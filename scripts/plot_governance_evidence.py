"""Render champion-challenger promotion-governance evidence (SF-S5-SL-MR5).

The figure answers the question a reviewer should ask of any promotion rule:
*under a true null, how often does it recommend promoting anyway?*

Three panels, all from seeded synthetic data generated in this script:

1. **Null calibration under intraday dependence.** Cohorts are simulated with no
   true difference between arms and a shared per-day shock. The date-block
   bootstrap that :mod:`quant_platform.governance.inference` actually uses is
   compared against a naive per-forecast bootstrap. The naive curve climbs above
   the nominal alpha as dependence strengthens; the block curve does not.
2. **Interval width.** The same cohorts, showing that the block interval is
   wider because the effective sample size is genuinely smaller -- the naive
   interval is not more precise, it is wrong.
3. **Holm correction across a family.** Raw versus adjusted p-values for a
   four-test family, marking which cross alpha before and after correction.
4. **Gate outcomes and cohort accounting.** Every absolute gate for two example
   cohorts -- one clearing the floor, one thin -- together with the coverage and
   exclusion counts behind them. Unfavourable and insufficient outcomes are
   shown, not omitted: a gate panel that only ever displays passes is decoration.

The output contains aggregate synthetic engineering measurements only: no market
observations, identifiers, credentials, or host paths.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, Final

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from quant_platform.governance.comparison import PairedCohort, PairedScore, PairKey
from quant_platform.governance.gates import FrozenPolicy, evaluate_gates
from quant_platform.governance.inference import (
    BOOTSTRAP_REPLICATES,
    Margin,
    holm_adjust,
)

#: Alpha the study is calibrated against, matching the default frozen policy.
ALPHA: Final = 0.05

#: Shared-shock strengths swept in panel 1. Zero is independence; higher values
#: mean a larger share of each day's score difference is common to that day.
DEPENDENCE_LEVELS: Final = (0.0, 0.25, 0.5, 0.75, 1.0)

#: Simulated cohorts per dependence level. Enough that a false-positive rate of
#: 0.05 has a standard error near 0.011, so a naive rate above ~0.09 is not noise.
COHORTS_PER_LEVEL: Final = 400

DAYS: Final = 30
PER_DAY: Final = 8

#: Replicates for the study. Lower than the production default because the study
#: runs thousands of bootstraps; the tail quantiles at this alpha are stable here.
STUDY_REPLICATES: Final = 2_000

MAX_FIGURE_BYTES: Final = 32 * 1024 * 1024


class EvidenceError(RuntimeError):
    """Raised when the study cannot produce trustworthy evidence."""


@dataclass(frozen=True, slots=True)
class CalibrationPoint:
    """One dependence level's measured operating characteristics."""

    dependence: float
    block_false_positive_rate: float
    naive_false_positive_rate: float
    block_median_width: float
    naive_median_width: float


def _simulate_null_cohort(
    generator: np.random.Generator, *, dependence: float
) -> tuple[np.ndarray, np.ndarray]:
    """Return per-forecast differences and their day index under a true null.

    Each day draws a shared shock; each forecast adds independent noise. The
    mean difference is zero in expectation, so any rejection is a false positive.
    Total variance is held constant across dependence levels, so the panels
    compare calibration rather than signal strength.
    """
    shared = generator.normal(0.0, 1.0, size=DAYS) * np.sqrt(dependence)
    idiosyncratic = generator.normal(0.0, 1.0, size=(DAYS, PER_DAY)) * np.sqrt(1.0 - dependence)
    differences = (shared[:, None] + idiosyncratic) * 0.01
    day_index = np.repeat(np.arange(DAYS), PER_DAY)
    return differences.reshape(-1), day_index


def _block_interval(
    values: np.ndarray, day_index: np.ndarray, generator: np.random.Generator
) -> tuple[float, float]:
    """Two-sided interval from resampling whole days, as the module does."""
    days = np.unique(day_index)
    per_day = np.array([values[day_index == day].mean() for day in days], dtype=float)
    counts = np.array([int(np.sum(day_index == day)) for day in days], dtype=float)
    draws = generator.integers(0, len(days), size=(STUDY_REPLICATES, len(days)))
    sampled = per_day[draws]
    weights = counts[draws]
    means = np.sum(sampled * weights, axis=1) / np.sum(weights, axis=1)
    low, high = np.percentile(means, [100 * ALPHA / 2, 100 * (1 - ALPHA / 2)])
    return float(low), float(high)


def _naive_interval(values: np.ndarray, generator: np.random.Generator) -> tuple[float, float]:
    """Two-sided interval from resampling individual forecasts.

    Present only as the contrast: it assumes forecasts made on the same day are
    independent, which they are not.
    """
    draws = generator.integers(0, len(values), size=(STUDY_REPLICATES, len(values)))
    means = values[draws].mean(axis=1)
    low, high = np.percentile(means, [100 * ALPHA / 2, 100 * (1 - ALPHA / 2)])
    return float(low), float(high)


def run_calibration_study(seed: int = 20260811) -> list[CalibrationPoint]:
    """Measure false-positive rate and interval width against dependence."""
    generator = np.random.default_rng(seed)
    points: list[CalibrationPoint] = []
    for dependence in DEPENDENCE_LEVELS:
        block_rejections = 0
        naive_rejections = 0
        block_widths: list[float] = []
        naive_widths: list[float] = []
        for _ in range(COHORTS_PER_LEVEL):
            values, day_index = _simulate_null_cohort(generator, dependence=dependence)
            block_low, block_high = _block_interval(values, day_index, generator)
            naive_low, naive_high = _naive_interval(values, generator)
            # A rejection under a true null is a false positive either way it points.
            block_rejections += int(block_high < 0.0 or block_low > 0.0)
            naive_rejections += int(naive_high < 0.0 or naive_low > 0.0)
            block_widths.append(block_high - block_low)
            naive_widths.append(naive_high - naive_low)
        points.append(
            CalibrationPoint(
                dependence=dependence,
                block_false_positive_rate=block_rejections / COHORTS_PER_LEVEL,
                naive_false_positive_rate=naive_rejections / COHORTS_PER_LEVEL,
                block_median_width=float(np.median(block_widths)),
                naive_median_width=float(np.median(naive_widths)),
            )
        )
    return points


def _holm_family() -> pd.DataFrame:
    """A four-test family where correction changes two of the conclusions."""
    raw = {
        "brier_superiority": 0.004,
        "log_superiority": 0.021,
        "brier_non_inferiority": 0.033,
        "log_non_inferiority": 0.048,
    }
    correction = holm_adjust(raw, alpha=ALPHA)
    adjusted: dict[str, float] = correction["adjusted"]
    # Readable axis labels come from the data, not from relabelling ticks after
    # the fact: set_yticklabels on a categorical axis warns that it may mislabel.
    return pd.DataFrame(
        [
            {"test": name.replace("_", " "), "stage": "raw", "p_value": value}
            for name, value in sorted(raw.items())
        ]
        + [
            {"test": name.replace("_", " "), "stage": "Holm-adjusted", "p_value": adjusted[name]}
            for name in sorted(raw)
        ]
    )


EVIDENCE_BASE: Final = datetime(2026, 8, 1, tzinfo=UTC)


def _example_cohort(*, days: int, per_day: int, unmatched: int) -> PairedCohort:
    """Build a cohort with a chosen size and a chosen number of exclusions."""
    pairs = []
    for day in range(days):
        as_of = EVIDENCE_BASE + timedelta(days=day)
        for slot in range(per_day):
            pairs.append(
                PairedScore(
                    key=PairKey(campaign_symbol=f"S{slot:02d}", as_of=as_of, horizon_days=5),
                    as_of_date=as_of.date(),
                    champion_brier=0.50,
                    challenger_brier=0.47,
                    champion_log=0.70,
                    challenger_log=0.66,
                )
            )
    return PairedCohort(
        pairs=tuple(pairs),
        champion_total=len(pairs) + unmatched,
        challenger_total=len(pairs),
        champion_only=unmatched,
        challenger_only=0,
        champion_unscored=0,
        challenger_unscored=0,
        comparable=True,
        incomparable_reason=None,
    )


def gate_evidence() -> tuple[pd.DataFrame, dict[str, Any]]:
    """Return per-gate outcomes for a sufficient and an insufficient cohort.

    Both are shown deliberately. A gate panel that only ever displays passing
    gates communicates nothing about what the gates would refuse.
    """
    policy = FrozenPolicy(
        version="promotion-1", alpha=ALPHA, margin=Margin(metric="brier", value=0.01)
    )
    cases = {
        "sufficient": {
            "cohort": _example_cohort(days=30, per_day=8, unmatched=1),
            "class_counts": {"up": 160, "down": 80},
            "achieved_power": 0.86,
        },
        "thin": {
            "cohort": _example_cohort(days=9, per_day=4, unmatched=12),
            "class_counts": {"up": 30, "down": 6},
            "achieved_power": 0.41,
        },
    }
    rows: list[dict[str, Any]] = []
    accounting: dict[str, Any] = {}
    for label, case in cases.items():
        cohort: PairedCohort = case["cohort"]
        results = evaluate_gates(
            cohort,
            policy,
            class_counts=case["class_counts"],
            achieved_power=case["achieved_power"],
        )
        for result in results:
            rows.append(
                {
                    "cohort": label,
                    "gate": result.name.replace("_", " "),
                    "satisfied": bool(result.satisfied),
                    "status": "satisfied" if result.satisfied else "failed",
                }
            )
        accounting[label] = {
            "matched": cohort.matched,
            "coverage": cohort.coverage,
            "champion_only": cohort.champion_only,
            "challenger_only": cohort.challenger_only,
            "distinct_days": len({item.as_of_date for item in cohort.pairs}),
            "gates_failed": sum(1 for item in results if not item.satisfied),
            "gates_total": len(results),
        }
    return pd.DataFrame(rows), accounting


def _validate(points: list[CalibrationPoint]) -> None:
    """Refuse to publish a figure whose own premise did not hold.

    The study exists to show the block bootstrap stays calibrated. If it did
    not, the honest outcome is a failure, not a plotted claim.
    """
    if not points:
        raise EvidenceError("the calibration study produced no points")
    independent = points[0]
    dependent = points[-1]
    if independent.dependence != 0.0 or dependent.dependence != 1.0:
        raise EvidenceError("the dependence sweep must span independence to full sharing")
    if dependent.block_false_positive_rate > 3 * ALPHA:
        raise EvidenceError(
            "the date-block bootstrap did not hold its nominal level under dependence "
            f"({dependent.block_false_positive_rate:.3f} against alpha {ALPHA}); "
            "publishing this figure would assert a property the study disproved"
        )
    if dependent.naive_false_positive_rate <= dependent.block_false_positive_rate:
        raise EvidenceError(
            "the naive bootstrap was not anticonservative under dependence, so the "
            "figure's central contrast is not present in the data"
        )


def render(points: list[CalibrationPoint], destination: Path, *, seed: int) -> dict[str, Any]:
    """Render the three-panel figure and return its aggregate summary."""
    sns.set_theme(style="whitegrid", context="talk", palette="colorblind")
    figure, axes = plt.subplots(1, 4, figsize=(25, 5.8))

    frame = pd.DataFrame(
        [
            {
                "dependence": point.dependence,
                "estimator": estimator,
                "false_positive_rate": rate,
                "median_width": width,
            }
            for point in points
            for estimator, rate, width in (
                ("date-block (used)", point.block_false_positive_rate, point.block_median_width),
                ("per-forecast (naive)", point.naive_false_positive_rate, point.naive_median_width),
            )
        ]
    )

    sns.lineplot(
        data=frame,
        x="dependence",
        y="false_positive_rate",
        hue="estimator",
        marker="o",
        ax=axes[0],
    )
    axes[0].axhline(ALPHA, color="crimson", linestyle="--", linewidth=1.4)
    axes[0].text(0.28, ALPHA + 0.034, f"nominal alpha = {ALPHA}", color="crimson", fontsize=11)
    axes[0].set_title("False positives under a true null", fontsize=14)
    axes[0].set_xlabel("share of variance common to the day")
    axes[0].set_ylabel("rejection rate")
    axes[0].set_ylim(0, max(0.32, frame["false_positive_rate"].max() * 1.15))
    axes[0].legend(title="", loc="upper left", fontsize=11)

    sns.lineplot(
        data=frame,
        x="dependence",
        y="median_width",
        hue="estimator",
        marker="o",
        ax=axes[1],
        legend=False,
    )
    axes[1].set_title("Median interval width", fontsize=14)
    axes[1].set_xlabel("share of variance common to the day")
    axes[1].set_ylabel("width of the 95% interval")

    family = _holm_family()
    sns.barplot(data=family, x="p_value", y="test", hue="stage", orient="h", ax=axes[2])
    axes[2].axvline(ALPHA, color="crimson", linestyle="--", linewidth=1.4)
    axes[2].set_title("Holm correction across the family", fontsize=14)
    axes[2].set_xlabel(f"p-value (dashed line: alpha = {ALPHA})")
    axes[2].set_ylabel("")
    # Outside the axes: at these p-values every in-axes corner sits on a bar.
    axes[2].legend(title="", loc="upper left", bbox_to_anchor=(1.02, 1.0), fontsize=11)

    gates, accounting = gate_evidence()
    palette = {"satisfied": "#2a7f62", "failed": "#b2182b"}
    sns.scatterplot(
        data=gates,
        x="cohort",
        y="gate",
        hue="status",
        style="status",
        palette=palette,
        markers={"satisfied": "o", "failed": "X"},
        s=260,
        ax=axes[3],
        legend="full",
    )
    axes[3].set_title("Absolute gates on two cohorts", fontsize=14)
    axes[3].set_xlabel("excluded rows are champion-only, counted not dropped", fontsize=11)
    axes[3].set_ylabel("")
    axes[3].margins(x=0.55, y=0.08)
    axes[3].legend(title="", loc="upper left", bbox_to_anchor=(1.02, 1.0), fontsize=11)
    # The cohort accounting belongs on the axis itself: a separate annotation
    # block collides with the tick labels at any figure size worth reading.
    ordered = sorted(accounting.items())
    axes[3].set_xticks(range(len(ordered)))
    axes[3].set_xticklabels(
        [
            f"{label}\n{record['matched']} pairs / {record['distinct_days']} days\n"
            f"coverage {record['coverage']:.3f}\n"
            f"{record['champion_only']} excluded\n"
            f"{record['gates_failed']} of {record['gates_total']} gates failed"
            for label, record in ordered
        ],
        fontsize=10,
    )

    figure.suptitle(
        "SF-S5-SL-MR5 promotion governance: calibrated inference and absolute gates",
        fontsize=15,
    )
    # The honest caveat, stated on the figure rather than only in the ADR: a
    # percentile bootstrap over ~30 blocks is itself mildly anticonservative,
    # which is why the block curve sits slightly above alpha rather than on it.
    figure.text(
        0.5,
        0.015,
        f"Simulated evidence, not measured or traded: {COHORTS_PER_LEVEL} synthetic null "
        f"cohorts per level, {DAYS} days x {PER_DAY} forecasts, seed {seed}, "
        f"{STUDY_REPLICATES} bootstrap replicates "
        f"(binomial standard error ~{np.sqrt(ALPHA * (1 - ALPHA) / COHORTS_PER_LEVEL):.3f}). "
        f"A percentile bootstrap over {DAYS} blocks is itself mildly anticonservative; the "
        "block curve tracks alpha within noise, it does not sit exactly on it.",
        ha="center",
        fontsize=10,
        color="dimgray",
    )
    figure.tight_layout(rect=(0, 0.07, 1, 0.94))
    destination.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(destination, dpi=140, bbox_inches="tight")
    plt.close(figure)

    size = destination.stat().st_size
    if size > MAX_FIGURE_BYTES:
        raise EvidenceError(f"figure is {size} bytes, above the {MAX_FIGURE_BYTES} ceiling")

    return {
        "alpha": ALPHA,
        "seed": seed,
        "cohorts_per_level": COHORTS_PER_LEVEL,
        "days": DAYS,
        "forecasts_per_day": PER_DAY,
        "study_replicates": STUDY_REPLICATES,
        "production_replicates": BOOTSTRAP_REPLICATES,
        "calibration": [
            {
                "dependence": point.dependence,
                "block_false_positive_rate": point.block_false_positive_rate,
                "naive_false_positive_rate": point.naive_false_positive_rate,
                "block_median_width": point.block_median_width,
                "naive_median_width": point.naive_median_width,
            }
            for point in points
        ],
        "gate_accounting": accounting,
        "figure_bytes": size,
    }


def main(argv: list[str] | None = None) -> int:
    """Run the study, render the figure, and emit its aggregate summary."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("reports/figures/governance_promotion_evidence.png"),
        help="Destination PNG path.",
    )
    parser.add_argument(
        "--summary", type=Path, default=None, help="Optional JSON summary destination."
    )
    parser.add_argument("--seed", type=int, default=20260811, help="Study seed.")
    arguments = parser.parse_args(argv)

    try:
        points = run_calibration_study(seed=arguments.seed)
        _validate(points)
        summary = render(points, arguments.output, seed=arguments.seed)
    except EvidenceError as error:
        print(f"governance evidence refused: {error}")
        return 2

    if arguments.summary is not None:
        arguments.summary.parent.mkdir(parents=True, exist_ok=True)
        arguments.summary.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    raise SystemExit(main())
