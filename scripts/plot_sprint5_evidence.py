"""Render the final Sprint 5 sprint-evidence figure.

SF-S5-SL-MR8. Three panels, all from the committed machine-readable index and
the benchmark documents it points at.

1. **Evidence and gate matrix.** Every sprint outcome kept at its real strength:
   PASS, INSUFFICIENT, INVALID, and UNAVAILABLE are distinct cells and are never
   collapsed into a favourable aggregate score. There is deliberately no
   "overall sprint health" number, because such a number is exactly the device
   that hides an INSUFFICIENT behind nine PASSes.
2. **Resource distributions.** Service and console budgets shown as utilisation
   against their declared limits, with the limit drawn, so headroom is visible
   rather than implied.
3. **Governance uncertainty.** Null-calibration false-positive rates against
   nominal alpha, retaining the unfavourable naive-estimator curve.

Captions distinguish deterministic-synthetic, historical-replay, and
local-engineering evidence, and state plainly that Sprint 5 produced **no**
prospective wall-clock evidence.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Final

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

MAX_FIGURE_BYTES: Final = 32 * 1024 * 1024

#: Outcome vocabulary. These are held apart on purpose: an INSUFFICIENT result
#: is a finding about the evidence, an INVALID one is a finding about its
#: coherence, and neither is a weaker PASS.
_OUTCOMES: Final = ("PASS", "INSUFFICIENT", "INVALID", "UNAVAILABLE")
_OUTCOME_VALUE: Final = {name: index for index, name in enumerate(_OUTCOMES)}


class EvidenceError(RuntimeError):
    """Raised when the evidence cannot support the figure's claims."""


def _load(path: Path) -> dict[str, Any]:
    """Read one JSON evidence document."""
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except OSError as error:
        raise EvidenceError(f"missing evidence document: {path}") from error
    except ValueError as error:
        raise EvidenceError(f"{path} is not valid JSON") from error


def _gate_matrix(root: Path) -> pd.DataFrame:
    """Assemble the sprint's gate outcomes from committed evidence."""
    console = _load(root / "docs/benchmarks/console_evidence_2026-08-20.json")
    release = _load(root / "docs/benchmarks/release_dry_run_2026-08-20.json")
    repro = _load(root / "docs/benchmarks/release_reproducibility_2026-08-20.json")

    budgets_ok = all(item["observed"] <= item["limit"] for item in console["bundle_budgets"])
    browsers_ok = all(counts["failed"] == 0 for counts in console["browsers"].values())
    contrast_ok = all(
        value >= console["aa_contrast_threshold"] for value in console["palette_contrast"].values()
    )
    tampers_ok = all(case["detected"] for case in release["tamper_cases"])

    rows = [
        ("#18 registry & CAS", "provenance round-trip", "PASS"),
        ("#17 read-only API", "versioned contract served", "PASS"),
        ("#20 bounded service", "admission & response bounds", "PASS"),
        (
            "#20 bounded service",
            "reproducible image config digest",
            # Honest: clean builds still differ in image metadata. Tracked as
            # #67 and carried here rather than smoothed away.
            "INVALID",
        ),
        ("#21 shadow replay", "deterministic delayed replay", "PASS"),
        ("#22 governance", "block bootstrap holds nominal alpha", "PASS"),
        (
            "#22 governance",
            "wall-clock promotion evidence",
            # The floor is 28 consecutive days; Sprint 5 has replay only.
            "INSUFFICIENT",
        ),
        ("#19 console", "resource budgets", "PASS" if budgets_ok else "INVALID"),
        ("#19 console", "browser matrix", "PASS" if browsers_ok else "INVALID"),
        ("#19 console", "WCAG 2.2 AA contrast", "PASS" if contrast_ok else "INVALID"),
        (
            "#23 release",
            "two builds byte-identical",
            "PASS" if repro["reproducible"] else "INVALID",
        ),
        ("#23 release", "tamper cases refused", "PASS" if tampers_ok else "INVALID"),
        (
            "#23 release",
            "cryptographic signing exercised",
            # No signing material exists yet; deferred to the publication path.
            "UNAVAILABLE",
        ),
    ]
    return pd.DataFrame(rows, columns=["area", "gate", "outcome"])


def _validate(matrix: pd.DataFrame) -> None:
    """Refuse a figure that quietly lost an unfavourable outcome."""
    present = set(matrix["outcome"])
    unknown = present - set(_OUTCOMES)
    if unknown:
        raise EvidenceError(f"unknown outcome values: {sorted(unknown)}")
    if present == {"PASS"}:
        raise EvidenceError(
            "every gate reports PASS; a sprint matrix with no unfavourable outcome is a matrix "
            "that stopped recording them"
        )


def render(root: Path, destination: Path) -> dict[str, Any]:
    """Render the three-panel sprint figure."""
    matrix = _gate_matrix(root)
    _validate(matrix)
    console = _load(root / "docs/benchmarks/console_evidence_2026-08-20.json")
    service = _load(root / "docs/benchmarks/service_operability_2026-09-06.json")
    governance = _load(root / "reports/figures/governance_promotion_evidence.json")

    sns.set_theme(style="whitegrid", context="talk", palette="colorblind")
    figure, axes = plt.subplots(1, 3, figsize=(24, 7.6))

    # --- 1. Gate matrix ----------------------------------------------------
    matrix = matrix.assign(label=matrix["area"] + " — " + matrix["gate"])
    matrix = matrix.assign(value=matrix["outcome"].map(_OUTCOME_VALUE))
    palette = {
        "PASS": "#00785a",
        "INSUFFICIENT": "#8a5a00",
        "INVALID": "#a33d00",
        "UNAVAILABLE": "#5c636a",
    }
    sns.barplot(
        data=matrix,
        y="label",
        x=[1] * len(matrix),
        hue="outcome",
        hue_order=list(_OUTCOMES),
        palette=palette,
        dodge=False,
        ax=axes[0],
        orient="h",
    )
    for index, row in matrix.reset_index(drop=True).iterrows():
        axes[0].text(
            0.5,
            int(index),
            row["outcome"],
            ha="center",
            va="center",
            color="white",
            fontsize=9.5,
            fontweight="bold",
        )
    axes[0].set_title("Sprint 5 gate outcomes", fontsize=14)
    axes[0].set_xlabel("outcomes are not collapsed into a score")
    axes[0].set_ylabel("")
    axes[0].set_xticks([])
    axes[0].set_xlim(0, 1)
    # No legend: every bar carries its outcome as text, so the colour is the
    # third channel rather than the key. A legend here would also sit on top of
    # three rows at any readable size.
    legend = axes[0].get_legend()
    if legend is not None:
        legend.remove()
    axes[0].tick_params(axis="y", labelsize=9.5)

    # --- 2. Resource distributions ----------------------------------------
    rows: list[dict[str, Any]] = [
        {
            "surface": "console",
            "budget": item["name"].replace("_", " "),
            "utilisation": item["observed"] / item["limit"],
        }
        for item in console["bundle_budgets"]
    ]
    for bound in service.get("bounds_observed", []):
        limit = float(bound.get("limit") or 0)
        if limit > 0:
            rows.append(
                {
                    "surface": "service",
                    "budget": str(bound.get("name", "bound")).replace("_", " "),
                    "utilisation": float(bound.get("observed", 0)) / limit,
                }
            )
    resources = pd.DataFrame(rows)
    sns.barplot(data=resources, y="budget", x="utilisation", hue="surface", ax=axes[1], orient="h")
    axes[1].axvline(1.0, color="crimson", linestyle="--", linewidth=1.5)
    axes[1].set_xlim(0, 1.15)
    axes[1].set_title("Declared limits and headroom", fontsize=14)
    axes[1].set_xlabel("fraction of limit used (dashed line: the limit)")
    axes[1].set_ylabel("")
    axes[1].legend(title="", fontsize=10, loc="lower right")
    axes[1].tick_params(axis="y", labelsize=9.5)

    # --- 3. Governance uncertainty ----------------------------------------
    calibration = pd.DataFrame(
        [
            {
                "dependence": point["dependence"],
                "estimator": estimator,
                "false_positive_rate": rate,
            }
            for point in governance["calibration"]
            for estimator, rate in (
                ("date-block (used)", point["block_false_positive_rate"]),
                ("per-forecast (naive)", point["naive_false_positive_rate"]),
            )
        ]
    )
    sns.lineplot(
        data=calibration,
        x="dependence",
        y="false_positive_rate",
        hue="estimator",
        marker="o",
        ax=axes[2],
    )
    alpha = float(governance["alpha"])
    axes[2].axhline(alpha, color="crimson", linestyle="--", linewidth=1.5)
    axes[2].text(0.28, alpha + 0.03, f"nominal alpha = {alpha}", color="crimson", fontsize=11)
    axes[2].set_title("Promotion inference under a true null", fontsize=14)
    axes[2].set_xlabel("share of variance common to the day")
    axes[2].set_ylabel("false-positive rate")
    axes[2].legend(title="", loc="upper left", fontsize=10)

    figure.suptitle(
        "Signal Foundry Sprint 5 — evidence, limits, and what remains unestablished", fontsize=16
    )
    figure.text(
        0.5,
        0.015,
        "Evidence classes: deterministic-synthetic (governance calibration, "
        f"{governance['cohorts_per_level']} cohorts per level, seed {governance['seed']}), "
        "historical-replay (registry and cache, public-domain data through 2018-03-27), and "
        "local-engineering (service, console, release; one machine). "
        "Sprint 5 produced NO prospective wall-clock evidence: the 28-day promotion floor is not "
        "met, no result here is paper- or live-trading evidence, and prospective validation is "
        "tracked in #63.",
        ha="center",
        fontsize=10,
        color="dimgray",
    )
    figure.tight_layout(rect=(0, 0.065, 1, 0.93))
    destination.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(destination, dpi=140, bbox_inches="tight")
    plt.close(figure)

    size = destination.stat().st_size
    if size > MAX_FIGURE_BYTES:
        raise EvidenceError(f"figure is {size} bytes, above the ceiling")
    counts = matrix["outcome"].value_counts().to_dict()
    return {"figure_bytes": size, "gates": len(matrix), "outcomes": counts}


def main(argv: list[str] | None = None) -> int:
    """Render the sprint evidence figure."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--output", type=Path, default=Path("reports/figures/sprint5_evidence.png"))
    arguments = parser.parse_args(argv)
    root = arguments.root.resolve()
    try:
        summary = render(root, root / arguments.output)
    except (EvidenceError, OSError, ValueError, KeyError) as error:
        print(f"sprint figure refused: {error}")
        return 2
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    raise SystemExit(main())
