"""Render the console's sprint evidence figure (SF-S5-SL-MR6).

Every value plotted comes from ``collect_console_evidence.py``, which reads real
build and test artifacts. Nothing here is hand-entered, and the script refuses to
render if the evidence contradicts what the figure would claim.

Four panels:

1. **Resource budgets.** Observed against limit for the initial JavaScript, the
   initial stylesheet, and the whole emitted bundle, drawn as utilisation so the
   headroom is the visible quantity.
2. **Browser matrix.** Passed and skipped counts per engine, with skips shown
   rather than folded into the pass count -- a skipped test is not a passing one.
3. **Palette contrast.** Every light-palette token against the WCAG 2.2 AA
   threshold, including the two that failed before this work and forced the
   ramp to be darkened.
4. **Honest states per view.** Which of the nine states each of the seven views
   can report. This is a static property of the source, not a coverage claim,
   and the caption says so.
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


class EvidenceError(RuntimeError):
    """Raised when the evidence cannot support the figure's claims."""


def _validate(document: dict[str, Any]) -> None:
    """Refuse to publish a figure whose own premise the evidence contradicts.

    The figure asserts that budgets hold, that no browser failed, and that the
    palette clears AA. If any of those is false the honest outcome is a failure,
    not a plotted claim.
    """
    budgets = document.get("bundle_budgets") or []
    if not budgets:
        raise EvidenceError("no bundle budgets were measured")
    over = [item["name"] for item in budgets if item["observed"] > item["limit"]]
    if over:
        raise EvidenceError(f"budgets exceeded: {', '.join(over)}")

    browsers = document.get("browsers") or {}
    if not browsers:
        raise EvidenceError("no browser results were recorded")
    failed = {name: counts["failed"] for name, counts in browsers.items() if counts["failed"]}
    if failed:
        raise EvidenceError(f"browser failures present: {failed}")

    threshold = float(document["aa_contrast_threshold"])
    contrast = document.get("palette_contrast") or {}
    if not contrast:
        raise EvidenceError("no palette contrast was computed")
    below = {name: value for name, value in contrast.items() if value < threshold}
    if below:
        raise EvidenceError(f"palette tokens below WCAG AA: {below}")

    states = document.get("route_states") or {}
    if len(states) != 7:
        raise EvidenceError(f"expected exactly seven views, found {len(states)}")


def render(document: dict[str, Any], destination: Path) -> dict[str, Any]:
    """Render the four-panel figure and return its aggregate summary."""
    sns.set_theme(style="whitegrid", context="talk", palette="colorblind")
    figure, axes = plt.subplots(1, 4, figsize=(26, 6.4))

    # --- 1. Budgets -------------------------------------------------------
    budgets = pd.DataFrame(document["bundle_budgets"])
    budgets["label"] = budgets["name"].str.replace("_", " ")
    sns.barplot(data=budgets, y="label", x="utilisation", ax=axes[0], orient="h")
    axes[0].axvline(1.0, color="crimson", linestyle="--", linewidth=1.5)
    axes[0].set_xlim(0, 1.75)
    axes[0].set_title("Resource budgets", fontsize=14)
    axes[0].set_xlabel("fraction of limit used (dashed line: the limit)")
    axes[0].set_ylabel("")
    for index, row in budgets.reset_index(drop=True).iterrows():
        axes[0].text(
            float(row["utilisation"]) + 0.02,
            int(index),
            f"{int(row['observed']):,} / {int(row['limit']):,} B",
            va="center",
            fontsize=9.5,
        )

    # --- 2. Browser matrix ------------------------------------------------
    rows = [
        {"engine": engine, "outcome": outcome, "tests": counts[outcome]}
        for engine, counts in sorted(document["browsers"].items())
        for outcome in ("passed", "skipped", "failed")
    ]
    matrix = pd.DataFrame(rows)
    sns.barplot(data=matrix, x="engine", y="tests", hue="outcome", ax=axes[1])
    axes[1].set_title("Browser matrix", fontsize=14)
    axes[1].set_xlabel("")
    axes[1].set_ylabel("browser tests")
    axes[1].tick_params(axis="x", rotation=20)
    axes[1].legend(title="", fontsize=10, loc="upper right")

    # --- 3. Contrast ------------------------------------------------------
    threshold = float(document["aa_contrast_threshold"])
    contrast = pd.DataFrame(
        [
            {"token": name.replace("state-", "").replace("-", " "), "ratio": value}
            for name, value in sorted(document["palette_contrast"].items())
        ]
    )
    sns.barplot(data=contrast, y="token", x="ratio", ax=axes[2], orient="h")
    axes[2].axvline(threshold, color="crimson", linestyle="--", linewidth=1.5)
    axes[2].set_title("Palette contrast on the light surface", fontsize=14)
    axes[2].set_xlabel(f"contrast ratio (dashed line: WCAG 2.2 AA = {threshold})")
    axes[2].set_ylabel("")

    # --- 4. Honest states per view ---------------------------------------
    states = document["evidence_states"]
    # Abbreviated so nine columns fit without the labels running into one
    # another; the full names are in the evidence document and the ADR.
    short = {
        "LOADING": "load",
        "READY": "ready",
        "EMPTY": "empty",
        "PARTIAL": "partial",
        "INSUFFICIENT_EVIDENCE": "insuff.",
        "INVALID": "invalid",
        "STALE": "stale",
        "UNAVAILABLE": "unavail.",
        "ERROR": "error",
    }
    grid = pd.DataFrame(
        [
            {
                "view": view.replace("Uncertainty", "").replace("Operations", "Ops"),
                "state": short[state],
                "reported": 1 if state in reported else 0,
            }
            for view, reported in sorted(document["route_states"].items())
            for state in states
        ]
    )
    pivot = grid.pivot(index="view", columns="state", values="reported")
    pivot = pivot[[short[state] for state in states]]
    sns.heatmap(
        pivot,
        ax=axes[3],
        cmap=sns.color_palette(["#e9edf1", "#005f96"], as_cmap=True),
        cbar=False,
        linewidths=1,
        linecolor="white",
        annot=pivot.replace({1: "yes", 0: "—"}),
        fmt="",
        annot_kws={"fontsize": 9},
    )
    axes[3].set_title("States each view can report", fontsize=14)
    axes[3].set_xlabel("")
    axes[3].set_ylabel("")
    axes[3].set_xticklabels(axes[3].get_xticklabels(), rotation=40, ha="right", fontsize=10)
    axes[3].set_yticklabels(axes[3].get_yticklabels(), rotation=0, fontsize=10)

    coverage = document["coverage"]
    figure.suptitle(
        "SF-S5-SL-MR6 console: budgets, browser matrix, contrast, and honest states",
        fontsize=16,
    )
    figure.text(
        0.5,
        0.015,
        "Measured local engineering evidence, not a benchmark or a trading claim. "
        f"Front-end branch coverage {coverage.get('branches', 0):.1f}% against a 90% gate; "
        f"{sum(c['passed'] for c in document['browsers'].values())} browser tests passed and "
        f"{sum(c['skipped'] for c in document['browsers'].values())} skipped with recorded reasons. "
        "Panel 4 is a static property of the view sources -- which states each view is able to "
        "report -- not a claim about which were exercised.",
        ha="center",
        fontsize=10,
        color="dimgray",
    )
    figure.tight_layout(rect=(0, 0.06, 1, 0.93))
    destination.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(destination, dpi=140, bbox_inches="tight")
    plt.close(figure)

    size = destination.stat().st_size
    if size > MAX_FIGURE_BYTES:
        raise EvidenceError(f"figure is {size} bytes, above the {MAX_FIGURE_BYTES} ceiling")
    return {
        "figure_bytes": size,
        "budgets": len(budgets),
        "engines": len(document["browsers"]),
        "views": len(document["route_states"]),
    }


def main(argv: list[str] | None = None) -> int:
    """Render the console evidence figure from a collected document."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("reports/figures/console_evidence.png"))
    arguments = parser.parse_args(argv)

    try:
        document = json.loads(arguments.evidence.read_text(encoding="utf-8"))
        _validate(document)
        summary = render(document, arguments.output)
    except (EvidenceError, OSError, ValueError) as error:
        print(f"console evidence figure refused: {error}")
        return 2
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    raise SystemExit(main())
