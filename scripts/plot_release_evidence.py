"""Render the supply-chain evidence figure for the release dry run.

SF-S5-SL-MR7. Every value comes from a real non-publishing dry run, collected by
``collect_release_evidence.py`` and ``check_release_reproducible.py``.

Four panels:

1. **Artifact set.** Subject count and total bytes by kind, so the release's
   shape is visible rather than a single "N artifacts" figure.
2. **Tamper detection.** Every deliberate mutation and whether verification
   refused it. A supply-chain figure showing only successes documents that
   nothing was tested, so the failures are the point of this panel.
3. **Reproducibility.** How many artifacts were byte-identical across two clean
   builds of one commit, which is the property every other guarantee rests on.
4. **Dependency inventory.** SBOM components by ecosystem.

The script refuses to render if the dry run did not verify, if any tamper went
undetected, or if the build was not reproducible -- publishing a figure whose
premise the evidence contradicts would be worse than publishing none.
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


def _validate(evidence: dict[str, Any], reproducibility: dict[str, Any]) -> None:
    """Refuse a figure whose own premise the evidence contradicts."""
    if not evidence.get("verified"):
        raise EvidenceError(f"the dry run did not verify: {evidence.get('verification_detail')}")
    undetected = [case["name"] for case in evidence.get("tamper_cases", []) if not case["detected"]]
    if undetected:
        raise EvidenceError(f"tamper cases went undetected: {', '.join(undetected)}")
    if not evidence.get("tamper_cases"):
        raise EvidenceError("no tamper cases were exercised")
    if not reproducibility.get("reproducible"):
        raise EvidenceError(f"the build is not reproducible: {reproducibility.get('differing')}")


def render(
    evidence: dict[str, Any], reproducibility: dict[str, Any], destination: Path
) -> dict[str, Any]:
    """Render the four-panel figure and return its summary."""
    sns.set_theme(style="whitegrid", context="talk", palette="colorblind")
    figure, axes = plt.subplots(1, 4, figsize=(26, 6.6))

    # --- 1. Artifact set ---------------------------------------------------
    kinds = pd.DataFrame(
        [
            {"kind": kind, "count": stats["count"], "kilobytes": stats["bytes"] / 1024}
            for kind, stats in evidence["subjects_by_kind"].items()
        ]
    ).sort_values("kilobytes", ascending=False)
    sns.barplot(data=kinds, y="kind", x="kilobytes", ax=axes[0], orient="h")
    axes[0].set_title("Release artifacts by kind", fontsize=14)
    axes[0].set_xlabel("total size (KiB)")
    axes[0].set_ylabel("")
    for index, row in kinds.reset_index(drop=True).iterrows():
        axes[0].text(
            float(row["kilobytes"]) * 1.02,
            int(index),
            f"{int(row['count'])} file{'s' if int(row['count']) != 1 else ''}",
            va="center",
            fontsize=10,
        )
    axes[0].set_xlim(0, kinds["kilobytes"].max() * 1.35)

    # --- 2. Tamper detection ----------------------------------------------
    cases = pd.DataFrame(
        [
            {
                "case": case["name"].replace("_", " "),
                "outcome": "refused" if case["detected"] else "MISSED",
                "value": 1,
            }
            for case in evidence["tamper_cases"]
        ]
    )
    sns.barplot(
        data=cases,
        y="case",
        x="value",
        hue="outcome",
        ax=axes[1],
        orient="h",
        dodge=False,
        palette={"refused": "#00785a", "MISSED": "#a33d00"},
    )
    axes[1].set_title("Deliberate tampering", fontsize=14)
    axes[1].set_xlabel("each case applied to a fresh copy and verified")
    axes[1].set_ylabel("")
    axes[1].set_xlim(0, 1.4)
    axes[1].set_xticks([])
    axes[1].legend(title="", loc="lower right", fontsize=10)

    # --- 3. Reproducibility -----------------------------------------------
    total = int(reproducibility["artifacts"])
    identical = int(reproducibility["identical"])
    repro = pd.DataFrame(
        [
            {"outcome": "byte-identical", "artifacts": identical},
            {"outcome": "differing", "artifacts": total - identical},
        ]
    )
    sns.barplot(
        data=repro,
        x="outcome",
        y="artifacts",
        ax=axes[2],
        palette={"byte-identical": "#00785a", "differing": "#a33d00"},
        hue="outcome",
        legend=False,
    )
    axes[2].set_title("Two clean builds of one commit", fontsize=14)
    axes[2].set_xlabel("")
    axes[2].set_ylabel("artifacts")
    axes[2].set_ylim(0, max(total * 1.2, 1))
    for index, row in repro.reset_index(drop=True).iterrows():
        axes[2].text(
            int(index),
            int(row["artifacts"]) + total * 0.03,
            str(int(row["artifacts"])),
            ha="center",
            fontsize=11,
        )

    # --- 4. Dependency inventory ------------------------------------------
    components = pd.DataFrame(
        [
            {"ecosystem": "python + node", "components": int(evidence["sbom_components"])},
        ]
    )
    sns.barplot(data=components, x="ecosystem", y="components", ax=axes[3])
    axes[3].set_title("SBOM components", fontsize=14)
    axes[3].set_xlabel("CycloneDX 1.6, from committed lockfiles")
    axes[3].set_ylabel("components")
    axes[3].text(
        0,
        int(evidence["sbom_components"]) * 1.03,
        str(evidence["sbom_components"]),
        ha="center",
        fontsize=12,
    )
    axes[3].set_ylim(0, int(evidence["sbom_components"]) * 1.25)

    figure.suptitle(
        f"SF-S5-SL-MR7 release dry run {evidence['release_version']}: artifacts, tampering, "
        "reproducibility, and dependencies",
        fontsize=15,
    )
    figure.text(
        0.5,
        0.015,
        f"Non-publishing dry run at commit {evidence['source_commit'][:12]}; build kind "
        f"'{evidence['build_kind']}'. No tag or GitHub Release was created. "
        "Digests establish content identity, not economic validity, independent review, or "
        "production readiness. A signature would prove only that the authorized process signed "
        "these bytes under the documented trust policy.",
        ha="center",
        fontsize=10,
        color="dimgray",
    )
    figure.tight_layout(rect=(0, 0.055, 1, 0.93))
    destination.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(destination, dpi=140, bbox_inches="tight")
    plt.close(figure)

    size = destination.stat().st_size
    if size > MAX_FIGURE_BYTES:
        raise EvidenceError(f"figure is {size} bytes, above the ceiling")
    return {
        "figure_bytes": size,
        "subjects": evidence["subject_count"],
        "tamper_cases": len(evidence["tamper_cases"]),
        "reproducible_artifacts": identical,
    }


def main(argv: list[str] | None = None) -> int:
    """Render the release supply-chain figure."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence", type=Path, required=True)
    parser.add_argument("--reproducibility", type=Path, required=True)
    parser.add_argument(
        "--output", type=Path, default=Path("reports/figures/release_supply_chain.png")
    )
    arguments = parser.parse_args(argv)

    try:
        evidence = json.loads(arguments.evidence.read_text(encoding="utf-8"))
        reproducibility = json.loads(arguments.reproducibility.read_text(encoding="utf-8"))
        _validate(evidence, reproducibility)
        summary = render(evidence, reproducibility, arguments.output)
    except (EvidenceError, OSError, ValueError) as error:
        print(f"release figure refused: {error}")
        return 2
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    raise SystemExit(main())
