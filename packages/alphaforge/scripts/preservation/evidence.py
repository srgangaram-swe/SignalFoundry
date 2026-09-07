"""Seaborn evidence for static preservation, never completed-migration claims."""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any

from scripts.preservation.git import PreservationError
from scripts.preservation.inventory import canonical, sha256
from scripts.preservation.publish import read_ledger


def summarize(directories: list[Path]) -> dict[str, Any]:
    """Derive aggregates from integrity-checked ledgers in deterministic order."""
    sources = []
    for directory in directories:
        ledger = read_ledger(directory)
        tree = ledger["ref_trees"]["refs/heads/dev"]
        files = ledger["trees"][tree]
        kinds = Counter(row["category"] for row in files)
        records = [
            record
            for value in {row["oid"] for row in files}
            for record in ledger["interfaces"].get(value, [])
        ]
        sources.append(
            {
                "source": ledger["source"],
                "ledger_sha256": sha256(canonical(ledger)),
                "dev": ledger["refs"]["refs/heads/dev"],
                "tracked_dev_files": len(files),
                "mapped_dev_files": sum(
                    row["target"] == f"packages/{ledger['source']}/{row['path']}" for row in files
                ),
                "capabilities": dict(sorted(kinds.items())),
                "public_declarations": dict(
                    sorted(Counter(row["kind"] for row in records).items())
                ),
                "dynamic_cli_api_declarations": sum(row.get("dynamic", False) for row in records),
                "summary": ledger["summary"],
                "migration_gate": ledger["migration_gate"],
            }
        )
    if len({row["source"] for row in sources}) != len(sources):
        raise PreservationError("duplicate-evidence-source")
    return {
        "schema_version": 1,
        "evidence_class": "offline-static-migration-plan",
        "sources": sorted(sources, key=lambda row: row["source"]),
        "runtime_parity": "NOT_RUN",
        "migration": "NOT_PERFORMED",
    }


def plot(summary: dict[str, Any], destination: Path) -> None:
    """Four safe aggregate panels; source counts, exclusions and pending gates."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import pandas as pd
    import seaborn as sns

    sns.set_theme(style="whitegrid", palette="colorblind", context="notebook")
    figure, axes = plt.subplots(2, 2, figsize=(15, 11))
    specifications = [
        ("capabilities", "Tracked dev files by capability", "Files"),
        ("object_kinds", "Complete reachable object closure", "Distinct Git objects"),
        (
            "public_declarations",
            "Static declarations at dev (not runtime discovery)",
            "Declarations",
        ),
        (
            "gates",
            "Unresolved migration checks — lower is not readiness",
            "Recorded findings / declarations",
        ),
    ]
    for axis, (field, title, units) in zip(axes.flat, specifications, strict=True):
        rows: list[dict[str, Any]] = []
        for source in summary["sources"]:
            if field == "object_kinds":
                values = source["summary"]["object_kinds"]
            elif field == "gates":
                values = {
                    "static blockers": source["summary"]["blockers"],
                    "dynamic CLI/API": source["dynamic_cli_api_declarations"],
                }
            else:
                values = source[field]
            rows.extend(
                {"Category": key, "Count": value, "Source": source["source"]}
                for key, value in values.items()
            )
        sns.barplot(
            data=pd.DataFrame(rows), x="Count", y="Category", hue="Source", ax=axis, errorbar=None
        )
        axis.set(title=title, xlabel=units, ylabel="")
        axis.set_xlim(left=0)
        axis.legend(title="Frozen source", loc="best")
    figure.suptitle("Signal Foundry: preservation evidence before import", fontsize=18)
    figure.text(
        0.5,
        0.025,
        "Complete enumeration of frozen public Git objects; no sampling or statistical uncertainty.\n"
        "File mappings are planned, not imported. Historical license gaps block import; dynamic interfaces need contract tests.\n"
        "Runtime parity NOT RUN · Migration NOT PERFORMED · No trading or model-performance conclusion.",
        ha="center",
        fontsize=10,
    )
    figure.tight_layout(rect=(0, 0.09, 1, 0.96))
    try:
        figure.savefig(
            destination, dpi=140, metadata={"Software": "Signal Foundry preservation evidence"}
        )
    finally:
        plt.close(figure)
