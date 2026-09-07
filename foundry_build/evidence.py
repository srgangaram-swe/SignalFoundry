"""Reproducible Seaborn assembly evidence from verified Git identities."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from foundry_build.assembly import canonical, read_json, verify
from foundry_build.git import AssemblyError, git


def measurements(root: Path, manifest: dict[str, Any]) -> dict[str, Any]:
    """Verify before measuring; count identities, never infer runtime correctness."""
    verified = verify(root, manifest)
    for source, result in verified.items():
        files = git(root, "ls-tree", "-rz", manifest["sources"][source]["tree"])
        result["tracked_files"] = len([entry for entry in files.split(b"\0") if entry])
    return {
        "schema_version": 1,
        "scope": "Git preservation counts; not performance or trading evidence",
        "manifest_sha256": hashlib.sha256(canonical(manifest)).hexdigest(),
        "sources": verified,
        "runtime_parity": "reported separately by source tests and CI",
        "source_container_limitation": "srgangaram-swe/Signalattice#67 remains open",
    }


def plot(report: dict[str, Any], path: Path) -> None:
    """Plot before/after counts with zero-based axes and textual values."""
    with sns.axes_style("whitegrid"), sns.plotting_context("notebook"):
        figure, axes = plt.subplots(1, 3, figsize=(13, 4.4), layout="constrained")
        try:
            for axis, metric, title in zip(
                axes,
                ("commits", "objects", "tracked_files"),
                (
                    "Original commits retained",
                    "Original objects verified",
                    "Current files preserved",
                ),
                strict=True,
            ):
                rows = [
                    {"source": source, "phase": phase, "count": values[metric]}
                    for source, values in report["sources"].items()
                    for phase in ("Frozen source", "Unified checkout")
                ]
                sns.barplot(
                    data={
                        key: [row[key] for row in rows]
                        for key in ("source", "phase", "count")
                    },
                    x="source",
                    y="count",
                    hue="phase",
                    ax=axis,
                    palette="colorblind",
                    errorbar=None,
                )
                axis.set(title=title, xlabel="", ylabel="Count", ylim=(0, None))
                axis.legend(title="", fontsize=8)
                for container in axis.containers:
                    axis.bar_label(container, padding=3, fontsize=9)
                axis.margins(y=0.18)
            figure.suptitle(
                "Signal Foundry assembly: exact identities, "
                "zero source-file exclusions\n"
                "Offline Git verification; runtime and live-trading readiness "
                "are separate gates",
                fontsize=12,
            )
            figure.savefig(
                path, dpi=150, metadata={"Software": "Signal Foundry assembly"}
            )
        finally:
            plt.close(figure)


def publish(report: dict[str, Any], destination: Path) -> None:
    """Publish a new directory atomically; never overwrite reviewed evidence."""
    destination = destination.parent.resolve() / destination.name
    if destination.exists() or destination.is_symlink():
        raise AssemblyError("evidence-already-exists")
    reservation = destination.with_name(f".{destination.name}.reservation")
    try:
        reservation.mkdir()
    except OSError as exc:
        raise AssemblyError("evidence-unavailable") from exc
    stage = None
    try:
        stage = Path(
            tempfile.mkdtemp(prefix=".assembly-evidence-", dir=destination.parent)
        )
        plot(report, stage / "preservation.png")
        (stage / "summary.json").write_bytes(canonical(report))
        hashes = {
            path.name: hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sorted(stage.iterdir())
        }
        (stage / "manifest.json").write_bytes(canonical(hashes))
        if destination.exists() or destination.is_symlink():
            raise AssemblyError("evidence-destination-race")
        os.rename(stage, destination)
        stage = None
    except OSError as exc:
        raise AssemblyError("evidence-publication-failed") from exc
    finally:
        if stage is not None:
            shutil.rmtree(stage)
        reservation.rmdir()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    try:
        report = measurements(root, read_json(root / "provenance/assembly.json"))
        publish(report, args.output)
    except (AssemblyError, OSError, KeyError, TypeError) as exc:
        parser.exit(2, f"assembly evidence refused ({type(exc).__name__})\n")
    print(json.dumps(report, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
