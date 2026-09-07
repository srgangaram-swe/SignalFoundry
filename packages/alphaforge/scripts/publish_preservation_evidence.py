"""Publish verified ledger aggregates and a Seaborn figure, exclusively."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import tempfile
from pathlib import Path

from scripts.preservation.evidence import plot, summarize
from scripts.preservation.git import PreservationError
from scripts.preservation.inventory import canonical
from scripts.preservation.publish import read_ledger


def publish_evidence(
    ledgers: list[Path], output: Path, snapshots: list[Path] | None = None
) -> None:
    """Verify inputs, reserve a new output and publish complete artifacts atomically."""
    summary = summarize(ledgers)
    metadata = {}
    if snapshots:
        for path, directory in zip(snapshots, ledgers, strict=True):
            with path.open("rb") as stream:
                content = stream.read(2_000_001)
            if len(content) > 2_000_000:
                raise PreservationError("metadata-byte-limit")
            record = json.loads(content)
            ledger = read_ledger(directory)
            if record["advertised_refs"] != ledger["refs"]:
                raise PreservationError("metadata-ref-mismatch")
            metadata[ledger["source"]] = record
    if output.exists() or output.is_symlink():
        raise PreservationError("destination-unavailable")
    reservation = output.parent / f".{output.name}.reservation"
    reservation.mkdir(exist_ok=False)
    stage: Path | None = None
    try:
        stage = Path(tempfile.mkdtemp(prefix=".preservation-evidence-", dir=output.parent))
        (stage / "summary.json").write_bytes(canonical(summary))
        for source, record in metadata.items():
            (stage / f"{source}-github.json").write_bytes(canonical(record))
        plot(summary, stage / "preservation.png")
        if output.exists() or output.is_symlink():
            raise PreservationError("destination-race")
        os.rename(stage, output)
        stage = None
    finally:
        if stage is not None:
            shutil.rmtree(stage)
        reservation.rmdir()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ledgers", nargs=2, type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--metadata", nargs=2, type=Path)
    args = parser.parse_args()
    try:
        publish_evidence(args.ledgers, args.output, args.metadata)
    except (OSError, ValueError, KeyError, TypeError):
        parser.exit(2, "preservation evidence: publication failed; no source mutation performed\n")


if __name__ == "__main__":
    main()
