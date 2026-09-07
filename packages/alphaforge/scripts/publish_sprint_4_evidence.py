"""Publish the deterministic Sprint 4 close-out evidence bundle."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from alphaforge.research.sprint_4_evidence import publish_sprint_4_evidence


def parse_args() -> argparse.Namespace:
    """Parse the bounded offline publisher arguments."""
    parser = argparse.ArgumentParser(
        description=(
            "Run the frozen SF-S4-MR8 perturbation study and SF-S4-MR9 qualification "
            "decision on deterministic synthetic development data, then write the "
            "evidence bundle and Seaborn figure. Offline; no credential is accepted."
        )
    )
    parser.add_argument(
        "--output",
        required=True,
        help="New or empty directory for the JSON, CSV, dossier, and figure.",
    )
    return parser.parse_args()


def main() -> None:
    """Publish the bundle and print its machine-readable summary."""
    args = parse_args()
    summary = publish_sprint_4_evidence(Path(args.output))
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
