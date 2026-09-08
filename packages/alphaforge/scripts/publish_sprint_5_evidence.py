"""Publish the Sprint 5 close-out evidence bundle and figure.

Run::

    python scripts/publish_sprint_5_evidence.py \
        --output docs/evidence/signal_foundry_sprint_5/closeout

Deterministic and network-independent. The bounded raw benchmark input and
frozen Git objects are verified before a transactional publication into an
absent repository-local destination.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from alphaforge.readiness.sprint_5_evidence import publish_sprint_5_evidence


def main() -> None:
    """Parse arguments and publish the bundle."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="repository-local destination directory; must not exist",
    )
    parser.add_argument(
        "--benchmark-input",
        type=Path,
        default=Path(
            "docs/evidence/signal_foundry_sprint_5/inputs/" "distribution_crossover_raw.json"
        ),
        help="committed bounded raw benchmark JSON",
    )
    parser.add_argument(
        "--repository-root",
        type=Path,
        default=Path.cwd(),
        help="AlphaForge Git worktree (defaults to the current directory)",
    )
    arguments = parser.parse_args()
    manifest = publish_sprint_5_evidence(
        repository_root=arguments.repository_root,
        benchmark_input=arguments.benchmark_input,
        output=arguments.output,
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
