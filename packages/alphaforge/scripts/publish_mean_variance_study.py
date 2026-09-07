"""Publish the frozen SF-S4-MR2 synthetic development evidence."""

from __future__ import annotations

import argparse
from pathlib import Path

from alphaforge.optimization.study import publish_mean_variance_study


def parse_args() -> argparse.Namespace:
    """Parse the bounded offline publisher arguments."""
    parser = argparse.ArgumentParser(
        description=(
            "Run every SF-S4-MR2 evidence arm and sensitivity on deterministic "
            "synthetic development data; no holdout, order, or credential is accepted."
        )
    )
    parser.add_argument(
        "--config",
        default="configs/mean_variance_study.yaml",
        help="Strict frozen mean-variance study YAML.",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="New directory for aggregate CSV, JSON, manifest, and Seaborn evidence.",
    )
    return parser.parse_args()


def main() -> None:
    """Run the publication and report identities without performance claims."""
    args = parse_args()
    result = publish_mean_variance_study(Path(args.config), Path(args.output))
    print(f"evidence={result.output_dir}")
    print(f"config_sha256={result.config_sha256}")
    print(f"manifest_sha256={result.manifest_sha256}")
    print("scope=synthetic_development_only")
    print("holdout_accessible=false")
    print("candidate_selected=false")
    print("profit_claim=false")


if __name__ == "__main__":
    main()
