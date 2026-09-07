"""Publish redistribution-safe aggregate evidence from a governed local run."""

from __future__ import annotations

import argparse

from alphaforge.research.public_evidence import publish_signal_foundry_evidence


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Publish aggregate-only Signal Foundry evidence and Seaborn plots."
    )
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--bundle-dir", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--performance-profile",
        help="Optional bounded macOS /usr/bin/time -l output for aggregate profiling evidence.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    destination = publish_signal_foundry_evidence(
        run_dir=args.run_dir,
        bundle_dir=args.bundle_dir,
        config_path=args.config,
        output_dir=args.output_dir,
        performance_profile=args.performance_profile,
    )
    print(f"published aggregate Signal Foundry evidence: {destination}")


if __name__ == "__main__":
    main()
