"""Publish aggregate-only Sprint 2 evidence from local governed artifacts."""

from __future__ import annotations

import argparse

from alphaforge.research.baseline_study_evidence import publish_baseline_study_evidence


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Publish redistribution-safe aggregate Sprint 2 study evidence."
    )
    parser.add_argument("study", help="Local governed study directory.")
    parser.add_argument("run", help="Local governed research-run directory.")
    parser.add_argument("bundle", help="Verified local Signal Foundry bundle directory.")
    parser.add_argument("performance_profile", help="Bounded macOS /usr/bin/time -l output.")
    parser.add_argument(
        "--config",
        default="configs/signal_foundry_sprint_2_study.yaml",
        help="Frozen study configuration.",
    )
    parser.add_argument(
        "--output",
        default="docs/evidence/signal_foundry_sprint_2",
        help="New aggregate-only publication directory.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output = publish_baseline_study_evidence(
        study_dir=args.study,
        run_dir=args.run,
        bundle_dir=args.bundle,
        config_path=args.config,
        output_dir=args.output,
        performance_profile=args.performance_profile,
    )
    print(f"published={output}")


if __name__ == "__main__":
    main()
