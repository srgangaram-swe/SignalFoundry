"""Run the frozen SF-S3-MR7 synthetic engineering benchmark."""

from __future__ import annotations

import argparse
from pathlib import Path

from alphaforge.research.time_frequency_study import (
    load_time_frequency_study_config,
    run_synthetic_time_frequency_study,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run the mandatory small-CNN and gated ResNet/ViT progression on "
            "the deterministic offline time-frequency engineering reference."
        )
    )
    parser.add_argument(
        "--config",
        default="configs/time_frequency_vision_benchmark.yaml",
        help="Frozen SF-S3-MR7 study configuration.",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="New output directory for aggregate evidence and its Seaborn plot.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = run_synthetic_time_frequency_study(
        load_time_frequency_study_config(args.config),
        Path(args.output),
    )
    print(f"evidence={result.output_dir}")
    print(f"evaluated={int(result.summary['status'].eq('evaluated').sum())}")
    print(f"gates={len(result.gates)}")
    print("scope=synthetic-engineering-only")


if __name__ == "__main__":
    main()
