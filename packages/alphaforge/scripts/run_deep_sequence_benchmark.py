"""Run the frozen SF-S3-MR6 development-only benchmark."""

from __future__ import annotations

import argparse
from pathlib import Path

from alphaforge.data import load_signal_foundry_dataset
from alphaforge.research.deep_sequence_study import (
    load_deep_sequence_study_config,
    run_deep_sequence_study,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare controlled sequence architectures with LightGBM on matched "
            "development-only folds."
        )
    )
    parser.add_argument("bundle", help="Verified local Signal Foundry bundle directory.")
    parser.add_argument(
        "--config",
        default="configs/deep_sequence_benchmark.yaml",
        help="Frozen deep-sequence study configuration.",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="New output directory for aggregate evidence and its Seaborn plot.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = run_deep_sequence_study(
        load_signal_foundry_dataset(Path(args.bundle)),
        load_deep_sequence_study_config(args.config),
        Path(args.output),
    )
    print(f"evidence={result.output_dir}")
    print(f"candidates={len(result.summary)}")
    print("scope=development-only")


if __name__ == "__main__":
    main()
