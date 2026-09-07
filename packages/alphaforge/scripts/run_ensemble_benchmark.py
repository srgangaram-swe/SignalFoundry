"""Run the frozen SF-S3-MR9 governed-ensemble engineering study."""

from __future__ import annotations

import argparse
from pathlib import Path

from alphaforge.research.ensemble_study import (
    load_ensemble_study_config,
    run_synthetic_ensemble_study,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run all governed ensemble policies on a deterministic synthetic "
            "temporal-OOF reference and publish aggregate-only evidence."
        )
    )
    parser.add_argument(
        "--config",
        default="configs/ensemble_benchmark.yaml",
        help="Frozen SF-S3-MR9 study configuration.",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="New output directory for aggregate tables and the Seaborn plot.",
    )
    return parser.parse_args()


def main() -> None:
    arguments = parse_args()
    result = run_synthetic_ensemble_study(
        load_ensemble_study_config(arguments.config),
        Path(arguments.output),
    )
    print(f"evidence={result.output_dir}")
    print(f"models={len(result.summary)}")
    print("scope=synthetic-engineering-only")
    print("published_row_predictions=0")


if __name__ == "__main__":
    main()
