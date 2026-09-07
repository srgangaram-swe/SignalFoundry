"""Run the frozen SF-S3-MR8 synthetic representation benchmark."""

from __future__ import annotations

import argparse
from pathlib import Path

from alphaforge.research.representation_study import (
    load_representation_study_config,
    run_synthetic_representation_study,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare the frozen raw, PCA, autoencoder, VAE, and contrastive "
            "representation family on deterministic synthetic engineering data."
        )
    )
    parser.add_argument(
        "--config",
        default="configs/latent_representation_benchmark.yaml",
        help="Frozen SF-S3-MR8 study configuration.",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="New directory for aggregate-only evidence and its Seaborn plot.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = run_synthetic_representation_study(
        load_representation_study_config(args.config),
        Path(args.output),
    )
    selected = result.metadata["study"]["selected_candidate"]
    print(f"evidence={result.output_dir}")
    print(f"candidates={len(result.summary)}")
    print(f"selected_on_validation={selected}")
    print("scope=synthetic-engineering-only")


if __name__ == "__main__":
    main()
