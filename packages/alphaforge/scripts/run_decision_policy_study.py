"""Run the frozen offline SF-S3-MR10 aggregate study."""

from __future__ import annotations

import argparse
from pathlib import Path

from alphaforge.research.decision_policy_study import (
    load_decision_policy_study_config,
    run_decision_policy_study,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate cost/uncertainty abstention against always-trade and "
            "never-trade synthetic baselines; emits no orders."
        )
    )
    parser.add_argument(
        "--config",
        default="configs/decision_policy.yaml",
        help="Strict frozen decision-policy study configuration.",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="New directory for aggregate-only CSV, JSON, and Seaborn evidence.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = run_decision_policy_study(
        load_decision_policy_study_config(args.config),
        Path(args.output),
    )
    print(f"evidence={result.output_dir}")
    print(f"study_id={result.study_id}")
    print(f"policy_id={result.policy_id}")
    print("orders_emitted=0")
    print("scope=synthetic_engineering_only")


if __name__ == "__main__":
    main()
