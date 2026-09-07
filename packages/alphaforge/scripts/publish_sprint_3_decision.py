"""Publish the frozen, aggregate-only Signal Foundry Sprint 3 decision."""

from __future__ import annotations

import argparse
from pathlib import Path

from alphaforge.research.sprint_3_decision import (
    load_sprint_3_evaluation_plan,
    publish_sprint_3_decision,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Verify the frozen ten-family Sprint 3 plan and atomically publish "
            "aggregate decision evidence without reopening model holdouts."
        )
    )
    parser.add_argument(
        "--config",
        default="configs/sprint_3_decision.yaml",
        help="Content-addressed strict synthesis plan.",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="New repository-local output directory; overwrite is prohibited.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    repository = Path.cwd()
    plan = load_sprint_3_evaluation_plan(
        args.config,
        repository_root=repository,
    )
    output = publish_sprint_3_decision(
        repository_root=repository,
        plan=plan,
        output_dir=args.output,
    )
    print(f"study_id={plan.study_id}")
    print(f"plan_id={plan.plan_id}")
    print(f"families={len(plan.families)}")
    print(f"output={output}")
    print("decision=NOT_READY")
    print("orders_emitted=0")
    print("capital_deployed=0")


if __name__ == "__main__":
    main()
