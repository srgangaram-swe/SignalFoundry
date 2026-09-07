"""Run the frozen seven-candidate Signal Foundry Sprint 2 study."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from alphaforge.config import (
    load_research_governance_config,
    load_signal_foundry_research_config,
)
from alphaforge.data import load_signal_foundry_dataset
from alphaforge.evaluation import ReadinessThresholds
from alphaforge.research import GovernedResearchConfig
from alphaforge.research.baseline_study import run_governed_baseline_study


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run one immutable, governed seven-candidate Sprint 2 baseline study."
    )
    parser.add_argument("bundle", help="Verified local Signal Foundry bundle directory.")
    parser.add_argument(
        "--config",
        default="configs/signal_foundry_sprint_2_study.yaml",
        help="Frozen study configuration.",
    )
    parser.add_argument(
        "--governance-config",
        default="configs/research_governance.yaml",
        help="Frozen correction, kill-criterion, and ledger policy.",
    )
    parser.add_argument("--study-output", default="runs/signal-foundry-sprint-2")
    parser.add_argument("--research-output", default="runs/signal-foundry")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_signal_foundry_research_config(args.config)
    governance = load_research_governance_config(args.governance_config)
    research_values = dict(config["research"])
    research_values["horizons"] = tuple(research_values["horizons"])
    result = run_governed_baseline_study(
        dataset=load_signal_foundry_dataset(Path(args.bundle)),
        model_specs=list(config["models"]),
        feature_config=dict(config["features"]),
        walk_forward_config=dict(config["walk_forward"]),
        backtest_config=dict(config["backtest"]),
        research_config=GovernedResearchConfig(**research_values),
        readiness_thresholds=ReadinessThresholds.from_mapping(dict(config["readiness"])),
        governance_config=governance,
        output_root=Path(args.study_output),
        research_output_root=Path(args.research_output),
        invocation={
            "entrypoint": "scripts/run_sprint_2_study.py",
            "arguments": sys.argv[1:],
        },
    )
    print(f"study_id={result.study_id}")
    print(f"study_dir={result.study_dir}")
    print(f"run_id={result.research_run.run_id}")
    print(f"run_dir={result.research_run.run_dir}")
    print(f"candidate={result.research_run.candidate_model}")
    print(f"decision={result.decision}")


if __name__ == "__main__":
    main()
