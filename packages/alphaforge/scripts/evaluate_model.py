from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from _common import latest_run_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Print model evaluation summary.")
    parser.add_argument("--run-dir")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_dir = Path(args.run_dir) if args.run_dir else latest_run_dir()
    metrics = pd.read_csv(run_dir / "model_metrics.csv")
    summary = (
        metrics.groupby("model").mean(numeric_only=True).sort_values("rank_ic", ascending=False)
    )
    print(summary.to_string())


if __name__ == "__main__":
    main()
