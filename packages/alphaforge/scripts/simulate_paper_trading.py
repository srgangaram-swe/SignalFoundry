from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from _common import latest_run_dir

from alphaforge.config import load_backtest_config
from alphaforge.paper import simulate_paper_trading
from alphaforge.research import read_frame_artifact, refresh_experiment_manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Replay latest targets as paper-trading orders.")
    parser.add_argument("--config", default="configs/backtest.yaml")
    parser.add_argument("--run-dir")
    parser.add_argument("--capital", type=float)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_backtest_config(args.config)
    run_dir = Path(args.run_dir) if args.run_dir else latest_run_dir()
    panel = read_frame_artifact(run_dir / "panel.table.json")
    weights_path = run_dir / "target_weights.csv"
    if not weights_path.exists():
        raise FileNotFoundError("target_weights.csv not found; run scripts/run_backtest.py first")
    weights = pd.read_csv(weights_path, parse_dates=["date"])
    orders, state = simulate_paper_trading(
        panel,
        weights,
        capital=args.capital or float(cfg.get("initial_capital", 1_000_000)),
        costs=cfg.get("costs", {}),
        execution=cfg.get("execution", {}),
        execution_lag=int(cfg.get("execution_lag", 1)),
        rebalance_frequency=int(cfg.get("rebalance_frequency", 1)),
    )
    orders.to_csv(run_dir / "paper_orders.csv", index=False)
    state.to_csv(run_dir / "paper_state.csv", index=False)
    refresh_experiment_manifest(run_dir)
    print("SIMULATED PAPER TRADING ONLY - no real orders were placed")
    print(f"orders: {len(orders):,}; output={run_dir}")


if __name__ == "__main__":
    main()
