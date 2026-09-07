from __future__ import annotations

import argparse
from pathlib import Path

from _common import latest_run_dir

from alphaforge.config import load_backtest_config, load_portfolio_config
from alphaforge.portfolio import construct_portfolio
from alphaforge.research import read_frame_artifact, refresh_experiment_manifest
from alphaforge.signals import build_signals, select_model_predictions


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Construct target portfolio weights.")
    parser.add_argument("--config", default="configs/portfolio.yaml")
    parser.add_argument("--backtest-config", default="configs/backtest.yaml")
    parser.add_argument("--run-dir")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_dir = Path(args.run_dir) if args.run_dir else latest_run_dir()
    portfolio_cfg = load_portfolio_config(args.config)
    backtest_cfg = load_backtest_config(args.backtest_config)
    predictions = read_frame_artifact(run_dir / "predictions.table.json")
    features = read_frame_artifact(run_dir / "features.table.json")
    selected = select_model_predictions(predictions)
    signals = build_signals(
        selected,
        strategy=backtest_cfg.get("strategy", "long_short"),
        params=backtest_cfg.get("strategy_params", {}),
    )
    weights = construct_portfolio(signals, features=features, config=portfolio_cfg)
    weights.to_csv(run_dir / "target_weights.csv", index=False)
    refresh_experiment_manifest(run_dir)
    print(f"target weights written: {run_dir / 'target_weights.csv'}")


if __name__ == "__main__":
    main()
