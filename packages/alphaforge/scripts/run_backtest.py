from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
from _common import latest_run_dir

from alphaforge.backtesting import run_backtest
from alphaforge.config import load_backtest_config, load_risk_config
from alphaforge.evaluation import (
    CapacityColumns,
    CapacityConfig,
    deflated_sharpe_ratio,
    estimate_capacity,
)
from alphaforge.portfolio import construct_portfolio
from alphaforge.research import read_frame_artifact, refresh_experiment_manifest
from alphaforge.risk import (
    exposure_summary,
    monthly_returns,
    performance_summary,
    regime_performance,
    stress_test_summary,
)
from alphaforge.signals import apply_regime_filter, build_signals, select_model_predictions
from alphaforge.utils import save_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a realistic OOS backtest from predictions.")
    parser.add_argument("--config", default="configs/backtest.yaml")
    parser.add_argument("--risk-config", default="configs/risk.yaml")
    parser.add_argument("--run-dir")
    parser.add_argument("--latest", action="store_true")
    parser.add_argument("--model", help="Prediction model to backtest.")
    return parser.parse_args()


def _resolve_run_dir(args: argparse.Namespace) -> Path:
    if args.run_dir:
        return Path(args.run_dir)
    return latest_run_dir()


def main() -> None:
    args = parse_args()
    run_dir = _resolve_run_dir(args)
    cfg = load_backtest_config(args.config)
    risk_cfg = load_risk_config(args.risk_config)
    meta = json.loads((run_dir / "run_meta.json").read_text())

    panel = read_frame_artifact(run_dir / "panel.table.json")
    features = read_frame_artifact(run_dir / "features.table.json")
    predictions = read_frame_artifact(run_dir / "predictions.table.json")
    selected = select_model_predictions(predictions, model=args.model)
    signals = build_signals(
        selected, strategy=cfg.get("strategy", "long_short"), params=cfg.get("strategy_params", {})
    )
    if cfg.get("risk", {}).get("regime_filter"):
        signals = apply_regime_filter(signals, features)
    weights = construct_portfolio(signals, features=features, config=cfg.get("portfolio", {}))

    result = run_backtest(
        panel=panel,
        target_weights=weights,
        benchmark_symbol=meta.get("benchmark_symbol"),
        initial_capital=float(cfg.get("initial_capital", 1_000_000)),
        execution_lag=int(cfg.get("execution_lag", 1)),
        rebalance_frequency=int(cfg.get("rebalance_frequency", 1)),
        costs=cfg.get("costs", {}),
        risk=cfg.get("risk", {}),
        execution=cfg.get("execution", {}),
        latency=cfg.get("latency", {}),
        carry=cfg.get("borrow_financing", {}),
        liquidate_at_end=bool(cfg.get("liquidate_at_end", True)),
    )
    summary = performance_summary(result.equity_curve)
    summary.update({f"latest_{k}": v for k, v in exposure_summary(result.weights).items()})

    # Deflated Sharpe: deflate by the number of model variants that competed
    # for selection in this run. Understating n_trials overstates the result.
    n_trials = int(predictions["model"].nunique()) if "model" in predictions else 1
    active_returns = (
        result.equity_curve.loc[result.equity_curve["active"], "return"]
        if "active" in result.equity_curve
        else result.equity_curve["return"]
    )
    summary.update(deflated_sharpe_ratio(active_returns, n_trials=n_trials))

    signals.to_csv(run_dir / "signals.csv", index=False)
    weights.to_csv(run_dir / "target_weights.csv", index=False)
    result.equity_curve.to_csv(run_dir / "equity_curve.csv", index=False)
    result.weights.to_csv(run_dir / "executed_weights.csv", index=False)
    result.trades.to_csv(run_dir / "trades.csv", index=False)
    result.orders.to_csv(run_dir / "orders.csv", index=False)
    result.fills.to_csv(run_dir / "fills.csv", index=False)
    result.pnl_attribution.to_csv(run_dir / "pnl_attribution.csv", index=False)
    result.events.to_csv(run_dir / "execution_events.csv", index=False)
    result.accounting.to_csv(run_dir / "accounting.csv", index=False)
    result.friction_model_manifest.to_csv(run_dir / "friction_model_manifest.csv", index=False)
    result.friction_attribution.to_csv(run_dir / "friction_attribution.csv", index=False)
    result.latency_schedule.to_csv(run_dir / "latency_schedule.csv", index=False)
    monthly_returns(result.equity_curve).to_csv(run_dir / "monthly_returns.csv", index=False)

    if not result.fills.empty:
        requested = float(result.fills["requested_notional"].sum())
        traded = float(result.fills["traded_notional"].sum())
        component_totals = (
            result.friction_attribution.groupby("component", sort=True)["amount_usd"]
            .sum()
            .to_dict()
        )
        slippage_components = (
            "fixed_slippage",
            "spread_slippage",
            "participation_slippage",
            "volatility_slippage",
        )
        summary.update(
            {
                "execution_orders": int(len(result.fills)),
                "execution_fill_ratio": traded / requested if requested > 0 else 0.0,
                "execution_partial_fill_rate": float((result.fills["status"] == "partial").mean()),
                "execution_rejected_rate": float((result.fills["status"] == "rejected").mean()),
                "total_trading_cost_dollars": float(result.equity_curve["trading_cost"].sum()),
                "total_commission_dollars": float(component_totals.get("commission", 0.0)),
                "total_exchange_fee_dollars": float(component_totals.get("exchange_fee", 0.0)),
                "total_spread_cost_dollars": float(component_totals.get("spread", 0.0)),
                "total_slippage_cost_dollars": float(
                    sum(component_totals.get(name, 0.0) for name in slippage_components)
                ),
                "total_impact_cost_dollars": float(component_totals.get("market_impact", 0.0)),
                "total_financing_cost_dollars": float(component_totals.get("cash_financing", 0.0)),
                "total_borrow_cost_dollars": float(component_totals.get("short_borrow", 0.0)),
            }
        )

    capacity_cfg = cfg.get("capacity", {})
    if capacity_cfg.get("enabled", True) and not result.fills.empty:
        eligible = result.fills.loc[
            (result.fills["requested_notional"] > 0)
            & np.isfinite(result.fills["lagged_adv_notional"])
            & (result.fills["lagged_adv_notional"] > 0)
        ].copy()
        if not eligible.empty:
            reference_aum = float(cfg.get("initial_capital", 1_000_000))
            multiples = tuple(
                float(value) for value in capacity_cfg.get("aum_multiples", [0.5, 1, 2, 5, 10])
            )
            capacity_result = estimate_capacity(
                eligible,
                CapacityConfig(
                    reference_aum=reference_aum,
                    aum_values=tuple(reference_aum * multiple for multiple in multiples),
                    max_participation_rate=float(capacity_cfg.get("max_participation_rate", 0.10)),
                    impact_exponent=float(capacity_cfg.get("impact_exponent", 0.50)),
                    variable_cost_fraction=float(capacity_cfg.get("variable_cost_fraction", 0.50)),
                    columns=CapacityColumns.for_fill_records(),
                ),
            )
            capacity_result.curve.to_csv(run_dir / "capacity_curve.csv", index=False)
            capacity_result.scenario_trades.to_csv(run_dir / "capacity_scenarios.csv", index=False)
            diagnostics = asdict(capacity_result.diagnostics)
            diagnostics["excluded_rows_missing_lagged_adv"] = int(len(result.fills) - len(eligible))
            save_json(diagnostics, run_dir / "capacity_diagnostics.json")
            minimum_fill_ratio = float(capacity_cfg.get("minimum_fill_ratio", 0.95))
            feasible = capacity_result.curve.loc[
                capacity_result.curve["fill_ratio"] >= minimum_fill_ratio,
                "scenario_aum",
            ]
            summary["capacity_sensitivity_max_aum"] = (
                float(feasible.max()) if not feasible.empty else np.nan
            )
            summary["capacity_minimum_fill_ratio"] = minimum_fill_ratio

    # regime-conditional performance (causal HMM stress feature when present)
    regime_col = "hmm_stress_prob" if "hmm_stress_prob" in features.columns else "high_vol_regime"
    if regime_col in features.columns:
        regime = (features.groupby("date")[regime_col].first() > 0.5).map(
            {True: "stress", False: "calm"}
        )
        regime_performance(result.equity_curve, regime, regime_name="regime").to_csv(
            run_dir / "regime_performance.csv", index=False
        )

    betas = (
        features.sort_values("date").groupby("symbol")["beta"].last()
        if "beta" in features.columns
        else None
    )
    stress_test_summary(result.weights, risk_cfg.get("stress_scenarios", []), betas=betas).to_csv(
        run_dir / "stress_tests.csv", index=False
    )
    save_json(summary, run_dir / "backtest_summary.json")
    refresh_experiment_manifest(run_dir)

    print(f"backtest run: {run_dir}")
    print(f"selected model: {selected['model'].iloc[0] if 'model' in selected else 'single'}")
    print(f"summary: {summary}")


if __name__ == "__main__":
    main()
