"""Paper-trading replay built on the historical engine's execution contract.

This module never routes real orders.  It replays close-time targets through
the same close-decision -> future-open-fill policy, causal lagged-liquidity
inputs, transaction costs, and self-financing ledger used by ``run_backtest``.
The optional C++ order book remains a separate, uncalibrated systems demo; it
is not presented as a reconstruction of historical daily-bar liquidity.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from alphaforge.backtesting import run_backtest
from alphaforge.execution import CostModel, ExecutionPolicy


def simulate_paper_trading(
    panel: pd.DataFrame,
    target_weights: pd.DataFrame,
    capital: float = 1_000_000.0,
    lookback_days: int = 20,
    half_spread_bps: float = 2.5,
    *,
    costs: CostModel | dict | None = None,
    execution: ExecutionPolicy | dict | None = None,
    execution_lag: int = 1,
    rebalance_frequency: int = 1,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Replay targets as simulated DAY orders and return recent fills/state."""
    if lookback_days < 1:
        raise ValueError("lookback_days must be >= 1")
    cost_config: CostModel | dict = (
        costs
        if costs is not None
        else {
            "commission_bps": 0.0,
            "half_spread_bps": float(half_spread_bps),
            "slippage_bps": 0.0,
        }
    )
    result = run_backtest(
        panel=panel,
        target_weights=target_weights,
        initial_capital=capital,
        execution_lag=execution_lag,
        rebalance_frequency=rebalance_frequency,
        costs=cost_config,
        execution=execution,
        risk={},
    )

    fills = result.fills.copy()
    if fills.empty:
        orders = pd.DataFrame(
            columns=[
                "date",
                "symbol",
                "side",
                "decision_date",
                "status",
                "simulated_notional",
                "reference_price",
                "simulated_fill_price",
                "simulated_shares",
                "residual_shares",
                "slippage_bps",
                "total_cost",
                "lagged_adv_shares",
                "participation_rate",
            ]
        )
    else:
        fills["fill_date"] = pd.to_datetime(fills["fill_date"])
        # Anchor the audit window at the most recent order rather than the end
        # of a market panel that may extend beyond the last OOS prediction.
        latest_fill_date = pd.Timestamp(fills["fill_date"].max())
        recent_dates = pd.DatetimeIndex(result.equity_curve["date"])
        recent_dates = recent_dates[recent_dates <= latest_fill_date].sort_values()[-lookback_days:]
        fills = fills.loc[fills["fill_date"].isin(recent_dates)].copy()
        direction = np.sign(fills["requested_shares"])
        execution_shortfall_bps = np.where(
            direction >= 0,
            (fills["fill_price"] / fills["reference_price"] - 1.0) * 10_000.0,
            (1.0 - fills["fill_price"] / fills["reference_price"]) * 10_000.0,
        )
        orders = pd.DataFrame(
            {
                "date": fills["fill_date"],
                "symbol": fills["symbol"],
                "side": np.where(direction > 0, "BUY", "SELL"),
                "decision_date": fills["decision_date"],
                "status": fills["status"]
                .str.upper()
                .map(
                    {
                        "FILLED": "SIMULATED_FILL",
                        "PARTIAL": "SIMULATED_PARTIAL",
                        "REJECTED": "SIMULATED_REJECTED",
                    }
                ),
                "simulated_notional": fills["traded_notional"],
                "reference_price": fills["reference_price"],
                "simulated_fill_price": fills["fill_price"],
                "simulated_shares": fills["filled_shares"].abs(),
                "residual_shares": fills["residual_shares"].abs(),
                "slippage_bps": execution_shortfall_bps,
                "total_cost": fills["total_cost"],
                "lagged_adv_shares": fills["lagged_adv_shares"],
                "participation_rate": fills["participation_rate"],
            }
        ).sort_values(["date", "symbol"])
        orders = orders.reset_index(drop=True)

    latest_date = pd.Timestamp(result.weights["date"].max())
    state = result.weights.loc[result.weights["date"] == latest_date].copy()
    state = state[
        ["date", "symbol", "target_weight", "weight", "shares", "mark_price", "market_value"]
    ]
    state = state.rename(columns={"market_value": "simulated_notional"})
    return orders, state.reset_index(drop=True)
