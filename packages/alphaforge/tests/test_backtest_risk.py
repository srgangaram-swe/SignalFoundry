from __future__ import annotations

import pandas as pd
import pytest

from alphaforge.backtesting import run_backtest
from alphaforge.portfolio import construct_portfolio
from alphaforge.risk import performance_summary, stress_test_summary


def test_transaction_costs_reduce_equity(small_panel):
    dates = sorted(small_panel["date"].unique())[20:120]
    symbols = ["SYN000", "SYN001", "SYN002", "SYN003"]
    rows = []
    for i, date in enumerate(dates):
        for symbol in symbols:
            direction = 1.0 if (i + symbols.index(symbol)) % 2 == 0 else -1.0
            rows.append({"date": date, "symbol": symbol, "signal": direction})
    signals = pd.DataFrame(rows)
    weights = construct_portfolio(
        signals,
        config={"max_weight": 0.25, "max_gross_exposure": 1.0, "inverse_vol_scaling": False},
    )

    free = run_backtest(
        small_panel,
        weights,
        benchmark_symbol="BENCH",
        costs={"commission_bps": 0, "half_spread_bps": 0, "slippage_bps": 0},
    )
    costly = run_backtest(
        small_panel,
        weights,
        benchmark_symbol="BENCH",
        costs={"commission_bps": 20, "half_spread_bps": 20, "slippage_bps": 20},
    )
    assert costly.equity_curve["equity"].iloc[-1] < free.equity_curve["equity"].iloc[-1]
    summary = performance_summary(costly.equity_curve)
    assert "max_drawdown" in summary
    assert summary["transaction_cost_impact"] > 0


def test_portfolio_contract_enforces_cash_gross_and_net_limits() -> None:
    date = pd.Timestamp("2026-01-02")
    balanced = pd.DataFrame(
        {
            "date": [date] * 4,
            "symbol": ["A", "B", "C", "D"],
            "signal": [1.0, 1.0, -1.0, -1.0],
        }
    )
    weights = construct_portfolio(
        balanced,
        config={
            "scheme": "equal_weight",
            "max_weight": 0.5,
            "max_gross_exposure": 1.0,
            "max_net_exposure": 0.2,
            "turnover_cap": None,
            "vol_lookback": 60,
            "cash_buffer": 0.2,
        },
    )
    assert weights["target_weight"].abs().sum() == pytest.approx(0.8)
    assert abs(weights["target_weight"].sum()) <= 0.2

    long_only = balanced.assign(signal=1.0)
    capped = construct_portfolio(
        long_only,
        config={
            "max_weight": 0.5,
            "max_gross_exposure": 1.0,
            "max_net_exposure": 0.2,
            "inverse_vol_scaling": False,
            "turnover_cap": None,
        },
    )
    assert capped["target_weight"].sum() == pytest.approx(0.2)
    with pytest.raises(ValueError, match="unknown portfolio"):
        construct_portfolio(balanced, config={"silent_leverage": 10.0})


def test_stress_report_uses_declared_volatility_multiplier() -> None:
    weights = pd.DataFrame(
        {
            "date": pd.to_datetime(["2026-01-02"]),
            "symbol": ["A"],
            "weight": [0.5],
        }
    )
    report = stress_test_summary(
        weights,
        [{"name": "crash", "market_shock": -0.4, "vol_multiplier": 3.0}],
    )
    assert report.loc[0, "estimated_portfolio_return"] == pytest.approx(-0.2)
    assert report.loc[0, "volatility_multiplier"] == pytest.approx(3.0)
