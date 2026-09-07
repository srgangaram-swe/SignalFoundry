from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from alphaforge.backtesting import run_backtest


def _panel(
    prices: dict[str, list[tuple[float, float]]],
    *,
    start: str = "2024-01-02",
    volume: float = 1_000_000.0,
) -> pd.DataFrame:
    dates = pd.bdate_range(start, periods=len(next(iter(prices.values()))))
    rows = []
    for symbol, bars in prices.items():
        for date, (open_price, close_price) in zip(dates, bars, strict=True):
            rows.append(
                {
                    "date": date,
                    "symbol": symbol,
                    "open": open_price,
                    "high": max(open_price, close_price),
                    "low": min(open_price, close_price),
                    "close": close_price,
                    "volume": volume,
                }
            )
    return pd.DataFrame(rows)


def test_next_open_fill_cannot_capture_the_pre_fill_overnight_gap() -> None:
    panel = _panel({"AAA": [(100.0, 100.0), (200.0, 200.0), (220.0, 220.0)]})
    dates = sorted(panel["date"].unique())
    targets = pd.DataFrame({"date": [dates[0]], "symbol": ["AAA"], "target_weight": [1.0]})

    result = run_backtest(
        panel, targets, costs={"commission_bps": 0, "half_spread_bps": 0, "slippage_bps": 0}
    )
    curve = result.equity_curve.set_index("date")

    # The target was decided after the 100 close and filled at the next 200
    # open. It cannot own that +100% gap; it does own the following gap.
    assert curve.loc[dates[1], "return"] == pytest.approx(0.0)
    assert curve.loc[dates[1], "equity"] == pytest.approx(1_000_000.0)
    assert curve.loc[dates[2], "return"] == pytest.approx(0.10)
    assert result.fills.loc[0, "decision_date"] == dates[0]
    assert result.fills.loc[0, "fill_date"] == dates[1]
    assert result.fills.loc[0, "reference_price"] == 200.0


def test_positions_drift_and_rebalancing_the_same_target_trades() -> None:
    panel = _panel(
        {
            "AAA": [(100.0, 100.0), (100.0, 200.0), (200.0, 200.0)],
            "BBB": [(100.0, 100.0), (100.0, 100.0), (100.0, 100.0)],
        }
    )
    dates = sorted(panel["date"].unique())
    targets = pd.DataFrame(
        [{"date": dates[0], "symbol": symbol, "target_weight": 0.5} for symbol in ("AAA", "BBB")]
        + [{"date": dates[1], "symbol": symbol, "target_weight": 0.5} for symbol in ("AAA", "BBB")]
    )

    result = run_backtest(
        panel,
        targets,
        costs={"commission_bps": 10, "half_spread_bps": 0, "slippage_bps": 0},
    )
    positions = result.weights.pivot(index="date", columns="symbol", values="weight")

    assert positions.loc[dates[1], "AAA"] > 0.65
    assert positions.loc[dates[1], "BBB"] < 0.34
    second_rebalance = result.trades[result.trades["date"] == dates[2]]
    assert set(second_rebalance["symbol"]) == {"AAA", "BBB"}
    assert second_rebalance["trade_weight"].abs().sum() > 0.30
    assert result.equity_curve.set_index("date").loc[dates[2], "trading_cost"] > 0


def test_cash_positions_and_symbol_pnl_reconcile_every_day() -> None:
    panel = _panel(
        {
            "AAA": [(100.0, 100.0), (101.0, 103.0), (104.0, 102.0), (102.0, 105.0)],
            "BBB": [(50.0, 50.0), (49.0, 48.0), (47.0, 49.0), (50.0, 48.0)],
        }
    )
    dates = sorted(panel["date"].unique())
    targets = pd.DataFrame(
        [
            {"date": dates[0], "symbol": "AAA", "target_weight": 0.6},
            {"date": dates[0], "symbol": "BBB", "target_weight": -0.4},
            {"date": dates[2], "symbol": "AAA", "target_weight": 0.2},
            {"date": dates[2], "symbol": "BBB", "target_weight": -0.2},
        ]
    )
    result = run_backtest(
        panel,
        targets,
        costs={"commission_bps": 1, "half_spread_bps": 2, "slippage_bps": 3},
    )

    marked = result.weights.groupby("date")["market_value"].sum()
    curve = result.equity_curve.set_index("date")
    np.testing.assert_allclose(
        curve["equity"].to_numpy(),
        (curve["cash"] + marked.reindex(curve.index, fill_value=0.0)).to_numpy(),
        rtol=1e-12,
        atol=1e-6,
    )

    attributed = (
        result.pnl_attribution.groupby("date")[["market_pnl", "trading_cost", "net_pnl"]]
        .sum()
        .reindex(curve.index, fill_value=0.0)
    )
    previous_equity = curve["equity"].shift(1).fillna(1_000_000.0)
    np.testing.assert_allclose(attributed["market_pnl"], curve["market_pnl"], atol=1e-6)
    np.testing.assert_allclose(attributed["trading_cost"], curve["trading_cost"], atol=1e-6)
    np.testing.assert_allclose(
        attributed["net_pnl"],
        curve["return"] * previous_equity,
        atol=1e-6,
    )


def test_execution_day_volume_cannot_change_a_lagged_adv_fill() -> None:
    n_days = 12
    panel = _panel({"AAA": [(100.0, 100.0)] * n_days}, volume=1_000.0)
    dates = sorted(panel["date"].unique())
    targets = pd.DataFrame({"date": [dates[8]], "symbol": ["AAA"], "target_weight": [1.0]})
    execution = {
        "adv_lookback": 5,
        "volatility_lookback": 2,
        "max_participation_rate": 0.10,
    }
    baseline = run_backtest(panel, targets, costs={}, execution=execution)

    mutated = panel.copy()
    fill_date = dates[9]
    mutated.loc[mutated["date"] == fill_date, "volume"] = 1_000_000_000.0
    changed = run_backtest(mutated, targets, costs={}, execution=execution)

    assert baseline.fills.loc[0, "status"] == "partial"
    assert baseline.fills.loc[0, "filled_shares"] == pytest.approx(100.0)
    assert changed.fills.loc[0, "filled_shares"] == pytest.approx(100.0)
    assert baseline.fills.loc[0, "lagged_adv_shares"] == changed.fills.loc[0, "lagged_adv_shares"]


def test_sparse_target_snapshot_explicitly_liquidates_omitted_names() -> None:
    panel = _panel(
        {
            "AAA": [(100.0, 100.0)] * 4,
            "BBB": [(100.0, 100.0)] * 4,
        }
    )
    dates = sorted(panel["date"].unique())
    targets = pd.DataFrame(
        [
            {"date": dates[0], "symbol": "AAA", "target_weight": 1.0},
            {"date": dates[1], "symbol": "BBB", "target_weight": 1.0},
        ]
    )
    result = run_backtest(
        panel,
        targets,
        costs={"commission_bps": 0, "half_spread_bps": 0, "slippage_bps": 0},
    )
    positions = result.weights.pivot(index="date", columns="symbol", values="shares")

    assert positions.loc[dates[1], "AAA"] > 0
    assert positions.loc[dates[2], "AAA"] == pytest.approx(0.0)
    assert positions.loc[dates[2], "BBB"] > 0


def test_missing_open_for_a_required_fill_fails_visibly() -> None:
    panel = _panel({"AAA": [(100.0, 100.0), (100.0, 100.0)]})
    dates = sorted(panel["date"].unique())
    panel.loc[panel["date"] == dates[1], "open"] = np.nan
    targets = pd.DataFrame({"date": [dates[0]], "symbol": ["AAA"], "target_weight": [1.0]})

    with pytest.raises(ValueError, match="invalid open prices"):
        run_backtest(panel, targets, costs={})


def test_terminal_liquidation_ends_the_research_period_without_stale_holdings() -> None:
    panel = _panel({"AAA": [(100.0, 100.0)] * 7})
    dates = sorted(panel["date"].unique())
    targets = pd.DataFrame({"date": [dates[0]], "symbol": ["AAA"], "target_weight": [1.0]})

    result = run_backtest(
        panel,
        targets,
        costs={"commission_bps": 0, "half_spread_bps": 0, "slippage_bps": 0},
        rebalance_frequency=2,
        liquidate_at_end=True,
    )

    # Decision day 0 fills day 1; terminal zero is decided day 2 and fills day 3.
    assert result.equity_curve["date"].iloc[-1] == dates[3]
    latest = result.weights[result.weights["date"] == dates[3]]
    assert latest["shares"].abs().sum() == pytest.approx(0.0)
    terminal_trade = result.trades[result.trades["date"] == dates[3]]
    assert terminal_trade["filled_shares"].sum() < 0


@pytest.mark.parametrize(
    ("target_factory", "message"),
    [
        (lambda date: pd.DataFrame({"date": [date], "symbol": ["AAA"]}), "missing columns"),
        (
            lambda date: pd.DataFrame(columns=["date", "symbol", "target_weight"]),
            "at least one row",
        ),
        (
            lambda date: pd.DataFrame(
                {
                    "date": [date, date],
                    "symbol": ["AAA", "AAA"],
                    "target_weight": [0.5, 0.5],
                }
            ),
            "duplicate",
        ),
        (
            lambda date: pd.DataFrame(
                {"date": [date], "symbol": ["AAA"], "target_weight": [np.nan]}
            ),
            "must be finite",
        ),
        (
            lambda date: pd.DataFrame(
                {"date": [date], "symbol": ["MISSING"], "target_weight": [1.0]}
            ),
            "missing from panel",
        ),
        (
            lambda date: pd.DataFrame(
                {
                    "date": [pd.Timestamp(date) + pd.Timedelta(days=5)],
                    "symbol": ["AAA"],
                    "target_weight": [1.0],
                }
            ),
            "not trading sessions",
        ),
    ],
)
def test_target_contract_rejects_ambiguous_panels(target_factory, message: str) -> None:
    panel = _panel({"AAA": [(100.0, 100.0)] * 3})
    targets = target_factory(panel["date"].min())
    with pytest.raises(ValueError, match=message):
        run_backtest(panel, targets)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"execution_lag": 0}, "execution_lag"),
        ({"execution_lag": True}, "execution_lag"),
        ({"rebalance_frequency": 0}, "rebalance_frequency"),
        ({"rebalance_frequency": True}, "rebalance_frequency"),
        ({"liquidate_at_end": "yes"}, "liquidate_at_end"),
        ({"initial_capital": np.inf}, "initial_capital"),
        ({"initial_capital": True}, "initial_capital"),
        (
            {"execution": {"missing_price_policy": "skip"}},
            "requires missing_price_policy='raise'",
        ),
        ({"risk": {"vol_target": -0.1}}, "volatility-target settings"),
        ({"risk": {"vol_target": "10%"}}, "vol_target must be numeric"),
        ({"risk": {"vol_target": 0.1, "vol_lookback": 1.5}}, "must be an integer"),
        ({"risk": {"drawdown_deleverage": 0.1, "drawdown_cut": 2.0}}, "drawdown"),
    ],
)
def test_backtest_configuration_fails_closed(kwargs, message: str) -> None:
    panel = _panel({"AAA": [(100.0, 100.0)] * 5})
    targets = pd.DataFrame(
        {"date": [panel["date"].min()], "symbol": ["AAA"], "target_weight": [1.0]}
    )
    with pytest.raises(ValueError, match=message):
        run_backtest(panel, targets, **kwargs)


def test_terminal_liquidation_requires_sufficient_future_sessions() -> None:
    panel = _panel({"AAA": [(100.0, 100.0)] * 2})
    targets = pd.DataFrame(
        {"date": [panel["date"].min()], "symbol": ["AAA"], "target_weight": [1.0]}
    )
    with pytest.raises(ValueError, match="enough sessions"):
        run_backtest(panel, targets, liquidate_at_end=True)
