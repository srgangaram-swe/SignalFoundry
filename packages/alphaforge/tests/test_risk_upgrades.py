"""Tests for backtest/risk upgrades: leverage costing, trimming, regimes, stress."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from alphaforge.backtesting import run_backtest
from alphaforge.models.baselines import MomentumBaseline, ZeroBaseline
from alphaforge.models.ensemble import EnsembleModel
from alphaforge.risk import (
    exposure_summary,
    performance_summary,
    regime_performance,
    stress_test_summary,
)


def _flat_panel(n_days: int = 300, seed: int = 4) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2021-01-04", periods=n_days)
    frames = []
    for sym in ["AAA", "BBB", "BENCH"]:
        close = 100 * np.exp(np.cumsum(rng.normal(0.0002, 0.01, n_days)))
        frames.append(
            pd.DataFrame(
                {
                    "date": dates,
                    "symbol": sym,
                    "open": close,
                    "high": close * 1.01,
                    "low": close * 0.99,
                    "close": close,
                    "volume": 1e6,
                }
            )
        )
    return pd.concat(frames, ignore_index=True)


def _constant_weights(panel: pd.DataFrame) -> pd.DataFrame:
    dates = sorted(panel["date"].unique())
    rows = []
    for d in dates[50:]:
        rows.append({"date": d, "symbol": "AAA", "target_weight": 0.5})
        rows.append({"date": d, "symbol": "BBB", "target_weight": -0.5})
    return pd.DataFrame(rows)


def test_vol_targeting_changes_generate_costed_turnover():
    panel = _flat_panel()
    weights = _constant_weights(panel)
    no_vt = run_backtest(panel, weights, costs={"commission_bps": 10}, risk={})
    with_vt = run_backtest(
        panel, weights, costs={"commission_bps": 10}, risk={"vol_target": 0.05, "vol_lookback": 20}
    )
    # A self-financing book drifts away from even constant target weights, so
    # ordinary rebalances are no longer (incorrectly) free. Volatility
    # targeting introduces additional exposure changes and therefore costs.
    later_costs_no_vt = no_vt.equity_curve["transaction_cost"].iloc[60:].sum()
    later_costs_vt = with_vt.equity_curve["transaction_cost"].iloc[60:].sum()
    assert later_costs_no_vt > 0.0
    assert later_costs_vt > later_costs_no_vt


def test_performance_summary_trims_dead_leading_period():
    panel = _flat_panel()
    weights = _constant_weights(panel)  # first position on day 51 (+lag)
    result = run_backtest(panel, weights, costs={}, risk={})
    trimmed = performance_summary(result.equity_curve, trim_inactive=True)
    untrimmed = performance_summary(result.equity_curve, trim_inactive=False)
    assert trimmed["n_days"] < untrimmed["n_days"]
    assert trimmed["average_gross_exposure"] > untrimmed["average_gross_exposure"]


def test_total_return_includes_the_first_active_day():
    curve = pd.DataFrame(
        {
            "date": pd.bdate_range("2024-01-02", periods=2),
            "return": [0.10, 0.10],
            "equity": [110.0, 121.0],
            "gross_exposure": [1.0, 1.0],
        }
    )
    assert performance_summary(curve)["total_return"] == pytest.approx(0.21)


def test_drawdown_deleverage_reduces_exposure():
    panel = _flat_panel(seed=9)
    weights = _constant_weights(panel)
    plain = run_backtest(panel, weights, costs={}, risk={})
    guarded = run_backtest(panel, weights, costs={}, risk={"drawdown_deleverage": 0.0001})
    # with an absurdly tight threshold the control must trigger somewhere
    assert guarded.equity_curve["leverage"].min() < plain.equity_curve["leverage"].min()


def test_regime_performance_splits_days():
    panel = _flat_panel()
    weights = _constant_weights(panel)
    result = run_backtest(panel, weights, costs={}, risk={})
    dates = pd.to_datetime(result.equity_curve["date"])
    regime = pd.Series(np.where(np.arange(len(dates)) % 2 == 0, "calm", "stress"), index=dates)
    table = regime_performance(result.equity_curve, regime)
    assert set(table["regime"]) == {"calm", "stress"}
    assert table["n_days"].sum() == performance_summary(result.equity_curve)["n_days"]


def test_stress_test_uses_betas():
    weights = pd.DataFrame(
        {
            "date": pd.to_datetime(["2021-06-01", "2021-06-01"]),
            "symbol": ["AAA", "BBB"],
            "weight": [0.5, 0.5],
        }
    )
    betas = pd.Series({"AAA": 2.0, "BBB": 0.0})
    table = stress_test_summary(weights, [{"name": "crash", "market_shock": -0.10}], betas=betas)
    assert table["portfolio_beta"].iloc[0] == pytest.approx(1.0)
    assert table["estimated_portfolio_return"].iloc[0] == pytest.approx(-0.10)


def test_exposure_summary_concentration():
    weights = pd.DataFrame(
        {
            "date": pd.to_datetime(["2021-06-01"] * 4),
            "symbol": list("ABCD"),
            "weight": [0.25, 0.25, 0.25, 0.25],
        }
    )
    summary = exposure_summary(weights)
    assert summary["n_positions"] == 4
    assert summary["hhi_concentration"] == pytest.approx(0.25)  # 1/n_eff with n_eff=4


def test_ic_weighted_ensemble_prefers_the_skilled_member():
    rng = np.random.default_rng(12)
    n = 2000
    X = pd.DataFrame({"momentum_20": rng.normal(0, 1, n), "noise": rng.normal(0, 1, n)})
    y = pd.Series(0.05 * X["momentum_20"] + rng.normal(0, 0.02, n))
    skilled = MomentumBaseline(feature="momentum_20", scale=0.05)
    unskilled = ZeroBaseline()
    ens = EnsembleModel([skilled, unskilled], weighting="ic").fit(X, y)
    assert ens.weights[0] > 0.7
    assert ens.member_ics_ is not None
    assert ens.member_ics_[0] > 0.5
