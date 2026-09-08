from __future__ import annotations

import pandas as pd
import pytest

from alphaforge.paper import simulate_paper_trading


def _paper_panel() -> pd.DataFrame:
    dates = pd.bdate_range("2024-03-01", periods=8)
    rows = []
    for i, date in enumerate(dates):
        open_price = 200.0 if i >= 5 else 100.0
        rows.append(
            {
                "date": date,
                "symbol": "AAA",
                "open": open_price,
                "high": open_price,
                "low": open_price,
                "close": open_price,
                "volume": 1_000.0,
            }
        )
    return pd.DataFrame(rows)


def test_paper_replay_uses_the_shared_future_open_contract() -> None:
    panel = _paper_panel()
    dates = sorted(panel["date"].unique())
    targets = pd.DataFrame({"date": [dates[4]], "symbol": ["AAA"], "target_weight": [1.0]})

    orders, state = simulate_paper_trading(
        panel,
        targets,
        costs={"commission_bps": 0, "half_spread_bps": 0, "slippage_bps": 0},
    )

    assert len(orders) == 1
    assert orders.loc[0, "decision_date"] == dates[4]
    assert orders.loc[0, "date"] == dates[5]
    assert orders.loc[0, "reference_price"] == 200.0
    assert orders.loc[0, "status"] == "SIMULATED_FILL"
    assert state.loc[0, "shares"] == pytest.approx(5_000.0)


def test_paper_replay_partial_fill_uses_lagged_not_same_day_volume() -> None:
    panel = _paper_panel()
    dates = sorted(panel["date"].unique())
    targets = pd.DataFrame({"date": [dates[4]], "symbol": ["AAA"], "target_weight": [1.0]})
    execution = {
        "adv_lookback": 3,
        "volatility_lookback": 2,
        "max_participation_rate": 0.10,
    }

    baseline, _ = simulate_paper_trading(panel, targets, execution=execution)
    mutated = panel.copy()
    mutated.loc[mutated["date"] == dates[5], "volume"] = 1_000_000_000.0
    changed, _ = simulate_paper_trading(mutated, targets, execution=execution)

    assert baseline.loc[0, "status"] == "SIMULATED_PARTIAL"
    assert baseline.loc[0, "simulated_shares"] == pytest.approx(100.0)
    assert changed.loc[0, "simulated_shares"] == pytest.approx(100.0)
    assert baseline.loc[0, "lagged_adv_shares"] == changed.loc[0, "lagged_adv_shares"]


def test_paper_replay_validates_lookback() -> None:
    panel = _paper_panel()
    first = panel["date"].min()
    targets = pd.DataFrame({"date": [first], "symbol": ["AAA"], "target_weight": [1.0]})
    with pytest.raises(ValueError, match="lookback_days"):
        simulate_paper_trading(panel, targets, lookback_days=0)
