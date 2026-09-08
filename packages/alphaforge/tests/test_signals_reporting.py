from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from alphaforge.reporting import write_markdown_report
from alphaforge.signals import apply_regime_filter, build_signals, select_model_predictions


def _predictions() -> pd.DataFrame:
    rows = []
    for date in pd.bdate_range("2024-01-02", periods=2):
        for i in range(10):
            rows.append(
                {
                    "date": date,
                    "symbol": f"S{i:02d}",
                    "prediction": float(i - 4.5),
                    "target": float(i) / 100,
                }
            )
    return pd.DataFrame(rows)


def test_model_selection_priority_and_validation() -> None:
    base = _predictions()
    panel = pd.concat(
        [base.assign(model=name) for name in ("zero_baseline", "gradient_boosting", "ensemble")],
        ignore_index=True,
    )
    assert set(select_model_predictions(panel)["model"]) == {"ensemble"}
    without_ensemble = panel[panel["model"] != "ensemble"]
    assert set(select_model_predictions(without_ensemble)["model"]) == {"gradient_boosting"}
    assert set(select_model_predictions(panel, model="zero_baseline")["model"]) == {"zero_baseline"}
    with pytest.raises(KeyError, match="not found"):
        select_model_predictions(panel, model="missing")
    pd.testing.assert_frame_equal(select_model_predictions(base), base)


@pytest.mark.parametrize(
    ("strategy", "params"),
    [
        ("long_short", {"quantile": 0.2}),
        ("long_only_topk", {"top_k": 3}),
        ("rank_weighted", {}),
        ("confidence_weighted", {"clip": 1.5}),
        ("threshold", {"threshold": 2.0, "allow_short": True}),
    ],
)
def test_signal_strategies_are_bounded_and_cross_sectional(strategy, params) -> None:
    signals = build_signals(_predictions(), strategy=strategy, params=params)
    assert len(signals) == 20
    assert signals["signal"].between(-1.0, 1.0).all()
    if strategy == "long_short":
        counts = signals.groupby("date")["signal"].agg(
            longs=lambda values: int((values > 0).sum()),
            shorts=lambda values: int((values < 0).sum()),
        )
        assert (counts == 2).all().all()
    if strategy == "long_only_topk":
        assert (signals.groupby("date")["signal"].sum() == 3.0).all()


def test_signal_edge_cases_and_errors_are_explicit() -> None:
    constant = _predictions().assign(prediction=1.0)
    assert build_signals(constant, strategy="rank_weighted")["signal"].eq(0).all()
    assert build_signals(constant, strategy="confidence_weighted")["signal"].eq(0).all()
    long_only = build_signals(
        _predictions(),
        strategy="threshold",
        params={"threshold": 2.0, "allow_short": False},
    )
    assert (long_only["signal"] >= 0).all()
    with pytest.raises(KeyError, match="missing columns"):
        build_signals(_predictions().drop(columns="symbol"))
    with pytest.raises(ValueError, match="unknown signal strategy"):
        build_signals(_predictions(), strategy="clairvoyant")
    empty = build_signals(_predictions().assign(prediction=np.nan))
    assert empty.empty


def test_regime_filter_uses_probabilities_fallback_and_noop() -> None:
    signals = build_signals(_predictions(), strategy="long_short")
    dates = sorted(signals["date"].unique())
    features = pd.DataFrame(
        {
            "date": np.repeat(dates, 2),
            "symbol": ["S00", "S01"] * 2,
            "hmm_stress_prob": [0.25, 0.25, 0.80, 0.80],
        }
    )
    filtered = apply_regime_filter(signals, features, max_stress=0.7)
    assert filtered.loc[filtered["date"] == dates[0], "signal"].abs().max() == pytest.approx(0.75)
    assert filtered.loc[filtered["date"] == dates[1], "signal"].eq(0).all()

    fallback = features.drop(columns="hmm_stress_prob").assign(high_vol_regime=1.0)
    assert apply_regime_filter(signals, fallback)["signal"].eq(0).all()
    pd.testing.assert_frame_equal(
        apply_regime_filter(signals, features.drop(columns="hmm_stress_prob")),
        signals,
    )


def test_research_report_includes_execution_attribution_and_capacity(tmp_path) -> None:
    summary = {
        "total_return": 0.05,
        "sharpe": 0.8,
        "execution_orders": 4,
        "execution_fill_ratio": 0.9,
        "total_trading_cost_dollars": 125.0,
        "deflated_sharpe_prob": 0.6,
        "n_trials": 3,
    }
    (tmp_path / "backtest_summary.json").write_text(json.dumps(summary))
    (tmp_path / "overfitting.json").write_text(
        json.dumps({"pbo": 0.2, "n_combinations": 10, "n_strategies": 3})
    )
    pd.DataFrame({"model": ["ridge"], "rank_ic": [0.03]}).to_csv(
        tmp_path / "model_metrics.csv", index=False
    )
    pd.DataFrame({"model": ["ridge"], "mean_ic": [0.03]}).to_csv(
        tmp_path / "ic_summary.csv", index=False
    )
    pd.DataFrame({"horizon": [1], "mean_rank_ic": [0.03]}).to_csv(
        tmp_path / "ic_decay.csv", index=False
    )
    pd.DataFrame({"quantile": [1, 5], "mean_return": [-0.01, 0.02]}).to_csv(
        tmp_path / "quantile_returns.csv", index=False
    )
    pd.DataFrame({"regime": ["calm"], "sharpe": [1.0]}).to_csv(
        tmp_path / "regime_performance.csv", index=False
    )
    pd.DataFrame(
        {
            "decision_date": ["2024-01-02"],
            "fill_date": ["2024-01-03"],
            "symbol": ["AAA"],
            "status": ["partial"],
            "requested_notional": [100_000.0],
            "traded_notional": [90_000.0],
            "participation_rate": [0.1],
            "total_cost": [125.0],
        }
    ).to_csv(tmp_path / "fills.csv", index=False)
    pd.DataFrame(
        {
            "symbol": ["AAA", "BBB"],
            "market_pnl": [500.0, -200.0],
            "trading_cost": [100.0, 25.0],
            "net_pnl": [400.0, -225.0],
        }
    ).to_csv(tmp_path / "pnl_attribution.csv", index=False)
    pd.DataFrame(
        {
            "scenario_aum": [1_000_000.0],
            "fill_ratio": [0.9],
            "capacity_constrained_fraction": [0.1],
            "participation_p95": [0.08],
            "modeled_cost_bps_per_traded_notional": [12.0],
        }
    ).to_csv(tmp_path / "capacity_curve.csv", index=False)
    (tmp_path / "capacity_diagnostics.json").write_text(
        json.dumps({"assumptions": ["Sensitivity, not a forecast."]})
    )

    report_path = write_markdown_report(tmp_path)
    report = report_path.read_text()

    assert "## Execution and Accounting" in report
    assert "### P&L attribution by symbol" in report
    assert "## Capacity Sensitivity" in report
    assert "Sensitivity, not a forecast." in report
    assert "Educational research output" not in report  # canonical disclaimer has fuller wording
    assert "not financial advice" in report.lower()


def test_report_handles_an_empty_run_directory(tmp_path) -> None:
    report = write_markdown_report(tmp_path).read_text()
    assert "Backtest summary not found" in report
    assert "No execution summary available" in report
    assert "No capacity sensitivity available" in report
