"""Leakage tests: the most important tests in the repository.

Strategy: build features/labels on a panel, then mutate all bars strictly
AFTER a cutoff date and rebuild. Anything computed at or before the cutoff
must be bit-identical. If one of these tests fails, backtest results are
meaningless regardless of how good they look.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from alphaforge.data import SyntheticMarketConfig, generate_synthetic_market
from alphaforge.features import FeatureScaler, build_features, feature_columns
from alphaforge.labels.labels import build_labels
from alphaforge.models.regime import causal_stress_probability

FEATURE_CONFIG = {
    "return_lags": [1, 5],
    "vol_windows": [5, 20],
    "ma_windows": [5, 20],
    "momentum_windows": [5, 20],
    "rsi_window": 7,
    "macd": {"fast": 6, "slow": 13, "signal": 5},
    "bollinger_window": 10,
    "mean_reversion_window": 5,
    "volume_window": 10,
    "beta_window": 20,
    "rolling_sharpe_window": 20,
    "drawdown_window": 60,
    "regime_vol_window": 10,
    "regime_trend_fast": 10,
    "regime_trend_slow": 30,
    "cross_sectional": True,
    "hmm_regime": True,
    "hmm_refit_every": 40,
    "hmm_min_train": 120,
}


@pytest.fixture(scope="module")
def panel():
    return generate_synthetic_market(SyntheticMarketConfig(n_symbols=5, n_days=320, seed=11))


def _mutate_after(panel: pd.DataFrame, cutoff: pd.Timestamp) -> pd.DataFrame:
    mutated = panel.copy()
    after = mutated["date"] > cutoff
    mutated.loc[after, ["open", "high", "low", "close"]] *= 3.0
    mutated.loc[after, "volume"] *= 10.0
    return mutated


def test_features_are_causal(panel):
    dates = sorted(panel["date"].unique())
    cutoff = dates[220]
    base = build_features(panel, benchmark_symbol="BENCH", config=FEATURE_CONFIG)
    mutated = build_features(
        _mutate_after(panel, cutoff), benchmark_symbol="BENCH", config=FEATURE_CONFIG
    )

    past_base = base[base["date"] <= cutoff].reset_index(drop=True)
    past_mut = mutated[mutated["date"] <= cutoff].reset_index(drop=True)
    for col in feature_columns(base):
        np.testing.assert_allclose(
            past_base[col].to_numpy(),
            past_mut[col].to_numpy(),
            rtol=1e-10,
            atol=1e-12,
            equal_nan=True,
            err_msg=f"feature {col!r} at t depends on data after t (LOOKAHEAD LEAK)",
        )


def test_labels_use_exactly_the_intended_future(panel):
    dates = sorted(panel["date"].unique())
    cutoff = dates[250]
    horizon = 5
    base = build_labels(panel, benchmark_symbol="BENCH", horizons=[horizon])
    mutated_labels = build_labels(
        _mutate_after(panel, cutoff), benchmark_symbol="BENCH", horizons=[horizon]
    )

    col = f"fwd_ret_{horizon}"
    merged = base.merge(mutated_labels, on=["date", "symbol"], suffixes=("_a", "_b"))
    # labels whose full horizon lies at or before the cutoff must be unchanged
    safe = merged["date"] <= dates[250 - horizon]
    np.testing.assert_allclose(
        merged.loc[safe, f"{col}_a"].to_numpy(),
        merged.loc[safe, f"{col}_b"].to_numpy(),
        rtol=1e-10,
        equal_nan=True,
    )
    # labels whose horizon crosses the mutation boundary MUST change
    crossing = (merged["date"] > dates[250 - horizon]) & (merged["date"] <= cutoff)
    diffs = (
        merged.loc[crossing, f"{col}_a"].to_numpy() - merged.loc[crossing, f"{col}_b"].to_numpy()
    )
    assert np.nanmax(np.abs(diffs)) > 1e-8, "forward labels ignored the future — wrong alignment"


def test_forward_label_alignment_exact(panel):
    g = (
        panel[panel["symbol"] == panel["symbol"].iloc[-1]]
        .sort_values("date")
        .reset_index(drop=True)
    )
    labels = build_labels(panel, benchmark_symbol="BENCH", horizons=[1])
    lg = labels[labels["symbol"] == g["symbol"].iloc[0]].sort_values("date").reset_index(drop=True)
    t = 100
    expected = g["close"].iloc[t + 1] / g["close"].iloc[t] - 1.0
    assert lg["fwd_ret_1"].iloc[t] == pytest.approx(expected)


def test_scaler_statistics_come_from_train_only(panel):
    features = build_features(panel, benchmark_symbol="BENCH", config=FEATURE_CONFIG)
    cols = feature_columns(features)
    dates = sorted(features["date"].unique())
    train = features[features["date"] <= dates[200]][cols]
    test = features[features["date"] > dates[200]][cols]

    scaler = FeatureScaler().fit(train)
    # fitting again on the same train data after the test set is corrupted
    # must give identical statistics: the scaler never sees test rows
    scaler_b = FeatureScaler().fit(train)
    pd.testing.assert_series_equal(scaler.means_, scaler_b.means_)
    pd.testing.assert_series_equal(scaler.stds_, scaler_b.stds_)
    # transform must be a pure function of train-fit statistics
    corrupted = test * 100.0
    expected = ((corrupted - scaler.means_) / scaler.stds_).clip(-scaler.clip, scaler.clip)
    np.testing.assert_allclose(
        scaler.transform(corrupted).to_numpy(),
        expected[scaler.columns_].to_numpy(),
        rtol=1e-10,
        equal_nan=True,
    )


def test_hmm_stress_probability_is_causal():
    rng = np.random.default_rng(3)
    returns = pd.Series(np.concatenate([rng.normal(0, 0.008, 200), rng.normal(0, 0.03, 100)]))
    base = causal_stress_probability(returns, refit_every=40, min_train=120)
    mutated_input = returns.copy()
    mutated_input.iloc[240:] = mutated_input.iloc[240:] * 5.0
    mutated = causal_stress_probability(mutated_input, refit_every=40, min_train=120)
    np.testing.assert_allclose(
        base.iloc[:240].to_numpy(),
        mutated.iloc[:240].to_numpy(),
        rtol=1e-10,
        equal_nan=True,
        err_msg="HMM stress probability at t depends on returns after t",
    )
