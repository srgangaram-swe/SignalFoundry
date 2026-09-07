from __future__ import annotations

import pytest

from alphaforge.data import SyntheticMarketConfig, generate_synthetic_market
from alphaforge.features import build_features
from alphaforge.labels.labels import build_labels

SMALL_FEATURE_CONFIG = {
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
}


@pytest.fixture(scope="session")
def small_panel():
    return generate_synthetic_market(SyntheticMarketConfig(n_symbols=6, n_days=280, seed=7))


@pytest.fixture(scope="session")
def small_features(small_panel):
    return build_features(small_panel, benchmark_symbol="BENCH", config=SMALL_FEATURE_CONFIG)


@pytest.fixture(scope="session")
def small_labels(small_panel):
    return build_labels(small_panel, benchmark_symbol="BENCH", horizons=[1, 5, 20])
