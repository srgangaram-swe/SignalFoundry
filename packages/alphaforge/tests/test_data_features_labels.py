from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from conftest import SMALL_FEATURE_CONFIG

from alphaforge.data import REQUIRED_COLUMNS, validate_panel
from alphaforge.features import build_features


def test_synthetic_panel_uses_canonical_schema(small_panel):
    validated = validate_panel(small_panel)
    assert list(validated.columns) == REQUIRED_COLUMNS
    assert not validated.duplicated(["date", "symbol"]).any()
    assert (validated["close"] > 0).all()


def test_features_are_invariant_to_future_data_mutation(small_panel):
    dates = sorted(small_panel["date"].unique())
    cutoff = dates[120]

    baseline = build_features(small_panel, "BENCH", SMALL_FEATURE_CONFIG)
    mutated = small_panel.copy()
    future = mutated["date"] > cutoff
    mutated.loc[future, "close"] = mutated.loc[future, "close"] * np.linspace(
        1.5, 0.6, future.sum()
    )
    mutated.loc[future, "volume"] = mutated.loc[future, "volume"] * 3.0
    changed = build_features(mutated, "BENCH", SMALL_FEATURE_CONFIG)

    cols = sorted(set(baseline.columns) - {"date", "symbol"})
    left = baseline.loc[baseline["date"] <= cutoff, ["date", "symbol", *cols]].reset_index(
        drop=True
    )
    right = changed.loc[changed["date"] <= cutoff, ["date", "symbol", *cols]].reset_index(drop=True)
    pd.testing.assert_frame_equal(
        left, right, check_dtype=False, check_exact=False, atol=1e-12, rtol=1e-12
    )


def test_forward_return_label_alignment(small_panel, small_labels):
    symbol = "SYN000"
    horizon = 5
    prices = (
        small_panel.loc[small_panel["symbol"] == symbol].sort_values("date").reset_index(drop=True)
    )
    labels = (
        small_labels.loc[small_labels["symbol"] == symbol]
        .sort_values("date")
        .reset_index(drop=True)
    )
    row = 40
    expected = prices.loc[row + horizon, "close"] / prices.loc[row, "close"] - 1.0
    actual = labels.loc[row, f"fwd_ret_{horizon}"]
    assert actual == pytest.approx(expected)
