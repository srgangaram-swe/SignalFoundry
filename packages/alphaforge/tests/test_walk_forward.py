from __future__ import annotations

import pandas as pd
import pytest

from alphaforge.training import make_walk_forward_splits, run_walk_forward


def test_walk_forward_requires_embargo_at_least_max_horizon():
    dates = pd.bdate_range("2020-01-01", periods=180)
    cfg = {"min_train_days": 60, "test_days": 20, "step_days": 20, "embargo_days": 5}
    with pytest.raises(ValueError, match="embargo_days"):
        make_walk_forward_splits(dates, cfg, max_horizon=20)


def test_walk_forward_splits_have_embargo_gap():
    dates = pd.bdate_range("2020-01-01", periods=220)
    cfg = {"min_train_days": 80, "test_days": 20, "step_days": 20, "embargo_days": 20}
    windows = make_walk_forward_splits(dates, cfg, max_horizon=20)
    date_index = pd.Index(dates)
    assert windows
    for window in windows:
        train_end_idx = date_index.get_loc(window.train_end)
        test_start_idx = date_index.get_loc(window.test_start)
        assert test_start_idx - train_end_idx - 1 >= 20
        assert window.train_end < window.test_start


def test_walk_forward_outputs_oos_predictions(small_features, small_labels):
    result = run_walk_forward(
        small_features,
        small_labels,
        model_specs=[
            {"name": "zero_baseline"},
            {"name": "ridge", "params": {"alpha": 5.0}},
        ],
        target="fwd_ret_5",
        config={
            "scheme": "expanding",
            "min_train_days": 100,
            "test_days": 30,
            "step_days": 30,
            "embargo_days": 20,
            "max_windows": 2,
        },
        max_horizon=20,
    )
    assert not result.predictions.empty
    assert set(result.predictions["model"]) == {"zero_baseline", "ridge"}
    assert {"date", "symbol", "prediction", "target", "window_id"}.issubset(result.predictions)
    for _, window in result.windows.iterrows():
        preds = result.predictions[result.predictions["window_id"] == window["window_id"]]
        assert preds["date"].min() >= pd.Timestamp(window["test_start"])
        assert preds["date"].max() <= pd.Timestamp(window["test_end"])
