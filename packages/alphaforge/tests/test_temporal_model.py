"""Unit + integration tests for TemporalAlphaNet.

All tests are skipped when torch is not installed (it is an optional extra);
CI installs it so these run there.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

torch = pytest.importorskip("torch")

from alphaforge.models.base import ModelError  # noqa: E402
from alphaforge.models.registry import create_model  # noqa: E402
from alphaforge.models.temporal import TemporalAlphaModel  # noqa: E402

FAST_PARAMS = dict(
    seq_len=10,
    hidden_size=16,
    n_blocks=2,
    max_epochs=15,
    patience=15,
    dates_per_batch=8,
    lr=3e-3,
    dropout=0.0,
    seed=7,
)


def make_learnable_data(
    n_symbols: int = 8, n_days: int = 160, n_features: int = 4, seed: int = 0
) -> tuple[pd.DataFrame, pd.Series]:
    """Panel where the target is the 5-day mean of feature 0 — learnable only
    by a model that actually uses the temporal dimension."""
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2022-01-03", periods=n_days)
    symbols = [f"S{i}" for i in range(n_symbols)]
    frames = []
    for sym in symbols:
        feats = rng.normal(0, 1, size=(n_days, n_features))
        signal = pd.Series(feats[:, 0]).rolling(5).mean().fillna(0).to_numpy()
        y = signal + rng.normal(0, 0.1, n_days)
        block = pd.DataFrame(feats, columns=[f"f{j}" for j in range(n_features)])
        block["date"] = dates
        block["symbol"] = sym
        block["y"] = y
        frames.append(block)
    data = pd.concat(frames).sort_values(["date", "symbol"]).reset_index(drop=True)
    X = data[[f"f{j}" for j in range(n_features)]].copy()
    X.index = pd.MultiIndex.from_frame(data[["date", "symbol"]])
    return X, pd.Series(data["y"].to_numpy(), index=X.index)


def _oos_rank_ic(model: TemporalAlphaModel, X: pd.DataFrame, y: pd.Series) -> float:
    preds = model.predict(X)
    frame = pd.DataFrame({"pred": preds, "y": y.to_numpy(), "date": X.index.get_level_values(0)})
    ics = [
        g["pred"].rank().corr(g["y"].rank())
        for _, g in frame.groupby("date")
        if g["pred"].std() > 0
    ]
    return float(np.nanmean(ics))


def test_registry_creates_temporal_model():
    model = create_model("temporal_alpha", max_epochs=1)
    assert isinstance(model, TemporalAlphaModel)
    assert model.needs_sequence_index is True


def test_training_loop_learns_a_temporal_signal():
    X, y = make_learnable_data()
    dates = X.index.get_level_values(0).unique().sort_values()
    train_dates, test_dates = dates[:120], dates[120:]
    X_tr = X[X.index.get_level_values(0).isin(train_dates)]
    y_tr = y[y.index.get_level_values(0).isin(train_dates)]
    X_te = X[X.index.get_level_values(0).isin(test_dates)]
    y_te = y[y.index.get_level_values(0).isin(test_dates)]

    model = TemporalAlphaModel(**FAST_PARAMS).fit(X_tr, y_tr)
    history = model.history_
    assert history is not None
    assert len(history.train_loss) >= 2
    assert history.best_epoch >= 1
    assert np.isfinite(history.val_rank_ic).any()
    # the signal is strong by construction; a working temporal model finds it
    assert _oos_rank_ic(model, X_te, y_te) > 0.3


def test_fit_and_predict_are_deterministic():
    X, y = make_learnable_data()
    params = dict(FAST_PARAMS, max_epochs=3)
    a = TemporalAlphaModel(**params).fit(X, y).predict(X)
    b = TemporalAlphaModel(**params).fit(X, y).predict(X)
    np.testing.assert_allclose(a, b, rtol=1e-6)


def test_predictions_are_causal():
    X, y = make_learnable_data()
    params = dict(FAST_PARAMS, max_epochs=3)
    model = TemporalAlphaModel(**params).fit(X, y)
    base = model.predict(X)

    dates = X.index.get_level_values(0)
    cutoff = dates.unique().sort_values()[100]
    mutated = X.copy()
    mutated.loc[dates > cutoff, :] = mutated.loc[dates > cutoff, :] * 7.0
    changed = model.predict(mutated)
    before = np.asarray(dates <= cutoff)
    np.testing.assert_allclose(
        base[before],
        changed[before],
        rtol=1e-6,
        err_msg="prediction at t depends on features after t (LOOKAHEAD LEAK)",
    )
    assert not np.allclose(base[~before], changed[~before])


def test_save_load_roundtrip(tmp_path):
    X, y = make_learnable_data()
    model = TemporalAlphaModel(**dict(FAST_PARAMS, max_epochs=2)).fit(X, y)
    path = tmp_path / "checkpoint.pt"
    model.save(path)
    with pytest.raises(ModelError, match="trusted=True"):
        TemporalAlphaModel.load(path)
    restored = TemporalAlphaModel.load(path, trusted=True)
    np.testing.assert_allclose(model.predict(X), restored.predict(X), rtol=1e-6)


def test_multi_task_aux_targets():
    X, y = make_learnable_data()
    aux = pd.DataFrame({"aux_1": y.to_numpy() * 0.5 + 0.01}, index=X.index)
    model = TemporalAlphaModel(**dict(FAST_PARAMS, max_epochs=2)).fit(X, y, aux_y=aux)
    assert model.n_outputs_ == 2
    assert len(model.predict(X)) == len(X)  # predictions come from the primary head


def test_walk_forward_integration(small_features, small_labels):
    """The model must run through the real walk-forward driver unchanged."""
    from alphaforge.training import run_walk_forward

    result = run_walk_forward(
        small_features,
        small_labels,
        model_specs=[
            {
                "name": "temporal_alpha",
                "params": dict(FAST_PARAMS, max_epochs=2, seq_len=8),
            }
        ],
        target="fwd_ret_5",
        config={
            "scheme": "expanding",
            "min_train_days": 180,
            "test_days": 40,
            "step_days": 40,
            "embargo_days": 20,
            "max_windows": 1,
        },
        max_horizon=20,
    )
    preds = result.predictions
    assert not preds.empty
    assert set(preds["model"]) == {"temporal_alpha"}
    assert preds["prediction"].notna().all()
    assert not result.metrics.empty
