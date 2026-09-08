from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from alphaforge.features import (
    FeatureCache,
    FeatureCacheError,
    FeatureContractError,
    FittedFeatureTransformer,
    FittedTransformSpec,
    build_default_registry,
    materialize_feature_set,
    validate_feature_frame,
)
from alphaforge.training import run_walk_forward


def _without_hmm() -> dict:
    return {
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
        "hmm_regime": False,
        "cross_sectional": True,
    }


def test_registry_semantic_identity_is_stable_and_parameter_sensitive():
    config = _without_hmm()
    reordered = dict(reversed(list(config.items())))
    first = build_default_registry(config)
    second = build_default_registry(reordered)
    changed = build_default_registry({**config, "rsi_window": 8})

    assert first.registry_id == second.registry_id
    assert first.registry_id != changed.registry_id
    assert first.resolve("rsi", "1.0.0").parameters == {
        "window": 7,
        "smoothing": "wilder-ewm",
    }
    with pytest.raises(TypeError):
        first.resolve("rsi", "1.0.0").parameters["window"] = 99  # type: ignore[index]
    with pytest.raises(AttributeError):
        first.version = "9.0.0"  # type: ignore[misc]
    with pytest.raises(FeatureContractError, match="unknown feature/version"):
        first.resolve("rsi", "9.0.0")
    with pytest.raises(FeatureContractError, match="unsupported feature registry"):
        build_default_registry({**config, "registry_version": "9.0.0"})


def test_materialization_cache_key_binds_data_code_dates_universe_and_parameters(
    small_panel, tmp_path
):
    config = _without_hmm()
    cache = FeatureCache(tmp_path / "cache")
    first = materialize_feature_set(
        small_panel,
        "BENCH",
        config,
        dataset_id="synthetic-test-panel",
        code_version="a" * 40,
        cache=cache,
    )
    replay = materialize_feature_set(
        small_panel,
        "BENCH",
        config,
        dataset_id="synthetic-test-panel",
        code_version="a" * 40,
        cache=cache,
    )
    mutated = small_panel.copy()
    mutated.loc[mutated.index[-1], "close"] *= 1.01
    changed = materialize_feature_set(
        mutated,
        "BENCH",
        config,
        dataset_id="synthetic-test-panel",
        code_version="a" * 40,
    )

    assert not first.cache_hit
    assert replay.cache_hit
    assert first.cache_key == replay.cache_key
    assert first.cache_key != changed.cache_key
    pd.testing.assert_frame_equal(first.frame, replay.frame)


def test_corrupt_cache_and_feature_output_fail_closed(small_panel, tmp_path):
    config = _without_hmm()
    cache = FeatureCache(tmp_path / "cache")
    result = materialize_feature_set(
        small_panel,
        "BENCH",
        config,
        dataset_id="synthetic-test-panel",
        code_version="b" * 40,
        cache=cache,
    )
    with pytest.raises(FeatureCacheError, match="failed validation"):
        FeatureCache(tmp_path / "cache", max_artifact_bytes=1).load(result.lineage, result.registry)
    manifest_path = tmp_path / "cache" / result.cache_key / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["cache_key"] = "0" * 64
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(FeatureCacheError, match="mismatch"):
        cache.load(result.lineage, result.registry)

    invalid = result.frame.copy()
    invalid.loc[invalid.index[-1], result.registry.output_columns[0]] = np.inf
    with pytest.raises(FeatureContractError, match="infinity"):
        validate_feature_frame(invalid, result.registry)

    incompatible = result.frame.drop(columns=[result.registry.output_columns[-1]])
    with pytest.raises(FeatureContractError, match="schema mismatch"):
        validate_feature_frame(incompatible, result.registry)


def test_materialization_rejects_insufficient_warmup(small_panel):
    short = small_panel[small_panel["date"].isin(sorted(small_panel["date"].unique())[:30])]
    with pytest.raises(FeatureContractError, match="insufficient feature warm-up"):
        materialize_feature_set(
            short,
            "BENCH",
            _without_hmm(),
            dataset_id="short-panel",
            code_version="c" * 40,
        )


def test_fitted_transform_has_reference_values_and_deterministic_state():
    frame = pd.DataFrame(
        {
            "a": [1.0, 2.0, np.nan, 4.0, 5.0],
            "b": [10.0, 20.0, 30.0, 40.0, 50.0],
        }
    )
    dates = pd.bdate_range("2025-01-01", periods=len(frame))
    spec = FittedTransformSpec(enabled=True, min_fit_rows=5)
    first = FittedFeatureTransformer(spec)
    transformed = first.fit_transform(frame, dates)
    second = FittedFeatureTransformer(spec).fit(frame, dates)

    expected_a = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    expected_a = (expected_a - expected_a.mean()) / expected_a.std(ddof=0)
    np.testing.assert_allclose(transformed["a"], expected_a, atol=1e-12)
    assert first.state_ == second.state_
    assert first.state_ is not None
    assert first.state_.fit_end == dates[-1].isoformat()


def test_holdout_mutation_cannot_change_fitted_state():
    rng = np.random.default_rng(19)
    train = pd.DataFrame(rng.normal(size=(80, 4)), columns=list("abcd"))
    holdout = pd.DataFrame(rng.normal(size=(20, 4)), columns=list("abcd"))
    dates = pd.bdate_range("2024-01-01", periods=len(train))
    transformer = FittedFeatureTransformer(
        FittedTransformSpec(enabled=True, pca_components=2, min_fit_rows=64)
    ).fit(train, dates)
    state_before = transformer.state_
    train_before = transformer.transform(train)
    transformer.transform(holdout * 1_000_000)

    assert transformer.state_ == state_before
    pd.testing.assert_frame_equal(transformer.transform(train), train_before)


@pytest.mark.parametrize(
    ("frame", "message"),
    [
        (pd.DataFrame({"a": [1.0, np.inf]}), "infinity"),
        (pd.DataFrame({"a": [np.nan, np.nan]}), "at least one finite"),
    ],
)
def test_fitted_transform_rejects_nonfinite_or_unlearnable_columns(frame, message):
    transformer = FittedFeatureTransformer(FittedTransformSpec(enabled=True, min_fit_rows=2))
    with pytest.raises(FeatureContractError, match=message):
        transformer.fit(frame, pd.bdate_range("2024-01-01", periods=2))


def test_fitted_transform_rejects_inference_before_fit_and_schema_drift():
    transformer = FittedFeatureTransformer(FittedTransformSpec(enabled=True, min_fit_rows=3))
    frame = pd.DataFrame({"a": [1.0, 2.0, 3.0], "b": [3.0, 4.0, 5.0]})
    with pytest.raises(FeatureContractError, match="fit before transform"):
        transformer.transform(frame)
    transformer.fit(frame, pd.bdate_range("2024-01-01", periods=3))
    with pytest.raises(FeatureContractError, match="schema mismatch"):
        transformer.transform(frame[["b", "a"]])


@pytest.mark.parametrize(
    "config",
    [
        {"version": "9.0.0"},
        {"variance_threshold": np.nan},
        {"pca_components": 1.5},
        {"pca_components": True},
        {"pca_whiten": True},
        {"min_fit_rows": 0},
        {"undeclared_policy": "unsafe"},
    ],
)
def test_fitted_transform_rejects_unsafe_parameters(config: dict[str, object]):
    with pytest.raises(FeatureContractError):
        FittedTransformSpec.from_config(config)


def test_walk_forward_records_train_only_fitted_state(small_features, small_labels):
    result = run_walk_forward(
        small_features.drop(columns=["hmm_stress_prob"]),
        small_labels,
        model_specs=[{"name": "ridge", "params": {"alpha": 2.0}}],
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
        transform_config={
            "version": "1.0.0",
            "enabled": True,
            "imputation": "median",
            "standardize": True,
            "variance_threshold": None,
            "pca_components": 0.95,
            "pca_whiten": False,
            "min_fit_rows": 64,
        },
    )

    assert len(result.transformations) == len(result.windows) == 2
    merged = result.transformations.merge(result.windows, on="window_id")
    assert (pd.to_datetime(merged["fit_end"]) <= pd.to_datetime(merged["train_end"])).all()
    assert (pd.to_datetime(merged["fit_end"]) < pd.to_datetime(merged["test_start"])).all()
    assert merged["state_id"].str.fullmatch(r"[0-9a-f]{64}").all()
