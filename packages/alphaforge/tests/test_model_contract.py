"""Tests for the unified model contract and naive baselines (SF-S2-MR4).

Covers, for every baseline and registry path: round-trip serialization and
byte-determinism, not-fitted / unsupported-shape / non-finite / label negative
cases, feature-schema checks, versioned metadata, deterministic predictions, and
the probability/uncertainty interface.
"""

from __future__ import annotations

import json

import joblib
import numpy as np
import pandas as pd
import pytest

from alphaforge.models import (
    CONTRACT_VERSION,
    AlphaModel,
    FeatureSchemaError,
    InvalidLabelError,
    ModelError,
    NotFittedError,
    ProbabilityNotSupportedError,
    available_models,
    create_model,
)
from alphaforge.models.base import ModelMetadata

# Baselines with JSON-safe, deterministic serialization.
BASELINE_NAMES = [
    "zero_baseline",
    "historical_mean",
    "lag_baseline",
    "moving_average_baseline",
    "momentum_baseline",
    "equal_probability",
    "buy_and_hold",
    "equal_weight",
]
CLASSIFIER_NAMES = {"equal_probability"}


@pytest.fixture
def frame() -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    rng = np.random.default_rng(7)
    n = 240
    X = pd.DataFrame(
        {
            "ret_1d": rng.normal(0, 0.01, n),
            "ma_ratio_20": rng.normal(0, 0.02, n),
            "momentum_20": rng.normal(0, 0.05, n),
            "extra": rng.normal(0, 1, n),
        }
    )
    y_reg = pd.Series(rng.normal(0, 0.01, n))
    y_cls = pd.Series((rng.random(n) > 0.5).astype(float))
    return X, y_reg, y_cls


def _fit(name: str, frame: tuple[pd.DataFrame, pd.Series, pd.Series]) -> AlphaModel:
    X, y_reg, y_cls = frame
    y = y_cls if name in CLASSIFIER_NAMES else y_reg
    return create_model(name).fit(X, y)


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


def test_all_baselines_registered() -> None:
    assert set(BASELINE_NAMES) <= set(available_models())


def test_unknown_model_raises() -> None:
    with pytest.raises(KeyError):
        create_model("no_such_model")


# ---------------------------------------------------------------------------
# Every baseline: fit / predict / metadata
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("name", BASELINE_NAMES)
def test_fit_predict_shape_and_determinism(
    name: str, frame: tuple[pd.DataFrame, pd.Series, pd.Series]
) -> None:
    X = frame[0]
    model = _fit(name, frame)
    first = model.predict(X)
    second = model.predict(X)
    assert first.shape == (len(X),)
    assert np.isfinite(first).all()
    np.testing.assert_array_equal(first, second)  # deterministic


@pytest.mark.parametrize("name", BASELINE_NAMES)
def test_metadata_is_versioned(name: str, frame: tuple[pd.DataFrame, pd.Series, pd.Series]) -> None:
    model = _fit(name, frame)
    meta = model.metadata()
    assert isinstance(meta, ModelMetadata)
    assert meta.name == name
    assert meta.contract_version == CONTRACT_VERSION
    assert meta.fitted is True
    assert meta.task in {"regression", "classification"}
    # Round-trips through a plain dict (deterministic under sort_keys).
    assert meta.to_dict()["name"] == name


# ---------------------------------------------------------------------------
# Serialization: round-trip + byte determinism (JSON baselines)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("name", BASELINE_NAMES)
def test_json_roundtrip_and_byte_determinism(
    name: str, frame: tuple[pd.DataFrame, pd.Series, pd.Series], tmp_path
) -> None:
    X = frame[0]
    model = _fit(name, frame)
    path_a = tmp_path / f"{name}_a.json"
    path_b = tmp_path / f"{name}_b.json"
    model.save(path_a)
    model.save(path_b)
    assert path_a.read_bytes() == path_b.read_bytes()  # byte-identical

    restored = AlphaModel.load(path_a)
    assert restored.metadata().fitted is True
    np.testing.assert_array_equal(restored.predict(X), model.predict(X))
    if name in CLASSIFIER_NAMES:
        np.testing.assert_array_equal(restored.predict_proba(X), model.predict_proba(X))


def test_sklearn_model_uses_joblib_fallback(
    frame: tuple[pd.DataFrame, pd.Series, pd.Series], tmp_path
) -> None:
    X, y_reg, _ = frame
    model = create_model("linear").fit(X, y_reg)
    path = tmp_path / "linear.joblib"
    model.save(path)
    assert path.read_bytes()[:1] != b"{"  # not the JSON container
    with pytest.raises(ModelError, match="trusted=True"):
        AlphaModel.load(path)
    restored = AlphaModel.load(path, trusted=True)
    np.testing.assert_allclose(restored.predict(X), model.predict(X), rtol=1e-9)


def test_load_rejects_non_model_payload(tmp_path) -> None:
    path = tmp_path / "junk.joblib"
    joblib.dump({"not": "a model"}, path)
    with pytest.raises(ModelError):
        AlphaModel.load(path, trusted=True)


def test_save_before_fit_is_rejected(tmp_path) -> None:
    with pytest.raises(NotFittedError):
        create_model("zero_baseline").save(tmp_path / "unfitted.json")


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("format", "unknown", "format"),
        ("format_version", 999, "version"),
        ("metadata.fitted", False, "not fitted"),
        ("metadata.contract_version", "999.0.0", "contract version"),
    ],
)
def test_json_artifact_schema_fails_closed(
    field: str,
    value: object,
    message: str,
    frame: tuple[pd.DataFrame, pd.Series, pd.Series],
    tmp_path,
) -> None:
    model = _fit("zero_baseline", frame)
    path = tmp_path / "model.json"
    model.save(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    if "." in field:
        parent, child = field.split(".", maxsplit=1)
        payload[parent][child] = value
    else:
        payload[field] = value
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ModelError, match=message):
        AlphaModel.load(path)


# ---------------------------------------------------------------------------
# Fitted-state semantics
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("name", BASELINE_NAMES)
def test_predict_before_fit_raises(
    name: str, frame: tuple[pd.DataFrame, pd.Series, pd.Series]
) -> None:
    X = frame[0]
    with pytest.raises(NotFittedError):
        create_model(name).predict(X)


def test_predict_during_fit_is_allowed(frame: tuple[pd.DataFrame, pd.Series, pd.Series]) -> None:
    X, y_reg, _ = frame

    class PredictsDuringFit(AlphaModel):
        name = "predicts_during_fit"
        feature_agnostic = True

        def fit(self, X: pd.DataFrame, y: pd.Series) -> PredictsDuringFit:
            # A model may call its own predict during fit (e.g. for early
            # stopping) without tripping the not-fitted guard.
            self.peek_ = self.predict(X)
            return self

        def predict(self, X: pd.DataFrame) -> np.ndarray:
            return np.zeros(len(X))

    model = PredictsDuringFit().fit(X, y_reg)
    assert model.peek_.shape == (len(X),)
    assert model.metadata().fitted is True


# ---------------------------------------------------------------------------
# Negative cases: shape, finiteness, labels, schema
# ---------------------------------------------------------------------------


def test_non_dataframe_x_rejected(frame: tuple[pd.DataFrame, pd.Series, pd.Series]) -> None:
    _, y_reg, _ = frame
    with pytest.raises(FeatureSchemaError):
        create_model("zero_baseline").fit(np.zeros((len(y_reg), 3)), y_reg)


def test_non_dataframe_predict_rejected(
    frame: tuple[pd.DataFrame, pd.Series, pd.Series],
) -> None:
    X, y_reg, _ = frame
    model = create_model("zero_baseline").fit(X, y_reg)
    with pytest.raises(FeatureSchemaError):
        model.predict(np.zeros((len(X), 3)))


def test_length_mismatch_rejected(frame: tuple[pd.DataFrame, pd.Series, pd.Series]) -> None:
    X, y_reg, _ = frame
    with pytest.raises(FeatureSchemaError):
        create_model("zero_baseline").fit(X, y_reg.iloc[:-1])


def test_misaligned_indices_rejected(frame: tuple[pd.DataFrame, pd.Series, pd.Series]) -> None:
    X, y_reg, _ = frame
    shifted = y_reg.copy()
    shifted.index = shifted.index + 1
    with pytest.raises(FeatureSchemaError, match="indices"):
        create_model("zero_baseline").fit(X, shifted)


def test_duplicate_feature_names_rejected(
    frame: tuple[pd.DataFrame, pd.Series, pd.Series],
) -> None:
    X, y_reg, _ = frame
    duplicate = pd.concat([X["ret_1d"], X["ret_1d"]], axis=1)
    with pytest.raises(FeatureSchemaError, match="unique"):
        create_model("zero_baseline").fit(duplicate, y_reg)


def test_empty_training_set_rejected() -> None:
    X = pd.DataFrame({"ret_1d": pd.Series(dtype=float)})
    y = pd.Series(dtype=float)
    with pytest.raises(FeatureSchemaError):
        create_model("zero_baseline").fit(X, y)


def test_non_finite_features_rejected(frame: tuple[pd.DataFrame, pd.Series, pd.Series]) -> None:
    X, y_reg, _ = frame
    corrupt = X.copy()
    corrupt.iloc[0, 0] = np.inf
    with pytest.raises(FeatureSchemaError):
        create_model("zero_baseline").fit(corrupt, y_reg)


def test_infinite_predict_features_rejected(
    frame: tuple[pd.DataFrame, pd.Series, pd.Series],
) -> None:
    X, y_reg, _ = frame
    model = create_model("momentum_baseline").fit(X, y_reg)
    corrupt = X.copy()
    corrupt.iloc[0, corrupt.columns.get_loc("momentum_20")] = np.inf
    with pytest.raises(FeatureSchemaError, match="infinite"):
        model.predict(corrupt)


def test_non_finite_labels_rejected(frame: tuple[pd.DataFrame, pd.Series, pd.Series]) -> None:
    X, y_reg, _ = frame
    for bad in (np.nan, np.inf):
        corrupt = y_reg.copy()
        corrupt.iloc[0] = bad
        with pytest.raises(InvalidLabelError):
            create_model("zero_baseline").fit(X, corrupt)


def test_non_numeric_features_rejected(frame: tuple[pd.DataFrame, pd.Series, pd.Series]) -> None:
    X, y_reg, _ = frame
    corrupt = X.copy()
    corrupt["ret_1d"] = "text"
    with pytest.raises(FeatureSchemaError):
        create_model("zero_baseline").fit(corrupt, y_reg)


def test_invalid_classification_labels_rejected(
    frame: tuple[pd.DataFrame, pd.Series, pd.Series],
) -> None:
    X, y_reg, _ = frame
    with pytest.raises(InvalidLabelError):
        create_model("equal_probability").fit(X, pd.Series(np.full(len(X), 2.0)))


@pytest.mark.parametrize("name", ["lag_baseline", "moving_average_baseline", "momentum_baseline"])
def test_missing_required_feature_rejected(
    name: str, frame: tuple[pd.DataFrame, pd.Series, pd.Series]
) -> None:
    _, y_reg, _ = frame
    empty = pd.DataFrame({"unrelated": np.zeros(len(y_reg))})
    with pytest.raises(FeatureSchemaError):
        create_model(name).fit(empty, y_reg)


def test_missing_feature_at_predict_rejected(
    frame: tuple[pd.DataFrame, pd.Series, pd.Series],
) -> None:
    X, y_reg, _ = frame
    model = create_model("momentum_baseline").fit(X, y_reg)
    without_feature = X.drop(columns=["momentum_20"])
    with pytest.raises(FeatureSchemaError):
        model.predict(without_feature)


def test_feature_agnostic_baseline_ignores_columns(
    frame: tuple[pd.DataFrame, pd.Series, pd.Series],
) -> None:
    X, y_reg, _ = frame
    model = create_model("zero_baseline").fit(X, y_reg)
    # A constant baseline predicts even when the columns differ from training.
    other = pd.DataFrame({"totally": np.zeros(10), "different": np.zeros(10)})
    assert model.predict(other).shape == (10,)


def test_failed_refit_invalidates_model(
    frame: tuple[pd.DataFrame, pd.Series, pd.Series],
) -> None:
    X, y_reg, _ = frame
    model = create_model("momentum_baseline").fit(X, y_reg)
    with pytest.raises(FeatureSchemaError):
        model.fit(X.drop(columns=["momentum_20"]), y_reg)
    with pytest.raises(NotFittedError):
        model.predict(X)


@pytest.mark.parametrize("kind", ["wrong_shape", "non_finite"])
def test_invalid_prediction_output_fails_closed(
    kind: str, frame: tuple[pd.DataFrame, pd.Series, pd.Series]
) -> None:
    X, y_reg, _ = frame

    class InvalidOutputModel(AlphaModel):
        name = "invalid_output"
        feature_agnostic = True

        def fit(self, X: pd.DataFrame, y: pd.Series) -> InvalidOutputModel:
            return self

        def predict(self, X: pd.DataFrame) -> np.ndarray:
            if kind == "wrong_shape":
                return np.zeros((len(X), 1))
            return np.full(len(X), np.nan)

    model = InvalidOutputModel().fit(X, y_reg)
    with pytest.raises(ModelError):
        model.predict(X)


# ---------------------------------------------------------------------------
# Probability and uncertainty interface
# ---------------------------------------------------------------------------


def test_regression_baselines_have_no_probability(
    frame: tuple[pd.DataFrame, pd.Series, pd.Series],
) -> None:
    X = frame[0]
    model = _fit("historical_mean", frame)
    with pytest.raises(ProbabilityNotSupportedError):
        model.predict_proba(X)


def test_equal_probability_predicts_half(frame: tuple[pd.DataFrame, pd.Series, pd.Series]) -> None:
    X = frame[0]
    model = _fit("equal_probability", frame)
    np.testing.assert_allclose(model.predict_proba(X), 0.5)


def test_probability_before_fit_raises(
    frame: tuple[pd.DataFrame, pd.Series, pd.Series],
) -> None:
    with pytest.raises(NotFittedError):
        create_model("equal_probability").predict_proba(frame[0])


def test_historical_mean_reports_uncertainty(
    frame: tuple[pd.DataFrame, pd.Series, pd.Series],
) -> None:
    X, y_reg, _ = frame
    model = create_model("historical_mean").fit(X, y_reg)
    unc = model.predict_uncertainty(X)
    assert unc is not None
    np.testing.assert_allclose(unc, float(y_reg.std(ddof=0)))


def test_default_uncertainty_is_none(frame: tuple[pd.DataFrame, pd.Series, pd.Series]) -> None:
    X = frame[0]
    assert _fit("zero_baseline", frame).predict_uncertainty(X) is None
