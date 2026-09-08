"""Adversarial and numerical evidence for governed core benchmark models."""

from __future__ import annotations

import builtins
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

from alphaforge.models import AlphaModel, ModelError, NotFittedError, create_model
from alphaforge.models.sklearn_models import SklearnModel
from alphaforge.training import run_walk_forward


def _regression_data(rows: int = 160) -> tuple[pd.DataFrame, pd.Series]:
    generator = np.random.default_rng(20260725)
    values = generator.normal(size=(rows, 4))
    frame = pd.DataFrame(values, columns=["value", "quality", "trend", "risk"])
    frame.loc[::17, "quality"] = np.nan
    target = pd.Series(
        0.7 * values[:, 0] - 0.35 * values[:, 2] + 0.1 * values[:, 3],
        index=frame.index,
        name="forward_return",
    )
    return frame, target


CORE_MODEL_CASES: tuple[tuple[str, dict[str, Any]], ...] = (
    ("linear", {}),
    ("ridge", {"alpha": 0.5, "max_iter": 100}),
    ("lasso", {"alpha": 1e-4, "max_iter": 1_000}),
    ("elastic_net", {"alpha": 1e-4, "max_iter": 1_000}),
    ("huber", {"max_iter": 500}),
    (
        "random_forest",
        {"n_estimators": 12, "max_depth": 4, "min_samples_leaf": 2, "n_jobs": 1},
    ),
    (
        "extra_trees",
        {"n_estimators": 12, "max_depth": 4, "min_samples_leaf": 2, "n_jobs": 1},
    ),
    (
        "gradient_boosting",
        {"max_iter": 12, "max_depth": 3, "min_samples_leaf": 2},
    ),
    (
        "small_mlp",
        {"hidden_layer_sizes": (8,), "batch_size": 32, "max_iter": 50},
    ),
)


@pytest.mark.parametrize(("name", "params"), CORE_MODEL_CASES)
def test_core_model_contract_is_finite_reproducible_and_diagnostic(
    name: str, params: dict[str, Any]
) -> None:
    features, target = _regression_data()
    first = create_model(name, **params)
    second = create_model(name, **params)

    with pytest.raises(NotFittedError):
        first.training_diagnostics()
    first.fit(features, target)
    second.fit(features, target)

    np.testing.assert_allclose(
        first.predict(features),
        second.predict(features),
        rtol=0.0,
        atol=1e-12,
    )
    diagnostics = first.training_diagnostics()
    assert diagnostics is not None
    assert diagnostics.status in {"converged", "completed", "max_iterations", "warning"}
    assert diagnostics.iteration_limit is None or diagnostics.iteration_limit > 0
    assert diagnostics.seed is None or diagnostics.seed >= 0
    assert diagnostics.to_dict()["backend"]


def test_linear_model_agrees_with_independent_least_squares_reference() -> None:
    features, target = _regression_data()
    complete = features.fillna(features.median())
    design = np.column_stack([np.ones(len(complete)), complete.to_numpy()])
    reference_coefficients, *_ = np.linalg.lstsq(design, target.to_numpy(), rcond=None)
    reference = design @ reference_coefficients

    model = create_model("linear").fit(features, target)

    np.testing.assert_allclose(model.predict(features), reference, rtol=1e-10, atol=1e-10)


def test_huber_is_less_sensitive_than_least_squares_to_target_outliers() -> None:
    feature = np.linspace(-2.0, 2.0, 120)
    features = pd.DataFrame({"signal": feature})
    clean = 3.0 * feature - 0.25
    contaminated = clean.copy()
    contaminated[::12] += 40.0
    target = pd.Series(contaminated)

    linear = create_model("linear").fit(features, target)
    huber = create_model("huber", max_iter=1_000).fit(features, target)

    linear_error = np.mean(np.square(linear.predict(features) - clean))
    huber_error = np.mean(np.square(huber.predict(features) - clean))
    assert huber_error < linear_error * 0.05


def test_preprocessing_state_is_train_fold_only_and_immutable_at_predict() -> None:
    features, target = _regression_data()
    model = create_model("ridge", alpha=1.0).fit(features, target)
    assert isinstance(model, SklearnModel)
    before = model.pipeline.named_steps["impute"].statistics_.copy()
    adversarial_test = pd.DataFrame(
        {
            "value": [1e12, -1e12],
            "quality": [np.nan, np.nan],
            "trend": [1e12, -1e12],
            "risk": [1e12, -1e12],
        }
    )

    model.predict(adversarial_test)

    np.testing.assert_array_equal(model.pipeline.named_steps["impute"].statistics_, before)


def test_small_mlp_reports_exhausted_iteration_budget() -> None:
    features, target = _regression_data()
    model = create_model(
        "small_mlp",
        hidden_layer_sizes=(8,),
        batch_size=32,
        max_iter=1,
        tol=1e-12,
    ).fit(features, target)

    diagnostics = model.training_diagnostics()
    assert diagnostics is not None
    assert diagnostics.status == "max_iterations"
    assert diagnostics.iterations == 1
    assert diagnostics.warnings


@pytest.mark.parametrize(
    ("name", "params", "message"),
    (
        ("random_forest", {"n_jobs": -1}, "n_jobs"),
        ("extra_trees", {"n_estimators": 2_001}, "n_estimators"),
        ("gradient_boosting", {"learning_rate": float("inf")}, "learning_rate"),
        ("xgboost", {"min_child_weight": -1.0}, "min_child_weight"),
        ("small_mlp", {"hidden_layer_sizes": (512,)}, "hidden layer width"),
        ("small_mlp", {"hidden_layer_sizes": "8"}, "one or two layers"),
        ("linear", {"fit_intercept": 1}, "boolean"),
    ),
)
def test_governed_hyperparameters_fail_closed(
    name: str, params: dict[str, Any], message: str
) -> None:
    with pytest.raises((TypeError, ValueError), match=message):
        create_model(name, **params)


@pytest.mark.parametrize("dependency", ("lightgbm", "xgboost", "catboost"))
def test_missing_optional_model_dependency_has_actionable_error(
    dependency: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    original_import = builtins.__import__

    def guarded_import(name: str, *args: Any, **kwargs: Any) -> Any:
        if name == dependency:
            raise ImportError(f"blocked {dependency} for boundary test")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded_import)
    with pytest.raises(ImportError, match=r"alphaforge\[ml\]"):
        create_model(dependency)


@pytest.mark.parametrize(("name", "params"), CORE_MODEL_CASES)
def test_trusted_model_persistence_preserves_predictions_and_diagnostics(
    tmp_path: Path, name: str, params: dict[str, Any]
) -> None:
    features, target = _regression_data()
    expected = create_model(name, **params).fit(features, target)
    destination = tmp_path / f"{name}.joblib"
    expected.save(destination)

    with pytest.raises(ModelError, match="executable binary"):
        AlphaModel.load(destination)
    restored = AlphaModel.load(destination, trusted=True)

    np.testing.assert_allclose(restored.predict(features), expected.predict(features))
    assert restored.training_diagnostics() == expected.training_diagnostics()


def test_walk_forward_publishes_termination_evidence(small_features, small_labels) -> None:
    result = run_walk_forward(
        small_features,
        small_labels,
        model_specs=[
            {
                "name": "extra_trees",
                "params": {
                    "n_estimators": 8,
                    "max_depth": 3,
                    "min_samples_leaf": 2,
                    "n_jobs": 1,
                },
            }
        ],
        target="fwd_ret_5",
        config={
            "scheme": "expanding",
            "min_train_days": 100,
            "test_days": 30,
            "step_days": 30,
            "embargo_days": 20,
            "max_windows": 1,
        },
        max_horizon=20,
    )

    row = result.metrics.iloc[0]
    assert row["training_backend"] == "sklearn.ensemble.ExtraTreesRegressor"
    assert row["training_status"] == "completed"
    assert row["training_iterations"] == 8
    assert row["training_iteration_limit"] == 8
    assert row["training_seed"] == 42
    assert row["training_warning_count"] == 0
