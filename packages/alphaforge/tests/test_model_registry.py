"""Model registry seed-policy tests."""

from __future__ import annotations

import pytest
from sklearn.ensemble import HistGradientBoostingRegressor

from alphaforge.models.registry import available_models, create_model, seed_model_specs
from alphaforge.models.sklearn_models import SklearnModel


def test_model_seed_injection_is_order_independent_and_non_mutating() -> None:
    original = [
        {"name": "gradient_boosting", "params": {"max_iter": 10}},
        {
            "name": "ensemble",
            "params": {
                "members": [
                    {"name": "random_forest", "params": {}},
                    {"name": "ridge", "params": {"alpha": 1.0}},
                ]
            },
        },
    ]

    first = seed_model_specs(original, 17)
    second = seed_model_specs(list(reversed(original)), 17)

    assert original[0]["params"] == {"max_iter": 10}
    assert first[0]["params"]["random_state"] == 17
    assert first[1]["params"]["members"][0]["params"]["random_state"] == 17
    assert {spec["name"]: spec["params"] for spec in first} == {
        spec["name"]: spec["params"] for spec in second
    }


def test_explicit_model_seed_is_preserved() -> None:
    seeded = seed_model_specs(
        [{"name": "random_forest", "params": {"random_state": 99}}],
        17,
    )
    assert seeded[0]["params"]["random_state"] == 99


def test_model_seed_policy_rejects_invalid_boundaries() -> None:
    with pytest.raises(ValueError, match="non-negative"):
        seed_model_specs([], -1)
    with pytest.raises(TypeError, match="params"):
        seed_model_specs([{"name": "ridge", "params": []}], 1)
    with pytest.raises(TypeError, match="ensemble members"):
        seed_model_specs([{"name": "ensemble", "params": {}}], 1)


def test_gradient_boosting_backend_is_explicit_and_environment_independent() -> None:
    model = create_model("gradient_boosting", max_iter=10)

    assert isinstance(model, SklearnModel)
    assert isinstance(model.pipeline.named_steps["model"], HistGradientBoostingRegressor)
    with pytest.raises(ValueError, match="backend must be"):
        create_model("gradient_boosting", backend="automatic")


def test_model_catalog_filters_by_declared_task_without_optional_imports() -> None:
    regression = available_models(task="regression")
    classification = available_models(task="classification")

    assert "random_forest" in regression
    assert {
        "huber",
        "extra_trees",
        "lightgbm",
        "xgboost",
        "catboost",
        "small_mlp",
    }.issubset(regression)
    assert "equal_probability" not in regression
    assert classification == ["equal_probability"]
    assert set(regression) | set(classification) == set(available_models())
    with pytest.raises(ValueError, match="task must be"):
        available_models(task="ranking")


def test_seed_injection_covers_every_stochastic_governed_backend() -> None:
    names = [
        "random_forest",
        "extra_trees",
        "gradient_boosting",
        "lightgbm",
        "xgboost",
        "catboost",
        "small_mlp",
    ]

    seeded = seed_model_specs([{"name": name, "params": {}} for name in names], 2026)

    assert all(spec["params"]["random_state"] == 2026 for spec in seeded)
