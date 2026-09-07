"""Config-driven model registry.

configs/models.yaml lists models by registry name + params; the walk-forward
driver instantiates them here. Optional dependencies (lightgbm, torch) fail
with actionable messages only when actually requested.
"""

from __future__ import annotations

from collections.abc import Callable
from copy import deepcopy
from typing import Any

from alphaforge.models.base import AlphaModel
from alphaforge.models.baselines import (
    BuyAndHoldBaseline,
    EqualProbabilityClassifier,
    EqualWeightBaseline,
    HistoricalMeanBaseline,
    LagBaseline,
    MomentumBaseline,
    MovingAverageBaseline,
    ZeroBaseline,
)
from alphaforge.models.ensemble import EnsembleModel
from alphaforge.models.sklearn_models import (
    make_catboost,
    make_elastic_net,
    make_extra_trees,
    make_gradient_boosting,
    make_huber,
    make_lasso,
    make_lightgbm,
    make_linear,
    make_random_forest,
    make_ridge,
    make_small_mlp,
    make_xgboost,
)


def _make_torch(kind: str) -> Callable[..., AlphaModel]:
    def factory(**params: Any) -> AlphaModel:
        from alphaforge.models import torch_models as tm

        cls = {"mlp": tm.TorchMLP, "gru": tm.TorchGRU, "tcn": tm.TorchTemporalCNN}[kind]
        return cls(**params)

    return factory


def _make_temporal(**params: Any) -> AlphaModel:
    from alphaforge.models.temporal import TemporalAlphaModel

    return TemporalAlphaModel(**params)


def _make_deep_sequence(architecture: str) -> Callable[..., AlphaModel]:
    def factory(**params: Any) -> AlphaModel:
        from alphaforge.models.deep_sequence import ControlledSequenceModel

        return ControlledSequenceModel(architecture, **params)  # type: ignore[arg-type]

    return factory


def _make_ensemble(members: list[dict], **kwargs: Any) -> EnsembleModel:
    built = [create_model(m["name"], **m.get("params", {})) for m in members]
    return EnsembleModel(built, **kwargs)


MODEL_REGISTRY: dict[str, Callable[..., AlphaModel]] = {
    "zero_baseline": ZeroBaseline,
    "historical_mean": HistoricalMeanBaseline,
    "lag_baseline": LagBaseline,
    "moving_average_baseline": MovingAverageBaseline,
    "momentum_baseline": MomentumBaseline,
    "equal_probability": EqualProbabilityClassifier,
    "buy_and_hold": BuyAndHoldBaseline,
    "equal_weight": EqualWeightBaseline,
    "linear": make_linear,
    "ridge": make_ridge,
    "lasso": make_lasso,
    "elastic_net": make_elastic_net,
    "huber": make_huber,
    "random_forest": make_random_forest,
    "extra_trees": make_extra_trees,
    "gradient_boosting": make_gradient_boosting,
    "lightgbm": make_lightgbm,
    "xgboost": make_xgboost,
    "catboost": make_catboost,
    "small_mlp": make_small_mlp,
    "torch_mlp": _make_torch("mlp"),
    "torch_gru": _make_torch("gru"),
    "torch_tcn": _make_torch("tcn"),
    "temporal_alpha": _make_temporal,
    "sequence_cnn": _make_deep_sequence("cnn"),
    "sequence_tcn": _make_deep_sequence("tcn"),
    "sequence_lstm": _make_deep_sequence("lstm"),
    "sequence_gru": _make_deep_sequence("gru"),
    "sequence_transformer": _make_deep_sequence("transformer"),
    "ensemble": _make_ensemble,
}

# Registry-level task metadata lets callers reject an incompatible model before
# instantiation. Keep this explicit: importing optional model ecosystems merely
# to discover their task would make catalog operations environment-dependent.
MODEL_TASKS: dict[str, str] = {
    name: "classification" if name == "equal_probability" else "regression"
    for name in MODEL_REGISTRY
}


def create_model(name: str, **params: Any) -> AlphaModel:
    if name not in MODEL_REGISTRY:
        raise KeyError(f"unknown model {name!r}; available: {sorted(MODEL_REGISTRY)}")
    model = MODEL_REGISTRY[name](**params)
    model.name = name
    return model


def available_models(*, task: str | None = None) -> list[str]:
    """Return deterministic registry names, optionally filtered by model task."""
    if task is not None and task not in {"regression", "classification"}:
        raise ValueError("task must be 'regression', 'classification', or None")
    return sorted(name for name in MODEL_REGISTRY if task is None or MODEL_TASKS[name] == task)


def seed_model_specs(model_specs: list[dict[str, Any]], root_seed: int) -> list[dict[str, Any]]:
    """Inject the declared root seed into every applicable model backend.

    The transformation is independent of candidate order and never overwrites
    an explicitly pre-registered model seed.
    """

    if root_seed < 0:
        raise ValueError("root_seed must be non-negative")
    seeded = deepcopy(model_specs)
    for spec in seeded:
        name = str(spec.get("name", ""))
        params = spec.setdefault("params", {})
        if not isinstance(params, dict):
            raise TypeError(f"model {name!r} params must be a mapping")
        if name in {
            "random_forest",
            "extra_trees",
            "gradient_boosting",
            "lightgbm",
            "xgboost",
            "catboost",
            "small_mlp",
        }:
            params.setdefault("random_state", root_seed)
        elif name in {
            "torch_mlp",
            "torch_gru",
            "torch_tcn",
            "temporal_alpha",
            "sequence_cnn",
            "sequence_tcn",
            "sequence_lstm",
            "sequence_gru",
            "sequence_transformer",
        }:
            params.setdefault("seed", root_seed)
        elif name == "ensemble":
            members = params.get("members")
            if not isinstance(members, list):
                raise TypeError("ensemble members must be a list")
            params["members"] = seed_model_specs(members, root_seed)
    return seeded
