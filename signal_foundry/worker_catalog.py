"""Discover the actual AlphaForge registries and preflight scientific parameters."""

from __future__ import annotations

import importlib.util
import inspect
from typing import Any

from signal_foundry.boundary import FoundryError
from signal_foundry.contracts import Capability, Catalog, ResearchRequest
from signal_foundry.policy import (
    OPTIONAL_MODULES,
    PARAMETERS,
    STRATEGIES,
    model_parameters,
)


def catalog() -> Catalog:
    from alphaforge.models.registry import MODEL_REGISTRY, available_models
    from alphaforge.service import available_baselines

    models: list[Capability] = []
    for name in available_models(task="regression"):
        optional = OPTIONAL_MODULES.get(name)
        available = optional is None or importlib.util.find_spec(optional) is not None
        reason = "Registered regression model; parameters require preflight."
        if not available:
            reason = (
                "Optional backend is not installed in the locked research environment."
            )
        if name == "ensemble":
            available = False
            reason = (
                "Structured ensemble members require the retained advanced source CLI."
            )
        signature = inspect.signature(MODEL_REGISTRY[name])
        generic = any(
            item.kind == item.VAR_KEYWORD for item in signature.parameters.values()
        )
        parameters = tuple(
            sorted(key for key in PARAMETERS if generic or key in signature.parameters)
        )
        models.append(
            Capability(
                name=name, available=available, reason=reason, parameters=parameters
            )
        )
    return Catalog(
        models=tuple(models),
        strategies=tuple(
            Capability(name=item.name, available=True, reason=item.description)
            for item in STRATEGIES.values()
        ),
        baselines=tuple(available_baselines()),
        datasets=(),
        limitations=(
            (
                "Interactive walk-forward results are development simulations, not a"
                " frozen final holdout."
            ),
            (
                "No live-order or broker route exists; successful research does not"
                " qualify a strategy."
            ),
            (
                "Daily bars cannot establish intraday queue position or executable"
                " market latency."
            ),
        ),
    )


def specifications(request: ResearchRequest) -> list[dict[str, Any]]:
    """Resolve exact registered models and seed policy before any training starts."""
    from alphaforge.models.registry import create_model, seed_model_specs

    choices = catalog()
    available = {item.name for item in choices.models if item.available}
    if request.model.name not in available:
        raise FoundryError(
            "model_unavailable",
            "The selected model is unavailable in this environment.",
        )
    if request.strategy not in STRATEGIES:
        raise FoundryError(
            "strategy_unavailable", "The selected strategy is not registered."
        )
    if any(name not in choices.baselines for name in request.baselines):
        raise FoundryError(
            "baseline_unavailable", "Select only registered comparison baselines."
        )
    specs: list[dict[str, Any]] = seed_model_specs(
        [
            {"name": request.model.name, "params": model_parameters(request.model)},
            *({"name": name, "params": {}} for name in request.baselines),
        ],
        request.seed,
    )
    # Optional imports and incompatible parameter/model combinations fail here,
    # before admission into the job queue. Instantiation does not fit a model.
    try:
        for spec in specs:
            create_model(spec["name"], **spec["params"])
    except (ImportError, TypeError, ValueError, KeyError) as exc:
        raise FoundryError(
            "model_parameters", "The source model rejected this parameter combination."
        ) from exc
    return specs
