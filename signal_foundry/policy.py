"""Reviewed interactive parameter/strategy registry; no executable configuration."""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Any

from signal_foundry.boundary import FoundryError
from signal_foundry.contracts import ModelChoice


@dataclass(frozen=True)
class Strategy:
    name: str
    source_name: str
    description: str


# Registration is reviewed Python policy, never browser-supplied import strings.
STRATEGIES = MappingProxyType(
    {
        item.name: item
        for item in (
            Strategy(
                "long_short", "long_short", "Capped long/short cross-sectional ranks."
            ),
            Strategy(
                "long_only_topk", "long_only_topk", "Capped long-only top-k scores."
            ),
            Strategy(
                "rank_weighted",
                "rank_weighted",
                "Magnitude-independent centered ranks.",
            ),
            Strategy(
                "confidence_weighted",
                "confidence_weighted",
                "Clipped within-date score magnitudes.",
            ),
            Strategy(
                "threshold",
                "threshold",
                "Signed score threshold, using the source implementation.",
            ),
        )
    }
)

PARAMETERS = MappingProxyType(
    {
        "alpha": (0.0, 10_000.0),
        "l1_ratio": (0.0, 1.0),
        "n_estimators": (1.0, 256.0),
        "max_depth": (1.0, 16.0),
        "max_iter": (1.0, 1000.0),
        "learning_rate": (0.00001, 1.0),
        "l2_regularization": (0.0, 1000.0),
        "hidden_size": (4.0, 128.0),
        "lookback": (2.0, 128.0),
        "epochs": (1.0, 50.0),
        "batch_size": (8.0, 512.0),
        "dropout": (0.0, 0.8),
        "min_samples_leaf": (1.0, 128.0),
        "epsilon": (1.01, 5.0),
        "tol": (0.0000001, 0.1),
        "scale": (0.00001, 1.0),
    }
)
INTEGER_PARAMETERS = frozenset(
    {
        "n_estimators",
        "max_depth",
        "max_iter",
        "hidden_size",
        "lookback",
        "epochs",
        "batch_size",
        "min_samples_leaf",
    }
)
OPTIONAL_MODULES = MappingProxyType(
    {
        "lightgbm": "lightgbm",
        "xgboost": "xgboost",
        "catboost": "catboost",
        "torch_mlp": "torch",
        "torch_gru": "torch",
        "torch_tcn": "torch",
        "temporal_alpha": "torch",
        "sequence_cnn": "torch",
        "sequence_tcn": "torch",
        "sequence_lstm": "torch",
        "sequence_gru": "torch",
        "sequence_transformer": "torch",
    }
)


def model_parameters(choice: ModelChoice) -> dict[str, Any]:
    """Validate resource-sensitive scalars; source factories check model semantics."""
    result: dict[str, Any] = {}
    for parameter in choice.parameters:
        limits = PARAMETERS.get(parameter.name)
        value = parameter.value
        if (
            limits is None
            or isinstance(value, bool)
            or not isinstance(value, int | float)
        ):
            raise FoundryError(
                "unsupported_parameter",
                "Only registered numerical hyperparameters are supported.",
            )
        if not limits[0] <= value <= limits[1]:
            raise FoundryError(
                "parameter_limit",
                "A hyperparameter exceeds its interactive resource range.",
            )
        if parameter.name in INTEGER_PARAMETERS and type(value) is not int:
            raise FoundryError(
                "parameter_type", "This hyperparameter requires an exact integer."
            )
        result[parameter.name] = value
    return result
