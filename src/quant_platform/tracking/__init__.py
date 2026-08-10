"""Backward-compatible experiment adapters and durable registry components.

Backends:

- ``sqlite`` (default) — a single self-contained ``experiments.sqlite`` DB.
- ``json``   — one JSON file per run under ``experiments/runs/``.
- ``mlflow`` — optional, if MLflow is installed.
- ``none``   — no-op (useful for tests).

The legacy factory remains intentionally small and backward compatible.  New
control-plane code imports the explicit ``contracts``, ``registry``, ``cas``,
``read_ports``, and ``retention`` modules so authority-bearing storage APIs are
never initialized as a side effect of importing this package.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Final

if TYPE_CHECKING:
    from quant_platform.tracking.experiment import (
        ExperimentTracker,
        LegacyTrackingError,
        LegacyTrackingReadError,
        LegacyTrackingWriteError,
        RunContext,
    )

__all__ = [
    "ExperimentTracker",
    "LegacyTrackingError",
    "LegacyTrackingReadError",
    "LegacyTrackingWriteError",
    "RunContext",
    "get_tracker",
]

_LEGACY_EXPORTS: Final = frozenset(__all__)


def __getattr__(name: str) -> Any:
    """Load legacy experiment adapters only when that public API is requested.

    Durable registry/service imports must not initialize the numerical research
    dependency graph as a package side effect. PEP 562 lazy attributes preserve
    the historic ``quant_platform.tracking`` API for callers that explicitly use
    it while keeping the read-only service dependency closure narrow.
    """

    if name not in _LEGACY_EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    from quant_platform.tracking import experiment

    value = getattr(experiment, name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    """Expose stable lazy exports to introspection without importing them."""

    return sorted(set(globals()).union(_LEGACY_EXPORTS))
