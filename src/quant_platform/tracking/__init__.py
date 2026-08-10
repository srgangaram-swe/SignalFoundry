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

from quant_platform.tracking.experiment import (
    ExperimentTracker,
    LegacyTrackingError,
    LegacyTrackingReadError,
    LegacyTrackingWriteError,
    RunContext,
    get_tracker,
)

__all__ = [
    "ExperimentTracker",
    "LegacyTrackingError",
    "LegacyTrackingReadError",
    "LegacyTrackingWriteError",
    "RunContext",
    "get_tracker",
]
