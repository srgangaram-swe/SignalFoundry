"""UI-facing services that compose the AlphaForge research stack.

The backtest service is the seam that lets a thin app (API or dashboard) run a
full, leakage-safe research backtest — optionally on a Signal Foundry data bundle
produced by Signalattice — without touching engine internals.
"""

from __future__ import annotations

from alphaforge.service.backtest_service import (
    DISCLAIMER,
    SERVICE_VERSION,
    BacktestRequest,
    BacktestResourceNotFoundError,
    BacktestResult,
    BacktestServiceError,
    StrategyOutcome,
    available_baselines,
    available_strategy_models,
    discover_bundles,
    run_backtest_service,
)

__all__ = [
    "DISCLAIMER",
    "SERVICE_VERSION",
    "BacktestRequest",
    "BacktestResourceNotFoundError",
    "BacktestResult",
    "BacktestServiceError",
    "StrategyOutcome",
    "available_baselines",
    "available_strategy_models",
    "discover_bundles",
    "run_backtest_service",
]
