"""Signalattice probabilistic forecasting platform.

An end-to-end quantitative research platform covering data lineage, causal
panel features, calibrated and temporal forecasting, conservative backtesting,
decision-readiness evaluation, experiment tracking, and evidence reporting.

This package is organised into focused sub-packages:

- :mod:`quant_platform.data`       — market-data ingestion & validation
- :mod:`quant_platform.features`   — technical / cross-sectional feature pipeline
- :mod:`quant_platform.models`     — baseline & ML models, time-series CV
- :mod:`quant_platform.backtest`   — vectorized backtesting engine
- :mod:`quant_platform.evaluation` — cost, delay, capacity, latency & gates
- :mod:`quant_platform.risk`       — risk & performance analytics
- :mod:`quant_platform.tracking`   — lightweight experiment tracking
- :mod:`quant_platform.reporting`  — plots and Markdown report generation

DISCLAIMER: Research-use software only. Not financial advice and not an
authorization for live trading.
"""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as _distribution_version

#: The release version, derived from installed package metadata rather than
#: re-typed here. A hardcoded copy drifts from ``pyproject.toml`` the first time
#: someone bumps one and not the other, and a release whose package and module
#: disagree about their own version cannot be verified.
#:
#: The fallback covers a source checkout that was never installed; a release
#: gate asserts the two paths agree, so the fallback cannot quietly go stale.
try:
    __version__ = _distribution_version("signalattice")
except PackageNotFoundError:  # pragma: no cover - exercised only in a bare checkout
    __version__ = "0.3.0"

__all__ = ["__version__"]
