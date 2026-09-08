"""Pre-registered, machine-readable gate for zero-capital paper research.

The gate never authorizes live trading. It converts an immutable final-holdout
result and explicit source limitations into either ``READY_FOR_PAPER`` or
``NOT_READY`` using thresholds fixed before the holdout is evaluated.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import pandas as pd

from alphaforge.evaluation.overfitting import deflated_sharpe_ratio
from alphaforge.risk.metrics import performance_summary
from alphaforge.utils import ANNUALIZATION_DAYS

READY_FOR_PAPER = "READY_FOR_PAPER"
NOT_READY = "NOT_READY"


@dataclass(frozen=True)
class ReadinessThresholds:
    """Immutable thresholds committed before a final-holdout run."""

    rubric_version: str = "1.0.0"
    minimum_holdout_days: int = 252
    minimum_deflated_sharpe_probability: float = 0.95
    maximum_probability_of_backtest_overfitting: float = 0.50
    maximum_drawdown: float = 0.25
    minimum_annual_excess_return: float = 0.0
    maximum_average_turnover: float = 0.50
    require_complete_point_in_time: bool = True

    def __post_init__(self) -> None:
        if self.rubric_version != "1.0.0":
            raise ValueError("unsupported readiness rubric version")
        if self.minimum_holdout_days < 20:
            raise ValueError("minimum_holdout_days must be at least 20")
        probabilities = (
            self.minimum_deflated_sharpe_probability,
            self.maximum_probability_of_backtest_overfitting,
        )
        if not all(0.0 <= value <= 1.0 for value in probabilities):
            raise ValueError("readiness probabilities must be in [0, 1]")
        if not 0.0 < self.maximum_drawdown < 1.0:
            raise ValueError("maximum_drawdown must be in (0, 1)")
        if self.maximum_average_turnover <= 0.0:
            raise ValueError("maximum_average_turnover must be positive")

    @classmethod
    def from_mapping(cls, value: dict[str, Any]) -> ReadinessThresholds:
        """Construct strictly, rejecting misspelled or unknown thresholds."""
        unknown = set(value) - set(cls.__dataclass_fields__)
        if unknown:
            raise ValueError(f"unknown readiness thresholds: {sorted(unknown)}")
        return cls(**value)


def _annual_return(returns: pd.Series) -> float:
    clean = returns.astype(float).replace([np.inf, -np.inf], np.nan).dropna()
    if clean.empty or (clean <= -1.0).any():
        return np.nan
    years = len(clean) / ANNUALIZATION_DAYS
    return float((1.0 + clean).prod() ** (1.0 / years) - 1.0)


def assess_paper_readiness(
    *,
    equity_curve: pd.DataFrame,
    benchmark_returns: pd.Series,
    n_trials: int,
    probability_of_backtest_overfitting: float,
    point_in_time_limits: dict[str, bool],
    accounting_reconciled: bool,
    stress_scenarios_passed: bool,
    thresholds: ReadinessThresholds,
    additional_gates: Mapping[str, bool] | None = None,
) -> dict[str, Any]:
    """Evaluate a final-holdout result against the pre-registered rubric."""
    if n_trials < 1:
        raise ValueError("n_trials must include every attempted candidate")
    summary = performance_summary(equity_curve)
    active_returns = equity_curve["return"].astype(float)
    if "active" in equity_curve:
        active_returns = equity_curve.loc[equity_curve["active"], "return"].astype(float)
    dsr = deflated_sharpe_ratio(active_returns, n_trials=n_trials)
    benchmark_annual_return = _annual_return(benchmark_returns)
    annual_excess = float(summary["annual_return"] - benchmark_annual_return)
    point_in_time_complete = bool(point_in_time_limits) and all(point_in_time_limits.values())

    observed = {
        **summary,
        **dsr,
        "benchmark_annual_return": benchmark_annual_return,
        "annual_excess_return": annual_excess,
        "probability_of_backtest_overfitting": float(probability_of_backtest_overfitting),
        "point_in_time_complete": point_in_time_complete,
        "accounting_reconciled": bool(accounting_reconciled),
        "stress_scenarios_passed": bool(stress_scenarios_passed),
    }
    gates = {
        "holdout_sample": int(summary["n_days"]) >= thresholds.minimum_holdout_days,
        "deflated_sharpe": bool(
            np.isfinite(dsr["deflated_sharpe_prob"])
            and dsr["deflated_sharpe_prob"] >= thresholds.minimum_deflated_sharpe_probability
        ),
        "selection_bias": bool(
            np.isfinite(probability_of_backtest_overfitting)
            and probability_of_backtest_overfitting
            <= thresholds.maximum_probability_of_backtest_overfitting
        ),
        "drawdown": bool(
            np.isfinite(summary["max_drawdown"])
            and summary["max_drawdown"] >= -thresholds.maximum_drawdown
        ),
        "benchmark_excess": bool(
            np.isfinite(annual_excess) and annual_excess >= thresholds.minimum_annual_excess_return
        ),
        "turnover": bool(
            np.isfinite(summary["average_turnover"])
            and summary["average_turnover"] <= thresholds.maximum_average_turnover
        ),
        "point_in_time_evidence": (
            point_in_time_complete if thresholds.require_complete_point_in_time else True
        ),
        "accounting_reconciliation": bool(accounting_reconciled),
        "stress_scenarios": bool(stress_scenarios_passed),
    }
    for name, passed in sorted((additional_gates or {}).items()):
        if not isinstance(name, str) or not name.strip() or name in gates:
            raise ValueError(f"invalid or duplicate additional readiness gate: {name!r}")
        if not isinstance(passed, bool):
            raise ValueError(f"additional readiness gate {name!r} must be boolean")
        gates[name] = passed
    decision = READY_FOR_PAPER if all(gates.values()) else NOT_READY
    return {
        "rubric": asdict(thresholds),
        "decision": decision,
        "gates": gates,
        "failed_gates": sorted(name for name, passed in gates.items() if not passed),
        "metrics": observed,
        "scope": (
            "Zero-capital offline/shadow paper evaluation only. This decision does not "
            "authorize broker connectivity, live orders, capital deployment, or risk expansion."
        ),
    }
