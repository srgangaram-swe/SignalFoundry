"""Pre-registered paper-readiness decision tests."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from alphaforge.evaluation import (
    NOT_READY,
    READY_FOR_PAPER,
    ReadinessThresholds,
    assess_paper_readiness,
)


def _equity_curve(returns: np.ndarray) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "date": pd.bdate_range("2024-01-02", periods=len(returns)),
            "return": returns,
            "equity": 1_000_000.0 * np.cumprod(1.0 + returns),
            "gross_exposure": np.ones(len(returns)),
            "net_exposure": np.zeros(len(returns)),
            "turnover": np.full(len(returns), 0.05),
            "transaction_cost": np.full(len(returns), 0.0001),
            "active": np.ones(len(returns), dtype=bool),
        }
    )


def test_strict_threshold_config_rejects_unknown_keys() -> None:
    with pytest.raises(ValueError, match="unknown readiness"):
        ReadinessThresholds.from_mapping({"minimum_holdout_dayz": 252})


def test_all_pre_registered_gates_can_authorize_paper_only() -> None:
    rng = np.random.default_rng(42)
    returns = rng.normal(0.001, 0.004, size=504)
    result = assess_paper_readiness(
        equity_curve=_equity_curve(returns),
        benchmark_returns=pd.Series(rng.normal(0.0001, 0.003, size=504)),
        n_trials=2,
        probability_of_backtest_overfitting=0.10,
        point_in_time_limits={
            "historical_revisions_complete": True,
            "universe_membership_point_in_time": True,
            "corporate_actions_complete": True,
        },
        accounting_reconciled=True,
        stress_scenarios_passed=True,
        thresholds=ReadinessThresholds(
            minimum_deflated_sharpe_probability=0.50,
            maximum_average_turnover=0.10,
        ),
    )

    assert result["decision"] == READY_FOR_PAPER
    assert not result["failed_gates"]
    assert "does not authorize" in result["scope"]


def test_incomplete_point_in_time_evidence_forces_not_ready() -> None:
    returns = np.tile(np.array([0.002, -0.0005]), 252)
    result = assess_paper_readiness(
        equity_curve=_equity_curve(returns),
        benchmark_returns=pd.Series(np.zeros(len(returns))),
        n_trials=2,
        probability_of_backtest_overfitting=0.10,
        point_in_time_limits={
            "historical_revisions_complete": False,
            "universe_membership_point_in_time": False,
            "corporate_actions_complete": False,
        },
        accounting_reconciled=True,
        stress_scenarios_passed=True,
        thresholds=ReadinessThresholds(
            minimum_deflated_sharpe_probability=0.50,
            maximum_average_turnover=0.10,
        ),
    )

    assert result["decision"] == NOT_READY
    assert "point_in_time_evidence" in result["failed_gates"]


def test_reconciliation_or_stress_failure_forces_not_ready() -> None:
    returns = np.tile(np.array([0.002, -0.0005]), 252)
    result = assess_paper_readiness(
        equity_curve=_equity_curve(returns),
        benchmark_returns=pd.Series(np.zeros(len(returns))),
        n_trials=2,
        probability_of_backtest_overfitting=0.10,
        point_in_time_limits={
            "historical_revisions_complete": True,
            "universe_membership_point_in_time": True,
            "corporate_actions_complete": True,
        },
        accounting_reconciled=False,
        stress_scenarios_passed=False,
        thresholds=ReadinessThresholds(
            minimum_deflated_sharpe_probability=0.50,
            maximum_average_turnover=0.10,
        ),
    )

    assert result["decision"] == NOT_READY
    assert {"accounting_reconciliation", "stress_scenarios"}.issubset(result["failed_gates"])


def test_additional_operational_gate_is_fail_closed_and_typed() -> None:
    returns = np.tile(np.array([0.002, -0.0005]), 252)
    arguments = {
        "equity_curve": _equity_curve(returns),
        "benchmark_returns": pd.Series(np.zeros(len(returns))),
        "n_trials": 2,
        "probability_of_backtest_overfitting": 0.10,
        "point_in_time_limits": {
            "historical_revisions_complete": True,
            "universe_membership_point_in_time": True,
            "corporate_actions_complete": True,
        },
        "accounting_reconciled": True,
        "stress_scenarios_passed": True,
        "thresholds": ReadinessThresholds(
            minimum_deflated_sharpe_probability=0.50,
            maximum_average_turnover=0.10,
        ),
    }
    result = assess_paper_readiness(
        **arguments,
        additional_gates={"capacity_liquidity": False},
    )
    assert result["decision"] == NOT_READY
    assert result["failed_gates"] == ["capacity_liquidity"]
    with pytest.raises(ValueError, match="must be boolean"):
        assess_paper_readiness(
            **arguments,
            additional_gates={"capacity_liquidity": 1},  # type: ignore[dict-item]
        )
