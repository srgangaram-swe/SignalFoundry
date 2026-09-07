"""Adversarial evidence-accounting tests for SF-S4-MR2.

These tests isolate the evidence boundary from solver internals. Their governing
invariant is that a failed decision changes no position: the existing book still
earns its observed return, then drifts self-financing into the next decision.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import numpy as np
import pandas as pd
import pytest

import alphaforge.optimization.evidence as optimization_evidence
import alphaforge.portfolio.evidence as portfolio_evidence
from alphaforge.optimization.evidence import (
    MAX_PERTURBATIONS,
    compare_markowitz_variants,
    run_markowitz_backtest,
    run_no_trade_backtest,
    sensitivity_to_input_error,
)
from alphaforge.optimization.mean_variance import OptimizerError
from alphaforge.optimization.risk_model import validate_covariance
from alphaforge.portfolio.allocation import top_k_portfolio
from alphaforge.portfolio.contracts import (
    AllocationResult,
    PortfolioConstraints,
    PortfolioError,
)
from alphaforge.portfolio.evidence import (
    MAX_HAC_LAG,
    BacktestPanel,
    Fold,
    run_allocation_backtest,
    summarize_record,
)


def _panel(*, n_dates: int = 12, n_assets: int = 4) -> BacktestPanel:
    dates = pd.bdate_range("2024-01-02", periods=n_dates)
    symbols = [f"S{index}" for index in range(n_assets)]
    scores = pd.DataFrame(
        np.tile(np.linspace(0.1, 1.0, n_assets), (n_dates, 1)),
        index=dates,
        columns=symbols,
    )
    forward = pd.DataFrame(0.0, index=dates, columns=symbols)
    volatility = pd.DataFrame(0.01, index=dates, columns=symbols)
    adv = pd.DataFrame(1e9, index=dates, columns=symbols)
    return BacktestPanel(
        scores=scores,
        forward_returns=forward,
        volatility=volatility,
        adv=adv,
    )


def _constraints() -> PortfolioConstraints:
    return PortfolioConstraints(
        max_position=1.0,
        max_gross=1.0,
        max_net=1.0,
        max_leverage=1.0,
        long_only=True,
    )


def _zero_record(panel: BacktestPanel) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "feasible": True,
            "carried": False,
            "reason": "",
            "gross_return": 0.0,
            "net_return": 0.0,
            "turnover": 0.0,
            "cost": 0.0,
            "gross": 1.0,
            "net_exposure": 1.0,
            "n_positions": len(panel.scores.columns),
            "concentration": 1.0 / len(panel.scores.columns),
            "effective_n": float(len(panel.scores.columns)),
            "condition_number": 1.0,
            "ridge_applied": 0.0,
            "n_observations": 252.0,
            "observations_considered": 252.0,
            "complete_observation_fraction": 1.0,
            "primal_residual": 1e-9,
            "dual_residual": 2e-9,
            "complementarity_residual": 3e-9,
            "objective_residual": 4e-9,
            "duality_gap": 5e-9,
            "max_certificate_residual": 5e-9,
            "ex_ante_volatility": 0.1,
            "solver_status": "optimal",
        },
        index=panel.scores.index,
    )


def test_no_trade_book_drifts_instead_of_receiving_free_rebalancing() -> None:
    panel = _panel(n_dates=2, n_assets=2)
    panel.forward_returns.iloc[0] = [0.10, 0.0]
    panel.forward_returns.iloc[1] = [0.10, 0.0]
    initial = pd.Series([0.5, 0.5], index=panel.scores.columns)

    record = run_no_trade_backtest(panel, initial, cost_bps=0.0)

    assert record["gross_return"].iloc[0] == pytest.approx(0.05)
    assert record["gross_return"].iloc[1] == pytest.approx(0.1 * (0.55 / 1.05))
    assert record["gross_return"].iloc[1] > record["gross_return"].iloc[0]
    assert list(record["turnover"]) == pytest.approx([1.0, 0.0])


def test_universe_exit_is_charged_as_liquidation_turnover() -> None:
    panel = _panel(n_dates=2, n_assets=2)
    panel.scores.iloc[1, 0] = np.nan

    record = run_allocation_backtest(
        panel,
        lambda scores, volatility, constraints, previous, caps: top_k_portfolio(
            scores,
            constraints,
            k=len(scores),
            weighting="equal",
            previous=previous,
            liquidity_caps=caps,
        ),
        _constraints(),
        capital=1e6,
        cost_bps=10.0,
    )

    # Sell the departed 50% position and buy the remaining name from 50% to 100%.
    assert record["turnover"].iloc[1] == pytest.approx(1.0)
    assert record["cost"].iloc[1] == pytest.approx(0.001)


def test_empty_and_infeasible_dates_carry_the_existing_book() -> None:
    panel = _panel(n_dates=3, n_assets=2)
    panel.forward_returns.iloc[1] = [0.10, 0.0]
    panel.scores.iloc[1] = np.nan
    panel.adv.iloc[2] = 0.0

    record = run_allocation_backtest(
        panel,
        lambda scores, volatility, constraints, previous, caps: top_k_portfolio(
            scores,
            constraints,
            k=len(scores),
            weighting="equal",
            previous=previous,
            liquidity_caps=caps,
        ),
        _constraints(),
        capital=1e6,
        cost_bps=0.0,
    )

    assert len(record) == 3
    assert not bool(record["feasible"].iloc[1])
    assert not bool(record["feasible"].iloc[2])
    assert bool(record["carried"].iloc[1])
    assert record["gross"].iloc[1] == pytest.approx(1.0)
    assert record["gross_return"].iloc[1] == pytest.approx(0.05)
    assert record["turnover"].iloc[1] == 0.0


def test_max_iterations_result_is_never_consumed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    panel = _panel(n_dates=12, n_assets=2)
    assets = tuple(str(column) for column in panel.scores.columns)
    risk_model = validate_covariance(np.eye(2) * 1e-4, assets=assets)
    monkeypatch.setattr(
        optimization_evidence,
        "shrinkage_covariance",
        lambda *args, **kwargs: risk_model,
    )
    monkeypatch.setattr(
        optimization_evidence,
        "solve_mean_variance",
        lambda *args, **kwargs: SimpleNamespace(
            status="max_iterations",
            audit_passed=True,
            audit_violations=(),
            risk_diagnostics={"condition_number": 1.0},
        ),
    )

    record = run_markowitz_backtest(
        panel,
        panel.forward_returns,
        _constraints(),
        capital=1e6,
        covariance_window=10,
        dates=panel.scores.index[-1:],
        max_iterations=1,
    )

    assert not bool(record["feasible"].iloc[0])
    assert record["solver_status"].iloc[0] == "max_iterations"
    assert record["turnover"].iloc[0] == 0.0
    assert pd.isna(record["ridge_applied"].iloc[0])
    assert pd.isna(record["max_certificate_residual"].iloc[0])
    assert "not eligible" in record["reason"].iloc[0]


def test_comparison_includes_required_arms_on_identical_dates(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    panel = _panel()
    monkeypatch.setattr(
        optimization_evidence,
        "run_markowitz_backtest",
        lambda panel, *args, **kwargs: _zero_record(panel),
    )

    comparison = compare_markowitz_variants(
        panel,
        panel.forward_returns,
        _constraints(),
        folds=(
            Fold(
                name="development",
                start=panel.scores.index[0],
                end=panel.scores.index[-1],
            ),
        ),
        capital_levels=(1e6,),
        formulations=("maximum_utility",),
        target_return=0.0001,
    )

    assert {
        "maximum_utility",
        "target_return",
        "equal_weight",
        "inverse_volatility",
        "rank_weighted",
        "rank_uncertainty_vol_target",
        "no_trade",
    } <= set(comparison["arm"])
    assert (comparison["n_dates"] == len(panel.scores.index)).all()


def test_successful_markowitz_rows_propagate_risk_and_certificate_diagnostics(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    panel = _panel(n_dates=12, n_assets=2)
    assets = tuple(str(column) for column in panel.scores.columns)
    risk_model = validate_covariance(np.eye(2) * 1e-4, assets=assets)
    monkeypatch.setattr(
        optimization_evidence,
        "shrinkage_covariance",
        lambda *args, **kwargs: risk_model,
    )
    monkeypatch.setattr(
        optimization_evidence,
        "solve_mean_variance",
        lambda *args, **kwargs: SimpleNamespace(
            status="optimal",
            audit_passed=True,
            audit_violations=(),
            weights=pd.Series([0.5, 0.5], index=assets),
            ex_ante_volatility=0.12,
            risk_diagnostics={
                "condition_number": 3.0,
                "ridge_applied": 1e-12,
                "n_observations": 11,
                "observations_considered": 12,
                "complete_observation_fraction": 11.0 / 12.0,
            },
            primal_residual=1e-8,
            dual_residual=2e-8,
            complementarity_residual=3e-8,
            objective_residual=4e-8,
            duality_gap=5e-8,
        ),
    )

    record = run_markowitz_backtest(
        panel,
        panel.forward_returns,
        _constraints(),
        capital=1e6,
        covariance_window=10,
        dates=panel.scores.index[-1:],
    )

    row = record.iloc[0]
    assert bool(row["feasible"])
    assert row["ridge_applied"] == pytest.approx(1e-12)
    assert row["n_observations"] == pytest.approx(11.0)
    assert row["complete_observation_fraction"] == pytest.approx(11.0 / 12.0)
    assert row["max_certificate_residual"] == pytest.approx(5e-8)
    assert row["duality_gap"] == pytest.approx(5e-8)


def test_summary_reports_tail_exposure_conditioning_and_failures() -> None:
    panel = _panel(n_dates=4, n_assets=2)
    record = _zero_record(panel)
    record.loc[record.index[-1], "feasible"] = False
    record.loc[record.index[-1], "carried"] = True
    record.loc[record.index[-1], "solver_status"] = "max_iterations"
    record.loc[record.index[-1], "reason"] = "iteration budget exhausted"
    record["net_return"] = [0.01, -0.02, 0.005, -0.03]
    record["gross_return"] = record["net_return"]

    summary = summarize_record(record)

    required = {
        "realized_volatility",
        "var_95_loss",
        "cvar_95_loss",
        "worst_bar_return",
        "mean_concentration",
        "mean_effective_n",
        "mean_gross_exposure",
        "mean_net_exposure",
        "mean_condition_number",
        "coverage_fraction",
        "solver_failures",
        "failure_reasons",
        "net_return_ci95_low",
        "net_return_ci95_high",
        "uncertainty_method",
        "hac_lag",
        "mean_ridge_applied",
        "mean_complete_observation_fraction",
        "p95_max_certificate_residual",
        "max_duality_gap",
    }
    assert required <= summary.keys()
    assert summary["solver_failures"] == 1
    assert summary["n_failures"] == 1
    assert summary["cvar_95_loss"] >= summary["var_95_loss"]
    assert summary["uncertainty_method"] == "newey_west_hac_bartlett"
    assert 0 < summary["hac_lag"] <= MAX_HAC_LAG
    assert summary["max_certificate_residual"] == pytest.approx(5e-9)
    assert summary["max_duality_gap"] == pytest.approx(5e-9)


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda frame: frame.__setitem__("feasible", "False"), "strict booleans"),
        (
            lambda frame: frame.__setitem__("carried", True),
            "inverse of feasible",
        ),
        (
            lambda frame: frame.__setitem__("net_return", 0.01),
            "reconcile",
        ),
        (lambda frame: frame.__setitem__("cost", -0.01), "non-negative"),
        (lambda frame: frame.__setitem__("net_return", -1.0), "positive equity"),
        (
            lambda frame: frame.__setitem__("condition_number", np.inf),
            "non-negative or unavailable",
        ),
    ],
)
def test_summary_rejects_malformed_or_unreconciled_evidence(mutation: Any, message: str) -> None:
    record = _zero_record(_panel(n_dates=4, n_assets=2))
    mutation(record)
    with pytest.raises(PortfolioError, match=message):
        summarize_record(record)


def test_all_failure_record_cannot_claim_zero_return_performance() -> None:
    record = _zero_record(_panel(n_dates=4, n_assets=2))
    record["feasible"] = False
    record["carried"] = True
    record["reason"] = "no portfolio was established"
    record["solver_status"] = "failed"

    summary = summarize_record(record)

    assert summary["coverage_fraction"] == 0.0
    assert summary["performance_evidence_available"] is False
    assert np.isnan(summary["net_return"])
    assert np.isnan(summary["net_sharpe"])


def test_sensitivity_runs_independent_alpha_and_covariance_perturbations(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    panel = _panel()
    panel.forward_returns.iloc[:, 0] = np.linspace(-0.01, 0.01, len(panel.scores.index))
    seen_returns: list[pd.DataFrame] = []

    def fake_backtest(
        panel: BacktestPanel, returns: pd.DataFrame, *args: Any, **kwargs: Any
    ) -> pd.DataFrame:
        seen_returns.append(returns.copy())
        return _zero_record(panel)

    monkeypatch.setattr(optimization_evidence, "run_markowitz_backtest", fake_backtest)
    evidence = sensitivity_to_input_error(
        panel,
        panel.forward_returns,
        _constraints(),
        capital=1e6,
        perturbations=(0.0, 0.25),
    )

    assert list(evidence["alpha_error"]) == [0.0, 0.25]
    assert list(evidence["covariance_error"]) == [0.0, 0.25]
    assert "covariance_realized_volatility" in evidence
    assert len(seen_returns) == 4
    assert not seen_returns[-1].equals(panel.forward_returns)


def test_covariance_perturbation_scale_cannot_see_future_returns(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    panel = _panel(n_dates=12, n_assets=2)
    returns_size = panel.scores.size
    returns = pd.DataFrame(
        np.linspace(-0.02, 0.03, returns_size).reshape(panel.scores.shape),
        index=panel.scores.index,
        columns=panel.scores.columns,
    )
    assert returns_size == 24
    captured: list[pd.DataFrame] = []

    def fake_backtest(
        panel: BacktestPanel, risk_returns: pd.DataFrame, *args: Any, **kwargs: Any
    ) -> pd.DataFrame:
        captured.append(risk_returns.copy())
        return _zero_record(panel)

    monkeypatch.setattr(optimization_evidence, "run_markowitz_backtest", fake_backtest)
    sensitivity_to_input_error(
        panel,
        returns,
        _constraints(),
        capital=1e6,
        perturbations=(0.25,),
    )
    first_covariance_input = captured[1]

    cutoff = returns.index[7]
    mutated = returns.copy()
    mutated.loc[mutated.index >= cutoff] *= 50.0
    captured.clear()
    sensitivity_to_input_error(
        panel,
        mutated,
        _constraints(),
        capital=1e6,
        perturbations=(0.25,),
    )
    mutated_covariance_input = captured[1]

    pd.testing.assert_frame_equal(
        first_covariance_input.loc[first_covariance_input.index < cutoff],
        mutated_covariance_input.loc[mutated_covariance_input.index < cutoff],
    )
    assert not first_covariance_input.loc[cutoff:].equals(mutated_covariance_input.loc[cutoff:])


def test_panel_detaches_inputs_and_rejects_invalid_numeric_content() -> None:
    dates = pd.bdate_range("2024-01-02", periods=2)
    scores = pd.DataFrame([[1.0], [2.0]], index=dates, columns=["AAA"])
    forward = pd.DataFrame(0.0, index=dates, columns=["AAA"])
    volatility = pd.DataFrame(0.01, index=dates, columns=["AAA"])
    adv = pd.DataFrame(1e6, index=dates, columns=["AAA"])
    panel = BacktestPanel(scores, forward, volatility, adv)

    scores.iloc[0, 0] = 99.0
    assert panel.scores.iloc[0, 0] == 1.0

    invalid = forward.copy()
    invalid.iloc[0, 0] = np.inf
    with pytest.raises(PortfolioError, match="infinite"):
        BacktestPanel(scores, invalid, volatility, adv)


def test_fold_evidence_rejects_overlap_and_empty_intervals(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    panel = _panel()
    monkeypatch.setattr(
        optimization_evidence,
        "run_markowitz_backtest",
        lambda panel, *args, **kwargs: _zero_record(panel),
    )
    dates = panel.scores.index
    overlapping = (
        Fold("first", dates[0], dates[6]),
        Fold("second", dates[6], dates[-1]),
    )
    with pytest.raises(OptimizerError, match="non-overlapping"):
        compare_markowitz_variants(
            panel,
            panel.forward_returns,
            _constraints(),
            folds=overlapping,
            capital_levels=(1e6,),
            formulations=("maximum_utility",),
        )

    outside = (
        Fold(
            "outside",
            dates[-1] + pd.Timedelta(days=1),
            dates[-1] + pd.Timedelta(days=2),
        ),
    )
    with pytest.raises(OptimizerError, match="contains no panel dates"):
        compare_markowitz_variants(
            panel,
            panel.forward_returns,
            _constraints(),
            folds=outside,
            capital_levels=(1e6,),
            formulations=("maximum_utility",),
        )


def test_policy_cannot_allocate_outside_the_decision_universe() -> None:
    panel = _panel(n_dates=2, n_assets=2)

    def rogue_policy(
        scores: pd.Series,
        volatility: pd.Series,
        constraints: PortfolioConstraints,
        previous: pd.Series | None,
        caps: pd.Series,
    ) -> AllocationResult:
        del volatility
        result = top_k_portfolio(
            scores,
            constraints,
            k=len(scores),
            weighting="equal",
            previous=previous,
            liquidity_caps=caps,
        )
        result.weights.loc["OUTSIDE"] = 0.1
        return result

    record = run_allocation_backtest(
        panel,
        rogue_policy,
        _constraints(),
        capital=1e6,
        cost_bps=0.0,
    )

    assert (~record["feasible"]).all()
    assert record["reason"].str.contains("outside the decision universe").all()


def test_panel_cell_ceiling_is_enforced_before_evidence_use(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(portfolio_evidence, "MAX_PANEL_CELLS", 3)
    with pytest.raises(PortfolioError, match="cell"):
        _panel(n_dates=2, n_assets=2)


@pytest.mark.parametrize(
    "perturbations",
    [(), (-0.1,), tuple(float(index) / 100 for index in range(MAX_PERTURBATIONS + 1))],
)
def test_sensitivity_grid_is_bounded(perturbations: tuple[float, ...]) -> None:
    panel = _panel()
    with pytest.raises(OptimizerError, match="perturbations"):
        sensitivity_to_input_error(
            panel,
            panel.forward_returns,
            _constraints(),
            capital=1e6,
            perturbations=perturbations,
        )
