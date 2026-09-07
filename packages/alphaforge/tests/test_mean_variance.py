"""Tests for constrained Markowitz optimization (SF-S4-MR2).

Grouped by acceptance criterion. The two that carry the most weight:

* **Analytic agreement** — on well-conditioned problems where no inequality
  binds, the constrained solver must reproduce the closed-form solution. That is
  the only check that validates the optimizer rather than merely exercising it.
* **Independent feasibility audit** — every returned book is re-verified by code
  that does not share the solver's internals, so a projection bug cannot hide.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import pytest

from alphaforge.optimization.evidence import (
    compare_markowitz_variants,
    run_markowitz_backtest,
    run_no_trade_backtest,
    sensitivity_to_input_error,
)
from alphaforge.optimization.mean_variance import (
    CostModel,
    ExposureConstraint,
    MeanVarianceProblem,
    OptimizerError,
    analytic_maximum_utility,
    analytic_minimum_variance,
    audit_solution,
    solve_mean_variance,
)
from alphaforge.optimization.risk_model import (
    RiskModelError,
    shrinkage_covariance,
    validate_covariance,
)
from alphaforge.portfolio.contracts import PortfolioConstraints
from alphaforge.portfolio.evidence import BacktestPanel, chronological_folds

N = 6
SYMBOLS = [f"A{index}" for index in range(N)]

#: Analytic agreement tolerance, declared in advance. The solver converges to a
#: fixed-point residual of 1e-14; 1e-8 on the weights leaves six orders of margin
#: and is far below any economically meaningful position size.
ANALYTIC_TOLERANCE = 1e-8


@pytest.fixture
def risk_model():
    rng = np.random.default_rng(3)
    factor = rng.normal(0.0, 1.0, (N, N))
    matrix = (factor @ factor.T / N + np.eye(N) * 0.5) * 1e-4
    return validate_covariance(pd.DataFrame(matrix, index=SYMBOLS, columns=SYMBOLS))


@pytest.fixture
def alpha() -> pd.Series:
    rng = np.random.default_rng(17)
    return pd.Series(rng.normal(0.0, 2e-4, N), index=SYMBOLS)


@pytest.fixture
def budget_constraints() -> PortfolioConstraints:
    return PortfolioConstraints(max_position=1.0, max_gross=1.0, max_net=1.0, max_leverage=1.0)


# ---------------------------------------------------------------------------
# AC: analytic references
# ---------------------------------------------------------------------------


def test_minimum_variance_matches_the_closed_form(risk_model, budget_constraints) -> None:
    """Only the budget equality binds, so the analytic solution is reachable."""
    problem = MeanVarianceProblem(risk_model=risk_model, constraints=budget_constraints, budget=1.0)
    result = solve_mean_variance(
        problem, formulation="minimum_variance", max_iterations=20_000, tolerance=1e-14
    )
    reference = analytic_minimum_variance(risk_model)
    assert result.status == "optimal"
    np.testing.assert_allclose(result.weights.to_numpy(), reference, atol=ANALYTIC_TOLERANCE)
    assert result.audit_passed


def test_maximum_utility_matches_the_closed_form(risk_model, alpha) -> None:
    """With every inequality slack the stationary point is exactly reproduced."""
    # Scale risk aversion so the unconstrained stationary point lies strictly
    # inside the shared one-unit gross/leverage ceiling. The earlier test used
    # gross >16 and therefore compared a constrained solve to an unreachable
    # unconstrained reference.
    risk_aversion = 100.0
    reference = analytic_maximum_utility(risk_model, alpha.to_numpy(), risk_aversion)
    span = float(np.abs(reference).sum())
    slack = PortfolioConstraints(max_position=1.0, max_gross=1.0, max_net=1.0, max_leverage=1.0)
    problem = MeanVarianceProblem(
        risk_model=risk_model,
        expected_returns=alpha,
        constraints=slack,
        risk_aversion=risk_aversion,
    )
    result = solve_mean_variance(
        problem, formulation="maximum_utility", max_iterations=20_000, tolerance=1e-14
    )
    assert span < 1.0
    np.testing.assert_allclose(result.weights.to_numpy(), reference, atol=ANALYTIC_TOLERANCE)


def test_minimum_variance_beats_equal_weight_on_variance(risk_model, budget_constraints) -> None:
    """The defining property: nothing budget-feasible has lower variance."""
    result = solve_mean_variance(
        MeanVarianceProblem(risk_model=risk_model, constraints=budget_constraints, budget=1.0),
        formulation="minimum_variance",
        max_iterations=20_000,
        tolerance=1e-14,
    )
    equal = np.full(N, 1.0 / N)
    assert (
        risk_model.portfolio_variance(result.weights.to_numpy())
        <= risk_model.portfolio_variance(equal) + 1e-15
    )


def test_target_return_is_attained(risk_model, alpha, budget_constraints) -> None:
    target = float(alpha.max() * 0.5)
    problem = MeanVarianceProblem(
        risk_model=risk_model,
        expected_returns=alpha,
        constraints=budget_constraints,
        budget=1.0,
        target_return=target,
    )
    result = solve_mean_variance(problem, formulation="target_return", max_iterations=4_000)
    assert result.status == "optimal"
    assert result.expected_return >= target - 1e-9
    assert result.audit_passed


def test_unreachable_target_return_is_reported_not_relaxed(
    risk_model, alpha, budget_constraints
) -> None:
    """No silent relaxation: an impossible target is `infeasible`, not a guess."""
    problem = MeanVarianceProblem(
        risk_model=risk_model,
        expected_returns=alpha,
        constraints=budget_constraints,
        budget=1.0,
        target_return=float(alpha.max() * 100.0),
    )
    result = solve_mean_variance(problem, formulation="target_return", max_iterations=2_000)
    assert result.status == "infeasible"


# ---------------------------------------------------------------------------
# AC: independent feasibility audit
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("formulation", ["minimum_variance", "maximum_utility", "alpha_risk_cost"])
def test_every_solution_passes_the_independent_audit(risk_model, alpha, formulation: str) -> None:
    constraints = PortfolioConstraints(
        max_position=0.30, max_gross=1.0, max_net=1.0, max_leverage=1.0, long_only=True
    )
    result = solve_mean_variance(
        MeanVarianceProblem(
            risk_model=risk_model,
            expected_returns=alpha,
            constraints=constraints,
            budget=1.0,
            risk_aversion=5.0,
        ),
        formulation=formulation,  # type: ignore[arg-type]
        max_iterations=8_000,
    )
    assert result.audit_passed, result.audit_violations
    assert result.audit_violations == ()
    weights = result.weights.to_numpy()
    assert np.max(np.abs(weights)) <= constraints.max_position + 1e-7
    assert float(np.sum(weights)) == pytest.approx(1.0, abs=1e-6)
    assert (weights >= -1e-7).all()


def test_audit_detects_every_breach(risk_model, budget_constraints) -> None:
    """The audit must fail a book that violates a limit, or it proves nothing."""
    constraints = PortfolioConstraints(
        max_position=0.2, max_gross=1.0, max_net=1.0, max_leverage=1.0, long_only=True
    )
    problem = MeanVarianceProblem(risk_model=risk_model, constraints=constraints, budget=1.0)
    passed, violations = audit_solution(np.full(N, 0.9), problem)
    assert not passed
    joined = " ".join(violations)
    for expected in ("max_position", "max_gross", "budget"):
        assert expected in joined, joined

    # max_net on its own, with a budget that is compatible with it.
    neutral = MeanVarianceProblem(
        risk_model=risk_model,
        constraints=PortfolioConstraints(
            max_position=1.0, max_gross=2.0, max_net=0.1, max_leverage=2.0
        ),
    )
    net_passed, net_violations = audit_solution(np.full(N, 0.3), neutral)
    assert not net_passed
    assert any("max_net" in item for item in net_violations)

    long_passed, long_violations = audit_solution(
        np.array([-0.5, 0.3, 0.3, 0.3, 0.3, 0.3]), problem
    )
    assert not long_passed
    assert any("long_only" in item for item in long_violations)

    nan_passed, _ = audit_solution(np.full(N, np.nan), problem)
    assert not nan_passed


def test_exposure_constraints_are_enforced_and_audited(risk_model, alpha) -> None:
    """A sector cap is just a bounded linear loading."""
    sector = np.array([1.0, 1.0, 1.0, 0.0, 0.0, 0.0])
    exposure = ExposureConstraint(name="tech", loadings=sector, lower=0.0, upper=0.35)
    constraints = PortfolioConstraints(
        max_position=0.5, max_gross=1.0, max_net=1.0, max_leverage=1.0, long_only=True
    )
    result = solve_mean_variance(
        MeanVarianceProblem(
            risk_model=risk_model,
            expected_returns=alpha,
            constraints=constraints,
            exposures=(exposure,),
            budget=1.0,
            risk_aversion=1.0,
        ),
        formulation="maximum_utility",
        max_iterations=10_000,
    )
    assert result.audit_passed, result.audit_violations
    assert exposure.value(result.weights.to_numpy()) <= 0.35 + 1e-7


# ---------------------------------------------------------------------------
# AC: fail closed on bad inputs, no silent relaxation
# ---------------------------------------------------------------------------


def test_non_symmetric_covariance_is_refused() -> None:
    matrix = np.eye(3) * 1e-4
    matrix[0, 1] = 5e-5
    with pytest.raises(RiskModelError, match="not symmetric"):
        validate_covariance(matrix, assets=("A", "B", "C"))


def test_non_finite_covariance_is_refused() -> None:
    matrix = np.eye(3) * 1e-4
    matrix[0, 0] = np.nan
    with pytest.raises(RiskModelError, match="non-finite"):
        validate_covariance(matrix, assets=("A", "B", "C"))


def test_dimension_mismatch_is_refused() -> None:
    with pytest.raises(RiskModelError, match="does not match asset labels"):
        validate_covariance(np.eye(3) * 1e-4, assets=("A", "B"))
    with pytest.raises(RiskModelError, match="square"):
        validate_covariance(np.ones((2, 3)), assets=("A", "B"))


def test_materially_non_psd_covariance_is_refused_even_if_repair_is_enabled() -> None:
    """Stabilization cannot turn a different strategy into a valid risk model."""
    matrix = np.array([[1e-4, 2e-4], [2e-4, 1e-4]])  # indefinite
    with pytest.raises(RiskModelError, match="materially indefinite"):
        validate_covariance(matrix, assets=("A", "B"), allow_ridge=False)
    with pytest.raises(RiskModelError, match="materially indefinite"):
        validate_covariance(matrix, assets=("A", "B"), allow_ridge=True)


def test_singular_covariance_is_refused() -> None:
    matrix = np.outer(np.ones(4), np.ones(4)) * 1e-4  # rank 1
    with pytest.raises(RiskModelError, match="singular|positive definite"):
        validate_covariance(matrix, assets=tuple("ABCD"))


def test_infeasible_constraints_fail_closed(risk_model) -> None:
    tight = PortfolioConstraints(max_position=0.05, max_gross=1.0, max_net=1.0)
    with pytest.raises(OptimizerError, match="infeasible"):
        solve_mean_variance(
            MeanVarianceProblem(risk_model=risk_model, constraints=tight, budget=1.0),
            formulation="minimum_variance",
        )


def test_budget_incompatible_with_net_is_refused(risk_model) -> None:
    constraints = PortfolioConstraints(max_position=0.5, max_gross=1.0, max_net=0.1)
    with pytest.raises(OptimizerError, match="budget .* max_net"):
        MeanVarianceProblem(risk_model=risk_model, constraints=constraints, budget=1.0)


def test_formulations_requiring_alpha_refuse_without_it(risk_model, budget_constraints) -> None:
    problem = MeanVarianceProblem(risk_model=risk_model, constraints=budget_constraints)
    for formulation in ("maximum_utility", "alpha_risk_cost", "target_return"):
        with pytest.raises(OptimizerError, match="requires"):
            solve_mean_variance(problem, formulation=formulation)


def test_misaligned_expected_returns_are_refused(risk_model, budget_constraints) -> None:
    with pytest.raises(OptimizerError, match="indexed by the risk-model assets"):
        MeanVarianceProblem(
            risk_model=risk_model,
            expected_returns=pd.Series(0.0, index=list(reversed(SYMBOLS))),
            constraints=budget_constraints,
        )


def test_non_finite_alpha_is_refused(risk_model, budget_constraints) -> None:
    broken = pd.Series(np.nan, index=SYMBOLS)
    with pytest.raises(OptimizerError, match="expected_returns must be finite"):
        MeanVarianceProblem(
            risk_model=risk_model, expected_returns=broken, constraints=budget_constraints
        )


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"max_iterations": 0}, "max_iterations"),
        ({"max_iterations": 10**9}, "max_iterations"),
        ({"tolerance": 0.0}, "tolerance"),
    ],
)
def test_solver_budget_is_bounded(risk_model, budget_constraints, kwargs, message) -> None:
    problem = MeanVarianceProblem(risk_model=risk_model, constraints=budget_constraints, budget=1.0)
    with pytest.raises(OptimizerError, match=message):
        solve_mean_variance(problem, formulation="minimum_variance", **kwargs)


def test_non_convergence_is_reported(risk_model, budget_constraints) -> None:
    """A spent iteration budget is `max_iterations`, never `optimal`."""
    result = solve_mean_variance(
        MeanVarianceProblem(risk_model=risk_model, constraints=budget_constraints, budget=1.0),
        formulation="minimum_variance",
        max_iterations=2,
        tolerance=1e-18,
    )
    assert result.status == "max_iterations"
    assert not result.converged
    assert result.iterations == 2


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"name": " "}, "requires a name"),
        ({"lower": 1.0, "upper": 0.0}, "below its lower bound"),
        ({"loadings": np.array([np.nan, 0.0])}, "must be finite"),
    ],
)
def test_exposure_validation(kwargs, message) -> None:
    fields: dict[str, Any] = {"name": "x", "loadings": np.ones(2), "lower": 0.0, "upper": 1.0}
    fields.update(kwargs)
    with pytest.raises(OptimizerError, match=message):
        ExposureConstraint(**fields)


@pytest.mark.parametrize("field", ["linear_bps", "quadratic_bps"])
def test_cost_model_validation(field: str) -> None:
    with pytest.raises(OptimizerError, match=field):
        CostModel(**{field: -1.0})


# ---------------------------------------------------------------------------
# AC: property / metamorphic
# ---------------------------------------------------------------------------


def test_asset_permutation_permutes_the_solution(risk_model, alpha) -> None:
    """A relabelling must not change what the optimizer holds."""
    constraints = PortfolioConstraints(
        max_position=0.4, max_gross=1.0, max_net=1.0, max_leverage=1.0, long_only=True
    )
    base = solve_mean_variance(
        MeanVarianceProblem(
            risk_model=risk_model, expected_returns=alpha, constraints=constraints, budget=1.0
        ),
        formulation="maximum_utility",
        max_iterations=10_000,
    )
    order = [3, 0, 5, 1, 4, 2]
    permuted_symbols = [SYMBOLS[index] for index in order]
    permuted_model = validate_covariance(
        risk_model.to_frame().loc[permuted_symbols, permuted_symbols]
    )
    permuted = solve_mean_variance(
        MeanVarianceProblem(
            risk_model=permuted_model,
            expected_returns=alpha.reindex(permuted_symbols),
            constraints=constraints,
            budget=1.0,
        ),
        formulation="maximum_utility",
        max_iterations=10_000,
    )
    pd.testing.assert_series_equal(
        base.weights.sort_index(), permuted.weights.sort_index(), atol=1e-7
    )


def test_scaling_the_alpha_scales_the_effective_risk_aversion(risk_model, alpha) -> None:
    """Doubling mu and doubling lambda leaves the optimum unchanged."""
    span = float(np.abs(analytic_maximum_utility(risk_model, alpha.to_numpy(), 1.0)).sum())
    slack = PortfolioConstraints(
        max_position=span, max_gross=span * 4, max_net=span * 4, max_leverage=span * 4
    )
    base = solve_mean_variance(
        MeanVarianceProblem(
            risk_model=risk_model, expected_returns=alpha, constraints=slack, risk_aversion=1.0
        ),
        formulation="maximum_utility",
        max_iterations=20_000,
        tolerance=1e-14,
    )
    scaled = solve_mean_variance(
        MeanVarianceProblem(
            risk_model=risk_model,
            expected_returns=alpha * 2.0,
            constraints=slack,
            risk_aversion=2.0,
        ),
        formulation="maximum_utility",
        max_iterations=20_000,
        tolerance=1e-14,
    )
    np.testing.assert_allclose(
        base.weights.to_numpy(), scaled.weights.to_numpy(), atol=ANALYTIC_TOLERANCE
    )


def test_duplicate_assets_are_refused() -> None:
    matrix = np.eye(2) * 1e-4
    with pytest.raises(RiskModelError, match="unique"):
        validate_covariance(matrix, assets=("A", "A"))


def test_identical_assets_receive_identical_weights(alpha) -> None:
    """Deterministic ties: two indistinguishable assets must be treated alike."""
    matrix = np.full((4, 4), 2e-5) + np.eye(4) * 8e-5
    model = validate_covariance(matrix, assets=tuple("ABCD"))
    constraints = PortfolioConstraints(
        max_position=0.5, max_gross=1.0, max_net=1.0, max_leverage=1.0, long_only=True
    )
    result = solve_mean_variance(
        MeanVarianceProblem(risk_model=model, constraints=constraints, budget=1.0),
        formulation="minimum_variance",
        max_iterations=20_000,
        tolerance=1e-14,
    )
    np.testing.assert_allclose(result.weights.to_numpy(), 0.25, atol=1e-7)


def test_near_zero_variance_asset_is_refused_as_ill_conditioned() -> None:
    matrix = np.diag([1e-14, 1e-4, 1e-4, 1e-4, 1e-4, 1e-4])
    with pytest.raises(RiskModelError, match="ill-conditioned"):
        validate_covariance(matrix, assets=tuple(SYMBOLS))


def test_extreme_correlation_is_solvable(alpha) -> None:
    matrix = np.full((3, 3), 0.999) + np.eye(3) * 0.001
    model = validate_covariance(matrix * 1e-4, assets=tuple("ABC"))
    constraints = PortfolioConstraints(
        max_position=1.0, max_gross=1.0, max_net=1.0, max_leverage=1.0
    )
    result = solve_mean_variance(
        MeanVarianceProblem(risk_model=model, constraints=constraints, budget=1.0),
        formulation="minimum_variance",
        max_iterations=20_000,
    )
    assert result.audit_passed
    assert model.condition_number > 100.0


def test_single_asset_problem_is_solvable() -> None:
    model = validate_covariance(np.array([[1e-4]]), assets=("A",))
    constraints = PortfolioConstraints(
        max_position=1.0, max_gross=1.0, max_net=1.0, max_leverage=1.0
    )
    result = solve_mean_variance(
        MeanVarianceProblem(risk_model=model, constraints=constraints, budget=1.0),
        formulation="minimum_variance",
    )
    assert result.weights.iloc[0] == pytest.approx(1.0, abs=1e-7)


def test_empty_covariance_is_refused() -> None:
    with pytest.raises(RiskModelError, match="at least one asset"):
        validate_covariance(np.zeros((0, 0)), assets=())


def test_turnover_penalty_monotonically_reduces_trading(risk_model, alpha) -> None:
    """The property a subgradient implementation silently gets backwards."""
    constraints = PortfolioConstraints(
        max_position=1.0, max_gross=1.0, max_net=1.0, max_leverage=1.0
    )
    previous = pd.Series(np.full(N, 1.0 / N), index=SYMBOLS)
    turnovers = []
    for bps in (0.0, 5.0, 20.0, 100.0):
        result = solve_mean_variance(
            MeanVarianceProblem(
                risk_model=risk_model,
                expected_returns=alpha,
                constraints=constraints,
                budget=1.0,
                previous_weights=previous,
                cost_model=CostModel(linear_bps=bps),
                risk_aversion=2.0,
            ),
            formulation="alpha_risk_cost",
            max_iterations=12_000,
        )
        turnovers.append(result.turnover)
    assert turnovers == sorted(turnovers, reverse=True), turnovers
    assert turnovers[-1] == pytest.approx(0.0, abs=1e-6)


def test_identity_changes_with_every_input(risk_model, alpha, budget_constraints) -> None:
    base = MeanVarianceProblem(
        risk_model=risk_model,
        expected_returns=alpha,
        constraints=budget_constraints,
        budget=1.0,
    )
    assert len(base.identity) == 64
    same = MeanVarianceProblem(
        risk_model=risk_model,
        expected_returns=alpha,
        constraints=budget_constraints,
        budget=1.0,
    )
    assert base.identity == same.identity
    different = MeanVarianceProblem(
        risk_model=risk_model,
        expected_returns=alpha * 1.000001,
        constraints=budget_constraints,
        budget=1.0,
    )
    assert base.identity != different.identity


def test_solver_is_deterministic(risk_model, alpha, budget_constraints) -> None:
    problem = MeanVarianceProblem(
        risk_model=risk_model,
        expected_returns=alpha,
        constraints=budget_constraints,
        budget=1.0,
    )
    first = solve_mean_variance(problem, formulation="maximum_utility", max_iterations=5_000)
    second = solve_mean_variance(problem, formulation="maximum_utility", max_iterations=5_000)
    np.testing.assert_array_equal(first.weights.to_numpy(), second.weights.to_numpy())
    assert first.to_dict() == second.to_dict()


# ---------------------------------------------------------------------------
# AC: mutation tests — no future information
# ---------------------------------------------------------------------------


@pytest.fixture
def market():
    rng = np.random.default_rng(9)
    n_dates, n_assets = 420, 8
    dates = pd.bdate_range("2019-01-02", periods=n_dates)
    symbols = [f"S{index:02d}" for index in range(n_assets)]
    volatility = pd.DataFrame(
        np.exp(rng.normal(-4.0, 0.25, (n_dates, n_assets))), index=dates, columns=symbols
    )
    returns = pd.DataFrame(
        rng.normal(0.0, 1.0, (n_dates, n_assets)) * volatility.to_numpy(),
        index=dates,
        columns=symbols,
    )
    forward = returns.shift(-1).fillna(0.0)
    scores = pd.DataFrame(rng.normal(0.0, 2e-4, (n_dates, n_assets)), index=dates, columns=symbols)
    adv = pd.DataFrame(3e8, index=dates, columns=symbols)
    panel = BacktestPanel(scores=scores, forward_returns=forward, volatility=volatility, adv=adv)
    return {"panel": panel, "returns": returns, "dates": dates, "symbols": symbols}


def test_future_returns_cannot_change_an_earlier_covariance(market) -> None:
    returns = market["returns"]
    as_of = market["dates"][300]
    baseline = shrinkage_covariance(returns, as_of=as_of, window=252)
    mutated = returns.copy()
    mutated.loc[mutated.index >= as_of] *= 50.0
    perturbed = shrinkage_covariance(mutated, as_of=as_of, window=252)
    np.testing.assert_array_equal(baseline.covariance, perturbed.covariance)


def test_future_observations_cannot_change_an_earlier_allocation(market) -> None:
    """The governing leakage test for the whole optimizer path."""
    panel, returns, dates = market["panel"], market["returns"], market["dates"]
    constraints = PortfolioConstraints(
        max_position=0.3, max_gross=1.0, max_net=1.0, max_leverage=1.0, long_only=True
    )
    evaluation = dates[260:340]
    baseline = run_markowitz_backtest(
        panel,
        returns,
        constraints,
        formulation="maximum_utility",
        capital=5e7,
        budget=1.0,
        dates=evaluation,
    )
    cutoff = evaluation[40]
    mutated_returns = returns.copy()
    mutated_returns.loc[mutated_returns.index >= cutoff] *= 30.0
    mutated_scores = panel.scores.copy()
    mutated_scores.loc[mutated_scores.index >= cutoff] *= -100.0
    mutated_panel = BacktestPanel(
        scores=mutated_scores,
        forward_returns=panel.forward_returns,
        volatility=panel.volatility,
        adv=panel.adv,
    )
    perturbed = run_markowitz_backtest(
        mutated_panel,
        mutated_returns,
        constraints,
        formulation="maximum_utility",
        capital=5e7,
        budget=1.0,
        dates=evaluation,
    )
    earlier = baseline.loc[baseline.index < cutoff]
    later = perturbed.loc[perturbed.index < cutoff]
    np.testing.assert_allclose(earlier["gross"].to_numpy(), later["gross"].to_numpy(), atol=1e-12)
    np.testing.assert_allclose(
        earlier["turnover"].to_numpy(), later["turnover"].to_numpy(), atol=1e-12
    )


def test_future_universe_membership_cannot_change_an_earlier_allocation(market) -> None:
    panel, returns, dates = market["panel"], market["returns"], market["dates"]
    constraints = PortfolioConstraints(
        max_position=0.3, max_gross=1.0, max_net=1.0, max_leverage=1.0, long_only=True
    )
    evaluation = dates[260:320]
    baseline = run_markowitz_backtest(
        panel, returns, constraints, capital=5e7, budget=1.0, dates=evaluation
    )
    cutoff = evaluation[30]
    dropped = panel.scores.copy()
    dropped.loc[dropped.index >= cutoff, market["symbols"][:3]] = np.nan
    mutated = BacktestPanel(
        scores=dropped,
        forward_returns=panel.forward_returns,
        volatility=panel.volatility,
        adv=panel.adv,
    )
    perturbed = run_markowitz_backtest(
        mutated, returns, constraints, capital=5e7, budget=1.0, dates=evaluation
    )
    earlier = baseline.loc[baseline.index < cutoff]["gross"].to_numpy()
    later = perturbed.loc[perturbed.index < cutoff]["gross"].to_numpy()
    np.testing.assert_allclose(earlier, later, atol=1e-12)


# ---------------------------------------------------------------------------
# AC: walk-forward evidence
# ---------------------------------------------------------------------------


def test_comparison_includes_a_no_trade_baseline(market) -> None:
    """The arm that is easiest to omit and hardest to beat net of costs."""
    panel, returns, dates = market["panel"], market["returns"], market["dates"]
    constraints = PortfolioConstraints(
        max_position=0.3, max_gross=1.0, max_net=1.0, max_leverage=1.0, long_only=True
    )
    evaluation = dates[260:340]
    window = BacktestPanel(
        scores=panel.scores.loc[evaluation],
        forward_returns=panel.forward_returns.loc[evaluation],
        volatility=panel.volatility.loc[evaluation],
        adv=panel.adv.loc[evaluation],
    )
    comparison = compare_markowitz_variants(
        window,
        returns,
        constraints,
        folds=chronological_folds(evaluation, n_folds=2),
        capital_levels=(5e7,),
        turnover_budgets=(None, 0.2),
        formulations=("minimum_variance", "maximum_utility"),
        budget=1.0,
    )
    assert "no_trade" in set(comparison["arm"])
    assert set(comparison["turnover_budget"]) == {"None", "0.2"}
    ordered = comparison.sort_values(
        ["arm", "capital", "turnover_budget", "fold", "regime"]
    ).reset_index(drop=True)
    pd.testing.assert_frame_equal(comparison, ordered)


def test_turnover_budget_reduces_realized_turnover(market) -> None:
    panel, returns, dates = market["panel"], market["returns"], market["dates"]
    base = PortfolioConstraints(
        max_position=0.3, max_gross=1.0, max_net=1.0, max_leverage=1.0, long_only=True
    )
    from dataclasses import replace

    evaluation = dates[260:330]
    # The fixture's alpha is pure noise, so a risk-averse optimizer barely trades
    # and no turnover cap could bind. Amplify the score so the unconstrained arm
    # genuinely rebalances, otherwise the test would pass vacuously.
    active = BacktestPanel(
        scores=panel.scores * 200.0,
        forward_returns=panel.forward_returns,
        volatility=panel.volatility,
        adv=panel.adv,
    )
    loose = run_markowitz_backtest(
        active, returns, base, capital=5e7, budget=1.0, dates=evaluation, risk_aversion=1.0
    )
    tight = run_markowitz_backtest(
        active,
        returns,
        replace(base, max_turnover=0.05),
        capital=5e7,
        budget=1.0,
        dates=evaluation,
        risk_aversion=1.0,
    )
    assert float(loose["turnover"].iloc[1:].mean()) > 0.0, "fixture must actually trade"
    assert float(tight["turnover"].mean()) < float(loose["turnover"].mean())


def test_capacity_exhaustion_is_recorded_with_a_reason(market) -> None:
    panel, returns, dates = market["panel"], market["returns"], market["dates"]
    constraints = PortfolioConstraints(
        max_position=0.3, max_gross=1.0, max_net=1.0, max_leverage=1.0, long_only=True
    )
    record = run_markowitz_backtest(
        panel, returns, constraints, capital=1e12, budget=1.0, dates=dates[260:280]
    )
    assert not record.empty
    assert (~record["feasible"]).all()
    assert record["reason"].str.contains("liquidity").all()


def test_no_trade_pays_cost_once(market) -> None:
    panel, dates = market["panel"], market["dates"]
    initial = pd.Series(1.0 / len(market["symbols"]), index=market["symbols"])
    record = run_no_trade_backtest(panel, initial, cost_bps=10.0, dates=dates[260:280])
    assert record["turnover"].iloc[0] == pytest.approx(1.0)
    assert (record["turnover"].iloc[1:] == 0.0).all()
    assert (record["cost"].iloc[1:] == 0.0).all()


def test_sensitivity_sweep_is_reproducible_and_ordered(market) -> None:
    panel, returns, dates = market["panel"], market["returns"], market["dates"]
    constraints = PortfolioConstraints(
        max_position=0.3, max_gross=1.0, max_net=1.0, max_leverage=1.0, long_only=True
    )
    evaluation = dates[260:310]
    window = BacktestPanel(
        scores=panel.scores.loc[evaluation],
        forward_returns=panel.forward_returns.loc[evaluation],
        volatility=panel.volatility.loc[evaluation],
        adv=panel.adv.loc[evaluation],
    )
    first = sensitivity_to_input_error(
        window, returns, constraints, capital=5e7, perturbations=(0.0, 0.25), budget=1.0
    )
    second = sensitivity_to_input_error(
        window, returns, constraints, capital=5e7, perturbations=(0.0, 0.25), budget=1.0
    )
    pd.testing.assert_frame_equal(first, second)
    assert list(first["alpha_error"]) == [0.0, 0.25]


def test_comparison_requires_folds_and_bounded_budgets(market) -> None:
    panel, returns = market["panel"], market["returns"]
    constraints = PortfolioConstraints(max_position=0.3, max_gross=1.0, max_net=1.0)
    with pytest.raises(OptimizerError, match="at least one evaluation fold"):
        compare_markowitz_variants(panel, returns, constraints, folds=(), capital_levels=(5e7,))
    with pytest.raises(OptimizerError, match="turnover_budgets"):
        compare_markowitz_variants(
            panel,
            returns,
            constraints,
            folds=chronological_folds(panel.scores.index, n_folds=2),
            capital_levels=(5e7,),
            turnover_budgets=(),
        )


# ---------------------------------------------------------------------------
# Risk model estimator
# ---------------------------------------------------------------------------


def test_shrinkage_is_between_sample_and_target(market) -> None:
    returns = market["returns"]
    model = shrinkage_covariance(returns, as_of=market["dates"][300], window=252)
    assert model.shrinkage_intensity is not None
    assert 0.0 <= model.shrinkage_intensity <= 1.0
    assert model.min_eigenvalue > 0.0
    assert model.estimator == "constant_correlation_shrinkage_v1"


def test_shrinkage_reduces_the_condition_number(market) -> None:
    """The reason shrinkage exists: the optimizer inverts this matrix."""
    returns = market["returns"].iloc[:60]
    sample = validate_covariance(returns.cov(), assets=tuple(market["symbols"]), estimator="sample")
    shrunk = shrinkage_covariance(returns, window=60, intensity=0.5)
    assert shrunk.condition_number < sample.condition_number


def test_insufficient_history_is_refused(market) -> None:
    with pytest.raises(RiskModelError, match="complete observations"):
        shrinkage_covariance(market["returns"], as_of=market["dates"][3], window=252)
    with pytest.raises(RiskModelError, match="window must"):
        shrinkage_covariance(market["returns"], window=2)


def test_invalid_shrinkage_intensity_is_refused(market) -> None:
    with pytest.raises(RiskModelError, match="intensity"):
        shrinkage_covariance(market["returns"], window=252, intensity=1.5)


def test_risk_model_rejects_misaligned_weights(risk_model) -> None:
    with pytest.raises(RiskModelError, match="align"):
        risk_model.portfolio_variance(np.ones(N + 1))
