"""Adversarial and differential tests for the certified sparse Markowitz QP.

These tests target failure modes that feasibility-only tests cannot expose:
suboptimal fixed points, incorrect objective bookkeeping, incomplete turnover
universes, mutable audit records, and solver statuses accepted without an
independent optimality certificate.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pandas as pd
import pytest
from scipy.optimize import minimize

import alphaforge.optimization.mean_variance as mean_variance
from alphaforge.optimization.mean_variance import (
    CostModel,
    ExposureConstraint,
    MeanVarianceProblem,
    OptimizerError,
    audit_solution,
    solve_mean_variance,
)
from alphaforge.optimization.risk_model import validate_covariance
from alphaforge.portfolio.contracts import PortfolioConstraints


def _risk(covariance: np.ndarray, assets: tuple[str, ...], **kwargs):
    """Build a strictly validated covariance with stable labelled order."""
    frame = pd.DataFrame(covariance, index=assets, columns=assets)
    return validate_covariance(frame, allow_ridge=False, **kwargs)


def _unit_constraints(*, long_only: bool = True) -> PortfolioConstraints:
    return PortfolioConstraints(
        max_position=1.0,
        max_gross=1.0,
        max_net=1.0,
        max_leverage=1.0,
        long_only=long_only,
    )


def test_projection_counterexample_reaches_the_exact_boundary_optimum() -> None:
    """Regress the feasible-but-suboptimal cyclic-projection result from MR2 review."""
    assets = ("A", "B", "C")
    optimum_a = 0.548410671798
    optimum_objective = 0.0049874889502
    variance_a = 2.0 * optimum_objective / optimum_a
    variance_c = variance_a * optimum_a / (1.0 - optimum_a)

    # This diagonal B variance reproduces the exact objective of the historical
    # bad candidate while leaving the KKT optimum on B's long-only boundary.
    old_c = 0.45158925
    old_a = (1.0 - old_c) / 2.0
    old_objective = 0.00842046
    alpha_b = -0.01
    variance_b = (
        2.0 * (old_objective + alpha_b * old_a) - variance_a * old_a**2 - variance_c * old_c**2
    ) / old_a**2
    covariance = np.diag([variance_a, variance_b, variance_c])
    alpha = pd.Series([0.0, alpha_b, 0.0], index=assets)
    problem = MeanVarianceProblem(
        risk_model=_risk(covariance, assets),
        expected_returns=alpha,
        constraints=_unit_constraints(),
        budget=1.0,
    )

    result = solve_mean_variance(
        problem,
        formulation="maximum_utility",
        max_iterations=20_000,
        tolerance=1e-12,
    )

    expected = np.array([optimum_a, 0.0, 1.0 - optimum_a])
    rejected_fixed_point = np.array([old_a, old_a, old_c])
    rejected_objective = (
        0.5 * rejected_fixed_point @ covariance @ rejected_fixed_point
        - alpha.to_numpy() @ rejected_fixed_point
    )
    assert result.status == "optimal", result.audit_violations
    np.testing.assert_allclose(result.weights.to_numpy(), expected, atol=2e-8, rtol=0.0)
    assert result.objective == pytest.approx(optimum_objective, abs=2e-10)
    assert rejected_objective == pytest.approx(old_objective, abs=1e-14)
    assert result.objective < rejected_objective - 0.003
    assert result.kkt_passed and result.audit_passed
    assert result.duality_gap <= 1e-7


@pytest.mark.parametrize("seed", range(5))
def test_osqp_agrees_with_independent_slsqp_on_random_constrained_problems(seed: int) -> None:
    """Differentially check the sparse QP against an independent dense optimizer."""
    rng = np.random.default_rng(seed)
    n_assets = 4
    assets = tuple(f"A{index}" for index in range(n_assets))
    factor = rng.normal(size=(n_assets, n_assets))
    covariance = (factor @ factor.T / n_assets + np.eye(n_assets) * 0.5) * 1e-3
    alpha_values = rng.normal(0.0, 1e-4, n_assets)
    risk_aversion = 2.5
    loading = np.array([1.0, 1.0, 0.0, 0.0])
    exposure = ExposureConstraint("first_pair", loading, 0.0, 0.65)
    constraints = PortfolioConstraints(
        max_position=0.70,
        max_gross=1.0,
        max_net=1.0,
        max_leverage=1.0,
        long_only=True,
    )
    risk_model = _risk(covariance, assets)
    problem = MeanVarianceProblem(
        risk_model=risk_model,
        expected_returns=pd.Series(alpha_values, index=assets),
        constraints=constraints,
        exposures=(exposure,),
        risk_aversion=risk_aversion,
        budget=1.0,
    )

    result = solve_mean_variance(
        problem,
        formulation="maximum_utility",
        max_iterations=20_000,
        tolerance=1e-11,
    )

    def objective(weights: np.ndarray) -> float:
        return float(0.5 * risk_aversion * weights @ covariance @ weights - alpha_values @ weights)

    def gradient(weights: np.ndarray) -> np.ndarray:
        return risk_aversion * covariance @ weights - alpha_values

    reference = minimize(
        objective,
        np.full(n_assets, 1.0 / n_assets),
        jac=gradient,
        method="SLSQP",
        bounds=[(0.0, 0.70)] * n_assets,
        constraints=(
            {"type": "eq", "fun": lambda weights: float(np.sum(weights) - 1.0)},
            {
                "type": "ineq",
                "fun": lambda weights: float(0.65 - loading @ weights),
            },
        ),
        options={"ftol": 1e-13, "maxiter": 1_000, "disp": False},
    )

    assert reference.success, reference.message
    assert result.status == "optimal", result.audit_violations
    assert result.objective <= objective(reference.x) + 2e-9
    np.testing.assert_allclose(result.weights.to_numpy(), reference.x, atol=2e-5, rtol=0.0)
    assert result.primal_residual <= 1e-7
    assert result.dual_residual <= 1e-7
    assert result.complementarity_residual <= 1e-7
    assert result.duality_gap <= 1e-7


def test_changing_universe_liquidations_enter_turnover_cost_and_feasibility() -> None:
    """An asset removed from the risk universe remains an explicit closing trade."""
    assets = ("A", "B")
    risk_model = _risk(np.eye(2) * 1e-4, assets)
    previous = pd.Series({"A": 1.0 / 3.0, "B": 1.0 / 3.0, "C": 1.0 / 3.0})
    costs = CostModel(linear_bps=100.0, quadratic_bps=100.0)
    problem = MeanVarianceProblem(
        risk_model=risk_model,
        expected_returns=pd.Series(0.0, index=assets),
        constraints=_unit_constraints(),
        previous_weights=previous,
        cost_model=costs,
        budget=1.0,
    )

    result = solve_mean_variance(
        problem,
        formulation="alpha_risk_cost",
        max_iterations=20_000,
        tolerance=1e-11,
    )

    np.testing.assert_allclose(result.weights.to_numpy(), [0.5, 0.5], atol=2e-8)
    assert result.exit_turnover == pytest.approx(1.0 / 3.0)
    assert result.turnover == pytest.approx(2.0 / 3.0, abs=1e-8)
    expected_cost = 0.01 * (2.0 / 3.0) + 0.01 * (1.0 / 6.0)
    assert result.cost_term == pytest.approx(expected_cost, abs=1e-10)
    assert problem.cost(result.weights.to_numpy()) == pytest.approx(expected_cost, abs=1e-10)

    impossible = MeanVarianceProblem(
        risk_model=risk_model,
        constraints=replace(_unit_constraints(), max_turnover=0.30),
        previous_weights=previous,
        budget=1.0,
    )
    with pytest.raises(OptimizerError, match="mandatory universe exits"):
        solve_mean_variance(impossible, formulation="minimum_variance")


@pytest.mark.parametrize(
    "formulation",
    ["minimum_variance", "target_return", "maximum_utility", "alpha_risk_cost"],
)
def test_each_formulation_reports_its_declared_objective(formulation: str) -> None:
    """Reported terms are reconstructed from weights, never trusted from OSQP."""
    assets = ("A", "B", "C")
    covariance = np.array(
        [
            [2.0e-4, 0.2e-4, 0.1e-4],
            [0.2e-4, 3.0e-4, 0.1e-4],
            [0.1e-4, 0.1e-4, 4.0e-4],
        ]
    )
    alpha = pd.Series([3e-4, 2e-4, 1e-4], index=assets)
    problem = MeanVarianceProblem(
        risk_model=_risk(covariance, assets),
        expected_returns=alpha,
        constraints=_unit_constraints(),
        previous_weights=pd.Series([0.2, 0.5, 0.3], index=assets),
        cost_model=CostModel(linear_bps=12.0, quadratic_bps=8.0),
        risk_aversion=3.0,
        target_return=1.5e-4,
        budget=1.0,
    )
    result = solve_mean_variance(
        problem,
        formulation=formulation,  # type: ignore[arg-type]
        max_iterations=20_000,
        tolerance=1e-11,
    )
    weights = result.weights.to_numpy()
    variance = float(weights @ covariance @ weights)
    achieved_return = float(alpha.to_numpy() @ weights)

    assert result.status == "optimal", result.audit_violations
    if formulation in {"minimum_variance", "target_return"}:
        expected_variance_term = 0.5 * variance
        expected_return_term = 0.0
        expected_cost_term = 0.0
    else:
        expected_variance_term = 0.5 * problem.risk_aversion * variance
        expected_return_term = achieved_return
        expected_cost_term = 0.0 if formulation == "maximum_utility" else problem.cost(weights)
    assert result.variance_term == pytest.approx(expected_variance_term, abs=1e-12)
    assert result.return_term == pytest.approx(expected_return_term, abs=1e-12)
    assert result.cost_term == pytest.approx(expected_cost_term, abs=1e-12)
    assert result.expected_return == pytest.approx(achieved_return, abs=1e-12)
    assert result.objective == pytest.approx(
        expected_variance_term - expected_return_term + expected_cost_term,
        abs=1e-12,
    )
    assert result.objective_residual <= 1e-8


def test_cash_buffer_scales_the_mr2_gross_and_leverage_ceiling() -> None:
    """MR2 applies deployable capital to its gross and leverage ceilings."""
    assets = ("A", "B", "C")
    constraints = PortfolioConstraints(
        max_position=0.4,
        max_gross=0.8,
        max_net=0.8,
        max_leverage=1.0,
        cash_buffer=0.25,
        long_only=True,
    )
    risk_model = _risk(np.eye(3) * 1e-4, assets)
    result = solve_mean_variance(
        MeanVarianceProblem(risk_model=risk_model, constraints=constraints, budget=0.6),
        formulation="minimum_variance",
        max_iterations=20_000,
    )

    assert result.status == "optimal", result.audit_violations
    assert result.gross == pytest.approx(0.6, abs=1e-8)
    assert "max_gross" in result.active_constraints
    with pytest.raises(OptimizerError, match="deployable gross/leverage"):
        MeanVarianceProblem(risk_model=risk_model, constraints=constraints, budget=0.6001)


def test_problem_and_solve_identities_include_exact_values_settings_and_time() -> None:
    assets = ("A", "B")
    covariance = np.array([[2e-4, 0.2e-4], [0.2e-4, 3e-4]])
    as_of = pd.Timestamp("2025-01-03T16:00:00Z")
    risk_model = _risk(covariance, assets, as_of=as_of)
    alpha = pd.Series([2e-4, 1e-4], index=assets)
    problem = MeanVarianceProblem(
        risk_model=risk_model,
        expected_returns=alpha,
        constraints=_unit_constraints(),
        budget=1.0,
        expected_returns_available_at=as_of,
    )
    tiny_change = alpha.copy()
    tiny_change.iloc[0] = np.nextafter(tiny_change.iloc[0], np.inf)
    changed_problem = replace(problem, expected_returns=tiny_change)

    first = solve_mean_variance(problem, formulation="maximum_utility", max_iterations=5_000)
    changed_budget = solve_mean_variance(
        problem, formulation="maximum_utility", max_iterations=5_001
    )
    changed_formulation = solve_mean_variance(
        problem, formulation="alpha_risk_cost", max_iterations=5_000
    )
    later_risk = _risk(covariance, assets, as_of=as_of + pd.Timedelta(days=1))
    later_problem = replace(
        problem,
        risk_model=later_risk,
        decision_timestamp=as_of + pd.Timedelta(days=1),
        expected_returns_available_at=as_of + pd.Timedelta(days=1),
    )

    assert problem.identity != changed_problem.identity
    assert problem.identity != later_problem.identity
    assert first.solver_identity != changed_budget.solver_identity
    assert first.solver_identity != changed_formulation.solver_identity
    assert first.decision_timestamp == as_of.isoformat()
    assert first.solver_name == "osqp"
    assert first.solver_version

    with pytest.raises(OptimizerError, match="not available"):
        replace(problem, expected_returns_available_at=as_of + pd.Timedelta(nanoseconds=1))
    with pytest.raises(OptimizerError, match="timezone awareness"):
        replace(problem, expected_returns_available_at=pd.Timestamp("2025-01-03 16:00:00"))


def test_inputs_results_and_nested_diagnostics_are_defensively_immutable() -> None:
    assets = ("A", "B")
    covariance = np.eye(2) * 1e-4
    alpha = pd.Series([1e-4, 0.0], index=assets)
    loading = np.array([1.0, 0.0])
    exposure = ExposureConstraint("first", loading, 0.0, 1.0)
    problem = MeanVarianceProblem(
        risk_model=_risk(covariance, assets, dropped_assets=("OLD",)),
        expected_returns=alpha,
        constraints=_unit_constraints(),
        exposures=(exposure,),
        budget=1.0,
    )
    alpha.iloc[0] = 99.0
    loading[0] = 99.0
    assert problem.alpha()[0] == pytest.approx(1e-4)
    assert problem.exposures[0].loadings[0] == pytest.approx(1.0)
    with pytest.raises(ValueError):
        problem.expected_returns.to_numpy(copy=False)[0] = 0.0  # type: ignore[union-attr]

    result = solve_mean_variance(problem, formulation="maximum_utility")
    with pytest.raises(ValueError):
        result.weights.to_numpy(copy=False)[0] = 0.0
    with pytest.raises(TypeError):
        result.risk_diagnostics["condition_number"] = 0.0  # type: ignore[index]
    assert result.risk_diagnostics["dropped_assets"] == ("OLD",)
    with pytest.raises(OptimizerError, match="objective does not reconcile"):
        replace(result, objective=result.objective + 1.0)


def test_nonoptimal_status_is_never_a_usable_portfolio() -> None:
    assets = ("A", "B", "C")
    problem = MeanVarianceProblem(
        risk_model=_risk(np.eye(3) * 1e-4, assets),
        constraints=_unit_constraints(),
        budget=1.0,
    )
    result = solve_mean_variance(
        problem,
        formulation="minimum_variance",
        max_iterations=1,
        tolerance=1e-18,
    )

    assert result.status != "optimal"
    assert not result.converged
    assert not result.audit_passed
    assert result.audit_violations
    passed, violations = audit_solution(result.weights.to_numpy(), problem)
    assert not passed
    assert any("budget" in violation for violation in violations)


def test_problem_dimensions_and_public_numeric_controls_are_bounded(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    assets = ("A", "B")
    risk_model = _risk(np.eye(2) * 1e-4, assets)
    monkeypatch.setattr(mean_variance, "MAX_OPTIMIZATION_ASSETS", 1)
    with pytest.raises(OptimizerError, match="asset resource ceiling"):
        MeanVarianceProblem(risk_model=risk_model)
    monkeypatch.setattr(mean_variance, "MAX_OPTIMIZATION_ASSETS", 512)

    exposures = tuple(
        ExposureConstraint(f"factor-{index:03d}", np.zeros(2), 0.0, 0.0) for index in range(257)
    )
    with pytest.raises(OptimizerError, match="exposure resource ceiling"):
        MeanVarianceProblem(risk_model=risk_model, exposures=exposures)

    problem = MeanVarianceProblem(
        risk_model=risk_model,
        constraints=_unit_constraints(),
        budget=1.0,
    )
    with pytest.raises(OptimizerError, match="tolerance"):
        solve_mean_variance(problem, formulation="minimum_variance", tolerance=1e-4)
    with pytest.raises(OptimizerError, match="cannot be converted"):
        ExposureConstraint("bad", object(), 0.0, 1.0)  # type: ignore[arg-type]


def test_public_dataclass_boundaries_raise_only_structured_domain_errors() -> None:
    """Adversarial scalar/container types never leak NumPy or Python internals."""
    assets = ("A", "B")
    risk_model = _risk(np.eye(2) * 1e-4, assets)

    invalid_calls = (
        lambda: ExposureConstraint(None, np.ones(2), 0.0, 1.0),  # type: ignore[arg-type]
        lambda: ExposureConstraint("factor", np.ones(2), object(), 1.0),  # type: ignore[arg-type]
        lambda: CostModel(linear_bps=np.array([1.0, 2.0])),  # type: ignore[arg-type]
        lambda: MeanVarianceProblem(risk_model=risk_model, risk_aversion=object()),  # type: ignore[arg-type]
        lambda: MeanVarianceProblem(risk_model=risk_model, target_return=[]),  # type: ignore[arg-type]
        lambda: MeanVarianceProblem(
            risk_model=risk_model,
            expected_returns=np.ones(2),
        ),
        lambda: MeanVarianceProblem(
            risk_model=risk_model,
            liquidity_caps=pd.Series(["invalid", "1.0"], index=assets),
        ),
    )
    for invalid_call in invalid_calls:
        with pytest.raises(OptimizerError):
            invalid_call()

    problem = MeanVarianceProblem(
        risk_model=risk_model,
        constraints=_unit_constraints(),
        budget=1.0,
    )
    result = solve_mean_variance(problem, formulation="minimum_variance")
    with pytest.raises(OptimizerError):
        solve_mean_variance(object())  # type: ignore[arg-type]
    with pytest.raises(OptimizerError):
        mean_variance.analytic_minimum_variance(object())  # type: ignore[arg-type]
    with pytest.raises(OptimizerError):
        mean_variance.analytic_maximum_utility(risk_model, object(), 1.0)  # type: ignore[arg-type]
    with pytest.raises(OptimizerError):
        solve_mean_variance(
            problem,
            formulation="minimum_variance",
            tolerance=[],  # type: ignore[arg-type]
        )
    with pytest.raises(OptimizerError):
        replace(result, objective=[])  # type: ignore[arg-type]
    with pytest.raises(OptimizerError):
        replace(result, primal_residual=[])  # type: ignore[arg-type]
    with pytest.raises(OptimizerError):
        replace(result, audit_passed=1)  # type: ignore[arg-type]


def test_sized_public_inputs_refuse_resource_exhaustion_before_conversion() -> None:
    """Cheap size metadata is honored before defensive float64 allocation."""

    class OversizedVector:
        size = mean_variance.MAX_OPTIMIZATION_ASSETS + 1

        def __len__(self) -> int:
            return self.size

        def __array__(self, *args, **kwargs):  # pragma: no cover - must remain unreachable
            raise AssertionError("oversized input was converted before its size refusal")

    with pytest.raises(OptimizerError, match="resource ceiling"):
        ExposureConstraint("oversized", OversizedVector(), 0.0, 1.0)  # type: ignore[arg-type]

    assets = ("A", "B")
    risk_model = _risk(np.eye(2) * 1e-4, assets)
    too_many_previous = pd.Series(
        np.zeros(mean_variance.MAX_PREVIOUS_ASSETS + 1),
        index=[f"A{index}" for index in range(mean_variance.MAX_PREVIOUS_ASSETS + 1)],
    )
    with pytest.raises(OptimizerError, match="entry resource ceiling"):
        MeanVarianceProblem(risk_model=risk_model, previous_weights=too_many_previous)


def test_public_numeric_storage_cannot_reenable_numpy_write_flags() -> None:
    """Bytes-backed arrays stay immutable even through ``setflags(write=True)``."""
    assets = ("A", "B")
    exposure = ExposureConstraint("first", np.array([1.0, 0.0]), 0.0, 1.0)
    problem = MeanVarianceProblem(
        risk_model=_risk(np.eye(2) * 1e-4, assets),
        expected_returns=pd.Series([1e-4, 0.0], index=assets),
        previous_weights=pd.Series([0.4, 0.6], index=assets),
        liquidity_caps=pd.Series([1.0, 1.0], index=assets),
        constraints=_unit_constraints(),
        exposures=(exposure,),
        budget=1.0,
    )
    result = solve_mean_variance(problem, formulation="maximum_utility")

    stored_arrays = (
        exposure.loadings,
        problem.expected_returns.to_numpy(copy=False),  # type: ignore[union-attr]
        problem.previous_weights.to_numpy(copy=False),  # type: ignore[union-attr]
        problem.liquidity_caps.to_numpy(copy=False),  # type: ignore[union-attr]
        result.weights.to_numpy(copy=False),
    )
    for stored in stored_arrays:
        with pytest.raises(ValueError):
            stored.setflags(write=True)

    diagnostic_source = pd.Series([1.0, 2.0], index=["x", "y"])
    with_diagnostics = replace(
        result,
        risk_diagnostics={"series": diagnostic_source, "array": np.array([3.0, 4.0])},
    )
    frozen_series = with_diagnostics.risk_diagnostics["series"]
    with pytest.raises(ValueError):
        frozen_series.to_numpy(copy=False).setflags(write=True)
    diagnostic_source.iloc[0] = 99.0
    assert frozen_series.iloc[0] == pytest.approx(1.0)
    assert with_diagnostics.risk_diagnostics["array"] == (3.0, 4.0)


def test_forged_solver_certificates_cannot_claim_portfolio_eligibility() -> None:
    assets = ("A", "B")
    problem = MeanVarianceProblem(
        risk_model=_risk(np.eye(2) * 1e-4, assets),
        constraints=_unit_constraints(),
        budget=1.0,
    )
    result = solve_mean_variance(problem, formulation="minimum_variance")
    forged_primal = 2.0 * mean_variance.KKT_TOLERANCE
    forged_residual = max(
        forged_primal,
        result.dual_residual,
        result.complementarity_residual,
        result.objective_residual,
        result.duality_gap,
    )

    with pytest.raises(OptimizerError, match="certificate exceeds"):
        replace(result, primal_residual=forged_primal, residual=forged_residual)
    with pytest.raises(OptimizerError, match="non-optimal result"):
        replace(
            result,
            status="failed",
            audit_passed=False,
            kkt_passed=True,
            audit_violations=("forged",),
        )
    with pytest.raises(OptimizerError, match="non-optimal result"):
        replace(
            result,
            status="failed",
            audit_passed=False,
            kkt_passed=False,
            audit_violations=(),
        )


def test_tiny_objective_scale_is_solved_and_certified_relatively() -> None:
    """A 1e-16 covariance keeps its optimizer geometry instead of looking flat."""
    assets = ("A", "B")
    problem = MeanVarianceProblem(
        risk_model=_risk(np.diag([1e-16, 4e-16]), assets),
        constraints=_unit_constraints(),
        budget=1.0,
    )
    result = solve_mean_variance(
        problem,
        formulation="minimum_variance",
        tolerance=1e-12,
        max_iterations=20_000,
    )

    assert result.status == "optimal", result.audit_violations
    np.testing.assert_allclose(result.weights.to_numpy(), [0.8, 0.2], atol=2e-9, rtol=0.0)

    qp = mean_variance._build_qp(problem, "minimum_variance")
    feasible_but_suboptimal = np.array([0.5, 0.5, 0.5, 0.5])
    forged = mean_variance._certificate(
        qp,
        feasible_but_suboptimal,
        np.zeros(qp.lower.size),
        problem,
        "minimum_variance",
    )
    assert forged.portfolio_passed
    assert not forged.kkt_passed
    assert forged.dual_residual > mean_variance.KKT_TOLERANCE
    assert forged.duality_gap > mean_variance.KKT_TOLERANCE


def test_tiny_weight_constraints_fail_closed_at_their_actual_scale() -> None:
    assets = ("A", "B")
    risk_model = _risk(np.diag([1e-4, 4e-4]), assets)
    with pytest.raises(OptimizerError, match="float64 audit floor"):
        MeanVarianceProblem(
            risk_model=risk_model,
            constraints=_unit_constraints(),
            budget=mean_variance.FLOAT_ROUNDOFF_FLOOR / 2.0,
        )

    problem = MeanVarianceProblem(
        risk_model=risk_model,
        constraints=_unit_constraints(),
        budget=1e-12,
    )
    passed, violations = audit_solution(
        np.zeros(2),
        problem,
        formulation="minimum_variance",
    )
    assert not passed
    assert any("budget" in violation for violation in violations)
