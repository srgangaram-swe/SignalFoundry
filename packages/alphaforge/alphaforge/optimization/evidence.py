"""Walk-forward evidence for the Markowitz optimizer (SF-S4-MR2).

Runs the optimizer chronologically against the SF-S4-MR1 baselines — equal
weight, inverse volatility, ranking portfolios — plus a **no-trade** baseline,
net of costs, across folds, regimes, turnover budgets, and capacity levels.

The no-trade arm matters more than it looks. An optimizer that rebalances hard
can beat a static book gross and lose to it net; without a do-nothing baseline
in the table there is no way to see that, and "our optimizer beat equal weight"
can be true while "our optimizer beat leaving it alone" is false.

**Causality.** At each date the covariance is estimated from returns strictly
before that date and the alpha is the score available at it. A mutation test
asserts that rewriting all later returns, later covariance observations, and
later universe membership cannot change an earlier allocation.

Nothing here selects a winner, and the final holdout is not an argument to the
comparison — see :mod:`alphaforge.portfolio.evidence` for the same discipline.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import replace
from typing import Any

import numpy as np
import pandas as pd

from alphaforge.optimization.mean_variance import (
    CostModel,
    Formulation,
    MeanVarianceProblem,
    OptimizerError,
    solve_mean_variance,
)
from alphaforge.optimization.risk_model import RiskModelError, shrinkage_covariance
from alphaforge.portfolio.allocation import (
    inverse_volatility_portfolio,
    rank_weighted_portfolio,
    top_k_portfolio,
)
from alphaforge.portfolio.contracts import (
    InfeasibleConstraintsError,
    PortfolioConstraints,
    PortfolioError,
    liquidity_caps_from_adv,
)
from alphaforge.portfolio.evidence import (
    MAX_CAPITAL_LEVELS,
    BacktestPanel,
    Fold,
    _carried_failure_row,
    _period_row,
    _returns_for_book,
    _turnover_between,
    _validated_evaluation_dates,
    _validated_folds,
    _validated_regimes,
    run_allocation_backtest,
    summarize_record,
)

#: Refusal thresholds, not tuning knobs.
MAX_TURNOVER_BUDGETS = 8
MAX_FORMULATIONS = 4
MAX_COMPARISON_RUNS = 256
MAX_PERTURBATIONS = 16
MAX_COVARIANCE_WINDOW = 5_000
MAX_RETURN_OBSERVATIONS = 100_000
MAX_RETURN_ASSETS = 5_000
MAX_RETURN_CELLS = 10_000_000
MAX_SENSITIVITY_CELLS = 2_000_000
SUPPORTED_FORMULATIONS: frozenset[str] = frozenset(
    {"minimum_variance", "target_return", "maximum_utility", "alpha_risk_cost"}
)


def _finite_scalar(value: Any, *, name: str) -> float:
    """Return one finite non-boolean numeric parameter."""
    if isinstance(value, (bool, np.bool_)):
        raise OptimizerError(f"{name} must be numeric, not boolean")
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise OptimizerError(f"{name} must be numeric") from exc
    if not np.isfinite(result):
        raise OptimizerError(f"{name} must be finite")
    return result


def _validate_returns_frame(panel: BacktestPanel, returns: pd.DataFrame) -> None:
    """Validate the bounded structural contract required by causal risk estimation."""
    if not isinstance(returns, pd.DataFrame):
        raise OptimizerError("returns must be a pandas DataFrame")
    if returns.index.has_duplicates or returns.columns.has_duplicates:
        raise OptimizerError("returns dates and asset labels must be unique")
    if not returns.index.is_monotonic_increasing:
        raise OptimizerError("returns dates must be sorted ascending")
    if not 1 <= len(returns.index) <= MAX_RETURN_OBSERVATIONS:
        raise OptimizerError(f"returns must hold 1..{MAX_RETURN_OBSERVATIONS} observations")
    if not 1 <= len(returns.columns) <= MAX_RETURN_ASSETS:
        raise OptimizerError(f"returns must hold 1..{MAX_RETURN_ASSETS} assets")
    if returns.size > MAX_RETURN_CELLS:
        raise OptimizerError(f"returns exceed the {MAX_RETURN_CELLS}-cell evidence ceiling")
    if bool(pd.isna(returns.index).any()):
        raise OptimizerError("returns dates must not contain missing labels")
    missing_assets = panel.scores.columns.difference(returns.columns)
    if len(missing_assets):
        raise OptimizerError(
            f"returns are missing {len(missing_assets)} assets required by the panel"
        )


def _validate_folds_and_regimes(
    panel: BacktestPanel, folds: tuple[Fold, ...], regimes: pd.Series | None
) -> pd.Series | None:
    """Bound evidence partitions and reject ambiguous duplicate fold identities."""
    try:
        _validated_folds(panel, folds)
        return _validated_regimes(panel, regimes)
    except PortfolioError as exc:
        raise OptimizerError(str(exc)) from exc


def _optional_finite_diagnostic(diagnostics: Any, name: str) -> float:
    """Return one optional numeric diagnostic, rejecting invalid supplied values."""
    if not isinstance(diagnostics, Mapping):
        raise OptimizerError("risk diagnostics must be a mapping")
    value = diagnostics.get(name)
    if value is None:
        return float("nan")
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise OptimizerError(f"risk diagnostic {name!r} must be numeric when supplied") from exc
    if not np.isfinite(number):
        raise OptimizerError(f"risk diagnostic {name!r} must be finite when supplied")
    return number


def _nonnegative_diagnostic(diagnostics: Any, name: str) -> float:
    """Return one optional finite diagnostic with a non-negative domain."""
    value = _optional_finite_diagnostic(diagnostics, name)
    if np.isfinite(value) and value < 0.0:
        raise OptimizerError(f"risk diagnostic {name!r} must be non-negative")
    return value


def run_markowitz_backtest(
    panel: BacktestPanel,
    returns: pd.DataFrame,
    constraints: PortfolioConstraints,
    *,
    formulation: Formulation = "alpha_risk_cost",
    capital: float,
    cost_bps: float = 10.0,
    participation: float = 0.05,
    risk_aversion: float = 5.0,
    covariance_window: int = 252,
    budget: float | None = None,
    target_return: float | None = None,
    dates: pd.Index | None = None,
    max_iterations: int = 2_000,
) -> pd.DataFrame:
    """Run the optimizer walk-forward and return its per-date net record.

    The record shares its columns with
    :func:`alphaforge.portfolio.evidence.run_allocation_backtest`, so Markowitz
    variants and the MR1 baselines summarize identically and are directly
    comparable.

    A date whose risk model cannot be estimated, whose constraints are
    infeasible, or whose solve fails its audit is recorded with
    ``feasible=False`` and its reason — never dropped, because dropping the hard
    dates is how an optimizer acquires a survivorship-flattered record.
    """
    if not isinstance(panel, BacktestPanel):
        raise OptimizerError("panel must be a BacktestPanel")
    if not isinstance(constraints, PortfolioConstraints):
        raise OptimizerError("constraints must be PortfolioConstraints")
    _validate_returns_frame(panel, returns)
    evaluation_dates = _validated_evaluation_dates(panel, dates)
    if formulation not in SUPPORTED_FORMULATIONS:
        raise OptimizerError(f"unsupported formulation {formulation!r}")
    cost_bps = _finite_scalar(cost_bps, name="cost_bps")
    capital = _finite_scalar(capital, name="capital")
    participation = _finite_scalar(participation, name="participation")
    risk_aversion = _finite_scalar(risk_aversion, name="risk_aversion")
    CostModel(linear_bps=cost_bps)
    if capital <= 0.0:
        raise OptimizerError("capital must be finite and positive")
    if not 0.0 < participation <= 1.0:
        raise OptimizerError("participation must be finite and lie in (0, 1]")
    if risk_aversion <= 0.0:
        raise OptimizerError("risk_aversion must be finite and positive")
    if budget is not None:
        budget = _finite_scalar(budget, name="budget")
        if budget <= 0.0:
            raise OptimizerError("budget must be finite and positive when supplied")
    if target_return is not None:
        target_return = _finite_scalar(target_return, name="target_return")
    if not isinstance(covariance_window, int) or isinstance(covariance_window, bool):
        raise OptimizerError("covariance_window must be an integer")
    if not 10 <= covariance_window <= MAX_COVARIANCE_WINDOW:
        raise OptimizerError(f"covariance_window must lie in [10, {MAX_COVARIANCE_WINDOW}]")
    if formulation == "target_return" and target_return is None:
        raise OptimizerError("target_return formulation requires target_return")
    previous: pd.Series | None = None
    rows: list[dict[str, Any]] = []

    for date in evaluation_dates:
        scores = panel.scores.loc[date].dropna()
        if scores.empty:
            row, previous = _carried_failure_row(
                panel, date, previous, reason="no finite scores on the decision date"
            )
            rows.append(row)
            continue
        if not np.isfinite(scores.to_numpy(dtype=float)).all():
            row, previous = _carried_failure_row(
                panel, date, previous, reason="scores are non-finite on the decision date"
            )
            rows.append(row)
            continue
        solver_status = "failed"
        # A turnover cap limits *rebalancing*. The initial build is not a
        # rebalance — there is no prior book to trade away from — so applying the
        # cap to it makes every subsequent date unreachable and the whole arm
        # reads as infeasible. The first allocation is therefore uncapped, and
        # every later one is capped.
        active_constraints = (
            replace(constraints, max_turnover=None) if previous is None else constraints
        )
        try:
            risk_model = shrinkage_covariance(
                returns.loc[:, scores.index], as_of=date, window=covariance_window
            )
            usable = [asset for asset in risk_model.assets if asset in scores.index]
            alpha = scores.reindex(usable)
            caps = liquidity_caps_from_adv(
                panel.adv.loc[date].reindex(usable).fillna(0.0),
                capital=capital,
                participation=participation,
            )
            problem = MeanVarianceProblem(
                risk_model=risk_model,
                expected_returns=alpha if formulation != "minimum_variance" else None,
                constraints=active_constraints,
                previous_weights=previous,
                cost_model=CostModel(linear_bps=cost_bps),
                risk_aversion=risk_aversion,
                target_return=(target_return if formulation == "target_return" else None),
                liquidity_caps=caps,
                budget=budget,
            )
            result = solve_mean_variance(
                problem, formulation=formulation, max_iterations=max_iterations
            )
            if not isinstance(result.status, str) or not result.status.strip():
                raise OptimizerError("solver status must be a non-empty string")
            solver_status = result.status
            if result.status != "optimal":
                raise OptimizerError(
                    f"solver status {result.status!r} is not eligible for evidence consumption"
                )
            if not isinstance(result.audit_passed, (bool, np.bool_)):
                raise OptimizerError("solver audit result must be boolean")
            if not result.audit_passed:
                raise OptimizerError(f"audit failed: {result.audit_violations}")
            condition_number = _nonnegative_diagnostic(result.risk_diagnostics, "condition_number")
            ridge_applied = _nonnegative_diagnostic(result.risk_diagnostics, "ridge_applied")
            n_observations = _nonnegative_diagnostic(result.risk_diagnostics, "n_observations")
            observations_considered = _nonnegative_diagnostic(
                result.risk_diagnostics, "observations_considered"
            )
            complete_observation_fraction = _nonnegative_diagnostic(
                result.risk_diagnostics, "complete_observation_fraction"
            )
            if np.isfinite(complete_observation_fraction) and complete_observation_fraction > 1.0:
                raise OptimizerError(
                    "risk diagnostic 'complete_observation_fraction' must lie in [0, 1]"
                )
            certificate_values = np.asarray(
                (
                    _finite_scalar(result.primal_residual, name="primal_residual"),
                    _finite_scalar(result.dual_residual, name="dual_residual"),
                    _finite_scalar(
                        result.complementarity_residual,
                        name="complementarity_residual",
                    ),
                    _finite_scalar(result.objective_residual, name="objective_residual"),
                    _finite_scalar(result.duality_gap, name="duality_gap"),
                ),
                dtype=np.float64,
            )
            if (certificate_values < 0.0).any():
                raise OptimizerError(
                    "optimal result must expose finite non-negative certificate residuals"
                )
            max_certificate_residual = float(certificate_values.max())
            if not isinstance(result.weights, pd.Series):
                raise OptimizerError("optimal result weights must be a pandas Series")
            if result.weights.index.has_duplicates:
                raise OptimizerError("optimal result weights must have unique asset labels")
            if not np.isfinite(result.weights.to_numpy(dtype=float)).all():
                raise OptimizerError("optimal result weights must be finite")
            outside = result.weights.index.difference(scores.index)
            if len(outside):
                raise OptimizerError(
                    "optimal result allocated assets outside the decision universe"
                )
            ex_ante_volatility = _finite_scalar(
                result.ex_ante_volatility, name="ex_ante_volatility"
            )
            if ex_ante_volatility < 0.0:
                raise OptimizerError("ex_ante_volatility must be non-negative")
            weights = result.weights.copy()
            turnover = _turnover_between(weights, previous)
            if (
                previous is not None
                and constraints.max_turnover is not None
                and turnover > constraints.max_turnover + 1e-9
            ):
                raise OptimizerError(
                    "target breaches max_turnover after exited-name liquidation accounting "
                    f"({turnover:.6f} > {constraints.max_turnover:.6f})"
                )
            cost = turnover * cost_bps / 10_000.0
            realized = _returns_for_book(panel, date, weights)
            row, drifted = _period_row(
                date=date,
                weights=weights,
                realized=realized,
                feasible=True,
                reason="",
                turnover=turnover,
                cost=cost,
                solver_status=result.status,
                condition_number=condition_number,
                ridge_applied=ridge_applied,
                n_observations=n_observations,
                observations_considered=observations_considered,
                complete_observation_fraction=complete_observation_fraction,
                primal_residual=float(result.primal_residual),
                dual_residual=float(result.dual_residual),
                complementarity_residual=float(result.complementarity_residual),
                objective_residual=float(result.objective_residual),
                duality_gap=float(result.duality_gap),
                max_certificate_residual=max_certificate_residual,
                ex_ante_volatility=ex_ante_volatility,
            )
        except (RiskModelError, OptimizerError, InfeasibleConstraintsError, PortfolioError) as exc:
            row, previous = _carried_failure_row(
                panel,
                date,
                previous,
                reason=str(exc),
                solver_status=solver_status,
            )
            rows.append(row)
            continue
        rows.append(row)
        previous = drifted

    return pd.DataFrame(rows).set_index("date") if rows else pd.DataFrame()


def run_no_trade_backtest(
    panel: BacktestPanel,
    initial_weights: pd.Series,
    *,
    cost_bps: float = 10.0,
    dates: pd.Index | None = None,
    initial_failure_reason: str | None = None,
) -> pd.DataFrame:
    """Run the do-nothing baseline: buy once, then never rebalance.

    The arm that is easiest to omit and hardest to beat net of costs. It pays
    turnover exactly once, on the initial purchase, and nothing afterwards.
    """
    if not isinstance(panel, BacktestPanel):
        raise OptimizerError("panel must be a BacktestPanel")
    cost_bps = _finite_scalar(cost_bps, name="cost_bps")
    if cost_bps < 0.0:
        raise OptimizerError("cost_bps must be finite and non-negative")
    if not isinstance(initial_weights, pd.Series):
        raise OptimizerError("initial_weights must be a pandas Series")
    if initial_weights.index.has_duplicates:
        raise OptimizerError("initial_weights must have unique asset labels")
    values = initial_weights.to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise OptimizerError("initial_weights must be finite")
    gross = float(np.abs(values).sum())
    if not np.isfinite(gross):
        raise OptimizerError("initial_weights gross exposure must be finite")
    missing = initial_weights.index.difference(panel.scores.columns)
    if len(missing):
        raise OptimizerError("initial_weights contain assets outside the panel")
    if initial_failure_reason is not None and (
        not isinstance(initial_failure_reason, str) or not initial_failure_reason.strip()
    ):
        raise OptimizerError("initial_failure_reason must be a non-empty string when supplied")
    evaluation_dates = _validated_evaluation_dates(panel, dates)
    rows: list[dict[str, Any]] = []
    held = initial_weights.copy()
    for position, date in enumerate(evaluation_dates):
        if initial_failure_reason is not None:
            row, _ = _carried_failure_row(
                panel,
                date,
                None,
                reason=initial_failure_reason,
                solver_status="not_applicable",
            )
            rows.append(row)
            continue
        realized = _returns_for_book(panel, date, held)
        turnover = float(held.abs().sum()) if position == 0 else 0.0
        cost = turnover * cost_bps / 10_000.0
        row, held = _period_row(
            date=date,
            weights=held,
            realized=realized,
            feasible=True,
            reason="",
            turnover=turnover,
            cost=cost,
        )
        rows.append(row)
    return pd.DataFrame(rows).set_index("date") if rows else pd.DataFrame()


def compare_markowitz_variants(
    panel: BacktestPanel,
    returns: pd.DataFrame,
    constraints: PortfolioConstraints,
    *,
    folds: tuple[Fold, ...],
    capital_levels: tuple[float, ...],
    turnover_budgets: tuple[float | None, ...] = (None,),
    formulations: tuple[Formulation, ...] = (
        "minimum_variance",
        "maximum_utility",
        "alpha_risk_cost",
    ),
    cost_bps: float = 10.0,
    risk_aversion: float = 5.0,
    budget: float | None = 1.0,
    target_return: float | None = None,
    regimes: pd.Series | None = None,
    uncertainty: pd.DataFrame | None = None,
    uncertainty_strength: float = 0.5,
    volatility_target: float = 0.10,
    participation: float = 0.05,
    include_no_trade: bool = True,
) -> pd.DataFrame:
    """Compare Markowitz formulations across folds, regimes, budgets, and capacity.

    Rows sort by ``(arm, capital, turnover_budget, fold, regime)`` — never by
    performance, because ranking a comparison by its own metric invites reading
    the top row as a decision. The rejection rule belongs to SF-S4-MR9.
    """
    _validate_returns_frame(panel, returns)
    regimes = _validate_folds_and_regimes(panel, folds, regimes)
    if not isinstance(include_no_trade, (bool, np.bool_)):
        raise OptimizerError("include_no_trade must be boolean")
    if not 1 <= len(capital_levels) <= MAX_CAPITAL_LEVELS:
        raise OptimizerError(f"capital_levels must hold 1..{MAX_CAPITAL_LEVELS} entries")
    normalized_capital = tuple(
        _finite_scalar(level, name="capital level") for level in capital_levels
    )
    if any(level <= 0.0 for level in normalized_capital):
        raise OptimizerError("every capital level must be finite and positive")
    if len(set(normalized_capital)) != len(normalized_capital):
        raise OptimizerError("capital_levels must be unique")
    if not 1 <= len(turnover_budgets) <= MAX_TURNOVER_BUDGETS:
        raise OptimizerError(f"turnover_budgets must hold 1..{MAX_TURNOVER_BUDGETS} entries")
    normalized_turnover = tuple(
        None if cap is None else _finite_scalar(cap, name="turnover budget")
        for cap in turnover_budgets
    )
    if len(set(normalized_turnover)) != len(normalized_turnover):
        raise OptimizerError("turnover_budgets must be unique")
    if any(cap is not None and cap < 0.0 for cap in normalized_turnover):
        raise OptimizerError("turnover budgets must be finite and non-negative when set")
    if not 1 <= len(formulations) <= MAX_FORMULATIONS:
        raise OptimizerError(f"formulations must hold 1..{MAX_FORMULATIONS} entries")
    configured = list(formulations)
    if target_return is not None and "target_return" not in configured:
        configured.append("target_return")
    if len(set(configured)) != len(configured):
        raise OptimizerError("formulations must be unique")
    unknown_formulations = set(configured) - SUPPORTED_FORMULATIONS
    if unknown_formulations:
        raise OptimizerError(f"unsupported formulations: {sorted(unknown_formulations)}")
    if "target_return" in configured and target_return is None:
        raise OptimizerError("target_return formulation requires target_return")
    runs_per_grid = len(configured) + 4 + int(include_no_trade)
    total_runs = runs_per_grid * len(capital_levels) * len(turnover_budgets)
    if total_runs > MAX_COMPARISON_RUNS:
        raise OptimizerError(
            f"comparison requires {total_runs} runs, above the {MAX_COMPARISON_RUNS}-run ceiling"
        )
    if uncertainty is not None:
        if not isinstance(uncertainty, pd.DataFrame):
            raise OptimizerError("uncertainty must be a pandas DataFrame when supplied")
        if not uncertainty.index.equals(panel.scores.index):
            raise OptimizerError("uncertainty index must match the score index")
        if list(uncertainty.columns) != list(panel.scores.columns):
            raise OptimizerError("uncertainty columns must match the score columns")
    # A dedicated predictive-uncertainty panel is preferred. Until supplied,
    # trailing asset volatility is used explicitly as a risk-dispersion proxy;
    # it is not represented as calibrated forecast uncertainty.
    composite_uncertainty = panel.volatility if uncertainty is None else uncertainty

    records: dict[tuple[str, float, str], pd.DataFrame] = {}
    # Budget appears in the key as its string form so `None` and a numeric cap
    # share one column type in the emitted comparison frame.
    for formulation in configured:
        for capital in sorted(normalized_capital):
            for cap in normalized_turnover:
                limited = replace(constraints, max_turnover=cap)
                records[(formulation, capital, str(cap))] = run_markowitz_backtest(
                    panel,
                    returns,
                    limited,
                    formulation=formulation,
                    capital=capital,
                    cost_bps=cost_bps,
                    participation=participation,
                    risk_aversion=risk_aversion,
                    budget=budget,
                    target_return=target_return,
                )

    baseline_policies = {
        "equal_weight": (
            lambda scores, volatility, active, previous, caps: top_k_portfolio(
                scores,
                active,
                k=len(scores),
                weighting="equal",
                previous=previous,
                liquidity_caps=caps,
            )
        ),
        "inverse_volatility": (
            lambda scores, volatility, active, previous, caps: inverse_volatility_portfolio(
                scores,
                volatility,
                active,
                previous=previous,
                liquidity_caps=caps,
            )
        ),
        "rank_weighted": (
            lambda scores, volatility, active, previous, caps: rank_weighted_portfolio(
                scores, active, previous=previous, liquidity_caps=caps
            )
        ),
    }
    for arm, policy in baseline_policies.items():
        for capital in sorted(normalized_capital):
            for cap in normalized_turnover:
                limited = replace(constraints, max_turnover=cap)
                records[(arm, capital, str(cap))] = run_allocation_backtest(
                    panel,
                    policy,
                    limited,
                    capital=capital,
                    cost_bps=cost_bps,
                    participation=participation,
                )
    for capital in sorted(normalized_capital):
        for cap in normalized_turnover:
            limited = replace(constraints, max_turnover=cap)
            records[("rank_uncertainty_vol_target", capital, str(cap))] = run_allocation_backtest(
                panel,
                baseline_policies["rank_weighted"],
                limited,
                capital=capital,
                cost_bps=cost_bps,
                participation=participation,
                uncertainty=composite_uncertainty,
                uncertainty_strength=uncertainty_strength,
                volatility_target=volatility_target,
            )
    if include_no_trade:
        first = panel.scores.iloc[0].dropna()
        if not np.isfinite(first.to_numpy(dtype=float)).all():
            raise OptimizerError("initial no-trade scores must be finite when observed")
        for capital in sorted(normalized_capital):
            caps = liquidity_caps_from_adv(
                panel.adv.iloc[0].reindex(first.index).fillna(0.0),
                capital=capital,
                participation=participation,
            )
            try:
                initial = top_k_portfolio(
                    first,
                    replace(constraints, max_turnover=None),
                    k=len(first),
                    weighting="equal",
                    liquidity_caps=caps,
                ).weights
                initial_failure_reason = None
            except PortfolioError as exc:
                initial = pd.Series(dtype=float)
                initial_failure_reason = str(exc)
            for cap in normalized_turnover:
                records[("no_trade", capital, str(cap))] = run_no_trade_backtest(
                    panel,
                    initial,
                    cost_bps=cost_bps,
                    initial_failure_reason=initial_failure_reason,
                )

    rows: list[dict[str, Any]] = []
    expected_dates = panel.scores.index
    for (arm, capital, budget_label), record in records.items():
        if record.empty:
            continue
        if not record.index.equals(expected_dates):
            raise OptimizerError(f"arm {arm!r} did not emit the identical evaluation dates")
        for fold in folds:
            window = record.loc[(record.index >= fold.start) & (record.index <= fold.end)]
            slices: list[tuple[str, pd.DataFrame]] = [("all", window)]
            if regimes is not None:
                aligned = regimes.reindex(window.index)
                string_labels = aligned.astype("string")
                for label in sorted(set(string_labels.dropna()) - {"unknown"}):
                    slices.append((str(label), window.loc[string_labels == label]))
            for regime_label, block in slices:
                if block.empty:
                    continue
                summary = summarize_record(block)
                summary.update(
                    {
                        "arm": arm,
                        "capital": capital,
                        "turnover_budget": budget_label,
                        "fold": fold.name,
                        "regime": regime_label,
                    }
                )
                rows.append(summary)
    frame = pd.DataFrame(rows)
    if frame.empty:
        return frame
    ordered = ["arm", "capital", "turnover_budget", "fold", "regime"]
    return (
        frame.loc[:, ordered + [column for column in frame.columns if column not in ordered]]
        .sort_values(ordered)
        .reset_index(drop=True)
    )


def sensitivity_to_input_error(
    panel: BacktestPanel,
    returns: pd.DataFrame,
    constraints: PortfolioConstraints,
    *,
    capital: float,
    perturbations: tuple[float, ...] = (0.0, 0.1, 0.25, 0.5),
    seed: int = 11,
    formulation: Formulation = "maximum_utility",
    budget: float | None = 1.0,
    risk_aversion: float = 5.0,
) -> pd.DataFrame:
    """Measure independent alpha and covariance-estimation error sensitivity.

    The single most important diagnostic for mean-variance: it is famously more
    sensitive to expected-return error than to covariance error, and an
    optimizer whose result collapses under a 10% alpha perturbation is reporting
    the precision of its inputs, not skill.

    Alpha perturbations multiply scores by seeded Gaussian noise. Covariance
    perturbations add separately seeded noise, scaled by each return series'
    sample volatility, to the risk-estimation input only; realized forward
    returns remain unchanged. Each row carries the ordinary alpha-perturbation
    summary plus a ``covariance_*`` summary for the matched error level.
    """
    _validate_returns_frame(panel, returns)
    if returns.size > MAX_SENSITIVITY_CELLS:
        raise OptimizerError(
            "sensitivity input exceeds the " f"{MAX_SENSITIVITY_CELLS}-cell materialization ceiling"
        )
    if not isinstance(seed, int) or isinstance(seed, bool) or not 0 <= seed <= 2**32 - 1:
        raise OptimizerError("seed must be an integer in [0, 2**32 - 1]")
    if not 1 <= len(perturbations) <= MAX_PERTURBATIONS:
        raise OptimizerError(f"perturbations must hold 1..{MAX_PERTURBATIONS} entries")
    normalized_perturbations = tuple(
        _finite_scalar(level, name="perturbation") for level in perturbations
    )
    if len(set(normalized_perturbations)) != len(normalized_perturbations):
        raise OptimizerError("perturbations must be unique")
    if any(not 0.0 <= level <= 2.0 for level in normalized_perturbations):
        raise OptimizerError("perturbations must be finite and lie in [0, 2]")
    try:
        return_values = returns.to_numpy(dtype=np.float64, copy=False)
    except (TypeError, ValueError) as exc:
        raise OptimizerError("sensitivity returns must contain numeric values") from exc
    if np.isinf(return_values).any():
        raise OptimizerError("sensitivity returns must not contain infinite values")
    # Each perturbation row is scaled only by observations strictly before it.
    # A full-sample standard deviation would let future volatility rewrite the
    # perturbed covariance history used by an earlier decision.
    causal_scales = returns.expanding(min_periods=2).std(ddof=1).shift(1).fillna(0.0)
    rows: list[dict[str, Any]] = []
    for level in normalized_perturbations:
        level_bits = int(np.asarray(level, dtype=np.float64).view(np.uint64))
        seed_words = (level_bits & 0xFFFFFFFF, level_bits >> 32)
        generator = np.random.default_rng([seed, 1, *seed_words])
        noisy = panel.scores.copy()
        if level > 0.0:
            noise = generator.normal(1.0, level, noisy.shape)
            noisy = pd.DataFrame(noisy.to_numpy() * noise, index=noisy.index, columns=noisy.columns)
        perturbed = BacktestPanel(
            scores=noisy,
            forward_returns=panel.forward_returns,
            volatility=panel.volatility,
            adv=panel.adv,
        )
        alpha_record = run_markowitz_backtest(
            perturbed,
            returns,
            constraints,
            formulation=formulation,
            capital=capital,
            budget=budget,
            risk_aversion=risk_aversion,
        )
        covariance_returns = returns.copy()
        if level > 0.0:
            covariance_generator = np.random.default_rng([seed, 2, *seed_words])
            scales = causal_scales.to_numpy(dtype=np.float64)
            covariance_noise = covariance_generator.normal(0.0, level * scales, size=returns.shape)
            covariance_returns = pd.DataFrame(
                returns.to_numpy(dtype=float) + covariance_noise,
                index=returns.index,
                columns=returns.columns,
            )
        covariance_record = run_markowitz_backtest(
            panel,
            covariance_returns,
            constraints,
            formulation=formulation,
            capital=capital,
            budget=budget,
            risk_aversion=risk_aversion,
        )
        summary = summarize_record(alpha_record)
        covariance_summary = summarize_record(covariance_record)
        summary["alpha_error"] = level
        summary["covariance_error"] = level
        summary.update({f"covariance_{name}": value for name, value in covariance_summary.items()})
        rows.append(summary)
    return pd.DataFrame(rows).sort_values("alpha_error").reset_index(drop=True)
