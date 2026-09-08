"""Tests for ranking portfolios and uncertainty-aware sizing (SF-S4-MR1).

Organised by the property each defends. The central one is that **no book is
ever returned that violates a declared limit** — the projection either satisfies
every constraint or raises. Everything else (ties, missing assets, zero
volatility, changing universes, infeasible limits, extreme scores, rebalance
boundaries) is a way that invariant is usually broken in practice.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from alphaforge.portfolio.allocation import (
    apply_uncertainty_sizing,
    apply_volatility_target,
    equal_weight_fallback,
    inverse_volatility_portfolio,
    long_short_spread_portfolio,
    quantile_portfolio,
    rank_weighted_portfolio,
    score_weighted_portfolio,
    top_k_portfolio,
)
from alphaforge.portfolio.contracts import (
    AllocationResult,
    InfeasibleConstraintsError,
    PortfolioConstraints,
    PortfolioError,
    liquidity_caps_from_adv,
    project_to_feasible,
)
from alphaforge.portfolio.evidence import (
    BacktestPanel,
    Fold,
    capacity_frontier,
    chronological_folds,
    compare_allocation_policies,
    run_allocation_backtest,
    score_final_holdout,
    summarize_record,
    volatility_regimes,
)

N_SYMBOLS = 20
SYMBOLS = [f"S{index:02d}" for index in range(N_SYMBOLS)]


@pytest.fixture
def scores() -> pd.Series:
    rng = np.random.default_rng(4)
    return pd.Series(rng.normal(0.0, 1.0, N_SYMBOLS), index=SYMBOLS)


@pytest.fixture
def volatility() -> pd.Series:
    rng = np.random.default_rng(5)
    return pd.Series(np.exp(rng.normal(-4.0, 0.3, N_SYMBOLS)), index=SYMBOLS)


@pytest.fixture
def long_short() -> PortfolioConstraints:
    return PortfolioConstraints(max_position=0.12, max_gross=1.0, max_net=0.10, max_leverage=1.0)


@pytest.fixture
def long_only() -> PortfolioConstraints:
    return PortfolioConstraints(max_position=0.15, max_gross=1.0, max_net=1.0, long_only=True)


def _assert_feasible(result: AllocationResult) -> None:
    """Assert every declared limit actually holds on the returned book."""
    constraints = result.constraints
    weights = result.weights.to_numpy(dtype=float)
    assert np.isfinite(weights).all()
    assert np.max(np.abs(weights)) <= constraints.max_position + 1e-9
    assert result.gross <= min(constraints.max_gross, constraints.deployable) + 1e-9
    assert abs(result.net) <= constraints.max_net + 1e-9
    assert result.gross <= constraints.max_leverage + 1e-9
    if constraints.long_only:
        assert (weights >= -1e-12).all()


def _all_policies(
    scores: pd.Series, volatility: pd.Series, constraints: PortfolioConstraints
) -> list[AllocationResult]:
    results = [
        top_k_portfolio(scores, constraints, k=6, weighting="equal"),
        top_k_portfolio(scores, constraints, k=6, weighting="score"),
        top_k_portfolio(scores, constraints, k=6, weighting="rank"),
        score_weighted_portfolio(scores, constraints),
        rank_weighted_portfolio(scores, constraints),
        inverse_volatility_portfolio(scores, volatility, constraints),
    ]
    if not constraints.long_only:
        results.append(long_short_spread_portfolio(scores, constraints, k=5))
        results.append(quantile_portfolio(scores, constraints, quantile=0.25))
    return results


# ---------------------------------------------------------------------------
# The central invariant
# ---------------------------------------------------------------------------


def test_every_policy_returns_a_feasible_book(
    scores: pd.Series, volatility: pd.Series, long_short: PortfolioConstraints
) -> None:
    for result in _all_policies(scores, volatility, long_short):
        _assert_feasible(result)


def test_every_long_only_policy_returns_a_feasible_book(
    scores: pd.Series, volatility: pd.Series, long_only: PortfolioConstraints
) -> None:
    for result in _all_policies(scores, volatility, long_only):
        _assert_feasible(result)


def test_projection_is_not_clip_then_rescale() -> None:
    """The bug this module exists to prevent.

    Rescaling a clipped vector to hit a gross target re-inflates exactly the
    names the position cap just pulled down. A correct projection satisfies both
    at once.
    """
    constraints = PortfolioConstraints(max_position=0.10, max_gross=1.0, max_net=1.0)
    target = np.array([10.0, 1.0, 1.0, 1.0, 1.0] + [0.1] * 15)
    projected = project_to_feasible(target, constraints)
    assert np.max(np.abs(projected)) <= constraints.max_position + 1e-9
    assert float(np.sum(np.abs(projected))) <= constraints.max_gross + 1e-9


def test_projection_deploys_available_capital() -> None:
    """Clipping frees capital; it must be redistributed, not silently held."""
    constraints = PortfolioConstraints(max_position=0.10, max_gross=1.0, max_net=1.0)
    target = np.array([10.0] + [0.01] * 19)
    projected = project_to_feasible(target, constraints)
    assert float(np.sum(np.abs(projected))) == pytest.approx(1.0, abs=1e-6)


def test_gross_shortfall_is_reported_when_unavoidable(long_only: PortfolioConstraints) -> None:
    """5 names at a 0.15 cap cannot reach gross 1.0 — say so, do not hide it."""
    result = top_k_portfolio(
        pd.Series(np.arange(N_SYMBOLS, dtype=float), index=SYMBOLS), long_only, k=5
    )
    assert result.diagnostics["gross_shortfall"] is True
    assert result.diagnostics["deployed_fraction"] == pytest.approx(0.75)
    assert "position cap" in result.diagnostics["shortfall_reason"]
    assert result.cash_weight == pytest.approx(0.25)


# ---------------------------------------------------------------------------
# Property tests named by the acceptance criteria
# ---------------------------------------------------------------------------


def test_ties_are_broken_deterministically(long_only: PortfolioConstraints) -> None:
    """An all-equal cross-section must give the same book every time."""
    tied = pd.Series(1.0, index=SYMBOLS)
    first = top_k_portfolio(tied, long_only, k=6)
    second = top_k_portfolio(tied.sample(frac=1.0, random_state=7), long_only, k=6)
    pd.testing.assert_series_equal(first.weights.sort_index(), second.weights.sort_index())


def test_shuffled_input_order_does_not_change_the_book(
    scores: pd.Series, long_short: PortfolioConstraints
) -> None:
    baseline = rank_weighted_portfolio(scores, long_short)
    shuffled = rank_weighted_portfolio(scores.sample(frac=1.0, random_state=3), long_short)
    pd.testing.assert_series_equal(
        baseline.weights.sort_index(), shuffled.weights.sort_index(), atol=1e-12
    )


def test_missing_and_non_finite_scores_are_dropped(long_short: PortfolioConstraints) -> None:
    values = pd.Series(np.arange(N_SYMBOLS, dtype=float), index=SYMBOLS)
    values.iloc[0] = np.nan
    values.iloc[1] = np.inf
    result = rank_weighted_portfolio(values, long_short)
    assert result.weights.index.tolist() == SYMBOLS[2:]
    _assert_feasible(result)


def test_all_missing_scores_are_refused(long_short: PortfolioConstraints) -> None:
    with pytest.raises(PortfolioError, match="no finite scores"):
        rank_weighted_portfolio(pd.Series(np.nan, index=SYMBOLS), long_short)


def test_duplicate_symbols_are_refused(long_short: PortfolioConstraints) -> None:
    duplicated = pd.Series([1.0, 2.0], index=["A", "A"])
    with pytest.raises(PortfolioError, match="unique symbol index"):
        rank_weighted_portfolio(duplicated, long_short)


def test_zero_volatility_names_are_excluded_not_imputed(
    scores: pd.Series, volatility: pd.Series, long_short: PortfolioConstraints
) -> None:
    """An unmeasurable risk is not a small risk."""
    broken = volatility.copy()
    broken.iloc[:3] = 0.0
    broken.iloc[3] = np.nan
    result = inverse_volatility_portfolio(scores, broken, long_short)
    assert (result.weights.iloc[:4].abs() <= 1e-12).all()
    assert result.diagnostics["excluded_unmeasurable_volatility"] == 4
    _assert_feasible(result)


def test_all_zero_volatility_is_refused(
    scores: pd.Series, long_short: PortfolioConstraints
) -> None:
    with pytest.raises(PortfolioError, match="usable positive volatility"):
        inverse_volatility_portfolio(scores, pd.Series(0.0, index=SYMBOLS), long_short)


def test_changing_universe_charges_turnover_for_exits(
    long_short: PortfolioConstraints,
) -> None:
    """A name that leaves the universe must be charged for being exited."""
    constrained = PortfolioConstraints(
        max_position=0.12, max_gross=1.0, max_net=0.10, max_turnover=None
    )
    previous = pd.Series(0.1, index=SYMBOLS[:10])
    scores = pd.Series(np.arange(10, dtype=float), index=SYMBOLS[10:])
    result = rank_weighted_portfolio(scores, constrained, previous=previous)
    # Every prior name exited and every new name entered.
    assert result.turnover > 1.0
    _assert_feasible(result)


def test_extreme_scores_do_not_dominate_a_rank_book(
    long_short: PortfolioConstraints,
) -> None:
    """The reason rank weighting exists."""
    values = pd.Series(np.arange(N_SYMBOLS, dtype=float), index=SYMBOLS)
    values.iloc[-1] = 1e9
    ranked = rank_weighted_portfolio(values, long_short)
    scored = score_weighted_portfolio(values, long_short)
    top_rank = float(abs(ranked.weights.iloc[-1]))
    top_score = float(abs(scored.weights.iloc[-1]))
    assert top_rank < top_score or top_score == pytest.approx(long_short.max_position, abs=1e-9)
    _assert_feasible(ranked)
    _assert_feasible(scored)


def test_flat_cross_section_falls_back_to_equal_weight(
    long_short: PortfolioConstraints,
) -> None:
    flat = pd.Series(3.0, index=SYMBOLS)
    result = score_weighted_portfolio(flat, long_short)
    assert result.policy == "equal_weight"
    assert result.diagnostics["reason"].startswith("cross-section carried no dispersion")


def test_turnover_cap_binds_at_a_rebalance(long_short: PortfolioConstraints) -> None:
    capped = PortfolioConstraints(max_position=0.12, max_gross=1.0, max_net=0.10, max_turnover=0.20)
    previous = pd.Series(0.0, index=SYMBOLS)
    scores = pd.Series(np.arange(N_SYMBOLS, dtype=float), index=SYMBOLS)
    result = rank_weighted_portfolio(scores, capped, previous=previous)
    assert capped.max_turnover is not None
    assert result.turnover <= capped.max_turnover + 1e-9
    _assert_feasible(result)


def test_zero_turnover_cap_freezes_an_already_feasible_book() -> None:
    frozen = PortfolioConstraints(max_position=0.12, max_gross=1.0, max_net=0.10, max_turnover=0.0)
    # The starting book must itself satisfy every other limit, or the frozen
    # constraint set is genuinely empty — see the companion test below.
    previous = pd.Series(np.r_[np.full(10, 0.05), np.full(10, -0.05)], index=SYMBOLS)
    scores = pd.Series(np.arange(N_SYMBOLS, dtype=float), index=SYMBOLS)
    result = rank_weighted_portfolio(scores, frozen, previous=previous)
    np.testing.assert_allclose(result.weights.to_numpy(), previous.to_numpy(), atol=1e-9)
    assert result.turnover == pytest.approx(0.0, abs=1e-9)


def test_turnover_budget_too_small_to_fix_the_book_is_refused() -> None:
    """A trade budget that cannot correct an out-of-limit book is infeasible.

    Left unchecked this surfaces as a confusing non-convergence; it is really an
    actionable statement that the constraints conflict.
    """
    frozen = PortfolioConstraints(max_position=0.12, max_gross=1.0, max_net=0.10, max_turnover=0.0)
    # Net 1.0 against a 0.10 limit, with no budget to trade back.
    previous = pd.Series(0.05, index=SYMBOLS)
    scores = pd.Series(np.arange(N_SYMBOLS, dtype=float), index=SYMBOLS)
    with pytest.raises(InfeasibleConstraintsError, match="turnover budget"):
        rank_weighted_portfolio(scores, frozen, previous=previous)


# ---------------------------------------------------------------------------
# Infeasible limits
# ---------------------------------------------------------------------------


def test_position_cap_too_tight_for_the_universe_is_refused() -> None:
    constraints = PortfolioConstraints(max_position=0.02, max_gross=1.0)
    with pytest.raises(InfeasibleConstraintsError, match="relax the position cap"):
        constraints.check_feasible(20)


def test_liquidity_caps_below_the_gross_target_are_refused(
    scores: pd.Series, long_only: PortfolioConstraints
) -> None:
    adv = pd.Series(1e6, index=SYMBOLS)
    caps = liquidity_caps_from_adv(adv, capital=1e9, participation=0.05)
    with pytest.raises(InfeasibleConstraintsError, match="liquidity caps permit"):
        top_k_portfolio(scores, long_only, k=6, liquidity_caps=caps)


def test_illiquid_names_receive_no_weight(
    scores: pd.Series, long_only: PortfolioConstraints
) -> None:
    """An unknown or zero ADV is untradeable, not unconstrained."""
    adv = pd.Series(2e8, index=SYMBOLS)
    adv.iloc[:3] = 0.0
    caps = liquidity_caps_from_adv(adv, capital=1e8, participation=0.25)
    result = top_k_portfolio(scores, long_only, k=8, liquidity_caps=caps)
    assert (result.weights.iloc[:3].abs() <= 1e-12).all()
    _assert_feasible(result)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"max_position": 0.0}, "max_position"),
        ({"max_gross": -1.0}, "max_gross"),
        ({"max_net": -0.1}, "max_net"),
        ({"max_turnover": -1.0}, "max_turnover"),
        ({"cash_buffer": 1.0}, "cash_buffer"),
        ({"max_net": 2.0, "max_gross": 1.0}, "max_net cannot exceed"),
        ({"max_gross": 2.0, "max_leverage": 1.0}, "max_gross cannot exceed"),
        ({"max_position": 2.0, "max_gross": 1.0}, "max_position cannot exceed"),
    ],
)
def test_constraint_validation(kwargs: dict, message: str) -> None:
    with pytest.raises(PortfolioError, match=message):
        PortfolioConstraints(**kwargs)


def test_long_short_policies_refuse_a_long_only_mandate(
    scores: pd.Series, long_only: PortfolioConstraints
) -> None:
    with pytest.raises(PortfolioError, match="long_only=False"):
        long_short_spread_portfolio(scores, long_only, k=5)


@pytest.mark.parametrize("k", [0, N_SYMBOLS + 1])
def test_out_of_range_k_is_refused(
    scores: pd.Series, long_only: PortfolioConstraints, k: int
) -> None:
    with pytest.raises(PortfolioError, match="k must be in"):
        top_k_portfolio(scores, long_only, k=k)


def test_overlapping_quantile_sleeves_are_refused(
    scores: pd.Series, long_short: PortfolioConstraints
) -> None:
    with pytest.raises(PortfolioError, match="quantile must lie"):
        quantile_portfolio(scores, long_short, quantile=0.9)


# ---------------------------------------------------------------------------
# Uncertainty sizing and volatility targeting
# ---------------------------------------------------------------------------


def test_uncertainty_sizing_preserves_gross_and_shifts_concentration(
    scores: pd.Series, long_short: PortfolioConstraints
) -> None:
    """It redistributes the book; it must not lever it."""
    rng = np.random.default_rng(11)
    uncertainty = pd.Series(np.abs(rng.normal(0.02, 0.008, N_SYMBOLS)), index=SYMBOLS)
    base = rank_weighted_portfolio(scores, long_short).weights
    sized = apply_uncertainty_sizing(base, uncertainty, strength=1.0)
    assert float(sized.abs().sum()) == pytest.approx(float(base.abs().sum()), rel=1e-9)
    assert not np.allclose(sized.to_numpy(), base.to_numpy())


def test_zero_strength_is_a_no_op(scores: pd.Series, long_short: PortfolioConstraints) -> None:
    uncertainty = pd.Series(np.linspace(0.01, 0.05, N_SYMBOLS), index=SYMBOLS)
    base = rank_weighted_portfolio(scores, long_short).weights
    np.testing.assert_allclose(
        apply_uncertainty_sizing(base, uncertainty, strength=0.0).to_numpy(),
        base.to_numpy(),
        atol=1e-12,
    )


def test_uncertainty_tilt_is_capped(scores: pd.Series, long_short: PortfolioConstraints) -> None:
    """An implausibly certain name must not take over the book."""
    uncertainty = pd.Series(0.02, index=SYMBOLS)
    uncertainty.iloc[0] = 1e-12
    base = rank_weighted_portfolio(scores, long_short).weights
    sized = apply_uncertainty_sizing(base, uncertainty, strength=1.0, max_multiple=2.0)
    ratio = abs(sized.iloc[0]) / max(abs(base.iloc[0]), 1e-15)
    assert ratio <= 2.0 + 1e-9


def test_missing_uncertainty_leaves_weights_untouched(
    scores: pd.Series, long_short: PortfolioConstraints
) -> None:
    base = rank_weighted_portfolio(scores, long_short).weights
    empty = pd.Series(np.nan, index=SYMBOLS)
    np.testing.assert_allclose(
        apply_uncertainty_sizing(base, empty, strength=1.0).to_numpy(), base.to_numpy()
    )


@pytest.mark.parametrize(("strength", "multiple"), [(-0.1, 2.0), (1.5, 2.0), (0.5, 0.5)])
def test_uncertainty_sizing_bounds(
    scores: pd.Series, long_short: PortfolioConstraints, strength: float, multiple: float
) -> None:
    base = rank_weighted_portfolio(scores, long_short).weights
    with pytest.raises(PortfolioError):
        apply_uncertainty_sizing(
            base, pd.Series(0.01, index=SYMBOLS), strength=strength, max_multiple=multiple
        )


def test_volatility_target_scales_to_the_target_when_leverage_allows(
    scores: pd.Series, volatility: pd.Series, long_short: PortfolioConstraints
) -> None:
    base = rank_weighted_portfolio(scores, long_short).weights
    covariance = pd.DataFrame(np.diag(volatility.to_numpy() ** 2), index=SYMBOLS, columns=SYMBOLS)
    scaled, diagnostics = apply_volatility_target(
        base, covariance, target_volatility=0.05, max_leverage=4.0
    )
    assert diagnostics["target_met"] is True
    assert diagnostics["leverage_capped"] is False
    assert diagnostics["scaled_ex_ante_volatility"] == pytest.approx(0.05)


def test_volatility_target_is_bounded_by_leverage(
    scores: pd.Series, volatility: pd.Series, long_short: PortfolioConstraints
) -> None:
    """A vol target without a leverage ceiling is how a calm regime ends badly."""
    base = rank_weighted_portfolio(scores, long_short).weights
    covariance = pd.DataFrame(np.diag(volatility.to_numpy() ** 2), index=SYMBOLS, columns=SYMBOLS)
    scaled, diagnostics = apply_volatility_target(
        base, covariance, target_volatility=5.0, max_leverage=1.5
    )
    assert diagnostics["leverage_capped"] is True
    assert diagnostics["target_met"] is False
    assert float(scaled.abs().sum()) <= 1.5 + 1e-9


def test_volatility_target_requires_full_covariance(
    scores: pd.Series, long_short: PortfolioConstraints
) -> None:
    base = rank_weighted_portfolio(scores, long_short).weights
    partial = pd.DataFrame(np.eye(3), index=SYMBOLS[:3], columns=SYMBOLS[:3])
    with pytest.raises(PortfolioError, match="cover every allocated symbol"):
        apply_volatility_target(base, partial, target_volatility=0.1, max_leverage=2.0)


@pytest.mark.parametrize(("target", "leverage"), [(0.0, 2.0), (0.1, 0.0)])
def test_volatility_target_bounds(
    scores: pd.Series,
    volatility: pd.Series,
    long_short: PortfolioConstraints,
    target: float,
    leverage: float,
) -> None:
    base = rank_weighted_portfolio(scores, long_short).weights
    covariance = pd.DataFrame(np.diag(volatility.to_numpy() ** 2), index=SYMBOLS, columns=SYMBOLS)
    with pytest.raises(PortfolioError):
        apply_volatility_target(base, covariance, target_volatility=target, max_leverage=leverage)


# ---------------------------------------------------------------------------
# Evidence harness
# ---------------------------------------------------------------------------


@pytest.fixture
def panel() -> BacktestPanel:
    rng = np.random.default_rng(9)
    n_dates = 260
    dates = pd.bdate_range("2020-01-02", periods=n_dates)
    volatility = pd.DataFrame(
        np.exp(rng.normal(-4.0, 0.3, (n_dates, N_SYMBOLS))), index=dates, columns=SYMBOLS
    )
    forward = pd.DataFrame(
        rng.normal(0.0, 1.0, (n_dates, N_SYMBOLS)) * volatility.to_numpy(),
        index=dates,
        columns=SYMBOLS,
    )
    scores = pd.DataFrame(
        0.35 * forward.to_numpy() / volatility.to_numpy()
        + rng.normal(0.0, 1.0, (n_dates, N_SYMBOLS)),
        index=dates,
        columns=SYMBOLS,
    )
    adv = pd.DataFrame(2e8, index=dates, columns=SYMBOLS)
    return BacktestPanel(scores=scores, forward_returns=forward, volatility=volatility, adv=adv)


def test_panel_requires_aligned_inputs(panel: BacktestPanel) -> None:
    with pytest.raises(PortfolioError, match="index must match"):
        BacktestPanel(
            scores=panel.scores,
            forward_returns=panel.forward_returns.iloc[:-1],
            volatility=panel.volatility,
            adv=panel.adv,
        )
    with pytest.raises(PortfolioError, match="columns must match"):
        BacktestPanel(
            scores=panel.scores,
            forward_returns=panel.forward_returns.rename(columns={SYMBOLS[0]: "X"}),
            volatility=panel.volatility,
            adv=panel.adv,
        )


def test_folds_are_contiguous_and_ordered(panel: BacktestPanel) -> None:
    folds = chronological_folds(panel.scores.index, n_folds=4)
    assert len(folds) == 4
    for earlier, later in zip(folds, folds[1:], strict=False):
        assert earlier.end < later.start


def test_fold_count_is_bounded(panel: BacktestPanel) -> None:
    with pytest.raises(PortfolioError, match="n_folds"):
        chronological_folds(panel.scores.index, n_folds=0)
    with pytest.raises(PortfolioError, match="shorter than"):
        chronological_folds(panel.scores.index[:3], n_folds=10)


def test_regimes_never_use_future_volatility(panel: BacktestPanel) -> None:
    labels = volatility_regimes(panel.forward_returns, window=63)
    assert set(labels.unique()) <= {"calm", "stress", "unknown"}
    assert (labels.iloc[:62] == "unknown").all()
    mutated = panel.forward_returns.copy()
    mutated.iloc[200:] *= 50.0
    perturbed = volatility_regimes(mutated, window=63)
    pd.testing.assert_series_equal(labels.iloc[:200], perturbed.iloc[:200])


def test_comparison_spans_policies_folds_regimes_and_capacity(panel: BacktestPanel) -> None:
    constraints = PortfolioConstraints(
        max_position=0.15, max_gross=1.0, max_net=0.15, max_turnover=0.5, max_leverage=1.0
    )
    folds = chronological_folds(panel.scores.index, n_folds=2)
    regimes = volatility_regimes(panel.forward_returns, window=63)
    comparison = compare_allocation_policies(
        panel, constraints, folds=folds, capital_levels=(5e7, 1e8), regimes=regimes
    )
    assert not comparison.empty
    assert set(comparison["regime"]) <= {"all", "calm", "stress"}
    assert comparison["capital"].nunique() == 2
    assert comparison["fold"].nunique() == 2
    # Sorted by name, never by performance.
    expected = comparison.sort_values(["policy", "capital", "fold", "regime"]).reset_index(
        drop=True
    )
    pd.testing.assert_frame_equal(comparison, expected)


def test_costs_reduce_net_below_gross(panel: BacktestPanel) -> None:
    constraints = PortfolioConstraints(
        max_position=0.15, max_gross=1.0, max_net=0.15, max_turnover=0.5, max_leverage=1.0
    )
    record = run_allocation_backtest(
        panel,
        lambda s, v, c, p, caps: rank_weighted_portfolio(s, c, previous=p, liquidity_caps=caps),
        constraints,
        capital=5e7,
        cost_bps=25.0,
    )
    assert (record["net_return"] <= record["gross_return"] + 1e-15).all()
    assert (record["cost"] >= 0.0).all()
    summary = summarize_record(record)
    assert summary["cost_drag"] > 0.0
    assert summary["n_dates"] == len(record)


def test_zero_cost_leaves_net_equal_to_gross(panel: BacktestPanel) -> None:
    constraints = PortfolioConstraints(max_position=0.15, max_gross=1.0, max_net=0.15)
    record = run_allocation_backtest(
        panel,
        lambda s, v, c, p, caps: rank_weighted_portfolio(s, c, previous=p, liquidity_caps=caps),
        constraints,
        capital=5e7,
        cost_bps=0.0,
    )
    np.testing.assert_allclose(
        record["net_return"].to_numpy(), record["gross_return"].to_numpy(), atol=1e-15
    )


def test_capacity_exhaustion_is_recorded_not_dropped(panel: BacktestPanel) -> None:
    """An infeasible date must stay in the sample, marked infeasible."""
    constraints = PortfolioConstraints(max_position=0.15, max_gross=1.0, max_net=0.15)
    record = run_allocation_backtest(
        panel,
        lambda s, v, c, p, caps: rank_weighted_portfolio(s, c, previous=p, liquidity_caps=caps),
        constraints,
        capital=1e12,
        cost_bps=10.0,
    )
    assert not record.empty
    assert (~record["feasible"]).all()
    assert summarize_record(record)["feasible_fraction"] == 0.0


def test_capacity_frontier_reports_each_level(panel: BacktestPanel) -> None:
    constraints = PortfolioConstraints(max_position=0.15, max_gross=1.0, max_net=0.15)
    comparison = compare_allocation_policies(
        panel,
        constraints,
        folds=chronological_folds(panel.scores.index, n_folds=2),
        capital_levels=(5e7, 1e8, 1e12),
    )
    frontier = capacity_frontier(comparison)
    assert set(frontier["capital"]) == {5e7, 1e8, 1e12}
    exhausted = frontier.loc[frontier["capital"] == 1e12, "feasible_fraction"]
    assert (exhausted == 0.0).all()


def test_holdout_is_scored_separately_and_once(panel: BacktestPanel) -> None:
    """There is no code path by which the holdout can influence a choice."""
    constraints = PortfolioConstraints(max_position=0.15, max_gross=1.0, max_net=0.15)
    development = panel.scores.index[:200]
    holdout = panel.scores.index[200:]
    comparison = compare_allocation_policies(
        panel,
        constraints,
        folds=chronological_folds(development, n_folds=2),
        capital_levels=(5e7,),
    )
    assert comparison["fold"].nunique() == 2
    summary = score_final_holdout(
        panel,
        lambda s, v, c, p, caps: rank_weighted_portfolio(s, c, previous=p, liquidity_caps=caps),
        constraints,
        holdout_dates=holdout,
        capital=5e7,
    )
    assert summary["holdout"] is True
    assert summary["n_dates"] == len(holdout)


def test_empty_record_summarizes_without_raising() -> None:
    summary = summarize_record(pd.DataFrame())
    assert summary["n_dates"] == 0
    assert np.isnan(summary["net_return"])


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"capital_levels": ()}, "capital_levels"),
        ({"capital_levels": (-1.0,)}, "finite and positive"),
    ],
)
def test_comparison_bounds(panel: BacktestPanel, kwargs: dict, message: str) -> None:
    constraints = PortfolioConstraints(max_position=0.15, max_gross=1.0, max_net=0.15)
    settings: dict = {
        "folds": chronological_folds(panel.scores.index, n_folds=2),
        "capital_levels": (5e7,),
    }
    settings.update(kwargs)
    with pytest.raises(PortfolioError, match=message):
        compare_allocation_policies(panel, constraints, **settings)


def test_comparison_requires_a_fold(panel: BacktestPanel) -> None:
    constraints = PortfolioConstraints(max_position=0.15, max_gross=1.0, max_net=0.15)
    with pytest.raises(PortfolioError, match="at least one evaluation fold"):
        compare_allocation_policies(panel, constraints, folds=(), capital_levels=(5e7,))


def test_fold_ordering_is_validated() -> None:
    with pytest.raises(PortfolioError, match="ends before it starts"):
        Fold(name="bad", start=pd.Timestamp("2021-01-01"), end=pd.Timestamp("2020-01-01"))


def test_equal_weight_fallback_is_feasible(long_only: PortfolioConstraints) -> None:
    result = equal_weight_fallback(pd.Series(1.0, index=SYMBOLS), long_only)
    _assert_feasible(result)
    assert result.n_positions == N_SYMBOLS


def test_liquidity_cap_bounds() -> None:
    adv = pd.Series(1e6, index=SYMBOLS)
    with pytest.raises(PortfolioError, match="capital"):
        liquidity_caps_from_adv(adv, capital=0.0, participation=0.1)
    with pytest.raises(PortfolioError, match="participation"):
        liquidity_caps_from_adv(adv, capital=1e6, participation=1.5)
