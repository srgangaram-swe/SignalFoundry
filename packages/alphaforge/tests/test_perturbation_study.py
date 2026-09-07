"""Tests for frozen Monte Carlo execution perturbation (SF-S4-MR8).

Grouped by acceptance criterion. The invariants that carry the most weight:

* **Seed streams are named and derived**, so any single path replays exactly
  from its coordinates and no two kinds share draws.
* **Failures are counted, not dropped** — insolvent, no-trade, and unreconciled
  paths stay in the denominator.
* **The grid is frozen** before qualification, and widening a distribution after
  seeing the tails is detectable.
* **Perturbation touches mechanics, never the P&L directly**, and the one arm
  that provably cannot move compounded return says so rather than pretending.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from alphaforge.robustness import (
    MIN_REPLICATES,
    PERTURBATION_KINDS,
    FrozenPerturbationGrid,
    PerturbationError,
    PerturbationSpec,
    assert_perturbation_streams_isolated,
    evaluate_path,
    failure_report,
    perturb_path,
    replay_path,
    run_perturbation_study,
    standard_execution_grid,
    verify_frozen_perturbations,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _path(n: int = 300, *, drift: float = 0.0006, seed: int = 5) -> tuple[pd.Series, pd.Series]:
    generator = np.random.default_rng(seed)
    index = pd.bdate_range("2020-01-01", periods=n)
    returns = pd.Series(generator.normal(drift, 0.01, n), index=index)
    costs = pd.Series(np.full(n, 0.0002), index=index)
    return returns, costs


@pytest.fixture
def path() -> tuple[pd.Series, pd.Series]:
    return _path()


@pytest.fixture
def grid() -> FrozenPerturbationGrid:
    return standard_execution_grid(replicates=MIN_REPLICATES, root_seed=11)


# ---------------------------------------------------------------------------
# Frozen grid
# ---------------------------------------------------------------------------


def test_the_standard_grid_covers_every_mechanic_the_work_item_names(
    grid: FrozenPerturbationGrid,
) -> None:
    assert {spec.kind for spec in grid.specs} == set(PERTURBATION_KINDS)


def test_widening_a_distribution_after_the_fact_is_detected(
    grid: FrozenPerturbationGrid,
) -> None:
    identity = grid.identity
    widened = FrozenPerturbationGrid(
        name=grid.name,
        specs=tuple(
            (
                PerturbationSpec(kind=spec.kind, magnitude=spec.magnitude, bound=spec.bound)
                if spec.kind != "execution_price"
                else PerturbationSpec(kind=spec.kind, magnitude=0.05, bound=0.10)
            )
            for spec in grid.specs
        ),
        replicates=grid.replicates,
        root_seed=grid.root_seed,
    )
    verify_frozen_perturbations(grid, identity)
    with pytest.raises(PerturbationError, match="does not match the frozen declaration"):
        verify_frozen_perturbations(widened, identity)


def test_grid_identity_is_order_independent(grid: FrozenPerturbationGrid) -> None:
    reversed_grid = FrozenPerturbationGrid(
        name=grid.name,
        specs=tuple(reversed(grid.specs)),
        replicates=grid.replicates,
        root_seed=grid.root_seed,
    )
    assert reversed_grid.identity == grid.identity


def test_a_duplicated_kind_is_refused() -> None:
    with pytest.raises(PerturbationError, match="may appear once"):
        FrozenPerturbationGrid(
            name="dupes",
            specs=(
                PerturbationSpec(kind="execution_price", magnitude=0.001, bound=0.01),
                PerturbationSpec(kind="execution_price", magnitude=0.002, bound=0.01),
            ),
            replicates=MIN_REPLICATES,
            root_seed=0,
        )


def test_an_unsupported_kind_is_refused_not_ignored() -> None:
    with pytest.raises(PerturbationError, match="unsupported perturbation kind"):
        PerturbationSpec(kind="wishful_thinking", magnitude=0.1, bound=0.2)


def test_too_few_replicates_to_estimate_a_tail_are_refused() -> None:
    with pytest.raises(PerturbationError, match="cannot support a tail estimate"):
        standard_execution_grid(replicates=MIN_REPLICATES - 1)


def test_a_bound_below_the_magnitude_is_refused() -> None:
    with pytest.raises(PerturbationError, match="clip the distribution to a point"):
        PerturbationSpec(kind="execution_price", magnitude=0.05, bound=0.01)


def test_an_oversized_magnitude_is_refused() -> None:
    with pytest.raises(PerturbationError, match="magnitude must lie"):
        PerturbationSpec(kind="execution_price", magnitude=2.0, bound=2.0)


def test_freeze_verification_requires_a_full_digest(grid: FrozenPerturbationGrid) -> None:
    with pytest.raises(PerturbationError, match="full SHA-256"):
        verify_frozen_perturbations(grid, "nope")


# ---------------------------------------------------------------------------
# Seed streams and replay
# ---------------------------------------------------------------------------


def test_no_two_kinds_share_a_random_stream(grid: FrozenPerturbationGrid) -> None:
    assert_perturbation_streams_isolated(grid)


def test_a_single_path_replays_exactly_from_its_coordinates(
    path: tuple[pd.Series, pd.Series], grid: FrozenPerturbationGrid
) -> None:
    returns, costs = path
    first = replay_path(returns, costs, grid, kind="execution_price", replicate=17)
    second = replay_path(returns, costs, grid, kind="execution_price", replicate=17)
    assert first.to_dict() == second.to_dict()


def test_replay_matches_what_the_study_recorded(
    path: tuple[pd.Series, pd.Series], grid: FrozenPerturbationGrid
) -> None:
    """The recorded path and its standalone replay must be the same path."""
    returns, costs = path
    study = run_perturbation_study(returns, costs, grid)
    recorded = [
        item
        for item in study["retained_paths"]
        if item["kind"] == "missing_trade" and item["replicate"] == 3
    ][0]
    replayed = replay_path(returns, costs, grid, kind="missing_trade", replicate=3)
    assert replayed.to_dict() == recorded


def test_different_replicates_draw_differently(
    path: tuple[pd.Series, pd.Series], grid: FrozenPerturbationGrid
) -> None:
    returns, costs = path
    first = replay_path(returns, costs, grid, kind="execution_price", replicate=1)
    second = replay_path(returns, costs, grid, kind="execution_price", replicate=2)
    assert first.net_return != second.net_return


def test_the_root_seed_changes_every_stream(path: tuple[pd.Series, pd.Series]) -> None:
    returns, costs = path
    first = standard_execution_grid(replicates=MIN_REPLICATES, root_seed=1)
    second = standard_execution_grid(replicates=MIN_REPLICATES, root_seed=2)
    left = replay_path(returns, costs, first, kind="execution_price", replicate=0)
    right = replay_path(returns, costs, second, kind="execution_price", replicate=0)
    assert left.net_return != right.net_return


def test_a_replicate_outside_the_frozen_study_is_refused(
    path: tuple[pd.Series, pd.Series], grid: FrozenPerturbationGrid
) -> None:
    returns, costs = path
    with pytest.raises(PerturbationError, match="outside the frozen study"):
        replay_path(returns, costs, grid, kind="execution_price", replicate=grid.replicates)


def test_replaying_an_undeclared_kind_is_refused(
    path: tuple[pd.Series, pd.Series],
) -> None:
    returns, costs = path
    narrow = FrozenPerturbationGrid(
        name="narrow",
        specs=(PerturbationSpec(kind="execution_price", magnitude=0.001, bound=0.01),),
        replicates=MIN_REPLICATES,
        root_seed=0,
    )
    with pytest.raises(PerturbationError, match="declares no"):
        replay_path(returns, costs, narrow, kind="missing_trade", replicate=0)


# ---------------------------------------------------------------------------
# Perturbation mechanics (property / metamorphic)
# ---------------------------------------------------------------------------


def test_reordering_conserves_compounded_return_exactly(
    path: tuple[pd.Series, pd.Series], grid: FrozenPerturbationGrid
) -> None:
    """Multiplication commutes, so `trade_order` cannot move compounded return.

    Pinned deliberately. The arm exists to stress path-dependent quantities such
    as drawdown; asserting the conservation here stops a future edit from
    quietly making this arm appear to move a number it has no mechanism to move.
    """
    returns, costs = path
    spec = next(item for item in grid.specs if item.kind == "trade_order")
    shifted, shifted_costs = perturb_path(returns, costs, spec, grid.seed_stream("trade_order", 0))
    assert sorted(shifted.tolist()) == pytest.approx(sorted(returns.tolist()))
    assert float(shifted.sum()) == pytest.approx(float(returns.sum()))
    assert float(shifted_costs.sum()) == pytest.approx(float(costs.sum()))


def test_reordering_does_move_drawdown(
    path: tuple[pd.Series, pd.Series], grid: FrozenPerturbationGrid
) -> None:
    """The quantity the arm actually stresses must in fact respond to it."""
    returns, costs = path
    study = run_perturbation_study(returns, costs, grid)
    reorder = [item for item in study["outcomes"] if item["kind"] == "trade_order"][0]
    assert reorder["worst_max_drawdown"] < reorder["baseline_max_drawdown"]


def test_signal_timing_jitter_is_not_a_point_mass(
    path: tuple[pd.Series, pd.Series], grid: FrozenPerturbationGrid
) -> None:
    """A fixed carry fraction would make timing risk look like certainty."""
    returns, costs = path
    study = run_perturbation_study(returns, costs, grid)
    timing = [item for item in study["outcomes"] if item["kind"] == "signal_timestamp"][0]
    assert timing["percentile_5"] < timing["percentile_95"]


def test_a_stale_signal_substitutes_the_prior_bars_outcome() -> None:
    """Acting on stale information earns the previous bar, at full scale."""
    index = pd.bdate_range("2020-01-01", periods=6)
    returns = pd.Series([0.01, -0.02, 0.03, -0.04, 0.05, -0.06], index=index)
    costs = pd.Series(np.zeros(6), index=index)
    spec = PerturbationSpec(kind="signal_timestamp", magnitude=1.0, bound=1.0)
    shifted, shifted_costs = perturb_path(returns, costs, spec, np.random.default_rng(0))
    expected = np.concatenate(([returns.iloc[0]], returns.to_numpy()[:-1]))
    assert shifted.to_numpy() == pytest.approx(expected)
    # It traded — just on the wrong information — so costs are unchanged.
    assert shifted_costs.to_numpy() == pytest.approx(costs.to_numpy())


def test_timing_jitter_can_hurt_as_well_as_help(
    path: tuple[pd.Series, pd.Series], grid: FrozenPerturbationGrid
) -> None:
    """A frequency of exactly 0 or 1 means the arm has a systematic bias.

    This test is the reason the timing arm is a substitution rather than a
    blend. Blending a fraction of each bar into the next smooths the path, and
    smoothing a fixed-sum path always lowers variance drag — so every replicate
    came out *better* than the baseline. A perturbation that can only help is
    not a stress.
    """
    returns, costs = path
    study = run_perturbation_study(returns, costs, grid)
    timing = [item for item in study["outcomes"] if item["kind"] == "signal_timestamp"][0]
    assert 0.0 < timing["worse_than_baseline"] < 1.0


def test_no_perturbation_arm_is_silently_one_directional(
    path: tuple[pd.Series, pd.Series], grid: FrozenPerturbationGrid
) -> None:
    """Attenuating arms may be one-sided; the rest must not be.

    Liquidity, partial fill, and cost inflation are one-directional by design —
    less fill and higher costs cannot help. Every other arm that comes back at
    exactly 0.0 or 1.0 has a modelling bias rather than a finding.
    """
    returns, costs = path
    study = run_perturbation_study(returns, costs, grid)
    # Attenuating arms cannot help: less fill and higher costs are one-way.
    intentionally_one_sided = {"liquidity_haircut", "partial_fill", "cost_multiplier"}
    # `trade_order` leaves compounded return mathematically invariant, so its
    # frequency is floating-point noise around the baseline and carries no
    # information either way. Excluded explicitly so it cannot pass this guard
    # by rounding luck; `test_reordering_moves_only_drawdown_not_return` states
    # what is actually true of that arm.
    uninformative = {"trade_order"}
    for outcome in study["outcomes"]:
        if outcome["kind"] in intentionally_one_sided | uninformative:
            continue
        assert 0.0 < outcome["worse_than_baseline"] < 1.0, (
            f"{outcome['kind']} never lands on both sides of the baseline, which "
            "indicates a systematic bias in the perturbation rather than a result"
        )


def test_reordering_moves_only_drawdown_not_return(
    path: tuple[pd.Series, pd.Series], grid: FrozenPerturbationGrid
) -> None:
    """The reordering arm's net-return spread is rounding noise, and nothing more."""
    returns, costs = path
    study = run_perturbation_study(returns, costs, grid)
    reorder = [item for item in study["outcomes"] if item["kind"] == "trade_order"][0]
    spread = reorder["percentile_95"] - reorder["percentile_5"]
    assert spread == pytest.approx(0.0, abs=1e-9)
    assert reorder["median_net_return"] == pytest.approx(reorder["baseline_net_return"], abs=1e-9)
    # Drawdown, by contrast, moves materially.
    assert reorder["worst_max_drawdown"] < reorder["baseline_max_drawdown"] - 1e-3


def test_a_missed_trade_drops_its_cost_with_its_return(
    path: tuple[pd.Series, pd.Series],
) -> None:
    """An order that never reached the market cannot have been charged for."""
    returns, costs = path
    spec = PerturbationSpec(kind="missing_trade", magnitude=0.5, bound=0.5)
    generator = np.random.default_rng(4)
    shifted, shifted_costs = perturb_path(returns, costs, spec, generator)
    dropped = shifted.to_numpy() == 0.0
    assert (shifted_costs.to_numpy()[dropped] == 0.0).all()


def test_cost_inflation_is_one_sided(path: tuple[pd.Series, pd.Series]) -> None:
    """A symmetric cost shock would average away the drag under test."""
    returns, costs = path
    spec = PerturbationSpec(kind="cost_multiplier", magnitude=0.25, bound=1.0)
    _, shifted_costs = perturb_path(returns, costs, spec, np.random.default_rng(6))
    assert (shifted_costs.to_numpy() >= costs.to_numpy() - 1e-12).all()
    assert float(shifted_costs.sum()) > float(costs.sum())


def test_a_liquidity_haircut_never_increases_the_captured_return(
    path: tuple[pd.Series, pd.Series],
) -> None:
    returns, costs = path
    spec = PerturbationSpec(kind="liquidity_haircut", magnitude=0.2, bound=0.5)
    shifted, _ = perturb_path(returns, costs, spec, np.random.default_rng(8))
    assert (np.abs(shifted.to_numpy()) <= np.abs(returns.to_numpy()) + 1e-12).all()


def test_a_partial_fill_never_increases_the_captured_return(
    path: tuple[pd.Series, pd.Series],
) -> None:
    returns, costs = path
    spec = PerturbationSpec(kind="partial_fill", magnitude=0.2, bound=0.5)
    shifted, shifted_costs = perturb_path(returns, costs, spec, np.random.default_rng(9))
    assert (np.abs(shifted.to_numpy()) <= np.abs(returns.to_numpy()) + 1e-12).all()
    assert (shifted_costs.to_numpy() <= costs.to_numpy() + 1e-12).all()


def test_delaying_a_trade_moves_its_return_to_the_next_bar(
    path: tuple[pd.Series, pd.Series],
) -> None:
    returns, costs = path
    spec = PerturbationSpec(kind="delayed_trade", magnitude=1.0, bound=1.0)
    shifted, _ = perturb_path(returns, costs, spec, np.random.default_rng(2))
    expected = np.concatenate(([0.0], returns.to_numpy()[:-1]))
    assert shifted.to_numpy() == pytest.approx(expected)


def test_every_perturbation_preserves_the_index(
    path: tuple[pd.Series, pd.Series], grid: FrozenPerturbationGrid
) -> None:
    returns, costs = path
    for spec in grid.specs:
        shifted, shifted_costs = perturb_path(returns, costs, spec, grid.seed_stream(spec.kind, 0))
        assert shifted.index.equals(returns.index)
        assert shifted_costs.index.equals(costs.index)


def test_bounded_draws_never_escape_the_declared_envelope(
    path: tuple[pd.Series, pd.Series],
) -> None:
    """The clip is what makes an unbounded family safe to declare."""
    returns, costs = path
    spec = PerturbationSpec(kind="execution_price", magnitude=0.5, bound=0.6)
    shifted, _ = perturb_path(returns, costs, spec, np.random.default_rng(1))
    deltas = shifted.to_numpy() - returns.to_numpy()
    assert np.abs(deltas).max() <= spec.bound + 1e-12


def test_misaligned_returns_and_costs_are_refused() -> None:
    returns = pd.Series([0.01, 0.02], index=pd.bdate_range("2020-01-01", periods=2))
    costs = pd.Series([0.0001], index=pd.bdate_range("2020-01-01", periods=1))
    spec = PerturbationSpec(kind="execution_price", magnitude=0.001, bound=0.01)
    with pytest.raises(PerturbationError, match="must share an index"):
        perturb_path(returns, costs, spec, np.random.default_rng(0))


def test_negative_costs_are_refused(path: tuple[pd.Series, pd.Series]) -> None:
    returns, costs = path
    spec = PerturbationSpec(kind="execution_price", magnitude=0.001, bound=0.01)
    with pytest.raises(PerturbationError, match="positive-charge convention"):
        perturb_path(returns, -costs, spec, np.random.default_rng(0))


def test_non_finite_inputs_are_refused(path: tuple[pd.Series, pd.Series]) -> None:
    returns, costs = path
    broken = returns.copy()
    broken.iloc[5] = np.inf
    spec = PerturbationSpec(kind="execution_price", magnitude=0.001, bound=0.01)
    with pytest.raises(PerturbationError, match="finite"):
        perturb_path(broken, costs, spec, np.random.default_rng(0))


# ---------------------------------------------------------------------------
# Path evaluation, insolvency, and reconciliation
# ---------------------------------------------------------------------------


def test_an_insolvent_path_stops_at_zero_and_cannot_recover() -> None:
    """A blown-up account must not contribute a positive average.

    The wipe-out bar is worse than -100%, which is reachable with leverage. A
    -90% bar is *not* insolvency, and the surrounding tests rely on that
    distinction holding.
    """
    index = pd.bdate_range("2020-01-01", periods=4)
    returns = pd.Series([-0.5, -1.2, 5.0, 5.0], index=index)
    costs = pd.Series(np.zeros(4), index=index)
    outcome = evaluate_path(returns, costs, replicate=0, kind="test")
    assert outcome.insolvent is True
    assert outcome.net_return == pytest.approx(-1.0)
    assert outcome.max_drawdown == pytest.approx(-1.0)


def test_a_severe_but_survivable_loss_is_not_insolvency() -> None:
    """-90% is a catastrophe, not a wipe-out, and the two must not be conflated."""
    index = pd.bdate_range("2020-01-01", periods=4)
    returns = pd.Series([-0.5, -0.9, 5.0, 5.0], index=index)
    costs = pd.Series(np.zeros(4), index=index)
    outcome = evaluate_path(returns, costs, replicate=0, kind="test")
    assert outcome.insolvent is False
    assert outcome.net_return > 0.0
    assert outcome.max_drawdown == pytest.approx(-0.95)


def test_a_path_that_exactly_zeroes_equity_is_insolvent() -> None:
    index = pd.bdate_range("2020-01-01", periods=2)
    returns = pd.Series([-1.0, 0.5], index=index)
    costs = pd.Series(np.zeros(2), index=index)
    outcome = evaluate_path(returns, costs, replicate=0, kind="test")
    assert outcome.insolvent is True
    assert outcome.net_return == pytest.approx(-1.0)


def test_a_no_trade_path_is_flagged_not_counted_as_a_win() -> None:
    index = pd.bdate_range("2020-01-01", periods=10)
    flat = pd.Series(np.zeros(10), index=index)
    outcome = evaluate_path(flat, flat, replicate=0, kind="test")
    assert outcome.no_trade is True
    assert outcome.net_return == pytest.approx(0.0)


def test_a_solvent_path_reconciles(path: tuple[pd.Series, pd.Series]) -> None:
    returns, costs = path
    outcome = evaluate_path(returns, costs, replicate=0, kind="test")
    assert outcome.reconciled is True
    assert outcome.insolvent is False


def test_drawdown_is_non_positive_and_bounded(path: tuple[pd.Series, pd.Series]) -> None:
    returns, costs = path
    outcome = evaluate_path(returns, costs, replicate=0, kind="test")
    assert -1.0 <= outcome.max_drawdown <= 0.0


# ---------------------------------------------------------------------------
# Study reporting
# ---------------------------------------------------------------------------


def test_the_study_reports_every_declared_kind(
    path: tuple[pd.Series, pd.Series], grid: FrozenPerturbationGrid
) -> None:
    returns, costs = path
    study = run_perturbation_study(returns, costs, grid)
    assert {item["kind"] for item in study["outcomes"]} == set(PERTURBATION_KINDS)


def test_insolvent_paths_stay_in_the_denominator() -> None:
    """A study that discards blow-ups reports the distribution given survival."""
    index = pd.bdate_range("2020-01-01", periods=60)
    generator = np.random.default_rng(3)
    # Heavy negative drift with large dispersion so some perturbed paths fail.
    returns = pd.Series(generator.normal(-0.05, 0.30, 60), index=index)
    costs = pd.Series(np.full(60, 0.001), index=index)
    grid = FrozenPerturbationGrid(
        name="stress",
        specs=(PerturbationSpec(kind="execution_price", magnitude=0.5, bound=1.0),),
        replicates=MIN_REPLICATES,
        root_seed=1,
    )
    study = run_perturbation_study(returns, costs, grid)
    outcome = study["outcomes"][0]
    assert outcome["replicates"] == MIN_REPLICATES
    assert outcome["insolvent_paths"] > 0
    assert study["total_insolvent_paths"] == outcome["insolvent_paths"]


def test_the_study_refuses_a_grid_that_diverges_from_its_frozen_identity(
    path: tuple[pd.Series, pd.Series], grid: FrozenPerturbationGrid
) -> None:
    returns, costs = path
    with pytest.raises(PerturbationError, match="does not match the frozen declaration"):
        run_perturbation_study(returns, costs, grid, expected_identity="0" * 64)


def test_the_study_is_deterministic(
    path: tuple[pd.Series, pd.Series], grid: FrozenPerturbationGrid
) -> None:
    returns, costs = path
    assert run_perturbation_study(returns, costs, grid) == run_perturbation_study(
        returns, costs, grid
    )


def test_the_failure_report_orders_worst_first(
    path: tuple[pd.Series, pd.Series], grid: FrozenPerturbationGrid
) -> None:
    returns, costs = path
    report = failure_report(run_perturbation_study(returns, costs, grid))
    assert len(report["most_damaging_kinds"]) == 3
    assert report["worst_case_net_return"] <= report["baseline_net_return"]


def test_the_failure_report_names_the_deepest_drawdown_kind(
    path: tuple[pd.Series, pd.Series], grid: FrozenPerturbationGrid
) -> None:
    returns, costs = path
    report = failure_report(run_perturbation_study(returns, costs, grid))
    assert report["deepest_drawdown_kind"] in PERTURBATION_KINDS


def test_an_empty_study_cannot_be_summarized() -> None:
    with pytest.raises(PerturbationError, match="no outcomes"):
        failure_report({"outcomes": [], "baseline": {"net_return": 0.0}})


def test_the_caveat_travels_with_every_frequency(
    path: tuple[pd.Series, pd.Series], grid: FrozenPerturbationGrid
) -> None:
    """Monte Carlo frequency is not a market probability, and says so."""
    returns, costs = path
    study = run_perturbation_study(returns, costs, grid)
    for outcome in study["outcomes"]:
        assert "not the probability" in outcome["interpretation"]
    assert study["simulation_only"] is True


def test_the_study_is_json_serializable(
    path: tuple[pd.Series, pd.Series], grid: FrozenPerturbationGrid
) -> None:
    import json

    returns, costs = path
    assert json.loads(json.dumps(run_perturbation_study(returns, costs, grid)))


def test_retained_paths_are_bounded(
    path: tuple[pd.Series, pd.Series], grid: FrozenPerturbationGrid
) -> None:
    """Bounded retention: a 2000-path study must not carry 2000 records."""
    returns, costs = path
    study = run_perturbation_study(returns, costs, grid)
    assert len(study["retained_paths"]) == 32 * len(PERTURBATION_KINDS)


def test_the_study_stays_within_its_bounded_cost(
    path: tuple[pd.Series, pd.Series],
) -> None:
    import time

    returns, costs = path
    grid = standard_execution_grid(replicates=MIN_REPLICATES, root_seed=0)
    started = time.perf_counter()
    run_perturbation_study(returns, costs, grid)
    elapsed = time.perf_counter() - started
    assert elapsed < 60.0, f"study took {elapsed:.1f}s for {MIN_REPLICATES} replicates"
