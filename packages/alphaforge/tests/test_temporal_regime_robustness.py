"""Tests for temporal, regime, sector, and universe robustness (SF-S4-MR7).

Grouped by acceptance criterion. The invariants that carry the most weight:

* **A regime is never defined from the outcomes it explains** — labels are
  causal, cut points come from strictly-prior observations, and a conditioning
  series that is really the candidate's own P&L is refused.
* **Universe membership is point-in-time** — a later constituent list cannot
  backfill an earlier session, and delistings survive the round trip.
* **Uncertainty accounts for temporal dependence** — on autocorrelated returns
  the block-bootstrap and HAC intervals are materially wider than the i.i.d.
  one, which is the whole reason they exist.
* **No portfolio-level claim without a qualified candidate** — the refusal is
  structural, not a matter of remembering.
* **Removing winners, delistings, empty sectors, sparse regimes, and boundary
  observations cannot corrupt accounting**.
"""

from __future__ import annotations

from datetime import date, datetime
from typing import Any, cast

import numpy as np
import pandas as pd
import pytest

from alphaforge.robustness import (
    MIN_BOOTSTRAP_OBSERVATIONS,
    FrozenPeriodSet,
    MembershipRecord,
    PeriodContractError,
    PeriodInterval,
    PointInTimeUniverse,
    QualifiedCandidate,
    RegimeDefinition,
    TemporalEvidenceError,
    UniverseContractError,
    UnqualifiedCandidateError,
    apply_liquidity_floor,
    assert_conditioning_is_independent,
    assert_no_future_membership,
    block_bootstrap_interval,
    calendar_years,
    compare_uncertainty,
    concentration_profile,
    coverage_report,
    drop_sector,
    drop_top_contributors,
    exclude_inactive,
    label_regimes,
    matched_family_evidence,
    naive_interval,
    newey_west_interval,
    newey_west_standard_error,
    period_evidence,
    portfolio_dependence_evidence,
    regime_definition_sensitivity,
    regime_evidence,
    standard_regime_definitions,
    temporal_robustness_report,
    verify_frozen_periods,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _ar1(n: int, rho: float, *, seed: int = 11, scale: float = 0.01) -> pd.Series:
    """Return an AR(1) series with a known persistence."""
    generator = np.random.default_rng(seed)
    shocks = generator.normal(0.0, scale, n)
    values = np.zeros(n, dtype=float)
    for position in range(1, n):
        values[position] = rho * values[position - 1] + shocks[position]
    return pd.Series(values, index=pd.bdate_range("2018-01-01", periods=n))


@pytest.fixture
def returns() -> pd.Series:
    return _ar1(1_200, 0.0, seed=5)


@pytest.fixture
def conditioning() -> pd.Series:
    generator = np.random.default_rng(23)
    return pd.Series(
        generator.normal(0.0, 0.012, 1_200), index=pd.bdate_range("2018-01-01", periods=1_200)
    )


@pytest.fixture
def two_regime() -> RegimeDefinition:
    return RegimeDefinition(
        name="volatility_two_state",
        statistic="volatility",
        window=60,
        thresholds=(0.5,),
        labels=("calm", "stormy"),
    )


@pytest.fixture
def universe() -> PointInTimeUniverse:
    return PointInTimeUniverse(
        name="test_universe",
        records=(
            MembershipRecord("AAA", date(2018, 1, 1), sector="tech"),
            MembershipRecord(
                "BBB", date(2018, 1, 1), date(2020, 6, 1), sector="tech", delisted=True
            ),
            MembershipRecord("CCC", date(2019, 3, 1), sector="energy"),
            MembershipRecord("DDD", date(2018, 1, 1), date(2021, 1, 1), sector="energy"),
        ),
    )


# ---------------------------------------------------------------------------
# Frozen calendar intervals
# ---------------------------------------------------------------------------


def test_calendar_years_tile_without_gaps_or_overlaps() -> None:
    period_set = calendar_years(2018, 2022)
    assert len(period_set.intervals) == 5
    for earlier, later in zip(period_set.intervals, period_set.intervals[1:], strict=False):
        assert earlier.end == later.start


def test_a_boundary_observation_joins_exactly_one_interval() -> None:
    period_set = calendar_years(2019, 2020)
    index = pd.to_datetime(["2019-12-31", "2020-01-01", "2020-12-31", "2021-01-01"])
    labels = period_set.assign(index)
    assert list(labels) == ["2019", "2020", "2020", None]


def test_an_observation_outside_every_interval_is_unlabelled_not_absorbed() -> None:
    period_set = calendar_years(2020, 2020)
    labels = period_set.assign(pd.to_datetime(["2019-06-01", "2020-06-01", "2021-06-01"]))
    assert list(labels) == [None, "2020", None]


def test_accidental_overlap_is_refused() -> None:
    with pytest.raises(PeriodContractError, match="overlap"):
        FrozenPeriodSet(
            name="overlapping",
            intervals=(
                PeriodInterval("a", date(2020, 1, 1), date(2020, 7, 1)),
                PeriodInterval("b", date(2020, 6, 1), date(2021, 1, 1)),
            ),
        )


def test_a_deliberate_overlay_is_permitted_when_declared() -> None:
    period_set = FrozenPeriodSet(
        name="overlay",
        intervals=(
            PeriodInterval("year", date(2020, 1, 1), date(2021, 1, 1)),
            PeriodInterval("stress", date(2020, 3, 1), date(2020, 5, 1), kind="overlay"),
        ),
        allow_overlap=True,
    )
    assert len(period_set.intervals) == 2


def test_an_empty_or_inverted_interval_is_refused() -> None:
    with pytest.raises(PeriodContractError, match="empty or inverted"):
        PeriodInterval("bad", date(2020, 6, 1), date(2020, 6, 1))


def test_duplicate_interval_labels_are_refused() -> None:
    with pytest.raises(PeriodContractError, match="unique"):
        FrozenPeriodSet(
            name="dupes",
            intervals=(
                PeriodInterval("x", date(2020, 1, 1), date(2020, 6, 1)),
                PeriodInterval("x", date(2020, 6, 1), date(2021, 1, 1)),
            ),
        )


def test_recutting_the_calendar_after_the_fact_is_detected() -> None:
    frozen = calendar_years(2018, 2022)
    identity = frozen.identity
    # A "crisis-excluding" recut of the same span.
    recut = FrozenPeriodSet(
        name="calendar_years",
        intervals=tuple(interval for interval in frozen.intervals if interval.label != "2020"),
    )
    verify_frozen_periods(frozen, identity)
    with pytest.raises(PeriodContractError, match="does not match the frozen declaration"):
        verify_frozen_periods(recut, identity)


def test_period_identity_is_stable_across_construction_order() -> None:
    intervals = calendar_years(2018, 2021).intervals
    forward = FrozenPeriodSet(name="calendar_years", intervals=intervals)
    reverse = FrozenPeriodSet(name="calendar_years", intervals=tuple(reversed(intervals)))
    assert forward.identity == reverse.identity


# ---------------------------------------------------------------------------
# Causal regime labelling
# ---------------------------------------------------------------------------


def test_regime_labels_do_not_depend_on_future_observations(
    conditioning: pd.Series, two_regime: RegimeDefinition
) -> None:
    """Labels for a prefix must not change when the future is appended.

    This is the sharpest available test of causality: if any label moved, some
    part of the labelling read data that had not happened yet.
    """
    cut = 800
    full = label_regimes(conditioning, two_regime)
    prefix = label_regimes(conditioning.iloc[:cut], two_regime)
    pd.testing.assert_series_equal(full.iloc[:cut], prefix)


def test_a_full_sample_quantile_would_have_broken_that_invariant(
    conditioning: pd.Series, two_regime: RegimeDefinition
) -> None:
    """Guard the reason expanding cut points are used rather than full-sample ones.

    A full-sample threshold looks causal — the statistic is trailing — but the
    cut point knows how the series ends, so prefix labels shift when more data
    arrives. Demonstrating that shift here keeps the expanding-quantile choice
    from being 'simplified' away later.
    """
    cut = 800
    values = conditioning.astype(float)
    statistic = values.rolling(two_regime.window, min_periods=two_regime.window).std()

    def full_sample_labels(series: pd.Series) -> list[str | None]:
        threshold = series.quantile(0.5)
        return [
            None if not np.isfinite(item) else ("calm" if item <= threshold else "stormy")
            for item in series
        ]

    whole = full_sample_labels(statistic)[:cut]
    prefix = full_sample_labels(statistic.iloc[:cut])
    assert whole != prefix


def test_warmup_bars_carry_no_label_rather_than_a_guessed_one(
    conditioning: pd.Series, two_regime: RegimeDefinition
) -> None:
    labels = label_regimes(conditioning, two_regime)
    assert labels.iloc[: two_regime.window].isna().all()
    assert labels.notna().any()


def test_a_warmup_policy_other_than_unknown_is_refused() -> None:
    with pytest.raises(PeriodContractError, match="warmup_policy"):
        RegimeDefinition(
            name="guessing",
            statistic="volatility",
            window=20,
            thresholds=(0.5,),
            labels=("low", "high"),
            warmup_policy="forward_fill",
        )


def test_a_regime_defined_from_the_candidates_own_returns_is_refused(
    returns: pd.Series,
) -> None:
    with pytest.raises(PeriodContractError, match="circular"):
        assert_conditioning_is_independent(returns, returns)


def test_a_near_duplicate_of_the_candidate_is_also_refused(returns: pd.Series) -> None:
    disguised = returns * 3.0 + 0.0001
    with pytest.raises(PeriodContractError, match="circular"):
        assert_conditioning_is_independent(disguised, returns)


def test_an_independent_conditioning_series_passes(
    returns: pd.Series, conditioning: pd.Series
) -> None:
    assert_conditioning_is_independent(conditioning, returns)


def test_label_count_must_match_the_threshold_count() -> None:
    with pytest.raises(PeriodContractError, match="labels were supplied"):
        RegimeDefinition(
            name="mismatched",
            statistic="volatility",
            window=20,
            thresholds=(0.33, 0.66),
            labels=("low", "high"),
        )


def test_unsorted_conditioning_is_refused(two_regime: RegimeDefinition) -> None:
    series = pd.Series(
        [1.0, 2.0, 3.0], index=pd.to_datetime(["2020-03-01", "2020-01-01", "2020-02-01"])
    )
    with pytest.raises(PeriodContractError, match="sorted ascending"):
        label_regimes(series, two_regime)


def test_sparse_periods_are_reported_with_counts_not_dropped() -> None:
    labels = pd.Series(["calm"] * 100 + ["stormy"] * 3, dtype="object")
    report = coverage_report(labels)
    assert report["counts"] == {"calm": 100, "stormy": 3}
    assert report["sparse"] == ["stormy"]


# ---------------------------------------------------------------------------
# Point-in-time universe
# ---------------------------------------------------------------------------


def test_membership_resolves_as_of_a_session(universe: PointInTimeUniverse) -> None:
    assert universe.members(date(2018, 6, 1)) == ("AAA", "BBB", "DDD")
    assert universe.members(date(2019, 6, 1)) == ("AAA", "BBB", "CCC", "DDD")
    assert universe.members(date(2020, 7, 1)) == ("AAA", "CCC", "DDD")
    assert universe.members(date(2021, 6, 1)) == ("AAA", "CCC")


def test_a_symbol_is_not_investable_before_it_lists(universe: PointInTimeUniverse) -> None:
    assert "CCC" not in universe.members(date(2019, 2, 28))
    assert "CCC" in universe.members(date(2019, 3, 1))


def test_the_expiry_session_is_exclusive(universe: PointInTimeUniverse) -> None:
    assert "BBB" in universe.members(date(2020, 5, 31))
    assert "BBB" not in universe.members(date(2020, 6, 1))


def test_a_later_constituent_list_cannot_backfill_an_earlier_session(
    universe: PointInTimeUniverse,
) -> None:
    """The survivorship check: today's names must not appear in 2018's panel."""
    todays_names = ["AAA", "BBB", "CCC", "DDD"]
    with pytest.raises(UniverseContractError, match="backfills the universe"):
        assert_no_future_membership(universe, todays_names, date(2018, 6, 1))
    assert_no_future_membership(universe, ["AAA", "BBB", "DDD"], date(2018, 6, 1))


def test_delistings_survive_the_record(universe: PointInTimeUniverse) -> None:
    assert universe.delistings() == (("BBB", date(2020, 6, 1)),)
    assert universe.to_dict()["delistings"] == [{"symbol": "BBB", "session": "2020-06-01"}]


def test_a_delisting_without_an_exit_session_is_refused() -> None:
    with pytest.raises(UniverseContractError, match="session it left"):
        MembershipRecord("XXX", date(2018, 1, 1), delisted=True)


def test_overlapping_membership_for_one_symbol_is_refused() -> None:
    with pytest.raises(UniverseContractError, match="member twice"):
        PointInTimeUniverse(
            name="overlapping",
            records=(
                MembershipRecord("AAA", date(2018, 1, 1), date(2020, 1, 1)),
                MembershipRecord("AAA", date(2019, 1, 1)),
            ),
        )


def test_a_symbol_may_rejoin_after_it_leaves() -> None:
    rejoined = PointInTimeUniverse(
        name="rejoin",
        records=(
            MembershipRecord("AAA", date(2018, 1, 1), date(2019, 1, 1)),
            MembershipRecord("AAA", date(2020, 1, 1)),
        ),
    )
    assert rejoined.members(date(2018, 6, 1)) == ("AAA",)
    assert rejoined.members(date(2019, 6, 1)) == ()
    assert rejoined.members(date(2020, 6, 1)) == ("AAA",)


def test_an_inverted_membership_interval_is_refused() -> None:
    with pytest.raises(UniverseContractError, match="must follow"):
        MembershipRecord("AAA", date(2020, 1, 1), date(2019, 1, 1))


def test_universe_identity_is_order_independent(universe: PointInTimeUniverse) -> None:
    shuffled = PointInTimeUniverse(name=universe.name, records=tuple(reversed(universe.records)))
    assert shuffled.identity == universe.identity


# ---------------------------------------------------------------------------
# Ablations
# ---------------------------------------------------------------------------


def test_dropping_top_contributors_is_marked_hindsight_based() -> None:
    contributions = {"AAA": 0.10, "BBB": 0.04, "CCC": 0.01, "DDD": -0.02}
    result = drop_top_contributors(contributions, count=2, baseline_metric=0.13)
    assert result.hindsight_based is True
    assert result.removed == ("AAA", "BBB")
    assert result.ablated_metric == pytest.approx(-0.01)
    assert "NOT an achievable return" in result.interpretation


def test_ablation_accounting_reconciles_exactly() -> None:
    """Removed plus retained must equal the whole, with no leakage either way."""
    contributions = {"AAA": 0.10, "BBB": 0.04, "CCC": 0.01, "DDD": -0.02}
    total = sum(contributions.values())
    result = drop_top_contributors(contributions, count=1, baseline_metric=total)
    removed_total = sum(contributions[symbol] for symbol in result.removed)
    assert result.ablated_metric + removed_total == pytest.approx(total)
    assert result.retained_count + len(result.removed) == len(contributions)


def test_top_contributor_ties_break_deterministically() -> None:
    contributions = {"BBB": 0.05, "AAA": 0.05, "CCC": 0.01}
    first = drop_top_contributors(contributions, count=1, baseline_metric=0.11)
    second = drop_top_contributors(
        dict(reversed(list(contributions.items()))), count=1, baseline_metric=0.11
    )
    assert first.removed == second.removed == ("AAA",)


def test_removing_every_contributor_is_refused() -> None:
    with pytest.raises(UniverseContractError, match="nothing to measure"):
        drop_top_contributors({"AAA": 1.0, "BBB": 2.0}, count=2, baseline_metric=3.0)


def test_dropping_a_sector_is_not_hindsight_based(universe: PointInTimeUniverse) -> None:
    contributions = {"AAA": 0.10, "CCC": 0.04, "DDD": -0.01}
    result = drop_sector(
        contributions, universe, sector="energy", session=date(2020, 1, 1), baseline_metric=0.13
    )
    assert result.hindsight_based is False
    assert result.removed == ("CCC", "DDD")
    assert result.ablated_metric == pytest.approx(0.10)


def test_dropping_an_empty_sector_leaves_the_total_unchanged(
    universe: PointInTimeUniverse,
) -> None:
    """A sector with no members on a session is a legitimate, non-fatal outcome."""
    contributions = {"AAA": 0.10, "CCC": 0.04}
    result = drop_sector(
        contributions,
        universe,
        sector="utilities",
        session=date(2020, 1, 1),
        baseline_metric=0.14,
    )
    assert result.removed == ()
    assert result.ablated_metric == pytest.approx(0.14)
    assert result.delta == pytest.approx(0.0)


def test_a_name_with_unknown_liquidity_is_removed_not_retained() -> None:
    contributions = {"AAA": 0.10, "BBB": 0.05, "CCC": 0.01}
    liquidity = {"AAA": 5_000_000.0, "BBB": 1_000.0}
    result = apply_liquidity_floor(
        contributions, liquidity, minimum=100_000.0, baseline_metric=0.16
    )
    assert result.removed == ("BBB", "CCC")
    assert result.ablated_metric == pytest.approx(0.10)


def test_excluding_inactive_names_removes_contributions_never_earnable(
    universe: PointInTimeUniverse,
) -> None:
    contributions = {"AAA": 0.10, "BBB": 0.05, "CCC": 0.01}
    result = exclude_inactive(
        contributions, universe, session=date(2018, 6, 1), baseline_metric=0.16
    )
    assert result.removed == ("CCC",)  # CCC had not listed yet
    assert result.ablated_metric == pytest.approx(0.15)


def test_a_delisted_name_is_excluded_after_its_exit(universe: PointInTimeUniverse) -> None:
    contributions = {"AAA": 0.10, "BBB": 0.05}
    result = exclude_inactive(
        contributions, universe, session=date(2020, 7, 1), baseline_metric=0.15
    )
    assert result.removed == ("BBB",)


def test_concentration_reports_how_few_names_carry_the_result() -> None:
    concentrated = concentration_profile({"AAA": 0.90, "BBB": 0.05, "CCC": 0.05})
    spread = concentration_profile({f"S{index:02d}": 0.05 for index in range(20)})
    assert concentrated["top_shares"]["top_1"] > 0.85
    assert spread["top_shares"]["top_1"] == pytest.approx(0.05)
    assert concentrated["herfindahl"] > spread["herfindahl"]


def test_concentration_is_undefined_rather_than_fabricated_at_zero() -> None:
    profile = concentration_profile({"AAA": 0.0, "BBB": 0.0})
    assert np.isnan(profile["herfindahl"])
    assert profile["top_shares"] == {}


# ---------------------------------------------------------------------------
# Dependence-aware uncertainty
# ---------------------------------------------------------------------------


def test_dependence_aware_intervals_are_wider_on_autocorrelated_returns() -> None:
    """The reason this module exists: the i.i.d. interval is too narrow."""
    persistent = _ar1(1_000, 0.85, seed=17)
    comparison = compare_uncertainty(persistent, block_length=40, seed=3)
    assert comparison["width_inflation"]["newey_west_over_iid"] > 1.5
    assert comparison["width_inflation"]["block_over_iid"] > 1.5


def test_the_intervals_agree_when_returns_are_actually_independent() -> None:
    """No dependence to account for means no material widening."""
    independent = _ar1(1_500, 0.0, seed=29)
    comparison = compare_uncertainty(independent, block_length=20, seed=3)
    assert 0.7 < comparison["width_inflation"]["newey_west_over_iid"] < 1.4
    assert 0.7 < comparison["width_inflation"]["block_over_iid"] < 1.4


def test_the_hac_error_grows_with_persistence() -> None:
    mild = newey_west_standard_error(_ar1(1_000, 0.1, seed=41), lag=20)
    strong = newey_west_standard_error(_ar1(1_000, 0.9, seed=41), lag=20)
    assert strong > mild


def test_the_hac_error_is_never_negative_under_bartlett_weights() -> None:
    alternating = pd.Series(
        [0.01 if index % 2 == 0 else -0.01 for index in range(200)],
        index=pd.bdate_range("2020-01-01", periods=200),
    )
    assert newey_west_standard_error(alternating, lag=10) >= 0.0


def test_the_hac_error_reduces_to_the_iid_error_at_zero_lag(returns: pd.Series) -> None:
    hac = newey_west_standard_error(returns, lag=0)
    values = returns.to_numpy(dtype=float)
    population = float(values.std(ddof=0) / np.sqrt(values.size))
    assert hac == pytest.approx(population)


def test_the_block_bootstrap_is_reproducible(returns: pd.Series) -> None:
    first = block_bootstrap_interval(returns, block_length=20, replicates=400, seed=9)
    second = block_bootstrap_interval(returns, block_length=20, replicates=400, seed=9)
    assert first.to_dict() == second.to_dict()


def test_a_different_seed_moves_the_interval_but_not_the_point(returns: pd.Series) -> None:
    first = block_bootstrap_interval(returns, block_length=20, replicates=400, seed=1)
    second = block_bootstrap_interval(returns, block_length=20, replicates=400, seed=2)
    assert first.point == pytest.approx(second.point)
    assert first.lower != second.lower


def test_a_sample_too_short_for_a_dependence_aware_interval_is_refused() -> None:
    short = pd.Series(
        np.zeros(MIN_BOOTSTRAP_OBSERVATIONS - 1),
        index=pd.bdate_range("2020-01-01", periods=MIN_BOOTSTRAP_OBSERVATIONS - 1),
    )
    with pytest.raises(TemporalEvidenceError, match="at least"):
        block_bootstrap_interval(short, block_length=5)


def test_a_block_longer_than_the_sample_is_refused(returns: pd.Series) -> None:
    with pytest.raises(TemporalEvidenceError, match="cannot exceed the sample length"):
        block_bootstrap_interval(returns.iloc[:100], block_length=200)


def test_every_interval_carries_the_assumption_that_produced_it(returns: pd.Series) -> None:
    for interval in (
        naive_interval(returns),
        newey_west_interval(returns),
        block_bootstrap_interval(returns, block_length=20, replicates=200, seed=1),
    ):
        assert interval.assumption
        assert interval.lower <= interval.point <= interval.upper


def test_non_finite_returns_are_refused() -> None:
    values = pd.Series([0.01, np.inf, 0.02], index=pd.bdate_range("2020-01-01", periods=3))
    with pytest.raises(TemporalEvidenceError, match="infinite"):
        naive_interval(values)


def test_a_duplicated_timestamp_is_refused() -> None:
    index = pd.to_datetime(["2020-01-01", "2020-01-01", "2020-01-02"])
    with pytest.raises(TemporalEvidenceError, match="unique"):
        naive_interval(pd.Series([0.01, 0.02, 0.03], index=index))


# ---------------------------------------------------------------------------
# Period and regime evidence
# ---------------------------------------------------------------------------


def test_every_declared_period_is_reported_including_losing_ones() -> None:
    index = pd.bdate_range("2019-01-01", "2020-12-31")
    values = np.where(index.year == 2019, 0.001, -0.001)
    losing_year = pd.Series(values, index=index)
    report = period_evidence(losing_year, calendar_years(2019, 2020), block_length=20)
    labels = [outcome["label"] for outcome in report["outcomes"]]
    assert labels == ["2019", "2020"]
    assert report["failure_periods"] == ["2020"]


def test_a_period_with_no_observations_is_reported_as_empty_not_omitted(
    returns: pd.Series,
) -> None:
    report = period_evidence(returns, calendar_years(2016, 2019), block_length=20)
    empty = [outcome for outcome in report["outcomes"] if outcome["label"] == "2016"]
    assert len(empty) == 1
    assert empty[0]["n_observations"] == 0
    assert empty[0]["interval"] is None
    assert "reported as empty" in empty[0]["note"]


def test_a_sparse_period_keeps_its_count_and_declines_an_interval() -> None:
    index = pd.bdate_range("2019-12-20", "2020-12-31")
    series = pd.Series(np.full(len(index), 0.001), index=index)
    report = period_evidence(series, calendar_years(2019, 2020), block_length=20)
    sparse = [outcome for outcome in report["outcomes"] if outcome["label"] == "2019"][0]
    assert 0 < sparse["n_observations"] < MIN_BOOTSTRAP_OBSERVATIONS
    assert sparse["sparse"] is True
    assert sparse["interval"] is None
    assert "misleadingly tight" in sparse["note"]


def test_period_accounting_partitions_every_observation(returns: pd.Series) -> None:
    """Totals across intervals plus unlabelled must equal the whole sample."""
    period_set = calendar_years(2018, 2023)
    report = period_evidence(returns, period_set, block_length=20)
    counted = sum(outcome["n_observations"] for outcome in report["outcomes"])
    assert counted + report["unlabelled_observations"] == len(returns)


def test_a_boundary_observation_is_counted_exactly_once() -> None:
    index = pd.to_datetime(["2019-12-31", "2020-01-01"])
    series = pd.Series([0.01, 0.02], index=index)
    report = period_evidence(series, calendar_years(2019, 2020), block_length=20)
    per_label = {outcome["label"]: outcome["n_observations"] for outcome in report["outcomes"]}
    assert per_label == {"2019": 1, "2020": 1}


def test_regime_evidence_reports_every_declared_regime(
    returns: pd.Series, conditioning: pd.Series, two_regime: RegimeDefinition
) -> None:
    report = regime_evidence(returns, conditioning, two_regime, block_length=20)
    labels = [outcome["label"] for outcome in report["outcomes"]]
    assert labels == ["calm", "stormy"]
    assert report["unlabelled_observations"] > 0


def test_an_empty_regime_is_reported_rather_than_crashing(
    returns: pd.Series, conditioning: pd.Series
) -> None:
    """An extreme threshold can leave a regime with no members; that is data, not an error."""
    lopsided = RegimeDefinition(
        name="rare_tail",
        statistic="volatility",
        window=60,
        thresholds=(0.001, 0.999),
        labels=("floor", "middle", "ceiling"),
    )
    report = regime_evidence(returns, conditioning, lopsided, block_length=20)
    counts = {outcome["label"]: outcome["n_observations"] for outcome in report["outcomes"]}
    assert set(counts) == {"floor", "middle", "ceiling"}
    assert min(counts.values()) >= 0


def test_regime_evidence_refuses_a_disjoint_conditioning_series(
    returns: pd.Series, two_regime: RegimeDefinition
) -> None:
    elsewhere = pd.Series(np.zeros(300), index=pd.bdate_range("2050-01-01", periods=300))
    with pytest.raises(TemporalEvidenceError, match="share no timestamps"):
        regime_evidence(returns, elsewhere, two_regime)


def test_regime_conclusions_are_reported_across_alternative_definitions(
    returns: pd.Series, conditioning: pd.Series, two_regime: RegimeDefinition
) -> None:
    alternative = RegimeDefinition(
        name="volatility_wide_window",
        statistic="volatility",
        window=120,
        thresholds=(0.5,),
        labels=("calm", "stormy"),
    )
    report = regime_definition_sensitivity(
        returns, conditioning, [two_regime, alternative], block_length=20
    )
    assert len(report["reports"]) == 2
    assert isinstance(report["conclusion_is_definition_dependent"], bool)


def test_duplicate_regime_definitions_are_refused(
    returns: pd.Series, conditioning: pd.Series, two_regime: RegimeDefinition
) -> None:
    with pytest.raises(TemporalEvidenceError, match="unique"):
        regime_definition_sensitivity(returns, conditioning, [two_regime, two_regime])


# ---------------------------------------------------------------------------
# The portfolio-level gate
# ---------------------------------------------------------------------------


def test_no_portfolio_claim_is_made_without_a_qualified_candidate(
    returns: pd.Series, universe: PointInTimeUniverse
) -> None:
    with pytest.raises(UnqualifiedCandidateError, match="no portfolio-level claim"):
        portfolio_dependence_evidence(returns, {"AAA": 0.1}, universe, (), qualified=None)


def test_a_qualified_candidate_unlocks_the_portfolio_report(
    returns: pd.Series, universe: PointInTimeUniverse
) -> None:
    qualified = QualifiedCandidate(
        candidate_id="cand-1",
        decision_id="SF-S4-MR9-decision",
        plan_hash="a" * 64,
        adjusted_p_value=0.004,
        alpha=0.05,
    )
    contributions = {"AAA": 0.10, "BBB": 0.03, "CCC": -0.01}
    ablation = drop_top_contributors(contributions, count=1, baseline_metric=0.12)
    report = portfolio_dependence_evidence(
        returns, contributions, universe, (ablation,), qualified=qualified, block_length=20
    )
    assert report["candidate"]["candidate_id"] == "cand-1"
    assert report["hindsight_based_ablations"] == ["drop_top_1_contributors"]
    assert report["simulation_only"] is True


def test_a_candidate_above_alpha_cannot_be_declared_qualified() -> None:
    with pytest.raises(UnqualifiedCandidateError, match="did not clear qualification"):
        QualifiedCandidate(
            candidate_id="cand-2",
            decision_id="decision",
            plan_hash="b" * 64,
            adjusted_p_value=0.20,
            alpha=0.05,
        )


def test_qualification_requires_a_real_plan_hash() -> None:
    with pytest.raises(UnqualifiedCandidateError, match="plan_hash"):
        QualifiedCandidate(
            candidate_id="cand-3",
            decision_id="decision",
            plan_hash="short",
            adjusted_p_value=0.01,
            alpha=0.05,
        )


def test_the_candidate_level_report_withholds_the_portfolio_claim(
    returns: pd.Series, conditioning: pd.Series, two_regime: RegimeDefinition
) -> None:
    period_set = calendar_years(2018, 2022)
    report = temporal_robustness_report(
        returns,
        period_set,
        conditioning,
        [two_regime],
        expected_period_identity=period_set.identity,
        block_length=20,
    )
    assert "withheld" in report["portfolio_claim"]
    assert report["simulation_only"] is True


def test_the_report_refuses_a_period_set_that_was_recut(
    returns: pd.Series, conditioning: pd.Series, two_regime: RegimeDefinition
) -> None:
    period_set = calendar_years(2018, 2022)
    with pytest.raises(PeriodContractError, match="does not match the frozen declaration"):
        temporal_robustness_report(
            returns,
            period_set,
            conditioning,
            [two_regime],
            expected_period_identity="0" * 64,
            block_length=20,
        )


def test_the_full_report_is_deterministic(
    returns: pd.Series, conditioning: pd.Series, two_regime: RegimeDefinition
) -> None:
    period_set = calendar_years(2018, 2022)

    def build() -> dict[str, Any]:
        return temporal_robustness_report(
            returns,
            period_set,
            conditioning,
            [two_regime],
            expected_period_identity=period_set.identity,
            block_length=20,
            seed=4,
        )

    assert build() == build()


def test_period_and_regime_records_are_json_serializable(
    returns: pd.Series, conditioning: pd.Series, two_regime: RegimeDefinition
) -> None:
    import json

    period_set = calendar_years(2018, 2022)
    report = temporal_robustness_report(
        returns,
        period_set,
        conditioning,
        [two_regime],
        expected_period_identity=period_set.identity,
        block_length=20,
    )
    assert json.loads(json.dumps(report))


def test_a_timestamp_is_not_accepted_where_a_session_date_is_required() -> None:
    with pytest.raises(UniverseContractError, match="datetime.date"):
        MembershipRecord("AAA", datetime(2018, 1, 1, 9, 30))  # noqa: DTZ001


# ---------------------------------------------------------------------------
# Matched per-family evidence
# ---------------------------------------------------------------------------


def test_families_are_evaluated_on_identical_windows() -> None:
    """Families evaluated over different spans compare the spans, not the families."""
    long_family = _ar1(600, 0.0, seed=61)
    short_family = long_family.iloc[100:400] + 0.0005
    report = matched_family_evidence(
        {"long": long_family, "short": short_family}, calendar_years(2018, 2020), block_length=20
    )
    assert report["matched_observations"] == 300
    per_family_counts = {
        name: [outcome["n_observations"] for outcome in family["outcomes"]]
        for name, family in report["families"].items()
    }
    assert per_family_counts["long"] == per_family_counts["short"]


def test_matching_reports_what_each_family_gave_up() -> None:
    long_family = _ar1(600, 0.0, seed=63)
    short_family = long_family.iloc[100:400]
    report = matched_family_evidence(
        {"long": long_family, "short": short_family}, calendar_years(2018, 2020), block_length=20
    )
    assert report["coverage_sacrificed"] == {"long": 300, "short": 0}


def test_a_family_with_gaps_matches_on_the_observations_it_actually_has() -> None:
    """Missing prices shrink the matched window rather than corrupting the counts."""
    complete = _ar1(400, 0.0, seed=67)
    gapped = complete.copy()
    gapped.iloc[50:100] = np.nan
    report = matched_family_evidence(
        {"complete": complete, "gapped": gapped}, calendar_years(2018, 2019), block_length=20
    )
    assert report["matched_observations"] == 350
    for family in report["families"].values():
        counted = sum(outcome["n_observations"] for outcome in family["outcomes"])
        assert counted + family["unlabelled_observations"] == 350


def test_a_single_family_cannot_be_matched() -> None:
    with pytest.raises(TemporalEvidenceError, match="at least two families"):
        matched_family_evidence({"only": _ar1(200, 0.0)}, calendar_years(2018, 2019))


def test_families_that_share_no_timestamps_are_refused() -> None:
    first = _ar1(200, 0.0, seed=71)
    second = pd.Series(np.zeros(200), index=pd.bdate_range("2050-01-01", periods=200))
    with pytest.raises(TemporalEvidenceError, match="share no timestamps"):
        matched_family_evidence({"first": first, "second": second}, calendar_years(2018, 2019))


# ---------------------------------------------------------------------------
# The standard regime set
# ---------------------------------------------------------------------------


def test_the_standard_set_covers_the_regimes_the_work_item_names() -> None:
    labels = {label for definition in standard_regime_definitions() for label in definition.labels}
    assert {"bull", "bear", "sideways", "crisis", "recovery"} <= labels
    assert {"high volatility", "low volatility"} <= labels


def test_every_standard_definition_is_causal(conditioning: pd.Series) -> None:
    """Prefix invariance must hold for each supplied definition, not just one."""
    cut = 700
    for definition in standard_regime_definitions():
        full = label_regimes(conditioning, definition)
        prefix = label_regimes(conditioning.iloc[:cut], definition)
        pd.testing.assert_series_equal(full.iloc[:cut], prefix)


def test_the_standard_definitions_have_distinct_identities() -> None:
    definitions = standard_regime_definitions()
    assert len({definition.identity for definition in definitions}) == len(definitions)


def test_the_standard_set_runs_end_to_end_as_a_sensitivity_sweep(
    returns: pd.Series, conditioning: pd.Series
) -> None:
    report = regime_definition_sensitivity(
        returns, conditioning, list(standard_regime_definitions()), block_length=20
    )
    assert len(report["reports"]) == 3


# ---------------------------------------------------------------------------
# Fault injection and bounded performance
# ---------------------------------------------------------------------------


def test_missing_prices_do_not_corrupt_period_accounting() -> None:
    index = pd.bdate_range("2019-01-01", "2020-12-31")
    series = pd.Series(np.full(len(index), 0.001), index=index)
    series.iloc[100:160] = np.nan
    report = period_evidence(series, calendar_years(2019, 2020), block_length=20)
    counted = sum(outcome["n_observations"] for outcome in report["outcomes"])
    assert counted == int(series.notna().sum())


def test_an_all_missing_period_is_reported_as_empty_not_as_a_zero_return() -> None:
    index = pd.bdate_range("2019-01-01", "2020-12-31")
    series = pd.Series(np.full(len(index), 0.001), index=index)
    series.loc[series.index.year == 2019] = np.nan
    report = period_evidence(series, calendar_years(2019, 2020), block_length=20)
    blank = [outcome for outcome in report["outcomes"] if outcome["label"] == "2019"][0]
    assert blank["n_observations"] == 0
    assert np.isnan(blank["mean_return"])


def test_the_bootstrap_stays_within_its_bounded_cost() -> None:
    """A bounded-performance guard: the resample is vectorized, not per-observation."""
    import time

    series = _ar1(2_000, 0.5, seed=83)
    started = time.perf_counter()
    block_bootstrap_interval(series, block_length=40, replicates=2_000, seed=1)
    elapsed = time.perf_counter() - started
    assert elapsed < 10.0, f"block bootstrap took {elapsed:.2f}s for 2000 replicates"


def test_replicate_and_lag_ceilings_are_enforced(returns: pd.Series) -> None:
    with pytest.raises(TemporalEvidenceError, match="ceiling"):
        block_bootstrap_interval(returns, block_length=20, replicates=1_000_000)
    with pytest.raises(TemporalEvidenceError, match="ceiling"):
        newey_west_standard_error(returns, lag=10_000)


# ---------------------------------------------------------------------------
# Contract refusals and documented abstentions
# ---------------------------------------------------------------------------


def test_the_independence_screen_abstains_when_it_cannot_judge(returns: pd.Series) -> None:
    """Documented abstention: too little overlap carries no evidence either way.

    Pinned deliberately. The screen catches the dynamically-built accident; the
    structural separation in ``label_regimes`` is what enforces the rule, so an
    abstention here is not a hole in the enforcement.
    """
    barely_overlapping = returns.iloc[:2]
    assert_conditioning_is_independent(barely_overlapping, returns)

    mostly_missing = returns.copy()
    mostly_missing.iloc[2:] = np.nan
    assert_conditioning_is_independent(mostly_missing, returns)


def test_a_constant_conditioning_series_is_not_treated_as_circular(
    returns: pd.Series,
) -> None:
    constant = pd.Series(np.full(len(returns), 0.01), index=returns.index)
    assert_conditioning_is_independent(constant, returns)


def test_an_out_of_range_correlation_ceiling_is_refused(returns: pd.Series) -> None:
    with pytest.raises(PeriodContractError, match="max_abs_correlation"):
        assert_conditioning_is_independent(returns, returns, max_abs_correlation=0.0)


def test_label_regimes_refuses_a_non_series_and_an_empty_series(
    two_regime: RegimeDefinition,
) -> None:
    with pytest.raises(PeriodContractError, match="pandas Series"):
        # cast rather than `type: ignore`: the violation is deliberate, and
        # whether a bare list needs an ignore here varies by pandas-stubs version.
        label_regimes(cast(pd.Series, [0.1, 0.2]), two_regime)
    with pytest.raises(PeriodContractError, match="non-empty"):
        label_regimes(pd.Series([], dtype=float), two_regime)


def test_calendar_years_refuses_a_non_integer_or_inverted_range() -> None:
    with pytest.raises(PeriodContractError, match="years must be ints"):
        calendar_years(cast(int, 2018.5), 2020)
    # bool is a subtype of int, so this needs no ignore — and the refusal is
    # exactly why the constructor checks for bool explicitly.
    with pytest.raises(PeriodContractError, match="years must be ints"):
        calendar_years(True, 2020)
    with pytest.raises(PeriodContractError, match="on or after"):
        calendar_years(2020, 2018)


def test_an_oversized_year_range_is_refused() -> None:
    with pytest.raises(PeriodContractError, match="ceiling"):
        calendar_years(1000, 2600)


def test_freeze_verification_requires_a_full_digest() -> None:
    with pytest.raises(PeriodContractError, match="full SHA-256"):
        verify_frozen_periods(calendar_years(2018, 2020), "not-a-digest")


def test_an_unsupported_regime_statistic_is_refused() -> None:
    with pytest.raises(PeriodContractError, match="unsupported regime statistic"):
        RegimeDefinition(
            name="bogus",
            statistic="astrology",
            window=20,
            thresholds=(0.5,),
            labels=("low", "high"),
        )


def test_thresholds_must_be_strictly_ascending_quantiles() -> None:
    with pytest.raises(PeriodContractError, match="ascending"):
        RegimeDefinition(
            name="unsorted",
            statistic="volatility",
            window=20,
            thresholds=(0.7, 0.3),
            labels=("a", "b", "c"),
        )
    with pytest.raises(PeriodContractError, match="quantiles"):
        RegimeDefinition(
            name="out_of_range",
            statistic="volatility",
            window=20,
            thresholds=(1.5,),
            labels=("a", "b"),
        )


def test_the_universe_refuses_an_empty_record_set() -> None:
    with pytest.raises(UniverseContractError, match="at least one"):
        PointInTimeUniverse(name="empty", records=())


def test_malformed_symbols_are_refused() -> None:
    with pytest.raises(UniverseContractError, match="padding"):
        MembershipRecord(" AAA", date(2018, 1, 1))
    with pytest.raises(UniverseContractError, match="ASCII"):
        MembershipRecord("AA A", date(2018, 1, 1))


def test_a_non_numeric_contribution_is_refused() -> None:
    with pytest.raises(UniverseContractError, match="real number"):
        drop_top_contributors(
            cast(dict[str, float], {"AAA": "big", "BBB": 1.0}), count=1, baseline_metric=1.0
        )
    with pytest.raises(UniverseContractError, match="finite"):
        drop_top_contributors({"AAA": np.nan, "BBB": 1.0}, count=1, baseline_metric=1.0)


def test_a_negative_liquidity_floor_is_refused() -> None:
    with pytest.raises(UniverseContractError, match="non-negative"):
        apply_liquidity_floor({"AAA": 1.0}, {"AAA": 5.0}, minimum=-1.0, baseline_metric=1.0)


def test_an_out_of_range_confidence_is_refused(returns: pd.Series) -> None:
    with pytest.raises(TemporalEvidenceError, match="confidence"):
        naive_interval(returns, confidence=0.2)
    with pytest.raises(TemporalEvidenceError, match="confidence"):
        newey_west_interval(returns, confidence=1.0)


def test_an_unsorted_return_series_is_refused() -> None:
    index = pd.to_datetime(["2020-03-01", "2020-01-01", "2020-02-01"])
    with pytest.raises(TemporalEvidenceError, match="sorted ascending"):
        naive_interval(pd.Series([0.01, 0.02, 0.03], index=index))
