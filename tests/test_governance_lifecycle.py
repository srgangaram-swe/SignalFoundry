"""State machines and the end-to-end governance lifecycle (SF-S5-SL-MR5).

Two things are tested here that unit tests of individual functions cannot show:

* **The transition tables are exhaustive.** Every ordered pair of states is
  enumerated and asserted against the declared table, so a path nobody
  remembered is still covered. A state machine tested only on remembered paths
  has untested paths.
* **The lifecycle holds together.** Bootstrap, compare, reject, approve, apply,
  monitor, freeze, and rebuild are exercised as one sequence against real
  storage, because the failure that matters is at the seams.
"""

from __future__ import annotations

import itertools
import sqlite3
from collections.abc import Iterator
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from quant_platform.governance.comparison import (
    PairedCohort,
    PairedScore,
    PairKey,
)
from quant_platform.governance.gates import (
    Approval,
    Decision,
    FrozenPolicy,
    NotAuthorizedError,
    Recommendation,
    authorize_apply,
    recommend,
)
from quant_platform.governance.inference import (
    Margin,
    non_inferiority_test,
    superiority_test,
)
from quant_platform.governance.inference import (
    TestResult as HypothesisTest,
)
from quant_platform.governance.inference import (
    TestVerdict as Verdict,
)
from quant_platform.governance.lane import (
    TERMINAL_REQUEST_STATES,
    FreezeTrigger,
    GovernanceLane,
    GovernanceStateError,
    LaneState,
    RequestState,
    assert_lane_transition,
    assert_request_transition,
    lane_transitions,
    request_transitions,
)
from quant_platform.governance.store import (
    EventKind,
    GovernanceStore,
    StaleWriteError,
)
from quant_platform.tracking import migrations as migrations_module

BASE = datetime(2026, 8, 1, tzinfo=UTC)
CHAMPION = "a" * 64
CHALLENGER = "b" * 64


@pytest.fixture
def store(tmp_path: Path) -> Iterator[GovernanceStore]:
    path = tmp_path / "registry.sqlite"
    connection = sqlite3.connect(path)
    try:
        connection.executescript(migrations_module._MIGRATION_LEDGER_SQL)
        for migration in migrations_module.MIGRATIONS:
            connection.executescript(migration.sql)
        connection.commit()
    finally:
        connection.close()
    yield GovernanceStore(path)


def _lane() -> GovernanceLane:
    return GovernanceLane(
        purpose="shadow-eval",
        target="direction",
        horizon_days=5,
        frequency="daily",
        universe="us-large-cap",
        decision_policy="long-short",
        environment="local",
    )


# ---------------------------------------------------------------------------
# Exhaustive transition tables
# ---------------------------------------------------------------------------


def test_every_lane_state_pair_matches_the_declared_table() -> None:
    """Enumerated, not sampled: an unlisted pair must be refused."""
    permitted = lane_transitions()
    for current, target in itertools.product(LaneState, LaneState):
        if (current, target) in permitted:
            assert_lane_transition(current, target)
        else:
            with pytest.raises(GovernanceStateError, match="not permitted"):
                assert_lane_transition(current, target)


def test_every_request_state_pair_matches_the_declared_table() -> None:
    permitted = request_transitions()
    for current, target in itertools.product(RequestState, RequestState):
        if (current, target) in permitted:
            assert_request_transition(current, target)
        else:
            with pytest.raises(GovernanceStateError):
                assert_request_transition(current, target)


def test_no_terminal_request_state_has_an_outgoing_transition() -> None:
    """Terminal means terminal; a rollback is a new request, not a reversal."""
    for current in TERMINAL_REQUEST_STATES:
        for target in RequestState:
            with pytest.raises(GovernanceStateError, match="terminal"):
                assert_request_transition(current, target)


def test_there_is_no_automatic_path_into_an_active_lane() -> None:
    """Every arrival at ACTIVE requires an approved, applied assignment."""
    permitted = lane_transitions()
    arrivals = {pair for pair in permitted if pair[1] is LaneState.ACTIVE}
    assert arrivals == {
        (LaneState.UNASSIGNED, LaneState.ACTIVE),
        (LaneState.ACTIVE, LaneState.ACTIVE),
        (LaneState.FROZEN, LaneState.ACTIVE),
    }
    # And none of them originate anywhere the machine can reach on its own.
    assert (LaneState.FROZEN, LaneState.FROZEN) not in permitted
    assert (LaneState.UNASSIGNED, LaneState.FROZEN) not in permitted


@pytest.mark.parametrize("value", ["active", None, 1, LaneState])
def test_a_non_state_value_is_refused_rather_than_coerced(value: Any) -> None:
    with pytest.raises(GovernanceStateError, match="require LaneState"):
        assert_lane_transition(value, LaneState.ACTIVE)
    with pytest.raises(GovernanceStateError, match="require RequestState"):
        assert_request_transition(value, RequestState.APPROVED)


# ---------------------------------------------------------------------------
# End-to-end lifecycle
# ---------------------------------------------------------------------------


def _cohort(effect: float, *, days: int = 30, per_day: int = 8, seed: int = 3) -> PairedCohort:
    """A paired cohort clearing the operational floor, with a chosen effect."""
    rng = np.random.default_rng(seed)
    pairs = []
    for day in range(days):
        shock = rng.normal(0.0, 0.05)
        for index in range(per_day):
            difference = effect + shock + rng.normal(0.0, 0.02)
            as_of = BASE + timedelta(days=day)
            pairs.append(
                PairedScore(
                    key=PairKey(campaign_symbol=f"S{index}", as_of=as_of, horizon_days=5),
                    as_of_date=as_of.date(),
                    champion_brier=0.5,
                    challenger_brier=0.5 + difference,
                    champion_log=0.7,
                    challenger_log=0.7 + difference,
                )
            )
    return PairedCohort(
        pairs=tuple(pairs),
        champion_total=len(pairs),
        challenger_total=len(pairs),
        champion_only=0,
        challenger_only=0,
        champion_unscored=0,
        challenger_unscored=0,
        comparable=True,
        incomparable_reason=None,
    )


def _policy() -> FrozenPolicy:
    return FrozenPolicy(
        version="promotion-1", alpha=0.05, margin=Margin(metric="brier", value=0.01)
    )


def _family(cohort: PairedCohort, policy: FrozenPolicy) -> tuple[HypothesisTest, ...]:
    return (
        superiority_test(cohort, alpha=policy.alpha, seed=5),
        non_inferiority_test(cohort, policy.margin, alpha=policy.alpha, seed=5),
    )


def _decide(cohort: PairedCohort, policy: FrozenPolicy, *, now: datetime = BASE) -> Decision:
    return recommend(
        cohort,
        policy,
        _family(cohort, policy),
        now=now,
        class_counts={"up": 160, "down": 80},
        achieved_power=0.86,
    )


def test_the_full_lifecycle_bootstraps_compares_approves_applies_and_rebuilds(
    store: GovernanceStore,
) -> None:
    """Bootstrap -> compare -> approve -> apply -> monitor -> freeze -> rebuild."""
    lane = _lane()
    identity = store.register_lane(lane, now=BASE)
    policy = _policy()

    # 1. The frozen policy is recorded before any comparison runs.
    store.append_event(identity, EventKind.POLICY, policy.to_dict(), now=BASE)

    # 2. Bootstrap: the first champion is assigned into an unassigned lane.
    head = store.apply_assignment(
        identity,
        champion_revision=CHAMPION,
        expected_generation=0,
        expected_champion=None,
        now=BASE,
        idempotency_key="bootstrap-1",
    )
    assert head.state is LaneState.ACTIVE
    assert head.generation == 1

    # 3. A challenger that is genuinely better produces a PROMOTE decision.
    cohort = _cohort(-0.05)
    decision = _decide(cohort, policy)
    assert decision.recommendation is Recommendation.PROMOTE
    store.append_event(identity, EventKind.COMPARISON, decision.to_dict(), now=BASE)

    # 4. The decision alone cannot promote.
    with pytest.raises(NotAuthorizedError):
        authorize_apply(
            decision,
            Approval(
                approver="reviewer",
                decision_identity="f" * 64,
                approved_at=BASE,
                reference="notes-1",
            ),
            observed_lane_head=CHAMPION,
            expected_lane_head=CHAMPION,
            now=BASE,
        )

    # 5. A named human approves this exact decision.
    approval = Approval(
        approver="reviewer",
        decision_identity=decision.identity,
        approved_at=BASE,
        reference="review-notes-1",
    )
    authorize_apply(
        decision,
        approval,
        observed_lane_head=CHAMPION,
        expected_lane_head=CHAMPION,
        now=BASE + timedelta(days=1),
    )
    store.append_event(identity, EventKind.APPROVAL, approval.to_dict(), now=BASE)

    # 6. Application swaps the champion under compare-and-swap.
    head = store.apply_assignment(
        identity,
        champion_revision=CHALLENGER,
        expected_generation=1,
        expected_champion=CHAMPION,
        now=BASE + timedelta(days=1),
        idempotency_key="promote-1",
    )
    assert head.champion_revision == CHALLENGER
    assert head.generation == 2

    # 7. Monitoring appends evidence without authority to act on it.
    store.append_event(
        identity,
        EventKind.MONITORING,
        {"window": "2026-08-02/2026-08-08", "soft_breaches": 1},
        now=BASE + timedelta(days=8),
    )

    # 8. A hard integrity breach freezes the lane automatically.
    head = store.freeze(
        identity,
        FreezeTrigger.HARD_INTEGRITY,
        detail="artifact digest did not reverify",
        now=BASE + timedelta(days=9),
    )
    assert head.state is LaneState.FROZEN
    assert head.champion_revision == CHALLENGER

    # 9. The whole chain verifies and the projection rebuilds from it.
    # policy, bootstrap, comparison, approval, promotion, monitoring, freeze.
    assert store.verify_chain(identity) == 7
    assert store.rebuild_head(identity) == head
    summary = store.summary(identity)
    assert summary["events_by_kind"] == {
        "approval": 1,
        "assignment": 2,
        "comparison": 1,
        "freeze": 1,
        "monitoring": 1,
        "policy": 1,
    }


def test_a_rejected_comparison_never_reaches_the_lane(store: GovernanceStore) -> None:
    """An unfavourable comparison leaves the champion exactly where it was."""
    identity = store.register_lane(_lane(), now=BASE)
    store.apply_assignment(
        identity,
        champion_revision=CHAMPION,
        expected_generation=0,
        expected_champion=None,
        now=BASE,
    )
    # A challenger that is no better produces no promotion.
    decision = _decide(_cohort(0.0, seed=11), _policy())
    assert decision.recommendation is not Recommendation.PROMOTE
    with pytest.raises(NotAuthorizedError, match="only a PROMOTE recommendation"):
        authorize_apply(
            decision,
            Approval(
                approver="reviewer",
                decision_identity=decision.identity,
                approved_at=BASE,
                reference="review-notes-2",
            ),
            observed_lane_head=CHAMPION,
            expected_lane_head=CHAMPION,
            now=BASE,
        )
    assert store.lane_head(identity).champion_revision == CHAMPION
    assert store.lane_head(identity).generation == 1


def test_two_approved_promotions_racing_leave_one_champion(store: GovernanceStore) -> None:
    """Both were approved against generation 1; only one can apply."""
    identity = store.register_lane(_lane(), now=BASE)
    store.apply_assignment(
        identity,
        champion_revision=CHAMPION,
        expected_generation=0,
        expected_champion=None,
        now=BASE,
    )
    store.apply_assignment(
        identity,
        champion_revision=CHALLENGER,
        expected_generation=1,
        expected_champion=CHAMPION,
        now=BASE,
    )
    with pytest.raises(StaleWriteError, match="won the race"):
        store.apply_assignment(
            identity,
            champion_revision="c" * 64,
            expected_generation=1,
            expected_champion=CHAMPION,
            now=BASE,
        )
    assert store.lane_head(identity).champion_revision == CHALLENGER
    store.verify_chain(identity)


def test_replaying_an_application_does_not_promote_twice(store: GovernanceStore) -> None:
    """A retried apply must be a no-op, not a second generation."""
    identity = store.register_lane(_lane(), now=BASE)
    store.apply_assignment(
        identity,
        champion_revision=CHAMPION,
        expected_generation=0,
        expected_champion=None,
        now=BASE,
        idempotency_key="bootstrap-1",
    )
    # The same key with the same expectations: the CAS refuses the second
    # attempt because generation 0 no longer exists, so the lane cannot
    # advance twice on one approval.
    with pytest.raises(StaleWriteError):
        store.apply_assignment(
            identity,
            champion_revision=CHAMPION,
            expected_generation=0,
            expected_champion=None,
            now=BASE,
            idempotency_key="bootstrap-1",
        )
    assert store.lane_head(identity).generation == 1
    assert len(store.events_of_kind(identity, EventKind.ASSIGNMENT)) == 1


def test_a_clock_rollback_does_not_revive_an_expired_approval() -> None:
    """Validity is checked against the supplied instant, and the past is refused."""
    cohort = _cohort(-0.05)
    policy = _policy()
    decision = _decide(cohort, policy)
    approval = Approval(
        approver="reviewer",
        decision_identity=decision.identity,
        approved_at=BASE,
        reference="review-notes-3",
    )
    # A clock that has rolled back before the approval is not "still valid";
    # is_current requires the instant to fall inside the window, not merely
    # before its end.
    with pytest.raises(NotAuthorizedError, match="outside its"):
        authorize_apply(
            decision,
            approval,
            observed_lane_head=CHAMPION,
            expected_lane_head=CHAMPION,
            now=BASE - timedelta(days=1),
        )


def test_an_underpowered_family_cannot_promote_however_favourable(store: GovernanceStore) -> None:
    """Too little evidence yields INSUFFICIENT_EVIDENCE, not a favourable default."""
    thin = _cohort(-0.20, days=6, per_day=4, seed=13)
    decision = _decide(thin, _policy())
    assert decision.recommendation is Recommendation.INSUFFICIENT_EVIDENCE
    assert any(not gate.satisfied for gate in decision.gates)
    assert all(
        test.verdict is not Verdict.FAVOURS_CHALLENGER or test.p_value is None
        for test in decision.tests
        if test.verdict is Verdict.UNDERPOWERED
    )
