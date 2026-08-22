"""Integration tests for append-only governance persistence (SF-S5-SL-MR5).

These run against real temporary SQLite databases, not fakes: the invariants
under test are enforced partly by triggers and CHECK constraints, and a mock
would assert that the mock works.

The properties carrying the most weight:

* **History is never rewritten.** Updates and deletes on the event chain are
  refused by the database itself.
* **The chain detects divergence and says where.** A tampered payload, a
  removed link, or a relabelled event breaks verification at a named sequence.
* **Compare-and-swap has exactly one winner**, verified with a barrier rather
  than a sleep.
* **The head is a projection.** It rebuilds from events, and a projection that
  disagrees with its events is a reported fault, never a preferred value.
"""

from __future__ import annotations

import json
import sqlite3
import threading
from collections.abc import Iterator
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pytest

from quant_platform.governance.lane import (
    FreezeTrigger,
    GovernanceLane,
    GovernanceStateError,
    LaneHead,
    LaneState,
)
from quant_platform.governance.store import (
    GENESIS_DIGEST,
    MAX_PAYLOAD_BYTES,
    ChainIntegrityError,
    EventKind,
    GovernanceStore,
    GovernanceStoreError,
    StaleWriteError,
    chain_digest,
    idempotency_digest,
    replay_head,
)
from quant_platform.shadow.contracts import ShadowValidationError
from quant_platform.tracking import migrations as migrations_module

BASE = datetime(2026, 8, 1, tzinfo=UTC)
REVISION_A = "a" * 64
REVISION_B = "b" * 64
REVISION_C = "c" * 64


def _lane(**overrides: Any) -> GovernanceLane:
    base: dict[str, Any] = {
        "purpose": "shadow-eval",
        "target": "direction",
        "horizon_days": 5,
        "frequency": "daily",
        "universe": "us-large-cap",
        "decision_policy": "long-short",
        "environment": "local",
    }
    base.update(overrides)
    return GovernanceLane(**base)


@pytest.fixture
def database(tmp_path: Path) -> Iterator[Path]:
    """A real registry database migrated to the latest schema."""
    path = tmp_path / "registry.sqlite"
    connection = sqlite3.connect(path)
    try:
        connection.executescript(migrations_module._MIGRATION_LEDGER_SQL)
        for migration in migrations_module.MIGRATIONS:
            connection.executescript(migration.sql)
        connection.commit()
    finally:
        connection.close()
    yield path


@pytest.fixture
def store(database: Path) -> GovernanceStore:
    return GovernanceStore(database)


@pytest.fixture
def lane_identity(store: GovernanceStore) -> str:
    return store.register_lane(_lane(), now=BASE)


# ---------------------------------------------------------------------------
# Registration and the empty lane
# ---------------------------------------------------------------------------


def test_a_new_lane_starts_unassigned_with_no_champion(
    store: GovernanceStore, lane_identity: str
) -> None:
    head = store.lane_head(lane_identity)
    assert head.state is LaneState.UNASSIGNED
    assert head.champion_revision is None
    assert head.generation == 0
    assert store.load_events(lane_identity) == ()


def test_registering_the_same_lane_twice_is_the_same_lane(store: GovernanceStore) -> None:
    """The lane key is content-derived, so a second registration is a no-op."""
    first = store.register_lane(_lane(), now=BASE)
    second = store.register_lane(_lane(), now=BASE + timedelta(days=1))
    assert first == second
    assert store.lane_head(first).generation == 0


def test_lanes_differing_in_any_key_component_are_different_lanes() -> None:
    """One revision may be champion in one lane and rejected in another."""
    base = _lane()
    for field, value in (
        ("purpose", "other-purpose"),
        ("target", "volatility"),
        ("horizon_days", 20),
        ("frequency", "weekly"),
        ("universe", "us-small-cap"),
        ("decision_policy", "long-only"),
        ("environment", "ci"),
    ):
        assert _lane(**{field: value}).identity != base.identity


def test_an_unknown_lane_is_refused_rather_than_created(store: GovernanceStore) -> None:
    with pytest.raises(GovernanceStoreError, match="not registered"):
        store.lane_head("f" * 64)
    with pytest.raises(GovernanceStoreError, match="not registered"):
        store.append_event("f" * 64, EventKind.POLICY, {"a": 1}, now=BASE)


# ---------------------------------------------------------------------------
# Append-only history
# ---------------------------------------------------------------------------


def test_events_cannot_be_updated_or_deleted(
    store: GovernanceStore, database: Path, lane_identity: str
) -> None:
    """Governance history is evidence; the database refuses to rewrite it."""
    store.append_event(lane_identity, EventKind.POLICY, {"version": "p1"}, now=BASE)
    connection = sqlite3.connect(database)
    try:
        with pytest.raises(sqlite3.IntegrityError, match="immutable"):
            connection.execute("UPDATE sl_governance_events SET payload = '{}'")
        with pytest.raises(sqlite3.IntegrityError, match="append-only"):
            connection.execute("DELETE FROM sl_governance_events")
    finally:
        connection.close()


def test_the_lane_head_cannot_move_backwards(
    store: GovernanceStore, database: Path, lane_identity: str
) -> None:
    """A decreasing generation means a stale writer won a race."""
    store.apply_assignment(
        lane_identity,
        champion_revision=REVISION_A,
        expected_generation=0,
        expected_champion=None,
        now=BASE,
    )
    connection = sqlite3.connect(database)
    try:
        with pytest.raises(sqlite3.IntegrityError, match="may not move backwards"):
            connection.execute("UPDATE sl_governance_lane_head SET generation = 0")
    finally:
        connection.close()


def test_the_first_event_links_to_genesis_and_the_chain_advances(
    store: GovernanceStore, lane_identity: str
) -> None:
    first = store.append_event(lane_identity, EventKind.POLICY, {"n": 1}, now=BASE)
    second = store.append_event(lane_identity, EventKind.COMPARISON, {"n": 2}, now=BASE)
    assert first.sequence == 1
    assert first.previous_digest == GENESIS_DIGEST
    assert second.sequence == 2
    assert second.previous_digest == first.chain_digest
    assert store.verify_chain(lane_identity) == 2


def test_an_oversized_payload_is_refused(store: GovernanceStore, lane_identity: str) -> None:
    with pytest.raises(GovernanceStoreError, match="byte ceiling"):
        store.append_event(
            lane_identity,
            EventKind.POLICY,
            {"blob": "x" * (MAX_PAYLOAD_BYTES + 1)},
            now=BASE,
        )


def test_a_non_finite_payload_value_is_refused(store: GovernanceStore, lane_identity: str) -> None:
    """A NaN reaching the chain would produce a digest over a value nothing equals."""
    with pytest.raises(ValueError, match="JSON compliant"):
        store.append_event(lane_identity, EventKind.MONITORING, {"metric": float("nan")}, now=BASE)


def test_a_naive_instant_is_refused(store: GovernanceStore, lane_identity: str) -> None:
    """The shared validator raises, so the whole platform refuses naive time alike."""
    with pytest.raises(ShadowValidationError, match="timezone-aware"):
        store.append_event(
            lane_identity,
            EventKind.POLICY,
            {"n": 1},
            now=datetime(2026, 8, 1),  # noqa: DTZ001
        )


# ---------------------------------------------------------------------------
# Idempotency
# ---------------------------------------------------------------------------


def test_replaying_an_idempotency_key_returns_the_existing_event(
    store: GovernanceStore, lane_identity: str
) -> None:
    first = store.append_event(
        lane_identity, EventKind.REQUEST, {"n": 1}, now=BASE, idempotency_key="req-1"
    )
    second = store.append_event(
        lane_identity, EventKind.REQUEST, {"n": 1}, now=BASE, idempotency_key="req-1"
    )
    assert first.chain_digest == second.chain_digest
    assert len(store.load_events(lane_identity)) == 1


def test_reusing_a_key_for_different_content_is_a_conflict(
    store: GovernanceStore, lane_identity: str
) -> None:
    """Silently returning the old event would discard the new payload."""
    store.append_event(
        lane_identity, EventKind.REQUEST, {"n": 1}, now=BASE, idempotency_key="req-1"
    )
    with pytest.raises(GovernanceStoreError, match="already used for different content"):
        store.append_event(
            lane_identity, EventKind.REQUEST, {"n": 2}, now=BASE, idempotency_key="req-1"
        )


def test_only_the_digest_of_an_idempotency_key_is_stored(
    store: GovernanceStore, database: Path, lane_identity: str
) -> None:
    """A reader of the table must not be able to replay a caller's key.

    The literal below is deliberately repetitive and low-entropy: it stands in
    for arbitrary caller-supplied text, and a high-entropy stand-in trips the
    repository's secret scanner on a value that is not a credential.
    """
    caller_reference = "replay-me-replay-me"
    store.append_event(
        lane_identity, EventKind.REQUEST, {"n": 1}, now=BASE, idempotency_key=caller_reference
    )
    connection = sqlite3.connect(database)
    try:
        stored = connection.execute(
            "SELECT idempotency_digest FROM sl_governance_events"
        ).fetchone()[0]
    finally:
        connection.close()
    assert stored == idempotency_digest(caller_reference)
    assert caller_reference not in stored
    # The strongest form of the claim: the key's bytes are nowhere in the file.
    assert caller_reference.encode() not in Path(database).read_bytes()


@pytest.mark.parametrize("key", ["", "   ", "k" * 257])
def test_an_unusable_idempotency_key_is_refused(key: str) -> None:
    with pytest.raises(GovernanceStoreError, match="idempotency key"):
        idempotency_digest(key)


# ---------------------------------------------------------------------------
# Chain verification and tamper detection
# ---------------------------------------------------------------------------


def test_a_tampered_payload_breaks_verification_at_its_sequence(
    store: GovernanceStore, database: Path, lane_identity: str
) -> None:
    store.append_event(lane_identity, EventKind.POLICY, {"n": 1}, now=BASE)
    store.append_event(lane_identity, EventKind.COMPARISON, {"n": 2}, now=BASE)
    # The trigger blocks UPDATE, so tamper the way an attacker with file access
    # would: drop the trigger first. That is the threat this detects.
    connection = sqlite3.connect(database)
    try:
        connection.execute("DROP TRIGGER sl_governance_events_no_update")
        connection.execute(
            "UPDATE sl_governance_events SET payload = '{\"n\":99}' WHERE sequence=2"
        )
        connection.commit()
    finally:
        connection.close()
    with pytest.raises(ChainIntegrityError, match="sequence 2 does not match its recorded digest"):
        store.verify_chain(lane_identity)


def test_a_removed_event_is_detected_as_a_sequence_gap(
    store: GovernanceStore, database: Path, lane_identity: str
) -> None:
    for index in range(3):
        store.append_event(lane_identity, EventKind.POLICY, {"n": index}, now=BASE)
    connection = sqlite3.connect(database)
    try:
        connection.execute("DROP TRIGGER sl_governance_events_no_delete")
        connection.execute("DELETE FROM sl_governance_events WHERE sequence = 2")
        connection.commit()
    finally:
        connection.close()
    with pytest.raises(ChainIntegrityError, match="sequence gap"):
        store.verify_chain(lane_identity)


def test_a_relabelled_event_breaks_its_chain_digest(
    store: GovernanceStore, database: Path, lane_identity: str
) -> None:
    """The kind is inside the digest, so an event cannot be retyped."""
    store.append_event(lane_identity, EventKind.MONITORING, {"n": 1}, now=BASE)
    connection = sqlite3.connect(database)
    try:
        connection.execute("DROP TRIGGER sl_governance_events_no_update")
        connection.execute("UPDATE sl_governance_events SET kind = 'approval'")
        connection.commit()
    finally:
        connection.close()
    with pytest.raises(ChainIntegrityError, match="chain digest at sequence 1 does not reproduce"):
        store.verify_chain(lane_identity)


def test_chain_digest_binds_position_and_kind() -> None:
    """Two events with identical payloads at different positions differ."""
    common = {"previous_digest": GENESIS_DIGEST, "payload_digest": "d" * 64}
    assert chain_digest(**common, sequence=1, kind="policy") != chain_digest(
        **common, sequence=2, kind="policy"
    )
    assert chain_digest(**common, sequence=1, kind="policy") != chain_digest(
        **common, sequence=1, kind="approval"
    )


# ---------------------------------------------------------------------------
# Compare-and-swap
# ---------------------------------------------------------------------------


def test_an_assignment_advances_the_generation_and_records_its_event(
    store: GovernanceStore, lane_identity: str
) -> None:
    head = store.apply_assignment(
        lane_identity,
        champion_revision=REVISION_A,
        expected_generation=0,
        expected_champion=None,
        now=BASE,
    )
    assert head.state is LaneState.ACTIVE
    assert head.champion_revision == REVISION_A
    assert head.generation == 1
    events = store.events_of_kind(lane_identity, EventKind.ASSIGNMENT)
    assert len(events) == 1
    assert events[0].payload["previous_champion"] is None


def test_an_assignment_against_a_stale_generation_is_refused(
    store: GovernanceStore, lane_identity: str
) -> None:
    store.apply_assignment(
        lane_identity,
        champion_revision=REVISION_A,
        expected_generation=0,
        expected_champion=None,
        now=BASE,
    )
    with pytest.raises(StaleWriteError, match="another application won the race"):
        store.apply_assignment(
            lane_identity,
            champion_revision=REVISION_B,
            expected_generation=0,
            expected_champion=None,
            now=BASE,
        )


def test_an_assignment_against_the_wrong_champion_is_refused(
    store: GovernanceStore, lane_identity: str
) -> None:
    """Generation alone is not enough: the champion is what was compared against."""
    store.apply_assignment(
        lane_identity,
        champion_revision=REVISION_A,
        expected_generation=0,
        expected_champion=None,
        now=BASE,
    )
    with pytest.raises(StaleWriteError, match="not the one this approval was granted against"):
        store.apply_assignment(
            lane_identity,
            champion_revision=REVISION_C,
            expected_generation=1,
            expected_champion=REVISION_B,
            now=BASE,
        )


def test_a_refused_assignment_leaves_no_event_behind(
    store: GovernanceStore, lane_identity: str
) -> None:
    """A rejected application must not record that it happened."""
    store.apply_assignment(
        lane_identity,
        champion_revision=REVISION_A,
        expected_generation=0,
        expected_champion=None,
        now=BASE,
    )
    before = store.load_events(lane_identity)
    with pytest.raises(StaleWriteError):
        store.apply_assignment(
            lane_identity,
            champion_revision=REVISION_B,
            expected_generation=0,
            expected_champion=None,
            now=BASE,
        )
    assert store.load_events(lane_identity) == before
    assert store.lane_head(lane_identity).champion_revision == REVISION_A


def test_concurrent_applications_have_exactly_one_winner(
    database: Path, lane_identity: str
) -> None:
    """A barrier, not a sleep: both threads reach the swap at the same moment."""
    contenders = 8
    barrier = threading.Barrier(contenders)
    outcomes: list[str] = []
    lock = threading.Lock()

    def attempt(index: int) -> None:
        own_store = GovernanceStore(database)
        revision = f"{index:x}" * 64
        barrier.wait(timeout=30)
        try:
            own_store.apply_assignment(
                lane_identity,
                champion_revision=revision[:64],
                expected_generation=0,
                expected_champion=None,
                now=BASE,
            )
            result = "won"
        except StaleWriteError:
            result = "lost"
        except sqlite3.OperationalError:  # pragma: no cover - lock contention
            result = "busy"
        with lock:
            outcomes.append(result)

    threads = [threading.Thread(target=attempt, args=(index,)) for index in range(contenders)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=60)

    assert outcomes.count("won") == 1, outcomes
    assert outcomes.count("busy") == 0, outcomes
    store = GovernanceStore(database)
    assert store.lane_head(lane_identity).generation == 1
    assert len(store.events_of_kind(lane_identity, EventKind.ASSIGNMENT)) == 1
    store.verify_chain(lane_identity)


@pytest.mark.parametrize("revision", ["short", "Z" * 64, "A" * 64, 123, None])
def test_a_malformed_champion_revision_is_refused(
    store: GovernanceStore, lane_identity: str, revision: Any
) -> None:
    with pytest.raises(GovernanceStoreError, match="SHA-256 digest"):
        store.apply_assignment(
            lane_identity,
            champion_revision=revision,
            expected_generation=0,
            expected_champion=None,
            now=BASE,
        )


# ---------------------------------------------------------------------------
# Freeze
# ---------------------------------------------------------------------------


def test_freezing_an_active_lane_preserves_its_champion(
    store: GovernanceStore, lane_identity: str
) -> None:
    store.apply_assignment(
        lane_identity,
        champion_revision=REVISION_A,
        expected_generation=0,
        expected_champion=None,
        now=BASE,
    )
    head = store.freeze(
        lane_identity,
        FreezeTrigger.HARD_INTEGRITY,
        detail="artifact digest mismatch",
        now=BASE,
    )
    assert head.state is LaneState.FROZEN
    assert head.champion_revision == REVISION_A
    assert head.freeze_trigger is FreezeTrigger.HARD_INTEGRITY


def test_an_unassigned_lane_cannot_be_frozen(store: GovernanceStore, lane_identity: str) -> None:
    with pytest.raises(GovernanceStateError, match="not permitted"):
        store.freeze(lane_identity, FreezeTrigger.HARD_INTEGRITY, detail="x", now=BASE)


def test_freezing_twice_is_refused_rather_than_silently_repeated(
    store: GovernanceStore, lane_identity: str
) -> None:
    store.apply_assignment(
        lane_identity,
        champion_revision=REVISION_A,
        expected_generation=0,
        expected_champion=None,
        now=BASE,
    )
    store.freeze(lane_identity, FreezeTrigger.HARD_INTEGRITY, detail="first", now=BASE)
    with pytest.raises(GovernanceStateError, match="not permitted"):
        store.freeze(lane_identity, FreezeTrigger.HARD_INTEGRITY, detail="second", now=BASE)


@pytest.mark.parametrize("detail", ["", "   ", "d" * 513])
def test_an_unusable_freeze_detail_is_refused(
    store: GovernanceStore, lane_identity: str, detail: str
) -> None:
    with pytest.raises(GovernanceStoreError, match="freeze detail"):
        store.freeze(lane_identity, FreezeTrigger.HARD_INTEGRITY, detail=detail, now=BASE)


def test_a_frozen_lane_reactivates_only_through_a_new_assignment(
    store: GovernanceStore, lane_identity: str
) -> None:
    """There is no unfreeze method. Clearing a freeze is a new assignment."""
    assert not hasattr(store, "unfreeze")
    store.apply_assignment(
        lane_identity,
        champion_revision=REVISION_A,
        expected_generation=0,
        expected_champion=None,
        now=BASE,
    )
    store.freeze(lane_identity, FreezeTrigger.CONSECUTIVE_SOFT_BREACH, detail="drift", now=BASE)
    head = store.apply_assignment(
        lane_identity,
        champion_revision=REVISION_A,
        expected_generation=1,
        expected_champion=REVISION_A,
        now=BASE,
    )
    assert head.state is LaneState.ACTIVE
    assert head.freeze_trigger is None


# ---------------------------------------------------------------------------
# Projection rebuild
# ---------------------------------------------------------------------------


def test_the_head_rebuilds_from_its_events(store: GovernanceStore, lane_identity: str) -> None:
    store.apply_assignment(
        lane_identity,
        champion_revision=REVISION_A,
        expected_generation=0,
        expected_champion=None,
        now=BASE,
    )
    store.apply_assignment(
        lane_identity,
        champion_revision=REVISION_B,
        expected_generation=1,
        expected_champion=REVISION_A,
        now=BASE,
    )
    store.freeze(lane_identity, FreezeTrigger.HARD_INTEGRITY, detail="stop", now=BASE)
    rebuilt = store.rebuild_head(lane_identity)
    assert rebuilt.state is LaneState.FROZEN
    assert rebuilt.champion_revision == REVISION_B
    assert rebuilt.generation == 2


def test_a_projection_that_disagrees_with_its_events_is_a_reported_fault(
    store: GovernanceStore, database: Path, lane_identity: str
) -> None:
    """The events are the authority; a divergent cache is never preferred."""
    store.apply_assignment(
        lane_identity,
        champion_revision=REVISION_A,
        expected_generation=0,
        expected_champion=None,
        now=BASE,
    )
    connection = sqlite3.connect(database)
    try:
        connection.execute(
            "UPDATE sl_governance_lane_head SET champion_revision = ?", (REVISION_C,)
        )
        connection.commit()
    finally:
        connection.close()
    with pytest.raises(ChainIntegrityError, match="disagrees with the one its events produce"):
        store.rebuild_head(lane_identity)


def test_rebuild_verifies_the_chain_before_trusting_it(
    store: GovernanceStore, database: Path, lane_identity: str
) -> None:
    """A projection rebuilt from a broken chain would launder the break."""
    store.apply_assignment(
        lane_identity,
        champion_revision=REVISION_A,
        expected_generation=0,
        expected_champion=None,
        now=BASE,
    )
    connection = sqlite3.connect(database)
    try:
        connection.execute("DROP TRIGGER sl_governance_events_no_update")
        connection.execute(
            "UPDATE sl_governance_events SET payload = ?",
            (json.dumps({"champion_revision": REVISION_C, "generation": 1}),),
        )
        connection.commit()
    finally:
        connection.close()
    with pytest.raises(ChainIntegrityError, match="does not match its recorded digest"):
        store.rebuild_head(lane_identity)


def test_replay_head_is_pure_and_matches_the_store(
    store: GovernanceStore, lane_identity: str
) -> None:
    store.apply_assignment(
        lane_identity,
        champion_revision=REVISION_A,
        expected_generation=0,
        expected_champion=None,
        now=BASE,
    )
    events = store.load_events(lane_identity)
    assert replay_head(events, lane_identity=lane_identity) == store.lane_head(lane_identity)
    assert replay_head((), lane_identity=lane_identity).state is LaneState.UNASSIGNED


def test_a_store_reopened_after_restart_sees_the_same_history(
    database: Path, lane_identity: str
) -> None:
    """Recovery is reading the file again, because nothing is held in memory."""
    first = GovernanceStore(database)
    first.apply_assignment(
        lane_identity,
        champion_revision=REVISION_A,
        expected_generation=0,
        expected_champion=None,
        now=BASE,
    )
    tip = first.summary(lane_identity)["chain_tip"]
    reopened = GovernanceStore(database)
    assert reopened.verify_chain(lane_identity) == 1
    assert reopened.summary(lane_identity)["chain_tip"] == tip
    assert reopened.rebuild_head(lane_identity).champion_revision == REVISION_A


# ---------------------------------------------------------------------------
# Contract invariants
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"lane_identity": "short"}, "full SHA-256 digest"),
        ({"generation": -1}, "generation must lie"),
        ({"generation": True}, "generation must be an int"),
        ({"state": LaneState.UNASSIGNED, "champion_revision": REVISION_A}, "cannot name a"),
        ({"state": LaneState.ACTIVE, "champion_revision": None}, "must name its champion"),
        ({"state": LaneState.FROZEN, "freeze_trigger": None}, "must record its trigger"),
        (
            {"state": LaneState.ACTIVE, "freeze_trigger": FreezeTrigger.HARD_INTEGRITY},
            "only a frozen lane may carry one",
        ),
    ],
)
def test_an_incoherent_lane_head_cannot_be_constructed(
    kwargs: dict[str, Any], message: str
) -> None:
    base: dict[str, Any] = {
        "lane_identity": "a" * 64,
        "state": LaneState.ACTIVE,
        "champion_revision": REVISION_A,
        "generation": 1,
        "freeze_trigger": None,
    }
    base.update(kwargs)
    with pytest.raises(Exception, match=message):
        LaneHead(**base)


def test_the_summary_is_bounded_and_states_its_own_limits(
    store: GovernanceStore, lane_identity: str
) -> None:
    store.append_event(lane_identity, EventKind.POLICY, {"n": 1}, now=BASE)
    store.append_event(lane_identity, EventKind.MONITORING, {"n": 2}, now=BASE)
    summary = store.summary(lane_identity)
    assert summary["events"] == 2
    assert summary["events_by_kind"] == {"monitoring": 1, "policy": 1}
    assert "not externally tamper-proof" in summary["note"]
    assert json.loads(json.dumps(summary, allow_nan=False))


def test_an_empty_lane_summarizes_to_the_genesis_tip(
    store: GovernanceStore, lane_identity: str
) -> None:
    assert store.summary(lane_identity)["chain_tip"] == GENESIS_DIGEST


# ---------------------------------------------------------------------------
# Remaining refusal paths
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("purpose", "Shadow-Eval", "purpose"),
        ("target", "", "target"),
        ("frequency", "d" * 65, "frequency"),
        ("universe", 7, "universe"),
        ("decision_policy", None, "decision_policy"),
        ("environment", "1local", "environment"),
    ],
)
def test_a_malformed_lane_key_component_is_refused(field: str, value: Any, message: str) -> None:
    with pytest.raises(ShadowValidationError, match=message):
        _lane(**{field: value})


@pytest.mark.parametrize("horizon", [0, 366, -1, "5", None])
def test_an_unusable_lane_horizon_is_refused(horizon: Any) -> None:
    with pytest.raises(ShadowValidationError, match="horizon_days"):
        _lane(horizon_days=horizon)


def test_a_non_lane_object_cannot_be_registered(store: GovernanceStore) -> None:
    with pytest.raises(GovernanceStoreError, match="must be a GovernanceLane"):
        store.register_lane("shadow-eval", now=BASE)  # type: ignore[arg-type]


def test_a_non_event_kind_is_refused(store: GovernanceStore, lane_identity: str) -> None:
    with pytest.raises(GovernanceStoreError, match="must be an EventKind"):
        store.append_event(lane_identity, "policy", {"n": 1}, now=BASE)  # type: ignore[arg-type]


def test_a_non_freeze_trigger_is_refused(store: GovernanceStore, lane_identity: str) -> None:
    with pytest.raises(GovernanceStoreError, match="must be a FreezeTrigger"):
        store.freeze(lane_identity, "hard_integrity", detail="x", now=BASE)  # type: ignore[arg-type]


@pytest.mark.parametrize("generation", ["1", True, None, 1.5])
def test_a_non_integer_expected_generation_is_refused(
    store: GovernanceStore, lane_identity: str, generation: Any
) -> None:
    with pytest.raises(GovernanceStoreError, match="expected_generation must be an int"):
        store.apply_assignment(
            lane_identity,
            champion_revision=REVISION_A,
            expected_generation=generation,
            expected_champion=None,
            now=BASE,
        )


def test_assigning_or_freezing_an_unregistered_lane_is_refused(store: GovernanceStore) -> None:
    for call in (
        lambda: store.apply_assignment(
            "f" * 64,
            champion_revision=REVISION_A,
            expected_generation=0,
            expected_champion=None,
            now=BASE,
        ),
        lambda: store.freeze("f" * 64, FreezeTrigger.HARD_INTEGRITY, detail="x", now=BASE),
    ):
        with pytest.raises(GovernanceStoreError, match="not registered"):
            call()


@pytest.mark.parametrize("stored", ["not-a-valid-date-here", "2026-08-01T00:00:00.0"])
def test_an_unusable_stored_instant_is_refused(
    store: GovernanceStore, database: Path, lane_identity: str, stored: str
) -> None:
    """A row whose timestamp cannot be ordered is a fault, not a default.

    Both values are long enough to satisfy the column's length CHECK, which is
    the schema's own first line of defence; what is under test here is the
    decoder behind it.
    """
    store.append_event(lane_identity, EventKind.POLICY, {"n": 1}, now=BASE)
    connection = sqlite3.connect(database)
    try:
        connection.execute("DROP TRIGGER sl_governance_events_no_update")
        connection.execute("UPDATE sl_governance_events SET recorded_at = ?", (stored,))
        connection.commit()
    finally:
        connection.close()
    with pytest.raises(GovernanceStoreError, match="unparseable|naive"):
        store.load_events(lane_identity)


def test_a_lane_over_the_event_ceiling_refuses_to_truncate(
    store: GovernanceStore, lane_identity: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A silently truncated history would read as a complete one."""
    for index in range(3):
        store.append_event(lane_identity, EventKind.MONITORING, {"n": index}, now=BASE)
    monkeypatch.setattr("quant_platform.governance.store.MAX_EVENTS_PER_QUERY", 2)
    with pytest.raises(GovernanceStoreError, match="silently truncated"):
        store.load_events(lane_identity)


def test_a_broken_predecessor_link_is_reported_with_its_sequence(
    store: GovernanceStore, database: Path, lane_identity: str
) -> None:
    """The reader must learn where history diverged, not only that it did."""
    store.append_event(lane_identity, EventKind.POLICY, {"n": 1}, now=BASE)
    store.append_event(lane_identity, EventKind.POLICY, {"n": 2}, now=BASE)
    connection = sqlite3.connect(database)
    try:
        connection.execute("DROP TRIGGER sl_governance_events_no_update")
        connection.execute(
            "UPDATE sl_governance_events SET previous_digest = ? WHERE sequence = 2",
            ("e" * 64,),
        )
        connection.commit()
    finally:
        connection.close()
    with pytest.raises(ChainIntegrityError, match="chain breaks at sequence 2"):
        store.verify_chain(lane_identity)


def test_the_event_record_serializes(store: GovernanceStore, lane_identity: str) -> None:
    event = store.append_event(lane_identity, EventKind.POLICY, {"n": 1}, now=BASE)
    payload = event.to_dict()
    assert payload["kind"] == "policy"
    assert payload["sequence"] == 1
    assert json.loads(json.dumps(payload, allow_nan=False))


def test_replay_head_reconstructs_a_freeze_from_events(lane_identity: str) -> None:
    """The freeze branch must replay, not only the assignment branch."""
    from quant_platform.governance.store import GovernanceEvent

    events = (
        GovernanceEvent(
            lane_identity=lane_identity,
            sequence=1,
            kind=EventKind.ASSIGNMENT,
            payload={"champion_revision": REVISION_A, "generation": 1},
            payload_digest="a" * 64,
            previous_digest=GENESIS_DIGEST,
            chain_digest="b" * 64,
            recorded_at=BASE,
        ),
        GovernanceEvent(
            lane_identity=lane_identity,
            sequence=2,
            kind=EventKind.FREEZE,
            payload={"trigger": "consecutive_soft_breach", "detail": "drift"},
            payload_digest="c" * 64,
            previous_digest="b" * 64,
            chain_digest="d" * 64,
            recorded_at=BASE,
        ),
    )
    head = replay_head(events, lane_identity=lane_identity)
    assert head.state is LaneState.FROZEN
    assert head.freeze_trigger is FreezeTrigger.CONSECUTIVE_SOFT_BREACH
    assert head.champion_revision == REVISION_A
