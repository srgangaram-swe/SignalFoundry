"""Tests for the bounded governance read projection (SF-S5-SL-MR6).

These run against real temporary SQLite databases. The projection's job is to
describe stored governance evidence to a browser without becoming a second
source of truth, so the properties under test are:

* **It cannot write.** The connection is opened read-only, and an attempted
  write fails at the database rather than relying on this module's restraint.
* **It reports chain faults instead of hiding them.** A lane whose chain does
  not verify is projected with the fault, because that is the lane an operator
  needs to see.
* **Everything is bounded.** Pages, events, comparisons, gates, tests, and text
  all have ceilings, and truncation is visible rather than silent.
* **Malformed stored evidence fails closed**, never as a favourable default.
"""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Iterator
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pytest

from quant_platform.governance.lane import FreezeTrigger, GovernanceLane, LaneState
from quant_platform.governance.read_ports import (
    MAX_COMPARISONS_PROJECTED,
    MAX_GATES_PROJECTED,
    MAX_LANES_PER_PAGE,
    GovernanceReadError,
    GovernanceReadPorts,
    project_lane_states,
)
from quant_platform.governance.store import EventKind, GovernanceStore
from quant_platform.tracking import migrations as migrations_module

BASE = datetime(2026, 8, 1, tzinfo=UTC)
CHAMPION = "a" * 64
CHALLENGER = "b" * 64
POLICY_DIGEST = "c" * 64
COHORT_DIGEST = "d" * 64


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
def ports(database: Path) -> GovernanceReadPorts:
    return GovernanceReadPorts(database)


def _decision_payload(recommendation: str = "promote", *, gates: int = 3, tests: int = 2) -> dict:
    return {
        "recommendation": recommendation,
        "policy_identity": POLICY_DIGEST,
        "cohort_identity": COHORT_DIGEST,
        "decided_at": BASE.isoformat(),
        "gates": [
            {"name": f"gate_{index}", "satisfied": True, "detail": f"detail {index}"}
            for index in range(gates)
        ],
        "tests": [
            {
                "name": "superiority" if index == 0 else "non_inferiority",
                "metric": "brier",
                "verdict": "favours_challenger",
                "point_estimate": -0.02,
                "interval": [-0.04, -0.01] if index == 0 else [None, -0.005],
                "p_value_uncorrected": 0.001,
                "blocks": 30,
                "observations": 240,
                "margin": None if index == 0 else 0.01,
            }
            for index in range(tests)
        ],
        "correction": {"method": "holm_bonferroni", "alpha": 0.05, "family_size": tests},
    }


@pytest.fixture
def populated(store: GovernanceStore) -> str:
    """A lane carrying one of every event kind the console renders."""
    identity = store.register_lane(_lane(), now=BASE)
    store.append_event(identity, EventKind.POLICY, {"version": "promotion-1"}, now=BASE)
    store.apply_assignment(
        identity,
        champion_revision=CHAMPION,
        expected_generation=0,
        expected_champion=None,
        now=BASE,
    )
    store.append_event(identity, EventKind.COMPARISON, _decision_payload(), now=BASE)
    store.append_event(identity, EventKind.APPROVAL, {"approver": "reviewer"}, now=BASE)
    store.apply_assignment(
        identity,
        champion_revision=CHALLENGER,
        expected_generation=1,
        expected_champion=CHAMPION,
        now=BASE + timedelta(days=1),
    )
    store.append_event(identity, EventKind.MONITORING, {"window": "w1"}, now=BASE)
    return identity


# ---------------------------------------------------------------------------
# Read-only enforcement
# ---------------------------------------------------------------------------


def test_the_projection_connection_refuses_writes(
    ports: GovernanceReadPorts, populated: str
) -> None:
    """Read-only is enforced by the database, not by this module's restraint."""
    connection = ports._connect()
    try:
        with pytest.raises(sqlite3.OperationalError):
            connection.execute("DELETE FROM sl_governance_events")
        with pytest.raises(sqlite3.OperationalError):
            connection.execute(
                "INSERT INTO sl_governance_lanes VALUES (?, '{}', ?)",
                ("f" * 64, BASE.isoformat()),
            )
    finally:
        connection.close()


def test_the_read_port_exposes_no_write_method(ports: GovernanceReadPorts) -> None:
    """A projection that could assign would put promotion behind a GET."""
    forbidden = {
        "append_event",
        "apply_assignment",
        "freeze",
        "register_lane",
        "rebuild_head",
        "unfreeze",
    }
    assert not forbidden & set(dir(ports))


# ---------------------------------------------------------------------------
# Projection correctness
# ---------------------------------------------------------------------------


def test_a_lane_projects_its_head_and_event_counts(
    ports: GovernanceReadPorts, populated: str
) -> None:
    detail = ports.get_lane(populated)
    summary = detail.summary
    assert summary.state is LaneState.ACTIVE
    assert summary.champion_revision == CHALLENGER
    assert summary.generation == 2
    assert summary.purpose == "shadow-eval"
    assert summary.horizon_days == 5
    assert summary.chain_verified is True
    assert summary.chain_fault is None
    assert summary.event_count == 6
    assert summary.events_by_kind == {
        "approval": 1,
        "assignment": 2,
        "comparison": 1,
        "monitoring": 1,
        "policy": 1,
    }
    assert detail.truncated is False
    assert [event.sequence for event in detail.events] == [1, 2, 3, 4, 5, 6]


def test_every_event_kind_gets_a_fixed_summary(ports: GovernanceReadPorts, populated: str) -> None:
    """A generic repr of the payload would let stored data shape the wire."""
    summaries = {event.kind: event.summary for event in ports.get_lane(populated).events}
    assert "champion set to" in summaries["assignment"]
    assert summaries["policy"] == "policy promotion-1 frozen"
    assert summaries["comparison"] == "comparison recommends promote"
    assert summaries["approval"] == "approved by reviewer"
    assert summaries["monitoring"] == "monitoring window appended"


def test_a_frozen_lane_keeps_its_champion_and_trigger(
    store: GovernanceStore, ports: GovernanceReadPorts, populated: str
) -> None:
    store.freeze(populated, FreezeTrigger.HARD_INTEGRITY, detail="artifact mismatch", now=BASE)
    summary = ports.get_lane(populated).summary
    assert summary.state is LaneState.FROZEN
    assert summary.freeze_trigger is FreezeTrigger.HARD_INTEGRITY
    assert summary.champion_revision == CHALLENGER


def test_an_unassigned_lane_projects_without_a_champion(
    store: GovernanceStore, ports: GovernanceReadPorts
) -> None:
    identity = store.register_lane(_lane(purpose="empty-lane"), now=BASE)
    summary = ports.get_lane(identity).summary
    assert summary.state is LaneState.UNASSIGNED
    assert summary.champion_revision is None
    assert summary.generation == 0
    assert summary.event_count == 0


def test_the_console_must_handle_every_lane_state() -> None:
    """The console renders a closed set; the projection publishes it."""
    assert set(project_lane_states()) == {"unassigned", "active", "frozen"}


# ---------------------------------------------------------------------------
# Chain health is reported, not hidden
# ---------------------------------------------------------------------------


def test_a_broken_chain_is_projected_with_its_fault(
    ports: GovernanceReadPorts, database: Path, populated: str
) -> None:
    """The lane worth investigating is exactly the one a filter would drop."""
    connection = sqlite3.connect(database)
    try:
        connection.execute("DROP TRIGGER sl_governance_events_no_update")
        connection.execute(
            "UPDATE sl_governance_events SET payload = '{\"tampered\":true}' WHERE sequence = 3"
        )
        connection.commit()
    finally:
        connection.close()
    summary = ports.get_lane(populated).summary
    assert summary.chain_verified is False
    assert summary.chain_fault is not None
    assert "sequence 3" in summary.chain_fault
    # The lane is still described rather than withheld.
    assert summary.champion_revision == CHALLENGER


def test_a_sequence_gap_is_reported(
    ports: GovernanceReadPorts, database: Path, populated: str
) -> None:
    connection = sqlite3.connect(database)
    try:
        connection.execute("DROP TRIGGER sl_governance_events_no_delete")
        connection.execute("DELETE FROM sl_governance_events WHERE sequence = 2")
        connection.commit()
    finally:
        connection.close()
    summary = ports.get_lane(populated).summary
    assert summary.chain_verified is False
    assert "sequence gap" in (summary.chain_fault or "")


def test_an_over_long_chain_is_reported_unverified_not_verified_on_a_prefix(
    ports: GovernanceReadPorts, populated: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Verifying a prefix and reporting success would be a false assurance."""
    monkeypatch.setattr("quant_platform.governance.read_ports.MAX_CHAIN_ROWS_VERIFIED", 2)
    summary = ports.get_lane(populated).summary
    assert summary.chain_verified is False
    assert "verification ceiling" in (summary.chain_fault or "")


# ---------------------------------------------------------------------------
# Bounds and truncation
# ---------------------------------------------------------------------------


def test_lane_pages_are_bounded_and_cursor_stable(
    store: GovernanceStore, ports: GovernanceReadPorts
) -> None:
    identities = sorted(
        store.register_lane(_lane(purpose=f"lane-{index:02d}"), now=BASE) for index in range(7)
    )
    first = ports.list_lanes(page_size=3)
    assert [item.lane_identity for item in first.items] == identities[:3]
    assert first.next_cursor == identities[2]
    second = ports.list_lanes(page_size=3, cursor=first.next_cursor)
    assert [item.lane_identity for item in second.items] == identities[3:6]
    last = ports.list_lanes(page_size=3, cursor=second.next_cursor)
    assert [item.lane_identity for item in last.items] == identities[6:]
    assert last.next_cursor is None


@pytest.mark.parametrize("page_size", [0, -1, MAX_LANES_PER_PAGE + 1, True, "10", None])
def test_an_unusable_page_size_is_refused(ports: GovernanceReadPorts, page_size: Any) -> None:
    with pytest.raises(GovernanceReadError, match="page_size"):
        ports.list_lanes(page_size=page_size)


@pytest.mark.parametrize("cursor", ["short", "Z" * 64, "A" * 64, 42])
def test_a_malformed_cursor_is_refused(ports: GovernanceReadPorts, cursor: Any) -> None:
    with pytest.raises(GovernanceReadError, match="cursor"):
        ports.list_lanes(cursor=cursor)


@pytest.mark.parametrize("identity", ["short", "Z" * 64, 7, None])
def test_a_malformed_lane_identity_is_refused(ports: GovernanceReadPorts, identity: Any) -> None:
    with pytest.raises(GovernanceReadError, match="lane_identity"):
        ports.get_lane(identity)


def test_an_unknown_lane_is_refused_rather_than_projected_empty(
    ports: GovernanceReadPorts,
) -> None:
    with pytest.raises(GovernanceReadError, match="not registered"):
        ports.get_lane("f" * 64)


def test_event_history_is_truncated_visibly(
    store: GovernanceStore, ports: GovernanceReadPorts, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A clipped history must not read as a complete one."""
    identity = store.register_lane(_lane(purpose="busy-lane"), now=BASE)
    for index in range(5):
        store.append_event(identity, EventKind.MONITORING, {"n": index}, now=BASE)
    monkeypatch.setattr("quant_platform.governance.read_ports.MAX_EVENTS_PROJECTED", 3)
    detail = ports.get_lane(identity)
    assert detail.truncated is True
    assert len(detail.events) == 3
    # The newest events are kept, and they stay in ascending order.
    assert [event.sequence for event in detail.events] == [3, 4, 5]


def test_gate_and_test_collections_report_their_own_truncation(
    store: GovernanceStore, ports: GovernanceReadPorts
) -> None:
    identity = store.register_lane(_lane(purpose="wide-decision"), now=BASE)
    store.append_event(
        identity,
        EventKind.COMPARISON,
        _decision_payload(gates=MAX_GATES_PROJECTED + 4, tests=2),
        now=BASE,
    )
    comparison = ports.list_comparisons(identity)[0]
    assert len(comparison.gates) == MAX_GATES_PROJECTED
    assert comparison.truncated_gates is True
    assert comparison.truncated_tests is False


def test_comparisons_are_bounded_and_ordered_oldest_first(
    store: GovernanceStore, ports: GovernanceReadPorts
) -> None:
    identity = store.register_lane(_lane(purpose="many-decisions"), now=BASE)
    total = MAX_COMPARISONS_PROJECTED + 5
    for index in range(total):
        store.append_event(
            identity,
            EventKind.COMPARISON,
            _decision_payload() | {"decided_at": str(index)},
            now=BASE,
        )
    comparisons = ports.list_comparisons(identity)
    assert len(comparisons) == MAX_COMPARISONS_PROJECTED
    sequences = [item.sequence for item in comparisons]
    assert sequences == sorted(sequences)
    # The most recent decisions are the ones retained.
    assert sequences[-1] == total


def test_long_detail_text_is_truncated_visibly(
    store: GovernanceStore, ports: GovernanceReadPorts
) -> None:
    identity = store.register_lane(_lane(purpose="verbose"), now=BASE)
    payload = _decision_payload()
    payload["gates"][0]["detail"] = "x" * 5_000
    store.append_event(identity, EventKind.COMPARISON, payload, now=BASE)
    detail = ports.list_comparisons(identity)[0].gates[0].detail
    assert len(detail) <= 512
    assert detail.endswith("…")


# ---------------------------------------------------------------------------
# Malformed stored evidence fails closed
# ---------------------------------------------------------------------------


def test_a_one_sided_interval_keeps_its_unbounded_end_null(
    store: GovernanceStore, ports: GovernanceReadPorts
) -> None:
    """Reporting a number there would assert a bound the test never made."""
    identity = store.register_lane(_lane(purpose="one-sided"), now=BASE)
    store.append_event(identity, EventKind.COMPARISON, _decision_payload(), now=BASE)
    tests = ports.list_comparisons(identity)[0].tests
    two_sided, one_sided = tests[0], tests[1]
    assert two_sided.interval_low == pytest.approx(-0.04)
    assert one_sided.interval_low is None
    assert one_sided.interval_high == pytest.approx(-0.005)


def test_a_malformed_interval_is_refused(
    store: GovernanceStore, ports: GovernanceReadPorts
) -> None:
    identity = store.register_lane(_lane(purpose="bad-interval"), now=BASE)
    payload = _decision_payload()
    payload["tests"][0]["interval"] = "not-an-interval"
    store.append_event(identity, EventKind.COMPARISON, payload, now=BASE)
    with pytest.raises(GovernanceReadError, match="two-element array"):
        ports.list_comparisons(identity)


def test_non_array_gates_are_refused(store: GovernanceStore, ports: GovernanceReadPorts) -> None:
    identity = store.register_lane(_lane(purpose="bad-gates"), now=BASE)
    payload = _decision_payload()
    payload["gates"] = {"gate": True}
    store.append_event(identity, EventKind.COMPARISON, payload, now=BASE)
    with pytest.raises(GovernanceReadError, match="must be arrays"):
        ports.list_comparisons(identity)


def test_an_unreadable_lane_key_is_refused(
    store: GovernanceStore, ports: GovernanceReadPorts, database: Path
) -> None:
    identity = store.register_lane(_lane(purpose="corrupt-key"), now=BASE)
    connection = sqlite3.connect(database)
    try:
        connection.execute("DROP TRIGGER sl_governance_lanes_no_update")
        connection.execute(
            "UPDATE sl_governance_lanes SET lane_key = 'not json' WHERE lane_identity = ?",
            (identity,),
        )
        connection.commit()
    finally:
        connection.close()
    with pytest.raises(GovernanceReadError, match="unreadable key record"):
        ports.get_lane(identity)


def test_a_naive_stored_instant_is_refused(
    store: GovernanceStore, ports: GovernanceReadPorts, database: Path
) -> None:
    """An instant that cannot be ordered is a fault, not a value to default."""
    identity = store.register_lane(_lane(purpose="naive-time"), now=BASE)
    connection = sqlite3.connect(database)
    try:
        connection.execute("DROP TRIGGER sl_governance_lanes_no_update")
        connection.execute(
            # 21 characters: long enough to satisfy the column's length CHECK,
            # which is the schema's own guard, so the decoder is what is tested.
            "UPDATE sl_governance_lanes SET created_at = '2026-08-01T00:00:00.0' "
            "WHERE lane_identity = ?",
            (identity,),
        )
        connection.commit()
    finally:
        connection.close()
    with pytest.raises(GovernanceReadError, match="naive"):
        ports.get_lane(identity)


def test_a_non_finite_stored_estimate_is_refused(
    store: GovernanceStore, ports: GovernanceReadPorts, database: Path
) -> None:
    """A NaN reaching a JSON response is not serialisable and must not be coerced."""
    identity = store.register_lane(_lane(purpose="nan-estimate"), now=BASE)
    store.append_event(identity, EventKind.COMPARISON, _decision_payload(), now=BASE)
    connection = sqlite3.connect(database)
    try:
        connection.execute("DROP TRIGGER sl_governance_events_no_update")
        # Written as the raw JSON ``Infinity`` token: json.dumps refuses to
        # emit it, which is itself a guard, so the stored-evidence path is
        # exercised by constructing the document text directly.
        payload = _decision_payload()
        document = json.dumps(payload).replace(
            '"point_estimate": -0.02', '"point_estimate": Infinity', 1
        )
        connection.execute(
            "UPDATE sl_governance_events SET payload = ? WHERE lane_identity = ?",
            (document, identity),
        )
        connection.commit()
    finally:
        connection.close()
    # Chain verification reports the fault rather than raising out of the
    # request, and the projection still refuses to hand the value to a client.
    assert ports.get_lane(identity).summary.chain_verified is False
    with pytest.raises(GovernanceReadError, match="finite"):
        ports.list_comparisons(identity)
