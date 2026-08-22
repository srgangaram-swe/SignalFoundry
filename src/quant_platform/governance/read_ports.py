"""Bounded read-only projection of governance lanes for the evidence service.

SF-S5-SL-MR6. The console needs to render lane state, gate outcomes, and
promotion history. It must not gain a second source of truth, and it must not
gain any authority the API does not already have.

Three properties shape this module:

**Read-only by construction.** Every statement here is a ``SELECT``. There is no
write path, no transaction, and no method that could append, assign, freeze, or
repair. The service that consumes it therefore cannot promote a model even if a
caller found a way to reach it.

**The projection reports chain health rather than asserting it.** A lane whose
event chain does not verify is projected with ``chain_verified=False`` and the
sequence at which it broke, not omitted and not silently repaired. A console
that renders only healthy lanes would hide exactly the lanes worth looking at.

**Payloads are summarised, never forwarded raw.** Event payloads carry policy
declarations and decision records. Those are aggregate evidence and safe to
show, but forwarding an arbitrary stored blob to a browser would make the
response shape depend on whatever a writer happened to store. Each event kind is
projected into a fixed, bounded shape instead.
"""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Sequence
from contextlib import closing
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Final, assert_never

from quant_platform.governance.lane import FreezeTrigger, LaneState
from quant_platform.governance.store import (
    GENESIS_DIGEST,
    EventKind,
    chain_digest,
)
from quant_platform.shadow.contracts import canonical_digest
from quant_platform.tracking.contracts import ValidationError

#: Ceilings for one read-port instance. Refusal thresholds, not tuning knobs.
MAX_LANES_PER_PAGE: Final = 100
MAX_EVENTS_PROJECTED: Final = 200
MAX_COMPARISONS_PROJECTED: Final = 50
MAX_GATES_PROJECTED: Final = 32
MAX_TESTS_PROJECTED: Final = 16
MAX_DETAIL_CHARS: Final = 512

#: Read queries are bounded in time as well as in rows: a governance lane with a
#: pathological index state must fail the request rather than hold a service
#: worker open.
QUERY_TIMEOUT_MS: Final = 500

#: Events verified per lane before the projection reports the chain unverified.
MAX_CHAIN_ROWS_VERIFIED: Final = 10_000


class GovernanceReadError(ValidationError):
    """Raised when governance evidence cannot be projected safely."""


@dataclass(frozen=True, slots=True)
class GateProjection:
    """One absolute gate as the console renders it."""

    name: str
    satisfied: bool
    detail: str


@dataclass(frozen=True, slots=True)
class TestProjection:
    """One hypothesis test, with its interval kept as an explicit pair.

    ``interval_low`` and ``interval_high`` are independently nullable because a
    one-sided test genuinely has one unbounded end. Collapsing that to a number
    would invent a bound the test never established.
    """

    name: str
    metric: str
    verdict: str
    point_estimate: float | None
    interval_low: float | None
    interval_high: float | None
    p_value_uncorrected: float | None
    blocks: int
    observations: int
    margin: float | None


@dataclass(frozen=True, slots=True)
class ComparisonProjection:
    """One promotion decision with every gate and test that produced it."""

    sequence: int
    recorded_at: datetime
    recommendation: str
    policy_identity: str
    cohort_identity: str
    decided_at: str
    gates: tuple[GateProjection, ...]
    tests: tuple[TestProjection, ...]
    correction_method: str | None
    correction_alpha: float | None
    family_size: int | None
    truncated_gates: bool = False
    truncated_tests: bool = False


@dataclass(frozen=True, slots=True)
class LaneEventProjection:
    """One chain event reduced to a fixed, bounded shape."""

    sequence: int
    kind: str
    recorded_at: datetime
    chain_digest: str
    summary: str


@dataclass(frozen=True, slots=True)
class LaneSummary:
    """A lane's head projection plus the health of the chain behind it."""

    lane_identity: str
    purpose: str
    target: str
    horizon_days: int
    frequency: str
    universe: str
    decision_policy: str
    environment: str
    state: LaneState
    champion_revision: str | None
    generation: int
    freeze_trigger: FreezeTrigger | None
    created_at: datetime
    event_count: int
    chain_verified: bool
    chain_fault: str | None = None
    events_by_kind: dict[str, int] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class LaneDetail:
    """One lane with its bounded event history."""

    summary: LaneSummary
    events: tuple[LaneEventProjection, ...]
    truncated: bool


@dataclass(frozen=True, slots=True)
class LanePage:
    """A bounded page of lane summaries with a stable next cursor."""

    items: tuple[LaneSummary, ...]
    next_cursor: str | None


def _decode_instant(value: object, *, field_name: str) -> datetime:
    """Return an aware UTC datetime from a stored text instant.

    Raises:
        GovernanceReadError: On a missing, unparseable, or naive value. A row
            whose timestamp cannot be ordered is a fault to report, never a
            value to default.
    """
    if not isinstance(value, str):
        raise GovernanceReadError(f"{field_name} must be stored as text")
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError as error:
        raise GovernanceReadError(f"{field_name} is not a valid instant") from error
    if parsed.tzinfo is None:
        raise GovernanceReadError(f"{field_name} is naive and cannot be ordered")
    return parsed.astimezone(UTC)


def _bounded_text(value: object, *, field_name: str) -> str:
    """Return stored text truncated to the projection ceiling.

    Truncation is visible (an ellipsis is appended) rather than silent, because
    a reader must be able to tell a short detail from a clipped one.
    """
    if not isinstance(value, str):
        raise GovernanceReadError(f"{field_name} must be stored as text")
    if len(value) <= MAX_DETAIL_CHARS:
        return value
    return value[: MAX_DETAIL_CHARS - 1] + "…"


def _optional_float(payload: dict[str, Any], key: str) -> float | None:
    """Return a finite float from a payload, or ``None``.

    Non-finite values are rejected rather than forwarded: a NaN reaching a JSON
    response body is not serialisable, and a caller that coerced it would be
    rendering a number nothing equals.
    """
    raw = payload.get(key)
    if raw is None:
        return None
    if isinstance(raw, bool) or not isinstance(raw, (int, float)):
        raise GovernanceReadError(f"{key} must be a real number or null")
    value = float(raw)
    if value != value or value in (float("inf"), float("-inf")):
        raise GovernanceReadError(f"{key} must be finite")
    return value


def _summarize_event(kind: EventKind, payload: dict[str, Any]) -> str:
    """Return a fixed, bounded one-line description of an event.

    Each kind gets an explicit branch. A generic ``str(payload)`` would let the
    response shape follow whatever a writer stored, which is the opposite of a
    projection.
    """
    if kind is EventKind.ASSIGNMENT:
        champion = str(payload.get("champion_revision", ""))[:12]
        generation = payload.get("generation")
        return f"champion set to {champion} at generation {generation}"
    if kind is EventKind.FREEZE:
        return f"frozen: {_bounded_text(payload.get('detail', ''), field_name='detail')}"
    if kind is EventKind.POLICY:
        return f"policy {payload.get('version', 'unknown')} frozen"
    if kind is EventKind.COMPARISON:
        return f"comparison recommends {payload.get('recommendation', 'unknown')}"
    if kind is EventKind.APPROVAL:
        approver = _bounded_text(payload.get("approver", ""), field_name="approver")
        return f"approved by {approver}"
    if kind is EventKind.MONITORING:
        return "monitoring window appended"
    if kind is EventKind.REQUEST:
        return "change request recorded"
    # Exhaustiveness is enforced by the type checker rather than by a runtime
    # fallback: adding an EventKind without a branch here fails mypy, which is
    # strictly better than shipping a generic label nobody notices is wrong.
    assert_never(kind)


def _verify_chain_readonly(connection: sqlite3.Connection, lane_identity: str) -> str | None:
    """Recompute one lane's chain from stored rows, returning the first fault.

    Returns ``None`` when the chain reproduces end to end, otherwise a bounded
    description naming the sequence at which it diverged. A fault is a value to
    report, never an exception to swallow: the caller renders it.

    The scan is bounded by :data:`MAX_CHAIN_ROWS_VERIFIED`; a longer chain is
    reported as unverified rather than silently verified on a prefix.
    """
    rows = connection.execute(
        "SELECT sequence, kind, payload, payload_digest, previous_digest, chain_digest "
        "FROM sl_governance_events WHERE lane_identity = ? ORDER BY sequence ASC LIMIT ?",
        (lane_identity, MAX_CHAIN_ROWS_VERIFIED + 1),
    ).fetchall()
    if len(rows) > MAX_CHAIN_ROWS_VERIFIED:
        return (
            f"chain exceeds the {MAX_CHAIN_ROWS_VERIFIED}-event verification ceiling; "
            "reported as unverified rather than verified on a prefix"
        )
    previous = GENESIS_DIGEST
    for position, row in enumerate(rows, start=1):
        if int(row["sequence"]) != position:
            return f"sequence gap: expected {position}, found {row['sequence']}"
        try:
            payload = json.loads(row["payload"])
        except (TypeError, ValueError):
            return f"payload at sequence {position} is unreadable"
        try:
            recomputed_payload = canonical_digest(payload)
        except ValueError:
            # A payload carrying a non-finite value cannot be canonically
            # digested, so it cannot match its recorded digest either. That is a
            # chain fault to report, not an exception to escape verification:
            # a crash here would take down a request that asked a fair question.
            return (
                f"payload at sequence {position} is not canonically representable "
                "(non-finite value)"
            )
        if recomputed_payload != row["payload_digest"]:
            return f"payload at sequence {position} does not match its recorded digest"
        if row["previous_digest"] != previous:
            return f"chain breaks at sequence {position}"
        recomputed_link = chain_digest(
            previous_digest=previous,
            payload_digest=recomputed_payload,
            sequence=position,
            kind=str(row["kind"]),
        )
        if recomputed_link != row["chain_digest"]:
            return f"chain digest at sequence {position} does not reproduce"
        previous = str(row["chain_digest"])
    return None


class GovernanceReadPorts:
    """Read-only projection over one registry database's governance tables.

    The instance holds a path, not a connection: each call opens, reads, and
    closes, so a projection can never hold a write lock or outlive its request.
    """

    def __init__(self, database: str | Path) -> None:
        self._path = Path(database).expanduser()

    def _connect(self) -> sqlite3.Connection:
        """Open a read-only, time-bounded connection.

        Opened through the ``file:...?mode=ro`` URI so the operating system
        refuses a write attempt, rather than relying on this module never
        issuing one.
        """
        uri = f"file:{self._path}?mode=ro"
        connection = sqlite3.connect(uri, uri=True, timeout=QUERY_TIMEOUT_MS / 1000)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA query_only = ON")
        return connection

    def list_lanes(self, *, page_size: int = 25, cursor: str | None = None) -> LanePage:
        """Return one bounded page of lane summaries ordered by identity.

        Ordering is by ``lane_identity``, which is content-derived and therefore
        stable: a lane cannot move within the ordering as its state changes, so
        keyset pagination cannot skip or repeat a row.

        Raises:
            GovernanceReadError: On an unusable page size or cursor.
        """
        if isinstance(page_size, bool) or not isinstance(page_size, int):
            raise GovernanceReadError("page_size must be an int")
        if not 1 <= page_size <= MAX_LANES_PER_PAGE:
            raise GovernanceReadError(f"page_size must be in [1, {MAX_LANES_PER_PAGE}]")
        if cursor is not None and (
            not isinstance(cursor, str)
            or len(cursor) != 64
            or not all(character in "0123456789abcdef" for character in cursor)
        ):
            raise GovernanceReadError("cursor must be a full lowercase SHA-256 digest")

        with closing(self._connect()) as connection:
            rows = connection.execute(
                "SELECT lane_identity, lane_key, created_at FROM sl_governance_lanes "
                "WHERE (? IS NULL OR lane_identity > ?) ORDER BY lane_identity ASC LIMIT ?",
                (cursor, cursor, page_size + 1),
            ).fetchall()
            has_more = len(rows) > page_size
            visible = rows[:page_size]
            items = tuple(self._summarize_lane(connection, row) for row in visible)
        next_cursor = items[-1].lane_identity if has_more and items else None
        return LanePage(items=items, next_cursor=next_cursor)

    def get_lane(self, lane_identity: str) -> LaneDetail:
        """Return one lane with a bounded slice of its most recent events.

        The newest events are returned because a reader inspecting a lane cares
        about what happened last; ``truncated`` says plainly when older history
        exists rather than implying the list is complete.

        Raises:
            GovernanceReadError: If the lane is unknown or malformed.
        """
        if (
            not isinstance(lane_identity, str)
            or len(lane_identity) != 64
            or not all(character in "0123456789abcdef" for character in lane_identity)
        ):
            raise GovernanceReadError("lane_identity must be a full lowercase SHA-256 digest")
        with closing(self._connect()) as connection:
            row = connection.execute(
                "SELECT lane_identity, lane_key, created_at FROM sl_governance_lanes "
                "WHERE lane_identity = ?",
                (lane_identity,),
            ).fetchone()
            if row is None:
                raise GovernanceReadError(f"governance lane {lane_identity[:12]} is not registered")
            summary = self._summarize_lane(connection, row)
            event_rows = connection.execute(
                "SELECT sequence, kind, payload, chain_digest, recorded_at "
                "FROM sl_governance_events WHERE lane_identity = ? "
                "ORDER BY sequence DESC LIMIT ?",
                (lane_identity, MAX_EVENTS_PROJECTED + 1),
            ).fetchall()
        truncated = len(event_rows) > MAX_EVENTS_PROJECTED
        events = tuple(
            self._project_event(row) for row in reversed(event_rows[:MAX_EVENTS_PROJECTED])
        )
        return LaneDetail(summary=summary, events=events, truncated=truncated)

    def list_comparisons(self, lane_identity: str) -> tuple[ComparisonProjection, ...]:
        """Return the lane's recorded promotion decisions, newest last.

        Raises:
            GovernanceReadError: If the lane is unknown or a decision payload is
                not shaped like a decision.
        """
        detail_guard = self.get_lane(lane_identity)  # validates identity and existence
        del detail_guard
        with closing(self._connect()) as connection:
            rows = connection.execute(
                "SELECT sequence, payload, recorded_at FROM sl_governance_events "
                "WHERE lane_identity = ? AND kind = ? ORDER BY sequence DESC LIMIT ?",
                (lane_identity, EventKind.COMPARISON.value, MAX_COMPARISONS_PROJECTED),
            ).fetchall()
        return tuple(self._project_comparison(row) for row in reversed(rows))

    # -- internals --------------------------------------------------------

    def _summarize_lane(self, connection: sqlite3.Connection, row: sqlite3.Row) -> LaneSummary:
        """Build one lane summary, reporting chain health rather than asserting it."""
        lane_identity = str(row["lane_identity"])
        try:
            key = json.loads(row["lane_key"])
        except (TypeError, ValueError) as error:
            raise GovernanceReadError(
                f"lane {lane_identity[:12]} has an unreadable key record"
            ) from error
        if not isinstance(key, dict):
            raise GovernanceReadError(f"lane {lane_identity[:12]} key is not an object")

        head = connection.execute(
            "SELECT state, champion_revision, generation, freeze_trigger "
            "FROM sl_governance_lane_head WHERE lane_identity = ?",
            (lane_identity,),
        ).fetchone()
        if head is None:
            raise GovernanceReadError(
                f"lane {lane_identity[:12]} has no head projection; the registry is "
                "inconsistent and the lane cannot be described"
            )
        counts_rows = connection.execute(
            "SELECT kind, COUNT(*) AS total FROM sl_governance_events "
            "WHERE lane_identity = ? GROUP BY kind",
            (lane_identity,),
        ).fetchall()
        events_by_kind = {str(item["kind"]): int(item["total"]) for item in counts_rows}

        # Verified through this port's own read-only connection rather than by
        # delegating to the writable store. A read projection that held a
        # write-capable object would put an assignment path one attribute away
        # from a request handler.
        chain_fault = _verify_chain_readonly(connection, lane_identity)
        chain_verified = chain_fault is None

        return LaneSummary(
            lane_identity=lane_identity,
            purpose=str(key.get("purpose", "")),
            target=str(key.get("target", "")),
            horizon_days=int(key.get("horizon_days", 0)),
            frequency=str(key.get("frequency", "")),
            universe=str(key.get("universe", "")),
            decision_policy=str(key.get("decision_policy", "")),
            environment=str(key.get("environment", "")),
            state=LaneState(str(head["state"])),
            champion_revision=head["champion_revision"],
            generation=int(head["generation"]),
            freeze_trigger=(
                FreezeTrigger(str(head["freeze_trigger"])) if head["freeze_trigger"] else None
            ),
            created_at=_decode_instant(row["created_at"], field_name="created_at"),
            event_count=sum(events_by_kind.values()),
            chain_verified=chain_verified,
            chain_fault=chain_fault,
            events_by_kind=dict(sorted(events_by_kind.items())),
        )

    def _payload(self, row: sqlite3.Row) -> dict[str, Any]:
        """Return a decoded event payload object.

        Raises:
            GovernanceReadError: If the payload is not a JSON object.
        """
        try:
            payload = json.loads(row["payload"])
        except (TypeError, ValueError) as error:
            raise GovernanceReadError(
                f"event {row['sequence']} has an unreadable payload"
            ) from error
        if not isinstance(payload, dict):
            raise GovernanceReadError(f"event {row['sequence']} payload is not an object")
        return payload

    def _project_event(self, row: sqlite3.Row) -> LaneEventProjection:
        """Reduce one stored event to its fixed console shape."""
        kind = EventKind(str(row["kind"]))
        return LaneEventProjection(
            sequence=int(row["sequence"]),
            kind=kind.value,
            recorded_at=_decode_instant(row["recorded_at"], field_name="recorded_at"),
            chain_digest=str(row["chain_digest"]),
            summary=_summarize_event(kind, self._payload(row)),
        )

    def _project_comparison(self, row: sqlite3.Row) -> ComparisonProjection:
        """Reduce one decision record to its fixed console shape.

        Gates and tests are bounded independently, and truncation is reported
        per collection so a reader can tell a short decision from a clipped one.
        """
        payload = self._payload(row)
        raw_gates = payload.get("gates", [])
        raw_tests = payload.get("tests", [])
        if not isinstance(raw_gates, list) or not isinstance(raw_tests, list):
            raise GovernanceReadError("decision gates and tests must be arrays")

        gates = tuple(
            GateProjection(
                name=_bounded_text(item.get("name", ""), field_name="gate name"),
                satisfied=bool(item.get("satisfied", False)),
                detail=_bounded_text(item.get("detail", ""), field_name="gate detail"),
            )
            for item in raw_gates[:MAX_GATES_PROJECTED]
            if isinstance(item, dict)
        )
        tests = tuple(
            self._project_test(item)
            for item in raw_tests[:MAX_TESTS_PROJECTED]
            if isinstance(item, dict)
        )
        correction = payload.get("correction")
        correction = correction if isinstance(correction, dict) else {}
        family_size = correction.get("family_size")
        return ComparisonProjection(
            sequence=int(row["sequence"]),
            recorded_at=_decode_instant(row["recorded_at"], field_name="recorded_at"),
            recommendation=str(payload.get("recommendation", "unknown")),
            policy_identity=str(payload.get("policy_identity", "")),
            cohort_identity=str(payload.get("cohort_identity", "")),
            decided_at=str(payload.get("decided_at", "")),
            gates=gates,
            tests=tests,
            correction_method=(
                str(correction["method"]) if isinstance(correction.get("method"), str) else None
            ),
            correction_alpha=_optional_float(correction, "alpha"),
            family_size=(
                int(family_size)
                if isinstance(family_size, int) and not isinstance(family_size, bool)
                else None
            ),
            truncated_gates=len(raw_gates) > MAX_GATES_PROJECTED,
            truncated_tests=len(raw_tests) > MAX_TESTS_PROJECTED,
        )

    def _project_test(self, item: dict[str, Any]) -> TestProjection:
        """Reduce one test record, keeping an unbounded interval end as null."""
        interval = item.get("interval")
        low: float | None = None
        high: float | None = None
        if isinstance(interval, list) and len(interval) == 2:
            low = _optional_float({"low": interval[0]}, "low")
            high = _optional_float({"high": interval[1]}, "high")
        elif interval is not None:
            raise GovernanceReadError("a test interval must be a two-element array or null")
        return TestProjection(
            name=_bounded_text(item.get("name", ""), field_name="test name"),
            metric=_bounded_text(item.get("metric", ""), field_name="test metric"),
            verdict=str(item.get("verdict", "unknown")),
            point_estimate=_optional_float(item, "point_estimate"),
            interval_low=low,
            interval_high=high,
            p_value_uncorrected=_optional_float(item, "p_value_uncorrected"),
            blocks=int(item.get("blocks", 0) or 0),
            observations=int(item.get("observations", 0) or 0),
            margin=_optional_float(item, "margin"),
        )


def project_lane_states() -> Sequence[str]:
    """Return every lane state the console must be able to render.

    Exposed so a console test can assert it handles the complete set rather
    than the subset a fixture happened to contain.
    """
    return tuple(state.value for state in LaneState)


__all__ = [
    "MAX_COMPARISONS_PROJECTED",
    "MAX_EVENTS_PROJECTED",
    "MAX_GATES_PROJECTED",
    "MAX_LANES_PER_PAGE",
    "MAX_TESTS_PROJECTED",
    "ComparisonProjection",
    "GateProjection",
    "GovernanceReadError",
    "GovernanceReadPorts",
    "LaneDetail",
    "LaneEventProjection",
    "LanePage",
    "LaneSummary",
    "TestProjection",
    "project_lane_states",
]
