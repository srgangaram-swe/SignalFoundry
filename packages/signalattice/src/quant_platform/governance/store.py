"""Append-only governance persistence with a rebuildable lane-head projection.

SF-S5-SL-MR5. Governance history is evidence, so it is never rewritten. The
event table is the authority and the lane head is a cache derived from it.

**The chain is per lane and monotonic.** Each event carries the digest of its
payload, the digest of the previous event, and a chain digest over both. A
missing, reordered, or altered event breaks the chain at that point and
:meth:`GovernanceStore.verify_chain` reports where. This detects accidental
divergence and careless edits; it is *not* externally tamper-proof, because
anyone who can rewrite a row can also recompute the rest of the chain. Anchoring
belongs to the release work in #23.

**Compare-and-swap is the concurrency control.** Applying an assignment names
the generation and champion it was approved against. Two applications racing to
promote different challengers cannot both win: the loser sees a head that has
moved and is refused as stale, without sleeping, retrying, or repairing.

**Idempotency keys are stored one-way.** Only the digest of a caller's key is
persisted, so replaying the same request resolves to the existing event while a
reader of the table cannot reconstruct the key itself.

Governance evidence is retention-ineligible: these tables carry the record of
who approved what, and expiring them would destroy the audit trail the lane
exists to produce. Nothing here registers with the retention planner.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
from collections.abc import Mapping, Sequence
from contextlib import closing
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from typing import Any, Final

from quant_platform.governance.lane import (
    FreezeTrigger,
    GovernanceLane,
    GovernanceStateError,
    LaneHead,
    LaneState,
    assert_lane_transition,
)
from quant_platform.shadow.contracts import (
    ShadowValidationError,
    canonical_digest,
    utc_instant,
)

#: SQLite busy timeout. Short, because governance writes are small and a caller
#: waiting longer than this is contending with something that should be
#: investigated rather than waited out.
BUSY_TIMEOUT_MS: Final = 5_000

#: Refusal thresholds, not tuning knobs.
MAX_PAYLOAD_BYTES: Final = 1_048_576
MAX_EVENTS_PER_QUERY: Final = 100_000

#: The digest recorded as the predecessor of a lane's first event. A constant
#: rather than a zero-length string so every row has a full-width digest and the
#: schema can require one.
GENESIS_DIGEST: Final = "0" * 64


class GovernanceStoreError(ShadowValidationError):
    """Raised when the governance store cannot record or read evidence."""


class ChainIntegrityError(GovernanceStoreError):
    """Raised when the event chain does not verify.

    Distinct because it means recorded history disagrees with itself, which is
    never something a caller should handle by retrying.
    """


class StaleWriteError(GovernanceStoreError):
    """Raised when a compare-and-swap lost its race.

    Terminal for this attempt: the request must be re-evaluated against the new
    lane head, not retried against the old one.
    """


class EventKind(StrEnum):
    """The kinds of record a governance lane accumulates."""

    POLICY = "policy"
    COMPARISON = "comparison"
    REQUEST = "request"
    APPROVAL = "approval"
    ASSIGNMENT = "assignment"
    MONITORING = "monitoring"
    FREEZE = "freeze"


@dataclass(frozen=True, slots=True)
class GovernanceEvent:
    """One immutable link in a lane's chain."""

    lane_identity: str
    sequence: int
    kind: EventKind
    payload: Mapping[str, Any]
    payload_digest: str
    previous_digest: str
    chain_digest: str
    recorded_at: datetime

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-friendly record."""
        return {
            "lane_identity": self.lane_identity,
            "sequence": self.sequence,
            "kind": self.kind.value,
            "payload": dict(self.payload),
            "payload_digest": self.payload_digest,
            "previous_digest": self.previous_digest,
            "chain_digest": self.chain_digest,
            "recorded_at": self.recorded_at.isoformat(),
        }


def chain_digest(*, previous_digest: str, payload_digest: str, sequence: int, kind: str) -> str:
    """Return the chain digest binding an event to its predecessor.

    The sequence and kind are inside the digest, so an event cannot be moved to
    a different position or relabelled while keeping its links intact.
    """
    material = f"{previous_digest}:{payload_digest}:{sequence}:{kind}"
    return hashlib.sha256(material.encode("utf-8")).hexdigest()


def idempotency_digest(key: str) -> str:
    """Return the one-way digest recorded for an idempotency key.

    Raises:
        GovernanceStoreError: On an empty or oversized key.
    """
    if not isinstance(key, str) or not key.strip() or len(key) > 256:
        raise GovernanceStoreError(
            "idempotency key must be a non-empty string of at most 256 chars"
        )
    return hashlib.sha256(f"sl-governance-idempotency:{key}".encode()).hexdigest()


def _require_digest(value: object, *, field_name: str) -> str:
    """Return a validated full lowercase SHA-256 digest.

    Raises:
        GovernanceStoreError: On anything that is not one.
    """
    if (
        not isinstance(value, str)
        or len(value) != 64
        or not all(character in "0123456789abcdef" for character in value)
    ):
        raise GovernanceStoreError(f"{field_name} must be a full lowercase SHA-256 digest")
    return value


def _encode_instant(value: datetime) -> str:
    """Return the canonical UTC text form stored in the database."""
    return value.astimezone(UTC).isoformat()


def _decode_instant(value: str) -> datetime:
    """Return an aware UTC datetime from its stored text form.

    Raises:
        GovernanceStoreError: On an unparseable or naive stored value.
    """
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError as error:
        raise GovernanceStoreError(f"stored instant {value!r} is unparseable") from error
    if parsed.tzinfo is None:
        raise GovernanceStoreError(f"stored instant {value!r} is naive")
    return parsed.astimezone(UTC)


class GovernanceStore:
    """Append-only storage for one registry database's governance lanes.

    The store owns no policy. It records what it is given, refuses what would
    corrupt the chain, and rebuilds the head projection on demand.
    """

    def __init__(self, database: str | Path) -> None:
        self._path = Path(database).expanduser()

    def _connect(self) -> sqlite3.Connection:
        """Open a bounded connection with foreign keys and WAL enforced."""
        connection = sqlite3.connect(
            self._path, timeout=BUSY_TIMEOUT_MS / 1000, isolation_level=None
        )
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        connection.execute("PRAGMA journal_mode = WAL")
        connection.execute("PRAGMA synchronous = FULL")
        connection.execute(f"PRAGMA busy_timeout = {BUSY_TIMEOUT_MS}")
        return connection

    # -- lanes ------------------------------------------------------------

    def register_lane(self, lane: GovernanceLane, *, now: datetime) -> str:
        """Register a lane and its unassigned head, returning its identity.

        Registering an existing lane is a no-op rather than an error: the lane
        key is content-derived, so a second registration describes the same lane.

        Raises:
            GovernanceStoreError: On a malformed lane or instant.
        """
        if not isinstance(lane, GovernanceLane):
            raise GovernanceStoreError("lane must be a GovernanceLane")
        moment = utc_instant(now, field_name="now")
        identity = lane.identity
        with closing(self._connect()) as connection:
            connection.execute("BEGIN IMMEDIATE")
            try:
                connection.execute(
                    "INSERT OR IGNORE INTO sl_governance_lanes "
                    "(lane_identity, lane_key, created_at) VALUES (?, ?, ?)",
                    (
                        identity,
                        json.dumps(lane.to_dict(), sort_keys=True, separators=(",", ":")),
                        _encode_instant(moment),
                    ),
                )
                connection.execute(
                    "INSERT OR IGNORE INTO sl_governance_lane_head "
                    "(lane_identity, state, champion_revision, generation, freeze_trigger, "
                    "sequence, updated_at) VALUES (?, ?, NULL, 0, NULL, 0, ?)",
                    (identity, LaneState.UNASSIGNED.value, _encode_instant(moment)),
                )
                connection.execute("COMMIT")
            except sqlite3.DatabaseError:
                connection.execute("ROLLBACK")
                raise
        return identity

    def lane_head(self, lane_identity: str) -> LaneHead:
        """Return the current head projection.

        Raises:
            GovernanceStoreError: If the lane is unknown.
        """
        with closing(self._connect()) as connection:
            row = connection.execute(
                "SELECT state, champion_revision, generation, freeze_trigger "
                "FROM sl_governance_lane_head WHERE lane_identity = ?",
                (lane_identity,),
            ).fetchone()
        if row is None:
            raise GovernanceStoreError(f"lane {lane_identity[:12]} is not registered")
        return LaneHead(
            lane_identity=lane_identity,
            state=LaneState(row["state"]),
            champion_revision=row["champion_revision"],
            generation=row["generation"],
            freeze_trigger=FreezeTrigger(row["freeze_trigger"]) if row["freeze_trigger"] else None,
        )

    # -- events -----------------------------------------------------------

    def _append_locked(
        self,
        connection: sqlite3.Connection,
        lane_identity: str,
        kind: EventKind,
        payload: Mapping[str, Any],
        *,
        now: datetime,
        idempotency_key: str | None,
    ) -> GovernanceEvent:
        """Append one event inside an already-open IMMEDIATE transaction.

        Separate from :meth:`append_event` so a head update and its event can
        share one transaction. A head that moved without its event, or an event
        without its head update, is exactly the split record the chain exists to
        make impossible.
        """
        encoded = json.dumps(dict(payload), sort_keys=True, separators=(",", ":"), allow_nan=False)
        if len(encoded.encode("utf-8")) > MAX_PAYLOAD_BYTES:
            raise GovernanceStoreError(f"payload exceeds the {MAX_PAYLOAD_BYTES}-byte ceiling")
        payload_digest = canonical_digest(dict(payload))
        key_digest = idempotency_digest(idempotency_key) if idempotency_key is not None else None

        if key_digest is not None:
            existing = connection.execute(
                "SELECT sequence, kind, payload, payload_digest, previous_digest, "
                "chain_digest, recorded_at FROM sl_governance_events "
                "WHERE lane_identity = ? AND idempotency_digest = ?",
                (lane_identity, key_digest),
            ).fetchone()
            if existing is not None:
                if existing["payload_digest"] != payload_digest:
                    raise GovernanceStoreError(
                        "idempotency key was already used for different content; "
                        "reusing a key for a new payload would silently discard it"
                    )
                return self._row_to_event(lane_identity, existing)

        registered = connection.execute(
            "SELECT 1 FROM sl_governance_lanes WHERE lane_identity = ?",
            (lane_identity,),
        ).fetchone()
        if registered is None:
            raise GovernanceStoreError(f"lane {lane_identity[:12]} is not registered")

        last = connection.execute(
            "SELECT sequence, chain_digest FROM sl_governance_events "
            "WHERE lane_identity = ? ORDER BY sequence DESC LIMIT 1",
            (lane_identity,),
        ).fetchone()
        sequence = 1 if last is None else last["sequence"] + 1
        previous = GENESIS_DIGEST if last is None else last["chain_digest"]
        link = chain_digest(
            previous_digest=previous,
            payload_digest=payload_digest,
            sequence=sequence,
            kind=kind.value,
        )
        connection.execute(
            "INSERT INTO sl_governance_events (lane_identity, sequence, kind, payload, "
            "payload_digest, previous_digest, chain_digest, idempotency_digest, "
            "recorded_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                lane_identity,
                sequence,
                kind.value,
                encoded,
                payload_digest,
                previous,
                link,
                key_digest,
                _encode_instant(now),
            ),
        )
        return GovernanceEvent(
            lane_identity=lane_identity,
            sequence=sequence,
            kind=kind,
            payload=dict(payload),
            payload_digest=payload_digest,
            previous_digest=previous,
            chain_digest=link,
            recorded_at=now,
        )

    def append_event(
        self,
        lane_identity: str,
        kind: EventKind,
        payload: Mapping[str, Any],
        *,
        now: datetime,
        idempotency_key: str | None = None,
    ) -> GovernanceEvent:
        """Append one event to a lane's chain.

        Replaying the same ``idempotency_key`` within a lane returns the event
        already recorded rather than appending a duplicate. The payload is
        checked against the stored one, so the same key carrying different
        content is a conflict rather than a silent no-op.

        Raises:
            GovernanceStoreError: On an unknown lane, oversized payload, or an
                idempotency key reused with different content.
        """
        if not isinstance(kind, EventKind):
            raise GovernanceStoreError("kind must be an EventKind")
        moment = utc_instant(now, field_name="now")
        with closing(self._connect()) as connection:
            connection.execute("BEGIN IMMEDIATE")
            try:
                event = self._append_locked(
                    connection,
                    lane_identity,
                    kind,
                    payload,
                    now=moment,
                    idempotency_key=idempotency_key,
                )
                connection.execute("COMMIT")
            except (GovernanceStoreError, sqlite3.DatabaseError):
                connection.execute("ROLLBACK")
                raise
        return event

    def _row_to_event(self, lane_identity: str, row: sqlite3.Row) -> GovernanceEvent:
        """Return the event a stored row describes."""
        return GovernanceEvent(
            lane_identity=lane_identity,
            sequence=row["sequence"],
            kind=EventKind(row["kind"]),
            payload=json.loads(row["payload"]),
            payload_digest=row["payload_digest"],
            previous_digest=row["previous_digest"],
            chain_digest=row["chain_digest"],
            recorded_at=_decode_instant(row["recorded_at"]),
        )

    def load_events(self, lane_identity: str) -> tuple[GovernanceEvent, ...]:
        """Return a lane's events in chain order.

        Raises:
            GovernanceStoreError: If the lane holds more events than the query
                ceiling, which means the caller must page rather than receive a
                silently truncated history.
        """
        with closing(self._connect()) as connection:
            rows = connection.execute(
                "SELECT sequence, kind, payload, payload_digest, previous_digest, chain_digest, "
                "recorded_at FROM sl_governance_events WHERE lane_identity = ? "
                "ORDER BY sequence ASC LIMIT ?",
                (lane_identity, MAX_EVENTS_PER_QUERY + 1),
            ).fetchall()
        if len(rows) > MAX_EVENTS_PER_QUERY:
            raise GovernanceStoreError(
                f"lane holds more than {MAX_EVENTS_PER_QUERY} events; refusing to return a "
                "silently truncated history"
            )
        return tuple(self._row_to_event(lane_identity, row) for row in rows)

    def verify_chain(self, lane_identity: str) -> int:
        """Verify a lane's chain end to end and return the events checked.

        Raises:
            ChainIntegrityError: Naming the first sequence at which the chain
                does not reproduce, so a reader knows where history diverged
                rather than only that it did.
        """
        events = self.load_events(lane_identity)
        previous = GENESIS_DIGEST
        for position, event in enumerate(events, start=1):
            if event.sequence != position:
                raise ChainIntegrityError(
                    f"lane {lane_identity[:12]} has a sequence gap: expected {position}, "
                    f"found {event.sequence}"
                )
            recomputed_payload = canonical_digest(dict(event.payload))
            if recomputed_payload != event.payload_digest:
                raise ChainIntegrityError(
                    f"payload at sequence {event.sequence} does not match its recorded digest"
                )
            if event.previous_digest != previous:
                raise ChainIntegrityError(
                    f"chain breaks at sequence {event.sequence}: recorded predecessor "
                    f"{event.previous_digest[:12]} but computed {previous[:12]}"
                )
            recomputed_link = chain_digest(
                previous_digest=previous,
                payload_digest=recomputed_payload,
                sequence=event.sequence,
                kind=event.kind.value,
            )
            if recomputed_link != event.chain_digest:
                raise ChainIntegrityError(
                    f"chain digest at sequence {event.sequence} does not reproduce"
                )
            previous = event.chain_digest
        return len(events)

    # -- head projection --------------------------------------------------

    def apply_assignment(
        self,
        lane_identity: str,
        *,
        champion_revision: str,
        expected_generation: int,
        expected_champion: str | None,
        now: datetime,
        idempotency_key: str | None = None,
    ) -> LaneHead:
        """Assign a champion under compare-and-swap, returning the new head.

        The expected generation and champion are what the approval was granted
        against. If either has moved, another application won the race and this
        one is refused: it must be re-evaluated against the new champion rather
        than applied on top of it.

        The head update and its event are written in **one** transaction. A head
        that moved without recording why, or an event describing an assignment
        that never took effect, are both states no reader could reconcile.

        Raises:
            StaleWriteError: If the lane head moved since approval.
            GovernanceStateError: If the resulting transition is not permitted.
            GovernanceStoreError: On an unknown lane or malformed revision.
        """
        _require_digest(champion_revision, field_name="champion_revision")
        if expected_champion is not None:
            _require_digest(expected_champion, field_name="expected_champion")
        if isinstance(expected_generation, bool) or not isinstance(expected_generation, int):
            raise GovernanceStoreError("expected_generation must be an int")
        moment = utc_instant(now, field_name="now")

        with closing(self._connect()) as connection:
            connection.execute("BEGIN IMMEDIATE")
            try:
                row = connection.execute(
                    "SELECT state, champion_revision, generation, freeze_trigger "
                    "FROM sl_governance_lane_head WHERE lane_identity = ?",
                    (lane_identity,),
                ).fetchone()
                if row is None:
                    raise GovernanceStoreError(f"lane {lane_identity[:12]} is not registered")
                if row["generation"] != expected_generation:
                    raise StaleWriteError(
                        f"lane head is at generation {row['generation']} but the approval "
                        f"named {expected_generation}; another application won the race"
                    )
                if row["champion_revision"] != expected_champion:
                    raise StaleWriteError(
                        "lane champion is not the one this approval was granted against; "
                        "the comparison must be re-run against the current champion"
                    )
                assert_lane_transition(LaneState(row["state"]), LaneState.ACTIVE)
                generation = row["generation"] + 1
                event = self._append_locked(
                    connection,
                    lane_identity,
                    EventKind.ASSIGNMENT,
                    {
                        "champion_revision": champion_revision,
                        "generation": generation,
                        "previous_champion": expected_champion,
                        "previous_generation": expected_generation,
                    },
                    now=moment,
                    idempotency_key=idempotency_key,
                )
                updated = connection.execute(
                    "UPDATE sl_governance_lane_head SET state = ?, champion_revision = ?, "
                    "generation = ?, freeze_trigger = NULL, sequence = ?, "
                    "updated_at = ? WHERE lane_identity = ? AND generation = ?",
                    (
                        LaneState.ACTIVE.value,
                        champion_revision,
                        generation,
                        event.sequence,
                        _encode_instant(moment),
                        lane_identity,
                        expected_generation,
                    ),
                ).rowcount
                if updated != 1:
                    # The generation moved between the read and the write. The
                    # transaction is IMMEDIATE so this should be unreachable;
                    # it is checked because a silent zero-row update would
                    # commit an event describing an assignment that never
                    # happened.
                    raise StaleWriteError(
                        "compare-and-swap matched no row; the lane head moved mid-transaction"
                    )
                connection.execute("COMMIT")
            except (GovernanceStoreError, GovernanceStateError, sqlite3.DatabaseError):
                connection.execute("ROLLBACK")
                raise
        return self.lane_head(lane_identity)

    def freeze(
        self,
        lane_identity: str,
        trigger: FreezeTrigger,
        *,
        detail: str,
        now: datetime,
        idempotency_key: str | None = None,
    ) -> LaneHead:
        """Freeze a lane. Automation may do this; it may not unfreeze.

        Freezing preserves the champion so a reader can still see which model
        the lane was running when it stopped.

        Raises:
            GovernanceStateError: If the lane is not active.
            GovernanceStoreError: On an unknown lane or malformed detail.
        """
        if not isinstance(trigger, FreezeTrigger):
            raise GovernanceStoreError("trigger must be a FreezeTrigger")
        if not isinstance(detail, str) or not detail.strip() or len(detail) > 512:
            raise GovernanceStoreError("freeze detail must be a non-empty string under 512 chars")
        moment = utc_instant(now, field_name="now")

        with closing(self._connect()) as connection:
            connection.execute("BEGIN IMMEDIATE")
            try:
                row = connection.execute(
                    "SELECT state FROM sl_governance_lane_head WHERE lane_identity = ?",
                    (lane_identity,),
                ).fetchone()
                if row is None:
                    raise GovernanceStoreError(f"lane {lane_identity[:12]} is not registered")
                assert_lane_transition(LaneState(row["state"]), LaneState.FROZEN)
                event = self._append_locked(
                    connection,
                    lane_identity,
                    EventKind.FREEZE,
                    {"trigger": trigger.value, "detail": detail},
                    now=moment,
                    idempotency_key=idempotency_key,
                )
                connection.execute(
                    "UPDATE sl_governance_lane_head SET state = ?, freeze_trigger = ?, "
                    "sequence = ?, updated_at = ? WHERE lane_identity = ?",
                    (
                        LaneState.FROZEN.value,
                        trigger.value,
                        event.sequence,
                        _encode_instant(moment),
                        lane_identity,
                    ),
                )
                connection.execute("COMMIT")
            except (GovernanceStoreError, GovernanceStateError, sqlite3.DatabaseError):
                connection.execute("ROLLBACK")
                raise
        return self.lane_head(lane_identity)

    def rebuild_head(self, lane_identity: str) -> LaneHead:
        """Recompute the head from the event chain and compare it to the cache.

        The chain is verified first: a projection rebuilt from a broken chain
        would launder the break into a plausible-looking head.

        Raises:
            ChainIntegrityError: If the chain does not verify, or if the stored
                projection disagrees with the one the events produce.
        """
        self.verify_chain(lane_identity)
        events = self.load_events(lane_identity)
        state = LaneState.UNASSIGNED
        champion: str | None = None
        generation = 0
        trigger: FreezeTrigger | None = None
        for event in events:
            if event.kind is EventKind.ASSIGNMENT:
                state = LaneState.ACTIVE
                champion = str(event.payload["champion_revision"])
                generation = int(event.payload["generation"])
                trigger = None
            elif event.kind is EventKind.FREEZE:
                state = LaneState.FROZEN
                trigger = FreezeTrigger(event.payload["trigger"])
        rebuilt = LaneHead(
            lane_identity=lane_identity,
            state=state,
            champion_revision=champion,
            generation=generation,
            freeze_trigger=trigger,
        )
        stored = self.lane_head(lane_identity)
        if rebuilt.to_dict() != stored.to_dict():
            raise ChainIntegrityError(
                "the stored lane head disagrees with the one its events produce: "
                f"stored {stored.to_dict()}, rebuilt {rebuilt.to_dict()}. The events are "
                "the authority; the projection must be repaired from them, and the "
                "divergence investigated rather than overwritten."
            )
        return rebuilt

    def events_of_kind(self, lane_identity: str, kind: EventKind) -> tuple[GovernanceEvent, ...]:
        """Return a lane's events of one kind, in chain order."""
        return tuple(event for event in self.load_events(lane_identity) if event.kind is kind)

    def summary(self, lane_identity: str) -> dict[str, Any]:
        """Return a bounded, JSON-friendly description of a lane."""
        events = self.load_events(lane_identity)
        head = self.lane_head(lane_identity)
        counts: dict[str, int] = {}
        for event in events:
            counts[event.kind.value] = counts.get(event.kind.value, 0) + 1
        return {
            "head": head.to_dict(),
            "events": len(events),
            "events_by_kind": dict(sorted(counts.items())),
            "chain_tip": events[-1].chain_digest if events else GENESIS_DIGEST,
            "note": (
                "The event chain is the authority and the head is a projection of it. "
                "Local hash chains detect accidental divergence; they are not externally "
                "tamper-proof."
            ),
        }


def replay_head(events: Sequence[GovernanceEvent], *, lane_identity: str) -> LaneHead:
    """Return the head a sequence of events produces, without touching storage.

    Exposed separately so the projection logic can be tested against
    hand-built chains, including ones no store would ever have written.

    Raises:
        ShadowValidationError: If the events do not produce a coherent head.
    """
    state = LaneState.UNASSIGNED
    champion: str | None = None
    generation = 0
    trigger: FreezeTrigger | None = None
    for event in events:
        if event.kind is EventKind.ASSIGNMENT:
            state = LaneState.ACTIVE
            champion = str(event.payload["champion_revision"])
            generation = int(event.payload["generation"])
            trigger = None
        elif event.kind is EventKind.FREEZE:
            state = LaneState.FROZEN
            trigger = FreezeTrigger(event.payload["trigger"])
    return LaneHead(
        lane_identity=lane_identity,
        state=state,
        champion_revision=champion,
        generation=generation,
        freeze_trigger=trigger,
    )


__all__ = [
    "BUSY_TIMEOUT_MS",
    "GENESIS_DIGEST",
    "MAX_EVENTS_PER_QUERY",
    "MAX_PAYLOAD_BYTES",
    "ChainIntegrityError",
    "EventKind",
    "GovernanceEvent",
    "GovernanceStore",
    "GovernanceStoreError",
    "StaleWriteError",
    "chain_digest",
    "idempotency_digest",
    "replay_head",
]
