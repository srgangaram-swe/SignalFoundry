"""Transactional session state with restart recovery that refuses bad snapshots.

SF-S5-MR4. MR3 made a replayed decision cycle a no-op *within one process*: the
adapter held submitted client order IDs in a dict. A process restart empties that
dict, and the next cycle resubmits every order it had already sent. This module
is what makes the idempotency survive the restart.

Three properties carry the weight:

**Writes are atomic or absent.** A snapshot is written inside one
``BEGIN IMMEDIATE`` transaction with ``synchronous=FULL``. A crash mid-write
leaves the previous snapshot intact rather than a half-updated one, and the
partial-write tests assert exactly that.

**Recovery refuses rather than guesses.** A snapshot that is tampered, partial,
schema-incompatible, stale beyond a declared bound, or ambiguous (two heads at
the same sequence) raises. The alternative — loading the best available
interpretation — resumes trading against a position book nobody has verified.

**Clock rollback is detected, not tolerated.** The sequence is monotonic and
independent of wall time, and a snapshot stamped in the future relative to the
recovering process is refused. Wall clocks move backwards on NTP correction, and
"latest by timestamp" silently picks the wrong snapshot when they do.

The store persists *intent* — what the system decided and what it believes the
broker did. It is never the authority on broker state; that is
:mod:`alphaforge.broker.reconciliation`, which compares the two and halts when
they disagree.
"""

from __future__ import annotations

import hashlib
import json
import os
import sqlite3
import stat
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from pathlib import Path
from types import TracebackType
from typing import Any, Final, Self

from alphaforge.broker.contracts import (
    BrokerContractError,
    OrderState,
    Position,
    utc_timestamp,
    validate_client_order_id,
    validate_symbol,
)

#: Schema version. A snapshot written under a different version is refused
#: rather than migrated: a silent migration of trading state is how a field
#: acquires a new meaning without anyone deciding that it should.
SCHEMA_VERSION: Final = 1

#: Refusal thresholds, not tuning knobs.
MAX_TRACKED_INTENTS: Final = 100_000
MAX_SNAPSHOT_BYTES: Final = 32 * 1024 * 1024
MAX_POSITIONS_PER_SNAPSHOT: Final = 5_000
MAX_IDENTIFIER_BYTES: Final = 256

#: Default staleness bound for a recovered snapshot. A day-old view of the book
#: cannot authorize an order; the operator must reconcile first.
DEFAULT_MAX_SNAPSHOT_AGE: Final = timedelta(hours=12)


class DurableStateError(BrokerContractError):
    """Raised when session state cannot be persisted or trusted."""


class SnapshotIntegrityError(DurableStateError):
    """Raised when a stored snapshot fails its integrity check.

    Distinct from a merely stale or incompatible snapshot: this one has been
    modified, truncated, or corrupted since it was written, and no part of it
    may be trusted.
    """


class SnapshotIncompatibleError(DurableStateError):
    """Raised when a snapshot was written under a different schema version."""


class SnapshotStaleError(DurableStateError):
    """Raised when a snapshot is too old to authorize action."""


class ClockRollbackError(DurableStateError):
    """Raised when the recovering process's clock precedes the snapshot's."""


def _canonical_bytes(payload: Any) -> bytes:
    """Return deterministic JSON bytes for hashing."""
    return json.dumps(
        payload, sort_keys=True, separators=(",", ":"), allow_nan=False, ensure_ascii=True
    ).encode("utf-8")


def _digest(payload: Any) -> str:
    """Return the SHA-256 of a canonical payload."""
    return hashlib.sha256(_canonical_bytes(payload)).hexdigest()


def _identifier(value: object, *, field_name: str) -> str:
    """Return a bounded non-empty identifier."""
    if not isinstance(value, str) or not value.strip() or value != value.strip():
        raise DurableStateError(f"{field_name} must be a non-empty unpadded string")
    if len(value.encode("utf-8")) > MAX_IDENTIFIER_BYTES:
        raise DurableStateError(f"{field_name} exceeds {MAX_IDENTIFIER_BYTES} bytes")
    return value


def _decimal_text(value: object, *, field_name: str) -> str:
    """Return a Decimal serialized as text, refusing floats."""
    if isinstance(value, float):
        raise DurableStateError(
            f"{field_name} must not be a float; persisted money is Decimal text so a "
            "restart reads back exactly what was written"
        )
    if isinstance(value, bool) or not isinstance(value, (int, str, Decimal)):
        raise DurableStateError(f"{field_name} must be a Decimal, int, or str")
    amount = Decimal(value)
    if not amount.is_finite():
        raise DurableStateError(f"{field_name} must be finite")
    return str(amount)


@dataclass(frozen=True, slots=True)
class OrderIntent:
    """One decision to place an order, and what became of it.

    The record exists from the moment the decision is made — *before* the broker
    is contacted — so a crash between "decided" and "acknowledged" leaves
    evidence that the order may exist. Recording only after acknowledgement
    would make that window invisible, and it is the window in which duplicates
    are born.
    """

    client_order_id: str
    decision_id: str
    symbol: str
    side: str
    quantity: str
    submitted_state: str
    recorded_at: datetime
    broker_order_id: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "client_order_id", validate_client_order_id(self.client_order_id))
        object.__setattr__(
            self, "decision_id", _identifier(self.decision_id, field_name="decision_id")
        )
        object.__setattr__(self, "symbol", validate_symbol(self.symbol))
        if self.side not in ("buy", "sell"):
            raise DurableStateError(f"side must be 'buy' or 'sell', got {self.side!r}")
        object.__setattr__(self, "quantity", _decimal_text(self.quantity, field_name="quantity"))
        if self.submitted_state not in {state.value for state in OrderState}:
            raise DurableStateError(f"unknown order state {self.submitted_state!r}")
        object.__setattr__(
            self, "recorded_at", utc_timestamp(self.recorded_at, field_name="recorded_at")
        )
        if self.broker_order_id is not None:
            object.__setattr__(
                self,
                "broker_order_id",
                _identifier(self.broker_order_id, field_name="broker_order_id"),
            )

    @property
    def is_terminal(self) -> bool:
        """Whether the recorded state admits no further update."""
        return self.submitted_state in {
            OrderState.FILLED.value,
            OrderState.CANCELED.value,
            OrderState.REJECTED.value,
            OrderState.EXPIRED.value,
        }

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-friendly record."""
        return {
            "client_order_id": self.client_order_id,
            "decision_id": self.decision_id,
            "symbol": self.symbol,
            "side": self.side,
            "quantity": self.quantity,
            "submitted_state": self.submitted_state,
            "recorded_at": self.recorded_at.isoformat(),
            "broker_order_id": self.broker_order_id,
        }


@dataclass(frozen=True)
class SessionSnapshot:
    """A complete, self-describing view of session state at one sequence point.

    ``content_hash`` covers every field below it. ``previous_hash`` chains to the
    prior snapshot, so a snapshot deleted from the middle of the history is
    detectable — not merely a modified one.
    """

    sequence: int
    schema_version: int
    strategy_id: str
    model_version: str
    config_identity: str
    written_at: datetime
    cash: str
    positions: tuple[Position, ...]
    intents: tuple[OrderIntent, ...]
    previous_hash: str
    content_hash: str

    def __post_init__(self) -> None:
        if isinstance(self.sequence, bool) or not isinstance(self.sequence, int):
            raise DurableStateError("sequence must be an int")
        if self.sequence < 1:
            raise DurableStateError("sequence must be positive")
        if self.schema_version != SCHEMA_VERSION:
            raise SnapshotIncompatibleError(
                f"snapshot schema version {self.schema_version} does not match the running "
                f"version {SCHEMA_VERSION}; trading state is never silently migrated"
            )
        for field_name in ("strategy_id", "model_version", "config_identity"):
            object.__setattr__(
                self, field_name, _identifier(getattr(self, field_name), field_name=field_name)
            )
        object.__setattr__(self, "cash", _decimal_text(self.cash, field_name="cash"))
        object.__setattr__(
            self, "written_at", utc_timestamp(self.written_at, field_name="written_at")
        )
        positions = tuple(self.positions)
        if len(positions) > MAX_POSITIONS_PER_SNAPSHOT:
            raise DurableStateError(
                f"snapshot exceeds the {MAX_POSITIONS_PER_SNAPSHOT}-position ceiling"
            )
        symbols = [item.symbol for item in positions]
        if len(set(symbols)) != len(symbols):
            raise DurableStateError("duplicate symbol in snapshot positions")
        object.__setattr__(self, "positions", tuple(sorted(positions, key=lambda p: p.symbol)))
        intents = tuple(self.intents)
        if len(intents) > MAX_TRACKED_INTENTS:
            raise DurableStateError(f"snapshot exceeds the {MAX_TRACKED_INTENTS}-intent ceiling")
        ids = [item.client_order_id for item in intents]
        if len(set(ids)) != len(ids):
            raise DurableStateError(
                "duplicate client_order_id in snapshot intents; two records for one order "
                "would let a replay resubmit it"
            )
        object.__setattr__(self, "intents", tuple(sorted(intents, key=lambda i: i.client_order_id)))
        for field_name in ("previous_hash", "content_hash"):
            value = getattr(self, field_name)
            if not isinstance(value, str) or len(value) != 64:
                raise DurableStateError(f"{field_name} must be a full SHA-256 hex digest")

    def hashable_payload(self) -> dict[str, Any]:
        """Return exactly the fields the content hash covers."""
        return {
            "sequence": self.sequence,
            "schema_version": self.schema_version,
            "strategy_id": self.strategy_id,
            "model_version": self.model_version,
            "config_identity": self.config_identity,
            "written_at": self.written_at.isoformat(),
            "cash": self.cash,
            "positions": [item.to_dict() for item in self.positions],
            "intents": [item.to_dict() for item in self.intents],
            "previous_hash": self.previous_hash,
        }

    def verify_integrity(self) -> None:
        """Recompute the content hash and refuse a mismatch.

        Raises:
            SnapshotIntegrityError: If the recomputed hash differs.
        """
        recomputed = _digest(self.hashable_payload())
        if recomputed != self.content_hash:
            raise SnapshotIntegrityError(
                f"snapshot {self.sequence} failed its integrity check: stored "
                f"{self.content_hash[:12]}, recomputed {recomputed[:12]}. The record has been "
                "modified since it was written and no part of it may be trusted."
            )

    def intent_for(self, client_order_id: str) -> OrderIntent | None:
        """Return the recorded intent for an order, or ``None``."""
        target = validate_client_order_id(client_order_id)
        for item in self.intents:
            if item.client_order_id == target:
                return item
        return None

    def open_intents(self) -> tuple[OrderIntent, ...]:
        """Return intents that are not in a terminal state."""
        return tuple(item for item in self.intents if not item.is_terminal)

    def to_dict(self) -> dict[str, Any]:
        """Return the complete JSON-friendly record including its hash."""
        payload = self.hashable_payload()
        payload["content_hash"] = self.content_hash
        return payload


def _validate_store_path(path: str | os.PathLike[str]) -> str:
    """Refuse an unsafe store path, matching the journal's posture.

    Raises:
        DurableStateError: On a symlink, a non-regular file, a hard-linked
            alias, foreign ownership, or group/other-readable permissions.
    """
    resolved = Path(path).expanduser()
    if resolved.exists():
        info = resolved.lstat()
        if stat.S_ISLNK(info.st_mode) or not stat.S_ISREG(info.st_mode):
            raise DurableStateError("state store must be a regular file, not a symlink")
        if info.st_nlink != 1:
            raise DurableStateError("state store must not have hard-link aliases")
        if hasattr(os, "getuid") and info.st_uid != os.getuid():
            raise DurableStateError("state store must be owned by the current user")
        if stat.S_IMODE(info.st_mode) & 0o077:
            raise DurableStateError("state store permissions must be owner-only")
    return str(resolved)


class SessionStateStore:
    """Durable, hash-chained session state backed by hardened SQLite.

    Opened on one thread and used from that thread. Writes serialize under
    ``BEGIN IMMEDIATE`` with ``synchronous=FULL``, so a crash mid-write leaves
    the previous snapshot intact.

    Raises:
        DurableStateError: On an unsafe path or a database that cannot be
            opened in the required mode.
    """

    def __init__(self, path: str | os.PathLike[str], *, busy_timeout_ms: int = 5_000) -> None:
        if isinstance(busy_timeout_ms, bool) or not isinstance(busy_timeout_ms, int):
            raise DurableStateError("busy_timeout_ms must be an int")
        if not 0 <= busy_timeout_ms <= 60_000:
            raise DurableStateError("busy_timeout_ms must lie in [0, 60000]")
        self._path = _validate_store_path(path)
        self._closed = False
        self._connection = sqlite3.connect(
            self._path,
            timeout=busy_timeout_ms / 1000.0,
            isolation_level=None,
            check_same_thread=True,
        )
        try:
            self._connection.execute(f"PRAGMA busy_timeout = {busy_timeout_ms}")
            self._connection.execute("PRAGMA trusted_schema = OFF")
            mode = self._connection.execute("PRAGMA journal_mode = WAL").fetchone()
            if mode is None or str(mode[0]).casefold() != "wal":
                raise DurableStateError("state store could not enable WAL mode")
            self._connection.execute("PRAGMA synchronous = FULL")
            self._connection.execute("""
                CREATE TABLE IF NOT EXISTS snapshots (
                    sequence INTEGER PRIMARY KEY,
                    schema_version INTEGER NOT NULL,
                    payload TEXT NOT NULL,
                    content_hash TEXT NOT NULL UNIQUE,
                    previous_hash TEXT NOT NULL
                )
                """)
        except Exception:
            self._connection.close()
            raise
        with self._suppress_chmod_error():
            os.chmod(self._path, 0o600)

    @staticmethod
    def _suppress_chmod_error() -> Any:
        from contextlib import suppress

        return suppress(OSError)

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.close()

    def close(self) -> None:
        """Close the connection. Idempotent."""
        if not self._closed:
            self._connection.close()
            self._closed = True

    def _require_open(self) -> None:
        if self._closed:
            raise DurableStateError("state store is closed")

    def latest_sequence(self) -> int:
        """Return the highest stored sequence, or 0 when empty."""
        self._require_open()
        row = self._connection.execute("SELECT MAX(sequence) FROM snapshots").fetchone()
        return int(row[0]) if row and row[0] is not None else 0

    def append(
        self,
        *,
        strategy_id: str,
        model_version: str,
        config_identity: str,
        cash: Decimal | int | str,
        positions: Sequence[Position],
        intents: Sequence[OrderIntent],
        written_at: datetime,
    ) -> SessionSnapshot:
        """Append one snapshot atomically and return it.

        The whole write is one transaction. A crash before it commits leaves the
        previous snapshot as the head; a crash after leaves the new one. There is
        no intermediate state in which the head is half-written.

        Raises:
            DurableStateError: On oversized payloads or a closed store.
            ClockRollbackError: If ``written_at`` precedes the current head.
        """
        self._require_open()
        moment = utc_timestamp(written_at, field_name="written_at")
        head = self._read_head()
        sequence = 1 if head is None else head.sequence + 1
        previous_hash = "0" * 64 if head is None else head.content_hash
        if head is not None and moment < head.written_at:
            raise ClockRollbackError(
                f"refusing to append a snapshot stamped {moment.isoformat()} before the "
                f"current head {head.written_at.isoformat()}. A wall clock that moved "
                "backwards makes 'latest by timestamp' pick the wrong record."
            )

        draft = {
            "sequence": sequence,
            "schema_version": SCHEMA_VERSION,
            "strategy_id": _identifier(strategy_id, field_name="strategy_id"),
            "model_version": _identifier(model_version, field_name="model_version"),
            "config_identity": _identifier(config_identity, field_name="config_identity"),
            "written_at": moment.isoformat(),
            "cash": _decimal_text(cash, field_name="cash"),
            "positions": [item.to_dict() for item in sorted(positions, key=lambda p: p.symbol)],
            "intents": [
                item.to_dict() for item in sorted(intents, key=lambda i: i.client_order_id)
            ],
            "previous_hash": previous_hash,
        }
        content_hash = _digest(draft)
        encoded = _canonical_bytes(draft)
        if len(encoded) > MAX_SNAPSHOT_BYTES:
            raise DurableStateError(
                f"snapshot payload is {len(encoded)} bytes, exceeding the "
                f"{MAX_SNAPSHOT_BYTES}-byte ceiling"
            )

        try:
            self._connection.execute("BEGIN IMMEDIATE")
            self._connection.execute(
                "INSERT INTO snapshots (sequence, schema_version, payload, content_hash, "
                "previous_hash) VALUES (?, ?, ?, ?, ?)",
                (sequence, SCHEMA_VERSION, encoded.decode("utf-8"), content_hash, previous_hash),
            )
            self._connection.execute("COMMIT")
        except Exception:
            with self._suppress_chmod_error():
                self._connection.execute("ROLLBACK")
            raise

        return _snapshot_from_payload(draft, content_hash=content_hash)

    def _read_head(self) -> SessionSnapshot | None:
        """Return the highest-sequence snapshot without verifying its chain."""
        row = self._connection.execute(
            "SELECT payload, content_hash FROM snapshots ORDER BY sequence DESC LIMIT 1"
        ).fetchone()
        if row is None:
            return None
        payload = json.loads(row[0])
        return _snapshot_from_payload(payload, content_hash=str(row[1]))

    def recover(
        self,
        *,
        now: datetime,
        expected_strategy_id: str,
        expected_config_identity: str,
        max_age: timedelta = DEFAULT_MAX_SNAPSHOT_AGE,
    ) -> SessionSnapshot | None:
        """Return the verified head snapshot, or ``None`` when the store is empty.

        Every refusal below is a deliberate stop rather than a best-effort load:
        resuming from a snapshot nobody has verified means trading against a
        position book nobody has verified.

        Raises:
            SnapshotIntegrityError: On a hash mismatch or a broken chain.
            SnapshotIncompatibleError: On a schema-version mismatch, or when the
                snapshot belongs to a different strategy or configuration.
            SnapshotStaleError: When the head exceeds ``max_age``.
            ClockRollbackError: When ``now`` precedes the head's timestamp.
            DurableStateError: On an ambiguous store.
        """
        self._require_open()
        moment = utc_timestamp(now, field_name="now")
        if max_age <= timedelta(0):
            raise DurableStateError("max_age must be positive")

        rows = self._connection.execute(
            "SELECT sequence, payload, content_hash, previous_hash FROM snapshots "
            "ORDER BY sequence ASC"
        ).fetchall()
        if not rows:
            return None

        sequences = [int(row[0]) for row in rows]
        if len(set(sequences)) != len(sequences):
            raise DurableStateError(
                "ambiguous store: duplicate sequence numbers, so there is no single head"
            )
        if sequences != list(range(sequences[0], sequences[0] + len(sequences))):
            raise SnapshotIntegrityError(
                f"snapshot sequence has a gap: {sequences[:8]}... A removed record is not "
                "recoverable from the remaining ones."
            )

        previous_hash = None
        head: SessionSnapshot | None = None
        for row in rows:
            payload = json.loads(row[1])
            snapshot = _snapshot_from_payload(payload, content_hash=str(row[2]))
            snapshot.verify_integrity()
            if previous_hash is not None and snapshot.previous_hash != previous_hash:
                raise SnapshotIntegrityError(
                    f"snapshot {snapshot.sequence} does not chain to its predecessor: "
                    f"expects {snapshot.previous_hash[:12]}, found {previous_hash[:12]}"
                )
            previous_hash = snapshot.content_hash
            head = snapshot

        assert head is not None  # rows was non-empty
        if head.strategy_id != expected_strategy_id:
            raise SnapshotIncompatibleError(
                f"snapshot belongs to strategy {head.strategy_id!r}, not "
                f"{expected_strategy_id!r}; recovering another strategy's positions would "
                "attribute them to this one"
            )
        if head.config_identity != expected_config_identity:
            raise SnapshotIncompatibleError(
                f"snapshot was written under configuration {head.config_identity[:12]}, "
                f"running {expected_config_identity[:12]}. The configuration that produced "
                "these positions is not the one about to act on them."
            )
        if moment < head.written_at:
            raise ClockRollbackError(
                f"current time {moment.isoformat()} precedes the snapshot's "
                f"{head.written_at.isoformat()}; the clock moved backwards and staleness "
                "cannot be judged"
            )
        age = moment - head.written_at
        if age > max_age:
            raise SnapshotStaleError(
                f"head snapshot is {age} old, exceeding {max_age}. A stale view of the book "
                "cannot authorize an order; reconcile against the broker first."
            )
        return head

    def verify_chain(self) -> int:
        """Verify every snapshot and return how many were checked.

        Raises:
            SnapshotIntegrityError: On the first record that fails.
        """
        self._require_open()
        rows = self._connection.execute(
            "SELECT payload, content_hash FROM snapshots ORDER BY sequence ASC"
        ).fetchall()
        previous_hash = None
        for row in rows:
            snapshot = _snapshot_from_payload(json.loads(row[0]), content_hash=str(row[1]))
            snapshot.verify_integrity()
            if previous_hash is not None and snapshot.previous_hash != previous_hash:
                raise SnapshotIntegrityError(f"snapshot {snapshot.sequence} breaks the hash chain")
            previous_hash = snapshot.content_hash
        return len(rows)


def _snapshot_from_payload(payload: Mapping[str, Any], *, content_hash: str) -> SessionSnapshot:
    """Rebuild a snapshot from its stored payload."""
    if not isinstance(payload, Mapping):
        raise SnapshotIntegrityError("snapshot payload is not an object")
    missing = {
        "sequence",
        "schema_version",
        "strategy_id",
        "model_version",
        "config_identity",
        "written_at",
        "cash",
        "positions",
        "intents",
        "previous_hash",
    } - set(payload)
    if missing:
        raise SnapshotIntegrityError(
            f"snapshot payload is missing {sorted(missing)}; a truncated record is not a "
            "partially usable one"
        )
    positions = tuple(
        Position(
            symbol=item["symbol"],
            quantity=Decimal(item["quantity"]),
            average_entry_price=Decimal(item["average_entry_price"]),
        )
        for item in payload["positions"]
    )
    intents = tuple(
        OrderIntent(
            client_order_id=item["client_order_id"],
            decision_id=item["decision_id"],
            symbol=item["symbol"],
            side=item["side"],
            quantity=item["quantity"],
            submitted_state=item["submitted_state"],
            recorded_at=datetime.fromisoformat(item["recorded_at"]).astimezone(UTC),
            broker_order_id=item.get("broker_order_id"),
        )
        for item in payload["intents"]
    )
    return SessionSnapshot(
        sequence=int(payload["sequence"]),
        schema_version=int(payload["schema_version"]),
        strategy_id=str(payload["strategy_id"]),
        model_version=str(payload["model_version"]),
        config_identity=str(payload["config_identity"]),
        written_at=datetime.fromisoformat(str(payload["written_at"])).astimezone(UTC),
        cash=str(payload["cash"]),
        positions=positions,
        intents=intents,
        previous_hash=str(payload["previous_hash"]),
        content_hash=content_hash,
    )


__all__ = [
    "DEFAULT_MAX_SNAPSHOT_AGE",
    "MAX_SNAPSHOT_BYTES",
    "MAX_TRACKED_INTENTS",
    "SCHEMA_VERSION",
    "ClockRollbackError",
    "DurableStateError",
    "OrderIntent",
    "SessionSnapshot",
    "SessionStateStore",
    "SnapshotIncompatibleError",
    "SnapshotIntegrityError",
    "SnapshotStaleError",
]
