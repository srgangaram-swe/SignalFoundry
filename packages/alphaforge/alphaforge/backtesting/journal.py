"""Bounded, tamper-evident journals for deterministic event replay.

The journal stores the canonical representation supplied by
``ExecutionEvent.canonical_bytes``.  Each record is chained to its predecessor
with SHA-256, so restart verification detects reordered, removed, modified, or
injected records before replay begins.  The SQLite implementation is a local
research persistence boundary: it does not provide Byzantine integrity or
protection from an attacker who can replace the database and every trusted
reference to it.
"""

from __future__ import annotations

import hashlib
import os
import sqlite3
import stat
import threading
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path
from types import TracebackType
from typing import TYPE_CHECKING, Protocol, Self, runtime_checkable

if TYPE_CHECKING:
    from alphaforge.execution.events import ExecutionEvent


_SCHEMA_VERSION = 1
_APPLICATION_ID = 0x41464A52  # "AFJR"
_HASH_ALGORITHM = "sha256-chain-v1"
_HASH_DOMAIN = b"alphaforge.execution-journal.v1\x00"
_GENESIS_HASH = bytes(hashlib.sha256().digest_size)
_HASH_SIZE = len(_GENESIS_HASH)

_DEFAULT_MAX_EVENTS = 1_000_000
_DEFAULT_MAX_EVENT_BYTES = 4 * 1024 * 1024
_DEFAULT_MAX_TOTAL_PAYLOAD_BYTES = 512 * 1024 * 1024
_DEFAULT_MAX_DATABASE_BYTES = 1024 * 1024 * 1024
_DEFAULT_MAX_EVENT_ID_BYTES = 512
_DEFAULT_BUSY_TIMEOUT_MS = 5_000

_HARD_MAX_EVENTS = 10_000_000
_HARD_MAX_EVENT_BYTES = 64 * 1024 * 1024
_HARD_MAX_TOTAL_PAYLOAD_BYTES = 16 * 1024 * 1024 * 1024
_HARD_MAX_DATABASE_BYTES = 64 * 1024 * 1024 * 1024
_HARD_MAX_EVENT_ID_BYTES = 4_096
_MIN_DATABASE_BYTES = 128 * 1024
_MAX_BUSY_TIMEOUT_MS = 60_000
_MAX_PATH_BYTES = 4_096


class JournalError(RuntimeError):
    """Base class for journal failures."""


class JournalClosedError(JournalError):
    """Raised when a closed journal is accessed."""


class JournalPathError(JournalError):
    """Raised when a durable journal path violates the local security policy."""


class JournalSchemaError(JournalError):
    """Raised when a durable journal has an unsupported or malformed schema."""


class JournalIntegrityError(JournalError):
    """Raised when canonical data or its hash chain fails verification."""


class JournalCollisionError(JournalError):
    """Raised when an event identifier is reused with different canonical bytes."""


class JournalOrderingError(JournalError):
    """Raised when an append would move backward in logical event order."""


class JournalResourceLimitError(JournalError):
    """Raised before an operation would exceed a configured resource limit."""


class JournalBusyError(JournalError):
    """Raised when another writer prevents a bounded SQLite transaction."""


@dataclass(frozen=True, slots=True)
class JournalHead:
    """Verified journal position.

    ``event_hash`` is the 32-byte SHA-256 chain head.  An empty journal uses a
    fixed all-zero genesis value rather than the digest of an implicit event.
    """

    count: int
    event_hash: bytes

    def __post_init__(self) -> None:
        if isinstance(self.count, bool) or not isinstance(self.count, int) or self.count < 0:
            raise ValueError("journal head count must be a non-negative integer")
        if not isinstance(self.event_hash, bytes) or len(self.event_hash) != _HASH_SIZE:
            raise ValueError(f"journal head hash must contain exactly {_HASH_SIZE} bytes")


@runtime_checkable
class Journal(Protocol):
    """Minimal persistence contract consumed by the event engine.

    ``append`` returns ``True`` only for a newly stored event.  Re-appending the
    exact same identifier and canonical bytes is an idempotent no-op returning
    ``False``; identifier reuse with a different body is an error.
    """

    @property
    def count(self) -> int:
        """Return the number of committed events."""

    @property
    def head_hash(self) -> bytes:
        """Return the current SHA-256 chain head."""

    def append(self, event: ExecutionEvent) -> bool:
        """Atomically append ``event``, or report an exact duplicate."""

    def events(self) -> tuple[ExecutionEvent, ...]:
        """Verify and decode every event in deterministic ordinal order."""

    def export_canonical(self) -> tuple[bytes, ...]:
        """Verify and export canonical event bytes in deterministic order."""

    def verify(self) -> JournalHead:
        """Verify the complete journal and return its trusted head."""

    def close(self) -> None:
        """Release resources; repeated calls are safe."""

    def __enter__(self) -> Self:
        """Return an open context-managed journal."""

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        """Close the journal without suppressing exceptions."""


@dataclass(frozen=True, slots=True)
class _Limits:
    max_events: int
    max_event_bytes: int
    max_total_payload_bytes: int
    max_event_id_bytes: int
    max_database_bytes: int | None = None


@dataclass(frozen=True, slots=True)
class _Record:
    ordinal: int
    event_id: str
    payload: bytes
    previous_hash: bytes
    event_hash: bytes


def _bounded_integer(name: str, value: object, *, minimum: int, maximum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer")
    if not minimum <= value <= maximum:
        raise ValueError(f"{name} must be in [{minimum}, {maximum}]")
    return value


def _limits(
    *,
    max_events: int,
    max_event_bytes: int,
    max_total_payload_bytes: int,
    max_event_id_bytes: int,
    max_database_bytes: int | None = None,
) -> _Limits:
    events = _bounded_integer("max_events", max_events, minimum=1, maximum=_HARD_MAX_EVENTS)
    event_bytes = _bounded_integer(
        "max_event_bytes", max_event_bytes, minimum=1, maximum=_HARD_MAX_EVENT_BYTES
    )
    total_bytes = _bounded_integer(
        "max_total_payload_bytes",
        max_total_payload_bytes,
        minimum=event_bytes,
        maximum=_HARD_MAX_TOTAL_PAYLOAD_BYTES,
    )
    event_id_bytes = _bounded_integer(
        "max_event_id_bytes",
        max_event_id_bytes,
        minimum=1,
        maximum=_HARD_MAX_EVENT_ID_BYTES,
    )
    database_bytes = None
    if max_database_bytes is not None:
        database_bytes = _bounded_integer(
            "max_database_bytes",
            max_database_bytes,
            minimum=_MIN_DATABASE_BYTES,
            maximum=_HARD_MAX_DATABASE_BYTES,
        )
        if total_bytes > database_bytes:
            raise ValueError("max_total_payload_bytes cannot exceed max_database_bytes")
    return _Limits(
        max_events=events,
        max_event_bytes=event_bytes,
        max_total_payload_bytes=total_bytes,
        max_event_id_bytes=event_id_bytes,
        max_database_bytes=database_bytes,
    )


def _event_id(event: object, *, maximum_bytes: int) -> str:
    value = getattr(event, "event_id", None)
    if not isinstance(value, str):
        raise JournalIntegrityError("event_id must be a string")
    try:
        encoded = value.encode("utf-8", errors="strict")
    except UnicodeError as exc:
        raise JournalIntegrityError("event_id must be valid UTF-8") from exc
    if not encoded or len(encoded) > maximum_bytes:
        raise JournalResourceLimitError(
            f"event_id must contain between 1 and {maximum_bytes} UTF-8 bytes"
        )
    if any(character < " " or character == "\x7f" for character in value):
        raise JournalIntegrityError("event_id must not contain control characters")
    return value


def _decode_event(payload: bytes) -> ExecutionEvent:
    try:
        from alphaforge.execution.events import ExecutionEvent
    except ImportError as exc:  # pragma: no cover - integration packaging guard
        raise JournalIntegrityError("execution event decoder is unavailable") from exc
    try:
        event = ExecutionEvent.from_canonical_bytes(payload)
    except (TypeError, ValueError, UnicodeError) as exc:
        raise JournalIntegrityError("canonical event payload cannot be decoded") from exc
    return event


def _canonical_event(event: object, limits: _Limits) -> tuple[str, bytes]:
    event_id = _event_id(event, maximum_bytes=limits.max_event_id_bytes)
    serializer = getattr(event, "canonical_bytes", None)
    if not callable(serializer):
        raise JournalIntegrityError("event must expose canonical_bytes()")
    try:
        payload = serializer()
    except (TypeError, ValueError, UnicodeError) as exc:
        raise JournalIntegrityError("event canonical serialization failed") from exc
    if not isinstance(payload, bytes):
        raise JournalIntegrityError("canonical_bytes() must return immutable bytes")
    if len(payload) > limits.max_event_bytes:
        raise JournalResourceLimitError(
            f"canonical event exceeds the {limits.max_event_bytes}-byte event limit"
        )
    decoded = _decode_event(payload)
    decoded_id = _event_id(decoded, maximum_bytes=limits.max_event_id_bytes)
    if decoded_id != event_id:
        raise JournalIntegrityError("canonical payload event_id does not match the event")
    try:
        round_trip = decoded.canonical_bytes()
    except (TypeError, ValueError, UnicodeError) as exc:
        raise JournalIntegrityError("decoded event cannot be serialized canonically") from exc
    if not isinstance(round_trip, bytes) or round_trip != payload:
        raise JournalIntegrityError("event payload is not in canonical form")
    return event_id, payload


def _chain_hash(previous_hash: bytes, event_id: str, payload: bytes) -> bytes:
    identifier = event_id.encode("utf-8", errors="strict")
    digest = hashlib.sha256()
    digest.update(_HASH_DOMAIN)
    digest.update(previous_hash)
    digest.update(len(identifier).to_bytes(8, byteorder="big", signed=False))
    digest.update(identifier)
    digest.update(len(payload).to_bytes(8, byteorder="big", signed=False))
    digest.update(payload)
    return digest.digest()


def _event_order_key(event: object) -> tuple[object, str]:
    coordinate = getattr(event, "coordinate", None)
    if coordinate is None:
        raise JournalIntegrityError("event must expose a logical coordinate")
    event_id = getattr(event, "event_id", None)
    if not isinstance(event_id, str):
        raise JournalIntegrityError("event must expose a string event_id")
    return coordinate, event_id


def _verified_records(
    records: tuple[_Record, ...],
    *,
    expected_head: JournalHead,
    expected_total_payload_bytes: int,
    limits: _Limits,
) -> tuple[bytes, ...]:
    if len(records) != expected_head.count:
        raise JournalIntegrityError("journal metadata count does not match stored events")
    if len(records) > limits.max_events:
        raise JournalResourceLimitError("journal exceeds the configured event-count limit")

    previous_hash = _GENESIS_HASH
    total_payload_bytes = 0
    seen_ids: set[str] = set()
    payloads: list[bytes] = []
    previous_order_key: tuple[object, str] | None = None
    for expected_ordinal, record in enumerate(records, start=1):
        if record.ordinal != expected_ordinal:
            raise JournalIntegrityError("journal ordinals are not contiguous")
        if record.event_id in seen_ids:
            raise JournalIntegrityError("journal contains duplicate event identifiers")
        seen_ids.add(record.event_id)
        if record.previous_hash != previous_hash:
            raise JournalIntegrityError("journal previous-hash linkage is invalid")
        if len(record.payload) > limits.max_event_bytes:
            raise JournalResourceLimitError("stored event exceeds the configured event-size limit")
        if len(record.event_id.encode("utf-8", errors="strict")) > limits.max_event_id_bytes:
            raise JournalResourceLimitError("stored event_id exceeds the configured size limit")
        calculated = _chain_hash(previous_hash, record.event_id, record.payload)
        if record.event_hash != calculated:
            raise JournalIntegrityError("journal event hash does not match canonical content")
        decoded = _decode_event(record.payload)
        if _event_id(decoded, maximum_bytes=limits.max_event_id_bytes) != record.event_id:
            raise JournalIntegrityError("stored event_id does not match canonical content")
        if decoded.canonical_bytes() != record.payload:
            raise JournalIntegrityError("stored event is not canonically encoded")
        order_key = _event_order_key(decoded)
        try:
            if previous_order_key is not None and order_key <= previous_order_key:
                raise JournalIntegrityError("journal events are not in strict logical order")
        except TypeError as exc:
            raise JournalIntegrityError("journal event coordinates are not comparable") from exc
        previous_order_key = order_key
        total_payload_bytes += len(record.payload)
        if total_payload_bytes > limits.max_total_payload_bytes:
            raise JournalResourceLimitError("journal exceeds the aggregate payload-size limit")
        payloads.append(record.payload)
        previous_hash = calculated

    if total_payload_bytes != expected_total_payload_bytes:
        raise JournalIntegrityError("journal payload-byte metadata does not match stored events")
    if previous_hash != expected_head.event_hash:
        raise JournalIntegrityError("journal metadata head does not match the verified chain")
    if expected_head.count == 0 and expected_head.event_hash != _GENESIS_HASH:
        raise JournalIntegrityError("empty journal does not use the required genesis hash")
    return tuple(payloads)


class InMemoryJournal:
    """Bounded process-local journal with the same replay semantics as SQLite."""

    def __init__(
        self,
        *,
        max_events: int = _DEFAULT_MAX_EVENTS,
        max_event_bytes: int = _DEFAULT_MAX_EVENT_BYTES,
        max_total_payload_bytes: int = _DEFAULT_MAX_TOTAL_PAYLOAD_BYTES,
        max_event_id_bytes: int = _DEFAULT_MAX_EVENT_ID_BYTES,
    ) -> None:
        self._limits = _limits(
            max_events=max_events,
            max_event_bytes=max_event_bytes,
            max_total_payload_bytes=max_total_payload_bytes,
            max_event_id_bytes=max_event_id_bytes,
        )
        self._records: list[_Record] = []
        self._indices: dict[str, int] = {}
        self._total_payload_bytes = 0
        self._closed = False
        self._lock = threading.RLock()

    def _check_open(self) -> None:
        if self._closed:
            raise JournalClosedError("journal is closed")

    @property
    def count(self) -> int:
        with self._lock:
            self._check_open()
            return len(self._records)

    @property
    def head_hash(self) -> bytes:
        with self._lock:
            self._check_open()
            return self._records[-1].event_hash if self._records else _GENESIS_HASH

    def append(self, event: ExecutionEvent) -> bool:
        event_id, payload = _canonical_event(event, self._limits)
        canonical_event = _decode_event(payload)
        with self._lock:
            self._check_open()
            existing_index = self._indices.get(event_id)
            if existing_index is not None:
                existing = self._records[existing_index]
                if existing.payload == payload:
                    return False
                raise JournalCollisionError(
                    f"event_id {event_id!r} already identifies different canonical bytes"
                )
            if len(self._records) >= self._limits.max_events:
                raise JournalResourceLimitError("journal event-count limit would be exceeded")
            next_total = self._total_payload_bytes + len(payload)
            if next_total > self._limits.max_total_payload_bytes:
                raise JournalResourceLimitError("journal payload-size limit would be exceeded")
            if self._records:
                previous_event = _decode_event(self._records[-1].payload)
                try:
                    if _event_order_key(canonical_event) <= _event_order_key(previous_event):
                        raise JournalOrderingError(
                            "event does not follow the committed logical coordinate"
                        )
                except TypeError as exc:
                    raise JournalIntegrityError("event coordinates are not comparable") from exc
            previous_hash = self._records[-1].event_hash if self._records else _GENESIS_HASH
            record = _Record(
                ordinal=len(self._records) + 1,
                event_id=event_id,
                payload=payload,
                previous_hash=previous_hash,
                event_hash=_chain_hash(previous_hash, event_id, payload),
            )
            self._indices[event_id] = len(self._records)
            self._records.append(record)
            self._total_payload_bytes = next_total
            return True

    def _verified_payloads(self) -> tuple[bytes, ...]:
        records = tuple(self._records)
        head = JournalHead(
            count=len(records),
            event_hash=records[-1].event_hash if records else _GENESIS_HASH,
        )
        return _verified_records(
            records,
            expected_head=head,
            expected_total_payload_bytes=self._total_payload_bytes,
            limits=self._limits,
        )

    def events(self) -> tuple[ExecutionEvent, ...]:
        return tuple(_decode_event(payload) for payload in self.export_canonical())

    def export_canonical(self) -> tuple[bytes, ...]:
        with self._lock:
            self._check_open()
            return self._verified_payloads()

    def verify(self) -> JournalHead:
        with self._lock:
            self._check_open()
            self._verified_payloads()
            return JournalHead(count=len(self._records), event_hash=self.head_hash)

    def close(self) -> None:
        with self._lock:
            self._closed = True

    def __enter__(self) -> Self:
        with self._lock:
            self._check_open()
            return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.close()


_CREATE_METADATA_SQL = f"""
CREATE TABLE journal_metadata (
    singleton INTEGER PRIMARY KEY CHECK (singleton = 1),
    schema_version INTEGER NOT NULL CHECK (schema_version = {_SCHEMA_VERSION}),
    hash_algorithm TEXT NOT NULL CHECK (hash_algorithm = '{_HASH_ALGORITHM}'),
    event_count INTEGER NOT NULL CHECK (event_count >= 0),
    total_payload_bytes INTEGER NOT NULL CHECK (total_payload_bytes >= 0),
    head_hash BLOB NOT NULL CHECK (length(head_hash) = {_HASH_SIZE})
)
""".strip()

_CREATE_EVENTS_SQL = f"""
CREATE TABLE journal_events (
    ordinal INTEGER PRIMARY KEY CHECK (ordinal >= 1),
    event_id TEXT NOT NULL UNIQUE
        CHECK (length(event_id) >= 1 AND length(CAST(event_id AS BLOB)) <= {_HARD_MAX_EVENT_ID_BYTES}),
    payload BLOB NOT NULL CHECK (length(payload) <= {_HARD_MAX_EVENT_BYTES}),
    payload_size INTEGER NOT NULL CHECK (payload_size = length(payload)),
    previous_hash BLOB NOT NULL CHECK (length(previous_hash) = {_HASH_SIZE}),
    event_hash BLOB NOT NULL UNIQUE CHECK (length(event_hash) = {_HASH_SIZE})
)
""".strip()

_TRIGGER_SQL = {
    "journal_events_no_update": """
        CREATE TRIGGER journal_events_no_update
        BEFORE UPDATE ON journal_events
        BEGIN
            SELECT RAISE(ABORT, 'journal events are append-only');
        END
    """.strip(),
    "journal_events_no_delete": """
        CREATE TRIGGER journal_events_no_delete
        BEFORE DELETE ON journal_events
        BEGIN
            SELECT RAISE(ABORT, 'journal events are append-only');
        END
    """.strip(),
    "journal_events_ordered_insert": """
        CREATE TRIGGER journal_events_ordered_insert
        BEFORE INSERT ON journal_events
        WHEN NEW.ordinal != (SELECT event_count + 1 FROM journal_metadata WHERE singleton = 1)
          OR NEW.previous_hash != (SELECT head_hash FROM journal_metadata WHERE singleton = 1)
        BEGIN
            SELECT RAISE(ABORT, 'journal event order does not extend the committed head');
        END
    """.strip(),
    "journal_metadata_no_delete": """
        CREATE TRIGGER journal_metadata_no_delete
        BEFORE DELETE ON journal_metadata
        BEGIN
            SELECT RAISE(ABORT, 'journal metadata is append-only');
        END
    """.strip(),
    "journal_metadata_identity_immutable": """
        CREATE TRIGGER journal_metadata_identity_immutable
        BEFORE UPDATE OF singleton, schema_version, hash_algorithm ON journal_metadata
        WHEN NEW.singleton != OLD.singleton
          OR NEW.schema_version != OLD.schema_version
          OR NEW.hash_algorithm != OLD.hash_algorithm
        BEGIN
            SELECT RAISE(ABORT, 'journal metadata identity is immutable');
        END
    """.strip(),
}

_EXPECTED_TABLE_COLUMNS = {
    "journal_metadata": (
        ("singleton", "INTEGER", 0, 1),
        ("schema_version", "INTEGER", 1, 0),
        ("hash_algorithm", "TEXT", 1, 0),
        ("event_count", "INTEGER", 1, 0),
        ("total_payload_bytes", "INTEGER", 1, 0),
        ("head_hash", "BLOB", 1, 0),
    ),
    "journal_events": (
        ("ordinal", "INTEGER", 0, 1),
        ("event_id", "TEXT", 1, 0),
        ("payload", "BLOB", 1, 0),
        ("payload_size", "INTEGER", 1, 0),
        ("previous_hash", "BLOB", 1, 0),
        ("event_hash", "BLOB", 1, 0),
    ),
}


def _normalized_sql(value: str | None) -> str:
    return " ".join((value or "").split()).casefold()


def _validate_path(path: str | os.PathLike[str]) -> tuple[Path, bool]:
    try:
        raw = os.fspath(path)
    except TypeError as exc:
        raise JournalPathError("journal path must be path-like") from exc
    if not isinstance(raw, str) or not raw or "\x00" in raw:
        raise JournalPathError("journal path must be a non-empty string without NUL bytes")
    if len(os.fsencode(raw)) > _MAX_PATH_BYTES:
        raise JournalPathError(f"journal path exceeds {_MAX_PATH_BYTES} filesystem bytes")

    unresolved = Path(os.path.abspath(raw))
    candidate = unresolved
    parent = candidate.parent
    try:
        parent_stat = parent.stat(follow_symlinks=False)
    except FileNotFoundError as exc:
        raise JournalPathError("journal parent directory does not exist") from exc
    if stat.S_ISLNK(parent_stat.st_mode) or not stat.S_ISDIR(parent_stat.st_mode):
        raise JournalPathError("journal parent must be a real directory, not a symlink")
    if hasattr(os, "getuid") and parent_stat.st_uid != os.getuid():
        raise JournalPathError("journal parent directory must be owned by the current user")
    if stat.S_IMODE(parent_stat.st_mode) & 0o022:
        raise JournalPathError("journal parent directory must not be group/world writable")

    try:
        unresolved_target_stat = unresolved.stat(follow_symlinks=False)
    except FileNotFoundError:
        unresolved_target_stat = None
    if unresolved_target_stat is not None and stat.S_ISLNK(unresolved_target_stat.st_mode):
        raise JournalPathError("journal database path must not be a symlink")

    current = parent
    while current != current.parent:
        try:
            current_stat = current.stat(follow_symlinks=False)
        except FileNotFoundError as exc:
            raise JournalPathError("journal path contains a missing ancestor") from exc
        if stat.S_ISLNK(current_stat.st_mode):
            link_parent_stat = current.parent.stat(follow_symlinks=False)
            trusted_system_link = current_stat.st_uid == 0 and not (
                stat.S_IMODE(link_parent_stat.st_mode) & 0o022
            )
            if not trusted_system_link:
                raise JournalPathError("journal path must not traverse untrusted symlinks")
        current = current.parent

    # macOS exposes trusted system paths such as /var through root-owned links.
    # Resolve those only after the ownership and mutability checks above; user-
    # controlled parent or target links remain rejected.
    candidate = Path(os.path.realpath(unresolved))

    flags = os.O_RDWR | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    created = False
    try:
        descriptor = os.open(candidate, flags)
    except FileNotFoundError:
        try:
            descriptor = os.open(candidate, flags | os.O_CREAT | os.O_EXCL, 0o600)
            created = True
        except OSError as exc:
            raise JournalPathError("cannot securely create the journal database") from exc
    except OSError as exc:
        raise JournalPathError("cannot securely open the journal database") from exc
    try:
        file_stat = os.fstat(descriptor)
        if not stat.S_ISREG(file_stat.st_mode):
            raise JournalPathError("journal path must identify a regular file")
        if file_stat.st_nlink != 1:
            raise JournalPathError("journal database must not have hard-link aliases")
        if hasattr(os, "getuid") and file_stat.st_uid != os.getuid():
            raise JournalPathError("journal database must be owned by the current user")
        if stat.S_IMODE(file_stat.st_mode) & 0o077:
            raise JournalPathError("journal database permissions must be owner-only")
    finally:
        os.close(descriptor)
    return candidate, created


def _validate_sidecar(path: Path, *, permit_hardening: bool) -> None:
    try:
        path_stat = path.stat(follow_symlinks=False)
    except FileNotFoundError:
        return
    if stat.S_ISLNK(path_stat.st_mode) or not stat.S_ISREG(path_stat.st_mode):
        raise JournalPathError("SQLite journal sidecars must be regular files, not symlinks")
    if path_stat.st_nlink != 1:
        raise JournalPathError("SQLite journal sidecars must not have hard-link aliases")
    if hasattr(os, "getuid") and path_stat.st_uid != os.getuid():
        raise JournalPathError("SQLite journal sidecars must be owned by the current user")
    if stat.S_IMODE(path_stat.st_mode) & 0o077:
        if not permit_hardening:
            raise JournalPathError("SQLite journal sidecar permissions must be owner-only")
        try:
            path.chmod(0o600, follow_symlinks=False)
        except OSError as exc:
            raise JournalPathError("cannot harden SQLite journal sidecar permissions") from exc


class SQLiteJournal:
    """Durable append-only event journal backed by a hardened SQLite database.

    Writes use ``BEGIN IMMEDIATE`` under WAL mode with ``synchronous=FULL``.
    SQLite therefore serializes writers, while ``busy_timeout_ms`` bounds lock
    waiting.  One instance and its connection must remain on the creating
    thread; callers needing cross-thread ingestion should serialize requests at
    the event-engine boundary.
    """

    def __init__(
        self,
        path: str | os.PathLike[str],
        *,
        max_events: int = _DEFAULT_MAX_EVENTS,
        max_event_bytes: int = _DEFAULT_MAX_EVENT_BYTES,
        max_total_payload_bytes: int = _DEFAULT_MAX_TOTAL_PAYLOAD_BYTES,
        max_database_bytes: int = _DEFAULT_MAX_DATABASE_BYTES,
        max_event_id_bytes: int = _DEFAULT_MAX_EVENT_ID_BYTES,
        busy_timeout_ms: int = _DEFAULT_BUSY_TIMEOUT_MS,
    ) -> None:
        self._limits = _limits(
            max_events=max_events,
            max_event_bytes=max_event_bytes,
            max_total_payload_bytes=max_total_payload_bytes,
            max_event_id_bytes=max_event_id_bytes,
            max_database_bytes=max_database_bytes,
        )
        self._busy_timeout_ms = _bounded_integer(
            "busy_timeout_ms", busy_timeout_ms, minimum=0, maximum=_MAX_BUSY_TIMEOUT_MS
        )
        self._path, created = _validate_path(path)
        self._owner_thread = threading.get_ident()
        self._closed = False
        for suffix in ("-wal", "-shm"):
            _validate_sidecar(Path(f"{self._path}{suffix}"), permit_hardening=False)

        try:
            self._connection = sqlite3.connect(
                self._path,
                timeout=self._busy_timeout_ms / 1000.0,
                isolation_level=None,
                check_same_thread=True,
            )
            self._connection.execute(f"PRAGMA busy_timeout = {self._busy_timeout_ms}")
            self._connection.execute("PRAGMA foreign_keys = ON")
            self._connection.execute("PRAGMA trusted_schema = OFF")
            self._connection.execute("PRAGMA temp_store = MEMORY")
            mode = self._connection.execute("PRAGMA journal_mode = WAL").fetchone()
            if mode is None or str(mode[0]).casefold() != "wal":
                raise JournalSchemaError("SQLite journal could not enable WAL mode")
            self._connection.execute("PRAGMA synchronous = FULL")
            synchronous = self._connection.execute("PRAGMA synchronous").fetchone()
            if synchronous is None or int(synchronous[0]) != 2:
                raise JournalSchemaError("SQLite journal could not enable FULL synchronization")
            self._connection.execute("PRAGMA wal_autocheckpoint = 1")
            self._connection.execute("PRAGMA journal_size_limit = 0")
            self._configure_page_limit()
            if created:
                self._initialize_schema()
            else:
                self._validate_schema()
            self._harden_sidecars()
            self.verify()
            self._check_database_size()
        except JournalError:
            connection = getattr(self, "_connection", None)
            if connection is not None:
                connection.close()
            self._closed = True
            raise
        except sqlite3.Error as exc:
            connection = getattr(self, "_connection", None)
            if connection is not None:
                connection.close()
            self._closed = True
            raise self._mapped_sqlite_error(exc, operation="open") from exc
        except Exception:
            connection = getattr(self, "_connection", None)
            if connection is not None:
                connection.close()
            self._closed = True
            raise

    def _check_open(self) -> None:
        if self._closed:
            raise JournalClosedError("journal is closed")
        if threading.get_ident() != self._owner_thread:
            raise JournalBusyError("SQLiteJournal is restricted to its creating thread")

    def _configure_page_limit(self) -> None:
        database_limit = self._limits.max_database_bytes
        if database_limit is None:  # pragma: no cover - constructor invariant
            raise JournalResourceLimitError("SQLite journal lacks a database-size limit")
        page_row = self._connection.execute("PRAGMA page_size").fetchone()
        if page_row is None or int(page_row[0]) <= 0:
            raise JournalSchemaError("SQLite database reports an invalid page size")
        self._page_size = int(page_row[0])
        maximum_pages = max(1, database_limit // self._page_size)
        result = self._connection.execute(f"PRAGMA max_page_count = {maximum_pages}").fetchone()
        if result is None or int(result[0]) > maximum_pages:
            raise JournalResourceLimitError("existing SQLite database exceeds its page limit")

    def _initialize_schema(self) -> None:
        try:
            self._connection.execute("BEGIN EXCLUSIVE")
            self._connection.execute(_CREATE_METADATA_SQL)
            self._connection.execute(_CREATE_EVENTS_SQL)
            for statement in _TRIGGER_SQL.values():
                self._connection.execute(statement)
            self._connection.execute(
                """
                INSERT INTO journal_metadata (
                    singleton, schema_version, hash_algorithm, event_count,
                    total_payload_bytes, head_hash
                ) VALUES (?, ?, ?, ?, ?, ?)
                """,
                (1, _SCHEMA_VERSION, _HASH_ALGORITHM, 0, 0, _GENESIS_HASH),
            )
            self._connection.execute(f"PRAGMA application_id = {_APPLICATION_ID}")
            self._connection.execute(f"PRAGMA user_version = {_SCHEMA_VERSION}")
            self._connection.execute("COMMIT")
        except sqlite3.Error as exc:
            self._rollback_quietly()
            raise self._mapped_sqlite_error(exc, operation="initialize") from exc
        self._validate_schema()

    def _validate_schema(self) -> None:
        try:
            application_id = self._connection.execute("PRAGMA application_id").fetchone()
            user_version = self._connection.execute("PRAGMA user_version").fetchone()
            if application_id is None or int(application_id[0]) != _APPLICATION_ID:
                raise JournalSchemaError("database is not an AlphaForge event journal")
            if user_version is None or int(user_version[0]) != _SCHEMA_VERSION:
                raise JournalSchemaError(f"journal schema version must be {_SCHEMA_VERSION}")

            objects = self._connection.execute("""
                SELECT type, name, sql
                FROM sqlite_master
                WHERE name NOT LIKE 'sqlite_%'
                ORDER BY type, name
                """).fetchall()
            tables = {str(name): sql for kind, name, sql in objects if kind == "table"}
            triggers = {str(name): sql for kind, name, sql in objects if kind == "trigger"}
            unexpected = {
                str(name) for kind, name, _ in objects if kind not in {"table", "trigger"}
            }
            if set(tables) != set(_EXPECTED_TABLE_COLUMNS) or unexpected:
                raise JournalSchemaError("journal tables do not match the supported schema")
            if set(triggers) != set(_TRIGGER_SQL):
                raise JournalSchemaError("journal append-only triggers are missing or unexpected")
            expected_table_sql = {
                "journal_metadata": _CREATE_METADATA_SQL,
                "journal_events": _CREATE_EVENTS_SQL,
            }
            for name, expected_sql in expected_table_sql.items():
                if _normalized_sql(tables[name]) != _normalized_sql(expected_sql):
                    raise JournalSchemaError(f"journal table {name!r} definition is invalid")
            for name, expected_sql in _TRIGGER_SQL.items():
                if _normalized_sql(triggers[name]) != _normalized_sql(expected_sql):
                    raise JournalSchemaError(f"journal trigger {name!r} definition is invalid")

            for table, expected in _EXPECTED_TABLE_COLUMNS.items():
                columns = self._connection.execute(f"PRAGMA table_info({table})").fetchall()
                actual = tuple(
                    (str(row[1]), str(row[2]).upper(), int(row[3]), int(row[5])) for row in columns
                )
                if actual != expected:
                    raise JournalSchemaError(f"journal table {table!r} columns are invalid")

            metadata = self._connection.execute(
                """
                SELECT schema_version, hash_algorithm
                FROM journal_metadata
                WHERE singleton = ?
                """,
                (1,),
            ).fetchall()
            if metadata != [(_SCHEMA_VERSION, _HASH_ALGORITHM)]:
                raise JournalSchemaError("journal metadata identity is invalid")
        except JournalError:
            raise
        except sqlite3.Error as exc:
            raise self._mapped_sqlite_error(exc, operation="validate schema") from exc

    def _metadata(self) -> tuple[int, int, bytes]:
        row = self._connection.execute(
            """
            SELECT event_count, total_payload_bytes, head_hash
            FROM journal_metadata
            WHERE singleton = ?
            """,
            (1,),
        ).fetchone()
        if row is None:
            raise JournalIntegrityError("journal metadata row is missing")
        count, total_payload_bytes, head_hash = row
        if isinstance(count, bool) or not isinstance(count, int) or count < 0:
            raise JournalIntegrityError("journal metadata count is invalid")
        if (
            isinstance(total_payload_bytes, bool)
            or not isinstance(total_payload_bytes, int)
            or total_payload_bytes < 0
        ):
            raise JournalIntegrityError("journal payload-byte metadata is invalid")
        if not isinstance(head_hash, bytes) or len(head_hash) != _HASH_SIZE:
            raise JournalIntegrityError("journal metadata head hash is invalid")
        return count, total_payload_bytes, head_hash

    def _records_from_database(self) -> tuple[_Record, ...]:
        rows = self._connection.execute("""
            SELECT ordinal, event_id, payload, payload_size, previous_hash, event_hash
            FROM journal_events
            ORDER BY ordinal ASC
            """).fetchall()
        records: list[_Record] = []
        for ordinal, event_id, payload, payload_size, previous_hash, event_hash in rows:
            if not isinstance(ordinal, int) or not isinstance(payload_size, int):
                raise JournalIntegrityError("journal row contains invalid integer metadata")
            if not isinstance(event_id, str):
                raise JournalIntegrityError("journal row event_id is not text")
            if not isinstance(payload, bytes) or payload_size != len(payload):
                raise JournalIntegrityError("journal row payload size is invalid")
            if not isinstance(previous_hash, bytes) or not isinstance(event_hash, bytes):
                raise JournalIntegrityError("journal row hash is not binary")
            if len(previous_hash) != _HASH_SIZE or len(event_hash) != _HASH_SIZE:
                raise JournalIntegrityError("journal row hash has an invalid length")
            records.append(
                _Record(
                    ordinal=ordinal,
                    event_id=event_id,
                    payload=payload,
                    previous_hash=previous_hash,
                    event_hash=event_hash,
                )
            )
        return tuple(records)

    def _verified_snapshot(self) -> tuple[tuple[bytes, ...], JournalHead]:
        self._validate_schema()
        try:
            integrity = self._connection.execute("PRAGMA integrity_check(1)").fetchall()
            if integrity != [("ok",)]:
                raise JournalIntegrityError("SQLite database integrity check failed")
            self._connection.execute("BEGIN")
            count, total_payload_bytes, head_hash = self._metadata()
            records = self._records_from_database()
            head = JournalHead(count=count, event_hash=head_hash)
            payloads = _verified_records(
                records,
                expected_head=head,
                expected_total_payload_bytes=total_payload_bytes,
                limits=self._limits,
            )
            self._connection.execute("COMMIT")
            return payloads, head
        except JournalError:
            self._rollback_quietly()
            raise
        except sqlite3.Error as exc:
            self._rollback_quietly()
            raise self._mapped_sqlite_error(exc, operation="verify") from exc

    def _verified_payloads(self) -> tuple[bytes, ...]:
        return self._verified_snapshot()[0]

    @property
    def count(self) -> int:
        self._check_open()
        try:
            return self._metadata()[0]
        except sqlite3.Error as exc:
            raise self._mapped_sqlite_error(exc, operation="read count") from exc

    @property
    def head_hash(self) -> bytes:
        self._check_open()
        try:
            return self._metadata()[2]
        except sqlite3.Error as exc:
            raise self._mapped_sqlite_error(exc, operation="read head") from exc

    def append(self, event: ExecutionEvent) -> bool:
        self._check_open()
        event_id, payload = _canonical_event(event, self._limits)
        canonical_event = _decode_event(payload)
        try:
            self._connection.execute("BEGIN IMMEDIATE")
            existing = self._connection.execute(
                """
                SELECT payload, previous_hash, event_hash
                FROM journal_events
                WHERE event_id = ?
                """,
                (event_id,),
            ).fetchone()
            if existing is not None:
                existing_payload, previous_hash, event_hash = existing
                if not all(isinstance(value, bytes) for value in existing):
                    raise JournalIntegrityError("existing journal event is malformed")
                if event_hash != _chain_hash(previous_hash, event_id, existing_payload):
                    raise JournalIntegrityError("existing journal event hash is invalid")
                self._connection.execute("ROLLBACK")
                if existing_payload == payload:
                    return False
                raise JournalCollisionError(
                    f"event_id {event_id!r} already identifies different canonical bytes"
                )

            count, total_payload_bytes, previous_hash = self._metadata()
            if count >= self._limits.max_events:
                raise JournalResourceLimitError("journal event-count limit would be exceeded")
            next_total = total_payload_bytes + len(payload)
            if next_total > self._limits.max_total_payload_bytes:
                raise JournalResourceLimitError("journal payload-size limit would be exceeded")
            if count:
                last_payload_row = self._connection.execute(
                    "SELECT payload FROM journal_events WHERE ordinal = ?",
                    (count,),
                ).fetchone()
                if last_payload_row is None or not isinstance(last_payload_row[0], bytes):
                    raise JournalIntegrityError("committed journal tail is missing or malformed")
                previous_event = _decode_event(last_payload_row[0])
                try:
                    if _event_order_key(canonical_event) <= _event_order_key(previous_event):
                        raise JournalOrderingError(
                            "event does not follow the committed logical coordinate"
                        )
                except TypeError as exc:
                    raise JournalIntegrityError("event coordinates are not comparable") from exc
            self._check_database_growth(len(payload))
            ordinal = count + 1
            event_hash = _chain_hash(previous_hash, event_id, payload)
            self._connection.execute(
                """
                INSERT INTO journal_events (
                    ordinal, event_id, payload, payload_size, previous_hash, event_hash
                ) VALUES (?, ?, ?, ?, ?, ?)
                """,
                (ordinal, event_id, payload, len(payload), previous_hash, event_hash),
            )
            updated = self._connection.execute(
                """
                UPDATE journal_metadata
                SET event_count = ?, total_payload_bytes = ?, head_hash = ?
                WHERE singleton = ? AND event_count = ? AND head_hash = ?
                """,
                (ordinal, next_total, event_hash, 1, count, previous_hash),
            )
            if updated.rowcount != 1:
                raise JournalIntegrityError("journal head changed during append")
            self._connection.execute("COMMIT")
            self._harden_sidecars()
            self._check_database_size()
            return True
        except JournalError:
            self._rollback_quietly()
            raise
        except sqlite3.Error as exc:
            self._rollback_quietly()
            raise self._mapped_sqlite_error(exc, operation="append") from exc

    def events(self) -> tuple[ExecutionEvent, ...]:
        return tuple(_decode_event(payload) for payload in self.export_canonical())

    def export_canonical(self) -> tuple[bytes, ...]:
        self._check_open()
        return self._verified_payloads()

    def verify(self) -> JournalHead:
        self._check_open()
        return self._verified_snapshot()[1]

    def _rollback_quietly(self) -> None:
        connection = getattr(self, "_connection", None)
        if connection is None or not connection.in_transaction:
            return
        with suppress(sqlite3.Error):
            connection.execute("ROLLBACK")

    def _mapped_sqlite_error(self, error: sqlite3.Error, *, operation: str) -> JournalError:
        message = str(error).casefold()
        if "locked" in message or "busy" in message:
            return JournalBusyError(f"SQLite journal is busy during {operation}")
        if isinstance(error, (sqlite3.IntegrityError, sqlite3.DatabaseError)):
            return JournalIntegrityError(f"SQLite journal failed integrity during {operation}")
        return JournalError(f"SQLite journal failed during {operation}")

    def _database_size(self) -> int:
        total = 0
        for candidate in (self._path, Path(f"{self._path}-wal"), Path(f"{self._path}-shm")):
            try:
                total += candidate.stat(follow_symlinks=False).st_size
            except FileNotFoundError:
                continue
        return total

    def _check_database_size(self) -> None:
        database_limit = self._limits.max_database_bytes
        if database_limit is None:  # pragma: no cover - constructor invariant
            raise JournalResourceLimitError("SQLite journal lacks a database-size limit")
        if self._database_size() > database_limit:
            raise JournalResourceLimitError("SQLite journal exceeds the database-size limit")

    def _check_database_growth(self, payload_bytes: int) -> None:
        database_limit = self._limits.max_database_bytes
        if database_limit is None:  # pragma: no cover - constructor invariant
            raise JournalResourceLimitError("SQLite journal lacks a database-size limit")
        conservative_growth = payload_bytes + 16 * self._page_size
        if self._database_size() + conservative_growth > database_limit:
            raise JournalResourceLimitError("append could exceed the SQLite database-size limit")

    def _harden_sidecars(self) -> None:
        for suffix in ("-wal", "-shm"):
            _validate_sidecar(Path(f"{self._path}{suffix}"), permit_hardening=True)

    def close(self) -> None:
        if self._closed:
            return
        self._check_open()
        failure: JournalError | None = None
        try:
            if self._connection.in_transaction:
                self._connection.execute("ROLLBACK")
            self._connection.execute("PRAGMA wal_checkpoint(TRUNCATE)")
        except sqlite3.Error as exc:
            failure = self._mapped_sqlite_error(exc, operation="close")
        finally:
            try:
                self._connection.close()
                self._harden_sidecars()
            except (sqlite3.Error, JournalError) as exc:
                if failure is None:
                    failure = (
                        exc
                        if isinstance(exc, JournalError)
                        else self._mapped_sqlite_error(exc, operation="close")
                    )
            self._closed = True
        if failure is not None:
            raise failure

    def __enter__(self) -> Self:
        self._check_open()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.close()
