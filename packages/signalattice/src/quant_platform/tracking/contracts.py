"""Strict immutable contracts for the local experiment control plane.

The registry treats every value crossing its boundary as untrusted.  Contracts therefore
validate lengths, identifiers, timestamps, and recursive JSON before SQLite sees them.  Nested
JSON is frozen as tuples and read-only mappings so a request cannot change after its digest is
computed.
"""

from __future__ import annotations

import json
import math
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from enum import StrEnum
from types import MappingProxyType
from typing import Protocol, cast

type JsonScalar = None | bool | int | float | str
type JsonValue = JsonScalar | tuple[JsonValue, ...] | Mapping[str, JsonValue]

_IDENTIFIER = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:/-]{0,127}$")
_DIGEST = re.compile(r"^[0-9a-f]{64}$")
_MEDIA_TYPE = re.compile(r"^[a-z0-9][a-z0-9!#$&^_.+-]*/[a-z0-9][a-z0-9!#$&^_.+-]*$")
_MAX_MEDIA_TYPE_BYTES = 128
_MAX_JSON_DEPTH = 16
_MAX_JSON_NODES = 16_384
_MAX_JSON_STRING_BYTES = 65_536
_MAX_CANONICAL_JSON_BYTES = 16_777_216
_MAX_JSON_STRUCTURAL_TOKENS = 50_001
_SAFE_FAILURE_SUMMARY = re.compile(r"^[A-Za-z0-9][A-Za-z0-9 .,;:()_'-]{0,511}$")
_SECRETISH_TOKEN = re.compile(r"[A-Za-z0-9_-]{32,}")
_SECRETISH_WORD = re.compile(
    r"(?i)\b(api[ _-]?key|authorization|bearer|password|private[ _-]?key|secret|token)\b"
)


class RegistryError(RuntimeError):
    """Base error carrying a stable machine-readable code and retry guidance."""

    code = "registry_error"
    retryable = False

    def __init__(self, message: str) -> None:
        super().__init__(message)


class ValidationError(RegistryError):
    """A boundary value violated a documented contract."""

    code = "validation_error"


class NotFoundError(RegistryError):
    """A requested durable record does not exist."""

    code = "not_found"


class ConflictError(RegistryError):
    """A stable key was reused with different semantics."""

    code = "conflict"


class CapacityError(RegistryError):
    """The configured durable queue bound has been reached."""

    code = "capacity_exhausted"
    retryable = True


class BusyError(RegistryError):
    """SQLite did not grant the bounded write lock in time."""

    code = "registry_busy"
    retryable = True


class VerificationTimeoutError(RegistryError):
    """Startup or readiness integrity verification exceeded its query budget."""

    code = "registry_verification_timeout"
    retryable = True


class RetentionPendingError(RegistryError):
    """New execution is paused while durable deletion intent is unresolved."""

    code = "retention_pending"
    retryable = True


class IntegrityError(RegistryError):
    """The database is corrupt, partial, or violates a durable invariant."""

    code = "integrity_error"


class RegistryAuthorityMismatchError(IntegrityError):
    """The supplied process secret does not authenticate this registry."""

    code = "registry_authority_mismatch"


class RetentionPlanAuthenticationError(IntegrityError):
    """A retention plan HMAC is invalid for the bound registry/CAS authority."""

    code = "retention_plan_authentication_failed"


class MigrationError(RegistryError):
    """A migration could not be verified or applied atomically."""

    code = "migration_error"


class MigrationDriftError(MigrationError):
    """An applied migration differs from the compiled migration."""

    code = "migration_drift"


class UnsupportedSchemaError(MigrationError):
    """The database was written by an unknown newer registry."""

    code = "unsupported_schema"


class InvalidTransitionError(RegistryError):
    """A requested job state transition is not permitted."""

    code = "invalid_transition"


class LeaseLostError(RegistryError):
    """A lease is absent, expired, or owned by another claimant."""

    code = "lease_lost"


class InvalidCursorError(ValidationError):
    """A pagination cursor is malformed, stale-versioned, or unauthenticated."""

    code = "invalid_cursor"


class JobState(StrEnum):
    """Closed job lifecycle; cancellation intent is represented separately."""

    QUEUED = "queued"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    CANCELLED = "cancelled"

    @property
    def terminal(self) -> bool:
        """Return whether no further lifecycle transition is legal."""

        return self in {self.SUCCEEDED, self.FAILED, self.CANCELLED}


class EventKind(StrEnum):
    """Append-only lifecycle event kinds."""

    SUBMITTED = "submitted"
    CLAIMED = "claimed"
    LEASE_EXPIRED = "lease_expired"
    RETRIED = "retried"
    CANCEL_REQUESTED = "cancel_requested"
    CANCELLED = "cancelled"
    SUCCEEDED = "succeeded"
    FAILED = "failed"


class CancellationReasonCode(StrEnum):
    """Closed, non-sensitive operator reasons for cancellation intent."""

    OPERATOR_REQUEST = "operator_request"
    SUPERSEDED = "superseded"
    DATA_QUALITY = "data_quality"
    RESOURCE_POLICY = "resource_policy"
    RISK_CONTROL = "risk_control"
    SHUTDOWN = "shutdown"


class FailureReasonCode(StrEnum):
    """Closed public failure taxonomy; raw exceptions never cross this boundary."""

    DATA_UNAVAILABLE = "data_unavailable"
    DEPENDENCY_UNAVAILABLE = "dependency_unavailable"
    EXECUTION_ERROR = "execution_error"
    INTERNAL_ERROR = "internal_error"
    INVALID_INPUT = "invalid_input"
    LEASE_ATTEMPTS_EXHAUSTED = "lease_attempts_exhausted"
    RESOURCE_EXHAUSTED = "resource_exhausted"
    TRANSIENT_IO = "transient_io"


class ArtifactClass(StrEnum):
    """Closed artifact classes accepted by the content-addressed store."""

    INPUT = "input"
    OUTPUT = "output"
    MODEL = "model"
    REPORT = "report"
    PLOT = "plot"
    LOG = "log"
    METADATA = "metadata"


class RunStatus(StrEnum):
    """Closed terminal status for an immutable execution-run record."""

    SUCCEEDED = "succeeded"
    FAILED = "failed"
    CANCELLED = "cancelled"


class EvidenceClass(StrEnum):
    """Honest provenance classification for execution evidence."""

    MEASURED = "measured"
    SIMULATED = "simulated"
    BACKTESTED = "backtested"
    PAPER_TRADED = "paper_traded"
    LIVE = "live"
    NOT_APPLICABLE = "not_applicable"


class Clock(Protocol):
    """Internal time authority injected once at registry construction.

    Production implementations return trusted wall-clock UTC. Tests may inject a deterministic
    clock; callers cannot provide timestamps to individual authority-bearing operations.
    """

    def now(self) -> datetime:
        """Return one timezone-aware instant."""
        ...


def require_identifier(value: str, field_name: str = "identifier") -> str:
    """Validate and return a bounded control-plane identifier."""

    if type(value) is not str or not _IDENTIFIER.fullmatch(value):
        raise ValidationError(f"{field_name} must match {_IDENTIFIER.pattern}")
    return value


def require_digest(value: str, field_name: str = "digest") -> str:
    """Validate and return a lowercase SHA-256 hex digest."""

    if type(value) is not str or not _DIGEST.fullmatch(value):
        raise ValidationError(f"{field_name} must be a lowercase SHA-256 digest")
    return value


def require_utf8_size(value: object, field_name: str) -> int:
    """Return exact UTF-8 bytes or raise a typed boundary-validation failure.

    Python ``str`` values may contain lone surrogates that cannot be encoded as UTF-8.  Every
    public text boundary uses this helper so those values, and allocation failure while sizing
    them, cannot escape as raw runtime exceptions.
    """

    if type(value) is not str:
        raise ValidationError(f"{field_name} must be exact text")
    try:
        return len(value.encode("utf-8"))
    except (MemoryError, UnicodeEncodeError):
        raise ValidationError(f"{field_name} must contain valid bounded UTF-8 text") from None


def require_failure_summary(value: str) -> str:
    """Accept deliberately public, low-entropy diagnostic prose only.

    Paths, URLs, assignments, multiline tracebacks, and credential-shaped tokens are rejected
    instead of being persisted and later redacted imperfectly.
    """

    if type(value) is not str:
        raise ValidationError("failure_summary must be bounded non-sensitive public prose")
    if (
        not 1 <= require_utf8_size(value, "failure_summary") <= 512
        or not value.isascii()
        or _SAFE_FAILURE_SUMMARY.fullmatch(value) is None
        or _SECRETISH_TOKEN.search(value) is not None
        or _SECRETISH_WORD.search(value) is not None
    ):
        raise ValidationError("failure_summary must be bounded non-sensitive public prose")
    return value


def _validate_cursor_token(value: str) -> None:
    if type(value) is not str or not 16 <= len(value) <= 1_024 or not value.isascii():
        raise InvalidCursorError("cursor token must be bounded ASCII text")


def _require_int(value: int, field_name: str, lower: int, upper: int) -> int:
    if type(value) is not int or not lower <= value <= upper:
        raise ValidationError(f"{field_name} must be an integer in [{lower}, {upper}]")
    return value


def _require_optional_text(value: str | None, field_name: str, maximum: int) -> None:
    if value is None:
        return
    if type(value) is not str or not 1 <= require_utf8_size(value, field_name) <= maximum:
        raise ValidationError(f"{field_name} must be null or contain 1 to {maximum} UTF-8 bytes")


def require_utc(value: datetime, field_name: str) -> datetime:
    """Validate an aware timestamp and normalize it to UTC."""

    if type(value) is not datetime or value.tzinfo is None:
        raise ValidationError(f"{field_name} must be a timezone-aware datetime")
    try:
        offset = value.utcoffset()
    except Exception:
        raise ValidationError(f"{field_name} cannot be normalized safely to UTC") from None
    if offset is None:
        raise ValidationError(f"{field_name} must be a timezone-aware datetime")
    try:
        return value.astimezone(UTC)
    except Exception:
        raise ValidationError(f"{field_name} cannot be normalized safely to UTC") from None


def require_stored_int(value: object, field_name: str, lower: int, upper: int) -> int:
    """Require exact SQLite INTEGER storage and a closed semantic domain."""

    if type(value) is not int or not lower <= value <= upper:
        raise IntegrityError(f"stored {field_name} is not an exact integer in its domain")
    return value


def parse_stored_utc(value: object, field_name: str) -> datetime:
    """Parse only the canonical UTC-microsecond text emitted by registry writes."""

    if type(value) is not str or len(value) != 27 or not value.endswith("Z"):
        raise IntegrityError(f"stored {field_name} is not canonical UTC text")
    try:
        parsed = datetime.fromisoformat(value[:-1] + "+00:00")
    except (MemoryError, OverflowError, ValueError):
        raise IntegrityError(f"stored {field_name} is not canonical UTC text") from None
    if parsed.tzinfo is None or parsed.utcoffset() != timedelta(0):
        raise IntegrityError(f"stored {field_name} is not canonical UTC text")
    canonical = parsed.astimezone(UTC).isoformat(timespec="microseconds").replace("+00:00", "Z")
    if canonical != value:
        raise IntegrityError(f"stored {field_name} is not canonical UTC text")
    return parsed


@dataclass(slots=True)
class _JsonBudget:
    """Bound recursive work and the exact canonical UTF-8 representation before allocation."""

    nodes: int = 0
    encoded_bytes: int = 0

    def add_node(self) -> None:
        self.nodes += 1
        if self.nodes > _MAX_JSON_NODES:
            raise ValidationError(f"JSON exceeds {_MAX_JSON_NODES} nodes")

    def add_encoded_bytes(self, count: int) -> None:
        if count < 0 or self.encoded_bytes > _MAX_CANONICAL_JSON_BYTES - count:
            raise ValidationError(
                "JSON exceeds the cumulative canonical byte budget of "
                f"{_MAX_CANONICAL_JSON_BYTES} bytes"
            )
        self.encoded_bytes += count


def _json_text_size(value: str, *, field_name: str) -> tuple[int, int]:
    """Return raw and JSON-encoded UTF-8 sizes without leaking encoding failures."""

    try:
        raw_size = len(value.encode("utf-8"))
        encoded_size = len(
            json.dumps(value, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
        )
    except (MemoryError, UnicodeEncodeError, ValueError):
        raise ValidationError(f"{field_name} must contain valid UTF-8 text") from None
    return raw_size, encoded_size


def _json_scalar_size(value: JsonScalar) -> int:
    """Return the exact canonical byte length for an already validated scalar."""

    if type(value) is str:
        raw_size, encoded_size = _json_text_size(value, field_name="JSON string")
        if raw_size > _MAX_JSON_STRING_BYTES:
            raise ValidationError(f"JSON strings may not exceed {_MAX_JSON_STRING_BYTES} bytes")
        return encoded_size
    return len(
        json.dumps(
            value,
            ensure_ascii=False,
            allow_nan=False,
            separators=(",", ":"),
        ).encode("utf-8")
    )


def _freeze_json(
    value: object,
    *,
    depth: int = 0,
    budget: _JsonBudget | None = None,
) -> JsonValue:
    if budget is None:
        budget = _JsonBudget()
    budget.add_node()
    if depth > _MAX_JSON_DEPTH:
        raise ValidationError(f"JSON exceeds depth {_MAX_JSON_DEPTH}")
    if value is None:
        budget.add_encoded_bytes(_json_scalar_size(value))
        return None
    if type(value) is bool:
        budget.add_encoded_bytes(_json_scalar_size(value))
        return value
    if type(value) is int:
        if -(2**63) <= value <= 2**63 - 1:
            budget.add_encoded_bytes(_json_scalar_size(value))
            return value
        raise ValidationError("JSON integer is outside signed 64-bit range")
    if type(value) is float:
        if math.isfinite(value):
            budget.add_encoded_bytes(_json_scalar_size(value))
            return value
        raise ValidationError("JSON numbers must be finite")
    if type(value) is str:
        budget.add_encoded_bytes(_json_scalar_size(value))
        return value
    if type(value) in {dict, MappingProxyType}:
        budget.add_encoded_bytes(2)  # Opening and closing braces.
        frozen: dict[str, JsonValue] = {}
        mapping = cast(Mapping[object, object], value)
        for index, (key, item) in enumerate(mapping.items()):
            if type(key) is not str or not key:
                raise ValidationError(
                    "JSON object keys must be non-empty strings at most 256 bytes"
                )
            raw_key_size, encoded_key_size = _json_text_size(key, field_name="JSON object key")
            if raw_key_size > 256:
                raise ValidationError(
                    "JSON object keys must be non-empty strings at most 256 bytes"
                )
            budget.add_encoded_bytes(encoded_key_size + 1 + int(index > 0))
            frozen[key] = _freeze_json(item, depth=depth + 1, budget=budget)
        return MappingProxyType(frozen)
    if type(value) in {list, tuple}:
        budget.add_encoded_bytes(2)  # Opening and closing brackets.
        frozen_items: list[JsonValue] = []
        sequence = cast(Sequence[object], value)
        for index, item in enumerate(sequence):
            budget.add_encoded_bytes(int(index > 0))
            frozen_items.append(_freeze_json(item, depth=depth + 1, budget=budget))
        return tuple(frozen_items)
    # Do not dispatch into arbitrary Mapping/Sequence implementations here.
    # Their iteration hooks are executable code and may raise attacker-chosen
    # exceptions or perform side effects before the typed boundary can respond.
    raise ValidationError("unsupported JSON value type")


def _thaw_json(value: JsonValue) -> object:
    if isinstance(value, Mapping):
        return {key: _thaw_json(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_thaw_json(item) for item in value]
    return value


def canonical_json(value: object) -> str:
    """Return bounded, deterministic JSON suitable for hashing and persistence."""

    try:
        frozen = _freeze_json(value)
        return json.dumps(
            _thaw_json(frozen),
            ensure_ascii=False,
            allow_nan=False,
            separators=(",", ":"),
            sort_keys=True,
        )
    except (MemoryError, RecursionError):
        raise ValidationError("JSON canonicalization exceeded its resource bound") from None


def _strict_json_pairs(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, item in pairs:
        if key in result:
            raise ValueError("duplicate JSON object key")
        result[key] = item
    return result


def _reject_json_constant(_value: str) -> None:
    raise ValueError("non-finite JSON constant")


def _preflight_json_text(text: str, *, maximum_bytes: int) -> None:
    """Bound hostile JSON bytes and nesting before the recursive stdlib decoder runs."""

    if type(maximum_bytes) is not int or not 1 <= maximum_bytes <= _MAX_CANONICAL_JSON_BYTES:
        raise ValidationError("maximum_bytes is outside the supported JSON bound")
    try:
        encoded_size = len(text.encode("utf-8"))
    except (MemoryError, UnicodeEncodeError):
        raise ValidationError("JSON text is not bounded valid UTF-8") from None
    if not 1 <= encoded_size <= maximum_bytes:
        raise ValidationError("JSON text exceeds its byte bound")

    depth = 0
    structural_tokens = 0
    in_string = False
    escaped = False
    for character in text:
        if in_string:
            if escaped:
                escaped = False
            elif character == "\\":
                escaped = True
            elif character == '"':
                in_string = False
            continue
        if character == '"':
            in_string = True
            structural_tokens += 1
        elif character in "[{":
            depth += 1
            structural_tokens += 1
            if depth > _MAX_JSON_DEPTH:
                raise ValidationError(f"JSON exceeds depth {_MAX_JSON_DEPTH}")
        elif character in "]}":
            depth -= 1
            if depth < 0:
                raise ValidationError("JSON structure is malformed")
        elif character in ",:":
            structural_tokens += 1
        if structural_tokens > _MAX_JSON_STRUCTURAL_TOKENS:
            raise ValidationError("JSON exceeds its structural token bound")
    if in_string or escaped or depth != 0:
        raise ValidationError("JSON structure is malformed")


def decode_bounded_json(
    value: str | bytes,
    *,
    maximum_bytes: int,
    require_canonical: bool = False,
) -> object:
    """Decode hostile JSON with duplicate, finite, byte, depth, node, and memory guards."""

    if type(value) is bytes:
        try:
            text = value.decode("utf-8")
        except (MemoryError, UnicodeDecodeError):
            raise ValidationError("JSON bytes are not bounded valid UTF-8") from None
    elif type(value) is str:
        text = value
    else:
        raise ValidationError("JSON input must be exact text or bytes")
    _preflight_json_text(text, maximum_bytes=maximum_bytes)
    try:
        decoded = json.loads(
            text,
            object_pairs_hook=_strict_json_pairs,
            parse_constant=_reject_json_constant,
        )
        canonical = canonical_json(decoded)
    except (
        json.JSONDecodeError,
        MemoryError,
        RecursionError,
        TypeError,
        UnicodeError,
        ValueError,
        ValidationError,
    ):
        raise ValidationError("JSON violates its bounded canonical contract") from None
    if require_canonical and canonical != text:
        raise ValidationError("JSON text is not canonical")
    return decoded


@dataclass(frozen=True, slots=True)
class RegistryLimits:
    """Resource and contention bounds enforced by one registry instance."""

    queue_capacity: int = 1_024
    max_request_bytes: int = 262_144
    max_page_size: int = 100
    max_attempts: int = 10
    min_lease_seconds: int = 1
    max_lease_seconds: int = 3_600
    recovery_batch_size: int = 100
    max_artifacts_per_run: int = 100
    max_events_per_job: int = 512
    busy_timeout_ms: int = 250
    verification_timeout_ms: int = 2_000

    def __post_init__(self) -> None:
        bounds = {
            "queue_capacity": (self.queue_capacity, 1, 1_000_000),
            "max_request_bytes": (self.max_request_bytes, 1_024, 16_777_216),
            "max_page_size": (self.max_page_size, 1, 1_000),
            "max_attempts": (self.max_attempts, 1, 100),
            "min_lease_seconds": (self.min_lease_seconds, 1, 86_400),
            "max_lease_seconds": (self.max_lease_seconds, 1, 86_400),
            "recovery_batch_size": (self.recovery_batch_size, 1, 10_000),
            "max_artifacts_per_run": (self.max_artifacts_per_run, 1, 10_000),
            "max_events_per_job": (self.max_events_per_job, 8, 10_000),
            "busy_timeout_ms": (self.busy_timeout_ms, 0, 60_000),
            "verification_timeout_ms": (self.verification_timeout_ms, 1, 60_000),
        }
        for name, (value, lower, upper) in bounds.items():
            if type(value) is not int or not lower <= value <= upper:
                raise ValidationError(f"{name} must be in [{lower}, {upper}]")
        if self.min_lease_seconds > self.max_lease_seconds:
            raise ValidationError("min_lease_seconds may not exceed max_lease_seconds")
        # A maximally retried job can emit SUBMITTED, one CLAIMED event per attempt, one
        # retry/expiry transition per non-final attempt, then CANCEL_REQUESTED and a terminal
        # event while its final lease is live.  Reserving this exact worst-case bound prevents
        # event admission from ever stranding a RUNNING job that still has a legal exit.
        minimum_event_capacity = 2 * self.max_attempts + 2
        if self.max_events_per_job < minimum_event_capacity:
            raise ValidationError(
                "max_events_per_job must be at least "
                f"2 * max_attempts + 2 ({minimum_event_capacity})"
            )


@dataclass(frozen=True, slots=True)
class SubmissionRequest:
    """Immutable job semantics covered by the idempotency conflict digest."""

    kind: str
    payload: Mapping[str, JsonValue]
    priority: int = 0
    max_attempts: int = 3
    schema_version: int = 1

    def __post_init__(self) -> None:
        require_identifier(self.kind, "kind")
        if type(self.priority) is not int or not -100 <= self.priority <= 100:
            raise ValidationError("priority must be an integer in [-100, 100]")
        if type(self.max_attempts) is not int or not 1 <= self.max_attempts <= 100:
            raise ValidationError("max_attempts must be an integer in [1, 100]")
        if type(self.schema_version) is not int or self.schema_version != 1:
            raise ValidationError("schema_version must equal 1")
        frozen = _freeze_json(self.payload)
        if not isinstance(frozen, Mapping):
            raise ValidationError("payload must be a JSON object")
        object.__setattr__(self, "payload", frozen)

    def canonical_json(self) -> str:
        """Serialize all execution semantics in a stable representation."""

        return canonical_json(
            {
                "kind": self.kind,
                "max_attempts": self.max_attempts,
                "payload": self.payload,
                "priority": self.priority,
                "schema_version": self.schema_version,
            }
        )


@dataclass(frozen=True, slots=True)
class JobSnapshot:
    """Redacted durable job state; raw idempotency and lease tokens never appear."""

    sequence: int
    job_id: str
    kind: str
    state: JobState
    priority: int
    attempt_count: int
    max_attempts: int
    created_at: datetime
    updated_at: datetime
    lease_owner: str | None = None
    lease_expires_at: datetime | None = None
    cancel_requested_at: datetime | None = None
    cancel_reason: CancellationReasonCode | None = None
    terminal_at: datetime | None = None
    result_run_id: str | None = None
    failure_code: FailureReasonCode | None = None

    def __post_init__(self) -> None:
        _require_int(self.sequence, "sequence", 1, 2**63 - 1)
        require_identifier(self.job_id, "job_id")
        require_identifier(self.kind, "kind")
        if type(self.state) is not JobState:
            raise ValidationError("state must be a JobState")
        _require_int(self.priority, "priority", -100, 100)
        _require_int(self.attempt_count, "attempt_count", 0, 100)
        _require_int(self.max_attempts, "max_attempts", 1, 100)
        if self.attempt_count > self.max_attempts:
            raise ValidationError("attempt_count may not exceed max_attempts")
        object.__setattr__(self, "created_at", require_utc(self.created_at, "created_at"))
        object.__setattr__(self, "updated_at", require_utc(self.updated_at, "updated_at"))
        if self.updated_at < self.created_at:
            raise ValidationError("updated_at may not precede created_at")
        for field_name in ("lease_expires_at", "cancel_requested_at", "terminal_at"):
            value = getattr(self, field_name)
            if value is not None:
                normalized = require_utc(value, field_name)
                if normalized < self.created_at:
                    raise ValidationError(f"{field_name} may not precede created_at")
                object.__setattr__(self, field_name, normalized)
        if self.lease_expires_at is not None and self.lease_expires_at < self.updated_at:
            raise ValidationError("lease_expires_at may not precede updated_at")
        _require_optional_text(self.lease_owner, "lease_owner", 128)
        if (
            self.cancel_reason is not None
            and type(self.cancel_reason) is not CancellationReasonCode
        ):
            raise ValidationError("cancel_reason must be a CancellationReasonCode or null")
        _require_optional_text(self.result_run_id, "result_run_id", 128)
        if self.failure_code is not None and type(self.failure_code) is not FailureReasonCode:
            raise ValidationError("failure_code must be a FailureReasonCode or null")
        if (self.lease_owner is None) != (self.lease_expires_at is None):
            raise ValidationError(
                "lease owner and expiry must either both be present or both be null"
            )
        if (self.state is JobState.RUNNING) != (self.lease_owner is not None):
            raise ValidationError("only running jobs may expose lease metadata")
        if self.state.terminal != (self.terminal_at is not None):
            raise ValidationError("terminal_at must exactly match terminal state")
        if (self.state is JobState.SUCCEEDED) != (self.result_run_id is not None):
            raise ValidationError("result_run_id must exactly match succeeded state")
        if (self.state is JobState.FAILED) != (self.failure_code is not None):
            raise ValidationError("failure_code must exactly match failed state")
        if (self.cancel_requested_at is None) != (self.cancel_reason is None):
            raise ValidationError("cancellation timestamp and reason must have matching presence")


@dataclass(frozen=True, slots=True)
class LifecycleEvent:
    """One immutable state or lease event."""

    sequence: int
    job_id: str
    kind: EventKind
    from_state: JobState | None
    to_state: JobState
    occurred_at: datetime
    attempt: int
    actor: str
    details: Mapping[str, JsonValue] = field(default_factory=lambda: MappingProxyType({}))

    def __post_init__(self) -> None:
        _require_int(self.sequence, "sequence", 1, 2**63 - 1)
        require_identifier(self.job_id, "job_id")
        if type(self.kind) is not EventKind:
            raise ValidationError("kind must be an EventKind")
        if self.from_state is not None and type(self.from_state) is not JobState:
            raise ValidationError("from_state must be null or a JobState")
        if type(self.to_state) is not JobState:
            raise ValidationError("to_state must be a JobState")
        object.__setattr__(self, "occurred_at", require_utc(self.occurred_at, "occurred_at"))
        _require_int(self.attempt, "attempt", 0, 100)
        require_identifier(self.actor, "actor")
        frozen = _freeze_json(self.details)
        if not isinstance(frozen, Mapping):
            raise ValidationError("event details must be a JSON object")
        object.__setattr__(self, "details", frozen)


@dataclass(frozen=True, slots=True)
class Lease:
    """Secret-bearing lease capability returned only to the successful claimant."""

    job_id: str
    worker_id: str
    token: str = field(repr=False)
    attempt: int = 0
    expires_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    cancel_requested: bool = False

    def __post_init__(self) -> None:
        require_identifier(self.job_id, "job_id")
        require_identifier(self.worker_id, "worker_id")
        if (
            type(self.token) is not str
            or not 32 <= len(self.token) <= 256
            or not self.token.isascii()
        ):
            raise ValidationError("lease token must be bounded ASCII text")
        _require_int(self.attempt, "attempt", 1, 100)
        object.__setattr__(self, "expires_at", require_utc(self.expires_at, "expires_at"))
        if type(self.cancel_requested) is not bool:
            raise ValidationError("cancel_requested must be a boolean")


@dataclass(frozen=True, slots=True)
class ClaimedJob:
    """One secret lease capability paired with its digest-verified immutable request."""

    lease: Lease
    request: SubmissionRequest

    def __post_init__(self) -> None:
        if type(self.lease) is not Lease:
            raise ValidationError("lease must be a Lease")
        if type(self.request) is not SubmissionRequest:
            raise ValidationError("request must be a SubmissionRequest")


@dataclass(frozen=True, slots=True)
class JobCursor:
    """Opaque authenticated stable-snapshot job cursor."""

    token: str = field(repr=False)

    def __post_init__(self) -> None:
        _validate_cursor_token(self.token)


@dataclass(frozen=True, slots=True)
class EventCursor:
    """Opaque authenticated stable-snapshot event cursor."""

    token: str = field(repr=False)

    def __post_init__(self) -> None:
        _validate_cursor_token(self.token)


@dataclass(frozen=True, slots=True)
class RunCursor:
    """Opaque authenticated stable-snapshot run cursor."""

    token: str = field(repr=False)

    def __post_init__(self) -> None:
        _validate_cursor_token(self.token)


@dataclass(frozen=True, slots=True)
class TerminalRunRequest:
    """Caller-supplied immutable evidence for one terminal execution attempt."""

    run_id: str
    evidence_class: EvidenceClass
    started_at: datetime | None = None
    source_commit: str | None = None
    data_identity: str | None = None
    limitation_summary: str | None = None

    def __post_init__(self) -> None:
        require_identifier(self.run_id, "run_id")
        if type(self.evidence_class) is not EvidenceClass:
            raise ValidationError("evidence_class must be an EvidenceClass")
        if self.started_at is not None:
            object.__setattr__(self, "started_at", require_utc(self.started_at, "started_at"))
        _require_optional_text(self.source_commit, "source_commit", 64)
        if self.source_commit is not None and len(self.source_commit) < 7:
            raise ValidationError("source_commit must contain at least 7 characters")
        _require_optional_text(self.data_identity, "data_identity", 256)
        _require_optional_text(self.limitation_summary, "limitation_summary", 4_096)


@dataclass(frozen=True, slots=True, order=True)
class ArtifactLink:
    """One immutable role-to-content link requested for a terminal run."""

    role: str
    artifact_digest: str

    def __post_init__(self) -> None:
        require_identifier(self.role, "artifact role")
        require_digest(self.artifact_digest, "artifact_digest")


@dataclass(frozen=True, slots=True)
class ArtifactCursor:
    """Opaque authenticated stable-snapshot run-artifact cursor."""

    token: str = field(repr=False)

    def __post_init__(self) -> None:
        _validate_cursor_token(self.token)


@dataclass(frozen=True, slots=True)
class Page[ItemT, CursorT]:
    """Immutable bounded keyset page."""

    items: tuple[ItemT, ...]
    next_cursor: CursorT | None

    def __post_init__(self) -> None:
        if type(self.items) is not tuple or len(self.items) > 1_000:
            raise ValidationError("page items must be a tuple of at most 1,000 entries")


@dataclass(frozen=True, slots=True)
class RunSnapshot:
    """Redacted immutable execution-run read model."""

    sequence: int
    run_id: str
    job_id: str
    attempt: int
    status: RunStatus
    evidence_class: EvidenceClass
    schema_version: int
    created_at: datetime
    started_at: datetime | None
    ended_at: datetime
    source_commit: str | None
    data_identity: str | None
    limitation_summary: str | None

    def __post_init__(self) -> None:
        _require_int(self.sequence, "sequence", 1, 2**63 - 1)
        require_identifier(self.run_id, "run_id")
        require_identifier(self.job_id, "job_id")
        _require_int(self.attempt, "attempt", 1, 100)
        if type(self.status) is not RunStatus:
            raise ValidationError("status must be a RunStatus")
        if type(self.evidence_class) is not EvidenceClass:
            raise ValidationError("evidence_class must be an EvidenceClass")
        _require_int(self.schema_version, "schema_version", 1, 1)
        object.__setattr__(self, "created_at", require_utc(self.created_at, "created_at"))
        if self.started_at is not None:
            object.__setattr__(self, "started_at", require_utc(self.started_at, "started_at"))
        object.__setattr__(self, "ended_at", require_utc(self.ended_at, "ended_at"))
        if self.started_at is not None and self.started_at < self.created_at:
            raise ValidationError("started_at must not precede created_at")
        if self.ended_at < (self.started_at or self.created_at):
            raise ValidationError("ended_at must not precede created_at or started_at")
        _require_optional_text(self.source_commit, "source_commit", 64)
        if self.source_commit is not None and len(self.source_commit) < 7:
            raise ValidationError("source_commit must contain at least 7 characters")
        _require_optional_text(self.data_identity, "data_identity", 256)
        _require_optional_text(self.limitation_summary, "limitation_summary", 4_096)


@dataclass(frozen=True, slots=True)
class ArtifactMetadata:
    """Content-addressed artifact metadata without host-absolute paths."""

    digest: str
    artifact_class: ArtifactClass
    byte_size: int
    media_type: str
    storage_relpath: str
    created_at: datetime
    pinned: bool = False

    def __post_init__(self) -> None:
        require_digest(self.digest)
        if type(self.artifact_class) is not ArtifactClass:
            raise ValidationError("artifact_class must be an ArtifactClass")
        if type(self.byte_size) is not int or not 0 <= self.byte_size <= 2**63 - 1:
            raise ValidationError("byte_size must be a non-negative signed 64-bit integer")
        if (
            type(self.media_type) is not str
            or not self.media_type.isascii()
            or not 3 <= len(self.media_type) <= _MAX_MEDIA_TYPE_BYTES
            or not _MEDIA_TYPE.fullmatch(self.media_type)
        ):
            raise ValidationError(
                "media_type must be an ASCII lowercase type/subtype at most 128 bytes"
            )
        path = self.storage_relpath
        path_size = None if type(path) is not str else require_utf8_size(path, "storage_relpath")
        if (
            type(path) is not str
            or not path
            or path_size is None
            or path_size > 1_024
            or path.startswith(("/", "\\"))
            or ".." in path.replace("\\", "/").split("/")
        ):
            raise ValidationError("storage_relpath must be a bounded safe relative path")
        object.__setattr__(self, "created_at", require_utc(self.created_at, "created_at"))
        if type(self.pinned) is not bool:
            raise ValidationError("pinned must be a boolean")


@dataclass(frozen=True, slots=True)
class RetentionPlan:
    """Immutable deletion intent; execution and tombstoning remain separate operations."""

    artifact_digests: tuple[str, ...]
    planned_at: datetime
    reason: str

    def __post_init__(self) -> None:
        if not self.artifact_digests or len(self.artifact_digests) > 10_000:
            raise ValidationError("artifact_digests must contain between 1 and 10,000 entries")
        if tuple(sorted(set(self.artifact_digests))) != self.artifact_digests:
            raise ValidationError("artifact_digests must be unique and sorted")
        for digest in self.artifact_digests:
            require_digest(digest, "artifact_digest")
        object.__setattr__(self, "planned_at", require_utc(self.planned_at, "planned_at"))
        if (
            type(self.reason) is not str
            or not 1 <= require_utf8_size(self.reason, "reason") <= 1_024
        ):
            raise ValidationError("reason must contain between 1 and 1,024 UTF-8 bytes")


@dataclass(frozen=True, slots=True)
class RegistryReadiness:
    """Read-only readiness verdict suitable for fail-closed health boundaries."""

    ready: bool
    schema_version: int | None
    journal_mode: str | None
    reason: str | None = None

    def __post_init__(self) -> None:
        if type(self.ready) is not bool:
            raise ValidationError("ready must be a boolean")
        if self.schema_version is not None:
            _require_int(self.schema_version, "schema_version", 0, 2**31 - 1)
        if self.journal_mode is not None and self.journal_mode not in {"wal", "delete"}:
            raise ValidationError("journal_mode is unknown")
        _require_optional_text(self.reason, "reason", 1_024)
        if self.ready and self.reason is not None:
            raise ValidationError("ready verdicts may not contain a failure reason")
