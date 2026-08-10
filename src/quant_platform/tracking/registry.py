"""Durable SQLite job registry with idempotency, leases, and guarded transitions."""

from __future__ import annotations

import base64
import binascii
import hashlib
import hmac
import inspect
import os
import secrets
import sqlite3
import uuid
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import NoReturn, Protocol, cast

from quant_platform.tracking.cas import ArtifactStoreError, PublishedArtifact
from quant_platform.tracking.contracts import (
    ArtifactClass,
    ArtifactCursor,
    ArtifactLink,
    ArtifactMetadata,
    BusyError,
    CancellationReasonCode,
    CapacityError,
    ClaimedJob,
    Clock,
    ConflictError,
    EventCursor,
    EventKind,
    EvidenceClass,
    FailureReasonCode,
    IntegrityError,
    InvalidCursorError,
    InvalidTransitionError,
    JobCursor,
    JobSnapshot,
    JobState,
    JsonValue,
    Lease,
    LeaseLostError,
    LifecycleEvent,
    NotFoundError,
    Page,
    RegistryError,
    RegistryLimits,
    RegistryReadiness,
    RetentionPendingError,
    RetentionPlanAuthenticationError,
    RunCursor,
    RunSnapshot,
    RunStatus,
    SubmissionRequest,
    TerminalRunRequest,
    ValidationError,
    canonical_json,
    decode_bounded_json,
    parse_stored_utc,
    require_digest,
    require_failure_summary,
    require_identifier,
    require_stored_int,
    require_utc,
    require_utf8_size,
)
from quant_platform.tracking.migrations import (
    initialize_database,
    open_database,
    probe_database,
    verify_database_binding,
)

_ACTIVE_STATES = (JobState.QUEUED.value, JobState.RUNNING.value)
_IDEMPOTENCY_DOMAIN = b"signalattice.registry.idempotency.v1\0"
_LEASE_DOMAIN = b"signalattice.registry.lease.v1\0"
_JOB_CURSOR_DOMAIN = b"signalattice.registry.job-cursor.v1\0"
_EVENT_CURSOR_DOMAIN = b"signalattice.registry.event-cursor.v1\0"
_RUN_CURSOR_DOMAIN = b"signalattice.registry.run-cursor.v1\0"
_ARTIFACT_CURSOR_DOMAIN = b"signalattice.registry.artifact-cursor.v1\0"
_KEY_VERIFIER_DOMAIN = b"signalattice.registry.key-verifier.v1\0"
_RETENTION_PLAN_DOMAIN = b"signalattice.registry.retention-plan.v1\0"
_CURSOR_VERSION = "v1"
_MAX_RETENTION_PAYLOAD_BYTES = 1_048_576
_CONTRACT_VALIDATION_INSTANT = datetime(1970, 1, 1, tzinfo=UTC)


class ArtifactVerifier(Protocol):
    """Narrow CAS authority required before durable metadata or links are committed."""

    @property
    def store_id(self) -> str:
        """Return the initialized canonical CAS store identity without artifact I/O."""
        ...

    def verify(self, artifact: PublishedArtifact) -> None:
        """Verify canonical bytes, digest, size, and store identity or raise."""
        ...


@dataclass(frozen=True, slots=True)
class _ArtifactPreflight:
    """Full CAS verification completed before acquiring the SQLite writer lock."""

    verifier: ArtifactVerifier
    store_id: str
    artifacts: tuple[ArtifactMetadata, ...] = ()


class SystemClock:
    """Production UTC wall-clock authority."""

    def now(self) -> datetime:
        """Return the current UTC instant."""

        return datetime.now(UTC)


def _has_static_callable(value: object, name: str) -> bool:
    """Inspect one protocol method without executing attacker-controlled descriptors."""

    try:
        candidate = inspect.getattr_static(value, name)
    except Exception:
        return False
    return callable(candidate)


def _db_time(value: datetime, field_name: str = "now") -> str:
    return require_utc(value, field_name).isoformat(timespec="microseconds").replace("+00:00", "Z")


def _parse_time(value: object, field_name: str) -> datetime:
    return parse_stored_utc(value, field_name)


def _optional_time(value: object, field_name: str) -> datetime | None:
    if value is None:
        return None
    return _parse_time(value, field_name)


def _stored_text(value: object, field_name: str, maximum: int) -> str:
    if type(value) is not str:
        raise IntegrityError(f"stored {field_name} is not bounded text")
    try:
        size = require_utf8_size(value, field_name)
    except ValidationError:
        raise IntegrityError(f"stored {field_name} is not bounded text") from None
    if not 1 <= size <= maximum:
        raise IntegrityError(f"stored {field_name} is not bounded text")
    return value


def _digest(secret: bytes, domain: bytes, value: str) -> str:
    return hmac.digest(secret, domain + value.encode("utf-8"), "sha256").hex()


def _validate_digest_secret(value: bytes) -> bytes:
    if type(value) is not bytes or not 32 <= len(value) <= 4_096:
        raise ValidationError("digest_secret must contain between 32 and 4,096 bytes")
    return bytes(value)


def _key_verifier(secret: bytes) -> str:
    return hmac.digest(secret, _KEY_VERIFIER_DOMAIN + b"bound", "sha256").hex()


def _validate_store_id(value: str) -> str:
    try:
        return require_digest(value, "store_id")
    except ValidationError:
        raise ValidationError("store_id must be 64 lowercase hexadecimal characters") from None


def _canonical_retention_payload(value: bytes) -> bytes:
    if type(value) is not bytes or not 2 <= len(value) <= _MAX_RETENTION_PAYLOAD_BYTES:
        raise ValidationError(
            "retention payload must contain between 2 and 1,048,576 canonical JSON bytes"
        )
    try:
        decode_bounded_json(
            value,
            maximum_bytes=_MAX_RETENTION_PAYLOAD_BYTES,
            require_canonical=True,
        )
    except ValidationError:
        raise ValidationError("retention payload must be canonical JSON") from None
    return bytes(value)


def _validate_secret(value: str, field_name: str) -> str:
    if type(value) is not str:
        raise ValidationError(f"{field_name} must be a string")
    length = require_utf8_size(value, field_name)
    if not 8 <= length <= 256 or "\x00" in value:
        raise ValidationError(f"{field_name} must contain between 8 and 256 UTF-8 bytes")
    return value


def _bounded_text(value: str, field_name: str, maximum: int) -> str:
    if type(value) is not str or not 1 <= require_utf8_size(value, field_name) <= maximum:
        raise ValidationError(f"{field_name} must contain between 1 and {maximum} UTF-8 bytes")
    return value


class CursorCodec:
    """Versioned HMAC codec for bounded, filter-bound snapshot cursors.

    Tokens authenticate but do not encrypt their small canonical payload.  Callers must treat the
    token as opaque and must never place secrets or user payloads in a cursor filter.
    """

    def __init__(self, digest_secret: bytes) -> None:
        self._secret = _validate_digest_secret(digest_secret)

    def encode_job(
        self,
        snapshot: int,
        event_snapshot: int,
        after: int,
        state: JobState | None,
    ) -> JobCursor:
        """Encode one job cursor bound to its optional state filter."""

        self._validate_position(snapshot, after)
        self._validate_position(event_snapshot, 0)
        return JobCursor(
            self._encode(
                _JOB_CURSOR_DOMAIN,
                {
                    "after": after,
                    "event_snapshot": event_snapshot,
                    "filter": None if state is None else state.value,
                    "snapshot": snapshot,
                },
            )
        )

    def decode_job(self, cursor: JobCursor) -> tuple[int, int, int, JobState | None]:
        """Authenticate and decode one job cursor."""

        if type(cursor) is not JobCursor:
            raise InvalidCursorError("job cursor has the wrong contract type")
        payload = self._decode_payload(cursor.token, _JOB_CURSOR_DOMAIN)
        if set(payload) != {"after", "event_snapshot", "filter", "snapshot"}:
            raise InvalidCursorError("job cursor payload has an unexpected shape")
        snapshot = payload["snapshot"]
        event_snapshot = payload["event_snapshot"]
        after = payload["after"]
        filter_value = payload["filter"]
        self._validate_position(snapshot, after)
        self._validate_position(event_snapshot, 0)
        if filter_value is not None and type(filter_value) is not str:
            raise InvalidCursorError("job cursor filter must be text or null")
        try:
            state = None if filter_value is None else JobState(filter_value)
        except ValueError as exc:
            raise InvalidCursorError("job cursor contains an unknown state filter") from exc
        return cast(int, snapshot), cast(int, event_snapshot), cast(int, after), state

    def encode_event(self, snapshot: int, after: int, job_id: str | None) -> EventCursor:
        """Encode one event cursor bound to its optional job filter."""

        self._validate_position(snapshot, after)
        if job_id is not None:
            require_identifier(job_id, "job_id")
        return EventCursor(
            self._encode(
                _EVENT_CURSOR_DOMAIN,
                {"after": after, "filter": job_id, "snapshot": snapshot},
            )
        )

    def decode_event(self, cursor: EventCursor) -> tuple[int, int, str | None]:
        """Authenticate and decode one event cursor."""

        if type(cursor) is not EventCursor:
            raise InvalidCursorError("event cursor has the wrong contract type")
        snapshot, after, job_id = self._decode(cursor.token, _EVENT_CURSOR_DOMAIN)
        if job_id is not None:
            try:
                require_identifier(job_id, "cursor job_id")
            except ValidationError as exc:
                raise InvalidCursorError("event cursor contains an invalid job filter") from exc
        return snapshot, after, job_id

    def encode_run(
        self,
        snapshot: int,
        after: int,
        filter_value: str | None = None,
    ) -> RunCursor:
        """Encode a run cursor bound to a caller-owned canonical filter identity."""

        self._validate_position(snapshot, after)
        if filter_value is not None:
            _bounded_text(filter_value, "filter_value", 128)
        return RunCursor(
            self._encode(
                _RUN_CURSOR_DOMAIN,
                {"after": after, "filter": filter_value, "snapshot": snapshot},
            )
        )

    def decode_run(self, cursor: RunCursor) -> tuple[int, int, str | None]:
        """Authenticate and decode one run cursor for a read-port implementation."""

        if type(cursor) is not RunCursor:
            raise InvalidCursorError("run cursor has the wrong contract type")
        snapshot, after, filter_value = self._decode(cursor.token, _RUN_CURSOR_DOMAIN)
        if filter_value is not None:
            try:
                _bounded_text(filter_value, "cursor filter", 128)
            except ValidationError as exc:
                raise InvalidCursorError("run cursor contains an invalid filter") from exc
        return snapshot, after, filter_value

    def encode_artifact(self, snapshot: int, after: int, run_id: str) -> ArtifactCursor:
        """Encode a run-artifact cursor bound to exactly one run identifier."""

        self._validate_position(snapshot, after)
        require_identifier(run_id, "run_id")
        return ArtifactCursor(
            self._encode(
                _ARTIFACT_CURSOR_DOMAIN,
                {"after": after, "filter": run_id, "snapshot": snapshot},
            )
        )

    def decode_artifact(self, cursor: ArtifactCursor) -> tuple[int, int, str]:
        """Authenticate and decode one run-bound artifact cursor."""

        if type(cursor) is not ArtifactCursor:
            raise InvalidCursorError("artifact cursor has the wrong contract type")
        snapshot, after, run_id = self._decode(cursor.token, _ARTIFACT_CURSOR_DOMAIN)
        if run_id is None:
            raise InvalidCursorError("artifact cursor is missing its run filter")
        try:
            require_identifier(run_id, "cursor run_id")
        except ValidationError as exc:
            raise InvalidCursorError("artifact cursor contains an invalid run filter") from exc
        return snapshot, after, run_id

    def _encode(self, domain: bytes, payload: object) -> str:
        encoded = canonical_json(payload).encode("utf-8")
        if len(encoded) > 512:
            raise InvalidCursorError("cursor payload exceeds 512 bytes")
        signature = hmac.digest(self._secret, domain + encoded, "sha256")
        body = base64.urlsafe_b64encode(encoded).rstrip(b"=").decode("ascii")
        mac = base64.urlsafe_b64encode(signature).rstrip(b"=").decode("ascii")
        return f"{_CURSOR_VERSION}.{body}.{mac}"

    def _decode(self, token: str, domain: bytes) -> tuple[int, int, str | None]:
        payload = self._decode_payload(token, domain)
        if set(payload) != {"after", "filter", "snapshot"}:
            raise InvalidCursorError("cursor payload has an unexpected shape")
        snapshot = payload["snapshot"]
        after = payload["after"]
        filter_value = payload["filter"]
        self._validate_position(snapshot, after)
        if filter_value is not None and type(filter_value) is not str:
            raise InvalidCursorError("cursor filter must be text or null")
        return cast(int, snapshot), cast(int, after), filter_value

    def _decode_payload(self, token: str, domain: bytes) -> dict[str, object]:
        try:
            version, body_text, mac_text = token.split(".")
        except ValueError as exc:
            raise InvalidCursorError("cursor token has an invalid envelope") from exc
        if version != _CURSOR_VERSION:
            raise InvalidCursorError("cursor token version is unsupported")
        body = self._decode_segment(body_text, maximum=768)
        signature = self._decode_segment(mac_text, maximum=64)
        if len(body) > 512 or len(signature) != hashlib.sha256().digest_size:
            raise InvalidCursorError("cursor token has invalid component lengths")
        expected = hmac.digest(self._secret, domain + body, "sha256")
        if not hmac.compare_digest(signature, expected):
            raise InvalidCursorError("cursor token authentication failed")
        try:
            payload = decode_bounded_json(body, maximum_bytes=512, require_canonical=True)
        except ValidationError:
            raise InvalidCursorError("cursor payload is not canonical JSON") from None
        if type(payload) is not dict or any(type(key) is not str for key in payload):
            raise InvalidCursorError("cursor payload has an unexpected shape")
        return cast(dict[str, object], payload)

    @staticmethod
    def _decode_segment(value: str, *, maximum: int) -> bytes:
        if not value or len(value) > maximum or not value.isascii():
            raise InvalidCursorError("cursor component is malformed")
        padding = "=" * ((4 - len(value) % 4) % 4)
        try:
            return base64.b64decode(value + padding, altchars=b"-_", validate=True)
        except (binascii.Error, ValueError) as exc:
            raise InvalidCursorError("cursor component is not valid base64url") from exc

    @staticmethod
    def _validate_position(snapshot: object, after: object) -> None:
        if (
            type(snapshot) is not int
            or type(after) is not int
            or snapshot < 0
            or after < 0
            or after > snapshot
            or snapshot > 2**63 - 1
        ):
            raise InvalidCursorError("cursor sequences are invalid")


class RunRegistry:
    """Local-first control-plane kernel backed by one SQLite database.

    Call :meth:`initialize` explicitly during deployment or process startup.  Operational methods
    never migrate implicitly, and :meth:`probe_readiness` is strictly read-only.  Every mutation
    holds ``BEGIN IMMEDIATE`` only for bounded indexed reads and writes, yielding at-least-once
    execution with one active lease winner rather than exactly-once execution.
    """

    def __init__(
        self,
        path: str | Path,
        *,
        digest_secret: bytes,
        limits: RegistryLimits | None = None,
        clock: Clock | None = None,
        artifact_verifier: ArtifactVerifier | None = None,
    ) -> None:
        try:
            raw_path = os.fspath(path)
        except Exception:
            raise ValidationError("path must expose a valid filesystem path") from None
        if type(raw_path) is not str:
            raise ValidationError("path must resolve to exact text")
        self.path = Path(raw_path)
        self._digest_secret = _validate_digest_secret(digest_secret)
        self._key_verifier = _key_verifier(self._digest_secret)
        self.cursor_codec = CursorCodec(self._digest_secret)
        if limits is None:
            self.limits = RegistryLimits()
        elif type(limits) is RegistryLimits:
            self.limits = limits
        else:
            raise ValidationError("limits must be a RegistryLimits")
        candidate_clock: Clock = SystemClock() if clock is None else clock
        if not _has_static_callable(candidate_clock, "now"):
            raise ValidationError("clock must expose a callable now method")
        if artifact_verifier is not None and not _has_static_callable(
            artifact_verifier,
            "verify",
        ):
            raise ValidationError("artifact_verifier must expose a callable verify method")
        self._clock = candidate_clock
        self._artifact_verifier = artifact_verifier

    def initialize(self) -> None:
        """Apply verified forward-only migrations; safe to call repeatedly."""

        initialize_database(
            self.path,
            busy_timeout_ms=self.limits.busy_timeout_ms,
            verification_timeout_ms=self.limits.verification_timeout_ms,
            key_verifier=self._key_verifier,
            retention_plan_authenticator=self._authenticate_retention_plan,
        )
        if self._artifact_verifier is not None:
            self.bind_artifact_store(self._artifact_verifier_store_id())

    @property
    def artifact_verifier(self) -> ArtifactVerifier | None:
        """Return the exact verifier object supplied at construction, without adaptation."""

        return self._artifact_verifier

    @property
    def registry_id(self) -> str:
        """Return this database's verified immutable path-free instance identity."""

        connection = self._connect(readonly=True)
        try:
            return verify_database_binding(connection, key_verifier=self._key_verifier)
        finally:
            connection.close()

    @property
    def artifact_store_id(self) -> str | None:
        """Return the immutable CAS binding, or ``None`` for a registry-only database."""

        connection = self._connect(readonly=True)
        try:
            return self._bound_artifact_store_id(connection)
        finally:
            connection.close()

    def bind_artifact_store(self, store_id: str) -> None:
        """Bind exactly one CAS identity, idempotently, without requiring a verifier object."""

        canonical = _validate_store_id(store_id)
        with self._write() as connection:
            observed = self._bound_artifact_store_id(connection)
            if observed is None:
                connection.execute(
                    "INSERT INTO sl_registry_cas_binding(singleton, store_id) VALUES(1, ?)",
                    (canonical,),
                )
            elif not hmac.compare_digest(observed, canonical):
                raise ConflictError("registry is already bound to another CAS store identity")

    def sign_retention_payload(self, payload: bytes, *, store_id: str) -> str:
        """Return an authority-, instance-, and CAS-bound HMAC plan identity."""

        canonical = _canonical_retention_payload(payload)
        requested_store_id = _validate_store_id(store_id)
        connection = self._connect(readonly=True)
        try:
            registry_id = verify_database_binding(connection, key_verifier=self._key_verifier)
            self._require_artifact_store_binding(connection, requested_store_id)
        finally:
            connection.close()
        return self._retention_plan_digest(canonical, registry_id, requested_store_id)

    def verify_retention_payload_signature(
        self,
        payload: bytes,
        *,
        store_id: str,
        signature: str,
    ) -> None:
        """Fail closed unless ``signature`` is this registry's HMAC plan identity."""

        observed = require_digest(signature, "signature")
        expected = self.sign_retention_payload(payload, store_id=store_id)
        if not hmac.compare_digest(observed, expected):
            raise RetentionPlanAuthenticationError("retention plan authentication failed")

    def probe_readiness(self) -> RegistryReadiness:
        """Return a read-only fail-closed schema and integrity verdict."""

        return probe_database(
            self.path,
            busy_timeout_ms=self.limits.busy_timeout_ms,
            verification_timeout_ms=self.limits.verification_timeout_ms,
            key_verifier=self._key_verifier,
            retention_plan_authenticator=self._authenticate_retention_plan,
        )

    def _connect(
        self,
        *,
        readonly: bool = False,
        busy_timeout_ms: int | None = None,
    ) -> sqlite3.Connection:
        """Open authenticated state with an optional stricter lock-wait ceiling."""

        effective_busy_timeout = self.limits.busy_timeout_ms
        if busy_timeout_ms is not None:
            if (
                type(busy_timeout_ms) is not int
                or not 0 <= busy_timeout_ms <= effective_busy_timeout
            ):
                raise ValidationError(
                    "busy_timeout_ms override must be an integer in "
                    f"[0, {effective_busy_timeout}]"
                )
            effective_busy_timeout = busy_timeout_ms
        connection = open_database(
            self.path,
            busy_timeout_ms=effective_busy_timeout,
            readonly=readonly,
        )
        try:
            verify_database_binding(connection, key_verifier=self._key_verifier)
        except RegistryError:
            connection.close()
            raise
        return connection

    @staticmethod
    def _bound_artifact_store_id(connection: sqlite3.Connection) -> str | None:
        rows = connection.execute(
            "SELECT singleton, store_id FROM sl_registry_cas_binding"
        ).fetchall()
        if not rows:
            return None
        if (
            len(rows) != 1
            or require_stored_int(rows[0]["singleton"], "CAS binding singleton", 1, 1) != 1
        ):
            raise IntegrityError("registry CAS binding metadata is malformed")
        value = rows[0]["store_id"]
        if type(value) is not str:
            raise IntegrityError("registry CAS store identity is not text")
        try:
            return _validate_store_id(value)
        except ValidationError:
            raise IntegrityError("registry CAS store identity is not canonical") from None

    @classmethod
    def _require_artifact_store_binding(
        cls,
        connection: sqlite3.Connection,
        store_id: str,
    ) -> None:
        observed = cls._bound_artifact_store_id(connection)
        if observed is None:
            raise IntegrityError("registry is not bound to a CAS store identity")
        if not hmac.compare_digest(observed, _validate_store_id(store_id)):
            raise IntegrityError("registry CAS store identity does not match this operation")

    def _artifact_verifier_store_id(self) -> str:
        verifier = self._artifact_verifier
        if verifier is None:
            raise IntegrityError("artifact operations require a bound CAS verifier")
        try:
            value = verifier.store_id
        except Exception:
            raise IntegrityError("artifact verifier has no initialized CAS identity") from None
        try:
            return _validate_store_id(value)
        except ValidationError:
            raise IntegrityError("artifact verifier CAS identity is not canonical") from None

    def _retention_plan_digest(
        self,
        payload: bytes,
        registry_id: str,
        store_id: str,
    ) -> str:
        message = (
            _RETENTION_PLAN_DOMAIN
            + registry_id.encode("ascii")
            + b"\0"
            + store_id.encode("ascii")
            + b"\0"
            + payload
        )
        return hmac.digest(self._digest_secret, message, "sha256").hex()

    def _authenticate_retention_plan(
        self,
        payload: bytes,
        registry_id: str,
        store_id: str,
        signature: str,
    ) -> bool:
        """Authenticate snapshot-bound retention bytes without opening another connection."""

        expected = self._retention_plan_digest(payload, registry_id, store_id)
        return hmac.compare_digest(signature, expected)

    def _now(self) -> datetime:
        """Read and validate one injected authority instant."""

        try:
            value = self._clock.now()
            return require_utc(value, "clock.now")
        except Exception:
            raise IntegrityError("registry time authority failed") from None

    @contextmanager
    def _write(self) -> Iterator[sqlite3.Connection]:
        connection = self._connect()
        primary_error: BaseException | None = None
        try:
            connection.execute("BEGIN IMMEDIATE")
            yield connection
            connection.execute("COMMIT")
        except RegistryError as exc:
            primary_error = exc
            raise
        except sqlite3.Error as exc:
            try:
                self._raise_sqlite(exc)
            except RegistryError as mapped:
                primary_error = mapped
                raise
        except BaseException as exc:
            primary_error = exc
            raise
        finally:
            cleanup_failures: list[BaseException] = []
            try:
                if connection.in_transaction:
                    connection.execute("ROLLBACK")
            except sqlite3.Error as exc:
                cleanup_failures.append(exc)
            try:
                connection.close()
            except sqlite3.Error as exc:
                cleanup_failures.append(exc)
            if cleanup_failures:
                if primary_error is not None:
                    names = ",".join(type(error).__name__ for error in cleanup_failures)
                    primary_error.add_note(f"registry write cleanup also failed: {names}")
                else:
                    raise IntegrityError("registry write cleanup failed") from cleanup_failures[0]

    @staticmethod
    def _raise_sqlite(exc: sqlite3.Error) -> NoReturn:
        code = getattr(exc, "sqlite_errorcode", None)
        if type(code) is int and (code & 0xFF) in {sqlite3.SQLITE_BUSY, sqlite3.SQLITE_LOCKED}:
            raise BusyError("registry write remained busy past its configured bound") from None
        raise IntegrityError("SQLite rejected a registry operation") from None

    def submit(
        self,
        request: SubmissionRequest,
        *,
        idempotency_key: str,
    ) -> JobSnapshot:
        """Durably enqueue once, or return the prior job for identical key semantics.

        Only a domain-separated SHA-256 digest of ``idempotency_key`` is persisted.  Reusing the
        key with a different canonical request raises :class:`ConflictError`.
        """

        if type(request) is not SubmissionRequest:
            raise ValidationError("request must be a SubmissionRequest")
        key = _validate_secret(idempotency_key, "idempotency_key")
        canonical = request.canonical_json()
        encoded = canonical.encode("utf-8")
        if len(encoded) > self.limits.max_request_bytes:
            raise ValidationError(
                f"canonical request exceeds the {self.limits.max_request_bytes}-byte bound"
            )
        if request.max_attempts > self.limits.max_attempts:
            raise ValidationError("request max_attempts exceeds registry limits")
        request_digest = hashlib.sha256(encoded).hexdigest()
        key_digest = _digest(self._digest_secret, _IDEMPOTENCY_DOMAIN, key)
        with self._write() as connection:
            prior = connection.execute(
                "SELECT * FROM sl_registry_jobs WHERE idempotency_digest = ?", (key_digest,)
            ).fetchone()
            if prior is not None:
                self._request(prior)
                if _stored_text(prior["request_digest"], "request_digest", 64) != request_digest:
                    raise ConflictError("idempotency key was already used for another request")
                return self._job(prior)
            active = require_stored_int(
                connection.execute(
                    "SELECT count(*) FROM sl_registry_jobs WHERE state IN (?, ?)", _ACTIVE_STATES
                ).fetchone()[0],
                "active job count",
                0,
                2**63 - 1,
            )
            if active >= self.limits.queue_capacity:
                raise CapacityError("durable queue capacity has been reached")
            instant = _db_time(self._now())
            job_id = uuid.uuid4().hex
            connection.execute(
                """
                INSERT INTO sl_registry_jobs(
                    job_id, kind, request_schema_version, request_json, request_digest,
                    idempotency_digest, state, priority, max_attempts, created_at, updated_at
                ) VALUES(?,?,?,?,?,?,?,?,?,?,?)
                """,
                (
                    job_id,
                    request.kind,
                    request.schema_version,
                    canonical,
                    request_digest,
                    key_digest,
                    JobState.QUEUED.value,
                    request.priority,
                    request.max_attempts,
                    instant,
                    instant,
                ),
            )
            self._event(
                connection,
                job_id=job_id,
                kind=EventKind.SUBMITTED,
                from_state=None,
                to_state=JobState.QUEUED,
                occurred_at=instant,
                attempt=0,
                actor="submitter",
                details={"request_digest": request_digest},
            )
            row = connection.execute(
                "SELECT * FROM sl_registry_jobs WHERE job_id = ?", (job_id,)
            ).fetchone()
            if row is None:
                raise IntegrityError("submitted job disappeared inside its transaction")
            return self._job(cast(sqlite3.Row, row))

    def get_job(self, job_id: str) -> JobSnapshot:
        """Return one redacted job snapshot."""

        require_identifier(job_id, "job_id")
        connection = self._connect(readonly=True)
        try:
            row = connection.execute(
                "SELECT * FROM sl_registry_jobs WHERE job_id = ?", (job_id,)
            ).fetchone()
            if row is None:
                raise NotFoundError(f"job {job_id!r} does not exist")
            return self._job(row)
        except sqlite3.Error as exc:
            self._raise_sqlite(exc)
        finally:
            connection.close()

    def list_jobs(
        self,
        *,
        page_size: int = 50,
        cursor: JobCursor | None = None,
        state: JobState | None = None,
    ) -> Page[JobSnapshot, JobCursor]:
        """List a stable snapshot using an ascending sequence keyset."""

        self._validate_page_size(page_size)
        if state is not None and type(state) is not JobState:
            raise ValidationError("state must be a JobState")
        connection = self._connect(readonly=True)
        try:
            connection.execute("BEGIN")
            current_job_max = self._sequence_snapshot(connection, "sl_registry_jobs")
            current_event_max = self._sequence_snapshot(connection, "sl_registry_events")
            if cursor is None:
                snapshot = current_job_max
                event_snapshot = current_event_max
                after = 0
            else:
                snapshot, event_snapshot, after, cursor_state = self.cursor_codec.decode_job(cursor)
                if cursor_state is not state:
                    raise InvalidCursorError("cursor state filter does not match this request")
                if current_job_max < snapshot or current_event_max < event_snapshot:
                    raise InvalidCursorError("job cursor snapshot source regressed")
                if state is not None:
                    changed = connection.execute(
                        """
                        SELECT 1
                        FROM sl_registry_events AS events
                        JOIN sl_registry_jobs AS jobs ON jobs.job_id = events.job_id
                        WHERE jobs.sequence > ? AND jobs.sequence <= ?
                          AND events.sequence > ?
                        LIMIT 1
                        """,
                        (after, snapshot, event_snapshot),
                    ).fetchone()
                    if changed is not None:
                        raise InvalidCursorError(
                            "state-filtered snapshot changed; restart pagination"
                        )
            parameters: list[object] = [after, snapshot]
            where = "sequence > ? AND sequence <= ?"
            if state is not None:
                where += " AND state = ?"
                parameters.append(state.value)
            parameters.append(page_size + 1)
            rows = connection.execute(
                f"SELECT * FROM sl_registry_jobs WHERE {where} ORDER BY sequence LIMIT ?",
                parameters,
            ).fetchall()
            visible = rows[:page_size]
            items = tuple(self._job(row) for row in visible)
            next_cursor = None
            if len(rows) > page_size and visible:
                next_cursor = self.cursor_codec.encode_job(
                    snapshot,
                    event_snapshot,
                    require_stored_int(
                        visible[-1]["sequence"],
                        "job sequence",
                        1,
                        2**63 - 1,
                    ),
                    state,
                )
            return Page(items, next_cursor)
        except sqlite3.Error as exc:
            self._raise_sqlite(exc)
        finally:
            connection.close()

    def list_events(
        self,
        *,
        job_id: str | None = None,
        page_size: int = 50,
        cursor: EventCursor | None = None,
    ) -> Page[LifecycleEvent, EventCursor]:
        """List append-only events through a stable sequence keyset."""

        self._validate_page_size(page_size)
        if job_id is not None:
            require_identifier(job_id, "job_id")
        connection = self._connect(readonly=True)
        try:
            current_event_max = self._sequence_snapshot(connection, "sl_registry_events")
            if cursor is None:
                snapshot = current_event_max
                after = 0
            else:
                snapshot, after, cursor_job_id = self.cursor_codec.decode_event(cursor)
                if cursor_job_id != job_id:
                    raise InvalidCursorError("cursor job filter does not match this request")
                if current_event_max < snapshot:
                    raise InvalidCursorError("event cursor snapshot source regressed")
            parameters: list[object] = [after, snapshot]
            where = "sequence > ? AND sequence <= ?"
            if job_id is not None:
                where += " AND job_id = ?"
                parameters.append(job_id)
            parameters.append(page_size + 1)
            rows = connection.execute(
                f"SELECT * FROM sl_registry_events WHERE {where} ORDER BY sequence LIMIT ?",
                parameters,
            ).fetchall()
            visible = rows[:page_size]
            items = tuple(self._lifecycle_event(row) for row in visible)
            next_cursor = None
            if len(rows) > page_size and visible:
                next_cursor = self.cursor_codec.encode_event(
                    snapshot,
                    require_stored_int(
                        visible[-1]["sequence"],
                        "event sequence",
                        1,
                        2**63 - 1,
                    ),
                    job_id,
                )
            return Page(items, next_cursor)
        except sqlite3.Error as exc:
            self._raise_sqlite(exc)
        finally:
            connection.close()

    @staticmethod
    def _sequence_snapshot(connection: sqlite3.Connection, table: str) -> int:
        """Return a bounded high-water mark only when every durable sequence is in-domain."""

        if table not in {"sl_registry_jobs", "sl_registry_events"}:
            raise IntegrityError("unsupported registry snapshot source")
        row = cast(
            sqlite3.Row,
            connection.execute(
                f"SELECT MIN(sequence) AS minimum, MAX(sequence) AS maximum FROM {table}"
            ).fetchone(),
        )
        minimum_raw = row["minimum"]
        maximum_raw = row["maximum"]
        if minimum_raw is None and maximum_raw is None:
            return 0
        if minimum_raw is None or maximum_raw is None:
            raise IntegrityError("registry sequence range is malformed")
        minimum = require_stored_int(minimum_raw, f"{table} minimum sequence", -(2**63), 2**63 - 1)
        maximum = require_stored_int(maximum_raw, f"{table} maximum sequence", -(2**63), 2**63 - 1)
        if minimum < 1 or maximum < minimum:
            raise IntegrityError("registry sequence exceeds the supported snapshot range")
        return maximum

    def claim(
        self,
        *,
        worker_id: str,
        lease_seconds: int,
    ) -> ClaimedJob | None:
        """Claim one job and return its lease plus digest-verified immutable request."""

        require_identifier(worker_id, "worker_id")
        self._validate_lease_seconds(lease_seconds)
        with self._write() as connection:
            pending_retention = connection.execute(
                "SELECT 1 FROM sl_registry_retention_tombstones " "WHERE deleted_at IS NULL LIMIT 1"
            ).fetchone()
            if pending_retention is not None:
                raise RetentionPendingError(
                    "job claims pause while durable retention intent remains pending"
                )
            current = self._now()
            instant = _db_time(current)
            try:
                expires_at = current + timedelta(seconds=lease_seconds)
            except OverflowError:
                raise ValidationError("lease expiry exceeds the supported datetime range") from None
            expires = _db_time(expires_at, "lease_expires_at")
            self._recover_expired(connection, instant)
            row = connection.execute("""
                SELECT * FROM sl_registry_jobs
                WHERE state = 'queued'
                ORDER BY priority DESC, sequence ASC
                LIMIT 1
                """).fetchone()
            if row is None:
                return None
            self._require_monotonic(row, current)
            request = self._request(row)
            queued = self._job(row)
            if queued.state is not JobState.QUEUED:
                raise IntegrityError("queue selection returned a non-queued job")
            job_id = queued.job_id
            attempt = queued.attempt_count + 1
            token = secrets.token_urlsafe(32)
            updated = connection.execute(
                """
                UPDATE sl_registry_jobs
                SET state = 'running', attempt_count = ?, updated_at = ?, lease_owner = ?,
                    lease_token_digest = ?, lease_expires_at = ?, heartbeat_at = ?
                WHERE job_id = ? AND state = 'queued'
                """,
                (
                    attempt,
                    instant,
                    worker_id,
                    _digest(self._digest_secret, _LEASE_DOMAIN, token),
                    expires,
                    instant,
                    job_id,
                ),
            ).rowcount
            if updated != 1:
                raise IntegrityError("guarded queue claim did not update exactly one job")
            self._event(
                connection,
                job_id=job_id,
                kind=EventKind.CLAIMED,
                from_state=JobState.QUEUED,
                to_state=JobState.RUNNING,
                occurred_at=instant,
                attempt=attempt,
                actor=worker_id,
                details={"lease_expires_at": expires},
            )
            return ClaimedJob(
                Lease(job_id, worker_id, token, attempt, _parse_time(expires, "expires_at")),
                request,
            )

    def heartbeat(
        self,
        lease: Lease,
        *,
        lease_seconds: int,
    ) -> Lease:
        """Extend a live lease from ``now`` and surface durable cancellation intent."""

        if type(lease) is not Lease:
            raise ValidationError("lease must be a Lease")
        self._validate_lease_seconds(lease_seconds)
        with self._write() as connection:
            current = self._now()
            instant = _db_time(current)
            try:
                expires_at = current + timedelta(seconds=lease_seconds)
            except OverflowError:
                raise ValidationError("lease expiry exceeds the supported datetime range") from None
            expires = _db_time(expires_at, "lease_expires_at")
            self._recover_expired(connection, instant)
            row = self._owned_running(connection, lease, instant)
            self._require_monotonic(row, current)
            connection.execute(
                """
                UPDATE sl_registry_jobs
                SET updated_at = ?, heartbeat_at = ?, lease_expires_at = ?
                WHERE job_id = ? AND state = 'running'
                """,
                (instant, instant, expires, lease.job_id),
            )
            return Lease(
                lease.job_id,
                lease.worker_id,
                lease.token,
                require_stored_int(row["attempt_count"], "attempt_count", 1, 100),
                _parse_time(expires, "expires_at"),
                row["cancel_requested_at"] is not None,
            )

    def recover_expired(self) -> int:
        """Requeue or terminally resolve at most one configured batch of expired leases."""

        with self._write() as connection:
            instant = _db_time(self._now())
            return self._recover_expired(connection, instant)

    def request_cancel(
        self,
        job_id: str,
        *,
        reason: CancellationReasonCode,
    ) -> JobSnapshot:
        """Persist cancellation intent; queued work is cancelled immediately."""

        require_identifier(job_id, "job_id")
        if type(reason) is not CancellationReasonCode:
            raise ValidationError("reason must be a CancellationReasonCode")
        with self._write() as connection:
            current = self._now()
            instant = _db_time(current)
            self._recover_expired(connection, instant, prioritized_job_id=job_id)
            row = self._require_job(connection, job_id)
            self._require_monotonic(row, current)
            snapshot = self._job(row)
            state = snapshot.state
            if state.terminal or row["cancel_requested_at"] is not None:
                return snapshot
            connection.execute(
                """
                UPDATE sl_registry_jobs
                SET cancel_requested_at = ?, cancel_reason = ?, updated_at = ?
                WHERE job_id = ?
                """,
                (instant, reason.value, instant, job_id),
            )
            self._event(
                connection,
                job_id=job_id,
                kind=EventKind.CANCEL_REQUESTED,
                from_state=state,
                to_state=state,
                occurred_at=instant,
                attempt=require_stored_int(row["attempt_count"], "attempt_count", 0, 100),
                actor="requester",
                details={"reason_code": reason.value},
            )
            if state is JobState.QUEUED:
                connection.execute(
                    """
                    UPDATE sl_registry_jobs
                    SET state = 'cancelled', terminal_at = ?, updated_at = ?
                    WHERE job_id = ? AND state = 'queued'
                    """,
                    (instant, instant, job_id),
                )
                self._event(
                    connection,
                    job_id=job_id,
                    kind=EventKind.CANCELLED,
                    from_state=JobState.QUEUED,
                    to_state=JobState.CANCELLED,
                    occurred_at=instant,
                    attempt=require_stored_int(row["attempt_count"], "attempt_count", 0, 100),
                    actor="registry",
                    details={},
                )
            return self._job(self._require_job(connection, job_id))

    def register_artifact(
        self,
        published: PublishedArtifact,
        *,
        artifact_class: ArtifactClass,
        media_type: str,
        pinned: bool = False,
    ) -> ArtifactMetadata:
        """Persist verified CAS metadata without accepting a caller-selected path.

        CAS publication must complete before this call. Re-registering the same digest is
        idempotent only when every durable semantic field is identical; conflicting metadata
        fails closed and the existing record remains unchanged.
        """

        if type(published) is not PublishedArtifact:
            raise ValidationError("published must be a PublishedArtifact")
        # Validate every caller-controlled metadata field before spending O(bytes) on CAS proof.
        ArtifactMetadata(
            digest=published.digest,
            artifact_class=artifact_class,
            byte_size=published.byte_size,
            media_type=media_type,
            storage_relpath=published.storage_key,
            created_at=_CONTRACT_VALIDATION_INSTANT,
            pinned=pinned,
        )
        preflight = self._preflight_published_artifact(published)
        with self._write() as connection:
            instant = self._now()
            metadata = ArtifactMetadata(
                digest=published.digest,
                artifact_class=artifact_class,
                byte_size=published.byte_size,
                media_type=media_type,
                storage_relpath=published.storage_key,
                created_at=instant,
                pinned=pinned,
            )
            self._require_preflight_authority(connection, preflight)
            prior = connection.execute(
                "SELECT * FROM sl_registry_artifacts WHERE digest = ?",
                (metadata.digest,),
            ).fetchone()
            if prior is not None:
                observed = self._artifact(prior)
                if (
                    observed.digest,
                    observed.artifact_class,
                    observed.byte_size,
                    observed.media_type,
                    observed.storage_relpath,
                    observed.pinned,
                ) != (
                    metadata.digest,
                    metadata.artifact_class,
                    metadata.byte_size,
                    metadata.media_type,
                    metadata.storage_relpath,
                    metadata.pinned,
                ):
                    raise ConflictError("artifact digest is already registered with other metadata")
                return observed
            connection.execute(
                """
                INSERT INTO sl_registry_artifacts(
                    digest, artifact_class, byte_size, media_type, storage_relpath,
                    created_at, pinned
                ) VALUES(?,?,?,?,?,?,?)
                """,
                (
                    metadata.digest,
                    metadata.artifact_class.value,
                    metadata.byte_size,
                    metadata.media_type,
                    metadata.storage_relpath,
                    _db_time(metadata.created_at, "created_at"),
                    int(metadata.pinned),
                ),
            )
            return metadata

    def acknowledge_cancel(
        self,
        lease: Lease,
        *,
        run: TerminalRunRequest,
        artifact_links: tuple[ArtifactLink, ...] = (),
    ) -> JobSnapshot:
        """Atomically preserve the cancelled attempt and acknowledge cancellation."""

        self._validate_terminal_request(lease, run)
        preflight = self._preflight_terminal_artifacts(artifact_links)
        with self._write() as connection:
            instant = _db_time(self._now())
            self._recover_expired(connection, instant)
            row = self._owned_running(connection, lease, instant)
            if row["cancel_requested_at"] is None:
                raise InvalidTransitionError("cancellation has not been requested")
            self._insert_terminal_run(
                connection,
                row,
                request=run,
                status=RunStatus.CANCELLED,
                artifact_links=artifact_links,
                artifact_preflight=preflight,
                instant=instant,
            )
            self._set_terminal(
                connection,
                row,
                to_state=JobState.CANCELLED,
                event_kind=EventKind.CANCELLED,
                actor=lease.worker_id,
                instant=instant,
            )
            return self._job(self._require_job(connection, lease.job_id))

    def complete(
        self,
        lease: Lease,
        *,
        run: TerminalRunRequest,
        artifact_links: tuple[ArtifactLink, ...] = (),
    ) -> JobSnapshot:
        """Atomically preserve one succeeded run and terminate its live job."""

        self._validate_terminal_request(lease, run)
        preflight = self._preflight_terminal_artifacts(artifact_links)
        with self._write() as connection:
            instant = _db_time(self._now())
            self._recover_expired(connection, instant)
            row = self._owned_running(connection, lease, instant)
            if row["cancel_requested_at"] is not None:
                raise InvalidTransitionError("cancel-requested work may not complete successfully")
            self._insert_terminal_run(
                connection,
                row,
                request=run,
                status=RunStatus.SUCCEEDED,
                artifact_links=artifact_links,
                artifact_preflight=preflight,
                instant=instant,
            )
            self._set_terminal(
                connection,
                row,
                to_state=JobState.SUCCEEDED,
                event_kind=EventKind.SUCCEEDED,
                actor=lease.worker_id,
                instant=instant,
                result_run_id=run.run_id,
            )
            return self._job(self._require_job(connection, lease.job_id))

    def fail(
        self,
        lease: Lease,
        *,
        failure_code: FailureReasonCode,
        failure_summary: str,
        retryable: bool,
        run: TerminalRunRequest,
        artifact_links: tuple[ArtifactLink, ...] = (),
    ) -> JobSnapshot:
        """Record a bounded failure, requeueing only while attempts remain."""

        if type(failure_code) is not FailureReasonCode:
            raise ValidationError("failure_code must be a FailureReasonCode")
        failure_summary = require_failure_summary(failure_summary)
        if type(retryable) is not bool:
            raise ValidationError("retryable must be a boolean")
        self._validate_terminal_request(lease, run)
        preflight = self._preflight_terminal_artifacts(artifact_links)
        with self._write() as connection:
            instant = _db_time(self._now())
            self._recover_expired(connection, instant)
            row = self._owned_running(connection, lease, instant)
            terminal_status = (
                RunStatus.CANCELLED if row["cancel_requested_at"] is not None else RunStatus.FAILED
            )
            self._insert_terminal_run(
                connection,
                row,
                request=run,
                status=terminal_status,
                artifact_links=artifact_links,
                artifact_preflight=preflight,
                instant=instant,
            )
            if row["cancel_requested_at"] is not None:
                self._set_terminal(
                    connection,
                    row,
                    to_state=JobState.CANCELLED,
                    event_kind=EventKind.CANCELLED,
                    actor=lease.worker_id,
                    instant=instant,
                )
            elif retryable and require_stored_int(
                row["attempt_count"], "attempt_count", 1, 100
            ) < require_stored_int(row["max_attempts"], "max_attempts", 1, 100):
                connection.execute(
                    """
                    UPDATE sl_registry_jobs
                    SET state = 'queued', updated_at = ?, lease_owner = NULL,
                        lease_token_digest = NULL, lease_expires_at = NULL, heartbeat_at = NULL
                    WHERE job_id = ? AND state = 'running'
                    """,
                    (instant, lease.job_id),
                )
                self._event(
                    connection,
                    job_id=lease.job_id,
                    kind=EventKind.RETRIED,
                    from_state=JobState.RUNNING,
                    to_state=JobState.QUEUED,
                    occurred_at=instant,
                    attempt=require_stored_int(row["attempt_count"], "attempt_count", 1, 100),
                    actor=lease.worker_id,
                    details={"failure_code": failure_code.value},
                )
            else:
                self._set_terminal(
                    connection,
                    row,
                    to_state=JobState.FAILED,
                    event_kind=EventKind.FAILED,
                    actor=lease.worker_id,
                    instant=instant,
                    failure_code=failure_code,
                    failure_summary=failure_summary,
                )
            return self._job(self._require_job(connection, lease.job_id))

    def _insert_terminal_run(
        self,
        connection: sqlite3.Connection,
        job: sqlite3.Row,
        *,
        request: TerminalRunRequest,
        status: RunStatus,
        artifact_links: tuple[ArtifactLink, ...],
        artifact_preflight: _ArtifactPreflight | None,
        instant: str,
    ) -> RunSnapshot:
        """Insert one terminal attempt and its verified metadata links atomically."""

        if type(request) is not TerminalRunRequest:
            raise ValidationError("run must be a TerminalRunRequest")
        if type(status) is not RunStatus:
            raise ValidationError("status must be a RunStatus")
        if type(artifact_links) is not tuple:
            raise ValidationError("artifact_links must be an immutable tuple")
        if len(artifact_links) > self.limits.max_artifacts_per_run:
            raise CapacityError("run artifact-link capacity has been reached")
        if any(type(link) is not ArtifactLink for link in artifact_links):
            raise ValidationError("artifact_links entries must be ArtifactLink values")
        if tuple(sorted(set(artifact_links))) != artifact_links:
            raise ValidationError("artifact_links must be unique and canonically sorted")
        self._validate_artifact_preflight(connection, artifact_links, artifact_preflight)
        ended_at = _parse_time(instant, "ended_at")
        created_at = _parse_time(job["created_at"], "job.created_at")
        claimed_at = self._current_attempt_claimed_at(connection, job)
        if claimed_at < created_at or claimed_at > ended_at:
            raise IntegrityError("current attempt claim chronology is inconsistent")
        started_at = request.started_at or claimed_at
        if started_at < claimed_at or started_at > ended_at:
            raise ValidationError(
                "run chronology must satisfy claimed_at <= started_at <= ended_at"
            )
        prior = connection.execute(
            """
            SELECT 1 FROM sl_registry_runs
            WHERE run_id = ? OR (job_id = ? AND attempt = ?)
            LIMIT 1
            """,
            (
                request.run_id,
                _stored_text(job["job_id"], "job_id", 128),
                require_stored_int(job["attempt_count"], "attempt_count", 1, 100),
            ),
        ).fetchone()
        if prior is not None:
            raise ConflictError("a terminal run already exists for this identifier or attempt")
        legacy_table = connection.execute(
            "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = 'runs'"
        ).fetchone()
        if legacy_table is not None:
            legacy_collision = connection.execute(
                "SELECT 1 FROM runs WHERE run_id = ? LIMIT 1",
                (request.run_id,),
            ).fetchone()
            if legacy_collision is not None:
                raise ConflictError("run_id already exists in the legacy experiment registry")
        for expected_artifact in (
            () if artifact_preflight is None else artifact_preflight.artifacts
        ):
            artifact = connection.execute(
                "SELECT * FROM sl_registry_artifacts WHERE digest = ?",
                (expected_artifact.digest,),
            ).fetchone()
            if artifact is None:
                raise IntegrityError("preflighted artifact metadata disappeared")
            stored_artifact = self._artifact(artifact)
            if stored_artifact != expected_artifact:
                raise IntegrityError("registered artifact metadata changed after CAS preflight")
            tombstone = connection.execute(
                "SELECT 1 FROM sl_registry_retention_tombstones WHERE artifact_digest = ?",
                (expected_artifact.digest,),
            ).fetchone()
            if tombstone is not None:
                raise ConflictError("run artifact is already governed by retention intent")
        connection.execute(
            """
            INSERT INTO sl_registry_runs(
                run_id, job_id, attempt, status, evidence_class, schema_version,
                created_at, started_at, ended_at, source_commit, data_identity,
                limitation_summary
            ) VALUES(?,?,?,?,?,1,?,?,?,?,?,?)
            """,
            (
                request.run_id,
                _stored_text(job["job_id"], "job_id", 128),
                require_stored_int(job["attempt_count"], "attempt_count", 1, 100),
                status.value,
                request.evidence_class.value,
                _db_time(created_at, "created_at"),
                _db_time(started_at, "started_at"),
                instant,
                request.source_commit,
                request.data_identity,
                request.limitation_summary,
            ),
        )
        for link in artifact_links:
            connection.execute(
                """
                INSERT INTO sl_registry_run_artifacts(
                    run_id, role, artifact_digest, linked_at
                ) VALUES(?,?,?,?)
                """,
                (request.run_id, link.role, link.artifact_digest, instant),
            )
        row = connection.execute(
            "SELECT * FROM sl_registry_runs WHERE run_id = ?",
            (request.run_id,),
        ).fetchone()
        if row is None:
            raise IntegrityError("terminal run insert did not produce a durable row")
        return self._run(row)

    @staticmethod
    def _current_attempt_claimed_at(
        connection: sqlite3.Connection,
        job: sqlite3.Row,
    ) -> datetime:
        """Return the sole canonical claim instant for the job's current attempt."""

        job_id = _stored_text(job["job_id"], "job_id", 128)
        attempt = require_stored_int(job["attempt_count"], "attempt_count", 1, 100)
        rows = connection.execute(
            """
            SELECT occurred_at, actor, from_state, to_state
            FROM sl_registry_events
            WHERE job_id = ? AND attempt = ? AND kind = 'claimed'
            ORDER BY sequence
            LIMIT 2
            """,
            (job_id, attempt),
        ).fetchall()
        if len(rows) != 1:
            raise IntegrityError("current attempt must have exactly one claim event")
        row = rows[0]
        if (
            row["actor"] != job["lease_owner"]
            or row["from_state"] != JobState.QUEUED.value
            or row["to_state"] != JobState.RUNNING.value
        ):
            raise IntegrityError("current attempt claim event is inconsistent with its lease")
        return _parse_time(row["occurred_at"], "claimed_at")

    def _recover_expired(
        self,
        connection: sqlite3.Connection,
        instant: str,
        *,
        prioritized_job_id: str | None = None,
    ) -> int:
        """Recover at most one configured batch, optionally selecting one target first."""

        rows: list[sqlite3.Row] = []
        if prioritized_job_id is not None:
            target = connection.execute(
                """
                SELECT * FROM sl_registry_jobs
                WHERE job_id = ? AND state = 'running' AND lease_expires_at <= ?
                """,
                (prioritized_job_id, instant),
            ).fetchone()
            if target is not None:
                rows.append(cast(sqlite3.Row, target))
        remaining = self.limits.recovery_batch_size - len(rows)
        if remaining > 0:
            if prioritized_job_id is None:
                rows.extend(
                    connection.execute(
                        """
                        SELECT * FROM sl_registry_jobs
                        WHERE state = 'running' AND lease_expires_at <= ?
                        ORDER BY sequence
                        LIMIT ?
                        """,
                        (instant, remaining),
                    ).fetchall()
                )
            else:
                rows.extend(
                    connection.execute(
                        """
                        SELECT * FROM sl_registry_jobs
                        WHERE state = 'running' AND lease_expires_at <= ? AND job_id <> ?
                        ORDER BY sequence
                        LIMIT ?
                        """,
                        (instant, prioritized_job_id, remaining),
                    ).fetchall()
                )
        for row in rows:
            snapshot = self._job(row)
            if snapshot.state is not JobState.RUNNING:
                raise IntegrityError("expiry selection returned a non-running job")
            job_id = snapshot.job_id
            attempt = snapshot.attempt_count
            if row["cancel_requested_at"] is not None:
                state = JobState.CANCELLED
                failure_code = None
                failure_summary = None
            elif attempt >= snapshot.max_attempts:
                state = JobState.FAILED
                failure_code = FailureReasonCode.LEASE_ATTEMPTS_EXHAUSTED
                failure_summary = "lease expired after the final permitted attempt"
            else:
                state = JobState.QUEUED
                failure_code = None
                failure_summary = None
            terminal = instant if state.terminal else None
            connection.execute(
                """
                UPDATE sl_registry_jobs
                SET state = ?, updated_at = ?, lease_owner = NULL, lease_token_digest = NULL,
                    lease_expires_at = NULL, heartbeat_at = NULL, terminal_at = ?,
                    failure_code = ?, failure_summary = ?
                WHERE job_id = ? AND state = 'running' AND lease_expires_at <= ?
                """,
                (
                    state.value,
                    instant,
                    terminal,
                    None if failure_code is None else failure_code.value,
                    failure_summary,
                    job_id,
                    instant,
                ),
            )
            self._event(
                connection,
                job_id=job_id,
                kind=EventKind.LEASE_EXPIRED,
                from_state=JobState.RUNNING,
                to_state=state,
                occurred_at=instant,
                attempt=attempt,
                actor="registry",
                details={},
            )
        return len(rows)

    def _owned_running(
        self,
        connection: sqlite3.Connection,
        lease: Lease,
        instant: str,
    ) -> sqlite3.Row:
        if type(lease) is not Lease:
            raise ValidationError("lease must be a Lease")
        row = self._require_job(connection, lease.job_id)
        current = _parse_time(instant, "now")
        self._require_monotonic(row, current)
        snapshot = self._job(row)
        if (
            snapshot.state is not JobState.RUNNING
            or row["lease_owner"] != lease.worker_id
            or not secrets.compare_digest(
                _stored_text(row["lease_token_digest"], "lease_token_digest", 64),
                _digest(self._digest_secret, _LEASE_DOMAIN, lease.token),
            )
            or snapshot.lease_expires_at is None
            or snapshot.lease_expires_at <= current
            or snapshot.attempt_count != lease.attempt
        ):
            raise LeaseLostError("lease is absent, expired, or no longer owned by this claimant")
        return row

    @staticmethod
    def _require_monotonic(row: sqlite3.Row, current: datetime) -> None:
        if current < _parse_time(row["updated_at"], "updated_at"):
            raise IntegrityError("registry time authority moved behind durable job state")

    @staticmethod
    def _validate_terminal_request(lease: Lease, run: TerminalRunRequest) -> None:
        if type(lease) is not Lease:
            raise ValidationError("lease must be a Lease")
        if type(run) is not TerminalRunRequest:
            raise ValidationError("run must be a TerminalRunRequest")

    def _artifact_preflight_authority(self) -> _ArtifactPreflight:
        """Capture and verify the immutable registry/CAS binding before full hashing."""

        verifier = self._artifact_verifier
        if verifier is None:
            raise IntegrityError("artifact operations require a bound CAS verifier")
        store_id = self._artifact_verifier_store_id()
        connection = self._connect(readonly=True)
        try:
            self._require_artifact_store_binding(connection, store_id)
        finally:
            connection.close()
        return _ArtifactPreflight(verifier=verifier, store_id=store_id)

    def _preflight_published_artifact(
        self,
        published: PublishedArtifact,
    ) -> _ArtifactPreflight:
        """Hash one CAS object without holding SQLite's process-wide writer lock."""

        preflight = self._artifact_preflight_authority()
        try:
            preflight.verifier.verify(published)
        except Exception:
            raise IntegrityError("CAS bytes failed verification at the registry boundary") from None
        self._require_preflight_process_authority(preflight)
        return preflight

    def _preflight_terminal_artifacts(
        self,
        artifact_links: tuple[ArtifactLink, ...],
    ) -> _ArtifactPreflight | None:
        """Verify each distinct linked object fully before entering a write transaction."""

        if type(artifact_links) is not tuple:
            raise ValidationError("artifact_links must be an immutable tuple")
        if len(artifact_links) > self.limits.max_artifacts_per_run:
            raise CapacityError("run artifact-link capacity has been reached")
        if any(type(link) is not ArtifactLink for link in artifact_links):
            raise ValidationError("artifact_links entries must be ArtifactLink values")
        if tuple(sorted(set(artifact_links))) != artifact_links:
            raise ValidationError("artifact_links must be unique and canonically sorted")
        if not artifact_links:
            return None

        authority = self._artifact_preflight_authority()
        digests = tuple(sorted({link.artifact_digest for link in artifact_links}))
        metadata: list[ArtifactMetadata] = []
        connection = self._connect(readonly=True)
        try:
            self._require_artifact_store_binding(connection, authority.store_id)
            for digest in digests:
                row = connection.execute(
                    "SELECT * FROM sl_registry_artifacts WHERE digest = ?",
                    (digest,),
                ).fetchone()
                if row is None:
                    raise NotFoundError("run artifact metadata is not registered")
                artifact = self._artifact(row)
                tombstone = connection.execute(
                    "SELECT 1 FROM sl_registry_retention_tombstones WHERE artifact_digest = ?",
                    (digest,),
                ).fetchone()
                if tombstone is not None:
                    raise ConflictError("run artifact is already governed by retention intent")
                metadata.append(artifact)
        except sqlite3.Error as exc:
            self._raise_sqlite(exc)
        finally:
            connection.close()

        verified = _ArtifactPreflight(
            verifier=authority.verifier,
            store_id=authority.store_id,
            artifacts=tuple(metadata),
        )
        for artifact in verified.artifacts:
            try:
                published = PublishedArtifact(
                    artifact.digest,
                    artifact.byte_size,
                    artifact.storage_relpath,
                )
            except ArtifactStoreError as exc:
                raise IntegrityError(
                    "stored artifact metadata is not a canonical CAS identity"
                ) from exc
            try:
                verified.verifier.verify(published)
            except Exception:
                raise IntegrityError(
                    "CAS bytes failed verification at the registry boundary"
                ) from None
        self._require_preflight_process_authority(verified)
        return verified

    def _require_preflight_process_authority(self, preflight: _ArtifactPreflight) -> None:
        """Reject verifier or store-identity replacement around an out-of-lock hash."""

        if self._artifact_verifier is not preflight.verifier:
            raise IntegrityError("artifact verifier authority changed during CAS preflight")
        observed_store_id = self._artifact_verifier_store_id()
        if not hmac.compare_digest(observed_store_id, preflight.store_id):
            raise IntegrityError("artifact verifier CAS identity changed during preflight")

    def _require_preflight_authority(
        self,
        connection: sqlite3.Connection,
        preflight: _ArtifactPreflight,
    ) -> None:
        """Recheck only bounded immutable authority state under the writer lock."""

        self._require_preflight_process_authority(preflight)
        self._require_artifact_store_binding(connection, preflight.store_id)

    def _validate_artifact_preflight(
        self,
        connection: sqlite3.Connection,
        artifact_links: tuple[ArtifactLink, ...],
        preflight: _ArtifactPreflight | None,
    ) -> None:
        """Bind a full preflight proof to the exact distinct link identities."""

        expected_digests = tuple(sorted({link.artifact_digest for link in artifact_links}))
        if not expected_digests:
            if preflight is not None:
                raise IntegrityError("empty artifact links received an unexpected CAS preflight")
            return
        if preflight is None:
            raise IntegrityError("artifact links are missing their CAS preflight")
        if tuple(artifact.digest for artifact in preflight.artifacts) != expected_digests:
            raise IntegrityError("artifact links differ from their CAS preflight identities")
        self._require_preflight_authority(connection, preflight)

    def _set_terminal(
        self,
        connection: sqlite3.Connection,
        row: sqlite3.Row,
        *,
        to_state: JobState,
        event_kind: EventKind,
        actor: str,
        instant: str,
        result_run_id: str | None = None,
        failure_code: FailureReasonCode | None = None,
        failure_summary: str | None = None,
    ) -> None:
        if to_state not in {JobState.SUCCEEDED, JobState.FAILED, JobState.CANCELLED}:
            raise IntegrityError("terminal helper received a non-terminal target")
        job_id = _stored_text(row["job_id"], "job_id", 128)
        updated = connection.execute(
            """
            UPDATE sl_registry_jobs
            SET state = ?, updated_at = ?, terminal_at = ?, result_run_id = ?,
                failure_code = ?, failure_summary = ?, lease_owner = NULL,
                lease_token_digest = NULL, lease_expires_at = NULL, heartbeat_at = NULL
            WHERE job_id = ? AND state = 'running'
            """,
            (
                to_state.value,
                instant,
                instant,
                result_run_id,
                None if failure_code is None else failure_code.value,
                failure_summary,
                job_id,
            ),
        ).rowcount
        if updated != 1:
            raise InvalidTransitionError("job is no longer running")
        self._event(
            connection,
            job_id=job_id,
            kind=event_kind,
            from_state=JobState.RUNNING,
            to_state=to_state,
            occurred_at=instant,
            attempt=require_stored_int(row["attempt_count"], "attempt_count", 1, 100),
            actor=actor,
            details={},
        )

    @staticmethod
    def _require_job(connection: sqlite3.Connection, job_id: str) -> sqlite3.Row:
        row = connection.execute(
            "SELECT * FROM sl_registry_jobs WHERE job_id = ?", (job_id,)
        ).fetchone()
        if row is None:
            raise NotFoundError(f"job {job_id!r} does not exist")
        return cast(sqlite3.Row, row)

    @staticmethod
    def _state(row: sqlite3.Row) -> JobState:
        value = row["state"]
        if type(value) is not str:
            raise IntegrityError("stored job state is unknown")
        try:
            return JobState(value)
        except ValueError as exc:
            raise IntegrityError("stored job state is unknown") from exc

    @classmethod
    def _job(cls, row: sqlite3.Row) -> JobSnapshot:
        try:
            return JobSnapshot(
                sequence=require_stored_int(row["sequence"], "job sequence", 1, 2**63 - 1),
                job_id=_stored_text(row["job_id"], "job_id", 128),
                kind=_stored_text(row["kind"], "job kind", 128),
                state=cls._state(row),
                priority=require_stored_int(row["priority"], "priority", -100, 100),
                attempt_count=require_stored_int(row["attempt_count"], "attempt_count", 0, 100),
                max_attempts=require_stored_int(row["max_attempts"], "max_attempts", 1, 100),
                created_at=_parse_time(row["created_at"], "created_at"),
                updated_at=_parse_time(row["updated_at"], "updated_at"),
                lease_owner=(
                    None
                    if row["lease_owner"] is None
                    else _stored_text(row["lease_owner"], "lease_owner", 128)
                ),
                lease_expires_at=_optional_time(row["lease_expires_at"], "lease_expires_at"),
                cancel_requested_at=_optional_time(
                    row["cancel_requested_at"], "cancel_requested_at"
                ),
                cancel_reason=(
                    None
                    if row["cancel_reason"] is None
                    else CancellationReasonCode(
                        _stored_text(row["cancel_reason"], "cancel_reason", 32)
                    )
                ),
                terminal_at=_optional_time(row["terminal_at"], "terminal_at"),
                result_run_id=(
                    None
                    if row["result_run_id"] is None
                    else _stored_text(row["result_run_id"], "result_run_id", 128)
                ),
                failure_code=(
                    None
                    if row["failure_code"] is None
                    else FailureReasonCode(_stored_text(row["failure_code"], "failure_code", 64))
                ),
            )
        except (TypeError, ValueError, ValidationError) as exc:
            raise IntegrityError("stored job state is malformed") from exc

    def _request(self, row: sqlite3.Row) -> SubmissionRequest:
        raw = row["request_json"]
        digest = row["request_digest"]
        if type(raw) is not str or type(digest) is not str:
            raise IntegrityError("stored canonical request fields are not text")
        try:
            encoded = raw.encode("utf-8")
        except (MemoryError, UnicodeEncodeError):
            raise IntegrityError("stored canonical request is not bounded UTF-8") from None
        if not 2 <= len(encoded) <= self.limits.max_request_bytes:
            raise IntegrityError("stored canonical request exceeds this registry's bound")
        expected = hashlib.sha256(encoded).hexdigest()
        if not hmac.compare_digest(digest, expected):
            raise IntegrityError("stored canonical request digest does not match its bytes")
        try:
            decoded = decode_bounded_json(
                raw,
                maximum_bytes=self.limits.max_request_bytes,
                require_canonical=True,
            )
            if type(decoded) is not dict or set(decoded) != {
                "kind",
                "max_attempts",
                "payload",
                "priority",
                "schema_version",
            }:
                raise ValueError("unexpected request shape")
            payload = cast(dict[str, object], decoded)
            request = SubmissionRequest(
                kind=cast(str, payload["kind"]),
                payload=cast(Mapping[str, JsonValue], payload["payload"]),
                priority=cast(int, payload["priority"]),
                max_attempts=cast(int, payload["max_attempts"]),
                schema_version=cast(int, payload["schema_version"]),
            )
        except (TypeError, ValueError, ValidationError):
            raise IntegrityError("stored canonical request is malformed") from None
        request_schema_version = require_stored_int(
            row["request_schema_version"], "request_schema_version", 1, 1
        )
        priority = require_stored_int(row["priority"], "request priority", -100, 100)
        max_attempts = require_stored_int(row["max_attempts"], "request max_attempts", 1, 100)
        if (
            request.kind != row["kind"]
            or request.schema_version != request_schema_version
            or request.priority != priority
            or request.max_attempts != max_attempts
        ):
            raise IntegrityError("stored request JSON disagrees with indexed job semantics")
        if request.max_attempts > self.limits.max_attempts:
            raise IntegrityError("stored request exceeds this registry's attempt bound")
        return request

    @staticmethod
    def _run(row: sqlite3.Row) -> RunSnapshot:
        try:
            return RunSnapshot(
                sequence=require_stored_int(row["sequence"], "run sequence", 1, 2**63 - 1),
                run_id=_stored_text(row["run_id"], "run_id", 128),
                job_id=_stored_text(row["job_id"], "job_id", 128),
                attempt=require_stored_int(row["attempt"], "run attempt", 1, 100),
                status=RunStatus(_stored_text(row["status"], "run status", 16)),
                evidence_class=EvidenceClass(
                    _stored_text(row["evidence_class"], "evidence_class", 32)
                ),
                schema_version=require_stored_int(
                    row["schema_version"], "run schema_version", 1, 1
                ),
                created_at=_parse_time(row["created_at"], "created_at"),
                started_at=_optional_time(row["started_at"], "started_at"),
                ended_at=_parse_time(row["ended_at"], "ended_at"),
                source_commit=(
                    None
                    if row["source_commit"] is None
                    else _stored_text(row["source_commit"], "source_commit", 64)
                ),
                data_identity=(
                    None
                    if row["data_identity"] is None
                    else _stored_text(row["data_identity"], "data_identity", 256)
                ),
                limitation_summary=(
                    None
                    if row["limitation_summary"] is None
                    else _stored_text(row["limitation_summary"], "limitation_summary", 4_096)
                ),
            )
        except (TypeError, ValueError, ValidationError) as exc:
            raise IntegrityError("stored terminal run is malformed") from exc

    @staticmethod
    def _artifact(row: sqlite3.Row) -> ArtifactMetadata:
        try:
            pinned = require_stored_int(row["pinned"], "pinned", 0, 1)
            return ArtifactMetadata(
                digest=_stored_text(row["digest"], "artifact digest", 64),
                artifact_class=ArtifactClass(
                    _stored_text(row["artifact_class"], "artifact_class", 16)
                ),
                byte_size=require_stored_int(row["byte_size"], "artifact byte_size", 0, 2**63 - 1),
                media_type=_stored_text(row["media_type"], "artifact media_type", 128),
                storage_relpath=_stored_text(
                    row["storage_relpath"], "artifact storage_relpath", 1_024
                ),
                created_at=_parse_time(row["created_at"], "created_at"),
                pinned=bool(pinned),
            )
        except (TypeError, ValueError, ValidationError) as exc:
            raise IntegrityError("stored artifact metadata is malformed") from exc

    def _event(
        self,
        connection: sqlite3.Connection,
        *,
        job_id: str,
        kind: EventKind,
        from_state: JobState | None,
        to_state: JobState,
        occurred_at: str,
        attempt: int,
        actor: str,
        details: object,
    ) -> None:
        count = require_stored_int(
            connection.execute(
                "SELECT count(*) FROM sl_registry_events WHERE job_id = ?",
                (job_id,),
            ).fetchone()[0],
            "job event count",
            0,
            2**63 - 1,
        )
        if count >= self.limits.max_events_per_job:
            raise CapacityError("job lifecycle event capacity has been reached")
        details_json = canonical_json(details)
        if len(details_json.encode("utf-8")) > 8_192:
            raise ValidationError("event details exceed 8,192 bytes")
        connection.execute(
            """
            INSERT INTO sl_registry_events(
                job_id, kind, from_state, to_state, occurred_at, attempt, actor, details_json
            ) VALUES(?,?,?,?,?,?,?,?)
            """,
            (
                job_id,
                kind.value,
                None if from_state is None else from_state.value,
                to_state.value,
                occurred_at,
                attempt,
                actor,
                details_json,
            ),
        )

    @staticmethod
    def _lifecycle_event(row: sqlite3.Row) -> LifecycleEvent:
        try:
            details = decode_bounded_json(
                row["details_json"], maximum_bytes=8_192, require_canonical=True
            )
            frozen_details = SubmissionRequest(
                "event", cast(Mapping[str, JsonValue], details)
            ).payload
            from_state = (
                None
                if row["from_state"] is None
                else JobState(_stored_text(row["from_state"], "event from_state", 16))
            )
            return LifecycleEvent(
                sequence=require_stored_int(row["sequence"], "event sequence", 1, 2**63 - 1),
                job_id=_stored_text(row["job_id"], "event job_id", 128),
                kind=EventKind(_stored_text(row["kind"], "event kind", 32)),
                from_state=from_state,
                to_state=JobState(_stored_text(row["to_state"], "event to_state", 16)),
                occurred_at=_parse_time(row["occurred_at"], "occurred_at"),
                attempt=require_stored_int(row["attempt"], "event attempt", 0, 100),
                actor=_stored_text(row["actor"], "event actor", 128),
                details=frozen_details,
            )
        except (TypeError, ValueError, ValidationError) as exc:
            raise IntegrityError("stored lifecycle event is malformed") from exc

    def _validate_page_size(self, page_size: int) -> None:
        if type(page_size) is not int or not 1 <= page_size <= self.limits.max_page_size:
            raise ValidationError(f"page_size must be in [1, {self.limits.max_page_size}]")

    def _validate_lease_seconds(self, lease_seconds: int) -> None:
        if type(lease_seconds) is not int or not (
            self.limits.min_lease_seconds <= lease_seconds <= self.limits.max_lease_seconds
        ):
            raise ValidationError(
                "lease_seconds must be within the configured inclusive lease bounds"
            )
