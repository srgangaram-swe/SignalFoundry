"""Checksummed, forward-only SQLite schema migrations for the run registry.

The unprefixed ``runs`` table belongs to the legacy experiment tracker and is intentionally absent
from every statement in this module.  Registry objects use the ``sl_registry_`` namespace so an
existing tracker database can be upgraded additively without changing legacy bytes or semantics.
"""

from __future__ import annotations

import ctypes
import errno
import hashlib
import hmac
import os
import platform
import secrets
import sqlite3
import stat
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from time import monotonic
from typing import cast

from quant_platform.tracking.contracts import (
    ArtifactClass,
    ArtifactMetadata,
    BusyError,
    CancellationReasonCode,
    EventKind,
    EvidenceClass,
    FailureReasonCode,
    IntegrityError,
    JobState,
    JsonValue,
    MigrationDriftError,
    MigrationError,
    RegistryAuthorityMismatchError,
    RegistryError,
    RegistryReadiness,
    RunStatus,
    SubmissionRequest,
    UnsupportedSchemaError,
    ValidationError,
    VerificationTimeoutError,
    canonical_json,
    decode_bounded_json,
    parse_stored_utc,
    require_failure_summary,
    require_identifier,
    require_stored_int,
    require_utf8_size,
)
from quant_platform.tracking.retention import (
    DecodedRetentionPlan,
    RetentionCandidate,
    RetentionIntegrityError,
    validate_decoded_retention_plan,
)

MIGRATION_TABLE = "sl_registry_schema_migrations"
_VERIFICATION_BATCH_SIZE = 256
_VERIFICATION_PROGRESS_INSTRUCTIONS = 1_000
_MAX_LIFECYCLE_EVENTS_PER_JOB = 202
_MAX_LIFECYCLE_RUNS_PER_JOB = 100

RetentionPlanAuthenticator = Callable[[bytes, str, str, str], bool]

_MIGRATION_1 = """
CREATE TRIGGER sl_registry_schema_migrations_no_update
BEFORE UPDATE ON sl_registry_schema_migrations
BEGIN
    SELECT RAISE(ABORT, 'sl_registry_schema_migrations is immutable');
END;

CREATE TRIGGER sl_registry_schema_migrations_no_delete
BEFORE DELETE ON sl_registry_schema_migrations
BEGIN
    SELECT RAISE(ABORT, 'sl_registry_schema_migrations is append-only');
END;

CREATE TABLE sl_registry_metadata (
    singleton INTEGER PRIMARY KEY CHECK(singleton = 1),
    hmac_version INTEGER NOT NULL CHECK(hmac_version = 1),
    key_verifier TEXT NOT NULL CHECK(
        length(key_verifier) = 64 AND key_verifier NOT GLOB '*[^0-9a-f]*'
    ),
    registry_id TEXT NOT NULL UNIQUE CHECK(
        length(registry_id) = 32 AND registry_id NOT GLOB '*[^0-9a-f]*'
    )
) WITHOUT ROWID;

CREATE TRIGGER sl_registry_metadata_no_update
BEFORE UPDATE ON sl_registry_metadata
BEGIN
    SELECT RAISE(ABORT, 'sl_registry_metadata is immutable');
END;

CREATE TRIGGER sl_registry_metadata_no_delete
BEFORE DELETE ON sl_registry_metadata
BEGIN
    SELECT RAISE(ABORT, 'sl_registry_metadata is immutable');
END;

CREATE TABLE sl_registry_cas_binding (
    singleton INTEGER PRIMARY KEY CHECK(singleton = 1),
    store_id TEXT NOT NULL UNIQUE CHECK(
        length(store_id) = 64 AND store_id NOT GLOB '*[^0-9a-f]*'
    )
) WITHOUT ROWID;

CREATE TRIGGER sl_registry_cas_binding_no_update
BEFORE UPDATE ON sl_registry_cas_binding
BEGIN
    SELECT RAISE(ABORT, 'sl_registry_cas_binding is immutable');
END;

CREATE TRIGGER sl_registry_cas_binding_no_delete
BEFORE DELETE ON sl_registry_cas_binding
BEGIN
    SELECT RAISE(ABORT, 'sl_registry_cas_binding is immutable');
END;

CREATE TABLE sl_registry_jobs (
    sequence INTEGER PRIMARY KEY AUTOINCREMENT,
    job_id TEXT NOT NULL UNIQUE CHECK(
        length(job_id) = 32 AND job_id NOT GLOB '*[^0-9a-f]*'
    ),
    kind TEXT NOT NULL CHECK(length(kind) BETWEEN 1 AND 128),
    request_schema_version INTEGER NOT NULL CHECK(request_schema_version = 1),
    request_json TEXT NOT NULL CHECK(length(request_json) BETWEEN 2 AND 262144),
    request_digest TEXT NOT NULL CHECK(
        length(request_digest) = 64 AND request_digest NOT GLOB '*[^0-9a-f]*'
    ),
    idempotency_digest TEXT NOT NULL UNIQUE CHECK(
        length(idempotency_digest) = 64 AND idempotency_digest NOT GLOB '*[^0-9a-f]*'
    ),
    state TEXT NOT NULL CHECK(state IN ('queued', 'running', 'succeeded', 'failed', 'cancelled')),
    priority INTEGER NOT NULL CHECK(priority BETWEEN -100 AND 100),
    attempt_count INTEGER NOT NULL DEFAULT 0 CHECK(attempt_count >= 0),
    max_attempts INTEGER NOT NULL CHECK(max_attempts BETWEEN 1 AND 100),
    created_at TEXT NOT NULL CHECK(length(created_at) = 27 AND substr(created_at, -1) = 'Z'),
    updated_at TEXT NOT NULL CHECK(length(updated_at) = 27 AND substr(updated_at, -1) = 'Z'),
    lease_owner TEXT CHECK(lease_owner IS NULL OR length(lease_owner) BETWEEN 1 AND 128),
    lease_token_digest TEXT CHECK(
        lease_token_digest IS NULL OR (
            length(lease_token_digest) = 64 AND lease_token_digest NOT GLOB '*[^0-9a-f]*'
        )
    ),
    lease_expires_at TEXT CHECK(
        lease_expires_at IS NULL OR (length(lease_expires_at) = 27 AND substr(lease_expires_at, -1) = 'Z')
    ),
    heartbeat_at TEXT CHECK(
        heartbeat_at IS NULL OR (length(heartbeat_at) = 27 AND substr(heartbeat_at, -1) = 'Z')
    ),
    cancel_requested_at TEXT CHECK(
        cancel_requested_at IS NULL OR (
            length(cancel_requested_at) = 27 AND substr(cancel_requested_at, -1) = 'Z'
        )
    ),
    cancel_reason TEXT CHECK(cancel_reason IS NULL OR cancel_reason IN (
        'operator_request', 'superseded', 'data_quality', 'resource_policy',
        'risk_control', 'shutdown'
    )),
    terminal_at TEXT CHECK(
        terminal_at IS NULL OR (length(terminal_at) = 27 AND substr(terminal_at, -1) = 'Z')
    ),
    result_run_id TEXT REFERENCES sl_registry_runs(run_id) ON DELETE RESTRICT
        CHECK(result_run_id IS NULL OR length(result_run_id) BETWEEN 1 AND 128),
    failure_code TEXT CHECK(failure_code IS NULL OR failure_code IN (
        'data_unavailable', 'dependency_unavailable', 'execution_error', 'internal_error',
        'invalid_input', 'lease_attempts_exhausted', 'resource_exhausted', 'transient_io'
    )),
    failure_summary TEXT CHECK(failure_summary IS NULL OR length(failure_summary) BETWEEN 1 AND 512),
    CHECK(attempt_count <= max_attempts),
    CHECK(
        (state = 'running' AND lease_owner IS NOT NULL AND lease_token_digest IS NOT NULL
            AND lease_expires_at IS NOT NULL AND heartbeat_at IS NOT NULL)
        OR
        (state <> 'running' AND lease_owner IS NULL AND lease_token_digest IS NULL
            AND lease_expires_at IS NULL AND heartbeat_at IS NULL)
    ),
    CHECK((state IN ('succeeded', 'failed', 'cancelled')) = (terminal_at IS NOT NULL)),
    CHECK((state = 'succeeded') = (result_run_id IS NOT NULL)),
    CHECK((state = 'failed') = (failure_code IS NOT NULL)),
    CHECK((failure_code IS NULL) = (failure_summary IS NULL)),
    CHECK(updated_at >= created_at),
    CHECK(cancel_requested_at IS NULL OR cancel_requested_at >= created_at),
    CHECK(terminal_at IS NULL OR terminal_at >= created_at),
    CHECK(heartbeat_at IS NULL OR heartbeat_at >= created_at),
    CHECK(lease_expires_at IS NULL OR lease_expires_at >= heartbeat_at)
);

CREATE INDEX sl_registry_jobs_queue_idx
    ON sl_registry_jobs(priority DESC, sequence ASC)
    WHERE state = 'queued';
CREATE INDEX sl_registry_jobs_state_sequence_idx
    ON sl_registry_jobs(state, sequence);
CREATE INDEX sl_registry_jobs_lease_expiry_idx
    ON sl_registry_jobs(lease_expires_at, sequence)
    WHERE state = 'running';

CREATE TRIGGER sl_registry_jobs_terminal_immutable
BEFORE UPDATE ON sl_registry_jobs
WHEN OLD.state IN ('succeeded', 'failed', 'cancelled')
BEGIN
    SELECT RAISE(ABORT, 'terminal sl_registry_jobs rows are immutable');
END;

CREATE TABLE sl_registry_events (
    sequence INTEGER PRIMARY KEY AUTOINCREMENT,
    job_id TEXT NOT NULL REFERENCES sl_registry_jobs(job_id) ON DELETE RESTRICT,
    kind TEXT NOT NULL CHECK(kind IN (
        'submitted', 'claimed', 'lease_expired', 'retried',
        'cancel_requested', 'cancelled', 'succeeded', 'failed'
    )),
    from_state TEXT CHECK(from_state IS NULL OR from_state IN (
        'queued', 'running', 'succeeded', 'failed', 'cancelled'
    )),
    to_state TEXT NOT NULL CHECK(to_state IN (
        'queued', 'running', 'succeeded', 'failed', 'cancelled'
    )),
    occurred_at TEXT NOT NULL CHECK(
        length(occurred_at) = 27 AND substr(occurred_at, -1) = 'Z'
    ),
    attempt INTEGER NOT NULL CHECK(attempt >= 0),
    actor TEXT NOT NULL CHECK(length(actor) BETWEEN 1 AND 128),
    details_json TEXT NOT NULL DEFAULT '{}' CHECK(length(details_json) BETWEEN 2 AND 8192)
);
CREATE INDEX sl_registry_events_job_sequence_idx
    ON sl_registry_events(job_id, sequence);

CREATE TRIGGER sl_registry_events_no_update
BEFORE UPDATE ON sl_registry_events
BEGIN
    SELECT RAISE(ABORT, 'sl_registry_events is append-only');
END;

CREATE TRIGGER sl_registry_events_no_delete
BEFORE DELETE ON sl_registry_events
BEGIN
    SELECT RAISE(ABORT, 'sl_registry_events is append-only');
END;

CREATE TABLE sl_registry_runs (
    sequence INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id TEXT NOT NULL UNIQUE CHECK(length(run_id) BETWEEN 1 AND 128),
    job_id TEXT NOT NULL REFERENCES sl_registry_jobs(job_id) ON DELETE RESTRICT,
    attempt INTEGER NOT NULL CHECK(attempt BETWEEN 1 AND 100),
    status TEXT NOT NULL CHECK(status IN ('succeeded', 'failed', 'cancelled')),
    evidence_class TEXT NOT NULL CHECK(evidence_class IN (
        'measured', 'simulated', 'backtested', 'paper_traded', 'live', 'not_applicable'
    )),
    schema_version INTEGER NOT NULL CHECK(schema_version = 1),
    created_at TEXT NOT NULL CHECK(length(created_at) = 27 AND substr(created_at, -1) = 'Z'),
    started_at TEXT CHECK(
        started_at IS NULL OR (length(started_at) = 27 AND substr(started_at, -1) = 'Z')
    ),
    ended_at TEXT NOT NULL CHECK(length(ended_at) = 27 AND substr(ended_at, -1) = 'Z'),
    source_commit TEXT CHECK(source_commit IS NULL OR length(source_commit) BETWEEN 7 AND 64),
    data_identity TEXT CHECK(data_identity IS NULL OR length(data_identity) BETWEEN 1 AND 256),
    limitation_summary TEXT CHECK(
        limitation_summary IS NULL OR length(limitation_summary) BETWEEN 1 AND 4096
    ),
    UNIQUE(job_id, attempt),
    CHECK(created_at <= coalesce(started_at, ended_at)),
    CHECK(coalesce(started_at, created_at) <= ended_at)
);
CREATE INDEX sl_registry_runs_job_sequence_idx
    ON sl_registry_runs(job_id, sequence);
CREATE INDEX sl_registry_runs_status_sequence_idx
    ON sl_registry_runs(status, sequence);

CREATE TRIGGER sl_registry_runs_requires_running_attempt
BEFORE INSERT ON sl_registry_runs
WHEN NOT EXISTS (
    SELECT 1
    FROM sl_registry_jobs
    WHERE job_id = NEW.job_id
      AND state = 'running'
      AND attempt_count = NEW.attempt
)
BEGIN
    SELECT RAISE(ABORT, 'terminal runs require the current running attempt');
END;

CREATE TRIGGER sl_registry_runs_no_update
BEFORE UPDATE ON sl_registry_runs
BEGIN
    SELECT RAISE(ABORT, 'sl_registry_runs is immutable');
END;

CREATE TRIGGER sl_registry_runs_no_delete
BEFORE DELETE ON sl_registry_runs
BEGIN
    SELECT RAISE(ABORT, 'sl_registry_runs is immutable');
END;

CREATE TABLE sl_registry_artifacts (
    digest TEXT PRIMARY KEY CHECK(length(digest) = 64 AND digest NOT GLOB '*[^0-9a-f]*'),
    artifact_class TEXT NOT NULL CHECK(artifact_class IN (
        'input', 'output', 'model', 'report', 'plot', 'log', 'metadata'
    )),
    byte_size INTEGER NOT NULL CHECK(byte_size BETWEEN 0 AND 9223372036854775807),
    media_type TEXT NOT NULL CHECK(
        length(media_type) BETWEEN 3 AND 128
        AND instr(media_type, '/') BETWEEN 2 AND 64
        AND media_type = lower(media_type)
    ),
    storage_relpath TEXT NOT NULL UNIQUE CHECK(
        length(storage_relpath) BETWEEN 1 AND 1024
        AND substr(storage_relpath, 1, 1) NOT IN ('/', char(92))
        AND instr('/' || replace(storage_relpath, char(92), '/') || '/', '/../') = 0
    ),
    created_at TEXT NOT NULL CHECK(length(created_at) = 27 AND substr(created_at, -1) = 'Z'),
    pinned INTEGER NOT NULL DEFAULT 0 CHECK(pinned IN (0, 1))
);
CREATE INDEX sl_registry_artifacts_created_idx
    ON sl_registry_artifacts(created_at DESC, digest);

CREATE TRIGGER sl_registry_artifacts_no_update
BEFORE UPDATE ON sl_registry_artifacts
BEGIN
    SELECT RAISE(ABORT, 'sl_registry_artifacts is immutable');
END;

CREATE TRIGGER sl_registry_artifacts_no_delete
BEFORE DELETE ON sl_registry_artifacts
BEGIN
    SELECT RAISE(ABORT, 'sl_registry_artifacts is immutable');
END;

CREATE TABLE sl_registry_run_artifacts (
    sequence INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id TEXT NOT NULL REFERENCES sl_registry_runs(run_id) ON DELETE RESTRICT,
    role TEXT NOT NULL CHECK(length(role) BETWEEN 1 AND 128),
    artifact_digest TEXT NOT NULL REFERENCES sl_registry_artifacts(digest) ON DELETE RESTRICT,
    linked_at TEXT NOT NULL CHECK(length(linked_at) = 27 AND substr(linked_at, -1) = 'Z'),
    UNIQUE(run_id, role, artifact_digest)
);
CREATE INDEX sl_registry_run_artifacts_run_sequence_idx
    ON sl_registry_run_artifacts(run_id, sequence);
CREATE INDEX sl_registry_run_artifacts_digest_idx
    ON sl_registry_run_artifacts(artifact_digest, sequence);

CREATE TRIGGER sl_registry_run_artifacts_no_update
BEFORE UPDATE ON sl_registry_run_artifacts
BEGIN
    SELECT RAISE(ABORT, 'sl_registry_run_artifacts is immutable');
END;

CREATE TRIGGER sl_registry_run_artifacts_no_delete
BEFORE DELETE ON sl_registry_run_artifacts
BEGIN
    SELECT RAISE(ABORT, 'sl_registry_run_artifacts is immutable');
END;

CREATE TABLE sl_registry_retention_plans (
    plan_digest TEXT PRIMARY KEY CHECK(
        length(plan_digest) = 64 AND plan_digest NOT GLOB '*[^0-9a-f]*'
    ),
    payload_digest TEXT NOT NULL CHECK(
        length(payload_digest) = 64 AND payload_digest NOT GLOB '*[^0-9a-f]*'
    ),
    payload_json TEXT NOT NULL CHECK(
        typeof(payload_json) = 'text'
        AND length(CAST(payload_json AS BLOB)) BETWEEN 2 AND 1048576
    ),
    registry_id TEXT NOT NULL REFERENCES sl_registry_metadata(registry_id) ON DELETE RESTRICT
        CHECK(length(registry_id) = 32 AND registry_id NOT GLOB '*[^0-9a-f]*'),
    cas_store_id TEXT NOT NULL REFERENCES sl_registry_cas_binding(store_id) ON DELETE RESTRICT
        CHECK(length(cas_store_id) = 64 AND cas_store_id NOT GLOB '*[^0-9a-f]*'),
    planned_at TEXT NOT NULL CHECK(
        length(planned_at) = 27 AND substr(planned_at, -1) = 'Z'
    ),
    schema_version INTEGER NOT NULL CHECK(schema_version = 1)
) WITHOUT ROWID;
CREATE INDEX sl_registry_retention_plans_time_idx
    ON sl_registry_retention_plans(planned_at, plan_digest);

CREATE TRIGGER sl_registry_retention_plans_no_update
BEFORE UPDATE ON sl_registry_retention_plans
BEGIN
    SELECT RAISE(ABORT, 'sl_registry_retention_plans is immutable');
END;

CREATE TRIGGER sl_registry_retention_plans_no_delete
BEFORE DELETE ON sl_registry_retention_plans
BEGIN
    SELECT RAISE(ABORT, 'sl_registry_retention_plans is append-only');
END;

CREATE TABLE sl_registry_retention_tombstones (
    artifact_digest TEXT PRIMARY KEY REFERENCES sl_registry_artifacts(digest)
        ON DELETE RESTRICT CHECK(
        length(artifact_digest) = 64 AND artifact_digest NOT GLOB '*[^0-9a-f]*'
    ),
    plan_digest TEXT NOT NULL REFERENCES sl_registry_retention_plans(plan_digest)
        ON DELETE RESTRICT CHECK(
        length(plan_digest) = 64 AND plan_digest NOT GLOB '*[^0-9a-f]*'
    ),
    planned_at TEXT NOT NULL CHECK(length(planned_at) = 27 AND substr(planned_at, -1) = 'Z'),
    deleted_at TEXT CHECK(
        deleted_at IS NULL OR (length(deleted_at) = 27 AND substr(deleted_at, -1) = 'Z')
    ),
    reason TEXT NOT NULL CHECK(length(reason) BETWEEN 1 AND 1024),
    CHECK(deleted_at IS NULL OR deleted_at >= planned_at)
);
CREATE INDEX sl_registry_retention_pending_idx
    ON sl_registry_retention_tombstones(planned_at, artifact_digest)
    WHERE deleted_at IS NULL;
CREATE INDEX sl_registry_retention_plan_pending_idx
    ON sl_registry_retention_tombstones(plan_digest, artifact_digest)
    WHERE deleted_at IS NULL;

CREATE TRIGGER sl_registry_run_artifacts_no_tombstoned_insert
BEFORE INSERT ON sl_registry_run_artifacts
WHEN EXISTS (
    SELECT 1 FROM sl_registry_retention_tombstones
    WHERE artifact_digest = NEW.artifact_digest
)
BEGIN
    SELECT RAISE(ABORT, 'tombstoned artifacts cannot be linked to runs');
END;

CREATE TRIGGER sl_registry_retention_tombstones_pending_insert
BEFORE INSERT ON sl_registry_retention_tombstones
WHEN NEW.deleted_at IS NOT NULL
BEGIN
    SELECT RAISE(ABORT, 'retention tombstones must begin pending');
END;

CREATE TRIGGER sl_registry_retention_tombstones_monotonic_update
BEFORE UPDATE ON sl_registry_retention_tombstones
WHEN NOT (
    OLD.artifact_digest = NEW.artifact_digest
    AND OLD.plan_digest = NEW.plan_digest
    AND OLD.planned_at = NEW.planned_at
    AND OLD.reason = NEW.reason
    AND OLD.deleted_at IS NULL
    AND NEW.deleted_at IS NOT NULL
)
BEGIN
    SELECT RAISE(ABORT, 'retention tombstones permit only deletion finalization');
END;

CREATE TRIGGER sl_registry_retention_tombstones_no_delete
BEFORE DELETE ON sl_registry_retention_tombstones
BEGIN
    SELECT RAISE(ABORT, 'retention tombstones are append-only');
END;
"""


@dataclass(frozen=True, slots=True)
class Migration:
    """One immutable, checksummed forward schema step."""

    version: int
    name: str
    sql: str

    @property
    def checksum(self) -> str:
        """Return the checksum committed to the migration ledger."""

        return hashlib.sha256(self.sql.encode("utf-8")).hexdigest()


MIGRATIONS = (Migration(1, "registry_control_plane", _MIGRATION_1),)
LATEST_SCHEMA_VERSION = MIGRATIONS[-1].version

_MIGRATION_LEDGER_SQL = f"""
CREATE TABLE IF NOT EXISTS {MIGRATION_TABLE} (
    version INTEGER PRIMARY KEY CHECK(version > 0),
    name TEXT NOT NULL UNIQUE CHECK(length(name) BETWEEN 1 AND 128),
    checksum TEXT NOT NULL CHECK(
        length(checksum) = 64 AND checksum NOT GLOB '*[^0-9a-f]*'
    ),
    applied_at TEXT NOT NULL CHECK(
        length(applied_at) = 27 AND substr(applied_at, -1) = 'Z'
    )
)
"""

_EXPECTED_COLUMNS = {
    "sl_registry_metadata": ("singleton", "hmac_version", "key_verifier", "registry_id"),
    "sl_registry_cas_binding": ("singleton", "store_id"),
    "sl_registry_jobs": (
        "sequence",
        "job_id",
        "kind",
        "request_schema_version",
        "request_json",
        "request_digest",
        "idempotency_digest",
        "state",
        "priority",
        "attempt_count",
        "max_attempts",
        "created_at",
        "updated_at",
        "lease_owner",
        "lease_token_digest",
        "lease_expires_at",
        "heartbeat_at",
        "cancel_requested_at",
        "cancel_reason",
        "terminal_at",
        "result_run_id",
        "failure_code",
        "failure_summary",
    ),
    "sl_registry_events": (
        "sequence",
        "job_id",
        "kind",
        "from_state",
        "to_state",
        "occurred_at",
        "attempt",
        "actor",
        "details_json",
    ),
    "sl_registry_runs": (
        "sequence",
        "run_id",
        "job_id",
        "attempt",
        "status",
        "evidence_class",
        "schema_version",
        "created_at",
        "started_at",
        "ended_at",
        "source_commit",
        "data_identity",
        "limitation_summary",
    ),
    "sl_registry_artifacts": (
        "digest",
        "artifact_class",
        "byte_size",
        "media_type",
        "storage_relpath",
        "created_at",
        "pinned",
    ),
    "sl_registry_run_artifacts": (
        "sequence",
        "run_id",
        "role",
        "artifact_digest",
        "linked_at",
    ),
    "sl_registry_retention_plans": (
        "plan_digest",
        "payload_digest",
        "payload_json",
        "registry_id",
        "cas_store_id",
        "planned_at",
        "schema_version",
    ),
    "sl_registry_retention_tombstones": (
        "artifact_digest",
        "plan_digest",
        "planned_at",
        "deleted_at",
        "reason",
    ),
}
_EXPECTED_TRIGGERS = {
    "sl_registry_schema_migrations_no_update",
    "sl_registry_schema_migrations_no_delete",
    "sl_registry_metadata_no_update",
    "sl_registry_metadata_no_delete",
    "sl_registry_cas_binding_no_update",
    "sl_registry_cas_binding_no_delete",
    "sl_registry_jobs_terminal_immutable",
    "sl_registry_events_no_update",
    "sl_registry_events_no_delete",
    "sl_registry_runs_requires_running_attempt",
    "sl_registry_runs_no_update",
    "sl_registry_runs_no_delete",
    "sl_registry_artifacts_no_update",
    "sl_registry_artifacts_no_delete",
    "sl_registry_run_artifacts_no_update",
    "sl_registry_run_artifacts_no_delete",
    "sl_registry_run_artifacts_no_tombstoned_insert",
    "sl_registry_retention_plans_no_update",
    "sl_registry_retention_plans_no_delete",
    "sl_registry_retention_tombstones_pending_insert",
    "sl_registry_retention_tombstones_monotonic_update",
    "sl_registry_retention_tombstones_no_delete",
}
_EXPECTED_INDEXES = {
    "sl_registry_jobs_queue_idx",
    "sl_registry_jobs_state_sequence_idx",
    "sl_registry_jobs_lease_expiry_idx",
    "sl_registry_events_job_sequence_idx",
    "sl_registry_runs_job_sequence_idx",
    "sl_registry_runs_status_sequence_idx",
    "sl_registry_artifacts_created_idx",
    "sl_registry_run_artifacts_run_sequence_idx",
    "sl_registry_run_artifacts_digest_idx",
    "sl_registry_retention_plans_time_idx",
    "sl_registry_retention_pending_idx",
    "sl_registry_retention_plan_pending_idx",
}


def _timestamp() -> str:
    return datetime.now(UTC).isoformat(timespec="microseconds").replace("+00:00", "Z")


def _is_busy(exc: sqlite3.Error) -> bool:
    """Classify SQLite contention from its numeric base result code only."""

    code = getattr(exc, "sqlite_errorcode", None)
    return type(code) is int and (code & 0xFF) in {sqlite3.SQLITE_BUSY, sqlite3.SQLITE_LOCKED}


def _is_interrupted(exc: sqlite3.Error) -> bool:
    """Classify SQLite interruption without trusting attacker-controlled messages."""

    code = getattr(exc, "sqlite_errorcode", None)
    return type(code) is int and (code & 0xFF) == sqlite3.SQLITE_INTERRUPT


def _validate_verification_timeout_ms(value: int) -> int:
    if type(value) is not int or not 1 <= value <= 60_000:
        raise ValidationError("verification_timeout_ms must be in [1, 60000]")
    return value


@dataclass(slots=True)
class _VerificationBudget:
    """One monotonic deadline shared by SQLite VM work and Python row validation."""

    deadline: float
    timed_out: bool = False

    def check(self) -> None:
        """Raise the stable public timeout once the verification deadline is exhausted."""

        if monotonic() >= self.deadline:
            self.timed_out = True
            raise VerificationTimeoutError(
                "registry integrity verification exceeded its configured deadline"
            )

    def sqlite_progress(self) -> int:
        """Interrupt SQLite cooperatively after the same monotonic deadline."""

        if monotonic() >= self.deadline:
            self.timed_out = True
            return 1
        return 0


def _execute_script(connection: sqlite3.Connection, script: str) -> None:
    statement = ""
    for line in script.splitlines(keepends=True):
        statement += line
        if sqlite3.complete_statement(statement):
            if statement.strip():
                connection.execute(statement)
            statement = ""
    if statement.strip():
        raise MigrationError("compiled migration contains an incomplete SQL statement")


def _assert_no_extended_acl(descriptor: int, *, role: str) -> None:
    """Reject non-mode authority where the host exposes a descriptor API."""

    if platform.system() == "Darwin":
        try:
            libc = ctypes.CDLL(None, use_errno=True)
            get_acl = libc.acl_get_fd_np
            get_acl.argtypes = [ctypes.c_int, ctypes.c_int]
            get_acl.restype = ctypes.c_void_p
            free_acl = libc.acl_free
            free_acl.argtypes = [ctypes.c_void_p]
            free_acl.restype = ctypes.c_int
            ctypes.set_errno(0)
            acl = get_acl(descriptor, 0x00000100)  # ACL_TYPE_EXTENDED
        except (AttributeError, OSError):
            raise IntegrityError(f"{role} access controls could not be inspected safely") from None
        if acl is None:
            error = ctypes.get_errno()
            if error in {
                errno.ENOENT,
                errno.ENOTSUP,
                getattr(errno, "EOPNOTSUPP", errno.ENOTSUP),
            }:
                return
            raise IntegrityError(f"{role} access controls could not be inspected safely") from None
        if free_acl(acl) != 0:
            raise IntegrityError(f"{role} access controls could not be inspected safely")
        raise IntegrityError(f"{role} must not grant authority through an extended ACL")
    list_attributes = getattr(os, "listxattr", None)
    if list_attributes is None:  # pragma: no cover - supported POSIX hosts expose this API
        return
    try:
        attributes = list_attributes(descriptor)
    except OSError as exc:
        if exc.errno in {errno.ENOTSUP, getattr(errno, "EOPNOTSUPP", errno.ENOTSUP)}:
            return
        raise IntegrityError(f"{role} access controls could not be inspected safely") from None
    normalized = {
        item.decode("utf-8", "surrogateescape") if isinstance(item, bytes) else item
        for item in attributes
    }
    if normalized.intersection({"system.posix_acl_access", "system.posix_acl_default"}):
        raise IntegrityError(f"{role} must not grant authority through an extended ACL")


def _same_identity(first: os.stat_result, second: os.stat_result) -> bool:
    return (first.st_dev, first.st_ino) == (second.st_dev, second.st_ino)


def _open_parent_directory(path: Path, *, create: bool = False) -> int:
    """Descriptor-walk every parent without following symlinks.

    Existing ancestors may carry conventional shared traversal modes, but the immediate database
    directory is an owner-only 0700 boundary with no non-mode ACL authority.
    """

    absolute = path.absolute()
    if absolute.name in {"", ".", ".."}:
        raise IntegrityError("registry database filename is invalid")
    directory_flags = (
        os.O_RDONLY
        | getattr(os, "O_DIRECTORY", 0)
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0)
    )
    try:
        current = os.open(os.sep, directory_flags)
    except OSError:
        raise IntegrityError("registry database parent could not be inspected safely") from None
    try:
        for component in absolute.parent.parts[1:]:
            try:
                before = os.stat(component, dir_fd=current, follow_symlinks=False)
            except FileNotFoundError:
                if not create:
                    raise IntegrityError("registry database parent does not exist") from None
                try:
                    os.mkdir(component, 0o700, dir_fd=current)
                    before = os.stat(component, dir_fd=current, follow_symlinks=False)
                except OSError:
                    raise IntegrityError(
                        "registry database parent could not be prepared safely"
                    ) from None
            except OSError:
                raise IntegrityError(
                    "registry database parent could not be inspected safely"
                ) from None
            if stat.S_ISLNK(before.st_mode) or not stat.S_ISDIR(before.st_mode):
                raise IntegrityError(
                    "registry database parent components must be non-symlink directories"
                )
            try:
                child = os.open(component, directory_flags, dir_fd=current)
            except OSError:
                raise IntegrityError(
                    "registry database parent components must be non-symlink directories"
                ) from None
            after = os.fstat(child)
            if not _same_identity(before, after) or not stat.S_ISDIR(after.st_mode):
                os.close(child)
                raise IntegrityError("registry database parent identity changed during inspection")
            os.close(current)
            current = child
        immediate = os.fstat(current)
        if immediate.st_uid != os.geteuid():
            raise IntegrityError("registry database directory must be owned by the effective user")
        if stat.S_IMODE(immediate.st_mode) != 0o700:
            raise IntegrityError("registry database directory must have mode 0700")
        _assert_no_extended_acl(current, role="registry database directory")
        return current
    except BaseException:
        os.close(current)
        raise


def _validate_private_file(
    path: Path, *, role: str, required: bool = True
) -> tuple[int, int] | None:
    """Reject absent, redirected, shared, ACL-granted, or non-owner registry files."""

    parent = _open_parent_directory(path)
    descriptor: int | None = None
    try:
        try:
            observed = os.stat(path.name, dir_fd=parent, follow_symlinks=False)
        except FileNotFoundError:
            if required:
                raise IntegrityError(f"{role} does not exist") from None
            return None
        except OSError:
            raise IntegrityError(f"{role} metadata could not be inspected safely") from None
        if stat.S_ISLNK(observed.st_mode) or not stat.S_ISREG(observed.st_mode):
            raise IntegrityError(f"{role} must be a regular non-symlink file")
        try:
            descriptor = os.open(
                path.name,
                os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0),
                dir_fd=parent,
            )
        except FileNotFoundError:
            # WAL and SHM files are lifecycle-managed by SQLite.  The last concurrent connection
            # may remove either one after the no-follow stat but before this descriptor open.  Only
            # that exact absence is benign for an optional companion; the primary database and all
            # other open failures remain fail-closed.  ``open_database`` repeats this validation
            # after its own connection is configured, so a recreated companion is checked before
            # the connection escapes this boundary.
            if not required:
                return None
            raise IntegrityError(f"{role} could not be opened safely") from None
        except OSError:
            raise IntegrityError(f"{role} could not be opened safely") from None
        opened = os.fstat(descriptor)
        if not _same_identity(observed, opened):
            raise IntegrityError(f"{role} identity changed during inspection")
        if opened.st_uid != os.geteuid():
            raise IntegrityError(f"{role} must be owned by the effective user")
        if stat.S_IMODE(opened.st_mode) != 0o600:
            raise IntegrityError(f"{role} must have mode 0600")
        if not required and opened.st_nlink == 0:
            # A concurrent last-close can unlink SQLite's transient companion after this process
            # has opened it but before ``fstat``.  The descriptor remains valid with a zero link
            # count; treating it as absent is equivalent to the exact ENOENT case above.  A link
            # count greater than one is never accepted, so attacker-created aliases still fail.
            return None
        if opened.st_nlink != 1:
            raise IntegrityError(f"{role} must have exactly one filesystem link")
        _assert_no_extended_acl(descriptor, role=role)
        return opened.st_dev, opened.st_ino
    finally:
        if descriptor is not None:
            os.close(descriptor)
        os.close(parent)


def _prepare_bootstrap_file(path: Path) -> None:
    """Create a missing database with mode 0600 without following its final component."""

    parent = _open_parent_directory(path, create=True)
    try:
        descriptor = os.open(
            path.name,
            os.O_RDWR
            | os.O_CREAT
            | os.O_EXCL
            | getattr(os, "O_CLOEXEC", 0)
            | getattr(os, "O_NOFOLLOW", 0),
            0o600,
            dir_fd=parent,
        )
    except FileExistsError:
        _validate_private_file(path, role="registry database")
    except OSError:
        raise IntegrityError(
            "registry database bootstrap file could not be created safely"
        ) from None
    else:
        os.close(descriptor)
        _validate_private_file(path, role="registry database")
    finally:
        os.close(parent)


def _validate_database_files(path: Path) -> tuple[int, int]:
    database_identity = _validate_private_file(path, role="registry database")
    if database_identity is None:  # pragma: no cover - required=True is fail-closed above
        raise IntegrityError("registry database identity is unavailable")
    _validate_private_file(Path(f"{path}-wal"), role="registry WAL", required=False)
    _validate_private_file(Path(f"{path}-shm"), role="registry shared-memory file", required=False)
    return database_identity


def open_database(
    path: Path,
    *,
    busy_timeout_ms: int,
    readonly: bool = False,
    initialize: bool = False,
) -> sqlite3.Connection:
    """Open one bounded SQLite connection with foreign-key enforcement.

    Only an explicit initialization connection may use SQLite ``mode=rwc``. Every operational
    writer uses ``mode=rw`` and every reader uses ``mode=ro``, so an omitted initialization step
    cannot create the primary database or schema. SQLite may create WAL/SHM coordination files for
    a live WAL database; the owner-only parent contains that unavoidable runtime behavior and every
    companion that exists before or immediately after open is validated as a private regular file.
    """

    if type(busy_timeout_ms) is not int or not 0 <= busy_timeout_ms <= 60_000:
        raise ValidationError("busy_timeout_ms must be in [0, 60000]")
    if type(readonly) is not bool or type(initialize) is not bool:
        raise ValidationError("readonly and initialize must be booleans")
    if readonly and initialize:
        raise ValidationError("an initialization connection cannot be read-only")
    database_path = path.absolute()
    if initialize:
        _prepare_bootstrap_file(database_path)
    initial_identity = _validate_database_files(database_path)
    connection: sqlite3.Connection | None = None
    try:
        mode = "ro" if readonly else ("rwc" if initialize else "rw")
        uri = f"{database_path.as_uri()}?mode={mode}"
        connection = sqlite3.connect(uri, uri=True, isolation_level=None, timeout=0)
        if readonly:
            connection.execute("PRAGMA query_only=ON")
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys=ON")
        connection.execute("PRAGMA trusted_schema=OFF")
        connection.execute(f"PRAGMA busy_timeout={busy_timeout_ms}")
        if not readonly:
            mode = str(connection.execute("PRAGMA journal_mode=WAL").fetchone()[0]).lower()
            if mode != "wal":
                connection.close()
                raise IntegrityError(f"SQLite refused WAL mode (reported {mode!r})")
            connection.execute("PRAGMA synchronous=FULL")
        if _validate_database_files(database_path) != initial_identity:
            connection.close()
            raise IntegrityError("registry database identity changed while it was being opened")
        return connection
    except RegistryError:
        if connection is not None:
            connection.close()
        raise
    except sqlite3.Error as exc:
        if connection is not None:
            connection.close()
        if _is_busy(exc):
            raise BusyError("registry database remained busy past its configured bound") from None
        raise IntegrityError("registry database could not be opened safely") from None


def _read_ledger(connection: sqlite3.Connection) -> dict[int, tuple[str, str]]:
    rows = connection.execute(
        f"SELECT version, name, checksum, applied_at FROM {MIGRATION_TABLE} ORDER BY version"
    ).fetchall()
    ledger: dict[int, tuple[str, str]] = {}
    for row in rows:
        version = _lifecycle_int(row["version"], lower=1, upper=2**31 - 1)
        name = row["name"]
        checksum = row["checksum"]
        if type(name) is not str:
            raise IntegrityError("migration ledger name is not canonical text")
        try:
            name_size = require_utf8_size(name, "migration ledger name")
        except ValidationError:
            raise IntegrityError("migration ledger name is not canonical text") from None
        if not 1 <= name_size <= 128:
            raise IntegrityError("migration ledger name is not canonical text")
        if (
            type(checksum) is not str
            or len(checksum) != 64
            or any(character not in "0123456789abcdef" for character in checksum)
        ):
            raise IntegrityError("migration ledger checksum is not canonical")
        _lifecycle_time(row["applied_at"])
        if version in ledger:
            raise IntegrityError("migration ledger version is not unique")
        ledger[version] = (name, checksum)
    return ledger


def _verify_ledger(applied: dict[int, tuple[str, str]]) -> None:
    compiled = {migration.version: migration for migration in MIGRATIONS}
    newer = sorted(set(applied) - set(compiled))
    if newer:
        raise UnsupportedSchemaError(
            f"database schema version {newer[-1]} is newer than supported {LATEST_SCHEMA_VERSION}"
        )
    for version, (name, checksum) in applied.items():
        migration = compiled[version]
        if name != migration.name or checksum != migration.checksum:
            raise MigrationDriftError(f"migration {version} checksum or name does not match")
    expected_prefix = list(range(1, len(applied) + 1))
    if sorted(applied) != expected_prefix:
        raise MigrationDriftError("migration ledger is not a contiguous prefix")


def _schema_objects(connection: sqlite3.Connection) -> dict[tuple[str, str], tuple[str, str]]:
    rows = connection.execute("""
        SELECT type, name, tbl_name, sql
        FROM sqlite_master
        WHERE (name LIKE 'sl_registry_%' OR tbl_name LIKE 'sl_registry_%')
          AND sql IS NOT NULL
        ORDER BY type, name
        """).fetchall()
    return {
        (_schema_text(row["type"]), _schema_text(row["name"])): (
            _schema_text(row["tbl_name"]),
            _schema_text(row["sql"]),
        )
        for row in rows
    }


def _schema_text(value: object) -> str:
    if type(value) is not str:
        raise IntegrityError("registry schema metadata is not exact text")
    return value


def _expected_schema_objects() -> dict[tuple[str, str], tuple[str, str]]:
    reference = sqlite3.connect(":memory:", isolation_level=None)
    reference.row_factory = sqlite3.Row
    try:
        reference.execute(_MIGRATION_LEDGER_SQL)
        for migration in MIGRATIONS:
            _execute_script(reference, migration.sql)
        return _schema_objects(reference)
    except sqlite3.Error:  # pragma: no cover - compiled source is exercised on every test
        raise MigrationError("compiled canonical registry schema is invalid") from None
    finally:
        reference.close()


def _validate_key_verifier(value: str) -> str:
    if (
        type(value) is not str
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValidationError("key_verifier must be a lowercase SHA-256 digest")
    return value


def _validate_registry_id(value: object) -> str:
    if (
        type(value) is not str
        or len(value) != 32
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise IntegrityError("registry instance identity is not canonical")
    return value


def verify_database_binding(
    connection: sqlite3.Connection,
    *,
    key_verifier: str,
) -> str:
    """Return the path-free instance ID after verifying the supplied HMAC authority."""

    expected = _validate_key_verifier(key_verifier)
    count = _lifecycle_int(
        connection.execute("SELECT count(*) FROM sl_registry_metadata").fetchone()[0],
        lower=0,
        upper=2,
    )
    row = connection.execute(
        "SELECT hmac_version, key_verifier, registry_id "
        "FROM sl_registry_metadata WHERE singleton = 1"
    ).fetchone()
    if count != 1 or row is None or _lifecycle_int(row["hmac_version"], lower=1, upper=1) != 1:
        raise IntegrityError("registry HMAC authority metadata is missing or unsupported")
    observed = row["key_verifier"]
    if type(observed) is not str:
        raise IntegrityError("registry HMAC authority metadata is malformed")
    if not hmac.compare_digest(observed, expected):
        raise RegistryAuthorityMismatchError("registry HMAC authority does not match this process")
    return _validate_registry_id(row["registry_id"])


def _decode_stored_json(raw: object, *, maximum_bytes: int) -> object:
    """Decode hostile durable JSON under pre/post budgets and canonical semantics."""

    if type(raw) is not str:
        raise IntegrityError("stored JSON is not text")
    try:
        return decode_bounded_json(raw, maximum_bytes=maximum_bytes)
    except ValidationError:
        raise IntegrityError("stored JSON violates its bounded canonical contract") from None


def _retention_digest(value: object, *, field_name: str) -> str:
    """Require one exact lowercase SHA-256 text identity."""

    if (
        type(value) is not str
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise IntegrityError(f"stored retention {field_name} is not canonical")
    return value


def _retention_snapshot_binding(connection: sqlite3.Connection) -> tuple[str, str]:
    """Read the exact registry/CAS binding inside the verifier's current SQLite snapshot."""

    metadata = connection.execute(
        "SELECT singleton, registry_id FROM sl_registry_metadata ORDER BY singleton LIMIT 2"
    ).fetchall()
    bindings = connection.execute(
        "SELECT singleton, store_id FROM sl_registry_cas_binding ORDER BY singleton LIMIT 2"
    ).fetchall()
    if len(metadata) != 1 or len(bindings) != 1:
        raise IntegrityError("stored retention authority binding is missing or ambiguous")
    _lifecycle_int(metadata[0]["singleton"], lower=1, upper=1)
    _lifecycle_int(bindings[0]["singleton"], lower=1, upper=1)
    registry_id = _validate_registry_id(metadata[0]["registry_id"])
    store_id = _retention_digest(bindings[0]["store_id"], field_name="CAS identity")
    return registry_id, store_id


def _authenticate_stored_retention_plan(
    authenticator: RetentionPlanAuthenticator | None,
    *,
    payload: bytes,
    registry_id: str,
    store_id: str,
    plan_digest: str,
) -> None:
    """Fail closed unless the in-process secret authority authenticates this exact envelope."""

    if authenticator is None:
        raise IntegrityError("stored retention plans require an authentication authority")
    try:
        authenticated = authenticator(payload, registry_id, store_id, plan_digest)
    except Exception:
        raise IntegrityError("stored retention plan authentication failed") from None
    if type(authenticated) is not bool or not authenticated:
        raise IntegrityError("stored retention plan authentication failed")


def _verify_retention_candidate(
    connection: sqlite3.Connection,
    candidate: RetentionCandidate,
    budget: _VerificationBudget,
) -> None:
    """Bind one signed candidate to exact immutable metadata and an unlinked registry object."""

    budget.check()
    rows = connection.execute(
        """
        SELECT digest, artifact_class, byte_size, media_type, storage_relpath,
               created_at, pinned
        FROM sl_registry_artifacts
        WHERE digest = ?
        LIMIT 2
        """,
        (candidate.digest,),
    ).fetchall()
    if len(rows) != 1:
        raise IntegrityError("stored retention candidate metadata is missing or ambiguous")
    row = rows[0]
    byte_size = _lifecycle_int(row["byte_size"], lower=0, upper=2**63 - 1)
    pinned = _lifecycle_int(row["pinned"], lower=0, upper=1)
    created_at = _lifecycle_time(row["created_at"])
    metadata = candidate.metadata
    if (
        row["digest"] != metadata.digest
        or row["artifact_class"] != metadata.artifact_class.value
        or byte_size != metadata.byte_size
        or row["media_type"] != metadata.media_type
        or row["storage_relpath"] != metadata.storage_relpath
        or created_at != metadata.created_at
        or bool(pinned) is not metadata.pinned
    ):
        raise IntegrityError("stored retention candidate disagrees with artifact metadata")
    linked = connection.execute(
        """
        SELECT 1
        FROM sl_registry_run_artifacts
        WHERE artifact_digest = ?
        LIMIT 1
        """,
        (candidate.digest,),
    ).fetchone()
    if linked is not None:
        raise IntegrityError("stored retention candidate is linked to immutable run evidence")


def _verify_retention_tombstones(
    connection: sqlite3.Connection,
    *,
    plan_digest: str,
    plan: DecodedRetentionPlan,
    budget: _VerificationBudget,
) -> None:
    """Require one exact pending-or-finalized tombstone for every signed candidate."""

    rows = connection.execute(
        """
        SELECT artifact_digest, plan_digest, planned_at, deleted_at, reason
        FROM sl_registry_retention_tombstones
        WHERE plan_digest = ?
        ORDER BY artifact_digest
        LIMIT ?
        """,
        (plan_digest, len(plan.candidates) + 1),
    ).fetchall()
    expected_digests = plan.plan.artifact_digests
    if len(rows) != len(expected_digests):
        raise IntegrityError("stored retention plan has an incomplete or oversized tombstone set")
    observed_digests: list[str] = []
    for row in rows:
        budget.check()
        artifact_digest = _retention_digest(
            row["artifact_digest"],
            field_name="tombstone artifact identity",
        )
        tombstone_plan = _retention_digest(
            row["plan_digest"],
            field_name="tombstone plan identity",
        )
        planned_at = _lifecycle_time(row["planned_at"])
        deleted_at = _optional_lifecycle_time(row["deleted_at"])
        reason = row["reason"]
        if (
            not hmac.compare_digest(tombstone_plan, plan_digest)
            or planned_at != plan.plan.planned_at
            or reason != plan.plan.reason
            or (deleted_at is not None and deleted_at < planned_at)
        ):
            raise IntegrityError("stored retention tombstone disagrees with its signed plan")
        observed_digests.append(artifact_digest)
    if tuple(observed_digests) != expected_digests:
        raise IntegrityError("stored retention tombstones do not match the signed candidate set")


def _verify_retention_plan_content(
    connection: sqlite3.Connection,
    budget: _VerificationBudget,
    authenticator: RetentionPlanAuthenticator | None,
) -> None:
    """Verify every retention envelope, authority binding, candidate, and tombstone cohort."""

    after_digest = ""
    snapshot_binding: tuple[str, str] | None = None
    while True:
        budget.check()
        rows = connection.execute(
            """
            SELECT plan_digest, payload_json, payload_digest, registry_id,
                   cas_store_id, planned_at, schema_version
            FROM sl_registry_retention_plans
            WHERE plan_digest > ?
            ORDER BY plan_digest
            LIMIT ?
            """,
            (after_digest, _VERIFICATION_BATCH_SIZE + 1),
        ).fetchall()
        page = rows[:_VERIFICATION_BATCH_SIZE]
        for row in page:
            budget.check()
            plan_digest = _retention_digest(row["plan_digest"], field_name="plan identity")
            if plan_digest <= after_digest:
                raise IntegrityError("stored retention plan sequence is not canonical")
            payload_json = row["payload_json"]
            if type(payload_json) is not str:
                raise IntegrityError("stored retention payload is not text")
            payload_digest = _retention_digest(
                row["payload_digest"],
                field_name="payload digest",
            )
            try:
                payload_bytes = payload_json.encode("utf-8")
            except (MemoryError, UnicodeEncodeError):
                raise IntegrityError("stored retention payload is not bounded UTF-8") from None
            if not hmac.compare_digest(
                payload_digest,
                hashlib.sha256(payload_bytes).hexdigest(),
            ):
                raise IntegrityError("stored retention payload digest does not match its bytes")
            decoded_json = _decode_stored_json(payload_json, maximum_bytes=1_048_576)
            try:
                plan = validate_decoded_retention_plan(
                    payload_json,
                    decoded_json,
                    checkpoint=budget.check,
                )
            except (
                RetentionIntegrityError,
                TypeError,
                ValueError,
                ValidationError,
            ):
                raise IntegrityError("stored retention payload is semantically invalid") from None

            row_registry_id = _validate_registry_id(row["registry_id"])
            row_store_id = _retention_digest(
                row["cas_store_id"],
                field_name="CAS identity",
            )
            row_planned_at = _lifecycle_time(row["planned_at"])
            row_schema_version = _lifecycle_int(row["schema_version"], lower=1, upper=1)
            if snapshot_binding is None:
                snapshot_binding = _retention_snapshot_binding(connection)
            snapshot_registry_id, snapshot_store_id = snapshot_binding
            if (
                not hmac.compare_digest(plan.registry_id, row_registry_id)
                or not hmac.compare_digest(plan.cas_store_id, row_store_id)
                or plan.plan.planned_at != row_planned_at
                or plan.schema_version != row_schema_version
                or not hmac.compare_digest(row_registry_id, snapshot_registry_id)
                or not hmac.compare_digest(row_store_id, snapshot_store_id)
            ):
                raise IntegrityError("stored retention envelope crosses its indexed authority")
            _authenticate_stored_retention_plan(
                authenticator,
                payload=payload_bytes,
                registry_id=snapshot_registry_id,
                store_id=snapshot_store_id,
                plan_digest=plan_digest,
            )
            for candidate in plan.candidates:
                _verify_retention_candidate(connection, candidate, budget)
            _verify_retention_tombstones(
                connection,
                plan_digest=plan_digest,
                plan=plan,
                budget=budget,
            )
            after_digest = plan_digest
        if len(rows) <= _VERIFICATION_BATCH_SIZE:
            return


def _verify_request_content(
    connection: sqlite3.Connection,
    budget: _VerificationBudget,
) -> None:
    after_sequence: int | None = None
    while True:
        budget.check()
        if after_sequence is None:
            rows = connection.execute(
                """
                SELECT sequence, kind, request_schema_version, request_json, request_digest,
                       priority, max_attempts, failure_summary
                FROM sl_registry_jobs
                ORDER BY sequence
                LIMIT ?
                """,
                (_VERIFICATION_BATCH_SIZE + 1,),
            ).fetchall()
        else:
            rows = connection.execute(
                """
                SELECT sequence, kind, request_schema_version, request_json, request_digest,
                       priority, max_attempts, failure_summary
                FROM sl_registry_jobs
                WHERE sequence > ?
                ORDER BY sequence
                LIMIT ?
                """,
                (after_sequence, _VERIFICATION_BATCH_SIZE + 1),
            ).fetchall()
        page = rows[:_VERIFICATION_BATCH_SIZE]
        last_sequence = after_sequence
        for row in page:
            budget.check()
            sequence = _lifecycle_int(row["sequence"], lower=1, upper=2**63 - 1)
            if after_sequence is not None and sequence <= after_sequence:
                raise IntegrityError("stored registry job sequence is not canonical")
            last_sequence = sequence
            raw = row["request_json"]
            digest = row["request_digest"]
            if type(raw) is not str or type(digest) is not str:
                raise IntegrityError("stored canonical request fields are not text")
            try:
                encoded = raw.encode("utf-8")
            except (MemoryError, UnicodeEncodeError):
                raise IntegrityError("stored canonical request is not bounded UTF-8") from None
            if not 2 <= len(encoded) <= 262_144:
                raise IntegrityError("stored canonical request exceeds its durable bound")
            expected = hashlib.sha256(encoded).hexdigest()
            if not hmac.compare_digest(digest, expected):
                raise IntegrityError("stored canonical request digest does not match its bytes")
            try:
                decoded = _decode_stored_json(raw, maximum_bytes=262_144)
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
            except (
                IntegrityError,
                MemoryError,
                RecursionError,
                TypeError,
                ValueError,
                ValidationError,
            ):
                raise IntegrityError("stored canonical request is malformed") from None
            if not hmac.compare_digest(request.canonical_json(), raw):
                raise IntegrityError("stored request JSON is not canonical")
            request_schema_version = _lifecycle_int(row["request_schema_version"], lower=1, upper=1)
            priority = _lifecycle_int(row["priority"], lower=-100, upper=100)
            max_attempts = _lifecycle_int(row["max_attempts"], lower=1, upper=100)
            if (
                request.kind != row["kind"]
                or request.schema_version != request_schema_version
                or request.priority != priority
                or request.max_attempts != max_attempts
            ):
                raise IntegrityError("stored request JSON disagrees with indexed job semantics")
            if row["failure_summary"] is not None:
                try:
                    require_failure_summary(row["failure_summary"])
                except ValidationError as exc:
                    raise IntegrityError(
                        "stored failure summary violates its public boundary"
                    ) from exc
        if len(rows) <= _VERIFICATION_BATCH_SIZE:
            return
        if last_sequence is None:
            raise IntegrityError("stored registry job page made no progress")
        after_sequence = last_sequence


def _lifecycle_int(value: object, *, lower: int, upper: int) -> int:
    """Require exact SQLite INTEGER storage without lossy REAL/text coercion."""

    return require_stored_int(value, "lifecycle integer", lower, upper)


def _lifecycle_time(value: object) -> datetime:
    """Require the exact canonical UTC-microsecond representation used by registry writes."""

    return parse_stored_utc(value, "lifecycle timestamp")


def _optional_lifecycle_time(value: object) -> datetime | None:
    return None if value is None else _lifecycle_time(value)


def _lifecycle_identifier(value: object) -> str:
    if type(value) is not str:
        raise IntegrityError("stored lifecycle identifier is malformed")
    try:
        return require_identifier(value, "stored lifecycle identifier")
    except ValidationError:
        raise IntegrityError("stored lifecycle identifier is malformed") from None


def _lifecycle_enum[EnumT: StrEnum](enum_type: type[EnumT], value: object) -> EnumT:
    if type(value) is not str:
        raise IntegrityError("stored lifecycle enum is malformed")
    try:
        return enum_type(value)
    except (TypeError, ValueError):
        raise IntegrityError("stored lifecycle enum is malformed") from None


def _event_details(value: object) -> dict[str, object]:
    decoded = _decode_stored_json(value, maximum_bytes=8_192)
    if type(decoded) is not dict or canonical_json(decoded) != value:
        raise IntegrityError("stored lifecycle event details are not canonical")
    return cast(dict[str, object], decoded)


@dataclass(frozen=True, slots=True)
class _VerifiedLifecycleRun:
    run_id: str
    status: RunStatus
    created_at: datetime
    started_at: datetime
    ended_at: datetime


@dataclass(frozen=True, slots=True)
class _AttemptClosure:
    kind: EventKind
    to_state: JobState
    occurred_at: datetime


def _load_lifecycle_runs(
    connection: sqlite3.Connection,
    *,
    job_id: str,
    created_at: datetime,
    budget: _VerificationBudget,
) -> dict[int, _VerifiedLifecycleRun]:
    budget.check()
    rows = connection.execute(
        """
        SELECT sequence, attempt, run_id, status, evidence_class, schema_version,
               created_at, started_at, ended_at
        FROM sl_registry_runs
        WHERE job_id = ?
        ORDER BY attempt
        LIMIT ?
        """,
        (job_id, _MAX_LIFECYCLE_RUNS_PER_JOB + 1),
    ).fetchall()
    if len(rows) > _MAX_LIFECYCLE_RUNS_PER_JOB:
        raise IntegrityError("stored job exceeds the terminal-run lifecycle bound")
    runs: dict[int, _VerifiedLifecycleRun] = {}
    previous_sequence = 0
    for row in rows:
        budget.check()
        sequence = _lifecycle_int(row["sequence"], lower=1, upper=2**63 - 1)
        if sequence <= previous_sequence:
            raise IntegrityError("stored terminal-run sequence is not canonical")
        previous_sequence = sequence
        attempt = _lifecycle_int(row["attempt"], lower=1, upper=100)
        if attempt in runs:
            raise IntegrityError("stored terminal-run attempts are not unique")
        _lifecycle_int(row["schema_version"], lower=1, upper=1)
        _lifecycle_enum(EvidenceClass, row["evidence_class"])
        run_created_at = _lifecycle_time(row["created_at"])
        started_at = _lifecycle_time(row["started_at"])
        ended_at = _lifecycle_time(row["ended_at"])
        if run_created_at != created_at or not run_created_at <= started_at <= ended_at:
            raise IntegrityError("stored terminal-run chronology is inconsistent")
        runs[attempt] = _VerifiedLifecycleRun(
            run_id=_lifecycle_identifier(row["run_id"]),
            status=_lifecycle_enum(RunStatus, row["status"]),
            created_at=run_created_at,
            started_at=started_at,
            ended_at=ended_at,
        )
    return runs


def _verify_job_lifecycle(
    connection: sqlite3.Connection,
    row: sqlite3.Row,
    budget: _VerificationBudget,
) -> None:
    """Replay one bounded append-only lifecycle and reconcile its snapshot and runs."""

    budget.check()
    job_id = _lifecycle_identifier(row["job_id"])
    if len(job_id) != 32 or any(character not in "0123456789abcdef" for character in job_id):
        raise IntegrityError("stored lifecycle job identity is not canonical")
    state = _lifecycle_enum(JobState, row["state"])
    attempt_count = _lifecycle_int(row["attempt_count"], lower=0, upper=100)
    max_attempts = _lifecycle_int(row["max_attempts"], lower=1, upper=100)
    if attempt_count > max_attempts:
        raise IntegrityError("stored job attempt count exceeds its maximum")
    request_digest = row["request_digest"]
    if (
        type(request_digest) is not str
        or len(request_digest) != 64
        or any(character not in "0123456789abcdef" for character in request_digest)
    ):
        raise IntegrityError("stored lifecycle request digest is not canonical")

    created_at = _lifecycle_time(row["created_at"])
    updated_at = _lifecycle_time(row["updated_at"])
    heartbeat_at = _optional_lifecycle_time(row["heartbeat_at"])
    lease_expires_at = _optional_lifecycle_time(row["lease_expires_at"])
    cancel_requested_at = _optional_lifecycle_time(row["cancel_requested_at"])
    terminal_at = _optional_lifecycle_time(row["terminal_at"])
    lease_owner = None if row["lease_owner"] is None else _lifecycle_identifier(row["lease_owner"])
    cancel_reason = (
        None
        if row["cancel_reason"] is None
        else _lifecycle_enum(CancellationReasonCode, row["cancel_reason"])
    )
    failure_code = (
        None
        if row["failure_code"] is None
        else _lifecycle_enum(FailureReasonCode, row["failure_code"])
    )
    result_run_id = (
        None if row["result_run_id"] is None else _lifecycle_identifier(row["result_run_id"])
    )
    if updated_at < created_at:
        raise IntegrityError("stored job chronology is inconsistent")

    runs = _load_lifecycle_runs(
        connection,
        job_id=job_id,
        created_at=created_at,
        budget=budget,
    )
    event_rows = connection.execute(
        """
        SELECT sequence, kind, from_state, to_state, occurred_at, attempt, actor, details_json
        FROM sl_registry_events
        WHERE job_id = ?
        ORDER BY sequence
        LIMIT ?
        """,
        (job_id, _MAX_LIFECYCLE_EVENTS_PER_JOB + 1),
    ).fetchall()
    if not event_rows or len(event_rows) > _MAX_LIFECYCLE_EVENTS_PER_JOB:
        raise IntegrityError("stored job lifecycle event count is invalid")

    current_state: JobState | None = None
    current_attempt = 0
    current_actor: str | None = None
    claims: dict[int, tuple[datetime, str]] = {}
    closures: dict[int, _AttemptClosure] = {}
    observed_cancel_at: datetime | None = None
    observed_cancel_reason: CancellationReasonCode | None = None
    previous_sequence = 0
    previous_time: datetime | None = None
    last_kind: EventKind | None = None
    last_from_state: JobState | None = None

    for index, event_row in enumerate(event_rows):
        budget.check()
        sequence = _lifecycle_int(event_row["sequence"], lower=1, upper=2**63 - 1)
        if sequence <= previous_sequence:
            raise IntegrityError("stored lifecycle event sequence is not monotonic")
        previous_sequence = sequence
        kind = _lifecycle_enum(EventKind, event_row["kind"])
        from_state = (
            None
            if event_row["from_state"] is None
            else _lifecycle_enum(JobState, event_row["from_state"])
        )
        to_state = _lifecycle_enum(JobState, event_row["to_state"])
        occurred_at = _lifecycle_time(event_row["occurred_at"])
        if previous_time is not None and occurred_at < previous_time:
            raise IntegrityError("stored lifecycle event time is not monotonic")
        previous_time = occurred_at
        attempt = _lifecycle_int(event_row["attempt"], lower=0, upper=100)
        actor = _lifecycle_identifier(event_row["actor"])
        details = _event_details(event_row["details_json"])

        if index == 0:
            if (
                kind is not EventKind.SUBMITTED
                or from_state is not None
                or to_state is not JobState.QUEUED
                or attempt != 0
                or actor != "submitter"
                or occurred_at != created_at
                or details != {"request_digest": request_digest}
            ):
                raise IntegrityError("stored job submission event is inconsistent")
            current_state = JobState.QUEUED
            last_kind = kind
            last_from_state = from_state
            continue

        if current_state is None or from_state is not current_state:
            raise IntegrityError("stored lifecycle transition chain is inconsistent")
        if kind is EventKind.SUBMITTED:
            raise IntegrityError("stored lifecycle contains a duplicate submission")
        if kind is EventKind.CLAIMED:
            expected_attempt = current_attempt + 1
            if (
                current_state is not JobState.QUEUED
                or to_state is not JobState.RUNNING
                or attempt != expected_attempt
                or attempt > max_attempts
                or set(details) != {"lease_expires_at"}
            ):
                raise IntegrityError("stored claim transition is inconsistent")
            claimed_expiry = _lifecycle_time(details["lease_expires_at"])
            if claimed_expiry < occurred_at:
                raise IntegrityError("stored claim expiry precedes its claim")
            current_attempt = attempt
            current_actor = actor
            claims[attempt] = (occurred_at, actor)
            current_state = JobState.RUNNING
        elif kind is EventKind.CANCEL_REQUESTED:
            if (
                current_state not in {JobState.QUEUED, JobState.RUNNING}
                or to_state is not current_state
                or attempt != current_attempt
                or observed_cancel_at is not None
                or actor != "requester"
                or set(details) != {"reason_code"}
            ):
                raise IntegrityError("stored cancellation request is inconsistent")
            observed_cancel_reason = _lifecycle_enum(CancellationReasonCode, details["reason_code"])
            observed_cancel_at = occurred_at
        elif kind is EventKind.RETRIED:
            if (
                current_state is not JobState.RUNNING
                or to_state is not JobState.QUEUED
                or attempt != current_attempt
                or observed_cancel_at is not None
                or actor != current_actor
                or set(details) != {"failure_code"}
            ):
                raise IntegrityError("stored retry transition is inconsistent")
            _lifecycle_enum(FailureReasonCode, details["failure_code"])
            closures[attempt] = _AttemptClosure(kind, to_state, occurred_at)
            current_state = JobState.QUEUED
            current_actor = None
        elif kind is EventKind.LEASE_EXPIRED:
            expected_state = (
                JobState.CANCELLED
                if observed_cancel_at is not None
                else (JobState.FAILED if attempt >= max_attempts else JobState.QUEUED)
            )
            if (
                current_state is not JobState.RUNNING
                or to_state is not expected_state
                or attempt != current_attempt
                or actor != "registry"
                or details
            ):
                raise IntegrityError("stored lease-expiry transition is inconsistent")
            closures[attempt] = _AttemptClosure(kind, to_state, occurred_at)
            current_state = to_state
            current_actor = None
        elif kind is EventKind.CANCELLED:
            if observed_cancel_at is None or attempt != current_attempt or details:
                raise IntegrityError("stored cancellation transition is inconsistent")
            if current_state is JobState.RUNNING:
                if to_state is not JobState.CANCELLED or actor != current_actor:
                    raise IntegrityError("stored running cancellation is inconsistent")
                closures[attempt] = _AttemptClosure(kind, to_state, occurred_at)
            elif current_state is JobState.QUEUED:
                if to_state is not JobState.CANCELLED or actor != "registry":
                    raise IntegrityError("stored queued cancellation is inconsistent")
            else:
                raise IntegrityError("stored cancellation source state is invalid")
            current_state = JobState.CANCELLED
            current_actor = None
        elif kind in {EventKind.SUCCEEDED, EventKind.FAILED}:
            expected_state = JobState.SUCCEEDED if kind is EventKind.SUCCEEDED else JobState.FAILED
            if (
                current_state is not JobState.RUNNING
                or to_state is not expected_state
                or attempt != current_attempt
                or observed_cancel_at is not None
                or actor != current_actor
                or details
            ):
                raise IntegrityError("stored terminal transition is inconsistent")
            closures[attempt] = _AttemptClosure(kind, to_state, occurred_at)
            current_state = to_state
            current_actor = None
        else:  # Exhaustive fail-closed guard for future enum additions.
            raise IntegrityError("stored lifecycle event kind is unsupported")
        last_kind = kind
        last_from_state = from_state

    if current_state is not state or current_attempt != attempt_count or previous_time is None:
        raise IntegrityError("stored job snapshot disagrees with lifecycle replay")
    if set(claims) != set(range(1, attempt_count + 1)):
        raise IntegrityError("stored lifecycle claims are not contiguous")
    for attempt in range(1, attempt_count + 1):
        closure = closures.get(attempt)
        if attempt < attempt_count and (closure is None or closure.to_state is not JobState.QUEUED):
            raise IntegrityError("stored prior attempt did not requeue canonically")
    current_closure = closures.get(attempt_count)
    if state is JobState.RUNNING and current_closure is not None:
        raise IntegrityError("stored running attempt is already closed")
    if (
        state is JobState.QUEUED
        and attempt_count > 0
        and (current_closure is None or current_closure.to_state is not JobState.QUEUED)
    ):
        raise IntegrityError("stored queued attempt has no requeue transition")
    if state in {JobState.SUCCEEDED, JobState.FAILED} and (
        current_closure is None or current_closure.to_state is not state
    ):
        raise IntegrityError("stored terminal attempt has no matching transition")
    if (
        state is JobState.CANCELLED
        and last_from_state is JobState.RUNNING
        and (current_closure is None or current_closure.to_state is not JobState.CANCELLED)
    ):
        raise IntegrityError("stored running cancellation has no matching transition")

    if observed_cancel_at is None:
        if (
            cancel_requested_at is not None
            or cancel_reason is not None
            or state is JobState.CANCELLED
        ):
            raise IntegrityError("stored cancellation fields disagree with lifecycle events")
    elif (
        cancel_requested_at != observed_cancel_at
        or cancel_reason is not observed_cancel_reason
        or state not in {JobState.RUNNING, JobState.CANCELLED}
    ):
        raise IntegrityError("stored cancellation fields disagree with lifecycle events")

    if state.terminal:
        if terminal_at != previous_time or updated_at != previous_time:
            raise IntegrityError("stored terminal chronology disagrees with lifecycle events")
    elif terminal_at is not None:
        raise IntegrityError("stored nonterminal job carries terminal chronology")
    elif state is JobState.QUEUED and updated_at != previous_time:
        raise IntegrityError("stored queued chronology disagrees with lifecycle events")

    if state is JobState.RUNNING:
        claim = claims.get(attempt_count)
        if (
            claim is None
            or lease_owner != claim[1]
            or heartbeat_at is None
            or lease_expires_at is None
            or not claim[0] <= heartbeat_at <= updated_at <= lease_expires_at
        ):
            raise IntegrityError("stored running lease disagrees with lifecycle events")
    elif lease_owner is not None or heartbeat_at is not None or lease_expires_at is not None:
        raise IntegrityError("stored non-running job carries lease state")

    for attempt, run in runs.items():
        claim = claims.get(attempt)
        closure = closures.get(attempt)
        if claim is None or closure is None or closure.kind is EventKind.LEASE_EXPIRED:
            raise IntegrityError("stored terminal run has no legal attempt closure")
        expected_status = {
            EventKind.RETRIED: RunStatus.FAILED,
            EventKind.SUCCEEDED: RunStatus.SUCCEEDED,
            EventKind.FAILED: RunStatus.FAILED,
            EventKind.CANCELLED: RunStatus.CANCELLED,
        }.get(closure.kind)
        if (
            expected_status is None
            or run.status is not expected_status
            or run.started_at < claim[0]
            or run.ended_at != closure.occurred_at
        ):
            raise IntegrityError("stored terminal run disagrees with its attempt closure")
    for attempt, closure in closures.items():
        if closure.kind is EventKind.LEASE_EXPIRED:
            if attempt in runs:
                raise IntegrityError("lease-expired attempt carries a forged terminal run")
        elif attempt not in runs:
            raise IntegrityError("explicit attempt closure is missing its terminal run")

    if state is JobState.SUCCEEDED:
        succeeded_run = runs.get(attempt_count)
        if (
            succeeded_run is None
            or result_run_id != succeeded_run.run_id
            or failure_code is not None
        ):
            raise IntegrityError("stored succeeded job disagrees with its terminal run")
    elif result_run_id is not None:
        raise IntegrityError("stored non-succeeded job carries a result run")
    if state is JobState.FAILED:
        if failure_code is None:
            raise IntegrityError("stored failed job lacks its failure code")
        if last_kind is EventKind.LEASE_EXPIRED and (
            failure_code is not FailureReasonCode.LEASE_ATTEMPTS_EXHAUSTED
        ):
            raise IntegrityError("stored lease-exhausted job has the wrong failure code")
    elif failure_code is not None:
        raise IntegrityError("stored non-failed job carries a failure code")


def _verify_lifecycle_content(
    connection: sqlite3.Connection,
    budget: _VerificationBudget,
) -> None:
    """Keyset-scan every job and replay bounded lifecycle/run semantics."""

    after_sequence = 0
    while True:
        budget.check()
        rows = connection.execute(
            """
            SELECT sequence, job_id, request_digest, state, attempt_count, max_attempts,
                   created_at, updated_at, lease_owner, lease_expires_at, heartbeat_at,
                   cancel_requested_at, cancel_reason, terminal_at, result_run_id,
                   failure_code
            FROM sl_registry_jobs
            WHERE sequence > ?
            ORDER BY sequence
            LIMIT ?
            """,
            (after_sequence, _VERIFICATION_BATCH_SIZE + 1),
        ).fetchall()
        page = rows[:_VERIFICATION_BATCH_SIZE]
        for row in page:
            sequence = _lifecycle_int(row["sequence"], lower=1, upper=2**63 - 1)
            if sequence <= after_sequence:
                raise IntegrityError("stored registry job sequence is not canonical")
            _verify_job_lifecycle(connection, row, budget)
            after_sequence = sequence
        if len(rows) <= _VERIFICATION_BATCH_SIZE:
            return


def _verify_artifact_content(
    connection: sqlite3.Connection,
    budget: _VerificationBudget,
) -> None:
    after_digest = ""
    while True:
        budget.check()
        rows = connection.execute(
            """
            SELECT digest, artifact_class, byte_size, media_type, storage_relpath,
                   created_at, pinned
            FROM sl_registry_artifacts
            WHERE digest > ?
            ORDER BY digest
            LIMIT ?
            """,
            (after_digest, _VERIFICATION_BATCH_SIZE + 1),
        ).fetchall()
        page = rows[:_VERIFICATION_BATCH_SIZE]
        for row in page:
            budget.check()
            digest = row["digest"]
            if (
                type(digest) is not str
                or len(digest) != 64
                or any(character not in "0123456789abcdef" for character in digest)
                or digest <= after_digest
            ):
                raise IntegrityError("stored artifact identity is not canonical")
            byte_size = _lifecycle_int(row["byte_size"], lower=0, upper=2**63 - 1)
            pinned = _lifecycle_int(row["pinned"], lower=0, upper=1)
            try:
                ArtifactMetadata(
                    digest=digest,
                    artifact_class=_lifecycle_enum(ArtifactClass, row["artifact_class"]),
                    byte_size=byte_size,
                    media_type=row["media_type"],
                    storage_relpath=row["storage_relpath"],
                    created_at=_lifecycle_time(row["created_at"]),
                    pinned=bool(pinned),
                )
            except (TypeError, ValidationError):
                raise IntegrityError("stored artifact metadata is malformed") from None
            expected_storage_key = f"objects/{digest[:2]}/{digest[2:4]}/{digest}"
            if row["storage_relpath"] != expected_storage_key:
                raise IntegrityError("stored artifact key does not match its digest")
            after_digest = digest
        if len(rows) <= _VERIFICATION_BATCH_SIZE:
            return


def _verify_link_content(
    connection: sqlite3.Connection,
    budget: _VerificationBudget,
) -> None:
    after_sequence = 0
    while True:
        budget.check()
        rows = connection.execute(
            """
            SELECT sequence, run_id, role, artifact_digest, linked_at
            FROM sl_registry_run_artifacts
            WHERE sequence > ?
            ORDER BY sequence
            LIMIT ?
            """,
            (after_sequence, _VERIFICATION_BATCH_SIZE + 1),
        ).fetchall()
        page = rows[:_VERIFICATION_BATCH_SIZE]
        for row in page:
            budget.check()
            sequence = _lifecycle_int(row["sequence"], lower=1, upper=2**63 - 1)
            if sequence <= after_sequence:
                raise IntegrityError("stored artifact-link sequence is not canonical")
            _lifecycle_identifier(row["run_id"])
            _lifecycle_identifier(row["role"])
            digest = row["artifact_digest"]
            if (
                type(digest) is not str
                or len(digest) != 64
                or any(character not in "0123456789abcdef" for character in digest)
            ):
                raise IntegrityError("stored artifact-link digest is not canonical")
            linked_at = _lifecycle_time(row["linked_at"])
            ended_row = connection.execute(
                "SELECT ended_at FROM sl_registry_runs WHERE run_id = ?",
                (row["run_id"],),
            ).fetchone()
            if ended_row is None or linked_at != _lifecycle_time(ended_row["ended_at"]):
                raise IntegrityError("stored artifact-link chronology is inconsistent")
            after_sequence = sequence
        if len(rows) <= _VERIFICATION_BATCH_SIZE:
            return


def _verify_retention_scalar_content(
    connection: sqlite3.Connection,
    budget: _VerificationBudget,
) -> None:
    after_digest = ""
    while True:
        budget.check()
        rows = connection.execute(
            """
            SELECT plan_digest, planned_at, schema_version
            FROM sl_registry_retention_plans
            WHERE plan_digest > ?
            ORDER BY plan_digest
            LIMIT ?
            """,
            (after_digest, _VERIFICATION_BATCH_SIZE + 1),
        ).fetchall()
        page = rows[:_VERIFICATION_BATCH_SIZE]
        for row in page:
            budget.check()
            digest = row["plan_digest"]
            if type(digest) is not str or digest <= after_digest:
                raise IntegrityError("stored retention-plan sequence is not canonical")
            _lifecycle_time(row["planned_at"])
            _lifecycle_int(row["schema_version"], lower=1, upper=1)
            after_digest = digest
        if len(rows) <= _VERIFICATION_BATCH_SIZE:
            break

    after_digest = ""
    while True:
        budget.check()
        rows = connection.execute(
            """
            SELECT artifact_digest, planned_at, deleted_at, reason
            FROM sl_registry_retention_tombstones
            WHERE artifact_digest > ?
            ORDER BY artifact_digest
            LIMIT ?
            """,
            (after_digest, _VERIFICATION_BATCH_SIZE + 1),
        ).fetchall()
        page = rows[:_VERIFICATION_BATCH_SIZE]
        for row in page:
            budget.check()
            digest = row["artifact_digest"]
            if type(digest) is not str or digest <= after_digest:
                raise IntegrityError("stored retention-tombstone sequence is not canonical")
            planned_at = _lifecycle_time(row["planned_at"])
            deleted_at = _optional_lifecycle_time(row["deleted_at"])
            if deleted_at is not None and deleted_at < planned_at:
                raise IntegrityError("stored retention-tombstone chronology is inconsistent")
            reason = row["reason"]
            if type(reason) is not str:
                raise IntegrityError("stored retention-tombstone reason is malformed")
            try:
                reason_size = require_utf8_size(reason, "retention-tombstone reason")
            except ValidationError:
                raise IntegrityError("stored retention-tombstone reason is malformed") from None
            if not 1 <= reason_size <= 1_024:
                raise IntegrityError("stored retention-tombstone reason is malformed")
            after_digest = digest
        if len(rows) <= _VERIFICATION_BATCH_SIZE:
            return


def _verify_immutable_record_content(
    connection: sqlite3.Connection,
    budget: _VerificationBudget,
) -> None:
    """Validate exact scalar storage and canonical time across immutable evidence tables."""

    _verify_artifact_content(connection, budget)
    _verify_link_content(connection, budget)
    _verify_retention_scalar_content(connection, budget)


def _verify_schema(
    connection: sqlite3.Connection,
    *,
    verification_timeout_ms: int,
    retention_plan_authenticator: RetentionPlanAuthenticator | None = None,
) -> None:
    timeout_ms = _validate_verification_timeout_ms(verification_timeout_ms)
    budget = _VerificationBudget(monotonic() + (timeout_ms / 1_000))
    handler_installed = False
    primary_error: BaseException | None = None
    try:
        connection.set_progress_handler(
            budget.sqlite_progress,
            _VERIFICATION_PROGRESS_INSTRUCTIONS,
        )
        handler_installed = True
        budget.check()
        for table, expected in _EXPECTED_COLUMNS.items():
            actual = tuple(
                _schema_text(row["name"])
                for row in connection.execute(f'PRAGMA table_info("{table}")').fetchall()
            )
            if actual != expected:
                raise IntegrityError(
                    f"registry table {table} is missing or has an unexpected shape"
                )
        budget.check()
        if _schema_objects(connection) != _expected_schema_objects():
            raise IntegrityError("registry sqlite_master schema differs from canonical DDL")
        budget.check()
        if connection.execute("PRAGMA foreign_key_check").fetchone() is not None:
            raise IntegrityError("SQLite foreign_key_check reported a registry violation")
        budget.check()
        check = connection.execute("PRAGMA quick_check(1)").fetchone()[0]
        if type(check) is not str or check != "ok":
            raise IntegrityError("SQLite quick_check reported registry corruption")
        budget.check()
        _verify_request_content(connection, budget)
        _verify_lifecycle_content(connection, budget)
        _verify_immutable_record_content(connection, budget)
        _verify_retention_plan_content(
            connection,
            budget,
            retention_plan_authenticator,
        )
        budget.check()
    except sqlite3.Error as exc:
        if _is_interrupted(exc) and budget.timed_out:
            mapped = VerificationTimeoutError(
                "registry integrity verification exceeded its configured deadline"
            )
            primary_error = mapped
            raise mapped from None
        primary_error = exc
        raise
    except BaseException as exc:
        primary_error = exc
        raise
    finally:
        if handler_installed:
            try:
                connection.set_progress_handler(None, 0)
            except sqlite3.Error:
                if primary_error is not None:
                    primary_error.add_note("registry verification handler cleanup also failed")
                else:
                    raise IntegrityError("registry verification handler cleanup failed") from None


def _rollback_migration(
    connection: sqlite3.Connection,
    *,
    primary: BaseException,
) -> None:
    """Rollback without replacing the typed primary failure or exposing SQLite text."""

    if connection.in_transaction:
        try:
            connection.execute("ROLLBACK")
        except sqlite3.Error:
            primary.add_note("registry migration rollback also failed closed")


def initialize_database(
    path: Path,
    *,
    busy_timeout_ms: int,
    verification_timeout_ms: int,
    key_verifier: str,
    retention_plan_authenticator: RetentionPlanAuthenticator | None = None,
) -> str:
    """Atomically apply every verified migration without touching legacy ``runs``.

    Applied checksums, gaps, partial schemas, and unknown newer versions fail closed.  All schema
    statements and their ledger record share one ``BEGIN IMMEDIATE`` transaction.
    """

    verifier = _validate_key_verifier(key_verifier)
    verification_timeout = _validate_verification_timeout_ms(verification_timeout_ms)
    connection = open_database(path, busy_timeout_ms=busy_timeout_ms, initialize=True)
    try:
        connection.execute("BEGIN IMMEDIATE")
        connection.execute(_MIGRATION_LEDGER_SQL)
        applied = _read_ledger(connection)
        _verify_ledger(applied)
        first_bootstrap = not applied
        if not applied:
            reserved = connection.execute(
                """
                SELECT name FROM sqlite_master
                WHERE name LIKE 'sl_registry_%' AND name <> ?
                    AND type IN ('table', 'index', 'trigger', 'view')
                LIMIT 1
                """,
                (MIGRATION_TABLE,),
            ).fetchone()
            if reserved is not None:
                raise IntegrityError("unledgered partial registry schema exists")
        for migration in MIGRATIONS:
            if migration.version in applied:
                continue
            if migration.version != len(applied) + 1:
                raise MigrationDriftError("migrations may only advance one version at a time")
            _execute_script(connection, migration.sql)
            connection.execute(
                f"INSERT INTO {MIGRATION_TABLE}(version, name, checksum, applied_at) VALUES(?,?,?,?)",
                (migration.version, migration.name, migration.checksum, _timestamp()),
            )
            applied[migration.version] = (migration.name, migration.checksum)
        if first_bootstrap:
            connection.execute(
                """
                INSERT INTO sl_registry_metadata(singleton, hmac_version, key_verifier, registry_id)
                VALUES(1, 1, ?, ?)
                """,
                (verifier, secrets.token_hex(16)),
            )
        registry_id = verify_database_binding(connection, key_verifier=verifier)
        _verify_schema(
            connection,
            verification_timeout_ms=verification_timeout,
            retention_plan_authenticator=retention_plan_authenticator,
        )
        connection.execute("COMMIT")
        _validate_database_files(path.absolute())
        return registry_id
    except RegistryError as exc:
        _rollback_migration(connection, primary=exc)
        raise
    except sqlite3.Error as exc:
        if _is_busy(exc):
            mapped: RegistryError = BusyError(
                "registry migration remained busy past its configured bound"
            )
        else:
            mapped = MigrationError("registry migration failed and was rolled back")
        _rollback_migration(connection, primary=mapped)
        raise mapped from None
    finally:
        connection.close()


def probe_database(
    path: Path,
    *,
    busy_timeout_ms: int,
    verification_timeout_ms: int,
    key_verifier: str,
    retention_plan_authenticator: RetentionPlanAuthenticator | None = None,
) -> RegistryReadiness:
    """Inspect schema readiness through a read-only connection without repair or creation."""

    try:
        path.lstat()
    except FileNotFoundError:
        return RegistryReadiness(False, None, None, "registry database does not exist")
    except OSError:
        return RegistryReadiness(False, None, None, "integrity_error")
    try:
        verifier = _validate_key_verifier(key_verifier)
        verification_timeout = _validate_verification_timeout_ms(verification_timeout_ms)
    except ValidationError as exc:
        return RegistryReadiness(False, None, None, exc.code)
    try:
        connection = open_database(path, busy_timeout_ms=busy_timeout_ms, readonly=True)
    except RegistryError as exc:
        return RegistryReadiness(False, None, None, exc.code)
    try:
        connection.execute("BEGIN")
        table = connection.execute(
            "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = ?", (MIGRATION_TABLE,)
        ).fetchone()
        if table is None:
            return RegistryReadiness(False, None, None, "migration ledger is missing")
        applied = _read_ledger(connection)
        _verify_ledger(applied)
        if len(applied) != len(MIGRATIONS):
            return RegistryReadiness(False, max(applied, default=0), None, "migrations are pending")
        verify_database_binding(connection, key_verifier=verifier)
        _verify_schema(
            connection,
            verification_timeout_ms=verification_timeout,
            retention_plan_authenticator=retention_plan_authenticator,
        )
        mode = str(connection.execute("PRAGMA journal_mode").fetchone()[0]).lower()
        if mode != "wal":
            return RegistryReadiness(False, LATEST_SCHEMA_VERSION, mode, "WAL mode is required")
        return RegistryReadiness(True, LATEST_SCHEMA_VERSION, mode)
    except RegistryError as exc:
        return RegistryReadiness(False, None, None, exc.code)
    except sqlite3.Error as exc:
        reason = BusyError.code if _is_busy(exc) else IntegrityError.code
        return RegistryReadiness(False, None, None, reason)
    finally:
        connection.close()
