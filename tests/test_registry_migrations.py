"""Fault, drift, and compatibility tests for registry migrations."""

from __future__ import annotations

import hashlib
import hmac
import os
import sqlite3
import subprocess
import sys
from contextlib import closing
from pathlib import Path

import pytest

from quant_platform.tracking import migrations as migrations_module
from quant_platform.tracking.contracts import (
    IntegrityError,
    MigrationDriftError,
    MigrationError,
    RegistryReadiness,
    SubmissionRequest,
    UnsupportedSchemaError,
    VerificationTimeoutError,
    canonical_json,
)
from quant_platform.tracking.migrations import (
    MIGRATION_TABLE,
    MIGRATIONS,
    Migration,
    initialize_database,
    open_database,
    probe_database,
)
from quant_platform.tracking.registry import RunRegistry

SECRET = b"registry-test-secret-material-32-bytes-minimum"
KEY_VERIFIER = hmac.digest(
    SECRET,
    b"signalattice.registry.key-verifier.v1\0bound",
    "sha256",
).hex()
_RETENTION_PLAN_DOMAIN = b"signalattice.registry.retention-plan.v1\0"


def _retention_plan_digest(payload: bytes, registry_id: str, store_id: str) -> str:
    message = (
        _RETENTION_PLAN_DOMAIN
        + registry_id.encode("ascii")
        + b"\0"
        + store_id.encode("ascii")
        + b"\0"
        + payload
    )
    return hmac.digest(SECRET, message, "sha256").hex()


def _authenticate_retention_plan(
    payload: bytes,
    registry_id: str,
    store_id: str,
    signature: str,
) -> bool:
    return hmac.compare_digest(
        signature,
        _retention_plan_digest(payload, registry_id, store_id),
    )


def _initialize(path: Path) -> None:
    initialize_database(
        path,
        busy_timeout_ms=2_000,
        verification_timeout_ms=2_000,
        key_verifier=KEY_VERIFIER,
        retention_plan_authenticator=_authenticate_retention_plan,
    )


def _probe(path: Path) -> RegistryReadiness:
    return probe_database(
        path,
        busy_timeout_ms=2_000,
        verification_timeout_ms=2_000,
        key_verifier=KEY_VERIFIER,
        retention_plan_authenticator=_authenticate_retention_plan,
    )


def _populate_verification_history(
    path: Path,
    *,
    count: int,
    corrupt_last_request_digest: bool = False,
    corrupt_last_retention_payload: bool = False,
) -> None:
    now = "2026-08-08T12:00:00.000000Z"
    created_at = "2026-07-01T12:00:00.000000Z"
    eligible_before = "2026-08-01T12:00:00.000000Z"
    reason = "verification-retention"
    with closing(sqlite3.connect(path)) as connection:
        connection.execute("PRAGMA foreign_keys=ON")
        registry_id = str(
            connection.execute(
                "SELECT registry_id FROM sl_registry_metadata WHERE singleton = 1"
            ).fetchone()[0]
        )
        store_id = "a" * 64
        connection.execute(
            "INSERT OR IGNORE INTO sl_registry_cas_binding(singleton, store_id) VALUES(1, ?)",
            (store_id,),
        )
        for index in range(1, count + 1):
            request = SubmissionRequest("verification", {"index": index})
            request_json = request.canonical_json()
            request_digest = hashlib.sha256(request_json.encode()).hexdigest()
            if corrupt_last_request_digest and index == count:
                request_digest = "0" * 64
            connection.execute(
                """
                INSERT INTO sl_registry_jobs(
                    job_id, kind, request_schema_version, request_json, request_digest,
                    idempotency_digest, state, priority, attempt_count, max_attempts,
                    created_at, updated_at
                ) VALUES(?, ?, 1, ?, ?, ?, 'queued', 0, 0, 3, ?, ?)
                """,
                (
                    f"{index:032x}",
                    request.kind,
                    request_json,
                    request_digest,
                    hashlib.sha256(f"idempotency:{index}".encode()).hexdigest(),
                    now,
                    now,
                ),
            )
            connection.execute(
                """
                INSERT INTO sl_registry_events(
                    job_id, kind, from_state, to_state, occurred_at,
                    attempt, actor, details_json
                ) VALUES(?, 'submitted', NULL, 'queued', ?, 0, 'submitter', ?)
                """,
                (
                    f"{index:032x}",
                    now,
                    canonical_json({"request_digest": request_digest}),
                ),
            )
            artifact_digest = hashlib.sha256(f"artifact:{index}".encode()).hexdigest()
            storage_key = f"objects/{artifact_digest[:2]}/{artifact_digest[2:4]}/{artifact_digest}"
            connection.execute(
                """
                INSERT INTO sl_registry_artifacts(
                    digest, artifact_class, byte_size, media_type,
                    storage_relpath, created_at, pinned
                ) VALUES(?, 'output', 1, 'application/octet-stream', ?, ?, 0)
                """,
                (artifact_digest, storage_key, created_at),
            )
            payload = canonical_json(
                {
                    "candidates": [
                        {
                            "artifact_class": "output",
                            "byte_size": 1,
                            "created_at": created_at,
                            "digest": artifact_digest,
                            "generation_token": hashlib.sha256(
                                f"generation:{index}".encode()
                            ).hexdigest(),
                            "last_changed_at": created_at,
                            "last_changed_ns": 1_782_907_200_000_000_000,
                            "media_type": "application/octet-stream",
                            "pinned": False,
                            "storage_relpath": storage_key,
                        }
                    ],
                    "cas_store_id": store_id,
                    "eligible_before": eligible_before,
                    "planned_at": now,
                    "policy": {
                        "enabled": True,
                        "grace_period_seconds": 7 * 24 * 60 * 60,
                        "max_candidates": 1,
                        "max_total_bytes": 1,
                    },
                    "reason": reason,
                    "registry_id": registry_id,
                    "schema_version": 1,
                }
            )
            if corrupt_last_retention_payload and index == count:
                payload = '{"index": 1}'
            plan_digest = _retention_plan_digest(
                payload.encode("utf-8"),
                registry_id,
                store_id,
            )
            connection.execute(
                """
                INSERT INTO sl_registry_retention_plans(
                    plan_digest, payload_digest, payload_json, registry_id,
                    cas_store_id, planned_at, schema_version
                ) VALUES(?, ?, ?, ?, ?, ?, 1)
                """,
                (
                    plan_digest,
                    hashlib.sha256(payload.encode()).hexdigest(),
                    payload,
                    registry_id,
                    store_id,
                    now,
                ),
            )
            connection.execute(
                """
                INSERT INTO sl_registry_retention_tombstones(
                    artifact_digest, plan_digest, planned_at, deleted_at, reason
                ) VALUES(?, ?, ?, NULL, ?)
                """,
                (artifact_digest, plan_digest, now, reason),
            )
        connection.commit()


def _legacy_database(path: Path) -> tuple[str, tuple[object, ...]]:
    connection = sqlite3.connect(path)
    connection.execute("""
        CREATE TABLE runs (
            run_id TEXT PRIMARY KEY,
            experiment TEXT,
            name TEXT,
            started_at TEXT,
            ended_at TEXT,
            status TEXT,
            git_commit TEXT,
            data_hash TEXT,
            tickers TEXT,
            features TEXT,
            params TEXT,
            metrics TEXT,
            tags TEXT,
            artifacts TEXT
        )
        """)
    row: tuple[object, ...] = (
        "legacy-1",
        "research",
        "baseline",
        "2026-08-01T00:00:00Z",
        "2026-08-01T00:01:00Z",
        "completed",
        "abcdef0",
        "data-v1",
        '["AAA"]',
        '["return"]',
        '{"seed":7}',
        '{"loss":0.1}',
        '{"scope":"legacy"}',
        '["report.json"]',
    )
    connection.execute("INSERT INTO runs VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?)", row)
    connection.commit()
    schema = str(
        connection.execute(
            "SELECT sql FROM sqlite_master WHERE type = 'table' AND name = 'runs'"
        ).fetchone()[0]
    )
    connection.close()
    os.chmod(path, 0o600)
    return schema, row


def test_migration_is_additive_and_preserves_legacy_runs_exactly(tmp_path: Path) -> None:
    path = tmp_path / "legacy.sqlite"
    schema, row = _legacy_database(path)

    _initialize(path)

    connection = sqlite3.connect(path)
    try:
        assert connection.execute("SELECT * FROM runs").fetchone() == row
        assert (
            connection.execute(
                "SELECT sql FROM sqlite_master WHERE type = 'table' AND name = 'runs'"
            ).fetchone()[0]
            == schema
        )
        tables = {
            value[0]
            for value in connection.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table'"
            ).fetchall()
        }
        assert "runs" in tables
        assert {
            MIGRATION_TABLE,
            "sl_registry_jobs",
            "sl_registry_events",
            "sl_registry_runs",
            "sl_registry_artifacts",
            "sl_registry_run_artifacts",
            "sl_registry_cas_binding",
            "sl_registry_retention_plans",
            "sl_registry_retention_tombstones",
        }.issubset(tables)
        run_artifact_columns = {
            value[1]: value[5]
            for value in connection.execute(
                "PRAGMA table_info(sl_registry_run_artifacts)"
            ).fetchall()
        }
        assert run_artifact_columns["sequence"] == 1
        assert (
            connection.execute(
                "SELECT sql FROM sqlite_master WHERE type = 'table' "
                "AND name = 'sl_registry_run_artifacts'"
            )
            .fetchone()[0]
            .find("WITHOUT ROWID")
            == -1
        )
    finally:
        connection.close()


def test_initialize_is_idempotent_and_enforces_connection_pragmas(tmp_path: Path) -> None:
    path = tmp_path / "registry.sqlite"
    _initialize(path)
    _initialize(path)

    connection = open_database(path, busy_timeout_ms=2_000)
    try:
        assert connection.execute("PRAGMA foreign_keys").fetchone()[0] == 1
        assert connection.execute("PRAGMA journal_mode").fetchone()[0] == "wal"
        assert connection.execute("PRAGMA synchronous").fetchone()[0] == 2
        assert connection.execute(f"SELECT count(*) FROM {MIGRATION_TABLE}").fetchone()[0] == 1
    finally:
        connection.close()


def test_registry_instance_identity_is_random_canonical_and_immutable(tmp_path: Path) -> None:
    first = tmp_path / "instance-first.sqlite"
    second = tmp_path / "instance-second.sqlite"
    first_id = initialize_database(
        first,
        busy_timeout_ms=2_000,
        verification_timeout_ms=2_000,
        key_verifier=KEY_VERIFIER,
    )
    repeated_id = initialize_database(
        first,
        busy_timeout_ms=2_000,
        verification_timeout_ms=2_000,
        key_verifier=KEY_VERIFIER,
    )
    second_id = initialize_database(
        second,
        busy_timeout_ms=2_000,
        verification_timeout_ms=2_000,
        key_verifier=KEY_VERIFIER,
    )

    assert first_id == repeated_id
    assert first_id != second_id
    assert len(first_id) == 32
    assert set(first_id) <= set("0123456789abcdef")
    with closing(sqlite3.connect(first)) as connection:
        with pytest.raises(sqlite3.IntegrityError, match="immutable"):
            connection.execute(
                "UPDATE sl_registry_metadata SET registry_id = ? WHERE singleton = 1",
                ("0" * 32,),
            )
        with pytest.raises(sqlite3.IntegrityError, match="immutable"):
            connection.execute("DELETE FROM sl_registry_metadata")


def test_checksum_drift_fails_closed(tmp_path: Path) -> None:
    path = tmp_path / "drift.sqlite"
    with closing(sqlite3.connect(path)) as connection:
        connection.executescript(migrations_module._MIGRATION_LEDGER_SQL)
        connection.execute(
            f"INSERT INTO {MIGRATION_TABLE}(version, name, checksum, applied_at) "
            "VALUES(1, ?, ?, ?)",
            (
                MIGRATIONS[0].name,
                "0" * 64,
                "2026-08-08T12:00:00.000000Z",
            ),
        )
        connection.commit()
    path.chmod(0o600)

    with pytest.raises(MigrationDriftError):
        _initialize(path)
    assert not _probe(path).ready


def test_migration_ledger_is_append_only_and_applied_time_is_canonical(tmp_path: Path) -> None:
    protected = tmp_path / "protected-ledger.sqlite"
    _initialize(protected)
    with closing(sqlite3.connect(protected)) as connection:
        with pytest.raises(sqlite3.IntegrityError, match="immutable"):
            connection.execute(
                f"UPDATE {MIGRATION_TABLE} SET applied_at = applied_at WHERE version = 1"
            )
        with pytest.raises(sqlite3.IntegrityError, match="append-only"):
            connection.execute(f"DELETE FROM {MIGRATION_TABLE} WHERE version = 1")

    malformed = tmp_path / "malformed-ledger-time.sqlite"
    with closing(sqlite3.connect(malformed)) as connection:
        connection.executescript(migrations_module._MIGRATION_LEDGER_SQL)
        connection.execute(
            f"INSERT INTO {MIGRATION_TABLE}(version, name, checksum, applied_at) "
            "VALUES(1, ?, ?, ?)",
            (MIGRATIONS[0].name, MIGRATIONS[0].checksum, "x" * 26 + "Z"),
        )
        connection.executescript(MIGRATIONS[0].sql)
        connection.commit()
    malformed.chmod(0o600)

    verdict = _probe(malformed)
    assert not verdict.ready
    assert verdict.reason == "integrity_error"
    with pytest.raises(IntegrityError, match="timestamp|canonical UTC"):
        _initialize(malformed)


def test_unknown_newer_schema_fails_closed(tmp_path: Path) -> None:
    path = tmp_path / "newer.sqlite"
    _initialize(path)
    connection = sqlite3.connect(path)
    connection.execute(
        f"INSERT INTO {MIGRATION_TABLE}(version, name, checksum, applied_at) VALUES(2,?,?,?)",
        ("future", "f" * 64, "2026-08-08T12:00:00.000000Z"),
    )
    connection.commit()
    connection.close()

    with pytest.raises(UnsupportedSchemaError):
        _initialize(path)
    assert not _probe(path).ready


def test_unledgered_partial_schema_is_not_adopted(tmp_path: Path) -> None:
    path = tmp_path / "partial.sqlite"
    connection = sqlite3.connect(path)
    connection.execute("CREATE TABLE sl_registry_jobs(untrusted TEXT)")
    connection.commit()
    connection.close()
    os.chmod(path, 0o600)

    with pytest.raises(IntegrityError, match="unledgered"):
        _initialize(path)
    connection = sqlite3.connect(path)
    try:
        assert (
            connection.execute(
                "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = ?", (MIGRATION_TABLE,)
            ).fetchone()
            is None
        )
    finally:
        connection.close()


def test_missing_immutability_trigger_is_detected_without_repair(tmp_path: Path) -> None:
    path = tmp_path / "missing-trigger.sqlite"
    _initialize(path)
    connection = sqlite3.connect(path)
    connection.execute("DROP TRIGGER sl_registry_events_no_delete")
    connection.commit()
    connection.close()

    with pytest.raises(IntegrityError, match="canonical DDL"):
        _initialize(path)
    assert not _probe(path).ready


def test_retention_envelope_schema_and_canonical_payload_are_verified(tmp_path: Path) -> None:
    drift = tmp_path / "retention-schema-drift.sqlite"
    _initialize(drift)
    with closing(sqlite3.connect(drift)) as connection:
        connection.execute("DROP TRIGGER sl_registry_retention_plans_no_delete")
        connection.commit()
    with pytest.raises(IntegrityError, match="canonical DDL"):
        _initialize(drift)

    malformed = tmp_path / "retention-payload-drift.sqlite"
    _initialize(malformed)
    payload = '{"schema_version": 1}'
    with closing(sqlite3.connect(malformed)) as connection:
        registry_id = str(
            connection.execute("SELECT registry_id FROM sl_registry_metadata").fetchone()[0]
        )
        connection.execute(
            "INSERT INTO sl_registry_cas_binding(singleton, store_id) VALUES(1, ?)",
            ("a" * 64,),
        )
        connection.execute(
            """
            INSERT INTO sl_registry_retention_plans(
                plan_digest, payload_digest, payload_json, registry_id,
                cas_store_id, planned_at, schema_version
            ) VALUES(?, ?, ?, ?, ?, ?, 1)
            """,
            (
                "b" * 64,
                hashlib.sha256(payload.encode()).hexdigest(),
                payload,
                registry_id,
                "a" * 64,
                "2026-08-08T12:00:00.000000Z",
            ),
        )
        connection.commit()
    assert not _probe(malformed).ready
    with pytest.raises(IntegrityError, match="semantically invalid"):
        _initialize(malformed)


def test_verification_keyset_batches_preserve_complete_history(tmp_path: Path) -> None:
    path = tmp_path / "bounded-verification-history.sqlite"
    _initialize(path)
    history_size = migrations_module._VERIFICATION_BATCH_SIZE * 2 + 3
    _populate_verification_history(path, count=history_size)

    readiness = _probe(path)

    assert readiness.ready
    assert initialize_database(
        path,
        busy_timeout_ms=2_000,
        verification_timeout_ms=2_000,
        key_verifier=KEY_VERIFIER,
        retention_plan_authenticator=_authenticate_retention_plan,
    )


@pytest.mark.parametrize(
    ("corruption", "expected_message"),
    [
        ("request", "request digest"),
        ("retention", "semantically invalid"),
    ],
)
def test_verification_checks_corruption_beyond_first_two_batches(
    tmp_path: Path,
    corruption: str,
    expected_message: str,
) -> None:
    path = tmp_path / f"late-{corruption}-corruption.sqlite"
    _initialize(path)
    history_size = migrations_module._VERIFICATION_BATCH_SIZE * 2 + 1
    _populate_verification_history(
        path,
        count=history_size,
        corrupt_last_request_digest=corruption == "request",
        corrupt_last_retention_payload=corruption == "retention",
    )

    readiness = _probe(path)

    assert not readiness.ready
    assert readiness.reason == "integrity_error"
    with pytest.raises(IntegrityError, match=expected_message):
        _initialize(path)


def test_verification_deadline_interrupt_is_typed_retryable_and_read_only(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "verification-timeout.sqlite"
    _initialize(path)
    before = path.read_bytes()
    monkeypatch.setattr(migrations_module, "_VERIFICATION_PROGRESS_INSTRUCTIONS", 1)
    calls = 0

    def expired_clock() -> float:
        nonlocal calls
        calls += 1
        return 0.0 if calls <= 2 else 1.0

    monkeypatch.setattr(migrations_module, "monotonic", expired_clock)
    with pytest.raises(VerificationTimeoutError) as interrupted:
        initialize_database(
            path,
            busy_timeout_ms=2_000,
            verification_timeout_ms=1,
            key_verifier=KEY_VERIFIER,
        )
    assert interrupted.value.retryable
    assert interrupted.value.code == "registry_verification_timeout"
    assert interrupted.value.__cause__ is None

    calls = 0
    readiness = probe_database(
        path,
        busy_timeout_ms=2_000,
        verification_timeout_ms=1,
        key_verifier=KEY_VERIFIER,
    )
    assert not readiness.ready
    assert readiness.reason == "registry_verification_timeout"
    assert path.read_bytes() == before


def test_verification_handler_cleanup_redacts_sqlite_failure(
    tmp_path: Path,
) -> None:
    path = tmp_path / "verification-cleanup.sqlite"
    _initialize(path)
    connection = open_database(path, busy_timeout_ms=2_000, readonly=True)
    attacker_text = "attacker-controlled-cleanup-detail"

    class CleanupFailingConnection:
        def __init__(self, wrapped: sqlite3.Connection) -> None:
            self._wrapped = wrapped

        def set_progress_handler(self, callback: object, instructions: int) -> None:
            if callback is None:
                raise sqlite3.OperationalError(attacker_text)
            self._wrapped.set_progress_handler(callback, instructions)  # type: ignore[arg-type]

        def __getattr__(self, name: str) -> object:
            return getattr(self._wrapped, name)

    try:
        with pytest.raises(IntegrityError) as captured:
            migrations_module._verify_schema(
                CleanupFailingConnection(connection),  # type: ignore[arg-type]
                verification_timeout_ms=2_000,
            )
        assert captured.value.__cause__ is None
        assert attacker_text not in str(captured.value)
        assert attacker_text not in repr(captured.value)
    finally:
        connection.set_progress_handler(None, 0)
        connection.close()


def test_migration_rollback_failure_preserves_typed_primary_without_raw_detail() -> None:
    attacker_text = "private-rollback-path"

    class RollbackFailingConnection:
        in_transaction = True

        @staticmethod
        def execute(statement: str) -> None:
            assert statement == "ROLLBACK"
            raise sqlite3.OperationalError(attacker_text)

    primary = VerificationTimeoutError("bounded verification timeout")
    migrations_module._rollback_migration(
        RollbackFailingConnection(),  # type: ignore[arg-type]
        primary=primary,
    )

    notes = tuple(getattr(primary, "__notes__", ()))
    assert notes == ("registry migration rollback also failed closed",)
    assert attacker_text not in " ".join(notes)


def test_same_name_noop_trigger_drift_is_detected_exactly(tmp_path: Path) -> None:
    path = tmp_path / "rewritten-trigger.sqlite"
    _initialize(path)
    with closing(sqlite3.connect(path)) as connection:
        connection.execute("DROP TRIGGER sl_registry_events_no_delete")
        connection.execute("""
            CREATE TRIGGER sl_registry_events_no_delete
            BEFORE DELETE ON sl_registry_events
            BEGIN
                SELECT 1;
            END
            """)
        connection.commit()

    with pytest.raises(IntegrityError, match="canonical DDL"):
        _initialize(path)
    assert not _probe(path).ready


def test_unexpected_nonprefixed_trigger_on_registry_table_is_detected(tmp_path: Path) -> None:
    path = tmp_path / "extra-trigger.sqlite"
    _initialize(path)
    with closing(sqlite3.connect(path)) as connection:
        connection.execute("""
            CREATE TRIGGER unexpected_mutator
            AFTER INSERT ON sl_registry_jobs
            BEGIN
                SELECT 1;
            END
            """)
        connection.commit()

    with pytest.raises(IntegrityError, match="canonical DDL"):
        _initialize(path)
    assert not _probe(path).ready


def test_secure_bootstrap_and_operational_open_modes(tmp_path: Path) -> None:
    path = tmp_path / "private.sqlite"
    _initialize(path)

    assert path.stat().st_mode & 0o777 == 0o600
    missing = tmp_path / "not-initialized.sqlite"
    with pytest.raises(IntegrityError, match="does not exist"):
        open_database(missing, busy_timeout_ms=2_000)
    assert not missing.exists()

    path.chmod(0o640)
    with pytest.raises(IntegrityError, match="mode 0600"):
        open_database(path, busy_timeout_ms=2_000)
    assert not _probe(path).ready
    path.chmod(0o600)


def test_database_parent_is_descriptor_walked_and_owner_only(tmp_path: Path) -> None:
    unsafe = tmp_path / "unsafe"
    unsafe.mkdir(mode=0o755)
    with pytest.raises(IntegrityError, match="mode 0700"):
        _initialize(unsafe / "registry.sqlite")
    assert not (unsafe / "registry.sqlite").exists()

    target = tmp_path / "target-directory"
    target.mkdir(mode=0o700)
    redirected = tmp_path / "redirected-directory"
    redirected.symlink_to(target, target_is_directory=True)
    with pytest.raises(IntegrityError, match="parent components"):
        _initialize(redirected / "registry.sqlite")
    assert not (target / "registry.sqlite").exists()


@pytest.mark.skipif(sys.platform != "darwin", reason="macOS extended ACL integration")
def test_database_parent_extended_acl_fails_closed(tmp_path: Path) -> None:
    private = tmp_path / "acl-directory"
    private.mkdir(mode=0o700)
    subprocess.run(
        ["chmod", "+a", "group:everyone allow read", str(private)],
        check=True,
        capture_output=True,
        text=True,
        timeout=5,
    )
    try:
        with pytest.raises(IntegrityError, match="extended ACL"):
            _initialize(private / "registry.sqlite")
        assert not (private / "registry.sqlite").exists()
    finally:
        subprocess.run(
            ["chmod", "-a#", "0", str(private)],
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        )


def test_database_and_companion_symlinks_or_broad_modes_fail_closed(tmp_path: Path) -> None:
    target = tmp_path / "target.sqlite"
    target.write_bytes(b"")
    target.chmod(0o600)
    redirected = tmp_path / "redirected.sqlite"
    redirected.symlink_to(target)

    with pytest.raises(IntegrityError, match="non-symlink"):
        _initialize(redirected)
    assert not _probe(redirected).ready

    hardlink_target = tmp_path / "hardlink-target.sqlite"
    _initialize(hardlink_target)
    os.link(hardlink_target, tmp_path / "second-hardlink.sqlite")
    with pytest.raises(IntegrityError, match="exactly one"):
        open_database(hardlink_target, busy_timeout_ms=2_000)

    path = tmp_path / "companions.sqlite"
    _initialize(path)
    connection = open_database(path, busy_timeout_ms=2_000)
    try:
        connection.execute("BEGIN IMMEDIATE")
        wal = Path(f"{path}-wal")
        shared = Path(f"{path}-shm")
        assert wal.is_file() and shared.is_file()
        shared.chmod(0o644)
        with pytest.raises(IntegrityError, match="shared-memory"):
            open_database(path, busy_timeout_ms=2_000)
        shared.chmod(0o600)
        connection.execute("ROLLBACK")
    finally:
        connection.close()


def test_live_wal_open_may_create_only_private_runtime_sidecars(tmp_path: Path) -> None:
    path = tmp_path / "wal-runtime.sqlite"
    _initialize(path)
    identity = (path.stat().st_dev, path.stat().st_ino)
    wal = Path(f"{path}-wal")
    shared = Path(f"{path}-shm")
    assert not wal.exists() and not shared.exists()

    connection = open_database(path, busy_timeout_ms=2_000)
    try:
        assert wal.is_file() and shared.is_file()
        assert wal.stat().st_mode & 0o777 == 0o600
        assert shared.stat().st_mode & 0o777 == 0o600
        assert (path.stat().st_dev, path.stat().st_ino) == identity
    finally:
        connection.close()


def test_optional_sidecar_disappearance_between_stat_and_open_is_revalidated(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "sidecar-race.sqlite"
    _initialize(path)
    anchor = open_database(path, busy_timeout_ms=2_000)
    original_open = migrations_module.os.open
    simulated = False

    def disappear_once(
        name: str,
        flags: int,
        mode: int = 0o777,
        *,
        dir_fd: int | None = None,
    ) -> int:
        nonlocal simulated
        if name == f"{path.name}-wal" and not simulated:
            simulated = True
            raise FileNotFoundError
        return original_open(name, flags, mode, dir_fd=dir_fd)

    monkeypatch.setattr(migrations_module.os, "open", disappear_once)
    concurrent: sqlite3.Connection | None = None
    try:
        concurrent = open_database(path, busy_timeout_ms=2_000, readonly=True)
        assert simulated
        assert concurrent.execute("SELECT 1").fetchone()[0] == 1
    finally:
        if concurrent is not None:
            concurrent.close()
        anchor.close()


def test_primary_database_disappearance_between_stat_and_open_fails_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "primary-race.sqlite"
    _initialize(path)
    original_open = migrations_module.os.open

    def disappear_primary(
        name: str,
        flags: int,
        mode: int = 0o777,
        *,
        dir_fd: int | None = None,
    ) -> int:
        if name == path.name:
            raise FileNotFoundError
        return original_open(name, flags, mode, dir_fd=dir_fd)

    monkeypatch.setattr(migrations_module.os, "open", disappear_primary)
    with pytest.raises(IntegrityError, match="database could not be opened safely"):
        open_database(path, busy_timeout_ms=2_000, readonly=True)


def test_unlinked_open_sidecar_descriptor_is_treated_as_exact_absence(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "unlinked-sidecar-race.sqlite"
    _initialize(path)
    anchor = open_database(path, busy_timeout_ms=2_000)
    original_open = migrations_module.os.open
    original_fstat = migrations_module.os.fstat
    sidecar_descriptor: int | None = None
    simulated = False

    def capture_sidecar(
        name: str,
        flags: int,
        mode: int = 0o777,
        *,
        dir_fd: int | None = None,
    ) -> int:
        nonlocal sidecar_descriptor
        descriptor = original_open(name, flags, mode, dir_fd=dir_fd)
        if name == f"{path.name}-shm" and sidecar_descriptor is None:
            sidecar_descriptor = descriptor
        return descriptor

    def report_concurrent_unlink(descriptor: int) -> os.stat_result:
        nonlocal simulated
        observed = original_fstat(descriptor)
        if descriptor == sidecar_descriptor and not simulated:
            simulated = True
            fields = list(observed)
            fields[3] = 0
            return os.stat_result(fields)
        return observed

    monkeypatch.setattr(migrations_module.os, "open", capture_sidecar)
    monkeypatch.setattr(migrations_module.os, "fstat", report_concurrent_unlink)
    concurrent: sqlite3.Connection | None = None
    try:
        concurrent = open_database(path, busy_timeout_ms=2_000, readonly=True)
        assert simulated
        assert concurrent.execute("SELECT 1").fetchone()[0] == 1
    finally:
        if concurrent is not None:
            concurrent.close()
        anchor.close()


def test_retention_tombstones_are_monotonic_and_prevent_relink(tmp_path: Path) -> None:
    path = tmp_path / "retention-invariants.sqlite"
    _initialize(path)
    connection = open_database(path, busy_timeout_ms=2_000)
    job_id = "a" * 32
    digest = "b" * 64
    now = "2026-08-08T12:00:00.000000Z"
    later = "2026-08-08T12:00:01.000000Z"
    try:
        connection.execute(
            """
                INSERT INTO sl_registry_jobs(
                    job_id, kind, request_schema_version, request_json, request_digest,
                    idempotency_digest, state, priority, attempt_count, max_attempts,
                    created_at, updated_at, lease_owner, lease_token_digest,
                    lease_expires_at, heartbeat_at
                ) VALUES(?, 'test', 1, '{}', ?, ?, 'running', 0, 1, 1, ?, ?,
                         'test-worker', ?, ?, ?)
                """,
            (job_id, "c" * 64, "d" * 64, now, now, "e" * 64, later, now),
        )
        connection.execute(
            """
            INSERT INTO sl_registry_runs(
                run_id, job_id, attempt, status, evidence_class, schema_version,
                created_at, started_at, ended_at
            ) VALUES('run-1', ?, 1, 'succeeded', 'measured', 1, ?, ?, ?)
            """,
            (job_id, now, now, later),
        )
        connection.execute(
            """
            INSERT INTO sl_registry_artifacts(
                digest, artifact_class, byte_size, media_type, storage_relpath,
                created_at, pinned
            ) VALUES(?, 'report', 1, 'application/json', ?, ?, 0)
            """,
            (digest, f"objects/{digest[:2]}/{digest[2:]}", now),
        )
        registry_id = str(
            connection.execute(
                "SELECT registry_id FROM sl_registry_metadata WHERE singleton = 1"
            ).fetchone()[0]
        )
        store_id = "f" * 64
        payload = "{}"
        plan_digest = "e" * 64
        connection.execute(
            "INSERT INTO sl_registry_cas_binding(singleton, store_id) VALUES(1, ?)",
            (store_id,),
        )
        connection.execute(
            """
            INSERT INTO sl_registry_retention_plans(
                plan_digest, payload_digest, payload_json, registry_id,
                cas_store_id, planned_at, schema_version
            ) VALUES(?, ?, ?, ?, ?, ?, 1)
            """,
            (
                plan_digest,
                hashlib.sha256(payload.encode()).hexdigest(),
                payload,
                registry_id,
                store_id,
                now,
            ),
        )
        with pytest.raises(sqlite3.IntegrityError, match="FOREIGN KEY"):
            connection.execute(
                """
                INSERT INTO sl_registry_retention_tombstones(
                    artifact_digest, plan_digest, planned_at, reason
                ) VALUES(?, ?, ?, 'unregistered digest')
                """,
                ("d" * 64, plan_digest, now),
            )
        with pytest.raises(sqlite3.IntegrityError, match="must begin pending"):
            connection.execute(
                """
                INSERT INTO sl_registry_retention_tombstones(
                    artifact_digest, plan_digest, planned_at, deleted_at, reason
                ) VALUES(?, ?, ?, ?, 'invalid finalized insert')
                """,
                (digest, plan_digest, now, later),
            )
        connection.execute(
            """
            INSERT INTO sl_registry_retention_tombstones(
                artifact_digest, plan_digest, planned_at, reason
            ) VALUES(?, ?, ?, 'bounded retention')
            """,
            (digest, plan_digest, now),
        )

        with pytest.raises(sqlite3.IntegrityError, match="cannot be linked"):
            connection.execute(
                """
                INSERT INTO sl_registry_run_artifacts(
                    run_id, role, artifact_digest, linked_at
                ) VALUES('run-1', 'report', ?, ?)
                """,
                (digest, now),
            )
        with pytest.raises(sqlite3.IntegrityError, match="only deletion finalization"):
            connection.execute(
                "UPDATE sl_registry_retention_tombstones SET reason = 'rewritten' "
                "WHERE artifact_digest = ?",
                (digest,),
            )
        connection.execute(
            "UPDATE sl_registry_retention_tombstones SET deleted_at = ? "
            "WHERE artifact_digest = ?",
            (later, digest),
        )
        with pytest.raises(sqlite3.IntegrityError, match="only deletion finalization"):
            connection.execute(
                "UPDATE sl_registry_retention_tombstones SET deleted_at = ? "
                "WHERE artifact_digest = ?",
                (later, digest),
            )
        with pytest.raises(sqlite3.IntegrityError, match="append-only"):
            connection.execute(
                "DELETE FROM sl_registry_retention_tombstones WHERE artifact_digest = ?",
                (digest,),
            )
        with pytest.raises(sqlite3.IntegrityError, match="immutable"):
            connection.execute(
                "UPDATE sl_registry_retention_plans SET payload_digest = ? WHERE plan_digest = ?",
                ("0" * 64, plan_digest),
            )
        with pytest.raises(sqlite3.IntegrityError, match="append-only"):
            connection.execute(
                "DELETE FROM sl_registry_retention_plans WHERE plan_digest = ?",
                (plan_digest,),
            )
        with pytest.raises(sqlite3.IntegrityError, match="immutable"):
            connection.execute(
                "UPDATE sl_registry_cas_binding SET store_id = ? WHERE singleton = 1",
                ("0" * 64,),
            )
    finally:
        connection.close()


def test_foreign_key_corruption_fails_readiness_and_startup(tmp_path: Path) -> None:
    path = tmp_path / "foreign-key-corruption.sqlite"
    _initialize(path)
    connection = sqlite3.connect(path)
    connection.execute(
        """
        INSERT INTO sl_registry_events(
            job_id, kind, from_state, to_state, occurred_at, attempt, actor, details_json
        ) VALUES(?, 'submitted', NULL, 'queued', ?, 0, 'test', '{}')
        """,
        ("0" * 32, "2026-08-08T12:00:00.000000Z"),
    )
    connection.commit()
    connection.close()

    assert not _probe(path).ready
    with pytest.raises(IntegrityError, match="foreign_key_check"):
        _initialize(path)


def test_failed_forward_migration_rolls_back_schema_and_ledger(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "rollback.sqlite"
    _initialize(path)
    broken = Migration(2, "broken", "CREATE TABLE sl_registry_partial(value TEXT); INVALID SQL;")
    monkeypatch.setattr(
        "quant_platform.tracking.migrations.MIGRATIONS",
        (MIGRATIONS[0], broken),
    )

    with pytest.raises(MigrationError):
        _initialize(path)
    connection = sqlite3.connect(path)
    try:
        assert connection.execute(f"SELECT max(version) FROM {MIGRATION_TABLE}").fetchone()[0] == 1
        assert (
            connection.execute(
                "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = 'sl_registry_partial'"
            ).fetchone()
            is None
        )
    finally:
        connection.close()


def test_corruption_returns_not_ready_and_initialize_fails(tmp_path: Path) -> None:
    path = tmp_path / "corrupt.sqlite"
    path.write_bytes(b"not a sqlite database")

    assert not _probe(path).ready
    with pytest.raises(IntegrityError):
        _initialize(path)


def test_request_digest_corruption_fails_readiness_startup_and_claim(tmp_path: Path) -> None:
    path = tmp_path / "request-corruption.sqlite"
    registry = RunRegistry(path, digest_secret=SECRET)
    registry.initialize()
    registry.submit(
        SubmissionRequest("forecast", {"seed": 1}),
        idempotency_key="request-corruption-key",
    )
    with closing(sqlite3.connect(path)) as connection:
        connection.execute(
            "UPDATE sl_registry_jobs SET request_json = replace(request_json, '\"seed\":1', "
            "'\"seed\":2')"
        )
        connection.commit()

    assert not registry.probe_readiness().ready
    with pytest.raises(IntegrityError, match="request digest"):
        registry.initialize()
    with pytest.raises(IntegrityError, match="request digest"):
        registry.claim(worker_id="worker-corruption", lease_seconds=10)


def test_readiness_probe_does_not_create_or_migrate(tmp_path: Path) -> None:
    missing = tmp_path / "missing.sqlite"
    verdict = _probe(missing)
    assert not verdict.ready
    assert not missing.exists()

    legacy = tmp_path / "legacy-only.sqlite"
    _legacy_database(legacy)
    before = legacy.stat().st_mtime_ns
    verdict = RunRegistry(legacy, digest_secret=SECRET).probe_readiness()
    assert not verdict.ready
    assert legacy.stat().st_mtime_ns == before
    connection = sqlite3.connect(legacy)
    try:
        assert (
            connection.execute(
                "SELECT 1 FROM sqlite_master WHERE name LIKE 'sl_registry_%'"
            ).fetchone()
            is None
        )
    finally:
        connection.close()


def _submitted_registry(tmp_path: Path, name: str) -> tuple[RunRegistry, str]:
    registry = RunRegistry(tmp_path / f"{name}.sqlite", digest_secret=SECRET)
    registry.initialize()
    job = registry.submit(
        SubmissionRequest("lifecycle-audit", {"case": name}, max_attempts=3),
        idempotency_key=f"lifecycle-audit-{name}",
    )
    return registry, job.job_id


@pytest.mark.parametrize(
    "mutation",
    ["forged-running", "forged-attempt", "forged-cancel", "broken-event-chain"],
)
def test_readiness_replays_nonterminal_lifecycle_and_rejects_schema_valid_tampering(
    tmp_path: Path,
    mutation: str,
) -> None:
    registry, job_id = _submitted_registry(tmp_path, mutation)
    with closing(sqlite3.connect(registry.path)) as connection:
        created_at = str(
            connection.execute(
                "SELECT created_at FROM sl_registry_jobs WHERE job_id = ?", (job_id,)
            ).fetchone()[0]
        )
        if mutation == "forged-running":
            connection.execute(
                """
                UPDATE sl_registry_jobs
                SET state = 'running', attempt_count = 1, lease_owner = 'forged-worker',
                    lease_token_digest = ?, heartbeat_at = ?, lease_expires_at = ?
                WHERE job_id = ?
                """,
                ("a" * 64, created_at, "2099-01-01T00:00:00.000000Z", job_id),
            )
        elif mutation == "forged-attempt":
            connection.execute(
                "UPDATE sl_registry_jobs SET attempt_count = 1 WHERE job_id = ?",
                (job_id,),
            )
        elif mutation == "forged-cancel":
            connection.execute(
                """
                UPDATE sl_registry_jobs
                SET cancel_requested_at = created_at, cancel_reason = 'operator_request'
                WHERE job_id = ?
                """,
                (job_id,),
            )
        else:
            connection.execute(
                """
                INSERT INTO sl_registry_events(
                    job_id, kind, from_state, to_state, occurred_at,
                    attempt, actor, details_json
                ) VALUES(?, 'retried', 'running', 'queued', ?, 0, 'forged-worker', ?)
                """,
                (
                    job_id,
                    created_at,
                    canonical_json({"failure_code": "execution_error"}),
                ),
            )
        connection.commit()

    verdict = registry.probe_readiness()
    assert not verdict.ready
    assert verdict.reason == "integrity_error"
    with pytest.raises(IntegrityError, match="lifecycle|attempt|cancellation|snapshot"):
        registry.initialize()


def test_readiness_rejects_real_attempts_and_noncanonical_lifecycle_timestamps(
    tmp_path: Path,
) -> None:
    path = tmp_path / "hostile-lifecycle-scalars.sqlite"
    _initialize(path)
    request = SubmissionRequest("scalar-audit", {})
    request_json = request.canonical_json()
    request_digest = hashlib.sha256(request_json.encode()).hexdigest()
    bad_time = "x" * 26 + "Z"
    with closing(sqlite3.connect(path)) as connection:
        connection.execute(
            """
            INSERT INTO sl_registry_jobs(
                job_id, kind, request_schema_version, request_json, request_digest,
                idempotency_digest, state, priority, attempt_count, max_attempts,
                created_at, updated_at
            ) VALUES(?, ?, 1, ?, ?, ?, 'queued', 0, 0.5, 3, ?, ?)
            """,
            (
                "a" * 32,
                request.kind,
                request_json,
                request_digest,
                "b" * 64,
                bad_time,
                bad_time,
            ),
        )
        connection.execute(
            """
            INSERT INTO sl_registry_events(
                job_id, kind, from_state, to_state, occurred_at,
                attempt, actor, details_json
            ) VALUES(?, 'submitted', NULL, 'queued', ?, 0.5, 'submitter', ?)
            """,
            (
                "a" * 32,
                bad_time,
                canonical_json({"request_digest": request_digest}),
            ),
        )
        connection.commit()

    verdict = _probe(path)
    assert not verdict.ready
    assert verdict.reason == "integrity_error"


def test_readiness_maps_deep_duplicate_and_nonfinite_request_json_to_integrity_error(
    tmp_path: Path,
) -> None:
    for name, request_json in {
        "deep": "[" * 10_000 + "]" * 10_000,
        "duplicate": '{"kind":"x","kind":"x"}',
        "nonfinite": '{"kind":NaN}',
    }.items():
        registry, job_id = _submitted_registry(tmp_path, f"hostile-json-{name}")
        with closing(sqlite3.connect(registry.path)) as connection:
            connection.execute(
                "UPDATE sl_registry_jobs SET request_json = ?, request_digest = ? "
                "WHERE job_id = ?",
                (request_json, hashlib.sha256(request_json.encode()).hexdigest(), job_id),
            )
            connection.commit()
        verdict = registry.probe_readiness()
        assert not verdict.ready
        assert verdict.reason == "integrity_error"


def test_run_insert_requires_current_running_attempt_and_heartbeat_event_is_unsupported(
    tmp_path: Path,
) -> None:
    registry, job_id = _submitted_registry(tmp_path, "run-trigger")
    with closing(sqlite3.connect(registry.path)) as connection:
        created_at = str(
            connection.execute(
                "SELECT created_at FROM sl_registry_jobs WHERE job_id = ?", (job_id,)
            ).fetchone()[0]
        )
        with pytest.raises(sqlite3.IntegrityError, match="current running attempt"):
            connection.execute(
                """
                INSERT INTO sl_registry_runs(
                    run_id, job_id, attempt, status, evidence_class, schema_version,
                    created_at, started_at, ended_at
                ) VALUES('forged-run', ?, 1, 'succeeded', 'measured', 1, ?, ?, ?)
                """,
                (job_id, created_at, created_at, created_at),
            )
        with pytest.raises(sqlite3.IntegrityError, match="CHECK constraint"):
            connection.execute(
                """
                INSERT INTO sl_registry_events(
                    job_id, kind, from_state, to_state, occurred_at,
                    attempt, actor, details_json
                ) VALUES(?, 'heartbeat', 'queued', 'queued', ?, 0, 'forged-worker', '{}')
                """,
                (job_id, created_at),
            )
