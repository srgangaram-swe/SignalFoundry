"""Integration and adversarial tests for framework-neutral registry read ports."""

from __future__ import annotations

import hashlib
import os
import sqlite3
import time
from contextlib import closing
from dataclasses import FrozenInstanceError, fields
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from quant_platform.tracking import migrations as migrations_module
from quant_platform.tracking.cas import ArtifactStore, PublishedArtifact
from quant_platform.tracking.contracts import (
    ArtifactClass,
    ArtifactCursor,
    ArtifactLink,
    EvidenceClass,
    IntegrityError,
    InvalidCursorError,
    NotFoundError,
    RegistryLimits,
    RunCursor,
    RunSnapshot,
    RunStatus,
    SubmissionRequest,
    TerminalRunRequest,
    ValidationError,
)
from quant_platform.tracking.read_ports import (
    ArtifactPageRequest,
    ArtifactView,
    LegacyRunSnapshot,
    ReadPortLimits,
    ReadTimeoutError,
    RegistryReadPorts,
    RunPageRequest,
    RunProvenance,
    RunQuery,
)
from quant_platform.tracking.registry import RunRegistry

NOW = datetime(2026, 8, 8, 12, 0, tzinfo=UTC)
SECRET = b"read-port-test-secret-material-32-bytes"


class _MutableClock:
    """Test-only clock controlled at operation boundaries without public time injection."""

    def __init__(self, current: datetime) -> None:
        self.current = current

    def now(self) -> datetime:
        return self.current


class _ExplosiveBool:
    """Wrong-type boundary input that proves validation never asks for truthiness."""

    def __bool__(self) -> bool:
        raise AssertionError("public validation executed attacker-controlled __bool__")


@pytest.fixture
def system(tmp_path: Path) -> tuple[RunRegistry, ArtifactStore, RegistryReadPorts]:
    root = tmp_path.resolve()
    store = ArtifactStore(root / "cas", max_artifact_bytes=1024 * 1024)
    store.initialize()
    registry = RunRegistry(
        root / "registry.sqlite",
        digest_secret=SECRET,
        limits=RegistryLimits(busy_timeout_ms=2_000),
        clock=_MutableClock(NOW),
        artifact_verifier=store,
    )
    registry.initialize()
    return registry, store, RegistryReadPorts(registry, store)


def _publish(
    registry: RunRegistry,
    store: ArtifactStore,
    root: Path,
    *,
    name: str,
    payload: bytes,
    media_type: str = "application/json",
) -> PublishedArtifact:
    source = root / name
    source.write_bytes(payload)
    published = store.publish(source)
    registry.register_artifact(
        published,
        artifact_class=ArtifactClass.METADATA,
        media_type=media_type,
    )
    return published


def _complete_run(
    registry: RunRegistry,
    run_id: str,
    *,
    offset: int,
    artifact_links: tuple[ArtifactLink, ...] = (),
) -> None:
    started_at = NOW + timedelta(minutes=offset)
    clock = registry._clock
    assert isinstance(clock, _MutableClock)
    clock.current = started_at
    registry.submit(
        SubmissionRequest("forecast", {"seed": offset}),
        idempotency_key=f"read-port-idempotency-{offset}",
    )
    claimed = registry.claim(worker_id=f"worker-{offset}", lease_seconds=60)
    assert claimed is not None
    clock.current = started_at + timedelta(seconds=5)
    registry.complete(
        claimed.lease,
        run=TerminalRunRequest(
            run_id,
            EvidenceClass.BACKTESTED,
            started_at=started_at,
            source_commit="abcdef0123456789",
            data_identity="sha256:" + "a" * 64,
            limitation_summary="Historical synthetic integration evidence only.",
        ),
        artifact_links=tuple(sorted(artifact_links)),
    )


def _create_legacy_table(path: Path) -> None:
    with closing(sqlite3.connect(path)) as connection, connection:
        connection.execute("""
            CREATE TABLE IF NOT EXISTS runs (
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


def _insert_legacy(
    path: Path,
    run_id: str,
    *,
    offset: int,
    status: str = "completed",
    params: str = '{"window":20}',
    metrics: str = '{"sharpe":0.5}',
    tags: str = '{"scope":"test"}',
    artifacts: str = '["../sensitive/report.json"]',
) -> None:
    _create_legacy_table(path)
    started = NOW + timedelta(hours=offset)
    ended = None if status == "running" else started + timedelta(seconds=2)
    with closing(sqlite3.connect(path)) as connection, connection:
        connection.execute(
            """
            INSERT INTO runs(
                run_id, experiment, name, started_at, ended_at, status, git_commit,
                data_hash, tickers, features, params, metrics, tags, artifacts
            ) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?)
            """,
            (
                run_id,
                "private-experiment-name",
                "private-run-name",
                started.isoformat(),
                None if ended is None else ended.isoformat(),
                status,
                "not-a-safe-commit-value",
                "not-a-safe-data-identity",
                '["AAPL"]',
                '["return_1d"]',
                params,
                metrics,
                tags,
                artifacts,
            ),
        )


def _tamper_cursor(token: str) -> str:
    version, body, mac = token.split(".")
    replacement = "A" if mac[0] != "A" else "B"
    return f"{version}.{body}.{replacement}{mac[1:]}"


def test_probe_is_read_only_and_never_bootstraps_missing_state(tmp_path: Path) -> None:
    root = tmp_path.resolve()
    database = root / "missing" / "registry.sqlite"
    cas_root = root / "missing-cas"
    registry = RunRegistry(database, digest_secret=SECRET)
    ports = RegistryReadPorts(registry, ArtifactStore(cas_root))

    readiness = ports.probe_readiness()

    assert not readiness.ready
    assert readiness.reason == "registry database does not exist"
    assert not database.exists()
    assert not database.parent.exists()
    assert not cas_root.exists()
    with pytest.raises(IntegrityError):
        ports.get_run("missing-run")
    assert not database.exists()
    assert not cas_root.exists()


@pytest.mark.parametrize("wrong", [0, (), _ExplosiveBool()])
def test_optional_read_contracts_reject_falsey_and_hostile_wrong_types(
    system: tuple[RunRegistry, ArtifactStore, RegistryReadPorts],
    wrong: object,
) -> None:
    registry, store, ports = system

    with pytest.raises(ValidationError, match="limits"):
        RegistryReadPorts(registry, store, limits=wrong)  # type: ignore[arg-type]
    with pytest.raises(ValidationError, match="query"):
        ports.list_runs(query=wrong)  # type: ignore[arg-type]
    with pytest.raises(ValidationError, match="page"):
        ports.list_runs(page=wrong)  # type: ignore[arg-type]
    with pytest.raises(ValidationError, match="page"):
        ports.list_run_artifacts("missing-run", page=wrong)  # type: ignore[arg-type]


def test_probe_delegates_the_registry_verification_deadline(
    system: tuple[RunRegistry, ArtifactStore, RegistryReadPorts],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    registry, store, _ = system
    bounded_registry = RunRegistry(
        registry.path,
        digest_secret=SECRET,
        limits=RegistryLimits(verification_timeout_ms=1),
        artifact_verifier=store,
    )
    ports = RegistryReadPorts(bounded_registry, store)
    monkeypatch.setattr(migrations_module, "_VERIFICATION_PROGRESS_INSTRUCTIONS", 1)
    calls = 0

    def expired_clock() -> float:
        nonlocal calls
        calls += 1
        return 0.0 if calls <= 2 else 1.0

    monkeypatch.setattr(migrations_module, "monotonic", expired_clock)

    readiness = ports.probe_readiness()

    assert not readiness.ready
    assert readiness.reason == "registry_verification_timeout"


def test_real_registry_and_cas_round_trip_is_typed_frozen_and_path_free(
    system: tuple[RunRegistry, ArtifactStore, RegistryReadPorts],
    tmp_path: Path,
) -> None:
    registry, store, ports = system
    payload = b'{"schema_version":1,"verdict":"not_ready"}'
    published = _publish(
        registry,
        store,
        tmp_path.resolve(),
        name="manifest.json",
        payload=payload,
    )
    _complete_run(
        registry,
        "run-cas-roundtrip",
        offset=0,
        artifact_links=(ArtifactLink("manifest", published.digest),),
    )

    run = ports.get_run("run-cas-roundtrip")
    metadata = ports.get_artifact(published.digest)
    links = ports.list_run_artifacts("run-cas-roundtrip")

    assert isinstance(run, RunSnapshot)
    assert run.status is RunStatus.SUCCEEDED
    assert run.limitation_summary == "Limitations recorded."
    assert metadata == links.items[0].artifact
    assert metadata.artifact_id == hashlib.sha256(payload).hexdigest()
    assert "storage_relpath" not in {item.name for item in fields(ArtifactView)}
    assert "storage_key" not in repr(metadata)
    assert str(store.root) not in repr((run, metadata, links))
    assert (
        ports.read_verified_manifest(published.digest, "application/json", max_bytes=len(payload))
        == payload
    )
    with pytest.raises(FrozenInstanceError):
        metadata.byte_size = 0  # type: ignore[misc]
    assert ports.probe_readiness().ready


def test_every_database_read_rejects_a_registry_secret_mismatch(
    system: tuple[RunRegistry, ArtifactStore, RegistryReadPorts],
    tmp_path: Path,
) -> None:
    registry, store, _ = system
    published = _publish(
        registry,
        store,
        tmp_path.resolve(),
        name="secret-bound.json",
        payload=b"{}",
    )
    _complete_run(
        registry,
        "secret-bound-run",
        offset=0,
        artifact_links=(ArtifactLink("manifest", published.digest),),
    )
    wrong_registry = RunRegistry(
        registry.path,
        digest_secret=b"different-read-port-secret-material",
        artifact_verifier=store,
    )
    ports = RegistryReadPorts(wrong_registry, store)

    assert not ports.probe_readiness().ready
    operations = (
        lambda: ports.get_run("secret-bound-run"),
        ports.list_runs,
        lambda: ports.get_artifact(published.digest),
        lambda: ports.list_run_artifacts("secret-bound-run"),
        lambda: ports.read_verified_manifest(
            published.digest,
            "application/json",
            2,
        ),
    )
    for operation in operations:
        with pytest.raises(IntegrityError, match="HMAC authority"):
            operation()


def test_run_cursor_is_filter_bound_tamper_evident_and_snapshot_stable(
    system: tuple[RunRegistry, ArtifactStore, RegistryReadPorts],
) -> None:
    registry, _, ports = system
    _complete_run(registry, "registry-1", offset=0)
    _complete_run(registry, "registry-2", offset=1)
    _insert_legacy(registry.path, "legacy-1", offset=1)
    _insert_legacy(registry.path, "legacy-2", offset=2)

    first = ports.list_runs(page=RunPageRequest(page_size=2))
    assert first.next_cursor is not None
    _complete_run(registry, "registry-after-snapshot", offset=2)
    _insert_legacy(registry.path, "legacy-after-snapshot", offset=3)
    second = ports.list_runs(page=RunPageRequest(page_size=2, cursor=first.next_cursor))

    ids = {item.run_id for item in first.items + second.items}
    assert ids == {"registry-1", "registry-2", "legacy-1", "legacy-2"}
    assert second.next_cursor is None
    with pytest.raises(InvalidCursorError, match="authentication"):
        ports.list_runs(
            page=RunPageRequest(
                page_size=2,
                cursor=RunCursor(_tamper_cursor(first.next_cursor.token)),
            )
        )
    with pytest.raises(InvalidCursorError, match="filter"):
        ports.list_runs(
            query=RunQuery(provenance=RunProvenance.REGISTRY_VERIFIED),
            page=RunPageRequest(page_size=2, cursor=first.next_cursor),
        )


def test_legacy_cursor_fails_closed_if_a_snapshotted_row_changes(
    system: tuple[RunRegistry, ArtifactStore, RegistryReadPorts],
) -> None:
    registry, _, ports = system
    _insert_legacy(registry.path, "legacy-before", offset=1)
    _insert_legacy(registry.path, "legacy-mutated", offset=2)
    query = RunQuery(provenance=RunProvenance.LEGACY_UNVERIFIED)
    first = ports.list_runs(query, RunPageRequest(page_size=1))
    assert first.next_cursor is not None
    with closing(sqlite3.connect(registry.path)) as connection, connection:
        connection.execute(
            "UPDATE runs SET metrics = ? WHERE run_id = ?",
            ('{"sharpe":0.9}', "legacy-mutated"),
        )

    with pytest.raises(InvalidCursorError, match="changed since"):
        ports.list_runs(query, RunPageRequest(page_size=1, cursor=first.next_cursor))


def test_run_filters_are_typed_and_legacy_active_state_is_not_fabricated(
    system: tuple[RunRegistry, ArtifactStore, RegistryReadPorts],
) -> None:
    registry, _, ports = system
    _complete_run(registry, "verified-run", offset=0)
    _insert_legacy(registry.path, "terminal-legacy", offset=2)
    _insert_legacy(registry.path, "active-legacy", offset=1, status="running")

    verified = ports.list_runs(RunQuery(RunStatus.SUCCEEDED, RunProvenance.REGISTRY_VERIFIED))
    legacy_terminal = ports.list_runs(
        RunQuery(RunStatus.SUCCEEDED, RunProvenance.LEGACY_UNVERIFIED)
    )
    active = ports.get_run("active-legacy")

    assert [item.run_id for item in verified.items] == ["verified-run"]
    assert [item.run_id for item in legacy_terminal.items] == ["terminal-legacy"]
    assert isinstance(active, LegacyRunSnapshot)
    assert active.provenance is RunProvenance.LEGACY_UNVERIFIED
    assert active.reported_terminal_status is None
    assert active.ended_at is None
    assert "no terminal evidence" in active.limitation_summary.lower()
    with pytest.raises(ValidationError):
        ports.list_runs(page=RunPageRequest(page_size=101))


def test_legacy_projection_validates_json_and_redacts_arbitrary_sensitive_fields(
    system: tuple[RunRegistry, ArtifactStore, RegistryReadPorts],
) -> None:
    registry, _, ports = system
    sensitive_value = "synthetic-provider-credential"
    _insert_legacy(
        registry.path,
        "legacy-redacted",
        offset=1,
        params='{"credential":"' + sensitive_value + '"}',
        tags='{"relative_path":"../sensitive/model"}',
        artifacts='["../sensitive/report.json"]',
    )

    observed = ports.get_run("legacy-redacted")

    assert isinstance(observed, LegacyRunSnapshot)
    assert sensitive_value not in repr(observed)
    assert "../sensitive" not in repr(observed)
    assert observed.source_commit is None
    assert observed.data_identity is None
    assert not hasattr(observed, "params")
    assert not hasattr(observed, "metrics")
    assert not hasattr(observed, "tags")
    assert not hasattr(observed, "artifacts")


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("params", "{not-json", "malformed"),
        ("params", '{"window":10,"window":20}', "malformed"),
        ("params", "[" * 5_000 + "]" * 5_000, "malformed"),
        ("metrics", '{"loss":NaN}', "non-finite"),
        ("metrics", '{"loss":"secret"}', "numeric"),
    ],
)
def test_malformed_or_nonfinite_legacy_json_fails_closed(
    system: tuple[RunRegistry, ArtifactStore, RegistryReadPorts],
    field: str,
    value: str,
    message: str,
) -> None:
    registry, _, ports = system
    kwargs = {field: value}
    _insert_legacy(registry.path, "legacy-malformed", offset=1, **kwargs)

    with pytest.raises(IntegrityError, match=message):
        ports.get_run("legacy-malformed")
    with pytest.raises(IntegrityError, match=message):
        ports.list_runs()


def test_oversized_legacy_json_fails_before_projection(
    system: tuple[RunRegistry, ArtifactStore, RegistryReadPorts],
) -> None:
    registry, store, _ = system
    ports = RegistryReadPorts(
        registry,
        store,
        limits=ReadPortLimits(max_legacy_json_bytes=1_024),
    )
    _insert_legacy(
        registry.path,
        "legacy-oversized",
        offset=1,
        params='{"blob":"' + "x" * 2_000 + '"}',
    )

    with pytest.raises(IntegrityError, match="byte bound"):
        ports.get_run("legacy-oversized")


@pytest.mark.parametrize(
    ("field", "storage_type"),
    [
        ("run_id", "text"),
        ("run_id", "blob"),
        ("status", "text"),
        ("status", "blob"),
    ],
)
def test_legacy_identity_and_status_are_bounded_before_sql_operations(
    system: tuple[RunRegistry, ArtifactStore, RegistryReadPorts],
    field: str,
    storage_type: str,
) -> None:
    registry, _, ports = system
    _create_legacy_table(registry.path)
    raw = b"x" * (2 * 1024 * 1024)
    hostile: object = raw.decode("ascii") if storage_type == "text" else raw
    run_id: object = hostile if field == "run_id" else "bounded-legacy-id"
    status: object = hostile if field == "status" else "completed"
    with closing(sqlite3.connect(registry.path)) as connection, connection:
        connection.execute(
            """
            INSERT INTO runs(
                run_id, experiment, name, started_at, ended_at, status, git_commit,
                data_hash, tickers, features, params, metrics, tags, artifacts
            ) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?)
            """,
            (
                run_id,
                "experiment",
                "run",
                NOW.isoformat(),
                (NOW + timedelta(seconds=1)).isoformat(),
                status,
                "abcdef0",
                "a" * 64,
                "[]",
                "[]",
                "{}",
                "{}",
                "{}",
                "[]",
            ),
        )

    for query in (
        RunQuery(provenance=RunProvenance.LEGACY_UNVERIFIED),
        RunQuery(
            status=RunStatus.SUCCEEDED,
            provenance=RunProvenance.LEGACY_UNVERIFIED,
        ),
    ):
        with pytest.raises(IntegrityError, match="legacy scalar field"):
            ports.list_runs(query=query)
    with pytest.raises(IntegrityError, match="legacy scalar field"):
        ports.get_run("bounded-legacy-id")


def test_duplicate_identity_across_sources_fails_closed(
    system: tuple[RunRegistry, ArtifactStore, RegistryReadPorts],
) -> None:
    registry, _, ports = system
    _complete_run(registry, "duplicate-run", offset=0)
    _insert_legacy(registry.path, "duplicate-run", offset=1)

    with pytest.raises(IntegrityError, match="ambiguous"):
        ports.get_run("duplicate-run")
    with pytest.raises(IntegrityError, match="ambiguous"):
        ports.list_runs()


def test_duplicate_legacy_identity_fails_get_and_list_without_selecting_a_row(
    system: tuple[RunRegistry, ArtifactStore, RegistryReadPorts],
) -> None:
    registry, _, ports = system
    with closing(sqlite3.connect(registry.path)) as connection, connection:
        connection.execute("""
            CREATE TABLE runs (
                run_id TEXT,
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
    _insert_legacy(registry.path, "ambiguous-legacy", offset=1)
    _insert_legacy(registry.path, "ambiguous-legacy", offset=2)

    with pytest.raises(IntegrityError, match="ambiguous"):
        ports.get_run("ambiguous-legacy")
    with pytest.raises(IntegrityError, match="ambiguous"):
        ports.list_runs()


def test_legacy_negative_rowid_cannot_be_omitted_from_a_list_snapshot(
    system: tuple[RunRegistry, ArtifactStore, RegistryReadPorts],
) -> None:
    registry, _, ports = system
    _insert_legacy(registry.path, "legacy-negative", offset=1)
    _insert_legacy(registry.path, "legacy-positive", offset=2)
    with closing(sqlite3.connect(registry.path)) as connection, connection:
        connection.execute(
            "UPDATE runs SET rowid = -1 WHERE run_id = ?",
            ("legacy-negative",),
        )

    with pytest.raises(IntegrityError, match="projected safely"):
        ports.get_run("legacy-negative")
    with pytest.raises(IntegrityError, match="sequence"):
        ports.list_runs()


def test_registry_negative_sequence_cannot_be_omitted_from_a_list_snapshot(
    system: tuple[RunRegistry, ArtifactStore, RegistryReadPorts],
) -> None:
    registry, _, ports = system
    _complete_run(registry, "registry-negative", offset=1)
    _complete_run(registry, "registry-positive", offset=2)
    with closing(sqlite3.connect(registry.path)) as connection, connection:
        connection.execute("DROP TRIGGER sl_registry_runs_no_update")
        connection.execute(
            "UPDATE sl_registry_runs SET sequence = -1 WHERE run_id = ?",
            ("registry-negative",),
        )

    with pytest.raises(IntegrityError, match="sequence"):
        ports.list_runs()


def test_registry_run_cursor_rejects_snapshot_source_regression(
    system: tuple[RunRegistry, ArtifactStore, RegistryReadPorts],
) -> None:
    registry, _, ports = system
    for offset in range(3):
        _complete_run(registry, f"regressed-run-{offset}", offset=offset)
    first = ports.list_runs(
        query=RunQuery(provenance=RunProvenance.REGISTRY_VERIFIED),
        page=RunPageRequest(page_size=1),
    )
    assert first.next_cursor is not None

    with closing(sqlite3.connect(registry.path)) as connection, connection:
        connection.execute("DROP TRIGGER sl_registry_runs_no_delete")
        connection.execute("DELETE FROM sl_registry_runs WHERE sequence > 1")

    with pytest.raises(InvalidCursorError, match="source regressed"):
        ports.list_runs(
            query=RunQuery(provenance=RunProvenance.REGISTRY_VERIFIED),
            page=RunPageRequest(page_size=1, cursor=first.next_cursor),
        )


def test_artifact_pagination_is_run_bound_tamper_evident_and_snapshot_stable(
    system: tuple[RunRegistry, ArtifactStore, RegistryReadPorts],
    tmp_path: Path,
) -> None:
    registry, store, ports = system
    published = tuple(
        _publish(
            registry,
            store,
            tmp_path.resolve(),
            name=f"manifest-{index}.json",
            payload=f'{{"index":{index}}}'.encode(),
        )
        for index in range(4)
    )
    _complete_run(
        registry,
        "artifacts-run",
        offset=0,
        artifact_links=tuple(
            ArtifactLink(f"manifest-{index}", artifact.digest)
            for index, artifact in enumerate(published[:3])
        ),
    )
    _complete_run(registry, "other-run", offset=1)

    first = ports.list_run_artifacts("artifacts-run", ArtifactPageRequest(page_size=1))
    assert first.next_cursor is not None
    with closing(sqlite3.connect(registry.path)) as connection, connection:
        connection.execute(
            """
            INSERT INTO sl_registry_run_artifacts(run_id, role, artifact_digest, linked_at)
            VALUES(?,?,?,?)
            """,
            (
                "artifacts-run",
                "manifest-after-snapshot",
                published[3].digest,
                NOW.isoformat(timespec="microseconds").replace("+00:00", "Z"),
            ),
        )
    second = ports.list_run_artifacts(
        "artifacts-run",
        ArtifactPageRequest(page_size=5, cursor=first.next_cursor),
    )

    assert len(first.items + second.items) == 3
    assert published[3].digest not in {
        link.artifact.artifact_id for link in first.items + second.items
    }
    with pytest.raises(InvalidCursorError, match="run filter"):
        ports.list_run_artifacts(
            "other-run",
            ArtifactPageRequest(page_size=1, cursor=first.next_cursor),
        )
    with pytest.raises(InvalidCursorError, match="authentication"):
        ports.list_run_artifacts(
            "artifacts-run",
            ArtifactPageRequest(
                page_size=1,
                cursor=ArtifactCursor(_tamper_cursor(first.next_cursor.token)),
            ),
        )


def test_negative_artifact_link_sequence_cannot_be_omitted_from_listing(
    system: tuple[RunRegistry, ArtifactStore, RegistryReadPorts],
    tmp_path: Path,
) -> None:
    registry, store, ports = system
    published = tuple(
        _publish(
            registry,
            store,
            tmp_path.resolve(),
            name=f"negative-sequence-{index}.json",
            payload=f'{{"index":{index}}}'.encode(),
        )
        for index in range(2)
    )
    _complete_run(
        registry,
        "negative-artifact-sequence",
        offset=0,
        artifact_links=tuple(
            ArtifactLink(f"manifest-{index}", artifact.digest)
            for index, artifact in enumerate(published)
        ),
    )
    with closing(sqlite3.connect(registry.path)) as connection, connection:
        connection.execute("DROP TRIGGER sl_registry_run_artifacts_no_update")
        connection.execute(
            "UPDATE sl_registry_run_artifacts SET sequence = -1 " "WHERE run_id = ? AND role = ?",
            ("negative-artifact-sequence", "manifest-0"),
        )

    with pytest.raises(IntegrityError, match="sequence"):
        ports.list_run_artifacts("negative-artifact-sequence")


def test_artifact_cursor_rejects_snapshot_source_regression(
    system: tuple[RunRegistry, ArtifactStore, RegistryReadPorts],
    tmp_path: Path,
) -> None:
    registry, store, ports = system
    published = tuple(
        _publish(
            registry,
            store,
            tmp_path.resolve(),
            name=f"regressed-artifact-{index}.json",
            payload=f'{{"index":{index}}}'.encode(),
        )
        for index in range(2)
    )
    run_id = "regressed-artifact-cursor"
    _complete_run(
        registry,
        run_id,
        offset=0,
        artifact_links=tuple(
            ArtifactLink(f"manifest-{index}", artifact.digest)
            for index, artifact in enumerate(published)
        ),
    )
    first = ports.list_run_artifacts(run_id, ArtifactPageRequest(page_size=1))
    assert first.next_cursor is not None

    with closing(sqlite3.connect(registry.path)) as connection, connection:
        connection.execute("DROP TRIGGER sl_registry_run_artifacts_no_delete")
        connection.execute(
            "DELETE FROM sl_registry_run_artifacts WHERE run_id = ? AND sequence > 1",
            (run_id,),
        )

    with pytest.raises(InvalidCursorError, match="source regressed"):
        ports.list_run_artifacts(
            run_id,
            ArtifactPageRequest(page_size=1, cursor=first.next_cursor),
        )


def test_missing_artifact_join_fails_closed_instead_of_silently_omitting_link(
    system: tuple[RunRegistry, ArtifactStore, RegistryReadPorts],
    tmp_path: Path,
) -> None:
    registry, store, ports = system
    published = _publish(
        registry,
        store,
        tmp_path.resolve(),
        name="orphan.json",
        payload=b"{}",
    )
    _complete_run(
        registry,
        "orphan-run",
        offset=0,
        artifact_links=(ArtifactLink("manifest", published.digest),),
    )
    with closing(sqlite3.connect(registry.path)) as connection, connection:
        connection.execute("PRAGMA foreign_keys=OFF")
        connection.execute("DROP TRIGGER sl_registry_artifacts_no_delete")
        connection.execute(
            "DELETE FROM sl_registry_artifacts WHERE digest = ?", (published.digest,)
        )

    with pytest.raises(IntegrityError, match="missing metadata"):
        ports.list_run_artifacts("orphan-run")


def test_legacy_run_has_no_verified_artifact_links(
    system: tuple[RunRegistry, ArtifactStore, RegistryReadPorts],
) -> None:
    registry, _, ports = system
    _insert_legacy(registry.path, "legacy-artifacts", offset=1)

    assert ports.list_run_artifacts("legacy-artifacts").items == ()
    with pytest.raises(NotFoundError):
        ports.list_run_artifacts("missing-run")


def test_manifest_media_and_byte_limits_fail_before_cas_read(
    system: tuple[RunRegistry, ArtifactStore, RegistryReadPorts],
    tmp_path: Path,
) -> None:
    registry, store, ports = system
    published = _publish(
        registry,
        store,
        tmp_path.resolve(),
        name="bounded.json",
        payload=b'{"safe":true}',
    )

    with pytest.raises(ValidationError, match="allowlisted"):
        ports.read_verified_manifest(published.digest, "text/plain", 100)
    with pytest.raises(IntegrityError, match="media type"):
        ports.read_verified_manifest(published.digest, "application/manifest+json", 100)
    with pytest.raises(ValidationError, match="byte limit"):
        ports.read_verified_manifest(published.digest, "application/json", 1)
    with pytest.raises(ValidationError, match="max_bytes"):
        ports.read_verified_manifest(published.digest, "application/json", 0)


def test_manifest_read_requires_metadata_artifact_authority(
    system: tuple[RunRegistry, ArtifactStore, RegistryReadPorts],
    tmp_path: Path,
) -> None:
    registry, store, ports = system
    source = tmp_path.resolve() / "model-labelled-json"
    source.write_bytes(b'{"not":"a manifest"}')
    published = store.publish(source)
    registry.register_artifact(
        published,
        artifact_class=ArtifactClass.MODEL,
        media_type="application/json",
    )

    with pytest.raises(IntegrityError, match="not authorized"):
        ports.read_verified_manifest(published.digest, "application/json", 100)


def test_manifest_read_rejects_registry_storage_key_substitution(
    system: tuple[RunRegistry, ArtifactStore, RegistryReadPorts],
    tmp_path: Path,
) -> None:
    registry, store, ports = system
    published = _publish(
        registry,
        store,
        tmp_path.resolve(),
        name="storage-key.json",
        payload=b"{}",
    )
    wrong_key = f"objects/ff/ff/{published.digest}"
    with closing(sqlite3.connect(registry.path)) as connection, connection:
        connection.execute("DROP TRIGGER sl_registry_artifacts_no_update")
        connection.execute(
            "UPDATE sl_registry_artifacts SET storage_relpath = ? WHERE digest = ?",
            (wrong_key, published.digest),
        )

    with pytest.raises(IntegrityError, match="noncanonical"):
        ports.get_artifact(published.digest)
    with pytest.raises(IntegrityError, match="noncanonical"):
        ports.read_verified_manifest(published.digest, "application/json", 10)


def test_manifest_read_rejects_a_different_initialized_cas_identity(
    system: tuple[RunRegistry, ArtifactStore, RegistryReadPorts],
    tmp_path: Path,
) -> None:
    registry, store, _ = system
    payload = b'{"store":"bound"}'
    published = _publish(
        registry,
        store,
        tmp_path.resolve(),
        name="bound-store.json",
        payload=payload,
    )
    other_store = ArtifactStore(tmp_path.resolve() / "other-cas")
    other_store.initialize()
    other_source = tmp_path.resolve() / "same-payload.json"
    other_source.write_bytes(payload)
    assert other_store.publish(other_source).digest == published.digest
    ports = RegistryReadPorts(registry, other_store)

    with pytest.raises(IntegrityError, match="does not match"):
        ports.read_verified_manifest(
            published.digest,
            "application/json",
            len(payload),
        )


def test_manifest_read_rejects_cas_digest_tamper_and_symlink(
    system: tuple[RunRegistry, ArtifactStore, RegistryReadPorts],
    tmp_path: Path,
) -> None:
    registry, store, ports = system
    published = _publish(
        registry,
        store,
        tmp_path.resolve(),
        name="tamper.json",
        payload=b'{"value":1}',
    )
    object_path = store.root.joinpath(*published.storage_key.split("/"))
    object_path.chmod(0o600)
    object_path.write_bytes(b'{"value":2}')

    with pytest.raises(IntegrityError, match="verified CAS read"):
        ports.read_verified_manifest(published.digest, "application/json", 100)

    object_path.unlink()
    os.symlink(tmp_path.resolve() / "tamper.json", object_path)
    with pytest.raises(IntegrityError, match="verified CAS read"):
        ports.read_verified_manifest(published.digest, "application/json", 100)


def test_read_deadline_interrupts_a_large_legacy_query_without_partial_results(
    system: tuple[RunRegistry, ArtifactStore, RegistryReadPorts],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    registry, store, _ = system
    _create_legacy_table(registry.path)
    row = (
        "experiment",
        "run",
        NOW.isoformat(),
        (NOW + timedelta(seconds=1)).isoformat(),
        "completed",
        "abcdef0",
        "a" * 64,
        "[]",
        "[]",
        "{}",
        "{}",
        "{}",
        "[]",
    )
    with closing(sqlite3.connect(registry.path)) as connection, connection:
        connection.executemany(
            """
            INSERT INTO runs(
                run_id, experiment, name, started_at, ended_at, status, git_commit,
                data_hash, tickers, features, params, metrics, tags, artifacts
            ) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?)
            """,
            ((f"deadline-{index}", *row) for index in range(500)),
        )
    ports = RegistryReadPorts(
        registry,
        store,
        limits=ReadPortLimits(query_timeout_ms=1),
    )
    calls = 0

    def expired_clock() -> float:
        nonlocal calls
        calls += 1
        return 0.0 if calls == 1 else 1.0

    monkeypatch.setattr(time, "monotonic", expired_clock)

    with pytest.raises(ReadTimeoutError):
        ports.list_runs(page=RunPageRequest(page_size=100))


def test_read_connection_caps_busy_wait_at_the_query_deadline(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path.resolve()
    registry = RunRegistry(
        root / "registry.sqlite",
        digest_secret=SECRET,
        limits=RegistryLimits(busy_timeout_ms=60_000),
    )
    registry.initialize()
    store = ArtifactStore(root / "cas")
    store.initialize()
    ports = RegistryReadPorts(
        registry,
        store,
        limits=ReadPortLimits(query_timeout_ms=17),
    )
    observed: list[int] = []
    real_connect = registry._connect

    def recording_connect(
        *,
        readonly: bool = False,
        busy_timeout_ms: int | None = None,
    ) -> sqlite3.Connection:
        assert busy_timeout_ms is not None
        observed.append(busy_timeout_ms)
        return real_connect(readonly=readonly, busy_timeout_ms=busy_timeout_ms)

    monkeypatch.setattr(registry, "_connect", recording_connect)

    assert ports.list_runs().items == ()
    assert observed == [17]


class _RollbackFaultConnection:
    """Proxy a real read connection while faulting query and rollback independently."""

    def __init__(self, connection: sqlite3.Connection, *, query_fails: bool) -> None:
        self._connection = connection
        self._query_fails = query_fails

    @property
    def in_transaction(self) -> bool:
        return self._connection.in_transaction

    def set_progress_handler(self, *args: object) -> None:
        self._connection.set_progress_handler(*args)  # type: ignore[arg-type]

    def execute(self, sql: str, parameters: object = ()) -> sqlite3.Cursor:
        if sql == "ROLLBACK":
            raise sqlite3.OperationalError("sensitive rollback detail")
        if self._query_fails and sql.startswith("SELECT"):
            raise sqlite3.DatabaseError("sensitive primary detail")
        return self._connection.execute(sql, parameters)  # type: ignore[arg-type]

    def close(self) -> None:
        self._connection.close()


@pytest.mark.parametrize("query_fails", [False, True])
def test_read_cleanup_failure_preserves_primary_error(
    system: tuple[RunRegistry, ArtifactStore, RegistryReadPorts],
    monkeypatch: pytest.MonkeyPatch,
    query_fails: bool,
) -> None:
    registry, _, ports = system
    real_connect = registry._connect

    def faulting_connect(
        *,
        readonly: bool = False,
        busy_timeout_ms: int | None = None,
    ) -> _RollbackFaultConnection:
        connection = real_connect(readonly=readonly, busy_timeout_ms=busy_timeout_ms)
        return _RollbackFaultConnection(connection, query_fails=query_fails)

    monkeypatch.setattr(registry, "_connect", faulting_connect)

    with pytest.raises(IntegrityError) as caught:
        ports.get_run("missing-run")

    if query_fails:
        assert str(caught.value) == "SQLite rejected a registry read"
        assert caught.value.__notes__ == ["registry read cleanup also failed: OperationalError"]
    else:
        assert str(caught.value) == "registry read cleanup failed"
    assert "sensitive" not in str(caught.value)


def test_public_validation_rejects_pathlike_identifiers_and_wrong_contracts(
    system: tuple[RunRegistry, ArtifactStore, RegistryReadPorts],
) -> None:
    _, _, ports = system

    with pytest.raises(ValidationError):
        ports.get_run("../escape")
    with pytest.raises(ValidationError):
        ports.get_artifact("not-a-digest")
    with pytest.raises(InvalidCursorError):
        RunPageRequest(cursor="not-a-cursor")  # type: ignore[arg-type]
    with pytest.raises(InvalidCursorError):
        ArtifactPageRequest(cursor=RunCursor("v1.aaaaaaaaaaaaaaaa.aaaaaaaaaaaaaaaa"))  # type: ignore[arg-type]
