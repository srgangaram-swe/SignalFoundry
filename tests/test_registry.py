"""Focused lifecycle, contention, and boundary tests for the durable registry."""

from __future__ import annotations

import hashlib
import sqlite3
import threading
from collections.abc import Iterator, Mapping, Sequence
from contextlib import closing
from dataclasses import FrozenInstanceError, dataclass
from datetime import UTC, datetime, timedelta, tzinfo
from pathlib import Path
from typing import overload

import pytest

from quant_platform.tracking import registry as registry_module
from quant_platform.tracking.cas import ArtifactStore, PublishedArtifact
from quant_platform.tracking.contracts import (
    ArtifactClass,
    ArtifactCursor,
    ArtifactLink,
    ArtifactMetadata,
    BusyError,
    CancellationReasonCode,
    CapacityError,
    ClaimedJob,
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
    NotFoundError,
    Page,
    RegistryLimits,
    RegistryReadiness,
    RetentionPendingError,
    RetentionPlan,
    RetentionPlanAuthenticationError,
    RunSnapshot,
    RunStatus,
    SubmissionRequest,
    TerminalRunRequest,
    ValidationError,
    canonical_json,
    require_failure_summary,
)
from quant_platform.tracking.migrations import _is_busy, open_database
from quant_platform.tracking.registry import CursorCodec, RunRegistry

NOW = datetime(2026, 8, 8, 12, 0, tzinfo=UTC)
SECRET = b"registry-test-secret-material-32-bytes-minimum"


@dataclass(slots=True)
class MutableClock:
    """Deterministic internal authority used only at registry construction."""

    current: datetime
    reads: int = 0

    def now(self) -> datetime:
        self.reads += 1
        return self.current

    def advance(self, seconds: int) -> None:
        self.current += timedelta(seconds=seconds)


class HostileTimezone(tzinfo):
    """Timezone that proves boundary normalization suppresses untrusted failures."""

    def utcoffset(self, value: datetime | None) -> timedelta | None:
        del value
        raise RuntimeError("attacker-controlled timezone secret")

    def dst(self, value: datetime | None) -> timedelta | None:
        del value
        return timedelta(0)

    def tzname(self, value: datetime | None) -> str | None:
        del value
        return "hostile-test-timezone"


class HostileMapping(Mapping[str, object]):
    """Mapping whose protocol hooks must never execute at the JSON boundary."""

    def __getitem__(self, key: str) -> object:
        del key
        raise RuntimeError("attacker-controlled mapping secret")

    def __iter__(self) -> Iterator[str]:
        raise RuntimeError("attacker-controlled mapping secret")

    def __len__(self) -> int:
        raise RuntimeError("attacker-controlled mapping secret")


class HostileSequence(Sequence[object]):
    """Sequence whose protocol hooks must never execute at the JSON boundary."""

    @overload
    def __getitem__(self, index: int) -> object: ...

    @overload
    def __getitem__(self, index: slice) -> Sequence[object]: ...

    def __getitem__(self, index: int | slice) -> object | Sequence[object]:
        del index
        raise RuntimeError("attacker-controlled sequence secret")

    def __len__(self) -> int:
        raise RuntimeError("attacker-controlled sequence secret")


class HostilePathLike:
    """Path protocol adversary whose exception must never escape construction."""

    def __fspath__(self) -> str:
        raise RuntimeError("attacker-controlled path secret")


class HostileStr(str):
    """Executable string subclass rejected before inherited string operations."""

    def __len__(self) -> int:
        raise RuntimeError("attacker-controlled string secret")


class HostileBool:
    """Wrong protocol object proving optional defaults never execute truthiness."""

    def __bool__(self) -> bool:
        raise RuntimeError("attacker-controlled truthiness secret")


class HostileClockDescriptor:
    @property
    def now(self) -> object:
        raise RuntimeError("attacker-controlled clock descriptor secret")


class HostileVerifierDescriptor:
    @property
    def verify(self) -> object:
        raise RuntimeError("attacker-controlled verifier descriptor secret")


class BeginSignallingConnection:
    """Test proxy that signals immediately before SQLite attempts its writer lock."""

    def __init__(self, connection: sqlite3.Connection, signal: threading.Event) -> None:
        self._connection = connection
        self._signal = signal

    def execute(self, sql: str, parameters: object = ()) -> sqlite3.Cursor:
        if sql.strip().upper() == "BEGIN IMMEDIATE":
            self._signal.set()
        return self._connection.execute(sql, parameters)  # type: ignore[arg-type]

    def __getattr__(self, name: str) -> object:
        return getattr(self._connection, name)


class BlockingArtifactVerifier:
    """Test verifier that can pause one full hash at a deterministic barrier."""

    def __init__(self, store: ArtifactStore) -> None:
        self._store = store
        self._lock = threading.Lock()
        self._armed = False
        self.entered = threading.Event()
        self.release = threading.Event()

    @property
    def store_id(self) -> str:
        return self._store.store_id

    def arm(self) -> None:
        with self._lock:
            self._armed = True
            self.entered.clear()
            self.release.clear()

    def verify(self, artifact: PublishedArtifact) -> None:
        with self._lock:
            should_block = self._armed
            self._armed = False
        if should_block:
            self.entered.set()
            if not self.release.wait(timeout=5):
                raise AssertionError("test verifier release barrier timed out")
        self._store.verify(artifact)


@pytest.fixture
def registry(tmp_path: Path) -> RunRegistry:
    clock = MutableClock(NOW)
    store = ArtifactStore(tmp_path / "cas", max_artifact_bytes=1_024)
    store.initialize()
    result = RunRegistry(
        tmp_path / "registry.sqlite",
        digest_secret=SECRET,
        limits=RegistryLimits(busy_timeout_ms=2_000),
        clock=clock,
        artifact_verifier=store,
    )
    result.initialize()
    return result


def _request(value: int = 1, *, max_attempts: int = 3) -> SubmissionRequest:
    return SubmissionRequest(
        "forecast", {"seed": value, "nested": {"enabled": True}}, max_attempts=max_attempts
    )


def _terminal_run(
    run_id: str,
    *,
    evidence_class: EvidenceClass = EvidenceClass.SIMULATED,
    started_at: datetime | None = NOW,
) -> TerminalRunRequest:
    return TerminalRunRequest(
        run_id=run_id,
        evidence_class=evidence_class,
        started_at=started_at,
        source_commit="abcdef0",
        data_identity="synthetic-v1",
        limitation_summary="network-free test evidence only",
    )


def _clock(registry: RunRegistry) -> MutableClock:
    assert isinstance(registry._clock, MutableClock)
    return registry._clock


def _store(registry: RunRegistry) -> ArtifactStore:
    assert isinstance(registry._artifact_verifier, ArtifactStore)
    return registry._artifact_verifier


def test_constructor_rejects_hostile_optional_protocols_without_side_effects(
    tmp_path: Path,
) -> None:
    before = frozenset(tmp_path.iterdir())

    for hostile_path in (HostilePathLike(), HostileStr("hostile.sqlite")):
        with pytest.raises(ValidationError, match="path") as caught:
            RunRegistry(hostile_path, digest_secret=SECRET)  # type: ignore[arg-type]
        assert caught.value.__cause__ is None

    for wrong_limits in (0, HostileBool()):
        with pytest.raises(ValidationError, match="limits"):
            RunRegistry(
                tmp_path / "registry.sqlite",
                digest_secret=SECRET,
                limits=wrong_limits,  # type: ignore[arg-type]
            )
    for wrong_clock in (0, HostileBool(), HostileClockDescriptor()):
        with pytest.raises(ValidationError, match="clock"):
            RunRegistry(
                tmp_path / "registry.sqlite",
                digest_secret=SECRET,
                clock=wrong_clock,  # type: ignore[arg-type]
            )
    for wrong_verifier in (0, HostileVerifierDescriptor()):
        with pytest.raises(ValidationError, match="artifact_verifier"):
            RunRegistry(
                tmp_path / "registry.sqlite",
                digest_secret=SECRET,
                artifact_verifier=wrong_verifier,  # type: ignore[arg-type]
            )

    assert frozenset(tmp_path.iterdir()) == before


def test_authenticated_connection_accepts_only_stricter_busy_timeout(
    registry: RunRegistry,
) -> None:
    with closing(registry._connect(readonly=True, busy_timeout_ms=17)) as connection:
        assert int(connection.execute("PRAGMA busy_timeout").fetchone()[0]) == 17

    for invalid in (True, -1, 2_001, 1.0):
        with pytest.raises(ValidationError):
            registry._connect(readonly=True, busy_timeout_ms=invalid)  # type: ignore[arg-type]


def _publish(registry: RunRegistry, payload: bytes = b"{}") -> PublishedArtifact:
    store = _store(registry)
    source = store.root.parent / f"source-{len(payload)}-{payload.hex()[:16]}.bin"
    source.write_bytes(payload)
    return store.publish(source)


def _stage_pending_retention(
    registry: RunRegistry,
    *,
    artifact_digest: str | None = None,
) -> str:
    if artifact_digest is None:
        published = _publish(registry, b"retention-claim-barrier")
        registry.register_artifact(
            published,
            artifact_class=ArtifactClass.OUTPUT,
            media_type="application/octet-stream",
        )
        artifact_digest = published.digest
    store_id = registry.artifact_store_id
    assert store_id is not None
    payload = canonical_json(
        {
            "candidates": [{"digest": artifact_digest, "generation": 1}],
            "planned_at": "2026-08-08T12:00:00.000000Z",
            "schema_version": 1,
        }
    ).encode()
    plan_digest = registry.sign_retention_payload(payload, store_id=store_id)
    registry_id = registry.registry_id
    with closing(registry._connect()) as connection:
        connection.execute("BEGIN IMMEDIATE")
        connection.execute(
            """
            INSERT INTO sl_registry_retention_plans(
                plan_digest, payload_digest, payload_json, registry_id,
                cas_store_id, planned_at, schema_version
            ) VALUES(?, ?, ?, ?, ?, ?, 1)
            """,
            (
                plan_digest,
                hashlib.sha256(payload).hexdigest(),
                payload.decode(),
                registry_id,
                store_id,
                "2026-08-08T12:00:00.000000Z",
            ),
        )
        connection.execute(
            """
            INSERT INTO sl_registry_retention_tombstones(
                artifact_digest, plan_digest, planned_at, reason
            ) VALUES(?, ?, ?, 'bounded test retention')
            """,
            (artifact_digest, plan_digest, "2026-08-08T12:00:00.000000Z"),
        )
        connection.execute("COMMIT")
    return plan_digest


def test_submission_contract_is_deeply_frozen_and_rejects_nonfinite_json() -> None:
    payload: dict[str, JsonValue] = {"nested": {"value": 1}, "items": (1, 2)}
    request = SubmissionRequest("forecast", payload)
    payload["nested"] = {"value": 99}

    assert request.canonical_json() == (
        '{"kind":"forecast","max_attempts":3,"payload":{"items":[1,2],'
        '"nested":{"value":1}},"priority":0,"schema_version":1}'
    )
    with pytest.raises(TypeError):
        request.payload["new"] = 1  # type: ignore[index]
    with pytest.raises(FrozenInstanceError):
        request.priority = 2  # type: ignore[misc]
    with pytest.raises(ValidationError, match="finite"):
        SubmissionRequest("forecast", {"value": float("nan")})
    nested: object = 1
    for _ in range(18):
        nested = {"next": nested}
    with pytest.raises(ValidationError, match="depth"):
        SubmissionRequest("forecast", {"nested": nested})  # type: ignore[dict-item]
    with pytest.raises(ValidationError, match="unsupported"):
        SubmissionRequest("forecast", {"bytes": b"not-json"})  # type: ignore[dict-item]

    shared_maximum_string = "x" * 65_536
    with pytest.raises(ValidationError, match="cumulative canonical byte budget"):
        SubmissionRequest(
            "forecast",
            {"chunks": [shared_maximum_string] * 10_000},  # type: ignore[dict-item]
        )


@pytest.mark.parametrize("hostile", [HostileMapping(), HostileSequence()])
def test_json_boundary_rejects_custom_containers_without_dispatch(hostile: object) -> None:
    with pytest.raises(ValidationError, match="unsupported JSON value type") as caught:
        canonical_json(hostile)

    assert caught.value.__cause__ is None
    assert "attacker-controlled" not in str(caught.value)
    assert "secret" not in str(caught.value)


def test_timestamp_contract_suppresses_hostile_timezone_failures() -> None:
    hostile = datetime(2026, 8, 8, 12, 0, tzinfo=HostileTimezone())

    with pytest.raises(ValidationError, match="normalized safely") as captured:
        TerminalRunRequest(
            run_id="hostile-timezone-run",
            evidence_class=EvidenceClass.SIMULATED,
            started_at=hostile,
        )

    assert captured.value.__cause__ is None
    assert "attacker-controlled" not in str(captured.value)
    assert "attacker-controlled" not in repr(captured.value)


def test_submit_is_durable_idempotent_and_never_persists_raw_key(registry: RunRegistry) -> None:
    key = "opaque-idempotency-key-123"
    first = registry.submit(_request(), idempotency_key=key)
    second = registry.submit(_request(), idempotency_key=key)

    assert second == first
    assert len(registry.list_jobs().items) == 1
    assert key.encode() not in registry.path.read_bytes()
    assert SECRET not in registry.path.read_bytes()
    with pytest.raises(ConflictError):
        registry.submit(_request(2), idempotency_key=key)


def test_queue_capacity_counts_nonterminal_work(tmp_path: Path) -> None:
    registry = RunRegistry(
        tmp_path / "bounded.sqlite",
        digest_secret=SECRET,
        limits=RegistryLimits(queue_capacity=1, busy_timeout_ms=2_000),
    )
    registry.initialize()
    first = registry.submit(_request(), idempotency_key="capacity-key-one")

    with pytest.raises(CapacityError):
        registry.submit(_request(2), idempotency_key="capacity-key-two")
    registry.request_cancel(first.job_id, reason=CancellationReasonCode.OPERATOR_REQUEST)
    assert registry.submit(_request(2), idempotency_key="capacity-key-two").state is JobState.QUEUED


def test_canonical_request_size_is_bounded_before_admission(tmp_path: Path) -> None:
    registry = RunRegistry(
        tmp_path / "request-bound.sqlite",
        digest_secret=SECRET,
        limits=RegistryLimits(max_request_bytes=1_024),
    )
    registry.initialize()

    with pytest.raises(ValidationError, match="1024-byte bound"):
        registry.submit(
            SubmissionRequest("forecast", {"blob": "x" * 1_024}),
            idempotency_key="request-size-bound-key",
        )
    assert registry.list_jobs().items == ()


def test_event_capacity_formula_is_exhaustive_at_every_attempt_boundary() -> None:
    for max_attempts in range(1, 101):
        formula_bound = 2 * max_attempts + 2
        required = max(8, formula_bound)
        assert (
            RegistryLimits(
                max_attempts=max_attempts,
                max_events_per_job=required,
            ).max_events_per_job
            == required
        )
        if formula_bound > 8:
            with pytest.raises(ValidationError, match=r"2 \* max_attempts \+ 2"):
                RegistryLimits(
                    max_attempts=max_attempts,
                    max_events_per_job=formula_bound - 1,
                )
    with pytest.raises(ValidationError, match=r"\[8, 10000\]"):
        RegistryLimits(max_attempts=1, max_events_per_job=7)


def test_exact_event_bound_allows_final_lease_cancellation(tmp_path: Path) -> None:
    clock = MutableClock(NOW)
    registry = RunRegistry(
        tmp_path / "exact-event-bound.sqlite",
        digest_secret=SECRET,
        limits=RegistryLimits(
            max_attempts=4,
            max_events_per_job=10,
            busy_timeout_ms=2_000,
        ),
        clock=clock,
    )
    registry.initialize()
    job = registry.submit(
        _request(max_attempts=4),
        idempotency_key="exact-event-capacity-key",
    )
    for attempt in range(1, 4):
        claimed = registry.claim(worker_id=f"worker-{attempt}", lease_seconds=10)
        assert claimed is not None
        registry.fail(
            claimed.lease,
            failure_code=FailureReasonCode.TRANSIENT_IO,
            failure_summary="Bounded retryable interruption.",
            retryable=True,
            run=_terminal_run(f"run-retry-{attempt}"),
        )
    final = registry.claim(worker_id="worker-final", lease_seconds=10)
    assert final is not None
    registry.request_cancel(job.job_id, reason=CancellationReasonCode.OPERATOR_REQUEST)
    cancelled = registry.acknowledge_cancel(
        final.lease,
        run=_terminal_run("run-final-cancel"),
    )

    assert cancelled.state is JobState.CANCELLED
    assert len(registry.list_events(job_id=job.job_id, page_size=10).items) == 10


def test_job_keyset_page_excludes_rows_created_after_first_page(registry: RunRegistry) -> None:
    for seed in range(3):
        registry.submit(_request(seed), idempotency_key=f"stable-page-key-{seed}")

    first = registry.list_jobs(page_size=2)
    assert first.next_cursor is not None
    registry.submit(_request(4), idempotency_key="stable-page-key-new")
    second = registry.list_jobs(page_size=2, cursor=first.next_cursor)

    assert [item.sequence for item in first.items + second.items] == [1, 2, 3]
    assert second.next_cursor is None


def test_state_filtered_cursor_fails_closed_when_remaining_membership_changes(
    registry: RunRegistry,
) -> None:
    jobs = tuple(
        registry.submit(_request(seed), idempotency_key=f"state-snapshot-key-{seed}")
        for seed in range(3)
    )
    first = registry.list_jobs(page_size=1, state=JobState.QUEUED)
    assert first.next_cursor is not None

    registry.request_cancel(
        jobs[1].job_id,
        reason=CancellationReasonCode.OPERATOR_REQUEST,
    )

    with pytest.raises(InvalidCursorError, match="snapshot changed"):
        registry.list_jobs(
            page_size=1,
            cursor=first.next_cursor,
            state=JobState.QUEUED,
        )


def test_cursor_authentication_version_and_filter_binding(registry: RunRegistry) -> None:
    for seed in range(3):
        registry.submit(_request(seed), idempotency_key=f"cursor-auth-key-{seed}")
    first = registry.list_jobs(page_size=1, state=JobState.QUEUED)
    assert first.next_cursor is not None
    token = first.next_cursor.token

    tampered = JobCursor(token[:-1] + ("A" if token[-1] != "A" else "B"))
    with pytest.raises(InvalidCursorError, match="authentication"):
        registry.list_jobs(page_size=1, cursor=tampered, state=JobState.QUEUED)
    with pytest.raises(InvalidCursorError, match="version"):
        registry.list_jobs(
            page_size=1,
            cursor=JobCursor(token.replace("v1.", "v2.", 1)),
            state=JobState.QUEUED,
        )
    with pytest.raises(InvalidCursorError, match="filter"):
        registry.list_jobs(page_size=1, cursor=first.next_cursor, state=None)

    other_secret = RunRegistry(registry.path, digest_secret=b"x" * 32)
    with pytest.raises(IntegrityError, match="authority"):
        other_secret.list_jobs(
            page_size=1,
            cursor=first.next_cursor,
            state=JobState.QUEUED,
        )


def test_job_and_event_keysets_reject_negative_durable_sequences(registry: RunRegistry) -> None:
    first = registry.submit(_request(), idempotency_key="negative-job-sequence-one")
    registry.submit(_request(2), idempotency_key="negative-job-sequence-two")
    with closing(sqlite3.connect(registry.path)) as connection, connection:
        connection.execute(
            "UPDATE sl_registry_jobs SET sequence = -1 WHERE job_id = ?",
            (first.job_id,),
        )

    with pytest.raises(IntegrityError, match="sequence"):
        registry.list_jobs()

    with closing(sqlite3.connect(registry.path)) as connection, connection:
        connection.execute("DROP TRIGGER sl_registry_events_no_update")
        connection.execute(
            "UPDATE sl_registry_events SET sequence = -1 WHERE job_id = ?",
            (first.job_id,),
        )

    with pytest.raises(IntegrityError, match="sequence"):
        registry.list_events()


def test_job_and_event_cursors_reject_snapshot_source_regression(registry: RunRegistry) -> None:
    for seed in range(3):
        registry.submit(_request(seed), idempotency_key=f"regressed-job-cursor-{seed}")
    job_page = registry.list_jobs(page_size=1)
    event_page = registry.list_events(page_size=1)
    assert job_page.next_cursor is not None
    assert event_page.next_cursor is not None

    with closing(sqlite3.connect(registry.path)) as connection, connection:
        connection.execute("DROP TRIGGER sl_registry_events_no_delete")
        connection.execute("DELETE FROM sl_registry_events WHERE sequence > 1")

    with pytest.raises(InvalidCursorError, match="source regressed"):
        registry.list_events(page_size=1, cursor=event_page.next_cursor)

    with closing(sqlite3.connect(registry.path)) as connection, connection:
        connection.execute("DELETE FROM sl_registry_jobs WHERE sequence > 1")

    with pytest.raises(InvalidCursorError, match="source regressed"):
        registry.list_jobs(page_size=1, cursor=job_page.next_cursor)


def test_cursor_codec_round_trips_each_query_family_and_rejects_malformed() -> None:
    codec = CursorCodec(SECRET)
    event = codec.encode_event(10, 4, "a" * 32)
    run = codec.encode_run(20, 7, "status=succeeded")
    artifact = codec.encode_artifact(30, 9, "run-1")

    assert codec.decode_event(event) == (10, 4, "a" * 32)
    assert codec.decode_run(run) == (20, 7, "status=succeeded")
    assert codec.decode_artifact(artifact) == (30, 9, "run-1")
    with pytest.raises(InvalidCursorError, match="wrong contract"):
        codec.decode_artifact(run)  # type: ignore[arg-type]
    with pytest.raises(InvalidCursorError, match="authentication"):
        codec.decode_artifact(
            ArtifactCursor(artifact.token[:-1] + ("A" if artifact.token[-1] != "A" else "B"))
        )
    with pytest.raises(InvalidCursorError, match="bounded ASCII"):
        JobCursor("short")
    with pytest.raises(InvalidCursorError, match="base64url"):
        codec.decode_event(
            EventCursor("v1.****************.aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa")
        )
    with pytest.raises(InvalidCursorError, match="sequences"):
        codec.encode_run(1, 2)
    with pytest.raises(ValidationError, match="UTF-8"):
        codec.encode_run(20, 7, "\ud800")


def test_claim_heartbeat_complete_and_append_only_events(registry: RunRegistry) -> None:
    request = _request()
    job = registry.submit(request, idempotency_key="lifecycle-key")
    claimed = registry.claim(worker_id="worker-1", lease_seconds=10)
    assert claimed is not None and claimed.request == request
    lease = claimed.lease
    published = _publish(registry)
    digest = published.digest
    registry.register_artifact(
        published,
        artifact_class=ArtifactClass.REPORT,
        media_type="application/json",
    )

    _clock(registry).advance(1)
    renewed = registry.heartbeat(lease, lease_seconds=10)
    _clock(registry).advance(1)
    completed = registry.complete(
        renewed,
        run=_terminal_run("run-1"),
        artifact_links=(ArtifactLink("report", digest),),
    )

    assert completed.state is JobState.SUCCEEDED
    assert completed.result_run_id == "run-1"
    assert [event.kind.value for event in registry.list_events(job_id=job.job_id).items] == [
        "submitted",
        "claimed",
        "succeeded",
    ]
    connection = sqlite3.connect(registry.path)
    try:
        with pytest.raises(sqlite3.IntegrityError, match="append-only"):
            connection.execute("DELETE FROM sl_registry_events")
        with pytest.raises(sqlite3.IntegrityError, match="terminal"):
            connection.execute(
                "UPDATE sl_registry_jobs SET state = 'queued', terminal_at = NULL, "
                "result_run_id = NULL WHERE job_id = ?",
                (job.job_id,),
            )
        with pytest.raises(sqlite3.IntegrityError, match="sl_registry_runs is immutable"):
            connection.execute("UPDATE sl_registry_runs SET status = 'failed'")
        with pytest.raises(sqlite3.IntegrityError, match="sl_registry_artifacts is immutable"):
            connection.execute("UPDATE sl_registry_artifacts SET pinned = 1")
        with pytest.raises(sqlite3.IntegrityError, match="sl_registry_run_artifacts is immutable"):
            connection.execute("DELETE FROM sl_registry_run_artifacts")
        assert connection.execute(
            "SELECT sequence, role, artifact_digest FROM sl_registry_run_artifacts"
        ).fetchone() == (1, "report", digest)
    finally:
        connection.close()


def test_artifact_registration_is_idempotent_only_for_exact_metadata(
    registry: RunRegistry,
) -> None:
    published = _publish(registry, b"plot")
    first = registry.register_artifact(
        published,
        artifact_class=ArtifactClass.PLOT,
        media_type="image/png",
        pinned=True,
    )
    repeated = registry.register_artifact(
        published,
        artifact_class=ArtifactClass.PLOT,
        media_type="image/png",
        pinned=True,
    )

    assert repeated == first
    with pytest.raises(ConflictError, match="other metadata"):
        registry.register_artifact(
            published,
            artifact_class=ArtifactClass.REPORT,
            media_type="application/json",
            pinned=True,
        )


def test_artifact_media_type_is_ascii_and_bounded_before_sqlite(
    registry: RunRegistry,
) -> None:
    published = _publish(registry, b"bounded-media-type")
    maximum_media_type = f"{'a' * 123}/json"
    assert len(maximum_media_type) == 128
    assert (
        ArtifactMetadata(
            digest=published.digest,
            artifact_class=ArtifactClass.REPORT,
            byte_size=published.byte_size,
            media_type=maximum_media_type,
            storage_relpath=published.storage_key,
            created_at=NOW,
        ).media_type
        == maximum_media_type
    )

    for invalid in (f"{'a' * 124}/json", "téxt/plain"):
        with pytest.raises(ValidationError, match="ASCII.*128 bytes") as captured:
            registry.register_artifact(
                published,
                artifact_class=ArtifactClass.REPORT,
                media_type=invalid,
            )
        assert captured.value.__cause__ is None

    with closing(registry._connect(readonly=True)) as connection:
        assert connection.execute("SELECT count(*) FROM sl_registry_artifacts").fetchone()[0] == 0


def test_registry_and_cas_identities_bind_retention_hmacs_without_paths(
    tmp_path: Path,
) -> None:
    first_store = ArtifactStore(tmp_path / "cas-first", max_artifact_bytes=1_024)
    first_store.initialize()
    first = RunRegistry(
        tmp_path / "registry-first.sqlite",
        digest_secret=SECRET,
        artifact_verifier=first_store,
    )
    first.initialize()
    second = RunRegistry(tmp_path / "registry-second.sqlite", digest_secret=SECRET)
    second.initialize()
    second.bind_artifact_store(first_store.store_id)
    payload = canonical_json(
        {"candidates": [], "planned_at": "2026-08-08T12:00:00.000000Z", "schema_version": 1}
    ).encode()

    first_digest = first.sign_retention_payload(payload, store_id=first_store.store_id)
    second_digest = second.sign_retention_payload(payload, store_id=first_store.store_id)

    assert first.artifact_verifier is first_store
    assert first.artifact_store_id == first_store.store_id
    assert first.registry_id != second.registry_id
    assert len(first.registry_id) == 32
    assert first_digest != second_digest
    first.verify_retention_payload_signature(
        payload,
        store_id=first_store.store_id,
        signature=first_digest,
    )
    with pytest.raises(RetentionPlanAuthenticationError, match="authentication"):
        first.verify_retention_payload_signature(
            payload,
            store_id=first_store.store_id,
            signature=second_digest,
        )
    with pytest.raises(ValidationError, match="canonical JSON"):
        first.sign_retention_payload(b'{"schema_version": 1}', store_id=first_store.store_id)
    first.bind_artifact_store(first_store.store_id)
    with pytest.raises(ConflictError, match="another CAS"):
        first.bind_artifact_store("a" * 64)

    reopened = RunRegistry(first.path, digest_secret=SECRET)
    assert reopened.registry_id == first.registry_id
    assert reopened.artifact_store_id == first_store.store_id


def test_claim_pauses_only_for_pending_retention_intent(registry: RunRegistry) -> None:
    job = registry.submit(_request(), idempotency_key="retention-barrier-key")
    _stage_pending_retention(registry)

    with pytest.raises(RetentionPendingError) as captured:
        registry.claim(worker_id="worker-retention-barrier", lease_seconds=10)
    assert captured.value.retryable
    assert registry.get_job(job.job_id).state is JobState.QUEUED

    with closing(registry._connect()) as connection:
        connection.execute(
            "UPDATE sl_registry_retention_tombstones SET deleted_at = ? "
            "WHERE deleted_at IS NULL",
            ("2026-08-08T12:00:01.000000Z",),
        )
    claimed = registry.claim(worker_id="worker-retention-barrier", lease_seconds=10)
    assert claimed is not None and claimed.lease.job_id == job.job_id


def test_artifact_registration_and_linking_reverify_actual_cas_bytes(
    registry: RunRegistry,
) -> None:
    forged = PublishedArtifact(
        "f" * 64,
        2,
        f"objects/ff/ff/{'f' * 64}",
    )
    with pytest.raises(IntegrityError, match="CAS bytes"):
        registry.register_artifact(
            forged,
            artifact_class=ArtifactClass.REPORT,
            media_type="application/json",
        )

    published = _publish(registry, b"verified before registration")
    registry.register_artifact(
        published,
        artifact_class=ArtifactClass.REPORT,
        media_type="application/json",
    )
    registry.submit(_request(), idempotency_key="tampered-link-key")
    claimed = registry.claim(worker_id="worker-tamper", lease_seconds=10)
    assert claimed is not None
    object_path = _store(registry).root / published.storage_key
    object_path.chmod(0o600)
    object_path.write_bytes(b"changed after metadata registration")

    with pytest.raises(IntegrityError, match="CAS bytes"):
        registry.complete(
            claimed.lease,
            run=_terminal_run("run-tampered-cas"),
            artifact_links=(ArtifactLink("report", published.digest),),
        )
    assert registry.get_job(claimed.lease.job_id).state is JobState.RUNNING


def test_artifact_registration_hash_does_not_hold_the_registry_writer(
    tmp_path: Path,
) -> None:
    store = ArtifactStore(tmp_path / "registration-cas", max_artifact_bytes=1_024)
    store.initialize()
    verifier = BlockingArtifactVerifier(store)
    registry = RunRegistry(
        tmp_path / "registration-registry.sqlite",
        digest_secret=SECRET,
        limits=RegistryLimits(busy_timeout_ms=0),
        clock=MutableClock(NOW),
        artifact_verifier=verifier,
    )
    registry.initialize()
    source = tmp_path / "registration-source.bin"
    source.write_bytes(b"registration hash outside writer")
    published = store.publish(source)
    results: list[ArtifactMetadata] = []
    failures: list[BaseException] = []

    def register() -> None:
        try:
            results.append(
                registry.register_artifact(
                    published,
                    artifact_class=ArtifactClass.REPORT,
                    media_type="application/octet-stream",
                )
            )
        except BaseException as exc:  # pragma: no cover - assertion reports thread failures
            failures.append(exc)

    verifier.arm()
    worker = threading.Thread(target=register)
    worker.start()
    try:
        assert verifier.entered.wait(timeout=5)
        unrelated = registry.submit(
            _request(),
            idempotency_key="writer-available-during-registration",
        )
        assert unrelated.state is JobState.QUEUED
    finally:
        verifier.release.set()
        worker.join(timeout=5)

    assert not worker.is_alive()
    assert not failures
    assert len(results) == 1 and results[0].digest == published.digest


def test_terminal_artifact_hash_does_not_hold_the_registry_writer(
    tmp_path: Path,
) -> None:
    store = ArtifactStore(tmp_path / "terminal-cas", max_artifact_bytes=1_024)
    store.initialize()
    verifier = BlockingArtifactVerifier(store)
    registry = RunRegistry(
        tmp_path / "terminal-registry.sqlite",
        digest_secret=SECRET,
        limits=RegistryLimits(busy_timeout_ms=0),
        clock=MutableClock(NOW),
        artifact_verifier=verifier,
    )
    registry.initialize()
    source = tmp_path / "terminal-source.bin"
    source.write_bytes(b"terminal hash outside writer")
    published = store.publish(source)
    registry.register_artifact(
        published,
        artifact_class=ArtifactClass.REPORT,
        media_type="application/octet-stream",
    )
    job = registry.submit(_request(), idempotency_key="terminal-preflight-job")
    claimed = registry.claim(worker_id="terminal-preflight-worker", lease_seconds=30)
    assert claimed is not None and claimed.lease.job_id == job.job_id
    results: list[JobSnapshot] = []
    failures: list[BaseException] = []

    def complete() -> None:
        try:
            results.append(
                registry.complete(
                    claimed.lease,
                    run=_terminal_run("terminal-preflight-run"),
                    artifact_links=(ArtifactLink("report", published.digest),),
                )
            )
        except BaseException as exc:  # pragma: no cover - assertion reports thread failures
            failures.append(exc)

    verifier.arm()
    worker = threading.Thread(target=complete)
    worker.start()
    try:
        assert verifier.entered.wait(timeout=5)
        unrelated = registry.submit(
            _request(2),
            idempotency_key="writer-available-during-terminal-preflight",
        )
        assert unrelated.state is JobState.QUEUED
    finally:
        verifier.release.set()
        worker.join(timeout=5)

    assert not worker.is_alive()
    assert not failures
    assert len(results) == 1 and results[0].state is JobState.SUCCEEDED


def test_heartbeats_update_one_row_without_unbounded_event_growth(
    registry: RunRegistry,
) -> None:
    job = registry.submit(_request(), idempotency_key="bounded-heartbeat-key")
    claimed = registry.claim(worker_id="worker-heartbeat", lease_seconds=10)
    assert claimed is not None

    lease = claimed.lease
    for _ in range(100):
        lease = registry.heartbeat(lease, lease_seconds=10)

    events = registry.list_events(job_id=job.job_id).items
    assert tuple(event.kind for event in events) == (EventKind.SUBMITTED, EventKind.CLAIMED)


def test_lease_expiry_requeues_then_fails_at_attempt_bound(registry: RunRegistry) -> None:
    job = registry.submit(_request(max_attempts=2), idempotency_key="expiry-attempt-key")
    first_claim = registry.claim(worker_id="worker-1", lease_seconds=1)
    assert first_claim is not None
    first = first_claim.lease

    _clock(registry).advance(1)
    assert registry.recover_expired() == 1
    assert registry.get_job(job.job_id).state is JobState.QUEUED
    with pytest.raises(LeaseLostError):
        registry.heartbeat(first, lease_seconds=1)

    second_claim = registry.claim(worker_id="worker-2", lease_seconds=1)
    assert second_claim is not None and second_claim.lease.attempt == 2
    _clock(registry).advance(1)
    assert registry.recover_expired() == 1
    exhausted = registry.get_job(job.job_id)
    assert exhausted.state is JobState.FAILED
    assert exhausted.failure_code is FailureReasonCode.LEASE_ATTEMPTS_EXHAUSTED


def test_expiry_recovery_batch_is_bounded(tmp_path: Path) -> None:
    clock = MutableClock(NOW)
    registry = RunRegistry(
        tmp_path / "recovery-bound.sqlite",
        digest_secret=SECRET,
        limits=RegistryLimits(recovery_batch_size=1, busy_timeout_ms=2_000),
        clock=clock,
    )
    registry.initialize()
    registry.submit(_request(1), idempotency_key="recovery-bound-key-1")
    first = registry.claim(worker_id="worker-1", lease_seconds=1)
    registry.submit(_request(2), idempotency_key="recovery-bound-key-2")
    second = registry.claim(worker_id="worker-2", lease_seconds=1)
    assert first is not None and second is not None

    clock.advance(1)
    assert registry.recover_expired() == 1
    assert registry.recover_expired() == 1
    assert registry.recover_expired() == 0


def test_cancel_prioritizes_its_expired_target_with_a_one_job_recovery_batch(
    tmp_path: Path,
) -> None:
    clock = MutableClock(NOW)
    registry = RunRegistry(
        tmp_path / "cancel-priority.sqlite",
        digest_secret=SECRET,
        limits=RegistryLimits(recovery_batch_size=1, busy_timeout_ms=2_000),
        clock=clock,
    )
    registry.initialize()
    running: list[JobSnapshot] = []
    for index in range(3):
        job = registry.submit(
            _request(index),
            idempotency_key=f"cancel-priority-key-{index}",
        )
        claimed = registry.claim(worker_id=f"worker-{index}", lease_seconds=1)
        assert claimed is not None and claimed.lease.job_id == job.job_id
        running.append(job)

    clock.advance(1)
    cancelled = registry.request_cancel(
        running[-1].job_id,
        reason=CancellationReasonCode.OPERATOR_REQUEST,
    )

    assert cancelled.state is JobState.CANCELLED
    assert registry.get_job(running[0].job_id).state is JobState.RUNNING
    assert registry.get_job(running[1].job_id).state is JobState.RUNNING
    assert [event.kind for event in registry.list_events(job_id=running[-1].job_id).items] == [
        EventKind.SUBMITTED,
        EventKind.CLAIMED,
        EventKind.LEASE_EXPIRED,
        EventKind.CANCEL_REQUESTED,
        EventKind.CANCELLED,
    ]


def test_retryable_failure_requeues_without_exposing_failure_on_snapshot(
    registry: RunRegistry,
) -> None:
    job = registry.submit(_request(), idempotency_key="retry-failure-key")
    claimed = registry.claim(worker_id="worker-1", lease_seconds=10)
    assert claimed is not None
    _clock(registry).advance(1)

    retried = registry.fail(
        claimed.lease,
        failure_code=FailureReasonCode.TRANSIENT_IO,
        failure_summary="bounded provider interruption",
        retryable=True,
        run=_terminal_run("run-failed-attempt-1"),
    )
    assert retried.state is JobState.QUEUED
    assert retried.failure_code is None
    assert registry.list_events(job_id=job.job_id).items[-1].kind.value == "retried"


def test_failure_taxonomy_and_summary_reject_sensitive_free_text(
    registry: RunRegistry,
) -> None:
    registry.submit(_request(), idempotency_key="safe-failure-key")
    claimed = registry.claim(worker_id="worker-safe-failure", lease_seconds=10)
    assert claimed is not None

    with pytest.raises(ValidationError, match="FailureReasonCode"):
        registry.fail(
            claimed.lease,
            failure_code="execution_error",  # type: ignore[arg-type]
            failure_summary="bounded public diagnostic",
            retryable=False,
            run=_terminal_run("run-invalid-code"),
        )
    with pytest.raises(ValidationError, match="non-sensitive"):
        registry.fail(
            claimed.lease,
            failure_code=FailureReasonCode.EXECUTION_ERROR,
            failure_summary="token abcdefghijklmnopqrstuvwxyz0123456789",
            retryable=False,
            run=_terminal_run("run-sensitive-summary"),
        )

    failed = registry.fail(
        claimed.lease,
        failure_code=FailureReasonCode.EXECUTION_ERROR,
        failure_summary="Worker exited before producing evidence.",
        retryable=False,
        run=_terminal_run("run-safe-summary"),
    )
    assert failed.failure_code is FailureReasonCode.EXECUTION_ERROR


def test_legacy_run_identifier_collision_rolls_back_terminal_transition(
    registry: RunRegistry,
) -> None:
    with closing(sqlite3.connect(registry.path)) as connection:
        connection.execute("CREATE TABLE runs(run_id TEXT PRIMARY KEY)")
        connection.execute("INSERT INTO runs(run_id) VALUES('shared-run-id')")
        connection.commit()
    job = registry.submit(_request(), idempotency_key="legacy-run-collision-key")
    claimed = registry.claim(worker_id="worker-legacy-collision", lease_seconds=10)
    assert claimed is not None

    with pytest.raises(ConflictError, match="legacy"):
        registry.complete(claimed.lease, run=_terminal_run("shared-run-id"))
    assert registry.get_job(job.job_id).state is JobState.RUNNING


def test_clock_is_internal_monotonic_and_overflow_safe(tmp_path: Path) -> None:
    clock = MutableClock(NOW)
    registry = RunRegistry(
        tmp_path / "clock.sqlite",
        digest_secret=SECRET,
        clock=clock,
    )
    registry.initialize()
    registry.submit(_request(), idempotency_key="clock-regression-key")
    clock.current = NOW - timedelta(seconds=1)
    with pytest.raises(IntegrityError, match="time authority"):
        registry.claim(worker_id="worker-clock", lease_seconds=1)

    max_clock = MutableClock(datetime.max.replace(tzinfo=UTC))
    overflow = RunRegistry(
        tmp_path / "clock-overflow.sqlite",
        digest_secret=SECRET,
        clock=max_clock,
    )
    overflow.initialize()
    overflow.submit(_request(), idempotency_key="clock-overflow-key")
    with pytest.raises(ValidationError, match="datetime range"):
        overflow.claim(worker_id="worker-overflow", lease_seconds=1)


def test_terminal_run_enforces_created_started_ended_chronology(
    registry: RunRegistry,
) -> None:
    clock = _clock(registry)
    clock.current = NOW + timedelta(seconds=10)
    job = registry.submit(_request(), idempotency_key="run-chronology-key")
    claimed = registry.claim(worker_id="worker-chronology", lease_seconds=10)
    assert claimed is not None

    with pytest.raises(ValidationError, match="chronology"):
        registry.complete(claimed.lease, run=_terminal_run("run-backdated"))
    assert registry.get_job(job.job_id).state is JobState.RUNNING


def test_terminal_run_start_defaults_to_and_cannot_precede_its_retry_claim(
    registry: RunRegistry,
) -> None:
    job = registry.submit(
        _request(max_attempts=2),
        idempotency_key="attempt-start-bound-key",
    )
    first = registry.claim(worker_id="worker-attempt-1", lease_seconds=30)
    assert first is not None
    first_claimed_at = next(
        event.occurred_at
        for event in registry.list_events(job_id=job.job_id).items
        if event.kind is EventKind.CLAIMED and event.attempt == 1
    )
    _clock(registry).advance(1)
    registry.fail(
        first.lease,
        failure_code=FailureReasonCode.TRANSIENT_IO,
        failure_summary="Bounded transient attempt failure.",
        retryable=True,
        run=_terminal_run("run-attempt-1", started_at=None),
    )

    _clock(registry).advance(5)
    second = registry.claim(worker_id="worker-attempt-2", lease_seconds=30)
    assert second is not None and second.lease.attempt == 2
    second_claimed_at = next(
        event.occurred_at
        for event in registry.list_events(job_id=job.job_id).items
        if event.kind is EventKind.CLAIMED and event.attempt == 2
    )
    assert second_claimed_at > first_claimed_at
    _clock(registry).advance(1)

    with pytest.raises(ValidationError, match="claimed_at <= started_at"):
        registry.complete(
            second.lease,
            run=_terminal_run("run-backdated-before-retry", started_at=first_claimed_at),
        )
    assert registry.get_job(job.job_id).state is JobState.RUNNING

    registry.complete(
        second.lease,
        run=_terminal_run("run-attempt-2", started_at=None),
    )
    with closing(registry._connect(readonly=True)) as connection:
        rows = connection.execute(
            "SELECT attempt, started_at FROM sl_registry_runs WHERE job_id = ? ORDER BY attempt",
            (job.job_id,),
        ).fetchall()
    assert [(int(row["attempt"]), str(row["started_at"])) for row in rows] == [
        (1, first_claimed_at.isoformat(timespec="microseconds").replace("+00:00", "Z")),
        (2, second_claimed_at.isoformat(timespec="microseconds").replace("+00:00", "Z")),
    ]


def test_terminal_transition_rejects_ambiguous_current_claim_history(
    registry: RunRegistry,
) -> None:
    job = registry.submit(_request(), idempotency_key="ambiguous-claim-history-key")
    claimed = registry.claim(worker_id="worker-ambiguous-claim", lease_seconds=30)
    assert claimed is not None
    claimed_at = registry.list_events(job_id=job.job_id).items[-1].occurred_at
    with closing(registry._connect()) as connection:
        connection.execute(
            """
            INSERT INTO sl_registry_events(
                job_id, kind, from_state, to_state, occurred_at, attempt, actor, details_json
            ) VALUES(?, 'claimed', 'queued', 'running', ?, ?, ?, '{}')
            """,
            (
                job.job_id,
                claimed_at.isoformat(timespec="microseconds").replace("+00:00", "Z"),
                claimed.lease.attempt,
                claimed.lease.worker_id,
            ),
        )

    with pytest.raises(IntegrityError, match="exactly one claim event"):
        registry.complete(
            claimed.lease,
            run=_terminal_run("run-ambiguous-claim"),
        )
    assert registry.get_job(job.job_id).state is JobState.RUNNING


def test_queued_and_running_cancellation_are_explicit(registry: RunRegistry) -> None:
    queued = registry.submit(_request(), idempotency_key="queued-cancel-key")
    cancelled = registry.request_cancel(
        queued.job_id, reason=CancellationReasonCode.OPERATOR_REQUEST
    )
    assert cancelled.state is JobState.CANCELLED
    assert [event.kind.value for event in registry.list_events(job_id=queued.job_id).items] == [
        "submitted",
        "cancel_requested",
        "cancelled",
    ]

    running = registry.submit(_request(2), idempotency_key="running-cancel-key")
    claimed = registry.claim(worker_id="worker-1", lease_seconds=10)
    assert claimed is not None and claimed.lease.job_id == running.job_id
    lease = claimed.lease
    intent = registry.request_cancel(running.job_id, reason=CancellationReasonCode.OPERATOR_REQUEST)
    assert intent.state is JobState.RUNNING
    _clock(registry).advance(1)
    renewed = registry.heartbeat(lease, lease_seconds=10)
    assert renewed.cancel_requested
    with pytest.raises(InvalidTransitionError):
        registry.complete(
            renewed,
            run=_terminal_run("run-not-successful"),
        )
    assert (
        registry.acknowledge_cancel(
            renewed,
            run=_terminal_run("run-cancelled"),
        ).state
        is JobState.CANCELLED
    )


def test_barrier_contention_has_one_claim_winner(registry: RunRegistry) -> None:
    registry.submit(_request(), idempotency_key="one-winner-key")
    barrier = threading.Barrier(3)
    results: list[object] = []
    failures: list[BaseException] = []

    def contend(worker: str) -> None:
        try:
            barrier.wait()
            results.append(registry.claim(worker_id=worker, lease_seconds=10))
        except BaseException as exc:  # pragma: no cover - assertion reports thread failures
            failures.append(exc)

    threads = [threading.Thread(target=contend, args=(f"worker-{index}",)) for index in range(2)]
    for thread in threads:
        thread.start()
    barrier.wait()
    for thread in threads:
        thread.join(timeout=5)

    assert not failures
    assert all(not thread.is_alive() for thread in threads)
    assert sum(result is not None for result in results) == 1


def test_barrier_idempotency_contention_creates_one_job(registry: RunRegistry) -> None:
    barrier = threading.Barrier(3)
    job_ids: list[str] = []
    failures: list[BaseException] = []

    def submit() -> None:
        try:
            barrier.wait()
            job_ids.append(
                registry.submit(_request(), idempotency_key="concurrent-idempotency-key").job_id
            )
        except BaseException as exc:  # pragma: no cover - assertion reports thread failures
            failures.append(exc)

    threads = [threading.Thread(target=submit) for _ in range(2)]
    for thread in threads:
        thread.start()
    barrier.wait()
    for thread in threads:
        thread.join(timeout=5)

    assert not failures
    assert len(job_ids) == 2
    assert len(set(job_ids)) == 1
    assert len(registry.list_jobs().items) == 1


@pytest.mark.parametrize(
    "operation_name",
    [
        "submit",
        "claim",
        "heartbeat",
        "cancel",
        "acknowledge",
        "complete",
        "fail",
        "register",
        "recover",
    ],
)
def test_mutation_clock_is_sampled_after_real_writer_lock_acquisition(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    operation_name: str,
) -> None:
    clock = MutableClock(NOW)
    store = ArtifactStore(tmp_path / f"cas-{operation_name}", max_artifact_bytes=1_024)
    store.initialize()
    registry = RunRegistry(
        tmp_path / f"clock-lock-{operation_name}.sqlite",
        digest_secret=SECRET,
        limits=RegistryLimits(busy_timeout_ms=2_000),
        clock=clock,
        artifact_verifier=store,
    )
    registry.initialize()
    job: JobSnapshot | None = None
    lease: Lease | None = None
    published: PublishedArtifact | None = None
    if operation_name not in {"submit", "register"}:
        job = registry.submit(
            _request(),
            idempotency_key=f"clock-lock-{operation_name}-key",
        )
    if operation_name in {"heartbeat", "acknowledge", "complete", "fail", "recover"}:
        claimed = registry.claim(
            worker_id=f"worker-{operation_name}",
            lease_seconds=1 if operation_name == "recover" else 10,
        )
        assert claimed is not None
        lease = claimed.lease
    if operation_name == "acknowledge":
        assert job is not None
        registry.request_cancel(job.job_id, reason=CancellationReasonCode.OPERATOR_REQUEST)
    if operation_name == "register":
        published = _publish(registry, b"clock-ordered-artifact")
    if operation_name == "recover":
        clock.current = NOW + timedelta(seconds=1)

    def operation() -> object:
        if operation_name == "submit":
            return registry.submit(_request(), idempotency_key="clock-lock-submit-key")
        if operation_name == "claim":
            return registry.claim(worker_id="worker-claim", lease_seconds=10)
        if operation_name == "heartbeat":
            assert lease is not None
            return registry.heartbeat(lease, lease_seconds=10)
        if operation_name == "cancel":
            assert job is not None
            return registry.request_cancel(
                job.job_id,
                reason=CancellationReasonCode.OPERATOR_REQUEST,
            )
        if operation_name == "acknowledge":
            assert lease is not None
            return registry.acknowledge_cancel(
                lease,
                run=_terminal_run("run-clock-acknowledge"),
            )
        if operation_name == "complete":
            assert lease is not None
            return registry.complete(lease, run=_terminal_run("run-clock-complete"))
        if operation_name == "fail":
            assert lease is not None
            return registry.fail(
                lease,
                failure_code=FailureReasonCode.EXECUTION_ERROR,
                failure_summary="Bounded deterministic execution failure.",
                retryable=False,
                run=_terminal_run("run-clock-fail"),
            )
        if operation_name == "register":
            assert published is not None
            return registry.register_artifact(
                published,
                artifact_class=ArtifactClass.REPORT,
                media_type="application/octet-stream",
            )
        return registry.recover_expired()

    blocker = open_database(registry.path, busy_timeout_ms=2_000)
    blocker.execute("BEGIN IMMEDIATE")
    began = threading.Event()
    original_open_database = registry_module.open_database

    def signalling_open_database(*args: object, **kwargs: object) -> BeginSignallingConnection:
        connection = original_open_database(*args, **kwargs)  # type: ignore[arg-type]
        return BeginSignallingConnection(connection, began)

    results: list[object] = []
    failures: list[BaseException] = []

    def run_operation() -> None:
        try:
            results.append(operation())
        except BaseException as exc:  # pragma: no cover - assertion reports thread failures
            failures.append(exc)

    baseline_reads = clock.reads
    try:
        with monkeypatch.context() as patcher:
            patcher.setattr(registry_module, "open_database", signalling_open_database)
            worker = threading.Thread(target=run_operation)
            worker.start()
            assert began.wait(timeout=5)
            assert clock.reads == baseline_reads
            clock.advance(1)
            expected = clock.current
            blocker.execute("ROLLBACK")
            worker.join(timeout=5)
            assert not worker.is_alive()
    finally:
        if blocker.in_transaction:
            blocker.execute("ROLLBACK")
        blocker.close()

    assert not failures
    assert len(results) == 1
    result = results[0]
    if operation_name == "submit":
        assert isinstance(result, JobSnapshot) and result.created_at == expected
    elif operation_name in {"claim", "heartbeat"}:
        observed_lease = result.lease if isinstance(result, ClaimedJob) else result
        assert isinstance(observed_lease, Lease)
        assert observed_lease.expires_at == expected + timedelta(seconds=10)
    elif operation_name == "register":
        assert isinstance(result, ArtifactMetadata) and result.created_at == expected
    elif operation_name == "recover":
        assert result == 1 and job is not None
        assert registry.get_job(job.job_id).updated_at == expected
    else:
        assert isinstance(result, JobSnapshot) and result.updated_at == expected


def test_write_lock_respects_zero_busy_bound(tmp_path: Path) -> None:
    registry = RunRegistry(
        tmp_path / "busy.sqlite",
        digest_secret=SECRET,
        limits=RegistryLimits(busy_timeout_ms=0),
    )
    registry.initialize()
    blocker = open_database(registry.path, busy_timeout_ms=0)
    blocker.execute("BEGIN IMMEDIATE")
    try:
        with pytest.raises(BusyError):
            registry.submit(_request(), idempotency_key="busy-bound-key")
    finally:
        blocker.execute("ROLLBACK")
        blocker.close()


def test_write_cleanup_preserves_primary_and_maps_cleanup_only_failure(
    registry: RunRegistry,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FaultingWriteConnection:
        def __init__(self, *, rollback_fails: bool, close_fails: bool) -> None:
            self.in_transaction = False
            self.rollback_fails = rollback_fails
            self.close_fails = close_fails
            self.close_attempted = False

        def execute(self, statement: str) -> None:
            if statement == "BEGIN IMMEDIATE":
                self.in_transaction = True
            elif statement == "COMMIT":
                self.in_transaction = False
            elif statement == "ROLLBACK":
                if self.rollback_fails:
                    raise sqlite3.OperationalError("sensitive injected rollback detail")
                self.in_transaction = False

        def close(self) -> None:
            self.close_attempted = True
            if self.close_fails:
                raise sqlite3.OperationalError("sensitive injected close detail")

    primary_connection = FaultingWriteConnection(rollback_fails=True, close_fails=True)
    monkeypatch.setattr(registry, "_connect", lambda: primary_connection)
    primary = ValidationError("typed primary")
    with pytest.raises(ValidationError) as captured, registry._write():
        raise primary

    assert captured.value is primary
    assert primary_connection.close_attempted
    assert captured.value.__notes__ == [
        "registry write cleanup also failed: OperationalError,OperationalError"
    ]
    assert "sensitive" not in str(captured.value)

    cleanup_connection = FaultingWriteConnection(rollback_fails=False, close_fails=True)
    monkeypatch.setattr(registry, "_connect", lambda: cleanup_connection)
    with pytest.raises(IntegrityError, match="cleanup failed") as cleanup, registry._write():
        pass
    assert cleanup_connection.close_attempted
    assert "sensitive" not in str(cleanup.value)


def test_sqlite_error_codes_drive_classification_and_raw_text_is_suppressed(
    tmp_path: Path,
) -> None:
    misleading = sqlite3.OperationalError(
        "database is locked by /private/attacker-controlled-trigger.sql"
    )
    misleading.sqlite_errorcode = sqlite3.SQLITE_CONSTRAINT
    assert not _is_busy(misleading)
    with pytest.raises(IntegrityError) as classified:
        RunRegistry._raise_sqlite(misleading)
    assert classified.value.__cause__ is None
    assert "attacker-controlled" not in repr(classified.value)

    extended_busy = sqlite3.OperationalError("non-sensitive provider text")
    extended_busy.sqlite_errorcode = sqlite3.SQLITE_BUSY | (1 << 8)
    assert _is_busy(extended_busy)
    with pytest.raises(BusyError) as busy:
        RunRegistry._raise_sqlite(extended_busy)
    assert busy.value.__cause__ is None

    registry = RunRegistry(tmp_path / "trigger-redaction.sqlite", digest_secret=SECRET)
    registry.initialize()
    with closing(sqlite3.connect(registry.path)) as connection:
        connection.execute("""
            CREATE TRIGGER attacker_controlled_message
            BEFORE INSERT ON sl_registry_jobs
            BEGIN
                SELECT RAISE(ABORT, 'database is locked at /private/attacker-secret');
            END
            """)
        connection.commit()
    with pytest.raises(IntegrityError) as translated:
        registry.submit(_request(), idempotency_key="trigger-redaction-key")
    assert translated.value.__cause__ is None
    assert "attacker-secret" not in str(translated.value)
    assert "attacker-secret" not in repr(translated.value)


def test_negative_boundary_errors_are_typed(registry: RunRegistry) -> None:
    with pytest.raises(NotFoundError):
        registry.get_job("missing-job")
    with pytest.raises(ValidationError):
        registry.submit(_request(), idempotency_key="short")
    with pytest.raises(ValidationError):
        registry.list_jobs(page_size=registry.limits.max_page_size + 1)
    with pytest.raises(ValidationError):
        registry.claim(worker_id="worker-1", lease_seconds=0)


def test_shared_read_contracts_are_strict_and_frozen() -> None:
    artifact = ArtifactMetadata(
        digest="a" * 64,
        artifact_class=ArtifactClass.REPORT,
        byte_size=12,
        media_type="application/json",
        storage_relpath="objects/aa/report.json",
        created_at=NOW,
    )
    plan = RetentionPlan(("a" * 64, "b" * 64), NOW, "expired unlinked evidence")
    run = RunSnapshot(
        sequence=1,
        run_id="run-1",
        job_id="a" * 32,
        attempt=1,
        status=RunStatus.SUCCEEDED,
        evidence_class=EvidenceClass.MEASURED,
        schema_version=1,
        created_at=NOW,
        started_at=NOW,
        ended_at=NOW + timedelta(seconds=1),
        source_commit="abcdef0",
        data_identity="dataset-v1",
        limitation_summary="historical evidence only",
    )

    assert artifact.storage_relpath.startswith("objects/")
    assert plan.artifact_digests == ("a" * 64, "b" * 64)
    assert run.status is RunStatus.SUCCEEDED
    assert RegistryReadiness(True, 1, "wal").ready
    assert Page((run,), None).items == (run,)
    assert "token" not in repr(ArtifactCursor(f"v1.{'a' * 16}.{'b' * 43}"))

    with pytest.raises(ValidationError, match="safe relative"):
        ArtifactMetadata(
            "a" * 64,
            ArtifactClass.REPORT,
            1,
            "application/json",
            "../escape",
            NOW,
        )
    with pytest.raises(ValidationError, match="unique and sorted"):
        RetentionPlan(("b" * 64, "a" * 64), NOW, "bad order")
    with pytest.raises(ValidationError, match="UTF-8|bounded"):
        require_failure_summary("\ud800")
    with pytest.raises(ValidationError, match="UTF-8"):
        TerminalRunRequest(
            "surrogate-run",
            EvidenceClass.MEASURED,
            source_commit="\ud800" * 7,
        )
    with pytest.raises(ValidationError, match="UTF-8"):
        RetentionPlan(("a" * 64,), NOW, "\ud800")
    with pytest.raises(ValidationError, match="ended_at"):
        RunSnapshot(
            sequence=1,
            run_id="run-1",
            job_id="a" * 32,
            attempt=1,
            status=RunStatus.SUCCEEDED,
            evidence_class=EvidenceClass.MEASURED,
            schema_version=1,
            created_at=NOW,
            started_at=NOW,
            ended_at=NOW - timedelta(seconds=1),
            source_commit=None,
            data_identity=None,
            limitation_summary=None,
        )
    with pytest.raises(ValidationError, match="terminal_at"):
        JobSnapshot(
            1,
            "a" * 32,
            "forecast",
            JobState.SUCCEEDED,
            0,
            1,
            3,
            NOW,
            NOW,
            result_run_id="run-1",
        )
    with pytest.raises(ValidationError, match="lease token"):
        Lease("a" * 32, "worker", "short", 1, NOW)
    with pytest.raises(ValidationError, match="ready verdicts"):
        RegistryReadiness(True, 1, "wal", "contradiction")
    with pytest.raises(ValidationError, match="min_lease_seconds"):
        RegistryLimits(min_lease_seconds=10, max_lease_seconds=5)
    with pytest.raises(ValidationError, match="verification_timeout_ms"):
        RegistryLimits(verification_timeout_ms=0)


def test_digest_secret_and_idempotency_key_boundaries(tmp_path: Path) -> None:
    with pytest.raises(ValidationError, match="digest_secret"):
        RunRegistry(tmp_path / "short-secret.sqlite", digest_secret=b"x" * 31)
    with pytest.raises(ValidationError, match="digest_secret"):
        RunRegistry(tmp_path / "mutable-secret.sqlite", digest_secret=bytearray(b"x" * 32))  # type: ignore[arg-type]

    registry = RunRegistry(tmp_path / "key-bounds.sqlite", digest_secret=SECRET)
    registry.initialize()
    accepted = registry.submit(_request(), idempotency_key="k" * 8)
    registry.request_cancel(accepted.job_id, reason=CancellationReasonCode.OPERATOR_REQUEST)
    assert registry.submit(_request(2), idempotency_key="k" * 256).state is JobState.QUEUED
    with pytest.raises(ValidationError, match="idempotency_key"):
        registry.submit(_request(3), idempotency_key="k" * 7)
    with pytest.raises(ValidationError, match="idempotency_key"):
        registry.submit(_request(3), idempotency_key="k" * 257)
    with pytest.raises(ValidationError, match="UTF-8"):
        registry.submit(_request(3), idempotency_key="\ud800" * 8)
    assert len(registry.list_jobs().items) == 2


def test_wrong_digest_secret_fails_startup_readiness_and_operations(tmp_path: Path) -> None:
    path = tmp_path / "key-binding.sqlite"
    owner = RunRegistry(path, digest_secret=SECRET)
    owner.initialize()
    owner.submit(_request(), idempotency_key="key-binding-job")

    wrong = RunRegistry(path, digest_secret=b"z" * 32)
    assert not wrong.probe_readiness().ready
    with pytest.raises(IntegrityError, match="authority"):
        wrong.initialize()
    with pytest.raises(IntegrityError, match="authority"):
        wrong.list_jobs()


def test_public_registry_reads_reject_real_time_and_json_storage_tampering(
    tmp_path: Path,
) -> None:
    real_registry = RunRegistry(tmp_path / "real.sqlite", digest_secret=SECRET)
    real_registry.initialize()
    real_job = real_registry.submit(_request(), idempotency_key="real-storage-tamper")
    connection = sqlite3.connect(real_registry.path)
    try:
        connection.execute(
            "UPDATE sl_registry_jobs SET attempt_count = 0.5 WHERE job_id = ?",
            (real_job.job_id,),
        )
        connection.commit()
    finally:
        connection.close()
    with pytest.raises(IntegrityError, match="exact integer"):
        real_registry.get_job(real_job.job_id)

    time_registry = RunRegistry(tmp_path / "time.sqlite", digest_secret=SECRET)
    time_registry.initialize()
    time_job = time_registry.submit(_request(), idempotency_key="time-storage-tamper")
    connection = sqlite3.connect(time_registry.path)
    try:
        connection.execute(
            "UPDATE sl_registry_jobs SET created_at = ?, updated_at = ? WHERE job_id = ?",
            ("x" * 26 + "Z", "x" * 26 + "Z", time_job.job_id),
        )
        connection.commit()
    finally:
        connection.close()
    with pytest.raises(IntegrityError, match="canonical UTC"):
        time_registry.get_job(time_job.job_id)

    request_registry = RunRegistry(tmp_path / "request.sqlite", digest_secret=SECRET)
    request_registry.initialize()
    request_job = request_registry.submit(_request(), idempotency_key="request-storage-tamper")
    deep_json = "[" * 5_000 + "]" * 5_000
    connection = sqlite3.connect(request_registry.path)
    try:
        connection.execute(
            "UPDATE sl_registry_jobs SET request_json = ?, request_digest = ? WHERE job_id = ?",
            (deep_json, hashlib.sha256(deep_json.encode()).hexdigest(), request_job.job_id),
        )
        connection.commit()
    finally:
        connection.close()
    with pytest.raises(IntegrityError, match="canonical request"):
        request_registry.claim(worker_id="bounded-reader", lease_seconds=10)

    event_registry = RunRegistry(tmp_path / "event.sqlite", digest_secret=SECRET)
    event_registry.initialize()
    event_job = event_registry.submit(_request(), idempotency_key="event-storage-tamper")
    connection = sqlite3.connect(event_registry.path)
    try:
        occurred_at = connection.execute(
            "SELECT created_at FROM sl_registry_jobs WHERE job_id = ?", (event_job.job_id,)
        ).fetchone()[0]
        connection.execute(
            """
            INSERT INTO sl_registry_events(
                job_id, kind, from_state, to_state, occurred_at,
                attempt, actor, details_json
            ) VALUES(?, 'cancel_requested', 'queued', 'queued', ?, 0, 'requester', ?)
            """,
            (
                event_job.job_id,
                occurred_at,
                '{"reason_code":"operator_request","reason_code":"shutdown"}',
            ),
        )
        connection.commit()
    finally:
        connection.close()
    with pytest.raises(IntegrityError, match="lifecycle event"):
        event_registry.list_events(job_id=event_job.job_id)
