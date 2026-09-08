"""Deterministic, fault-injected tests for digest-confirmed CAS retention."""

from __future__ import annotations

import hashlib
import json
import os
import sqlite3
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from pathlib import Path
from threading import Barrier
from typing import Any

import pytest

import quant_platform.tracking.retention as retention_module
from quant_platform.tracking.cas import (
    ArtifactGeneration,
    ArtifactStore,
    ArtifactStoreError,
    PublishedArtifact,
)
from quant_platform.tracking.contracts import (
    ArtifactClass,
    ArtifactLink,
    ArtifactMetadata,
    BusyError,
    ConflictError,
    EvidenceClass,
    IntegrityError,
    RegistryLimits,
    RetentionPendingError,
    RetentionPlan,
    SubmissionRequest,
    TerminalRunRequest,
    ValidationError,
    canonical_json,
)
from quant_platform.tracking.migrations import open_database, probe_database
from quant_platform.tracking.registry import RunRegistry
from quant_platform.tracking.retention import (
    DigestConfirmedRetentionPlan,
    RetentionActiveWorkError,
    RetentionCandidate,
    RetentionConfirmationError,
    RetentionController,
    RetentionDisabledError,
    RetentionDriftError,
    RetentionExecution,
    RetentionIntegrityError,
    RetentionIntent,
    RetentionPartialFailure,
    RetentionPolicy,
)

# Keep the deterministic registry clock beyond any physical test-file generation. This makes the
# normal fixture satisfy the independent CAS mtime/ctime grace requirement without wall-clock sleeps.
NOW = datetime(2030, 8, 8, 12, tzinfo=UTC)
OLD = NOW - timedelta(days=30)
SECRET = b"retention-test-secret-material!!"


class _MutableClock:
    """Deterministic test-only implementation of the registry clock authority."""

    def __init__(self, instant: datetime) -> None:
        self._instant = instant

    def now(self) -> datetime:
        return self._instant

    def set(self, instant: datetime) -> None:
        self._instant = instant


class _CleanupFaultConnection:
    """Minimal SQLite-shaped fault injector for connection lifecycle assertions."""

    def __init__(self, *, fail_progress: bool, fail_close: bool) -> None:
        self.fail_progress = fail_progress
        self.fail_close = fail_close
        self.events: list[str] = []

    def set_progress_handler(self, callback: object, instructions: int) -> None:
        del callback, instructions
        self.events.append("remove-progress-handler")
        if self.fail_progress:
            raise sqlite3.OperationalError("sensitive injected progress cleanup detail")

    def close(self) -> None:
        self.events.append("close")
        if self.fail_close:
            raise sqlite3.OperationalError("sensitive injected close detail")


class _TestRegistry(RunRegistry):
    """RunRegistry with an explicit, test-bound time-control surface."""

    def __init__(
        self,
        path: Path,
        *,
        artifact_verifier: ArtifactStore,
        limits: RegistryLimits | None = None,
    ) -> None:
        self._test_clock = _MutableClock(NOW)
        super().__init__(
            path,
            digest_secret=SECRET,
            limits=limits,
            clock=self._test_clock,
            artifact_verifier=artifact_verifier,
        )

    def set_time(self, instant: datetime) -> None:
        """Set the next deterministic authority instant for a registry operation."""

        self._test_clock.set(instant)


@pytest.fixture
def store(tmp_path: Path) -> ArtifactStore:
    instance = ArtifactStore(tmp_path / "cas", max_artifact_bytes=1_024)
    instance.initialize()
    return instance


@pytest.fixture
def registry(tmp_path: Path, store: ArtifactStore) -> RunRegistry:
    # Concurrency tests exercise idempotency after a legitimate writer releases its lock. Give
    # those tests the same bounded two-second scheduling allowance as the registry concurrency
    # suite; zero-wait fail-closed behavior is covered independently below with an explicit limit.
    instance = _TestRegistry(
        tmp_path / "registry.sqlite",
        artifact_verifier=store,
        limits=RegistryLimits(busy_timeout_ms=2_000),
    )
    instance.initialize()
    return instance


def _set_registry_time(registry: RunRegistry, instant: datetime) -> None:
    if not isinstance(registry, _TestRegistry):
        raise AssertionError("retention tests require the deterministic registry fixture")
    registry.set_time(instant)


def _enabled(
    registry: RunRegistry,
    store: ArtifactStore,
    *,
    grace_period_seconds: int = 24 * 60 * 60,
    max_candidates: int = 100,
    max_total_bytes: int = 1_024,
) -> RetentionController:
    if isinstance(registry, _TestRegistry):
        registry.set_time(NOW)
    return RetentionController(
        registry,
        store,
        policy=RetentionPolicy(
            enabled=True,
            grace_period_seconds=grace_period_seconds,
            max_candidates=max_candidates,
            max_total_bytes=max_total_bytes,
        ),
    )


def _register(
    registry: RunRegistry,
    store: ArtifactStore,
    tmp_path: Path,
    payload: bytes,
    *,
    created_at: datetime = OLD,
    pinned: bool = False,
    artifact_class: ArtifactClass = ArtifactClass.OUTPUT,
) -> PublishedArtifact:
    digest = hashlib.sha256(payload).hexdigest()
    source = tmp_path / f"source-{digest}.bin"
    source.write_bytes(payload)
    published = store.publish(source)
    _set_registry_time(registry, created_at)
    registry.register_artifact(
        published,
        artifact_class=artifact_class,
        media_type="application/octet-stream",
        pinned=pinned,
    )
    return published


def _link_artifact(
    registry: RunRegistry,
    artifact: PublishedArtifact,
    *,
    suffix: str,
    now: datetime = OLD,
) -> None:
    _set_registry_time(registry, now)
    registry.submit(
        SubmissionRequest(kind="retention-test", payload={"suffix": suffix}),
        idempotency_key=f"retention-idempotency-{suffix}",
    )
    _set_registry_time(registry, now + timedelta(seconds=1))
    claimed = registry.claim(worker_id=f"worker-{suffix}", lease_seconds=60)
    assert claimed is not None
    _set_registry_time(registry, now + timedelta(seconds=3))
    registry.complete(
        claimed.lease,
        run=TerminalRunRequest(
            run_id=f"run-{suffix}",
            evidence_class=EvidenceClass.SIMULATED,
            started_at=now + timedelta(seconds=2),
            limitation_summary="synthetic retention fixture only",
        ),
        artifact_links=(ArtifactLink("output", artifact.digest),),
    )


def _tombstones(registry: RunRegistry) -> list[sqlite3.Row]:
    connection = registry._connect(readonly=True)
    try:
        return connection.execute("""
            SELECT artifact_digest, plan_digest, planned_at, deleted_at, reason
            FROM sl_registry_retention_tombstones
            ORDER BY artifact_digest
            """).fetchall()
    finally:
        connection.close()


def _count(registry: RunRegistry, table: str) -> int:
    allowed = {
        "sl_registry_artifacts",
        "sl_registry_jobs",
        "sl_registry_runs",
        "sl_registry_events",
        "sl_registry_run_artifacts",
        "sl_registry_retention_plans",
        "sl_registry_retention_tombstones",
    }
    if table not in allowed:
        raise AssertionError("test requested an unapproved table")
    connection = registry._connect(readonly=True)
    try:
        return int(connection.execute(f'SELECT count(*) FROM "{table}"').fetchone()[0])
    finally:
        connection.close()


def _insert_plan_envelope(
    connection: sqlite3.Connection,
    plan: DigestConfirmedRetentionPlan,
) -> None:
    """Insert one otherwise-valid signed envelope for adversarial row fixtures."""

    connection.execute(
        """
        INSERT INTO sl_registry_retention_plans(
            plan_digest, payload_digest, payload_json, registry_id,
            cas_store_id, planned_at, schema_version
        ) VALUES(?, ?, ?, ?, ?, ?, ?)
        """,
        (
            plan.digest,
            plan.payload_digest,
            plan.canonical_payload_bytes().decode("utf-8"),
            plan.registry_id,
            plan.cas_store_id,
            plan.plan.planned_at.isoformat(timespec="microseconds").replace("+00:00", "Z"),
            plan.schema_version,
        ),
    )


def test_retention_is_disabled_by_default(
    registry: RunRegistry,
    store: ArtifactStore,
    tmp_path: Path,
) -> None:
    artifact = _register(registry, store, tmp_path, b"disabled")
    controller = RetentionController(registry, store)

    with pytest.raises(RetentionDisabledError):
        controller.plan(reason="disabled-policy-test")

    enabled = _enabled(registry, store)
    plan = enabled.plan(reason="disabled-policy-test")
    assert plan is not None
    with pytest.raises(RetentionDisabledError):
        controller.execute(plan, confirmed_digest=plan.digest)

    store.verify(artifact)
    assert _tombstones(registry) == []


@pytest.mark.parametrize(
    "kwargs",
    [
        {"enabled": 1},
        {"grace_period_seconds": 0},
        {"max_candidates": 0},
        {"max_total_bytes": 0},
        {"max_total_bytes": True},
    ],
)
def test_policy_rejects_unsafe_or_unbounded_values(kwargs: dict[str, Any]) -> None:
    with pytest.raises(ValidationError):
        RetentionPolicy(**kwargs)


def test_connection_cleanup_attempts_every_action_and_maps_cleanup_only_failure() -> None:
    connection = _CleanupFaultConnection(fail_progress=True, fail_close=True)

    with pytest.raises(RetentionIntegrityError, match="connection cleanup failed") as captured:
        retention_module._close_connection(connection)  # type: ignore[arg-type]

    assert connection.events == ["remove-progress-handler", "close"]
    assert captured.value.__cause__ is None
    assert captured.value.__context__ is None
    assert "sensitive" not in str(captured.value)


def test_connection_cleanup_preserves_primary_and_adds_sanitized_failure_note() -> None:
    connection = _CleanupFaultConnection(fail_progress=True, fail_close=True)
    primary = RetentionDriftError("authoritative primary failure")

    with pytest.raises(RetentionDriftError) as captured:
        try:
            raise primary
        finally:
            retention_module._close_connection(connection)  # type: ignore[arg-type]

    assert captured.value is primary
    assert connection.events == ["remove-progress-handler", "close"]
    notes = getattr(captured.value, "__notes__", [])
    assert notes == [
        "retention connection cleanup also failed closed: "
        "progress-handler removal (OperationalError), connection close (OperationalError)"
    ]
    assert "sensitive" not in " ".join(notes)


def test_plan_returns_none_without_old_unlinked_candidates(
    registry: RunRegistry,
    store: ArtifactStore,
    tmp_path: Path,
) -> None:
    _register(registry, store, tmp_path, b"too-young", created_at=NOW)
    controller = _enabled(registry, store)

    assert controller.plan(reason="nothing-eligible") is None
    with pytest.raises(ValidationError, match="reason"):
        controller.plan(reason="")
    with pytest.raises(ValidationError, match="policy identifier"):
        controller.plan(reason="../operator/path")
    with pytest.raises(ValidationError, match="policy identifier"):
        controller.plan(reason="credential value")


def test_plan_stops_before_crossing_aggregate_byte_bound(
    registry: RunRegistry,
    store: ArtifactStore,
    tmp_path: Path,
) -> None:
    artifacts = (
        _register(registry, store, tmp_path, b"12"),
        _register(registry, store, tmp_path, b"34"),
    )
    plan = _enabled(registry, store, max_total_bytes=3).plan(reason="aggregate-bound")

    assert plan is not None
    assert len(plan.candidates) == 1
    assert plan.total_bytes == 2
    assert plan.plan.artifact_digests[0] in {artifact.digest for artifact in artifacts}


def test_plan_is_deterministic_bounded_and_preserves_ineligible_evidence(
    registry: RunRegistry,
    store: ArtifactStore,
    tmp_path: Path,
) -> None:
    eligible_a = _register(registry, store, tmp_path, b"a")
    eligible_b = _register(registry, store, tmp_path, b"bb")
    pinned = _register(registry, store, tmp_path, b"pinned", pinned=True)
    young = _register(registry, store, tmp_path, b"young", created_at=NOW)
    linked = _register(registry, store, tmp_path, b"linked")
    _link_artifact(registry, linked, suffix="linked")
    controller = _enabled(registry, store, max_total_bytes=3)

    first = controller.plan(reason="expired-unlinked-objects")
    second = controller.plan(reason="expired-unlinked-objects")

    assert first is not None
    assert second == first
    assert first.digest == second.digest
    assert len(first.policy_digest) == 64
    assert first.total_bytes == 3
    assert first.plan.artifact_digests == tuple(sorted((eligible_a.digest, eligible_b.digest)))
    assert tuple(candidate.digest for candidate in first.candidates) == first.plan.artifact_digests
    assert first.eligible_before == NOW - timedelta(days=1)
    different = DigestConfirmedRetentionPlan.create(
        plan=replace(first.plan, reason="different-bounded-reason"),
        candidates=first.candidates,
        eligible_before=first.eligible_before,
        policy=controller.policy,
        registry=registry,
        cas_store_id=store.store_id,
    )
    assert first.digest != different.digest
    assert first.payload_digest != different.payload_digest
    for artifact in (pinned, young, linked):
        store.verify(artifact)


def test_stage_requires_exact_confirmation_and_commits_bytes_free_intent(
    registry: RunRegistry,
    store: ArtifactStore,
    tmp_path: Path,
) -> None:
    artifacts = (
        _register(registry, store, tmp_path, b"stage-a"),
        _register(registry, store, tmp_path, b"stage-b"),
    )
    controller = _enabled(registry, store)
    plan = controller.plan(reason="stage-boundary")
    assert plan is not None

    with pytest.raises(RetentionConfirmationError):
        controller.stage(plan, confirmed_digest="0" * 64)
    assert _tombstones(registry) == []

    intent = controller.stage(plan, confirmed_digest=plan.digest)
    assert intent.pending_digests == plan.plan.artifact_digests
    assert intent.finalized_digests == ()
    assert controller.stage(plan, confirmed_digest=plan.digest) == intent
    rows = _tombstones(registry)
    assert {row["plan_digest"] for row in rows} == {plan.digest}
    assert all(row["deleted_at"] is None for row in rows)
    for artifact in artifacts:
        store.verify(artifact)


def test_plan_contract_and_confirmation_reject_forgery(
    registry: RunRegistry,
    store: ArtifactStore,
    tmp_path: Path,
) -> None:
    for payload in (b"contract-a", b"contract-b"):
        _register(registry, store, tmp_path, payload)
    controller = _enabled(registry, store)
    plan = controller.plan(reason="contract-validation")
    assert plan is not None

    with pytest.raises(ValidationError, match="plan must"):
        replace(plan, plan="invalid")  # type: ignore[arg-type]
    with pytest.raises(ValidationError, match="non-empty tuple"):
        replace(plan, candidates=[])  # type: ignore[arg-type]
    with pytest.raises(ValidationError, match="may not exceed"):
        replace(plan, candidates=(plan.candidates[0],) * 1_001)
    with pytest.raises(ValidationError, match="RetentionCandidate"):
        replace(plan, candidates=(object(),))  # type: ignore[arg-type]
    with pytest.raises(ValidationError, match="unique and sorted"):
        replace(plan, candidates=(plan.candidates[0], plan.candidates[0]))
    with pytest.raises(ValidationError, match="sorted"):
        replace(
            plan,
            plan=RetentionPlan(
                tuple(reversed(plan.plan.artifact_digests)),
                NOW,
                "invalid-order",
            ),
            candidates=tuple(reversed(plan.candidates)),
        )
    with pytest.raises(ValidationError, match="exactly match"):
        replace(plan, candidates=plan.candidates[:1])
    pinned_metadata = replace(plan.candidates[0].metadata, pinned=True)
    with pytest.raises(ValidationError, match="pinned"):
        RetentionCandidate(
            pinned_metadata,
            plan.candidates[0].last_changed_at,
            plan.candidates[0].last_changed_ns,
            plan.candidates[0].generation,
        )
    with pytest.raises(ValidationError, match="follow planned_at"):
        replace(plan, eligible_before=NOW + timedelta(seconds=1))
    with pytest.raises(ValidationError, match="schema_version"):
        replace(plan, schema_version=2)
    with pytest.raises(ValidationError, match="payload_digest"):
        replace(plan, payload_digest="0" * 64)
    with pytest.raises(ValidationError, match="RetentionPolicy"):
        DigestConfirmedRetentionPlan.create(
            plan=plan.plan,
            candidates=plan.candidates,
            eligible_before=plan.eligible_before,
            policy=object(),  # type: ignore[arg-type]
            registry=registry,
            cas_store_id=store.store_id,
        )
    forged_signature = replace(plan, digest="0" * 64)
    with pytest.raises(RetentionConfirmationError, match="signature"):
        controller.stage(forged_signature, confirmed_digest=forged_signature.digest)
    with pytest.raises(RetentionConfirmationError):
        controller.stage(plan, confirmed_digest="not-a-digest")
    with pytest.raises(ValidationError, match="plan must"):
        controller.stage(object(), confirmed_digest=plan.digest)  # type: ignore[arg-type]


def test_planning_fails_closed_when_clock_underflows_the_grace_period(
    registry: RunRegistry,
    store: ArtifactStore,
) -> None:
    controller = _enabled(registry, store)
    _set_registry_time(registry, datetime.min.replace(tzinfo=UTC))

    with pytest.raises(ValidationError, match="too early"):
        controller.plan(reason="clock-underflow")


def test_confirmation_rejects_plan_from_different_policy(
    registry: RunRegistry,
    store: ArtifactStore,
    tmp_path: Path,
) -> None:
    for payload in (b"policy-a", b"policy-b"):
        _register(registry, store, tmp_path, payload)
    source_controller = _enabled(registry, store, max_candidates=2, max_total_bytes=1_024)
    plan = source_controller.plan(reason="policy-binding")
    assert plan is not None

    with pytest.raises(RetentionConfirmationError, match="candidate bound"):
        _enabled(registry, store, max_candidates=1).stage(
            plan,
            confirmed_digest=plan.digest,
        )
    with pytest.raises(RetentionConfirmationError, match="byte bound"):
        _enabled(registry, store, max_total_bytes=1).stage(
            plan,
            confirmed_digest=plan.digest,
        )
    with pytest.raises(RetentionConfirmationError, match="grace period"):
        _enabled(registry, store, grace_period_seconds=2 * 24 * 60 * 60).stage(
            plan,
            confirmed_digest=plan.digest,
        )
    earliest = datetime.min.replace(tzinfo=UTC)
    earliest_plan = DigestConfirmedRetentionPlan.create(
        plan=RetentionPlan(
            plan.plan.artifact_digests,
            earliest,
            "policy-overflow",
        ),
        candidates=plan.candidates,
        eligible_before=earliest,
        policy=source_controller.policy,
        registry=registry,
        cas_store_id=store.store_id,
    )
    with pytest.raises(RetentionConfirmationError, match="timestamp"):
        source_controller.stage(earliest_plan, confirmed_digest=earliest_plan.digest)
    _set_registry_time(registry, NOW - timedelta(seconds=1))
    with pytest.raises(RetentionConfirmationError, match="future"):
        source_controller.stage(plan, confirmed_digest=plan.digest)
    _set_registry_time(registry, NOW)
    assert all(row["deleted_at"] is None for row in _tombstones(registry))


def test_forged_young_plan_cannot_weaken_grace_period(
    registry: RunRegistry,
    store: ArtifactStore,
    tmp_path: Path,
) -> None:
    published = _register(registry, store, tmp_path, b"young-forgery", created_at=NOW)
    controller = _enabled(registry, store)
    assert controller.plan(reason="young-forgery") is None
    candidate = ArtifactMetadata(
        digest=published.digest,
        artifact_class=ArtifactClass.OUTPUT,
        byte_size=published.byte_size,
        media_type="application/octet-stream",
        storage_relpath=published.storage_key,
        created_at=NOW,
    )
    future = NOW + timedelta(days=2)
    forged_contract = RetentionPlan(
        (published.digest,),
        future,
        "young-forgery",
    )
    inventory = store.inspect(published)
    forged_candidate = RetentionCandidate(
        candidate,
        inventory.last_changed_at,
        inventory.last_changed_ns,
        inventory.generation,
    )
    forged = DigestConfirmedRetentionPlan.create(
        plan=forged_contract,
        candidates=(forged_candidate,),
        eligible_before=future - timedelta(days=1),
        policy=controller.policy,
        registry=registry,
        cas_store_id=store.store_id,
    )

    with pytest.raises(RetentionConfirmationError, match="future"):
        controller.stage(forged, confirmed_digest=forged.digest)
    store.verify(published)
    assert _tombstones(registry) == []


def test_running_work_blocks_planning_and_new_intent(
    registry: RunRegistry,
    store: ArtifactStore,
    tmp_path: Path,
) -> None:
    artifact = _register(registry, store, tmp_path, b"active-evidence")
    controller = _enabled(registry, store)
    plan = controller.plan(reason="active-evidence")
    assert plan is not None
    _set_registry_time(registry, NOW)
    registry.submit(
        SubmissionRequest(kind="retention-test", payload={}),
        idempotency_key="retention-idempotency-active-evidence",
    )
    _set_registry_time(registry, NOW + timedelta(seconds=1))
    claimed = registry.claim(worker_id="worker-active-evidence", lease_seconds=60)
    assert claimed is not None

    with pytest.raises(RetentionActiveWorkError):
        controller.plan(reason="active-evidence")
    with pytest.raises(RetentionActiveWorkError):
        controller.stage(plan, confirmed_digest=plan.digest)

    store.verify(artifact)
    assert _tombstones(registry) == []


def test_phase_a_fault_rolls_back_the_entire_plan(
    registry: RunRegistry,
    store: ArtifactStore,
    tmp_path: Path,
) -> None:
    _register(registry, store, tmp_path, b"phase-a-first")
    _register(registry, store, tmp_path, b"phase-a-second")
    controller = _enabled(registry, store)
    plan = controller.plan(reason="phase-a-atomicity")
    assert plan is not None
    failed_digest = plan.plan.artifact_digests[-1]
    connection = open_database(registry.path, busy_timeout_ms=registry.limits.busy_timeout_ms)
    try:
        connection.execute(f"""
            CREATE TRIGGER test_retention_phase_a_failure
            BEFORE INSERT ON sl_registry_retention_tombstones
            WHEN NEW.artifact_digest = '{failed_digest}'
            BEGIN
                SELECT RAISE(ABORT, 'injected phase-a failure');
            END;
            """)
    finally:
        connection.close()

    with pytest.raises(RetentionIntegrityError):
        controller.stage(plan, confirmed_digest=plan.digest)
    assert _tombstones(registry) == []

    connection = open_database(registry.path, busy_timeout_ms=registry.limits.busy_timeout_ms)
    try:
        connection.execute("DROP TRIGGER test_retention_phase_a_failure")
    finally:
        connection.close()
    assert controller.stage(plan, confirmed_digest=plan.digest).pending_digests


def test_candidate_drift_aborts_before_durable_intent_or_unlink(
    registry: RunRegistry,
    store: ArtifactStore,
    tmp_path: Path,
) -> None:
    artifact = _register(registry, store, tmp_path, b"drift")
    controller = _enabled(registry, store)
    plan = controller.plan(reason="drift-rejection")
    assert plan is not None
    _link_artifact(registry, artifact, suffix="drift")
    _set_registry_time(registry, NOW)

    with pytest.raises(RetentionDriftError):
        controller.stage(plan, confirmed_digest=plan.digest)

    assert _tombstones(registry) == []
    store.verify(artifact)


def test_stage_rejects_another_plan_and_mixed_plan_cohort(
    registry: RunRegistry,
    store: ArtifactStore,
    tmp_path: Path,
) -> None:
    first = _register(registry, store, tmp_path, b"other-plan-first")
    _register(registry, store, tmp_path, b"other-plan-second")
    controller = _enabled(registry, store, max_candidates=1)
    plan = controller.plan(reason="other-plan")
    assert plan is not None
    controller.stage(plan, confirmed_digest=plan.digest)
    competing_contract = replace(plan.plan, reason="different-plan")
    competing = DigestConfirmedRetentionPlan.create(
        plan=competing_contract,
        candidates=plan.candidates,
        eligible_before=plan.eligible_before,
        policy=controller.policy,
        registry=registry,
        cas_store_id=store.store_id,
    )

    with pytest.raises(RetentionDriftError, match="another durable plan"):
        controller.stage(competing, confirmed_digest=competing.digest)
    store.verify(first)

    # A second isolated registry proves an extra row under the same digest also fails closed.
    other_root = tmp_path / "mixed"
    other_store = ArtifactStore(other_root / "cas", max_artifact_bytes=1_024)
    other_store.initialize()
    other_registry = _TestRegistry(
        other_root / "registry.sqlite",
        artifact_verifier=other_store,
    )
    other_registry.initialize()
    mixed_artifacts = (
        _register(other_registry, other_store, other_root, b"mixed-first"),
        _register(other_registry, other_store, other_root, b"mixed-extra"),
    )
    other_controller = _enabled(other_registry, other_store, max_candidates=1)
    other_plan = other_controller.plan(reason="mixed-cohort")
    assert other_plan is not None
    other_controller.stage(other_plan, confirmed_digest=other_plan.digest)
    extra = next(
        artifact
        for artifact in mixed_artifacts
        if artifact.digest not in other_plan.plan.artifact_digests
    )
    connection = open_database(
        other_registry.path,
        busy_timeout_ms=other_registry.limits.busy_timeout_ms,
    )
    try:
        connection.execute(
            """
            INSERT INTO sl_registry_retention_tombstones(
                artifact_digest, plan_digest, planned_at, deleted_at, reason
            ) VALUES(?, ?, ?, NULL, ?)
            """,
            (
                extra.digest,
                other_plan.digest,
                NOW.isoformat(timespec="microseconds").replace("+00:00", "Z"),
                other_plan.plan.reason,
            ),
        )
    finally:
        connection.close()
    with pytest.raises(RetentionDriftError, match="candidate set"):
        other_controller.stage(other_plan, confirmed_digest=other_plan.digest)


def test_plan_rejects_noncanonical_stored_cas_key(
    registry: RunRegistry,
    store: ArtifactStore,
) -> None:
    digest = hashlib.sha256(b"not-published").hexdigest()
    connection = open_database(registry.path, busy_timeout_ms=registry.limits.busy_timeout_ms)
    try:
        connection.execute(
            """
            INSERT INTO sl_registry_artifacts(
                digest, artifact_class, byte_size, media_type, storage_relpath,
                created_at, pinned
            ) VALUES(?, 'output', 1, 'application/octet-stream', ?, ?, 0)
            """,
            (
                digest,
                "objects/aa/not-the-content-digest",
                OLD.isoformat(timespec="microseconds").replace("+00:00", "Z"),
            ),
        )
    finally:
        connection.close()

    with pytest.raises(RetentionIntegrityError, match="canonical"):
        _enabled(registry, store).plan(reason="invalid-key")


def test_plan_rejects_permissive_sqlite_numeric_storage(
    registry: RunRegistry,
    store: ArtifactStore,
) -> None:
    digest = hashlib.sha256(b"non-integer-size").hexdigest()
    connection = open_database(registry.path, busy_timeout_ms=registry.limits.busy_timeout_ms)
    try:
        connection.execute(
            """
            INSERT INTO sl_registry_artifacts(
                digest, artifact_class, byte_size, media_type, storage_relpath,
                created_at, pinned
            ) VALUES(?, 'output', 1.5, 'application/octet-stream', ?, ?, 0)
            """,
            (
                digest,
                f"objects/{digest[:2]}/{digest[2:4]}/{digest}",
                OLD.isoformat(timespec="microseconds").replace("+00:00", "Z"),
            ),
        )
    finally:
        connection.close()

    with pytest.raises(RetentionIntegrityError, match="metadata"):
        _enabled(registry, store).plan(reason="invalid-sqlite-type")


def test_plan_rejects_structurally_plausible_invalid_database_timestamp(
    registry: RunRegistry,
    store: ArtifactStore,
) -> None:
    digest = hashlib.sha256(b"invalid-created-at").hexdigest()
    connection = open_database(registry.path, busy_timeout_ms=registry.limits.busy_timeout_ms)
    try:
        connection.execute(
            """
            INSERT INTO sl_registry_artifacts(
                digest, artifact_class, byte_size, media_type, storage_relpath,
                created_at, pinned
            ) VALUES(?, 'output', 1, 'application/octet-stream', ?, ?, 0)
            """,
            (
                digest,
                f"objects/{digest[:2]}/{digest[2:4]}/{digest}",
                "0000-00-00T00:00:00.000000Z",
            ),
        )
    finally:
        connection.close()

    with pytest.raises(RetentionIntegrityError, match="ISO-8601"):
        _enabled(registry, store).plan(reason="invalid-created-at")


def test_stage_rejects_noncanonical_persisted_tombstone_fields(
    registry: RunRegistry,
    store: ArtifactStore,
    tmp_path: Path,
) -> None:
    artifact = _register(registry, store, tmp_path, b"invalid-tombstone")
    controller = _enabled(registry, store)
    plan = controller.plan(reason="strict-tombstone")
    assert plan is not None
    connection = open_database(registry.path, busy_timeout_ms=registry.limits.busy_timeout_ms)
    try:
        _insert_plan_envelope(connection, plan)
        connection.execute(
            """
            INSERT INTO sl_registry_retention_tombstones(
                artifact_digest, plan_digest, planned_at, deleted_at, reason
            ) VALUES(?, ?, ?, NULL, ?)
            """,
            (
                artifact.digest,
                plan.digest,
                NOW.isoformat(timespec="microseconds").replace("+00:00", "Z"),
                "../private/path",
            ),
        )
    finally:
        connection.close()

    with pytest.raises(RetentionIntegrityError, match="tombstone"):
        controller.stage(plan, confirmed_digest=plan.digest)
    store.verify(artifact)


def test_execute_retains_all_metadata_and_is_idempotent_for_exact_plan(
    registry: RunRegistry,
    store: ArtifactStore,
    tmp_path: Path,
) -> None:
    outside = tmp_path / "outside-cas.txt"
    outside.write_text("must remain", encoding="utf-8")
    artifacts = (
        _register(registry, store, tmp_path, b"delete-a"),
        _register(registry, store, tmp_path, b"delete-b"),
    )
    linked = _register(registry, store, tmp_path, b"retain-linked")
    _link_artifact(registry, linked, suffix="retained")
    before_counts = {
        table: _count(registry, table)
        for table in (
            "sl_registry_artifacts",
            "sl_registry_jobs",
            "sl_registry_runs",
            "sl_registry_events",
            "sl_registry_run_artifacts",
        )
    }
    controller = _enabled(registry, store)
    plan = controller.plan(reason="verified-unlinked-cleanup")
    assert plan is not None

    result = controller.execute(plan, confirmed_digest=plan.digest)

    assert result.newly_deleted_digests == plan.plan.artifact_digests
    assert result.previously_deleted_digests == ()
    assert result.deleted_bytes == sum(artifact.byte_size for artifact in artifacts)
    for artifact in artifacts:
        with pytest.raises(ArtifactStoreError):
            store.verify(artifact)
    store.verify(linked)
    assert outside.read_text(encoding="utf-8") == "must remain"
    assert {table: _count(registry, table) for table in before_counts} == before_counts
    rows = _tombstones(registry)
    assert all(row["deleted_at"] is not None for row in rows)

    replay = controller.execute(plan, confirmed_digest=plan.digest)
    assert replay.newly_deleted_digests == ()
    assert replay.previously_deleted_digests == plan.plan.artifact_digests
    assert replay.deleted_bytes == 0


def test_preflight_corruption_prevents_every_unlink(
    registry: RunRegistry,
    store: ArtifactStore,
    tmp_path: Path,
) -> None:
    artifacts = (
        _register(registry, store, tmp_path, b"healthy-object"),
        _register(registry, store, tmp_path, b"corrupt-object"),
    )
    controller = _enabled(registry, store)
    plan = controller.plan(reason="preflight-all-objects")
    assert plan is not None
    corrupt = artifacts[1]
    corrupt_path = store.root.joinpath(*corrupt.storage_key.split("/"))
    corrupt_path.chmod(0o600)
    corrupt_path.write_bytes(b"substituted")
    corrupt_path.chmod(0o400)

    with pytest.raises(RetentionDriftError, match="generation preflight"):
        controller.execute(plan, confirmed_digest=plan.digest)

    healthy = next(artifact for artifact in artifacts if artifact != corrupt)
    store.verify(healthy)
    assert corrupt_path.exists()
    assert _count(registry, "sl_registry_retention_plans") == 0
    assert _tombstones(registry) == []


def test_phase_b_stops_on_first_failure_and_resumes_pending_candidates(
    registry: RunRegistry,
    store: ArtifactStore,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    for payload in (b"resume-one", b"resume-two", b"resume-three"):
        _register(registry, store, tmp_path, payload)
    controller = _enabled(registry, store)
    plan = controller.plan(reason="bounded-resumable-failure")
    assert plan is not None
    original_unlink = store.unlink_verified
    calls: list[str] = []

    def fail_second(
        artifact: PublishedArtifact,
        *,
        expected_generation: ArtifactGeneration | None = None,
    ) -> None:
        calls.append(artifact.digest)
        if len(calls) == 2:
            raise ArtifactStoreError("injected bounded unlink failure")
        original_unlink(
            artifact,
            expected_generation=expected_generation,
        )

    monkeypatch.setattr(store, "unlink_verified", fail_second)
    with pytest.raises(RetentionPartialFailure) as captured:
        controller.execute(plan, confirmed_digest=plan.digest)

    first = plan.plan.artifact_digests[0]
    assert calls == list(plan.plan.artifact_digests[:2])
    assert captured.value.finalized_digests == (first,)
    assert captured.value.pending_digests == plan.plan.artifact_digests[1:]
    monkeypatch.setattr(store, "unlink_verified", original_unlink)

    resumed = controller.execute(
        plan,
        confirmed_digest=plan.digest,
    )
    assert resumed.newly_deleted_digests == plan.plan.artifact_digests[1:]
    assert resumed.previously_deleted_digests == (first,)
    assert all(row["deleted_at"] is not None for row in _tombstones(registry))


def test_phase_b_registry_failure_is_machine_coded_and_preserves_bytes(
    registry: RunRegistry,
    store: ArtifactStore,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    artifact = _register(registry, store, tmp_path, b"registry-phase-b-failure")
    controller = _enabled(registry, store)
    plan = controller.plan(reason="phase-b-registry-failure")
    assert plan is not None

    def fail_closed(
        _: DigestConfirmedRetentionPlan,
        __: RetentionCandidate,
    ) -> bool:
        raise RetentionDriftError("injected registry drift")

    monkeypatch.setattr(controller, "_finalize_candidate", fail_closed)
    with pytest.raises(RetentionPartialFailure) as captured:
        controller.execute(plan, confirmed_digest=plan.digest)

    assert captured.value.cause_code == RetentionDriftError.code
    assert captured.value.pending_digests == plan.plan.artifact_digests
    store.verify(artifact)


def test_phase_b_refuses_to_report_success_without_finalized_tombstones(
    registry: RunRegistry,
    store: ArtifactStore,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    artifact = _register(registry, store, tmp_path, b"incomplete-finalization")
    controller = _enabled(registry, store)
    plan = controller.plan(reason="incomplete-finalization")
    assert plan is not None

    def leave_pending(
        _: DigestConfirmedRetentionPlan,
        __: RetentionCandidate,
    ) -> bool:
        return False

    monkeypatch.setattr(controller, "_finalize_candidate", leave_pending)
    with pytest.raises(RetentionPartialFailure) as captured:
        controller.execute(plan, confirmed_digest=plan.digest)

    assert captured.value.cause_code == "retention_finalize_incomplete"
    assert captured.value.pending_digests == plan.plan.artifact_digests
    store.verify(artifact)


def test_cas_unlink_never_holds_a_sqlite_writer_transaction(
    registry: RunRegistry,
    store: ArtifactStore,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _register(registry, store, tmp_path, b"short-write-transaction")
    controller = _enabled(registry, store)
    plan = controller.plan(reason="short-write-transaction")
    assert plan is not None
    original_unlink = store.unlink_verified
    lock_probes = 0

    def assert_writer_is_available(
        artifact: PublishedArtifact,
        *,
        expected_generation: ArtifactGeneration | None = None,
    ) -> None:
        nonlocal lock_probes
        probe = open_database(registry.path, busy_timeout_ms=0)
        try:
            probe.execute("BEGIN IMMEDIATE")
            lock_probes += 1
            probe.execute("ROLLBACK")
        finally:
            probe.close()
        original_unlink(
            artifact,
            expected_generation=expected_generation,
        )

    monkeypatch.setattr(store, "unlink_verified", assert_writer_is_available)

    result = controller.execute(plan, confirmed_digest=plan.digest)

    assert result.newly_deleted_digests == plan.plan.artifact_digests
    assert lock_probes == 1


def test_unlink_then_failure_remains_explicitly_pending_and_fail_closed(
    registry: RunRegistry,
    store: ArtifactStore,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _register(registry, store, tmp_path, b"uncertain-crash-window")
    controller = _enabled(registry, store)
    plan = controller.plan(reason="uncertain-unlink-window")
    assert plan is not None
    original_unlink = store.unlink_verified

    def unlink_then_fail(
        artifact: PublishedArtifact,
        *,
        expected_generation: ArtifactGeneration | None = None,
    ) -> None:
        original_unlink(
            artifact,
            expected_generation=expected_generation,
        )
        raise ArtifactStoreError("injected post-unlink failure")

    monkeypatch.setattr(store, "unlink_verified", unlink_then_fail)
    with pytest.raises(RetentionPartialFailure) as captured:
        controller.execute(plan, confirmed_digest=plan.digest)
    assert captured.value.finalized_digests == ()
    assert captured.value.pending_digests == plan.plan.artifact_digests
    assert _tombstones(registry)[0]["deleted_at"] is None

    monkeypatch.setattr(store, "unlink_verified", original_unlink)
    with pytest.raises(RetentionPartialFailure) as retried:
        controller.execute(
            plan,
            confirmed_digest=plan.digest,
        )
    assert retried.value.pending_digests == plan.plan.artifact_digests


def test_pending_intent_prevents_a_later_evidence_link(
    registry: RunRegistry,
    store: ArtifactStore,
    tmp_path: Path,
) -> None:
    artifact = _register(registry, store, tmp_path, b"no-relink")
    controller = _enabled(registry, store)
    plan = controller.plan(reason="link-exclusion")
    assert plan is not None
    controller.stage(plan, confirmed_digest=plan.digest)

    _set_registry_time(registry, NOW)
    registry.submit(
        SubmissionRequest(kind="retention-test", payload={}),
        idempotency_key="retention-idempotency-no-relink",
    )
    _set_registry_time(registry, NOW + timedelta(seconds=1))
    with pytest.raises(RetentionPendingError, match="retention"):
        registry.claim(worker_id="worker-no-relink", lease_seconds=60)
    assert _count(registry, "sl_registry_run_artifacts") == 0
    store.verify(artifact)


def test_concurrent_stage_is_one_atomic_idempotent_cohort(
    registry: RunRegistry,
    store: ArtifactStore,
    tmp_path: Path,
) -> None:
    for payload in (b"concurrent-a", b"concurrent-b"):
        _register(registry, store, tmp_path, payload)
    controller = _enabled(registry, store)
    plan = controller.plan(reason="concurrent-stage")
    assert plan is not None
    barrier = Barrier(3)

    def stage() -> object:
        barrier.wait()
        return controller.stage(plan, confirmed_digest=plan.digest)

    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(stage) for _ in range(2)]
        barrier.wait()
        intents = [future.result(timeout=5) for future in futures]

    assert intents[0] == intents[1]
    assert len(_tombstones(registry)) == len(plan.candidates)
    assert all(row["deleted_at"] is None for row in _tombstones(registry))


def test_concurrent_execution_is_safe_for_the_same_confirmed_plan(
    registry: RunRegistry,
    store: ArtifactStore,
    tmp_path: Path,
) -> None:
    _register(registry, store, tmp_path, b"concurrent-execute")
    controller = _enabled(registry, store)
    plan = controller.plan(reason="concurrent-execute")
    assert plan is not None
    controller.stage(plan, confirmed_digest=plan.digest)
    barrier = Barrier(3)

    def execute() -> RetentionExecution | RetentionPartialFailure:
        barrier.wait()
        try:
            return controller.execute(plan, confirmed_digest=plan.digest)
        except RetentionPartialFailure as exc:
            return exc

    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(execute) for _ in range(2)]
        barrier.wait()
        results = [future.result(timeout=5) for future in futures]

    completed = [result for result in results if isinstance(result, RetentionExecution)]
    failed_closed = [result for result in results if isinstance(result, RetentionPartialFailure)]
    assert len(completed) >= 1
    assert sum(len(result.newly_deleted_digests) for result in completed) == 1
    assert all(result.plan_digest == plan.digest for result in failed_closed)
    assert all(result.pending_digests for result in failed_closed)
    assert all(row["deleted_at"] is not None for row in _tombstones(registry))

    replay = controller.execute(plan, confirmed_digest=plan.digest)
    assert replay.newly_deleted_digests == ()
    assert replay.previously_deleted_digests == plan.plan.artifact_digests


def test_plan_and_link_race_has_one_safe_winner(
    registry: RunRegistry,
    store: ArtifactStore,
    tmp_path: Path,
) -> None:
    artifact = _register(registry, store, tmp_path, b"race-link")
    controller = _enabled(registry, store)
    plan = controller.plan(reason="link-race")
    assert plan is not None
    _set_registry_time(registry, NOW)
    registry.submit(
        SubmissionRequest(kind="retention-test", payload={}),
        idempotency_key="retention-idempotency-race-link",
    )
    _set_registry_time(registry, NOW + timedelta(seconds=1))
    claimed = registry.claim(worker_id="worker-race-link", lease_seconds=60)
    assert claimed is not None
    _set_registry_time(registry, NOW + timedelta(seconds=3))
    barrier = Barrier(3)

    def stage() -> str:
        barrier.wait()
        try:
            controller.stage(plan, confirmed_digest=plan.digest)
            return "retention"
        except (BusyError, RetentionActiveWorkError, RetentionDriftError):
            return "link"

    def link() -> str:
        barrier.wait()
        try:
            registry.complete(
                claimed.lease,
                run=TerminalRunRequest(
                    run_id="run-race-link",
                    evidence_class=EvidenceClass.SIMULATED,
                    started_at=NOW + timedelta(seconds=2),
                ),
                artifact_links=(ArtifactLink("output", artifact.digest),),
            )
            return "link"
        except ConflictError:
            return "retention"

    with ThreadPoolExecutor(max_workers=2) as pool:
        stage_future = pool.submit(stage)
        link_future = pool.submit(link)
        barrier.wait()
        outcomes = (stage_future.result(timeout=5), link_future.result(timeout=5))

    assert outcomes[0] == outcomes[1]
    if outcomes[0] == "retention":
        assert len(_tombstones(registry)) == 1
        assert _count(registry, "sl_registry_run_artifacts") == 0
    else:
        assert _tombstones(registry) == []
        assert _count(registry, "sl_registry_run_artifacts") == 1
    store.verify(artifact)


def test_controller_requires_registry_and_store_contracts(
    registry: RunRegistry,
    store: ArtifactStore,
) -> None:
    with pytest.raises(ValidationError, match="registry"):
        RetentionController(object(), store)  # type: ignore[arg-type]
    with pytest.raises(ValidationError, match="artifact_store"):
        RetentionController(registry, object())  # type: ignore[arg-type]
    with pytest.raises(ValidationError, match="policy"):
        RetentionController(registry, store, policy=object())  # type: ignore[arg-type]
    altered_limits = replace(registry.limits, busy_timeout_ms=0)
    assert type(altered_limits) is RegistryLimits

    uninitialized = ArtifactStore(store.root.parent / "uninitialized-cas")
    uninitialized_registry = _TestRegistry(
        store.root.parent / "uninitialized.sqlite",
        artifact_verifier=uninitialized,
    )
    with pytest.raises(ValidationError, match="initialized"):
        RetentionController(uninitialized_registry, uninitialized)


def test_retention_rejects_a_registry_opened_with_the_wrong_hmac_authority(
    registry: RunRegistry,
    store: ArtifactStore,
    tmp_path: Path,
) -> None:
    _register(registry, store, tmp_path, b"authority-bound")
    wrong_authority = RunRegistry(
        registry.path,
        digest_secret=b"different-retention-authority!!!",
        clock=_MutableClock(NOW),
        artifact_verifier=store,
    )
    with pytest.raises(IntegrityError, match="authority"):
        _enabled(wrong_authority, store)


def test_controller_requires_exact_verifier_identity_and_durable_store_binding(
    registry: RunRegistry,
    store: ArtifactStore,
    tmp_path: Path,
) -> None:
    same_root = ArtifactStore(store.root, max_artifact_bytes=1_024)
    same_root.initialize()
    with pytest.raises(ValidationError, match="exact ArtifactStore"):
        RetentionController(registry, same_root)

    other_store = ArtifactStore(tmp_path / "other-cas", max_artifact_bytes=1_024)
    other_store.initialize()
    rebound_registry = _TestRegistry(
        registry.path,
        artifact_verifier=other_store,
    )
    with pytest.raises(ConflictError, match="another CAS"):
        RetentionController(rebound_registry, other_store)


def test_plan_hmac_rejects_cross_registry_replay_with_same_or_different_secret(
    registry: RunRegistry,
    store: ArtifactStore,
    tmp_path: Path,
) -> None:
    _register(registry, store, tmp_path, b"authority-scoped-plan")
    controller = _enabled(registry, store)
    plan = controller.plan(reason="authority-scoped-plan")
    assert plan is not None

    same_secret = RunRegistry(
        tmp_path / "same-secret.sqlite",
        digest_secret=SECRET,
        clock=_MutableClock(NOW),
        artifact_verifier=store,
    )
    same_secret.initialize()
    same_controller = RetentionController(same_secret, store, policy=controller.policy)
    different_secret = RunRegistry(
        tmp_path / "different-secret.sqlite",
        digest_secret=b"different-retention-secret-value!!",
        clock=_MutableClock(NOW),
        artifact_verifier=store,
    )
    different_secret.initialize()
    different_controller = RetentionController(
        different_secret,
        store,
        policy=controller.policy,
    )

    payload = plan.canonical_payload_bytes()
    assert same_secret.sign_retention_payload(payload, store_id=store.store_id) != plan.digest
    assert different_secret.sign_retention_payload(payload, store_id=store.store_id) != plan.digest
    for foreign_controller in (same_controller, different_controller):
        with pytest.raises(RetentionConfirmationError, match="registry identity"):
            foreign_controller.stage(plan, confirmed_digest=plan.digest)
    assert _tombstones(registry) == []


def test_restart_load_list_and_resume_recover_exact_signed_plan(
    registry: RunRegistry,
    store: ArtifactStore,
    tmp_path: Path,
) -> None:
    artifact = _register(registry, store, tmp_path, b"restart-recovery")
    controller = _enabled(registry, store)
    plan = controller.plan(reason="restart-recovery")
    assert plan is not None
    controller.stage(plan, confirmed_digest=plan.digest)

    restarted_store = ArtifactStore(store.root, max_artifact_bytes=1_024)
    restarted_store.initialize()
    restarted_registry = _TestRegistry(
        registry.path,
        artifact_verifier=restarted_store,
    )
    restarted_registry.initialize()
    restarted = _enabled(restarted_registry, restarted_store)

    recoveries = restarted.list_recovery_plans(limit=10)
    assert len(recoveries) == 1
    assert recoveries[0].plan_digest == plan.digest
    assert recoveries[0].pending_count == 1
    assert recoveries[0].finalized_count == 0
    assert restarted.load_plan(plan.digest) == plan

    result = restarted.resume(plan.digest, confirmed_digest=plan.digest)
    assert result.newly_deleted_digests == (artifact.digest,)
    assert restarted.list_recovery_plans(limit=10) == ()


def test_execute_rechecks_running_work_for_an_already_staged_plan(
    registry: RunRegistry,
    store: ArtifactStore,
    tmp_path: Path,
) -> None:
    artifact = _register(registry, store, tmp_path, b"staged-active-work")
    controller = _enabled(registry, store)
    plan = controller.plan(reason="staged-active-work")
    assert plan is not None
    controller.stage(plan, confirmed_digest=plan.digest)
    _set_registry_time(registry, NOW)
    queued = registry.submit(
        SubmissionRequest(kind="retention-test", payload={}),
        idempotency_key="retention-idempotency-staged-active",
    )
    active_at = NOW + timedelta(seconds=1)
    connection = open_database(registry.path, busy_timeout_ms=registry.limits.busy_timeout_ms)
    try:
        changed = connection.execute(
            """
            UPDATE sl_registry_jobs
            SET state = 'running', attempt_count = 1, updated_at = ?,
                lease_owner = 'adversarial-worker', lease_token_digest = ?,
                lease_expires_at = ?, heartbeat_at = ?
            WHERE job_id = ? AND state = 'queued'
            """,
            (
                active_at.isoformat(timespec="microseconds").replace("+00:00", "Z"),
                "a" * 64,
                (active_at + timedelta(minutes=1))
                .isoformat(timespec="microseconds")
                .replace("+00:00", "Z"),
                active_at.isoformat(timespec="microseconds").replace("+00:00", "Z"),
                queued.job_id,
            ),
        ).rowcount
        assert changed == 1
    finally:
        connection.close()
    _set_registry_time(registry, active_at)

    with pytest.raises(RetentionActiveWorkError):
        controller.execute(plan, confirmed_digest=plan.digest)
    store.verify(artifact)


@pytest.mark.parametrize("chronology", ["before-plan", "future"])
def test_recovery_rejects_forged_finalized_chronology(
    registry: RunRegistry,
    store: ArtifactStore,
    tmp_path: Path,
    chronology: str,
) -> None:
    _register(registry, store, tmp_path, chronology.encode("ascii"))
    controller = _enabled(registry, store)
    plan = controller.plan(reason=f"forged-{chronology}")
    assert plan is not None
    controller.stage(plan, confirmed_digest=plan.digest)
    forged = (
        plan.plan.planned_at - timedelta(seconds=1)
        if chronology == "before-plan"
        else NOW + timedelta(days=1)
    )
    connection = open_database(registry.path, busy_timeout_ms=registry.limits.busy_timeout_ms)
    try:
        if chronology == "before-plan":
            connection.execute("PRAGMA ignore_check_constraints = ON")
        connection.execute(
            """
            UPDATE sl_registry_retention_tombstones
            SET deleted_at = ?
            WHERE plan_digest = ?
            """,
            (
                forged.isoformat(timespec="microseconds").replace("+00:00", "Z"),
                plan.digest,
            ),
        )
    finally:
        connection.close()

    with pytest.raises(RetentionIntegrityError, match="chronology|tombstone"):
        controller.load_plan(plan.digest)


@pytest.mark.parametrize("replacement", ["bytes", "symlink", "fifo"])
def test_finalized_replay_rejects_restored_or_special_exact_key(
    registry: RunRegistry,
    store: ArtifactStore,
    tmp_path: Path,
    replacement: str,
) -> None:
    payload = f"finalized-{replacement}".encode("ascii")
    artifact = _register(registry, store, tmp_path, payload)
    controller = _enabled(registry, store)
    plan = controller.plan(reason=f"finalized-{replacement}")
    assert plan is not None
    controller.execute(plan, confirmed_digest=plan.digest)
    object_path = store.root.joinpath(*artifact.storage_key.split("/"))
    if replacement == "bytes":
        object_path.write_bytes(payload)
        object_path.chmod(0o400)
    elif replacement == "symlink":
        target = tmp_path / "symlink-target"
        target.write_bytes(payload)
        object_path.symlink_to(target)
    else:
        os.mkfifo(object_path, 0o400)

    with pytest.raises(RetentionIntegrityError, match="CAS key"):
        controller.execute(plan, confirmed_digest=plan.digest)


def test_generation_preflight_checks_entire_cohort_before_persisting_intent(
    registry: RunRegistry,
    store: ArtifactStore,
    tmp_path: Path,
) -> None:
    artifacts = (
        _register(registry, store, tmp_path, b"preflight-stable"),
        _register(registry, store, tmp_path, b"preflight-republished"),
    )
    controller = _enabled(registry, store)
    plan = controller.plan(reason="fresh-republication")
    assert plan is not None
    artifact = artifacts[-1]
    candidate = next(item for item in plan.candidates if item.digest == artifact.digest)
    source = tmp_path / f"source-{artifact.digest}.bin"
    store.unlink_verified(
        artifact,
        expected_generation=candidate.generation,
    )
    republished = store.publish(source)
    assert republished.digest == artifact.digest

    with pytest.raises(RetentionDriftError, match="generation"):
        controller.execute(plan, confirmed_digest=plan.digest)
    assert _count(registry, "sl_registry_retention_plans") == 0
    assert _tombstones(registry) == []
    for published in artifacts[:-1]:
        store.verify(published)
    store.verify(republished)

    _set_registry_time(registry, NOW)
    registry.submit(
        SubmissionRequest(kind="retention-test", payload={}),
        idempotency_key="retention-idempotency-preflight-drift",
    )
    _set_registry_time(registry, NOW + timedelta(seconds=1))
    assert registry.claim(worker_id="worker-preflight-drift", lease_seconds=60) is not None


def test_physical_generation_grace_excludes_planning_and_prevents_intent(
    registry: RunRegistry,
    store: ArtifactStore,
    tmp_path: Path,
) -> None:
    physical_now = datetime.now(UTC)
    artifact = _register(
        registry,
        store,
        tmp_path,
        b"physical-grace",
        created_at=physical_now - timedelta(days=30),
    )
    _set_registry_time(registry, physical_now)
    controller = RetentionController(
        registry,
        store,
        policy=RetentionPolicy(
            enabled=True,
            grace_period_seconds=24 * 60 * 60,
            max_candidates=100,
            max_total_bytes=1_024,
        ),
    )
    plan = controller.plan(reason="physical-grace")
    assert plan is None

    metadata = ArtifactMetadata(
        digest=artifact.digest,
        artifact_class=ArtifactClass.OUTPUT,
        byte_size=artifact.byte_size,
        media_type="application/octet-stream",
        storage_relpath=artifact.storage_key,
        created_at=physical_now - timedelta(days=30),
    )
    inventory = store.inspect(artifact)
    physically_young = DigestConfirmedRetentionPlan.create(
        plan=RetentionPlan((artifact.digest,), physical_now, "physical-grace-preflight"),
        candidates=(
            RetentionCandidate(
                metadata,
                inventory.last_changed_at,
                inventory.last_changed_ns,
                inventory.generation,
            ),
        ),
        eligible_before=physical_now - timedelta(days=1),
        policy=controller.policy,
        registry=registry,
        cas_store_id=store.store_id,
    )
    with pytest.raises(RetentionDriftError, match="physical grace"):
        controller.execute(physically_young, confirmed_digest=physically_young.digest)
    assert _count(registry, "sl_registry_retention_plans") == 0
    assert _tombstones(registry) == []
    store.verify(artifact)

    registry.submit(
        SubmissionRequest(kind="retention-test", payload={}),
        idempotency_key="retention-idempotency-physical-grace",
    )
    _set_registry_time(registry, physical_now + timedelta(seconds=1))
    assert registry.claim(worker_id="worker-physical-grace", lease_seconds=60) is not None


def test_oversized_corrupt_tombstone_cohort_is_bounded_and_rejected(
    registry: RunRegistry,
    store: ArtifactStore,
    tmp_path: Path,
) -> None:
    _register(registry, store, tmp_path, b"oversized-cohort")
    controller = _enabled(registry, store)
    plan = controller.plan(reason="oversized-cohort")
    assert plan is not None
    controller.stage(plan, confirmed_digest=plan.digest)
    extras: list[tuple[str, str, str, str]] = []
    index = 0
    while len(extras) < 1_000:
        digest = hashlib.sha256(f"extra-{index}".encode("ascii")).hexdigest()
        index += 1
        if digest not in plan.plan.artifact_digests:
            extras.append(
                (
                    digest,
                    plan.digest,
                    plan.plan.planned_at.isoformat(timespec="microseconds").replace("+00:00", "Z"),
                    plan.plan.reason,
                )
            )
    connection = open_database(registry.path, busy_timeout_ms=registry.limits.busy_timeout_ms)
    try:
        connection.execute("PRAGMA foreign_keys = OFF")
        connection.executemany(
            """
            INSERT INTO sl_registry_retention_tombstones(
                artifact_digest, plan_digest, planned_at, deleted_at, reason
            ) VALUES(?, ?, ?, NULL, ?)
            """,
            extras,
        )
    finally:
        connection.close()

    with pytest.raises(RetentionIntegrityError, match="row bound|exceeds"):
        controller.list_recovery_plans(limit=10)
    with pytest.raises(RetentionIntegrityError, match="cohort exceeds"):
        controller.stage(plan, confirmed_digest=plan.digest)
    with pytest.raises(RetentionIntegrityError, match="row bound|exceeds"):
        controller.load_plan(plan.digest)


def test_sqlite_failure_redacts_attacker_controlled_cause(
    registry: RunRegistry,
    store: ArtifactStore,
    tmp_path: Path,
) -> None:
    _register(registry, store, tmp_path, b"redacted-sqlite-cause")
    controller = _enabled(registry, store)
    plan = controller.plan(reason="redacted-sqlite-cause")
    assert plan is not None
    attacker_text = "private-path-and-credential-value"
    connection = open_database(registry.path, busy_timeout_ms=registry.limits.busy_timeout_ms)
    try:
        connection.execute(f"""
            CREATE TRIGGER test_retention_redacted_failure
            BEFORE INSERT ON sl_registry_retention_tombstones
            BEGIN
                SELECT RAISE(ABORT, '{attacker_text}');
            END;
            """)
    finally:
        connection.close()

    with pytest.raises(RetentionIntegrityError) as captured:
        controller.stage(plan, confirmed_digest=plan.digest)
    assert captured.value.__cause__ is None
    assert attacker_text not in str(captured.value)


@pytest.mark.parametrize(
    "corruption",
    [
        "invalid-json",
        "wrong-shape",
        "noncanonical-json",
        "policy-shape",
        "schema-version",
        "empty-candidates",
        "candidate-shape",
        "payload-hash",
        "column-time",
        "signature",
    ],
)
def test_persisted_plan_envelope_decoder_fails_closed_on_corruption(
    registry: RunRegistry,
    store: ArtifactStore,
    tmp_path: Path,
    corruption: str,
) -> None:
    _register(registry, store, tmp_path, corruption.encode("ascii"))
    controller = _enabled(registry, store)
    plan = controller.plan(reason=f"envelope-{corruption}")
    assert plan is not None
    original = plan.canonical_payload_bytes().decode("utf-8")
    payload = original
    if corruption == "invalid-json":
        payload = "{x"
    elif corruption == "wrong-shape":
        payload = '{"unexpected":true}'
    elif corruption == "noncanonical-json":
        payload = original.replace('{"candidates"', '{ "candidates"', 1)
    elif corruption in {
        "policy-shape",
        "schema-version",
        "empty-candidates",
        "candidate-shape",
    }:
        parsed = json.loads(original)
        if corruption == "policy-shape":
            parsed["policy"].pop("max_candidates")
        elif corruption == "schema-version":
            parsed["schema_version"] = 2
        elif corruption == "empty-candidates":
            parsed["candidates"] = []
        else:
            parsed["candidates"][0].pop("media_type")
        payload = canonical_json(parsed)
    plan_digest = hashlib.sha256(f"plan-{corruption}".encode("ascii")).hexdigest()
    payload_digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()
    if corruption == "payload-hash":
        payload_digest = "0" * 64
    stored_planned_at = plan.plan.planned_at
    if corruption == "column-time":
        stored_planned_at += timedelta(seconds=1)
    connection = open_database(registry.path, busy_timeout_ms=registry.limits.busy_timeout_ms)
    try:
        connection.execute(
            """
            INSERT INTO sl_registry_retention_plans(
                plan_digest, payload_digest, payload_json, registry_id,
                cas_store_id, planned_at, schema_version
            ) VALUES(?, ?, ?, ?, ?, ?, 1)
            """,
            (
                plan_digest,
                payload_digest,
                payload,
                plan.registry_id,
                plan.cas_store_id,
                stored_planned_at.isoformat(timespec="microseconds").replace("+00:00", "Z"),
            ),
        )
    finally:
        connection.close()

    with pytest.raises(IntegrityError):
        controller.load_plan(plan_digest)


def test_persisted_plan_envelope_rejects_lone_surrogate_as_integrity_failure(
    registry: RunRegistry,
    store: ArtifactStore,
    tmp_path: Path,
) -> None:
    """A hostile SQLite text value cannot leak a raw UTF-8 codec failure."""

    _register(registry, store, tmp_path, b"envelope-lone-surrogate")
    controller = _enabled(registry, store)
    plan = controller.plan(reason="envelope-lone-surrogate")
    assert plan is not None
    stored_row: Any = {
        "plan_digest": plan.digest,
        "payload_digest": plan.payload_digest,
        "payload_json": "\ud800",
        "registry_id": plan.registry_id,
        "cas_store_id": plan.cas_store_id,
        "planned_at": plan.plan.planned_at.isoformat(timespec="microseconds").replace(
            "+00:00", "Z"
        ),
        "schema_version": plan.schema_version,
    }

    with pytest.raises(RetentionIntegrityError, match="valid bounded UTF-8") as captured:
        controller._plan_from_envelope_row(
            stored_row,
            budget=retention_module._OperationBudget.start(),
        )

    assert captured.value.__cause__ is None


def test_operation_deadlines_and_sqlite_interrupts_fail_closed_without_raw_causes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    backwards = retention_module._OperationBudget(deadline=10.0, last_observed=5.0)
    monkeypatch.setattr(retention_module.time, "monotonic", lambda: 4.0)
    assert backwards.expired() == 1
    with pytest.raises(RetentionIntegrityError, match="moved backwards"):
        backwards.checkpoint()

    expired = retention_module._OperationBudget(deadline=5.0, last_observed=4.0)
    monkeypatch.setattr(retention_module.time, "monotonic", lambda: 5.0)
    with pytest.raises(RetentionIntegrityError, match="deadline"):
        expired.checkpoint()

    sqlite_failure = sqlite3.OperationalError("attacker-controlled sqlite text")
    sqlite_failure.sqlite_errorcode = sqlite3.SQLITE_INTERRUPT
    with pytest.raises(RetentionIntegrityError, match="deadline") as captured:
        retention_module._raise_sqlite(sqlite_failure, operation="bounded test")
    assert captured.value.__cause__ is None
    assert "attacker-controlled" not in str(captured.value)


def test_stage_and_load_reject_incomplete_or_missing_durable_envelopes(
    registry: RunRegistry,
    store: ArtifactStore,
    tmp_path: Path,
) -> None:
    _register(registry, store, tmp_path, b"envelope-without-cohort")
    controller = _enabled(registry, store)
    plan = controller.plan(reason="envelope-without-cohort")
    assert plan is not None
    connection = open_database(registry.path, busy_timeout_ms=registry.limits.busy_timeout_ms)
    try:
        _insert_plan_envelope(connection, plan)
    finally:
        connection.close()

    with pytest.raises(RetentionDriftError, match="no candidate cohort"):
        controller.stage(plan, confirmed_digest=plan.digest)
    missing = hashlib.sha256(b"missing-plan").hexdigest()
    with pytest.raises(RetentionDriftError, match="missing"):
        controller.load_plan(missing)


def test_confirmation_rechecks_policy_verifier_and_busy_authority(
    registry: RunRegistry,
    store: ArtifactStore,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _register(registry, store, tmp_path, b"confirmation-rechecks")
    controller = _enabled(registry, store)
    plan = controller.plan(reason="confirmation-rechecks")
    assert plan is not None

    expanded_policy = RetentionPolicy(
        enabled=True,
        grace_period_seconds=controller.policy.grace_period_seconds,
        max_candidates=controller.policy.max_candidates + 1,
        max_total_bytes=controller.policy.max_total_bytes + 1,
    )
    expanded = RetentionController(registry, store, policy=expanded_policy)
    with pytest.raises(RetentionConfirmationError, match="retention policy"):
        expanded.stage(plan, confirmed_digest=plan.digest)

    monkeypatch.setattr(registry, "_artifact_verifier", object())
    with pytest.raises(RetentionConfirmationError, match="verifier"):
        controller.stage(plan, confirmed_digest=plan.digest)
    monkeypatch.setattr(registry, "_artifact_verifier", store)

    def busy_verification(
        _: bytes,
        *,
        store_id: str,
        signature: str,
    ) -> None:
        del store_id, signature
        raise BusyError("bounded injected contention")

    monkeypatch.setattr(registry, "verify_retention_payload_signature", busy_verification)
    with pytest.raises(BusyError):
        controller.stage(plan, confirmed_digest=plan.digest)


def test_plan_reports_missing_exact_cas_bytes_as_path_free_integrity_failure(
    registry: RunRegistry,
    store: ArtifactStore,
    tmp_path: Path,
) -> None:
    artifact = _register(registry, store, tmp_path, b"missing-during-plan")
    store.unlink_verified(artifact)

    with pytest.raises(RetentionIntegrityError, match="descriptor-safe") as captured:
        _enabled(registry, store).plan(reason="missing-during-plan")
    assert captured.value.__cause__ is None
    assert str(store.root) not in str(captured.value)


def test_unlink_clock_regression_leaves_ambiguous_intent_pending(
    registry: RunRegistry,
    store: ArtifactStore,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _register(registry, store, tmp_path, b"unlink-clock-regression")
    controller = _enabled(registry, store)
    plan = controller.plan(reason="unlink-clock-regression")
    assert plan is not None
    original_unlink = store.unlink_verified

    def unlink_then_regress(
        artifact: PublishedArtifact,
        *,
        expected_generation: ArtifactGeneration | None = None,
    ) -> None:
        original_unlink(
            artifact,
            expected_generation=expected_generation,
        )
        _set_registry_time(registry, plan.plan.planned_at - timedelta(microseconds=1))

    monkeypatch.setattr(store, "unlink_verified", unlink_then_regress)
    with pytest.raises(RetentionPartialFailure) as captured:
        controller.execute(plan, confirmed_digest=plan.digest)
    assert captured.value.pending_digests == plan.plan.artifact_digests
    assert _tombstones(registry)[0]["deleted_at"] is None


def test_result_contracts_reject_ambiguous_or_unbounded_state() -> None:
    digest = "a" * 64
    with pytest.raises(ValidationError, match="ArtifactMetadata"):
        RetentionCandidate(  # type: ignore[arg-type]
            object(),
            NOW,
            0,
            ArtifactGeneration("a" * 64),
        )
    with pytest.raises(ValidationError, match="cohort"):
        retention_module.RetentionRecovery(digest, NOW, "invalid-cohort", 1_000, 1)
    with pytest.raises(ValidationError, match="disjoint"):
        RetentionIntent(digest, (digest,), (digest,))
    with pytest.raises(ValidationError, match="disjoint"):
        RetentionExecution(digest, (digest,), (digest,), 1, NOW)
    with pytest.raises(ValidationError, match="disjoint"):
        RetentionPartialFailure(
            plan_digest=digest,
            finalized_digests=(digest,),
            pending_digests=(digest,),
            cause_code="artifact_integrity",
        )
    with pytest.raises(ValidationError, match="cause_code"):
        RetentionPartialFailure(
            plan_digest=digest,
            finalized_digests=(),
            pending_digests=(digest,),
            cause_code="not-ascii-£",
        )
    with pytest.raises(ValidationError, match="cause_code"):
        RetentionPartialFailure(
            plan_digest=digest,
            finalized_digests=(),
            pending_digests=(digest,),
            cause_code="../private/path",
        )
    with pytest.raises(ValidationError, match="possibly empty tuple"):
        RetentionIntent(digest, [], ())  # type: ignore[arg-type]
    with pytest.raises(ValidationError, match="may not exceed"):
        RetentionIntent(digest, (digest,) * 1_001, ())
    with pytest.raises(ValidationError, match="unique sorted"):
        RetentionIntent(digest, (digest, digest), ())


def test_stage_respects_bounded_database_lock_wait(tmp_path: Path) -> None:
    store = ArtifactStore(tmp_path / "cas", max_artifact_bytes=1_024)
    store.initialize()
    registry = _TestRegistry(
        tmp_path / "busy.sqlite",
        artifact_verifier=store,
        limits=RegistryLimits(busy_timeout_ms=0),
    )
    registry.initialize()
    _register(registry, store, tmp_path, b"busy-retention")
    controller = _enabled(registry, store)
    plan = controller.plan(reason="busy-bound")
    assert plan is not None
    blocker = open_database(registry.path, busy_timeout_ms=0)
    blocker.execute("BEGIN IMMEDIATE")
    try:
        with pytest.raises(BusyError):
            controller.stage(plan, confirmed_digest=plan.digest)
    finally:
        blocker.execute("ROLLBACK")
        blocker.close()


def _persist_readiness_envelope(
    registry: RunRegistry,
    plan: DigestConfirmedRetentionPlan,
    *,
    payload_json: str,
    plan_digest: str,
    include_tombstones: bool,
) -> None:
    """Persist a controlled hostile envelope while retaining valid SQLite relationships."""

    payload = json.loads(payload_json)
    if type(payload) is not dict or type(payload.get("candidates")) is not list:
        raise AssertionError("readiness fixture requires a candidate-bearing envelope")
    candidate_digests = tuple(
        candidate["digest"] for candidate in payload["candidates"] if type(candidate) is dict
    )
    planned_at = plan.plan.planned_at.isoformat(timespec="microseconds").replace("+00:00", "Z")
    connection = open_database(
        registry.path,
        busy_timeout_ms=registry.limits.busy_timeout_ms,
    )
    try:
        connection.execute(
            """
            INSERT INTO sl_registry_retention_plans(
                plan_digest, payload_digest, payload_json, registry_id,
                cas_store_id, planned_at, schema_version
            ) VALUES(?, ?, ?, ?, ?, ?, ?)
            """,
            (
                plan_digest,
                hashlib.sha256(payload_json.encode("utf-8")).hexdigest(),
                payload_json,
                plan.registry_id,
                plan.cas_store_id,
                planned_at,
                plan.schema_version,
            ),
        )
        if include_tombstones:
            connection.executemany(
                """
                INSERT INTO sl_registry_retention_tombstones(
                    artifact_digest, plan_digest, planned_at, deleted_at, reason
                ) VALUES(?, ?, ?, NULL, ?)
                """,
                (
                    (digest, plan_digest, planned_at, plan.plan.reason)
                    for digest in candidate_digests
                ),
            )
    finally:
        connection.close()


def test_readiness_rejects_forged_retention_digest_and_missing_authority(
    registry: RunRegistry,
    store: ArtifactStore,
    tmp_path: Path,
) -> None:
    _register(registry, store, tmp_path, b"readiness-forged-digest")
    controller = _enabled(registry, store)
    plan = controller.plan(reason="readiness-forged-digest")
    assert plan is not None
    forged_digest = hashlib.sha256(b"not-an-authority-mac").hexdigest()
    assert forged_digest != plan.digest
    _persist_readiness_envelope(
        registry,
        plan,
        payload_json=plan.canonical_payload_bytes().decode("utf-8"),
        plan_digest=forged_digest,
        include_tombstones=True,
    )

    authenticated = registry.probe_readiness()
    assert not authenticated.ready
    assert authenticated.reason == "integrity_error"
    unauthenticated = probe_database(
        registry.path,
        busy_timeout_ms=registry.limits.busy_timeout_ms,
        verification_timeout_ms=registry.limits.verification_timeout_ms,
        key_verifier=registry._key_verifier,
    )
    assert not unauthenticated.ready
    assert unauthenticated.reason == "integrity_error"


@pytest.mark.parametrize(
    "corruption",
    [
        "extra-field",
        "registry-binding",
        "cas-binding",
        "planned-at-binding",
        "schema-version",
        "artifact-metadata",
        "disabled-policy",
        "policy-count-bound",
        "policy-byte-bound",
        "grace-cutoff",
        "young-generation",
        "generation-token",
        "generation-nanoseconds",
    ],
)
def test_readiness_rejects_cross_bound_retention_payloads(
    registry: RunRegistry,
    store: ArtifactStore,
    tmp_path: Path,
    corruption: str,
) -> None:
    _register(registry, store, tmp_path, f"readiness-{corruption}".encode("ascii"))
    if corruption == "policy-count-bound":
        _register(registry, store, tmp_path, b"readiness-policy-count-bound-extra")
    controller = _enabled(registry, store)
    plan = controller.plan(reason=f"readiness-{corruption}")
    assert plan is not None
    payload = json.loads(plan.canonical_payload_bytes())
    if corruption == "extra-field":
        payload["unexpected"] = True
    elif corruption == "registry-binding":
        payload["registry_id"] = "0" * 32
    elif corruption == "cas-binding":
        payload["cas_store_id"] = "0" * 64
    elif corruption == "planned-at-binding":
        payload["planned_at"] = (
            (plan.plan.planned_at + timedelta(seconds=1))
            .isoformat(timespec="microseconds")
            .replace("+00:00", "Z")
        )
        payload["eligible_before"] = (
            (plan.eligible_before + timedelta(seconds=1))
            .isoformat(timespec="microseconds")
            .replace("+00:00", "Z")
        )
    elif corruption == "schema-version":
        payload["schema_version"] = 2
    elif corruption == "artifact-metadata":
        payload["candidates"][0]["media_type"] = "application/json"
    elif corruption == "disabled-policy":
        payload["policy"]["enabled"] = False
    elif corruption == "policy-count-bound":
        payload["policy"]["max_candidates"] = 1
    elif corruption == "policy-byte-bound":
        payload["policy"]["max_total_bytes"] = 1
    elif corruption == "grace-cutoff":
        payload["eligible_before"] = (
            (plan.eligible_before + timedelta(microseconds=1))
            .isoformat(timespec="microseconds")
            .replace("+00:00", "Z")
        )
    elif corruption == "young-generation":
        payload["candidates"][0]["last_changed_at"] = payload["planned_at"]
    elif corruption == "generation-token":
        payload["candidates"][0]["generation_token"] = "invalid"
    elif corruption == "generation-nanoseconds":
        payload["candidates"][0]["last_changed_ns"] = 1.5
    else:  # pragma: no cover - exhaustive parameter guard
        raise AssertionError("unknown corruption fixture")
    payload_json = canonical_json(payload)
    authenticated_digest = registry.sign_retention_payload(
        payload_json.encode("utf-8"),
        store_id=plan.cas_store_id,
    )
    _persist_readiness_envelope(
        registry,
        plan,
        payload_json=payload_json,
        plan_digest=authenticated_digest,
        include_tombstones=True,
    )

    readiness = registry.probe_readiness()
    assert not readiness.ready
    assert readiness.reason == "integrity_error"


def test_readiness_rejects_orphan_retention_plan_without_tombstones(
    registry: RunRegistry,
    store: ArtifactStore,
    tmp_path: Path,
) -> None:
    _register(registry, store, tmp_path, b"readiness-orphan-plan")
    controller = _enabled(registry, store)
    plan = controller.plan(reason="readiness-orphan-plan")
    assert plan is not None
    _persist_readiness_envelope(
        registry,
        plan,
        payload_json=plan.canonical_payload_bytes().decode("utf-8"),
        plan_digest=plan.digest,
        include_tombstones=False,
    )

    readiness = registry.probe_readiness()

    assert not readiness.ready
    assert readiness.reason == "integrity_error"


def test_readiness_accepts_valid_staged_retention_plan(
    registry: RunRegistry,
    store: ArtifactStore,
    tmp_path: Path,
) -> None:
    _register(registry, store, tmp_path, b"readiness-staged-plan")
    controller = _enabled(registry, store)
    plan = controller.plan(reason="readiness-staged-plan")
    assert plan is not None

    controller.stage(plan, confirmed_digest=plan.digest)

    assert registry.probe_readiness().ready


def test_readiness_accepts_mixed_resumable_retention_plan(
    registry: RunRegistry,
    store: ArtifactStore,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    for payload in (b"readiness-resume-a", b"readiness-resume-b", b"readiness-resume-c"):
        _register(registry, store, tmp_path, payload)
    controller = _enabled(registry, store)
    plan = controller.plan(reason="readiness-resumable-plan")
    assert plan is not None
    original_unlink = store.unlink_verified
    calls = 0

    def fail_second(
        artifact: PublishedArtifact,
        *,
        expected_generation: ArtifactGeneration | None = None,
    ) -> None:
        nonlocal calls
        calls += 1
        if calls == 2:
            raise ArtifactStoreError("bounded injected readiness failure")
        original_unlink(
            artifact,
            expected_generation=expected_generation,
        )

    monkeypatch.setattr(store, "unlink_verified", fail_second)
    with pytest.raises(RetentionPartialFailure):
        controller.execute(plan, confirmed_digest=plan.digest)

    rows = _tombstones(registry)
    assert any(row["deleted_at"] is None for row in rows)
    assert any(row["deleted_at"] is not None for row in rows)
    assert registry.probe_readiness().ready


@pytest.mark.parametrize(
    "payload_json",
    [
        "[" * 2_000 + "0" + "]" * 2_000,
        '{"duplicate":1,"duplicate":1}',
        '{"non_finite":NaN}',
    ],
)
def test_readiness_maps_hostile_retention_json_to_fail_closed_verdict(
    registry: RunRegistry,
    store: ArtifactStore,
    payload_json: str,
) -> None:
    registry_id = registry.registry_id
    plan_digest = registry._retention_plan_digest(
        payload_json.encode("utf-8"),
        registry_id,
        store.store_id,
    )
    connection = open_database(
        registry.path,
        busy_timeout_ms=registry.limits.busy_timeout_ms,
    )
    try:
        connection.execute(
            """
            INSERT INTO sl_registry_retention_plans(
                plan_digest, payload_digest, payload_json, registry_id,
                cas_store_id, planned_at, schema_version
            ) VALUES(?, ?, ?, ?, ?, ?, 1)
            """,
            (
                plan_digest,
                hashlib.sha256(payload_json.encode("utf-8")).hexdigest(),
                payload_json,
                registry_id,
                store.store_id,
                NOW.isoformat(timespec="microseconds").replace("+00:00", "Z"),
            ),
        )
    finally:
        connection.close()

    readiness = registry.probe_readiness()

    assert not readiness.ready
    assert readiness.reason == "integrity_error"


def test_readiness_rejects_noncanonical_indexed_and_tombstone_timestamps(
    registry: RunRegistry,
    store: ArtifactStore,
    tmp_path: Path,
) -> None:
    _register(registry, store, tmp_path, b"readiness-invalid-time")
    controller = _enabled(registry, store)
    plan = controller.plan(reason="readiness-invalid-time")
    assert plan is not None
    _persist_readiness_envelope(
        registry,
        plan,
        payload_json=plan.canonical_payload_bytes().decode("utf-8"),
        plan_digest=plan.digest,
        include_tombstones=True,
    )
    invalid_time = "x" * 26 + "Z"
    connection = open_database(
        registry.path,
        busy_timeout_ms=registry.limits.busy_timeout_ms,
    )
    try:
        connection.execute(
            """
            UPDATE sl_registry_retention_tombstones
            SET deleted_at = ?
            WHERE plan_digest = ?
            """,
            (invalid_time, plan.digest),
        )
    finally:
        connection.close()

    readiness = registry.probe_readiness()

    assert not readiness.ready
    assert readiness.reason == "integrity_error"


def test_readiness_rejects_real_artifact_size_in_retention_cohort(
    registry: RunRegistry,
    store: ArtifactStore,
) -> None:
    digest = hashlib.sha256(b"readiness-real-size").hexdigest()
    storage_key = f"objects/{digest[:2]}/{digest[2:4]}/{digest}"
    created_at = OLD.isoformat(timespec="microseconds").replace("+00:00", "Z")
    planned_at = NOW.isoformat(timespec="microseconds").replace("+00:00", "Z")
    eligible_before = (
        (NOW - timedelta(days=1)).isoformat(timespec="microseconds").replace("+00:00", "Z")
    )
    payload_json = canonical_json(
        {
            "candidates": [
                {
                    "artifact_class": "output",
                    "byte_size": 1,
                    "created_at": created_at,
                    "digest": digest,
                    "generation_token": hashlib.sha256(b"readiness-real-generation").hexdigest(),
                    "last_changed_at": created_at,
                    "last_changed_ns": retention_module._datetime_to_nanoseconds(OLD),
                    "media_type": "application/octet-stream",
                    "pinned": False,
                    "storage_relpath": storage_key,
                }
            ],
            "cas_store_id": store.store_id,
            "eligible_before": eligible_before,
            "planned_at": planned_at,
            "policy": {
                "enabled": True,
                "grace_period_seconds": 86_400,
                "max_candidates": 1,
                "max_total_bytes": 1_024,
            },
            "reason": "readiness-real-size",
            "registry_id": registry.registry_id,
            "schema_version": 1,
        }
    )
    plan_digest = registry.sign_retention_payload(
        payload_json.encode("utf-8"),
        store_id=store.store_id,
    )
    connection = open_database(
        registry.path,
        busy_timeout_ms=registry.limits.busy_timeout_ms,
    )
    try:
        connection.execute(
            """
            INSERT INTO sl_registry_artifacts(
                digest, artifact_class, byte_size, media_type, storage_relpath,
                created_at, pinned
            ) VALUES(?, 'output', 1.5, 'application/octet-stream', ?, ?, 0)
            """,
            (digest, storage_key, created_at),
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
                hashlib.sha256(payload_json.encode("utf-8")).hexdigest(),
                payload_json,
                registry.registry_id,
                store.store_id,
                planned_at,
            ),
        )
        connection.execute(
            """
            INSERT INTO sl_registry_retention_tombstones(
                artifact_digest, plan_digest, planned_at, deleted_at, reason
            ) VALUES(?, ?, ?, NULL, 'readiness-real-size')
            """,
            (digest, plan_digest, planned_at),
        )
    finally:
        connection.close()

    readiness = registry.probe_readiness()

    assert not readiness.ready
    assert readiness.reason == "integrity_error"


@pytest.mark.parametrize(
    "payload_json",
    [
        '{"duplicate":1,"duplicate":1}',
        "[" * 32 + "0" + "]" * 32,
        '"\\ud800"',
    ],
)
def test_retention_decoder_rejects_duplicate_deep_and_surrogate_json(
    payload_json: str,
) -> None:
    with pytest.raises(RetentionIntegrityError, match="JSON"):
        retention_module.decode_retention_plan_payload(
            payload_json,
            checkpoint=lambda: None,
        )


def test_retention_decoder_accepts_its_declared_maximum_candidate_bound() -> None:
    planned_at = retention_module._db_time(NOW)
    created_at = retention_module._db_time(OLD)
    last_changed_ns = retention_module._datetime_to_nanoseconds(OLD)
    candidates = []
    for index in range(1_000):
        digest = f"{index:064x}"
        candidates.append(
            {
                "artifact_class": "output",
                "byte_size": 1,
                "created_at": created_at,
                "digest": digest,
                "generation_token": hashlib.sha256(f"generation:{index}".encode()).hexdigest(),
                "last_changed_at": created_at,
                "last_changed_ns": last_changed_ns,
                "media_type": "application/octet-stream",
                "pinned": False,
                "storage_relpath": f"objects/{digest[:2]}/{digest[2:4]}/{digest}",
            }
        )
    payload_json = canonical_json(
        {
            "candidates": candidates,
            "cas_store_id": "a" * 64,
            "eligible_before": retention_module._db_time(NOW - timedelta(days=1)),
            "planned_at": planned_at,
            "policy": {
                "enabled": True,
                "grace_period_seconds": 86_400,
                "max_candidates": 1_000,
                "max_total_bytes": 1_000,
            },
            "reason": "maximum-candidate-bound",
            "registry_id": "b" * 32,
            "schema_version": 1,
        }
    )

    decoded = retention_module.decode_retention_plan_payload(
        payload_json,
        checkpoint=lambda: None,
    )

    assert len(decoded.candidates) == 1_000
    assert sum(candidate.metadata.byte_size for candidate in decoded.candidates) == 1_000
