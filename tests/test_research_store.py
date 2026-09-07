"""Atomic publication, corruption, idempotency, retention and restart invariants."""

from __future__ import annotations

import sqlite3
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

from signal_foundry.boundary import FoundryError
from signal_foundry.contracts import JobState, ResearchRequest
from signal_foundry.store import Store
from tests.research_helpers import evidence


@pytest.fixture
def store(tmp_path: Path):
    value = Store(tmp_path / "state")
    try:
        yield value
    finally:
        value.close()


def test_atomic_publish_and_audit(store: Store) -> None:
    request = ResearchRequest()
    job, created = store.submit(request, "a" * 16)
    assert created and job.state == JobState.QUEUED
    assert store.request(job.job_id) == request
    assert store.submit(request, "a" * 16) == (job, False)
    with pytest.raises(FoundryError, match="evidence_unavailable"):
        store.evidence(job.job_id)
    with pytest.raises(FoundryError, match="publication_conflict"):
        store.publish(job.job_id, evidence())
    store.transition(job.job_id, JobState.RUNNING)
    result = store.publish(job.job_id, evidence())
    assert result.evidence_hash == evidence().digest()
    assert store.evidence(job.job_id) == evidence()
    assert [event.state for event in store.audit(job.job_id).events] == [
        JobState.QUEUED,
        JobState.RUNNING,
        JobState.SUCCEEDED,
    ]
    assert store.list().jobs == (result,)
    with pytest.raises(FoundryError, match="terminal_job"):
        store.transition(job.job_id, JobState.CANCELLED)
    assert store.transition(job.job_id, JobState.SUCCEEDED) == result


def test_concurrent_idempotency_and_conflict(store: Store) -> None:
    with ThreadPoolExecutor(max_workers=8) as pool:
        results = list(
            pool.map(lambda _: store.submit(ResearchRequest(), "a" * 16), range(16))
        )
    assert sum(created for _, created in results) == 1
    assert len({job.job_id for job, _ in results}) == 1
    with pytest.raises(FoundryError, match="idempotency_conflict"):
        store.submit(ResearchRequest(seed=1), "a" * 16)
    with pytest.raises(FoundryError, match="idempotency_key"):
        store.submit(ResearchRequest(), "../bad")


def test_queue_and_retention_are_bounded(store: Store) -> None:
    jobs = [store.submit(ResearchRequest(), f"key_{i:016d}")[0] for i in range(8)]
    with pytest.raises(FoundryError, match="queue_capacity"):
        store.submit(ResearchRequest(), "b" * 16)
    for job in jobs:
        store.transition(job.job_id, JobState.CANCELLED, "cancelled")
    for index in range(8, 64):
        job, _ = store.submit(ResearchRequest(), f"key_{index:016d}")
        store.transition(job.job_id, JobState.FAILED, "test_failure")
    with pytest.raises(FoundryError, match="queue_capacity"):
        store.submit(ResearchRequest(), "b" * 16)
    assert len(store.list().jobs) == 64


def test_restart_marks_interrupted_and_excludes_second_owner(tmp_path: Path) -> None:
    path = tmp_path / "state"
    first = Store(path)
    job, _ = first.submit(ResearchRequest(), "a" * 16)
    first.transition(job.job_id, JobState.RUNNING)
    with pytest.raises(FoundryError, match="store_unavailable"):
        Store(path)
    first.close()
    restored = Store(path)
    try:
        assert restored.get(job.job_id).state == JobState.FAILED
        assert restored.get(job.job_id).error_code == "interrupted"
        assert len(restored.audit(job.job_id).events) == 3
    finally:
        restored.close()


def test_failure_rolls_back_artifact_and_audit(
    store: Store, monkeypatch: pytest.MonkeyPatch
) -> None:
    job, _ = store.submit(ResearchRequest(), "a" * 16)
    store.transition(job.job_id, JobState.RUNNING)

    def fail(*args: object) -> None:
        raise sqlite3.OperationalError("private SQL must not escape")

    monkeypatch.setattr(store, "_audit", fail)
    with pytest.raises(FoundryError, match="store_io") as error:
        store.publish(job.job_id, evidence())
    assert isinstance(error.value.__cause__, sqlite3.OperationalError)
    assert store.get(job.job_id).state == JobState.RUNNING
    assert store.get(job.job_id).evidence_hash is None
    assert len(store.audit(job.job_id).events) == 2


@pytest.mark.parametrize(
    "column,payload,code",
    [
        ("request_json", b"bad", "corrupt_store"),
        ("request_json", ResearchRequest(seed=1).canonical(), "corrupt_store"),
        ("evidence", b"bad", "corrupt_evidence"),
    ],
)
def test_corrupt_payload_is_not_evidence(
    store: Store, column: str, payload: bytes, code: str
) -> None:
    job, _ = store.submit(ResearchRequest(), "a" * 16)
    store.transition(job.job_id, JobState.RUNNING)
    store.publish(job.job_id, evidence())
    # Fixed test-only SQL identifiers; public store operations bind all values.
    with store.connection:
        store.connection.execute(
            f"UPDATE jobs SET {column}=? WHERE job_id=?", (payload, job.job_id)
        )
    with pytest.raises(FoundryError, match=code):
        (store.request if column == "request_json" else store.evidence)(job.job_id)


def test_invalid_transitions_and_paths(store: Store) -> None:
    for identity in ("../secret", "0" * 64):
        with pytest.raises(FoundryError, match="job_not_found"):
            store.get(identity)
    job, _ = store.submit(ResearchRequest(), "a" * 16)
    with pytest.raises(FoundryError, match="invalid_transition"):
        store.transition(job.job_id, JobState.SUCCEEDED)
    store.transition(job.job_id, JobState.RUNNING)
    with pytest.raises(FoundryError, match="invalid_transition"):
        store.transition(job.job_id, JobState.RUNNING)
    with pytest.raises(FoundryError, match="publication_conflict"):
        store.publish(job.job_id, evidence(ResearchRequest(seed=1)))
    store.close()
    with pytest.raises(FoundryError, match="store_closed"):
        store.get(job.job_id)
