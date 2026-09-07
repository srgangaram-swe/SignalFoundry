"""Deterministically ordered cancellation/admission/publication race tests."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

from signal_foundry.boundary import FoundryError
from signal_foundry.contracts import JobState, ResearchRequest
from signal_foundry.manager import Manager
from signal_foundry.store import Store
from tests.research_helpers import FakeRunner


@pytest.fixture
def manager(tmp_path: Path):
    runner = FakeRunner()
    value = Manager(Store(tmp_path / "state"), runner)
    try:
        yield value, runner
    finally:
        runner.release.set()
        value.close()


def test_success_and_retry_does_not_execute_twice(manager) -> None:
    owner, runner = manager
    with ThreadPoolExecutor(max_workers=8) as pool:
        jobs = list(
            pool.map(lambda _: owner.submit(ResearchRequest(), "a" * 16), range(16))
        )
    assert len({job.job_id for job in jobs}) == 1
    result = owner.wait(jobs[0].job_id, 5)
    assert result.state == JobState.SUCCEEDED
    assert runner.calls == 1
    assert owner.submit(ResearchRequest(), "a" * 16) == result
    assert owner.cancel(result.job_id) == result
    comparison = owner.compare(result.job_id, result.job_id)
    assert comparison.compatible


def test_cancel_running_and_queued_discards_late_success(manager) -> None:
    owner, runner = manager
    runner.release.clear()
    first = owner.submit(ResearchRequest(), "a" * 16)
    assert runner.started.wait(2)
    second = owner.submit(ResearchRequest(seed=2), "b" * 16)
    assert owner.cancel(second.job_id).state == JobState.CANCELLED
    assert owner.cancel(first.job_id).state == JobState.CANCELLED
    runner.release.set()
    assert owner.wait(first.job_id, 2).state == JobState.CANCELLED
    owner.close()
    assert runner.calls == 1


def test_failure_is_audited_and_next_job_survives(manager) -> None:
    owner, runner = manager
    runner.failure = FoundryError("worker_timeout", "Bounded timeout.", 504)
    job = owner.submit(ResearchRequest(), "a" * 16)
    assert owner.wait(job.job_id, 5).error_code == "worker_timeout"
    assert [event.state for event in owner.store.audit(job.job_id).events] == [
        JobState.QUEUED,
        JobState.RUNNING,
        JobState.FAILED,
    ]
    runner.failure = None
    other = owner.submit(ResearchRequest(), "b" * 16)
    assert owner.wait(other.job_id, 5).state == JobState.SUCCEEDED


def test_comparison_marks_policy_mismatch(manager) -> None:
    owner, _ = manager
    a = owner.submit(ResearchRequest(), "a" * 16)
    b = owner.submit(ResearchRequest(seed=1), "b" * 16)
    owner.wait(a.job_id, 5)
    owner.wait(b.job_id, 5)
    assert not owner.compare(a.job_id, b.job_id).compatible


def test_wait_deadline_and_shutdown(manager) -> None:
    owner, runner = manager
    runner.release.clear()
    job = owner.submit(ResearchRequest(), "a" * 16)
    assert runner.started.wait(2)
    with pytest.raises(FoundryError, match="wait_policy"):
        owner.wait(job.job_id, 0)
    with pytest.raises(FoundryError, match="wait_timeout"):
        owner.wait(job.job_id, 0.001)
    runner.release.set()
    owner.close()
    owner.close()
    with pytest.raises(FoundryError, match="scheduler_unavailable"):
        owner.submit(ResearchRequest(), "b" * 16)


def test_unexpected_fault_stops_admission_without_claiming_success(
    manager, caplog
) -> None:
    owner, runner = manager
    runner.failure = RuntimeError("private fault payload")
    job = owner.submit(ResearchRequest(), "a" * 16)
    with pytest.raises(FoundryError, match="scheduler_unavailable"):
        owner.wait(job.job_id, 5)
    assert owner.store.get(job.job_id).state == JobState.FAILED
    assert "RuntimeError" in caplog.text
    assert "private fault payload" not in caplog.text
    with pytest.raises(FoundryError, match="scheduler_unavailable"):
        owner.submit(ResearchRequest(), "b" * 16)
