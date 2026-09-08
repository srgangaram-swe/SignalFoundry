"""Tests for bounded distributed research execution (SF-S5-MR8).

Grouped by the guarantee each protects. The ones carrying the most weight:

* **Assembly is by task identity, never completion order.** A distributed run
  finishes tasks in a different order every time; anything that folds in arrival
  sequence produces results that vary run to run while every task is
  deterministic.
* **The local backend is the reference and needs no cluster.** Cluster access is
  never required for reproducibility — an explicit non-goal of the work item.
* **A backend that changes results is not an accelerator**, and the parity check
  names the divergent tasks rather than reporting that something differs.
* **Every terminal state is named and attributed** — failure, timeout,
  cancellation — never a bare exception from an anonymous worker.
"""

from __future__ import annotations

import json
from typing import Any

import pytest

from alphaforge.distributed import (
    MIN_USEFUL_PARALLEL_FRACTION,
    BatchReport,
    ExecutionError,
    ProfilingError,
    ResourceRequest,
    SerialProfile,
    StageTiming,
    TaskContractError,
    TaskOutcome,
    TaskResult,
    TaskSpec,
    assert_backend_parity,
    assert_unique_tasks,
    content_hash,
    execute_local,
    execute_process_pool,
    profile_stages,
)


# Module-scope so the process pool can pickle them.
def square(payload: dict[str, Any]) -> int:
    """Deterministic pure work."""
    return int(payload["value"]) ** 2


def always_fails(payload: dict[str, Any]) -> int:
    raise RuntimeError(f"deliberate failure for {payload['value']}")


def fails_then_succeeds(payload: dict[str, Any]) -> int:
    """Fails on the first attempt within a process, then succeeds."""
    marker = f"_attempted_{payload['value']}"
    if not getattr(fails_then_succeeds, marker, False):
        setattr(fails_then_succeeds, marker, True)
        raise RuntimeError("transient")
    return int(payload["value"])


def returns_unhashable(payload: dict[str, Any]) -> object:
    return object()


def slow(payload: dict[str, Any]) -> int:
    total = 0
    for index in range(200_000):
        total += index % 3
    return total


def _resources(**kw: Any) -> ResourceRequest:
    base: dict[str, Any] = {
        "cpus": 1.0,
        "memory_mb": 256,
        "gpus": 0,
        "scratch_mb": 0,
        "expected_seconds": 1.0,
    }
    base.update(kw)
    return ResourceRequest(**base)


def _task(value: int = 1, **kw: Any) -> TaskSpec:
    base: dict[str, Any] = {
        "name": f"task-{value}",
        "payload": {"value": value},
        "seed": value,
        "resources": _resources(),
        "timeout_seconds": 30.0,
    }
    base.update(kw)
    return TaskSpec(**base)


def _batch(count: int = 6) -> list[TaskSpec]:
    return [_task(value) for value in range(count)]


# ---------------------------------------------------------------------------
# Profiling comes before distributing
# ---------------------------------------------------------------------------


def test_the_parallel_fraction_bounds_achievable_speedup() -> None:
    """Amdahl: a 50%-parallel pipeline cannot exceed 2x at any worker count."""
    profile = SerialProfile(
        stages=(
            StageTiming(name="setup", seconds=1.0, parallelizable=False),
            StageTiming(name="sweep", seconds=1.0, parallelizable=True),
        ),
        repeats=1,
        environment={},
    )
    assert profile.parallel_fraction == pytest.approx(0.5)
    assert profile.amdahl_speedup_bound() == pytest.approx(2.0)
    assert profile.amdahl_speedup_bound(8) == pytest.approx(1.0 / (0.5 + 0.5 / 8))


def test_a_fully_parallel_pipeline_has_an_unbounded_ceiling() -> None:
    profile = SerialProfile(
        stages=(StageTiming(name="sweep", seconds=1.0, parallelizable=True),),
        repeats=1,
        environment={},
    )
    assert profile.amdahl_speedup_bound() == float("inf")
    assert profile.distribution_is_justified


def test_a_mostly_serial_pipeline_is_reported_as_unjustified() -> None:
    """The finding that stops someone distributing code that cannot benefit."""
    profile = SerialProfile(
        stages=(
            StageTiming(name="io", seconds=9.0, parallelizable=False),
            StageTiming(name="sweep", seconds=1.0, parallelizable=True),
        ),
        repeats=1,
        environment={},
    )
    assert profile.parallel_fraction == pytest.approx(0.1)
    assert not profile.distribution_is_justified
    assert profile.amdahl_speedup_bound() < 1.2


def test_a_zero_duration_profile_reports_nan_not_zero() -> None:
    """ "No measurable work" and "no parallel work" are different findings."""
    profile = SerialProfile(
        stages=(StageTiming(name="noop", seconds=0.0, parallelizable=True),),
        repeats=1,
        environment={},
    )
    assert profile.parallel_fraction != profile.parallel_fraction  # nan
    assert not profile.distribution_is_justified


def test_profiling_measures_real_stages() -> None:
    calls: list[str] = []
    profile = profile_stages(
        [
            ("a", lambda: calls.append("a"), False),
            ("b", lambda: calls.append("b"), True),
        ],
        repeats=2,
    )
    assert len(profile.stages) == 2
    assert calls.count("a") == 2
    assert profile.total_seconds >= 0.0
    assert json.loads(json.dumps(profile.to_dict()))


def test_duplicate_stage_names_are_refused() -> None:
    with pytest.raises(ProfilingError, match="unique"):
        SerialProfile(
            stages=(
                StageTiming(name="x", seconds=1.0, parallelizable=True),
                StageTiming(name="x", seconds=1.0, parallelizable=False),
            ),
            repeats=1,
            environment={},
        )


def test_a_negative_duration_is_refused() -> None:
    with pytest.raises(ProfilingError, match="non-negative"):
        StageTiming(name="x", seconds=-1.0, parallelizable=True)


def test_an_empty_profile_is_refused() -> None:
    with pytest.raises(ProfilingError, match="at least one stage"):
        SerialProfile(stages=(), repeats=1, environment={})


def test_the_justification_threshold_is_declared() -> None:
    assert 0.0 < MIN_USEFUL_PARALLEL_FRACTION < 1.0


# ---------------------------------------------------------------------------
# Task contracts
# ---------------------------------------------------------------------------


def test_task_identity_is_content_addressed() -> None:
    """The same logical work has the same identity on every machine."""
    assert _task(1).task_id == _task(1).task_id
    assert _task(1).task_id != _task(2).task_id


@pytest.mark.parametrize(
    "override",
    [
        {"seed": 99},
        {"payload": {"value": 1, "extra": True}},
        {"resources": {"cpus": 2.0}},
        {"name": "renamed"},
    ],
)
def test_every_declared_component_changes_identity(override: dict[str, Any]) -> None:
    if "resources" in override:
        override = {"resources": _resources(**override["resources"])}
    assert _task(1).task_id != _task(1, **override).task_id


def test_a_non_serializable_payload_is_refused_where_it_is_written() -> None:
    """A payload whose bytes vary between runs defeats caching and dedup."""
    with pytest.raises(TaskContractError, match="not deterministically serializable"):
        _task(1, payload={"bad": {1, 2, 3}})


def test_a_non_idempotent_task_cannot_request_retries() -> None:
    """Retrying a task with side effects is how one unit of work becomes two."""
    with pytest.raises(TaskContractError, match="non-idempotent"):
        _task(1, idempotent=False, max_retries=2)


def test_a_non_idempotent_task_with_no_retries_is_allowed() -> None:
    assert _task(1, idempotent=False, max_retries=0).max_retries == 0


def test_a_timeout_below_the_declared_duration_is_refused() -> None:
    """The task would be cancelled while behaving exactly as specified."""
    with pytest.raises(TaskContractError, match="below the declared expected duration"):
        _task(1, resources=_resources(expected_seconds=60.0), timeout_seconds=10.0)


@pytest.mark.parametrize(
    "override",
    [{"cpus": 0.0}, {"cpus": -1.0}, {"memory_mb": 0}, {"gpus": -1}, {"scratch_mb": -5}],
)
def test_unusable_resource_declarations_are_refused(override: dict[str, Any]) -> None:
    with pytest.raises(TaskContractError):
        _resources(**override)


def test_an_excessive_retry_budget_is_refused() -> None:
    with pytest.raises(TaskContractError, match="max_retries"):
        _task(1, max_retries=99)


def test_a_batch_with_duplicate_identities_is_refused() -> None:
    """Identical identity means identical work; the batch source has a bug."""
    with pytest.raises(TaskContractError, match="share identity"):
        assert_unique_tasks((_task(1), _task(1)))


def test_a_unique_batch_passes() -> None:
    assert_unique_tasks(tuple(_batch(4)))


def test_content_hash_is_order_independent_for_mappings() -> None:
    assert content_hash({"a": 1, "b": 2}) == content_hash({"b": 2, "a": 1})


# ---------------------------------------------------------------------------
# Local reference backend
# ---------------------------------------------------------------------------


def test_the_local_backend_runs_without_any_cluster() -> None:
    report = execute_local(square, _batch(5))
    assert report.all_succeeded
    assert report.backend == "local"
    assert len(report.results) == 5


def test_results_are_ordered_by_identity_not_submission_order() -> None:
    tasks = _batch(6)
    forward = execute_local(square, tasks)
    reverse = execute_local(square, list(reversed(tasks)))
    assert [r.task_id for r in forward.results] == [r.task_id for r in reverse.results]
    assert forward.assembly_hash() == reverse.assembly_hash()


def test_the_assembly_hash_is_stable_across_runs() -> None:
    assert (
        execute_local(square, _batch(5)).assembly_hash()
        == execute_local(square, _batch(5)).assembly_hash()
    )


def test_a_failing_task_is_attributed_by_name() -> None:
    report = execute_local(always_fails, _batch(3))
    assert not report.all_succeeded
    assert len(report.failures) == 3
    failure = report.failures[0]
    assert failure.outcome is TaskOutcome.FAILED
    assert failure.error is not None
    assert "deliberate failure" in failure.error
    assert failure.name.startswith("task-")


def test_one_failure_does_not_hide_the_other_results() -> None:
    tasks = [_task(0), _task(1)]
    report = execute_local(lambda p: 1 / (p["value"]), tasks)
    assert len(report.results) == 2
    assert len(report.failures) == 1


def test_a_retry_budget_is_honoured() -> None:
    task = _task(7, max_retries=2, idempotent=True)
    report = execute_local(fails_then_succeeds, [task])
    assert report.all_succeeded
    assert report.results[0].attempts == 2


def test_a_task_returning_an_unhashable_value_fails_rather_than_passing() -> None:
    """Parity cannot be checked on a result that has no stable bytes."""
    report = execute_local(returns_unhashable, [_task(1)])
    assert not report.all_succeeded
    error = report.results[0].error
    assert error is not None
    assert "parity cannot be checked" in error


def test_an_empty_batch_is_refused() -> None:
    with pytest.raises(ExecutionError, match="at least one task"):
        execute_local(square, [])


def test_a_total_budget_cancels_the_remainder() -> None:
    report = execute_local(slow, _batch(4), total_timeout_seconds=0.001)
    cancelled = [r for r in report.results if r.outcome is TaskOutcome.CANCELLED]
    assert cancelled
    assert all(r.error is not None for r in cancelled)


# ---------------------------------------------------------------------------
# Process-pool backend and parity
# ---------------------------------------------------------------------------


def test_the_process_pool_matches_the_local_reference_exactly() -> None:
    """A backend that changes results is not an accelerator."""
    tasks = _batch(8)
    reference = execute_local(square, tasks)
    candidate = execute_process_pool(square, tasks, workers=2)
    assert_backend_parity(reference, candidate)
    assert reference.assembly_hash() == candidate.assembly_hash()


def test_parity_holds_regardless_of_worker_count() -> None:
    tasks = _batch(8)
    reference = execute_local(square, tasks)
    for workers in (1, 2, 4):
        assert_backend_parity(reference, execute_process_pool(square, tasks, workers=workers))


def test_parity_failure_names_the_divergent_tasks() -> None:
    tasks = _batch(4)
    reference = execute_local(square, tasks)
    divergent = execute_local(lambda p: int(p["value"]) ** 3, tasks)
    with pytest.raises(ExecutionError, match="produced different output"):
        assert_backend_parity(reference, divergent)


def test_parity_failure_on_a_different_task_set_is_distinguished() -> None:
    with pytest.raises(ExecutionError, match="different task sets"):
        assert_backend_parity(execute_local(square, _batch(4)), execute_local(square, _batch(3)))


def test_a_worker_failure_is_attributed_not_swallowed() -> None:
    report = execute_process_pool(always_fails, _batch(3), workers=2)
    assert not report.all_succeeded
    assert len(report.failures) == 3
    assert all(item.error for item in report.failures)


def test_an_invalid_worker_count_is_refused() -> None:
    with pytest.raises(ExecutionError, match="workers must be a positive int"):
        execute_process_pool(square, _batch(2), workers=0)


def test_an_excessive_worker_count_is_refused() -> None:
    with pytest.raises(ExecutionError, match="ceiling"):
        execute_process_pool(square, _batch(2), workers=10_000)


def test_a_duplicate_task_is_refused_before_any_worker_starts() -> None:
    with pytest.raises(TaskContractError, match="share identity"):
        execute_process_pool(square, [_task(1), _task(1)], workers=2)


def test_a_saturating_batch_still_assembles_deterministically() -> None:
    """More tasks than workers: completion order varies, assembly must not."""
    tasks = _batch(24)
    first = execute_process_pool(square, tasks, workers=4)
    second = execute_process_pool(square, tasks, workers=4)
    assert first.assembly_hash() == second.assembly_hash()
    assert [r.task_id for r in first.results] == [r.task_id for r in second.results]


# ---------------------------------------------------------------------------
# Report contracts
# ---------------------------------------------------------------------------


def test_a_report_cannot_hold_two_results_for_one_task() -> None:
    result = TaskResult(
        task_id="a" * 64, name="x", outcome=TaskOutcome.SUCCEEDED, output_hash="b" * 64
    )
    with pytest.raises(ExecutionError, match="two results for one task"):
        BatchReport(results=(result, result), backend="local", workers=1, wall_seconds=0.0)


def test_a_success_without_an_output_hash_is_refused() -> None:
    with pytest.raises(ExecutionError, match="without an output hash"):
        TaskResult(task_id="a" * 64, name="x", outcome=TaskOutcome.SUCCEEDED)


def test_a_failure_without_a_description_is_refused() -> None:
    """An unattributed failure cannot be acted on."""
    with pytest.raises(ExecutionError, match="without an error"):
        TaskResult(task_id="a" * 64, name="x", outcome=TaskOutcome.FAILED)


def test_the_report_serializes_without_values() -> None:
    payload = execute_local(square, _batch(3)).to_dict()
    assert json.loads(json.dumps(payload))
    assert "value" not in payload["results"][0]
    assert "identity" in payload["ordering_note"]


def test_values_are_returned_in_identity_order() -> None:
    report = execute_local(square, _batch(5))
    assert len(report.values_in_identity_order()) == 5
