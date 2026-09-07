"""Contract tests for repeated distributed-crossover evidence."""

from __future__ import annotations

import hashlib
import inspect
import json
import os
from collections.abc import Callable, Iterator, Sequence
from dataclasses import replace
from functools import partial
from pathlib import Path
from typing import Any, cast

import pytest

import alphaforge.distributed.benchmark_evidence as benchmark_module
import benchmarks.benchmark_distributed_crossover as production_benchmark
from alphaforge.distributed.benchmark_evidence import (
    BENCHMARK_NAME,
    MAX_BENCHMARK_SECONDS,
    MAX_DECLARED_ITERATIONS,
    MAX_EVIDENCE_BYTES,
    MAX_ITERATION_COUNTS,
    MAX_ITERATIONS_PER_TASK,
    MAX_REPETITIONS,
    MAX_TASK_COUNT,
    MAX_WARMUPS,
    MAX_WORKERS,
    MIN_REPETITIONS,
    PRODUCTION_SOURCE_BINDINGS,
    SCHEMA_VERSION,
    TEST_EXECUTION_PROFILE,
    BenchmarkConfig,
    BenchmarkEnvironment,
    BenchmarkEvidence,
    BenchmarkEvidenceError,
    BenchmarkImplementation,
    BenchmarkSample,
    collect_benchmark_environment,
    load_benchmark_evidence,
    parse_benchmark_evidence_bytes,
    run_crossover_benchmark,
    summarize_benchmark,
    task_declaration_graph_sha256,
    verify_production_implementation_sources,
    write_benchmark_evidence,
)
from alphaforge.distributed.executor import BatchReport, TaskOutcome, TaskResult
from alphaforge.distributed.tasks import ResourceRequest, TaskSpec, content_hash


def _sum_workload(payload: dict[str, int]) -> int:
    return payload["index"] + payload["iterations"]


def _index_workload(payload: dict[str, int]) -> int:
    return payload["index"]


def _identity_workload(payload: object) -> object:
    return payload


def _benchmark_harness() -> int:
    return 0


def _alternate_harness() -> int:
    return 0


def _callable_entrypoint(candidate: Callable[..., object]) -> str:
    return f"{candidate.__module__}.{candidate.__qualname__}"


def _implementation(
    *,
    workload: Callable[..., object] = _sum_workload,
    task_builder: Callable[..., object] | None = None,
    harness: Callable[..., object] = _benchmark_harness,
    **changes: object,
) -> BenchmarkImplementation:
    builder = task_builder or _tasks
    repository = Path(__file__).resolve().parents[1]

    def source_sha256(relative: str) -> str:
        return hashlib.sha256((repository / relative).read_bytes()).hexdigest()

    workload_sha256 = source_sha256("tests/test_distributed_benchmark_evidence.py")
    values: dict[str, object] = {
        "workload_name": "deterministic-test-workload",
        "workload_version": "1.2.3",
        "execution_profile": TEST_EXECUTION_PROFILE,
        "workload_entrypoint": _callable_entrypoint(workload),
        "task_builder_entrypoint": _callable_entrypoint(builder),
        "harness_name": "deterministic-test-harness",
        "harness_version": "2.0.0",
        "harness_entrypoint": _callable_entrypoint(harness),
        "serial_executor_entrypoint": "tests.injected_runtime.serial_executor",
        "pool_executor_entrypoint": "tests.injected_runtime.pool_executor",
        "timing_clock_entrypoint": "tests.injected_runtime.timing_clock_ns",
        "budget_clock_entrypoint": "tests.injected_runtime.budget_clock",
        "workload_source_sha256": workload_sha256,
        "task_builder_source_sha256": workload_sha256,
        "harness_source_sha256": workload_sha256,
        "evidence_contract_source_sha256": source_sha256(
            "alphaforge/distributed/benchmark_evidence.py"
        ),
        "executor_source_sha256": source_sha256("alphaforge/distributed/executor.py"),
        "task_contract_source_sha256": source_sha256("alphaforge/distributed/tasks.py"),
        "dependency_lock_sha256": source_sha256("uv.lock"),
    }
    values.update(changes)
    return BenchmarkImplementation(**cast(Any, values))


def _environment() -> BenchmarkEnvironment:
    return BenchmarkEnvironment(
        python_implementation="CPython",
        python_version="3.13.1",
        platform="TestOS",
        platform_release="1.0",
        machine="test64",
        logical_cpu_count=8,
        process_start_method="spawn",
        timing_clock_implementation="deterministic_test_clock",
        timing_clock_resolution_seconds=1e-9,
        timing_clock_monotonic=True,
        timing_clock_adjustable=False,
    )


def _evidence(
    *,
    iteration_counts: tuple[int, ...] = (10, 20),
    implementation: BenchmarkImplementation | None = None,
) -> BenchmarkEvidence:
    config = BenchmarkConfig(
        implementation=implementation or _implementation(),
        task_count=2,
        workers=2,
        warmups=1,
        repetitions=MIN_REPETITIONS,
        iteration_counts=iteration_counts,
    )
    environment = _environment()
    samples: list[BenchmarkSample] = []
    for iterations in config.iteration_counts:
        digest = content_hash({"iterations": iterations})
        task_graph_sha256 = task_declaration_graph_sha256(_tasks(config.task_count, iterations))
        for repetition in range(config.repetitions):
            samples.append(
                BenchmarkSample(
                    iterations=iterations,
                    repetition=repetition,
                    backend_order=(
                        "serial_then_pool" if repetition % 2 == 0 else "pool_then_serial"
                    ),
                    task_count=config.task_count,
                    workers=config.workers,
                    config_sha256=config.identity,
                    environment_sha256=environment.identity,
                    task_graph_sha256=task_graph_sha256,
                    serial_ns=2_000 + 100 * repetition + iterations,
                    pool_ns=3_000 + 50 * repetition + iterations,
                    serial_hash=digest,
                    pool_hash=digest,
                    parity=True,
                )
            )
    return BenchmarkEvidence.from_samples(
        config=config,
        environment=environment,
        samples=samples,
    )


def _tasks(count: int, iterations: int) -> tuple[TaskSpec, ...]:
    return tuple(
        TaskSpec(
            name=f"task-{index}",
            payload={"index": index, "iterations": iterations},
            seed=index,
            resources=ResourceRequest(
                cpus=1.0,
                memory_mb=1,
                gpus=0,
                scratch_mb=0,
                expected_seconds=0.1,
            ),
            timeout_seconds=1.0,
        )
        for index in range(count)
    )


def _alternate_tasks(count: int, iterations: int) -> tuple[TaskSpec, ...]:
    return _tasks(count, iterations)


_UNBOUNDED_YIELDS: list[int] = []


def _unbounded_tasks(count: int, iterations: int) -> Iterator[TaskSpec]:
    del count
    index = 0
    while True:
        _UNBOUNDED_YIELDS.append(index)
        yield _tasks(index + 1, iterations)[-1]
        index += 1


def _short_tasks(count: int, iterations: int) -> tuple[TaskSpec, ...]:
    return _tasks(max(1, count - 1), iterations)


def _noniterable_tasks(count: int, iterations: int) -> Any:
    del count, iterations
    return 1


def _report(
    backend: str,
    workers: int,
    function: Callable[[Any], Any],
    tasks: Sequence[TaskSpec],
    *,
    reverse: bool = False,
    corrupt: bool = False,
) -> BatchReport:
    ordered = tuple(reversed(tasks)) if reverse else tuple(tasks)
    results: list[TaskResult] = []
    for index, task in enumerate(ordered):
        value = function(task.payload)
        digest = content_hash(value)
        if corrupt and index == 0:
            digest = "f" * 64 if digest != "f" * 64 else "e" * 64
        results.append(
            TaskResult(
                task_id=task.task_id,
                name=task.name,
                outcome=TaskOutcome.SUCCEEDED,
                value=value,
                output_hash=digest,
                seconds=0.001,
            )
        )
    return BatchReport(
        results=tuple(results),
        backend=backend,
        workers=workers,
        wall_seconds=0.001,
    )


def _run_test_benchmark(config: BenchmarkConfig, **kwargs: Any) -> BenchmarkEvidence:
    """Run explicit test-profile evidence through the private injection seam."""

    return benchmark_module._run_crossover_benchmark_for_testing(config, **kwargs)


def test_task_graph_identity_is_order_independent_and_binds_full_declarations() -> None:
    tasks = _tasks(3, 11)
    baseline = task_declaration_graph_sha256(tasks)

    assert task_declaration_graph_sha256(tuple(reversed(tasks))) == baseline
    assert (
        task_declaration_graph_sha256((replace(tasks[0], timeout_seconds=2.0), *tasks[1:]))
        != baseline
    )
    assert (
        task_declaration_graph_sha256(
            (replace(tasks[0], payload={"index": 0, "iterations": 12}), *tasks[1:])
        )
        != baseline
    )
    assert (
        task_declaration_graph_sha256((replace(tasks[0], cancellable=False), *tasks[1:]))
        != baseline
    )


def test_task_graph_identity_rejects_empty_malformed_and_duplicate_graphs() -> None:
    with pytest.raises(BenchmarkEvidenceError, match="at least one"):
        task_declaration_graph_sha256(())
    with pytest.raises(BenchmarkEvidenceError, match="TaskSpec"):
        task_declaration_graph_sha256(cast(Any, (object(),)))
    duplicate = _tasks(1, 1)[0]
    with pytest.raises(ValueError, match="share identity"):
        task_declaration_graph_sha256((duplicate, duplicate))


def test_schema_round_trip_summary_and_digests(tmp_path: Path) -> None:
    evidence = _evidence()
    document = evidence.to_dict()

    assert document["schema_version"] == SCHEMA_VERSION
    assert document["benchmark"] == BENCHMARK_NAME
    assert len(document["benchmark_id"]) == 64
    assert len(document["raw_samples_sha256"]) == 64
    assert document["config"]["implementation"] == _implementation().to_dict()
    assert len(document["config"]["implementation"]["workload_source_sha256"]) == 64
    assert document["config"]["repetitions"] == MIN_REPETITIONS
    assert document["config"]["max_total_seconds"] > 0.0
    assert len(document["samples"]) == 2 * MIN_REPETITIONS
    assert document["task_graphs"] == [binding.to_dict() for binding in evidence.task_graphs]
    assert len(document["task_graphs"]) == 2
    assert all(len(row["task_graph_sha256"]) == 64 for row in document["task_graphs"])
    assert document["summary"] == [record.to_dict() for record in evidence.summary]
    assert summarize_benchmark(evidence) == evidence.summary
    assert evidence.summary[0].per_task_ms.median > 0.0
    assert evidence.summary[0].speedup.q1 <= evidence.summary[0].speedup.q3

    path = tmp_path / "crossover.json"
    assert write_benchmark_evidence(evidence, path) == path
    loaded = load_benchmark_evidence(path)
    assert loaded == evidence
    assert parse_benchmark_evidence_bytes(path.read_bytes()) == evidence
    assert loaded.canonical_bytes() == path.read_bytes()
    with pytest.raises(FileExistsError):
        write_benchmark_evidence(evidence, path)


def test_environment_contains_only_public_declared_fields() -> None:
    environment = collect_benchmark_environment().to_dict()

    assert set(environment) == {
        "python_implementation",
        "python_version",
        "platform",
        "platform_release",
        "machine",
        "logical_cpu_count",
        "process_start_method",
        "timing_clock_implementation",
        "timing_clock_resolution_seconds",
        "timing_clock_monotonic",
        "timing_clock_adjustable",
    }
    assert not {"username", "home", "environment", "hostname"}.intersection(environment)


def test_public_runner_has_no_runtime_injection_boundary() -> None:
    config = BenchmarkConfig(
        implementation=_implementation(),
        task_count=2,
        workers=2,
        iteration_counts=(1,),
    )
    assert set(inspect.signature(run_crossover_benchmark).parameters) == {
        "config",
        "function",
        "task_builder",
        "harness",
    }
    for field in (
        "serial_executor",
        "pool_executor",
        "clock_ns",
        "budget_clock",
        "environment",
    ):
        with pytest.raises(TypeError, match="unexpected keyword"):
            run_crossover_benchmark(
                config,
                function=_sum_workload,
                task_builder=_tasks,
                harness=_benchmark_harness,
                **cast(Any, {field: lambda *args: 1}),
            )


def test_public_and_private_runners_refuse_the_opposite_execution_profiles() -> None:
    test_config = BenchmarkConfig(
        implementation=_implementation(),
        task_count=2,
        workers=2,
        iteration_counts=(1,),
    )
    with pytest.raises(BenchmarkEvidenceError, match="production execution profile"):
        run_crossover_benchmark(
            test_config,
            function=_sum_workload,
            task_builder=_tasks,
            harness=_benchmark_harness,
        )

    production_config = BenchmarkConfig(
        implementation=production_benchmark.benchmark_implementation(),
        task_count=2,
        workers=2,
        iteration_counts=(1,),
    )
    with pytest.raises(BenchmarkEvidenceError, match="test_injected"):
        _run_test_benchmark(
            production_config,
            function=production_benchmark.busy_work,
            task_builder=production_benchmark.build_batch,
            harness=production_benchmark.main,
            environment=_environment(),
        )


@pytest.mark.parametrize(
    ("target", "match"),
    [
        ("execute_local", "local executor binding"),
        ("execute_process_pool", "process-pool executor binding"),
        ("_default_serial_executor", "serial adapter binding"),
        ("_default_pool_executor", "process-pool adapter binding"),
        ("collect_benchmark_environment", "environment collector binding"),
        ("_run_crossover_benchmark_with_runtime", "benchmark runtime binding"),
        ("_validate_production_runtime_bindings", "runtime validator binding"),
        ("perf_counter_ns", "timing clock binding"),
        ("monotonic", "budget clock binding"),
    ],
)
def test_public_runner_refuses_replaced_production_runtime_bindings(
    monkeypatch: pytest.MonkeyPatch,
    target: str,
    match: str,
) -> None:
    config = BenchmarkConfig(
        implementation=production_benchmark.benchmark_implementation(),
        task_count=2,
        workers=2,
        iteration_counts=(1,),
    )
    owner = (
        benchmark_module.time if target in {"perf_counter_ns", "monotonic"} else benchmark_module
    )
    monkeypatch.setattr(owner, target, lambda *args, **kwargs: 1)
    with pytest.raises(BenchmarkEvidenceError, match=match):
        run_crossover_benchmark(
            config,
            function=production_benchmark.busy_work,
            task_builder=production_benchmark.build_batch,
            harness=production_benchmark.main,
        )


def test_production_contract_rejects_forged_entrypoints_and_rehashed_sources() -> None:
    implementation = production_benchmark.benchmark_implementation()
    with pytest.raises(BenchmarkEvidenceError, match="workload_entrypoint"):
        replace(
            implementation,
            workload_entrypoint="benchmarks.benchmark_distributed_crossover.other_work",
        )

    repository = Path(__file__).resolve().parents[1]
    records = {
        path: {
            "bytes": (repository / path).stat().st_size,
            "sha256": hashlib.sha256((repository / path).read_bytes()).hexdigest(),
        }
        for _, path in PRODUCTION_SOURCE_BINDINGS
    }
    verify_production_implementation_sources(implementation, records)
    for field, path in PRODUCTION_SOURCE_BINDINGS:
        forged_digest = "a" * 64
        if forged_digest == getattr(implementation, field):
            forged_digest = "b" * 64
        forged = replace(implementation, **{field: forged_digest})
        with pytest.raises(BenchmarkEvidenceError, match=f"{field}.*{path}"):
            verify_production_implementation_sources(forged, records)


@pytest.mark.parametrize(
    ("field", "value", "match"),
    [
        ("workload_name", " padded", "workload_name"),
        ("execution_profile", "production-like", "execution_profile"),
        ("workload_version", "v1", "semantic version"),
        ("harness_version", "01.0.0", "semantic version"),
        ("harness_entrypoint", "main", "fully-qualified"),
        ("workload_entrypoint", "busy_work", "fully-qualified"),
        ("task_builder_entrypoint", "module.bad-name", "fully-qualified"),
        ("serial_executor_entrypoint", "executor", "fully-qualified"),
        ("workload_source_sha256", "A" * 64, "SHA-256"),
        ("dependency_lock_sha256", True, "SHA-256"),
    ],
)
def test_implementation_bindings_fail_closed(field: str, value: object, match: str) -> None:
    with pytest.raises(BenchmarkEvidenceError, match=match):
        _implementation(**cast(Any, {field: value}))


def test_source_identity_changes_config_and_complete_benchmark_identity() -> None:
    original = _evidence()
    changed = _evidence(
        implementation=_implementation(workload_source_sha256="a" * 64),
    )

    assert original.config.identity != changed.config.identity
    assert original.benchmark_id != changed.benchmark_id
    assert original.samples[0].config_sha256 != changed.samples[0].config_sha256


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("task_count", True),
        ("task_count", 0),
        ("task_count", MAX_TASK_COUNT + 1),
        ("workers", False),
        ("workers", 0),
        ("workers", MAX_WORKERS + 1),
        ("warmups", 0),
        ("warmups", MAX_WARMUPS + 1),
        ("repetitions", MIN_REPETITIONS - 1),
        ("repetitions", MAX_REPETITIONS + 1),
        ("max_total_seconds", True),
        ("max_total_seconds", 0.0),
        ("max_total_seconds", float("inf")),
        ("max_total_seconds", MAX_BENCHMARK_SECONDS + 1.0),
    ],
)
def test_config_scalar_bounds_reject_bool_and_overflow(field: str, value: object) -> None:
    settings: dict[str, object] = {
        "implementation": _implementation(),
        "task_count": 2,
        "workers": 2,
        "warmups": 1,
        "repetitions": MIN_REPETITIONS,
        "iteration_counts": (1,),
    }
    settings[field] = value
    with pytest.raises(BenchmarkEvidenceError, match=field):
        BenchmarkConfig(**cast(Any, settings))


@pytest.mark.parametrize(
    "counts",
    [
        (),
        (True,),
        (0,),
        (MAX_ITERATIONS_PER_TASK + 1,),
        (1, 1),
        tuple(range(1, MAX_ITERATION_COUNTS + 2)),
    ],
)
def test_config_iteration_counts_fail_closed(counts: tuple[object, ...]) -> None:
    with pytest.raises(BenchmarkEvidenceError, match="iteration_counts"):
        BenchmarkConfig(
            implementation=_implementation(),
            task_count=2,
            workers=2,
            warmups=1,
            repetitions=MIN_REPETITIONS,
            iteration_counts=cast(Any, counts),
        )


def test_config_rejects_workers_above_tasks_and_aggregate_work() -> None:
    with pytest.raises(BenchmarkEvidenceError, match="workers cannot exceed"):
        BenchmarkConfig(implementation=_implementation(), task_count=1, workers=2)
    assert MAX_DECLARED_ITERATIONS < (
        MAX_ITERATIONS_PER_TASK * MAX_TASK_COUNT * (MAX_WARMUPS + MAX_REPETITIONS) * 2
    )
    with pytest.raises(BenchmarkEvidenceError, match="declared iterations"):
        BenchmarkConfig(
            implementation=_implementation(),
            task_count=MAX_TASK_COUNT,
            workers=1,
            warmups=MAX_WARMUPS,
            repetitions=MAX_REPETITIONS,
            iteration_counts=(MAX_ITERATIONS_PER_TASK,),
        )


@pytest.mark.parametrize(
    "candidate",
    [
        lambda payload: payload,
        partial(_sum_workload),
        abs,
    ],
)
def test_runner_rejects_lambda_partial_and_builtin_workloads(candidate: object) -> None:
    with pytest.raises(BenchmarkEvidenceError, match="module-level|named"):
        _run_test_benchmark(
            BenchmarkConfig(
                implementation=_implementation(),
                task_count=2,
                workers=2,
                iteration_counts=(1,),
            ),
            function=cast(Any, candidate),
            task_builder=_tasks,
            harness=_benchmark_harness,
            environment=_environment(),
        )


def test_runner_rejects_nested_and_mismatched_callable_identities() -> None:
    def nested(payload: object) -> object:
        return payload

    config = BenchmarkConfig(
        implementation=_implementation(),
        task_count=2,
        workers=2,
        iteration_counts=(1,),
    )
    with pytest.raises(BenchmarkEvidenceError, match="named module-level"):
        _run_test_benchmark(
            config,
            function=nested,
            task_builder=_tasks,
            harness=_benchmark_harness,
            environment=_environment(),
        )
    with pytest.raises(BenchmarkEvidenceError, match="declared entrypoint"):
        _run_test_benchmark(
            config,
            function=_index_workload,
            task_builder=_tasks,
            harness=_benchmark_harness,
            environment=_environment(),
        )
    with pytest.raises(BenchmarkEvidenceError, match="declared entrypoint"):
        _run_test_benchmark(
            config,
            function=_sum_workload,
            task_builder=_tasks,
            harness=_alternate_harness,
            environment=_environment(),
        )
    with pytest.raises(BenchmarkEvidenceError, match="declared entrypoint"):
        _run_test_benchmark(
            config,
            function=_sum_workload,
            task_builder=_alternate_tasks,
            harness=_benchmark_harness,
            environment=_environment(),
        )


def test_runner_rejects_callable_source_hash_mismatch() -> None:
    with pytest.raises(BenchmarkEvidenceError, match="declared SHA-256"):
        _run_test_benchmark(
            BenchmarkConfig(
                implementation=_implementation(workload_source_sha256="a" * 64),
                task_count=2,
                workers=2,
                iteration_counts=(1,),
            ),
            function=_sum_workload,
            task_builder=_tasks,
            harness=_benchmark_harness,
            environment=_environment(),
        )


@pytest.mark.parametrize(
    ("field", "match"),
    [
        ("task_builder_source_sha256", "task builder source"),
        ("harness_source_sha256", "harness source"),
        ("evidence_contract_source_sha256", "evidence contract source"),
        ("executor_source_sha256", "executor source"),
        ("task_contract_source_sha256", "task contract source"),
        ("dependency_lock_sha256", "dependency lock source"),
    ],
)
def test_runner_rejects_internal_source_binding_mismatch(field: str, match: str) -> None:
    with pytest.raises(BenchmarkEvidenceError, match=match):
        _run_test_benchmark(
            BenchmarkConfig(
                implementation=_implementation(**cast(Any, {field: "a" * 64})),
                task_count=2,
                workers=2,
                iteration_counts=(1,),
            ),
            function=_sum_workload,
            task_builder=_tasks,
            harness=_benchmark_harness,
            environment=_environment(),
        )


def test_runner_collects_at_most_task_count_plus_one_from_unbounded_builder() -> None:
    _UNBOUNDED_YIELDS.clear()
    with pytest.raises(BenchmarkEvidenceError, match="more than the declared 2"):
        _run_test_benchmark(
            BenchmarkConfig(
                implementation=_implementation(task_builder=_unbounded_tasks),
                task_count=2,
                workers=2,
                iteration_counts=(1,),
            ),
            function=_sum_workload,
            task_builder=_unbounded_tasks,
            harness=_benchmark_harness,
            environment=_environment(),
        )
    assert _UNBOUNDED_YIELDS == [0, 1, 2]


@pytest.mark.parametrize(
    ("builder", "match"),
    [
        (_short_tasks, "returned 1 tasks, expected 2"),
        (_noniterable_tasks, "must return an iterable"),
    ],
)
def test_runner_rejects_short_and_noniterable_task_builders(
    builder: Callable[..., object], match: str
) -> None:
    with pytest.raises(BenchmarkEvidenceError, match=match):
        _run_test_benchmark(
            BenchmarkConfig(
                implementation=_implementation(task_builder=builder),
                task_count=2,
                workers=2,
                iteration_counts=(1,),
            ),
            function=_sum_workload,
            task_builder=cast(Any, builder),
            harness=_benchmark_harness,
            environment=_environment(),
        )


def test_runner_checks_budget_after_non_preemptive_builder_return() -> None:
    readings = iter((0.0, 0.0, 1.0))
    with pytest.raises(BenchmarkEvidenceError, match="total wall-time budget"):
        _run_test_benchmark(
            BenchmarkConfig(
                implementation=_implementation(),
                task_count=2,
                workers=2,
                iteration_counts=(1,),
                max_total_seconds=0.5,
            ),
            function=_sum_workload,
            task_builder=_tasks,
            harness=_benchmark_harness,
            budget_clock=lambda: next(readings),
            environment=_environment(),
        )


def test_runner_alternates_backend_order_and_is_completion_order_independent() -> None:
    calls: list[str] = []

    def serial(
        operation: Callable[[Any], Any],
        tasks: Sequence[TaskSpec],
        total_timeout_seconds: float,
    ) -> BatchReport:
        assert 0.0 < total_timeout_seconds <= 3_600.0
        calls.append("serial")
        return _report("local", 1, operation, tasks)

    def pool(
        operation: Callable[[Any], Any],
        tasks: Sequence[TaskSpec],
        workers: int,
        total_timeout_seconds: float,
    ) -> BatchReport:
        assert 0.0 < total_timeout_seconds <= 3_600.0
        calls.append("pool")
        # Reverse completion order; BatchReport must still assemble by identity.
        return _report("process_pool", workers, operation, tasks, reverse=True)

    clock_values = iter(range(100, 100_000, 100))
    evidence = _run_test_benchmark(
        BenchmarkConfig(
            implementation=_implementation(),
            task_count=2,
            workers=2,
            warmups=1,
            repetitions=MIN_REPETITIONS,
            iteration_counts=(3,),
        ),
        function=_sum_workload,
        task_builder=_tasks,
        harness=_benchmark_harness,
        serial_executor=serial,
        pool_executor=pool,
        clock_ns=lambda: next(clock_values),
        environment=_environment(),
    )

    assert calls[:6] == ["serial", "pool", "serial", "pool", "pool", "serial"]
    assert [row.backend_order for row in evidence.samples[:3]] == [
        "serial_then_pool",
        "pool_then_serial",
        "serial_then_pool",
    ]
    assert all(row.parity and row.serial_hash == row.pool_hash for row in evidence.samples)
    assert all(row.serial_ns == 100 and row.pool_ns == 100 for row in evidence.samples)
    assert evidence.task_graphs[0].task_graph_sha256 == task_declaration_graph_sha256(_tasks(2, 3))


def test_runner_refuses_backend_parity_failure() -> None:
    def serial(
        operation: Callable[[Any], Any],
        tasks: Sequence[TaskSpec],
        total_timeout_seconds: float,
    ) -> BatchReport:
        assert total_timeout_seconds > 0.0
        return _report("local", 1, operation, tasks)

    def corrupt_pool(
        operation: Callable[[Any], Any],
        tasks: Sequence[TaskSpec],
        workers: int,
        total_timeout_seconds: float,
    ) -> BatchReport:
        assert total_timeout_seconds > 0.0
        return _report("process_pool", workers, operation, tasks, corrupt=True)

    with pytest.raises(ValueError, match="changes results|different output"):
        _run_test_benchmark(
            BenchmarkConfig(
                implementation=_implementation(workload=_index_workload),
                task_count=2,
                workers=2,
                warmups=1,
                repetitions=MIN_REPETITIONS,
                iteration_counts=(1,),
            ),
            function=_index_workload,
            task_builder=_tasks,
            harness=_benchmark_harness,
            serial_executor=serial,
            pool_executor=corrupt_pool,
            clock_ns=lambda: 1,
            environment=_environment(),
        )


def test_runner_refuses_results_from_a_different_task_graph() -> None:
    def wrong_serial(
        operation: Callable[[Any], Any],
        tasks: Sequence[TaskSpec],
        total_timeout_seconds: float,
    ) -> BatchReport:
        assert total_timeout_seconds > 0.0
        wrong = _tasks(len(tasks), 2)
        return _report("local", 1, operation, wrong)

    def pool(
        operation: Callable[[Any], Any],
        tasks: Sequence[TaskSpec],
        workers: int,
        total_timeout_seconds: float,
    ) -> BatchReport:
        assert total_timeout_seconds > 0.0
        return _report("process_pool", workers, operation, tasks)

    with pytest.raises(BenchmarkEvidenceError, match="bound task graph"):
        _run_test_benchmark(
            BenchmarkConfig(
                implementation=_implementation(),
                task_count=2,
                workers=2,
                iteration_counts=(1,),
            ),
            function=_sum_workload,
            task_builder=_tasks,
            harness=_benchmark_harness,
            serial_executor=wrong_serial,
            pool_executor=pool,
            environment=_environment(),
        )


def test_runner_enforces_one_wall_deadline_across_warmups_and_samples() -> None:
    calls: list[tuple[str, float]] = []

    def serial(
        operation: Callable[[Any], Any],
        tasks: Sequence[TaskSpec],
        total_timeout_seconds: float,
    ) -> BatchReport:
        calls.append(("serial", total_timeout_seconds))
        return _report("local", 1, operation, tasks)

    def pool(
        operation: Callable[[Any], Any],
        tasks: Sequence[TaskSpec],
        workers: int,
        total_timeout_seconds: float,
    ) -> BatchReport:
        calls.append(("pool", total_timeout_seconds))
        return _report("process_pool", workers, operation, tasks)

    # Start, bounded graph collection, pre-serial warmup, then post-serial expiry.
    budget_readings = iter((*([0.0] * 9), 1.0))
    with pytest.raises(BenchmarkEvidenceError, match="total wall-time budget"):
        _run_test_benchmark(
            BenchmarkConfig(
                implementation=_implementation(workload=_index_workload),
                task_count=2,
                workers=2,
                warmups=1,
                repetitions=MIN_REPETITIONS,
                iteration_counts=(1,),
                max_total_seconds=0.5,
            ),
            function=_index_workload,
            task_builder=_tasks,
            harness=_benchmark_harness,
            serial_executor=serial,
            pool_executor=pool,
            budget_clock=lambda: next(budget_readings),
            environment=_environment(),
        )

    assert calls == [("serial", 0.5)]


def test_runner_rejects_non_monotonic_budget_clock() -> None:
    readings = iter((2.0, 1.0))
    with pytest.raises(BenchmarkEvidenceError, match="monotonic"):
        _run_test_benchmark(
            BenchmarkConfig(
                implementation=_implementation(workload=_identity_workload),
                task_count=2,
                workers=2,
                warmups=1,
                repetitions=MIN_REPETITIONS,
                iteration_counts=(1,),
            ),
            function=_identity_workload,
            task_builder=_tasks,
            harness=_benchmark_harness,
            budget_clock=lambda: next(readings),
            environment=_environment(),
        )


def test_raw_order_does_not_change_summary_or_identity() -> None:
    evidence = _evidence(iteration_counts=(20, 10))
    reordered = BenchmarkEvidence(
        config=evidence.config,
        environment=evidence.environment,
        samples=tuple(reversed(evidence.samples)),
        summary=tuple(reversed(evidence.summary)),
        limitations=evidence.limitations,
    )

    assert reordered.benchmark_id == evidence.benchmark_id
    assert reordered.canonical_bytes() == evidence.canonical_bytes()


@pytest.mark.parametrize(
    ("mutation", "match"),
    [
        (lambda doc: doc.update({"unknown": 1}), "unknown"),
        (lambda doc: doc.update({"schema_version": "4.0.0"}), "unsupported schema"),
        (lambda doc: doc["config"].update({"task_count": True}), "task_count"),
        (lambda doc: doc["samples"][0].update({"serial_ns": -1}), "serial_ns"),
        (lambda doc: doc["samples"][0].update({"parity": False}), "parity"),
        (
            lambda doc: doc["samples"][0].update({"environment_sha256": "a" * 64}),
            "environment identity",
        ),
        (
            lambda doc: doc["samples"][0].update({"task_graph_sha256": "a" * 64}),
            "exact task graph",
        ),
        (
            lambda doc: doc["task_graphs"][0].update({"task_graph_sha256": "b" * 64}),
            "task graph index",
        ),
        (
            lambda doc: doc["config"]["implementation"].update({"harness_source_sha256": "c" * 64}),
            "config identity",
        ),
        (
            lambda doc: doc["config"]["implementation"].update(
                {
                    "execution_profile": "production",
                    "workload_entrypoint": "benchmarks.benchmark_distributed_crossover.other_work",
                }
            ),
            "production implementation requires",
        ),
        (lambda doc: doc["summary"][0]["speedup"].update({"median": 999.0}), "summary"),
        (lambda doc: doc.update({"raw_samples_sha256": "a" * 64}), "raw sample digest"),
        (lambda doc: doc.update({"benchmark_id": "b" * 64}), "benchmark identity"),
    ],
)
def test_malformed_documents_fail_closed(
    mutation: Callable[[dict[str, Any]], None], match: str
) -> None:
    document = json.loads(json.dumps(_evidence().to_dict()))
    mutation(document)
    with pytest.raises(BenchmarkEvidenceError, match=match):
        BenchmarkEvidence.from_dict(document)


def test_duplicate_and_missing_samples_fail_closed() -> None:
    evidence = _evidence()
    with pytest.raises(BenchmarkEvidenceError, match="duplicate"):
        BenchmarkEvidence(
            config=evidence.config,
            environment=evidence.environment,
            samples=(*evidence.samples[:-1], evidence.samples[0]),
            summary=evidence.summary,
        )
    with pytest.raises(BenchmarkEvidenceError, match="expected"):
        BenchmarkEvidence(
            config=evidence.config,
            environment=evidence.environment,
            samples=evidence.samples[:-1],
            summary=evidence.summary,
        )


@pytest.mark.parametrize(
    "payload",
    [
        '{"schema_version":"1.0.0","schema_version":"1.0.0"}',
        '{"schema_version":NaN}',
    ],
)
def test_loader_rejects_duplicate_keys_and_nonfinite_json(tmp_path: Path, payload: str) -> None:
    path = tmp_path / "malformed.json"
    path.write_text(payload, encoding="utf-8")
    with pytest.raises(BenchmarkEvidenceError):
        load_benchmark_evidence(path)


def test_snapshot_parser_rejects_mutable_empty_oversized_and_malformed_bytes() -> None:
    with pytest.raises(BenchmarkEvidenceError, match="immutable bytes"):
        parse_benchmark_evidence_bytes(cast(Any, bytearray(b"{}")))
    with pytest.raises(BenchmarkEvidenceError, match="bytes must lie"):
        parse_benchmark_evidence_bytes(b"")
    with pytest.raises(BenchmarkEvidenceError, match="bytes must lie"):
        parse_benchmark_evidence_bytes(b" " * (MAX_EVIDENCE_BYTES + 1))
    with pytest.raises(BenchmarkEvidenceError, match="strict JSON"):
        parse_benchmark_evidence_bytes(b'{"schema_version":"2.0.0","schema_version":"2.0.0"}')


def test_loader_rejects_symlink(tmp_path: Path) -> None:
    source = tmp_path / "source.json"
    source.write_bytes(_evidence().canonical_bytes())
    link = tmp_path / "link.json"
    link.symlink_to(source)
    with pytest.raises(BenchmarkEvidenceError, match="regular file"):
        load_benchmark_evidence(link)


def test_loader_rejects_oversized_and_deep_documents(tmp_path: Path) -> None:
    oversized = tmp_path / "oversized.json"
    oversized.write_bytes(b" " * (MAX_EVIDENCE_BYTES + 1))
    with pytest.raises(BenchmarkEvidenceError, match="bounded regular file"):
        load_benchmark_evidence(oversized)

    deeply_nested = tmp_path / "deep.json"
    deeply_nested.write_text("[" * 20 + "0" + "]" * 20, encoding="utf-8")
    with pytest.raises(BenchmarkEvidenceError, match="depth"):
        load_benchmark_evidence(deeply_nested)


def test_writer_rejects_symlinked_parent(tmp_path: Path) -> None:
    actual = tmp_path / "actual"
    actual.mkdir()
    alias = tmp_path / "alias"
    alias.symlink_to(actual, target_is_directory=True)

    with pytest.raises(BenchmarkEvidenceError, match="traverses a symlink"):
        write_benchmark_evidence(_evidence(), alias / "crossover.json")
    assert not (actual / "crossover.json").exists()


def test_writer_does_not_create_descendants_through_symlinked_ancestry(tmp_path: Path) -> None:
    outside = tmp_path / "outside"
    outside.mkdir()
    alias = tmp_path / "alias"
    alias.symlink_to(outside, target_is_directory=True)

    with pytest.raises(BenchmarkEvidenceError, match="traverses a symlink"):
        write_benchmark_evidence(_evidence(), alias / "new" / "crossover.json")

    assert list(outside.iterdir()) == []


def test_writer_safely_creates_nested_non_symlink_parents(tmp_path: Path) -> None:
    destination = tmp_path / "one" / "two" / "crossover.json"

    assert write_benchmark_evidence(_evidence(), destination) == destination
    assert load_benchmark_evidence(destination) == _evidence()


def _replace_directory_entry(
    directory_descriptor: int,
    name: str,
    payload: bytes,
) -> None:
    os.unlink(name, dir_fd=directory_descriptor)
    descriptor = os.open(
        name,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL,
        0o600,
        dir_fd=directory_descriptor,
    )
    with os.fdopen(descriptor, "wb") as handle:
        handle.write(payload)


def test_writer_detects_replacement_immediately_after_link_without_deleting_it(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    destination = tmp_path / "crossover.json"
    replacement = b"concurrent replacement after link\n"
    real_link = benchmark_module.os.link

    def replacing_link(*args: Any, **kwargs: Any) -> None:
        real_link(*args, **kwargs)
        _replace_directory_entry(
            cast(int, kwargs["dst_dir_fd"]),
            cast(str, args[1]),
            replacement,
        )

    monkeypatch.setattr(benchmark_module.os, "link", replacing_link)
    with pytest.raises(BenchmarkEvidenceError, match="destination changed"):
        write_benchmark_evidence(_evidence(), destination)

    assert destination.read_bytes() == replacement
    assert not list(tmp_path.glob(".distributed-benchmark-*.tmp"))


def test_writer_detects_replacement_after_verified_read_without_deleting_it(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    destination = tmp_path / "crossover.json"
    replacement = b"concurrent replacement after verified read\n"
    real_load = benchmark_module._load_published_evidence

    def replacing_load(
        directory_descriptor: int,
        name: str,
        *,
        expected: os.stat_result,
    ) -> tuple[BenchmarkEvidence, os.stat_result]:
        result = real_load(directory_descriptor, name, expected=expected)
        _replace_directory_entry(directory_descriptor, name, replacement)
        return result

    monkeypatch.setattr(benchmark_module, "_load_published_evidence", replacing_load)
    with pytest.raises(BenchmarkEvidenceError, match="destination changed"):
        write_benchmark_evidence(_evidence(), destination)

    assert destination.read_bytes() == replacement
    assert not list(tmp_path.glob(".distributed-benchmark-*.tmp"))
