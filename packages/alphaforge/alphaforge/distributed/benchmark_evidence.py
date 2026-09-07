"""Strict evidence contract for the local distributed-crossover benchmark.

The crossover is performance *evidence*, not a performance gate.  This module
keeps every measured ``perf_counter_ns`` duration, proves local/process-pool
semantic parity for every repetition, derives summaries from those raw rows,
and gives the complete document a canonical SHA-256 identity.

Only public runtime metadata is collected.  User names, home paths, environment
variables, credentials, account identifiers, and market data are never read.
"""

from __future__ import annotations

import errno
import hashlib
import inspect
import json
import math
import multiprocessing
import os
import platform as platform_module
import re
import secrets
import stat
import statistics
import time
from collections.abc import Callable, Iterable, Mapping, Sequence
from contextlib import suppress
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Any, Final

from alphaforge.distributed.executor import (
    BatchReport,
    assert_backend_parity,
    execute_local,
    execute_process_pool,
)
from alphaforge.distributed.tasks import TaskSpec, assert_unique_tasks
from alphaforge.research._bounded_io import (
    BoundedIOError,
    parse_strict_json,
    read_regular_file_snapshot,
)

SCHEMA_VERSION: Final = "3.0.0"
BENCHMARK_NAME: Final = "local_process_pool_distributed_crossover"
TIMING_CLOCK: Final = "perf_counter_ns"
PRODUCTION_EXECUTION_PROFILE: Final = "production"
TEST_EXECUTION_PROFILE: Final = "test_injected"
PRODUCTION_WORKLOAD_NAME: Final = "deterministic-sqrt-modulo-seven-cpu-probe"
PRODUCTION_WORKLOAD_VERSION: Final = "1.0.0"
PRODUCTION_WORKLOAD_ENTRYPOINT: Final = "benchmarks.benchmark_distributed_crossover.busy_work"
PRODUCTION_TASK_BUILDER_ENTRYPOINT: Final = "benchmarks.benchmark_distributed_crossover.build_batch"
PRODUCTION_HARNESS_NAME: Final = "local-process-pool-distributed-crossover"
PRODUCTION_HARNESS_VERSION: Final = "3.0.0"
PRODUCTION_HARNESS_ENTRYPOINT: Final = "benchmarks.benchmark_distributed_crossover.main"
PRODUCTION_SERIAL_EXECUTOR_ENTRYPOINT: Final = "alphaforge.distributed.executor.execute_local"
PRODUCTION_POOL_EXECUTOR_ENTRYPOINT: Final = "alphaforge.distributed.executor.execute_process_pool"
PRODUCTION_TIMING_CLOCK_ENTRYPOINT: Final = "time.perf_counter_ns"
PRODUCTION_BUDGET_CLOCK_ENTRYPOINT: Final = "time.monotonic"
PRODUCTION_IMPLEMENTATION_BINDINGS: Final[tuple[tuple[str, str], ...]] = (
    ("workload_name", PRODUCTION_WORKLOAD_NAME),
    ("workload_version", PRODUCTION_WORKLOAD_VERSION),
    ("workload_entrypoint", PRODUCTION_WORKLOAD_ENTRYPOINT),
    ("task_builder_entrypoint", PRODUCTION_TASK_BUILDER_ENTRYPOINT),
    ("harness_name", PRODUCTION_HARNESS_NAME),
    ("harness_version", PRODUCTION_HARNESS_VERSION),
    ("harness_entrypoint", PRODUCTION_HARNESS_ENTRYPOINT),
    ("serial_executor_entrypoint", PRODUCTION_SERIAL_EXECUTOR_ENTRYPOINT),
    ("pool_executor_entrypoint", PRODUCTION_POOL_EXECUTOR_ENTRYPOINT),
    ("timing_clock_entrypoint", PRODUCTION_TIMING_CLOCK_ENTRYPOINT),
    ("budget_clock_entrypoint", PRODUCTION_BUDGET_CLOCK_ENTRYPOINT),
)
PRODUCTION_SOURCE_BINDINGS: Final[tuple[tuple[str, str], ...]] = (
    ("workload_source_sha256", "benchmarks/benchmark_distributed_crossover.py"),
    ("task_builder_source_sha256", "benchmarks/benchmark_distributed_crossover.py"),
    ("harness_source_sha256", "benchmarks/benchmark_distributed_crossover.py"),
    (
        "evidence_contract_source_sha256",
        "alphaforge/distributed/benchmark_evidence.py",
    ),
    ("executor_source_sha256", "alphaforge/distributed/executor.py"),
    ("task_contract_source_sha256", "alphaforge/distributed/tasks.py"),
    ("dependency_lock_sha256", "uv.lock"),
)
MIN_REPETITIONS: Final = 7
MAX_REPETITIONS: Final = 100
MIN_WARMUPS: Final = 1
MAX_WARMUPS: Final = 20
MAX_ITERATION_COUNTS: Final = 32
MAX_ITERATIONS_PER_TASK: Final = 10_000_000
MAX_TASK_COUNT: Final = 4_096
MAX_WORKERS: Final = 256
MAX_TOTAL_TASK_RUNS: Final = 1_000_000
MAX_DECLARED_ITERATIONS: Final = 5_000_000_000
MAX_SAMPLE_NS: Final = 24 * 60 * 60 * 1_000_000_000
DEFAULT_MAX_TOTAL_SECONDS: Final = 60 * 60.0
MAX_BENCHMARK_SECONDS: Final = 24 * 60 * 60.0
MAX_EVIDENCE_BYTES: Final = 8_000_000
MAX_JSON_DEPTH: Final = 16
MAX_JSON_NODES: Final = 100_000
MAX_TEXT_CHARS: Final = 512
MAX_IDENTITY_SOURCE_BYTES: Final = 20_000_000
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_SEMANTIC_VERSION = re.compile(
    r"^(?:0|[1-9][0-9]*)\.(?:0|[1-9][0-9]*)\.(?:0|[1-9][0-9]*)"
    r"(?:-[0-9A-Za-z]+(?:[.-][0-9A-Za-z]+)*)?(?:\+[0-9A-Za-z]+(?:[.-][0-9A-Za-z]+)*)?$"
)
_ENTRYPOINT = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z_][A-Za-z0-9_]*)+$")
_BACKEND_ORDERS = frozenset({"serial_then_pool", "pool_then_serial"})

BenchmarkFunction = Callable[[Any], Any]
TaskBuilder = Callable[[int, int], Iterable[TaskSpec]]
SerialExecutor = Callable[[BenchmarkFunction, Sequence[TaskSpec], float], BatchReport]
PoolExecutor = Callable[[BenchmarkFunction, Sequence[TaskSpec], int, float], BatchReport]
Clock = Callable[[], int]
BudgetClock = Callable[[], float]

_PRODUCTION_LOCAL_EXECUTOR: Final = execute_local
_PRODUCTION_POOL_EXECUTOR: Final = execute_process_pool
_PRODUCTION_TIMING_CLOCK: Final = time.perf_counter_ns
_PRODUCTION_BUDGET_CLOCK: Final = time.monotonic


class BenchmarkEvidenceError(ValueError):
    """Raised when benchmark evidence is unsafe, inconsistent, or malformed."""


def _canonical_bytes(value: object) -> bytes:
    try:
        return json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise BenchmarkEvidenceError("evidence must be finite canonical JSON") from exc


def _digest(value: object) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _mapping(value: object, expected: set[str], *, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise BenchmarkEvidenceError(f"{name} must be an object")
    keys = set(value)
    missing = sorted(expected - keys)
    unknown = sorted(keys - expected)
    if missing or unknown:
        raise BenchmarkEvidenceError(f"{name} keys differ: missing={missing}, unknown={unknown}")
    if any(not isinstance(key, str) for key in value):
        raise BenchmarkEvidenceError(f"{name} keys must be strings")
    return value


def _integer(value: object, *, name: str, minimum: int, maximum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise BenchmarkEvidenceError(f"{name} must be an integer, not a bool")
    if not minimum <= value <= maximum:
        raise BenchmarkEvidenceError(f"{name} must lie in [{minimum}, {maximum}]")
    return value


def _real(
    value: object,
    *,
    name: str,
    minimum: float = 0.0,
    maximum: float | None = None,
) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise BenchmarkEvidenceError(f"{name} must be a real number, not a bool")
    result = float(value)
    if not math.isfinite(result) or result < minimum:
        raise BenchmarkEvidenceError(f"{name} must be finite and at least {minimum}")
    if maximum is not None and result > maximum:
        raise BenchmarkEvidenceError(f"{name} must be at most {maximum}")
    return result


def _text(value: object, *, name: str, maximum: int = MAX_TEXT_CHARS) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        raise BenchmarkEvidenceError(f"{name} must be a non-empty, trimmed string")
    if len(value) > maximum or any(ord(character) < 32 for character in value):
        raise BenchmarkEvidenceError(f"{name} exceeds its safe text contract")
    return value


def _sha256(value: object, *, name: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise BenchmarkEvidenceError(f"{name} must be a lowercase SHA-256 digest")
    return value


def _semantic_version(value: object, *, name: str) -> str:
    version = _text(value, name=name, maximum=128)
    if _SEMANTIC_VERSION.fullmatch(version) is None:
        raise BenchmarkEvidenceError(f"{name} must be a semantic version")
    return version


def _entrypoint(value: object, *, name: str) -> str:
    entrypoint = _text(value, name=name, maximum=256)
    if _ENTRYPOINT.fullmatch(entrypoint) is None:
        raise BenchmarkEvidenceError(f"{name} must be a fully-qualified Python entrypoint")
    return entrypoint


@dataclass(frozen=True, slots=True)
class BenchmarkImplementation:
    """Versioned workload and exact source identities used by one run.

    The benchmark is reproducible only when readers can distinguish a changed
    workload, task builder, harness, execution contract, or dependency lock.
    These hashes identify immutable bytes; the human-readable names and semantic
    versions communicate which contract those bytes claim to implement.
    """

    workload_name: str
    workload_version: str
    execution_profile: str
    workload_entrypoint: str
    task_builder_entrypoint: str
    harness_name: str
    harness_version: str
    harness_entrypoint: str
    serial_executor_entrypoint: str
    pool_executor_entrypoint: str
    timing_clock_entrypoint: str
    budget_clock_entrypoint: str
    workload_source_sha256: str
    task_builder_source_sha256: str
    harness_source_sha256: str
    evidence_contract_source_sha256: str
    executor_source_sha256: str
    task_contract_source_sha256: str
    dependency_lock_sha256: str

    def __post_init__(self) -> None:
        profile = _text(self.execution_profile, name="execution_profile", maximum=64)
        if profile not in {PRODUCTION_EXECUTION_PROFILE, TEST_EXECUTION_PROFILE}:
            raise BenchmarkEvidenceError("execution_profile is unsupported")
        object.__setattr__(self, "execution_profile", profile)
        for name in ("workload_name", "harness_name"):
            object.__setattr__(self, name, _text(getattr(self, name), name=name, maximum=128))
        for name in ("workload_version", "harness_version"):
            object.__setattr__(
                self,
                name,
                _semantic_version(getattr(self, name), name=name),
            )
        for name in (
            "workload_entrypoint",
            "task_builder_entrypoint",
            "harness_entrypoint",
            "serial_executor_entrypoint",
            "pool_executor_entrypoint",
            "timing_clock_entrypoint",
            "budget_clock_entrypoint",
        ):
            object.__setattr__(self, name, _entrypoint(getattr(self, name), name=name))
        for name in (
            "workload_source_sha256",
            "task_builder_source_sha256",
            "harness_source_sha256",
            "evidence_contract_source_sha256",
            "executor_source_sha256",
            "task_contract_source_sha256",
            "dependency_lock_sha256",
        ):
            object.__setattr__(self, name, _sha256(getattr(self, name), name=name))
        if profile == PRODUCTION_EXECUTION_PROFILE:
            for name, expected in PRODUCTION_IMPLEMENTATION_BINDINGS:
                if getattr(self, name) != expected:
                    raise BenchmarkEvidenceError(
                        f"production implementation requires {name}={expected!r}"
                    )

    @property
    def identity(self) -> str:
        """Return the canonical implementation/source identity."""

        return _digest(self.to_dict())

    def to_dict(self) -> dict[str, str]:
        """Return the strict versioned source-binding document."""

        return {
            "workload_name": self.workload_name,
            "workload_version": self.workload_version,
            "execution_profile": self.execution_profile,
            "workload_entrypoint": self.workload_entrypoint,
            "task_builder_entrypoint": self.task_builder_entrypoint,
            "harness_name": self.harness_name,
            "harness_version": self.harness_version,
            "harness_entrypoint": self.harness_entrypoint,
            "serial_executor_entrypoint": self.serial_executor_entrypoint,
            "pool_executor_entrypoint": self.pool_executor_entrypoint,
            "timing_clock_entrypoint": self.timing_clock_entrypoint,
            "budget_clock_entrypoint": self.budget_clock_entrypoint,
            "workload_source_sha256": self.workload_source_sha256,
            "task_builder_source_sha256": self.task_builder_source_sha256,
            "harness_source_sha256": self.harness_source_sha256,
            "evidence_contract_source_sha256": self.evidence_contract_source_sha256,
            "executor_source_sha256": self.executor_source_sha256,
            "task_contract_source_sha256": self.task_contract_source_sha256,
            "dependency_lock_sha256": self.dependency_lock_sha256,
        }

    @classmethod
    def from_dict(cls, value: object) -> BenchmarkImplementation:
        """Parse a strict implementation/source-binding document."""

        expected = {
            "workload_name",
            "workload_version",
            "execution_profile",
            "workload_entrypoint",
            "task_builder_entrypoint",
            "harness_name",
            "harness_version",
            "harness_entrypoint",
            "serial_executor_entrypoint",
            "pool_executor_entrypoint",
            "timing_clock_entrypoint",
            "budget_clock_entrypoint",
            "workload_source_sha256",
            "task_builder_source_sha256",
            "harness_source_sha256",
            "evidence_contract_source_sha256",
            "executor_source_sha256",
            "task_contract_source_sha256",
            "dependency_lock_sha256",
        }
        data = _mapping(value, expected, name="implementation")
        return cls(**{key: data[key] for key in expected})


def require_production_implementation(implementation: BenchmarkImplementation) -> None:
    """Require the exact public benchmark execution contract.

    A test-injected run is valid internal test evidence, but it must never be
    published as the measured Sprint 5 production benchmark.
    """

    if not isinstance(implementation, BenchmarkImplementation):
        raise BenchmarkEvidenceError("implementation must be BenchmarkImplementation")
    if implementation.execution_profile != PRODUCTION_EXECUTION_PROFILE:
        raise BenchmarkEvidenceError(
            "published benchmark requires the production execution profile"
        )
    for name, expected in PRODUCTION_IMPLEMENTATION_BINDINGS:
        if getattr(implementation, name) != expected:
            raise BenchmarkEvidenceError(f"published benchmark implementation differs at {name}")


def verify_production_implementation_sources(
    implementation: BenchmarkImplementation,
    source_records: Mapping[str, object],
) -> None:
    """Reconcile every declared implementation digest to an exact source path.

    ``source_records`` is the already repository-verified manifest mapping.  Its
    byte-size fields remain part of that outer manifest contract; this function
    binds each benchmark implementation role to the corresponding SHA-256.
    """

    require_production_implementation(implementation)
    if not isinstance(source_records, Mapping) or any(
        not isinstance(path, str) for path in source_records
    ):
        raise BenchmarkEvidenceError("implementation source records must be a string-keyed object")
    for field, path in PRODUCTION_SOURCE_BINDINGS:
        if path not in source_records:
            raise BenchmarkEvidenceError(f"implementation source record is missing: {path}")
        record = _mapping(
            source_records[path],
            {"bytes", "sha256"},
            name=f"implementation source record {path}",
        )
        _integer(
            record["bytes"],
            name=f"implementation source record {path}.bytes",
            minimum=1,
            maximum=MAX_IDENTITY_SOURCE_BYTES,
        )
        observed = _sha256(
            record["sha256"],
            name=f"implementation source record {path}.sha256",
        )
        if getattr(implementation, field) != observed:
            raise BenchmarkEvidenceError(
                f"benchmark {field} does not match repository source {path}"
            )


@dataclass(frozen=True, slots=True)
class BenchmarkConfig:
    """Finite workload and repetition bounds for one benchmark run."""

    implementation: BenchmarkImplementation
    task_count: int = 32
    workers: int = 8
    warmups: int = 1
    repetitions: int = 7
    iteration_counts: tuple[int, ...] = (1_000, 50_000, 500_000, 2_000_000)
    timing_clock: str = TIMING_CLOCK
    max_total_seconds: float = DEFAULT_MAX_TOTAL_SECONDS

    def __post_init__(self) -> None:
        if not isinstance(self.implementation, BenchmarkImplementation):
            raise BenchmarkEvidenceError("implementation must be BenchmarkImplementation")
        task_count = _integer(self.task_count, name="task_count", minimum=1, maximum=MAX_TASK_COUNT)
        workers = _integer(self.workers, name="workers", minimum=1, maximum=MAX_WORKERS)
        if workers > task_count:
            raise BenchmarkEvidenceError("workers cannot exceed task_count")
        warmups = _integer(self.warmups, name="warmups", minimum=MIN_WARMUPS, maximum=MAX_WARMUPS)
        repetitions = _integer(
            self.repetitions,
            name="repetitions",
            minimum=MIN_REPETITIONS,
            maximum=MAX_REPETITIONS,
        )
        if isinstance(self.iteration_counts, (str, bytes)):
            raise BenchmarkEvidenceError("iteration_counts must be a sequence of integers")
        counts = tuple(
            _integer(
                value,
                name=f"iteration_counts[{index}]",
                minimum=1,
                maximum=MAX_ITERATIONS_PER_TASK,
            )
            for index, value in enumerate(self.iteration_counts)
        )
        if not counts or len(counts) > MAX_ITERATION_COUNTS:
            raise BenchmarkEvidenceError(
                f"iteration_counts must contain between 1 and {MAX_ITERATION_COUNTS} values"
            )
        if len(set(counts)) != len(counts):
            raise BenchmarkEvidenceError("iteration_counts must not contain duplicates")
        counts = tuple(sorted(counts))
        total_task_runs = len(counts) * (warmups + repetitions) * task_count * 2
        if total_task_runs > MAX_TOTAL_TASK_RUNS:
            raise BenchmarkEvidenceError(
                f"declared task runs exceed the {MAX_TOTAL_TASK_RUNS} ceiling"
            )
        declared_iterations = sum(counts) * (warmups + repetitions) * task_count * 2
        if declared_iterations > MAX_DECLARED_ITERATIONS:
            raise BenchmarkEvidenceError(
                f"declared iterations exceed the {MAX_DECLARED_ITERATIONS} ceiling"
            )
        if self.timing_clock != TIMING_CLOCK:
            raise BenchmarkEvidenceError(f"timing_clock must be {TIMING_CLOCK!r}")
        max_total_seconds = _real(
            self.max_total_seconds,
            name="max_total_seconds",
            minimum=0.0,
            maximum=MAX_BENCHMARK_SECONDS,
        )
        if max_total_seconds <= 0.0:
            raise BenchmarkEvidenceError("max_total_seconds must be positive")
        object.__setattr__(self, "task_count", task_count)
        object.__setattr__(self, "workers", workers)
        object.__setattr__(self, "warmups", warmups)
        object.__setattr__(self, "repetitions", repetitions)
        object.__setattr__(self, "iteration_counts", counts)
        object.__setattr__(self, "max_total_seconds", max_total_seconds)

    @property
    def identity(self) -> str:
        """Return the canonical workload identity."""

        return _digest(self.to_dict())

    def to_dict(self) -> dict[str, Any]:
        """Return the strict JSON configuration."""

        return {
            "implementation": self.implementation.to_dict(),
            "task_count": self.task_count,
            "workers": self.workers,
            "warmups": self.warmups,
            "repetitions": self.repetitions,
            "iteration_counts": list(self.iteration_counts),
            "timing_clock": self.timing_clock,
            "max_total_seconds": self.max_total_seconds,
        }

    @classmethod
    def from_dict(cls, value: object) -> BenchmarkConfig:
        """Parse a strict configuration object."""

        data = _mapping(
            value,
            {
                "implementation",
                "task_count",
                "workers",
                "warmups",
                "repetitions",
                "iteration_counts",
                "timing_clock",
                "max_total_seconds",
            },
            name="config",
        )
        counts = data["iteration_counts"]
        if not isinstance(counts, list):
            raise BenchmarkEvidenceError("config.iteration_counts must be an array")
        return cls(
            implementation=BenchmarkImplementation.from_dict(data["implementation"]),
            task_count=data["task_count"],
            workers=data["workers"],
            warmups=data["warmups"],
            repetitions=data["repetitions"],
            iteration_counts=tuple(counts),
            timing_clock=data["timing_clock"],
            max_total_seconds=data["max_total_seconds"],
        )


@dataclass(frozen=True, slots=True)
class BenchmarkEnvironment:
    """Public, non-user-specific runtime metadata for one benchmark process."""

    python_implementation: str
    python_version: str
    platform: str
    platform_release: str
    machine: str
    logical_cpu_count: int
    process_start_method: str
    timing_clock_implementation: str
    timing_clock_resolution_seconds: float
    timing_clock_monotonic: bool
    timing_clock_adjustable: bool

    def __post_init__(self) -> None:
        for name in (
            "python_implementation",
            "python_version",
            "platform",
            "platform_release",
            "machine",
            "process_start_method",
            "timing_clock_implementation",
        ):
            object.__setattr__(self, name, _text(getattr(self, name), name=name))
        object.__setattr__(
            self,
            "logical_cpu_count",
            _integer(
                self.logical_cpu_count,
                name="logical_cpu_count",
                minimum=1,
                maximum=MAX_TASK_COUNT,
            ),
        )
        object.__setattr__(
            self,
            "timing_clock_resolution_seconds",
            _real(
                self.timing_clock_resolution_seconds,
                name="timing_clock_resolution_seconds",
                minimum=0.0,
            ),
        )
        if self.timing_clock_resolution_seconds <= 0.0:
            raise BenchmarkEvidenceError("timing clock resolution must be positive")
        for name in ("timing_clock_monotonic", "timing_clock_adjustable"):
            if not isinstance(getattr(self, name), bool):
                raise BenchmarkEvidenceError(f"{name} must be a bool")

    @property
    def identity(self) -> str:
        """Return the canonical environment identity."""

        return _digest(self.to_dict())

    def to_dict(self) -> dict[str, Any]:
        """Return public environment metadata only."""

        return {
            "python_implementation": self.python_implementation,
            "python_version": self.python_version,
            "platform": self.platform,
            "platform_release": self.platform_release,
            "machine": self.machine,
            "logical_cpu_count": self.logical_cpu_count,
            "process_start_method": self.process_start_method,
            "timing_clock_implementation": self.timing_clock_implementation,
            "timing_clock_resolution_seconds": self.timing_clock_resolution_seconds,
            "timing_clock_monotonic": self.timing_clock_monotonic,
            "timing_clock_adjustable": self.timing_clock_adjustable,
        }

    @classmethod
    def from_dict(cls, value: object) -> BenchmarkEnvironment:
        """Parse a strict environment object."""

        expected = {
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
        data = _mapping(value, expected, name="environment")
        return cls(**{key: data[key] for key in expected})


def collect_benchmark_environment() -> BenchmarkEnvironment:
    """Collect only the public runtime fields declared by the evidence schema."""

    cpu_count = os.cpu_count()
    if cpu_count is None:
        raise BenchmarkEvidenceError("logical CPU count is unavailable")
    clock = time.get_clock_info("perf_counter")
    start_method = multiprocessing.get_context().get_start_method()
    return BenchmarkEnvironment(
        python_implementation=platform_module.python_implementation(),
        python_version=platform_module.python_version(),
        platform=platform_module.system() or "unknown",
        platform_release=platform_module.release() or "unknown",
        machine=platform_module.machine() or "unknown",
        logical_cpu_count=cpu_count,
        process_start_method=start_method,
        timing_clock_implementation=clock.implementation,
        timing_clock_resolution_seconds=clock.resolution,
        timing_clock_monotonic=clock.monotonic,
        timing_clock_adjustable=clock.adjustable,
    )


@dataclass(frozen=True, slots=True)
class TaskGraphBinding:
    """Content identity of every declared task for one workload size."""

    iterations: int
    task_count: int
    task_graph_sha256: str

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "iterations",
            _integer(
                self.iterations,
                name="task_graph.iterations",
                minimum=1,
                maximum=MAX_ITERATIONS_PER_TASK,
            ),
        )
        object.__setattr__(
            self,
            "task_count",
            _integer(
                self.task_count,
                name="task_graph.task_count",
                minimum=1,
                maximum=MAX_TASK_COUNT,
            ),
        )
        object.__setattr__(
            self,
            "task_graph_sha256",
            _sha256(self.task_graph_sha256, name="task_graph.task_graph_sha256"),
        )

    def to_dict(self) -> dict[str, Any]:
        """Return the strict task-graph binding record."""

        return {
            "iterations": self.iterations,
            "task_count": self.task_count,
            "task_graph_sha256": self.task_graph_sha256,
        }

    @classmethod
    def from_dict(cls, value: object) -> TaskGraphBinding:
        """Parse one strict task-graph binding record."""

        expected = {"iterations", "task_count", "task_graph_sha256"}
        data = _mapping(value, expected, name="task_graph")
        return cls(**{key: data[key] for key in expected})


def task_declaration_graph_sha256(tasks: Sequence[TaskSpec]) -> str:
    """Hash the exact order-independent set of full task declarations.

    ``TaskSpec.task_id`` binds payload, seed, and resources.  Hashing the full
    declaration additionally binds timeout, retry, idempotency, and cancellation
    policy so an operationally different graph cannot reuse benchmark evidence.

    Raises:
        BenchmarkEvidenceError: If the graph is empty, oversized, malformed, or
            contains a duplicate task identity.
    """

    rows = tuple(tasks)
    if not rows:
        raise BenchmarkEvidenceError("task graph must contain at least one task")
    if len(rows) > MAX_TASK_COUNT:
        raise BenchmarkEvidenceError(f"task graph exceeds the {MAX_TASK_COUNT}-task ceiling")
    if any(not isinstance(task, TaskSpec) for task in rows):
        raise BenchmarkEvidenceError("task graph entries must be TaskSpec values")
    assert_unique_tasks(rows)
    declarations = sorted((task.to_dict() for task in rows), key=lambda row: row["task_id"])
    return _digest(declarations)


@dataclass(frozen=True, slots=True)
class BenchmarkSample:
    """One serial/process-pool measurement and its semantic parity proof."""

    iterations: int
    repetition: int
    backend_order: str
    task_count: int
    workers: int
    config_sha256: str
    environment_sha256: str
    task_graph_sha256: str
    serial_ns: int
    pool_ns: int
    serial_hash: str
    pool_hash: str
    parity: bool

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "iterations",
            _integer(
                self.iterations,
                name="sample.iterations",
                minimum=1,
                maximum=MAX_ITERATIONS_PER_TASK,
            ),
        )
        object.__setattr__(
            self,
            "repetition",
            _integer(
                self.repetition,
                name="sample.repetition",
                minimum=0,
                maximum=MAX_REPETITIONS - 1,
            ),
        )
        if self.backend_order not in _BACKEND_ORDERS:
            raise BenchmarkEvidenceError("sample.backend_order is unsupported")
        object.__setattr__(
            self,
            "task_count",
            _integer(
                self.task_count,
                name="sample.task_count",
                minimum=1,
                maximum=MAX_TASK_COUNT,
            ),
        )
        object.__setattr__(
            self,
            "workers",
            _integer(self.workers, name="sample.workers", minimum=1, maximum=MAX_WORKERS),
        )
        object.__setattr__(self, "config_sha256", _sha256(self.config_sha256, name="config_sha256"))
        object.__setattr__(
            self,
            "environment_sha256",
            _sha256(self.environment_sha256, name="environment_sha256"),
        )
        object.__setattr__(
            self,
            "task_graph_sha256",
            _sha256(self.task_graph_sha256, name="task_graph_sha256"),
        )
        for name in ("serial_ns", "pool_ns"):
            object.__setattr__(
                self,
                name,
                _integer(
                    getattr(self, name),
                    name=f"sample.{name}",
                    minimum=1,
                    maximum=MAX_SAMPLE_NS,
                ),
            )
        object.__setattr__(self, "serial_hash", _sha256(self.serial_hash, name="serial_hash"))
        object.__setattr__(self, "pool_hash", _sha256(self.pool_hash, name="pool_hash"))
        if not isinstance(self.parity, bool):
            raise BenchmarkEvidenceError("sample.parity must be a bool")
        if not self.parity or self.serial_hash != self.pool_hash:
            raise BenchmarkEvidenceError("sample failed backend parity")

    def to_dict(self) -> dict[str, Any]:
        """Return the strict raw-sample record."""

        return {
            "iterations": self.iterations,
            "repetition": self.repetition,
            "backend_order": self.backend_order,
            "task_count": self.task_count,
            "workers": self.workers,
            "config_sha256": self.config_sha256,
            "environment_sha256": self.environment_sha256,
            "task_graph_sha256": self.task_graph_sha256,
            "serial_ns": self.serial_ns,
            "pool_ns": self.pool_ns,
            "serial_hash": self.serial_hash,
            "pool_hash": self.pool_hash,
            "parity": self.parity,
        }

    @classmethod
    def from_dict(cls, value: object) -> BenchmarkSample:
        """Parse a strict raw-sample record."""

        expected = {
            "iterations",
            "repetition",
            "backend_order",
            "task_count",
            "workers",
            "config_sha256",
            "environment_sha256",
            "task_graph_sha256",
            "serial_ns",
            "pool_ns",
            "serial_hash",
            "pool_hash",
            "parity",
        }
        data = _mapping(value, expected, name="sample")
        return cls(**{key: data[key] for key in expected})


@dataclass(frozen=True, slots=True)
class DistributionSummary:
    """Median, raw MAD, quartiles, and range for one measured quantity."""

    median: float
    mad: float
    minimum: float
    q1: float
    q3: float
    maximum: float

    def __post_init__(self) -> None:
        for name in ("median", "mad", "minimum", "q1", "q3", "maximum"):
            object.__setattr__(self, name, _real(getattr(self, name), name=name))
        if not self.minimum <= self.q1 <= self.median <= self.q3 <= self.maximum:
            raise BenchmarkEvidenceError("summary quantiles are not ordered")

    def to_dict(self) -> dict[str, float]:
        """Return the JSON summary fields."""

        return {
            "median": self.median,
            "mad": self.mad,
            "min": self.minimum,
            "q1": self.q1,
            "q3": self.q3,
            "max": self.maximum,
        }

    @classmethod
    def from_dict(cls, value: object, *, name: str) -> DistributionSummary:
        """Parse one strict distribution summary."""

        data = _mapping(value, {"median", "mad", "min", "q1", "q3", "max"}, name=name)
        return cls(
            median=data["median"],
            mad=data["mad"],
            minimum=data["min"],
            q1=data["q1"],
            q3=data["q3"],
            maximum=data["max"],
        )


def _percentile(values: Sequence[float], quantile: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * quantile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return float(ordered[lower])
    fraction = position - lower
    return float(ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction)


def _distribution(values: Sequence[float]) -> DistributionSummary:
    if not values:
        raise BenchmarkEvidenceError("cannot summarize an empty sample")
    numeric = [_real(value, name="raw summary value") for value in values]
    median = float(statistics.median(numeric))
    return DistributionSummary(
        median=median,
        mad=float(statistics.median(abs(value - median) for value in numeric)),
        minimum=float(min(numeric)),
        q1=_percentile(numeric, 0.25),
        q3=_percentile(numeric, 0.75),
        maximum=float(max(numeric)),
    )


@dataclass(frozen=True, slots=True)
class BenchmarkSummary:
    """Derived statistics for one declared iteration count."""

    iterations: int
    sample_count: int
    serial_ms: DistributionSummary
    pool_ms: DistributionSummary
    per_task_ms: DistributionSummary
    speedup: DistributionSummary

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "iterations",
            _integer(
                self.iterations,
                name="summary.iterations",
                minimum=1,
                maximum=MAX_ITERATIONS_PER_TASK,
            ),
        )
        object.__setattr__(
            self,
            "sample_count",
            _integer(
                self.sample_count,
                name="summary.sample_count",
                minimum=MIN_REPETITIONS,
                maximum=MAX_REPETITIONS,
            ),
        )
        for name in ("serial_ms", "pool_ms", "per_task_ms", "speedup"):
            if not isinstance(getattr(self, name), DistributionSummary):
                raise BenchmarkEvidenceError(f"summary.{name} is malformed")

    def to_dict(self) -> dict[str, Any]:
        """Return a publisher-safe summary independent of dataclass internals."""

        return {
            "iterations": self.iterations,
            "sample_count": self.sample_count,
            "serial_ms": self.serial_ms.to_dict(),
            "pool_ms": self.pool_ms.to_dict(),
            "per_task_ms": self.per_task_ms.to_dict(),
            "speedup": self.speedup.to_dict(),
        }

    @classmethod
    def from_dict(cls, value: object) -> BenchmarkSummary:
        """Parse one strict summary record."""

        data = _mapping(
            value,
            {"iterations", "sample_count", "serial_ms", "pool_ms", "per_task_ms", "speedup"},
            name="summary",
        )
        return cls(
            iterations=data["iterations"],
            sample_count=data["sample_count"],
            serial_ms=DistributionSummary.from_dict(data["serial_ms"], name="summary.serial_ms"),
            pool_ms=DistributionSummary.from_dict(data["pool_ms"], name="summary.pool_ms"),
            per_task_ms=DistributionSummary.from_dict(
                data["per_task_ms"], name="summary.per_task_ms"
            ),
            speedup=DistributionSummary.from_dict(data["speedup"], name="summary.speedup"),
        )


def _summaries_from_samples(
    config: BenchmarkConfig, samples: Sequence[BenchmarkSample]
) -> tuple[BenchmarkSummary, ...]:
    summaries: list[BenchmarkSummary] = []
    for iterations in config.iteration_counts:
        rows = sorted(
            (sample for sample in samples if sample.iterations == iterations),
            key=lambda sample: sample.repetition,
        )
        if len(rows) != config.repetitions:
            raise BenchmarkEvidenceError(
                f"iterations={iterations} has {len(rows)} samples, expected {config.repetitions}"
            )
        serial_ms = [sample.serial_ns / 1_000_000.0 for sample in rows]
        pool_ms = [sample.pool_ns / 1_000_000.0 for sample in rows]
        per_task_ms = [value / config.task_count for value in serial_ms]
        speedup = [sample.serial_ns / sample.pool_ns for sample in rows]
        summaries.append(
            BenchmarkSummary(
                iterations=iterations,
                sample_count=len(rows),
                serial_ms=_distribution(serial_ms),
                pool_ms=_distribution(pool_ms),
                per_task_ms=_distribution(per_task_ms),
                speedup=_distribution(speedup),
            )
        )
    return tuple(summaries)


DEFAULT_LIMITATIONS: Final[tuple[str, ...]] = (
    "Single-machine local Python measurement; not a cluster benchmark or performance SLA.",
    "Synthetic CPU work only; not market, broker, execution-quality, capacity, or trading evidence.",
    "Process-pool timing includes process startup, scheduling, serialization, and result assembly.",
    "Timing distributions describe this recorded environment and may differ on another machine.",
)


def _task_graphs_from_samples(
    config: BenchmarkConfig,
    samples: Sequence[BenchmarkSample],
) -> tuple[TaskGraphBinding, ...]:
    bindings: list[TaskGraphBinding] = []
    for iterations in config.iteration_counts:
        identities = {
            sample.task_graph_sha256 for sample in samples if sample.iterations == iterations
        }
        if len(identities) != 1:
            raise BenchmarkEvidenceError(
                f"iterations={iterations} does not bind one exact task graph"
            )
        bindings.append(
            TaskGraphBinding(
                iterations=iterations,
                task_count=config.task_count,
                task_graph_sha256=identities.pop(),
            )
        )
    return tuple(bindings)


@dataclass(frozen=True)
class BenchmarkEvidence:
    """Complete immutable raw and derived benchmark evidence."""

    config: BenchmarkConfig
    environment: BenchmarkEnvironment
    samples: tuple[BenchmarkSample, ...]
    summary: tuple[BenchmarkSummary, ...]
    limitations: tuple[str, ...] = DEFAULT_LIMITATIONS

    def __post_init__(self) -> None:
        if not isinstance(self.config, BenchmarkConfig):
            raise BenchmarkEvidenceError("config must be BenchmarkConfig")
        if not isinstance(self.environment, BenchmarkEnvironment):
            raise BenchmarkEvidenceError("environment must be BenchmarkEnvironment")
        samples = tuple(sorted(self.samples, key=lambda row: (row.iterations, row.repetition)))
        expected_count = len(self.config.iteration_counts) * self.config.repetitions
        if len(samples) != expected_count:
            raise BenchmarkEvidenceError(
                f"samples contains {len(samples)} rows, expected {expected_count}"
            )
        keys = [(sample.iterations, sample.repetition) for sample in samples]
        if len(set(keys)) != len(keys):
            raise BenchmarkEvidenceError("samples contains a duplicate iterations/repetition row")
        expected_keys = {
            (iterations, repetition)
            for iterations in self.config.iteration_counts
            for repetition in range(self.config.repetitions)
        }
        if set(keys) != expected_keys:
            raise BenchmarkEvidenceError("samples does not cover every declared repetition")
        for sample in samples:
            if sample.task_count != self.config.task_count or sample.workers != self.config.workers:
                raise BenchmarkEvidenceError("sample workload bounds differ from config")
            if sample.config_sha256 != self.config.identity:
                raise BenchmarkEvidenceError("sample config identity is inconsistent")
            if sample.environment_sha256 != self.environment.identity:
                raise BenchmarkEvidenceError("sample environment identity is inconsistent")
        _task_graphs_from_samples(self.config, samples)
        summaries = tuple(sorted(self.summary, key=lambda row: row.iterations))
        expected_summary = _summaries_from_samples(self.config, samples)
        if summaries != expected_summary:
            raise BenchmarkEvidenceError("summary is not derived exactly from raw samples")
        limitations = tuple(
            _text(value, name=f"limitations[{index}]")
            for index, value in enumerate(self.limitations)
        )
        if not limitations or len(limitations) > 16 or len(set(limitations)) != len(limitations):
            raise BenchmarkEvidenceError("limitations must contain 1-16 unique statements")
        object.__setattr__(self, "samples", samples)
        object.__setattr__(self, "summary", summaries)
        object.__setattr__(self, "limitations", limitations)

    @classmethod
    def from_samples(
        cls,
        *,
        config: BenchmarkConfig,
        environment: BenchmarkEnvironment,
        samples: Sequence[BenchmarkSample],
        limitations: Sequence[str] = DEFAULT_LIMITATIONS,
    ) -> BenchmarkEvidence:
        """Build evidence while deriving every summary from raw rows."""

        rows = tuple(samples)
        return cls(
            config=config,
            environment=environment,
            samples=rows,
            summary=_summaries_from_samples(config, rows),
            limitations=tuple(limitations),
        )

    @property
    def raw_samples_sha256(self) -> str:
        """Return the digest of the canonical raw-sample array."""

        return _digest([sample.to_dict() for sample in self.samples])

    @property
    def task_graphs(self) -> tuple[TaskGraphBinding, ...]:
        """Return the exact per-work-size task graph identities."""

        return _task_graphs_from_samples(self.config, self.samples)

    def _identity_payload(self) -> dict[str, Any]:
        return {
            "schema_version": SCHEMA_VERSION,
            "benchmark": BENCHMARK_NAME,
            "raw_samples_sha256": self.raw_samples_sha256,
            "config": self.config.to_dict(),
            "environment": self.environment.to_dict(),
            "task_graphs": [binding.to_dict() for binding in self.task_graphs],
            "samples": [sample.to_dict() for sample in self.samples],
            "summary": [record.to_dict() for record in self.summary],
            "limitations": list(self.limitations),
        }

    @property
    def benchmark_id(self) -> str:
        """Return the full canonical SHA-256 evidence identity."""

        return _digest(self._identity_payload())

    def to_dict(self) -> dict[str, Any]:
        """Return the complete strict schema, including its self-excluding digest."""

        return {**self._identity_payload(), "benchmark_id": self.benchmark_id}

    def canonical_bytes(self) -> bytes:
        """Return byte-stable canonical JSON with one trailing newline."""

        return _canonical_bytes(self.to_dict()) + b"\n"

    @classmethod
    def from_dict(cls, value: object) -> BenchmarkEvidence:
        """Validate and reconstruct a strict current-schema evidence document."""

        expected = {
            "schema_version",
            "benchmark",
            "benchmark_id",
            "raw_samples_sha256",
            "config",
            "environment",
            "task_graphs",
            "samples",
            "summary",
            "limitations",
        }
        data = _mapping(value, expected, name="benchmark evidence")
        if data["schema_version"] != SCHEMA_VERSION:
            raise BenchmarkEvidenceError(
                f"unsupported schema_version {data['schema_version']!r}; expected {SCHEMA_VERSION!r}"
            )
        if data["benchmark"] != BENCHMARK_NAME:
            raise BenchmarkEvidenceError("benchmark name is unsupported")
        samples_data = data["samples"]
        task_graphs_data = data["task_graphs"]
        summary_data = data["summary"]
        limitations_data = data["limitations"]
        if (
            not isinstance(samples_data, list)
            or not isinstance(task_graphs_data, list)
            or not isinstance(summary_data, list)
        ):
            raise BenchmarkEvidenceError("samples, task_graphs, and summary must be arrays")
        if not isinstance(limitations_data, list):
            raise BenchmarkEvidenceError("limitations must be an array")
        evidence = cls(
            config=BenchmarkConfig.from_dict(data["config"]),
            environment=BenchmarkEnvironment.from_dict(data["environment"]),
            samples=tuple(BenchmarkSample.from_dict(row) for row in samples_data),
            summary=tuple(BenchmarkSummary.from_dict(row) for row in summary_data),
            limitations=tuple(limitations_data),
        )
        if _sha256(data["raw_samples_sha256"], name="raw_samples_sha256") != (
            evidence.raw_samples_sha256
        ):
            raise BenchmarkEvidenceError("raw sample digest mismatch")
        declared_graphs = tuple(TaskGraphBinding.from_dict(row) for row in task_graphs_data)
        if declared_graphs != evidence.task_graphs:
            raise BenchmarkEvidenceError("task graph index is inconsistent with raw samples")
        if _sha256(data["benchmark_id"], name="benchmark_id") != evidence.benchmark_id:
            raise BenchmarkEvidenceError("benchmark identity mismatch")
        return evidence


def summarize_benchmark(evidence: BenchmarkEvidence) -> tuple[BenchmarkSummary, ...]:
    """Recompute and return summaries from the evidence's raw samples."""

    if not isinstance(evidence, BenchmarkEvidence):
        raise BenchmarkEvidenceError("evidence must be BenchmarkEvidence")
    return _summaries_from_samples(evidence.config, evidence.samples)


def _default_serial_executor(
    function: BenchmarkFunction,
    tasks: Sequence[TaskSpec],
    total_timeout_seconds: float,
) -> BatchReport:
    return _PRODUCTION_LOCAL_EXECUTOR(
        function,
        tasks,
        total_timeout_seconds=total_timeout_seconds,
    )


def _default_pool_executor(
    function: BenchmarkFunction,
    tasks: Sequence[TaskSpec],
    workers: int,
    total_timeout_seconds: float,
) -> BatchReport:
    return _PRODUCTION_POOL_EXECUTOR(
        function,
        tasks,
        workers=workers,
        total_timeout_seconds=total_timeout_seconds,
    )


_PRODUCTION_SERIAL_ADAPTER: Final = _default_serial_executor
_PRODUCTION_POOL_ADAPTER: Final = _default_pool_executor
_PRODUCTION_ENVIRONMENT_COLLECTOR: Final = collect_benchmark_environment


def _validated_report(
    report: BatchReport,
    *,
    backend: str,
    workers: int,
    tasks: Sequence[TaskSpec],
) -> BatchReport:
    if not isinstance(report, BatchReport):
        raise BenchmarkEvidenceError(f"{backend} executor returned a non-BatchReport")
    if report.backend != backend or report.workers != workers:
        raise BenchmarkEvidenceError(f"{backend} executor reported inconsistent metadata")
    if len(report.results) != len(tasks) or not report.all_succeeded:
        raise BenchmarkEvidenceError(f"{backend} executor did not complete every task")
    expected = {(task.task_id, task.name) for task in tasks}
    observed = {(result.task_id, result.name) for result in report.results}
    if len(observed) != len(report.results) or observed != expected:
        raise BenchmarkEvidenceError(
            f"{backend} executor results do not match the bound task graph"
        )
    return report


def _elapsed_ns(clock_ns: Clock, operation: Callable[[], BatchReport]) -> tuple[int, BatchReport]:
    start = clock_ns()
    if isinstance(start, bool) or not isinstance(start, int):
        raise BenchmarkEvidenceError("perf_counter_ns clock must return integers")
    report = operation()
    stop = clock_ns()
    if isinstance(stop, bool) or not isinstance(stop, int):
        raise BenchmarkEvidenceError("perf_counter_ns clock must return integers")
    elapsed = _integer(stop - start, name="sample elapsed_ns", minimum=1, maximum=MAX_SAMPLE_NS)
    return elapsed, report


@dataclass(frozen=True, slots=True)
class _RunBudget:
    """One monotonic wall-time budget shared by all warmup and measured batches."""

    clock: BudgetClock
    started: float
    maximum_seconds: float

    @classmethod
    def start(cls, clock: BudgetClock, maximum_seconds: float) -> _RunBudget:
        if not callable(clock):
            raise BenchmarkEvidenceError("budget_clock must be callable")
        started = _real(clock(), name="budget_clock reading")
        return cls(clock=clock, started=started, maximum_seconds=maximum_seconds)

    def remaining_seconds(self) -> float:
        """Return positive remaining time or fail closed at the run deadline."""

        current = _real(self.clock(), name="budget_clock reading")
        if current < self.started:
            raise BenchmarkEvidenceError("budget_clock must be monotonic")
        remaining = self.maximum_seconds - (current - self.started)
        if remaining <= 0.0:
            raise BenchmarkEvidenceError(
                f"benchmark exceeded its {self.maximum_seconds:g}s total wall-time budget"
            )
        return min(self.maximum_seconds, remaining)


def _execute_with_budget(
    budget: _RunBudget,
    operation: Callable[[float], BatchReport],
) -> BatchReport:
    remaining = budget.remaining_seconds()
    report = operation(remaining)
    budget.remaining_seconds()
    return report


def _measure_with_budget(
    budget: _RunBudget,
    clock_ns: Clock,
    operation: Callable[[float], BatchReport],
) -> tuple[int, BatchReport]:
    remaining = budget.remaining_seconds()
    elapsed, report = _elapsed_ns(clock_ns, lambda: operation(remaining))
    budget.remaining_seconds()
    return elapsed, report


def _collect_tasks_with_budget(
    budget: _RunBudget,
    task_builder: TaskBuilder,
    *,
    task_count: int,
    iterations: int,
) -> tuple[TaskSpec, ...]:
    """Collect exactly ``task_count`` tasks without exhausting an unbounded iterable.

    The monotonic budget is checked before and after the builder call and every
    iterator advance.  These checks are cooperative: arbitrary Python executed
    inside the builder cannot be preempted while it is running.
    """

    budget.remaining_seconds()
    try:
        produced = task_builder(task_count, iterations)
    except Exception as exc:
        raise BenchmarkEvidenceError("task_builder failed while declaring the task graph") from exc
    budget.remaining_seconds()
    try:
        iterator = iter(produced)
    except TypeError as exc:
        raise BenchmarkEvidenceError(
            "task_builder must return an iterable of TaskSpec values"
        ) from exc

    tasks: list[TaskSpec] = []
    while len(tasks) <= task_count:
        budget.remaining_seconds()
        try:
            task = next(iterator)
        except StopIteration:
            break
        except Exception as exc:
            raise BenchmarkEvidenceError("task_builder iterable failed during collection") from exc
        budget.remaining_seconds()
        if len(tasks) == task_count:
            raise BenchmarkEvidenceError(
                f"task_builder returned more than the declared {task_count} tasks"
            )
        if not isinstance(task, TaskSpec):
            raise BenchmarkEvidenceError("task_builder entries must be TaskSpec values")
        tasks.append(task)
    if len(tasks) != task_count:
        raise BenchmarkEvidenceError(
            f"task_builder returned {len(tasks)} tasks, expected {task_count}"
        )
    return tuple(tasks)


def _validate_callable_binding(
    candidate: object,
    *,
    expected_entrypoint: str,
    expected_source_sha256: str,
    role: str,
) -> str:
    """Refuse a callable whose identity or source differs from its binding."""

    if not inspect.isfunction(candidate):
        raise BenchmarkEvidenceError(
            f"{role} must be a module-level Python function, not a wrapper or builtin"
        )
    module = candidate.__module__
    qualified_name = candidate.__qualname__
    if (
        not isinstance(module, str)
        or not module
        or not isinstance(qualified_name, str)
        or not qualified_name
        or qualified_name == "<lambda>"
        or "<locals>" in qualified_name
    ):
        raise BenchmarkEvidenceError(f"{role} must be a named module-level Python function")
    actual_entrypoint = f"{module}.{qualified_name}"
    source_name = inspect.getsourcefile(candidate)
    if source_name is None:
        raise BenchmarkEvidenceError(f"{role} source file is unavailable")
    source = Path(os.path.abspath(source_name))

    entrypoint_matches = actual_entrypoint == expected_entrypoint
    if not entrypoint_matches and module in {"__main__", "__mp_main__"}:
        suffix = f".{qualified_name}"
        if expected_entrypoint.endswith(suffix):
            expected_module = expected_entrypoint[: -len(suffix)]
            expected_relative = Path(*expected_module.split(".")).with_suffix(".py")
            entrypoint_matches = source.as_posix().endswith(f"/{expected_relative.as_posix()}")
    if not entrypoint_matches:
        raise BenchmarkEvidenceError(f"{role} does not match its declared entrypoint")

    try:
        source_identity = read_regular_file_snapshot(
            source,
            max_bytes=MAX_IDENTITY_SOURCE_BYTES,
            root=source.parent,
        ).sha256
    except BoundedIOError as exc:
        raise BenchmarkEvidenceError(
            f"{role} source must be a bounded non-symlink regular file"
        ) from exc
    if source_identity != expected_source_sha256:
        raise BenchmarkEvidenceError(f"{role} source does not match its declared SHA-256")
    return source_identity


def _validate_source_binding(source: Path, *, expected_sha256: str, role: str) -> None:
    """Verify one internal benchmark source without exposing its local path."""

    lexical = Path(os.path.abspath(source))
    try:
        observed = read_regular_file_snapshot(
            lexical,
            max_bytes=MAX_IDENTITY_SOURCE_BYTES,
            root=lexical.parent,
        ).sha256
    except BoundedIOError as exc:
        raise BenchmarkEvidenceError(
            f"{role} source must be a bounded non-symlink regular file"
        ) from exc
    if observed != expected_sha256:
        raise BenchmarkEvidenceError(f"{role} source does not match its declared SHA-256")


def _validate_runtime_source_bindings(implementation: BenchmarkImplementation) -> None:
    """Bind internal execution semantics and dependency resolution to exact bytes."""

    evidence_source = Path(__file__)
    executor_source_name = inspect.getsourcefile(execute_local)
    task_source_name = inspect.getsourcefile(TaskSpec)
    if executor_source_name is None or task_source_name is None:
        raise BenchmarkEvidenceError("benchmark runtime source files are unavailable")
    repository = Path(os.path.abspath(evidence_source)).parents[2]
    _validate_source_binding(
        evidence_source,
        expected_sha256=implementation.evidence_contract_source_sha256,
        role="evidence contract",
    )
    _validate_source_binding(
        Path(executor_source_name),
        expected_sha256=implementation.executor_source_sha256,
        role="executor",
    )
    _validate_source_binding(
        Path(task_source_name),
        expected_sha256=implementation.task_contract_source_sha256,
        role="task contract",
    )
    _validate_source_binding(
        repository / "uv.lock",
        expected_sha256=implementation.dependency_lock_sha256,
        role="dependency lock",
    )


def _validate_production_runtime_bindings(implementation: BenchmarkImplementation) -> None:
    """Prove that the public runner still names and invokes its fixed runtime."""

    require_production_implementation(implementation)
    if execute_local is not _PRODUCTION_LOCAL_EXECUTOR:
        raise BenchmarkEvidenceError("production local executor binding was replaced")
    if execute_process_pool is not _PRODUCTION_POOL_EXECUTOR:
        raise BenchmarkEvidenceError("production process-pool executor binding was replaced")
    if _default_serial_executor is not _PRODUCTION_SERIAL_ADAPTER:
        raise BenchmarkEvidenceError("production serial adapter binding was replaced")
    if _default_pool_executor is not _PRODUCTION_POOL_ADAPTER:
        raise BenchmarkEvidenceError("production process-pool adapter binding was replaced")
    if collect_benchmark_environment is not _PRODUCTION_ENVIRONMENT_COLLECTOR:
        raise BenchmarkEvidenceError("production environment collector binding was replaced")
    if time.perf_counter_ns is not _PRODUCTION_TIMING_CLOCK:
        raise BenchmarkEvidenceError("production timing clock binding was replaced")
    if time.monotonic is not _PRODUCTION_BUDGET_CLOCK:
        raise BenchmarkEvidenceError("production budget clock binding was replaced")
    _validate_callable_binding(
        _PRODUCTION_LOCAL_EXECUTOR,
        expected_entrypoint=implementation.serial_executor_entrypoint,
        expected_source_sha256=implementation.executor_source_sha256,
        role="production local executor",
    )
    _validate_callable_binding(
        _PRODUCTION_POOL_EXECUTOR,
        expected_entrypoint=implementation.pool_executor_entrypoint,
        expected_source_sha256=implementation.executor_source_sha256,
        role="production process-pool executor",
    )


def _run_crossover_benchmark_with_runtime(
    config: BenchmarkConfig,
    *,
    function: BenchmarkFunction,
    task_builder: TaskBuilder,
    harness: Callable[..., object],
    serial_executor: SerialExecutor,
    pool_executor: PoolExecutor,
    clock_ns: Clock,
    budget_clock: BudgetClock,
    environment: BenchmarkEnvironment,
    required_profile: str,
) -> BenchmarkEvidence:
    """Execute one profile-locked benchmark runtime.

    Backend order alternates by repetition so cache/thermal order cannot always
    favor the same backend. Timing is descriptive; no numeric SLA is asserted.
    The shared budget is checked between builder advances and executor returns
    and is passed into each backend. It does not preempt arbitrary in-process
    Python; task and batch timeout behavior is cooperative or post-return unless
    a backend independently terminates its worker. Injected clocks must supply
    matching explicit environment metadata so test evidence cannot be mistaken
    for ``perf_counter_ns`` output.
    """

    if not isinstance(config, BenchmarkConfig):
        raise BenchmarkEvidenceError("config must be BenchmarkConfig")
    if config.implementation.execution_profile != required_profile:
        raise BenchmarkEvidenceError(
            f"benchmark runtime requires execution_profile={required_profile!r}"
        )
    if not callable(function) or not callable(task_builder) or not callable(harness):
        raise BenchmarkEvidenceError("function, task_builder, and harness must be callable")
    if not callable(serial_executor) or not callable(pool_executor):
        raise BenchmarkEvidenceError("serial_executor and pool_executor must be callable")
    if not callable(clock_ns) or not callable(budget_clock):
        raise BenchmarkEvidenceError("benchmark clocks must be callable")
    if not isinstance(environment, BenchmarkEnvironment):
        raise BenchmarkEvidenceError("environment must be BenchmarkEnvironment")
    _validate_callable_binding(
        function,
        expected_entrypoint=config.implementation.workload_entrypoint,
        expected_source_sha256=config.implementation.workload_source_sha256,
        role="benchmark function",
    )
    _validate_callable_binding(
        task_builder,
        expected_entrypoint=config.implementation.task_builder_entrypoint,
        expected_source_sha256=config.implementation.task_builder_source_sha256,
        role="task builder",
    )
    _validate_callable_binding(
        harness,
        expected_entrypoint=config.implementation.harness_entrypoint,
        expected_source_sha256=config.implementation.harness_source_sha256,
        role="benchmark harness",
    )
    _validate_runtime_source_bindings(config.implementation)
    budget = _RunBudget.start(budget_clock, config.max_total_seconds)
    samples: list[BenchmarkSample] = []
    for iterations in config.iteration_counts:
        tasks = _collect_tasks_with_budget(
            budget,
            task_builder,
            task_count=config.task_count,
            iterations=iterations,
        )
        task_graph_sha256 = task_declaration_graph_sha256(tasks)
        serial_operation = partial(serial_executor, function, tasks)
        pool_operation = partial(pool_executor, function, tasks, config.workers)

        for warmup in range(config.warmups):
            if warmup % 2 == 0:
                serial = _execute_with_budget(budget, serial_operation)
                pool = _execute_with_budget(budget, pool_operation)
            else:
                pool = _execute_with_budget(budget, pool_operation)
                serial = _execute_with_budget(budget, serial_operation)
            serial = _validated_report(serial, backend="local", workers=1, tasks=tasks)
            pool = _validated_report(
                pool,
                backend="process_pool",
                workers=config.workers,
                tasks=tasks,
            )
            assert_backend_parity(serial, pool)

        for repetition in range(config.repetitions):
            backend_order = "serial_then_pool" if repetition % 2 == 0 else "pool_then_serial"
            if backend_order == "serial_then_pool":
                serial_ns, serial = _measure_with_budget(budget, clock_ns, serial_operation)
                pool_ns, pool = _measure_with_budget(budget, clock_ns, pool_operation)
            else:
                pool_ns, pool = _measure_with_budget(budget, clock_ns, pool_operation)
                serial_ns, serial = _measure_with_budget(budget, clock_ns, serial_operation)
            serial = _validated_report(serial, backend="local", workers=1, tasks=tasks)
            pool = _validated_report(
                pool,
                backend="process_pool",
                workers=config.workers,
                tasks=tasks,
            )
            assert_backend_parity(serial, pool)
            serial_hash = serial.assembly_hash()
            pool_hash = pool.assembly_hash()
            samples.append(
                BenchmarkSample(
                    iterations=iterations,
                    repetition=repetition,
                    backend_order=backend_order,
                    task_count=config.task_count,
                    workers=config.workers,
                    config_sha256=config.identity,
                    environment_sha256=environment.identity,
                    task_graph_sha256=task_graph_sha256,
                    serial_ns=serial_ns,
                    pool_ns=pool_ns,
                    serial_hash=serial_hash,
                    pool_hash=pool_hash,
                    parity=serial_hash == pool_hash,
                )
            )
    return BenchmarkEvidence.from_samples(
        config=config,
        environment=environment,
        samples=samples,
    )


_PRODUCTION_RUNTIME_RUNNER: Final = _run_crossover_benchmark_with_runtime
_PRODUCTION_RUNTIME_VALIDATOR: Final = _validate_production_runtime_bindings


def run_crossover_benchmark(
    config: BenchmarkConfig,
    *,
    function: BenchmarkFunction,
    task_builder: TaskBuilder,
    harness: Callable[..., object],
) -> BenchmarkEvidence:
    """Run the source-bound production crossover benchmark.

    Runtime dependencies are deliberately not injectable at this public
    boundary.  The exact local/process-pool executors and the two clocks are
    identity-checked immediately before use so test doubles cannot emit evidence
    claiming the production execution profile.
    """

    if not isinstance(config, BenchmarkConfig):
        raise BenchmarkEvidenceError("config must be BenchmarkConfig")
    if _run_crossover_benchmark_with_runtime is not _PRODUCTION_RUNTIME_RUNNER:
        raise BenchmarkEvidenceError("production benchmark runtime binding was replaced")
    if _validate_production_runtime_bindings is not _PRODUCTION_RUNTIME_VALIDATOR:
        raise BenchmarkEvidenceError("production runtime validator binding was replaced")
    _PRODUCTION_RUNTIME_VALIDATOR(config.implementation)
    return _PRODUCTION_RUNTIME_RUNNER(
        config,
        function=function,
        task_builder=task_builder,
        harness=harness,
        serial_executor=_PRODUCTION_SERIAL_ADAPTER,
        pool_executor=_PRODUCTION_POOL_ADAPTER,
        clock_ns=_PRODUCTION_TIMING_CLOCK,
        budget_clock=_PRODUCTION_BUDGET_CLOCK,
        environment=_PRODUCTION_ENVIRONMENT_COLLECTOR(),
        required_profile=PRODUCTION_EXECUTION_PROFILE,
    )


def _run_crossover_benchmark_for_testing(
    config: BenchmarkConfig,
    *,
    function: BenchmarkFunction,
    task_builder: TaskBuilder,
    harness: Callable[..., object],
    environment: BenchmarkEnvironment,
    serial_executor: SerialExecutor = _default_serial_executor,
    pool_executor: PoolExecutor = _default_pool_executor,
    clock_ns: Clock = _PRODUCTION_TIMING_CLOCK,
    budget_clock: BudgetClock = _PRODUCTION_BUDGET_CLOCK,
) -> BenchmarkEvidence:
    """Exercise benchmark mechanics with explicit non-publishable test evidence."""

    return _run_crossover_benchmark_with_runtime(
        config,
        function=function,
        task_builder=task_builder,
        harness=harness,
        serial_executor=serial_executor,
        pool_executor=pool_executor,
        clock_ns=clock_ns,
        budget_clock=budget_clock,
        environment=environment,
        required_profile=TEST_EXECUTION_PROFILE,
    )


def parse_benchmark_evidence_bytes(payload: bytes) -> BenchmarkEvidence:
    """Validate evidence from one immutable, already-read bounded snapshot.

    Callers that own a stronger file-boundary policy can read once and pass the
    exact snapshot bytes here, avoiding a pathname reopen between provenance
    hashing and semantic validation.
    """

    if not isinstance(payload, bytes):
        raise BenchmarkEvidenceError("benchmark evidence snapshot must be immutable bytes")
    if not 0 < len(payload) <= MAX_EVIDENCE_BYTES:
        raise BenchmarkEvidenceError(
            f"benchmark evidence snapshot bytes must lie in [1, {MAX_EVIDENCE_BYTES}]"
        )
    try:
        document = parse_strict_json(
            payload,
            maximum_depth=MAX_JSON_DEPTH,
            maximum_nodes=MAX_JSON_NODES,
        )
    except BoundedIOError as exc:
        raise BenchmarkEvidenceError(
            f"benchmark evidence snapshot is not strict JSON: {exc}"
        ) from exc
    return BenchmarkEvidence.from_dict(document)


def load_benchmark_evidence(path: str | Path) -> BenchmarkEvidence:
    """Load one immutable bounded snapshot and validate its schema and digests."""

    source = Path(os.path.abspath(path))
    try:
        snapshot = read_regular_file_snapshot(
            source,
            max_bytes=MAX_EVIDENCE_BYTES,
            root=source.parent,
        )
    except BoundedIOError as exc:
        raise BenchmarkEvidenceError(
            f"benchmark evidence must be valid bounded regular file JSON: {exc}"
        ) from exc
    return parse_benchmark_evidence_bytes(snapshot.data)


def _directory_open_flags() -> int:
    return (
        os.O_RDONLY
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_DIRECTORY", 0)
        | getattr(os, "O_NOFOLLOW", 0)
    )


def _same_inode(left: os.stat_result, right: os.stat_result) -> bool:
    return left.st_dev == right.st_dev and left.st_ino == right.st_ino


def _safe_publication_parent(path: Path) -> tuple[Path, int]:
    """Anchor and create each directory component without following symlinks.

    A recursive ``mkdir`` before ancestry validation can create content through
    a pre-existing symlink.  This walker starts at the filesystem root, opens
    every component relative to the already trusted directory descriptor, and
    returns the final descriptor for all subsequent publication operations.
    """

    parent = Path(os.path.abspath(path))
    descriptor = -1
    try:
        descriptor = os.open(parent.anchor, _directory_open_flags())
        for component in parent.parts[1:]:
            try:
                metadata = os.stat(component, dir_fd=descriptor, follow_symlinks=False)
            except FileNotFoundError:
                try:
                    os.mkdir(component, mode=0o700, dir_fd=descriptor)
                    _fsync_directory(descriptor)
                except FileExistsError:
                    # A concurrent creator won the race.  The lstat/open/inode
                    # checks below still decide whether its object is trusted.
                    pass
                metadata = os.stat(component, dir_fd=descriptor, follow_symlinks=False)
            if stat.S_ISLNK(metadata.st_mode):
                raise BenchmarkEvidenceError("benchmark evidence directory traverses a symlink")
            if not stat.S_ISDIR(metadata.st_mode):
                raise BenchmarkEvidenceError(
                    "benchmark evidence parent component must be a directory"
                )
            child = os.open(component, _directory_open_flags(), dir_fd=descriptor)
            opened = os.fstat(child)
            current = os.stat(component, dir_fd=descriptor, follow_symlinks=False)
            if (
                not stat.S_ISDIR(opened.st_mode)
                or not stat.S_ISDIR(current.st_mode)
                or not _same_inode(opened, current)
            ):
                os.close(child)
                raise BenchmarkEvidenceError(
                    "benchmark evidence directory changed while being anchored"
                )
            os.close(descriptor)
            descriptor = child
    except BenchmarkEvidenceError:
        if descriptor >= 0:
            os.close(descriptor)
        raise
    except OSError as exc:
        if descriptor >= 0:
            os.close(descriptor)
        if exc.errno == errno.ELOOP:
            raise BenchmarkEvidenceError(
                "benchmark evidence directory traverses a symlink"
            ) from exc
        raise BenchmarkEvidenceError(
            "unable to create and anchor benchmark evidence directory"
        ) from exc
    return parent, descriptor


def _create_temporary_file(directory_descriptor: int) -> tuple[int, str]:
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_CLOEXEC", 0)
    for _ in range(32):
        name = f".distributed-benchmark-{secrets.token_hex(16)}.tmp"
        try:
            return os.open(name, flags, 0o600, dir_fd=directory_descriptor), name
        except FileExistsError:
            continue
        except OSError as exc:
            raise BenchmarkEvidenceError(
                "unable to create benchmark evidence temporary file"
            ) from exc
    raise BenchmarkEvidenceError("unable to allocate a unique benchmark evidence temporary file")


def _fsync_directory(descriptor: int) -> None:
    try:
        os.fsync(descriptor)
    except OSError as exc:
        if exc.errno not in {errno.EINVAL, getattr(errno, "ENOTSUP", errno.EINVAL)}:
            raise


def _named_regular_file(
    directory_descriptor: int,
    name: str,
    *,
    expected: os.stat_result,
    role: str,
) -> os.stat_result:
    """Return metadata only when ``name`` still identifies the expected file."""

    try:
        observed = os.stat(name, dir_fd=directory_descriptor, follow_symlinks=False)
    except OSError as exc:
        raise BenchmarkEvidenceError(f"{role} is unavailable") from exc
    if not stat.S_ISREG(observed.st_mode) or not _same_inode(observed, expected):
        raise BenchmarkEvidenceError(f"{role} changed during publication")
    return observed


def _unlink_owned_name(
    directory_descriptor: int,
    name: str,
    *,
    expected: os.stat_result,
) -> None:
    """Best-effort cleanup that refuses a name rebound to another inode."""

    try:
        observed = os.stat(name, dir_fd=directory_descriptor, follow_symlinks=False)
    except FileNotFoundError:
        return
    if not stat.S_ISREG(observed.st_mode) or not _same_inode(observed, expected):
        return
    os.unlink(name, dir_fd=directory_descriptor)
    _fsync_directory(directory_descriptor)


def _load_published_evidence(
    directory_descriptor: int,
    name: str,
    *,
    expected: os.stat_result,
) -> tuple[BenchmarkEvidence, os.stat_result]:
    """Validate one publication and return the inode metadata actually read."""

    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    descriptor = -1
    payload = b""
    try:
        descriptor = os.open(name, flags, dir_fd=directory_descriptor)
        initial = os.fstat(descriptor)
        if (
            not stat.S_ISREG(initial.st_mode)
            or not _same_inode(initial, expected)
            or not 0 < initial.st_size <= MAX_EVIDENCE_BYTES
        ):
            raise BenchmarkEvidenceError("published benchmark evidence is not a bounded file")
        chunks: list[bytes] = []
        remaining = MAX_EVIDENCE_BYTES + 1
        while remaining > 0:
            chunk = os.read(descriptor, min(1_048_576, remaining))
            if not chunk:
                break
            chunks.append(chunk)
            remaining -= len(chunk)
        payload = b"".join(chunks)
        final = os.fstat(descriptor)
        if (
            len(payload) != initial.st_size
            or initial.st_size != final.st_size
            or not _same_inode(initial, final)
            or initial.st_mtime_ns != final.st_mtime_ns
            or initial.st_ctime_ns != final.st_ctime_ns
        ):
            raise BenchmarkEvidenceError("published benchmark evidence changed during verification")
    except BenchmarkEvidenceError:
        raise
    except OSError as exc:
        raise BenchmarkEvidenceError("unable to verify published benchmark evidence") from exc
    finally:
        if descriptor >= 0:
            os.close(descriptor)
    return parse_benchmark_evidence_bytes(payload), final


def write_benchmark_evidence(evidence: BenchmarkEvidence, path: str | Path) -> Path:
    """Atomically create, sync, and verify evidence without following symlinks."""

    if not isinstance(evidence, BenchmarkEvidence):
        raise BenchmarkEvidenceError("evidence must be BenchmarkEvidence")
    requested = Path(path)
    if not requested.name:
        raise BenchmarkEvidenceError("benchmark evidence destination must name a file")
    payload = evidence.canonical_bytes()
    if len(payload) > MAX_EVIDENCE_BYTES:
        raise BenchmarkEvidenceError("serialized benchmark evidence exceeds its byte ceiling")
    parent, directory_descriptor = _safe_publication_parent(requested.parent)
    destination = parent / requested.name

    temporary_descriptor = -1
    temporary_name: str | None = None
    temporary_metadata: os.stat_result | None = None
    published_metadata: os.stat_result | None = None
    try:
        try:
            os.stat(destination.name, dir_fd=directory_descriptor, follow_symlinks=False)
        except FileNotFoundError:
            pass
        else:
            raise FileExistsError(f"benchmark evidence destination already exists: {destination}")
        temporary_descriptor, temporary_name = _create_temporary_file(directory_descriptor)
        temporary_metadata = os.fstat(temporary_descriptor)
        if not stat.S_ISREG(temporary_metadata.st_mode):
            raise BenchmarkEvidenceError("benchmark evidence temporary object is not a file")
        with os.fdopen(temporary_descriptor, "wb") as handle:
            temporary_descriptor = -1
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
            published_metadata = os.fstat(handle.fileno())
            if not _same_inode(published_metadata, temporary_metadata):
                raise BenchmarkEvidenceError(
                    "benchmark evidence temporary file changed during write"
                )
        try:
            os.link(
                temporary_name,
                destination.name,
                src_dir_fd=directory_descriptor,
                dst_dir_fd=directory_descriptor,
                follow_symlinks=False,
            )
        except FileExistsError as exc:
            raise FileExistsError(
                f"benchmark evidence destination already exists: {destination}"
            ) from exc
        if published_metadata is None:
            raise BenchmarkEvidenceError("benchmark evidence file metadata is unavailable")
        _named_regular_file(
            directory_descriptor,
            destination.name,
            expected=published_metadata,
            role="benchmark evidence destination",
        )
        _fsync_directory(directory_descriptor)

        # Verify through the descriptor used to publish, so a concurrent
        # pathname substitution cannot redirect the verification read.
        loaded, loaded_metadata = _load_published_evidence(
            directory_descriptor,
            destination.name,
            expected=published_metadata,
        )
        if not _same_inode(loaded_metadata, published_metadata):
            raise BenchmarkEvidenceError("published benchmark evidence inode is inconsistent")
        pathname = os.stat(parent, follow_symlinks=False)
        opened = os.fstat(directory_descriptor)
        if not stat.S_ISDIR(pathname.st_mode) or not _same_inode(pathname, opened):
            raise BenchmarkEvidenceError("benchmark evidence directory changed during verification")
        if loaded.benchmark_id != evidence.benchmark_id:
            raise BenchmarkEvidenceError(
                "published benchmark evidence failed identity verification"
            )
        if temporary_name is None or temporary_metadata is None:
            raise BenchmarkEvidenceError("benchmark evidence temporary identity is unavailable")
        _unlink_owned_name(
            directory_descriptor,
            temporary_name,
            expected=temporary_metadata,
        )
        temporary_name = None
        temporary_metadata = None
        _named_regular_file(
            directory_descriptor,
            destination.name,
            expected=published_metadata,
            role="benchmark evidence destination",
        )
    except Exception:
        # Once linked, the destination is intentionally preserved on failure.
        # POSIX has no portable inode-conditional unlink; retaining the name is
        # safer than deleting a concurrent replacement between stat and unlink.
        raise
    finally:
        if temporary_descriptor >= 0:
            os.close(temporary_descriptor)
        if temporary_name is not None and temporary_metadata is not None:
            with suppress(FileNotFoundError):
                _unlink_owned_name(
                    directory_descriptor,
                    temporary_name,
                    expected=temporary_metadata,
                )
        os.close(directory_descriptor)
    return requested


__all__ = [
    "BENCHMARK_NAME",
    "DEFAULT_LIMITATIONS",
    "DEFAULT_MAX_TOTAL_SECONDS",
    "MAX_BENCHMARK_SECONDS",
    "MAX_DECLARED_ITERATIONS",
    "MAX_EVIDENCE_BYTES",
    "MAX_ITERATION_COUNTS",
    "MAX_ITERATIONS_PER_TASK",
    "MAX_REPETITIONS",
    "MAX_TASK_COUNT",
    "MAX_TOTAL_TASK_RUNS",
    "MAX_WARMUPS",
    "MAX_WORKERS",
    "MIN_REPETITIONS",
    "MIN_WARMUPS",
    "PRODUCTION_BUDGET_CLOCK_ENTRYPOINT",
    "PRODUCTION_EXECUTION_PROFILE",
    "PRODUCTION_HARNESS_ENTRYPOINT",
    "PRODUCTION_HARNESS_NAME",
    "PRODUCTION_HARNESS_VERSION",
    "PRODUCTION_IMPLEMENTATION_BINDINGS",
    "PRODUCTION_POOL_EXECUTOR_ENTRYPOINT",
    "PRODUCTION_SERIAL_EXECUTOR_ENTRYPOINT",
    "PRODUCTION_SOURCE_BINDINGS",
    "PRODUCTION_TASK_BUILDER_ENTRYPOINT",
    "PRODUCTION_TIMING_CLOCK_ENTRYPOINT",
    "PRODUCTION_WORKLOAD_ENTRYPOINT",
    "PRODUCTION_WORKLOAD_NAME",
    "PRODUCTION_WORKLOAD_VERSION",
    "SCHEMA_VERSION",
    "TEST_EXECUTION_PROFILE",
    "TIMING_CLOCK",
    "BenchmarkConfig",
    "BenchmarkEnvironment",
    "BenchmarkEvidence",
    "BenchmarkEvidenceError",
    "BenchmarkImplementation",
    "BenchmarkSample",
    "BenchmarkSummary",
    "DistributionSummary",
    "TaskGraphBinding",
    "collect_benchmark_environment",
    "load_benchmark_evidence",
    "parse_benchmark_evidence_bytes",
    "require_production_implementation",
    "run_crossover_benchmark",
    "summarize_benchmark",
    "task_declaration_graph_sha256",
    "verify_production_implementation_sources",
    "write_benchmark_evidence",
]
