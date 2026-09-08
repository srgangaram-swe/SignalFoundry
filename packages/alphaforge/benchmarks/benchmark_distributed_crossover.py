"""Generate repeated raw evidence for the local distribution crossover.

The timed region is one complete bounded batch execution.  Every work size has
at least one warmup and seven measured serial/process-pool repetitions, with
backend order alternated and semantic parity required for every pair.  Results
are descriptive single-machine evidence, never a numeric CI gate or SLA.

Run::

    python benchmarks/benchmark_distributed_crossover.py \
        --output runs/distributed-crossover.json
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path
from typing import Any, Final

from alphaforge.distributed.benchmark_evidence import (
    DEFAULT_MAX_TOTAL_SECONDS,
    PRODUCTION_BUDGET_CLOCK_ENTRYPOINT,
    PRODUCTION_EXECUTION_PROFILE,
    PRODUCTION_HARNESS_ENTRYPOINT,
    PRODUCTION_HARNESS_NAME,
    PRODUCTION_HARNESS_VERSION,
    PRODUCTION_POOL_EXECUTOR_ENTRYPOINT,
    PRODUCTION_SERIAL_EXECUTOR_ENTRYPOINT,
    PRODUCTION_TASK_BUILDER_ENTRYPOINT,
    PRODUCTION_TIMING_CLOCK_ENTRYPOINT,
    PRODUCTION_WORKLOAD_ENTRYPOINT,
    PRODUCTION_WORKLOAD_NAME,
    PRODUCTION_WORKLOAD_VERSION,
    BenchmarkConfig,
    BenchmarkEvidenceError,
    BenchmarkImplementation,
    run_crossover_benchmark,
    summarize_benchmark,
    write_benchmark_evidence,
)
from alphaforge.distributed.tasks import ResourceRequest, TaskSpec
from alphaforge.research._bounded_io import BoundedIOError, read_regular_file_snapshot

DEFAULT_ITERATION_COUNTS: Final = (1_000, 50_000, 500_000, 2_000_000)
DEFAULT_TASK_COUNT: Final = 32
DEFAULT_WORKERS: Final = 8
DEFAULT_WARMUPS: Final = 1
DEFAULT_REPETITIONS: Final = 7
MAX_SOURCE_BYTES: Final = 20_000_000


def benchmark_implementation() -> BenchmarkImplementation:
    """Bind the run to exact workload, harness, contracts, and dependency bytes."""

    repository = Path(__file__).resolve().parents[1]
    relative_sources = {
        "benchmark": Path("benchmarks/benchmark_distributed_crossover.py"),
        "evidence": Path("alphaforge/distributed/benchmark_evidence.py"),
        "executor": Path("alphaforge/distributed/executor.py"),
        "tasks": Path("alphaforge/distributed/tasks.py"),
        "lock": Path("uv.lock"),
    }
    try:
        identities = {
            name: read_regular_file_snapshot(
                repository / relative,
                max_bytes=MAX_SOURCE_BYTES,
                root=repository,
            ).sha256
            for name, relative in relative_sources.items()
        }
    except BoundedIOError as exc:
        raise BenchmarkEvidenceError(
            "benchmark source identity inputs must be bounded regular repository files"
        ) from exc
    return BenchmarkImplementation(
        workload_name=PRODUCTION_WORKLOAD_NAME,
        workload_version=PRODUCTION_WORKLOAD_VERSION,
        execution_profile=PRODUCTION_EXECUTION_PROFILE,
        workload_entrypoint=PRODUCTION_WORKLOAD_ENTRYPOINT,
        task_builder_entrypoint=PRODUCTION_TASK_BUILDER_ENTRYPOINT,
        harness_name=PRODUCTION_HARNESS_NAME,
        harness_version=PRODUCTION_HARNESS_VERSION,
        harness_entrypoint=PRODUCTION_HARNESS_ENTRYPOINT,
        serial_executor_entrypoint=PRODUCTION_SERIAL_EXECUTOR_ENTRYPOINT,
        pool_executor_entrypoint=PRODUCTION_POOL_EXECUTOR_ENTRYPOINT,
        timing_clock_entrypoint=PRODUCTION_TIMING_CLOCK_ENTRYPOINT,
        budget_clock_entrypoint=PRODUCTION_BUDGET_CLOCK_ENTRYPOINT,
        workload_source_sha256=identities["benchmark"],
        task_builder_source_sha256=identities["benchmark"],
        harness_source_sha256=identities["benchmark"],
        evidence_contract_source_sha256=identities["evidence"],
        executor_source_sha256=identities["executor"],
        task_contract_source_sha256=identities["tasks"],
        dependency_lock_sha256=identities["lock"],
    )


def busy_work(payload: dict[str, Any]) -> float:
    """Return deterministic CPU work whose cost is controlled by ``iters``."""

    total = 0.0
    for index in range(int(payload["iters"])):
        total += (index % 7) ** 0.5
    return round(total, 6)


def build_batch(count: int, iterations: int) -> tuple[TaskSpec, ...]:
    """Build one content-addressed synthetic task batch."""

    return tuple(
        TaskSpec(
            name=f"probe-{index}",
            payload={"index": index, "iters": iterations},
            seed=index,
            resources=ResourceRequest(
                cpus=1.0,
                memory_mb=256,
                gpus=0,
                scratch_mb=0,
                expected_seconds=60.0,
            ),
            timeout_seconds=300.0,
        )
        for index in range(count)
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Measure the bounded local serial/process-pool crossover.",
    )
    parser.add_argument("--tasks", type=int, default=DEFAULT_TASK_COUNT)
    parser.add_argument("--workers", type=int, default=DEFAULT_WORKERS)
    parser.add_argument("--warmups", type=int, default=DEFAULT_WARMUPS)
    parser.add_argument("--repetitions", type=int, default=DEFAULT_REPETITIONS)
    parser.add_argument(
        "--max-total-seconds",
        type=float,
        default=DEFAULT_MAX_TOTAL_SECONDS,
        help=(
            "total wall-time budget checked between completed calls and passed to backends; "
            "in-process Python is not preempted"
        ),
    )
    parser.add_argument(
        "--iterations",
        type=int,
        nargs="+",
        default=list(DEFAULT_ITERATION_COUNTS),
        metavar="COUNT",
        help="one or more deterministic iteration counts per task",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="new JSON evidence file to create (existing files are refused)",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the benchmark once, publish raw JSON, and print concise medians."""

    parser = _parser()
    args = parser.parse_args(argv)
    try:
        config = BenchmarkConfig(
            implementation=benchmark_implementation(),
            task_count=args.tasks,
            workers=args.workers,
            warmups=args.warmups,
            repetitions=args.repetitions,
            iteration_counts=tuple(args.iterations),
            max_total_seconds=args.max_total_seconds,
        )
        evidence = run_crossover_benchmark(
            config,
            function=busy_work,
            task_builder=build_batch,
            harness=main,
        )
        output = write_benchmark_evidence(evidence, args.output)
    except (BenchmarkEvidenceError, FileExistsError, OSError, ValueError) as exc:
        parser.error(str(exc))
    print(
        f"wrote {len(evidence.samples)} raw samples to {output} "
        f"(benchmark_id={evidence.benchmark_id[:12]})"
    )
    for record in summarize_benchmark(evidence):
        print(
            f"iterations={record.iterations} n={record.sample_count} "
            f"per_task_median_ms={record.per_task_ms.median:.6g} "
            f"speedup_median={record.speedup.median:.6g}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
