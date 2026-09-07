"""Backend-neutral bounded execution with order-independent result assembly.

SF-S5-MR8. The contract this module enforces is narrow and load-bearing:

**Results are assembled by task identity, never by completion order.** A
distributed run completes tasks in whatever order workers finish them, and that
order varies between runs on the same inputs. Any assembly that depends on it —
appending to a list, folding in arrival sequence — produces results that differ
run to run while every individual task is deterministic. Sorting by content
identity makes the assembled output a function of the inputs alone.

**Cluster access is never required for reproducibility.** The local backend is
the reference implementation and is always available. A distributed backend is an
optional accelerator that must produce byte-identical results; the parity test
asserts exactly that. The work item names "making cluster access required for
reproducibility" as an explicit non-goal, and a research result that can only be
reproduced with a cluster is not reproducible.

**Failures are bounded and attributed.** Every task carries a timeout and a
retry budget. A worker failure, a straggler exceeding its timeout, and a
cancelled batch all produce a structured outcome naming the task, not a bare
exception from an anonymous worker.

The distributed backend is selected in
:doc:`ADR 0018 </adr/0018-bounded-distributed-research-execution>`; this module
does not import it unless it is requested, so the dependency stays optional.
"""

from __future__ import annotations

import time
import traceback
from collections.abc import Callable, Sequence
from concurrent.futures import FIRST_COMPLETED, Future, ProcessPoolExecutor, wait
from dataclasses import dataclass
from enum import StrEnum
from typing import Any, Final

from alphaforge.distributed.tasks import (
    TaskContractError,
    TaskSpec,
    assert_unique_tasks,
    content_hash,
)

#: Refusal thresholds, not tuning knobs.
MAX_WORKERS: Final = 256
MAX_TOTAL_SECONDS: Final = 24 * 60 * 60


class ExecutionError(ValueError):
    """Raised when a batch cannot be executed as specified."""


class TaskOutcome(StrEnum):
    """How one task ended. Every terminal state is named."""

    SUCCEEDED = "succeeded"
    FAILED = "failed"
    TIMED_OUT = "timed_out"
    CANCELLED = "cancelled"


@dataclass(frozen=True, slots=True)
class TaskResult:
    """The outcome of one task, carrying enough to attribute a failure.

    ``output_hash`` is present only on success and is what the parity check
    compares: two backends agree when every task's output hashes identically,
    which is a stronger claim than the assembled outputs matching.
    """

    task_id: str
    name: str
    outcome: TaskOutcome
    value: Any = None
    output_hash: str | None = None
    seconds: float = 0.0
    attempts: int = 1
    error: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.outcome, TaskOutcome):
            raise ExecutionError("outcome must be a TaskOutcome member")
        if self.outcome is TaskOutcome.SUCCEEDED and self.output_hash is None:
            raise ExecutionError(
                f"task {self.name!r} succeeded without an output hash; the hash is what "
                "makes cross-backend parity checkable"
            )
        if self.outcome is not TaskOutcome.SUCCEEDED and self.error is None:
            raise ExecutionError(
                f"task {self.name!r} ended as {self.outcome.value} without an error "
                "description; an unattributed failure cannot be acted on"
            )
        if isinstance(self.attempts, bool) or not isinstance(self.attempts, int):
            raise ExecutionError("attempts must be an int")
        if self.attempts < 1:
            raise ExecutionError("attempts must be at least 1")

    @property
    def succeeded(self) -> bool:
        """Whether this task produced a value."""
        return self.outcome is TaskOutcome.SUCCEEDED

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-friendly record. Excludes the value itself."""
        return {
            "task_id": self.task_id,
            "name": self.name,
            "outcome": self.outcome.value,
            "output_hash": self.output_hash,
            "seconds": self.seconds,
            "attempts": self.attempts,
            "error": self.error,
        }


@dataclass(frozen=True)
class BatchReport:
    """The complete, deterministically ordered outcome of one batch."""

    results: tuple[TaskResult, ...]
    backend: str
    workers: int
    wall_seconds: float

    def __post_init__(self) -> None:
        results = tuple(self.results)
        identities = [item.task_id for item in results]
        if len(set(identities)) != len(identities):
            raise ExecutionError("a batch report cannot contain two results for one task")
        # Deterministic by identity, never by completion order.
        object.__setattr__(self, "results", tuple(sorted(results, key=lambda r: r.task_id)))

    @property
    def all_succeeded(self) -> bool:
        """Whether every task produced a value."""
        return all(item.succeeded for item in self.results)

    @property
    def failures(self) -> tuple[TaskResult, ...]:
        """Every non-successful result, in identity order."""
        return tuple(item for item in self.results if not item.succeeded)

    def assembly_hash(self) -> str:
        """Content hash over every task's identity and output hash.

        Independent of completion order by construction. Two backends that agree
        on this agree on the whole computation.
        """
        return content_hash(
            [{"task_id": item.task_id, "output_hash": item.output_hash} for item in self.results]
        )

    def values_in_identity_order(self) -> tuple[Any, ...]:
        """Return successful values ordered by task identity."""
        return tuple(item.value for item in self.results if item.succeeded)

    def to_dict(self) -> dict[str, Any]:
        """Return the JSON-friendly report."""
        return {
            "backend": self.backend,
            "workers": self.workers,
            "wall_seconds": self.wall_seconds,
            "task_count": len(self.results),
            "all_succeeded": self.all_succeeded,
            "failure_count": len(self.failures),
            "assembly_hash": self.assembly_hash(),
            "results": [item.to_dict() for item in self.results],
            "ordering_note": (
                "Results are ordered by content-addressed task identity, never by "
                "completion order, so the assembled output is a function of the inputs "
                "alone."
            ),
        }


def _run_one(function: Callable[[Any], Any], task: TaskSpec) -> TaskResult:
    """Run one task in-process with its declared bounds and retry budget."""
    attempts = 0
    last_error: str | None = None
    started = time.perf_counter()
    while attempts <= task.max_retries:
        attempts += 1
        try:
            value = function(task.payload)
        except Exception as error:  # noqa: BLE001 - attributed and re-reported
            last_error = f"{type(error).__name__}: {error}"
            if attempts > task.max_retries:
                break
            continue
        elapsed = time.perf_counter() - started
        if elapsed > task.timeout_seconds:
            return TaskResult(
                task_id=task.task_id,
                name=task.name,
                outcome=TaskOutcome.TIMED_OUT,
                seconds=elapsed,
                attempts=attempts,
                error=(
                    f"exceeded its declared {task.timeout_seconds}s timeout after "
                    f"{elapsed:.3f}s"
                ),
            )
        try:
            digest = content_hash(value)
        except TaskContractError as error:
            return TaskResult(
                task_id=task.task_id,
                name=task.name,
                outcome=TaskOutcome.FAILED,
                seconds=time.perf_counter() - started,
                attempts=attempts,
                error=(f"produced a non-hashable result, so parity cannot be checked: {error}"),
            )
        return TaskResult(
            task_id=task.task_id,
            name=task.name,
            outcome=TaskOutcome.SUCCEEDED,
            value=value,
            output_hash=digest,
            seconds=elapsed,
            attempts=attempts,
        )
    return TaskResult(
        task_id=task.task_id,
        name=task.name,
        outcome=TaskOutcome.FAILED,
        seconds=time.perf_counter() - started,
        attempts=attempts,
        error=last_error or "failed without an exception",
    )


def execute_local(
    function: Callable[[Any], Any],
    tasks: Sequence[TaskSpec],
    *,
    total_timeout_seconds: float = MAX_TOTAL_SECONDS,
) -> BatchReport:
    """Run every task in-process. The reference implementation.

    Always available, requires no cluster, and defines the correct answer that
    every other backend must match.

    Raises:
        ExecutionError: On a malformed batch or an exceeded total budget.
    """
    ordered = tuple(tasks)
    if not ordered:
        raise ExecutionError("a batch must contain at least one task")
    assert_unique_tasks(ordered)
    if (
        isinstance(total_timeout_seconds, bool)
        or not isinstance(total_timeout_seconds, (int, float))
        or not 0 < total_timeout_seconds <= MAX_TOTAL_SECONDS
    ):
        raise ExecutionError(f"total_timeout_seconds must lie in (0, {MAX_TOTAL_SECONDS}]")

    started = time.perf_counter()
    results: list[TaskResult] = []
    for task in ordered:
        if time.perf_counter() - started > total_timeout_seconds:
            results.append(
                TaskResult(
                    task_id=task.task_id,
                    name=task.name,
                    outcome=TaskOutcome.CANCELLED,
                    attempts=1,
                    error="batch exceeded its total time budget before this task started",
                )
            )
            continue
        results.append(_run_one(function, task))
    return BatchReport(
        results=tuple(results),
        backend="local",
        workers=1,
        wall_seconds=time.perf_counter() - started,
    )


def execute_process_pool(
    function: Callable[[Any], Any],
    tasks: Sequence[TaskSpec],
    *,
    workers: int,
    total_timeout_seconds: float = MAX_TOTAL_SECONDS,
) -> BatchReport:
    """Run tasks across local processes, assembling by identity.

    The intermediate rung between in-process and a cluster: real parallelism,
    real serialization, no external service. It is where serialization bugs and
    order-dependence surface, and it needs no deployment to run in CI.

    ``function`` must be importable at module scope, since it is pickled to the
    worker processes.

    Raises:
        ExecutionError: On a malformed batch or worker count.
    """
    ordered = tuple(tasks)
    if not ordered:
        raise ExecutionError("a batch must contain at least one task")
    assert_unique_tasks(ordered)
    if isinstance(workers, bool) or not isinstance(workers, int) or workers < 1:
        raise ExecutionError("workers must be a positive int")
    if workers > MAX_WORKERS:
        raise ExecutionError(f"workers exceeds the {MAX_WORKERS} ceiling")

    started = time.perf_counter()
    results: list[TaskResult] = []
    with ProcessPoolExecutor(max_workers=workers) as pool:
        pending: dict[Future[TaskResult], TaskSpec] = {
            pool.submit(_run_one, function, task): task for task in ordered
        }
        remaining = set(pending)
        while remaining:
            elapsed = time.perf_counter() - started
            budget = total_timeout_seconds - elapsed
            if budget <= 0:
                for future in remaining:
                    task = pending[future]
                    future.cancel()
                    results.append(
                        TaskResult(
                            task_id=task.task_id,
                            name=task.name,
                            outcome=TaskOutcome.CANCELLED,
                            attempts=1,
                            error="batch exceeded its total time budget",
                        )
                    )
                break
            done, remaining = wait(remaining, timeout=budget, return_when=FIRST_COMPLETED)
            for future in done:
                task = pending[future]
                try:
                    results.append(future.result())
                except Exception as error:  # noqa: BLE001 - worker died
                    results.append(
                        TaskResult(
                            task_id=task.task_id,
                            name=task.name,
                            outcome=TaskOutcome.FAILED,
                            attempts=1,
                            error=(
                                f"worker failed: {type(error).__name__}: {error}\n"
                                f"{''.join(traceback.format_exception_only(type(error), error))}"
                            ),
                        )
                    )
    return BatchReport(
        results=tuple(results),
        backend="process_pool",
        workers=workers,
        wall_seconds=time.perf_counter() - started,
    )


def assert_backend_parity(reference: BatchReport, candidate: BatchReport) -> None:
    """Refuse when two backends disagree on any task's output.

    Compares per-task output hashes rather than only the assembly hash, so the
    failure message names the tasks that diverged instead of reporting that
    something, somewhere, differs.

    Raises:
        ExecutionError: Naming the divergent tasks.
    """
    reference_map = {item.task_id: item.output_hash for item in reference.results}
    candidate_map = {item.task_id: item.output_hash for item in candidate.results}
    if set(reference_map) != set(candidate_map):
        only_reference = sorted(set(reference_map) - set(candidate_map))
        only_candidate = sorted(set(candidate_map) - set(reference_map))
        raise ExecutionError(
            f"backends ran different task sets; only in {reference.backend}: "
            f"{[t[:12] for t in only_reference[:5]]}, only in {candidate.backend}: "
            f"{[t[:12] for t in only_candidate[:5]]}"
        )
    divergent = sorted(
        task_id for task_id, digest in reference_map.items() if candidate_map[task_id] != digest
    )
    if divergent:
        raise ExecutionError(
            f"{len(divergent)} task(s) produced different output under {candidate.backend} "
            f"than under {reference.backend}: {[t[:12] for t in divergent[:5]]}. A backend "
            "that changes results is not an accelerator."
        )


__all__ = [
    "MAX_TOTAL_SECONDS",
    "MAX_WORKERS",
    "BatchReport",
    "ExecutionError",
    "TaskOutcome",
    "TaskResult",
    "assert_backend_parity",
    "execute_local",
    "execute_process_pool",
]
