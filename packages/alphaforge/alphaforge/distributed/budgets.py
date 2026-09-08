"""Declared resource budgets, admitted before spending and enforced during.

SF-S5-MR9. A budget checked only while running is not a budget — by the time
the check fires the resource is already consumed. Admission control is what makes
the limit real: a batch whose declared requirements exceed the budget is refused
**before** any worker starts, so the failure costs nothing.

The work item's non-goal is "best-effort budgets". Every limit here is a hard
refusal, and there is no `soft`, `warn_only`, or `allow_overage` parameter.

Two distinct checks, and both are necessary:

**Admission** compares the sum of declared task requirements against the budget.
It catches the batch that could never have fit — the common case, and the cheap
one to catch.

**Runtime enforcement** compares observed consumption against the budget as work
proceeds. It catches the task that *declared* one hour and is now four hours in.
Admission alone would trust the declaration; enforcement alone would let a
hopeless batch burn a cluster before noticing.

Cancellation produces machine-readable evidence naming which limit was reached,
what was declared, and what was observed — because "the job was cancelled" is not
something an operator can act on.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import StrEnum
from typing import Any, Final

from alphaforge.distributed.tasks import TaskSpec

#: Refusal thresholds, not tuning knobs.
MAX_WALL_SECONDS: Final = 30 * 24 * 60 * 60
MAX_GPU_HOURS: Final = 10_000.0
MAX_CONCURRENT_TASKS: Final = 4_096
MAX_MEMORY_MB: Final = 64 * 1_024 * 1_024
MAX_STORAGE_MB: Final = 64 * 1_024 * 1_024
MAX_COST_UNITS: Final = 1_000_000.0


class BudgetError(ValueError):
    """Raised when a budget is malformed or cannot be satisfied."""


class BudgetExceededError(BudgetError):
    """Raised when a declared or observed requirement exceeds its limit.

    Distinct from a malformed budget: the budget is valid and the work does not
    fit inside it.
    """


class LimitKind(StrEnum):
    """Which declared limit was reached. Named so an operator can act on it."""

    WALL_SECONDS = "wall_seconds"
    GPU_HOURS = "gpu_hours"
    CONCURRENT_TASKS = "concurrent_tasks"
    MEMORY_MB = "memory_mb"
    STORAGE_MB = "storage_mb"
    COST_UNITS = "cost_units"


@dataclass(frozen=True)
class ExperimentBudget:
    """Hard limits for one experiment. Every field is a refusal threshold.

    ``cost_units`` is deliberately abstract rather than a currency: this
    repository has no billing integration, and denominating a limit in dollars
    would imply a price feed that does not exist. Callers map units to whatever
    their provider charges.

    Raises:
        BudgetError: On any non-positive, non-finite, or oversized limit.
    """

    wall_seconds: float
    gpu_hours: float
    concurrent_tasks: int
    memory_mb: int
    storage_mb: int
    cost_units: float

    def __post_init__(self) -> None:
        for name, ceiling in (
            ("wall_seconds", MAX_WALL_SECONDS),
            ("gpu_hours", MAX_GPU_HOURS),
            ("cost_units", MAX_COST_UNITS),
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise BudgetError(f"{name} must be a real number")
            if value != value or value <= 0.0 or value > ceiling:
                raise BudgetError(f"{name} must be finite and lie in (0, {ceiling}]")
        for name, ceiling in (
            ("concurrent_tasks", MAX_CONCURRENT_TASKS),
            ("memory_mb", MAX_MEMORY_MB),
            ("storage_mb", MAX_STORAGE_MB),
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int):
                raise BudgetError(f"{name} must be an int")
            if not 0 < value <= ceiling:
                raise BudgetError(f"{name} must lie in (0, {ceiling}]")

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-friendly declaration."""
        return {
            "wall_seconds": self.wall_seconds,
            "gpu_hours": self.gpu_hours,
            "concurrent_tasks": self.concurrent_tasks,
            "memory_mb": self.memory_mb,
            "storage_mb": self.storage_mb,
            "cost_units": self.cost_units,
            "enforcement": "hard refusal; there is no soft or best-effort mode",
        }


@dataclass(frozen=True, slots=True)
class ResourceUsage:
    """Observed consumption at one instant."""

    wall_seconds: float = 0.0
    gpu_hours: float = 0.0
    concurrent_tasks: int = 0
    memory_mb: int = 0
    storage_mb: int = 0
    cost_units: float = 0.0

    def __post_init__(self) -> None:
        for name in ("wall_seconds", "gpu_hours", "cost_units"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise BudgetError(f"observed {name} must be a real number")
            if value != value or value < 0.0:
                raise BudgetError(f"observed {name} must be finite and non-negative")
        for name in ("concurrent_tasks", "memory_mb", "storage_mb"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise BudgetError(f"observed {name} must be a non-negative int")

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-friendly record."""
        return {
            "wall_seconds": self.wall_seconds,
            "gpu_hours": self.gpu_hours,
            "concurrent_tasks": self.concurrent_tasks,
            "memory_mb": self.memory_mb,
            "storage_mb": self.storage_mb,
            "cost_units": self.cost_units,
        }


@dataclass(frozen=True, slots=True)
class BudgetBreach:
    """Machine-readable evidence of which limit was reached and by how much.

    "The job was cancelled" is not actionable. This names the limit, the
    declared bound, and the observed value.
    """

    kind: LimitKind
    limit: float
    observed: float
    stage: str
    detail: str

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-friendly record."""
        return {
            "kind": self.kind.value,
            "limit": self.limit,
            "observed": self.observed,
            "overage": self.observed - self.limit,
            "stage": self.stage,
            "detail": self.detail,
        }


@dataclass(frozen=True)
class AdmissionDecision:
    """Whether a batch may start, and why not when it may not."""

    admitted: bool
    budget: ExperimentBudget
    declared: ResourceUsage
    breaches: tuple[BudgetBreach, ...]
    decided_at: str

    def __post_init__(self) -> None:
        if self.admitted and self.breaches:
            raise BudgetError("a batch cannot be admitted while carrying breaches")
        if not self.admitted and not self.breaches:
            raise BudgetError(
                "a refused batch must name at least one breach; an unexplained refusal "
                "cannot be acted on"
            )

    def to_dict(self) -> dict[str, Any]:
        """Return the complete JSON-friendly decision."""
        return {
            "admitted": self.admitted,
            "decided_at": self.decided_at,
            "budget": self.budget.to_dict(),
            "declared": self.declared.to_dict(),
            "breaches": [item.to_dict() for item in self.breaches],
            "policy": (
                "Admission compares declared requirements against the budget before any "
                "worker starts, so a batch that could never fit fails at zero cost."
            ),
        }


def declared_usage(tasks: Sequence[TaskSpec], *, concurrency: int) -> ResourceUsage:
    """Sum what a batch says it needs.

    Wall time is the *critical path* under the declared concurrency, not the sum
    of durations: tasks run in parallel, and charging a batch the serial total
    would refuse work that fits comfortably. Memory is the peak concurrent
    requirement for the same reason. GPU-hours and storage accumulate, because
    those are consumed in total regardless of overlap.

    Raises:
        BudgetError: On an empty batch or non-positive concurrency.
    """
    ordered = tuple(tasks)
    if not ordered:
        raise BudgetError("cannot compute declared usage for an empty batch")
    if isinstance(concurrency, bool) or not isinstance(concurrency, int) or concurrency < 1:
        raise BudgetError("concurrency must be a positive int")

    total_seconds = sum(task.resources.expected_seconds for task in ordered)
    # Critical path lower bound: the longest single task, or the evenly divided
    # total, whichever is larger. A batch cannot finish faster than its slowest
    # member however many workers are available.
    longest = max(task.resources.expected_seconds for task in ordered)
    wall = max(longest, total_seconds / concurrency)

    gpu_hours = sum(
        task.resources.gpus * task.resources.expected_seconds / 3_600.0 for task in ordered
    )
    peak_memory = sum(
        sorted((task.resources.memory_mb for task in ordered), reverse=True)[:concurrency]
    )
    storage = sum(task.resources.scratch_mb for task in ordered)
    return ResourceUsage(
        wall_seconds=wall,
        gpu_hours=gpu_hours,
        concurrent_tasks=min(concurrency, len(ordered)),
        memory_mb=peak_memory,
        storage_mb=storage,
        cost_units=0.0,
    )


def _compare(
    budget: ExperimentBudget, usage: ResourceUsage, *, stage: str
) -> tuple[BudgetBreach, ...]:
    """Return every limit the usage exceeds."""
    checks: tuple[tuple[LimitKind, float, float, str], ...] = (
        (
            LimitKind.WALL_SECONDS,
            budget.wall_seconds,
            usage.wall_seconds,
            "elapsed or critical-path wall time",
        ),
        (LimitKind.GPU_HOURS, budget.gpu_hours, usage.gpu_hours, "accumulated GPU hours"),
        (
            LimitKind.CONCURRENT_TASKS,
            float(budget.concurrent_tasks),
            float(usage.concurrent_tasks),
            "simultaneously running tasks",
        ),
        (
            LimitKind.MEMORY_MB,
            float(budget.memory_mb),
            float(usage.memory_mb),
            "peak concurrent memory",
        ),
        (
            LimitKind.STORAGE_MB,
            float(budget.storage_mb),
            float(usage.storage_mb),
            "accumulated scratch and artifact storage",
        ),
        (LimitKind.COST_UNITS, budget.cost_units, usage.cost_units, "accumulated cost units"),
    )
    return tuple(
        BudgetBreach(
            kind=kind,
            limit=limit,
            observed=observed,
            stage=stage,
            detail=f"{detail}: {observed:g} exceeds the declared limit {limit:g}",
        )
        for kind, limit, observed, detail in checks
        if observed > limit
    )


def admit(
    budget: ExperimentBudget,
    tasks: Sequence[TaskSpec],
    *,
    concurrency: int,
    now: datetime | None = None,
) -> AdmissionDecision:
    """Decide whether a batch may start, before any worker does.

    Returns a decision rather than raising, so a caller can record the refusal
    as evidence. Use :func:`enforce` for the raising form during a run.
    """
    declared = declared_usage(tasks, concurrency=concurrency)
    breaches = _compare(budget, declared, stage="admission")
    moment = (now or datetime.now(UTC)).astimezone(UTC)
    return AdmissionDecision(
        admitted=not breaches,
        budget=budget,
        declared=declared,
        breaches=breaches,
        decided_at=moment.isoformat(),
    )


def enforce(budget: ExperimentBudget, usage: ResourceUsage, *, stage: str = "runtime") -> None:
    """Refuse to continue when observed consumption exceeds any limit.

    Admission trusts the declaration; this does not. A task that declared one
    hour and is now four hours in is caught here and nowhere else.

    Raises:
        BudgetExceededError: Naming every breached limit with its overage.
    """
    breaches = _compare(budget, usage, stage=stage)
    if breaches:
        summary = "; ".join(item.detail for item in breaches)
        raise BudgetExceededError(
            f"{len(breaches)} budget limit(s) exceeded at {stage}: {summary}. "
            "Budgets are hard refusals; there is no best-effort mode."
        )


def breach_report(breaches: Sequence[BudgetBreach]) -> str:
    """Return machine-readable cancellation evidence as canonical JSON."""
    return json.dumps(
        {
            "breach_count": len(breaches),
            "breaches": [item.to_dict() for item in breaches],
            "action": (
                "Cancelled. Raise the declared limit deliberately or reduce the work; "
                "the budget is not advisory."
            ),
        },
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


__all__ = [
    "MAX_CONCURRENT_TASKS",
    "MAX_COST_UNITS",
    "MAX_GPU_HOURS",
    "MAX_WALL_SECONDS",
    "AdmissionDecision",
    "BudgetBreach",
    "BudgetError",
    "BudgetExceededError",
    "ExperimentBudget",
    "LimitKind",
    "ResourceUsage",
    "admit",
    "breach_report",
    "declared_usage",
    "enforce",
]
