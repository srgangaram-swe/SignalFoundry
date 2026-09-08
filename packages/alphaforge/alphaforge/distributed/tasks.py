"""Resource-declaring task specifications with content-addressed identity.

SF-S5-MR8. A task that does not declare what it needs cannot be scheduled
safely: the scheduler either over-commits a machine or leaves it idle, and both
failures are invisible until the run is already slow or already dead.

Every :class:`TaskSpec` therefore declares CPU, RAM, GPU, scratch space,
expected duration, seed, input hash, timeout, retry budget, and whether it may be
cancelled. None of these has a permissive default that would let a task omit its
requirements by accident — the ones that matter for safety are required
arguments.

**Identity is content-addressed.** ``task_id`` derives from the payload, the
seed, and the declared resources, so the same logical unit of work has the same
identity on every machine and in every run. That is what makes duplicate
detection possible without a coordinator, and what makes a result cache sound.

**Retries are bounded and side-effect-aware.** ``max_retries`` is capped, and a
task that declares ``idempotent=False`` is refused a retry budget above zero:
retrying a non-idempotent task is how one logical unit of work becomes two.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any, Final

#: Refusal thresholds, not tuning knobs.
MAX_CPUS: Final = 256
MAX_MEMORY_MB: Final = 1_024 * 1_024
MAX_GPUS: Final = 16
MAX_SCRATCH_MB: Final = 4 * 1_024 * 1_024
MAX_TIMEOUT_SECONDS: Final = 24 * 60 * 60
MAX_RETRIES: Final = 5
MAX_TASKS_PER_BATCH: Final = 50_000
MAX_NAME_CHARS: Final = 96


class TaskContractError(ValueError):
    """Raised when a task specification is unsafe or incomplete."""


def _canonical(payload: Any) -> bytes:
    """Return deterministic bytes for content addressing."""
    return json.dumps(
        payload, sort_keys=True, separators=(",", ":"), allow_nan=False, ensure_ascii=True
    ).encode("utf-8")


def content_hash(payload: Any) -> str:
    """Return the SHA-256 of a JSON-serializable payload.

    Raises:
        TaskContractError: If the payload is not deterministically serializable.
            A payload containing a set, a float NaN, or an arbitrary object has
            no stable byte representation, and a task identity that varies
            between runs defeats both caching and duplicate detection.
    """
    try:
        return hashlib.sha256(_canonical(payload)).hexdigest()
    except (TypeError, ValueError) as error:
        raise TaskContractError(
            f"task payload is not deterministically serializable: {error}. A payload whose "
            "bytes vary between runs produces a different task identity each time, which "
            "defeats caching and duplicate detection."
        ) from error


@dataclass(frozen=True, slots=True)
class ResourceRequest:
    """What one task needs from a worker.

    Every field is required. A default would let a task omit its requirements by
    accident, and an under-declared task is the one that gets the machine killed.
    """

    cpus: float
    memory_mb: int
    gpus: int
    scratch_mb: int
    expected_seconds: float

    def __post_init__(self) -> None:
        if isinstance(self.cpus, bool) or not isinstance(self.cpus, (int, float)):
            raise TaskContractError("cpus must be a real number")
        if not 0 < self.cpus <= MAX_CPUS or self.cpus != self.cpus:
            raise TaskContractError(f"cpus must lie in (0, {MAX_CPUS}]")
        for name, ceiling in (
            ("memory_mb", MAX_MEMORY_MB),
            ("gpus", MAX_GPUS),
            ("scratch_mb", MAX_SCRATCH_MB),
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int):
                raise TaskContractError(f"{name} must be an int")
            if not 0 <= value <= ceiling:
                raise TaskContractError(f"{name} must lie in [0, {ceiling}]")
        if self.memory_mb <= 0:
            raise TaskContractError("memory_mb must be positive; every task needs memory")
        if (
            isinstance(self.expected_seconds, bool)
            or not isinstance(self.expected_seconds, (int, float))
            or not 0 < self.expected_seconds <= MAX_TIMEOUT_SECONDS
        ):
            raise TaskContractError(f"expected_seconds must lie in (0, {MAX_TIMEOUT_SECONDS}]")

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-friendly record."""
        return {
            "cpus": self.cpus,
            "memory_mb": self.memory_mb,
            "gpus": self.gpus,
            "scratch_mb": self.scratch_mb,
            "expected_seconds": self.expected_seconds,
        }


@dataclass(frozen=True)
class TaskSpec:
    """One unit of independent work, fully described before it is scheduled.

    Attributes:
        name: Human-readable label, not an identity.
        payload: JSON-serializable inputs. Hashed into ``task_id``.
        seed: The named seed for this unit, so a rerun reproduces it exactly.
        resources: Declared requirements.
        timeout_seconds: Hard bound. A task exceeding it is cancelled.
        max_retries: Bounded retry budget. Zero unless the task is idempotent.
        idempotent: Whether re-running produces the same effect. A task that is
            not idempotent may not be retried, because retrying it is how one
            logical unit of work becomes two.
        cancellable: Whether the scheduler may stop it mid-flight.

    Raises:
        TaskContractError: On any unsafe or incomplete declaration.
    """

    name: str
    payload: Any
    seed: int
    resources: ResourceRequest
    timeout_seconds: float
    max_retries: int = 0
    idempotent: bool = True
    cancellable: bool = True

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name.strip():
            raise TaskContractError("task name must be a non-empty string")
        if len(self.name) > MAX_NAME_CHARS:
            raise TaskContractError(f"task name exceeds {MAX_NAME_CHARS} characters")
        if isinstance(self.seed, bool) or not isinstance(self.seed, int) or self.seed < 0:
            raise TaskContractError("seed must be a non-negative int")
        if not isinstance(self.resources, ResourceRequest):
            raise TaskContractError("resources must be a ResourceRequest")
        if (
            isinstance(self.timeout_seconds, bool)
            or not isinstance(self.timeout_seconds, (int, float))
            or not 0 < self.timeout_seconds <= MAX_TIMEOUT_SECONDS
        ):
            raise TaskContractError(f"timeout_seconds must lie in (0, {MAX_TIMEOUT_SECONDS}]")
        if self.timeout_seconds < self.resources.expected_seconds:
            raise TaskContractError(
                f"timeout {self.timeout_seconds}s is below the declared expected duration "
                f"{self.resources.expected_seconds}s; the task would be cancelled while "
                "behaving exactly as specified"
            )
        for flag in ("idempotent", "cancellable"):
            if not isinstance(getattr(self, flag), bool):
                raise TaskContractError(f"{flag} must be a bool")
        if isinstance(self.max_retries, bool) or not isinstance(self.max_retries, int):
            raise TaskContractError("max_retries must be an int")
        if not 0 <= self.max_retries <= MAX_RETRIES:
            raise TaskContractError(f"max_retries must lie in [0, {MAX_RETRIES}]")
        if self.max_retries > 0 and not self.idempotent:
            raise TaskContractError(
                f"task {self.name!r} declares itself non-idempotent but requests "
                f"{self.max_retries} retries; retrying a task with side effects is how one "
                "logical unit of work becomes two"
            )
        # Validate serializability now rather than at submission, so a malformed
        # payload fails where it was written.
        content_hash(self.payload)

    @property
    def input_hash(self) -> str:
        """Content hash of the payload alone."""
        return content_hash(self.payload)

    @property
    def task_id(self) -> str:
        """Content-addressed identity over payload, seed, and resources.

        Two tasks with the same identity are the same work. That is what lets a
        duplicate be detected without a central coordinator.
        """
        return content_hash(
            {
                "name": self.name,
                "input_hash": self.input_hash,
                "seed": self.seed,
                "resources": self.resources.to_dict(),
            }
        )

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-friendly declaration."""
        return {
            "task_id": self.task_id,
            "name": self.name,
            "input_hash": self.input_hash,
            "seed": self.seed,
            "resources": self.resources.to_dict(),
            "timeout_seconds": self.timeout_seconds,
            "max_retries": self.max_retries,
            "idempotent": self.idempotent,
            "cancellable": self.cancellable,
        }


def assert_unique_tasks(tasks: tuple[TaskSpec, ...]) -> None:
    """Refuse a batch containing two tasks with the same identity.

    Identical identity means identical work. Submitting it twice wastes a worker
    at best, and at worst indicates the caller built its batch from a source that
    already contained duplicates — which usually means the sweep is wrong.

    Raises:
        TaskContractError: Naming the duplicated identity.
    """
    if len(tasks) > MAX_TASKS_PER_BATCH:
        raise TaskContractError(f"batch exceeds the {MAX_TASKS_PER_BATCH}-task ceiling")
    seen: dict[str, str] = {}
    for task in tasks:
        existing = seen.get(task.task_id)
        if existing is not None:
            raise TaskContractError(
                f"tasks {existing!r} and {task.name!r} share identity "
                f"{task.task_id[:12]}; identical identity means identical work, so the "
                "batch was built from a source that already contained duplicates"
            )
        seen[task.task_id] = task.name


__all__ = [
    "MAX_RETRIES",
    "MAX_TASKS_PER_BATCH",
    "MAX_TIMEOUT_SECONDS",
    "ResourceRequest",
    "TaskContractError",
    "TaskSpec",
    "assert_unique_tasks",
    "content_hash",
]
