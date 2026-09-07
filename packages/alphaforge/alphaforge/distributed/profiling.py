"""Measure the serial pipeline before distributing any part of it.

SF-S5-MR8. The work item's explicit non-goal is "distributing unprofiled code",
and the reason is that distribution has a fixed cost — serialization, scheduling,
worker startup — that is invisible until measured against the work it replaces.
A sweep whose per-point cost is milliseconds gets slower when distributed, and
the only way to know which regime you are in is to measure.

This module produces a :class:`SerialProfile`: per-stage wall time, the share of
total each stage holds, and the **parallel fraction** — the proportion of runtime
spent in work that is independent across grid points and therefore actually
distributable.

The parallel fraction is the number that decides. Amdahl's law bounds achievable
speedup at ``1 / (1 - p)`` regardless of worker count, so a pipeline that is 60%
parallel cannot exceed 2.5x however many machines are added. Reporting that bound
alongside the measurement is what keeps a distribution decision honest: the
speedup you can buy is capped by the serial remainder, not by the cluster.

Measurements are wall-clock on one machine and are labelled as such. They are not
a benchmark of the distributed backend; they are the evidence that distributing
is or is not worth its overhead.
"""

from __future__ import annotations

import platform
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Final

#: Refusal thresholds, not tuning knobs.
MAX_STAGES: Final = 64
MAX_REPEATS: Final = 100

#: Below this parallel fraction, distribution cannot repay its overhead at any
#: worker count: Amdahl caps speedup at 2x. Reported as a recommendation, never
#: enforced — the caller may have a reason this module cannot see.
MIN_USEFUL_PARALLEL_FRACTION: Final = 0.5


class ProfilingError(ValueError):
    """Raised when a profile cannot be measured or is not interpretable."""


@dataclass(frozen=True, slots=True)
class StageTiming:
    """Wall time attributed to one named pipeline stage."""

    name: str
    seconds: float
    parallelizable: bool
    note: str = ""

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name.strip():
            raise ProfilingError("stage name must be a non-empty string")
        if isinstance(self.seconds, bool) or not isinstance(self.seconds, (int, float)):
            raise ProfilingError(f"stage {self.name!r} seconds must be a real number")
        if self.seconds < 0 or self.seconds != self.seconds or self.seconds == float("inf"):
            raise ProfilingError(f"stage {self.name!r} seconds must be finite and non-negative")
        if not isinstance(self.parallelizable, bool):
            raise ProfilingError("parallelizable must be a bool")

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-friendly record."""
        return {
            "name": self.name,
            "seconds": self.seconds,
            "parallelizable": self.parallelizable,
            "note": self.note,
        }


@dataclass(frozen=True)
class SerialProfile:
    """A measured serial baseline and the speedup bound it implies."""

    stages: tuple[StageTiming, ...]
    repeats: int
    environment: Mapping[str, str]

    def __post_init__(self) -> None:
        stages = tuple(self.stages)
        if not stages:
            raise ProfilingError("a profile must contain at least one stage")
        if len(stages) > MAX_STAGES:
            raise ProfilingError(f"profile exceeds the {MAX_STAGES}-stage ceiling")
        names = [item.name for item in stages]
        if len(set(names)) != len(names):
            raise ProfilingError("stage names must be unique within a profile")
        if isinstance(self.repeats, bool) or not isinstance(self.repeats, int):
            raise ProfilingError("repeats must be an int")
        if not 1 <= self.repeats <= MAX_REPEATS:
            raise ProfilingError(f"repeats must lie in [1, {MAX_REPEATS}]")
        object.__setattr__(self, "stages", stages)

    @property
    def total_seconds(self) -> float:
        """Total measured wall time across all stages."""
        return sum(item.seconds for item in self.stages)

    @property
    def parallel_seconds(self) -> float:
        """Wall time in stages that are independent across grid points."""
        return sum(item.seconds for item in self.stages if item.parallelizable)

    @property
    def parallel_fraction(self) -> float:
        """Proportion of runtime that is actually distributable.

        Returns ``nan`` for a zero-duration profile rather than 0.0, because
        "no measurable work" and "no parallel work" are different findings.
        """
        total = self.total_seconds
        if total <= 0.0:
            return float("nan")
        return self.parallel_seconds / total

    def amdahl_speedup_bound(self, workers: int | None = None) -> float:
        """Return the speedup ceiling under Amdahl's law.

        With ``workers`` omitted, returns the infinite-worker bound
        ``1 / (1 - p)`` — the speedup no amount of hardware can exceed.

        Raises:
            ProfilingError: If ``workers`` is not a positive int.
        """
        fraction = self.parallel_fraction
        if fraction != fraction:  # nan
            return float("nan")
        if workers is None:
            if fraction >= 1.0:
                return float("inf")
            return 1.0 / (1.0 - fraction)
        if isinstance(workers, bool) or not isinstance(workers, int) or workers < 1:
            raise ProfilingError("workers must be a positive int")
        return 1.0 / ((1.0 - fraction) + fraction / workers)

    @property
    def distribution_is_justified(self) -> bool:
        """Whether the measured parallel fraction can repay distribution overhead."""
        fraction = self.parallel_fraction
        return fraction == fraction and fraction >= MIN_USEFUL_PARALLEL_FRACTION

    def to_dict(self) -> dict[str, Any]:
        """Return the complete JSON-friendly profile."""
        return {
            "stages": [item.to_dict() for item in self.stages],
            "repeats": self.repeats,
            "environment": dict(self.environment),
            "total_seconds": self.total_seconds,
            "parallel_seconds": self.parallel_seconds,
            "parallel_fraction": self.parallel_fraction,
            "amdahl_bound_infinite_workers": self.amdahl_speedup_bound(),
            "amdahl_bound_8_workers": self.amdahl_speedup_bound(8),
            "distribution_is_justified": self.distribution_is_justified,
            "measurement_note": (
                "Wall-clock on a single machine, not a benchmark of any distributed "
                "backend. Speedup is bounded by the serial remainder, not by worker count."
            ),
        }


def profile_stages(
    stages: Sequence[tuple[str, Callable[[], Any], bool]], *, repeats: int = 3
) -> SerialProfile:
    """Time each stage and return the measured serial profile.

    Each stage is ``(name, callable, parallelizable)``. The callable runs
    ``repeats`` times and the **minimum** is taken, not the mean: the minimum is
    the closest estimate of the work itself, since scheduling noise and
    contention only ever add time.

    Raises:
        ProfilingError: On a malformed stage specification.
    """
    if isinstance(repeats, bool) or not isinstance(repeats, int) or repeats < 1:
        raise ProfilingError("repeats must be a positive int")
    if repeats > MAX_REPEATS:
        raise ProfilingError(f"repeats exceeds the {MAX_REPEATS} ceiling")
    if not stages:
        raise ProfilingError("at least one stage is required")

    timings: list[StageTiming] = []
    for entry in stages:
        if not isinstance(entry, tuple) or len(entry) != 3:
            raise ProfilingError("each stage must be (name, callable, parallelizable)")
        name, function, parallelizable = entry
        if not callable(function):
            raise ProfilingError(f"stage {name!r} must supply a callable")
        best = float("inf")
        for _ in range(repeats):
            started = time.perf_counter()
            function()
            best = min(best, time.perf_counter() - started)
        timings.append(
            StageTiming(name=str(name), seconds=best, parallelizable=bool(parallelizable))
        )

    return SerialProfile(
        stages=tuple(timings),
        repeats=repeats,
        environment={
            "python": platform.python_version(),
            "platform": platform.platform(),
            "machine": platform.machine(),
            "processor": platform.processor() or "unknown",
        },
    )


__all__ = [
    "MAX_REPEATS",
    "MAX_STAGES",
    "MIN_USEFUL_PARALLEL_FRACTION",
    "ProfilingError",
    "SerialProfile",
    "StageTiming",
    "profile_stages",
]
