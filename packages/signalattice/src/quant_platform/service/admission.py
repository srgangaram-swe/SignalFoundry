"""Deterministic, fixed-cardinality admission control for the local service.

The controller owns exactly two global token buckets and two concurrency
counters.  It deliberately has no client-keyed state: hostile identifiers can
therefore neither allocate memory nor alter admission-policy cardinality.
All state transitions are constant-time and protected by one short critical
section that never performs I/O.
"""

from __future__ import annotations

import math
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from enum import StrEnum


class AdmissionLane(StrEnum):
    """Closed request classes with separately reserved capacity."""

    DATA = "data"
    OPERATIONS = "operations"


class AdmissionRejection(StrEnum):
    """Stable, non-reflective reasons an exchange was not admitted."""

    RATE_LIMITED = "rate_limited"
    DATA_CONCURRENCY = "data_concurrency"
    GLOBAL_CONCURRENCY = "global_concurrency"
    SHUTTING_DOWN = "shutting_down"


@dataclass(frozen=True, slots=True)
class AdmissionLimits:
    """Resource policy for the two fixed admission lanes.

    Rates are tokens per monotonic second and bursts are exact token counts.
    The production default keeps the data limit below the global limit so
    probes and the private metrics scrape retain bounded headroom during
    data-route saturation.  Smaller explicit test profiles may set them equal.
    """

    global_concurrency: int = 32
    data_concurrency: int = 24
    api_rate_per_second: float = 20.0
    api_burst: int = 40
    operations_rate_per_second: float = 2.0
    operations_burst: int = 4

    def __post_init__(self) -> None:
        integer_bounds = {
            "global_concurrency": (self.global_concurrency, 1, 1_024),
            "data_concurrency": (self.data_concurrency, 1, 1_024),
            "api_burst": (self.api_burst, 1, 10_000),
            "operations_burst": (self.operations_burst, 1, 1_000),
        }
        for name, (value, lower, upper) in integer_bounds.items():
            if type(value) is not int or not lower <= value <= upper:
                raise ValueError(f"{name} must be an integer in [{lower}, {upper}]")
        if self.data_concurrency > self.global_concurrency:
            raise ValueError("data_concurrency cannot exceed global_concurrency")
        rate_values: tuple[tuple[str, object], ...] = (
            ("api_rate_per_second", self.api_rate_per_second),
            ("operations_rate_per_second", self.operations_rate_per_second),
        )
        for rate_name, rate_value in rate_values:
            if (
                type(rate_value) is not float
                or not math.isfinite(rate_value)
                or not 0.01 <= rate_value <= 10_000
            ):
                raise ValueError(f"{rate_name} must be a finite positive float")


@dataclass(frozen=True, slots=True)
class AdmissionDecision:
    """Result of one admission attempt; accepted decisions require release."""

    accepted: bool
    lane: AdmissionLane
    rejection: AdmissionRejection | None = None

    def __post_init__(self) -> None:
        if type(self.accepted) is not bool or type(self.lane) is not AdmissionLane:
            raise TypeError("admission decisions require exact closed types")
        if self.accepted == (self.rejection is not None):
            raise ValueError("accepted decisions cannot carry a rejection reason")


@dataclass(frozen=True, slots=True)
class AdmissionSnapshot:
    """Bounded operational counters without request or business identifiers."""

    accepting: bool
    active_total: int
    active_data: int
    active_operations: int


@dataclass(slots=True)
class _TokenBucket:
    rate: float
    capacity: float
    tokens: float
    updated_at: float

    def consume(self, now: float) -> bool:
        if not math.isfinite(now) or now < 0:
            raise RuntimeError("monotonic admission clock returned an invalid value")
        if now < self.updated_at:
            # A regressing monotonic source is an internal integrity failure.  Do
            # not refill from a negative interval or silently reset the policy.
            raise RuntimeError("monotonic admission clock regressed")
        elapsed = now - self.updated_at
        self.tokens = min(self.capacity, self.tokens + elapsed * self.rate)
        self.updated_at = now
        if self.tokens < 1.0:
            return False
        self.tokens -= 1.0
        return True


class AdmissionController:
    """Thread-safe constant-space token-bucket and concurrency controller."""

    def __init__(
        self,
        limits: AdmissionLimits | None = None,
        *,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        resolved = AdmissionLimits() if limits is None else limits
        if type(resolved) is not AdmissionLimits:
            raise TypeError("limits must be an AdmissionLimits value")
        if not callable(clock):
            raise TypeError("clock must be callable")
        initial = clock()
        if type(initial) is not float or not math.isfinite(initial) or initial < 0:
            raise ValueError("clock must return a finite non-negative float")
        self._limits = resolved
        self._clock = clock
        self._buckets = {
            AdmissionLane.DATA: _TokenBucket(
                resolved.api_rate_per_second,
                float(resolved.api_burst),
                float(resolved.api_burst),
                initial,
            ),
            AdmissionLane.OPERATIONS: _TokenBucket(
                resolved.operations_rate_per_second,
                float(resolved.operations_burst),
                float(resolved.operations_burst),
                initial,
            ),
        }
        self._lock = threading.Lock()
        self._accepting = True
        self._active_total = 0
        self._active_data = 0

    @property
    def limits(self) -> AdmissionLimits:
        """Return the immutable resource policy."""

        return self._limits

    def try_acquire(self, lane: AdmissionLane) -> AdmissionDecision:
        """Attempt one non-blocking admission transition in constant time."""

        if type(lane) is not AdmissionLane:
            raise TypeError("lane must be an AdmissionLane")
        with self._lock:
            if not self._accepting:
                return AdmissionDecision(False, lane, AdmissionRejection.SHUTTING_DOWN)
            if self._active_total >= self._limits.global_concurrency:
                return AdmissionDecision(False, lane, AdmissionRejection.GLOBAL_CONCURRENCY)
            if lane is AdmissionLane.DATA and self._active_data >= self._limits.data_concurrency:
                return AdmissionDecision(False, lane, AdmissionRejection.DATA_CONCURRENCY)
            now = self._clock()
            if type(now) is not float:
                raise RuntimeError("monotonic admission clock changed value type")
            if not self._buckets[lane].consume(now):
                return AdmissionDecision(False, lane, AdmissionRejection.RATE_LIMITED)
            self._active_total += 1
            if lane is AdmissionLane.DATA:
                self._active_data += 1
            return AdmissionDecision(True, lane)

    def release(self, lane: AdmissionLane) -> None:
        """Release one previously accepted request, detecting accounting bugs."""

        if type(lane) is not AdmissionLane:
            raise TypeError("lane must be an AdmissionLane")
        with self._lock:
            if self._active_total <= 0:
                raise RuntimeError("global admission accounting underflow")
            if lane is AdmissionLane.DATA:
                if self._active_data <= 0:
                    raise RuntimeError("data admission accounting underflow")
                self._active_data -= 1
            elif self._active_total - self._active_data <= 0:
                raise RuntimeError("operations admission accounting underflow")
            self._active_total -= 1

    def begin_shutdown(self) -> None:
        """Fail future requests closed while allowing admitted work to drain."""

        with self._lock:
            self._accepting = False

    def snapshot(self) -> AdmissionSnapshot:
        """Return only fixed-cardinality aggregate state."""

        with self._lock:
            operations = self._active_total - self._active_data
            if operations < 0:
                raise RuntimeError("admission counters violated their invariant")
            return AdmissionSnapshot(
                accepting=self._accepting,
                active_total=self._active_total,
                active_data=self._active_data,
                active_operations=operations,
            )
