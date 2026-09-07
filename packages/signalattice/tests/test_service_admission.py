"""Invariant and adversarial tests for fixed-cardinality admission policy."""

from __future__ import annotations

import threading

import pytest

from quant_platform.service.admission import (
    AdmissionController,
    AdmissionLane,
    AdmissionLimits,
    AdmissionRejection,
)


class _FakeClock:
    def __init__(self, initial: float = 10.0) -> None:
        self.value = initial

    def __call__(self) -> float:
        return self.value

    def advance(self, seconds: float) -> None:
        self.value += seconds


@pytest.mark.parametrize(
    "overrides",
    [
        {"global_concurrency": True},
        {"global_concurrency": 1},
        {"data_concurrency": 33},
        {"data_concurrency": 0},
        {"api_burst": 0},
        {"operations_burst": 0},
        {"api_rate_per_second": 20},
        {"api_rate_per_second": float("nan")},
        {"operations_rate_per_second": float("inf")},
    ],
)
def test_limits_reject_coercion_nonfinite_rates_and_missing_probe_reserve(
    overrides: dict[str, object],
) -> None:
    with pytest.raises(ValueError):
        AdmissionLimits(**overrides)  # type: ignore[arg-type]


def test_two_token_buckets_are_independent_and_refill_with_fake_time() -> None:
    clock = _FakeClock()
    controller = AdmissionController(
        AdmissionLimits(
            global_concurrency=3,
            data_concurrency=2,
            api_rate_per_second=2.0,
            api_burst=2,
            operations_rate_per_second=1.0,
            operations_burst=1,
        ),
        clock=clock,
    )

    for _ in range(2):
        accepted = controller.try_acquire(AdmissionLane.DATA)
        assert accepted.accepted
        controller.release(AdmissionLane.DATA)
    assert controller.try_acquire(AdmissionLane.DATA).rejection is AdmissionRejection.RATE_LIMITED

    operation = controller.try_acquire(AdmissionLane.OPERATIONS)
    assert operation.accepted
    controller.release(AdmissionLane.OPERATIONS)
    assert (
        controller.try_acquire(AdmissionLane.OPERATIONS).rejection
        is AdmissionRejection.RATE_LIMITED
    )

    clock.advance(1.0)
    assert controller.try_acquire(AdmissionLane.DATA).accepted
    controller.release(AdmissionLane.DATA)
    assert controller.try_acquire(AdmissionLane.OPERATIONS).accepted
    controller.release(AdmissionLane.OPERATIONS)


def test_data_saturation_reserves_operations_capacity_without_allocating_clients() -> None:
    controller = AdmissionController(
        AdmissionLimits(
            global_concurrency=4,
            data_concurrency=2,
            api_rate_per_second=100.0,
            api_burst=100,
            operations_rate_per_second=100.0,
            operations_burst=100,
        )
    )
    first = controller.try_acquire(AdmissionLane.DATA)
    second = controller.try_acquire(AdmissionLane.DATA)
    rejected = controller.try_acquire(AdmissionLane.DATA)

    assert first.accepted and second.accepted
    assert rejected.rejection is AdmissionRejection.DATA_CONCURRENCY
    probe = controller.try_acquire(AdmissionLane.OPERATIONS)
    metrics = controller.try_acquire(AdmissionLane.OPERATIONS)
    assert probe.accepted and metrics.accepted
    assert (
        controller.try_acquire(AdmissionLane.OPERATIONS).rejection
        is AdmissionRejection.GLOBAL_CONCURRENCY
    )
    assert controller.snapshot().active_total == 4

    for lane in (
        AdmissionLane.DATA,
        AdmissionLane.DATA,
        AdmissionLane.OPERATIONS,
        AdmissionLane.OPERATIONS,
    ):
        controller.release(lane)
    assert controller.snapshot().active_total == 0


def test_shutdown_is_idempotent_and_future_work_fails_closed() -> None:
    controller = AdmissionController()
    admitted = controller.try_acquire(AdmissionLane.DATA)
    assert admitted.accepted

    controller.begin_shutdown()
    controller.begin_shutdown()

    rejected = controller.try_acquire(AdmissionLane.OPERATIONS)
    assert rejected.rejection is AdmissionRejection.SHUTTING_DOWN
    assert controller.snapshot().active_total == 1
    controller.release(AdmissionLane.DATA)
    assert controller.snapshot().active_total == 0


def test_accounting_underflow_and_regressing_clock_are_detected() -> None:
    controller = AdmissionController()
    with pytest.raises(RuntimeError, match="underflow"):
        controller.release(AdmissionLane.DATA)

    clock = _FakeClock()
    timed = AdmissionController(clock=clock)
    clock.value = 9.0
    with pytest.raises(RuntimeError, match="regressed"):
        timed.try_acquire(AdmissionLane.DATA)


def test_concurrent_acquisition_never_exceeds_global_or_data_limits() -> None:
    controller = AdmissionController(
        AdmissionLimits(
            global_concurrency=8,
            data_concurrency=6,
            api_rate_per_second=10_000.0,
            api_burst=10_000,
            operations_rate_per_second=10_000.0,
            operations_burst=1_000,
        )
    )
    barrier = threading.Barrier(33)
    accepted: list[AdmissionLane] = []
    result_lock = threading.Lock()

    def contend(index: int) -> None:
        lane = AdmissionLane.OPERATIONS if index % 8 == 0 else AdmissionLane.DATA
        barrier.wait()
        decision = controller.try_acquire(lane)
        if decision.accepted:
            with result_lock:
                accepted.append(lane)

    threads = [threading.Thread(target=contend, args=(index,)) for index in range(32)]
    for thread in threads:
        thread.start()
    barrier.wait()
    for thread in threads:
        thread.join(timeout=2)
        assert not thread.is_alive()

    assert len(accepted) <= 8
    assert accepted.count(AdmissionLane.DATA) <= 6
    snapshot = controller.snapshot()
    assert snapshot.active_total == len(accepted)
    assert snapshot.active_data == accepted.count(AdmissionLane.DATA)
    for lane in accepted:
        controller.release(lane)
    assert controller.snapshot().active_total == 0
