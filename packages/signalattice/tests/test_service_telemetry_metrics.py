"""Cardinality, byte-bound, thread-safety, and hostile-input metric tests."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor

import pytest

from quant_platform.service.metrics import MAX_METRIC_SERIES, MAX_METRICS_BYTES, ServiceMetrics
from quant_platform.service.telemetry_contracts import (
    DropReason,
    ExportOutcome,
    Operation,
    Outcome,
    RejectionReason,
    RequestObservation,
    RouteTemplate,
    TelemetryChannel,
    TelemetryContractError,
    classify_route,
    operation_for_route,
)


def _success_observation(route: RouteTemplate = RouteTemplate.RUN) -> RequestObservation:
    return RequestObservation(
        route=route,
        operation=operation_for_route(route),
        outcome=Outcome.SUCCESS,
        status_code=200,
        duration_seconds=0.0125,
        response_bytes=2_048,
    )


def _sample_value(body: bytes, prefix: bytes) -> int:
    line = next(line for line in body.splitlines() if line.startswith(prefix))
    return int(line.rsplit(b" ", 1)[1])


def test_empty_registry_has_fixed_sub_600_series_and_sub_256_kib_exposition() -> None:
    first = ServiceMetrics().snapshot()
    second = ServiceMetrics().snapshot()

    assert first == second
    assert first.series_count == 296
    assert first.series_count <= MAX_METRIC_SERIES
    assert len(first.body) < MAX_METRICS_BYTES
    assert first.body.endswith(b"\n")
    assert b"# TYPE signalattice_http_requests_total counter\n" in first.body
    assert b"client" not in first.body
    assert b"ticker" not in first.body
    assert b"exception" not in first.body


def test_hostile_high_uniqueness_paths_cannot_create_series_or_disclose_values() -> None:
    metrics = ServiceMetrics()
    initial = metrics.snapshot()
    canary_prefix = "UNIQUE-CANARY-"

    for index in range(10_000):
        route = classify_route(f"/api/v1/runs/{canary_prefix}{index}")
        metrics.record_request(_success_observation(route))

    observed = metrics.snapshot()
    expected_prefix = (
        b'signalattice_http_requests_total{outcome="success",' b'route="/api/v1/runs/{run_id}"}'
    )
    assert route is RouteTemplate.RUN
    assert observed.series_count == initial.series_count == 296
    assert len(observed.body) < MAX_METRICS_BYTES
    assert canary_prefix.encode("ascii") not in observed.body
    assert _sample_value(observed.body, expected_prefix) == 10_000


def test_request_histograms_are_cumulative_and_finite() -> None:
    metrics = ServiceMetrics()
    metrics.record_request(_success_observation(RouteTemplate.RUN))
    snapshot = metrics.snapshot().body

    assert (
        b'signalattice_http_request_duration_seconds_bucket{le="0.005",operation="entity_read"} 0'
        in snapshot
    )
    assert (
        b'signalattice_http_request_duration_seconds_bucket{le="0.025",operation="entity_read"} 1'
        in snapshot
    )
    assert (
        b'signalattice_http_request_duration_seconds_count{operation="entity_read"} 1' in snapshot
    )
    assert b" NaN\n" not in snapshot
    assert b" nan\n" not in snapshot


def test_rejections_drops_exports_and_queue_depth_use_only_closed_labels() -> None:
    metrics = ServiceMetrics()
    rejected = RequestObservation(
        route=RouteTemplate.RUNS,
        operation=Operation.COLLECTION_READ,
        outcome=Outcome.REJECTED,
        status_code=429,
        duration_seconds=0.001,
        response_bytes=512,
        rejection=RejectionReason.API_RATE_LIMIT,
    )

    metrics.record_request(rejected)
    metrics.record_drop(TelemetryChannel.LOG, DropReason.QUEUE_FULL, count=2)
    metrics.record_export_attempt(TelemetryChannel.TRACE, ExportOutcome.TIMEOUT)
    metrics.set_queue_depth(TelemetryChannel.LOG, 3)
    metrics.set_in_flight(Operation.COLLECTION_READ, 4)
    metrics.record_worker_termination_failure(TelemetryChannel.LOG)
    metrics.record_delivery_indeterminate(TelemetryChannel.TRACE, count=3)
    body = metrics.snapshot().body

    assert b'signalattice_admission_rejections_total{reason="api_rate_limit"} 1' in body
    assert (
        b'signalattice_telemetry_dropped_records_total{channel="log",reason="queue_full"} 2' in body
    )
    assert (
        b'signalattice_telemetry_export_attempts_total{channel="trace",outcome="timeout"} 1' in body
    )
    assert b'signalattice_telemetry_queue_depth{channel="log"} 3' in body
    assert b'signalattice_telemetry_worker_termination_failures_total{channel="log"} 1' in body
    assert b'signalattice_telemetry_delivery_indeterminate_records_total{channel="trace"} 3' in body
    assert b'signalattice_http_in_flight{operation="collection_read"} 4' in body


def test_in_flight_adjustment_is_atomic_and_detects_underflow() -> None:
    metrics = ServiceMetrics()

    metrics.adjust_in_flight(Operation.ENTITY_READ, 1)
    assert b'signalattice_http_in_flight{operation="entity_read"} 1' in metrics.snapshot().body
    metrics.adjust_in_flight(Operation.ENTITY_READ, -1)

    with pytest.raises(TelemetryContractError, match="leave its bound"):
        metrics.adjust_in_flight(Operation.ENTITY_READ, -1)
    with pytest.raises(TelemetryContractError, match="exactly -1 or 1"):
        metrics.adjust_in_flight(Operation.ENTITY_READ, 0)


def test_metric_mutators_reject_coercion_non_enums_and_invalid_counts() -> None:
    metrics = ServiceMetrics()

    with pytest.raises(TelemetryContractError, match="exact RequestObservation"):
        metrics.record_request(object())  # type: ignore[arg-type]
    with pytest.raises(TelemetryContractError, match="exact telemetry enum"):
        metrics.record_drop("log", DropReason.QUEUE_FULL)  # type: ignore[arg-type]
    with pytest.raises(TelemetryContractError, match="integer"):
        metrics.record_drop(TelemetryChannel.LOG, DropReason.QUEUE_FULL, count=True)
    with pytest.raises(TelemetryContractError, match="integer"):
        metrics.set_queue_depth(TelemetryChannel.LOG, -1)
    with pytest.raises(TelemetryContractError, match="exact telemetry enum"):
        metrics.record_worker_termination_failure("log")  # type: ignore[arg-type]


def test_concurrent_updates_are_not_lost() -> None:
    metrics = ServiceMetrics()
    observation = _success_observation()
    workers = 8
    updates_per_worker = 1_000

    def update() -> None:
        for _ in range(updates_per_worker):
            metrics.record_request(observation)

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(update) for _ in range(workers)]
        for future in futures:
            future.result(timeout=5)

    prefix = (
        b'signalattice_http_requests_total{outcome="success",' b'route="/api/v1/runs/{run_id}"}'
    )
    assert _sample_value(metrics.snapshot().body, prefix) == workers * updates_per_worker
