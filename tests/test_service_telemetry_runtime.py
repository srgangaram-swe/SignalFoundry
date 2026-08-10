"""Fake-clock, queue, retry, timeout, and shutdown tests for telemetry runtime."""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass

import pytest

from quant_platform.service.exporter import ExporterConfig
from quant_platform.service.metrics import ServiceMetrics
from quant_platform.service.telemetry import (
    EmissionResult,
    LifecyclePhase,
    RuntimeState,
    ServiceTelemetry,
    TelemetryRuntimeError,
)
from quant_platform.service.telemetry_contracts import (
    LifecycleState,
    Outcome,
    RejectionReason,
    RouteTemplate,
    TelemetryChannel,
    lifecycle_record,
)


@dataclass
class _FakeClock:
    wall: float = 1_786_291_200.0
    elapsed: float = 10.0

    def time(self) -> float:
        return self.wall

    def monotonic(self) -> float:
        return self.elapsed


class _RecordingExporter:
    def __init__(self, *, failures: int = 0) -> None:
        self.failures = failures
        self.calls: list[tuple[TelemetryChannel, tuple[bytes, ...]]] = []

    async def export(
        self,
        channel: TelemetryChannel,
        records: tuple[bytes, ...],
    ) -> None:
        self.calls.append((channel, records))
        if len(self.calls) <= self.failures:
            raise RuntimeError("REMOTE-CANARY must never reach telemetry")


def _enabled_config(**overrides: object) -> ExporterConfig:
    values: dict[str, object] = {
        "endpoint": "https://telemetry.example/v1/records",
        "timeout_seconds": 0.25,
        "queue_capacity": 4,
        "batch_size": 2,
        "shutdown_timeout_seconds": 1.0,
    }
    values.update(overrides)
    return ExporterConfig(**values)  # type: ignore[arg-type]


def _metric_value(body: bytes, prefix: bytes) -> int:
    line = next(line for line in body.splitlines() if line.startswith(prefix))
    return int(line.rsplit(b" ", 1)[1])


def test_disabled_export_retains_local_records_and_discards_raw_path() -> None:
    metrics = ServiceMetrics()
    clock = _FakeClock()
    telemetry = ServiceTelemetry(metrics, clock=clock)
    canary = "PRIVATE-BUSINESS-ID"

    timer = telemetry.begin_request(f"/api/v1/runs/{canary}")
    clock.elapsed += 0.125
    result = telemetry.finish_request(
        timer,
        outcome=Outcome.SUCCESS,
        status_code=200,
        response_bytes=1_024,
    )
    snapshot = telemetry.metrics.snapshot()

    assert telemetry.state is RuntimeState.DISABLED
    assert telemetry.metrics is metrics
    assert timer.route is RouteTemplate.RUN
    assert canary not in repr(timer)
    assert result.log is EmissionResult.LOCAL_ONLY
    assert result.trace is EmissionResult.LOCAL_ONLY
    assert canary.encode("ascii") not in snapshot.body
    local = telemetry.local_snapshot()
    assert len(local.records) == 2
    assert local.total_bytes <= local.capacity * 8 * 1_024
    assert {json.loads(record)["channel"] for record in local.records} == {"log", "trace"}
    assert canary.encode("ascii") not in b"".join(local.records)
    assert (
        b'signalattice_http_request_duration_seconds_sum{operation="entity_read"} 0.125'
        in snapshot.body
    )


def test_fake_clock_regression_and_invalid_values_fail_explicitly() -> None:
    clock = _FakeClock()
    telemetry = ServiceTelemetry(ServiceMetrics(), clock=clock)
    timer = telemetry.begin_request("/health/live")
    clock.elapsed -= 1.0

    with pytest.raises(TelemetryRuntimeError, match="regressed"):
        telemetry.finish_request(
            timer,
            outcome=Outcome.SUCCESS,
            status_code=200,
            response_bytes=32,
        )

    clock.wall = float("nan")
    with pytest.raises(TelemetryRuntimeError, match="invalid request timestamp"):
        telemetry.begin_request("/health/live")


def test_runtime_requires_exporter_exactly_when_endpoint_is_enabled() -> None:
    with pytest.raises(TelemetryRuntimeError, match="required exactly"):
        ServiceTelemetry(ServiceMetrics(), _enabled_config())
    with pytest.raises(TelemetryRuntimeError, match="required exactly"):
        ServiceTelemetry(ServiceMetrics(), exporter=_RecordingExporter())


def test_successful_export_is_async_batched_and_lifecycle_is_single_use() -> None:
    async def scenario() -> None:
        exporter = _RecordingExporter()
        telemetry = ServiceTelemetry(ServiceMetrics(), _enabled_config(), exporter)

        await telemetry.start()
        assert telemetry.state is RuntimeState.RUNNING
        with pytest.raises(TelemetryRuntimeError, match="exactly once"):
            await telemetry.start()
        assert (
            telemetry.emit(lifecycle_record(LifecycleState.READY, occurred_at_unix_ms=1))
            is EmissionResult.ACCEPTED
        )
        report = await telemetry.stop()

        assert report.flushed is True
        assert report.dropped_records == 0
        assert report.indeterminate_records == 0
        assert report.workers_terminated is True
        assert telemetry.state is RuntimeState.CLOSED
        assert len(exporter.calls) == 1
        assert exporter.calls[0][0] is TelemetryChannel.LOG
        assert exporter.calls[0][1][0].endswith(b"\n")

    asyncio.run(scenario())


def test_local_lifecycle_records_startup_draining_and_shutdown_in_order() -> None:
    async def scenario() -> None:
        telemetry = ServiceTelemetry(ServiceMetrics(), clock=_FakeClock())

        await telemetry.start()
        assert telemetry.lifecycle_phase is LifecyclePhase.RUNNING
        telemetry.begin_draining()
        assert telemetry.lifecycle_phase is LifecyclePhase.DRAINING
        report = await telemetry.stop()

        assert report.flushed is True
        assert report.workers_terminated is True
        assert telemetry.lifecycle_phase is LifecyclePhase.CLOSED
        lifecycle = [
            json.loads(record)["attributes"]["lifecycle"]
            for record in telemetry.local_snapshot().records
            if json.loads(record)["event"] == "service_lifecycle"
        ]
        assert lifecycle == ["starting", "ready", "draining", "stopping", "stopped"]
        with pytest.raises(TelemetryRuntimeError, match="exactly once"):
            await telemetry.start()
        with pytest.raises(TelemetryRuntimeError, match="running service"):
            telemetry.begin_draining()
        with pytest.raises(TelemetryRuntimeError, match="running or draining"):
            await telemetry.stop()

    asyncio.run(scenario())


def test_local_ring_is_bounded_and_eviction_is_observable() -> None:
    metrics = ServiceMetrics()
    telemetry = ServiceTelemetry(metrics, local_record_capacity=2)
    records = tuple(
        lifecycle_record(state, occurred_at_unix_ms=index)
        for index, state in enumerate(
            (LifecycleState.STARTING, LifecycleState.READY, LifecycleState.DRAINING),
            start=1,
        )
    )

    assert all(telemetry.emit(record) is EmissionResult.LOCAL_ONLY for record in records)

    snapshot = telemetry.local_snapshot()
    assert len(snapshot.records) == snapshot.capacity == 2
    assert snapshot.evicted_records == 1
    assert b'"lifecycle":"starting"' not in b"".join(snapshot.records)
    assert (
        b'signalattice_telemetry_dropped_records_total{channel="log",reason="local_sink_capacity"} 1'
        in metrics.snapshot().body
    )


def test_local_sink_failure_is_contained_counted_and_redacted(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    telemetry = ServiceTelemetry(ServiceMetrics())

    def fail_retention(
        _sink: object,
        _channel: TelemetryChannel,
        _record: bytes,
    ) -> TelemetryChannel | None:
        raise RuntimeError("LOCAL-SINK-PRIVATE-CANARY")

    monkeypatch.setattr(type(telemetry._local_sink), "retain", fail_retention)
    result = telemetry.emit(lifecycle_record(LifecycleState.READY, occurred_at_unix_ms=1))
    body = telemetry.metrics.snapshot().body

    assert result is EmissionResult.LOCAL_SINK_FAILED
    assert (
        b'signalattice_telemetry_dropped_records_total{channel="log",reason="local_sink_failure"} 1'
        in body
    )
    assert b"LOCAL-SINK-PRIVATE-CANARY" not in body


def test_full_queue_drops_immediately_without_invoking_exporter_on_request_task() -> None:
    class BlockingExporter:
        def __init__(self) -> None:
            self.entered = asyncio.Event()
            self.release = asyncio.Event()

        async def export(
            self,
            _channel: TelemetryChannel,
            _records: tuple[bytes, ...],
        ) -> None:
            self.entered.set()
            await self.release.wait()

    async def scenario() -> None:
        exporter = BlockingExporter()
        metrics = ServiceMetrics()
        telemetry = ServiceTelemetry(
            metrics,
            _enabled_config(queue_capacity=2, batch_size=1),
            exporter,
        )
        record = lifecycle_record(LifecycleState.READY, occurred_at_unix_ms=1)
        await telemetry.start()

        assert telemetry.emit(record) is EmissionResult.ACCEPTED
        await exporter.entered.wait()
        assert telemetry.emit(record) is EmissionResult.ACCEPTED
        assert telemetry.emit(record) is EmissionResult.ACCEPTED
        assert telemetry.emit(record) is EmissionResult.QUEUE_FULL
        exporter.release.set()
        report = await telemetry.stop()

        assert report.flushed is True
        body = metrics.snapshot().body
        assert (
            b'signalattice_telemetry_dropped_records_total{channel="log",reason="queue_full"} 1'
            in body
        )

    asyncio.run(scenario())


def test_export_failure_is_redacted_counted_and_never_raised_by_request_finish() -> None:
    async def scenario() -> None:
        exporter = _RecordingExporter(failures=10)
        metrics = ServiceMetrics()
        clock = _FakeClock()
        telemetry = ServiceTelemetry(metrics, _enabled_config(), exporter, clock=clock)
        await telemetry.start()
        timer = telemetry.begin_request("/api/v1/runs/public")
        clock.elapsed += 0.01

        result = telemetry.finish_request(
            timer,
            outcome=Outcome.SUCCESS,
            status_code=200,
            response_bytes=100,
        )
        report = await telemetry.stop()
        body = metrics.snapshot().body

        assert result.log is EmissionResult.ACCEPTED
        assert result.trace is EmissionResult.ACCEPTED
        assert report.flushed is True
        assert (
            _metric_value(
                body,
                b'signalattice_telemetry_export_attempts_total{channel="log",outcome="failure"}',
            )
            == 1
        )
        assert (
            _metric_value(
                body,
                b'signalattice_telemetry_export_attempts_total{channel="trace",outcome="failure"}',
            )
            == 1
        )
        assert b"REMOTE-CANARY" not in body

    asyncio.run(scenario())


def test_exporter_originated_cancellation_is_contained_counted_and_worker_survives() -> None:
    class CancellingExporter:
        def __init__(self) -> None:
            self.calls = 0
            self.first_attempted = asyncio.Event()

        async def export(
            self,
            _channel: TelemetryChannel,
            _records: tuple[bytes, ...],
        ) -> None:
            self.calls += 1
            if self.calls == 1:
                self.first_attempted.set()
                raise asyncio.CancelledError

    async def scenario() -> None:
        exporter = CancellingExporter()
        metrics = ServiceMetrics()
        telemetry = ServiceTelemetry(
            metrics,
            _enabled_config(batch_size=1),
            exporter,
        )
        record = lifecycle_record(LifecycleState.READY, occurred_at_unix_ms=1)
        await telemetry.start()
        assert telemetry.emit(record) is EmissionResult.ACCEPTED
        await exporter.first_attempted.wait()
        await asyncio.sleep(0)
        assert telemetry.emit(record) is EmissionResult.ACCEPTED

        report = await telemetry.stop()
        body = metrics.snapshot().body

        assert report.flushed is True
        assert report.dropped_records == 0
        assert report.indeterminate_records == 0
        assert report.workers_terminated is True
        assert exporter.calls == 2
        assert (
            _metric_value(
                body,
                b'signalattice_telemetry_export_attempts_total{channel="log",outcome="failure"}',
            )
            == 1
        )
        assert (
            _metric_value(
                body,
                b'signalattice_telemetry_export_attempts_total{channel="log",outcome="success"}',
            )
            == 1
        )
        assert (
            _metric_value(
                body,
                b'signalattice_telemetry_dropped_records_total{channel="log",reason="export_failed"}',
            )
            == 1
        )

    asyncio.run(scenario())


def test_export_retry_count_is_bounded_and_success_stops_retrying() -> None:
    async def scenario() -> None:
        exporter = _RecordingExporter(failures=1)
        metrics = ServiceMetrics()
        telemetry = ServiceTelemetry(
            metrics,
            _enabled_config(max_retries=1),
            exporter,
        )
        await telemetry.start()
        telemetry.emit(lifecycle_record(LifecycleState.READY, occurred_at_unix_ms=1))
        report = await telemetry.stop()
        body = metrics.snapshot().body

        assert report.flushed is True
        assert len(exporter.calls) == 2
        assert (
            _metric_value(
                body,
                b'signalattice_telemetry_export_attempts_total{channel="log",outcome="failure"}',
            )
            == 1
        )
        assert (
            _metric_value(
                body,
                b'signalattice_telemetry_export_attempts_total{channel="log",outcome="success"}',
            )
            == 1
        )

    asyncio.run(scenario())


def test_export_timeout_is_bounded_and_accounted_without_error_text() -> None:
    class HangingExporter:
        async def export(
            self,
            _channel: TelemetryChannel,
            _records: tuple[bytes, ...],
        ) -> None:
            await asyncio.Event().wait()

    async def scenario() -> None:
        metrics = ServiceMetrics()
        telemetry = ServiceTelemetry(
            metrics,
            _enabled_config(timeout_seconds=0.01, shutdown_timeout_seconds=0.5),
            HangingExporter(),
        )
        await telemetry.start()
        telemetry.emit(lifecycle_record(LifecycleState.READY, occurred_at_unix_ms=1))
        report = await telemetry.stop()
        body = metrics.snapshot().body

        assert report.flushed is True
        assert (
            _metric_value(
                body,
                b'signalattice_telemetry_export_attempts_total{channel="log",outcome="timeout"}',
            )
            == 1
        )
        assert (
            _metric_value(
                body,
                b'signalattice_telemetry_dropped_records_total{channel="log",reason="export_failed"}',
            )
            == 1
        )

    asyncio.run(scenario())


def test_shutdown_timeout_cancels_export_and_reports_in_flight_drop() -> None:
    class HangingExporter:
        def __init__(self) -> None:
            self.entered = asyncio.Event()

        async def export(
            self,
            _channel: TelemetryChannel,
            _records: tuple[bytes, ...],
        ) -> None:
            self.entered.set()
            await asyncio.Event().wait()

    async def scenario() -> None:
        exporter = HangingExporter()
        metrics = ServiceMetrics()
        telemetry = ServiceTelemetry(
            metrics,
            _enabled_config(timeout_seconds=1.0, shutdown_timeout_seconds=0.01),
            exporter,
        )
        await telemetry.start()
        telemetry.emit(lifecycle_record(LifecycleState.READY, occurred_at_unix_ms=1))
        await exporter.entered.wait()
        telemetry.emit(lifecycle_record(LifecycleState.READY, occurred_at_unix_ms=2))
        telemetry.emit(lifecycle_record(LifecycleState.READY, occurred_at_unix_ms=3))
        report = await telemetry.stop()
        body = metrics.snapshot().body

        assert report.flushed is False
        assert report.dropped_records == 2
        assert report.indeterminate_records == 1
        assert report.workers_terminated is True
        assert telemetry.state is RuntimeState.CLOSED
        assert (
            _metric_value(
                body,
                b'signalattice_telemetry_dropped_records_total{channel="log",reason="shutdown_timeout"}',
            )
            == 2
        )
        assert (
            _metric_value(
                body,
                b'signalattice_telemetry_delivery_indeterminate_records_total{channel="log"}',
            )
            == 1
        )

    asyncio.run(scenario())


def test_cancellation_resistant_worker_has_an_independent_termination_deadline() -> None:
    class ResistsFirstCancellationExporter:
        def __init__(self) -> None:
            self.entered = asyncio.Event()
            self.cancellations = 0

        async def export(
            self,
            _channel: TelemetryChannel,
            _records: tuple[bytes, ...],
        ) -> None:
            self.entered.set()
            while True:
                try:
                    await asyncio.Event().wait()
                except asyncio.CancelledError:
                    self.cancellations += 1
                    if self.cancellations == 1:
                        continue
                    raise

    async def scenario() -> None:
        exporter = ResistsFirstCancellationExporter()
        metrics = ServiceMetrics()
        telemetry = ServiceTelemetry(
            metrics,
            _enabled_config(
                timeout_seconds=1.0,
                shutdown_timeout_seconds=0.01,
                worker_termination_timeout_seconds=0.01,
            ),
            exporter,
        )
        await telemetry.start()
        telemetry.emit(lifecycle_record(LifecycleState.READY, occurred_at_unix_ms=1))
        await exporter.entered.wait()

        report = await asyncio.wait_for(telemetry.stop(), timeout=0.25)
        await asyncio.sleep(0)
        body = metrics.snapshot().body

        assert report.flushed is False
        assert report.dropped_records == 0
        assert report.indeterminate_records == 1
        assert report.workers_terminated is False
        assert telemetry.state is RuntimeState.STOPPING
        assert telemetry.lifecycle_phase is LifecyclePhase.TERMINATION_DEGRADED
        assert exporter.cancellations == 2
        assert not {
            task.get_name()
            for task in asyncio.all_tasks()
            if task is not asyncio.current_task()
            and not task.done()
            and task.get_name().startswith("signalattice-telemetry-")
        }
        assert b'signalattice_telemetry_worker_termination_failures_total{channel="log"} 1' in body
        assert (
            b'signalattice_telemetry_delivery_indeterminate_records_total{channel="log"} 1' in body
        )
        lifecycle = [
            json.loads(record)["attributes"]["lifecycle"]
            for record in telemetry.local_snapshot().records
            if json.loads(record)["event"] == "service_lifecycle"
        ]
        assert lifecycle[-2:] == ["stopping", "termination_timeout"]
        assert "stopped" not in lifecycle

    asyncio.run(scenario())


def test_cancellation_suppressing_exporter_remains_indeterminate_until_it_cooperates() -> None:
    class StubbornUntilReleasedExporter:
        def __init__(self) -> None:
            self.entered = asyncio.Event()
            self.release = asyncio.Event()
            self.completed = asyncio.Event()
            self.cancellations = 0

        async def export(
            self,
            _channel: TelemetryChannel,
            _records: tuple[bytes, ...],
        ) -> None:
            self.entered.set()
            while not self.release.is_set():
                try:
                    await self.release.wait()
                except asyncio.CancelledError:
                    self.cancellations += 1
            self.completed.set()

    async def scenario() -> None:
        exporter = StubbornUntilReleasedExporter()
        metrics = ServiceMetrics()
        telemetry = ServiceTelemetry(
            metrics,
            _enabled_config(
                timeout_seconds=1.0,
                shutdown_timeout_seconds=0.01,
                worker_termination_timeout_seconds=0.01,
            ),
            exporter,
        )
        await telemetry.start()
        telemetry.emit(lifecycle_record(LifecycleState.READY, occurred_at_unix_ms=1))
        await exporter.entered.wait()

        try:
            report = await asyncio.wait_for(telemetry.stop(), timeout=0.25)
            at_report = metrics.snapshot().body
        finally:
            # Never leave a deliberately cancellation-suppressing test double
            # alive if an assertion or bounded stop unexpectedly fails.
            exporter.release.set()
        await asyncio.wait_for(exporter.completed.wait(), timeout=0.25)
        for _ in range(3):
            await asyncio.sleep(0)

        assert report.flushed is False
        assert report.dropped_records == 0
        assert report.indeterminate_records == 1
        assert report.workers_terminated is False
        assert telemetry.state is RuntimeState.STOPPING
        assert telemetry.lifecycle_phase is LifecyclePhase.TERMINATION_DEGRADED
        assert exporter.cancellations >= 1
        assert (
            b'signalattice_telemetry_export_attempts_total{channel="log",outcome="success"} 0'
            in at_report
        )
        assert (
            b'signalattice_telemetry_dropped_records_total{channel="log",reason="shutdown_timeout"} 0'
            in at_report
        )
        assert (
            b'signalattice_telemetry_delivery_indeterminate_records_total{channel="log"} 1'
            in at_report
        )
        assert not {
            task.get_name()
            for task in asyncio.all_tasks()
            if task is not asyncio.current_task()
            and not task.done()
            and task.get_name().startswith("signalattice-telemetry-")
        }
        assert (
            b'signalattice_telemetry_export_attempts_total{channel="log",outcome="success"} 1'
            in metrics.snapshot().body
        )

    asyncio.run(scenario())


def test_structured_perimeter_rejection_records_reason_without_raw_path() -> None:
    metrics = ServiceMetrics()
    clock = _FakeClock()
    telemetry = ServiceTelemetry(metrics, clock=clock)
    timer = telemetry.begin_request("/private/canary?secret=yes")

    result = telemetry.finish_request(
        timer,
        outcome=Outcome.REJECTED,
        status_code=413,
        response_bytes=256,
        rejection=RejectionReason.BODY_FORBIDDEN,
    )

    assert result.log is EmissionResult.LOCAL_ONLY
    assert timer.route is RouteTemplate.UNMATCHED
    body = metrics.snapshot().body
    assert b"private" not in body
    assert b"secret" not in body
    assert b'signalattice_admission_rejections_total{reason="body_forbidden"} 1' in body
