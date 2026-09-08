"""Non-blocking telemetry facade and bounded asynchronous export runtime.

Request tasks synchronously update fixed in-memory metrics and retain canonical
JSON log/span records in a 512-record process-local ring. Optional external
export uses ``put_nowait``; request tasks never call an exporter, wait for queue
space, retry, perform network I/O, or surface exporter failures. Two independent
queues are drained by background tasks with per-attempt timeouts, bounded
immediate retries, explicit drop accounting, and a bounded shutdown flush.
"""

from __future__ import annotations

import asyncio
import math
import threading
import time
from collections import deque
from dataclasses import dataclass
from enum import StrEnum
from typing import Final, Protocol

from quant_platform.service.exporter import ExporterConfig, TelemetryExporter
from quant_platform.service.metrics import ServiceMetrics
from quant_platform.service.telemetry_contracts import (
    MAX_TELEMETRY_RECORD_BYTES,
    DropReason,
    ExportOutcome,
    LifecycleState,
    Outcome,
    RejectionReason,
    RequestObservation,
    RouteTemplate,
    TelemetryChannel,
    TelemetryContractError,
    TelemetryRecord,
    classify_route,
    lifecycle_record,
    operation_for_route,
    records_for_request,
)

LOCAL_TELEMETRY_RECORD_CAPACITY: Final = 512
MAX_LOCAL_TELEMETRY_RECORDS: Final = 4_096


class TelemetryRuntimeError(RuntimeError):
    """The telemetry lifecycle or injected clock violates its contract."""

    code = "telemetry_runtime_error"


class Clock(Protocol):
    """Injectable clock for deterministic request-duration evidence."""

    def time(self) -> float:
        """Return finite Unix seconds."""

    def monotonic(self) -> float:
        """Return finite process-monotonic seconds."""


class SystemClock:
    """Production clock backed by Python's system and monotonic clocks."""

    @staticmethod
    def time() -> float:
        return time.time()

    @staticmethod
    def monotonic() -> float:
        return time.monotonic()


class EmissionResult(StrEnum):
    """Machine-readable result of local retention and optional export enqueue."""

    ACCEPTED = "accepted"
    LOCAL_ONLY = "local_only"
    LOCAL_SINK_FAILED = "local_sink_failed"
    INVALID_RECORD = "invalid_record"
    NOT_RUNNING = "not_running"
    QUEUE_FULL = "queue_full"


class RuntimeState(StrEnum):
    """Explicit lifecycle for the optional exporter workers."""

    DISABLED = "disabled"
    CREATED = "created"
    RUNNING = "running"
    STOPPING = "stopping"
    CLOSED = "closed"


class LifecyclePhase(StrEnum):
    """Process lifecycle independent of whether external export is enabled."""

    CREATED = "created"
    RUNNING = "running"
    DRAINING = "draining"
    STOPPING = "stopping"
    CLOSED = "closed"
    TERMINATION_DEGRADED = "termination_degraded"


@dataclass(frozen=True, slots=True)
class RequestTimer:
    """Privacy-safe timer token; the raw path is discarded at creation."""

    route: RouteTemplate
    started_at_monotonic: float
    occurred_at_unix_ms: int

    def __post_init__(self) -> None:
        if type(self.route) is not RouteTemplate:
            raise TelemetryRuntimeError("timer route must be an exact RouteTemplate")
        if (
            type(self.started_at_monotonic) is not float
            or not math.isfinite(self.started_at_monotonic)
            or self.started_at_monotonic < 0.0
        ):
            raise TelemetryRuntimeError("timer monotonic value is invalid")
        if type(self.occurred_at_unix_ms) is not int or self.occurred_at_unix_ms < 0:
            raise TelemetryRuntimeError("timer wall-clock value is invalid")


@dataclass(frozen=True, slots=True)
class ObservationResult:
    """Non-blocking enqueue outcomes after metrics were recorded."""

    log: EmissionResult
    trace: EmissionResult


@dataclass(frozen=True, slots=True)
class ShutdownReport:
    """Bounded shutdown evidence without endpoint or failure details."""

    flushed: bool
    dropped_records: int
    indeterminate_records: int
    workers_terminated: bool

    def __post_init__(self) -> None:
        if type(self.flushed) is not bool:
            raise TelemetryRuntimeError("flushed must be an exact boolean")
        if type(self.dropped_records) is not int or self.dropped_records < 0:
            raise TelemetryRuntimeError("dropped_records must be a nonnegative integer")
        if type(self.indeterminate_records) is not int or self.indeterminate_records < 0:
            raise TelemetryRuntimeError("indeterminate_records must be a nonnegative integer")
        if type(self.workers_terminated) is not bool:
            raise TelemetryRuntimeError("workers_terminated must be an exact boolean")


@dataclass(frozen=True, slots=True)
class LocalTelemetrySnapshot:
    """Immutable bounded snapshot of canonical process-local JSON records.

    The sink is an operational diagnostic ring, not durable business evidence.
    It keeps at most ``capacity`` records, each already constrained by the
    closed telemetry schema and 8-KiB serialization ceiling.
    """

    records: tuple[bytes, ...]
    capacity: int
    evicted_records: int

    def __post_init__(self) -> None:
        if type(self.capacity) is not int or not 1 <= self.capacity <= MAX_LOCAL_TELEMETRY_RECORDS:
            raise TelemetryRuntimeError("local telemetry capacity is invalid")
        if type(self.records) is not tuple or len(self.records) > self.capacity:
            raise TelemetryRuntimeError("local telemetry snapshot exceeds its record capacity")
        if type(self.evicted_records) is not int or self.evicted_records < 0:
            raise TelemetryRuntimeError("local telemetry eviction count is invalid")
        for record in self.records:
            if (
                type(record) is not bytes
                or not 2 <= len(record) <= MAX_TELEMETRY_RECORD_BYTES
                or not record.endswith(b"\n")
                or any(byte < 0x20 for byte in record[:-1])
            ):
                raise TelemetryRuntimeError("local telemetry snapshot contains an invalid record")

    @property
    def total_bytes(self) -> int:
        """Return retained bytes under ``capacity * 8 KiB`` by construction."""

        return sum(len(record) for record in self.records)


class _BoundedLocalTelemetrySink:
    """Thread-safe fixed-record ring storing only canonical telemetry bytes."""

    def __init__(self, capacity: int) -> None:
        if type(capacity) is not int or not 1 <= capacity <= MAX_LOCAL_TELEMETRY_RECORDS:
            raise TelemetryRuntimeError(
                f"local_record_capacity must be an integer in [1, {MAX_LOCAL_TELEMETRY_RECORDS}]"
            )
        self._capacity = capacity
        self._records: deque[tuple[TelemetryChannel, bytes]] = deque()
        self._evicted_records = 0
        self._lock = threading.Lock()

    def retain(self, channel: TelemetryChannel, record: bytes) -> TelemetryChannel | None:
        """Retain one canonical record and return the channel evicted, if any."""

        if type(channel) is not TelemetryChannel:
            raise TelemetryRuntimeError("local sink channel must be an exact TelemetryChannel")
        if (
            type(record) is not bytes
            or not 2 <= len(record) <= MAX_TELEMETRY_RECORD_BYTES
            or not record.endswith(b"\n")
            or any(byte < 0x20 for byte in record[:-1])
        ):
            raise TelemetryRuntimeError("local sink accepts only bounded canonical JSON lines")
        with self._lock:
            evicted_channel = None
            if len(self._records) == self._capacity:
                evicted_channel, _ = self._records.popleft()
                self._evicted_records += 1
            self._records.append((channel, record))
            return evicted_channel

    def snapshot(self) -> LocalTelemetrySnapshot:
        """Copy the bounded ring without retaining channel keys separately."""

        with self._lock:
            records = tuple(record for _, record in self._records)
            evicted_records = self._evicted_records
        return LocalTelemetrySnapshot(
            records=records,
            capacity=self._capacity,
            evicted_records=evicted_records,
        )


class ServiceTelemetry:
    """Own fixed metrics, local records, and optional exporter worker tasks."""

    def __init__(
        self,
        metrics: ServiceMetrics,
        config: ExporterConfig | None = None,
        exporter: TelemetryExporter | None = None,
        *,
        clock: Clock | None = None,
        local_record_capacity: int = LOCAL_TELEMETRY_RECORD_CAPACITY,
    ) -> None:
        if type(metrics) is not ServiceMetrics:
            raise TelemetryRuntimeError("metrics must be an exact ServiceMetrics")
        resolved_config = ExporterConfig() if config is None else config
        if type(resolved_config) is not ExporterConfig:
            raise TelemetryRuntimeError("config must be an exact ExporterConfig")
        if resolved_config.enabled != (exporter is not None):
            raise TelemetryRuntimeError(
                "an exporter is required exactly when external export is enabled"
            )
        resolved_clock = SystemClock() if clock is None else clock
        if not callable(getattr(resolved_clock, "time", None)) or not callable(
            getattr(resolved_clock, "monotonic", None)
        ):
            raise TelemetryRuntimeError("clock must implement time and monotonic")

        self._metrics = metrics
        self._config = resolved_config
        self._exporter = exporter
        self._clock = resolved_clock
        self._state = RuntimeState.CREATED if resolved_config.enabled else RuntimeState.DISABLED
        self._lifecycle_phase = LifecyclePhase.CREATED
        self._local_sink = _BoundedLocalTelemetrySink(local_record_capacity)
        self._queues = {
            channel: asyncio.Queue[bytes](maxsize=resolved_config.queue_capacity)
            for channel in TelemetryChannel
        }
        self._workers: dict[TelemetryChannel, asyncio.Task[None]] = {}
        self._in_flight = dict.fromkeys(TelemetryChannel, 0)
        self._shutdown_cancellations: set[asyncio.Task[None]] = set()

    @property
    def state(self) -> RuntimeState:
        """Return the current explicit exporter lifecycle state."""

        return self._state

    @property
    def metrics(self) -> ServiceMetrics:
        """Return the exact registry used by request and exporter accounting."""

        return self._metrics

    @property
    def lifecycle_phase(self) -> LifecyclePhase:
        """Return the service lifecycle independently of exporter configuration."""

        return self._lifecycle_phase

    def local_snapshot(self) -> LocalTelemetrySnapshot:
        """Return the bounded process-local canonical log/span ring."""

        return self._local_sink.snapshot()

    def begin_request(self, raw_path: object) -> RequestTimer:
        """Start a privacy-safe request timer and immediately discard ``raw_path``."""

        wall = self._clock.time()
        monotonic = self._clock.monotonic()
        if (
            type(wall) is not float
            or not math.isfinite(wall)
            or not 0.0 <= wall <= 32_503_680_000.0
            or type(monotonic) is not float
            or not math.isfinite(monotonic)
            or monotonic < 0.0
        ):
            raise TelemetryRuntimeError("clock produced an invalid request timestamp")
        return RequestTimer(
            route=classify_route(raw_path),
            started_at_monotonic=monotonic,
            occurred_at_unix_ms=int(wall * 1_000.0),
        )

    def finish_request(
        self,
        timer: RequestTimer,
        *,
        outcome: Outcome,
        status_code: int,
        response_bytes: int,
        rejection: RejectionReason | None = None,
    ) -> ObservationResult:
        """Record metrics and enqueue log/span records without awaiting export.

        This method performs no exporter I/O.  Contract errors identify an
        internal adapter bug and are explicit; queue saturation and exporter
        lifecycle state are returned as data and never fail the request.
        """

        if type(timer) is not RequestTimer:
            raise TelemetryRuntimeError("timer must be an exact RequestTimer")
        finished = self._clock.monotonic()
        if (
            type(finished) is not float
            or not math.isfinite(finished)
            or finished < timer.started_at_monotonic
        ):
            raise TelemetryRuntimeError("monotonic clock regressed during a request")
        observation = RequestObservation(
            route=timer.route,
            operation=operation_for_route(timer.route),
            outcome=outcome,
            status_code=status_code,
            duration_seconds=finished - timer.started_at_monotonic,
            response_bytes=response_bytes,
            rejection=rejection,
        )
        self._metrics.record_request(observation)
        log_record, trace_record = records_for_request(
            observation,
            occurred_at_unix_ms=timer.occurred_at_unix_ms,
        )
        return ObservationResult(
            log=self.emit(log_record),
            trace=self.emit(trace_record),
        )

    def emit(self, record: TelemetryRecord) -> EmissionResult:
        """Retain locally, optionally enqueue for export, and never wait for I/O."""

        if type(record) is not TelemetryRecord:
            for channel in TelemetryChannel:
                self._metrics.record_drop(channel, DropReason.INVALID_RECORD)
            return EmissionResult.INVALID_RECORD
        try:
            encoded = record.json_line()
        except TelemetryContractError:
            if type(record.channel) is TelemetryChannel:
                self._metrics.record_drop(record.channel, DropReason.INVALID_RECORD)
            else:
                for channel in TelemetryChannel:
                    self._metrics.record_drop(channel, DropReason.INVALID_RECORD)
            return EmissionResult.INVALID_RECORD
        local_retained = self._retain_local(record.channel, encoded)
        if self._state is RuntimeState.DISABLED:
            return EmissionResult.LOCAL_ONLY if local_retained else EmissionResult.LOCAL_SINK_FAILED
        if self._state is not RuntimeState.RUNNING:
            self._metrics.record_drop(record.channel, DropReason.NOT_RUNNING)
            return (
                EmissionResult.NOT_RUNNING if local_retained else EmissionResult.LOCAL_SINK_FAILED
            )
        queue = self._queues[record.channel]
        try:
            queue.put_nowait(encoded)
        except asyncio.QueueFull:
            self._metrics.record_drop(record.channel, DropReason.QUEUE_FULL)
            return EmissionResult.QUEUE_FULL if local_retained else EmissionResult.LOCAL_SINK_FAILED
        self._metrics.set_queue_depth(record.channel, queue.qsize())
        return EmissionResult.ACCEPTED if local_retained else EmissionResult.LOCAL_SINK_FAILED

    def _retain_local(self, channel: TelemetryChannel, encoded: bytes) -> bool:
        try:
            evicted_channel = self._local_sink.retain(channel, encoded)
        except Exception:
            # Local retention is an availability boundary. Exception identity
            # and text are discarded because an injected allocator/runtime
            # failure may itself contain sensitive process context.
            self._metrics.record_drop(channel, DropReason.LOCAL_SINK_FAILURE)
            return False
        if evicted_channel is not None:
            self._metrics.record_drop(evicted_channel, DropReason.LOCAL_SINK_CAPACITY)
        return True

    def _wall_timestamp_milliseconds(self) -> int:
        wall = self._clock.time()
        if (
            type(wall) is not float
            or not math.isfinite(wall)
            or not 0.0 <= wall <= 32_503_680_000.0
        ):
            raise TelemetryRuntimeError("clock produced an invalid lifecycle timestamp")
        return int(wall * 1_000.0)

    def _emit_lifecycle(self, state: LifecycleState, *, local_only: bool = False) -> None:
        try:
            record = lifecycle_record(
                state,
                occurred_at_unix_ms=self._wall_timestamp_milliseconds(),
            )
        except (TelemetryContractError, TelemetryRuntimeError):
            self._metrics.record_drop(TelemetryChannel.LOG, DropReason.INVALID_RECORD)
            return
        if local_only:
            try:
                encoded = record.json_line()
            except TelemetryContractError:
                self._metrics.record_drop(TelemetryChannel.LOG, DropReason.INVALID_RECORD)
                return
            self._retain_local(TelemetryChannel.LOG, encoded)
            return
        self.emit(record)

    async def start(self) -> None:
        """Record startup and start exporter tasks when export is configured."""

        if self._lifecycle_phase is not LifecyclePhase.CREATED:
            raise TelemetryRuntimeError("telemetry runtime can be started exactly once")
        if self._state is RuntimeState.CREATED:
            self._state = RuntimeState.RUNNING
            for channel in TelemetryChannel:
                self._workers[channel] = asyncio.create_task(
                    self._worker(channel),
                    name=f"signalattice-telemetry-{channel.value}",
                )
        self._lifecycle_phase = LifecyclePhase.RUNNING
        self._emit_lifecycle(LifecycleState.STARTING, local_only=True)
        self._emit_lifecycle(LifecycleState.READY, local_only=True)

    def begin_draining(self) -> None:
        """Enter the single explicit draining phase before admission closes."""

        if self._lifecycle_phase is not LifecyclePhase.RUNNING:
            raise TelemetryRuntimeError("only a running service can begin draining")
        self._lifecycle_phase = LifecyclePhase.DRAINING
        self._emit_lifecycle(LifecycleState.DRAINING, local_only=True)

    async def stop(self) -> ShutdownReport:
        """Flush queued records within the configured shutdown ceiling.

        Queued records are counted as definite drops on timeout. In-flight
        records are reported separately as delivery-indeterminate because an
        injected transport may suppress cancellation and complete after this
        bounded report. Worker cancellation never propagates to the caller,
        and no endpoint or exception text is retained.
        """

        if self._lifecycle_phase is LifecyclePhase.RUNNING:
            self.begin_draining()
        elif self._lifecycle_phase is not LifecyclePhase.DRAINING:
            raise TelemetryRuntimeError("only a running or draining service can stop")
        self._emit_lifecycle(LifecycleState.STOPPING, local_only=True)
        self._lifecycle_phase = LifecyclePhase.STOPPING
        if self._state is RuntimeState.DISABLED:
            self._lifecycle_phase = LifecyclePhase.CLOSED
            self._emit_lifecycle(LifecycleState.STOPPED, local_only=True)
            return ShutdownReport(
                flushed=True,
                dropped_records=0,
                indeterminate_records=0,
                workers_terminated=True,
            )
        if self._state is not RuntimeState.RUNNING:
            raise TelemetryRuntimeError("only a running exporter can stop")
        self._state = RuntimeState.STOPPING
        flushed = True
        try:
            async with asyncio.timeout(self._config.shutdown_timeout_seconds):
                await asyncio.gather(*(queue.join() for queue in self._queues.values()))
        except TimeoutError:
            flushed = False

        indeterminate_by_channel = {
            channel: self._in_flight[channel] if not flushed else 0 for channel in TelemetryChannel
        }
        if not flushed:
            for channel, count in indeterminate_by_channel.items():
                if count:
                    self._metrics.record_delivery_indeterminate(
                        channel,
                        count=count,
                    )
        workers = dict(self._workers)
        for task in workers.values():
            self._shutdown_cancellations.add(task)
            task.cancel()
        done, pending = await asyncio.wait(
            tuple(workers.values()),
            timeout=self._config.worker_termination_timeout_seconds,
        )
        for task in done:
            self._consume_worker_result(task)
        for channel, task in workers.items():
            if task not in pending:
                continue
            self._metrics.record_worker_termination_failure(channel)
            task.cancel()
            task.add_done_callback(self._consume_worker_result)
        dropped_by_channel = dict.fromkeys(TelemetryChannel, 0)
        for channel in TelemetryChannel:
            queue = self._queues[channel]
            while True:
                try:
                    queue.get_nowait()
                except asyncio.QueueEmpty:
                    break
                else:
                    queue.task_done()
                    dropped_by_channel[channel] += 1
            if dropped_by_channel[channel]:
                self._metrics.record_drop(
                    channel,
                    DropReason.SHUTDOWN_TIMEOUT,
                    count=dropped_by_channel[channel],
                )
            self._metrics.set_queue_depth(channel, 0)
        self._workers.clear()
        workers_terminated = not pending
        if workers_terminated:
            self._state = RuntimeState.CLOSED
            self._lifecycle_phase = LifecyclePhase.CLOSED
            self._emit_lifecycle(LifecycleState.STOPPED, local_only=True)
        else:
            self._lifecycle_phase = LifecyclePhase.TERMINATION_DEGRADED
            self._emit_lifecycle(LifecycleState.TERMINATION_TIMEOUT, local_only=True)
        return ShutdownReport(
            flushed=flushed,
            dropped_records=sum(dropped_by_channel.values()),
            indeterminate_records=sum(indeterminate_by_channel.values()),
            workers_terminated=workers_terminated,
        )

    def _consume_worker_result(self, task: asyncio.Task[None]) -> None:
        """Retrieve a worker result without retaining exception details."""

        self._shutdown_cancellations.discard(task)
        try:
            task.result()
        except asyncio.CancelledError:
            return
        except Exception:
            return

    async def _worker(self, channel: TelemetryChannel) -> None:
        queue = self._queues[channel]
        while True:
            first = await queue.get()
            batch = [first]
            while len(batch) < self._config.batch_size:
                try:
                    batch.append(queue.get_nowait())
                except asyncio.QueueEmpty:
                    break
            self._in_flight[channel] = len(batch)
            self._metrics.set_queue_depth(channel, queue.qsize())
            try:
                await self._deliver(channel, tuple(batch))
            finally:
                self._in_flight[channel] = 0
                for _ in batch:
                    queue.task_done()
            task = asyncio.current_task()
            if task is not None and task in self._shutdown_cancellations:
                return

    async def _deliver(
        self,
        channel: TelemetryChannel,
        records: tuple[bytes, ...],
    ) -> None:
        exporter = self._exporter
        if exporter is None:
            self._metrics.record_drop(
                channel,
                DropReason.EXPORT_FAILED,
                count=len(records),
            )
            return
        delivered = False
        for _ in range(self._config.max_retries + 1):
            try:
                async with asyncio.timeout(self._config.timeout_seconds):
                    await exporter.export(channel, records)
            except asyncio.CancelledError:
                task = asyncio.current_task()
                if task is not None and task in self._shutdown_cancellations:
                    raise
                # An injected exporter may raise or self-schedule cancellation.
                # It is not allowed to kill the channel worker silently. Clear
                # only cancellation not issued by this runtime and account the
                # attempt through the same closed failure outcome.
                if task is not None:
                    while task.cancelling():
                        task.uncancel()
                self._metrics.record_export_attempt(channel, ExportOutcome.FAILURE)
            except TimeoutError:
                self._metrics.record_export_attempt(channel, ExportOutcome.TIMEOUT)
            except Exception:
                # Exporters are outside the request path and are an explicit
                # failure boundary.  Error text may contain endpoint or remote
                # content, so only this bounded outcome is retained.
                self._metrics.record_export_attempt(channel, ExportOutcome.FAILURE)
            else:
                self._metrics.record_export_attempt(channel, ExportOutcome.SUCCESS)
                delivered = True
                break
        if not delivered:
            self._metrics.record_drop(
                channel,
                DropReason.EXPORT_FAILED,
                count=len(records),
            )
