"""Fixed-cardinality Prometheus metrics for the local read service.

The registry has no generic ``register`` or arbitrary-label API.  Every series
is allocated from closed telemetry enums at construction, so hostile route and
business identifiers cannot increase cardinality.  Exposition is rendered from
an immutable snapshot under a hard 256-KiB ceiling and reports its exact series
count for service and benchmark assertions.
"""

from __future__ import annotations

import math
import threading
from dataclasses import dataclass
from typing import Final

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
)

MAX_METRIC_SERIES: Final = 599
MAX_METRICS_BYTES: Final = 256 * 1024
METRICS_CONTENT_TYPE: Final = "text/plain; version=0.0.4; charset=utf-8"

_DURATION_BUCKETS: Final = (0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.0)
_RESPONSE_BUCKETS: Final = (
    256.0,
    1_024.0,
    4_096.0,
    16_384.0,
    65_536.0,
    262_144.0,
    1_048_576.0,
    2_097_152.0,
)


class MetricsBoundError(RuntimeError):
    """The fixed metric schema violated its audited size or cardinality bound."""

    code = "metrics_bound_violation"


@dataclass(frozen=True, slots=True)
class MetricsSnapshot:
    """Immutable bounded Prometheus exposition and audited series count."""

    body: bytes
    series_count: int
    content_type: str = METRICS_CONTENT_TYPE

    def __post_init__(self) -> None:
        if type(self.body) is not bytes or not 1 <= len(self.body) <= MAX_METRICS_BYTES:
            raise MetricsBoundError("metrics exposition exceeds the byte ceiling")
        if type(self.series_count) is not int or not 1 <= self.series_count <= MAX_METRIC_SERIES:
            raise MetricsBoundError("metrics exposition exceeds the series ceiling")
        if self.content_type != METRICS_CONTENT_TYPE:
            raise MetricsBoundError("metrics content type changed unexpectedly")


@dataclass(slots=True)
class _Histogram:
    buckets: tuple[float, ...]
    bucket_counts: list[int]
    count: int = 0
    total: float = 0.0

    @classmethod
    def empty(cls, buckets: tuple[float, ...]) -> _Histogram:
        return cls(buckets=buckets, bucket_counts=[0 for _ in buckets])

    def observe(self, value: float) -> None:
        self.count += 1
        self.total += value
        for index, upper in enumerate(self.buckets):
            if value <= upper:
                self.bucket_counts[index] += 1

    def clone(self) -> _Histogram:
        return _Histogram(
            buckets=self.buckets,
            bucket_counts=list(self.bucket_counts),
            count=self.count,
            total=self.total,
        )


def _exact_enum(value: object, expected: type[object], name: str) -> None:
    if type(value) is not expected:
        raise TelemetryContractError(f"{name} must use its exact telemetry enum")


def _positive_count(value: int, name: str) -> int:
    if type(value) is not int or not 1 <= value <= 1_000_000_000:
        raise TelemetryContractError(f"{name} must be an integer in [1, 1000000000]")
    return value


def _nonnegative_count(value: int, name: str) -> int:
    if type(value) is not int or not 0 <= value <= 1_000_000_000:
        raise TelemetryContractError(f"{name} must be an integer in [0, 1000000000]")
    return value


class ServiceMetrics:
    """Thread-safe, fixed-shape metrics owned by one service process.

    Mutation is constant time apart from two short fixed histogram scans.  The
    lock protects consistent snapshots; user-controlled strings are never
    stored, hashed, or compared inside the registry.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._requests = {(route, outcome): 0 for route in RouteTemplate for outcome in Outcome}
        self._duration = {operation: _Histogram.empty(_DURATION_BUCKETS) for operation in Operation}
        self._response_size = {
            operation: _Histogram.empty(_RESPONSE_BUCKETS) for operation in Operation
        }
        self._in_flight = dict.fromkeys(Operation, 0)
        self._rejections = dict.fromkeys(RejectionReason, 0)
        self._drops = {
            (channel, reason): 0 for channel in TelemetryChannel for reason in DropReason
        }
        self._exports = {
            (channel, outcome): 0 for channel in TelemetryChannel for outcome in ExportOutcome
        }
        self._queue_depth = dict.fromkeys(TelemetryChannel, 0)
        self._worker_termination_failures = dict.fromkeys(TelemetryChannel, 0)
        self._delivery_indeterminate = dict.fromkeys(TelemetryChannel, 0)

    def record_request(self, observation: RequestObservation) -> None:
        """Record one already-validated bounded request observation."""

        if type(observation) is not RequestObservation:
            raise TelemetryContractError("observation must be an exact RequestObservation")
        with self._lock:
            self._requests[(observation.route, observation.outcome)] += 1
            self._duration[observation.operation].observe(observation.duration_seconds)
            self._response_size[observation.operation].observe(float(observation.response_bytes))
            if observation.rejection is not None:
                self._rejections[observation.rejection] += 1

    def set_in_flight(self, operation: Operation, count: int) -> None:
        """Set an operation's bounded process-local concurrency gauge."""

        _exact_enum(operation, Operation, "operation")
        resolved = _nonnegative_count(count, "count")
        with self._lock:
            self._in_flight[operation] = resolved

    def adjust_in_flight(self, operation: Operation, delta: int) -> None:
        """Atomically adjust one operation's gauge and reject accounting bugs.

        Middleware calls this exactly once after admission and once during its
        corresponding release path.  The mutation is kept inside the registry
        lock so simultaneous request starts and finishes cannot lose updates.
        """

        _exact_enum(operation, Operation, "operation")
        if type(delta) is not int or delta not in {-1, 1}:
            raise TelemetryContractError("delta must be exactly -1 or 1")
        with self._lock:
            resolved = self._in_flight[operation] + delta
            if not 0 <= resolved <= 1_000_000_000:
                raise TelemetryContractError("in-flight accounting would leave its bound")
            self._in_flight[operation] = resolved

    def record_drop(
        self,
        channel: TelemetryChannel,
        reason: DropReason,
        *,
        count: int = 1,
    ) -> None:
        """Account for records intentionally discarded at a bounded boundary."""

        _exact_enum(channel, TelemetryChannel, "channel")
        _exact_enum(reason, DropReason, "reason")
        resolved = _positive_count(count, "count")
        with self._lock:
            self._drops[(channel, reason)] += resolved

    def record_export_attempt(
        self,
        channel: TelemetryChannel,
        outcome: ExportOutcome,
    ) -> None:
        """Account for one exporter attempt without retaining its endpoint/error."""

        _exact_enum(channel, TelemetryChannel, "channel")
        _exact_enum(outcome, ExportOutcome, "outcome")
        with self._lock:
            self._exports[(channel, outcome)] += 1

    def set_queue_depth(self, channel: TelemetryChannel, count: int) -> None:
        """Set one bounded telemetry channel's current queue depth."""

        _exact_enum(channel, TelemetryChannel, "channel")
        resolved = _nonnegative_count(count, "count")
        with self._lock:
            self._queue_depth[channel] = resolved

    def record_worker_termination_failure(self, channel: TelemetryChannel) -> None:
        """Account for a worker that missed its independent termination deadline."""

        _exact_enum(channel, TelemetryChannel, "channel")
        with self._lock:
            self._worker_termination_failures[channel] += 1

    def record_delivery_indeterminate(
        self,
        channel: TelemetryChannel,
        *,
        count: int,
    ) -> None:
        """Account for records whose remote delivery is unknown at shutdown."""

        _exact_enum(channel, TelemetryChannel, "channel")
        resolved = _positive_count(count, "count")
        with self._lock:
            self._delivery_indeterminate[channel] += resolved

    def snapshot(self) -> MetricsSnapshot:
        """Return deterministic fixed-cardinality Prometheus text.

        Serialization operates on copies, so exporters and metrics scrapes do
        not hold the mutation lock.  A schema programming error fails closed
        rather than emitting truncated or partially ambiguous metrics.
        """

        with self._lock:
            requests = dict(self._requests)
            duration = {key: value.clone() for key, value in self._duration.items()}
            response_size = {key: value.clone() for key, value in self._response_size.items()}
            in_flight = dict(self._in_flight)
            rejections = dict(self._rejections)
            drops = dict(self._drops)
            exports = dict(self._exports)
            queue_depth = dict(self._queue_depth)
            worker_termination_failures = dict(self._worker_termination_failures)
            delivery_indeterminate = dict(self._delivery_indeterminate)

        lines: list[str] = []
        _header(
            lines,
            "signalattice_http_requests_total",
            "Completed bounded HTTP requests by reviewed route template and outcome.",
            "counter",
        )
        for route in RouteTemplate:
            for outcome in Outcome:
                _sample(
                    lines,
                    "signalattice_http_requests_total",
                    requests[(route, outcome)],
                    route=route.value,
                    outcome=outcome.value,
                )

        _header(
            lines,
            "signalattice_http_request_duration_seconds",
            "Request duration by bounded operation family.",
            "histogram",
        )
        for operation in Operation:
            _histogram_samples(
                lines,
                "signalattice_http_request_duration_seconds",
                duration[operation],
                operation=operation.value,
            )

        _header(
            lines,
            "signalattice_http_response_bytes",
            "Buffered response bytes by bounded operation family.",
            "histogram",
        )
        for operation in Operation:
            _histogram_samples(
                lines,
                "signalattice_http_response_bytes",
                response_size[operation],
                operation=operation.value,
            )

        _header(
            lines,
            "signalattice_http_in_flight",
            "Current admitted requests by bounded operation family.",
            "gauge",
        )
        for operation in Operation:
            _sample(
                lines,
                "signalattice_http_in_flight",
                in_flight[operation],
                operation=operation.value,
            )

        _header(
            lines,
            "signalattice_admission_rejections_total",
            "Structurally rejected requests by reviewed reason.",
            "counter",
        )
        for reason in RejectionReason:
            _sample(
                lines,
                "signalattice_admission_rejections_total",
                rejections[reason],
                reason=reason.value,
            )

        _header(
            lines,
            "signalattice_telemetry_dropped_records_total",
            "Telemetry records dropped at bounded non-blocking boundaries.",
            "counter",
        )
        for channel in TelemetryChannel:
            for drop_reason in DropReason:
                _sample(
                    lines,
                    "signalattice_telemetry_dropped_records_total",
                    drops[(channel, drop_reason)],
                    channel=channel.value,
                    reason=drop_reason.value,
                )

        _header(
            lines,
            "signalattice_telemetry_export_attempts_total",
            "Bounded telemetry export attempts by channel and outcome.",
            "counter",
        )
        for channel in TelemetryChannel:
            for export_outcome in ExportOutcome:
                _sample(
                    lines,
                    "signalattice_telemetry_export_attempts_total",
                    exports[(channel, export_outcome)],
                    channel=channel.value,
                    outcome=export_outcome.value,
                )

        _header(
            lines,
            "signalattice_telemetry_queue_depth",
            "Current records in each bounded telemetry queue.",
            "gauge",
        )
        for channel in TelemetryChannel:
            _sample(
                lines,
                "signalattice_telemetry_queue_depth",
                queue_depth[channel],
                channel=channel.value,
            )

        _header(
            lines,
            "signalattice_telemetry_worker_termination_failures_total",
            "Telemetry workers that exceeded the bounded termination deadline.",
            "counter",
        )
        for channel in TelemetryChannel:
            _sample(
                lines,
                "signalattice_telemetry_worker_termination_failures_total",
                worker_termination_failures[channel],
                channel=channel.value,
            )

        _header(
            lines,
            "signalattice_telemetry_delivery_indeterminate_records_total",
            "Records whose remote delivery could not be determined at shutdown.",
            "counter",
        )
        for channel in TelemetryChannel:
            _sample(
                lines,
                "signalattice_telemetry_delivery_indeterminate_records_total",
                delivery_indeterminate[channel],
                channel=channel.value,
            )

        body = ("\n".join(lines) + "\n").encode("ascii")
        series_count = sum(1 for line in lines if line and not line.startswith("#"))
        return MetricsSnapshot(body=body, series_count=series_count)


def _header(lines: list[str], name: str, help_text: str, metric_type: str) -> None:
    lines.extend((f"# HELP {name} {help_text}", f"# TYPE {name} {metric_type}"))


def _escape_label(value: str) -> str:
    return value.replace("\\", "\\\\").replace("\n", "\\n").replace('"', '\\"')


def _format_number(value: int | float) -> str:
    if type(value) is int:
        return str(value)
    if not math.isfinite(value):
        raise MetricsBoundError("non-finite metric value reached exposition")
    if value == 0.0:
        return "0"
    return repr(value)


def _sample(
    lines: list[str],
    name: str,
    value: int | float,
    **labels: str,
) -> None:
    rendered_labels = ",".join(
        f'{key}="{_escape_label(label)}"' for key, label in sorted(labels.items())
    )
    suffix = f"{{{rendered_labels}}}" if rendered_labels else ""
    lines.append(f"{name}{suffix} {_format_number(value)}")


def _histogram_samples(
    lines: list[str],
    name: str,
    histogram: _Histogram,
    **labels: str,
) -> None:
    for upper, count in zip(histogram.buckets, histogram.bucket_counts, strict=True):
        _sample(lines, f"{name}_bucket", count, **labels, le=_format_number(upper))
    _sample(lines, f"{name}_bucket", histogram.count, **labels, le="+Inf")
    _sample(lines, f"{name}_sum", histogram.total, **labels)
    _sample(lines, f"{name}_count", histogram.count, **labels)
