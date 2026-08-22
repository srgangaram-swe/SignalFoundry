"""Closed, privacy-safe contracts for local read-service telemetry.

Telemetry is a disclosure boundary, not a dumping ground for request context.
Every string that can reach a metric label, span, or structured log is selected
from an enum defined here.  Raw paths are collapsed to route templates before a
request timer is created; URLs, queries, headers, client metadata, business
identifiers, artifact paths, and exception text are therefore unrepresentable.

Records use deterministic single-line JSON.  They contain at most 32 total
fields and 8 KiB, reject non-finite values, and use JSON escaping for every C0
or C1 control character.  These bounds are rechecked at serialization time so
even an object mutated through unsupported reflection cannot bypass the export
boundary.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from enum import StrEnum
from typing import Final

MAX_TELEMETRY_FIELDS: Final = 32
MAX_TELEMETRY_RECORD_BYTES: Final = 8 * 1024
MAX_TELEMETRY_PATH_BYTES: Final = 512
MAX_RESPONSE_BYTES: Final = 2 * 1024 * 1024
MAX_REQUEST_DURATION_SECONDS: Final = 300.0
TELEMETRY_SCHEMA_VERSION: Final = 1
_MAX_UNIX_MILLISECONDS: Final = 32_503_680_000_000  # 3000-01-01T00:00:00Z


class TelemetryContractError(ValueError):
    """Telemetry input cannot be represented without disclosure or ambiguity."""

    code = "invalid_telemetry_contract"


class RouteTemplate(StrEnum):
    """Complete low-cardinality route inventory for the read-only service."""

    LIVE = "/health/live"
    READY = "/health/ready"
    METRICS = "/internal/metrics"
    OPENAPI = "/api/v1/openapi.json"
    RUNS = "/api/v1/runs"
    RUN = "/api/v1/runs/{run_id}"
    FORECAST_SUMMARIES = "/api/v1/runs/{run_id}/forecast-summaries"
    DIAGNOSTICS = "/api/v1/runs/{run_id}/diagnostics"
    RUN_ARTIFACTS = "/api/v1/runs/{run_id}/artifacts"
    ARTIFACT = "/api/v1/artifacts/{artifact_id}"
    MODEL_CARDS = "/api/v1/model-cards"
    MODEL_CARD = "/api/v1/model-cards/{card_id}"
    UNMATCHED = "unmatched"


class Operation(StrEnum):
    """Bounded service operation families used by histograms and spans."""

    PROBE = "probe"
    METRICS = "metrics"
    COLLECTION_READ = "collection_read"
    ENTITY_READ = "entity_read"
    DIAGNOSTIC_READ = "diagnostic_read"
    SPECIFICATION = "specification"
    UNMATCHED = "unmatched"


class Outcome(StrEnum):
    """Closed request outcomes; no exception or provider value becomes a label."""

    SUCCESS = "success"
    NOT_FOUND = "not_found"
    INVALID = "invalid"
    UNAVAILABLE = "unavailable"
    REJECTED = "rejected"
    INTERNAL_ERROR = "internal_error"
    CANCELLED = "cancelled"


class RejectionReason(StrEnum):
    """Reviewed admission and resource-limit rejection reasons."""

    API_RATE_LIMIT = "api_rate_limit"
    PROBE_RATE_LIMIT = "probe_rate_limit"
    GLOBAL_CONCURRENCY = "global_concurrency"
    DATA_CONCURRENCY = "data_concurrency"
    APPLICATION_CAPACITY = "application_capacity"
    QUERY_LIMIT = "query_limit"
    HEADER_LIMIT = "header_limit"
    BODY_FORBIDDEN = "body_forbidden"
    RESPONSE_LIMIT = "response_limit"
    INVALID_METADATA = "invalid_metadata"
    SHUTTING_DOWN = "shutting_down"


class TelemetryChannel(StrEnum):
    """Independently bounded asynchronous record channels."""

    LOG = "log"
    TRACE = "trace"


class DropReason(StrEnum):
    """Observable bounded reasons that a telemetry record was not delivered."""

    QUEUE_FULL = "queue_full"
    INVALID_RECORD = "invalid_record"
    LOCAL_SINK_CAPACITY = "local_sink_capacity"
    LOCAL_SINK_FAILURE = "local_sink_failure"
    EXPORT_FAILED = "export_failed"
    SHUTDOWN_TIMEOUT = "shutdown_timeout"
    NOT_RUNNING = "not_running"


class ExportOutcome(StrEnum):
    """Low-cardinality outcome of one bounded exporter attempt."""

    SUCCESS = "success"
    FAILURE = "failure"
    TIMEOUT = "timeout"


class TelemetryEvent(StrEnum):
    """Only events permitted in structured logs and spans."""

    REQUEST_COMPLETED = "request_completed"
    REQUEST_REJECTED = "request_rejected"
    REQUEST_SPAN = "request_span"
    SERVICE_LIFECYCLE = "service_lifecycle"


class Severity(StrEnum):
    """Closed JSON-log severity vocabulary."""

    INFO = "info"
    WARNING = "warning"
    ERROR = "error"


class LifecycleState(StrEnum):
    """Auditable local service lifecycle transitions."""

    STARTING = "starting"
    READY = "ready"
    DRAINING = "draining"
    STOPPING = "stopping"
    STOPPED = "stopped"
    TERMINATION_TIMEOUT = "termination_timeout"


class StatusClass(StrEnum):
    """Bounded HTTP status projection that cannot disclose response content."""

    INFORMATIONAL = "1xx"
    SUCCESS = "2xx"
    REDIRECTION = "3xx"
    CLIENT_ERROR = "4xx"
    SERVER_ERROR = "5xx"
    CANCELLED = "cancelled"


type TelemetryScalar = str | int | float | bool

_ROUTE_OPERATIONS: Final = {
    RouteTemplate.LIVE: Operation.PROBE,
    RouteTemplate.READY: Operation.PROBE,
    RouteTemplate.METRICS: Operation.METRICS,
    RouteTemplate.OPENAPI: Operation.SPECIFICATION,
    RouteTemplate.RUNS: Operation.COLLECTION_READ,
    RouteTemplate.RUN: Operation.ENTITY_READ,
    RouteTemplate.FORECAST_SUMMARIES: Operation.COLLECTION_READ,
    RouteTemplate.DIAGNOSTICS: Operation.DIAGNOSTIC_READ,
    RouteTemplate.RUN_ARTIFACTS: Operation.COLLECTION_READ,
    RouteTemplate.ARTIFACT: Operation.ENTITY_READ,
    RouteTemplate.MODEL_CARDS: Operation.COLLECTION_READ,
    RouteTemplate.MODEL_CARD: Operation.ENTITY_READ,
    RouteTemplate.UNMATCHED: Operation.UNMATCHED,
}
_FIXED_ROUTES: Final = {
    RouteTemplate.LIVE.value: RouteTemplate.LIVE,
    RouteTemplate.READY.value: RouteTemplate.READY,
    RouteTemplate.METRICS.value: RouteTemplate.METRICS,
    RouteTemplate.OPENAPI.value: RouteTemplate.OPENAPI,
    RouteTemplate.RUNS.value: RouteTemplate.RUNS,
    RouteTemplate.MODEL_CARDS.value: RouteTemplate.MODEL_CARDS,
}

_REQUEST_FIELDS: Final = frozenset(
    {
        "duration_ms",
        "operation",
        "outcome",
        "response_bytes",
        "route",
        "status_class",
    }
)
_REJECTION_FIELDS: Final = _REQUEST_FIELDS | {"rejection"}
_LIFECYCLE_FIELDS: Final = frozenset({"lifecycle"})
_EVENT_CHANNELS: Final = {
    TelemetryEvent.REQUEST_COMPLETED: TelemetryChannel.LOG,
    TelemetryEvent.REQUEST_REJECTED: TelemetryChannel.LOG,
    TelemetryEvent.REQUEST_SPAN: TelemetryChannel.TRACE,
    TelemetryEvent.SERVICE_LIFECYCLE: TelemetryChannel.LOG,
}
_EVENT_FIELDS: Final = {
    TelemetryEvent.REQUEST_COMPLETED: _REQUEST_FIELDS,
    TelemetryEvent.REQUEST_REJECTED: _REJECTION_FIELDS,
    TelemetryEvent.REQUEST_SPAN: _REJECTION_FIELDS,
    TelemetryEvent.SERVICE_LIFECYCLE: _LIFECYCLE_FIELDS,
}
_EVENT_REQUIRED_FIELDS: Final = {
    TelemetryEvent.REQUEST_COMPLETED: _REQUEST_FIELDS,
    TelemetryEvent.REQUEST_REJECTED: _REJECTION_FIELDS,
    TelemetryEvent.REQUEST_SPAN: _REQUEST_FIELDS,
    TelemetryEvent.SERVICE_LIFECYCLE: _LIFECYCLE_FIELDS,
}


def classify_route(path: object) -> RouteTemplate:
    """Collapse one untrusted path to a reviewed template without retaining it.

    The function is deliberately total: malformed types, controls, non-ASCII
    input, and oversized paths map to ``UNMATCHED``.  Identifier syntax is
    validated by the HTTP boundary; telemetry needs only the route shape.
    """

    if type(path) is not str or not path:
        return RouteTemplate.UNMATCHED
    # ASCII bytes and code points have identical length.  Guard the immutable
    # string before encoding so an oversized, untrusted ASGI path cannot force
    # a second allocation merely to be classified as unmatched.
    if len(path) > MAX_TELEMETRY_PATH_BYTES:
        return RouteTemplate.UNMATCHED
    try:
        encoded = path.encode("ascii")
    except (UnicodeEncodeError, MemoryError):
        return RouteTemplate.UNMATCHED
    if any(byte < 0x20 or byte == 0x7F for byte in encoded):
        return RouteTemplate.UNMATCHED

    matched = _FIXED_ROUTES.get(path)
    if matched is not None:
        return matched

    segments = path.split("/")
    if len(segments) == 6 and segments[:4] == ["", "api", "v1", "runs"] and segments[4]:
        suffix = segments[5]
        if suffix == "forecast-summaries":
            return RouteTemplate.FORECAST_SUMMARIES
        if suffix == "diagnostics":
            return RouteTemplate.DIAGNOSTICS
        if suffix == "artifacts":
            return RouteTemplate.RUN_ARTIFACTS
    if len(segments) == 5 and segments[:4] == ["", "api", "v1", "runs"] and segments[4]:
        return RouteTemplate.RUN
    if len(segments) == 5 and segments[:4] == ["", "api", "v1", "artifacts"] and segments[4]:
        return RouteTemplate.ARTIFACT
    if len(segments) == 5 and segments[:4] == ["", "api", "v1", "model-cards"] and segments[4]:
        return RouteTemplate.MODEL_CARD
    return RouteTemplate.UNMATCHED


def operation_for_route(route: RouteTemplate) -> Operation:
    """Return the fixed operation family for an exact route enum."""

    if type(route) is not RouteTemplate:
        raise TelemetryContractError("route must be an exact RouteTemplate")
    return _ROUTE_OPERATIONS[route]


def _status_class(status_code: int, outcome: Outcome) -> StatusClass:
    if outcome is Outcome.CANCELLED:
        if status_code != 499:
            raise TelemetryContractError("cancelled observations use the synthetic status 499")
        return StatusClass.CANCELLED
    if not 100 <= status_code <= 599:
        raise TelemetryContractError("status_code must be in [100, 599]")
    return (
        StatusClass.INFORMATIONAL,
        StatusClass.SUCCESS,
        StatusClass.REDIRECTION,
        StatusClass.CLIENT_ERROR,
        StatusClass.SERVER_ERROR,
    )[status_code // 100 - 1]


def _validate_outcome_status(outcome: Outcome, status_code: int) -> None:
    allowed = {
        Outcome.SUCCESS: 200 <= status_code <= 299,
        Outcome.NOT_FOUND: status_code == 404,
        Outcome.INVALID: status_code in {400, 405, 413, 422},
        Outcome.UNAVAILABLE: status_code == 503,
        Outcome.REJECTED: status_code in {400, 405, 413, 422, 429, 503},
        Outcome.INTERNAL_ERROR: status_code == 500,
        Outcome.CANCELLED: status_code == 499,
    }
    if not allowed[outcome]:
        raise TelemetryContractError("outcome and status_code are inconsistent")


@dataclass(frozen=True, slots=True)
class RequestObservation:
    """One completed request projected to non-disclosing bounded values."""

    route: RouteTemplate
    operation: Operation
    outcome: Outcome
    status_code: int
    duration_seconds: float
    response_bytes: int
    rejection: RejectionReason | None = None

    def __post_init__(self) -> None:
        if type(self.route) is not RouteTemplate:
            raise TelemetryContractError("route must be an exact RouteTemplate")
        if type(self.operation) is not Operation or self.operation is not operation_for_route(
            self.route
        ):
            raise TelemetryContractError("operation does not match the route template")
        if type(self.outcome) is not Outcome:
            raise TelemetryContractError("outcome must be an exact Outcome")
        if type(self.status_code) is not int:
            raise TelemetryContractError("status_code must be an exact integer")
        _status_class(self.status_code, self.outcome)
        _validate_outcome_status(self.outcome, self.status_code)
        if (
            type(self.duration_seconds) is not float
            or not math.isfinite(self.duration_seconds)
            or not 0.0 <= self.duration_seconds <= MAX_REQUEST_DURATION_SECONDS
        ):
            raise TelemetryContractError("duration_seconds is outside the finite service bound")
        if (
            type(self.response_bytes) is not int
            or not 0 <= self.response_bytes <= MAX_RESPONSE_BYTES
        ):
            raise TelemetryContractError("response_bytes is outside the service response bound")
        if self.outcome is Outcome.REJECTED:
            if type(self.rejection) is not RejectionReason:
                raise TelemetryContractError("rejected observations require a rejection reason")
        elif self.rejection is not None:
            raise TelemetryContractError(
                "non-rejected observations cannot carry a rejection reason"
            )

    @property
    def status_class(self) -> StatusClass:
        """Return the status projection used in logs and spans."""

        return _status_class(self.status_code, self.outcome)


def _record_value_is_valid(key: str, value: TelemetryScalar) -> bool:
    if key == "route":
        return type(value) is str and value in {member.value for member in RouteTemplate}
    if key == "operation":
        return type(value) is str and value in {member.value for member in Operation}
    if key == "outcome":
        return type(value) is str and value in {member.value for member in Outcome}
    if key == "status_class":
        return type(value) is str and value in {member.value for member in StatusClass}
    if key == "rejection":
        return type(value) is str and value in {member.value for member in RejectionReason}
    if key == "lifecycle":
        return type(value) is str and value in {member.value for member in LifecycleState}
    if key == "duration_ms":
        return type(value) is float and math.isfinite(value) and 0.0 <= value <= 300_000.0
    if key == "response_bytes":
        return type(value) is int and 0 <= value <= MAX_RESPONSE_BYTES
    return False


@dataclass(frozen=True, slots=True)
class TelemetryRecord:
    """One immutable allowlisted JSON log or span record."""

    channel: TelemetryChannel
    event: TelemetryEvent
    severity: Severity
    occurred_at_unix_ms: int
    attributes: tuple[tuple[str, TelemetryScalar], ...]

    def __post_init__(self) -> None:
        self._validate()

    def _validate(self) -> None:
        if type(self.channel) is not TelemetryChannel:
            raise TelemetryContractError("channel must be an exact TelemetryChannel")
        if (
            type(self.event) is not TelemetryEvent
            or _EVENT_CHANNELS.get(self.event) is not self.channel
        ):
            raise TelemetryContractError("event is not permitted on this telemetry channel")
        if type(self.severity) is not Severity:
            raise TelemetryContractError("severity must be an exact Severity")
        if (
            type(self.occurred_at_unix_ms) is not int
            or not 0 <= self.occurred_at_unix_ms <= _MAX_UNIX_MILLISECONDS
        ):
            raise TelemetryContractError("occurred_at_unix_ms is outside the supported bound")
        if type(self.attributes) is not tuple:
            raise TelemetryContractError("attributes must be an immutable tuple")
        if len(self.attributes) + 6 > MAX_TELEMETRY_FIELDS:
            raise TelemetryContractError("telemetry record exceeds the 32-field bound")
        observed_keys: list[str] = []
        for pair in self.attributes:
            if type(pair) is not tuple or len(pair) != 2 or type(pair[0]) is not str:
                raise TelemetryContractError("telemetry attributes must be exact key/value tuples")
            key, value = pair
            if not _record_value_is_valid(key, value):
                raise TelemetryContractError("telemetry attribute is not allowlisted")
            observed_keys.append(key)
        if observed_keys != sorted(observed_keys) or len(observed_keys) != len(set(observed_keys)):
            raise TelemetryContractError("telemetry attributes must have unique canonical ordering")
        observed = frozenset(observed_keys)
        if not _EVENT_REQUIRED_FIELDS[self.event] <= observed <= _EVENT_FIELDS[self.event]:
            raise TelemetryContractError("telemetry attributes do not match the event schema")

    def json_line(self) -> bytes:
        """Return deterministic ASCII JSON plus one escaped record delimiter."""

        self._validate()
        payload = {
            "attributes": dict(self.attributes),
            "channel": self.channel.value,
            "event": self.event.value,
            "occurred_at_unix_ms": self.occurred_at_unix_ms,
            "schema_version": TELEMETRY_SCHEMA_VERSION,
            "severity": self.severity.value,
        }
        try:
            encoded = (
                json.dumps(
                    payload,
                    allow_nan=False,
                    ensure_ascii=True,
                    separators=(",", ":"),
                    sort_keys=True,
                ).encode("ascii")
                + b"\n"
            )
        except (MemoryError, RecursionError, TypeError, UnicodeError, ValueError):
            raise TelemetryContractError("telemetry record cannot be serialized safely") from None
        if not 1 <= len(encoded) <= MAX_TELEMETRY_RECORD_BYTES:
            raise TelemetryContractError("telemetry record exceeds the 8-KiB bound")
        # The only literal control byte permitted is the final record delimiter;
        # json.dumps escapes controls inside every JSON string.
        if any(byte < 0x20 for byte in encoded[:-1]):
            raise TelemetryContractError("telemetry JSON contains an unescaped control character")
        return encoded


def _severity_for(outcome: Outcome) -> Severity:
    if outcome in {Outcome.UNAVAILABLE, Outcome.INTERNAL_ERROR}:
        return Severity.ERROR
    if outcome in {Outcome.INVALID, Outcome.REJECTED}:
        return Severity.WARNING
    return Severity.INFO


def records_for_request(
    observation: RequestObservation,
    *,
    occurred_at_unix_ms: int,
) -> tuple[TelemetryRecord, TelemetryRecord]:
    """Build the structured log and span records for one observation."""

    if type(observation) is not RequestObservation:
        raise TelemetryContractError("observation must be an exact RequestObservation")
    attributes: tuple[tuple[str, TelemetryScalar], ...] = tuple(
        sorted(
            {
                "duration_ms": observation.duration_seconds * 1_000.0,
                "operation": observation.operation.value,
                "outcome": observation.outcome.value,
                "response_bytes": observation.response_bytes,
                "route": observation.route.value,
                "status_class": observation.status_class.value,
            }.items()
        )
    )
    log_event = TelemetryEvent.REQUEST_COMPLETED
    if observation.rejection is not None:
        attributes = tuple(sorted((*attributes, ("rejection", observation.rejection.value))))
        log_event = TelemetryEvent.REQUEST_REJECTED
    severity = _severity_for(observation.outcome)
    return (
        TelemetryRecord(
            channel=TelemetryChannel.LOG,
            event=log_event,
            severity=severity,
            occurred_at_unix_ms=occurred_at_unix_ms,
            attributes=attributes,
        ),
        TelemetryRecord(
            channel=TelemetryChannel.TRACE,
            event=TelemetryEvent.REQUEST_SPAN,
            severity=severity,
            occurred_at_unix_ms=occurred_at_unix_ms,
            attributes=attributes,
        ),
    )


def lifecycle_record(
    state: LifecycleState,
    *,
    occurred_at_unix_ms: int,
) -> TelemetryRecord:
    """Build one bounded lifecycle log without host or process identifiers."""

    if type(state) is not LifecycleState:
        raise TelemetryContractError("state must be an exact LifecycleState")
    return TelemetryRecord(
        channel=TelemetryChannel.LOG,
        event=TelemetryEvent.SERVICE_LIFECYCLE,
        severity=(Severity.ERROR if state is LifecycleState.TERMINATION_TIMEOUT else Severity.INFO),
        occurred_at_unix_ms=occurred_at_unix_ms,
        attributes=(("lifecycle", state.value),),
    )
