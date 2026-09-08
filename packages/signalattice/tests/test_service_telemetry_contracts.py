"""Privacy, strictness, and serialization tests for service telemetry contracts."""

from __future__ import annotations

import json
import tracemalloc

import pytest

from quant_platform.service.telemetry_contracts import (
    MAX_RESPONSE_BYTES,
    MAX_TELEMETRY_FIELDS,
    MAX_TELEMETRY_RECORD_BYTES,
    LifecycleState,
    Operation,
    Outcome,
    RejectionReason,
    RequestObservation,
    RouteTemplate,
    Severity,
    StatusClass,
    TelemetryChannel,
    TelemetryContractError,
    TelemetryEvent,
    TelemetryRecord,
    classify_route,
    lifecycle_record,
    operation_for_route,
    records_for_request,
)


@pytest.mark.parametrize(
    ("path", "expected"),
    [
        ("/health/live", RouteTemplate.LIVE),
        ("/health/ready", RouteTemplate.READY),
        ("/internal/metrics", RouteTemplate.METRICS),
        ("/api/v1/openapi.json", RouteTemplate.OPENAPI),
        ("/api/v1/runs", RouteTemplate.RUNS),
        ("/api/v1/runs/r1_public", RouteTemplate.RUN),
        (
            "/api/v1/runs/r1_public/forecast-summaries",
            RouteTemplate.FORECAST_SUMMARIES,
        ),
        ("/api/v1/runs/r1_public/diagnostics", RouteTemplate.DIAGNOSTICS),
        ("/api/v1/runs/r1_public/artifacts", RouteTemplate.RUN_ARTIFACTS),
        ("/api/v1/artifacts/" + "a" * 64, RouteTemplate.ARTIFACT),
        ("/api/v1/model-cards", RouteTemplate.MODEL_CARDS),
        ("/api/v1/model-cards/public-card", RouteTemplate.MODEL_CARD),
    ],
)
def test_route_classification_collapses_identifiers_to_templates(
    path: str,
    expected: RouteTemplate,
) -> None:
    assert classify_route(path) is expected


@pytest.mark.parametrize(
    "path",
    [
        None,
        b"/health/live",
        "",
        "/not-a-route",
        "/api/v1/runs/",
        "/api/v1/runs/id/unknown",
        "/openapi.json",
        "/health/live?canary=secret",
        "/health/live\nINJECTED",
        "/api/v1/runs/canary-\N{SNOWMAN}",
        "/" + "x" * 513,
    ],
)
def test_route_classification_is_total_and_non_disclosing(path: object) -> None:
    assert classify_route(path) is RouteTemplate.UNMATCHED


def test_route_classification_rejects_oversize_before_ascii_encoding() -> None:
    path = "/" + "x" * 2_000_000

    tracemalloc.start()
    try:
        assert classify_route(path) is RouteTemplate.UNMATCHED
        _, peak_bytes = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()

    # Encoding the exact ASCII string would allocate roughly two MiB.  The
    # pre-encoding length guard keeps classification scratch space tiny.
    assert peak_bytes < 64 * 1024


def test_operation_mapping_is_closed_and_exact() -> None:
    assert operation_for_route(RouteTemplate.LIVE) is Operation.PROBE
    assert operation_for_route(RouteTemplate.RUNS) is Operation.COLLECTION_READ
    assert operation_for_route(RouteTemplate.DIAGNOSTICS) is Operation.DIAGNOSTIC_READ
    with pytest.raises(TelemetryContractError, match="exact RouteTemplate"):
        operation_for_route("/health/live")  # type: ignore[arg-type]


def test_request_observation_enforces_status_operation_and_resource_bounds() -> None:
    observation = RequestObservation(
        route=RouteTemplate.RUN,
        operation=Operation.ENTITY_READ,
        outcome=Outcome.SUCCESS,
        status_code=200,
        duration_seconds=0.025,
        response_bytes=4_096,
    )

    assert observation.status_class is StatusClass.SUCCESS
    with pytest.raises(TelemetryContractError, match="operation"):
        RequestObservation(
            route=RouteTemplate.RUN,
            operation=Operation.PROBE,
            outcome=Outcome.SUCCESS,
            status_code=200,
            duration_seconds=0.025,
            response_bytes=4_096,
        )
    with pytest.raises(TelemetryContractError, match="inconsistent"):
        RequestObservation(
            route=RouteTemplate.RUN,
            operation=Operation.ENTITY_READ,
            outcome=Outcome.SUCCESS,
            status_code=500,
            duration_seconds=0.025,
            response_bytes=4_096,
        )
    with pytest.raises(TelemetryContractError, match="duration"):
        RequestObservation(
            route=RouteTemplate.RUN,
            operation=Operation.ENTITY_READ,
            outcome=Outcome.SUCCESS,
            status_code=200,
            duration_seconds=float("nan"),
            response_bytes=4_096,
        )
    with pytest.raises(TelemetryContractError, match="response_bytes"):
        RequestObservation(
            route=RouteTemplate.RUN,
            operation=Operation.ENTITY_READ,
            outcome=Outcome.SUCCESS,
            status_code=200,
            duration_seconds=0.025,
            response_bytes=MAX_RESPONSE_BYTES + 1,
        )


def test_rejection_reason_is_required_only_for_rejected_outcomes() -> None:
    rejected = RequestObservation(
        route=RouteTemplate.RUNS,
        operation=Operation.COLLECTION_READ,
        outcome=Outcome.REJECTED,
        status_code=429,
        duration_seconds=0.0,
        response_bytes=256,
        rejection=RejectionReason.API_RATE_LIMIT,
    )

    assert rejected.rejection is RejectionReason.API_RATE_LIMIT
    with pytest.raises(TelemetryContractError, match="require"):
        RequestObservation(
            route=RouteTemplate.RUNS,
            operation=Operation.COLLECTION_READ,
            outcome=Outcome.REJECTED,
            status_code=429,
            duration_seconds=0.0,
            response_bytes=256,
        )
    with pytest.raises(TelemetryContractError, match="cannot carry"):
        RequestObservation(
            route=RouteTemplate.RUNS,
            operation=Operation.COLLECTION_READ,
            outcome=Outcome.SUCCESS,
            status_code=200,
            duration_seconds=0.0,
            response_bytes=256,
            rejection=RejectionReason.API_RATE_LIMIT,
        )


@pytest.mark.parametrize(
    ("status_code", "reason"),
    [
        (400, RejectionReason.INVALID_METADATA),
        (405, RejectionReason.INVALID_METADATA),
        (413, RejectionReason.BODY_FORBIDDEN),
        (422, RejectionReason.QUERY_LIMIT),
        (429, RejectionReason.API_RATE_LIMIT),
        (429, RejectionReason.APPLICATION_CAPACITY),
        (503, RejectionReason.SHUTTING_DOWN),
    ],
)
def test_perimeter_rejections_support_their_structured_http_statuses(
    status_code: int,
    reason: RejectionReason,
) -> None:
    observation = RequestObservation(
        route=RouteTemplate.UNMATCHED,
        operation=Operation.UNMATCHED,
        outcome=Outcome.REJECTED,
        status_code=status_code,
        duration_seconds=0.0,
        response_bytes=256,
        rejection=reason,
    )

    assert observation.rejection is reason


def test_records_are_canonical_bounded_and_contain_only_reviewed_fields() -> None:
    observation = RequestObservation(
        route=RouteTemplate.DIAGNOSTICS,
        operation=Operation.DIAGNOSTIC_READ,
        outcome=Outcome.UNAVAILABLE,
        status_code=503,
        duration_seconds=0.125,
        response_bytes=512,
    )

    log_record, span_record = records_for_request(
        observation,
        occurred_at_unix_ms=1_786_291_200_000,
    )
    log_line = log_record.json_line()
    span_line = span_record.json_line()
    decoded = json.loads(log_line)

    assert log_record.channel is TelemetryChannel.LOG
    assert span_record.channel is TelemetryChannel.TRACE
    assert log_record.severity is Severity.ERROR
    assert len(log_record.attributes) + 6 <= MAX_TELEMETRY_FIELDS
    assert len(log_line) <= MAX_TELEMETRY_RECORD_BYTES
    assert len(span_line) <= MAX_TELEMETRY_RECORD_BYTES
    assert log_line.endswith(b"\n")
    assert not any(byte < 0x20 for byte in log_line[:-1])
    assert decoded["attributes"] == {
        "duration_ms": 125.0,
        "operation": "diagnostic_read",
        "outcome": "unavailable",
        "response_bytes": 512,
        "route": "/api/v1/runs/{run_id}/diagnostics",
        "status_class": "5xx",
    }
    assert b"status_code" not in log_line


def test_rejection_log_and_trace_carry_only_the_closed_reason() -> None:
    observation = RequestObservation(
        route=RouteTemplate.UNMATCHED,
        operation=Operation.UNMATCHED,
        outcome=Outcome.REJECTED,
        status_code=503,
        duration_seconds=0.001,
        response_bytes=300,
        rejection=RejectionReason.GLOBAL_CONCURRENCY,
    )

    log_record, span_record = records_for_request(observation, occurred_at_unix_ms=1)

    assert log_record.event is TelemetryEvent.REQUEST_REJECTED
    assert dict(log_record.attributes)["rejection"] == "global_concurrency"
    assert dict(span_record.attributes)["rejection"] == "global_concurrency"


def test_arbitrary_fields_and_control_character_values_are_unrepresentable() -> None:
    canary = "SECRET\r\nforged-log-line"
    with pytest.raises(TelemetryContractError, match="allowlisted") as failure:
        TelemetryRecord(
            channel=TelemetryChannel.LOG,
            event=TelemetryEvent.SERVICE_LIFECYCLE,
            severity=Severity.INFO,
            occurred_at_unix_ms=1,
            attributes=(("lifecycle", canary),),
        )
    assert "SECRET" not in str(failure.value)

    with pytest.raises(TelemetryContractError, match="event schema"):
        TelemetryRecord(
            channel=TelemetryChannel.LOG,
            event=TelemetryEvent.SERVICE_LIFECYCLE,
            severity=Severity.INFO,
            occurred_at_unix_ms=1,
            attributes=(
                ("lifecycle", LifecycleState.READY.value),
                ("route", RouteTemplate.LIVE.value),
            ),
        )


def test_record_revalidates_after_unsupported_reflection_mutation() -> None:
    record = lifecycle_record(LifecycleState.READY, occurred_at_unix_ms=1)
    object.__setattr__(record, "attributes", (("lifecycle", "unsafe\nvalue"),))

    with pytest.raises(TelemetryContractError, match="allowlisted"):
        record.json_line()


def test_lifecycle_record_is_deterministic_and_non_identifying() -> None:
    first = lifecycle_record(LifecycleState.STARTING, occurred_at_unix_ms=123)
    second = lifecycle_record(LifecycleState.STARTING, occurred_at_unix_ms=123)

    assert first.json_line() == second.json_line()
    assert json.loads(first.json_line())["attributes"] == {"lifecycle": "starting"}
