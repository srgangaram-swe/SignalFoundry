"""Raw-ASGI resource-bound and fault tests for the service perimeter."""

from __future__ import annotations

import asyncio
import json
import tracemalloc
from collections.abc import Awaitable, Callable
from typing import Any

import h11
import pytest

from quant_platform.service.admission import AdmissionController, AdmissionLimits
from quant_platform.service.metrics import ServiceMetrics
from quant_platform.service.middleware import SecurityBoundaryMiddleware
from quant_platform.service.telemetry import ServiceTelemetry
from quant_platform.service.telemetry_contracts import RejectionReason

type Message = dict[str, Any]
type InnerApp = Callable[
    [dict[str, Any], Callable[[], Awaitable[Message]], Callable[[Message], Awaitable[None]]],
    Awaitable[None],
]


class _FakeClock:
    value = 100.0

    def __call__(self) -> float:
        return self.value


async def _ok_app(
    _scope: dict[str, Any],
    _receive: Callable[[], Awaitable[Message]],
    send: Callable[[Message], Awaitable[None]],
) -> None:
    await send(
        {
            "type": "http.response.start",
            "status": 200,
            "headers": [(b"content-type", b"application/json")],
        }
    )
    await send({"type": "http.response.body", "body": b"{}"})


async def _exchange(
    app: InnerApp,
    path: bytes,
    *,
    decoded_path: str | None = None,
    query: bytes = b"",
    headers: list[tuple[bytes, bytes]] | None = None,
    events: list[Message] | None = None,
) -> list[Message]:
    pending = list(
        events
        if events is not None
        else [{"type": "http.request", "body": b"", "more_body": False}]
    )
    emitted: list[Message] = []

    async def receive() -> Message:
        if not pending:
            return {"type": "http.disconnect"}
        return pending.pop(0)

    async def send(message: Message) -> None:
        emitted.append(message)

    await app(
        {
            "type": "http",
            "asgi": {"version": "3.0"},
            "http_version": "1.1",
            "method": "GET",
            "scheme": "http",
            "path": path.decode("ascii") if decoded_path is None else decoded_path,
            "raw_path": path,
            "query_string": query,
            "headers": [(b"host", b"localhost")] if headers is None else headers,
            "client": ("127.0.0.1", 12345),
            "server": ("127.0.0.1", 8765),
            "state": {},
        },
        receive,
        send,
    )
    return emitted


def _problem(messages: list[Message]) -> tuple[int, dict[str, object]]:
    assert len(messages) == 2
    return int(messages[0]["status"]), json.loads(messages[1]["body"])


def test_fixed_rate_buckets_have_exact_bursts_without_client_state() -> None:
    clock = _FakeClock()
    admission = AdmissionController(clock=clock)
    boundary = SecurityBoundaryMiddleware(_ok_app, admission=admission)

    async def scenario() -> tuple[list[Message], list[Message]]:
        for index in range(40):
            messages = await _exchange(boundary, f"/api/v1/runs/{index}".encode())
            assert messages[0]["status"] == 200
        data_rejected = await _exchange(boundary, b"/api/v1/runs/excess")
        for _ in range(4):
            messages = await _exchange(boundary, b"/health/live")
            assert messages[0]["status"] == 200
        operations_rejected = await _exchange(boundary, b"/internal/metrics")
        return data_rejected, operations_rejected

    data, operations = asyncio.run(scenario())
    assert _problem(data)[0] == 429
    assert _problem(data)[1]["code"] == "rate_limited"
    assert _problem(operations)[0] == 429
    assert _problem(operations)[1]["code"] == "rate_limited"


def test_invalid_metadata_consumes_the_fixed_data_rate_budget() -> None:
    clock = _FakeClock()
    admission = AdmissionController(
        AdmissionLimits(
            global_concurrency=2,
            data_concurrency=2,
            api_rate_per_second=1.0,
            api_burst=2,
            operations_rate_per_second=1.0,
            operations_burst=2,
        ),
        clock=clock,
    )
    boundary = SecurityBoundaryMiddleware(_ok_app, admission=admission)

    async def scenario() -> tuple[list[list[Message]], list[Message]]:
        invalid = [
            await _exchange(
                boundary,
                b"/api/v1/runs",
                headers=[(b"host", f"attacker-{index}.invalid".encode("ascii"))],
            )
            for index in range(2)
        ]
        excess = await _exchange(
            boundary,
            b"/api/v1/runs",
            headers=[(b"host", b"attacker-excess.invalid")],
        )
        return invalid, excess

    invalid, excess = asyncio.run(scenario())
    assert [_problem(response)[0] for response in invalid] == [400, 400]
    assert _problem(excess)[:1] == (429,)
    assert _problem(excess)[1]["code"] == "rate_limited"
    assert (b"connection", b"close") in excess[0]["headers"]
    assert admission.snapshot().active_total == 0


def test_invalid_metadata_is_held_inside_the_global_concurrency_bound() -> None:
    release = asyncio.Event()
    all_started = asyncio.Event()
    started = 0
    admission = AdmissionController(
        AdmissionLimits(
            global_concurrency=2,
            data_concurrency=2,
            api_rate_per_second=1_000.0,
            api_burst=1_000,
            operations_rate_per_second=1_000.0,
            operations_burst=1_000,
        )
    )
    boundary = SecurityBoundaryMiddleware(_ok_app, admission=admission)

    async def blocked_invalid_exchange(index: int) -> list[Message]:
        nonlocal started
        emitted: list[Message] = []

        async def receive() -> Message:
            return {"type": "http.request", "body": b"", "more_body": False}

        async def send(message: Message) -> None:
            nonlocal started
            emitted.append(message)
            if message.get("type") == "http.response.start":
                started += 1
                if started == 2:
                    all_started.set()
                await release.wait()

        await boundary(
            {
                "type": "http",
                "asgi": {"version": "3.0"},
                "http_version": "1.1",
                "method": "GET",
                "scheme": "http",
                "path": "/api/v1/runs",
                "raw_path": b"/api/v1/runs",
                "query_string": b"",
                "headers": [(b"host", f"invalid-{index}.example".encode("ascii"))],
                "client": ("127.0.0.1", 12_345),
                "server": ("127.0.0.1", 8_765),
                "state": {},
            },
            receive,
            send,
        )
        return emitted

    async def scenario() -> tuple[list[Message], list[list[Message]]]:
        tasks = [asyncio.create_task(blocked_invalid_exchange(index)) for index in range(2)]
        await asyncio.wait_for(all_started.wait(), timeout=1)
        assert admission.snapshot().active_total == 2
        excess = await _exchange(
            boundary,
            b"/api/v1/runs",
            headers=[(b"host", b"invalid-excess.example")],
        )
        release.set()
        completed = await asyncio.wait_for(asyncio.gather(*tasks), timeout=1)
        return excess, completed

    excess, completed = asyncio.run(scenario())
    assert _problem(excess)[0] == 429
    assert _problem(excess)[1]["code"] == "service_saturated"
    assert [_problem(response)[0] for response in completed] == [400, 400]
    assert admission.snapshot().active_total == 0


def test_percent_encoded_operation_alias_is_rejected_inside_operations_budget() -> None:
    clock = _FakeClock()
    admission = AdmissionController(clock=clock)
    boundary = SecurityBoundaryMiddleware(_ok_app, admission=admission)

    async def scenario() -> tuple[list[list[Message]], list[Message], list[Message]]:
        invalid_aliases = [
            await _exchange(
                boundary,
                b"/health/%6cive",
                decoded_path="/health/live",
            )
            for _ in range(4)
        ]
        excess_alias = await _exchange(
            boundary,
            b"/health/%6cive",
            decoded_path="/health/live",
        )
        unaffected_data = await _exchange(boundary, b"/api/v1/runs")
        return invalid_aliases, excess_alias, unaffected_data

    invalid_aliases, excess_alias, unaffected_data = asyncio.run(scenario())
    assert [_problem(response)[0] for response in invalid_aliases] == [400] * 4
    assert _problem(excess_alias)[0] == 429
    assert _problem(excess_alias)[1]["code"] == "rate_limited"
    assert unaffected_data[0]["status"] == 200


def test_data_saturation_retains_probe_capacity_under_barrier() -> None:
    release = asyncio.Event()
    entered = 0

    async def blocking_app(
        scope: dict[str, Any],
        _receive: Callable[[], Awaitable[Message]],
        send: Callable[[Message], Awaitable[None]],
    ) -> None:
        nonlocal entered
        if bytes(scope["raw_path"]).startswith(b"/api/"):
            entered += 1
            await release.wait()
        await _ok_app(scope, _receive, send)

    admission = AdmissionController(
        AdmissionLimits(
            global_concurrency=32,
            data_concurrency=24,
            api_rate_per_second=1_000.0,
            api_burst=1_000,
            operations_rate_per_second=1_000.0,
            operations_burst=1_000,
        )
    )
    boundary = SecurityBoundaryMiddleware(blocking_app, admission=admission)

    async def scenario() -> tuple[list[Message], list[Message]]:
        tasks = [
            asyncio.create_task(_exchange(boundary, f"/api/v1/runs/{index}".encode()))
            for index in range(24)
        ]
        for _ in range(10_000):
            if entered == 24:
                break
            await asyncio.sleep(0)
        assert entered == 24
        excess = await _exchange(boundary, b"/api/v1/runs/excess")
        probe = await _exchange(boundary, b"/health/live")
        release.set()
        await asyncio.wait_for(asyncio.gather(*tasks), timeout=2)
        return excess, probe

    excess, probe = asyncio.run(scenario())
    assert _problem(excess)[0] == 429
    assert _problem(excess)[1]["code"] == "service_saturated"
    assert probe[0]["status"] == 200
    assert admission.snapshot().active_total == 0


def test_body_without_length_oversized_metadata_and_response_fail_closed() -> None:
    boundary = SecurityBoundaryMiddleware(_ok_app)

    async def oversized_response(
        _scope: dict[str, Any],
        _receive: Callable[[], Awaitable[Message]],
        send: Callable[[Message], Awaitable[None]],
    ) -> None:
        await send({"type": "http.response.start", "status": 200, "headers": []})
        await send({"type": "http.response.body", "body": b"x" * (2 * 1_024 * 1_024 + 1)})

    async def scenario() -> tuple[list[Message], list[Message], list[Message], list[Message]]:
        body = await _exchange(
            boundary,
            b"/health/live",
            events=[
                {"type": "http.request", "body": b"", "more_body": True},
                {"type": "http.request", "body": b"canary-private", "more_body": False},
            ],
        )
        query = await _exchange(boundary, b"/health/live", query=b"q" * (4 * 1_024 + 1))
        headers = await _exchange(
            boundary,
            b"/health/live",
            headers=[(b"host", b"localhost"), (b"x-padding", b"p" * (16 * 1_024))],
        )
        response = await _exchange(SecurityBoundaryMiddleware(oversized_response), b"/health/live")
        return body, query, headers, response

    body, query, headers, response = asyncio.run(scenario())
    assert _problem(body)[0] == 413
    assert b"canary-private" not in bytes(body[1]["body"])
    assert _problem(query)[0] == 400
    assert _problem(headers)[0] == 400
    assert _problem(response)[0] == 503
    assert _problem(response)[1]["code"] == "response_limit_exceeded"


def test_scope_rejects_oversized_path_before_ascii_encoding() -> None:
    boundary = SecurityBoundaryMiddleware(_ok_app)
    path = "/" + "x" * 2_000_000
    scope = {
        "method": "GET",
        "path": path,
        "raw_path": b"/",
        "query_string": b"",
        "headers": [(b"host", b"localhost")],
    }

    tracemalloc.start()
    try:
        problem, reason = boundary._validate_scope(scope)
        _, peak_bytes = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()

    assert problem is not None and problem.code == "invalid_request"
    assert reason is RejectionReason.INVALID_METADATA
    assert peak_bytes < 64 * 1024


def test_scope_rejects_header_budget_before_scanning_oversized_value(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    boundary = SecurityBoundaryMiddleware(_ok_app)
    original = boundary._valid_header_value

    def guarded_scan(value: bytes) -> bool:
        if len(value) > 16 * 1024:
            raise AssertionError("oversized value must not be scanned")
        return original(value)

    monkeypatch.setattr(boundary, "_valid_header_value", guarded_scan)
    problem, reason = boundary._validate_scope(
        {
            "method": "GET",
            "path": "/health/live",
            "raw_path": b"/health/live",
            "query_string": b"",
            "headers": [
                (b"host", b"localhost"),
                (b"x-padding", b"p" * 2_000_000),
            ],
        }
    )

    assert problem is not None and problem.code == "invalid_request"
    assert reason is RejectionReason.HEADER_LIMIT


@pytest.mark.parametrize(
    "headers",
    [
        [(b"host", b"localhost"), (b"bad(name", b"value")],
        [(b"host", b"localhost"), (b"bad name", b"value")],
        [(b"host", b"localhost"), (b":authority", b"localhost")],
        [(b"host", b"localhost"), (b"x-test", b"nul\x00value")],
        [(b"host", b"localhost"), (b"x-test", b"tab\tvalue")],
        [(b"host", b"localhost"), (b"x-test", b"escape\x1bvalue")],
        [(b"host", b"localhost"), (b"x-test", b"delete\x7fvalue")],
        [(b"host", b"localhost"), (b"x-test", b"obs-text-\x80")],
    ],
)
def test_request_header_names_and_values_reject_ambiguous_bytes(
    headers: list[tuple[bytes, bytes]],
) -> None:
    response = asyncio.run(
        _exchange(SecurityBoundaryMiddleware(_ok_app), b"/health/live", headers=headers)
    )

    assert _problem(response)[0] == 400
    assert _problem(response)[1]["code"] == "invalid_request"
    assert not any(value in bytes(response[1]["body"]) for _, value in headers[1:])


def test_request_headers_accept_the_complete_rfc_token_and_visible_ascii_contract() -> None:
    token = b"!#$%&'*+-.^_`|~0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"
    visible_ascii = bytes(range(0x20, 0x7F))

    response = asyncio.run(
        _exchange(
            SecurityBoundaryMiddleware(_ok_app),
            b"/health/live",
            headers=[(b"host", b"localhost"), (token, visible_ascii)],
        )
    )

    assert response[0]["status"] == 200


@pytest.mark.parametrize(
    "control",
    [b"\x01", b"\t", b"\x1b", b"\x1f", b"\x7f", b"\x80"],
)
def test_real_h11_parsed_control_values_still_fail_the_stricter_asgi_boundary(
    control: bytes,
) -> None:
    marker = b"canary" + control + b"value"
    connection = h11.Connection(h11.SERVER)
    connection.receive_data(
        b"GET /health/live HTTP/1.1\r\n" b"Host: localhost\r\n" b"X-Test: " + marker + b"\r\n\r\n"
    )
    event = connection.next_event()
    assert isinstance(event, h11.Request)

    response = asyncio.run(
        _exchange(
            SecurityBoundaryMiddleware(_ok_app),
            b"/health/live",
            headers=list(event.headers),
        )
    )

    assert _problem(response)[0] == 400
    assert marker not in bytes(response[1]["body"])


@pytest.mark.parametrize(
    ("name", "value"),
    [
        (b"bad(name", b"value"),
        (b"x-test", b"nul\x00value"),
        (b"x-test", b"tab\tvalue"),
        (b"x-test", b"escape\x1bvalue"),
        (b"x-test", b"delete\x7fvalue"),
        (b"x-test", b"obs-text-\x80"),
    ],
)
def test_application_response_headers_fail_closed_on_ambiguous_bytes(
    name: bytes,
    value: bytes,
) -> None:
    async def invalid_response(
        _scope: dict[str, Any],
        _receive: Callable[[], Awaitable[Message]],
        send: Callable[[Message], Awaitable[None]],
    ) -> None:
        await send({"type": "http.response.start", "status": 200, "headers": [(name, value)]})
        await send({"type": "http.response.body", "body": b"{}"})

    response = asyncio.run(_exchange(SecurityBoundaryMiddleware(invalid_response), b"/health/live"))

    status, document = _problem(response)
    assert status == 500
    assert document["code"] == "internal_error"
    assert value not in bytes(response[1]["body"])


@pytest.mark.parametrize("status", [True, 99, 100, 199, 600, "secret-status", None])
def test_application_response_status_requires_an_exact_bounded_integer(status: object) -> None:
    async def invalid_response(
        _scope: dict[str, Any],
        _receive: Callable[[], Awaitable[Message]],
        send: Callable[[Message], Awaitable[None]],
    ) -> None:
        await send({"type": "http.response.start", "status": status, "headers": []})
        await send({"type": "http.response.body", "body": b"secret-status"})

    response = asyncio.run(_exchange(SecurityBoundaryMiddleware(invalid_response), b"/health/live"))

    assert _problem(response)[0] == 500
    assert _problem(response)[1]["code"] == "internal_error"
    assert b"secret-status" not in bytes(response[1]["body"])


@pytest.mark.parametrize("status", [204, 304])
def test_application_no_body_status_rejects_payload_bytes(status: int) -> None:
    async def invalid_response(
        _scope: dict[str, Any],
        _receive: Callable[[], Awaitable[Message]],
        send: Callable[[Message], Awaitable[None]],
    ) -> None:
        await send({"type": "http.response.start", "status": status, "headers": []})
        await send({"type": "http.response.body", "body": b"secret-body"})

    response = asyncio.run(_exchange(SecurityBoundaryMiddleware(invalid_response), b"/health/live"))

    assert _problem(response)[0] == 500
    assert _problem(response)[1]["code"] == "internal_error"
    assert b"secret-body" not in bytes(response[1]["body"])


@pytest.mark.parametrize("more_body", [1, "false", None, [], {}])
def test_application_more_body_requires_an_exact_boolean(more_body: object) -> None:
    async def invalid_response(
        _scope: dict[str, Any],
        _receive: Callable[[], Awaitable[Message]],
        send: Callable[[Message], Awaitable[None]],
    ) -> None:
        await send({"type": "http.response.start", "status": 200, "headers": []})
        await send(
            {
                "type": "http.response.body",
                "body": b"secret-body",
                "more_body": more_body,
            }
        )

    response = asyncio.run(_exchange(SecurityBoundaryMiddleware(invalid_response), b"/health/live"))

    assert _problem(response)[0] == 500
    assert _problem(response)[1]["code"] == "internal_error"
    assert b"secret-body" not in bytes(response[1]["body"])


def test_admission_is_held_while_waiting_for_the_first_body_event_and_released_on_cancel() -> None:
    admission = AdmissionController(
        AdmissionLimits(
            global_concurrency=1,
            data_concurrency=1,
            api_rate_per_second=1_000.0,
            api_burst=1_000,
            operations_rate_per_second=1_000.0,
            operations_burst=1_000,
        )
    )
    telemetry = ServiceTelemetry(ServiceMetrics())
    boundary = SecurityBoundaryMiddleware(
        _ok_app,
        admission=admission,
        telemetry=telemetry,
    )
    entered = asyncio.Event()

    async def scenario() -> None:
        async def receive() -> Message:
            entered.set()
            await asyncio.Event().wait()
            raise AssertionError("unreachable")

        async def send(_message: Message) -> None:
            raise AssertionError("cancelled request must not emit a response")

        task = asyncio.create_task(
            boundary(
                {
                    "type": "http",
                    "asgi": {"version": "3.0"},
                    "http_version": "1.1",
                    "method": "GET",
                    "scheme": "http",
                    "path": "/api/v1/runs/public-id",
                    "raw_path": b"/api/v1/runs/public-id",
                    "query_string": b"",
                    "headers": [(b"host", b"localhost")],
                    "client": ("127.0.0.1", 12345),
                    "server": ("127.0.0.1", 8765),
                    "state": {},
                },
                receive,
                send,
            )
        )
        await asyncio.wait_for(entered.wait(), timeout=1)
        assert admission.snapshot().active_total == 1
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

    asyncio.run(scenario())

    assert admission.snapshot().active_total == 0
    body = telemetry.metrics.snapshot().body
    assert (
        b'signalattice_http_requests_total{outcome="cancelled",route="/api/v1/runs/{run_id}"} 1'
        in body
    )


def test_application_failure_has_only_closed_structured_telemetry(
    caplog: pytest.LogCaptureFixture,
) -> None:
    marker = b"PRIVATE-EXCEPTION-CANARY"
    telemetry = ServiceTelemetry(ServiceMetrics())

    async def exploding_app(
        _scope: dict[str, Any],
        _receive: Callable[[], Awaitable[Message]],
        _send: Callable[[Message], Awaitable[None]],
    ) -> None:
        raise RuntimeError(marker.decode("ascii"))

    response = asyncio.run(
        _exchange(
            SecurityBoundaryMiddleware(exploding_app, telemetry=telemetry),
            b"/api/v1/runs/private-id",
        )
    )

    assert _problem(response)[0] == 500
    assert caplog.records == []
    records = telemetry.local_snapshot().records
    assert len(records) == 2
    assert {json.loads(record)["channel"] for record in records} == {"log", "trace"}
    assert {json.loads(record)["attributes"]["outcome"] for record in records} == {"internal_error"}
    assert marker not in b"".join(records)


def test_shutdown_rejection_is_structured_and_retryable() -> None:
    admission = AdmissionController()
    boundary = SecurityBoundaryMiddleware(_ok_app, admission=admission)
    admission.begin_shutdown()

    response = asyncio.run(_exchange(boundary, b"/health/live"))

    status, document = _problem(response)
    assert status == 503
    assert document["code"] == "service_draining"
    start_headers = dict(response[0]["headers"])
    assert start_headers[b"retry-after"] == b"1"


def test_perimeter_telemetry_records_closed_outcomes_and_rejections() -> None:
    metrics = ServiceMetrics()
    telemetry = ServiceTelemetry(metrics)

    async def oversized_response(
        _scope: dict[str, Any],
        _receive: Callable[[], Awaitable[Message]],
        send: Callable[[Message], Awaitable[None]],
    ) -> None:
        await send({"type": "http.response.start", "status": 200, "headers": []})
        await send({"type": "http.response.body", "body": b"x" * (2 * 1_024 * 1_024 + 1)})

    async def capacity_response(
        _scope: dict[str, Any],
        _receive: Callable[[], Awaitable[Message]],
        send: Callable[[Message], Awaitable[None]],
    ) -> None:
        await send({"type": "http.response.start", "status": 429, "headers": []})
        await send({"type": "http.response.body", "body": b"{}"})

    async def scenario() -> None:
        success = await _exchange(
            SecurityBoundaryMiddleware(_ok_app, telemetry=telemetry),
            b"/api/v1/runs/public-id",
        )
        forbidden_body = await _exchange(
            SecurityBoundaryMiddleware(_ok_app, telemetry=telemetry),
            b"/health/live",
            events=[{"type": "http.request", "body": b"private-canary"}],
        )
        oversized = await _exchange(
            SecurityBoundaryMiddleware(oversized_response, telemetry=telemetry),
            b"/health/live",
        )
        capacity = await _exchange(
            SecurityBoundaryMiddleware(capacity_response, telemetry=telemetry),
            b"/health/live",
        )
        assert success[0]["status"] == 200
        assert forbidden_body[0]["status"] == 413
        assert oversized[0]["status"] == 503
        assert capacity[0]["status"] == 429

    asyncio.run(scenario())
    body = metrics.snapshot().body

    assert (
        b'signalattice_http_requests_total{outcome="success",route="/api/v1/runs/{run_id}"} 1'
        in body
    )
    assert b'signalattice_http_requests_total{outcome="rejected",route="/health/live"} 3' in body
    assert b'signalattice_admission_rejections_total{reason="body_forbidden"} 1' in body
    assert b'signalattice_admission_rejections_total{reason="response_limit"} 1' in body
    assert b'signalattice_admission_rejections_total{reason="application_capacity"} 1' in body
    assert b"private-canary" not in body
    assert b'signalattice_http_in_flight{operation="entity_read"} 0' in body


def test_telemetry_clock_failure_cannot_replace_an_admitted_response() -> None:
    class InvalidClock:
        @staticmethod
        def time() -> float:
            return float("nan")

        @staticmethod
        def monotonic() -> float:
            return 0.0

    metrics = ServiceMetrics()
    telemetry = ServiceTelemetry(metrics, clock=InvalidClock())
    boundary = SecurityBoundaryMiddleware(_ok_app, telemetry=telemetry)

    response = asyncio.run(_exchange(boundary, b"/health/live"))
    body = metrics.snapshot().body

    assert response[0]["status"] == 200
    assert (
        b'signalattice_telemetry_dropped_records_total{channel="log",reason="invalid_record"} 1'
        in body
    )
    assert (
        b'signalattice_telemetry_dropped_records_total{channel="trace",reason="invalid_record"} 1'
        in body
    )


def test_client_cancellation_releases_admission_and_records_bounded_outcome() -> None:
    entered = asyncio.Event()
    blocker = asyncio.Event()
    admission = AdmissionController()
    metrics = ServiceMetrics()

    async def blocked_app(
        _scope: dict[str, Any],
        _receive: Callable[[], Awaitable[Message]],
        _send: Callable[[Message], Awaitable[None]],
    ) -> None:
        entered.set()
        await blocker.wait()

    boundary = SecurityBoundaryMiddleware(
        blocked_app,
        admission=admission,
        telemetry=ServiceTelemetry(metrics),
    )

    async def scenario() -> None:
        task = asyncio.create_task(_exchange(boundary, b"/api/v1/runs/public-id"))
        await asyncio.wait_for(entered.wait(), timeout=1)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

    asyncio.run(scenario())
    body = metrics.snapshot().body

    assert admission.snapshot().active_total == 0
    assert (
        b'signalattice_http_requests_total{outcome="cancelled",route="/api/v1/runs/{run_id}"} 1'
        in body
    )
    assert b'signalattice_http_in_flight{operation="entity_read"} 0' in body
