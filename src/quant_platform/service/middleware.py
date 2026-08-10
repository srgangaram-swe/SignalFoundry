"""Raw ASGI trust boundary for bounded loopback-only HTTP reads.

The middleware sits outside FastAPI so hostile Host values, unsupported
methods, oversized metadata, request bodies, proxy hints, application failures,
and response-size overruns all receive the same redacted RFC 9457 contract.
Responses are buffered only up to a fixed ceiling; this API has no streaming or
download route.
"""

from __future__ import annotations

import asyncio
import secrets
from collections.abc import Awaitable, Callable
from typing import Any, Final

from quant_platform.service.admission import (
    AdmissionController,
    AdmissionLane,
    AdmissionLimits,
    AdmissionRejection,
)
from quant_platform.service.problems import (
    BAD_REQUEST,
    INTERNAL,
    METHOD_NOT_ALLOWED,
    PAYLOAD_TOO_LARGE,
    RATE_LIMITED,
    RESPONSE_LIMIT_EXCEEDED,
    SATURATED,
    SERVICE_DRAINING,
    ProblemSpec,
    build_problem,
)
from quant_platform.service.telemetry import (
    RequestTimer,
    ServiceTelemetry,
    TelemetryRuntimeError,
)
from quant_platform.service.telemetry_contracts import (
    DropReason,
    Operation,
    Outcome,
    RejectionReason,
    TelemetryChannel,
    TelemetryContractError,
    operation_for_route,
)

type Scope = dict[str, Any]
type Message = dict[str, Any]
type Receive = Callable[[], Awaitable[Message]]
type Send = Callable[[Message], Awaitable[None]]
type ASGIApp = Callable[[Scope, Receive, Send], Awaitable[None]]

_PROBLEM_MEDIA_TYPE: Final = b"application/problem+json"
_SECURITY_HEADERS: Final = (
    (b"cache-control", b"no-store"),
    (
        b"content-security-policy",
        b"default-src 'none'; base-uri 'none'; form-action 'none'; frame-ancestors 'none'",
    ),
    (b"cross-origin-resource-policy", b"same-origin"),
    (b"permissions-policy", b"accelerometer=(), camera=(), geolocation=(), microphone=()"),
    (b"referrer-policy", b"no-referrer"),
    (b"x-content-type-options", b"nosniff"),
    (b"x-frame-options", b"DENY"),
)
_PROXY_HEADERS: Final = frozenset(
    {
        b"forwarded",
        b"x-forwarded-for",
        b"x-forwarded-host",
        b"x-forwarded-port",
        b"x-forwarded-proto",
    }
)
_OPERATIONS_PATHS: Final = frozenset({"/health/live", "/health/ready", "/internal/metrics"})
_REQUEST_EVENT_TIMEOUT_SECONDS: Final = 0.25
_MAX_EMPTY_REQUEST_EVENTS: Final = 8
_HEADER_NAME_BYTES: Final = frozenset(
    b"!#$%&'*+-.^_`|~0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"
)


class _ResponseLimitError(RuntimeError):
    """The inner application attempted to exceed the fixed response ceiling."""


class SecurityBoundaryMiddleware:
    """Enforce the local service's method, metadata, capacity, and output bounds."""

    def __init__(
        self,
        app: ASGIApp,
        *,
        max_concurrency: int = 32,
        max_data_concurrency: int = 24,
        max_headers: int = 64,
        max_header_bytes: int = 16 * 1024,
        max_query_bytes: int = 4 * 1024,
        max_path_bytes: int = 512,
        max_response_bytes: int = 2 * 1024 * 1024,
        allowed_port: int | None = None,
        admission: AdmissionController | None = None,
        telemetry: ServiceTelemetry | None = None,
    ) -> None:
        bounds = {
            "max_concurrency": (max_concurrency, 1, 1_024),
            "max_data_concurrency": (max_data_concurrency, 1, 1_024),
            "max_headers": (max_headers, 1, 256),
            "max_header_bytes": (max_header_bytes, 1_024, 64 * 1024),
            "max_query_bytes": (max_query_bytes, 0, 16 * 1024),
            "max_path_bytes": (max_path_bytes, 64, 4 * 1024),
            "max_response_bytes": (max_response_bytes, 4 * 1024, 4 * 1024 * 1024),
        }
        for name, (value, lower, upper) in bounds.items():
            if type(value) is not int or not lower <= value <= upper:
                raise ValueError(f"{name} must be an integer in [{lower}, {upper}]")
        if allowed_port is not None and (
            type(allowed_port) is not int or not 1 <= allowed_port <= 65_535
        ):
            raise ValueError("allowed_port must be null or an integer in [1, 65535]")
        if max_data_concurrency > max_concurrency:
            raise ValueError("max_data_concurrency cannot exceed max_concurrency")
        if admission is not None and type(admission) is not AdmissionController:
            raise TypeError("admission must be an AdmissionController")
        if telemetry is not None and type(telemetry) is not ServiceTelemetry:
            raise TypeError("telemetry must be a ServiceTelemetry")
        self._app = app
        self._max_headers = max_headers
        self._max_header_bytes = max_header_bytes
        self._max_query_bytes = max_query_bytes
        self._max_path_bytes = max_path_bytes
        self._max_response_bytes = max_response_bytes
        self._allowed_port = allowed_port
        self._admission = (
            AdmissionController(
                AdmissionLimits(
                    global_concurrency=max_concurrency,
                    data_concurrency=max_data_concurrency,
                )
            )
            if admission is None
            else admission
        )
        self._telemetry = telemetry

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        """Validate one ASGI exchange and emit a bounded secured response."""

        if scope.get("type") != "http":
            await self._app(scope, receive, send)
            return
        timer = self._begin_telemetry(scope.get("path"))
        status_code = 500
        response_bytes = 0
        forced_outcome: Outcome | None = None
        rejection_reason: RejectionReason | None = None
        admitted_operation: Operation | None = None

        async def observed_send(message: Message) -> None:
            nonlocal status_code, response_bytes
            if type(message) is dict:
                if message.get("type") == "http.response.start":
                    candidate = message.get("status")
                    if type(candidate) is int and 100 <= candidate <= 599:
                        status_code = candidate
                elif message.get("type") == "http.response.body":
                    candidate_body = message.get("body", b"")
                    if type(candidate_body) is bytes:
                        response_bytes = min(
                            self._max_response_bytes,
                            response_bytes + len(candidate_body),
                        )
            await send(message)

        request_id = secrets.token_hex(16)
        try:
            lane = self._classify_lane(scope)
            try:
                decision = self._admission.try_acquire(lane)
            except RuntimeError:
                await self._send_problem(observed_send, INTERNAL, request_id)
                return
            if not decision.accepted:
                forced_outcome = Outcome.REJECTED
                rejection_reason = self._admission_rejection_reason(
                    decision.rejection,
                    lane=lane,
                )
                if decision.rejection is AdmissionRejection.RATE_LIMITED:
                    rejection_problem = RATE_LIMITED
                elif decision.rejection is AdmissionRejection.SHUTTING_DOWN:
                    rejection_problem = SERVICE_DRAINING
                else:
                    rejection_problem = SATURATED
                await self._send_problem(observed_send, rejection_problem, request_id)
                return
            try:
                admitted_operation = None if timer is None else operation_for_route(timer.route)
                if admitted_operation is not None:
                    self._adjust_in_flight(admitted_operation, 1)
                validation_problem, rejection_reason = self._validate_scope(scope)
                if validation_problem is not None:
                    forced_outcome = Outcome.REJECTED
                    await self._send_problem(observed_send, validation_problem, request_id)
                    return
                initial_message, body_problem, rejection_reason = await self._receive_empty_request(
                    receive
                )
                if body_problem is not None:
                    forced_outcome = Outcome.REJECTED
                    await self._send_problem(observed_send, body_problem, request_id)
                    return
                replayed = False

                async def replay_receive() -> Message:
                    nonlocal replayed
                    if not replayed:
                        replayed = True
                        return initial_message
                    return await receive()

                state = scope.setdefault("state", {})
                if type(state) is not dict:
                    forced_outcome = Outcome.REJECTED
                    rejection_reason = RejectionReason.INVALID_METADATA
                    await self._send_problem(observed_send, BAD_REQUEST, request_id)
                    return
                state["signalattice_request_id"] = request_id
                state["signalattice_admission_lane"] = lane.value
                scope["headers"] = [
                    pair for pair in scope.get("headers", ()) if pair[0].lower() != b"x-request-id"
                ]
                bounded_rejection = await self._call_bounded(
                    scope,
                    replay_receive,
                    observed_send,
                    request_id,
                )
                if bounded_rejection is not None:
                    forced_outcome = Outcome.REJECTED
                    rejection_reason = bounded_rejection
            finally:
                if admitted_operation is not None:
                    self._adjust_in_flight(admitted_operation, -1)
                self._admission.release(lane)
        except asyncio.CancelledError:
            status_code = 499
            response_bytes = 0
            forced_outcome = Outcome.CANCELLED
            rejection_reason = None
            raise
        finally:
            if timer is not None:
                if forced_outcome is None and status_code == 429:
                    forced_outcome = Outcome.REJECTED
                    rejection_reason = RejectionReason.APPLICATION_CAPACITY
                self._finish_telemetry(
                    timer,
                    outcome=(
                        forced_outcome
                        if forced_outcome is not None
                        else self._outcome_for_status(status_code)
                    ),
                    status_code=status_code,
                    response_bytes=response_bytes,
                    rejection=rejection_reason,
                )

    @staticmethod
    async def _receive_empty_request(
        receive: Receive,
    ) -> tuple[Message, ProblemSpec | None, RejectionReason | None]:
        try:
            async with asyncio.timeout(_REQUEST_EVENT_TIMEOUT_SECONDS):
                for _ in range(_MAX_EMPTY_REQUEST_EVENTS):
                    message = await receive()
                    if type(message) is not dict or message.get("type") != "http.request":
                        return {}, BAD_REQUEST, RejectionReason.INVALID_METADATA
                    body = message.get("body", b"")
                    more_body = message.get("more_body", False)
                    if type(body) is not bytes or type(more_body) is not bool:
                        return {}, BAD_REQUEST, RejectionReason.INVALID_METADATA
                    if body:
                        return {}, PAYLOAD_TOO_LARGE, RejectionReason.BODY_FORBIDDEN
                    if not more_body:
                        return (
                            {"type": "http.request", "body": b"", "more_body": False},
                            None,
                            None,
                        )
        except TimeoutError:
            return {}, BAD_REQUEST, RejectionReason.INVALID_METADATA
        return {}, PAYLOAD_TOO_LARGE, RejectionReason.BODY_FORBIDDEN

    @staticmethod
    def _classify_lane(scope: Scope) -> AdmissionLane:
        path = scope.get("path")
        return (
            AdmissionLane.OPERATIONS
            if type(path) is str and path in _OPERATIONS_PATHS
            else AdmissionLane.DATA
        )

    def _validate_scope(
        self,
        scope: Scope,
    ) -> tuple[ProblemSpec | None, RejectionReason | None]:
        method = scope.get("method")
        if method != "GET":
            return METHOD_NOT_ALLOWED, RejectionReason.INVALID_METADATA
        path = scope.get("path")
        raw_path = scope.get("raw_path", b"")
        query = scope.get("query_string", b"")
        headers = scope.get("headers", ())
        if type(path) is not str or not 1 <= len(path) <= self._max_path_bytes:
            canonical_path = b""
        else:
            try:
                canonical_path = path.encode("ascii")
            except UnicodeEncodeError:
                canonical_path = b""
        if (
            not canonical_path
            or type(raw_path) is not bytes
            or not 1 <= len(raw_path) <= self._max_path_bytes
            or not raw_path.isascii()
            or canonical_path != raw_path
            or type(query) is not bytes
        ):
            return BAD_REQUEST, RejectionReason.INVALID_METADATA
        if len(query) > self._max_query_bytes:
            return BAD_REQUEST, RejectionReason.QUERY_LIMIT
        if type(headers) is not list:
            return BAD_REQUEST, RejectionReason.INVALID_METADATA
        if not 1 <= len(headers) <= self._max_headers:
            return BAD_REQUEST, RejectionReason.HEADER_LIMIT
        header_bytes = 0
        normalized: dict[bytes, list[bytes]] = {}
        for pair in headers:
            if (
                type(pair) is not tuple
                or len(pair) != 2
                or type(pair[0]) is not bytes
                or type(pair[1]) is not bytes
            ):
                return BAD_REQUEST, RejectionReason.INVALID_METADATA
            name, value = pair
            pair_bytes = len(name) + len(value)
            if pair_bytes > self._max_header_bytes - header_bytes:
                return BAD_REQUEST, RejectionReason.HEADER_LIMIT
            header_bytes += pair_bytes
            if not self._valid_header_name(name) or not self._valid_header_value(value):
                return BAD_REQUEST, RejectionReason.INVALID_METADATA
            normalized.setdefault(name.lower(), []).append(value)
        if _PROXY_HEADERS.intersection(normalized):
            return BAD_REQUEST, RejectionReason.INVALID_METADATA
        if not self._valid_host(normalized.get(b"host"), allowed_port=self._allowed_port):
            return BAD_REQUEST, RejectionReason.INVALID_METADATA
        if (
            b"transfer-encoding" in normalized
            or b"expect" in normalized
            or b"content-encoding" in normalized
        ):
            return BAD_REQUEST, RejectionReason.INVALID_METADATA
        if b"x-request-id" in normalized:
            return BAD_REQUEST, RejectionReason.INVALID_METADATA
        content_lengths = normalized.get(b"content-length", [])
        if len(content_lengths) > 1:
            return BAD_REQUEST, RejectionReason.INVALID_METADATA
        if content_lengths and content_lengths[0] != b"0":
            return PAYLOAD_TOO_LARGE, RejectionReason.BODY_FORBIDDEN
        if query.count(b"&") >= 32:
            return BAD_REQUEST, RejectionReason.QUERY_LIMIT
        return None, None

    @staticmethod
    def _admission_rejection_reason(
        rejection: AdmissionRejection | None,
        *,
        lane: AdmissionLane,
    ) -> RejectionReason:
        if rejection is AdmissionRejection.RATE_LIMITED:
            return (
                RejectionReason.PROBE_RATE_LIMIT
                if lane is AdmissionLane.OPERATIONS
                else RejectionReason.API_RATE_LIMIT
            )
        if rejection is AdmissionRejection.DATA_CONCURRENCY:
            return RejectionReason.DATA_CONCURRENCY
        if rejection is AdmissionRejection.GLOBAL_CONCURRENCY:
            return RejectionReason.GLOBAL_CONCURRENCY
        if rejection is AdmissionRejection.SHUTTING_DOWN:
            return RejectionReason.SHUTTING_DOWN
        raise RuntimeError("admission returned an unknown rejection")

    @staticmethod
    def _outcome_for_status(status_code: int) -> Outcome:
        if 200 <= status_code <= 299:
            return Outcome.SUCCESS
        if status_code == 404:
            return Outcome.NOT_FOUND
        if status_code in {400, 405, 413, 422}:
            return Outcome.INVALID
        if status_code == 503:
            return Outcome.UNAVAILABLE
        return Outcome.INTERNAL_ERROR

    def _begin_telemetry(self, raw_path: object) -> RequestTimer | None:
        if self._telemetry is None:
            return None
        try:
            return self._telemetry.begin_request(raw_path)
        except (TelemetryContractError, TelemetryRuntimeError):
            self._record_telemetry_failure()
            return None

    def _finish_telemetry(
        self,
        timer: RequestTimer,
        *,
        outcome: Outcome,
        status_code: int,
        response_bytes: int,
        rejection: RejectionReason | None,
    ) -> None:
        if self._telemetry is None:
            return
        try:
            self._telemetry.finish_request(
                timer,
                outcome=outcome,
                status_code=status_code,
                response_bytes=response_bytes,
                rejection=rejection,
            )
        except (TelemetryContractError, TelemetryRuntimeError):
            self._record_telemetry_failure()

    def _adjust_in_flight(self, operation: Operation, delta: int) -> None:
        if self._telemetry is None:
            return
        try:
            self._telemetry.metrics.adjust_in_flight(operation, delta)
        except TelemetryContractError:
            self._record_telemetry_failure()

    def _record_telemetry_failure(self) -> None:
        if self._telemetry is not None:
            try:
                for channel in TelemetryChannel:
                    self._telemetry.metrics.record_drop(channel, DropReason.INVALID_RECORD)
            except TelemetryContractError:
                pass

    @staticmethod
    def _valid_header_name(name: bytes) -> bool:
        """Accept only non-empty RFC token bytes for field names."""

        return bool(name) and all(byte in _HEADER_NAME_BYTES for byte in name)

    @staticmethod
    def _valid_header_value(value: bytes) -> bool:
        """Accept SP and visible ASCII; reject HTAB, C0, DEL, and obs-text."""

        return all(0x20 <= byte <= 0x7E for byte in value)

    @staticmethod
    def _valid_host(values: list[bytes] | None, *, allowed_port: int | None) -> bool:
        if values is None or len(values) != 1:
            return False
        try:
            authority = values[0].decode("ascii").lower()
        except UnicodeDecodeError:
            return False
        host, separator, port = authority.partition(":")
        if host not in {"127.0.0.1", "localhost"}:
            return False
        if not separator:
            return True
        if not port.isdigit() or not 1 <= int(port) <= 65_535 or str(int(port)) != port:
            return False
        return allowed_port is None or int(port) == allowed_port

    async def _call_bounded(
        self,
        scope: Scope,
        receive: Receive,
        send: Send,
        request_id: str,
    ) -> RejectionReason | None:
        start: Message | None = None
        body = bytearray()
        completed = False

        async def capture(message: Message) -> None:
            nonlocal start, completed
            message_type = message.get("type")
            if message_type == "http.response.start":
                if start is not None or completed:
                    raise RuntimeError("application emitted duplicate response start")
                status = message.get("status")
                # The boundary buffers one final response; it deliberately does not
                # implement ASGI informational-response sequencing.  Rejecting 1xx
                # here also prevents the h11 adapter from receiving a final
                # ``Response`` with an informational status.
                if type(status) is not int or not 200 <= status <= 599:
                    raise RuntimeError("application emitted an invalid response status")
                start = message
                return
            if message_type != "http.response.body" or start is None or completed:
                raise RuntimeError("application emitted an invalid HTTP response sequence")
            chunk = message.get("body", b"")
            more_body = message.get("more_body", False)
            if type(chunk) is not bytes or type(more_body) is not bool:
                raise RuntimeError("application emitted an invalid response body")
            if len(chunk) > self._max_response_bytes - len(body):
                raise _ResponseLimitError("application response exceeded its byte ceiling")
            body.extend(chunk)
            if not more_body:
                completed = True

        try:
            await self._app(scope, receive, capture)
            if start is None or not completed:
                raise RuntimeError("application did not complete one bounded HTTP response")
        except _ResponseLimitError:
            await self._send_problem(send, RESPONSE_LIMIT_EXCEEDED, request_id)
            return RejectionReason.RESPONSE_LIMIT
        except Exception:
            # The outer boundary intentionally discards exception identity and
            # text. The final request observation records only the fixed
            # internal-error outcome in the canonical telemetry sink.
            await self._send_problem(send, INTERNAL, request_id)
            return None
        status = start["status"]
        if status in {204, 304} and body:
            # h11 forbids payload bytes for these final statuses.  Detect the
            # invalid application response while it is still inside the
            # redacting boundary so protocol-specific exceptions cannot escape.
            await self._send_problem(send, INTERNAL, request_id)
            return None
        raw_headers = start.get("headers", [])
        if type(raw_headers) is not list or len(raw_headers) > self._max_headers:
            await self._send_problem(send, INTERNAL, request_id)
            return None
        response_header_bytes = 0
        for pair in raw_headers:
            if (
                type(pair) is not tuple
                or len(pair) != 2
                or type(pair[0]) is not bytes
                or type(pair[1]) is not bytes
            ):
                await self._send_problem(send, INTERNAL, request_id)
                return None
            name, value = pair
            response_header_bytes += len(name) + len(value)
            normalized_name = name.lower()
            if (
                not self._valid_header_name(name)
                or not self._valid_header_value(value)
                or normalized_name == b"set-cookie"
                or normalized_name.startswith(b"access-control-")
            ):
                await self._send_problem(send, INTERNAL, request_id)
                return None
        if response_header_bytes > self._max_header_bytes:
            await self._send_problem(send, INTERNAL, request_id)
            return None
        stripped = [
            pair
            for pair in raw_headers
            if pair[0].lower()
            not in {
                b"cache-control",
                b"content-length",
                b"content-security-policy",
                b"cross-origin-resource-policy",
                b"permissions-policy",
                b"referrer-policy",
                b"x-content-type-options",
                b"x-frame-options",
                b"x-request-id",
            }
        ]
        stripped.extend(_SECURITY_HEADERS)
        stripped.append((b"content-length", str(len(body)).encode("ascii")))
        stripped.append((b"x-request-id", request_id.encode("ascii")))
        await send(
            {
                "type": "http.response.start",
                "status": start["status"],
                "headers": stripped,
            }
        )
        await send({"type": "http.response.body", "body": bytes(body)})
        return None

    @staticmethod
    async def _send_problem(send: Send, spec: ProblemSpec, request_id: str) -> None:
        document = build_problem(spec, request_id)
        payload = document.canonical_json_bytes(maximum_bytes=4_096)
        headers = [
            (b"content-length", str(len(payload)).encode("ascii")),
            (b"content-type", _PROBLEM_MEDIA_TYPE),
            *_SECURITY_HEADERS,
            (b"x-request-id", request_id.encode("ascii")),
            (b"connection", b"close"),
        ]
        if spec.retry_after_seconds is not None:
            headers.append((b"retry-after", str(spec.retry_after_seconds).encode("ascii")))
        await send({"type": "http.response.start", "status": spec.status, "headers": headers})
        await send({"type": "http.response.body", "body": payload})
