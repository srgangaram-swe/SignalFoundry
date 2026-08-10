"""Raw ASGI trust boundary for bounded loopback-only HTTP reads.

The middleware sits outside FastAPI so hostile Host values, unsupported
methods, oversized metadata, request bodies, proxy hints, application failures,
and response-size overruns all receive the same redacted RFC 9457 contract.
Responses are buffered only up to a fixed ceiling; this API has no streaming or
download route.
"""

from __future__ import annotations

import asyncio
import logging
import secrets
from collections.abc import Awaitable, Callable
from typing import Any, Final

from quant_platform.service.problems import (
    BAD_REQUEST,
    INTERNAL,
    METHOD_NOT_ALLOWED,
    PAYLOAD_TOO_LARGE,
    SATURATED,
    ProblemSpec,
    build_problem,
)

type Scope = dict[str, Any]
type Message = dict[str, Any]
type Receive = Callable[[], Awaitable[Message]]
type Send = Callable[[Message], Awaitable[None]]
type ASGIApp = Callable[[Scope, Receive, Send], Awaitable[None]]

_LOGGER = logging.getLogger(__name__)
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


class SecurityBoundaryMiddleware:
    """Enforce the local service's method, metadata, capacity, and output bounds."""

    def __init__(
        self,
        app: ASGIApp,
        *,
        max_concurrency: int = 32,
        max_headers: int = 64,
        max_header_bytes: int = 8 * 1024,
        max_query_bytes: int = 4 * 1024,
        max_path_bytes: int = 512,
        max_response_bytes: int = 1024 * 1024,
        allowed_port: int | None = None,
    ) -> None:
        bounds = {
            "max_concurrency": (max_concurrency, 1, 1_024),
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
        self._app = app
        self._max_concurrency = max_concurrency
        self._max_headers = max_headers
        self._max_header_bytes = max_header_bytes
        self._max_query_bytes = max_query_bytes
        self._max_path_bytes = max_path_bytes
        self._max_response_bytes = max_response_bytes
        self._allowed_port = allowed_port
        self._active = 0
        self._active_lock = asyncio.Lock()

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        """Validate one ASGI exchange and emit a bounded secured response."""

        if scope.get("type") != "http":
            await self._app(scope, receive, send)
            return
        request_id = secrets.token_hex(16)
        validation_problem = self._validate_scope(scope)
        if validation_problem is not None:
            await self._send_problem(send, validation_problem, request_id)
            return
        if not await self._try_enter():
            await self._send_problem(send, SATURATED, request_id)
            return
        state = scope.setdefault("state", {})
        if type(state) is not dict:
            await self._leave()
            await self._send_problem(send, BAD_REQUEST, request_id)
            return
        state["signalattice_request_id"] = request_id
        scope["headers"] = [
            pair for pair in scope.get("headers", ()) if pair[0].lower() != b"x-request-id"
        ]
        try:
            await self._call_bounded(scope, receive, send, request_id)
        finally:
            await self._leave()

    async def _try_enter(self) -> bool:
        async with self._active_lock:
            if self._active >= self._max_concurrency:
                return False
            self._active += 1
            return True

    async def _leave(self) -> None:
        async with self._active_lock:
            if self._active <= 0:
                raise RuntimeError("service concurrency accounting underflow")
            self._active -= 1

    def _validate_scope(self, scope: Scope) -> ProblemSpec | None:
        method = scope.get("method")
        if method != "GET":
            return METHOD_NOT_ALLOWED
        raw_path = scope.get("raw_path", b"")
        query = scope.get("query_string", b"")
        headers = scope.get("headers", ())
        if (
            type(raw_path) is not bytes
            or not 1 <= len(raw_path) <= self._max_path_bytes
            or not raw_path.isascii()
            or type(query) is not bytes
            or len(query) > self._max_query_bytes
            or type(headers) is not list
            or not 1 <= len(headers) <= self._max_headers
        ):
            return BAD_REQUEST
        header_bytes = 0
        normalized: dict[bytes, list[bytes]] = {}
        for pair in headers:
            if (
                type(pair) is not tuple
                or len(pair) != 2
                or type(pair[0]) is not bytes
                or type(pair[1]) is not bytes
            ):
                return BAD_REQUEST
            name, value = pair
            header_bytes += len(name) + len(value)
            if (
                not name
                or not name.isascii()
                or not value.isascii()
                or b"\r" in value
                or b"\n" in value
            ):
                return BAD_REQUEST
            normalized.setdefault(name.lower(), []).append(value)
        if header_bytes > self._max_header_bytes or _PROXY_HEADERS.intersection(normalized):
            return BAD_REQUEST
        if not self._valid_host(normalized.get(b"host"), allowed_port=self._allowed_port):
            return BAD_REQUEST
        if (
            b"transfer-encoding" in normalized
            or b"expect" in normalized
            or b"content-encoding" in normalized
        ):
            return BAD_REQUEST
        if b"x-request-id" in normalized:
            return BAD_REQUEST
        content_lengths = normalized.get(b"content-length", [])
        if len(content_lengths) > 1:
            return BAD_REQUEST
        if content_lengths and content_lengths[0] != b"0":
            return PAYLOAD_TOO_LARGE
        if query.count(b"&") >= 32:
            return BAD_REQUEST
        return None

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
    ) -> None:
        start: Message | None = None
        body = bytearray()
        completed = False

        async def capture(message: Message) -> None:
            nonlocal start, completed
            message_type = message.get("type")
            if message_type == "http.response.start":
                if start is not None or completed:
                    raise RuntimeError("application emitted duplicate response start")
                start = message
                return
            if message_type != "http.response.body" or start is None or completed:
                raise RuntimeError("application emitted an invalid HTTP response sequence")
            chunk = message.get("body", b"")
            if type(chunk) is not bytes or len(chunk) > self._max_response_bytes - len(body):
                raise RuntimeError("application response exceeded its byte ceiling")
            body.extend(chunk)
            if not bool(message.get("more_body", False)):
                completed = True

        try:
            await self._app(scope, receive, capture)
            if start is None or not completed:
                raise RuntimeError("application did not complete one bounded HTTP response")
        except Exception as error:
            _LOGGER.error(
                "read service request failed request_id=%s error_class=%s",
                request_id,
                type(error).__name__,
            )
            await self._send_problem(send, INTERNAL, request_id)
            return
        raw_headers = start.get("headers", [])
        if type(raw_headers) is not list or len(raw_headers) > self._max_headers:
            await self._send_problem(send, INTERNAL, request_id)
            return
        response_header_bytes = 0
        for pair in raw_headers:
            if (
                type(pair) is not tuple
                or len(pair) != 2
                or type(pair[0]) is not bytes
                or type(pair[1]) is not bytes
            ):
                await self._send_problem(send, INTERNAL, request_id)
                return
            name, value = pair
            response_header_bytes += len(name) + len(value)
            normalized_name = name.lower()
            if (
                not name
                or not name.isascii()
                or b"\r" in value
                or b"\n" in value
                or normalized_name == b"set-cookie"
                or normalized_name.startswith(b"access-control-")
            ):
                await self._send_problem(send, INTERNAL, request_id)
                return
        if response_header_bytes > self._max_header_bytes:
            await self._send_problem(send, INTERNAL, request_id)
            return
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
                "status": start.get("status", 500),
                "headers": stripped,
            }
        )
        await send({"type": "http.response.body", "body": bytes(body)})

    @staticmethod
    async def _send_problem(send: Send, spec: ProblemSpec, request_id: str) -> None:
        document = build_problem(spec, request_id)
        payload = document.canonical_json_bytes(maximum_bytes=4_096)
        headers = [
            (b"content-length", str(len(payload)).encode("ascii")),
            (b"content-type", _PROBLEM_MEDIA_TYPE),
            *_SECURITY_HEADERS,
            (b"x-request-id", request_id.encode("ascii")),
        ]
        if spec.retry_after_seconds is not None:
            headers.append((b"retry-after", str(spec.retry_after_seconds).encode("ascii")))
        await send({"type": "http.response.start", "status": spec.status, "headers": headers})
        await send({"type": "http.response.body", "body": payload})
