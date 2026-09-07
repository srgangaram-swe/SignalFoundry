"""Pinned h11 adapter for structured parser failures and ASGI-owned admission.

Uvicorn 0.52.1 emits its own plain-text 400/503 responses before the ASGI
application can apply Signalattice's privacy, telemetry, and problem-detail
contracts.  This isolated adapter keeps Uvicorn's parser and lifecycle logic,
corrects its inclusive concurrency decision at the configured transport bound,
and replaces both saturation and malformed-request output with bounded,
non-reflective RFC 9457 responses.

The compatibility surface is intentionally small and is guarded by real-socket
tests.  Revalidating it is mandatory before changing the pinned Uvicorn minor
line.
"""

from __future__ import annotations

import asyncio
import secrets
from collections.abc import Awaitable, Callable
from typing import Any, Final

from uvicorn.config import Config
from uvicorn.protocols.http.h11_impl import H11Protocol
from uvicorn.server import ServerState

from quant_platform.service.problems import BAD_REQUEST, SATURATED, build_problem
from quant_platform.service.telemetry import (
    RequestTimer,
    ServiceTelemetry,
    TelemetryRuntimeError,
)
from quant_platform.service.telemetry_contracts import (
    DropReason,
    Outcome,
    RejectionReason,
    TelemetryChannel,
    TelemetryContractError,
)

_MAX_PROTOCOL_RESPONSE_BYTES: Final = 8 * 1024
_PROBLEM_MEDIA_TYPE: Final = b"application/problem+json"
_ASGI_SECURITY_HEADERS: Final = (
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
_STATIC_SECURITY_HEADERS: Final = (
    b"cache-control: no-store\r\n"
    b"content-security-policy: default-src 'none'; base-uri 'none'; "
    b"form-action 'none'; frame-ancestors 'none'\r\n"
    b"cross-origin-resource-policy: same-origin\r\n"
    b"permissions-policy: accelerometer=(), camera=(), geolocation=(), microphone=()\r\n"
    b"referrer-policy: no-referrer\r\n"
    b"x-content-type-options: nosniff\r\n"
    b"x-frame-options: DENY\r\n"
)

type Scope = dict[str, Any]
type Message = dict[str, Any]
type Receive = Callable[[], Awaitable[Message]]
type Send = Callable[[Message], Awaitable[None]]


class SignalatticeH11Protocol(H11Protocol):
    """H11 protocol with exact transport admission and structured failures."""

    def __init__(
        self,
        config: Config,
        server_state: ServerState,
        app_state: dict[str, Any],
        _loop: asyncio.AbstractEventLoop | None = None,
        *,
        telemetry: ServiceTelemetry,
    ) -> None:
        if type(telemetry) is not ServiceTelemetry:
            raise TypeError("telemetry must be a ServiceTelemetry")
        self._signalattice_telemetry = telemetry
        super().__init__(config, server_state, app_state, _loop)

    def handle_events(self) -> None:
        """Apply an exact transport gate without Uvicorn's raw responder.

        Uvicorn includes the current connection in ``connections`` and uses a
        greater-than-or-equal comparison, so its native responder admits only
        ``limit - 1`` connections and emits uninstrumented ``text/plain``.  The
        pinned adapter instead admits exactly ``limit`` normal parsed request
        exchanges and routes the next parsed request through a canonical ASGI
        overload responder.  The temporary mutations are synchronous and
        per-protocol; the selected bound method is captured by Uvicorn's newly
        created rejection task before the original application and configured
        limit are restored.
        """

        configured_app = self.app
        configured_limit = self.limit_concurrency
        if self._transport_is_saturated():
            self.app = self._send_saturation_response
        self.limit_concurrency = None
        try:
            super().handle_events()
        finally:
            self.app = configured_app
            self.limit_concurrency = configured_limit

    def _transport_is_saturated(self) -> bool:
        """Return whether the next request would exceed the configured limit.

        Uvicorn re-enters ``handle_events`` from ``on_response_complete`` for a
        buffered keep-alive request before the retiring task's done callback
        removes it from ``server_state.tasks``. Discount only that exact current
        task so a next exchange that becomes number ``limit`` is not rejected
        as number ``limit + 1``.
        """

        limit = self.limit_concurrency
        if limit is None:
            return False
        current = asyncio.current_task()
        active_tasks = len(self.tasks) - (1 if current is not None and current in self.tasks else 0)
        return len(self.connections) > limit or active_tasks >= limit

    async def _send_saturation_response(
        self,
        scope: Scope,
        _receive: Receive,
        send: Send,
    ) -> None:
        """Reject one excess transport exchange with bounded telemetry."""

        timer = self._begin_request(scope.get("path"))
        request_id = secrets.token_hex(16)
        payload = build_problem(SATURATED, request_id).canonical_json_bytes(maximum_bytes=4_096)
        headers = [
            (b"content-length", str(len(payload)).encode("ascii")),
            (b"content-type", _PROBLEM_MEDIA_TYPE),
            *_ASGI_SECURITY_HEADERS,
            (b"x-request-id", request_id.encode("ascii")),
            (b"retry-after", b"1"),
            (b"connection", b"close"),
        ]
        try:
            await send({"type": "http.response.start", "status": 429, "headers": headers})
            await send({"type": "http.response.body", "body": payload})
        finally:
            if timer is not None:
                try:
                    self._signalattice_telemetry.finish_request(
                        timer,
                        outcome=Outcome.REJECTED,
                        status_code=429,
                        response_bytes=len(payload),
                        rejection=RejectionReason.GLOBAL_CONCURRENCY,
                    )
                except (TelemetryContractError, TelemetryRuntimeError):
                    self._record_telemetry_failure()

    def send_400_response(self, _message: str) -> None:
        """Emit one canonical bounded 400 without retaining parser diagnostics."""

        timer = self._begin_request(None)
        request_id = secrets.token_hex(16)
        payload = build_problem(BAD_REQUEST, request_id).canonical_json_bytes(maximum_bytes=4_096)
        response = b"".join(
            (
                b"HTTP/1.1 400 Bad Request\r\n",
                b"content-length: " + str(len(payload)).encode("ascii") + b"\r\n",
                b"content-type: application/problem+json\r\n",
                _STATIC_SECURITY_HEADERS,
                b"x-request-id: " + request_id.encode("ascii") + b"\r\n",
                b"connection: close\r\n\r\n",
                payload,
            )
        )
        if len(response) > _MAX_PROTOCOL_RESPONSE_BYTES:
            self.transport.close()
            self._record_telemetry_failure()
            return
        self.transport.write(response)
        self.transport.close()
        if timer is not None:
            try:
                self._signalattice_telemetry.finish_request(
                    timer,
                    outcome=Outcome.REJECTED,
                    status_code=400,
                    response_bytes=len(payload),
                    rejection=RejectionReason.INVALID_METADATA,
                )
            except (TelemetryContractError, TelemetryRuntimeError):
                self._record_telemetry_failure()

    def _begin_request(self, raw_path: object) -> RequestTimer | None:
        try:
            return self._signalattice_telemetry.begin_request(raw_path)
        except (TelemetryContractError, TelemetryRuntimeError):
            self._record_telemetry_failure()
            return None

    def _record_telemetry_failure(self) -> None:
        for channel in TelemetryChannel:
            try:
                self._signalattice_telemetry.metrics.record_drop(
                    channel,
                    DropReason.INVALID_RECORD,
                )
            except TelemetryContractError:
                # The response boundary must remain available even if the fixed
                # metrics implementation itself violates its internal contract.
                return


def configured_h11_protocol(
    telemetry: ServiceTelemetry,
) -> type[SignalatticeH11Protocol]:
    """Bind one telemetry instance to Uvicorn's supported protocol-class hook."""

    if type(telemetry) is not ServiceTelemetry:
        raise TypeError("telemetry must be a ServiceTelemetry")

    class ConfiguredSignalatticeH11Protocol(SignalatticeH11Protocol):
        def __init__(
            self,
            config: Config,
            server_state: ServerState,
            app_state: dict[str, Any],
            _loop: asyncio.AbstractEventLoop | None = None,
        ) -> None:
            super().__init__(
                config,
                server_state,
                app_state,
                _loop,
                telemetry=telemetry,
            )

    ConfiguredSignalatticeH11Protocol.__name__ = "ConfiguredSignalatticeH11Protocol"
    ConfiguredSignalatticeH11Protocol.__qualname__ = "ConfiguredSignalatticeH11Protocol"
    return ConfiguredSignalatticeH11Protocol
