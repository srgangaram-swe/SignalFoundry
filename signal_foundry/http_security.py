"""ASGI loopback/browser boundary with bounded body and admission.

This is not authentication against another local process. No remote bind or CORS
opt-out is offered. Middleware instances own their counters on one event loop.
"""

from __future__ import annotations

import asyncio
import logging

from starlette.datastructures import Headers, MutableHeaders
from starlette.responses import JSONResponse
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from signal_foundry.boundary import MAX_REQUEST_BYTES, FoundryError, decode
from signal_foundry.contracts import Problem

LOGGER = logging.getLogger(__name__)

SECURITY_HEADERS = {
    "content-security-policy": (
        "default-src 'none'; script-src 'self'; style-src 'self'; img-src 'self';"
        " connect-src 'self'; font-src 'self'; object-src 'none'; base-uri 'none';"
        " frame-ancestors 'none'; form-action 'none'; worker-src 'none'"
    ),
    "x-content-type-options": "nosniff",
    "referrer-policy": "no-referrer",
    "cache-control": "no-store",
    "permissions-policy": "camera=(), microphone=(), geolocation=()",
    "cross-origin-opener-policy": "same-origin",
    "cross-origin-resource-policy": "same-origin",
    "x-frame-options": "DENY",
}


class LocalBoundary:
    """Reject foreign origins, duplicates, oversized/slow payloads and overload."""

    def __init__(self, app: ASGIApp, *, port: int) -> None:
        self.app = app
        self.hosts = frozenset({f"127.0.0.1:{port}", f"localhost:{port}"})
        self.origins = frozenset("http://" + host for host in self.hosts)
        self._active = 0
        self._stop_active = 0

    def _headers(self, scope: Scope) -> None:
        headers = Headers(scope=scope)
        if len(headers.getlist("host")) != 1 or headers["host"] not in self.hosts:
            raise FoundryError(
                "foreign_host", "Only the configured loopback host is accepted.", 403
            )
        if len(headers.getlist("origin")) > 1 or (
            "origin" in headers and headers["origin"] not in self.origins
        ):
            raise FoundryError(
                "foreign_origin", "Cross-origin requests are not accepted.", 403
            )
        if headers.get("sec-fetch-site") not in {None, "none", "same-origin"}:
            raise FoundryError(
                "cross_site", "Cross-site browser requests are not accepted.", 403
            )
        if scope["method"] not in {"GET", "HEAD", "POST"}:
            raise FoundryError(
                "method_not_allowed",
                "This interface permits read or explicit research requests only.",
                405,
            )
        if scope["method"] == "POST":
            if headers.getlist("x-signal-foundry-client") != ["nexus"]:
                raise FoundryError(
                    "client_header",
                    "An explicit local research client header is required.",
                    403,
                )
            if headers.getlist("content-type") != ["application/json"]:
                raise FoundryError(
                    "content_type", "Use application/json for research mutations.", 415
                )
        lengths = headers.getlist("content-length")
        if len(lengths) > 1 or (
            lengths
            and (
                not lengths[0].isascii()
                or not lengths[0].isdigit()
                or len(lengths[0]) > 8
            )
        ):
            raise FoundryError("content_length", "Invalid content length.", 400)
        if lengths and int(lengths[0]) > MAX_REQUEST_BYTES:
            raise FoundryError(
                "payload_size", "Request exceeds the body byte limit.", 413
            )

    async def _body(self, scope: Scope, receive: Receive) -> bytes:
        body = bytearray()
        try:
            async with asyncio.timeout(5):
                while True:
                    message = await receive()
                    if message["type"] == "http.disconnect":
                        raise FoundryError(
                            "client_disconnected",
                            "Client disconnected before request admission.",
                            400,
                        )
                    chunk = message.get("body", b"")
                    if len(body) + len(chunk) > MAX_REQUEST_BYTES:
                        raise FoundryError(
                            "payload_size", "Request exceeds the body byte limit.", 413
                        )
                    body.extend(chunk)
                    if not message.get("more_body", False):
                        break
        except TimeoutError as exc:
            raise FoundryError(
                "body_timeout", "Request body exceeded its time budget.", 408
            ) from exc
        length = Headers(scope=scope).get("content-length")
        if length is not None and int(length) != len(body):
            raise FoundryError(
                "content_length", "Request length does not match its body.", 400
            )
        if scope["method"] == "POST":
            decode(bytes(body))
        elif body:
            raise FoundryError(
                "unexpected_body", "Read requests do not accept bodies.", 400
            )
        return bytes(body)

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        started = False

        async def secured(message: Message) -> None:
            nonlocal started
            if message["type"] == "http.response.start":
                started = True
                headers = MutableHeaders(scope=message)
                for key, value in SECURITY_HEADERS.items():
                    headers[key] = value
            await send(message)

        admitted = False
        emergency = scope["method"] == "POST" and scope["path"] == "/api/v1/paper/stop"
        try:
            self._headers(scope)
            occupied = self._stop_active >= 1 if emergency else self._active >= 8
            if occupied:
                raise FoundryError(
                    "http_capacity",
                    "Local request capacity is occupied; retry later.",
                    429,
                )
            if emergency:
                self._stop_active += 1
            else:
                self._active += 1
            admitted = True
            body = await self._body(scope, receive)
            consumed = False

            async def replay() -> Message:
                nonlocal consumed
                if not consumed:
                    consumed = True
                    return {"type": "http.request", "body": body, "more_body": False}
                return await receive()

            await self.app(scope, replay, secured)
        except FoundryError as exc:
            response = JSONResponse(
                Problem(code=exc.code, detail=exc.detail).model_dump(),
                status_code=exc.status,
            )
            await response(scope, receive, secured)
        except Exception as exc:
            # HTTP ownership boundary only. Never reflect/log exception payloads
            # from an unexpected failure; explicit status and cause class remain
            # observable. Mid-response transport failures cannot send new headers.
            LOGGER.error("HTTP operation failed (%s)", type(exc).__name__)
            if started:
                raise FoundryError(
                    "response_interrupted", "Response transport was interrupted.", 500
                ) from None
            response = JSONResponse(
                Problem(
                    code="internal_error",
                    detail=(
                        "The operation failed; no successful research outcome is"
                        " implied."
                    ),
                ).model_dump(),
                status_code=500,
            )
            await response(scope, receive, secured)
        finally:
            if admitted:
                if emergency:
                    self._stop_active -= 1
                else:
                    self._active -= 1
