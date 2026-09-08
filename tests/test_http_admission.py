"""Deterministic ASGI concurrency, incremental-body and disconnect probes."""

from __future__ import annotations

import asyncio
import json

import pytest
from starlette.responses import JSONResponse

from signal_foundry.http_security import LocalBoundary


def scope():
    return {
        "type": "http",
        "method": "POST",
        "path": "/test",
        "headers": [
            (b"host", b"127.0.0.1:8765"),
            (b"content-type", b"application/json"),
            (b"x-signal-foundry-client", b"nexus"),
        ],
    }


async def invoke(boundary, messages=None):
    incoming = iter(
        messages or [{"type": "http.request", "body": b"{}", "more_body": False}]
    )
    responses = []

    async def receive():
        return next(incoming)

    async def send(message):
        responses.append(message)

    await boundary(scope(), receive, send)
    return responses


def test_ninth_request_is_rejected_while_eight_are_admitted() -> None:
    async def scenario() -> None:
        release = asyncio.Event()
        ready = asyncio.Event()
        entered = 0

        async def app(scope, receive, send):
            nonlocal entered
            entered += 1
            if entered == 8:
                ready.set()
            assert (await receive())["body"] == b"{}"
            await release.wait()
            await JSONResponse({"ok": True})(scope, receive, send)

        boundary = LocalBoundary(app, port=8765)
        tasks = [asyncio.create_task(invoke(boundary)) for _ in range(8)]
        await asyncio.wait_for(ready.wait(), 2)
        rejected = await invoke(boundary)
        assert rejected[0]["status"] == 429
        assert json.loads(rejected[1]["body"])["code"] == "http_capacity"
        release.set()
        results = await asyncio.wait_for(asyncio.gather(*tasks), 2)
        assert all(result[0]["status"] == 200 for result in results)
        assert boundary._active == 0

    asyncio.run(scenario())


@pytest.mark.parametrize(
    "messages,status,code",
    [
        ([{"type": "http.disconnect"}], 400, "client_disconnected"),
        (
            [
                {"type": "http.request", "body": b"x" * 10000, "more_body": True},
                {"type": "http.request", "body": b"x" * 10000},
            ],
            413,
            "payload_size",
        ),
    ],
)
def test_incomplete_or_oversized_body_never_reaches_app(messages, status, code) -> None:
    async def forbidden(scope, receive, send):
        raise AssertionError("invalid body reached application")

    result = asyncio.run(invoke(LocalBoundary(forbidden, port=8765), messages))
    assert result[0]["status"] == status
    assert json.loads(result[1]["body"])["code"] == code


def test_slow_body_timeout_is_bounded(monkeypatch) -> None:
    original = asyncio.timeout

    def short_timeout(seconds):
        assert seconds == 5
        return original(0.001)

    monkeypatch.setattr(asyncio, "timeout", short_timeout)

    async def scenario() -> None:
        never = asyncio.Event()
        responses = []

        async def receive():
            await never.wait()

        async def send(message):
            responses.append(message)

        async def forbidden(scope, receive, send):
            raise AssertionError("slow body was admitted")

        await LocalBoundary(forbidden, port=8765)(scope(), receive, send)
        assert responses[0]["status"] == 408
        assert json.loads(responses[1]["body"])["code"] == "body_timeout"

    asyncio.run(scenario())
