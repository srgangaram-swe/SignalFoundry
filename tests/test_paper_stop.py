"""Emergency admission stays bounded and reaches the real persistent service."""

import asyncio
import json

import pytest
from fastapi.testclient import TestClient
from starlette.responses import JSONResponse

from signal_foundry.api import create_app
from signal_foundry.http_security import LocalBoundary
from signal_foundry.manager import Manager
from signal_foundry.store import Store
from signal_foundry.trading.service import Action
from tests.research_helpers import FakeRunner
from tests.test_http_admission import scope
from tests.test_paper_integration import configured as configured
from tests.test_research_api import HEADERS


def test_stop_reaches_persistent_service_under_saturation(configured):
    async def scenario():
        release = asyncio.Event()
        release_stop = asyncio.Event()
        full = asyncio.Event()
        stopped = asyncio.Event()
        entered = 0

        async def app(request, receive, send):
            nonlocal entered
            assert (await receive())["body"] == b"{}"
            if request["path"] == "/api/v1/paper/stop":
                result = configured.action(Action(operation="stop")).wire()
                stopped.set()
                await release_stop.wait()
            else:
                entered += 1
                if entered == 8:
                    full.set()
                await release.wait()
                result = {"ok": True}
            await JSONResponse(result)(request, receive, send)

        boundary = LocalBoundary(app, port=8765)

        async def request(path="/ordinary", foreign=False):
            incoming = scope()
            incoming["path"] = path
            if foreign:
                incoming["headers"].append((b"origin", b"https://evil.example"))
            output = []

            async def receive():
                return {"type": "http.request", "body": b"{}", "more_body": False}

            async def send(value):
                output.append(value)

            await boundary(incoming, receive, send)
            return output

        ordinary = [asyncio.create_task(request()) for _ in range(8)]
        await asyncio.wait_for(full.wait(), 2)
        assert (await request())[0]["status"] == 429
        emergency = asyncio.create_task(request("/api/v1/paper/stop"))
        await asyncio.wait_for(stopped.wait(), 2)
        assert configured.status().stopped
        assert boundary._active == 8 and boundary._stop_active == 1
        assert (await request("/api/v1/paper/stop"))[0]["status"] == 429
        assert (await request("/api/v1/paper/stop", foreign=True))[0]["status"] == 403
        release_stop.set()
        response = await asyncio.wait_for(emergency, 2)
        assert json.loads(response[1]["body"])["status"]["stopped"]
        assert (await request("/api/v1/paper/stop"))[0]["status"] == 200
        release.set()
        await asyncio.wait_for(asyncio.gather(*ordinary), 2)
        assert boundary._active == boundary._stop_active == 0

    asyncio.run(scenario())


def test_real_stop_http_route_keeps_closed_input_and_origin_policy(
    configured, tmp_path
):
    manager = Manager(Store(tmp_path / "research"), FakeRunner())
    with TestClient(
        create_app(lambda: manager, paper_factory=lambda: configured),
        base_url="http://127.0.0.1:8765",
    ) as http:
        for value in ({"operation": "cycle"}, {"url": "https://evil.example"}, []):
            assert (
                http.post("/api/v1/paper/stop", headers=HEADERS, json=value).status_code
                == 422
            )
        assert not configured.status().stopped
        assert (
            http.post(
                "/api/v1/paper/stop",
                headers={**HEADERS, "Origin": "https://evil.example"},
                json={},
            ).status_code
            == 403
        )
        result = http.post("/api/v1/paper/stop", headers=HEADERS, json={})
        assert result.status_code == 200 and result.json()["status"]["stopped"]
        assert http.post("/api/v1/paper/stop", headers=HEADERS, json={}).json()[
            "status"
        ]["stopped"]


@pytest.mark.parametrize("body", [b"{", b"x" * 20000])
def test_reserved_slot_released_after_body_refusal(body):
    async def scenario():
        async def forbidden(scope, receive, send):
            raise AssertionError("Malformed stop reached application")

        boundary = LocalBoundary(forbidden, port=8765)
        output = []
        incoming = scope()
        incoming["path"] = "/api/v1/paper/stop"

        async def receive():
            return {"type": "http.request", "body": body, "more_body": False}

        async def send(message):
            output.append(message)

        await boundary(incoming, receive, send)
        assert output[0]["status"] in {413, 422}
        assert boundary._stop_active == boundary._active == 0

    asyncio.run(scenario())
