"""Public HTTP contract, browser abuse and guided research lifecycle tests."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

from signal_foundry.api import create_app
from signal_foundry.contracts import JobState, ResearchRequest
from signal_foundry.manager import Manager
from signal_foundry.store import Store
from tests.research_helpers import FakeRunner

HEADERS = {"X-Signal-Foundry-Client": "nexus", "Content-Type": "application/json"}


@pytest.fixture
def client(tmp_path: Path):
    runner = FakeRunner()
    manager = Manager(Store(tmp_path / "state"), runner)
    with TestClient(
        create_app(lambda: manager), base_url="http://127.0.0.1:8765"
    ) as client:
        yield client, manager, runner


def test_guided_lifecycle_and_typed_contract(client) -> None:
    http, manager, runner = client
    assert http.get("/api/v1/catalog").json()["live_readiness"] == "NOT_READY"
    config = ResearchRequest().model_dump(mode="json")
    validation = http.post("/api/v1/validate", headers=HEADERS, json=config)
    assert validation.status_code == 200, validation.text
    assert validation.json()["request_hash"] == ResearchRequest().digest()
    response = http.post(
        "/api/v1/jobs", headers={**HEADERS, "Idempotency-Key": "a" * 16}, json=config
    )
    assert response.status_code == 202, response.text
    identity = response.json()["job_id"]
    assert manager.wait(identity, 5).state == JobState.SUCCEEDED
    assert http.get(f"/api/v1/jobs/{identity}").json()["state"] == "succeeded"
    assert len(http.get("/api/v1/jobs").json()["jobs"]) == 1
    assert (
        http.get(f"/api/v1/jobs/{identity}/evidence").json()["mode"]
        == "development_simulation"
    )
    assert len(http.get(f"/api/v1/jobs/{identity}/audit").json()["events"]) == 3
    assert http.get(f"/api/v1/compare/{identity}/{identity}").json()["compatible"]
    assert (
        http.post(f"/api/v1/jobs/{identity}/cancel", headers=HEADERS, json={}).json()[
            "state"
        ]
        == "succeeded"
    )
    schema = http.get("/api/v1/openapi.json").json()
    assert "ResearchRequest" in schema["components"]["schemas"]
    assert all(
        "broker" not in route and "live" not in route for route in schema["paths"]
    )
    assert runner.calls == 1


@pytest.mark.parametrize(
    "headers,status,code",
    [
        ({"Host": "evil.example"}, 403, "foreign_host"),
        ({"Origin": "https://evil.example"}, 403, "foreign_origin"),
        ({"Origin": "null"}, 403, "foreign_origin"),
        ({"Sec-Fetch-Site": "cross-site"}, 403, "cross_site"),
        ({"X-Signal-Foundry-Client": "wrong"}, 403, "client_header"),
        ({"Content-Type": "text/plain"}, 415, "content_type"),
        ({"Content-Length": "999999"}, 413, "payload_size"),
        ({"Content-Length": "abc"}, 400, "content_length"),
        ({"Content-Length": "2"}, 400, "content_length"),
    ],
)
def test_reject_browser_and_length_abuse(
    client, headers: dict[str, str], status: int, code: str
) -> None:
    http, _, _ = client
    result = http.post(
        "/api/v1/validate", headers={**HEADERS, **headers}, content=b'{"seed":42}'
    )
    assert result.status_code == status
    assert result.json()["code"] == code
    assert result.headers["x-content-type-options"] == "nosniff"
    assert result.headers["cache-control"] == "no-store"
    assert "access-control-allow-origin" not in result.headers
    assert "evil.example" not in result.text


@pytest.mark.parametrize(
    "payload,status",
    [
        (b"{}", 200),
        (b'{"seed":true}', 422),
        (b'{"seed":1,"seed":2}', 422),
        (b'{"seed":NaN}', 422),
        (b"[1]", 422),
        (b"x" * 16385, 413),
    ],
)
def test_malformed_json_never_reaches_research(
    client, payload: bytes, status: int
) -> None:
    http, _, runner = client
    result = http.post("/api/v1/validate", headers=HEADERS, content=payload)
    assert result.status_code == status, result.text
    assert runner.calls == 0


def test_errors_do_not_reflect_input_and_no_live_route(client) -> None:
    http, _, _ = client
    response = http.post(
        "/api/v1/jobs", headers=HEADERS, json={"private_key": "MARKER_SECRET"}
    )
    assert response.status_code == 422
    assert "MARKER_SECRET" not in response.text
    assert http.get("/api/v1/jobs/no-such-job").status_code == 404
    assert http.get("/api/v1/broker/orders").status_code == 404
    assert http.delete("/api/v1/jobs").status_code == 405
    assert http.request("GET", "/api/v1/jobs", content=b"bad").status_code == 400
    assert http.post("/api/v1/validate", json={}).status_code == 403


@pytest.mark.parametrize(
    "name,values",
    [
        ("Host", ["127.0.0.1:8765", "evil.example"]),
        ("Origin", ["http://127.0.0.1:8765", "http://localhost:8765"]),
        ("Content-Length", ["2", "2"]),
    ],
)
def test_duplicate_security_headers_fail_closed(
    client, name: str, values: list[str]
) -> None:
    http, _, _ = client
    headers = [*HEADERS.items(), *((name, value) for value in values)]
    result = http.post("/api/v1/validate", headers=headers, content=b"{}")
    assert result.status_code in {400, 403}


def test_schema_is_deterministic_without_acquiring_state() -> None:
    def forbidden() -> Any:
        raise AssertionError("schema generation must not acquire state")

    assert create_app(forbidden).openapi() == create_app(forbidden).openapi()


def test_unexpected_errors_are_structured_without_payload_or_trace(
    client, monkeypatch, caplog
) -> None:
    http, _, runner = client

    def fail():
        raise RuntimeError("MARKER_SECRET")

    monkeypatch.setattr(runner, "catalog", fail)
    result = http.get("/api/v1/catalog")
    assert result.status_code == 500
    assert result.json()["code"] == "internal_error"
    assert result.headers["x-content-type-options"] == "nosniff"
    assert "MARKER_SECRET" not in result.text + caplog.text
    assert "RuntimeError" in caplog.text


def test_cancel_rejects_extra_body_fields(client) -> None:
    http, _, _ = client
    result = http.post(
        "/api/v1/jobs/unknown/cancel", headers=HEADERS, json={"secret": "MARKER_SECRET"}
    )
    assert result.status_code == 422
    assert "MARKER_SECRET" not in result.text
