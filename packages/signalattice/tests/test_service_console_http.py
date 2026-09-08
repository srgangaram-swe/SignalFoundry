"""HTTP-boundary tests for the console and governance reads (SF-S5-SL-MR6).

These drive the real ASGI stack -- admission control, the security boundary, the
console handler, and the FastAPI routes -- rather than calling handlers
directly, because the properties that matter here are properties of the
composed boundary.

Asserted:

* Console assets are reachable by GET and HEAD, and by nothing else.
* The JSON API stays GET-only; HEAD there is refused.
* Every mutation verb is refused everywhere, with no route to override it.
* Governance evidence is projected read-only, and a browser is given no field
  it could use to promote anything.
"""

from __future__ import annotations

import sqlite3
from collections.abc import Iterator
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

from quant_platform.governance.lane import GovernanceLane
from quant_platform.governance.read_ports import GovernanceReadPorts
from quant_platform.governance.store import EventKind, GovernanceStore
from quant_platform.service.api import assert_read_only_route_inventory, create_app
from quant_platform.service.console import CONSOLE_CSP, load_console_bundle
from quant_platform.tracking import migrations as migrations_module

BASE = datetime(2026, 8, 1, tzinfo=UTC)
CHAMPION = "a" * 64
DIGEST = "c" * 64


class _StubEvidencePorts:
    """Minimal structural stand-in for the evidence read contract.

    The console and governance surfaces under test never touch run evidence, so
    a stub keeps this module focused on the boundary rather than re-building a
    registry fixture that other suites already cover.
    """

    def probe_evidence_readiness(self) -> Any:  # pragma: no cover - not exercised here
        raise NotImplementedError

    def get_run(self, run_id: str) -> Any:  # pragma: no cover
        raise NotImplementedError

    def list_runs(self, *args: Any, **kwargs: Any) -> Any:  # pragma: no cover
        raise NotImplementedError

    def get_artifact(self, digest: str) -> Any:  # pragma: no cover
        raise NotImplementedError

    def list_run_artifacts(self, *args: Any, **kwargs: Any) -> Any:  # pragma: no cover
        raise NotImplementedError

    def read_verified_manifest(self, *args: Any, **kwargs: Any) -> Any:  # pragma: no cover
        raise NotImplementedError


def _bundle_dir(tmp_path: Path) -> Path:
    root = tmp_path / "dist"
    (root / "assets").mkdir(parents=True)
    (root / "index.html").write_bytes(b"<!doctype html><title>Console</title>")
    (root / "assets" / "app-abc123.js").write_bytes(b"export const ready = true;\n")
    return root


@pytest.fixture
def governance_database(tmp_path: Path) -> Iterator[Path]:
    path = tmp_path / "registry.sqlite"
    connection = sqlite3.connect(path)
    try:
        connection.executescript(migrations_module._MIGRATION_LEDGER_SQL)
        for migration in migrations_module.MIGRATIONS:
            connection.executescript(migration.sql)
        connection.commit()
    finally:
        connection.close()
    store = GovernanceStore(path)
    lane = GovernanceLane(
        purpose="shadow-eval",
        target="direction",
        horizon_days=5,
        frequency="daily",
        universe="us-large-cap",
        decision_policy="long-short",
        environment="local",
    )
    identity = store.register_lane(lane, now=BASE)
    store.apply_assignment(
        identity,
        champion_revision=CHAMPION,
        expected_generation=0,
        expected_champion=None,
        now=BASE,
    )
    store.append_event(
        identity,
        EventKind.COMPARISON,
        {
            "recommendation": "retain_champion",
            "policy_identity": DIGEST,
            "cohort_identity": "d" * 64,
            "decided_at": BASE.isoformat(),
            "gates": [{"name": "minimum_pairs", "satisfied": False, "detail": "12 of 200"}],
            "tests": [
                {
                    "name": "superiority",
                    "metric": "brier",
                    "verdict": "inconclusive",
                    "point_estimate": 0.001,
                    "interval": [-0.01, 0.02],
                    "p_value_uncorrected": 0.4,
                    "blocks": 30,
                    "observations": 240,
                    "margin": None,
                }
            ],
            "correction": {"method": "holm_bonferroni", "alpha": 0.05, "family_size": 1},
        },
        now=BASE,
    )
    yield path


@pytest.fixture
def client(tmp_path: Path, governance_database: Path) -> Iterator[TestClient]:
    app = create_app(
        _StubEvidencePorts(),
        governance=GovernanceReadPorts(governance_database),
        console=load_console_bundle(_bundle_dir(tmp_path)),
    )
    assert_read_only_route_inventory(app)
    # Loopback authority: the boundary refuses any other Host, so the client
    # must present one the local service actually accepts.
    with TestClient(app, base_url="http://127.0.0.1") as test_client:
        yield test_client


@pytest.fixture
def lane_identity(governance_database: Path) -> str:
    return GovernanceReadPorts(governance_database).list_lanes().items[0].lane_identity


# ---------------------------------------------------------------------------
# Console assets
# ---------------------------------------------------------------------------


def test_the_console_document_is_served_with_its_strict_policy(client: TestClient) -> None:
    response = client.get("/console")
    assert response.status_code == 200
    assert response.headers["content-type"] == "text/html; charset=utf-8"
    assert response.headers["content-security-policy"] == CONSOLE_CSP
    assert response.headers["cache-control"] == "no-store"
    assert response.headers["x-content-type-options"] == "nosniff"
    assert response.headers["x-frame-options"] == "DENY"


def test_every_declared_route_serves_the_application_document(client: TestClient) -> None:
    for route in (
        "/console",
        "/console/runs",
        "/console/evidence",
        "/console/comparison",
        "/console/calibration",
        "/console/operations",
        "/console/governance",
    ):
        response = client.get(route)
        assert response.status_code == 200, route
        assert response.headers["content-type"] == "text/html; charset=utf-8"


def test_a_hashed_asset_is_immutable_and_typed(client: TestClient) -> None:
    response = client.get("/console/assets/app-abc123.js")
    assert response.status_code == 200
    assert response.headers["content-type"] == "text/javascript; charset=utf-8"
    assert response.headers["cache-control"] == "public, max-age=31536000, immutable"
    assert response.headers["etag"].startswith('"')


def test_head_on_a_console_asset_returns_headers_without_a_body(client: TestClient) -> None:
    head = client.head("/console/assets/app-abc123.js")
    get = client.get("/console/assets/app-abc123.js")
    assert head.status_code == 200
    assert head.content == b""
    assert head.headers["content-length"] == get.headers["content-length"]
    assert head.headers["content-security-policy"] == get.headers["content-security-policy"]


def test_an_unknown_console_path_is_a_bounded_problem_not_the_document(
    client: TestClient,
) -> None:
    """A wildcard fallback would make the console answer 200 for anything."""
    response = client.get("/console/does-not-exist")
    assert response.status_code == 404
    assert response.headers["content-type"].startswith("application/problem+json")
    assert b"<!doctype" not in response.content.lower()


@pytest.mark.parametrize("path", ["/console/../pyproject.toml", "/console/%2e%2e/secret"])
def test_traversal_attempts_do_not_escape_the_bundle(client: TestClient, path: str) -> None:
    response = client.get(path)
    assert response.status_code in {400, 404}
    assert b"pyproject" not in response.content


# ---------------------------------------------------------------------------
# Method discipline
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("method", ["POST", "PUT", "PATCH", "DELETE", "OPTIONS"])
@pytest.mark.parametrize("path", ["/console", "/console/assets/app-abc123.js", "/api/v1/runs"])
def test_every_mutation_verb_is_refused_everywhere(
    client: TestClient, method: str, path: str
) -> None:
    response = client.request(method, path)
    assert response.status_code == 405


def test_head_is_refused_on_the_json_api(client: TestClient) -> None:
    """HEAD there would be a second evidence path with no body to verify."""
    assert client.head("/api/v1/governance/lanes").status_code == 405


# ---------------------------------------------------------------------------
# Governance projection over HTTP
# ---------------------------------------------------------------------------


def test_lanes_are_listed_without_a_total_count(client: TestClient) -> None:
    """A count over an append-only chain is stale before it renders."""
    response = client.get("/api/v1/governance/lanes")
    assert response.status_code == 200
    body = response.json()
    assert set(body) == {"schema_version", "items", "next_cursor"}
    assert "total" not in body
    assert body["items"][0]["state"] == "active"
    assert body["items"][0]["champion_revision"] == CHAMPION
    assert body["items"][0]["chain_verified"] is True


def test_lane_detail_states_the_authority_boundary(client: TestClient, lane_identity: str) -> None:
    response = client.get(f"/api/v1/governance/lanes/{lane_identity}")
    assert response.status_code == 200
    body = response.json()
    assert "no console action can approve" in body["authority"].lower()
    assert body["lane"]["lane_identity"] == lane_identity
    assert [event["kind"] for event in body["events"]] == ["assignment", "comparison"]


def test_the_projection_exposes_no_field_usable_to_promote(
    client: TestClient, lane_identity: str
) -> None:
    """A console that cannot express a promotion cannot perform one."""
    body = client.get(f"/api/v1/governance/lanes/{lane_identity}").text.lower()
    for forbidden in (
        "idempotency",
        "approval_token",
        "expected_generation",
        "expected_champion",
        "secret",
        "password",
    ):
        assert forbidden not in body


def test_a_failed_gate_reaches_the_console_intact(client: TestClient, lane_identity: str) -> None:
    """Unfavourable evidence must render, not be filtered on the way out."""
    response = client.get(f"/api/v1/governance/lanes/{lane_identity}/comparisons")
    assert response.status_code == 200
    comparison = response.json()["items"][0]
    assert comparison["recommendation"] == "retain_champion"
    assert comparison["gates"][0]["satisfied"] is False
    assert comparison["gates"][0]["detail"] == "12 of 200"
    assert comparison["tests"][0]["verdict"] == "inconclusive"
    assert comparison["correction_method"] == "holm_bonferroni"


@pytest.mark.parametrize("lane", ["short", "Z" * 64, "0" * 63])
def test_a_malformed_lane_identity_is_refused_at_the_boundary(
    client: TestClient, lane: str
) -> None:
    assert client.get(f"/api/v1/governance/lanes/{lane}").status_code in {400, 404, 422}


def test_an_unknown_lane_yields_a_bounded_problem_document(client: TestClient) -> None:
    response = client.get(f"/api/v1/governance/lanes/{'f' * 64}")
    assert response.status_code in {400, 404}
    assert response.headers["content-type"].startswith("application/problem+json")
    # No host path or internal trace escapes with the failure.
    assert "/private/" not in response.text
    assert "Traceback" not in response.text


def test_page_size_is_bounded_at_the_boundary(client: TestClient) -> None:
    assert client.get("/api/v1/governance/lanes?page_size=0").status_code == 422
    assert client.get("/api/v1/governance/lanes?page_size=101").status_code == 422
    assert client.get("/api/v1/governance/lanes?page_size=100").status_code == 200


def test_governance_routes_are_absent_when_no_port_is_injected(tmp_path: Path) -> None:
    """The console surface is additive: without governance it simply is not there."""
    app = create_app(_StubEvidencePorts())
    assert_read_only_route_inventory(app)
    with TestClient(app, base_url="http://127.0.0.1") as client:
        assert client.get("/api/v1/governance/lanes").status_code == 404


@pytest.mark.parametrize("host", ["evil.example", "127.0.0.1.evil.example", "0.0.0.0", "[::1]"])
def test_a_non_loopback_host_cannot_reach_the_console(
    tmp_path: Path, governance_database: Path, host: str
) -> None:
    """Local-only enforcement is not weakened by adding a static surface."""
    app = create_app(
        _StubEvidencePorts(),
        governance=GovernanceReadPorts(governance_database),
        console=load_console_bundle(_bundle_dir(tmp_path)),
    )
    with TestClient(app, base_url="http://127.0.0.1") as client:
        response = client.get("/console", headers={"host": host})
    assert response.status_code == 400


def test_proxy_headers_are_refused_on_the_console_surface(
    tmp_path: Path, governance_database: Path
) -> None:
    """Trusting a forwarded header would let a proxy forge the local authority."""
    app = create_app(
        _StubEvidencePorts(),
        governance=GovernanceReadPorts(governance_database),
        console=load_console_bundle(_bundle_dir(tmp_path)),
    )
    with TestClient(app, base_url="http://127.0.0.1") as client:
        response = client.get("/console", headers={"x-forwarded-for": "10.0.0.1"})
    assert response.status_code == 400
