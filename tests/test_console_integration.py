"""End-to-end integration of the built console with the real service.

SF-S5-SL-MR6. This is the test that proves the pieces fit: the front-end is
built from source with its committed lockfile, the resulting bundle is loaded by
the real delivery boundary, and the real read-only service answers from a
temporary registry containing synthetic evidence.

Nothing here reaches a provider or the network. The build runs offline against
``node_modules`` that must already be installed, and the service is driven
in-process.

The build is genuinely executed rather than mocked because the acceptance
criterion is about *the exact frontend*: a test against a hand-written fixture
directory would pass while the real bundle emitted something the boundary
refuses to serve.
"""

from __future__ import annotations

import json
import os
import shutil
import sqlite3
import subprocess
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
from quant_platform.service.console import CONSOLE_ROUTES, load_console_bundle
from quant_platform.tracking import migrations as migrations_module

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
WEB_ROOT = REPOSITORY_ROOT / "web"
BASE = datetime(2026, 8, 1, tzinfo=UTC)
CHAMPION = "a" * 64

#: The build is bounded: a front-end build that takes minutes is a build that
#: has started doing something other than bundling.
BUILD_TIMEOUT_SECONDS = 300


class _StubEvidencePorts:
    """Structural stand-in for the evidence read contract.

    The console's governance and asset surfaces are what this module exercises;
    run-evidence projections have their own dedicated suites.
    """

    def probe_evidence_readiness(self) -> Any:  # pragma: no cover - not exercised
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


def _npm() -> str | None:
    """Return the npm executable, or ``None`` when the toolchain is absent."""
    return shutil.which("npm")


@pytest.fixture(scope="module")
def built_console() -> Path:
    """Build the console exactly as a release would, and return its bundle root.

    Skipped -- with a stated reason -- when the Node toolchain or its installed
    dependencies are unavailable. The hardened service container deliberately
    ships without Node, so this test cannot be a hard requirement everywhere it
    might run.
    """
    npm = _npm()
    if npm is None:
        pytest.skip("npm is unavailable; the console build boundary needs the Node toolchain")
    if not (WEB_ROOT / "node_modules").is_dir():
        pytest.skip("web/node_modules is absent; run `npm ci` in web/ to enable this test")

    environment = dict(os.environ)
    # Offline and non-interactive: the build must not reach a registry.
    environment["npm_config_offline"] = "true"
    environment["CI"] = "1"
    result = subprocess.run(  # noqa: S603 - fixed argv, no shell
        [npm, "run", "build"],
        cwd=WEB_ROOT,
        capture_output=True,
        text=True,
        timeout=BUILD_TIMEOUT_SECONDS,
        env=environment,
        check=False,
    )
    if result.returncode != 0:
        pytest.fail(f"console build failed:\n{result.stdout[-2000:]}\n{result.stderr[-2000:]}")
    distribution = WEB_ROOT / "dist"
    assert (distribution / "index.html").is_file(), "the build produced no document"
    return distribution


@pytest.fixture
def registry(tmp_path: Path) -> Iterator[Path]:
    """A temporary registry holding synthetic governance evidence."""
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
    identity = store.register_lane(
        GovernanceLane(
            purpose="shadow-eval",
            target="direction",
            horizon_days=5,
            frequency="daily",
            universe="us-large-cap",
            decision_policy="long-short",
            environment="local",
        ),
        now=BASE,
    )
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
            "recommendation": "insufficient_evidence",
            "policy_identity": "c" * 64,
            "cohort_identity": "d" * 64,
            "decided_at": BASE.isoformat(),
            "gates": [{"name": "minimum_pairs", "satisfied": False, "detail": "12 of 200"}],
            "tests": [
                {
                    "name": "superiority",
                    "metric": "brier",
                    "verdict": "underpowered",
                    "point_estimate": None,
                    "interval": None,
                    "p_value_uncorrected": None,
                    "blocks": 3,
                    "observations": 12,
                    "margin": None,
                }
            ],
            "correction": {},
        },
        now=BASE,
    )
    yield path


@pytest.fixture
def client(built_console: Path, registry: Path) -> Iterator[TestClient]:
    app = create_app(
        _StubEvidencePorts(),
        governance=GovernanceReadPorts(registry),
        console=load_console_bundle(built_console),
    )
    assert_read_only_route_inventory(app)
    with TestClient(app, base_url="http://127.0.0.1") as test_client:
        yield test_client


# ---------------------------------------------------------------------------
# The built bundle is servable
# ---------------------------------------------------------------------------


def test_the_real_build_is_admitted_by_the_delivery_boundary(built_console: Path) -> None:
    """The boundary refuses unexpected file types, so this proves the build agrees."""
    bundle = load_console_bundle(built_console)
    assert bundle.document.route == "/console/index.html"
    scripts = [route for route in bundle.assets if route.endswith(".js")]
    assert scripts, "the build emitted no JavaScript"
    # Hashed assets are immutable; the document never is.
    assert all(bundle.assets[route].immutable for route in scripts)
    assert not bundle.document.immutable


def test_the_build_emits_no_source_map_or_unservable_artifact(built_console: Path) -> None:
    """Source maps would publish the console's source to any local reader."""
    emitted = {path.suffix for path in built_console.rglob("*") if path.is_file()}
    assert ".map" not in emitted
    assert emitted <= {".html", ".js", ".css", ".svg", ".json", ".woff2", ".webmanifest"}


def test_every_console_route_serves_the_built_document(client: TestClient) -> None:
    for route in CONSOLE_ROUTES:
        response = client.get(route)
        assert response.status_code == 200, route
        assert response.headers["content-type"] == "text/html; charset=utf-8"
        assert b'<div id="root">' in response.content


def test_the_document_references_only_assets_the_boundary_serves(
    client: TestClient, built_console: Path
) -> None:
    """A reference the boundary refuses would leave the console blank."""
    import re

    document = (built_console / "index.html").read_text(encoding="utf-8")
    referenced = set(re.findall(r'(?:src|href)="(/console/[^"]+)"', document))
    assert referenced, "the document references no assets"
    for target in sorted(referenced):
        response = client.get(target)
        assert response.status_code == 200, target


# ---------------------------------------------------------------------------
# The console's data surface, driven through the real stack
# ---------------------------------------------------------------------------


def test_the_governance_view_reads_synthetic_evidence(client: TestClient) -> None:
    response = client.get("/api/v1/governance/lanes")
    assert response.status_code == 200
    body = response.json()
    assert len(body["items"]) == 1
    lane = body["items"][0]
    assert lane["state"] == "active"
    assert lane["champion_revision"] == CHAMPION
    assert lane["chain_verified"] is True


def test_an_unfavourable_comparison_reaches_the_console_intact(client: TestClient) -> None:
    """The console must be able to render insufficiency; the API must send it."""
    lanes = client.get("/api/v1/governance/lanes").json()["items"]
    identity = lanes[0]["lane_identity"]
    body = client.get(f"/api/v1/governance/lanes/{identity}/comparisons").json()
    decision = body["items"][0]
    assert decision["recommendation"] == "insufficient_evidence"
    assert decision["gates"][0]["satisfied"] is False
    assert decision["tests"][0]["verdict"] == "underpowered"
    # An underpowered test carries no estimate at all.
    assert decision["tests"][0]["p_value_uncorrected"] is None
    assert decision["tests"][0]["interval_low"] is None


def test_the_served_contract_matches_the_committed_one(client: TestClient) -> None:
    """The console's generated bindings are built from the committed document."""
    served = client.get("/api/v1/openapi.json").json()
    committed = json.loads((REPOSITORY_ROOT / "docs/api/openapi-v1.json").read_text())
    assert served["info"]["version"] == committed["info"]["version"]
    assert set(served["paths"]) >= set(committed["paths"])
    # And it resolves offline: no remote reference anywhere in it.
    assert "https://spec.openapis.org" not in json.dumps(served)


# ---------------------------------------------------------------------------
# The composed boundary keeps its guarantees
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("method", ["POST", "PUT", "PATCH", "DELETE"])
def test_no_mutation_reaches_the_console_or_the_api(client: TestClient, method: str) -> None:
    for path in ("/console", "/api/v1/governance/lanes"):
        assert client.request(method, path).status_code == 405, (method, path)


def test_the_console_document_carries_its_strict_policy(client: TestClient) -> None:
    headers = client.get("/console").headers
    policy = headers["content-security-policy"]
    assert "unsafe-inline" not in policy
    assert "unsafe-eval" not in policy
    assert "connect-src 'self'" in policy
    assert headers["x-content-type-options"] == "nosniff"


def test_the_bundle_contains_no_workstation_path_or_credential(built_console: Path) -> None:
    """A build that embedded a local path would publish it to every reader."""
    suspicious = ("/Users/", "/home/runner/", "BEGIN PRIVATE KEY", "api_key", "secret_key")
    for path in built_console.rglob("*"):
        if not path.is_file():
            continue
        text = path.read_bytes()
        for needle in suspicious:
            assert needle.encode() not in text, f"{path.name} contains {needle!r}"
