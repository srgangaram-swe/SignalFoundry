"""Manifest, filesystem, same-origin and immutable delivery abuse regressions."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from signal_foundry import cli
from signal_foundry.api import create_app
from signal_foundry.boundary import FoundryError
from signal_foundry.manager import Manager
from signal_foundry.nexus import load_bundle
from signal_foundry.store import Store
from tests.research_helpers import FakeRunner


@pytest.fixture
def build(tmp_path: Path):
    root = tmp_path / "dist"
    (root / "assets").mkdir(parents=True)
    contract = tmp_path / "openapi.json"
    contract.write_bytes(b"{}")
    bodies = {
        "index.html": b"<!doctype html><title>Nexus test fixture</title>",
        "assets/app-123.js": b"export const test = true;",
        "assets/app-123.css": b"body { color: black; }",
    }
    for name, body in bodies.items():
        (root / name).write_bytes(body)
    manifest = {
        "schema_version": "1.0.0",
        "source_sha256": "a" * 64,
        "lock_sha256": "b" * 64,
        "contract_sha256": hashlib.sha256(contract.read_bytes()).hexdigest(),
        "assets": [
            {
                "path": name,
                "bytes": len(body),
                "sha256": hashlib.sha256(body).hexdigest(),
            }
            for name, body in bodies.items()
        ],
        "measured": {
            "javascript_gzip": 50,
            "css_gzip": 50,
            "total_bytes": sum(map(len, bodies.values())),
        },
        "budgets": {
            "javascript_gzip": 256000,
            "css_gzip": 51200,
            "total_bytes": 1048576,
        },
    }
    (root / "manifest.json").write_text(json.dumps(manifest))
    return root, contract, manifest


def test_verified_snapshot_never_rereads_mutated_assets(build, tmp_path: Path) -> None:
    root, contract, _ = build
    bundle = load_bundle(root, contract=contract)
    original = (root / "assets/app-123.js").read_bytes()
    (root / "assets/app-123.js").write_text("changed after startup")
    manager = Manager(Store(tmp_path / "state"), FakeRunner())
    with TestClient(
        create_app(lambda: manager, nexus=bundle), base_url="http://127.0.0.1:8765"
    ) as client:
        assert client.get("/", follow_redirects=False).headers["location"] == "/nexus"
        assert client.get("/nexus").status_code == 200
        assert client.get("/nexus/").content == client.get("/nexus").content
        response = client.get("/nexus/assets/app-123.js")
        assert response.content == original
        assert response.headers["content-type"].startswith("text/javascript")
        head = client.head("/nexus/assets/app-123.js")
        assert not head.content
        assert int(head.headers["content-length"]) == len(original)
        assert "unsafe-inline" not in response.headers["content-security-policy"]
        assert "unsafe-eval" not in response.headers["content-security-policy"]
        assert response.headers["x-frame-options"] == "DENY"
        assert response.headers["cache-control"] == "no-store"
        assert client.get("/nexus/manifest.json").status_code == 200
        assert client.get("/api/v1/catalog").status_code == 200
        for path in ("missing", "ai.md", "assets/app-123.js.map", "%2e%2e/ai.md"):
            assert client.get("/nexus/" + path).status_code == 404
        assert (
            client.get(
                "/nexus", headers={"Origin": "https://foreign.invalid"}
            ).status_code
            == 403
        )
        assert client.post("/nexus", json={}).status_code == 403
        schema = client.get("/api/v1/openapi.json").json()
        assert all(not route.startswith("/nexus") for route in schema["paths"])
    with pytest.raises(TypeError):
        bundle.assets["injected"] = bundle.assets["index.html"]


@pytest.mark.parametrize(
    "fault",
    [
        "extra",
        "map",
        "symlink",
        "directory",
        "mutated",
        "truncated",
        "stale",
        "inventory_limit",
    ],
)
def test_build_faults_fail_closed(build, fault: str) -> None:
    root, contract, _ = build
    asset = root / "assets/app-123.js"
    if fault == "extra":
        (root / ".env").write_text("private fixture")
    elif fault == "map":
        (root / "assets/app-123.js.map").write_text("fixture map")
    elif fault == "symlink":
        asset.unlink()
        asset.symlink_to(contract)
    elif fault == "directory":
        asset.unlink()
        asset.mkdir()
    elif fault == "mutated":
        asset.write_text("x" * len(asset.read_bytes()))
    elif fault == "truncated":
        asset.write_text("x")
    elif fault == "stale":
        contract.write_text('{"changed":true}')
    else:
        for index in range(65):
            (root / "assets" / f"extra-{index}.js").write_text("x")
    with pytest.raises(FoundryError):
        load_bundle(root, contract=contract)


@pytest.mark.parametrize(
    "fault",
    [
        "duplicate",
        "missing_index",
        "traversal",
        "version",
        "bytes",
        "boolean",
        "budget",
        "unknown",
    ],
)
def test_manifest_semantic_faults(build, fault: str) -> None:
    root, contract, manifest = build
    if fault == "duplicate":
        manifest["assets"][1] = manifest["assets"][0]
    elif fault == "missing_index":
        manifest["assets"][0]["path"] = "assets/index.js"
    elif fault == "traversal":
        manifest["assets"][1]["path"] = "assets/../../private.js"
    elif fault == "version":
        manifest["schema_version"] = "2.0.0"
    elif fault == "bytes":
        manifest["measured"]["total_bytes"] += 1
    elif fault == "boolean":
        manifest["assets"][0]["bytes"] = True
    elif fault == "budget":
        manifest["budgets"]["javascript_gzip"] *= 2
    else:
        manifest["secret"] = "must not escape"
    (root / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(FoundryError, match="invalid_bundle") as raised:
        load_bundle(root, contract=contract)
    assert "must not escape" not in raised.value.detail


def test_asset_scan_fault_is_structured(build, monkeypatch) -> None:
    root, contract, _ = build

    def denied(*args):
        raise PermissionError("private operating-system detail")

    monkeypatch.setattr("signal_foundry.nexus.os.scandir", denied)
    with pytest.raises(FoundryError, match="bundle_unavailable") as raised:
        load_bundle(root, contract=contract)
    assert isinstance(raised.value.__cause__, PermissionError)


def test_duplicate_manifest_keys_are_rejected(build) -> None:
    root, contract, _ = build
    (root / "manifest.json").write_bytes(
        b'{"schema_version":"1.0.0","schema_version":"1.0.0"}'
    )
    with pytest.raises(FoundryError, match="invalid_json"):
        load_bundle(root, contract=contract)


def test_cli_verifies_bundle_before_server_or_state_start(build, tmp_path, monkeypatch):
    import uvicorn

    root, contract, _ = build
    destination = tmp_path / "apps" / "nexus" / "dist"
    destination.parent.mkdir(parents=True)
    root.rename(destination)
    schemas = tmp_path / "contracts"
    schemas.mkdir()
    contract.rename(schemas / "openapi-v1.json")
    calls = []
    monkeypatch.setattr(
        uvicorn, "run", lambda app, **options: calls.append((app, options))
    )
    state = tmp_path / "private-state"
    arguments = ["--root", str(tmp_path), "--state", str(state), "serve", "--nexus"]
    assert cli.main(arguments) == 0
    assert len(calls) == 1
    assert calls[0][1]["host"] == "127.0.0.1"
    assert any(route.path == "/nexus" for route in calls[0][0].routes)
    assert not state.exists()
    (destination / "index.html").write_text("corrupt")
    assert cli.main(arguments) == 1
    assert len(calls) == 1
    assert not state.exists()
