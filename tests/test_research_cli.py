"""CLI uses the same validated lifecycle and never offers a remote bind."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

import foundry_build.contracts as schema_module
import signal_foundry.cli as cli
from signal_foundry.boundary import FoundryError
from signal_foundry.contracts import ResearchRequest
from tests.research_helpers import FakeRunner


def test_cli_example_and_validation(tmp_path: Path, monkeypatch, capsys) -> None:
    monkeypatch.setattr(cli, "Runner", lambda *args: FakeRunner())
    assert cli.main(["example"]) == 0
    request = ResearchRequest.model_validate_json(capsys.readouterr().out)
    path = tmp_path / "request.json"
    path.write_bytes(request.canonical())
    assert cli.main(["validate", str(path)]) == 0
    assert json.loads(capsys.readouterr().out)["request_hash"] == request.digest()
    assert cli.main(["catalog"]) == 0
    assert json.loads(capsys.readouterr().out)["live_readiness"] == "NOT_READY"
    assert (
        cli.main(
            [
                "--state",
                str(tmp_path / "state"),
                "run",
                str(path),
                "--key",
                "cli_test_request_001",
            ]
        )
        == 0
    )
    assert json.loads(capsys.readouterr().out)["job"]["state"] == "succeeded"


def test_cli_failure_and_invalid_configuration(
    tmp_path: Path, monkeypatch, capsys
) -> None:
    runner = FakeRunner()
    runner.failure = FoundryError("test_failure", "Test failure.")
    monkeypatch.setattr(cli, "Runner", lambda *args: runner)
    path = tmp_path / "request.json"
    path.write_bytes(ResearchRequest().canonical())
    assert cli.main(["--state", str(tmp_path / "state"), "run", str(path)]) == 1
    assert json.loads(capsys.readouterr().out)["job"]["state"] == "failed"
    path.write_text('{"seed":true}')
    assert cli.main(["validate", str(path)]) == 1
    assert json.loads(capsys.readouterr().err)["code"] == "invalid_request"
    assert cli.main(["validate", str(tmp_path / "missing")]) == 1
    assert json.loads(capsys.readouterr().err)["code"] == "resource_unavailable"


def test_serve_is_loopback_only(tmp_path: Path, monkeypatch) -> None:
    import uvicorn
    from fastapi.testclient import TestClient

    monkeypatch.setattr(cli, "Runner", lambda *args: FakeRunner())

    def serve(app, **options) -> None:
        assert options["host"] == "127.0.0.1"
        assert options["workers"] == 1
        assert options["proxy_headers"] is False
        assert options["access_log"] is False
        with TestClient(app, base_url="http://127.0.0.1:8765") as client:
            assert client.get("/api/v1/catalog").status_code == 200

    monkeypatch.setattr(uvicorn, "run", serve)
    assert cli.main(["--state", str(tmp_path / "state"), "serve"]) == 0
    assert cli.main(["serve", "--port", "80"]) == 1
    with pytest.raises(SystemExit) as error:
        cli.main(["serve", "--host", "0.0.0.0"])
    assert error.value.code == 2


def test_schema_cli_detects_drift_without_runtime(tmp_path: Path, monkeypatch) -> None:
    path = tmp_path / "openapi.json"
    monkeypatch.setattr("sys.argv", ["contracts", "--output", str(path)])
    assert schema_module.main() == 0
    first = path.read_bytes()
    monkeypatch.setattr("sys.argv", ["contracts", "--output", str(path), "--check"])
    assert schema_module.main() == 0
    assert schema_module.schema() == first
    path.write_bytes(b"{}")
    with pytest.raises(SystemExit) as error:
        schema_module.main()
    assert error.value.code == 2
