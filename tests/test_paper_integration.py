"""Real CLI/worker/HTTP contracts and source qualification; broker I/O is a fixture."""

from __future__ import annotations

import io
import json
import subprocess
from datetime import timedelta
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

import signal_foundry.cli as cli
import signal_foundry.trading.engine as engine_module
import signal_foundry.trading.entry as entry
import tests.paper_helpers as helpers
from signal_foundry.api import create_app
from signal_foundry.boundary import FoundryError, code_identity, encode
from signal_foundry.manager import Manager
from signal_foundry.store import Store
from signal_foundry.trading.alpaca import Alpaca
from signal_foundry.trading.engine import Engine
from signal_foundry.trading.service import (
    Action,
    PaperService,
)
from signal_foundry.trading.store import Journal
from tests.paper_helpers import NOW, ROOT, FixedTime, Vendor, config
from tests.research_helpers import FakeRunner
from tests.test_paper_core import setup as setup
from tests.test_research_api import HEADERS
from tests.test_research_integration import runner as runner


@pytest.fixture
def configured(tmp_path):
    path = tmp_path / "paper.json"
    path.write_bytes(encode(config().wire()))
    return PaperService(ROOT, tmp_path / "paper", path)


def test_real_subprocess_initialize_status_and_persistent_stop(configured):
    service = configured
    assert service.status().state == "uninitialized"
    assert service.action(Action(operation="initialize")).status.state == "initialized"
    assert service.action(Action(operation="initialize")).status.state == "initialized"
    assert service.action(Action(operation="stop")).status.stopped
    with pytest.raises(FoundryError, match="paper_disabled"):
        service.action(Action(operation="start"))
    assert service.status().stopped


def test_same_service_http_contract_and_browser_abuse(configured, tmp_path):
    manager = Manager(Store(tmp_path / "research"), FakeRunner())
    with TestClient(
        create_app(lambda: manager, paper_factory=lambda: configured),
        base_url="http://127.0.0.1:8765",
    ) as http:
        assert http.get("/api/v1/paper").json()["live_authorized"] is False
        result = http.post(
            "/api/v1/paper", headers=HEADERS, json={"operation": "initialize"}
        )
        assert result.status_code == 200 and result.json()["status"]["configured"]
        for value in (
            {"operation": "live"},
            {"operation": "cycle", "quantity": "1"},
            {"operation": "cycle", "symbol": "../../etc"},
        ):
            assert (
                http.post("/api/v1/paper", headers=HEADERS, json=value).status_code
                == 422
            )
        assert (
            http.post(
                "/api/v1/paper",
                headers={**HEADERS, "Origin": "https://evil.example"},
                json={"operation": "stop"},
            ).status_code
            == 403
        )
        assert http.post(
            "/api/v1/paper", headers=HEADERS, json={"operation": "stop"}
        ).json()["status"]["stopped"]
    manager = Manager(Store(tmp_path / "other"), FakeRunner())
    with TestClient(
        create_app(lambda: manager), base_url="http://127.0.0.1:8765"
    ) as http:
        assert not http.get("/api/v1/paper").json()["configured"]
        assert (
            http.post(
                "/api/v1/paper", headers=HEADERS, json={"operation": "start"}
            ).status_code
            == 409
        )


def test_parent_capacity_credential_policy_and_response(configured, monkeypatch):
    service = configured
    service._capacity.acquire()
    try:
        with pytest.raises(FoundryError, match="paper_busy"):
            service.action(Action(operation="initialize"))
        assert service.action(Action(operation="stop")).status.stopped
    finally:
        service._capacity.release()
    monkeypatch.setenv("ALPACA_API_KEY", "private-value")
    with pytest.raises(FoundryError, match="credential_policy"):
        service.action(Action(operation="probe"))


def test_cli_uses_real_worker_and_private_artifact_boundary(configured, capsys):
    args = [
        "--root",
        str(ROOT),
        "--paper-state",
        str(configured.state),
        "--paper-config",
        str(configured.config_path),
        "paper",
    ]
    assert cli.main([*args, "initialize"]) == 0
    assert json.loads(capsys.readouterr().out)["status"]["state"] == "initialized"
    assert cli.main([*args, "status"]) == 0
    assert json.loads(capsys.readouterr().out)["live_authorized"] is False
    evidence = configured.state / "input.json"
    evidence.write_bytes(
        encode(
            {
                "plan_identity": configured.config.plan.identity,
                "config_identity": configured.config.identity,
                "evidence_kind": "capacity",
            }
        )
    )
    assert cli.main([*args, "import-evidence", "--file", str(evidence)]) == 0
    identity = json.loads(capsys.readouterr().out)["artifact"]
    assert cli.main([*args, "show-artifact", "--artifact", identity]) == 0
    assert json.loads(capsys.readouterr().out)["evidence_kind"] == "capacity"
    assert cli.main([*args, "import-evidence"]) == 1
    capsys.readouterr()
    evidence.write_text("{}")
    assert cli.main([*args, "import-evidence", "--file", str(evidence)]) == 1
    capsys.readouterr()
    assert cli.main(["paper", "status"]) == 1
    assert "paper_configuration" in capsys.readouterr().err


def test_intraday_acquisition_is_immutable_cache_first_and_session_aware(setup):
    engine, journal, vendor = setup
    selected = config().wire()
    selected["plan"].update(
        start="2026-09-08T13:30:00Z",
        selection_end="2026-09-08T14:00:00Z",
        end="2026-09-08T14:30:00Z",
    )
    engine.config = config(**selected)
    journal.set("config", engine.config.wire(), kind="test_config")
    for symbol in engine.config.plan.symbols:
        engine.acquire(symbol)
    calls = len(vendor.calls)
    engine.acquire("AAA")
    assert len(vendor.calls) == calls
    data = journal.read_artifact(journal.get("dataset/AAA"))
    assert len(data["bars"]) == 3 and data["feed"] == "iex"
    assert not data["point_in_time_universe_complete"]
    engine.research()
    result = journal.read_artifact(journal.get("research"))
    assert set(result["reports"]) == {"AAA", "BBB", "CCC"}
    assert result["decision"] == "NO_GO"
    with pytest.raises(FoundryError, match="paper_symbol"):
        engine.acquire("ZZZ")


def test_missing_or_future_data_does_not_turn_into_a_result(setup):
    engine, _, vendor = setup
    with pytest.raises(FoundryError, match="data_missing"):
        engine.research()
    engine.config = config().model_copy(
        update={
            "plan": config().plan.model_copy(update={"end": NOW + timedelta(days=1)})
        }
    )
    with pytest.raises(FoundryError, match="future_data"):
        engine.acquire("AAA")
    assert vendor.calls == []


def test_real_elapsed_time_and_missing_sessions_cannot_be_manufactured(
    setup, monkeypatch
):
    engine, journal, vendor = setup
    engine.start()
    with pytest.raises(FoundryError, match="session_incomplete"):
        engine.record_session()
    monkeypatch.setattr(helpers, "NOW", NOW.replace(hour=21))
    engine.record_session()
    with pytest.raises(FoundryError, match="duplicate_session"):
        engine.record_session()
    assert engine.status().paper_sessions == 1
    identity = engine.campaign()
    report = journal.read_artifact(identity)
    assert report["decision"] == "NO_GO" and report["elapsed_days"] == 0
    assert len(report["observations"]) == 1
    journal.set("observed_dates", [], kind="test_fault")
    vendor.calendar.append({"date": "2026-09-09", "open": "09:30", "close": "16:00"})
    monkeypatch.setattr(helpers, "NOW", NOW.replace(hour=21) + timedelta(days=1))
    report = journal.read_artifact(engine.campaign())
    assert report["missing_dates"] == ["2026-09-09"]
    assert report["decision"] == "NO_GO"


@pytest.fixture
def dossier(tmp_path, monkeypatch):
    monkeypatch.setattr(engine_module, "datetime", FixedTime)
    journal = Journal(tmp_path / "paper")
    engine = Engine(ROOT, journal, config())
    engine.initialize()
    command = [
        str(ROOT / "packages/alphaforge/.venv/bin/python"),
        "-c",
        "import json; from alphaforge.research.qualification "
        "import standard_paper_rubric; "
        "print(json.dumps(standard_paper_rubric().to_dict()))",
    ]
    rubric = json.loads(
        subprocess.run(
            command,
            cwd=ROOT / "packages/alphaforge",
            capture_output=True,
            text=True,
            check=True,
            timeout=10,
        ).stdout
    )
    criteria = []
    for criterion in rubric["criteria"]:
        links = []
        for kind in criterion["required_evidence"]:
            identity = journal.artifact(
                {
                    "config_identity": engine.config.identity,
                    "plan_identity": engine.config.plan.identity,
                    "evidence_kind": kind,
                    "evidence_class": "measured",
                }
            )
            links.append({"kind": kind, "identifier": kind, "content_hash": identity})
        criteria.append(
            {
                "name": criterion["name"],
                "observed": criterion["threshold"],
                "evidence": links,
                "passed": False,
            }
        )
    value = {
        "config_identity": engine.config.identity,
        "code_identity": code_identity(ROOT),
        "valid_until": (NOW + timedelta(days=1)).isoformat(),
        "decision": {
            "plan_hash": engine.config.plan.identity,
            "candidate_id": engine.config.candidate,
            "rubric_identity": rubric["identity"],
            "decided_at": NOW.isoformat(),
            "criteria": criteria,
            "verdict": "REJECTED",
        },
    }
    path = journal.root / "qualification.json"
    path.write_bytes(encode(value))
    yield engine, journal, value, path
    journal.close()


def test_recompute_actual_preserved_source_qualification(dossier, runner):
    engine, journal, value, path = dossier
    # Threshold-boundary synthetic observations exercise the source gate only;
    # no broker/transport exists here, and this fixture is never exported.
    result = engine.qualification()
    assert result["verdict"] == "QUALIFIED_FOR_PAPER"
    value["decision"]["criteria"][0]["observed"] = None
    path.write_bytes(encode(value))
    with pytest.raises(FoundryError, match="paper_qualification"):
        engine.qualification()


@pytest.mark.parametrize(
    "fault",
    [
        "envelope",
        "code",
        "expiry",
        "plan",
        "candidate",
        "count",
        "duplicate",
        "links",
        "artifact",
    ],
)
def test_qualification_rejects_tampering_before_source_or_network(dossier, fault):
    engine, journal, value, path = dossier
    if fault == "envelope":
        value["qualified"] = True
    if fault == "code":
        value["code_identity"] = "f" * 64
    if fault == "expiry":
        value["valid_until"] = (NOW - timedelta(seconds=1)).isoformat()
    if fault == "plan":
        value["decision"]["plan_hash"] = "f" * 64
    if fault == "candidate":
        value["decision"]["candidate_id"] = "other"
    if fault == "count":
        value["decision"]["criteria"] = []
    if fault == "duplicate":
        value["decision"]["criteria"][1] = value["decision"]["criteria"][0]
    if fault == "links":
        value["decision"]["criteria"][0]["evidence"] = []
    if fault == "artifact":
        link = value["decision"]["criteria"][0]["evidence"][0]
        link["content_hash"] = journal.artifact({"wrong": True})
    path.write_bytes(encode(value))
    with pytest.raises(FoundryError, match="paper_qualification"):
        engine.qualification()


def test_entry_dispatch_uses_real_engine_and_serialized_envelope(
    configured, monkeypatch
):
    vendor = Vendor()
    monkeypatch.setattr(entry, "limits", lambda: None)
    monkeypatch.setattr(Engine, "broker", property(lambda _: Alpaca(vendor)))
    monkeypatch.setattr(engine_module, "datetime", FixedTime)
    # Initialize through the actual process first; exercise in-process dispatch
    # afterward so coverage includes its ownership/error-handling boundary.
    configured.action(Action(operation="initialize"))
    for operation in ("probe", "research", "qualify", "campaign", "stop"):
        payload = {
            "state": str(configured.state),
            "config": str(configured.config_path),
            "action": {"operation": operation},
        }
        output = io.BytesIO()
        monkeypatch.setattr(
            entry.sys, "stdin", SimpleNamespace(buffer=io.BytesIO(encode(payload)))
        )
        monkeypatch.setattr(entry.sys, "stdout", SimpleNamespace(buffer=output))
        assert entry.main() == 0
        result = json.loads(output.getvalue())
        if operation == "probe":
            assert (
                result["result"]["account_digest"] == configured.config.account_digest
            )
        if operation in {"research", "qualify"}:
            assert "error" in result
    output = io.BytesIO()
    monkeypatch.setattr(entry.sys, "stdin", SimpleNamespace(buffer=io.BytesIO(b"{}")))
    monkeypatch.setattr(entry.sys, "stdout", SimpleNamespace(buffer=output))
    assert (
        entry.main() == 0
        and json.loads(output.getvalue())["error"]["code"] == "paper_envelope"
    )


def test_entry_owned_lifecycle_dispatch_and_contract_failure(configured, monkeypatch):
    vendor = Vendor()
    monkeypatch.setattr(entry, "limits", lambda: None)
    monkeypatch.setattr(Engine, "broker", property(lambda _: Alpaca(vendor)))
    monkeypatch.setattr(engine_module, "datetime", FixedTime)
    monkeypatch.setattr("signal_foundry.trading.alpaca.datetime", FixedTime)
    monkeypatch.setattr(
        Engine, "qualification", lambda _: {"verdict": "QUALIFIED_FOR_PAPER"}
    )
    for operation in (
        "initialize",
        "qualify",
        "start",
        "cycle",
        "reconcile",
        "record-session",
        "cancel",
    ):
        payload = {
            "state": str(configured.state),
            "config": str(configured.config_path),
            "action": {"operation": operation, "symbol": "AAA"},
        }
        output = io.BytesIO()
        monkeypatch.setattr(
            entry.sys, "stdin", SimpleNamespace(buffer=io.BytesIO(encode(payload)))
        )
        monkeypatch.setattr(entry.sys, "stdout", SimpleNamespace(buffer=output))
        assert entry.main() == 0
        value = json.loads(output.getvalue())
        if operation == "record-session":
            assert value["error"]["code"] == "session_incomplete"
        else:
            assert value["result"]["status"]["last_action"] == operation
    assert len(vendor.orders) == 1
    output = io.BytesIO()
    monkeypatch.setattr(
        entry.sys,
        "stdin",
        SimpleNamespace(
            buffer=io.BytesIO(
                encode(
                    {"state": [], "config": None, "action": {"operation": "initialize"}}
                )
            )
        ),
    )
    monkeypatch.setattr(entry.sys, "stdout", SimpleNamespace(buffer=output))
    assert entry.main() == 0
    assert json.loads(output.getvalue())["error"]["code"] == "paper_contract"
