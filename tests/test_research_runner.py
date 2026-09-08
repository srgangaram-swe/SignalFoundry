"""Adversarial worker envelopes and typed provenance composition boundaries."""

from __future__ import annotations

import threading
from pathlib import Path
from typing import Any

import pytest

import signal_foundry.runner as module
from signal_foundry.boundary import FoundryError, encode
from signal_foundry.contracts import DataChoice, ResearchRequest
from signal_foundry.runner import Runner
from tests.research_helpers import FakeRunner, evidence


@pytest.fixture
def runner(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Runner:
    monkeypatch.setattr(module, "code_identity", lambda _: "a" * 64)
    return Runner(tmp_path, tmp_path / "state")


@pytest.mark.parametrize(
    "envelope,code",
    [
        ([], "worker_contract"),
        ({"result": {}, "other": 1}, "worker_contract"),
        ({"error": []}, "worker_contract"),
        ({"error": {"code": "x"}}, "worker_contract"),
        (
            {"error": {"code": "../x", "detail": "test", "status": 422}},
            "worker_contract",
        ),
        ({"error": {"code": "x", "detail": "test", "status": True}}, "worker_contract"),
        ({"error": {"code": "x", "detail": "test", "status": 200}}, "worker_contract"),
        (
            {
                "error": {
                    "code": "source_failure",
                    "detail": "Test error.",
                    "status": 422,
                }
            },
            "source_failure",
        ),
    ],
)
def test_reject_worker_envelope(
    runner: Runner, monkeypatch: pytest.MonkeyPatch, envelope: Any, code: str
) -> None:
    monkeypatch.setattr(module, "execute", lambda *args, **kwargs: encode(envelope))
    with pytest.raises(FoundryError, match=code):
        runner.call("alphaforge", "catalog", {})
    # Failure releases both capacity slots.
    assert runner._capacity.acquire(blocking=False)
    assert runner._capacity.acquire(blocking=False)
    runner._capacity.release()
    runner._capacity.release()


def test_worker_selection_limits_and_source_change(
    runner: Runner, monkeypatch: pytest.MonkeyPatch
) -> None:
    with pytest.raises(FoundryError, match="unknown_operation"):
        runner.call("../../bin", "exec", {})
    runner._capacity.acquire()
    runner._capacity.acquire()
    with pytest.raises(FoundryError, match="workers_busy"):
        runner.call("alphaforge", "catalog", {})
    runner._capacity.release()
    runner._capacity.release()
    identities = iter(["a" * 64, "b" * 64])
    monkeypatch.setattr(module, "code_identity", lambda _: next(identities))
    monkeypatch.setattr(
        module, "execute", lambda *args, **kwargs: encode({"result": {}})
    )
    with pytest.raises(FoundryError, match="code_changed"):
        runner.call("alphaforge", "catalog", {})


@pytest.mark.parametrize(
    "method,code",
    [
        ("catalog", "catalog_contract"),
        ("validate", "validation_contract"),
        ("run", "evidence_contract"),
    ],
)
def test_malformed_typed_results(
    runner: Runner, monkeypatch: pytest.MonkeyPatch, method: str, code: str
) -> None:
    monkeypatch.setattr(runner, "call", lambda *args, **kwargs: {"unknown": True})
    args = () if method == "catalog" else (ResearchRequest(),)
    if method == "run":
        args += (threading.Event(),)
    with pytest.raises(FoundryError, match=code):
        getattr(runner, method)(*args)


def test_mismatched_validation_and_evidence_identity(
    runner: Runner, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        runner,
        "call",
        lambda *args, **kwargs: FakeRunner()
        .validate(ResearchRequest(seed=1))
        .model_dump(mode="json"),
    )
    with pytest.raises(FoundryError, match="validation_identity"):
        runner.validate(ResearchRequest())
    monkeypatch.setattr(
        runner,
        "call",
        lambda *args, **kwargs: evidence(ResearchRequest(seed=1)).model_dump(
            mode="json"
        ),
    )
    with pytest.raises(FoundryError, match="evidence_identity"):
        runner.run(ResearchRequest(), threading.Event())


def test_producer_identity_is_not_inferred(
    runner: Runner, monkeypatch: pytest.MonkeyPatch
) -> None:
    producer = {
        "bundle_id": "b" * 64,
        "rows": 4500,
        "symbols": ["BENCH", "AAA"],
        "date_min": "2020-01-01",
        "date_max": "2021-01-01",
        "limitations": [],
    }
    monkeypatch.setattr(runner, "call", lambda *args, **kwargs: producer)
    request = ResearchRequest(data=DataChoice(kind="bundle", bundle_id="a" * 64))
    with pytest.raises(FoundryError, match="dataset_identity"):
        runner.validate(request)
