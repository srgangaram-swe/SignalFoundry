"""Actual locked producer/consumer workers, deterministic research and HTTP E2E.

No optional skip: qualification must install both source environments. Coverage
instrumentation is injected at this explicit test boundary, never into the API.
"""

from __future__ import annotations

import os
import subprocess
import threading
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

import signal_foundry.runner as runner_module
from signal_foundry.api import create_app
from signal_foundry.boundary import FoundryError
from signal_foundry.contracts import DataChoice, JobState, ModelChoice, ResearchRequest
from signal_foundry.manager import Manager
from signal_foundry.runner import Runner
from signal_foundry.store import Store
from tests.test_research_api import HEADERS

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def runner(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Runner:
    original = runner_module.execute

    def instrumented(command: list[str], **options: Any) -> bytes:
        if os.environ.get("FOUNDRY_TEST_COVERAGE") == "1":
            command = [
                command[0],
                "-I",
                "-m",
                "coverage",
                "run",
                "--branch",
                "--parallel-mode",
                f"--source={ROOT / 'signal_foundry'}",
                *command[2:],
            ]
            options["environment"] = {
                **options["environment"],
                "COVERAGE_FILE": str(ROOT / ".coverage"),
            }
        return original(command, **options)

    monkeypatch.setattr(runner_module, "execute", instrumented)
    return Runner(ROOT, tmp_path / "state")


def test_real_registry_preflight_and_repeatable_research(runner: Runner) -> None:
    choices = runner.catalog()
    assert {item.name for item in choices.strategies} == {
        "long_short",
        "long_only_topk",
        "rank_weighted",
        "confidence_weighted",
        "threshold",
    }
    assert not next(
        item for item in choices.models if item.name == "ensemble"
    ).available
    request = ResearchRequest()
    assert runner.validate(request).observations == 4500
    first = runner.run(request, threading.Event())
    second = runner.run(request, threading.Event())
    assert first.canonical() == second.canonical()
    tables = {table.name: table for table in first.tables}
    assert {
        "folds",
        "learning",
        "comparison",
        "uncertainty",
        "capacity",
        "stress",
        "concentration",
        "attribution",
    } <= tables.keys()
    assert tables["comparison"].total_rows == 3
    assert first.live_readiness == "NOT_READY"
    assert first.request == request
    assert all(
        column.name not in {"close", "open", "price", "api_key"}
        for table in first.tables
        for column in table.columns
    )
    curve = tables["equity_ridge"]
    assert any(row[1] != 0 for row in curve.rows)
    assert tables["equity_historical_mean"].rows[0][0] == curve.rows[0][0]
    assert tables["equity_momentum_baseline"].rows[-1][0] == curve.rows[-1][0]


@pytest.mark.parametrize(
    "strategy", ["long_only_topk", "rank_weighted", "confidence_weighted", "threshold"]
)
def test_real_strategy_dispatch_and_causal_regime(
    runner: Runner, strategy: str
) -> None:
    request = ResearchRequest(
        strategy=strategy, baselines=("zero_baseline",), regime="causal_volatility"
    )
    assert runner.validate(request).request_hash == request.digest()
    result = runner.run(request, threading.Event())
    assert result.request.strategy == strategy
    assert (
        next(table for table in result.tables if table.name == "comparison").total_rows
        == 2
    )


@pytest.mark.parametrize("model", ["not_registered", "ensemble"])
def test_real_unavailable_model_never_executes(runner: Runner, model: str) -> None:
    with pytest.raises(FoundryError, match="model_unavailable"):
        runner.validate(ResearchRequest(model=ModelChoice(name=model)))


def publish_bundle(parent: Path, *, constant_volume: bool = False) -> str:
    exported = subprocess.run(
        [
            str(ROOT / "packages/signalattice/.venv/bin/python"),
            "-I",
            str(ROOT / "tests/fixtures/publish_bundle.py"),
            str(parent),
            *(["--constant-volume"] if constant_volume else []),
        ],
        capture_output=True,
        text=True,
        timeout=30,
        check=True,
        env={
            "PATH": "/usr/bin:/bin",
            "OMP_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
            "VECLIB_MAXIMUM_THREADS": "1",
        },
    )
    identity = exported.stdout.strip().splitlines()[-1]
    assert len(identity) == 64
    return identity


def test_real_producer_consumer_bundle_to_http_evidence(
    runner: Runner, tmp_path: Path
) -> None:
    parent = tmp_path / "bundles"
    identity = publish_bundle(parent)
    runner.bundles = parent
    catalog = runner.catalog()
    assert catalog.datasets[0].bundle_id == identity
    request = ResearchRequest(
        data=DataChoice(kind="bundle", bundle_id=identity), baselines=("zero_baseline",)
    )
    manager = Manager(Store(runner.cache.parent), runner)
    with TestClient(
        create_app(lambda: manager), base_url="http://127.0.0.1:8765"
    ) as client:
        validated = client.post(
            "/api/v1/validate", json=request.model_dump(mode="json"), headers=HEADERS
        )
        assert validated.status_code == 200, validated.text
        assert validated.json()["observations"] == 2500
        submitted = client.post(
            "/api/v1/jobs",
            json=request.model_dump(mode="json"),
            headers={**HEADERS, "Idempotency-Key": "integration_request_001"},
        )
        assert submitted.status_code == 202, submitted.text
        job = manager.wait(submitted.json()["job_id"], 60)
        assert job.state == JobState.SUCCEEDED, job
        result = client.get(f"/api/v1/jobs/{job.job_id}/evidence")
        assert result.status_code == 200
        assert result.json()["data_identity"] == "bundle:" + identity
        assert "survivorship" in str(result.json()["limitations"]).lower()
        assert not any(
            token in result.text for token in ("api_key", "adj_close", str(tmp_path))
        )


def test_incompatible_training_features_fail_during_preflight(
    runner: Runner, tmp_path: Path
) -> None:
    parent = tmp_path / "constant-volume"
    identity = publish_bundle(parent, constant_volume=True)
    runner.bundles = parent
    request = ResearchRequest(data=DataChoice(kind="bundle", bundle_id=identity))
    with pytest.raises(FoundryError, match="feature_incompatible"):
        runner.validate(request)


def test_real_catalog_empty_state_permissions(runner: Runner) -> None:
    assert runner.cache.parent.stat().st_mode & 0o077 == 0
    owner = Store(runner.cache.parent)
    try:
        assert owner.list().jobs == ()
    finally:
        owner.close()
