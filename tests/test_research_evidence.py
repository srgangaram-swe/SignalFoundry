"""Measured-harness behavior, deterministic rendering and atomic fault cleanup."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import matplotlib.pyplot as plt
import pytest

import foundry_build.research_evidence as module
from signal_foundry.boundary import FoundryError
from signal_foundry.contracts import ResearchEvidence

REFERENCE = (
    Path(__file__).resolve().parents[1]
    / "docs/evidence/control-plane/measurements.json"
)


def test_reference_figures_reproduce_and_refuse_overwrite(tmp_path: Path) -> None:
    report = json.loads(REFERENCE.read_text())
    module.publish(report, tmp_path / "first")
    module.publish(report, tmp_path / "second")
    hashes = json.loads((tmp_path / "first/manifest.json").read_text())
    for name, digest in hashes.items():
        data = (tmp_path / "first" / name).read_bytes()
        assert hashlib.sha256(data).hexdigest() == digest
        assert (tmp_path / "second" / name).read_bytes() == data
    assert not plt.get_fignums()
    with pytest.raises(FoundryError, match="evidence_exists"):
        module.publish(report, tmp_path / "first")


def test_publication_faults_never_leave_completed_directory(
    tmp_path: Path, monkeypatch
) -> None:
    report = json.loads(REFERENCE.read_text())

    def fail(*args, **kwargs) -> None:
        plt.figure()
        raise OSError("injected plot failure")

    monkeypatch.setattr(module, "_plot", fail)
    with pytest.raises(FoundryError, match="publication_io"):
        module.publish(report, tmp_path / "result")
    assert not list(tmp_path.iterdir())
    assert not plt.get_fignums()
    (tmp_path / ".result.reservation").mkdir()
    with pytest.raises(FoundryError, match="evidence_reserved"):
        module.publish(report, tmp_path / "result")


def test_measurement_reports_admission_and_propagates_other_faults() -> None:
    success = module.measure("test", "test", lambda: None)
    assert success["status"] == "completed" and success["rss_samples"] >= 1
    assert success["wall_seconds"] >= 0 and success["peak_tree_rss_mib"] > 0

    def rejected() -> None:
        raise FoundryError("workers_busy", "Test capacity denial.", 429)

    assert (
        module.measure("test", "saturated", rejected)["status"] == "capacity_rejected"
    )

    def failure() -> None:
        raise FoundryError("not_capacity", "Test unexpected failure.")

    with pytest.raises(FoundryError, match="not_capacity"):
        module.measure("test", "test", failure)


def test_benchmark_harness_keeps_every_sample(tmp_path: Path, monkeypatch) -> None:
    report = json.loads(REFERENCE.read_text())
    result = ResearchEvidence.model_validate_json(json.dumps(report["research"]))

    class Runner:
        def __init__(self, *args) -> None:
            pass

        def catalog(self):
            return None

        def validate(self, request):
            return None

        def run(self, request, cancel):
            return result

    monkeypatch.setattr(module, "Runner", Runner)
    measured = module.benchmark(tmp_path, tmp_path / "state")
    assert len(measured["samples"]) == 22
    assert measured["repeatable_research_hash"] == result.digest()
    assert measured["method"]["saturation_clients"] == 8


def test_render_cli_uses_saved_measurements(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(
        "sys.argv",
        [
            "research-evidence",
            "--input",
            str(REFERENCE),
            "--output",
            str(tmp_path / "result"),
        ],
    )
    assert module.main() == 0
    assert (tmp_path / "result/manifest.json").is_file()
