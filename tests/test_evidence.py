"""Reproducibility and transaction tests for redistribution-safe visual evidence."""

from __future__ import annotations

import hashlib
import sys
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import pytest

import foundry_build.evidence as evidence
from foundry_build.assembly import read_json
from foundry_build.git import AssemblyError


def test_measured_artifacts_reproduce_and_refuse_overwrite(
    imported: tuple[Path, dict[str, Any]], tmp_path: Path
) -> None:
    root, manifest = imported
    report = evidence.measurements(root, manifest)
    assert report["sources"]["alphaforge"]["tracked_files"] == 2
    evidence.publish(report, tmp_path / "a")
    evidence.publish(report, tmp_path / "b")
    hashes = read_json(tmp_path / "a/manifest.json")
    for name, digest in hashes.items():
        payload = (tmp_path / "a" / name).read_bytes()
        assert hashlib.sha256(payload).hexdigest() == digest
        assert payload == (tmp_path / "b" / name).read_bytes()
    with pytest.raises(AssemblyError, match="already-exists"):
        evidence.publish(report, tmp_path / "a")
    assert plt.get_fignums() == []


def test_publication_fault_cleanup(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def fail(*args: Any) -> None:
        raise OSError("injected save failure")

    monkeypatch.setattr(evidence, "plot", fail)
    with pytest.raises(AssemblyError, match="publication-failed"):
        evidence.publish({}, tmp_path / "result")
    assert list(tmp_path.iterdir()) == []
    (tmp_path / ".result.reservation").mkdir()
    with pytest.raises(AssemblyError, match="unavailable"):
        evidence.publish({}, tmp_path / "result")
    assert (tmp_path / ".result.reservation").is_dir()


def test_evidence_cli_and_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    report = {"sources": {"fixture": {"commits": 3, "objects": 7, "tracked_files": 2}}}
    monkeypatch.setattr(evidence, "measurements", lambda *args: report)
    monkeypatch.setattr(sys, "argv", ["evidence", "--output", str(tmp_path / "result")])
    assert evidence.main() == 0
    with pytest.raises(SystemExit) as error:
        evidence.main()
    assert error.value.code == 2
