"""Public engineering evidence preserves weak modules and every test observation."""

import json

import pytest

from foundry_build import paper_evidence


def test_evidence_collection_retains_every_result_and_weak_module(monkeypatch):
    def run(command, **kwargs):
        assert kwargs["timeout"] == 180 and kwargs["check"]
        assert all("packages/" not in arg for arg in command)
        coverage = next(
            arg.split(":", 1)[1]
            for arg in command
            if arg.startswith("--cov-report=json:")
        )
        report = next(
            arg.split("=", 1)[1] for arg in command if arg.startswith("--junitxml=")
        )
        from pathlib import Path

        Path(coverage).write_text(
            json.dumps(
                {
                    "totals": {"percent_covered": 91},
                    "files": {
                        "weak.py": {
                            "summary": {"num_branches": 4, "covered_branches": 1}
                        },
                        "empty.py": {"summary": {"num_branches": 0}},
                    },
                }
            )
        )
        Path(report).write_text(
            '<testsuite><testcase name="one" classname="paper_core" time="0"/>'
            '<testcase name="two" classname="paper_core" time="1">'
            "<failure/></testcase></testsuite>"
        )

    monkeypatch.setattr(paper_evidence.subprocess, "run", run)
    value = paper_evidence.collect()
    assert value["broker_requests"] == value["live_orders"] == 0
    assert [row["passed"] for row in value["tests"]] == [True, False]
    assert value["modules"] == [
        {"module": "weak", "covered_branches": 1, "branches": 4}
    ]


def test_committed_evidence_renders_all_samples(tmp_path):
    value = json.loads((paper_evidence.DESTINATION / "measurements.json").read_text())
    assert value["evidence_class"] == "software_fixture"
    assert all(row["passed"] for row in value["tests"])
    assert len(value["tests"]) >= 100
    paper_evidence.render(value, tmp_path)
    assert (tmp_path / "software-checks.png").stat().st_size > 20_000


def test_collection_does_not_hide_test_failure(monkeypatch):
    def fail(*args, **kwargs):
        raise paper_evidence.subprocess.CalledProcessError(1, "pytest")

    monkeypatch.setattr(paper_evidence.subprocess, "run", fail)
    with pytest.raises(paper_evidence.subprocess.CalledProcessError):
        paper_evidence.collect()
