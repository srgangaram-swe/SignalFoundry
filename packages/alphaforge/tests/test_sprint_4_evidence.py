"""Tests for the Sprint 4 close-out evidence bundle (SF-S4-MR8/MR9).

The evidence publisher is itself a claim-making artifact, so it gets the same
treatment as the modules it reports on: deterministic, atomic, non-overwriting,
and honest about what it did not measure.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from alphaforge.research.sprint_4_evidence import (
    STUDY_REPLICATES,
    SYNTHETIC_BARS,
    SprintEvidenceError,
    publish_sprint_4_evidence,
    sprint_4_decision,
    synthetic_candidate_path,
)
from alphaforge.robustness.perturbation import run_perturbation_study, standard_execution_grid


def test_the_synthetic_path_is_deterministic() -> None:
    first_returns, first_costs = synthetic_candidate_path()
    second_returns, second_costs = synthetic_candidate_path()
    assert first_returns.equals(second_returns)
    assert first_costs.equals(second_costs)
    assert len(first_returns) == SYNTHETIC_BARS


def test_sprint_4_has_no_qualified_candidate() -> None:
    """The honest outcome: no trial ledger, no capacity, no uncertainty evidence.

    If this ever returns QUALIFIED_FOR_PAPER without those evidence families
    being supplied, the gate has stopped working.
    """
    returns, costs = synthetic_candidate_path()
    grid = standard_execution_grid(replicates=STUDY_REPLICATES, root_seed=4)
    decision = sprint_4_decision(run_perturbation_study(returns, costs, grid))
    assert decision.verdict == "REJECTED"
    assert decision.qualified is False


def test_the_unmeasured_criteria_are_the_ones_that_block() -> None:
    """Rejection must be traceable to specific missing evidence, not vague failure."""
    returns, costs = synthetic_candidate_path()
    grid = standard_execution_grid(replicates=STUDY_REPLICATES, root_seed=4)
    decision = sprint_4_decision(run_perturbation_study(returns, costs, grid))
    unmeasured = {
        item.name for item in decision.results if item.observed is None and not item.passed
    }
    assert {
        "adjusted_p_value",
        "capacity_utilization",
        "top_name_concentration",
        "uncertainty_lower_bound",
    } <= unmeasured


def test_publishing_writes_every_artifact(tmp_path: Path) -> None:
    destination = tmp_path / "bundle"
    summary = publish_sprint_4_evidence(destination)
    for name in (
        "perturbation_study.json",
        "qualification_decision.json",
        "qualification_dossier.md",
        "summary.json",
        "perturbation_outcomes.csv",
        "sprint_4_evidence.png",
    ):
        assert (destination / name).exists(), name
    assert summary["verdict"] == "REJECTED"
    assert summary["simulation_only"] is True


def test_publishing_refuses_a_non_empty_destination(tmp_path: Path) -> None:
    """A republish must not silently overwrite evidence a report already cites."""
    destination = tmp_path / "bundle"
    destination.mkdir()
    (destination / "existing.json").write_text("{}", encoding="utf-8")
    with pytest.raises(SprintEvidenceError, match="not empty"):
        publish_sprint_4_evidence(destination)


def test_the_published_summary_carries_its_limitations(tmp_path: Path) -> None:
    summary = publish_sprint_4_evidence(tmp_path / "bundle")
    text = " ".join(summary["limitations"])
    assert "Synthetic development data" in text
    assert "not probabilities of loss in the market" in text
    assert "no live or paper trading is authorized" in text


def test_the_published_bundle_is_reproducible(tmp_path: Path) -> None:
    """Two runs of the committed script must agree on every number."""
    first = publish_sprint_4_evidence(tmp_path / "one")
    second = publish_sprint_4_evidence(tmp_path / "two")
    assert first == second
    left = json.loads((tmp_path / "one" / "perturbation_study.json").read_text())
    right = json.loads((tmp_path / "two" / "perturbation_study.json").read_text())
    assert left == right


def test_the_dossier_states_the_verdict_plainly(tmp_path: Path) -> None:
    publish_sprint_4_evidence(tmp_path / "bundle")
    text = (tmp_path / "bundle" / "qualification_dossier.md").read_text(encoding="utf-8")
    assert "**Verdict: REJECTED**" in text
    assert "No paper or live evaluation is authorized" in text
