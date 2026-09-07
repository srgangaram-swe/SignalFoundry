"""Tests for the frozen qualification rubric and dossier (SF-S4-MR9).

Grouped by acceptance criterion. The invariants that carry the most weight:

* **The rubric is frozen before scoring** — lowering a threshold to admit a
  candidate changes the digest and is refused.
* **Failure is closed and total** — any failed criterion, missing observation,
  missing evidence, or reconciliation failure forces ``REJECTED``. There is no
  weighted score and no override.
* **An unevidenced metric is a failure**, not a weaker pass.
* **``QUALIFIED_FOR_PAPER`` never authorizes capital.**
"""

from __future__ import annotations

import json
from typing import Any

import pytest

from alphaforge.research.qualification import (
    EVIDENCE_KINDS,
    Criterion,
    EvidenceLink,
    QualificationError,
    QualificationRubric,
    qualify,
    render_dossier,
    standard_paper_rubric,
    verify_frozen_rubric,
)

PLAN_HASH = "a" * 64
CONTENT_HASH = "b" * 64
DECIDED_AT = "2026-08-02T12:00:00"

PASSING = {
    "net_return_over_baseline": 0.08,
    "adjusted_p_value": 0.01,
    "stress_downside_net_return": 0.02,
    "stress_insolvent_paths": 0.0,
    "max_drawdown": -0.11,
    "top_name_concentration": 0.22,
    "capacity_utilization": 0.60,
    "uncertainty_lower_bound": 0.004,
}


@pytest.fixture
def rubric() -> QualificationRubric:
    return standard_paper_rubric()


def _evidence(rubric: QualificationRubric) -> dict[str, tuple[EvidenceLink, ...]]:
    return {
        criterion.name: tuple(
            EvidenceLink(
                kind=kind, identifier=f"{criterion.name}-{kind}", content_hash=CONTENT_HASH
            )
            for kind in criterion.required_evidence
        )
        for criterion in rubric.criteria
    }


def _decide(rubric: QualificationRubric, **overrides: Any) -> Any:
    kwargs: dict[str, Any] = {
        "candidate_id": "cand-A",
        "rubric": rubric,
        "expected_rubric_identity": rubric.identity,
        "observations": dict(PASSING),
        "evidence": _evidence(rubric),
        "plan_hash": PLAN_HASH,
        "decided_at": DECIDED_AT,
    }
    kwargs.update(overrides)
    return qualify(**kwargs)


# ---------------------------------------------------------------------------
# The frozen rubric
# ---------------------------------------------------------------------------


def test_the_standard_rubric_asks_every_question_the_work_item_names(
    rubric: QualificationRubric,
) -> None:
    names = {criterion.name for criterion in rubric.criteria}
    assert {
        "net_return_over_baseline",
        "adjusted_p_value",
        "stress_downside_net_return",
        "max_drawdown",
        "top_name_concentration",
        "capacity_utilization",
        "uncertainty_lower_bound",
    } <= names


def test_lowering_a_threshold_after_the_fact_is_detected(
    rubric: QualificationRubric,
) -> None:
    """The failure this module exists to prevent."""
    identity = rubric.identity
    relaxed = QualificationRubric(
        version=rubric.version,
        criteria=tuple(
            (
                criterion
                if criterion.name != "adjusted_p_value"
                else Criterion(
                    name=criterion.name,
                    question=criterion.question,
                    threshold=0.50,
                    direction=criterion.direction,
                    required_evidence=criterion.required_evidence,
                )
            )
            for criterion in rubric.criteria
        ),
    )
    verify_frozen_rubric(rubric, identity)
    with pytest.raises(QualificationError, match="does not match the frozen declaration"):
        verify_frozen_rubric(relaxed, identity)


def test_scoring_against_a_moved_rubric_is_refused(rubric: QualificationRubric) -> None:
    with pytest.raises(QualificationError, match="does not match the frozen declaration"):
        _decide(rubric, expected_rubric_identity="0" * 64)


def test_rubric_identity_is_order_independent(rubric: QualificationRubric) -> None:
    reversed_rubric = QualificationRubric(
        version=rubric.version, criteria=tuple(reversed(rubric.criteria))
    )
    assert reversed_rubric.identity == rubric.identity


def test_duplicate_criterion_names_are_refused() -> None:
    criterion = Criterion(
        name="same",
        question="Does it?",
        threshold=0.0,
        direction="at_least",
        required_evidence=("dataset",),
    )
    with pytest.raises(QualificationError, match="unique"):
        QualificationRubric(version="v", criteria=(criterion, criterion))


def test_an_empty_rubric_is_refused() -> None:
    with pytest.raises(QualificationError, match="at least one criterion"):
        QualificationRubric(version="v", criteria=())


def test_a_criterion_requiring_no_evidence_is_refused() -> None:
    with pytest.raises(QualificationError, match="cannot be checked"):
        Criterion(
            name="unfalsifiable",
            question="Is it good?",
            threshold=0.0,
            direction="at_least",
            required_evidence=(),
        )


def test_an_unsupported_direction_is_refused() -> None:
    with pytest.raises(QualificationError, match="direction"):
        Criterion(
            name="sideways",
            question="Which way?",
            threshold=0.0,
            direction="roughly",  # type: ignore[arg-type]
            required_evidence=("dataset",),
        )


def test_freeze_verification_requires_a_full_digest(rubric: QualificationRubric) -> None:
    with pytest.raises(QualificationError, match="full SHA-256"):
        verify_frozen_rubric(rubric, "short")


# ---------------------------------------------------------------------------
# Evidence links
# ---------------------------------------------------------------------------


def test_evidence_must_pin_a_content_hash() -> None:
    with pytest.raises(QualificationError, match="pin a full SHA-256"):
        EvidenceLink(kind="dataset", identifier="prices", content_hash="v1")


def test_evidence_hash_must_be_hexadecimal() -> None:
    with pytest.raises(QualificationError, match="hexadecimal"):
        EvidenceLink(kind="dataset", identifier="prices", content_hash="z" * 64)


def test_an_unsupported_evidence_kind_is_refused() -> None:
    with pytest.raises(QualificationError, match="unsupported evidence kind"):
        EvidenceLink(kind="vibes", identifier="hunch", content_hash=CONTENT_HASH)


def test_every_declared_evidence_kind_is_constructible() -> None:
    for kind in EVIDENCE_KINDS:
        assert EvidenceLink(kind=kind, identifier="x", content_hash=CONTENT_HASH).kind == kind


# ---------------------------------------------------------------------------
# Scoring: fail-closed behaviour
# ---------------------------------------------------------------------------


def test_a_fully_evidenced_passing_candidate_qualifies(rubric: QualificationRubric) -> None:
    decision = _decide(rubric)
    assert decision.verdict == "QUALIFIED_FOR_PAPER"
    assert decision.qualified is True
    assert decision.blocking_failures == ()


def test_a_candidate_with_no_evidence_at_all_is_rejected(
    rubric: QualificationRubric,
) -> None:
    """Sprint 4's actual state: no candidate, therefore no qualification."""
    decision = _decide(rubric, observations={}, evidence={})
    assert decision.verdict == "REJECTED"
    assert len(decision.blocking_failures) == len(rubric.criteria)


def test_an_unevidenced_metric_fails_rather_than_passing_weakly(
    rubric: QualificationRubric,
) -> None:
    decision = _decide(rubric, evidence={})
    assert decision.verdict == "REJECTED"
    reasons = [item.failure_reason for item in decision.results if not item.passed]
    assert all("missing required evidence" in str(reason) for reason in reasons)


def test_partial_evidence_still_fails(rubric: QualificationRubric) -> None:
    """Citing one of three required kinds is not two-thirds of a pass."""
    evidence = _evidence(rubric)
    evidence["net_return_over_baseline"] = (
        EvidenceLink(kind="dataset", identifier="prices", content_hash=CONTENT_HASH),
    )
    decision = _decide(rubric, evidence=evidence)
    assert decision.verdict == "REJECTED"
    assert "net_return_over_baseline" in decision.blocking_failures


def test_a_missing_observation_cannot_pass(rubric: QualificationRubric) -> None:
    observations = dict(PASSING)
    del observations["max_drawdown"]
    decision = _decide(rubric, observations=observations)
    assert decision.verdict == "REJECTED"
    assert "max_drawdown" in decision.blocking_failures
    failed = [item for item in decision.results if item.name == "max_drawdown"][0]
    assert failed.observed is None
    assert "no observation supplied" in str(failed.failure_reason)


def test_a_non_finite_observation_cannot_pass(rubric: QualificationRubric) -> None:
    observations = dict(PASSING)
    observations["net_return_over_baseline"] = float("nan")
    decision = _decide(rubric, observations=observations)
    assert decision.verdict == "REJECTED"
    assert "net_return_over_baseline" in decision.blocking_failures


def test_a_single_failed_criterion_rejects_the_whole_candidate(
    rubric: QualificationRubric,
) -> None:
    """No weighted score: seven of eight is a rejection."""
    observations = dict(PASSING)
    observations["adjusted_p_value"] = 0.20
    decision = _decide(rubric, observations=observations)
    assert decision.verdict == "REJECTED"
    assert decision.blocking_failures == ("adjusted_p_value",)
    assert sum(1 for item in decision.results if item.passed) == len(rubric.criteria) - 1


def test_reconciliation_failure_rejects_regardless_of_every_other_number(
    rubric: QualificationRubric,
) -> None:
    """An account that does not balance makes every metric meaningless."""
    decision = _decide(rubric, reconciliation_ok=False)
    assert decision.verdict == "REJECTED"
    assert "ledger_reconciliation" in decision.blocking_failures


def test_an_insolvent_stress_path_rejects(rubric: QualificationRubric) -> None:
    observations = dict(PASSING)
    observations["stress_insolvent_paths"] = 1.0
    decision = _decide(rubric, observations=observations)
    assert decision.verdict == "REJECTED"
    assert "stress_insolvent_paths" in decision.blocking_failures


def test_a_negative_stress_downside_rejects(rubric: QualificationRubric) -> None:
    observations = dict(PASSING)
    observations["stress_downside_net_return"] = -0.01
    decision = _decide(rubric, observations=observations)
    assert decision.verdict == "REJECTED"


def test_both_threshold_directions_are_enforced(rubric: QualificationRubric) -> None:
    at_most = [item for item in rubric.criteria if item.direction == "at_most"]
    at_least = [item for item in rubric.criteria if item.direction == "at_least"]
    assert at_most and at_least
    for criterion in at_most:
        assert criterion.satisfied_by(criterion.threshold) is True
        assert criterion.satisfied_by(criterion.threshold + 1.0) is False
    for criterion in at_least:
        assert criterion.satisfied_by(criterion.threshold) is True
        assert criterion.satisfied_by(criterion.threshold - 1.0) is False


def test_a_boundary_observation_satisfies_its_criterion(
    rubric: QualificationRubric,
) -> None:
    """Thresholds are inclusive; exactly meeting the bar passes."""
    observations = dict(PASSING)
    observations["adjusted_p_value"] = 0.05
    observations["net_return_over_baseline"] = 0.0
    decision = _decide(rubric, observations=observations)
    assert decision.verdict == "QUALIFIED_FOR_PAPER"


def test_a_malformed_plan_hash_is_refused(rubric: QualificationRubric) -> None:
    with pytest.raises(QualificationError, match="plan_hash"):
        _decide(rubric, plan_hash="nope")


def test_a_malformed_timestamp_is_refused(rubric: QualificationRubric) -> None:
    with pytest.raises(QualificationError, match="ISO-8601"):
        _decide(rubric, decided_at="last tuesday")


# ---------------------------------------------------------------------------
# The dossier
# ---------------------------------------------------------------------------


def test_the_machine_readable_dossier_is_json_serializable(
    rubric: QualificationRubric,
) -> None:
    payload = json.dumps(_decide(rubric).to_dict())
    assert json.loads(payload)["verdict"] == "QUALIFIED_FOR_PAPER"


def test_a_qualified_dossier_never_authorizes_capital(
    rubric: QualificationRubric,
) -> None:
    decision = _decide(rubric)
    authorization = decision.to_dict()["authorization"]
    assert "zero-capital paper evaluation only" in authorization
    assert "not authorization for live capital" in authorization
    text = render_dossier(decision)
    assert "not a statement that the strategy will be profitable" in text


def test_a_rejected_dossier_authorizes_nothing(rubric: QualificationRubric) -> None:
    decision = _decide(rubric, observations={}, evidence={})
    assert "No paper or live evaluation is authorized" in decision.to_dict()["authorization"]
    assert "No paper or live evaluation is authorized" in render_dossier(decision)


def test_the_dossier_lists_failures_before_passes(rubric: QualificationRubric) -> None:
    """A reader who stops after the first screen must see what is wrong."""
    observations = dict(PASSING)
    observations["adjusted_p_value"] = 0.30
    text = render_dossier(_decide(rubric, observations=observations))
    assert text.index("Blocking failures") < text.index("Satisfied criteria")


def test_the_dossier_states_the_observed_value_and_the_bar_it_missed(
    rubric: QualificationRubric,
) -> None:
    observations = dict(PASSING)
    observations["adjusted_p_value"] = 0.30
    text = render_dossier(_decide(rubric, observations=observations))
    assert "0.3" in text
    assert "fails required <= 0.05" in text


def test_the_dossier_records_the_rubric_and_plan_identities(
    rubric: QualificationRubric,
) -> None:
    text = render_dossier(_decide(rubric))
    assert rubric.identity[:12] in text
    assert PLAN_HASH[:12] in text


def test_the_dossier_carries_its_limitations(rubric: QualificationRubric) -> None:
    text = render_dossier(_decide(rubric))
    assert "Limitations" in text
    assert "not that it will earn money" in text
    assert "not probabilities of loss in the market" in text


def test_an_unmeasured_criterion_renders_as_not_measured(
    rubric: QualificationRubric,
) -> None:
    text = render_dossier(_decide(rubric, observations={}, evidence={}))
    assert "not measured" in text


def test_the_decision_is_deterministic(rubric: QualificationRubric) -> None:
    assert _decide(rubric).to_dict() == _decide(rubric).to_dict()


def test_every_criterion_appears_in_the_dossier(rubric: QualificationRubric) -> None:
    decision = _decide(rubric)
    assert len(decision.results) == len(rubric.criteria)
    text = render_dossier(decision)
    for criterion in rubric.criteria:
        assert criterion.name in text
