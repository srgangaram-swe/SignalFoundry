"""Frozen strategy-qualification rubric and dossier (SF-S4-MR9).

The last gate before a candidate may be paper traded. It exists because the
failure it prevents is the most expensive one in quantitative research: deciding
what "good enough" means *after* seeing how good the result was.

Three properties make that structurally hard rather than merely discouraged.

**The rubric is frozen and versioned before the candidate is scored.** A
:class:`QualificationRubric` publishes a content-derived SHA-256 over every
criterion, threshold, and direction. :func:`qualify` requires the identity that
was recorded beforehand and refuses to score against a rubric that has since
moved. Lowering a threshold to admit a candidate changes the digest.

**Every claim carries an evidence link.** A criterion without an
:class:`EvidenceLink` to immutable data, code, configuration, ledger, stress, or
uncertainty evidence cannot pass — it is recorded as unevidenced and forces
rejection. An unsupported metric is not a weaker pass; it is a failure.

**Failure is closed and total.** Any failed criterion, any missing evidence, any
reconciliation error, and any unreconciled stress path forces ``REJECTED``.
There is no weighted score, no "mostly passed", and no override parameter,
because every one of those is a mechanism for talking oneself into a trade.

The only two outcomes are ``QUALIFIED_FOR_PAPER`` and ``REJECTED``.
``QUALIFIED_FOR_PAPER`` authorizes **zero-capital paper evaluation only**. It is
not permission to deploy capital, not a broker authorization, and not a
statement that the strategy will make money.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Final, Literal

import numpy as np

#: Refusal thresholds, not tuning knobs.
MAX_CRITERIA: Final = 64
MAX_NAME_CHARS: Final = 80
MAX_TEXT_CHARS: Final = 400

#: The evidence kinds a criterion may cite. A criterion citing anything else is
#: refused, because "evidence: trust me" is the failure mode this list exists to
#: prevent.
EVIDENCE_KINDS: Final = (
    "dataset",
    "code",
    "configuration",
    "trial_ledger",
    "stress_study",
    "uncertainty",
    "risk_attribution",
    "capacity",
)

#: Comparison directions. Stated per criterion so a threshold can never be
#: silently reinterpreted as the opposite bound.
Direction = Literal["at_least", "at_most"]

Verdict = Literal["QUALIFIED_FOR_PAPER", "REJECTED"]


class QualificationError(ValueError):
    """Raised when a rubric, evidence link, or scoring request is unusable."""


def _name(value: object, *, field_name: str) -> str:
    if not isinstance(value, str):
        raise QualificationError(f"{field_name} must be a string, got {type(value).__name__}")
    text = value.strip()
    if not text or text != value:
        raise QualificationError(f"{field_name} must be non-empty and free of padding")
    if len(text) > MAX_NAME_CHARS:
        raise QualificationError(f"{field_name} exceeds {MAX_NAME_CHARS} characters")
    if not text.isascii():
        raise QualificationError(f"{field_name} must be ASCII")
    return text


def _text(value: object, *, field_name: str) -> str:
    if not isinstance(value, str):
        raise QualificationError(f"{field_name} must be a string")
    text = value.strip()
    if not text:
        raise QualificationError(f"{field_name} must be non-empty")
    if len(text) > MAX_TEXT_CHARS:
        raise QualificationError(f"{field_name} exceeds {MAX_TEXT_CHARS} characters")
    return text


def _digest(payload: Any) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _finite(value: object, *, field_name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise QualificationError(f"{field_name} must be a real number")
    numeric = float(value)
    if not np.isfinite(numeric):
        raise QualificationError(f"{field_name} must be finite")
    return numeric


@dataclass(frozen=True, slots=True)
class EvidenceLink:
    """An immutable pointer to the artifact backing one claim.

    ``content_hash`` is required. A link naming a file without pinning its
    contents lets the evidence change after the claim is made, which is the
    same failure as having no evidence at all.
    """

    kind: str
    identifier: str
    content_hash: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "kind", _name(self.kind, field_name="evidence kind"))
        if self.kind not in EVIDENCE_KINDS:
            raise QualificationError(
                f"unsupported evidence kind {self.kind!r}; supported: "
                f"{', '.join(EVIDENCE_KINDS)}"
            )
        object.__setattr__(
            self, "identifier", _name(self.identifier, field_name="evidence identifier")
        )
        if not isinstance(self.content_hash, str) or len(self.content_hash) != 64:
            raise QualificationError(
                f"evidence for {self.identifier!r} must pin a full SHA-256 content hash; "
                "a link that does not pin contents lets the evidence change after the claim"
            )
        if not all(character in "0123456789abcdef" for character in self.content_hash):
            raise QualificationError("content_hash must be lowercase hexadecimal")

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-friendly link."""
        return {
            "kind": self.kind,
            "identifier": self.identifier,
            "content_hash": self.content_hash,
        }


@dataclass(frozen=True, slots=True)
class Criterion:
    """One frozen, checkable qualification requirement.

    Attributes:
        name: Identifier, e.g. ``net_return_over_baseline``.
        question: The plain-language question this criterion answers, carried
            into the dossier so a reader need not reverse-engineer intent from
            a threshold.
        threshold: The bound.
        direction: Whether the observed value must be ``at_least`` or
            ``at_most`` the threshold.
        required_evidence: Evidence kinds that must be cited for a pass.
    """

    name: str
    question: str
    threshold: float
    direction: Direction
    required_evidence: tuple[str, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "name", _name(self.name, field_name="criterion name"))
        object.__setattr__(self, "question", _text(self.question, field_name="question"))
        object.__setattr__(self, "threshold", _finite(self.threshold, field_name="threshold"))
        if self.direction not in ("at_least", "at_most"):
            raise QualificationError("direction must be 'at_least' or 'at_most'")
        required = tuple(self.required_evidence)
        if not required:
            raise QualificationError(
                f"criterion {self.name!r} must require at least one evidence kind; a "
                "criterion that needs no evidence cannot be checked"
            )
        for kind in required:
            if kind not in EVIDENCE_KINDS:
                raise QualificationError(f"unsupported required evidence kind {kind!r}")
        if len(set(required)) != len(required):
            raise QualificationError("required evidence kinds must be unique")
        object.__setattr__(self, "required_evidence", tuple(sorted(required)))

    def satisfied_by(self, observed: float) -> bool:
        """Whether ``observed`` meets this criterion's bound."""
        if self.direction == "at_least":
            return observed >= self.threshold
        return observed <= self.threshold

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-friendly declaration."""
        return {
            "name": self.name,
            "question": self.question,
            "threshold": self.threshold,
            "direction": self.direction,
            "required_evidence": list(self.required_evidence),
        }


@dataclass(frozen=True)
class QualificationRubric:
    """The complete, versioned rubric, frozen before any candidate is scored.

    Raises:
        QualificationError: On duplicate or missing criteria.
    """

    version: str
    criteria: tuple[Criterion, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "version", _name(self.version, field_name="rubric version"))
        criteria = tuple(self.criteria)
        if not criteria:
            raise QualificationError("a rubric must declare at least one criterion")
        if len(criteria) > MAX_CRITERIA:
            raise QualificationError(f"rubric exceeds the {MAX_CRITERIA}-criterion ceiling")
        names = [item.name for item in criteria]
        if len(set(names)) != len(names):
            raise QualificationError("criterion names must be unique within a rubric")
        object.__setattr__(self, "criteria", tuple(sorted(criteria, key=lambda x: x.name)))

    @property
    def identity(self) -> str:
        """Content identity frozen before qualification."""
        return _digest(
            {
                "version": self.version,
                "criteria": [item.to_dict() for item in self.criteria],
            }
        )

    def to_dict(self) -> dict[str, Any]:
        """Return the JSON-friendly frozen declaration."""
        return {
            "version": self.version,
            "identity": self.identity,
            "criteria": [item.to_dict() for item in self.criteria],
            "frozen_before_scoring": True,
        }


def verify_frozen_rubric(rubric: QualificationRubric, expected_identity: str) -> None:
    """Refuse a rubric that differs from the one frozen before scoring.

    Raises:
        QualificationError: On any divergence, naming both identities.
    """
    if not isinstance(expected_identity, str) or len(expected_identity) != 64:
        raise QualificationError("expected_identity must be a full SHA-256 digest")
    if rubric.identity != expected_identity:
        raise QualificationError(
            f"rubric {rubric.version!r} does not match the frozen declaration: "
            f"scoring {rubric.identity[:12]}, frozen {expected_identity[:12]}. "
            "Adjusting a threshold after seeing the candidate's numbers makes the "
            "rubric a function of the result it is supposed to judge."
        )


def standard_paper_rubric() -> QualificationRubric:
    """Return the default rubric for paper-trading qualification.

    The thresholds encode the questions the work item names: does it beat an
    investable baseline **after costs**, does the edge survive multiple-testing
    correction, does it survive parameter/regime/universe robustness and
    execution stress, is it too concentrated, and does capacity exist. Every one
    is deliberately a bound a mediocre-but-lucky candidate fails.
    """
    return QualificationRubric(
        version="paper-v1",
        criteria=(
            Criterion(
                name="net_return_over_baseline",
                question=(
                    "Does the candidate beat a simple investable baseline after all "
                    "modeled costs?"
                ),
                threshold=0.0,
                direction="at_least",
                required_evidence=("dataset", "code", "configuration"),
            ),
            Criterion(
                name="adjusted_p_value",
                question=(
                    "Does the edge survive multiple-testing correction across the "
                    "complete trial family?"
                ),
                threshold=0.05,
                direction="at_most",
                required_evidence=("trial_ledger",),
            ),
            Criterion(
                name="stress_downside_net_return",
                question=(
                    "Under frozen execution perturbation, does the 5th-percentile path "
                    "remain non-negative?"
                ),
                threshold=0.0,
                direction="at_least",
                required_evidence=("stress_study",),
            ),
            Criterion(
                name="stress_insolvent_paths",
                question="Did any perturbed path drive the account to insolvency?",
                threshold=0.0,
                direction="at_most",
                required_evidence=("stress_study",),
            ),
            Criterion(
                name="max_drawdown",
                question="Is the worst peak-to-trough loss within the declared tolerance?",
                threshold=-0.25,
                direction="at_least",
                required_evidence=("dataset", "risk_attribution"),
            ),
            Criterion(
                name="top_name_concentration",
                question="Is the result carried by more than a handful of names?",
                threshold=0.35,
                direction="at_most",
                required_evidence=("risk_attribution",),
            ),
            Criterion(
                name="capacity_utilization",
                question=(
                    "Does the strategy fit inside its measured liquidity and borrow " "capacity?"
                ),
                threshold=1.0,
                direction="at_most",
                required_evidence=("capacity",),
            ),
            Criterion(
                name="uncertainty_lower_bound",
                question=(
                    "Is the dependence-aware lower confidence bound on mean return " "above zero?"
                ),
                threshold=0.0,
                direction="at_least",
                required_evidence=("uncertainty",),
            ),
        ),
    )


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class CriterionResult:
    """One criterion's outcome with the evidence that backs it."""

    name: str
    question: str
    observed: float | None
    threshold: float
    direction: str
    passed: bool
    evidence: tuple[EvidenceLink, ...]
    failure_reason: str | None

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-friendly result row."""
        return {
            "name": self.name,
            "question": self.question,
            "observed": self.observed,
            "threshold": self.threshold,
            "direction": self.direction,
            "passed": self.passed,
            "evidence": [link.to_dict() for link in self.evidence],
            "failure_reason": self.failure_reason,
        }


@dataclass(frozen=True)
class QualificationDecision:
    """The final, immutable verdict.

    ``QUALIFIED_FOR_PAPER`` authorizes zero-capital paper evaluation only. It is
    never authorization for live capital and never a profit guarantee.
    """

    verdict: Verdict
    candidate_id: str
    rubric_version: str
    rubric_identity: str
    plan_hash: str
    decided_at: str
    results: tuple[CriterionResult, ...]
    blocking_failures: tuple[str, ...]

    @property
    def qualified(self) -> bool:
        """Whether the candidate may proceed to paper evaluation."""
        return self.verdict == "QUALIFIED_FOR_PAPER"

    def to_dict(self) -> dict[str, Any]:
        """Return the complete machine-readable dossier."""
        return {
            "verdict": self.verdict,
            "candidate_id": self.candidate_id,
            "rubric_version": self.rubric_version,
            "rubric_identity": self.rubric_identity,
            "plan_hash": self.plan_hash,
            "decided_at": self.decided_at,
            "criteria": [item.to_dict() for item in self.results],
            "blocking_failures": list(self.blocking_failures),
            "authorization": (
                "QUALIFIED_FOR_PAPER authorizes zero-capital paper evaluation only. "
                "It is not authorization for live capital, not broker access, and not "
                "a statement that the strategy will be profitable."
                if self.qualified
                else "REJECTED. No paper or live evaluation is authorized."
            ),
            "simulation_only": True,
        }


def qualify(
    *,
    candidate_id: str,
    rubric: QualificationRubric,
    expected_rubric_identity: str,
    observations: Mapping[str, float],
    evidence: Mapping[str, Sequence[EvidenceLink]],
    plan_hash: str,
    decided_at: str,
    reconciliation_ok: bool = True,
) -> QualificationDecision:
    """Score one candidate against the frozen rubric and return the verdict.

    Fail-closed at every step. A criterion is rejected when its observation is
    missing, non-finite, unevidenced, or outside its bound; the decision is
    ``REJECTED`` if *any* criterion fails or if ledger reconciliation failed.
    There is no override.

    Args:
        observations: Observed value per criterion name. A criterion with no
            observation fails; it does not default to a pass.
        evidence: Evidence links per criterion name.
        reconciliation_ok: Whether the ledger reconciled. ``False`` forces
            rejection regardless of every other number, because an account that
            does not balance makes all of them meaningless.

    Raises:
        QualificationError: If the rubric diverges from its frozen identity, or
            the plan hash or timestamp is malformed.
    """
    verify_frozen_rubric(rubric, expected_rubric_identity)
    candidate = _name(candidate_id, field_name="candidate_id")
    if not isinstance(plan_hash, str) or len(plan_hash) != 64:
        raise QualificationError(
            "plan_hash must be the full SHA-256 of the frozen research plan the "
            "decision is made under"
        )
    try:
        datetime.fromisoformat(decided_at)
    except (TypeError, ValueError) as exc:
        raise QualificationError("decided_at must be an ISO-8601 timestamp") from exc

    results: list[CriterionResult] = []
    blocking: list[str] = []

    for criterion in rubric.criteria:
        links = tuple(evidence.get(criterion.name, ()))
        cited = {link.kind for link in links}
        missing_evidence = sorted(set(criterion.required_evidence) - cited)
        raw = observations.get(criterion.name)

        observed: float | None = None
        reason: str | None = None
        passed = False

        if raw is None:
            reason = "no observation supplied; an unmeasured criterion cannot pass"
        elif isinstance(raw, bool) or not isinstance(raw, (int, float)):
            reason = "observation is not a real number"
        elif not np.isfinite(float(raw)):
            reason = "observation is not finite"
        else:
            observed = float(raw)
            if missing_evidence:
                reason = (
                    f"missing required evidence: {', '.join(missing_evidence)}; an "
                    "unevidenced metric is a failure, not a weaker pass"
                )
            elif not criterion.satisfied_by(observed):
                comparison = ">=" if criterion.direction == "at_least" else "<="
                reason = (
                    f"observed {observed:.6g} fails required {comparison} "
                    f"{criterion.threshold:.6g}"
                )
            else:
                passed = True

        if not passed:
            blocking.append(criterion.name)
        results.append(
            CriterionResult(
                name=criterion.name,
                question=criterion.question,
                observed=observed,
                threshold=criterion.threshold,
                direction=criterion.direction,
                passed=passed,
                evidence=links,
                failure_reason=reason,
            )
        )

    if not reconciliation_ok:
        blocking.append("ledger_reconciliation")

    verdict: Verdict = "QUALIFIED_FOR_PAPER" if not blocking else "REJECTED"
    return QualificationDecision(
        verdict=verdict,
        candidate_id=candidate,
        rubric_version=rubric.version,
        rubric_identity=rubric.identity,
        plan_hash=plan_hash,
        decided_at=decided_at,
        results=tuple(results),
        blocking_failures=tuple(sorted(blocking)),
    )


def render_dossier(decision: QualificationDecision) -> str:
    """Render the human-readable dossier.

    Failures are listed **before** passes. A reader who stops after the first
    screen should see what is wrong with the candidate, not what is right.
    """
    lines: list[str] = []
    lines.append(f"# Qualification dossier — {decision.candidate_id}")
    lines.append("")
    lines.append(f"**Verdict: {decision.verdict}**")
    lines.append("")
    if decision.qualified:
        lines.append(
            "> Authorizes **zero-capital paper evaluation only**. Not authorization "
            "for live capital, not broker access, and not a statement that the "
            "strategy will be profitable."
        )
    else:
        lines.append(
            "> **No paper or live evaluation is authorized.** Every blocking failure "
            "below must be resolved and the candidate re-scored against a rubric "
            "frozen before the new evidence was seen."
        )
    lines.append("")
    lines.append(f"- Rubric: `{decision.rubric_version}` (`{decision.rubric_identity[:12]}`)")
    lines.append(f"- Frozen research plan: `{decision.plan_hash[:12]}`")
    lines.append(f"- Decided at: {decision.decided_at}")
    lines.append("")

    failures = [item for item in decision.results if not item.passed]
    passes = [item for item in decision.results if item.passed]

    if failures:
        lines.append(f"## Blocking failures ({len(failures)})")
        lines.append("")
        for item in failures:
            observed = "not measured" if item.observed is None else f"{item.observed:.6g}"
            lines.append(f"### {item.name}")
            lines.append("")
            lines.append(f"{item.question}")
            lines.append("")
            lines.append(f"- Observed: **{observed}**")
            lines.append(f"- Required: {item.direction.replace('_', ' ')} {item.threshold:.6g}")
            lines.append(f"- Reason: {item.failure_reason}")
            lines.append("")

    if passes:
        lines.append(f"## Satisfied criteria ({len(passes)})")
        lines.append("")
        lines.append("| Criterion | Observed | Required | Evidence |")
        lines.append("| --- | --- | --- | --- |")
        for item in passes:
            observed = "—" if item.observed is None else f"{item.observed:.6g}"
            requirement = f"{item.direction.replace('_', ' ')} {item.threshold:.6g}"
            citations = ", ".join(
                f"`{link.kind}:{link.content_hash[:8]}`" for link in item.evidence
            )
            lines.append(f"| {item.name} | {observed} | {requirement} | {citations} |")
        lines.append("")

    lines.append("## Limitations")
    lines.append("")
    lines.append("- Every number above is simulated. No live or paper capital has been at risk.")
    lines.append(
        "- Monte Carlo frequencies come from a declared perturbation family and are "
        "not probabilities of loss in the market."
    )
    lines.append(
        "- A frozen rubric prevents moving the bar; it does not make the bar correct. "
        "The thresholds are judgements, recorded so they can be argued with."
    )
    lines.append(
        "- Passing every criterion establishes that the candidate was not obviously "
        "broken, not that it will earn money."
    )
    return "\n".join(lines) + "\n"


__all__ = [
    "EVIDENCE_KINDS",
    "MAX_CRITERIA",
    "Criterion",
    "CriterionResult",
    "Direction",
    "EvidenceLink",
    "QualificationDecision",
    "QualificationError",
    "QualificationRubric",
    "Verdict",
    "qualify",
    "render_dossier",
    "standard_paper_rubric",
    "verify_frozen_rubric",
]
