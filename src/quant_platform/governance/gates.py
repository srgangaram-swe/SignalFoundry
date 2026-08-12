"""Promotion gates and the human-authority boundary.

SF-S5-SL-MR5. Two separate ideas live here, and keeping them separate is the
point.

**Gates decide whether the evidence permits a recommendation.** They are
absolute and independent: each is evaluated on its own, none can be traded off
against another, and there is no weighted score. A challenger that wins on
Brier but ran for six days does not average its way to a recommendation.

**Only a human can approve.** :func:`recommend` produces a recommendation and
nothing else — it cannot approve, apply, roll back, or unfreeze. Applying a
recommendation requires an :class:`Approval` carrying a named approver, and the
apply path re-verifies every artifact identity and uses a compare-and-swap
against the lane head so two approvals racing cannot both win.

The distinction matters because a system that can promote its own challenger
will eventually promote one on evidence nobody examined. There is deliberately
no function in this module that both evaluates and applies.

**A failed gate is not overridable.** There is no ``force``, ``waive``, or
``override`` parameter, and a test parses the module AST to prove it. A gate
that should not apply is removed from a new frozen policy version, which leaves
a record.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from enum import StrEnum
from typing import Any, Final

from quant_platform.governance.comparison import PairedCohort
from quant_platform.governance.inference import (
    Margin,
    TestResult,
    TestVerdict,
    holm_adjust,
)
from quant_platform.shadow.contracts import canonical_digest

#: An approval older than this is stale. Evidence moves; a sign-off from a month
#: ago approved a comparison that no longer describes the lane.
APPROVAL_VALIDITY: Final = timedelta(days=7)

#: Refusal thresholds, not tuning knobs.
MAX_GATES: Final = 32
MAX_NAME_CHARS: Final = 64


class GovernanceError(ValueError):
    """Raised when a governance record or transition is unusable."""


class NotAuthorizedError(GovernanceError):
    """Raised when application is attempted without valid human approval.

    A distinct type so generic error handling cannot retry past it.
    """


class Recommendation(StrEnum):
    """What the evidence supports. Never an authorization to act."""

    PROMOTE = "promote"
    RETAIN_CHAMPION = "retain_champion"
    INSUFFICIENT_EVIDENCE = "insufficient_evidence"
    INVALID = "invalid"


@dataclass(frozen=True, slots=True)
class GateResult:
    """One absolute gate's outcome."""

    name: str
    satisfied: bool
    detail: str

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-friendly record."""
        return {"name": self.name, "satisfied": self.satisfied, "detail": self.detail}


@dataclass(frozen=True)
class FrozenPolicy:
    """Preregistered decision rules, fixed before any comparison is run.

    Raises:
        GovernanceError: On unusable thresholds.
    """

    version: str
    alpha: float
    minimum_days: int
    minimum_pairs: int
    minimum_coverage: float
    margin: Margin

    def __post_init__(self) -> None:
        if not isinstance(self.version, str) or not self.version.strip():
            raise GovernanceError("policy version must be a non-empty string")
        if not 0.0 < self.alpha < 0.5:
            raise GovernanceError("alpha must lie in (0, 0.5)")
        for field_name in ("minimum_days", "minimum_pairs"):
            value = getattr(self, field_name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise GovernanceError(f"{field_name} must be a positive int")
        if not 0.0 < self.minimum_coverage <= 1.0:
            raise GovernanceError("minimum_coverage must lie in (0, 1]")
        if not isinstance(self.margin, Margin):
            raise GovernanceError("margin must be a Margin")

    @property
    def identity(self) -> str:
        """Content identity, so a policy edited after results is detectable."""
        return canonical_digest(
            {
                "version": self.version,
                "alpha": self.alpha,
                "minimum_days": self.minimum_days,
                "minimum_pairs": self.minimum_pairs,
                "minimum_coverage": self.minimum_coverage,
                "margin": self.margin.to_dict(),
            }
        )

    def to_dict(self) -> dict[str, Any]:
        """Return the JSON-friendly frozen declaration."""
        return {
            "version": self.version,
            "identity": self.identity,
            "alpha": self.alpha,
            "minimum_days": self.minimum_days,
            "minimum_pairs": self.minimum_pairs,
            "minimum_coverage": self.minimum_coverage,
            "margin": self.margin.to_dict(),
            "policy": (
                "Gates are absolute and independent. There is no weighted score and no "
                "override; removing a gate requires publishing a new policy version."
            ),
        }


@dataclass(frozen=True)
class Decision:
    """A recommendation with every gate and test that produced it.

    A ``PROMOTE`` recommendation is evidence that promotion is *permissible*,
    never that it has happened. Applying requires a separate human approval.
    """

    recommendation: Recommendation
    policy_identity: str
    cohort_identity: str
    gates: tuple[GateResult, ...]
    tests: tuple[TestResult, ...]
    correction: Mapping[str, Any]
    decided_at: str

    def __post_init__(self) -> None:
        if self.recommendation is Recommendation.PROMOTE and any(
            not gate.satisfied for gate in self.gates
        ):
            raise GovernanceError(
                "cannot recommend promotion while a gate is unsatisfied; gates are "
                "absolute and there is no override"
            )

    @property
    def identity(self) -> str:
        """Content identity binding policy, cohort, gates, and tests."""
        return canonical_digest(self.to_dict())

    def to_dict(self) -> dict[str, Any]:
        """Return the complete JSON-friendly decision."""
        return {
            "recommendation": self.recommendation.value,
            "policy_identity": self.policy_identity,
            "cohort_identity": self.cohort_identity,
            "gates": [gate.to_dict() for gate in self.gates],
            "tests": [test.to_dict() for test in self.tests],
            "correction": dict(self.correction),
            "decided_at": self.decided_at,
            "authority": (
                "A recommendation is not an authorization. Promotion requires a separate "
                "human approval and a compare-and-swap against the lane head. No "
                "automation in this package can approve, apply, roll back, or unfreeze."
            ),
        }


def evaluate_gates(cohort: PairedCohort, policy: FrozenPolicy) -> tuple[GateResult, ...]:
    """Evaluate every absolute gate independently.

    Each gate is checked on its own and all results are returned, so a reader
    sees every failure rather than only the first. Short-circuiting would hide
    a coverage problem behind a duration problem.
    """
    distinct_days = len({item.as_of_date for item in cohort.pairs})
    return (
        GateResult(
            name="cohort_comparable",
            satisfied=cohort.comparable,
            detail=cohort.incomparable_reason or "arms have symmetric missingness",
        ),
        GateResult(
            name="minimum_duration",
            satisfied=distinct_days >= policy.minimum_days,
            detail=f"{distinct_days} distinct days against a minimum of {policy.minimum_days}",
        ),
        GateResult(
            name="minimum_pairs",
            satisfied=cohort.matched >= policy.minimum_pairs,
            detail=f"{cohort.matched} matched pairs against a minimum of {policy.minimum_pairs}",
        ),
        GateResult(
            name="minimum_coverage",
            satisfied=cohort.coverage >= policy.minimum_coverage,
            detail=(
                f"coverage {cohort.coverage:.3f} against a minimum of "
                f"{policy.minimum_coverage:.3f}"
            ),
        ),
    )


def recommend(
    cohort: PairedCohort,
    policy: FrozenPolicy,
    tests: Sequence[TestResult],
    *,
    now: datetime,
    expected_policy_identity: str | None = None,
) -> Decision:
    """Produce a recommendation. This function cannot promote anything.

    Args:
        cohort: The paired sample.
        policy: Preregistered rules.
        tests: The complete family of tests run against this cohort.
        now: Decision instant.
        expected_policy_identity: When supplied, the policy must match it, so a
            policy edited after seeing results is refused.

    Raises:
        GovernanceError: On a policy identity mismatch or an empty test family.
    """
    if expected_policy_identity is not None and policy.identity != expected_policy_identity:
        raise GovernanceError(
            f"policy {policy.version!r} does not match the preregistered identity: "
            f"running {policy.identity[:12]}, expected {expected_policy_identity[:12]}. "
            "Changing thresholds after seeing results is what the identity detects."
        )
    if not tests:
        raise GovernanceError("the test family must be non-empty")

    gates = evaluate_gates(cohort, policy)
    if not all(gate.satisfied for gate in gates):
        failing = [gate.name for gate in gates if not gate.satisfied]
        recommendation = (
            Recommendation.INVALID
            if "cohort_comparable" in failing
            else Recommendation.INSUFFICIENT_EVIDENCE
        )
        return Decision(
            recommendation=recommendation,
            policy_identity=policy.identity,
            cohort_identity=cohort.identity(),
            gates=gates,
            tests=tuple(tests),
            correction={},
            decided_at=now.astimezone(UTC).isoformat(),
        )

    if any(item.verdict is TestVerdict.UNDERPOWERED for item in tests):
        return Decision(
            recommendation=Recommendation.INSUFFICIENT_EVIDENCE,
            policy_identity=policy.identity,
            cohort_identity=cohort.identity(),
            gates=gates,
            tests=tuple(tests),
            correction={},
            decided_at=now.astimezone(UTC).isoformat(),
        )

    family = {item.name: item.p_value for item in tests if item.p_value is not None}
    if not family:
        raise GovernanceError("no test produced a p-value to correct")
    correction = holm_adjust(family, alpha=policy.alpha)
    promoted = all(correction["rejected"].values()) and all(
        item.verdict is TestVerdict.FAVOURS_CHALLENGER for item in tests
    )
    return Decision(
        recommendation=(Recommendation.PROMOTE if promoted else Recommendation.RETAIN_CHAMPION),
        policy_identity=policy.identity,
        cohort_identity=cohort.identity(),
        gates=gates,
        tests=tuple(tests),
        correction=correction,
        decided_at=now.astimezone(UTC).isoformat(),
    )


@dataclass(frozen=True)
class Approval:
    """A named human's sign-off on one specific decision.

    Bound to the decision identity, so an approval cannot be recycled onto a
    later comparison that produced a different result.

    Raises:
        GovernanceError: On a malformed or unbound approval.
    """

    approver: str
    decision_identity: str
    approved_at: datetime
    reference: str

    def __post_init__(self) -> None:
        for field_name in ("approver", "reference"):
            value = getattr(self, field_name)
            if not isinstance(value, str) or not value.strip() or value != value.strip():
                raise GovernanceError(f"{field_name} must be a non-empty unpadded string")
            if len(value) > MAX_NAME_CHARS:
                raise GovernanceError(f"{field_name} exceeds {MAX_NAME_CHARS} characters")
        if not isinstance(self.decision_identity, str) or len(self.decision_identity) != 64:
            raise GovernanceError("decision_identity must be a full SHA-256 digest")
        if not isinstance(self.approved_at, datetime) or self.approved_at.tzinfo is None:
            raise GovernanceError("approved_at must be a timezone-aware datetime")

    def is_current(self, *, now: datetime) -> bool:
        """Whether the approval is inside its validity window."""
        moment = now.astimezone(UTC)
        approved = self.approved_at.astimezone(UTC)
        return approved <= moment <= approved + APPROVAL_VALIDITY

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-friendly record."""
        return {
            "approver": self.approver,
            "decision_identity": self.decision_identity,
            "approved_at": self.approved_at.astimezone(UTC).isoformat(),
            "reference": self.reference,
        }


def authorize_apply(
    decision: Decision,
    approval: Approval,
    *,
    observed_lane_head: str,
    expected_lane_head: str,
    now: datetime,
) -> None:
    """Refuse unless a human approved *this* decision and the lane has not moved.

    The compare-and-swap is the concurrency control: two approvals racing to
    promote different challengers cannot both succeed, because the second sees
    a lane head that no longer matches what it was approved against.

    Raises:
        NotAuthorizedError: Naming which condition failed.
    """
    if not isinstance(decision, Decision):
        raise NotAuthorizedError("decision must be a Decision produced by recommend()")
    if not isinstance(approval, Approval):
        raise NotAuthorizedError("approval must be an Approval")
    if decision.recommendation is not Recommendation.PROMOTE:
        raise NotAuthorizedError(
            f"decision recommends {decision.recommendation.value!r}; only a PROMOTE "
            "recommendation may be applied, and a failed gate cannot be overridden"
        )
    if approval.decision_identity != decision.identity:
        raise NotAuthorizedError(
            f"approval is bound to decision {approval.decision_identity[:12]} but this "
            f"decision is {decision.identity[:12]}; an approval cannot be recycled onto "
            "a different comparison"
        )
    if not approval.is_current(now=now):
        raise NotAuthorizedError(
            f"approval by {approval.approver} is outside its "
            f"{APPROVAL_VALIDITY.days}-day window; evidence moves and a stale sign-off "
            "approved a comparison that no longer describes the lane"
        )
    if observed_lane_head != expected_lane_head:
        raise NotAuthorizedError(
            f"lane head moved from {expected_lane_head[:12]} to {observed_lane_head[:12]} "
            "since approval; another promotion won the race and this one must be "
            "re-evaluated against the new champion"
        )


__all__ = [
    "APPROVAL_VALIDITY",
    "Approval",
    "Decision",
    "FrozenPolicy",
    "GateResult",
    "GovernanceError",
    "NotAuthorizedError",
    "Recommendation",
    "authorize_apply",
    "evaluate_gates",
    "recommend",
]
