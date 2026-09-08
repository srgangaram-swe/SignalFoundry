"""A versioned readiness checklist that cannot be talked into saying READY.

SF-S5-MR10. This is the last gate before capital, and it is designed around one
assumption: the person running it will want it to pass. That person may be
tired, may have spent a sprint building the infrastructure, and may be genuinely
convinced the remaining items are formalities. The checklist is written so that
conviction cannot move it.

**There is no score.** Every item is required. A weighted score would let nine
strong items outvote one missing legal review, and the missing legal review is
the one that matters.

**There is no override.** No `force`, `waive`, `skip`, or `acknowledge_risk`
parameter exists, and a test parses the module AST to prove it. An item that
should not apply must be removed from a **new version** of the checklist, which
leaves a record of who removed it and when.

**Attestations are recorded, not verified.** Employment-policy and legal review
cannot be checked by software. The framework records that a named human attested
on a date, and says plainly that it cannot confirm the attestation is true. An
attestation that is stale, or that names no attester, is refused — those *are*
checkable, and checking what can be checked is the honest boundary.

**The default verdict is NOT_READY**, produced by an empty evidence set. Readiness
is something you demonstrate, not something you inherit by omission.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from enum import StrEnum
from typing import Any, Final

#: Refusal thresholds, not tuning knobs.
MAX_ITEMS: Final = 128
MAX_TEXT_CHARS: Final = 512
MAX_NAME_CHARS: Final = 64

#: An attestation older than this is stale. Employment policy, legal posture, and
#: personal circumstances change; a two-year-old sign-off is not current consent.
DEFAULT_ATTESTATION_VALIDITY: Final = timedelta(days=180)


class ReadinessError(ValueError):
    """Raised when a checklist, evidence item, or decision is malformed."""


class Verdict(StrEnum):
    """The only two outcomes. There is no conditional or partial readiness."""

    NOT_READY = "NOT_READY"
    READY_FOR_MINIMAL_CAPITAL = "READY_FOR_MINIMAL_CAPITAL"


class Category(StrEnum):
    """What kind of failure an unmet item represents.

    Categories exist because the *remedy* differs: an evidence gap is closed by
    running something, a legal gap is not.
    """

    EVIDENCE = "evidence"
    OPERATIONAL = "operational"
    SECURITY = "security"
    RECONCILIATION = "reconciliation"
    POLICY = "policy"
    LEGAL = "legal"


#: Categories whose items can only be satisfied by a human attestation. Software
#: can record these; it cannot verify them.
ATTESTATION_ONLY: Final[frozenset[Category]] = frozenset({Category.POLICY, Category.LEGAL})


def _text(value: object, *, field_name: str, limit: int = MAX_TEXT_CHARS) -> str:
    if not isinstance(value, str) or not value.strip() or value != value.strip():
        raise ReadinessError(f"{field_name} must be a non-empty unpadded string")
    if len(value) > limit:
        raise ReadinessError(f"{field_name} exceeds {limit} characters")
    return value


def _digest(payload: Any) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    ).hexdigest()


@dataclass(frozen=True, slots=True)
class ChecklistItem:
    """One required condition. Every item is mandatory; none is weighted."""

    key: str
    category: Category
    requirement: str
    verified_by: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "key", _text(self.key, field_name="key", limit=MAX_NAME_CHARS))
        if not isinstance(self.category, Category):
            raise ReadinessError("category must be a Category member")
        object.__setattr__(self, "requirement", _text(self.requirement, field_name="requirement"))
        object.__setattr__(self, "verified_by", _text(self.verified_by, field_name="verified_by"))

    @property
    def is_attestation_only(self) -> bool:
        """Whether only a human can satisfy this item."""
        return self.category in ATTESTATION_ONLY

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-friendly declaration."""
        return {
            "key": self.key,
            "category": self.category.value,
            "requirement": self.requirement,
            "verified_by": self.verified_by,
            "attestation_only": self.is_attestation_only,
        }


@dataclass(frozen=True)
class ReadinessChecklist:
    """A versioned, frozen set of required conditions.

    Raises:
        ReadinessError: On duplicate keys, an empty set, or an oversized one.
    """

    version: str
    items: tuple[ChecklistItem, ...]

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "version", _text(self.version, field_name="version", limit=MAX_NAME_CHARS)
        )
        items = tuple(self.items)
        if not items:
            raise ReadinessError(
                "a checklist must contain at least one item; an empty checklist would make "
                "every candidate trivially ready"
            )
        if len(items) > MAX_ITEMS:
            raise ReadinessError(f"checklist exceeds the {MAX_ITEMS}-item ceiling")
        keys = [item.key for item in items]
        if len(set(keys)) != len(keys):
            raise ReadinessError("checklist item keys must be unique")
        object.__setattr__(self, "items", tuple(sorted(items, key=lambda item: item.key)))

    @property
    def identity(self) -> str:
        """Content identity, so a silently edited checklist is detectable."""
        return _digest({"version": self.version, "items": [i.to_dict() for i in self.items]})

    def keys(self) -> tuple[str, ...]:
        """Every required key, in deterministic order."""
        return tuple(item.key for item in self.items)

    def to_dict(self) -> dict[str, Any]:
        """Return the JSON-friendly declaration."""
        return {
            "version": self.version,
            "identity": self.identity,
            "item_count": len(self.items),
            "items": [item.to_dict() for item in self.items],
            "policy": (
                "Every item is required. There is no score, no weighting, and no override; "
                "removing an item requires publishing a new checklist version."
            ),
        }


@dataclass(frozen=True, slots=True)
class Attestation:
    """A named human's dated sign-off on something software cannot verify."""

    attester: str
    statement: str
    attested_at: datetime
    reference: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "attester", _text(self.attester, field_name="attester", limit=MAX_NAME_CHARS)
        )
        object.__setattr__(self, "statement", _text(self.statement, field_name="statement"))
        if not isinstance(self.attested_at, datetime) or self.attested_at.tzinfo is None:
            raise ReadinessError("attested_at must be a timezone-aware datetime")
        object.__setattr__(self, "attested_at", self.attested_at.astimezone(UTC))
        if self.reference is not None:
            object.__setattr__(self, "reference", _text(self.reference, field_name="reference"))

    def is_current(self, *, now: datetime, validity: timedelta) -> bool:
        """Whether this attestation is still within its validity window."""
        moment = now.astimezone(UTC)
        if moment < self.attested_at:
            return False
        return moment - self.attested_at <= validity

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-friendly record."""
        return {
            "attester": self.attester,
            "statement": self.statement,
            "attested_at": self.attested_at.isoformat(),
            "reference": self.reference,
            "verification_note": (
                "Recorded, not verified. Software cannot confirm that a policy or legal "
                "review actually occurred or reached this conclusion."
            ),
        }


@dataclass(frozen=True, slots=True)
class ItemResult:
    """Whether one item is satisfied, and why not when it is not."""

    key: str
    satisfied: bool
    category: Category
    detail: str

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-friendly record."""
        return {
            "key": self.key,
            "satisfied": self.satisfied,
            "category": self.category.value,
            "detail": self.detail,
        }


@dataclass(frozen=True)
class ReadinessDecision:
    """The machine-readable readiness verdict.

    ``READY_FOR_MINIMAL_CAPITAL`` authorizes a separately configured, hard-capped
    minimal-capital deployment and nothing else. It is not a profit expectation,
    not permission to raise the cap, and not permanent.
    """

    verdict: Verdict
    checklist_version: str
    checklist_identity: str
    results: tuple[ItemResult, ...]
    unmet: tuple[str, ...]
    decided_at: str

    def __post_init__(self) -> None:
        if self.verdict is Verdict.READY_FOR_MINIMAL_CAPITAL and self.unmet:
            raise ReadinessError(
                f"cannot be READY with {len(self.unmet)} unmet item(s): {list(self.unmet)[:5]}. "
                "Every item is required and there is no override."
            )
        if self.verdict is Verdict.NOT_READY and not self.unmet:
            raise ReadinessError(
                "a NOT_READY verdict must name the unmet items; an unexplained refusal "
                "cannot be acted on"
            )

    @property
    def ready(self) -> bool:
        """Whether minimal-capital deployment is authorized."""
        return self.verdict is Verdict.READY_FOR_MINIMAL_CAPITAL

    def unmet_by_category(self) -> dict[str, list[str]]:
        """Unmet keys grouped by category, since remedies differ by kind."""
        grouped: dict[str, list[str]] = {}
        for result in self.results:
            if not result.satisfied:
                grouped.setdefault(result.category.value, []).append(result.key)
        return {key: sorted(value) for key, value in sorted(grouped.items())}

    def to_dict(self) -> dict[str, Any]:
        """Return the complete machine-readable decision."""
        return {
            "verdict": self.verdict.value,
            "ready": self.ready,
            "checklist_version": self.checklist_version,
            "checklist_identity": self.checklist_identity,
            "decided_at": self.decided_at,
            "satisfied_count": sum(1 for item in self.results if item.satisfied),
            "total_count": len(self.results),
            "unmet": list(self.unmet),
            "unmet_by_category": self.unmet_by_category(),
            "results": [item.to_dict() for item in self.results],
            "authorization": (
                "READY_FOR_MINIMAL_CAPITAL authorizes a separately configured, hard-capped, "
                "owner-approved minimal-capital deployment only. It is not a profit "
                "expectation, not permission to raise the cap, and not permanent."
                if self.ready
                else "NOT_READY. No capital deployment is authorized."
            ),
            "simulation_only": not self.ready,
        }


def render_readiness_report(decision: ReadinessDecision) -> str:
    """Render a human-readable report, unmet items first.

    Failures lead. A report that opens with everything that passed invites the
    reader to skim to a conclusion the evidence does not support.
    """
    lines = [
        f"# Live readiness: {decision.verdict.value}",
        "",
        f"Checklist `{decision.checklist_version}` ({decision.checklist_identity[:12]})",
        f"Decided {decision.decided_at}",
        f"Satisfied {sum(1 for i in decision.results if i.satisfied)} of {len(decision.results)}",
        "",
    ]
    unmet = [item for item in decision.results if not item.satisfied]
    if unmet:
        lines.append(f"## Unmet ({len(unmet)}) — every one of these blocks deployment")
        lines.append("")
        for item in unmet:
            lines.append(f"- **{item.key}** [{item.category.value}] — {item.detail}")
        lines.append("")
    met = [item for item in decision.results if item.satisfied]
    if met:
        lines.append(f"## Satisfied ({len(met)})")
        lines.append("")
        for item in met:
            lines.append(f"- {item.key} [{item.category.value}] — {item.detail}")
        lines.append("")
    lines.append("---")
    lines.append("")
    lines.append(decision.to_dict()["authorization"])
    return "\n".join(lines)


def minimal_capital_checklist() -> ReadinessChecklist:
    """Return the standard checklist for a first minimal-capital deployment.

    Derived from the capital-authorization gate in the broker connectivity
    requirements (SF-S5-MR2), extended with the policy and legal reviews that
    work item #49 names.
    """
    return ReadinessChecklist(
        version="minimal-capital-1",
        items=(
            ChecklistItem(
                key="qualified_candidate",
                category=Category.EVIDENCE,
                requirement="A strategy holds a QUALIFIED_FOR_PAPER verdict under a frozen rubric",
                verified_by="qualification dossier with rubric identity and evidence hashes",
            ),
            ChecklistItem(
                key="paper_duration",
                category=Category.EVIDENCE,
                requirement="Paper trading has run for the declared minimum duration with no gaps",
                verified_by="paper run records covering the whole interval",
            ),
            ChecklistItem(
                key="paper_stability",
                category=Category.EVIDENCE,
                requirement="Paper results are stable across the evaluation period",
                verified_by="per-period evidence with dependence-aware uncertainty",
            ),
            ChecklistItem(
                key="cost_model_validated",
                category=Category.EVIDENCE,
                requirement="Modelled costs are validated against observed paper fills",
                verified_by="slippage and commission comparison over the paper period",
            ),
            ChecklistItem(
                key="reconciliation_clean",
                category=Category.RECONCILIATION,
                requirement="Every reconciliation over the paper period passed with zero "
                "unexplained divergence",
                verified_by="reconciliation log across the paper period",
            ),
            ChecklistItem(
                key="broker_state_known",
                category=Category.RECONCILIATION,
                requirement="Broker positions, cash, and open orders are currently known and "
                "match local state",
                verified_by="a passing reconciliation report dated within the freshness bound",
            ),
            ChecklistItem(
                key="operational_rehearsal",
                category=Category.OPERATIONAL,
                requirement="An operational rehearsal including a deliberate failure drill has "
                "been performed",
                verified_by="rehearsal record with injected failure and observed recovery",
            ),
            ChecklistItem(
                key="broker_failure_drill",
                category=Category.OPERATIONAL,
                requirement="Behaviour under broker outage and rejected orders has been exercised",
                verified_by="fault-injection record showing halt rather than retry",
            ),
            ChecklistItem(
                key="kill_switch_verified",
                category=Category.OPERATIONAL,
                requirement="The kill switch stops submission within the declared bound",
                verified_by="kill-switch test record with measured latency",
            ),
            ChecklistItem(
                key="deactivation_procedure",
                category=Category.OPERATIONAL,
                requirement="A deactivation and rollback procedure exists and has been exercised",
                verified_by="runbook plus an execution record",
            ),
            ChecklistItem(
                key="capital_cap_enforced",
                category=Category.SECURITY,
                requirement="A capital-at-risk cap is set and enforced in code",
                verified_by="configuration plus the test proving a breach is refused",
            ),
            ChecklistItem(
                key="risk_limits_enforced",
                category=Category.SECURITY,
                requirement="Position, notional, drawdown, and daily-loss limits are enforced "
                "in code",
                verified_by="configuration plus enforcement tests",
            ),
            ChecklistItem(
                key="credential_custody",
                category=Category.SECURITY,
                requirement="Credentials are in the secret store only, with rotation recorded",
                verified_by="custody policy plus a negative test that environment credentials "
                "are refused",
            ),
            ChecklistItem(
                key="audit_and_tax_export",
                category=Category.OPERATIONAL,
                requirement="Order, fill, and position history exports in a form suitable for "
                "audit and tax reporting",
                verified_by="a generated export over the paper period",
            ),
            ChecklistItem(
                key="employment_policy_review",
                category=Category.POLICY,
                requirement="Personal-trading and outside-activity policy has been reviewed and "
                "permits this activity",
                verified_by="dated attestation by the owner naming the policy reviewed",
            ),
            ChecklistItem(
                key="legal_regulatory_review",
                category=Category.LEGAL,
                requirement="Applicable legal, regulatory, and tax obligations have been reviewed",
                verified_by="dated attestation by the owner naming the review performed",
            ),
            ChecklistItem(
                key="owner_approval",
                category=Category.POLICY,
                requirement="The owner has approved this deployment, naming the cap and the date",
                verified_by="written approval referencing every other item",
            ),
        ),
    )


def evaluate_readiness(
    checklist: ReadinessChecklist,
    *,
    evidence: Mapping[str, bool],
    attestations: Mapping[str, Attestation] | None = None,
    now: datetime,
    attestation_validity: timedelta = DEFAULT_ATTESTATION_VALIDITY,
    expected_identity: str | None = None,
) -> ReadinessDecision:
    """Evaluate every checklist item and return a machine-readable decision.

    An item absent from ``evidence`` is **unmet**, not skipped: readiness is
    demonstrated, never inherited by omission. Attestation-only items require a
    current :class:`Attestation` in addition to their evidence flag, because a
    boolean is not a record of who decided what, and when.

    Args:
        checklist: The frozen checklist to evaluate against.
        evidence: Item key → whether the underlying condition is demonstrated.
        attestations: Item key → the human sign-off for attestation-only items.
        now: Evaluation time, used for attestation freshness.
        attestation_validity: How long a sign-off remains current.
        expected_identity: When supplied, the checklist must match it, so a
            silently edited checklist is detectable.

    Raises:
        ReadinessError: On a checklist identity mismatch, an unknown evidence
            key, or a malformed input.
    """
    if expected_identity is not None:
        if not isinstance(expected_identity, str) or len(expected_identity) != 64:
            raise ReadinessError("expected_identity must be a full SHA-256 digest")
        if checklist.identity != expected_identity:
            raise ReadinessError(
                f"checklist {checklist.version!r} does not match the expected identity: "
                f"running {checklist.identity[:12]}, expected {expected_identity[:12]}. "
                "Editing the checklist to admit a candidate is exactly what the identity "
                "exists to catch."
            )
    if not isinstance(evidence, Mapping):
        raise ReadinessError("evidence must be a mapping of item key to bool")
    unknown = sorted(set(evidence) - set(checklist.keys()))
    if unknown:
        raise ReadinessError(
            f"evidence names {unknown} which are not in checklist {checklist.version!r}; "
            "supplying evidence for a nonexistent item usually means the wrong checklist "
            "version is in use"
        )
    supplied = dict(attestations or {})
    unknown_attestations = sorted(set(supplied) - set(checklist.keys()))
    if unknown_attestations:
        raise ReadinessError(f"attestations name unknown items {unknown_attestations}")
    moment = now.astimezone(UTC)

    results: list[ItemResult] = []
    for item in checklist.items:
        demonstrated = evidence.get(item.key)
        if demonstrated is None:
            results.append(
                ItemResult(
                    key=item.key,
                    satisfied=False,
                    category=item.category,
                    detail="no evidence supplied; readiness is demonstrated, not assumed",
                )
            )
            continue
        if not isinstance(demonstrated, bool):
            raise ReadinessError(f"evidence for {item.key!r} must be a bool")
        if not demonstrated:
            results.append(
                ItemResult(
                    key=item.key,
                    satisfied=False,
                    category=item.category,
                    detail=f"not satisfied: {item.requirement}",
                )
            )
            continue
        if item.is_attestation_only:
            attestation = supplied.get(item.key)
            if attestation is None:
                results.append(
                    ItemResult(
                        key=item.key,
                        satisfied=False,
                        category=item.category,
                        detail=(
                            "requires a named, dated attestation; a boolean does not record "
                            "who decided what, and when"
                        ),
                    )
                )
                continue
            if not attestation.is_current(now=moment, validity=attestation_validity):
                results.append(
                    ItemResult(
                        key=item.key,
                        satisfied=False,
                        category=item.category,
                        detail=(
                            f"attestation by {attestation.attester} dated "
                            f"{attestation.attested_at.date()} is outside the "
                            f"{attestation_validity.days}-day validity window; policy and "
                            "legal posture change, and an old sign-off is not current consent"
                        ),
                    )
                )
                continue
        results.append(
            ItemResult(
                key=item.key,
                satisfied=True,
                category=item.category,
                detail=f"satisfied via {item.verified_by}",
            )
        )

    unmet = tuple(item.key for item in results if not item.satisfied)
    return ReadinessDecision(
        verdict=Verdict.NOT_READY if unmet else Verdict.READY_FOR_MINIMAL_CAPITAL,
        checklist_version=checklist.version,
        checklist_identity=checklist.identity,
        results=tuple(results),
        unmet=unmet,
        decided_at=moment.isoformat(),
    )


__all__ = [
    "ATTESTATION_ONLY",
    "DEFAULT_ATTESTATION_VALIDITY",
    "MAX_ITEMS",
    "Attestation",
    "Category",
    "ChecklistItem",
    "ItemResult",
    "ReadinessChecklist",
    "ReadinessDecision",
    "ReadinessError",
    "Verdict",
    "evaluate_readiness",
    "minimal_capital_checklist",
    "render_readiness_report",
]
