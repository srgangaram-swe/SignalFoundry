"""Versioned HTTP projections of governance lanes for the console.

SF-S5-SL-MR6. These contracts carry the evidence the console needs to render
lane state, gate outcomes, and promotion history, and nothing more.

Three deliberate omissions:

* **No authority-bearing field.** Nothing here names an approval token, an
  idempotency key, an expected generation for a compare-and-swap, or any other
  value that would let a browser participate in an assignment. A console that
  cannot express a promotion cannot accidentally perform one.
* **No raw event payload.** Each event is projected into a fixed shape by the
  read port. Forwarding a stored blob would make the wire contract depend on
  whatever a writer happened to record.
* **No host path, database identifier, or request identity.** Those describe the
  machine rather than the evidence.

``chain_verified`` is reported rather than filtered on. A lane whose event chain
fails verification is exactly the lane an operator needs to see, so the contract
carries the failure to the console instead of hiding the lane.
"""

from __future__ import annotations

from datetime import datetime
from typing import TYPE_CHECKING, Annotated, Final, Literal

from pydantic import Field

if TYPE_CHECKING:  # pragma: no cover - typing-only, keeps the container closure clean
    # Imported for annotations only. The hardened service image ships the
    # service package without the governance package, so a runtime import here
    # would drag governance -- including its writable store -- into an image
    # whose whole point is that it cannot write.
    from quant_platform.governance.read_ports import (
        ComparisonProjection,
        LaneDetail,
        LaneEventProjection,
        LanePage,
        LaneSummary,
    )

from quant_platform.service.contracts import (
    SCHEMA_VERSION,
    StrictServiceContract,
    require_bounded_text,
    require_utc_datetime,
)

#: Console pages stay well inside the middleware's response ceiling. A lane
#: summary is small, but a page of them plus their per-kind counts is not free.
LANE_RESPONSE_PAGE_LIMIT: Final = 50

MAX_SUMMARY_CHARS: Final = 512

Digest = Annotated[
    str,
    Field(min_length=64, max_length=64, pattern=r"^[0-9a-f]{64}$"),
]

OptionalDigest = Annotated[
    str | None,
    Field(default=None, min_length=64, max_length=64, pattern=r"^[0-9a-f]{64}$"),
]


class GateResultResponse(StrictServiceContract):
    """One absolute gate outcome.

    ``satisfied`` is the whole verdict. There is no score, weight, or partial
    credit field, because a gate that could be partially satisfied would not be
    absolute.
    """

    schema_version: Literal[1] = SCHEMA_VERSION
    name: Annotated[str, Field(min_length=1, max_length=64)]
    satisfied: bool
    detail: Annotated[str, Field(min_length=0, max_length=MAX_SUMMARY_CHARS)]

    @classmethod
    def from_projection(cls, projection: object) -> GateResultResponse:
        """Build the wire contract from a read-port gate projection."""
        name = require_bounded_text(getattr(projection, "name", ""), "gate name", maximum_bytes=64)
        detail = require_bounded_text(
            getattr(projection, "detail", ""),
            "gate detail",
            minimum_bytes=0,
            maximum_bytes=MAX_SUMMARY_CHARS,
        )
        return cls(
            name=name, satisfied=bool(getattr(projection, "satisfied", False)), detail=detail
        )


class HypothesisTestResponse(StrictServiceContract):
    """One dependence-aware test with its interval kept as an explicit pair.

    ``interval_low`` and ``interval_high`` are independently nullable. A
    one-sided test has one genuinely unbounded end, and reporting a number there
    would assert a bound the test never established.

    ``p_value_uncorrected`` is named for what it is. The familywise decision
    lives in the comparison's correction block, so a reader cannot mistake a raw
    p-value from a family for the corrected one.
    """

    schema_version: Literal[1] = SCHEMA_VERSION
    name: Annotated[str, Field(min_length=1, max_length=64)]
    metric: Annotated[str, Field(min_length=1, max_length=64)]
    verdict: Literal[
        "favours_challenger", "favours_champion", "inconclusive", "underpowered", "unknown"
    ]
    point_estimate: float | None = None
    interval_low: float | None = None
    interval_high: float | None = None
    p_value_uncorrected: Annotated[float | None, Field(default=None, ge=0.0, le=1.0)]
    blocks: Annotated[int, Field(ge=0, le=100_000)]
    observations: Annotated[int, Field(ge=0, le=10_000_000)]
    margin: float | None = None

    @classmethod
    def from_projection(cls, projection: object) -> HypothesisTestResponse:
        """Build the wire contract from a read-port test projection."""
        verdict = str(getattr(projection, "verdict", "unknown"))
        allowed = {
            "favours_challenger",
            "favours_champion",
            "inconclusive",
            "underpowered",
        }
        return cls(
            name=require_bounded_text(
                getattr(projection, "name", "test"), "test name", maximum_bytes=64
            ),
            metric=require_bounded_text(
                getattr(projection, "metric", "metric"), "test metric", maximum_bytes=64
            ),
            # An unrecognised verdict becomes the explicit "unknown" member
            # rather than a favourable default: the console renders it as
            # incompatible evidence instead of silently reading it as a pass.
            verdict=verdict if verdict in allowed else "unknown",  # type: ignore[arg-type]
            point_estimate=getattr(projection, "point_estimate", None),
            interval_low=getattr(projection, "interval_low", None),
            interval_high=getattr(projection, "interval_high", None),
            p_value_uncorrected=getattr(projection, "p_value_uncorrected", None),
            blocks=int(getattr(projection, "blocks", 0)),
            observations=int(getattr(projection, "observations", 0)),
            margin=getattr(projection, "margin", None),
        )


class ComparisonResponse(StrictServiceContract):
    """One recorded promotion decision with every gate and test behind it."""

    schema_version: Literal[1] = SCHEMA_VERSION
    sequence: Annotated[int, Field(ge=1, le=1_000_000)]
    recorded_at: datetime
    recommendation: Literal[
        "promote", "retain_champion", "insufficient_evidence", "invalid", "unknown"
    ]
    policy_identity: Digest
    cohort_identity: Digest
    decided_at: Annotated[str, Field(min_length=0, max_length=64)]
    gates: tuple[GateResultResponse, ...]
    tests: tuple[HypothesisTestResponse, ...]
    correction_method: Annotated[str | None, Field(default=None, max_length=64)]
    correction_alpha: Annotated[float | None, Field(default=None, gt=0.0, lt=0.5)]
    family_size: Annotated[int | None, Field(default=None, ge=1, le=1_000)]
    truncated_gates: bool = False
    truncated_tests: bool = False

    @classmethod
    def from_projection(cls, projection: ComparisonProjection) -> ComparisonResponse:
        """Build the wire contract from a read-port comparison projection."""
        recommendation = projection.recommendation
        allowed = {"promote", "retain_champion", "insufficient_evidence", "invalid"}
        return cls(
            sequence=projection.sequence,
            recorded_at=require_utc_datetime(projection.recorded_at, "recorded_at"),
            recommendation=(
                recommendation if recommendation in allowed else "unknown"  # type: ignore[arg-type]
            ),
            policy_identity=projection.policy_identity,
            cohort_identity=projection.cohort_identity,
            decided_at=require_bounded_text(
                projection.decided_at, "decided_at", minimum_bytes=0, maximum_bytes=64
            ),
            gates=tuple(GateResultResponse.from_projection(item) for item in projection.gates),
            tests=tuple(HypothesisTestResponse.from_projection(item) for item in projection.tests),
            correction_method=projection.correction_method,
            correction_alpha=projection.correction_alpha,
            family_size=projection.family_size,
            truncated_gates=projection.truncated_gates,
            truncated_tests=projection.truncated_tests,
        )


class LaneEventResponse(StrictServiceContract):
    """One chain event reduced to its fixed console shape."""

    schema_version: Literal[1] = SCHEMA_VERSION
    sequence: Annotated[int, Field(ge=1, le=1_000_000)]
    kind: Literal[
        "policy", "comparison", "request", "approval", "assignment", "monitoring", "freeze"
    ]
    recorded_at: datetime
    chain_digest: Digest
    summary: Annotated[str, Field(min_length=1, max_length=MAX_SUMMARY_CHARS)]

    @classmethod
    def from_projection(cls, projection: LaneEventProjection) -> LaneEventResponse:
        """Build the wire contract from a read-port event projection."""
        return cls(
            sequence=projection.sequence,
            kind=projection.kind,  # type: ignore[arg-type]
            recorded_at=require_utc_datetime(projection.recorded_at, "recorded_at"),
            chain_digest=projection.chain_digest,
            summary=require_bounded_text(
                projection.summary, "summary", maximum_bytes=MAX_SUMMARY_CHARS
            ),
        )


class LaneSummaryResponse(StrictServiceContract):
    """A lane's identity, head state, and the health of the chain behind it."""

    schema_version: Literal[1] = SCHEMA_VERSION
    lane_identity: Digest
    purpose: Annotated[str, Field(min_length=1, max_length=64)]
    target: Annotated[str, Field(min_length=1, max_length=64)]
    horizon_days: Annotated[int, Field(ge=1, le=365)]
    frequency: Annotated[str, Field(min_length=1, max_length=64)]
    universe: Annotated[str, Field(min_length=1, max_length=64)]
    decision_policy: Annotated[str, Field(min_length=1, max_length=64)]
    environment: Annotated[str, Field(min_length=1, max_length=64)]
    state: Literal["unassigned", "active", "frozen"]
    champion_revision: OptionalDigest
    generation: Annotated[int, Field(ge=0, le=1_000_000)]
    freeze_trigger: Literal["hard_integrity", "consecutive_soft_breach"] | None = None
    created_at: datetime
    event_count: Annotated[int, Field(ge=0, le=1_000_000)]
    chain_verified: bool
    chain_fault: Annotated[str | None, Field(default=None, max_length=MAX_SUMMARY_CHARS)]
    events_by_kind: tuple[tuple[str, int], ...] = ()

    @classmethod
    def from_projection(cls, projection: LaneSummary) -> LaneSummaryResponse:
        """Build the wire contract from a read-port lane summary.

        ``events_by_kind`` becomes a sorted tuple of pairs rather than a mapping
        so the serialization is deterministic and the contract stays frozen.
        """
        return cls(
            lane_identity=projection.lane_identity,
            purpose=require_bounded_text(projection.purpose, "purpose", maximum_bytes=64),
            target=require_bounded_text(projection.target, "target", maximum_bytes=64),
            horizon_days=projection.horizon_days,
            frequency=require_bounded_text(projection.frequency, "frequency", maximum_bytes=64),
            universe=require_bounded_text(projection.universe, "universe", maximum_bytes=64),
            decision_policy=require_bounded_text(
                projection.decision_policy, "decision_policy", maximum_bytes=64
            ),
            environment=require_bounded_text(
                projection.environment, "environment", maximum_bytes=64
            ),
            state=projection.state.value,  # type: ignore[arg-type]
            champion_revision=projection.champion_revision,
            generation=projection.generation,
            freeze_trigger=(
                projection.freeze_trigger.value  # type: ignore[arg-type]
                if projection.freeze_trigger is not None
                else None
            ),
            created_at=require_utc_datetime(projection.created_at, "created_at"),
            event_count=projection.event_count,
            chain_verified=projection.chain_verified,
            chain_fault=projection.chain_fault,
            events_by_kind=tuple(sorted(projection.events_by_kind.items())),
        )


class LanePageResponse(StrictServiceContract):
    """A bounded page of lane summaries with a stable next cursor.

    There is no total count. A count over an append-only chain is a snapshot
    that is stale before it renders, and the console must not present one as
    fact.
    """

    schema_version: Literal[1] = SCHEMA_VERSION
    items: tuple[LaneSummaryResponse, ...]
    next_cursor: OptionalDigest

    @classmethod
    def from_projection(cls, page: LanePage) -> LanePageResponse:
        """Build the wire contract from a read-port lane page."""
        return cls(
            items=tuple(LaneSummaryResponse.from_projection(item) for item in page.items),
            next_cursor=page.next_cursor,
        )


class LaneDetailResponse(StrictServiceContract):
    """One lane with a bounded slice of its most recent history."""

    schema_version: Literal[1] = SCHEMA_VERSION
    lane: LaneSummaryResponse
    events: tuple[LaneEventResponse, ...]
    truncated: bool
    authority: Annotated[str, Field(min_length=1, max_length=MAX_SUMMARY_CHARS)] = (
        "This projection is read-only evidence. No console action can approve, apply, "
        "roll back, or unfreeze a lane; promotion requires a separate local human action."
    )

    @classmethod
    def from_projection(cls, detail: LaneDetail) -> LaneDetailResponse:
        """Build the wire contract from a read-port lane detail."""
        return cls(
            lane=LaneSummaryResponse.from_projection(detail.summary),
            events=tuple(LaneEventResponse.from_projection(item) for item in detail.events),
            truncated=detail.truncated,
        )


class ComparisonPageResponse(StrictServiceContract):
    """Every recorded decision for one lane, oldest first."""

    schema_version: Literal[1] = SCHEMA_VERSION
    items: tuple[ComparisonResponse, ...]
    truncated: bool = False

    @classmethod
    def from_projections(
        cls, projections: tuple[ComparisonProjection, ...], *, truncated: bool = False
    ) -> ComparisonPageResponse:
        """Build the wire contract from read-port comparison projections."""
        return cls(
            items=tuple(ComparisonResponse.from_projection(item) for item in projections),
            truncated=truncated,
        )


__all__ = [
    "LANE_RESPONSE_PAGE_LIMIT",
    "ComparisonPageResponse",
    "ComparisonResponse",
    "GateResultResponse",
    "HypothesisTestResponse",
    "LaneDetailResponse",
    "LaneEventResponse",
    "LanePageResponse",
    "LaneSummaryResponse",
]
