"""Lane identity and the two governance state machines.

SF-S5-SL-MR5. A *lane* is the unit a champion is assigned to. It is keyed by
purpose, target, horizon, frequency, universe contract, decision policy, and
environment, and that key is content-derived rather than assigned.

**Model state is never a mutable global.** One revision may be champion in one
lane and a rejected challenger in another, so "is this model the champion?" is
only answerable relative to a lane. Storing a `champion` flag on the revision
would make the question look answerable when it is not.

Two state machines live here, and both are declared as explicit transition
tables rather than as scattered ``if`` statements:

* :class:`LaneState` — where a lane is: unassigned, active with a champion, or
  frozen. Freezing is automatic; **unfreezing is not**.
* :class:`RequestState` — the lifecycle of a change request. Every terminal
  state is terminal: a rejected request is never reopened, and an applied one is
  never re-applied. Rollback is a *new* request, never a reversal of history.

A transition that is not in the table is refused with a typed error naming both
states, because the alternative -- a silent no-op or a permissive default -- is
how a lane ends up in a state nobody approved.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import StrEnum
from typing import Any, Final

from quant_platform.shadow.contracts import (
    ShadowValidationError,
    canonical_digest,
)

#: Refusal thresholds, not tuning knobs.
MAX_LANE_FIELD_CHARS: Final = 64
MAX_GENERATION: Final = 1_000_000

_LANE_FIELD_PATTERN: Final = re.compile(r"^[a-z][a-z0-9_\-]{0,63}$")


class GovernanceStateError(ShadowValidationError):
    """Raised when a governance transition is not permitted.

    A distinct type so a caller cannot conflate "this transition is illegal"
    with "this record is malformed": the first means the lane is fine and the
    request is wrong, the second means the input never should have been built.
    """


class LaneState(StrEnum):
    """Where a governance lane is."""

    UNASSIGNED = "unassigned"
    ACTIVE = "active"
    FROZEN = "frozen"


class RequestState(StrEnum):
    """The lifecycle of a change request."""

    AWAITING_APPROVAL = "awaiting_approval"
    APPROVED = "approved"
    APPLIED = "applied"
    REJECTED = "rejected"
    WITHDRAWN = "withdrawn"
    EXPIRED = "expired"
    STALE = "stale"
    REVOKED = "revoked"


class FreezeTrigger(StrEnum):
    """Why a lane froze. Recorded because it determines what clears the freeze."""

    HARD_INTEGRITY = "hard_integrity"
    CONSECUTIVE_SOFT_BREACH = "consecutive_soft_breach"


#: Permitted lane transitions. Absent pairs are refused.
#:
#: ``ACTIVE -> ACTIVE`` is present because promoting a new champion keeps the
#: lane active while changing which revision holds it; the compare-and-swap on
#: generation is what makes that safe, not the state machine.
_LANE_TRANSITIONS: Final[frozenset[tuple[LaneState, LaneState]]] = frozenset(
    {
        (LaneState.UNASSIGNED, LaneState.ACTIVE),
        (LaneState.ACTIVE, LaneState.ACTIVE),
        (LaneState.ACTIVE, LaneState.FROZEN),
        (LaneState.FROZEN, LaneState.ACTIVE),
    }
)

#: Terminal request states. Reaching one ends the request permanently.
TERMINAL_REQUEST_STATES: Final[frozenset[RequestState]] = frozenset(
    {
        RequestState.APPLIED,
        RequestState.REJECTED,
        RequestState.WITHDRAWN,
        RequestState.EXPIRED,
        RequestState.STALE,
        RequestState.REVOKED,
    }
)

#: Permitted request transitions, exactly as the governance policy declares.
_REQUEST_TRANSITIONS: Final[frozenset[tuple[RequestState, RequestState]]] = frozenset(
    {
        (RequestState.AWAITING_APPROVAL, RequestState.APPROVED),
        (RequestState.AWAITING_APPROVAL, RequestState.REJECTED),
        (RequestState.AWAITING_APPROVAL, RequestState.WITHDRAWN),
        (RequestState.AWAITING_APPROVAL, RequestState.EXPIRED),
        (RequestState.AWAITING_APPROVAL, RequestState.STALE),
        (RequestState.APPROVED, RequestState.APPLIED),
        (RequestState.APPROVED, RequestState.REVOKED),
        (RequestState.APPROVED, RequestState.EXPIRED),
        (RequestState.APPROVED, RequestState.STALE),
    }
)


def _lane_field(value: object, *, field_name: str) -> str:
    """Return a validated lane key component.

    Raises:
        ShadowValidationError: On a malformed or oversized component.
    """
    if not isinstance(value, str) or not _LANE_FIELD_PATTERN.match(value):
        raise ShadowValidationError(
            f"lane {field_name} {value!r} is malformed; expected lowercase alphanumeric "
            f"with _ or -, at most {MAX_LANE_FIELD_CHARS} characters"
        )
    return value


@dataclass(frozen=True, slots=True)
class GovernanceLane:
    """The identity a champion is assigned to.

    The identity is derived from the key rather than assigned, so two lanes
    that differ in any component are different lanes and cannot silently share
    a champion. Comparing a model trained for a 5-day horizon against one
    trained for 20 is a different question, not a closer contest.

    Raises:
        ShadowValidationError: On a malformed component or horizon.
    """

    purpose: str
    target: str
    horizon_days: int
    frequency: str
    universe: str
    decision_policy: str
    environment: str

    def __post_init__(self) -> None:
        for field_name in (
            "purpose",
            "target",
            "frequency",
            "universe",
            "decision_policy",
            "environment",
        ):
            _lane_field(getattr(self, field_name), field_name=field_name)
        if isinstance(self.horizon_days, bool) or not isinstance(self.horizon_days, int):
            raise ShadowValidationError("horizon_days must be an int")
        if not 1 <= self.horizon_days <= 365:
            raise ShadowValidationError("horizon_days must lie in [1, 365]")

    @property
    def identity(self) -> str:
        """Content identity over the complete lane key."""
        return canonical_digest(self.to_dict())

    def to_dict(self) -> dict[str, Any]:
        """Return the JSON-friendly lane key."""
        return {
            "purpose": self.purpose,
            "target": self.target,
            "horizon_days": self.horizon_days,
            "frequency": self.frequency,
            "universe": self.universe,
            "decision_policy": self.decision_policy,
            "environment": self.environment,
        }


@dataclass(frozen=True, slots=True)
class LaneHead:
    """The current projection of a lane, rebuildable from its event chain.

    ``generation`` is the compare-and-swap token. It increments on every
    assignment, so an approval that names generation 4 cannot be applied once
    the lane has reached generation 5 -- somebody else won the race.

    Raises:
        ShadowValidationError: On an internally inconsistent head.
    """

    lane_identity: str
    state: LaneState
    champion_revision: str | None
    generation: int
    freeze_trigger: FreezeTrigger | None

    def __post_init__(self) -> None:
        if not isinstance(self.lane_identity, str) or len(self.lane_identity) != 64:
            raise ShadowValidationError("lane_identity must be a full SHA-256 digest")
        if isinstance(self.generation, bool) or not isinstance(self.generation, int):
            raise ShadowValidationError("generation must be an int")
        if not 0 <= self.generation <= MAX_GENERATION:
            raise ShadowValidationError(f"generation must lie in [0, {MAX_GENERATION}]")
        if self.state is LaneState.UNASSIGNED and self.champion_revision is not None:
            raise ShadowValidationError("an unassigned lane cannot name a champion")
        if self.state is LaneState.ACTIVE and not self.champion_revision:
            raise ShadowValidationError(
                "an active lane must name its champion; an active lane without one is a "
                "lane whose current model nobody can identify"
            )
        if (self.state is LaneState.FROZEN) != (self.freeze_trigger is not None):
            raise ShadowValidationError(
                "a frozen lane must record its trigger and only a frozen lane may carry "
                "one; the trigger determines what is required to clear the freeze"
            )

    def to_dict(self) -> dict[str, Any]:
        """Return the JSON-friendly projection."""
        return {
            "lane_identity": self.lane_identity,
            "state": self.state.value,
            "champion_revision": self.champion_revision,
            "generation": self.generation,
            "freeze_trigger": self.freeze_trigger.value if self.freeze_trigger else None,
        }


def assert_lane_transition(current: LaneState, target: LaneState) -> None:
    """Refuse a lane transition that the policy does not permit.

    Raises:
        GovernanceStateError: Naming both states, so the log says what was
            attempted rather than only that something was refused.
    """
    if not isinstance(current, LaneState) or not isinstance(target, LaneState):
        raise GovernanceStateError("lane transitions require LaneState values")
    if (current, target) not in _LANE_TRANSITIONS:
        raise GovernanceStateError(
            f"lane transition {current.value} -> {target.value} is not permitted; "
            "there is no automatic activation, failover, or unfreeze"
        )


def assert_request_transition(current: RequestState, target: RequestState) -> None:
    """Refuse a request transition that the policy does not permit.

    Raises:
        GovernanceStateError: On a terminal-state mutation or an unlisted pair.
    """
    if not isinstance(current, RequestState) or not isinstance(target, RequestState):
        raise GovernanceStateError("request transitions require RequestState values")
    if current in TERMINAL_REQUEST_STATES:
        raise GovernanceStateError(
            f"request is already {current.value}, which is terminal; a rollback is a new "
            "request, never a reversal of a recorded one"
        )
    if (current, target) not in _REQUEST_TRANSITIONS:
        raise GovernanceStateError(
            f"request transition {current.value} -> {target.value} is not permitted"
        )


def lane_transitions() -> frozenset[tuple[LaneState, LaneState]]:
    """Return the permitted lane transitions.

    Exposed so tests can assert the table exhaustively rather than sampling it:
    a state machine tested only on the paths someone remembered is a state
    machine with untested paths.
    """
    return _LANE_TRANSITIONS


def request_transitions() -> frozenset[tuple[RequestState, RequestState]]:
    """Return the permitted request transitions."""
    return _REQUEST_TRANSITIONS


__all__ = [
    "MAX_GENERATION",
    "MAX_LANE_FIELD_CHARS",
    "TERMINAL_REQUEST_STATES",
    "FreezeTrigger",
    "GovernanceLane",
    "GovernanceStateError",
    "LaneHead",
    "LaneState",
    "RequestState",
    "assert_lane_transition",
    "assert_request_transition",
    "lane_transitions",
    "request_transitions",
]
