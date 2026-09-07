"""Causal borrow, locate, and liquidity authorization (SF-S4-MR5).

One policy, applied at two boundaries: before a target or order is created, and
again before a fill is admitted. Applying it once is not enough — a decision made
against yesterday's borrow book can become unauthorized by the time it executes,
and an engine that only checks at decision time will happily fill a short whose
locate expired overnight.

The resolution rules are deliberately asymmetric, because the risks are:

* **Opening a short requires positive evidence.** Active availability, an
  unexpired locate with remaining quantity, fresh liquidity, and remaining
  participation and book budget. Any one missing yields zero.
* **Covering requires none of it.** A cover reduces risk, and blocking it because
  new borrow is unavailable would trap the book in exactly the position the
  restriction was warning about. Covers remain subject to liquidity,
  participation, cost, and accounting limits — but never to borrow availability.

Causality is enforced by construction: :meth:`CapacityPolicy.resolve` filters
every record to those whose ``as_of_session`` is **strictly earlier** than the
session being decided. A record published on the decision session cannot
influence it, so a future observation cannot change an earlier decision. That is
asserted by mutation test rather than assumed.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, timedelta
from typing import Any, Literal

from alphaforge.capacity.budgets import SessionCapacityLedger
from alphaforge.capacity.contracts import (
    SHORTABLE_STATUSES,
    BorrowAvailability,
    CapacityContractError,
    CapacityPolicyDeclaration,
    LiquidityObservation,
    LocateRecord,
    book_digest,
    finite_quantity,
    identifier,
    session_date,
    validate_unique_records,
)

#: Why a request was reduced or refused. A closed vocabulary rather than free
#: text so the evidence can be aggregated and compared across runs.
DenialReason = Literal[
    "authorized",
    "new_shorts_disabled",
    "no_borrow_record",
    "borrow_stale",
    "borrow_not_shortable",
    "borrow_exhausted",
    "no_locate",
    "locate_expired",
    "locate_exhausted",
    "no_liquidity_record",
    "liquidity_stale",
    "participation_exhausted",
    "book_budget_exhausted",
]


@dataclass(frozen=True, slots=True)
class CapacityDecision:
    """The authorized quantity for one request, with its exact reason.

    ``requested`` and ``authorized`` are share counts. ``authorized`` is never
    greater than ``requested`` and never negative; a fully refused request
    returns ``0.0`` with the binding reason, never an exception, because refusal
    is a normal outcome the caller must record rather than an error.
    """

    symbol: str
    session: date
    side: Literal["open_short", "cover_short", "long"]
    requested: float
    authorized: float
    reason: DenialReason
    borrow_digest: str | None = None
    locate_id: str | None = None
    detail: str = ""

    def __post_init__(self) -> None:
        if self.authorized > self.requested + 1e-9:
            raise CapacityContractError("authorized quantity cannot exceed the request")
        if self.authorized < 0.0:
            raise CapacityContractError("authorized quantity cannot be negative")

    @property
    def fully_authorized(self) -> bool:
        """Whether the entire request was granted."""
        return abs(self.authorized - self.requested) <= 1e-9

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-friendly evidence row."""
        return {
            "symbol": self.symbol,
            "session": self.session.isoformat(),
            "side": self.side,
            "requested": self.requested,
            "authorized": self.authorized,
            "shortfall": self.requested - self.authorized,
            "reason": self.reason,
            "borrow_digest": self.borrow_digest,
            "locate_id": self.locate_id,
            "detail": self.detail,
        }


@dataclass(frozen=True, slots=True)
class ResolvedCapacity:
    """The point-in-time view a single session may legally consult."""

    session: date
    borrow: dict[str, BorrowAvailability]
    locates: dict[str, tuple[LocateRecord, ...]]
    liquidity: dict[str, LiquidityObservation]
    policy: CapacityPolicyDeclaration
    book_identity: str
    #: Symbols whose only candidate rows were dropped for exceeding the declared
    #: freshness horizon. Both stale and absent yield zero capacity, but the
    #: evidence must say which, because "we had no data" and "our data went
    #: stale" call for different operational fixes.
    stale_borrow: frozenset[str] = frozenset()
    stale_liquidity: frozenset[str] = frozenset()
    stale_locates: frozenset[str] = frozenset()


class CapacityPolicy:
    """Resolves and enforces one frozen borrow/liquidity/capacity policy.

    Args:
        declaration: The frozen policy. Frozen before evaluation by contract —
            a participation cap chosen after seeing fills is a fitted parameter.
        borrow: Point-in-time borrow availability rows.
        locates: Simulated locate grants.
        liquidity: Lagged ADV observations.

    Raises:
        CapacityContractError: On duplicate or conflicting records, or any
            malformed row.
    """

    def __init__(
        self,
        declaration: CapacityPolicyDeclaration,
        *,
        borrow: tuple[BorrowAvailability, ...] = (),
        locates: tuple[LocateRecord, ...] = (),
        liquidity: tuple[LiquidityObservation, ...] = (),
    ) -> None:
        if not isinstance(declaration, CapacityPolicyDeclaration):
            raise CapacityContractError("declaration must be a CapacityPolicyDeclaration")
        self.declaration = declaration
        self.borrow = validate_unique_records(borrow, name="borrow")
        self.locates = validate_unique_records(locates, name="locates")
        self.liquidity = validate_unique_records(liquidity, name="liquidity")
        self.book_identity = book_digest(self.borrow, self.locates, self.liquidity, declaration)
        # Locate consumption is tracked across the run, not per session: a locate
        # authorizes a total quantity, and spending it twice in two sessions
        # would double-count the same grant.
        self._locate_used: dict[str, float] = {}

    # ----- resolution ------------------------------------------------------
    def resolve(self, session: date) -> ResolvedCapacity:
        """Return the records legally visible when deciding ``session``.

        Only rows observed **strictly before** ``session`` are admitted. A row
        stamped with the decision session itself is excluded, because in daily-bar
        evidence there is no way to establish it was published before the decision
        was taken.
        """
        session = session_date(session, name="session")
        horizon = self.declaration.max_staleness_sessions

        borrow: dict[str, BorrowAvailability] = {}
        stale_borrow: set[str] = set()
        for record in self.borrow:
            if record.as_of_session >= session or not record.covers(session):
                continue
            if _stale(record.as_of_session, session, horizon):
                stale_borrow.add(record.symbol)
                continue
            incumbent = borrow.get(record.symbol)
            # Most recently observed row wins; ties are impossible because
            # duplicate (symbol, effective_session) pairs were already refused.
            if incumbent is None or record.as_of_session > incumbent.as_of_session:
                borrow[record.symbol] = record

        liquidity: dict[str, LiquidityObservation] = {}
        stale_liquidity: set[str] = set()
        for observation in self.liquidity:
            if observation.as_of_session >= session:
                continue
            if _stale(observation.as_of_session, session, horizon):
                stale_liquidity.add(observation.symbol)
                continue
            seen_liquidity = liquidity.get(observation.symbol)
            if seen_liquidity is None or observation.as_of_session > seen_liquidity.as_of_session:
                liquidity[observation.symbol] = observation

        locates: dict[str, list[LocateRecord]] = {}
        expired_locates: set[str] = set()
        for locate in self.locates:
            if locate.granted_session >= session:
                continue
            if not locate.active(session):
                expired_locates.add(locate.symbol)
                continue
            locates.setdefault(locate.symbol, []).append(locate)
        ordered_locates = {
            symbol: tuple(sorted(items, key=lambda item: (item.expiry_session, item.locate_id)))
            for symbol, items in locates.items()
        }

        return ResolvedCapacity(
            session=session,
            borrow=borrow,
            locates=ordered_locates,
            liquidity=liquidity,
            policy=self.declaration,
            book_identity=self.book_identity,
            stale_borrow=frozenset(stale_borrow - borrow.keys()),
            stale_liquidity=frozenset(stale_liquidity - liquidity.keys()),
            stale_locates=frozenset(expired_locates - ordered_locates.keys()),
        )

    # ----- limits ----------------------------------------------------------
    def participation_limits(self, resolved: ResolvedCapacity) -> dict[str, float]:
        """Return per-symbol session share ceilings from lagged ADV.

        A symbol with no fresh liquidity observation is **absent** from the
        mapping rather than present with a default, so the ledger's
        "unknown symbol means zero" rule applies to it.
        """
        return {
            symbol: observation.adv_shares * self.declaration.max_participation
            for symbol, observation in resolved.liquidity.items()
        }

    def open_session_ledger(self, resolved: ResolvedCapacity) -> SessionCapacityLedger:
        """Return a conserved ledger for ``resolved``'s session."""
        return SessionCapacityLedger(
            session=resolved.session,
            participation_limits=self.participation_limits(resolved),
            book_notional_budget=self.declaration.max_session_notional,
        )

    # ----- authorization ---------------------------------------------------
    def authorize(
        self,
        resolved: ResolvedCapacity,
        ledger: SessionCapacityLedger,
        *,
        symbol: str,
        side: Literal["open_short", "cover_short", "long"],
        quantity: float,
        price: float,
        reservation_id: str,
    ) -> CapacityDecision:
        """Authorize one request against borrow, locate, liquidity, and budget.

        The same call is used at decision time and again at fill time; passing
        the fill session's ``resolved`` view is what catches a locate that
        expired between the two.

        Covers are exempt from borrow and locate checks by design — see the
        module docstring — but still consume participation and book budget,
        because the market impact of buying back is real regardless of why.
        """
        symbol = identifier(symbol, name="symbol")
        quantity = finite_quantity(quantity, name="quantity")
        price = finite_quantity(price, name="price")
        if quantity <= 0.0:
            return CapacityDecision(
                symbol=symbol,
                session=resolved.session,
                side=side,
                requested=quantity,
                authorized=0.0,
                reason="authorized",
                detail="zero-quantity request",
            )

        allowed = quantity
        reason: DenialReason = "authorized"
        borrow_digest: str | None = None
        locate_id: str | None = None
        detail = ""

        if side == "open_short":
            if not self.declaration.allow_new_shorts:
                return _refuse(symbol, resolved, side, quantity, "new_shorts_disabled")
            record = resolved.borrow.get(symbol)
            if record is None:
                return _refuse(
                    symbol,
                    resolved,
                    side,
                    quantity,
                    "borrow_stale" if symbol in resolved.stale_borrow else "no_borrow_record",
                )
            borrow_digest = record.digest
            if record.status not in SHORTABLE_STATUSES:
                return _refuse(
                    symbol,
                    resolved,
                    side,
                    quantity,
                    "borrow_not_shortable",
                    detail=f"status={record.status}",
                    borrow_digest=borrow_digest,
                )
            if record.shortable_quantity <= 0.0:
                return _refuse(
                    symbol,
                    resolved,
                    side,
                    quantity,
                    "borrow_exhausted",
                    borrow_digest=borrow_digest,
                )
            allowed = min(allowed, record.shortable_quantity)

            grants = resolved.locates.get(symbol, ())
            if not grants:
                return _refuse(
                    symbol,
                    resolved,
                    side,
                    quantity,
                    "locate_expired" if symbol in resolved.stale_locates else "no_locate",
                    borrow_digest=borrow_digest,
                )
            remaining = 0.0
            chosen: LocateRecord | None = None
            for grant in grants:
                available = grant.quantity - self._locate_used.get(grant.locate_id, 0.0)
                if available > remaining:
                    remaining, chosen = available, grant
            if chosen is None or remaining <= 0.0:
                return _refuse(
                    symbol,
                    resolved,
                    side,
                    quantity,
                    "locate_exhausted",
                    borrow_digest=borrow_digest,
                )
            locate_id = chosen.locate_id
            allowed = min(allowed, remaining)
            if allowed < quantity:
                reason = (
                    "locate_exhausted"
                    if remaining < record.shortable_quantity
                    else "borrow_exhausted"
                )

        if symbol not in resolved.liquidity:
            return _refuse(
                symbol,
                resolved,
                side,
                quantity,
                "liquidity_stale" if symbol in resolved.stale_liquidity else "no_liquidity_record",
                borrow_digest=borrow_digest,
                locate_id=locate_id,
            )

        reservation = ledger.reserve(reservation_id, symbol, allowed, allowed * price)
        granted = reservation.quantity
        if granted <= 0.0:
            binding: DenialReason = (
                "book_budget_exhausted"
                if ledger.remaining_participation(symbol) > 0.0
                else "participation_exhausted"
            )
            return _refuse(
                symbol,
                resolved,
                side,
                quantity,
                binding,
                borrow_digest=borrow_digest,
                locate_id=locate_id,
            )
        if granted < allowed:
            reason = (
                "book_budget_exhausted"
                if ledger.remaining_participation(symbol) > 0.0
                else "participation_exhausted"
            )
        elif granted < quantity and reason == "authorized":
            reason = "borrow_exhausted"

        if side == "open_short" and locate_id is not None:
            self._locate_used[locate_id] = self._locate_used.get(locate_id, 0.0) + granted

        return CapacityDecision(
            symbol=symbol,
            session=resolved.session,
            side=side,
            requested=quantity,
            authorized=granted,
            reason=reason,
            borrow_digest=borrow_digest,
            locate_id=locate_id,
            detail=detail,
        )

    def locate_utilization(self) -> dict[str, float]:
        """Return consumed quantity per locate, for the evidence record."""
        return dict(sorted(self._locate_used.items()))


def _stale(observed: date, session: date, horizon: int) -> bool:
    """Whether ``observed`` is older than the declared freshness horizon."""
    return (session - observed) > timedelta(days=horizon)


def _refuse(
    symbol: str,
    resolved: ResolvedCapacity,
    side: str,
    quantity: float,
    reason: DenialReason,
    *,
    detail: str = "",
    borrow_digest: str | None = None,
    locate_id: str | None = None,
) -> CapacityDecision:
    """Return a fully-refused decision carrying its binding reason."""
    return CapacityDecision(
        symbol=symbol,
        session=resolved.session,
        side=side,  # type: ignore[arg-type]
        requested=quantity,
        authorized=0.0,
        reason=reason,
        borrow_digest=borrow_digest,
        locate_id=locate_id,
        detail=detail,
    )


@dataclass
class CapacityEvidence:
    """Additive, normalized evidence for one run's capacity behaviour."""

    decisions: list[CapacityDecision] = field(default_factory=list)
    reconciliations: list[dict[str, Any]] = field(default_factory=list)

    def record(self, decision: CapacityDecision) -> None:
        """Append one authorization decision."""
        self.decisions.append(decision)

    def close_session(self, ledger: SessionCapacityLedger) -> None:
        """Append one session's budget reconciliation."""
        self.reconciliations.append(ledger.reconciliation())

    def summary(self) -> dict[str, Any]:
        """Return aggregate counters for the run manifest."""
        refused = [item for item in self.decisions if item.authorized <= 0.0]
        constrained = [item for item in self.decisions if 0.0 < item.authorized < item.requested]
        by_reason: dict[str, int] = {}
        for item in self.decisions:
            by_reason[item.reason] = by_reason.get(item.reason, 0) + 1
        return {
            "decisions": len(self.decisions),
            "refused": len(refused),
            "constrained": len(constrained),
            "requested_quantity": sum(item.requested for item in self.decisions),
            "authorized_quantity": sum(item.authorized for item in self.decisions),
            "by_reason": dict(sorted(by_reason.items())),
            "sessions_reconciled": sum(
                1 for entry in self.reconciliations if entry["all_symbols_reconcile"]
            ),
            "sessions_total": len(self.reconciliations),
            "all_sessions_reconcile": all(
                entry["all_symbols_reconcile"] for entry in self.reconciliations
            ),
        }
