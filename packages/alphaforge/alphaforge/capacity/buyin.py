"""Forced buy-ins on recall or restriction (SF-S4-MR5).

When a held short becomes recalled or restricted, the position must be bought
back. This module schedules that buy-in at the next causally eligible session
and tracks it to resolution under a bounded policy.

Two rules carry the weight:

**A buy-in is an ordinary trade.** It is not a free unwind. It consumes
participation and book budget, pays the same frictions, and can partially fill —
which is precisely why it can fail to complete. Modelling it as an instant
costless exit would hide the risk that makes recalls dangerous.

**An unresolved residual halts publication.** If the bounded resolution window
expires with shares still outstanding, the run raises
:class:`ForcedBuyInHalt` and no artifact is published. It is never silently
carried into the next session, and never dropped. A backtest that quietly
absorbs an unresolvable recall is reporting a position it could not have held.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from typing import Any

from alphaforge.capacity.contracts import (
    CapacityContractError,
    finite_quantity,
    identifier,
    session_date,
)


class ForcedBuyInHalt(RuntimeError):
    """Raised when a forced buy-in cannot resolve within its declared window.

    A halt rather than a warning: the surviving position is unauthorized, so any
    downstream P&L, capacity, or risk figure computed from it would be describing
    a book that could not legally exist.
    """


@dataclass(frozen=True, slots=True)
class BuyInOrder:
    """One scheduled forced repurchase."""

    buy_in_id: str
    symbol: str
    triggered_session: date
    scheduled_session: date
    quantity: float
    trigger: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "buy_in_id", identifier(self.buy_in_id, name="buy_in_id"))
        object.__setattr__(self, "symbol", identifier(self.symbol, name="symbol"))
        object.__setattr__(
            self,
            "triggered_session",
            session_date(self.triggered_session, name="triggered_session"),
        )
        object.__setattr__(
            self,
            "scheduled_session",
            session_date(self.scheduled_session, name="scheduled_session"),
        )
        object.__setattr__(self, "quantity", finite_quantity(self.quantity, name="quantity"))
        if self.quantity <= 0.0:
            raise CapacityContractError("a buy-in must cover a positive quantity")
        if self.scheduled_session < self.triggered_session:
            raise CapacityContractError(
                "a buy-in cannot be scheduled before the session that triggered it"
            )
        if self.trigger not in ("recalled", "restricted", "unavailable"):
            raise CapacityContractError(f"unsupported buy-in trigger {self.trigger!r}")


@dataclass(slots=True)
class _BuyInState:
    """Mutable progress for one outstanding buy-in."""

    order: BuyInOrder
    outstanding: float
    filled: float = 0.0
    sessions_elapsed: int = 0
    fills: list[tuple[str, float]] = field(default_factory=list)


class ForcedBuyInBook:
    """Tracks scheduled buy-ins from trigger through resolution or halt.

    Args:
        resolution_sessions: Bounded number of sessions a buy-in may remain
            outstanding. ``0`` requires same-session resolution.
    """

    def __init__(self, *, resolution_sessions: int) -> None:
        if isinstance(resolution_sessions, bool) or not isinstance(resolution_sessions, int):
            raise CapacityContractError("resolution_sessions must be an int")
        if resolution_sessions < 0:
            raise CapacityContractError("resolution_sessions must be non-negative")
        self.resolution_sessions = resolution_sessions
        self._open: dict[str, _BuyInState] = {}
        self._resolved: list[_BuyInState] = []

    def schedule(
        self,
        *,
        buy_in_id: str,
        symbol: str,
        triggered_session: date,
        scheduled_session: date,
        quantity: float,
        trigger: str,
    ) -> BuyInOrder:
        """Schedule a buy-in, idempotently by ``buy_in_id``.

        Idempotent because journal replay re-presents the same trigger; a second
        schedule under the same identifier must not double the quantity that has
        to be repurchased.
        """
        buy_in_id = identifier(buy_in_id, name="buy_in_id")
        if existing := self._open.get(buy_in_id):
            return existing.order
        order = BuyInOrder(
            buy_in_id=buy_in_id,
            symbol=symbol,
            triggered_session=triggered_session,
            scheduled_session=scheduled_session,
            quantity=quantity,
            trigger=trigger,
        )
        self._open[buy_in_id] = _BuyInState(order=order, outstanding=order.quantity)
        return order

    def outstanding(self) -> tuple[BuyInOrder, ...]:
        """Return buy-ins still awaiting completion, in deterministic order."""
        return tuple(
            state.order for _, state in sorted(self._open.items(), key=lambda item: item[0])
        )

    def apply_fill(self, buy_in_id: str, quantity: float, fill_id: str) -> float:
        """Apply a (possibly partial) repurchase and return the amount applied."""
        state = self._require(buy_in_id)
        quantity = finite_quantity(quantity, name="quantity")
        applied = min(quantity, state.outstanding)
        state.outstanding -= applied
        state.filled += applied
        state.fills.append((identifier(fill_id, name="fill_id"), applied))
        if state.outstanding <= 1e-9:
            self._resolved.append(state)
            del self._open[state.order.buy_in_id]
        return applied

    def advance_session(self, session: date) -> None:
        """Age every outstanding buy-in by one session and halt on expiry.

        Raises:
            ForcedBuyInHalt: When a buy-in exceeds its declared resolution
                window with shares still outstanding.
        """
        session = session_date(session, name="session")
        for state in list(self._open.values()):
            if session <= state.order.scheduled_session:
                continue
            state.sessions_elapsed += 1
            if state.sessions_elapsed > self.resolution_sessions:
                raise ForcedBuyInHalt(
                    f"forced buy-in {state.order.buy_in_id!r} for {state.order.symbol!r} "
                    f"left {state.outstanding} shares outstanding after "
                    f"{self.resolution_sessions} session(s); publication is halted because "
                    "the surviving short is unauthorized"
                )

    def assert_resolved(self) -> None:
        """Refuse to finish a run with outstanding buy-ins.

        Raises:
            ForcedBuyInHalt: If any buy-in remains open.
        """
        if self._open:
            names = ", ".join(sorted(self._open))
            raise ForcedBuyInHalt(
                f"run ended with unresolved forced buy-ins: {names}; artifacts are not "
                "publishable while an unauthorized short remains open"
            )

    def _require(self, buy_in_id: str) -> _BuyInState:
        state = self._open.get(identifier(buy_in_id, name="buy_in_id"))
        if state is None:
            raise CapacityContractError(f"unknown or already-resolved buy-in {buy_in_id!r}")
        return state

    def evidence(self) -> dict[str, Any]:
        """Return the buy-in record for the run manifest."""
        resolved = [
            {
                "buy_in_id": state.order.buy_in_id,
                "symbol": state.order.symbol,
                "trigger": state.order.trigger,
                "triggered_session": state.order.triggered_session.isoformat(),
                "scheduled_session": state.order.scheduled_session.isoformat(),
                "quantity": state.order.quantity,
                "filled": state.filled,
                "fills": [{"fill_id": name, "quantity": amount} for name, amount in state.fills],
            }
            for state in sorted(self._resolved, key=lambda item: item.order.buy_in_id)
        ]
        return {
            "resolution_sessions": self.resolution_sessions,
            "resolved": resolved,
            "resolved_count": len(resolved),
            "outstanding_count": len(self._open),
            "outstanding": sorted(self._open),
        }


def detect_recalls(
    positions: dict[str, float],
    resolved_borrow: dict[str, Any],
    *,
    session: date,
    next_session: date,
) -> tuple[tuple[str, float, str], ...]:
    """Return open shorts whose borrow became recalled, restricted, or unknown.

    A symbol that simply *disappeared* from the borrow book counts as
    ``unavailable``: the absence of a record is not evidence that the borrow
    survived, and treating silence as continuation is the same permissive
    default this MR exists to remove.
    """
    triggers: list[tuple[str, float, str]] = []
    for symbol, quantity in sorted(positions.items()):
        if quantity >= 0.0:
            continue
        record = resolved_borrow.get(symbol)
        if record is None:
            triggers.append((symbol, abs(quantity), "unavailable"))
        elif record.status == "recalled":
            triggers.append((symbol, abs(quantity), "recalled"))
        elif record.status in ("restricted", "unknown"):
            triggers.append((symbol, abs(quantity), "restricted"))
    return tuple(triggers)
