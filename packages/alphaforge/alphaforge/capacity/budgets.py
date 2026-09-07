"""Conserved per-session capacity budgets (SF-S4-MR5).

A capacity limit that is only checked is not a budget. If an order reserves
participation, partially fills, and is then cancelled, the unfilled remainder
must return to the pool — otherwise the book quietly loses capacity it never
used. If a replay re-applies the same fill, the budget must not be charged
twice. Both failures are invisible in aggregate statistics and both change what
a strategy appears able to trade.

This module therefore models capacity as a ledger with an exact conservation
identity, checked after every mutation:

    reserved_outstanding + consumed + released + rejected == requested

Reservations are idempotent by ``reservation_id``. Replaying a journal
re-presents the same identifiers, and a second reservation under an identifier
already seen is a no-op returning the original decision rather than a second
charge against the budget. That is what makes the ledger safe under the
deterministic replay the event engine performs.

Every quantity is a **share count**, and every notional a **currency amount**.
The two are tracked separately because participation is a share constraint while
the book-level budget is a notional one, and conflating them is how a
high-priced symbol quietly consumes a low-priced symbol's allowance.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import date
from typing import Any, Final

from alphaforge.capacity.contracts import (
    MAX_RECORDS,
    CapacityContractError,
    CapacityPolicyViolation,
    finite_quantity,
    identifier,
)

#: Absolute tolerance for the conservation identity. Tighter than any share or
#: cent a simulation can represent, loose enough to survive float accumulation
#: across a session's reservations.
CONSERVATION_TOLERANCE: Final = 1e-9


@dataclass(frozen=True, slots=True)
class Reservation:
    """One idempotent claim against a symbol's session capacity."""

    reservation_id: str
    symbol: str
    session: date
    quantity: float
    notional: float

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "reservation_id", identifier(self.reservation_id, name="reservation_id")
        )
        object.__setattr__(self, "symbol", identifier(self.symbol, name="symbol"))
        object.__setattr__(self, "quantity", finite_quantity(self.quantity, name="quantity"))
        object.__setattr__(self, "notional", finite_quantity(self.notional, name="notional"))


@dataclass(slots=True)
class _SymbolLedger:
    """Mutable per-symbol accounting for one session."""

    requested: float = 0.0
    reserved: float = 0.0
    consumed: float = 0.0
    released: float = 0.0
    rejected: float = 0.0


@dataclass
class SessionCapacityLedger:
    """Conserved capacity accounting for one logical session.

    Args:
        session: The logical session this ledger governs.
        participation_limits: Per-symbol share ceilings, already resolved from
            lagged ADV by the policy. A symbol absent from this mapping has
            **zero** capacity, never unlimited.
        book_notional_budget: Book-level gross traded-notional ceiling.

    Raises:
        CapacityContractError: On malformed limits.
    """

    session: date
    participation_limits: dict[str, float]
    book_notional_budget: float
    _symbols: dict[str, _SymbolLedger] = field(default_factory=dict, init=False, repr=False)
    _reservations: dict[str, Reservation] = field(default_factory=dict, init=False, repr=False)
    _settled: set[str] = field(default_factory=set, init=False, repr=False)
    _book_reserved: float = field(default=0.0, init=False, repr=False)
    _book_consumed: float = field(default=0.0, init=False, repr=False)

    def __post_init__(self) -> None:
        if len(self.participation_limits) > MAX_RECORDS:
            raise CapacityContractError("participation limits exceed the record ceiling")
        cleaned: dict[str, float] = {}
        for symbol, limit in self.participation_limits.items():
            cleaned[identifier(symbol, name="symbol")] = finite_quantity(
                limit, name=f"participation_limit[{symbol}]"
            )
        self.participation_limits = cleaned
        self.book_notional_budget = finite_quantity(
            self.book_notional_budget, name="book_notional_budget"
        )

    # ----- queries ---------------------------------------------------------
    def remaining_participation(self, symbol: str) -> float:
        """Shares still available for ``symbol`` this session.

        An unknown symbol returns ``0.0``: absence of a limit is absence of
        permission, which is the single most important default in this module.
        """
        limit = self.participation_limits.get(symbol)
        if limit is None:
            return 0.0
        ledger = self._symbols.get(symbol)
        used = 0.0 if ledger is None else ledger.reserved + ledger.consumed
        return max(limit - used, 0.0)

    def remaining_book_notional(self) -> float:
        """Currency still available against the book-level session budget."""
        return max(self.book_notional_budget - self._book_reserved - self._book_consumed, 0.0)

    # ----- mutations -------------------------------------------------------
    def reserve(
        self, reservation_id: str, symbol: str, quantity: float, notional: float
    ) -> Reservation:
        """Reserve capacity, returning the granted (possibly reduced) claim.

        Idempotent: presenting an identifier already reserved returns the
        original claim unchanged rather than charging the budget again. This is
        what makes journal replay safe.

        The granted quantity is the largest that fits **both** the symbol's
        remaining participation and the book's remaining notional; the shortfall
        is recorded as rejected so the conservation identity still closes.
        """
        reservation_id = identifier(reservation_id, name="reservation_id")
        symbol = identifier(symbol, name="symbol")
        quantity = finite_quantity(quantity, name="quantity")
        notional = finite_quantity(notional, name="notional")
        if existing := self._reservations.get(reservation_id):
            if existing.symbol != symbol:
                raise CapacityPolicyViolation(
                    f"reservation {reservation_id!r} already claimed for {existing.symbol!r}"
                )
            return existing

        ledger = self._symbols.setdefault(symbol, _SymbolLedger())
        ledger.requested += quantity

        allowed = min(quantity, self.remaining_participation(symbol))
        if quantity > 0.0 and notional > 0.0:
            # Scale the notional with the granted share count so the two stay
            # consistent, then clip again against the book budget.
            unit_notional = notional / quantity
            book_allowed = self.remaining_book_notional() / unit_notional if unit_notional else 0.0
            allowed = min(allowed, book_allowed)
        allowed = max(allowed, 0.0)
        granted_notional = (notional / quantity * allowed) if quantity > 0.0 else 0.0

        ledger.reserved += allowed
        ledger.rejected += quantity - allowed
        self._book_reserved += granted_notional

        reservation = Reservation(
            reservation_id=reservation_id,
            symbol=symbol,
            session=self.session,
            quantity=allowed,
            notional=granted_notional,
        )
        self._reservations[reservation_id] = reservation
        self._assert_conserved(symbol)
        return reservation

    def consume(self, reservation_id: str, quantity: float) -> float:
        """Convert reserved capacity into consumed capacity on a fill.

        Returns the quantity actually consumed, which never exceeds what was
        reserved. A fill larger than its reservation is a bug in the caller, and
        clipping it here — rather than raising — would let the engine overtrade
        its budget silently, so it raises.
        """
        reservation = self._require_reservation(reservation_id)
        quantity = finite_quantity(quantity, name="quantity")
        if quantity > reservation.quantity + CONSERVATION_TOLERANCE:
            raise CapacityPolicyViolation(
                f"fill of {quantity} exceeds reservation {reservation_id!r} "
                f"of {reservation.quantity}"
            )
        ledger = self._symbols[reservation.symbol]
        applied = min(quantity, reservation.quantity)
        ledger.reserved -= applied
        ledger.consumed += applied
        unit = reservation.notional / reservation.quantity if reservation.quantity > 0.0 else 0.0
        self._book_reserved -= unit * applied
        self._book_consumed += unit * applied
        # Shrink the outstanding claim so a later release returns only the part
        # that was genuinely unused.
        self._reservations[reservation_id] = Reservation(
            reservation_id=reservation.reservation_id,
            symbol=reservation.symbol,
            session=reservation.session,
            quantity=reservation.quantity - applied,
            notional=reservation.notional - unit * applied,
        )
        self._assert_conserved(reservation.symbol)
        return applied

    def release(self, reservation_id: str) -> float:
        """Return the unconsumed remainder of a reservation to the pool.

        Idempotent by reservation: a second release is a no-op returning ``0.0``.
        Without that, a cancellation replayed from the journal would credit the
        budget twice and manufacture capacity out of nothing.
        """
        reservation = self._require_reservation(reservation_id)
        if reservation_id in self._settled:
            return 0.0
        ledger = self._symbols[reservation.symbol]
        returned = reservation.quantity
        ledger.reserved -= returned
        ledger.released += returned
        self._book_reserved -= reservation.notional
        self._settled.add(reservation_id)
        self._reservations[reservation_id] = Reservation(
            reservation_id=reservation.reservation_id,
            symbol=reservation.symbol,
            session=reservation.session,
            quantity=0.0,
            notional=0.0,
        )
        self._assert_conserved(reservation.symbol)
        return returned

    def _require_reservation(self, reservation_id: str) -> Reservation:
        reservation = self._reservations.get(identifier(reservation_id, name="reservation_id"))
        if reservation is None:
            raise CapacityPolicyViolation(f"unknown reservation {reservation_id!r}")
        return reservation

    def _assert_conserved(self, symbol: str) -> None:
        """Check the ledger identity after every mutation.

        Deliberately an invariant check rather than a test-only assertion: a
        capacity leak that only manifests after thousands of events would
        otherwise surface as a slightly wrong backtest rather than an error.
        """
        ledger = self._symbols[symbol]
        total = ledger.reserved + ledger.consumed + ledger.released + ledger.rejected
        if not math.isclose(total, ledger.requested, abs_tol=CONSERVATION_TOLERANCE):
            raise CapacityPolicyViolation(
                f"capacity conservation violated for {symbol!r} in {self.session}: "
                f"reserved {ledger.reserved} + consumed {ledger.consumed} + released "
                f"{ledger.released} + rejected {ledger.rejected} != requested {ledger.requested}"
            )
        if self._book_consumed > self.book_notional_budget + CONSERVATION_TOLERANCE:
            raise CapacityPolicyViolation(
                f"book notional budget exceeded in {self.session}: consumed "
                f"{self._book_consumed} against {self.book_notional_budget}"
            )

    # ----- evidence --------------------------------------------------------
    def reconciliation(self) -> dict[str, Any]:
        """Return the per-symbol and book-level accounting for the evidence record."""
        symbols = {
            symbol: {
                "requested": ledger.requested,
                "reserved_outstanding": ledger.reserved,
                "consumed": ledger.consumed,
                "released": ledger.released,
                "rejected": ledger.rejected,
                "limit": self.participation_limits.get(symbol, 0.0),
                "reconciles": math.isclose(
                    ledger.reserved + ledger.consumed + ledger.released + ledger.rejected,
                    ledger.requested,
                    abs_tol=CONSERVATION_TOLERANCE,
                ),
            }
            for symbol, ledger in sorted(self._symbols.items())
        }
        return {
            "session": self.session.isoformat(),
            "symbols": symbols,
            "book_notional_budget": self.book_notional_budget,
            "book_notional_reserved": self._book_reserved,
            "book_notional_consumed": self._book_consumed,
            "book_notional_remaining": self.remaining_book_notional(),
            "all_symbols_reconcile": all(entry["reconciles"] for entry in symbols.values()),
        }
