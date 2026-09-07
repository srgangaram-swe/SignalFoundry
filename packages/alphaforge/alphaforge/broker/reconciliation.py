"""Compare persisted intent against broker-reported state, and halt on divergence.

SF-S5-MR4. The store in :mod:`alphaforge.broker.durable` records what the system
*believes*. The broker reports what it *did*. This module compares them and, on
any material difference, halts.

**It never repairs.** The issue's non-goal says so directly, and the reason is
mechanical: a repair acts on the state that is known to be wrong. If the local
book says 100 shares and the broker says 150, the difference is either an
unrecorded fill, a duplicate submission, a manual intervention, or a bug — and
those four have different correct responses. Automatically writing 150 into the
local book picks one interpretation and destroys the evidence needed to
distinguish it from the others.

**It never liquidates.** An automatic flatten on divergence is a market order
sized from an unverified position, which is the one action guaranteed to be wrong
when the position is what you are unsure about.

The output is a report: what diverged, by how much, and which category the
divergence falls into. A human decides what to do with it.

Recovery also has to survive fills that arrive **out of order**.
:func:`apply_fills_idempotently` folds a fill sequence into a position book
keyed by fill identity, so a replayed or reordered delivery converges to the same
book — the property that makes a dropped-and-redelivered acknowledgement safe.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal
from enum import StrEnum
from typing import Any, Final

from alphaforge.broker.contracts import (
    QUANTITY_EPSILON,
    AccountSnapshot,
    BrokerContractError,
    Fill,
    OrderSide,
    OrderState,
    OrderStatus,
    Position,
    utc_timestamp,
)
from alphaforge.broker.durable import OrderIntent, SessionSnapshot

#: Cash may legitimately differ by accrual and rounding. Anything above this is
#: a divergence. Deliberately tight: this is a tolerance for representation,
#: not for disagreement.
DEFAULT_CASH_TOLERANCE: Final = Decimal("0.01")

#: Bound on reported divergences. A report that cannot be read is not actionable.
MAX_REPORTED_DIVERGENCES: Final = 500


class ReconciliationError(BrokerContractError):
    """Raised when reconciliation cannot be performed at all."""


class DivergenceKind(StrEnum):
    """What sort of disagreement was found.

    The categories exist because they have different correct responses, and
    collapsing them into "mismatch" would discard the information a human needs.
    """

    POSITION_QUANTITY = "position_quantity"
    POSITION_ONLY_LOCAL = "position_only_local"
    POSITION_ONLY_BROKER = "position_only_broker"
    CASH = "cash"
    ORDER_ONLY_LOCAL = "order_only_local"
    ORDER_ONLY_BROKER = "order_only_broker"
    ORDER_STATE = "order_state"
    UNRECORDED_FILL = "unrecorded_fill"


@dataclass(frozen=True, slots=True)
class Divergence:
    """One disagreement between local intent and broker-reported state."""

    kind: DivergenceKind
    subject: str
    local_value: str
    broker_value: str
    interpretation: str

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-friendly record."""
        return {
            "kind": self.kind.value,
            "subject": self.subject,
            "local_value": self.local_value,
            "broker_value": self.broker_value,
            "interpretation": self.interpretation,
        }


@dataclass(frozen=True)
class ReconciliationReport:
    """The outcome of one comparison. Advisory to a human, never self-acting."""

    reconciled: bool
    checked_at: datetime
    divergences: tuple[Divergence, ...]
    positions_compared: int
    orders_compared: int

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "checked_at", utc_timestamp(self.checked_at, field_name="checked_at")
        )
        divergences = tuple(self.divergences)
        if len(divergences) > MAX_REPORTED_DIVERGENCES:
            raise ReconciliationError(
                f"divergence count exceeds the {MAX_REPORTED_DIVERGENCES} ceiling; the "
                "comparison is being run against the wrong account or the wrong snapshot"
            )
        object.__setattr__(self, "divergences", divergences)
        if self.reconciled and divergences:
            raise ReconciliationError(
                "a report cannot be marked reconciled while carrying divergences"
            )

    @property
    def must_halt(self) -> bool:
        """Whether trading must stop. True whenever anything diverged."""
        return not self.reconciled

    def kinds(self) -> tuple[DivergenceKind, ...]:
        """Return the distinct divergence kinds, in declaration order."""
        seen = [kind for kind in DivergenceKind if any(d.kind is kind for d in self.divergences)]
        return tuple(seen)

    def to_dict(self) -> dict[str, Any]:
        """Return the complete JSON-friendly report."""
        return {
            "reconciled": self.reconciled,
            "must_halt": self.must_halt,
            "checked_at": self.checked_at.isoformat(),
            "positions_compared": self.positions_compared,
            "orders_compared": self.orders_compared,
            "divergence_count": len(self.divergences),
            "divergence_kinds": [kind.value for kind in self.kinds()],
            "divergences": [item.to_dict() for item in self.divergences],
            "action": (
                "No divergence. Trading may proceed."
                if self.reconciled
                else "HALT. Stop submitting, preserve state, require operator review. "
                "Do not retry and do not liquidate: an automatic action here would be "
                "sized from exactly the state that is known to be wrong."
            ),
            "repair_policy": (
                "This module never repairs local state from broker state. A quantity "
                "difference may be an unrecorded fill, a duplicate submission, a manual "
                "intervention, or a bug, and those have different correct responses; "
                "overwriting picks one and destroys the evidence for the others."
            ),
        }


def reconcile(
    snapshot: SessionSnapshot,
    account: AccountSnapshot,
    broker_orders: Mapping[str, OrderStatus],
    *,
    checked_at: datetime,
    cash_tolerance: Decimal = DEFAULT_CASH_TOLERANCE,
) -> ReconciliationReport:
    """Compare persisted intent against broker state and report every difference.

    Args:
        snapshot: The recovered local view.
        account: The broker's authoritative account state.
        broker_orders: Broker order status keyed by client order ID.
        checked_at: When the comparison ran.
        cash_tolerance: Absolute cash difference treated as representation
            rather than disagreement.

    Returns:
        A report. ``must_halt`` is true whenever anything diverged.

    Raises:
        ReconciliationError: If the inputs cannot be compared at all.
    """
    if not isinstance(snapshot, SessionSnapshot):
        raise ReconciliationError("snapshot must be a SessionSnapshot")
    if not isinstance(account, AccountSnapshot):
        raise ReconciliationError("account must be an AccountSnapshot")
    if isinstance(cash_tolerance, float):
        raise ReconciliationError("cash_tolerance must be a Decimal, not a float")
    tolerance = Decimal(cash_tolerance)
    if not tolerance.is_finite() or tolerance < 0:
        raise ReconciliationError("cash_tolerance must be finite and non-negative")

    divergences: list[Divergence] = []

    local_positions = {item.symbol: item for item in snapshot.positions}
    broker_positions = {item.symbol: item for item in account.positions}
    for symbol in sorted(set(local_positions) | set(broker_positions)):
        local = local_positions.get(symbol)
        remote = broker_positions.get(symbol)
        if local is not None and remote is None:
            divergences.append(
                Divergence(
                    kind=DivergenceKind.POSITION_ONLY_LOCAL,
                    subject=symbol,
                    local_value=str(local.quantity),
                    broker_value="absent",
                    interpretation=(
                        "the system believes it holds this security and the broker does not "
                        "report it; either a fill was never executed or the position was "
                        "closed outside this system"
                    ),
                )
            )
        elif local is None and remote is not None:
            divergences.append(
                Divergence(
                    kind=DivergenceKind.POSITION_ONLY_BROKER,
                    subject=symbol,
                    local_value="absent",
                    broker_value=str(remote.quantity),
                    interpretation=(
                        "the broker reports a position this system never opened; a manual "
                        "trade, another process on the same account, or an unrecorded fill"
                    ),
                )
            )
        elif local is not None and remote is not None:
            difference = remote.quantity - local.quantity
            if abs(difference) > QUANTITY_EPSILON:
                divergences.append(
                    Divergence(
                        kind=DivergenceKind.POSITION_QUANTITY,
                        subject=symbol,
                        local_value=str(local.quantity),
                        broker_value=str(remote.quantity),
                        interpretation=(
                            f"quantities differ by {difference}; an unrecorded fill, a "
                            "duplicate submission, or a partial fill applied on one side only"
                        ),
                    )
                )

    local_cash = Decimal(snapshot.cash)
    cash_difference = account.cash - local_cash
    if abs(cash_difference) > tolerance:
        divergences.append(
            Divergence(
                kind=DivergenceKind.CASH,
                subject="cash",
                local_value=str(local_cash),
                broker_value=str(account.cash),
                interpretation=(
                    f"cash differs by {cash_difference}, beyond the {tolerance} tolerance; "
                    "an unrecorded fill, a fee, or a financing charge"
                ),
            )
        )

    local_open = {item.client_order_id: item for item in snapshot.open_intents()}
    broker_open = {
        key: value
        for key, value in broker_orders.items()
        if value.state
        not in {OrderState.FILLED, OrderState.CANCELED, OrderState.REJECTED, OrderState.EXPIRED}
    }
    for client_order_id in sorted(set(local_open) | set(broker_open)):
        local_intent = local_open.get(client_order_id)
        remote_order = broker_open.get(client_order_id)
        if local_intent is not None and remote_order is None:
            reported = broker_orders.get(client_order_id)
            if reported is None:
                divergences.append(
                    Divergence(
                        kind=DivergenceKind.ORDER_ONLY_LOCAL,
                        subject=client_order_id,
                        local_value=local_intent.submitted_state,
                        broker_value="unknown",
                        interpretation=(
                            "the system recorded an intent the broker has never heard of; the "
                            "submission may have been lost in flight, and resubmitting without "
                            "confirming would risk a duplicate"
                        ),
                    )
                )
            else:
                divergences.append(
                    Divergence(
                        kind=DivergenceKind.ORDER_STATE,
                        subject=client_order_id,
                        local_value=local_intent.submitted_state,
                        broker_value=reported.state.value,
                        interpretation=(
                            "the order reached a terminal state the system has not recorded; "
                            "its fill may be missing from local accounting"
                        ),
                    )
                )
        elif local_intent is None and remote_order is not None:
            divergences.append(
                Divergence(
                    kind=DivergenceKind.ORDER_ONLY_BROKER,
                    subject=client_order_id,
                    local_value="absent",
                    broker_value=remote_order.state.value,
                    interpretation=(
                        "the broker holds a working order this system has no record of; it "
                        "may fill at any moment against an unmodelled position"
                    ),
                )
            )
        elif (
            local_intent is not None
            and remote_order is not None
            and local_intent.submitted_state != remote_order.state.value
        ):
            divergences.append(
                Divergence(
                    kind=DivergenceKind.ORDER_STATE,
                    subject=client_order_id,
                    local_value=local_intent.submitted_state,
                    broker_value=remote_order.state.value,
                    interpretation="local and broker order states disagree",
                )
            )

    return ReconciliationReport(
        reconciled=not divergences,
        checked_at=checked_at,
        divergences=tuple(divergences[:MAX_REPORTED_DIVERGENCES]),
        positions_compared=len(set(local_positions) | set(broker_positions)),
        orders_compared=len(set(local_open) | set(broker_open)),
    )


def apply_fills_idempotently(
    fills: Iterable[Fill], *, opening: Sequence[Position] = ()
) -> tuple[Position, ...]:
    """Fold fills into a position book, ignoring duplicates and order of arrival.

    Keyed by ``fill_id``, so a fill delivered twice counts once and a sequence
    delivered out of order produces the same book as the in-order sequence. That
    convergence is what makes a dropped-and-redelivered acknowledgement safe to
    process without first proving it is new.

    A symbol whose net quantity reaches zero is **removed**, not kept at zero: a
    flat symbol that still appears in the book reconciles as a spurious
    position-only-local divergence.

    Raises:
        ReconciliationError: If two distinct fills share a ``fill_id``.
    """
    unique: dict[str, Fill] = {}
    for fill in fills:
        if not isinstance(fill, Fill):
            raise ReconciliationError("every element must be a Fill")
        existing = unique.get(fill.fill_id)
        if existing is None:
            unique[fill.fill_id] = fill
            continue
        if existing != fill:
            raise ReconciliationError(
                f"two different fills share fill_id {fill.fill_id!r}; identity is what makes "
                "deduplication safe, and a reused id makes it unsafe"
            )

    book: dict[str, tuple[Decimal, Decimal]] = {
        item.symbol: (item.quantity, item.average_entry_price) for item in opening
    }
    # Sorted by identity so the fold is deterministic regardless of arrival order.
    for fill in sorted(unique.values(), key=lambda item: item.fill_id):
        signed = fill.quantity if fill.side is OrderSide.BUY else -fill.quantity
        current = book.get(fill.symbol)
        if current is None:
            book[fill.symbol] = (signed, fill.price)
            continue
        quantity, average = current
        combined = quantity + signed
        if abs(combined) <= QUANTITY_EPSILON:
            del book[fill.symbol]
            continue
        if (quantity > 0) == (signed > 0):
            total = average * abs(quantity) + fill.price * abs(signed)
            average = total / abs(combined)
        elif (combined > 0) != (quantity > 0):
            average = fill.price
        book[fill.symbol] = (combined, average)

    return tuple(
        Position(symbol=symbol, quantity=quantity, average_entry_price=average)
        for symbol, (quantity, average) in sorted(book.items())
    )


def intents_from_orders(
    orders: Mapping[str, OrderStatus], *, decision_id: str, recorded_at: datetime
) -> tuple[OrderIntent, ...]:
    """Project broker order statuses into persistable intents.

    Used after recovery to write back what the broker confirmed, so the next
    restart begins from a reconciled view rather than the pre-crash one.
    """
    return tuple(
        OrderIntent(
            client_order_id=key,
            decision_id=decision_id,
            symbol=status.symbol,
            side=status.side.value,
            quantity=str(status.requested_quantity),
            submitted_state=status.state.value,
            recorded_at=recorded_at,
            broker_order_id=status.broker_order_id,
        )
        for key, status in sorted(orders.items())
    )


__all__ = [
    "DEFAULT_CASH_TOLERANCE",
    "MAX_REPORTED_DIVERGENCES",
    "Divergence",
    "DivergenceKind",
    "ReconciliationError",
    "ReconciliationReport",
    "apply_fills_idempotently",
    "intents_from_orders",
    "reconcile",
]
