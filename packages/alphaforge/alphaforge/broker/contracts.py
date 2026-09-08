"""Broker-neutral typed contracts for account, order, fill, and clock state.

SF-S5-MR3. These types name no vendor. They exist so an adapter can be swapped
without the callers above it changing, and so the semantics that matter for
correctness — idempotency, state transitions, time, and failure classification —
are stated once rather than re-derived per integration.

**Time is always UTC and always explicit.** A naive datetime is refused rather
than assumed to be local or assumed to be UTC; both assumptions are wrong
somewhere, and a silently mis-zoned order timestamp is the kind of defect that
only surfaces on a holiday boundary.

**Failures are classified, not merely raised.** The distinction between
:class:`RetryableBrokerError` and :class:`TerminalBrokerError` is the one that
decides whether a caller may resubmit. Getting it wrong in the safe direction
costs a missed cycle; getting it wrong in the unsafe direction duplicates an
order. Anything unrecognized is therefore terminal — a broker error the adapter
has never seen is not evidence that retrying is safe.

**Money and quantity are decimals.** Binary floating point cannot represent
0.01, and an accumulated cash balance that drifts by fractions of a cent will
eventually fail a reconciliation that is working correctly.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from datetime import UTC, datetime
from decimal import Decimal, InvalidOperation
from enum import StrEnum
from typing import Any, Final

#: Refusal thresholds, not tuning knobs.
MAX_SYMBOL_CHARS: Final = 16
MAX_CLIENT_ORDER_ID_CHARS: Final = 64
MAX_REASON_CHARS: Final = 512
MAX_POSITIONS: Final = 5_000

#: A quantity below this is treated as zero. Brokers report fractional shares to
#: a bounded precision; a residual smaller than this is rounding, not a position.
QUANTITY_EPSILON: Final = Decimal("0.000001")

_SYMBOL_PATTERN: Final = re.compile(r"^[A-Z][A-Z0-9.\-]{0,15}$")
_CLIENT_ORDER_ID_PATTERN: Final = re.compile(r"^[A-Za-z0-9._\-]{1,64}$")


class BrokerContractError(ValueError):
    """Raised when a broker value violates its contract before semantic use."""


class RetryableBrokerError(BrokerContractError):
    """A transient condition. The same request may be retried under a budget.

    Retryable does not mean "retry immediately" or "retry forever". The caller
    owns the backoff schedule and the attempt budget; exhausting either converts
    the condition into a terminal failure.
    """


class TerminalBrokerError(BrokerContractError):
    """A condition that will not be fixed by retrying the same request.

    Insufficient buying power, an unshortable symbol, a malformed order, a
    rejected authentication, or an unrecognized broker error. The decision cycle
    halts and surfaces the broker's reason verbatim.
    """


class OrderSide(StrEnum):
    """Direction of an order."""

    BUY = "buy"
    SELL = "sell"


class OrderType(StrEnum):
    """Order types the contract supports.

    Deliberately narrow. The strategy family needs market and limit orders; stop
    and trailing types are not included because an unused code path in an
    order-routing boundary is a liability rather than an option.
    """

    MARKET = "market"
    LIMIT = "limit"


class TimeInForce(StrEnum):
    """Time-in-force values the contract supports."""

    DAY = "day"
    GTC = "gtc"
    OPG = "opg"
    CLS = "cls"


class OrderState(StrEnum):
    """Lifecycle state of an order.

    The state machine is explicit because "what may follow what" is the property
    that keeps a fill from being applied twice or a canceled order from being
    treated as live.
    """

    PENDING_NEW = "pending_new"
    ACCEPTED = "accepted"
    PARTIALLY_FILLED = "partially_filled"
    FILLED = "filled"
    CANCELED = "canceled"
    REJECTED = "rejected"
    EXPIRED = "expired"


#: Terminal states. Nothing may follow them.
TERMINAL_ORDER_STATES: Final[frozenset[OrderState]] = frozenset(
    {OrderState.FILLED, OrderState.CANCELED, OrderState.REJECTED, OrderState.EXPIRED}
)

#: The only permitted transitions. Absence from this map is a refusal, not a
#: gap: an unlisted transition means the adapter and the broker disagree about
#: the order's history, and continuing would apply an update out of order.
ALLOWED_ORDER_TRANSITIONS: Final[dict[OrderState, frozenset[OrderState]]] = {
    OrderState.PENDING_NEW: frozenset(
        {
            OrderState.ACCEPTED,
            OrderState.REJECTED,
            OrderState.CANCELED,
            OrderState.PARTIALLY_FILLED,
            OrderState.FILLED,
            OrderState.EXPIRED,
        }
    ),
    OrderState.ACCEPTED: frozenset(
        {
            OrderState.PARTIALLY_FILLED,
            OrderState.FILLED,
            OrderState.CANCELED,
            OrderState.REJECTED,
            OrderState.EXPIRED,
        }
    ),
    OrderState.PARTIALLY_FILLED: frozenset(
        {OrderState.PARTIALLY_FILLED, OrderState.FILLED, OrderState.CANCELED, OrderState.EXPIRED}
    ),
    OrderState.FILLED: frozenset(),
    OrderState.CANCELED: frozenset(),
    OrderState.REJECTED: frozenset(),
    OrderState.EXPIRED: frozenset(),
}


def assert_transition_allowed(current: OrderState, proposed: OrderState) -> None:
    """Refuse an order-state transition the lifecycle does not permit.

    Raises:
        BrokerContractError: If the transition is not in
            :data:`ALLOWED_ORDER_TRANSITIONS`, naming both states.
    """
    if not isinstance(current, OrderState) or not isinstance(proposed, OrderState):
        raise BrokerContractError("order states must be OrderState members")
    permitted = ALLOWED_ORDER_TRANSITIONS[current]
    if proposed not in permitted:
        if current in TERMINAL_ORDER_STATES:
            raise BrokerContractError(
                f"order is already terminal in {current.value!r}; {proposed.value!r} cannot "
                "follow it. A late update for a settled order means local and broker "
                "history disagree and must be reconciled, not applied."
            )
        raise BrokerContractError(
            f"{current.value!r} -> {proposed.value!r} is not a permitted transition"
        )


def utc_timestamp(value: object, *, field_name: str) -> datetime:
    """Return a timezone-aware UTC datetime, refusing anything ambiguous.

    A naive datetime is refused rather than localized. Assuming local time and
    assuming UTC are both wrong somewhere, and the resulting defect surfaces at a
    session boundary where it is most expensive to diagnose.

    Raises:
        BrokerContractError: If the value is not an aware datetime.
    """
    if not isinstance(value, datetime):
        raise BrokerContractError(f"{field_name} must be a datetime, got {type(value).__name__}")
    if value.tzinfo is None or value.tzinfo.utcoffset(value) is None:
        raise BrokerContractError(
            f"{field_name} must be timezone-aware; a naive datetime is refused rather "
            "than assumed to be UTC or local, because both assumptions are wrong somewhere"
        )
    return value.astimezone(UTC)


def _decimal(value: object, *, field_name: str, allow_negative: bool = False) -> Decimal:
    """Return a finite Decimal from a Decimal, int, or str.

    ``int`` and ``str`` are accepted because broker payloads carry numbers as
    JSON strings, and forcing every call site to convert would move the parsing
    outward without making it safer. ``float`` is refused: it is the one input
    that cannot round-trip a cent exactly.

    Fields are annotated ``Decimal`` because that is what they hold after
    construction; a caller passing an int or str is normalized here.
    """
    if isinstance(value, bool):
        raise BrokerContractError(f"{field_name} must be numeric, not a bool")
    if isinstance(value, float):
        raise BrokerContractError(
            f"{field_name} must be a Decimal, int, or str — not a float. Binary floating "
            "point cannot represent 0.01, and an accumulated balance will drift out of "
            "reconciliation."
        )
    try:
        amount = Decimal(value) if isinstance(value, (int, str, Decimal)) else None
    except (InvalidOperation, ValueError) as error:
        raise BrokerContractError(f"{field_name} is not a valid decimal: {value!r}") from error
    if amount is None:
        raise BrokerContractError(f"{field_name} must be a Decimal, int, or str")
    if not amount.is_finite():
        raise BrokerContractError(f"{field_name} must be finite, got {amount}")
    if not allow_negative and amount < 0:
        raise BrokerContractError(f"{field_name} must not be negative, got {amount}")
    return amount


def validate_symbol(value: object) -> str:
    """Return a validated ticker symbol.

    Raises:
        BrokerContractError: On a malformed or oversized symbol.
    """
    if not isinstance(value, str):
        raise BrokerContractError(f"symbol must be a string, got {type(value).__name__}")
    if not _SYMBOL_PATTERN.match(value):
        raise BrokerContractError(
            f"symbol {value!r} is malformed; expected uppercase alphanumeric with . or -, "
            f"at most {MAX_SYMBOL_CHARS} characters"
        )
    return value


def validate_client_order_id(value: object) -> str:
    """Return a validated client order ID.

    Raises:
        BrokerContractError: On a malformed or oversized identifier.
    """
    if not isinstance(value, str):
        raise BrokerContractError(f"client_order_id must be a string, got {type(value).__name__}")
    if not _CLIENT_ORDER_ID_PATTERN.match(value):
        raise BrokerContractError(
            f"client_order_id {value!r} is malformed; expected 1-{MAX_CLIENT_ORDER_ID_CHARS} "
            "characters of ASCII alphanumerics, dot, underscore, or hyphen"
        )
    return value


def derive_client_order_id(
    *, strategy_id: str, decision_timestamp: datetime, symbol: str, side: OrderSide, sequence: int
) -> str:
    """Derive a deterministic client order ID from decision content.

    The same decision replayed produces the same identifier, which is what makes
    a resubmission a broker-side no-op instead of a second order. Derivation is
    content-based rather than random precisely so that a process restart cannot
    forget what it already sent.

    Raises:
        BrokerContractError: On any malformed component.
    """
    if not isinstance(strategy_id, str) or not strategy_id.strip():
        raise BrokerContractError("strategy_id must be a non-empty string")
    if isinstance(sequence, bool) or not isinstance(sequence, int) or sequence < 0:
        raise BrokerContractError("sequence must be a non-negative int")
    if not isinstance(side, OrderSide):
        raise BrokerContractError("side must be an OrderSide member")
    moment = utc_timestamp(decision_timestamp, field_name="decision_timestamp")
    payload = "|".join(
        (
            strategy_id.strip(),
            moment.isoformat(),
            validate_symbol(symbol),
            side.value,
            str(sequence),
        )
    )
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()
    return f"af-{digest[:32]}"


@dataclass(frozen=True, slots=True)
class OrderRequest:
    """An intent to trade, before any broker has seen it.

    Attributes:
        client_order_id: Deterministic identity from :func:`derive_client_order_id`.
        quantity: Share count. Must be strictly positive; direction lives in
            ``side`` so a negative quantity cannot silently invert an order.
        limit_price: Required for a limit order, refused for a market order. A
            limit price on a market order is a caller confusion, not a hint.
    """

    client_order_id: str
    symbol: str
    side: OrderSide
    quantity: Decimal
    order_type: OrderType
    time_in_force: TimeInForce
    limit_price: Decimal | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "client_order_id", validate_client_order_id(self.client_order_id))
        object.__setattr__(self, "symbol", validate_symbol(self.symbol))
        if not isinstance(self.side, OrderSide):
            raise BrokerContractError("side must be an OrderSide member")
        if not isinstance(self.order_type, OrderType):
            raise BrokerContractError("order_type must be an OrderType member")
        if not isinstance(self.time_in_force, TimeInForce):
            raise BrokerContractError("time_in_force must be a TimeInForce member")
        quantity = _decimal(self.quantity, field_name="quantity")
        if quantity <= QUANTITY_EPSILON:
            raise BrokerContractError(
                f"quantity must exceed {QUANTITY_EPSILON}; direction belongs in `side`, so a "
                "zero or negative quantity is a malformed order rather than a sell"
            )
        object.__setattr__(self, "quantity", quantity)
        if self.order_type is OrderType.LIMIT:
            if self.limit_price is None:
                raise BrokerContractError("a limit order requires limit_price")
            price = _decimal(self.limit_price, field_name="limit_price")
            if price <= 0:
                raise BrokerContractError("limit_price must be positive")
            object.__setattr__(self, "limit_price", price)
        elif self.limit_price is not None:
            raise BrokerContractError(
                "limit_price is not permitted on a market order; supplying one means the "
                "caller intended a limit order and would otherwise get an unbounded fill"
            )

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-friendly record with decimals as strings."""
        return {
            "client_order_id": self.client_order_id,
            "symbol": self.symbol,
            "side": self.side.value,
            "quantity": str(self.quantity),
            "order_type": self.order_type.value,
            "time_in_force": self.time_in_force.value,
            "limit_price": None if self.limit_price is None else str(self.limit_price),
        }


@dataclass(frozen=True, slots=True)
class Fill:
    """One execution against an order."""

    client_order_id: str
    symbol: str
    side: OrderSide
    quantity: Decimal
    price: Decimal
    filled_at: datetime
    fill_id: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "client_order_id", validate_client_order_id(self.client_order_id))
        object.__setattr__(self, "symbol", validate_symbol(self.symbol))
        if not isinstance(self.side, OrderSide):
            raise BrokerContractError("side must be an OrderSide member")
        quantity = _decimal(self.quantity, field_name="fill quantity")
        if quantity <= QUANTITY_EPSILON:
            raise BrokerContractError("a fill must have positive quantity")
        price = _decimal(self.price, field_name="fill price")
        if price <= 0:
            raise BrokerContractError("a fill must have a positive price")
        object.__setattr__(self, "quantity", quantity)
        object.__setattr__(self, "price", price)
        object.__setattr__(self, "filled_at", utc_timestamp(self.filled_at, field_name="filled_at"))
        if not isinstance(self.fill_id, str) or not self.fill_id.strip():
            raise BrokerContractError("fill_id must be a non-empty string")

    @property
    def notional(self) -> Decimal:
        """Signed cash effect: negative for a buy, positive for a sell."""
        gross = self.quantity * self.price
        return -gross if self.side is OrderSide.BUY else gross

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-friendly record."""
        return {
            "fill_id": self.fill_id,
            "client_order_id": self.client_order_id,
            "symbol": self.symbol,
            "side": self.side.value,
            "quantity": str(self.quantity),
            "price": str(self.price),
            "filled_at": self.filled_at.isoformat(),
            "notional": str(self.notional),
        }


@dataclass(frozen=True, slots=True)
class OrderStatus:
    """The broker's authoritative view of one order."""

    client_order_id: str
    broker_order_id: str
    symbol: str
    side: OrderSide
    state: OrderState
    requested_quantity: Decimal
    filled_quantity: Decimal
    average_fill_price: Decimal | None
    submitted_at: datetime
    updated_at: datetime
    reason: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "client_order_id", validate_client_order_id(self.client_order_id))
        object.__setattr__(self, "symbol", validate_symbol(self.symbol))
        if not isinstance(self.state, OrderState):
            raise BrokerContractError("state must be an OrderState member")
        if not isinstance(self.side, OrderSide):
            raise BrokerContractError("side must be an OrderSide member")
        if not isinstance(self.broker_order_id, str) or not self.broker_order_id.strip():
            raise BrokerContractError("broker_order_id must be a non-empty string")
        requested = _decimal(self.requested_quantity, field_name="requested_quantity")
        filled = _decimal(self.filled_quantity, field_name="filled_quantity")
        if filled > requested:
            raise BrokerContractError(
                f"filled_quantity {filled} exceeds requested_quantity {requested}; the broker "
                "reports having executed more than was asked for, which is a reconciliation "
                "fault rather than a fill"
            )
        object.__setattr__(self, "requested_quantity", requested)
        object.__setattr__(self, "filled_quantity", filled)
        if self.average_fill_price is not None:
            price = _decimal(self.average_fill_price, field_name="average_fill_price")
            if price <= 0:
                raise BrokerContractError("average_fill_price must be positive when present")
            object.__setattr__(self, "average_fill_price", price)
        if self.state is OrderState.FILLED and filled < requested:
            raise BrokerContractError(
                f"state is FILLED but {filled} of {requested} executed; a partially executed "
                "order is PARTIALLY_FILLED, and mislabelling it hides an unfilled residual"
            )
        if filled > 0 and self.average_fill_price is None:
            raise BrokerContractError(
                "a partially or fully filled order must report average_fill_price"
            )
        object.__setattr__(
            self, "submitted_at", utc_timestamp(self.submitted_at, field_name="submitted_at")
        )
        object.__setattr__(
            self, "updated_at", utc_timestamp(self.updated_at, field_name="updated_at")
        )
        if self.updated_at < self.submitted_at:
            raise BrokerContractError("updated_at precedes submitted_at")
        if self.reason is not None:
            if not isinstance(self.reason, str):
                raise BrokerContractError("reason must be a string when present")
            object.__setattr__(self, "reason", self.reason[:MAX_REASON_CHARS])

    @property
    def is_terminal(self) -> bool:
        """Whether no further update may follow."""
        return self.state in TERMINAL_ORDER_STATES

    @property
    def unfilled_quantity(self) -> Decimal:
        """Quantity requested but not executed."""
        return self.requested_quantity - self.filled_quantity

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-friendly record."""
        return {
            "client_order_id": self.client_order_id,
            "broker_order_id": self.broker_order_id,
            "symbol": self.symbol,
            "side": self.side.value,
            "state": self.state.value,
            "requested_quantity": str(self.requested_quantity),
            "filled_quantity": str(self.filled_quantity),
            "unfilled_quantity": str(self.unfilled_quantity),
            "average_fill_price": (
                None if self.average_fill_price is None else str(self.average_fill_price)
            ),
            "submitted_at": self.submitted_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "reason": self.reason,
            "is_terminal": self.is_terminal,
        }


@dataclass(frozen=True, slots=True)
class Position:
    """A held position. Quantity is signed: negative is short."""

    symbol: str
    quantity: Decimal
    average_entry_price: Decimal

    def __post_init__(self) -> None:
        object.__setattr__(self, "symbol", validate_symbol(self.symbol))
        object.__setattr__(
            self,
            "quantity",
            _decimal(self.quantity, field_name="position quantity", allow_negative=True),
        )
        price = _decimal(self.average_entry_price, field_name="average_entry_price")
        if price <= 0:
            raise BrokerContractError("average_entry_price must be positive")
        object.__setattr__(self, "average_entry_price", price)

    @property
    def is_short(self) -> bool:
        """Whether this is a short position."""
        return self.quantity < 0

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-friendly record."""
        return {
            "symbol": self.symbol,
            "quantity": str(self.quantity),
            "average_entry_price": str(self.average_entry_price),
            "is_short": self.is_short,
        }


@dataclass(frozen=True)
class AccountSnapshot:
    """The broker's authoritative account state at one instant.

    ``observed_at`` is mandatory: an account snapshot without a timestamp cannot
    be checked for staleness, and a stale snapshot is exactly what makes an
    exposure check pass when it should not.
    """

    account_id_digest: str
    cash: Decimal
    equity: Decimal
    buying_power: Decimal
    positions: tuple[Position, ...]
    observed_at: datetime
    is_paper: bool = True

    def __post_init__(self) -> None:
        # The account identifier never appears in clear: it is an account
        # identifier under the credential-custody policy, and a digest is
        # sufficient to detect a switched account.
        if not isinstance(self.account_id_digest, str) or len(self.account_id_digest) != 64:
            raise BrokerContractError(
                "account_id_digest must be a full SHA-256 hex digest; the raw account "
                "identifier must never be stored or logged"
            )
        object.__setattr__(
            self, "cash", _decimal(self.cash, field_name="cash", allow_negative=True)
        )
        object.__setattr__(
            self, "equity", _decimal(self.equity, field_name="equity", allow_negative=True)
        )
        object.__setattr__(
            self, "buying_power", _decimal(self.buying_power, field_name="buying_power")
        )
        positions = tuple(self.positions)
        if len(positions) > MAX_POSITIONS:
            raise BrokerContractError(f"position count exceeds the {MAX_POSITIONS} ceiling")
        symbols = [item.symbol for item in positions]
        if len(set(symbols)) != len(symbols):
            raise BrokerContractError(
                "duplicate symbol in positions; two rows for one symbol would double-count it"
            )
        object.__setattr__(self, "positions", tuple(sorted(positions, key=lambda p: p.symbol)))
        object.__setattr__(
            self, "observed_at", utc_timestamp(self.observed_at, field_name="observed_at")
        )
        if not isinstance(self.is_paper, bool):
            raise BrokerContractError("is_paper must be a bool")

    def position_for(self, symbol: str) -> Position | None:
        """Return the position in ``symbol``, or ``None`` if flat."""
        target = validate_symbol(symbol)
        for item in self.positions:
            if item.symbol == target:
                return item
        return None

    def age_seconds(self, *, now: datetime) -> float:
        """Return the snapshot's age in seconds against ``now``."""
        return (utc_timestamp(now, field_name="now") - self.observed_at).total_seconds()

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-friendly record. Never contains the raw account ID."""
        return {
            "account_id_digest": self.account_id_digest,
            "cash": str(self.cash),
            "equity": str(self.equity),
            "buying_power": str(self.buying_power),
            "positions": [item.to_dict() for item in self.positions],
            "observed_at": self.observed_at.isoformat(),
            "is_paper": self.is_paper,
        }


@dataclass(frozen=True, slots=True)
class MarketClock:
    """Broker-reported market state.

    Read from the broker rather than inferred from local time: holidays and early
    closes are not derivable from a weekday check, and a local-calendar guess is
    wrong a dozen times a year.
    """

    is_open: bool
    server_time: datetime
    next_open: datetime | None = None
    next_close: datetime | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.is_open, bool):
            raise BrokerContractError("is_open must be a bool")
        object.__setattr__(
            self, "server_time", utc_timestamp(self.server_time, field_name="server_time")
        )
        for field_name in ("next_open", "next_close"):
            value = getattr(self, field_name)
            if value is not None:
                object.__setattr__(self, field_name, utc_timestamp(value, field_name=field_name))

    def skew_seconds(self, *, local_time: datetime) -> float:
        """Return signed skew between local clock and broker server time."""
        local = utc_timestamp(local_time, field_name="local_time")
        return (local - self.server_time).total_seconds()

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-friendly record."""
        return {
            "is_open": self.is_open,
            "server_time": self.server_time.isoformat(),
            "next_open": None if self.next_open is None else self.next_open.isoformat(),
            "next_close": None if self.next_close is None else self.next_close.isoformat(),
        }


@dataclass(frozen=True, slots=True)
class Quote:
    """A price observation with a mandatory observation time."""

    symbol: str
    bid: Decimal
    ask: Decimal
    observed_at: datetime

    def __post_init__(self) -> None:
        object.__setattr__(self, "symbol", validate_symbol(self.symbol))
        bid = _decimal(self.bid, field_name="bid")
        ask = _decimal(self.ask, field_name="ask")
        if bid <= 0 or ask <= 0:
            raise BrokerContractError("bid and ask must be positive")
        if ask < bid:
            raise BrokerContractError(
                f"crossed quote: ask {ask} is below bid {bid}. A crossed book is a data "
                "fault, and pricing an order from it produces a fill that could not occur."
            )
        object.__setattr__(self, "bid", bid)
        object.__setattr__(self, "ask", ask)
        object.__setattr__(
            self, "observed_at", utc_timestamp(self.observed_at, field_name="observed_at")
        )

    @property
    def midpoint(self) -> Decimal:
        """Midpoint of the spread."""
        return (self.bid + self.ask) / Decimal(2)

    def age_seconds(self, *, now: datetime) -> float:
        """Return the quote's age in seconds against ``now``."""
        return (utc_timestamp(now, field_name="now") - self.observed_at).total_seconds()

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-friendly record."""
        return {
            "symbol": self.symbol,
            "bid": str(self.bid),
            "ask": str(self.ask),
            "midpoint": str(self.midpoint),
            "observed_at": self.observed_at.isoformat(),
        }


__all__ = [
    "ALLOWED_ORDER_TRANSITIONS",
    "MAX_CLIENT_ORDER_ID_CHARS",
    "MAX_POSITIONS",
    "MAX_REASON_CHARS",
    "MAX_SYMBOL_CHARS",
    "QUANTITY_EPSILON",
    "TERMINAL_ORDER_STATES",
    "AccountSnapshot",
    "BrokerContractError",
    "Fill",
    "MarketClock",
    "OrderRequest",
    "OrderSide",
    "OrderState",
    "OrderStatus",
    "OrderType",
    "Position",
    "Quote",
    "RetryableBrokerError",
    "TerminalBrokerError",
    "TimeInForce",
    "assert_transition_allowed",
    "derive_client_order_id",
    "utc_timestamp",
    "validate_client_order_id",
    "validate_symbol",
]
