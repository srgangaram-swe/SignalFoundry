"""Immutable, canonical execution events for deterministic daily-bar replay.

The event contract records logical market-session coordinates rather than wall
clock arrival time.  Its phase ordering encodes AlphaForge's causal daily-bar
timeline: open marks precede orders and fills; close marks precede close-time
signals and target decisions.  A content-derived identifier and strict
canonical JSON representation make duplicate detection and journal replay
independent of process identity.

``PortfolioMarked`` carries the bounded price snapshot required to replay and
independently reconcile invested positions.  An all-cash portfolio legitimately
has an empty snapshot.  Runtime journals can therefore contain licensed
observations and must remain private generated artifacts; this module, its
tests, and committed fixtures contain synthetic values only.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import dataclass
from datetime import date, datetime
from enum import IntEnum, StrEnum
from functools import total_ordering
from typing import Any, ClassVar, Literal, cast

SCHEMA_VERSION = "1.0.0"

MAX_IDENTIFIER_LENGTH = 128
MAX_SYMBOL_LENGTH = 64
MAX_REASON_LENGTH = 64
MAX_DETAIL_LENGTH = 512
MAX_TARGET_ASSETS = 512
MAX_MARK_PRICES = 5_000
MAX_BAR_INDEX = 10_000_000
MAX_PHASE_ORDINAL = 1_000_000
MAX_CANONICAL_BYTES = 131_072
MAX_ABSOLUTE_WEIGHT = 100.0
MAX_QUANTITY = 1.0e15
MAX_PRICE = 1.0e12
MAX_MONEY = 1.0e18

_IDENTIFIER = re.compile(r"[A-Za-z0-9][A-Za-z0-9._:-]{0,127}\Z", re.ASCII)
_SYMBOL = re.compile(r"[A-Za-z0-9][A-Za-z0-9._:/-]{0,63}\Z", re.ASCII)
_REASON = re.compile(r"[a-z][a-z0-9_]{0,63}\Z", re.ASCII)
_SHA256 = re.compile(r"[0-9a-f]{64}\Z", re.ASCII)
_CURRENCY = re.compile(r"[A-Z]{3}\Z", re.ASCII)


class ExecutionEventError(ValueError):
    """Raised when an execution event violates its canonical contract."""


class EventPhase(IntEnum):
    """Causal ordering for a single logical daily bar."""

    OPEN_MARK = 10
    ORDER_SUBMISSION = 20
    EXECUTION = 30
    DAY_CANCEL = 40
    CHARGE = 50
    CLOSE_MARK = 60
    SIGNAL = 70
    TARGET_DECISION = 80
    CONTROL = 90


class FeeCategory(StrEnum):
    """Stable fill-fee categories; MR4 may calculate each independently."""

    COMMISSION = "commission"
    EXCHANGE_FEE = "exchange_fee"
    OTHER_FEE = "other_fee"


type Side = Literal["buy", "sell"]
type OrderType = Literal["market", "limit"]
type TimeInForce = Literal["day"]
type MarkType = Literal["open", "close"]
type ChargeType = Literal["fees", "financing", "borrow", "other"]


def _identifier(value: Any, *, name: str) -> str:
    if not isinstance(value, str) or _IDENTIFIER.fullmatch(value) is None:
        raise ExecutionEventError(
            f"{name} must match {_IDENTIFIER.pattern!r} and be at most "
            f"{MAX_IDENTIFIER_LENGTH} ASCII characters"
        )
    return value


def _symbol(value: Any, *, name: str = "symbol") -> str:
    if not isinstance(value, str) or _SYMBOL.fullmatch(value) is None:
        raise ExecutionEventError(f"{name} must be a non-empty bounded ASCII market identifier")
    return value


def _reason(value: Any, *, name: str) -> str:
    if not isinstance(value, str) or _REASON.fullmatch(value) is None:
        raise ExecutionEventError(
            f"{name} must be a lowercase machine reason no longer than "
            f"{MAX_REASON_LENGTH} characters"
        )
    return value


def _digest(value: Any, *, name: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise ExecutionEventError(f"{name} must be a lowercase SHA-256 digest")
    return value


def _bounded_text(value: Any, *, name: str) -> str:
    if (
        not isinstance(value, str)
        or not value
        or value != value.strip()
        or len(value) > MAX_DETAIL_LENGTH
        or not value.isprintable()
    ):
        raise ExecutionEventError(
            f"{name} must be non-empty printable text no longer than {MAX_DETAIL_LENGTH} characters"
        )
    return value


def _bounded_integer(value: Any, *, name: str, maximum: int) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or not 0 <= value <= maximum:
        raise ExecutionEventError(f"{name} must be an integer in [0, {maximum}]")
    return value


def _finite_float(
    value: Any,
    *,
    name: str,
    maximum_absolute: float,
    positive: bool = False,
    nonnegative: bool = False,
) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ExecutionEventError(f"{name} must be a finite real number")
    try:
        result = float(value)
    except OverflowError as exc:
        raise ExecutionEventError(f"{name} must be a finite real number") from exc
    if not math.isfinite(result) or abs(result) > maximum_absolute:
        raise ExecutionEventError(
            f"{name} must be finite with magnitude at most {maximum_absolute:.3g}"
        )
    if positive and result <= 0.0:
        raise ExecutionEventError(f"{name} must be positive")
    if nonnegative and result < 0.0:
        raise ExecutionEventError(f"{name} must be non-negative")
    return 0.0 if result == 0.0 else result


def _daily_session(value: Any, *, name: str) -> date:
    if isinstance(value, datetime) or not isinstance(value, date):
        raise ExecutionEventError(f"{name} must be a datetime.date without a time component")
    return value


def _reconcile(
    left: float,
    right: float,
    *,
    name: str,
    terms: tuple[float, ...] = (),
) -> None:
    """Reconcile arithmetic at its actual scale without a currency-unit floor."""
    if left == right:
        return
    scale = max((abs(value) for value in (left, right, *terms)), default=0.0)
    if scale == 0.0:
        raise ExecutionEventError(f"{name} does not reconcile at exact zero scale")
    tolerance = 32.0 * max(len(terms), 1) * math.ulp(scale)
    if abs(left - right) > tolerance:
        raise ExecutionEventError(
            f"{name} does not reconcile: actual={left:.17g}, expected={right:.17g}"
        )


@dataclass(frozen=True, slots=True, order=True)
class EventCoordinate:
    """Logical location in the causal daily-bar event sequence.

    ``bar_index`` is the zero-based index in the run's frozen trading calendar.
    ``ordinal`` deterministically orders multiple events in the same phase.
    Calendar membership and monotonicity across events belong to the engine.
    """

    session: date
    bar_index: int
    phase: EventPhase
    ordinal: int = 0

    def __post_init__(self) -> None:
        object.__setattr__(self, "session", _daily_session(self.session, name="session"))
        object.__setattr__(
            self,
            "bar_index",
            _bounded_integer(self.bar_index, name="bar_index", maximum=MAX_BAR_INDEX),
        )
        if not isinstance(self.phase, EventPhase):
            raise ExecutionEventError("phase must be an EventPhase")
        object.__setattr__(
            self,
            "ordinal",
            _bounded_integer(self.ordinal, name="ordinal", maximum=MAX_PHASE_ORDINAL),
        )


@dataclass(frozen=True, slots=True)
class FeeComponent:
    """One non-negative fill fee in account-currency units."""

    category: FeeCategory
    amount: float

    def __post_init__(self) -> None:
        if not isinstance(self.category, FeeCategory):
            raise ExecutionEventError("fee category must be a FeeCategory")
        object.__setattr__(
            self,
            "amount",
            _finite_float(
                self.amount,
                name=f"{self.category.value} amount",
                maximum_absolute=MAX_MONEY,
                nonnegative=True,
            ),
        )


@dataclass(frozen=True, slots=True)
class SignalAvailable:
    """A signal artifact became available; raw signal values are excluded."""

    event_type: ClassVar[str] = "signal_available"
    signal_id: str
    model_id: str
    signal_digest: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "signal_id", _identifier(self.signal_id, name="signal_id"))
        object.__setattr__(self, "model_id", _identifier(self.model_id, name="model_id"))
        object.__setattr__(
            self,
            "signal_digest",
            _digest(self.signal_digest, name="signal_digest"),
        )


@dataclass(frozen=True, slots=True)
class TargetDecided:
    """An immutable target book eligible strictly after its decision session."""

    event_type: ClassVar[str] = "target_decided"
    target_id: str
    portfolio_id: str
    solver_id: str
    eligible_session: date
    cash_weight: float
    weights: tuple[tuple[str, float], ...]
    configuration_digest: str
    data_digest: str
    problem_digest: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "target_id", _identifier(self.target_id, name="target_id"))
        object.__setattr__(
            self,
            "portfolio_id",
            _identifier(self.portfolio_id, name="portfolio_id"),
        )
        object.__setattr__(self, "solver_id", _identifier(self.solver_id, name="solver_id"))
        object.__setattr__(
            self,
            "eligible_session",
            _daily_session(self.eligible_session, name="eligible_session"),
        )
        cash = _finite_float(
            self.cash_weight,
            name="cash_weight",
            maximum_absolute=MAX_ABSOLUTE_WEIGHT,
        )
        if not isinstance(self.weights, tuple) or len(self.weights) > MAX_TARGET_ASSETS:
            raise ExecutionEventError(
                f"weights must be a tuple with at most {MAX_TARGET_ASSETS} entries"
            )
        normalized: list[tuple[str, float]] = []
        seen: set[str] = set()
        for index, entry in enumerate(self.weights):
            if not isinstance(entry, tuple) or len(entry) != 2:
                raise ExecutionEventError(f"weights[{index}] must be an (asset, weight) tuple")
            asset = _symbol(entry[0], name=f"weights[{index}] asset")
            if asset in seen:
                raise ExecutionEventError(f"duplicate target asset {asset!r}")
            seen.add(asset)
            weight = _finite_float(
                entry[1],
                name=f"weight for {asset!r}",
                maximum_absolute=MAX_ABSOLUTE_WEIGHT,
            )
            normalized.append((asset, weight))
        normalized.sort(key=lambda item: item[0])
        try:
            net = math.fsum(weight for _, weight in normalized)
        except OverflowError as exc:
            raise ExecutionEventError("target weights overflowed") from exc
        _reconcile(
            net + cash,
            1.0,
            name="target weights plus cash",
            terms=(*[weight for _, weight in normalized], cash),
        )
        object.__setattr__(self, "cash_weight", cash)
        object.__setattr__(self, "weights", tuple(normalized))
        for field_name in ("configuration_digest", "data_digest", "problem_digest"):
            object.__setattr__(
                self,
                field_name,
                _digest(getattr(self, field_name), name=field_name),
            )


@dataclass(frozen=True, slots=True)
class OrderSubmitted:
    """One bounded daily order submitted after an open mark."""

    event_type: ClassVar[str] = "order_submitted"
    order_id: str
    symbol: str
    side: Side
    quantity: float
    order_type: OrderType = "market"
    time_in_force: TimeInForce = "day"
    limit_price: float | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "order_id", _identifier(self.order_id, name="order_id"))
        object.__setattr__(self, "symbol", _symbol(self.symbol))
        if not isinstance(self.side, str) or self.side not in {"buy", "sell"}:
            raise ExecutionEventError("side must be 'buy' or 'sell'")
        object.__setattr__(
            self,
            "quantity",
            _finite_float(
                self.quantity,
                name="quantity",
                maximum_absolute=MAX_QUANTITY,
                positive=True,
            ),
        )
        if not isinstance(self.order_type, str) or self.order_type not in {"market", "limit"}:
            raise ExecutionEventError("order_type must be 'market' or 'limit'")
        if not isinstance(self.time_in_force, str) or self.time_in_force != "day":
            raise ExecutionEventError("time_in_force must be 'day'")
        if self.order_type == "market":
            if self.limit_price is not None:
                raise ExecutionEventError("market orders cannot carry a limit_price")
        elif self.limit_price is None:
            raise ExecutionEventError("limit orders require a limit_price")
        else:
            object.__setattr__(
                self,
                "limit_price",
                _finite_float(
                    self.limit_price,
                    name="limit_price",
                    maximum_absolute=MAX_PRICE,
                    positive=True,
                ),
            )


@dataclass(frozen=True, slots=True)
class OrderAccepted:
    """A broker simulation accepted a bounded order quantity."""

    event_type: ClassVar[str] = "order_accepted"
    order_id: str
    accepted_quantity: float

    def __post_init__(self) -> None:
        object.__setattr__(self, "order_id", _identifier(self.order_id, name="order_id"))
        object.__setattr__(
            self,
            "accepted_quantity",
            _finite_float(
                self.accepted_quantity,
                name="accepted_quantity",
                maximum_absolute=MAX_QUANTITY,
                positive=True,
            ),
        )


@dataclass(frozen=True, slots=True)
class OrderRejected:
    """A broker simulation rejected an order with a stable reason code."""

    event_type: ClassVar[str] = "order_rejected"
    order_id: str
    reason_code: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "order_id", _identifier(self.order_id, name="order_id"))
        object.__setattr__(
            self,
            "reason_code",
            _reason(self.reason_code, name="reason_code"),
        )


@dataclass(frozen=True, slots=True)
class FillApplied:
    """One signed-side fill with reference price and categorized fees."""

    event_type: ClassVar[str] = "fill_applied"
    fill_id: str
    order_id: str
    symbol: str
    side: Side
    quantity: float
    reference_price: float
    price: float
    fees: tuple[FeeComponent, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "fill_id", _identifier(self.fill_id, name="fill_id"))
        object.__setattr__(self, "order_id", _identifier(self.order_id, name="order_id"))
        object.__setattr__(self, "symbol", _symbol(self.symbol))
        if not isinstance(self.side, str) or self.side not in {"buy", "sell"}:
            raise ExecutionEventError("side must be 'buy' or 'sell'")
        quantity = _finite_float(
            self.quantity,
            name="quantity",
            maximum_absolute=MAX_QUANTITY,
            positive=True,
        )
        reference = _finite_float(
            self.reference_price,
            name="reference_price",
            maximum_absolute=MAX_PRICE,
            positive=True,
        )
        price = _finite_float(
            self.price,
            name="price",
            maximum_absolute=MAX_PRICE,
            positive=True,
        )
        if quantity * max(reference, price) > MAX_MONEY:
            raise ExecutionEventError("fill notional exceeds the resource ceiling")
        if not isinstance(self.fees, tuple) or len(self.fees) > len(FeeCategory):
            raise ExecutionEventError("fees must be a bounded tuple of FeeComponent records")
        if any(not isinstance(item, FeeComponent) for item in self.fees):
            raise ExecutionEventError("fees must contain only FeeComponent records")
        normalized = tuple(sorted(self.fees, key=lambda item: item.category.value))
        categories = tuple(item.category for item in normalized)
        if len(set(categories)) != len(categories):
            raise ExecutionEventError("fill fee categories must be unique")
        try:
            total_fees = math.fsum(item.amount for item in normalized)
        except OverflowError as exc:
            raise ExecutionEventError("fill fees overflowed") from exc
        if not math.isfinite(total_fees) or total_fees > MAX_MONEY:
            raise ExecutionEventError("fill fees exceed the resource ceiling")
        object.__setattr__(self, "quantity", quantity)
        object.__setattr__(self, "reference_price", reference)
        object.__setattr__(self, "price", price)
        object.__setattr__(self, "fees", normalized)

    @property
    def total_fees(self) -> float:
        """Return total categorized fees in account-currency units."""
        return math.fsum(item.amount for item in self.fees)

    @property
    def signed_quantity(self) -> float:
        """Return positive buy or negative sell quantity."""
        return self.quantity if self.side == "buy" else -self.quantity


@dataclass(frozen=True, slots=True)
class OrderCancelled:
    """The unfilled remainder of a day order was cancelled."""

    event_type: ClassVar[str] = "order_cancelled"
    order_id: str
    reason_code: str
    cancelled_quantity: float

    def __post_init__(self) -> None:
        object.__setattr__(self, "order_id", _identifier(self.order_id, name="order_id"))
        object.__setattr__(
            self,
            "reason_code",
            _reason(self.reason_code, name="reason_code"),
        )
        object.__setattr__(
            self,
            "cancelled_quantity",
            _finite_float(
                self.cancelled_quantity,
                name="cancelled_quantity",
                maximum_absolute=MAX_QUANTITY,
                positive=True,
            ),
        )


@dataclass(frozen=True, slots=True)
class CashChargeAccrued:
    """A positive USD fee, financing, or borrow charge applied to cash."""

    event_type: ClassVar[str] = "cash_charge_accrued"
    charge_id: str
    charge_type: ChargeType
    amount: float
    currency: str = "USD"

    def __post_init__(self) -> None:
        object.__setattr__(self, "charge_id", _identifier(self.charge_id, name="charge_id"))
        if not isinstance(self.charge_type, str) or self.charge_type not in {
            "fees",
            "financing",
            "borrow",
            "other",
        }:
            raise ExecutionEventError("charge_type must be one of: fees, financing, borrow, other")
        object.__setattr__(
            self,
            "amount",
            _finite_float(
                self.amount,
                name="amount",
                maximum_absolute=MAX_MONEY,
                positive=True,
            ),
        )
        if not isinstance(self.currency, str) or _CURRENCY.fullmatch(self.currency) is None:
            raise ExecutionEventError("currency must be a three-letter uppercase ASCII code")
        if self.currency != "USD":
            raise ExecutionEventError(
                "cash charges support USD only; FX conversion is outside this schema"
            )


@dataclass(frozen=True, slots=True)
class PortfolioMarked:
    """Bounded open/close price snapshot plus independently auditable aggregates.

    ``prices`` may be empty only in the sense that no instrument needs a mark;
    the event engine remains responsible for requiring every held instrument.
    ``cash`` is already net of applied charges, so ``equity`` reconciles to
    ``cash + holdings_value``. ``accrued_charges`` is a non-negative cumulative
    diagnostic and is not subtracted twice.
    """

    event_type: ClassVar[str] = "portfolio_marked"
    mark_id: str
    mark_type: MarkType
    prices: tuple[tuple[str, float], ...]
    cash: float
    holdings_value: float
    accrued_charges: float
    equity: float

    def __post_init__(self) -> None:
        object.__setattr__(self, "mark_id", _identifier(self.mark_id, name="mark_id"))
        if not isinstance(self.mark_type, str) or self.mark_type not in {"open", "close"}:
            raise ExecutionEventError("mark_type must be 'open' or 'close'")
        if not isinstance(self.prices, tuple) or len(self.prices) > MAX_MARK_PRICES:
            raise ExecutionEventError(
                f"prices must be a tuple with at most {MAX_MARK_PRICES} entries"
            )
        normalized: list[tuple[str, float]] = []
        seen: set[str] = set()
        for index, entry in enumerate(self.prices):
            if not isinstance(entry, tuple) or len(entry) != 2:
                raise ExecutionEventError(f"prices[{index}] must be a (symbol, price) tuple")
            symbol = _symbol(entry[0], name=f"prices[{index}] symbol")
            if symbol in seen:
                raise ExecutionEventError(f"duplicate mark symbol {symbol!r}")
            seen.add(symbol)
            price = _finite_float(
                entry[1],
                name=f"price for {symbol!r}",
                maximum_absolute=MAX_PRICE,
                positive=True,
            )
            normalized.append((symbol, price))
        normalized.sort(key=lambda item: item[0])
        cash = _finite_float(
            self.cash,
            name="cash",
            maximum_absolute=MAX_MONEY,
        )
        holdings = _finite_float(
            self.holdings_value,
            name="holdings_value",
            maximum_absolute=MAX_MONEY,
        )
        charges = _finite_float(
            self.accrued_charges,
            name="accrued_charges",
            maximum_absolute=MAX_MONEY,
            nonnegative=True,
        )
        equity = _finite_float(
            self.equity,
            name="equity",
            maximum_absolute=MAX_MONEY,
        )
        _reconcile(
            equity,
            cash + holdings,
            name="marked equity",
            terms=(cash, holdings),
        )
        object.__setattr__(self, "prices", tuple(normalized))
        object.__setattr__(self, "cash", cash)
        object.__setattr__(self, "holdings_value", holdings)
        object.__setattr__(self, "accrued_charges", charges)
        object.__setattr__(self, "equity", equity)


@dataclass(frozen=True, slots=True)
class EngineHalted:
    """A terminal fail-closed control event."""

    event_type: ClassVar[str] = "engine_halted"
    reason_code: str
    detail: str
    recoverable: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "reason_code",
            _reason(self.reason_code, name="reason_code"),
        )
        object.__setattr__(self, "detail", _bounded_text(self.detail, name="detail"))
        if not isinstance(self.recoverable, bool):
            raise ExecutionEventError("recoverable must be a boolean")


type EventPayload = (
    SignalAvailable
    | TargetDecided
    | OrderSubmitted
    | OrderAccepted
    | OrderRejected
    | FillApplied
    | OrderCancelled
    | CashChargeAccrued
    | PortfolioMarked
    | EngineHalted
)

_PAYLOAD_TYPES: dict[str, type[EventPayload]] = {
    payload_type.event_type: payload_type
    for payload_type in (
        SignalAvailable,
        TargetDecided,
        OrderSubmitted,
        OrderAccepted,
        OrderRejected,
        FillApplied,
        OrderCancelled,
        CashChargeAccrued,
        PortfolioMarked,
        EngineHalted,
    )
}


def _expected_phase(payload: EventPayload) -> EventPhase:
    if isinstance(payload, PortfolioMarked):
        return EventPhase.OPEN_MARK if payload.mark_type == "open" else EventPhase.CLOSE_MARK
    mappings: tuple[tuple[type[Any], EventPhase], ...] = (
        (SignalAvailable, EventPhase.SIGNAL),
        (TargetDecided, EventPhase.TARGET_DECISION),
        (OrderSubmitted, EventPhase.ORDER_SUBMISSION),
        (OrderAccepted, EventPhase.EXECUTION),
        (OrderRejected, EventPhase.EXECUTION),
        (FillApplied, EventPhase.EXECUTION),
        (OrderCancelled, EventPhase.DAY_CANCEL),
        (CashChargeAccrued, EventPhase.CHARGE),
        (EngineHalted, EventPhase.CONTROL),
    )
    for payload_type, phase in mappings:
        if isinstance(payload, payload_type):
            return phase
    raise ExecutionEventError(f"unsupported payload type {type(payload).__name__!r}")


def _payload_to_dict(payload: EventPayload) -> dict[str, Any]:
    if isinstance(payload, SignalAvailable):
        return {
            "model_id": payload.model_id,
            "signal_digest": payload.signal_digest,
            "signal_id": payload.signal_id,
        }
    if isinstance(payload, TargetDecided):
        return {
            "cash_weight": payload.cash_weight,
            "configuration_digest": payload.configuration_digest,
            "data_digest": payload.data_digest,
            "eligible_session": payload.eligible_session.isoformat(),
            "portfolio_id": payload.portfolio_id,
            "problem_digest": payload.problem_digest,
            "solver_id": payload.solver_id,
            "target_id": payload.target_id,
            "weights": [[asset, weight] for asset, weight in payload.weights],
        }
    if isinstance(payload, OrderSubmitted):
        return {
            "limit_price": payload.limit_price,
            "order_id": payload.order_id,
            "order_type": payload.order_type,
            "quantity": payload.quantity,
            "side": payload.side,
            "symbol": payload.symbol,
            "time_in_force": payload.time_in_force,
        }
    if isinstance(payload, OrderAccepted):
        return {
            "accepted_quantity": payload.accepted_quantity,
            "order_id": payload.order_id,
        }
    if isinstance(payload, OrderRejected):
        return {"order_id": payload.order_id, "reason_code": payload.reason_code}
    if isinstance(payload, FillApplied):
        return {
            "fees": [
                {"amount": item.amount, "category": item.category.value} for item in payload.fees
            ],
            "fill_id": payload.fill_id,
            "order_id": payload.order_id,
            "price": payload.price,
            "quantity": payload.quantity,
            "reference_price": payload.reference_price,
            "side": payload.side,
            "symbol": payload.symbol,
        }
    if isinstance(payload, OrderCancelled):
        return {
            "cancelled_quantity": payload.cancelled_quantity,
            "order_id": payload.order_id,
            "reason_code": payload.reason_code,
        }
    if isinstance(payload, CashChargeAccrued):
        return {
            "amount": payload.amount,
            "charge_id": payload.charge_id,
            "charge_type": payload.charge_type,
            "currency": payload.currency,
        }
    if isinstance(payload, PortfolioMarked):
        return {
            "accrued_charges": payload.accrued_charges,
            "cash": payload.cash,
            "equity": payload.equity,
            "holdings_value": payload.holdings_value,
            "mark_id": payload.mark_id,
            "mark_type": payload.mark_type,
            "prices": [[symbol, price] for symbol, price in payload.prices],
        }
    if isinstance(payload, EngineHalted):
        return {
            "detail": payload.detail,
            "reason_code": payload.reason_code,
            "recoverable": payload.recoverable,
        }
    raise ExecutionEventError(f"unsupported payload type {type(payload).__name__!r}")


def _canonical_json_bytes(value: Any) -> bytes:
    try:
        return json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError, OverflowError) as exc:
        raise ExecutionEventError("event is not representable as canonical JSON") from exc


@total_ordering
@dataclass(frozen=True, slots=True)
class ExecutionEvent:
    """Canonical event envelope with content-derived identity and lineage."""

    run_id: str
    correlation_id: str
    entity_id: str
    coordinate: EventCoordinate
    payload: EventPayload
    causation_id: str | None = None

    def __post_init__(self) -> None:
        for field_name in ("run_id", "correlation_id", "entity_id"):
            object.__setattr__(
                self,
                field_name,
                _identifier(getattr(self, field_name), name=field_name),
            )
        if not isinstance(self.coordinate, EventCoordinate):
            raise ExecutionEventError("coordinate must be an EventCoordinate")
        if type(self.payload) not in _PAYLOAD_TYPES.values():
            raise ExecutionEventError("payload must be one supported immutable event payload")
        expected = _expected_phase(self.payload)
        if self.coordinate.phase is not expected:
            raise ExecutionEventError(
                f"{self.payload.event_type} requires phase {expected.name}, "
                f"got {self.coordinate.phase.name}"
            )
        if isinstance(self.payload, TargetDecided) and (
            self.payload.eligible_session <= self.coordinate.session
        ):
            raise ExecutionEventError(
                "target eligible_session must be strictly after its decision session"
            )
        if self.causation_id is not None:
            object.__setattr__(
                self,
                "causation_id",
                _digest(self.causation_id, name="causation_id"),
            )
        if len(self.canonical_bytes()) > MAX_CANONICAL_BYTES:
            raise ExecutionEventError(
                f"canonical event exceeds the {MAX_CANONICAL_BYTES}-byte ceiling"
            )

    @property
    def event_type(self) -> str:
        """Return the stable machine-readable payload type."""
        return self.payload.event_type

    def _body(self) -> dict[str, Any]:
        return {
            "causation_id": self.causation_id,
            "coordinate": {
                "bar_index": self.coordinate.bar_index,
                "ordinal": self.coordinate.ordinal,
                "phase": self.coordinate.phase.name.lower(),
                "session": self.coordinate.session.isoformat(),
            },
            "correlation_id": self.correlation_id,
            "entity_id": self.entity_id,
            "event_type": self.event_type,
            "payload": _payload_to_dict(self.payload),
            "run_id": self.run_id,
            "schema_version": SCHEMA_VERSION,
        }

    @property
    def event_id(self) -> str:
        """Return SHA-256 over the canonical envelope excluding the identifier."""
        return hashlib.sha256(_canonical_json_bytes(self._body())).hexdigest()

    def canonical_bytes(self) -> bytes:
        """Serialize the event as strict deterministic UTF-8 JSON without a newline."""
        document = {"event_id": self.event_id, **self._body()}
        encoded = _canonical_json_bytes(document)
        if len(encoded) > MAX_CANONICAL_BYTES:
            raise ExecutionEventError(
                f"canonical event exceeds the {MAX_CANONICAL_BYTES}-byte ceiling"
            )
        return encoded

    @classmethod
    def from_canonical_bytes(cls, value: bytes) -> ExecutionEvent:
        """Parse, validate, re-identify, and require an exact canonical encoding."""
        if not isinstance(value, bytes):
            raise ExecutionEventError("canonical event input must be bytes")
        if not value or len(value) > MAX_CANONICAL_BYTES:
            raise ExecutionEventError(
                f"canonical event byte length must lie in [1, {MAX_CANONICAL_BYTES}]"
            )
        try:
            text = value.decode("utf-8", errors="strict")
        except UnicodeDecodeError as exc:
            raise ExecutionEventError("canonical event must be valid UTF-8") from exc
        try:
            document = json.loads(
                text,
                object_pairs_hook=_unique_object,
                parse_constant=_reject_json_constant,
            )
        except (json.JSONDecodeError, ExecutionEventError, RecursionError, ValueError) as exc:
            if isinstance(exc, ExecutionEventError):
                raise
            raise ExecutionEventError("canonical event is not valid strict JSON") from exc
        if not isinstance(document, dict):
            raise ExecutionEventError("canonical event must be a JSON object")
        _exact_keys(
            document,
            {
                "causation_id",
                "coordinate",
                "correlation_id",
                "entity_id",
                "event_id",
                "event_type",
                "payload",
                "run_id",
                "schema_version",
            },
            name="event envelope",
        )
        if document["schema_version"] != SCHEMA_VERSION:
            raise ExecutionEventError(f"schema_version must be {SCHEMA_VERSION!r}")
        supplied_event_id = _digest(document["event_id"], name="event_id")
        coordinate = _coordinate_from_dict(document["coordinate"])
        payload = _payload_from_dict(document["event_type"], document["payload"])
        event = cls(
            run_id=document["run_id"],
            correlation_id=document["correlation_id"],
            entity_id=document["entity_id"],
            coordinate=coordinate,
            payload=payload,
            causation_id=document["causation_id"],
        )
        if event.event_id != supplied_event_id:
            raise ExecutionEventError("event_id does not match canonical event content")
        if event.canonical_bytes() != value:
            raise ExecutionEventError("event bytes are valid JSON but not the canonical encoding")
        return event

    def __lt__(self, other: object) -> bool:
        if not isinstance(other, ExecutionEvent):
            return NotImplemented
        return (self.coordinate, self.event_id) < (other.coordinate, other.event_id)


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ExecutionEventError(f"duplicate JSON key {key!r}")
        result[key] = value
    return result


def _reject_json_constant(value: str) -> None:
    raise ExecutionEventError(f"non-finite JSON constant {value!r} is forbidden")


def _exact_keys(value: Any, expected: set[str], *, name: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ExecutionEventError(f"{name} must be a JSON object")
    actual = set(value)
    if actual != expected:
        raise ExecutionEventError(
            f"{name} keys differ: missing={sorted(expected - actual)}, "
            f"unexpected={sorted(actual - expected)}"
        )
    return cast(dict[str, Any], value)


def _date_from_json(value: Any, *, name: str) -> date:
    if not isinstance(value, str):
        raise ExecutionEventError(f"{name} must be an ISO-8601 date")
    try:
        parsed = date.fromisoformat(value)
    except ValueError as exc:
        raise ExecutionEventError(f"{name} must be an ISO-8601 date") from exc
    if parsed.isoformat() != value:
        raise ExecutionEventError(f"{name} must use canonical ISO-8601 date form")
    return parsed


def _coordinate_from_dict(value: Any) -> EventCoordinate:
    document = _exact_keys(
        value,
        {"bar_index", "ordinal", "phase", "session"},
        name="event coordinate",
    )
    if not isinstance(document["phase"], str):
        raise ExecutionEventError("event coordinate phase must be a string")
    try:
        phase = EventPhase[document["phase"].upper()]
    except KeyError as exc:
        raise ExecutionEventError("event coordinate contains an unknown phase") from exc
    return EventCoordinate(
        session=_date_from_json(document["session"], name="coordinate session"),
        bar_index=document["bar_index"],
        phase=phase,
        ordinal=document["ordinal"],
    )


def _pair_tuple(value: Any, *, name: str) -> tuple[tuple[Any, Any], ...]:
    if not isinstance(value, list):
        raise ExecutionEventError(f"{name} must be a JSON array")
    pairs: list[tuple[Any, Any]] = []
    for index, entry in enumerate(value):
        if not isinstance(entry, list) or len(entry) != 2:
            raise ExecutionEventError(f"{name}[{index}] must be a two-item JSON array")
        pairs.append((entry[0], entry[1]))
    return tuple(pairs)


def _payload_from_dict(event_type: Any, value: Any) -> EventPayload:
    if not isinstance(event_type, str) or event_type not in _PAYLOAD_TYPES:
        raise ExecutionEventError("event_type is unknown")
    document = _exact_keys(value, _PAYLOAD_KEYS[event_type], name=f"{event_type} payload")
    if event_type == SignalAvailable.event_type:
        return SignalAvailable(**document)
    if event_type == TargetDecided.event_type:
        return TargetDecided(
            **{
                **document,
                "eligible_session": _date_from_json(
                    document["eligible_session"],
                    name="eligible_session",
                ),
                "weights": _pair_tuple(document["weights"], name="weights"),
            }
        )
    if event_type == OrderSubmitted.event_type:
        return OrderSubmitted(**document)
    if event_type == OrderAccepted.event_type:
        return OrderAccepted(**document)
    if event_type == OrderRejected.event_type:
        return OrderRejected(**document)
    if event_type == FillApplied.event_type:
        raw_fees = document["fees"]
        if not isinstance(raw_fees, list):
            raise ExecutionEventError("fill fees must be a JSON array")
        fees: list[FeeComponent] = []
        for index, raw_fee in enumerate(raw_fees):
            fee = _exact_keys(
                raw_fee,
                {"amount", "category"},
                name=f"fill fees[{index}]",
            )
            try:
                category = FeeCategory(fee["category"])
            except (TypeError, ValueError) as exc:
                raise ExecutionEventError("fill fee category is unknown") from exc
            fees.append(FeeComponent(category=category, amount=fee["amount"]))
        return FillApplied(**{**document, "fees": tuple(fees)})
    if event_type == OrderCancelled.event_type:
        return OrderCancelled(**document)
    if event_type == CashChargeAccrued.event_type:
        return CashChargeAccrued(**document)
    if event_type == PortfolioMarked.event_type:
        return PortfolioMarked(
            **{**document, "prices": _pair_tuple(document["prices"], name="prices")}
        )
    if event_type == EngineHalted.event_type:
        return EngineHalted(**document)
    raise ExecutionEventError("event_type is unsupported")


_PAYLOAD_KEYS: dict[str, set[str]] = {
    SignalAvailable.event_type: {"model_id", "signal_digest", "signal_id"},
    TargetDecided.event_type: {
        "cash_weight",
        "configuration_digest",
        "data_digest",
        "eligible_session",
        "portfolio_id",
        "problem_digest",
        "solver_id",
        "target_id",
        "weights",
    },
    OrderSubmitted.event_type: {
        "limit_price",
        "order_id",
        "order_type",
        "quantity",
        "side",
        "symbol",
        "time_in_force",
    },
    OrderAccepted.event_type: {"accepted_quantity", "order_id"},
    OrderRejected.event_type: {"order_id", "reason_code"},
    FillApplied.event_type: {
        "fees",
        "fill_id",
        "order_id",
        "price",
        "quantity",
        "reference_price",
        "side",
        "symbol",
    },
    OrderCancelled.event_type: {"cancelled_quantity", "order_id", "reason_code"},
    CashChargeAccrued.event_type: {"amount", "charge_id", "charge_type", "currency"},
    PortfolioMarked.event_type: {
        "accrued_charges",
        "cash",
        "equity",
        "holdings_value",
        "mark_id",
        "mark_type",
        "prices",
    },
    EngineHalted.event_type: {"detail", "reason_code", "recoverable"},
}


__all__ = [
    "CashChargeAccrued",
    "EngineHalted",
    "EventCoordinate",
    "EventPayload",
    "EventPhase",
    "ExecutionEvent",
    "ExecutionEventError",
    "FeeCategory",
    "FeeComponent",
    "FillApplied",
    "OrderAccepted",
    "OrderCancelled",
    "OrderRejected",
    "OrderSubmitted",
    "PortfolioMarked",
    "SignalAvailable",
    "TargetDecided",
]
