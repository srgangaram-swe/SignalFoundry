"""Versioned paper contracts, with bounded decimals and explicit UTC timestamps.

Wire decimals are strings: binary floats never enter cash/quantity arithmetic.
Unknown fields, non-finite values and ambiguous timestamps fail before use.
"""

from __future__ import annotations

import hashlib
import re
from datetime import UTC, datetime
from decimal import Decimal
from typing import Annotated, Literal, Self

from pydantic import (
    BaseModel,
    BeforeValidator,
    ConfigDict,
    Field,
    WithJsonSchema,
    model_validator,
)

from signal_foundry.boundary import encode


def money(value: object) -> Decimal:
    """Parse at most 18 integral and eight fractional digits without rounding."""
    if not isinstance(value, (str, Decimal)) or not re.fullmatch(
        r"-?\d{1,18}(?:\.\d{1,8})?", str(value)
    ):
        raise ValueError("Expected a bounded decimal string")
    return Decimal(value)


def timestamp(value: object) -> datetime:
    """Normalize an explicit offset to UTC; never guess a date's timezone."""
    if isinstance(value, str) and len(value) <= 40:
        value = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if not isinstance(value, datetime) or value.utcoffset() is None:
        raise ValueError("Expected a timezone-aware timestamp")
    try:
        return value.astimezone(UTC)
    except OverflowError as exc:
        raise ValueError("Timestamp is outside the UTC range") from exc


Amount = Annotated[Decimal, BeforeValidator(money)]
Positive = Annotated[Amount, Field(gt=0)]
Nonnegative = Annotated[
    Amount,
    Field(ge=0),
    WithJsonSchema(
        {"type": "string", "pattern": r"^\d{1,18}(?:\.\d{1,8})?$"}, mode="serialization"
    ),
]
Instant = Annotated[datetime, BeforeValidator(timestamp)]
Digest = Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]
Symbol = Annotated[str, Field(pattern=r"^[A-Z][A-Z0-9.\-]{0,15}$")]
Identifier = Annotated[str, Field(pattern=r"^[A-Za-z0-9_-]{1,48}$")]
Side = Literal["buy", "sell"]
OrderState = Literal[
    "new",
    "accepted",
    "pending_new",
    "partially_filled",
    "filled",
    "canceled",
    "expired",
    "rejected",
    "pending_cancel",
    "pending_replace",
    "replaced",
    "done_for_day",
    "stopped",
    "suspended",
    "calculated",
    "accepted_for_bidding",
]
TERMINAL = frozenset({"filled", "canceled", "expired", "rejected", "replaced"})


class Record(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, validate_default=True)

    def wire(self) -> dict[str, object]:
        return self.model_dump(mode="json")

    @property
    def identity(self) -> str:
        return hashlib.sha256(encode(self.wire())).hexdigest()


class Plan(Record):
    """Freeze the small candidate family and split before acquiring observations."""

    schema_version: Literal["intraday-plan-1"] = "intraday-plan-1"
    symbols: tuple[Symbol, ...] = Field(min_length=3, max_length=5)
    feed: Literal["iex", "sip"]
    start: Instant
    selection_end: Instant
    end: Instant
    window: int = Field(default=20, ge=2, le=120, strict=True)
    threshold_bps: int = Field(default=10, ge=0, le=100, strict=True)
    cost_bps_per_side: int = Field(default=10, ge=1, le=100, strict=True)
    seed: int = Field(default=17, ge=0, le=2**32 - 1, strict=True)

    @model_validator(mode="after")
    def dates(self) -> Self:
        if len(set(self.symbols)) != len(self.symbols):
            raise ValueError("Duplicate universe symbol")
        if not self.start < self.selection_end < self.end:
            raise ValueError("Expected start < selection_end < end")
        if (self.end - self.start).days > 366:
            raise ValueError("The acquisition interval exceeds one year")
        return self


class PaperConfig(Record):
    """Trusted local configuration; no URL, credentials or qualification flag."""

    schema_version: Literal["paper-config-1"] = "paper-config-1"
    environment: Literal["alpaca-paper"] = "alpaca-paper"
    enabled: bool = Field(default=False, strict=True)
    plan: Plan
    candidate: Literal["momentum", "mean_reversion"] = "momentum"
    account_digest: Digest
    maximum_order_notional: Positive = Decimal("100")
    maximum_position_notional: Positive = Decimal("500")
    maximum_session_loss: Positive = Decimal("25")
    maximum_spread_bps: Positive = Decimal("25")
    quantity: Positive = Decimal("1")
    max_orders: int = Field(default=100, ge=1, le=1000, strict=True)
    quote_age_seconds: int = Field(default=10, ge=1, le=30, strict=True)

    @model_validator(mode="after")
    def limits(self) -> Self:
        if not self.quantity == self.quantity.to_integral_value():
            raise ValueError("This version supports whole shares only")
        if not (
            self.maximum_order_notional
            <= self.maximum_position_notional
            <= Decimal("10000")
            and self.maximum_session_loss <= Decimal("1000")
        ):
            raise ValueError("Paper exposure ceilings are inconsistent")
        return self


class Bar(Record):
    """Raw one-minute bar, timestamped at its start; revisions remain explicit."""

    at: Instant
    open: Positive
    high: Positive
    low: Positive
    close: Positive
    volume: Nonnegative

    @model_validator(mode="after")
    def range(self) -> Self:
        if (
            self.low > min(self.open, self.close)
            or self.high < max(self.open, self.close)
            or self.low > self.high
        ):
            raise ValueError("Invalid OHLC range")
        return self


class Quote(Record):
    at: Instant
    bid: Positive
    ask: Positive
    bid_size: Positive
    ask_size: Positive

    @model_validator(mode="after")
    def spread(self) -> Self:
        if self.bid > self.ask:
            raise ValueError("Crossed quote")
        return self


class Clock(Record):
    at: Instant
    is_open: bool = Field(strict=True)
    next_open: Instant
    next_close: Instant


class Position(Record):
    symbol: Symbol
    quantity: Nonnegative
    market_value: Nonnegative


class Account(Record):
    digest: Digest
    cash: Amount
    equity: Positive
    buying_power: Nonnegative
    blocked: bool = Field(strict=True)
    positions: tuple[Position, ...] = Field(max_length=100)
    observed_at: Instant

    @model_validator(mode="after")
    def unique(self) -> Self:
        if len({p.symbol for p in self.positions}) != len(self.positions):
            raise ValueError("Duplicate position")
        return self


class Intent(Record):
    client_order_id: Identifier
    symbol: Symbol
    side: Side
    quantity: Positive
    limit_price: Positive


class Order(Record):
    broker_id: Annotated[str, Field(pattern=r"^[0-9a-fA-F-]{36}$")]
    intent: Intent
    state: OrderState
    filled_quantity: Nonnegative
    average_price: Positive | None

    @model_validator(mode="after")
    def fill(self) -> Self:
        if self.filled_quantity > self.intent.quantity:
            raise ValueError("Overfill")
        if bool(self.filled_quantity) != (self.average_price is not None):
            raise ValueError("Fill quantity and price disagree")
        if self.state == "filled" and self.filled_quantity != self.intent.quantity:
            raise ValueError("Incomplete terminal fill")
        if self.state == "partially_filled" and not (
            0 < self.filled_quantity < self.intent.quantity
        ):
            raise ValueError("Invalid partial fill quantity")
        return self


class PaperStatus(Record):
    environment: Literal["alpaca-paper"] = "alpaca-paper"
    configured: bool
    enabled: bool
    stopped: bool
    config_identity: str = Field(pattern=r"^(?:[0-9a-f]{64})?$")
    symbols: tuple[Symbol, ...] = Field(max_length=5)
    feed: Literal["iex", "sip", "none"]
    maximum_order_notional: str
    maximum_position_notional: str
    maximum_session_loss: str
    state: str = Field(max_length=64, pattern=r"^[a-z_]+$")
    orders: int = Field(ge=0, le=1000)
    events: int = Field(ge=0, le=100000)
    blockers: tuple[Annotated[str, Field(max_length=256)], ...] = Field(max_length=32)
    last_action: str = Field(max_length=64)
    paper_sessions: int = Field(ge=0)
    cash: Annotated[str, Field(max_length=40)] | None = None
    equity: Annotated[str, Field(max_length=40)] | None = None
    account_observed_at: Annotated[str, Field(max_length=40)] | None = None
    positions: tuple[Position, ...] = Field(default=(), max_length=100)
    live_authorized: Literal[False] = False
