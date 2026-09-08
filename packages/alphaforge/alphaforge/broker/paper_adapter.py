"""A strictly simulated paper broker behind the broker-neutral contract.

SF-S5-MR3. This adapter implements :mod:`alphaforge.broker.contracts` against an
in-process simulated book. **It contains no network client, no socket, no HTTP
library, and no credential read.** That is not a temporary state pending a real
implementation — it is the property that makes the module safe to run anywhere,
and a test asserts the module imports no networking library.

The simulation is deliberately pessimistic where reality is uncertain:

- A market order fills at the far side of the spread, never the midpoint. The
  midpoint is the price you get when someone else pays the spread.
- A limit order fills only when the market is already through the limit; it
  never fills "close enough".
- Nothing fills while the market is closed. Orders rest.

**Idempotency is enforced by the adapter, not assumed of the caller.**
Resubmitting a known client order ID returns the existing order unchanged and
does not create a second one. This mirrors what the broker must do and makes a
replayed decision cycle a no-op rather than double exposure.

**Paper fills are not evidence of executable performance.** This simulation has
no queue position, no contention, no partial-fill dynamics under real depth, and
no borrow scarcity. It bounds *operational* readiness — that the plumbing works,
reconciles, and fails closed — and nothing more.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from decimal import Decimal
from typing import Any, Final

from alphaforge.broker.config import (
    BrokerConfigurationError,
    BrokerSessionConfig,
    authorize_paper_session,
)
from alphaforge.broker.contracts import (
    QUANTITY_EPSILON,
    AccountSnapshot,
    BrokerContractError,
    Fill,
    MarketClock,
    OrderRequest,
    OrderSide,
    OrderState,
    OrderStatus,
    OrderType,
    Position,
    Quote,
    RetryableBrokerError,
    TerminalBrokerError,
    assert_transition_allowed,
    utc_timestamp,
)
from alphaforge.research.qualification import QualificationDecision

#: Bounds. Exceeding one is a refusal, not a degradation.
MAX_TRACKED_ORDERS: Final = 10_000
MAX_TRACKED_FILLS: Final = 50_000


class KillSwitchEngagedError(TerminalBrokerError):
    """Raised when the kill switch has halted the session.

    Terminal and one-way. Nothing re-enables a session whose kill switch fired;
    a new session object must be constructed deliberately.
    """


@dataclass
class PaperBrokerAdapter:
    """A simulated broker implementing the broker-neutral contract.

    Construction alone does not authorize anything: :meth:`connect` runs the full
    :func:`~alphaforge.broker.config.authorize_paper_session` check, and every
    order-facing method refuses until it has succeeded.

    Args:
        config: Session configuration. Defaults refuse.
        qualification: A ``QUALIFIED_FOR_PAPER`` decision. ``None`` refuses.
        opening_cash: Simulated starting cash.

    Raises:
        BrokerConfigurationError: On an unsafe configuration or missing
            qualification.
    """

    config: BrokerSessionConfig
    qualification: QualificationDecision | None
    account_id_digest: str
    opening_cash: Decimal = Decimal("100000")
    _connected: bool = field(default=False, init=False)
    _kill_switch_engaged: bool = field(default=False, init=False)
    _orders: dict[str, OrderStatus] = field(default_factory=dict, init=False)
    _fills: list[Fill] = field(default_factory=list, init=False)
    _positions: dict[str, Position] = field(default_factory=dict, init=False)
    _cash: Decimal = field(default=Decimal("0"), init=False)
    _next_broker_id: int = field(default=1, init=False)
    _next_fill_id: int = field(default=1, init=False)

    def __post_init__(self) -> None:
        if not isinstance(self.config, BrokerSessionConfig):
            raise BrokerConfigurationError("config must be a BrokerSessionConfig")
        if isinstance(self.opening_cash, float):
            raise BrokerContractError("opening_cash must be a Decimal, int, or str, not a float")
        cash = Decimal(self.opening_cash)
        if not cash.is_finite() or cash <= 0:
            raise BrokerContractError("opening_cash must be finite and positive")
        if not isinstance(self.account_id_digest, str) or len(self.account_id_digest) != 64:
            raise BrokerConfigurationError(
                "account_id_digest must be a full SHA-256 digest; the raw identifier is "
                "never stored"
            )
        self._cash = cash

    # -- lifecycle ---------------------------------------------------------

    def connect(self, *, environment: dict[str, str] | None = None) -> None:
        """Authorize and open the session.

        Raises:
            BrokerConfigurationError: If any authorization condition fails.
            LiveTradingNotAuthorizedError: If the endpoint is not paper-only.
        """
        authorize_paper_session(
            self.config, qualification=self.qualification, environment=environment
        )
        self._connected = True

    def engage_kill_switch(self) -> None:
        """Halt the session permanently.

        One-way for the lifetime of this object. Every subsequent order-facing
        call raises :class:`KillSwitchEngagedError`. There is no corresponding
        ``disengage``: a switch that can be flipped back by code is not a kill
        switch, and re-enabling must be a deliberate human act that constructs a
        new session.
        """
        self._kill_switch_engaged = True

    @property
    def kill_switch_engaged(self) -> bool:
        """Whether the kill switch has fired."""
        return self._kill_switch_engaged

    def _require_ready(self) -> None:
        """Refuse unless connected and not killed."""
        if self._kill_switch_engaged:
            raise KillSwitchEngagedError(
                "kill switch is engaged; this session is permanently halted and cannot be "
                "re-enabled. Construct a new session deliberately."
            )
        if not self._connected:
            raise BrokerConfigurationError(
                "session is not connected; call connect() so authorization is checked before "
                "any order-facing operation"
            )

    # -- market state ------------------------------------------------------

    def assert_quote_usable(self, quote: Quote, *, now: datetime) -> None:
        """Refuse a quote too old to price an order against.

        Raises:
            TerminalBrokerError: If the quote exceeds the configured age bound.
        """
        age = quote.age_seconds(now=now)
        if age < 0:
            raise TerminalBrokerError(
                f"quote for {quote.symbol} is timestamped {-age:.1f}s in the future; a clock "
                "fault or a fabricated observation, neither safe to trade on"
            )
        if age > self.config.max_quote_age_seconds:
            raise TerminalBrokerError(
                f"quote for {quote.symbol} is {age:.1f}s old, exceeding the "
                f"{self.config.max_quote_age_seconds:.1f}s bound; pricing an order from a "
                "stale quote produces a fill that could not occur"
            )

    def assert_clock_usable(self, clock: MarketClock, *, local_time: datetime) -> None:
        """Refuse when local and broker clocks disagree beyond the bound.

        Raises:
            TerminalBrokerError: If skew exceeds the configured bound.
        """
        skew = abs(clock.skew_seconds(local_time=local_time))
        if skew > self.config.max_clock_skew_seconds:
            raise TerminalBrokerError(
                f"clock skew {skew:.1f}s exceeds the {self.config.max_clock_skew_seconds:.1f}s "
                "bound; order timestamps, time-in-force, and market-hours logic all become "
                "unreliable"
            )

    def assert_account_fresh(self, snapshot: AccountSnapshot, *, now: datetime) -> None:
        """Refuse an account snapshot too old to support an exposure check.

        Raises:
            TerminalBrokerError: If the snapshot exceeds the configured age bound.
        """
        age = snapshot.age_seconds(now=now)
        if age > self.config.max_account_snapshot_age_seconds:
            raise TerminalBrokerError(
                f"account snapshot is {age:.1f}s old, exceeding the "
                f"{self.config.max_account_snapshot_age_seconds:.1f}s bound; a stale snapshot "
                "is what makes an exposure check pass when it should not"
            )

    # -- orders ------------------------------------------------------------

    def submit_order(
        self, request: OrderRequest, *, quote: Quote, clock: MarketClock, now: datetime
    ) -> OrderStatus:
        """Submit an order, or return the existing one for a known ID.

        Resubmitting a known ``client_order_id`` is a **success** that returns
        the existing order unchanged. That is what makes a replayed decision
        cycle a no-op instead of double exposure, and it mirrors the behaviour
        the real broker must provide.

        Raises:
            KillSwitchEngagedError: If the kill switch has fired.
            TerminalBrokerError: On a stale quote, clock skew, insufficient
                buying power, or an order-count breach.
        """
        self._require_ready()
        moment = utc_timestamp(now, field_name="now")

        existing = self._orders.get(request.client_order_id)
        if existing is not None:
            # Idempotent replay. Not an error: the caller may be recovering
            # from a connection loss and cannot know whether we saw this.
            return existing

        if len(self._orders) >= MAX_TRACKED_ORDERS:
            raise TerminalBrokerError(f"order count exceeds the {MAX_TRACKED_ORDERS} ceiling")
        if request.symbol != quote.symbol:
            raise TerminalBrokerError(
                f"quote is for {quote.symbol} but the order is for {request.symbol}"
            )
        self.assert_quote_usable(quote, now=moment)
        self.assert_clock_usable(clock, local_time=moment)

        if request.side is OrderSide.BUY:
            cost = request.quantity * quote.ask
            if cost > self._cash:
                raise TerminalBrokerError(
                    f"insufficient buying power: {request.symbol} costs {cost} against "
                    f"{self._cash} available. Terminal, not retryable — retrying an "
                    "underfunded order cannot succeed."
                )

        status = OrderStatus(
            client_order_id=request.client_order_id,
            broker_order_id=f"paper-{self._next_broker_id}",
            symbol=request.symbol,
            side=request.side,
            state=OrderState.ACCEPTED,
            requested_quantity=request.quantity,
            filled_quantity=Decimal("0"),
            average_fill_price=None,
            submitted_at=moment,
            updated_at=moment,
        )
        self._next_broker_id += 1
        self._orders[request.client_order_id] = status

        if not clock.is_open:
            # Rests until the market opens. Simulating a closed-market fill
            # would invent liquidity that did not exist.
            return status
        return self._attempt_fill(request, status, quote=quote, now=moment)

    def _attempt_fill(
        self, request: OrderRequest, status: OrderStatus, *, quote: Quote, now: datetime
    ) -> OrderStatus:
        """Fill an accepted order against the simulated book, pessimistically."""
        if request.order_type is OrderType.MARKET:
            # Far side of the spread. The midpoint is what you get when someone
            # else pays the spread, and assuming it flatters every result.
            price = quote.ask if request.side is OrderSide.BUY else quote.bid
        else:
            assert request.limit_price is not None  # guaranteed by OrderRequest
            marketable = (
                quote.ask <= request.limit_price
                if request.side is OrderSide.BUY
                else quote.bid >= request.limit_price
            )
            if not marketable:
                return status
            price = quote.ask if request.side is OrderSide.BUY else quote.bid

        fill = Fill(
            client_order_id=request.client_order_id,
            symbol=request.symbol,
            side=request.side,
            quantity=request.quantity,
            price=price,
            filled_at=now,
            fill_id=f"fill-{self._next_fill_id}",
        )
        self._next_fill_id += 1
        return self._apply_fill(fill, status)

    def _apply_fill(self, fill: Fill, status: OrderStatus) -> OrderStatus:
        """Apply a fill to order, position, and cash state atomically."""
        if len(self._fills) >= MAX_TRACKED_FILLS:
            raise TerminalBrokerError(f"fill count exceeds the {MAX_TRACKED_FILLS} ceiling")
        filled = status.filled_quantity + fill.quantity
        proposed = (
            OrderState.FILLED
            if filled >= status.requested_quantity - QUANTITY_EPSILON
            else OrderState.PARTIALLY_FILLED
        )
        assert_transition_allowed(status.state, proposed)

        if status.average_fill_price is None:
            average = fill.price
        else:
            prior_notional = status.average_fill_price * status.filled_quantity
            average = (prior_notional + fill.price * fill.quantity) / filled

        updated = OrderStatus(
            client_order_id=status.client_order_id,
            broker_order_id=status.broker_order_id,
            symbol=status.symbol,
            side=status.side,
            state=proposed,
            requested_quantity=status.requested_quantity,
            filled_quantity=filled,
            average_fill_price=average,
            submitted_at=status.submitted_at,
            updated_at=fill.filled_at,
        )
        self._orders[status.client_order_id] = updated
        self._fills.append(fill)
        self._cash += fill.notional
        self._apply_position(fill)
        return updated

    def _apply_position(self, fill: Fill) -> None:
        """Update the position book for one fill."""
        signed = fill.quantity if fill.side is OrderSide.BUY else -fill.quantity
        existing = self._positions.get(fill.symbol)
        if existing is None:
            self._positions[fill.symbol] = Position(
                symbol=fill.symbol, quantity=signed, average_entry_price=fill.price
            )
            return
        combined = existing.quantity + signed
        if abs(combined) <= QUANTITY_EPSILON:
            # Flat. Removed rather than kept at zero so a flat symbol cannot
            # look like a position during reconciliation.
            del self._positions[fill.symbol]
            return
        if (existing.quantity > 0) == (signed > 0):
            # Adding to the position: blend the entry price.
            total_cost = existing.average_entry_price * abs(existing.quantity) + fill.price * abs(
                signed
            )
            average = total_cost / abs(combined)
        else:
            # Reducing or flipping: the entry price of the survivor is the fill
            # price when flipped, otherwise unchanged.
            average = (
                fill.price
                if (combined > 0) != (existing.quantity > 0)
                else existing.average_entry_price
            )
        self._positions[fill.symbol] = Position(
            symbol=fill.symbol, quantity=combined, average_entry_price=average
        )

    def cancel_order(self, client_order_id: str, *, now: datetime) -> OrderStatus:
        """Cancel a resting order.

        Raises:
            TerminalBrokerError: If the order is unknown or already terminal.
        """
        self._require_ready()
        status = self._orders.get(client_order_id)
        if status is None:
            raise TerminalBrokerError(f"unknown client_order_id {client_order_id!r}")
        if status.is_terminal:
            raise TerminalBrokerError(
                f"order {client_order_id!r} is already terminal in {status.state.value!r}; "
                "cancelling a settled order would rewrite history"
            )
        assert_transition_allowed(status.state, OrderState.CANCELED)
        canceled = OrderStatus(
            client_order_id=status.client_order_id,
            broker_order_id=status.broker_order_id,
            symbol=status.symbol,
            side=status.side,
            state=OrderState.CANCELED,
            requested_quantity=status.requested_quantity,
            filled_quantity=status.filled_quantity,
            average_fill_price=status.average_fill_price,
            submitted_at=status.submitted_at,
            updated_at=utc_timestamp(now, field_name="now"),
            reason="canceled_by_client",
        )
        self._orders[client_order_id] = canceled
        return canceled

    def get_order(self, client_order_id: str) -> OrderStatus:
        """Return the authoritative status of one order.

        This is the method a caller uses after a connection loss to establish
        what actually happened, rather than assuming and resubmitting.

        Raises:
            TerminalBrokerError: If the order is unknown.
        """
        self._require_ready()
        status = self._orders.get(client_order_id)
        if status is None:
            raise TerminalBrokerError(f"unknown client_order_id {client_order_id!r}")
        return status

    def list_fills(self) -> tuple[Fill, ...]:
        """Return every fill in deterministic order."""
        self._require_ready()
        return tuple(self._fills)

    def account_snapshot(self, *, now: datetime) -> AccountSnapshot:
        """Return the current simulated account state."""
        self._require_ready()
        positions = tuple(self._positions.values())
        equity = self._cash + sum(
            (item.quantity * item.average_entry_price for item in positions), Decimal("0")
        )
        return AccountSnapshot(
            account_id_digest=self.account_id_digest,
            cash=self._cash,
            equity=equity,
            buying_power=max(self._cash, Decimal("0")),
            positions=positions,
            observed_at=utc_timestamp(now, field_name="now"),
            is_paper=True,
        )

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-friendly session summary. Contains no credential."""
        return {
            "connected": self._connected,
            "kill_switch_engaged": self._kill_switch_engaged,
            "config": self.config.to_dict(),
            "order_count": len(self._orders),
            "fill_count": len(self._fills),
            "position_count": len(self._positions),
            "simulated": True,
            "network_capable": False,
            "evidence_note": (
                "Paper fills are simulated. There is no queue position, no contention, no "
                "partial-fill dynamics under real depth, and no borrow scarcity. This bounds "
                "operational readiness, not executable performance."
            ),
        }


__all__ = [
    "MAX_TRACKED_FILLS",
    "MAX_TRACKED_ORDERS",
    "KillSwitchEngagedError",
    "PaperBrokerAdapter",
    "RetryableBrokerError",
]
