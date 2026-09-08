"""Deterministic execution-event reducer with fail-closed replay semantics.

The engine owns order lifecycle state and is the only supported mutation path
for its portfolio ledger.  Event payloads are validated independently by
``alphaforge.execution.events``; this reducer additionally validates causation,
logical time, order transitions, accounting effects, and resource ceilings.

Ledger mutations are validated on an independent candidate before the journal
append.  The candidate becomes authoritative only after the event is durable.
If a durable append fails or has an uncertain outcome, the engine is permanently
poisoned: callers must discard it and reconstruct from ``replay`` over the
journal's verified committed prefix.
"""

from __future__ import annotations

import hashlib
import heapq
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from datetime import date, datetime
from enum import StrEnum
from types import MappingProxyType
from typing import Final

from alphaforge.backtesting.journal import InMemoryJournal, Journal, JournalError
from alphaforge.backtesting.ledger import LedgerSnapshot, PortfolioLedger, reconciles
from alphaforge.execution.events import (
    CashChargeAccrued,
    EngineHalted,
    EventPhase,
    ExecutionEvent,
    FillApplied,
    OrderAccepted,
    OrderCancelled,
    OrderRejected,
    OrderSubmitted,
    PortfolioMarked,
    SignalAvailable,
    TargetDecided,
)

DEFAULT_MAX_EVENTS: Final = 1_000_000
DEFAULT_MAX_PENDING_EVENTS: Final = 100_000
DEFAULT_MAX_ORDERS: Final = 500_000
DEFAULT_MAX_OPEN_ORDERS: Final = 50_000

type EventSortKey = tuple[date, int, EventPhase, int, str]


class EventEngineError(RuntimeError):
    """Base class for deterministic event-engine failures."""


class EventOrderingError(EventEngineError):
    """Raised when an event attempts logical time travel."""


class EventCausationError(EventEngineError):
    """Raised when an event has an absent or incompatible cause."""


class EventTransitionError(EventEngineError):
    """Raised when an order or control transition is invalid."""


class EventResourceLimitError(EventEngineError):
    """Raised before an event would exceed a configured resource ceiling."""


class EventReplayError(EventEngineError):
    """Raised when a verified byte chain contains invalid event semantics."""


class EngineRecoveryRequiredError(EventEngineError):
    """Raised after a journal failure makes in-memory state non-publishable."""


class OrderLifecycle(StrEnum):
    """Finite order lifecycle accepted by the reducer."""

    SUBMITTED = "submitted"
    ACCEPTED = "accepted"
    PARTIALLY_FILLED = "partially_filled"
    FILLED = "filled"
    REJECTED = "rejected"
    CANCELLED = "cancelled"

    @property
    def terminal(self) -> bool:
        """Return whether no further order transition is permitted."""
        return self in {self.FILLED, self.REJECTED, self.CANCELLED}


@dataclass(frozen=True, slots=True)
class OrderState:
    """Immutable reducer state for one submitted order."""

    order_id: str
    symbol: str
    side: str
    requested_quantity: float
    accepted_quantity: float
    filled_quantity: float
    cancelled_quantity: float
    status: OrderLifecycle
    correlation_id: str
    submitted_session: date
    submitted_bar_index: int

    @property
    def residual_quantity(self) -> float:
        """Return the requested quantity not filled or explicitly cancelled."""
        residual = self.requested_quantity - self.filled_quantity - self.cancelled_quantity
        return (
            0.0
            if reconciles(
                residual,
                0.0,
                operands=(
                    self.requested_quantity,
                    self.filled_quantity,
                    self.cancelled_quantity,
                ),
            )
            else residual
        )


@dataclass(frozen=True, slots=True)
class EngineSnapshot:
    """Immutable externally inspectable event-engine state."""

    run_id: str
    processed_events: int
    pending_events: int
    orders: Mapping[str, OrderState]
    portfolio: LedgerSnapshot | None
    halted: bool
    halt_reason: str | None
    recovery_required: bool


@dataclass(frozen=True, slots=True)
class _EventReference:
    payload_type: type[object]
    order_id: str | None
    correlation_id: str
    sort_key: EventSortKey
    session: date
    bar_index: int
    eligible_session: date | None = None
    target_symbols: frozenset[str] = frozenset()


@dataclass(frozen=True, slots=True)
class _PlannedTransition:
    order_update: OrderState | None = None
    ledger_update: PortfolioLedger | None = None
    portfolio: LedgerSnapshot | None = None
    halt_pending: str | None = None
    halted: bool = False
    halt_reason: str | None = None


def _bounded_positive_int(value: object, *, name: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _validated_calendar(value: object) -> tuple[date, ...]:
    """Return one bounded, strictly increasing date-only trading calendar."""

    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise TypeError("calendar must be a finite sequence of date values")
    if not value:
        raise ValueError("calendar must contain at least one session")
    if len(value) > 10_000_001:
        raise ValueError("calendar exceeds the supported bar-index range")
    sessions: list[date] = []
    for session in value:
        if isinstance(session, datetime) or not isinstance(session, date):
            raise TypeError("calendar sessions must be datetime.date values without time")
        if sessions and session <= sessions[-1]:
            raise ValueError("calendar sessions must be unique and strictly increasing")
        sessions.append(session)
    return tuple(sessions)


def _event_sort_key(event: ExecutionEvent) -> EventSortKey:
    coordinate = event.coordinate
    return (
        coordinate.session,
        coordinate.bar_index,
        coordinate.phase,
        coordinate.ordinal,
        event.event_id,
    )


def _fingerprint(event: ExecutionEvent) -> bytes:
    return hashlib.sha256(event.canonical_bytes()).digest()


def _is_single_addition_roundoff_overrun(actual: float, expected: float) -> bool:
    """Return whether a positive overrun is at most one representable ULP."""

    if not math.isfinite(actual) or not math.isfinite(expected) or actual <= expected:
        return False
    scale = max(abs(actual), abs(expected))
    return actual - expected <= math.ulp(scale)


class DeterministicEventEngine:
    """Bounded order-state and accounting reducer for one immutable run.

    Queue insertion is ``O(log Q)``. Order-only transitions are ``O(1)``;
    fill/charge candidates and portfolio marks are ``O(P + R)`` because the
    current ledger intentionally rechecks cumulative position and realized-P&L
    invariants. Journal verification on replay is ``O(E)`` and intentionally
    precedes semantic state reconstruction.
    """

    def __init__(
        self,
        run_id: str,
        *,
        calendar: Sequence[date],
        initial_cash: float = 1_000_000.0,
        journal: Journal | None = None,
        max_events: int = DEFAULT_MAX_EVENTS,
        max_pending_events: int = DEFAULT_MAX_PENDING_EVENTS,
        max_orders: int = DEFAULT_MAX_ORDERS,
        max_open_orders: int = DEFAULT_MAX_OPEN_ORDERS,
    ) -> None:
        if not isinstance(run_id, str) or not run_id.strip():
            raise ValueError("run_id must be a non-empty string")
        self._calendar = _validated_calendar(calendar)
        self._calendar_index = {session: index for index, session in enumerate(self._calendar)}
        self._max_events = _bounded_positive_int(max_events, name="max_events")
        self._max_pending = _bounded_positive_int(max_pending_events, name="max_pending_events")
        self._max_orders = _bounded_positive_int(max_orders, name="max_orders")
        self._max_open_orders = _bounded_positive_int(max_open_orders, name="max_open_orders")
        if self._max_open_orders > self._max_orders:
            raise ValueError("max_open_orders cannot exceed max_orders")
        owned_journal = InMemoryJournal(max_events=self._max_events)
        selected_journal = owned_journal if journal is None else journal
        if selected_journal.count != 0:
            raise ValueError("non-empty journals must be opened through replay()")

        self._run_id = run_id
        self._journal = selected_journal
        self._owns_journal = journal is None
        self._ledger = PortfolioLedger(initial_cash)
        self._orders: dict[str, OrderState] = {}
        self._open_order_count = 0
        self._references: dict[str, _EventReference] = {}
        self._fingerprints: dict[str, bytes] = {}
        self._identity_origins: dict[tuple[str, str], str] = {}
        self._pending: list[tuple[EventSortKey, ExecutionEvent]] = []
        self._pending_fingerprints: dict[str, bytes] = {}
        self._cursor: EventSortKey | None = None
        self._open_mark_bar: int | None = None
        self._open_prices: Mapping[str, float] = MappingProxyType({})
        self._close_mark_bar: int | None = None
        self._last_portfolio: LedgerSnapshot | None = None
        self._halt_pending: str | None = None
        self._halt_cause_event_id: str | None = None
        self._halted = False
        self._halt_reason: str | None = None
        self._recovery_required = False

    @classmethod
    def replay(
        cls,
        journal: Journal,
        *,
        calendar: Sequence[date],
        initial_cash: float = 1_000_000.0,
        expected_run_id: str | None = None,
        max_events: int = DEFAULT_MAX_EVENTS,
        max_pending_events: int = DEFAULT_MAX_PENDING_EVENTS,
        max_orders: int = DEFAULT_MAX_ORDERS,
        max_open_orders: int = DEFAULT_MAX_OPEN_ORDERS,
    ) -> DeterministicEventEngine:
        """Verify ``journal`` and reconstruct its exact semantic state."""
        events = journal.events()
        if not events:
            if expected_run_id is None:
                raise EventReplayError("an empty journal requires expected_run_id")
            return cls(
                expected_run_id,
                calendar=calendar,
                initial_cash=initial_cash,
                journal=journal,
                max_events=max_events,
                max_pending_events=max_pending_events,
                max_orders=max_orders,
                max_open_orders=max_open_orders,
            )
        run_id = events[0].run_id
        if expected_run_id is not None and run_id != expected_run_id:
            raise EventReplayError("journal run_id does not match expected_run_id")

        # Construct against a temporary empty journal, then attach the already
        # verified durable journal after semantic replay.
        engine = cls(
            run_id,
            calendar=calendar,
            initial_cash=initial_cash,
            max_events=max_events,
            max_pending_events=max_pending_events,
            max_orders=max_orders,
            max_open_orders=max_open_orders,
        )
        try:
            for event in events:
                engine._process(event, append=False)
        except EventEngineError as exc:
            raise EventReplayError("journal event semantics failed replay") from exc
        if engine._owns_journal:
            engine._journal.close()
        engine._journal = journal
        engine._owns_journal = False
        return engine

    @property
    def run_id(self) -> str:
        return self._run_id

    @property
    def journal(self) -> Journal:
        """Return the journal for verified export and caller-managed closure."""
        return self._journal

    @property
    def positions(self) -> Mapping[str, float]:
        """Return a detached read-only copy of authoritative signed positions."""

        self._require_usable()
        return self._ledger.positions

    @property
    def cash(self) -> float:
        """Return authoritative cash without exposing a mutation capability."""

        self._require_usable()
        return self._ledger.cash

    @property
    def total_charges(self) -> float:
        """Return cumulative separately debited charges."""

        self._require_usable()
        return self._ledger.total_charges

    def value_portfolio(
        self,
        session: date,
        prices: Mapping[str, float],
    ) -> LedgerSnapshot:
        """Return an immutable valuation without mutating authoritative state."""

        self._require_usable()
        return self._ledger.snapshot(session, prices)

    def target_orders(
        self,
        target_weights: Mapping[str, float],
        reference_prices: Mapping[str, float],
    ) -> Mapping[str, float]:
        """Return detached target quantities from one authoritative pretrade NAV."""

        self._require_usable()
        return MappingProxyType(self._ledger.target_orders(target_weights, reference_prices))

    def snapshot(self) -> EngineSnapshot:
        """Return an immutable view of reducer state."""
        return EngineSnapshot(
            run_id=self._run_id,
            processed_events=len(self._fingerprints),
            pending_events=len(self._pending),
            orders=MappingProxyType(dict(self._orders)),
            portfolio=self._last_portfolio,
            halted=self._halted,
            halt_reason=self._halt_reason,
            recovery_required=self._recovery_required,
        )

    def submit(self, event: ExecutionEvent) -> bool:
        """Queue one event and return ``False`` for an exact duplicate."""
        self._require_usable()
        self._require_event(event)
        fingerprint = _fingerprint(event)
        existing = self._fingerprints.get(event.event_id)
        if existing is not None:
            if existing != fingerprint:
                raise EventTransitionError("event_id was reused with different content")
            return False
        pending = self._pending_fingerprints.get(event.event_id)
        if pending is not None:
            if pending != fingerprint:
                raise EventTransitionError("queued event_id was reused with different content")
            return False
        key = _event_sort_key(event)
        if self._cursor is not None and key <= self._cursor:
            raise EventOrderingError("event attempts to enter at or before the committed cursor")
        if len(self._pending) >= self._max_pending:
            raise EventResourceLimitError("pending-event limit exceeded")
        if len(self._fingerprints) + len(self._pending) >= self._max_events:
            raise EventResourceLimitError("event limit exceeded")
        heapq.heappush(self._pending, (key, event))
        self._pending_fingerprints[event.event_id] = fingerprint
        return True

    def drain(self, *, maximum: int | None = None) -> tuple[ExecutionEvent, ...]:
        """Process queued events in canonical order."""
        self._require_usable()
        if maximum is not None:
            maximum = _bounded_positive_int(maximum, name="maximum")
        processed: list[ExecutionEvent] = []
        while self._pending and (maximum is None or len(processed) < maximum):
            _, event = heapq.heappop(self._pending)
            self._pending_fingerprints.pop(event.event_id, None)
            if self._process(event, append=True):
                processed.append(event)
        return tuple(processed)

    def process(self, event: ExecutionEvent) -> bool:
        """Process one already ordered event without queueing it."""
        self._require_usable()
        self._require_event(event)
        return self._process(event, append=True)

    def close(self) -> None:
        """Close only an internally owned journal."""
        if self._owns_journal:
            self._journal.close()

    def _require_usable(self) -> None:
        if self._recovery_required:
            raise EngineRecoveryRequiredError(
                "journal outcome is uncertain; discard the engine and replay its journal"
            )

    def _require_event(self, event: object) -> None:
        if not isinstance(event, ExecutionEvent):
            raise TypeError("event must be an ExecutionEvent")
        if event.run_id != self._run_id:
            raise EventTransitionError("event run_id does not match the engine")
        expected_index = self._calendar_index.get(event.coordinate.session)
        if expected_index is None:
            raise EventOrderingError("event session is absent from the frozen calendar")
        if event.coordinate.bar_index != expected_index:
            raise EventOrderingError("event bar_index does not match its frozen-calendar session")

    def _process(self, event: ExecutionEvent, *, append: bool) -> bool:
        self._require_event(event)
        fingerprint = _fingerprint(event)
        existing = self._fingerprints.get(event.event_id)
        if existing is not None:
            if existing != fingerprint:
                raise EventTransitionError("event_id was reused with different content")
            return False
        if len(self._fingerprints) >= self._max_events:
            raise EventResourceLimitError("event limit exceeded")
        key = _event_sort_key(event)
        if self._cursor is not None and key <= self._cursor:
            raise EventOrderingError("event attempts to enter at or before the committed cursor")
        if self._halted:
            raise EventTransitionError("engine is halted and rejects new events")
        if self._halt_pending is not None and not isinstance(event.payload, EngineHalted):
            raise EventTransitionError("a required engine halt must be recorded before more events")

        parent = self._validate_causation(event, key)
        if self._halt_pending is not None and event.causation_id != self._halt_cause_event_id:
            raise EventCausationError(
                "a required engine halt must cite the exact causal accounting event"
            )
        identity_key = self._identity_key(event)
        if identity_key is not None and identity_key in self._identity_origins:
            raise EventTransitionError(f"{identity_key[0]} identity is already in use")

        # Accounting mutations are evaluated on a detached candidate. Raw
        # ledger errors are translated at this trust boundary so replay never
        # leaks implementation-specific exceptions.
        try:
            transition = self._plan_transition(event, parent)
        except EventEngineError:
            raise
        except (ArithmeticError, RuntimeError, ValueError) as exc:
            raise EventTransitionError(
                f"{event.event_type} failed portfolio accounting validation"
            ) from exc
        if append:
            try:
                inserted = self._journal.append(event)
            except JournalError as exc:
                self._recovery_required = True
                raise EngineRecoveryRequiredError(
                    "journal append failed; in-memory state is non-publishable"
                ) from exc
            if not inserted:
                self._recovery_required = True
                raise EngineRecoveryRequiredError(
                    "journal already contained an event absent from reducer state"
                )

        if transition.order_update is not None:
            previous_order = self._orders.get(transition.order_update.order_id)
            if previous_order is None:
                self._open_order_count += 1
            elif not previous_order.status.terminal and transition.order_update.status.terminal:
                self._open_order_count -= 1
            self._orders[transition.order_update.order_id] = transition.order_update
        if transition.ledger_update is not None:
            self._ledger = transition.ledger_update
        if transition.portfolio is not None:
            self._last_portfolio = transition.portfolio
        if isinstance(event.payload, PortfolioMarked):
            if event.payload.mark_type == "open":
                self._open_mark_bar = event.coordinate.bar_index
                self._open_prices = MappingProxyType(dict(event.payload.prices))
            else:
                self._close_mark_bar = event.coordinate.bar_index
        self._halt_pending = transition.halt_pending
        if (
            transition.halt_pending is not None
            and transition.portfolio is not None
            and transition.portfolio.bankrupt
        ):
            self._halt_cause_event_id = event.event_id
        elif transition.halted or transition.halt_pending is None:
            self._halt_cause_event_id = None
        if transition.halted:
            self._halted = True
            self._halt_reason = transition.halt_reason
            self._halt_pending = None
        if identity_key is not None:
            self._identity_origins[identity_key] = event.event_id
        self._references[event.event_id] = _EventReference(
            payload_type=type(event.payload),
            order_id=self._payload_order_id(event.payload),
            correlation_id=event.correlation_id,
            sort_key=key,
            session=event.coordinate.session,
            bar_index=event.coordinate.bar_index,
            eligible_session=(
                event.payload.eligible_session if isinstance(event.payload, TargetDecided) else None
            ),
            target_symbols=(
                frozenset(symbol for symbol, _ in event.payload.weights)
                if isinstance(event.payload, TargetDecided)
                else frozenset()
            ),
        )
        self._fingerprints[event.event_id] = fingerprint
        self._cursor = key
        return True

    def _validate_causation(
        self,
        event: ExecutionEvent,
        key: EventSortKey,
    ) -> _EventReference | None:
        payload = event.payload
        required: tuple[type[object], ...] | None
        if isinstance(payload, TargetDecided):
            required = (SignalAvailable,)
        elif isinstance(payload, OrderSubmitted):
            required = (TargetDecided,)
        elif isinstance(payload, (OrderAccepted, OrderRejected)):
            required = (OrderSubmitted,)
        elif isinstance(payload, FillApplied):
            required = (OrderAccepted, FillApplied)
        elif isinstance(payload, OrderCancelled):
            required = (OrderSubmitted, OrderAccepted, FillApplied)
        else:
            required = None

        cause_id = event.causation_id
        if required is not None and cause_id is None:
            raise EventCausationError(f"{event.event_type} requires a causation_id")
        if cause_id is None:
            return None
        parent = self._references.get(cause_id)
        if parent is None:
            raise EventCausationError("causation_id does not reference an earlier event")
        if parent.sort_key >= key:
            raise EventCausationError("causation must precede the caused event")
        if required is not None and parent.payload_type not in required:
            raise EventCausationError(
                f"{event.event_type} cannot be caused by {parent.payload_type.__name__}"
            )
        if parent.correlation_id != event.correlation_id:
            raise EventCausationError("caused events must retain correlation_id")
        if isinstance(payload, OrderSubmitted):
            if parent.eligible_session != event.coordinate.session:
                raise EventCausationError(
                    "order submission must occur on the target eligible_session"
                )
            if payload.symbol not in parent.target_symbols:
                raise EventCausationError("order symbol is absent from the causing target book")
        if isinstance(
            payload,
            (OrderAccepted, OrderRejected, FillApplied, OrderCancelled),
        ) and (
            parent.session != event.coordinate.session
            or parent.bar_index != event.coordinate.bar_index
        ):
            raise EventCausationError("DAY-order transitions must remain on the submission bar")
        if (
            isinstance(payload, FillApplied)
            and parent.payload_type in {OrderAccepted, FillApplied}
            and parent.sort_key[2] == key[2]
            and parent.sort_key[3] >= key[3]
        ):
            raise EventCausationError(
                "fill ordinals must increase explicitly within the execution phase"
            )
        return parent

    def _plan_transition(
        self,
        event: ExecutionEvent,
        parent: _EventReference | None,
    ) -> _PlannedTransition:
        payload = event.payload
        if isinstance(payload, OrderSubmitted):
            if len(self._orders) >= self._max_orders:
                raise EventResourceLimitError("order limit exceeded")
            if self._open_order_count >= self._max_open_orders:
                raise EventResourceLimitError("open-order limit exceeded")
            if payload.order_id in self._orders:
                raise EventTransitionError("order_id is already in use")
            if self._open_mark_bar != event.coordinate.bar_index:
                raise EventTransitionError("order submission requires the current open mark")
            if payload.symbol not in self._open_prices:
                raise EventTransitionError(
                    "order symbol is missing from the current open price snapshot"
                )
            return _PlannedTransition(
                order_update=OrderState(
                    order_id=payload.order_id,
                    symbol=payload.symbol,
                    side=payload.side,
                    requested_quantity=payload.quantity,
                    accepted_quantity=0.0,
                    filled_quantity=0.0,
                    cancelled_quantity=0.0,
                    status=OrderLifecycle.SUBMITTED,
                    correlation_id=event.correlation_id,
                    submitted_session=event.coordinate.session,
                    submitted_bar_index=event.coordinate.bar_index,
                ),
                portfolio=self._last_portfolio,
                halt_pending=self._halt_pending,
            )
        if isinstance(payload, OrderAccepted):
            order = self._require_order(payload.order_id, parent, event)
            if order.status is not OrderLifecycle.SUBMITTED:
                raise EventTransitionError("only a submitted order can be accepted")
            if payload.accepted_quantity != order.requested_quantity:
                raise EventTransitionError(
                    "accepted quantity must equal the requested DAY-order quantity"
                )
            return _PlannedTransition(
                order_update=replace(
                    order,
                    accepted_quantity=payload.accepted_quantity,
                    status=OrderLifecycle.ACCEPTED,
                ),
                portfolio=self._last_portfolio,
                halt_pending=self._halt_pending,
            )
        if isinstance(payload, OrderRejected):
            order = self._require_order(payload.order_id, parent, event)
            if order.status is not OrderLifecycle.SUBMITTED:
                raise EventTransitionError("only a submitted order can be rejected")
            return _PlannedTransition(
                order_update=replace(order, status=OrderLifecycle.REJECTED),
                portfolio=self._last_portfolio,
                halt_pending=self._halt_pending,
            )
        if isinstance(payload, FillApplied):
            order = self._require_order(payload.order_id, parent, event)
            if order.status not in {OrderLifecycle.ACCEPTED, OrderLifecycle.PARTIALLY_FILLED}:
                raise EventTransitionError("fills require an accepted open order")
            if payload.symbol != order.symbol or payload.side != order.side:
                raise EventTransitionError("fill symbol and side must match the order")
            reference_price = self._open_prices.get(payload.symbol)
            if reference_price is None or not reconciles(
                payload.reference_price,
                reference_price,
                operands=(payload.reference_price, reference_price),
            ):
                raise EventTransitionError(
                    "fill reference price does not match the current open mark"
                )
            new_filled = math.fsum((order.filled_quantity, payload.quantity))
            if new_filled > order.accepted_quantity:
                if not _is_single_addition_roundoff_overrun(
                    new_filled,
                    order.accepted_quantity,
                ):
                    raise EventTransitionError("fill quantity exceeds accepted order residual")
                # Canonicalize a tiny addition-roundoff overrun. Positive
                # underfills are never rounded up, so hard caps remain hard.
                new_filled = order.accepted_quantity
            signed_quantity = payload.quantity if payload.side == "buy" else -payload.quantity
            candidate = self._ledger.clone()
            candidate.apply_fill(
                payload.symbol,
                signed_quantity,
                payload.price,
                charges={"fees": payload.total_fees},
            )
            status = (
                OrderLifecycle.FILLED
                if new_filled == order.accepted_quantity
                else OrderLifecycle.PARTIALLY_FILLED
            )
            portfolio = candidate.snapshot(
                event.coordinate.session,
                dict(self._open_prices),
            )
            return _PlannedTransition(
                order_update=replace(order, filled_quantity=new_filled, status=status),
                ledger_update=candidate,
                portfolio=portfolio,
                halt_pending="bankruptcy" if portfolio.bankrupt else self._halt_pending,
            )
        if isinstance(payload, OrderCancelled):
            order = self._require_order(payload.order_id, parent, event)
            if order.status.terminal:
                raise EventTransitionError("terminal orders cannot be cancelled")
            expected = order.requested_quantity - order.filled_quantity
            if payload.cancelled_quantity != expected:
                raise EventTransitionError("cancelled quantity does not match order residual")
            return _PlannedTransition(
                order_update=replace(
                    order,
                    cancelled_quantity=payload.cancelled_quantity,
                    status=OrderLifecycle.CANCELLED,
                ),
                portfolio=self._last_portfolio,
                halt_pending=self._halt_pending,
            )
        if isinstance(payload, CashChargeAccrued):
            if self._open_mark_bar != event.coordinate.bar_index:
                raise EventTransitionError("cash charges require the current open mark")
            if self._close_mark_bar == event.coordinate.bar_index:
                raise EventTransitionError("cash charges cannot follow the close mark")
            if self._open_order_count != 0:
                raise EventTransitionError("all DAY orders must be terminal before cash charges")
            candidate = self._ledger.clone()
            candidate.apply_charge(payload.charge_type, payload.amount)
            portfolio = candidate.snapshot(
                event.coordinate.session,
                dict(self._open_prices),
            )
            return _PlannedTransition(
                ledger_update=candidate,
                portfolio=portfolio,
                halt_pending="bankruptcy" if portfolio.bankrupt else self._halt_pending,
            )
        if isinstance(payload, PortfolioMarked):
            if payload.mark_type == "open":
                if self._open_order_count != 0:
                    raise EventTransitionError(
                        "a new open mark cannot advance past outstanding DAY orders"
                    )
                if self._open_mark_bar == event.coordinate.bar_index:
                    raise EventTransitionError("a bar can contain only one open mark")
            else:
                if self._open_mark_bar != event.coordinate.bar_index:
                    raise EventTransitionError("close mark requires the current open mark")
                if self._close_mark_bar == event.coordinate.bar_index:
                    raise EventTransitionError("a bar can contain only one close mark")
                if self._open_order_count != 0:
                    raise EventTransitionError(
                        "all DAY orders must be terminal before the close mark"
                    )
            portfolio = self._ledger.snapshot(
                event.coordinate.session,
                dict(payload.prices),
            )
            holdings = math.fsum(portfolio.market_values.values())
            comparisons = (
                (portfolio.cash, payload.cash, "cash", (portfolio.cash, payload.cash)),
                (
                    holdings,
                    payload.holdings_value,
                    "holdings value",
                    (*portfolio.market_values.values(), holdings, payload.holdings_value),
                ),
                (
                    portfolio.total_charges,
                    payload.accrued_charges,
                    "accrued charges",
                    (*portfolio.charges.values(), payload.accrued_charges),
                ),
                (
                    portfolio.equity,
                    payload.equity,
                    "equity",
                    (portfolio.cash, holdings, portfolio.equity, payload.equity),
                ),
            )
            for actual, declared, name, operands in comparisons:
                if not reconciles(actual, declared, operands=operands):
                    raise EventTransitionError(f"portfolio mark {name} does not reconcile")
            return _PlannedTransition(
                portfolio=portfolio,
                halt_pending="bankruptcy" if portfolio.bankrupt else self._halt_pending,
            )
        if isinstance(payload, EngineHalted):
            if payload.reason_code == "bankruptcy" and self._halt_pending != "bankruptcy":
                raise EventTransitionError(
                    "bankruptcy halt requires a preceding non-positive accounting event"
                )
            if self._halt_pending is not None and payload.reason_code != self._halt_pending:
                raise EventTransitionError("halt reason does not match the required control halt")
            return _PlannedTransition(
                portfolio=self._last_portfolio,
                halted=True,
                halt_reason=payload.reason_code,
            )
        # Signal and target events are immutable lineage/control records.  Their
        # payload constructors and causation checks carry the semantic work.
        if isinstance(payload, (SignalAvailable, TargetDecided)):
            if self._close_mark_bar != event.coordinate.bar_index:
                raise EventTransitionError(
                    "signal and target events require the current close mark"
                )
            return _PlannedTransition(
                portfolio=self._last_portfolio,
                halt_pending=self._halt_pending,
            )
        raise EventTransitionError(f"unsupported payload type: {type(payload).__name__}")

    def _require_order(
        self,
        order_id: str,
        parent: _EventReference | None,
        event: ExecutionEvent,
    ) -> OrderState:
        order = self._orders.get(order_id)
        if order is None:
            raise EventTransitionError("event references an unknown order")
        if parent is not None and parent.order_id != order_id:
            raise EventCausationError("causation references a different order")
        if (
            event.coordinate.session != order.submitted_session
            or event.coordinate.bar_index != order.submitted_bar_index
        ):
            raise EventTransitionError("DAY-order transition escaped its submission bar")
        return order

    @staticmethod
    def _payload_order_id(payload: object) -> str | None:
        if isinstance(
            payload,
            (OrderSubmitted, OrderAccepted, OrderRejected, FillApplied, OrderCancelled),
        ):
            return payload.order_id
        return None

    @staticmethod
    def _identity_key(event: ExecutionEvent) -> tuple[str, str] | None:
        payload = event.payload
        if isinstance(payload, SignalAvailable):
            return ("signal", payload.signal_id)
        if isinstance(payload, TargetDecided):
            return ("target", payload.target_id)
        if isinstance(payload, FillApplied):
            return ("fill", payload.fill_id)
        if isinstance(payload, CashChargeAccrued):
            return ("charge", payload.charge_id)
        if isinstance(payload, PortfolioMarked):
            return ("mark", payload.mark_id)
        return None
