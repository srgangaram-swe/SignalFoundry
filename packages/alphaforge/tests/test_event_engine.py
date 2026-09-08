"""State-machine, accounting, replay, and failure tests for the MR3 reducer."""

from __future__ import annotations

from datetime import date
from typing import Any

import pytest

from alphaforge.backtesting.event_engine import (
    DeterministicEventEngine,
    EngineRecoveryRequiredError,
    EventCausationError,
    EventOrderingError,
    EventReplayError,
    EventResourceLimitError,
    EventTransitionError,
    OrderLifecycle,
)
from alphaforge.backtesting.journal import InMemoryJournal, JournalIntegrityError
from alphaforge.execution.events import (
    CashChargeAccrued,
    ChargeType,
    EngineHalted,
    EventCoordinate,
    EventPhase,
    ExecutionEvent,
    FeeCategory,
    FeeComponent,
    FillApplied,
    OrderAccepted,
    OrderCancelled,
    OrderSubmitted,
    PortfolioMarked,
    SignalAvailable,
    TargetDecided,
)

_DIGEST = "a" * 64
_RUN_ID = "synthetic-run-001"
_CALENDAR = tuple(date(2025, 1, day) for day in range(2, 6))


def _event(
    payload: Any,
    *,
    session: date,
    bar_index: int,
    phase: EventPhase,
    ordinal: int = 0,
    correlation_id: str = "target-chain-001",
    entity_id: str = "entity-001",
    causation_id: str | None = None,
) -> ExecutionEvent:
    return ExecutionEvent(
        run_id=_RUN_ID,
        correlation_id=correlation_id,
        entity_id=entity_id,
        coordinate=EventCoordinate(
            session=session,
            bar_index=bar_index,
            phase=phase,
            ordinal=ordinal,
        ),
        payload=payload,
        causation_id=causation_id,
    )


def _mark(
    *,
    mark_id: str,
    mark_type: str,
    session: date,
    bar_index: int,
    price: float,
    cash: float,
    holdings: float,
    charges: float,
    equity: float,
) -> ExecutionEvent:
    phase = EventPhase.OPEN_MARK if mark_type == "open" else EventPhase.CLOSE_MARK
    return _event(
        PortfolioMarked(
            mark_id=mark_id,
            mark_type=mark_type,  # type: ignore[arg-type]
            prices=(("A", price),),
            cash=cash,
            holdings_value=holdings,
            accrued_charges=charges,
            equity=equity,
        ),
        session=session,
        bar_index=bar_index,
        phase=phase,
        correlation_id="mark-chain",
        entity_id=mark_id,
    )


def _partial_fill_flow() -> tuple[ExecutionEvent, ...]:
    first = date(2025, 1, 2)
    second = date(2025, 1, 3)
    first_open = _mark(
        mark_id="mark-open-000",
        mark_type="open",
        session=first,
        bar_index=0,
        price=100.0,
        cash=1_000.0,
        holdings=0.0,
        charges=0.0,
        equity=1_000.0,
    )
    first_close = _mark(
        mark_id="mark-close-000",
        mark_type="close",
        session=first,
        bar_index=0,
        price=100.0,
        cash=1_000.0,
        holdings=0.0,
        charges=0.0,
        equity=1_000.0,
    )
    signal = _event(
        SignalAvailable(
            signal_id="signal-001",
            model_id="synthetic-noncandidate",
            signal_digest=_DIGEST,
        ),
        session=first,
        bar_index=0,
        phase=EventPhase.SIGNAL,
        entity_id="signal-001",
    )
    target = _event(
        TargetDecided(
            target_id="target-001",
            portfolio_id="synthetic-targets",
            solver_id="no-solver",
            eligible_session=second,
            cash_weight=0.6,
            weights=(("A", 0.4),),
            configuration_digest=_DIGEST,
            data_digest="b" * 64,
            problem_digest="c" * 64,
        ),
        session=first,
        bar_index=0,
        phase=EventPhase.TARGET_DECISION,
        entity_id="target-001",
        causation_id=signal.event_id,
    )
    second_open = _mark(
        mark_id="mark-open-001",
        mark_type="open",
        session=second,
        bar_index=1,
        price=100.0,
        cash=1_000.0,
        holdings=0.0,
        charges=0.0,
        equity=1_000.0,
    )
    submitted = _event(
        OrderSubmitted(
            order_id="order-001",
            symbol="A",
            side="buy",
            quantity=10.0,
        ),
        session=second,
        bar_index=1,
        phase=EventPhase.ORDER_SUBMISSION,
        entity_id="order-001",
        causation_id=target.event_id,
    )
    accepted = _event(
        OrderAccepted(order_id="order-001", accepted_quantity=10.0),
        session=second,
        bar_index=1,
        phase=EventPhase.EXECUTION,
        ordinal=0,
        entity_id="order-001",
        causation_id=submitted.event_id,
    )
    fill = _event(
        FillApplied(
            fill_id="fill-001",
            order_id="order-001",
            symbol="A",
            side="buy",
            quantity=4.0,
            reference_price=100.0,
            price=100.0,
            fees=(FeeComponent(FeeCategory.COMMISSION, 2.0),),
        ),
        session=second,
        bar_index=1,
        phase=EventPhase.EXECUTION,
        ordinal=1,
        entity_id="fill-001",
        causation_id=accepted.event_id,
    )
    cancelled = _event(
        OrderCancelled(
            order_id="order-001",
            reason_code="day_expired",
            cancelled_quantity=6.0,
        ),
        session=second,
        bar_index=1,
        phase=EventPhase.DAY_CANCEL,
        entity_id="order-001",
        causation_id=fill.event_id,
    )
    second_close = _mark(
        mark_id="mark-close-001",
        mark_type="close",
        session=second,
        bar_index=1,
        price=101.0,
        cash=598.0,
        holdings=404.0,
        charges=2.0,
        equity=1_002.0,
    )
    return (
        first_open,
        first_close,
        signal,
        target,
        second_open,
        submitted,
        accepted,
        fill,
        cancelled,
        second_close,
    )


def test_permuted_queue_reduces_to_reconciled_partial_fill_state() -> None:
    events = _partial_fill_flow()
    journal = InMemoryJournal()
    engine = DeterministicEventEngine(
        _RUN_ID,
        calendar=_CALENDAR,
        initial_cash=1_000.0,
        journal=journal,
    )

    for event in reversed(events):
        assert engine.submit(event)
    processed = engine.drain()

    assert processed == events
    state = engine.snapshot()
    order = state.orders["order-001"]
    assert order.status is OrderLifecycle.CANCELLED
    assert order.filled_quantity == pytest.approx(4.0)
    assert order.cancelled_quantity == pytest.approx(6.0)
    assert order.residual_quantity == 0.0
    assert state.portfolio is not None
    assert state.portfolio.cash == pytest.approx(598.0)
    assert state.portfolio.equity == pytest.approx(1_002.0)
    assert state.portfolio.unrealized_pnl == pytest.approx(4.0)
    assert state.portfolio.total_charges == pytest.approx(2.0)
    assert state.portfolio.reconciliation_error <= state.portfolio.reconciliation_tolerance
    assert journal.count == len(events)


def test_cash_pnl_and_exposure_reconcile_after_every_processed_event() -> None:
    engine = DeterministicEventEngine(_RUN_ID, calendar=_CALENDAR, initial_cash=1_000.0)

    for event in _partial_fill_flow():
        engine.process(event)
        portfolio = engine.snapshot().portfolio
        assert portfolio is not None
        expected_net = sum(portfolio.market_values.values())
        expected_gross = sum(abs(value) for value in portfolio.market_values.values())
        assert portfolio.net_exposure == pytest.approx(expected_net)
        assert portfolio.gross_exposure == pytest.approx(expected_gross)
        assert portfolio.equity == pytest.approx(portfolio.cash + expected_net)
        assert portfolio.net_pnl == pytest.approx(
            portfolio.realized_pnl + portfolio.unrealized_pnl - portfolio.total_charges
        )
        assert portfolio.reconciliation_error <= portfolio.reconciliation_tolerance


def test_exact_duplicate_is_idempotent_but_time_travel_is_rejected() -> None:
    engine = DeterministicEventEngine(_RUN_ID, calendar=_CALENDAR, initial_cash=1_000.0)
    events = _partial_fill_flow()
    for event in events:
        assert engine.process(event)
    assert not engine.process(events[-1])
    assert engine.snapshot().processed_events == len(events)
    earlier_control = _event(
        EngineHalted(reason_code="operator_stop", detail="Synthetic control event."),
        session=date(2025, 1, 2),
        bar_index=0,
        phase=EventPhase.CONTROL,
        correlation_id="control-chain",
        entity_id="earlier-control",
    )
    with pytest.raises(EventOrderingError, match="committed cursor"):
        engine.process(earlier_control)


def test_orphan_and_wrong_order_causation_fail_before_journaling() -> None:
    engine = DeterministicEventEngine(_RUN_ID, calendar=_CALENDAR, initial_cash=1_000.0)
    events = _partial_fill_flow()
    for event in events[:6]:
        engine.process(event)
    orphan = _event(
        OrderAccepted(order_id="order-001", accepted_quantity=10.0),
        session=date(2025, 1, 3),
        bar_index=1,
        phase=EventPhase.EXECUTION,
        ordinal=2,
        entity_id="order-001",
        causation_id="d" * 64,
    )
    with pytest.raises(EventCausationError, match="earlier event"):
        engine.process(orphan)
    assert engine.journal.count == 6


def test_day_order_refuses_next_session_fill() -> None:
    engine = DeterministicEventEngine(_RUN_ID, calendar=_CALENDAR, initial_cash=1_000.0)
    events = _partial_fill_flow()
    for event in events[:-1]:
        engine.process(event)
    late = _event(
        FillApplied(
            fill_id="fill-late",
            order_id="order-001",
            symbol="A",
            side="buy",
            quantity=1.0,
            reference_price=100.0,
            price=100.0,
        ),
        session=date(2025, 1, 4),
        bar_index=2,
        phase=EventPhase.EXECUTION,
        entity_id="fill-late",
        causation_id=events[-3].event_id,
    )
    with pytest.raises(EventCausationError, match="submission bar"):
        engine.process(late)


def test_charge_cannot_cross_day_order_cancellation_boundary() -> None:
    engine = DeterministicEventEngine(_RUN_ID, calendar=_CALENDAR, initial_cash=1_000.0)
    events = _partial_fill_flow()
    for event in events[:8]:
        engine.process(event)
    charge = _event(
        CashChargeAccrued(
            charge_id="premature-charge",
            charge_type="other",
            amount=1.0,
        ),
        session=date(2025, 1, 3),
        bar_index=1,
        phase=EventPhase.CHARGE,
        entity_id="premature-charge",
    )

    with pytest.raises(EventTransitionError, match="terminal before cash charges"):
        engine.process(charge)
    assert engine.journal.count == 8
    assert engine.snapshot().orders["order-001"].status is OrderLifecycle.PARTIALLY_FILLED


def test_acceptance_and_cancellation_quantities_are_exact_hard_bounds() -> None:
    engine = DeterministicEventEngine(_RUN_ID, calendar=_CALENDAR, initial_cash=1_000.0)
    base = _partial_fill_flow()
    for event in base[:5]:
        engine.process(event)
    submitted = _event(
        OrderSubmitted(
            order_id="bounded-order",
            symbol="A",
            side="buy",
            quantity=1_000_000_000_000_000.0,
        ),
        session=date(2025, 1, 3),
        bar_index=1,
        phase=EventPhase.ORDER_SUBMISSION,
        entity_id="bounded-order",
        causation_id=base[3].event_id,
    )
    engine.process(submitted)
    understated_acceptance = _event(
        OrderAccepted(
            order_id="bounded-order",
            accepted_quantity=999_999_999_999_999.0,
        ),
        session=date(2025, 1, 3),
        bar_index=1,
        phase=EventPhase.EXECUTION,
        ordinal=0,
        entity_id="bounded-order",
        causation_id=submitted.event_id,
    )
    with pytest.raises(EventTransitionError, match="must equal the requested"):
        engine.process(understated_acceptance)

    accepted = _event(
        OrderAccepted(
            order_id="bounded-order",
            accepted_quantity=1_000_000_000_000_000.0,
        ),
        session=date(2025, 1, 3),
        bar_index=1,
        phase=EventPhase.EXECUTION,
        ordinal=0,
        entity_id="bounded-order",
        causation_id=submitted.event_id,
    )
    engine.process(accepted)
    understated_cancellation = _event(
        OrderCancelled(
            order_id="bounded-order",
            reason_code="day_expired",
            cancelled_quantity=999_999_999_999_999.0,
        ),
        session=date(2025, 1, 3),
        bar_index=1,
        phase=EventPhase.DAY_CANCEL,
        entity_id="bounded-order",
        causation_id=accepted.event_id,
    )
    with pytest.raises(EventTransitionError, match="does not match order residual"):
        engine.process(understated_cancellation)

    state = engine.snapshot().orders["bounded-order"]
    assert state.status is OrderLifecycle.ACCEPTED
    assert state.accepted_quantity == 1_000_000_000_000_000.0
    assert state.residual_quantity == 1_000_000_000_000_000.0


def test_next_open_cannot_advance_past_expired_day_order() -> None:
    engine = DeterministicEventEngine(_RUN_ID, calendar=_CALENDAR, initial_cash=1_000.0)
    events = _partial_fill_flow()
    for event in events[:7]:
        engine.process(event)
    next_open = _mark(
        mark_id="mark-open-002",
        mark_type="open",
        session=date(2025, 1, 4),
        bar_index=2,
        price=100.0,
        cash=1_000.0,
        holdings=0.0,
        charges=0.0,
        equity=1_000.0,
    )

    with pytest.raises(EventTransitionError, match="outstanding DAY orders"):
        engine.process(next_open)
    assert engine.journal.count == 7
    assert engine.snapshot().orders["order-001"].status is OrderLifecycle.ACCEPTED


def test_replay_reconstructs_identical_state_and_canonical_history() -> None:
    journal = InMemoryJournal()
    engine = DeterministicEventEngine(
        _RUN_ID,
        calendar=_CALENDAR,
        initial_cash=1_000.0,
        journal=journal,
    )
    for event in _partial_fill_flow():
        engine.process(event)
    canonical = journal.export_canonical()

    replayed = DeterministicEventEngine.replay(
        journal,
        calendar=_CALENDAR,
        initial_cash=1_000.0,
        expected_run_id=_RUN_ID,
    )

    assert replayed.snapshot() == engine.snapshot()
    assert replayed.journal.export_canonical() == canonical


def test_forged_portfolio_mark_fails_without_a_journal_row() -> None:
    engine = DeterministicEventEngine(_RUN_ID, calendar=_CALENDAR, initial_cash=1_000.0)
    forged = _mark(
        mark_id="forged-mark",
        mark_type="open",
        session=date(2025, 1, 2),
        bar_index=0,
        price=100.0,
        cash=1_000.0,
        holdings=0.0,
        charges=0.0,
        equity=1_000.0,
    )
    # The payload is internally coherent; the reducer independently rejects a
    # forged cash diagnostic against its own ledger.
    object.__setattr__(forged.payload, "cash", 999.0)
    object.__setattr__(forged.payload, "equity", 999.0)
    with pytest.raises(EventTransitionError, match="cash does not reconcile"):
        engine.process(forged)
    assert engine.journal.count == 0


def test_bankruptcy_requires_explicit_terminal_halt() -> None:
    engine = DeterministicEventEngine(_RUN_ID, calendar=_CALENDAR, initial_cash=100.0)
    session = date(2025, 1, 2)
    opened = _mark(
        mark_id="bankrupt-open",
        mark_type="open",
        session=session,
        bar_index=0,
        price=100.0,
        cash=100.0,
        holdings=0.0,
        charges=0.0,
        equity=100.0,
    )
    charge = _event(
        CashChargeAccrued(
            charge_id="charge-001",
            charge_type="other",
            amount=150.0,
        ),
        session=session,
        bar_index=0,
        phase=EventPhase.CHARGE,
        correlation_id="mark-chain",
        entity_id="charge-001",
    )
    engine.process(opened)
    engine.process(charge)
    with pytest.raises(EventTransitionError, match="required engine halt"):
        engine.process(
            _event(
                SignalAvailable(
                    signal_id="late-signal",
                    model_id="synthetic-noncandidate",
                    signal_digest=_DIGEST,
                ),
                session=session,
                bar_index=0,
                phase=EventPhase.SIGNAL,
                entity_id="late-signal",
            )
        )
    for bad_cause, entity_id in (
        (None, "engine-control-missing-cause"),
        (opened.event_id, "engine-control-wrong-cause"),
    ):
        invalid_halt = _event(
            EngineHalted(
                reason_code="bankruptcy",
                detail="Marked equity is non-positive.",
            ),
            session=session,
            bar_index=0,
            phase=EventPhase.CONTROL,
            correlation_id="mark-chain",
            entity_id=entity_id,
            causation_id=bad_cause,
        )
        with pytest.raises(EventCausationError, match="exact causal accounting event"):
            engine.process(invalid_halt)
    assert engine.journal.count == 2
    halt = _event(
        EngineHalted(
            reason_code="bankruptcy",
            detail="Marked equity is non-positive.",
        ),
        session=session,
        bar_index=0,
        phase=EventPhase.CONTROL,
        correlation_id="mark-chain",
        entity_id="engine-control",
        causation_id=charge.event_id,
    )
    engine.process(halt)
    assert engine.snapshot().halted
    assert engine.snapshot().halt_reason == "bankruptcy"


def test_bankruptcy_reason_is_reserved_for_a_non_positive_accounting_event() -> None:
    engine = DeterministicEventEngine(_RUN_ID, calendar=_CALENDAR, initial_cash=100.0)
    session = date(2025, 1, 2)
    engine.process(
        _mark(
            mark_id="healthy-open",
            mark_type="open",
            session=session,
            bar_index=0,
            price=100.0,
            cash=100.0,
            holdings=0.0,
            charges=0.0,
            equity=100.0,
        )
    )
    false_halt = _event(
        EngineHalted(
            reason_code="bankruptcy",
            detail="Unsubstantiated bankruptcy assertion.",
        ),
        session=session,
        bar_index=0,
        phase=EventPhase.CONTROL,
        correlation_id="mark-chain",
        entity_id="false-bankruptcy-halt",
    )

    with pytest.raises(EventTransitionError, match="non-positive accounting event"):
        engine.process(false_halt)
    assert engine.journal.count == 1
    assert not engine.snapshot().halted


def test_fill_insolvency_requires_exactly_caused_halt_before_other_orders() -> None:
    engine = DeterministicEventEngine(_RUN_ID, calendar=_CALENDAR, initial_cash=1_000.0)
    base = _partial_fill_flow()
    for event in base[:5]:
        engine.process(event)
    first_submitted = _event(
        OrderSubmitted(order_id="insolvent-order", symbol="A", side="buy", quantity=1.0),
        session=date(2025, 1, 3),
        bar_index=1,
        phase=EventPhase.ORDER_SUBMISSION,
        ordinal=0,
        entity_id="insolvent-order",
        causation_id=base[3].event_id,
    )
    second_submitted = _event(
        OrderSubmitted(order_id="blocked-order", symbol="A", side="buy", quantity=1.0),
        session=date(2025, 1, 3),
        bar_index=1,
        phase=EventPhase.ORDER_SUBMISSION,
        ordinal=1,
        entity_id="blocked-order",
        causation_id=base[3].event_id,
    )
    accepted = _event(
        OrderAccepted(order_id="insolvent-order", accepted_quantity=1.0),
        session=date(2025, 1, 3),
        bar_index=1,
        phase=EventPhase.EXECUTION,
        ordinal=0,
        entity_id="insolvent-order",
        causation_id=first_submitted.event_id,
    )
    insolvent_fill = _event(
        FillApplied(
            fill_id="insolvent-fill",
            order_id="insolvent-order",
            symbol="A",
            side="buy",
            quantity=1.0,
            reference_price=100.0,
            price=100.0,
            fees=(FeeComponent(FeeCategory.COMMISSION, 2_000.0),),
        ),
        session=date(2025, 1, 3),
        bar_index=1,
        phase=EventPhase.EXECUTION,
        ordinal=1,
        entity_id="insolvent-fill",
        causation_id=accepted.event_id,
    )
    blocked_acceptance = _event(
        OrderAccepted(order_id="blocked-order", accepted_quantity=1.0),
        session=date(2025, 1, 3),
        bar_index=1,
        phase=EventPhase.EXECUTION,
        ordinal=2,
        entity_id="blocked-order",
        causation_id=second_submitted.event_id,
    )

    for event in (first_submitted, second_submitted, accepted, insolvent_fill):
        engine.process(event)
    portfolio = engine.snapshot().portfolio
    assert portfolio is not None and portfolio.bankrupt
    with pytest.raises(EventTransitionError, match="required engine halt"):
        engine.process(blocked_acceptance)

    halt = _event(
        EngineHalted(reason_code="bankruptcy", detail="Fill made equity non-positive."),
        session=date(2025, 1, 3),
        bar_index=1,
        phase=EventPhase.CONTROL,
        correlation_id=insolvent_fill.correlation_id,
        entity_id="fill-bankruptcy-halt",
        causation_id=insolvent_fill.event_id,
    )
    engine.process(halt)
    assert engine.snapshot().halted
    assert engine.snapshot().halt_reason == "bankruptcy"


def test_fractional_split_fills_canonicalize_only_tiny_addition_overrun() -> None:
    engine = DeterministicEventEngine(_RUN_ID, calendar=_CALENDAR, initial_cash=1_000.0)
    base = _partial_fill_flow()
    for event in base[:5]:
        engine.process(event)
    submitted = _event(
        OrderSubmitted(order_id="fractional-order", symbol="A", side="buy", quantity=0.3),
        session=date(2025, 1, 3),
        bar_index=1,
        phase=EventPhase.ORDER_SUBMISSION,
        entity_id="fractional-order",
        causation_id=base[3].event_id,
    )
    accepted = _event(
        OrderAccepted(order_id="fractional-order", accepted_quantity=0.3),
        session=date(2025, 1, 3),
        bar_index=1,
        phase=EventPhase.EXECUTION,
        ordinal=0,
        entity_id="fractional-order",
        causation_id=submitted.event_id,
    )
    first = _event(
        FillApplied(
            fill_id="fractional-fill-1",
            order_id="fractional-order",
            symbol="A",
            side="buy",
            quantity=0.1,
            reference_price=100.0,
            price=100.0,
        ),
        session=date(2025, 1, 3),
        bar_index=1,
        phase=EventPhase.EXECUTION,
        ordinal=1,
        entity_id="fractional-fill-1",
        causation_id=accepted.event_id,
    )
    second = _event(
        FillApplied(
            fill_id="fractional-fill-2",
            order_id="fractional-order",
            symbol="A",
            side="buy",
            quantity=0.2,
            reference_price=100.0,
            price=100.0,
        ),
        session=date(2025, 1, 3),
        bar_index=1,
        phase=EventPhase.EXECUTION,
        ordinal=2,
        entity_id="fractional-fill-2",
        causation_id=first.event_id,
    )

    for event in (submitted, accepted, first, second):
        engine.process(event)

    order = engine.snapshot().orders["fractional-order"]
    assert order.status is OrderLifecycle.FILLED
    assert order.filled_quantity == 0.3
    assert order.residual_quantity == 0.0


def test_split_fill_cannot_hide_material_overfill_at_large_quantity() -> None:
    engine = DeterministicEventEngine(_RUN_ID, calendar=_CALENDAR, initial_cash=1_000.0)
    base = _partial_fill_flow()
    for event in base[:5]:
        engine.process(event)
    submitted = _event(
        OrderSubmitted(
            order_id="large-order",
            symbol="A",
            side="buy",
            quantity=1_000_000_000_000_000.0,
        ),
        session=date(2025, 1, 3),
        bar_index=1,
        phase=EventPhase.ORDER_SUBMISSION,
        entity_id="large-order",
        causation_id=base[3].event_id,
    )
    accepted = _event(
        OrderAccepted(
            order_id="large-order",
            accepted_quantity=1_000_000_000_000_000.0,
        ),
        session=date(2025, 1, 3),
        bar_index=1,
        phase=EventPhase.EXECUTION,
        ordinal=0,
        entity_id="large-order",
        causation_id=submitted.event_id,
    )
    first = _event(
        FillApplied(
            fill_id="large-fill-1",
            order_id="large-order",
            symbol="A",
            side="buy",
            quantity=999_999_999_999_999.0,
            reference_price=100.0,
            price=100.0,
        ),
        session=date(2025, 1, 3),
        bar_index=1,
        phase=EventPhase.EXECUTION,
        ordinal=1,
        entity_id="large-fill-1",
        causation_id=accepted.event_id,
    )
    overfill = _event(
        FillApplied(
            fill_id="large-fill-2",
            order_id="large-order",
            symbol="A",
            side="buy",
            quantity=2.0,
            reference_price=100.0,
            price=100.0,
        ),
        session=date(2025, 1, 3),
        bar_index=1,
        phase=EventPhase.EXECUTION,
        ordinal=2,
        entity_id="large-fill-2",
        causation_id=first.event_id,
    )

    for event in (submitted, accepted, first):
        engine.process(event)
    with pytest.raises(EventTransitionError, match="exceeds accepted order residual"):
        engine.process(overfill)

    order = engine.snapshot().orders["large-order"]
    assert order.status is OrderLifecycle.PARTIALLY_FILLED
    assert order.filled_quantity == 999_999_999_999_999.0
    assert engine.positions["A"] == 999_999_999_999_999.0


@pytest.mark.parametrize("charge_type", ("fees", "financing", "borrow", "other"))
def test_cash_charge_categories_reconcile_and_replay(charge_type: ChargeType) -> None:
    session = date(2025, 1, 2)
    journal = InMemoryJournal()
    engine = DeterministicEventEngine(
        _RUN_ID,
        calendar=_CALENDAR,
        initial_cash=1_000.0,
        journal=journal,
    )
    opened = _mark(
        mark_id="charge-open",
        mark_type="open",
        session=session,
        bar_index=0,
        price=100.0,
        cash=1_000.0,
        holdings=0.0,
        charges=0.0,
        equity=1_000.0,
    )
    charge = _event(
        CashChargeAccrued(
            charge_id=f"{charge_type}-charge",
            charge_type=charge_type,
            amount=10.0,
        ),
        session=session,
        bar_index=0,
        phase=EventPhase.CHARGE,
        correlation_id="charge-chain",
        entity_id=f"{charge_type}-charge",
    )
    closed = _mark(
        mark_id="charge-close",
        mark_type="close",
        session=session,
        bar_index=0,
        price=100.0,
        cash=990.0,
        holdings=0.0,
        charges=10.0,
        equity=990.0,
    )

    for event in (opened, charge, closed):
        engine.process(event)

    expected_charges = {
        category: 10.0 if category == charge_type else 0.0
        for category in ("fees", "financing", "borrow", "other")
    }
    portfolio = engine.snapshot().portfolio
    assert portfolio is not None
    assert portfolio.cash == pytest.approx(990.0)
    assert portfolio.equity == pytest.approx(990.0)
    assert portfolio.net_pnl == pytest.approx(-10.0)
    assert portfolio.charges == pytest.approx(expected_charges)

    replayed = DeterministicEventEngine.replay(
        journal,
        calendar=_CALENDAR,
        initial_cash=1_000.0,
        expected_run_id=_RUN_ID,
    )
    assert replayed.snapshot() == engine.snapshot()


class _FailOnFillJournal(InMemoryJournal):
    def append(self, event: ExecutionEvent) -> bool:
        if isinstance(event.payload, FillApplied):
            raise JournalIntegrityError("injected append failure")
        return super().append(event)


def test_journal_failure_poisons_mutated_engine_until_replay() -> None:
    journal = _FailOnFillJournal()
    engine = DeterministicEventEngine(
        _RUN_ID,
        calendar=_CALENDAR,
        initial_cash=1_000.0,
        journal=journal,
    )
    events = _partial_fill_flow()
    for event in events[:7]:
        engine.process(event)
    with pytest.raises(EngineRecoveryRequiredError, match="non-publishable"):
        engine.process(events[7])
    assert engine.snapshot().recovery_required
    with pytest.raises(EngineRecoveryRequiredError, match="discard"):
        engine.process(events[7])
    assert journal.count == 7


def test_pending_and_order_resource_limits_fail_before_growth() -> None:
    engine = DeterministicEventEngine(
        _RUN_ID,
        calendar=_CALENDAR,
        initial_cash=1_000.0,
        max_pending_events=1,
        max_orders=1,
        max_open_orders=1,
    )
    first = _partial_fill_flow()[0]
    assert engine.submit(first)
    with pytest.raises(EventResourceLimitError, match="pending-event"):
        engine.submit(_partial_fill_flow()[1])


def test_cross_run_event_is_rejected() -> None:
    event = _partial_fill_flow()[0]
    engine = DeterministicEventEngine(
        "other-run",
        calendar=_CALENDAR,
        initial_cash=1_000.0,
    )
    with pytest.raises(EventTransitionError, match="run_id"):
        engine.process(event)


def test_frozen_calendar_rejects_session_index_mismatch() -> None:
    engine = DeterministicEventEngine(_RUN_ID, calendar=_CALENDAR, initial_cash=1_000.0)
    mismatched = _event(
        PortfolioMarked(
            mark_id="mismatched-open",
            mark_type="open",
            prices=(),
            cash=1_000.0,
            holdings_value=0.0,
            accrued_charges=0.0,
            equity=1_000.0,
        ),
        session=_CALENDAR[0],
        bar_index=1,
        phase=EventPhase.OPEN_MARK,
        entity_id="mismatched-open",
    )

    with pytest.raises(EventOrderingError, match="frozen-calendar"):
        engine.process(mismatched)


def test_order_cannot_execute_before_target_eligible_session() -> None:
    engine = DeterministicEventEngine(_RUN_ID, calendar=_CALENDAR, initial_cash=1_000.0)
    base = _partial_fill_flow()
    for event in base[:3]:
        engine.process(event)
    delayed_target = _event(
        TargetDecided(
            target_id="target-delayed",
            portfolio_id="synthetic-targets",
            solver_id="no-solver",
            eligible_session=_CALENDAR[2],
            cash_weight=0.6,
            weights=(("A", 0.4),),
            configuration_digest=_DIGEST,
            data_digest="b" * 64,
            problem_digest="c" * 64,
        ),
        session=_CALENDAR[0],
        bar_index=0,
        phase=EventPhase.TARGET_DECISION,
        entity_id="target-delayed",
        causation_id=base[2].event_id,
    )
    engine.process(delayed_target)
    engine.process(base[4])
    early_order = _event(
        OrderSubmitted(order_id="order-early", symbol="A", side="buy", quantity=1.0),
        session=_CALENDAR[1],
        bar_index=1,
        phase=EventPhase.ORDER_SUBMISSION,
        entity_id="order-early",
        causation_id=delayed_target.event_id,
    )

    with pytest.raises(EventCausationError, match="eligible_session"):
        engine.process(early_order)


def test_fill_reference_price_must_match_current_open_mark() -> None:
    engine = DeterministicEventEngine(_RUN_ID, calendar=_CALENDAR, initial_cash=1_000.0)
    events = _partial_fill_flow()
    for event in events[:7]:
        engine.process(event)
    mismatched = _event(
        FillApplied(
            fill_id="fill-mismatched-reference",
            order_id="order-001",
            symbol="A",
            side="buy",
            quantity=1.0,
            reference_price=99.0,
            price=100.0,
        ),
        session=_CALENDAR[1],
        bar_index=1,
        phase=EventPhase.EXECUTION,
        ordinal=1,
        entity_id="fill-mismatched-reference",
        causation_id=events[6].event_id,
    )

    with pytest.raises(EventTransitionError, match="reference price"):
        engine.process(mismatched)
    assert engine.journal.count == 7


def test_large_account_mark_forgery_is_not_hidden_by_relative_tolerance() -> None:
    engine = DeterministicEventEngine(_RUN_ID, calendar=_CALENDAR, initial_cash=1.0e12)
    forged = _event(
        PortfolioMarked(
            mark_id="large-forged-open",
            mark_type="open",
            prices=(),
            cash=1.0e12 - 0.5,
            holdings_value=0.0,
            accrued_charges=0.0,
            equity=1.0e12 - 0.5,
        ),
        session=_CALENDAR[0],
        bar_index=0,
        phase=EventPhase.OPEN_MARK,
        entity_id="large-forged-open",
    )

    with pytest.raises(EventTransitionError, match="cash does not reconcile"):
        engine.process(forged)
    assert engine.journal.count == 0


def test_cash_only_mark_and_read_only_portfolio_facade() -> None:
    engine = DeterministicEventEngine(_RUN_ID, calendar=_CALENDAR, initial_cash=1_000.0)
    cash_mark = _event(
        PortfolioMarked(
            mark_id="cash-only-open",
            mark_type="open",
            prices=(),
            cash=1_000.0,
            holdings_value=0.0,
            accrued_charges=0.0,
            equity=1_000.0,
        ),
        session=_CALENDAR[0],
        bar_index=0,
        phase=EventPhase.OPEN_MARK,
        entity_id="cash-only-open",
    )

    engine.process(cash_mark)
    assert engine.positions == {}
    assert engine.cash == 1_000.0
    assert engine.value_portfolio(_CALENDAR[0], {}).equity == 1_000.0
    assert not hasattr(engine, "ledger")
    with pytest.raises(TypeError):
        engine.positions["A"] = 1.0  # type: ignore[index]


def test_replay_wraps_semantically_invalid_but_canonical_journal() -> None:
    journal = InMemoryJournal()
    signal_without_marks = _event(
        SignalAvailable(
            signal_id="signal-without-close",
            model_id="synthetic-noncandidate",
            signal_digest=_DIGEST,
        ),
        session=_CALENDAR[0],
        bar_index=0,
        phase=EventPhase.SIGNAL,
        entity_id="signal-without-close",
    )
    journal.append(signal_without_marks)

    with pytest.raises(EventReplayError, match="journal event semantics failed replay"):
        DeterministicEventEngine.replay(
            journal,
            calendar=_CALENDAR,
            initial_cash=1_000.0,
            expected_run_id=_RUN_ID,
        )
