from __future__ import annotations

import hashlib
import json
from dataclasses import FrozenInstanceError
from datetime import date, datetime
from typing import Any, cast

import pytest

from alphaforge.execution.events import (
    MAX_CANONICAL_BYTES,
    MAX_MARK_PRICES,
    MAX_TARGET_ASSETS,
    CashChargeAccrued,
    EngineHalted,
    EventCoordinate,
    EventPhase,
    ExecutionEvent,
    ExecutionEventError,
    FeeCategory,
    FeeComponent,
    FillApplied,
    OrderAccepted,
    OrderCancelled,
    OrderRejected,
    OrderSubmitted,
    PortfolioMarked,
    SignalAvailable,
    TargetDecided,
)

SESSION = date(2026, 8, 3)
NEXT_SESSION = date(2026, 8, 4)
DIGEST_A = "a" * 64
DIGEST_B = "b" * 64
DIGEST_C = "c" * 64


def _event(
    payload: object,
    phase: EventPhase,
    *,
    ordinal: int = 0,
    causation_id: str | None = None,
) -> ExecutionEvent:
    return ExecutionEvent(
        run_id="run-001",
        correlation_id="rebalance-001",
        entity_id="portfolio-primary",
        coordinate=EventCoordinate(
            session=SESSION,
            bar_index=17,
            phase=phase,
            ordinal=ordinal,
        ),
        payload=payload,  # type: ignore[arg-type]
        causation_id=causation_id,
    )


def _target() -> TargetDecided:
    return TargetDecided(
        target_id="target-001",
        portfolio_id="markowitz-v1",
        solver_id="osqp-1.1.3",
        eligible_session=NEXT_SESSION,
        cash_weight=0.2,
        weights=(("MSFT", 0.3), ("AAPL", 0.5)),
        configuration_digest=DIGEST_A,
        data_digest=DIGEST_B,
        problem_digest=DIGEST_C,
    )


def _fill() -> FillApplied:
    return FillApplied(
        fill_id="fill-001",
        order_id="order-001",
        symbol="AAPL",
        side="buy",
        quantity=25.0,
        reference_price=99.5,
        price=100.0,
        fees=(
            FeeComponent(FeeCategory.OTHER_FEE, 0.25),
            FeeComponent(FeeCategory.COMMISSION, 1.0),
            FeeComponent(FeeCategory.EXCHANGE_FEE, 0.5),
        ),
    )


def _all_events() -> tuple[ExecutionEvent, ...]:
    open_mark = _event(
        PortfolioMarked(
            mark_id="open-001",
            mark_type="open",
            prices=(("MSFT", 200.0), ("AAPL", 100.0)),
            cash=1_000.0,
            holdings_value=0.0,
            accrued_charges=0.0,
            equity=1_000.0,
        ),
        EventPhase.OPEN_MARK,
    )
    submitted = _event(
        OrderSubmitted(
            order_id="order-001",
            symbol="AAPL",
            side="buy",
            quantity=5.0,
        ),
        EventPhase.ORDER_SUBMISSION,
        causation_id=open_mark.event_id,
    )
    accepted = _event(
        OrderAccepted(order_id="order-001", accepted_quantity=5.0),
        EventPhase.EXECUTION,
        ordinal=0,
        causation_id=submitted.event_id,
    )
    fill = _event(
        _fill(),
        EventPhase.EXECUTION,
        ordinal=1,
        causation_id=accepted.event_id,
    )
    cancelled = _event(
        OrderCancelled(
            order_id="order-001",
            reason_code="day_expired",
            cancelled_quantity=1.0,
        ),
        EventPhase.DAY_CANCEL,
        causation_id=fill.event_id,
    )
    charge = _event(
        CashChargeAccrued(
            charge_id="charge-001",
            charge_type="financing",
            amount=0.75,
        ),
        EventPhase.CHARGE,
        causation_id=cancelled.event_id,
    )
    close_mark = _event(
        PortfolioMarked(
            mark_id="close-001",
            mark_type="close",
            prices=(("AAPL", 101.0), ("MSFT", 201.0)),
            cash=499.25,
            holdings_value=505.0,
            accrued_charges=0.75,
            equity=1_004.25,
        ),
        EventPhase.CLOSE_MARK,
        causation_id=charge.event_id,
    )
    signal = _event(
        SignalAvailable(
            signal_id="signal-001",
            model_id="baseline-v1",
            signal_digest=DIGEST_A,
        ),
        EventPhase.SIGNAL,
        causation_id=close_mark.event_id,
    )
    target = _event(
        _target(),
        EventPhase.TARGET_DECISION,
        causation_id=signal.event_id,
    )
    halted = _event(
        EngineHalted(
            reason_code="completed",
            detail="Synthetic replay completed.",
            recoverable=False,
        ),
        EventPhase.CONTROL,
        causation_id=target.event_id,
    )
    return (
        open_mark,
        submitted,
        accepted,
        fill,
        cancelled,
        charge,
        close_mark,
        signal,
        target,
        halted,
    )


@pytest.mark.parametrize("event", _all_events())
def test_every_payload_round_trips_exact_canonical_bytes(event: ExecutionEvent) -> None:
    encoded = event.canonical_bytes()
    restored = ExecutionEvent.from_canonical_bytes(encoded)

    assert restored == event
    assert restored.event_id == event.event_id
    assert restored.canonical_bytes() == encoded
    assert len(encoded) <= MAX_CANONICAL_BYTES


def test_rejected_order_payload_round_trips() -> None:
    event = _event(
        OrderRejected(order_id="order-002", reason_code="missing_price"),
        EventPhase.EXECUTION,
    )

    assert ExecutionEvent.from_canonical_bytes(event.canonical_bytes()) == event


def test_phase_values_encode_the_daily_bar_timeline() -> None:
    observed = [
        EventPhase.OPEN_MARK,
        EventPhase.ORDER_SUBMISSION,
        EventPhase.EXECUTION,
        EventPhase.DAY_CANCEL,
        EventPhase.CHARGE,
        EventPhase.CLOSE_MARK,
        EventPhase.SIGNAL,
        EventPhase.TARGET_DECISION,
        EventPhase.CONTROL,
    ]

    assert observed == sorted(observed)
    assert [event.coordinate.phase for event in sorted(reversed(_all_events()))] == [
        EventPhase.OPEN_MARK,
        EventPhase.ORDER_SUBMISSION,
        EventPhase.EXECUTION,
        EventPhase.EXECUTION,
        EventPhase.DAY_CANCEL,
        EventPhase.CHARGE,
        EventPhase.CLOSE_MARK,
        EventPhase.SIGNAL,
        EventPhase.TARGET_DECISION,
        EventPhase.CONTROL,
    ]


def test_same_coordinate_uses_event_id_as_stable_tie_breaker() -> None:
    first = _event(
        OrderAccepted(order_id="order-a", accepted_quantity=1.0),
        EventPhase.EXECUTION,
    )
    second = _event(
        OrderAccepted(order_id="order-b", accepted_quantity=1.0),
        EventPhase.EXECUTION,
    )

    assert sorted((second, first)) == sorted((first, second))
    assert [item.event_id for item in sorted((second, first))] == sorted(
        (first.event_id, second.event_id)
    )


def test_content_identity_is_idempotent_and_sensitive_to_lineage() -> None:
    payload = SignalAvailable("signal-001", "model-001", DIGEST_A)
    first = _event(payload, EventPhase.SIGNAL)
    duplicate = _event(payload, EventPhase.SIGNAL)
    correlated_elsewhere = ExecutionEvent(
        run_id=first.run_id,
        correlation_id="rebalance-002",
        entity_id=first.entity_id,
        coordinate=first.coordinate,
        payload=payload,
    )

    assert duplicate.event_id == first.event_id
    assert duplicate.canonical_bytes() == first.canonical_bytes()
    assert correlated_elsewhere.event_id != first.event_id


def test_canonical_json_has_a_stable_exact_schema() -> None:
    event = ExecutionEvent(
        run_id="run",
        correlation_id="correlation",
        entity_id="signal-1",
        coordinate=EventCoordinate(SESSION, 0, EventPhase.SIGNAL),
        payload=SignalAvailable("signal-1", "model-1", DIGEST_A),
    )
    document = json.loads(event.canonical_bytes())
    event_id = document.pop("event_id")
    expected_body = (
        b'{"causation_id":null,"coordinate":{"bar_index":0,"ordinal":0,'
        b'"phase":"signal","session":"2026-08-03"},'
        b'"correlation_id":"correlation","entity_id":"signal-1",'
        b'"event_type":"signal_available","payload":{"model_id":"model-1",'
        b'"signal_digest":"aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",'
        b'"signal_id":"signal-1"},"run_id":"run","schema_version":"1.0.0"}'
    )

    assert (
        json.dumps(
            document,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
        ).encode()
        == expected_body
    )
    assert event_id == hashlib.sha256(expected_body).hexdigest()


def test_event_id_hashes_canonical_content_without_self_reference() -> None:
    event = _event(_target(), EventPhase.TARGET_DECISION)
    document = json.loads(event.canonical_bytes())
    supplied = document.pop("event_id")
    body = json.dumps(
        document,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode()

    assert supplied == hashlib.sha256(body).hexdigest() == event.event_id


def test_collections_are_canonically_sorted_and_detached() -> None:
    target_input = (("MSFT", 0.3), ("AAPL", 0.5))
    target = _target()
    prices_input = (("MSFT", 200.0), ("AAPL", 100.0))
    mark = PortfolioMarked(
        mark_id="open-001",
        mark_type="open",
        prices=prices_input,
        cash=1_000.0,
        holdings_value=0.0,
        accrued_charges=0.0,
        equity=1_000.0,
    )
    fill = _fill()

    assert target.weights == (("AAPL", 0.5), ("MSFT", 0.3))
    assert target_input == (("MSFT", 0.3), ("AAPL", 0.5))
    assert mark.prices == (("AAPL", 100.0), ("MSFT", 200.0))
    assert prices_input == (("MSFT", 200.0), ("AAPL", 100.0))
    assert tuple(item.category for item in fill.fees) == (
        FeeCategory.COMMISSION,
        FeeCategory.EXCHANGE_FEE,
        FeeCategory.OTHER_FEE,
    )
    assert fill.total_fees == pytest.approx(1.75)
    assert fill.signed_quantity == 25.0


def test_sell_fill_has_negative_signed_quantity() -> None:
    fill = FillApplied(
        fill_id="fill-sell",
        order_id="order-sell",
        symbol="AAPL",
        side="sell",
        quantity=2.0,
        reference_price=100.0,
        price=99.0,
    )

    assert fill.signed_quantity == -2.0


def test_cash_charge_rejects_unknown_accounting_category() -> None:
    with pytest.raises(ExecutionEventError, match="charge_type"):
        CashChargeAccrued(
            charge_id="charge-invalid",
            charge_type="tax",  # type: ignore[arg-type]
            amount=1.0,
        )


def test_payloads_and_envelope_are_frozen_and_slotted() -> None:
    payload = SignalAvailable("signal-001", "model-001", DIGEST_A)
    event = _event(payload, EventPhase.SIGNAL)

    assert not hasattr(payload, "__dict__")
    assert not hasattr(event, "__dict__")
    with pytest.raises(FrozenInstanceError):
        event.run_id = "different"  # type: ignore[misc]
    with pytest.raises(FrozenInstanceError):
        payload.signal_id = "different"  # type: ignore[misc]


@pytest.mark.parametrize(
    ("coordinate", "message"),
    [
        ((datetime(2026, 8, 3), 0, EventPhase.OPEN_MARK, 0), "datetime.date"),
        ((SESSION, True, EventPhase.OPEN_MARK, 0), "bar_index"),
        ((SESSION, -1, EventPhase.OPEN_MARK, 0), "bar_index"),
        ((SESSION, 0, 10, 0), "EventPhase"),
        ((SESSION, 0, EventPhase.OPEN_MARK, -1), "ordinal"),
    ],
)
def test_coordinate_rejects_invalid_state(
    coordinate: tuple[object, object, object, object],
    message: str,
) -> None:
    with pytest.raises(ExecutionEventError, match=message):
        EventCoordinate(*coordinate)  # type: ignore[arg-type]


@pytest.mark.parametrize("field", ["run_id", "correlation_id", "entity_id"])
def test_envelope_rejects_invalid_identifiers(field: str) -> None:
    values: dict[str, Any] = {
        "run_id": "run-001",
        "correlation_id": "rebalance-001",
        "entity_id": "portfolio-primary",
        "coordinate": EventCoordinate(SESSION, 0, EventPhase.SIGNAL),
        "payload": SignalAvailable("signal-001", "model-001", DIGEST_A),
    }
    values[field] = "invalid value"

    with pytest.raises(ExecutionEventError, match=field):
        ExecutionEvent(**values)


def test_envelope_rejects_bad_causation_and_wrong_phase() -> None:
    with pytest.raises(ExecutionEventError, match="causation_id"):
        _event(
            SignalAvailable("signal-001", "model-001", DIGEST_A),
            EventPhase.SIGNAL,
            causation_id="root",
        )
    with pytest.raises(ExecutionEventError, match="requires phase"):
        _event(SignalAvailable("signal-001", "model-001", DIGEST_A), EventPhase.OPEN_MARK)


def test_mark_type_selects_open_or_close_phase() -> None:
    open_payload = PortfolioMarked("mark", "open", (("AAPL", 1.0),), 1.0, 0.0, 0.0, 1.0)
    close_payload = PortfolioMarked("mark", "close", (("AAPL", 1.0),), 1.0, 0.0, 0.0, 1.0)

    with pytest.raises(ExecutionEventError, match="OPEN_MARK"):
        _event(open_payload, EventPhase.CLOSE_MARK)
    with pytest.raises(ExecutionEventError, match="CLOSE_MARK"):
        _event(close_payload, EventPhase.OPEN_MARK)


def test_target_must_be_eligible_after_decision_session() -> None:
    payload = TargetDecided(
        target_id="target-001",
        portfolio_id="portfolio-001",
        solver_id="solver-001",
        eligible_session=SESSION,
        cash_weight=1.0,
        weights=(),
        configuration_digest=DIGEST_A,
        data_digest=DIGEST_B,
        problem_digest=DIGEST_C,
    )

    with pytest.raises(ExecutionEventError, match="strictly after"):
        _event(payload, EventPhase.TARGET_DECISION)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"signal_id": "", "model_id": "model", "signal_digest": DIGEST_A},
        {"signal_id": "signal", "model_id": "model", "signal_digest": "A" * 64},
        {"signal_id": "signal\n", "model_id": "model", "signal_digest": DIGEST_A},
    ],
)
def test_signal_rejects_unbounded_or_noncanonical_identifiers(kwargs: dict[str, Any]) -> None:
    with pytest.raises(ExecutionEventError):
        SignalAvailable(**kwargs)


def test_target_validates_budget_uniqueness_types_and_bounds() -> None:
    common: dict[str, Any] = {
        "target_id": "target",
        "portfolio_id": "portfolio",
        "solver_id": "solver",
        "eligible_session": NEXT_SESSION,
        "configuration_digest": DIGEST_A,
        "data_digest": DIGEST_B,
        "problem_digest": DIGEST_C,
    }
    with pytest.raises(ExecutionEventError, match="reconcile"):
        TargetDecided(**common, cash_weight=0.0, weights=(("AAPL", 0.5),))
    with pytest.raises(ExecutionEventError, match="duplicate"):
        TargetDecided(
            **common,
            cash_weight=0.0,
            weights=(("AAPL", 0.5), ("AAPL", 0.5)),
        )
    with pytest.raises(ExecutionEventError, match="tuple"):
        TargetDecided(**common, cash_weight=1.0, weights=[])  # type: ignore[arg-type]
    with pytest.raises(ExecutionEventError, match="magnitude"):
        TargetDecided(**common, cash_weight=-100.0, weights=(("AAPL", 101.0),))
    with pytest.raises(ExecutionEventError, match="at most"):
        TargetDecided(
            **common,
            cash_weight=1.0,
            weights=tuple((f"A{index:04d}", 0.0) for index in range(MAX_TARGET_ASSETS + 1)),
        )


@pytest.mark.parametrize(
    "payload",
    [
        lambda: OrderSubmitted("order", "AAPL", cast(Any, "hold"), 1.0),
        lambda: OrderSubmitted("order", "AAPL", cast(Any, ["buy"]), 1.0),
        lambda: OrderSubmitted("order", "AAPL", "buy", 0.0),
        lambda: OrderSubmitted("order", "AAPL", "buy", float("nan")),
        lambda: OrderSubmitted("order", "AAPL", "buy", 1.0, limit_price=1.0),
        lambda: OrderSubmitted("order", "AAPL", "buy", 1.0, order_type="limit"),
        lambda: OrderSubmitted("order", "AAPL", "buy", 1.0, order_type="limit", limit_price=-1.0),
        lambda: OrderSubmitted("order", "AAPL", "buy", 1.0, order_type=cast(Any, ["market"])),
        lambda: OrderSubmitted("order", "AAPL", "buy", 1.0, time_in_force=cast(Any, "gtc")),
        lambda: OrderAccepted("order", 0.0),
        lambda: OrderRejected("order", "Not Stable"),
        lambda: OrderCancelled("order", "day_expired", 0.0),
        lambda: CashChargeAccrued("charge", "borrow", 0.0),
        lambda: CashChargeAccrued("charge", "borrow", 1.0, "usd"),
        lambda: CashChargeAccrued("charge", "borrow", 1.0, "EUR"),
    ],
)
def test_order_and_charge_payloads_fail_closed(payload: Any) -> None:
    with pytest.raises(ExecutionEventError):
        payload()


def test_bounded_limit_order_round_trips() -> None:
    order = OrderSubmitted(
        "limit-001",
        "BRK.B",
        "sell",
        3.0,
        order_type="limit",
        limit_price=512.25,
    )
    event = _event(order, EventPhase.ORDER_SUBMISSION)

    assert ExecutionEvent.from_canonical_bytes(event.canonical_bytes()) == event


def test_fill_validates_reference_price_fees_and_notional() -> None:
    with pytest.raises(ExecutionEventError, match="side"):
        FillApplied("fill", "order", "AAPL", cast(Any, ["buy"]), 1.0, 1.0, 1.0)
    with pytest.raises(ExecutionEventError, match="reference_price"):
        FillApplied("fill", "order", "AAPL", "buy", 1.0, 0.0, 1.0)
    with pytest.raises(ExecutionEventError, match="fee categories"):
        FillApplied(
            "fill",
            "order",
            "AAPL",
            "buy",
            1.0,
            1.0,
            1.0,
            fees=(
                FeeComponent(FeeCategory.COMMISSION, 1.0),
                FeeComponent(FeeCategory.COMMISSION, 2.0),
            ),
        )
    with pytest.raises(ExecutionEventError, match="non-negative"):
        FeeComponent(FeeCategory.OTHER_FEE, -1.0)
    with pytest.raises(ExecutionEventError, match="resource ceiling"):
        FillApplied("fill", "order", "AAPL", "buy", 1.0e15, 1.0e12, 1.0e12)
    with pytest.raises(ExecutionEventError, match="tuple"):
        FillApplied(
            "fill",
            "order",
            "AAPL",
            "buy",
            1.0,
            1.0,
            1.0,
            fees=[FeeComponent(FeeCategory.COMMISSION, 1.0)],  # type: ignore[arg-type]
        )


def test_portfolio_mark_validates_prices_reconciliation_and_bounds() -> None:
    with pytest.raises(ExecutionEventError, match="mark_type"):
        PortfolioMarked("mark", cast(Any, ["open"]), (), 1.0, 0.0, 0.0, 1.0)
    with pytest.raises(ExecutionEventError, match="duplicate"):
        PortfolioMarked(
            "mark",
            "open",
            (("AAPL", 1.0), ("AAPL", 2.0)),
            1.0,
            0.0,
            0.0,
            1.0,
        )
    with pytest.raises(ExecutionEventError, match="positive"):
        PortfolioMarked("mark", "open", (("AAPL", 0.0),), 1.0, 0.0, 0.0, 1.0)
    with pytest.raises(ExecutionEventError, match="does not reconcile"):
        PortfolioMarked("mark", "open", (("AAPL", 1.0),), 1.0, 1.0, 0.0, 3.0)
    with pytest.raises(ExecutionEventError, match="non-negative"):
        PortfolioMarked("mark", "open", (("AAPL", 1.0),), 1.0, 0.0, -1.0, 1.0)
    with pytest.raises(ExecutionEventError, match="at most"):
        PortfolioMarked(
            "mark",
            "open",
            tuple((f"A{index:04d}", 1.0) for index in range(MAX_MARK_PRICES + 1)),
            1.0,
            0.0,
            0.0,
            1.0,
        )


def test_all_cash_portfolio_mark_allows_empty_prices_and_round_trips() -> None:
    event = _event(
        PortfolioMarked(
            mark_id="all-cash-open",
            mark_type="open",
            prices=(),
            cash=1_000.0,
            holdings_value=0.0,
            accrued_charges=0.0,
            equity=1_000.0,
        ),
        EventPhase.OPEN_MARK,
    )

    restored = ExecutionEvent.from_canonical_bytes(event.canonical_bytes())

    assert restored == event
    assert isinstance(restored.payload, PortfolioMarked)
    assert restored.payload.prices == ()


def test_portfolio_mark_reconciliation_has_no_absolute_currency_floor() -> None:
    tiny = PortfolioMarked(
        "tiny",
        "close",
        (("AAPL", 1.0),),
        1.0e-12,
        2.0e-12,
        0.0,
        3.0e-12,
    )
    large = PortfolioMarked(
        "large",
        "close",
        (("AAPL", 1.0),),
        5.0e17,
        5.0e17,
        0.0,
        1.0e18,
    )

    assert tiny.equity == 3.0e-12
    assert large.equity == 1.0e18
    with pytest.raises(ExecutionEventError, match="does not reconcile"):
        PortfolioMarked(
            "tiny-mismatch",
            "close",
            (("AAPL", 1.0),),
            1.0e-12,
            2.0e-12,
            0.0,
            3.1e-12,
        )
    with pytest.raises(ExecutionEventError, match="does not reconcile"):
        PortfolioMarked(
            "large-mismatch",
            "close",
            (("AAPL", 1.0),),
            5.0e17,
            5.0e17,
            0.0,
            1.0e18 - 1.0e6,
        )


def test_halt_detail_is_bounded_printable_and_recoverable_is_boolean() -> None:
    with pytest.raises(ExecutionEventError, match="printable"):
        EngineHalted("fault", "contains\nnewline")
    with pytest.raises(ExecutionEventError, match="printable"):
        EngineHalted("fault", "x" * 513)
    with pytest.raises(ExecutionEventError, match="boolean"):
        EngineHalted("fault", "Synthetic fault.", recoverable=1)  # type: ignore[arg-type]


def _reidentify(document: dict[str, Any]) -> bytes:
    body = {key: value for key, value in document.items() if key != "event_id"}
    encoded_body = json.dumps(
        body,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode()
    document["event_id"] = hashlib.sha256(encoded_body).hexdigest()
    return json.dumps(
        document,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode()


@pytest.mark.parametrize(
    "value",
    [
        b"",
        b"[]",
        b"not-json",
        b'"scalar"',
        b'{"value":NaN}',
        b"\xff",
        b"{" * (MAX_CANONICAL_BYTES + 1),
    ],
)
def test_deserializer_rejects_noncanonical_or_unbounded_documents(value: bytes) -> None:
    with pytest.raises(ExecutionEventError):
        ExecutionEvent.from_canonical_bytes(value)


def test_deserializer_maps_excessive_json_nesting_to_domain_error() -> None:
    nested = (b'{"x":' * 2_000) + b"0" + (b"}" * 2_000)

    with pytest.raises(ExecutionEventError):
        ExecutionEvent.from_canonical_bytes(nested)


def test_deserializer_requires_bytes() -> None:
    with pytest.raises(ExecutionEventError, match="must be bytes"):
        ExecutionEvent.from_canonical_bytes(bytearray(b"{}"))  # type: ignore[arg-type]


def test_deserializer_rejects_whitespace_trailing_newline_and_duplicate_keys() -> None:
    encoded = _event(
        SignalAvailable("signal-001", "model-001", DIGEST_A), EventPhase.SIGNAL
    ).canonical_bytes()
    document = json.loads(encoded)
    pretty = json.dumps(document, indent=2, sort_keys=True).encode()
    duplicate = encoded.replace(b'{"causation_id":', b'{"run_id":"other","causation_id":', 1)

    with pytest.raises(ExecutionEventError, match="canonical encoding"):
        ExecutionEvent.from_canonical_bytes(pretty)
    with pytest.raises(ExecutionEventError, match="canonical encoding"):
        ExecutionEvent.from_canonical_bytes(encoded + b"\n")
    with pytest.raises(ExecutionEventError, match="duplicate JSON key"):
        ExecutionEvent.from_canonical_bytes(duplicate)


def test_deserializer_rejects_tampered_content_and_identifier() -> None:
    encoded = _event(
        SignalAvailable("signal-001", "model-001", DIGEST_A), EventPhase.SIGNAL
    ).canonical_bytes()
    document = json.loads(encoded)
    document["payload"]["model_id"] = "tampered"
    tampered_without_reidentifying = json.dumps(
        document,
        sort_keys=True,
        separators=(",", ":"),
    ).encode()

    with pytest.raises(ExecutionEventError, match="event_id"):
        ExecutionEvent.from_canonical_bytes(tampered_without_reidentifying)


@pytest.mark.parametrize(
    "mutation",
    [
        lambda value: value.update(schema_version="2.0.0"),
        lambda value: value.update(event_type="unknown_event"),
        lambda value: value["coordinate"].update(phase="open_mark"),
        lambda value: value.update(unexpected=True),
        lambda value: value["payload"].update(unexpected=True),
    ],
)
def test_deserializer_rejects_reidentified_semantically_invalid_documents(mutation: Any) -> None:
    event = _event(SignalAvailable("signal-001", "model-001", DIGEST_A), EventPhase.SIGNAL)
    document = json.loads(event.canonical_bytes())
    mutation(document)

    with pytest.raises(ExecutionEventError):
        ExecutionEvent.from_canonical_bytes(_reidentify(document))


def test_deserializer_rejects_noncanonical_negative_zero() -> None:
    event = _event(
        OrderSubmitted("order", "AAPL", "buy", 1.0),
        EventPhase.ORDER_SUBMISSION,
    )
    document = json.loads(event.canonical_bytes())
    document["payload"]["quantity"] = -0.0

    with pytest.raises(ExecutionEventError):
        ExecutionEvent.from_canonical_bytes(_reidentify(document))


@pytest.mark.parametrize(
    ("event", "field", "invalid_value"),
    [
        (
            _event(OrderSubmitted("order", "AAPL", "buy", 1.0), EventPhase.ORDER_SUBMISSION),
            "side",
            ["buy"],
        ),
        (
            _event(
                PortfolioMarked("mark", "open", (), 1.0, 0.0, 0.0, 1.0),
                EventPhase.OPEN_MARK,
            ),
            "mark_type",
            ["open"],
        ),
        (
            _event(
                CashChargeAccrued("charge", "borrow", 1.0),
                EventPhase.CHARGE,
            ),
            "charge_type",
            [],
        ),
    ],
)
def test_deserializer_maps_unhashable_enum_values_to_domain_error(
    event: ExecutionEvent,
    field: str,
    invalid_value: object,
) -> None:
    document = json.loads(event.canonical_bytes())
    document["payload"][field] = invalid_value

    with pytest.raises(ExecutionEventError, match=field):
        ExecutionEvent.from_canonical_bytes(_reidentify(document))


def test_canonical_payload_contains_no_raw_signal_or_arbitrary_metadata_field() -> None:
    encoded = _event(
        SignalAvailable("signal-001", "model-001", DIGEST_A), EventPhase.SIGNAL
    ).canonical_bytes()
    payload = json.loads(encoded)["payload"]

    assert set(payload) == {"model_id", "signal_digest", "signal_id"}
    assert "value" not in payload
    assert "metadata" not in payload
