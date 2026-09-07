"""Tests for the broker-neutral contracts (SF-S5-MR3).

Grouped by the invariant each protects. The ones that carry the most weight:

* **A naive datetime is refused**, never localized or assumed UTC.
* **Money and quantity are decimal**; a float is refused at the boundary.
* **The order state machine is closed**: a terminal order accepts nothing.
* **A client order ID is content-derived**, so a replay is a no-op.
* **Impossible broker responses are refused**, not recorded — overfills,
  crossed quotes, and FILLED-with-a-residual are reconciliation faults.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta, timezone
from decimal import Decimal
from typing import cast

import pytest

from alphaforge.broker import (
    ALLOWED_ORDER_TRANSITIONS,
    TERMINAL_ORDER_STATES,
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
    TimeInForce,
    assert_transition_allowed,
    derive_client_order_id,
    utc_timestamp,
    validate_client_order_id,
    validate_symbol,
)

NOW = datetime(2026, 8, 4, 15, 0, tzinfo=UTC)
DIGEST = "a" * 64


# ---------------------------------------------------------------------------
# Time
# ---------------------------------------------------------------------------


def test_a_naive_datetime_is_refused_rather_than_assumed() -> None:
    """Both "it's local" and "it's UTC" are wrong somewhere."""
    with pytest.raises(BrokerContractError, match="timezone-aware"):
        utc_timestamp(datetime(2026, 8, 4, 15, 0), field_name="t")  # noqa: DTZ001


def test_a_non_utc_aware_datetime_is_normalized_not_refused() -> None:
    eastern = timezone(timedelta(hours=-4))
    converted = utc_timestamp(datetime(2026, 8, 4, 11, 0, tzinfo=eastern), field_name="t")
    assert converted == NOW
    assert converted.tzinfo is UTC


def test_a_non_datetime_is_refused() -> None:
    with pytest.raises(BrokerContractError, match="must be a datetime"):
        utc_timestamp("2026-08-04T15:00:00Z", field_name="t")


# ---------------------------------------------------------------------------
# Decimal money
# ---------------------------------------------------------------------------


def test_a_float_price_is_refused_at_the_boundary() -> None:
    """0.01 is not representable in binary floating point.

    ``cast`` rather than ``type: ignore`` for the deliberate violation: the
    static type already forbids a float, and this proves the runtime guard
    holds for callers who reach the constructor from untyped broker JSON.
    """
    with pytest.raises(BrokerContractError, match="not a float"):
        Position(symbol="AAPL", quantity=Decimal("1"), average_entry_price=cast(Decimal, 100.01))


def test_decimal_strings_and_ints_are_accepted() -> None:
    """Broker payloads carry numbers as strings; the constructor normalizes them."""
    assert Position(
        symbol="AAPL",
        quantity=cast(Decimal, 1),
        average_entry_price=cast(Decimal, "100.01"),
    ).average_entry_price == Decimal("100.01")


def test_a_non_finite_quantity_is_refused() -> None:
    with pytest.raises(BrokerContractError, match="finite"):
        Position(symbol="AAPL", quantity=Decimal("NaN"), average_entry_price=Decimal("1"))


def test_a_bool_is_not_a_number() -> None:
    with pytest.raises(BrokerContractError, match="not a bool"):
        Position(symbol="AAPL", quantity=cast(Decimal, True), average_entry_price=Decimal("1"))


# ---------------------------------------------------------------------------
# Symbols and identifiers
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("symbol", ["AAPL", "BRK.B", "RDS-A", "A"])
def test_well_formed_symbols_are_accepted(symbol: str) -> None:
    assert validate_symbol(symbol) == symbol


@pytest.mark.parametrize(
    "symbol", ["aapl", "", "TOOLONGSYMBOLNAME1", "AA PL", "AA;DROP", "123", "../etc"]
)
def test_malformed_symbols_are_refused(symbol: str) -> None:
    with pytest.raises(BrokerContractError):
        validate_symbol(symbol)


def test_a_client_order_id_is_derived_from_decision_content() -> None:
    first = derive_client_order_id(
        strategy_id="s1", decision_timestamp=NOW, symbol="AAPL", side=OrderSide.BUY, sequence=0
    )
    second = derive_client_order_id(
        strategy_id="s1", decision_timestamp=NOW, symbol="AAPL", side=OrderSide.BUY, sequence=0
    )
    assert first == second, "a replayed decision must produce the same identifier"
    assert validate_client_order_id(first) == first


@pytest.mark.parametrize(
    "kwargs",
    [
        {"strategy_id": "s2"},
        {"symbol": "MSFT"},
        {"side": OrderSide.SELL},
        {"sequence": 1},
        {"decision_timestamp": NOW + timedelta(seconds=1)},
    ],
)
def test_every_component_changes_the_derived_id(kwargs: dict[str, object]) -> None:
    base = {
        "strategy_id": "s1",
        "decision_timestamp": NOW,
        "symbol": "AAPL",
        "side": OrderSide.BUY,
        "sequence": 0,
    }
    assert derive_client_order_id(**base) != derive_client_order_id(**{**base, **kwargs})  # type: ignore[arg-type]


def test_a_malformed_client_order_id_is_refused() -> None:
    with pytest.raises(BrokerContractError, match="malformed"):
        validate_client_order_id("has spaces")
    with pytest.raises(BrokerContractError, match="malformed"):
        validate_client_order_id("x" * 65)


# ---------------------------------------------------------------------------
# Order state machine
# ---------------------------------------------------------------------------


def test_every_terminal_state_accepts_nothing() -> None:
    for state in TERMINAL_ORDER_STATES:
        assert ALLOWED_ORDER_TRANSITIONS[state] == frozenset()


def test_the_transition_table_covers_every_state() -> None:
    assert set(ALLOWED_ORDER_TRANSITIONS) == set(OrderState)


def test_a_late_update_for_a_settled_order_is_refused() -> None:
    with pytest.raises(BrokerContractError, match="already terminal"):
        assert_transition_allowed(OrderState.FILLED, OrderState.PARTIALLY_FILLED)


def test_an_unlisted_transition_is_refused() -> None:
    with pytest.raises(BrokerContractError, match="not a permitted transition"):
        assert_transition_allowed(OrderState.ACCEPTED, OrderState.PENDING_NEW)


def test_partial_fills_may_repeat() -> None:
    assert_transition_allowed(OrderState.PARTIALLY_FILLED, OrderState.PARTIALLY_FILLED)


# ---------------------------------------------------------------------------
# OrderRequest
# ---------------------------------------------------------------------------


def _request(**overrides: object) -> OrderRequest:
    base: dict[str, object] = {
        "client_order_id": "af-test-1",
        "symbol": "AAPL",
        "side": OrderSide.BUY,
        "quantity": Decimal("10"),
        "order_type": OrderType.MARKET,
        "time_in_force": TimeInForce.DAY,
    }
    base.update(overrides)
    return OrderRequest(**base)  # type: ignore[arg-type]


def test_a_zero_or_negative_quantity_is_refused() -> None:
    """Direction lives in `side`, so a negative quantity is malformed, not a sell."""
    for quantity in (Decimal("0"), Decimal("-5")):
        with pytest.raises(BrokerContractError, match="quantity"):
            _request(quantity=quantity)


def test_a_limit_order_requires_a_limit_price() -> None:
    with pytest.raises(BrokerContractError, match="requires limit_price"):
        _request(order_type=OrderType.LIMIT)


def test_a_market_order_refuses_a_limit_price() -> None:
    """Supplying one means the caller wanted a limit order and would get an unbounded fill."""
    with pytest.raises(BrokerContractError, match="not permitted on a market order"):
        _request(limit_price=Decimal("100"))


def test_a_non_positive_limit_price_is_refused() -> None:
    with pytest.raises(BrokerContractError, match="limit_price must be positive"):
        _request(order_type=OrderType.LIMIT, limit_price=Decimal("0"))


def test_a_request_serializes_decimals_as_strings() -> None:
    payload = _request().to_dict()
    assert payload["quantity"] == "10"
    assert isinstance(payload["quantity"], str)


# ---------------------------------------------------------------------------
# Fills
# ---------------------------------------------------------------------------


def test_a_buy_fill_has_negative_cash_effect() -> None:
    fill = Fill(
        client_order_id="af-test-1",
        symbol="AAPL",
        side=OrderSide.BUY,
        quantity=Decimal("10"),
        price=Decimal("100"),
        filled_at=NOW,
        fill_id="f1",
    )
    assert fill.notional == Decimal("-1000")


def test_a_sell_fill_has_positive_cash_effect() -> None:
    fill = Fill(
        client_order_id="af-test-1",
        symbol="AAPL",
        side=OrderSide.SELL,
        quantity=Decimal("10"),
        price=Decimal("100"),
        filled_at=NOW,
        fill_id="f1",
    )
    assert fill.notional == Decimal("1000")


def test_a_zero_quantity_fill_is_refused() -> None:
    with pytest.raises(BrokerContractError, match="positive quantity"):
        Fill(
            client_order_id="af-test-1",
            symbol="AAPL",
            side=OrderSide.BUY,
            quantity=Decimal("0"),
            price=Decimal("100"),
            filled_at=NOW,
            fill_id="f1",
        )


# ---------------------------------------------------------------------------
# OrderStatus: impossible broker responses
# ---------------------------------------------------------------------------


def _status(**overrides: object) -> OrderStatus:
    base: dict[str, object] = {
        "client_order_id": "af-test-1",
        "broker_order_id": "b1",
        "symbol": "AAPL",
        "side": OrderSide.BUY,
        "state": OrderState.ACCEPTED,
        "requested_quantity": Decimal("10"),
        "filled_quantity": Decimal("0"),
        "average_fill_price": None,
        "submitted_at": NOW,
        "updated_at": NOW,
    }
    base.update(overrides)
    return OrderStatus(**base)  # type: ignore[arg-type]


def test_an_overfill_is_refused_as_a_reconciliation_fault() -> None:
    """The broker claiming to have executed more than asked is not a fill."""
    with pytest.raises(BrokerContractError, match="exceeds requested_quantity"):
        _status(
            filled_quantity=Decimal("11"),
            average_fill_price=Decimal("100"),
            state=OrderState.FILLED,
        )


def test_filled_with_an_unfilled_residual_is_refused() -> None:
    """Mislabelling a partial as FILLED hides the residual."""
    with pytest.raises(BrokerContractError, match="partially executed"):
        _status(
            state=OrderState.FILLED,
            filled_quantity=Decimal("5"),
            average_fill_price=Decimal("100"),
        )


def test_a_fill_without_an_average_price_is_refused() -> None:
    with pytest.raises(BrokerContractError, match="average_fill_price"):
        _status(state=OrderState.PARTIALLY_FILLED, filled_quantity=Decimal("5"))


def test_an_update_before_submission_is_refused() -> None:
    with pytest.raises(BrokerContractError, match="precedes submitted_at"):
        _status(updated_at=NOW - timedelta(seconds=1))


def test_unfilled_quantity_is_reported() -> None:
    status = _status(
        state=OrderState.PARTIALLY_FILLED,
        filled_quantity=Decimal("4"),
        average_fill_price=Decimal("100"),
    )
    assert status.unfilled_quantity == Decimal("6")
    assert not status.is_terminal


def test_an_oversized_reason_is_truncated_not_refused() -> None:
    status = _status(state=OrderState.REJECTED, reason="x" * 5_000)
    assert status.reason is not None
    assert len(status.reason) == 512


# ---------------------------------------------------------------------------
# Quotes
# ---------------------------------------------------------------------------


def test_a_crossed_quote_is_refused() -> None:
    """Pricing an order from a crossed book produces a fill that could not occur."""
    with pytest.raises(BrokerContractError, match="crossed quote"):
        Quote(symbol="AAPL", bid=Decimal("101"), ask=Decimal("100"), observed_at=NOW)


def test_quote_age_is_measured_against_a_supplied_now() -> None:
    quote = Quote(symbol="AAPL", bid=Decimal("100"), ask=Decimal("101"), observed_at=NOW)
    assert quote.age_seconds(now=NOW + timedelta(seconds=30)) == 30.0
    assert quote.midpoint == Decimal("100.5")


def test_a_locked_quote_is_permitted() -> None:
    """bid == ask is legal; only ask < bid is a fault."""
    assert Quote(
        symbol="AAPL", bid=Decimal("100"), ask=Decimal("100"), observed_at=NOW
    ).midpoint == Decimal("100")


# ---------------------------------------------------------------------------
# Account snapshot
# ---------------------------------------------------------------------------


def _snapshot(**overrides: object) -> AccountSnapshot:
    base: dict[str, object] = {
        "account_id_digest": DIGEST,
        "cash": Decimal("1000"),
        "equity": Decimal("1000"),
        "buying_power": Decimal("1000"),
        "positions": (),
        "observed_at": NOW,
    }
    base.update(overrides)
    return AccountSnapshot(**base)  # type: ignore[arg-type]


def test_a_raw_account_identifier_is_refused() -> None:
    """Only a digest may be stored; the raw identifier must never persist."""
    with pytest.raises(BrokerContractError, match="SHA-256"):
        _snapshot(account_id_digest="PA3XYZACCOUNT")


def test_duplicate_position_rows_are_refused() -> None:
    duplicate = (
        Position(symbol="AAPL", quantity=Decimal("1"), average_entry_price=Decimal("100")),
        Position(symbol="AAPL", quantity=Decimal("2"), average_entry_price=Decimal("101")),
    )
    with pytest.raises(BrokerContractError, match="duplicate symbol"):
        _snapshot(positions=duplicate)


def test_positions_are_sorted_deterministically() -> None:
    snapshot = _snapshot(
        positions=(
            Position(symbol="MSFT", quantity=Decimal("1"), average_entry_price=Decimal("100")),
            Position(symbol="AAPL", quantity=Decimal("1"), average_entry_price=Decimal("100")),
        )
    )
    assert [item.symbol for item in snapshot.positions] == ["AAPL", "MSFT"]


def test_a_short_position_is_identified_by_sign() -> None:
    short = Position(symbol="AAPL", quantity=Decimal("-5"), average_entry_price=Decimal("100"))
    assert short.is_short


def test_position_lookup_returns_none_when_flat() -> None:
    assert _snapshot().position_for("AAPL") is None


def test_negative_cash_is_permitted_but_negative_buying_power_is_not() -> None:
    """A margin account can hold negative cash; negative buying power is nonsense."""
    assert _snapshot(cash=Decimal("-500")).cash == Decimal("-500")
    with pytest.raises(BrokerContractError, match="buying_power"):
        _snapshot(buying_power=Decimal("-1"))


def test_a_snapshot_serializes_without_the_raw_identifier() -> None:
    payload = _snapshot().to_dict()
    assert payload["account_id_digest"] == DIGEST
    assert payload["is_paper"] is True


# ---------------------------------------------------------------------------
# Market clock
# ---------------------------------------------------------------------------


def test_clock_skew_is_signed_against_local_time() -> None:
    clock = MarketClock(is_open=True, server_time=NOW)
    assert clock.skew_seconds(local_time=NOW + timedelta(seconds=3)) == 3.0
    assert clock.skew_seconds(local_time=NOW - timedelta(seconds=3)) == -3.0


def test_a_closed_clock_may_report_the_next_open() -> None:
    clock = MarketClock(is_open=False, server_time=NOW, next_open=NOW + timedelta(hours=12))
    assert clock.to_dict()["is_open"] is False
    assert clock.next_open is not None
