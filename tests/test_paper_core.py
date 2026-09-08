"""Adversarial money/time, durable ambiguity, reconciliation and causal tests."""

from __future__ import annotations

import sqlite3
from datetime import timedelta
from decimal import Decimal

import pytest
from hypothesis import given
from hypothesis import strategies as st
from pydantic import ValidationError

import signal_foundry.trading.alpaca as alpaca_module
import signal_foundry.trading.engine as engine_module
from signal_foundry.boundary import FoundryError
from signal_foundry.trading.alpaca import Alpaca, order
from signal_foundry.trading.engine import Engine
from signal_foundry.trading.models import Bar, Intent, money, timestamp
from signal_foundry.trading.research import daily_pnl, evaluate, interval, target
from signal_foundry.trading.store import Journal
from tests.paper_helpers import NOW, ROOT, FixedTime, Vendor, config


@pytest.fixture
def setup(tmp_path, monkeypatch):
    vendor = Vendor()
    journal = Journal(tmp_path / "paper")
    engine = Engine(ROOT, journal, config(), Alpaca(vendor))
    monkeypatch.setattr(engine_module, "datetime", FixedTime)
    monkeypatch.setattr(alpaca_module, "datetime", FixedTime)
    # Operational fault injection only. Separate tests exercise the actual
    # qualification worker; no production switch admits this fixture verdict.
    monkeypatch.setattr(
        engine, "qualification", lambda: {"verdict": "QUALIFIED_FOR_PAPER"}
    )
    engine.initialize()
    yield engine, journal, vendor
    journal.close()


@pytest.mark.parametrize(
    "value", [True, 1.0, "NaN", "Infinity", "1e9", "0.000000001", "1" * 19, {}, None]
)
def test_decimal_refuses_ambiguous_or_unbounded_values(value):
    with pytest.raises(ValueError):
        money(value)


@given(st.integers(min_value=-(10**12), max_value=10**12))
def test_decimal_string_roundtrip_is_exact(cents):
    value = Decimal(cents) / 100
    assert money(str(value)) == value


@pytest.mark.parametrize(
    "changes",
    [
        {"enabled": 1},
        {"environment": "live"},
        {"quantity": "0.5"},
        {"maximum_position_notional": "10001"},
        {"max_orders": True},
        {"plan": {"symbols": ["AAA"]}},
        {"account_digest": "account-id"},
        {"url": "https://api.alpaca.markets"},
    ],
)
def test_configuration_fails_closed(changes):
    with pytest.raises(ValidationError):
        config(**changes)


def test_timestamp_normalizes_offset_and_rejects_naive():
    assert timestamp("2026-09-08T11:00:00-04:00") == NOW
    with pytest.raises(ValueError):
        timestamp("2026-09-08T15:00:00")
    with pytest.raises(ValueError):
        timestamp(0)


def test_persisted_ambiguous_acceptance_recovers_without_duplicate(setup):
    engine, journal, vendor = setup
    engine.start()
    vendor.submit_failure = True
    with pytest.raises(FoundryError, match="provider_uncertain"):
        engine.cycle("AAA")
    assert journal.get("state") == "reconciliation_required"
    keys = journal.order_keys()
    assert len(keys) == 1 and journal.get(keys[0])["order"] is None
    vendor.submit_failure = False
    account = engine.reconcile()
    assert account.positions[0].quantity == 1
    assert journal.get("state") == "ready"
    engine.cycle("AAA")
    assert sum(method == "POST" for _, method, *_ in vendor.calls) == 1
    assert engine.reconcile().cash == account.cash
    journal.verify()


def test_unknown_submission_never_retries(setup):
    engine, journal, vendor = setup
    engine.start()
    intent = Intent(
        client_order_id="sf_unknown",
        symbol="AAA",
        side="buy",
        quantity="1",
        limit_price="100",
    )
    journal.set(
        "order/sf_unknown", {"intent": intent.wire(), "order": None}, kind="intent"
    )
    with pytest.raises(FoundryError, match="submission_unknown"):
        engine.reconcile()
    assert not any(method == "POST" for _, method, *_ in vendor.calls)
    assert journal.get("state") == "reconciliation_required"


def test_restart_stop_and_config_binding(setup):
    engine, journal, vendor = setup
    engine.start()
    other = Journal(journal.root)
    try:
        with journal.exclusive():
            with pytest.raises(FoundryError, match="paper_busy"):
                with other.exclusive():
                    pytest.fail("Concurrent operator admitted")
            other.stop()
            other.stop()
        with pytest.raises(FoundryError, match="paper_disabled"):
            engine.cycle("AAA")
        assert engine.status().stopped
        changed = Engine(ROOT, other, config(max_orders=99), Alpaca(vendor))
        with pytest.raises(FoundryError, match="paper_identity"):
            changed.initialize()
    finally:
        other.close()


@pytest.mark.parametrize(
    "fault,code",
    [
        ("cash", "account_divergence"),
        ("position", "account_divergence"),
        ("loss", "loss_limit"),
        ("unknown_order", "external_order"),
    ],
)
def test_reconciliation_detects_breaks(setup, fault, code):
    engine, journal, vendor = setup
    engine.start()
    if fault == "cash":
        vendor.cash += 1
    elif fault == "position":
        vendor.positions["BBB"] = Decimal(1)
    elif fault == "loss":
        baseline = journal.get("baseline")
        baseline["equity"] = "11000"
        journal.set("baseline", baseline, kind="test_fault")
    else:
        vendor.status = "accepted"
        vendor.request(
            "paper",
            "POST",
            "/v2/orders",
            body={
                "client_order_id": "external",
                "symbol": "AAA",
                "qty": "1",
                "side": "buy",
                "type": "limit",
                "time_in_force": "day",
                "limit_price": "100",
            },
        )
    with pytest.raises(FoundryError, match=code):
        engine.reconcile()
    assert journal.get("state") == "reconciliation_required"
    assert bool(journal.get("stopped")) == (fault == "loss")


@pytest.mark.parametrize(
    "fault,code",
    [
        ("closed", "market_session"),
        ("old_quote", "stale_quote"),
        ("future_quote", "stale_quote"),
        ("spread", "spread_limit"),
        ("symbol", "paper_symbol"),
        ("notional", "notional_limit"),
    ],
)
def test_admission_guards_precede_submission(setup, fault, code):
    engine, journal, vendor = setup
    engine.start()
    symbol = "AAA"
    if fault == "closed":
        vendor.clock_open = False
    if fault == "old_quote":
        vendor.quote_at = (NOW - timedelta(seconds=11)).isoformat()
    if fault == "future_quote":
        vendor.quote_at = (NOW + timedelta(seconds=1)).isoformat()
    if fault == "spread":
        vendor.quote_ask = 110
    if fault == "symbol":
        symbol = "ZZZ"
    if fault == "notional":
        engine.config = config(maximum_order_notional="1")
        journal.set("config", engine.config.wire(), kind="test_fault")
    with pytest.raises(FoundryError, match=code):
        engine.cycle(symbol)
    assert not any(method == "POST" for _, method, *_ in vendor.calls)


def test_cancel_only_owned_orders_and_keep_positions(setup):
    engine, journal, vendor = setup
    engine.start()
    vendor.status = "accepted"
    engine.cycle("AAA")
    with pytest.raises(FoundryError, match="working_order"):
        engine.cycle("BBB")
    owned = next(iter(vendor.orders.values()))
    vendor.orders["external"] = {
        **owned,
        "client_order_id": "external",
        "id": "22222222-2222-2222-2222-222222222222",
    }
    engine.cancel()
    assert owned["status"] == "canceled"
    assert vendor.orders["external"]["status"] == "accepted"
    assert journal.get("stopped") is True


def test_state_hash_append_only_and_artifact_corruption(tmp_path):
    journal = Journal(tmp_path / "state")
    journal.set("example", {"safe": True}, kind="test")
    with pytest.raises(sqlite3.IntegrityError):
        journal.connection.execute("DELETE FROM events")
    identity = journal.artifact({"safe": True})
    assert journal.artifact({"safe": True}) == identity
    assert journal.read_artifact(identity) == {"safe": True}
    (journal.root / "artifacts" / f"{identity}.json").write_bytes(b"{}")
    with pytest.raises(FoundryError, match="paper_integrity"):
        journal.read_artifact(identity)
    with pytest.raises(FoundryError, match="paper_integrity"):
        journal.read_artifact("../wrong")
    journal.connection.execute("UPDATE state SET value=? WHERE key='example'", (b"{}",))
    with pytest.raises(FoundryError, match="paper_integrity"):
        journal.get("example")
    journal.close()


def test_audit_chain_corruption_and_rate_budget(tmp_path):
    journal = Journal(tmp_path / "state")
    for _ in range(100):
        journal.request(1000)
    with pytest.raises(FoundryError, match="provider_rate"):
        journal.request(1000)
    with pytest.raises(FoundryError, match="clock_rollback"):
        journal.request(999)
    journal.request(1060)
    journal.append("test", {})
    journal.connection.execute("DROP TRIGGER no_update")
    journal.connection.execute("UPDATE events SET previous=?", ("f" * 64,))
    with pytest.raises(FoundryError, match="paper_integrity"):
        journal.verify()
    journal.close()
    with pytest.raises(FoundryError, match="paper_store"):
        Journal(tmp_path / "state")


def bars():
    start = NOW - timedelta(minutes=10)
    return tuple(
        Bar(
            at=start + timedelta(minutes=i),
            open=str(100 + i),
            high=str(102 + i),
            low=str(99 + i),
            close=str(101 + i),
            volume="100",
        )
        for i in range(10)
    )


def test_causal_next_open_costs_and_prefix_invariance():
    plan = config().plan.model_copy(
        update={
            "start": NOW - timedelta(days=1),
            "selection_end": NOW - timedelta(hours=1),
            "end": NOW + timedelta(days=1),
        }
    )
    rows = bars()
    baseline = daily_pnl(rows, plan, "momentum", 0)
    expensive = daily_pnl(rows, plan, "momentum", 100)
    assert sum(r["gross"] for r in baseline.values()) == sum(
        r["gross"] for r in expensive.values()
    )
    assert sum(r["gross"] for r in baseline.values()) == 7
    appended = rows + tuple(
        b.model_copy(update={"at": b.at + timedelta(days=1)}) for b in rows
    )
    assert (
        daily_pnl(appended, plan, "momentum", 0)["2026-09-08"] == baseline["2026-09-08"]
    )
    assert sum(r["net"] for r in expensive.values()) < sum(
        r["net"] for r in baseline.values()
    )
    assert (
        target(tuple(b.close for b in rows[:3]), candidate="momentum", threshold_bps=10)
        == 1
    )
    assert (
        target(
            tuple(b.close for b in rows[:3]),
            candidate="mean_reversion",
            threshold_bps=10,
        )
        == 0
    )
    result = evaluate(rows, plan)
    assert len(result["records"]) == 6 and result["decision"] == "NO_GO"
    assert result["selected_on_pretest_dates"] is None
    assert all(row["net_interval"] is None for row in result["records"])
    with pytest.raises(FoundryError, match="research_order"):
        evaluate(rows[::-1], plan)
    with pytest.raises(FoundryError, match="research_split"):
        evaluate(rows, config().plan)
    assert interval([0.1] * 20, 1) == pytest.approx((0.1, 0.1))
    with pytest.raises(FoundryError, match="research_sample"):
        interval([float("nan")] * 20, 1)


def test_bar_semantics_and_order_mapping():
    with pytest.raises(ValidationError):
        Bar(at=NOW, open="10", high="9", low="8", close="10", volume="1")
    vendor = Vendor()
    alpaca = Alpaca(vendor)
    intent = Intent(
        client_order_id="one", symbol="AAA", side="buy", quantity="1", limit_price="100"
    )
    current = alpaca.submit(intent)
    assert current.filled_quantity == 1
    changed = intent.model_copy(
        update={"client_order_id": "two", "limit_price": Decimal("99")}
    )
    assert alpaca.replace(current, changed).intent == changed
    with pytest.raises(FoundryError, match="replacement_contract"):
        alpaca.replace(current, changed.model_copy(update={"symbol": "BBB"}))
    with pytest.raises(FoundryError, match="unsupported_order"):
        order({"type": "market"})


def test_vendor_pagination_refuses_duplicate_and_out_of_range():
    vendor = Vendor()
    vendor.page = {"bars": [], "next_page_token": "repeated"}
    with pytest.raises(FoundryError, match="data_pagination"):
        Alpaca(vendor).bars("AAA", "iex", NOW - timedelta(minutes=3), NOW)
    vendor.page = {"bars": [], "next_page_token": None}
    with pytest.raises(FoundryError, match="data_order"):
        Alpaca(vendor).bars("AAA", "iex", NOW - timedelta(minutes=3), NOW)
