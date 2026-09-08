"""Session cadence, malformed provider state and capacity regression evidence."""

from datetime import timedelta
from decimal import Decimal
from types import SimpleNamespace

import pytest
from pydantic import ValidationError

import signal_foundry.trading.session as session_module
from signal_foundry.boundary import FoundryError, encode
from signal_foundry.trading.models import Order, Plan, Quote, timestamp
from signal_foundry.trading.service import Action, unavailable
from signal_foundry.trading.session import run_session
from signal_foundry.trading.store import Journal
from tests.paper_helpers import NOW, config
from tests.test_paper_core import setup as setup
from tests.test_paper_integration import configured as configured
from tests.test_paper_transport import wire as wire


def test_session_cadence_is_bounded_and_never_retries(monkeypatch):
    clock = [0.0]
    calls = []
    sleeps = []

    def action(value):
        calls.append(value)
        clock[0] += 2

    def sleep(seconds):
        sleeps.append(seconds)
        clock[0] += seconds

    service = SimpleNamespace(config=config(), action=action, status=unavailable)
    monkeypatch.setattr(session_module.time, "monotonic", lambda: clock[0])
    monkeypatch.setattr(session_module.time, "sleep", sleep)
    assert not run_session(service, 2).status.live_authorized
    assert [c.symbol for c in calls] == ["AAA", "BBB", "CCC"] * 2
    assert sleeps == [54]
    for invalid in (0, 391, True):
        with pytest.raises(FoundryError, match="session_budget"):
            run_session(service, invalid)
    assert len(calls) == 6


@pytest.mark.parametrize("failure", ["overrun", "uncertain", "interrupt"])
def test_session_failure_persists_stop_without_retry(monkeypatch, failure):
    calls = []
    clock = iter([0, 61])
    monkeypatch.setattr(session_module.time, "monotonic", lambda: next(clock))

    def action(value):
        calls.append(value.operation)
        if value.operation == "cycle":
            if failure == "uncertain":
                raise FoundryError("submission_unknown", "Test uncertainty")
            if failure == "interrupt":
                raise KeyboardInterrupt

    service = SimpleNamespace(config=config(), action=action, status=unavailable)
    with pytest.raises((FoundryError, KeyboardInterrupt)):
        run_session(service, 2)
    assert calls[-1] == "stop"
    assert calls.count("cycle") == (3 if failure == "overrun" else 1)


def test_stop_is_rechecked_at_transport_dispatch(wire):
    transport, connection, _ = wire
    transport.journal.stop()
    with pytest.raises(FoundryError, match="paper_disabled"):
        transport.request("paper", "POST", "/v2/orders", body={})
    assert connection.calls == []


def test_artifact_replay_remains_available_at_capacity(tmp_path):
    journal = Journal(tmp_path / "paper")
    try:
        identity = journal.artifact({"first": True})
        for index in range(255):
            journal.artifact({"index": index})
        assert journal.artifact({"first": True}) == identity
        with pytest.raises(FoundryError, match="paper_capacity"):
            journal.artifact({"extra": True})
    finally:
        journal.close()


@pytest.mark.parametrize("payload", [[], {}, {"result": {}, "error": {}}])
def test_worker_envelope_is_closed(configured, monkeypatch, payload):
    monkeypatch.setattr(
        "signal_foundry.trading.service.execute",
        lambda *args, **kwargs: encode(payload),
    )
    with pytest.raises(FoundryError, match="paper_envelope"):
        configured.action(Action(operation="initialize"))


def test_partial_fill_cash_reconciliation_and_cancellation(setup):
    engine, journal, vendor = setup
    engine.start()
    vendor.status = "new"
    engine.cycle("AAA")
    order = next(iter(vendor.orders.values()))
    order.update(status="partially_filled", filled_qty="0.5", filled_avg_price="101.01")
    vendor.positions["AAA"] = Decimal("0.5")
    vendor.cash -= Decimal("50.505")
    first = engine.reconcile()
    assert engine.reconcile().cash == first.cash == Decimal("9949.495")
    assert first.positions[0].quantity == Decimal("0.5")
    engine.cancel()
    assert engine.reconcile().positions[0].quantity == Decimal("0.5")
    assert journal.get("stopped") and order["status"] == "canceled"
    parsed = engine.broker.lookup(order["client_order_id"])
    for quantity, price in (("0", None), ("1", "101.01"), ("2", "101.01")):
        value = parsed.wire()
        value.update(
            state="partially_filled", filled_quantity=quantity, average_price=price
        )
        with pytest.raises(ValidationError):
            Order.model_validate(value)


@pytest.mark.parametrize("fault", ["empty", "duplicate", "inverted", "long"])
def test_official_session_rejects_malformed_calendar(setup, fault):
    engine, _, vendor = setup
    if fault == "empty":
        vendor.calendar = []
    elif fault == "duplicate":
        vendor.calendar *= 2
    elif fault == "inverted":
        vendor.calendar[0]["close"] = "09:00"
    else:
        vendor.calendar[0]["close"] = "20:00"
    with pytest.raises(FoundryError, match="market_session|calendar_contract"):
        engine.session("2026-09-08")


def test_model_boundary_rejects_ambiguous_ranges():
    with pytest.raises(ValueError, match="UTC range"):
        timestamp("0001-01-01T00:00:00+01:00")
    value = config().plan.wire()
    for change in (
        {"symbols": ["AAA"] * 3},
        {"end": value["start"]},
        {"end": "2028-01-01T00:00:00Z"},
    ):
        with pytest.raises(ValidationError):
            Plan.model_validate({**value, **change})
    with pytest.raises(ValidationError, match="Crossed"):
        Quote(at=NOW, bid="2", ask="1", bid_size="1", ask_size="1")


def test_campaign_future_clock_is_not_evidence(setup):
    engine, journal, _ = setup
    journal.set("created", (NOW + timedelta(days=1)).isoformat(), kind="test_future")
    with pytest.raises(FoundryError, match="campaign_clock"):
        engine.campaign()


@pytest.mark.parametrize(
    "fault,code",
    [
        ("window", "feature_freshness"),
        ("gap", "feature_gap"),
        ("session", "feature_session"),
        ("account", "account_identity"),
        ("budget", "order_limit"),
        ("expiry", "admission_expired"),
    ],
)
def test_boundary_changes_never_cross_submission(setup, monkeypatch, fault, code):
    engine, journal, vendor = setup
    engine.start()
    if fault == "account":
        engine.config = config(account_digest="f" * 64)
        journal.set("config", engine.config.wire(), kind="test_fault")
    elif fault == "budget":
        engine.config = config(max_orders=1)
        journal.set("config", engine.config.wire(), kind="test_fault")
        engine.cycle("BBB")
        vendor.calls.clear()
    elif fault == "expiry":
        prior = engine.account
        monkeypatch.setattr(
            engine,
            "account",
            lambda: prior().model_copy(
                update={"observed_at": NOW - timedelta(seconds=16)}
            ),
        )
    else:
        original = engine.broker.bars

        def bars(*args, **kwargs):
            values, pages = original(*args, **kwargs)
            if fault == "window":
                values = values[:-1]
            elif fault == "gap":
                values = (
                    values[0].model_copy(
                        update={"at": values[0].at - timedelta(minutes=1)}
                    ),
                    *values[1:],
                )
            else:
                values = (
                    values[0].model_copy(update={"at": NOW.replace(hour=13, minute=0)}),
                    *values[1:],
                )
            return values, pages

        monkeypatch.setattr(engine.broker, "bars", bars)
    with pytest.raises(FoundryError, match=code):
        engine.cycle("AAA")
    assert not any(method == "POST" for _, method, *_ in vendor.calls)


def test_baseline_and_no_order_paths_remain_explicit(setup):
    engine, journal, vendor = setup
    with pytest.raises(FoundryError, match="paper_baseline"):
        engine.reconcile()
    vendor.positions["AAA"] = Decimal("1")
    with pytest.raises(FoundryError, match="paper_baseline"):
        engine.start()
    vendor.positions.clear()
    engine.start()
    engine.start()
    engine.cycle("AAA")
    journal.set("decision/AAA", None, kind="test_next_decision")
    engine.cycle("AAA")
    assert sum(method == "POST" for _, method, *_ in vendor.calls) == 1
    journal.set("last_error", "test_failure", kind="test_fault")
    assert "Last operation failed: test_failure" in engine.status().blockers


def test_provider_changed_intent_is_not_accepted(setup):
    engine, journal, vendor = setup
    engine.start()
    vendor.status = "new"
    engine.cycle("AAA")
    order = next(iter(vendor.orders.values()))
    order["qty"] = "2"
    with pytest.raises(FoundryError, match="order_divergence"):
        engine.reconcile()
    assert journal.get("state") == "reconciliation_required"


def test_private_audit_pages_preserve_sequence_and_chain(setup):
    _, journal, _ = setup
    first = journal.audit(limit=2)
    second = journal.audit(after=first["next_after"], limit=2)
    assert [row["sequence"] for row in first["events"] + second["events"]] == [
        1,
        2,
        3,
        4,
    ]
    assert second["events"][0]["previous"] == first["events"][-1]["digest"]
    assert journal.audit(after=journal.event_count())["events"] == []
    for after, limit in ((-1, 1), (True, 1), (0, 257), (0, False)):
        with pytest.raises(FoundryError, match="audit_cursor"):
            journal.audit(after, limit)


def test_provider_inclusive_end_cannot_admit_current_minute(setup):
    engine, _, vendor = setup
    start = NOW - timedelta(minutes=3)
    bars, _ = engine.broker.bars("AAA", "iex", start, NOW)
    query = vendor.calls[-1][3]
    assert timestamp(query["end"]) == NOW - timedelta(microseconds=1)
    assert query["asof"] == "-"
    assert [bar.at for bar in bars] == [start + timedelta(minutes=i) for i in range(3)]
