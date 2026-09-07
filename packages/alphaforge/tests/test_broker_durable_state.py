"""Fault-injection and recovery tests for durable session state (SF-S5-MR4).

Grouped by the guarantee each protects. The ones carrying the most weight:

* **A restart cannot resubmit an order.** This is the whole point: MR3's
  idempotency lived in a dict that a restart empties.
* **Recovery refuses rather than guesses** — tampered, truncated, gapped,
  schema-incompatible, foreign-strategy, stale, and clock-rolled-back snapshots
  all raise.
* **A crash mid-write leaves the previous snapshot intact**, never a half one.
* **Reconciliation halts and never repairs**, because a repair acts on exactly
  the state known to be wrong.
* **Fills converge regardless of arrival order or duplication.**
"""

from __future__ import annotations

import json
import sqlite3
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from pathlib import Path
from typing import Any, cast

import pytest

from alphaforge.broker import (
    AccountSnapshot,
    ClockRollbackError,
    DivergenceKind,
    DurableStateError,
    Fill,
    OrderIntent,
    OrderSide,
    OrderState,
    OrderStatus,
    Position,
    ReconciliationError,
    SessionStateStore,
    SnapshotIncompatibleError,
    SnapshotIntegrityError,
    SnapshotStaleError,
    apply_fills_idempotently,
    intents_from_orders,
    reconcile,
)

NOW = datetime(2026, 8, 6, 15, 0, tzinfo=UTC)
DIGEST = "a" * 64
STRATEGY = "sf-s5-candidate"
CONFIG = "c" * 64


@pytest.fixture
def store_path(tmp_path: Path) -> Path:
    return tmp_path / "session.sqlite3"


@pytest.fixture
def store(store_path: Path) -> Any:
    with SessionStateStore(store_path) as opened:
        yield opened


def _intent(cid: str = "af-order-1", state: str = "accepted", **kw: Any) -> OrderIntent:
    base: dict[str, Any] = {
        "client_order_id": cid,
        "decision_id": "decision-1",
        "symbol": "AAPL",
        "side": "buy",
        "quantity": "10",
        "submitted_state": state,
        "recorded_at": NOW,
    }
    base.update(kw)
    return OrderIntent(**base)


def _position(symbol: str = "AAPL", quantity: str = "10", price: str = "100") -> Position:
    return Position(symbol=symbol, quantity=Decimal(quantity), average_entry_price=Decimal(price))


def _append(store: SessionStateStore, **kw: Any) -> Any:
    base: dict[str, Any] = {
        "strategy_id": STRATEGY,
        "model_version": "m1",
        "config_identity": CONFIG,
        "cash": Decimal("99000"),
        "positions": (_position(),),
        "intents": (_intent(),),
        "written_at": NOW,
    }
    base.update(kw)
    return store.append(**base)


def _recover(store: SessionStateStore, **kw: Any) -> Any:
    base: dict[str, Any] = {
        "now": NOW,
        "expected_strategy_id": STRATEGY,
        "expected_config_identity": CONFIG,
    }
    base.update(kw)
    return store.recover(**base)


# ---------------------------------------------------------------------------
# The core guarantee: a restart cannot resubmit
# ---------------------------------------------------------------------------


def test_a_restart_recovers_the_orders_already_sent(store_path: Path) -> None:
    """MR3's idempotency lived in a dict; a restart emptied it. This is the fix."""
    with SessionStateStore(store_path) as first:
        _append(first, intents=(_intent("af-order-1"), _intent("af-order-2")))

    with SessionStateStore(store_path) as second:
        recovered = _recover(second)
        assert recovered is not None
        assert recovered.intent_for("af-order-1") is not None
        assert recovered.intent_for("af-order-2") is not None
        assert recovered.intent_for("af-order-never-sent") is None


def test_an_intent_is_recorded_before_the_broker_is_contacted(store: SessionStateStore) -> None:
    """The crash window between decided and acknowledged must leave evidence."""
    snapshot = _append(store, intents=(_intent("af-order-1", state="pending_new"),))
    recorded = snapshot.intent_for("af-order-1")
    assert recorded is not None
    assert recorded.submitted_state == "pending_new"
    assert recorded.broker_order_id is None
    assert not recorded.is_terminal


def test_open_intents_exclude_terminal_orders(store: SessionStateStore) -> None:
    snapshot = _append(
        store,
        intents=(
            _intent("af-open-1", state="accepted"),
            _intent("af-done-1", state="filled"),
            _intent("af-dead-1", state="rejected"),
        ),
    )
    assert {item.client_order_id for item in snapshot.open_intents()} == {"af-open-1"}


def test_duplicate_client_order_ids_in_one_snapshot_are_refused(
    store: SessionStateStore,
) -> None:
    with pytest.raises(DurableStateError, match="duplicate client_order_id"):
        _append(store, intents=(_intent("af-dup"), _intent("af-dup")))


# ---------------------------------------------------------------------------
# Atomicity and the hash chain
# ---------------------------------------------------------------------------


def test_snapshots_chain_to_their_predecessor(store: SessionStateStore) -> None:
    first = _append(store)
    second = _append(store, written_at=NOW + timedelta(minutes=1))
    assert first.previous_hash == "0" * 64
    assert second.previous_hash == first.content_hash
    assert second.sequence == first.sequence + 1
    assert store.verify_chain() == 2


def test_a_tampered_payload_is_detected(store_path: Path) -> None:
    with SessionStateStore(store_path) as opened:
        _append(opened)
    # Simulate an edit that preserves structure but changes a value.
    connection = sqlite3.connect(store_path)
    row = connection.execute("SELECT sequence, payload FROM snapshots").fetchone()
    payload = json.loads(row[1])
    payload["cash"] = "999999"
    connection.execute(
        "UPDATE snapshots SET payload = ? WHERE sequence = ?",
        (json.dumps(payload, sort_keys=True, separators=(",", ":")), row[0]),
    )
    connection.commit()
    connection.close()

    with (
        SessionStateStore(store_path) as reopened,
        pytest.raises(SnapshotIntegrityError, match="integrity check"),
    ):
        _recover(reopened)


def test_a_truncated_payload_is_not_partially_usable(store_path: Path) -> None:
    with SessionStateStore(store_path) as opened:
        _append(opened)
    connection = sqlite3.connect(store_path)
    row = connection.execute("SELECT sequence, payload FROM snapshots").fetchone()
    payload = json.loads(row[1])
    del payload["positions"]
    connection.execute(
        "UPDATE snapshots SET payload = ? WHERE sequence = ?",
        (json.dumps(payload, sort_keys=True, separators=(",", ":")), row[0]),
    )
    connection.commit()
    connection.close()

    with (
        SessionStateStore(store_path) as reopened,
        pytest.raises(SnapshotIntegrityError, match="missing"),
    ):
        _recover(reopened)


def test_a_removed_middle_snapshot_breaks_the_chain(store_path: Path) -> None:
    """A deleted record is detectable, not merely a modified one."""
    with SessionStateStore(store_path) as opened:
        _append(opened)
        _append(opened, written_at=NOW + timedelta(minutes=1))
        _append(opened, written_at=NOW + timedelta(minutes=2))
    connection = sqlite3.connect(store_path)
    connection.execute("DELETE FROM snapshots WHERE sequence = 2")
    connection.commit()
    connection.close()

    with (
        SessionStateStore(store_path) as reopened,
        pytest.raises(SnapshotIntegrityError, match="gap"),
    ):
        _recover(reopened)


class _FailingConnection:
    """Delegates to a real connection but fails one statement prefix.

    A proxy rather than a monkeypatch because ``sqlite3.Connection.execute`` is
    a read-only C attribute and cannot be replaced in place.
    """

    def __init__(self, real: sqlite3.Connection, fail_prefix: str) -> None:
        self._real = real
        self._fail_prefix = fail_prefix

    def execute(self, sql: str, *args: Any, **kwargs: Any) -> Any:
        if sql.startswith(self._fail_prefix):
            raise sqlite3.OperationalError("disk I/O error")
        return self._real.execute(sql, *args, **kwargs)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._real, name)


def test_a_failed_write_leaves_the_previous_head_intact(store: SessionStateStore) -> None:
    """A crash mid-write must not produce a half-updated head."""
    good = _append(store)

    real = store._connection  # noqa: SLF001
    store._connection = _FailingConnection(real, "INSERT INTO snapshots")  # type: ignore[assignment]  # noqa: SLF001
    try:
        with pytest.raises(sqlite3.OperationalError):
            _append(store, written_at=NOW + timedelta(minutes=1))
    finally:
        store._connection = real  # noqa: SLF001

    recovered = _recover(store)
    assert recovered is not None
    assert recovered.sequence == good.sequence
    assert recovered.content_hash == good.content_hash
    assert store.verify_chain() == 1


def test_an_empty_store_recovers_to_none(store: SessionStateStore) -> None:
    assert _recover(store) is None
    assert store.latest_sequence() == 0


# ---------------------------------------------------------------------------
# Recovery refusals
# ---------------------------------------------------------------------------


def test_a_stale_snapshot_cannot_authorize_action(store: SessionStateStore) -> None:
    _append(store)
    with pytest.raises(SnapshotStaleError, match="cannot authorize"):
        _recover(store, now=NOW + timedelta(days=3))


def test_a_snapshot_within_the_age_bound_recovers(store: SessionStateStore) -> None:
    _append(store)
    assert _recover(store, now=NOW + timedelta(hours=1)) is not None


def test_a_clock_rollback_is_refused_on_recovery(store: SessionStateStore) -> None:
    """Wall clocks move backwards on NTP correction; staleness becomes unjudgeable."""
    _append(store)
    with pytest.raises(ClockRollbackError, match="precedes"):
        _recover(store, now=NOW - timedelta(hours=1))


def test_a_clock_rollback_is_refused_on_append(store: SessionStateStore) -> None:
    _append(store, written_at=NOW)
    with pytest.raises(ClockRollbackError, match="moved backwards"):
        _append(store, written_at=NOW - timedelta(minutes=5))


def test_another_strategys_snapshot_is_refused(store: SessionStateStore) -> None:
    _append(store)
    with pytest.raises(SnapshotIncompatibleError, match="belongs to strategy"):
        _recover(store, expected_strategy_id="a-different-strategy")


def test_a_snapshot_from_a_different_configuration_is_refused(
    store: SessionStateStore,
) -> None:
    """The configuration that produced these positions is not the one about to act."""
    _append(store)
    with pytest.raises(SnapshotIncompatibleError, match="configuration"):
        _recover(store, expected_config_identity="d" * 64)


def test_an_incompatible_schema_version_is_refused(store_path: Path) -> None:
    with SessionStateStore(store_path) as opened:
        _append(opened)
    connection = sqlite3.connect(store_path)
    row = connection.execute("SELECT sequence, payload FROM snapshots").fetchone()
    payload = json.loads(row[1])
    payload["schema_version"] = 99
    connection.execute(
        "UPDATE snapshots SET payload = ? WHERE sequence = ?",
        (json.dumps(payload, sort_keys=True, separators=(",", ":")), row[0]),
    )
    connection.commit()
    connection.close()

    with (
        SessionStateStore(store_path) as reopened,
        pytest.raises(SnapshotIncompatibleError, match="schema version"),
    ):
        _recover(reopened)


def test_a_non_positive_max_age_is_refused(store: SessionStateStore) -> None:
    _append(store)
    with pytest.raises(DurableStateError, match="max_age"):
        _recover(store, max_age=timedelta(0))


def test_a_closed_store_refuses_every_operation(store_path: Path) -> None:
    opened = SessionStateStore(store_path)
    opened.close()
    opened.close()  # idempotent
    with pytest.raises(DurableStateError, match="closed"):
        opened.latest_sequence()


# ---------------------------------------------------------------------------
# Path hardening
# ---------------------------------------------------------------------------


def test_a_symlinked_store_path_is_refused(tmp_path: Path) -> None:
    real = tmp_path / "real.sqlite3"
    real.write_bytes(b"")
    link = tmp_path / "link.sqlite3"
    link.symlink_to(real)
    with pytest.raises(DurableStateError, match="symlink"):
        SessionStateStore(link)


def test_a_group_readable_store_is_refused(tmp_path: Path) -> None:
    path = tmp_path / "loose.sqlite3"
    path.write_bytes(b"")
    path.chmod(0o644)
    with pytest.raises(DurableStateError, match="owner-only"):
        SessionStateStore(path)


def test_a_hard_linked_store_is_refused(tmp_path: Path) -> None:
    path = tmp_path / "state.sqlite3"
    path.write_bytes(b"")
    path.chmod(0o600)
    (tmp_path / "alias.sqlite3").hardlink_to(path)
    with pytest.raises(DurableStateError, match="hard-link"):
        SessionStateStore(path)


# ---------------------------------------------------------------------------
# Field validation
# ---------------------------------------------------------------------------


def test_persisted_money_refuses_a_float(store: SessionStateStore) -> None:
    """A float would not read back as what was written."""
    with pytest.raises(DurableStateError, match="float"):
        _append(store, cash=99000.55)


def test_an_unknown_order_state_is_refused() -> None:
    with pytest.raises(DurableStateError, match="unknown order state"):
        _intent(state="teleported")


def test_an_invalid_side_is_refused() -> None:
    with pytest.raises(DurableStateError, match="side must be"):
        _intent(side="sideways")


def test_an_oversized_identifier_is_refused(store: SessionStateStore) -> None:
    with pytest.raises(DurableStateError, match="exceeds"):
        _append(store, strategy_id="x" * 300)


# ---------------------------------------------------------------------------
# Reconciliation: halts, never repairs
# ---------------------------------------------------------------------------


def _account(**kw: Any) -> AccountSnapshot:
    base: dict[str, Any] = {
        "account_id_digest": DIGEST,
        "cash": Decimal("99000"),
        "equity": Decimal("100000"),
        "buying_power": Decimal("99000"),
        "positions": (_position(),),
        "observed_at": NOW,
    }
    base.update(kw)
    return AccountSnapshot(**base)


def _order_status(cid: str = "af-order-1", state: OrderState = OrderState.ACCEPTED) -> OrderStatus:
    return OrderStatus(
        client_order_id=cid,
        broker_order_id="b1",
        symbol="AAPL",
        side=OrderSide.BUY,
        state=state,
        requested_quantity=Decimal("10"),
        filled_quantity=Decimal("0"),
        average_fill_price=None,
        submitted_at=NOW,
        updated_at=NOW,
    )


def test_matching_state_reconciles(store: SessionStateStore) -> None:
    snapshot = _append(store)
    report = reconcile(snapshot, _account(), {"af-order-1": _order_status()}, checked_at=NOW)
    assert report.reconciled
    assert not report.must_halt
    assert report.divergences == ()


def test_a_quantity_difference_halts_and_does_not_repair(store: SessionStateStore) -> None:
    """The book is not overwritten; the difference is reported for a human."""
    snapshot = _append(store)
    report = reconcile(
        snapshot,
        _account(positions=(_position(quantity="15"),)),
        {"af-order-1": _order_status()},
        checked_at=NOW,
    )
    assert report.must_halt
    assert DivergenceKind.POSITION_QUANTITY in report.kinds()
    # The local snapshot is untouched.
    assert snapshot.positions[0].quantity == Decimal("10")
    assert "never repairs" in report.to_dict()["repair_policy"]
    assert "do not liquidate" in report.to_dict()["action"].lower()


def test_a_position_only_the_broker_reports_halts(store: SessionStateStore) -> None:
    snapshot = _append(store)
    report = reconcile(
        snapshot,
        _account(positions=(_position(), _position(symbol="MSFT", quantity="5"))),
        {"af-order-1": _order_status()},
        checked_at=NOW,
    )
    assert report.must_halt
    assert DivergenceKind.POSITION_ONLY_BROKER in report.kinds()


def test_a_position_only_the_system_believes_in_halts(store: SessionStateStore) -> None:
    snapshot = _append(store)
    report = reconcile(
        snapshot, _account(positions=()), {"af-order-1": _order_status()}, checked_at=NOW
    )
    assert report.must_halt
    assert DivergenceKind.POSITION_ONLY_LOCAL in report.kinds()


def test_a_cash_difference_beyond_tolerance_halts(store: SessionStateStore) -> None:
    snapshot = _append(store)
    report = reconcile(
        snapshot,
        _account(cash=Decimal("98000")),
        {"af-order-1": _order_status()},
        checked_at=NOW,
    )
    assert report.must_halt
    assert DivergenceKind.CASH in report.kinds()


def test_a_rounding_sized_cash_difference_is_tolerated(store: SessionStateStore) -> None:
    snapshot = _append(store)
    report = reconcile(
        snapshot,
        _account(cash=Decimal("99000.005")),
        {"af-order-1": _order_status()},
        checked_at=NOW,
    )
    assert report.reconciled


def test_an_order_the_broker_never_heard_of_halts(store: SessionStateStore) -> None:
    """Resubmitting without confirming would risk a duplicate."""
    snapshot = _append(store)
    report = reconcile(snapshot, _account(), {}, checked_at=NOW)
    assert report.must_halt
    assert DivergenceKind.ORDER_ONLY_LOCAL in report.kinds()
    assert "lost in flight" in report.divergences[0].interpretation


def test_a_working_order_the_system_does_not_know_about_halts(
    store: SessionStateStore,
) -> None:
    snapshot = _append(store)
    report = reconcile(
        snapshot,
        _account(),
        {"af-order-1": _order_status(), "af-ghost": _order_status("af-ghost")},
        checked_at=NOW,
    )
    assert report.must_halt
    assert DivergenceKind.ORDER_ONLY_BROKER in report.kinds()


def test_a_terminal_order_the_system_recorded_as_open_halts(
    store: SessionStateStore,
) -> None:
    """The order settled while the system still believes it is working."""
    snapshot = _append(store)
    filled = OrderStatus(
        client_order_id="af-order-1",
        broker_order_id="b1",
        symbol="AAPL",
        side=OrderSide.BUY,
        state=OrderState.FILLED,
        requested_quantity=Decimal("10"),
        filled_quantity=Decimal("10"),
        average_fill_price=Decimal("100"),
        submitted_at=NOW,
        updated_at=NOW,
    )
    report = reconcile(snapshot, _account(), {"af-order-1": filled}, checked_at=NOW)
    assert report.must_halt
    assert DivergenceKind.ORDER_STATE in report.kinds()
    assert "fill may be missing" in report.divergences[0].interpretation


def test_a_report_cannot_claim_reconciled_while_diverged() -> None:
    from alphaforge.broker.reconciliation import Divergence, ReconciliationReport

    with pytest.raises(ReconciliationError, match="cannot be marked reconciled"):
        ReconciliationReport(
            reconciled=True,
            checked_at=NOW,
            divergences=(
                Divergence(
                    kind=DivergenceKind.CASH,
                    subject="cash",
                    local_value="1",
                    broker_value="2",
                    interpretation="x",
                ),
            ),
            positions_compared=0,
            orders_compared=0,
        )


def test_a_float_cash_tolerance_is_refused(store: SessionStateStore) -> None:
    snapshot = _append(store)
    with pytest.raises(ReconciliationError, match="not a float"):
        reconcile(snapshot, _account(), {}, checked_at=NOW, cash_tolerance=cast(Decimal, 0.01))


# ---------------------------------------------------------------------------
# Out-of-order and duplicate fills
# ---------------------------------------------------------------------------


def _fill(
    fill_id: str,
    symbol: str = "AAPL",
    side: OrderSide = OrderSide.BUY,
    qty: str = "5",
    price: str = "100",
) -> Fill:
    return Fill(
        client_order_id="af-order-1",
        symbol=symbol,
        side=side,
        quantity=Decimal(qty),
        price=Decimal(price),
        filled_at=NOW,
        fill_id=fill_id,
    )


def test_duplicate_fills_count_once() -> None:
    """A redelivered acknowledgement must not double the position."""
    single = apply_fills_idempotently([_fill("f1")])
    doubled = apply_fills_idempotently([_fill("f1"), _fill("f1"), _fill("f1")])
    assert single == doubled
    assert single[0].quantity == Decimal("5")


def test_fills_converge_regardless_of_arrival_order() -> None:
    fills = [_fill("f1"), _fill("f2", qty="3"), _fill("f3", side=OrderSide.SELL, qty="2")]
    forward = apply_fills_idempotently(fills)
    reverse = apply_fills_idempotently(list(reversed(fills)))
    assert forward == reverse
    assert forward[0].quantity == Decimal("6")


def test_a_reused_fill_id_with_different_content_is_refused() -> None:
    """Identity is what makes deduplication safe; reuse makes it unsafe."""
    with pytest.raises(ReconciliationError, match="share fill_id"):
        apply_fills_idempotently([_fill("f1", qty="5"), _fill("f1", qty="9")])


def test_a_closing_fill_removes_the_symbol() -> None:
    """A flat symbol left in the book reconciles as a spurious divergence."""
    book = apply_fills_idempotently(
        [_fill("f1", qty="5"), _fill("f2", side=OrderSide.SELL, qty="5")]
    )
    assert book == ()


def test_fills_apply_on_top_of_an_opening_book() -> None:
    book = apply_fills_idempotently([_fill("f1", qty="5")], opening=(_position(quantity="10"),))
    assert book[0].quantity == Decimal("15")


def test_a_non_fill_element_is_refused() -> None:
    with pytest.raises(ReconciliationError, match="must be a Fill"):
        apply_fills_idempotently(["not-a-fill"])  # type: ignore[list-item]


def test_intents_project_from_broker_orders() -> None:
    intents = intents_from_orders(
        {"af-order-1": _order_status()}, decision_id="recovery-1", recorded_at=NOW
    )
    assert len(intents) == 1
    assert intents[0].client_order_id == "af-order-1"
    assert intents[0].broker_order_id == "b1"


# ---------------------------------------------------------------------------
# Determinism and bounds
# ---------------------------------------------------------------------------


def test_the_same_state_produces_the_same_hash(tmp_path: Path) -> None:
    hashes = []
    for name in ("a.sqlite3", "b.sqlite3"):
        with SessionStateStore(tmp_path / name) as opened:
            hashes.append(_append(opened).content_hash)
    assert hashes[0] == hashes[1]


def test_snapshot_serializes_to_json(store: SessionStateStore) -> None:
    payload = _append(store).to_dict()
    assert json.loads(json.dumps(payload))
    assert payload["content_hash"]


def test_the_report_serializes_to_json(store: SessionStateStore) -> None:
    snapshot = _append(store)
    report = reconcile(snapshot, _account(positions=()), {}, checked_at=NOW)
    assert json.loads(json.dumps(report.to_dict()))
