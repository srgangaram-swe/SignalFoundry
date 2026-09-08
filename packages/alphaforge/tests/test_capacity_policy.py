"""Tests for point-in-time borrow, liquidity, and capacity policy (SF-S4-MR5).

Grouped by acceptance criterion. The invariants that carry the most weight:

* **Absence is never permission** — every missing, stale, expired, unknown, or
  conflicting record must yield zero new-short capacity.
* **Capacity is conserved** — reserved + consumed + released + rejected equals
  requested, under partial fills, cancellations, and replay.
* **A cover is never blocked by borrow** — the risk-reducing direction stays open
  even when every borrow signal says stop.
"""

from __future__ import annotations

import datetime as dt
from typing import Any

import pytest

from alphaforge.capacity import (
    BorrowAvailability,
    CapacityContractError,
    CapacityEvidence,
    CapacityPolicy,
    CapacityPolicyDeclaration,
    CapacityPolicyViolation,
    ForcedBuyInBook,
    ForcedBuyInHalt,
    LiquidityObservation,
    LocateRecord,
    ScenarioResult,
    SessionCapacityLedger,
    book_digest,
    capacity_frontier,
    detect_recalls,
    frontier_summary,
    validate_scenarios,
    validate_unique_records,
    verify_row_aggregation,
)

MONDAY = dt.date(2024, 1, 1)


def session(offset: int) -> dt.date:
    """Return a session ``offset`` days after the fixture epoch."""
    return MONDAY + dt.timedelta(days=offset)


def borrow_row(**overrides: Any) -> BorrowAvailability:
    fields: dict[str, Any] = {
        "symbol": "AAA",
        "as_of_session": session(1),
        "effective_session": session(2),
        "expiry_session": session(30),
        "status": "available",
        "shortable_quantity": 10_000.0,
        "source": "synthetic-book",
        "source_version": "v1",
    }
    fields.update(overrides)
    return BorrowAvailability(**fields)


def locate_row(**overrides: Any) -> LocateRecord:
    fields: dict[str, Any] = {
        "locate_id": "L1",
        "symbol": "AAA",
        "granted_session": session(1),
        "expiry_session": session(10),
        "quantity": 6_000.0,
        "source": "synthetic",
    }
    fields.update(overrides)
    return LocateRecord(**fields)


def liquidity_row(**overrides: Any) -> LiquidityObservation:
    fields: dict[str, Any] = {
        "symbol": "AAA",
        "as_of_session": session(1),
        "adv_shares": 200_000.0,
        "adv_notional": 1e7,
        "lookback_sessions": 21,
        "source": "synthetic",
    }
    fields.update(overrides)
    return LiquidityObservation(**fields)


def declaration(**overrides: Any) -> CapacityPolicyDeclaration:
    fields: dict[str, Any] = {
        "max_participation": 0.05,
        "max_session_notional": 1e7,
        "max_staleness_sessions": 5,
        "buy_in_resolution_sessions": 1,
    }
    fields.update(overrides)
    return CapacityPolicyDeclaration(**fields)


@pytest.fixture
def policy() -> CapacityPolicy:
    return CapacityPolicy(
        declaration(),
        borrow=(borrow_row(),),
        locates=(locate_row(),),
        liquidity=(liquidity_row(),),
    )


# ---------------------------------------------------------------------------
# AC1: contracts declare units, temporality, bounds, failure, identity
# ---------------------------------------------------------------------------


def test_records_publish_deterministic_content_identity() -> None:
    assert borrow_row().digest == borrow_row().digest
    assert borrow_row().digest != borrow_row(shortable_quantity=9_999.0).digest
    assert locate_row().digest != locate_row(quantity=1.0).digest
    assert liquidity_row().digest != liquidity_row(adv_shares=1.0).digest


def test_book_identity_is_order_independent() -> None:
    """Identity must track content, not file layout."""
    first = book_digest((borrow_row(),), (locate_row(),), (liquidity_row(),), declaration())
    second_borrow = (borrow_row(symbol="BBB"), borrow_row())
    reversed_borrow = (borrow_row(), borrow_row(symbol="BBB"))
    assert book_digest(second_borrow, (locate_row(),), (liquidity_row(),), declaration()) == (
        book_digest(reversed_borrow, (locate_row(),), (liquidity_row(),), declaration())
    )
    assert first != book_digest(second_borrow, (locate_row(),), (liquidity_row(),), declaration())


def test_policy_declaration_publishes_units_and_failure_behavior() -> None:
    record = declaration().to_dict()
    assert set(record["units"]) == {
        "max_participation",
        "max_session_notional",
        "max_staleness_sessions",
        "buy_in_resolution_sessions",
    }
    assert "zero new-short capacity" in record["failure_behavior"]
    assert len(record["digest"]) == 64


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"symbol": ""}, "symbol"),
        ({"symbol": "A B"}, "symbol"),
        ({"symbol": "x" * 100}, "symbol"),
        ({"shortable_quantity": -1.0}, "non-negative"),
        ({"shortable_quantity": float("nan")}, "finite"),
        ({"shortable_quantity": float("inf")}, "finite"),
        ({"shortable_quantity": True}, "bool"),
        ({"shortable_quantity": 1e20}, "ceiling"),
        ({"status": "maybe"}, "status"),
        ({"schema_version": "9.9.9"}, "unsupported borrow schema"),
        ({"as_of_session": dt.datetime(2024, 1, 1)}, "datetime.date"),
        ({"as_of_session": "2024-01-01"}, "datetime.date"),
    ],
)
def test_borrow_rows_fail_closed(overrides: dict[str, Any], message: str) -> None:
    with pytest.raises(CapacityContractError, match=message):
        borrow_row(**overrides)


def test_borrow_row_rejects_inverted_or_acausal_sessions() -> None:
    with pytest.raises(CapacityContractError, match="expiry_session precedes"):
        borrow_row(effective_session=session(10), expiry_session=session(2))
    with pytest.raises(CapacityContractError, match="observed after the session"):
        borrow_row(as_of_session=session(5), effective_session=session(2))


def test_non_shortable_status_cannot_carry_capacity() -> None:
    """A restricted symbol with positive quantity is a contradiction, not a value."""
    with pytest.raises(CapacityContractError, match="no new-short capacity by definition"):
        borrow_row(status="restricted", shortable_quantity=100.0)


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"quantity": 0.0}, "positive quantity"),
        ({"expiry_session": session(0)}, "expiry_session precedes"),
        ({"locate_id": "bad id"}, "locate_id"),
    ],
)
def test_locate_rows_fail_closed(overrides: dict[str, Any], message: str) -> None:
    with pytest.raises(CapacityContractError, match=message):
        locate_row(**overrides)


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"lookback_sessions": 0}, "lookback_sessions"),
        ({"lookback_sessions": True}, "lookback_sessions must be an int"),
        ({"adv_notional": -1.0}, "non-negative"),
    ],
)
def test_liquidity_rows_fail_closed(overrides: dict[str, Any], message: str) -> None:
    with pytest.raises(CapacityContractError, match=message):
        liquidity_row(**overrides)


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"max_participation": 0.0}, "strictly positive"),
        ({"max_participation": 1.5}, "ceiling"),
        ({"max_session_notional": 0.0}, "strictly positive"),
        ({"max_staleness_sessions": -1}, "max_staleness_sessions"),
        ({"buy_in_resolution_sessions": True}, "must be an int"),
    ],
)
def test_policy_declaration_fails_closed(overrides: dict[str, Any], message: str) -> None:
    with pytest.raises(CapacityContractError, match=message):
        declaration(**overrides)


def test_duplicate_and_conflicting_records_are_refused() -> None:
    """Ordering must never decide how much a strategy may short."""
    with pytest.raises(CapacityContractError, match="duplicate record"):
        validate_unique_records((borrow_row(), borrow_row()), name="borrow")
    with pytest.raises(CapacityContractError, match="conflicting records"):
        validate_unique_records((borrow_row(), borrow_row(shortable_quantity=5.0)), name="borrow")
    with pytest.raises(CapacityContractError, match="unsupported record type"):
        validate_unique_records((object(),), name="borrow")


# ---------------------------------------------------------------------------
# AC2: causality, absence-is-not-permission, monotonicity
# ---------------------------------------------------------------------------


def test_records_published_on_or_after_the_decision_are_invisible(policy: CapacityPolicy) -> None:
    """The governing causality rule: strictly-earlier observation only."""
    resolved = policy.resolve(session(1))
    assert resolved.borrow == {}
    assert resolved.liquidity == {}
    assert policy.resolve(session(2)).borrow != {}


def test_future_observations_cannot_change_an_earlier_decision() -> None:
    """Mutation test: rewrite the future, the earlier authorization is identical."""
    base = CapacityPolicy(
        declaration(),
        borrow=(borrow_row(),),
        locates=(locate_row(),),
        liquidity=(liquidity_row(),),
    )
    resolved = base.resolve(session(3))
    ledger = base.open_session_ledger(resolved)
    before = base.authorize(
        resolved,
        ledger,
        symbol="AAA",
        side="open_short",
        quantity=1_000.0,
        price=50.0,
        reservation_id="R1",
    )

    mutated = CapacityPolicy(
        declaration(),
        borrow=(
            borrow_row(),
            borrow_row(
                as_of_session=session(4),
                effective_session=session(5),
                status="recalled",
                shortable_quantity=0.0,
            ),
        ),
        locates=(
            locate_row(),
            locate_row(
                locate_id="L2", granted_session=session(6), expiry_session=session(9), quantity=1e6
            ),
        ),
        liquidity=(liquidity_row(), liquidity_row(as_of_session=session(6), adv_shares=1e9)),
    )
    resolved_after = mutated.resolve(session(3))
    ledger_after = mutated.open_session_ledger(resolved_after)
    after = mutated.authorize(
        resolved_after,
        ledger_after,
        symbol="AAA",
        side="open_short",
        quantity=1_000.0,
        price=50.0,
        reservation_id="R1",
    )
    assert before.authorized == after.authorized
    assert before.reason == after.reason
    assert before.locate_id == after.locate_id


@pytest.mark.parametrize(
    ("kwargs", "reason"),
    [
        ({"borrow": ()}, "no_borrow_record"),
        ({"locates": ()}, "no_locate"),
        ({"liquidity": ()}, "no_liquidity_record"),
    ],
)
def test_missing_records_give_zero_new_short_capacity(kwargs: dict, reason: str) -> None:
    """Absence is never permission — the failure mode this MR exists to close."""
    fields: dict[str, Any] = {
        "borrow": (borrow_row(),),
        "locates": (locate_row(),),
        "liquidity": (liquidity_row(),),
    }
    fields.update(kwargs)
    policy = CapacityPolicy(declaration(), **fields)
    resolved = policy.resolve(session(3))
    decision = policy.authorize(
        resolved,
        policy.open_session_ledger(resolved),
        symbol="AAA",
        side="open_short",
        quantity=100.0,
        price=50.0,
        reservation_id="R1",
    )
    assert decision.authorized == 0.0
    assert decision.reason == reason


def test_stale_records_are_named_as_stale_not_missing() -> None:
    """Both give zero, but the operator needs to know which."""
    policy = CapacityPolicy(
        declaration(max_staleness_sessions=1),
        borrow=(borrow_row(),),
        locates=(locate_row(),),
        liquidity=(liquidity_row(),),
    )
    resolved = policy.resolve(session(20))
    decision = policy.authorize(
        resolved,
        policy.open_session_ledger(resolved),
        symbol="AAA",
        side="open_short",
        quantity=100.0,
        price=50.0,
        reservation_id="R1",
    )
    assert decision.authorized == 0.0
    assert decision.reason == "borrow_stale"


def test_expired_locate_cannot_authorize_a_fill() -> None:
    policy = CapacityPolicy(
        declaration(),
        borrow=(borrow_row(),),
        locates=(locate_row(expiry_session=session(3)),),
        liquidity=(liquidity_row(),),
    )
    resolved = policy.resolve(session(5))
    decision = policy.authorize(
        resolved,
        policy.open_session_ledger(resolved),
        symbol="AAA",
        side="open_short",
        quantity=100.0,
        price=50.0,
        reservation_id="R1",
    )
    assert decision.authorized == 0.0
    assert decision.reason == "locate_expired"


@pytest.mark.parametrize("status", ["restricted", "recalled", "unknown"])
def test_non_available_status_blocks_new_shorts(status: str) -> None:
    policy = CapacityPolicy(
        declaration(),
        borrow=(borrow_row(status=status, shortable_quantity=0.0),),
        locates=(locate_row(),),
        liquidity=(liquidity_row(),),
    )
    resolved = policy.resolve(session(3))
    decision = policy.authorize(
        resolved,
        policy.open_session_ledger(resolved),
        symbol="AAA",
        side="open_short",
        quantity=100.0,
        price=50.0,
        reservation_id="R1",
    )
    assert decision.authorized == 0.0
    assert decision.reason == "borrow_not_shortable"


@pytest.mark.parametrize("status", ["restricted", "recalled", "unknown"])
def test_a_cover_is_never_blocked_by_borrow(status: str) -> None:
    """The asymmetry that keeps the book from being trapped in a bad short."""
    policy = CapacityPolicy(
        declaration(),
        borrow=(borrow_row(status=status, shortable_quantity=0.0),),
        locates=(),
        liquidity=(liquidity_row(),),
    )
    resolved = policy.resolve(session(3))
    decision = policy.authorize(
        resolved,
        policy.open_session_ledger(resolved),
        symbol="AAA",
        side="cover_short",
        quantity=500.0,
        price=50.0,
        reservation_id="R1",
    )
    assert decision.authorized == 500.0
    assert decision.reason == "authorized"


def test_new_shorts_can_be_disabled_by_declaration_only() -> None:
    """Long-only is the rollback posture; it must be declared, never reached by failure."""
    policy = CapacityPolicy(
        declaration(allow_new_shorts=False),
        borrow=(borrow_row(),),
        locates=(locate_row(),),
        liquidity=(liquidity_row(),),
    )
    resolved = policy.resolve(session(3))
    decision = policy.authorize(
        resolved,
        policy.open_session_ledger(resolved),
        symbol="AAA",
        side="open_short",
        quantity=100.0,
        price=50.0,
        reservation_id="R1",
    )
    assert decision.reason == "new_shorts_disabled"


@pytest.mark.parametrize("factor", [0.5, 0.1, 0.0])
def test_reducing_inputs_cannot_increase_feasible_quantity(factor: float) -> None:
    """Monotonicity: less availability, ADV, or budget never buys more capacity."""
    request = 5_000.0

    def authorized(scale: float) -> float:
        policy = CapacityPolicy(
            declaration(max_session_notional=1e7 * max(scale, 1e-9)),
            borrow=(borrow_row(shortable_quantity=10_000.0 * scale),),
            locates=(
                (locate_row(quantity=6_000.0 * max(scale, 1e-9)),)
                if scale > 0.0
                else (locate_row(quantity=1e-9),)
            ),
            liquidity=(liquidity_row(adv_shares=200_000.0 * scale),),
        )
        resolved = policy.resolve(session(3))
        return policy.authorize(
            resolved,
            policy.open_session_ledger(resolved),
            symbol="AAA",
            side="open_short",
            quantity=request,
            price=50.0,
            reservation_id="R1",
        ).authorized

    assert authorized(factor) <= authorized(1.0) + 1e-9


def test_participation_limit_derives_from_lagged_adv(policy: CapacityPolicy) -> None:
    resolved = policy.resolve(session(3))
    assert policy.participation_limits(resolved) == {"AAA": 200_000.0 * 0.05}


# ---------------------------------------------------------------------------
# AC3: conservation, no double-spend, replay safety
# ---------------------------------------------------------------------------


@pytest.fixture
def ledger() -> SessionCapacityLedger:
    return SessionCapacityLedger(
        session=session(3),
        participation_limits={"AAA": 1_000.0},
        book_notional_budget=1e6,
    )


def test_unknown_symbol_has_zero_capacity(ledger: SessionCapacityLedger) -> None:
    assert ledger.remaining_participation("ZZZ") == 0.0


def test_reservation_consumption_and_release_conserve(ledger: SessionCapacityLedger) -> None:
    ledger.reserve("R1", "AAA", 400.0, 20_000.0)
    ledger.consume("R1", 250.0)
    ledger.release("R1")
    entry = ledger.reconciliation()["symbols"]["AAA"]
    assert entry["consumed"] == pytest.approx(250.0)
    assert entry["released"] == pytest.approx(150.0)
    assert entry["reconciles"]
    assert ledger.reconciliation()["all_symbols_reconcile"]


def test_over_request_is_rejected_not_granted(ledger: SessionCapacityLedger) -> None:
    reservation = ledger.reserve("R1", "AAA", 5_000.0, 250_000.0)
    assert reservation.quantity == pytest.approx(1_000.0)
    entry = ledger.reconciliation()["symbols"]["AAA"]
    assert entry["rejected"] == pytest.approx(4_000.0)
    assert entry["reconciles"]


def test_reservation_is_idempotent_under_replay(ledger: SessionCapacityLedger) -> None:
    """Journal replay re-presents identifiers; a second charge would leak capacity."""
    first = ledger.reserve("R1", "AAA", 400.0, 20_000.0)
    second = ledger.reserve("R1", "AAA", 400.0, 20_000.0)
    assert first == second
    assert ledger.remaining_participation("AAA") == pytest.approx(600.0)


def test_release_is_idempotent(ledger: SessionCapacityLedger) -> None:
    ledger.reserve("R1", "AAA", 400.0, 20_000.0)
    assert ledger.release("R1") == pytest.approx(400.0)
    assert ledger.release("R1") == 0.0
    assert ledger.reconciliation()["symbols"]["AAA"]["reconciles"]


def test_fill_larger_than_its_reservation_is_refused(ledger: SessionCapacityLedger) -> None:
    ledger.reserve("R1", "AAA", 100.0, 5_000.0)
    with pytest.raises(CapacityPolicyViolation, match="exceeds reservation"):
        ledger.consume("R1", 150.0)


def test_book_notional_budget_binds_across_symbols() -> None:
    ledger = SessionCapacityLedger(
        session=session(3),
        participation_limits={"AAA": 1e9, "BBB": 1e9},
        book_notional_budget=100_000.0,
    )
    ledger.reserve("R1", "AAA", 1_000.0, 90_000.0)
    second = ledger.reserve("R2", "BBB", 1_000.0, 90_000.0)
    assert second.notional <= 10_000.0 + 1e-6
    assert ledger.remaining_book_notional() == pytest.approx(0.0, abs=1e-6)


def test_unknown_reservation_is_refused(ledger: SessionCapacityLedger) -> None:
    with pytest.raises(CapacityPolicyViolation, match="unknown reservation"):
        ledger.consume("nope", 1.0)


def test_reservation_id_cannot_be_reused_for_another_symbol(
    ledger: SessionCapacityLedger,
) -> None:
    ledger.reserve("R1", "AAA", 10.0, 500.0)
    with pytest.raises(CapacityPolicyViolation, match="already claimed"):
        ledger.reserve("R1", "BBB", 10.0, 500.0)


def test_evidence_summarizes_decisions_and_reconciliation(policy: CapacityPolicy) -> None:
    evidence = CapacityEvidence()
    resolved = policy.resolve(session(3))
    ledger = policy.open_session_ledger(resolved)
    evidence.record(
        policy.authorize(
            resolved,
            ledger,
            symbol="AAA",
            side="open_short",
            quantity=1_000.0,
            price=50.0,
            reservation_id="R1",
        )
    )
    evidence.record(
        policy.authorize(
            resolved,
            ledger,
            symbol="ZZZ",
            side="open_short",
            quantity=100.0,
            price=10.0,
            reservation_id="R2",
        )
    )
    evidence.close_session(ledger)
    summary = evidence.summary()
    assert summary["decisions"] == 2
    assert summary["refused"] == 1
    assert summary["all_sessions_reconcile"]
    assert "no_borrow_record" in summary["by_reason"]


# ---------------------------------------------------------------------------
# AC4: forced buy-ins and halts
# ---------------------------------------------------------------------------


def test_recall_detection_covers_absence_and_status() -> None:
    """A vanished record is not evidence the borrow survived."""
    resolved = {"AAA": borrow_row(status="recalled", shortable_quantity=0.0)}
    triggers = detect_recalls(
        {"AAA": -100.0, "BBB": -50.0, "CCC": 25.0},
        resolved,
        session=session(3),
        next_session=session(4),
    )
    assert triggers == (("AAA", 100.0, "recalled"), ("BBB", 50.0, "unavailable"))


def test_buy_in_resolves_through_partial_fills() -> None:
    book = ForcedBuyInBook(resolution_sessions=2)
    book.schedule(
        buy_in_id="B1",
        symbol="AAA",
        triggered_session=session(3),
        scheduled_session=session(4),
        quantity=100.0,
        trigger="recalled",
    )
    assert book.apply_fill("B1", 60.0, "F1") == pytest.approx(60.0)
    assert len(book.outstanding()) == 1
    assert book.apply_fill("B1", 40.0, "F2") == pytest.approx(40.0)
    assert book.outstanding() == ()
    book.assert_resolved()
    record = book.evidence()
    assert record["resolved_count"] == 1
    assert record["resolved"][0]["filled"] == pytest.approx(100.0)


def test_buy_in_scheduling_is_idempotent() -> None:
    book = ForcedBuyInBook(resolution_sessions=1)
    first = book.schedule(
        buy_in_id="B1",
        symbol="AAA",
        triggered_session=session(3),
        scheduled_session=session(4),
        quantity=100.0,
        trigger="recalled",
    )
    second = book.schedule(
        buy_in_id="B1",
        symbol="AAA",
        triggered_session=session(3),
        scheduled_session=session(4),
        quantity=100.0,
        trigger="recalled",
    )
    assert first == second
    assert len(book.outstanding()) == 1


def test_unresolved_buy_in_halts_publication() -> None:
    """An unauthorized surviving short must stop the run, not be carried."""
    book = ForcedBuyInBook(resolution_sessions=1)
    book.schedule(
        buy_in_id="B1",
        symbol="AAA",
        triggered_session=session(3),
        scheduled_session=session(4),
        quantity=100.0,
        trigger="recalled",
    )
    book.advance_session(session(5))
    with pytest.raises(ForcedBuyInHalt, match="publication is halted"):
        book.advance_session(session(6))


def test_run_cannot_finish_with_outstanding_buy_ins() -> None:
    book = ForcedBuyInBook(resolution_sessions=5)
    book.schedule(
        buy_in_id="B1",
        symbol="AAA",
        triggered_session=session(3),
        scheduled_session=session(4),
        quantity=100.0,
        trigger="recalled",
    )
    with pytest.raises(ForcedBuyInHalt, match="not publishable"):
        book.assert_resolved()


def test_buy_in_validation() -> None:
    book = ForcedBuyInBook(resolution_sessions=1)
    with pytest.raises(CapacityContractError, match="positive quantity"):
        book.schedule(
            buy_in_id="B1",
            symbol="AAA",
            triggered_session=session(3),
            scheduled_session=session(4),
            quantity=0.0,
            trigger="recalled",
        )
    with pytest.raises(CapacityContractError, match="before the session that triggered"):
        book.schedule(
            buy_in_id="B2",
            symbol="AAA",
            triggered_session=session(5),
            scheduled_session=session(3),
            quantity=10.0,
            trigger="recalled",
        )
    with pytest.raises(CapacityContractError, match="unsupported buy-in trigger"):
        book.schedule(
            buy_in_id="B3",
            symbol="AAA",
            triggered_session=session(3),
            scheduled_session=session(4),
            quantity=10.0,
            trigger="whim",
        )
    with pytest.raises(CapacityContractError, match="unknown or already-resolved"):
        book.apply_fill("absent", 1.0, "F1")


# ---------------------------------------------------------------------------
# AC5: capacity frontier by complete rerun
# ---------------------------------------------------------------------------


def scenario(aum: float, **overrides: Any) -> ScenarioResult:
    fields: dict[str, Any] = {
        "aum": aum,
        "feasible": True,
        "desired_notional": aum,
        "filled_notional": aum,
        "shortfall_notional": 0.0,
        "fill_ratio": 1.0,
        "participation_utilization": 0.2,
        "book_budget_utilization": 0.2,
        "locate_rejections": 0,
        "forced_buy_ins": 0,
        "turnover": 0.5,
        "gross_return": 0.10,
        "net_return": 0.08,
        "costs": 0.02,
        "max_drawdown": -0.05,
        "max_concentration": 0.1,
    }
    fields.update(overrides)
    return ScenarioResult(**fields)


def test_frontier_reruns_every_scenario_exactly_once() -> None:
    """Never scale a completed series — call the simulation per scenario."""
    calls: list[float] = []

    def simulate(aum: float, _declaration: CapacityPolicyDeclaration) -> ScenarioResult:
        calls.append(aum)
        return scenario(aum, fill_ratio=min(1.0, 1e8 / aum))

    frame = capacity_frontier(
        simulate, declaration(), aum_levels=(1e7, 1e8, 1e9), reference_aum=1e8
    )
    assert calls == [1e7, 1e8, 1e9]
    assert list(frame["aum"]) == [1e7, 1e8, 1e9]
    assert frame["is_reference"].sum() == 1
    assert frame.loc[frame["is_reference"], "aum"].iloc[0] == 1e8
    assert frame["policy_digest"].nunique() == 1


def test_frontier_rows_aggregate_exactly() -> None:
    results = [scenario(level) for level in (1e7, 1e8)]
    frame = capacity_frontier(
        lambda aum, _d: next(item for item in results if item.aum == aum),
        declaration(),
        aum_levels=(1e7, 1e8),
        reference_aum=1e7,
    )
    verify_row_aggregation(frame, results)
    with pytest.raises(CapacityContractError, match="rows against"):
        verify_row_aggregation(frame, results[:1])


def test_infeasible_scenarios_stay_in_the_frontier() -> None:
    """Dropping hard scenarios is how a capacity curve grows an optimistic tail."""

    def simulate(aum: float, _d: CapacityPolicyDeclaration) -> ScenarioResult:
        if aum >= 1e9:
            return scenario(aum, feasible=False, fill_ratio=0.0, reason="liquidity exhausted")
        return scenario(aum)

    frame = capacity_frontier(simulate, declaration(), aum_levels=(1e8, 1e9), reference_aum=1e8)
    assert len(frame) == 2
    assert not bool(frame.loc[frame["aum"] == 1e9, "feasible"].iloc[0])
    summary = frontier_summary(frame)
    assert summary["first_infeasible_aum"] == 1e9
    assert summary["max_feasible_aum"] == 1e8


def test_frontier_summary_refuses_a_deployable_reading() -> None:
    frame = capacity_frontier(
        lambda aum, _d: scenario(aum), declaration(), aum_levels=(1e8,), reference_aum=1e8
    )
    interpretation = frontier_summary(frame)["interpretation"]
    for phrase in ("NOT a deployable AUM", "broker capacity", "expected"):
        assert phrase in interpretation


@pytest.mark.parametrize(
    ("levels", "message"),
    [
        ((), "at least one AUM"),
        ((1e8, 1e7), "strictly increasing"),
        ((1e8, 1e8), "strictly increasing"),
        ((0.0,), "strictly positive"),
        ((float("inf"),), "finite"),
        (tuple(float(index + 1) for index in range(30)), "ceiling"),
    ],
)
def test_scenario_grid_validation(levels: tuple, message: str) -> None:
    with pytest.raises(CapacityContractError, match=message):
        validate_scenarios(levels)


def test_reference_aum_must_be_in_the_grid() -> None:
    with pytest.raises(CapacityContractError, match="absent from the scenario grid"):
        capacity_frontier(
            lambda aum, _d: scenario(aum), declaration(), aum_levels=(1e8,), reference_aum=1e9
        )


def test_simulation_must_report_the_capital_it_ran() -> None:
    with pytest.raises(CapacityContractError, match="must report the capital"):
        capacity_frontier(
            lambda aum, _d: scenario(1.0), declaration(), aum_levels=(1e8,), reference_aum=1e8
        )

    def broken(aum: float, _declaration: CapacityPolicyDeclaration) -> Any:
        """A simulation that forgets to return a result must be refused, not trusted."""
        return None

    with pytest.raises(CapacityContractError, match="must return a ScenarioResult"):
        capacity_frontier(broken, declaration(), aum_levels=(1e8,), reference_aum=1e8)


def test_scenarios_are_candidate_order_isolated() -> None:
    """A scenario's result must not depend on which ran before it."""

    def simulate(aum: float, _d: CapacityPolicyDeclaration) -> ScenarioResult:
        return scenario(aum, net_return=0.10 - aum / 1e10)

    forward = capacity_frontier(
        simulate, declaration(), aum_levels=(1e8, 5e8, 1e9), reference_aum=1e8
    )
    single = capacity_frontier(simulate, declaration(), aum_levels=(1e9,), reference_aum=1e9)
    assert forward.loc[forward["aum"] == 1e9, "net_return"].iloc[0] == pytest.approx(
        single["net_return"].iloc[0]
    )
