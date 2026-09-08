from __future__ import annotations

import math
import random
from dataclasses import FrozenInstanceError

import pytest

from alphaforge.backtesting.ledger import PortfolioLedger


class _IndependentReferenceLedger:
    """Minimal conservation-law reference used only for randomized comparison."""

    def __init__(self, initial_cash: float) -> None:
        self.initial_cash = initial_cash
        self.cash = initial_cash
        self.positions: dict[str, float] = {}
        self.signed_basis: dict[str, float] = {}
        self.realized: dict[str, float] = {}
        self.fees = 0.0

    def apply_fill(self, symbol: str, quantity: float, price: float, fee: float) -> None:
        old_quantity = self.positions.get(symbol, 0.0)
        old_basis = self.signed_basis.get(symbol, 0.0)
        new_quantity = old_quantity + quantity

        if old_quantity == 0.0 or math.copysign(1.0, old_quantity) == math.copysign(1.0, quantity):
            new_basis = math.fsum((old_basis, quantity * price))
        elif new_quantity == 0.0:
            new_basis = 0.0
        elif math.copysign(1.0, new_quantity) == math.copysign(1.0, old_quantity):
            new_basis = new_quantity * (old_basis / old_quantity)
        else:
            new_basis = new_quantity * price

        if old_quantity != 0.0 and math.copysign(1.0, old_quantity) != math.copysign(1.0, quantity):
            realized_delta = math.fsum((-quantity * price, new_basis, -old_basis))
            self.realized[symbol] = math.fsum((self.realized.get(symbol, 0.0), realized_delta))

        self.cash = math.fsum((self.cash, -quantity * price, -fee))
        self.fees = math.fsum((self.fees, fee))
        if new_quantity == 0.0:
            self.positions.pop(symbol, None)
            self.signed_basis.pop(symbol, None)
        else:
            self.positions[symbol] = new_quantity
            self.signed_basis[symbol] = new_basis

    @property
    def average_costs(self) -> dict[str, float]:
        return {
            symbol: self.signed_basis[symbol] / quantity
            for symbol, quantity in self.positions.items()
        }

    @property
    def realized_pnl(self) -> float:
        return math.fsum(self.realized.values())


def _accounting_state(ledger: PortfolioLedger) -> tuple[object, ...]:
    return (
        ledger.initial_cash,
        ledger.cash,
        dict(ledger.positions),
        dict(ledger.average_costs),
        dict(ledger.realized_pnl_by_symbol),
        dict(ledger.charges),
    )


def test_long_and_short_fills_update_cash_and_equity() -> None:
    ledger = PortfolioLedger(initial_cash=1_000.0)

    ledger.apply_fill("LONG", signed_quantity=5.0, fill_price=100.0, commission=1.0)
    assert ledger.cash == pytest.approx(499.0)
    assert ledger.positions == {"LONG": 5.0}
    assert ledger.equity({"LONG": 100.0}) == pytest.approx(999.0)

    ledger.apply_fill("SHORT", signed_quantity=-2.0, fill_price=50.0, commission=0.5)
    assert ledger.cash == pytest.approx(598.5)
    assert ledger.positions == {"LONG": 5.0, "SHORT": -2.0}
    assert ledger.equity({"LONG": 100.0, "SHORT": 50.0}) == pytest.approx(998.5)

    ledger.apply_fill("SHORT", signed_quantity=2.0, fill_price=45.0, commission=0.5)
    assert ledger.cash == pytest.approx(508.0)
    assert ledger.positions == {"LONG": 5.0}
    assert ledger.equity({"LONG": 100.0}) == pytest.approx(1_008.0)


def test_average_cost_realized_pnl_and_flip_match_hand_reference() -> None:
    ledger = PortfolioLedger(initial_cash=1_000.0)

    ledger.apply_fill("A", 10.0, 10.0, commission=1.0)
    ledger.apply_fill("A", 5.0, 20.0, charges={"fees": 2.0})
    ledger.apply_fill("A", -6.0, 25.0, commission=1.0)

    assert ledger.cash == pytest.approx(946.0)
    assert ledger.positions == pytest.approx({"A": 9.0})
    assert ledger.average_costs == pytest.approx({"A": 200.0 / 15.0})
    assert ledger.realized_pnl_by_symbol == pytest.approx({"A": 70.0})
    assert ledger.realized_pnl == pytest.approx(70.0)
    assert ledger.charges == pytest.approx(
        {"fees": 4.0, "financing": 0.0, "borrow": 0.0, "other": 0.0}
    )

    marked = ledger.snapshot("2026-01-02", {"A": 30.0})
    assert marked.equity == pytest.approx(1_216.0)
    assert marked.unrealized_pnl_by_symbol == pytest.approx({"A": 150.0})
    assert marked.realized_pnl == pytest.approx(70.0)
    assert marked.unrealized_pnl == pytest.approx(150.0)
    assert marked.total_charges == pytest.approx(4.0)
    assert marked.net_pnl == pytest.approx(216.0)
    assert marked.reconciliation_error <= marked.reconciliation_tolerance

    # Selling through zero closes nine long shares and opens three short at $5.
    ledger.apply_fill("A", -12.0, 5.0)
    flipped = ledger.snapshot("2026-01-03", {"A": 4.0})
    assert ledger.cash == pytest.approx(1_006.0)
    assert ledger.positions == pytest.approx({"A": -3.0})
    assert ledger.average_costs == pytest.approx({"A": 5.0})
    assert ledger.realized_pnl == pytest.approx(-5.0)
    assert flipped.unrealized_pnl == pytest.approx(3.0)
    assert flipped.equity == pytest.approx(994.0)
    assert flipped.net_pnl == pytest.approx(-6.0)


def test_short_average_cost_partial_close_and_flip_match_hand_reference() -> None:
    ledger = PortfolioLedger(initial_cash=1_000.0)

    ledger.apply_fill("S", -10.0, 20.0)
    ledger.apply_fill("S", 4.0, 15.0)
    assert ledger.positions == pytest.approx({"S": -6.0})
    assert ledger.average_costs == pytest.approx({"S": 20.0})
    assert ledger.realized_pnl == pytest.approx(20.0)

    ledger.apply_fill("S", 8.0, 25.0)
    snapshot = ledger.snapshot("2026-01-04", {"S": 30.0})
    assert ledger.positions == pytest.approx({"S": 2.0})
    assert ledger.average_costs == pytest.approx({"S": 25.0})
    assert ledger.realized_pnl == pytest.approx(-10.0)
    assert snapshot.unrealized_pnl == pytest.approx(10.0)
    assert snapshot.net_pnl == pytest.approx(0.0)
    assert snapshot.equity == pytest.approx(1_000.0)


def test_all_in_fill_price_and_commission_are_both_reflected_in_equity() -> None:
    ledger = PortfolioLedger(initial_cash=1_000.0)

    # The one-dollar difference between the reference and fill prices represents
    # spread/impact supplied by the execution model; the ledger adds no implicit cost.
    ledger.apply_fill("A", signed_quantity=10.0, fill_price=101.0, commission=1.0)

    assert ledger.cash == pytest.approx(-11.0)
    assert ledger.equity({"A": 100.0}) == pytest.approx(989.0)


def test_separate_charge_categories_are_debited_and_reported_once() -> None:
    ledger = PortfolioLedger(initial_cash=1_000.0)
    ledger.apply_charge("fees", 2.0)
    ledger.apply_charge("financing", 3.0)
    ledger.apply_charge("borrow", 4.0)
    ledger.apply_charge("other", 5.0)
    ledger.apply_fill(
        "A",
        1.0,
        10.0,
        commission=1.0,
        charges={"fees": 2.0, "financing": 0.5},
    )

    snapshot = ledger.snapshot("2026-01-02", {"A": 12.0})
    assert ledger.cash == pytest.approx(972.5)
    assert snapshot.charges == pytest.approx(
        {"fees": 5.0, "financing": 3.5, "borrow": 4.0, "other": 5.0}
    )
    assert snapshot.total_charges == pytest.approx(17.5)
    assert snapshot.unrealized_pnl == pytest.approx(2.0)
    assert snapshot.net_pnl == pytest.approx(-15.5)
    assert snapshot.equity == pytest.approx(984.5)
    assert snapshot.reconciliation_error <= snapshot.reconciliation_tolerance


def test_clone_is_exact_and_shares_no_mutable_accounting_state() -> None:
    source = PortfolioLedger(initial_cash=2_000.0)
    source.apply_fill("A", 5.0, 100.0, commission=1.0)
    source.apply_fill("A", -2.0, 110.0)
    source.apply_fill("B", -4.0, 50.0)
    source.apply_charge("borrow", 0.75)

    candidate = source.clone()
    assert candidate is not source
    assert candidate.initial_cash == source.initial_cash
    assert candidate.cash == source.cash
    assert candidate.positions == source.positions
    assert candidate.average_costs == source.average_costs
    assert candidate.realized_pnl_by_symbol == source.realized_pnl_by_symbol
    assert candidate.charges == source.charges

    candidate.apply_fill("A", 1.0, 120.0)
    candidate.apply_charge("financing", 2.0)
    assert source.positions == {"A": 3.0, "B": -4.0}
    assert source.charges["financing"] == 0.0

    source.apply_fill("B", 1.0, 45.0)
    source.apply_charge("other", 3.0)
    assert candidate.positions == {"A": 4.0, "B": -4.0}
    assert candidate.charges["other"] == 0.0


def test_target_orders_use_one_pretrade_nav_and_include_liquidations() -> None:
    ledger = PortfolioLedger(initial_cash=1_000.0)
    ledger.apply_fill("A", signed_quantity=4.0, fill_price=100.0)

    # A has appreciated, so pre-trade NAV is 600 cash + 4 * 125 = 1,100.
    orders = ledger.target_orders(
        target_weights={"A": 0.5, "B": -0.25},
        reference_prices={"A": 125.0, "B": 50.0},
    )

    assert orders == pytest.approx({"A": 0.4, "B": -5.5})

    for symbol, quantity in orders.items():
        ledger.apply_fill(symbol, quantity, {"A": 125.0, "B": 50.0}[symbol])
    snapshot = ledger.snapshot("2026-01-02", {"A": 125.0, "B": 50.0})
    assert snapshot.equity == pytest.approx(1_100.0)
    assert snapshot.weights == pytest.approx({"A": 0.5, "B": -0.25})

    liquidation = ledger.target_orders(
        target_weights={"B": 0.0},
        reference_prices={"A": 125.0, "B": 50.0},
    )
    assert liquidation == pytest.approx({"A": -4.4, "B": 5.5})


def test_snapshot_captures_drift_and_is_detached_from_later_state() -> None:
    ledger = PortfolioLedger(initial_cash=1_000.0)
    ledger.apply_fill("A", signed_quantity=6.0, fill_price=100.0)
    ledger.apply_fill("B", signed_quantity=8.0, fill_price=50.0)

    snapshot = ledger.snapshot("2026-01-03", {"A": 110.0, "B": 40.0})

    assert snapshot.cash == pytest.approx(0.0)
    assert snapshot.positions == {"A": 6.0, "B": 8.0}
    assert snapshot.market_values == pytest.approx({"A": 660.0, "B": 320.0})
    assert snapshot.equity == pytest.approx(980.0)
    assert snapshot.weights == pytest.approx({"A": 660.0 / 980.0, "B": 320.0 / 980.0})
    assert snapshot.average_costs == pytest.approx({"A": 100.0, "B": 50.0})
    assert snapshot.unrealized_pnl_by_symbol == pytest.approx({"A": 60.0, "B": -80.0})
    assert snapshot.equity == pytest.approx(snapshot.cash + sum(snapshot.market_values.values()))

    ledger.apply_fill("A", signed_quantity=-1.0, fill_price=110.0)
    assert snapshot.positions == {"A": 6.0, "B": 8.0}
    with pytest.raises(TypeError):
        snapshot.positions["A"] = 99.0  # type: ignore[index]
    for mapping in (
        snapshot.average_costs,
        snapshot.realized_pnl_by_symbol,
        snapshot.unrealized_pnl_by_symbol,
        snapshot.charges,
    ):
        with pytest.raises(TypeError):
            mapping["A"] = 99.0  # type: ignore[index]
    with pytest.raises(FrozenInstanceError):
        snapshot.cash = 1.0  # type: ignore[misc]


def test_nonpositive_equity_is_observable_for_fail_closed_engine_halt() -> None:
    zero_equity = PortfolioLedger(initial_cash=100.0)
    zero_equity.apply_fill("A", 1.0, 100.0)
    zero_equity.apply_charge("fees", 100.0)

    zero_snapshot = zero_equity.snapshot("2026-01-05", {"A": 100.0})
    assert zero_snapshot.equity == 0.0
    assert zero_snapshot.bankrupt
    assert not zero_snapshot.weights_defined
    assert zero_snapshot.weights == {}
    assert zero_snapshot.net_pnl == pytest.approx(-100.0)

    negative_equity = PortfolioLedger(initial_cash=100.0)
    negative_equity.apply_fill("A", 1.0, 100.0)
    negative_equity.apply_charge("fees", 101.0)
    negative_snapshot = negative_equity.snapshot("2026-01-05", {"A": 100.0})
    assert negative_snapshot.equity == pytest.approx(-1.0)
    assert negative_snapshot.bankrupt
    assert negative_snapshot.weights_defined
    assert negative_snapshot.weights == pytest.approx({"A": -100.0})


def test_near_zero_position_cleanup_preserves_fill_reconciliation() -> None:
    ledger = PortfolioLedger(initial_cash=1_000.0)
    ledger.apply_fill("A", 0.3, 100.0)
    ledger.apply_fill("A", -(0.1 + 0.2), 100.0)

    assert ledger.positions == {}
    assert ledger.cash == pytest.approx(1_000.0)
    assert ledger.equity({}) == pytest.approx(1_000.0)

    large_ledger = PortfolioLedger(initial_cash=1e12)
    large_ledger.apply_fill("EXPENSIVE", 5e-13, 1e10)
    assert large_ledger.positions == {"EXPENSIVE": 5e-13}
    assert large_ledger.equity({"EXPENSIVE": 1e10}) == pytest.approx(1e12)


@pytest.mark.parametrize("price", [0.0, -1.0, float("nan"), float("inf")])
def test_prices_must_be_finite_and_positive(price: float) -> None:
    ledger = PortfolioLedger(initial_cash=1_000.0)

    with pytest.raises(ValueError):
        ledger.apply_fill("A", 1.0, price)

    assert ledger.cash == pytest.approx(1_000.0)
    assert ledger.positions == {}


def test_invalid_accounting_inputs_are_rejected_without_mutation() -> None:
    ledger = PortfolioLedger(initial_cash=1_000.0)
    ledger.apply_fill("A", 1.0, 100.0)

    with pytest.raises(ValueError, match="missing prices"):
        ledger.equity({})
    with pytest.raises(ValueError, match="target weight"):
        ledger.target_orders({"A": float("nan")}, {"A": 100.0})
    with pytest.raises(ValueError, match="gross and net"):
        ledger.target_orders({"A": 1e308, "B": 1e308}, {"A": 100.0, "B": 100.0})
    with pytest.raises(ValueError, match="commission"):
        ledger.apply_fill("A", 1.0, 100.0, commission=-1.0)

    assert ledger.cash == pytest.approx(900.0)
    assert ledger.positions == {"A": 1.0}

    oversized = PortfolioLedger(initial_cash=1e308)
    oversized.apply_fill("A", 1e308, 1.0)
    with pytest.raises(ValueError, match="market value"):
        oversized.apply_fill("A", 0.0, 2.0)
    assert oversized.cash == pytest.approx(0.0)
    assert oversized.positions == {"A": 1e308}


def test_invalid_fill_and_charge_inputs_fail_atomically() -> None:
    ledger = PortfolioLedger(initial_cash=1_000.0)
    ledger.apply_fill("A", 2.0, 100.0)
    ledger.apply_charge("fees", 1.0)

    invalid_operations = (
        lambda: ledger.apply_fill("A", True, 100.0),
        lambda: ledger.apply_fill("A", 1.0, True),
        lambda: ledger.apply_fill("A", 1.0, 100.0, commission=True),
        lambda: ledger.apply_fill("A", 1.0, 100.0, charges={"tax": 1.0}),
        lambda: ledger.apply_fill("A", 1.0, 100.0, charges={"fees": -1.0}),
        lambda: ledger.apply_fill("A", 1.0, 100.0, charges={"fees": float("nan")}),
        lambda: ledger.apply_charge("tax", 1.0),
        lambda: ledger.apply_charge("fees", True),
        lambda: ledger.apply_charge("fees", -1.0),
        lambda: ledger.apply_charge("fees", float("inf")),
    )
    for operation in invalid_operations:
        before = _accounting_state(ledger)
        with pytest.raises(ValueError):
            operation()
        assert _accounting_state(ledger) == before


def test_overflowing_fill_and_cumulative_charge_fail_atomically() -> None:
    fill_ledger = PortfolioLedger(initial_cash=1e308)
    before_fill = _accounting_state(fill_ledger)
    with pytest.raises(ValueError, match="fill notional"):
        fill_ledger.apply_fill("A", 1e308, 2.0)
    assert _accounting_state(fill_ledger) == before_fill

    charge_ledger = PortfolioLedger(initial_cash=1e308)
    charge_ledger.apply_charge("fees", 1e308)
    before_charge = _accounting_state(charge_ledger)
    with pytest.raises(ValueError, match="cumulative fees"):
        charge_ledger.apply_charge("fees", 1e308)
    assert _accounting_state(charge_ledger) == before_charge


def test_randomized_fills_match_independent_conservation_reference() -> None:
    seed = 0x5F4
    generator = random.Random(seed)
    ledger = PortfolioLedger(initial_cash=100_000.0)
    reference = _IndependentReferenceLedger(initial_cash=100_000.0)
    symbols = ("A", "B", "C", "D")

    for _ in range(400):
        symbol = generator.choice(symbols)
        quantity = float(generator.choice((-7, -5, -3, -2, -1, 1, 2, 3, 5, 7)))
        price = generator.randint(20, 400) + generator.choice((0.0, 0.25, 0.5, 0.75))
        fee = generator.choice((0.0, 0.01, 0.05, 0.10))
        ledger.apply_fill(symbol, quantity, price, commission=fee)
        reference.apply_fill(symbol, quantity, price, fee)

    assert ledger.cash == pytest.approx(reference.cash, rel=2e-13, abs=2e-10)
    assert ledger.positions == pytest.approx(reference.positions)
    assert ledger.average_costs == pytest.approx(reference.average_costs, rel=2e-13)
    assert ledger.realized_pnl_by_symbol == pytest.approx(reference.realized, abs=2e-10)
    assert ledger.realized_pnl == pytest.approx(reference.realized_pnl, abs=2e-10)
    assert ledger.charges["fees"] == pytest.approx(reference.fees)

    marks = {symbol: float(generator.randint(25, 350)) for symbol in ledger.positions}
    snapshot = ledger.snapshot(f"seed={seed}", marks)
    reference_market_value = math.fsum(
        reference.positions[symbol] * marks[symbol] for symbol in reference.positions
    )
    assert snapshot.equity == pytest.approx(
        math.fsum((reference.cash, reference_market_value)), abs=2e-10
    )
    assert snapshot.reconciliation_error <= snapshot.reconciliation_tolerance


def test_split_fill_and_symbol_permutation_are_accounting_invariant() -> None:
    whole = PortfolioLedger(initial_cash=10_000.0)
    split = PortfolioLedger(initial_cash=10_000.0)
    whole.apply_fill("A", 7.0, 11.0)
    split.apply_fill("A", 7.0, 11.0)
    whole.apply_fill("A", -12.0, 15.0, commission=3.0, charges={"financing": 0.5})
    split.apply_fill("A", -4.0, 15.0, commission=1.0, charges={"financing": 0.2})
    split.apply_fill("A", -8.0, 15.0, commission=2.0, charges={"financing": 0.3})
    assert _accounting_state(whole) == pytest.approx(_accounting_state(split))

    fills = {
        "A": ((3.0, 10.0), (-1.0, 15.0)),
        "B": ((-4.0, 20.0), (2.0, 18.0)),
        "C": ((5.0, 7.0), (1.0, 9.0)),
    }
    forward = PortfolioLedger(initial_cash=10_000.0)
    reversed_symbols = PortfolioLedger(initial_cash=10_000.0)
    for symbol in fills:
        for quantity, price in fills[symbol]:
            forward.apply_fill(symbol, quantity, price)
    for symbol in reversed(tuple(fills)):
        for quantity, price in fills[symbol]:
            reversed_symbols.apply_fill(symbol, quantity, price)

    assert forward.cash == pytest.approx(reversed_symbols.cash)
    assert forward.positions == reversed_symbols.positions
    assert forward.average_costs == reversed_symbols.average_costs
    assert forward.realized_pnl_by_symbol == reversed_symbols.realized_pnl_by_symbol
    marks = {"A": 12.0, "B": 17.0, "C": 8.0}
    assert forward.snapshot("forward", marks).equity == pytest.approx(
        reversed_symbols.snapshot("reversed", marks).equity
    )


def test_reconciliation_tolerance_scales_below_one_without_absolute_floor() -> None:
    ledger = PortfolioLedger(initial_cash=1e-12)
    ledger.apply_fill("MICRO", 1e-6, 1e-6)
    ledger.apply_charge("fees", 1e-18)

    snapshot = ledger.snapshot("2026-01-06", {"MICRO": 2e-6})
    assert snapshot.equity == pytest.approx(1.999999e-12)
    assert snapshot.unrealized_pnl == pytest.approx(1e-12)
    assert snapshot.total_charges == pytest.approx(1e-18)
    assert snapshot.reconciliation_error <= snapshot.reconciliation_tolerance
    assert snapshot.reconciliation_tolerance < 1e-20


@pytest.mark.parametrize("initial_cash", [True, 0.0, -1.0, float("nan"), float("inf")])
def test_initial_cash_must_be_finite_and_positive(initial_cash: float) -> None:
    with pytest.raises(ValueError):
        PortfolioLedger(initial_cash=initial_cash)
