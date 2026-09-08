"""Self-financing, cost-basis-aware portfolio accounting.

The ledger owns accounting mechanism only: cash, signed share quantities,
average cost, realized profit and loss (P&L), and separately charged costs.
Execution policy remains responsible for all-in fill prices and for deciding
which charges apply.  Keeping that boundary explicit prevents spread or impact
embedded in a fill price from being debited a second time.

Every accepted mutation is constructed off to the side and must satisfy both
the local self-financing equation and the cumulative cost-basis identity before
the live state changes.  Reconciliation tolerances scale with the ULP of the
actual operands; they never introduce a one-dollar or unit-scale floor that
could hide errors in a small account.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Literal, cast

ChargeCategory = Literal["fees", "financing", "borrow", "other"]

CHARGE_CATEGORIES: tuple[ChargeCategory, ...] = (
    "fees",
    "financing",
    "borrow",
    "other",
)

# Covers a conservative number of rounded additions/multiplications while
# remaining proportional to the represented values.  This is a numerical audit
# bound, not an economic tolerance.
_ULP_SAFETY_FACTOR = 64


def _empty_mapping() -> Mapping[str, float]:
    """Return an independently owned immutable empty mapping."""

    return MappingProxyType({})


@dataclass(frozen=True, slots=True)
class LedgerSnapshot:
    """An immutable mark-to-market view of a portfolio.

    Position quantities and market values are signed, so short positions have
    negative values and weights.  Cash is not included in ``weights``; its
    implicit portfolio weight is ``cash / equity``.
    """

    date: object
    cash: float
    equity: float
    positions: Mapping[str, float]
    market_values: Mapping[str, float]
    weights: Mapping[str, float]
    average_costs: Mapping[str, float] = field(default_factory=_empty_mapping)
    realized_pnl_by_symbol: Mapping[str, float] = field(default_factory=_empty_mapping)
    unrealized_pnl_by_symbol: Mapping[str, float] = field(default_factory=_empty_mapping)
    realized_pnl: float = 0.0
    unrealized_pnl: float = 0.0
    charges: Mapping[str, float] = field(default_factory=_empty_mapping)
    total_charges: float = 0.0
    net_pnl: float = 0.0
    reconciliation_error: float = 0.0
    reconciliation_tolerance: float = 0.0
    weights_defined: bool = True
    bankrupt: bool = False

    def __post_init__(self) -> None:
        """Detach mappings so a frozen snapshot is deeply immutable."""

        for name in (
            "positions",
            "market_values",
            "weights",
            "average_costs",
            "realized_pnl_by_symbol",
            "unrealized_pnl_by_symbol",
            "charges",
        ):
            value = getattr(self, name)
            try:
                detached = dict(value)
            except (TypeError, ValueError) as exc:
                raise ValueError(f"snapshot {name} must be a finite mapping") from exc
            if any(
                not isinstance(key, str)
                or not key.strip()
                or isinstance(item, bool)
                or not isinstance(item, (int, float))
                or not math.isfinite(float(item))
                for key, item in detached.items()
            ):
                raise ValueError(f"snapshot {name} must be a finite string-keyed mapping")
            object.__setattr__(
                self,
                name,
                MappingProxyType({key: float(item) for key, item in detached.items()}),
            )

        for name in (
            "cash",
            "equity",
            "realized_pnl",
            "unrealized_pnl",
            "total_charges",
            "net_pnl",
            "reconciliation_error",
            "reconciliation_tolerance",
        ):
            value = _finite_float(getattr(self, name), name=f"snapshot {name}")
            if (
                name
                in {
                    "total_charges",
                    "reconciliation_error",
                    "reconciliation_tolerance",
                }
                and value < 0.0
            ):
                raise ValueError(f"snapshot {name} must be non-negative")
            object.__setattr__(self, name, value)
        if not isinstance(self.weights_defined, bool) or not isinstance(self.bankrupt, bool):
            raise ValueError("snapshot state flags must be boolean")

    @property
    def gross_exposure(self) -> float:
        """Return absolute marked position exposure in account-currency units."""

        return _finite_sum(
            (abs(value) for value in self.market_values.values()),
            name="gross marked exposure",
        )

    @property
    def net_exposure(self) -> float:
        """Return signed marked position exposure in account-currency units."""

        return _finite_sum(self.market_values.values(), name="net marked exposure")


class PortfolioLedger:
    """Track cash and signed shares under self-financing accounting.

    Positive fill quantities buy shares and negative quantities sell shares.
    For every fill the cash balance changes according to

    ``cash -= signed_quantity * fill_price + separate_charges``.

    The fill price is assumed to be all-in: spread, slippage, and market impact
    must already be embedded in it.  Such implementation shortfall therefore
    enters average cost and must not also be passed as a separate charge.
    """

    def __init__(self, initial_cash: float = 1_000_000.0) -> None:
        cash = _finite_float(initial_cash, name="initial_cash")
        if cash <= 0.0:
            raise ValueError("initial_cash must be positive")
        self._initial_cash = cash
        self._cash = cash
        self._positions: dict[str, float] = {}
        self._average_costs: dict[str, float] = {}
        self._realized_pnl_by_symbol: dict[str, float] = {}
        self._charges: dict[ChargeCategory, float] = {
            category: 0.0 for category in CHARGE_CATEGORIES
        }

    @property
    def initial_cash(self) -> float:
        """Initial equity for cumulative P&L reconciliation."""

        return self._initial_cash

    @property
    def cash(self) -> float:
        """Current cash balance, which may be negative when the book is financed."""
        return self._cash

    @property
    def positions(self) -> Mapping[str, float]:
        """A read-only copy of current signed share quantities."""
        return MappingProxyType(dict(self._positions))

    @property
    def average_costs(self) -> Mapping[str, float]:
        """A detached read-only average fill price for every open position."""

        return MappingProxyType(dict(self._average_costs))

    @property
    def realized_pnl_by_symbol(self) -> Mapping[str, float]:
        """Cumulative realized P&L before separately charged costs."""

        return MappingProxyType(dict(self._realized_pnl_by_symbol))

    @property
    def realized_pnl(self) -> float:
        """Cumulative realized P&L before separately charged costs."""

        return _finite_sum(self._realized_pnl_by_symbol.values(), name="realized P&L")

    @property
    def charges(self) -> Mapping[str, float]:
        """Detached cumulative charges by stable accounting category."""

        return MappingProxyType(
            {str(category): amount for category, amount in self._charges.items()}
        )

    @property
    def total_charges(self) -> float:
        """All separately debited fees, financing, borrow, and other costs."""

        return _finite_sum(self._charges.values(), name="total charges")

    def clone(self) -> PortfolioLedger:
        """Return an exact, independently mutable copy of this ledger.

        Event reducers use a clone as a transaction candidate: accounting can be
        validated on the candidate before a journal append makes the event
        durable.  No mutable container is shared with the source ledger.

        Raises:
            RuntimeError: If the source ledger's internal accounting invariant
                is not reconciled.
        """

        _assert_cost_basis_reconciled(
            initial_cash=self._initial_cash,
            cash=self._cash,
            positions=self._positions,
            average_costs=self._average_costs,
            realized_pnl_by_symbol=self._realized_pnl_by_symbol,
            charges=self._charges,
        )
        candidate = PortfolioLedger(self._initial_cash)
        candidate._cash = self._cash
        candidate._positions = dict(self._positions)
        candidate._average_costs = dict(self._average_costs)
        candidate._realized_pnl_by_symbol = dict(self._realized_pnl_by_symbol)
        candidate._charges = dict(self._charges)
        return candidate

    def market_values(self, prices: Mapping[str, float]) -> dict[str, float]:
        """Mark all open positions at ``prices`` and return signed values."""
        validated_prices = _validate_prices(prices, required=set(self._positions))
        values = {
            symbol: quantity * validated_prices[symbol]
            for symbol, quantity in sorted(self._positions.items())
        }
        if not all(math.isfinite(value) for value in values.values()):
            raise ValueError("position market values must be finite")
        return values

    def equity(self, prices: Mapping[str, float]) -> float:
        """Return marked equity as cash plus signed position market values."""
        market_values = self.market_values(prices)
        return _finite_sum((self._cash, *market_values.values()), name="marked equity")

    def target_orders(
        self,
        target_weights: Mapping[str, float],
        reference_prices: Mapping[str, float],
    ) -> dict[str, float]:
        """Compute signed share trades needed to reach target weights.

        Every target is sized from the same pre-trade marked equity.  Existing
        positions omitted from ``target_weights`` receive a zero target and are
        therefore liquidated.  The method does not impose leverage, gross, or
        net exposure policy; it only requires finite weights.

        Returns only non-zero orders.  Positive quantities are buys and
        negative quantities are sells.
        """
        weights = _validate_weights(target_weights)
        symbols = set(self._positions) | set(weights)
        prices = _validate_prices(reference_prices, required=symbols)
        pretrade_equity = self.equity(prices)
        if pretrade_equity <= 0.0:
            raise ValueError("pre-trade equity must be positive to size target orders")

        orders: dict[str, float] = {}
        for symbol in sorted(symbols):
            target_quantity = weights.get(symbol, 0.0) * pretrade_equity / prices[symbol]
            signed_quantity = target_quantity - self._positions.get(symbol, 0.0)
            if not math.isfinite(signed_quantity):
                raise ValueError(f"target order quantity for {symbol!r} must be finite")
            if signed_quantity != 0.0:
                orders[symbol] = signed_quantity
        return orders

    def apply_fill(
        self,
        symbol: str,
        signed_quantity: float,
        fill_price: float,
        commission: float = 0.0,
        *,
        charges: Mapping[str, float] | None = None,
    ) -> None:
        """Apply one fill atomically and update average cost and realized P&L.

        No spread or impact adjustment is performed here; callers must supply
        the actual all-in execution price. ``commission`` is retained for API
        compatibility and enters the ``fees`` category. Additional category
        totals may be supplied through ``charges`` and are validated in full
        before any accounting state changes.

        A fill that increases a position updates weighted average cost. A fill
        in the opposite direction realizes P&L on the closed quantity; a fill
        crossing through zero assigns its price to the new side's residual.

        Raises:
            ValueError: If any input is invalid or arithmetic is non-finite.
            RuntimeError: If either accounting invariant fails. State remains
                unchanged for both failure classes.
        """
        _validate_symbol(symbol)
        quantity = _finite_float(signed_quantity, name="signed_quantity")
        price = _positive_price(fill_price, name="fill_price")
        commission_value = _finite_float(commission, name="commission")
        if commission_value < 0.0:
            raise ValueError("commission must be non-negative")
        event_charges = _validate_charges(charges)
        event_charges["fees"] = _finite_sum(
            (event_charges["fees"], commission_value),
            name="fill fees",
        )
        event_charge_total = _finite_sum(event_charges.values(), name="fill charges")

        old_quantity = self._positions.get(symbol, 0.0)
        old_average = self._average_costs.get(symbol, 0.0)
        notional = _finite_product(quantity, price, name="fill notional")
        new_cash = _finite_sum(
            (self._cash, -notional, -event_charge_total),
            name="cash balance after fill",
        )
        new_quantity = _finite_sum(
            (old_quantity, quantity),
            name="position quantity after fill",
        )
        stored_quantity = _clean_position_residue(
            new_quantity,
            price=price,
            operands=(old_quantity, quantity),
        )
        realized_delta, new_average = _cost_basis_transition(
            old_quantity=old_quantity,
            old_average=old_average,
            fill_quantity=quantity,
            fill_price=price,
            new_quantity=stored_quantity,
        )

        positions = dict(self._positions)
        average_costs = dict(self._average_costs)
        realized = dict(self._realized_pnl_by_symbol)
        charge_totals = dict(self._charges)
        if stored_quantity == 0.0:
            positions.pop(symbol, None)
            average_costs.pop(symbol, None)
        else:
            positions[symbol] = stored_quantity
            average_costs[symbol] = new_average
        if _opposite_sign(old_quantity, quantity):
            realized[symbol] = _finite_sum(
                (realized.get(symbol, 0.0), realized_delta),
                name=f"realized P&L for {symbol!r}",
            )
        for category, amount in event_charges.items():
            charge_totals[category] = _finite_sum(
                (charge_totals[category], amount),
                name=f"cumulative {category} charges",
            )

        old_value_at_fill = _finite_product(
            old_quantity,
            price,
            name="position market value at fill price",
        )
        new_value_at_fill = _finite_product(
            stored_quantity,
            price,
            name="post-fill position market value",
        )
        before_at_fill = _finite_sum(
            (self._cash, old_value_at_fill),
            name="pre-fill accounting value",
        )
        after_at_fill = _finite_sum(
            (new_cash, new_value_at_fill),
            name="post-fill accounting value",
        )
        expected_after = _finite_sum(
            (before_at_fill, -event_charge_total),
            name="expected post-fill accounting value",
        )
        _assert_close(
            after_at_fill,
            expected_after,
            operands=(
                self._cash,
                old_value_at_fill,
                new_cash,
                new_value_at_fill,
                event_charge_total,
            ),
            message="fill failed self-financing accounting reconciliation",
        )
        _assert_cost_basis_reconciled(
            initial_cash=self._initial_cash,
            cash=new_cash,
            positions=positions,
            average_costs=average_costs,
            realized_pnl_by_symbol=realized,
            charges=charge_totals,
        )

        # Every operation that can fail completed above. Publishing these owned
        # containers is consequently one logically atomic state transition.
        self._cash = new_cash
        self._positions = positions
        self._average_costs = average_costs
        self._realized_pnl_by_symbol = realized
        self._charges = charge_totals

    def apply_charge(self, category: str, amount: float) -> None:
        """Debit one separately charged cost atomically.

        ``category`` must be one of :data:`CHARGE_CATEGORIES`; unknown labels
        are refused so misspellings cannot create an unreported cost bucket.
        Charges are non-negative debits. Credits and external cash flows are a
        different accounting contract and are intentionally unsupported.
        """

        validated_category = _validate_charge_category(category)
        charge = _finite_float(amount, name=f"{validated_category} charge")
        if charge < 0.0:
            raise ValueError("charges must be non-negative")
        new_cash = _finite_sum((self._cash, -charge), name="cash balance after charge")
        charge_totals = dict(self._charges)
        charge_totals[validated_category] = _finite_sum(
            (charge_totals[validated_category], charge),
            name=f"cumulative {validated_category} charges",
        )
        _assert_cost_basis_reconciled(
            initial_cash=self._initial_cash,
            cash=new_cash,
            positions=self._positions,
            average_costs=self._average_costs,
            realized_pnl_by_symbol=self._realized_pnl_by_symbol,
            charges=charge_totals,
        )
        self._cash = new_cash
        self._charges = charge_totals

    def snapshot(self, date: object, prices: Mapping[str, float]) -> LedgerSnapshot:
        """Return an immutable, fully reconciled marked snapshot.

        Non-positive equity is returned with ``bankrupt=True`` so an event
        engine can journal and halt on the observed state. Weights remain
        available for negative equity. Exactly-zero equity, or an unrepresentable
        division, sets ``weights_defined=False`` and returns an empty weight map
        rather than concealing the accounting state behind an exception.
        """
        validated_prices = _validate_prices(prices, required=set(self._positions))
        position_values = {
            symbol: _finite_product(
                quantity,
                validated_prices[symbol],
                name=f"market value for {symbol!r}",
            )
            for symbol, quantity in sorted(self._positions.items())
        }
        equity = _finite_sum((self._cash, *position_values.values()), name="marked equity")
        unrealized_by_symbol: dict[str, float] = {}
        for symbol, quantity in sorted(self._positions.items()):
            price = validated_prices[symbol]
            difference = _finite_sum(
                (price, -self._average_costs[symbol]),
                name=f"price change for {symbol!r}",
            )
            unrealized_by_symbol[symbol] = _finite_product(
                quantity,
                difference,
                name=f"unrealized P&L for {symbol!r}",
            )
        realized_pnl = self.realized_pnl
        unrealized_pnl = _finite_sum(unrealized_by_symbol.values(), name="unrealized P&L")
        total_charges = self.total_charges
        net_pnl = _finite_sum(
            (realized_pnl, unrealized_pnl, -total_charges),
            name="net P&L",
        )
        expected_equity = _finite_sum(
            (self._initial_cash, net_pnl),
            name="expected marked equity",
        )
        accounting_error, accounting_tolerance = reconciliation_error(
            equity,
            expected_equity,
            operands=(
                self._initial_cash,
                self._cash,
                *position_values.values(),
                *self._realized_pnl_by_symbol.values(),
                *unrealized_by_symbol.values(),
                *self._charges.values(),
            ),
        )
        if accounting_error > accounting_tolerance:
            raise RuntimeError("marked ledger failed cumulative P&L reconciliation")

        weights_defined = equity != 0.0
        weights: dict[str, float] = {}
        if weights_defined:
            candidate_weights = {
                symbol: value / equity for symbol, value in position_values.items()
            }
            weights_defined = all(math.isfinite(weight) for weight in candidate_weights.values())
            if weights_defined:
                weights = candidate_weights

        return LedgerSnapshot(
            date=date,
            cash=self._cash,
            equity=equity,
            positions=self._positions,
            market_values=position_values,
            weights=weights,
            average_costs=self._average_costs,
            realized_pnl_by_symbol=self._realized_pnl_by_symbol,
            unrealized_pnl_by_symbol=unrealized_by_symbol,
            realized_pnl=realized_pnl,
            unrealized_pnl=unrealized_pnl,
            charges={str(category): amount for category, amount in self._charges.items()},
            total_charges=total_charges,
            net_pnl=net_pnl,
            reconciliation_error=accounting_error,
            reconciliation_tolerance=accounting_tolerance,
            weights_defined=weights_defined,
            bankrupt=equity <= 0.0,
        )


def _validate_charge_category(category: object) -> ChargeCategory:
    if not isinstance(category, str) or category not in CHARGE_CATEGORIES:
        allowed = ", ".join(CHARGE_CATEGORIES)
        raise ValueError(f"charge category must be one of: {allowed}")
    return cast(ChargeCategory, category)


def _validate_charges(charges: Mapping[str, float] | None) -> dict[ChargeCategory, float]:
    """Return a complete validated category record without unbounded iteration."""

    validated: dict[ChargeCategory, float] = {category: 0.0 for category in CHARGE_CATEGORIES}
    if charges is None:
        return validated
    try:
        iterator = iter(charges.items())
    except AttributeError as exc:
        raise ValueError("charges must be a category-to-amount mapping") from exc

    seen: set[ChargeCategory] = set()
    for index, item in enumerate(iterator):
        if index >= len(CHARGE_CATEGORIES):
            raise ValueError("charges contains more entries than supported categories")
        try:
            raw_category, raw_amount = item
        except (TypeError, ValueError) as exc:
            raise ValueError("charges must contain category/amount pairs") from exc
        category = _validate_charge_category(raw_category)
        if category in seen:
            raise ValueError(f"charges contains duplicate category {category!r}")
        amount = _finite_float(raw_amount, name=f"{category} charge")
        if amount < 0.0:
            raise ValueError("charges must be non-negative")
        validated[category] = amount
        seen.add(category)
    return validated


def _finite_product(left: float, right: float, *, name: str) -> float:
    try:
        result = left * right
    except OverflowError as exc:
        raise ValueError(f"{name} must be finite") from exc
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _opposite_sign(left: float, right: float) -> bool:
    return left != 0.0 and right != 0.0 and math.copysign(1.0, left) != math.copysign(1.0, right)


def _ulp_tolerance(values: Iterable[float], *, operations: int) -> float:
    """Return an operation-count-adjusted ULP bound at the actual value scale."""

    finite = tuple(abs(value) for value in values if math.isfinite(value))
    if not finite:
        return 0.0
    scale = max(finite)
    if scale == 0.0:
        return 0.0
    factor = max(_ULP_SAFETY_FACTOR, 8 * max(operations, len(finite)))
    tolerance = math.ulp(scale) * factor
    if not math.isfinite(tolerance):
        return math.ulp(scale)
    return tolerance


def reconciliation_error(
    observed: float,
    expected: float,
    *,
    operands: Iterable[float],
) -> tuple[float, float]:
    """Return absolute error and an operation-aware ULP tolerance.

    This is the shared numerical boundary for event and daily accounting.
    It deliberately has no currency- or unit-sized absolute floor.
    """

    error = abs(_finite_sum((observed, -expected), name="accounting reconciliation error"))
    materialized = tuple(operands)
    tolerance = _ulp_tolerance(
        (*materialized, observed, expected),
        operations=len(materialized) + 4,
    )
    return error, tolerance


def reconciles(
    observed: float,
    expected: float,
    *,
    operands: Iterable[float],
) -> bool:
    """Return whether two accounting values differ only by bounded roundoff."""

    error, tolerance = reconciliation_error(observed, expected, operands=operands)
    return error <= tolerance


def _assert_close(
    observed: float,
    expected: float,
    *,
    operands: Iterable[float],
    message: str,
) -> None:
    error, tolerance = reconciliation_error(observed, expected, operands=operands)
    if error > tolerance:
        raise RuntimeError(message)


def _clean_position_residue(
    quantity: float,
    *,
    price: float,
    operands: Iterable[float],
) -> float:
    """Remove only arithmetic zero residue whose marked value is also roundoff."""

    materialized = tuple(operands)
    quantity_tolerance = _ulp_tolerance(
        (*materialized, quantity),
        operations=len(materialized) + 1,
    )
    if abs(quantity) > quantity_tolerance:
        return quantity
    marked_residue = _finite_product(quantity, price, name="position residue market value")
    notional_operands = tuple(
        _finite_product(value, price, name="position residue operand") for value in materialized
    )
    value_tolerance = _ulp_tolerance(
        (*notional_operands, marked_residue),
        operations=len(notional_operands) + 2,
    )
    return 0.0 if abs(marked_residue) <= value_tolerance else quantity


def _cost_basis_transition(
    *,
    old_quantity: float,
    old_average: float,
    fill_quantity: float,
    fill_price: float,
    new_quantity: float,
) -> tuple[float, float]:
    """Return ``(realized_delta, new_average)`` for one signed fill."""

    if old_quantity == 0.0:
        return 0.0, 0.0 if new_quantity == 0.0 else fill_price
    if fill_quantity == 0.0:
        return 0.0, old_average
    if not _opposite_sign(old_quantity, fill_quantity):
        old_basis = _finite_product(abs(old_quantity), old_average, name="old position basis")
        fill_basis = _finite_product(abs(fill_quantity), fill_price, name="fill basis")
        total_basis = _finite_sum((old_basis, fill_basis), name="increased position basis")
        denominator = abs(new_quantity)
        if denominator == 0.0:
            raise RuntimeError("same-side fill unexpectedly produced a flat position")
        new_average = total_basis / denominator
        if not math.isfinite(new_average) or new_average <= 0.0:
            raise ValueError("average cost after fill must be finite and positive")
        return 0.0, new_average

    closing_quantity = min(abs(old_quantity), abs(fill_quantity))
    price_change = _finite_sum((fill_price, -old_average), name="realized price change")
    realized_delta = _finite_product(
        closing_quantity,
        price_change,
        name="realized P&L change",
    )
    if old_quantity < 0.0:
        realized_delta = -realized_delta
    if new_quantity == 0.0:
        return realized_delta, 0.0
    if math.copysign(1.0, new_quantity) == math.copysign(1.0, old_quantity):
        return realized_delta, old_average
    return realized_delta, fill_price


def _assert_cost_basis_reconciled(
    *,
    initial_cash: float,
    cash: float,
    positions: Mapping[str, float],
    average_costs: Mapping[str, float],
    realized_pnl_by_symbol: Mapping[str, float],
    charges: Mapping[ChargeCategory, float],
) -> None:
    if set(positions) != set(average_costs):
        raise RuntimeError("open positions and average costs are misaligned")
    basis_values = tuple(
        _finite_product(
            positions[symbol],
            average_costs[symbol],
            name=f"cost-basis value for {symbol!r}",
        )
        for symbol in sorted(positions)
    )
    realized_values = tuple(realized_pnl_by_symbol.values())
    charge_values = tuple(charges.values())
    observed = _finite_sum((cash, *basis_values), name="cost-basis accounting value")
    expected = _finite_sum(
        (initial_cash, *realized_values, *(-value for value in charge_values)),
        name="expected cost-basis accounting value",
    )
    _assert_close(
        observed,
        expected,
        operands=(
            initial_cash,
            cash,
            *basis_values,
            *realized_values,
            *charge_values,
        ),
        message="ledger failed cumulative cost-basis reconciliation",
    )


def _validate_symbol(symbol: object) -> None:
    if not isinstance(symbol, str) or not symbol.strip():
        raise ValueError("symbols must be non-empty strings")


def _finite_float(value: Any, *, name: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a finite number")
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must be a finite number") from exc
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _positive_price(value: object, *, name: str) -> float:
    price = _finite_float(value, name=name)
    if price <= 0.0:
        raise ValueError(f"{name} must be positive")
    return price


def _finite_sum(values: Iterable[float], *, name: str) -> float:
    try:
        result = math.fsum(values)
    except OverflowError as exc:
        raise ValueError(f"{name} must be finite") from exc
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _validate_prices(
    prices: Mapping[str, float],
    *,
    required: set[str],
) -> dict[str, float]:
    try:
        items = prices.items()
    except AttributeError as exc:
        raise ValueError("prices must be a symbol-to-price mapping") from exc

    validated: dict[str, float] = {}
    for symbol, value in items:
        _validate_symbol(symbol)
        validated[symbol] = _positive_price(value, name=f"price for {symbol!r}")

    missing = sorted(required - set(validated))
    if missing:
        raise ValueError(f"missing prices for symbols: {', '.join(missing)}")
    return validated


def _validate_weights(target_weights: Mapping[str, float]) -> dict[str, float]:
    try:
        items = target_weights.items()
    except AttributeError as exc:
        raise ValueError("target_weights must be a symbol-to-weight mapping") from exc

    validated: dict[str, float] = {}
    for symbol, value in items:
        _validate_symbol(symbol)
        validated[symbol] = _finite_float(value, name=f"target weight for {symbol!r}")

    try:
        gross_weight = math.fsum(abs(weight) for weight in validated.values())
        net_weight = math.fsum(validated.values())
    except OverflowError as exc:
        raise ValueError("target gross and net weights must be finite") from exc
    if not math.isfinite(gross_weight) or not math.isfinite(net_weight):
        raise ValueError("target gross and net weights must be finite")
    return validated
