from __future__ import annotations

import pandas as pd
import pytest

from alphaforge.execution.costs import CostModel
from alphaforge.execution.models import BarExecutionModel, ExecutionPolicy, Order


def _order(shares: float = 1_000.0) -> Order:
    return Order(
        order_id=1,
        symbol="AAA",
        decision_date=pd.Timestamp("2024-01-02"),
        fill_date=pd.Timestamp("2024-01-03"),
        requested_shares=shares,
        target_weight=0.5,
        pretrade_equity=1_000_000.0,
    )


def test_buy_fill_reconciles_explicit_cost_components() -> None:
    model = BarExecutionModel(
        CostModel(commission_bps=1.0, half_spread_bps=2.0, slippage_bps=3.0),
        ExecutionPolicy(impact_coefficient=0.10),
    )
    fill = model.execute(
        _order(),
        reference_price=100.0,
        lagged_adv_shares=100_000.0,
        lagged_volatility=0.02,
    )

    assert fill.status == "filled"
    assert fill.participation_rate == pytest.approx(0.01)
    assert fill.impact_bps == pytest.approx(2.0)
    assert fill.fill_price == pytest.approx(100.07)
    assert fill.total_cost == pytest.approx(80.0)
    assert fill.traded_notional == pytest.approx(100_000.0)


def test_sell_receives_a_price_below_the_reference() -> None:
    model = BarExecutionModel(
        CostModel(commission_bps=0.0, half_spread_bps=2.0, slippage_bps=3.0),
        ExecutionPolicy(),
    )
    fill = model.execute(_order(-100.0), reference_price=50.0)

    assert fill.filled_shares == -100.0
    assert fill.fill_price == pytest.approx(49.975)
    assert fill.total_cost == pytest.approx(2.5)


def test_lagged_adv_participation_cap_produces_a_partial_fill() -> None:
    model = BarExecutionModel(
        CostModel(commission_bps=0.0, half_spread_bps=0.0, slippage_bps=0.0),
        ExecutionPolicy(max_participation_rate=0.05),
    )
    fill = model.execute(
        _order(10_000.0),
        reference_price=20.0,
        lagged_adv_shares=100_000.0,
    )

    assert fill.status == "partial"
    assert fill.filled_shares == pytest.approx(5_000.0)
    assert fill.residual_shares == pytest.approx(5_000.0)
    assert fill.participation_rate == pytest.approx(0.05)


def test_large_order_residual_is_not_hidden_by_relative_tolerance() -> None:
    model = BarExecutionModel(
        CostModel(),
        ExecutionPolicy(max_participation_rate=0.999999),
    )

    fill = model.execute(
        _order(1_000_000_000.0),
        reference_price=1.0,
        lagged_adv_shares=1_000_000_000.0,
    )

    assert fill.status == "partial"
    assert fill.filled_shares == 999_999_000.0
    assert fill.residual_shares == 1_000.0


def test_participation_cap_is_never_rounded_up_within_ulp_tolerance() -> None:
    policy = ExecutionPolicy(max_participation_rate=0.999999999999999)
    model = BarExecutionModel(CostModel(), policy)

    fill = model.execute(
        _order(1_000_000_000_000_000.0),
        reference_price=1.0,
        lagged_adv_shares=1_000_000_000_000_000.0,
    )

    assert fill.status == "partial"
    assert fill.filled_shares == 999_999_999_999_999.0
    assert fill.residual_shares == 1.0
    assert policy.max_participation_rate is not None
    assert fill.participation_rate <= policy.max_participation_rate


def test_missing_adv_rejects_when_participation_is_enforced() -> None:
    model = BarExecutionModel(
        CostModel(),
        ExecutionPolicy(max_participation_rate=0.05),
    )
    fill = model.execute(_order(), reference_price=20.0)
    assert fill.status == "rejected"
    assert fill.filled_shares == 0.0
    assert fill.total_cost == 0.0


def test_execution_config_rejects_unknown_or_unsafe_settings() -> None:
    with pytest.raises(ValueError, match="unknown execution settings"):
        ExecutionPolicy.from_config({"lookahead_volume": True})
    with pytest.raises(ValueError, match="max_participation_rate"):
        ExecutionPolicy(max_participation_rate=1.1)
    with pytest.raises(ValueError, match="non-negative"):
        CostModel(slippage_bps=-1.0)
    with pytest.raises(ValueError, match="unknown transaction-cost settings"):
        CostModel.from_config({"free_money_bps": 10.0})


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"price_field": "close"}, "next-open"),
        ({"adv_lookback": 0}, "lookbacks"),
        ({"volatility_lookback": 1}, "lookbacks"),
        ({"impact_coefficient": -0.1}, "impact_coefficient"),
        ({"missing_price_policy": "invent"}, "missing_price_policy"),
    ],
)
def test_execution_policy_validation(kwargs, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        ExecutionPolicy(**kwargs)


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"order_id": 0}, "order_id"),
        ({"order_id": True}, "order_id"),
        ({"symbol": ""}, "symbol"),
        ({"symbol": "BAD SYMBOL"}, "symbol"),
        ({"decision_date": "2024-01-02"}, "decision_date"),
        ({"fill_date": pd.Timestamp("2024-01-03", tz="UTC")}, "fill_date"),
        ({"fill_date": pd.Timestamp("2024-01-02")}, "strictly after"),
        ({"requested_shares": float("nan")}, "finite"),
        ({"requested_shares": True}, "requested_shares"),
        ({"requested_shares": 3 + 4j}, "requested_shares"),
        ({"requested_shares": 1.0e15 + 1.0}, "requested_shares"),
        ({"target_weight": True}, "target_weight"),
        ({"target_weight": 3 + 4j}, "target_weight"),
        ({"pretrade_equity": 0.0}, "pretrade_equity"),
        ({"pretrade_equity": True}, "pretrade_equity"),
    ],
)
def test_order_contract_validation(overrides, message: str) -> None:
    values = {
        "order_id": 1,
        "symbol": "AAA",
        "decision_date": pd.Timestamp("2024-01-02"),
        "fill_date": pd.Timestamp("2024-01-03"),
        "requested_shares": 100.0,
        "target_weight": 0.5,
        "pretrade_equity": 1_000_000.0,
    }
    values.update(overrides)
    with pytest.raises(ValueError, match=message):
        Order(**values)


def test_missing_price_policy_is_explicit() -> None:
    strict = BarExecutionModel(CostModel(), ExecutionPolicy(missing_price_policy="raise"))
    with pytest.raises(ValueError, match="invalid open price"):
        strict.execute(_order(), reference_price=float("nan"))

    permissive = BarExecutionModel(CostModel(), ExecutionPolicy(missing_price_policy="skip"))
    rejected = permissive.execute(_order(), reference_price=float("nan"))
    assert rejected.status == "rejected"
    assert rejected.residual_shares == rejected.requested_shares


def test_factory_accepts_typed_configs_and_zero_order_is_rejected() -> None:
    costs = CostModel()
    policy = ExecutionPolicy()
    model = BarExecutionModel.from_config(costs=costs, execution=policy)
    fill = model.execute(_order(0.0), reference_price=100.0)
    assert fill.status == "rejected"


def test_composite_fill_costs_match_an_independent_reference() -> None:
    model = BarExecutionModel(
        CostModel(
            commission_bps=1.0,
            commission_per_share_usd=0.005,
            minimum_commission_usd=1.0,
            exchange_fee_bps=0.2,
            exchange_fee_per_share_usd=0.001,
            half_spread_bps=2.0,
            slippage_bps=1.0,
            spread_slippage_multiplier=0.5,
            participation_slippage_bps=4.0,
            participation_slippage_exponent=0.5,
            volatility_slippage_bps_per_1pct=0.5,
        ),
        ExecutionPolicy(impact_coefficient=0.10, impact_exponent=0.5),
    )

    fill = model.execute(
        _order(1_000.0),
        reference_price=100.0,
        lagged_adv_shares=4_000.0,
        lagged_volatility=0.02,
    )

    assert fill.status == "filled"
    assert fill.commission == pytest.approx(15.0)
    assert fill.exchange_fee == pytest.approx(3.0)
    assert fill.spread_cost == pytest.approx(20.0)
    assert fill.fixed_slippage_cost == pytest.approx(10.0)
    assert fill.spread_slippage_cost == pytest.approx(10.0)
    assert fill.participation_slippage_cost == pytest.approx(20.0)
    assert fill.volatility_slippage_cost == pytest.approx(10.0)
    assert fill.impact_cost == pytest.approx(100.0)
    assert fill.fill_price == pytest.approx(100.17)
    assert fill.total_cost == pytest.approx(188.0)
    assert fill.cost_breakdown is not None
    assert fill.cost_breakdown.cash_fees == pytest.approx(18.0)
    assert fill.cost_breakdown.embedded_cost == pytest.approx(170.0)


def test_active_models_fail_closed_when_causal_inputs_are_missing() -> None:
    participation = BarExecutionModel(
        CostModel(participation_slippage_bps=5.0),
        ExecutionPolicy(),
    ).execute(_order(), reference_price=100.0, lagged_volatility=0.01)
    volatility = BarExecutionModel(
        CostModel(volatility_slippage_bps_per_1pct=2.0),
        ExecutionPolicy(),
    ).execute(_order(), reference_price=100.0, lagged_adv_shares=10_000.0)

    assert participation.status == "rejected"
    assert participation.rejection_reason == "required_lagged_adv_unavailable"
    assert volatility.status == "rejected"
    assert volatility.rejection_reason == "required_lagged_volatility_unavailable"
    assert participation.total_cost == 0.0
    assert volatility.total_cost == 0.0


def test_extreme_sell_shortfall_rejects_non_positive_fill_price() -> None:
    model = BarExecutionModel(
        CostModel(commission_bps=0.0, half_spread_bps=0.0, slippage_bps=0.0),
        ExecutionPolicy(impact_coefficient=1.0),
    )

    fill = model.execute(
        _order(-100.0),
        reference_price=50.0,
        lagged_adv_shares=100.0,
        lagged_volatility=1.0,
    )

    assert fill.status == "rejected"
    assert fill.rejection_reason == "non_positive_or_non_finite_fill_price"
    assert fill.total_cost == 0.0


def test_composite_cost_is_monotone_in_filled_size() -> None:
    model = BarExecutionModel(
        CostModel(
            commission_bps=1.0,
            exchange_fee_bps=0.5,
            participation_slippage_bps=8.0,
            participation_slippage_exponent=0.5,
        ),
        ExecutionPolicy(impact_coefficient=0.05),
    )
    costs = [
        model.execute(
            _order(shares),
            reference_price=10.0,
            lagged_adv_shares=1_000_000.0,
            lagged_volatility=0.02,
        ).total_cost
        for shares in (100.0, 1_000.0, 10_000.0, 100_000.0)
    ]

    assert costs == sorted(costs)
    assert all(later > earlier for earlier, later in zip(costs, costs[1:]))


@pytest.mark.parametrize(
    "factory",
    [
        lambda: CostModel.from_config({"commission_bps": True}),
        lambda: CostModel.from_config({"participation_slippage_exponent": False}),
        lambda: ExecutionPolicy.from_config({"adv_lookback": True}),
        lambda: ExecutionPolicy.from_config({"impact_exponent": False}),
    ],
)
def test_numeric_configuration_rejects_booleans(factory) -> None:
    with pytest.raises(ValueError):
        factory()


@pytest.mark.parametrize("invalid", [False, 0, "", []])
def test_model_factories_reject_falsy_non_mapping_configuration(invalid: object) -> None:
    with pytest.raises(ValueError, match="mapping"):
        CostModel.from_config(invalid)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="mapping"):
        ExecutionPolicy.from_config(invalid)  # type: ignore[arg-type]


def test_provenance_is_strict_and_identity_preserving() -> None:
    with pytest.raises(ValueError, match="never normalized"):
        CostModel(calibration_provenance="  source A")
    with pytest.raises(ValueError, match="never normalized"):
        ExecutionPolicy(calibration_provenance="source A\n")

    compact_cost = CostModel(calibration_provenance="source A")
    spaced_cost = CostModel(calibration_provenance="source   A")
    compact_policy = ExecutionPolicy(calibration_provenance="source A")
    spaced_policy = ExecutionPolicy(calibration_provenance="source   A")
    assert compact_cost.calibration_provenance == "source A"
    assert spaced_cost.calibration_provenance == "source   A"
    assert compact_cost.configuration_digest != spaced_cost.configuration_digest
    assert compact_policy.configuration_digest != spaced_policy.configuration_digest


def test_cost_resource_failure_propagates_instead_of_becoming_market_rejection() -> None:
    model = BarExecutionModel(
        CostModel(
            commission_bps=1_000_000.0,
            half_spread_bps=0.0,
            slippage_bps=0.0,
        ),
        ExecutionPolicy(),
    )
    order = _order(1.0e15)

    with pytest.raises(ValueError, match="commission|resource ceiling"):
        model.execute(order, reference_price=1_000.0)


@pytest.mark.parametrize("filled_shares", [True, 3 + 4j, 1.0e15 + 1.0])
def test_fill_cost_boundary_rejects_coerced_or_unbounded_quantity(
    filled_shares: object,
) -> None:
    with pytest.raises(ValueError, match="filled_shares"):
        CostModel().evaluate_fill(
            filled_shares=filled_shares,  # type: ignore[arg-type]
            reference_price=100.0,
            participation_rate=0.01,
            lagged_volatility=0.02,
            impact_bps=1.0,
        )


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"reference_price": True}, "reference_price"),
        ({"reference_price": 3 + 4j}, "reference_price"),
        ({"lagged_adv_shares": True}, "lagged_adv_shares"),
        ({"lagged_adv_shares": 3 + 4j}, "lagged_adv_shares"),
        ({"lagged_volatility": True}, "lagged_volatility"),
        ({"lagged_volatility": 3 + 4j}, "lagged_volatility"),
    ],
)
def test_execution_boundary_rejects_boolean_and_complex_inputs(
    kwargs: dict[str, object],
    message: str,
) -> None:
    values: dict[str, object] = {
        "reference_price": 100.0,
        "lagged_adv_shares": 10_000.0,
        "lagged_volatility": 0.02,
    }
    values.update(kwargs)
    model = BarExecutionModel(CostModel(), ExecutionPolicy(impact_coefficient=0.1))
    with pytest.raises(ValueError, match=message):
        model.execute(_order(), **values)  # type: ignore[arg-type]
