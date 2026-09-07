"""Focused reconciliation, causality, and monotonicity tests for capacity scenarios."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pandas.testing as pdt
import pytest

from alphaforge.evaluation.capacity import (
    CapacityColumns,
    CapacityConfig,
    estimate_capacity,
)


def _fill_panel() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "date": pd.to_datetime(["2025-01-06", "2025-01-06", "2025-01-07"]),
            "information_date": pd.to_datetime(["2025-01-03", "2025-01-03", "2025-01-06"]),
            "symbol": ["AAA", "BBB", "AAA"],
            "desired_notional": [100_000.0, -80_000.0, 50_000.0],
            "traded_notional": [80_000.0, -80_000.0, 40_000.0],
            "lagged_adv_notional": [2_000_000.0, 1_000_000.0, 500_000.0],
            "total_cost": [80.0, 120.0, 80.0],
        }
    )


def _config() -> CapacityConfig:
    return CapacityConfig(
        reference_aum=1_000_000.0,
        # Deliberately unsorted with a duplicate: output must be canonical.
        aum_values=(4_000_000.0, 500_000.0, 1_000_000.0, 1_000_000.0),
        max_participation_rate=0.10,
        columns=CapacityColumns(information_date="information_date"),
    )


def test_capacity_curve_reconciles_to_row_scenarios_and_reference_costs() -> None:
    result = estimate_capacity(_fill_panel(), _config())

    assert result.curve["scenario_aum"].tolist() == [500_000.0, 1_000_000.0, 4_000_000.0]
    assert result.diagnostics.cost_source == "realized_total_cost"
    assert result.diagnostics.temporal_validation == "information_date_verified"
    assert "not deployable-AUM forecasts" in result.diagnostics.assumptions[0]

    sum_columns = [
        "desired_gross_notional",
        "empirical_fill_gross_notional",
        "modeled_fill_gross_notional",
        "total_shortfall_notional",
        "capacity_shortfall_notional",
        "modeled_cost",
    ]
    reconciled = (
        result.scenario_trades.groupby("scenario_aum", sort=True)[sum_columns].sum().reset_index()
    )
    pdt.assert_frame_equal(
        result.curve[["scenario_aum", *sum_columns]],
        reconciled,
        check_exact=False,
        rtol=1e-12,
        atol=1e-12,
    )

    reference = result.curve.set_index("scenario_aum").loc[1_000_000.0]
    assert reference["modeled_fill_gross_notional"] == pytest.approx(200_000.0)
    assert reference["modeled_cost"] == pytest.approx(_fill_panel()["total_cost"].sum())
    assert reference["modeled_cost_bps_per_traded_notional"] == pytest.approx(14.0)
    assert reference["fill_ratio"] == pytest.approx(200_000.0 / 230_000.0)


def test_capacity_curve_has_expected_monotonic_scenario_invariants() -> None:
    curve = estimate_capacity(_fill_panel(), _config()).curve

    nondecreasing = [
        "desired_gross_notional",
        "empirical_fill_gross_notional",
        "modeled_fill_gross_notional",
        "capacity_shortfall_notional",
        "aggregate_participation_rate",
        "participation_p95",
        "participation_max",
        "modeled_cost",
        "n_capacity_constrained",
        "capacity_constrained_fraction",
    ]
    for column in nondecreasing:
        assert np.all(np.diff(curve[column]) >= -1e-12), column
    assert np.all(np.diff(curve["fill_ratio"]) <= 1e-12)
    assert curve["participation_max"].max() <= _config().max_participation_rate


def test_capacity_is_order_invariant_and_supports_engine_column_mapping() -> None:
    panel = pd.DataFrame(
        {
            "fill_date": pd.to_datetime(["2025-02-04", "2025-02-03"]),
            "symbol": ["BBB", "AAA"],
            "requested_notional": [-25_000.0, 50_000.0],
            "filled_notional": [-20_000.0, 40_000.0],
            "adv_shares_lag1": [50_000.0, 100_000.0],
            "reference_price": [20.0, 10.0],
            "cost_estimate_bps_lag1": [12.0, 8.0],
        }
    )
    columns = CapacityColumns(
        date="fill_date",
        desired_notional="requested_notional",
        traded_notional="filled_notional",
        lagged_adv_notional=None,
        lagged_adv_shares="adv_shares_lag1",
        lagged_cost_bps="cost_estimate_bps_lag1",
    )
    config = CapacityConfig(
        reference_aum=500_000.0,
        aum_values=(500_000.0, 2_000_000.0),
        columns=columns,
    )

    result = estimate_capacity(panel, config)
    shuffled = estimate_capacity(panel.sample(frac=1.0, random_state=19), config)

    pdt.assert_frame_equal(result.curve, shuffled.curve, check_exact=True)
    pdt.assert_frame_equal(result.scenario_trades, shuffled.scenario_trades, check_exact=True)
    assert result.diagnostics.liquidity_source == "lagged_adv_shares_x_reference_price"
    assert result.diagnostics.cost_source == "lagged_cost_bps"
    assert result.diagnostics.temporal_validation == "caller_attested_lagged_inputs"
    assert result.scenario_trades["lagged_adv_notional"].unique().tolist() == [1_000_000.0]


def test_fill_record_preset_accepts_backtest_result_schema() -> None:
    fills = (
        _fill_panel()
        .drop(columns="information_date")
        .rename(columns={"date": "fill_date", "desired_notional": "requested_notional"})
    )
    config = CapacityConfig(
        reference_aum=1_000_000.0,
        aum_values=(1_000_000.0,),
        columns=CapacityColumns.for_fill_records(),
    )

    result = estimate_capacity(fills, config)

    assert result.curve.loc[0, "modeled_cost"] == pytest.approx(fills["total_cost"].sum())
    assert result.diagnostics.cost_source == "realized_total_cost"
    assert "ex-post anchored" in result.diagnostics.assumptions[-1]


@pytest.mark.parametrize(
    ("mutator", "message"),
    [
        (
            lambda frame: frame.assign(traded_notional=[101_000.0, -80_000.0, 40_000.0]),
            "cannot exceed desired",
        ),
        (
            lambda frame: frame.assign(traded_notional=[-80_000.0, -80_000.0, 40_000.0]),
            "same direction",
        ),
        (
            lambda frame: frame.assign(lagged_adv_notional=[np.nan, 1_000_000.0, 500_000.0]),
            "finite numeric",
        ),
        (
            lambda frame: frame.assign(information_date=["2025-01-06", "2025-01-03", "2025-01-06"]),
            "strictly before",
        ),
    ],
)
def test_capacity_rejects_inconsistent_or_noncausal_inputs(mutator, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        estimate_capacity(mutator(_fill_panel()), _config())


def test_capacity_config_rejects_false_precision_inputs() -> None:
    with pytest.raises(ValueError, match="aum_values"):
        CapacityConfig(reference_aum=1_000_000.0, aum_values=())
    with pytest.raises(ValueError, match="max_participation_rate"):
        CapacityConfig(
            reference_aum=1_000_000.0,
            aum_values=(1_000_000.0,),
            max_participation_rate=1.01,
        )
    with pytest.raises(ValueError, match="variable_cost_fraction"):
        CapacityConfig(
            reference_aum=1_000_000.0,
            aum_values=(1_000_000.0,),
            variable_cost_fraction=-0.1,
        )
