"""Invariant-focused tests for portfolio risk attribution (SF-S4-MR2/#10)."""

from __future__ import annotations

import math
from dataclasses import FrozenInstanceError, replace
from typing import Any

import numpy as np
import pandas as pd
import pytest

from alphaforge.optimization import risk_attribution
from alphaforge.optimization.risk_attribution import (
    MAX_ATTRIBUTION_ASSETS,
    MAX_SCENARIO_SIMPLE_RETURN,
    MAX_SCENARIOS,
    AssetRiskContribution,
    AssetWeightDrift,
    ExAnteRiskAttribution,
    FactorExposureDrift,
    FactorRiskContribution,
    RealizedAssetContribution,
    RiskAttributionError,
    ScenarioAnalysis,
    ScenarioAssetContribution,
    SpecificRiskContribution,
    attribute_ex_ante_risk,
    attribute_exposure_drift,
    attribute_realized_performance,
    evaluate_return_scenarios,
)
from alphaforge.optimization.risk_model import FactorRiskModel, validate_covariance

ASSETS = ("A", "B", "C")
FACTORS = ("market", "quality")


@pytest.fixture
def factor_model() -> FactorRiskModel:
    as_of = pd.Timestamp("2024-01-03", tz="UTC")
    loadings = np.array(
        [
            [1.0, 0.0],
            [0.5, 1.0],
            [-0.2, 0.3],
        ]
    )
    factor_covariance = np.array(
        [
            [4.0e-4, 1.0e-4],
            [1.0e-4, 3.0e-4],
        ]
    )
    specific = np.array([2.0e-4, 1.5e-4, 2.5e-4])
    covariance = loadings @ factor_covariance @ loadings.T + np.diag(specific)
    risk_model = validate_covariance(
        covariance,
        assets=ASSETS,
        periods_per_year=252,
        estimator="test_factor_model",
        as_of=as_of,
    )
    return FactorRiskModel(
        risk_model=risk_model,
        factors=FACTORS,
        loadings=loadings,
        factor_covariance=factor_covariance,
        specific_variances=specific,
        exposure_vintage=as_of - pd.Timedelta(days=1),
        exposure_available_at=as_of,
    )


def weights(values: tuple[float, float, float] = (0.5, 0.3, 0.2)) -> pd.Series:
    return pd.Series(values, index=ASSETS, dtype=float)


def returns(values: tuple[float, float, float] = (0.02, -0.01, 0.03)) -> pd.Series:
    return pd.Series(values, index=ASSETS, dtype=float)


def independently_observed_ending(
    allocation: pd.Series,
    observations: pd.Series,
    *,
    capital: float,
    cash_weight: float,
    cash_return: float,
    cost_return: float,
) -> float:
    """Compute an ending ledger without calling attribution code."""
    return math.fsum(
        [
            *(
                capital * float(weight) * (1.0 + float(asset_return))
                for weight, asset_return in zip(
                    allocation.to_numpy(), observations.to_numpy(), strict=True
                )
            ),
            capital * cash_weight * (1.0 + cash_return),
            -capital * cost_return,
        ]
    )


# ---------------------------------------------------------------------------
# Ex-ante Euler and factor/specific risk
# ---------------------------------------------------------------------------


def test_asset_euler_components_reconcile_variance_and_volatility(
    factor_model: FactorRiskModel,
) -> None:
    allocation = weights()
    result = attribute_ex_ante_risk(factor_model.risk_model, allocation)
    vector = allocation.to_numpy()
    covariance = factor_model.risk_model.covariance
    expected_variance = float(vector @ covariance @ vector)
    expected_volatility = float(np.sqrt(expected_variance))

    assert result.portfolio_variance == pytest.approx(expected_variance)
    assert result.portfolio_volatility == pytest.approx(expected_volatility)
    assert result.annualized_volatility == pytest.approx(expected_volatility * np.sqrt(252))
    assert sum(item.component_variance for item in result.asset_contributions) == pytest.approx(
        expected_variance
    )
    assert sum(item.component_volatility for item in result.asset_contributions) == pytest.approx(
        expected_volatility
    )
    np.testing.assert_allclose(
        [item.marginal_variance for item in result.asset_contributions],
        covariance @ vector,
    )
    assert result.variance_reconciliation_error == pytest.approx(0.0, abs=1e-15)
    assert result.volatility_reconciliation_error == pytest.approx(0.0, abs=1e-15)
    assert result.factor_variance is None
    assert result.specific_variance is None
    assert result.factor_contributions == ()
    assert result.specific_contributions == ()


def test_factor_and_specific_components_reconcile_total_risk(
    factor_model: FactorRiskModel,
) -> None:
    allocation = weights()
    result = attribute_ex_ante_risk(factor_model, allocation)
    vector = allocation.to_numpy()
    expected_exposures = factor_model.loadings.T @ vector
    expected_factor_variance = float(
        expected_exposures @ factor_model.factor_covariance @ expected_exposures
    )
    expected_specific = float(np.sum(vector**2 * factor_model.specific_variances))

    np.testing.assert_allclose(
        [item.exposure for item in result.factor_contributions],
        expected_exposures,
    )
    assert result.factor_variance == pytest.approx(expected_factor_variance)
    assert result.specific_variance == pytest.approx(expected_specific)
    assert result.factor_variance is not None
    assert result.specific_variance is not None
    assert result.factor_variance + result.specific_variance == pytest.approx(
        result.portfolio_variance
    )
    assert sum(item.component_variance for item in result.factor_contributions) == pytest.approx(
        expected_factor_variance
    )
    assert sum(item.component_variance for item in result.specific_contributions) == pytest.approx(
        expected_specific
    )
    assert sum(item.component_volatility for item in result.factor_contributions) + sum(
        item.component_volatility for item in result.specific_contributions
    ) == pytest.approx(result.portfolio_volatility)
    assert result.factor_specific_reconciliation_error == pytest.approx(0.0, abs=1e-15)
    assert list(result.asset_frame()["asset"]) == list(ASSETS)
    assert list(result.factor_frame()["factor"]) == list(FACTORS)


def test_zero_book_has_zero_finite_attribution(factor_model: FactorRiskModel) -> None:
    result = attribute_ex_ante_risk(factor_model, weights((0.0, 0.0, 0.0)))
    assert result.portfolio_variance == 0.0
    assert result.portfolio_volatility == 0.0
    assert result.annualized_volatility == 0.0
    assert all(item.marginal_volatility == 0.0 for item in result.asset_contributions)
    assert all(item.component_volatility == 0.0 for item in result.factor_contributions)
    assert all(item.component_volatility == 0.0 for item in result.specific_contributions)


def test_ex_ante_result_is_deeply_immutable(factor_model: FactorRiskModel) -> None:
    result = attribute_ex_ante_risk(factor_model, weights())
    with pytest.raises(FrozenInstanceError):
        result.portfolio_variance = 0.0  # type: ignore[misc]
    with pytest.raises(FrozenInstanceError):
        result.asset_contributions[0].weight = 0.0  # type: ignore[misc]
    assert isinstance(result.asset_contributions, tuple)
    assert result.to_dict()["risk_model_identity"] == factor_model.identity


@pytest.mark.parametrize(
    ("allocation", "message"),
    [
        (pd.Series([0.5, 0.3, 0.2], index=("B", "A", "C")), "exactly match"),
        (pd.Series([0.5, np.nan, 0.2], index=ASSETS), "finite"),
        (pd.Series([0.5, 0.3, 0.2], index=("A", "A", "C")), "unique"),
    ],
)
def test_ex_ante_attribution_rejects_misaligned_or_nonfinite_weights(
    factor_model: FactorRiskModel,
    allocation: pd.Series,
    message: str,
) -> None:
    with pytest.raises(RiskAttributionError, match=message):
        attribute_ex_ante_risk(factor_model, allocation)


# ---------------------------------------------------------------------------
# Realized return/P&L attribution with explicit aggregate cost
# ---------------------------------------------------------------------------


def test_realized_asset_and_cost_contributions_reconcile() -> None:
    allocation = weights()
    observations = returns()
    capital = 2_000_000.0
    observed_ending = independently_observed_ending(
        allocation,
        observations,
        capital=capital,
        cash_weight=0.0,
        cash_return=0.0,
        cost_return=0.001,
    )
    result = attribute_realized_performance(
        allocation,
        observations,
        initial_capital=capital,
        observed_ending_equity=observed_ending,
        cash_weight=0.0,
        cash_return=0.0,
        cost_return=0.001,
    )
    expected_gross = float(allocation @ observations)
    assert result.gross_return == pytest.approx(expected_gross)
    assert result.cost_return_contribution == pytest.approx(-0.001)
    assert result.net_return == pytest.approx(expected_gross - 0.001)
    assert result.gross_pnl == pytest.approx(expected_gross * 2_000_000.0)
    assert result.cost_pnl == pytest.approx(2_000.0)
    assert result.cost_pnl_contribution == pytest.approx(-2_000.0)
    assert result.net_pnl == pytest.approx(result.net_return * 2_000_000.0)
    assert result.modeled_ending_equity == pytest.approx(observed_ending)
    assert result.observed_ending_equity == observed_ending
    assert sum(
        item.return_contribution for item in result.asset_contributions
    ) + result.cash_return_contribution + result.cost_return_contribution == pytest.approx(
        result.net_return
    )
    assert sum(
        item.pnl_contribution for item in result.asset_contributions
    ) + result.cash_pnl_contribution + result.cost_pnl_contribution == pytest.approx(result.net_pnl)
    frame = result.contribution_frame()
    assert list(frame["asset"]) == [*ASSETS, "__cash__", "__cost__"]
    assert frame.iloc[-1]["pnl_contribution"] == pytest.approx(-2_000.0)


def test_realized_long_short_book_reconciles_explicit_financing_cash() -> None:
    allocation = weights((0.6, -0.2, 0.3))
    observations = returns((0.10, -0.05, 0.02))
    capital = 500_000.0
    cash_weight = 0.3
    cash_return = 0.002
    cost_return = 0.0005
    observed_ending = independently_observed_ending(
        allocation,
        observations,
        capital=capital,
        cash_weight=cash_weight,
        cash_return=cash_return,
        cost_return=cost_return,
    )

    result = attribute_realized_performance(
        allocation,
        observations,
        initial_capital=capital,
        observed_ending_equity=observed_ending,
        cash_weight=cash_weight,
        cash_return=cash_return,
        cost_return=cost_return,
    )

    assert result.starting_cash_weight == cash_weight
    assert result.cash_return_contribution == pytest.approx(cash_weight * cash_return)
    assert result.cash_pnl_contribution == pytest.approx(capital * cash_weight * cash_return)
    assert result.modeled_ending_equity == pytest.approx(observed_ending)
    assert result.ending_equity_reconciliation_error == pytest.approx(0.0, abs=1e-12)


def test_realized_attribution_rejects_observed_ending_mismatch() -> None:
    with pytest.raises(RiskAttributionError, match="did not reconcile"):
        attribute_realized_performance(
            weights(),
            returns(),
            initial_capital=1_000_000.0,
            observed_ending_equity=1_013_001.0,
            cash_weight=0.0,
            cash_return=0.0,
            cost_return=0.0,
        )


@pytest.mark.parametrize(
    ("observations", "kwargs", "message"),
    [
        (pd.Series([0.0, 0.0, 0.0], index=("B", "A", "C")), {}, "exactly match"),
        (pd.Series([0.0, np.nan, 0.0], index=ASSETS), {}, "finite"),
        (pd.Series([0.0, -1.01, 0.0], index=ASSETS), {}, "below -1"),
        (returns(), {"initial_capital": 0.0}, "positive"),
        (returns(), {"cost_return": -0.01}, "non-negative"),
        (returns(), {"cost_return": True}, "finite real"),
    ],
)
def test_realized_attribution_rejects_invalid_accounting_inputs(
    observations: pd.Series,
    kwargs: dict[str, float | bool],
    message: str,
) -> None:
    fields: dict[str, float | bool] = {
        "initial_capital": 1_000_000.0,
        "observed_ending_equity": 1_013_000.0,
        "cash_weight": 0.0,
        "cash_return": 0.0,
        "cost_return": 0.0,
    }
    fields.update(kwargs)
    with pytest.raises(RiskAttributionError, match=message):
        attribute_realized_performance(weights(), observations, **fields)


# ---------------------------------------------------------------------------
# Pre/post-return factor exposure drift
# ---------------------------------------------------------------------------


def test_return_drift_reconciles_cash_weights_and_factor_exposures(
    factor_model: FactorRiskModel,
) -> None:
    allocation = weights((0.6, -0.2, 0.3))
    observations = returns((0.10, -0.05, 0.02))
    result = attribute_exposure_drift(
        factor_model,
        allocation,
        observations,
        cash_weight=0.3,
        cash_return=0.001,
    )
    ending_values = allocation.to_numpy() * (1.0 + observations.to_numpy())
    ending_cash = 0.3 * 1.001
    ending_equity = float(ending_values.sum() + ending_cash)
    post_weights = ending_values / ending_equity

    assert result.ending_equity_multiplier == pytest.approx(ending_equity)
    np.testing.assert_allclose(
        [item.post_return_weight for item in result.asset_drifts],
        post_weights,
    )
    np.testing.assert_allclose(
        [item.pre_return_exposure for item in result.factor_drifts],
        factor_model.loadings.T @ allocation.to_numpy(),
    )
    np.testing.assert_allclose(
        [item.post_return_exposure for item in result.factor_drifts],
        factor_model.loadings.T @ post_weights,
    )
    assert post_weights.sum() + result.post_return_cash_weight == pytest.approx(1.0)
    assert result.gross_drift == pytest.approx(result.post_return_gross - result.pre_return_gross)
    assert result.net_drift == pytest.approx(result.post_return_net - result.pre_return_net)
    assert result.accounting_reconciliation_error == pytest.approx(0.0, abs=1e-15)
    assert list(result.asset_frame()["asset"]) == list(ASSETS)
    assert list(result.factor_frame()["factor"]) == list(FACTORS)


def test_return_drift_requires_explicit_balancing_cash(factor_model: FactorRiskModel) -> None:
    with pytest.raises(RiskAttributionError, match="cash_weight did not reconcile"):
        attribute_exposure_drift(
            factor_model,
            weights((0.6, -0.2, 0.3)),
            returns(),
            cash_weight=0.0,
        )


def test_return_drift_rejects_nonpositive_ending_equity(factor_model: FactorRiskModel) -> None:
    with pytest.raises(RiskAttributionError, match="positive"):
        attribute_exposure_drift(
            factor_model,
            weights((1.0, 0.0, 0.0)),
            returns((-1.0, 0.0, 0.0)),
            cash_weight=0.0,
        )


@pytest.mark.parametrize("cash_return", [-1.01, np.inf])
def test_return_drift_rejects_invalid_cash_return(
    factor_model: FactorRiskModel,
    cash_return: float,
) -> None:
    with pytest.raises(RiskAttributionError, match="cash_return"):
        attribute_exposure_drift(
            factor_model,
            weights(),
            returns(),
            cash_weight=0.0,
            cash_return=cash_return,
        )


# ---------------------------------------------------------------------------
# Bounded named scenarios
# ---------------------------------------------------------------------------


def test_named_scenarios_are_sorted_and_contributions_reconcile() -> None:
    allocation = weights((0.6, -0.2, 0.3))
    capital = 5_000_000.0
    analysis = evaluate_return_scenarios(
        allocation,
        {
            "selloff": returns((-0.20, -0.10, -0.15)),
            "rally": returns((0.10, 0.08, 0.12)),
        },
        initial_capital=capital,
        cash_weight=0.3,
        cash_return=0.001,
        cost_return=0.0004,
    )
    assert [result.name for result in analysis.results] == ["rally", "selloff"]
    for result in analysis.results:
        assert (
            sum(item.return_contribution for item in result.contributions)
            + result.cash_return_contribution
            + result.cost_return_contribution
        ) == pytest.approx(result.modeled_portfolio_return)
        assert (
            sum(item.pnl_contribution for item in result.contributions)
            + result.cash_pnl_contribution
            + result.cost_pnl_contribution
        ) == pytest.approx(result.modeled_scenario_pnl)
        assert result.modeled_scenario_pnl == pytest.approx(
            result.modeled_portfolio_return * capital
        )
        assert result.modeled_ending_equity == pytest.approx(capital + result.modeled_scenario_pnl)
        assert result.ending_value_reconciliation_error == pytest.approx(0.0, abs=1e-12)
    frame = analysis.contribution_frame()
    assert list(frame["scenario"].drop_duplicates()) == ["rally", "selloff"]
    assert len(frame) == (len(ASSETS) + 2) * 2
    assert set(frame["asset"].tail(2)) == {"__cash__", "__cost__"}
    with pytest.raises(FrozenInstanceError):
        analysis.results[0].modeled_portfolio_return = 0.0  # type: ignore[misc]


@pytest.mark.parametrize("name", ["", " leading", "trailing ", "x" * 129, 7])
def test_scenarios_reject_invalid_names(name: object) -> None:
    with pytest.raises(RiskAttributionError, match="scenario name"):
        evaluate_return_scenarios(
            weights(),
            {name: returns()},  # type: ignore[dict-item]
            initial_capital=1_000_000.0,
            cash_weight=0.0,
            cash_return=0.0,
            cost_return=0.0,
        )


def test_scenarios_enforce_count_and_asset_resource_bounds() -> None:
    with pytest.raises(RiskAttributionError, match="scenario count"):
        evaluate_return_scenarios(
            weights(),
            {},
            initial_capital=1_000_000.0,
            cash_weight=0.0,
            cash_return=0.0,
            cost_return=0.0,
        )
    too_many = {f"scenario-{index:03d}": returns() for index in range(MAX_SCENARIOS + 1)}
    with pytest.raises(RiskAttributionError, match="scenario count"):
        evaluate_return_scenarios(
            weights(),
            too_many,
            initial_capital=1_000_000.0,
            cash_weight=0.0,
            cash_return=0.0,
            cost_return=0.0,
        )

    large_weights = pd.Series(
        np.zeros(MAX_ATTRIBUTION_ASSETS + 1),
        index=[f"A{index}" for index in range(MAX_ATTRIBUTION_ASSETS + 1)],
    )
    with pytest.raises(RiskAttributionError, match="weight assets count"):
        evaluate_return_scenarios(
            large_weights,
            {"flat": pd.Series(0.0, index=large_weights.index)},
            initial_capital=1_000_000.0,
            cash_weight=1.0,
            cash_return=0.0,
            cost_return=0.0,
        )


@pytest.mark.parametrize(
    ("scenario", "message"),
    [
        (pd.Series([0.0, 0.0, 0.0], index=("B", "A", "C")), "exactly match"),
        (pd.Series([0.0, np.nan, 0.0], index=ASSETS), "finite"),
        (pd.Series([-1.01, 0.0, 0.0], index=ASSETS), "below -1"),
        (
            pd.Series([MAX_SCENARIO_SIMPLE_RETURN + 0.01, 0.0, 0.0], index=ASSETS),
            "exceeds",
        ),
    ],
)
def test_scenarios_reject_invalid_shock_vectors(
    scenario: pd.Series,
    message: str,
) -> None:
    with pytest.raises(RiskAttributionError, match=message):
        evaluate_return_scenarios(
            weights(),
            {"invalid": scenario},
            initial_capital=1_000_000.0,
            cash_weight=0.0,
            cash_return=0.0,
            cost_return=0.0,
        )


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"cash_weight": 0.1}, "starting weights plus cash"),
        ({"cash_return": -1.01}, "cash_return"),
        ({"cost_return": -0.001}, "non-negative"),
    ],
)
def test_scenarios_reject_invalid_financing_inputs(
    kwargs: dict[str, float],
    message: str,
) -> None:
    fields = {
        "initial_capital": 1_000_000.0,
        "cash_weight": 0.0,
        "cash_return": 0.0,
        "cost_return": 0.0,
    }
    fields.update(kwargs)
    with pytest.raises(RiskAttributionError, match=message):
        evaluate_return_scenarios(weights(), {"flat": returns()}, **fields)


# ---------------------------------------------------------------------------
# Direct-construction and numerical-boundary hardening
# ---------------------------------------------------------------------------


def test_scale_aware_reconciliation_rejects_tiny_fixed_atol_attack() -> None:
    variance = 2.0e-36
    volatility = math.sqrt(variance)
    asset = AssetRiskContribution(
        asset="A",
        weight=1.0,
        marginal_variance=variance,
        component_variance=variance,
        marginal_volatility=volatility,
        component_volatility=volatility,
        variance_fraction=1.0,
    )
    factor = FactorRiskContribution(
        factor="market",
        exposure=1.0,
        marginal_variance=1.0e-32,
        component_variance=1.0e-32,
        component_volatility=1.0e-32 / volatility,
        variance_fraction=5_000.0,
    )
    specific = SpecificRiskContribution(
        asset="A",
        specific_variance=1.0e-32,
        component_variance=1.0e-32,
        component_volatility=1.0e-32 / volatility,
        variance_fraction=5_000.0,
    )

    with pytest.raises(RiskAttributionError, match="factor plus specific variance"):
        ExAnteRiskAttribution(
            risk_model_identity="0" * 64,
            assets=("A",),
            periods_per_year=252,
            portfolio_variance=variance,
            portfolio_volatility=volatility,
            annualized_volatility=volatility * math.sqrt(252.0),
            variance_reconciliation_error=0.0,
            volatility_reconciliation_error=0.0,
            factor_variance=1.0e-32,
            specific_variance=1.0e-32,
            factor_specific_reconciliation_error=0.0,
            asset_contributions=(asset,),
            factor_contributions=(factor,),
            specific_contributions=(specific,),
        )


def test_scale_aware_reconciliation_handles_exact_zero_and_rejects_tiny_mismatch() -> None:
    assert (
        risk_attribution._reconciliation_error(  # noqa: SLF001 - invariant regression
            0.0,
            0.0,
            name="exact zero",
            term_magnitudes=(0.0,),
            operations=1,
        )
        == 0.0
    )
    with pytest.raises(RiskAttributionError, match="tiny mismatch did not reconcile"):
        risk_attribution._reconciliation_error(  # noqa: SLF001 - invariant regression
            1.0e-300,
            0.0,
            name="tiny mismatch",
            term_magnitudes=(1.0e-300,),
            operations=1,
        )


def test_exported_records_canonicalize_direct_mutable_sequences(
    factor_model: FactorRiskModel,
) -> None:
    allocation = weights()
    observations = returns()
    ex_ante = attribute_ex_ante_risk(factor_model, allocation)
    observed_ending = independently_observed_ending(
        allocation,
        observations,
        capital=1_000_000.0,
        cash_weight=0.0,
        cash_return=0.0,
        cost_return=0.0,
    )
    realized = attribute_realized_performance(
        allocation,
        observations,
        initial_capital=1_000_000.0,
        observed_ending_equity=observed_ending,
        cash_weight=0.0,
        cash_return=0.0,
        cost_return=0.0,
    )
    drift = attribute_exposure_drift(
        factor_model,
        allocation,
        observations,
        cash_weight=0.0,
    )
    scenarios = evaluate_return_scenarios(
        allocation,
        {"flat": returns((0.0, 0.0, 0.0))},
        initial_capital=1_000_000.0,
        cash_weight=0.0,
        cash_return=0.0,
        cost_return=0.0,
    )

    rebuilt_ex_ante = replace(
        ex_ante,
        assets=list(ex_ante.assets),  # type: ignore[arg-type]
        asset_contributions=list(ex_ante.asset_contributions),  # type: ignore[arg-type]
        factor_contributions=list(ex_ante.factor_contributions),  # type: ignore[arg-type]
        specific_contributions=list(ex_ante.specific_contributions),  # type: ignore[arg-type]
    )
    rebuilt_realized = replace(
        realized,
        assets=list(realized.assets),  # type: ignore[arg-type]
        asset_contributions=list(realized.asset_contributions),  # type: ignore[arg-type]
    )
    rebuilt_drift = replace(
        drift,
        assets=list(drift.assets),  # type: ignore[arg-type]
        factors=list(drift.factors),  # type: ignore[arg-type]
        asset_drifts=list(drift.asset_drifts),  # type: ignore[arg-type]
        factor_drifts=list(drift.factor_drifts),  # type: ignore[arg-type]
    )
    rebuilt_scenario = replace(
        scenarios.results[0],
        contributions=list(scenarios.results[0].contributions),  # type: ignore[arg-type]
    )
    rebuilt_analysis = ScenarioAnalysis(
        assets=list(scenarios.assets),  # type: ignore[arg-type]
        initial_capital=np.float64(scenarios.initial_capital),
        starting_cash_weight=np.float64(scenarios.starting_cash_weight),
        cash_return=np.float64(scenarios.cash_return),
        cost_return=np.float64(scenarios.cost_return),
        results=[rebuilt_scenario],  # type: ignore[arg-type]
    )

    for sequence in (
        rebuilt_ex_ante.assets,
        rebuilt_ex_ante.asset_contributions,
        rebuilt_realized.assets,
        rebuilt_realized.asset_contributions,
        rebuilt_drift.assets,
        rebuilt_drift.factors,
        rebuilt_drift.asset_drifts,
        rebuilt_drift.factor_drifts,
        rebuilt_scenario.contributions,
        rebuilt_analysis.results,
    ):
        assert isinstance(sequence, tuple)
    assert isinstance(rebuilt_analysis.initial_capital, float)


def test_every_exported_record_rejects_inconsistent_direct_state(
    factor_model: FactorRiskModel,
) -> None:
    allocation = weights()
    observations = returns()
    ex_ante = attribute_ex_ante_risk(factor_model, allocation)
    observed_ending = independently_observed_ending(
        allocation,
        observations,
        capital=1_000_000.0,
        cash_weight=0.0,
        cash_return=0.0,
        cost_return=0.0,
    )
    realized = attribute_realized_performance(
        allocation,
        observations,
        initial_capital=1_000_000.0,
        observed_ending_equity=observed_ending,
        cash_weight=0.0,
        cash_return=0.0,
        cost_return=0.0,
    )
    drift = attribute_exposure_drift(
        factor_model,
        allocation,
        observations,
        cash_weight=0.0,
    )
    scenarios = evaluate_return_scenarios(
        allocation,
        {"flat": returns((0.0, 0.0, 0.0))},
        initial_capital=1_000_000.0,
        cash_weight=0.0,
        cash_return=0.0,
        cost_return=0.0,
    )

    invalid_builders: tuple[Any, ...] = (
        lambda: replace(
            ex_ante.asset_contributions[0],
            component_variance=ex_ante.asset_contributions[0].component_variance + 1.0,
        ),
        lambda: replace(
            ex_ante.factor_contributions[0],
            component_variance=ex_ante.factor_contributions[0].component_variance + 1.0,
        ),
        lambda: replace(ex_ante.specific_contributions[0], component_variance=-1.0),
        lambda: replace(ex_ante, portfolio_variance=ex_ante.portfolio_variance + 1.0),
        lambda: replace(
            realized.asset_contributions[0],
            return_contribution=realized.asset_contributions[0].return_contribution + 1.0,
        ),
        lambda: replace(realized, observed_ending_equity=observed_ending + 1.0),
        lambda: replace(drift.asset_drifts[0], drift=drift.asset_drifts[0].drift + 1.0),
        lambda: replace(drift.factor_drifts[0], drift=drift.factor_drifts[0].drift + 1.0),
        lambda: replace(
            drift,
            post_return_cash_weight=drift.post_return_cash_weight + 0.1,
        ),
        lambda: replace(
            scenarios.results[0].contributions[0],
            return_contribution=1.0,
        ),
        lambda: replace(
            scenarios.results[0],
            modeled_ending_equity=scenarios.results[0].modeled_ending_equity + 1.0,
        ),
        lambda: ScenarioAnalysis(
            assets=["A"],  # type: ignore[arg-type]
            initial_capital=-1.0,
            starting_cash_weight=0.0,
            cash_return=0.0,
            cost_return=0.0,
            results=[],  # type: ignore[arg-type]
        ),
    )
    for build in invalid_builders:
        with pytest.raises(RiskAttributionError):
            build()


class _OverflowingFloat:
    def __float__(self) -> float:
        raise OverflowError("synthetic overflow")


def test_numeric_boundaries_translate_overflow_to_domain_errors() -> None:
    value: Any = _OverflowingFloat()
    with pytest.raises(RiskAttributionError, match="finite real"):
        AssetWeightDrift(
            asset="A",
            pre_return_weight=value,
            post_return_weight=0.0,
            drift=0.0,
        )

    huge = pd.Series([10**10_000, 0, 0], index=ASSETS, dtype=object)
    with pytest.raises(RiskAttributionError, match="numeric"):
        evaluate_return_scenarios(
            huge,
            {"flat": returns((0.0, 0.0, 0.0))},
            initial_capital=1_000_000.0,
            cash_weight=0.0,
            cash_return=0.0,
            cost_return=0.0,
        )


def test_component_records_canonicalize_numpy_scalars() -> None:
    realized = RealizedAssetContribution(
        asset="A",
        starting_weight=np.float64(0.5),
        asset_return=np.float64(0.1),
        return_contribution=np.float64(0.05),
        pnl_contribution=np.float64(5.0),
    )
    weight_drift = AssetWeightDrift(
        asset="A",
        pre_return_weight=np.float64(0.5),
        post_return_weight=np.float64(0.6),
        drift=np.float64(0.1),
    )
    factor_drift = FactorExposureDrift(
        factor="market",
        pre_return_exposure=np.float64(0.2),
        post_return_exposure=np.float64(0.3),
        drift=np.float64(0.1),
    )
    scenario = ScenarioAssetContribution(
        asset="A",
        starting_weight=np.float64(0.5),
        scenario_return=np.float64(-0.1),
        return_contribution=np.float64(-0.05),
        pnl_contribution=np.float64(-5.0),
    )

    for value in (
        realized.starting_weight,
        realized.asset_return,
        weight_drift.drift,
        factor_drift.drift,
        scenario.scenario_return,
    ):
        assert type(value) is float


# ---------------------------------------------------------------------------
# Regression: residual-vs-residual tolerance (surfaced on CPython 3.14)
# ---------------------------------------------------------------------------


def test_residual_comparison_is_scaled_by_the_reconciled_quantity() -> None:
    """A stored residual must be judged against the scale of what it reconciles.

    Both operands of `_canonical_error` are roundoff residuals, so the residual's
    own magnitude is noise and cannot bound the comparison: a residual of 1.9e-9
    would earn a bound of 1.9e-18 and any two legitimately-different float
    accumulation paths would be reported as a reconciliation failure. This
    reproduces the exact values that failed on CPython 3.14.
    """
    from alphaforge.optimization.risk_attribution import _canonical_error

    observed_residual = -1.862645149230957e-09  # 2**-29, from an equity near 1e6

    # Without a scale the comparison is judged against the residual itself and
    # is unsatisfiable.
    with pytest.raises(RiskAttributionError, match="did not reconcile"):
        _canonical_error(observed_residual, 0.0, name="ending_value_reconciliation_error")

    # Given the equity it reconciles, the same difference is ordinary roundoff.
    assert (
        _canonical_error(
            observed_residual,
            0.0,
            name="ending_value_reconciliation_error",
            scale=1_000_000.0,
        )
        == 0.0
    )


def test_residual_tolerance_still_rejects_a_materially_wrong_value() -> None:
    """Widening the bound must not make the integrity check vacuous."""
    from alphaforge.optimization.risk_attribution import _canonical_error

    with pytest.raises(RiskAttributionError, match="did not reconcile"):
        _canonical_error(1e-3, 0.0, name="ending_value_reconciliation_error", scale=1_000_000.0)
