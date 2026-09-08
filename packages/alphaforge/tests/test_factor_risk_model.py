"""Adversarial and causal tests for the Sprint 4 factor-risk contract."""

from __future__ import annotations

from dataclasses import replace
from typing import Any, cast

import numpy as np
import pandas as pd
import pytest

from alphaforge.optimization.risk_model import (
    CONDITION_LIMIT,
    MAX_ASSETS,
    MAX_SOURCE_CELLS,
    MAX_SOURCE_OBSERVATIONS,
    FactorRiskModel,
    RiskModel,
    RiskModelError,
    estimate_factor_risk_model,
    shrinkage_covariance,
    validate_covariance,
)


def _factor_market(seed: int = 71) -> tuple[pd.DataFrame, pd.DataFrame, pd.Timestamp]:
    generator = np.random.default_rng(seed)
    dates = pd.date_range("2023-01-02", periods=180, freq="B", tz="UTC")
    assets = [f"A{position:02d}" for position in range(8)]
    factors = ["market", "quality"]
    loadings = pd.DataFrame(
        generator.normal(size=(len(assets), len(factors))),
        index=assets,
        columns=factors,
    )
    factor_returns = generator.normal(scale=[0.009, 0.006], size=(len(dates), 2))
    specific = generator.normal(scale=0.004, size=(len(dates), len(assets)))
    returns = pd.DataFrame(
        factor_returns @ loadings.to_numpy().T + specific,
        index=dates,
        columns=assets,
    )
    return returns, loadings, dates[140]


def _estimate(
    returns: pd.DataFrame,
    loadings: pd.DataFrame,
    *,
    as_of: pd.Timestamp,
    window: int = 100,
    factor_shrinkage: float = 0.2,
    specific_shrinkage: float = 0.2,
    exposure_vintage: pd.Timestamp | None = None,
    exposure_available_at: pd.Timestamp | None = None,
) -> FactorRiskModel:
    """Estimate with an explicit, causal synthetic exposure publication."""
    return estimate_factor_risk_model(
        returns,
        loadings,
        as_of=as_of,
        exposure_vintage=(
            as_of - pd.Timedelta(days=1) if exposure_vintage is None else exposure_vintage
        ),
        exposure_available_at=(as_of if exposure_available_at is None else exposure_available_at),
        window=window,
        factor_shrinkage=factor_shrinkage,
        specific_shrinkage=specific_shrinkage,
    )


def test_factor_model_reconstructs_covariance_and_carries_coverage() -> None:
    returns, loadings, as_of = _factor_market()
    model = _estimate(
        returns,
        loadings,
        as_of=as_of,
        window=100,
        factor_shrinkage=0.3,
        specific_shrinkage=0.25,
    )

    reconstructed = model.loadings @ model.factor_covariance @ model.loadings.T + np.diag(
        model.specific_variances
    )
    np.testing.assert_allclose(reconstructed, model.risk_model.covariance, rtol=1e-10)
    assert model.risk_model.as_of == as_of
    assert model.risk_model.window_end is not None
    assert model.risk_model.window_end < as_of
    assert model.risk_model.n_observations == 100
    assert model.risk_model.source_hash is not None
    assert len(model.risk_model.source_hash) == 64


def test_future_returns_cannot_change_factor_risk_snapshot() -> None:
    returns, loadings, as_of = _factor_market()
    baseline = _estimate(returns, loadings, as_of=as_of, window=100)
    mutated = returns.copy()
    mutated.loc[mutated.index >= as_of] = 1_000.0
    changed = _estimate(mutated, loadings, as_of=as_of, window=100)

    np.testing.assert_array_equal(baseline.risk_model.covariance, changed.risk_model.covariance)
    np.testing.assert_array_equal(baseline.loadings, changed.loadings)
    assert baseline.risk_model.identity == changed.risk_model.identity


def test_covariance_coverage_reports_missing_rows_and_dropped_assets() -> None:
    returns, _, as_of = _factor_market()
    returns["ALL_MISSING"] = np.nan
    returns.loc[returns.index[80], "A00"] = np.nan
    model = shrinkage_covariance(returns, as_of=as_of, window=100)
    diagnostics = model.diagnostics()

    assert diagnostics["observations_considered"] == 100
    assert diagnostics["n_observations"] == 99
    assert diagnostics["observations_dropped"] == 1
    assert diagnostics["complete_observation_fraction"] == pytest.approx(0.99)
    assert diagnostics["dropped_assets"] == ["ALL_MISSING"]


def test_factor_model_is_deterministic_under_asset_permutation() -> None:
    returns, loadings, as_of = _factor_market()
    baseline = _estimate(returns, loadings, as_of=as_of, window=100)
    order = list(reversed(returns.columns))
    permuted = _estimate(returns.loc[:, order], loadings.reindex(order), as_of=as_of, window=100)
    restored = permuted.risk_model.to_frame().reindex(
        index=baseline.risk_model.assets, columns=baseline.risk_model.assets
    )
    np.testing.assert_allclose(restored, baseline.risk_model.covariance, rtol=1e-11, atol=1e-14)


def test_factor_exposures_match_direct_matrix_product() -> None:
    returns, loadings, as_of = _factor_market()
    model = _estimate(returns, loadings, as_of=as_of, window=100)
    weights = np.linspace(-0.2, 0.2, len(model.risk_model.assets))
    np.testing.assert_allclose(model.factor_exposures(weights), model.loadings.T @ weights)


@pytest.mark.parametrize(
    "matrix, message",
    [
        (np.diag([1.0, -100.0]), "materially indefinite"),
        (np.array([[1.0, 1.0], [1.0, 1.0]]), "singular"),
        (np.diag([1.0, 1e-12]), "ill-conditioned"),
    ],
)
def test_invalid_covariance_fails_without_silent_strategy_change(
    matrix: np.ndarray, message: str
) -> None:
    with pytest.raises(RiskModelError, match=message):
        validate_covariance(matrix, assets=("A", "B"), allow_ridge=True)


def test_direct_constructor_cannot_bypass_covariance_validation() -> None:
    with pytest.raises(RiskModelError, match="symmetric"):
        RiskModel(
            assets=("A", "B"),
            covariance=np.array([[1.0, 2.0], [0.0, 1.0]]),
            periods_per_year=252,
            min_eigenvalue=1.0,
            condition_number=1.0,
            ridge_applied=0.0,
            estimator="forged",
        )


def test_validated_arrays_are_defensively_copied_and_read_only() -> None:
    source = np.diag([0.02, 0.03])
    model = validate_covariance(source, assets=("A", "B"))
    source[0, 0] = np.nan
    assert np.isfinite(model.covariance).all()
    with pytest.raises(ValueError, match="read-only"):
        model.covariance[0, 0] = 5.0


def test_source_identity_changes_for_a_single_float_bit() -> None:
    returns, _, as_of = _factor_market()
    baseline = shrinkage_covariance(returns, as_of=as_of, window=100)
    mutated = returns.copy()
    # Pick an observation inside the 100-row training window ending at as_of.
    location = mutated.index[80], mutated.columns[0]
    mutated.loc[location] = np.nextafter(mutated.loc[location], np.inf)
    changed = shrinkage_covariance(mutated, as_of=as_of, window=100)
    assert baseline.source_hash != changed.source_hash
    assert baseline.identity != changed.identity


def test_factor_input_failures_are_actionable() -> None:
    returns, loadings, as_of = _factor_market()
    with pytest.raises(RiskModelError, match="rank deficient"):
        duplicated = loadings.assign(copy=loadings["market"])
        _estimate(returns, duplicated, as_of=as_of, window=100)
    with pytest.raises(RiskModelError, match="missing return assets"):
        _estimate(returns, loadings.drop(index=returns.columns[0]), as_of=as_of, window=100)


def test_factor_snapshot_reports_temporal_provenance_and_stabilization() -> None:
    returns, loadings, as_of = _factor_market()
    vintage = as_of - pd.Timedelta(days=2)
    available = as_of - pd.Timedelta(hours=1)
    model = _estimate(
        returns,
        loadings,
        as_of=as_of,
        exposure_vintage=vintage,
        exposure_available_at=available,
    )
    diagnostics = model.diagnostics()

    assert diagnostics["exposure_vintage"] == vintage.isoformat()
    assert diagnostics["exposure_available_at"] == available.isoformat()
    assert diagnostics["factor_variance_floor"] > 0.0
    assert diagnostics["specific_variance_floor"] > 0.0
    assert diagnostics["factor_directions_stabilized"] >= 0
    assert diagnostics["specific_assets_stabilized"] >= 0


def test_temporal_provenance_changes_identity_and_rejects_future_information() -> None:
    returns, loadings, as_of = _factor_market()
    baseline = _estimate(returns, loadings, as_of=as_of)
    earlier = _estimate(
        returns,
        loadings,
        as_of=as_of,
        exposure_vintage=as_of - pd.Timedelta(days=2),
    )
    assert baseline.identity != earlier.identity
    assert baseline.risk_model.source_hash != earlier.risk_model.source_hash

    with pytest.raises(RiskModelError, match="available_at must not follow as_of"):
        _estimate(
            returns,
            loadings,
            as_of=as_of,
            exposure_available_at=as_of + pd.Timedelta(seconds=1),
        )
    with pytest.raises(RiskModelError, match="vintage must not follow"):
        _estimate(
            returns,
            loadings,
            as_of=as_of,
            exposure_vintage=as_of,
            exposure_available_at=as_of - pd.Timedelta(seconds=1),
        )
    with pytest.raises(RiskModelError, match="timezone semantics"):
        estimate_factor_risk_model(
            returns,
            loadings,
            as_of=as_of,
            exposure_vintage=as_of.tz_localize(None),
            exposure_available_at=as_of,
            window=100,
        )


def test_factor_names_are_part_of_source_and_factor_identity() -> None:
    returns, loadings, as_of = _factor_market()
    baseline = _estimate(returns, loadings, as_of=as_of)
    renamed = loadings.rename(columns={"market": "renamed-market"})
    changed = _estimate(returns, renamed, as_of=as_of)

    assert baseline.risk_model.source_hash != changed.risk_model.source_hash
    assert baseline.identity != changed.identity


def test_immutable_array_backing_cannot_be_reenabled_with_setflags() -> None:
    returns, loadings, as_of = _factor_market()
    factor_model = _estimate(returns, loadings, as_of=as_of)

    for array in (
        factor_model.risk_model.covariance,
        factor_model.loadings,
        factor_model.factor_covariance,
        factor_model.specific_variances,
    ):
        with pytest.raises(ValueError):
            array.setflags(write=True)


def test_stale_asset_is_classified_inside_the_trailing_window() -> None:
    returns, loadings, as_of = _factor_market()
    stale = np.full(len(returns), np.nan)
    stale[:20] = np.linspace(-0.01, 0.01, 20)
    returns["STALE"] = stale

    model = _estimate(returns, loadings, as_of=as_of, window=100)

    assert "STALE" not in model.risk_model.assets
    assert model.risk_model.dropped_assets == ("STALE",)


def test_raw_source_dimensions_are_refused_before_estimation() -> None:
    too_many_rows = pd.DataFrame(np.zeros((MAX_SOURCE_OBSERVATIONS + 1, 1)), columns=["A"])
    with pytest.raises(RiskModelError, match="raw return observation count"):
        shrinkage_covariance(too_many_rows, window=10)

    too_many_assets = pd.DataFrame(
        np.zeros((10, MAX_ASSETS + 1)),
        columns=[f"A{position}" for position in range(MAX_ASSETS + 1)],
    )
    with pytest.raises(RiskModelError, match="raw return asset count"):
        shrinkage_covariance(too_many_assets, window=10)

    jointly_oversized = pd.DataFrame(
        np.zeros((MAX_SOURCE_CELLS // MAX_ASSETS + 1, MAX_ASSETS)),
        columns=[f"A{position}" for position in range(MAX_ASSETS)],
    )
    with pytest.raises(RiskModelError, match="cell ceiling"):
        shrinkage_covariance(jointly_oversized, window=10)


def test_ndarray_covariance_size_is_checked_before_defensive_copy() -> None:
    scalar = np.array([1.0])
    oversized_view = np.lib.stride_tricks.as_strided(
        scalar,
        shape=(MAX_ASSETS + 1, MAX_ASSETS + 1),
        strides=(0, 0),
        writeable=False,
    )
    with pytest.raises(RiskModelError, match="2000-asset ceiling"):
        validate_covariance(
            oversized_view,
            assets=tuple(f"A{position}" for position in range(MAX_ASSETS + 1)),
        )


def test_exposure_dimensions_are_refused_before_return_alignment() -> None:
    returns, _, as_of = _factor_market()
    oversized = pd.DataFrame(
        np.ones((MAX_ASSETS + 1, 1)),
        index=[f"A{position}" for position in range(MAX_ASSETS + 1)],
        columns=["factor"],
    )
    with pytest.raises(RiskModelError, match="exposure asset count"):
        _estimate(returns, oversized, as_of=as_of)


def _rank_one_factor_market() -> tuple[pd.DataFrame, pd.DataFrame, pd.Timestamp]:
    generator = np.random.default_rng(991)
    dates = pd.date_range("2024-01-02", periods=90, freq="B", tz="UTC")
    assets = [f"A{position}" for position in range(6)]
    loadings_array = generator.normal(size=(6, 2))
    loadings = pd.DataFrame(loadings_array, index=assets, columns=["f1", "f2"])
    _, _, right_vectors = np.linalg.svd(loadings_array.T, full_matrices=True)
    residual_direction = right_vectors[-1]
    common = generator.normal(scale=0.01, size=len(dates))
    factor_returns = np.column_stack((common, common))
    residual_scale = generator.normal(scale=0.003, size=len(dates))
    returns = pd.DataFrame(
        factor_returns @ loadings_array.T + np.outer(residual_scale, residual_direction),
        index=dates,
        columns=assets,
    )
    return returns, loadings, dates[-1]


def test_factor_floor_fails_closed_at_zero_shrinkage_and_is_disclosed() -> None:
    returns, loadings, as_of = _rank_one_factor_market()
    with pytest.raises(RiskModelError, match="factor_shrinkage is zero"):
        _estimate(
            returns,
            loadings,
            as_of=as_of,
            window=60,
            factor_shrinkage=0.0,
        )

    model = _estimate(
        returns,
        loadings,
        as_of=as_of,
        window=60,
        factor_shrinkage=1e-12,
    )
    assert model.factor_directions_stabilized == 1
    assert model.factor_ridge_applied > 0.0
    assert model.factor_variance_floor > 0.0
    assert np.linalg.cond(model.factor_covariance) < CONDITION_LIMIT * (1.0 - 1e-6)


def _zero_specific_variance_market() -> tuple[pd.DataFrame, pd.DataFrame, pd.Timestamp]:
    generator = np.random.default_rng(781)
    dates = pd.date_range("2024-01-02", periods=90, freq="B", tz="UTC")
    assets = ["A", "B", "C", "D", "E", "F"]
    loadings = pd.DataFrame(np.ones((6, 1)), index=assets, columns=["market"])
    # Keep common risk below the residual scale so the disclosed specific floor
    # is sufficient for the assembled asset covariance's condition limit.
    common = generator.normal(scale=1e-8, size=len(dates))
    residual = generator.normal(scale=0.003, size=len(dates))
    residual_direction = np.array([0.0, 1.0, 1.0, -1.0, -1.0, 0.0])
    returns = pd.DataFrame(
        np.outer(common, np.ones(6)) + np.outer(residual, residual_direction),
        index=dates,
        columns=assets,
    )
    return returns, loadings, dates[-1]


def test_specific_floor_fails_closed_at_zero_shrinkage_and_is_disclosed() -> None:
    returns, loadings, as_of = _zero_specific_variance_market()
    with pytest.raises(RiskModelError, match="specific_shrinkage is zero"):
        _estimate(
            returns,
            loadings,
            as_of=as_of,
            window=60,
            specific_shrinkage=0.0,
        )

    model = _estimate(
        returns,
        loadings,
        as_of=as_of,
        window=60,
        specific_shrinkage=1e-12,
    )
    assert model.specific_assets_stabilized == 2
    assert model.specific_stabilization_total > 0.0
    assert model.specific_variance_floor > 0.0
    with pytest.raises(RiskModelError, match="plausible bound"):
        replace(
            model,
            specific_stabilization_total=(
                model.specific_assets_stabilized * model.specific_variance_floor * 2.0
            ),
        )


def test_direct_factor_model_uses_scale_relative_reconciliation() -> None:
    as_of = pd.Timestamp("2025-01-02", tz="UTC")
    loadings = np.array([[1.0], [2.0]])
    factor_covariance = np.array([[1e-20]])
    specific = np.array([1e-22, 2e-22])
    covariance = loadings @ factor_covariance @ loadings.T + np.diag(specific)
    risk_model = validate_covariance(covariance, assets=("A", "B"), as_of=as_of)

    exact = FactorRiskModel(
        risk_model=risk_model,
        factors=("tiny",),
        loadings=loadings,
        factor_covariance=factor_covariance,
        specific_variances=specific,
        exposure_vintage=as_of - pd.Timedelta(days=1),
        exposure_available_at=as_of,
    )
    assert exact.factor_covariance[0, 0] == pytest.approx(1e-20)

    with pytest.raises(RiskModelError, match="factor decomposition does not reconcile"):
        FactorRiskModel(
            risk_model=risk_model,
            factors=("tiny",),
            loadings=loadings,
            factor_covariance=np.array([[1e-24]]),
            specific_variances=specific,
            exposure_vintage=as_of - pd.Timedelta(days=1),
            exposure_available_at=as_of,
        )


def test_direct_factor_model_rejects_rank_deficient_exposures() -> None:
    as_of = pd.Timestamp("2025-01-02", tz="UTC")
    loadings = np.array([[1.0, 1.0], [2.0, 2.0], [3.0, 3.0]])
    factor_covariance = np.eye(2) * 0.01
    specific = np.full(3, 0.001)
    covariance = loadings @ factor_covariance @ loadings.T + np.diag(specific)
    risk_model = validate_covariance(covariance, assets=("A", "B", "C"), as_of=as_of)

    with pytest.raises(RiskModelError, match="rank deficient"):
        FactorRiskModel(
            risk_model=risk_model,
            factors=("f1", "f2"),
            loadings=loadings,
            factor_covariance=factor_covariance,
            specific_variances=specific,
            exposure_vintage=as_of - pd.Timedelta(days=1),
            exposure_available_at=as_of,
        )


def test_public_input_conversion_failures_are_structured() -> None:
    returns, loadings, as_of = _factor_market()
    with pytest.raises(RiskModelError, match="returns must be a pandas DataFrame"):
        estimate_factor_risk_model(
            cast(Any, []),
            loadings,
            as_of=as_of,
            exposure_vintage=as_of - pd.Timedelta(days=1),
            exposure_available_at=as_of,
        )

    malformed = loadings.astype(object)
    malformed.iloc[0, 0] = "not-a-number"
    with pytest.raises(RiskModelError, match="cannot be converted to float64"):
        _estimate(returns, malformed, as_of=as_of)

    with pytest.raises(RiskModelError, match="finite real number"):
        estimate_factor_risk_model(
            returns,
            loadings,
            as_of=as_of,
            exposure_vintage=as_of - pd.Timedelta(days=1),
            exposure_available_at=as_of,
            factor_shrinkage=cast(Any, "invalid"),
        )
    with pytest.raises(RiskModelError, match="finite real number"):
        shrinkage_covariance(returns, as_of=as_of, intensity=cast(Any, "invalid"))

    malformed_index = returns.copy()
    malformed_index.index = [f"not-a-timestamp-{position}" for position in range(len(returns))]
    with pytest.raises(RiskModelError, match="invalid timestamp"):
        shrinkage_covariance(malformed_index, window=100)
