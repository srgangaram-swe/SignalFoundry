"""Numerical, adversarial, and replay contracts for SF-S2-MR7."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from alphaforge.config import ConfigValidationError, load_config
from alphaforge.evaluation import (
    BlockBootstrapConfig,
    MetricContract,
    MetricError,
    MetricEstimate,
    MetricSuiteConfig,
    evaluate_metric_suite,
    metric_contracts,
)


def _config(
    *,
    minimum_prediction_samples: int = 20,
    minimum_trading_periods: int = 20,
    block_length: int = 5,
) -> MetricSuiteConfig:
    return MetricSuiteConfig(
        bootstrap=BlockBootstrapConfig(
            n_resamples=200,
            block_length=block_length,
            confidence_level=0.90,
            seed=73,
        ),
        minimum_prediction_samples=minimum_prediction_samples,
        minimum_trading_periods=minimum_trading_periods,
        reliability_bins=5,
        benchmark_name="synthetic benchmark",
    )


def _predictions(n_dates: int = 30, n_symbols: int = 6) -> pd.DataFrame:
    rng = np.random.default_rng(12)
    dates = np.repeat(pd.bdate_range("2022-01-03", periods=n_dates), n_symbols)
    latent = rng.normal(0.0, 0.02, size=n_dates * n_symbols)
    prediction = latent + rng.normal(0.0, 0.005, size=len(latent))
    probability = 1.0 / (1.0 + np.exp(-prediction * 80.0))
    outcome = (latent > 0.0).astype(int)
    return pd.DataFrame(
        {
            "date": dates,
            "symbol": np.tile([f"S{index}" for index in range(n_symbols)], n_dates),
            "target": latent,
            "prediction": prediction,
            "probability": probability,
            "outcome": outcome,
        }
    )


def _trading(n_periods: int = 80) -> pd.DataFrame:
    index = np.arange(n_periods)
    returns = 0.0004 + 0.006 * np.sin(index / 5.0)
    turnover = np.where(index % 4 == 0, 0.2, 0.04)
    traded_notional = 1_000_000.0 * turnover
    transaction_cost = traded_notional * 0.0007
    return pd.DataFrame(
        {
            "date": pd.bdate_range("2022-06-01", periods=n_periods),
            "return": returns,
            "benchmark_return": 0.0002 + 0.004 * np.sin(index / 6.0),
            "turnover": turnover,
            "gross_exposure": 0.8 + 0.1 * np.sin(index / 9.0),
            "net_exposure": 0.05 * np.cos(index / 8.0),
            "transaction_cost": transaction_cost,
            "traded_notional": traded_notional,
            "capacity_fill_ratio": np.full(n_periods, 0.95),
            "capacity_constrained_fraction": np.full(n_periods, 0.05),
        }
    )


def test_contract_registry_declares_every_interpretation_dimension() -> None:
    contracts = metric_contracts(_config())

    assert len(contracts) == len({contract.name for contract in contracts})
    assert {
        "mse",
        "rank_ic",
        "brier_score",
        "annualized_return",
        "max_drawdown",
        "average_turnover",
        "average_gross_exposure",
        "capacity_fill_ratio",
        "cost_bps_per_traded_notional",
    }.issubset({contract.name for contract in contracts})
    for contract in contracts:
        assert contract.unit
        assert contract.annualization
        assert contract.benchmark
        assert contract.missingness
        assert contract.minimum_samples >= 2
        assert contract.aggregation
        assert contract.invalid_state


def test_public_metric_records_reject_contradictory_direct_construction() -> None:
    contract = MetricContract(
        name="test",
        unit="ratio",
        annualization="none",
        benchmark="none",
        missingness="reject",
        minimum_samples=2,
        aggregation="mean",
        invalid_state="undefined",
    )
    with pytest.raises(MetricError, match="requires time-series uncertainty"):
        MetricEstimate(
            contract=contract,
            value=1.0,
            status="ok",
            n_observations=2,
            uncertainty=None,
        )
    with pytest.raises(MetricError, match="cannot carry"):
        MetricEstimate(
            contract=contract,
            value=0.0,
            status="undefined",
            n_observations=2,
            uncertainty=None,
            note="not defined",
        )
    with pytest.raises(MetricError, match="block_length"):
        _config(
            minimum_prediction_samples=4,
            minimum_trading_periods=4,
            block_length=5,
        )


def test_unified_metrics_match_independent_numerical_references() -> None:
    prediction_panel = _predictions()
    trading_frame = _trading()

    report = evaluate_metric_suite(prediction_panel, trading_frame, _config())
    metrics = report.by_name()

    errors = prediction_panel["prediction"] - prediction_panel["target"]
    returns = trading_frame["return"].to_numpy()
    wealth = np.concatenate(([1.0], np.cumprod(1.0 + returns)))
    expected_drawdown = np.min(wealth / np.maximum.accumulate(wealth) - 1.0)
    elapsed_days = (
        trading_frame["date"].iloc[-1] - trading_frame["date"].iloc[0]
    ).total_seconds() / 86_400.0
    expected_frequency = (len(trading_frame) - 1) * 365.2425 / elapsed_days

    assert metrics["mse"].value == pytest.approx(float(np.mean(errors**2)))
    assert metrics["mae"].value == pytest.approx(float(np.mean(np.abs(errors))))
    assert metrics["total_return"].value == pytest.approx(float(np.prod(1.0 + returns) - 1.0))
    total_return = metrics["total_return"].value
    assert total_return is not None
    expected_annual_return = (1.0 + total_return) ** (365.2425 / elapsed_days) - 1.0
    assert metrics["annualized_return"].value == pytest.approx(expected_annual_return)
    assert metrics["max_drawdown"].value == pytest.approx(float(expected_drawdown))
    assert metrics["cost_bps_per_traded_notional"].value == pytest.approx(7.0)
    assert metrics["capacity_fill_ratio"].value == pytest.approx(0.95)
    assert report.effective_periods_per_year == pytest.approx(expected_frequency)
    assert metrics["rank_ic"].status == "ok"
    assert metrics["brier_score"].status == "ok"
    assert report.assumptions


def test_block_distributions_are_deterministic_and_publish_policy_and_variance() -> None:
    first = evaluate_metric_suite(_predictions(), _trading(), _config())
    second = evaluate_metric_suite(_predictions(), _trading(), _config())

    first_distribution = first.by_name()["annualized_return"].uncertainty
    second_distribution = second.by_name()["annualized_return"].uncertainty

    assert first_distribution == second_distribution
    assert first_distribution is not None
    assert first_distribution.n_resamples == 200
    assert first_distribution.block_length == 5
    mse_distribution = first.by_name()["mse"].uncertainty
    assert mse_distribution is not None
    assert mse_distribution.n_observations == 30
    assert first_distribution.seed == 73
    assert first_distribution.variance >= 0.0
    assert first.by_name()["annualized_return"].uncertainty_variance == pytest.approx(
        first_distribution.variance
    )
    assert first_distribution.assumptions


def test_no_trades_and_constant_returns_have_explicit_undefined_ratios() -> None:
    frame = _trading().assign(
        turnover=0.0,
        transaction_cost=0.0,
        traded_notional=0.0,
        **{"return": 0.001},
    )

    metrics = evaluate_metric_suite(_predictions(), frame, _config()).by_name()

    assert metrics["average_turnover"].value == 0.0
    assert metrics["total_transaction_cost"].value == 0.0
    assert metrics["sharpe"].status == "undefined"
    assert metrics["sharpe"].value is None
    assert metrics["cost_bps_per_traded_notional"].status == "undefined"
    assert "zero" in str(metrics["cost_bps_per_traded_notional"].note)


def test_bankrupt_path_is_bounded_and_cannot_recover_geometrically() -> None:
    frame = _trading().copy()
    frame.loc[20, "return"] = -1.0
    frame.loc[21:, "return"] = 0.01

    metrics = evaluate_metric_suite(_predictions(), frame, _config()).by_name()

    assert metrics["total_return"].value == pytest.approx(-1.0)
    assert metrics["annualized_return"].value == pytest.approx(-1.0)
    assert metrics["max_drawdown"].value == pytest.approx(-1.0)

    with pytest.raises(MetricError, match="greater than or equal"):
        evaluate_metric_suite(_predictions(), frame.assign(**{"return": -1.01}), _config())


def test_irregular_calendar_uses_elapsed_time_not_a_hard_coded_frequency() -> None:
    frame = _trading(25).iloc[[0, 1, 3, 4, 8, 9, 12, 14, 16, 18, 20, 22, 24]].reset_index(drop=True)
    config = _config(minimum_trading_periods=10, block_length=3)

    report = evaluate_metric_suite(_predictions(), frame, config)

    elapsed = (frame["date"].iloc[-1] - frame["date"].iloc[0]).days
    assert report.effective_periods_per_year == pytest.approx((len(frame) - 1) * 365.2425 / elapsed)


def test_tied_cross_sections_report_ic_as_undefined_without_inventing_signal() -> None:
    prediction_panel = _predictions().assign(prediction=0.0)

    metrics = evaluate_metric_suite(prediction_panel, _trading(), _config()).by_name()

    assert metrics["pearson_ic"].status == "undefined"
    assert metrics["rank_ic"].status == "undefined"
    assert metrics["pearson_ic"].value is None
    assert metrics["rank_ic"].uncertainty is None


@pytest.mark.parametrize(
    ("mutator", "message"),
    [
        (lambda frame: frame.assign(**{"return": np.nan}), "finite"),
        (lambda frame: frame.assign(turnover=-0.1), "non-negative"),
        (lambda frame: frame.assign(capacity_fill_ratio=1.1), r"\[0, 1\]"),
        (lambda frame: frame.assign(transaction_cost=-1.0), "non-negative"),
    ],
)
def test_trading_metrics_reject_missing_or_invalid_states(mutator, message: str) -> None:
    with pytest.raises(MetricError, match=message):
        evaluate_metric_suite(_predictions(), mutator(_trading()), _config())


def test_prediction_metrics_reject_nan_bad_probability_and_partial_calibration() -> None:
    with pytest.raises(MetricError, match="finite"):
        evaluate_metric_suite(
            _predictions().assign(target=np.nan),
            _trading(),
            _config(),
        )
    with pytest.raises((MetricError, ValueError), match=r"\[0, 1\]"):
        evaluate_metric_suite(
            _predictions().assign(probability=1.1),
            _trading(),
            _config(),
        )
    with pytest.raises(MetricError, match="together"):
        evaluate_metric_suite(
            _predictions().drop(columns="outcome"),
            _trading(),
            _config(),
        )


def test_short_samples_ordering_and_duplicate_trading_dates_fail_closed() -> None:
    with pytest.raises(MetricError, match="at least"):
        evaluate_metric_suite(_predictions().iloc[:10], _trading(), _config())
    with pytest.raises(MetricError, match="at least"):
        evaluate_metric_suite(_predictions(), _trading().iloc[:10], _config())
    with pytest.raises(MetricError, match="monotonically"):
        evaluate_metric_suite(_predictions().iloc[::-1], _trading(), _config())
    duplicated = _trading().copy()
    duplicated.loc[1, "date"] = duplicated.loc[0, "date"]
    with pytest.raises(MetricError, match="unique"):
        evaluate_metric_suite(_predictions(), duplicated, _config())


def test_metric_scaling_and_cost_ratio_metamorphic_properties() -> None:
    predictions = _predictions()
    baseline = evaluate_metric_suite(predictions, _trading(), _config()).by_name()
    shifted = predictions.assign(
        target=predictions["target"] + 0.25,
        prediction=predictions["prediction"] + 0.25,
    )
    shifted_metrics = evaluate_metric_suite(shifted, _trading(), _config()).by_name()
    scaled_trading = _trading().assign(
        transaction_cost=_trading()["transaction_cost"] * 3.0,
        traded_notional=_trading()["traded_notional"] * 3.0,
    )
    scaled_metrics = evaluate_metric_suite(predictions, scaled_trading, _config()).by_name()

    assert shifted_metrics["mse"].value == pytest.approx(baseline["mse"].value)
    assert scaled_metrics["cost_bps_per_traded_notional"].value == pytest.approx(
        baseline["cost_bps_per_traded_notional"].value
    )


def test_sparse_trade_cost_ratio_uses_active_trades_and_rejects_unreconciled_cost() -> None:
    frame = _trading()
    inactive = np.arange(len(frame)) % 2 == 0
    frame.loc[inactive, ["transaction_cost", "traded_notional"]] = 0.0

    metric = evaluate_metric_suite(_predictions(), frame, _config()).by_name()[
        "cost_bps_per_traded_notional"
    ]

    assert metric.value == pytest.approx(7.0)
    corrupt = frame.copy()
    corrupt.loc[0, "transaction_cost"] = 1.0
    with pytest.raises(MetricError, match="must be zero"):
        evaluate_metric_suite(_predictions(), corrupt, _config())


def test_metrics_configuration_is_strict_and_cross_validated(tmp_path) -> None:
    config = load_config("configs/metrics.yaml", "metrics")
    assert config["bootstrap"]["block_length"] == 20
    assert config["minimum_prediction_samples"] == 20

    import yaml

    malformed = {**config, "bootstrap": {**config["bootstrap"], "block_length": 21}}
    path = tmp_path / "metrics.yaml"
    path.write_text(yaml.safe_dump(malformed, sort_keys=False), encoding="utf-8")
    with pytest.raises(ConfigValidationError, match="block_length"):
        load_config(path, "metrics")
