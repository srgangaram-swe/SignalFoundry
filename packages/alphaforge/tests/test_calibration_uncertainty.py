"""Adversarial and numerical contracts for SF-S2-MR6."""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from alphaforge.config import ConfigValidationError, load_config
from alphaforge.evaluation import (
    BlockBootstrapConfig,
    BlockConformalInterval,
    CalibrationError,
    CalibrationEvaluationData,
    OOFProbabilityData,
    OOFRegressionData,
    ProbabilityCalibrator,
    QuantileRegressionConfig,
    QuantileRegressionInterval,
    UncertaintyError,
    block_bootstrap_interval,
    brier_score,
    compare_calibration,
    expected_calibration_error,
    paired_block_bootstrap_difference,
    reliability_diagram,
)


def _probability_data() -> tuple[OOFProbabilityData, CalibrationEvaluationData]:
    rng = np.random.default_rng(19)
    score = np.linspace(-2.5, 2.5, 400)
    true_probability = 1.0 / (1.0 + np.exp(-score))
    raw_probability = 1.0 / (1.0 + np.exp(-2.2 * score))
    outcome = rng.binomial(1, true_probability)
    dates = pd.date_range("2020-01-01", periods=400, freq="D")
    fit = OOFProbabilityData.from_arrays(
        raw_probability[:300],
        outcome[:300],
        dates[:300],
        np.repeat([0, 1, 2], 100),
    )
    evaluation = CalibrationEvaluationData.from_arrays(
        raw_probability[300:],
        outcome[300:],
        dates[300:],
    )
    return fit, evaluation


def _regression_oof(n: int = 120) -> OOFRegressionData:
    rng = np.random.default_rng(29)
    prediction = np.linspace(-0.02, 0.02, n)
    residual = rng.normal(0.0, 0.01, size=n)
    return OOFRegressionData.from_arrays(
        prediction,
        prediction + residual,
        pd.date_range("2021-01-01", periods=n, freq="D"),
        np.repeat(np.arange(3), n // 3),
    )


def test_brier_and_reliability_match_independent_reference() -> None:
    probabilities = np.array([0.0, 0.1, 0.8, 1.0])
    outcomes = np.array([0, 1, 1, 1])
    expected = float(np.mean((probabilities - outcomes) ** 2))

    result = reliability_diagram(probabilities, outcomes, n_bins=5)

    assert brier_score(probabilities, outcomes) == pytest.approx(expected)
    assert result.brier_score == pytest.approx(expected)
    assert result.n_observations == 4
    assert result.bins["count"].sum() == 4
    assert (result.bins["count"] == 0).any()
    assert result.bins.iloc[-1]["count"] == 2
    assert expected_calibration_error(probabilities, outcomes, n_bins=5) == pytest.approx(
        result.expected_calibration_error
    )


@pytest.mark.parametrize(
    ("probabilities", "outcomes", "message"),
    [
        ([0.1, np.nan], [0, 1], "finite"),
        ([-0.1, 0.5], [0, 1], r"\[0, 1\]"),
        ([0.1, 1.1], [0, 1], r"\[0, 1\]"),
        ([0.1, 0.9], [0, 2], "binary"),
        ([0.1], [0, 1], "equal length"),
    ],
)
def test_probability_metrics_reject_malformed_inputs(
    probabilities: object, outcomes: object, message: str
) -> None:
    with pytest.raises(CalibrationError, match=message):
        brier_score(probabilities, outcomes)


def test_reliability_rejects_unbounded_or_noninteger_bins() -> None:
    with pytest.raises(CalibrationError, match="n_bins"):
        reliability_diagram([0.5], [1], n_bins=1)
    with pytest.raises(TypeError, match="integer"):
        reliability_diagram([0.5], [1], n_bins=True)


def test_oof_probability_contract_rejects_false_provenance_shapes_and_time() -> None:
    probability = np.linspace(0.1, 0.9, 20)
    outcome = np.tile([0, 1], 10)
    dates = pd.date_range("2020-01-01", periods=20)

    with pytest.raises(CalibrationError, match="two folds"):
        OOFProbabilityData.from_arrays(probability, outcome, dates, np.zeros(20, dtype=int))
    with pytest.raises(CalibrationError, match="monotonically"):
        OOFProbabilityData.from_arrays(
            probability,
            outcome,
            dates[::-1],
            np.repeat([0, 1], 10),
        )
    with pytest.raises(CalibrationError, match="fold_ids"):
        OOFProbabilityData.from_arrays(probability, outcome, dates, np.arange(19))
    with pytest.raises(CalibrationError, match="source"):
        OOFProbabilityData(
            tuple(probability),
            tuple(outcome),
            tuple(timestamp.isoformat() for timestamp in dates),
            tuple(np.repeat([0, 1], 10)),
            source="training_oof_typo",  # type: ignore[arg-type]
        )


@pytest.mark.parametrize("method", ["platt", "isotonic"])
def test_calibrators_are_deterministic_and_record_oof_provenance(method: str) -> None:
    fit, _ = _probability_data()
    first = ProbabilityCalibrator(method).fit(fit)  # type: ignore[arg-type]
    second = ProbabilityCalibrator(method).fit(fit)  # type: ignore[arg-type]
    grid = np.linspace(0.0, 1.0, 101)

    assert np.array_equal(first.predict(grid), second.predict(grid))
    assert first.metadata_ is not None
    assert first.metadata_.fit_start == fit.fit_start
    assert first.metadata_.fit_end == fit.fit_end
    assert first.metadata_.n_samples == 300
    assert first.metadata_.n_folds == 3
    assert first.metadata_.limitations


def test_calibrator_rejects_constant_scores_single_class_and_non_oof_data() -> None:
    dates = pd.date_range("2020-01-01", periods=20)
    folds = np.repeat([0, 1], 10)
    constant = OOFProbabilityData.from_arrays(np.full(20, 0.4), np.tile([0, 1], 10), dates, folds)
    one_class = OOFProbabilityData.from_arrays(
        np.linspace(0.1, 0.9, 20), np.zeros(20), dates, folds
    )

    with pytest.raises(CalibrationError, match="constant"):
        ProbabilityCalibrator().fit(constant)
    with pytest.raises(CalibrationError, match="both"):
        ProbabilityCalibrator().fit(one_class)
    with pytest.raises(TypeError, match="OOFProbabilityData"):
        ProbabilityCalibrator().fit(object())  # type: ignore[arg-type]


@pytest.mark.parametrize("method", ["platt", "isotonic"])
def test_calibrator_json_round_trip_is_non_executable(method: str, tmp_path: Path) -> None:
    fit, _ = _probability_data()
    calibrator = ProbabilityCalibrator(method).fit(fit)  # type: ignore[arg-type]
    path = calibrator.save(tmp_path / f"{method}.json")

    loaded = ProbabilityCalibrator.load(path)

    assert np.array_equal(
        loaded.predict([0.0, 0.25, 0.5, 0.75, 1.0]),
        calibrator.predict([0.0, 0.25, 0.5, 0.75, 1.0]),
    )
    assert loaded.metadata_ == calibrator.metadata_
    assert json.loads(path.read_text())["format"] == "alphaforge-calibration/json"


def test_calibration_comparison_is_post_fit_and_dependence_aware() -> None:
    fit, evaluation = _probability_data()
    calibrator = ProbabilityCalibrator("platt").fit(fit)
    bootstrap = BlockBootstrapConfig(n_resamples=300, block_length=10, seed=7)

    result = compare_calibration(calibrator, evaluation, bootstrap, n_bins=10)

    assert result.method == "platt"
    assert result.n_observations == 100
    assert result.calibrated_brier < result.raw_brier
    assert result.brier_improvement_interval.estimate == pytest.approx(
        result.raw_brier - result.calibrated_brier
    )
    assert result.brier_improvement_interval.block_length == 10
    assert result.limitations

    overlapping = CalibrationEvaluationData.from_arrays(
        evaluation.probabilities,
        evaluation.outcomes,
        pd.date_range("2020-01-01", periods=100),
    )
    with pytest.raises(CalibrationError, match="strictly after"):
        compare_calibration(calibrator, overlapping, bootstrap)


def test_moving_block_bootstrap_is_reproducible_and_matches_mean() -> None:
    values = np.sin(np.arange(200) / 8.0)
    config = BlockBootstrapConfig(n_resamples=300, block_length=12, seed=91)

    first = block_bootstrap_interval(values, config)
    second = block_bootstrap_interval(values, config)

    assert first == second
    assert first.estimate == pytest.approx(np.mean(values))
    assert first.lower <= first.estimate <= first.upper
    assert first.standard_error > 0
    assert first.assumptions


def test_paired_bootstrap_preserves_pairing_and_rejects_short_blocks() -> None:
    first = np.arange(40, dtype=float) / 100
    second = first - 0.02
    config = BlockBootstrapConfig(n_resamples=200, block_length=5, seed=1)

    result = paired_block_bootstrap_difference(first, second, config)

    assert result.estimate == pytest.approx(0.02)
    with pytest.raises(UncertaintyError, match="equal length"):
        paired_block_bootstrap_difference(first, second[:-1], config)
    with pytest.raises(UncertaintyError, match="must not exceed"):
        block_bootstrap_interval(first[:4], BlockBootstrapConfig(block_length=5))


def test_block_conformal_requires_sufficient_oof_blocks_and_future_dates(
    tmp_path: Path,
) -> None:
    data = _regression_oof()
    conformal = BlockConformalInterval(alpha=0.1, block_length=10, min_blocks=5).fit(data)
    future_dates = pd.date_range("2021-05-01", periods=5)

    interval = conformal.predict_interval(np.zeros(5), future_dates)

    assert conformal.metadata_ is not None
    assert conformal.metadata_.n_blocks == 12
    assert (interval["upper"] >= interval["prediction"]).all()
    assert (interval["lower"] <= interval["prediction"]).all()
    assert conformal.metadata_.limitations
    loaded = BlockConformalInterval.load(conformal.save(tmp_path / "conformal.json"))
    pd.testing.assert_frame_equal(
        loaded.predict_interval(np.zeros(5), future_dates),
        interval,
    )

    with pytest.raises(UncertaintyError, match="strictly after"):
        conformal.predict_interval(np.zeros(5), pd.date_range("2021-01-01", periods=5))
    with pytest.raises(UncertaintyError, match="complete blocks"):
        BlockConformalInterval(block_length=20, min_blocks=7).fit(data)


def test_oof_regression_contract_rejects_nonfinite_and_single_fold() -> None:
    dates = pd.date_range("2021-01-01", periods=10)
    with pytest.raises(UncertaintyError, match="finite"):
        OOFRegressionData.from_arrays(
            [0.0] * 9 + [np.nan],
            np.zeros(10),
            dates,
            np.repeat([0, 1], 5),
        )
    with pytest.raises(UncertaintyError, match="two folds"):
        OOFRegressionData.from_arrays(np.zeros(10), np.zeros(10), dates, np.zeros(10, dtype=int))
    with pytest.raises(UncertaintyError, match="source"):
        OOFRegressionData(
            tuple(np.zeros(10)),
            tuple(np.zeros(10)),
            tuple(timestamp.isoformat() for timestamp in dates),
            tuple(np.repeat([0, 1], 5)),
            source="test_predictions",  # type: ignore[arg-type]
        )


def test_quantile_regression_interval_is_bounded_deterministic_and_persistent(
    tmp_path: Path,
) -> None:
    rng = np.random.default_rng(123)
    features = pd.DataFrame({"constant": np.zeros(240)})
    outcomes = rng.standard_t(df=5, size=240)
    dates = pd.date_range("2019-01-01", periods=240)
    config = QuantileRegressionConfig(
        lower_quantile=0.1,
        upper_quantile=0.9,
        alpha=0.001,
        max_iter=5_000,
    )
    first = QuantileRegressionInterval(config).fit(features, outcomes, dates)
    second = QuantileRegressionInterval(config).fit(features, outcomes, dates)
    query = pd.DataFrame({"constant": np.zeros(8)})

    first_prediction = first.predict_interval(query)

    pd.testing.assert_frame_equal(first_prediction, second.predict_interval(query))
    assert (first_prediction["lower"] <= first_prediction["median"]).all()
    assert (first_prediction["median"] <= first_prediction["upper"]).all()
    assert first.metadata_ is not None
    assert first.metadata_.fit_start == dates[0].isoformat()
    assert first.metadata_.limitations
    loaded = QuantileRegressionInterval.load(first.save(tmp_path / "quantile.json"))
    pd.testing.assert_frame_equal(loaded.predict_interval(query), first_prediction)


def test_quantile_regression_rejects_schema_nonfinite_and_bad_bounds() -> None:
    features = pd.DataFrame({"x": np.arange(20, dtype=float)})
    outcomes = np.arange(20, dtype=float)
    dates = pd.date_range("2020-01-01", periods=20)
    model = QuantileRegressionInterval().fit(features, outcomes, dates)

    with pytest.raises(UncertaintyError, match="schema"):
        model.predict_interval(pd.DataFrame({"other": [0.0]}))
    with pytest.raises(UncertaintyError, match="finite"):
        model.predict_interval(pd.DataFrame({"x": [np.nan]}))
    with pytest.raises(UncertaintyError, match="lower"):
        QuantileRegressionConfig(lower_quantile=0.6, upper_quantile=0.9)


def test_calibration_configuration_is_strict_and_cross_validated(tmp_path: Path) -> None:
    config = load_config("configs/calibration.yaml", "calibration")
    assert config["probability"]["method"] == "platt"
    assert config["bootstrap"]["block_length"] == config["conformal"]["block_length"]

    malformed = {
        **config,
        "bootstrap": {**config["bootstrap"], "block_length": 15},
    }
    path = tmp_path / "calibration.yaml"
    import yaml

    path.write_text(yaml.safe_dump(malformed, sort_keys=False), encoding="utf-8")
    with pytest.raises(ConfigValidationError, match="block lengths"):
        load_config(path, "calibration")


def test_artifact_loaders_fail_closed_on_unknown_or_malformed_state(tmp_path: Path) -> None:
    unknown = tmp_path / "unknown.json"
    unknown.write_text('{"format":"other","version":1}', encoding="utf-8")
    with pytest.raises(CalibrationError, match="unsupported"):
        ProbabilityCalibrator.load(unknown)
    with pytest.raises(UncertaintyError, match="unsupported"):
        BlockConformalInterval.load(unknown)

    fit, _ = _probability_data()
    path = ProbabilityCalibrator("platt").fit(fit).save(tmp_path / "calibrator.json")
    payload = json.loads(path.read_text())
    payload["state"] = {}
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(CalibrationError, match="state"):
        ProbabilityCalibrator.load(path)

    fit, _ = _probability_data()
    isotonic_path = ProbabilityCalibrator("isotonic").fit(fit).save(tmp_path / "isotonic.json")
    isotonic_payload = json.loads(isotonic_path.read_text())
    isotonic_payload["state"]["x_thresholds"] = list(
        reversed(isotonic_payload["state"]["x_thresholds"])
    )
    isotonic_path.write_text(json.dumps(isotonic_payload), encoding="utf-8")
    with pytest.raises(CalibrationError, match="thresholds"):
        ProbabilityCalibrator.load(isotonic_path)

    malformed_date_path = (
        ProbabilityCalibrator("platt").fit(fit).save(tmp_path / "malformed-date.json")
    )
    malformed_date_payload = json.loads(malformed_date_path.read_text())
    malformed_date_payload["metadata"]["fit_start"] = "not-a-date"
    malformed_date_path.write_text(json.dumps(malformed_date_payload), encoding="utf-8")
    with pytest.raises(CalibrationError, match="metadata"):
        ProbabilityCalibrator.load(malformed_date_path)

    conformal_path = (
        BlockConformalInterval(block_length=10)
        .fit(_regression_oof())
        .save(tmp_path / "conformal-corrupt.json")
    )
    conformal_payload = json.loads(conformal_path.read_text())
    conformal_payload["metadata"]["fit_end"] = "not-a-date"
    conformal_path.write_text(json.dumps(conformal_payload), encoding="utf-8")
    with pytest.raises(UncertaintyError, match="metadata"):
        BlockConformalInterval.load(conformal_path)

    features = pd.DataFrame({"x": np.arange(20, dtype=float)})
    quantile_path = (
        QuantileRegressionInterval()
        .fit(
            features,
            np.arange(20, dtype=float),
            pd.date_range("2020-01-01", periods=20),
        )
        .save(tmp_path / "quantile-corrupt.json")
    )
    quantile_payload = json.loads(quantile_path.read_text())
    quantile_payload["metadata"]["fit_start"] = "not-a-date"
    quantile_path.write_text(json.dumps(quantile_payload), encoding="utf-8")
    with pytest.raises(UncertaintyError, match="state"):
        QuantileRegressionInterval.load(quantile_path)


def test_metadata_records_are_json_compatible() -> None:
    fit, _ = _probability_data()
    calibrator = ProbabilityCalibrator("platt").fit(fit)
    assert calibrator.metadata_ is not None
    json.dumps(asdict(calibrator.metadata_), sort_keys=True)
