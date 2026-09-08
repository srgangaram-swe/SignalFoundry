"""Probability calibration with explicit training-OOF provenance.

Calibration is learned only from :class:`OOFProbabilityData`.  The type fixes
the provenance to ``training_oof`` and requires multiple temporal folds.  A
separate evaluation contract requires every reported observation to occur
strictly after the calibration period, preventing accidental calibration on a
test or final-holdout outcome.
"""

from __future__ import annotations

import json
import os
import tempfile
from contextlib import suppress
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal, cast

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression

from alphaforge.evaluation.uncertainty import (
    BlockBootstrapConfig,
    BootstrapInterval,
    paired_block_bootstrap_difference,
)

_CALIBRATION_FORMAT = "alphaforge-calibration/json"
_CALIBRATION_VERSION = 1
_MAX_ARTIFACT_BYTES = 4 * 1024 * 1024
_MAX_OBSERVATIONS = 10_000_000
_PROBABILITY_EPSILON = 1e-12


class CalibrationError(ValueError):
    """Raised when calibration inputs, provenance, or state are invalid."""


def _finite_vector(values: object, name: str, minimum: int = 1) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    if array.ndim != 1:
        raise CalibrationError(f"{name} must be one-dimensional")
    if len(array) < minimum:
        raise CalibrationError(f"{name} must contain at least {minimum} observations")
    if len(array) > _MAX_OBSERVATIONS:
        raise CalibrationError(f"{name} exceeds the {_MAX_OBSERVATIONS} observation limit")
    if not np.isfinite(array).all():
        raise CalibrationError(f"{name} must contain only finite values")
    return array


def _probabilities(values: object, name: str = "probabilities", minimum: int = 1) -> np.ndarray:
    array = _finite_vector(values, name, minimum)
    if ((array < 0.0) | (array > 1.0)).any():
        raise CalibrationError(f"{name} must lie in [0, 1]")
    return array


def _binary_outcomes(values: object, expected: int, minimum: int = 1) -> np.ndarray:
    array = _finite_vector(values, "outcomes", minimum)
    if len(array) != expected:
        raise CalibrationError("probabilities and outcomes must have equal length")
    if not np.isin(array, (0.0, 1.0)).all():
        raise CalibrationError("outcomes must contain only binary values 0 and 1")
    return array.astype(int)


def _ordered_dates(values: object, expected: int, name: str = "dates") -> pd.DatetimeIndex:
    try:
        dates = pd.DatetimeIndex(pd.to_datetime(values, errors="raise"))
    except (TypeError, ValueError) as exc:
        raise CalibrationError(f"{name} must contain valid timestamps") from exc
    if len(dates) != expected:
        raise CalibrationError(f"{name} length must match the observations")
    if dates.hasnans:
        raise CalibrationError(f"{name} must not contain missing timestamps")
    if not dates.is_monotonic_increasing:
        raise CalibrationError(f"{name} must be monotonically non-decreasing")
    return dates


@dataclass(frozen=True)
class OOFProbabilityData:
    """Training-derived out-of-fold probabilities and binary outcomes."""

    probabilities: tuple[float, ...]
    outcomes: tuple[int, ...]
    dates: tuple[str, ...]
    fold_ids: tuple[int, ...]
    source: Literal["training_oof"] = "training_oof"

    def __post_init__(self) -> None:
        if self.source != "training_oof":
            raise CalibrationError("OOF probability source must be 'training_oof'")
        probability = _probabilities(self.probabilities, minimum=20)
        _binary_outcomes(self.outcomes, len(probability), minimum=20)
        _ordered_dates(self.dates, len(probability))
        folds = np.asarray(self.fold_ids)
        if folds.ndim != 1 or len(folds) != len(probability):
            raise CalibrationError("fold_ids must be one-dimensional and match observations")
        if not np.issubdtype(folds.dtype, np.integer) or (folds < 0).any():
            raise CalibrationError("fold_ids must contain non-negative integers")
        if len(np.unique(folds)) < 2:
            raise CalibrationError("training OOF data must contain at least two folds")

    @classmethod
    def from_arrays(
        cls,
        probabilities: object,
        outcomes: object,
        dates: object,
        fold_ids: object,
    ) -> OOFProbabilityData:
        probability = _probabilities(probabilities, minimum=20)
        outcome = _binary_outcomes(outcomes, len(probability), minimum=20)
        parsed_dates = _ordered_dates(dates, len(probability))
        folds = np.asarray(fold_ids)
        if folds.ndim != 1 or len(folds) != len(probability):
            raise CalibrationError("fold_ids must be one-dimensional and match observations")
        if not np.issubdtype(folds.dtype, np.integer) or (folds < 0).any():
            raise CalibrationError("fold_ids must contain non-negative integers")
        if len(np.unique(folds)) < 2:
            raise CalibrationError("training OOF data must contain at least two folds")
        return cls(
            probabilities=tuple(float(value) for value in probability),
            outcomes=tuple(int(value) for value in outcome),
            dates=tuple(timestamp.isoformat() for timestamp in parsed_dates),
            fold_ids=tuple(int(value) for value in folds),
        )

    @property
    def fit_start(self) -> str:
        return self.dates[0]

    @property
    def fit_end(self) -> str:
        return self.dates[-1]


@dataclass(frozen=True)
class CalibrationEvaluationData:
    """Post-fit observations used only for honest calibration comparison."""

    probabilities: tuple[float, ...]
    outcomes: tuple[int, ...]
    dates: tuple[str, ...]

    def __post_init__(self) -> None:
        probability = _probabilities(self.probabilities, minimum=10)
        _binary_outcomes(self.outcomes, len(probability), minimum=10)
        _ordered_dates(self.dates, len(probability))

    @classmethod
    def from_arrays(
        cls, probabilities: object, outcomes: object, dates: object
    ) -> CalibrationEvaluationData:
        probability = _probabilities(probabilities, minimum=10)
        outcome = _binary_outcomes(outcomes, len(probability), minimum=10)
        parsed_dates = _ordered_dates(dates, len(probability))
        return cls(
            probabilities=tuple(float(value) for value in probability),
            outcomes=tuple(int(value) for value in outcome),
            dates=tuple(timestamp.isoformat() for timestamp in parsed_dates),
        )


@dataclass(frozen=True)
class ReliabilityResult:
    """Fixed-bin reliability table and aggregate calibration metrics."""

    bins: pd.DataFrame
    brier_score: float
    expected_calibration_error: float
    n_observations: int
    n_bins: int


def brier_score(probabilities: object, outcomes: object) -> float:
    """Return the mean squared probability error for binary outcomes."""

    probability = _probabilities(probabilities)
    outcome = _binary_outcomes(outcomes, len(probability))
    return float(np.mean((probability - outcome) ** 2))


def reliability_diagram(
    probabilities: object, outcomes: object, *, n_bins: int = 10
) -> ReliabilityResult:
    """Return fixed-width reliability bins, including explicitly empty bins."""

    if isinstance(n_bins, bool) or not isinstance(n_bins, int):
        raise TypeError("n_bins must be an integer")
    if not 2 <= n_bins <= 100:
        raise CalibrationError("n_bins must be in [2, 100]")
    probability = _probabilities(probabilities)
    outcome = _binary_outcomes(outcomes, len(probability))
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    assignments = np.minimum(np.searchsorted(edges, probability, side="right") - 1, n_bins - 1)
    rows: list[dict[str, float | int]] = []
    weighted_error = 0.0
    for bin_id in range(n_bins):
        mask = assignments == bin_id
        count = int(mask.sum())
        mean_probability = float(np.mean(probability[mask])) if count else np.nan
        observed_rate = float(np.mean(outcome[mask])) if count else np.nan
        absolute_gap = abs(mean_probability - observed_rate) if count else np.nan
        if count:
            weighted_error += count / len(probability) * absolute_gap
        rows.append(
            {
                "bin_id": bin_id,
                "lower": float(edges[bin_id]),
                "upper": float(edges[bin_id + 1]),
                "count": count,
                "mean_probability": mean_probability,
                "observed_rate": observed_rate,
                "absolute_gap": absolute_gap,
            }
        )
    return ReliabilityResult(
        bins=pd.DataFrame(rows),
        brier_score=brier_score(probability, outcome),
        expected_calibration_error=float(weighted_error),
        n_observations=len(probability),
        n_bins=n_bins,
    )


def expected_calibration_error(
    probabilities: object, outcomes: object, *, n_bins: int = 10
) -> float:
    """Return count-weighted expected calibration error over fixed bins."""

    return reliability_diagram(probabilities, outcomes, n_bins=n_bins).expected_calibration_error


@dataclass(frozen=True)
class CalibrationMetadata:
    """Fit provenance and limitations for a probability calibrator."""

    method: Literal["platt", "isotonic"]
    fit_start: str
    fit_end: str
    n_samples: int
    n_folds: int
    positive_rate: float
    limitations: tuple[str, ...]


class ProbabilityCalibrator:
    """Platt or isotonic calibration fit exclusively on OOF training data."""

    def __init__(self, method: Literal["platt", "isotonic"] = "platt") -> None:
        if method not in {"platt", "isotonic"}:
            raise CalibrationError("method must be 'platt' or 'isotonic'")
        self.method = method
        self.metadata_: CalibrationMetadata | None = None
        self._state: dict[str, object] | None = None

    def fit(self, data: OOFProbabilityData) -> ProbabilityCalibrator:
        if not isinstance(data, OOFProbabilityData) or data.source != "training_oof":
            raise TypeError("data must be OOFProbabilityData with source='training_oof'")
        probability = np.asarray(data.probabilities, dtype=float)
        outcome = np.asarray(data.outcomes, dtype=int)
        if len(np.unique(outcome)) != 2:
            raise CalibrationError("calibration requires both binary outcome classes")
        if np.ptp(probability) <= np.finfo(float).eps:
            raise CalibrationError("constant scores cannot support probability calibration")
        if self.method == "platt":
            log_odds = np.log(
                np.clip(probability, _PROBABILITY_EPSILON, 1.0 - _PROBABILITY_EPSILON)
                / np.clip(1.0 - probability, _PROBABILITY_EPSILON, 1.0)
            )
            estimator = LogisticRegression(
                C=1.0,
                solver="lbfgs",
                max_iter=10_000,
                random_state=0,
            ).fit(log_odds.reshape(-1, 1), outcome)
            if int(estimator.n_iter_[0]) >= estimator.max_iter:
                raise CalibrationError("Platt calibration did not converge")
            self._state = {
                "coefficient": float(estimator.coef_[0, 0]),
                "intercept": float(estimator.intercept_[0]),
            }
            method_limitations = (
                "Platt scaling assumes a logistic relationship in raw-probability log-odds",
            )
        else:
            estimator = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip").fit(
                probability, outcome
            )
            self._state = {
                "x_thresholds": [float(value) for value in estimator.X_thresholds_],
                "y_thresholds": [float(value) for value in estimator.y_thresholds_],
            }
            method_limitations = (
                "isotonic calibration can overfit small or weakly supported score regions",
            )
        self.metadata_ = CalibrationMetadata(
            method=self.method,
            fit_start=data.fit_start,
            fit_end=data.fit_end,
            n_samples=len(probability),
            n_folds=len(set(data.fold_ids)),
            positive_rate=float(np.mean(outcome)),
            limitations=method_limitations
            + (
                "calibration validity can fail under prevalence or regime shift",
                "OOF provenance prevents in-sample fitting but does not prove future calibration",
            ),
        )
        return self

    def predict(self, probabilities: object) -> np.ndarray:
        if self.metadata_ is None or self._state is None:
            raise CalibrationError("probability calibrator is not fitted")
        probability = _probabilities(probabilities)
        if self.method == "platt":
            coefficient = float(cast(float, self._state["coefficient"]))
            intercept = float(cast(float, self._state["intercept"]))
            log_odds = np.log(
                np.clip(probability, _PROBABILITY_EPSILON, 1.0 - _PROBABILITY_EPSILON)
                / np.clip(1.0 - probability, _PROBABILITY_EPSILON, 1.0)
            )
            linear = np.clip(coefficient * log_odds + intercept, -709.0, 709.0)
            return 1.0 / (1.0 + np.exp(-linear))
        x_thresholds = np.asarray(cast(list[float], self._state["x_thresholds"]), dtype=float)
        y_thresholds = np.asarray(cast(list[float], self._state["y_thresholds"]), dtype=float)
        return np.interp(probability, x_thresholds, y_thresholds)

    def save(self, path: str | Path) -> Path:
        if self.metadata_ is None or self._state is None:
            raise CalibrationError("probability calibrator is not fitted")
        payload = {
            "format": _CALIBRATION_FORMAT,
            "version": _CALIBRATION_VERSION,
            "method": self.method,
            "metadata": asdict(self.metadata_),
            "state": self._state,
        }
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        descriptor, temporary_name = tempfile.mkstemp(
            dir=destination.parent, prefix=f".{destination.name}.", suffix=".tmp"
        )
        try:
            with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
                json.dump(payload, handle, sort_keys=True, separators=(",", ":"))
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary_name, destination)
        except BaseException:
            with suppress(FileNotFoundError):
                os.unlink(temporary_name)
            raise
        return destination

    @classmethod
    def load(cls, path: str | Path) -> ProbabilityCalibrator:
        source = Path(path)
        try:
            if source.stat().st_size > _MAX_ARTIFACT_BYTES:
                raise CalibrationError("calibration artifact exceeds the size limit")
            payload = json.loads(source.read_text(encoding="utf-8"))
        except CalibrationError:
            raise
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            raise CalibrationError("could not read calibration artifact") from exc
        if not isinstance(payload, dict):
            raise CalibrationError("calibration artifact root must be an object")
        if (
            payload.get("format") != _CALIBRATION_FORMAT
            or payload.get("version") != _CALIBRATION_VERSION
        ):
            raise CalibrationError("unsupported calibration artifact")
        method = payload.get("method")
        metadata = payload.get("metadata")
        state = payload.get("state")
        if (
            method not in {"platt", "isotonic"}
            or not isinstance(metadata, dict)
            or not isinstance(state, dict)
        ):
            raise CalibrationError("malformed calibration artifact")
        instance = cls(method)
        try:
            instance.metadata_ = CalibrationMetadata(
                **{**metadata, "limitations": tuple(metadata["limitations"])}
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise CalibrationError("malformed calibration metadata") from exc
        if instance.metadata_.method != instance.method:
            raise CalibrationError("calibration metadata method does not match the artifact")
        try:
            fit_start = pd.Timestamp(instance.metadata_.fit_start)
            fit_end = pd.Timestamp(instance.metadata_.fit_end)
            invalid_metadata = (
                fit_start is pd.NaT
                or fit_end is pd.NaT
                or instance.metadata_.n_samples < 20
                or instance.metadata_.n_folds < 2
                or not 0.0 <= instance.metadata_.positive_rate <= 1.0
                or fit_start > fit_end
            )
        except (TypeError, ValueError, OverflowError) as exc:
            raise CalibrationError("malformed calibration metadata") from exc
        if invalid_metadata:
            raise CalibrationError("calibration metadata violates fitted-state invariants")
        if method == "platt":
            if set(state) != {"coefficient", "intercept"}:
                raise CalibrationError("malformed Platt calibration state")
            try:
                values = np.asarray([state["coefficient"], state["intercept"]], dtype=float)
            except (TypeError, ValueError, OverflowError) as exc:
                raise CalibrationError("malformed Platt calibration state") from exc
            if not np.isfinite(values).all():
                raise CalibrationError("Platt calibration state must be finite")
        else:
            if set(state) != {"x_thresholds", "y_thresholds"}:
                raise CalibrationError("malformed isotonic calibration state")
            try:
                x_thresholds = np.asarray(state["x_thresholds"], dtype=float)
                y_thresholds = np.asarray(state["y_thresholds"], dtype=float)
            except (TypeError, ValueError, OverflowError) as exc:
                raise CalibrationError("malformed isotonic calibration state") from exc
            if (
                x_thresholds.ndim != 1
                or y_thresholds.ndim != 1
                or len(x_thresholds) < 2
                or len(x_thresholds) != len(y_thresholds)
                or not np.isfinite(x_thresholds).all()
                or not np.isfinite(y_thresholds).all()
                or not (np.diff(x_thresholds) > 0).all()
                or not (np.diff(y_thresholds) >= 0).all()
                or ((y_thresholds < 0) | (y_thresholds > 1)).any()
            ):
                raise CalibrationError("isotonic calibration thresholds violate invariants")
        instance._state = state
        try:
            _ = instance.predict([0.5])
        except (KeyError, TypeError, ValueError) as exc:
            raise CalibrationError("malformed calibration state") from exc
        return instance


@dataclass(frozen=True)
class CalibrationComparison:
    """Raw-versus-calibrated evidence on a post-fit evaluation period."""

    method: Literal["platt", "isotonic"]
    evaluation_start: str
    evaluation_end: str
    n_observations: int
    raw_brier: float
    calibrated_brier: float
    raw_ece: float
    calibrated_ece: float
    brier_improvement_interval: BootstrapInterval
    limitations: tuple[str, ...]


def compare_calibration(
    calibrator: ProbabilityCalibrator,
    evaluation: CalibrationEvaluationData,
    bootstrap: BlockBootstrapConfig,
    *,
    n_bins: int = 10,
) -> CalibrationComparison:
    """Compare raw and calibrated probabilities after the fit period."""

    if calibrator.metadata_ is None:
        raise CalibrationError("probability calibrator is not fitted")
    if not isinstance(evaluation, CalibrationEvaluationData):
        raise TypeError("evaluation must be CalibrationEvaluationData")
    evaluation_dates = pd.DatetimeIndex(evaluation.dates)
    if evaluation_dates[0] <= pd.Timestamp(calibrator.metadata_.fit_end):
        raise CalibrationError("evaluation dates must be strictly after the calibration fit period")
    raw = np.asarray(evaluation.probabilities, dtype=float)
    outcome = np.asarray(evaluation.outcomes, dtype=int)
    calibrated = calibrator.predict(raw)
    raw_reliability = reliability_diagram(raw, outcome, n_bins=n_bins)
    calibrated_reliability = reliability_diagram(calibrated, outcome, n_bins=n_bins)
    interval = paired_block_bootstrap_difference(
        (raw - outcome) ** 2,
        (calibrated - outcome) ** 2,
        bootstrap,
    )
    return CalibrationComparison(
        method=calibrator.metadata_.method,
        evaluation_start=evaluation.dates[0],
        evaluation_end=evaluation.dates[-1],
        n_observations=len(raw),
        raw_brier=raw_reliability.brier_score,
        calibrated_brier=calibrated_reliability.brier_score,
        raw_ece=raw_reliability.expected_calibration_error,
        calibrated_ece=calibrated_reliability.expected_calibration_error,
        brier_improvement_interval=interval,
        limitations=(
            "positive improvement is evidence for this period, not a future guarantee",
            "ECE depends on the declared binning policy",
            "the interval inherits moving-block stationarity and block-length assumptions",
        ),
    )
