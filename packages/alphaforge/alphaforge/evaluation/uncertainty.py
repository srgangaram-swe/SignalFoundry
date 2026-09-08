"""Dependence-aware uncertainty baselines for governed time-series research.

The contracts in this module deliberately separate training-derived
out-of-fold (OOF) residuals from future application data.  They do not claim
distribution-free coverage for arbitrary financial time series: the moving
block bootstrap assumes local stationarity, while block conformal intervals
assume exchangeable residual blocks.  Those assumptions are published in
every result so downstream reports cannot silently present an interval as a
guarantee.
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
from sklearn.linear_model import QuantileRegressor

_MAX_ARTIFACT_BYTES = 4 * 1024 * 1024
_MAX_BOOTSTRAP_RESAMPLES = 100_000
_MAX_FEATURES = 10_000
_MAX_SAMPLES = 10_000_000
_UNCERTAINTY_FORMAT = "alphaforge-uncertainty/json"
_UNCERTAINTY_VERSION = 1


class UncertaintyError(ValueError):
    """Raised when an uncertainty contract or assumption is violated."""


def _bounded_int(name: str, value: int, minimum: int, maximum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer")
    if not minimum <= value <= maximum:
        raise UncertaintyError(f"{name} must be in [{minimum}, {maximum}]")
    return value


def _open_unit(name: str, value: float) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise TypeError(f"{name} must be numeric")
    number = float(value)
    if not np.isfinite(number) or not 0.0 < number < 1.0:
        raise UncertaintyError(f"{name} must be finite and in (0, 1)")
    return number


def _finite_vector(values: object, name: str, *, minimum: int = 1) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    if array.ndim != 1:
        raise UncertaintyError(f"{name} must be one-dimensional")
    if len(array) < minimum:
        raise UncertaintyError(f"{name} must contain at least {minimum} observations")
    if len(array) > _MAX_SAMPLES:
        raise UncertaintyError(f"{name} exceeds the {_MAX_SAMPLES} observation limit")
    if not np.isfinite(array).all():
        raise UncertaintyError(f"{name} must contain only finite values")
    return array


def _ordered_dates(values: object, expected: int, name: str = "dates") -> pd.DatetimeIndex:
    try:
        dates = pd.DatetimeIndex(pd.to_datetime(values, errors="raise"))
    except (TypeError, ValueError) as exc:
        raise UncertaintyError(f"{name} must contain valid timestamps") from exc
    if len(dates) != expected:
        raise UncertaintyError(f"{name} length must match the observations")
    if dates.hasnans:
        raise UncertaintyError(f"{name} must not contain missing timestamps")
    if not dates.is_monotonic_increasing:
        raise UncertaintyError(f"{name} must be monotonically non-decreasing")
    return dates


@dataclass(frozen=True)
class BlockBootstrapConfig:
    """Bounded deterministic moving-block bootstrap settings."""

    n_resamples: int = 2_000
    block_length: int = 20
    confidence_level: float = 0.95
    seed: int = 42
    circular: bool = True

    def __post_init__(self) -> None:
        _bounded_int("n_resamples", self.n_resamples, 100, _MAX_BOOTSTRAP_RESAMPLES)
        _bounded_int("block_length", self.block_length, 2, _MAX_SAMPLES)
        _open_unit("confidence_level", self.confidence_level)
        _bounded_int("seed", self.seed, 0, 2**32 - 1)
        if not isinstance(self.circular, bool):
            raise TypeError("circular must be a boolean")


@dataclass(frozen=True)
class BootstrapInterval:
    """One scalar estimate and its dependence-aware uncertainty interval."""

    estimate: float
    lower: float
    upper: float
    standard_error: float
    statistic: Literal["mean", "median"]
    n_observations: int
    n_resamples: int
    block_length: int
    confidence_level: float
    seed: int
    assumptions: tuple[str, ...]


def _bootstrap_indices(
    n_observations: int, config: BlockBootstrapConfig, rng: np.random.Generator
) -> np.ndarray:
    if config.block_length > n_observations:
        raise UncertaintyError("block_length must not exceed the observation count")
    n_blocks = int(np.ceil(n_observations / config.block_length))
    max_start = n_observations if config.circular else n_observations - config.block_length + 1
    starts = rng.integers(0, max_start, size=n_blocks)
    offsets = np.arange(config.block_length)
    indices = (starts[:, None] + offsets[None, :]).reshape(-1)
    if config.circular:
        indices %= n_observations
    return indices[:n_observations]


def block_bootstrap_interval(
    values: object,
    config: BlockBootstrapConfig,
    *,
    statistic: Literal["mean", "median"] = "mean",
) -> BootstrapInterval:
    """Estimate a scalar interval using a deterministic moving-block bootstrap."""

    array = _finite_vector(values, "values", minimum=4)
    if not isinstance(config, BlockBootstrapConfig):
        raise TypeError("config must be a BlockBootstrapConfig")
    if statistic not in {"mean", "median"}:
        raise UncertaintyError("statistic must be 'mean' or 'median'")
    reducer = np.mean if statistic == "mean" else np.median
    rng = np.random.default_rng(config.seed)
    samples = np.empty(config.n_resamples, dtype=float)
    for index in range(config.n_resamples):
        samples[index] = float(reducer(array[_bootstrap_indices(len(array), config, rng)]))
    tail = (1.0 - config.confidence_level) / 2.0
    lower, upper = np.quantile(samples, [tail, 1.0 - tail])
    return BootstrapInterval(
        estimate=float(reducer(array)),
        lower=float(lower),
        upper=float(upper),
        standard_error=float(np.std(samples, ddof=1)),
        statistic=statistic,
        n_observations=len(array),
        n_resamples=config.n_resamples,
        block_length=config.block_length,
        confidence_level=config.confidence_level,
        seed=config.seed,
        assumptions=(
            "ordered observations preserve their original temporal spacing",
            "local dependence is represented by contiguous blocks",
            "the series is sufficiently stationary over the evaluated period",
        ),
    )


def paired_block_bootstrap_difference(
    first_losses: object,
    second_losses: object,
    config: BlockBootstrapConfig,
) -> BootstrapInterval:
    """Return an interval for ``mean(first_losses - second_losses)``.

    Pairing preserves observation-level dependence between two methods.  A
    positive estimate means the second method has lower loss.
    """

    first = _finite_vector(first_losses, "first_losses", minimum=4)
    second = _finite_vector(second_losses, "second_losses", minimum=4)
    if len(first) != len(second):
        raise UncertaintyError("paired loss vectors must have equal length")
    return block_bootstrap_interval(first - second, config, statistic="mean")


@dataclass(frozen=True)
class OOFRegressionData:
    """Training-derived OOF regression predictions with temporal provenance."""

    predictions: tuple[float, ...]
    outcomes: tuple[float, ...]
    dates: tuple[str, ...]
    fold_ids: tuple[int, ...]
    source: Literal["training_oof"] = "training_oof"

    def __post_init__(self) -> None:
        if self.source != "training_oof":
            raise UncertaintyError("OOF regression source must be 'training_oof'")
        prediction = _finite_vector(self.predictions, "predictions", minimum=10)
        outcome = _finite_vector(self.outcomes, "outcomes", minimum=10)
        if len(prediction) != len(outcome):
            raise UncertaintyError("predictions and outcomes must have equal length")
        _ordered_dates(self.dates, len(prediction))
        folds = np.asarray(self.fold_ids)
        if folds.ndim != 1 or len(folds) != len(prediction):
            raise UncertaintyError("fold_ids must be one-dimensional and match observations")
        if not np.issubdtype(folds.dtype, np.integer) or (folds < 0).any():
            raise UncertaintyError("fold_ids must contain non-negative integers")
        if len(np.unique(folds)) < 2:
            raise UncertaintyError("training OOF data must contain at least two folds")

    @classmethod
    def from_arrays(
        cls,
        predictions: object,
        outcomes: object,
        dates: object,
        fold_ids: object,
    ) -> OOFRegressionData:
        pred = _finite_vector(predictions, "predictions", minimum=10)
        target = _finite_vector(outcomes, "outcomes", minimum=10)
        if len(pred) != len(target):
            raise UncertaintyError("predictions and outcomes must have equal length")
        parsed_dates = _ordered_dates(dates, len(pred))
        folds = np.asarray(fold_ids)
        if folds.ndim != 1 or len(folds) != len(pred):
            raise UncertaintyError("fold_ids must be one-dimensional and match observations")
        if not np.issubdtype(folds.dtype, np.integer) or (folds < 0).any():
            raise UncertaintyError("fold_ids must contain non-negative integers")
        if len(np.unique(folds)) < 2:
            raise UncertaintyError("training OOF data must contain at least two folds")
        return cls(
            predictions=tuple(float(value) for value in pred),
            outcomes=tuple(float(value) for value in target),
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
class ConformalMetadata:
    """Published assumptions and provenance for a fitted conformal interval."""

    method: Literal["absolute_residual_block_max"]
    fit_start: str
    fit_end: str
    n_observations: int
    n_folds: int
    n_blocks: int
    block_length: int
    alpha: float
    radius: float
    limitations: tuple[str, ...]


class BlockConformalInterval:
    """Conservative split-conformal interval over OOF residual blocks."""

    def __init__(self, *, alpha: float = 0.10, block_length: int = 20, min_blocks: int = 5) -> None:
        self.alpha = _open_unit("alpha", alpha)
        self.block_length = _bounded_int("block_length", block_length, 2, _MAX_SAMPLES)
        self.min_blocks = _bounded_int("min_blocks", min_blocks, 3, 10_000)
        self.metadata_: ConformalMetadata | None = None

    def fit(self, data: OOFRegressionData) -> BlockConformalInterval:
        """Fit only from training-derived OOF residuals."""

        if not isinstance(data, OOFRegressionData) or data.source != "training_oof":
            raise TypeError("data must be OOFRegressionData with source='training_oof'")
        residuals = np.abs(np.asarray(data.outcomes) - np.asarray(data.predictions))
        n_blocks = len(residuals) // self.block_length
        if n_blocks < self.min_blocks:
            raise UncertaintyError(
                f"block conformal requires at least {self.min_blocks} complete blocks"
            )
        trimmed = residuals[: n_blocks * self.block_length]
        block_scores = trimmed.reshape(n_blocks, self.block_length).max(axis=1)
        quantile_level = min(1.0, np.ceil((n_blocks + 1) * (1.0 - self.alpha)) / n_blocks)
        radius = float(np.quantile(block_scores, quantile_level, method="higher"))
        self.metadata_ = ConformalMetadata(
            method="absolute_residual_block_max",
            fit_start=data.fit_start,
            fit_end=data.fit_end,
            n_observations=len(residuals),
            n_folds=len(set(data.fold_ids)),
            n_blocks=n_blocks,
            block_length=self.block_length,
            alpha=self.alpha,
            radius=radius,
            limitations=(
                "coverage requires exchangeable residual blocks after temporal ordering",
                "coverage can fail under regime shift, drift, or a changed prediction policy",
                "the block length must exceed material residual dependence",
                "the radius is marginal and not a simultaneous path guarantee",
            ),
        )
        return self

    def predict_interval(self, predictions: object, dates: object) -> pd.DataFrame:
        """Apply the fitted radius only to observations after the OOF fit period."""

        if self.metadata_ is None:
            raise UncertaintyError("block conformal interval is not fitted")
        point = _finite_vector(predictions, "predictions")
        future_dates = _ordered_dates(dates, len(point))
        if future_dates[0] <= pd.Timestamp(self.metadata_.fit_end):
            raise UncertaintyError("application dates must be strictly after the OOF fit period")
        radius = self.metadata_.radius
        return pd.DataFrame(
            {
                "date": future_dates,
                "prediction": point,
                "lower": point - radius,
                "upper": point + radius,
            }
        )

    def save(self, path: str | Path) -> Path:
        if self.metadata_ is None:
            raise UncertaintyError("block conformal interval is not fitted")
        payload = {
            "format": _UNCERTAINTY_FORMAT,
            "version": _UNCERTAINTY_VERSION,
            "kind": "block_conformal",
            "parameters": {
                "alpha": self.alpha,
                "block_length": self.block_length,
                "min_blocks": self.min_blocks,
            },
            "metadata": asdict(self.metadata_),
        }
        return _write_json_atomic(path, payload)

    @classmethod
    def load(cls, path: str | Path) -> BlockConformalInterval:
        payload = _read_json(path)
        if (
            payload.get("format") != _UNCERTAINTY_FORMAT
            or payload.get("version") != _UNCERTAINTY_VERSION
            or payload.get("kind") != "block_conformal"
        ):
            raise UncertaintyError("unsupported block conformal artifact")
        params = payload.get("parameters")
        metadata = payload.get("metadata")
        if not isinstance(params, dict) or not isinstance(metadata, dict):
            raise UncertaintyError("malformed block conformal artifact")
        try:
            instance = cls(**params)
            instance.metadata_ = ConformalMetadata(
                **{**metadata, "limitations": tuple(metadata["limitations"])}
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise UncertaintyError("malformed block conformal metadata") from exc
        fitted = instance.metadata_
        try:
            fit_start = pd.Timestamp(fitted.fit_start)
            fit_end = pd.Timestamp(fitted.fit_end)
            invalid_metadata = (
                fitted.method != "absolute_residual_block_max"
                or fit_start is pd.NaT
                or fit_end is pd.NaT
                or fitted.alpha != instance.alpha
                or fitted.block_length != instance.block_length
                or fitted.n_blocks < instance.min_blocks
                or fitted.n_observations < fitted.n_blocks * fitted.block_length
                or fitted.n_folds < 2
                or not np.isfinite(fitted.radius)
                or fitted.radius < 0
                or fit_start > fit_end
            )
        except (TypeError, ValueError, OverflowError) as exc:
            raise UncertaintyError("malformed block conformal metadata") from exc
        if invalid_metadata:
            raise UncertaintyError("block conformal metadata violates fitted-state invariants")
        return instance


@dataclass(frozen=True)
class QuantileRegressionConfig:
    """Bounded linear quantile-regression interval settings."""

    lower_quantile: float = 0.10
    upper_quantile: float = 0.90
    alpha: float = 0.001
    max_iter: int = 10_000
    max_samples: int = 1_000_000
    max_features: int = 10_000

    def __post_init__(self) -> None:
        lower = _open_unit("lower_quantile", self.lower_quantile)
        upper = _open_unit("upper_quantile", self.upper_quantile)
        if not lower < 0.5 < upper:
            raise UncertaintyError("quantiles must satisfy lower < 0.5 < upper")
        if isinstance(self.alpha, bool) or not isinstance(self.alpha, int | float):
            raise TypeError("alpha must be numeric")
        if not np.isfinite(self.alpha) or self.alpha < 0:
            raise UncertaintyError("alpha must be finite and non-negative")
        _bounded_int("max_iter", self.max_iter, 100, 1_000_000)
        _bounded_int("max_samples", self.max_samples, 10, _MAX_SAMPLES)
        _bounded_int("max_features", self.max_features, 1, _MAX_FEATURES)


@dataclass(frozen=True)
class QuantileRegressionMetadata:
    fit_start: str
    fit_end: str
    n_samples: int
    n_features: int
    feature_names: tuple[str, ...]
    quantiles: tuple[float, float, float]
    alpha: float
    iterations: tuple[int | None, int | None, int | None]
    limitations: tuple[str, ...]


class QuantileRegressionInterval:
    """Three bounded linear quantile regressions fit on one training fold."""

    def __init__(self, config: QuantileRegressionConfig | None = None) -> None:
        self.config = config or QuantileRegressionConfig()
        self.metadata_: QuantileRegressionMetadata | None = None
        self._coefficients: tuple[tuple[float, ...], ...] | None = None
        self._intercepts: tuple[float, ...] | None = None

    def fit(
        self, features: pd.DataFrame, outcomes: object, dates: object
    ) -> QuantileRegressionInterval:
        """Fit lower, median, and upper models on one declared training fold."""

        if not isinstance(features, pd.DataFrame):
            raise TypeError("features must be a pandas DataFrame")
        if not 1 <= features.shape[1] <= self.config.max_features:
            raise UncertaintyError("feature count exceeds configured bounds")
        if not 10 <= len(features) <= self.config.max_samples:
            raise UncertaintyError("training sample count exceeds configured bounds")
        matrix = features.to_numpy(dtype=float)
        if not np.isfinite(matrix).all():
            raise UncertaintyError("features must contain only finite values")
        target = _finite_vector(outcomes, "outcomes", minimum=10)
        if len(target) != len(features):
            raise UncertaintyError("features and outcomes must have equal length")
        train_dates = _ordered_dates(dates, len(features))
        quantiles = (self.config.lower_quantile, 0.5, self.config.upper_quantile)
        estimators = [
            QuantileRegressor(
                quantile=quantile,
                alpha=float(self.config.alpha),
                solver="highs",
                solver_options={"maxiter": self.config.max_iter},
            ).fit(matrix, target)
            for quantile in quantiles
        ]
        self._coefficients = tuple(
            tuple(float(value) for value in estimator.coef_) for estimator in estimators
        )
        self._intercepts = tuple(float(estimator.intercept_) for estimator in estimators)
        iteration_values = tuple(
            int(estimator.n_iter_) if hasattr(estimator, "n_iter_") else None
            for estimator in estimators
        )
        self.metadata_ = QuantileRegressionMetadata(
            fit_start=train_dates[0].isoformat(),
            fit_end=train_dates[-1].isoformat(),
            n_samples=len(features),
            n_features=features.shape[1],
            feature_names=tuple(str(column) for column in features.columns),
            quantiles=quantiles,
            alpha=float(self.config.alpha),
            iterations=(iteration_values[0], iteration_values[1], iteration_values[2]),
            limitations=(
                "linear conditional quantiles can be misspecified",
                "finite-sample coverage is not guaranteed by quantile regression alone",
                "prediction intervals can degrade under drift or changed feature semantics",
            ),
        )
        return self

    def predict_interval(self, features: pd.DataFrame) -> pd.DataFrame:
        if self.metadata_ is None or self._coefficients is None or self._intercepts is None:
            raise UncertaintyError("quantile regression interval is not fitted")
        if not isinstance(features, pd.DataFrame):
            raise TypeError("features must be a pandas DataFrame")
        if tuple(str(column) for column in features.columns) != self.metadata_.feature_names:
            raise UncertaintyError("prediction feature schema differs from the fitted schema")
        matrix = features.to_numpy(dtype=float)
        if not np.isfinite(matrix).all():
            raise UncertaintyError("features must contain only finite values")
        predictions = np.column_stack(
            [
                matrix @ np.asarray(coefficient) + intercept
                for coefficient, intercept in zip(self._coefficients, self._intercepts, strict=True)
            ]
        )
        crossed = (predictions[:, 0] > predictions[:, 1]) | (predictions[:, 1] > predictions[:, 2])
        if crossed.any():
            raise UncertaintyError(
                "quantile crossing detected; the interval is invalid and was not reordered"
            )
        return pd.DataFrame(
            predictions,
            columns=["lower", "median", "upper"],
            index=features.index,
        )

    def save(self, path: str | Path) -> Path:
        if self.metadata_ is None or self._coefficients is None or self._intercepts is None:
            raise UncertaintyError("quantile regression interval is not fitted")
        payload = {
            "format": _UNCERTAINTY_FORMAT,
            "version": _UNCERTAINTY_VERSION,
            "kind": "quantile_regression",
            "config": asdict(self.config),
            "metadata": asdict(self.metadata_),
            "coefficients": self._coefficients,
            "intercepts": self._intercepts,
        }
        return _write_json_atomic(path, payload)

    @classmethod
    def load(cls, path: str | Path) -> QuantileRegressionInterval:
        payload = _read_json(path)
        if (
            payload.get("format") != _UNCERTAINTY_FORMAT
            or payload.get("version") != _UNCERTAINTY_VERSION
            or payload.get("kind") != "quantile_regression"
        ):
            raise UncertaintyError("unsupported quantile-regression artifact")
        config = payload.get("config")
        metadata = payload.get("metadata")
        coefficients = payload.get("coefficients")
        intercepts = payload.get("intercepts")
        if not all(
            isinstance(value, dict | list) for value in (config, metadata, coefficients, intercepts)
        ):
            raise UncertaintyError("malformed quantile-regression artifact")
        config_dict = cast(dict[str, object], config)
        metadata_dict = cast(dict[str, object], metadata)
        coefficient_rows = cast(list[object], coefficients)
        intercept_values = cast(list[object], intercepts)
        try:
            instance = cls(
                QuantileRegressionConfig(
                    lower_quantile=float(cast(float, config_dict["lower_quantile"])),
                    upper_quantile=float(cast(float, config_dict["upper_quantile"])),
                    alpha=float(cast(float, config_dict["alpha"])),
                    max_iter=int(cast(int, config_dict["max_iter"])),
                    max_samples=int(cast(int, config_dict["max_samples"])),
                    max_features=int(cast(int, config_dict["max_features"])),
                )
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise UncertaintyError("malformed quantile-regression configuration") from exc
        try:
            instance.metadata_ = QuantileRegressionMetadata(
                fit_start=str(metadata_dict["fit_start"]),
                fit_end=str(metadata_dict["fit_end"]),
                n_samples=int(cast(int, metadata_dict["n_samples"])),
                n_features=int(cast(int, metadata_dict["n_features"])),
                feature_names=tuple(cast(list[str], metadata_dict["feature_names"])),
                quantiles=cast(
                    tuple[float, float, float],
                    tuple(float(value) for value in cast(list[float], metadata_dict["quantiles"])),
                ),
                alpha=float(cast(float, metadata_dict["alpha"])),
                iterations=cast(
                    tuple[int | None, int | None, int | None],
                    tuple(
                        cast(int | None, value)
                        for value in cast(list[object], metadata_dict["iterations"])
                    ),
                ),
                limitations=tuple(cast(list[str], metadata_dict["limitations"])),
            )
            instance._coefficients = tuple(
                tuple(float(value) for value in cast(list[float], row)) for row in coefficient_rows
            )
            instance._intercepts = tuple(float(cast(float, value)) for value in intercept_values)
        except (KeyError, TypeError, ValueError) as exc:
            raise UncertaintyError("malformed quantile-regression state") from exc
        if len(instance._coefficients) != 3 or len(instance._intercepts) != 3:
            raise UncertaintyError("quantile-regression state must contain three models")
        fitted = instance.metadata_
        try:
            coefficient_array = np.asarray(instance._coefficients, dtype=float)
            intercept_array = np.asarray(instance._intercepts, dtype=float)
            fit_start = pd.Timestamp(fitted.fit_start)
            fit_end = pd.Timestamp(fitted.fit_end)
            invalid_state = (
                fit_start is pd.NaT
                or fit_end is pd.NaT
                or fitted.n_samples < 10
                or fitted.n_samples > instance.config.max_samples
                or fitted.n_features != len(fitted.feature_names)
                or fitted.n_features < 1
                or fitted.n_features > instance.config.max_features
                or coefficient_array.shape != (3, fitted.n_features)
                or intercept_array.shape != (3,)
                or not np.isfinite(coefficient_array).all()
                or not np.isfinite(intercept_array).all()
                or fitted.quantiles
                != (
                    instance.config.lower_quantile,
                    0.5,
                    instance.config.upper_quantile,
                )
                or fitted.alpha != instance.config.alpha
                or fit_start > fit_end
            )
        except (TypeError, ValueError, OverflowError) as exc:
            raise UncertaintyError("malformed quantile-regression state") from exc
        if invalid_state:
            raise UncertaintyError("quantile-regression state violates fitted-state invariants")
        return instance


def _write_json_atomic(path: str | Path, payload: dict[str, object]) -> Path:
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


def _read_json(path: str | Path) -> dict[str, object]:
    source = Path(path)
    try:
        if source.stat().st_size > _MAX_ARTIFACT_BYTES:
            raise UncertaintyError("uncertainty artifact exceeds the size limit")
        payload = json.loads(source.read_text(encoding="utf-8"))
    except UncertaintyError:
        raise
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise UncertaintyError("could not read uncertainty artifact") from exc
    if not isinstance(payload, dict):
        raise UncertaintyError("uncertainty artifact root must be an object")
    return payload
