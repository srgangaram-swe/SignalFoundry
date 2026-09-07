"""Leakage-safe fitted transformations with immutable learned-state evidence."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.feature_selection import VarianceThreshold
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from alphaforge.features.registry import FeatureContractError, semantic_hash

TRANSFORM_VERSION = "1.0.0"


@dataclass(frozen=True)
class FittedTransformSpec:
    """Validated policy for preprocessing learned within one temporal fold."""

    version: str = TRANSFORM_VERSION
    enabled: bool = False
    imputation: str = "median"
    standardize: bool = True
    variance_threshold: float | None = None
    pca_components: int | float | None = None
    pca_whiten: bool = False
    min_fit_rows: int = 64

    def __post_init__(self) -> None:
        if self.version != TRANSFORM_VERSION:
            raise FeatureContractError(f"unsupported fitted-transform version {self.version!r}")
        if self.imputation not in {"median", "mean"}:
            raise FeatureContractError("imputation must be 'median' or 'mean'")
        if self.min_fit_rows <= 0:
            raise FeatureContractError("min_fit_rows must be positive")
        if self.variance_threshold is not None and (
            not np.isfinite(self.variance_threshold) or self.variance_threshold < 0
        ):
            raise FeatureContractError("variance_threshold must be finite and non-negative")
        if self.pca_components is not None:
            components = self.pca_components
            valid_int = (
                isinstance(components, int) and not isinstance(components, bool) and components > 0
            )
            valid_float = (
                isinstance(components, float) and np.isfinite(components) and 0 < components < 1
            )
            if not (valid_int or valid_float):
                raise FeatureContractError(
                    "pca_components must be a positive integer or float strictly between zero and one"
                )
        if self.pca_whiten and self.pca_components is None:
            raise FeatureContractError("pca_whiten requires pca_components")

    @classmethod
    def from_config(cls, config: dict[str, Any] | None) -> FittedTransformSpec:
        try:
            return cls(**(config or {}))
        except TypeError as exc:
            raise FeatureContractError("fitted-transform configuration fields mismatch") from exc

    @property
    def spec_id(self) -> str:
        return semantic_hash(asdict(self))


@dataclass(frozen=True)
class FittedTransformState:
    """Audit record proving where and how a transformation was fitted."""

    transform_version: str
    spec_id: str
    input_columns: tuple[str, ...]
    output_columns: tuple[str, ...]
    fit_start: str
    fit_end: str
    fit_rows: int
    state_id: str


def _finite_or_missing(frame: pd.DataFrame) -> np.ndarray:
    numeric = frame.apply(pd.to_numeric, errors="coerce")
    introduced = numeric.isna() & ~frame.isna()
    if introduced.any().any():
        raise FeatureContractError("fitted-transform inputs must be numeric")
    array = numeric.to_numpy(dtype=float)
    if np.isinf(array).any():
        raise FeatureContractError("fitted-transform inputs must not contain infinity")
    return array


def _array_record(value: np.ndarray | None) -> list[Any] | None:
    if value is None:
        return None
    array = np.asarray(value)
    if np.issubdtype(array.dtype, np.number) and not np.isfinite(array.astype(float)).all():
        raise FeatureContractError("learned transform state contains non-finite values")
    return array.tolist()


class FittedFeatureTransformer:
    """Fit imputation/scaling/selection/PCA on training observations only.

    ``fit`` records the exact temporal boundary and learned parameters.
    ``transform`` accepts only the identical ordered input schema. The class
    performs no implicit refit, fallback, or schema repair.
    """

    def __init__(self, spec: FittedTransformSpec | dict[str, Any] | None = None) -> None:
        self.spec = (
            spec if isinstance(spec, FittedTransformSpec) else FittedTransformSpec.from_config(spec)
        )
        self._pipeline: Pipeline | None = None
        self.state_: FittedTransformState | None = None

    def _build_pipeline(self) -> Pipeline:
        steps: list[tuple[str, Any]] = [
            (
                "imputer",
                SimpleImputer(strategy=self.spec.imputation, keep_empty_features=True),
            )
        ]
        if self.spec.standardize:
            steps.append(("scaler", StandardScaler()))
        if self.spec.variance_threshold is not None:
            steps.append(("variance", VarianceThreshold(self.spec.variance_threshold)))
        if self.spec.pca_components is not None:
            steps.append(
                (
                    "pca",
                    PCA(
                        n_components=self.spec.pca_components,
                        whiten=self.spec.pca_whiten,
                        svd_solver="full",
                    ),
                )
            )
        return Pipeline(steps)

    def fit(self, frame: pd.DataFrame, dates: pd.Series | pd.Index) -> FittedFeatureTransformer:
        if not isinstance(frame, pd.DataFrame) or frame.empty:
            raise FeatureContractError("fitted-transform training frame must be non-empty")
        if len(frame) < self.spec.min_fit_rows:
            raise FeatureContractError(
                f"insufficient fitted-transform history: {len(frame)} < {self.spec.min_fit_rows}"
            )
        if frame.columns.has_duplicates or any(
            not isinstance(column, str) or not column for column in frame.columns
        ):
            raise FeatureContractError("fitted-transform columns must be unique non-empty strings")
        date_index = pd.DatetimeIndex(pd.to_datetime(dates, errors="raise"))
        if len(date_index) != len(frame) or date_index.hasnans:
            raise FeatureContractError("fit dates must align one-for-one with training rows")
        values = _finite_or_missing(frame)
        if np.isnan(values).all(axis=0).any():
            raise FeatureContractError("training columns must contain at least one finite value")

        pipeline = self._build_pipeline()
        try:
            transformed = pipeline.fit_transform(frame)
        except ValueError as exc:
            raise FeatureContractError("fitted transformation rejected the training data") from exc
        if transformed.ndim != 2 or transformed.shape[1] == 0:
            raise FeatureContractError("fitted transformation removed every feature")
        if not np.isfinite(transformed).all():
            raise FeatureContractError("fitted transformation produced non-finite output")

        output_columns = self._output_columns(tuple(frame.columns), pipeline, transformed.shape[1])
        learned = self._learned_record(pipeline)
        state_id = semantic_hash(
            {
                "spec_id": self.spec.spec_id,
                "input_columns": tuple(frame.columns),
                "output_columns": output_columns,
                "learned": learned,
            }
        )
        self._pipeline = pipeline
        self.state_ = FittedTransformState(
            transform_version=TRANSFORM_VERSION,
            spec_id=self.spec.spec_id,
            input_columns=tuple(frame.columns),
            output_columns=output_columns,
            fit_start=date_index.min().isoformat(),
            fit_end=date_index.max().isoformat(),
            fit_rows=len(frame),
            state_id=state_id,
        )
        return self

    @staticmethod
    def _output_columns(
        input_columns: tuple[str, ...], pipeline: Pipeline, output_width: int
    ) -> tuple[str, ...]:
        if "pca" in pipeline.named_steps:
            return tuple(f"pc_{index + 1:03d}" for index in range(output_width))
        if "variance" in pipeline.named_steps:
            support = pipeline.named_steps["variance"].get_support()
            return tuple(
                column for column, keep in zip(input_columns, support, strict=True) if keep
            )
        return input_columns

    @staticmethod
    def _learned_record(pipeline: Pipeline) -> dict[str, Any]:
        imputer = pipeline.named_steps["imputer"]
        record: dict[str, Any] = {
            "imputer_statistics": _array_record(imputer.statistics_),
        }
        scaler = pipeline.named_steps.get("scaler")
        if scaler is not None:
            record["scaler_mean"] = _array_record(scaler.mean_)
            record["scaler_scale"] = _array_record(scaler.scale_)
        selector = pipeline.named_steps.get("variance")
        if selector is not None:
            record["variance_support"] = _array_record(selector.get_support())
            record["variances"] = _array_record(selector.variances_)
        pca = pipeline.named_steps.get("pca")
        if pca is not None:
            record["pca_components"] = _array_record(pca.components_)
            record["pca_mean"] = _array_record(pca.mean_)
            record["pca_explained_variance"] = _array_record(pca.explained_variance_)
        return record

    def transform(self, frame: pd.DataFrame) -> pd.DataFrame:
        if self._pipeline is None or self.state_ is None:
            raise FeatureContractError("fitted transformer must be fit before transform")
        if tuple(frame.columns) != self.state_.input_columns:
            raise FeatureContractError("fitted-transform input schema mismatch")
        _finite_or_missing(frame)
        try:
            transformed = self._pipeline.transform(frame)
        except ValueError as exc:
            raise FeatureContractError("fitted transformation rejected input data") from exc
        if not np.isfinite(transformed).all():
            raise FeatureContractError("fitted transformation produced non-finite output")
        return pd.DataFrame(
            transformed,
            index=frame.index,
            columns=list(self.state_.output_columns),
        )

    def fit_transform(self, frame: pd.DataFrame, dates: pd.Series | pd.Index) -> pd.DataFrame:
        return self.fit(frame, dates).transform(frame)
