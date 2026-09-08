"""Frozen synthetic engineering study for SF-S3-MR8 representations.

The study proves leakage boundaries and comparative mechanics on a deterministic
reference with planted factors, regimes, and anomalies.  It is not market
evidence.  All candidates share complete-date splits and fixed downstream
evaluators; selection is finalized from validation evidence before the test
partition is summarized.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import platform
import shutil
import sys
import tempfile
from dataclasses import dataclass
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any, Literal

import matplotlib
import numpy as np
import pandas as pd
import seaborn as sns
import yaml
from numpy.typing import NDArray
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

matplotlib.use("Agg")
from matplotlib import pyplot as plt  # noqa: E402

from alphaforge.representations import (
    REPRESENTATION_KINDS,
    RepresentationBatch,
    RepresentationCapabilityError,
    RepresentationConfig,
    RepresentationKind,
    create_representation,
    named_seed,
)
from alphaforge.representations.base import (
    BaseRepresentation,
    FloatArray,
    _readonly_float_array,
    stable_identity,
)

REPRESENTATION_STUDY_SCHEMA_VERSION = "1.0.0"
REPRESENTATION_CANDIDATES: tuple[RepresentationKind, ...] = REPRESENTATION_KINDS
SelectionMetric = Literal["validation_rank_ic"]

_ROOT_FIELDS = frozenset({"study", "synthetic_reference", "representation"})
_STUDY_FIELDS = frozenset(
    {
        "schema_version",
        "seed",
        "train_fraction",
        "validation_fraction",
        "selection_metric",
        "prediction_alpha",
        "interpretation",
    }
)
_SYNTHETIC_FIELDS = frozenset(
    {
        "dates",
        "symbols",
        "features",
        "noise_scale",
        "anomaly_fraction",
        "regime_period",
    }
)
_REPRESENTATION_FIELDS = frozenset(
    {
        "candidates",
        "latent_dim",
        "hidden_dim",
        "sequence_length",
        "batch_size",
        "incremental_batch_size",
        "max_epochs",
        "patience",
        "learning_rate",
        "weight_decay",
        "inner_validation_fraction",
        "corruption_std",
        "vae_beta",
        "contrastive_temperature",
        "augmentation_jitter_std",
        "augmentation_scale_std",
        "augmentation_mask_probability",
        "min_embedding_variance",
        "max_samples",
        "max_features",
        "max_tensor_bytes",
        "max_parameters",
    }
)


@dataclass(frozen=True)
class RepresentationStudyConfig:
    """Resolved pre-registration for a deterministic representation study."""

    seed: int
    train_fraction: float
    validation_fraction: float
    selection_metric: SelectionMetric
    prediction_alpha: float
    interpretation: str
    synthetic_reference: dict[str, Any]
    candidate_configs: tuple[RepresentationConfig, ...]

    def __post_init__(self) -> None:
        if isinstance(self.seed, bool) or not isinstance(self.seed, int) or self.seed < 0:
            raise ValueError("study seed must be a non-negative integer")
        if not 0.4 <= self.train_fraction <= 0.75:
            raise ValueError("train_fraction must be in [0.4, 0.75]")
        if not 0.10 <= self.validation_fraction <= 0.30:
            raise ValueError("validation_fraction must be in [0.10, 0.30]")
        if self.train_fraction + self.validation_fraction > 0.90:
            raise ValueError("at least 10% of dates must remain for test evaluation")
        if self.selection_metric != "validation_rank_ic":
            raise ValueError("selection_metric must be validation_rank_ic")
        if not math.isfinite(self.prediction_alpha) or self.prediction_alpha <= 0.0:
            raise ValueError("prediction_alpha must be finite and positive")
        if not isinstance(self.interpretation, str) or not self.interpretation.strip():
            raise ValueError("interpretation must be a non-empty string")
        kinds = tuple(config.kind for config in self.candidate_configs)
        if kinds != REPRESENTATION_CANDIDATES:
            raise ValueError(
                f"candidate family must be frozen as {list(REPRESENTATION_CANDIDATES)}"
            )

    @property
    def config_id(self) -> str:
        return stable_identity(
            {
                "schema_version": REPRESENTATION_STUDY_SCHEMA_VERSION,
                "seed": self.seed,
                "train_fraction": self.train_fraction,
                "validation_fraction": self.validation_fraction,
                "selection_metric": self.selection_metric,
                "prediction_alpha": self.prediction_alpha,
                "interpretation": self.interpretation,
                "synthetic_reference": self.synthetic_reference,
                "candidate_config_ids": [config.config_id for config in self.candidate_configs],
            }
        )


@dataclass(frozen=True)
class SyntheticRepresentationReference:
    """Aligned synthetic features and evaluation-only outcomes."""

    batch: RepresentationBatch
    target: FloatArray
    regime: NDArray[np.int64]
    anomaly: NDArray[np.int64]

    def __post_init__(self) -> None:
        target = _readonly_float_array(self.target, name="synthetic target", dimensions=1)
        regime = np.asarray(self.regime)
        anomaly = np.asarray(self.anomaly)
        for name, values in (("regime", regime), ("anomaly", anomaly)):
            if values.ndim != 1 or len(values) != self.batch.n_rows:
                raise ValueError(f"synthetic {name} must align with feature rows")
            if not np.issubdtype(values.dtype, np.integer):
                raise ValueError(f"synthetic {name} must contain integers")
        if len(target) != self.batch.n_rows:
            raise ValueError("synthetic target must align with feature rows")
        if set(np.unique(regime)) != {0, 1}:
            raise ValueError("synthetic reference must contain both regimes")
        if not set(np.unique(anomaly)).issubset({0, 1}) or anomaly.sum() == 0:
            raise ValueError("synthetic anomaly labels must be binary and non-empty")
        regime_copy = np.ascontiguousarray(regime.astype(np.int64))
        anomaly_copy = np.ascontiguousarray(anomaly.astype(np.int64))
        regime_copy.setflags(write=False)
        anomaly_copy.setflags(write=False)
        object.__setattr__(self, "target", target)
        object.__setattr__(self, "regime", regime_copy)
        object.__setattr__(self, "anomaly", anomaly_copy)


@dataclass(frozen=True)
class RepresentationStudySplit:
    """Complete-date fit, validation-selection, and test masks."""

    train: NDArray[np.bool_]
    validation: NDArray[np.bool_]
    test: NDArray[np.bool_]
    train_end: str
    validation_end: str
    test_end: str

    def __post_init__(self) -> None:
        masks = tuple(
            np.asarray(mask, dtype=bool) for mask in (self.train, self.validation, self.test)
        )
        if any(mask.ndim != 1 for mask in masks):
            raise ValueError("study split masks must be one-dimensional")
        if len({len(mask) for mask in masks}) != 1:
            raise ValueError("study split masks must have equal length")
        if any(
            (left & right).any()
            for left, right in ((masks[0], masks[1]), (masks[0], masks[2]), (masks[1], masks[2]))
        ):
            raise ValueError("study split masks must be disjoint")
        if not np.logical_or.reduce(masks).all() or any(not mask.any() for mask in masks):
            raise ValueError("study split masks must cover every row and each be non-empty")
        for field, mask in zip(("train", "validation", "test"), masks, strict=True):
            copy_mask = np.ascontiguousarray(mask)
            copy_mask.setflags(write=False)
            object.__setattr__(self, field, copy_mask)


@dataclass(frozen=True)
class RepresentationStudyResult:
    """Aggregate-only evidence published by one study run."""

    output_dir: Path
    summary: pd.DataFrame
    metadata: dict[str, Any]
    manifest: dict[str, Any]


def _mapping(value: object, name: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be a mapping")
    return dict(value)


def _exact_fields(value: dict[str, Any], expected: frozenset[str], name: str) -> None:
    if set(value) != expected:
        raise ValueError(
            f"{name} fields mismatch: expected {sorted(expected)}, got {sorted(value)}"
        )


def load_representation_study_config(path: str | Path) -> RepresentationStudyConfig:
    """Load an exact-field YAML pre-registration without importing Torch."""

    source = Path(path)
    try:
        payload = yaml.safe_load(source.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, yaml.YAMLError) as exc:
        raise ValueError(f"cannot load representation study config: {source}") from exc
    root = _mapping(payload, "root")
    _exact_fields(root, _ROOT_FIELDS, "root")
    study = _mapping(root["study"], "study")
    synthetic = _mapping(root["synthetic_reference"], "synthetic_reference")
    representation = _mapping(root["representation"], "representation")
    _exact_fields(study, _STUDY_FIELDS, "study")
    _exact_fields(synthetic, _SYNTHETIC_FIELDS, "synthetic_reference")
    _exact_fields(representation, _REPRESENTATION_FIELDS, "representation")
    if study["schema_version"] != REPRESENTATION_STUDY_SCHEMA_VERSION:
        raise ValueError("unsupported representation study schema version")
    candidates = representation["candidates"]
    if not isinstance(candidates, list) or tuple(candidates) != REPRESENTATION_CANDIDATES:
        raise ValueError(
            f"representation candidates must be exactly {list(REPRESENTATION_CANDIDATES)}"
        )
    seed = study["seed"]
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ValueError("study.seed must be an integer")
    common = {
        "latent_dim": representation["latent_dim"],
        "hidden_dim": representation["hidden_dim"],
        "sequence_length": representation["sequence_length"],
        "batch_size": representation["batch_size"],
        "incremental_batch_size": representation["incremental_batch_size"],
        "max_epochs": representation["max_epochs"],
        "patience": representation["patience"],
        "learning_rate": representation["learning_rate"],
        "weight_decay": representation["weight_decay"],
        "validation_fraction": representation["inner_validation_fraction"],
        "corruption_std": representation["corruption_std"],
        "vae_beta": representation["vae_beta"],
        "contrastive_temperature": representation["contrastive_temperature"],
        "augmentation_jitter_std": representation["augmentation_jitter_std"],
        "augmentation_scale_std": representation["augmentation_scale_std"],
        "augmentation_mask_probability": representation["augmentation_mask_probability"],
        "min_embedding_variance": representation["min_embedding_variance"],
        "max_samples": representation["max_samples"],
        "max_features": representation["max_features"],
        "max_tensor_bytes": representation["max_tensor_bytes"],
        "max_parameters": representation["max_parameters"],
    }
    candidate_configs = tuple(
        RepresentationConfig(
            kind=kind,
            seed=named_seed(seed, f"representation:{kind}"),
            **common,
        )
        for kind in REPRESENTATION_CANDIDATES
    )
    for name in ("dates", "symbols", "features", "regime_period"):
        value = synthetic[name]
        if isinstance(value, bool) or not isinstance(value, int) or value < 2:
            raise ValueError(f"synthetic_reference.{name} must be an integer >= 2")
    for name in ("noise_scale", "anomaly_fraction"):
        value = synthetic[name]
        if not isinstance(value, (int, float)) or not math.isfinite(float(value)):
            raise ValueError(f"synthetic_reference.{name} must be finite")
    if float(synthetic["noise_scale"]) <= 0.0:
        raise ValueError("synthetic_reference.noise_scale must be positive")
    if not 0.01 <= float(synthetic["anomaly_fraction"]) <= 0.25:
        raise ValueError("synthetic_reference.anomaly_fraction must be in [0.01, 0.25]")
    if int(synthetic["features"]) < int(representation["latent_dim"]):
        raise ValueError("synthetic features must be at least latent_dim")
    return RepresentationStudyConfig(
        seed=seed,
        train_fraction=float(study["train_fraction"]),
        validation_fraction=float(study["validation_fraction"]),
        selection_metric=str(study["selection_metric"]),  # type: ignore[arg-type]
        prediction_alpha=float(study["prediction_alpha"]),
        interpretation=str(study["interpretation"]),
        synthetic_reference=synthetic,
        candidate_configs=candidate_configs,
    )


def build_synthetic_representation_reference(
    config: RepresentationStudyConfig,
) -> SyntheticRepresentationReference:
    """Create a planted nonlinear factor reference with regimes and anomalies."""

    spec = config.synthetic_reference
    n_dates = int(spec["dates"])
    n_symbols = int(spec["symbols"])
    n_features = int(spec["features"])
    n_rows = n_dates * n_symbols
    dates_unique = pd.bdate_range("2020-01-02", periods=n_dates)
    dates = np.repeat(dates_unique.to_numpy(), n_symbols)
    symbol_names = tuple(f"S{index:02d}" for index in range(n_symbols))
    symbols = np.tile(np.asarray(symbol_names, dtype=object), n_dates)

    factor_rng = np.random.default_rng(named_seed(config.seed, "synthetic:factors"))
    loading_rng = np.random.default_rng(named_seed(config.seed, "synthetic:loadings"))
    noise_rng = np.random.default_rng(named_seed(config.seed, "synthetic:noise"))
    anomaly_rng = np.random.default_rng(named_seed(config.seed, "synthetic:anomalies"))

    date_axis = np.linspace(0.0, 4.0 * np.pi, n_dates)
    regime_by_date = ((np.arange(n_dates) // int(spec["regime_period"])) % 2).astype(int)
    common = np.sin(date_axis) + 0.35 * np.cos(0.5 * date_axis)
    local = np.cos(1.7 * date_axis)
    symbol_loading = loading_rng.normal(0.0, 0.45, size=(n_symbols, 3))
    factor_rows = np.empty((n_rows, 4), dtype=float)
    for date_index in range(n_dates):
        for symbol_index in range(n_symbols):
            row = date_index * n_symbols + symbol_index
            regime = regime_by_date[date_index]
            factor_rows[row] = (
                common[date_index] + symbol_loading[symbol_index, 0],
                local[date_index] + symbol_loading[symbol_index, 1],
                float(regime) + 0.25 * symbol_loading[symbol_index, 2],
                common[date_index] * (1.0 if regime else -0.5),
            )
    factor_rows += factor_rng.normal(0.0, 0.08, size=factor_rows.shape)
    feature_loadings = loading_rng.normal(0.0, 0.7, size=(factor_rows.shape[1], n_features))
    values = factor_rows @ feature_loadings
    values += (
        0.25
        * np.square(factor_rows[:, [0]])
        @ loading_rng.normal(
            0.0,
            0.3,
            size=(1, n_features),
        )
    )
    values += noise_rng.normal(0.0, float(spec["noise_scale"]), size=values.shape)

    anomaly = (anomaly_rng.random(n_rows) < float(spec["anomaly_fraction"])).astype(np.int64)
    # Guarantee every chronological partition contains anomalies without making
    # their exact row locations dependent on the split configuration.
    anomaly[np.arange(0, n_rows, max(n_symbols * 5, 1))] = 1
    anomaly_direction = anomaly_rng.choice((-1.0, 1.0), size=(n_rows, 1))
    values += (
        anomaly[:, None]
        * anomaly_direction
        * loading_rng.normal(
            2.0,
            0.25,
            size=(n_rows, n_features),
        )
    )
    target = (
        0.06 * factor_rows[:, 0]
        - 0.04 * factor_rows[:, 1]
        + 0.05 * factor_rows[:, 3]
        + noise_rng.normal(0.0, 0.025, size=n_rows)
    )
    regime = np.repeat(regime_by_date, n_symbols).astype(np.int64)
    batch = RepresentationBatch(
        values=values,
        dates=tuple(str(value) for value in dates),
        symbols=tuple(str(value) for value in symbols),
        feature_names=tuple(f"feature_{index + 1:02d}" for index in range(n_features)),
    )
    return SyntheticRepresentationReference(
        batch=batch,
        target=target,
        regime=regime,
        anomaly=anomaly,
    )


def chronological_representation_split(
    reference: SyntheticRepresentationReference,
    config: RepresentationStudyConfig,
) -> RepresentationStudySplit:
    """Split on complete dates before any learned transform is fitted."""

    dates = pd.DatetimeIndex(reference.batch.dates)
    unique_dates = dates.unique().sort_values()
    if len(unique_dates) < 10:
        raise ValueError("representation study requires at least ten distinct dates")
    train_count = max(2, int(math.floor(len(unique_dates) * config.train_fraction)))
    validation_count = max(
        1,
        int(math.floor(len(unique_dates) * config.validation_fraction)),
    )
    if train_count + validation_count >= len(unique_dates):
        raise ValueError("representation split leaves no test dates")
    train_end = unique_dates[train_count - 1]
    validation_end = unique_dates[train_count + validation_count - 1]
    train = np.asarray(dates <= train_end)
    validation = np.asarray((dates > train_end) & (dates <= validation_end))
    test = np.asarray(dates > validation_end)
    return RepresentationStudySplit(
        train=train,
        validation=validation,
        test=test,
        train_end=train_end.isoformat(),
        validation_end=validation_end.isoformat(),
        test_end=unique_dates[-1].isoformat(),
    )


def _mean_daily_rank_ic(
    dates: tuple[str, ...],
    target: FloatArray,
    prediction: FloatArray,
) -> tuple[float, float, int]:
    """Return the mean, standard error, and date count of daily rank IC.

    The standard error describes variation across the observed dates.  It is
    not a confidence interval and does not correct for serial dependence.
    """

    frame = pd.DataFrame(
        {
            "date": pd.to_datetime(dates),
            "target": target,
            "prediction": prediction,
        }
    )
    values: list[float] = []
    for _, group in frame.groupby("date", sort=True):
        if group["target"].nunique() > 1 and group["prediction"].nunique() > 1:
            value = group["target"].rank().corr(group["prediction"].rank())
            if pd.notna(value) and np.isfinite(value):
                values.append(float(value))
    if not values:
        return 0.0, 0.0, 0
    daily = np.asarray(values, dtype=np.float64)
    standard_error = 0.0 if len(daily) < 2 else float(np.std(daily, ddof=1) / np.sqrt(len(daily)))
    return float(np.mean(daily)), standard_error, len(daily)


def _prediction_metrics(
    train_values: FloatArray,
    train_target: FloatArray,
    evaluation_values: FloatArray,
    evaluation_target: FloatArray,
    evaluation_dates: tuple[str, ...],
    *,
    alpha: float,
) -> tuple[float, float, float, int, FloatArray]:
    model = make_pipeline(StandardScaler(), Ridge(alpha=alpha))
    model.fit(train_values, train_target)
    prediction = _readonly_float_array(
        model.predict(evaluation_values),
        name="downstream predictions",
        dimensions=1,
    )
    mse = float(np.mean(np.square(prediction - evaluation_target)))
    rank_ic, rank_ic_daily_se, rank_ic_dates = _mean_daily_rank_ic(
        evaluation_dates,
        evaluation_target,
        prediction,
    )
    return mse, rank_ic, rank_ic_daily_se, rank_ic_dates, prediction


def _regime_accuracy(
    train_values: FloatArray,
    train_regime: NDArray[np.int64],
    test_values: FloatArray,
    test_regime: NDArray[np.int64],
    *,
    seed: int,
) -> float:
    model = make_pipeline(
        StandardScaler(),
        LogisticRegression(
            C=1.0,
            solver="liblinear",
            max_iter=500,
            random_state=seed,
        ),
    )
    model.fit(train_values, train_regime)
    return float(np.mean(model.predict(test_values) == test_regime))


def _scaled_train_test(
    train_values: FloatArray,
    test_values: FloatArray,
) -> tuple[FloatArray, FloatArray]:
    center = np.mean(train_values, axis=0)
    scale = np.std(train_values, axis=0, ddof=0)
    scale = np.where(scale > np.finfo(float).eps, scale, 1.0)
    return (
        _readonly_float_array(
            (train_values - center) / scale,
            name="scaled train embeddings",
            dimensions=2,
        ),
        _readonly_float_array(
            (test_values - center) / scale,
            name="scaled test embeddings",
            dimensions=2,
        ),
    )


def _similarity_regime_recall(
    train_values: FloatArray,
    train_regime: NDArray[np.int64],
    test_values: FloatArray,
    test_regime: NDArray[np.int64],
) -> float:
    scaled_train, scaled_test = _scaled_train_test(train_values, test_values)
    distances = (
        np.sum(np.square(scaled_test), axis=1)[:, None]
        + np.sum(np.square(scaled_train), axis=1)[None, :]
        - 2.0 * scaled_test @ scaled_train.T
    )
    nearest = np.argmin(distances, axis=1)
    return float(np.mean(train_regime[nearest] == test_regime))


def _anomaly_auc(
    train_values: FloatArray,
    test_values: FloatArray,
    test_anomaly: NDArray[np.int64],
) -> float:
    scaled_train, scaled_test = _scaled_train_test(train_values, test_values)
    center = np.mean(scaled_train, axis=0)
    score = np.sum(np.square(scaled_test - center), axis=1)
    if len(np.unique(test_anomaly)) < 2:
        raise ValueError("test partition must contain normal and anomalous rows")
    return float(roc_auc_score(test_anomaly, score))


def _diversification(values: FloatArray) -> tuple[float, float]:
    if len(values) < 2:
        return 1.0, 0.0
    covariance = np.cov(values, rowvar=False, ddof=1)
    covariance = np.atleast_2d(covariance)
    eigenvalues = np.clip(np.linalg.eigvalsh(covariance), 0.0, None)
    total = float(eigenvalues.sum())
    if total <= np.finfo(float).eps:
        return 1.0, 0.0
    probabilities = eigenvalues[eigenvalues > 0.0] / total
    effective_rank = float(np.exp(-np.sum(probabilities * np.log(probabilities))))
    deviations = np.std(values, axis=0, ddof=0)
    variable = deviations > np.finfo(float).eps
    if variable.sum() < 2:
        return effective_rank, 0.0
    correlation: FloatArray = np.asarray(
        np.corrcoef(values[:, variable], rowvar=False),
        dtype=np.float64,
    )
    row_indices, column_indices = np.triu_indices(correlation.shape[0], k=1)
    upper = np.abs(correlation[row_indices, column_indices])
    return effective_rank, float(np.mean(upper))


def evaluate_representation_candidates(
    reference: SyntheticRepresentationReference,
    split: RepresentationStudySplit,
    config: RepresentationStudyConfig,
    *,
    candidate_order: tuple[RepresentationKind, ...] | None = None,
) -> pd.DataFrame:
    """Fit candidates independently and return canonical aggregate evidence."""

    order = candidate_order or REPRESENTATION_CANDIDATES
    if len(order) != len(set(order)) or set(order) != set(REPRESENTATION_CANDIDATES):
        raise ValueError("candidate_order must contain every frozen candidate exactly once")
    config_by_kind = {candidate.kind: candidate for candidate in config.candidate_configs}
    train_batch = reference.batch.take(split.train)
    development_mask = split.train | split.validation
    development_batch = reference.batch.take(development_mask)
    development_train_mask = split.train[development_mask]
    development_validation_mask = split.validation[development_mask]
    validation_dates = tuple(
        np.asarray(reference.batch.dates, dtype=object)[split.validation].tolist()
    )
    fitted: dict[RepresentationKind, BaseRepresentation] = {}
    training_embeddings: dict[RepresentationKind, FloatArray] = {}
    validation_records: dict[RepresentationKind, dict[str, Any]] = {}
    for kind in order:
        representation = create_representation(config_by_kind[kind])
        representation.fit(train_batch)
        if representation.state_ is None:  # pragma: no cover - fit contract
            raise RuntimeError("representation did not publish fitted state")
        # Validation sequence rows receive earlier observed training history,
        # but the test feature partition remains untouched until selection.
        development_embeddings = representation.transform(development_batch).values
        train_values = development_embeddings[development_train_mask]
        validation_values = development_embeddings[development_validation_mask]
        (
            validation_mse,
            validation_rank_ic,
            validation_rank_ic_daily_se,
            validation_rank_ic_dates,
            _,
        ) = _prediction_metrics(
            train_values,
            reference.target[split.train],
            validation_values,
            reference.target[split.validation],
            validation_dates,
            alpha=config.prediction_alpha,
        )
        fitted[kind] = representation
        training_embeddings[kind] = train_values
        validation_records[kind] = {
            "candidate": kind,
            "validation_prediction_mse": validation_mse,
            "validation_rank_ic": validation_rank_ic,
            "validation_rank_ic_daily_se": validation_rank_ic_daily_se,
            "validation_rank_ic_dates": validation_rank_ic_dates,
        }

    selection_frame = pd.DataFrame.from_records(list(validation_records.values()))
    selected = str(
        selection_frame.sort_values(
            ["validation_rank_ic", "validation_prediction_mse", "candidate"],
            ascending=[False, True, True],
            kind="mergesort",
        ).iloc[0]["candidate"]
    )

    rows: list[dict[str, Any]] = []
    test_dates = tuple(np.asarray(reference.batch.dates, dtype=object)[split.test].tolist())
    for kind in order:
        representation = fitted[kind]
        state = representation.state_
        if state is None:  # pragma: no cover - retained fit contract
            raise RuntimeError("representation lost fitted state")
        # Selection is now immutable. Sequence candidates receive the complete
        # feature history so each test row may use earlier observed features.
        test_values = representation.transform(reference.batch).values[split.test]
        train_values = training_embeddings[kind]
        (
            test_mse,
            test_rank_ic,
            test_rank_ic_daily_se,
            test_rank_ic_dates,
            test_prediction,
        ) = _prediction_metrics(
            train_values,
            reference.target[split.train],
            test_values,
            reference.target[split.test],
            test_dates,
            alpha=config.prediction_alpha,
        )
        regime_accuracy = _regime_accuracy(
            train_values,
            reference.regime[split.train],
            test_values,
            reference.regime[split.test],
            seed=named_seed(config.seed, f"downstream-regime:{kind}"),
        )
        similarity_recall = _similarity_regime_recall(
            train_values,
            reference.regime[split.train],
            test_values,
            reference.regime[split.test],
        )
        anomaly_auc = _anomaly_auc(
            train_values,
            test_values,
            reference.anomaly[split.test],
        )
        effective_rank, mean_abs_correlation = _diversification(test_values)
        try:
            reconstruction = representation.reconstruct(reference.batch)
        except RepresentationCapabilityError:
            test_reconstruction_mse = None
        else:
            test_reconstruction_mse = float(
                np.mean(np.square(reconstruction[split.test] - reference.batch.values[split.test]))
            )
        row = dict(validation_records[kind])
        row.update(
            {
                "state_id": state.state_id,
                "config_id": state.config_id,
                "output_dimensions": len(state.output_features),
                "test_prediction_mse": test_mse,
                "test_rank_ic": test_rank_ic,
                "test_rank_ic_daily_se": test_rank_ic_daily_se,
                "test_rank_ic_dates": test_rank_ic_dates,
                "test_regime_accuracy": regime_accuracy,
                "test_similarity_regime_recall": similarity_recall,
                "test_anomaly_auc": anomaly_auc,
                "test_effective_rank": effective_rank,
                "test_mean_abs_embedding_correlation": mean_abs_correlation,
                "test_reconstruction_mse": test_reconstruction_mse,
                "test_prediction_mean": float(np.mean(test_prediction)),
                "fit_rows": state.fit_rows,
                "inner_validation_rows": state.validation_rows,
                "iterations": state.iterations,
                "converged": state.converged,
                "stopping_reason": state.stopping_reason,
                "embedding_variance": state.embedding_variance,
                "parameter_count": state.parameter_count,
                "parameter_bytes": state.parameter_bytes,
                "fit_wall_seconds": state.fit_wall_seconds,
                "fit_cpu_seconds": state.fit_cpu_seconds,
            }
        )
        rows.append(row)
    summary = (
        pd.DataFrame.from_records(rows).set_index("candidate").loc[list(REPRESENTATION_CANDIDATES)]
    )
    summary["selected_on_validation"] = summary.index == selected
    raw = summary.loc["raw"]
    summary["test_rank_ic_delta_vs_raw"] = summary["test_rank_ic"] - raw["test_rank_ic"]
    summary["test_prediction_mse_delta_vs_raw"] = (
        summary["test_prediction_mse"] - raw["test_prediction_mse"]
    )
    return summary.reset_index()


_DISPLAY_NAMES = {
    "raw": "Raw control",
    "pca": "PCA",
    "incremental_pca": "Incremental PCA",
    "robust_pca": "Robust-scale PCA",
    "dense_autoencoder": "Dense AE",
    "sequence_autoencoder": "Sequence AE",
    "denoising_autoencoder": "Denoising AE",
    "variational_autoencoder": "VAE",
    "contrastive_timeseries": "Contrastive",
}


def _plot_summary(
    summary: pd.DataFrame,
    destination: Path,
    *,
    config: RepresentationStudyConfig,
    test_rows: int,
    test_dates: int,
) -> None:
    plot = summary.copy()
    plot["display_candidate"] = plot["candidate"].map(_DISPLAY_NAMES)
    raw_rank_ic = float(plot.loc[plot["candidate"].eq("raw"), "test_rank_ic"].iloc[0])
    selected = str(plot.loc[plot["selected_on_validation"], "display_candidate"].iloc[0])
    sns.set_theme(style="whitegrid", context="talk", palette="colorblind")
    figure, axes = plt.subplots(2, 2, figsize=(20, 14))
    sns.barplot(
        data=plot,
        x="test_rank_ic",
        y="display_candidate",
        ax=axes[0, 0],
        color=sns.color_palette("colorblind")[0],
    )
    axes[0, 0].axvline(raw_rank_ic, color="black", linestyle="--", label="Raw control")
    axes[0, 0].axvline(0.0, color="black", linewidth=0.8)
    axes[0, 0].errorbar(
        x=plot["test_rank_ic"],
        y=np.arange(len(plot)),
        xerr=plot["test_rank_ic_daily_se"],
        fmt="none",
        ecolor="black",
        capsize=3,
        linewidth=1.2,
        label="Daily standard error",
    )
    axes[0, 0].set(
        title="Prediction on untouched synthetic test", xlabel="Mean daily rank IC", ylabel=""
    )
    axes[0, 0].legend(fontsize=10)

    transfer = plot.melt(
        id_vars=["display_candidate"],
        value_vars=[
            "test_regime_accuracy",
            "test_similarity_regime_recall",
            "test_anomaly_auc",
        ],
        var_name="task",
        value_name="score",
    )
    transfer["task"] = transfer["task"].map(
        {
            "test_regime_accuracy": "Regime accuracy",
            "test_similarity_regime_recall": "Neighbor regime recall",
            "test_anomaly_auc": "Anomaly AUROC",
        }
    )
    sns.barplot(
        data=transfer,
        x="score",
        y="display_candidate",
        hue="task",
        ax=axes[0, 1],
    )
    axes[0, 1].axvline(0.5, color="black", linewidth=0.8, linestyle=":")
    axes[0, 1].set(title="Transfer diagnostics on synthetic test", xlabel="Score", ylabel="")
    axes[0, 1].legend(fontsize=9, title="")

    reconstructive = plot.dropna(subset=["test_reconstruction_mse"])
    sns.barplot(
        data=reconstructive,
        x="test_reconstruction_mse",
        y="display_candidate",
        ax=axes[1, 0],
        color=sns.color_palette("colorblind")[2],
    )
    axes[1, 0].set(title="Reconstruction diagnostic", xlabel="Test feature MSE", ylabel="")

    sns.barplot(
        data=plot,
        x="fit_wall_seconds",
        y="display_candidate",
        ax=axes[1, 1],
        color=sns.color_palette("colorblind")[3],
    )
    axes[1, 1].set(title="Measured fit cost", xlabel="Wall seconds on reference CPU", ylabel="")
    figure.suptitle(
        "SF-S3-MR8 latent representations — deterministic synthetic engineering evidence only\n"
        f"test n={test_rows}, dates={test_dates}, seed={config.seed}; "
        f"validation-selected={selected}",
        fontsize=17,
    )
    figure.tight_layout()
    figure.savefig(
        destination,
        dpi=160,
        bbox_inches="tight",
        metadata={"Software": "AlphaForge"},
    )
    plt.close(figure)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    encoded = json.dumps(
        payload,
        indent=2,
        sort_keys=True,
        ensure_ascii=True,
        allow_nan=False,
    )
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(encoded)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _installed_version(distribution: str) -> str:
    try:
        return version(distribution)
    except PackageNotFoundError:
        return "unavailable"


def _runtime_environment() -> dict[str, Any]:
    return {
        "python": platform.python_version(),
        "python_implementation": platform.python_implementation(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor() or "not reported by operating system",
        "logical_cpu_count": os.cpu_count(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "scikit_learn": _installed_version("scikit-learn"),
        "torch": _installed_version("torch"),
        "seaborn": sns.__version__,
        "reference_device": "cpu",
        "warmup_runs": 0,
        "measurement_samples_per_fit": 1,
        "executable_platform": sys.platform,
    }


def run_synthetic_representation_study(
    config: RepresentationStudyConfig,
    output_dir: str | Path,
) -> RepresentationStudyResult:
    """Run the frozen family and publish only aggregate evidence atomically."""

    destination = Path(output_dir)
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(f"study destination already exists: {destination}")
    staging = destination.parent / f".publishing-{destination.name}"
    if staging.exists() or staging.is_symlink():
        raise FileExistsError(f"stale study staging exists: {staging}")
    staging.mkdir(parents=True)
    try:
        reference = build_synthetic_representation_reference(config)
        split = chronological_representation_split(reference, config)
        summary = evaluate_representation_candidates(reference, split, config)
        selected = str(summary.loc[summary["selected_on_validation"], "candidate"].iloc[0])
        raw = summary.loc[summary["candidate"].eq("raw")].iloc[0]
        selected_row = summary.loc[summary["candidate"].eq(selected)].iloc[0]
        (staging / "plots").mkdir()
        summary_path = staging / "candidate_summary.csv"
        summary.to_csv(summary_path, index=False, float_format="%.12g")
        plot_path = staging / "plots" / "representation_comparison.png"
        test_dates = int(pd.Series(np.asarray(reference.batch.dates)[split.test]).nunique())
        _plot_summary(
            summary,
            plot_path,
            config=config,
            test_rows=int(split.test.sum()),
            test_dates=test_dates,
        )
        metadata = {
            "schema_version": REPRESENTATION_STUDY_SCHEMA_VERSION,
            "study": {
                "config_id": config.config_id,
                "candidate_order": list(REPRESENTATION_CANDIDATES),
                "selection_metric": config.selection_metric,
                "selected_candidate": selected,
                "seed": config.seed,
                "interpretation": config.interpretation,
                "test_rank_ic_delta_vs_raw": float(
                    selected_row["test_rank_ic"] - raw["test_rank_ic"]
                ),
                "test_prediction_mse_delta_vs_raw": float(
                    selected_row["test_prediction_mse"] - raw["test_prediction_mse"]
                ),
            },
            "data": {
                "kind": "deterministic_synthetic_engineering_reference",
                "rows": reference.batch.n_rows,
                "dates": int(pd.Series(reference.batch.dates).nunique()),
                "symbols": len(set(reference.batch.symbols)),
                "features": reference.batch.n_features,
                "anomaly_rows": int(reference.anomaly.sum()),
                "raw_rows_published": 0,
                "target_rows_published": 0,
                "embedding_rows_published": 0,
                "model_weights_published": 0,
            },
            "split": {
                "train_rows": int(split.train.sum()),
                "validation_rows": int(split.validation.sum()),
                "test_rows": int(split.test.sum()),
                "train_end": split.train_end,
                "validation_end": split.validation_end,
                "test_end": split.test_end,
            },
            "evidence": {
                "aggregate_rows": len(summary),
                "summary": "candidate_summary.csv",
                "plot": "plots/representation_comparison.png",
                "axes": [
                    "prediction",
                    "regime",
                    "similarity",
                    "anomaly",
                    "diversification",
                    "reconstruction",
                    "validation-only selection",
                    "compute",
                ],
            },
            "environment": _runtime_environment(),
            "limitations": [
                "Synthetic engineering evidence is not evidence of market predictability.",
                "The planted factors, regimes, and anomalies are cleaner than real markets.",
                "Validation selects architecture; the test partition does not tune fitted state.",
                "Rank-IC standard errors describe observed daily variation and do not correct for serial dependence.",
                "No licensed row, target, embedding, model weight, tensor, or prediction is published.",
                "No result authorizes paper or live trading or guarantees profit.",
                "Licensed point-in-time evaluation with costs remains a later governed study.",
            ],
        }
        summary_json = staging / "summary.json"
        _atomic_json(summary_json, metadata)
        evidence_files = (summary_path, summary_json, plot_path)
        manifest = {
            "schema_version": REPRESENTATION_STUDY_SCHEMA_VERSION,
            "kind": "aggregate_only_representation_evidence",
            "config_id": config.config_id,
            "files": [
                {
                    "path": str(path.relative_to(staging)),
                    "sha256": _sha256(path),
                    "bytes": path.stat().st_size,
                }
                for path in evidence_files
            ],
            "publication": {
                "raw_rows": False,
                "targets": False,
                "embeddings": False,
                "predictions": False,
                "model_weights": False,
                "credentials": False,
            },
        }
        _atomic_json(staging / "manifest.json", manifest)
        os.replace(staging, destination)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return RepresentationStudyResult(destination, summary, metadata, manifest)
