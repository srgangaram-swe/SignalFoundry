"""Frozen, gated study for time-frequency CNN progression (SF-S3-MR7).

The study has one decision path: tabular baselines, mandatory small CNN,
conditional ResNet, and conditional ViT.  Both gates use only the chronological
validation partition.  Test predictions are produced after the progression is
fully decided, preventing the final evidence from influencing model capacity.

The committed reference profile is deterministic synthetic engineering
evidence with a known frequency-localized response.  It proves mechanics,
alignment, gates, reproducibility, resource reporting, and failure behavior; it
does not establish a market edge.  The same typed batch accepts local
Signalattice-derived tensors for later licensed-data studies.
"""

from __future__ import annotations

import json
import os
import platform
import shutil
import tempfile
import time
from dataclasses import asdict, dataclass
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import yaml

from alphaforge.models.deep_sequence import evaluate_oos_predictions
from alphaforge.models.sklearn_models import make_lightgbm
from alphaforge.models.time_frequency_vision import (
    ControlledVisionModel,
    PredictionMetrics,
    ProgressionGateEvidence,
    ProgressionGatePolicy,
    TimeFrequencyBatch,
    VisionAugmentationConfig,
    VisionResourceEvidence,
    VisionTrainingConfig,
    evaluate_progression_gate,
    prediction_metrics,
)

STUDY_SCHEMA_VERSION = "1.0.0"
TIME_FREQUENCY_CANDIDATES = (
    "lightgbm_time",
    "lightgbm_spectral",
    "small_cnn",
    "resnet",
    "vit",
)
_ROOT_FIELDS = frozenset({"study", "synthetic_reference", "lightgbm", "vision", "gates"})
_STUDY_FIELDS = frozenset(
    {
        "schema_version",
        "seed",
        "train_fraction",
        "validation_fraction",
        "transaction_cost_bps",
        "selection_fraction",
        "interpretation",
    }
)
_SYNTHETIC_FIELDS = frozenset(
    {"dates", "symbols", "channels", "frequency_bins", "time_steps", "noise_scale"}
)
_LIGHTGBM_FIELDS = frozenset(
    {
        "n_estimators",
        "max_depth",
        "learning_rate",
        "min_samples_leaf",
        "l2_regularization",
        "n_jobs",
        "random_state",
    }
)
_VISION_FIELDS = frozenset(
    {
        "width",
        "residual_blocks",
        "vit_layers",
        "vit_heads",
        "patch_frequency",
        "patch_time",
        "dropout",
        "learning_rate",
        "weight_decay",
        "max_epochs",
        "patience",
        "batch_size",
        "gradient_clip",
        "max_parameters",
        "max_tensor_bytes",
        "seed",
        "device",
        "augmentation",
    }
)
_AUGMENTATION_FIELDS = frozenset({"probability", "log_amplitude_std", "max_frequency_mask_bins"})
_GATE_FIELDS = frozenset(
    {
        "minimum_observations",
        "minimum_dates",
        "minimum_rank_ic",
        "minimum_incremental_rank_ic",
        "maximum_rmse_ratio",
    }
)


@dataclass(frozen=True)
class TimeFrequencyStudyConfig:
    """Fully resolved study and capacity policy."""

    seed: int
    train_fraction: float
    validation_fraction: float
    transaction_cost_bps: float
    selection_fraction: float
    interpretation: str
    synthetic_reference: dict[str, Any]
    lightgbm: dict[str, Any]
    vision: VisionTrainingConfig
    small_cnn_gate: ProgressionGatePolicy
    resnet_gate: ProgressionGatePolicy

    def __post_init__(self) -> None:
        if self.seed < 0:
            raise ValueError("seed must be non-negative")
        if not 0.4 <= self.train_fraction <= 0.8:
            raise ValueError("train_fraction must be in [0.4, 0.8]")
        if not 0.1 <= self.validation_fraction <= 0.3:
            raise ValueError("validation_fraction must be in [0.1, 0.3]")
        if self.train_fraction + self.validation_fraction >= 0.9:
            raise ValueError("at least 10% of dates must remain untouched for test")
        if not np.isfinite(self.transaction_cost_bps) or self.transaction_cost_bps < 0.0:
            raise ValueError("transaction_cost_bps must be finite and non-negative")
        if not 0.0 < self.selection_fraction <= 0.5:
            raise ValueError("selection_fraction must be in (0, 0.5]")
        if not self.interpretation.strip():
            raise ValueError("interpretation must be non-empty")


@dataclass(frozen=True)
class TimeFrequencyStudyResult:
    """Published aggregate evidence and gate decisions."""

    output_dir: Path
    summary: pd.DataFrame
    gates: tuple[ProgressionGateEvidence, ...]
    metadata: dict[str, Any]


@dataclass(frozen=True)
class _ChronologicalSplit:
    train: np.ndarray
    validation: np.ndarray
    test: np.ndarray
    train_end: str
    validation_end: str
    test_end: str


@dataclass(frozen=True)
class _FittedCandidate:
    name: str
    validation_prediction: np.ndarray
    resource: dict[str, Any]
    model: Any


def load_time_frequency_study_config(path: str | Path) -> TimeFrequencyStudyConfig:
    """Load and strictly validate the frozen YAML study profile."""
    payload = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("time-frequency study config must be a mapping")
    _exact_fields(payload, _ROOT_FIELDS, "root")
    study = _mapping(payload["study"], "study")
    _exact_fields(study, _STUDY_FIELDS, "study")
    if study["schema_version"] != STUDY_SCHEMA_VERSION:
        raise ValueError(f"study.schema_version must be {STUDY_SCHEMA_VERSION}")
    synthetic = _mapping(payload["synthetic_reference"], "synthetic_reference")
    _exact_fields(synthetic, _SYNTHETIC_FIELDS, "synthetic_reference")
    lightgbm = _mapping(payload["lightgbm"], "lightgbm")
    _exact_fields(lightgbm, _LIGHTGBM_FIELDS, "lightgbm")
    vision_raw = _mapping(payload["vision"], "vision")
    _exact_fields(vision_raw, _VISION_FIELDS, "vision")
    augmentation_raw = _mapping(vision_raw["augmentation"], "vision.augmentation")
    _exact_fields(augmentation_raw, _AUGMENTATION_FIELDS, "vision.augmentation")
    vision = VisionTrainingConfig(
        **{key: value for key, value in vision_raw.items() if key != "augmentation"},
        augmentation=VisionAugmentationConfig(**augmentation_raw),
    )
    gates = _mapping(payload["gates"], "gates")
    _exact_fields(gates, frozenset({"small_cnn_to_resnet", "resnet_to_vit"}), "gates")
    small_gate = _mapping(gates["small_cnn_to_resnet"], "gates.small_cnn_to_resnet")
    resnet_gate = _mapping(gates["resnet_to_vit"], "gates.resnet_to_vit")
    _exact_fields(small_gate, _GATE_FIELDS, "gates.small_cnn_to_resnet")
    _exact_fields(resnet_gate, _GATE_FIELDS, "gates.resnet_to_vit")
    seed = study["seed"]
    if not isinstance(seed, int) or isinstance(seed, bool):
        raise ValueError("study.seed must be an integer")
    return TimeFrequencyStudyConfig(
        seed=seed,
        train_fraction=float(study["train_fraction"]),
        validation_fraction=float(study["validation_fraction"]),
        transaction_cost_bps=float(study["transaction_cost_bps"]),
        selection_fraction=float(study["selection_fraction"]),
        interpretation=str(study["interpretation"]),
        synthetic_reference=dict(synthetic),
        lightgbm=dict(lightgbm),
        vision=vision,
        small_cnn_gate=ProgressionGatePolicy(**small_gate),
        resnet_gate=ProgressionGatePolicy(**resnet_gate),
    )


def build_synthetic_time_frequency_reference(
    config: TimeFrequencyStudyConfig,
) -> TimeFrequencyBatch:
    """Build a reproducible reference with a known localized spectral response.

    The response combines low-frequency channel energy, a recent-time contrast,
    and independent noise.  Conventional features expose coarse moments;
    spectral descriptors expose explicit band summaries.  This construction is
    deliberately inspectable and is not intended to resemble market returns.
    """
    specification = config.synthetic_reference
    n_dates = _bounded_int(specification["dates"], "dates", 20, 2_000)
    n_symbols = _bounded_int(specification["symbols"], "symbols", 4, 500)
    n_channels = _bounded_int(specification["channels"], "channels", 1, 16)
    n_frequency = _bounded_int(specification["frequency_bins"], "frequency_bins", 4, 128)
    n_time = _bounded_int(specification["time_steps"], "time_steps", 2, 64)
    noise_scale = float(specification["noise_scale"])
    if not np.isfinite(noise_scale) or not 0.0 < noise_scale <= 1.0:
        raise ValueError("noise_scale must be finite and in (0, 1]")
    samples = n_dates * n_symbols
    cells = samples * n_channels * n_frequency * n_time
    if cells * np.dtype(np.float32).itemsize > config.vision.max_tensor_bytes:
        raise ValueError("synthetic reference exceeds the vision tensor budget")

    base_seed, noise_seed = np.random.SeedSequence(config.seed).spawn(2)
    base_generator = np.random.default_rng(base_seed)
    noise_generator = np.random.default_rng(noise_seed)
    dates = np.repeat(pd.bdate_range("2021-01-04", periods=n_dates).to_numpy(), n_symbols)
    symbols = np.tile(np.asarray([f"S{index:03d}" for index in range(n_symbols)]), n_dates)
    base = base_generator.lognormal(
        mean=-1.0,
        sigma=0.55,
        size=(samples, n_channels, n_frequency, n_time),
    ).astype(np.float32)
    date_cycle = np.repeat(
        np.sin(np.arange(n_dates, dtype=np.float64) * np.pi / 6.0),
        n_symbols,
    )
    symbol_loading = np.tile(
        np.linspace(-0.5, 0.5, n_symbols, dtype=np.float64),
        n_dates,
    )
    base[:, 0, :2, -1] += np.maximum(date_cycle + symbol_loading, 0.0)[:, None].astype(np.float32)
    if n_channels > 1:
        base[:, 1, -2:, :] += np.maximum(-date_cycle + symbol_loading, 0.0)[:, None, None].astype(
            np.float32
        )
    low_recent = base[:, 0, :2, -1].mean(axis=1)
    high_history = base[:, 0, -2:, :-1].mean(axis=(1, 2))
    channel_contrast = (
        base[:, 1].mean(axis=(1, 2)) if n_channels > 1 else base[:, 0].mean(axis=(1, 2))
    )
    signal = low_recent - 0.45 * high_history - 0.20 * channel_contrast
    signal = (signal - signal.mean()) / max(float(signal.std()), 1e-12)
    target = 0.01 * signal + noise_generator.normal(0.0, noise_scale * 0.01, samples)

    time_features = np.column_stack(
        [
            base.mean(axis=(1, 2, 3)),
            base.std(axis=(1, 2, 3)),
            base[:, :, :, -1].mean(axis=(1, 2)),
            date_cycle,
            symbol_loading,
        ]
    )
    spectral_features = _spectral_descriptors(base)
    frequency_values = tuple(np.linspace(0.0, 0.5, n_frequency, dtype=np.float64))
    return TimeFrequencyBatch(
        values=base,
        observed_mask=np.ones((samples, n_channels), dtype=bool),
        dates=dates,
        symbols=symbols.astype(object),
        target=target,
        time_features=time_features,
        spectral_features=spectral_features,
        channels=tuple(f"channel_{index}" for index in range(n_channels)),
        frequency_values=frequency_values,
        representation="spectrogram",
        max_tensor_bytes=config.vision.max_tensor_bytes,
    )


def _spectral_descriptors(values: np.ndarray) -> np.ndarray:
    power = np.asarray(values, dtype=np.float64)
    frequency = np.linspace(0.0, 1.0, power.shape[2], dtype=np.float64)
    total = np.maximum(power.sum(axis=(2, 3)), 1e-12)
    marginal = power.sum(axis=3)
    centroid = (marginal * frequency[None, None, :]).sum(axis=2) / total
    distribution = marginal / np.maximum(marginal.sum(axis=2, keepdims=True), 1e-12)
    entropy = -(distribution * np.log(np.maximum(distribution, 1e-12))).sum(axis=2)
    low = power[:, :, : max(1, power.shape[2] // 3), :].mean(axis=(2, 3))
    high = power[:, :, -max(1, power.shape[2] // 3) :, :].mean(axis=(2, 3))
    recent = power[:, :, :, -1].mean(axis=2)
    return np.concatenate([centroid, entropy, low, high, recent], axis=1)


def _chronological_split(
    batch: TimeFrequencyBatch,
    config: TimeFrequencyStudyConfig,
) -> _ChronologicalSplit:
    dates = pd.to_datetime(batch.dates)
    unique = np.sort(dates.unique())
    if len(unique) < 10:
        raise ValueError("time-frequency study needs at least ten distinct dates")
    train_count = int(np.floor(len(unique) * config.train_fraction))
    validation_count = int(np.floor(len(unique) * config.validation_fraction))
    test_count = len(unique) - train_count - validation_count
    if min(train_count, validation_count, test_count) < 2:
        raise ValueError("every chronological partition needs at least two dates")
    train_end = unique[train_count - 1]
    validation_end = unique[train_count + validation_count - 1]
    train = np.asarray(dates <= train_end)
    validation = np.asarray((dates > train_end) & (dates <= validation_end))
    test = np.asarray(dates > validation_end)
    complete = batch.complete_samples
    train &= complete
    validation &= complete
    test &= complete
    if min(int(train.sum()), int(validation.sum()), int(test.sum())) < 2:
        raise ValueError("observability filtering emptied a chronological partition")
    return _ChronologicalSplit(
        train=train,
        validation=validation,
        test=test,
        train_end=str(pd.Timestamp(train_end).date()),
        validation_end=str(pd.Timestamp(validation_end).date()),
        test_end=str(pd.Timestamp(unique[-1]).date()),
    )


def _fit_lightgbm_candidate(
    name: str,
    features: np.ndarray,
    batch: TimeFrequencyBatch,
    split: _ChronologicalSplit,
    params: dict[str, Any],
) -> _FittedCandidate:
    columns = [f"feature_{index:03d}" for index in range(features.shape[1])]
    frame = pd.DataFrame(features, columns=columns)
    target = pd.Series(batch.target, name="target")
    start_wall = time.perf_counter()
    start_cpu = time.process_time()
    model = make_lightgbm(**params).fit(frame.loc[split.train], target.loc[split.train])
    validation_prediction = model.predict(frame.loc[split.validation])
    resource = {
        "backend": model.training_diagnostics().backend,
        "fit_wall_seconds": time.perf_counter() - start_wall,
        "fit_cpu_seconds": time.process_time() - start_cpu,
        "parameter_count": None,
        "parameter_bytes": None,
        "epochs_completed": model.training_diagnostics().iterations,
        "best_epoch": None,
        "best_validation_loss": None,
        "stopped_early": None,
    }
    return _FittedCandidate(name, validation_prediction, resource, (model, frame))


def _fit_vision_candidate(
    name: str,
    batch: TimeFrequencyBatch,
    split: _ChronologicalSplit,
    config: VisionTrainingConfig,
    policy: ProgressionGatePolicy,
    prior_gate: ProgressionGateEvidence | None,
) -> _FittedCandidate:
    model = ControlledVisionModel(
        name,  # type: ignore[arg-type]
        config,
        policy=policy,
        prior_gate=prior_gate,
    ).fit(
        batch.values[split.train],
        batch.target[split.train],
        batch.values[split.validation],
        batch.target[split.validation],
    )
    validation_prediction = model.predict(batch.values[split.validation])
    evidence: VisionResourceEvidence = model.resource_evidence()
    return _FittedCandidate(name, validation_prediction, evidence.to_dict(), model)


def _candidate_metrics(
    candidate: _FittedCandidate,
    batch: TimeFrequencyBatch,
    split: _ChronologicalSplit,
) -> PredictionMetrics:
    return prediction_metrics(
        batch.target[split.validation],
        candidate.validation_prediction,
        batch.dates[split.validation],
        partition="validation",
    )


def _progression(
    batch: TimeFrequencyBatch,
    split: _ChronologicalSplit,
    config: TimeFrequencyStudyConfig,
) -> tuple[list[_FittedCandidate], tuple[ProgressionGateEvidence, ...], dict[str, str]]:
    time_candidate = _fit_lightgbm_candidate(
        "lightgbm_time", batch.time_features, batch, split, config.lightgbm
    )
    spectral_candidate = _fit_lightgbm_candidate(
        "lightgbm_spectral", batch.spectral_features, batch, split, config.lightgbm
    )
    small = _fit_vision_candidate(
        "small_cnn",
        batch,
        split,
        config.vision,
        config.small_cnn_gate,
        None,
    )
    candidates = [time_candidate, spectral_candidate, small]
    statuses = {name: "evaluated" for name in TIME_FREQUENCY_CANDIDATES[:3]}
    small_gate = evaluate_progression_gate(
        candidate="small_cnn",
        baseline="lightgbm_spectral",
        candidate_metrics=_candidate_metrics(small, batch, split),
        baseline_metrics=_candidate_metrics(spectral_candidate, batch, split),
        policy=config.small_cnn_gate,
    )
    gates = [small_gate]
    if not small_gate.passed:
        statuses["resnet"] = "blocked_by_small_cnn_gate"
        statuses["vit"] = "blocked_by_small_cnn_gate"
        return candidates, tuple(gates), statuses

    resnet = _fit_vision_candidate(
        "resnet",
        batch,
        split,
        config.vision,
        config.small_cnn_gate,
        small_gate,
    )
    candidates.append(resnet)
    statuses["resnet"] = "evaluated"
    resnet_gate = evaluate_progression_gate(
        candidate="resnet",
        baseline="small_cnn",
        candidate_metrics=_candidate_metrics(resnet, batch, split),
        baseline_metrics=_candidate_metrics(small, batch, split),
        policy=config.resnet_gate,
    )
    gates.append(resnet_gate)
    if not resnet_gate.passed:
        statuses["vit"] = "blocked_by_resnet_gate"
        return candidates, tuple(gates), statuses
    vit = _fit_vision_candidate(
        "vit",
        batch,
        split,
        config.vision,
        config.resnet_gate,
        resnet_gate,
    )
    candidates.append(vit)
    statuses["vit"] = "evaluated"
    return candidates, tuple(gates), statuses


def _test_prediction(
    candidate: _FittedCandidate,
    batch: TimeFrequencyBatch,
    split: _ChronologicalSplit,
) -> np.ndarray:
    if candidate.name.startswith("lightgbm_"):
        model, frame = candidate.model
        return np.asarray(model.predict(frame.loc[split.test]), dtype=np.float64)
    return np.asarray(candidate.model.predict(batch.values[split.test]), dtype=np.float64)


def _summary(
    candidates: list[_FittedCandidate],
    statuses: dict[str, str],
    batch: TimeFrequencyBatch,
    split: _ChronologicalSplit,
    config: TimeFrequencyStudyConfig,
) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    by_name = {candidate.name: candidate for candidate in candidates}
    for name in TIME_FREQUENCY_CANDIDATES:
        status = statuses.get(name, "blocked")
        if name not in by_name:
            records.append({"model": name, "status": status})
            continue
        candidate = by_name[name]
        validation = _candidate_metrics(candidate, batch, split)
        prediction = _test_prediction(candidate, batch, split)
        test = prediction_metrics(
            batch.target[split.test],
            prediction,
            batch.dates[split.test],
            partition="test",
        )
        panel = pd.DataFrame(
            {
                "date": pd.to_datetime(batch.dates[split.test]),
                "symbol": batch.symbols[split.test],
                "target": batch.target[split.test],
                "prediction": prediction,
            }
        )
        costed = evaluate_oos_predictions(
            panel,
            model=name,
            transaction_cost_bps=config.transaction_cost_bps,
            selection_fraction=config.selection_fraction,
        )
        records.append(
            {
                "model": name,
                "status": status,
                "validation_observations": validation.observations,
                "validation_dates": validation.dates,
                "validation_rank_ic": validation.rank_ic,
                "validation_rmse": validation.rmse,
                "test_observations": test.observations,
                "test_dates": test.dates,
                "test_rank_ic": test.rank_ic,
                "test_rmse": test.rmse,
                "test_mae": test.mae,
                "test_rank_ic_standard_error": _rank_ic_standard_error(panel),
                "net_mean_daily_return": costed.net_mean_daily_return,
                "mean_daily_turnover": costed.mean_daily_turnover,
                **candidate.resource,
            }
        )
    return pd.DataFrame.from_records(records)


def _rank_ic_standard_error(panel: pd.DataFrame) -> float:
    values: list[float] = []
    for _, group in panel.groupby("date", sort=True):
        if group["target"].nunique() > 1 and group["prediction"].nunique() > 1:
            value = group["target"].rank().corr(group["prediction"].rank())
            if np.isfinite(value):
                values.append(float(value))
    if len(values) < 2:
        return 0.0
    return float(np.std(values, ddof=1) / np.sqrt(len(values)))


def _plot_summary(summary: pd.DataFrame, output: Path, *, seed: int) -> None:
    evaluated = summary.loc[summary["status"].eq("evaluated")].copy()
    evaluated["display_model"] = evaluated["model"].map(
        {
            "lightgbm_time": "LightGBM · time",
            "lightgbm_spectral": "LightGBM · spectral",
            "small_cnn": "Small CNN",
            "resnet": "ResNet",
            "vit": "Vision Transformer",
        }
    )
    sns.set_theme(style="whitegrid", context="talk", palette="colorblind")
    figure, axes = plt.subplots(1, 3, figsize=(20, 6))
    sns.barplot(
        data=evaluated,
        x="test_rank_ic",
        y="display_model",
        ax=axes[0],
        color=sns.color_palette("colorblind")[0],
    )
    axes[0].errorbar(
        evaluated["test_rank_ic"],
        np.arange(len(evaluated)),
        xerr=1.96 * evaluated["test_rank_ic_standard_error"],
        fmt="none",
        ecolor="black",
        capsize=4,
        linewidth=1.5,
        label="±1.96 daily-rank-IC SE",
    )
    axes[0].axvline(0.0, color="black", linewidth=1)
    axes[0].set(title="Untouched synthetic test rank IC", xlabel="Mean daily rank IC", ylabel="")
    axes[0].legend(loc="lower right", fontsize=10)
    sns.barplot(
        data=evaluated,
        x="test_rmse",
        y="display_model",
        ax=axes[1],
        color=sns.color_palette("colorblind")[1],
    )
    axes[1].set(title="Untouched synthetic test error", xlabel="RMSE", ylabel="")
    resources = evaluated.dropna(subset=["fit_wall_seconds"])
    sns.barplot(
        data=resources,
        x="fit_wall_seconds",
        y="display_model",
        ax=axes[2],
        color=sns.color_palette("colorblind")[2],
    )
    axes[2].set(title="Measured fit wall time", xlabel="Seconds on reference host", ylabel="")
    blocked = ", ".join(
        f"{row.model}: {row.status.replace('_', ' ')}"
        for row in summary.loc[~summary["status"].eq("evaluated")].itertuples()
    )
    suffix = f" Blocked: {blocked}." if blocked else ""
    test_rows = int(evaluated["test_observations"].iloc[0])
    test_dates = int(evaluated["test_dates"].iloc[0])
    figure.suptitle(
        "Gated time-frequency progression — synthetic engineering evidence only "
        f"(test n={test_rows}, dates={test_dates}, seed={seed})." + suffix,
        fontsize=15,
    )
    figure.tight_layout()
    figure.savefig(output, dpi=160, bbox_inches="tight")
    plt.close(figure)


def run_time_frequency_study(
    batch: TimeFrequencyBatch,
    config: TimeFrequencyStudyConfig,
    output_dir: str | Path,
) -> TimeFrequencyStudyResult:
    """Run the frozen progression and atomically publish aggregate evidence."""
    destination = Path(output_dir)
    if destination.exists():
        raise FileExistsError(f"study destination already exists: {destination}")
    staging = destination.parent / f".publishing-{destination.name}"
    if staging.exists():
        raise FileExistsError(f"stale study staging exists: {staging}")
    staging.mkdir(parents=True)
    try:
        split = _chronological_split(batch, config)
        candidates, gates, statuses = _progression(batch, split, config)
        # Test data is first accessed inside `_summary`, after `_progression`
        # has finalized all architecture gates.
        summary = _summary(candidates, statuses, batch, split, config)
        summary.to_csv(staging / "model_summary.csv", index=False)
        _plot_summary(summary, staging / "model_comparison.png", seed=config.seed)
        metadata = {
            "schema_version": STUDY_SCHEMA_VERSION,
            "study": {
                "candidate_order": list(TIME_FREQUENCY_CANDIDATES),
                "seed": config.seed,
                "interpretation": config.interpretation,
                "transaction_cost_bps": config.transaction_cost_bps,
                "selection_fraction": config.selection_fraction,
                "small_cnn_gate_policy_id": config.small_cnn_gate.identity,
                "resnet_gate_policy_id": config.resnet_gate.identity,
            },
            "data": {
                "kind": "deterministic_synthetic_engineering_reference",
                "samples": int(len(batch.target)),
                "dates": int(pd.Series(batch.dates).nunique()),
                "symbols": int(pd.Series(batch.symbols).nunique()),
                "shape": list(batch.values.shape),
                "channels": list(batch.channels),
                "representation": batch.representation,
                "raw_observations_published": 0,
            },
            "split": asdict(split)
            | {
                "train": int(split.train.sum()),
                "validation": int(split.validation.sum()),
                "test": int(split.test.sum()),
            },
            "gates": [gate.payload() | {"evidence_sha256": gate.evidence_sha256} for gate in gates],
            "statuses": statuses,
            "environment": _runtime_environment(),
            "evidence": {
                "aggregate_rows": len(summary),
                "summary": "model_summary.csv",
                "plot": "model_comparison.png",
                "prediction_rows_published": 0,
            },
            "limitations": [
                "Synthetic engineering evidence is not evidence of market predictability.",
                "No licensed observations, model weights, tensors, or row predictions are published.",
                "The simplified costed diagnostic is not the event-driven execution simulator.",
                "No result authorizes paper or live trading or guarantees profit.",
                "Licensed point-in-time evaluation remains a later governed study.",
            ],
        }
        _atomic_json(staging / "summary.json", metadata)
        os.replace(staging, destination)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return TimeFrequencyStudyResult(destination, summary, gates, metadata)


def run_synthetic_time_frequency_study(
    config: TimeFrequencyStudyConfig,
    output_dir: str | Path,
) -> TimeFrequencyStudyResult:
    """Build and run the deterministic offline engineering reference."""
    return run_time_frequency_study(
        build_synthetic_time_frequency_reference(config),
        config,
        output_dir,
    )


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
        "seaborn": sns.__version__,
        "torch": _installed_version("torch"),
        "lightgbm": _installed_version("lightgbm"),
        "reference_device": "cpu",
        "measurement_samples_per_fit": 1,
        "warmup_runs": 0,
    }


def _installed_version(distribution: str) -> str:
    try:
        return version(distribution)
    except PackageNotFoundError:
        return "unavailable"


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    encoded = json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=True, allow_nan=False)
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


def _mapping(value: Any, name: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be a mapping")
    return dict(value)


def _exact_fields(value: dict[str, Any], expected: frozenset[str], name: str) -> None:
    missing = sorted(expected - set(value))
    unknown = sorted(set(value) - expected)
    if missing or unknown:
        raise ValueError(f"{name} fields mismatch: missing={missing}, unknown={unknown}")


def _bounded_int(value: Any, name: str, minimum: int, maximum: int) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or not minimum <= value <= maximum:
        raise ValueError(f"{name} must be an integer in [{minimum}, {maximum}]")
    return value
