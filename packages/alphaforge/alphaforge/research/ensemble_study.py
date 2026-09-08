"""Deterministic aggregate engineering evidence for governed ensembles.

The reference is deliberately synthetic and offline.  Row predictions and
targets exist only in memory; the publisher emits aggregate diagnostics,
stable fitted-state identities, and one Seaborn figure.  Nothing in this
module is market evidence or authorization for paper/live trading.
"""

from __future__ import annotations

import hashlib
import itertools
import json
import os
import platform
import shutil
import tempfile
from dataclasses import asdict, dataclass
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from alphaforge.evaluation.metric_suite import MetricSuiteConfig, evaluate_metric_suite
from alphaforge.evaluation.uncertainty import (
    BlockBootstrapConfig,
    block_bootstrap_interval,
)
from alphaforge.models.ensemble import (
    GovernedEnsembleConfig,
    GovernedEnsembleState,
    fit_governed_ensemble,
)
from alphaforge.models.ensemble_contracts import (
    EnsembleDecision,
    InferenceBatch,
    TemporalOOFFold,
    TrainingOOFPanel,
)
from alphaforge.visualization.ensemble_plots import plot_ensemble_evidence

ENSEMBLE_STUDY_VERSION = "1.0.0"
REFERENCE_EXPERTS = ("defensive", "redundant", "stable", "trend")
ENSEMBLE_METHODS = (
    "static",
    "rank_vote",
    "stacking",
    "bayesian",
    "dynamic",
    "regime_gate",
)

_ROOT_FIELDS = frozenset({"version", "study", "resources", "policies", "bootstrap"})
_STUDY_FIELDS = frozenset(
    {
        "seed",
        "training_dates",
        "holdout_dates",
        "symbols",
        "folds",
        "transaction_cost_bps",
        "interpretation",
    }
)
_RESOURCE_FIELDS = frozenset({"max_prediction_records", "max_audit_records"})
_BOOTSTRAP_FIELDS = frozenset(
    {"n_resamples", "block_length", "confidence_level", "seed", "circular"}
)
_POLICY_FIELDS = frozenset(
    {
        "ridge_penalty",
        "bayesian_temperature",
        "dynamic_decay",
        "dynamic_temperature",
        "min_weight",
        "regime_threshold",
        "regime_min_confidence",
        "min_regime_rows",
        "fallback_prediction",
    }
)


@dataclass(frozen=True)
class EnsembleStudyConfig:
    """Frozen deterministic reference-study configuration."""

    seed: int
    training_dates: int
    holdout_dates: int
    symbols: int
    folds: int
    transaction_cost_bps: float
    interpretation: str
    max_prediction_records: int
    max_audit_records: int
    bootstrap_resamples: int
    bootstrap_block_length: int
    bootstrap_confidence_level: float
    bootstrap_seed: int
    bootstrap_circular: bool
    ridge_penalty: float
    bayesian_temperature: float
    dynamic_decay: float
    dynamic_temperature: float
    min_weight: float
    regime_threshold: float
    regime_min_confidence: float
    min_regime_rows: int
    fallback_prediction: float

    def __post_init__(self) -> None:
        integer_fields = (
            ("seed", self.seed, 0, 2**32 - 1),
            ("training_dates", self.training_dates, 32, 2_000),
            ("holdout_dates", self.holdout_dates, 20, 1_000),
            ("symbols", self.symbols, 4, 500),
            ("folds", self.folds, 2, 20),
            ("max_prediction_records", self.max_prediction_records, 1, 10_000_000),
            ("max_audit_records", self.max_audit_records, 1, 100_000),
            ("bootstrap_resamples", self.bootstrap_resamples, 100, 100_000),
            ("bootstrap_seed", self.bootstrap_seed, 0, 2**32 - 1),
        )
        for integer_field, integer_value, integer_minimum, integer_maximum in integer_fields:
            object.__setattr__(
                self,
                integer_field,
                _integer(
                    integer_value,
                    integer_field,
                    integer_minimum,
                    integer_maximum,
                ),
            )
        if self.training_dates % self.folds:
            raise ValueError("training_dates must be divisible by folds")
        object.__setattr__(
            self,
            "bootstrap_block_length",
            _integer(
                self.bootstrap_block_length,
                "bootstrap_block_length",
                2,
                20,
            ),
        )
        if self.interpretation != "synthetic_engineering_only":
            raise ValueError("interpretation must be 'synthetic_engineering_only'")
        required_records = (
            (self.training_dates + self.holdout_dates) * self.symbols * len(REFERENCE_EXPERTS)
        )
        if self.max_prediction_records < required_records:
            raise ValueError("max_prediction_records cannot hold the reference panel")
        if self.max_audit_records < self.training_dates:
            raise ValueError("max_audit_records must cover every training date")
        if not isinstance(self.bootstrap_circular, bool):
            raise ValueError("bootstrap_circular must be a boolean")
        object.__setattr__(
            self,
            "min_regime_rows",
            _integer(
                self.min_regime_rows,
                "min_regime_rows",
                2,
                self.training_dates * self.symbols,
            ),
        )
        finite_fields = (
            ("transaction_cost_bps", self.transaction_cost_bps, 0.0, 1_000.0),
            ("ridge_penalty", self.ridge_penalty, 1e-12, 1e9),
            ("bayesian_temperature", self.bayesian_temperature, 1e-6, 1e6),
            ("dynamic_decay", self.dynamic_decay, 0.0, 0.999999),
            ("dynamic_temperature", self.dynamic_temperature, 1e-6, 1e6),
            ("min_weight", self.min_weight, 0.0, 0.2),
            ("regime_threshold", self.regime_threshold, 0.01, 0.99),
            ("regime_min_confidence", self.regime_min_confidence, 0.0, 0.49),
            ("fallback_prediction", self.fallback_prediction, -1.0, 1.0),
            (
                "bootstrap_confidence_level",
                self.bootstrap_confidence_level,
                0.500001,
                0.999999,
            ),
        )
        for finite_field, finite_value, finite_minimum, finite_maximum in finite_fields:
            object.__setattr__(
                self,
                finite_field,
                _finite(
                    finite_value,
                    finite_field,
                    finite_minimum,
                    finite_maximum,
                ),
            )

    @property
    def identity(self) -> str:
        payload = json.dumps(asdict(self), sort_keys=True, separators=(",", ":"), allow_nan=False)
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class EnsembleStudyResult:
    """Published aggregate evidence and in-memory summary."""

    output_dir: Path
    summary: pd.DataFrame
    metadata: dict[str, Any]


@dataclass(frozen=True)
class _SyntheticReference:
    panel: TrainingOOFPanel
    inference: InferenceBatch
    training_frame: pd.DataFrame
    inference_frame: pd.DataFrame
    outcomes: pd.DataFrame


def _exact_fields(mapping: dict[str, Any], expected: frozenset[str], name: str) -> None:
    missing = sorted(expected - set(mapping))
    unknown = sorted(set(mapping) - expected)
    if missing or unknown:
        raise ValueError(f"{name} fields mismatch: missing={missing}, unknown={unknown}")


def _mapping(value: Any, name: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be a mapping")
    return dict(value)


def _integer(value: Any, name: str, minimum: int, maximum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or not minimum <= value <= maximum:
        raise ValueError(f"{name} must be an integer in [{minimum}, {maximum}]")
    return value


def _finite(value: Any, name: str, minimum: float, maximum: float) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be numeric")
    result = float(value)
    if not np.isfinite(result) or not minimum <= result <= maximum:
        raise ValueError(f"{name} must be finite and in [{minimum}, {maximum}]")
    return result


def load_ensemble_study_config(path: str | Path) -> EnsembleStudyConfig:
    """Load the exact-schema SF-S3-MR9 reference configuration."""

    source = Path(path)
    try:
        payload = yaml.safe_load(source.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, yaml.YAMLError) as exc:
        raise ValueError(f"could not parse ensemble study configuration {source}") from exc
    root = _mapping(payload, "configuration")
    _exact_fields(root, _ROOT_FIELDS, "configuration")
    if root["version"] != ENSEMBLE_STUDY_VERSION:
        raise ValueError("unsupported ensemble study version")
    study = _mapping(root["study"], "study")
    resources = _mapping(root["resources"], "resources")
    policies = _mapping(root["policies"], "policies")
    bootstrap = _mapping(root["bootstrap"], "bootstrap")
    _exact_fields(study, _STUDY_FIELDS, "study")
    _exact_fields(resources, _RESOURCE_FIELDS, "resources")
    _exact_fields(policies, _POLICY_FIELDS, "policies")
    _exact_fields(bootstrap, _BOOTSTRAP_FIELDS, "bootstrap")

    seed = _integer(study["seed"], "study.seed", 0, 2**32 - 1)
    training_dates = _integer(study["training_dates"], "study.training_dates", 32, 2_000)
    holdout_dates = _integer(study["holdout_dates"], "study.holdout_dates", 20, 1_000)
    symbols = _integer(study["symbols"], "study.symbols", 4, 500)
    folds = _integer(study["folds"], "study.folds", 2, 20)
    if training_dates % folds:
        raise ValueError("study.training_dates must be divisible by study.folds")
    interpretation = study["interpretation"]
    if interpretation != "synthetic_engineering_only":
        raise ValueError("study.interpretation must be 'synthetic_engineering_only'")
    max_records = _integer(
        resources["max_prediction_records"],
        "resources.max_prediction_records",
        1,
        10_000_000,
    )
    required_records = (training_dates + holdout_dates) * symbols * len(REFERENCE_EXPERTS)
    if max_records < required_records:
        raise ValueError("resources.max_prediction_records cannot hold the reference panel")
    max_audits = _integer(
        resources["max_audit_records"],
        "resources.max_audit_records",
        training_dates,
        100_000,
    )
    min_regime_rows = _integer(
        policies["min_regime_rows"],
        "policies.min_regime_rows",
        2,
        training_dates * symbols,
    )
    return EnsembleStudyConfig(
        seed=seed,
        training_dates=training_dates,
        holdout_dates=holdout_dates,
        symbols=symbols,
        folds=folds,
        transaction_cost_bps=_finite(
            study["transaction_cost_bps"], "study.transaction_cost_bps", 0.0, 1_000.0
        ),
        interpretation=interpretation,
        max_prediction_records=max_records,
        max_audit_records=max_audits,
        bootstrap_resamples=_integer(
            bootstrap["n_resamples"],
            "bootstrap.n_resamples",
            100,
            100_000,
        ),
        bootstrap_block_length=_integer(
            bootstrap["block_length"],
            "bootstrap.block_length",
            2,
            20,
        ),
        bootstrap_confidence_level=_finite(
            bootstrap["confidence_level"],
            "bootstrap.confidence_level",
            0.500001,
            0.999999,
        ),
        bootstrap_seed=_integer(
            bootstrap["seed"],
            "bootstrap.seed",
            0,
            2**32 - 1,
        ),
        bootstrap_circular=bootstrap["circular"],
        ridge_penalty=_finite(policies["ridge_penalty"], "policies.ridge_penalty", 1e-12, 1e9),
        bayesian_temperature=_finite(
            policies["bayesian_temperature"],
            "policies.bayesian_temperature",
            1e-6,
            1e6,
        ),
        dynamic_decay=_finite(policies["dynamic_decay"], "policies.dynamic_decay", 0.0, 0.999999),
        dynamic_temperature=_finite(
            policies["dynamic_temperature"],
            "policies.dynamic_temperature",
            1e-6,
            1e6,
        ),
        min_weight=_finite(policies["min_weight"], "policies.min_weight", 0.0, 0.2),
        regime_threshold=_finite(
            policies["regime_threshold"], "policies.regime_threshold", 0.01, 0.99
        ),
        regime_min_confidence=_finite(
            policies["regime_min_confidence"],
            "policies.regime_min_confidence",
            0.0,
            0.49,
        ),
        min_regime_rows=min_regime_rows,
        fallback_prediction=_finite(
            policies["fallback_prediction"],
            "policies.fallback_prediction",
            -1.0,
            1.0,
        ),
    )


def _named_child_seed(seed: int, name: str) -> int:
    digest = hashlib.sha256(f"{seed}:{name}".encode("ascii")).digest()
    return int.from_bytes(digest[:8], byteorder="big", signed=False)


def _named_generator(seed: int, name: str) -> np.random.Generator:
    return np.random.default_rng(_named_child_seed(seed, name))


def _build_synthetic_reference(config: EnsembleStudyConfig) -> _SyntheticReference:
    dates = pd.bdate_range(
        "2021-01-04",
        periods=config.training_dates + config.holdout_dates,
    )
    symbols = tuple(f"S{index:03d}" for index in range(config.symbols))
    target_rng = _named_generator(config.seed, "target")
    expert_rngs = {
        expert: _named_generator(config.seed, f"expert:{expert}") for expert in REFERENCE_EXPERTS
    }
    rows: list[dict[str, Any]] = []
    outcomes: list[dict[str, Any]] = []
    inference_rows: list[dict[str, Any]] = []
    fold_width = config.training_dates // config.folds
    for date_index, date_value in enumerate(dates):
        phase = np.sin(2.0 * np.pi * date_index / 37.0)
        stress = phase < -0.15
        regime_probability = 0.85 if stress else 0.15
        # Sparse ambiguous gates deliberately exercise fail-closed abstention.
        if date_index % 19 == 0:
            regime_probability = 0.5
        for symbol_index, symbol in enumerate(symbols):
            cross_section = (symbol_index - (config.symbols - 1) / 2.0) / config.symbols
            latent = (
                0.006 * phase
                + 0.005 * cross_section
                + (0.003 if stress else -0.001) * np.cos(symbol_index)
            )
            target = latent + target_rng.normal(0.0, 0.0035)
            expert_spec = {
                "trend": (0.0025 if not stress else 0.0090, 0.0005 if stress else 0.0),
                "defensive": (0.0090 if not stress else 0.0023, -0.0004 if not stress else 0.0),
                "stable": (0.0048, 0.0),
                "redundant": (0.0028 if not stress else 0.0092, 0.0002),
            }
            independent_predictions: dict[str, tuple[float, float]] = {}
            for expert in REFERENCE_EXPERTS:
                noise_scale, bias = expert_spec[expert]
                prediction = latent + bias + expert_rngs[expert].normal(0.0, noise_scale)
                independent_predictions[expert] = (float(prediction), noise_scale)
            redundant_prediction, redundant_scale = independent_predictions["redundant"]
            predictions = {
                **independent_predictions,
                "redundant": (
                    0.93 * independent_predictions["defensive"][0] + 0.07 * redundant_prediction,
                    redundant_scale,
                ),
            }
            if date_index < config.training_dates:
                fold_id = date_index // fold_width
                for expert in REFERENCE_EXPERTS:
                    prediction, uncertainty = predictions[expert]
                    rows.append(
                        {
                            "date": date_value,
                            "symbol": symbol,
                            "fold_id": fold_id,
                            "expert": expert,
                            "prediction": prediction,
                            "target": target,
                            "uncertainty": uncertainty,
                            "regime_probability": regime_probability,
                        }
                    )
            else:
                outcomes.append(
                    {
                        "date": date_value.date().isoformat(),
                        "symbol": symbol,
                        "target": target,
                        "regime_probability": regime_probability,
                    }
                )
                for expert in REFERENCE_EXPERTS:
                    prediction, uncertainty = predictions[expert]
                    inference_rows.append(
                        {
                            "date": date_value,
                            "symbol": symbol,
                            "expert": expert,
                            "prediction": prediction,
                            "uncertainty": uncertainty,
                            "regime_probability": regime_probability,
                        }
                    )
    training = pd.DataFrame(rows)
    inference_frame = pd.DataFrame(inference_rows)
    holdout_start = dates[config.training_dates].date().isoformat()
    folds = tuple(
        TemporalOOFFold(
            fold_id=fold_id,
            training_end=(dates[fold_id * fold_width] - pd.offsets.BDay(5)).date().isoformat(),
            validation_start=dates[fold_id * fold_width].date().isoformat(),
            validation_end=dates[(fold_id + 1) * fold_width - 1].date().isoformat(),
            embargo_days=5,
        )
        for fold_id in range(config.folds)
    )
    panel = TrainingOOFPanel.from_frame(
        training,
        folds=folds,
        expected_experts=REFERENCE_EXPERTS,
        holdout_start=holdout_start,
        source_id=f"ensemble-reference-{config.identity[:16]}",
        max_records=config.max_prediction_records,
    )
    expected_inference_keys = tuple(
        (date_value.date().isoformat(), symbol)
        for date_value in dates[config.training_dates :]
        for symbol in symbols
    )
    inference = InferenceBatch.from_frame(
        inference_frame,
        expected_keys=expected_inference_keys,
        max_records=config.max_prediction_records,
    )
    return _SyntheticReference(
        panel=panel,
        inference=inference,
        training_frame=training,
        inference_frame=inference_frame,
        outcomes=pd.DataFrame(outcomes),
    )


def _policy_config(
    config: EnsembleStudyConfig,
    method: str,
    experts: tuple[str, ...],
) -> GovernedEnsembleConfig:
    return GovernedEnsembleConfig(
        method=method,  # type: ignore[arg-type]
        experts=experts,
        ridge_penalty=config.ridge_penalty,
        bayesian_temperature=config.bayesian_temperature,
        dynamic_decay=config.dynamic_decay,
        dynamic_temperature=config.dynamic_temperature,
        min_weight=config.min_weight,
        regime_threshold=config.regime_threshold,
        regime_min_confidence=config.regime_min_confidence,
        min_regime_rows=config.min_regime_rows,
        fallback_prediction=config.fallback_prediction,
        max_audit_records=config.max_audit_records,
    )


def _decision_frame(decisions: tuple[EnsembleDecision, ...], model: str) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "date": [decision.date for decision in decisions],
            "symbol": [decision.symbol for decision in decisions],
            "model": model,
            "prediction": [decision.prediction for decision in decisions],
            "uncertainty": [decision.uncertainty for decision in decisions],
            "status": [decision.status for decision in decisions],
        }
    )


def _single_expert_frame(reference: _SyntheticReference, expert: str) -> pd.DataFrame:
    return (
        reference.inference_frame.loc[
            reference.inference_frame["expert"] == expert,
            ["date", "symbol", "prediction", "uncertainty"],
        ]
        .assign(
            date=lambda frame: pd.to_datetime(frame["date"]).dt.date.astype(str),
            model=expert,
            status="combined",
        )[["date", "symbol", "model", "prediction", "uncertainty", "status"]]
        .reset_index(drop=True)
    )


def _portfolio_diagnostic(
    frame: pd.DataFrame,
    *,
    transaction_cost_bps: float,
) -> tuple[float, float, float, float, pd.DataFrame]:
    ordered = frame.sort_values(["date", "symbol"]).copy()
    denominator = ordered.groupby("date")["prediction"].transform(
        lambda values: max(float(values.abs().sum()), 1e-12)
    )
    ordered["weight"] = ordered["prediction"] / denominator
    weights = ordered.pivot(index="date", columns="symbol", values="weight").fillna(0.0)
    turnover = 0.5 * weights.diff().abs().sum(axis=1)
    turnover.iloc[0] = 0.5 * weights.iloc[0].abs().sum()
    ordered["gross_contribution"] = ordered["weight"] * ordered["target"]
    gross = ordered.groupby("date")["gross_contribution"].sum()
    cost = turnover * transaction_cost_bps / 10_000.0
    daily = pd.DataFrame(
        {
            "date": gross.index.astype(str),
            "return": (gross - cost).to_numpy(dtype=float),
        }
    )
    return (
        float(turnover.mean()),
        float(cost.mean()),
        float(gross.mean()),
        float((gross - cost).mean()),
        daily,
    )


def _evaluate_prediction_frames(
    predictions: dict[str, pd.DataFrame],
    outcomes: pd.DataFrame,
    config: EnsembleStudyConfig,
) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    joined: dict[str, pd.DataFrame] = {}
    provisional: list[dict[str, Any]] = []
    bootstrap = BlockBootstrapConfig(
        n_resamples=config.bootstrap_resamples,
        block_length=config.bootstrap_block_length,
        confidence_level=config.bootstrap_confidence_level,
        seed=config.bootstrap_seed,
        circular=config.bootstrap_circular,
    )
    metric_config = MetricSuiteConfig(
        bootstrap=bootstrap,
        minimum_prediction_samples=20,
        minimum_trading_periods=20,
        benchmark_name="none_synthetic_reference",
    )
    for model in sorted(predictions):
        frame = predictions[model].merge(
            outcomes,
            on=["date", "symbol"],
            how="inner",
            validate="one_to_one",
        )
        if len(frame) != len(outcomes):
            raise RuntimeError(f"{model} did not cover the complete synthetic holdout")
        frame["error"] = frame["prediction"] - frame["target"]
        joined[model] = frame
        turnover, cost, gross, net, daily_returns = _portfolio_diagnostic(
            frame,
            transaction_cost_bps=config.transaction_cost_bps,
        )
        metric_report = evaluate_metric_suite(
            frame[["date", "target", "prediction"]].sort_values(["date"]),
            daily_returns,
            metric_config,
        )
        estimates = metric_report.by_name()
        mse_estimate = estimates["mse"]
        mae_estimate = estimates["mae"]
        rank_estimate = estimates["rank_ic"]
        if (
            mse_estimate.status != "ok"
            or mse_estimate.value is None
            or mse_estimate.uncertainty is None
            or mae_estimate.status != "ok"
            or mae_estimate.value is None
        ):
            raise RuntimeError(f"{model} predictive metrics were unexpectedly undefined")
        mse_distribution = mse_estimate.uncertainty
        if rank_estimate.status == "ok":
            if rank_estimate.value is None or rank_estimate.uncertainty is None:
                raise RuntimeError(f"{model} rank IC is missing defined uncertainty")
            rank_value = rank_estimate.value
            rank_distribution = rank_estimate.uncertainty
            rank_status = "defined"
            rank_lower = rank_distribution.lower
            rank_upper = rank_distribution.upper
            rank_standard_error = rank_distribution.standard_error
            rank_variance = rank_distribution.variance
        else:
            rank_value = float("nan")
            rank_status = f"undefined:{rank_estimate.note}"
            rank_lower = float("nan")
            rank_upper = float("nan")
            rank_standard_error = float("nan")
            rank_variance = float("nan")
        net_distribution = block_bootstrap_interval(
            daily_returns["return"].to_numpy(dtype=float),
            bootstrap,
        )
        uncertainty = pd.to_numeric(frame["uncertainty"], errors="coerce")
        available = uncertainty.notna()
        coverage = (
            float((frame.loc[available, "error"].abs() <= uncertainty[available]).mean())
            if available.any()
            else float("nan")
        )
        dispersion_rows = int(available.sum())
        if dispersion_rows >= 3 and uncertainty[available].std() > 0:
            dispersion_error_correlation = float(
                uncertainty[available].corr(frame.loc[available, "error"].abs())
            )
            dispersion_correlation_status = "defined"
        else:
            dispersion_error_correlation = float("nan")
            dispersion_correlation_status = "undefined_constant_or_insufficient_dispersion"
        provisional.append(
            {
                "model": model,
                "kind": "single" if model in REFERENCE_EXPERTS else "ensemble",
                "rows": len(frame),
                "mse": mse_estimate.value,
                "mse_ci_lower": mse_distribution.lower,
                "mse_ci_upper": mse_distribution.upper,
                "mse_standard_error": mse_distribution.standard_error,
                "mse_variance": mse_distribution.variance,
                "mae": mae_estimate.value,
                "rank_ic": rank_value,
                "rank_ic_status": rank_status,
                "rank_ic_valid_dates": rank_estimate.n_observations,
                "rank_ic_total_dates": int(frame["date"].nunique()),
                "rank_ic_ci_lower": rank_lower,
                "rank_ic_ci_upper": rank_upper,
                "rank_ic_standard_error": rank_standard_error,
                "rank_ic_variance": rank_variance,
                "bootstrap_resamples": bootstrap.n_resamples,
                "bootstrap_block_dates": bootstrap.block_length,
                "bootstrap_confidence_level": bootstrap.confidence_level,
                "bootstrap_seed": bootstrap.seed,
                "bootstrap_circular": bootstrap.circular,
                "fallback_rate": float((frame["status"] != "combined").mean()),
                "mean_heuristic_dispersion": (
                    float(uncertainty[available].mean()) if available.any() else float("nan")
                ),
                "heuristic_interval_coverage": coverage,
                "dispersion_absolute_error_correlation": dispersion_error_correlation,
                "dispersion_correlation_status": dispersion_correlation_status,
                "dispersion_rows": dispersion_rows,
                "mean_turnover": turnover,
                "mean_cost_drag": cost,
                "gross_directional_return": gross,
                "net_directional_return": net,
                "net_directional_return_ci_lower": net_distribution.lower,
                "net_directional_return_ci_upper": net_distribution.upper,
                "net_directional_return_standard_error": net_distribution.standard_error,
                "net_directional_return_variance": net_distribution.standard_error**2,
            }
        )
    summary = pd.DataFrame(provisional)
    single_rows = summary.loc[summary["kind"] == "single"].sort_values(["mse", "model"])
    best_single_model = str(single_rows.iloc[0]["model"])
    best_single = float(single_rows.iloc[0]["mse"])
    baseline_loss = (
        joined[best_single_model]
        .assign(squared_error=lambda frame: frame["error"] ** 2)
        .groupby("date", sort=True, as_index=False)["squared_error"]
        .mean()
        .rename(columns={"squared_error": "baseline_squared_error"})
    )
    paired_intervals: dict[str, Any] = {}
    for model, frame in joined.items():
        daily_loss = (
            frame.assign(squared_error=lambda values: values["error"] ** 2)
            .groupby("date", sort=True, as_index=False)["squared_error"]
            .mean()
        )
        paired = daily_loss.merge(
            baseline_loss,
            on="date",
            how="inner",
            validate="one_to_one",
        )
        if len(paired) != config.holdout_dates:
            raise RuntimeError(f"{model} could not align paired daily losses")
        paired_intervals[model] = block_bootstrap_interval(
            (paired["squared_error"] - paired["baseline_squared_error"]).to_numpy(dtype=float),
            bootstrap,
        )
    summary["best_single_model"] = best_single_model
    summary["best_single_mse"] = best_single
    summary["mse_delta_vs_best_single"] = summary["mse"] - best_single
    summary["mse_delta_ci_lower"] = summary["model"].map(
        lambda model: paired_intervals[str(model)].lower
    )
    summary["mse_delta_ci_upper"] = summary["model"].map(
        lambda model: paired_intervals[str(model)].upper
    )
    summary["mse_delta_standard_error"] = summary["model"].map(
        lambda model: paired_intervals[str(model)].standard_error
    )
    summary["mse_delta_variance"] = summary["mse_delta_standard_error"] ** 2
    for row in summary.itertuples(index=False):
        interval = paired_intervals[row.model]
        if not np.isclose(interval.estimate, row.mse_delta_vs_best_single, rtol=1e-10, atol=1e-15):
            raise RuntimeError(f"{row.model} paired MSE-delta replay mismatch")
    return summary.sort_values(["kind", "model"]).reset_index(drop=True), joined


def _correlations(joined: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for first in sorted(joined):
        left = joined[first].sort_values(["date", "symbol"])
        for second in sorted(joined):
            right = joined[second].sort_values(["date", "symbol"])
            for kind, column in (("prediction", "prediction"), ("error", "error")):
                correlation = float(left[column].corr(right[column]))
                rows.append(
                    {
                        "kind": kind,
                        "model_a": first,
                        "model_b": second,
                        "correlation": correlation,
                        "correlation_status": (
                            "defined"
                            if np.isfinite(correlation)
                            else "undefined_constant_or_insufficient_variation"
                        ),
                        "rows": len(left),
                    }
                )
    return pd.DataFrame(rows).sort_values(["kind", "model_a", "model_b"]).reset_index(drop=True)


def _regime_overlap(joined: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    models = sorted(joined)
    for regime, lower, upper in (
        ("calm", 0.0, 0.4),
        ("uncertain", 0.4, 0.6),
        ("stress", 0.6, 1.01),
    ):
        for first, second in itertools.combinations_with_replacement(models, 2):
            left = joined[first].sort_values(["date", "symbol"])
            right = joined[second].sort_values(["date", "symbol"])
            mask = (left["regime_probability"] >= lower) & (left["regime_probability"] < upper)
            signed_overlap = (
                float(
                    (
                        np.sign(left.loc[mask, "prediction"])
                        == np.sign(right.loc[mask, "prediction"])
                    ).mean()
                )
                if mask.any()
                else float("nan")
            )
            error_correlation = (
                float(left.loc[mask, "error"].corr(right.loc[mask, "error"]))
                if int(mask.sum()) >= 3
                else float("nan")
            )
            rows.append(
                {
                    "regime": regime,
                    "model_a": first,
                    "model_b": second,
                    "rows": int(mask.sum()),
                    "signed_prediction_overlap": signed_overlap,
                    "signed_prediction_overlap_status": (
                        "defined" if np.isfinite(signed_overlap) else "undefined_empty_regime"
                    ),
                    "error_correlation": error_correlation,
                    "error_correlation_status": (
                        "defined"
                        if np.isfinite(error_correlation)
                        else "undefined_constant_or_insufficient_variation"
                    ),
                }
            )
    return pd.DataFrame(rows).sort_values(["regime", "model_a", "model_b"]).reset_index(drop=True)


def _subset_reference(
    reference: _SyntheticReference,
    experts: tuple[str, ...],
    config: EnsembleStudyConfig,
) -> tuple[TrainingOOFPanel, InferenceBatch]:
    experts = tuple(sorted(experts))
    training = reference.training_frame[
        reference.training_frame["expert"].isin(experts)
    ].reset_index(drop=True)
    panel = TrainingOOFPanel.from_frame(
        training,
        folds=reference.panel.folds,
        expected_experts=experts,
        holdout_start=reference.panel.holdout_start,
        source_id=f"drop-one-{hashlib.sha256(','.join(experts).encode()).hexdigest()[:16]}",
        max_records=config.max_prediction_records,
    )
    inference_frame = reference.inference_frame[
        reference.inference_frame["expert"].isin(experts)
    ].reset_index(drop=True)
    return panel, InferenceBatch.from_frame(
        inference_frame,
        expected_keys=reference.inference.expected_keys,
        max_records=config.max_prediction_records,
    )


def _marginal_contribution(
    reference: _SyntheticReference,
    states: dict[str, GovernedEnsembleState],
    joined: dict[str, pd.DataFrame],
    config: EnsembleStudyConfig,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    full_mse = {method: float(np.mean(joined[method]["error"] ** 2)) for method in ENSEMBLE_METHODS}
    for method in ENSEMBLE_METHODS:
        for omitted in REFERENCE_EXPERTS:
            experts = tuple(expert for expert in REFERENCE_EXPERTS if expert != omitted)
            panel, inference = _subset_reference(reference, experts, config)
            state = fit_governed_ensemble(panel, _policy_config(config, method, experts))
            decisions = _decision_frame(state.predict(inference), method)
            evaluated = decisions.merge(
                reference.outcomes,
                on=["date", "symbol"],
                how="inner",
                validate="one_to_one",
            )
            mse = float(np.mean((evaluated["prediction"] - evaluated["target"]) ** 2))
            rows.append(
                {
                    "model": method,
                    "omitted_expert": omitted,
                    "full_mse": full_mse[method],
                    "mse_without_expert": mse,
                    "mse_increase_without_expert": mse - full_mse[method],
                    "full_state_id": states[method].identity,
                    "reduced_state_id": state.identity,
                    "reduced_fit_status": state.fit_status,
                }
            )
    return pd.DataFrame(rows).sort_values(["model", "omitted_expert"]).reset_index(drop=True)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


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
        "matplotlib": _installed_version("matplotlib"),
        "seaborn": _installed_version("seaborn"),
        "reference_device": "cpu",
    }


def _write_evidence_readme(path: Path) -> None:
    path.write_text(
        """# Governed ensemble reference evidence

This directory contains deterministic **synthetic engineering evidence** for
SF-S3-MR9. It verifies the temporal-OOF contracts, six combination policies,
fallback paths, aggregate diagnostics, and evidence publisher. It is not market
evidence, a backtest on licensed observations, paper trading, or live trading.

Reproduce into a new directory:

```bash
make ensemble-evidence OUTPUT=/absolute/path/to/new-evidence-directory
```

The frozen input is
[`configs/ensemble_benchmark.yaml`](../../../../configs/ensemble_benchmark.yaml).
The publisher refuses overwrite and writes no row predictions, targets, fitted
state, tensors, or model artifacts.

Artifacts:

- [`model_summary.csv`](model_summary.csv) — holdout error, within-date rank IC,
  moving-block variance/intervals, fallback, heuristic dispersion,
  turnover/cost, and paired best-single comparisons;
- [`prediction_error_correlations.csv`](prediction_error_correlations.csv) —
  pairwise prediction and error correlations with explicit undefined states;
- [`regime_overlap.csv`](regime_overlap.csv) — pairwise signed overlap and error
  correlation for calm, uncertain, and stress groups;
- [`marginal_contribution.csv`](marginal_contribution.csv) — drop-one expert
  ablations;
- [`turnover_costs.csv`](turnover_costs.csv) — transparent configured synthetic
  turnover-cost diagnostic with its complete moving-block policy;
- [`uncertainty_diagnostics.csv`](uncertainty_diagnostics.csv) — explicitly
  heuristic interval coverage and dispersion/error association;
- [`ensemble_evidence.png`](ensemble_evidence.png) — inspected Seaborn summary;
  and
- [`summary.json`](summary.json) — resolved config, generator/bootstrap seed
  map, resampling assumptions, runtime, identities, resource counts, artifact
  integrity hashes, and limitations.

See the [engineering report](../../../sprint_3_ensemble_report.md) and
[governed ensemble contract](../../../governed_ensembles.md) for interpretation.
""",
        encoding="utf-8",
    )


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    encoded = json.dumps(payload, sort_keys=True, indent=2, ensure_ascii=True, allow_nan=False)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
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


def run_synthetic_ensemble_study(
    config: EnsembleStudyConfig,
    output_dir: str | Path,
) -> EnsembleStudyResult:
    """Run and atomically publish the offline aggregate reference study."""

    destination = Path(output_dir)
    if destination.exists():
        raise FileExistsError(f"refusing to overwrite existing evidence directory {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{destination.name}.", dir=destination.parent))
    try:
        reference = _build_synthetic_reference(config)
        predictions = {
            expert: _single_expert_frame(reference, expert) for expert in REFERENCE_EXPERTS
        }
        states: dict[str, GovernedEnsembleState] = {}
        for method in ENSEMBLE_METHODS:
            state = fit_governed_ensemble(
                reference.panel,
                _policy_config(config, method, REFERENCE_EXPERTS),
            )
            states[method] = state
            predictions[method] = _decision_frame(state.predict(reference.inference), method)
        summary, joined = _evaluate_prediction_frames(
            predictions,
            reference.outcomes,
            config,
        )
        correlations = _correlations(joined)
        regime_overlap = _regime_overlap(joined)
        marginal = _marginal_contribution(reference, states, joined, config)
        uncertainty = summary[
            [
                "model",
                "kind",
                "rows",
                "mean_heuristic_dispersion",
                "heuristic_interval_coverage",
                "dispersion_absolute_error_correlation",
                "dispersion_correlation_status",
                "dispersion_rows",
                "fallback_rate",
            ]
        ].copy()
        turnover = summary[
            [
                "model",
                "kind",
                "mean_turnover",
                "mean_cost_drag",
                "gross_directional_return",
                "net_directional_return",
                "net_directional_return_ci_lower",
                "net_directional_return_ci_upper",
                "net_directional_return_standard_error",
                "net_directional_return_variance",
                "bootstrap_resamples",
                "bootstrap_block_dates",
                "bootstrap_confidence_level",
                "bootstrap_seed",
                "bootstrap_circular",
            ]
        ].copy()
        outputs = {
            "model_summary.csv": summary,
            "prediction_error_correlations.csv": correlations,
            "regime_overlap.csv": regime_overlap,
            "marginal_contribution.csv": marginal,
            "turnover_costs.csv": turnover,
            "uncertainty_diagnostics.csv": uncertainty,
        }
        for name, frame in outputs.items():
            frame.to_csv(staging / name, index=False, lineterminator="\n")
        _write_evidence_readme(staging / "README.md")
        plot_ensemble_evidence(
            summary,
            correlations,
            marginal,
            staging / "ensemble_evidence.png",
            generator_seed=config.seed,
            bootstrap_seed=config.bootstrap_seed,
            holdout_rows=len(reference.outcomes),
            transaction_cost_bps=config.transaction_cost_bps,
        )
        artifacts = {
            path.name: {
                "sha256": _sha256_file(path),
                "bytes": path.stat().st_size,
            }
            for path in sorted(staging.iterdir())
        }
        metadata: dict[str, Any] = {
            "schema_version": ENSEMBLE_STUDY_VERSION,
            "scope": config.interpretation,
            "config_id": config.identity,
            "resolved_config": asdict(config),
            "seed_map": {
                "generator_root_seed": config.seed,
                "derivation": (
                    "unsigned big-endian integer from the first eight bytes of "
                    "SHA-256('<root-seed>:<stream-name>')"
                ),
                "children": {
                    name: _named_child_seed(config.seed, name)
                    for name in (
                        "target",
                        *(f"expert:{expert}" for expert in sorted(REFERENCE_EXPERTS)),
                    )
                },
                "bootstrap_seed": config.bootstrap_seed,
            },
            "environment": _runtime_environment(),
            "metric_policy": {
                "rank_ic": (
                    "Spearman rank correlation within each date, arithmetic mean over "
                    "non-constant valid dates; valid and total date counts are published."
                ),
                "variability": (
                    "MSE, paired MSE differences, rank IC, and mean net directional-return "
                    "intervals use the predeclared moving-block bootstrap over complete dates."
                ),
                "resampling": {
                    "method": "moving_block_bootstrap",
                    "n_resamples": config.bootstrap_resamples,
                    "block_length_dates": config.bootstrap_block_length,
                    "confidence_level": config.bootstrap_confidence_level,
                    "seed": config.bootstrap_seed,
                    "circular": config.bootstrap_circular,
                    "interval": "equal-tailed empirical quantiles",
                    "standard_error": "sample standard deviation across bootstrap statistics",
                    "variance": "sample variance across bootstrap statistics",
                },
                "assumptions": [
                    "observations are ordered by date before resampling",
                    "prediction statistics retain complete within-date cross-sections",
                    "contiguous date blocks represent material short-range dependence",
                    "the evaluated synthetic process is sufficiently stable over the holdout",
                    "intervals are descriptive evidence, not future-performance guarantees",
                ],
                "dispersion": (
                    "Heuristic combination of supplied expert dispersion, training-OOF "
                    "residual MSE, and expert disagreement; not a calibrated predictive "
                    "standard deviation and not comparable to single-expert noise scales."
                ),
                "undefined": (
                    "Undefined correlations are published with an explicit status and a "
                    "blank numeric CSV value; they are never replaced by zero."
                ),
            },
            "training_boundary": {
                "panel_id": reference.panel.identity,
                "training_oof_rows": len(reference.panel.predictions),
                "training_dates": config.training_dates,
                "folds": config.folds,
                "holdout_start": reference.panel.holdout_start,
                "complete_experts": list(reference.panel.expected_experts),
            },
            "holdout": {
                "rows_evaluated_in_memory": len(reference.outcomes),
                "dates": config.holdout_dates,
                "row_predictions_published": 0,
                "row_targets_published": 0,
            },
            "states": {
                method: {
                    "state_id": state.identity,
                    "fit_status": state.fit_status,
                    "fallback_reason": state.fallback_reason,
                    "audit_records": len(state.audits),
                    "condition_number": state.condition_number,
                }
                for method, state in sorted(states.items())
            },
            "artifacts": artifacts,
            "limitations": [
                "Deterministic synthetic engineering evidence is not market evidence.",
                "No raw observations, row predictions, targets, fitted state, or model weights are published.",
                "The turnover/cost diagnostic is simplified and is not the event-driven execution engine.",
                "Final-holdout outcomes are evaluated outside the immutable target-free inference boundary.",
                "Dispersion and interval coverage are heuristic, not calibrated predictive uncertainty.",
                "Moving-block intervals do not cover generator, selection, or regime uncertainty.",
                "No result authorizes paper/live trading or implies persistent profitability.",
            ],
        }
        _atomic_json(staging / "summary.json", metadata)
        os.replace(staging, destination)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return EnsembleStudyResult(destination, summary, metadata)


__all__ = [
    "ENSEMBLE_METHODS",
    "ENSEMBLE_STUDY_VERSION",
    "REFERENCE_EXPERTS",
    "EnsembleStudyConfig",
    "EnsembleStudyResult",
    "load_ensemble_study_config",
    "run_synthetic_ensemble_study",
]
