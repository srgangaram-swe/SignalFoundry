"""Pre-registered, development-only deep-sequence comparison study."""

from __future__ import annotations

import json
import os
import platform
import shutil
import sys
import tempfile
from dataclasses import asdict, dataclass
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np
import pandas as pd
import seaborn as sns
import yaml

matplotlib.use("Agg")
from matplotlib import pyplot as plt  # noqa: E402

from alphaforge.data import SignalFoundryDataset
from alphaforge.features import build_features
from alphaforge.labels import build_labels
from alphaforge.models.deep_sequence import evaluate_oos_predictions
from alphaforge.training import run_walk_forward

STUDY_SCHEMA_VERSION = "1.0.0"
DEEP_SEQUENCE_CANDIDATES = (
    "lightgbm",
    "sequence_cnn",
    "sequence_tcn",
    "sequence_lstm",
    "sequence_gru",
    "sequence_transformer",
)
_ROOT_FIELDS = frozenset({"study", "models", "features", "walk_forward"})
_STUDY_FIELDS = frozenset(
    {
        "schema_version",
        "benchmark_symbol",
        "target",
        "horizons",
        "transaction_cost_bps",
        "selection_fraction",
        "seed",
        "interpretation",
    }
)


@dataclass(frozen=True)
class DeepSequenceStudyConfig:
    """Validated pre-registration for a development-only model comparison."""

    benchmark_symbol: str
    target: str
    horizons: tuple[int, ...]
    transaction_cost_bps: float
    selection_fraction: float
    seed: int
    interpretation: str
    models: tuple[dict[str, Any], ...]
    features: dict[str, Any]
    walk_forward: dict[str, Any]


@dataclass(frozen=True)
class DeepSequenceStudyResult:
    """Published aggregate evidence for one comparison."""

    output_dir: Path
    summary: pd.DataFrame
    metadata: dict[str, Any]


def load_deep_sequence_study_config(path: str | Path) -> DeepSequenceStudyConfig:
    """Load an exact-field YAML pre-registration or fail closed."""
    source = Path(path)
    try:
        payload = yaml.safe_load(source.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, yaml.YAMLError) as exc:
        raise ValueError(f"cannot load deep-sequence study config: {source}") from exc
    if not isinstance(payload, dict) or set(payload) != _ROOT_FIELDS:
        raise ValueError(f"study root fields must be exactly {sorted(_ROOT_FIELDS)}")
    study = payload["study"]
    if not isinstance(study, dict) or set(study) != _STUDY_FIELDS:
        raise ValueError(f"study fields must be exactly {sorted(_STUDY_FIELDS)}")
    if study["schema_version"] != STUDY_SCHEMA_VERSION:
        raise ValueError("unsupported deep-sequence study schema version")
    models = payload["models"]
    if not isinstance(models, list) or any(
        not isinstance(item, dict)
        or set(item) - {"name", "params"}
        or not isinstance(item.get("params", {}), dict)
        for item in models
    ):
        raise ValueError("every model requires only name and parameter mapping fields")
    if tuple(item.get("name") for item in models) != DEEP_SEQUENCE_CANDIDATES:
        raise ValueError(f"model family must be frozen as {list(DEEP_SEQUENCE_CANDIDATES)}")
    horizons = tuple(study["horizons"])
    if not horizons or any(not isinstance(value, int) or value < 1 for value in horizons):
        raise ValueError("horizons must contain positive integers")
    cost = float(study["transaction_cost_bps"])
    fraction = float(study["selection_fraction"])
    seed = study["seed"]
    if not np.isfinite(cost) or cost < 0.0:
        raise ValueError("transaction_cost_bps must be finite and non-negative")
    if not 0.0 < fraction <= 0.5:
        raise ValueError("selection_fraction must be in (0, 0.5]")
    if not isinstance(seed, int) or isinstance(seed, bool) or seed < 0:
        raise ValueError("seed must be a non-negative integer")
    for field in ("benchmark_symbol", "target", "interpretation"):
        if not isinstance(study[field], str) or not study[field].strip():
            raise ValueError(f"{field} must be a non-empty string")
    if not isinstance(payload["features"], dict) or not isinstance(payload["walk_forward"], dict):
        raise ValueError("features and walk_forward must be mappings")
    return DeepSequenceStudyConfig(
        benchmark_symbol=study["benchmark_symbol"],
        target=study["target"],
        horizons=horizons,
        transaction_cost_bps=cost,
        selection_fraction=fraction,
        seed=seed,
        interpretation=study["interpretation"],
        models=tuple(dict(item) for item in models),
        features=dict(payload["features"]),
        walk_forward=dict(payload["walk_forward"]),
    )


def _summarize(
    predictions: pd.DataFrame,
    metrics: pd.DataFrame,
    config: DeepSequenceStudyConfig,
) -> pd.DataFrame:
    if predictions.empty or metrics.empty:
        raise ValueError("deep-sequence study produced no OOS evidence")
    records: list[dict[str, Any]] = []
    for name in DEEP_SEQUENCE_CANDIDATES:
        candidate = predictions.loc[predictions["model"].eq(name)]
        if candidate.empty:
            raise ValueError(f"candidate {name} produced no OOS predictions")
        evidence = evaluate_oos_predictions(
            candidate,
            model=name,
            transaction_cost_bps=config.transaction_cost_bps,
            selection_fraction=config.selection_fraction,
        ).to_dict()
        folds = metrics.loc[metrics["model"].eq(name)]
        if folds.empty:
            raise ValueError(f"candidate {name} produced no fold metrics")
        evidence.update(
            {
                "fold_count": int(folds["window_id"].nunique()),
                "training_iterations": int(_finite_values(folds, "training_iterations").sum()),
                "parameter_count": _finite_mean(folds, "parameter_count"),
                "parameter_bytes": _finite_mean(folds, "parameter_bytes"),
                "fit_wall_seconds": _finite_sum(folds, "fit_wall_seconds"),
                "fit_cpu_seconds": _finite_sum(folds, "fit_cpu_seconds"),
                "peak_device_bytes": _finite_max(folds, "peak_device_bytes"),
                "best_validation_loss": _finite_mean(folds, "best_validation_loss"),
            }
        )
        records.append(evidence)
    return pd.DataFrame.from_records(records)


def _finite_values(frame: pd.DataFrame, column: str) -> pd.Series:
    if column not in frame:
        return pd.Series(dtype=float)
    values = pd.to_numeric(frame[column], errors="coerce")
    return values.loc[np.isfinite(values)]


def _finite_mean(frame: pd.DataFrame, column: str) -> float | None:
    values = _finite_values(frame, column)
    return None if values.empty else float(values.mean())


def _finite_sum(frame: pd.DataFrame, column: str) -> float | None:
    values = _finite_values(frame, column)
    return None if values.empty else float(values.sum())


def _finite_max(frame: pd.DataFrame, column: str) -> float | None:
    values = _finite_values(frame, column)
    return None if values.empty else float(values.max())


def _plot_summary(summary: pd.DataFrame, destination: Path) -> None:
    sns.set_theme(style="whitegrid", context="talk", palette="colorblind")
    figure, axes = plt.subplots(1, 3, figsize=(19, 6))
    sns.barplot(data=summary, x="rank_ic", y="model", ax=axes[0], color=sns.color_palette()[0])
    axes[0].axvline(0.0, color="black", linewidth=1)
    axes[0].set(title="Development OOS rank IC", xlabel="Mean per-date rank IC", ylabel="")
    sns.barplot(
        data=summary,
        x="net_mean_daily_return",
        y="model",
        ax=axes[1],
        color=sns.color_palette()[1],
    )
    axes[1].axvline(0.0, color="black", linewidth=1)
    axes[1].set(
        title="Simple costed long-short diagnostic",
        xlabel="Mean daily net return",
        ylabel="",
    )
    resource = summary.dropna(subset=["fit_wall_seconds"])
    sns.barplot(
        data=resource,
        x="fit_wall_seconds",
        y="model",
        ax=axes[2],
        color=sns.color_palette()[2],
    )
    axes[2].set(title="Measured model-fit wall time", xlabel="Seconds across folds", ylabel="")
    figure.suptitle(
        "Controlled deep-sequence benchmark — development folds only; no trading-readiness claim",
        fontsize=16,
    )
    figure.tight_layout()
    figure.savefig(destination, dpi=160, bbox_inches="tight")
    plt.close(figure)


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    encoded = json.dumps(
        payload,
        indent=2,
        sort_keys=True,
        ensure_ascii=True,
        allow_nan=False,
    )
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


def _runtime_environment() -> dict[str, Any]:
    """Record the minimum environment needed to interpret compute evidence."""
    return {
        "python": platform.python_version(),
        "python_implementation": platform.python_implementation(),
        "executable_platform": sys.platform,
        "operating_system": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor() or "not reported by operating system",
        "logical_cpu_count": os.cpu_count(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "seaborn": sns.__version__,
        "torch": _installed_version("torch"),
        "lightgbm": _installed_version("lightgbm"),
        "reference_device": "cpu",
        "warmup_runs": 0,
        "measurement_samples_per_fit": 1,
    }


def _installed_version(distribution: str) -> str:
    """Return package metadata without importing an optional native runtime."""
    try:
        return version(distribution)
    except PackageNotFoundError:
        return "unavailable"


def run_deep_sequence_study(
    dataset: SignalFoundryDataset,
    config: DeepSequenceStudyConfig,
    output_dir: str | Path,
) -> DeepSequenceStudyResult:
    """Run matched development folds and publish aggregate evidence atomically."""
    destination = Path(output_dir)
    if destination.exists():
        raise FileExistsError(f"study destination already exists: {destination}")
    staging = destination.parent / f".publishing-{destination.name}"
    if staging.exists():
        raise FileExistsError(f"stale study staging exists: {staging}")
    staging.mkdir(parents=True)
    try:
        panel = dataset.decision_panel if dataset.decision_panel is not None else dataset.panel
        features = build_features(
            panel,
            benchmark_symbol=config.benchmark_symbol,
            config=config.features,
        )
        labels = build_labels(
            dataset.panel,
            benchmark_symbol=config.benchmark_symbol,
            horizons=list(config.horizons),
        )
        result = run_walk_forward(
            features,
            labels,
            model_specs=[dict(item) for item in config.models],
            target=config.target,
            config=config.walk_forward,
            max_horizon=max(config.horizons),
            transform_config=config.features.get("fitted_transform"),
        )
        summary = _summarize(result.predictions, result.metrics, config)
        summary.to_csv(staging / "model_summary.csv", index=False)
        _plot_summary(summary, staging / "model_comparison.png")
        diagnostics = dataset.point_in_time_diagnostics
        metadata = {
            "schema_version": STUDY_SCHEMA_VERSION,
            "study": {
                "benchmark_symbol": config.benchmark_symbol,
                "target": config.target,
                "horizons": list(config.horizons),
                "transaction_cost_bps": config.transaction_cost_bps,
                "selection_fraction": config.selection_fraction,
                "seed": config.seed,
                "interpretation": config.interpretation,
                "candidate_order": list(DEEP_SEQUENCE_CANDIDATES),
            },
            "dataset": {
                "bundle_id": dataset.bundle_id,
                "rows": int(len(dataset.panel)),
                "symbols": int(dataset.panel["symbol"].nunique()),
                "date_min": str(pd.Timestamp(dataset.panel["date"].min()).date()),
                "date_max": str(pd.Timestamp(dataset.panel["date"].max()).date()),
                "schema_version": dataset.manifest.get("schema_version"),
                "point_in_time": None if diagnostics is None else asdict(diagnostics),
            },
            "evidence": {
                "folds": int(result.metrics["window_id"].nunique()),
                "prediction_rows_published": 0,
                "aggregate_rows": len(summary),
                "plot": "model_comparison.png",
                "summary": "model_summary.csv",
            },
            "environment": _runtime_environment(),
            "limitations": [
                "Development-only evidence; no protected final holdout was accessed.",
                "The simple costed diagnostic is not the event-driven execution simulator.",
                "WIKI bootstrap data is stale and incomplete point-in-time engineering data.",
                "No historical result authorizes paper or live trading or guarantees profit.",
            ],
        }
        _atomic_json(staging / "summary.json", metadata)
        os.replace(staging, destination)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return DeepSequenceStudyResult(destination, summary, metadata)
