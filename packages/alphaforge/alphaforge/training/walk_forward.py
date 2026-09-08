"""Leakage-aware walk-forward training and OOS prediction generation."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import pandas as pd

from alphaforge.evaluation import regression_metrics
from alphaforge.features import FittedFeatureTransformer, FittedTransformSpec, feature_columns
from alphaforge.models.registry import create_model

ID_COLUMNS = ["date", "symbol"]


@dataclass(frozen=True)
class WalkForwardConfig:
    scheme: str = "expanding"
    min_train_days: int = 756
    test_days: int = 126
    step_days: int = 126
    embargo_days: int = 21
    max_windows: int | None = None


@dataclass(frozen=True)
class WalkForwardWindow:
    window_id: int
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp
    embargo_days: int
    train_rows: int = 0
    test_rows: int = 0


@dataclass
class WalkForwardResult:
    predictions: pd.DataFrame
    metrics: pd.DataFrame
    feature_importance: pd.DataFrame
    windows: pd.DataFrame
    transformations: pd.DataFrame


def _coerce_config(config: WalkForwardConfig | dict | None) -> WalkForwardConfig:
    if config is None:
        return WalkForwardConfig()
    if isinstance(config, WalkForwardConfig):
        return config
    return WalkForwardConfig(**config)


def make_walk_forward_splits(
    dates: pd.Series | pd.Index | list,
    config: WalkForwardConfig | dict | None = None,
    max_horizon: int | None = None,
) -> list[WalkForwardWindow]:
    """Create expanding or rolling time splits with an explicit embargo gap."""
    cfg = _coerce_config(config)
    if cfg.embargo_days < 0:
        raise ValueError("embargo_days must be non-negative")
    if max_horizon is not None and cfg.embargo_days < max_horizon:
        raise ValueError(
            f"embargo_days ({cfg.embargo_days}) must be >= max label horizon ({max_horizon})"
        )
    if cfg.scheme not in {"expanding", "rolling"}:
        raise ValueError("walk-forward scheme must be 'expanding' or 'rolling'")

    unique_dates = pd.Index(pd.to_datetime(pd.Series(dates).drop_duplicates()).sort_values())
    if len(unique_dates) <= cfg.min_train_days + cfg.embargo_days:
        return []

    windows: list[WalkForwardWindow] = []
    test_start_idx = cfg.min_train_days + cfg.embargo_days
    window_id = 0
    while test_start_idx < len(unique_dates):
        train_end_idx = test_start_idx - cfg.embargo_days - 1
        test_end_idx = min(test_start_idx + cfg.test_days - 1, len(unique_dates) - 1)
        train_start_idx = 0
        if cfg.scheme == "rolling":
            train_start_idx = max(0, train_end_idx - cfg.min_train_days + 1)

        train_len = train_end_idx - train_start_idx + 1
        if train_len >= cfg.min_train_days and test_end_idx >= test_start_idx:
            windows.append(
                WalkForwardWindow(
                    window_id=window_id,
                    train_start=unique_dates[train_start_idx],
                    train_end=unique_dates[train_end_idx],
                    test_start=unique_dates[test_start_idx],
                    test_end=unique_dates[test_end_idx],
                    embargo_days=cfg.embargo_days,
                )
            )
            window_id += 1
            if cfg.max_windows is not None and len(windows) >= cfg.max_windows:
                break

        test_start_idx += cfg.step_days
    return windows


def supervised_frame(
    features: pd.DataFrame,
    labels: pd.DataFrame,
    target: str,
) -> tuple[pd.DataFrame, list[str]]:
    """Merge features and labels into one time-sorted modeling table."""
    if target not in labels.columns:
        raise KeyError(f"target {target!r} not found in labels")
    x_cols = feature_columns(features)
    data = features.merge(labels[ID_COLUMNS + [target]], on=ID_COLUMNS, how="inner")
    data["date"] = pd.to_datetime(data["date"])
    for col in x_cols:
        data[col] = pd.to_numeric(data[col], errors="coerce")
    return data.sort_values(ID_COLUMNS).reset_index(drop=True), x_cols


def _model_matrix(frame: pd.DataFrame, columns: list[str], model) -> pd.DataFrame:
    """Feature matrix for one model; sequence models get a (date, symbol) index."""
    X = frame[columns].copy()
    if getattr(model, "needs_sequence_index", False):
        X.index = pd.MultiIndex.from_frame(frame[ID_COLUMNS])
    return X


def _index_model_matrix(matrix: pd.DataFrame, frame: pd.DataFrame, model) -> pd.DataFrame:
    """Attach temporal identifiers without changing transformed values."""

    result = matrix.copy()
    if getattr(model, "needs_sequence_index", False):
        result.index = pd.MultiIndex.from_frame(frame[ID_COLUMNS])
    return result


def run_walk_forward(
    features: pd.DataFrame,
    labels: pd.DataFrame,
    model_specs: list[dict],
    target: str,
    config: WalkForwardConfig | dict | None = None,
    max_horizon: int | None = None,
    transform_config: FittedTransformSpec | dict | None = None,
) -> WalkForwardResult:
    """Train each model per window and return OOS-only predictions.

    Enabled fitted transformations are learned once from each window's
    training rows and then applied unchanged to its test rows.
    """
    cfg = _coerce_config(config)
    transform_spec = (
        transform_config
        if isinstance(transform_config, FittedTransformSpec)
        else FittedTransformSpec.from_config(transform_config)
    )
    data, x_cols = supervised_frame(features, labels, target)
    windows = make_walk_forward_splits(data["date"], cfg, max_horizon=max_horizon)
    if not windows:
        raise ValueError("not enough dates to create any walk-forward window")

    pred_frames: list[pd.DataFrame] = []
    metric_rows: list[dict] = []
    importance_frames: list[pd.DataFrame] = []
    window_rows: list[dict] = []
    transformation_rows: list[dict] = []

    specs = model_specs or [{"name": "zero_baseline", "params": {}}]
    for window in windows:
        train_mask = (data["date"] >= window.train_start) & (data["date"] <= window.train_end)
        test_mask = (data["date"] >= window.test_start) & (data["date"] <= window.test_end)
        train = data.loc[train_mask].dropna(subset=[target]).copy()
        test = data.loc[test_mask].dropna(subset=[target]).copy()
        if train.empty or test.empty:
            continue

        wrow = asdict(window)
        wrow["train_rows"] = int(len(train))
        wrow["test_rows"] = int(len(test))
        window_rows.append(wrow)

        transformed_train: pd.DataFrame | None = None
        transformed_test: pd.DataFrame | None = None
        if transform_spec.enabled:
            transformer = FittedFeatureTransformer(transform_spec)
            transformed_train = transformer.fit_transform(train[x_cols], train["date"])
            transformed_test = transformer.transform(test[x_cols])
            if transformer.state_ is None:  # pragma: no cover - fit_transform guarantees state
                raise RuntimeError("fitted transformer did not publish state")
            transformation_rows.append(
                {"window_id": window.window_id, **asdict(transformer.state_)}
            )

        for spec in specs:
            name = spec["name"]
            params = spec.get("params", {})
            model = create_model(name, **params)
            if transform_spec.enabled and not getattr(model, "requires_raw_features", False):
                if transformed_train is None or transformed_test is None:  # pragma: no cover
                    raise RuntimeError("enabled fitted transformation is unavailable")
                X_train = _index_model_matrix(transformed_train, train, model)
                X_test = _index_model_matrix(transformed_test, test, model)
            else:
                X_train = _model_matrix(train, x_cols, model)
                X_test = _model_matrix(test, x_cols, model)
            y_train = train[target].astype(float)
            y_test = test[target].astype(float)

            fit_target = y_train
            if getattr(model, "needs_sequence_index", False):
                # Sequence matrices replace the row index with the temporal
                # (date, symbol) identity. Carry the identical index onto the
                # positional target before crossing the model contract.
                fit_target = y_train.set_axis(X_train.index)
            model.fit(X_train, fit_target)
            pred = pd.Series(model.predict(X_test), index=test.index, dtype=float)

            block = test[ID_COLUMNS + [target]].copy()
            block = block.rename(columns={target: "target"})
            block["target_name"] = target
            block["prediction"] = pred.to_numpy()
            block["model"] = name
            block["window_id"] = window.window_id
            pred_frames.append(block)

            metrics: dict[str, Any] = dict(regression_metrics(y_test, pred))
            metrics.update({"model": name, "window_id": window.window_id})
            diagnostics = model.training_diagnostics()
            if diagnostics is not None:
                metrics.update(
                    {
                        "training_backend": diagnostics.backend,
                        "training_status": diagnostics.status,
                        "training_iterations": diagnostics.iterations,
                        "training_iteration_limit": diagnostics.iteration_limit,
                        "training_seed": diagnostics.seed,
                        "training_warning_count": len(diagnostics.warnings),
                    }
                )
            resource_evidence = getattr(model, "resource_evidence", None)
            if callable(resource_evidence):
                resources = resource_evidence()
                metrics.update(
                    {
                        "parameter_count": resources.parameter_count,
                        "parameter_bytes": resources.parameter_bytes,
                        "fit_wall_seconds": resources.fit_wall_seconds,
                        "fit_cpu_seconds": resources.fit_cpu_seconds,
                        "peak_device_bytes": resources.peak_device_bytes,
                        "best_epoch": resources.best_epoch,
                        "best_validation_loss": resources.best_validation_loss,
                    }
                )
            metric_rows.append(metrics)

            importance = model.feature_importance()
            if importance is not None:
                importance_frames.append(
                    importance.rename("importance")
                    .reset_index()
                    .rename(columns={"index": "feature"})
                    .assign(model=name, window_id=window.window_id)
                )

    predictions = pd.concat(pred_frames, ignore_index=True) if pred_frames else pd.DataFrame()
    metrics = pd.DataFrame(metric_rows)
    feature_importance = (
        pd.concat(importance_frames, ignore_index=True) if importance_frames else pd.DataFrame()
    )
    windows_frame = pd.DataFrame(window_rows)
    transformations = pd.DataFrame(transformation_rows)
    return WalkForwardResult(
        predictions,
        metrics,
        feature_importance,
        windows_frame,
        transformations,
    )
