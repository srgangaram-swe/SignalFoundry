"""Train the neural temporal alpha model with a full, honest protocol.

This is the standalone training entry point (`make train`). Protocol:

1. Load panel -> features -> multi-horizon labels (synthetic or real data
   per configs/data.yaml — the same path used for yfinance/CSV universes).
2. Chronological split by DATE: train+val | embargo (>= max horizon) | test.
   The model performs its own chronological validation inside train+val for
   early stopping; the test block is touched exactly once, at the end.
3. Fit TemporalAlphaNet (multi-task across horizons) with the real training
   loop: date-batched sampling, AdamW + cosine schedule, gradient clipping,
   early stopping on validation rank IC, best-checkpoint restore.
4. Evaluate on the held-out test block; persist checkpoint, per-epoch
   training history, metrics, predictions, and evaluation plots.

Note: a single chronological split is a *model development* tool. The
statistically serious evaluation remains scripts/run_walk_forward.py; this
script exists so there is a real, inspectable training loop with artifacts.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict

import pandas as pd
from _common import (
    load_configs,
    make_run_dir,
    record_pipeline_manifest,
    save_meta,
    utc_timestamp,
)

from alphaforge.data import load_prices
from alphaforge.evaluation import (
    ic_decay,
    ic_summary,
    information_coefficient_by_date,
    quantile_return_table,
    regression_metrics,
)
from alphaforge.features import FittedFeatureTransformer, FittedTransformSpec, build_features
from alphaforge.labels.labels import build_labels
from alphaforge.models.registry import seed_model_specs
from alphaforge.research import write_frame_artifact
from alphaforge.training.walk_forward import ID_COLUMNS, supervised_frame
from alphaforge.utils import save_json, set_seed
from alphaforge.visualization import save_evaluation_plots


def parse_args() -> argparse.Namespace:
    def positive_int(value: str) -> int:
        parsed = int(value)
        if parsed < 1:
            raise argparse.ArgumentTypeError("must be positive")
        return parsed

    def open_unit_float(value: str) -> float:
        parsed = float(value)
        if not 0.0 < parsed < 1.0:
            raise argparse.ArgumentTypeError("must be in (0, 1)")
        return parsed

    parser = argparse.ArgumentParser(description="Train the temporal alpha model end to end.")
    parser.add_argument("--config", default="configs/models.yaml")
    parser.add_argument("--data-config", default="configs/data.yaml")
    parser.add_argument("--feature-config", default="configs/features.yaml")
    parser.add_argument("--synthetic", action="store_true", help="Force the synthetic source.")
    parser.add_argument("--target", help="Label column (default from config).")
    parser.add_argument("--epochs", type=positive_int, help="Override max_epochs.")
    parser.add_argument("--test-fraction", type=open_unit_float, default=0.15)
    parser.add_argument("--runs-dir", help="Override the validated models-config run root.")
    return parser.parse_args()


def main() -> None:
    from alphaforge.models.temporal import TemporalAlphaModel

    started_at = utc_timestamp()
    args = parse_args()
    model_cfg, data_cfg, feature_cfg = load_configs(
        args.config, args.data_config, args.feature_config
    )
    if args.synthetic:
        data_cfg["source"] = "synthetic"

    horizons = list(model_cfg.get("horizons", [1, 5, 20]))
    target = args.target or model_cfg.get("target", f"fwd_ret_{horizons[0]}")
    temporal_params = dict(model_cfg.get("temporal", {}))
    if args.epochs is not None:
        temporal_params["max_epochs"] = args.epochs
    root_seed = int(model_cfg["seed"])
    set_seed(root_seed)
    model_cfg["models"] = seed_model_specs(model_cfg["models"], root_seed)
    model_cfg["target"] = target
    model_cfg["temporal"] = temporal_params

    run_dir = make_run_dir(args.runs_dir or model_cfg["runs_dir"], prefix="train")
    panel, benchmark = load_prices(data_cfg)
    features = build_features(panel, benchmark_symbol=benchmark, config=feature_cfg)
    labels = build_labels(panel, benchmark_symbol=benchmark, horizons=horizons)

    data, x_cols = supervised_frame(features, labels, target)
    aux_cols = [f"fwd_ret_{h}" for h in horizons if f"fwd_ret_{h}" != target]
    data = data.merge(labels[ID_COLUMNS + aux_cols], on=ID_COLUMNS, how="left")
    data = data.dropna(subset=[target]).reset_index(drop=True)

    # chronological train+val | embargo | test split on dates
    dates = data["date"].sort_values().unique()
    n_test = max(1, int(len(dates) * args.test_fraction))
    embargo = max(horizons)
    test_dates = dates[-n_test:]
    trainval_dates = dates[: -(n_test + embargo)]
    if len(trainval_dates) < 100:
        raise SystemExit("not enough history to train; extend the data range")
    train_frame = data[data["date"].isin(trainval_dates)]
    test_frame = data[data["date"].isin(test_dates)]

    transform_spec = FittedTransformSpec.from_config(feature_cfg.get("fitted_transform"))
    transformer: FittedFeatureTransformer | None = None
    if transform_spec.enabled:
        transformer = FittedFeatureTransformer(transform_spec)
        train_values = transformer.fit_transform(train_frame[x_cols], train_frame["date"])
        test_values = transformer.transform(test_frame[x_cols])
    else:
        train_values = train_frame[x_cols].copy()
        test_values = test_frame[x_cols].copy()

    def matrix(values: pd.DataFrame, frame: pd.DataFrame) -> pd.DataFrame:
        X = values.copy()
        X.index = pd.MultiIndex.from_frame(frame[ID_COLUMNS])
        return X

    model = TemporalAlphaModel(**temporal_params)
    print(
        f"training temporal_alpha: {len(train_frame):,} rows, "
        f"{len(trainval_dates)} dates, target={target}, aux={aux_cols}"
    )
    model.fit(
        matrix(train_values, train_frame),
        train_frame[target],
        aux_y=train_frame[aux_cols],
    )
    history_record = model.history_
    if history_record is None:
        raise RuntimeError("temporal training completed without a training history")
    history = history_record.to_frame()
    best = int(history_record.best_epoch)
    print(
        f"trained {len(history)} epochs; best epoch {best} "
        f"(val rank IC {history['val_rank_ic'].max():.4f})"
    )

    # single-touch test evaluation
    predictions = test_frame[ID_COLUMNS].copy()
    predictions["target"] = test_frame[target].to_numpy()
    predictions["prediction"] = model.predict(matrix(test_values, test_frame))
    predictions["model"] = "temporal_alpha"
    metrics = regression_metrics(predictions["target"], predictions["prediction"])
    by_date = information_coefficient_by_date(predictions)
    ic_stats = ic_summary(by_date)

    # artifacts
    model.save(run_dir / "checkpoint.pt")
    write_frame_artifact(panel, run_dir / "panel.table.json")
    history.to_csv(run_dir / "training_history.csv", index=False)
    write_frame_artifact(predictions, run_dir / "predictions.table.json")
    pd.DataFrame([{"model": "temporal_alpha", **ic_stats}]).to_csv(
        run_dir / "ic_summary.csv", index=False
    )
    quantile_return_table(predictions).to_csv(run_dir / "quantile_returns.csv", index=False)
    ic_decay(predictions, labels, horizons).to_csv(run_dir / "ic_decay.csv", index=False)
    save_json({**metrics, **ic_stats}, run_dir / "test_metrics.json")
    if transformer is not None and transformer.state_ is not None:
        save_json(asdict(transformer.state_), run_dir / "fitted_transformation.json")
    save_meta(
        run_dir,
        kind="temporal_training",
        target=target,
        aux_targets=aux_cols,
        temporal_config=temporal_params,
        best_epoch=best,
        embargo_days=embargo,
        test_dates=[str(test_dates[0]), str(test_dates[-1])],
        data_config=data_cfg,
    )
    plots = save_evaluation_plots(run_dir, model="temporal_alpha")
    manifest = record_pipeline_manifest(
        run_dir=run_dir,
        panel=panel,
        data_config=data_cfg,
        model_config=model_cfg,
        feature_config=feature_cfg,
        started_at=started_at,
    )

    print(f"run dir: {run_dir}")
    print(f"experiment id: {manifest.experiment_id}")
    print(
        f"test rank IC {ic_stats['mean_ic']:.4f} (NW t={ic_stats['t_stat_nw']:.2f}), "
        f"directional accuracy {metrics['directional_accuracy']:.3f}"
    )
    print(f"plots: {[p.name for p in plots]}")


if __name__ == "__main__":
    main()
