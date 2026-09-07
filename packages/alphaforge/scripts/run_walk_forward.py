from __future__ import annotations

import argparse

import pandas as pd
from _common import (
    configure_fast_demo,
    load_configs,
    make_run_dir,
    record_pipeline_manifest,
    save_meta,
    utc_timestamp,
    write_latest,
)

from alphaforge.data import data_quality_report, load_prices
from alphaforge.evaluation import (
    ic_decay,
    ic_summary,
    information_coefficient_by_date,
    probability_of_backtest_overfitting,
    quantile_return_table,
)
from alphaforge.features import build_features
from alphaforge.labels.labels import build_labels
from alphaforge.models.registry import seed_model_specs
from alphaforge.research import write_frame_artifact
from alphaforge.signals import select_model_predictions
from alphaforge.training import run_walk_forward
from alphaforge.utils import save_json, set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run leakage-safe walk-forward validation.")
    parser.add_argument("--config", default="configs/models.yaml")
    parser.add_argument("--data-config", default="configs/data.yaml")
    parser.add_argument("--feature-config", default="configs/features.yaml")
    parser.add_argument(
        "--synthetic", action="store_true", help="Force the no-network synthetic source."
    )
    parser.add_argument(
        "--fast", action="store_true", help="Use a small CI-friendly synthetic experiment."
    )
    parser.add_argument("--runs-dir", help="Override the validated models-config run root.")
    return parser.parse_args()


def main() -> None:
    started_at = utc_timestamp()
    args = parse_args()
    model_cfg, data_cfg, feature_cfg = load_configs(
        args.config, args.data_config, args.feature_config
    )
    if args.synthetic:
        data_cfg["source"] = "synthetic"
    if args.fast:
        configure_fast_demo(model_cfg, data_cfg)
    root_seed = int(model_cfg["seed"])
    set_seed(root_seed)
    model_cfg["models"] = seed_model_specs(model_cfg["models"], root_seed)

    run_dir = make_run_dir(args.runs_dir or model_cfg["runs_dir"], prefix="wf")
    panel, benchmark = load_prices(data_cfg)
    quality = data_quality_report(panel)
    features = build_features(panel, benchmark_symbol=benchmark, config=feature_cfg)
    horizons = list(model_cfg.get("horizons", [1, 5, 20]))
    labels = build_labels(panel, benchmark_symbol=benchmark, horizons=horizons)

    result = run_walk_forward(
        features=features,
        labels=labels,
        model_specs=model_cfg.get("models", []),
        target=model_cfg.get("target", f"fwd_ret_{horizons[0]}"),
        config=model_cfg.get("walk_forward", {}),
        max_horizon=max(horizons),
        transform_config=feature_cfg.get("fitted_transform"),
    )

    write_frame_artifact(panel, run_dir / "panel.table.json")
    write_frame_artifact(features, run_dir / "features.table.json")
    write_frame_artifact(labels, run_dir / "labels.table.json")
    write_frame_artifact(result.predictions, run_dir / "predictions.table.json")
    quality.to_csv(run_dir / "data_quality.csv", index=False)
    result.metrics.to_csv(run_dir / "model_metrics.csv", index=False)
    result.windows.to_csv(run_dir / "walk_forward_windows.csv", index=False)
    if not result.transformations.empty:
        result.transformations.to_csv(run_dir / "fitted_transformations.csv", index=False)
    if not result.feature_importance.empty:
        result.feature_importance.to_csv(run_dir / "feature_importance.csv", index=False)

    selected = select_model_predictions(result.predictions)
    quantiles = quantile_return_table(selected)
    quantiles.to_csv(run_dir / "quantile_returns.csv", index=False)

    # --- IC inference per model (Newey-West robust to overlapping labels) ---
    ic_rows = []
    ic_panels = {}
    for name, g in result.predictions.groupby("model"):
        by_date = information_coefficient_by_date(g)
        ic_panels[name] = by_date.set_index("date")["rank_ic"]
        ic_rows.append({"model": name, **ic_summary(by_date)})
    ic_table = pd.DataFrame(ic_rows).sort_values("mean_ic", ascending=False)
    ic_table.to_csv(run_dir / "ic_summary.csv", index=False)

    # --- IC decay: how fast does the selected signal's edge fade? ---
    ic_decay(selected, labels, horizons).to_csv(run_dir / "ic_decay.csv", index=False)

    # --- PBO across models: is the winner real or selection bias? ---
    ic_matrix = pd.DataFrame(ic_panels).sort_index()
    pbo = probability_of_backtest_overfitting(ic_matrix, n_blocks=min(16, len(ic_matrix) // 4))
    save_json(
        {k: v for k, v in pbo.items() if k != "logits"},
        run_dir / "overfitting.json",
    )

    # --- evaluation plots (PNG) for the report and dashboard ---
    from alphaforge.visualization import save_evaluation_plots

    plot_paths = save_evaluation_plots(run_dir)
    print(f"evaluation plots: {len(plot_paths)} written to {run_dir / 'plots'}")

    save_meta(
        run_dir,
        benchmark_symbol=benchmark,
        target=model_cfg.get("target", f"fwd_ret_{horizons[0]}"),
        horizons=horizons,
        data_config=data_cfg,
        model_config=model_cfg,
        feature_config=feature_cfg,
    )
    manifest = record_pipeline_manifest(
        run_dir=run_dir,
        panel=panel,
        data_config=data_cfg,
        model_config=model_cfg,
        feature_config=feature_cfg,
        started_at=started_at,
    )
    write_latest(run_dir)
    print(f"walk-forward run: {run_dir}")
    print(f"experiment id: {manifest.experiment_id}")
    print(f"oos predictions: {len(result.predictions):,}")
    print(f"models: {sorted(result.predictions['model'].unique())}")


if __name__ == "__main__":
    main()
