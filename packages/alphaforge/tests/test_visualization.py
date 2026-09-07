"""Tests for the evaluation plots module (no torch required)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from alphaforge.research import write_frame_artifact
from alphaforge.visualization import save_evaluation_plots
from alphaforge.visualization.plots import plot_quantile_returns, plot_training_history


def _fake_run_dir(tmp_path):
    rng = np.random.default_rng(0)
    dates = pd.bdate_range("2023-01-02", periods=60)

    pd.DataFrame(
        {
            "epoch": np.arange(1, 11),
            "train_loss": np.linspace(1.0, 0.4, 10),
            "val_loss": np.linspace(1.1, 0.6, 10),
            "val_rank_ic": np.linspace(0.0, 0.08, 10),
            "lr": np.linspace(3e-4, 1e-5, 10),
        }
    ).to_csv(tmp_path / "training_history.csv", index=False)

    pd.DataFrame(
        {
            "model": ["ridge", "temporal_alpha"],
            "mean_ic": [0.03, 0.05],
            "ic_std": [0.2, 0.2],
            "icir": [0.15, 0.25],
            "t_stat_nw": [2.1, 3.3],
            "pct_positive": [0.55, 0.58],
            "n_dates": [60, 60],
        }
    ).to_csv(tmp_path / "ic_summary.csv", index=False)

    pd.DataFrame(
        {
            "horizon": [1, 5, 20],
            "mean_rank_ic": [0.02, 0.04, 0.06],
            "t_stat_nw": [1, 2, 3],
            "n_dates": [60, 60, 60],
        }
    ).to_csv(tmp_path / "ic_decay.csv", index=False)

    pd.DataFrame(
        {
            "quantile": [1, 2, 3, 4, 5],
            "mean_return": [-0.004, -0.001, 0.0, 0.002, 0.005],
            "median_return": [-0.003, -0.001, 0.0, 0.001, 0.004],
            "count": [100] * 5,
        }
    ).to_csv(tmp_path / "quantile_returns.csv", index=False)

    rows = []
    for date in dates:
        for i in range(6):
            target = rng.normal(0, 0.02)
            rows.append(
                {
                    "date": date,
                    "symbol": f"S{i}",
                    "target": target,
                    "prediction": target * 0.3 + rng.normal(0, 0.01),
                    "model": "temporal_alpha",
                }
            )
    write_frame_artifact(pd.DataFrame(rows), tmp_path / "predictions.table.json")
    return tmp_path


def test_save_evaluation_plots_writes_all_pngs(tmp_path):
    run_dir = _fake_run_dir(tmp_path)
    written = save_evaluation_plots(run_dir, model="temporal_alpha")
    names = {p.name for p in written}
    assert {
        "training_history.png",
        "model_comparison.png",
        "ic_decay.png",
        "quantile_returns.png",
        "ic_timeseries.png",
        "prediction_scatter.png",
    } <= names
    for path in written:
        assert path.exists() and path.stat().st_size > 1_000


def test_plots_skip_missing_artifacts(tmp_path):
    assert save_evaluation_plots(tmp_path) == []


def test_individual_plot_functions_return_paths(tmp_path):
    history = pd.DataFrame(
        {
            "epoch": [1, 2],
            "train_loss": [1.0, 0.5],
            "val_loss": [1.0, 0.6],
            "val_rank_ic": [0.01, 0.02],
            "lr": [1e-3, 1e-4],
        }
    )
    out = plot_training_history(history, tmp_path / "h.png")
    assert out.exists()
    quantiles = pd.DataFrame({"quantile": [1, 2], "mean_return": [-0.01, 0.01]})
    out = plot_quantile_returns(quantiles, tmp_path / "q.png")
    assert out.exists()


def test_report_embeds_plots_section(tmp_path):
    from alphaforge.reporting import write_markdown_report

    run_dir = _fake_run_dir(tmp_path)
    save_evaluation_plots(run_dir, model="temporal_alpha")
    report = write_markdown_report(run_dir)
    text = report.read_text()
    assert "## Evaluation Plots" in text
    assert "![training history](plots/training_history.png)" in text
