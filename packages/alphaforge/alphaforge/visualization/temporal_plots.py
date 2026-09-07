"""Seaborn visual evidence for interval-aware temporal validation."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from alphaforge.training.temporal_validation import TemporalFold, fold_assignments
from alphaforge.visualization.plots import _RC, _save

ROLE_ORDER = [
    "train",
    "purge",
    "validation",
    "embargo",
    "test",
    "overlap",
    "final_holdout",
]
ROLE_PALETTE = {
    "train": "#0072B2",
    "purge": "#56B4E9",
    "validation": "#009E73",
    "embargo": "#F0E442",
    "test": "#E69F00",
    "overlap": "#D55E00",
    "final_holdout": "#CC79A7",
}


def plot_temporal_folds(folds: list[TemporalFold], path: str | Path) -> Path:
    """Render every fold role without exposing labels, features, or returns."""

    frame = fold_assignments(folds)
    frame["role"] = pd.Categorical(frame["role"], categories=ROLE_ORDER, ordered=True)
    with plt.rc_context(_RC):
        figure_height = max(4.8, min(12.0, 1.0 + 0.65 * len(folds)))
        fig, ax = plt.subplots(figsize=(12.0, figure_height))
        sns.scatterplot(
            data=frame,
            x="date",
            y="fold_id",
            hue="role",
            hue_order=ROLE_ORDER,
            palette=ROLE_PALETTE,
            marker="s",
            s=38,
            linewidth=0,
            ax=ax,
        )
        ax.set_title("Synthetic temporal-validation roles — final holdout is never selected on")
        ax.set_xlabel("market session")
        ax.set_ylabel("fold id")
        ax.xaxis.set_major_locator(mdates.AutoDateLocator(minticks=5, maxticks=10))
        ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(ax.xaxis.get_major_locator()))
        ax.set_yticks(sorted(frame["fold_id"].unique()))
        ax.grid(True, axis="x", alpha=0.35)
        ax.legend(
            title="role",
            loc="upper center",
            bbox_to_anchor=(0.5, -0.14),
            ncol=4,
            frameon=False,
        )
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
    return _save(fig, Path(path))
