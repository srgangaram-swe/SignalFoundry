"""Model-evaluation plots saved as static PNGs into a run directory.

Conventions (deliberate, not decorative):

- One measure per axis — never dual-axis; related measures stack as subplots.
- Categorical hues follow a fixed order (blue, aqua, ...) and never cycle.
- Diverging blue/red is used only where sign carries meaning (quantile
  returns); magnitude comparisons use a single hue.
- Grids and axes are recessive; ink goes to the data.
- Text stays in text colors; colored marks carry series identity.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal, cast

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from matplotlib.container import BarContainer

from alphaforge.research import read_frame_artifact

# Seaborn's colorblind palette is the categorical default for sprint evidence.
sns.set_theme(context="notebook", style="whitegrid", palette="colorblind")
BLUE, YELLOW, AQUA, RED, VIOLET, *_ = sns.color_palette("colorblind", 10)
SURFACE = "#fcfcfb"
GRID = "#e5e4e1"
TEXT = "#0b0b0b"
TEXT_2 = "#52514e"

_RC = matplotlib.RcParams(
    {
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "axes.edgecolor": GRID,
        "axes.labelcolor": TEXT_2,
        "axes.titlecolor": TEXT,
        "axes.titlesize": 11,
        "axes.labelsize": 9,
        "xtick.color": TEXT_2,
        "ytick.color": TEXT_2,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "grid.color": GRID,
        "grid.linewidth": 0.8,
        "legend.frameon": False,
        "legend.fontsize": 8,
        "font.size": 9,
    }
)


def _style(ax: plt.Axes) -> None:
    ax.grid(True, axis="y", zorder=0)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)


def _save(fig: plt.Figure, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def _label_bars(
    ax: plt.Axes,
    *,
    fmt: str,
    label_type: Literal["center", "edge"] = "edge",
    color: object = TEXT_2,
) -> None:
    for container in ax.containers:
        if isinstance(container, BarContainer):
            ax.bar_label(
                container,
                fmt=fmt,
                fontsize=8,
                color=color,
                padding=2,
                label_type=label_type,
            )


def plot_training_history(history: pd.DataFrame, path: str | Path) -> Path:
    """Loss curves and validation rank IC per epoch (stacked, never dual-axis)."""
    with plt.rc_context(_RC):
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(7, 5), sharex=True)
        losses = history[["epoch", "train_loss", "val_loss"]].melt(
            id_vars="epoch", var_name="series", value_name="loss"
        )
        losses["series"] = losses["series"].map(
            {"train_loss": "Train loss", "val_loss": "Validation loss"}
        )
        sns.lineplot(
            data=losses,
            x="epoch",
            y="loss",
            hue="series",
            palette={"Train loss": BLUE, "Validation loss": AQUA},
            linewidth=2,
            ax=ax1,
        )
        ax1.set_title("Training loss")
        ax1.legend(loc="upper right")
        _style(ax1)

        sns.lineplot(
            data=history,
            x="epoch",
            y="val_rank_ic",
            color=BLUE,
            linewidth=2,
            ax=ax2,
        )
        best = history["val_rank_ic"].idxmax()
        sns.scatterplot(
            x=[history.loc[best, "epoch"]],
            y=[history.loc[best, "val_rank_ic"]],
            color=BLUE,
            s=30,
            zorder=3,
            ax=ax2,
        )
        ax2.annotate(
            f"best {history.loc[best, 'val_rank_ic']:.3f}",
            (history.loc[best, "epoch"], history.loc[best, "val_rank_ic"]),
            textcoords="offset points",
            xytext=(6, 4),
            fontsize=8,
            color=TEXT_2,
        )
        ax2.axhline(0, color=GRID, lw=1)
        ax2.set_title("Validation rank IC (early-stopping metric)")
        ax2.set_xlabel("epoch")
        _style(ax2)
    return _save(fig, Path(path))


def plot_ic_timeseries(ic_by_date: pd.DataFrame, path: str | Path, window: int = 63) -> Path:
    """Daily rank IC with a rolling mean — is the signal stable or episodic?"""
    frame = ic_by_date.copy()
    frame["date"] = pd.to_datetime(frame["date"])
    frame["daily rank IC"] = frame["rank_ic"]
    frame[f"{window}d mean"] = (
        frame["rank_ic"].rolling(window, min_periods=max(1, window // 3)).mean()
    )
    lines = frame.melt(
        id_vars="date",
        value_vars=["daily rank IC", f"{window}d mean"],
        var_name="series",
        value_name="rank IC",
    )
    with plt.rc_context(_RC):
        fig, ax = plt.subplots(figsize=(8, 3.4))
        sns.lineplot(
            data=lines,
            x="date",
            y="rank IC",
            hue="series",
            palette={"daily rank IC": GRID, f"{window}d mean": BLUE},
            linewidth=1.4,
            ax=ax,
        )
        ax.axhline(0, color=TEXT_2, lw=0.8)
        ax.set_title("Out-of-sample daily rank IC")
        ax.legend(loc="upper right")
        _style(ax)
    return _save(fig, Path(path))


def plot_ic_decay(decay: pd.DataFrame, path: str | Path) -> Path:
    """Mean rank IC by label horizon — how fast the edge fades."""
    with plt.rc_context(_RC):
        fig, ax = plt.subplots(figsize=(5, 3.2))
        frame = decay.assign(horizon=decay["horizon"].astype(str))
        sns.barplot(
            data=frame,
            x="horizon",
            y="mean_rank_ic",
            color=BLUE,
            errorbar=None,
            ax=ax,
        )
        _label_bars(ax, fmt="%.3f")
        ax.axhline(0, color=TEXT_2, lw=0.8)
        ax.set_xlabel("horizon (days)")
        ax.set_ylabel("mean rank IC")
        ax.set_title("IC decay by forward horizon")
        _style(ax)
    return _save(fig, Path(path))


def plot_quantile_returns(quantiles: pd.DataFrame, path: str | Path) -> Path:
    """Mean forward return per prediction quantile. Sign carries meaning,
    so bars use the diverging blue/red pair around zero."""
    with plt.rc_context(_RC):
        fig, ax = plt.subplots(figsize=(5, 3.2))
        frame = quantiles.assign(
            quantile=quantiles["quantile"].astype(str),
            direction=quantiles["mean_return"].ge(0).map({True: "non-negative", False: "negative"}),
        )
        sns.barplot(
            data=frame,
            x="quantile",
            y="mean_return",
            hue="direction",
            palette={"non-negative": BLUE, "negative": RED},
            errorbar=None,
            dodge=False,
            legend=False,
            ax=ax,
        )
        _label_bars(ax, fmt="%.4f")
        ax.axhline(0, color=TEXT_2, lw=0.8)
        ax.set_xlabel("prediction quantile (1 = lowest)")
        ax.set_ylabel("mean forward return")
        ax.set_title("Realized forward return by prediction quantile")
        _style(ax)
    return _save(fig, Path(path))


def plot_model_comparison(ic_summary: pd.DataFrame, path: str | Path) -> Path:
    """Mean rank IC per model with Newey-West t-stats as direct labels."""
    frame = ic_summary.dropna(subset=["mean_ic"]).sort_values("mean_ic").copy()
    frame["model_label"] = frame["model"].str.replace("_", " ", regex=False)
    with plt.rc_context(_RC):
        fig, ax = plt.subplots(figsize=(7, 0.5 * len(frame) + 1.6))
        sns.barplot(
            data=frame,
            x="mean_ic",
            y="model_label",
            color=BLUE,
            errorbar=None,
            ax=ax,
        )
        bars = cast(BarContainer, ax.containers[0])
        labels = [f"t={t:.1f}" for t in frame["t_stat_nw"]]
        ax.bar_label(
            bars,
            labels=labels,
            fontsize=8,
            color="white",
            padding=0,
            label_type="center",
        )
        ax.axvline(0, color=TEXT_2, lw=0.8)
        ax.set_xlabel("mean daily rank IC (out-of-sample)")
        ax.set_ylabel("model")
        ax.set_title("Model comparison — walk-forward OOS")
        ax.grid(True, axis="x", zorder=0)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
    return _save(fig, Path(path))


def plot_prediction_scatter(
    predictions: pd.DataFrame, path: str | Path, max_points: int = 20_000
) -> Path:
    """Prediction vs realized forward return (2D histogram, sequential hue)."""
    frame = predictions[["prediction", "target"]].dropna()
    if len(frame) > max_points:
        frame = frame.sample(max_points, random_state=42)
    with plt.rc_context(_RC):
        fig, ax = plt.subplots(figsize=(5, 4.2))
        sns.histplot(
            data=frame,
            x="prediction",
            y="target",
            bins=40,
            cmap=sns.light_palette(BLUE, as_cmap=True),
            cbar=True,
            pthresh=0.01,
            ax=ax,
        )
        ax.axhline(0, color=TEXT_2, lw=0.8)
        ax.axvline(0, color=TEXT_2, lw=0.8)
        ax.set_xlabel("prediction")
        ax.set_ylabel("realized forward return")
        ax.set_title(f"Prediction vs realized (OOS, n={len(frame):,})")
        if len(fig.axes) > 1:
            fig.axes[-1].set_ylabel("observations")
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
    return _save(fig, Path(path))


def save_evaluation_plots(run_dir: str | Path, model: str | None = None) -> list[Path]:
    """Render every evaluation plot the run artifacts support.

    Reads the standard walk-forward artifacts and writes PNGs to
    ``<run_dir>/plots/``. Missing artifacts are skipped, not errors.
    """
    run_dir = Path(run_dir)
    plots_dir = run_dir / "plots"
    written: list[Path] = []

    history_path = run_dir / "training_history.csv"
    if history_path.exists():
        written.append(
            plot_training_history(pd.read_csv(history_path), plots_dir / "training_history.png")
        )

    ic_summary_path = run_dir / "ic_summary.csv"
    if ic_summary_path.exists():
        written.append(
            plot_model_comparison(pd.read_csv(ic_summary_path), plots_dir / "model_comparison.png")
        )

    decay_path = run_dir / "ic_decay.csv"
    if decay_path.exists():
        written.append(plot_ic_decay(pd.read_csv(decay_path), plots_dir / "ic_decay.png"))

    quantile_path = run_dir / "quantile_returns.csv"
    if quantile_path.exists():
        written.append(
            plot_quantile_returns(pd.read_csv(quantile_path), plots_dir / "quantile_returns.png")
        )

    predictions_path = run_dir / "predictions.table.json"
    if predictions_path.exists():
        preds = read_frame_artifact(predictions_path)
        if model is None and "model" in preds.columns and not preds.empty:
            from alphaforge.signals import select_model_predictions

            preds = select_model_predictions(preds)
        elif model is not None and "model" in preds.columns:
            preds = preds[preds["model"] == model]
        if not preds.empty:
            from alphaforge.evaluation import information_coefficient_by_date

            written.append(
                plot_ic_timeseries(
                    information_coefficient_by_date(preds), plots_dir / "ic_timeseries.png"
                )
            )
            written.append(plot_prediction_scatter(preds, plots_dir / "prediction_scatter.png"))
    return written
