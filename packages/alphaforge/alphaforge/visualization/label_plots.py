"""Seaborn visual evidence for financial-label diagnostics."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from alphaforge.labels import LabelDiagnostics
from alphaforge.visualization.plots import _RC, AQUA, BLUE, GRID, RED, TEXT_2, _save, _style


def plot_label_dependence(summary: pd.DataFrame, path: str | Path) -> Path:
    """Show overlap, serial dependence, and effective information fraction."""

    frame = summary.copy()
    autocorrelation_column = next(
        column for column in frame if column.startswith("autocorrelation_lag_")
    )
    frame["effective_information_fraction"] = frame["effective_sample_size"] / frame["observations"]
    long = frame.melt(
        id_vars=["label"],
        value_vars=[
            "overlap_rate",
            autocorrelation_column,
            "effective_information_fraction",
        ],
        var_name="diagnostic",
        value_name="fraction",
    )
    long["diagnostic"] = long["diagnostic"].map(
        {
            "overlap_rate": "overlapping adjacent events",
            autocorrelation_column: autocorrelation_column.replace("_", " "),
            "effective_information_fraction": "effective information / observations",
        }
    )
    with plt.rc_context(_RC):
        fig, ax = plt.subplots(figsize=(9, 4.8))
        sns.barplot(
            data=long,
            x="fraction",
            y="label",
            hue="diagnostic",
            palette=[BLUE, AQUA, RED],
            errorbar=None,
            ax=ax,
        )
        ax.axvline(0, color=TEXT_2, linewidth=0.8)
        ax.set_xlim(-1, 1)
        ax.set_xlabel("fraction / correlation (bounded to [-1, 1])")
        ax.set_ylabel("label")
        ax.set_title(
            "Synthetic label dependence — overlap and autocorrelation reduce effective evidence"
        )
        ax.legend(loc="lower left", title=None)
        ax.grid(True, axis="x", zorder=0)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
    return _save(fig, Path(path))


def plot_label_class_balance(class_balance: pd.DataFrame, path: str | Path) -> Path:
    """Show all observed classes instead of reporting only majority accuracy."""

    with plt.rc_context(_RC):
        fig, ax = plt.subplots(figsize=(8.5, 4.5))
        sns.barplot(
            data=class_balance,
            x="fraction",
            y="label",
            hue="class",
            palette="colorblind",
            errorbar=None,
            ax=ax,
        )
        ax.set_xlim(0, 1)
        ax.set_xlabel("fraction of observable synthetic events")
        ax.set_ylabel("categorical label")
        ax.set_title("Synthetic categorical-label balance (all classes shown)")
        ax.legend(loc="lower right", title="class")
        ax.grid(True, axis="x", zorder=0)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
    return _save(fig, Path(path))


def plot_label_temporal_stability(
    temporal_stability: pd.DataFrame,
    path: str | Path,
    summary: pd.DataFrame | None = None,
) -> Path:
    """Plot period means to expose distribution drift across chronological blocks."""

    frame = temporal_stability.copy()
    y_column = "mean"
    y_label = "label mean (native units)"
    title = "Synthetic label means across equal-session chronological blocks"
    if summary is not None:
        reference = summary.set_index("label")[["mean", "standard_deviation"]]
        frame = frame.join(reference, on="label", rsuffix="_full")
        scale = frame["standard_deviation_full"].where(frame["standard_deviation_full"].ne(0))
        frame["standardized_mean_shift"] = (
            (frame["mean"] - frame["mean_full"]).divide(scale).fillna(0.0)
        )
        y_column = "standardized_mean_shift"
        y_label = "mean shift from full sample (standard deviations)"
        title = "Synthetic label stability across equal-session chronological blocks"
    with plt.rc_context(_RC):
        fig, ax = plt.subplots(figsize=(9, 4.8))
        sns.lineplot(
            data=frame,
            x="period",
            y=y_column,
            hue="label",
            marker="o",
            palette="colorblind",
            estimator=None,
            linewidth=1.6,
            ax=ax,
        )
        ax.axhline(0, color=GRID, linewidth=0.9)
        ax.set_xlabel("chronological block (1 = earliest)")
        ax.set_ylabel(y_label)
        ax.set_title(title)
        ax.legend(loc="center left", bbox_to_anchor=(1.01, 0.5), title=None)
        _style(ax)
    return _save(fig, Path(path))


def plot_label_parameter_sensitivity(sensitivity: pd.DataFrame, path: str | Path) -> Path:
    """Show how categorical outcomes change as predeclared magnitudes move."""

    frame = sensitivity.dropna(subset=["class_flip_rate"]).copy()
    varied = frame.groupby("label")["parameter_scale"].transform("nunique").gt(1)
    frame = frame.loc[varied]
    with plt.rc_context(_RC):
        fig, ax = plt.subplots(figsize=(8, 4.2))
        sns.lineplot(
            data=frame,
            x="parameter_scale",
            y="class_flip_rate",
            hue="label",
            marker="o",
            palette="colorblind",
            estimator=None,
            linewidth=1.8,
            ax=ax,
        )
        ax.axvline(1.0, color=GRID, linewidth=1.0, linestyle="--")
        ax.set_ylim(0, max(0.05, float(frame["class_flip_rate"].max()) * 1.15))
        ax.set_xlabel("predeclared threshold / barrier scale (1.0 = baseline)")
        ax.set_ylabel("class flip rate vs baseline")
        ax.set_title("Synthetic categorical-label parameter sensitivity")
        ax.legend(loc="upper right", title=None)
        _style(ax)
    return _save(fig, Path(path))


def save_label_diagnostic_plots(
    diagnostics: LabelDiagnostics, output_dir: str | Path
) -> list[Path]:
    """Render the complete deterministic label-evidence plot set."""

    destination = Path(output_dir)
    return [
        plot_label_dependence(diagnostics.summary, destination / "dependence.png"),
        plot_label_class_balance(diagnostics.class_balance, destination / "class_balance.png"),
        plot_label_temporal_stability(
            diagnostics.temporal_stability,
            destination / "temporal_stability.png",
            diagnostics.summary,
        ),
        plot_label_parameter_sensitivity(
            diagnostics.parameter_sensitivity,
            destination / "parameter_sensitivity.png",
        ),
    ]
