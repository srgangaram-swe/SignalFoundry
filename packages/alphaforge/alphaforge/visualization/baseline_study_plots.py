"""Seaborn visual evidence for the governed Sprint 2 baseline study."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

_PALETTE = "colorblind"


def _prepare(path: str | Path) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    sns.set_theme(context="notebook", style="whitegrid", palette=_PALETTE)
    return destination


def plot_fold_rank_ic(frame: pd.DataFrame, path: str | Path) -> Path:
    """Plot every matched walk-forward fold, including unfavorable values."""

    destination = _prepare(path)
    model_order = frame["model"].astype(str).drop_duplicates().tolist()
    figure, axis = plt.subplots(figsize=(11, 6.5))
    sns.boxplot(
        data=frame,
        x="rank_ic",
        y="model",
        hue="model",
        order=model_order,
        hue_order=model_order,
        palette=_PALETTE,
        legend=False,
        whis=(0, 100),
        width=0.55,
        ax=axis,
    )
    sns.stripplot(
        data=frame,
        x="rank_ic",
        y="model",
        order=model_order,
        color="#222222",
        alpha=0.72,
        size=5,
        jitter=0.10,
        ax=axis,
    )
    axis.set_yticks(range(len(model_order)), labels=model_order)
    axis.tick_params(axis="y", colors="#222222")
    axis.axvline(0.0, color="#444444", linestyle="--", linewidth=1.2)
    axis.set(
        title="Development-only rank IC across matched walk-forward folds",
        xlabel="Spearman rank IC (decimal correlation; zero is no association)",
        ylabel="Pre-registered candidate",
    )
    axis.text(
        0.0,
        -0.16,
        "Points are every fold; boxes span the full observed range. "
        "Stale WIKI engineering data, not paper/live evidence.",
        transform=axis.transAxes,
        fontsize=9,
    )
    figure.tight_layout()
    figure.subplots_adjust(left=0.24, bottom=0.20)
    figure.savefig(destination, dpi=180)
    plt.close(figure)
    return destination


def plot_costed_returns(frame: pd.DataFrame, path: str | Path) -> Path:
    """Compare gross and net annualized development backtest returns."""

    destination = _prepare(path)
    model_order = frame["model"].astype(str).tolist()
    long = frame.melt(
        id_vars=["model"],
        value_vars=["gross_annual_return", "net_annual_return"],
        var_name="return_basis",
        value_name="annualized_return",
    )
    long["return_basis"] = long["return_basis"].map(
        {
            "gross_annual_return": "Gross (before modeled costs)",
            "net_annual_return": "Net (after modeled costs)",
        }
    )
    figure, axis = plt.subplots(figsize=(11, 6.5))
    sns.barplot(
        data=long,
        x="annualized_return",
        y="model",
        hue="return_basis",
        order=model_order,
        palette=_PALETTE,
        errorbar=None,
        ax=axis,
    )
    axis.set_yticks(range(len(model_order)), labels=model_order)
    axis.tick_params(axis="y", colors="#222222")
    axis.axvline(0.0, color="#444444", linestyle="--", linewidth=1.2)
    axis.set(
        title="Development OOS economics before and after declared trading costs",
        xlabel="Annualized return (decimal; historical backtest)",
        ylabel="Pre-registered candidate",
    )
    axis.legend(title=None, loc="best")
    axis.text(
        0.0,
        -0.16,
        "One cost policy and portfolio rule is applied to every candidate. "
        "Returns are simulated and do not predict profit.",
        transform=axis.transAxes,
        fontsize=9,
    )
    figure.tight_layout()
    figure.subplots_adjust(left=0.24, bottom=0.20)
    figure.savefig(destination, dpi=180)
    plt.close(figure)
    return destination


def plot_multiplicity(frame: pd.DataFrame, path: str | Path, *, alpha: float) -> Path:
    """Show all adjusted p-values and the frozen family-wise threshold."""

    destination = _prepare(path)
    ordered = frame.sort_values(["adjusted_p_value", "model"], kind="stable")
    figure, axis = plt.subplots(figsize=(11, 6.5))
    sns.barplot(
        data=ordered,
        x="adjusted_p_value",
        y="model",
        hue="multiplicity_result",
        palette={"survives correction": "#0173b2", "does not survive": "#de8f05"},
        errorbar=None,
        dodge=False,
        ax=axis,
    )
    axis.axvline(alpha, color="#222222", linestyle="--", linewidth=1.4, label=f"alpha={alpha:g}")
    axis.set(
        title="Holm-Bonferroni correction across the complete seven-candidate family",
        xlabel="Multiplicity-adjusted one-sided p-value",
        ylabel="Pre-registered candidate",
        xlim=(0.0, 1.0),
    )
    handles, labels = axis.get_legend_handles_labels()
    axis.legend(handles=handles, labels=labels, title=None, loc="lower right")
    axis.text(
        0.0,
        -0.16,
        "Tests use matched development-fold rank-IC differences. "
        "The final holdout was not used to form these p-values.",
        transform=axis.transAxes,
        fontsize=9,
    )
    figure.tight_layout()
    figure.savefig(destination, dpi=180, bbox_inches="tight")
    plt.close(figure)
    return destination


def plot_compute_accounting(frame: pd.DataFrame, path: str | Path) -> Path:
    """Plot bounded estimator-iteration accounting and disclose warnings."""

    destination = _prepare(path)
    figure, axis = plt.subplots(figsize=(11, 6.5))
    sns.barplot(
        data=frame,
        x="training_iterations",
        y="model",
        hue="training_status",
        palette={"clean": "#029e73", "warning recorded": "#d55e00"},
        errorbar=None,
        dodge=False,
        ax=axis,
    )
    axis.set_xscale("symlog", linthresh=1.0)
    axis.set(
        title="Recorded estimator iterations across all development folds",
        xlabel="Summed reported estimator iterations (symlog; zero means not applicable)",
        ylabel="Pre-registered candidate",
    )
    axis.legend(title=None, loc="best")
    axis.text(
        0.0,
        -0.16,
        "Iteration counts are bounded audit proxies, not comparable FLOPs. "
        "Total wall/CPU/RSS is reported separately in summary.json.",
        transform=axis.transAxes,
        fontsize=9,
    )
    figure.tight_layout()
    figure.savefig(destination, dpi=180, bbox_inches="tight")
    plt.close(figure)
    return destination
