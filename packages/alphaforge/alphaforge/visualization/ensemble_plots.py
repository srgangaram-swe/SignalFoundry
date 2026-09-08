"""Seaborn visual evidence for the governed ensemble engineering study."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

_RC = matplotlib.RcParams(
    {
        "figure.facecolor": "#fcfcfb",
        "axes.facecolor": "#fcfcfb",
        "savefig.facecolor": "#fcfcfb",
        "axes.edgecolor": "#e5e4e1",
        "axes.labelcolor": "#52514e",
        "axes.titlecolor": "#0b0b0b",
        "grid.color": "#e5e4e1",
        "legend.frameon": False,
    }
)


def _save(figure: plt.Figure, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.tight_layout()
    figure.savefig(path, dpi=150)
    plt.close(figure)
    return path


def plot_ensemble_evidence(
    summary: pd.DataFrame,
    correlations: pd.DataFrame,
    marginal: pd.DataFrame,
    path: str | Path,
    *,
    generator_seed: int,
    bootstrap_seed: int,
    holdout_rows: int,
    transaction_cost_bps: float,
) -> Path:
    """Plot accuracy, dependence, contribution, and cost evidence.

    Every mark is derived from aggregate synthetic holdout diagnostics.  No
    row-level prediction, target, or fitted model is embedded in the image.
    """

    method_order = summary.sort_values(["kind", "model"])["model"].tolist()
    error = correlations[correlations["kind"] == "error"].pivot(
        index="model_a",
        columns="model_b",
        values="correlation",
    )
    contribution = marginal.groupby("model", as_index=False)["mse_increase_without_expert"].mean()
    with plt.rc_context(_RC):
        sns.set_theme(style="whitegrid", context="notebook", palette="colorblind")
        figure, axes = plt.subplots(2, 2, figsize=(13.0, 9.0))

        relative = summary.copy()
        relative["MSE relative to best single"] = relative["mse"] / relative["best_single_mse"]
        relative["MSE CI lower relative"] = relative["mse_ci_lower"] / relative["best_single_mse"]
        relative["MSE CI upper relative"] = relative["mse_ci_upper"] / relative["best_single_mse"]
        sns.barplot(
            data=relative,
            x="MSE relative to best single",
            y="model",
            hue="kind",
            order=method_order,
            dodge=False,
            palette="colorblind",
            ax=axes[0, 0],
        )
        ordered_relative = relative.set_index("model").loc[method_order]
        intervals = zip(
            ordered_relative["MSE CI lower relative"],
            ordered_relative["MSE CI upper relative"],
            strict=True,
        )
        for position, (lower, upper) in enumerate(intervals):
            axes[0, 0].hlines(
                position,
                lower,
                upper,
                color="#202020",
                linewidth=1.2,
            )
            axes[0, 0].vlines(
                [lower, upper],
                position - 0.08,
                position + 0.08,
                color="#202020",
                linewidth=1.0,
            )
        axes[0, 0].axvline(1.0, color="#444444", linestyle="--", linewidth=1.0)
        confidence = 100.0 * float(summary["bootstrap_confidence_level"].iloc[0])
        axes[0, 0].set_title(
            f"Untouched synthetic holdout MSE with {confidence:g}% block intervals"
        )
        axes[0, 0].set_ylabel("")

        sns.heatmap(
            error,
            cmap="vlag",
            center=0.0,
            vmin=-1.0,
            vmax=1.0,
            square=True,
            cbar_kws={"label": "error correlation"},
            ax=axes[0, 1],
        )
        axes[0, 1].set_title("Holdout error dependence")
        axes[0, 1].set_xlabel("")
        axes[0, 1].set_ylabel("")

        sns.barplot(
            data=contribution,
            x="mse_increase_without_expert",
            y="model",
            color=sns.color_palette("colorblind")[2],
            ax=axes[1, 0],
        )
        axes[1, 0].axvline(0.0, color="#444444", linewidth=0.8)
        axes[1, 0].set_title("Mean marginal contribution across experts")
        axes[1, 0].set_xlabel("MSE(without expert) − MSE(full)")
        axes[1, 0].set_ylabel("")

        sns.scatterplot(
            data=summary,
            x="mean_turnover",
            y="net_directional_return",
            hue="kind",
            style="kind",
            s=80,
            palette="colorblind",
            ax=axes[1, 1],
        )
        annotation_offsets = {
            "bayesian": (5, 10),
            "stable": (5, -12),
            "defensive": (5, -11),
            "redundant": (-45, 8),
            "dynamic": (-35, -12),
            "stacking": (5, 10),
            "static": (6, -10),
        }
        for row in summary.itertuples(index=False):
            axes[1, 1].annotate(
                row.model,
                (row.mean_turnover, row.net_directional_return),
                xytext=annotation_offsets.get(row.model, (4, 3)),
                textcoords="offset points",
                fontsize=7,
            )
        axes[1, 1].axhline(0.0, color="#444444", linewidth=0.8)
        axes[1, 1].set_title(f"Turnover and {transaction_cost_bps:g} bps synthetic cost diagnostic")
        axes[1, 1].set_xlabel("mean one-way turnover")
        axes[1, 1].set_ylabel("mean signed return after simplified cost")

        figure.suptitle(
            "Governed ensembles — deterministic synthetic engineering evidence\n"
            f"generator seed={generator_seed}, bootstrap seed={bootstrap_seed}, "
            f"holdout rows={holdout_rows}, "
            f"{int(summary['bootstrap_resamples'].iloc[0])} resamples × "
            f"{int(summary['bootstrap_block_dates'].iloc[0])}-date circular blocks; "
            "not market or trading evidence",
            fontsize=13,
        )
    return _save(figure, Path(path))


__all__ = ["plot_ensemble_evidence"]
