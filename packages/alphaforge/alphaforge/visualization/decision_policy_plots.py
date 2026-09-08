"""Seaborn evidence for the bounded abstention-policy study."""

from __future__ import annotations

from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
import seaborn as sns

matplotlib.use("Agg")
from matplotlib import pyplot as plt  # noqa: E402

_METRIC_COLUMNS = frozenset(
    {
        "policy",
        "coverage",
        "selective_risk",
        "mean_net_value",
        "missed_opportunity",
        "mean_turnover_per_period",
        "peak_capacity_demand",
    }
)
_REASON_COLUMNS = frozenset({"reason", "count"})


def plot_decision_policy_study(
    metrics: pd.DataFrame,
    reason_counts: pd.DataFrame,
    destination: str | Path,
    *,
    observation_count: int,
    seed: int,
) -> Path:
    """Render aggregate synthetic evidence without accepting row-level inputs."""
    missing_metrics = _METRIC_COLUMNS - set(metrics)
    missing_reasons = _REASON_COLUMNS - set(reason_counts)
    if missing_metrics:
        raise ValueError(f"metrics missing columns: {sorted(missing_metrics)}")
    if missing_reasons:
        raise ValueError(f"reason_counts missing columns: {sorted(missing_reasons)}")
    if len(metrics) != 3:
        raise ValueError("decision-policy plot requires exactly three aggregate policies")
    if not isinstance(observation_count, int) or observation_count < 1:
        raise ValueError("observation_count must be a positive integer")
    if not isinstance(seed, int) or seed < 0:
        raise ValueError("seed must be a non-negative integer")
    for column in _METRIC_COLUMNS - {"policy", "selective_risk"}:
        values = pd.to_numeric(metrics[column], errors="coerce").to_numpy(dtype=float)
        if not np.isfinite(values).all():
            raise ValueError(f"{column} must contain only finite aggregate values")
    risk = pd.to_numeric(metrics["selective_risk"], errors="coerce").to_numpy(dtype=float)
    if np.isinf(risk).any():
        raise ValueError("selective_risk cannot contain infinity")

    output = Path(destination)
    sns.set_theme(style="whitegrid", context="talk", palette="colorblind")
    palette = dict(
        zip(
            metrics["policy"].astype(str),
            sns.color_palette("colorblind", n_colors=len(metrics)),
            strict=True,
        )
    )
    figure, axes = plt.subplots(2, 3, figsize=(20, 12))

    defined_risk = metrics.loc[metrics["selective_risk"].notna()]
    sns.scatterplot(
        data=defined_risk,
        x="coverage",
        y="selective_risk",
        hue="policy",
        palette=palette,
        s=180,
        ax=axes[0, 0],
    )
    axes[0, 0].set(
        title="Coverage–risk tradeoff",
        xlabel="Coverage (eligible fraction)",
        ylabel="Conditional loss frequency",
        xlim=(-0.03, 1.03),
        ylim=(-0.03, 1.03),
    )
    axes[0, 0].annotate(
        "never trade:\nrisk undefined",
        xy=(0.0, 0.0),
        xytext=(0.08, 0.12),
        arrowprops={"arrowstyle": "->", "color": "0.35"},
        fontsize=10,
    )

    panels = (
        ("mean_net_value", "Mean realized net value", "Decimal return / opportunity"),
        ("missed_opportunity", "Missed positive opportunity", "Decimal return / opportunity"),
        ("mean_turnover_per_period", "Turnover demand", "Turnover units / period"),
        ("peak_capacity_demand", "Peak capacity demand", "Fraction of period capacity"),
    )
    for axis, (column, title, ylabel) in zip(axes.flat[1:5], panels, strict=True):
        sns.barplot(
            data=metrics,
            x="policy",
            y=column,
            hue="policy",
            palette=palette,
            legend=False,
            ax=axis,
        )
        axis.axhline(0.0, color="black", linewidth=1)
        axis.set(title=title, xlabel="", ylabel=ylabel)
        axis.tick_params(axis="x", rotation=18)

    if reason_counts.empty:
        axes[1, 2].text(0.5, 0.5, "No abstention reasons", ha="center", va="center")
        axes[1, 2].set_axis_off()
    else:
        sns.barplot(
            data=reason_counts,
            x="count",
            y="reason",
            color=sns.color_palette("colorblind")[0],
            ax=axes[1, 2],
        )
        axes[1, 2].set(
            title="Policy abstention reasons",
            xlabel="Decision count (non-exclusive)",
            ylabel="",
        )

    figure.suptitle(
        "Cost/uncertainty abstention — "
        f"synthetic engineering evidence (n={observation_count:,}, seed={seed}, CPU-only)",
        fontsize=17,
    )
    figure.text(
        0.5,
        0.01,
        "Eligibility decisions only; realized outcomes are evaluation labels and no orders "
        "or market claims are produced.",
        ha="center",
        fontsize=11,
    )
    figure.tight_layout(rect=(0, 0.035, 1, 0.96))
    figure.savefig(output, dpi=160, bbox_inches="tight")
    plt.close(figure)
    return output
