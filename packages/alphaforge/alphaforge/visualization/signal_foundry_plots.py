"""Seaborn plots for redistribution-safe Signal Foundry sprint evidence."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from matplotlib.ticker import PercentFormatter

from alphaforge.visualization.plots import _RC, BLUE, RED, TEXT_2, _save, _style


def plot_readiness_gates(gates: pd.DataFrame, path: str | Path) -> Path:
    """Plot every pre-registered gate as an equally visible pass/fail result."""
    required = {"gate", "passed"}
    if required - set(gates):
        raise ValueError(f"readiness gates missing columns: {sorted(required - set(gates))}")
    frame = gates.copy().sort_values("gate", kind="stable")
    frame["status"] = frame["passed"].map({True: "PASS", False: "FAIL"})
    frame["bar"] = 1.0
    with matplotlib.rc_context(_RC):
        fig, ax = plt.subplots(figsize=(9.0, max(4.2, 0.42 * len(frame) + 1.5)))
        sns.barplot(
            data=frame,
            x="bar",
            y="gate",
            hue="status",
            hue_order=["PASS", "FAIL"],
            palette={"PASS": BLUE, "FAIL": RED},
            dodge=False,
            ax=ax,
        )
        ax.set(
            title="Pre-registered paper-readiness gates — stale WIKI engineering bootstrap",
            xlabel="",
            ylabel="Gate",
            xlim=(0.0, 1.06),
        )
        ax.set_xticks([])
        _style(ax)
        ax.grid(False)
        for row, status in enumerate(frame["status"]):
            ax.text(
                0.98,
                row,
                status,
                ha="right",
                va="center",
                color="white",
                fontsize=8,
                fontweight="bold",
            )
        ax.legend(title="", loc="lower right")
        fig.text(
            0.01,
            0.005,
            "Historical backtest gate evidence only. Any failed gate forces NOT_READY.",
            color=TEXT_2,
            fontsize=8,
        )
        return _save(fig, Path(path))


def plot_scenario_returns(scenarios: pd.DataFrame, path: str | Path) -> Path:
    """Compare primary and adversarial annualized returns without hiding losses."""
    required = {"scenario", "annual_return"}
    if required - set(scenarios):
        raise ValueError(f"scenario evidence missing columns: {sorted(required - set(scenarios))}")
    frame = scenarios.copy()
    frame["sign"] = frame["annual_return"].map(
        lambda value: "nonnegative" if float(value) >= 0.0 else "negative"
    )
    with matplotlib.rc_context(_RC):
        fig, ax = plt.subplots(figsize=(10.0, max(4.5, 0.42 * len(frame) + 1.7)))
        sns.barplot(
            data=frame,
            x="annual_return",
            y="scenario",
            hue="sign",
            hue_order=["nonnegative", "negative"],
            palette={"nonnegative": BLUE, "negative": RED},
            dodge=False,
            ax=ax,
        )
        ax.axvline(0.0, color=TEXT_2, linewidth=1.0)
        ax.xaxis.set_major_formatter(PercentFormatter(1.0))
        ax.set(
            title="Primary and adversarial annualized returns — untouched 2017–2018 holdout",
            xlabel="Annualized net return",
            ylabel="Scenario",
        )
        _style(ax)
        ax.legend(title="")
        fig.text(
            0.01,
            0.005,
            "Stale, current-vintage WIKI data; results are backtested, not paper or live returns.",
            color=TEXT_2,
            fontsize=8,
        )
        return _save(fig, Path(path))


def plot_capacity_sensitivity(
    capacity: pd.DataFrame,
    path: str | Path,
    *,
    minimum_fill_ratio: float,
) -> Path:
    """Plot fill-ratio sensitivity across predeclared AUM scenarios."""
    required = {"scenario_aum", "fill_ratio"}
    if required - set(capacity):
        raise ValueError(f"capacity evidence missing columns: {sorted(required - set(capacity))}")
    frame = capacity.copy().sort_values("scenario_aum", kind="stable")
    frame["scenario_aum_millions"] = frame["scenario_aum"].astype(float) / 1_000_000.0
    with matplotlib.rc_context(_RC):
        fig, ax = plt.subplots(figsize=(8.0, 4.5))
        sns.lineplot(
            data=frame,
            x="scenario_aum_millions",
            y="fill_ratio",
            marker="o",
            color=BLUE,
            linewidth=2.0,
            ax=ax,
        )
        ax.axhline(
            minimum_fill_ratio,
            color=RED,
            linestyle="--",
            linewidth=1.2,
            label=f"pre-registered floor ({minimum_fill_ratio:.0%})",
        )
        ax.yaxis.set_major_formatter(PercentFormatter(1.0))
        ax.set(
            title="Modeled capacity sensitivity — ex-post liquidity anchor",
            xlabel="Scenario AUM (USD millions)",
            ylabel="Modeled fill ratio",
            ylim=(0.0, 1.05),
        )
        _style(ax)
        ax.legend()
        fig.text(
            0.01,
            0.005,
            "Sensitivity curve, not a deployable-capital forecast; daily bars omit queue position.",
            color=TEXT_2,
            fontsize=8,
        )
        return _save(fig, Path(path))
