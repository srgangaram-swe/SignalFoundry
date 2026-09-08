"""Markdown report generation for AlphaForge runs."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

DISCLAIMER = (
    "AlphaForge is an educational quantitative research and ML engineering project. "
    "It is not financial advice, does not guarantee profitability, and should not be "
    "used to trade real money without professional review, additional validation, and "
    "appropriate risk controls. Backtests are not live results and may not predict "
    "future performance."
)


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text()) if path.exists() else {}


def _read_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path) if path.exists() else pd.DataFrame()


def _table(frame: pd.DataFrame, max_rows: int = 12) -> str:
    if frame.empty:
        return "_No rows available._"
    return "```\n" + frame.head(max_rows).to_string(index=False) + "\n```"


def _kv_lines(d: dict) -> list[str]:
    return [
        f"- **{k}**: {v:.6g}" if isinstance(v, (int, float)) else f"- **{k}**: {v}"
        for k, v in d.items()
    ]


def write_markdown_report(run_dir: str | Path, output_path: str | Path | None = None) -> Path:
    """Write a compact research report for a completed synthetic or real run."""
    run_dir = Path(run_dir)
    output_path = Path(output_path) if output_path else run_dir / "report.md"
    summary = _read_json(run_dir / "backtest_summary.json")
    overfit = _read_json(run_dir / "overfitting.json")
    model_metrics = _read_csv(run_dir / "model_metrics.csv")
    quantiles = _read_csv(run_dir / "quantile_returns.csv")
    ic_table = _read_csv(run_dir / "ic_summary.csv")
    decay = _read_csv(run_dir / "ic_decay.csv")
    regimes = _read_csv(run_dir / "regime_performance.csv")
    fills = _read_csv(run_dir / "fills.csv")
    attribution = _read_csv(run_dir / "pnl_attribution.csv")
    capacity = _read_csv(run_dir / "capacity_curve.csv")
    capacity_diagnostics = _read_json(run_dir / "capacity_diagnostics.json")

    lines = ["# AlphaForge Research Report", "", DISCLAIMER, "", "## Backtest Summary", ""]
    lines.extend(_kv_lines(summary) if summary else ["_Backtest summary not found._"])

    lines += ["", "## Overfitting Diagnostics", ""]
    diag = {}
    if overfit:
        diag["probability_of_backtest_overfitting"] = overfit.get("pbo")
        diag["pbo_combinations"] = overfit.get("n_combinations")
        diag["pbo_strategies_compared"] = overfit.get("n_strategies")
    for key in ("deflated_sharpe_prob", "psr_vs_zero", "expected_max_sharpe", "n_trials"):
        if key in summary:
            diag[key] = summary[key]
    if diag:
        lines.extend(_kv_lines(diag))
        lines += [
            "",
            "_PBO is the probability the in-sample best model underperforms the median "
            "out-of-sample (CSCV). The deflated Sharpe probability is P(true Sharpe > 0) "
            "after correcting for multiple testing, sample length, skew, and fat tails._",
        ]
    else:
        lines.append("_No overfitting diagnostics available._")

    lines += ["", "## Execution and Accounting", ""]
    execution_keys = (
        "execution_orders",
        "execution_fill_ratio",
        "execution_partial_fill_rate",
        "execution_rejected_rate",
        "total_trading_cost_dollars",
        "total_commission_dollars",
        "total_exchange_fee_dollars",
        "total_spread_cost_dollars",
        "total_slippage_cost_dollars",
        "total_impact_cost_dollars",
        "total_financing_cost_dollars",
        "total_borrow_cost_dollars",
    )
    execution_summary = {key: summary[key] for key in execution_keys if key in summary}
    if execution_summary:
        lines.extend(_kv_lines(execution_summary))
    else:
        lines.append("_No execution summary available._")
    lines += [
        "",
        "_Targets are decided at a session close and fill no earlier than a future "
        "open. Shares and cash are tracked in a self-financing ledger; positions "
        "drift between explicit, costed rebalances._",
    ]
    if not fills.empty:
        fill_columns = [
            column
            for column in (
                "decision_date",
                "fill_date",
                "symbol",
                "status",
                "requested_notional",
                "traded_notional",
                "participation_rate",
                "total_cost",
            )
            if column in fills
        ]
        lines += ["", "### Recent fills", "", _table(fills[fill_columns].tail(12))]
    if not attribution.empty:
        contribution = (
            attribution.groupby("symbol", as_index=False)[["market_pnl", "trading_cost", "net_pnl"]]
            .sum()
            .sort_values("net_pnl", ascending=False)
        )
        lines += ["", "### P&L attribution by symbol", "", _table(contribution)]

    lines += ["", "## Capacity Sensitivity", ""]
    if capacity.empty:
        lines.append("_No capacity sensitivity available._")
    else:
        capacity_columns = [
            column
            for column in (
                "scenario_aum",
                "fill_ratio",
                "capacity_constrained_fraction",
                "participation_p95",
                "modeled_cost_bps_per_traded_notional",
            )
            if column in capacity
        ]
        lines.append(_table(capacity[capacity_columns]))
        assumptions = capacity_diagnostics.get("assumptions", [])
        if assumptions:
            lines += ["", *[f"- {assumption}" for assumption in assumptions]]

    lines += ["", "## Information Coefficient by Model", "", _table(ic_table)]
    if not decay.empty:
        lines += ["", "## IC Decay (selected model)", "", _table(decay)]
    if not regimes.empty:
        lines += ["", "## Regime-Conditional Performance", "", _table(regimes)]
    lines += [
        "",
        "## Model Metrics (per walk-forward window)",
        "",
        _table(model_metrics),
        "",
        "## Prediction Quantiles",
        "",
        _table(quantiles),
        "",
    ]

    plots_dir = run_dir / "plots"
    if plots_dir.exists():
        pngs = sorted(plots_dir.glob("*.png"))
        if pngs:
            lines += ["", "## Evaluation Plots", ""]
            for png in pngs:
                title = png.stem.replace("_", " ")
                lines.append(f"![{title}](plots/{png.name})")
                lines.append("")

    output_path.write_text("\n".join(lines))
    return output_path
