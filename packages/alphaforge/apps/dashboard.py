from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pandas as pd

if TYPE_CHECKING:
    from alphaforge.service import BacktestResult

# Wong colorblind-safe palette: strategy (blue), benchmark (orange), drawdown (vermillion).
_STRATEGY_COLOR = "#0072B2"
_BENCHMARK_COLOR = "#E69F00"
_DRAWDOWN_COLOR = "#D55E00"


def _latest_run() -> Path | None:
    pointer = Path("runs/latest_run.txt")
    return Path(pointer.read_text().strip()) if pointer.exists() else None


def _csv(run_dir: Path, name: str) -> pd.DataFrame | None:
    path = run_dir / name
    return pd.read_csv(path) if path.exists() else None


# --- Pure, testable presentation helpers (SF-S2-MR10c) ------------------------


def format_metric_tiles(result: BacktestResult) -> list[tuple[str, str]]:
    """Headline metric tiles as (label, formatted-value) pairs."""
    metrics = result.headline.metrics

    def pct(value: float | None) -> str:
        return "—" if value is None else f"{value:.2%}"

    def num(value: float | None) -> str:
        return "—" if value is None else f"{value:.2f}"

    return [
        ("Total return", pct(metrics.get("total_return"))),
        ("Sharpe", num(metrics.get("sharpe"))),
        ("Sortino", num(metrics.get("sortino"))),
        ("Max drawdown", pct(metrics.get("max_drawdown"))),
        ("Annual vol", pct(metrics.get("annual_volatility"))),
        ("Avg turnover", pct(metrics.get("average_turnover"))),
    ]


def build_comparison_table(result: BacktestResult) -> pd.DataFrame:
    """Model-vs-baselines metric table (headline row first)."""
    columns = [
        "name",
        "is_baseline",
        "total_return",
        "sharpe",
        "sortino",
        "max_drawdown",
        "annual_volatility",
        "hit_rate",
        "average_turnover",
    ]
    frame = pd.DataFrame(result.comparison)
    present = [column for column in columns if column in frame.columns]
    return frame[present]


def build_equity_figure(result: BacktestResult) -> Any:
    """Plotly cumulative-return vs benchmark figure for the headline strategy."""
    import plotly.graph_objects as go

    points = result.headline.equity_curve
    dates = [point["date"] for point in points]
    figure = go.Figure()
    figure.add_trace(
        go.Scatter(
            x=dates,
            y=[point["strategy_cum"] for point in points],
            name=result.headline.name,
            line={"color": _STRATEGY_COLOR, "width": 2},
        )
    )
    figure.add_trace(
        go.Scatter(
            x=dates,
            y=[point["benchmark_cum"] for point in points],
            name=f"benchmark ({result.benchmark_symbol})",
            line={"color": _BENCHMARK_COLOR, "width": 2, "dash": "dash"},
        )
    )
    figure.update_layout(
        title="Cumulative return vs benchmark (out-of-sample, simulated)",
        xaxis_title="Date",
        yaxis_title="Cumulative return",
        yaxis={"tickformat": ".0%"},
        legend={"orientation": "h"},
        margin={"t": 48, "b": 32, "l": 8, "r": 8},
    )
    return figure


def build_drawdown_figure(result: BacktestResult) -> Any:
    """Plotly drawdown area figure for the headline strategy."""
    import plotly.graph_objects as go

    points = result.headline.equity_curve
    figure = go.Figure(
        go.Scatter(
            x=[point["date"] for point in points],
            y=[point["drawdown"] for point in points],
            fill="tozeroy",
            name="drawdown",
            line={"color": _DRAWDOWN_COLOR},
        )
    )
    figure.update_layout(
        title="Drawdown",
        xaxis_title="Date",
        yaxis_title="Drawdown",
        yaxis={"tickformat": ".0%"},
        margin={"t": 48, "b": 32, "l": 8, "r": 8},
    )
    return figure


def _render_backtest_workspace(st: Any) -> None:
    """Configure → run → evidence backtesting workspace."""
    from alphaforge.service import (
        DISCLAIMER,
        BacktestRequest,
        BacktestServiceError,
        available_baselines,
        available_strategy_models,
        discover_bundles,
        run_backtest_service,
    )

    st.sidebar.header("Backtest configuration")
    data_source = st.sidebar.selectbox("Data source", ["synthetic", "signal_foundry"])
    bundle_dir: str | None = None
    benchmark = "BENCH"
    n_symbols, n_days = 8, 600
    if data_source == "signal_foundry":
        bundles = discover_bundles()
        if not bundles:
            st.warning("No Signal Foundry bundles found under data/signal-foundry-bundles/.")
            return
        chosen = st.sidebar.selectbox("Bundle (from Signalattice)", bundles)
        bundle_dir = f"data/signal-foundry-bundles/{chosen}"
        benchmark = st.sidebar.text_input("Benchmark symbol", "SPY")
    else:
        n_symbols = st.sidebar.slider("Symbols", 3, 30, 8)
        n_days = st.sidebar.slider("Trading days", 320, 1500, 600, step=20)

    models = available_strategy_models()
    model = st.sidebar.selectbox(
        "Model", models, index=models.index("random_forest") if "random_forest" in models else 0
    )
    baselines = st.sidebar.multiselect(
        "Baselines to compare",
        available_baselines(),
        default=["zero_baseline", "historical_mean", "momentum_baseline"],
    )
    strategy = st.sidebar.selectbox(
        "Signal strategy",
        ["long_short", "long_only_topk", "rank_weighted", "confidence_weighted"],
    )
    cost_bps = st.sidebar.slider("Transaction cost (bps)", 0.0, 20.0, 1.0, step=0.5)
    seed = int(st.sidebar.number_input("Seed", min_value=0, value=42, step=1))
    run = st.sidebar.button("Run backtest", type="primary")

    st.subheader("Interactive backtest")
    st.caption(DISCLAIMER)
    if not run:
        st.info("Configure on the left, then click **Run backtest**.")
        return

    try:
        request = BacktestRequest(
            data_source=data_source,
            bundle_dir=bundle_dir,
            n_symbols=n_symbols,
            n_days=n_days,
            benchmark_symbol=benchmark,
            model=model,
            baselines=tuple(baselines),
            strategy=strategy,
            cost_bps=cost_bps,
            seed=seed,
        )
        with st.spinner("Running leakage-safe walk-forward backtest…"):
            result = run_backtest_service(request)
    except BacktestServiceError as exc:
        st.error(f"Invalid configuration: {exc}")
        return

    tiles = format_metric_tiles(result)
    columns = st.columns(len(tiles))
    for column, (label, value) in zip(columns, tiles, strict=True):
        column.metric(label, value)

    st.plotly_chart(build_equity_figure(result), use_container_width=True)
    st.plotly_chart(build_drawdown_figure(result), use_container_width=True)

    st.subheader("Model vs baselines")
    st.caption("Same data, splits, and costs for every strategy — the honest comparison.")
    st.dataframe(build_comparison_table(result), use_container_width=True)

    if result.trades_tail:
        st.subheader("Recent simulated fills")
        st.dataframe(pd.DataFrame(result.trades_tail), use_container_width=True)

    repro = result.reproducibility
    st.caption(
        f"Reproduce — data: `{result.data_id}` · seed: {repro['seed']} · "
        f"config: `{repro['config_hash']}` · walk-forward windows: {result.n_windows}."
    )


def main() -> None:
    try:
        import streamlit as st
    except ImportError as exc:  # pragma: no cover
        raise SystemExit("Install app extras with: pip install -e '.[app]'") from exc

    st.set_page_config(page_title="AlphaForge", layout="wide")
    st.title("AlphaForge")
    st.caption("Educational research platform — simulated results only, not financial advice.")

    view = st.sidebar.radio("View", ["Run a backtest", "Latest run"], index=0)
    if view == "Run a backtest":
        _render_backtest_workspace(st)
        return

    run_dir = _latest_run()
    if run_dir is None:
        st.info("No run found. Run `make demo` first.")
        return

    st.caption(f"Run: {run_dir}")
    summary = {}
    summary_path = run_dir / "backtest_summary.json"
    if summary_path.exists():
        summary = json.loads(summary_path.read_text())
    overfit_path = run_dir / "overfitting.json"
    overfit = json.loads(overfit_path.read_text()) if overfit_path.exists() else {}

    cols = st.columns(6)
    cols[0].metric("Total return", f"{summary.get('total_return', 0):.2%}")
    cols[1].metric("Sharpe", f"{summary.get('sharpe', 0):.2f}")
    cols[2].metric("Max drawdown", f"{summary.get('max_drawdown', 0):.2%}")
    cols[3].metric("Deflated Sharpe P", f"{summary.get('deflated_sharpe_prob', float('nan')):.3f}")
    cols[4].metric("PBO", f"{overfit.get('pbo', float('nan')):.3f}")
    cols[5].metric("Avg turnover", f"{summary.get('average_turnover', 0):.2%}")

    tab_bt, tab_ic, tab_risk, tab_paper = st.tabs(
        ["Backtest", "Signal Quality", "Risk & Regimes", "Paper Trading"]
    )

    with tab_bt:
        curve_path = run_dir / "equity_curve.csv"
        if curve_path.exists():
            curve = pd.read_csv(curve_path, parse_dates=["date"])
            st.line_chart(curve.set_index("date")[["equity"]])
            st.area_chart(curve.set_index("date")[["gross_exposure"]])
        metrics = _csv(run_dir, "model_metrics.csv")
        if metrics is not None:
            st.subheader("Model metrics per walk-forward window")
            st.dataframe(metrics, use_container_width=True)
        fills = _csv(run_dir, "fills.csv")
        if fills is not None:
            st.subheader("Recent next-open fills")
            st.caption(
                "Decision and fill dates are explicit; liquidity inputs are lagged before the open."
            )
            st.dataframe(fills.tail(50), use_container_width=True)
        attribution = _csv(run_dir, "pnl_attribution.csv")
        if attribution is not None:
            by_symbol = (
                attribution.groupby("symbol")[["market_pnl", "trading_cost", "net_pnl"]]
                .sum()
                .sort_values("net_pnl")
            )
            st.subheader("P&L attribution by symbol")
            st.bar_chart(by_symbol["net_pnl"])

    with tab_ic:
        ic = _csv(run_dir, "ic_summary.csv")
        if ic is not None:
            st.subheader("Information coefficient by model (Newey-West t-stats)")
            st.dataframe(ic, use_container_width=True)
        decay = _csv(run_dir, "ic_decay.csv")
        if decay is not None:
            st.subheader("IC decay for the selected model")
            st.bar_chart(decay.set_index("horizon")["mean_rank_ic"])
        quantiles = _csv(run_dir, "quantile_returns.csv")
        if quantiles is not None:
            st.subheader("Forward return by prediction quantile")
            st.bar_chart(quantiles.set_index("quantile")["mean_return"])

    with tab_risk:
        regimes = _csv(run_dir, "regime_performance.csv")
        if regimes is not None:
            st.subheader("Regime-conditional performance")
            st.dataframe(regimes, use_container_width=True)
        stress = _csv(run_dir, "stress_tests.csv")
        if stress is not None:
            st.subheader("Beta-aware stress scenarios")
            st.dataframe(stress, use_container_width=True)
        capacity = _csv(run_dir, "capacity_curve.csv")
        if capacity is not None:
            st.subheader("Capacity sensitivity (not an AUM forecast)")
            st.line_chart(capacity.set_index("scenario_aum")[["fill_ratio"]])
            st.dataframe(capacity, use_container_width=True)
        weights = _csv(run_dir, "executed_weights.csv")
        if weights is not None:
            st.subheader("Latest holdings")
            weights["date"] = pd.to_datetime(weights["date"])
            latest = weights[weights["date"] == weights["date"].max()]
            st.dataframe(
                latest[latest["weight"] != 0].sort_values("weight"),
                use_container_width=True,
            )

    with tab_paper:
        st.warning("SIMULATED PAPER TRADING ONLY — no real orders are ever placed.")
        orders = _csv(run_dir, "paper_orders.csv")
        if orders is not None:
            st.subheader("Simulated causal next-open fills")
            st.dataframe(orders.tail(50), use_container_width=True)
        else:
            st.info("Run `make paper` to generate simulated orders.")


if __name__ == "__main__":
    main()
