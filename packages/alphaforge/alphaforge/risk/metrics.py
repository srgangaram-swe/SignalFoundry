"""Risk and performance analytics."""

from __future__ import annotations

import numpy as np
import pandas as pd

from alphaforge.utils import ANNUALIZATION_DAYS


def drawdown_series(equity: pd.Series) -> pd.Series:
    wealth = equity.astype(float)
    peak = wealth.cummax()
    return wealth / peak - 1.0


def _annual_return(returns: pd.Series) -> float:
    returns = returns.dropna()
    if returns.empty:
        return np.nan
    total = float((1.0 + returns).prod() - 1.0)
    years = len(returns) / ANNUALIZATION_DAYS
    return (1.0 + total) ** (1.0 / years) - 1.0 if years > 0 else np.nan


def _trim_to_active(equity_curve: pd.DataFrame) -> pd.DataFrame:
    """Drop leading rows before the strategy holds its first position.

    Walk-forward runs produce weights only from the first test window onward;
    including the dead ramp-up period dilutes every statistic (hit rate,
    Sharpe, exposure). Metrics should describe the period the strategy
    actually traded.
    """
    if "gross_exposure" not in equity_curve.columns:
        return equity_curve
    active = equity_curve["gross_exposure"].astype(float) > 0
    if not active.any():
        return equity_curve
    return equity_curve.loc[active.idxmax() :]


def performance_summary(equity_curve: pd.DataFrame, trim_inactive: bool = True) -> dict[str, float]:
    """Summarize backtest performance and risk.

    ``trim_inactive=True`` (default) starts the clock at the first day with
    non-zero exposure. Sharpe/Sortino use arithmetic daily means (industry
    convention); annual_return remains geometric.
    """
    frame = _trim_to_active(equity_curve) if trim_inactive else equity_curve
    r = frame["return"].astype(float).fillna(0.0)
    equity = frame["equity"].astype(float)
    dd = drawdown_series(equity)
    daily_vol = float(r.std())
    ann_vol = daily_vol * np.sqrt(ANNUALIZATION_DAYS)
    downside_daily = r[r < 0].std()
    ann_ret = _annual_return(r)
    bench = frame.get("benchmark_return", pd.Series(0.0, index=frame.index)).astype(float)
    beta = np.nan
    alpha = np.nan
    if bench.std() > 0 and len(bench) == len(r):
        beta = float(np.cov(r, bench)[0, 1] / np.var(bench))
        alpha = float((r.mean() - beta * bench.mean()) * ANNUALIZATION_DAYS)

    var_95 = float(r.quantile(0.05))
    tail = r[r <= var_95]
    turnover = frame.get("turnover", pd.Series(0.0)).astype(float)
    avg_turnover = float(turnover.mean())
    return {
        "n_days": int(len(r)),
        # Compounding daily returns retains the first active day's P&L. Using
        # final_equity / first_active_equity silently drops that day because
        # the first marked equity is already post-return.
        "total_return": float((1.0 + r).prod() - 1.0) if len(r) else 0.0,
        "annual_return": float(ann_ret),
        "annual_volatility": ann_vol,
        "sharpe": (
            np.nan if daily_vol == 0 else float(r.mean() / daily_vol * np.sqrt(ANNUALIZATION_DAYS))
        ),
        "sortino": (
            np.nan
            if downside_daily == 0 or np.isnan(downside_daily)
            else float(r.mean() / downside_daily * np.sqrt(ANNUALIZATION_DAYS))
        ),
        "max_drawdown": float(dd.min()),
        "calmar": np.nan if dd.min() == 0 else float(ann_ret / abs(dd.min())),
        "hit_rate": float((r > 0).mean()),
        "average_turnover": avg_turnover,
        "avg_holding_period_days": np.nan if avg_turnover == 0 else float(1.0 / avg_turnover),
        "transaction_cost_impact": float(frame.get("transaction_cost", pd.Series(0.0)).sum()),
        "average_gross_exposure": float(frame.get("gross_exposure", pd.Series(0.0)).mean()),
        "average_net_exposure": float(frame.get("net_exposure", pd.Series(0.0)).mean()),
        "beta_to_benchmark": beta,
        "alpha_estimate": alpha,
        "var_95_daily": var_95,
        "expected_shortfall_95_daily": float(tail.mean()) if not tail.empty else np.nan,
    }


def monthly_returns(equity_curve: pd.DataFrame) -> pd.DataFrame:
    frame = equity_curve.copy()
    frame["date"] = pd.to_datetime(frame["date"])
    monthly = (1.0 + frame.set_index("date")["return"]).resample("ME").prod() - 1.0
    return monthly.rename("return").reset_index()


def regime_performance(
    equity_curve: pd.DataFrame,
    regime: pd.Series,
    regime_name: str = "regime",
) -> pd.DataFrame:
    """Performance split by a date-indexed regime label (e.g. HMM stress state).

    ``regime`` is indexed by date; values are labels (0/1, 'calm'/'stress', ...).
    A strategy that only works in one regime is a different (and riskier)
    proposition than one that works in both — this table makes that visible.
    """
    frame = _trim_to_active(equity_curve).copy()
    frame["date"] = pd.to_datetime(frame["date"])
    regime = regime.copy()
    regime.index = pd.to_datetime(regime.index)
    frame[regime_name] = frame["date"].map(regime)
    rows = []
    for label, g in frame.dropna(subset=[regime_name]).groupby(regime_name):
        r = g["return"].astype(float)
        daily_vol = float(r.std())
        eq = (1.0 + r).cumprod()
        rows.append(
            {
                regime_name: label,
                "n_days": int(len(r)),
                "annual_return": _annual_return(r),
                "annual_volatility": daily_vol * np.sqrt(ANNUALIZATION_DAYS),
                "sharpe": (
                    np.nan
                    if daily_vol == 0
                    else float(r.mean() / daily_vol * np.sqrt(ANNUALIZATION_DAYS))
                ),
                "max_drawdown": float(drawdown_series(eq).min()),
                "hit_rate": float((r > 0).mean()),
            }
        )
    return pd.DataFrame(rows)


def exposure_summary(weights: pd.DataFrame) -> dict[str, float]:
    """Concentration diagnostics on the latest portfolio snapshot."""
    if weights.empty or "weight" not in weights.columns:
        return {
            "gross_exposure": 0.0,
            "net_exposure": 0.0,
            "n_positions": 0,
            "max_abs_weight": 0.0,
            "hhi_concentration": np.nan,
        }
    latest_date = weights["date"].max()
    latest = weights.loc[weights["date"] == latest_date]
    w = latest.set_index("symbol")["weight"].astype(float)
    w = w[w != 0]
    gross = float(w.abs().sum())
    return {
        "gross_exposure": gross,
        "net_exposure": float(w.sum()),
        "n_positions": int(len(w)),
        "max_abs_weight": float(w.abs().max()) if len(w) else 0.0,
        # HHI of gross-normalized weights: 1/n_eff, higher = more concentrated
        "hhi_concentration": float(((w.abs() / gross) ** 2).sum()) if gross > 0 else np.nan,
    }


def stress_test_summary(
    weights: pd.DataFrame,
    scenarios: list[dict] | None = None,
    betas: pd.Series | None = None,
) -> pd.DataFrame:
    """First-order scenario PnL and volatility multiplier diagnostics.

    ``betas``: per-symbol beta to the benchmark (e.g. the latest rolling-beta
    feature). Without betas, every name is assumed beta 1 and the estimate
    reduces to net_exposure * shock.
    """
    scenarios = scenarios or [{"name": "market -10%", "market_shock": -0.10}]
    latest = weights.sort_values("date").groupby("symbol").tail(1)
    w = (
        latest.set_index("symbol")["weight"].astype(float)
        if "weight" in latest
        else pd.Series(dtype=float)
    )
    gross = float(w.abs().sum())
    net = float(w.sum())
    if betas is not None:
        b = betas.reindex(w.index).fillna(1.0).astype(float)
    else:
        b = pd.Series(1.0, index=w.index)
    portfolio_beta = float((w * b).sum())
    rows = []
    for scenario in scenarios:
        shock = float(scenario.get("market_shock", 0.0))
        volatility_multiplier = float(scenario.get("vol_multiplier", 1.0))
        rows.append(
            {
                "scenario": scenario.get("name", "scenario"),
                "market_shock": shock,
                "portfolio_beta": portfolio_beta,
                "estimated_portfolio_return": portfolio_beta * shock,
                "volatility_multiplier": volatility_multiplier,
                "gross_exposure": gross,
                "net_exposure": net,
            }
        )
    return pd.DataFrame(rows)
