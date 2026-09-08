"""Per-symbol technical features.

Every function takes a single symbol's bars sorted by date and returns
columns indexed like the input. All computations are causal: value at row t
depends only on rows <= t. This invariant is enforced by
tests/test_leakage.py, which mutates future bars and asserts features at t
are unchanged.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from alphaforge.utils import ANNUALIZATION_DAYS


def compute_symbol_features(g: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """Compute all single-name features for one symbol's sorted bars."""
    out = pd.DataFrame(index=g.index)
    close, volume = g["close"], g["volume"]
    ret = close.pct_change()
    logret = np.log(close).diff()

    # --- lagged returns (past k-day total return, known at t) ---
    for k in cfg.get("return_lags", [1, 5, 20]):
        out[f"ret_{k}"] = close.pct_change(k)

    out["logret_1"] = logret

    # --- rolling volatility (annualized) ---
    for w in cfg.get("vol_windows", [5, 20, 60]):
        out[f"vol_{w}"] = ret.rolling(w).std() * np.sqrt(ANNUALIZATION_DAYS)

    # --- moving-average ratios ---
    for w in cfg.get("ma_windows", [10, 20, 50, 200]):
        out[f"ma_ratio_{w}"] = close / close.rolling(w).mean() - 1.0

    # --- momentum ---
    for w in cfg.get("momentum_windows", [20, 60, 120]):
        out[f"momentum_{w}"] = close.pct_change(w)

    # --- RSI (Wilder smoothing) ---
    n = cfg.get("rsi_window", 14)
    delta = close.diff()
    gain = delta.clip(lower=0).ewm(alpha=1 / n, min_periods=n).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1 / n, min_periods=n).mean()
    rs = gain / loss.replace(0, np.nan)
    out["rsi"] = (100 - 100 / (1 + rs)) / 100.0  # scaled to [0, 1]

    # --- MACD (normalized by price to be scale-free) ---
    macd_cfg = cfg.get("macd", {"fast": 12, "slow": 26, "signal": 9})
    ema_fast = close.ewm(span=macd_cfg["fast"], min_periods=macd_cfg["fast"]).mean()
    ema_slow = close.ewm(span=macd_cfg["slow"], min_periods=macd_cfg["slow"]).mean()
    macd = (ema_fast - ema_slow) / close
    macd_signal = macd.ewm(span=macd_cfg["signal"], min_periods=macd_cfg["signal"]).mean()
    out["macd"] = macd
    out["macd_signal"] = macd_signal
    out["macd_hist"] = macd - macd_signal

    # --- Bollinger z-score ---
    w = cfg.get("bollinger_window", 20)
    ma, sd = close.rolling(w).mean(), close.rolling(w).std()
    out["bollinger_z"] = (close - ma) / (2 * sd)

    # --- short-horizon mean reversion: z-score of recent return ---
    w = cfg.get("mean_reversion_window", 5)
    r_w = close.pct_change(w)
    out["meanrev_z"] = (r_w - r_w.rolling(60).mean()) / r_w.rolling(60).std()

    # --- volume features ---
    w = cfg.get("volume_window", 20)
    vol_ma = volume.rolling(w).mean()
    out["volume_z"] = (volume - vol_ma) / volume.rolling(w).std()
    out["volume_ratio"] = volume / vol_ma - 1.0
    out["log_dollar_volume"] = np.log1p(close * volume)

    # --- drawdown from rolling high ---
    w = cfg.get("drawdown_window", 252)
    out["drawdown"] = close / close.rolling(w, min_periods=20).max() - 1.0

    # --- rolling Sharpe-style ratio ---
    w = cfg.get("rolling_sharpe_window", 60)
    out["rolling_sharpe"] = (
        ret.rolling(w).mean() / ret.rolling(w).std() * np.sqrt(ANNUALIZATION_DAYS)
    )

    return out


def compute_benchmark_relative(g: pd.DataFrame, bench: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """Rolling beta/correlation to benchmark for one symbol.

    ``bench`` must contain columns date, bench_ret.
    """
    merged = g[["date", "close"]].merge(bench, on="date", how="left")
    ret = merged["close"].pct_change()
    b = merged["bench_ret"]
    w = cfg.get("beta_window", 60)
    cov = ret.rolling(w).cov(b)
    var = b.rolling(w).var()
    out = pd.DataFrame(index=g.index)
    out["beta"] = (cov / var).to_numpy()
    out["bench_corr"] = ret.rolling(w).corr(b).to_numpy()
    return out


def compute_market_regime(bench_bars: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """Market-level regime features derived from the benchmark series.

    Returns a frame keyed by date, later joined onto every symbol:
    - bench_vol: trailing annualized benchmark volatility
    - bench_vol_pctile: expanding percentile of that vol (regime intensity)
    - bench_trend: fast MA vs slow MA of the benchmark (risk-on/off)
    - high_vol_regime: indicator that vol is in its top quartile so far
    """
    b = bench_bars.sort_values("date").reset_index(drop=True)
    ret = b["close"].pct_change()
    w = cfg.get("regime_vol_window", 20)
    vol = ret.rolling(w).std() * np.sqrt(ANNUALIZATION_DAYS)
    # expanding percentile: causal by construction
    vol_pctile = vol.expanding(min_periods=w * 3).rank(pct=True)
    fast = b["close"].rolling(cfg.get("regime_trend_fast", 50)).mean()
    slow = b["close"].rolling(cfg.get("regime_trend_slow", 200)).mean()
    return pd.DataFrame(
        {
            "date": b["date"],
            "bench_ret": ret,
            "bench_vol": vol,
            "bench_vol_pctile": vol_pctile,
            "bench_trend": fast / slow - 1.0,
            "high_vol_regime": (vol_pctile > 0.75).astype(float),
        }
    )
