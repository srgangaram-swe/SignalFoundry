"""Label generation: strictly forward-looking targets.

Convention: a label at row (t, symbol) describes what happens *after* the
close of bar t. ``fwd_ret_h`` at t is the total return from close(t) to
close(t+h). Rows too close to the end of a symbol's history get NaN and are
dropped at training time.

Labels intentionally look into the future — that is their job. What must
never happen is a *feature* looking forward, or a label horizon overlapping
a training window without an embargo. Both are covered by tests.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from alphaforge.utils import ANNUALIZATION_DAYS


def build_labels(
    panel: pd.DataFrame,
    benchmark_symbol: str,
    horizons: list[int] | None = None,
    vol_window: int = 20,
) -> pd.DataFrame:
    """Build the multi-horizon label frame keyed by (date, symbol)."""
    horizons = horizons or [1, 5, 20]
    panel = panel.sort_values(["symbol", "date"]).reset_index(drop=True)

    bench = panel[panel["symbol"] == benchmark_symbol].sort_values("date")
    bench_fwd = {h: (bench["close"].shift(-h) / bench["close"] - 1.0) for h in horizons}
    bench_frame = pd.DataFrame({"date": bench["date"].values})
    for h in horizons:
        bench_frame[f"bench_fwd_{h}"] = bench_fwd[h].values

    frames = []
    for symbol, g in panel.groupby("symbol"):
        if symbol == benchmark_symbol:
            continue
        g = g.sort_values("date").reset_index(drop=True)
        close = g["close"]
        ret = close.pct_change()
        # trailing vol for volatility adjustment — uses PAST data only, so
        # the vol-adjusted label leaks nothing beyond the raw forward return
        trailing_vol = ret.rolling(vol_window).std() * np.sqrt(ANNUALIZATION_DAYS)

        out = g[["date", "symbol"]].copy()
        for h in horizons:
            fwd = close.shift(-h) / close - 1.0
            out[f"fwd_ret_{h}"] = fwd
            out[f"fwd_dir_{h}"] = np.where(fwd.isna(), np.nan, (fwd > 0).astype(float))
            out[f"fwd_ret_{h}_voladj"] = fwd / trailing_vol.replace(0, np.nan)
            # realized vol over the NEXT h days (a risk label, not a feature)
            out[f"fwd_vol_{h}"] = ret.shift(-h).rolling(h).std().to_numpy() * np.sqrt(
                ANNUALIZATION_DAYS
            )
        frames.append(out)

    labels = pd.concat(frames, ignore_index=True)
    labels = labels.merge(bench_frame, on="date", how="left")
    for h in horizons:
        labels[f"fwd_excess_{h}"] = labels[f"fwd_ret_{h}"] - labels[f"bench_fwd_{h}"]
        # cross-sectional quantile rank of future return (a relative label)
        labels[f"fwd_rank_{h}"] = labels.groupby("date")[f"fwd_ret_{h}"].rank(pct=True)
    labels = labels.drop(columns=[f"bench_fwd_{h}" for h in horizons])
    return labels.sort_values(["date", "symbol"]).reset_index(drop=True)


def label_columns(labels: pd.DataFrame) -> list[str]:
    return [c for c in labels.columns if c not in ("date", "symbol")]
