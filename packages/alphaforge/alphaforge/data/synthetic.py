"""Synthetic market generator: regime-switching factor model.

Used for tests, CI, and the no-network demo pipeline. The generator is
deliberately *not* pure noise — it embeds weak, realistic structure so that
the research stack has something honest to find:

- a two-state (calm/stress) Markov market regime driving drift and vol,
- per-asset market betas (cross-sectional correlation structure),
- a weak short-horizon mean-reversion effect in idiosyncratic returns,
- a weak medium-horizon momentum effect,
- volume that co-moves with absolute returns.

Signal strength is calibrated to be faint (IC on the order of a few percent),
which is representative of real equity alpha, so pipeline results on
synthetic data look like plausible research output rather than a rigged demo.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from alphaforge.data.schemas import validate_panel

TRADING_DAYS = 252


@dataclass
class SyntheticMarketConfig:
    n_symbols: int = 20
    n_days: int = 2500
    seed: int = 42
    start: str = "2015-01-02"
    benchmark_symbol: str = "BENCH"
    # regime parameters (annualized drift, daily vol)
    calm_mu: float = 0.08 / TRADING_DAYS
    calm_vol: float = 0.11 / np.sqrt(TRADING_DAYS)
    stress_mu: float = -0.20 / TRADING_DAYS
    stress_vol: float = 0.32 / np.sqrt(TRADING_DAYS)
    p_calm_to_stress: float = 0.01
    p_stress_to_calm: float = 0.05
    # embedded (weak) predictable structure — calibrated so cross-sectional
    # rank ICs land in the mid-single-digits, comparable to real equity alpha
    mean_reversion: float = -0.012  # loading of next-day idio return on past 5d idio return
    momentum: float = 0.004  # loading of next-day return on past 60d return
    idio_vol_range: tuple[float, float] = (0.010, 0.028)
    beta_range: tuple[float, float] = (0.5, 1.5)
    symbols: list[str] = field(default_factory=list)

    def symbol_names(self) -> list[str]:
        if self.symbols:
            return list(self.symbols)
        return [f"SYN{i:03d}" for i in range(self.n_symbols)]


def generate_synthetic_market(config: SyntheticMarketConfig | None = None) -> pd.DataFrame:
    """Generate a canonical long-format OHLCV panel including the benchmark.

    Deterministic given ``config.seed``.
    """
    cfg = config or SyntheticMarketConfig()
    rng = np.random.default_rng(cfg.seed)
    n, t = cfg.n_symbols, cfg.n_days
    dates = pd.bdate_range(cfg.start, periods=t)

    # --- market factor with 2-state Markov regime ---
    regime = np.zeros(t, dtype=int)  # 0 = calm, 1 = stress
    u = rng.random(t)
    for i in range(1, t):
        if regime[i - 1] == 0:
            regime[i] = 1 if u[i] < cfg.p_calm_to_stress else 0
        else:
            regime[i] = 0 if u[i] < cfg.p_stress_to_calm else 1
    mu = np.where(regime == 0, cfg.calm_mu, cfg.stress_mu)
    vol = np.where(regime == 0, cfg.calm_vol, cfg.stress_vol)
    mkt_ret = mu + vol * rng.standard_normal(t)

    # --- per-asset parameters ---
    betas = rng.uniform(*cfg.beta_range, size=n)
    idio_vols = rng.uniform(*cfg.idio_vol_range, size=n)

    # --- asset returns with weak mean reversion + momentum ---
    idio = np.zeros((t, n))
    rets = np.zeros((t, n))
    eps = rng.standard_normal((t, n)) * idio_vols
    for i in range(t):
        past5 = idio[max(0, i - 5) : i].sum(axis=0) if i > 0 else np.zeros(n)
        past60 = rets[max(0, i - 60) : i].sum(axis=0) if i > 0 else np.zeros(n)
        idio[i] = eps[i] + cfg.mean_reversion * past5
        rets[i] = betas * mkt_ret[i] + idio[i] + cfg.momentum * past60
    rets = np.clip(rets, -0.4, 0.4)

    frames = []
    all_rets = np.column_stack([mkt_ret, rets])
    all_syms = [cfg.benchmark_symbol] + cfg.symbol_names()
    base_prices = np.concatenate([[100.0], rng.uniform(20, 400, size=n)])
    base_volumes = np.concatenate([[5e8], rng.uniform(1e6, 5e7, size=n)])

    for j, sym in enumerate(all_syms):
        r = all_rets[:, j]
        close = base_prices[j] * np.exp(np.cumsum(np.log1p(r)))
        open_ = np.empty_like(close)
        open_[0] = base_prices[j]
        # next open gaps a fraction of daily vol away from prior close
        gap = rng.standard_normal(t - 1) * 0.25 * np.std(r)
        open_[1:] = close[:-1] * (1 + gap)
        intrabar = np.abs(rng.standard_normal(t)) * 0.5 * np.std(r) * close
        high = np.maximum(open_, close) + intrabar
        low = np.minimum(open_, close) - intrabar
        low = np.maximum(low, 0.01)
        volume = base_volumes[j] * np.exp(
            2.0 * np.abs(r) / (np.std(r) + 1e-12) * 0.3 + rng.standard_normal(t) * 0.3
        )
        frames.append(
            pd.DataFrame(
                {
                    "date": dates,
                    "symbol": sym,
                    "open": open_,
                    "high": high,
                    "low": low,
                    "close": close,
                    "volume": volume.round(0),
                }
            )
        )

    panel = pd.concat(frames, ignore_index=True)
    return validate_panel(panel)
