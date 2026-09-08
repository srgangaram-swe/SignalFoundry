"""Backtest-overfitting statistics: PSR, DSR, PBO, Newey-West IC inference.

The most common failure in quant research is not a bad model — it is an
overstated backtest. This module quantifies that risk:

- **PSR** (Probabilistic Sharpe Ratio): probability the true Sharpe exceeds a
  benchmark, adjusting for sample length, skew, and fat tails
  (Bailey & Lopez de Prado, 2012).
- **DSR** (Deflated Sharpe Ratio): PSR against the Sharpe you'd expect from
  the *best of N* unskilled trials — the honest multiple-testing correction
  for "we tried N models and report the winner" (Bailey & LdP, 2014).
- **PBO** (Probability of Backtest Overfitting) via CSCV: how often the
  in-sample winner underperforms the median out-of-sample
  (Bailey, Borwein, LdP & Zhu, 2015).
- **Newey-West t-statistics** for daily IC series, robust to the serial
  correlation induced by overlapping multi-day labels.
"""

from __future__ import annotations

import itertools
from math import e as EULER_E

import numpy as np
import pandas as pd
from scipy.stats import kurtosis, norm, skew

EULER_MASCHERONI = 0.5772156649015329


# ---------------------------------------------------------------------------
# Sharpe inference
# ---------------------------------------------------------------------------


def probabilistic_sharpe_ratio(returns: pd.Series | np.ndarray, sr_benchmark: float = 0.0) -> float:
    """P(true SR > sr_benchmark), higher-moment adjusted. SRs are per-period."""
    r = pd.Series(returns).dropna().to_numpy(dtype=float)
    if len(r) < 3 or np.std(r, ddof=1) == 0:
        return np.nan
    sr = float(np.mean(r) / np.std(r, ddof=1))
    g3 = float(skew(r))
    g4 = float(kurtosis(r, fisher=False))  # raw kurtosis; 3 for a normal
    denom = np.sqrt(max(1e-12, 1.0 - g3 * sr + (g4 - 1.0) / 4.0 * sr**2))
    z = (sr - sr_benchmark) * np.sqrt(len(r) - 1) / denom
    return float(norm.cdf(z))


def expected_max_sharpe(n_trials: int, sr_variance: float) -> float:
    """E[max SR] across n unskilled trials with SR estimator variance sr_variance."""
    if n_trials <= 1 or sr_variance <= 0:
        return 0.0
    g = EULER_MASCHERONI
    return float(
        np.sqrt(sr_variance)
        * ((1 - g) * norm.ppf(1 - 1.0 / n_trials) + g * norm.ppf(1 - 1.0 / (n_trials * EULER_E)))
    )


def deflated_sharpe_ratio(
    returns: pd.Series | np.ndarray,
    n_trials: int,
    trial_sharpes: list[float] | None = None,
    periods_per_year: int = 252,
) -> dict[str, float]:
    """DSR: PSR of the reported strategy against the best-of-N null.

    ``n_trials`` must count every configuration evaluated before selecting
    this result (models x targets x strategy variants). Understating it is
    self-deception; it is the single most important input here.
    """
    r = pd.Series(returns).dropna().to_numpy(dtype=float)
    if len(r) < 3 or np.std(r, ddof=1) == 0:
        return {
            "sharpe_annualized": np.nan,
            "psr_vs_zero": np.nan,
            "expected_max_sharpe": np.nan,
            "deflated_sharpe_prob": np.nan,
            "n_trials": n_trials,
            "n_obs": len(r),
        }
    sr = float(np.mean(r) / np.std(r, ddof=1))
    if trial_sharpes is not None and len(trial_sharpes) > 1:
        sr_var = float(np.var(np.asarray(trial_sharpes, dtype=float), ddof=1))
    else:
        # variance of the SR estimator itself (Lo, 2002 with moment adjustment)
        g3 = float(skew(r))
        g4 = float(kurtosis(r, fisher=False))
        sr_var = (1.0 - g3 * sr + (g4 - 1.0) / 4.0 * sr**2) / max(1, len(r) - 1)
    sr0 = expected_max_sharpe(n_trials, sr_var)
    return {
        "sharpe_annualized": sr * np.sqrt(periods_per_year),
        "psr_vs_zero": probabilistic_sharpe_ratio(r, 0.0),
        "expected_max_sharpe": sr0,
        "deflated_sharpe_prob": probabilistic_sharpe_ratio(r, sr0),
        "n_trials": n_trials,
        "n_obs": len(r),
    }


# ---------------------------------------------------------------------------
# Probability of Backtest Overfitting (CSCV)
# ---------------------------------------------------------------------------


def probability_of_backtest_overfitting(
    performance: pd.DataFrame,
    n_blocks: int = 16,
    max_combinations: int = 3000,
    seed: int = 42,
) -> dict:
    """PBO via combinatorially symmetric cross-validation.

    ``performance``: T x N frame — rows are time-ordered observations (daily
    returns or daily ICs), columns are strategy/model variants. For every
    half-and-half split of the T rows into IS/OOS blocks, pick the IS winner
    and record the logit of its OOS relative rank. PBO is the fraction of
    splits where the IS winner lands in the bottom half OOS.
    """
    # drop degenerate strategies (all-NaN, e.g. a constant-prediction baseline)
    # before row filtering, or one dead column erases the whole sample
    M = performance.dropna(axis=1, how="all").dropna(axis=0, how="any")
    T, N = M.shape
    if N < 2 or n_blocks * 2 > T:
        return {"pbo": np.nan, "n_combinations": 0, "logits": np.array([]), "n_obs": T}

    values = M.to_numpy(dtype=float)
    blocks = np.array_split(np.arange(T), n_blocks)
    block_sum = np.vstack([values[b].sum(axis=0) for b in blocks])
    block_sumsq = np.vstack([(values[b] ** 2).sum(axis=0) for b in blocks])
    block_n = np.array([len(b) for b in blocks], dtype=float)

    combos = list(itertools.combinations(range(n_blocks), n_blocks // 2))
    if len(combos) > max_combinations:
        rng = np.random.default_rng(seed)
        combos = [combos[i] for i in rng.choice(len(combos), max_combinations, replace=False)]

    all_ids = frozenset(range(n_blocks))
    logits = np.empty(len(combos))
    for i, combo in enumerate(combos):
        is_ids = list(combo)
        oos_ids = list(all_ids - set(combo))
        logits[i] = _oos_rank_logit(block_sum, block_sumsq, block_n, is_ids, oos_ids, N)

    logits = logits[np.isfinite(logits)]
    return {
        "pbo": float(np.mean(logits <= 0)) if len(logits) else np.nan,
        "n_combinations": int(len(logits)),
        "logits": logits,
        "mean_logit": float(np.mean(logits)) if len(logits) else np.nan,
        "n_obs": T,
        "n_strategies": N,
    }


def _sharpe_from_blocks(block_sum, block_sumsq, block_n, ids) -> np.ndarray:
    n = block_n[ids].sum()
    mean = block_sum[ids].sum(axis=0) / n
    var = block_sumsq[ids].sum(axis=0) / n - mean**2
    sd = np.sqrt(np.maximum(var, 1e-18))
    return mean / sd


def _oos_rank_logit(block_sum, block_sumsq, block_n, is_ids, oos_ids, n_strategies) -> float:
    is_sharpe = _sharpe_from_blocks(block_sum, block_sumsq, block_n, is_ids)
    oos_sharpe = _sharpe_from_blocks(block_sum, block_sumsq, block_n, oos_ids)
    best = int(np.argmax(is_sharpe))
    # relative OOS rank of the IS winner, in (0, 1)
    omega = (np.sum(oos_sharpe <= oos_sharpe[best])) / (n_strategies + 1.0)
    return float(np.log(omega / (1.0 - omega)))


# ---------------------------------------------------------------------------
# IC inference
# ---------------------------------------------------------------------------


def newey_west_tstat(series: pd.Series | np.ndarray, lags: int | None = None) -> float:
    """t-stat of the series mean with HAC (Newey-West) standard errors."""
    x = pd.Series(series).dropna().to_numpy(dtype=float)
    T = len(x)
    if T < 3:
        return np.nan
    if lags is None:
        lags = int(np.floor(4 * (T / 100.0) ** (2.0 / 9.0)))
    lags = max(0, min(lags, T - 2))
    e = x - x.mean()
    s = float(np.mean(e * e))
    for lag in range(1, lags + 1):
        gamma = float(np.mean(e[lag:] * e[:-lag]))
        s += 2.0 * (1.0 - lag / (lags + 1.0)) * gamma
    if s <= 0:
        return np.nan
    return float(x.mean() / np.sqrt(s / T))


def ic_summary(ic_by_date: pd.DataFrame, col: str = "rank_ic") -> dict[str, float]:
    """Aggregate a per-date IC frame into mean IC, ICIR, and a NW t-stat."""
    ic = ic_by_date[col].dropna()
    if ic.empty:
        return {
            "mean_ic": np.nan,
            "ic_std": np.nan,
            "icir": np.nan,
            "t_stat_nw": np.nan,
            "pct_positive": np.nan,
            "n_dates": 0,
        }
    return {
        "mean_ic": float(ic.mean()),
        "ic_std": float(ic.std()),
        "icir": float(ic.mean() / ic.std()) if ic.std() > 0 else np.nan,
        "t_stat_nw": newey_west_tstat(ic),
        "pct_positive": float((ic > 0).mean()),
        "n_dates": int(len(ic)),
    }


def ic_decay(
    predictions: pd.DataFrame,
    labels: pd.DataFrame,
    horizons: list[int],
    prediction_col: str = "prediction",
) -> pd.DataFrame:
    """Mean per-date rank IC of one signal against each label horizon.

    A signal whose IC decays slowly can be traded at lower frequency for the
    same alpha — this curve directly informs the rebalance choice.
    """
    from alphaforge.evaluation.metrics import _rank_corr

    merged = predictions.merge(labels, on=["date", "symbol"], how="left")
    rows = []
    for h in horizons:
        col = f"fwd_ret_{h}"
        if col not in merged.columns:
            continue
        daily = []
        for _, g in merged.groupby("date"):
            valid = g[[prediction_col, col]].dropna()
            if len(valid) >= 3:
                daily.append(_rank_corr(valid[col].to_numpy(), valid[prediction_col].to_numpy()))
        daily = pd.Series(daily).dropna()
        rows.append(
            {
                "horizon": h,
                "mean_rank_ic": float(daily.mean()) if len(daily) else np.nan,
                "t_stat_nw": newey_west_tstat(daily),
                "n_dates": int(len(daily)),
            }
        )
    return pd.DataFrame(rows)
