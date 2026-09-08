"""Two-state Gaussian HMM regime model (calm / stress), no external deps.

Markets exhibit volatility clustering: tranquil trends punctuated by
high-volatility drawdown episodes. A 2-state Gaussian HMM on daily benchmark
returns captures this with five interpretable parameters per state, fit by
Baum-Welch EM with scaled forward-backward recursions.

Leakage discipline — the part most implementations get wrong:

- *Parameters* are refit on an expanding window every ``refit_every`` days,
  using only data before the refit date.
- *State inference* uses **filtered** probabilities P(state_t | x_{1..t}),
  never smoothed ones: smoothing conditions on the full sample and leaks the
  future. ``causal_stress_probability`` combines both rules, so its output at
  date t is a deterministic function of returns up to t. Verified by
  tests/test_leakage.py.

State 1 is always the higher-variance ("stress") state.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

_LOG_SQRT_2PI = 0.5 * np.log(2.0 * np.pi)


def _emission_probs(x: np.ndarray, means: np.ndarray, variances: np.ndarray) -> np.ndarray:
    """Gaussian emission densities, T x 2, floored for numerical safety."""
    var = np.maximum(variances, 1e-12)
    log_b = (
        -_LOG_SQRT_2PI
        - 0.5 * np.log(var)[None, :]
        - 0.5 * (x[:, None] - means[None, :]) ** 2 / var[None, :]
    )
    return np.exp(np.clip(log_b, -700, 50))


class GaussianHMM2:
    """2-state Gaussian hidden Markov model fit with EM."""

    def __init__(self, n_iter: int = 100, tol: float = 1e-6, seed: int = 42):
        self.n_iter = n_iter
        self.tol = tol
        self.seed = seed
        self.pi_: np.ndarray | None = None
        self.transition_: np.ndarray | None = None
        self.means_: np.ndarray | None = None
        self.variances_: np.ndarray | None = None
        self.loglik_: float = -np.inf

    def fit(self, returns: pd.Series | np.ndarray) -> GaussianHMM2:
        x = pd.Series(returns).dropna().to_numpy(dtype=float)
        if len(x) < 30:
            raise ValueError("need at least 30 observations to fit the HMM")

        # deterministic, informative init: split by absolute move size
        med = np.median(np.abs(x))
        calm, wild = x[np.abs(x) <= med], x[np.abs(x) > med]
        means = np.array([calm.mean(), wild.mean()])
        variances = np.array([max(calm.var(), 1e-10), max(wild.var(), 1e-10) * 2.0])
        pi = np.array([0.8, 0.2])
        A = np.array([[0.97, 0.03], [0.06, 0.94]])

        prev_ll = -np.inf
        for _ in range(self.n_iter):
            B = _emission_probs(x, means, variances)
            # scaled forward
            alpha = np.empty((len(x), 2))
            scale = np.empty(len(x))
            alpha[0] = pi * B[0]
            scale[0] = alpha[0].sum() + 1e-300
            alpha[0] /= scale[0]
            for t in range(1, len(x)):
                alpha[t] = (alpha[t - 1] @ A) * B[t]
                scale[t] = alpha[t].sum() + 1e-300
                alpha[t] /= scale[t]
            ll = float(np.log(scale).sum())
            # scaled backward
            beta = np.empty((len(x), 2))
            beta[-1] = 1.0
            for t in range(len(x) - 2, -1, -1):
                beta[t] = (A @ (B[t + 1] * beta[t + 1])) / scale[t + 1]
            gamma = alpha * beta
            gamma /= gamma.sum(axis=1, keepdims=True) + 1e-300
            # transition expectations, vectorized over t:
            # xi[t] = A ∘ (alpha[t] ⊗ (B[t+1] * beta[t+1] / scale[t+1]))
            w = (B[1:] * beta[1:]) / scale[1:, None]
            xi_num = (alpha[:-1].T @ w) * A
            # M-step
            pi = gamma[0] / gamma[0].sum()
            A = xi_num / (gamma[:-1].sum(axis=0)[:, None] + 1e-300)
            A /= A.sum(axis=1, keepdims=True)
            weights = gamma.sum(axis=0)
            means = (gamma * x[:, None]).sum(axis=0) / (weights + 1e-300)
            variances = (gamma * (x[:, None] - means[None, :]) ** 2).sum(axis=0) / (
                weights + 1e-300
            )
            variances = np.maximum(variances, 1e-12)
            if abs(ll - prev_ll) < self.tol * max(1.0, abs(prev_ll)):
                prev_ll = ll
                break
            prev_ll = ll

        # canonical ordering: state 1 = stress = higher variance
        if variances[0] > variances[1]:
            order = [1, 0]
            pi, means, variances = pi[order], means[order], variances[order]
            A = A[np.ix_(order, order)]

        self.pi_, self.transition_ = pi, A
        self.means_, self.variances_ = means, variances
        self.loglik_ = prev_ll
        return self

    def filtered_probabilities(self, returns: pd.Series | np.ndarray) -> np.ndarray:
        """P(state_t | x_{1..t}) — causal given fixed parameters. T x 2."""
        if (
            self.pi_ is None
            or self.transition_ is None
            or self.means_ is None
            or self.variances_ is None
        ):
            raise RuntimeError("fit the model first")
        x = pd.Series(returns).to_numpy(dtype=float)
        B = _emission_probs(np.nan_to_num(x, nan=0.0), self.means_, self.variances_)
        probs = np.empty((len(x), 2))
        state = self.pi_.copy()
        for t in range(len(x)):
            state = (state @ self.transition_) if t > 0 else state
            state = state * B[t]
            total = state.sum()
            state = state / total if total > 0 else np.array([0.5, 0.5])
            probs[t] = state
        return probs


def causal_stress_probability(
    returns: pd.Series,
    refit_every: int = 63,
    min_train: int = 252,
    n_iter: int = 100,
) -> pd.Series:
    """Filtered stress-state probability with expanding parameter refits.

    For each block of ``refit_every`` days starting at k: fit parameters on
    returns[:k], filter forward through returns[:block_end], and keep only the
    block's outputs. Output at t therefore depends only on returns up to t.
    Dates before ``min_train`` are NaN.
    """
    r = returns.astype(float)
    values = r.to_numpy()
    out = np.full(len(r), np.nan)
    start = min_train
    while start < len(r):
        end = min(start + refit_every, len(r))
        train = values[:start]
        train = train[np.isfinite(train)]
        if len(train) >= 30:
            model = GaussianHMM2(n_iter=n_iter).fit(train)
            probs = model.filtered_probabilities(np.nan_to_num(values[:end], nan=0.0))
            out[start:end] = probs[start:end, 1]
        start = end
    return pd.Series(out, index=r.index, name="hmm_stress_prob")
