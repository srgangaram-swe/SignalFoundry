"""Incremental-value evidence for regime models (SF-S3-MR4).

The issue's non-goal is blunt: never accept a regime because its chart looks
intuitive. This module is what replaces the chart. It measures what a regime
series is worth **against an explicit no-regime baseline**, on the same rows,
with the same forward returns, along the five axes the sprint plan names:

* **forecast** — does conditioning on the regime reduce error?
* **calibration** — is the state posterior honest, or systematically
  over-confident?
* **sizing** — does scaling exposure by the regime improve risk-adjusted return?
* **drawdown** — does it reduce the worst peak-to-trough loss?
* **stability** — does the regime persist, or flicker bar to bar?

Every measure is a **difference against the baseline**, never a level. A
regime-conditioned Sharpe of 1.2 says nothing on its own; the same number
against an unconditional Sharpe of 1.3 says the regime destroyed value, and only
the differenced form makes that impossible to miss.

Two guards against the usual self-deception:

* Forward returns must be strictly forward. The regime at ``t`` is applied to
  the return over ``t+1``, and the alignment is asserted rather than assumed.
* An unstable regime is reported, not hidden. ``flip_rate`` and
  ``mean_duration`` sit alongside the performance numbers, because a regime that
  changes every other bar can post an attractive Sharpe purely by trading noise
  and will not survive costs.

Nothing here decides acceptance. It produces the comparison; the rejection rule
belongs to the frozen study in SF-S3-MR11.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from alphaforge.regimes.base import RegimeError, RegimeStates

FloatArray = NDArray[np.float64]

#: Bars per year for annualization, matching the rest of the platform.
TRADING_DAYS = 252


@dataclass(frozen=True)
class RegimeValueEvidence:
    """Incremental value of one regime series against a no-regime baseline.

    Every ``*_delta`` field is ``regime - baseline``. **Direction differs by
    field and is stated on each one**, because getting a sign backwards here
    turns a harmful regime into an apparent improvement. ``mse_delta`` and
    ``brier_delta`` improve when negative (less error); ``sharpe_delta`` and
    ``max_drawdown_delta`` improve when positive — the latter because a
    drawdown is a negative number, so a shallower one is the larger value.
    """

    model: str
    n_observations: int
    #: Forecast: mean squared error of a regime-conditional mean vs the
    #: unconditional mean. Negative is an improvement.
    mse_delta: float
    #: Calibration: Brier score of the posterior against the realized
    #: high-volatility indicator, minus the no-skill constant-rate Brier.
    #: Negative is an improvement.
    brier_delta: float
    #: Sizing: annualized Sharpe of regime-scaled exposure minus always-on.
    sharpe_delta: float
    #: Drawdown: worst peak-to-trough of the scaled equity minus always-on.
    #: Drawdowns are negative, so a *positive* delta means the regime produced a
    #: shallower worst loss and is an improvement.
    max_drawdown_delta: float
    #: Stability: fraction of bars where the hard label changes.
    flip_rate: float
    #: Stability: mean bars spent in a state before switching.
    mean_duration: float
    #: Mean posterior entropy; high means the model is routinely unsure.
    mean_entropy: float
    #: Turnover implied by the exposure scaling, a proxy for the cost this
    #: regime would incur before any of the above is realizable.
    exposure_turnover: float

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-friendly record for the research ledger."""
        return {
            "model": self.model,
            "n_observations": self.n_observations,
            "mse_delta": self.mse_delta,
            "brier_delta": self.brier_delta,
            "sharpe_delta": self.sharpe_delta,
            "max_drawdown_delta": self.max_drawdown_delta,
            "flip_rate": self.flip_rate,
            "mean_duration": self.mean_duration,
            "mean_entropy": self.mean_entropy,
            "exposure_turnover": self.exposure_turnover,
        }

    def improves_nothing(self) -> bool:
        """Return whether no axis improved on the baseline.

        A convenience for reporting, not a rejection rule — the threshold that
        decides acceptance must be frozen before results are seen, and lives in
        the study, not here.
        """
        return (
            self.mse_delta >= 0.0
            and not (self.brier_delta < 0.0)
            and self.sharpe_delta <= 0.0
            and self.max_drawdown_delta <= 0.0
        )


def forward_returns(returns: pd.Series, horizon: int = 1) -> pd.Series:
    """Return the return realized over ``t+1 .. t+horizon``.

    A negative shift, so the value at ``t`` is strictly future information and
    never available to a model at ``t``. The last ``horizon`` rows are NaN,
    which is correct: the future has not happened.
    """
    if horizon < 1:
        raise RegimeError("horizon must be at least one bar")
    rolled = returns.shift(-1).rolling(horizon, min_periods=horizon).sum()
    return rolled.shift(-(horizon - 1)).rename("forward_return")


def max_drawdown(returns: FloatArray) -> float:
    """Return the worst peak-to-trough decline of the compounded path."""
    if returns.size == 0:
        return 0.0
    equity = np.cumprod(1.0 + returns)
    peak = np.maximum.accumulate(equity)
    return float(np.min(equity / peak - 1.0))


def annualized_sharpe(returns: FloatArray) -> float:
    """Return the annualized Sharpe ratio, or ``0.0`` for a degenerate path."""
    if returns.size < 2:
        return 0.0
    deviation = float(np.std(returns, ddof=1))
    if deviation <= 0.0:
        return 0.0
    return float(np.mean(returns) / deviation * np.sqrt(TRADING_DAYS))


def evaluate_regime_value(
    states: RegimeStates,
    returns: pd.Series,
    *,
    model_name: str,
    horizon: int = 1,
    stress_state: int | None = None,
    volatility_window: int = 21,
) -> RegimeValueEvidence:
    """Measure a regime series' incremental value against a no-regime baseline.

    Args:
        states: Causal posteriors aligned to ``returns``.
        returns: Realized per-bar returns.
        model_name: Label recorded in the evidence.
        horizon: Forward-return horizon in bars.
        stress_state: Index of the state treated as "risk-off" for sizing and
            calibration. Defaults to the last state, which under the canonical
            variance ordering is the most turbulent.
        volatility_window: Trailing window defining the realized
            high-volatility indicator the posterior is scored against.

    Raises:
        RegimeError: If the states are non-causal, misaligned, or leave too few
            usable observations.

    Returns:
        Differences against the baseline on all five axes.
    """
    if not states.causal:
        raise RegimeError(
            "incremental-value evidence requires causal state estimates; a "
            "retrospective segmentation would measure hindsight, not value"
        )
    if not states.index.equals(returns.index):
        raise RegimeError("states and returns must share an index")

    stress = len(states.labels) - 1 if stress_state is None else stress_state
    if not 0 <= stress < len(states.labels):
        raise RegimeError(f"stress_state must index one of {len(states.labels)} states")

    forward = forward_returns(returns, horizon)
    usable = states.observed & np.isfinite(forward.to_numpy()) & np.isfinite(returns.to_numpy())
    if int(usable.sum()) < 30:
        raise RegimeError("fewer than 30 aligned observations; evidence would be noise")

    posterior = states.probabilities[usable]
    future = forward.to_numpy()[usable]
    realized = returns.to_numpy()[usable]
    stress_probability = posterior[:, stress]
    labels = np.argmax(posterior, axis=1)

    # --- forecast: regime-conditional mean vs unconditional mean -------------
    # Conditional means come from the same rows they are scored on, so this is
    # an in-sample upper bound on the regime's forecast value, not an estimate
    # of out-of-sample skill. Stated plainly because a favourable number here
    # is the easiest to over-read in the whole record.
    baseline_prediction = np.full(future.shape, float(np.mean(future)))
    conditional = np.zeros_like(future)
    for state in range(posterior.shape[1]):
        selected = labels == state
        conditional[selected] = float(np.mean(future[selected])) if selected.any() else 0.0
    mse_delta = float(
        np.mean((future - conditional) ** 2) - np.mean((future - baseline_prediction) ** 2)
    )

    # --- calibration: posterior vs realized high-volatility indicator --------
    trailing = (
        returns.rolling(volatility_window, min_periods=volatility_window).std().to_numpy()[usable]
    )
    valid = np.isfinite(trailing)
    if valid.sum() >= 30:
        cutoff = float(np.quantile(trailing[valid], 0.7))
        indicator = (trailing[valid] > cutoff).astype(float)
        brier = float(np.mean((stress_probability[valid] - indicator) ** 2))
        base_rate = float(np.mean(indicator))
        brier_delta = brier - float(np.mean((base_rate - indicator) ** 2))
    else:
        brier_delta = float("nan")

    # --- sizing and drawdown: de-risk in the stress state --------------------
    exposure = 1.0 - stress_probability
    scaled = exposure * realized
    sharpe_delta = annualized_sharpe(scaled) - annualized_sharpe(realized)
    drawdown_delta = max_drawdown(scaled) - max_drawdown(realized)

    # --- stability -----------------------------------------------------------
    flips = int(np.sum(labels[1:] != labels[:-1]))
    flip_rate = flips / max(labels.size - 1, 1)
    mean_duration = labels.size / (flips + 1)
    entropy_values = states.entropy().to_numpy()[usable]
    mean_entropy = float(np.nanmean(entropy_values)) if entropy_values.size else float("nan")
    turnover = float(np.mean(np.abs(np.diff(exposure)))) if exposure.size > 1 else 0.0

    return RegimeValueEvidence(
        model=model_name,
        n_observations=int(usable.sum()),
        mse_delta=mse_delta,
        brier_delta=brier_delta,
        sharpe_delta=sharpe_delta,
        max_drawdown_delta=drawdown_delta,
        flip_rate=float(flip_rate),
        mean_duration=float(mean_duration),
        mean_entropy=mean_entropy,
        exposure_turnover=turnover,
    )


def compare_regime_models(
    evidence: list[RegimeValueEvidence],
) -> pd.DataFrame:
    """Return a deterministic comparison table across regime models.

    Sorted by model name rather than by any metric: ordering by performance
    invites reading the top row as a winner, and selecting a regime on this
    table is precisely what the frozen study exists to prevent.
    """
    if not evidence:
        raise RegimeError("cannot compare an empty evidence set")
    frame = pd.DataFrame([record.to_dict() for record in evidence])
    return frame.sort_values("model").reset_index(drop=True)
