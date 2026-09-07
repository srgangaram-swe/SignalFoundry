"""Gaussian mixture and hidden Markov regime models (SF-S3-MR4).

Two probabilistic state models over a single observed series, both fit by EM
from scratch so their initialization, convergence, and labelling are auditable.

The pair is deliberately complementary, and comparing them is informative:

* **GMM** treats observations as independent draws from a mixture. It knows
  nothing about time, so its posterior can flip state on a single outlier.
* **HMM** adds a transition matrix, so persistence is a *fitted* parameter
  rather than an assumption. On volatility-clustered returns this is usually the
  difference between a usable state series and a flickering one.

If the HMM's estimated transition matrix comes out near-uniform, it has found no
persistence and its extra parameters bought nothing — which the transition
matrix in :meth:`fitted_parameters` makes checkable rather than assumed.

Both order states by **increasing variance**, so state 0 is always the calmest
and the top state is always the most turbulent. EM has no preferred labelling,
so without this two runs finding identical states could swap their indices and
silently mix definitions across refit boundaries.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray

from alphaforge.regimes.base import (
    MAX_EM_ITERATIONS,
    MAX_STATES,
    RegimeError,
    RegimeModel,
    RegimeReport,
    canonical_order,
)

FloatArray = NDArray[np.float64]

_LOG_SQRT_2PI = 0.5 * np.log(2.0 * np.pi)

#: Variance floor. A mixture component can otherwise collapse onto a single
#: point, driving its variance to zero and its likelihood to infinity — the
#: classic degenerate EM solution, which looks like spectacular convergence.
VARIANCE_FLOOR = 1e-12


def _log_gaussian(values: FloatArray, means: FloatArray, variances: FloatArray) -> FloatArray:
    """Return ``(n, k)`` Gaussian log densities."""
    variance = np.maximum(variances, VARIANCE_FLOOR)
    return (
        -_LOG_SQRT_2PI
        - 0.5 * np.log(variance)[None, :]
        - 0.5 * (values[:, None] - means[None, :]) ** 2 / variance[None, :]
    )


def _log_sum_exp(values: FloatArray, axis: int) -> FloatArray:
    """Numerically stable log-sum-exp.

    Working in log space throughout is what keeps a long series from
    underflowing to zero likelihood; the shift by the row maximum is what keeps
    the exponentials themselves in range.
    """
    peak = np.max(values, axis=axis, keepdims=True)
    peak = np.where(np.isfinite(peak), peak, 0.0)
    total = np.log(np.sum(np.exp(values - peak), axis=axis, keepdims=True)) + peak
    return np.asarray(np.squeeze(total, axis=axis), dtype=np.float64)


def _quantile_initialization(values: FloatArray, n_states: int) -> tuple[FloatArray, FloatArray]:
    """Return deterministic initial means and variances.

    Splitting on quantiles of absolute deviation seeds the components across the
    dispersion range the data actually shows. It is deterministic — a random
    restart would make the fit irreproducible, and the identity hash meaningless.
    """
    magnitude = np.abs(values - np.median(values))
    edges = np.quantile(magnitude, np.linspace(0.0, 1.0, n_states + 1))
    means = np.empty(n_states, dtype=np.float64)
    variances = np.empty(n_states, dtype=np.float64)
    for state in range(n_states):
        low, high = edges[state], edges[state + 1]
        selected = values[(magnitude >= low) & (magnitude <= high)]
        if selected.size < 2:
            selected = values
        means[state] = float(np.mean(selected))
        variances[state] = max(float(np.var(selected)), VARIANCE_FLOOR)
    # Separate identical variances so components do not start superimposed and
    # collapse into one another.
    variances = variances * np.linspace(1.0, 1.0 + 0.5 * n_states, n_states)
    return means, variances


class GaussianMixtureRegime(RegimeModel):
    """Independent Gaussian mixture over one observed series.

    Args:
        n_states: Mixture components, at most :data:`MAX_STATES`.
        max_iterations: EM budget.
        tolerance: Relative log-likelihood change ending the fit.
    """

    name = "gmm_regime"
    causal = True
    ordering_rule = "states sorted by increasing fitted variance"

    def __init__(
        self, n_states: int = 2, *, max_iterations: int = 200, tolerance: float = 1e-6
    ) -> None:
        super().__init__()
        if not 1 <= n_states <= MAX_STATES:
            raise RegimeError(f"n_states must be in [1, {MAX_STATES}], got {n_states}")
        if not 1 <= max_iterations <= MAX_EM_ITERATIONS:
            raise RegimeError(f"max_iterations must be in [1, {MAX_EM_ITERATIONS}]")
        if tolerance <= 0.0 or not np.isfinite(tolerance):
            raise RegimeError("tolerance must be finite and positive")
        self.n_states = n_states
        self.max_iterations = max_iterations
        self.tolerance = tolerance
        self.weights_: FloatArray = np.empty(0)
        self.means_: FloatArray = np.empty(0)
        self.variances_: FloatArray = np.empty(0)

    def _fit(self, values: FloatArray) -> RegimeReport:
        means, variances = _quantile_initialization(values, self.n_states)
        weights = np.full(self.n_states, 1.0 / self.n_states)
        previous = -np.inf
        iterations = 0
        converged = False
        for iterations in range(1, self.max_iterations + 1):  # noqa: B007 - reported
            log_joint = np.log(np.maximum(weights, 1e-300))[None, :] + _log_gaussian(
                values, means, variances
            )
            log_evidence = _log_sum_exp(log_joint, axis=1)
            responsibility = np.exp(log_joint - log_evidence[:, None])
            total = float(np.sum(log_evidence))

            mass = responsibility.sum(axis=0)
            weights = mass / mass.sum()
            means = (responsibility * values[:, None]).sum(axis=0) / np.maximum(mass, 1e-300)
            variances = np.maximum(
                (responsibility * (values[:, None] - means[None, :]) ** 2).sum(axis=0)
                / np.maximum(mass, 1e-300),
                VARIANCE_FLOOR,
            )
            if abs(total - previous) < self.tolerance * max(1.0, abs(previous)):
                previous = total
                converged = True
                break
            previous = total

        order = canonical_order(variances)
        self.weights_, self.means_, self.variances_ = weights[order], means[order], variances[order]
        return RegimeReport(
            model=self.name,
            n_states=self.n_states,
            iterations=iterations,
            max_iterations=self.max_iterations,
            converged=converged,
            stopping_reason="log_likelihood_tolerance" if converged else "max_iterations",
            n_train_observations=int(values.size),
            log_likelihood=float(previous) if np.isfinite(previous) else None,
        )

    def _filter(self, values: FloatArray) -> FloatArray:
        observed = np.isfinite(values)
        probabilities = np.full((values.size, self.n_states), np.nan)
        if observed.any():
            log_joint = np.log(np.maximum(self.weights_, 1e-300))[None, :] + _log_gaussian(
                values[observed], self.means_, self.variances_
            )
            probabilities[observed] = np.exp(log_joint - _log_sum_exp(log_joint, axis=1)[:, None])
        return probabilities

    @property
    def state_labels(self) -> tuple[str, ...]:
        return _variance_ordered_labels(self.n_states)

    def configuration(self) -> dict[str, Any]:
        return {
            "n_states": self.n_states,
            "max_iterations": self.max_iterations,
            "tolerance": self.tolerance,
        }

    def fitted_parameters(self) -> dict[str, Any]:
        return {
            "weights": [float(value) for value in self.weights_],
            "means": [float(value) for value in self.means_],
            "variances": [float(value) for value in self.variances_],
        }


class GaussianHMMRegime(RegimeModel):
    """K-state Gaussian hidden Markov model fit by Baum-Welch.

    Inference uses **filtered** posteriors ``P(state_t | x_1..t)``, never
    smoothed ones. Smoothing conditions on the entire sample and is the single
    most common way a regime feature acquires lookahead: the smoothed series
    looks cleaner precisely because it has seen the future.

    Args:
        n_states: Hidden states, at most :data:`MAX_STATES`.
        max_iterations: Baum-Welch budget.
        tolerance: Relative log-likelihood change ending the fit.
        self_transition: Initial diagonal mass. Seeding persistence high matches
            the volatility-clustering prior; the value is still re-estimated.
    """

    name = "hmm_regime"
    causal = True
    ordering_rule = "states sorted by increasing emission variance"

    def __init__(
        self,
        n_states: int = 2,
        *,
        max_iterations: int = 200,
        tolerance: float = 1e-6,
        self_transition: float = 0.95,
    ) -> None:
        super().__init__()
        if not 1 <= n_states <= MAX_STATES:
            raise RegimeError(f"n_states must be in [1, {MAX_STATES}], got {n_states}")
        if not 1 <= max_iterations <= MAX_EM_ITERATIONS:
            raise RegimeError(f"max_iterations must be in [1, {MAX_EM_ITERATIONS}]")
        if tolerance <= 0.0 or not np.isfinite(tolerance):
            raise RegimeError("tolerance must be finite and positive")
        if not 0.0 < self_transition < 1.0:
            raise RegimeError("self_transition must lie strictly in (0, 1)")
        self.n_states = n_states
        self.max_iterations = max_iterations
        self.tolerance = tolerance
        self.self_transition = self_transition
        self.initial_: FloatArray = np.empty(0)
        self.transition_: FloatArray = np.empty((0, 0))
        self.means_: FloatArray = np.empty(0)
        self.variances_: FloatArray = np.empty(0)

    def _initial_transition(self) -> FloatArray:
        if self.n_states == 1:
            return np.ones((1, 1))
        off = (1.0 - self.self_transition) / (self.n_states - 1)
        matrix = np.full((self.n_states, self.n_states), off)
        np.fill_diagonal(matrix, self.self_transition)
        return matrix

    def _fit(self, values: FloatArray) -> RegimeReport:
        means, variances = _quantile_initialization(values, self.n_states)
        transition = self._initial_transition()
        initial = np.full(self.n_states, 1.0 / self.n_states)
        previous = -np.inf
        iterations = 0
        converged = False
        length = values.size

        for iterations in range(1, self.max_iterations + 1):  # noqa: B007 - reported
            emission = np.exp(np.clip(_log_gaussian(values, means, variances), -700.0, 50.0))
            alpha = np.empty((length, self.n_states))
            scale = np.empty(length)
            alpha[0] = initial * emission[0]
            scale[0] = alpha[0].sum() + 1e-300
            alpha[0] /= scale[0]
            for step in range(1, length):
                alpha[step] = (alpha[step - 1] @ transition) * emission[step]
                scale[step] = alpha[step].sum() + 1e-300
                alpha[step] /= scale[step]
            total = float(np.log(scale).sum())

            beta = np.empty((length, self.n_states))
            beta[-1] = 1.0
            for step in range(length - 2, -1, -1):
                beta[step] = (transition @ (emission[step + 1] * beta[step + 1])) / scale[step + 1]

            gamma = alpha * beta
            gamma /= gamma.sum(axis=1, keepdims=True) + 1e-300
            forward = (emission[1:] * beta[1:]) / scale[1:, None]
            transition_counts = (alpha[:-1].T @ forward) * transition

            initial = gamma[0] / gamma[0].sum()
            transition = transition_counts / (gamma[:-1].sum(axis=0)[:, None] + 1e-300)
            transition /= transition.sum(axis=1, keepdims=True)
            mass = gamma.sum(axis=0)
            means = (gamma * values[:, None]).sum(axis=0) / (mass + 1e-300)
            variances = np.maximum(
                (gamma * (values[:, None] - means[None, :]) ** 2).sum(axis=0) / (mass + 1e-300),
                VARIANCE_FLOOR,
            )
            if abs(total - previous) < self.tolerance * max(1.0, abs(previous)):
                previous = total
                converged = True
                break
            previous = total

        order = canonical_order(variances)
        self.initial_ = initial[order]
        self.transition_ = transition[np.ix_(order, order)]
        self.means_, self.variances_ = means[order], variances[order]
        return RegimeReport(
            model=self.name,
            n_states=self.n_states,
            iterations=iterations,
            max_iterations=self.max_iterations,
            converged=converged,
            stopping_reason="log_likelihood_tolerance" if converged else "max_iterations",
            n_train_observations=int(length),
            log_likelihood=float(previous) if np.isfinite(previous) else None,
        )

    def _filter(self, values: FloatArray) -> FloatArray:
        # A missing observation carries no likelihood, so the filter propagates
        # the prior through the transition matrix without updating on evidence —
        # the honest treatment. Imputing a value would fabricate evidence.
        emission = np.exp(
            np.clip(
                _log_gaussian(np.nan_to_num(values, nan=0.0), self.means_, self.variances_),
                -700.0,
                50.0,
            )
        )
        observed = np.isfinite(values)
        probabilities = np.full((values.size, self.n_states), np.nan)
        state = self.initial_.copy()
        for step in range(values.size):
            if step > 0:
                state = state @ self.transition_
            if observed[step]:
                state = state * emission[step]
                total = state.sum()
                state = (
                    state / total if total > 0.0 else np.full(self.n_states, 1.0 / self.n_states)
                )
                probabilities[step] = state
            else:
                total = state.sum()
                state = (
                    state / total if total > 0.0 else np.full(self.n_states, 1.0 / self.n_states)
                )
        return probabilities

    @property
    def state_labels(self) -> tuple[str, ...]:
        return _variance_ordered_labels(self.n_states)

    def configuration(self) -> dict[str, Any]:
        return {
            "n_states": self.n_states,
            "max_iterations": self.max_iterations,
            "tolerance": self.tolerance,
            "self_transition": self.self_transition,
        }

    def fitted_parameters(self) -> dict[str, Any]:
        return {
            "initial": [float(value) for value in self.initial_],
            "transition": [[float(value) for value in row] for row in self.transition_],
            "means": [float(value) for value in self.means_],
            "variances": [float(value) for value in self.variances_],
        }

    def expected_durations(self) -> list[float]:
        """Return each state's expected dwell time in bars, ``1/(1 - a_ii)``.

        The most interpretable summary of a fitted HMM: a state with an expected
        duration of 1.2 bars is not a regime, it is a relabelled outlier
        detector, and this number says so directly.
        """
        self._ensure_fitted()
        diagonal = np.clip(np.diag(self.transition_), 0.0, 1.0 - 1e-12)
        return [float(1.0 / (1.0 - value)) for value in diagonal]


def _variance_ordered_labels(n_states: int) -> tuple[str, ...]:
    """Return canonical labels for variance-ordered states."""
    if n_states == 1:
        return ("single",)
    if n_states == 2:
        return ("calm", "stress")
    if n_states == 3:
        return ("calm", "normal", "stress")
    return tuple(f"state_{index}" for index in range(n_states))
