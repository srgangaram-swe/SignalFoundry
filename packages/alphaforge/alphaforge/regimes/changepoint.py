"""Change-point detection: CUSUM, BOCPD, and retrospective segmenters (SF-S3-MR4).

Change-point methods split cleanly into two kinds, and conflating them is the
leakage bug this module is arranged to prevent.

**Online (causal).** :class:`CusumRegime` and :class:`BayesianOnlineChangePoint`
decide at each bar using only bars up to that point. They may be used as
features.

**Offline (retrospective).** :func:`segment_bayesian` and
:func:`segment_kernel` place change points by optimizing over the *entire*
series. They are strictly better at locating a break — because they can see
what came after it — and are therefore useless as features and dangerous if
mistaken for one. They return :class:`Segmentation`, not a
:class:`~alphaforge.regimes.base.RegimeModel`, so they cannot be handed to
``expanding_state_probabilities`` at all; the type system enforces the
separation rather than a docstring warning.

Retrospective segmentation is still worth having: it is the reference against
which an online detector's lag and false-alarm rate are measured. The point is
that the comparison is the deliverable, not the segmentation itself.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray

from alphaforge.regimes.base import RegimeError, RegimeModel, RegimeReport

FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]

#: Ceilings. Retrospective segmentation is quadratic in series length, so an
#: unbounded request is a configuration mistake rather than a long wait.
MAX_SEGMENT_SERIES = 5_000
MAX_CHANGE_POINTS = 50


class CusumRegime(RegimeModel):
    """Two-sided CUSUM change detector as a two-state regime model.

    CUSUM accumulates standardized deviations from the in-control mean and
    flags a change when the running sum crosses ``threshold``. It is the
    classical sequential test: minimal state, no distributional fitting beyond a
    mean and scale, and a detection delay that is provably near-optimal for a
    step change of known size.

    ``drift`` is the slack that makes it usable. Without it the statistic
    accumulates noise indefinitely and eventually crosses any threshold; ``drift``
    is the per-observation allowance below which a deviation is treated as noise,
    conventionally set near half the change magnitude worth detecting.

    The posterior is a bounded, monotone map of the exceedance rather than a
    calibrated probability — CUSUM is a test statistic, not a generative model.
    :meth:`fitted_parameters` records that so the number is never mistaken for
    one. State ``0`` is in-control, state ``1`` is changed.

    Args:
        threshold: Decision threshold in standardized units.
        drift: Per-observation slack in standardized units.
    """

    name = "cusum_regime"
    causal = True
    ordering_rule = "state 0 in-control, state 1 changed (fixed by construction)"

    def __init__(self, *, threshold: float = 5.0, drift: float = 0.5) -> None:
        super().__init__()
        if threshold <= 0.0 or not np.isfinite(threshold):
            raise RegimeError("threshold must be finite and positive")
        if drift < 0.0 or not np.isfinite(drift):
            raise RegimeError("drift must be finite and non-negative")
        self.threshold = threshold
        self.drift = drift
        self.center_: float = 0.0
        self.scale_: float = 1.0
        self.n_states = 2

    def _fit(self, values: FloatArray) -> RegimeReport:
        # Median and MAD rather than mean and standard deviation: the in-control
        # baseline must not be dragged by the very excursions CUSUM exists to
        # detect. MAD is scaled to be consistent for the Gaussian standard
        # deviation so `threshold` keeps its conventional interpretation.
        center = float(np.median(values))
        deviation = float(np.median(np.abs(values - center))) * 1.4826
        self.center_ = center
        self.scale_ = deviation if deviation > 0.0 else float(np.std(values)) or 1.0
        return RegimeReport(
            model=self.name,
            n_states=2,
            iterations=1,
            max_iterations=1,
            converged=True,
            stopping_reason="closed_form",
            n_train_observations=int(values.size),
        )

    def _filter(self, values: FloatArray) -> FloatArray:
        standardized = (values - self.center_) / self.scale_
        probabilities = np.full((values.size, 2), np.nan)
        high = 0.0
        low = 0.0
        for step, value in enumerate(standardized):
            if not np.isfinite(value):
                # No evidence: hold the statistic rather than resetting it, and
                # emit nothing for this bar.
                continue
            high = max(0.0, high + value - self.drift)
            low = max(0.0, low - value - self.drift)
            exceedance = max(high, low) / self.threshold
            changed = float(np.clip(exceedance, 0.0, 1.0))
            probabilities[step] = (1.0 - changed, changed)
            if exceedance >= 1.0:
                # Alarm: reset so the detector can find the next change instead
                # of latching permanently after the first one.
                high = 0.0
                low = 0.0
        return probabilities

    @property
    def state_labels(self) -> tuple[str, ...]:
        return ("in_control", "changed")

    def configuration(self) -> dict[str, Any]:
        return {"threshold": self.threshold, "drift": self.drift}

    def fitted_parameters(self) -> dict[str, Any]:
        return {
            "center": self.center_,
            "scale": self.scale_,
            "posterior_is_calibrated": False,
            "posterior_definition": "clipped CUSUM exceedance over threshold",
        }


class BayesianOnlineChangePoint(RegimeModel):
    """Bayesian online change-point detection (Adams & MacKay, 2007).

    Maintains a posterior over the **run length** — how many bars since the last
    change — updated recursively at each observation. It is causal by
    construction rather than by discipline: the recursion has no access to
    future data, so there is no smoothed variant to accidentally use.

    The observation model is Gaussian with a Normal-Inverse-Gamma conjugate
    prior, so the predictive is Student-t and parameters are updated in closed
    form. Student-t matters here: its heavy tails stop a single large return
    from being read as a certain regime change, which a Gaussian predictive
    would do routinely on financial data.

    The reported two-state posterior collapses the run-length distribution to
    ``P(run length < short_run)`` — the probability of being recently changed.
    The full run-length posterior is available from :meth:`run_length_posterior`.

    Args:
        hazard: Constant per-bar change probability; the prior expected regime
            length is ``1 / hazard``.
        short_run: Run length below which a bar counts as recently changed.
        max_run_length: Truncation of the run-length posterior, bounding both
            memory and time to ``O(max_run_length)`` per bar.
    """

    name = "bocpd_regime"
    causal = True
    ordering_rule = "state 0 established, state 1 recently changed (fixed by construction)"

    def __init__(
        self,
        *,
        hazard: float = 1.0 / 250.0,
        short_run: int = 21,
        max_run_length: int = 500,
        prior_strength: float = 1.0,
    ) -> None:
        super().__init__()
        if not 0.0 < hazard < 1.0:
            raise RegimeError("hazard must lie strictly in (0, 1)")
        if short_run < 1:
            raise RegimeError("short_run must be at least one bar")
        if not 2 <= max_run_length <= 10_000:
            raise RegimeError("max_run_length must be in [2, 10000]")
        if prior_strength <= 0.0 or not np.isfinite(prior_strength):
            raise RegimeError("prior_strength must be finite and positive")
        self.hazard = hazard
        self.short_run = short_run
        self.max_run_length = max_run_length
        self.prior_strength = prior_strength
        self.prior_mean_: float = 0.0
        self.prior_variance_: float = 1.0
        self.n_states = 2
        self._run_length_posterior: FloatArray | None = None

    def _fit(self, values: FloatArray) -> RegimeReport:
        self.prior_mean_ = float(np.median(values))
        spread = float(np.median(np.abs(values - self.prior_mean_))) * 1.4826
        self.prior_variance_ = float(spread**2) if spread > 0.0 else float(np.var(values)) or 1.0
        return RegimeReport(
            model=self.name,
            n_states=2,
            iterations=1,
            max_iterations=1,
            converged=True,
            stopping_reason="closed_form",
            n_train_observations=int(values.size),
        )

    def _filter(self, values: FloatArray) -> FloatArray:
        width = self.max_run_length
        # Normal-Inverse-Gamma sufficient statistics per run length.
        alpha = np.full(width, self.prior_strength)
        beta = np.full(width, self.prior_strength * self.prior_variance_)
        kappa = np.full(width, self.prior_strength)
        mu = np.full(width, self.prior_mean_)

        posterior = np.zeros(width)
        posterior[0] = 1.0
        probabilities = np.full((values.size, 2), np.nan)
        history = np.full((values.size, width), np.nan)

        for step, value in enumerate(values):
            if not np.isfinite(value):
                continue
            predictive = _student_t_pdf(value, mu, alpha, beta, kappa)
            growth = posterior * predictive * (1.0 - self.hazard)
            change = float(np.sum(posterior * predictive * self.hazard))

            updated = np.zeros(width)
            updated[0] = change
            updated[1:] = growth[:-1]
            # The truncated tail would otherwise leak probability mass out of
            # the distribution; folding it into the last bin conserves it.
            updated[-1] += growth[-1]
            total = updated.sum()
            posterior = updated / total if total > 0.0 else _uniform(width)

            new_kappa = np.concatenate([[self.prior_strength], kappa[:-1] + 1.0])
            new_mu = np.concatenate(
                [[self.prior_mean_], (kappa[:-1] * mu[:-1] + value) / (kappa[:-1] + 1.0)]
            )
            new_alpha = np.concatenate([[self.prior_strength], alpha[:-1] + 0.5])
            new_beta = np.concatenate(
                [
                    [self.prior_strength * self.prior_variance_],
                    beta[:-1] + kappa[:-1] * (value - mu[:-1]) ** 2 / (2.0 * (kappa[:-1] + 1.0)),
                ]
            )
            kappa, mu, alpha, beta = new_kappa, new_mu, new_alpha, new_beta

            history[step] = posterior
            recent = float(posterior[: self.short_run].sum())
            probabilities[step] = (1.0 - recent, recent)

        self._run_length_posterior = history
        return probabilities

    def run_length_posterior(self) -> FloatArray:
        """Return the ``(n, max_run_length)`` run-length posterior from the last filter."""
        if self._run_length_posterior is None:
            raise RegimeError("call filter before requesting the run-length posterior")
        return self._run_length_posterior

    @property
    def state_labels(self) -> tuple[str, ...]:
        return ("established", "recently_changed")

    def configuration(self) -> dict[str, Any]:
        return {
            "hazard": self.hazard,
            "short_run": self.short_run,
            "max_run_length": self.max_run_length,
            "prior_strength": self.prior_strength,
        }

    def fitted_parameters(self) -> dict[str, Any]:
        return {"prior_mean": self.prior_mean_, "prior_variance": self.prior_variance_}


def _uniform(width: int) -> FloatArray:
    return np.full(width, 1.0 / width)


def _student_t_pdf(
    value: float, mu: FloatArray, alpha: FloatArray, beta: FloatArray, kappa: FloatArray
) -> FloatArray:
    """Posterior predictive of a Normal-Inverse-Gamma model: a Student-t."""
    from scipy.special import gammaln

    degrees = 2.0 * alpha
    scale = beta * (kappa + 1.0) / (alpha * kappa)
    standardized = (value - mu) ** 2 / (degrees * scale)
    log_density = (
        gammaln((degrees + 1.0) / 2.0)
        - gammaln(degrees / 2.0)
        - 0.5 * np.log(np.pi * degrees * scale)
        - (degrees + 1.0) / 2.0 * np.log1p(standardized)
    )
    return np.asarray(np.exp(np.clip(log_density, -700.0, 50.0)), dtype=np.float64)


@dataclass(frozen=True)
class Segmentation:
    """Retrospective change points over a complete series.

    **Not causal.** Deliberately not a :class:`RegimeModel`, so it cannot be
    passed to the causal expanding-window driver. Use it to measure an online
    detector's lag, never as a feature.

    Attributes:
        change_points: Indices where a new segment begins, ascending, excluding
            zero.
        method: Segmenter that produced them.
        cost: Total penalized cost of the chosen segmentation.
        penalty: Per-change-point penalty applied.
    """

    change_points: IntArray
    method: str
    cost: float
    penalty: float
    causal: bool = False

    def __post_init__(self) -> None:
        if self.change_points.ndim != 1:
            raise RegimeError("change points must be one-dimensional")
        if self.change_points.size and (np.diff(self.change_points) <= 0).any():
            raise RegimeError("change points must be strictly ascending")
        if self.causal:
            raise RegimeError("retrospective segmentation is never causal")

    def segment_labels(self, length: int) -> IntArray:
        """Return a segment index per observation."""
        labels = np.zeros(length, dtype=np.int64)
        for position, point in enumerate(self.change_points, start=1):
            labels[int(point) :] = position
        return labels


def _validate_segment_input(
    values: FloatArray, max_change_points: int, penalty: float
) -> FloatArray:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1:
        raise RegimeError("segmentation expects a one-dimensional series")
    if not np.isfinite(array).all():
        raise RegimeError("segmentation requires a finite series; it fails closed on gaps")
    if array.size < 4:
        raise RegimeError("segmentation needs at least four observations")
    if array.size > MAX_SEGMENT_SERIES:
        raise RegimeError(
            f"series of {array.size} exceeds the {MAX_SEGMENT_SERIES} segmentation ceiling; "
            "retrospective segmentation is quadratic in length"
        )
    if not 1 <= max_change_points <= MAX_CHANGE_POINTS:
        raise RegimeError(f"max_change_points must be in [1, {MAX_CHANGE_POINTS}]")
    if penalty < 0.0 or not np.isfinite(penalty):
        raise RegimeError("penalty must be finite and non-negative")
    return array


def segment_bayesian(
    values: FloatArray, *, max_change_points: int = 5, penalty: float = 10.0
) -> Segmentation:
    """Retrospective segmentation by penalized Gaussian model evidence.

    Exact dynamic programming over the ``O(n^2)`` segment-cost matrix, so the
    result is the global optimum for the stated cost — not a greedy
    approximation whose output depends on scan order.

    Segment cost is the Gaussian negative log-likelihood at the segment's own
    maximum-likelihood mean and variance, so a split is only worth its penalty
    if it genuinely reduces within-segment dispersion. ``penalty`` is the price
    of each extra change point; without it the optimum is one segment per
    observation.

    Raises:
        RegimeError: On a non-finite, too-short, or oversized series.
    """
    array = _validate_segment_input(values, max_change_points, penalty)
    length = array.size
    cost = _gaussian_segment_costs(array)

    # best[k, t] = optimal cost of segmenting the first t observations with k cuts.
    best = np.full((max_change_points + 1, length + 1), np.inf)
    previous = np.zeros((max_change_points + 1, length + 1), dtype=np.int64)
    best[0, 1:] = cost[0, 1:]
    for cuts in range(1, max_change_points + 1):
        for end in range(2, length + 1):
            candidates = best[cuts - 1, 1:end] + cost[1:end, end] + penalty
            if candidates.size == 0:
                continue
            position = int(np.argmin(candidates))
            best[cuts, end] = candidates[position]
            previous[cuts, end] = position + 1

    total = best[:, length]
    chosen = int(np.argmin(total))
    points: list[int] = []
    end = length
    for cuts in range(chosen, 0, -1):
        start = int(previous[cuts, end])
        points.append(start)
        end = start
    return Segmentation(
        change_points=np.asarray(sorted(points), dtype=np.int64),
        method="bayesian_dp",
        cost=float(total[chosen]),
        penalty=penalty,
    )


def _gaussian_segment_costs(values: FloatArray) -> FloatArray:
    """Return ``cost[i, j]`` = Gaussian NLL of ``values[i:j]`` at its own MLE."""
    length = values.size
    prefix = np.concatenate([[0.0], np.cumsum(values)])
    prefix_square = np.concatenate([[0.0], np.cumsum(values**2)])
    cost = np.full((length + 1, length + 1), np.inf)
    for start in range(length):
        ends = np.arange(start + 1, length + 1)
        count = ends - start
        total = prefix[ends] - prefix[start]
        total_square = prefix_square[ends] - prefix_square[start]
        variance = np.maximum(total_square / count - (total / count) ** 2, 1e-12)
        cost[start, ends] = 0.5 * count * (np.log(2.0 * np.pi * variance) + 1.0)
    return cost


def segment_kernel(
    values: FloatArray,
    *,
    max_change_points: int = 5,
    penalty: float = 1.0,
    bandwidth: float | None = None,
) -> Segmentation:
    """Retrospective kernel segmentation (ruptures-style, RBF kernel).

    Where :func:`segment_bayesian` assumes Gaussian segments, this compares
    segments by their **distributions** in a reproducing-kernel Hilbert space,
    so it detects a change in shape — skew, tails, multimodality — that leaves
    mean and variance almost untouched. Financial regime shifts frequently look
    exactly like that.

    Segment cost is the within-segment kernel scatter
    ``sum_i k(x_i,x_i) - (1/n) sum_{i,j} k(x_i,x_j)``, minimized exactly by the
    same dynamic program.

    Args:
        bandwidth: RBF bandwidth. ``None`` uses the median pairwise distance —
            the standard heuristic, and deterministic, so the segmentation does
            not depend on an unstated scale choice.
    """
    array = _validate_segment_input(values, max_change_points, penalty)
    gram = _rbf_gram(array, bandwidth)
    length = array.size
    cost = _kernel_segment_costs(gram)

    best = np.full((max_change_points + 1, length + 1), np.inf)
    previous = np.zeros((max_change_points + 1, length + 1), dtype=np.int64)
    best[0, 1:] = cost[0, 1:]
    for cuts in range(1, max_change_points + 1):
        for end in range(2, length + 1):
            candidates = best[cuts - 1, 1:end] + cost[1:end, end] + penalty
            if candidates.size == 0:
                continue
            position = int(np.argmin(candidates))
            best[cuts, end] = candidates[position]
            previous[cuts, end] = position + 1

    total = best[:, length]
    chosen = int(np.argmin(total))
    points: list[int] = []
    end = length
    for cuts in range(chosen, 0, -1):
        start = int(previous[cuts, end])
        points.append(start)
        end = start
    return Segmentation(
        change_points=np.asarray(sorted(points), dtype=np.int64),
        method="kernel_rbf",
        cost=float(total[chosen]),
        penalty=penalty,
    )


def _rbf_gram(values: FloatArray, bandwidth: float | None) -> FloatArray:
    """Return the RBF Gram matrix with a median-heuristic bandwidth."""
    distance = np.abs(values[:, None] - values[None, :])
    if bandwidth is None:
        upper = distance[np.triu_indices_from(distance, k=1)]
        median = float(np.median(upper)) if upper.size else 1.0
        bandwidth = median if median > 0.0 else 1.0
    if bandwidth <= 0.0 or not np.isfinite(bandwidth):
        raise RegimeError("bandwidth must be finite and positive")
    return np.asarray(np.exp(-(distance**2) / (2.0 * bandwidth**2)), dtype=np.float64)


def _kernel_segment_costs(gram: FloatArray) -> FloatArray:
    """Return ``cost[i, j]`` = within-segment kernel scatter of ``values[i:j]``."""
    length = gram.shape[0]
    # Two-dimensional prefix sums make each segment's block sum O(1).
    block = np.zeros((length + 1, length + 1))
    block[1:, 1:] = np.cumsum(np.cumsum(gram, axis=0), axis=1)
    diagonal = np.concatenate([[0.0], np.cumsum(np.diag(gram))])
    cost = np.full((length + 1, length + 1), np.inf)
    for start in range(length):
        ends = np.arange(start + 1, length + 1)
        count = ends - start
        total = block[ends, ends] - block[start, ends] - block[ends, start] + block[start, start]
        cost[start, ends] = (diagonal[ends] - diagonal[start]) - total / count
    return cost
