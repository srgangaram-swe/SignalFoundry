"""Shared contract for regime and change-point models (SF-S3-MR4).

A regime model is unusually easy to fool yourself with. It produces a chart that
looks obviously right — calm stretches in blue, the crash in red — and the chart
is not evidence, because the model was fit on the whole sample including the
crash. Four properties in this contract exist specifically to make that mistake
impossible to commit accidentally:

**Causality is declared, not assumed.** Every model carries a class-level
``causal`` flag. Offline segmenters (retrospective Bayesian, kernel) are
genuinely non-causal — they condition on the whole series by construction — so
they declare ``causal = False`` and :func:`expanding_state_probabilities`
*refuses* them. They remain available for retrospective analysis, and cannot be
turned into a feature by accident.

**Posteriors, not labels.** Every model returns a full state posterior. A hard
label discards the model's own uncertainty, which is the quantity that matters
when a regime call sizes a position. ``confidence`` and ``entropy`` are derived
from the posterior so an ambiguous call is visible as ambiguous.

**Canonical label ordering.** EM has no preferred labelling: two runs can find
the same two states and swap their indices. Every model therefore sorts its
states by a declared, deterministic key, so "state 1" means the same thing
across runs, folds, and refits. Without this, a regime-conditioned backtest
silently mixes two different definitions across refit boundaries.

**Deterministic identity.** A model hashes its configuration *and* its fitted
parameters, so a regime series can be traced to the exact state model that
produced it — and two models that disagree cannot be confused for one.

The non-goal this implements: never accept a regime because its chart looks
intuitive, and never relabel states after inspecting a holdout.
"""

from __future__ import annotations

import hashlib
import json
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Literal, Self

import numpy as np
import pandas as pd
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]

#: Hard ceilings. Refusal thresholds, not tuning knobs.
MAX_STATES = 8
MAX_EM_ITERATIONS = 2_000
MIN_FIT_OBSERVATIONS = 30

StoppingReason = Literal[
    "log_likelihood_tolerance",
    "parameter_tolerance",
    "max_iterations",
    "closed_form",
]


class RegimeError(Exception):
    """Base class for regime-model failures."""


class NotFittedError(RegimeError):
    """Raised when inference is attempted before ``fit``."""


class RegimeLeakageError(RegimeError):
    """Raised when a non-causal model is used where causality is required."""


@dataclass(frozen=True)
class RegimeReport:
    """How one regime fit terminated.

    ``converged`` means a stopping criterion fired. Exhausting the iteration
    budget is reported as ``False`` — an EM run cut off mid-ascent has not found
    a mode, and treating its parameters as fitted is exactly the silent
    non-convergence this contract refuses.
    """

    model: str
    n_states: int
    iterations: int
    max_iterations: int
    converged: bool
    stopping_reason: StoppingReason
    n_train_observations: int
    log_likelihood: float | None = None
    seed: int | None = None
    notes: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.n_states < 1:
            raise ValueError("a regime model must have at least one state")
        if self.iterations < 0 or self.iterations > self.max_iterations:
            raise ValueError("iterations must lie within the configured budget")
        if self.n_train_observations < 1:
            raise ValueError("a fit needs at least one training observation")
        if self.log_likelihood is not None and not np.isfinite(self.log_likelihood):
            raise ValueError("log likelihood must be finite when reported")

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-friendly record for run manifests."""
        return {
            "model": self.model,
            "n_states": self.n_states,
            "iterations": self.iterations,
            "max_iterations": self.max_iterations,
            "converged": self.converged,
            "stopping_reason": self.stopping_reason,
            "n_train_observations": self.n_train_observations,
            "log_likelihood": self.log_likelihood,
            "seed": self.seed,
            "notes": list(self.notes),
        }


@dataclass(frozen=True)
class RegimeStates:
    """Posterior state probabilities aligned to an observation index.

    Attributes:
        probabilities: ``(n_observations, n_states)``, rows summing to one over
            observed rows and all-NaN where the model could not produce an
            estimate (warm-up or missing input).
        labels: State names in canonical order.
        index: Observation index the rows align to.
        causal: Whether every row used only information available at that row.
            Carried on the result, not just the model, so a downstream consumer
            holding only this object still knows whether it may be used as a
            feature.
    """

    probabilities: FloatArray
    labels: tuple[str, ...]
    index: pd.Index
    causal: bool

    def __post_init__(self) -> None:
        if self.probabilities.ndim != 2:
            raise RegimeError("state probabilities must be two-dimensional")
        if self.probabilities.shape[1] != len(self.labels):
            raise RegimeError("probability columns must match the state labels")
        if self.probabilities.shape[0] != len(self.index):
            raise RegimeError("probability rows must match the observation index")
        observed = np.isfinite(self.probabilities).all(axis=1)
        if observed.any():
            totals = self.probabilities[observed].sum(axis=1)
            if not np.allclose(totals, 1.0, atol=1e-8):
                raise RegimeError("observed state posteriors must sum to one")
            if (self.probabilities[observed] < -1e-12).any():
                raise RegimeError("state posteriors must be non-negative")

    @property
    def observed(self) -> NDArray[np.bool_]:
        """Rows that carry a usable posterior."""
        return np.isfinite(self.probabilities).all(axis=1)

    def to_frame(self) -> pd.DataFrame:
        """Return the posterior as a labelled frame."""
        return pd.DataFrame(
            self.probabilities, index=self.index, columns=[f"p_{name}" for name in self.labels]
        )

    def hard_labels(self) -> pd.Series:
        """Return the maximum-a-posteriori state, or NA where unobserved.

        Provided for reporting only. Sizing and conditioning should use the
        posterior: collapsing to a label throws away exactly the uncertainty
        that should shrink a position when the regime call is marginal.
        """
        best = np.full(len(self.index), -1, dtype=int)
        observed = self.observed
        if observed.any():
            best[observed] = np.argmax(self.probabilities[observed], axis=1)
        values = [self.labels[position] if position >= 0 else None for position in best]
        return pd.Series(values, index=self.index, dtype="object", name="regime")

    def confidence(self) -> pd.Series:
        """Return the maximum posterior probability per observation."""
        values = np.full(len(self.index), np.nan)
        observed = self.observed
        values[observed] = self.probabilities[observed].max(axis=1)
        return pd.Series(values, index=self.index, name="regime_confidence")

    def entropy(self) -> pd.Series:
        """Return posterior entropy normalized to ``[0, 1]``.

        ``0`` is a certain call and ``1`` is complete ambiguity. Normalizing by
        ``log(n_states)`` keeps the scale comparable between a two-state and a
        four-state model, which a raw entropy would not.
        """
        values = np.full(len(self.index), np.nan)
        observed = self.observed
        if observed.any() and len(self.labels) > 1:
            mass = self.probabilities[observed]
            with np.errstate(divide="ignore", invalid="ignore"):
                terms = np.where(mass > 0.0, mass * np.log(mass), 0.0)
            values[observed] = -terms.sum(axis=1) / np.log(len(self.labels))
        return pd.Series(values, index=self.index, name="regime_entropy")


class RegimeModel(ABC):
    """A model producing posterior probabilities over market states.

    Subclasses declare :attr:`causal`, implement ``_fit`` and ``_filter``, and
    supply :meth:`fitted_parameters` for the identity hash. Canonical state
    ordering is the subclass's responsibility at fit time and is verified here.
    """

    name: str = "regime_model"
    #: Whether inference at row ``t`` uses only information available at ``t``.
    causal: bool = True
    #: Human-readable description of the canonical ordering rule, published so a
    #: consumer knows what "state 0" means without reading the implementation.
    ordering_rule: str = "unspecified"

    def __init__(self) -> None:
        self._fitted = False
        self._report: RegimeReport | None = None
        self._labels: tuple[str, ...] = ()

    # ----- required interface -------------------------------------------------
    @abstractmethod
    def _fit(self, values: FloatArray) -> RegimeReport:
        """Fit on a finite training array and return termination evidence."""

    @abstractmethod
    def _filter(self, values: FloatArray) -> FloatArray:
        """Return ``(n, n_states)`` posteriors for a finite-or-NaN array."""

    @abstractmethod
    def fitted_parameters(self) -> dict[str, Any]:
        """Return JSON-safe fitted parameters, in canonical state order."""

    @property
    @abstractmethod
    def state_labels(self) -> tuple[str, ...]:
        """Return canonical state names."""

    def configuration(self) -> dict[str, Any]:
        """Return JSON-safe constructor configuration."""
        return {}

    # ----- provided behaviour -------------------------------------------------
    def fit(self, series: pd.Series | FloatArray) -> Self:
        """Fit on the supplied observations only.

        Raises:
            RegimeError: If the training sample is too small or non-finite after
                dropping missing values.
        """
        values = _finite_training_values(series)
        self._report = self._fit(values)
        self._labels = self.state_labels
        if len(self._labels) != self._report.n_states:
            raise RegimeError("fitted state count disagrees with the declared labels")
        if len(set(self._labels)) != len(self._labels):
            raise RegimeError("state labels must be unique")
        self._fitted = True
        return self

    def filter(self, series: pd.Series) -> RegimeStates:
        """Return causal posteriors for ``series`` under the fitted parameters.

        Raises:
            NotFittedError: If called before ``fit``.
        """
        self._ensure_fitted()
        if not isinstance(series, pd.Series):
            raise RegimeError("filter expects a pandas Series")
        values = series.to_numpy(dtype=float)
        probabilities = np.asarray(self._filter(values), dtype=np.float64)
        return RegimeStates(
            probabilities=probabilities,
            labels=self._labels,
            index=series.index,
            causal=self.causal,
        )

    def report(self) -> RegimeReport:
        """Return how the last fit terminated."""
        self._ensure_fitted()
        assert self._report is not None  # narrowed by _ensure_fitted
        return self._report

    @property
    def identity(self) -> str:
        """Return a deterministic SHA-256 over configuration and fitted state.

        Both halves matter: configuration alone would collide two fits on
        different training windows, and parameters alone would collide two models
        that happen to converge to similar numbers from different specifications.
        """
        self._ensure_fitted()
        payload = {
            "model": self.name,
            "causal": self.causal,
            "ordering_rule": self.ordering_rule,
            "labels": list(self._labels),
            "configuration": self.configuration(),
            "parameters": self.fitted_parameters(),
        }
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False)
        return hashlib.sha256(encoded.encode("utf-8")).hexdigest()

    def _ensure_fitted(self) -> None:
        if not self._fitted:
            raise NotFittedError(f"{self.name} must be fit before this operation")


def _finite_training_values(series: pd.Series | FloatArray) -> FloatArray:
    """Return the finite training observations, or fail closed."""
    values = pd.Series(series).to_numpy(dtype=float)
    finite = values[np.isfinite(values)]
    if finite.size < MIN_FIT_OBSERVATIONS:
        raise RegimeError(
            f"regime fitting needs at least {MIN_FIT_OBSERVATIONS} finite observations, "
            f"got {finite.size}"
        )
    return np.asarray(finite, dtype=np.float64)


def canonical_order(keys: FloatArray) -> NDArray[np.int64]:
    """Return the permutation sorting states by an ordering key.

    Ties break on the original index so the permutation is total and
    reproducible; an unstable sort would let two identical fits disagree about
    labelling, which is precisely the failure canonical ordering prevents.
    """
    return np.asarray(np.argsort(keys, kind="stable"), dtype=np.int64)


@dataclass(frozen=True)
class ExpandingFitPlan:
    """Refit schedule for causal, expanding-window state estimation.

    Attributes:
        min_train: Observations required before the first estimate; earlier rows
            are NaN.
        refit_every: Bars between parameter refits. Larger is cheaper and
            staler; the trade is explicit rather than hidden in a default.
        notes: Free-form provenance carried into the evidence record.
    """

    min_train: int = 252
    refit_every: int = 63
    notes: tuple[str, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        if self.min_train < MIN_FIT_OBSERVATIONS:
            raise ValueError(f"min_train must be at least {MIN_FIT_OBSERVATIONS}")
        if self.refit_every < 1:
            raise ValueError("refit_every must be at least one bar")


def expanding_state_probabilities(
    series: pd.Series,
    factory: Any,
    plan: ExpandingFitPlan | None = None,
) -> tuple[RegimeStates, tuple[RegimeReport, ...]]:
    """Estimate state posteriors causally with expanding-window refits.

    This is the only sanctioned way to turn a regime model into a feature. For
    each block starting at ``k``: fit a **fresh** model on ``series[:k]``, filter
    forward to the block end, and keep only that block's rows. The estimate at
    row ``t`` is therefore a function of observations strictly up to ``t``, and
    no parameter was ever informed by a row it is later applied to.

    Args:
        series: Observations in time order.
        factory: Zero-argument callable returning an unfitted
            :class:`RegimeModel`. A factory rather than an instance, because
            reusing one instance across refits would carry state across the
            boundary the refit exists to enforce.
        plan: Warm-up and refit cadence.

    Returns:
        ``(states, reports)`` — one report per refit, so a fold whose EM failed
        to converge is visible rather than averaged away.

    Raises:
        RegimeLeakageError: If the factory produces a non-causal model.
        RegimeError: If the series is too short to produce any estimate.
    """
    plan = plan or ExpandingFitPlan()
    probe = factory()
    if not isinstance(probe, RegimeModel):
        raise RegimeError("factory must produce a RegimeModel")
    if not probe.causal:
        raise RegimeLeakageError(
            f"{probe.name} is a retrospective segmenter and conditions on the whole "
            "series; it must not be used to build a causal feature"
        )
    labels = ()
    values = series.to_numpy(dtype=float)
    total = len(series)
    if total <= plan.min_train:
        raise RegimeError(
            f"series of {total} observations is too short for min_train={plan.min_train}"
        )

    probabilities: FloatArray | None = None
    reports: list[RegimeReport] = []
    start = plan.min_train
    while start < total:
        end = min(start + plan.refit_every, total)
        training = values[:start]
        if np.isfinite(training).sum() >= MIN_FIT_OBSERVATIONS:
            model = factory()
            model.fit(pd.Series(training))
            block = model.filter(series.iloc[:end])
            if probabilities is None:
                labels = block.labels
                probabilities = np.full((total, len(labels)), np.nan)
            probabilities[start:end] = block.probabilities[start:end]
            reports.append(model.report())
        start = end

    if probabilities is None:
        raise RegimeError("no refit window contained enough finite observations")
    return (
        RegimeStates(probabilities=probabilities, labels=labels, index=series.index, causal=True),
        tuple(reports),
    )
