"""Temporal, regime, and universe evidence with dependence-aware uncertainty.

SF-S4-MR7. This module answers "does the result hold up when the calendar is cut
differently?" and reports the answer with error bars that survive contact with
financial data.

**Why the usual error bar is wrong here.** The textbook standard error assumes
independent observations. Returns are not independent: volatility clusters, so
adjacent bars carry overlapping information, and an i.i.d. interval counts each
one as fresh evidence. The result is an interval too narrow — often by a large
factor — which makes an ordinary stretch of luck look decisive. Two estimators
are provided instead, and both are reported side by side with the naive one so
the gap is visible rather than assumed away:

- :func:`block_bootstrap_interval` resamples *contiguous blocks*, preserving
  short-range dependence inside each block.
- :func:`newey_west_standard_error` widens the standard error analytically using
  autocovariances out to a declared lag.

**Failure periods are reported, never trimmed.** :func:`period_evidence` returns
every declared interval, including the ones where the candidate lost money and
the ones too sparse to support an estimate. A sparse interval is marked sparse
and keeps its count; it is not dropped, because dropping short intervals is
precisely how a strategy's worst stretch leaves the record.

**No portfolio-level claim without a qualified candidate.**
:func:`portfolio_dependence_evidence` refuses to emit asset, sector,
concentration, or universe dependence for a candidate that has not been formally
qualified. Sprint 4 has no qualified candidate — the qualification decision is
SF-S4-MR9 — so the gate is structural rather than a matter of remembering not to
make the claim.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from statistics import NormalDist
from typing import Any, Final

import numpy as np
import pandas as pd

from alphaforge.robustness.periods import (
    MIN_PERIOD_OBSERVATIONS,
    FrozenPeriodSet,
    PeriodContractError,
    RegimeDefinition,
    label_regimes,
)
from alphaforge.robustness.universe import (
    AblationResult,
    PointInTimeUniverse,
    concentration_profile,
)

#: Refusal thresholds, not tuning knobs.
MAX_REPLICATES: Final = 20_000
MAX_BLOCK_LENGTH: Final = 2_520
MAX_LAG: Final = 512

#: Fewer observations than this cannot support a dependence-aware interval: a
#: block bootstrap needs enough blocks for the resample to vary at all.
MIN_BOOTSTRAP_OBSERVATIONS: Final = 30


class TemporalEvidenceError(ValueError):
    """Raised when an evidence request cannot be answered honestly."""


class UnqualifiedCandidateError(TemporalEvidenceError):
    """Raised when a portfolio-level claim is requested without qualification."""


def _clean_returns(returns: pd.Series, *, field_name: str = "returns") -> pd.Series:
    if not isinstance(returns, pd.Series):
        raise TemporalEvidenceError(f"{field_name} must be a pandas Series")
    if returns.empty:
        raise TemporalEvidenceError(f"{field_name} must be non-empty")
    if not returns.index.is_monotonic_increasing:
        raise TemporalEvidenceError(f"{field_name} must be sorted ascending")
    if returns.index.has_duplicates:
        raise TemporalEvidenceError(f"{field_name} index must be unique")
    values = returns.astype(float)
    if np.isinf(values.to_numpy(dtype=float)).any():
        raise TemporalEvidenceError(f"{field_name} contains infinite values")
    return values


# ---------------------------------------------------------------------------
# Dependence-aware uncertainty
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class UncertaintyInterval:
    """A mean estimate with an interval and the assumption that produced it.

    ``assumption`` travels with the number because the interval is only as good
    as the dependence structure it accounts for, and a reader comparing two
    intervals needs to know which one assumed independence.
    """

    point: float
    lower: float
    upper: float
    standard_error: float
    method: str
    assumption: str
    n_observations: int

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-friendly record."""
        return {
            "point": self.point,
            "lower": self.lower,
            "upper": self.upper,
            "standard_error": self.standard_error,
            "method": self.method,
            "assumption": self.assumption,
            "n_observations": self.n_observations,
        }


def naive_interval(returns: pd.Series, *, confidence: float = 0.95) -> UncertaintyInterval:
    """Return the i.i.d. interval — reported only as the baseline to beat.

    Included so the dependence-aware intervals can be compared against it. On
    autocorrelated data this interval is too narrow; it is never the interval to
    quote on its own.
    """
    values = _clean_returns(returns).dropna().to_numpy(dtype=float)
    if values.size < 2:
        raise TemporalEvidenceError("at least two observations are required for an interval")
    if not 0.5 <= confidence < 1.0:
        raise TemporalEvidenceError("confidence must lie in [0.5, 1.0)")
    point = float(values.mean())
    standard_error = float(values.std(ddof=1) / np.sqrt(values.size))
    half_width = _normal_quantile(confidence) * standard_error
    return UncertaintyInterval(
        point=point,
        lower=point - half_width,
        upper=point + half_width,
        standard_error=standard_error,
        method="iid_normal",
        assumption=(
            "observations independent and identically distributed; on autocorrelated "
            "returns this interval is too narrow and overstates significance"
        ),
        n_observations=int(values.size),
    )


def _normal_quantile(confidence: float) -> float:
    """Return the two-sided normal critical value for ``confidence``."""
    return float(NormalDist().inv_cdf(1.0 - (1.0 - confidence) / 2.0))


def newey_west_standard_error(returns: pd.Series, *, lag: int | None = None) -> float:
    """Return a HAC standard error of the mean, widened for autocorrelation.

    Uses Bartlett weights so the estimator is guaranteed non-negative. When
    ``lag`` is omitted it defaults to the standard ``floor(4 * (n/100)^(2/9))``
    rule, declared here rather than tuned per result.

    Raises:
        TemporalEvidenceError: On an unusable series or an out-of-range lag.
    """
    values = _clean_returns(returns).dropna().to_numpy(dtype=float)
    count = values.size
    if count < 2:
        raise TemporalEvidenceError("at least two observations are required for a HAC error")
    if lag is None:
        lag = int(np.floor(4.0 * (count / 100.0) ** (2.0 / 9.0)))
    if isinstance(lag, bool) or not isinstance(lag, int) or lag < 0:
        raise TemporalEvidenceError("lag must be a non-negative int")
    if lag > MAX_LAG:
        raise TemporalEvidenceError(f"lag exceeds the {MAX_LAG}-bar ceiling")
    lag = min(lag, count - 1)

    centered = values - values.mean()
    variance = float(centered @ centered) / count
    for shift in range(1, lag + 1):
        weight = 1.0 - shift / (lag + 1.0)
        covariance = float(centered[shift:] @ centered[:-shift]) / count
        variance += 2.0 * weight * covariance
    # Bartlett weights keep this non-negative in exact arithmetic; clamp only the
    # roundoff that can push a near-zero long-run variance fractionally below it.
    variance = max(variance, 0.0)
    return float(np.sqrt(variance / count))


def newey_west_interval(
    returns: pd.Series, *, lag: int | None = None, confidence: float = 0.95
) -> UncertaintyInterval:
    """Return a HAC interval for the mean return."""
    values = _clean_returns(returns).dropna()
    if not 0.5 <= confidence < 1.0:
        raise TemporalEvidenceError("confidence must lie in [0.5, 1.0)")
    standard_error = newey_west_standard_error(values, lag=lag)
    point = float(values.mean())
    half_width = _normal_quantile(confidence) * standard_error
    return UncertaintyInterval(
        point=point,
        lower=point - half_width,
        upper=point + half_width,
        standard_error=standard_error,
        method="newey_west_hac",
        assumption=(
            "autocorrelation decays within the declared lag; Bartlett weights keep the "
            "long-run variance non-negative"
        ),
        n_observations=int(len(values)),
    )


def block_bootstrap_interval(
    returns: pd.Series,
    *,
    block_length: int,
    replicates: int = 2_000,
    confidence: float = 0.95,
    seed: int = 0,
) -> UncertaintyInterval:
    """Return a moving-block bootstrap interval for the mean return.

    Contiguous blocks are resampled with replacement, so dependence *within* a
    block survives the resample. ``block_length`` must exceed the horizon over
    which returns stay correlated; too short a block reproduces the i.i.d.
    interval's overconfidence, which is why it is a required argument rather
    than a defaulted one.

    Raises:
        TemporalEvidenceError: On too few observations, an unusable block
            length, or an out-of-range replicate count.
    """
    values = _clean_returns(returns).dropna().to_numpy(dtype=float)
    count = values.size
    if count < MIN_BOOTSTRAP_OBSERVATIONS:
        raise TemporalEvidenceError(
            f"block bootstrap requires at least {MIN_BOOTSTRAP_OBSERVATIONS} observations, "
            f"got {count}; a shorter sample cannot support a dependence-aware interval"
        )
    if isinstance(block_length, bool) or not isinstance(block_length, int) or block_length < 1:
        raise TemporalEvidenceError("block_length must be a positive int")
    if block_length > MAX_BLOCK_LENGTH:
        raise TemporalEvidenceError(f"block_length exceeds the {MAX_BLOCK_LENGTH}-bar ceiling")
    if block_length > count:
        raise TemporalEvidenceError("block_length cannot exceed the sample length")
    if isinstance(replicates, bool) or not isinstance(replicates, int) or replicates < 100:
        raise TemporalEvidenceError("replicates must be an int of at least 100")
    if replicates > MAX_REPLICATES:
        raise TemporalEvidenceError(f"replicates exceeds the {MAX_REPLICATES} ceiling")
    if not 0.5 <= confidence < 1.0:
        raise TemporalEvidenceError("confidence must lie in [0.5, 1.0)")

    generator = np.random.default_rng(np.random.SeedSequence([int(seed), block_length, count]))
    n_blocks = int(np.ceil(count / block_length))
    n_starts = count - block_length + 1
    offsets = np.arange(block_length)
    means = np.empty(replicates, dtype=float)
    for replicate in range(replicates):
        starts = generator.integers(0, n_starts, size=n_blocks)
        sample = values[(starts[:, None] + offsets[None, :]).ravel()][:count]
        means[replicate] = sample.mean()

    tail = (1.0 - confidence) / 2.0
    lower, upper = np.quantile(means, [tail, 1.0 - tail])
    return UncertaintyInterval(
        point=float(values.mean()),
        lower=float(lower),
        upper=float(upper),
        standard_error=float(means.std(ddof=1)),
        method=f"moving_block_bootstrap_{block_length}",
        assumption=(
            f"dependence decays within {block_length} bars; a block shorter than the "
            "true correlation horizon reproduces the i.i.d. interval's overconfidence"
        ),
        n_observations=int(count),
    )


def compare_uncertainty(returns: pd.Series, *, block_length: int, seed: int = 0) -> dict[str, Any]:
    """Report all three intervals together, with the width inflation between them.

    The ratio is the point of the exercise: it says how much of the naive
    interval's tightness was an artefact of assuming independence.
    """
    naive = naive_interval(returns)
    hac = newey_west_interval(returns)
    block = block_bootstrap_interval(returns, block_length=block_length, seed=seed)
    naive_width = naive.upper - naive.lower
    return {
        "iid": naive.to_dict(),
        "newey_west": hac.to_dict(),
        "block_bootstrap": block.to_dict(),
        "width_inflation": {
            "newey_west_over_iid": (
                float((hac.upper - hac.lower) / naive_width) if naive_width > 0.0 else float("nan")
            ),
            "block_over_iid": (
                float((block.upper - block.lower) / naive_width)
                if naive_width > 0.0
                else float("nan")
            ),
        },
        "note": (
            "an inflation factor above 1 is the portion of the i.i.d. interval's tightness "
            "that came from assuming independence rather than from the data"
        ),
    }


# ---------------------------------------------------------------------------
# Per-period and per-regime evidence
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class PeriodOutcome:
    """One interval's outcome, including the ones that failed."""

    label: str
    n_observations: int
    mean_return: float
    total_return: float
    sparse: bool
    interval: UncertaintyInterval | None
    note: str

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-friendly record."""
        return {
            "label": self.label,
            "n_observations": self.n_observations,
            "mean_return": self.mean_return,
            "total_return": self.total_return,
            "sparse": self.sparse,
            "interval": None if self.interval is None else self.interval.to_dict(),
            "note": self.note,
        }


def _outcome_for(
    label: str, slice_returns: pd.Series, *, block_length: int, seed: int
) -> PeriodOutcome:
    count = int(slice_returns.notna().sum())
    if count == 0:
        return PeriodOutcome(
            label=label,
            n_observations=0,
            mean_return=float("nan"),
            total_return=float("nan"),
            sparse=True,
            interval=None,
            note="no observations in this interval; reported as empty rather than omitted",
        )
    clean = slice_returns.dropna()
    mean_return = float(clean.mean())
    total_return = float(clean.sum())
    sparse = count < MIN_PERIOD_OBSERVATIONS
    interval: UncertaintyInterval | None = None
    note = ""
    if count >= MIN_BOOTSTRAP_OBSERVATIONS and block_length <= count:
        interval = block_bootstrap_interval(clean, block_length=block_length, seed=seed)
        note = "dependence-aware interval from a moving-block bootstrap"
    else:
        note = (
            f"too short for a dependence-aware interval ({count} observations, "
            f"{MIN_BOOTSTRAP_OBSERVATIONS} required); the point estimate is reported "
            "without one rather than with a misleadingly tight i.i.d. interval"
        )
    return PeriodOutcome(
        label=label,
        n_observations=count,
        mean_return=mean_return,
        total_return=total_return,
        sparse=sparse,
        interval=interval,
        note=note,
    )


def period_evidence(
    returns: pd.Series,
    period_set: FrozenPeriodSet,
    *,
    block_length: int = 20,
    seed: int = 0,
) -> dict[str, Any]:
    """Report every declared interval's outcome, failures included.

    Intervals where the candidate lost money, and intervals too sparse to
    support an interval estimate, appear in the output with their counts. None
    are dropped: an evidence set that contains only the good periods is not
    evidence.
    """
    values = _clean_returns(returns)
    labels = period_set.assign(values.index)
    outcomes: list[PeriodOutcome] = []
    for interval in period_set.intervals:
        mask = labels == interval.label
        outcomes.append(
            _outcome_for(
                interval.label,
                values.loc[mask.to_numpy(dtype=bool)],
                block_length=block_length,
                seed=seed,
            )
        )
    unlabelled = int((labels.isna()).sum())
    losing = [item.label for item in outcomes if item.n_observations > 0 and item.mean_return < 0.0]
    return {
        "period_set": period_set.name,
        "period_set_identity": period_set.identity,
        "outcomes": [item.to_dict() for item in outcomes],
        "failure_periods": losing,
        "sparse_periods": [item.label for item in outcomes if item.sparse],
        "unlabelled_observations": unlabelled,
        "reporting_rule": (
            "every declared interval is reported, including losing and sparse ones; "
            "selecting favourable periods after inspecting results invalidates the evidence"
        ),
    }


def matched_family_evidence(
    families: Mapping[str, pd.Series],
    period_set: FrozenPeriodSet,
    *,
    block_length: int = 20,
    seed: int = 0,
) -> dict[str, Any]:
    """Report every family on the **same** windows, matched observation by observation.

    Two families evaluated over different spans cannot be compared: the one that
    happened to cover a calmer stretch looks better for a reason that has nothing
    to do with the family. This function restricts every family to the timestamps
    they *all* share, reports the matched span and how much each family gave up
    to reach it, and evaluates them on identical intervals.

    Dropping to the shared span is itself a decision worth seeing, so
    ``coverage_sacrificed`` names the observations each family lost — a family
    that loses most of its history to matching is being compared on a fragment.

    Raises:
        TemporalEvidenceError: If fewer than two families are supplied or the
            families share no timestamps.
    """
    if not isinstance(families, Mapping) or len(families) < 2:
        raise TemporalEvidenceError(
            "matched evidence requires at least two families; a single family has "
            "nothing to be matched against"
        )
    cleaned = {
        name: _clean_returns(series, field_name=f"family {name!r}")
        for name, series in families.items()
    }
    series_iter = iter(cleaned.values())
    shared = next(series_iter).dropna().index
    for series in series_iter:
        shared = shared.intersection(series.dropna().index)
    if len(shared) == 0:
        raise TemporalEvidenceError(
            "families share no timestamps; there is no window on which they can be "
            "compared, and evaluating them on different spans compares the spans"
        )
    shared = shared.sort_values()
    reports = {
        name: period_evidence(series.loc[shared], period_set, block_length=block_length, seed=seed)
        for name, series in cleaned.items()
    }
    return {
        "period_set": period_set.name,
        "period_set_identity": period_set.identity,
        "matched_observations": int(len(shared)),
        "matched_span": [str(shared[0]), str(shared[-1])],
        "coverage_sacrificed": {
            name: int(len(series.dropna()) - len(shared)) for name, series in cleaned.items()
        },
        "families": reports,
        "matching_rule": (
            "every family is evaluated on the identical shared timestamps; families "
            "compared over different spans compare the spans, not the families"
        ),
    }


def regime_evidence(
    returns: pd.Series,
    conditioning: pd.Series,
    definition: RegimeDefinition,
    *,
    block_length: int = 20,
    seed: int = 0,
) -> dict[str, Any]:
    """Report per-regime outcomes under one frozen, causal regime definition.

    The regime labels come from :func:`~alphaforge.robustness.periods.label_regimes`,
    which sees only the conditioning series. Bars in the warm-up window carry no
    label and are counted separately rather than folded into a neighbouring
    regime.
    """
    values = _clean_returns(returns)
    labels = label_regimes(conditioning, definition)
    shared = values.index.intersection(labels.index)
    if len(shared) == 0:
        raise TemporalEvidenceError(
            "returns and conditioning series share no timestamps; a regime label cannot "
            "be attached to a return it does not cover"
        )
    aligned_returns = values.loc[shared]
    aligned_labels = labels.loc[shared]
    outcomes = [
        _outcome_for(
            label,
            aligned_returns.loc[(aligned_labels == label).to_numpy(dtype=bool)],
            block_length=block_length,
            seed=seed,
        )
        for label in definition.labels
    ]
    return {
        "definition": definition.to_dict(),
        "outcomes": [item.to_dict() for item in outcomes],
        "unlabelled_observations": int(aligned_labels.isna().sum()),
        "failure_regimes": [
            item.label for item in outcomes if item.n_observations > 0 and item.mean_return < 0.0
        ],
        "coverage_note": (
            "warm-up bars carry no regime label and are counted separately rather than "
            "folded into a neighbouring regime"
        ),
    }


def regime_definition_sensitivity(
    returns: pd.Series,
    conditioning: pd.Series,
    definitions: Sequence[RegimeDefinition],
    *,
    block_length: int = 20,
    seed: int = 0,
) -> dict[str, Any]:
    """Report how the regime conclusion moves across alternative frozen definitions.

    Regime boundaries are a modelling choice, and a conclusion that survives only
    one particular window and threshold set is a property of that choice rather
    than of the market. All definitions must be declared up front — this function
    is for reporting the spread, not for picking the definition that reads best.
    """
    if not definitions:
        raise TemporalEvidenceError("at least one regime definition is required")
    if len({definition.name for definition in definitions}) != len(definitions):
        raise TemporalEvidenceError("regime definition names must be unique")
    reports = [
        regime_evidence(returns, conditioning, definition, block_length=block_length, seed=seed)
        for definition in definitions
    ]
    failure_counts = [len(report["failure_regimes"]) for report in reports]
    return {
        "definitions": [definition.identity for definition in definitions],
        "reports": reports,
        "conclusion_is_definition_dependent": len(set(failure_counts)) > 1,
        "note": (
            "a conclusion that holds under one regime definition but not another is a "
            "property of the definition, not of the market"
        ),
    }


# ---------------------------------------------------------------------------
# The portfolio-level gate
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class QualifiedCandidate:
    """Proof that a candidate cleared the formal qualification decision.

    Constructing one of these by hand does not qualify a candidate — the
    qualification decision is SF-S4-MR9, and this record is the receipt it
    issues. It exists so :func:`portfolio_dependence_evidence` can require
    something more specific than a boolean flag: a named decision, a plan hash,
    and the corrected p-value the decision rested on.
    """

    candidate_id: str
    decision_id: str
    plan_hash: str
    adjusted_p_value: float
    alpha: float

    def __post_init__(self) -> None:
        for field_name in ("candidate_id", "decision_id"):
            value = getattr(self, field_name)
            if not isinstance(value, str) or not value.strip() or value != value.strip():
                raise UnqualifiedCandidateError(f"{field_name} must be a non-empty identifier")
        if not isinstance(self.plan_hash, str) or len(self.plan_hash) != 64:
            raise UnqualifiedCandidateError(
                "plan_hash must be the full SHA-256 of the frozen research plan the "
                "qualification decision was made under"
            )
        for field_name in ("adjusted_p_value", "alpha"):
            value = getattr(self, field_name)
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise UnqualifiedCandidateError(f"{field_name} must be a real number")
        if not 0.0 <= self.adjusted_p_value <= 1.0:
            raise UnqualifiedCandidateError("adjusted_p_value must lie in [0, 1]")
        if not 0.0 < self.alpha < 1.0:
            raise UnqualifiedCandidateError("alpha must lie in (0, 1)")
        if self.adjusted_p_value > self.alpha:
            raise UnqualifiedCandidateError(
                f"candidate {self.candidate_id!r} has adjusted p-value "
                f"{self.adjusted_p_value:.4f} above alpha {self.alpha:.4f}; it did not "
                "clear qualification and no portfolio-level claim may be made for it"
            )

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-friendly receipt."""
        return {
            "candidate_id": self.candidate_id,
            "decision_id": self.decision_id,
            "plan_hash": self.plan_hash,
            "adjusted_p_value": self.adjusted_p_value,
            "alpha": self.alpha,
        }


def portfolio_dependence_evidence(
    returns: pd.Series,
    contributions: Mapping[str, float],
    universe: PointInTimeUniverse,
    ablations: Sequence[AblationResult],
    *,
    qualified: QualifiedCandidate | None,
    block_length: int = 20,
    seed: int = 0,
) -> dict[str, Any]:
    """Report portfolio-level dependence — only for a qualified candidate.

    Asset, sector, concentration, universe, liquidity, and top-contributor
    dependence are all statements *about a portfolio that is claimed to work*.
    Emitting them for an unqualified candidate dresses an unproven result in the
    apparatus of a proven one, so ``qualified=None`` raises rather than returning
    a hedged report. Sprint 4 has no qualified candidate; the refusal is the
    expected path until SF-S4-MR9 issues one.

    Raises:
        UnqualifiedCandidateError: When ``qualified`` is ``None``.
    """
    if qualified is None:
        raise UnqualifiedCandidateError(
            "no qualified candidate: portfolio-level asset, sector, concentration, and "
            "universe dependence claims require a candidate that cleared the formal "
            "qualification decision. Absent one, no portfolio-level claim is made."
        )
    values = _clean_returns(returns)
    hindsight = [item for item in ablations if item.hindsight_based]
    return {
        "candidate": qualified.to_dict(),
        "uncertainty": compare_uncertainty(values, block_length=block_length, seed=seed),
        "concentration": concentration_profile(contributions),
        "universe": universe.to_dict(),
        "ablations": [item.to_dict() for item in ablations],
        "hindsight_based_ablations": [item.name for item in hindsight],
        "interpretation": (
            "hindsight-based ablations are fragility probes, not achievable returns; "
            "the removed names could only be identified after the fact"
        ),
        "simulation_only": True,
    }


def temporal_robustness_report(
    returns: pd.Series,
    period_set: FrozenPeriodSet,
    conditioning: pd.Series,
    definitions: Sequence[RegimeDefinition],
    *,
    expected_period_identity: str,
    block_length: int = 20,
    seed: int = 0,
) -> dict[str, Any]:
    """Assemble the full candidate-level temporal report against frozen artifacts.

    The period set is verified against the identity frozen before evaluation, so
    a calendar re-cut after seeing results fails here rather than passing
    unnoticed.

    Raises:
        PeriodContractError: If the period set diverges from its frozen identity.
    """
    from alphaforge.robustness.periods import verify_frozen_periods

    verify_frozen_periods(period_set, expected_period_identity)
    values = _clean_returns(returns)
    return {
        "uncertainty": compare_uncertainty(values, block_length=block_length, seed=seed),
        "periods": period_evidence(values, period_set, block_length=block_length, seed=seed),
        "regimes": regime_definition_sensitivity(
            values, conditioning, definitions, block_length=block_length, seed=seed
        ),
        "portfolio_claim": (
            "withheld: portfolio-level dependence requires a qualified candidate "
            "(see portfolio_dependence_evidence)"
        ),
        "simulation_only": True,
    }


__all__ = [
    "MAX_BLOCK_LENGTH",
    "MAX_LAG",
    "MAX_REPLICATES",
    "MIN_BOOTSTRAP_OBSERVATIONS",
    "PeriodContractError",
    "PeriodOutcome",
    "QualifiedCandidate",
    "TemporalEvidenceError",
    "UncertaintyInterval",
    "UnqualifiedCandidateError",
    "block_bootstrap_interval",
    "compare_uncertainty",
    "matched_family_evidence",
    "naive_interval",
    "newey_west_interval",
    "newey_west_standard_error",
    "period_evidence",
    "portfolio_dependence_evidence",
    "regime_definition_sensitivity",
    "regime_evidence",
    "temporal_robustness_report",
]
