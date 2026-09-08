"""Dependence-aware inference for champion-challenger decisions.

SF-S5-SL-MR5. The statistics here are deliberately conservative, because the
failure they guard against is approving a challenger that is not actually better.

**Resampling is by date block, not by forecast.** Forecasts made on the same day
share market conditions. Treating them as independent inflates the effective
sample size and narrows every interval, which is exactly the error that makes a
coin-flip challenger look significant.

The scheme is a **circular moving-block** bootstrap over dates. Two properties
matter and both are deliberate:

* *Moving blocks* rather than independent days, because the day-level series is
  itself serially dependent -- a regime lasting a week correlates consecutive
  days' score differences, and resampling days independently destroys it.
* *Circular* wrapping, so every date can start a block. A non-circular scheme
  can only start within the first ``n - L + 1`` dates, under-sampling the end of
  the window, which is the most recent evidence and the part a promotion leans
  on hardest.

Block length is **derived, never supplied**: ``max(horizon, ceil(n ** (1/3)))``.
A caller who could choose it could choose the length that produced the narrowest
interval. Everything is driven by a seeded generator, so a rerun reproduces the
interval exactly.

**Superiority and non-inferiority are different questions.** Superiority asks
whether the challenger is better than the champion. Non-inferiority asks whether
it is no worse by more than a margin declared *before* the comparison. Promoting
on non-inferiority requires a margin; promoting on a superiority test that
happened to clear zero is how a tie becomes a promotion.

**Multiplicity is corrected across the whole family.** Testing two metrics on
two questions is four tests, and the probability that at least one clears by
chance is not the nominal alpha. Holm-Bonferroni controls the familywise error
rate under arbitrary dependence, which matters because the metrics here are
correlated and independence-assuming corrections would be anticonservative.

**Low power returns a verdict, not a p-value.** A test with too few blocks to
detect the declared margin cannot produce evidence either way, and reporting its
p-value invites reading a non-significant result as evidence of equivalence.
"""

from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import date
from enum import StrEnum
from typing import Any, Final

import numpy as np

from quant_platform.governance.comparison import ComparisonError, PairedCohort

#: Replicates for the date-block bootstrap. Enough for stable tail quantiles at
#: the alphas used here without making a decision expensive to reproduce.
BOOTSTRAP_REPLICATES: Final = 10_000

#: Fewer distinct days than this cannot support a block bootstrap: the resample
#: degenerates toward the observed blocks and the interval understates spread.
MIN_BLOCKS: Final = 10

#: A test with fewer effective observations than this is underpowered for any
#: margin worth declaring, and returns a verdict instead of a p-value.
MIN_EFFECTIVE_OBSERVATIONS: Final = 50


class InferenceError(ComparisonError):
    """Raised when an inference request cannot be answered honestly."""


class TestVerdict(StrEnum):
    """Outcome of one hypothesis test."""

    FAVOURS_CHALLENGER = "favours_challenger"
    FAVOURS_CHAMPION = "favours_champion"
    INCONCLUSIVE = "inconclusive"
    UNDERPOWERED = "underpowered"


@dataclass(frozen=True, slots=True)
class Margin:
    """A pre-declared indifference margin for a non-inferiority test.

    ``value`` is expressed in the metric's own units and must be positive: a
    zero margin turns non-inferiority into a superiority test with a misleading
    name, and a negative one asserts the challenger must be worse.
    """

    metric: str
    value: float

    def __post_init__(self) -> None:
        if not isinstance(self.metric, str) or not self.metric.strip():
            raise InferenceError("margin metric must be a non-empty string")
        if isinstance(self.value, bool) or not isinstance(self.value, (int, float)):
            raise InferenceError("margin value must be a real number")
        if not math.isfinite(self.value) or self.value <= 0.0:
            raise InferenceError(
                "margin must be finite and strictly positive; a zero margin makes a "
                "non-inferiority test a superiority test wearing the wrong name"
            )

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-friendly declaration."""
        return {"metric": self.metric, "value": self.value}


@dataclass(frozen=True)
class TestResult:
    """One dependence-aware test with its interval and raw p-value.

    ``p_value`` is uncorrected. Family correction happens in
    :func:`holm_adjust`, so a caller cannot accidentally report a raw p-value
    from a family of tests as though it were the corrected one.
    """

    name: str
    metric: str
    verdict: TestVerdict
    point_estimate: float | None
    # A one-sided test leaves one end unbounded. That end is None rather than
    # an infinity: infinities are not JSON-representable, and 'no bound' is
    # the honest description of a bound that does not exist.
    interval: tuple[float | None, float | None] | None
    p_value: float | None
    blocks: int
    observations: int
    margin: float | None

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-friendly record."""
        return {
            "name": self.name,
            "metric": self.metric,
            "verdict": self.verdict.value,
            "point_estimate": self.point_estimate,
            "interval": list(self.interval) if self.interval is not None else None,
            "p_value_uncorrected": self.p_value,
            "blocks": self.blocks,
            "observations": self.observations,
            "margin": self.margin,
            "direction": "negative point estimate favours the challenger",
        }


def _blocks_from(cohort: PairedCohort) -> dict[date, list[float]]:
    """Group per-pair Brier differences into whole-day blocks."""
    grouped: dict[date, list[float]] = defaultdict(list)
    for item in cohort.pairs:
        grouped[item.as_of_date].append(item.brier_difference)
    return dict(grouped)


def _cohort_horizon(cohort: PairedCohort) -> int:
    """Return the cohort's single forecast horizon.

    Raises:
        InferenceError: If the cohort mixes horizons. A block length derived
            from one horizon does not describe a sample containing another, and
            averaging the two would understate dependence for the longer one.
    """
    horizons = {item.key.horizon_days for item in cohort.pairs}
    if len(horizons) != 1:
        raise InferenceError(
            f"cohort mixes forecast horizons {sorted(horizons)}; the block length is "
            "derived from the horizon and cannot describe several at once"
        )
    return horizons.pop()


def block_length(*, horizon_days: int, date_count: int) -> int:
    """Return the frozen block length for a cohort.

    ``max(horizon, ceil(n ** (1/3)))`` -- the horizon because overlapping
    forecasts remain dependent for at least that long, and the cube root because
    it is the standard rate at which block length must grow with sample size for
    a moving-block bootstrap to stay consistent. Taking the larger keeps both
    guarantees rather than trading one for the other.

    The length is derived, never passed in: a caller who could choose it could
    choose the one that produced the narrowest interval.

    Raises:
        InferenceError: On a non-positive horizon or date count.
    """
    if isinstance(horizon_days, bool) or not isinstance(horizon_days, int) or horizon_days < 1:
        raise InferenceError("horizon_days must be a positive int")
    if isinstance(date_count, bool) or not isinstance(date_count, int) or date_count < 1:
        raise InferenceError("date_count must be a positive int")
    return max(horizon_days, int(math.ceil(date_count ** (1 / 3))))


def _block_bootstrap(
    blocks: Mapping[date, Sequence[float]],
    *,
    seed: int,
    replicates: int,
    horizon_days: int,
) -> np.ndarray:
    """Return replicate means from a circular moving-block bootstrap over dates.

    **Circular** so every date starts a block equally often. A non-circular
    moving-block scheme can only start a block at one of the first ``n - L + 1``
    dates, which under-samples the end of the window -- and the end of the
    window is the most recent evidence, the part a promotion decision leans on
    hardest.

    **Moving blocks** rather than independent days, because the day-level series
    is itself serially dependent: a regime that lasts a week makes consecutive
    days' score differences correlated, and resampling days independently
    destroys exactly that structure.

    Blocks are weighted by the number of forecasts on each resampled date, so a
    day carrying 40 forecasts counts for more than one carrying 4.
    """
    keys = sorted(blocks)
    date_count = len(keys)
    means = np.array([float(np.mean(blocks[key])) for key in keys], dtype=float)
    weights = np.array([len(blocks[key]) for key in keys], dtype=float)

    length = block_length(horizon_days=horizon_days, date_count=date_count)
    block_count = math.ceil(date_count / length)
    generator = np.random.default_rng(np.random.SeedSequence([seed, date_count, length]))

    # One random start per block per replicate; the offsets within a block are
    # deterministic, which is what makes it a *block* rather than a resample.
    starts = generator.integers(0, date_count, size=(replicates, block_count))
    offsets = np.arange(length)
    indices = (starts[:, :, None] + offsets[None, None, :]) % date_count
    flattened = indices.reshape(replicates, block_count * length)[:, :date_count]

    sampled = means[flattened]
    sampled_weights = weights[flattened]
    return np.asarray(np.sum(sampled * sampled_weights, axis=1) / np.sum(sampled_weights, axis=1))


def superiority_test(cohort: PairedCohort, *, alpha: float = 0.05, seed: int = 0) -> TestResult:
    """Test whether the challenger's mean Brier is below the champion's.

    The p-value is the bootstrap proportion of replicates in which the
    challenger did **not** win, computed as ``(exceedances + 1) / (replicates + 1)``
    so it can never be reported as exactly zero. A p-value of zero asserts
    impossibility, which a finite resample cannot establish.

    Raises:
        InferenceError: On an unusable alpha or an incomparable cohort.
    """
    if not 0.0 < alpha < 0.5:
        raise InferenceError("alpha must lie in (0, 0.5)")
    if not cohort.comparable:
        raise InferenceError(f"cohort is not comparable: {cohort.incomparable_reason}")
    blocks = _blocks_from(cohort)
    observations = len(cohort.pairs)
    if len(blocks) < MIN_BLOCKS or observations < MIN_EFFECTIVE_OBSERVATIONS:
        return TestResult(
            name="superiority",
            metric="brier",
            verdict=TestVerdict.UNDERPOWERED,
            point_estimate=None,
            interval=None,
            p_value=None,
            blocks=len(blocks),
            observations=observations,
            margin=None,
        )
    replicates = _block_bootstrap(
        blocks,
        seed=seed,
        replicates=BOOTSTRAP_REPLICATES,
        horizon_days=_cohort_horizon(cohort),
    )
    point = float(np.mean(replicates))
    low, high = np.percentile(replicates, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    exceedances = int(np.sum(replicates >= 0.0))
    p_value = (exceedances + 1) / (BOOTSTRAP_REPLICATES + 1)
    if high < 0.0:
        verdict = TestVerdict.FAVOURS_CHALLENGER
    elif low > 0.0:
        verdict = TestVerdict.FAVOURS_CHAMPION
    else:
        verdict = TestVerdict.INCONCLUSIVE
    return TestResult(
        name="superiority",
        metric="brier",
        verdict=verdict,
        point_estimate=point,
        interval=(float(low), float(high)),
        p_value=p_value,
        blocks=len(blocks),
        observations=observations,
        margin=None,
    )


def non_inferiority_test(
    cohort: PairedCohort, margin: Margin, *, alpha: float = 0.05, seed: int = 0
) -> TestResult:
    """Test whether the challenger is worse by no more than a declared margin.

    The margin must have been declared before the comparison; this function
    cannot verify that, which is why the frozen policy records it and the gate
    checks the policy identity.

    Raises:
        InferenceError: On an unusable alpha or an incomparable cohort.
    """
    if not 0.0 < alpha < 0.5:
        raise InferenceError("alpha must lie in (0, 0.5)")
    if not cohort.comparable:
        raise InferenceError(f"cohort is not comparable: {cohort.incomparable_reason}")
    blocks = _blocks_from(cohort)
    observations = len(cohort.pairs)
    if len(blocks) < MIN_BLOCKS or observations < MIN_EFFECTIVE_OBSERVATIONS:
        return TestResult(
            name="non_inferiority",
            metric=margin.metric,
            verdict=TestVerdict.UNDERPOWERED,
            point_estimate=None,
            interval=None,
            p_value=None,
            blocks=len(blocks),
            observations=observations,
            margin=margin.value,
        )
    replicates = _block_bootstrap(
        blocks,
        seed=seed,
        replicates=BOOTSTRAP_REPLICATES,
        horizon_days=_cohort_horizon(cohort),
    )
    point = float(np.mean(replicates))
    # One-sided: the whole question is whether the difference stays below +margin.
    high = float(np.percentile(replicates, 100 * (1 - alpha)))
    exceedances = int(np.sum(replicates >= margin.value))
    p_value = (exceedances + 1) / (BOOTSTRAP_REPLICATES + 1)
    verdict = TestVerdict.FAVOURS_CHALLENGER if high < margin.value else TestVerdict.INCONCLUSIVE
    return TestResult(
        name="non_inferiority",
        metric=margin.metric,
        verdict=verdict,
        point_estimate=point,
        interval=(None, high),
        p_value=p_value,
        blocks=len(blocks),
        observations=observations,
        margin=margin.value,
    )


def holm_adjust(p_values: Mapping[str, float], *, alpha: float = 0.05) -> dict[str, Any]:
    """Apply Holm-Bonferroni across a complete family of tests.

    Holm controls the familywise error rate under **arbitrary** dependence,
    which is required here: Brier and log score on the same cohort are strongly
    correlated, and a correction assuming independence would be anticonservative
    exactly when it matters.

    Raises:
        InferenceError: On an empty family or a p-value outside ``[0, 1]``.
    """
    if not p_values:
        raise InferenceError(
            "the family must be complete and non-empty; correcting a subset chosen "
            "after seeing results is not a correction"
        )
    for name, value in p_values.items():
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise InferenceError(f"p-value for {name!r} must be a real number")
        if not math.isfinite(value) or not 0.0 <= value <= 1.0:
            raise InferenceError(f"p-value for {name!r} must be finite in [0, 1]")
    ordered = sorted(p_values.items(), key=lambda item: (item[1], item[0]))
    count = len(ordered)
    adjusted: dict[str, float] = {}
    running = 0.0
    for index, (name, value) in enumerate(ordered):
        running = max(running, min(1.0, (count - index) * value))
        adjusted[name] = running
    return {
        "method": "holm_bonferroni",
        "alpha": alpha,
        "family_size": count,
        "adjusted": {name: adjusted[name] for name, _ in sorted(p_values.items())},
        "rejected": {name: adjusted[name] <= alpha for name, _ in sorted(p_values.items())},
        "note": (
            "Holm controls the familywise error rate under arbitrary dependence. The "
            "metrics tested here are correlated, so an independence-assuming correction "
            "would be anticonservative."
        ),
    }


__all__ = [
    "BOOTSTRAP_REPLICATES",
    "MIN_BLOCKS",
    "MIN_EFFECTIVE_OBSERVATIONS",
    "InferenceError",
    "Margin",
    "TestResult",
    "TestVerdict",
    "block_length",
    "holm_adjust",
    "non_inferiority_test",
    "superiority_test",
]
