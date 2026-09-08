"""Proper scores and completeness reporting for shadow campaigns.

SF-S5-SL-MR4. Two reports can be computed from the same campaign and they answer
different questions:

**First-eligible** uses the earliest revision of each outcome that qualified —
what a consumer acting live could have known. **Latest-known** uses the newest
revision, including corrections published afterwards. Quoting a latest-known
number as though it had been available live is the most common way shadow
evidence flatters itself, so the two carry distinct identities, are never merged,
and the report records the delta between them explicitly.

**Scores are proper.** Brier and logarithmic scoring are both strictly proper:
the expected score is optimised by reporting your true belief, so a model cannot
improve its score by hedging. Log score is clipped at a declared epsilon because
an unbounded ``-log(0)`` from a single confident miss would otherwise dominate
every aggregate and destroy comparability.

**Too little evidence returns a verdict, not a number.** Below
``MIN_SCORED_FORECASTS`` the report is ``INSUFFICIENT_EVIDENCE``. A Brier score
over four forecasts is arithmetic, not evidence, and printing it invites someone
to compare it against a number computed over four hundred.

Uncertainty is reported by a deterministic date-block bootstrap: forecasts made
on the same day share market conditions and are not independent, so resampling
individual forecasts would understate the interval. Blocks are whole days,
resampled with a seeded generator so a rerun reproduces the interval exactly.
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

from quant_platform.shadow.contracts import (
    MIN_SCORED_FORECASTS,
    ProbabilityVector,
    ReportBasis,
    ShadowForecast,
    ShadowOutcome,
    ShadowValidationError,
    canonical_digest,
)

#: Log score is clipped here. One confident miss would otherwise contribute
#: infinity and make the aggregate meaningless rather than merely bad.
LOG_SCORE_EPSILON: Final = 1e-15

#: Bootstrap replicates for the date-block interval. Enough for a stable 5th and
#: 95th percentile without making a report expensive to produce.
BOOTSTRAP_REPLICATES: Final = 2_000

#: Percentiles reported for every scored metric.
INTERVAL_PERCENTILES: Final = (5.0, 95.0)


class ReportVerdict(StrEnum):
    """Whether a campaign carries enough evidence to report a score."""

    REPORTED = "reported"
    INSUFFICIENT_EVIDENCE = "insufficient_evidence"


def brier_score(distribution: ProbabilityVector, realized_label: str) -> float:
    """Return the multiclass Brier score for one forecast.

    The sum of squared differences between the predicted vector and the one-hot
    realisation. Lower is better; the range is ``[0, 2]`` for a multiclass vector.
    Strictly proper, so hedging cannot improve the expected value.

    Raises:
        ShadowValidationError: If ``realized_label`` is not a class in the
            distribution. A realised label outside the declared classes means
            the campaign and the outcome source disagree about the label space,
            which is a reconciliation fault rather than a score of zero.
    """
    if realized_label not in distribution.labels:
        raise ShadowValidationError(
            f"realised label {realized_label!r} is not among the forecast classes "
            f"{list(distribution.labels)}; the outcome source and the campaign disagree "
            "about the label space"
        )
    total = 0.0
    for label, probability in zip(distribution.labels, distribution.probabilities, strict=True):
        indicator = 1.0 if label == realized_label else 0.0
        total += (probability - indicator) ** 2
    return total


def log_score(distribution: ProbabilityVector, realized_label: str) -> float:
    """Return the clipped negative log score for one forecast.

    Lower is better. Clipped at :data:`LOG_SCORE_EPSILON` so a single confident
    miss contributes a large but finite penalty instead of infinity.

    Raises:
        ShadowValidationError: If the realised label is not a declared class.
    """
    probability = distribution.probability_of(realized_label)
    return -math.log(max(probability, LOG_SCORE_EPSILON))


@dataclass(frozen=True, slots=True)
class ScoredForecast:
    """One forecast paired with the outcome revision used to score it."""

    forecast_id: str
    symbol: str
    as_of_date: date
    revision: int
    brier: float
    log_loss: float

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-friendly record."""
        return {
            "forecast_id": self.forecast_id,
            "symbol": self.symbol,
            "as_of_date": self.as_of_date.isoformat(),
            "revision": self.revision,
            "brier": self.brier,
            "log_loss": self.log_loss,
        }


@dataclass(frozen=True)
class CampaignReport:
    """Scores, completeness, and uncertainty for one basis.

    ``verdict`` is ``INSUFFICIENT_EVIDENCE`` when fewer than
    :data:`~quant_platform.shadow.contracts.MIN_SCORED_FORECASTS` forecasts
    scored, and the score fields are ``None`` rather than a computed number, so
    a caller cannot accidentally format an unsupported figure.
    """

    campaign: str
    basis: ReportBasis
    verdict: ReportVerdict
    forecast_count: int
    scored_count: int
    pending_count: int
    missing_count: int
    mean_brier: float | None
    mean_log_loss: float | None
    brier_interval: tuple[float, float] | None
    baseline_brier: float | None
    distinct_days: int
    revision_count: int

    def to_dict(self) -> dict[str, Any]:
        """Return the complete JSON-friendly report."""
        return {
            "campaign": self.campaign,
            "basis": self.basis.value,
            "verdict": self.verdict.value,
            "forecast_count": self.forecast_count,
            "scored_count": self.scored_count,
            "pending_count": self.pending_count,
            "missing_count": self.missing_count,
            "completeness": (
                self.scored_count / self.forecast_count if self.forecast_count else 0.0
            ),
            "mean_brier": self.mean_brier,
            "mean_log_loss": self.mean_log_loss,
            "brier_interval": (
                list(self.brier_interval) if self.brier_interval is not None else None
            ),
            "baseline_brier": self.baseline_brier,
            "brier_minus_baseline": (
                self.mean_brier - self.baseline_brier
                if self.mean_brier is not None and self.baseline_brier is not None
                else None
            ),
            "distinct_days": self.distinct_days,
            "revision_count": self.revision_count,
            "minimum_scored_forecasts": MIN_SCORED_FORECASTS,
            "interpretation": (
                "Replay evidence over a delayed shadow campaign. Lower Brier and log "
                "loss are better. The interval is a deterministic date-block bootstrap: "
                "forecasts sharing a day share market conditions and are not "
                "independent. This is not prospective evidence, not proof of "
                "calibration, and not a claim of economic edge."
            ),
        }

    @property
    def identity(self) -> str:
        """Content identity, so two bases cannot be confused for one another."""
        return canonical_digest(self.to_dict())


def _select_revision(
    outcomes: Sequence[ShadowOutcome], *, basis: ReportBasis
) -> ShadowOutcome | None:
    """Return the outcome revision this basis scores against.

    First-eligible takes the lowest revision that was actually observed; a
    ``PENDING`` or ``MISSING`` revision is not eligible and does not become
    eligible by being earliest. Latest-known takes the highest observed revision.
    """
    observed = [item for item in outcomes if item.status.value == "observed"]
    if not observed:
        return None
    ordered = sorted(observed, key=lambda item: item.revision)
    return ordered[0] if basis is ReportBasis.FIRST_ELIGIBLE else ordered[-1]


def _date_block_interval(
    scored: Sequence[ScoredForecast], *, seed: int
) -> tuple[float, float] | None:
    """Return a deterministic date-block bootstrap interval for mean Brier.

    Resamples whole days with replacement rather than individual forecasts,
    because forecasts sharing an as-of date share market conditions. Treating
    them as independent would produce an interval narrower than the evidence
    supports.

    Returns ``None`` when fewer than two distinct days exist: a block bootstrap
    over one block resamples the same block every time and would report a
    zero-width interval, which asserts a precision the data cannot support.
    """
    blocks: dict[date, list[float]] = defaultdict(list)
    for item in scored:
        blocks[item.as_of_date].append(item.brier)
    if len(blocks) < 2:
        return None
    keys = sorted(blocks)
    block_means = np.array([float(np.mean(blocks[key])) for key in keys], dtype=float)
    weights = np.array([len(blocks[key]) for key in keys], dtype=float)
    generator = np.random.default_rng(np.random.SeedSequence([seed, len(keys)]))
    draws = generator.integers(0, len(keys), size=(BOOTSTRAP_REPLICATES, len(keys)))
    sampled_means = block_means[draws]
    sampled_weights = weights[draws]
    replicate_means = np.sum(sampled_means * sampled_weights, axis=1) / np.sum(
        sampled_weights, axis=1
    )
    low, high = np.percentile(replicate_means, INTERVAL_PERCENTILES)
    return (float(low), float(high))


def _climatology_baseline(
    forecasts: Sequence[ShadowForecast], realised: Mapping[str, str]
) -> float | None:
    """Return the Brier score of a constant base-rate forecast.

    The comparison that matters: a model that beats nothing beats a forecaster
    who ignores every feature and predicts the observed class frequency. Without
    it a Brier score is a number with no scale.
    """
    labels = forecasts[0].distribution.labels if forecasts else ()
    if not labels:
        return None
    counts = dict.fromkeys(labels, 0)
    for label in realised.values():
        if label in counts:
            counts[label] += 1
    total = sum(counts.values())
    if total == 0:
        return None
    rates = ProbabilityVector(
        labels=labels,
        probabilities=tuple(counts[label] / total for label in labels),
    )
    scores = [brier_score(rates, label) for label in realised.values() if label in counts]
    return float(np.mean(scores)) if scores else None


def build_report(
    campaign: str,
    forecasts: Sequence[ShadowForecast],
    outcomes: Mapping[str, Sequence[ShadowOutcome]],
    *,
    basis: ReportBasis,
    seed: int = 0,
) -> CampaignReport:
    """Score a campaign under one basis.

    Args:
        campaign: Campaign name, carried into the report identity.
        forecasts: Every sealed forecast in the campaign.
        outcomes: Forecast identity to its revision history, newest or oldest
            order irrelevant; the basis selects.
        basis: Which revision of each outcome to score against.
        seed: Seeds the date-block bootstrap so an interval reproduces exactly.

    Raises:
        ShadowValidationError: If an outcome references an unknown forecast, or
            a realised label falls outside the declared class space.
    """
    if not isinstance(basis, ReportBasis):
        raise ShadowValidationError("basis must be a ReportBasis member")
    known = {item.forecast_id: item for item in forecasts}
    unknown = sorted(set(outcomes) - set(known))
    if unknown:
        raise ShadowValidationError(
            f"{len(unknown)} outcome(s) reference forecasts absent from this campaign, "
            f"e.g. {unknown[:3]}"
        )

    scored: list[ScoredForecast] = []
    realised: dict[str, str] = {}
    pending = 0
    missing = 0
    revisions = 0
    for forecast_id, forecast in known.items():
        history = list(outcomes.get(forecast_id, ()))
        revisions += len(history)
        chosen = _select_revision(history, basis=basis)
        if chosen is None:
            if any(item.status.value == "pending" for item in history):
                pending += 1
            else:
                missing += 1
            continue
        forecast.assert_precedes(chosen.observed_at)
        realised[forecast_id] = chosen.realized_label
        scored.append(
            ScoredForecast(
                forecast_id=forecast_id,
                symbol=forecast.symbol,
                as_of_date=forecast.as_of.date(),
                revision=chosen.revision,
                brier=brier_score(forecast.distribution, chosen.realized_label),
                log_loss=log_score(forecast.distribution, chosen.realized_label),
            )
        )

    distinct_days = len({item.as_of_date for item in scored})
    if len(scored) < MIN_SCORED_FORECASTS:
        return CampaignReport(
            campaign=campaign,
            basis=basis,
            verdict=ReportVerdict.INSUFFICIENT_EVIDENCE,
            forecast_count=len(known),
            scored_count=len(scored),
            pending_count=pending,
            missing_count=missing,
            mean_brier=None,
            mean_log_loss=None,
            brier_interval=None,
            baseline_brier=None,
            distinct_days=distinct_days,
            revision_count=revisions,
        )

    return CampaignReport(
        campaign=campaign,
        basis=basis,
        verdict=ReportVerdict.REPORTED,
        forecast_count=len(known),
        scored_count=len(scored),
        pending_count=pending,
        missing_count=missing,
        mean_brier=float(np.mean([item.brier for item in scored])),
        mean_log_loss=float(np.mean([item.log_loss for item in scored])),
        brier_interval=_date_block_interval(scored, seed=seed),
        baseline_brier=_climatology_baseline(
            [known[item.forecast_id] for item in scored], realised
        ),
        distinct_days=distinct_days,
        revision_count=revisions,
    )


def revision_delta(first: CampaignReport, latest: CampaignReport) -> dict[str, Any]:
    """Report how much later revisions moved the score.

    A large delta means the campaign's apparent skill depends on corrections
    published after the fact. That is not necessarily wrong, but it is never the
    number to quote as live-available performance.

    Raises:
        ShadowValidationError: If the two reports are not the same campaign on
            opposite bases.
    """
    if first.campaign != latest.campaign:
        raise ShadowValidationError("reports describe different campaigns")
    if (
        first.basis is not ReportBasis.FIRST_ELIGIBLE
        or latest.basis is not ReportBasis.LATEST_KNOWN
    ):
        raise ShadowValidationError(
            "revision_delta compares a first-eligible report against a latest-known one"
        )
    movement = (
        latest.mean_brier - first.mean_brier
        if first.mean_brier is not None and latest.mean_brier is not None
        else None
    )
    return {
        "campaign": first.campaign,
        "first_eligible_identity": first.identity,
        "latest_known_identity": latest.identity,
        "identities_differ": first.identity != latest.identity,
        "first_eligible_brier": first.mean_brier,
        "latest_known_brier": latest.mean_brier,
        "brier_movement": movement,
        "scored_delta": latest.scored_count - first.scored_count,
        "note": (
            "A negative movement means later revisions improved the apparent score. "
            "Only the first-eligible figure was available to a live consumer."
        ),
    }


__all__ = [
    "BOOTSTRAP_REPLICATES",
    "INTERVAL_PERCENTILES",
    "LOG_SCORE_EPSILON",
    "CampaignReport",
    "ReportVerdict",
    "ScoredForecast",
    "brier_score",
    "build_report",
    "log_score",
    "revision_delta",
]
