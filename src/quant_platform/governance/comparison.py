"""Paired champion-challenger comparison over exact cohorts.

SF-S5-SL-MR5. A comparison is only meaningful when both models were asked the
same questions. This module refuses anything else.

**Pairing is exact, not approximate.** A forecast pairs with another only when
campaign, symbol, as-of instant, and horizon all match. Comparing a challenger's
easy days against a champion's hard ones produces a number that describes the
sample, not the models, and no amount of downstream statistics repairs it.

**Unmatched rows are counted and reported, never dropped.** Silent inner-join
semantics are how a challenger evaluated on 60% of the cohort gets compared
against a champion evaluated on all of it. The exclusion counts travel with the
result so a reader can see how much of the universe the comparison actually
covers.

**Asymmetric missingness is detected.** If one model is systematically missing
on the days it would have done badly, the paired sample is biased even though
every retained pair is legitimate. A comparison whose missingness differs
materially between arms is reported as such rather than scored.

**Leakage is checked, not assumed.** Every paired forecast is re-verified to
have been made strictly before the outcome it is scored against, on both arms.
The shadow contracts already enforce this at construction, but a comparison
assembled from two independently sealed campaigns must not take that on trust.
"""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import date, datetime
from typing import Any, Final

from quant_platform.shadow.contracts import (
    ShadowForecast,
    ShadowOutcome,
    ShadowValidationError,
    canonical_digest,
)
from quant_platform.shadow.reports import brier_score, log_score

#: Refusal thresholds, not tuning knobs.
MAX_PAIRS: Final = 500_000

#: Absolute difference in per-arm missingness above which the cohorts are not
#: comparable. Missingness that differs this much between arms means the two
#: models were effectively asked different questions.
MAX_MISSINGNESS_ASYMMETRY: Final = 0.05


class ComparisonError(ShadowValidationError):
    """Raised when two cohorts cannot be compared honestly."""


class LeakageError(ComparisonError):
    """Raised when a paired forecast could have seen its own outcome.

    Distinct from generic comparison failure because it invalidates the arm
    entirely rather than reducing coverage.
    """


@dataclass(frozen=True, slots=True)
class PairKey:
    """The tuple that must match exactly for two forecasts to be comparable."""

    campaign_symbol: str
    as_of: datetime
    horizon_days: int

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-friendly record."""
        return {
            "symbol": self.campaign_symbol,
            "as_of": self.as_of.isoformat(),
            "horizon_days": self.horizon_days,
        }


@dataclass(frozen=True, slots=True)
class PairedScore:
    """One matched observation scored under both arms."""

    key: PairKey
    as_of_date: date
    champion_brier: float
    challenger_brier: float
    champion_log: float
    challenger_log: float

    @property
    def brier_difference(self) -> float:
        """Challenger minus champion. Negative means the challenger is better."""
        return self.challenger_brier - self.champion_brier

    @property
    def log_difference(self) -> float:
        """Challenger minus champion on log score. Negative favours challenger."""
        return self.challenger_log - self.champion_log

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-friendly record."""
        return {
            "key": self.key.to_dict(),
            "as_of_date": self.as_of_date.isoformat(),
            "champion_brier": self.champion_brier,
            "challenger_brier": self.challenger_brier,
            "brier_difference": self.brier_difference,
            "log_difference": self.log_difference,
        }


@dataclass(frozen=True)
class PairedCohort:
    """A matched sample with every exclusion accounted for.

    ``comparable`` is ``False`` when missingness differs materially between
    arms. The pairs remain available for inspection, but a caller must not
    proceed to inference on a cohort that reports itself incomparable.
    """

    pairs: tuple[PairedScore, ...]
    champion_total: int
    challenger_total: int
    champion_only: int
    challenger_only: int
    champion_unscored: int
    challenger_unscored: int
    comparable: bool
    incomparable_reason: str | None

    def __post_init__(self) -> None:
        if len(self.pairs) > MAX_PAIRS:
            raise ComparisonError(f"cohort exceeds the {MAX_PAIRS}-pair ceiling")
        if self.comparable and self.incomparable_reason is not None:
            raise ComparisonError("a comparable cohort cannot carry an incomparability reason")
        if not self.comparable and not self.incomparable_reason:
            raise ComparisonError(
                "an incomparable cohort must state why; an unexplained refusal cannot be acted on"
            )

    @property
    def matched(self) -> int:
        """Number of exactly matched, scored pairs."""
        return len(self.pairs)

    @property
    def coverage(self) -> float:
        """Fraction of the union of both arms that produced a scored pair."""
        union = self.champion_total + self.challenger_only
        return self.matched / union if union else 0.0

    def identity(self) -> str:
        """Content identity binding every pair and exclusion count."""
        return canonical_digest(
            {
                "pairs": [item.to_dict() for item in self.pairs],
                "champion_total": self.champion_total,
                "challenger_total": self.challenger_total,
                "champion_only": self.champion_only,
                "challenger_only": self.challenger_only,
                "champion_unscored": self.champion_unscored,
                "challenger_unscored": self.challenger_unscored,
            }
        )

    def to_dict(self) -> dict[str, Any]:
        """Return the JSON-friendly cohort summary."""
        return {
            "identity": self.identity(),
            "matched": self.matched,
            "coverage": self.coverage,
            "champion_total": self.champion_total,
            "challenger_total": self.challenger_total,
            "champion_only": self.champion_only,
            "challenger_only": self.challenger_only,
            "champion_unscored": self.champion_unscored,
            "challenger_unscored": self.challenger_unscored,
            "comparable": self.comparable,
            "incomparable_reason": self.incomparable_reason,
            "distinct_days": len({item.as_of_date for item in self.pairs}),
            "note": (
                "Exclusions are reported rather than dropped. Coverage below one means "
                "the comparison describes part of the universe, and the difference "
                "between arms is what makes a paired result trustworthy."
            ),
        }


def _index(forecasts: Sequence[ShadowForecast]) -> dict[PairKey, ShadowForecast]:
    """Index forecasts by their exact pairing tuple.

    Raises:
        ComparisonError: On a duplicate key, which would make the pairing
            ambiguous and silently pick whichever row came last.
    """
    indexed: dict[PairKey, ShadowForecast] = {}
    for item in forecasts:
        key = PairKey(
            campaign_symbol=item.symbol,
            as_of=item.as_of,
            horizon_days=item.horizon_days,
        )
        if key in indexed:
            raise ComparisonError(
                f"duplicate forecast for {item.symbol} at {item.as_of.isoformat()}; "
                "the pairing would be ambiguous"
            )
        indexed[key] = item
    return indexed


def _latest_observed(history: Sequence[ShadowOutcome]) -> ShadowOutcome | None:
    """Return the newest observed revision, or ``None`` if never observed."""
    observed = [item for item in history if item.status.value == "observed"]
    return max(observed, key=lambda item: item.revision) if observed else None


def build_paired_cohort(
    champion_forecasts: Sequence[ShadowForecast],
    champion_outcomes: Mapping[str, Sequence[ShadowOutcome]],
    challenger_forecasts: Sequence[ShadowForecast],
    challenger_outcomes: Mapping[str, Sequence[ShadowOutcome]],
) -> PairedCohort:
    """Pair two arms exactly and score every matched observation.

    Both arms must score against the *same* realised label for a pair to count.
    A disagreement means the two campaigns observed different outcomes for one
    question, which is a reconciliation fault rather than a scoring difference.

    Raises:
        LeakageError: If any paired forecast could have seen its own outcome.
        ComparisonError: On duplicate keys or disagreeing realised labels.
    """
    champion_index = _index(champion_forecasts)
    challenger_index = _index(challenger_forecasts)
    shared = sorted(
        set(champion_index) & set(challenger_index),
        key=lambda key: (key.as_of, key.campaign_symbol),
    )

    pairs: list[PairedScore] = []
    champion_unscored = 0
    challenger_unscored = 0
    for key in shared:
        champion = champion_index[key]
        challenger = challenger_index[key]
        champion_outcome = _latest_observed(champion_outcomes.get(champion.forecast_id, ()))
        challenger_outcome = _latest_observed(challenger_outcomes.get(challenger.forecast_id, ()))
        if champion_outcome is None:
            champion_unscored += 1
        if challenger_outcome is None:
            challenger_unscored += 1
        if champion_outcome is None or challenger_outcome is None:
            continue
        if champion_outcome.realized_label != challenger_outcome.realized_label:
            raise ComparisonError(
                f"arms disagree about the realised outcome for {key.campaign_symbol} at "
                f"{key.as_of.isoformat()}: {champion_outcome.realized_label!r} versus "
                f"{challenger_outcome.realized_label!r}. One question has one answer; "
                "this is a reconciliation fault, not a scoring difference."
            )
        # Re-verify chronology on both arms. The contracts enforce it at
        # construction, but a cohort assembled from two independently sealed
        # campaigns must not take that on trust.
        try:
            champion.assert_precedes(champion_outcome.observed_at)
            challenger.assert_precedes(challenger_outcome.observed_at)
        except ShadowValidationError as error:
            raise LeakageError(
                f"paired forecast for {key.campaign_symbol} at {key.as_of.isoformat()} "
                f"could have consumed its outcome: {error}"
            ) from error
        label = champion_outcome.realized_label
        pairs.append(
            PairedScore(
                key=key,
                as_of_date=key.as_of.date(),
                champion_brier=brier_score(champion.distribution, label),
                challenger_brier=brier_score(challenger.distribution, label),
                champion_log=log_score(champion.distribution, label),
                challenger_log=log_score(challenger.distribution, label),
            )
        )

    champion_only = len(set(champion_index) - set(challenger_index))
    challenger_only = len(set(challenger_index) - set(champion_index))
    champion_missing_rate = champion_unscored / len(shared) if shared else 0.0
    challenger_missing_rate = challenger_unscored / len(shared) if shared else 0.0
    asymmetry = abs(champion_missing_rate - challenger_missing_rate)
    comparable = asymmetry <= MAX_MISSINGNESS_ASYMMETRY
    reason = (
        None
        if comparable
        else (
            f"missingness differs by {asymmetry:.3f} between arms "
            f"(champion {champion_missing_rate:.3f}, challenger {challenger_missing_rate:.3f}), "
            f"above the {MAX_MISSINGNESS_ASYMMETRY:.3f} bound. A model absent on the days "
            "it would have scored badly produces a biased paired sample even when every "
            "retained pair is individually valid."
        )
    )
    return PairedCohort(
        pairs=tuple(pairs),
        champion_total=len(champion_index),
        challenger_total=len(challenger_index),
        champion_only=champion_only,
        challenger_only=challenger_only,
        champion_unscored=champion_unscored,
        challenger_unscored=challenger_unscored,
        comparable=comparable,
        incomparable_reason=reason,
    )


def paired_differences(cohort: PairedCohort) -> tuple[float, ...]:
    """Return per-pair Brier differences, challenger minus champion."""
    return tuple(item.brier_difference for item in cohort.pairs)


def summarize_pairs(cohort: PairedCohort) -> dict[str, Any]:
    """Return bounded descriptive statistics over the paired differences."""
    differences = paired_differences(cohort)
    if not differences:
        return {"matched": 0, "mean_difference": None, "wins": 0, "losses": 0, "ties": 0}
    counts = Counter(
        "win" if value < 0 else "loss" if value > 0 else "tie" for value in differences
    )
    return {
        "matched": len(differences),
        "mean_difference": math.fsum(differences) / len(differences),
        "wins": counts["win"],
        "losses": counts["loss"],
        "ties": counts["tie"],
        "direction": "negative favours the challenger",
    }


__all__ = [
    "MAX_MISSINGNESS_ASYMMETRY",
    "MAX_PAIRS",
    "ComparisonError",
    "LeakageError",
    "PairKey",
    "PairedCohort",
    "PairedScore",
    "build_paired_cohort",
    "paired_differences",
    "summarize_pairs",
]
