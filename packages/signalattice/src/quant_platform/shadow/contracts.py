"""Typed records for delayed shadow forecast evidence.

SF-S5-SL-MR4. A shadow campaign records what a model predicted *before* the
outcome existed, then records the outcome separately and never rewrites either.
Everything in this module exists to make the "before" provable rather than
asserted.

**Forecast identity is content-derived, not assigned.** A forecast's identity
covers the campaign, the as-of instant, the horizon, the symbol, and the
predicted distribution. Recomputing it from the record reproduces it, so a
forecast cannot be edited and still match the identity that was sealed.

**Every timestamp is timezone-aware UTC and every ordering is checked.** A naive
datetime is refused rather than assumed. `as_of` must strictly precede the
outcome instant it will eventually be scored against; a forecast whose as-of is
not strictly earlier is not a forecast.

**Probabilities are validated as probabilities.** Finite, within ``[0, 1]``, and
summing to one within a declared tolerance. NaN and infinity are refused before
they can reach a proper score, where they would silently poison an aggregate.

**Booleans are not numbers.** ``isinstance(True, int)`` is true in Python, so
every numeric field rejects ``bool`` explicitly. A ``True`` that slips into a
probability vector would validate as ``1.0`` and be scored as certainty.

Errors extend the existing registry taxonomy so the retryable/non-retryable
split callers already depend on continues to hold: capacity and busy conditions
stay retryable, integrity and schema conditions stay terminal.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import StrEnum
from typing import Any, Final

from quant_platform.tracking.contracts import (
    IntegrityError,
    RegistryError,
    ValidationError,
)

#: Refusal thresholds, not tuning knobs.
MAX_SYMBOLS_PER_BATCH: Final = 5_000
MAX_CAMPAIGN_NAME_CHARS: Final = 64
MAX_SYMBOL_CHARS: Final = 24
MAX_HORIZON_DAYS: Final = 365
MAX_CLASSES: Final = 16

#: Probability vectors must sum to one within this tolerance. Tight enough that
#: a genuinely malformed vector is refused, loose enough to admit the rounding
#: a float64 renormalisation leaves behind.
PROBABILITY_SUM_TOLERANCE: Final = 1e-9

#: A campaign below this many scored forecasts cannot support a calibration
#: claim. Reports return INSUFFICIENT_EVIDENCE rather than a number.
MIN_SCORED_FORECASTS: Final = 30

_SYMBOL_PATTERN: Final = re.compile(r"^[A-Z][A-Z0-9.\-]{0,23}$")
_CAMPAIGN_PATTERN: Final = re.compile(r"^[a-z][a-z0-9_\-]{0,63}$")


class ShadowError(RegistryError):
    """Base for shadow-campaign failures."""


class ShadowValidationError(ShadowError, ValidationError):
    """A record is malformed. Terminal: retrying identical input cannot help."""


class ChronologyError(ShadowValidationError):
    """A record violates the forecast-before-outcome ordering.

    Separated from generic validation because it is the invariant the whole
    campaign exists to protect, and a caller may reasonably want to catch it
    specifically.
    """


class SealedCampaignError(ShadowError, IntegrityError):
    """A sealed batch was asked to change. Terminal and non-repairable."""


class CampaignState(StrEnum):
    """Lifecycle of one shadow campaign.

    Forward-only. There is no transition back to ``ACTIVE`` from any later
    state: reopening a campaign would let a forecast be added after its outcome
    window opened, which is the one thing this module prevents.
    """

    DRAFT = "draft"
    ACTIVE = "active"
    SEALED = "sealed"
    RECONCILING = "reconciling"
    CLOSED = "closed"
    ABANDONED = "abandoned"


#: The only permitted transitions. Absence is refusal, not omission.
ALLOWED_CAMPAIGN_TRANSITIONS: Final[dict[CampaignState, frozenset[CampaignState]]] = {
    CampaignState.DRAFT: frozenset({CampaignState.ACTIVE, CampaignState.ABANDONED}),
    CampaignState.ACTIVE: frozenset({CampaignState.SEALED, CampaignState.ABANDONED}),
    CampaignState.SEALED: frozenset({CampaignState.RECONCILING, CampaignState.ABANDONED}),
    CampaignState.RECONCILING: frozenset({CampaignState.CLOSED, CampaignState.ABANDONED}),
    CampaignState.CLOSED: frozenset(),
    CampaignState.ABANDONED: frozenset(),
}


class OutcomeStatus(StrEnum):
    """Why an outcome is or is not available for a forecast."""

    OBSERVED = "observed"
    PENDING = "pending"
    MISSING = "missing"
    LATE = "late"
    WRONG_HORIZON = "wrong_horizon"


class ReportBasis(StrEnum):
    """Which revision of the outcome record a report was computed from.

    The two bases answer different questions and must never be conflated: the
    first-eligible basis is what a live consumer could have known at the first
    moment the outcome qualified, while the latest-known basis includes every
    later revision. Reporting a revised number as if it were available live is
    the most common way shadow evidence flatters itself.
    """

    FIRST_ELIGIBLE = "first_eligible"
    LATEST_KNOWN = "latest_known"


def utc_instant(value: object, *, field_name: str) -> datetime:
    """Return a timezone-aware UTC datetime, refusing anything ambiguous.

    Raises:
        ShadowValidationError: If the value is not an aware datetime.
    """
    if not isinstance(value, datetime):
        raise ShadowValidationError(f"{field_name} must be a datetime, got {type(value).__name__}")
    if value.tzinfo is None or value.tzinfo.utcoffset(value) is None:
        raise ShadowValidationError(
            f"{field_name} must be timezone-aware; a naive instant cannot be ordered "
            "against an outcome, and the ordering is the evidence"
        )
    return value.astimezone(UTC)


def _finite(value: object, *, field_name: str) -> float:
    """Return a finite float, refusing bools and non-finite values."""
    if isinstance(value, bool):
        raise ShadowValidationError(
            f"{field_name} must be a real number, not a bool; True would validate as 1.0 "
            "and be scored as certainty"
        )
    if not isinstance(value, (int, float)):
        raise ShadowValidationError(f"{field_name} must be a real number")
    numeric = float(value)
    if not math.isfinite(numeric):
        raise ShadowValidationError(
            f"{field_name} must be finite; a NaN reaching a proper score poisons the "
            "aggregate without failing"
        )
    return numeric


def validate_symbol(value: object) -> str:
    """Return a validated symbol.

    Raises:
        ShadowValidationError: On a malformed or oversized symbol.
    """
    if not isinstance(value, str) or not _SYMBOL_PATTERN.match(value):
        raise ShadowValidationError(
            f"symbol {value!r} is malformed; expected uppercase alphanumeric with . or -, "
            f"at most {MAX_SYMBOL_CHARS} characters"
        )
    return value


def validate_campaign_name(value: object) -> str:
    """Return a validated campaign name.

    Raises:
        ShadowValidationError: On a malformed or oversized name.
    """
    if not isinstance(value, str) or not _CAMPAIGN_PATTERN.match(value):
        raise ShadowValidationError(
            f"campaign name {value!r} is malformed; expected lowercase alphanumeric with "
            f"_ or -, at most {MAX_CAMPAIGN_NAME_CHARS} characters"
        )
    return value


def canonical_digest(payload: Any) -> str:
    """Return a deterministic SHA-256 over a canonical JSON payload.

    ``sort_keys`` and ``allow_nan=False`` together mean the digest depends on
    content rather than on dictionary insertion order, and that a non-finite
    value fails here rather than producing a digest over ``NaN``.
    """
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


@dataclass(frozen=True, slots=True)
class ProbabilityVector:
    """A discrete predictive distribution over ordered class labels.

    Stored as labels plus probabilities rather than a mapping so the class order
    is part of the record: a proper score computed against a different ordering
    is a different number, and the ordering must therefore be sealed with the
    forecast rather than recovered from a dict at scoring time.
    """

    labels: tuple[str, ...]
    probabilities: tuple[float, ...]

    def __post_init__(self) -> None:
        labels = tuple(self.labels)
        if not 2 <= len(labels) <= MAX_CLASSES:
            raise ShadowValidationError(
                f"a distribution needs between 2 and {MAX_CLASSES} classes, got {len(labels)}"
            )
        if any(not isinstance(item, str) or not item.strip() for item in labels):
            raise ShadowValidationError("class labels must be non-empty strings")
        if len(set(labels)) != len(labels):
            raise ShadowValidationError("class labels must be unique")
        probabilities = tuple(
            _finite(item, field_name=f"probability[{index}]")
            for index, item in enumerate(self.probabilities)
        )
        if len(probabilities) != len(labels):
            raise ShadowValidationError(
                f"{len(labels)} labels but {len(probabilities)} probabilities"
            )
        if any(item < 0.0 or item > 1.0 for item in probabilities):
            raise ShadowValidationError("probabilities must lie in [0, 1]")
        total = math.fsum(probabilities)
        if abs(total - 1.0) > PROBABILITY_SUM_TOLERANCE:
            raise ShadowValidationError(
                f"probabilities sum to {total!r}, outside 1 +/- {PROBABILITY_SUM_TOLERANCE}"
            )
        object.__setattr__(self, "labels", labels)
        object.__setattr__(self, "probabilities", probabilities)

    def probability_of(self, label: str) -> float:
        """Return the probability assigned to ``label``.

        Raises:
            ShadowValidationError: If the label is not in this distribution.
        """
        try:
            return self.probabilities[self.labels.index(label)]
        except ValueError:
            raise ShadowValidationError(
                f"label {label!r} is not in this distribution: {list(self.labels)}"
            ) from None

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-friendly record preserving class order."""
        return {"labels": list(self.labels), "probabilities": list(self.probabilities)}


@dataclass(frozen=True, slots=True)
class ShadowForecast:
    """One sealed prediction, made strictly before its outcome could exist.

    ``as_of`` is the instant the feature prefix ended; ``target_instant`` is when
    the outcome becomes knowable. The gap between them is the horizon, and it is
    recorded rather than inferred so a later horizon change cannot silently
    rescore old forecasts.
    """

    campaign: str
    symbol: str
    as_of: datetime
    target_instant: datetime
    horizon_days: int
    distribution: ProbabilityVector
    feature_prefix_digest: str
    model_identity: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "campaign", validate_campaign_name(self.campaign))
        object.__setattr__(self, "symbol", validate_symbol(self.symbol))
        object.__setattr__(self, "as_of", utc_instant(self.as_of, field_name="as_of"))
        object.__setattr__(
            self, "target_instant", utc_instant(self.target_instant, field_name="target_instant")
        )
        if self.target_instant <= self.as_of:
            raise ChronologyError(
                f"target_instant {self.target_instant.isoformat()} does not follow as_of "
                f"{self.as_of.isoformat()}; a forecast whose target is not strictly later "
                "is a retrospective statement, not a prediction"
            )
        if isinstance(self.horizon_days, bool) or not isinstance(self.horizon_days, int):
            raise ShadowValidationError("horizon_days must be an int")
        if not 1 <= self.horizon_days <= MAX_HORIZON_DAYS:
            raise ShadowValidationError(f"horizon_days must lie in [1, {MAX_HORIZON_DAYS}]")
        if not isinstance(self.distribution, ProbabilityVector):
            raise ShadowValidationError("distribution must be a ProbabilityVector")
        for field_name in ("feature_prefix_digest", "model_identity"):
            value = getattr(self, field_name)
            if not isinstance(value, str) or len(value) != 64:
                raise ShadowValidationError(f"{field_name} must be a full SHA-256 digest")
            if any(character not in "0123456789abcdef" for character in value):
                raise ShadowValidationError(f"{field_name} must be lowercase hexadecimal")

    @property
    def forecast_id(self) -> str:
        """Content identity over everything that defines this prediction.

        Excludes nothing that affects the score: campaign, symbol, both
        instants, horizon, the full distribution with its class order, the
        feature prefix, and the model. Editing any of them produces a different
        identity, so a sealed forecast cannot be revised in place.
        """
        return canonical_digest(
            {
                "campaign": self.campaign,
                "symbol": self.symbol,
                "as_of": self.as_of.isoformat(),
                "target_instant": self.target_instant.isoformat(),
                "horizon_days": self.horizon_days,
                "distribution": self.distribution.to_dict(),
                "feature_prefix_digest": self.feature_prefix_digest,
                "model_identity": self.model_identity,
            }
        )

    def assert_precedes(self, observed_at: datetime) -> None:
        """Refuse an outcome observed at or before the forecast's as-of.

        Raises:
            ChronologyError: If the outcome could have been known when the
                forecast was made.
        """
        instant = utc_instant(observed_at, field_name="observed_at")
        if instant <= self.as_of:
            raise ChronologyError(
                f"outcome observed at {instant.isoformat()} is not after as_of "
                f"{self.as_of.isoformat()}; the forecast could have consumed it"
            )

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-friendly record including the derived identity."""
        payload = {
            "forecast_id": self.forecast_id,
            "campaign": self.campaign,
            "symbol": self.symbol,
            "as_of": self.as_of.isoformat(),
            "target_instant": self.target_instant.isoformat(),
            "horizon_days": self.horizon_days,
            "distribution": self.distribution.to_dict(),
            "feature_prefix_digest": self.feature_prefix_digest,
            "model_identity": self.model_identity,
        }
        return payload


@dataclass(frozen=True, slots=True)
class ShadowOutcome:
    """An observed outcome for one forecast, appended and never mutated.

    ``revision`` starts at zero and increases. A later revision does not replace
    an earlier one; both are retained so first-eligible and latest-known reports
    can differ and the delta between them stays visible.
    """

    forecast_id: str
    realized_label: str
    observed_at: datetime
    recorded_at: datetime
    revision: int
    status: OutcomeStatus = OutcomeStatus.OBSERVED

    def __post_init__(self) -> None:
        if not isinstance(self.forecast_id, str) or len(self.forecast_id) != 64:
            raise ShadowValidationError("forecast_id must be a full SHA-256 digest")
        if not isinstance(self.realized_label, str) or not self.realized_label.strip():
            raise ShadowValidationError("realized_label must be a non-empty string")
        object.__setattr__(
            self, "observed_at", utc_instant(self.observed_at, field_name="observed_at")
        )
        object.__setattr__(
            self, "recorded_at", utc_instant(self.recorded_at, field_name="recorded_at")
        )
        if self.recorded_at < self.observed_at:
            raise ChronologyError(
                f"recorded_at {self.recorded_at.isoformat()} precedes observed_at "
                f"{self.observed_at.isoformat()}; a clock reversal makes the publication "
                "lag meaningless"
            )
        if isinstance(self.revision, bool) or not isinstance(self.revision, int):
            raise ShadowValidationError("revision must be an int")
        if self.revision < 0:
            raise ShadowValidationError("revision must be non-negative")
        if not isinstance(self.status, OutcomeStatus):
            raise ShadowValidationError("status must be an OutcomeStatus member")

    @property
    def publication_lag_seconds(self) -> float:
        """Seconds between the outcome occurring and being recorded."""
        return (self.recorded_at - self.observed_at).total_seconds()

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-friendly record."""
        return {
            "forecast_id": self.forecast_id,
            "realized_label": self.realized_label,
            "observed_at": self.observed_at.isoformat(),
            "recorded_at": self.recorded_at.isoformat(),
            "revision": self.revision,
            "status": self.status.value,
            "publication_lag_seconds": self.publication_lag_seconds,
        }


@dataclass(frozen=True)
class SealedBatch:
    """A complete, immutable set of forecasts sealed as one unit.

    Sealing is all-or-nothing by construction: the batch validates its whole
    universe on creation, so a partially populated batch cannot exist to be
    committed.
    """

    campaign: str
    as_of: datetime
    expected_universe: tuple[str, ...]
    forecasts: tuple[ShadowForecast, ...]
    sealed_at: datetime

    def __post_init__(self) -> None:
        object.__setattr__(self, "campaign", validate_campaign_name(self.campaign))
        object.__setattr__(self, "as_of", utc_instant(self.as_of, field_name="as_of"))
        object.__setattr__(self, "sealed_at", utc_instant(self.sealed_at, field_name="sealed_at"))
        universe = tuple(validate_symbol(item) for item in self.expected_universe)
        if not universe:
            raise ShadowValidationError("expected_universe must not be empty")
        if len(universe) > MAX_SYMBOLS_PER_BATCH:
            raise ShadowValidationError(
                f"expected_universe exceeds the {MAX_SYMBOLS_PER_BATCH}-symbol ceiling"
            )
        if len(set(universe)) != len(universe):
            raise ShadowValidationError("expected_universe contains a duplicate symbol")
        forecasts = tuple(self.forecasts)
        if any(item.campaign != self.campaign for item in forecasts):
            raise ShadowValidationError("a forecast belongs to a different campaign")
        if any(item.as_of != self.as_of for item in forecasts):
            raise ChronologyError(
                "a forecast carries a different as_of than its batch; one batch is one "
                "decision instant"
            )
        covered = [item.symbol for item in forecasts]
        if len(set(covered)) != len(covered):
            raise ShadowValidationError("two forecasts cover the same symbol")
        missing = sorted(set(universe) - set(covered))
        if missing:
            raise ShadowValidationError(
                f"batch is incomplete: {len(missing)} symbol(s) in the expected universe have "
                f"no forecast, e.g. {missing[:5]}. Sealing a partial universe would let a "
                "campaign silently drop the names a model could not score."
            )
        extra = sorted(set(covered) - set(universe))
        if extra:
            raise ShadowValidationError(
                f"batch covers {extra[:5]}, absent from the expected universe"
            )
        if self.sealed_at < self.as_of:
            raise ChronologyError(
                f"sealed_at {self.sealed_at.isoformat()} precedes as_of "
                f"{self.as_of.isoformat()}; a batch cannot be sealed before the decision "
                "instant it represents"
            )
        object.__setattr__(self, "expected_universe", tuple(sorted(universe)))
        object.__setattr__(
            self, "forecasts", tuple(sorted(forecasts, key=lambda item: item.symbol))
        )

    @property
    def batch_id(self) -> str:
        """Content identity over the whole sealed set."""
        return canonical_digest(
            {
                "campaign": self.campaign,
                "as_of": self.as_of.isoformat(),
                "expected_universe": list(self.expected_universe),
                "forecast_ids": [item.forecast_id for item in self.forecasts],
            }
        )

    def forecast_ids(self) -> tuple[str, ...]:
        """Every forecast identity in deterministic order."""
        return tuple(item.forecast_id for item in self.forecasts)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-friendly record."""
        return {
            "batch_id": self.batch_id,
            "campaign": self.campaign,
            "as_of": self.as_of.isoformat(),
            "sealed_at": self.sealed_at.isoformat(),
            "expected_universe": list(self.expected_universe),
            "forecast_count": len(self.forecasts),
            "forecasts": [item.to_dict() for item in self.forecasts],
        }


def assert_campaign_transition(current: CampaignState, proposed: CampaignState) -> None:
    """Refuse a campaign-state transition the lifecycle does not permit.

    Raises:
        ShadowValidationError: If the transition is not permitted, naming both
            states and why terminal states accept nothing.
    """
    if not isinstance(current, CampaignState) or not isinstance(proposed, CampaignState):
        raise ShadowValidationError("campaign states must be CampaignState members")
    permitted = ALLOWED_CAMPAIGN_TRANSITIONS[current]
    if proposed not in permitted:
        if not permitted:
            raise ShadowValidationError(
                f"campaign is terminal in {current.value!r}; {proposed.value!r} cannot follow. "
                "Reopening would allow a forecast after its outcome window opened."
            )
        raise ShadowValidationError(
            f"{current.value!r} -> {proposed.value!r} is not a permitted campaign transition"
        )


def assert_universe_complete(
    expected: Sequence[str], covered: Mapping[str, Any], *, context: str
) -> None:
    """Refuse an incomplete universe.

    Raises:
        ShadowValidationError: Naming the missing symbols.
    """
    missing = sorted(set(expected) - set(covered))
    if missing:
        raise ShadowValidationError(
            f"{context}: {len(missing)} expected symbol(s) absent, e.g. {missing[:5]}"
        )


__all__ = [
    "ALLOWED_CAMPAIGN_TRANSITIONS",
    "MAX_CLASSES",
    "MAX_HORIZON_DAYS",
    "MAX_SYMBOLS_PER_BATCH",
    "MIN_SCORED_FORECASTS",
    "PROBABILITY_SUM_TOLERANCE",
    "CampaignState",
    "ChronologyError",
    "OutcomeStatus",
    "ProbabilityVector",
    "ReportBasis",
    "SealedBatch",
    "SealedCampaignError",
    "ShadowError",
    "ShadowForecast",
    "ShadowOutcome",
    "ShadowValidationError",
    "assert_campaign_transition",
    "assert_universe_complete",
    "canonical_digest",
    "utc_instant",
    "validate_campaign_name",
    "validate_symbol",
]
