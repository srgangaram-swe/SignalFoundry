"""Cost- and uncertainty-aware abstention before portfolio construction.

The policy is a pure function of one immutable signal and immutable
thresholds. It emits research eligibility only: no quantity, price, venue,
account, order type, or executable instruction crosses this boundary.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Iterable
from dataclasses import dataclass, fields
from datetime import UTC, datetime
from enum import StrEnum
from typing import Any

_IDENTIFIER_MAX_LENGTH = 128
_POLICY_SCHEMA_VERSION = "1.0.0"


class DecisionAction(StrEnum):
    """Permitted decision outcomes."""

    TRADE = "trade"
    ABSTAIN = "abstain"


class SignalDirection(StrEnum):
    """Forecast sign retained for audit, never an executable side."""

    LONG = "long"
    SHORT = "short"
    FLAT = "flat"


class RegimeSupport(StrEnum):
    """Whether the fitted research model supports the detected regime."""

    SUPPORTED = "supported"
    UNSUPPORTED = "unsupported"
    UNKNOWN = "unknown"


class DecisionReason(StrEnum):
    """Stable, ordered, machine-readable abstention reasons."""

    NON_FINITE_INPUT = "non_finite_input"
    OUT_OF_BOUNDS_INPUT = "out_of_bounds_input"
    FUTURE_DATA = "future_data"
    STALE_DATA = "stale_data"
    EXCESS_COST = "excess_cost"
    HIGH_DISAGREEMENT = "high_disagreement"
    UNSUPPORTED_REGIME = "unsupported_regime"
    UNCERTAIN_REGIME = "uncertain_regime"
    DRIFT_DETECTED = "drift_detected"
    EXCESS_UNCERTAINTY = "excess_uncertainty"
    INSUFFICIENT_MARGIN = "insufficient_margin"


@dataclass(frozen=True, slots=True)
class DecisionThresholds:
    """Finite, bounded thresholds for one versioned decision policy.

    All return and cost quantities are decimal returns per proposed unit of
    exposure. Equality passes independent maximum gates, but the penalized
    expected value must *strictly exceed* ``required_margin``.
    """

    schema_version: str
    required_margin: float
    cost_multiplier: float
    cost_uncertainty_multiplier: float
    uncertainty_penalty: float
    maximum_total_cost: float
    maximum_prediction_uncertainty: float
    maximum_model_disagreement: float
    maximum_regime_uncertainty: float
    maximum_drift_score: float
    maximum_data_age_seconds: int
    maximum_absolute_expected_return: float
    maximum_batch_size: int

    def __post_init__(self) -> None:
        if self.schema_version != _POLICY_SCHEMA_VERSION:
            raise ValueError(f"schema_version must be {_POLICY_SCHEMA_VERSION!r}")
        _finite_in_range("required_margin", self.required_margin, 0.0, 1.0)
        _finite_in_range("cost_multiplier", self.cost_multiplier, 1.0, 100.0)
        _finite_in_range(
            "cost_uncertainty_multiplier",
            self.cost_uncertainty_multiplier,
            0.0,
            100.0,
        )
        _finite_in_range("uncertainty_penalty", self.uncertainty_penalty, 0.0, 100.0)
        _finite_in_range("maximum_total_cost", self.maximum_total_cost, 0.0, 1.0)
        _finite_in_range(
            "maximum_prediction_uncertainty",
            self.maximum_prediction_uncertainty,
            0.0,
            1.0,
        )
        _finite_in_range(
            "maximum_model_disagreement",
            self.maximum_model_disagreement,
            0.0,
            1.0,
        )
        _finite_in_range(
            "maximum_regime_uncertainty",
            self.maximum_regime_uncertainty,
            0.0,
            1.0,
        )
        _finite_in_range("maximum_drift_score", self.maximum_drift_score, 0.0, 1.0)
        _finite_in_range(
            "maximum_absolute_expected_return",
            self.maximum_absolute_expected_return,
            0.0,
            1.0,
            lower_open=True,
        )
        _bounded_integer(
            "maximum_data_age_seconds",
            self.maximum_data_age_seconds,
            minimum=1,
            maximum=31_536_000,
        )
        _bounded_integer(
            "maximum_batch_size",
            self.maximum_batch_size,
            minimum=1,
            maximum=100_000,
        )
        if self.required_margin >= self.maximum_absolute_expected_return:
            raise ValueError("required_margin must be below maximum_absolute_expected_return")

    @property
    def policy_id(self) -> str:
        """Return a stable identity over every policy field."""
        payload = {
            field.name: _canonical_value(getattr(self, field.name)) for field in fields(self)
        }
        return f"policy-{_sha256(payload)}"


@dataclass(frozen=True, slots=True)
class DecisionSignal:
    """One untrusted model opportunity evaluated at a known UTC instant.

    Numeric estimates may be non-finite or outside semantic ranges so the
    policy can record a deterministic fail-closed decision. Finite integers
    normalize to floats; integers outside the float range normalize to signed
    infinity and therefore abstain. Structural type, identifier, and timezone
    violations are rejected at construction.
    """

    signal_id: str
    model_id: str
    decision_time: datetime
    data_available_at: datetime
    expected_return: float
    expected_cost: float
    cost_uncertainty: float
    prediction_uncertainty: float
    model_disagreement: float
    regime: RegimeSupport
    regime_uncertainty: float
    drift_score: float

    def __post_init__(self) -> None:
        _identifier("signal_id", self.signal_id)
        _identifier("model_id", self.model_id)
        _aware_datetime("decision_time", self.decision_time)
        _aware_datetime("data_available_at", self.data_available_at)
        for name in (
            "expected_return",
            "expected_cost",
            "cost_uncertainty",
            "prediction_uncertainty",
            "model_disagreement",
            "regime_uncertainty",
            "drift_score",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int | float):
                raise TypeError(f"{name} must be numeric")
            try:
                normalized = float(value)
            except OverflowError:
                normalized = -math.inf if value < 0 else math.inf
            object.__setattr__(self, name, normalized)
        if not isinstance(self.regime, RegimeSupport):
            raise TypeError("regime must be a RegimeSupport")


@dataclass(frozen=True, slots=True)
class Decision:
    """Immutable eligibility evidence; deliberately not an order."""

    decision_id: str
    policy_id: str
    signal_id: str
    action: DecisionAction
    direction: SignalDirection
    reasons: tuple[DecisionReason, ...]
    failed_fields: tuple[str, ...]
    conservative_cost: float | None
    uncertainty_charge: float | None
    penalized_expected_value: float | None
    required_margin: float
    data_age_seconds: float

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-safe, stable-key representation."""
        return {
            "action": self.action.value,
            "conservative_cost": self.conservative_cost,
            "data_age_seconds": self.data_age_seconds,
            "decision_id": self.decision_id,
            "direction": self.direction.value,
            "failed_fields": list(self.failed_fields),
            "penalized_expected_value": self.penalized_expected_value,
            "policy_id": self.policy_id,
            "reasons": [reason.value for reason in self.reasons],
            "required_margin": self.required_margin,
            "signal_id": self.signal_id,
            "uncertainty_charge": self.uncertainty_charge,
        }

    def to_json(self) -> str:
        """Serialize deterministically without NaN or Infinity extensions."""
        return json.dumps(
            self.to_dict(),
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        )


@dataclass(frozen=True, slots=True)
class DecisionPolicy:
    """Evaluate bounded batches without portfolio or execution side effects."""

    thresholds: DecisionThresholds

    def __post_init__(self) -> None:
        if not isinstance(self.thresholds, DecisionThresholds):
            raise TypeError("thresholds must be DecisionThresholds")

    @property
    def policy_id(self) -> str:
        return self.thresholds.policy_id

    def evaluate(self, signal: DecisionSignal) -> Decision:
        """Evaluate one signal in constant time and allocate no external state."""
        if not isinstance(signal, DecisionSignal):
            raise TypeError("signal must be DecisionSignal")
        threshold = self.thresholds
        numeric = {
            name: float(getattr(signal, name))
            for name in (
                "expected_return",
                "expected_cost",
                "cost_uncertainty",
                "prediction_uncertainty",
                "model_disagreement",
                "regime_uncertainty",
                "drift_score",
            )
        }
        failed_fields: list[str] = []
        reasons: list[DecisionReason] = []

        non_finite = [name for name, value in numeric.items() if not math.isfinite(value)]
        if non_finite:
            reasons.append(DecisionReason.NON_FINITE_INPUT)
            failed_fields.extend(non_finite)

        out_of_bounds: list[str] = []
        if math.isfinite(numeric["expected_return"]) and (
            abs(numeric["expected_return"]) > threshold.maximum_absolute_expected_return
        ):
            out_of_bounds.append("expected_return")
        for name in (
            "expected_cost",
            "cost_uncertainty",
            "prediction_uncertainty",
            "model_disagreement",
            "regime_uncertainty",
            "drift_score",
        ):
            if math.isfinite(numeric[name]) and not 0.0 <= numeric[name] <= 1.0:
                out_of_bounds.append(name)
        if out_of_bounds:
            reasons.append(DecisionReason.OUT_OF_BOUNDS_INPUT)
            failed_fields.extend(out_of_bounds)
        invalid_fields = set(non_finite) | set(out_of_bounds)

        decision_time = signal.decision_time.astimezone(UTC)
        available_at = signal.data_available_at.astimezone(UTC)
        data_age_seconds = (decision_time - available_at).total_seconds()
        if data_age_seconds < 0.0:
            reasons.append(DecisionReason.FUTURE_DATA)
            failed_fields.append("data_available_at")
        elif data_age_seconds > threshold.maximum_data_age_seconds:
            reasons.append(DecisionReason.STALE_DATA)
            failed_fields.append("data_available_at")

        conservative_cost: float | None = None
        uncertainty_charge: float | None = None
        penalized_value: float | None = None
        if not invalid_fields.intersection({"expected_cost", "cost_uncertainty"}):
            conservative_cost = threshold.cost_multiplier * math.fsum(
                (
                    numeric["expected_cost"],
                    threshold.cost_uncertainty_multiplier * numeric["cost_uncertainty"],
                )
            )
        if "prediction_uncertainty" not in invalid_fields:
            uncertainty_charge = threshold.uncertainty_penalty * numeric["prediction_uncertainty"]
        if (
            "expected_return" not in invalid_fields
            and conservative_cost is not None
            and uncertainty_charge is not None
        ):
            penalized_value = math.fsum(
                (
                    abs(numeric["expected_return"]),
                    -conservative_cost,
                    -uncertainty_charge,
                )
            )

        if conservative_cost is not None and conservative_cost > threshold.maximum_total_cost:
            reasons.append(DecisionReason.EXCESS_COST)
            failed_fields.append("conservative_cost")
        if (
            "model_disagreement" not in invalid_fields
            and numeric["model_disagreement"] > threshold.maximum_model_disagreement
        ):
            reasons.append(DecisionReason.HIGH_DISAGREEMENT)
            failed_fields.append("model_disagreement")
        if signal.regime is not RegimeSupport.SUPPORTED:
            reasons.append(DecisionReason.UNSUPPORTED_REGIME)
            failed_fields.append("regime")
        if (
            "regime_uncertainty" not in invalid_fields
            and numeric["regime_uncertainty"] > threshold.maximum_regime_uncertainty
        ):
            reasons.append(DecisionReason.UNCERTAIN_REGIME)
            failed_fields.append("regime_uncertainty")
        if (
            "drift_score" not in invalid_fields
            and numeric["drift_score"] > threshold.maximum_drift_score
        ):
            reasons.append(DecisionReason.DRIFT_DETECTED)
            failed_fields.append("drift_score")
        if (
            "prediction_uncertainty" not in invalid_fields
            and numeric["prediction_uncertainty"] > threshold.maximum_prediction_uncertainty
        ):
            reasons.append(DecisionReason.EXCESS_UNCERTAINTY)
            failed_fields.append("prediction_uncertainty")
        if penalized_value is not None and penalized_value <= threshold.required_margin:
            reasons.append(DecisionReason.INSUFFICIENT_MARGIN)
            failed_fields.append("penalized_expected_value")

        direction = _direction(numeric["expected_return"])
        action = DecisionAction.ABSTAIN if reasons else DecisionAction.TRADE
        payload = _signal_identity_payload(signal)
        payload["policy_id"] = self.policy_id
        return Decision(
            decision_id=f"decision-{_sha256(payload)}",
            policy_id=self.policy_id,
            signal_id=signal.signal_id,
            action=action,
            direction=direction,
            reasons=tuple(reasons),
            failed_fields=tuple(dict.fromkeys(failed_fields)),
            conservative_cost=conservative_cost,
            uncertainty_charge=uncertainty_charge,
            penalized_expected_value=penalized_value,
            required_margin=threshold.required_margin,
            data_age_seconds=data_age_seconds,
        )

    def evaluate_many(self, signals: Iterable[DecisionSignal]) -> tuple[Decision, ...]:
        """Evaluate at most ``maximum_batch_size`` unique signals.

        Results are sorted by stable decision identity, making replay output
        independent of caller iteration order. The iterable is consumed only
        through the first disallowed item.
        """
        bounded: list[DecisionSignal] = []
        for signal in signals:
            if len(bounded) >= self.thresholds.maximum_batch_size:
                raise ValueError("signal batch exceeds maximum_batch_size")
            if not isinstance(signal, DecisionSignal):
                raise TypeError("every batch item must be DecisionSignal")
            bounded.append(signal)
        identifiers = [signal.signal_id for signal in bounded]
        if len(identifiers) != len(set(identifiers)):
            raise ValueError("signal batch contains duplicate signal_id values")
        return tuple(
            sorted((self.evaluate(signal) for signal in bounded), key=lambda item: item.decision_id)
        )


def _direction(expected_return: float) -> SignalDirection:
    if not math.isfinite(expected_return) or expected_return == 0.0:
        return SignalDirection.FLAT
    return SignalDirection.LONG if expected_return > 0.0 else SignalDirection.SHORT


def _signal_identity_payload(signal: DecisionSignal) -> dict[str, Any]:
    return {
        "cost_uncertainty": _canonical_value(signal.cost_uncertainty),
        "data_available_at": _canonical_value(signal.data_available_at),
        "decision_time": _canonical_value(signal.decision_time),
        "drift_score": _canonical_value(signal.drift_score),
        "expected_cost": _canonical_value(signal.expected_cost),
        "expected_return": _canonical_value(signal.expected_return),
        "model_disagreement": _canonical_value(signal.model_disagreement),
        "model_id": signal.model_id,
        "prediction_uncertainty": _canonical_value(signal.prediction_uncertainty),
        "regime": signal.regime.value,
        "regime_uncertainty": _canonical_value(signal.regime_uncertainty),
        "signal_id": signal.signal_id,
    }


def _canonical_value(value: Any) -> Any:
    if isinstance(value, datetime):
        return value.astimezone(UTC).isoformat().replace("+00:00", "Z")
    if isinstance(value, float):
        if math.isnan(value):
            return "float:nan"
        if value == math.inf:
            return "float:+inf"
        if value == -math.inf:
            return "float:-inf"
        return f"float:{value.hex()}"
    return value


def _sha256(payload: dict[str, Any]) -> str:
    encoded = json.dumps(
        payload,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("ascii")
    return hashlib.sha256(encoded).hexdigest()


def _identifier(name: str, value: str) -> None:
    if (
        not isinstance(value, str)
        or not value
        or len(value) > _IDENTIFIER_MAX_LENGTH
        or value != value.strip()
        or not value.isascii()
        or not all(character.isalnum() or character in "._:-" for character in value)
    ):
        raise ValueError(
            f"{name} must be a non-empty safe ASCII identifier of at most "
            f"{_IDENTIFIER_MAX_LENGTH} characters"
        )


def _aware_datetime(name: str, value: datetime) -> None:
    if not isinstance(value, datetime) or value.tzinfo is None or value.utcoffset() is None:
        raise ValueError(f"{name} must be a timezone-aware datetime")


def _finite_in_range(
    name: str,
    value: float,
    lower: float,
    upper: float,
    *,
    lower_open: bool = False,
) -> None:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise TypeError(f"{name} must be numeric")
    number = float(value)
    lower_ok = number > lower if lower_open else number >= lower
    if not math.isfinite(number) or not lower_ok or number > upper:
        opening = "(" if lower_open else "["
        raise ValueError(f"{name} must be finite and in {opening}{lower}, {upper}]")


def _bounded_integer(name: str, value: int, *, minimum: int, maximum: int) -> None:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer")
    if not minimum <= value <= maximum:
        raise ValueError(f"{name} must be in [{minimum}, {maximum}]")
