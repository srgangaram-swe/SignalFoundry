"""Fail-closed controls for offline paper-decision replay.

The module emits decisions only; it contains no broker, credential, network,
or order-routing interface. A kill switch is one-way for the lifetime of the
state object so an automated path cannot silently re-enable evaluation.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from types import MappingProxyType


@dataclass(frozen=True)
class PaperRiskLimits:
    """Pre-registered bounds for zero-capital paper decisions."""

    maximum_data_age: timedelta = timedelta(days=4)
    maximum_gross_exposure: float = 1.0
    maximum_net_exposure: float = 1.0
    maximum_position_weight: float = 0.10
    maximum_turnover: float = 0.50
    maximum_notional: float = 1_000_000.0
    maximum_drawdown: float = 0.15
    maximum_daily_loss: float = 0.05

    def __post_init__(self) -> None:
        if self.maximum_data_age <= timedelta(0):
            raise ValueError("maximum_data_age must be positive")
        positive = (
            self.maximum_gross_exposure,
            self.maximum_net_exposure,
            self.maximum_position_weight,
            self.maximum_turnover,
            self.maximum_notional,
            self.maximum_drawdown,
            self.maximum_daily_loss,
        )
        if not all(math.isfinite(value) and value > 0.0 for value in positive):
            raise ValueError("paper risk limits must be finite and positive")


@dataclass(frozen=True)
class PaperControlDecision:
    """Auditable allow/halt result without an executable order."""

    decision_id: str
    allowed: bool
    reasons: tuple[str, ...]
    observed: Mapping[str, float]


@dataclass
class PaperControlState:
    """Stateful idempotency, loss, stale-data, and kill-switch boundary."""

    limits: PaperRiskLimits = field(default_factory=PaperRiskLimits)
    _seen_decisions: set[str] = field(default_factory=set, init=False)
    _kill_switch_active: bool = field(default=False, init=False)
    _peak_equity: float | None = field(default=None, init=False)

    def activate_kill_switch(self) -> None:
        """Permanently halt this paper-control state."""
        self._kill_switch_active = True

    def evaluate(
        self,
        *,
        decision_id: str,
        decision_time: datetime,
        data_available_at: datetime,
        target_weights: Mapping[str, float],
        current_weights: Mapping[str, float],
        equity: float,
        previous_equity: float,
    ) -> PaperControlDecision:
        """Evaluate one proposed paper state transition and fail closed."""
        if not isinstance(decision_id, str) or not decision_id.strip():
            raise ValueError("decision_id must be a non-empty string")
        if decision_id in self._seen_decisions:
            return PaperControlDecision(
                decision_id=decision_id,
                allowed=False,
                reasons=("duplicate_decision_id",),
                observed=MappingProxyType({}),
            )
        self._seen_decisions.add(decision_id)

        decision_utc = _utc(decision_time, "decision_time")
        available_utc = _utc(data_available_at, "data_available_at")
        weights = _finite_weights(target_weights, "target_weights")
        current = _finite_weights(current_weights, "current_weights")
        equity_value = _positive(equity, "equity")
        previous_value = _positive(previous_equity, "previous_equity")
        self._peak_equity = max(self._peak_equity or equity_value, equity_value)

        gross = math.fsum(abs(weight) for weight in weights.values())
        net = abs(math.fsum(weights.values()))
        max_position = max((abs(weight) for weight in weights.values()), default=0.0)
        symbols = set(weights) | set(current)
        turnover = math.fsum(
            abs(weights.get(symbol, 0.0) - current.get(symbol, 0.0)) for symbol in symbols
        )
        notional = gross * equity_value
        drawdown = equity_value / self._peak_equity - 1.0
        daily_return = equity_value / previous_value - 1.0
        data_age_seconds = (decision_utc - available_utc).total_seconds()
        observed = {
            "gross_exposure": gross,
            "absolute_net_exposure": net,
            "maximum_position_weight": max_position,
            "turnover": turnover,
            "notional": notional,
            "drawdown": drawdown,
            "daily_return": daily_return,
            "data_age_seconds": data_age_seconds,
        }

        reasons: list[str] = []
        if self._kill_switch_active:
            reasons.append("manual_kill_switch")
        if data_age_seconds < 0:
            reasons.append("future_data_availability")
        elif data_age_seconds > self.limits.maximum_data_age.total_seconds():
            reasons.append("stale_data")
        if gross > self.limits.maximum_gross_exposure:
            reasons.append("gross_exposure_limit")
        if net > self.limits.maximum_net_exposure:
            reasons.append("net_exposure_limit")
        if max_position > self.limits.maximum_position_weight:
            reasons.append("position_limit")
        if turnover > self.limits.maximum_turnover:
            reasons.append("turnover_limit")
        if notional > self.limits.maximum_notional:
            reasons.append("notional_limit")
        if drawdown < -self.limits.maximum_drawdown:
            reasons.append("drawdown_limit")
        if daily_return < -self.limits.maximum_daily_loss:
            reasons.append("daily_loss_limit")
        return PaperControlDecision(
            decision_id=decision_id,
            allowed=not reasons,
            reasons=tuple(reasons),
            observed=MappingProxyType(observed),
        )


def _utc(value: datetime, name: str) -> datetime:
    if not isinstance(value, datetime):
        raise ValueError(f"{name} must be a datetime")
    if value.tzinfo is None:
        raise ValueError(f"{name} must be timezone-aware")
    return value.astimezone(UTC)


def _positive(value: float, name: str) -> float:
    result = float(value)
    if not math.isfinite(result) or result <= 0.0:
        raise ValueError(f"{name} must be finite and positive")
    return result


def _finite_weights(value: Mapping[str, float], name: str) -> dict[str, float]:
    try:
        items = value.items()
    except AttributeError as exc:
        raise ValueError(f"{name} must be a symbol-to-weight mapping") from exc
    result: dict[str, float] = {}
    for symbol, weight in items:
        if not isinstance(symbol, str) or not symbol.strip():
            raise ValueError(f"{name} contains an invalid symbol")
        numeric = float(weight)
        if not math.isfinite(numeric):
            raise ValueError(f"{name} contains a non-finite weight")
        result[symbol] = numeric
    return result
