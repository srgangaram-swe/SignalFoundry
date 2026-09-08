"""Inert-by-default minimal-capital configuration that cannot expand its own risk.

SF-S5-MR10. Everything else in this repository computes. This module is the only
place that would describe real money, and it is written on the assumption that
every other control has already failed.

**Inert by default.** ``enabled`` is ``False`` and there is no constructor that
produces an active configuration without an explicit authorization record. A
configuration that forgets to mention capital deploys none.

**It cannot expand risk autonomously.** There is no method that raises a cap.
Not a guarded one, not a validated one — none. Increasing exposure requires
constructing a new configuration with a new authorization naming the new cap,
which leaves a record. A mutable cap with a permission check is one bug away from
an unbounded one; an absent method is not.

**Authorization is bound to what it authorizes.** A
:class:`CapitalAuthorization` names the cap, the readiness decision identity, the
approver, and an expiry. Presenting it for a different cap or a different
readiness decision is refused, so an approval cannot be recycled onto a larger
deployment than the one it was given for.

**It expires.** An authorization is a decision about a moment. Markets, policy,
and personal circumstances change, and an approval that never expires eventually
authorizes something nobody looked at.

**Everything is reversible.** :meth:`LiveCapitalConfig.deactivated` returns an
inert copy; there is no state to unwind and no partial teardown.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from typing import Any, Final

from alphaforge.readiness.checklist import ReadinessDecision, ReadinessError, Verdict

#: Absolute ceiling on any minimal-capital deployment, independent of what an
#: authorization requests. A first deployment is an operational test; sizing it
#: for return defeats the purpose.
ABSOLUTE_MAX_CAPITAL: Final = Decimal("1000")

#: An authorization may not run longer than this regardless of its stated expiry.
MAX_AUTHORIZATION_DAYS: Final = 30

#: Ceilings on the risk limits themselves, so a configuration cannot be written
#: with limits so loose they impose nothing.
MAX_POSITION_FRACTION: Final = Decimal("0.25")
MAX_DAILY_LOSS_FRACTION: Final = Decimal("0.10")
MAX_DRAWDOWN_FRACTION: Final = Decimal("0.20")


class CapitalError(ReadinessError):
    """Raised when a capital configuration or authorization is unsafe."""


class NotAuthorizedError(CapitalError):
    """Raised when activation is attempted without valid authorization.

    A distinct type because this is the one failure that must never be caught
    and retried by generic error handling.
    """


def _decimal(value: object, *, field_name: str) -> Decimal:
    """Return a finite non-negative Decimal, refusing floats."""
    if isinstance(value, float):
        raise CapitalError(
            f"{field_name} must be a Decimal, int, or str — not a float. Money that cannot "
            "be represented exactly has no place in a capital limit."
        )
    if isinstance(value, bool) or not isinstance(value, (int, str, Decimal)):
        raise CapitalError(f"{field_name} must be a Decimal, int, or str")
    amount = Decimal(value)
    if not amount.is_finite():
        raise CapitalError(f"{field_name} must be finite")
    if amount < 0:
        raise CapitalError(f"{field_name} must not be negative")
    return amount


@dataclass(frozen=True)
class CapitalAuthorization:
    """A named approval for one specific deployment, bound to what it authorizes.

    Raises:
        CapitalError: On a malformed, oversized, or over-long authorization.
    """

    approver: str
    approved_cap: Decimal
    readiness_identity: str
    approved_at: datetime
    expires_at: datetime
    reference: str

    def __post_init__(self) -> None:
        for field_name in ("approver", "reference"):
            value = getattr(self, field_name)
            if not isinstance(value, str) or not value.strip() or value != value.strip():
                raise CapitalError(f"{field_name} must be a non-empty unpadded string")
        object.__setattr__(
            self, "approved_cap", _decimal(self.approved_cap, field_name="approved_cap")
        )
        if self.approved_cap <= 0:
            raise CapitalError("approved_cap must be positive; a zero cap authorizes nothing")
        if self.approved_cap > ABSOLUTE_MAX_CAPITAL:
            raise CapitalError(
                f"approved_cap {self.approved_cap} exceeds the absolute ceiling "
                f"{ABSOLUTE_MAX_CAPITAL}. A first deployment is an operational test; sizing "
                "it for return defeats the purpose."
            )
        if not isinstance(self.readiness_identity, str) or len(self.readiness_identity) != 64:
            raise CapitalError("readiness_identity must be a full SHA-256 digest")
        for field_name in ("approved_at", "expires_at"):
            value = getattr(self, field_name)
            if not isinstance(value, datetime) or value.tzinfo is None:
                raise CapitalError(f"{field_name} must be a timezone-aware datetime")
            object.__setattr__(self, field_name, value.astimezone(UTC))
        if self.expires_at <= self.approved_at:
            raise CapitalError("expires_at must follow approved_at")
        if self.expires_at - self.approved_at > timedelta(days=MAX_AUTHORIZATION_DAYS):
            raise CapitalError(
                f"authorization spans more than {MAX_AUTHORIZATION_DAYS} days; an approval "
                "that runs long enough eventually authorizes something nobody looked at"
            )

    def is_current(self, *, now: datetime) -> bool:
        """Whether this authorization is within its window."""
        moment = now.astimezone(UTC)
        return self.approved_at <= moment < self.expires_at

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-friendly record."""
        return {
            "approver": self.approver,
            "approved_cap": str(self.approved_cap),
            "readiness_identity": self.readiness_identity,
            "approved_at": self.approved_at.isoformat(),
            "expires_at": self.expires_at.isoformat(),
            "reference": self.reference,
        }


@dataclass(frozen=True)
class LiveCapitalConfig:
    """A minimal-capital deployment configuration. Inert unless activated.

    Construct with :meth:`inert` and obtain an active copy only through
    :func:`activate`, which requires a matching readiness decision and a current
    authorization. There is deliberately **no** method that raises a cap.

    Raises:
        CapitalError: On unsafe limits or an internally inconsistent state.
    """

    enabled: bool
    capital_cap: Decimal
    max_position_fraction: Decimal
    max_daily_loss_fraction: Decimal
    max_drawdown_fraction: Decimal
    authorization: CapitalAuthorization | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.enabled, bool):
            raise CapitalError("enabled must be a bool")
        object.__setattr__(
            self, "capital_cap", _decimal(self.capital_cap, field_name="capital_cap")
        )
        if self.capital_cap > ABSOLUTE_MAX_CAPITAL:
            raise CapitalError(
                f"capital_cap {self.capital_cap} exceeds the absolute ceiling "
                f"{ABSOLUTE_MAX_CAPITAL}"
            )
        bounds = (
            ("max_position_fraction", MAX_POSITION_FRACTION),
            ("max_daily_loss_fraction", MAX_DAILY_LOSS_FRACTION),
            ("max_drawdown_fraction", MAX_DRAWDOWN_FRACTION),
        )
        for field_name, ceiling in bounds:
            value = _decimal(getattr(self, field_name), field_name=field_name)
            if value <= 0:
                raise CapitalError(
                    f"{field_name} must be positive; a zero limit permits no position at all "
                    "and is almost certainly a mistake"
                )
            if value > ceiling:
                raise CapitalError(
                    f"{field_name} {value} exceeds {ceiling}; a limit that loose imposes "
                    "nothing and would pass review while constraining nothing"
                )
            object.__setattr__(self, field_name, value)
        if self.enabled:
            if self.authorization is None:
                raise NotAuthorizedError(
                    "an enabled configuration requires an authorization; there is no path "
                    "from 'no approval' to 'deploying capital'"
                )
            if self.capital_cap > self.authorization.approved_cap:
                raise NotAuthorizedError(
                    f"configured cap {self.capital_cap} exceeds the approved cap "
                    f"{self.authorization.approved_cap}; an approval cannot be recycled onto "
                    "a larger deployment than it was given for"
                )
        elif self.capital_cap != 0 and self.authorization is None:
            # A disabled config may carry a cap for review, but it deploys nothing.
            pass

    @classmethod
    def inert(cls) -> LiveCapitalConfig:
        """Return a configuration that deploys nothing. The default state."""
        return cls(
            enabled=False,
            capital_cap=Decimal("0"),
            max_position_fraction=Decimal("0.05"),
            max_daily_loss_fraction=Decimal("0.02"),
            max_drawdown_fraction=Decimal("0.05"),
            authorization=None,
        )

    def deactivated(self) -> LiveCapitalConfig:
        """Return an inert copy. Reversal is total and has no partial state."""
        return LiveCapitalConfig(
            enabled=False,
            capital_cap=Decimal("0"),
            max_position_fraction=self.max_position_fraction,
            max_daily_loss_fraction=self.max_daily_loss_fraction,
            max_drawdown_fraction=self.max_drawdown_fraction,
            authorization=None,
        )

    def max_position_value(self) -> Decimal:
        """The largest single position this configuration permits."""
        return self.capital_cap * self.max_position_fraction

    def identity(self) -> str:
        """Content identity, so a deployed configuration is auditable."""
        return hashlib.sha256(
            json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-friendly, auditable record."""
        return {
            "enabled": self.enabled,
            "capital_cap": str(self.capital_cap),
            "max_position_fraction": str(self.max_position_fraction),
            "max_daily_loss_fraction": str(self.max_daily_loss_fraction),
            "max_drawdown_fraction": str(self.max_drawdown_fraction),
            "max_position_value": str(self.max_position_value()),
            "authorization": None if self.authorization is None else self.authorization.to_dict(),
            "absolute_ceiling": str(ABSOLUTE_MAX_CAPITAL),
            "expansion_policy": (
                "There is no method that raises a cap. Increasing exposure requires a new "
                "configuration with a new authorization naming the new cap."
            ),
        }


def activate(
    decision: ReadinessDecision,
    authorization: CapitalAuthorization,
    *,
    capital_cap: Decimal,
    max_position_fraction: Decimal,
    max_daily_loss_fraction: Decimal,
    max_drawdown_fraction: Decimal,
    now: datetime,
) -> LiveCapitalConfig:
    """Return an active configuration, or refuse.

    Four independent conditions, each alone sufficient to refuse: the readiness
    decision is READY, the authorization is bound to *that* decision, the
    authorization is current, and the requested cap is within it.

    Raises:
        NotAuthorizedError: When any condition fails, naming which.
    """
    if not isinstance(decision, ReadinessDecision):
        raise NotAuthorizedError(
            "readiness must be a ReadinessDecision produced by the checklist, not a "
            "caller-constructed stand-in"
        )
    if not decision.ready:
        raise NotAuthorizedError(
            f"readiness verdict is {decision.verdict.value} with {len(decision.unmet)} unmet "
            f"item(s): {list(decision.unmet)[:5]}. No capital deployment is authorized, and "
            "this is not a configuration problem."
        )
    if not isinstance(authorization, CapitalAuthorization):
        raise NotAuthorizedError("authorization must be a CapitalAuthorization")
    if authorization.readiness_identity != decision.checklist_identity:
        raise NotAuthorizedError(
            f"authorization was granted against checklist "
            f"{authorization.readiness_identity[:12]}, but this decision used "
            f"{decision.checklist_identity[:12]}. An approval cannot be recycled onto a "
            "deployment evaluated under a different checklist."
        )
    if not authorization.is_current(now=now):
        raise NotAuthorizedError(
            f"authorization by {authorization.approver} is outside its window "
            f"({authorization.approved_at.date()} to {authorization.expires_at.date()}); an "
            "approval is a decision about a moment"
        )
    requested = _decimal(capital_cap, field_name="capital_cap")
    if requested > authorization.approved_cap:
        raise NotAuthorizedError(
            f"requested cap {requested} exceeds the approved cap " f"{authorization.approved_cap}"
        )
    return LiveCapitalConfig(
        enabled=True,
        capital_cap=requested,
        max_position_fraction=max_position_fraction,
        max_daily_loss_fraction=max_daily_loss_fraction,
        max_drawdown_fraction=max_drawdown_fraction,
        authorization=authorization,
    )


def assert_within_limits(
    config: LiveCapitalConfig,
    *,
    position_value: Decimal,
    daily_loss: Decimal,
    drawdown: Decimal,
) -> None:
    """Refuse an exposure that breaches any configured limit.

    Checked against the configuration rather than against the broker, so a
    breach is refused before an order is built rather than after it fills.

    Raises:
        CapitalError: Naming the breached limit.
        NotAuthorizedError: If the configuration is inert.
    """
    if not config.enabled:
        raise NotAuthorizedError("configuration is inert; no exposure is permitted at all")
    checks = (
        (
            "position value",
            _decimal(position_value, field_name="position_value"),
            config.max_position_value(),
        ),
        (
            "daily loss",
            _decimal(daily_loss, field_name="daily_loss"),
            config.capital_cap * config.max_daily_loss_fraction,
        ),
        (
            "drawdown",
            _decimal(drawdown, field_name="drawdown"),
            config.capital_cap * config.max_drawdown_fraction,
        ),
    )
    breached = [(name, observed, limit) for name, observed, limit in checks if observed > limit]
    if breached:
        detail = "; ".join(
            f"{name} {observed} exceeds {limit}" for name, observed, limit in breached
        )
        raise CapitalError(f"{len(breached)} limit(s) breached: {detail}")


__all__ = [
    "ABSOLUTE_MAX_CAPITAL",
    "MAX_AUTHORIZATION_DAYS",
    "MAX_DAILY_LOSS_FRACTION",
    "MAX_DRAWDOWN_FRACTION",
    "MAX_POSITION_FRACTION",
    "CapitalAuthorization",
    "CapitalError",
    "LiveCapitalConfig",
    "NotAuthorizedError",
    "Verdict",
    "activate",
    "assert_within_limits",
]
