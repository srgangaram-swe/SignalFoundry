"""Deny-by-default broker configuration and the paper-only endpoint boundary.

SF-S5-MR3. This module decides whether a broker session may exist at all. It is
the smallest and most security-relevant part of the broker package, and it is
separate from the adapter so that the enabling decision cannot be made
incidentally by code whose job is order routing.

Three independent conditions must hold before a session is constructed, and each
one alone is sufficient to refuse:

1. **Explicitly enabled.** ``enabled`` defaults to ``False``. A configuration
   that forgets to mention trading does not trade.
2. **Paper-only endpoint.** The endpoint must appear on
   :data:`ALLOWED_PAPER_ENDPOINTS`. This is an allowlist, not a denylist of live
   hosts: a denylist fails open the moment a vendor adds a hostname.
3. **A qualified candidate.** A ``QUALIFIED_FOR_PAPER`` decision must be
   supplied. Sprint 4's verdict is ``REJECTED``, so this condition is currently
   unsatisfiable — which is the correct state, not an obstacle to work around.

There is no ``force``, ``override``, ``allow_live``, or ``skip_checks``
parameter anywhere in this module. Live capability is not disabled by a flag
here; it is **absent**, and enabling it is a future change that must pass its own
review rather than a value someone can set.
"""

from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass
from typing import Any, Final
from urllib.parse import urlparse

from alphaforge.broker.contracts import BrokerContractError
from alphaforge.research.qualification import QualificationDecision

#: Endpoints a paper session may address. An allowlist rather than a denylist:
#: a denylist of live hosts fails open as soon as a vendor introduces a new one,
#: and failing open here means routing a real order.
ALLOWED_PAPER_ENDPOINTS: Final[frozenset[str]] = frozenset(
    {
        "https://paper-api.alpaca.markets",
        "https://broker-api.sandbox.alpaca.markets",
    }
)

#: Substrings that indicate a production trading host. Checked in addition to
#: the allowlist as defense in depth: if an allowlist entry is ever edited
#: carelessly, this catches the obvious mistake.
_LIVE_ENDPOINT_MARKERS: Final[tuple[str, ...]] = ("api.alpaca.markets", "live", "production")

#: Keychain service names permitted for credential lookup. Credentials are never
#: read from configuration, environment variables, or files.
ALLOWED_KEYCHAIN_SERVICES: Final[frozenset[str]] = frozenset({"com.signal-foundry.alpaca-paper"})

#: Environment variables that must NOT carry broker credentials. Their presence
#: is treated as a misconfiguration because an environment variable is readable
#: by every child process and lands in crash dumps and process listings.
FORBIDDEN_CREDENTIAL_ENVIRONMENT: Final[tuple[str, ...]] = (
    "ALPACA_API_KEY",
    "ALPACA_SECRET_KEY",
    "APCA_API_KEY_ID",
    "APCA_API_SECRET_KEY",
    "BROKER_API_KEY",
    "BROKER_SECRET_KEY",
)


class BrokerConfigurationError(BrokerContractError):
    """Raised when a broker configuration is unsafe or incomplete."""


class LiveTradingNotAuthorizedError(BrokerConfigurationError):
    """Raised when configuration would permit contact with a live endpoint.

    A distinct type because this is the one failure that must never be caught
    and retried by generic error handling.
    """


def digest_account_identifier(raw_account_id: str) -> str:
    """Return a SHA-256 digest of an account identifier.

    The raw identifier is an account identifier under the credential-custody
    policy and must not be stored, logged, or serialized. A digest is enough to
    detect that the account changed between snapshots, which is the only thing
    the system needs it for.

    Raises:
        BrokerConfigurationError: If the identifier is empty.
    """
    if not isinstance(raw_account_id, str) or not raw_account_id.strip():
        raise BrokerConfigurationError("account identifier must be a non-empty string")
    return hashlib.sha256(raw_account_id.strip().encode("utf-8")).hexdigest()


def assert_endpoint_is_paper(endpoint: object) -> str:
    """Refuse any endpoint that is not an allowlisted paper host.

    Raises:
        LiveTradingNotAuthorizedError: If the endpoint is not allowlisted or
            carries a live-host marker.
        BrokerConfigurationError: If the endpoint is malformed.
    """
    if not isinstance(endpoint, str) or not endpoint.strip():
        raise BrokerConfigurationError("endpoint must be a non-empty string")
    candidate = endpoint.strip().rstrip("/")
    parsed = urlparse(candidate)
    if parsed.scheme != "https":
        raise BrokerConfigurationError(
            f"endpoint must use https, got {parsed.scheme!r}; credentials must never "
            "traverse an unencrypted connection"
        )
    if candidate not in ALLOWED_PAPER_ENDPOINTS:
        raise LiveTradingNotAuthorizedError(
            f"endpoint {candidate!r} is not an allowlisted paper endpoint. Permitted: "
            f"{sorted(ALLOWED_PAPER_ENDPOINTS)}. This is an allowlist rather than a "
            "denylist of live hosts, because a denylist fails open when a vendor adds "
            "a hostname — and failing open here routes a real order."
        )
    host = (parsed.hostname or "").lower()
    for marker in _LIVE_ENDPOINT_MARKERS:
        if marker in host and not host.startswith(("paper-", "broker-api.sandbox")):
            raise LiveTradingNotAuthorizedError(
                f"endpoint host {host!r} carries the live marker {marker!r}"
            )
    return candidate


def assert_no_credentials_in_environment(
    environment: dict[str, str] | None = None,
) -> None:
    """Refuse to proceed when a broker credential is present in the environment.

    An environment variable is readable by every child process and appears in
    crash dumps and process listings. Credentials belong in the Keychain and are
    retrieved for the lifetime of a bounded operation.

    Raises:
        BrokerConfigurationError: Naming the offending variables. The values are
            never included in the message.
    """
    source = os.environ if environment is None else environment
    present = sorted(name for name in FORBIDDEN_CREDENTIAL_ENVIRONMENT if source.get(name))
    if present:
        raise BrokerConfigurationError(
            f"broker credentials found in environment variables {present}. Credentials must "
            "come from the Keychain only: an environment variable is readable by every "
            "child process and lands in crash dumps and process listings. "
            "(Values are deliberately not shown.)"
        )


@dataclass(frozen=True)
class BrokerSessionConfig:
    """Configuration that must be satisfied before a broker session may exist.

    Every field defaults to the refusing value, so an empty or partial
    configuration cannot produce a working session.

    Attributes:
        enabled: Master switch. Defaults to ``False``.
        endpoint: Must be an allowlisted paper endpoint.
        keychain_service: Must be an allowlisted Keychain service name.
        max_quote_age_seconds: Quotes older than this block submission. A quote
            without an observation time is treated as infinitely old.
        max_clock_skew_seconds: Local-versus-broker skew beyond this halts.
        max_account_snapshot_age_seconds: A snapshot older than this cannot
            support an exposure check.
        max_order_attempts: Bound on retries of a retryable failure.
        max_orders_per_cycle: Bound on submissions in one decision cycle, so a
            malformed rebalance cannot emit an unbounded burst.

    Raises:
        BrokerConfigurationError: On any unsafe or malformed value.
        LiveTradingNotAuthorizedError: If the endpoint is not paper-only.
    """

    enabled: bool = False
    endpoint: str = ""
    keychain_service: str = ""
    max_quote_age_seconds: float = 300.0
    max_clock_skew_seconds: float = 5.0
    max_account_snapshot_age_seconds: float = 60.0
    max_order_attempts: int = 3
    max_orders_per_cycle: int = 100

    def __post_init__(self) -> None:
        if not isinstance(self.enabled, bool):
            raise BrokerConfigurationError("enabled must be a bool")
        # An endpoint is validated whenever one is supplied, even while
        # disabled: a latent bad endpoint should fail when it is written, not
        # on the day someone flips the switch.
        if self.endpoint:
            object.__setattr__(self, "endpoint", assert_endpoint_is_paper(self.endpoint))
        elif self.enabled:
            raise BrokerConfigurationError("an enabled session requires an endpoint")
        if self.keychain_service:
            if self.keychain_service not in ALLOWED_KEYCHAIN_SERVICES:
                raise BrokerConfigurationError(
                    f"keychain_service {self.keychain_service!r} is not allowlisted; "
                    f"permitted: {sorted(ALLOWED_KEYCHAIN_SERVICES)}"
                )
        elif self.enabled:
            raise BrokerConfigurationError("an enabled session requires a keychain_service")

        for name in (
            "max_quote_age_seconds",
            "max_clock_skew_seconds",
            "max_account_snapshot_age_seconds",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise BrokerConfigurationError(f"{name} must be a real number")
            if not value > 0.0 or value != value or value == float("inf"):
                raise BrokerConfigurationError(f"{name} must be finite and positive")
        for name in ("max_order_attempts", "max_orders_per_cycle"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise BrokerConfigurationError(f"{name} must be a positive int")
        if self.max_order_attempts > 10:
            raise BrokerConfigurationError(
                "max_order_attempts above 10 is refused; an unbounded retry against an "
                "order endpoint is how a transient fault becomes duplicate exposure"
            )
        if self.max_orders_per_cycle > 1_000:
            raise BrokerConfigurationError("max_orders_per_cycle exceeds the 1000 ceiling")

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-friendly record. Contains no credential."""
        return {
            "enabled": self.enabled,
            "endpoint": self.endpoint,
            "keychain_service": self.keychain_service,
            "max_quote_age_seconds": self.max_quote_age_seconds,
            "max_clock_skew_seconds": self.max_clock_skew_seconds,
            "max_account_snapshot_age_seconds": self.max_account_snapshot_age_seconds,
            "max_order_attempts": self.max_order_attempts,
            "max_orders_per_cycle": self.max_orders_per_cycle,
            "paper_only": True,
            "live_capability_present": False,
        }


def authorize_paper_session(
    config: BrokerSessionConfig,
    *,
    qualification: QualificationDecision | None,
    environment: dict[str, str] | None = None,
) -> None:
    """Refuse unless every independent condition for a paper session holds.

    Checks, in order: credentials absent from the environment, session enabled,
    endpoint allowlisted, and a ``QUALIFIED_FOR_PAPER`` decision supplied.

    The qualification check is last so its message is the one a caller sees once
    the mechanical problems are fixed — and it is the one that cannot be fixed by
    configuration.

    Raises:
        BrokerConfigurationError: If credentials are in the environment, the
            session is disabled, or no qualified decision is supplied.
        LiveTradingNotAuthorizedError: If the endpoint is not paper-only.
    """
    assert_no_credentials_in_environment(environment)

    if not config.enabled:
        raise BrokerConfigurationError(
            "broker session is disabled. `enabled` defaults to False so a configuration "
            "that forgets to mention trading does not trade."
        )
    assert_endpoint_is_paper(config.endpoint)
    if not config.keychain_service:
        raise BrokerConfigurationError("an enabled session requires a keychain_service")

    if qualification is None:
        raise BrokerConfigurationError(
            "no qualification decision supplied. A paper session may operationalize only a "
            "strategy that cleared the formal qualification gate; absent a decision there "
            "is nothing to operationalize."
        )
    if not isinstance(qualification, QualificationDecision):
        raise BrokerConfigurationError(
            "qualification must be a QualificationDecision produced by the frozen rubric, "
            "not a caller-constructed stand-in"
        )
    if not qualification.qualified:
        raise BrokerConfigurationError(
            f"candidate {qualification.candidate_id!r} has verdict {qualification.verdict!r} "
            f"with {len(qualification.blocking_failures)} blocking failure(s): "
            f"{list(qualification.blocking_failures)[:5]}. Paper evaluation requires "
            "QUALIFIED_FOR_PAPER. This is not a configuration problem and cannot be "
            "resolved by changing a setting."
        )


__all__ = [
    "ALLOWED_KEYCHAIN_SERVICES",
    "ALLOWED_PAPER_ENDPOINTS",
    "FORBIDDEN_CREDENTIAL_ENVIRONMENT",
    "BrokerConfigurationError",
    "BrokerSessionConfig",
    "LiveTradingNotAuthorizedError",
    "assert_endpoint_is_paper",
    "assert_no_credentials_in_environment",
    "authorize_paper_session",
    "digest_account_identifier",
]
