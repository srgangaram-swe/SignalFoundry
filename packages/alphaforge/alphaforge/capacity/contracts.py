"""Point-in-time borrow, locate, and liquidity records (SF-S4-MR5).

MR4 models the borrow *rate*. A rate is a price, and a price is not permission:
it says what a short would cost, never that the security was borrowable or that a
locate existed. This module supplies the missing evidence — versioned, bounded,
immutable records that state, for a named session, whether a symbol could be
shorted at all, up to what size, under whose authority, and until when.

Every record is treated as adversarial input under the security contract, so the
validators are the trust boundary and their refusals are the point:

* **Temporal semantics are explicit.** Each record carries the session it was
  *observed* in (``as_of_session``) separately from the session it is *effective*
  for. Collapsing those two is how a backtest silently learns tomorrow's borrow
  book: an availability row stamped only with its effective date looks perfectly
  causal while having been published after the fact.
* **Absence is never permission.** A missing, stale, expired, unknown, or
  conflicting record yields **zero** new-short capacity, never "unconstrained".
  The whole failure mode this MR exists to close is a backtest that shorts
  freely because it had no borrow data to say otherwise.
* **Identity is content-derived.** Each record and the resolved policy publish a
  canonical SHA-256 over their normalized fields, so a run can be tied to the
  exact borrow book it consumed and two different books can never be confused.

Digests provide identity and integrity evidence. They are **not** signatures and
carry no external provenance guarantee.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from datetime import date
from typing import Any, Final, Literal

#: Refusal thresholds, not tuning knobs. Every collection an untrusted source can
#: grow is bounded so a malformed book cannot exhaust memory.
MAX_SYMBOLS: Final = 5_000
MAX_RECORDS: Final = 200_000
MAX_SESSIONS: Final = 20_000
MAX_IDENTIFIER_CHARS: Final = 64
MAX_TEXT_CHARS: Final = 256

#: Largest share quantity or currency amount any record may declare. Above this a
#: value is a data error rather than a position, and admitting it would let a
#: single malformed row authorize an unbounded short.
MAX_QUANTITY: Final = 1e12
MAX_NOTIONAL: Final = 1e15

#: Supported record schema. An unrecognized version is refused rather than
#: best-effort parsed, because a field whose meaning changed silently is worse
#: than a field that is missing.
SCHEMA_VERSION: Final = "1.0.0"

BorrowStatus = Literal["available", "restricted", "recalled", "unknown"]

#: Statuses under which a *new* short may be opened. `restricted` and `recalled`
#: permit covering only, and `unknown` permits nothing.
SHORTABLE_STATUSES: Final[frozenset[str]] = frozenset({"available"})


class CapacityContractError(ValueError):
    """Raised when a borrow, locate, or liquidity record is unusable.

    A distinct type because the caller's response differs from a numerical
    failure: the input must be corrected or the symbol excluded, never retried.
    """


class CapacityPolicyViolation(RuntimeError):
    """Raised when an action would breach a resolved capacity policy.

    Separated from :class:`CapacityContractError` so a *well-formed* book that
    simply cannot support a trade is distinguishable from a malformed one.
    """


# ---------------------------------------------------------------------------
# Field validators — the trust boundary
# ---------------------------------------------------------------------------


def identifier(value: object, *, name: str) -> str:
    """Return a bounded ASCII identifier, or fail closed."""
    if not isinstance(value, str):
        raise CapacityContractError(f"{name} must be a string, got {type(value).__name__}")
    text = value.strip()
    if not text or text != value:
        raise CapacityContractError(f"{name} must be non-empty and free of surrounding space")
    if len(text) > MAX_IDENTIFIER_CHARS:
        raise CapacityContractError(f"{name} exceeds {MAX_IDENTIFIER_CHARS} characters")
    if not text.isascii() or not all(part.isalnum() or part in "._-" for part in text):
        raise CapacityContractError(f"{name} must be ASCII alphanumeric with . _ - only")
    return text


def bounded_text(value: object, *, name: str) -> str:
    """Return bounded ASCII free text (provenance, reasons), or fail closed."""
    if not isinstance(value, str):
        raise CapacityContractError(f"{name} must be a string, got {type(value).__name__}")
    text = value.strip()
    if not text:
        raise CapacityContractError(f"{name} must be non-empty")
    if len(text) > MAX_TEXT_CHARS:
        raise CapacityContractError(f"{name} exceeds {MAX_TEXT_CHARS} characters")
    if not text.isascii() or "\x00" in text:
        raise CapacityContractError(f"{name} must be printable ASCII")
    return text


def finite_quantity(value: object, *, name: str, maximum: float = MAX_QUANTITY) -> float:
    """Return a finite non-negative quantity within its declared ceiling.

    ``bool`` is refused explicitly: ``isinstance(True, int)`` holds in Python, so
    a boolean smuggled into a quantity field would silently become ``1`` share.
    """
    if isinstance(value, bool):
        raise CapacityContractError(f"{name} must be a real number, got bool")
    if not isinstance(value, (int, float)):
        raise CapacityContractError(f"{name} must be a real number, got {type(value).__name__}")
    number = float(value)
    if not math.isfinite(number):
        raise CapacityContractError(f"{name} must be finite, got {number}")
    if number < 0.0:
        raise CapacityContractError(f"{name} must be non-negative, got {number}")
    if number > maximum:
        raise CapacityContractError(f"{name} exceeds its {maximum} ceiling, got {number}")
    return number


def session_date(value: object, *, name: str) -> date:
    """Return a calendar session, refusing datetimes and strings.

    A ``datetime`` carries a time-of-day this model does not resolve; accepting
    one would imply an intraday precision the daily-bar evidence cannot support.
    """
    if isinstance(value, date) and not hasattr(value, "hour"):
        return value
    raise CapacityContractError(f"{name} must be a datetime.date, got {type(value).__name__}")


def _canonical_digest(payload: dict[str, Any]) -> str:
    """Return a deterministic SHA-256 over a normalized JSON payload.

    ``allow_nan=False`` matters: a NaN would serialize to the non-standard
    ``NaN`` token and two different books could then share a digest.
    """
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# Records
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class BorrowAvailability:
    """Point-in-time borrowable supply for one symbol in one session.

    Attributes:
        symbol: Instrument identifier.
        as_of_session: Session in which this row was **observed**. Decisions may
            only consume rows whose ``as_of_session`` is strictly earlier than
            the decision session — that separation is what makes the record
            causal.
        effective_session: Session the availability applies to.
        expiry_session: Last session the row remains valid, inclusive.
        status: ``available`` permits new shorts; ``restricted`` and ``recalled``
            permit covering only; ``unknown`` permits nothing.
        shortable_quantity: Maximum new short size in shares, in the issue's
            declared units. Zero is meaningful and distinct from missing.
        source: Provenance label, recorded in the digest.
        source_version: Version of the source book.

    Raises:
        CapacityContractError: On any malformed or out-of-bounds field.
    """

    symbol: str
    as_of_session: date
    effective_session: date
    expiry_session: date
    status: BorrowStatus
    shortable_quantity: float
    source: str
    source_version: str
    schema_version: str = SCHEMA_VERSION

    def __post_init__(self) -> None:
        object.__setattr__(self, "symbol", identifier(self.symbol, name="symbol"))
        object.__setattr__(self, "source", bounded_text(self.source, name="source"))
        object.__setattr__(
            self, "source_version", identifier(self.source_version, name="source_version")
        )
        if self.schema_version != SCHEMA_VERSION:
            raise CapacityContractError(
                f"unsupported borrow schema {self.schema_version!r}; expected {SCHEMA_VERSION}"
            )
        if self.status not in ("available", "restricted", "recalled", "unknown"):
            raise CapacityContractError(f"unsupported borrow status {self.status!r}")
        object.__setattr__(
            self, "as_of_session", session_date(self.as_of_session, name="as_of_session")
        )
        object.__setattr__(
            self,
            "effective_session",
            session_date(self.effective_session, name="effective_session"),
        )
        object.__setattr__(
            self, "expiry_session", session_date(self.expiry_session, name="expiry_session")
        )
        object.__setattr__(
            self,
            "shortable_quantity",
            finite_quantity(self.shortable_quantity, name="shortable_quantity"),
        )
        if self.expiry_session < self.effective_session:
            raise CapacityContractError("expiry_session precedes effective_session")
        if self.as_of_session > self.effective_session:
            raise CapacityContractError(
                "as_of_session follows effective_session; a row cannot be observed "
                "after the session it claims to describe"
            )
        if self.status not in SHORTABLE_STATUSES and self.shortable_quantity > 0.0:
            raise CapacityContractError(
                f"status {self.status!r} cannot carry positive shortable_quantity; "
                "a non-available symbol has no new-short capacity by definition"
            )

    @property
    def digest(self) -> str:
        """Deterministic content identity for this record."""
        return _canonical_digest(
            {
                "kind": "borrow_availability",
                "schema_version": self.schema_version,
                "symbol": self.symbol,
                "as_of_session": self.as_of_session.isoformat(),
                "effective_session": self.effective_session.isoformat(),
                "expiry_session": self.expiry_session.isoformat(),
                "status": self.status,
                "shortable_quantity": self.shortable_quantity,
                "source": self.source,
                "source_version": self.source_version,
            }
        )

    def covers(self, session: date) -> bool:
        """Whether this row is in force for ``session``."""
        return self.effective_session <= session <= self.expiry_session


@dataclass(frozen=True, slots=True)
class LocateRecord:
    """A simulated locate authorizing a bounded new short.

    A locate is the artifact a broker issues to say a specific quantity has been
    sourced. It is deliberately modelled as a separate, expiring record rather
    than folded into availability: availability is a property of the market,
    a locate is a *grant* to one account, and the two expire independently.

    Attributes:
        locate_id: Synthetic identifier. Real broker locate identifiers must
            never be committed; this is a simulation artifact.
        granted_session: Session the locate was issued in.
        expiry_session: Last session it may authorize a fill, inclusive.
        quantity: Authorized share count.
    """

    locate_id: str
    symbol: str
    granted_session: date
    expiry_session: date
    quantity: float
    source: str
    schema_version: str = SCHEMA_VERSION

    def __post_init__(self) -> None:
        object.__setattr__(self, "locate_id", identifier(self.locate_id, name="locate_id"))
        object.__setattr__(self, "symbol", identifier(self.symbol, name="symbol"))
        object.__setattr__(self, "source", bounded_text(self.source, name="source"))
        if self.schema_version != SCHEMA_VERSION:
            raise CapacityContractError(
                f"unsupported locate schema {self.schema_version!r}; expected {SCHEMA_VERSION}"
            )
        object.__setattr__(
            self, "granted_session", session_date(self.granted_session, name="granted_session")
        )
        object.__setattr__(
            self, "expiry_session", session_date(self.expiry_session, name="expiry_session")
        )
        object.__setattr__(self, "quantity", finite_quantity(self.quantity, name="quantity"))
        if self.expiry_session < self.granted_session:
            raise CapacityContractError("locate expiry_session precedes granted_session")
        if self.quantity <= 0.0:
            raise CapacityContractError("a locate must authorize a positive quantity")

    @property
    def digest(self) -> str:
        """Deterministic content identity for this locate."""
        return _canonical_digest(
            {
                "kind": "locate",
                "schema_version": self.schema_version,
                "locate_id": self.locate_id,
                "symbol": self.symbol,
                "granted_session": self.granted_session.isoformat(),
                "expiry_session": self.expiry_session.isoformat(),
                "quantity": self.quantity,
                "source": self.source,
            }
        )

    def active(self, session: date) -> bool:
        """Whether this locate may authorize a fill in ``session``."""
        return self.granted_session <= session <= self.expiry_session


@dataclass(frozen=True, slots=True)
class LiquidityObservation:
    """Lagged average daily volume for one symbol, with its information time.

    ``as_of_session`` is the session the ADV was *computed through*. The policy
    admits a row only when that session is strictly earlier than the decision
    session, which is what prevents a trade being sized on volume it helped
    create.

    Attributes:
        adv_shares: Average daily volume in shares.
        adv_notional: Average daily traded notional in currency.
        lookback_sessions: Sessions the average covers, recorded so a stale or
            unusually short window is auditable rather than implicit.
    """

    symbol: str
    as_of_session: date
    adv_shares: float
    adv_notional: float
    lookback_sessions: int
    source: str
    schema_version: str = SCHEMA_VERSION

    def __post_init__(self) -> None:
        object.__setattr__(self, "symbol", identifier(self.symbol, name="symbol"))
        object.__setattr__(self, "source", bounded_text(self.source, name="source"))
        if self.schema_version != SCHEMA_VERSION:
            raise CapacityContractError(
                f"unsupported liquidity schema {self.schema_version!r}; expected {SCHEMA_VERSION}"
            )
        object.__setattr__(
            self, "as_of_session", session_date(self.as_of_session, name="as_of_session")
        )
        object.__setattr__(self, "adv_shares", finite_quantity(self.adv_shares, name="adv_shares"))
        object.__setattr__(
            self,
            "adv_notional",
            finite_quantity(self.adv_notional, name="adv_notional", maximum=MAX_NOTIONAL),
        )
        if isinstance(self.lookback_sessions, bool) or not isinstance(self.lookback_sessions, int):
            raise CapacityContractError("lookback_sessions must be an int")
        if not 1 <= self.lookback_sessions <= MAX_SESSIONS:
            raise CapacityContractError(
                f"lookback_sessions must be in [1, {MAX_SESSIONS}], got {self.lookback_sessions}"
            )

    @property
    def digest(self) -> str:
        """Deterministic content identity for this observation."""
        return _canonical_digest(
            {
                "kind": "liquidity",
                "schema_version": self.schema_version,
                "symbol": self.symbol,
                "as_of_session": self.as_of_session.isoformat(),
                "adv_shares": self.adv_shares,
                "adv_notional": self.adv_notional,
                "lookback_sessions": self.lookback_sessions,
                "source": self.source,
            }
        )


@dataclass(frozen=True, slots=True)
class CapacityPolicyDeclaration:
    """The frozen policy every capacity decision is evaluated against.

    Freezing this before evaluation is what makes the resulting frontier
    meaningful: a participation cap chosen after seeing the fills is a fitted
    parameter, not a constraint.

    Attributes:
        max_participation: Fraction of a symbol's lagged ADV the book may trade
            in one session, in ``[0, 1]``.
        max_session_notional: Book-level gross traded-notional budget per logical
            session. This is the conserved aggregate the issue requires.
        max_staleness_sessions: How old a liquidity or borrow observation may be
            before it is treated as unusable. Beyond this the symbol gets zero
            new-short capacity rather than a stale allowance.
        buy_in_resolution_sessions: Bounded number of sessions a forced buy-in
            may take before an unresolved residual halts publication.
        allow_new_shorts: Master switch. ``False`` is the documented rollback
            posture — long-only — and must never be reachable by *failure*, only
            by explicit declaration.
    """

    max_participation: float
    max_session_notional: float
    max_staleness_sessions: int
    buy_in_resolution_sessions: int
    allow_new_shorts: bool = True
    policy_id: str = "default"
    schema_version: str = SCHEMA_VERSION

    def __post_init__(self) -> None:
        object.__setattr__(self, "policy_id", identifier(self.policy_id, name="policy_id"))
        if self.schema_version != SCHEMA_VERSION:
            raise CapacityContractError(
                f"unsupported policy schema {self.schema_version!r}; expected {SCHEMA_VERSION}"
            )
        if isinstance(self.allow_new_shorts, bool) is False:
            raise CapacityContractError("allow_new_shorts must be a bool")
        participation = finite_quantity(
            self.max_participation, name="max_participation", maximum=1.0
        )
        if participation <= 0.0:
            raise CapacityContractError("max_participation must be strictly positive")
        object.__setattr__(self, "max_participation", participation)
        object.__setattr__(
            self,
            "max_session_notional",
            finite_quantity(
                self.max_session_notional, name="max_session_notional", maximum=MAX_NOTIONAL
            ),
        )
        if self.max_session_notional <= 0.0:
            raise CapacityContractError("max_session_notional must be strictly positive")
        for field_name in ("max_staleness_sessions", "buy_in_resolution_sessions"):
            value = getattr(self, field_name)
            if isinstance(value, bool) or not isinstance(value, int):
                raise CapacityContractError(f"{field_name} must be an int")
            if not 0 <= value <= MAX_SESSIONS:
                raise CapacityContractError(
                    f"{field_name} must be in [0, {MAX_SESSIONS}], got {value}"
                )

    @property
    def digest(self) -> str:
        """Deterministic content identity for the frozen policy."""
        return _canonical_digest(
            {
                "kind": "capacity_policy",
                "schema_version": self.schema_version,
                "policy_id": self.policy_id,
                "max_participation": self.max_participation,
                "max_session_notional": self.max_session_notional,
                "max_staleness_sessions": self.max_staleness_sessions,
                "buy_in_resolution_sessions": self.buy_in_resolution_sessions,
                "allow_new_shorts": self.allow_new_shorts,
            }
        )

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-friendly declaration for the evidence record."""
        return {
            "policy_id": self.policy_id,
            "schema_version": self.schema_version,
            "max_participation": self.max_participation,
            "max_session_notional": self.max_session_notional,
            "max_staleness_sessions": self.max_staleness_sessions,
            "buy_in_resolution_sessions": self.buy_in_resolution_sessions,
            "allow_new_shorts": self.allow_new_shorts,
            "units": {
                "max_participation": "fraction of lagged ADV shares per session",
                "max_session_notional": "currency, gross traded notional per session",
                "max_staleness_sessions": "sessions",
                "buy_in_resolution_sessions": "sessions",
            },
            "failure_behavior": (
                "missing, stale, expired, unknown, duplicated, or conflicting records "
                "yield zero new-short capacity; covers remain permitted"
            ),
            "digest": self.digest,
        }


def validate_unique_records(records: object, *, name: str) -> tuple[Any, ...]:
    """Return a bounded tuple of records, refusing duplicates and conflicts.

    Two rows describing the same ``(symbol, effective_session)`` are a conflict,
    not a preference. Silently taking the last one would let record ordering
    decide how much a strategy may short.
    """
    if not isinstance(records, (list, tuple)):
        raise CapacityContractError(f"{name} must be a list or tuple")
    ordered = tuple(records)
    if len(ordered) > MAX_RECORDS:
        raise CapacityContractError(f"{name} exceeds the {MAX_RECORDS}-record ceiling")
    seen: dict[tuple[str, str], str] = {}
    for record in ordered:
        if isinstance(record, BorrowAvailability):
            key = (record.symbol, record.effective_session.isoformat())
        elif isinstance(record, LiquidityObservation):
            key = (record.symbol, record.as_of_session.isoformat())
        elif isinstance(record, LocateRecord):
            key = (record.locate_id, record.symbol)
        else:
            raise CapacityContractError(f"{name} contains an unsupported record type")
        digest = record.digest
        if key in seen:
            if seen[key] == digest:
                raise CapacityContractError(f"{name} contains a duplicate record for {key}")
            raise CapacityContractError(
                f"{name} contains conflicting records for {key}; the book must be "
                "corrected rather than resolved by ordering"
            )
        seen[key] = digest
    return ordered


def book_digest(
    borrow: tuple[BorrowAvailability, ...],
    locates: tuple[LocateRecord, ...],
    liquidity: tuple[LiquidityObservation, ...],
    policy: CapacityPolicyDeclaration,
) -> str:
    """Return one identity binding an entire resolved capacity book.

    Sorted by record digest so the identity is independent of input ordering —
    two runs that consumed the same rows in a different order must produce the
    same book identity, or the identity would track file layout rather than
    content.
    """
    return _canonical_digest(
        {
            "kind": "capacity_book",
            "schema_version": SCHEMA_VERSION,
            "policy": policy.digest,
            "borrow": sorted(record.digest for record in borrow),
            "locates": sorted(record.digest for record in locates),
            "liquidity": sorted(record.digest for record in liquidity),
        }
    )
