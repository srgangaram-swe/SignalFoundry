"""Strict, immutable market-friction and logical-latency contracts.

The objects in this module are pure domain records.  They perform no network,
broker, calendar, or storage I/O and never infer unavailable market state.
Latency is expressed only in positions on a caller-supplied frozen trading
calendar.  Carry costs are declared simulation sensitivities: they calculate
USD financing and borrow charges but make no claim that stock was borrowable,
located, or executable at the supplied price.

Every public model exposes a content-derived declaration/configuration digest.
Those identities make assumptions auditable without presenting a proxy model
as observed execution quality or calibrating it from a strategy test outcome.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from datetime import date, datetime
from typing import ClassVar, Literal

MAX_IDENTIFIER_LENGTH = 128
MAX_METADATA_TEXT = 512
MAX_DECLARATION_ITEMS = 32
MAX_LIMITATIONS = 16
MAX_CALENDAR_SESSIONS = 1_000_000
MAX_STAGE_DELAY_SESSIONS = 2_520
MAX_TOTAL_DELAY_SESSIONS = 5_040
MAX_SESSIONS_PER_YEAR = 366
MAX_RATE_BPS_ANNUAL = 1_000_000.0
MAX_MULTIPLIER = 1_000.0
MAX_MONEY_USD = 1.0e18
MAX_QUANTITY_SHARES = 1.0e15
MAX_PRICE_USD = 1.0e12
MAX_CARRY_SYMBOLS = 10_000

_IDENTIFIER = re.compile(r"[A-Za-z0-9][A-Za-z0-9._:-]{0,127}\Z", re.ASCII)
_SYMBOL = re.compile(r"[A-Za-z0-9][A-Za-z0-9._:/-]{0,63}\Z", re.ASCII)
_SHA256 = re.compile(r"[0-9a-f]{64}\Z", re.ASCII)

_DEFAULT_PROVENANCE = (
    "predeclared deterministic simulation sensitivity; not calibrated from strategy "
    "test outcomes or presented as observed execution quality"
)

type MetadataPairs = tuple[tuple[str, str], ...]
type MoneyPairs = tuple[tuple[str, float], ...]


class FrictionContractError(ValueError):
    """Raised when a friction or logical-latency contract is invalid."""


def _identifier(value: object, *, name: str) -> str:
    if not isinstance(value, str) or _IDENTIFIER.fullmatch(value) is None:
        raise FrictionContractError(
            f"{name} must be a bounded ASCII identifier matching {_IDENTIFIER.pattern!r}"
        )
    return value


def _symbol(value: object, *, name: str = "symbol") -> str:
    if not isinstance(value, str) or _SYMBOL.fullmatch(value) is None:
        raise FrictionContractError(f"{name} must be a non-empty bounded ASCII market symbol")
    return value


def _text(value: object, *, name: str) -> str:
    if (
        not isinstance(value, str)
        or not value
        or value != value.strip()
        or len(value) > MAX_METADATA_TEXT
        or not value.isprintable()
    ):
        raise FrictionContractError(
            f"{name} must be non-empty printable text no longer than {MAX_METADATA_TEXT} characters"
        )
    return value


def _digest(value: object, *, name: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise FrictionContractError(f"{name} must be a lowercase SHA-256 digest")
    return value


def _integer(
    value: object,
    *,
    name: str,
    minimum: int = 0,
    maximum: int,
) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or not minimum <= value <= maximum:
        raise FrictionContractError(f"{name} must be an integer in [{minimum}, {maximum}]")
    return value


def _number(
    value: object,
    *,
    name: str,
    minimum: float,
    maximum: float,
    minimum_inclusive: bool = True,
) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise FrictionContractError(f"{name} must be a finite real number")
    try:
        number = float(value)
    except OverflowError as exc:
        raise FrictionContractError(f"{name} must be a finite real number") from exc
    lower_ok = number >= minimum if minimum_inclusive else number > minimum
    if not math.isfinite(number) or not lower_ok or number > maximum:
        bracket = "[" if minimum_inclusive else "("
        raise FrictionContractError(f"{name} must be finite and in {bracket}{minimum}, {maximum}]")
    return 0.0 if number == 0.0 else number


def _session(value: object, *, name: str) -> date:
    if isinstance(value, datetime):
        raise FrictionContractError(f"{name} must be a date without a time component")
    if isinstance(value, date):
        return value
    if isinstance(value, str):
        try:
            parsed = date.fromisoformat(value)
        except ValueError as exc:
            raise FrictionContractError(f"{name} must be an ISO-8601 date") from exc
        if parsed.isoformat() != value:
            raise FrictionContractError(f"{name} must use canonical YYYY-MM-DD form")
        return parsed
    raise FrictionContractError(f"{name} must be a date or canonical YYYY-MM-DD string")


def _strict_config(
    config: Mapping[str, object] | None,
    *,
    allowed: frozenset[str],
    name: str,
) -> dict[str, object]:
    if config is None:
        return {}
    if not isinstance(config, Mapping):
        raise FrictionContractError(f"{name} must be a mapping")
    if any(not isinstance(key, str) for key in config):
        raise FrictionContractError(f"{name} keys must be strings")
    unknown = set(config) - allowed
    if unknown:
        raise FrictionContractError(f"unknown {name} settings: {sorted(unknown)}")
    return dict(config)


def _metadata_pairs(value: object, *, name: str) -> MetadataPairs:
    if not isinstance(value, tuple) or len(value) > MAX_DECLARATION_ITEMS:
        raise FrictionContractError(
            f"{name} must be a tuple containing at most {MAX_DECLARATION_ITEMS} pairs"
        )
    normalized: list[tuple[str, str]] = []
    for index, item in enumerate(value):
        if not isinstance(item, tuple) or len(item) != 2:
            raise FrictionContractError(f"{name}[{index}] must be a two-item tuple")
        key = _identifier(item[0], name=f"{name}[{index}] key")
        text = _text(item[1], name=f"{name}[{index}] value")
        normalized.append((key, text))
    result = tuple(normalized)
    if result != tuple(sorted(result)) or len({key for key, _ in result}) != len(result):
        raise FrictionContractError(f"{name} must have unique keys in lexical order")
    return result


def _limitations(value: object) -> tuple[str, ...]:
    if not isinstance(value, tuple) or len(value) > MAX_LIMITATIONS:
        raise FrictionContractError(
            f"limitations must be a tuple containing at most {MAX_LIMITATIONS} items"
        )
    normalized = tuple(_text(item, name="limitation") for item in value)
    if len(set(normalized)) != len(normalized):
        raise FrictionContractError("limitations must be unique")
    return normalized


def _canonical_digest(payload: object, *, domain: str) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")
    digest = hashlib.sha256()
    digest.update(domain.encode("ascii"))
    digest.update(b"\x00")
    digest.update(encoded)
    return digest.hexdigest()


def _frozen_calendar(calendar: Iterable[date]) -> tuple[date, ...]:
    if isinstance(calendar, str):
        raise FrictionContractError("calendar must be an iterable of date-only sessions")
    sessions: list[date] = []
    try:
        for index, value in enumerate(calendar):
            if index >= MAX_CALENDAR_SESSIONS:
                raise FrictionContractError(
                    f"calendar exceeds the {MAX_CALENDAR_SESSIONS}-session resource ceiling"
                )
            sessions.append(_session(value, name=f"calendar[{index}]"))
    except TypeError as exc:
        raise FrictionContractError("calendar must be an iterable of date-only sessions") from exc
    result = tuple(sessions)
    if not result:
        raise FrictionContractError("calendar must contain at least one session")
    if any(left >= right for left, right in zip(result, result[1:], strict=False)):
        raise FrictionContractError("calendar sessions must be unique and strictly increasing")
    return result


@dataclass(frozen=True, slots=True)
class ModelDeclaration:
    """Machine-auditable metadata shared by every friction model.

    Bounds are explanatory strings because individual model constructors remain
    the executable source of truth.  Both units and bounds use sorted tuples so
    declarations are immutable, hashable, and serialization-order independent.
    """

    model_id: str
    version: str
    units: MetadataPairs
    calibration_provenance: str
    domain: str
    parameter_bounds: MetadataPairs
    execution_timestamp: str
    failure_behavior: str
    limitations: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "model_id", _identifier(self.model_id, name="model_id"))
        object.__setattr__(self, "version", _identifier(self.version, name="version"))
        object.__setattr__(self, "units", _metadata_pairs(self.units, name="units"))
        object.__setattr__(
            self,
            "calibration_provenance",
            _text(self.calibration_provenance, name="calibration_provenance"),
        )
        object.__setattr__(self, "domain", _text(self.domain, name="domain"))
        object.__setattr__(
            self,
            "parameter_bounds",
            _metadata_pairs(self.parameter_bounds, name="parameter_bounds"),
        )
        object.__setattr__(
            self,
            "execution_timestamp",
            _text(self.execution_timestamp, name="execution_timestamp"),
        )
        object.__setattr__(
            self,
            "failure_behavior",
            _text(self.failure_behavior, name="failure_behavior"),
        )
        object.__setattr__(self, "limitations", _limitations(self.limitations))

    def to_dict(self) -> dict[str, object]:
        """Return a canonical JSON-compatible declaration."""

        return {
            "model_id": self.model_id,
            "version": self.version,
            "units": dict(self.units),
            "calibration_provenance": self.calibration_provenance,
            "domain": self.domain,
            "parameter_bounds": dict(self.parameter_bounds),
            "execution_timestamp": self.execution_timestamp,
            "failure_behavior": self.failure_behavior,
            "limitations": list(self.limitations),
        }

    @property
    def digest(self) -> str:
        """Content identity for this declaration."""

        return _canonical_digest(self.to_dict(), domain="alphaforge.model-declaration.v1")


@dataclass(frozen=True, slots=True)
class LatencySchedule:
    """One bounded causal schedule on a frozen logical trading calendar."""

    origin_session: date
    data_available_session: date
    feature_available_session: date
    signal_available_session: date
    submission_session: date
    fill_session: date
    execution_lag_sessions: int
    model_digest: str

    _CONFIG_KEYS: ClassVar[frozenset[str]] = frozenset(
        {
            "origin_session",
            "data_available_session",
            "feature_available_session",
            "signal_available_session",
            "submission_session",
            "fill_session",
            "execution_lag_sessions",
            "model_digest",
        }
    )

    def __post_init__(self) -> None:
        names = (
            "origin_session",
            "data_available_session",
            "feature_available_session",
            "signal_available_session",
            "submission_session",
            "fill_session",
        )
        sessions = tuple(_session(getattr(self, name), name=name) for name in names)
        for name, session in zip(names, sessions, strict=True):
            object.__setattr__(self, name, session)
        if any(left > right for left, right in zip(sessions, sessions[1:], strict=False)):
            raise FrictionContractError(
                "latency stages must be monotonic: origin <= data <= feature <= signal "
                "<= submission < fill"
            )
        if self.fill_session <= self.submission_session:
            raise FrictionContractError("fill_session must be strictly after submission_session")
        object.__setattr__(
            self,
            "execution_lag_sessions",
            _integer(
                self.execution_lag_sessions,
                name="execution_lag_sessions",
                minimum=1,
                maximum=MAX_STAGE_DELAY_SESSIONS,
            ),
        )
        object.__setattr__(self, "model_digest", _digest(self.model_digest, name="model_digest"))

    @classmethod
    def from_config(
        cls,
        config: Mapping[str, object],
        *,
        calendar: Iterable[date],
    ) -> LatencySchedule:
        """Parse and verify a serialized schedule against its frozen calendar."""

        cfg = _strict_config(config, allowed=cls._CONFIG_KEYS, name="latency schedule")
        missing = cls._CONFIG_KEYS - set(cfg)
        if missing:
            raise FrictionContractError(f"latency schedule missing settings: {sorted(missing)}")
        schedule = cls(
            origin_session=_session(cfg["origin_session"], name="origin_session"),
            data_available_session=_session(
                cfg["data_available_session"], name="data_available_session"
            ),
            feature_available_session=_session(
                cfg["feature_available_session"], name="feature_available_session"
            ),
            signal_available_session=_session(
                cfg["signal_available_session"], name="signal_available_session"
            ),
            submission_session=_session(cfg["submission_session"], name="submission_session"),
            fill_session=_session(cfg["fill_session"], name="fill_session"),
            execution_lag_sessions=_integer(
                cfg["execution_lag_sessions"],
                name="execution_lag_sessions",
                minimum=1,
                maximum=MAX_STAGE_DELAY_SESSIONS,
            ),
            model_digest=_digest(cfg["model_digest"], name="model_digest"),
        )
        frozen = _frozen_calendar(calendar)
        positions = {session: index for index, session in enumerate(frozen)}
        missing_sessions = [
            session.isoformat() for session in schedule.stage_sessions if session not in positions
        ]
        if missing_sessions:
            raise FrictionContractError(
                f"latency schedule contains sessions outside the frozen calendar: {missing_sessions}"
            )
        indices = tuple(positions[session] for session in schedule.stage_sessions)
        if any(left > right for left, right in zip(indices, indices[1:], strict=False)):
            raise FrictionContractError("latency schedule is not monotonic on the frozen calendar")
        submission_to_fill = indices[-1] - indices[-2]
        if submission_to_fill < schedule.execution_lag_sessions:
            raise FrictionContractError(
                "fill stage must preserve the declared execution lag on the frozen calendar"
            )
        return schedule

    @property
    def stage_sessions(self) -> tuple[date, ...]:
        """Return the complete causal stage order, including its origin."""

        return (
            self.origin_session,
            self.data_available_session,
            self.feature_available_session,
            self.signal_available_session,
            self.submission_session,
            self.fill_session,
        )

    def to_dict(self) -> dict[str, object]:
        """Return the canonical serialized schedule."""

        return {
            "origin_session": self.origin_session.isoformat(),
            "data_available_session": self.data_available_session.isoformat(),
            "feature_available_session": self.feature_available_session.isoformat(),
            "signal_available_session": self.signal_available_session.isoformat(),
            "submission_session": self.submission_session.isoformat(),
            "fill_session": self.fill_session.isoformat(),
            "execution_lag_sessions": self.execution_lag_sessions,
            "model_digest": self.model_digest,
        }

    @property
    def digest(self) -> str:
        """Content identity for the resolved schedule."""

        return _canonical_digest(self.to_dict(), domain="alphaforge.latency-schedule.v1")


@dataclass(frozen=True, slots=True)
class LatencyModel:
    """Compose non-negative stage delays in logical trading sessions.

    The origin is the session at which the un-delayed source observation would
    be available.  Every configured stage is cumulative and non-negative.  A
    positive caller-supplied ``execution_lag`` remains the irreducible
    close-decision/future-session boundary; modeled fill delay is added to it.
    """

    data_delay_sessions: int = 0
    feature_delay_sessions: int = 0
    inference_delay_sessions: int = 0
    submission_delay_sessions: int = 0
    fill_delay_sessions: int = 0
    calibration_provenance: str = _DEFAULT_PROVENANCE

    _CONFIG_KEYS: ClassVar[frozenset[str]] = frozenset(
        {
            "data_delay_sessions",
            "feature_delay_sessions",
            "inference_delay_sessions",
            "submission_delay_sessions",
            "fill_delay_sessions",
            "calibration_provenance",
        }
    )

    def __post_init__(self) -> None:
        for name in (
            "data_delay_sessions",
            "feature_delay_sessions",
            "inference_delay_sessions",
            "submission_delay_sessions",
        ):
            object.__setattr__(
                self,
                name,
                _integer(
                    getattr(self, name),
                    name=name,
                    maximum=MAX_STAGE_DELAY_SESSIONS,
                ),
            )
        object.__setattr__(
            self,
            "fill_delay_sessions",
            _integer(
                self.fill_delay_sessions,
                name="fill_delay_sessions",
                maximum=MAX_STAGE_DELAY_SESSIONS,
            ),
        )
        if self.stage_delay_sessions >= MAX_TOTAL_DELAY_SESSIONS:
            raise FrictionContractError(
                f"total logical latency must be less than {MAX_TOTAL_DELAY_SESSIONS} sessions"
            )
        object.__setattr__(
            self,
            "calibration_provenance",
            _text(self.calibration_provenance, name="calibration_provenance"),
        )

    @classmethod
    def from_config(cls, config: Mapping[str, object] | None) -> LatencyModel:
        """Construct a strict model, rejecting unknown or coerced values."""

        cfg = _strict_config(config, allowed=cls._CONFIG_KEYS, name="latency")
        return cls(
            data_delay_sessions=_integer(
                cfg.get("data_delay_sessions", 0),
                name="data_delay_sessions",
                maximum=MAX_STAGE_DELAY_SESSIONS,
            ),
            feature_delay_sessions=_integer(
                cfg.get("feature_delay_sessions", 0),
                name="feature_delay_sessions",
                maximum=MAX_STAGE_DELAY_SESSIONS,
            ),
            inference_delay_sessions=_integer(
                cfg.get("inference_delay_sessions", 0),
                name="inference_delay_sessions",
                maximum=MAX_STAGE_DELAY_SESSIONS,
            ),
            submission_delay_sessions=_integer(
                cfg.get("submission_delay_sessions", 0),
                name="submission_delay_sessions",
                maximum=MAX_STAGE_DELAY_SESSIONS,
            ),
            fill_delay_sessions=_integer(
                cfg.get("fill_delay_sessions", 0),
                name="fill_delay_sessions",
                maximum=MAX_STAGE_DELAY_SESSIONS,
            ),
            calibration_provenance=_text(
                cfg.get("calibration_provenance", _DEFAULT_PROVENANCE),
                name="calibration_provenance",
            ),
        )

    def cumulative_offsets(self, *, execution_lag: int) -> tuple[int, int, int, int, int]:
        """Return cumulative stage offsets including the irreducible execution lag."""

        lag = _integer(
            execution_lag,
            name="execution_lag",
            minimum=1,
            maximum=MAX_STAGE_DELAY_SESSIONS,
        )
        data = self.data_delay_sessions
        feature = data + self.feature_delay_sessions
        signal = feature + self.inference_delay_sessions
        submission = signal + self.submission_delay_sessions
        fill = submission + lag + self.fill_delay_sessions
        if fill > MAX_TOTAL_DELAY_SESSIONS:
            raise FrictionContractError(
                f"resolved logical latency exceeds {MAX_TOTAL_DELAY_SESSIONS} sessions"
            )
        return data, feature, signal, submission, fill

    @property
    def stage_delay_sessions(self) -> int:
        """Total configurable stage delay, excluding the legacy execution lag."""

        return sum(
            (
                self.data_delay_sessions,
                self.feature_delay_sessions,
                self.inference_delay_sessions,
                self.submission_delay_sessions,
                self.fill_delay_sessions,
            )
        )

    @property
    def declaration(self) -> ModelDeclaration:
        """Return units, provenance, bounds, timing, and failure semantics."""

        return ModelDeclaration(
            model_id="logical-session-latency",
            version="1.0.0",
            units=(
                ("data_delay_sessions", "logical trading sessions"),
                ("feature_delay_sessions", "logical trading sessions"),
                ("fill_delay_sessions", "logical trading sessions"),
                ("inference_delay_sessions", "logical trading sessions"),
                ("submission_delay_sessions", "logical trading sessions"),
            ),
            calibration_provenance=self.calibration_provenance,
            domain="daily-bar simulation over a caller-supplied frozen trading calendar",
            parameter_bounds=(
                ("execution_lag", f"caller-supplied integer in [1, {MAX_STAGE_DELAY_SESSIONS}]"),
                ("fill_delay_sessions", f"integer in [0, {MAX_STAGE_DELAY_SESSIONS}]"),
                (
                    "non_fill_stage_delays",
                    f"integer in [0, {MAX_STAGE_DELAY_SESSIONS}] per stage",
                ),
                ("resolved_delay_sessions", f"integer in [1, {MAX_TOTAL_DELAY_SESSIONS}]"),
            ),
            execution_timestamp=(
                "fill_session on the frozen logical calendar; no wall-clock timestamp is inferred"
            ),
            failure_behavior=(
                "reject malformed stages, invalid calendars, missing origins, and schedules beyond "
                "the frozen calendar; never clamp or move a stage earlier"
            ),
            limitations=(
                "Does not model exchange, network, broker, auction, queue, or wall-clock latency.",
                "A daily bar cannot establish an intraday fill timestamp.",
            ),
        )

    @property
    def configuration_digest(self) -> str:
        """Content identity binding declaration and configured delays."""

        payload = {
            "declaration_digest": self.declaration.digest,
            "data_delay_sessions": self.data_delay_sessions,
            "feature_delay_sessions": self.feature_delay_sessions,
            "inference_delay_sessions": self.inference_delay_sessions,
            "submission_delay_sessions": self.submission_delay_sessions,
            "fill_delay_sessions": self.fill_delay_sessions,
        }
        return _canonical_digest(payload, domain="alphaforge.latency-model.v1")

    def schedule(
        self,
        origin_session: date,
        *,
        calendar: Iterable[date],
        execution_lag: int,
    ) -> LatencySchedule:
        """Resolve stages while preserving the caller's positive next-open lag.

        The fill offset is ``submission readiness + execution_lag + modeled
        fill delay``.  No modeled stage can reduce or replace the legacy lag.
        Resolution fails if any resulting stage exceeds the frozen calendar.
        """

        frozen = _frozen_calendar(calendar)
        origin = _session(origin_session, name="origin_session")
        positions = {session: index for index, session in enumerate(frozen)}
        if origin not in positions:
            raise FrictionContractError("origin_session is outside the frozen calendar")
        origin_index = positions[origin]
        lag = _integer(
            execution_lag,
            name="execution_lag",
            minimum=1,
            maximum=MAX_STAGE_DELAY_SESSIONS,
        )
        offsets = self.cumulative_offsets(execution_lag=lag)
        if origin_index + offsets[-1] >= len(frozen):
            raise FrictionContractError("latency schedule exceeds the frozen calendar")
        stage_sessions = tuple(frozen[origin_index + offset] for offset in offsets)
        return LatencySchedule(
            origin_session=origin,
            data_available_session=stage_sessions[0],
            feature_available_session=stage_sessions[1],
            signal_available_session=stage_sessions[2],
            submission_session=stage_sessions[3],
            fill_session=stage_sessions[4],
            execution_lag_sessions=lag,
            model_digest=self.configuration_digest,
        )


def _money_pairs(value: object, *, name: str, positive: bool) -> MoneyPairs:
    if not isinstance(value, tuple) or len(value) > MAX_CARRY_SYMBOLS:
        raise FrictionContractError(f"{name} must be a bounded tuple of symbol/value pairs")
    normalized: list[tuple[str, float]] = []
    for index, item in enumerate(value):
        if not isinstance(item, tuple) or len(item) != 2:
            raise FrictionContractError(f"{name}[{index}] must be a two-item tuple")
        symbol = _symbol(item[0], name=f"{name}[{index}] symbol")
        amount = _number(
            item[1],
            name=f"{name}[{index}] amount",
            minimum=0.0,
            maximum=MAX_MONEY_USD,
            minimum_inclusive=not positive,
        )
        normalized.append((symbol, amount))
    result = tuple(normalized)
    if result != tuple(sorted(result)) or len({symbol for symbol, _ in result}) != len(result):
        raise FrictionContractError(f"{name} must use unique symbols in lexical order")
    return result


@dataclass(frozen=True, slots=True)
class CarryAccrual:
    """Auditable USD financing and short-borrow charges for one logical session."""

    session: date
    financing_basis_usd: float
    financing_charge_usd: float
    short_market_values_usd: MoneyPairs
    borrow_charges_usd: MoneyPairs
    total_borrow_charge_usd: float
    total_charge_usd: float
    model_digest: str
    calibration_provenance: str
    currency: Literal["USD"] = "USD"
    borrow_availability: Literal["not_modeled"] = "not_modeled"
    locate_status: Literal["not_modeled"] = "not_modeled"

    def __post_init__(self) -> None:
        object.__setattr__(self, "session", _session(self.session, name="session"))
        object.__setattr__(
            self,
            "financing_basis_usd",
            _number(
                self.financing_basis_usd,
                name="financing_basis_usd",
                minimum=0.0,
                maximum=MAX_MONEY_USD,
            ),
        )
        object.__setattr__(
            self,
            "financing_charge_usd",
            _number(
                self.financing_charge_usd,
                name="financing_charge_usd",
                minimum=0.0,
                maximum=MAX_MONEY_USD,
            ),
        )
        market_values = _money_pairs(
            self.short_market_values_usd,
            name="short_market_values_usd",
            positive=True,
        )
        borrow_charges = _money_pairs(
            self.borrow_charges_usd,
            name="borrow_charges_usd",
            positive=False,
        )
        if tuple(symbol for symbol, _ in market_values) != tuple(
            symbol for symbol, _ in borrow_charges
        ):
            raise FrictionContractError(
                "short market values and borrow charges must cover the same sorted symbols"
            )
        object.__setattr__(self, "short_market_values_usd", market_values)
        object.__setattr__(self, "borrow_charges_usd", borrow_charges)
        total_borrow = _number(
            self.total_borrow_charge_usd,
            name="total_borrow_charge_usd",
            minimum=0.0,
            maximum=MAX_MONEY_USD,
        )
        total_charge = _number(
            self.total_charge_usd,
            name="total_charge_usd",
            minimum=0.0,
            maximum=MAX_MONEY_USD,
        )
        expected_borrow = math.fsum(amount for _, amount in borrow_charges)
        expected_total = math.fsum((self.financing_charge_usd, expected_borrow))
        if total_borrow != expected_borrow:
            raise FrictionContractError("total_borrow_charge_usd must equal exact math.fsum")
        if total_charge != expected_total:
            raise FrictionContractError("total_charge_usd must equal exact math.fsum")
        object.__setattr__(self, "total_borrow_charge_usd", total_borrow)
        object.__setattr__(self, "total_charge_usd", total_charge)
        object.__setattr__(self, "model_digest", _digest(self.model_digest, name="model_digest"))
        object.__setattr__(
            self,
            "calibration_provenance",
            _text(self.calibration_provenance, name="calibration_provenance"),
        )
        if self.currency != "USD":
            raise FrictionContractError("carry accrual currency must be USD")
        if self.borrow_availability != "not_modeled" or self.locate_status != "not_modeled":
            raise FrictionContractError(
                "carry accrual cannot claim borrow availability or locate status"
            )

    def to_dict(self) -> dict[str, object]:
        """Return a canonical JSON-compatible accrual record."""

        return {
            "session": self.session.isoformat(),
            "financing_basis_usd": self.financing_basis_usd.hex(),
            "financing_charge_usd": self.financing_charge_usd.hex(),
            "short_market_values_usd": [
                [symbol, amount.hex()] for symbol, amount in self.short_market_values_usd
            ],
            "borrow_charges_usd": [
                [symbol, amount.hex()] for symbol, amount in self.borrow_charges_usd
            ],
            "total_borrow_charge_usd": self.total_borrow_charge_usd.hex(),
            "total_charge_usd": self.total_charge_usd.hex(),
            "model_digest": self.model_digest,
            "calibration_provenance": self.calibration_provenance,
            "currency": self.currency,
            "borrow_availability": self.borrow_availability,
            "locate_status": self.locate_status,
        }

    @property
    def digest(self) -> str:
        """Content identity for this accrual."""

        return _canonical_digest(self.to_dict(), domain="alphaforge.carry-accrual.v1")


@dataclass(frozen=True, slots=True)
class CarryCostModel:
    """Per-session USD financing and borrow-cost sensitivity.

    Financing applies only to negative cash.  Borrow applies to the absolute
    marked value of each short position, ordered lexically by symbol.  Rates
    are annual basis-point assumptions divided by ``sessions_per_year``.
    Availability, locates, recalls, forced buy-ins, and security-specific rates
    remain outside this calculator.
    """

    cash_financing_bps_annual: float = 0.0
    short_borrow_bps_annual: float = 0.0
    sessions_per_year: int = 252
    calibration_provenance: str = _DEFAULT_PROVENANCE

    _CONFIG_KEYS: ClassVar[frozenset[str]] = frozenset(
        {
            "cash_financing_bps_annual",
            "short_borrow_bps_annual",
            "sessions_per_year",
            "calibration_provenance",
        }
    )

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "cash_financing_bps_annual",
            _number(
                self.cash_financing_bps_annual,
                name="cash_financing_bps_annual",
                minimum=0.0,
                maximum=MAX_RATE_BPS_ANNUAL,
            ),
        )
        object.__setattr__(
            self,
            "short_borrow_bps_annual",
            _number(
                self.short_borrow_bps_annual,
                name="short_borrow_bps_annual",
                minimum=0.0,
                maximum=MAX_RATE_BPS_ANNUAL,
            ),
        )
        object.__setattr__(
            self,
            "sessions_per_year",
            _integer(
                self.sessions_per_year,
                name="sessions_per_year",
                minimum=1,
                maximum=MAX_SESSIONS_PER_YEAR,
            ),
        )
        object.__setattr__(
            self,
            "calibration_provenance",
            _text(self.calibration_provenance, name="calibration_provenance"),
        )

    @classmethod
    def from_config(cls, config: Mapping[str, object] | None) -> CarryCostModel:
        """Construct a strict carry model without numeric coercion."""

        cfg = _strict_config(config, allowed=cls._CONFIG_KEYS, name="carry cost")
        return cls(
            cash_financing_bps_annual=_number(
                cfg.get("cash_financing_bps_annual", 0.0),
                name="cash_financing_bps_annual",
                minimum=0.0,
                maximum=MAX_RATE_BPS_ANNUAL,
            ),
            short_borrow_bps_annual=_number(
                cfg.get("short_borrow_bps_annual", 0.0),
                name="short_borrow_bps_annual",
                minimum=0.0,
                maximum=MAX_RATE_BPS_ANNUAL,
            ),
            sessions_per_year=_integer(
                cfg.get("sessions_per_year", 252),
                name="sessions_per_year",
                minimum=1,
                maximum=MAX_SESSIONS_PER_YEAR,
            ),
            calibration_provenance=_text(
                cfg.get("calibration_provenance", _DEFAULT_PROVENANCE),
                name="calibration_provenance",
            ),
        )

    @property
    def declaration(self) -> ModelDeclaration:
        """Return the explicit carry-cost contract and non-goals."""

        return ModelDeclaration(
            model_id="usd-session-carry-cost",
            version="1.0.0",
            units=(
                ("cash", "USD"),
                ("positions", "shares"),
                ("prices", "USD per share"),
                ("rates", "basis points per year"),
                ("result", "USD per logical trading session"),
            ),
            calibration_provenance=self.calibration_provenance,
            domain="single-currency USD daily-bar simulation",
            parameter_bounds=(
                ("amounts", f"finite USD amounts in [0, {MAX_MONEY_USD:.1e}]"),
                ("annual_rates", f"finite basis points in [0, {MAX_RATE_BPS_ANNUAL:g}]"),
                ("prices", f"finite USD prices in (0, {MAX_PRICE_USD:.1e}]"),
                ("sessions_per_year", f"integer in [1, {MAX_SESSIONS_PER_YEAR}]"),
                ("share_quantity", f"finite absolute shares in [0, {MAX_QUANTITY_SHARES:.1e}]"),
                ("symbols", f"at most {MAX_CARRY_SYMBOLS} positions and prices"),
            ),
            execution_timestamp="charge phase after DAY cancellation and before the close mark",
            failure_behavior=(
                "reject missing/non-positive short marks, malformed inputs, non-finite arithmetic, "
                "and resource-bound overflow; never invent a price or rate"
            ),
            limitations=(
                "Borrow availability, locates, recalls, restrictions, and forced buy-ins are not modeled.",
                "Rates are supplied sensitivities, not evidence of executable financing or borrow terms.",
                "Only USD accounting is supported; no FX conversion is inferred.",
            ),
        )

    @property
    def configuration_digest(self) -> str:
        """Content identity binding rates, basis, and declared provenance."""

        payload = {
            "declaration_digest": self.declaration.digest,
            "cash_financing_bps_annual": self.cash_financing_bps_annual.hex(),
            "short_borrow_bps_annual": self.short_borrow_bps_annual.hex(),
            "sessions_per_year": self.sessions_per_year,
        }
        return _canonical_digest(payload, domain="alphaforge.carry-cost-model.v1")

    def accrue(
        self,
        *,
        session: date,
        cash_usd: float,
        positions: Mapping[str, float],
        prices_usd: Mapping[str, float],
    ) -> CarryAccrual:
        """Calculate one session of USD carry from supplied causal marks.

        A price is required for every short symbol.  All supplied prices and
        positions are validated even if a value would otherwise be ignored.
        This fail-closed boundary prevents malformed extra inputs from being
        silently accepted.
        """

        accrued_session = _session(session, name="session")
        cash = _number(
            cash_usd,
            name="cash_usd",
            minimum=-MAX_MONEY_USD,
            maximum=MAX_MONEY_USD,
        )
        if not isinstance(positions, Mapping):
            raise FrictionContractError("positions must be a symbol-to-shares mapping")
        if not isinstance(prices_usd, Mapping):
            raise FrictionContractError("prices_usd must be a symbol-to-price mapping")
        if len(positions) > MAX_CARRY_SYMBOLS:
            raise FrictionContractError(
                f"positions exceeds the {MAX_CARRY_SYMBOLS}-symbol resource ceiling"
            )
        if len(prices_usd) > MAX_CARRY_SYMBOLS:
            raise FrictionContractError(
                f"prices_usd exceeds the {MAX_CARRY_SYMBOLS}-symbol resource ceiling"
            )

        normalized_positions: dict[str, float] = {}
        for index, (raw_symbol, raw_quantity) in enumerate(positions.items()):
            if index >= MAX_CARRY_SYMBOLS:
                raise FrictionContractError(
                    f"positions exceeds the {MAX_CARRY_SYMBOLS}-symbol resource ceiling"
                )
            symbol = _symbol(raw_symbol)
            if symbol in normalized_positions:
                raise FrictionContractError(f"duplicate normalized position symbol: {symbol}")
            normalized_positions[symbol] = _number(
                raw_quantity,
                name=f"position[{symbol}]",
                minimum=-MAX_QUANTITY_SHARES,
                maximum=MAX_QUANTITY_SHARES,
            )

        normalized_prices: dict[str, float] = {}
        for index, (raw_symbol, raw_price) in enumerate(prices_usd.items()):
            if index >= MAX_CARRY_SYMBOLS:
                raise FrictionContractError(
                    f"prices_usd exceeds the {MAX_CARRY_SYMBOLS}-symbol resource ceiling"
                )
            symbol = _symbol(raw_symbol, name="price symbol")
            if symbol in normalized_prices:
                raise FrictionContractError(f"duplicate normalized price symbol: {symbol}")
            normalized_prices[symbol] = _number(
                raw_price,
                name=f"price[{symbol}]",
                minimum=0.0,
                maximum=MAX_PRICE_USD,
                minimum_inclusive=False,
            )

        financing_basis = max(-cash, 0.0)
        financing_rate = self.cash_financing_bps_annual / 10_000.0 / float(self.sessions_per_year)
        financing_charge = financing_basis * financing_rate
        if not math.isfinite(financing_charge) or financing_charge > MAX_MONEY_USD:
            raise FrictionContractError("financing charge overflowed the USD resource ceiling")

        borrow_rate = self.short_borrow_bps_annual / 10_000.0 / float(self.sessions_per_year)
        market_values: list[tuple[str, float]] = []
        borrow_charges: list[tuple[str, float]] = []
        for symbol in sorted(normalized_positions):
            quantity = normalized_positions[symbol]
            if quantity >= 0.0:
                continue
            if symbol not in normalized_prices:
                raise FrictionContractError(f"missing positive USD price for short symbol {symbol}")
            market_value = abs(quantity) * normalized_prices[symbol]
            if not math.isfinite(market_value) or market_value > MAX_MONEY_USD:
                raise FrictionContractError(
                    f"short market value for {symbol} overflowed the USD resource ceiling"
                )
            charge = market_value * borrow_rate
            if not math.isfinite(charge) or charge > MAX_MONEY_USD:
                raise FrictionContractError(
                    f"borrow charge for {symbol} overflowed the USD resource ceiling"
                )
            market_values.append((symbol, market_value))
            borrow_charges.append((symbol, charge))

        total_borrow = math.fsum(amount for _, amount in borrow_charges)
        total_charge = math.fsum((financing_charge, total_borrow))
        if not math.isfinite(total_borrow) or not math.isfinite(total_charge):
            raise FrictionContractError("carry charge aggregation overflowed")
        if total_borrow > MAX_MONEY_USD or total_charge > MAX_MONEY_USD:
            raise FrictionContractError("carry charge exceeds the USD resource ceiling")
        return CarryAccrual(
            session=accrued_session,
            financing_basis_usd=financing_basis,
            financing_charge_usd=financing_charge,
            short_market_values_usd=tuple(market_values),
            borrow_charges_usd=tuple(borrow_charges),
            total_borrow_charge_usd=total_borrow,
            total_charge_usd=total_charge,
            model_digest=self.configuration_digest,
            calibration_provenance=self.calibration_provenance,
        )


@dataclass(frozen=True, slots=True)
class ExecutionStressProfile:
    """One predeclared adverse execution sensitivity.

    Multipliers are deliberately directional: costs, spreads, and capacity
    demand cannot fall below baseline; available liquidity and participation
    cannot exceed baseline; signal delay cannot be negative.  This makes a
    profile structurally incapable of presenting a favorable perturbation as
    a stress.
    """

    name: str
    cost_multiplier: float = 1.0
    spread_multiplier: float = 1.0
    liquidity_multiplier: float = 1.0
    signal_delay_sessions: int = 0
    participation_limit_multiplier: float = 1.0
    capacity_multiplier: float = 1.0
    calibration_provenance: str = _DEFAULT_PROVENANCE

    _CONFIG_KEYS: ClassVar[frozenset[str]] = frozenset(
        {
            "name",
            "cost_multiplier",
            "spread_multiplier",
            "liquidity_multiplier",
            "signal_delay_sessions",
            "participation_limit_multiplier",
            "capacity_multiplier",
            "calibration_provenance",
        }
    )

    def __post_init__(self) -> None:
        object.__setattr__(self, "name", _identifier(self.name, name="stress profile name"))
        for name in ("cost_multiplier", "spread_multiplier", "capacity_multiplier"):
            object.__setattr__(
                self,
                name,
                _number(
                    getattr(self, name),
                    name=name,
                    minimum=1.0,
                    maximum=MAX_MULTIPLIER,
                ),
            )
        for name in ("liquidity_multiplier", "participation_limit_multiplier"):
            object.__setattr__(
                self,
                name,
                _number(
                    getattr(self, name),
                    name=name,
                    minimum=0.0,
                    maximum=1.0,
                    minimum_inclusive=False,
                ),
            )
        object.__setattr__(
            self,
            "signal_delay_sessions",
            _integer(
                self.signal_delay_sessions,
                name="signal_delay_sessions",
                maximum=MAX_STAGE_DELAY_SESSIONS,
            ),
        )
        object.__setattr__(
            self,
            "calibration_provenance",
            _text(self.calibration_provenance, name="calibration_provenance"),
        )
        reserved = {
            "baseline": (1.0, 1.0, 1.0, 0, 1.0, 1.0),
            "doubled_costs": (2.0, 1.0, 1.0, 0, 1.0, 1.0),
            "tripled_costs": (3.0, 1.0, 1.0, 0, 1.0, 1.0),
            "adverse_spread": (1.0, 3.0, 1.0, 0, 1.0, 1.0),
            "reduced_liquidity": (1.0, 1.0, 0.5, 0, 1.0, 1.0),
            "delayed_signals": (1.0, 1.0, 1.0, 1, 1.0, 1.0),
            "partial_fill_pressure": (1.0, 1.0, 1.0, 0, 0.5, 1.0),
            "capacity_scaling": (1.0, 1.0, 1.0, 0, 1.0, 5.0),
        }
        observed = (
            self.cost_multiplier,
            self.spread_multiplier,
            self.liquidity_multiplier,
            self.signal_delay_sessions,
            self.participation_limit_multiplier,
            self.capacity_multiplier,
        )
        if self.name in reserved and observed != reserved[self.name]:
            raise FrictionContractError(
                f"reserved stress profile {self.name!r} must use its canonical settings"
            )

    @classmethod
    def from_config(cls, config: Mapping[str, object]) -> ExecutionStressProfile:
        """Parse one strict stress profile without implicit numeric coercion."""

        cfg = _strict_config(config, allowed=cls._CONFIG_KEYS, name="execution stress profile")
        if "name" not in cfg:
            raise FrictionContractError("execution stress profile requires name")
        return cls(
            name=_identifier(cfg["name"], name="stress profile name"),
            cost_multiplier=_number(
                cfg.get("cost_multiplier", 1.0),
                name="cost_multiplier",
                minimum=1.0,
                maximum=MAX_MULTIPLIER,
            ),
            spread_multiplier=_number(
                cfg.get("spread_multiplier", 1.0),
                name="spread_multiplier",
                minimum=1.0,
                maximum=MAX_MULTIPLIER,
            ),
            liquidity_multiplier=_number(
                cfg.get("liquidity_multiplier", 1.0),
                name="liquidity_multiplier",
                minimum=0.0,
                maximum=1.0,
                minimum_inclusive=False,
            ),
            signal_delay_sessions=_integer(
                cfg.get("signal_delay_sessions", 0),
                name="signal_delay_sessions",
                maximum=MAX_STAGE_DELAY_SESSIONS,
            ),
            participation_limit_multiplier=_number(
                cfg.get("participation_limit_multiplier", 1.0),
                name="participation_limit_multiplier",
                minimum=0.0,
                maximum=1.0,
                minimum_inclusive=False,
            ),
            capacity_multiplier=_number(
                cfg.get("capacity_multiplier", 1.0),
                name="capacity_multiplier",
                minimum=1.0,
                maximum=MAX_MULTIPLIER,
            ),
            calibration_provenance=_text(
                cfg.get("calibration_provenance", _DEFAULT_PROVENANCE),
                name="calibration_provenance",
            ),
        )

    @property
    def effective_spread_multiplier(self) -> float:
        """Return the product of all-cost and spread-specific stresses."""

        result = self.cost_multiplier * self.spread_multiplier
        if not math.isfinite(result) or result > MAX_MULTIPLIER * MAX_MULTIPLIER:
            raise FrictionContractError("effective spread multiplier overflowed")
        return result

    @property
    def declaration(self) -> ModelDeclaration:
        """Return the frozen stress-grid semantics and interpretation limits."""

        return ModelDeclaration(
            model_id="execution-stress-profile",
            version="1.0.0",
            units=(
                ("capacity_multiplier", "dimensionless demand multiplier"),
                ("cost_multiplier", "dimensionless adverse multiplier"),
                ("liquidity_multiplier", "dimensionless available-liquidity multiplier"),
                (
                    "participation_limit_multiplier",
                    "dimensionless participation-cap multiplier",
                ),
                ("signal_delay_sessions", "logical trading sessions"),
                ("spread_multiplier", "dimensionless adverse multiplier"),
            ),
            calibration_provenance=self.calibration_provenance,
            domain="deterministic daily-bar execution sensitivity analysis",
            parameter_bounds=(
                ("adverse_multipliers", f"finite values in [1, {MAX_MULTIPLIER:g}]"),
                ("constraint_multipliers", "finite values in (0, 1]"),
                (
                    "signal_delay_sessions",
                    f"integer in [0, {MAX_STAGE_DELAY_SESSIONS}]",
                ),
            ),
            execution_timestamp="resolved by the associated logical-session latency model",
            failure_behavior=(
                "reject favorable, non-finite, coerced, unknown, or resource-unbounded settings"
            ),
            limitations=(
                "Profiles are sensitivities, not forecasts of realized transaction costs or capacity.",
                "A partial-fill pressure profile does not assert historical fill probability.",
            ),
        )

    def to_dict(self) -> dict[str, object]:
        """Return canonical profile values with exact float encodings."""

        return {
            "name": self.name,
            "cost_multiplier": self.cost_multiplier.hex(),
            "spread_multiplier": self.spread_multiplier.hex(),
            "liquidity_multiplier": self.liquidity_multiplier.hex(),
            "signal_delay_sessions": self.signal_delay_sessions,
            "participation_limit_multiplier": self.participation_limit_multiplier.hex(),
            "capacity_multiplier": self.capacity_multiplier.hex(),
            "calibration_provenance": self.calibration_provenance,
            "declaration_digest": self.declaration.digest,
        }

    @property
    def digest(self) -> str:
        """Stable identity for this named, predeclared profile."""

        return _canonical_digest(self.to_dict(), domain="alphaforge.execution-stress-profile.v1")


STANDARD_STRESS_PROFILE_ORDER = (
    "baseline",
    "doubled_costs",
    "tripled_costs",
    "adverse_spread",
    "reduced_liquidity",
    "delayed_signals",
    "partial_fill_pressure",
    "capacity_scaling",
)


def standard_stress_profiles() -> tuple[ExecutionStressProfile, ...]:
    """Return the canonical, deterministic MR4 execution stress grid."""

    profiles = (
        ExecutionStressProfile(name="baseline"),
        ExecutionStressProfile(name="doubled_costs", cost_multiplier=2.0),
        ExecutionStressProfile(name="tripled_costs", cost_multiplier=3.0),
        ExecutionStressProfile(name="adverse_spread", spread_multiplier=3.0),
        ExecutionStressProfile(name="reduced_liquidity", liquidity_multiplier=0.5),
        ExecutionStressProfile(name="delayed_signals", signal_delay_sessions=1),
        ExecutionStressProfile(
            name="partial_fill_pressure",
            participation_limit_multiplier=0.5,
        ),
        ExecutionStressProfile(name="capacity_scaling", capacity_multiplier=5.0),
    )
    if tuple(profile.name for profile in profiles) != STANDARD_STRESS_PROFILE_ORDER:
        raise AssertionError("standard stress profile order is not canonical")
    return profiles


__all__ = [
    "CarryAccrual",
    "CarryCostModel",
    "ExecutionStressProfile",
    "FrictionContractError",
    "LatencyModel",
    "LatencySchedule",
    "ModelDeclaration",
    "STANDARD_STRESS_PROFILE_ORDER",
    "standard_stress_profiles",
]
