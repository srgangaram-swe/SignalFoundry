"""Point-in-time universe membership and ablations (SF-S4-MR7).

Universe composition is where survivorship bias enters a backtest, and it enters
quietly. A membership table without effective and expiry dates silently asserts
that today's constituents were always the constituents — which deletes every
delisting, every index removal, and every company that failed. The resulting
backtest earns returns on securities it could not have held.

:class:`PointInTimeUniverse` therefore stores membership as dated intervals and
resolves it *as of* a session, using only rows whose effective date precedes that
session. A symbol that leaves the universe stops being investable from that
session forward, and one that had not yet joined is not investable before it.

The ablations answer a different question and are labelled accordingly. Removing
the top contributors is **inherently hindsight-based** — you can only know which
names won after the fact — so it is a *fragility probe*, not a performance
estimate. It answers "how much of this result rested on a handful of names?", and
:class:`AblationResult` says so in its own record rather than leaving the reader
to infer it. Reporting a winners-removed return as an achievable return would be
a straightforwardly false claim.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import date
from typing import Any, Final

import numpy as np

#: Refusal thresholds, not tuning knobs.
MAX_SYMBOLS: Final = 5_000
MAX_MEMBERSHIP_ROWS: Final = 200_000
MAX_SECTORS: Final = 128
MAX_NAME_CHARS: Final = 64


class UniverseContractError(ValueError):
    """Raised when a membership record or ablation request is unusable."""


def _identifier(value: object, *, field_name: str) -> str:
    if not isinstance(value, str):
        raise UniverseContractError(f"{field_name} must be a string, got {type(value).__name__}")
    text = value.strip()
    if not text or text != value:
        raise UniverseContractError(f"{field_name} must be non-empty and free of padding")
    if len(text) > MAX_NAME_CHARS:
        raise UniverseContractError(f"{field_name} exceeds {MAX_NAME_CHARS} characters")
    if not text.isascii() or not all(part.isalnum() or part in "._-" for part in text):
        raise UniverseContractError(f"{field_name} must be ASCII alphanumeric with . _ - only")
    return text


def _session(value: object, *, field_name: str) -> date:
    if isinstance(value, date) and not hasattr(value, "hour"):
        return value
    raise UniverseContractError(f"{field_name} must be a datetime.date")


def _digest(payload: Any) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


@dataclass(frozen=True, slots=True)
class MembershipRecord:
    """One symbol's dated membership in the investable universe.

    Attributes:
        effective_session: First session the symbol is investable.
        expiry_session: First session it is **no longer** investable, exclusive.
            ``None`` means still a member at the end of the record. Exclusive so
            a delisting date and the next listing's effective date can coincide
            without overlapping.
        delisted: Whether the exit was a delisting rather than an index removal.
            Recorded separately because the two have different consequences: a
            delisted position must be liquidated at whatever the terminal price
            was, while an index removal leaves a still-tradeable security.
    """

    symbol: str
    effective_session: date
    expiry_session: date | None = None
    sector: str = "unclassified"
    delisted: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(self, "symbol", _identifier(self.symbol, field_name="symbol"))
        object.__setattr__(self, "sector", _identifier(self.sector, field_name="sector"))
        object.__setattr__(
            self,
            "effective_session",
            _session(self.effective_session, field_name="effective_session"),
        )
        if self.expiry_session is not None:
            object.__setattr__(
                self, "expiry_session", _session(self.expiry_session, field_name="expiry_session")
            )
            if self.expiry_session <= self.effective_session:
                raise UniverseContractError(
                    f"{self.symbol}: expiry_session must follow effective_session"
                )
        if not isinstance(self.delisted, bool):
            raise UniverseContractError("delisted must be a bool")
        if self.delisted and self.expiry_session is None:
            raise UniverseContractError(
                f"{self.symbol}: a delisted symbol must record the session it left"
            )

    def active(self, session: date) -> bool:
        """Whether the symbol is investable on ``session``."""
        if session < self.effective_session:
            return False
        return self.expiry_session is None or session < self.expiry_session

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-friendly record."""
        return {
            "symbol": self.symbol,
            "effective_session": self.effective_session.isoformat(),
            "expiry_session": (
                None if self.expiry_session is None else self.expiry_session.isoformat()
            ),
            "sector": self.sector,
            "delisted": self.delisted,
        }


@dataclass(frozen=True)
class PointInTimeUniverse:
    """Dated membership resolved as of a session, never in hindsight.

    Raises:
        UniverseContractError: On malformed, oversized, or overlapping records.
    """

    name: str
    records: tuple[MembershipRecord, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "name", _identifier(self.name, field_name="universe name"))
        records = tuple(self.records)
        if not records:
            raise UniverseContractError("a universe must contain at least one membership record")
        if len(records) > MAX_MEMBERSHIP_ROWS:
            raise UniverseContractError(
                f"universe exceeds the {MAX_MEMBERSHIP_ROWS}-record ceiling"
            )
        symbols = {record.symbol for record in records}
        if len(symbols) > MAX_SYMBOLS:
            raise UniverseContractError(f"universe exceeds the {MAX_SYMBOLS}-symbol ceiling")
        sectors = {record.sector for record in records}
        if len(sectors) > MAX_SECTORS:
            raise UniverseContractError(f"universe exceeds the {MAX_SECTORS}-sector ceiling")

        # Two membership rows for one symbol may not overlap: a symbol cannot be
        # a member twice at once, and permitting it would double-count the name.
        by_symbol: dict[str, list[MembershipRecord]] = {}
        for record in records:
            by_symbol.setdefault(record.symbol, []).append(record)
        for symbol, rows in by_symbol.items():
            ordered = sorted(rows, key=lambda item: item.effective_session)
            for earlier, later in zip(ordered, ordered[1:], strict=False):
                if (
                    earlier.expiry_session is None
                    or later.effective_session < earlier.expiry_session
                ):
                    raise UniverseContractError(
                        f"{symbol}: membership intervals overlap; a symbol cannot be a "
                        "member twice at the same session"
                    )
        object.__setattr__(
            self,
            "records",
            tuple(sorted(records, key=lambda item: (item.symbol, item.effective_session))),
        )

    @property
    def identity(self) -> str:
        """Content identity of the whole membership table."""
        return _digest(
            {"name": self.name, "records": [record.to_dict() for record in self.records]}
        )

    def members(self, session: date) -> tuple[str, ...]:
        """Return the symbols investable on ``session``, in deterministic order."""
        session = _session(session, field_name="session")
        return tuple(sorted({record.symbol for record in self.records if record.active(session)}))

    def sector_of(self, symbol: str, session: date) -> str | None:
        """Return a symbol's sector on ``session``, or ``None`` if not a member."""
        for record in self.records:
            if record.symbol == symbol and record.active(session):
                return record.sector
        return None

    def delistings(self) -> tuple[tuple[str, date], ...]:
        """Return every recorded delisting, in deterministic order."""
        return tuple(
            sorted(
                (record.symbol, record.expiry_session)
                for record in self.records
                if record.delisted and record.expiry_session is not None
            )
        )

    def to_dict(self) -> dict[str, Any]:
        """Return the JSON-friendly membership summary."""
        return {
            "name": self.name,
            "identity": self.identity,
            "n_records": len(self.records),
            "n_symbols": len({record.symbol for record in self.records}),
            "sectors": sorted({record.sector for record in self.records}),
            "delistings": [
                {"symbol": symbol, "session": session.isoformat()}
                for symbol, session in self.delistings()
            ],
            "membership_rule": (
                "resolved as of each session from dated records; a symbol is investable "
                "only between its effective session and its exclusive expiry session"
            ),
        }


def assert_no_future_membership(
    universe: PointInTimeUniverse, panel_columns: Sequence[str], session: date
) -> None:
    """Refuse a panel that offers securities not yet (or no longer) investable.

    This is the check that catches survivorship at the point it would enter: a
    panel built from *today's* constituents contains names that had not listed
    yet, and using it silently backfills the universe with future knowledge.

    Raises:
        UniverseContractError: If any supplied column is not a member.
    """
    investable = set(universe.members(session))
    intruders = sorted(set(panel_columns) - investable)
    if intruders:
        raise UniverseContractError(
            f"panel offers {len(intruders)} securities not investable on {session}: "
            f"{intruders[:8]}. A panel built from a later constituent list backfills "
            "the universe with future composition."
        )


# ---------------------------------------------------------------------------
# Ablations
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class AblationResult:
    """The outcome of removing part of the universe.

    ``hindsight_based`` marks probes whose selection could only be made after the
    fact — removing the top contributors is the canonical example. Such a result
    is a **fragility measure**, never an achievable return, and the flag travels
    with the number so a reader cannot mistake one for the other.
    """

    name: str
    removed: tuple[str, ...]
    retained_count: int
    baseline_metric: float
    ablated_metric: float
    delta: float
    hindsight_based: bool
    interpretation: str

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-friendly evidence row."""
        return {
            "ablation": self.name,
            "removed": list(self.removed),
            "n_removed": len(self.removed),
            "retained_count": self.retained_count,
            "baseline_metric": self.baseline_metric,
            "ablated_metric": self.ablated_metric,
            "delta": self.delta,
            "hindsight_based": self.hindsight_based,
            "interpretation": self.interpretation,
        }


def _validated_contributions(contributions: Mapping[str, float]) -> dict[str, float]:
    if not isinstance(contributions, Mapping) or not contributions:
        raise UniverseContractError("contributions must be a non-empty mapping")
    cleaned: dict[str, float] = {}
    for symbol, value in contributions.items():
        name = _identifier(symbol, field_name="contribution symbol")
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise UniverseContractError(f"contribution for {name} must be a real number")
        if not np.isfinite(value):
            raise UniverseContractError(f"contribution for {name} must be finite")
        cleaned[name] = float(value)
    return cleaned


def drop_top_contributors(
    contributions: Mapping[str, float], *, count: int, baseline_metric: float
) -> AblationResult:
    """Remove the ``count`` largest positive contributors and re-total.

    **A fragility probe, not a performance estimate.** The winners can only be
    identified after the fact, so the resulting number answers "how much of this
    rested on a handful of names?" and never "what could have been earned".
    Ties break on symbol name so the removal set is deterministic.
    """
    cleaned = _validated_contributions(contributions)
    if isinstance(count, bool) or not isinstance(count, int) or count < 1:
        raise UniverseContractError("count must be a positive int")
    if count >= len(cleaned):
        raise UniverseContractError(
            f"removing {count} of {len(cleaned)} contributors leaves nothing to measure"
        )
    ordered = sorted(cleaned.items(), key=lambda item: (-item[1], item[0]))
    removed = tuple(symbol for symbol, _ in ordered[:count])
    retained = {symbol: value for symbol, value in cleaned.items() if symbol not in removed}
    ablated = float(sum(retained.values()))
    return AblationResult(
        name=f"drop_top_{count}_contributors",
        removed=removed,
        retained_count=len(retained),
        baseline_metric=float(baseline_metric),
        ablated_metric=ablated,
        delta=ablated - float(baseline_metric),
        hindsight_based=True,
        interpretation=(
            "Fragility probe. The removed names could only be identified after the fact, "
            "so this is a measure of concentration dependence and NOT an achievable return."
        ),
    )


def drop_sector(
    contributions: Mapping[str, float],
    universe: PointInTimeUniverse,
    *,
    sector: str,
    session: date,
    baseline_metric: float,
) -> AblationResult:
    """Remove every member of one sector and re-total.

    Not hindsight-based: sector membership is known in advance, so the result is
    a genuine "what if this sector had been excluded" estimate under the stated
    universe policy.

    An empty sector is a legitimate outcome — a sector may have no members on a
    given session — and returns an unchanged total rather than raising, because
    refusing would make sector sweeps fail on exactly the sparse sessions worth
    examining.
    """
    cleaned = _validated_contributions(contributions)
    sector_name = _identifier(sector, field_name="sector")
    removed = tuple(
        sorted(symbol for symbol in cleaned if universe.sector_of(symbol, session) == sector_name)
    )
    retained = {symbol: value for symbol, value in cleaned.items() if symbol not in removed}
    ablated = float(sum(retained.values()))
    return AblationResult(
        name=f"drop_sector_{sector_name}",
        removed=removed,
        retained_count=len(retained),
        baseline_metric=float(baseline_metric),
        ablated_metric=ablated,
        delta=ablated - float(baseline_metric),
        hindsight_based=False,
        interpretation=(
            "Sector membership is known in advance, so this is a genuine exclusion "
            "estimate under the stated universe policy."
        ),
    )


def apply_liquidity_floor(
    contributions: Mapping[str, float],
    liquidity: Mapping[str, float],
    *,
    minimum: float,
    baseline_metric: float,
) -> AblationResult:
    """Remove names below a liquidity floor and re-total.

    A symbol with **missing** liquidity is removed, not retained. Treating an
    unknown as passing the floor is how an untradeable name keeps its
    contribution in an investability study.
    """
    cleaned = _validated_contributions(contributions)
    if not np.isfinite(minimum) or minimum < 0.0:
        raise UniverseContractError("liquidity floor must be finite and non-negative")
    removed = tuple(
        sorted(
            symbol
            for symbol in cleaned
            if not np.isfinite(liquidity.get(symbol, float("nan")))
            or float(liquidity.get(symbol, 0.0)) < minimum
        )
    )
    retained = {symbol: value for symbol, value in cleaned.items() if symbol not in removed}
    ablated = float(sum(retained.values()))
    return AblationResult(
        name=f"liquidity_floor_{minimum:g}",
        removed=removed,
        retained_count=len(retained),
        baseline_metric=float(baseline_metric),
        ablated_metric=ablated,
        delta=ablated - float(baseline_metric),
        hindsight_based=False,
        interpretation=(
            "Investability filter. A symbol with unknown liquidity is removed rather than "
            "retained, because treating an unknown as passing keeps an untradeable name "
            "in the result."
        ),
    )


def exclude_inactive(
    contributions: Mapping[str, float],
    universe: PointInTimeUniverse,
    *,
    session: date,
    baseline_metric: float,
) -> AblationResult:
    """Remove names that were not investable on ``session`` and re-total.

    The direct point-in-time check: any contribution attributed to a security
    outside the universe on that session was never earnable.
    """
    cleaned = _validated_contributions(contributions)
    investable = set(universe.members(session))
    removed = tuple(sorted(symbol for symbol in cleaned if symbol not in investable))
    retained = {symbol: value for symbol, value in cleaned.items() if symbol not in removed}
    ablated = float(sum(retained.values()))
    return AblationResult(
        name="exclude_inactive",
        removed=removed,
        retained_count=len(retained),
        baseline_metric=float(baseline_metric),
        ablated_metric=ablated,
        delta=ablated - float(baseline_metric),
        hindsight_based=False,
        interpretation=(
            "Point-in-time membership check. Any contribution attributed to a security "
            "outside the universe on this session was never earnable."
        ),
    )


def concentration_profile(contributions: Mapping[str, float]) -> dict[str, Any]:
    """Summarize how much of the total rests on how few names.

    Reports the Herfindahl index over absolute contributions and the share held
    by the top 1, 3, and 5 names. A result whose top three names carry most of it
    is a different proposition from one spread across the book, and the point
    estimate is identical in both cases.
    """
    cleaned = _validated_contributions(contributions)
    values = np.asarray(list(cleaned.values()), dtype=float)
    magnitude = np.abs(values)
    total = float(magnitude.sum())
    if total <= 0.0:
        return {
            "n_names": len(cleaned),
            "herfindahl": float("nan"),
            "top_shares": {},
            "note": "no non-zero contributions; concentration is undefined",
        }
    shares = magnitude / total
    ordered = np.sort(shares)[::-1]
    return {
        "n_names": len(cleaned),
        "herfindahl": float(np.sum(shares**2)),
        "top_shares": {
            f"top_{count}": float(ordered[:count].sum())
            for count in (1, 3, 5)
            if count <= len(ordered)
        },
        "note": (
            "shares are of total absolute contribution; a result carried by three names "
            "is a different proposition from one spread across the book"
        ),
    }
