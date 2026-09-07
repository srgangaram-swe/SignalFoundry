"""Deterministic, interval-aware temporal validation plans.

The planner separates model-development evidence into explicit training,
validation, and test roles while keeping a final holdout unavailable to model
selection.  Session-count purge and embargo gaps are supplemented by exact
label-event intervals: any sample whose required future information crosses a
role boundary is classified as ``overlap`` and excluded.

Dates are market-session labels rather than elapsed calendar days.  This makes
the contract valid on irregular exchange calendars without inventing missing
sessions.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any, Literal

import pandas as pd

TemporalScheme = Literal["expanding", "rolling"]
FoldRole = Literal[
    "train",
    "purge",
    "validation",
    "embargo",
    "test",
    "overlap",
    "final_holdout",
]

MAX_TEMPORAL_SESSIONS = 5_000_000
MAX_TEMPORAL_FOLDS = 10_000


class TemporalValidationError(ValueError):
    """Raised when a temporal plan is invalid, ambiguous, or leaks information."""


@dataclass(frozen=True)
class TemporalValidationConfig:
    """Bounded configuration for one rolling or expanding validation plan.

    All window lengths are counts of observed market sessions.  The final
    holdout is described for audit and visualization but is never returned as a
    training, validation, or test role.
    """

    scheme: TemporalScheme = "expanding"
    min_train_sessions: int = 756
    validation_sessions: int = 126
    test_sessions: int = 126
    step_sessions: int = 126
    purge_sessions: int = 20
    embargo_sessions: int = 5
    final_holdout_sessions: int = 252
    max_folds: int | None = None

    def __post_init__(self) -> None:
        if self.scheme not in {"expanding", "rolling"}:
            raise TemporalValidationError("scheme must be 'expanding' or 'rolling'")
        positive = {
            "min_train_sessions": self.min_train_sessions,
            "validation_sessions": self.validation_sessions,
            "test_sessions": self.test_sessions,
            "step_sessions": self.step_sessions,
            "final_holdout_sessions": self.final_holdout_sessions,
        }
        invalid_positive = [name for name, value in positive.items() if value <= 0]
        if invalid_positive:
            raise TemporalValidationError(
                f"session counts must be positive: {sorted(invalid_positive)}"
            )
        if self.purge_sessions < 0 or self.embargo_sessions < 0:
            raise TemporalValidationError(
                "purge_sessions and embargo_sessions must be non-negative"
            )
        if self.max_folds is not None and not 0 < self.max_folds <= MAX_TEMPORAL_FOLDS:
            raise TemporalValidationError(
                f"max_folds must be in [1, {MAX_TEMPORAL_FOLDS}] when supplied"
            )


@dataclass(frozen=True)
class TemporalFold:
    """One immutable, disjoint temporal fold allocation."""

    fold_id: int
    scheme: TemporalScheme
    train_dates: tuple[pd.Timestamp, ...]
    purge_dates: tuple[pd.Timestamp, ...]
    validation_dates: tuple[pd.Timestamp, ...]
    embargo_dates: tuple[pd.Timestamp, ...]
    test_dates: tuple[pd.Timestamp, ...]
    overlap_dates: tuple[pd.Timestamp, ...]
    final_holdout_dates: tuple[pd.Timestamp, ...]

    def role_dates(self) -> dict[FoldRole, tuple[pd.Timestamp, ...]]:
        """Return every auditable role in deterministic display order."""

        return {
            "train": self.train_dates,
            "purge": self.purge_dates,
            "validation": self.validation_dates,
            "embargo": self.embargo_dates,
            "test": self.test_dates,
            "overlap": self.overlap_dates,
            "final_holdout": self.final_holdout_dates,
        }


def _normalize_dates(dates: Any) -> pd.DatetimeIndex:
    try:
        values = pd.to_datetime(pd.Series(dates), errors="raise")
    except (TypeError, ValueError) as exc:
        raise TemporalValidationError("dates must contain valid market-session labels") from exc
    if values.empty:
        raise TemporalValidationError("dates must contain at least one session")
    if values.dt.tz is not None:
        raise TemporalValidationError("dates must be timezone-naive market-session labels")
    if values.isna().any():
        raise TemporalValidationError("dates must not contain missing timestamps")
    unique = pd.DatetimeIndex(values.dt.normalize().drop_duplicates().sort_values())
    if len(unique) > MAX_TEMPORAL_SESSIONS:
        raise TemporalValidationError(
            f"temporal plan exceeds the {MAX_TEMPORAL_SESSIONS:,}-session limit"
        )
    return unique


def event_end_by_date(label_events: pd.DataFrame | None) -> pd.Series:
    """Return the latest required future session for each observable sample date.

    Multiple assets and labels may share a date.  Taking the maximum end is
    deliberately conservative: a date is eligible only when every supplied
    event is contained by the relevant fold boundary.
    """

    if label_events is None:
        return pd.Series(dtype="datetime64[ns]")
    required = ("date", "required_future_start", "required_future_end", "observable")
    missing = sorted(set(required) - set(label_events.columns))
    if missing:
        raise TemporalValidationError(f"label events are missing required columns: {missing}")
    events = label_events.loc[
        label_events["observable"].eq(True), list(required)
    ].copy()  # noqa: E712
    if events.empty:
        return pd.Series(dtype="datetime64[ns]")
    for column in ("date", "required_future_start", "required_future_end"):
        try:
            events[column] = pd.to_datetime(events[column], errors="raise")
        except (TypeError, ValueError) as exc:
            raise TemporalValidationError(f"{column} must contain valid session labels") from exc
        if events[column].dt.tz is not None or events[column].isna().any():
            raise TemporalValidationError(f"{column} must be finite and timezone-naive")
        events[column] = events[column].dt.normalize()
    invalid = events["required_future_start"].le(events["date"]) | events["required_future_end"].lt(
        events["required_future_start"]
    )
    if invalid.any():
        raise TemporalValidationError(
            "observable label intervals must satisfy date < future_start <= future_end"
        )
    return events.groupby("date", sort=True)["required_future_end"].max()


def _contained_dates(
    candidates: pd.DatetimeIndex,
    *,
    boundary: pd.Timestamp,
    event_ends: pd.Series,
) -> tuple[pd.DatetimeIndex, pd.DatetimeIndex]:
    if candidates.empty or event_ends.empty:
        return candidates, pd.DatetimeIndex([])
    candidate_ends = event_ends.reindex(candidates)
    crossing = candidate_ends.notna() & candidate_ends.ge(boundary)
    return candidates[~crossing.to_numpy()], candidates[crossing.to_numpy()]


def _as_tuple(dates: pd.DatetimeIndex) -> tuple[pd.Timestamp, ...]:
    return tuple(pd.Timestamp(value) for value in dates)


def make_temporal_validation_plan(
    dates: Any,
    config: TemporalValidationConfig | dict[str, Any] | None = None,
    *,
    label_events: pd.DataFrame | None = None,
) -> list[TemporalFold]:
    """Construct interval-safe rolling or expanding train/validation/test folds.

    Label events are optional for backward-compatible horizon-only research.
    When supplied, exact future intervals supersede assumptions about a uniform
    horizon and crossing samples are moved to the explicit ``overlap`` role.

    Raises:
        TemporalValidationError: If configuration, dates, intervals, or the
            resulting fold allocation are invalid.
    """

    cfg = (
        config
        if isinstance(config, TemporalValidationConfig)
        else TemporalValidationConfig(**(config or {}))
    )
    sessions = _normalize_dates(dates)
    minimum = (
        cfg.min_train_sessions
        + cfg.purge_sessions
        + cfg.validation_sessions
        + cfg.embargo_sessions
        + cfg.test_sessions
        + cfg.final_holdout_sessions
    )
    if len(sessions) < minimum:
        raise TemporalValidationError(
            f"insufficient history: need at least {minimum} sessions, received {len(sessions)}"
        )

    event_ends = event_end_by_date(label_events)
    unknown_event_dates = event_ends.index.difference(sessions)
    if len(unknown_event_dates):
        raise TemporalValidationError("label events contain dates outside the supplied calendar")

    holdout = sessions[-cfg.final_holdout_sessions :]
    development = sessions[: -cfg.final_holdout_sessions]
    holdout_start = pd.Timestamp(holdout[0])
    cursor = cfg.min_train_sessions
    folds: list[TemporalFold] = []

    while cursor < len(development):
        purge_start = cursor
        validation_start = purge_start + cfg.purge_sessions
        validation_end = validation_start + cfg.validation_sessions
        embargo_start = validation_end
        test_start = embargo_start + cfg.embargo_sessions
        test_end = test_start + cfg.test_sessions
        if test_end > len(development):
            break

        raw_train = development[:cursor]
        if cfg.scheme == "rolling":
            raw_train = raw_train[-cfg.min_train_sessions :]
        purge = development[purge_start:validation_start]
        raw_validation = development[validation_start:validation_end]
        embargo = development[embargo_start:test_start]
        raw_test = development[test_start:test_end]

        train, train_overlap = _contained_dates(
            raw_train,
            boundary=pd.Timestamp(raw_validation[0]),
            event_ends=event_ends,
        )
        validation, validation_overlap = _contained_dates(
            raw_validation,
            boundary=pd.Timestamp(raw_test[0]),
            event_ends=event_ends,
        )
        test_boundary = (
            holdout_start if test_end == len(development) else pd.Timestamp(development[test_end])
        )
        test, test_overlap = _contained_dates(
            raw_test,
            boundary=test_boundary,
            event_ends=event_ends,
        )
        overlap = train_overlap.union(validation_overlap).union(test_overlap).sort_values()
        if train.empty or validation.empty or test.empty:
            raise TemporalValidationError(
                f"fold {len(folds)} became empty after exact interval purging"
            )

        fold = TemporalFold(
            fold_id=len(folds),
            scheme=cfg.scheme,
            train_dates=_as_tuple(train),
            purge_dates=_as_tuple(purge),
            validation_dates=_as_tuple(validation),
            embargo_dates=_as_tuple(embargo),
            test_dates=_as_tuple(test),
            overlap_dates=_as_tuple(overlap),
            final_holdout_dates=_as_tuple(holdout),
        )
        assert_temporal_integrity(fold, event_ends=event_ends)
        folds.append(fold)
        if cfg.max_folds is not None and len(folds) >= cfg.max_folds:
            break
        cursor += cfg.step_sessions

    if not folds:
        raise TemporalValidationError("configuration produced no complete temporal folds")
    return folds


def assert_temporal_integrity(
    fold: TemporalFold,
    *,
    event_ends: pd.Series | None = None,
) -> None:
    """Fail closed unless all roles are disjoint, ordered, and interval-safe."""

    role_dates = fold.role_dates()
    seen: set[pd.Timestamp] = set()
    for role, values in role_dates.items():
        if tuple(sorted(set(values))) != values:
            raise TemporalValidationError(f"{role} dates must be ordered and unique")
        overlap = seen.intersection(values)
        if overlap:
            raise TemporalValidationError(f"fold roles overlap on {min(overlap)}")
        seen.update(values)
    if not fold.train_dates or not fold.validation_dates or not fold.test_dates:
        raise TemporalValidationError("train, validation, and test roles must be non-empty")
    if not (
        max(fold.train_dates)
        < min(fold.validation_dates)
        <= max(fold.validation_dates)
        < min(fold.test_dates)
        <= max(fold.test_dates)
        < min(fold.final_holdout_dates)
    ):
        raise TemporalValidationError("temporal roles are not strictly ordered")

    if event_ends is None or event_ends.empty:
        return
    boundaries = (
        (fold.train_dates, min(fold.validation_dates), "train"),
        (fold.validation_dates, min(fold.test_dates), "validation"),
        (fold.test_dates, min(fold.final_holdout_dates), "test"),
    )
    for dates, boundary, boundary_role in boundaries:
        ends = event_ends.reindex(pd.DatetimeIndex(dates)).dropna()
        if ends.ge(boundary).any():
            raise TemporalValidationError(
                f"{boundary_role} label interval crosses its protected boundary"
            )


def fold_assignments(folds: list[TemporalFold]) -> pd.DataFrame:
    """Return one deterministic machine-readable row per fold, date, and role."""

    rows = [
        {
            "fold_id": fold.fold_id,
            "scheme": fold.scheme,
            "date": date,
            "role": role,
        }
        for fold in folds
        for role, dates in fold.role_dates().items()
        for date in dates
    ]
    return pd.DataFrame(rows).sort_values(["fold_id", "date", "role"]).reset_index(drop=True)


def fold_metadata(folds: list[TemporalFold]) -> pd.DataFrame:
    """Return compact role bounds and counts without observations or targets."""

    rows: list[dict[str, Any]] = []
    for fold in folds:
        for role, dates in fold.role_dates().items():
            rows.append(
                {
                    "fold_id": fold.fold_id,
                    "scheme": fold.scheme,
                    "role": role,
                    "date_start": None if not dates else dates[0],
                    "date_end": None if not dates else dates[-1],
                    "sessions": len(dates),
                }
            )
    return pd.DataFrame(rows)


def temporal_plan_identity(folds: list[TemporalFold]) -> str:
    """Hash the complete fold allocation for immutable run provenance."""

    records = fold_assignments(folds).copy()
    records["date"] = records["date"].dt.strftime("%Y-%m-%d")
    payload = records.to_dict(orient="records")
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("ascii")
    return hashlib.sha256(encoded).hexdigest()
