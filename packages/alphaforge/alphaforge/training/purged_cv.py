"""Purged K-Fold and Combinatorial Purged Cross-Validation (CPCV).

Standard K-fold is invalid for overlapping-label time series: a label at date
t spans (t, t+h], so samples near a train/test boundary share information.
Following Lopez de Prado (Advances in Financial Machine Learning, ch. 7 & 12):

- **Purging** removes training dates within ``purge`` days of a test block on
  both sides, eliminating label-interval overlap (set purge >= label horizon).
- **Embargo** drops an extra buffer *after* each test block to kill serial-
  correlation leakage from features computed on trailing windows.
- **CPCV** evaluates every combination of test groups, producing many
  out-of-sample paths instead of one — the input the PBO estimator
  (alphaforge.evaluation.overfitting) needs.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass
from math import comb

import numpy as np
import pandas as pd

from alphaforge.training.temporal_validation import (
    MAX_TEMPORAL_FOLDS,
    TemporalValidationError,
    event_end_by_date,
)


def _unique_dates(dates) -> pd.DatetimeIndex:
    try:
        values = pd.to_datetime(pd.Series(dates), errors="raise")
    except (TypeError, ValueError) as exc:
        raise ValueError("dates must contain valid market-session labels") from exc
    if values.empty or values.isna().any():
        raise ValueError("dates must contain finite market-session labels")
    if values.dt.tz is not None:
        raise ValueError("dates must be timezone-naive market-session labels")
    return pd.DatetimeIndex(values.dt.normalize().drop_duplicates().sort_values())


def _purged_train_mask(
    n: int, test_blocks: list[tuple[int, int]], purge: int, embargo: int
) -> np.ndarray:
    """True where a date index is usable for training given test blocks."""
    mask = np.ones(n, dtype=bool)
    for start, end in test_blocks:
        lo = max(0, start - purge)
        hi = min(n - 1, end + purge + embargo)
        mask[lo : hi + 1] = False
    return mask


def _exact_interval_mask(
    dates: pd.DatetimeIndex,
    test_blocks: list[tuple[int, int]],
    event_ends: pd.Series,
) -> np.ndarray:
    """Keep samples whose exact future interval cannot intersect the test span."""

    if event_ends.empty:
        return np.ones(len(dates), dtype=bool)
    train_ends = event_ends.reindex(dates)
    effective_ends = train_ends.where(train_ends.notna(), pd.Series(dates, index=dates))
    keep = np.ones(len(dates), dtype=bool)
    starts = pd.Series(dates, index=dates)
    for start, end in test_blocks:
        test_dates = dates[start : end + 1]
        test_ends = event_ends.reindex(test_dates).dropna()
        test_end = pd.Timestamp(test_dates.max())
        if not test_ends.empty:
            test_end = max(test_end, pd.Timestamp(test_ends.max()))
        test_start = pd.Timestamp(test_dates.min())
        overlaps = starts.le(test_end) & effective_ends.ge(test_start)
        keep &= ~overlaps.to_numpy()
    return keep


def _development_dates(
    dates: pd.DatetimeIndex,
    *,
    event_ends: pd.Series,
    final_holdout_start: str | pd.Timestamp | None,
) -> tuple[pd.DatetimeIndex, pd.Timestamp | None]:
    if final_holdout_start is None:
        return dates, None
    boundary = pd.Timestamp(final_holdout_start)
    if boundary.tzinfo is not None:
        raise ValueError("final_holdout_start must be timezone-naive")
    boundary = boundary.normalize()
    development = dates[dates < boundary]
    if development.empty or len(development) == len(dates):
        raise ValueError("final_holdout_start must split the supplied calendar")
    if not event_ends.empty:
        ends = event_ends.reindex(development)
        development = development[(~ends.notna() | ends.lt(boundary)).to_numpy()]
        if development.empty:
            raise ValueError("all development labels cross the final holdout boundary")
    return development, boundary


@dataclass(frozen=True)
class PurgedKFold:
    """Contiguous K-fold over dates with purging and embargo."""

    n_splits: int = 5
    purge: int = 0
    embargo: int = 0

    def split(
        self,
        dates,
        *,
        label_events: pd.DataFrame | None = None,
        final_holdout_start: str | pd.Timestamp | None = None,
    ):
        """Yield interval-safe ``(train_dates, test_dates)`` pairs.

        ``label_events`` enables exact purging for heterogeneous horizons.
        ``final_holdout_start`` removes that interval from cross-validation and
        also removes development samples whose labels cross into it.
        """

        if self.n_splits < 2:
            raise ValueError("n_splits must be >= 2")
        if self.purge < 0 or self.embargo < 0:
            raise ValueError("purge and embargo must be non-negative")
        event_ends = event_end_by_date(label_events)
        u, holdout = _development_dates(
            _unique_dates(dates),
            event_ends=event_ends,
            final_holdout_start=final_holdout_start,
        )
        if self.n_splits > len(u):
            raise ValueError("n_splits cannot exceed the number of development sessions")
        folds = np.array_split(np.arange(len(u)), self.n_splits)
        for fold in folds:
            mask = _purged_train_mask(len(u), [(fold[0], fold[-1])], self.purge, self.embargo)
            mask &= _exact_interval_mask(u, [(int(fold[0]), int(fold[-1]))], event_ends)
            if holdout is not None and not event_ends.empty:
                ends = event_ends.reindex(u)
                mask &= (~ends.notna() | ends.lt(holdout)).to_numpy()
            if not mask.any():
                raise ValueError("purging removed every training session from a fold")
            yield u[np.flatnonzero(mask)], u[fold]


@dataclass(frozen=True)
class CombinatorialPurgedCV:
    """CPCV: every C(n_groups, n_test_groups) combination of test groups.

    Each date group appears in many distinct test sets, so OOS predictions can
    be assembled into multiple backtest paths rather than a single trajectory.
    """

    n_groups: int = 8
    n_test_groups: int = 2
    purge: int = 0
    embargo: int = 0

    @property
    def n_splits(self) -> int:
        return comb(self.n_groups, self.n_test_groups)

    def split(
        self,
        dates,
        *,
        label_events: pd.DataFrame | None = None,
        final_holdout_start: str | pd.Timestamp | None = None,
    ):
        """Yield interval-safe train/test dates and deterministic group ids."""

        if not 0 < self.n_test_groups < self.n_groups:
            raise ValueError("need 0 < n_test_groups < n_groups")
        if self.purge < 0 or self.embargo < 0:
            raise ValueError("purge and embargo must be non-negative")
        if self.n_splits > MAX_TEMPORAL_FOLDS:
            raise ValueError(f"CPCV split count exceeds the {MAX_TEMPORAL_FOLDS:,}-fold limit")
        try:
            event_ends = event_end_by_date(label_events)
        except TemporalValidationError as exc:
            raise ValueError(str(exc)) from exc
        u, holdout = _development_dates(
            _unique_dates(dates),
            event_ends=event_ends,
            final_holdout_start=final_holdout_start,
        )
        if self.n_groups > len(u):
            raise ValueError("n_groups cannot exceed the number of development sessions")
        groups = np.array_split(np.arange(len(u)), self.n_groups)
        for combo in itertools.combinations(range(self.n_groups), self.n_test_groups):
            blocks = [(groups[c][0], groups[c][-1]) for c in combo]
            test_idx = np.concatenate([groups[c] for c in combo])
            mask = _purged_train_mask(len(u), blocks, self.purge, self.embargo)
            mask &= _exact_interval_mask(u, blocks, event_ends)
            if holdout is not None and not event_ends.empty:
                ends = event_ends.reindex(u)
                mask &= (~ends.notna() | ends.lt(holdout)).to_numpy()
            if not mask.any():
                raise ValueError("purging removed every training session from a CPCV split")
            yield u[np.flatnonzero(mask)], u[test_idx], combo


def run_purged_cv(
    features: pd.DataFrame,
    labels: pd.DataFrame,
    model_specs: list[dict],
    target: str,
    splitter: PurgedKFold | CombinatorialPurgedCV | None = None,
    *,
    label_events: pd.DataFrame | None = None,
    final_holdout_start: str | pd.Timestamp | None = None,
) -> pd.DataFrame:
    """Fit/predict every model over purged splits; return OOS predictions.

    Complements the chronological walk-forward driver: CPCV predictions feed
    the PBO estimator, while walk-forward remains the primary honest backtest.
    """
    from alphaforge.models.registry import create_model
    from alphaforge.training.walk_forward import ID_COLUMNS, _model_matrix, supervised_frame

    splitter = splitter or PurgedKFold(n_splits=5, purge=20, embargo=5)
    data, x_cols = supervised_frame(features, labels, target)
    blocks = []
    for split_id, split in enumerate(
        splitter.split(
            data["date"],
            label_events=label_events,
            final_holdout_start=final_holdout_start,
        )
    ):
        train_dates, test_dates = split[0], split[1]
        combo = split[2] if len(split) > 2 else (split_id,)
        train = data[data["date"].isin(train_dates)].dropna(subset=[target])
        test = data[data["date"].isin(test_dates)].dropna(subset=[target])
        if train.empty or test.empty:
            continue
        for spec in model_specs or [{"name": "zero_baseline"}]:
            name = spec["name"]
            model = create_model(name, **spec.get("params", {}))
            model.fit(_model_matrix(train, x_cols, model), train[target].astype(float))
            block = test[ID_COLUMNS + [target]].rename(columns={target: "target"})
            block = block.assign(
                prediction=model.predict(_model_matrix(test, x_cols, model)),
                model=name,
                split_id=split_id,
                test_groups=str(combo),
                train_start=pd.Timestamp(train_dates.min()),
                train_end=pd.Timestamp(train_dates.max()),
                test_start=pd.Timestamp(test_dates.min()),
                test_end=pd.Timestamp(test_dates.max()),
            )
            blocks.append(block)
    if not blocks:
        return pd.DataFrame(
            columns=ID_COLUMNS
            + [
                "target",
                "prediction",
                "model",
                "split_id",
                "test_groups",
                "train_start",
                "train_end",
                "test_start",
                "test_end",
            ]
        )
    return pd.concat(blocks, ignore_index=True)
