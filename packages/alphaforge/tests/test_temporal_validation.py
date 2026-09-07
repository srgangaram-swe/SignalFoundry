"""Interval, boundary, determinism, and visualization tests for temporal validation."""

from __future__ import annotations

from math import comb

import pandas as pd
import pytest

from alphaforge.training import (
    CombinatorialPurgedCV,
    PurgedKFold,
    TemporalValidationConfig,
    TemporalValidationError,
    assert_temporal_integrity,
    fold_assignments,
    fold_metadata,
    make_temporal_validation_plan,
    temporal_plan_identity,
)
from alphaforge.training.temporal_validation import event_end_by_date
from alphaforge.visualization import plot_temporal_folds


def _events(dates: pd.DatetimeIndex, horizon: int = 5) -> pd.DataFrame:
    frame = pd.DataFrame(
        {
            "date": dates,
            "required_future_start": pd.Series(dates).shift(-1),
            "required_future_end": pd.Series(dates).shift(-horizon),
        }
    )
    frame["observable"] = frame["required_future_end"].notna()
    return frame


def _config(**overrides: object) -> TemporalValidationConfig:
    values: dict[str, object] = {
        "scheme": "expanding",
        "min_train_sessions": 80,
        "validation_sessions": 20,
        "test_sessions": 20,
        "step_sessions": 20,
        "purge_sessions": 5,
        "embargo_sessions": 3,
        "final_holdout_sessions": 30,
    }
    values.update(overrides)
    return TemporalValidationConfig(**values)  # type: ignore[arg-type]


def test_plan_is_deterministic_disjoint_and_final_holdout_is_inaccessible() -> None:
    dates = pd.bdate_range("2020-01-02", periods=220)
    first = make_temporal_validation_plan(dates, _config(), label_events=_events(dates))
    second = make_temporal_validation_plan(dates[::-1], _config(), label_events=_events(dates))

    assert temporal_plan_identity(first) == temporal_plan_identity(second)
    assert fold_assignments(first).equals(fold_assignments(second))
    assert set(fold_assignments(first)["role"]) == {
        "train",
        "purge",
        "validation",
        "embargo",
        "test",
        "overlap",
        "final_holdout",
    }
    holdout = set(first[0].final_holdout_dates)
    for fold in first:
        assert_temporal_integrity(fold, event_ends=event_end_by_date(_events(dates)))
        assert not holdout.intersection(fold.train_dates)
        assert not holdout.intersection(fold.validation_dates)
        assert not holdout.intersection(fold.test_dates)


@pytest.mark.parametrize("scheme", ["expanding", "rolling"])
def test_irregular_calendar_sparse_assets_and_overlapping_labels(scheme: str) -> None:
    full = pd.bdate_range("2021-01-01", periods=260)
    dates = full.delete([7, 31, 72, 130, 201])
    base = _events(dates, horizon=12)
    sparse = _events(dates[::3], horizon=4)
    events = pd.concat(
        [
            base.assign(symbol="AAA"),
            sparse.assign(symbol="SPARSE"),
        ],
        ignore_index=True,
    )
    folds = make_temporal_validation_plan(
        dates,
        _config(scheme=scheme),
        label_events=events,
    )

    assert folds
    assert all(fold.overlap_dates for fold in folds)
    if scheme == "rolling":
        assert all(len(fold.train_dates) <= 80 for fold in folds)
    else:
        assert len(folds[-1].train_dates) > len(folds[0].train_dates)


def test_metadata_has_every_role_and_stable_bounds() -> None:
    dates = pd.bdate_range("2022-01-03", periods=220)
    folds = make_temporal_validation_plan(dates, _config(), label_events=_events(dates))
    metadata = fold_metadata(folds)

    assert len(metadata) == len(folds) * 7
    assert metadata.groupby("fold_id")["role"].nunique().eq(7).all()
    assert metadata.loc[metadata["role"].eq("final_holdout"), "sessions"].eq(30).all()


def test_invalid_counts_history_calendars_and_event_intervals_fail_closed() -> None:
    dates = pd.bdate_range("2023-01-02", periods=220)
    with pytest.raises(TemporalValidationError, match="session counts"):
        _config(validation_sessions=0)
    with pytest.raises(TemporalValidationError, match="insufficient history"):
        make_temporal_validation_plan(dates[:50], _config())
    with pytest.raises(TemporalValidationError, match="timezone-naive"):
        make_temporal_validation_plan(dates.tz_localize("UTC"), _config())

    invalid = _events(dates)
    invalid.loc[0, "required_future_start"] = dates[0]
    with pytest.raises(TemporalValidationError, match="date < future_start"):
        make_temporal_validation_plan(dates, _config(), label_events=invalid)

    outside = _events(dates).iloc[:1].copy()
    outside["date"] = pd.Timestamp("1999-01-01")
    with pytest.raises(TemporalValidationError, match="outside"):
        make_temporal_validation_plan(dates, _config(), label_events=outside)


def test_purged_kfold_uses_exact_events_and_excludes_final_holdout() -> None:
    dates = pd.bdate_range("2020-01-02", periods=180)
    events = _events(dates, horizon=15)
    holdout_start = dates[-30]
    splitter = PurgedKFold(n_splits=5, purge=2, embargo=1)

    splits = list(
        splitter.split(
            dates,
            label_events=events,
            final_holdout_start=holdout_start,
        )
    )
    assert len(splits) == 5
    ends = event_end_by_date(events)
    for train, test in splits:
        test_end = max(test.max(), ends.reindex(test).dropna().max())
        train_ends = ends.reindex(train).dropna()
        overlaps = (train_ends.index <= test_end) & (train_ends >= test.min())
        assert not overlaps.any()
        assert train.max() < holdout_start
        assert test.max() < holdout_start
        assert ends.reindex(train).dropna().lt(holdout_start).all()
        assert ends.reindex(test).dropna().lt(holdout_start).all()


def test_cpcv_is_bounded_deterministic_and_covers_combinations() -> None:
    dates = pd.bdate_range("2020-01-02", periods=180)
    cv = CombinatorialPurgedCV(n_groups=6, n_test_groups=2, purge=2, embargo=1)
    first = list(cv.split(dates, label_events=_events(dates, horizon=3)))
    second = list(cv.split(dates, label_events=_events(dates, horizon=3)))

    assert len(first) == comb(6, 2)
    assert [split[2] for split in first] == [split[2] for split in second]
    with pytest.raises(ValueError, match="cannot exceed"):
        list(CombinatorialPurgedCV(n_groups=181, n_test_groups=1).split(dates))
    with pytest.raises(ValueError, match="fold limit"):
        list(CombinatorialPurgedCV(n_groups=30, n_test_groups=15).split(dates))


def test_temporal_fold_visualization_is_a_nonempty_png(tmp_path) -> None:
    dates = pd.bdate_range("2020-01-02", periods=220)
    folds = make_temporal_validation_plan(dates, _config(), label_events=_events(dates))
    path = plot_temporal_folds(folds, tmp_path / "folds.png")

    assert path.is_file()
    assert path.read_bytes().startswith(b"\x89PNG")
    assert path.stat().st_size > 10_000
