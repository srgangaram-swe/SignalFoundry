"""Tests for purged CV splitters and overfitting statistics."""

from __future__ import annotations

from math import comb

import numpy as np
import pandas as pd

from alphaforge.evaluation import (
    deflated_sharpe_ratio,
    newey_west_tstat,
    probabilistic_sharpe_ratio,
    probability_of_backtest_overfitting,
)
from alphaforge.training import CombinatorialPurgedCV, PurgedKFold

DATES = pd.bdate_range("2020-01-01", periods=500)


def test_purged_kfold_no_overlap_and_purge_respected():
    purge, embargo = 10, 5
    splitter = PurgedKFold(n_splits=5, purge=purge, embargo=embargo)
    n_folds = 0
    for train_dates, test_dates in splitter.split(DATES):
        n_folds += 1
        train_set, test_set = set(train_dates), set(test_dates)
        assert not train_set & test_set
        # positional distance between any train date and the test block
        pos = {d: i for i, d in enumerate(DATES)}
        t0, t1 = pos[test_dates.min()], pos[test_dates.max()]
        for d in train_dates:
            i = pos[d]
            assert i < t0 - purge or i > t1 + purge + embargo
    assert n_folds == 5


def test_cpcv_combination_count_and_coverage():
    cv = CombinatorialPurgedCV(n_groups=6, n_test_groups=2, purge=5, embargo=2)
    assert cv.n_splits == comb(6, 2)
    seen = []
    for train_dates, test_dates, combo in cv.split(DATES):
        seen.append(combo)
        assert len(set(train_dates) & set(test_dates)) == 0
        assert len(test_dates) > 0
    assert len(seen) == comb(6, 2)
    # every group appears as test in exactly C(5,1) combinations
    counts = pd.Series([g for combo in seen for g in combo]).value_counts()
    assert (counts == comb(5, 1)).all()


def test_pbo_near_half_for_pure_noise_and_low_for_real_edge():
    # a single finite sample gives a noisy PBO (one lucky strategy can
    # dominate both halves), so average the estimator over realizations
    noise_pbos, skill_pbos = [], []
    for seed in range(10):
        rng = np.random.default_rng(seed)
        noise = pd.DataFrame(rng.normal(0, 0.01, size=(400, 10)))
        noise_pbos.append(probability_of_backtest_overfitting(noise, n_blocks=8)["pbo"])
        skilled = noise.copy()
        skilled[0] = skilled[0] + 0.004  # one strategy with a genuine edge
        skill_pbos.append(probability_of_backtest_overfitting(skilled, n_blocks=8)["pbo"])

    mean_noise, mean_skill = np.mean(noise_pbos), np.mean(skill_pbos)
    assert 0.3 <= mean_noise <= 0.7  # selection among noise is ~coin-flip OOS
    assert mean_skill < mean_noise  # a real edge survives the IS/OOS swap
    assert mean_skill <= 0.15


def test_deflated_sharpe_decreases_with_trials():
    rng = np.random.default_rng(1)
    returns = pd.Series(rng.normal(0.0004, 0.01, 1000))
    d1 = deflated_sharpe_ratio(returns, n_trials=1)
    d10 = deflated_sharpe_ratio(returns, n_trials=10)
    d100 = deflated_sharpe_ratio(returns, n_trials=100)
    assert d1["deflated_sharpe_prob"] >= d10["deflated_sharpe_prob"] >= d100["deflated_sharpe_prob"]
    assert d100["expected_max_sharpe"] > d10["expected_max_sharpe"] > 0


def test_psr_sane_bounds():
    rng = np.random.default_rng(2)
    strong = pd.Series(rng.normal(0.002, 0.01, 750))
    weak = pd.Series(rng.normal(-0.001, 0.01, 750))
    assert probabilistic_sharpe_ratio(strong) > 0.95
    assert probabilistic_sharpe_ratio(weak) < 0.5


def test_newey_west_tstat():
    rng = np.random.default_rng(3)
    noise = pd.Series(rng.normal(0, 1, 500))
    assert abs(newey_west_tstat(noise)) < 3.0
    shifted = noise + 1.0
    assert newey_west_tstat(shifted) > 10.0


def test_run_purged_cv_produces_oos_predictions(small_features, small_labels):
    from alphaforge.training import run_purged_cv

    preds = run_purged_cv(
        small_features,
        small_labels,
        model_specs=[{"name": "ridge", "params": {"alpha": 10.0}}],
        target="fwd_ret_5",
        splitter=PurgedKFold(n_splits=3, purge=5, embargo=2),
    )
    assert not preds.empty
    assert {"date", "symbol", "target", "prediction", "model", "split_id"} <= set(preds.columns)
    # each date should be predicted at most once per model (folds are disjoint)
    dupes = preds.groupby(["date", "symbol", "model"]).size()
    assert (dupes == 1).all()
