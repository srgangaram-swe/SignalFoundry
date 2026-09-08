"""Tests for the Gaussian HMM regime model."""

from __future__ import annotations

import numpy as np
import pandas as pd

from alphaforge.models.regime import GaussianHMM2, causal_stress_probability


def _two_regime_returns(seed: int = 5, calm_days: int = 300, stress_days: int = 120):
    rng = np.random.default_rng(seed)
    calm = rng.normal(0.0004, 0.006, calm_days)
    stress = rng.normal(-0.001, 0.030, stress_days)
    return pd.Series(np.concatenate([calm, stress, calm[:150]]))


def test_hmm_recovers_separated_volatility_regimes():
    r = _two_regime_returns()
    model = GaussianHMM2().fit(r)
    assert model.variances_ is not None
    assert model.transition_ is not None
    # canonical ordering: state 1 is the high-variance (stress) state
    assert model.variances_[1] > model.variances_[0] * 3
    # transition matrix should be persistent (regimes are sticky)
    assert model.transition_[0, 0] > 0.8 and model.transition_[1, 1] > 0.8

    probs = model.filtered_probabilities(r)
    stress_window = probs[320:400, 1]  # deep inside the stress segment
    calm_window = probs[50:250, 1]
    assert stress_window.mean() > 0.7
    assert calm_window.mean() < 0.3


def test_hmm_fit_is_deterministic():
    r = _two_regime_returns()
    a = GaussianHMM2().fit(r)
    b = GaussianHMM2().fit(r)
    assert a.means_ is not None and b.means_ is not None
    assert a.transition_ is not None and b.transition_ is not None
    np.testing.assert_allclose(a.means_, b.means_)
    np.testing.assert_allclose(a.transition_, b.transition_)


def test_causal_stress_probability_shape_and_warmup():
    r = _two_regime_returns()
    stress = causal_stress_probability(r, refit_every=50, min_train=150)
    assert len(stress) == len(r)
    assert stress.iloc[:150].isna().all()  # no output before min_train
    assert stress.iloc[150:].notna().all()
    assert ((stress.dropna() >= 0) & (stress.dropna() <= 1)).all()
    # stress segment (300..420) should read as high stress probability
    assert stress.iloc[330:420].mean() > 0.6
