"""Tests for regime and change-point model contracts (SF-S3-MR4).

Grouped by the property each defends: causality and leakage, deterministic
identity, canonical labelling, posterior/confidence outputs, synthetic recovery,
no-change and abrupt/gradual change behaviour, degenerate input, non-convergence
reporting, retrospective/causal separation, and incremental-value evidence.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import pytest

from alphaforge.regimes import (
    MAX_STATES,
    BayesianOnlineChangePoint,
    CusumRegime,
    ExpandingFitPlan,
    GaussianHMMRegime,
    GaussianMixtureRegime,
    NotFittedError,
    RegimeError,
    RegimeLeakageError,
    RegimeModel,
    RegimeStates,
    RuleBasedRegime,
    compare_regime_models,
    evaluate_regime_value,
    expanding_state_probabilities,
    forward_returns,
    rule_statistic,
    segment_bayesian,
    segment_kernel,
)
from alphaforge.regimes.base import MIN_FIT_OBSERVATIONS, RegimeReport

CAUSAL_FACTORIES = {
    "rule": lambda: RuleBasedRegime("volatility"),
    "gmm": lambda: GaussianMixtureRegime(2),
    "hmm": lambda: GaussianHMMRegime(2),
    "cusum": lambda: CusumRegime(),
    "bocpd": lambda: BayesianOnlineChangePoint(),
}
FITTED_MODELS = ("gmm", "hmm", "cusum", "bocpd")


def _switching_series(seed: int = 7, n: int = 900) -> pd.Series:
    """Calm / stress / calm volatility with a known switch at 400 and 600."""
    rng = np.random.default_rng(seed)
    volatility = np.concatenate([np.full(400, 0.006), np.full(200, 0.025), np.full(n - 600, 0.006)])
    values = rng.normal(0.0, 1.0, n) * volatility
    return pd.Series(values, index=pd.bdate_range("2019-01-01", periods=n), name="return")


def _constant_series(seed: int = 3, n: int = 700) -> pd.Series:
    rng = np.random.default_rng(seed)
    return pd.Series(
        rng.normal(0.0, 0.01, n), index=pd.bdate_range("2019-01-01", periods=n), name="return"
    )


@pytest.fixture
def switching() -> pd.Series:
    return _switching_series()


@pytest.fixture
def plan() -> ExpandingFitPlan:
    return ExpandingFitPlan(min_train=300, refit_every=100)


# ---------------------------------------------------------------------------
# Causality and leakage
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("name", sorted(CAUSAL_FACTORIES))
def test_future_observations_cannot_change_earlier_states(
    name: str, switching: pd.Series, plan: ExpandingFitPlan
) -> None:
    """The leakage-mutation test: rewrite the future, earlier rows must not move."""
    factory = CAUSAL_FACTORIES[name]
    baseline, _ = expanding_state_probabilities(switching, factory, plan)

    cutoff = 700
    mutated = switching.copy()
    rng = np.random.default_rng(99)
    mutated.iloc[cutoff:] = rng.normal(0.0, 0.5, len(mutated) - cutoff)
    perturbed, _ = expanding_state_probabilities(mutated, factory, plan)

    np.testing.assert_allclose(
        baseline.probabilities[:cutoff], perturbed.probabilities[:cutoff], rtol=1e-12, atol=1e-12
    )


@pytest.mark.parametrize("name", sorted(CAUSAL_FACTORIES))
def test_warmup_rows_carry_no_estimate(
    name: str, switching: pd.Series, plan: ExpandingFitPlan
) -> None:
    states, _ = expanding_state_probabilities(switching, CAUSAL_FACTORIES[name], plan)
    assert np.isnan(states.probabilities[: plan.min_train]).all()
    assert states.observed[plan.min_train :].any()
    assert states.causal is True


def test_expanding_driver_refuses_a_non_causal_model(switching: pd.Series) -> None:
    class Retrospective(RegimeModel):
        name = "retrospective"
        causal = False

        def _fit(self, values):  # pragma: no cover - refused before use
            raise AssertionError("must not be fit")

        def _filter(self, values):  # pragma: no cover - refused before use
            raise AssertionError("must not be filtered")

        def fitted_parameters(self):  # pragma: no cover
            return {}

        @property
        def state_labels(self):
            return ("a", "b")

    with pytest.raises(RegimeLeakageError, match="retrospective"):
        expanding_state_probabilities(switching, Retrospective)


def test_expanding_driver_requires_a_regime_model(switching: pd.Series) -> None:
    with pytest.raises(RegimeError, match="must produce a RegimeModel"):
        expanding_state_probabilities(switching, lambda: object())


def test_each_refit_is_reported_separately(switching: pd.Series, plan: ExpandingFitPlan) -> None:
    _states, reports = expanding_state_probabilities(switching, lambda: GaussianHMMRegime(2), plan)
    expected = int(np.ceil((len(switching) - plan.min_train) / plan.refit_every))
    assert len(reports) == expected
    assert all(isinstance(report, RegimeReport) for report in reports)


# ---------------------------------------------------------------------------
# Deterministic identity and canonical labelling
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("name", FITTED_MODELS)
def test_identity_is_deterministic_and_configuration_sensitive(
    name: str, switching: pd.Series
) -> None:
    training = switching.iloc[:400]
    first = CAUSAL_FACTORIES[name]().fit(training)
    second = CAUSAL_FACTORIES[name]().fit(training)
    assert first.identity == second.identity
    assert len(first.identity) == 64
    # A different training window is a different fitted model.
    other = CAUSAL_FACTORIES[name]().fit(switching.iloc[:500])
    assert first.identity != other.identity


def test_identity_separates_models_with_the_same_parameters(switching: pd.Series) -> None:
    training = switching.iloc[:400]
    gmm = GaussianMixtureRegime(2).fit(training)
    hmm = GaussianHMMRegime(2).fit(training)
    assert gmm.identity != hmm.identity


@pytest.mark.parametrize("n_states", [2, 3, 4])
@pytest.mark.parametrize("model", [GaussianMixtureRegime, GaussianHMMRegime])
def test_states_are_canonically_ordered_by_variance(
    model: type, n_states: int, switching: pd.Series
) -> None:
    """EM has no preferred labelling; the contract must impose one."""
    fitted = model(n_states).fit(switching.iloc[:600])
    variances = fitted.fitted_parameters()["variances"]
    assert variances == sorted(variances), variances
    assert len(fitted.state_labels) == n_states
    assert len(set(fitted.state_labels)) == n_states


def test_identity_before_fit_is_refused() -> None:
    with pytest.raises(NotFittedError):
        _ = GaussianHMMRegime(2).identity
    with pytest.raises(NotFittedError):
        GaussianHMMRegime(2).report()


def test_ordering_rule_is_published() -> None:
    for factory in CAUSAL_FACTORIES.values():
        assert factory().ordering_rule != "unspecified"


# ---------------------------------------------------------------------------
# Posterior and confidence outputs
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("name", sorted(CAUSAL_FACTORIES))
def test_posteriors_are_probability_distributions(
    name: str, switching: pd.Series, plan: ExpandingFitPlan
) -> None:
    states, _ = expanding_state_probabilities(switching, CAUSAL_FACTORIES[name], plan)
    observed = states.probabilities[states.observed]
    np.testing.assert_allclose(observed.sum(axis=1), 1.0, atol=1e-8)
    assert (observed >= -1e-12).all()


@pytest.mark.parametrize("name", sorted(CAUSAL_FACTORIES))
def test_confidence_and_entropy_are_derived_from_the_posterior(
    name: str, switching: pd.Series, plan: ExpandingFitPlan
) -> None:
    states, _ = expanding_state_probabilities(switching, CAUSAL_FACTORIES[name], plan)
    confidence = states.confidence().to_numpy()
    entropy = states.entropy().to_numpy()
    observed = states.observed
    assert np.all(confidence[observed] >= 0.5 - 1e-9)
    assert np.all(confidence[observed] <= 1.0 + 1e-9)
    assert np.all(entropy[observed] >= -1e-9)
    assert np.all(entropy[observed] <= 1.0 + 1e-9)
    assert np.isnan(confidence[~observed]).all()
    labels = states.hard_labels()
    assert set(labels.dropna().unique()) <= set(states.labels)
    assert labels[~observed].isna().all()


def test_entropy_is_maximal_for_a_uniform_posterior() -> None:
    index = pd.RangeIndex(10)
    states = RegimeStates(
        probabilities=np.tile([0.5, 0.5], (10, 1)),
        labels=("a", "b"),
        index=index,
        causal=True,
    )
    np.testing.assert_allclose(states.entropy().to_numpy(), 1.0)
    np.testing.assert_allclose(states.confidence().to_numpy(), 0.5)


def test_state_container_validates_its_inputs() -> None:
    index = pd.RangeIndex(4)
    with pytest.raises(RegimeError, match="sum to one"):
        RegimeStates(
            probabilities=np.full((4, 2), 0.9), labels=("a", "b"), index=index, causal=True
        )
    with pytest.raises(RegimeError, match="columns must match"):
        RegimeStates(
            probabilities=np.full((4, 3), 1 / 3), labels=("a", "b"), index=index, causal=True
        )
    with pytest.raises(RegimeError, match="rows must match"):
        RegimeStates(
            probabilities=np.full((5, 2), 0.5), labels=("a", "b"), index=index, causal=True
        )
    with pytest.raises(RegimeError, match="two-dimensional"):
        RegimeStates(probabilities=np.zeros(4), labels=("a",), index=index, causal=True)


# ---------------------------------------------------------------------------
# Synthetic recovery, no change, abrupt and gradual change
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("name", ["rule", "gmm", "hmm"])
def test_synthetic_regime_is_recovered(
    name: str, switching: pd.Series, plan: ExpandingFitPlan
) -> None:
    """The stress posterior must be higher inside the known stress window."""
    states, _ = expanding_state_probabilities(switching, CAUSAL_FACTORIES[name], plan)
    stress = states.probabilities[:, -1]
    inside = np.nanmean(stress[400:600])
    outside = np.nanmean(stress[600:])
    assert inside > outside + 0.2, f"{name}: inside={inside:.3f} outside={outside:.3f}"
    assert inside > 0.5


def test_no_change_series_produces_a_stable_regime(plan: ExpandingFitPlan) -> None:
    """A constant-volatility series must not manufacture persistent switching."""
    constant = _constant_series()
    states, _ = expanding_state_probabilities(constant, lambda: GaussianHMMRegime(2), plan)
    labels = states.hard_labels().dropna().to_numpy()
    flips = int(np.sum(labels[1:] != labels[:-1]))
    assert flips / max(labels.size - 1, 1) < 0.15


def test_abrupt_change_is_located_by_retrospective_segmentation() -> None:
    rng = np.random.default_rng(11)
    series = np.concatenate([rng.normal(0.0, 0.005, 300), rng.normal(0.0, 0.03, 300)])
    segmentation = segment_bayesian(series, max_change_points=3, penalty=20.0)
    assert segmentation.change_points.size >= 1
    nearest = int(np.min(np.abs(segmentation.change_points - 300)))
    assert nearest < 25, segmentation.change_points


def test_abrupt_change_raises_the_online_change_posterior() -> None:
    rng = np.random.default_rng(13)
    series = pd.Series(np.concatenate([rng.normal(0.0, 0.005, 300), rng.normal(0.0, 0.03, 300)]))
    model = BayesianOnlineChangePoint(hazard=1 / 200, short_run=15).fit(series.iloc[:250])
    states = model.filter(series)
    before = float(np.nanmean(states.probabilities[200:290, 1]))
    after = float(np.nanmax(states.probabilities[300:400, 1]))
    assert after > before


def test_gradual_change_is_detected_with_lag() -> None:
    """A ramp is harder than a step; detection is expected late, not absent."""
    rng = np.random.default_rng(17)
    ramp = np.concatenate([np.full(300, 0.005), np.linspace(0.005, 0.03, 300)])
    series = pd.Series(rng.normal(0.0, 1.0, 600) * ramp)
    model = RuleBasedRegime("volatility", quantile=0.7).fit(series.iloc[:300])
    states = model.filter(series)
    early = float(np.nanmean(states.probabilities[300:400, 1]))
    late = float(np.nanmean(states.probabilities[500:, 1]))
    assert late > early
    assert late > 0.5


def test_volatility_change_rule_leads_a_level_rule() -> None:
    """Expansion precedes the level a stress rule only confirms afterwards."""
    rng = np.random.default_rng(19)
    volatility = np.concatenate([np.full(300, 0.005), np.full(300, 0.03)])
    series = pd.Series(rng.normal(0.0, 1.0, 600) * volatility)
    statistic = rule_statistic(series, "volatility_change", window=10, long_window=60)
    assert np.isfinite(statistic.to_numpy()[320:360]).any()
    assert float(np.nanmean(statistic.to_numpy()[310:360])) > 0.0


# ---------------------------------------------------------------------------
# Degenerate input and non-convergence
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("name", FITTED_MODELS)
def test_short_series_are_refused(name: str) -> None:
    tiny = pd.Series(np.zeros(MIN_FIT_OBSERVATIONS - 1))
    with pytest.raises(RegimeError, match="at least"):
        CAUSAL_FACTORIES[name]().fit(tiny)


def test_rule_refuses_a_training_window_shorter_than_its_lookback() -> None:
    series = pd.Series(np.random.default_rng(2).normal(size=40))
    with pytest.raises(RegimeError, match="trailing window is longer"):
        RuleBasedRegime("volatility", window=35, long_window=60).fit(series)


@pytest.mark.parametrize("name", FITTED_MODELS)
def test_missing_observations_are_not_imputed(name: str, switching: pd.Series) -> None:
    model = CAUSAL_FACTORIES[name]().fit(switching.iloc[:400])
    gappy = switching.copy()
    gappy.iloc[500:510] = np.nan
    states = model.filter(gappy)
    assert np.isnan(states.probabilities[500:510]).all()
    assert states.observed[520:].any()


@pytest.mark.parametrize("model", [GaussianMixtureRegime, GaussianHMMRegime])
def test_exhausted_em_budget_is_reported_as_unconverged(model: type, switching: pd.Series) -> None:
    starved = model(2, max_iterations=1).fit(switching.iloc[:400])
    assert starved.report().converged is False
    assert starved.report().stopping_reason == "max_iterations"
    settled = model(2, max_iterations=500).fit(switching.iloc[:400])
    assert settled.report().converged is True
    assert settled.report().stopping_reason == "log_likelihood_tolerance"


def test_hmm_expected_durations_expose_a_degenerate_state(switching: pd.Series) -> None:
    """A state lasting ~1 bar is an outlier detector, not a regime."""
    model = GaussianHMMRegime(2).fit(switching.iloc[:400])
    durations = model.expected_durations()
    assert len(durations) == 2
    assert all(value >= 1.0 for value in durations)


@pytest.mark.parametrize(
    ("call", "message"),
    [
        (lambda: GaussianMixtureRegime(0), "n_states"),
        (lambda: GaussianMixtureRegime(MAX_STATES + 1), "n_states"),
        (lambda: GaussianMixtureRegime(2, max_iterations=0), "max_iterations"),
        (lambda: GaussianMixtureRegime(2, tolerance=0.0), "tolerance"),
        (lambda: GaussianHMMRegime(2, self_transition=1.0), "self_transition"),
        (lambda: CusumRegime(threshold=0.0), "threshold"),
        (lambda: CusumRegime(drift=-1.0), "drift"),
        (lambda: BayesianOnlineChangePoint(hazard=0.0), "hazard"),
        (lambda: BayesianOnlineChangePoint(short_run=0), "short_run"),
        (lambda: BayesianOnlineChangePoint(max_run_length=1), "max_run_length"),
        (lambda: RuleBasedRegime("volatility", quantile=0.0), "quantile"),
        (lambda: RuleBasedRegime("volatility", width=0.0), "width"),
        (lambda: ExpandingFitPlan(min_train=5), "min_train"),
        (lambda: ExpandingFitPlan(refit_every=0), "refit_every"),
    ],
)
def test_out_of_range_configuration_is_refused(call, message: str) -> None:
    with pytest.raises((RegimeError, ValueError), match=message):
        call()


def test_unknown_rule_is_refused() -> None:
    with pytest.raises(RegimeError, match="unknown rule"):
        rule_statistic(pd.Series(np.zeros(100)), "momentum")  # type: ignore[arg-type]


def test_report_invariants_fail_closed() -> None:
    fields: dict[str, Any] = {
        "model": "x",
        "n_states": 2,
        "iterations": 3,
        "max_iterations": 10,
        "converged": True,
        "stopping_reason": "closed_form",
        "n_train_observations": 100,
    }
    with pytest.raises(ValueError, match="at least one state"):
        RegimeReport(**{**fields, "n_states": 0})
    with pytest.raises(ValueError, match="within the configured budget"):
        RegimeReport(**{**fields, "iterations": 99})
    with pytest.raises(ValueError, match="log likelihood"):
        RegimeReport(**{**fields, "log_likelihood": float("nan")})


# ---------------------------------------------------------------------------
# Retrospective segmentation stays retrospective
# ---------------------------------------------------------------------------


def test_segmentation_is_never_causal() -> None:
    rng = np.random.default_rng(23)
    series = np.concatenate([rng.normal(0.0, 0.005, 200), rng.normal(0.0, 0.02, 200)])
    for segmentation in (
        segment_bayesian(series, max_change_points=2, penalty=20.0),
        segment_kernel(series, max_change_points=2, penalty=0.5),
    ):
        assert segmentation.causal is False
        assert not isinstance(segmentation, RegimeModel)
        labels = segmentation.segment_labels(series.size)
        assert labels.size == series.size
        assert labels[0] == 0


def test_kernel_segmentation_detects_a_distribution_change_at_equal_variance() -> None:
    """The case a Gaussian segmenter is blind to: same variance, different shape."""
    rng = np.random.default_rng(29)
    left = rng.normal(0.0, 1.0, 200)
    # A two-point mixture with matched mean and variance but very different shape.
    right = rng.choice([-1.0, 1.0], size=200).astype(float)
    series = np.concatenate([left / np.std(left), right])
    segmentation = segment_kernel(series, max_change_points=2, penalty=0.5)
    assert segmentation.change_points.size >= 1
    assert int(np.min(np.abs(segmentation.change_points - 200))) < 40


@pytest.mark.parametrize("segmenter", [segment_bayesian, segment_kernel])
def test_segmentation_refuses_unusable_input(segmenter) -> None:
    with pytest.raises(RegimeError, match="finite"):
        segmenter(np.array([1.0, np.nan, 3.0, 4.0]))
    with pytest.raises(RegimeError, match="at least four"):
        segmenter(np.array([1.0, 2.0]))
    with pytest.raises(RegimeError, match="max_change_points"):
        segmenter(np.zeros(100), max_change_points=0)
    with pytest.raises(RegimeError, match="penalty"):
        segmenter(np.zeros(100), penalty=-1.0)


def test_segmentation_refuses_an_oversized_series() -> None:
    with pytest.raises(RegimeError, match="ceiling"):
        segment_bayesian(np.zeros(6_000))


def test_segmentation_rejects_a_causal_claim() -> None:
    from alphaforge.regimes.changepoint import Segmentation

    with pytest.raises(RegimeError, match="never causal"):
        Segmentation(
            change_points=np.array([5], dtype=np.int64),
            method="x",
            cost=1.0,
            penalty=1.0,
            causal=True,
        )


def test_a_higher_penalty_yields_no_more_change_points() -> None:
    rng = np.random.default_rng(31)
    series = np.concatenate([rng.normal(0.0, 0.005, 200), rng.normal(0.0, 0.03, 200)])
    cheap = segment_bayesian(series, max_change_points=5, penalty=1.0)
    dear = segment_bayesian(series, max_change_points=5, penalty=5_000.0)
    assert dear.change_points.size <= cheap.change_points.size


# ---------------------------------------------------------------------------
# Incremental-value evidence
# ---------------------------------------------------------------------------


def test_forward_returns_are_strictly_forward() -> None:
    series = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
    ahead = forward_returns(series, horizon=1)
    np.testing.assert_allclose(ahead.to_numpy()[:-1], [2.0, 3.0, 4.0, 5.0])
    assert np.isnan(ahead.to_numpy()[-1])
    two = forward_returns(series, horizon=2)
    np.testing.assert_allclose(two.to_numpy()[:-2], [5.0, 7.0, 9.0])
    assert np.isnan(two.to_numpy()[-2:]).all()
    with pytest.raises(RegimeError, match="horizon"):
        forward_returns(series, horizon=0)


@pytest.mark.parametrize("name", sorted(CAUSAL_FACTORIES))
def test_evidence_is_produced_for_every_causal_model(
    name: str, switching: pd.Series, plan: ExpandingFitPlan
) -> None:
    states, _ = expanding_state_probabilities(switching, CAUSAL_FACTORIES[name], plan)
    evidence = evaluate_regime_value(states, switching, model_name=name)
    assert evidence.n_observations > 100
    assert 0.0 <= evidence.flip_rate <= 1.0
    assert evidence.mean_duration >= 1.0
    assert set(evidence.to_dict()) == {
        "model",
        "n_observations",
        "mse_delta",
        "brier_delta",
        "sharpe_delta",
        "max_drawdown_delta",
        "flip_rate",
        "mean_duration",
        "mean_entropy",
        "exposure_turnover",
    }


def test_a_regime_that_de_risks_the_stress_window_improves_drawdown(
    switching: pd.Series, plan: ExpandingFitPlan
) -> None:
    """Drawdown delta is positive when the worst loss gets shallower."""
    states, _ = expanding_state_probabilities(switching, lambda: GaussianHMMRegime(2), plan)
    evidence = evaluate_regime_value(states, switching, model_name="hmm")
    assert evidence.max_drawdown_delta > 0.0
    assert not evidence.improves_nothing()


def test_a_constant_posterior_adds_no_sizing_value(switching: pd.Series) -> None:
    """A regime carrying no information must not appear to help."""
    flat = RegimeStates(
        probabilities=np.tile([0.5, 0.5], (len(switching), 1)),
        labels=("calm", "stress"),
        index=switching.index,
        causal=True,
    )
    evidence = evaluate_regime_value(flat, switching, model_name="flat")
    assert evidence.sharpe_delta == pytest.approx(0.0, abs=1e-9)
    assert evidence.flip_rate == 0.0
    assert evidence.exposure_turnover == pytest.approx(0.0, abs=1e-12)


def test_evidence_refuses_non_causal_states(switching: pd.Series) -> None:
    retrospective = RegimeStates(
        probabilities=np.tile([0.5, 0.5], (len(switching), 1)),
        labels=("a", "b"),
        index=switching.index,
        causal=False,
    )
    with pytest.raises(RegimeError, match="causal"):
        evaluate_regime_value(retrospective, switching, model_name="bad")


def test_evidence_requires_alignment_and_enough_rows(switching: pd.Series) -> None:
    states = RegimeStates(
        probabilities=np.tile([0.5, 0.5], (len(switching), 1)),
        labels=("a", "b"),
        index=switching.index,
        causal=True,
    )
    with pytest.raises(RegimeError, match="share an index"):
        evaluate_regime_value(states, switching.reset_index(drop=True), model_name="x")
    with pytest.raises(RegimeError, match="stress_state"):
        evaluate_regime_value(states, switching, model_name="x", stress_state=5)
    short = switching.iloc[:20]
    tiny = RegimeStates(
        probabilities=np.tile([0.5, 0.5], (len(short), 1)),
        labels=("a", "b"),
        index=short.index,
        causal=True,
    )
    with pytest.raises(RegimeError, match="fewer than 30"):
        evaluate_regime_value(tiny, short, model_name="x")


def test_comparison_table_is_deterministic(switching: pd.Series, plan: ExpandingFitPlan) -> None:
    records = []
    for name in ("hmm", "gmm", "cusum"):
        states, _ = expanding_state_probabilities(switching, CAUSAL_FACTORIES[name], plan)
        records.append(evaluate_regime_value(states, switching, model_name=name))
    table = compare_regime_models(records)
    assert list(table["model"]) == sorted(record.model for record in records)
    again = compare_regime_models(list(reversed(records)))
    pd.testing.assert_frame_equal(table, again)
    with pytest.raises(RegimeError, match="empty evidence"):
        compare_regime_models([])


# ---------------------------------------------------------------------------
# Rule coverage across every statistic
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("rule", ["volatility", "trend", "drawdown", "volatility_change"])
def test_every_rule_produces_a_usable_regime(rule: str, switching: pd.Series) -> None:
    model = RuleBasedRegime(rule, window=21, long_window=63).fit(  # type: ignore[arg-type]
        switching.iloc[:400]
    )
    states = model.filter(switching)
    observed = states.probabilities[states.observed]
    assert observed.size > 0
    np.testing.assert_allclose(observed.sum(axis=1), 1.0, atol=1e-9)
    assert model.identity != ""


def test_drawdown_rule_inverts_its_direction() -> None:
    """A deeper drawdown is a lower number; the elevated state must follow it."""
    assert RuleBasedRegime("drawdown").elevated_above is False
    assert RuleBasedRegime("volatility").elevated_above is True


def test_rule_statistic_validates_its_windows() -> None:
    series = pd.Series(np.zeros(100))
    with pytest.raises(RegimeError, match="window must be at least two"):
        rule_statistic(series, "volatility", window=1)
    with pytest.raises(RegimeError, match="long_window must exceed"):
        rule_statistic(series, "volatility", window=21, long_window=21)


def test_rule_accepts_an_array_and_refuses_a_bare_array_at_filter(
    switching: pd.Series,
) -> None:
    model = RuleBasedRegime("volatility").fit(switching.iloc[:400].to_numpy())
    with pytest.raises(RegimeError, match="expects a pandas Series"):
        model.filter(switching.to_numpy())


def test_base_filter_refuses_a_bare_array(switching: pd.Series) -> None:
    model = GaussianHMMRegime(2).fit(switching.iloc[:400])
    with pytest.raises(RegimeError, match="expects a pandas Series"):
        model.filter(switching.to_numpy())


def test_bocpd_exposes_its_run_length_posterior(switching: pd.Series) -> None:
    model = BayesianOnlineChangePoint(max_run_length=120).fit(switching.iloc[:400])
    with pytest.raises(RegimeError, match="call filter"):
        model.run_length_posterior()
    model.filter(switching)
    posterior = model.run_length_posterior()
    assert posterior.shape == (len(switching), 120)
    rows = posterior[np.isfinite(posterior).all(axis=1)]
    np.testing.assert_allclose(rows.sum(axis=1), 1.0, atol=1e-8)


def test_cusum_declares_its_posterior_is_not_calibrated(switching: pd.Series) -> None:
    """A test statistic mapped to [0, 1] is not a probability, and says so."""
    model = CusumRegime().fit(switching.iloc[:400])
    assert model.fitted_parameters()["posterior_is_calibrated"] is False
