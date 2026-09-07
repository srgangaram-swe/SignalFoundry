"""Tests for champion-challenger comparison and promotion governance (SF-S5-SL-MR5).

The invariants carrying the most weight:

* **No automation can approve, apply, roll back, or unfreeze.** Asserted by
  parsing the module AST, not by reading the code.
* **Gates are absolute.** A failed gate cannot be overridden or averaged away.
* **Pairing is exact and exclusions are reported**, so a comparison cannot
  quietly describe a different sample than it claims.
* **Inference is dependence-aware and multiplicity-corrected**, verified under
  null, known-effect, and low-power conditions.
"""

from __future__ import annotations

import ast
import inspect
import json
from datetime import UTC, datetime, timedelta
from typing import Any

import numpy as np
import pytest

from quant_platform.governance import comparison as comparison_module
from quant_platform.governance import gates as gates_module
from quant_platform.governance.comparison import (
    ComparisonError,
    LeakageError,
    PairedCohort,
    PairedScore,
    PairKey,
    build_paired_cohort,
    summarize_pairs,
)
from quant_platform.governance.gates import (
    APPROVAL_VALIDITY,
    Approval,
    Decision,
    FrozenPolicy,
    GovernanceError,
    NotAuthorizedError,
    Recommendation,
    authorize_apply,
    evaluate_gates,
    recommend,
)
from quant_platform.governance.inference import (
    MIN_BLOCKS,
    InferenceError,
    Margin,
    block_length,
    holm_adjust,
    non_inferiority_test,
    superiority_test,
)
from quant_platform.governance.inference import (
    TestResult as HypothesisTest,
)
from quant_platform.governance.inference import (
    TestVerdict as Verdict,
)
from quant_platform.shadow.contracts import (
    ProbabilityVector,
    ShadowForecast,
    ShadowOutcome,
)

BASE = datetime(2026, 8, 1, tzinfo=UTC)
DIGEST_A = "a" * 64
DIGEST_C = "c" * 64


def _forecast(symbol: str, day: int, up: float) -> ShadowForecast:
    as_of = BASE + timedelta(days=day)
    return ShadowForecast(
        campaign="lane-a",
        symbol=symbol,
        as_of=as_of,
        target_instant=as_of + timedelta(days=5),
        horizon_days=5,
        distribution=ProbabilityVector(labels=("down", "up"), probabilities=(1 - up, up)),
        feature_prefix_digest=DIGEST_C,
        model_identity=DIGEST_A,
    )


def _outcome(forecast: ShadowForecast, label: str) -> ShadowOutcome:
    observed = forecast.as_of + timedelta(days=5)
    return ShadowOutcome(
        forecast_id=forecast.forecast_id,
        realized_label=label,
        observed_at=observed,
        recorded_at=observed,
        revision=0,
    )


def _arms(
    days: int = 30, per_day: int = 5, champion_up: float = 0.6, challenger_up: float = 0.75
) -> tuple[list[ShadowForecast], dict, list[ShadowForecast], dict]:
    """Build two arms over the same cohort; the challenger is better calibrated."""
    rng = np.random.default_rng(11)
    champ, chal = [], []
    champ_out: dict[str, Any] = {}
    chal_out: dict[str, Any] = {}
    for day in range(days):
        for index in range(per_day):
            symbol = f"SYM{index:02d}"
            label = "up" if rng.random() < 0.75 else "down"
            c = _forecast(symbol, day, champion_up)
            g = _forecast(symbol, day, challenger_up)
            champ.append(c)
            chal.append(g)
            champ_out[c.forecast_id] = (_outcome(c, label),)
            chal_out[g.forecast_id] = (_outcome(g, label),)
    return champ, champ_out, chal, chal_out


def _policy(**overrides: Any) -> FrozenPolicy:
    """A policy at the real preregistered floor.

    The floor is deliberately not lowered here. A test suite that relaxes the
    thresholds to make its fixtures pass proves only that the relaxed gates
    work.
    """
    base: dict[str, Any] = {
        "version": "promotion-1",
        "alpha": 0.05,
        "margin": Margin(metric="brier", value=0.01),
    }
    base.update(overrides)
    return FrozenPolicy(**base)


def _floor_evidence(**overrides: Any) -> dict[str, Any]:
    """Per-class counts and power that clear the operational floor."""
    base: dict[str, Any] = {
        "class_counts": {"up": 160, "down": 80},
        "achieved_power": 0.86,
    }
    base.update(overrides)
    return base


def _recommend(
    cohort: PairedCohort,
    policy: FrozenPolicy,
    tests: tuple[HypothesisTest, ...],
    **overrides: Any,
) -> Decision:
    """Recommend with floor evidence supplied, as a real caller must."""
    now = overrides.pop("now", BASE)
    floor = _floor_evidence(
        **{k: overrides.pop(k) for k in ("class_counts", "achieved_power") if k in overrides}
    )
    return recommend(cohort, policy, tests, now=now, **floor, **overrides)


# ---------------------------------------------------------------------------
# No automation may approve
# ---------------------------------------------------------------------------


def test_no_override_parameter_exists_in_the_governance_modules() -> None:
    """A failed gate is removed in a new policy version, never waived."""
    forbidden = {"force", "waive", "skip", "override", "bypass", "unsafe", "ignore_gates"}
    for module in (gates_module, comparison_module):
        tree = ast.parse(inspect.getsource(module))
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                names = {arg.arg for arg in node.args.args + node.args.kwonlyargs}
                assert not (names & forbidden), f"{node.name} exposes {names & forbidden}"


def test_recommend_cannot_apply_anything() -> None:
    """The evaluating function must not also be able to act."""
    source = inspect.getsource(gates_module.recommend)
    for verb in ("apply", "promote_lane", "write", "commit", "execute"):
        assert f"{verb}(" not in source


def test_a_promote_decision_cannot_carry_a_failed_gate() -> None:
    with pytest.raises(GovernanceError, match="gates are"):
        Decision(
            recommendation=Recommendation.PROMOTE,
            policy_identity="p" * 64,
            cohort_identity="c" * 64,
            gates=(gates_module.GateResult(name="x", satisfied=False, detail="no"),),
            tests=(),
            correction={},
            decided_at=BASE.isoformat(),
        )


# ---------------------------------------------------------------------------
# Pairing and exclusions
# ---------------------------------------------------------------------------


def test_exact_pairing_matches_only_identical_questions() -> None:
    champ, champ_out, chal, chal_out = _arms(days=3, per_day=2)
    cohort = build_paired_cohort(champ, champ_out, chal, chal_out)
    assert cohort.matched == 6
    assert cohort.coverage == pytest.approx(1.0)


def test_unmatched_rows_are_counted_not_dropped() -> None:
    champ, champ_out, chal, chal_out = _arms(days=3, per_day=2)
    extra = _forecast("EXTRA", 0, 0.5)
    champ.append(extra)
    champ_out[extra.forecast_id] = (_outcome(extra, "up"),)
    cohort = build_paired_cohort(champ, champ_out, chal, chal_out)
    assert cohort.champion_only == 1
    assert cohort.coverage < 1.0


def test_a_duplicate_key_makes_pairing_ambiguous_and_is_refused() -> None:
    champ, champ_out, chal, chal_out = _arms(days=2, per_day=1)
    champ.append(champ[0])
    with pytest.raises(ComparisonError, match="ambiguous"):
        build_paired_cohort(champ, champ_out, chal, chal_out)


def test_arms_disagreeing_about_an_outcome_is_a_reconciliation_fault() -> None:
    """One question has one answer."""
    champ, champ_out, chal, chal_out = _arms(days=2, per_day=1)
    first = chal[0]
    chal_out[first.forecast_id] = (_outcome(first, "down"),)
    champ_out[champ[0].forecast_id] = (_outcome(champ[0], "up"),)
    with pytest.raises(ComparisonError, match="one answer"):
        build_paired_cohort(champ, champ_out, chal, chal_out)


def test_asymmetric_missingness_makes_a_cohort_incomparable() -> None:
    """A model absent on its bad days biases an otherwise valid paired sample."""
    champ, champ_out, chal, chal_out = _arms(days=20, per_day=5)
    for forecast in chal[:40]:
        chal_out.pop(forecast.forecast_id, None)
    cohort = build_paired_cohort(champ, champ_out, chal, chal_out)
    assert cohort.comparable is False
    assert "missingness differs" in (cohort.incomparable_reason or "")


def test_leakage_in_either_arm_is_refused() -> None:
    champ, champ_out, chal, chal_out = _arms(days=2, per_day=1)
    leaking = chal[0]
    chal_out[leaking.forecast_id] = (
        ShadowOutcome(
            forecast_id=leaking.forecast_id,
            realized_label="up",
            observed_at=leaking.as_of,  # not strictly after as_of
            recorded_at=leaking.as_of,
            revision=0,
        ),
    )
    with pytest.raises((LeakageError, Exception)):
        build_paired_cohort(champ, champ_out, chal, chal_out)


def test_an_incomparable_cohort_must_state_why() -> None:
    with pytest.raises(ComparisonError, match="must state why"):
        PairedCohort(
            pairs=(),
            champion_total=0,
            challenger_total=0,
            champion_only=0,
            challenger_only=0,
            champion_unscored=0,
            challenger_unscored=0,
            comparable=False,
            incomparable_reason=None,
        )


def test_pair_summary_reports_wins_losses_and_ties() -> None:
    champ, champ_out, chal, chal_out = _arms(days=5, per_day=4)
    summary = summarize_pairs(build_paired_cohort(champ, champ_out, chal, chal_out))
    assert summary["matched"] == 20
    assert summary["wins"] + summary["losses"] + summary["ties"] == 20


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------


def _synthetic_cohort(effect: float, days: int = 30, per_day: int = 5, seed: int = 0):
    rng = np.random.default_rng(seed)
    pairs = []
    for day in range(days):
        shock = rng.normal(0.0, 0.05)  # shared within-day dependence
        for index in range(per_day):
            difference = effect + shock + rng.normal(0.0, 0.02)
            as_of = BASE + timedelta(days=day)
            pairs.append(
                PairedScore(
                    key=PairKey(campaign_symbol=f"S{index}", as_of=as_of, horizon_days=5),
                    as_of_date=as_of.date(),
                    champion_brier=0.5,
                    challenger_brier=0.5 + difference,
                    champion_log=0.7,
                    challenger_log=0.7 + difference,
                )
            )
    return PairedCohort(
        pairs=tuple(pairs),
        champion_total=len(pairs),
        challenger_total=len(pairs),
        champion_only=0,
        challenger_only=0,
        champion_unscored=0,
        challenger_unscored=0,
        comparable=True,
        incomparable_reason=None,
    )


def test_a_known_effect_is_detected() -> None:
    result = superiority_test(_synthetic_cohort(-0.05, seed=1), seed=1)
    assert result.verdict is Verdict.FAVOURS_CHALLENGER
    assert result.point_estimate is not None and result.point_estimate < 0


def test_the_null_is_not_routinely_rejected() -> None:
    """Date-block resampling must not manufacture significance from noise."""
    positives = sum(
        1
        for seed in range(40)
        if superiority_test(_synthetic_cohort(0.0, seed=seed), seed=seed).verdict
        is Verdict.FAVOURS_CHALLENGER
    )
    assert positives <= 4  # 10% of 40 trials at alpha=0.05, generous for flake safety


def test_too_few_blocks_returns_underpowered_not_a_p_value() -> None:
    result = superiority_test(_synthetic_cohort(-0.05, days=MIN_BLOCKS - 1, seed=2), seed=2)
    assert result.verdict is Verdict.UNDERPOWERED
    assert result.p_value is None


def test_a_p_value_is_never_exactly_zero() -> None:
    """A finite resample cannot establish impossibility."""
    result = superiority_test(_synthetic_cohort(-0.5, seed=3), seed=3)
    assert result.p_value is not None and result.p_value > 0.0


def test_the_interval_is_deterministic() -> None:
    a = superiority_test(_synthetic_cohort(-0.02, seed=4), seed=9)
    b = superiority_test(_synthetic_cohort(-0.02, seed=4), seed=9)
    assert a.interval == b.interval and a.p_value == b.p_value


def test_an_incomparable_cohort_cannot_be_tested() -> None:
    cohort = PairedCohort(
        pairs=(),
        champion_total=1,
        challenger_total=1,
        champion_only=0,
        challenger_only=0,
        champion_unscored=0,
        challenger_unscored=0,
        comparable=False,
        incomparable_reason="asymmetric",
    )
    with pytest.raises(InferenceError, match="not comparable"):
        superiority_test(cohort)


def test_a_zero_margin_is_refused() -> None:
    """It would be a superiority test wearing the wrong name."""
    with pytest.raises(InferenceError, match="strictly positive"):
        Margin(metric="brier", value=0.0)


def test_non_inferiority_uses_the_declared_margin() -> None:
    cohort = _synthetic_cohort(0.005, seed=5)  # slightly worse than champion
    tight = non_inferiority_test(cohort, Margin(metric="brier", value=0.001), seed=5)
    loose = non_inferiority_test(cohort, Margin(metric="brier", value=0.1), seed=5)
    assert tight.verdict is Verdict.INCONCLUSIVE
    assert loose.verdict is Verdict.FAVOURS_CHALLENGER


def test_holm_is_conservative_against_uncorrected_alpha() -> None:
    """Two raw p-values below alpha must not both pass after correction."""
    correction = holm_adjust({"brier": 0.03, "log": 0.04}, alpha=0.05)
    assert correction["adjusted"]["brier"] > 0.03
    assert not any(correction["rejected"].values())


def test_holm_refuses_an_empty_family() -> None:
    with pytest.raises(InferenceError, match="complete and non-empty"):
        holm_adjust({})


def test_holm_refuses_a_non_finite_p_value() -> None:
    with pytest.raises(InferenceError, match="finite"):
        holm_adjust({"brier": float("nan")})


# ---------------------------------------------------------------------------
# Gates and recommendation
# ---------------------------------------------------------------------------


def _tests_favouring_challenger() -> tuple[HypothesisTest, ...]:
    return (
        HypothesisTest(
            name="superiority",
            metric="brier",
            verdict=Verdict.FAVOURS_CHALLENGER,
            point_estimate=-0.05,
            interval=(-0.08, -0.02),
            p_value=0.0001,
            blocks=30,
            observations=150,
            margin=None,
        ),
        HypothesisTest(
            name="non_inferiority",
            metric="brier",
            verdict=Verdict.FAVOURS_CHALLENGER,
            point_estimate=-0.05,
            interval=(None, -0.01),
            p_value=0.0001,
            blocks=30,
            observations=150,
            margin=0.01,
        ),
    )


def _sparse_cohort(*, dates: int, spacing_days: int) -> PairedCohort:
    """A cohort with a chosen number of target dates at a chosen spacing."""
    pairs = []
    for index in range(dates):
        as_of = BASE + timedelta(days=index * spacing_days)
        for slot in range(10):
            pairs.append(
                PairedScore(
                    key=PairKey(campaign_symbol=f"S{slot}", as_of=as_of, horizon_days=5),
                    as_of_date=as_of.date(),
                    champion_brier=0.5,
                    challenger_brier=0.45,
                    champion_log=0.7,
                    challenger_log=0.65,
                )
            )
    return PairedCohort(
        pairs=tuple(pairs),
        champion_total=len(pairs),
        challenger_total=len(pairs),
        champion_only=0,
        challenger_only=0,
        champion_unscored=0,
        challenger_unscored=0,
        comparable=True,
        incomparable_reason=None,
    )


def _passing_cohort() -> PairedCohort:
    """A cohort that clears every floor gate: 30 days, 30 target dates, 240 pairs."""
    return _synthetic_cohort(-0.05, days=30, per_day=8, seed=6)


def test_all_gates_are_reported_not_short_circuited() -> None:
    """A coverage failure must not hide behind a duration failure."""
    results = evaluate_gates(_synthetic_cohort(0.0, days=2, per_day=1), _policy())
    # Every gate reports, and each name appears once: a reader fixing one
    # failure must be able to see the others in the same record.
    names = [item.name for item in results]
    assert len(names) == len(set(names))
    assert sum(1 for item in results if not item.satisfied) >= 4


def test_an_unevaluated_gate_is_not_a_satisfied_gate() -> None:
    """Omitting the evidence a gate needs must fail it, never skip it."""
    results = {item.name: item for item in evaluate_gates(_passing_cohort(), _policy())}
    assert results["minimum_per_class"].satisfied is False
    assert "cannot be evaluated" in results["minimum_per_class"].detail
    assert results["minimum_power"].satisfied is False
    assert "cannot be evaluated" in results["minimum_power"].detail


def test_a_thin_class_fails_the_per_class_gate() -> None:
    """A class seen rarely cannot support a calibration claim about it."""
    results = {
        item.name: item
        for item in evaluate_gates(
            _passing_cohort(),
            _policy(),
            class_counts={"up": 230, "down": 10},
            achieved_power=0.9,
        )
    }
    assert results["minimum_per_class"].satisfied is False
    assert "'down'" in results["minimum_per_class"].detail


def test_distinct_days_and_calendar_span_are_gated_separately() -> None:
    """Twenty forecasts across a year are not twenty consecutive days."""
    scattered = _sparse_cohort(dates=25, spacing_days=14)
    results = {item.name: item for item in evaluate_gates(scattered, _policy())}
    assert results["minimum_target_dates"].satisfied is True
    assert results["minimum_duration"].satisfied is True
    dense = _sparse_cohort(dates=25, spacing_days=1)
    dense_results = {item.name: item for item in evaluate_gates(dense, _policy())}
    # 25 consecutive days is 25 calendar days, below the 28-day floor.
    assert dense_results["minimum_target_dates"].satisfied is True
    assert dense_results["minimum_duration"].satisfied is False


def test_a_failed_duration_gate_yields_insufficient_evidence() -> None:
    decision = _recommend(
        _synthetic_cohort(-0.05, days=5, per_day=5, seed=7),
        _policy(),
        _tests_favouring_challenger(),
        now=BASE,
    )
    assert decision.recommendation is Recommendation.INSUFFICIENT_EVIDENCE


def test_an_incomparable_cohort_yields_invalid() -> None:
    cohort = _passing_cohort()
    broken = PairedCohort(
        pairs=cohort.pairs,
        champion_total=cohort.champion_total,
        challenger_total=cohort.challenger_total,
        champion_only=0,
        challenger_only=0,
        champion_unscored=0,
        challenger_unscored=0,
        comparable=False,
        incomparable_reason="asymmetric missingness",
    )
    decision = _recommend(broken, _policy(), _tests_favouring_challenger(), now=BASE)
    assert decision.recommendation is Recommendation.INVALID


def test_an_underpowered_test_blocks_promotion() -> None:
    underpowered = HypothesisTest(
        name="superiority",
        metric="brier",
        verdict=Verdict.UNDERPOWERED,
        point_estimate=None,
        interval=None,
        p_value=None,
        blocks=3,
        observations=9,
        margin=None,
    )
    decision = _recommend(_passing_cohort(), _policy(), (underpowered,), now=BASE)
    assert decision.recommendation is Recommendation.INSUFFICIENT_EVIDENCE


def test_a_clearly_better_challenger_is_recommended() -> None:
    decision = _recommend(_passing_cohort(), _policy(), _tests_favouring_challenger(), now=BASE)
    assert decision.recommendation is Recommendation.PROMOTE
    assert "not an authorization" in decision.to_dict()["authority"]


def test_editing_the_policy_after_results_is_detected() -> None:
    policy = _policy()
    identity = policy.identity
    relaxed = _policy(minimum_days=1)
    with pytest.raises(GovernanceError, match="after seeing results"):
        _recommend(
            _passing_cohort(),
            relaxed,
            _tests_favouring_challenger(),
            now=BASE,
            expected_policy_identity=identity,
        )


def test_the_decision_serializes() -> None:
    payload = _recommend(
        _passing_cohort(), _policy(), _tests_favouring_challenger(), now=BASE
    ).to_dict()
    assert json.loads(json.dumps(payload))
    assert payload["correction"]["method"] == "holm_bonferroni"


# ---------------------------------------------------------------------------
# Human authority and compare-and-swap
# ---------------------------------------------------------------------------


def _approved(decision: Decision, **overrides: Any) -> Approval:
    base: dict[str, Any] = {
        "approver": "owner",
        "decision_identity": decision.identity,
        "approved_at": BASE,
        "reference": "approval-1",
    }
    base.update(overrides)
    return Approval(**base)


def test_a_promote_decision_with_current_approval_and_stable_head_applies() -> None:
    decision = _recommend(_passing_cohort(), _policy(), _tests_favouring_challenger(), now=BASE)
    authorize_apply(
        decision,
        _approved(decision),
        observed_lane_head="h" * 64,
        expected_lane_head="h" * 64,
        now=BASE,
    )


def test_a_non_promote_decision_cannot_be_applied() -> None:
    decision = _recommend(
        _synthetic_cohort(-0.05, days=5, per_day=5, seed=8),
        _policy(),
        _tests_favouring_challenger(),
        now=BASE,
    )
    with pytest.raises(NotAuthorizedError, match="cannot be overridden"):
        authorize_apply(
            decision,
            _approved(decision),
            observed_lane_head="h" * 64,
            expected_lane_head="h" * 64,
            now=BASE,
        )


def test_an_approval_cannot_be_recycled_onto_another_decision() -> None:
    first = _recommend(_passing_cohort(), _policy(), _tests_favouring_challenger(), now=BASE)
    other = _recommend(
        _synthetic_cohort(-0.06, days=30, per_day=8, seed=12),
        _policy(),
        _tests_favouring_challenger(),
        now=BASE,
    )
    with pytest.raises(NotAuthorizedError, match="recycled"):
        authorize_apply(
            other,
            _approved(first),
            observed_lane_head="h" * 64,
            expected_lane_head="h" * 64,
            now=BASE,
        )


def test_a_stale_approval_is_refused() -> None:
    decision = _recommend(_passing_cohort(), _policy(), _tests_favouring_challenger(), now=BASE)
    with pytest.raises(NotAuthorizedError, match="outside its"):
        authorize_apply(
            decision,
            _approved(decision),
            observed_lane_head="h" * 64,
            expected_lane_head="h" * 64,
            now=BASE + APPROVAL_VALIDITY + timedelta(days=1),
        )


def test_a_moved_lane_head_loses_the_race() -> None:
    """Two approvals racing cannot both promote."""
    decision = _recommend(_passing_cohort(), _policy(), _tests_favouring_challenger(), now=BASE)
    with pytest.raises(NotAuthorizedError, match="lane head moved"):
        authorize_apply(
            decision,
            _approved(decision),
            observed_lane_head="z" * 64,
            expected_lane_head="h" * 64,
            now=BASE,
        )


def test_an_anonymous_approver_is_refused() -> None:
    decision = _recommend(_passing_cohort(), _policy(), _tests_favouring_challenger(), now=BASE)
    with pytest.raises(GovernanceError, match="approver"):
        _approved(decision, approver="   ")


# ---------------------------------------------------------------------------
# Refusal paths
#
# Every guard below rejects a state that would otherwise produce a decision
# record describing something other than what happened. They are tested because
# a validation branch that never runs in a test is a validation branch that is
# assumed to work.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"version": "  "}, "version must be a non-empty string"),
        ({"alpha": 0.0}, "alpha must lie"),
        ({"alpha": 0.5}, "alpha must lie"),
        ({"minimum_days": 0}, "minimum_days must be a positive int"),
        ({"minimum_pairs": True}, "minimum_pairs must be a positive int"),
        ({"minimum_coverage": 0.0}, "minimum_coverage must lie"),
        ({"minimum_coverage": 1.5}, "minimum_coverage must lie"),
        ({"margin": 0.01}, "margin must be a Margin"),
    ],
)
def test_an_unusable_policy_threshold_is_refused(overrides: dict[str, Any], message: str) -> None:
    with pytest.raises(GovernanceError, match=message):
        _policy(**overrides)


@pytest.mark.parametrize(
    ("metric", "value", "message"),
    [
        ("  ", 0.01, "metric must be a non-empty string"),
        ("brier", "0.01", "value must be a real number"),
        ("brier", True, "value must be a real number"),
        ("brier", 0.0, "strictly positive"),
        ("brier", -0.01, "strictly positive"),
        ("brier", float("inf"), "finite"),
    ],
)
def test_an_unusable_margin_is_refused(metric: str, value: Any, message: str) -> None:
    with pytest.raises(InferenceError, match=message):
        Margin(metric=metric, value=value)


@pytest.mark.parametrize("alpha", [0.0, 0.5, 1.0, -0.1])
def test_both_tests_refuse_an_alpha_outside_the_open_unit_interval(alpha: float) -> None:
    cohort = _passing_cohort()
    with pytest.raises(InferenceError, match="alpha must lie"):
        superiority_test(cohort, alpha=alpha)
    with pytest.raises(InferenceError, match="alpha must lie"):
        non_inferiority_test(cohort, Margin(metric="brier", value=0.01), alpha=alpha)


def test_non_inferiority_refuses_an_incomparable_cohort() -> None:
    """Inference must not run on a sample the pairing already rejected."""
    champ, champ_out, chal, chal_out = _arms(days=20, per_day=5)
    for forecast in chal[:40]:
        chal_out.pop(forecast.forecast_id, None)
    cohort = build_paired_cohort(champ, champ_out, chal, chal_out)
    assert cohort.comparable is False
    with pytest.raises(InferenceError, match="not comparable"):
        non_inferiority_test(cohort, Margin(metric="brier", value=0.01))


def test_non_inferiority_reports_underpowered_rather_than_a_p_value() -> None:
    """Too few blocks yields a verdict, never a number that invites over-reading."""
    champ, champ_out, chal, chal_out = _arms(days=3, per_day=2)
    cohort = build_paired_cohort(champ, champ_out, chal, chal_out)
    result = non_inferiority_test(cohort, Margin(metric="brier", value=0.01))
    assert result.verdict is Verdict.UNDERPOWERED
    assert result.p_value is None
    assert result.interval is None
    assert result.blocks < MIN_BLOCKS


@pytest.mark.parametrize("value", ["0.05", True, None])
def test_holm_refuses_a_non_numeric_p_value(value: Any) -> None:
    with pytest.raises(InferenceError, match="must be a real number"):
        holm_adjust({"only": value})


def test_recommend_refuses_an_empty_test_family() -> None:
    with pytest.raises(GovernanceError, match="family must be non-empty"):
        _recommend(_passing_cohort(), _policy(), (), now=BASE)


def test_recommend_refuses_a_family_in_which_no_test_produced_a_p_value() -> None:
    """A family of all-None p-values is not a family that can be corrected."""
    silent = HypothesisTest(
        name="superiority",
        metric="brier",
        verdict=Verdict.FAVOURS_CHALLENGER,
        point_estimate=-0.02,
        interval=(-0.04, -0.01),
        p_value=None,
        blocks=30,
        observations=150,
        margin=None,
    )
    with pytest.raises(GovernanceError, match="no test produced a p-value"):
        _recommend(_passing_cohort(), _policy(), (silent,), now=BASE)


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"approver": "n" * 65}, "exceeds"),
        ({"reference": "r" * 65}, "exceeds"),
        ({"reference": " padded "}, "unpadded"),
        ({"decision_identity": "short"}, "full SHA-256 digest"),
        ({"approved_at": datetime(2026, 8, 1)}, "timezone-aware"),  # noqa: DTZ001
    ],
)
def test_a_malformed_approval_is_refused(overrides: dict[str, Any], message: str) -> None:
    decision = _recommend(_passing_cohort(), _policy(), _tests_favouring_challenger(), now=BASE)
    with pytest.raises(GovernanceError, match=message):
        _approved(decision, **overrides)


@pytest.mark.parametrize(
    ("decision", "approval", "message"),
    [
        ("not a decision", None, "must be a Decision"),
        (None, "not an approval", "must be an Approval"),
    ],
)
def test_authorize_apply_refuses_objects_it_did_not_produce(
    decision: Any, approval: Any, message: str
) -> None:
    """A duck-typed stand-in must not be able to walk through the authority gate."""
    real = _recommend(_passing_cohort(), _policy(), _tests_favouring_challenger(), now=BASE)
    with pytest.raises(NotAuthorizedError, match=message):
        authorize_apply(
            real if decision is None else decision,
            _approved(real) if approval is None else approval,
            observed_lane_head="h" * 64,
            expected_lane_head="h" * 64,
            now=BASE,
        )


def test_a_comparable_cohort_carrying_a_reason_is_a_contradiction() -> None:
    with pytest.raises(ComparisonError, match="cannot carry an incomparability reason"):
        PairedCohort(
            pairs=(),
            champion_total=0,
            challenger_total=0,
            champion_only=0,
            challenger_only=0,
            champion_unscored=0,
            challenger_unscored=0,
            comparable=True,
            incomparable_reason="but it said it was fine",
        )


def test_summarize_reports_an_empty_cohort_as_empty_rather_than_as_a_tie() -> None:
    """Zero pairs is not a mean difference of zero."""
    empty = PairedCohort(
        pairs=(),
        champion_total=0,
        challenger_total=0,
        champion_only=0,
        challenger_only=0,
        champion_unscored=0,
        challenger_unscored=0,
        comparable=True,
        incomparable_reason=None,
    )
    summary = summarize_pairs(empty)
    assert summary["matched"] == 0
    assert summary["mean_difference"] is None
    assert empty.coverage == 0.0


def test_an_unscored_champion_arm_is_counted_as_unscored() -> None:
    """A pair missing the champion's outcome reduces coverage, it does not vanish."""
    champ, champ_out, chal, chal_out = _arms(days=12, per_day=5)
    champ_out.pop(champ[0].forecast_id, None)
    cohort = build_paired_cohort(champ, champ_out, chal, chal_out)
    assert cohort.champion_unscored == 1
    assert cohort.matched == len(champ) - 1


def test_the_serialized_records_carry_their_reasoning() -> None:
    """to_dict is the reviewable artifact, so it must state what it is."""
    policy = _policy()
    decision = _recommend(_passing_cohort(), policy, _tests_favouring_challenger(), now=BASE)
    assert "no override" in policy.to_dict()["policy"]
    assert "not an authorization" in decision.to_dict()["authority"]
    approval = _approved(decision)
    assert approval.to_dict()["decision_identity"] == decision.identity
    assert json.loads(json.dumps(decision.to_dict(), allow_nan=False))


def test_a_cohort_above_the_pair_ceiling_is_refused(monkeypatch: pytest.MonkeyPatch) -> None:
    """The ceiling is a refusal threshold, not a tuning knob, so it must fire."""
    cohort = _passing_cohort()
    monkeypatch.setattr(comparison_module, "MAX_PAIRS", len(cohort.pairs) - 1)
    with pytest.raises(ComparisonError, match="pair ceiling"):
        PairedCohort(
            pairs=cohort.pairs,
            champion_total=cohort.champion_total,
            challenger_total=cohort.challenger_total,
            champion_only=0,
            challenger_only=0,
            champion_unscored=0,
            challenger_unscored=0,
            comparable=True,
            incomparable_reason=None,
        )


def test_the_cohort_summary_states_its_exclusions_and_its_identity() -> None:
    """Coverage below one must be visible in the artifact, not inferable from it."""
    champ, champ_out, chal, chal_out = _arms(days=12, per_day=5)
    extra = _forecast("EXTRA", 0, 0.5)
    champ.append(extra)
    champ_out[extra.forecast_id] = (_outcome(extra, "up"),)
    cohort = build_paired_cohort(champ, champ_out, chal, chal_out)
    summary = cohort.to_dict()
    assert summary["identity"] == cohort.identity()
    assert summary["champion_only"] == 1
    assert summary["coverage"] < 1.0
    assert summary["distinct_days"] == 12
    assert "reported rather than dropped" in summary["note"]
    assert json.loads(json.dumps(summary, allow_nan=False))


# ---------------------------------------------------------------------------
# Circular moving-block resampling
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("horizon", "dates", "expected"),
    [
        (5, 27, 5),  # ceil(27 ** 1/3) == 3, so the horizon dominates
        (1, 27, 3),  # cube root dominates a one-day horizon
        (1, 1000, 10),
        (20, 30, 20),  # a long horizon keeps long blocks on a short window
        (1, 1, 1),
    ],
)
def test_block_length_is_the_larger_of_horizon_and_cube_root(
    horizon: int, dates: int, expected: int
) -> None:
    """Both guarantees are kept rather than one traded for the other."""
    assert block_length(horizon_days=horizon, date_count=dates) == expected


@pytest.mark.parametrize(
    ("horizon", "dates"),
    [(0, 30), (-1, 30), (True, 30), (5, 0), (5, -1), (5, True)],
)
def test_an_unusable_block_length_request_is_refused(horizon: Any, dates: Any) -> None:
    with pytest.raises(InferenceError, match="must be a positive int"):
        block_length(horizon_days=horizon, date_count=dates)


def test_the_block_length_is_derived_not_accepted() -> None:
    """A caller who could choose it could choose the narrowest interval."""
    signature = inspect.signature(superiority_test)
    assert "block_length" not in signature.parameters
    assert "blocks" not in signature.parameters
    signature = inspect.signature(non_inferiority_test)
    assert "block_length" not in signature.parameters


def test_a_cohort_mixing_horizons_is_refused() -> None:
    """One block length cannot describe two horizons."""
    cohort = _passing_cohort()
    mixed = PairedCohort(
        pairs=(
            *cohort.pairs,
            PairedScore(
                key=PairKey(campaign_symbol="X", as_of=BASE, horizon_days=20),
                as_of_date=BASE.date(),
                champion_brier=0.5,
                challenger_brier=0.5,
                champion_log=0.7,
                challenger_log=0.7,
            ),
        ),
        champion_total=cohort.champion_total + 1,
        challenger_total=cohort.challenger_total + 1,
        champion_only=0,
        challenger_only=0,
        champion_unscored=0,
        challenger_unscored=0,
        comparable=True,
        incomparable_reason=None,
    )
    with pytest.raises(InferenceError, match="mixes forecast horizons"):
        superiority_test(mixed)


def test_resampling_is_circular_so_every_date_starts_a_block_equally_often() -> None:
    """A non-circular scheme under-samples the most recent evidence."""
    from quant_platform.governance import inference as inference_module

    dates = sorted({item.as_of_date for item in _passing_cohort().pairs})
    # One unit of signal on the final date only. Under a circular scheme the
    # last date appears in resamples as often as any other, so the replicate
    # mean centres on its true share; a scheme that could not start a block
    # there would systematically under-weight it.
    blocks = {day: [1.0 if day == dates[-1] else 0.0] for day in dates}
    replicates = inference_module._block_bootstrap(blocks, seed=1, replicates=4000, horizon_days=5)
    assert replicates.mean() == pytest.approx(1 / len(dates), abs=0.01)


def test_serially_dependent_days_widen_the_interval() -> None:
    """Moving blocks must preserve day-to-day dependence, not average it away."""
    from quant_platform.governance import inference as inference_module

    rng = np.random.default_rng(4)
    dates = sorted({item.as_of_date for item in _passing_cohort().pairs})
    level = 0.0
    persistent = {}
    for day in dates:
        level = 0.9 * level + rng.normal(0.0, 0.02)
        persistent[day] = [level]
    independent = {day: [rng.normal(0.0, 0.02)] for day in dates}
    spread_persistent = inference_module._block_bootstrap(
        persistent, seed=2, replicates=4000, horizon_days=5
    ).std()
    spread_independent = inference_module._block_bootstrap(
        independent, seed=2, replicates=4000, horizon_days=5
    ).std()
    assert spread_persistent > spread_independent


def test_the_bootstrap_is_reproducible_under_a_fixed_seed() -> None:
    cohort = _passing_cohort()
    first = superiority_test(cohort, seed=99)
    second = superiority_test(cohort, seed=99)
    assert first.point_estimate == second.point_estimate
    assert first.p_value == second.p_value
    assert first.interval == second.interval


@pytest.mark.parametrize("field", ["minimum_coverage", "minimum_power"])
@pytest.mark.parametrize("value", ["0.9", None, True])
def test_a_non_numeric_floor_fraction_is_refused(field: str, value: Any) -> None:
    with pytest.raises(GovernanceError, match="must be a real number"):
        _policy(**{field: value})


def test_an_empty_cohort_spans_no_calendar_days() -> None:
    """Zero pairs spans zero days, not one."""
    empty = PairedCohort(
        pairs=(),
        champion_total=0,
        challenger_total=0,
        champion_only=0,
        challenger_only=0,
        champion_unscored=0,
        challenger_unscored=0,
        comparable=True,
        incomparable_reason=None,
    )
    results = {item.name: item for item in evaluate_gates(empty, _policy())}
    assert results["minimum_duration"].satisfied is False
    assert "0 consecutive calendar days" in results["minimum_duration"].detail
