"""Adversarial and mathematical tests for the pre-portfolio decision policy."""

from __future__ import annotations

import json
import math
from dataclasses import FrozenInstanceError
from datetime import UTC, datetime, timedelta, tzinfo

import pytest

from alphaforge.decision import (
    DecisionAction,
    DecisionPolicy,
    DecisionReason,
    DecisionSignal,
    DecisionThresholds,
    RegimeSupport,
    SignalDirection,
)

NOW = datetime(2026, 7, 26, 12, 0, tzinfo=UTC)


class _UndefinedOffset(tzinfo):
    """Timezone object that is present structurally but defines no UTC offset."""

    def utcoffset(self, dt: datetime | None) -> None:
        return None

    def dst(self, dt: datetime | None) -> None:
        return None

    def tzname(self, dt: datetime | None) -> str:
        return "undefined"


def _thresholds(**overrides: object) -> DecisionThresholds:
    values: dict[str, object] = {
        "schema_version": "1.0.0",
        "required_margin": 0.0005,
        "cost_multiplier": 1.25,
        "cost_uncertainty_multiplier": 2.0,
        "uncertainty_penalty": 1.5,
        "maximum_total_cost": 0.003,
        "maximum_prediction_uncertainty": 0.002,
        "maximum_model_disagreement": 0.30,
        "maximum_regime_uncertainty": 0.35,
        "maximum_drift_score": 0.25,
        "maximum_data_age_seconds": 3600,
        "maximum_absolute_expected_return": 0.20,
        "maximum_batch_size": 128,
    }
    values.update(overrides)
    return DecisionThresholds(**values)  # type: ignore[arg-type]


def _signal(**overrides: object) -> DecisionSignal:
    values: dict[str, object] = {
        "signal_id": "signal-001",
        "model_id": "ensemble-v1",
        "decision_time": NOW,
        "data_available_at": NOW - timedelta(minutes=5),
        "expected_return": 0.010,
        "expected_cost": 0.0004,
        "cost_uncertainty": 0.0001,
        "prediction_uncertainty": 0.001,
        "model_disagreement": 0.10,
        "regime": RegimeSupport.SUPPORTED,
        "regime_uncertainty": 0.10,
        "drift_score": 0.05,
    }
    values.update(overrides)
    return DecisionSignal(**values)  # type: ignore[arg-type]


def test_reference_equation_is_conservative_and_strictly_margin_aware() -> None:
    policy = DecisionPolicy(_thresholds())
    decision = policy.evaluate(_signal())

    expected_cost = 1.25 * (0.0004 + 2.0 * 0.0001)
    expected_uncertainty_charge = 1.5 * 0.001
    expected_value = 0.010 - expected_cost - expected_uncertainty_charge

    assert decision.action is DecisionAction.TRADE
    assert decision.direction is SignalDirection.LONG
    assert decision.reasons == ()
    assert math.isclose(decision.conservative_cost or 0.0, expected_cost)
    assert math.isclose(decision.uncertainty_charge or 0.0, expected_uncertainty_charge)
    assert math.isclose(decision.penalized_expected_value or 0.0, expected_value)


@pytest.mark.parametrize(
    "field",
    [
        "expected_return",
        "expected_cost",
        "cost_uncertainty",
        "prediction_uncertainty",
        "model_disagreement",
        "regime_uncertainty",
        "drift_score",
    ],
)
@pytest.mark.parametrize("value", [math.nan, math.inf, -math.inf])
def test_every_nonfinite_estimate_abstains_with_json_safe_evidence(
    field: str, value: float
) -> None:
    decision = DecisionPolicy(_thresholds()).evaluate(_signal(**{field: value}))

    assert decision.action is DecisionAction.ABSTAIN
    assert decision.reasons[0] is DecisionReason.NON_FINITE_INPUT
    assert field in decision.failed_fields
    if field in {
        "expected_return",
        "expected_cost",
        "cost_uncertainty",
        "prediction_uncertainty",
    }:
        assert decision.penalized_expected_value is None
    assert "NaN" not in decision.to_json()
    assert "Infinity" not in decision.to_json()


def test_oversized_integer_estimate_normalizes_to_a_bounded_abstention() -> None:
    signal = _signal(expected_return=10**10_000)
    decision = DecisionPolicy(_thresholds()).evaluate(signal)

    assert signal.expected_return == math.inf
    assert decision.action is DecisionAction.ABSTAIN
    assert decision.reasons == (DecisionReason.NON_FINITE_INPUT,)
    assert decision.failed_fields == ("expected_return",)
    assert "Infinity" not in decision.to_json()


def test_invalid_estimate_does_not_hide_unrelated_applicable_gates() -> None:
    decision = DecisionPolicy(_thresholds()).evaluate(
        _signal(
            expected_return=math.nan,
            expected_cost=0.004,
            prediction_uncertainty=0.004,
            model_disagreement=0.60,
            regime=RegimeSupport.UNSUPPORTED,
            regime_uncertainty=0.80,
            drift_score=0.70,
        )
    )

    assert decision.action is DecisionAction.ABSTAIN
    assert decision.reasons == (
        DecisionReason.NON_FINITE_INPUT,
        DecisionReason.EXCESS_COST,
        DecisionReason.HIGH_DISAGREEMENT,
        DecisionReason.UNSUPPORTED_REGIME,
        DecisionReason.UNCERTAIN_REGIME,
        DecisionReason.DRIFT_DETECTED,
        DecisionReason.EXCESS_UNCERTAINTY,
    )
    assert decision.penalized_expected_value is None


def test_excess_cost_and_all_independent_gates_accumulate_in_stable_order() -> None:
    decision = DecisionPolicy(_thresholds()).evaluate(
        _signal(
            data_available_at=NOW - timedelta(hours=2),
            expected_return=0.004,
            expected_cost=0.004,
            cost_uncertainty=0.001,
            prediction_uncertainty=0.004,
            model_disagreement=0.60,
            regime=RegimeSupport.UNSUPPORTED,
            regime_uncertainty=0.80,
            drift_score=0.70,
        )
    )

    assert decision.action is DecisionAction.ABSTAIN
    assert decision.reasons == (
        DecisionReason.STALE_DATA,
        DecisionReason.EXCESS_COST,
        DecisionReason.HIGH_DISAGREEMENT,
        DecisionReason.UNSUPPORTED_REGIME,
        DecisionReason.UNCERTAIN_REGIME,
        DecisionReason.DRIFT_DETECTED,
        DecisionReason.EXCESS_UNCERTAINTY,
        DecisionReason.INSUFFICIENT_MARGIN,
    )


@pytest.mark.parametrize(
    ("overrides", "reason"),
    [
        (
            {"data_available_at": NOW + timedelta(microseconds=1)},
            DecisionReason.FUTURE_DATA,
        ),
        (
            {"model_disagreement": 0.3000001},
            DecisionReason.HIGH_DISAGREEMENT,
        ),
        (
            {"regime": RegimeSupport.UNSUPPORTED},
            DecisionReason.UNSUPPORTED_REGIME,
        ),
        ({"regime": RegimeSupport.UNKNOWN}, DecisionReason.UNSUPPORTED_REGIME),
        ({"regime_uncertainty": 0.3500001}, DecisionReason.UNCERTAIN_REGIME),
        ({"drift_score": 0.2500001}, DecisionReason.DRIFT_DETECTED),
        (
            {"prediction_uncertainty": 0.0020001},
            DecisionReason.EXCESS_UNCERTAINTY,
        ),
    ],
)
def test_each_gate_fails_closed(overrides: dict[str, object], reason: DecisionReason) -> None:
    decision = DecisionPolicy(_thresholds()).evaluate(_signal(**overrides))

    assert decision.action is DecisionAction.ABSTAIN
    assert reason in decision.reasons


def test_maximum_gate_equalities_pass_but_margin_equality_abstains() -> None:
    threshold = _thresholds(
        required_margin=0.125,
        cost_multiplier=1.0,
        cost_uncertainty_multiplier=1.0,
        uncertainty_penalty=1.0,
        maximum_total_cost=0.125,
        maximum_prediction_uncertainty=0.0,
        maximum_model_disagreement=0.30,
        maximum_regime_uncertainty=0.35,
        maximum_drift_score=0.25,
        maximum_absolute_expected_return=0.50,
    )
    decision = DecisionPolicy(threshold).evaluate(
        _signal(
            expected_return=0.25,
            expected_cost=0.125,
            cost_uncertainty=0.0,
            prediction_uncertainty=0.0,
            model_disagreement=0.30,
            regime_uncertainty=0.35,
            drift_score=0.25,
            data_available_at=NOW - timedelta(seconds=3600),
        )
    )

    assert decision.penalized_expected_value == threshold.required_margin
    assert decision.reasons == (DecisionReason.INSUFFICIENT_MARGIN,)
    assert decision.action is DecisionAction.ABSTAIN


def test_negative_return_is_a_short_research_direction_not_an_order() -> None:
    decision = DecisionPolicy(_thresholds()).evaluate(_signal(expected_return=-0.010))

    assert decision.action is DecisionAction.TRADE
    assert decision.direction is SignalDirection.SHORT
    assert set(decision.to_dict()).isdisjoint(
        {"quantity", "price", "venue", "account", "order_type"}
    )


def test_out_of_range_values_abstain_before_arithmetic() -> None:
    decision = DecisionPolicy(_thresholds()).evaluate(
        _signal(expected_return=0.21, expected_cost=-0.001, drift_score=1.1)
    )

    assert decision.action is DecisionAction.ABSTAIN
    assert decision.reasons == (DecisionReason.OUT_OF_BOUNDS_INPUT,)
    assert decision.penalized_expected_value is None
    assert decision.failed_fields == (
        "expected_return",
        "expected_cost",
        "drift_score",
    )


def test_extreme_finite_inputs_are_bounded_before_arithmetic() -> None:
    decision = DecisionPolicy(_thresholds()).evaluate(
        _signal(
            expected_cost=1.0e308,
            cost_uncertainty=1.0e308,
            prediction_uncertainty=1.0e308,
        )
    )

    assert decision.action is DecisionAction.ABSTAIN
    assert decision.reasons == (DecisionReason.OUT_OF_BOUNDS_INPUT,)
    assert decision.penalized_expected_value is None
    assert decision.failed_fields == (
        "expected_cost",
        "cost_uncertainty",
        "prediction_uncertainty",
    )


def test_cost_and_uncertainty_monotonicity() -> None:
    policy = DecisionPolicy(_thresholds())
    base = policy.evaluate(_signal())
    higher_cost = policy.evaluate(_signal(signal_id="costlier", expected_cost=0.0025))
    higher_uncertainty = policy.evaluate(
        _signal(signal_id="uncertain", prediction_uncertainty=0.003)
    )

    assert base.action is DecisionAction.TRADE
    assert (higher_cost.penalized_expected_value or 0.0) < (base.penalized_expected_value or 0.0)
    assert higher_uncertainty.action is DecisionAction.ABSTAIN
    assert DecisionReason.EXCESS_UNCERTAINTY in higher_uncertainty.reasons


def test_policy_decisions_are_immutable_deterministic_and_serializable() -> None:
    policy = DecisionPolicy(_thresholds())
    first = policy.evaluate(_signal())
    second = DecisionPolicy(_thresholds()).evaluate(_signal())

    assert first == second
    assert first.decision_id == second.decision_id
    assert first.policy_id == second.policy_id
    assert json.loads(first.to_json()) == first.to_dict()
    with pytest.raises(FrozenInstanceError):
        first.action = DecisionAction.ABSTAIN  # type: ignore[misc]
    with pytest.raises(FrozenInstanceError):
        policy.thresholds.required_margin = 0.0  # type: ignore[misc]
    with pytest.raises(FrozenInstanceError):
        policy.thresholds = _thresholds()  # type: ignore[misc]


def test_batch_output_and_identities_are_isolated_from_input_order() -> None:
    policy = DecisionPolicy(_thresholds())
    signals = [
        _signal(signal_id=f"signal-{index}", expected_return=0.005 + 0.001 * index)
        for index in range(8)
    ]

    forward = policy.evaluate_many(signals)
    reverse = policy.evaluate_many(reversed(signals))

    assert forward == reverse
    assert len({decision.decision_id for decision in forward}) == len(signals)


def test_batch_resource_and_uniqueness_bounds_fail_before_unbounded_work() -> None:
    policy = DecisionPolicy(_thresholds(maximum_batch_size=2))

    with pytest.raises(ValueError, match="maximum_batch_size"):
        policy.evaluate_many(_signal(signal_id=f"signal-{index}") for index in range(3))
    with pytest.raises(ValueError, match="duplicate"):
        policy.evaluate_many([_signal(), _signal()])


def test_batch_evaluation_operation_count_benchmark_and_prework_bound(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Count the linear evaluation phase and prove overflow fails before it."""
    policy = DecisionPolicy(_thresholds(maximum_batch_size=4096))
    original = DecisionPolicy.evaluate
    evaluation_calls = 0

    def counting_evaluate(self: DecisionPolicy, signal: DecisionSignal):
        nonlocal evaluation_calls
        evaluation_calls += 1
        return original(self, signal)

    monkeypatch.setattr(DecisionPolicy, "evaluate", counting_evaluate)
    for sample_count in (1, 64, 4096):
        evaluation_calls = 0
        decisions = policy.evaluate_many(
            _signal(signal_id=f"benchmark-{index:04d}") for index in range(sample_count)
        )
        assert len(decisions) == sample_count
        assert evaluation_calls == sample_count

    pulled = 0

    def oversized_batch():
        nonlocal pulled
        for index in range(4097):
            pulled += 1
            yield _signal(signal_id=f"overflow-{index:04d}")

    evaluation_calls = 0
    with pytest.raises(ValueError, match="maximum_batch_size"):
        policy.evaluate_many(oversized_batch())
    assert pulled == 4097
    assert evaluation_calls == 0


def test_structural_input_and_threshold_errors_are_rejected() -> None:
    with pytest.raises(ValueError, match="timezone-aware"):
        _signal(decision_time=datetime(2026, 7, 26))
    with pytest.raises(ValueError, match="timezone-aware"):
        _signal(decision_time=datetime(2026, 7, 26, tzinfo=_UndefinedOffset()))
    with pytest.raises(ValueError, match="safe ASCII"):
        _signal(signal_id="../unsafe")
    with pytest.raises(TypeError, match="RegimeSupport"):
        _signal(regime="supported")
    with pytest.raises(ValueError, match="below"):
        _thresholds(required_margin=0.20, maximum_absolute_expected_return=0.20)
