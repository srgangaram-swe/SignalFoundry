"""Offline aggregate study, causality, and publication tests for SF-S3-MR10."""

from __future__ import annotations

import json
import math
from dataclasses import FrozenInstanceError, replace
from pathlib import Path

import pandas as pd
import pytest
import yaml

import alphaforge.research.decision_policy_study as study_module
from alphaforge.config import ConfigValidationError, load_config
from alphaforge.decision import DecisionAction, DecisionPolicy
from alphaforge.research import (
    POLICY_NAMES,
    build_synthetic_decision_reference,
    evaluate_decision_policy_study,
    load_decision_policy_study_config,
    run_decision_policy_study,
)


def _config():
    return load_decision_policy_study_config("configs/decision_policy.yaml")


def test_committed_config_is_strict_bounded_and_immutable() -> None:
    config = _config()

    assert config.baselines == ("always_trade", "never_trade")
    assert config.observation_count == 2400
    assert config.observation_count <= config.thresholds.maximum_batch_size
    assert config.scope == "synthetic_engineering_only"
    assert config.protected_holdout_access is False
    assert config.broker_access is False
    assert config.study_id.startswith("decision-study-")
    with pytest.raises(FrozenInstanceError):
        config.seed = 1


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"observation_count": 100_001}, "observation_count"),
        ({"period_count": 1}, "period_count"),
        ({"unit_notional": math.inf}, "unit_notional"),
        ({"interpretation": ""}, "interpretation"),
        ({"protected_holdout_access": True}, "protected_holdout_access"),
        ({"broker_access": True}, "broker_access"),
    ],
)
def test_direct_study_config_construction_cannot_bypass_bounds(
    changes: dict[str, object], message: str
) -> None:
    with pytest.raises((TypeError, ValueError), match=message):
        replace(_config(), **changes)


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda payload: payload["policy"].update({"unknown": 1}), "unknown"),
        (
            lambda payload: payload["policy"].update(
                {"required_margin": payload["policy"]["maximum_absolute_expected_return"]}
            ),
            "required_margin",
        ),
        (
            lambda payload: payload["study"].update(
                {"observation_count": payload["policy"]["maximum_batch_size"] + 1}
            ),
            "maximum_batch_size",
        ),
        (
            lambda payload: payload["study"].update({"baselines": ["never_trade", "always_trade"]}),
            "baselines",
        ),
        (
            lambda payload: payload["study"].update({"anchor_time": "2026-07-26T12:00:00"}),
            "timezone",
        ),
    ],
)
def test_invalid_study_configuration_fails_before_work(
    tmp_path: Path, mutation: object, message: str
) -> None:
    payload = yaml.safe_load(Path("configs/decision_policy.yaml").read_text())
    mutation(payload)  # type: ignore[operator]
    path = tmp_path / "bad.yaml"
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")

    with pytest.raises(ConfigValidationError, match=message):
        load_config(path, "decision_policy")


def test_synthetic_reference_is_deterministic_bounded_and_input_label_separated() -> None:
    config = _config()
    first = build_synthetic_decision_reference(config)
    second = build_synthetic_decision_reference(config)

    assert first == second
    assert len(first) == config.observation_count
    assert max(item.period for item in first) < config.period_count
    signal_fields = set(first[0].signal.__dataclass_fields__)
    assert signal_fields.isdisjoint({"realized_return", "realized_cost", "target", "label"})
    assert all(item.realized_cost >= 0.0 for item in first)


def test_observation_values_and_periods_fail_before_aggregation() -> None:
    config = _config()
    observations = build_synthetic_decision_reference(config)

    with pytest.raises(ValueError, match="realized_return"):
        replace(observations[0], realized_return=math.nan)
    invalid_period = replace(observations[0], period=config.period_count)
    with pytest.raises(ValueError, match="period"):
        evaluate_decision_policy_study(
            (invalid_period, *observations[1:]),
            config,
        )


def test_realized_labels_cannot_change_policy_decisions_or_reason_counts() -> None:
    config = _config()
    observations = build_synthetic_decision_reference(config)
    counterfactual = tuple(
        replace(
            item,
            realized_return=-100.0 * item.realized_return + 0.5,
            realized_cost=item.realized_cost + 0.25,
        )
        for item in observations
    )
    policy = DecisionPolicy(config.thresholds)

    original_decisions = policy.evaluate_many(item.signal for item in observations)
    counterfactual_decisions = policy.evaluate_many(item.signal for item in counterfactual)
    _, original_reasons = evaluate_decision_policy_study(observations, config)
    _, counterfactual_reasons = evaluate_decision_policy_study(counterfactual, config)

    assert original_decisions == counterfactual_decisions
    assert original_reasons == counterfactual_reasons


def test_aggregate_study_reports_exact_baselines_and_tradeoffs() -> None:
    config = _config()
    observations = build_synthetic_decision_reference(config)
    metrics, reason_counts = evaluate_decision_policy_study(observations, config)
    by_policy = {item.policy: item for item in metrics}

    assert tuple(item.policy for item in metrics) == POLICY_NAMES
    assert by_policy["always_trade"].trade_count == config.observation_count
    assert by_policy["always_trade"].coverage == 1.0
    assert by_policy["always_trade"].missed_opportunity == 0.0
    assert by_policy["never_trade"].trade_count == 0
    assert by_policy["never_trade"].coverage == 0.0
    assert by_policy["never_trade"].selective_risk is None
    assert by_policy["never_trade"].mean_net_value == 0.0
    assert 0.0 < by_policy["abstention_policy"].coverage < 1.0
    assert (
        by_policy["abstention_policy"].mean_turnover_per_period
        < by_policy["always_trade"].mean_turnover_per_period
    )
    assert (
        by_policy["abstention_policy"].peak_capacity_demand
        < by_policy["always_trade"].peak_capacity_demand
    )
    assert reason_counts
    assert (
        sum(count for _, count in reason_counts) > by_policy["abstention_policy"].abstention_count
    )


def test_publisher_is_aggregate_only_json_safe_and_uses_seaborn(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    scatter_calls = 0
    bar_calls = 0
    original_scatter = study_module.plot_decision_policy_study.__globals__["sns"].scatterplot
    original_bar = study_module.plot_decision_policy_study.__globals__["sns"].barplot

    def track_scatter(*args: object, **kwargs: object):
        nonlocal scatter_calls
        scatter_calls += 1
        return original_scatter(*args, **kwargs)

    def track_bar(*args: object, **kwargs: object):
        nonlocal bar_calls
        bar_calls += 1
        return original_bar(*args, **kwargs)

    plot_globals = study_module.plot_decision_policy_study.__globals__["sns"]
    monkeypatch.setattr(plot_globals, "scatterplot", track_scatter)
    monkeypatch.setattr(plot_globals, "barplot", track_bar)
    result = run_decision_policy_study(_config(), tmp_path / "evidence")

    assert scatter_calls == 1
    assert bar_calls == 5
    assert {path.name for path in result.output_dir.iterdir()} == {
        "abstention_reasons.csv",
        "aggregate_metrics.csv",
        "decision_policy_study.png",
        "summary.json",
    }
    summary = json.loads((result.output_dir / "summary.json").read_text())
    assert summary["evidence"]["row_level_signals_published"] == 0
    assert summary["evidence"]["row_level_outcomes_published"] == 0
    assert summary["evidence"]["orders_emitted"] == 0
    assert summary["evidence"]["broker_access"] is False
    assert summary["mathematics"]["never_trade_selective_risk"] is None
    serialized = (result.output_dir / "summary.json").read_text()
    assert "NaN" not in serialized
    assert "Infinity" not in serialized
    public_columns = set(pd.read_csv(result.output_dir / "aggregate_metrics.csv"))
    assert public_columns.isdisjoint(
        {
            "signal_id",
            "model_id",
            "decision_time",
            "data_available_at",
            "expected_return",
            "realized_return",
            "realized_cost",
        }
    )
    assert not list(result.output_dir.rglob("*prediction*"))
    assert not list(result.output_dir.rglob("*order*"))


def test_replay_is_byte_deterministic_in_one_environment(tmp_path: Path) -> None:
    first = run_decision_policy_study(_config(), tmp_path / "first")
    second = run_decision_policy_study(_config(), tmp_path / "second")

    assert first.study_id == second.study_id
    assert first.policy_id == second.policy_id
    assert first.metrics == second.metrics
    assert first.reason_counts == second.reason_counts
    for name in (
        "aggregate_metrics.csv",
        "abstention_reasons.csv",
        "decision_policy_study.png",
        "summary.json",
    ):
        assert (first.output_dir / name).read_bytes() == (second.output_dir / name).read_bytes()


def test_failed_publication_removes_staging_and_never_partial_publishes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def fail_plot(*args: object, **kwargs: object) -> None:
        raise OSError("injected plot failure")

    monkeypatch.setattr(study_module, "plot_decision_policy_study", fail_plot)
    output = tmp_path / "evidence"

    with pytest.raises(OSError, match="injected"):
        run_decision_policy_study(_config(), output)

    assert not output.exists()
    assert not (tmp_path / ".publishing-evidence").exists()


def test_policy_selection_never_constructs_execution_objects() -> None:
    observations = build_synthetic_decision_reference(_config())[:8]
    decisions = DecisionPolicy(_config().thresholds).evaluate_many(
        item.signal for item in observations
    )

    assert all(decision.action in DecisionAction for decision in decisions)
    assert all(
        set(decision.to_dict()).isdisjoint({"quantity", "price", "venue", "account", "order_type"})
        for decision in decisions
    )
