"""Operational-control tests for offline paper decisions."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import Any

import pytest

from alphaforge.paper import PaperControlState, PaperRiskLimits

NOW = datetime(2026, 7, 24, 21, tzinfo=UTC)


def _evaluate(state: PaperControlState, decision_id: str = "decision-1", **overrides):
    arguments: dict[str, Any] = {
        "decision_id": decision_id,
        "decision_time": NOW,
        "data_available_at": NOW - timedelta(hours=1),
        "target_weights": {"AAA": 0.10, "BBB": -0.10},
        "current_weights": {"AAA": 0.05, "BBB": -0.05},
        "equity": 1_000_000.0,
        "previous_equity": 1_000_000.0,
    }
    arguments.update(overrides)
    return state.evaluate(**arguments)


def test_safe_paper_decision_is_allowed_but_duplicate_is_halted() -> None:
    state = PaperControlState()

    first = _evaluate(state)
    duplicate = _evaluate(state)

    assert first.allowed
    assert not duplicate.allowed
    assert duplicate.reasons == ("duplicate_decision_id",)


def test_manual_kill_switch_is_one_way_for_state_lifetime() -> None:
    state = PaperControlState()
    state.activate_kill_switch()

    decision = _evaluate(state)

    assert not decision.allowed
    assert "manual_kill_switch" in decision.reasons


@pytest.mark.parametrize(
    ("overrides", "reason"),
    [
        ({"data_available_at": NOW + timedelta(minutes=1)}, "future_data_availability"),
        ({"data_available_at": NOW - timedelta(days=5)}, "stale_data"),
        ({"target_weights": {"AAA": 0.80, "BBB": 0.80}}, "gross_exposure_limit"),
        ({"target_weights": {"AAA": 0.20}}, "position_limit"),
        ({"target_weights": {"AAA": 0.10}, "current_weights": {"AAA": -0.60}}, "turnover_limit"),
        ({"equity": 900_000.0, "previous_equity": 1_000_000.0}, "daily_loss_limit"),
    ],
)
def test_limit_breaches_halt_with_machine_readable_reason(
    overrides: dict[str, Any],
    reason: str,
) -> None:
    decision = _evaluate(PaperControlState(), **overrides)

    assert not decision.allowed
    assert reason in decision.reasons


def test_drawdown_and_notional_limits_are_enforced_across_decisions() -> None:
    limits = PaperRiskLimits(maximum_notional=500_000.0)
    state = PaperControlState(limits)
    assert _evaluate(
        state,
        "peak",
        target_weights={"AAA": 0.10},
        equity=1_000_000.0,
    ).allowed

    decision = _evaluate(
        state,
        "drawdown",
        target_weights={"AAA": 0.70},
        current_weights={"AAA": 0.70},
        equity=800_000.0,
        previous_equity=850_000.0,
    )

    assert not decision.allowed
    assert {"notional_limit", "drawdown_limit"}.issubset(decision.reasons)


def test_invalid_inputs_fail_before_policy_evaluation() -> None:
    state = PaperControlState()
    with pytest.raises(ValueError, match="timezone-aware"):
        _evaluate(state, decision_time=NOW.replace(tzinfo=None))
    with pytest.raises(ValueError, match="non-empty"):
        _evaluate(PaperControlState(), decision_id="")
    with pytest.raises(ValueError, match="non-finite"):
        _evaluate(PaperControlState(), target_weights={"AAA": float("nan")})
