"""Deterministic evidence that the zero-capital paper boundary fails closed.

The audit exercises proposed state transitions only. It deliberately exposes
no broker, credential, network, or executable-order interface.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import Any

from alphaforge.paper.controls import PaperControlDecision, PaperControlState, PaperRiskLimits


def _record(name: str, decision: PaperControlDecision) -> dict[str, Any]:
    return {
        "scenario": name,
        "decision_id": decision.decision_id,
        "allowed": decision.allowed,
        "reasons": list(decision.reasons),
    }


def audit_offline_paper_controls(
    *,
    decision_time: datetime,
    maximum_notional: float,
) -> dict[str, Any]:
    """Exercise allow, idempotency, stale-data, risk, and kill-switch paths.

    Args:
        decision_time: Timezone-aware timestamp anchoring the deterministic audit.
        maximum_notional: Positive zero-capital shadow notional ceiling.

    Returns:
        JSON-compatible audit evidence with expected halt reasons.

    Raises:
        ValueError: If inputs violate the paper-control contract.
        RuntimeError: If any control does not produce its predeclared outcome.
    """
    if decision_time.tzinfo is None:
        raise ValueError("decision_time must be timezone-aware")
    anchor = decision_time.astimezone(UTC)
    limits = PaperRiskLimits(
        maximum_position_weight=0.25,
        maximum_notional=maximum_notional,
    )
    bounded_weights = {"LONG": 0.10, "SHORT": -0.10}
    current_weights = {"LONG": 0.05, "SHORT": -0.05}

    def evaluate(
        state: PaperControlState,
        *,
        decision_id: str,
        data_available_at: datetime,
        target_weights: dict[str, float] | None = None,
    ) -> PaperControlDecision:
        return state.evaluate(
            decision_id=decision_id,
            decision_time=anchor,
            data_available_at=data_available_at,
            target_weights=target_weights or bounded_weights,
            current_weights=current_weights,
            equity=maximum_notional,
            previous_equity=maximum_notional,
        )

    idempotency = PaperControlState(limits)
    allowed = evaluate(
        idempotency,
        decision_id="paper-audit-allow",
        data_available_at=anchor - timedelta(hours=1),
    )
    duplicate = evaluate(
        idempotency,
        decision_id="paper-audit-allow",
        data_available_at=anchor - timedelta(hours=1),
    )

    stale = evaluate(
        PaperControlState(limits),
        decision_id="paper-audit-stale",
        data_available_at=anchor - limits.maximum_data_age - timedelta(seconds=1),
    )
    risk = evaluate(
        PaperControlState(limits),
        decision_id="paper-audit-risk",
        data_available_at=anchor - timedelta(hours=1),
        target_weights={"LONG": 0.80, "SHORT": 0.80},
    )
    killed_state = PaperControlState(limits)
    killed_state.activate_kill_switch()
    killed = evaluate(
        killed_state,
        decision_id="paper-audit-killed",
        data_available_at=anchor - timedelta(hours=1),
    )

    records = [
        _record("bounded_proposal", allowed),
        _record("duplicate_decision", duplicate),
        _record("stale_data", stale),
        _record("risk_limit", risk),
        _record("manual_kill_switch", killed),
    ]
    expected = {
        "bounded_proposal": (True, set()),
        "duplicate_decision": (False, {"duplicate_decision_id"}),
        "stale_data": (False, {"stale_data"}),
        "risk_limit": (
            False,
            {"gross_exposure_limit", "net_exposure_limit"},
        ),
        "manual_kill_switch": (False, {"manual_kill_switch"}),
    }
    for record in records:
        expected_allowed, expected_reasons = expected[record["scenario"]]
        reasons = set(record["reasons"])
        # The risk scenario may trip additional bounded controls; the required
        # reasons must be present and every other scenario remains exact.
        reasons_match = (
            expected_reasons.issubset(reasons)
            if record["scenario"] == "risk_limit"
            else reasons == expected_reasons
        )
        if record["allowed"] is not expected_allowed or not reasons_match:
            raise RuntimeError(f"paper control audit failed for {record['scenario']}")

    return {
        "audit_schema_version": "1.0.0",
        "scope": "offline zero-capital proposed decisions only",
        "broker_adapter_present": False,
        "executable_orders_emitted": False,
        "all_controls_passed": True,
        "scenarios": records,
    }
