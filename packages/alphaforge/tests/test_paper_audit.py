"""Deterministic zero-capital paper-control audit tests."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from alphaforge.paper import audit_offline_paper_controls


def test_paper_audit_proves_fail_closed_controls_without_executable_orders() -> None:
    evidence = audit_offline_paper_controls(
        decision_time=datetime(2026, 7, 25, tzinfo=UTC),
        maximum_notional=1_000_000.0,
    )

    assert evidence["all_controls_passed"]
    assert evidence["broker_adapter_present"] is False
    assert evidence["executable_orders_emitted"] is False
    scenarios = {record["scenario"]: record for record in evidence["scenarios"]}
    assert scenarios["bounded_proposal"]["allowed"]
    assert scenarios["duplicate_decision"]["reasons"] == ["duplicate_decision_id"]
    assert scenarios["stale_data"]["reasons"] == ["stale_data"]
    assert "gross_exposure_limit" in scenarios["risk_limit"]["reasons"]
    assert scenarios["manual_kill_switch"]["reasons"] == ["manual_kill_switch"]


def test_paper_audit_rejects_naive_timestamp() -> None:
    with pytest.raises(ValueError, match="timezone-aware"):
        audit_offline_paper_controls(
            decision_time=datetime(2026, 7, 25),
            maximum_notional=1_000_000.0,
        )
