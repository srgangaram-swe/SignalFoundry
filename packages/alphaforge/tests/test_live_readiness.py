"""Negative and tabletop tests for the live-readiness gate (SF-S5-MR10).

This is the last gate before capital, and it is designed on the assumption that
whoever runs it wants it to pass. The tests are written the same way: most of
them try to get a READY verdict or an active configuration by a route that
should not work.

The guarantees carrying the most weight:

* **An empty evidence set is NOT_READY**, not vacuously ready.
* **A single unmet item blocks**, whatever the other sixteen say.
* **No override exists** — asserted by parsing the module AST.
* **An approval cannot be recycled** onto a different checklist or a larger cap.
* **Nothing can raise a cap** — there is no such method.
"""

from __future__ import annotations

import ast
import inspect
import json
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from typing import Any, cast

import pytest

from alphaforge.readiness import (
    ABSOLUTE_MAX_CAPITAL,
    ATTESTATION_ONLY,
    Attestation,
    CapitalAuthorization,
    CapitalError,
    Category,
    ChecklistItem,
    LiveCapitalConfig,
    NotAuthorizedError,
    ReadinessChecklist,
    ReadinessError,
    Verdict,
    activate,
    assert_within_limits,
    evaluate_readiness,
    minimal_capital_checklist,
    render_readiness_report,
)
from alphaforge.readiness import capital as capital_module
from alphaforge.readiness import checklist as checklist_module

NOW = datetime(2026, 8, 8, 12, 0, tzinfo=UTC)


def _all_satisfied() -> dict[str, bool]:
    return dict.fromkeys(minimal_capital_checklist().keys(), True)


def _attestations(at: datetime = NOW) -> dict[str, Attestation]:
    checklist = minimal_capital_checklist()
    return {
        item.key: Attestation(
            attester="owner",
            statement=f"reviewed: {item.requirement}",
            attested_at=at,
            reference="record-1",
        )
        for item in checklist.items
        if item.is_attestation_only
    }


def _ready_decision(now: datetime = NOW) -> Any:
    return evaluate_readiness(
        minimal_capital_checklist(),
        evidence=_all_satisfied(),
        attestations=_attestations(now),
        now=now,
    )


def _authorization(decision: Any, **kw: Any) -> CapitalAuthorization:
    base: dict[str, Any] = {
        "approver": "owner",
        "approved_cap": Decimal("500"),
        "readiness_identity": decision.checklist_identity,
        "approved_at": NOW,
        "expires_at": NOW + timedelta(days=7),
        "reference": "approval-1",
    }
    base.update(kw)
    return CapitalAuthorization(**base)


# ---------------------------------------------------------------------------
# The default is NOT_READY
# ---------------------------------------------------------------------------


def test_an_empty_evidence_set_is_not_ready() -> None:
    """Readiness is demonstrated, never inherited by omission."""
    decision = evaluate_readiness(minimal_capital_checklist(), evidence={}, now=NOW)
    assert decision.verdict is Verdict.NOT_READY
    assert not decision.ready
    assert len(decision.unmet) == len(minimal_capital_checklist().items)
    assert all("no evidence supplied" in r.detail for r in decision.results)


def test_the_current_state_of_this_repository_is_not_ready() -> None:
    """Sprint 5 closes with nothing qualified and no paper trading performed."""
    decision = evaluate_readiness(
        minimal_capital_checklist(),
        evidence={"qualified_candidate": False, "paper_duration": False},
        now=NOW,
    )
    assert decision.verdict is Verdict.NOT_READY
    assert "qualified_candidate" in decision.unmet
    assert decision.to_dict()["simulation_only"] is True


@pytest.mark.parametrize("missing", list(minimal_capital_checklist().keys()))
def test_any_single_unmet_item_blocks(missing: str) -> None:
    """Sixteen strong items cannot outvote one missing review."""
    evidence = _all_satisfied()
    evidence[missing] = False
    decision = evaluate_readiness(
        minimal_capital_checklist(), evidence=evidence, attestations=_attestations(), now=NOW
    )
    assert decision.verdict is Verdict.NOT_READY
    assert missing in decision.unmet


def test_a_fully_evidenced_candidate_is_ready() -> None:
    """The gate must be passable, or it teaches people to route around it."""
    decision = _ready_decision()
    assert decision.verdict is Verdict.READY_FOR_MINIMAL_CAPITAL
    assert decision.unmet == ()
    assert "not permanent" in decision.to_dict()["authorization"]


def test_a_decision_cannot_be_ready_while_carrying_unmet_items() -> None:
    from alphaforge.readiness.checklist import ItemResult, ReadinessDecision

    with pytest.raises(ReadinessError, match="cannot be READY"):
        ReadinessDecision(
            verdict=Verdict.READY_FOR_MINIMAL_CAPITAL,
            checklist_version="v",
            checklist_identity="a" * 64,
            results=(ItemResult(key="x", satisfied=False, category=Category.LEGAL, detail="no"),),
            unmet=("x",),
            decided_at=NOW.isoformat(),
        )


def test_a_not_ready_verdict_must_name_its_unmet_items() -> None:
    from alphaforge.readiness.checklist import ReadinessDecision

    with pytest.raises(ReadinessError, match="must name the unmet items"):
        ReadinessDecision(
            verdict=Verdict.NOT_READY,
            checklist_version="v",
            checklist_identity="a" * 64,
            results=(),
            unmet=(),
            decided_at=NOW.isoformat(),
        )


# ---------------------------------------------------------------------------
# No override exists
# ---------------------------------------------------------------------------


def test_no_override_parameter_exists_in_either_module() -> None:
    """An item that should not apply is removed in a new version, not waived."""
    forbidden = {
        "force",
        "waive",
        "skip",
        "override",
        "acknowledge_risk",
        "bypass",
        "unsafe",
        "ignore_unmet",
    }
    for module in (checklist_module, capital_module):
        tree = ast.parse(inspect.getsource(module))
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                names = {arg.arg for arg in node.args.args + node.args.kwonlyargs}
                assert not (
                    names & forbidden
                ), f"{module.__name__}.{node.name} exposes {names & forbidden}"


def test_there_is_no_method_that_raises_a_cap() -> None:
    """A mutable cap with a permission check is one bug from an unbounded one."""
    suspicious = [
        name
        for name in dir(LiveCapitalConfig)
        if any(word in name.lower() for word in ("raise", "increase", "expand", "set_cap", "grow"))
    ]
    assert suspicious == [], f"LiveCapitalConfig exposes {suspicious}"


def test_the_checklist_has_no_weighting() -> None:
    """A score would let strong items outvote a missing legal review."""
    item = minimal_capital_checklist().items[0]
    assert not hasattr(item, "weight")
    assert not hasattr(item, "score")
    assert "no score" in minimal_capital_checklist().to_dict()["policy"]


# ---------------------------------------------------------------------------
# Attestations are recorded, not verified
# ---------------------------------------------------------------------------


def test_policy_and_legal_items_require_a_named_dated_attestation() -> None:
    """A boolean does not record who decided what, and when."""
    decision = evaluate_readiness(
        minimal_capital_checklist(), evidence=_all_satisfied(), attestations={}, now=NOW
    )
    assert decision.verdict is Verdict.NOT_READY
    assert "employment_policy_review" in decision.unmet
    assert "legal_regulatory_review" in decision.unmet
    detail = next(r.detail for r in decision.results if r.key == "legal_regulatory_review")
    assert "named, dated attestation" in detail


def test_a_stale_attestation_is_not_current_consent() -> None:
    decision = evaluate_readiness(
        minimal_capital_checklist(),
        evidence=_all_satisfied(),
        attestations=_attestations(NOW - timedelta(days=400)),
        now=NOW,
    )
    assert decision.verdict is Verdict.NOT_READY
    detail = next(r.detail for r in decision.results if r.key == "legal_regulatory_review")
    assert "validity window" in detail


def test_a_future_dated_attestation_is_refused() -> None:
    attestation = Attestation(
        attester="owner", statement="reviewed", attested_at=NOW + timedelta(days=5)
    )
    assert not attestation.is_current(now=NOW, validity=timedelta(days=180))


def test_an_attestation_records_that_it_is_unverified() -> None:
    payload = Attestation(attester="owner", statement="reviewed", attested_at=NOW).to_dict()
    assert "Recorded, not verified" in payload["verification_note"]


def test_an_anonymous_attestation_is_refused() -> None:
    with pytest.raises(ReadinessError, match="attester"):
        Attestation(attester="  ", statement="reviewed", attested_at=NOW)


def test_a_naive_attestation_timestamp_is_refused() -> None:
    with pytest.raises(ReadinessError, match="timezone-aware"):
        Attestation(
            attester="owner", statement="x", attested_at=datetime(2026, 8, 8)  # noqa: DTZ001
        )


def test_only_policy_and_legal_are_attestation_only() -> None:
    assert frozenset({Category.POLICY, Category.LEGAL}) == ATTESTATION_ONLY


# ---------------------------------------------------------------------------
# Checklist integrity
# ---------------------------------------------------------------------------


def test_editing_the_checklist_to_admit_a_candidate_is_detected() -> None:
    checklist = minimal_capital_checklist()
    identity = checklist.identity
    trimmed = ReadinessChecklist(
        version=checklist.version,
        items=tuple(i for i in checklist.items if i.key != "legal_regulatory_review"),
    )
    with pytest.raises(ReadinessError, match="does not match the expected identity"):
        evaluate_readiness(trimmed, evidence={}, now=NOW, expected_identity=identity)


def test_the_identity_is_order_independent() -> None:
    checklist = minimal_capital_checklist()
    reversed_order = ReadinessChecklist(
        version=checklist.version, items=tuple(reversed(checklist.items))
    )
    assert reversed_order.identity == checklist.identity


def test_an_empty_checklist_is_refused() -> None:
    """It would make every candidate trivially ready."""
    with pytest.raises(ReadinessError, match="at least one item"):
        ReadinessChecklist(version="empty", items=())


def test_duplicate_keys_are_refused() -> None:
    item = ChecklistItem(key="dup", category=Category.EVIDENCE, requirement="r", verified_by="v")
    with pytest.raises(ReadinessError, match="unique"):
        ReadinessChecklist(version="dupes", items=(item, item))


def test_evidence_for_an_unknown_item_is_refused() -> None:
    """Usually means the wrong checklist version is in use."""
    with pytest.raises(ReadinessError, match="not in checklist"):
        evaluate_readiness(minimal_capital_checklist(), evidence={"invented_item": True}, now=NOW)


def test_non_boolean_evidence_is_refused() -> None:
    with pytest.raises(ReadinessError, match="must be a bool"):
        evaluate_readiness(
            minimal_capital_checklist(),
            evidence=cast(dict[str, bool], {"qualified_candidate": "yes"}),
            now=NOW,
        )


def test_the_checklist_covers_every_category() -> None:
    covered = {item.category for item in minimal_capital_checklist().items}
    assert covered == set(Category)


# ---------------------------------------------------------------------------
# Capital configuration is inert by default
# ---------------------------------------------------------------------------


def test_the_default_configuration_deploys_nothing() -> None:
    inert = LiveCapitalConfig.inert()
    assert inert.enabled is False
    assert inert.capital_cap == Decimal("0")
    assert inert.max_position_value() == Decimal("0")


def test_an_enabled_configuration_without_authorization_is_refused() -> None:
    """There is no path from 'no approval' to 'deploying capital'."""
    with pytest.raises(NotAuthorizedError, match="requires an authorization"):
        LiveCapitalConfig(
            enabled=True,
            capital_cap=Decimal("100"),
            max_position_fraction=Decimal("0.05"),
            max_daily_loss_fraction=Decimal("0.02"),
            max_drawdown_fraction=Decimal("0.05"),
            authorization=None,
        )


def test_activation_is_refused_when_not_ready() -> None:
    decision = evaluate_readiness(minimal_capital_checklist(), evidence={}, now=NOW)
    authorization = _authorization(decision)
    with pytest.raises(NotAuthorizedError, match="not a configuration problem"):
        activate(
            decision,
            authorization,
            capital_cap=Decimal("100"),
            max_position_fraction=Decimal("0.05"),
            max_daily_loss_fraction=Decimal("0.02"),
            max_drawdown_fraction=Decimal("0.05"),
            now=NOW,
        )


def test_an_approval_cannot_be_recycled_onto_a_different_checklist() -> None:
    decision = _ready_decision()
    foreign = _authorization(decision, readiness_identity="b" * 64)
    with pytest.raises(NotAuthorizedError, match="cannot be recycled"):
        activate(
            decision,
            foreign,
            capital_cap=Decimal("100"),
            max_position_fraction=Decimal("0.05"),
            max_daily_loss_fraction=Decimal("0.02"),
            max_drawdown_fraction=Decimal("0.05"),
            now=NOW,
        )


def test_an_expired_approval_is_refused() -> None:
    """An approval is a decision about a moment."""
    decision = _ready_decision()
    authorization = _authorization(decision)
    with pytest.raises(NotAuthorizedError, match="outside its window"):
        activate(
            decision,
            authorization,
            capital_cap=Decimal("100"),
            max_position_fraction=Decimal("0.05"),
            max_daily_loss_fraction=Decimal("0.02"),
            max_drawdown_fraction=Decimal("0.05"),
            now=NOW + timedelta(days=30),
        )


def test_a_cap_above_the_approval_is_refused() -> None:
    decision = _ready_decision()
    authorization = _authorization(decision, approved_cap=Decimal("100"))
    with pytest.raises(NotAuthorizedError, match="exceeds the approved cap"):
        activate(
            decision,
            authorization,
            capital_cap=Decimal("900"),
            max_position_fraction=Decimal("0.05"),
            max_daily_loss_fraction=Decimal("0.02"),
            max_drawdown_fraction=Decimal("0.05"),
            now=NOW,
        )


def test_an_approval_above_the_absolute_ceiling_is_refused() -> None:
    """A first deployment is an operational test; sizing it for return defeats that."""
    decision = _ready_decision()
    with pytest.raises(CapitalError, match="absolute ceiling"):
        _authorization(decision, approved_cap=ABSOLUTE_MAX_CAPITAL + Decimal("1"))


def test_an_over_long_approval_is_refused() -> None:
    decision = _ready_decision()
    with pytest.raises(CapitalError, match="more than 30 days"):
        _authorization(decision, expires_at=NOW + timedelta(days=365))


def test_a_duck_typed_decision_cannot_activate() -> None:
    class FakeDecision:
        ready = True
        verdict = Verdict.READY_FOR_MINIMAL_CAPITAL
        unmet: tuple[str, ...] = ()
        checklist_identity = "c" * 64

    with pytest.raises(NotAuthorizedError, match="not a\n?\\s*caller-constructed|stand-in"):
        activate(
            cast(Any, FakeDecision()),
            _authorization(_ready_decision()),
            capital_cap=Decimal("100"),
            max_position_fraction=Decimal("0.05"),
            max_daily_loss_fraction=Decimal("0.02"),
            max_drawdown_fraction=Decimal("0.05"),
            now=NOW,
        )


def test_a_valid_activation_produces_a_capped_configuration() -> None:
    decision = _ready_decision()
    config = activate(
        decision,
        _authorization(decision),
        capital_cap=Decimal("500"),
        max_position_fraction=Decimal("0.05"),
        max_daily_loss_fraction=Decimal("0.02"),
        max_drawdown_fraction=Decimal("0.05"),
        now=NOW,
    )
    assert config.enabled
    assert config.capital_cap == Decimal("500")
    assert config.max_position_value() == Decimal("25.00")
    assert "no method that raises a cap" in config.to_dict()["expansion_policy"]


def test_deactivation_is_total_and_reversible() -> None:
    decision = _ready_decision()
    config = activate(
        decision,
        _authorization(decision),
        capital_cap=Decimal("500"),
        max_position_fraction=Decimal("0.05"),
        max_daily_loss_fraction=Decimal("0.02"),
        max_drawdown_fraction=Decimal("0.05"),
        now=NOW,
    )
    inert = config.deactivated()
    assert inert.enabled is False
    assert inert.capital_cap == Decimal("0")
    assert inert.authorization is None


def test_a_float_cap_is_refused() -> None:
    with pytest.raises(CapitalError, match="not a float"):
        LiveCapitalConfig(
            enabled=False,
            capital_cap=cast(Decimal, 100.5),
            max_position_fraction=Decimal("0.05"),
            max_daily_loss_fraction=Decimal("0.02"),
            max_drawdown_fraction=Decimal("0.05"),
        )


@pytest.mark.parametrize(
    "field,value",
    [
        ("max_position_fraction", Decimal("0.9")),
        ("max_daily_loss_fraction", Decimal("0.9")),
        ("max_drawdown_fraction", Decimal("0.9")),
    ],
)
def test_a_limit_too_loose_to_impose_anything_is_refused(field: str, value: Decimal) -> None:
    kwargs: dict[str, Any] = {
        "enabled": False,
        "capital_cap": Decimal("100"),
        "max_position_fraction": Decimal("0.05"),
        "max_daily_loss_fraction": Decimal("0.02"),
        "max_drawdown_fraction": Decimal("0.05"),
    }
    kwargs[field] = value
    with pytest.raises(CapitalError, match="imposes\n?\\s*nothing|exceeds"):
        LiveCapitalConfig(**kwargs)


def test_a_zero_limit_is_refused() -> None:
    with pytest.raises(CapitalError, match="must be positive"):
        LiveCapitalConfig(
            enabled=False,
            capital_cap=Decimal("100"),
            max_position_fraction=Decimal("0"),
            max_daily_loss_fraction=Decimal("0.02"),
            max_drawdown_fraction=Decimal("0.05"),
        )


# ---------------------------------------------------------------------------
# Runtime limit enforcement
# ---------------------------------------------------------------------------


def _active() -> LiveCapitalConfig:
    decision = _ready_decision()
    return activate(
        decision,
        _authorization(decision),
        capital_cap=Decimal("500"),
        max_position_fraction=Decimal("0.05"),
        max_daily_loss_fraction=Decimal("0.02"),
        max_drawdown_fraction=Decimal("0.05"),
        now=NOW,
    )


def test_an_inert_configuration_permits_no_exposure() -> None:
    with pytest.raises(NotAuthorizedError, match="inert"):
        assert_within_limits(
            LiveCapitalConfig.inert(),
            position_value=Decimal("1"),
            daily_loss=Decimal("0"),
            drawdown=Decimal("0"),
        )


def test_exposure_within_limits_passes() -> None:
    assert_within_limits(
        _active(), position_value=Decimal("20"), daily_loss=Decimal("5"), drawdown=Decimal("10")
    )


@pytest.mark.parametrize(
    "kwargs,expected",
    [
        ({"position_value": Decimal("100")}, "position value"),
        ({"daily_loss": Decimal("50")}, "daily loss"),
        ({"drawdown": Decimal("100")}, "drawdown"),
    ],
)
def test_a_breach_is_refused_before_an_order_is_built(
    kwargs: dict[str, Decimal], expected: str
) -> None:
    base = {
        "position_value": Decimal("1"),
        "daily_loss": Decimal("0"),
        "drawdown": Decimal("0"),
    }
    base.update(kwargs)
    with pytest.raises(CapitalError, match=expected):
        assert_within_limits(_active(), **base)


def test_multiple_breaches_are_all_named() -> None:
    with pytest.raises(CapitalError, match="3 limit"):
        assert_within_limits(
            _active(),
            position_value=Decimal("400"),
            daily_loss=Decimal("400"),
            drawdown=Decimal("400"),
        )


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------


def test_the_report_leads_with_failures() -> None:
    """A report opening with passes invites skimming to an unsupported conclusion."""
    decision = evaluate_readiness(minimal_capital_checklist(), evidence={}, now=NOW)
    report = render_readiness_report(decision)
    assert report.index("## Unmet") < len(report)
    assert "blocks deployment" in report
    assert "NOT_READY" in report


def test_the_report_shows_both_sections_when_partially_satisfied() -> None:
    evidence = _all_satisfied()
    evidence["legal_regulatory_review"] = False
    decision = evaluate_readiness(
        minimal_capital_checklist(), evidence=evidence, attestations=_attestations(), now=NOW
    )
    report = render_readiness_report(decision)
    assert report.index("## Unmet") < report.index("## Satisfied")


def test_the_decision_is_machine_readable() -> None:
    decision = evaluate_readiness(minimal_capital_checklist(), evidence={}, now=NOW)
    payload = json.loads(json.dumps(decision.to_dict()))
    assert payload["verdict"] == "NOT_READY"
    assert payload["ready"] is False
    assert payload["unmet_by_category"]


def test_unmet_items_group_by_category() -> None:
    """Remedies differ by kind: an evidence gap is closed by running something."""
    grouped = evaluate_readiness(
        minimal_capital_checklist(), evidence={}, now=NOW
    ).unmet_by_category()
    assert "legal" in grouped
    assert "evidence" in grouped


def test_the_configuration_is_auditable_by_identity() -> None:
    config = _active()
    assert len(config.identity()) == 64
    assert config.identity() == _active().identity()
    assert config.identity() != config.deactivated().identity()
