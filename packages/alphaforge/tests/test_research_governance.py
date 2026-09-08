"""Adversarial tests for append-only research governance."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from alphaforge.research import (
    ROOT_TRIAL_ID,
    DuplicateTrialError,
    FrozenResearchPlan,
    IncompleteTrialFamilyError,
    KillCriterion,
    LedgerIntegrityError,
    MultipleTestingPolicy,
    ResearchGovernanceError,
    ResearchLedger,
    TrialSpec,
    adjust_p_values,
)

T0 = "2026-07-26T00:00:00Z"
T1 = "2026-07-26T00:00:01Z"
T2 = "2026-07-26T00:00:02Z"
T3 = "2026-07-26T00:00:03Z"
T4 = "2026-07-26T00:00:04Z"
T5 = "2026-07-26T00:00:05Z"


def _trials() -> tuple[TrialSpec, ...]:
    return (
        TrialSpec("naive", ROOT_TRIAL_ID, "naive", {"kind": "zero"}),
        TrialSpec("linear", "naive", "linear", {"alpha": 1.0}),
        TrialSpec("tree", "linear", "tree", {"depth": 3}),
    )


def _policy(method: str = "holm_bonferroni") -> MultipleTestingPolicy:
    return MultipleTestingPolicy(
        method=method,  # type: ignore[arg-type]
        alpha=0.05,
        family_size=3,
        assumptions=("complete frozen family", "dependence-aware raw tests"),
    )


def _plan(*, trials: tuple[TrialSpec, ...] | None = None) -> FrozenResearchPlan:
    return FrozenResearchPlan(
        hypothesis="Lagged price structure has incremental out-of-sample information.",
        mechanism="Slow information diffusion may create weak, decaying predictability.",
        dataset_id="a" * 64,
        features=("return_1", "momentum_20"),
        label="fwd_ret_5",
        test_plan={
            "primary_test": "moving_block_bootstrap_rank_ic",
            "holdout_access": "sealed_until_family_frozen",
        },
        validation={"scheme": "walk_forward", "embargo_sessions": 5},
        costs={"commission_bps": 1.0, "half_spread_bps": 2.5},
        uncertainty={"method": "moving_block_bootstrap", "block_length": 5},
        rejection_thresholds={"family_alpha": 0.05, "minimum_observations": 40},
        trials=_trials() if trials is None else trials,
        correction=_policy(),
        kill_criteria=(
            KillCriterion("no_incremental_ic", "incremental_rank_ic", "le", 0.0),
            KillCriterion("excessive_drawdown", "max_drawdown", "le", -0.25),
        ),
        frozen_at=T0,
    )


def _complete_success(
    ledger: ResearchLedger,
    trial_id: str,
    *,
    p_value: float,
    incremental_rank_ic: float = 0.02,
    max_drawdown: float = -0.10,
) -> None:
    ledger.register_trial(trial_id, occurred_at=T1)
    ledger.transition(
        trial_id,
        "STARTED",
        occurred_at=T1,
        details={"worker": "cpu", "attempt": 1},
    )
    ledger.transition(
        trial_id,
        "SUCCEEDED",
        occurred_at=T1,
        details={
            "p_value": p_value,
            "metrics": {
                "incremental_rank_ic": incremental_rank_ic,
                "max_drawdown": max_drawdown,
            },
        },
    )


def test_frozen_plan_is_deeply_immutable_and_identity_is_deterministic() -> None:
    plan = _plan()
    second = _plan()

    assert plan.plan_hash == second.plan_hash
    assert FrozenResearchPlan.from_dict(plan.to_dict()) == plan
    with pytest.raises(TypeError):
        plan.test_plan["primary_test"] = "changed"  # type: ignore[index]
    with pytest.raises(TypeError):
        plan.trials[0].configuration["kind"] = "changed"  # type: ignore[index]


def test_plan_rejects_incomplete_family_and_invalid_parent_lineage() -> None:
    with pytest.raises(ResearchGovernanceError, match="family_size"):
        _plan(trials=_trials()[:2])

    cyclic = (
        TrialSpec("a", "b", "a", {}),
        TrialSpec("b", "a", "b", {}),
        TrialSpec("c", ROOT_TRIAL_ID, "c", {}),
    )
    with pytest.raises(ResearchGovernanceError, match="cycle"):
        _plan(trials=cyclic)


def test_holm_and_bh_match_independent_numerical_references() -> None:
    p_values = {"naive": 0.01, "linear": 0.04, "tree": 0.03}
    eligible = ("naive", "linear", "tree")

    holm, holm_rejected = adjust_p_values(
        p_values,
        policy=_policy("holm_bonferroni"),
        eligible_trial_ids=eligible,
    )
    bh, bh_rejected = adjust_p_values(
        p_values,
        policy=_policy("benjamini_hochberg"),
        eligible_trial_ids=eligible,
    )

    assert holm == pytest.approx({"naive": 0.03, "linear": 0.06, "tree": 0.06})
    assert holm_rejected == {"naive": True, "linear": False, "tree": False}
    assert bh == pytest.approx({"naive": 0.03, "linear": 0.04, "tree": 0.04})
    assert bh_rejected == {"naive": True, "linear": True, "tree": True}


@pytest.mark.parametrize(
    "p_values",
    [
        {"naive": 0.01, "linear": 0.02},
        {"naive": 0.01, "linear": 0.02, "tree": 0.03, "extra": 0.04},
        {"naive": 0.01, "linear": float("nan"), "tree": 0.03},
        {"naive": 0.01, "linear": -0.1, "tree": 0.03},
    ],
)
def test_correction_fails_closed_on_missing_extra_or_invalid_evidence(
    p_values: dict[str, float],
) -> None:
    with pytest.raises(IncompleteTrialFamilyError):
        adjust_p_values(
            p_values,
            policy=_policy(),
            eligible_trial_ids=("naive", "linear", "tree"),
        )


def test_ledger_records_complete_family_failure_and_kill_decisions(tmp_path: Path) -> None:
    ledger = ResearchLedger.create(tmp_path / "ledger", _plan())
    _complete_success(ledger, "naive", p_value=0.01, incremental_rank_ic=-0.01)
    _complete_success(ledger, "linear", p_value=0.02)
    ledger.register_trial("tree", occurred_at=T1)
    ledger.transition(
        "tree",
        "FAILED",
        occurred_at=T2,
        details={"reason": "optional backend unavailable", "exception_type": "ImportError"},
    )

    evaluation = ledger.evaluate_family(occurred_at=T4)
    records = ledger.verify()

    assert evaluation.raw_p_values["tree"] == 1.0
    assert "trial_failed" in evaluation.killed["tree"]
    assert "no_incremental_ic" in evaluation.killed["naive"]
    assert evaluation.adjusted_p_values["linear"] == pytest.approx(0.04)
    assert records[-1]["event_type"] == "FAMILY_EVALUATED"

    with pytest.raises(ResearchGovernanceError, match="already been evaluated"):
        ledger.evaluate_family(occurred_at=T5)
    assert ledger.verify()[-1]["payload"]["reason"] == "family_already_evaluated"


def test_incomplete_evaluation_is_audited_before_failing(tmp_path: Path) -> None:
    ledger = ResearchLedger.create(tmp_path / "ledger", _plan())
    _complete_success(ledger, "naive", p_value=0.01)

    with pytest.raises(IncompleteTrialFamilyError, match="complete eligible family"):
        ledger.evaluate_family(occurred_at=T4)

    record = ledger.verify()[-1]
    assert record["event_type"] == "FAMILY_EVALUATION_REJECTED"
    assert record["payload"]["missing_or_nonterminal"] == ["linear", "tree"]


def test_interrupted_and_resumed_trial_remains_in_chain(tmp_path: Path) -> None:
    ledger = ResearchLedger.create(tmp_path / "ledger", _plan())
    ledger.register_trial("naive", occurred_at=T1)
    ledger.transition("naive", "STARTED", occurred_at=T2, details={"attempt": 1})
    ledger.transition(
        "naive",
        "INTERRUPTED",
        occurred_at=T3,
        details={"reason": "preempted", "checkpoint": "none"},
    )
    ledger.transition(
        "naive",
        "RESUMED",
        occurred_at=T4,
        details={"attempt": 2, "resume_from": "clean_start"},
    )
    ledger.transition(
        "naive",
        "SUCCEEDED",
        occurred_at=T5,
        details={
            "p_value": 0.01,
            "metrics": {"incremental_rank_ic": 0.01, "max_drawdown": -0.1},
        },
    )

    assert [record["event_type"] for record in ledger.verify()] == [
        "PLAN_FROZEN",
        "TRIAL_REGISTERED",
        "TRIAL_STARTED",
        "TRIAL_INTERRUPTED",
        "TRIAL_RESUMED",
        "TRIAL_SUCCEEDED",
    ]


def test_duplicate_registration_is_recorded_and_rejected(tmp_path: Path) -> None:
    ledger = ResearchLedger.create(tmp_path / "ledger", _plan())
    ledger.register_trial("naive", occurred_at=T1)

    with pytest.raises(DuplicateTrialError):
        ledger.register_trial("naive", occurred_at=T2)

    duplicate = ledger.verify()[-1]
    assert duplicate["event_type"] == "DUPLICATE_REJECTED"
    assert duplicate["payload"]["existing_status"] == "REGISTERED"


def test_hash_chain_and_receipt_detect_mutation_truncation_and_extension(tmp_path: Path) -> None:
    directory = tmp_path / "ledger"
    ledger = ResearchLedger.create(directory, _plan())
    ledger.register_trial("naive", occurred_at=T1)
    original = ledger.ledger_path.read_bytes()

    ledger.ledger_path.write_bytes(original.replace(b'"naive"', b'"other"', 1))
    with pytest.raises(LedgerIntegrityError, match="byte mutation"):
        ledger.verify()

    ledger.ledger_path.write_bytes(original.splitlines(keepends=True)[0])
    with pytest.raises(LedgerIntegrityError, match="truncation or extension"):
        ledger.verify()

    ledger.ledger_path.write_bytes(original + b"{}\n")
    with pytest.raises(LedgerIntegrityError, match="truncation or extension"):
        ledger.verify()


def test_receipt_mutation_wrong_plan_and_symlinks_fail_closed(tmp_path: Path) -> None:
    directory = tmp_path / "ledger"
    ledger = ResearchLedger.create(directory, _plan())
    head = json.loads(ledger.head_path.read_text(encoding="utf-8"))
    head["head_hash"] = "f" * 64
    ledger.head_path.write_text(json.dumps(head), encoding="utf-8")
    with pytest.raises(LedgerIntegrityError, match="ledger head"):
        ledger.verify()

    clean = ResearchLedger.create(tmp_path / "clean", _plan())
    with pytest.raises(LedgerIntegrityError, match="plan hash"):
        ResearchLedger.open(clean.directory, expected_plan_hash="f" * 64)

    symlink = tmp_path / "link"
    symlink.symlink_to(clean.directory, target_is_directory=True)
    with pytest.raises(LedgerIntegrityError, match="symbolic"):
        ResearchLedger.open(symlink)


def test_failed_head_publication_leaves_detectable_unreceipted_append(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    ledger = ResearchLedger.create(tmp_path / "ledger", _plan())

    def fail_head(record_count: int, head_hash: str) -> None:
        del record_count, head_hash
        raise OSError("simulated receipt failure")

    monkeypatch.setattr(ledger, "_write_head", fail_head)
    with pytest.raises(OSError, match="simulated"):
        ledger.register_trial("naive", occurred_at=T1)

    assert not ledger.lock_path.exists()
    with pytest.raises(LedgerIntegrityError, match="truncation or extension"):
        ledger.verify()


def test_ledger_resource_limits_and_invalid_transition_are_enforced(tmp_path: Path) -> None:
    with pytest.raises(LedgerIntegrityError, match="byte limit"):
        ResearchLedger.create(tmp_path / "tiny", _plan(), max_bytes=1)

    ledger = ResearchLedger.create(tmp_path / "ledger", _plan(), max_records=2)
    ledger.register_trial("naive", occurred_at=T1)
    with pytest.raises(LedgerIntegrityError, match="record limit"):
        ledger.transition("naive", "STARTED", occurred_at=T2, details={"attempt": 1})

    valid = ResearchLedger.create(tmp_path / "valid", _plan())
    with pytest.raises(ResearchGovernanceError, match="invalid trial transition"):
        valid.transition("naive", "STARTED", occurred_at=T1, details={"attempt": 1})


def test_nonfinite_metrics_are_rejected_before_append(tmp_path: Path) -> None:
    ledger = ResearchLedger.create(tmp_path / "ledger", _plan())
    ledger.register_trial("naive", occurred_at=T1)
    ledger.transition("naive", "STARTED", occurred_at=T2, details={"attempt": 1})

    with pytest.raises(ResearchGovernanceError, match="non-finite"):
        ledger.transition(
            "naive",
            "SUCCEEDED",
            occurred_at=T3,
            details={
                "p_value": 0.01,
                "metrics": {"incremental_rank_ic": float("nan")},
            },
        )

    assert ledger.verify()[-1]["event_type"] == "TRIAL_STARTED"


def test_backdated_event_is_rejected_without_corrupting_ledger(tmp_path: Path) -> None:
    ledger = ResearchLedger.create(tmp_path / "ledger", _plan())
    ledger.register_trial("naive", occurred_at=T2)

    with pytest.raises(ResearchGovernanceError, match="nondecreasing"):
        ledger.transition(
            "naive",
            "STARTED",
            occurred_at=T1,
            details={"attempt": 1},
        )

    assert ledger.verify()[-1]["event_type"] == "TRIAL_REGISTERED"
