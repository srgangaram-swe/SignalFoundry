"""Pre-registered seven-candidate Sprint 2 baseline study.

The study freezes the complete candidate family and all research policy before
dispatching the existing development/final-holdout workflow. Development
comparisons use matched walk-forward folds; the selected candidate alone
crosses the final-holdout boundary. Every candidate remains in the append-only
ledger, including failures, before Holm-Bonferroni correction is evaluated.
"""

from __future__ import annotations

import json
import math
import os
import tempfile
from collections.abc import Callable
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

from alphaforge.data import SignalFoundryDataset
from alphaforge.evaluation import ReadinessThresholds
from alphaforge.models.registry import seed_model_specs
from alphaforge.research.governance import (
    ROOT_TRIAL_ID,
    FrozenResearchPlan,
    KillCriterion,
    MultipleTestingPolicy,
    ResearchLedger,
    TrialSpec,
)
from alphaforge.research.signal_foundry import (
    GovernedResearchConfig,
    GovernedResearchResult,
    run_governed_signal_foundry_research,
)

SPRINT_2_CANDIDATES = (
    "momentum_baseline",
    "linear",
    "random_forest",
    "lightgbm",
    "xgboost",
    "catboost",
    "small_mlp",
)
NAIVE_CANDIDATE = SPRINT_2_CANDIDATES[0]
STUDY_SCHEMA_VERSION = "1.0.0"


@dataclass(frozen=True)
class BaselineStudyResult:
    """Immutable identities and decision for one completed governed study."""

    study_id: str
    study_dir: Path
    research_run: GovernedResearchResult
    decision: str


def _timestamp(clock: Callable[[], datetime]) -> str:
    value = clock()
    if value.tzinfo is None:
        raise ValueError("study clock must return a timezone-aware datetime")
    return value.astimezone(UTC).isoformat().replace("+00:00", "Z")


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    encoded = (
        json.dumps(payload, sort_keys=True, indent=2, ensure_ascii=True, allow_nan=False) + "\n"
    )
    temporary: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary = handle.name
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        if temporary is not None:
            Path(temporary).unlink(missing_ok=True)
        raise


def _validate_family(model_specs: list[dict[str, Any]]) -> None:
    names = tuple(str(spec.get("name", "")) for spec in model_specs)
    if names != SPRINT_2_CANDIDATES:
        raise ValueError(
            "Sprint 2 candidates must be frozen in order as "
            f"{list(SPRINT_2_CANDIDATES)}; received {list(names)}"
        )
    if any(not isinstance(spec.get("params", {}), dict) for spec in model_specs):
        raise TypeError("every candidate parameter set must be a mapping")


def _plan(
    *,
    dataset: SignalFoundryDataset,
    model_specs: list[dict[str, Any]],
    feature_config: dict[str, Any],
    walk_forward_config: dict[str, Any],
    backtest_config: dict[str, Any],
    research_config: GovernedResearchConfig,
    readiness_thresholds: ReadinessThresholds,
    governance_config: dict[str, Any],
    frozen_at: str,
) -> FrozenResearchPlan:
    correction = dict(governance_config["correction"])
    criteria = tuple(
        KillCriterion(
            name=str(item["name"]),
            metric=str(item["metric"]),
            operator=item["operator"],
            threshold=float(item["threshold"]),
        )
        for item in governance_config["kill_criteria"]
    )
    trial_ids = {
        name: f"{index:02d}-{name}" for index, name in enumerate(SPRINT_2_CANDIDATES, start=1)
    }
    trials = tuple(
        TrialSpec(
            trial_id=trial_ids[str(spec["name"])],
            parent_trial_id=(
                ROOT_TRIAL_ID if spec["name"] == NAIVE_CANDIDATE else trial_ids[NAIVE_CANDIDATE]
            ),
            candidate=str(spec["name"]),
            configuration=dict(spec.get("params", {})),
        )
        for spec in model_specs
    )
    return FrozenResearchPlan(
        hypothesis=(
            "At least one conventional model has positive development rank IC, positive "
            "costed annual return, and survives multiplicity correction relative to the "
            "predeclared momentum baseline."
        ),
        mechanism=(
            "Causal technical, cross-sectional, and regime features may encode weak "
            "predictability that survives walk-forward validation and realistic daily-bar costs."
        ),
        dataset_id=dataset.bundle_id,
        features=tuple(sorted(str(name) for name in feature_config)),
        label=research_config.target,
        test_plan={
            "primary_test": "one_sided_paired_student_t_on_walk_forward_rank_ic",
            "naive_reference": NAIVE_CANDIDATE,
            "feature_config": feature_config,
            "candidate_order": list(SPRINT_2_CANDIDATES),
            "final_holdout_access": "selected candidate only after development selection",
        },
        validation={
            "walk_forward": walk_forward_config,
            "research": asdict(research_config),
            "readiness": asdict(readiness_thresholds),
        },
        costs=backtest_config,
        uncertainty={
            "development": "matched-fold rank-IC distribution and Student-t interval",
            "final_holdout": "circular moving-block bootstrap over selected-candidate returns",
            "calibration": "OOF regression intercept and slope; descriptive, not probability calibration",
        },
        rejection_thresholds={
            "family_alpha": float(correction["alpha"]),
            "paper_readiness_requires_all_gates": True,
            "incomplete_point_in_time_forces_rejection": True,
        },
        trials=trials,
        correction=MultipleTestingPolicy(
            method=correction["method"],
            alpha=float(correction["alpha"]),
            family_size=int(correction["family_size"]),
            assumptions=tuple(str(value) for value in correction["assumptions"]),
            failed_trial_p_value=float(correction["failed_trial_p_value"]),
        ),
        kill_criteria=criteria,
        frozen_at=frozen_at,
    )


def _one_sided_p_value(differences: np.ndarray) -> float:
    values = np.asarray(differences, dtype=float)
    if values.ndim != 1 or len(values) < 2 or not np.isfinite(values).all():
        raise ValueError("paired fold differences require at least two finite values")
    mean = float(np.mean(values))
    if float(np.ptp(values)) == 0.0:
        return 0.0 if mean > 0.0 else 1.0
    standard_deviation = float(np.std(values, ddof=1))
    statistic = mean / (standard_deviation / math.sqrt(len(values)))
    return float(stats.t.sf(statistic, df=len(values) - 1))


def _calibration(predictions: pd.DataFrame) -> dict[str, float]:
    x = predictions["prediction"].to_numpy(dtype=float)
    y = predictions["target"].to_numpy(dtype=float)
    if len(x) < 2 or not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError("OOF calibration requires finite prediction/target pairs")
    if float(np.std(x)) == 0.0:
        return {
            "calibration_intercept": float(np.mean(y)),
            "calibration_slope": 0.0,
            "calibration_rmse": float(np.sqrt(np.mean(np.square(y - np.mean(y))))),
        }
    design = np.column_stack([np.ones(len(x)), x])
    coefficients, _, _, _ = np.linalg.lstsq(design, y, rcond=None)
    fitted = design @ coefficients
    return {
        "calibration_intercept": float(coefficients[0]),
        "calibration_slope": float(coefficients[1]),
        "calibration_rmse": float(np.sqrt(np.mean(np.square(y - fitted)))),
    }


def _trial_evidence(
    run_dir: Path,
    plan: FrozenResearchPlan,
) -> dict[str, dict[str, float | int]]:
    windows = pd.read_csv(run_dir / "development_windows.csv")
    predictions = pd.read_csv(run_dir / "development_predictions.csv")
    economics = pd.read_csv(run_dir / "development_economic_metrics.csv").set_index("model")
    baseline = (
        windows.loc[windows["model"].eq(NAIVE_CANDIDATE)]
        .set_index("window_id")["rank_ic"]
        .sort_index()
    )
    if len(baseline) < 2 or not np.isfinite(baseline.to_numpy(dtype=float)).all():
        raise ValueError("naive baseline lacks complete finite matched-fold evidence")

    evidence: dict[str, dict[str, float | int]] = {}
    for trial in plan.trials:
        model = trial.candidate
        model_windows = (
            windows.loc[windows["model"].eq(model)].set_index("window_id")["rank_ic"].sort_index()
        )
        if not model_windows.index.equals(baseline.index):
            raise ValueError(f"candidate {model!r} does not share the frozen baseline folds")
        differences = (
            model_windows.to_numpy(dtype=float)
            if model == NAIVE_CANDIDATE
            else model_windows.to_numpy(dtype=float) - baseline.to_numpy(dtype=float)
        )
        candidate_predictions = predictions.loc[predictions["model"].eq(model)]
        economic = economics.loc[model]
        metric_rows = windows.loc[windows["model"].eq(model)]
        iterations = pd.to_numeric(
            metric_rows.get("training_iterations", pd.Series(dtype=float)),
            errors="coerce",
        ).dropna()
        evidence[trial.trial_id] = {
            "mean_rank_ic": float(model_windows.mean()),
            "incremental_rank_ic": float(np.mean(differences)),
            "rank_ic_standard_error": float(
                np.std(model_windows.to_numpy(dtype=float), ddof=1) / math.sqrt(len(model_windows))
            ),
            "p_value": _one_sided_p_value(differences),
            "net_annual_return": float(economic["net_annual_return"]),
            "gross_annual_return": float(economic["gross_annual_return"]),
            "annual_cost_drag": float(economic["annual_cost_drag"]),
            "max_drawdown": float(economic["max_drawdown"]),
            "average_turnover": float(economic["average_turnover"]),
            "fold_count": int(len(model_windows)),
            "prediction_rows": int(economic["prediction_rows"]),
            "model_fit_count": int(len(metric_rows)),
            "training_iterations": int(iterations.sum()) if not iterations.empty else 0,
            "training_warning_count": int(
                pd.to_numeric(
                    metric_rows.get("training_warning_count", pd.Series(dtype=float)),
                    errors="coerce",
                )
                .fillna(0)
                .sum()
            ),
            **_calibration(candidate_predictions),
        }
    return evidence


def run_governed_baseline_study(
    *,
    dataset: SignalFoundryDataset,
    model_specs: list[dict[str, Any]],
    feature_config: dict[str, Any],
    walk_forward_config: dict[str, Any],
    backtest_config: dict[str, Any],
    research_config: GovernedResearchConfig,
    readiness_thresholds: ReadinessThresholds,
    governance_config: dict[str, Any],
    output_root: str | Path = "runs/signal-foundry-sprint-2",
    research_output_root: str | Path = "runs/signal-foundry",
    code_sha: str | None = None,
    invocation: dict[str, Any] | None = None,
    clock: Callable[[], datetime] | None = None,
) -> BaselineStudyResult:
    """Execute and audit the exact Sprint 2 candidate family.

    All state is local and immutable-by-convention under ignored ``runs/``
    directories. The function never acquires data, reads credentials, emits
    orders to a broker, or publishes row-level observations.
    """

    _validate_family(model_specs)
    now = clock or (lambda: datetime.now(UTC))
    frozen_at = _timestamp(now)
    seeded_specs = seed_model_specs(model_specs, research_config.seed)
    plan = _plan(
        dataset=dataset,
        model_specs=seeded_specs,
        feature_config=feature_config,
        walk_forward_config=walk_forward_config,
        backtest_config=backtest_config,
        research_config=research_config,
        readiness_thresholds=readiness_thresholds,
        governance_config=governance_config,
        frozen_at=frozen_at,
    )
    study_dir = Path(output_root) / plan.plan_hash
    if study_dir.exists():
        raise FileExistsError(f"governed study already exists: {study_dir}")
    ledger_settings = dict(governance_config["ledger"])
    ledger = ResearchLedger.create(
        study_dir,
        plan,
        max_bytes=int(ledger_settings["max_bytes"]),
        max_records=int(ledger_settings["max_records"]),
    )
    for trial in plan.trials:
        ledger.register_trial(trial.trial_id, occurred_at=_timestamp(now))
        ledger.transition(
            trial.trial_id,
            "STARTED",
            occurred_at=_timestamp(now),
            details={"dispatcher": "sequential_cpu", "candidate": trial.candidate},
        )

    try:
        research_result = run_governed_signal_foundry_research(
            dataset=dataset,
            model_specs=seeded_specs,
            feature_config=feature_config,
            walk_forward_config=walk_forward_config,
            backtest_config=backtest_config,
            research_config=research_config,
            readiness_thresholds=readiness_thresholds,
            output_root=research_output_root,
            code_sha=code_sha,
            invocation=invocation,
            clock=now,
        )
        trial_evidence = _trial_evidence(research_result.run_dir, plan)
        for trial in plan.trials:
            values = trial_evidence[trial.trial_id]
            p_value = float(values.pop("p_value"))
            ledger.transition(
                trial.trial_id,
                "SUCCEEDED",
                occurred_at=_timestamp(now),
                details={"metrics": values, "p_value": p_value},
            )
        family = ledger.evaluate_family(occurred_at=_timestamp(now))
    except BaseException as exc:
        terminal = {
            str(record["trial_id"])
            for record in ledger.verify()
            if record["event_type"] in {"TRIAL_SUCCEEDED", "TRIAL_FAILED"}
        }
        for trial in plan.trials:
            if trial.trial_id not in terminal:
                ledger.transition(
                    trial.trial_id,
                    "FAILED",
                    occurred_at=_timestamp(now),
                    details={
                        "reason": "batch_execution_failed",
                        "exception_type": type(exc).__name__,
                    },
                )
        ledger.evaluate_family(occurred_at=_timestamp(now))
        raise

    selected_trial = next(
        trial for trial in plan.trials if trial.candidate == research_result.candidate_model
    )
    selected_kills = list(family.killed[selected_trial.trial_id])
    decision = (
        "ADVANCE_TO_PAPER_EVALUATION"
        if research_result.dossier["decision"] == "READY_FOR_PAPER" and not selected_kills
        else "REJECT_PAPER_ADVANCEMENT"
    )
    records = ledger.verify()
    terminal_failures = [
        {
            "trial_id": record["trial_id"],
            "reason": record["payload"].get("reason"),
            "exception_type": record["payload"].get("exception_type"),
        }
        for record in records
        if record["event_type"] == "TRIAL_FAILED"
    ]
    summary = {
        "study_schema_version": STUDY_SCHEMA_VERSION,
        "study_id": plan.plan_hash,
        "plan_hash": plan.plan_hash,
        "research_run_id": research_result.run_id,
        "candidate_model": research_result.candidate_model,
        "decision": decision,
        "readiness_decision": research_result.dossier["decision"],
        "selected_candidate_kill_reasons": selected_kills,
        "family_evaluation": family.to_dict(),
        "failures": terminal_failures,
        "ledger": {
            "record_count": len(records),
            "head_hash": records[-1]["record_hash"],
        },
        "limitations": [
            "The cached Nasdaq WIKI panel ends in 2018 and is not current trading data.",
            "Universe membership, revisions, delistings, and corporate actions are incomplete.",
            "Development t-tests treat matched fold differences as the analysis unit and have low power.",
            "Regression calibration slope is descriptive and is not probability calibration.",
            "Backtested returns are not realized profits and do not establish a persistent edge.",
        ],
    }
    _atomic_json(study_dir / "study_summary.json", summary)
    return BaselineStudyResult(
        study_id=plan.plan_hash,
        study_dir=study_dir,
        research_run=research_result,
        decision=decision,
    )
