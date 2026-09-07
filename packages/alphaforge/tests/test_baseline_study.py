"""Tests for the governed seven-candidate Sprint 2 study and publisher."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import alphaforge.research.baseline_study as study_module
from alphaforge.config import load_signal_foundry_research_config
from alphaforge.data import SignalFoundryDataset
from alphaforge.evaluation import ReadinessThresholds
from alphaforge.research import GovernedResearchConfig, ResearchLedger
from alphaforge.research.baseline_study import (
    SPRINT_2_CANDIDATES,
    BaselineStudyResult,
    _one_sided_p_value,
    run_governed_baseline_study,
)
from alphaforge.research.baseline_study_evidence import publish_baseline_study_evidence
from alphaforge.research.signal_foundry import GovernedResearchResult


def _dataset(tmp_path: Path) -> SignalFoundryDataset:
    bundle = tmp_path / "bundle"
    bundle.mkdir(parents=True)
    manifest = {
        "bundle_id": "a" * 64,
        "schema_version": "1.1.0",
        "rows": 100,
        "tickers": ["REDACTED"],
        "date_min": "2013-01-02",
        "date_max": "2018-03-27",
        "license": {
            "observations_redistributable": False,
            "bundle_must_remain_local": True,
            "public_evidence_must_be_aggregate_or_synthetic": True,
        },
        "point_in_time_limits": {
            "historical_revisions_complete": False,
            "universe_membership_point_in_time": False,
            "corporate_actions_complete": False,
        },
    }
    (bundle / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return SignalFoundryDataset(
        bundle_dir=bundle,
        manifest=manifest,
        source_panel=pd.DataFrame(),
        panel=pd.DataFrame(),
    )


def _specs() -> list[dict[str, object]]:
    return [{"name": name, "params": {}} for name in SPRINT_2_CANDIDATES]


def _governance() -> dict[str, object]:
    return {
        "version": "1.0.0",
        "correction": {
            "method": "holm_bonferroni",
            "alpha": 0.05,
            "family_size": 7,
            "assumptions": ["complete family"],
            "failed_trial_p_value": 1.0,
        },
        "kill_criteria": [
            {
                "name": "non_positive_incremental_ic",
                "metric": "incremental_rank_ic",
                "operator": "le",
                "threshold": 0.0,
            },
            {
                "name": "non_positive_net_return",
                "metric": "net_annual_return",
                "operator": "le",
                "threshold": 0.0,
            },
            {
                "name": "excessive_drawdown",
                "metric": "max_drawdown",
                "operator": "le",
                "threshold": -0.25,
            },
        ],
        "ledger": {"max_records": 1_000, "max_bytes": 1_000_000},
    }


def _research_config() -> GovernedResearchConfig:
    return GovernedResearchConfig(
        holdout_start="2017-01-03",
        benchmark_symbol="AAPL",
        target="fwd_ret_5",
        horizons=(1, 5, 20),
        seed=42,
    )


def _fake_research_run(tmp_path: Path, dataset: SignalFoundryDataset) -> GovernedResearchResult:
    run = tmp_path / "research" / ("b" * 64)
    run.mkdir(parents=True)
    fold_rows: list[dict[str, object]] = []
    prediction_rows: list[dict[str, object]] = []
    economic_rows: list[dict[str, object]] = []
    for model_index, model in enumerate(SPRINT_2_CANDIDATES):
        for window_id, baseline_ic in enumerate((0.01, 0.02, 0.03)):
            rank_ic = baseline_ic + 0.002 * model_index
            fold_rows.append(
                {
                    "model": model,
                    "window_id": window_id,
                    "rank_ic": rank_ic,
                    "ic": rank_ic,
                    "mae": 0.02,
                    "training_iterations": 10 * model_index,
                    "training_warning_count": 0,
                }
            )
            for row in range(4):
                prediction = (row - 1.5) * 0.001 + model_index * 0.0001
                prediction_rows.append(
                    {
                        "date": f"2016-0{window_id + 1}-{row + 1:02d}",
                        "symbol": f"S{row}",
                        "target": 0.5 * prediction + 0.0002,
                        "prediction": prediction,
                        "model": model,
                        "window_id": window_id,
                    }
                )
        economic_rows.append(
            {
                "model": model,
                "prediction_rows": 12,
                "trading_sessions": 60,
                "order_count": 20,
                "fill_count": 20,
                "gross_annual_return": 0.04 + model_index * 0.01,
                "net_annual_return": 0.02 + model_index * 0.01,
                "annual_cost_drag": 0.02,
                "sharpe": 0.5,
                "max_drawdown": -0.10,
                "average_turnover": 0.2,
                "average_gross_exposure": 0.8,
                "average_net_exposure": 0.05,
            }
        )
    pd.DataFrame(fold_rows).to_csv(run / "development_windows.csv", index=False)
    pd.DataFrame(prediction_rows).to_csv(run / "development_predictions.csv", index=False)
    pd.DataFrame(economic_rows).to_csv(run / "development_economic_metrics.csv", index=False)
    dossier = {
        "bundle_id": dataset.bundle_id,
        "decision": "NOT_READY",
        "candidate_model": "small_mlp",
    }
    (run / "dossier.json").write_text(json.dumps(dossier), encoding="utf-8")
    run_manifest = {
        "experiment": {
            "experiment_id": run.name,
            "dataset": {
                "bundle_id": dataset.bundle_id,
                "point_in_time_limits": dataset.manifest["point_in_time_limits"],
            },
            "environment": {"python": "test", "hardware": {"processor": "test"}},
        }
    }
    (run / "run_manifest.json").write_text(json.dumps(run_manifest), encoding="utf-8")
    return GovernedResearchResult(
        run_id=run.name,
        run_dir=run,
        candidate_model="small_mlp",
        dossier=dossier,
    )


def _run(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[SignalFoundryDataset, BaselineStudyResult]:
    dataset = _dataset(tmp_path)
    fake = _fake_research_run(tmp_path, dataset)
    monkeypatch.setattr(
        study_module,
        "run_governed_signal_foundry_research",
        lambda **_: fake,
    )

    def fixed() -> datetime:
        return datetime(2026, 7, 26, 12, 0, tzinfo=UTC)

    result = run_governed_baseline_study(
        dataset=dataset,
        model_specs=_specs(),
        feature_config={"registry_version": "1.0.0"},
        walk_forward_config={
            "scheme": "expanding",
            "min_train_days": 20,
            "test_days": 5,
            "step_days": 5,
            "embargo_days": 5,
        },
        backtest_config={"costs": {"commission_bps": 1.0}},
        research_config=_research_config(),
        readiness_thresholds=ReadinessThresholds(),
        governance_config=_governance(),
        output_root=tmp_path / "studies",
        research_output_root=tmp_path / "unused",
        code_sha="c" * 40,
        clock=fixed,
    )
    return dataset, result


def test_one_sided_test_handles_positive_and_nonpositive_constants() -> None:
    assert _one_sided_p_value(np.array([0.1, 0.1, 0.1])) == 0.0
    assert _one_sided_p_value(np.array([0.0, 0.0, 0.0])) == 1.0
    with pytest.raises(ValueError, match="at least two finite"):
        _one_sided_p_value(np.array([np.nan, 0.1]))


def test_study_requires_exact_complete_candidate_family(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="candidates must be frozen"):
        run_governed_baseline_study(
            dataset=_dataset(tmp_path),
            model_specs=_specs()[:-1],
            feature_config={"registry_version": "1.0.0"},
            walk_forward_config={"embargo_days": 20},
            backtest_config={"costs": {}},
            research_config=_research_config(),
            readiness_thresholds=ReadinessThresholds(),
            governance_config=_governance(),
        )


def test_study_ledger_corrects_complete_family_and_rejects_paper(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, result = _run(tmp_path, monkeypatch)

    assert result.decision == "REJECT_PAPER_ADVANCEMENT"
    ledger = ResearchLedger.open(result.study_dir, expected_plan_hash=result.study_id)
    records = ledger.verify()
    assert len([record for record in records if record["event_type"] == "TRIAL_SUCCEEDED"]) == 7
    assert records[-1]["event_type"] == "FAMILY_EVALUATED"
    summary = json.loads((result.study_dir / "study_summary.json").read_text())
    assert summary["readiness_decision"] == "NOT_READY"
    assert summary["failures"] == []
    assert len(summary["family_evaluation"]["adjusted_p_values"]) == 7


def test_aggregate_publisher_excludes_rows_and_generates_all_seaborn_plots(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dataset, result = _run(tmp_path, monkeypatch)
    profile = tmp_path / "time.txt"
    profile.write_text(
        "        12.34 real         11.00 user         1.00 sys\n"
        "            123456 maximum resident set size\n"
        "            234567 peak memory footprint\n",
        encoding="utf-8",
    )
    config = tmp_path / "study.yaml"
    config.write_text("frozen: true\n", encoding="utf-8")
    output = publish_baseline_study_evidence(
        study_dir=result.study_dir,
        run_dir=result.research_run.run_dir,
        bundle_dir=dataset.bundle_dir,
        config_path=config,
        output_dir=tmp_path / "public",
        performance_profile=profile,
    )

    assert {path.name for path in (output / "plots").glob("*.png")} == {
        "compute_accounting.png",
        "costed_returns.png",
        "fold_rank_ic.png",
        "multiplicity_correction.png",
    }
    summary = json.loads((output / "summary.json").read_text())
    assert summary["source"]["licensed_observations_published"] is False
    assert summary["source"]["row_level_predictions_published"] is False
    assert summary["source"]["final_holdout_artifacts_published"] is False
    assert summary["compute"]["wall_seconds"] == 12.34
    public_columns = set(pd.read_csv(output / "model_summary.csv"))
    assert not public_columns.intersection({"date", "symbol", "target", "prediction"})
    assert not list(output.rglob("*prediction*"))


def test_committed_study_profile_freezes_exact_family_and_single_worker() -> None:
    profile = load_signal_foundry_research_config("configs/signal_foundry_sprint_2_study.yaml")

    assert tuple(model["name"] for model in profile["models"]) == SPRINT_2_CANDIDATES
    assert profile["walk_forward"] == {
        "scheme": "expanding",
        "min_train_days": 504,
        "test_days": 63,
        "step_days": 63,
        "embargo_days": 21,
        "max_windows": None,
    }
    for model in profile["models"]:
        if model["name"] in {"random_forest", "lightgbm", "xgboost", "catboost"}:
            assert model["params"]["n_jobs"] == 1
