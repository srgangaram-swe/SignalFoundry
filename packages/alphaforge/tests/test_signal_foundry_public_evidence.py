"""Aggregate-only Signal Foundry publication-boundary tests."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd
import pytest

from alphaforge.research.public_evidence import publish_signal_foundry_evidence

BUNDLE_ID = "6" * 64


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _sources(tmp_path: Path) -> tuple[Path, Path]:
    bundle = tmp_path / BUNDLE_ID
    _write_json(
        bundle / "manifest.json",
        {
            "bundle_id": BUNDLE_ID,
            "schema_version": "1.1.0",
            "rows": 13_169,
            "tickers": ["AAA", "BBB"],
            "date_min": "2013-01-02",
            "date_max": "2018-03-27",
            "files": [{}, {}],
            "license": {
                "bundle_must_remain_local": True,
                "observations_redistributable": False,
                "public_evidence_must_be_aggregate_or_synthetic": True,
            },
            "point_in_time_limits": {
                "historical_revisions_complete": False,
                "universe_membership_point_in_time": False,
                "corporate_actions_complete": False,
            },
        },
    )
    run = tmp_path / "run"
    _write_json(
        run / "run_manifest.json",
        {
            "experiment": {
                "code": {"sha": "a" * 40},
                "dataset": {
                    "bundle_id": BUNDLE_ID,
                    "point_in_time_limits": {
                        "historical_revisions_complete": False,
                        "universe_membership_point_in_time": False,
                        "corporate_actions_complete": False,
                    },
                },
                "models": [{"name": "zero_baseline"}, {"name": "ridge"}],
            }
        },
    )
    _write_json(
        run / "dossier.json",
        {
            "bundle_id": BUNDLE_ID,
            "run_id": "b" * 64,
            "candidate_model": "ridge",
            "holdout_start": "2017-01-03",
            "development_end": "2016-12-02",
            "decision": "NOT_READY",
            "failed_gates": ["point_in_time_evidence"],
            "gates": {
                "accounting_reconciliation": True,
                "point_in_time_evidence": False,
            },
            "metrics": {
                "n_days": 300,
                "accounting_reconciled": True,
                "annual_return": 0.01,
                "annual_volatility": 0.10,
                "sharpe": 0.10,
                "sortino": 0.12,
                "max_drawdown": -0.08,
                "average_turnover": 0.20,
            },
            "scenarios": [
                {
                    "scenario": "doubled_costs",
                    "accounting_reconciled": True,
                    "annual_return": -0.01,
                    "annual_volatility": 0.10,
                    "sharpe": -0.10,
                    "sortino": -0.12,
                    "max_drawdown": -0.10,
                    "average_turnover": 0.20,
                }
            ],
            "paper_controls": {
                "all_controls_passed": True,
                "broker_adapter_present": False,
                "executable_orders_emitted": False,
            },
        },
    )
    pd.DataFrame(
        {
            "scenario_aum": [500_000.0, 1_000_000.0],
            "fill_ratio": [1.0, 0.98],
            "aggregate_participation_rate": [0.01, 0.02],
            "participation_p95": [0.02, 0.04],
            "participation_max": [0.03, 0.05],
            "modeled_cost_bps_per_traded_notional": [5.0, 7.0],
            "capacity_constrained_fraction": [0.0, 0.05],
        }
    ).to_csv(run / "capacity_curve.csv", index=False)
    return run, bundle


def test_public_evidence_is_deterministic_and_never_copies_licensed_rows(
    tmp_path: Path,
) -> None:
    run, bundle = _sources(tmp_path)
    profile = tmp_path / "time.txt"
    profile.write_text(
        "       13.44 real        17.38 user         2.09 sys\n"
        "           585334784  maximum resident set size\n"
        "           389236104  peak memory footprint\n",
        encoding="utf-8",
    )
    first = publish_signal_foundry_evidence(
        run_dir=run,
        bundle_dir=bundle,
        config_path="configs/signal_foundry_wiki_bootstrap.yaml",
        output_dir=tmp_path / "first",
        performance_profile=profile,
    )
    second = publish_signal_foundry_evidence(
        run_dir=run,
        bundle_dir=bundle,
        config_path="configs/signal_foundry_wiki_bootstrap.yaml",
        output_dir=tmp_path / "second",
        performance_profile=profile,
    )

    first_hashes = {
        path.relative_to(first): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in first.rglob("*")
        if path.is_file()
    }
    second_hashes = {
        path.relative_to(second): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in second.rglob("*")
        if path.is_file()
    }
    assert first_hashes == second_hashes
    summary = json.loads((first / "summary.json").read_text(encoding="utf-8"))
    assert summary["decision"] == "NOT_READY"
    assert summary["source"]["provider_requests"] == 0
    assert summary["source"]["licensed_observations_published"] is False
    assert summary["source"]["consumer_exclusions"]["rows"] == 0
    assert summary["paper_controls"]["executable_orders_emitted"] is False
    assert summary["performance"]["wall_seconds"] == 13.44
    assert summary["performance"]["maximum_resident_set_bytes"] == 585_334_784
    assert summary["performance"]["compute_path"].startswith("CPU")
    text = "\n".join(
        path.read_text(encoding="utf-8")
        for path in first.rglob("*")
        if path.suffix in {".json", ".csv", ".md"}
    )
    assert "AAA" not in text
    assert "BBB" not in text


def test_public_evidence_rejects_mismatched_or_redistributable_source(
    tmp_path: Path,
) -> None:
    run, bundle = _sources(tmp_path)
    manifest_path = bundle / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["license"]["observations_redistributable"] = True
    _write_json(manifest_path, manifest)

    with pytest.raises(ValueError, match="non-redistributable"):
        publish_signal_foundry_evidence(
            run_dir=run,
            bundle_dir=bundle,
            config_path="configs/signal_foundry_wiki_bootstrap.yaml",
            output_dir=tmp_path / "evidence",
        )
