"""End-to-end governed Signal Foundry research tests."""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd
import pytest

from alphaforge.config import load_signal_foundry_research_config
from alphaforge.data import SignalFoundryDataset, SyntheticMarketConfig, generate_synthetic_market
from alphaforge.evaluation import NOT_READY, ReadinessThresholds
from alphaforge.execution import standard_stress_profiles
from alphaforge.research import GovernedResearchConfig, run_governed_signal_foundry_research
from alphaforge.research.signal_foundry import _capacity_config


def _dataset(tmp_path: Path) -> SignalFoundryDataset:
    panel = generate_synthetic_market(
        SyntheticMarketConfig(
            n_symbols=6,
            n_days=420,
            seed=7,
            benchmark_symbol="SPY",
        )
    )
    return SignalFoundryDataset(
        bundle_dir=tmp_path / "synthetic-bundle",
        manifest={
            "bundle_id": "a" * 64,
            "license": {
                "observations_redistributable": True,
                "bundle_must_remain_local": False,
                "public_evidence_must_be_aggregate_or_synthetic": False,
            },
            "point_in_time_limits": {
                "historical_revisions_complete": False,
                "universe_membership_point_in_time": False,
                "corporate_actions_complete": False,
            },
        },
        source_panel=pd.DataFrame(),
        panel=panel,
    )


def _run(tmp_path: Path, *, fitted_transform: bool = False):
    dataset = _dataset(tmp_path)
    dates = sorted(dataset.panel["date"].unique())
    return run_governed_signal_foundry_research(
        dataset=dataset,
        model_specs=(
            [{"name": "ridge", "params": {"alpha": 10.0}}]
            if fitted_transform
            else [
                {
                    "name": "momentum_baseline",
                    "params": {"feature": "momentum_20", "scale": 0.05},
                },
                {"name": "ridge", "params": {"alpha": 10.0}},
            ]
        ),
        feature_config={
            "return_lags": [1, 5],
            "vol_windows": [5, 20],
            "ma_windows": [5, 20],
            "momentum_windows": [5, 20],
            "rsi_window": 5,
            "macd": {"fast": 3, "slow": 8, "signal": 3},
            "bollinger_window": 5,
            "mean_reversion_window": 3,
            "volume_window": 5,
            "beta_window": 20,
            "rolling_sharpe_window": 20,
            "drawdown_window": 20,
            "regime_vol_window": 5,
            "regime_trend_fast": 5,
            "regime_trend_slow": 20,
            "hmm_regime": False,
            "cross_sectional": True,
            "fitted_transform": {
                "version": "1.0.0",
                "enabled": fitted_transform,
                "imputation": "median",
                "standardize": True,
                "variance_threshold": None,
                "pca_components": 0.95 if fitted_transform else None,
                "pca_whiten": False,
                "min_fit_rows": 64,
            },
        },
        walk_forward_config={
            "scheme": "expanding",
            "min_train_days": 160,
            "test_days": 40,
            "step_days": 40,
            "embargo_days": 5,
        },
        backtest_config={
            "strategy": "long_short",
            "strategy_params": {"quantile": 0.25},
            "rebalance_frequency": 5,
            "execution_lag": 1,
            "liquidate_at_end": True,
            "initial_capital": 1_000_000,
            "execution": {
                "adv_lookback": 5,
                "volatility_lookback": 5,
                "max_participation_rate": 0.05,
                "impact_coefficient": 0.10,
                "missing_price_policy": "raise",
            },
            "costs": {
                "commission_bps": 1.0,
                "half_spread_bps": 2.5,
                "slippage_bps": 2.0,
            },
            "portfolio": {
                "max_weight": 0.25,
                "max_gross_exposure": 1.0,
                "inverse_vol_scaling": True,
                "turnover_cap": 0.50,
            },
            "risk": {},
            "capacity": {
                "aum_multiples": [0.5, 1.0, 2.0],
                "max_participation_rate": 0.05,
                "minimum_fill_ratio": 0.95,
            },
            "borrow_financing": {
                "short_borrow_bps_annual": 1000.0,
                "cash_financing_bps_annual": 500.0,
            },
        },
        research_config=GovernedResearchConfig(
            holdout_start=str(pd.Timestamp(dates[300]).date()),
            benchmark_symbol="SPY",
            target="fwd_ret_5",
            horizons=(1, 5),
            seed=17,
        ),
        readiness_thresholds=ReadinessThresholds(
            minimum_holdout_days=40,
            minimum_deflated_sharpe_probability=0.50,
            maximum_average_turnover=1.0,
        ),
        output_root=tmp_path / "runs",
        code_sha="b" * 40,
        invocation={
            "entrypoint": "test_signal_foundry_research",
            "arguments": ["--config", "synthetic"],
        },
        clock=lambda: datetime(2026, 7, 24, tzinfo=UTC),
    )


def test_governed_run_is_transactional_auditable_and_not_ready_on_missing_pit(
    tmp_path: Path,
) -> None:
    result = _run(tmp_path)

    assert result.run_dir.name == result.run_id
    assert result.dossier["decision"] == NOT_READY
    assert "point_in_time_evidence" in result.dossier["failed_gates"]
    assert (result.run_dir / "dossier.md").is_file()
    assert (result.run_dir / "final_holdout_predictions.csv").is_file()
    assert (result.run_dir / "capacity_curve.csv").is_file()
    new_execution_artifacts = (
        "execution_events.csv",
        "accounting.csv",
        "friction_model_manifest.csv",
        "friction_attribution.csv",
        "latency_schedule.csv",
    )
    assert all((result.run_dir / name).is_file() for name in new_execution_artifacts)
    manifest = json.loads((result.run_dir / "run_manifest.json").read_text())
    assert manifest["run_manifest_version"] == "2.0.0"
    assert manifest["result"]["trial_ledger_head"]
    assert manifest["experiment"]["experiment_id"] == result.run_id
    assert manifest["experiment"]["code"]["sha"] == "b" * 40
    assert manifest["experiment"]["dataset"]["bundle_id"] == "a" * 64
    assert manifest["experiment"]["seeds"]
    assert manifest["experiment"]["environment"]["dependencies"]
    assert manifest["experiment"]["artifacts"]
    ledger = [
        json.loads(line)
        for line in (result.run_dir / "trial_ledger.jsonl").read_text().splitlines()
    ]
    assert ledger[0]["previous_hash"] == "0" * 64
    assert ledger[1]["previous_hash"] == ledger[0]["record_hash"]
    assert result.dossier["gates"]["missing_price_halt"]
    assert result.dossier["gates"]["paper_controls"]
    assert result.dossier["paper_controls"]["all_controls_passed"]
    assert result.dossier["paper_controls"]["broker_adapter_present"] is False
    assert result.dossier["paper_controls"]["executable_orders_emitted"] is False
    assert (
        result.dossier["metrics"]["gross_annual_return"]
        >= result.dossier["metrics"]["annual_return"]
    )
    assert result.dossier["concentration"]["gross_exposure"] >= 0.0
    assert all(scenario["accounting_reconciled"] for scenario in result.dossier["scenarios"])
    execution_profiles = standard_stress_profiles()[1:]
    execution_scenarios = result.dossier["scenarios"][: len(execution_profiles)]
    assert [scenario["scenario"] for scenario in execution_scenarios] == [
        profile.name for profile in execution_profiles
    ]
    assert [scenario["stress_profile_digest"] for scenario in execution_scenarios] == [
        profile.digest for profile in execution_profiles
    ]
    manifest_frame = pd.read_csv(result.run_dir / "friction_model_manifest.csv")
    assert set(manifest_frame["model_type"]) == {
        "fill_cost",
        "execution_policy",
        "carry_cost",
        "latency",
        "stress_profile",
    }
    friction_frame = pd.read_csv(result.run_dir / "friction_attribution.csv")
    assert friction_frame["model_digest"].str.fullmatch(r"[0-9a-f]{64}").all()
    assert friction_frame["record_digest"].str.fullmatch(r"[0-9a-f]{64}").all()
    assert not pd.read_csv(result.run_dir / "latency_schedule.csv").empty
    assert result.dossier["uncertainty"]["available"]
    assert result.dossier["year_stability"]
    assert result.dossier["regime_stability"]


def test_final_holdout_identity_cannot_be_overwritten_or_repeated(tmp_path: Path) -> None:
    first = _run(tmp_path)

    with pytest.raises(FileExistsError, match="cannot be repeated"):
        _run(tmp_path)

    assert first.run_dir.is_dir()
    assert not list((tmp_path / "runs").glob(".publishing-*"))


def test_clean_output_roots_produce_byte_identical_evidence(tmp_path: Path) -> None:
    first = _run(tmp_path / "first")
    second = _run(tmp_path / "second")

    assert first.run_id == second.run_id
    first_hashes = {
        path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in first.run_dir.iterdir()
    }
    second_hashes = {
        path.name: hashlib.sha256(path.read_bytes()).hexdigest()
        for path in second.run_dir.iterdir()
    }
    assert first_hashes == second_hashes


def test_governed_holdout_records_development_only_fitted_state(tmp_path: Path) -> None:
    result = _run(tmp_path, fitted_transform=True)

    development_path = result.run_dir / "development_fitted_transformations.csv"
    holdout_path = result.run_dir / "final_holdout_fitted_transformation.json"
    assert development_path.is_file()
    assert holdout_path.is_file()
    holdout_state = json.loads(holdout_path.read_text(encoding="utf-8"))
    assert pd.Timestamp(holdout_state["fit_end"]) <= pd.Timestamp(result.dossier["development_end"])
    assert pd.Timestamp(holdout_state["fit_end"]) < pd.Timestamp(result.dossier["holdout_start"])
    assert len(holdout_state["state_id"]) == 64


def test_committed_wiki_profile_maps_the_complete_capacity_contract() -> None:
    profile = load_signal_foundry_research_config("configs/signal_foundry_wiki_bootstrap.yaml")

    capacity, minimum_fill_ratio = _capacity_config(dict(profile["backtest"]))

    assert capacity.reference_aum == 1_000_000.0
    assert capacity.impact_exponent == 0.5
    assert capacity.variable_cost_fraction == 0.5
    assert minimum_fill_ratio == 0.95
