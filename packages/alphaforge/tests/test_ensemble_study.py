"""Integration and artifact tests for the governed ensemble study."""

from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml
from matplotlib import image as mpimg

import alphaforge.research.ensemble_study as ensemble_study
from alphaforge.research.ensemble_study import (
    ENSEMBLE_METHODS,
    REFERENCE_EXPERTS,
    load_ensemble_study_config,
    run_synthetic_ensemble_study,
)


def test_committed_config_is_strict_and_reference_runner_is_aggregate_only(
    tmp_path: Path,
) -> None:
    config = load_ensemble_study_config("configs/ensemble_benchmark.yaml")
    result = run_synthetic_ensemble_study(config, tmp_path / "evidence")

    assert set(result.summary["model"]) == set(REFERENCE_EXPERTS) | set(ENSEMBLE_METHODS)
    assert {
        "mse_delta_vs_best_single",
        "mse_ci_lower",
        "mse_ci_upper",
        "mse_standard_error",
        "mse_delta_ci_lower",
        "mse_delta_ci_upper",
        "mse_delta_standard_error",
        "rank_ic_valid_dates",
        "rank_ic_total_dates",
        "rank_ic_ci_lower",
        "rank_ic_ci_upper",
        "mean_heuristic_dispersion",
        "heuristic_interval_coverage",
        "mean_turnover",
        "mean_cost_drag",
        "fallback_rate",
    } <= set(result.summary)
    metadata = json.loads((result.output_dir / "summary.json").read_text(encoding="utf-8"))
    assert metadata["scope"] == "synthetic_engineering_only"
    assert metadata["holdout"]["row_predictions_published"] == 0
    assert metadata["holdout"]["row_targets_published"] == 0
    assert metadata["training_boundary"]["training_oof_rows"] > 0
    assert set(metadata["states"]) == set(ENSEMBLE_METHODS)
    assert metadata["resolved_config"]["seed"] == config.seed
    assert metadata["resolved_config"]["bootstrap_seed"] == config.bootstrap_seed
    assert metadata["seed_map"]["generator_root_seed"] == config.seed
    assert metadata["seed_map"]["bootstrap_seed"] == config.bootstrap_seed
    assert set(metadata["seed_map"]["children"]) == {
        "target",
        *(f"expert:{expert}" for expert in REFERENCE_EXPERTS),
    }
    assert metadata["environment"]["reference_device"] == "cpu"
    assert "within each date" in metadata["metric_policy"]["rank_ic"]
    published = {path.name for path in result.output_dir.iterdir()}
    assert not any("prediction_rows" in name or "targets" in name for name in published)
    assert {
        "README.md",
        "model_summary.csv",
        "prediction_error_correlations.csv",
        "regime_overlap.csv",
        "marginal_contribution.csv",
        "turnover_costs.csv",
        "uncertainty_diagnostics.csv",
        "ensemble_evidence.png",
        "summary.json",
    } == published


def test_reference_evidence_is_deterministic_and_hashes_are_integrity_checked(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = load_ensemble_study_config("configs/ensemble_benchmark.yaml")
    first = run_synthetic_ensemble_study(config, tmp_path / "first")
    second = run_synthetic_ensemble_study(config, tmp_path / "second")

    for first_path in sorted(first.output_dir.iterdir()):
        second_path = second.output_dir / first_path.name
        assert first_path.read_bytes() == second_path.read_bytes()
    metadata = json.loads((first.output_dir / "summary.json").read_text(encoding="utf-8"))
    for name, record in metadata["artifacts"].items():
        payload = (first.output_dir / name).read_bytes()
        assert hashlib.sha256(payload).hexdigest() == record["sha256"]
        assert len(payload) == record["bytes"]
    monkeypatch.setattr(
        ensemble_study,
        "REFERENCE_EXPERTS",
        tuple(reversed(REFERENCE_EXPERTS)),
    )
    monkeypatch.setattr(
        ensemble_study,
        "ENSEMBLE_METHODS",
        tuple(reversed(ENSEMBLE_METHODS)),
    )
    reordered = run_synthetic_ensemble_study(config, tmp_path / "reordered")
    for first_path in sorted(first.output_dir.iterdir()):
        assert first_path.read_bytes() == (reordered.output_dir / first_path.name).read_bytes()


def test_reference_tables_cover_required_diagnostics_and_plot_is_legible(
    tmp_path: Path,
) -> None:
    result = run_synthetic_ensemble_study(
        load_ensemble_study_config("configs/ensemble_benchmark.yaml"),
        tmp_path / "evidence",
    )
    correlations = pd.read_csv(result.output_dir / "prediction_error_correlations.csv")
    assert set(correlations["kind"]) == {"prediction", "error"}
    assert correlations["correlation"].between(-1.0, 1.0).all()
    assert set(correlations["correlation_status"]) == {"defined"}
    overlap = pd.read_csv(result.output_dir / "regime_overlap.csv")
    assert set(overlap["regime"]) == {"calm", "uncertain", "stress"}
    assert (
        overlap["signed_prediction_overlap_status"].str.startswith(("defined", "undefined")).all()
    )
    assert overlap["error_correlation_status"].str.startswith(("defined", "undefined")).all()
    marginal = pd.read_csv(result.output_dir / "marginal_contribution.csv")
    assert len(marginal) == len(ENSEMBLE_METHODS) * len(REFERENCE_EXPERTS)
    assert set(marginal["omitted_expert"]) == set(REFERENCE_EXPERTS)
    uncertainty = pd.read_csv(result.output_dir / "uncertainty_diagnostics.csv")
    assert uncertainty["heuristic_interval_coverage"].dropna().between(0.0, 1.0).all()
    stable = uncertainty.loc[uncertainty["model"] == "stable"].iloc[0]
    assert pd.isna(stable["dispersion_absolute_error_correlation"])
    assert stable["dispersion_correlation_status"].startswith("undefined")
    summary = result.summary
    assert (summary["mse_ci_lower"] <= summary["mse_ci_upper"]).all()
    assert (summary["mse_standard_error"] > 0.0).all()
    assert (summary["mse_delta_ci_lower"] <= summary["mse_delta_ci_upper"]).all()
    best_single = summary.loc[summary["model"] == summary["best_single_model"].iloc[0]].iloc[0]
    assert best_single["mse_delta_vs_best_single"] == pytest.approx(0.0)
    assert best_single["mse_delta_ci_lower"] == pytest.approx(0.0)
    assert best_single["mse_delta_ci_upper"] == pytest.approx(0.0)
    assert (summary["rank_ic_valid_dates"] <= summary["rank_ic_total_dates"]).all()
    gate = summary.loc[summary["model"] == "regime_gate"].iloc[0]
    assert gate["rank_ic_valid_dates"] < gate["rank_ic_total_dates"]
    image = mpimg.imread(result.output_dir / "ensemble_evidence.png")
    assert image.shape[0] >= 900
    assert image.shape[1] >= 1500
    assert float(image.std()) > 0.05


def test_published_csv_schemas_are_exact_and_exclude_row_level_fields(
    tmp_path: Path,
) -> None:
    result = run_synthetic_ensemble_study(
        load_ensemble_study_config("configs/ensemble_benchmark.yaml"),
        tmp_path / "evidence",
    )
    expected_schemas = {
        "model_summary.csv": {
            "model",
            "kind",
            "rows",
            "mse",
            "mse_ci_lower",
            "mse_ci_upper",
            "mse_standard_error",
            "mse_variance",
            "mae",
            "rank_ic",
            "rank_ic_status",
            "rank_ic_valid_dates",
            "rank_ic_total_dates",
            "rank_ic_ci_lower",
            "rank_ic_ci_upper",
            "rank_ic_standard_error",
            "rank_ic_variance",
            "bootstrap_resamples",
            "bootstrap_block_dates",
            "bootstrap_confidence_level",
            "bootstrap_seed",
            "bootstrap_circular",
            "fallback_rate",
            "mean_heuristic_dispersion",
            "heuristic_interval_coverage",
            "dispersion_absolute_error_correlation",
            "dispersion_correlation_status",
            "dispersion_rows",
            "mean_turnover",
            "mean_cost_drag",
            "gross_directional_return",
            "net_directional_return",
            "net_directional_return_ci_lower",
            "net_directional_return_ci_upper",
            "net_directional_return_standard_error",
            "net_directional_return_variance",
            "best_single_model",
            "best_single_mse",
            "mse_delta_vs_best_single",
            "mse_delta_ci_lower",
            "mse_delta_ci_upper",
            "mse_delta_standard_error",
            "mse_delta_variance",
        },
        "prediction_error_correlations.csv": {
            "kind",
            "model_a",
            "model_b",
            "correlation",
            "correlation_status",
            "rows",
        },
        "regime_overlap.csv": {
            "regime",
            "model_a",
            "model_b",
            "rows",
            "signed_prediction_overlap",
            "signed_prediction_overlap_status",
            "error_correlation",
            "error_correlation_status",
        },
        "marginal_contribution.csv": {
            "model",
            "omitted_expert",
            "full_mse",
            "mse_without_expert",
            "mse_increase_without_expert",
            "full_state_id",
            "reduced_state_id",
            "reduced_fit_status",
        },
        "turnover_costs.csv": {
            "model",
            "kind",
            "mean_turnover",
            "mean_cost_drag",
            "gross_directional_return",
            "net_directional_return",
            "net_directional_return_ci_lower",
            "net_directional_return_ci_upper",
            "net_directional_return_standard_error",
            "net_directional_return_variance",
            "bootstrap_resamples",
            "bootstrap_block_dates",
            "bootstrap_confidence_level",
            "bootstrap_seed",
            "bootstrap_circular",
        },
        "uncertainty_diagnostics.csv": {
            "model",
            "kind",
            "rows",
            "mean_heuristic_dispersion",
            "heuristic_interval_coverage",
            "dispersion_absolute_error_correlation",
            "dispersion_correlation_status",
            "dispersion_rows",
            "fallback_rate",
        },
    }
    forbidden = {"date", "symbol", "target", "prediction"}
    for name, expected in expected_schemas.items():
        columns = set(pd.read_csv(result.output_dir / name, nrows=0).columns)
        assert columns == expected
        assert columns.isdisjoint(forbidden)
    metadata = json.loads((result.output_dir / "summary.json").read_text(encoding="utf-8"))
    published = {path.name for path in result.output_dir.iterdir()}
    assert set(metadata["artifacts"]) == published - {"summary.json"}


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda payload: payload["study"].update({"future_target": True}), "unknown"),
        (lambda payload: payload["study"].update({"training_dates": 95}), "divisible"),
        (
            lambda payload: payload["resources"].update({"max_prediction_records": 1}),
            "cannot hold",
        ),
        (
            lambda payload: payload["study"].update({"interpretation": "market_evidence"}),
            "synthetic_engineering_only",
        ),
        (lambda payload: payload["study"].update({"holdout_dates": 19}), "holdout_dates"),
        (lambda payload: payload["bootstrap"].update({"block_length": 21}), "block_length"),
    ],
)
def test_config_rejects_unknown_unbounded_or_misleading_settings(
    tmp_path: Path,
    mutation: object,
    message: str,
) -> None:
    payload = yaml.safe_load(Path("configs/ensemble_benchmark.yaml").read_text(encoding="utf-8"))
    mutation(payload)  # type: ignore[operator]
    path = tmp_path / "invalid.yaml"
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")

    with pytest.raises(ValueError, match=message):
        load_ensemble_study_config(path)


def test_runner_refuses_to_overwrite_any_existing_destination(tmp_path: Path) -> None:
    config = load_ensemble_study_config("configs/ensemble_benchmark.yaml")
    destination = tmp_path / "evidence"
    destination.mkdir()

    with pytest.raises(FileExistsError, match="overwrite"):
        run_synthetic_ensemble_study(config, destination)


def test_direct_config_construction_cannot_bypass_strict_invariants() -> None:
    config = load_ensemble_study_config("configs/ensemble_benchmark.yaml")
    with pytest.raises(ValueError, match="synthetic_engineering_only"):
        replace(config, interpretation="market_evidence")
    with pytest.raises(ValueError, match="finite"):
        replace(config, transaction_cost_bps=np.nan)
    with pytest.raises(ValueError, match="cannot hold"):
        replace(config, max_prediction_records=1)
    with pytest.raises(ValueError, match="holdout_dates"):
        replace(config, holdout_dates=19)
    with pytest.raises(ValueError, match="bootstrap_block_length"):
        replace(config, bootstrap_block_length=21)
