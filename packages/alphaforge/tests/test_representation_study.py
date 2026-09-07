"""Governance, evaluation, publication, and evidence tests for SF-S3-MR8."""

from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from alphaforge.research import representation_study as study


def _config() -> study.RepresentationStudyConfig:
    return study.load_representation_study_config("configs/latent_representation_benchmark.yaml")


def _fast_config() -> study.RepresentationStudyConfig:
    config = _config()
    candidates = tuple(
        replace(candidate, max_epochs=2, patience=1) for candidate in config.candidate_configs
    )
    return replace(config, candidate_configs=candidates)


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_committed_profile_freezes_candidates_splits_budgets_and_cpu() -> None:
    config = _config()

    assert study.REPRESENTATION_CANDIDATES == (
        "raw",
        "pca",
        "incremental_pca",
        "robust_pca",
        "dense_autoencoder",
        "sequence_autoencoder",
        "denoising_autoencoder",
        "variational_autoencoder",
        "contrastive_timeseries",
    )
    assert config.train_fraction == 0.60
    assert config.validation_fraction == 0.20
    assert config.selection_metric == "validation_rank_ic"
    assert config.synthetic_reference["dates"] == 72
    assert all(candidate.max_parameters == 200_000 for candidate in config.candidate_configs)


def test_config_rejects_unknown_missing_reordered_and_unsafe_fields(tmp_path: Path) -> None:
    source = Path("configs/latent_representation_benchmark.yaml").read_text(encoding="utf-8")
    unknown = tmp_path / "unknown.yaml"
    unknown.write_text(source.replace("study:\n", "unknown: true\nstudy:\n"), encoding="utf-8")
    with pytest.raises(ValueError, match="root fields mismatch"):
        study.load_representation_study_config(unknown)

    missing = tmp_path / "missing.yaml"
    missing.write_text(source.replace("  seed: 20260726\n", "", 1), encoding="utf-8")
    with pytest.raises(ValueError, match="study fields mismatch"):
        study.load_representation_study_config(missing)

    reordered = tmp_path / "reordered.yaml"
    reordered.write_text(
        source.replace("    - raw\n    - pca\n", "    - pca\n    - raw\n"),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="candidates must be exactly"):
        study.load_representation_study_config(reordered)

    unsafe = tmp_path / "unsafe.yaml"
    unsafe.write_text(
        source.replace("  max_parameters: 200000", "  max_parameters: 0"),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="max_parameters"):
        study.load_representation_study_config(unsafe)


def test_synthetic_reference_is_deterministic_aligned_and_nontrivial() -> None:
    config = _config()
    first = study.build_synthetic_representation_reference(config)
    second = study.build_synthetic_representation_reference(config)

    np.testing.assert_array_equal(first.batch.values, second.batch.values)
    np.testing.assert_array_equal(first.target, second.target)
    np.testing.assert_array_equal(first.regime, second.regime)
    np.testing.assert_array_equal(first.anomaly, second.anomaly)
    assert first.batch.values.shape == (432, 10)
    assert len(set(first.batch.symbols)) == 6
    assert set(first.regime) == {0, 1}
    assert 0 < first.anomaly.sum() < len(first.anomaly)


def test_chronological_split_uses_complete_disjoint_dates() -> None:
    config = _config()
    reference = study.build_synthetic_representation_reference(config)
    split = study.chronological_representation_split(reference, config)
    dates = pd.DatetimeIndex(reference.batch.dates)

    assert (split.train.sum(), split.validation.sum(), split.test.sum()) == (258, 84, 90)
    assert set(dates[split.train]).isdisjoint(set(dates[split.validation]))
    assert set(dates[split.train]).isdisjoint(set(dates[split.test]))
    assert dates[split.train].max() < dates[split.validation].min()
    assert dates[split.validation].max() < dates[split.test].min()


@pytest.fixture(scope="module")
def evaluated_summary() -> pd.DataFrame:
    pytest.importorskip("torch")
    config = _fast_config()
    reference = study.build_synthetic_representation_reference(config)
    split = study.chronological_representation_split(reference, config)
    return study.evaluate_representation_candidates(reference, split, config)


def test_all_axes_are_evaluated_against_raw_and_selection_is_unique(
    evaluated_summary: pd.DataFrame,
) -> None:
    assert tuple(evaluated_summary["candidate"]) == study.REPRESENTATION_CANDIDATES
    required = {
        "validation_rank_ic",
        "validation_rank_ic_daily_se",
        "validation_rank_ic_dates",
        "test_rank_ic",
        "test_rank_ic_daily_se",
        "test_rank_ic_dates",
        "test_regime_accuracy",
        "test_similarity_regime_recall",
        "test_anomaly_auc",
        "test_effective_rank",
        "test_mean_abs_embedding_correlation",
        "test_reconstruction_mse",
        "test_rank_ic_delta_vs_raw",
        "test_prediction_mse_delta_vs_raw",
        "fit_wall_seconds",
        "parameter_count",
        "state_id",
    }
    assert required.issubset(evaluated_summary.columns)
    assert evaluated_summary["selected_on_validation"].sum() == 1
    raw = evaluated_summary.loc[evaluated_summary["candidate"].eq("raw")].iloc[0]
    assert raw["test_rank_ic_delta_vs_raw"] == pytest.approx(0.0)
    assert raw["test_prediction_mse_delta_vs_raw"] == pytest.approx(0.0)
    assert (evaluated_summary["test_rank_ic_daily_se"] >= 0.0).all()
    assert set(evaluated_summary["test_rank_ic_dates"]) == {15}
    assert evaluated_summary["test_anomaly_auc"].between(0.0, 1.0).all()
    contrastive = evaluated_summary.loc[
        evaluated_summary["candidate"].eq("contrastive_timeseries")
    ].iloc[0]
    assert pd.isna(contrastive["test_reconstruction_mse"])


def test_candidate_order_does_not_change_state_or_numeric_evidence() -> None:
    pytest.importorskip("torch")
    config = _fast_config()
    reference = study.build_synthetic_representation_reference(config)
    split = study.chronological_representation_split(reference, config)
    first = study.evaluate_representation_candidates(reference, split, config)
    second = study.evaluate_representation_candidates(
        reference,
        split,
        config,
        candidate_order=tuple(reversed(study.REPRESENTATION_CANDIDATES)),
    )

    assert tuple(first["state_id"]) == tuple(second["state_id"])
    deterministic_columns = [
        column
        for column in first.select_dtypes(include=[np.number, bool]).columns
        if column not in {"fit_wall_seconds", "fit_cpu_seconds"}
    ]
    pd.testing.assert_frame_equal(
        first[deterministic_columns],
        second[deterministic_columns],
        check_exact=False,
        atol=1e-12,
        rtol=1e-12,
    )


def test_test_target_mutation_cannot_change_fitted_states_or_validation_selection() -> None:
    pytest.importorskip("torch")
    config = _fast_config()
    reference = study.build_synthetic_representation_reference(config)
    split = study.chronological_representation_split(reference, config)
    first = study.evaluate_representation_candidates(reference, split, config)
    mutated_target = reference.target.copy()
    mutated_target[split.test] *= -1_000.0
    mutated = study.SyntheticRepresentationReference(
        batch=reference.batch,
        target=mutated_target,
        regime=reference.regime,
        anomaly=reference.anomaly,
    )
    second = study.evaluate_representation_candidates(mutated, split, config)

    assert tuple(first["state_id"]) == tuple(second["state_id"])
    assert tuple(first["validation_rank_ic"]) == tuple(second["validation_rank_ic"])
    assert tuple(first["selected_on_validation"]) == tuple(second["selected_on_validation"])
    assert not np.allclose(first["test_prediction_mse"], second["test_prediction_mse"])


def test_all_validation_scores_precede_any_test_access(monkeypatch) -> None:
    pytest.importorskip("torch")
    config = _fast_config()
    reference = study.build_synthetic_representation_reference(config)
    split = study.chronological_representation_split(reference, config)
    roles: list[str] = []
    original = study._prediction_metrics

    def record_partition(*args, **kwargs):
        evaluation_dates = args[4]
        maximum = pd.DatetimeIndex(evaluation_dates).max().isoformat()
        roles.append("validation" if maximum == split.validation_end else "test")
        return original(*args, **kwargs)

    monkeypatch.setattr(study, "_prediction_metrics", record_partition)
    study.evaluate_representation_candidates(reference, split, config)

    candidate_count = len(study.REPRESENTATION_CANDIDATES)
    assert roles == ["validation"] * candidate_count + ["test"] * candidate_count


def test_plot_calls_seaborn_and_discloses_synthetic_scope(
    tmp_path: Path,
    evaluated_summary: pd.DataFrame,
    monkeypatch,
) -> None:
    calls = 0
    original = study.sns.barplot

    def spy(*args, **kwargs):
        nonlocal calls
        calls += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(study.sns, "barplot", spy)
    output = tmp_path / "plot.png"
    study._plot_summary(
        evaluated_summary,
        output,
        config=_fast_config(),
        test_rows=90,
        test_dates=15,
    )

    assert calls == 4
    assert output.stat().st_size > 10_000


def test_publisher_is_atomic_aggregate_only_and_hashes_every_evidence_file(
    tmp_path: Path,
) -> None:
    pytest.importorskip("torch")
    output = tmp_path / "evidence"
    result = study.run_synthetic_representation_study(_fast_config(), output)

    assert result.output_dir == output
    assert {
        "candidate_summary.csv",
        "summary.json",
        "manifest.json",
        "plots/representation_comparison.png",
    } == {str(path.relative_to(output)) for path in output.rglob("*") if path.is_file()}
    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    for record in manifest["files"]:
        path = output / record["path"]
        assert record["sha256"] == _digest(path)
        assert record["bytes"] == path.stat().st_size
    assert not any(manifest["publication"].values())
    assert result.metadata["data"]["raw_rows_published"] == 0
    assert result.metadata["data"]["model_weights_published"] == 0
    assert len(result.summary) == len(study.REPRESENTATION_CANDIDATES)
    with pytest.raises(FileExistsError):
        study.run_synthetic_representation_study(_fast_config(), output)


def test_failed_publication_removes_staging_without_partial_destination(
    tmp_path: Path,
    monkeypatch,
) -> None:
    output = tmp_path / "failed"

    def fail(*args, **kwargs):
        del args, kwargs
        raise RuntimeError("injected study failure")

    monkeypatch.setattr(study, "evaluate_representation_candidates", fail)
    with pytest.raises(RuntimeError, match="injected"):
        study.run_synthetic_representation_study(_fast_config(), output)
    assert not output.exists()
    assert not (tmp_path / ".publishing-failed").exists()


def test_candidate_order_contract_rejects_missing_or_duplicate_candidates() -> None:
    config = _fast_config()
    reference = study.build_synthetic_representation_reference(config)
    split = study.chronological_representation_split(reference, config)
    with pytest.raises(ValueError, match="every frozen candidate"):
        study.evaluate_representation_candidates(
            reference,
            split,
            config,
            candidate_order=(study.REPRESENTATION_CANDIDATES[0],)
            * len(study.REPRESENTATION_CANDIDATES),
        )
