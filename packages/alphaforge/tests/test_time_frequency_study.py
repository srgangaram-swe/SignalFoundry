"""Governance, publication, and evidence tests for SF-S3-MR7."""

from __future__ import annotations

import json
from importlib.util import find_spec
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from alphaforge.models.time_frequency_vision import (
    PredictionMetrics,
    ProgressionGatePolicy,
    evaluate_progression_gate,
)
from alphaforge.research import time_frequency_study as study


def _config() -> study.TimeFrequencyStudyConfig:
    return study.load_time_frequency_study_config("configs/time_frequency_vision_benchmark.yaml")


def test_committed_profile_freezes_progression_split_and_compute() -> None:
    config = _config()

    assert study.TIME_FREQUENCY_CANDIDATES == (
        "lightgbm_time",
        "lightgbm_spectral",
        "small_cnn",
        "resnet",
        "vit",
    )
    assert config.train_fraction == 0.60
    assert config.validation_fraction == 0.20
    assert config.vision.device == "cpu"
    assert config.vision.max_parameters == 500_000
    assert config.synthetic_reference["dates"] == 80
    assert config.small_cnn_gate.minimum_incremental_rank_ic == -0.10
    assert config.resnet_gate.minimum_incremental_rank_ic == 0.01


def test_config_rejects_unknown_missing_or_incoherent_fields(tmp_path: Path) -> None:
    source = Path("configs/time_frequency_vision_benchmark.yaml").read_text(encoding="utf-8")
    unknown = tmp_path / "unknown.yaml"
    unknown.write_text(source.replace("study:\n", "unknown: true\nstudy:\n"), encoding="utf-8")
    with pytest.raises(ValueError, match="root fields mismatch"):
        study.load_time_frequency_study_config(unknown)

    missing = tmp_path / "missing.yaml"
    missing.write_text(source.replace("  seed: 20260726\n", "", 1), encoding="utf-8")
    with pytest.raises(ValueError, match="study fields mismatch"):
        study.load_time_frequency_study_config(missing)

    no_test = tmp_path / "no-test.yaml"
    no_test.write_text(
        source.replace("train_fraction: 0.60", "train_fraction: 0.75").replace(
            "validation_fraction: 0.20",
            "validation_fraction: 0.20",
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="10%"):
        study.load_time_frequency_study_config(no_test)


def test_synthetic_reference_is_deterministic_and_aligned() -> None:
    config = _config()
    first = study.build_synthetic_time_frequency_reference(config)
    second = study.build_synthetic_time_frequency_reference(config)
    np.testing.assert_array_equal(first.values, second.values)
    np.testing.assert_array_equal(first.target, second.target)
    assert first.values.shape == (960, 3, 8, 4)
    assert first.spectral_features.shape == (960, 15)
    assert first.complete_samples.all()


def test_chronological_split_is_disjoint_ordered_and_untouched() -> None:
    config = _config()
    batch = study.build_synthetic_time_frequency_reference(config)
    split = study._chronological_split(batch, config)

    assert not (split.train & split.validation).any()
    assert not (split.train & split.test).any()
    assert not (split.validation & split.test).any()
    assert (split.train | split.validation | split.test).all()
    assert pd.Timestamp(batch.dates[split.train].max()) < pd.Timestamp(
        batch.dates[split.validation].min()
    )
    assert pd.Timestamp(batch.dates[split.validation].max()) < pd.Timestamp(
        batch.dates[split.test].min()
    )
    assert (split.train.sum(), split.validation.sum(), split.test.sum()) == (576, 192, 192)


def test_progression_never_starts_with_vit_and_blocks_after_failed_resnet(
    monkeypatch,
) -> None:
    config = _config()
    batch = study.build_synthetic_time_frequency_reference(config)
    split = study._chronological_split(batch, config)
    order: list[str] = []
    validation_target = batch.target[split.validation]

    def tabular(name, features, batch_arg, split_arg, params):
        del features, batch_arg, split_arg, params
        order.append(name)
        return study._FittedCandidate(name, validation_target.copy(), {}, object())

    def vision(name, batch_arg, split_arg, vision_config, policy, prior_gate):
        del batch_arg, split_arg, vision_config, policy, prior_gate
        order.append(name)
        # Small CNN passes the relaxed first gate. ResNet is deliberately tied
        # with it, so the positive incremental second gate blocks ViT.
        return study._FittedCandidate(name, validation_target.copy(), {}, object())

    monkeypatch.setattr(study, "_fit_lightgbm_candidate", tabular)
    monkeypatch.setattr(study, "_fit_vision_candidate", vision)

    candidates, gates, statuses = study._progression(batch, split, config)

    assert order == ["lightgbm_time", "lightgbm_spectral", "small_cnn", "resnet"]
    assert [candidate.name for candidate in candidates] == order
    assert len(gates) == 2
    assert gates[0].passed
    assert not gates[1].passed
    assert statuses["vit"] == "blocked_by_resnet_gate"


def test_summary_lists_blocked_candidates_without_fabricated_metrics(monkeypatch) -> None:
    config = _config()
    batch = study.build_synthetic_time_frequency_reference(config)
    split = study._chronological_split(batch, config)
    validation = batch.target[split.validation].copy()
    candidate = study._FittedCandidate(
        "lightgbm_time",
        validation,
        {"fit_wall_seconds": 0.1},
        (
            SimpleNamespace(predict=lambda frame: batch.target[split.test].copy()),
            pd.DataFrame({"x": np.arange(len(batch.target))}),
        ),
    )
    monkeypatch.setattr(
        study,
        "evaluate_oos_predictions",
        lambda *args, **kwargs: SimpleNamespace(
            net_mean_daily_return=0.0,
            mean_daily_turnover=0.0,
        ),
    )

    summary = study._summary(
        [candidate],
        {
            "lightgbm_time": "evaluated",
            "lightgbm_spectral": "blocked",
            "small_cnn": "blocked",
            "resnet": "blocked",
            "vit": "blocked",
        },
        batch,
        split,
        config,
    )

    assert tuple(summary["model"]) == study.TIME_FREQUENCY_CANDIDATES
    assert summary.loc[summary["model"].eq("lightgbm_time"), "test_rank_ic"].notna().all()
    assert summary.loc[summary["model"].eq("resnet"), "test_rank_ic"].isna().all()


def test_plot_uses_seaborn_and_discloses_blocked_architectures(
    tmp_path: Path,
    monkeypatch,
) -> None:
    summary = pd.DataFrame(
        [
            {
                "model": "lightgbm_time",
                "status": "evaluated",
                "test_rank_ic": 0.1,
                "test_rank_ic_standard_error": 0.01,
                "test_rmse": 0.2,
                "test_observations": 20,
                "test_dates": 5,
                "fit_wall_seconds": 0.01,
            },
            {"model": "resnet", "status": "blocked_by_small_cnn_gate"},
        ]
    )
    calls: list[str] = []
    original = study.sns.barplot

    def recording_barplot(*args, **kwargs):
        calls.append(str(kwargs.get("x")))
        return original(*args, **kwargs)

    monkeypatch.setattr(study.sns, "barplot", recording_barplot)
    output = tmp_path / "comparison.png"

    study._plot_summary(summary, output, seed=42)

    assert calls == ["test_rank_ic", "test_rmse", "fit_wall_seconds"]
    assert output.stat().st_size > 10_000


def test_gate_records_are_json_safe_and_self_verifying() -> None:
    policy = ProgressionGatePolicy()
    metrics = PredictionMetrics(
        partition="validation",
        observations=128,
        dates=16,
        rmse=0.1,
        mae=0.08,
        rank_ic=0.2,
    )
    gate = evaluate_progression_gate(
        candidate="small_cnn",
        baseline="spectral",
        candidate_metrics=metrics,
        baseline_metrics=metrics,
        policy=policy,
    )

    encoded = json.dumps(gate.payload() | {"evidence_sha256": gate.evidence_sha256})

    assert gate.evidence_sha256 in encoded
    gate.verify()


def test_runtime_capture_uses_metadata_without_importing_optional_runtimes(monkeypatch) -> None:
    observed: list[str] = []

    def fake_version(name: str) -> str:
        observed.append(name)
        if name == "lightgbm":
            raise study.PackageNotFoundError
        return f"{name}-version"

    monkeypatch.setattr(study, "version", fake_version)

    environment = study._runtime_environment()

    assert observed == ["torch", "lightgbm"]
    assert environment["torch"] == "torch-version"
    assert environment["lightgbm"] == "unavailable"


@pytest.mark.skipif(
    find_spec("torch") is None or find_spec("lightgbm") is None,
    reason="optional torch and LightGBM dependencies are unavailable",
)
def test_reference_publication_is_atomic_aggregate_and_reproducible(tmp_path: Path) -> None:
    config = study.TimeFrequencyStudyConfig(
        **{
            **_config().__dict__,
            "synthetic_reference": {
                **_config().synthetic_reference,
                "dates": 24,
                "symbols": 6,
            },
            "vision": study.VisionTrainingConfig(
                **{
                    **_config().vision.__dict__,
                    "max_epochs": 2,
                    "patience": 2,
                    "width": 8,
                    "vit_heads": 2,
                }
            ),
            "small_cnn_gate": ProgressionGatePolicy(
                minimum_observations=8,
                minimum_dates=2,
                minimum_rank_ic=1.0,
                minimum_incremental_rank_ic=1.0,
                maximum_rmse_ratio=0.1,
            ),
        }
    )
    output = tmp_path / "evidence"

    result = study.run_synthetic_time_frequency_study(config, output)

    assert result.output_dir == output
    assert (output / "model_comparison.png").stat().st_size > 10_000
    assert (output / "model_summary.csv").is_file()
    metadata = json.loads((output / "summary.json").read_text(encoding="utf-8"))
    assert metadata["data"]["raw_observations_published"] == 0
    assert metadata["evidence"]["prediction_rows_published"] == 0
    assert not any("prediction" in path.name for path in output.iterdir())
    assert not (tmp_path / ".publishing-evidence").exists()
    with pytest.raises(FileExistsError, match="already exists"):
        study.run_synthetic_time_frequency_study(config, output)


def test_publication_failure_removes_staging(tmp_path: Path, monkeypatch) -> None:
    config = _config()
    batch = study.build_synthetic_time_frequency_reference(config)
    monkeypatch.setattr(
        study,
        "_progression",
        lambda *args: ([], (), {name: "blocked" for name in study.TIME_FREQUENCY_CANDIDATES}),
    )
    monkeypatch.setattr(
        study,
        "_summary",
        lambda *args: pd.DataFrame(
            [{"model": name, "status": "blocked"} for name in study.TIME_FREQUENCY_CANDIDATES]
        ),
    )
    monkeypatch.setattr(
        study,
        "_plot_summary",
        lambda *args, **kwargs: (_ for _ in ()).throw(OSError("injected plot failure")),
    )
    output = tmp_path / "evidence"

    with pytest.raises(OSError, match="injected"):
        study.run_time_frequency_study(batch, config, output)

    assert not output.exists()
    assert not (tmp_path / ".publishing-evidence").exists()
