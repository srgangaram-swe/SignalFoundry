"""Governance, publication, and visualization tests for SF-S3-MR6 evidence."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from alphaforge.data import SignalFoundryDataset
from alphaforge.research import deep_sequence_study as study


def _prediction_and_metric_frames() -> tuple[pd.DataFrame, pd.DataFrame]:
    predictions: list[pd.DataFrame] = []
    metrics: list[dict[str, object]] = []
    dates = np.repeat(pd.bdate_range("2025-01-02", periods=3), 4)
    symbols = np.tile(["A", "B", "C", "D"], 3)
    target = np.tile([-0.02, -0.01, 0.01, 0.02], 3)
    for index, name in enumerate(study.DEEP_SEQUENCE_CANDIDATES):
        predictions.append(
            pd.DataFrame(
                {
                    "date": dates,
                    "symbol": symbols,
                    "target": target,
                    "prediction": target * (0.5 + index * 0.05),
                    "model": name,
                    "window_id": 0,
                }
            )
        )
        record: dict[str, object] = {
            "model": name,
            "window_id": 0,
            "training_iterations": 3,
        }
        if name != "lightgbm":
            record.update(
                {
                    "parameter_count": 1_000 + index,
                    "parameter_bytes": 4_000 + index * 4,
                    "fit_wall_seconds": 0.5 + index,
                    "fit_cpu_seconds": 0.4 + index,
                    "peak_device_bytes": 0,
                    "best_validation_loss": 0.01 + index * 0.001,
                }
            )
        metrics.append(record)
    return pd.concat(predictions, ignore_index=True), pd.DataFrame(metrics)


def _dataset() -> SignalFoundryDataset:
    panel = pd.DataFrame(
        {
            "date": pd.bdate_range("2025-01-02", periods=4),
            "symbol": ["AAPL"] * 4,
        }
    )
    return SignalFoundryDataset(
        bundle_dir=Path("/unused"),
        manifest={"bundle_id": "synthetic-bundle", "schema_version": "1.1.0"},
        source_panel=pd.DataFrame(),
        panel=panel,
    )


def test_committed_study_config_freezes_complete_family() -> None:
    config = study.load_deep_sequence_study_config("configs/deep_sequence_benchmark.yaml")

    assert tuple(item["name"] for item in config.models) == study.DEEP_SEQUENCE_CANDIDATES
    assert config.target == "fwd_ret_5"
    assert config.walk_forward["max_windows"] == 1
    assert config.transaction_cost_bps == 5.5
    assert all(
        item["params"].get("device") == "cpu"
        for item in config.models
        if item["name"] != "lightgbm"
    )


def test_config_rejects_unknown_fields_and_candidate_reordering(tmp_path: Path) -> None:
    source = Path("configs/deep_sequence_benchmark.yaml").read_text(encoding="utf-8")
    unknown = tmp_path / "unknown.yaml"
    unknown.write_text(source.replace("study:\n", "unknown: true\nstudy:\n"), encoding="utf-8")
    with pytest.raises(ValueError, match="root fields"):
        study.load_deep_sequence_study_config(unknown)

    reordered = tmp_path / "reordered.yaml"
    reordered.write_text(
        source.replace("- name: lightgbm", "- name: wrong_model", 1),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="model family"):
        study.load_deep_sequence_study_config(reordered)


def test_summary_is_matched_and_reports_resources_only_when_measured() -> None:
    predictions, metrics = _prediction_and_metric_frames()
    config = study.load_deep_sequence_study_config("configs/deep_sequence_benchmark.yaml")

    summary = study._summarize(predictions, metrics, config)

    assert tuple(summary["model"]) == study.DEEP_SEQUENCE_CANDIDATES
    assert (summary["observations"] == 12).all()
    assert (summary["fold_count"] == 1).all()
    assert pd.isna(summary.loc[summary["model"].eq("lightgbm"), "parameter_count"]).all()
    assert summary.loc[summary["model"].ne("lightgbm"), "parameter_count"].notna().all()
    assert (summary["net_mean_daily_return"] < summary["gross_mean_daily_return"]).all()


def test_runtime_capture_does_not_import_optional_native_packages(monkeypatch) -> None:
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


def test_aggregate_publication_is_atomic_seaborn_based_and_excludes_rows(
    tmp_path: Path,
    monkeypatch,
) -> None:
    predictions, metrics = _prediction_and_metric_frames()
    config = study.load_deep_sequence_study_config("configs/deep_sequence_benchmark.yaml")
    monkeypatch.setattr(study, "build_features", lambda *args, **kwargs: pd.DataFrame())
    monkeypatch.setattr(study, "build_labels", lambda *args, **kwargs: pd.DataFrame())
    monkeypatch.setattr(
        study,
        "run_walk_forward",
        lambda *args, **kwargs: SimpleNamespace(predictions=predictions, metrics=metrics),
    )
    calls: list[str] = []
    original_barplot = study.sns.barplot

    def recording_barplot(*args, **kwargs):
        calls.append(str(kwargs.get("x")))
        return original_barplot(*args, **kwargs)

    monkeypatch.setattr(study.sns, "barplot", recording_barplot)
    output = tmp_path / "evidence"

    result = study.run_deep_sequence_study(_dataset(), config, output)

    assert result.output_dir == output
    assert calls == ["rank_ic", "net_mean_daily_return", "fit_wall_seconds"]
    assert (output / "model_comparison.png").stat().st_size > 10_000
    assert (output / "model_summary.csv").is_file()
    metadata = json.loads((output / "summary.json").read_text(encoding="utf-8"))
    assert metadata["evidence"]["prediction_rows_published"] == 0
    assert not any("prediction" in path.name for path in output.iterdir())
    assert not (tmp_path / ".publishing-evidence").exists()


def test_publication_failure_removes_staging_and_never_overwrites(
    tmp_path: Path,
    monkeypatch,
) -> None:
    predictions, metrics = _prediction_and_metric_frames()
    config = study.load_deep_sequence_study_config("configs/deep_sequence_benchmark.yaml")
    monkeypatch.setattr(study, "build_features", lambda *args, **kwargs: pd.DataFrame())
    monkeypatch.setattr(study, "build_labels", lambda *args, **kwargs: pd.DataFrame())
    monkeypatch.setattr(
        study,
        "run_walk_forward",
        lambda *args, **kwargs: SimpleNamespace(predictions=predictions, metrics=metrics),
    )
    monkeypatch.setattr(
        study,
        "_plot_summary",
        lambda *args, **kwargs: (_ for _ in ()).throw(OSError("injected plot failure")),
    )
    output = tmp_path / "evidence"

    with pytest.raises(OSError, match="injected"):
        study.run_deep_sequence_study(_dataset(), config, output)

    assert not output.exists()
    assert not (tmp_path / ".publishing-evidence").exists()
    output.mkdir()
    with pytest.raises(FileExistsError, match="already exists"):
        study.run_deep_sequence_study(_dataset(), config, output)
