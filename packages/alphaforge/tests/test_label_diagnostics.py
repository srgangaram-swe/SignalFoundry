"""Deterministic label diagnostics and Seaborn evidence tests."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

from alphaforge.config import load_labels_config
from alphaforge.labels import LabelContract, build_label_set, diagnose_labels
from alphaforge.visualization import save_label_diagnostic_plots


@pytest.fixture(scope="module")
def diagnostic_evidence(small_panel):
    config = load_labels_config("configs/labels.yaml")
    config["benchmark_symbol"] = "BENCH"
    contract = LabelContract.from_mapping(config)
    dataset = build_label_set(small_panel, contract)
    variants = {
        scale: dataset if scale == 1.0 else build_label_set(small_panel, contract.scaled(scale))
        for scale in config["diagnostics"]["sensitivity_scales"]
    }
    diagnostics = diagnose_labels(
        dataset,
        autocorrelation_lag=config["diagnostics"]["autocorrelation_lag"],
        periods=config["diagnostics"]["temporal_periods"],
        sensitivity_variants=variants,
    )
    return dataset, diagnostics


def test_diagnostics_cover_dependence_balance_stability_and_sensitivity(
    diagnostic_evidence,
) -> None:
    dataset, diagnostics = diagnostic_evidence

    assert set(diagnostics.summary["label"]) == {
        definition.name for definition in dataset.contract.definitions
    }
    assert diagnostics.summary["overlap_rate"].between(0, 1).all()
    assert diagnostics.summary["autocorrelation_lag_1"].between(-1, 1).all()
    assert (
        diagnostics.summary["effective_sample_size"] <= diagnostics.summary["observations"]
    ).all()
    balance_totals = diagnostics.class_balance.groupby("label")["fraction"].sum()
    assert balance_totals.eq(1.0).all()
    assert set(diagnostics.temporal_stability["period"]) == {1, 2, 3, 4}
    assert set(diagnostics.parameter_sensitivity["parameter_scale"]) == {0.5, 1.0, 2.0}
    assert diagnostics.parameter_sensitivity["observations"].gt(0).all()


def test_diagnostics_replay_is_deterministic(diagnostic_evidence) -> None:
    dataset, expected = diagnostic_evidence
    replay = diagnose_labels(
        dataset,
        autocorrelation_lag=1,
        periods=4,
        sensitivity_variants={1.0: dataset},
    )
    replay_again = diagnose_labels(
        dataset,
        autocorrelation_lag=1,
        periods=4,
        sensitivity_variants={1.0: dataset},
    )

    pd.testing.assert_frame_equal(replay.summary, expected.summary)
    pd.testing.assert_frame_equal(replay.summary, replay_again.summary)
    pd.testing.assert_frame_equal(replay.class_balance, replay_again.class_balance)
    pd.testing.assert_frame_equal(replay.temporal_stability, replay_again.temporal_stability)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"autocorrelation_lag": 0}, "positive"),
        ({"periods": 1}, "at least two"),
        ({"periods": 1_000_000}, "cannot exceed"),
    ],
)
def test_invalid_diagnostic_parameters_fail(
    diagnostic_evidence, kwargs: dict[str, Any], message: str
) -> None:
    dataset, _ = diagnostic_evidence
    with pytest.raises(ValueError, match=message):
        diagnose_labels(dataset, **kwargs)


def test_sensitivity_variants_require_exact_label_schema(
    diagnostic_evidence,
) -> None:
    dataset, _ = diagnostic_evidence
    subset_contract = LabelContract(
        benchmark_symbol=dataset.contract.benchmark_symbol,
        definitions=(dataset.contract.definitions[0],),
    )
    subset_panel = pd.concat(
        [
            dataset.values[["date", "symbol"]].assign(close=100.0),
            pd.DataFrame(
                {
                    "date": sorted(dataset.values["date"].unique()),
                    "symbol": dataset.contract.benchmark_symbol,
                    "close": 100.0,
                }
            ),
        ],
        ignore_index=True,
    )
    subset = build_label_set(subset_panel, subset_contract)
    with pytest.raises(ValueError, match="exact baseline label names"):
        diagnose_labels(dataset, sensitivity_variants={1.0: subset})


def test_seaborn_label_plots_write_complete_nonempty_evidence(
    tmp_path: Path, diagnostic_evidence
) -> None:
    _, diagnostics = diagnostic_evidence
    paths = save_label_diagnostic_plots(diagnostics, tmp_path / "plots")

    assert [path.name for path in paths] == [
        "dependence.png",
        "class_balance.png",
        "temporal_stability.png",
        "parameter_sensitivity.png",
    ]
    assert all(path.stat().st_size > 10_000 for path in paths)


def test_evidence_cli_publishes_once_and_records_only_synthetic_aggregates(
    tmp_path: Path,
) -> None:
    output = tmp_path / "label-evidence"
    command = [
        sys.executable,
        "scripts/generate_label_evidence.py",
        "--output-dir",
        str(output),
    ]
    completed = subprocess.run(command, check=True, capture_output=True, text=True)
    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))

    assert "published label evidence" in completed.stdout
    assert manifest["data"]["source"] == "alphaforge synthetic market"
    assert not manifest["data"]["licensed_observations"]
    assert not manifest["data"]["market_evidence"]
    assert len(manifest["artifacts"]) == 8
    assert not any(
        sensitive in json.dumps(manifest).lower()
        for sensitive in ("api_key", "api-token", "password")
    )

    repeated = subprocess.run(command, check=False, capture_output=True, text=True)
    assert repeated.returncode != 0
    assert "will not be overwritten" in repeated.stderr
