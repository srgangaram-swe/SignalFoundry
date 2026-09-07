"""Publication, governance, and reproducibility tests for SF-S4-MR2 evidence."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pandas as pd
import pytest
import yaml
from pydantic import ValidationError

from alphaforge.optimization import study

CONFIG_PATH = Path("configs/mean_variance_study.yaml")
EXPECTED_FILES = {
    "attribution_summary.csv",
    "comparison.csv",
    "manifest.json",
    "mean_variance_evidence.png",
    "resolved_config.json",
    "sensitivity.csv",
    "summary.json",
}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _config_document() -> dict[str, Any]:
    document = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    assert isinstance(document, dict)
    return document


def _write_config(tmp_path: Path, document: dict[str, Any], name: str = "study.yaml") -> Path:
    path = tmp_path / name
    path.write_text(yaml.safe_dump(document, sort_keys=False), encoding="utf-8")
    return path


@pytest.fixture(scope="module")
def published(
    tmp_path_factory: pytest.TempPathFactory,
) -> tuple[study.MeanVarianceStudyResult, Path]:
    root = tmp_path_factory.mktemp("mean-variance-study")
    output = root / "evidence"
    result = study.publish_mean_variance_study(CONFIG_PATH, output)
    return result, output


def test_committed_config_is_strict_complete_and_bounded() -> None:
    config = study.load_mean_variance_study_config(CONFIG_PATH)

    assert config.schema_version == study.STUDY_SCHEMA_VERSION
    assert config.scope == "synthetic_development_only"
    assert config.evidence.formulations == study.EXPECTED_FORMULATIONS
    assert config.evidence.perturbations[0] == 0.0
    assert config.synthetic.reserved_holdout_observations > 0
    assert study._estimated_solver_calls(config) == 323
    assert study._estimated_solver_calls(config) <= config.publication.max_solver_calls
    assert config.publication.max_artifacts >= len(EXPECTED_FILES)


@pytest.mark.parametrize(
    ("mutate", "match"),
    [
        (
            lambda document: document.update({"unknown_root": True}),
            "extra_forbidden",
        ),
        (
            lambda document: document["synthetic"].update({"unknown_nested": True}),
            "extra_forbidden",
        ),
        (
            lambda document: document["evidence"].update(
                {"formulations": list(reversed(document["evidence"]["formulations"]))}
            ),
            "frozen order",
        ),
        (
            lambda document: document["publication"].update({"max_solver_calls": 1}),
            "solver calls",
        ),
        (
            lambda document: document["synthetic"].update({"factor_volatility": float("nan")}),
            "finite number",
        ),
        (
            lambda document: document["evidence"].update({"risk_aversion": "5.0"}),
            "valid number",
        ),
        (
            lambda document: document["evidence"].update({"uncertainty_strength": 1.01}),
            "less than or equal to 1",
        ),
        (
            lambda document: (
                document["constraints"].update({"cash_buffer": 0.10}),
                document["evidence"].update({"budget": 0.95}),
            ),
            "budget exceeds",
        ),
    ],
)
def test_config_refuses_unknown_unbounded_or_incomplete_input(
    tmp_path: Path,
    mutate,
    match: str,
) -> None:
    document = _config_document()
    mutate(document)
    path = _write_config(tmp_path, document)

    with pytest.raises((ValidationError, ValueError), match=match):
        study.load_mean_variance_study_config(path)


def test_config_refuses_symlink_source(tmp_path: Path) -> None:
    link = tmp_path / "linked.yaml"
    link.symlink_to(CONFIG_PATH.resolve())

    with pytest.raises(study.MeanVarianceStudyError, match="non-symlink"):
        study.load_mean_variance_study_config(link)


def test_synthetic_development_is_deterministic_and_holdout_inaccessible() -> None:
    config = study.load_mean_variance_study_config(CONFIG_PATH)
    changed_reservation = config.synthetic.model_copy(
        update={
            "reserved_holdout_observations": config.synthetic.reserved_holdout_observations + 17
        }
    )

    first = study.build_synthetic_development_data(config.synthetic)
    second = study.build_synthetic_development_data(changed_reservation)

    pd.testing.assert_frame_equal(first.returns, second.returns, check_exact=True)
    pd.testing.assert_frame_equal(first.panel.scores, second.panel.scores, check_exact=True)
    pd.testing.assert_frame_equal(
        first.panel.forward_returns,
        second.panel.forward_returns,
        check_exact=True,
    )
    pd.testing.assert_frame_equal(first.factor_exposures, second.factor_exposures, check_exact=True)
    assert first.seed_map == second.seed_map
    assert not hasattr(first, "holdout")
    assert len(first.returns) == (
        config.synthetic.history_observations + config.synthetic.development_observations
    )
    assert len(first.panel.scores) == config.synthetic.development_observations - 1


def test_publication_contains_every_arm_sensitivity_and_reconciled_attribution(
    published: tuple[study.MeanVarianceStudyResult, Path],
) -> None:
    result, output = published
    comparison = pd.read_csv(output / "comparison.csv", keep_default_na=False)
    sensitivity = pd.read_csv(output / "sensitivity.csv")
    attribution = pd.read_csv(output / "attribution_summary.csv")

    assert set(comparison["arm"]) == study.EXPECTED_COMPARISON_ARMS
    assert set(comparison["regime"]) >= {"all"}
    assert set(comparison["turnover_budget"].astype(str)) == {"0.25", "None"}
    assert tuple(sensitivity["alpha_error"]) == (0.0, 0.25, 0.5)
    assert tuple(sensitivity["covariance_error"]) == (0.0, 0.25, 0.5)
    assert {
        "asset_risk",
        "factor_risk",
        "factor_drift",
        "realized",
        "reconciliation",
        "scenario",
        "scenario_cash",
        "scenario_cost",
    } <= set(attribution["section"])
    reconciliation = attribution.loc[attribution["section"].eq("reconciliation"), "value"].abs()
    assert (reconciliation <= 1e-12).all()
    assert {"broad_drawdown", "broad_rally", "factor_rotation"} <= set(attribution["name"])
    assert {
        "observed_ending_equity",
        "ending_equity_error",
        "modeled_portfolio_return",
        "modeled_scenario_pnl",
        "scenario_ending_value_error",
    } <= set(attribution["metric"])
    assert result.comparison_rows == len(comparison)
    assert result.sensitivity_rows == len(sensitivity)
    assert result.attribution_rows == len(attribution)


def test_manifest_hashes_exact_config_and_every_payload(
    published: tuple[study.MeanVarianceStudyResult, Path],
) -> None:
    result, output = published
    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    payload_names = {item["path"] for item in manifest["artifacts"]}

    assert {path.name for path in output.iterdir()} == EXPECTED_FILES
    assert payload_names == EXPECTED_FILES - {"manifest.json"}
    assert manifest["config_source"]["sha256"] == _sha256(CONFIG_PATH)
    assert manifest["config_source"]["bytes"] == CONFIG_PATH.stat().st_size
    assert result.config_sha256 == _sha256(CONFIG_PATH)
    assert result.manifest_sha256 == _sha256(output / "manifest.json")
    for record in manifest["artifacts"]:
        artifact = output / record["path"]
        assert record["bytes"] == artifact.stat().st_size
        assert record["sha256"] == _sha256(artifact)
    total = sum(path.stat().st_size for path in output.iterdir())
    config = study.load_mean_variance_study_config(CONFIG_PATH)
    assert total <= config.publication.max_output_bytes
    assert len(EXPECTED_FILES) <= config.publication.max_artifacts


def test_summary_makes_synthetic_nonselection_and_no_profit_scope_explicit(
    published: tuple[study.MeanVarianceStudyResult, Path],
) -> None:
    _, output = published
    summary = json.loads((output / "summary.json").read_text(encoding="utf-8"))

    assert summary["scope"] == {
        "candidate_selected": False,
        "data": "deterministic_redistribution_safe_synthetic",
        "development_folds_only": True,
        "holdout_accessible": False,
        "holdout_observations_generated": 0,
        "holdout_observations_reserved": 63,
        "paper_or_live_readiness_claim": False,
        "profit_claim": False,
    }
    assert summary["data"]["raw_rows_published"] == 0
    assert summary["evidence"]["row_level_returns_published"] == 0
    assert summary["evidence"]["row_level_predictions_published"] == 0
    limitations = " ".join(summary["limitations"]).lower()
    for phrase in ("synthetic", "holdout", "no arm", "profit", "not order"):
        assert phrase in limitations
    assert (output / "mean_variance_evidence.png").stat().st_size > 50_000


def test_publication_is_byte_reproducible(
    published: tuple[study.MeanVarianceStudyResult, Path],
    tmp_path: Path,
) -> None:
    first, first_output = published
    second_output = tmp_path / "same-study"
    second = study.publish_mean_variance_study(CONFIG_PATH, second_output)

    assert second.config_sha256 == first.config_sha256
    assert second.manifest_sha256 == first.manifest_sha256
    assert second.artifacts == first.artifacts
    for name in EXPECTED_FILES:
        assert (second_output / name).read_bytes() == (first_output / name).read_bytes()


def test_publisher_never_overwrites_or_leaves_staging(
    published: tuple[study.MeanVarianceStudyResult, Path],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, existing = published
    with pytest.raises(FileExistsError, match="already exists"):
        study.publish_mean_variance_study(CONFIG_PATH, existing)

    output = tmp_path / "failed"

    def injected_failure(*_args, **_kwargs):
        raise OSError("injected comparison failure")

    monkeypatch.setattr(study, "_run_comparison", injected_failure)
    with pytest.raises(OSError, match="injected"):
        study.publish_mean_variance_study(CONFIG_PATH, output)
    assert not output.exists()
    assert not (tmp_path / ".publishing-failed").exists()


def test_output_byte_ceiling_fails_closed(tmp_path: Path) -> None:
    document = _config_document()
    document["publication"]["max_output_bytes"] = study.MIN_OUTPUT_BYTES
    config_path = _write_config(tmp_path, document, "small-output.yaml")
    output = tmp_path / "too-large"

    with pytest.raises(study.MeanVarianceStudyError, match="max_output_bytes"):
        study.publish_mean_variance_study(config_path, output)

    assert not output.exists()
    assert not (tmp_path / ".publishing-too-large").exists()


def test_runtime_ceiling_is_checked_and_failure_is_atomic(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output = tmp_path / "expired"
    readings = iter((0.0, 1_000.0))
    monkeypatch.setattr(study.time, "monotonic", lambda: next(readings))

    with pytest.raises(study.MeanVarianceStudyError, match="runtime ceiling"):
        study.publish_mean_variance_study(CONFIG_PATH, output)

    assert not output.exists()
    assert not (tmp_path / ".publishing-expired").exists()


def test_plot_uses_seaborn_apis_with_declared_axes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[str] = []
    original_barplot = study.sns.barplot
    original_lineplot = study.sns.lineplot

    def record_barplot(*args, **kwargs):
        calls.append("barplot")
        return original_barplot(*args, **kwargs)

    def record_lineplot(*args, **kwargs):
        calls.append("lineplot")
        return original_lineplot(*args, **kwargs)

    monkeypatch.setattr(study.sns, "barplot", record_barplot)
    monkeypatch.setattr(study.sns, "lineplot", record_lineplot)
    comparison = pd.DataFrame(
        {
            "regime": ["all", "all"],
            "arm": ["minimum_variance", "no_trade"],
            "turnover_budget": ["None", "None"],
            "net_return": [0.01, 0.0],
            "mean_turnover": [0.1, 0.0],
            "cost_drag": [0.001, 0.0],
            "feasible_fraction": [0.5, 1.0],
            "n_dates": [2, 2],
            "fold": ["fold_1", "fold_1"],
        }
    )
    sensitivity = pd.DataFrame(
        {
            "alpha_error": [0.0, 0.25],
            "net_return": [0.01, 0.005],
            "covariance_error": [0.0, 0.25],
            "covariance_net_return": [0.01, 0.009],
        }
    )
    destination = tmp_path / "plot.png"

    study._plot_study(comparison, sensitivity, destination, seed=17, dpi=100)

    assert calls == ["barplot", "barplot", "lineplot", "barplot"]
    assert destination.stat().st_size > 10_000


def test_makefile_exposes_bounded_publication_target() -> None:
    makefile = Path("Makefile").read_text(encoding="utf-8")
    assert "mean-variance-evidence:" in makefile
    assert "OUTPUT must name a new evidence directory" in makefile
    assert "scripts/publish_mean_variance_study.py" in makefile
