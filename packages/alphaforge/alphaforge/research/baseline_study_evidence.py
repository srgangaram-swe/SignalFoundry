"""Publish aggregate-only evidence from a governed Sprint 2 baseline study."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from alphaforge.research.baseline_study import SPRINT_2_CANDIDATES, STUDY_SCHEMA_VERSION
from alphaforge.research.governance import ResearchLedger
from alphaforge.research.public_evidence import _load_time_profile
from alphaforge.visualization.baseline_study_plots import (
    plot_compute_accounting,
    plot_costed_returns,
    plot_fold_rank_ic,
    plot_multiplicity,
)

PUBLIC_STUDY_EVIDENCE_VERSION = "1.0.0"
_MAX_JSON_BYTES = 8 * 1024 * 1024
_MAX_CSV_BYTES = 16 * 1024 * 1024
_FOLD_COLUMNS = ("model", "window_id", "rank_ic", "mae")
_MODEL_METRICS = (
    "mean_rank_ic",
    "incremental_rank_ic",
    "rank_ic_standard_error",
    "net_annual_return",
    "gross_annual_return",
    "annual_cost_drag",
    "max_drawdown",
    "average_turnover",
    "fold_count",
    "prediction_rows",
    "model_fit_count",
    "training_iterations",
    "training_warning_count",
    "calibration_intercept",
    "calibration_slope",
    "calibration_rmse",
)


def _json(path: Path) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file() or not 0 < path.stat().st_size <= _MAX_JSON_BYTES:
        raise ValueError(f"JSON evidence input violates file policy: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read JSON evidence input: {path}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"JSON evidence input must be an object: {path}")
    return value


def _csv(path: Path) -> pd.DataFrame:
    if path.is_symlink() or not path.is_file() or not 0 < path.stat().st_size <= _MAX_CSV_BYTES:
        raise ValueError(f"CSV evidence input violates file policy: {path}")
    return pd.read_csv(path)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(value, sort_keys=True, indent=2, ensure_ascii=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _model_table(
    records: tuple[Any, ...],
    family: dict[str, Any],
) -> pd.DataFrame:
    plan = records[0]["payload"]["plan"]
    trial_to_model = {str(trial["trial_id"]): str(trial["candidate"]) for trial in plan["trials"]}
    terminal = {
        str(record["trial_id"]): record
        for record in records
        if record["event_type"] in {"TRIAL_SUCCEEDED", "TRIAL_FAILED"}
    }
    rows: list[dict[str, Any]] = []
    for trial_id, model in trial_to_model.items():
        record = terminal.get(trial_id)
        if record is None:
            raise ValueError(f"eligible trial {trial_id!r} is not terminal")
        failed = record["event_type"] == "TRIAL_FAILED"
        metrics = {} if failed else record["payload"]["metrics"]
        if not failed and set(_MODEL_METRICS) - set(metrics):
            raise ValueError(f"trial {trial_id!r} lacks required aggregate metrics")
        row = {
            "trial_id": trial_id,
            "model": model,
            "status": "failed" if failed else "succeeded",
            **{name: (np.nan if failed else metrics[name]) for name in _MODEL_METRICS},
            "raw_p_value": float(family["raw_p_values"][trial_id]),
            "adjusted_p_value": float(family["adjusted_p_values"][trial_id]),
            "survives_correction": bool(family["rejected"][trial_id]),
            "kill_reasons": ";".join(str(value) for value in family["killed"][trial_id]),
        }
        rows.append(row)
    table = pd.DataFrame(rows)
    if tuple(table["model"]) != SPRINT_2_CANDIDATES:
        raise ValueError("ledger candidate family is not the frozen Sprint 2 family")
    return table


def publish_baseline_study_evidence(
    *,
    study_dir: str | Path,
    run_dir: str | Path,
    bundle_dir: str | Path,
    config_path: str | Path,
    output_dir: str | Path,
    performance_profile: str | Path,
) -> Path:
    """Publish redistribution-safe aggregates and Seaborn plots atomically.

    No row-level prediction, target, date, symbol, order, fill, position, or
    return series crosses this boundary.
    """

    study = Path(study_dir).resolve()
    run = Path(run_dir).resolve()
    bundle = Path(bundle_dir).resolve()
    config = Path(config_path).resolve()
    destination = Path(output_dir).resolve()
    for source in (study, run, bundle):
        if source.is_symlink() or not source.is_dir():
            raise ValueError(f"evidence source must be a real directory: {source}")
    if config.is_symlink() or not config.is_file():
        raise ValueError("study configuration must be a real file")
    if destination.exists():
        raise FileExistsError(f"public evidence destination already exists: {destination}")

    study_summary = _json(study / "study_summary.json")
    run_manifest = _json(run / "run_manifest.json")
    dossier = _json(run / "dossier.json")
    bundle_manifest = _json(bundle / "manifest.json")
    if study_summary.get("study_schema_version") != STUDY_SCHEMA_VERSION:
        raise ValueError("unsupported governed-study schema")
    if study_summary.get("research_run_id") != run.name:
        raise ValueError("study and research-run identities disagree")
    experiment = run_manifest.get("experiment")
    if not isinstance(experiment, dict) or experiment.get("experiment_id") != run.name:
        raise ValueError("research-run manifest identity is invalid")
    dataset = experiment.get("dataset")
    if not isinstance(dataset, dict) or dataset.get("bundle_id") != bundle_manifest.get(
        "bundle_id"
    ):
        raise ValueError("research run and source bundle identities disagree")
    license_policy = bundle_manifest.get("license")
    if (
        not isinstance(license_policy, dict)
        or license_policy.get("observations_redistributable") is not False
        or license_policy.get("public_evidence_must_be_aggregate_or_synthetic") is not True
    ):
        raise ValueError("publisher requires an aggregate-only non-redistributable source policy")
    if dossier.get("bundle_id") != bundle_manifest.get("bundle_id"):
        raise ValueError("dossier and source bundle identities disagree")

    ledger = ResearchLedger.open(
        study,
        expected_plan_hash=str(study_summary["plan_hash"]),
    )
    records = ledger.verify()
    family_records = [record for record in records if record["event_type"] == "FAMILY_EVALUATED"]
    if len(family_records) != 1:
        raise ValueError("study ledger must contain exactly one complete-family evaluation")
    family = dict(family_records[0]["payload"])
    models = _model_table(records, family)

    folds_raw = _csv(run / "development_windows.csv")
    missing_fold_columns = set(_FOLD_COLUMNS) - set(folds_raw)
    if missing_fold_columns:
        raise ValueError(f"development evidence lacks columns: {sorted(missing_fold_columns)}")
    folds = folds_raw.loc[:, _FOLD_COLUMNS].copy()
    if set(folds["model"]) != set(SPRINT_2_CANDIDATES):
        raise ValueError("development evidence omits a pre-registered candidate")
    expected_windows: tuple[int, ...] | None = None
    for model, block in folds.groupby("model", sort=False):
        windows = tuple(sorted(int(value) for value in block["window_id"]))
        if len(windows) != len(set(windows)):
            raise ValueError(f"candidate {model!r} repeats a walk-forward fold")
        if expected_windows is None:
            expected_windows = windows
        elif windows != expected_windows:
            raise ValueError("candidate family was not evaluated on identical folds")
    numeric = folds[["rank_ic", "mae"]].apply(pd.to_numeric, errors="coerce")
    if not np.isfinite(numeric.to_numpy(dtype=float)).all():
        raise ValueError("fold aggregates contain missing or non-finite values")

    failures = models.loc[models["status"].eq("failed"), ["trial_id", "model", "status"]]
    profile = _load_time_profile(Path(performance_profile).resolve())
    alpha = float(family["alpha"])
    plot_models = models.copy()
    plot_models["multiplicity_result"] = np.where(
        plot_models["survives_correction"],
        "survives correction",
        "does not survive",
    )
    plot_models["training_status"] = np.where(
        plot_models["training_warning_count"].fillna(0).gt(0),
        "warning recorded",
        "clean",
    )

    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{destination.name}.", dir=destination.parent))
    try:
        folds.to_csv(staging / "fold_metrics.csv", index=False)
        models.to_csv(staging / "model_summary.csv", index=False)
        failures.to_csv(staging / "failures.csv", index=False)
        plot_fold_rank_ic(folds, staging / "plots/fold_rank_ic.png")
        plot_costed_returns(models, staging / "plots/costed_returns.png")
        plot_multiplicity(
            plot_models,
            staging / "plots/multiplicity_correction.png",
            alpha=alpha,
        )
        plot_compute_accounting(
            plot_models,
            staging / "plots/compute_accounting.png",
        )

        selected_model = str(study_summary["candidate_model"])
        selected = models.loc[models["model"].eq(selected_model)]
        if len(selected) != 1:
            raise ValueError("selected candidate is absent from the complete family")
        selected_row = selected.iloc[0]
        summary = {
            "evidence_schema_version": PUBLIC_STUDY_EVIDENCE_VERSION,
            "evidence_type": "aggregate governed seven-candidate historical baseline study",
            "decision": study_summary["decision"],
            "readiness_decision": study_summary["readiness_decision"],
            "selected_candidate": selected_model,
            "selected_candidate_evidence": {
                "mean_rank_ic": float(selected_row["mean_rank_ic"]),
                "incremental_rank_ic": float(selected_row["incremental_rank_ic"]),
                "raw_p_value": float(selected_row["raw_p_value"]),
                "adjusted_p_value": float(selected_row["adjusted_p_value"]),
                "net_annual_return": float(selected_row["net_annual_return"]),
                "max_drawdown": float(selected_row["max_drawdown"]),
                "kill_reasons": (
                    str(selected_row["kill_reasons"]).split(";")
                    if str(selected_row["kill_reasons"])
                    else []
                ),
            },
            "study": {
                "study_id": study_summary["study_id"],
                "plan_hash": study_summary["plan_hash"],
                "research_run_id": run.name,
                "producer_code_sha": experiment.get("code", {}).get("sha"),
                "ledger_head_hash": records[-1]["record_hash"],
                "ledger_record_count": len(records),
                "candidate_count": len(models),
                "fold_count_per_candidate": len(expected_windows or ()),
                "correction": family["method"],
                "family_alpha": alpha,
                "failed_candidate_count": int(len(failures)),
            },
            "source": {
                "bundle_id": bundle_manifest["bundle_id"],
                "schema_version": bundle_manifest.get("schema_version"),
                "rows": int(bundle_manifest.get("rows", 0)),
                "ticker_count": len(bundle_manifest.get("tickers", [])),
                "date_min": bundle_manifest.get("date_min"),
                "date_max": bundle_manifest.get("date_max"),
                "point_in_time_limits": dataset.get("point_in_time_limits"),
                "licensed_observations_published": False,
                "row_level_predictions_published": False,
                "final_holdout_artifacts_published": False,
                "provider_requests": 0,
            },
            "compute": {
                **profile,
                "scope": "one local seven-candidate cached-bundle governed study",
                "compute_path": "single-process CPU with each model backend bounded to one worker",
                "model_fit_count": int(models["model_fit_count"].fillna(0).sum()),
                "prediction_rows": int(models["prediction_rows"].fillna(0).sum()),
                "reported_training_iterations": int(models["training_iterations"].fillna(0).sum()),
                "environment": experiment.get("environment"),
            },
            "calibration": {
                "method": "OOF linear regression of realized target on prediction",
                "interpretation": (
                    "descriptive regression intercept/slope/RMSE; not probability calibration"
                ),
            },
            "limitations": study_summary["limitations"],
        }
        _write_json(staging / "summary.json", summary)
        (staging / "README.md").write_text(
            "# Signal Foundry Sprint 2 evidence\n\n"
            f"Decision: **{study_summary['decision']}**. Selected development candidate: "
            f"`{selected_model}`.\n\n"
            "This evidence is an aggregate historical study on the stale Nasdaq WIKI "
            "engineering bootstrap. It is not paper trading, live trading, a current market "
            "forecast, or a profit claim. The final holdout and every row-level observation, "
            "prediction, order, fill, position, and return remain outside Git.\n\n"
            "![Matched-fold rank IC](plots/fold_rank_ic.png)\n\n"
            "![Costed development returns](plots/costed_returns.png)\n\n"
            "![Multiplicity correction](plots/multiplicity_correction.png)\n\n"
            "![Compute accounting](plots/compute_accounting.png)\n\n"
            "See `summary.json`, `fold_metrics.csv`, and `model_summary.csv` for the "
            "machine-readable evidence, provenance, decision gates, and limitations.\n",
            encoding="utf-8",
        )
        artifacts = {
            path.relative_to(staging).as_posix(): {
                "bytes": path.stat().st_size,
                "sha256": _sha256(path),
            }
            for path in sorted(staging.rglob("*"))
            if path.is_file()
        }
        _write_json(
            staging / "manifest.json",
            {
                "evidence_schema_version": PUBLIC_STUDY_EVIDENCE_VERSION,
                "artifacts": artifacts,
                "source_bundle_id": bundle_manifest["bundle_id"],
                "source_study_id": study_summary["study_id"],
                "source_run_id": run.name,
                "producer_code_sha": experiment.get("code", {}).get("sha"),
                "config_sha256": _sha256(config),
                "publication_boundary": "aggregate-only; no licensed or row-level artifacts",
            },
        )
        os.replace(staging, destination)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return destination
