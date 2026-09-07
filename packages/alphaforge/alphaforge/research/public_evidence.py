"""Publish non-reconstructive evidence from a governed licensed-data run."""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import shutil
import tempfile
from pathlib import Path
from typing import Any

import pandas as pd

from alphaforge.config import load_signal_foundry_research_config
from alphaforge.visualization import (
    plot_capacity_sensitivity,
    plot_readiness_gates,
    plot_scenario_returns,
)

PUBLIC_EVIDENCE_SCHEMA_VERSION = "1.0.0"
MAX_JSON_BYTES = 8 * 1024 * 1024
MAX_CAPACITY_BYTES = 4 * 1024 * 1024
MAX_PROFILE_BYTES = 64 * 1024
PUBLIC_METRICS = (
    "n_days",
    "gross_total_return",
    "total_return",
    "gross_annual_return",
    "annual_return",
    "benchmark_annual_return",
    "annual_excess_return",
    "annual_volatility",
    "sharpe",
    "sortino",
    "max_drawdown",
    "hit_rate",
    "average_turnover",
    "average_gross_exposure",
    "average_net_exposure",
    "var_95_daily",
    "expected_shortfall_95_daily",
    "deflated_sharpe_prob",
    "probability_of_backtest_overfitting",
)
PUBLIC_SCENARIO_FIELDS = (
    "scenario",
    "accounting_reconciled",
    "annual_return",
    "annual_volatility",
    "sharpe",
    "sortino",
    "max_drawdown",
    "average_turnover",
)
PUBLIC_CAPACITY_FIELDS = (
    "scenario_aum",
    "fill_ratio",
    "aggregate_participation_rate",
    "participation_p95",
    "participation_max",
    "modeled_cost_bps_per_traded_notional",
    "capacity_constrained_fraction",
)


def _load_json(path: Path, *, max_bytes: int = MAX_JSON_BYTES) -> dict[str, Any]:
    if path.is_symlink():
        raise ValueError(f"evidence input must not be a symbolic link: {path}")
    size = path.stat().st_size
    if size <= 0 or size > max_bytes:
        raise ValueError(f"evidence input has invalid byte length {size}: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"evidence input must contain a JSON object: {path}")
    return payload


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_scalar(value: Any) -> bool | int | float | str | None:
    if value is None or isinstance(value, (bool, int, str)):
        return value
    numeric = float(value)
    return numeric if math.isfinite(numeric) else None


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(payload, sort_keys=True, indent=2, allow_nan=False, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )


def _load_time_profile(path: Path) -> dict[str, float | int]:
    """Parse the bounded fields emitted by macOS ``/usr/bin/time -l``."""
    if path.is_symlink() or not path.is_file():
        raise ValueError("performance profile must be a real file")
    size = path.stat().st_size
    if size <= 0 or size > MAX_PROFILE_BYTES:
        raise ValueError("performance profile has an invalid byte length")
    text = path.read_text(encoding="utf-8")
    duration = re.search(
        r"^\s*([0-9.]+) real\s+([0-9.]+) user\s+([0-9.]+) sys\s*$",
        text,
        flags=re.MULTILINE,
    )
    maximum_rss = re.search(
        r"^\s*(\d+)\s+maximum resident set size\s*$",
        text,
        flags=re.MULTILINE,
    )
    peak_footprint = re.search(
        r"^\s*(\d+)\s+peak memory footprint\s*$",
        text,
        flags=re.MULTILINE,
    )
    if duration is None or maximum_rss is None or peak_footprint is None:
        raise ValueError("performance profile lacks required macOS time fields")
    return {
        "wall_seconds": float(duration.group(1)),
        "user_cpu_seconds": float(duration.group(2)),
        "system_cpu_seconds": float(duration.group(3)),
        "maximum_resident_set_bytes": int(maximum_rss.group(1)),
        "peak_memory_footprint_bytes": int(peak_footprint.group(1)),
    }


def _validate_source(
    *,
    bundle: dict[str, Any],
    run_manifest: dict[str, Any],
    dossier: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    experiment = run_manifest.get("experiment")
    if not isinstance(experiment, dict):
        raise ValueError("run manifest lacks experiment metadata")
    dataset = experiment.get("dataset")
    if not isinstance(dataset, dict):
        raise ValueError("run manifest lacks dataset metadata")
    bundle_id = bundle.get("bundle_id")
    if (
        not isinstance(bundle_id, str)
        or len(bundle_id) != 64
        or dataset.get("bundle_id") != bundle_id
        or dossier.get("bundle_id") != bundle_id
    ):
        raise ValueError("bundle identity does not match governed run evidence")
    license_policy = bundle.get("license")
    if not isinstance(license_policy, dict):
        raise ValueError("bundle lacks license policy")
    if license_policy.get("observations_redistributable") is not False:
        raise ValueError("this publisher is restricted to non-redistributable source bundles")
    if license_policy.get("public_evidence_must_be_aggregate_or_synthetic") is not True:
        raise ValueError("bundle does not require the supported aggregate-only evidence policy")
    if dossier.get("decision") not in {"READY_FOR_PAPER", "NOT_READY"}:
        raise ValueError("dossier has an unsupported readiness decision")
    if dossier.get("decision") == "READY_FOR_PAPER" and not all(
        bool(value) for value in bundle.get("point_in_time_limits", {}).values()
    ):
        raise ValueError("incomplete point-in-time evidence cannot publish READY_FOR_PAPER")
    return experiment, dataset


def publish_signal_foundry_evidence(
    *,
    run_dir: str | Path,
    bundle_dir: str | Path,
    config_path: str | Path,
    output_dir: str | Path,
    performance_profile: str | Path | None = None,
) -> Path:
    """Publish aggregate JSON/CSV and Seaborn plots without licensed observations.

    The destination must not exist. Only an explicit allowlist of aggregate
    dossier and capacity fields crosses the publication boundary.
    """
    run = Path(run_dir).resolve()
    bundle_root = Path(bundle_dir).resolve()
    config_source = Path(config_path).resolve()
    destination = Path(output_dir).resolve()
    if destination.exists():
        raise FileExistsError(f"public evidence destination already exists: {destination}")
    for source in (run, bundle_root):
        if not source.is_dir() or source.is_symlink():
            raise ValueError(f"evidence source must be a real directory: {source}")

    bundle = _load_json(bundle_root / "manifest.json")
    run_manifest = _load_json(run / "run_manifest.json")
    dossier = _load_json(run / "dossier.json")
    experiment, dataset = _validate_source(
        bundle=bundle,
        run_manifest=run_manifest,
        dossier=dossier,
    )
    config = load_signal_foundry_research_config(config_source)
    profile = (
        _load_time_profile(Path(performance_profile).resolve())
        if performance_profile is not None
        else None
    )

    gates_raw = dossier.get("gates")
    scenarios_raw = dossier.get("scenarios")
    metrics_raw = dossier.get("metrics")
    if not isinstance(gates_raw, dict) or not all(
        isinstance(name, str) and isinstance(passed, bool) for name, passed in gates_raw.items()
    ):
        raise ValueError("dossier readiness gates are malformed")
    if not isinstance(scenarios_raw, list) or not all(
        isinstance(record, dict) for record in scenarios_raw
    ):
        raise ValueError("dossier scenario evidence is malformed")
    if not isinstance(metrics_raw, dict):
        raise ValueError("dossier metrics are malformed")

    capacity_path = run / "capacity_curve.csv"
    if capacity_path.is_symlink() or capacity_path.stat().st_size > MAX_CAPACITY_BYTES:
        raise ValueError("capacity evidence exceeds its safe input boundary")
    capacity_raw = pd.read_csv(capacity_path)
    missing_capacity = set(PUBLIC_CAPACITY_FIELDS) - set(capacity_raw)
    if missing_capacity:
        raise ValueError(f"capacity evidence missing columns: {sorted(missing_capacity)}")

    gates = pd.DataFrame(
        [{"gate": name, "passed": passed} for name, passed in sorted(gates_raw.items())]
    )
    primary = {
        "scenario": "primary",
        "accounting_reconciled": bool(metrics_raw.get("accounting_reconciled")),
        **{
            field: _json_scalar(metrics_raw.get(field))
            for field in PUBLIC_SCENARIO_FIELDS
            if field not in {"scenario", "accounting_reconciled"}
        },
    }
    scenarios = pd.DataFrame(
        [
            primary,
            *[
                {field: _json_scalar(record.get(field)) for field in PUBLIC_SCENARIO_FIELDS}
                for record in scenarios_raw
            ],
        ],
        columns=PUBLIC_SCENARIO_FIELDS,
    )
    capacity = capacity_raw.loc[:, PUBLIC_CAPACITY_FIELDS].copy()
    if capacity.empty or not all(
        pd.to_numeric(capacity[column], errors="coerce").notna().all()
        for column in PUBLIC_CAPACITY_FIELDS
    ):
        raise ValueError("capacity aggregate contains missing or nonnumeric values")

    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{destination.name}.", dir=destination.parent))
    try:
        gates.to_csv(staging / "readiness_gates.csv", index=False)
        scenarios.to_csv(staging / "scenario_metrics.csv", index=False)
        capacity.to_csv(staging / "capacity_sensitivity.csv", index=False)
        plot_readiness_gates(gates, staging / "plots/readiness_gates.png")
        plot_scenario_returns(scenarios, staging / "plots/scenario_returns.png")
        plot_capacity_sensitivity(
            capacity,
            staging / "plots/capacity_sensitivity.png",
            minimum_fill_ratio=float(config["backtest"]["capacity"]["minimum_fill_ratio"]),
        )

        summary = {
            "evidence_schema_version": PUBLIC_EVIDENCE_SCHEMA_VERSION,
            "evidence_type": "aggregate historical WIKI engineering-bootstrap backtest",
            "decision": dossier["decision"],
            "failed_gates": sorted(dossier.get("failed_gates", [])),
            "run": {
                "run_id": dossier.get("run_id"),
                "code_sha": experiment.get("code", {}).get("sha"),
                "candidate_model": dossier.get("candidate_model"),
                "holdout_start": dossier.get("holdout_start"),
                "development_end": dossier.get("development_end"),
                "trial_count": len(experiment.get("models", [])),
            },
            "source": {
                "bundle_id": bundle["bundle_id"],
                "schema_version": bundle.get("schema_version"),
                "rows": int(bundle.get("rows", 0)),
                "ticker_count": len(bundle.get("tickers", [])),
                "date_min": bundle.get("date_min"),
                "date_max": bundle.get("date_max"),
                "partition_count": len(bundle.get("files", [])),
                "universe_rows": int(bundle.get("universe_rows", 0)),
                "corporate_action_rows": int(bundle.get("corporate_action_rows", 0)),
                "point_in_time_limits": dataset.get("point_in_time_limits"),
                "licensed_observations_published": False,
                "provider_requests": 0,
                "consumer_exclusions": {
                    "rows": 0,
                    "tickers": 0,
                    "dates": 0,
                    "policy": "fail the complete bundle rather than silently exclude invalid input",
                    "producer_exclusions_declared": "not present in source manifest",
                },
            },
            "metrics": {
                field: _json_scalar(metrics_raw.get(field))
                for field in PUBLIC_METRICS
                if field in metrics_raw
            },
            "paper_controls": {
                "all_controls_passed": bool(
                    dossier.get("paper_controls", {}).get("all_controls_passed")
                ),
                "broker_adapter_present": bool(
                    dossier.get("paper_controls", {}).get("broker_adapter_present")
                ),
                "executable_orders_emitted": bool(
                    dossier.get("paper_controls", {}).get("executable_orders_emitted")
                ),
            },
            "performance": (
                {
                    **profile,
                    "scope": "single local cached-bundle governed run",
                    "provider_requests": 0,
                    "compute_path": (
                        "CPU sklearn/NumPy; accelerator inventory records availability, not use"
                    ),
                    "environment": {
                        "python": experiment.get("environment", {}).get("python"),
                        "operating_system": experiment.get("environment", {}).get(
                            "operating_system"
                        ),
                        "hardware": experiment.get("environment", {}).get("hardware"),
                    },
                }
                if profile is not None
                else None
            ),
            "limitations": [
                "The WIKI bundle ends in 2018 and is stale, current-vintage data.",
                "Universe membership, historical revisions, and corporate actions are incomplete.",
                "The evidence validates historical pipeline mechanics, not current paper/live readiness.",
                "Backtested returns are not realized profits and do not predict future performance.",
                "No licensed observation, ticker list, order, fill, position, or date-level return is public.",
                "The performance profile is one local run, not a latency distribution or capacity SLA.",
            ],
        }
        _write_json(staging / "summary.json", summary)
        (staging / "README.md").write_text(
            "# Signal Foundry Sprint 1 evidence\n\n"
            f"Decision: **{dossier['decision']}**.\n\n"
            "This is redistribution-safe aggregate evidence from the stale Nasdaq WIKI "
            "engineering bootstrap. It validates the governed historical research path; "
            "it is not paper-trading or live-trading evidence and is not a profit claim.\n\n"
            "![Readiness gates](plots/readiness_gates.png)\n\n"
            "![Adversarial scenarios](plots/scenario_returns.png)\n\n"
            "![Capacity sensitivity](plots/capacity_sensitivity.png)\n\n"
            "See `summary.json` for provenance, failed gates, and limitations. The licensed "
            "bundle and all row-level run artifacts remain outside Git.\n",
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
                "evidence_schema_version": PUBLIC_EVIDENCE_SCHEMA_VERSION,
                "artifacts": artifacts,
                "source_bundle_id": bundle["bundle_id"],
                "source_run_id": dossier.get("run_id"),
                "config_sha256": _sha256(config_source),
                "publication_boundary": "aggregate-only; no licensed rows",
            },
        )
        os.replace(staging, destination)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return destination
