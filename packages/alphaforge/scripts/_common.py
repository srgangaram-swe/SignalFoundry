from __future__ import annotations

import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd

from alphaforge.config import load_data_config, load_feature_config, load_models_config
from alphaforge.research import (
    ExperimentManifest,
    capture_environment,
    capture_git_context,
    inventory_artifacts,
    redact_cli_arguments,
    write_experiment_manifest,
)
from alphaforge.research.manifest import sha256_file
from alphaforge.utils import save_json, timestamp_id


def make_run_dir(runs_dir: str | Path = "runs", prefix: str = "run") -> Path:
    run_dir = Path(runs_dir) / timestamp_id(prefix)
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def write_latest(run_dir: Path) -> None:
    run_dir.parent.mkdir(parents=True, exist_ok=True)
    (run_dir.parent / "latest_run.txt").write_text(str(run_dir.resolve()))


def latest_run_dir(runs_dir: str | Path = "runs") -> Path:
    pointer = Path(runs_dir) / "latest_run.txt"
    if not pointer.exists():
        raise FileNotFoundError("no latest run found; run scripts/run_walk_forward.py first")
    run_dir = Path(pointer.read_text().strip())
    if not run_dir.exists():
        raise FileNotFoundError(f"latest run directory does not exist: {run_dir}")
    return run_dir


def load_configs(
    model_config: str | Path = "configs/models.yaml",
    data_config: str | Path = "configs/data.yaml",
    feature_config: str | Path = "configs/features.yaml",
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    return (
        load_models_config(model_config),
        load_data_config(data_config),
        load_feature_config(feature_config),
    )


def configure_fast_demo(model_cfg: dict[str, Any], data_cfg: dict[str, Any]) -> None:
    data_cfg["source"] = "synthetic"
    data_cfg.setdefault("synthetic", {})
    data_cfg["synthetic"].update({"n_symbols": 8, "n_days": 420, "seed": 42})
    model_cfg["models"] = [
        {"name": "zero_baseline"},
        {"name": "momentum_baseline", "params": {"feature": "momentum_20", "scale": 0.05}},
        {"name": "ridge", "params": {"alpha": 10.0}},
    ]
    model_cfg.setdefault("walk_forward", {})
    model_cfg["walk_forward"].update(
        {
            "min_train_days": 180,
            "test_days": 40,
            "step_days": 40,
            "embargo_days": max(model_cfg.get("horizons", [20])),
            "max_windows": 2,
        }
    )


def save_meta(run_dir: Path, **meta: Any) -> None:
    save_json(meta, run_dir / "run_meta.json")


def utc_timestamp() -> str:
    """Return the current UTC timestamp in the manifest's canonical form."""

    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def record_pipeline_manifest(
    *,
    run_dir: Path,
    panel: pd.DataFrame,
    data_config: dict[str, Any],
    model_config: dict[str, Any],
    feature_config: dict[str, Any],
    started_at: str,
    invocation: list[str] | None = None,
    transaction_costs: dict[str, Any] | None = None,
) -> ExperimentManifest:
    """Record a deterministic manifest for a local walk-forward or training run."""

    panel_path = run_dir / "panel.table.json"
    if not panel_path.is_file():
        raise FileNotFoundError("panel.table.json must exist before manifest publication")
    manifest = ExperimentManifest.build(
        code=capture_git_context(),
        dataset={
            "id": sha256_file(panel_path),
            "source": data_config["source"],
            "observations_redistributable": data_config["source"] == "synthetic",
        },
        universe=sorted(str(symbol) for symbol in panel["symbol"].unique()),
        date_range={
            "start": str(pd.Timestamp(panel["date"].min()).date()),
            "end": str(pd.Timestamp(panel["date"].max()).date()),
        },
        features=feature_config,
        label={
            "target": model_config["target"],
            "horizons": model_config["horizons"],
        },
        models=model_config["models"],
        validation=model_config["walk_forward"],
        transaction_costs=transaction_costs or {},
        root_seed=int(model_config["seed"]),
        environment=capture_environment(),
        invocation={
            "entrypoint": Path(sys.argv[0]).name,
            "arguments": redact_cli_arguments(invocation or sys.argv[1:]),
        },
        execution={"started_at": started_at, "finished_at": utc_timestamp()},
        artifacts=inventory_artifacts(run_dir, exclude=("run_manifest.json",)),
    )
    write_experiment_manifest(manifest, run_dir / "run_manifest.json")
    return manifest
