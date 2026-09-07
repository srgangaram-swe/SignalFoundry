"""Generate deterministic, redistribution-safe temporal-validation evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any

from alphaforge.data import SyntheticMarketConfig, generate_synthetic_market
from alphaforge.labels import LabelContract, LabelDefinition, build_label_set
from alphaforge.training import (
    TemporalValidationConfig,
    fold_assignments,
    fold_metadata,
    make_temporal_validation_plan,
    temporal_plan_identity,
)
from alphaforge.visualization import plot_temporal_folds

EVIDENCE_SCHEMA_VERSION = "1.0.0"
SYNTHETIC_SEED = 20260725
SYNTHETIC_SESSIONS = 756


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate synthetic temporal-fold metadata and a Seaborn visualization."
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="A new directory; existing paths fail closed to preserve evidence immutability.",
    )
    return parser.parse_args()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(payload, sort_keys=True, indent=2, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    args = parse_args()
    destination = Path(args.output_dir).resolve()
    if destination.exists():
        raise FileExistsError(
            f"output directory already exists and will not be overwritten: {destination}"
        )
    destination.parent.mkdir(parents=True, exist_ok=True)

    panel = generate_synthetic_market(
        SyntheticMarketConfig(
            n_symbols=6,
            n_days=SYNTHETIC_SESSIONS,
            seed=SYNTHETIC_SEED,
            benchmark_symbol="SPY",
        )
    )
    label_dataset = build_label_set(
        panel,
        LabelContract(
            benchmark_symbol="SPY",
            definitions=(
                LabelDefinition(
                    name="forward_return_20",
                    kind="regression",
                    horizon=20,
                ),
            ),
        ),
    )
    config = TemporalValidationConfig(
        scheme="expanding",
        min_train_sessions=252,
        validation_sessions=63,
        test_sessions=63,
        step_sessions=63,
        purge_sessions=20,
        embargo_sessions=5,
        final_holdout_sessions=126,
    )
    folds = make_temporal_validation_plan(
        panel["date"],
        config,
        label_events=label_dataset.events,
    )

    staging = Path(tempfile.mkdtemp(prefix=f".{destination.name}.", dir=destination.parent))
    try:
        assignments = fold_assignments(folds)
        metadata = fold_metadata(folds)
        assignments.to_csv(staging / "fold_assignments.csv", index=False)
        metadata.to_csv(staging / "fold_metadata.csv", index=False)
        plot_temporal_folds(folds, staging / "temporal_folds.png")
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
                "schema_version": EVIDENCE_SCHEMA_VERSION,
                "evidence_type": "deterministic synthetic temporal validation",
                "plan_identity": temporal_plan_identity(folds),
                "configuration": {
                    key: value for key, value in config.__dict__.items() if value is not None
                },
                "data": {
                    "source": "alphaforge synthetic market",
                    "seed": SYNTHETIC_SEED,
                    "sessions": SYNTHETIC_SESSIONS,
                    "licensed_observations": False,
                    "market_evidence": False,
                },
                "folds": len(folds),
                "limitations": [
                    "Synthetic folds prove temporal mechanics, not predictive or trading value.",
                    "Session-count gaps complement exact supplied label intervals.",
                    "The final holdout is visualized only as an inaccessible boundary.",
                ],
                "artifacts": artifacts,
            },
        )
        os.replace(staging, destination)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise

    print(
        f"published temporal-validation evidence: {destination} "
        f"(folds={len(folds)}, identity={temporal_plan_identity(folds)[:12]})"
    )


if __name__ == "__main__":
    main()
