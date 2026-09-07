"""Generate deterministic, redistribution-safe label diagnostics and Seaborn plots."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any

from alphaforge.config import load_labels_config
from alphaforge.data import SyntheticMarketConfig, generate_synthetic_market
from alphaforge.labels import LabelContract, build_label_set, diagnose_labels
from alphaforge.visualization import save_label_diagnostic_plots

EVIDENCE_SCHEMA_VERSION = "1.0.0"
SYNTHETIC_SEED = 20260725
SYNTHETIC_SYMBOLS = 10
SYNTHETIC_SESSIONS = 756


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate synthetic financial-label diagnostics without network access."
    )
    parser.add_argument("--config", default="configs/labels.yaml")
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

    config = load_labels_config(args.config)
    panel = generate_synthetic_market(
        SyntheticMarketConfig(
            n_symbols=SYNTHETIC_SYMBOLS,
            n_days=SYNTHETIC_SESSIONS,
            seed=SYNTHETIC_SEED,
        )
    )
    generated_benchmark = "BENCH"
    configured_benchmark = config["benchmark_symbol"]
    panel.loc[panel["symbol"].eq(generated_benchmark), "symbol"] = configured_benchmark

    contract = LabelContract.from_mapping(config)
    dataset = build_label_set(panel, contract)
    diagnostic_config = config["diagnostics"]
    scales = diagnostic_config["sensitivity_scales"]
    variants = {
        float(scale): (
            dataset
            if float(scale) == 1.0
            else build_label_set(panel, contract.scaled(float(scale)))
        )
        for scale in scales
    }
    diagnostics = diagnose_labels(
        dataset,
        autocorrelation_lag=diagnostic_config["autocorrelation_lag"],
        periods=diagnostic_config["temporal_periods"],
        sensitivity_variants=variants,
    )

    staging = Path(tempfile.mkdtemp(prefix=f".{destination.name}.", dir=destination.parent))
    try:
        diagnostics.summary.to_csv(staging / "summary.csv", index=False)
        diagnostics.class_balance.to_csv(staging / "class_balance.csv", index=False)
        diagnostics.temporal_stability.to_csv(staging / "temporal_stability.csv", index=False)
        diagnostics.parameter_sensitivity.to_csv(staging / "parameter_sensitivity.csv", index=False)
        save_label_diagnostic_plots(diagnostics, staging / "plots")

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
                "evidence_type": "deterministic synthetic label diagnostics",
                "data": {
                    "source": "alphaforge synthetic market",
                    "seed": SYNTHETIC_SEED,
                    "tradable_symbols": SYNTHETIC_SYMBOLS,
                    "sessions": SYNTHETIC_SESSIONS,
                    "licensed_observations": False,
                    "market_evidence": False,
                },
                "label_dataset": dataset.manifest(),
                "diagnostics": diagnostic_config,
                "limitations": [
                    "Synthetic distributions validate engineering behavior, not trading value.",
                    "Dependence and class balance on generated data do not predict live behavior.",
                    "Sensitivity scales were predeclared and were not selected on a final holdout.",
                ],
                "artifacts": artifacts,
            },
        )
        os.replace(staging, destination)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise

    print(
        f"published label evidence: {destination} "
        f"({len(dataset.values):,} rows, contract={contract.identity[:12]})"
    )


if __name__ == "__main__":
    main()
