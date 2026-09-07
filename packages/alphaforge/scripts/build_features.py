from __future__ import annotations

import argparse
from dataclasses import asdict
from pathlib import Path
from typing import Any

from alphaforge.config import (
    load_data_config,
    load_feature_config,
    load_labels_config,
    load_models_config,
)
from alphaforge.data import load_prices
from alphaforge.features import FeatureCache, fingerprint_frame, materialize_feature_set
from alphaforge.labels import LabelContract, build_label_set, build_labels
from alphaforge.research import write_frame_artifact
from alphaforge.research.manifest import capture_git_context
from alphaforge.utils import save_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build feature and label panels.")
    parser.add_argument("--config", default="configs/features.yaml")
    parser.add_argument("--data-config", default="configs/data.yaml")
    parser.add_argument("--models-config", default="configs/models.yaml")
    parser.add_argument(
        "--labels-config",
        default=None,
        help=(
            "opt into governed label contracts; omit to preserve the legacy "
            "models-config horizon behavior"
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    feature_cfg = load_feature_config(args.config)
    data_cfg = load_data_config(args.data_config)
    panel, benchmark = load_prices(data_cfg)
    git_context = capture_git_context()
    if git_context["dirty"]:
        raise RuntimeError("feature materialization requires a clean Git worktree")
    cache_dir = feature_cfg.get("cache_dir")
    feature_set = materialize_feature_set(
        panel,
        benchmark,
        feature_cfg,
        dataset_id=fingerprint_frame(panel),
        code_version=git_context["sha"],
        cache=None if cache_dir is None else FeatureCache(cache_dir),
    )
    features = feature_set.frame
    label_set = None
    if args.labels_config is None:
        model_cfg = load_models_config(args.models_config)
        labels = build_labels(panel, benchmark, horizons=model_cfg.get("horizons", [1, 5, 20]))
    else:
        label_cfg = load_labels_config(args.labels_config)
        if label_cfg["benchmark_symbol"] != benchmark:
            raise ValueError(
                "labels benchmark_symbol must match the canonical data benchmark "
                f"({label_cfg['benchmark_symbol']!r} != {benchmark!r})"
            )
        label_set = build_label_set(panel, LabelContract.from_mapping(label_cfg))
        labels = label_set.values
    output_dir = Path(feature_cfg.get("output_dir", "data/processed"))
    output_dir.mkdir(parents=True, exist_ok=True)
    write_frame_artifact(panel, output_dir / "panel.table.json")
    write_frame_artifact(features, output_dir / "features.table.json")
    write_frame_artifact(labels, output_dir / "labels.table.json")
    lineage: dict[str, Any] = {
        "cache_key": feature_set.cache_key,
        "cache_hit": feature_set.cache_hit,
        "lineage": asdict(feature_set.lineage),
        "registry": feature_set.registry.manifest(),
    }
    if label_set is not None:
        write_frame_artifact(label_set.events, output_dir / "label_events.table.json")
        lineage["labels"] = label_set.manifest()
    save_json(lineage, output_dir / "feature_lineage.json")
    print(
        f"features: {features.shape}; labels: {labels.shape}; output={output_dir}; "
        f"cache_key={feature_set.cache_key[:12]}; cache_hit={feature_set.cache_hit}"
    )


if __name__ == "__main__":
    main()
