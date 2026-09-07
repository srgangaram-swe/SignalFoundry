"""Validate every committed AlphaForge YAML configuration before execution."""

from __future__ import annotations

from pathlib import Path

from alphaforge.config import SCHEMAS, load_config
from alphaforge.optimization.study import load_mean_variance_study_config
from alphaforge.research.cross_repository_provenance import (
    CrossRepositoryProvenanceError,
    load_cross_repository_receipt,
)
from alphaforge.research.deep_sequence_study import load_deep_sequence_study_config
from alphaforge.research.ensemble_study import load_ensemble_study_config
from alphaforge.research.representation_study import load_representation_study_config
from alphaforge.research.sprint_3_decision import load_sprint_3_evaluation_plan
from alphaforge.research.time_frequency_study import load_time_frequency_study_config

_SIGNALATTICE_REPOSITORY = "srgangaram-swe/Signalattice"
_SIGNALATTICE_ORIGIN = "https://github.com/srgangaram-swe/Signalattice.git"


def main() -> None:
    for kind in sorted(SCHEMAS):
        path = Path("configs") / f"{kind}.yaml"
        load_config(path, kind)
        print(f"validated {kind}: {path}")
    for profile in (
        "signal_foundry_wiki_bootstrap.yaml",
        "signal_foundry_sprint_2_study.yaml",
    ):
        path = Path("configs") / profile
        load_config(path, "signal_foundry_research")
        print(f"validated signal_foundry_research: {path}")
    deep_sequence_path = Path("configs/deep_sequence_benchmark.yaml")
    load_deep_sequence_study_config(deep_sequence_path)
    print(f"validated deep_sequence_study: {deep_sequence_path}")
    time_frequency_path = Path("configs/time_frequency_vision_benchmark.yaml")
    load_time_frequency_study_config(time_frequency_path)
    print(f"validated time_frequency_study: {time_frequency_path}")
    representation_path = Path("configs/latent_representation_benchmark.yaml")
    load_representation_study_config(representation_path)
    print(f"validated representation_study: {representation_path}")

    ensemble_path = Path("configs/ensemble_benchmark.yaml")
    load_ensemble_study_config(ensemble_path)
    print(f"validated ensemble_study: {ensemble_path}")

    receipt_path = Path("docs/evidence/signal_foundry_sprint_3/cross_repository_provenance.json")
    receipt = load_cross_repository_receipt(receipt_path)
    if receipt.repository != _SIGNALATTICE_REPOSITORY or receipt.origin_url != _SIGNALATTICE_ORIGIN:
        raise CrossRepositoryProvenanceError(
            "committed Sprint 3 receipt must identify the governed Signalattice origin"
        )
    print(
        f"validated cross_repository_receipt: {receipt_path} "
        f"(sources={len(receipt.sources)}, bytes={sum(source.bytes for source in receipt.sources)})"
    )
    sprint_3_path = Path("configs/sprint_3_decision.yaml")
    sprint_3_plan = load_sprint_3_evaluation_plan(
        sprint_3_path,
        repository_root=Path.cwd(),
    )
    print(
        f"validated sprint_3_decision: {sprint_3_path} "
        f"(plan_id={sprint_3_plan.plan_id}, families={len(sprint_3_plan.families)})"
    )

    mean_variance_path = Path("configs/mean_variance_study.yaml")
    mean_variance_config = load_mean_variance_study_config(mean_variance_path)
    print(
        f"validated mean_variance_study: {mean_variance_path} "
        f"(profile_id={mean_variance_config.profile_id}, "
        f"formulations={len(mean_variance_config.evidence.formulations)})"
    )


if __name__ == "__main__":
    main()
