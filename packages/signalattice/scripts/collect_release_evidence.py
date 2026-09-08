"""Collect machine-readable release evidence from a real dry run.

SF-S5-SL-MR7. Reads the artifacts a dry run actually produced -- the descriptor,
the inventory, the SBOM, and the provenance -- and records both the successful
verification and a set of deliberately tampered cases.

The tamper cases matter as much as the success: a supply-chain figure showing
only green bars documents that nothing was tested. Each case here is a real
mutation applied to a copy of the staged release, verified, and recorded with
the refusal it produced.

Output is redistribution-safe: counts, sizes, digests of public artifacts, and
refusal reasons. No signing material, credential, host path, or private content.
"""

from __future__ import annotations

import argparse
import json
import shutil
import tempfile
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from quant_platform.release.descriptor import ReleaseDescriptor
from quant_platform.release.inventory import (
    InventoryError,
    build_inventory,
    compare_inventories,
    inventory_from_dicts,
)
from quant_platform.release.provenance import (
    BUILDER_DRY_RUN,
    BUILDER_PUBLICATION,
    ProvenanceError,
    verify_provenance,
)

DESCRIPTOR_NAME = "release-descriptor.json"
INVENTORY_NAME = "release-inventory.json"
PROVENANCE_NAME = "provenance.intoto.json"
SBOM_NAME = "sbom.cyclonedx.json"
MANIFESTS = (DESCRIPTOR_NAME, INVENTORY_NAME, PROVENANCE_NAME)


class EvidenceError(RuntimeError):
    """Raised when release evidence cannot be collected."""


@dataclass(frozen=True, slots=True)
class TamperCase:
    """One deliberate mutation and whether verification caught it."""

    name: str
    description: str
    detected: bool
    refusal: str


def _verify_stage(stage: Path) -> None:
    """Verify a staged release, raising on the first failure.

    Raises:
        InventoryError | ProvenanceError: On the first guarantee that fails.
    """
    descriptor = ReleaseDescriptor.model_validate_json(
        (stage / DESCRIPTOR_NAME).read_text(encoding="utf-8")
    )
    recorded = inventory_from_dicts(
        json.loads((stage / INVENTORY_NAME).read_text(encoding="utf-8"))
    )
    observed = build_inventory(stage, exclude=MANIFESTS)
    difference = compare_inventories(recorded, observed)
    if not difference.clean:
        raise InventoryError(difference.describe())
    statement = json.loads((stage / PROVENANCE_NAME).read_text(encoding="utf-8"))
    verify_provenance(
        statement,
        expected_subjects=observed,
        expected_commit=descriptor.source_commit,
        expected_builder=BUILDER_DRY_RUN,
    )


def _largest_artifact(stage: Path) -> Path:
    """Return the largest non-manifest artifact, used as a tamper target."""
    candidates = [
        path
        for path in stage.rglob("*")
        if path.is_file() and path.name not in {*MANIFESTS, SBOM_NAME}
    ]
    if not candidates:
        raise EvidenceError("the staged release contains no artifacts to tamper with")
    return max(candidates, key=lambda path: path.stat().st_size)


def _tamper_flip_one_byte(stage: Path) -> None:
    """Flip a single bit in the largest artifact."""
    target = _largest_artifact(stage)
    payload = bytearray(target.read_bytes())
    payload[len(payload) // 2] ^= 0x01
    target.write_bytes(bytes(payload))


def _tamper_remove_artifact(stage: Path) -> None:
    """Delete an artifact the inventory expects."""
    _largest_artifact(stage).unlink()


def _tamper_add_artifact(stage: Path) -> None:
    """Add a file the inventory does not list."""
    (stage / "unexpected.txt").write_bytes(b"not part of this release\n")


def _tamper_rename_artifact(stage: Path) -> None:
    """Rename an artifact, keeping its bytes."""
    target = _largest_artifact(stage)
    target.rename(target.with_name(f"renamed-{target.name}"))


def _tamper_substitute_provenance_commit(stage: Path) -> None:
    """Rewrite the provenance to claim a different source commit."""
    path = stage / PROVENANCE_NAME
    statement = json.loads(path.read_text(encoding="utf-8"))
    resolved = statement["predicate"]["buildDefinition"]["resolvedDependencies"]
    for entry in resolved:
        if "gitCommit" in entry.get("digest", {}):
            entry["digest"]["gitCommit"] = "0" * 40
    path.write_text(json.dumps(statement, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _tamper_claim_publication_builder(stage: Path) -> None:
    """Relabel a dry-run attestation as a publication one."""
    path = stage / PROVENANCE_NAME
    statement = json.loads(path.read_text(encoding="utf-8"))
    statement["predicate"]["runDetails"]["builder"]["id"] = BUILDER_PUBLICATION
    path.write_text(json.dumps(statement, indent=2, sort_keys=True) + "\n", encoding="utf-8")


_TAMPERS: tuple[tuple[str, str, Callable[[Path], None]], ...] = (
    ("one_byte_flip", "flip a single bit inside the largest artifact", _tamper_flip_one_byte),
    ("removed_artifact", "delete an artifact the inventory expects", _tamper_remove_artifact),
    ("extra_artifact", "add a file the inventory does not list", _tamper_add_artifact),
    ("renamed_artifact", "rename an artifact but keep its bytes", _tamper_rename_artifact),
    (
        "wrong_source_commit",
        "rewrite provenance to claim a different source commit",
        _tamper_substitute_provenance_commit,
    ),
    (
        "dry_run_as_publication",
        "relabel a dry-run attestation as a publication one",
        _tamper_claim_publication_builder,
    ),
)


def run_tamper_cases(stage: Path) -> tuple[TamperCase, ...]:
    """Apply each tamper to a fresh copy and record whether it was caught."""
    cases: list[TamperCase] = []
    for name, description, mutate in _TAMPERS:
        with tempfile.TemporaryDirectory() as scratch:
            copy = Path(scratch) / "stage"
            shutil.copytree(stage, copy, symlinks=False)
            mutate(copy)
            # The publication-builder case is only meaningful against a verifier
            # that expects a dry-run builder, which _verify_stage does.
            try:
                _verify_stage(copy)
            except (InventoryError, ProvenanceError, ValueError) as error:
                cases.append(
                    TamperCase(
                        name=name,
                        description=description,
                        detected=True,
                        refusal=str(error)[:200],
                    )
                )
            else:
                cases.append(
                    TamperCase(
                        name=name, description=description, detected=False, refusal="not detected"
                    )
                )
    return tuple(cases)


def collect(stage: Path) -> dict[str, Any]:
    """Assemble the redistribution-safe release evidence document."""
    descriptor = ReleaseDescriptor.model_validate_json(
        (stage / DESCRIPTOR_NAME).read_text(encoding="utf-8")
    )
    sbom = json.loads((stage / SBOM_NAME).read_text(encoding="utf-8"))

    try:
        _verify_stage(stage)
        verified = True
        verification_detail = "the staged release verifies end to end"
    except (InventoryError, ProvenanceError, ValueError) as error:
        verified = False
        verification_detail = str(error)[:200]

    by_kind: dict[str, dict[str, int]] = {}
    for subject in descriptor.subjects:
        bucket = by_kind.setdefault(subject.kind, {"count": 0, "bytes": 0})
        bucket["count"] += 1
        bucket["bytes"] += subject.size_bytes

    cases = run_tamper_cases(stage)
    return {
        "schema_version": 1,
        "evidence_class": "measured_local_release_dry_run",
        "note": (
            "Produced by a non-publishing dry run. Digests establish content identity, not "
            "economic validity, independent review, or production readiness."
        ),
        "release_version": descriptor.release_version,
        "source_commit": descriptor.source_commit,
        "build_kind": descriptor.build_kind,
        "verified": verified,
        "verification_detail": verification_detail,
        "subjects_by_kind": dict(sorted(by_kind.items())),
        "subject_count": len(descriptor.subjects),
        "sbom_components": len(sbom.get("components", [])),
        "tamper_cases": [
            {
                "name": case.name,
                "description": case.description,
                "detected": case.detected,
                "refusal": case.refusal,
            }
            for case in cases
        ],
        "limitations": list(descriptor.limitations),
    }


def main(argv: list[str] | None = None) -> int:
    """Write the release evidence document."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--staging", type=Path, default=Path("build/release"))
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args(argv)

    try:
        document = collect(arguments.staging)
    except (EvidenceError, OSError, ValueError) as error:
        print(f"release evidence refused: {error}")
        return 2

    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(json.dumps(document, indent=2, sort_keys=True) + "\n")
    print(f"wrote {arguments.output}")
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    raise SystemExit(main())
