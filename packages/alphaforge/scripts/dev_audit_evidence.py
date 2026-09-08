"""Publish metadata-only dependency-surface evidence from saved audit responses."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Any, NoReturn

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from packaging.utils import canonicalize_name
from packaging.version import InvalidVersion, Version

from scripts.dev_audit import (
    MAX_BYTES,
    AuditContractError,
    Key,
    LockGraph,
    read_bounded,
    validate_export,
)


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise AuditContractError("duplicate-audit-json-key")
        result[key] = value
    return result


def _reject_constant(value: str) -> NoReturn:
    raise AuditContractError("nonstandard-audit-json-constant")


def audit_findings(content: bytes, expected: set[Key]) -> list[dict[str, Any]]:
    """Require exactly one complete advisory result per active locked dependency."""
    if not content or len(content) > MAX_BYTES:
        raise AuditContractError("audit-size-limit")
    try:
        report = json.loads(
            content, object_pairs_hook=_unique_object, parse_constant=_reject_constant
        )
    except (ValueError, UnicodeError, RecursionError) as exc:
        raise AuditContractError("invalid-audit-json") from exc
    if not isinstance(report, dict) or not isinstance(report.get("dependencies"), list):
        raise AuditContractError("invalid-audit-schema")
    dependencies = report["dependencies"]
    if len(dependencies) != len(expected) or report.get("fixes") != []:
        raise AuditContractError("incomplete-or-mutating-audit")
    seen = set()
    findings = []
    for item in dependencies:
        if (
            not isinstance(item, dict)
            or not isinstance(item.get("name"), str)
            or not isinstance(item.get("version"), str)
            or len(item["version"]) > 128
        ):
            raise AuditContractError("invalid-audit-package")
        try:
            key = (canonicalize_name(item["name"]), Version(item["version"]))
        except InvalidVersion as exc:
            raise AuditContractError("invalid-audit-version") from exc
        if (
            key not in expected
            or key in seen
            or "skip_reason" in item
            or not isinstance(item.get("vulns"), list)
        ):
            raise AuditContractError("skipped-or-mismatched-audit-package")
        seen.add(key)
        for vulnerability in item["vulns"]:
            if not isinstance(vulnerability, dict):
                raise AuditContractError("invalid-advisory-record")
            identifier = vulnerability.get("id")
            if not isinstance(identifier, str) or not re.fullmatch(
                r"(?:GHSA|CVE|PYSEC)-[A-Za-z0-9-]{1,80}", identifier
            ):
                raise AuditContractError("invalid-advisory-identifier")
            findings.append({"name": key[0], "version": str(key[1]), "id": identifier})
    return sorted(findings, key=lambda finding: (finding["name"], finding["id"]))


def build_evidence(
    before: bytes, after: bytes, requirements: bytes, old_audit: bytes, new_audit: bytes
) -> dict[str, Any]:
    """Bind saved service responses, export, package graph and exact lock delta.

    Historical service responses are inputs, not regenerated predictions. The
    figure is deterministic for these metadata inputs; future advisories can
    change a fresh network audit. No descriptions or source package bytes export.
    """
    validated = validate_export(after, requirements)
    env = validated["environment"]
    old, new = LockGraph(before), LockGraph(after)
    old_packages, new_packages = old.closure(("dev",), env), new.closure(("dev",), env)

    def unchanged(graph: LockGraph) -> dict[Key, dict[str, Any]]:
        return {
            key: value
            for key, value in graph.packages.items()
            if key[0] not in {"pip", "alphaforge"}
        }

    if unchanged(old) != unchanged(new):
        raise AuditContractError("unrelated-lock-drift")
    old_findings = audit_findings(old_audit, old_packages)
    new_findings = audit_findings(new_audit, new_packages)
    if new_findings:
        raise AuditContractError("current-audit-has-advisories")
    original_surface = new.closure(("data", "ml"), env)
    additional = new_packages - original_surface
    changed = sorted(str(key[1]) for key in new.packages if key[0] == "pip")
    return {
        "schema_version": 1,
        "scope": "host known-advisory and audited-dependency surface; not a security proof",
        "environment": env,
        "inputs_sha256": {
            name: hashlib.sha256(content).hexdigest()
            for name, content in (
                ("before_lock", before),
                ("after_lock", after),
                ("dev_export", requirements),
                ("before_audit", old_audit),
                ("after_audit", new_audit),
            )
        },
        "original_runtime_data_ml_packages": len(original_surface),
        "new_runtime_dev_packages": len(new_packages),
        "newly_audited_dev_packages": len(additional),
        "combined_audited_packages": len(original_surface | new_packages),
        "unchanged_third_party_package_records": len(unchanged(old)),
        "changed_pip_records": int(
            {key: value for key, value in old.packages.items() if key[0] == "pip"}
            != {key: value for key, value in new.packages.items() if key[0] == "pip"}
        ),
        "patched_pip_versions": changed,
        "before_findings": old_findings,
        "after_findings": new_findings,
        "newly_audited_pins": [
            {"name": name, "version": str(version)} for name, version in sorted(additional)
        ],
    }


def plot_evidence(evidence: dict[str, Any], destination: Path) -> None:
    """Plot exact package counts and service findings; no inferred risk score."""
    with sns.axes_style("whitegrid"), sns.plotting_context("notebook"):
        fig, axes = plt.subplots(1, 3, figsize=(13, 4.8), constrained_layout=True)
        try:
            panels = (
                (
                    "Audited dependency surface",
                    ["Existing gate", "With dev gate"],
                    [
                        evidence["original_runtime_data_ml_packages"],
                        evidence["combined_audited_packages"],
                    ],
                    "Active package versions",
                ),
                (
                    "Dev audit: known advisories",
                    ["Before patch", "After patch"],
                    [len(evidence["before_findings"]), len(evidence["after_findings"])],
                    "Advisory findings",
                ),
                (
                    "Lock change isolation",
                    ["Pip updated", "Others unchanged"],
                    [
                        evidence["changed_pip_records"],
                        evidence["unchanged_third_party_package_records"],
                    ],
                    "Third-party package records",
                ),
            )
            for axis, (title, labels, values, unit) in zip(axes, panels, strict=True):
                sns.barplot(
                    x=labels, y=values, hue=labels, palette="colorblind", legend=False, ax=axis
                )
                axis.set(title=title, ylabel=unit, xlabel="", ylim=(0, max(values) * 1.2 + 0.2))
                for index, value in enumerate(values):
                    axis.text(index, value, str(value), ha="center", va="bottom")
            env = evidence["environment"]
            fig.suptitle(
                f"Signal Foundry S6 security prerequisite | CPython {env['python_version']} / {env['sys_platform']}\nSaved advisory responses; package counts are not a measure of exploitability",
                fontsize=12,
            )
            fig.savefig(
                destination,
                dpi=150,
                metadata={"Software": "Signal Foundry dependency audit evidence"},
            )
        finally:
            plt.close(fig)


def publish(evidence: dict[str, Any], destination: Path) -> None:
    """Atomically publish a new directory; a reservation excludes peer writers."""
    destination = destination.parent.resolve() / destination.name
    if destination.exists() or destination.is_symlink():
        raise AuditContractError("evidence-destination-exists")
    reservation = destination.with_name(f".{destination.name}.reservation")
    try:
        reservation.mkdir()
    except OSError as exc:
        raise AuditContractError("evidence-destination-unavailable") from exc
    stage = None
    try:
        stage = Path(tempfile.mkdtemp(prefix=".dev-audit-", dir=destination.parent))
        plot_evidence(evidence, stage / "audit_surface.png")
        (stage / "evidence.json").write_text(json.dumps(evidence, indent=2, sort_keys=True) + "\n")
        hashes = {
            path.name: hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sorted(stage.iterdir())
        }
        (stage / "manifest.json").write_text(json.dumps(hashes, indent=2, sort_keys=True) + "\n")
        if destination.exists() or destination.is_symlink():
            raise AuditContractError("evidence-destination-race")
        os.rename(stage, destination)
        stage = None
    except OSError as exc:
        raise AuditContractError("evidence-publication-failed") from exc
    finally:
        if stage is not None:
            shutil.rmtree(stage)
        reservation.rmdir()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in (
        "before-lock",
        "after-lock",
        "requirements",
        "before-audit",
        "after-audit",
        "output",
    ):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    try:
        evidence = build_evidence(
            *(
                read_bounded(path)
                for path in (
                    args.before_lock,
                    args.after_lock,
                    args.requirements,
                    args.before_audit,
                    args.after_audit,
                )
            )
        )
        publish(evidence, args.output)
    except AuditContractError as exc:
        print(f"development audit evidence: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
