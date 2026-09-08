"""In-toto / SLSA provenance and CycloneDX SBOM generation.

SF-S5-SL-MR7. Provenance answers *who built these bytes, from what, and how*.
It is only useful if a verifier can reject a statement that does not match, so
this module is written around the rejections:

* the statement names every subject by digest, so a subject that was not built
  cannot be attached to it;
* it names the exact source commit, the builder identity, and the locked
  dependency digest, so a statement from a different commit, a different
  workflow, or a different lockfile is detectable;
* dry-run and publication builder identities are **different URIs**, so a
  dry-run attestation can never be presented as a publication one.

The SBOM is CycloneDX 1.6 in JSON, generated from locked resolutions rather than
from an installed environment: an SBOM that described whatever happened to be
importable would drift from the artifact it claims to describe.
"""

from __future__ import annotations

import json
import re
import tomllib
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Final

from quant_platform.release.inventory import Subject, digest_file

#: In-toto statement type and SLSA predicate this repository emits.
STATEMENT_TYPE: Final = "https://in-toto.io/Statement/v1"
PREDICATE_TYPE: Final = "https://slsa.dev/provenance/v1"

#: Builder identities. These are deliberately distinct strings: a verifier that
#: expects the publication builder must reject a dry-run attestation outright.
BUILDER_DRY_RUN: Final = "https://github.com/srgangaram-swe/Signalattice/release/dry-run"
BUILDER_PUBLICATION: Final = "https://github.com/srgangaram-swe/Signalattice/release/publication"

#: CycloneDX specification this repository emits.
CYCLONEDX_SPEC_VERSION: Final = "1.6"

_PURL_SAFE: Final = re.compile(r"^[A-Za-z0-9._-]+$")


class ProvenanceError(ValueError):
    """Raised when provenance or an SBOM cannot be produced or verified."""


@dataclass(frozen=True, slots=True)
class BuildContext:
    """Everything a provenance statement binds a build to."""

    source_commit: str
    build_kind: str
    lockfile_digest: str
    invocation: tuple[str, ...]
    started_at: datetime
    finished_at: datetime

    def __post_init__(self) -> None:
        if len(self.source_commit) != 40:
            raise ProvenanceError("source_commit must be a full 40-character commit")
        if self.build_kind not in {"dry-run", "publication"}:
            raise ProvenanceError("build_kind must be 'dry-run' or 'publication'")
        if len(self.lockfile_digest) != 64:
            raise ProvenanceError("lockfile_digest must be a SHA-256 digest")
        if not self.invocation:
            raise ProvenanceError("provenance must record the commands that produced the build")
        if self.finished_at < self.started_at:
            raise ProvenanceError("build finished before it started")

    @property
    def builder_id(self) -> str:
        """Return the builder identity for this build kind."""
        return BUILDER_DRY_RUN if self.build_kind == "dry-run" else BUILDER_PUBLICATION


def build_provenance(
    subjects: Sequence[Subject], context: BuildContext, *, release_version: str
) -> dict[str, Any]:
    """Return an in-toto statement carrying a SLSA v1 provenance predicate.

    Raises:
        ProvenanceError: If there are no subjects to attest to.
    """
    if not subjects:
        raise ProvenanceError("provenance must attest to at least one subject")
    return {
        "_type": STATEMENT_TYPE,
        "subject": [
            {"name": item.path, "digest": {"sha256": item.sha256}}
            for item in sorted(subjects, key=lambda entry: entry.path)
        ],
        "predicateType": PREDICATE_TYPE,
        "predicate": {
            "buildDefinition": {
                "buildType": "https://github.com/srgangaram-swe/Signalattice/release/v1",
                "externalParameters": {
                    "releaseVersion": release_version,
                    "buildKind": context.build_kind,
                },
                "internalParameters": {"invocation": list(context.invocation)},
                "resolvedDependencies": [
                    {
                        "uri": "git+https://github.com/srgangaram-swe/Signalattice",
                        "digest": {"gitCommit": context.source_commit},
                    },
                    {"name": "uv.lock", "digest": {"sha256": context.lockfile_digest}},
                ],
            },
            "runDetails": {
                "builder": {"id": context.builder_id},
                "metadata": {
                    "invocationId": f"{context.build_kind}-{context.source_commit[:12]}",
                    "startedOn": context.started_at.astimezone(UTC).isoformat(),
                    "finishedOn": context.finished_at.astimezone(UTC).isoformat(),
                },
            },
        },
    }


def verify_provenance(
    statement: Mapping[str, Any],
    *,
    expected_subjects: Sequence[Subject],
    expected_commit: str,
    expected_builder: str,
) -> None:
    """Refuse a statement that does not bind exactly this build.

    Every check is a separate refusal with its own message, because "provenance
    failed" tells an operator nothing about which guarantee broke.

    Raises:
        ProvenanceError: Naming the first guarantee that does not hold.
    """
    if statement.get("_type") != STATEMENT_TYPE:
        raise ProvenanceError("statement is not an in-toto v1 statement")
    if statement.get("predicateType") != PREDICATE_TYPE:
        raise ProvenanceError("statement does not carry a SLSA v1 provenance predicate")

    predicate = statement.get("predicate")
    if not isinstance(predicate, Mapping):
        raise ProvenanceError("statement has no predicate object")

    builder = predicate.get("runDetails", {}).get("builder", {}).get("id")
    if builder != expected_builder:
        raise ProvenanceError(
            f"provenance was produced by {builder!r} but {expected_builder!r} was required; "
            "a dry-run attestation is not a publication attestation"
        )

    resolved = predicate.get("buildDefinition", {}).get("resolvedDependencies", [])
    commits = [
        item.get("digest", {}).get("gitCommit")
        for item in resolved
        if isinstance(item, Mapping) and "gitCommit" in item.get("digest", {})
    ]
    if expected_commit not in commits:
        raise ProvenanceError(f"provenance does not bind source commit {expected_commit[:12]}")

    attested: dict[str, str] = {}
    for item in statement.get("subject", []):
        if not isinstance(item, Mapping):
            continue
        name = item.get("name")
        recorded_digest = item.get("digest", {}).get("sha256")
        if isinstance(name, str) and isinstance(recorded_digest, str):
            attested[name] = recorded_digest
    for subject in expected_subjects:
        recorded = attested.get(subject.path)
        if recorded is None:
            raise ProvenanceError(f"provenance does not attest to {subject.path}")
        if recorded != subject.sha256:
            raise ProvenanceError(
                f"provenance digest for {subject.path} does not match the built artifact"
            )
    extra = set(attested) - {item.path for item in expected_subjects}
    if extra:
        raise ProvenanceError(
            f"provenance attests to subjects that were not built: {', '.join(sorted(extra))}"
        )


def _locked_packages(lockfile: Path) -> tuple[tuple[str, str], ...]:
    """Return ``(name, version)`` for every package in the uv lock.

    Raises:
        ProvenanceError: If the lockfile is unreadable or has no packages.
    """
    try:
        document = tomllib.loads(lockfile.read_text(encoding="utf-8"))
    except OSError as error:
        raise ProvenanceError(f"cannot read {lockfile}") from error
    except tomllib.TOMLDecodeError as error:
        raise ProvenanceError(f"{lockfile} is not valid TOML") from error
    packages = document.get("package")
    if not isinstance(packages, list) or not packages:
        raise ProvenanceError("lockfile declares no packages")
    resolved: list[tuple[str, str]] = []
    for entry in packages:
        name = entry.get("name")
        version = entry.get("version")
        if isinstance(name, str) and isinstance(version, str):
            resolved.append((name, version))
    return tuple(sorted(set(resolved)))


def _node_packages(lockfile: Path) -> tuple[tuple[str, str], ...]:
    """Return ``(name, version)`` for every package in an npm lockfile."""
    try:
        document = json.loads(lockfile.read_text(encoding="utf-8"))
    except OSError as error:
        raise ProvenanceError(f"cannot read {lockfile}") from error
    except ValueError as error:
        raise ProvenanceError(f"{lockfile} is not valid JSON") from error
    packages = document.get("packages")
    if not isinstance(packages, dict):
        raise ProvenanceError("npm lockfile declares no packages object")
    resolved: list[tuple[str, str]] = []
    for location, entry in packages.items():
        if not location or not isinstance(entry, dict):
            continue
        name = entry.get("name") or location.rsplit("node_modules/", 1)[-1]
        version = entry.get("version")
        if isinstance(name, str) and isinstance(version, str):
            resolved.append((name, version))
    return tuple(sorted(set(resolved)))


def _purl(ecosystem: str, name: str, version: str) -> str:
    """Return a package URL, or a bounded fallback for an unusual name."""
    if _PURL_SAFE.match(name) is None:
        # Names that need escaping are rare and a silent mangle would make the
        # SBOM unmatchable, so the raw name is preserved in the component and
        # the purl is marked as unavailable instead.
        return f"pkg:{ecosystem}/unencodable@{version}"
    return f"pkg:{ecosystem}/{name}@{version}"


def build_sbom(
    repository_root: Path, *, release_version: str, source_commit: str
) -> dict[str, Any]:
    """Return a CycloneDX 1.6 SBOM covering Python and Node dependencies.

    Generated from the committed lockfiles rather than an installed
    environment, so it describes what the release resolves to rather than what
    a build machine happened to have.

    Raises:
        ProvenanceError: If a required lockfile is missing or unusable.
    """
    python_packages = _locked_packages(repository_root / "uv.lock")
    node_lock = repository_root / "web" / "package-lock.json"
    node_packages = _node_packages(node_lock) if node_lock.is_file() else ()

    components = [
        {
            "type": "library",
            "name": name,
            "version": version,
            "purl": _purl("pypi", name, version),
            "scope": "required",
        }
        for name, version in python_packages
    ] + [
        {
            "type": "library",
            "name": name,
            "version": version,
            "purl": _purl("npm", name, version),
            "scope": "required",
        }
        for name, version in node_packages
    ]

    return {
        "bomFormat": "CycloneDX",
        "specVersion": CYCLONEDX_SPEC_VERSION,
        "version": 1,
        # A fixed serial derived from the release identity, not a random UUID:
        # two clean builds of the same release must produce identical bytes.
        "serialNumber": f"urn:uuid:{_deterministic_uuid(release_version, source_commit)}",
        "metadata": {
            "component": {
                "type": "application",
                "name": "signalattice",
                "version": release_version,
                "purl": _purl("pypi", "signalattice", release_version),
            },
            "properties": [
                {"name": "signalattice:sourceCommit", "value": source_commit},
                {"name": "signalattice:pythonPackages", "value": str(len(python_packages))},
                {"name": "signalattice:nodePackages", "value": str(len(node_packages))},
            ],
        },
        "components": components,
    }


def _deterministic_uuid(release_version: str, source_commit: str) -> str:
    """Return a stable UUID-shaped identifier for a release identity."""
    import hashlib

    material = hashlib.sha256(f"{release_version}:{source_commit}".encode()).hexdigest()
    return "-".join(
        (material[0:8], material[8:12], material[12:16], material[16:20], material[20:32])
    )


def lockfile_digest(repository_root: Path) -> str:
    """Return the SHA-256 of the committed Python lockfile."""
    digest, _size = digest_file(repository_root / "uv.lock")
    return digest


__all__ = [
    "BUILDER_DRY_RUN",
    "BUILDER_PUBLICATION",
    "CYCLONEDX_SPEC_VERSION",
    "PREDICATE_TYPE",
    "STATEMENT_TYPE",
    "BuildContext",
    "ProvenanceError",
    "build_provenance",
    "build_sbom",
    "lockfile_digest",
    "verify_provenance",
]
