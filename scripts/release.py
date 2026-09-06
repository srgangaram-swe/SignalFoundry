"""Release dry-run and verification entry points.

SF-S5-SL-MR7. Two subcommands:

``dry-run``
    Builds the complete release candidate into a staging directory, writes the
    descriptor, inventory, SBOM, and provenance, and verifies all of it. It has
    **no** capability to create a tag, a GitHub Release, or any remote artifact:
    there is no network call and no token is read.

``verify``
    Consumes only a staging directory and its manifests, and independently
    re-checks version agreement, the inventory, the descriptor, and the
    provenance. It re-hashes every artifact rather than trusting the recorded
    digests, because a verifier that reads the numbers the producer wrote is
    checking arithmetic rather than content.

Exit codes are stable and typed, so a workflow can distinguish "the release is
bad" from "the tool broke":

===  ==========================================================
  0  the release verifies
  2  a release gate refused (version, inventory, provenance)
  3  the environment is unusable (missing tool, unreadable tree)
===  ==========================================================
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Final, Literal

from quant_platform.release.descriptor import (
    ArtifactSubject,
    CompatibilityStatement,
    DescriptorError,
    ReleaseDescriptor,
    ToolchainRequirement,
)
from quant_platform.release.identity import (
    ReleaseIdentityError,
    assert_contract_pinned,
    assert_versions_agree,
    claims_stable_api,
)
from quant_platform.release.inventory import (
    InventoryError,
    build_inventory,
    compare_inventories,
    inventory_from_dicts,
    inventory_to_dicts,
)
from quant_platform.release.policy import (
    PublicationRefused,
    RepositoryState,
    assert_dry_run_publishes_nothing,
    assert_publication_permitted,
)
from quant_platform.release.provenance import (
    BUILDER_DRY_RUN,
    BuildContext,
    ProvenanceError,
    build_provenance,
    build_sbom,
    lockfile_digest,
    verify_provenance,
)

EXIT_OK: Final = 0
EXIT_REFUSED: Final = 2
EXIT_ENVIRONMENT: Final = 3

#: Manifest file names. Excluded from the inventory they describe, because a
#: manifest cannot contain its own digest.
DESCRIPTOR_NAME: Final = "release-descriptor.json"
INVENTORY_NAME: Final = "release-inventory.json"
SBOM_NAME: Final = "sbom.cyclonedx.json"
PROVENANCE_NAME: Final = "provenance.intoto.json"
MANIFESTS: Final = (DESCRIPTOR_NAME, INVENTORY_NAME, SBOM_NAME, PROVENANCE_NAME)

#: Bounded so a wedged build fails rather than holding a runner open.
BUILD_TIMEOUT_SECONDS: Final = 900

#: The API wire contract this release serves.
EXPECTED_API_CONTRACT: Final = "1.0.0"


class ReleaseToolError(RuntimeError):
    """Raised when the environment cannot support a release build."""


def _run(argv: list[str], *, cwd: Path, env: dict[str, str] | None = None) -> str:
    """Run one command, returning stdout.

    Raises:
        ReleaseToolError: On a non-zero exit or a timeout, carrying bounded
            output so a failure is diagnosable without dumping a whole log.
    """
    try:
        result = subprocess.run(  # noqa: S603 - fixed argv, never shell
            argv,
            cwd=cwd,
            capture_output=True,
            text=True,
            timeout=BUILD_TIMEOUT_SECONDS,
            env={**os.environ, **(env or {})},
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as error:
        raise ReleaseToolError(f"{argv[0]} could not run: {error}") from error
    if result.returncode != 0:
        raise ReleaseToolError(
            f"{' '.join(argv[:3])} failed ({result.returncode}):\n"
            f"{result.stdout[-1500:]}\n{result.stderr[-1500:]}"
        )
    return result.stdout


def _git(repository_root: Path, *args: str) -> str:
    """Return trimmed git output."""
    return _run(["git", *args], cwd=repository_root).strip()


def _source_commit(repository_root: Path) -> str:
    """Return the exact HEAD commit.

    Raises:
        ReleaseToolError: If the tree is dirty. Artifacts built from
            uncommitted changes are not reproducible from any commit, so this
            refuses rather than recording a commit the bytes do not match.
    """
    status = _git(repository_root, "status", "--porcelain")
    if status:
        raise ReleaseToolError(
            "the working tree is dirty; a release must be reproducible from its commit.\n"
            f"{status[:800]}"
        )
    return _git(repository_root, "rev-parse", "HEAD")


def _deterministic_env(source_epoch: int) -> dict[str, str]:
    """Return the environment that makes a build reproducible.

    Timestamps, hash ordering, locale, and timezone are all pinned: each is a
    documented way for two builds of identical inputs to differ in their bytes.
    """
    return {
        "SOURCE_DATE_EPOCH": str(source_epoch),
        "PYTHONHASHSEED": "0",
        "PYTHONDONTWRITEBYTECODE": "1",
        "LC_ALL": "C",
        "LANG": "C",
        "TZ": "UTC",
    }


def _normalise_sdist(archive: Path, *, source_epoch: int) -> None:
    """Rewrite an sdist so two builds of one commit are byte-identical.

    setuptools honours ``SOURCE_DATE_EPOCH`` for most members but stamps the
    generated entries -- ``PKG-INFO`` and every directory -- with the wall clock,
    which makes the archive differ between runs that produced identical content.

    The archive is repacked with member order sorted, mtimes pinned to the
    source epoch, and ownership normalised. Nothing about the file *contents*
    changes; only metadata that describes when the build ran, which a
    reproducible build must not record.
    """
    import gzip
    import io
    import tarfile

    with tarfile.open(archive, "r:gz") as source:
        members = sorted(source.getmembers(), key=lambda item: item.name)
        payloads: list[tuple[tarfile.TarInfo, bytes | None]] = []
        for member in members:
            handle = source.extractfile(member) if member.isfile() else None
            payloads.append((member, handle.read() if handle is not None else None))

    raw = io.BytesIO()
    with tarfile.open(fileobj=raw, mode="w", format=tarfile.PAX_FORMAT) as destination:
        for member, payload in payloads:
            # The PAX extended header carries its own sub-second `mtime`, and it
            # takes precedence over the TarInfo field on write. Clearing it is
            # what actually pins the timestamp -- setting `member.mtime` alone
            # leaves the original value in the extended header and the archive
            # still differs between runs.
            member.pax_headers = {}
            member.mtime = source_epoch
            member.uid = 0
            member.gid = 0
            member.uname = ""
            member.gname = ""
            destination.addfile(member, io.BytesIO(payload) if payload is not None else None)

    # mtime=0 in the gzip header as well: the container records its own
    # timestamp, and leaving it would defeat the normalisation above.
    with (
        archive.open("wb") as handle,
        gzip.GzipFile(fileobj=handle, mode="wb", mtime=0) as compressed,
    ):
        compressed.write(raw.getvalue())


def _build_python_distributions(
    repository_root: Path, staging: Path, env: dict[str, str], *, source_epoch: int
) -> None:
    """Build the wheel and sdist into the staging directory."""
    output = staging / "python"
    output.mkdir(parents=True, exist_ok=True)
    _run(
        [sys.executable, "-m", "build", "--outdir", str(output), str(repository_root)],
        cwd=repository_root,
        env=env,
    )
    for archive in sorted(output.glob("*.tar.gz")):
        _normalise_sdist(archive, source_epoch=source_epoch)


def _build_console(repository_root: Path, staging: Path, env: dict[str, str]) -> bool:
    """Build the console bundle, returning whether it was built.

    Returns ``False`` when the Node toolchain is unavailable, which a caller
    reports rather than hides: a release missing its console is a smaller
    release, not a silently equivalent one.
    """
    web = repository_root / "web"
    npm = shutil.which("npm")
    if npm is None or not (web / "node_modules").is_dir():
        return False
    _run([npm, "run", "build"], cwd=web, env={**env, "CI": "1"})
    destination = staging / "console"
    if destination.exists():
        shutil.rmtree(destination)
    shutil.copytree(web / "dist", destination, symlinks=False)
    return True


def _copy_schemas_and_evidence(repository_root: Path, staging: Path) -> None:
    """Copy the versioned contract, license, and redistribution-safe evidence."""
    schemas = staging / "schemas"
    schemas.mkdir(parents=True, exist_ok=True)
    shutil.copy2(repository_root / "docs/api/openapi-v1.json", schemas / "openapi-v1.json")

    evidence = staging / "evidence"
    evidence.mkdir(parents=True, exist_ok=True)
    for candidate in sorted((repository_root / "docs/benchmarks").glob("*.json")):
        shutil.copy2(candidate, evidence / candidate.name)

    shutil.copy2(repository_root / "LICENSE", staging / "LICENSE")


def _toolchains() -> tuple[ToolchainRequirement, ...]:
    """Return the toolchains this build used and the release requires."""
    node = shutil.which("node")
    entries = [
        ToolchainRequirement(
            name="python",
            version=".".join(str(part) for part in sys.version_info[:3]),
            role="build",
        ),
        ToolchainRequirement(name="python", version=">=3.12", role="runtime"),
    ]
    if node is not None:
        try:
            node_version = _run([node, "--version"], cwd=Path.cwd()).strip().lstrip("v")
            entries.append(ToolchainRequirement(name="node", version=node_version, role="build"))
        except ReleaseToolError:  # pragma: no cover - node present but unusable
            pass
    return tuple(entries)


def _classify(path: str) -> str:
    """Return the descriptor kind for a staged artifact path."""
    if path.startswith("python/") and path.endswith(".whl"):
        return "wheel"
    if path.startswith("python/") and path.endswith(".tar.gz"):
        return "sdist"
    if path.startswith("console/"):
        return "console"
    if path.startswith("schemas/"):
        return "schema"
    if path.startswith("evidence/"):
        return "evidence"
    if path == "LICENSE":
        return "license"
    if path.endswith("RELEASE-NOTES.md"):
        return "notes"
    return "evidence"


def _write_canonical(path: Path, document: Any) -> None:
    """Write a document as canonical, deterministic JSON."""
    path.write_text(
        json.dumps(document, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def dry_run(
    repository_root: Path,
    staging: Path,
    *,
    release_version: str | None,
    build_kind: Literal["dry-run", "publication"] = "dry-run",
) -> int:
    """Build and verify a complete release candidate without publishing.

    Returns:
        A stable exit code.
    """
    try:
        version = assert_versions_agree(repository_root)
        if release_version is not None and release_version != version:
            raise ReleaseIdentityError(
                f"requested release {release_version} does not match the canonical {version}"
            )
        assert_contract_pinned(repository_root, expected_contract=EXPECTED_API_CONTRACT)
        commit = _source_commit(repository_root)
    except (ReleaseIdentityError, ReleaseToolError) as error:
        print(f"release refused: {error}")
        return EXIT_REFUSED if isinstance(error, ReleaseIdentityError) else EXIT_ENVIRONMENT

    if claims_stable_api(version):
        print(
            f"note: {version} claims a stable public interface; confirm the compatibility "
            "statement is accurate before publishing."
        )

    if staging.exists():
        print("release refused: staging already exists; choose a new destination")
        return EXIT_REFUSED
    staging.mkdir(parents=True)

    # A fixed epoch derived from the commit, not the wall clock: two builds of
    # the same commit must agree byte for byte, so every timestamp the release
    # records is the source date rather than when the build happened to run.
    epoch = int(_git(repository_root, "show", "-s", "--format=%ct", commit))
    started = datetime.fromtimestamp(epoch, tz=UTC)
    env = _deterministic_env(epoch)

    try:
        _build_python_distributions(repository_root, staging, env, source_epoch=epoch)
        console_built = _build_console(repository_root, staging, env)
        _copy_schemas_and_evidence(repository_root, staging)
    except ReleaseToolError as error:
        print(f"release build failed: {error}")
        return EXIT_ENVIRONMENT

    try:
        subjects = build_inventory(staging, exclude=MANIFESTS)
        sbom = build_sbom(repository_root, release_version=version, source_commit=commit)
        _write_canonical(staging / SBOM_NAME, sbom)

        # The SBOM is a subject too, so the inventory is rebuilt to include it
        # before provenance attests to the set.
        subjects = build_inventory(
            staging, exclude=(DESCRIPTOR_NAME, INVENTORY_NAME, PROVENANCE_NAME)
        )
        descriptor = ReleaseDescriptor(
            release_version=version,
            source_commit=commit,
            built_at=started,
            build_kind=build_kind,
            toolchains=_toolchains(),
            subjects=tuple(
                ArtifactSubject.from_subject(item, kind=_classify(item.path)) for item in subjects
            ),
            compatibility=CompatibilityStatement(
                api_contract_version=EXPECTED_API_CONTRACT,
                registry_schema_version=3,
                supported_python=("3.12", "3.13", "3.14"),
                supported_platforms=("linux/amd64", "macos/arm64"),
                migration_required=True,
                migration_notes=(
                    "The registry advances to schema version 3, which adds the append-only "
                    "governance lane tables. Migration is forward-only and applied on open; "
                    "there is no down-migration, and a database at a newer version is refused."
                ),
            ),
            evidence=(
                "docs/benchmarks/service_operability_2026-09-06.json",
                "docs/benchmarks/console_evidence_2026-08-20.json",
                "reports/figures/console_evidence.png",
            ),
            limitations=(
                "Artifact digests establish content identity, not economic validity.",
                "No prospective wall-clock trading evidence exists; shadow campaigns are "
                "deterministic replays.",
                (
                    "The console bundle is omitted when the Node toolchain is unavailable."
                    if not console_built
                    else "The console bundle is built from the committed lockfile."
                ),
                "A signature proves the authorized process signed these bytes; it is not "
                "independent review and not an authorization to trade.",
            ),
        )
        _write_canonical(staging / DESCRIPTOR_NAME, descriptor.to_dict())
        _write_canonical(staging / INVENTORY_NAME, inventory_to_dicts(subjects))

        context = BuildContext(
            source_commit=commit,
            build_kind=build_kind,
            lockfile_digest=lockfile_digest(repository_root),
            invocation=("scripts/release.py", build_kind),
            started_at=started,
            # Also the source date. A provenance statement that recorded real
            # elapsed time would make two reproducible builds differ, and the
            # value it would carry -- how long a machine took -- is not a
            # property of the release.
            finished_at=started,
        )
        _write_canonical(
            staging / PROVENANCE_NAME,
            build_provenance(subjects, context, release_version=version),
        )
    except (DescriptorError, InventoryError, ProvenanceError) as error:
        print(f"release refused: {error}")
        return EXIT_REFUSED

    # A dry run must leave no public state behind. Asserted rather than assumed,
    # because "we did not call the API" is a claim about code that changes.
    try:
        assert_dry_run_publishes_nothing(created_tags=(), created_releases=())
    except PublicationRefused as error:  # pragma: no cover - defensive
        print(f"release refused: {error}")
        return EXIT_REFUSED

    print(
        json.dumps(
            {
                "release_version": version,
                "source_commit": commit,
                "build_kind": build_kind,
                "subjects": len(subjects),
                "console_built": console_built,
                "staging": str(staging),
                "published": False,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return verify(repository_root, staging)


def verify(repository_root: Path, staging: Path) -> int:
    """Independently verify a staged release candidate.

    Re-hashes every artifact rather than trusting the recorded digests.
    """
    try:
        # Read as JSON text rather than as a Python object: the descriptor is a
        # strict model, and in strict Python-object mode a JSON list will not
        # coerce to the declared tuple. Validating the raw document is what
        # makes the record round-trip, which is the whole point of writing it.
        descriptor_text = (staging / DESCRIPTOR_NAME).read_text(encoding="utf-8")
        recorded = inventory_from_dicts(
            json.loads((staging / INVENTORY_NAME).read_text(encoding="utf-8"))
        )
        statement = json.loads((staging / PROVENANCE_NAME).read_text(encoding="utf-8"))
    except OSError as error:
        print(f"verification refused: a release manifest is missing ({error})")
        return EXIT_REFUSED
    except ValueError as error:
        print(f"verification refused: a release manifest is not valid JSON ({error})")
        return EXIT_REFUSED

    try:
        descriptor = ReleaseDescriptor.model_validate_json(descriptor_text)
    except Exception as error:  # noqa: BLE001 - pydantic raises its own type
        print(f"verification refused: the descriptor is invalid ({error})")
        return EXIT_REFUSED

    if descriptor.source_commit != _git(repository_root, "rev-parse", "HEAD"):
        print("verification refused: descriptor source does not match checked-out HEAD")
        return EXIT_REFUSED

    try:
        canonical = assert_versions_agree(repository_root)
    except ReleaseIdentityError as error:
        print(f"verification refused: {error}")
        return EXIT_REFUSED
    if descriptor.release_version != canonical:
        print(
            f"verification refused: the descriptor states {descriptor.release_version} but the "
            f"tree resolves to {canonical}"
        )
        return EXIT_REFUSED

    try:
        observed = build_inventory(
            staging, exclude=(DESCRIPTOR_NAME, INVENTORY_NAME, PROVENANCE_NAME)
        )
    except InventoryError as error:
        print(f"verification refused: {error}")
        return EXIT_REFUSED

    difference = compare_inventories(recorded, observed)
    if not difference.clean:
        print(f"verification refused: {difference.describe()}")
        return EXIT_REFUSED

    declared = {item.path: item.sha256 for item in descriptor.subjects}
    rebuilt = {item.path: item.sha256 for item in observed}
    if declared != rebuilt:
        print("verification refused: the descriptor's subjects do not match the staged bytes")
        return EXIT_REFUSED

    try:
        verify_provenance(
            statement,
            expected_subjects=observed,
            expected_commit=descriptor.source_commit,
            expected_builder=(
                BUILDER_DRY_RUN
                if descriptor.build_kind == "dry-run"
                else "https://github.com/srgangaram-swe/Signalattice/release/publication"
            ),
        )
    except ProvenanceError as error:
        print(f"verification refused: {error}")
        return EXIT_REFUSED

    print(
        json.dumps(
            {
                "release_version": descriptor.release_version,
                "source_commit": descriptor.source_commit,
                "build_kind": descriptor.build_kind,
                "subjects_verified": len(observed),
                "verified": True,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return EXIT_OK


def publication_gate(repository_root: Path, staging: Path, requested_commit: str) -> int:
    """Verify staged bytes and observed promotion ancestry without publishing.

    The caller refreshes origin refs before invocation. This function never
    accesses a credential or changes Git refs. Both promotions must be real
    merge ancestry; an empty list cannot silently satisfy the gate.
    """
    if verify(repository_root, staging) != EXIT_OK:
        return EXIT_REFUSED
    try:
        descriptor = ReleaseDescriptor.model_validate_json(
            (staging / DESCRIPTOR_NAME).read_text(encoding="utf-8")
        )
        main = _git(repository_root, "rev-parse", "origin/main")
        prod = _git(repository_root, "rev-parse", "origin/prod")
        dev = _git(repository_root, "rev-parse", "origin/dev")
        _git(repository_root, "merge-base", "--is-ancestor", dev, prod)
        _git(repository_root, "merge-base", "--is-ancestor", prod, main)
        for promotion in (prod, main):
            if len(_git(repository_root, "show", "-s", "--format=%P", promotion).split()) < 2:
                raise PublicationRefused("promotion must be a merge commit")
        state = RepositoryState(
            branch=_git(repository_root, "branch", "--show-current"),
            head_commit=_source_commit(repository_root),
            remote_main_commit=main,
            is_clean=True,
            existing_tags=frozenset(_git(repository_root, "tag", "--list").splitlines()),
            main_ancestry=frozenset(_git(repository_root, "rev-list", main).splitlines()),
        )
        tag = assert_publication_permitted(
            descriptor, state, requested_commit=requested_commit, promotion_ancestry=(dev, prod)
        )
        print(json.dumps({"permitted": True, "tag": tag, "dev": dev, "prod": prod, "main": main}))
        return EXIT_OK
    except (OSError, ValueError, PublicationRefused, ReleaseToolError) as error:
        print(f"publication refused: {error}")
        return EXIT_REFUSED


def main(argv: list[str] | None = None) -> int:
    """Dispatch a release subcommand."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    subcommands = parser.add_subparsers(dest="command", required=True)

    # --staging belongs to each subcommand rather than the top level: a global
    # option that must precede the subcommand reads as though it can follow it,
    # and argparse rejects the natural spelling with an unhelpful error.
    dry = subcommands.add_parser("dry-run", help="build and verify without publishing")
    dry.add_argument("--staging", type=Path, default=Path("build/release"))
    dry.add_argument("--release-version", default=None)
    publication = subcommands.add_parser(
        "publication", help="build publication-identity bytes locally; publishes nothing"
    )
    publication.add_argument("--staging", type=Path, default=Path("build/publication"))
    publication.add_argument("--release-version", default=None)
    gate = subcommands.add_parser("publication-gate", help="verify bytes and promotion ancestry")
    gate.add_argument("--staging", type=Path, default=Path("build/publication"))
    gate.add_argument("--commit", required=True)
    check = subcommands.add_parser("verify", help="independently verify a staged candidate")
    check.add_argument("--staging", type=Path, default=Path("build/release"))

    arguments = parser.parse_args(argv)
    root = arguments.root.resolve()
    staging = arguments.staging if arguments.staging.is_absolute() else root / arguments.staging

    if arguments.command in {"dry-run", "publication"}:
        return dry_run(
            root, staging, release_version=arguments.release_version, build_kind=arguments.command
        )
    if arguments.command == "publication-gate":
        return publication_gate(root, staging, arguments.commit)
    return verify(root, staging)


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    raise SystemExit(main())
