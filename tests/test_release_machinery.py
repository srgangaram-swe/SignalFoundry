"""Tests for the release identity, inventory, provenance, and publication gate.

SF-S5-SL-MR7. A release is a claim about bytes, so the tests here are mostly
about what the machinery **refuses**. The positive path is one assertion; the
value is in the tamper, substitution, and authority cases.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pytest

from quant_platform.release.descriptor import (
    ArtifactSubject,
    CompatibilityStatement,
    ReleaseDescriptor,
    ToolchainRequirement,
)
from quant_platform.release.identity import (
    ReleaseIdentityError,
    assert_contract_pinned,
    assert_versions_agree,
    canonical_version,
    claims_stable_api,
    collect_version_sources,
    is_prerelease,
    parse_version,
)
from quant_platform.release.inventory import (
    InventoryError,
    Subject,
    assert_inventory_matches,
    build_inventory,
    compare_inventories,
    digest_file,
    inventory_from_dicts,
    inventory_to_dicts,
)
from quant_platform.release.policy import (
    PublicationRefused,
    RepositoryState,
    assert_dry_run_publishes_nothing,
    assert_publication_permitted,
    assert_tag_shape,
    tag_for_version,
)
from quant_platform.release.provenance import (
    BUILDER_DRY_RUN,
    BUILDER_PUBLICATION,
    BuildContext,
    ProvenanceError,
    build_provenance,
    build_sbom,
    verify_provenance,
)

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
COMMIT = "a" * 40
OTHER_COMMIT = "b" * 40
BASE = datetime(2026, 8, 1, tzinfo=UTC)


# ---------------------------------------------------------------------------
# Version identity
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("value", ["0.3.0", "1.0.0", "0.3.0rc1", "2.11.4b3"])
def test_supported_versions_parse(value: str) -> None:
    assert parse_version(value) == value


@pytest.mark.parametrize(
    "value", ["0.3", "v0.3.0", "0.3.0.post1", "0.3.0-rc1", "", "latest", None, 3]
)
def test_an_unrecognised_version_is_refused(value: Any) -> None:
    """A version the tooling had to reinterpret is one nobody can reproduce."""
    with pytest.raises(ReleaseIdentityError, match="major.minor.patch"):
        parse_version(value)


def test_the_repository_states_one_version_everywhere() -> None:
    """pyproject, the console, and installed metadata must agree."""
    version = assert_versions_agree(REPOSITORY_ROOT)
    assert version == canonical_version(REPOSITORY_ROOT)
    sources = collect_version_sources(REPOSITORY_ROOT)
    assert len(sources) >= 2
    assert {item.version for item in sources} == {version}


def test_the_module_version_is_not_a_second_source() -> None:
    """__version__ is derived from metadata, so it cannot drift from pyproject."""
    import quant_platform

    assert quant_platform.__version__ == canonical_version(REPOSITORY_ROOT)


def test_a_version_disagreement_is_reported_with_every_offender(tmp_path: Path) -> None:
    (tmp_path / "pyproject.toml").write_text('[project]\nname = "x"\nversion = "0.3.0"\n')
    (tmp_path / "web").mkdir()
    (tmp_path / "web" / "package.json").write_text('{"version": "0.1.0"}')
    with pytest.raises(ReleaseIdentityError, match="console"):
        assert_versions_agree(tmp_path)


def test_the_api_contract_version_is_tracked_separately() -> None:
    """A 0.x release can serve a stable v1 wire contract; both are recorded."""
    assert assert_contract_pinned(REPOSITORY_ROOT, expected_contract="1.0.0") == "1.0.0"
    with pytest.raises(ReleaseIdentityError, match="names the wire format"):
        assert_contract_pinned(REPOSITORY_ROOT, expected_contract="2.0.0")


def test_maturity_is_not_implied_by_release_machinery() -> None:
    """Reaching 1.0 is a compatibility promise, so the check is explicit."""
    assert claims_stable_api("1.0.0") is True
    assert claims_stable_api("0.3.0") is False
    assert is_prerelease("0.3.0rc1") is True
    assert is_prerelease("0.3.0") is False


def test_the_sprint_release_does_not_claim_stability() -> None:
    """Sprint 5 added capability; it makes no compatibility promise."""
    assert claims_stable_api(canonical_version(REPOSITORY_ROOT)) is False


# ---------------------------------------------------------------------------
# Inventory: detection of every way bytes can change
# ---------------------------------------------------------------------------


@pytest.fixture
def staged(tmp_path: Path) -> Path:
    root = tmp_path / "stage"
    (root / "python").mkdir(parents=True)
    (root / "python" / "pkg-0.3.0-py3-none-any.whl").write_bytes(b"wheel bytes\n")
    (root / "python" / "pkg-0.3.0.tar.gz").write_bytes(b"sdist bytes\n")
    (root / "LICENSE").write_bytes(b"MIT\n")
    return root


def test_an_inventory_records_every_file_in_stable_order(staged: Path) -> None:
    subjects = build_inventory(staged)
    assert [item.path for item in subjects] == [
        "LICENSE",
        "python/pkg-0.3.0-py3-none-any.whl",
        "python/pkg-0.3.0.tar.gz",
    ]
    assert all(len(item.sha256) == 64 for item in subjects)


def test_a_missing_artifact_is_detected(staged: Path) -> None:
    expected = build_inventory(staged)
    (staged / "LICENSE").unlink()
    with pytest.raises(InventoryError, match="missing: LICENSE"):
        assert_inventory_matches(expected, staged)


def test_an_extra_artifact_is_detected(staged: Path) -> None:
    """The inventory is closed, not a minimum."""
    expected = build_inventory(staged)
    (staged / "python" / "surprise.whl").write_bytes(b"not in the release\n")
    with pytest.raises(InventoryError, match="unexpected"):
        assert_inventory_matches(expected, staged)


def test_a_renamed_artifact_is_not_matched_by_digest(staged: Path) -> None:
    """A wheel under the wrong filename is not the release the inventory names."""
    expected = build_inventory(staged)
    source = staged / "python" / "pkg-0.3.0-py3-none-any.whl"
    source.rename(staged / "python" / "pkg-0.3.1-py3-none-any.whl")
    difference = compare_inventories(expected, build_inventory(staged))
    assert difference.missing == ("python/pkg-0.3.0-py3-none-any.whl",)
    assert difference.unexpected == ("python/pkg-0.3.1-py3-none-any.whl",)


def test_a_truncated_artifact_is_detected(staged: Path) -> None:
    expected = build_inventory(staged)
    (staged / "python" / "pkg-0.3.0.tar.gz").write_bytes(b"sdist")
    with pytest.raises(InventoryError, match="modified"):
        assert_inventory_matches(expected, staged)


def test_a_substituted_artifact_of_equal_size_is_detected(staged: Path) -> None:
    """Size alone is not identity."""
    expected = build_inventory(staged)
    (staged / "python" / "pkg-0.3.0.tar.gz").write_bytes(b"OTHER bytes\n")
    with pytest.raises(InventoryError, match="modified"):
        assert_inventory_matches(expected, staged)


def test_a_one_byte_modification_is_detected(staged: Path) -> None:
    """The weakest tamper worth naming: a size-only check would miss it."""
    expected = build_inventory(staged)
    target = staged / "LICENSE"
    payload = bytearray(target.read_bytes())
    payload[0] ^= 0x01
    target.write_bytes(bytes(payload))
    with pytest.raises(InventoryError, match="modified: LICENSE"):
        assert_inventory_matches(expected, staged)


def test_a_symlink_in_the_stage_is_refused(staged: Path, tmp_path: Path) -> None:
    """A stage holds regular files so its inventory describes what it ships."""
    outside = tmp_path / "outside.txt"
    outside.write_bytes(b"elsewhere\n")
    (staged / "link.txt").symlink_to(outside)
    with pytest.raises(InventoryError, match="symlink"):
        build_inventory(staged)


def test_an_inventory_round_trips_through_its_record(staged: Path) -> None:
    subjects = build_inventory(staged)
    rebuilt = inventory_from_dicts(json.loads(json.dumps(inventory_to_dicts(subjects))))
    assert rebuilt == subjects


@pytest.mark.parametrize(
    "record",
    [
        {"path": "", "sha256": "a" * 64, "size_bytes": 1},
        {"path": "x", "sha256": "short", "size_bytes": 1},
        {"path": "x", "sha256": "A" * 64, "size_bytes": 1},
        {"path": "x", "sha256": "a" * 64, "size_bytes": -1},
        {"path": "x", "sha256": "a" * 64},
    ],
)
def test_a_malformed_inventory_record_is_refused(record: dict[str, Any]) -> None:
    with pytest.raises(InventoryError):
        inventory_from_dicts([record])


def test_a_duplicate_path_in_an_inventory_is_refused() -> None:
    record = {"path": "x", "sha256": "a" * 64, "size_bytes": 1}
    with pytest.raises(InventoryError, match="duplicate"):
        inventory_from_dicts([record, dict(record)])


def test_an_oversized_artifact_is_refused(staged: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # LICENSE is 4 bytes, so the ceiling has to be below that to be crossed.
    monkeypatch.setattr("quant_platform.release.inventory.MAX_SUBJECT_BYTES", 2)
    with pytest.raises(InventoryError, match="above the subject ceiling"):
        digest_file(staged / "LICENSE")


# ---------------------------------------------------------------------------
# Descriptor
# ---------------------------------------------------------------------------


def _descriptor(**overrides: Any) -> ReleaseDescriptor:
    base: dict[str, Any] = {
        "release_version": "0.3.0",
        "source_commit": COMMIT,
        "built_at": BASE,
        "build_kind": "dry-run",
        "toolchains": (ToolchainRequirement(name="python", version="3.13.0", role="build"),),
        "subjects": (
            ArtifactSubject(path="python/a.whl", sha256="a" * 64, size_bytes=1, kind="wheel"),
            ArtifactSubject(path="python/a.tar.gz", sha256="b" * 64, size_bytes=1, kind="sdist"),
        ),
        "compatibility": CompatibilityStatement(
            api_contract_version="1.0.0",
            registry_schema_version=3,
            supported_python=("3.12",),
            supported_platforms=("linux/amd64",),
            migration_required=True,
            migration_notes="Forward-only to schema 3.",
        ),
        "evidence": ("docs/benchmarks/x.json",),
        "limitations": ("Digests are content identity, not economic validity.",),
    }
    base.update(overrides)
    return ReleaseDescriptor(**base)


def test_a_complete_descriptor_validates_and_round_trips() -> None:
    descriptor = _descriptor()
    rebuilt = ReleaseDescriptor.model_validate_json(json.dumps(descriptor.to_dict()))
    assert rebuilt == descriptor


def test_an_unknown_field_is_refused() -> None:
    """An extra key would be a claim no verifier reads."""
    payload = _descriptor().to_dict()
    payload["extra_claim"] = "trust me"
    with pytest.raises(Exception, match="extra_forbidden|Extra inputs"):
        ReleaseDescriptor.model_validate_json(json.dumps(payload))


@pytest.mark.parametrize("placeholder", ["TODO", "tbd", "  ", "CHANGEME", "none"])
def test_an_unresolved_placeholder_is_refused(placeholder: str) -> None:
    """A template that reached a release is the failure this catches."""
    with pytest.raises(Exception, match="placeholder|major.minor.patch"):
        _descriptor(release_version=placeholder)


def test_an_empty_version_is_refused_by_the_length_guard() -> None:
    """Caught by the field constraint before the placeholder check; either way
    an empty version never reaches a release."""
    with pytest.raises(Exception, match="at least 1 character"):
        _descriptor(release_version="")


def test_a_descriptor_without_a_wheel_or_sdist_is_refused() -> None:
    with pytest.raises(Exception, match="must include a"):
        _descriptor(
            subjects=(
                ArtifactSubject(path="LICENSE", sha256="a" * 64, size_bytes=1, kind="license"),
            )
        )


def test_a_descriptor_with_duplicate_subject_paths_is_refused() -> None:
    subject = ArtifactSubject(path="python/a.whl", sha256="a" * 64, size_bytes=1, kind="wheel")
    sdist = ArtifactSubject(path="python/a.tar.gz", sha256="b" * 64, size_bytes=1, kind="sdist")
    with pytest.raises(Exception, match="duplicate"):
        _descriptor(subjects=(subject, subject, sdist))


def test_empty_evidence_or_limitations_are_refused() -> None:
    """An empty field is a claim nobody made."""
    with pytest.raises(Exception, match="evidence|limitations|state its"):
        _descriptor(limitations=())


def test_a_naive_build_instant_is_refused() -> None:
    with pytest.raises(Exception, match="timezone-aware"):
        _descriptor(built_at=datetime(2026, 8, 1))  # noqa: DTZ001


def test_the_descriptor_states_what_a_signature_does_not_prove() -> None:
    meaning = _descriptor().signature_meaning.lower()
    assert "does not establish" in meaning
    assert "profitability" in meaning


def test_a_stable_release_must_document_its_breaking_changes() -> None:
    """1.0 is a compatibility promise, not a version bump."""
    compatibility = CompatibilityStatement(
        api_contract_version="1.0.0",
        registry_schema_version=3,
        supported_python=("3.12",),
        supported_platforms=("linux/amd64",),
        migration_required=False,
        migration_notes="None.",
        breaking_changes=("dropped the legacy reader",),
    )
    with pytest.raises(Exception, match="compatibility promise"):
        _descriptor(release_version="1.0.0", compatibility=compatibility)


# ---------------------------------------------------------------------------
# Provenance
# ---------------------------------------------------------------------------


def _context(**overrides: Any) -> BuildContext:
    base: dict[str, Any] = {
        "source_commit": COMMIT,
        "build_kind": "dry-run",
        "lockfile_digest": "c" * 64,
        "invocation": ("scripts/release.py", "dry-run"),
        "started_at": BASE,
        "finished_at": BASE,
    }
    base.update(overrides)
    return BuildContext(**base)


@pytest.fixture
def subjects() -> tuple[Subject, ...]:
    return (
        Subject(path="python/a.whl", sha256="a" * 64, size_bytes=10),
        Subject(path="python/a.tar.gz", sha256="b" * 64, size_bytes=20),
    )


def test_provenance_binds_commit_builder_and_every_subject(
    subjects: tuple[Subject, ...],
) -> None:
    statement = build_provenance(subjects, _context(), release_version="0.3.0")
    verify_provenance(
        statement,
        expected_subjects=subjects,
        expected_commit=COMMIT,
        expected_builder=BUILDER_DRY_RUN,
    )


def test_a_dry_run_attestation_cannot_pass_as_a_publication_one(
    subjects: tuple[Subject, ...],
) -> None:
    """The two builder identities are distinct strings for exactly this reason."""
    statement = build_provenance(subjects, _context(), release_version="0.3.0")
    with pytest.raises(ProvenanceError, match="not a publication attestation"):
        verify_provenance(
            statement,
            expected_subjects=subjects,
            expected_commit=COMMIT,
            expected_builder=BUILDER_PUBLICATION,
        )


def test_provenance_from_the_wrong_commit_is_refused(subjects: tuple[Subject, ...]) -> None:
    statement = build_provenance(subjects, _context(), release_version="0.3.0")
    with pytest.raises(ProvenanceError, match="does not bind source commit"):
        verify_provenance(
            statement,
            expected_subjects=subjects,
            expected_commit=OTHER_COMMIT,
            expected_builder=BUILDER_DRY_RUN,
        )


def test_a_tampered_subject_digest_is_refused(subjects: tuple[Subject, ...]) -> None:
    statement = build_provenance(subjects, _context(), release_version="0.3.0")
    tampered = (Subject(path="python/a.whl", sha256="f" * 64, size_bytes=10), subjects[1])
    with pytest.raises(ProvenanceError, match="does not match the built artifact"):
        verify_provenance(
            statement,
            expected_subjects=tampered,
            expected_commit=COMMIT,
            expected_builder=BUILDER_DRY_RUN,
        )


def test_a_missing_attestation_for_a_built_subject_is_refused(
    subjects: tuple[Subject, ...],
) -> None:
    statement = build_provenance(subjects[:1], _context(), release_version="0.3.0")
    with pytest.raises(ProvenanceError, match="does not attest to"):
        verify_provenance(
            statement,
            expected_subjects=subjects,
            expected_commit=COMMIT,
            expected_builder=BUILDER_DRY_RUN,
        )


def test_an_attestation_for_an_unbuilt_subject_is_refused(
    subjects: tuple[Subject, ...],
) -> None:
    """A statement may not carry a subject the build did not produce."""
    statement = build_provenance(subjects, _context(), release_version="0.3.0")
    with pytest.raises(ProvenanceError, match="were not built"):
        verify_provenance(
            statement,
            expected_subjects=subjects[:1],
            expected_commit=COMMIT,
            expected_builder=BUILDER_DRY_RUN,
        )


@pytest.mark.parametrize(
    "payload",
    [
        {"_type": "wrong", "predicateType": "x"},
        {"_type": "https://in-toto.io/Statement/v1", "predicateType": "wrong"},
        {
            "_type": "https://in-toto.io/Statement/v1",
            "predicateType": "https://slsa.dev/provenance/v1",
        },
    ],
)
def test_a_malformed_statement_is_refused(
    payload: dict[str, Any], subjects: tuple[Subject, ...]
) -> None:
    with pytest.raises(ProvenanceError):
        verify_provenance(
            payload,
            expected_subjects=subjects,
            expected_commit=COMMIT,
            expected_builder=BUILDER_DRY_RUN,
        )


@pytest.mark.parametrize(
    "overrides",
    [
        {"source_commit": "short"},
        {"build_kind": "publish"},
        {"lockfile_digest": "short"},
        {"invocation": ()},
        {"finished_at": BASE - timedelta(seconds=1)},
    ],
)
def test_an_unusable_build_context_is_refused(overrides: dict[str, Any]) -> None:
    with pytest.raises(ProvenanceError):
        _context(**overrides)


def test_provenance_requires_at_least_one_subject() -> None:
    with pytest.raises(ProvenanceError, match="at least one subject"):
        build_provenance((), _context(), release_version="0.3.0")


# ---------------------------------------------------------------------------
# SBOM
# ---------------------------------------------------------------------------


def test_the_sbom_covers_python_and_node_and_is_deterministic() -> None:
    first = build_sbom(REPOSITORY_ROOT, release_version="0.3.0", source_commit=COMMIT)
    second = build_sbom(REPOSITORY_ROOT, release_version="0.3.0", source_commit=COMMIT)
    assert first == second, "two SBOMs of one release must be byte-identical"
    assert first["bomFormat"] == "CycloneDX"
    assert first["specVersion"] == "1.6"
    purls = {component["purl"].split("/")[0] for component in first["components"]}
    assert "pkg:pypi" in purls
    assert "pkg:npm" in purls, "the console's dependencies belong in the SBOM"


def test_the_sbom_serial_changes_with_the_release_identity() -> None:
    """Two different releases must not share a serial number."""
    first = build_sbom(REPOSITORY_ROOT, release_version="0.3.0", source_commit=COMMIT)
    second = build_sbom(REPOSITORY_ROOT, release_version="0.4.0", source_commit=COMMIT)
    assert first["serialNumber"] != second["serialNumber"]


def test_an_unreadable_lockfile_is_refused(tmp_path: Path) -> None:
    with pytest.raises(ProvenanceError, match="cannot read"):
        build_sbom(tmp_path, release_version="0.3.0", source_commit=COMMIT)


# ---------------------------------------------------------------------------
# The publication gate
# ---------------------------------------------------------------------------


def _state(**overrides: Any) -> RepositoryState:
    base: dict[str, Any] = {
        "branch": "main",
        "head_commit": COMMIT,
        "remote_main_commit": COMMIT,
        "is_clean": True,
        "existing_tags": frozenset(),
        "main_ancestry": frozenset({COMMIT, OTHER_COMMIT}),
    }
    base.update(overrides)
    return RepositoryState(**base)


def test_a_correct_publication_is_permitted() -> None:
    descriptor = _descriptor(build_kind="publication")
    tag = assert_publication_permitted(descriptor, _state(), requested_commit=COMMIT)
    assert tag == "v0.3.0"


def test_a_dry_run_build_can_never_be_published() -> None:
    with pytest.raises(PublicationRefused, match="dry-run"):
        assert_publication_permitted(_descriptor(), _state(), requested_commit=COMMIT)


@pytest.mark.parametrize("branch", ["dev", "prod", "feat/x", "HEAD"])
def test_publication_from_any_branch_but_main_is_refused(branch: str) -> None:
    descriptor = _descriptor(build_kind="publication")
    with pytest.raises(PublicationRefused, match="only permitted from main"):
        assert_publication_permitted(descriptor, _state(branch=branch), requested_commit=COMMIT)


def test_a_dirty_tree_is_refused() -> None:
    descriptor = _descriptor(build_kind="publication")
    with pytest.raises(PublicationRefused, match="dirty"):
        assert_publication_permitted(descriptor, _state(is_clean=False), requested_commit=COMMIT)


def test_a_stale_commit_is_refused() -> None:
    """Tagging a commit main no longer points at would tag the wrong bytes."""
    descriptor = _descriptor(build_kind="publication")
    with pytest.raises(PublicationRefused, match="current origin/main"):
        assert_publication_permitted(
            descriptor,
            _state(head_commit=COMMIT, remote_main_commit=OTHER_COMMIT),
            requested_commit=COMMIT,
        )


def test_a_descriptor_built_from_a_different_commit_is_refused() -> None:
    descriptor = _descriptor(build_kind="publication", source_commit=OTHER_COMMIT)
    with pytest.raises(PublicationRefused, match="descriptor was built from"):
        assert_publication_permitted(descriptor, _state(), requested_commit=COMMIT)


def test_a_reused_tag_is_refused() -> None:
    """Published tags are immutable; a defect is corrected with a new version."""
    descriptor = _descriptor(build_kind="publication")
    with pytest.raises(PublicationRefused, match="already exists"):
        assert_publication_permitted(
            descriptor,
            _state(existing_tags=frozenset({"v0.3.0"})),
            requested_commit=COMMIT,
        )


def test_a_missing_promotion_ancestry_is_refused() -> None:
    descriptor = _descriptor(build_kind="publication")
    with pytest.raises(PublicationRefused, match="not reachable from main"):
        assert_publication_permitted(
            descriptor,
            _state(),
            requested_commit=COMMIT,
            promotion_ancestry=("d" * 40,),
        )


@pytest.mark.parametrize("tag", ["0.3.0", "release-0.3.0", "v0.3", "vlatest", "v0.3.0-hotfix"])
def test_a_free_form_tag_is_refused(tag: str) -> None:
    with pytest.raises(PublicationRefused, match="derived from the version"):
        assert_tag_shape(tag)


def test_the_tag_is_derived_from_the_version() -> None:
    assert tag_for_version("0.3.0") == "v0.3.0"
    assert tag_for_version("1.0.0rc1") == "v1.0.0rc1"


def test_a_dry_run_that_created_public_state_is_refused() -> None:
    assert_dry_run_publishes_nothing(created_tags=(), created_releases=())
    with pytest.raises(PublicationRefused, match="created public state"):
        assert_dry_run_publishes_nothing(created_tags=("v0.3.0",), created_releases=())


def test_the_release_package_exposes_no_publish_function() -> None:
    """Deciding whether publication is permitted is not the same as doing it."""
    import quant_platform.release as release_package
    from quant_platform.release import descriptor, identity, inventory, policy, provenance

    forbidden = {"publish", "create_tag", "upload", "push_release", "create_release"}
    for module in (release_package, descriptor, identity, inventory, policy, provenance):
        assert not forbidden & set(dir(module)), module.__name__
