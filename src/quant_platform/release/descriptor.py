"""The strict release descriptor.

SF-S5-SL-MR7. One record binds everything a release claims: its version, the
exact source commit, the toolchains it requires, the schema versions it serves,
the complete artifact inventory, its compatibility status, and the evidence that
supports it.

The descriptor is the thing verification checks *against*. That makes two
properties load-bearing:

* **Unknown fields are refused.** A descriptor that silently accepted an extra
  key would let a producer add a claim no verifier reads.
* **Unresolved placeholders are refused.** A version, digest, or commit that is
  still ``TODO``/``TBD``/empty is the single most likely way a template reaches
  a real release, so those are rejected explicitly rather than caught by
  whatever downstream check happens to notice.

The descriptor states what a signature does and does not establish, because the
temptation to read "signed" as "validated" is exactly what this record must not
encourage.
"""

from __future__ import annotations

import re
from datetime import UTC, datetime
from typing import Annotated, Any, Final, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from quant_platform.release.identity import claims_stable_api, parse_version
from quant_platform.release.inventory import Subject

#: Descriptor schema version. Independent of the release version: this is the
#: shape of the record, not the software it describes.
DESCRIPTOR_SCHEMA_VERSION: Final = 1

#: Tokens that mean "this was never filled in".
_PLACEHOLDERS: Final = frozenset({"", "todo", "tbd", "fixme", "xxx", "changeme", "none", "null"})

_COMMIT_PATTERN: Final = re.compile(r"^[0-9a-f]{40}$")

Digest = Annotated[str, Field(min_length=64, max_length=64, pattern=r"^[0-9a-f]{64}$")]
Commit = Annotated[str, Field(min_length=40, max_length=40, pattern=r"^[0-9a-f]{40}$")]


class DescriptorError(ValueError):
    """Raised when a release descriptor is unusable."""


def _reject_placeholder(value: str, field_name: str) -> str:
    """Return the value, refusing an unresolved template token.

    Raises:
        DescriptorError: If the value looks like a placeholder.
    """
    if value.strip().lower() in _PLACEHOLDERS:
        raise DescriptorError(
            f"{field_name} is still a placeholder ({value!r}); a descriptor with an "
            "unfilled field must not reach a release"
        )
    return value


class StrictRecord(BaseModel):
    """Immutable base that refuses coercion, unknown fields, and non-finite JSON."""

    model_config = ConfigDict(
        allow_inf_nan=False,
        extra="forbid",
        frozen=True,
        strict=True,
        validate_default=True,
    )


class ToolchainRequirement(StrictRecord):
    """One toolchain the release was built with and requires."""

    name: Annotated[str, Field(min_length=1, max_length=64)]
    version: Annotated[str, Field(min_length=1, max_length=64)]
    role: Literal["build", "runtime", "test"]

    @field_validator("name", "version")
    @classmethod
    def _no_placeholder(cls, value: str) -> str:
        return _reject_placeholder(value, "toolchain field")


class ArtifactSubject(StrictRecord):
    """One release artifact, bound by path, digest, and size."""

    path: Annotated[str, Field(min_length=1, max_length=512)]
    sha256: Digest
    size_bytes: Annotated[int, Field(ge=0, le=512 * 1024 * 1024)]
    kind: Literal[
        "wheel",
        "sdist",
        "console",
        "schema",
        "sbom",
        "provenance",
        "evidence",
        "notes",
        "license",
        "oci",
    ]

    @classmethod
    def from_subject(cls, subject: Subject, *, kind: str) -> ArtifactSubject:
        """Build a descriptor subject from an inventory subject."""
        return cls(
            path=subject.path,
            sha256=subject.sha256,
            size_bytes=subject.size_bytes,
            kind=kind,  # type: ignore[arg-type]
        )


class CompatibilityStatement(StrictRecord):
    """What the release promises about interfaces and upgrades."""

    api_contract_version: Annotated[str, Field(min_length=1, max_length=32)]
    registry_schema_version: Annotated[int, Field(ge=1, le=1000)]
    supported_python: tuple[Annotated[str, Field(min_length=1, max_length=16)], ...]
    supported_platforms: tuple[Annotated[str, Field(min_length=1, max_length=64)], ...]
    migration_required: bool
    migration_notes: Annotated[str, Field(min_length=1, max_length=2048)]
    breaking_changes: tuple[Annotated[str, Field(min_length=1, max_length=512)], ...] = ()

    @field_validator("supported_python", "supported_platforms")
    @classmethod
    def _non_empty(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        if not value:
            raise DescriptorError("a release must state which targets it supports")
        return value


class ReleaseDescriptor(StrictRecord):
    """The complete, strict statement of one release.

    ``signature_meaning`` is a required field rather than documentation because
    the descriptor travels with the release, and the claim most likely to be
    over-read is what a signature proves.
    """

    schema_version: Literal[1] = DESCRIPTOR_SCHEMA_VERSION
    release_version: Annotated[str, Field(min_length=1, max_length=32)]
    source_commit: Commit
    built_at: datetime
    #: "dry-run" and "publication" are separate identities on purpose: a dry-run
    #: artifact must never be mistakable for a published one.
    build_kind: Literal["dry-run", "publication"]
    toolchains: tuple[ToolchainRequirement, ...]
    subjects: tuple[ArtifactSubject, ...]
    compatibility: CompatibilityStatement
    evidence: tuple[Annotated[str, Field(min_length=1, max_length=512)], ...]
    limitations: tuple[Annotated[str, Field(min_length=1, max_length=512)], ...]
    signature_meaning: Annotated[str, Field(min_length=32, max_length=1024)] = (
        "A signature establishes that the authorized release process signed these exact bytes "
        "under the documented trust policy. It does not establish independent research review, "
        "economic validity, production readiness, or profitability."
    )

    @field_validator("release_version")
    @classmethod
    def _valid_version(cls, value: str) -> str:
        return parse_version(_reject_placeholder(value, "release_version"))

    @field_validator("source_commit")
    @classmethod
    def _valid_commit(cls, value: str) -> str:
        if _COMMIT_PATTERN.match(value) is None:
            raise DescriptorError("source_commit must be a full 40-character hex commit")
        return value

    @field_validator("built_at")
    @classmethod
    def _aware_instant(cls, value: datetime) -> datetime:
        if value.tzinfo is None:
            raise DescriptorError("built_at must be timezone-aware")
        return value.astimezone(UTC)

    @field_validator("toolchains", "subjects", "evidence", "limitations")
    @classmethod
    def _non_empty_tuple(cls, value: tuple[Any, ...]) -> tuple[Any, ...]:
        if not value:
            raise DescriptorError(
                "a descriptor must state its toolchains, subjects, evidence, and limitations; "
                "an empty field is a claim nobody made"
            )
        return value

    @field_validator("subjects")
    @classmethod
    def _unique_paths(cls, value: tuple[ArtifactSubject, ...]) -> tuple[ArtifactSubject, ...]:
        paths = [item.path for item in value]
        if len(set(paths)) != len(paths):
            raise DescriptorError("descriptor lists a duplicate subject path")
        return value

    def model_post_init(self, _context: object, /) -> None:
        """Refuse a descriptor whose own fields contradict each other."""
        kinds = {item.kind for item in self.subjects}
        for required in ("wheel", "sdist"):
            if required not in kinds:
                raise DescriptorError(f"a release must include a {required} subject")
        # 1.0 is a compatibility promise. Shipping breaking changes under it
        # without documenting each one is the failure this catches.
        if (
            claims_stable_api(self.release_version)
            and self.compatibility.breaking_changes
            and not all(
                "documented" in note.lower() for note in self.compatibility.breaking_changes
            )
        ):
            raise DescriptorError(
                "a 1.x release listing breaking changes must document each one explicitly; "
                "reaching 1.0 is a compatibility promise, not a version bump"
            )

    def subject_for(self, path: str) -> ArtifactSubject | None:
        """Return the subject at ``path``, or ``None``."""
        for item in self.subjects:
            if item.path == path:
                return item
        return None

    def to_dict(self) -> dict[str, Any]:
        """Return the canonical JSON-friendly record."""
        return self.model_dump(mode="json")


__all__ = [
    "DESCRIPTOR_SCHEMA_VERSION",
    "ArtifactSubject",
    "CompatibilityStatement",
    "DescriptorError",
    "ReleaseDescriptor",
    "StrictRecord",
    "ToolchainRequirement",
]
