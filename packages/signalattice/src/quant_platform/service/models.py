"""Versioned HTTP projections for the local read-only evidence service.

These models are deliberately distinct from registry persistence records.  The
projection omits host paths, legacy JSON, parameters, tags, raw observations,
and authority-bearing state.  It also replaces internal run identifiers with a
canonical URL-safe reference before any value reaches an HTTP response.
"""

from __future__ import annotations

import base64
import binascii
import json
from datetime import datetime
from typing import Annotated, Final, Literal

from pydantic import Field, field_validator

from quant_platform.service.contracts import (
    SCHEMA_VERSION,
    StrictServiceContract,
    require_bounded_text,
    require_utc_datetime,
)
from quant_platform.service.manifests import (
    DiagnosticsManifest,
    DiagnosticValue,
    ForecastAggregate,
    ForecastSummaryManifest,
    ManifestKind,
    ModelCardManifest,
    ModelCardSection,
)
from quant_platform.tracking.contracts import (
    ArtifactClass,
    EvidenceClass,
    RunSnapshot,
    RunStatus,
    require_identifier,
)
from quant_platform.tracking.read_ports import (
    ArtifactView,
    EvidenceReadinessCode,
    LegacyRunSnapshot,
    RunArtifactView,
    RunProvenance,
)

_RUN_REFERENCE_PREFIX = "r1_"
_MAX_INTERNAL_RUN_ID_BYTES = 128
_MAX_RUN_REFERENCE_BYTES = 176
# These conservative limits reserve at least 64 KiB inside the middleware's
# 1 MiB response ceiling after accepting one maximum-size manifest per item.
FORECAST_RESPONSE_PAGE_LIMIT: Final = 7
DIAGNOSTICS_RESPONSE_PAGE_LIMIT: Final = 1

RunReference = Annotated[
    str,
    Field(
        min_length=5,
        max_length=_MAX_RUN_REFERENCE_BYTES,
        pattern=r"^r1_[A-Za-z0-9_-]+$",
    ),
]
Digest = Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]
Cursor = Annotated[str, Field(min_length=16, max_length=1_024)]
CommitIdentity = Annotated[str, Field(pattern=r"^[0-9a-fA-F]{7,64}$")]
DataIdentity = Annotated[str, Field(pattern=r"^(?:sha256:)?[0-9a-f]{64}$")]
MediaType = Annotated[
    str,
    Field(pattern=r"^[a-z0-9][a-z0-9!#$&^_.+-]*/[a-z0-9][a-z0-9!#$&^_.+-]{0,126}$"),
]


def encode_run_reference(run_id: str) -> str:
    """Encode one validated internal run ID as a canonical URL path segment."""

    canonical = require_identifier(run_id, "run_id")
    payload = canonical.encode("ascii")
    token = base64.urlsafe_b64encode(payload).rstrip(b"=").decode("ascii")
    reference = f"{_RUN_REFERENCE_PREFIX}{token}"
    if len(reference) > _MAX_RUN_REFERENCE_BYTES:
        raise ValueError("run identifier cannot be represented by the public API")
    return reference


def decode_run_reference(reference: str) -> str:
    """Decode a canonical URL-safe run reference without accepting aliases."""

    if (
        type(reference) is not str
        or not reference.startswith(_RUN_REFERENCE_PREFIX)
        or not 5 <= len(reference) <= _MAX_RUN_REFERENCE_BYTES
        or not reference.isascii()
    ):
        raise ValueError("run reference is malformed")
    token = reference[len(_RUN_REFERENCE_PREFIX) :]
    if not token or "=" in token:
        raise ValueError("run reference is malformed")
    padding = "=" * (-len(token) % 4)
    try:
        decoded = base64.b64decode(token + padding, altchars=b"-_", validate=True)
    except (binascii.Error, ValueError):
        raise ValueError("run reference is malformed") from None
    if not 1 <= len(decoded) <= _MAX_INTERNAL_RUN_ID_BYTES:
        raise ValueError("run reference is outside its byte bound")
    try:
        run_id = decoded.decode("ascii")
    except UnicodeDecodeError:
        raise ValueError("run reference is not ASCII") from None
    canonical = require_identifier(run_id, "run_id")
    if encode_run_reference(canonical) != reference:
        raise ValueError("run reference is not canonical")
    return canonical


class LiveResponse(StrictServiceContract):
    """Storage-independent process liveness."""

    schema_version: Literal[1] = SCHEMA_VERSION
    status: Literal["live"] = "live"


class ReadyResponse(StrictServiceContract):
    """Redacted bounded registry/CAS readiness verdict."""

    schema_version: Literal[1] = SCHEMA_VERSION
    ready: bool
    code: EvidenceReadinessCode
    registry_schema_version: int | None = None
    journal_mode: Literal["wal", "delete"] | None = None
    retryable: bool


class RunEvidenceLinks(StrictServiceContract):
    """Relative versioned links available for one run projection."""

    forecast_summaries: str
    diagnostics: str
    artifacts: str

    @field_validator("forecast_summaries", "diagnostics", "artifacts")
    @classmethod
    def _relative_api_path(cls, value: str) -> str:
        if (
            type(value) is not str
            or not value.startswith("/api/v1/runs/r1_")
            or "?" in value
            or "#" in value
            or len(value) > 256
        ):
            raise ValueError("evidence link must be a bounded relative API path")
        return value


class RunResponse(StrictServiceContract):
    """Path-free public run projection for durable or legacy evidence."""

    schema_version: Literal[1] = SCHEMA_VERSION
    run_id: RunReference
    sequence: int = Field(ge=1, le=2**63 - 1)
    provenance: RunProvenance
    status: RunStatus | None
    evidence_class: EvidenceClass | None
    created_at: datetime | None
    started_at: datetime
    ended_at: datetime | None
    source_commit: CommitIdentity | None
    data_identity: DataIdentity | None
    limitation_summary: str | None
    links: RunEvidenceLinks

    @field_validator("created_at", "started_at", "ended_at")
    @classmethod
    def _utc_times(cls, value: datetime | None, info: object) -> datetime | None:
        if value is None:
            return None
        return require_utc_datetime(value, str(getattr(info, "field_name", "timestamp")))

    @field_validator("limitation_summary")
    @classmethod
    def _bounded_limitation(cls, value: str | None) -> str | None:
        if value is None:
            return None
        return require_bounded_text(value, "limitation summary", maximum_bytes=4_096)

    @classmethod
    def from_read_model(cls, run: RunSnapshot | LegacyRunSnapshot) -> RunResponse:
        """Create the sole approved HTTP projection from a read-port model."""

        reference = encode_run_reference(run.run_id)
        prefix = f"/api/v1/runs/{reference}"
        links = RunEvidenceLinks(
            forecast_summaries=f"{prefix}/forecast-summaries",
            diagnostics=f"{prefix}/diagnostics",
            artifacts=f"{prefix}/artifacts",
        )
        if type(run) is RunSnapshot:
            return cls(
                run_id=reference,
                sequence=run.sequence,
                provenance=RunProvenance.REGISTRY_VERIFIED,
                status=run.status,
                evidence_class=run.evidence_class,
                created_at=run.created_at,
                started_at=run.started_at or run.created_at,
                ended_at=run.ended_at,
                source_commit=run.source_commit,
                data_identity=run.data_identity,
                limitation_summary=run.limitation_summary,
                links=links,
            )
        if type(run) is LegacyRunSnapshot:
            return cls(
                run_id=reference,
                sequence=run.sequence,
                provenance=RunProvenance.LEGACY_UNVERIFIED,
                status=(
                    None if run.reported_terminal_status is None else run.reported_terminal_status
                ),
                evidence_class=None,
                created_at=None,
                started_at=run.started_at,
                ended_at=run.ended_at,
                source_commit=run.source_commit,
                data_identity=run.data_identity,
                limitation_summary=run.limitation_summary,
                links=links,
            )
        raise TypeError("unsupported run read model")


class RunPageResponse(StrictServiceContract):
    """Bounded keyset page of run projections."""

    schema_version: Literal[1] = SCHEMA_VERSION
    items: tuple[RunResponse, ...] = Field(max_length=100)
    next_cursor: Cursor | None = None


class ArtifactResponse(StrictServiceContract):
    """Verified artifact metadata without bytes or a storage pathname."""

    schema_version: Literal[1] = SCHEMA_VERSION
    artifact_id: Digest
    artifact_class: ArtifactClass
    byte_size: int = Field(ge=0, le=2**63 - 1)
    media_type: MediaType
    created_at: datetime
    pinned: bool

    @field_validator("created_at")
    @classmethod
    def _utc_created_at(cls, value: datetime) -> datetime:
        return require_utc_datetime(value, "created_at")

    @classmethod
    def from_view(cls, artifact: ArtifactView) -> ArtifactResponse:
        """Project path-free metadata returned by the registry read port."""

        return cls(
            artifact_id=artifact.artifact_id,
            artifact_class=artifact.artifact_class,
            byte_size=artifact.byte_size,
            media_type=artifact.media_type,
            created_at=artifact.created_at,
            pinned=artifact.pinned,
        )


class RunArtifactResponse(StrictServiceContract):
    """One immutable evidence role and its verified metadata projection."""

    sequence: int = Field(ge=1, le=2**63 - 1)
    role: str
    linked_at: datetime
    artifact: ArtifactResponse

    @field_validator("role")
    @classmethod
    def _bounded_role(cls, value: str) -> str:
        return require_bounded_text(value, "artifact role", maximum_bytes=128)

    @field_validator("linked_at")
    @classmethod
    def _utc_linked_at(cls, value: datetime) -> datetime:
        return require_utc_datetime(value, "linked_at")

    @classmethod
    def from_view(cls, link: RunArtifactView) -> RunArtifactResponse:
        """Project one role link without its internal storage key."""

        return cls(
            sequence=link.sequence,
            role=link.role,
            linked_at=link.linked_at,
            artifact=ArtifactResponse.from_view(link.artifact),
        )


class ArtifactPageResponse(StrictServiceContract):
    """Bounded keyset page of role-linked artifact metadata."""

    schema_version: Literal[1] = SCHEMA_VERSION
    items: tuple[RunArtifactResponse, ...] = Field(max_length=100)
    next_cursor: Cursor | None = None


class _EvidenceManifestResponse(StrictServiceContract):
    """Common public fields projected from one verified internal manifest."""

    schema_version: Literal[1] = SCHEMA_VERSION
    kind: ManifestKind
    run_id: RunReference
    generated_at: datetime
    limitations: tuple[str, ...] = Field(default=(), max_length=16)

    @field_validator("generated_at")
    @classmethod
    def _utc_manifest_time(cls, value: datetime) -> datetime:
        return require_utc_datetime(value, "generated_at")

    @field_validator("limitations")
    @classmethod
    def _bounded_manifest_limitations(cls, values: tuple[str, ...]) -> tuple[str, ...]:
        validated = tuple(
            require_bounded_text(value, "limitation", maximum_bytes=2_048) for value in values
        )
        if len(set(validated)) != len(validated):
            raise ValueError("limitations must be unique")
        return validated


class ForecastSummaryResponse(_EvidenceManifestResponse):
    """Aggregate forecast manifest with a canonical public run reference."""

    kind: Literal[ManifestKind.FORECAST_SUMMARY] = ManifestKind.FORECAST_SUMMARY
    evidence_granularity: Literal["aggregate"] = "aggregate"
    row_level_available: Literal[False] = False
    aggregates: tuple[ForecastAggregate, ...] = Field(min_length=1, max_length=100)

    @classmethod
    def from_manifest(cls, manifest: ForecastSummaryManifest) -> ForecastSummaryResponse:
        """Project one verified storage contract without its internal identifier."""

        return cls(
            run_id=encode_run_reference(manifest.run_id),
            generated_at=manifest.generated_at,
            limitations=manifest.limitations,
            aggregates=manifest.aggregates,
        )


class DiagnosticsResponse(_EvidenceManifestResponse):
    """Diagnostics manifest with a canonical public run reference."""

    kind: Literal[ManifestKind.DIAGNOSTICS] = ManifestKind.DIAGNOSTICS
    values: tuple[DiagnosticValue, ...] = Field(min_length=1, max_length=256)

    @classmethod
    def from_manifest(cls, manifest: DiagnosticsManifest) -> DiagnosticsResponse:
        """Project one verified storage contract without its internal identifier."""

        return cls(
            run_id=encode_run_reference(manifest.run_id),
            generated_at=manifest.generated_at,
            limitations=manifest.limitations,
            values=manifest.values,
        )


class ModelCardManifestResponse(_EvidenceManifestResponse):
    """Structured model card with a canonical public run reference."""

    kind: Literal[ManifestKind.MODEL_CARD] = ManifestKind.MODEL_CARD
    card_id: str
    model_name: str
    model_version: str
    text_format: Literal["plain_text"] = "plain_text"
    sections: tuple[ModelCardSection, ...] = Field(min_length=1, max_length=8)

    @field_validator("card_id", "model_name", "model_version")
    @classmethod
    def _bounded_card_identifiers(cls, value: str, info: object) -> str:
        return require_bounded_text(
            value,
            str(getattr(info, "field_name", "model-card field")),
            maximum_bytes=128,
        )

    @classmethod
    def from_manifest(cls, manifest: ModelCardManifest) -> ModelCardManifestResponse:
        """Project one verified storage contract without its internal identifier."""

        return cls(
            run_id=encode_run_reference(manifest.run_id),
            generated_at=manifest.generated_at,
            limitations=manifest.limitations,
            card_id=manifest.card_id,
            model_name=manifest.model_name,
            model_version=manifest.model_version,
            sections=manifest.sections,
        )


class ForecastSummaryPageResponse(StrictServiceContract):
    """Bounded page of aggregate-only forecast manifests."""

    schema_version: Literal[1] = SCHEMA_VERSION
    evidence_granularity: Literal["aggregate"] = "aggregate"
    row_level_available: Literal[False] = False
    items: tuple[ForecastSummaryResponse, ...] = Field(max_length=FORECAST_RESPONSE_PAGE_LIMIT)
    next_cursor: Cursor | None = None


class DiagnosticsPageResponse(StrictServiceContract):
    """Bounded page of strict diagnostic manifests."""

    schema_version: Literal[1] = SCHEMA_VERSION
    items: tuple[DiagnosticsResponse, ...] = Field(max_length=DIAGNOSTICS_RESPONSE_PAGE_LIMIT)
    next_cursor: Cursor | None = None


class ModelCardSummary(StrictServiceContract):
    """Small model-card listing projection."""

    artifact_id: Digest
    card_id: str
    model_name: str
    model_version: str
    generated_at: datetime
    limitation_count: int = Field(ge=0, le=16)

    @field_validator("card_id", "model_name", "model_version")
    @classmethod
    def _bounded_identifiers(cls, value: str, info: object) -> str:
        return require_bounded_text(
            value,
            str(getattr(info, "field_name", "model-card field")),
            maximum_bytes=128,
        )

    @field_validator("generated_at")
    @classmethod
    def _utc_generated_at(cls, value: datetime) -> datetime:
        return require_utc_datetime(value, "generated_at")


class ModelCardPageResponse(StrictServiceContract):
    """Bounded run-scoped page of model-card summaries."""

    schema_version: Literal[1] = SCHEMA_VERSION
    items: tuple[ModelCardSummary, ...] = Field(max_length=100)
    next_cursor: Cursor | None = None


class ModelCardResponse(StrictServiceContract):
    """One verified structured plain-text model card."""

    schema_version: Literal[1] = SCHEMA_VERSION
    artifact_id: Digest
    model_card: ModelCardManifestResponse


def canonical_openapi_bytes(document: dict[str, object]) -> bytes:
    """Serialize the checked public OpenAPI document deterministically."""

    return (
        json.dumps(document, allow_nan=False, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")
