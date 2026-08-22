"""Strict aggregate-only evidence manifests for the read-only service.

The service reads only these closed schemas from the verified CAS boundary.  It
does not deserialize arbitrary JSON, Markdown, reports, model files, or raw
forecast observations.  Canonical parsing is intentionally stricter than normal
JSON parsing: duplicate keys, non-finite numbers, semantic default omission,
alternate timestamp spellings, excessive nesting, and unknown fields all fail
closed before projection to an HTTP response.
"""

from __future__ import annotations

import hmac
from datetime import datetime
from enum import StrEnum
from typing import Final, Literal, cast

from pydantic import Field, ValidationInfo, field_validator, model_validator
from pydantic import ValidationError as PydanticValidationError

from quant_platform.service.contracts import (
    SCHEMA_VERSION,
    ServiceContractError,
    StrictServiceContract,
    require_bounded_text,
    require_finite_number,
    require_url_identifier,
    require_utc_datetime,
)
from quant_platform.tracking.contracts import ValidationError as TrackingValidationError
from quant_platform.tracking.contracts import decode_bounded_json, require_identifier

MAX_MANIFEST_DEPTH: Final = 16
MAX_FORECAST_AGGREGATES: Final = 100
MAX_DIAGNOSTIC_VALUES: Final = 256
MAX_MODEL_CARD_TEXT_BYTES: Final = 64 * 1024
MAX_LIMITATIONS: Final = 16

_MAX_FORECAST_MANIFEST_BYTES: Final = 128 * 1024
_MAX_DIAGNOSTICS_MANIFEST_BYTES: Final = 512 * 1024
_MAX_MODEL_CARD_MANIFEST_BYTES: Final = 96 * 1024


class ManifestValidationError(ValueError):
    """Untrusted evidence bytes violated the closed manifest boundary."""

    code = "invalid_evidence_manifest"


class ManifestKind(StrEnum):
    """Closed artifact semantics accepted by the aggregate evidence reader."""

    FORECAST_SUMMARY = "forecast_summary"
    DIAGNOSTICS = "diagnostics"
    MODEL_CARD = "model_card"


class ForecastSplit(StrEnum):
    """Closed temporal split labels for forecast evaluation aggregates."""

    TRAIN = "train"
    VALIDATION = "validation"
    TEST = "test"
    SHADOW = "shadow"


class HorizonUnit(StrEnum):
    """Units supported by aggregate forecast horizons."""

    BARS = "bars"
    CALENDAR_DAYS = "calendar_days"
    BUSINESS_DAYS = "business_days"


class DiagnosticCategory(StrEnum):
    """Closed diagnostic families exposed by the public projection."""

    CALIBRATION = "calibration"
    UNCERTAINTY = "uncertainty"
    LATENCY = "latency"
    LINEAGE = "lineage"
    READINESS = "readiness"


class DiagnosticStatus(StrEnum):
    """Decision state attached to one diagnostic value."""

    PASS = "pass"
    WARN = "warn"
    FAIL = "fail"
    INFORMATIONAL = "informational"


class ModelCardSectionName(StrEnum):
    """Ordered, closed sections allowed in a structured model card."""

    OVERVIEW = "overview"
    INTENDED_USE = "intended_use"
    OUT_OF_SCOPE_USE = "out_of_scope_use"
    DATA = "data"
    EVALUATION = "evaluation"
    LIMITATIONS = "limitations"
    ETHICAL_CONSIDERATIONS = "ethical_considerations"
    MONITORING = "monitoring"


_MODEL_CARD_SECTION_ORDER: Final = tuple(ModelCardSectionName)
_REQUIRED_MODEL_CARD_SECTIONS: Final = frozenset(
    {
        ModelCardSectionName.OVERVIEW,
        ModelCardSectionName.INTENDED_USE,
        ModelCardSectionName.OUT_OF_SCOPE_USE,
        ModelCardSectionName.DATA,
        ModelCardSectionName.EVALUATION,
        ModelCardSectionName.LIMITATIONS,
        ModelCardSectionName.MONITORING,
    }
)


class _EvidenceManifest(StrictServiceContract):
    """Fields common to every immutable aggregate evidence manifest."""

    schema_version: Literal[1] = SCHEMA_VERSION
    kind: ManifestKind
    run_id: str
    generated_at: datetime
    limitations: tuple[str, ...] = Field(default=(), max_length=MAX_LIMITATIONS)

    @field_validator("run_id")
    @classmethod
    def _run_identifier(cls, value: str) -> str:
        return require_identifier(value, "run_id")

    @field_validator("generated_at")
    @classmethod
    def _generated_at_utc(cls, value: datetime) -> datetime:
        return require_utc_datetime(value, "generated_at")

    @field_validator("limitations")
    @classmethod
    def _bounded_unique_limitations(cls, values: tuple[str, ...]) -> tuple[str, ...]:
        validated = tuple(
            require_bounded_text(value, "limitation", maximum_bytes=2_048, multiline=False)
            for value in values
        )
        if len(set(validated)) != len(validated):
            raise ValueError("limitations must be unique")
        return validated


class ForecastAggregate(StrictServiceContract):
    """Multi-observation forecast evidence with no row- or instrument-level values."""

    aggregate_id: str
    target: str
    split: ForecastSplit
    horizon_steps: int = Field(ge=1, le=1_000_000)
    horizon_unit: HorizonUnit
    window_start: datetime
    window_end: datetime
    sample_count: int = Field(ge=2, le=2**63 - 1)
    mean_prediction: float
    mean_observation: float
    mean_error: float
    mean_absolute_error: float = Field(ge=0.0)
    root_mean_squared_error: float = Field(ge=0.0)
    interval_coverage: float | None = Field(default=None, ge=0.0, le=1.0)
    mean_interval_width: float | None = Field(default=None, ge=0.0)

    @field_validator("aggregate_id", "target")
    @classmethod
    def _bounded_identifiers(cls, value: str, info: ValidationInfo) -> str:
        return require_url_identifier(value, cast(str, info.field_name))

    @field_validator("window_start", "window_end")
    @classmethod
    def _window_utc(cls, value: datetime, info: ValidationInfo) -> datetime:
        return require_utc_datetime(value, cast(str, info.field_name))

    @field_validator(
        "mean_prediction",
        "mean_observation",
        "mean_error",
        "mean_absolute_error",
        "root_mean_squared_error",
        "interval_coverage",
        "mean_interval_width",
        mode="before",
    )
    @classmethod
    def _finite_metrics(cls, value: object, info: ValidationInfo) -> float | None:
        if value is None:
            return None
        return require_finite_number(cast(float, value), cast(str, info.field_name))

    @model_validator(mode="after")
    def _consistent_aggregate(self) -> ForecastAggregate:
        if self.window_end < self.window_start:
            raise ValueError("window_end may not precede window_start")
        tolerance = max(1.0, self.root_mean_squared_error) * 1e-12
        if self.root_mean_squared_error + tolerance < self.mean_absolute_error:
            raise ValueError("root_mean_squared_error may not be smaller than mean_absolute_error")
        if (self.interval_coverage is None) != (self.mean_interval_width is None):
            raise ValueError("interval coverage and mean width must be present or absent together")
        return self


class ForecastSummaryManifest(_EvidenceManifest):
    """Version-1 aggregate forecast evidence; observation rows are unrepresentable."""

    kind: Literal[ManifestKind.FORECAST_SUMMARY] = ManifestKind.FORECAST_SUMMARY
    evidence_granularity: Literal["aggregate"] = "aggregate"
    row_level_available: Literal[False] = False
    aggregates: tuple[ForecastAggregate, ...] = Field(
        min_length=1,
        max_length=MAX_FORECAST_AGGREGATES,
    )

    @field_validator("aggregates")
    @classmethod
    def _unique_canonical_aggregates(
        cls,
        values: tuple[ForecastAggregate, ...],
    ) -> tuple[ForecastAggregate, ...]:
        identifiers = tuple(value.aggregate_id for value in values)
        if len(set(identifiers)) != len(identifiers):
            raise ValueError("forecast aggregate identifiers must be unique")
        if identifiers != tuple(sorted(identifiers)):
            raise ValueError("forecast aggregates must be sorted by aggregate_id")
        return values


type DiagnosticScalar = bool | int | float | str


class DiagnosticValue(StrictServiceContract):
    """One bounded scalar diagnostic; arbitrary recursive JSON is forbidden."""

    code: str
    category: DiagnosticCategory
    value: DiagnosticScalar
    unit: str | None = None
    status: DiagnosticStatus
    detail: str | None = None

    @field_validator("code")
    @classmethod
    def _diagnostic_code(cls, value: str) -> str:
        return require_url_identifier(value, "diagnostic code")

    @field_validator("value")
    @classmethod
    def _typed_bounded_scalar(cls, value: DiagnosticScalar) -> DiagnosticScalar:
        if type(value) is bool:
            return value
        if type(value) is int:
            if -(2**63) <= value <= 2**63 - 1:
                return value
            raise ValueError("diagnostic integer is outside signed 64-bit range")
        if type(value) is float:
            return require_finite_number(value, "diagnostic value")
        if type(value) is str:
            return require_bounded_text(
                value,
                "diagnostic value",
                maximum_bytes=256,
                multiline=False,
            )
        raise ValueError("diagnostic value must be a strict scalar")

    @field_validator("unit")
    @classmethod
    def _bounded_unit(cls, value: str | None) -> str | None:
        if value is None:
            return None
        return require_bounded_text(value, "diagnostic unit", maximum_bytes=32, multiline=False)

    @field_validator("detail")
    @classmethod
    def _bounded_detail(cls, value: str | None) -> str | None:
        if value is None:
            return None
        return require_bounded_text(
            value,
            "diagnostic detail",
            maximum_bytes=1_024,
            multiline=False,
        )


class DiagnosticsManifest(_EvidenceManifest):
    """Version-1 bounded calibration, uncertainty, latency, lineage, and readiness evidence."""

    kind: Literal[ManifestKind.DIAGNOSTICS] = ManifestKind.DIAGNOSTICS
    values: tuple[DiagnosticValue, ...] = Field(min_length=1, max_length=MAX_DIAGNOSTIC_VALUES)

    @field_validator("values")
    @classmethod
    def _unique_canonical_values(
        cls,
        values: tuple[DiagnosticValue, ...],
    ) -> tuple[DiagnosticValue, ...]:
        keys = tuple((value.category.value, value.code) for value in values)
        if len(set(keys)) != len(keys):
            raise ValueError("diagnostic category/code pairs must be unique")
        if keys != tuple(sorted(keys)):
            raise ValueError("diagnostic values must be sorted by category and code")
        return values


class ModelCardSection(StrictServiceContract):
    """One named plain-text model-card section with no rendering semantics."""

    name: ModelCardSectionName
    title: str
    text: str

    @field_validator("title")
    @classmethod
    def _bounded_title(cls, value: str) -> str:
        return require_bounded_text(value, "model-card section title", maximum_bytes=128)

    @field_validator("text")
    @classmethod
    def _bounded_text(cls, value: str) -> str:
        return require_bounded_text(
            value,
            "model-card section text",
            maximum_bytes=MAX_MODEL_CARD_TEXT_BYTES,
            multiline=True,
        )


class ModelCardManifest(_EvidenceManifest):
    """Version-1 structured plain-text model card bound to exactly one run."""

    kind: Literal[ManifestKind.MODEL_CARD] = ManifestKind.MODEL_CARD
    card_id: str
    model_name: str
    model_version: str
    text_format: Literal["plain_text"] = "plain_text"
    sections: tuple[ModelCardSection, ...] = Field(min_length=1, max_length=8)

    @field_validator("card_id", "model_name", "model_version")
    @classmethod
    def _card_identifiers(cls, value: str, info: ValidationInfo) -> str:
        return require_url_identifier(value, cast(str, info.field_name))

    @model_validator(mode="after")
    def _complete_bounded_card(self) -> ModelCardManifest:
        names = tuple(section.name for section in self.sections)
        if len(set(names)) != len(names):
            raise ValueError("model-card section names must be unique")
        order = {name: index for index, name in enumerate(_MODEL_CARD_SECTION_ORDER)}
        if tuple(sorted(names, key=order.__getitem__)) != names:
            raise ValueError("model-card sections must follow the canonical section order")
        missing = _REQUIRED_MODEL_CARD_SECTIONS.difference(names)
        if missing:
            raise ValueError("model card omits one or more required structured sections")
        text_values = (
            self.card_id,
            self.model_name,
            self.model_version,
            *self.limitations,
            *(section.title for section in self.sections),
            *(section.text for section in self.sections),
        )
        try:
            total_bytes = sum(len(value.encode("utf-8")) for value in text_values)
        except (MemoryError, UnicodeEncodeError):
            raise ValueError("model-card text must be bounded valid UTF-8") from None
        if total_bytes > MAX_MODEL_CARD_TEXT_BYTES:
            raise ValueError(
                f"cumulative model-card text exceeds {MAX_MODEL_CARD_TEXT_BYTES} UTF-8 bytes"
            )
        return self


type EvidenceManifest = ForecastSummaryManifest | DiagnosticsManifest | ModelCardManifest

_MANIFEST_BYTE_LIMIT: Final = {
    ManifestKind.FORECAST_SUMMARY: _MAX_FORECAST_MANIFEST_BYTES,
    ManifestKind.DIAGNOSTICS: _MAX_DIAGNOSTICS_MANIFEST_BYTES,
    ManifestKind.MODEL_CARD: _MAX_MODEL_CARD_MANIFEST_BYTES,
}


def parse_evidence_manifest(
    payload: bytes,
    *,
    expected_kind: ManifestKind,
    expected_run_id: str,
) -> EvidenceManifest:
    """Parse canonical verified-CAS bytes into one expected aggregate manifest.

    ``expected_kind`` is bound by the artifact role/media contract and
    ``expected_run_id`` is bound by the route.  Both must agree with the bytes;
    callers may not trust an attacker-selected discriminator or cross-run link.
    Parsing work is bounded before both stdlib and Pydantic decoding.
    """

    if type(payload) is not bytes:
        raise ManifestValidationError("manifest payload must be exact bytes")
    if type(expected_kind) is not ManifestKind:
        raise ManifestValidationError("expected manifest kind is invalid")
    try:
        canonical_run_id = require_identifier(expected_run_id, "expected_run_id")
        decoded = decode_bounded_json(
            payload,
            maximum_bytes=_MANIFEST_BYTE_LIMIT[expected_kind],
            require_canonical=True,
        )
    except (TrackingValidationError, ValueError):
        raise ManifestValidationError(
            "manifest bytes violate the bounded canonical JSON contract"
        ) from None
    if type(decoded) is not dict:
        raise ManifestValidationError("manifest root must be an object")
    version = decoded.get("schema_version")
    if type(version) is not int or version != SCHEMA_VERSION:
        raise ManifestValidationError("manifest schema version is unsupported")
    kind = decoded.get("kind")
    # Kind is public routing metadata, not a secret.  Ordinary equality safely
    # handles hostile non-ASCII strings; ``compare_digest`` rejects them with a
    # raw ``TypeError`` before the typed failure boundary can respond.
    if type(kind) is not str or kind != expected_kind.value:
        raise ManifestValidationError("manifest kind does not match the artifact role")
    try:
        if expected_kind is ManifestKind.FORECAST_SUMMARY:
            manifest: EvidenceManifest = ForecastSummaryManifest.model_validate_json(payload)
        elif expected_kind is ManifestKind.DIAGNOSTICS:
            manifest = DiagnosticsManifest.model_validate_json(payload)
        else:
            manifest = ModelCardManifest.model_validate_json(payload)
        canonical = manifest.canonical_json_bytes(
            maximum_bytes=_MANIFEST_BYTE_LIMIT[expected_kind],
        )
    except (
        MemoryError,
        PydanticValidationError,
        RecursionError,
        ServiceContractError,
        UnicodeError,
        ValueError,
    ):
        raise ManifestValidationError("manifest violates its strict typed contract") from None
    if not hmac.compare_digest(canonical, payload):
        raise ManifestValidationError("manifest bytes are not the canonical typed representation")
    if not hmac.compare_digest(manifest.run_id, canonical_run_id):
        raise ManifestValidationError("manifest run identity does not match the requested run")
    return manifest
