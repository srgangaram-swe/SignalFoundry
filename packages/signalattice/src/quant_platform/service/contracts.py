"""Shared invariants for versioned read-only service contracts.

Values crossing the service boundary are untrusted even when they came from the
local content-addressed store.  These helpers keep identifiers URL-segment safe,
text valid and resource-bounded, timestamps normalized to UTC, and JSON output
finite and deterministic.  Contracts are immutable and reject coercion and
unknown fields so stored evidence cannot silently acquire new semantics.
"""

from __future__ import annotations

import json
import math
import re
from datetime import UTC, datetime
from typing import Final

from pydantic import BaseModel, ConfigDict

SCHEMA_VERSION: Final = 1
MAX_IDENTIFIER_BYTES: Final = 128
MAX_CANONICAL_CONTRACT_BYTES: Final = 512 * 1024

_URL_SEGMENT_IDENTIFIER: Final = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._~-]{0,127}$")


class ServiceContractError(ValueError):
    """A service value cannot be represented by the public bounded contract."""

    code = "invalid_service_contract"


class StrictServiceContract(BaseModel):
    """Immutable Pydantic-v2 base that forbids coercion and non-finite JSON.

    Nested public structures use other frozen contracts or tuples rather than
    mutable mappings.  This makes the object graph stable after validation and
    keeps canonical serialization independent of caller-owned state.
    """

    model_config = ConfigDict(
        allow_inf_nan=False,
        extra="forbid",
        frozen=True,
        strict=True,
        validate_default=True,
    )

    def canonical_json_bytes(
        self,
        *,
        maximum_bytes: int = MAX_CANONICAL_CONTRACT_BYTES,
    ) -> bytes:
        """Return deterministic finite UTF-8 JSON within ``maximum_bytes``.

        The output contains every declared field, including explicit version and
        granularity constants.  Callers therefore cannot create equivalent but
        wire-incompatible representations by omitting defaults.
        """

        if type(maximum_bytes) is not int or not 1 <= maximum_bytes <= MAX_CANONICAL_CONTRACT_BYTES:
            raise ServiceContractError(
                "maximum_bytes is outside the supported canonical contract bound"
            )
        try:
            payload = json.dumps(
                self.model_dump(mode="json"),
                allow_nan=False,
                ensure_ascii=False,
                separators=(",", ":"),
                sort_keys=True,
            ).encode("utf-8")
        except (MemoryError, RecursionError, TypeError, UnicodeError, ValueError):
            raise ServiceContractError(
                "contract cannot be serialized as finite UTF-8 JSON"
            ) from None
        if not 1 <= len(payload) <= maximum_bytes:
            raise ServiceContractError("canonical contract exceeds its byte bound")
        return payload


def require_url_identifier(value: str, field_name: str) -> str:
    """Return one RFC-3986-unreserved, bounded path-segment identifier."""

    if type(value) is not str or not _URL_SEGMENT_IDENTIFIER.fullmatch(value):
        raise ValueError(
            f"{field_name} must be a 1 to {MAX_IDENTIFIER_BYTES}-byte URL-safe identifier"
        )
    # The accepted alphabet is ASCII, so regex length and encoded length agree.
    return value


def require_bounded_text(
    value: str,
    field_name: str,
    *,
    minimum_bytes: int = 1,
    maximum_bytes: int,
    multiline: bool = False,
) -> str:
    """Return valid UTF-8 plain text satisfying byte and control bounds.

    Newlines and horizontal tabs are permitted only for explicitly multiline
    fields.  All other C0/C1 controls are rejected.  The service transports this
    value as JSON text and never interprets it as Markdown or HTML.
    """

    if (
        type(minimum_bytes) is not int
        or type(maximum_bytes) is not int
        or minimum_bytes < 0
        or maximum_bytes < minimum_bytes
        or maximum_bytes > MAX_CANONICAL_CONTRACT_BYTES
        or type(multiline) is not bool
    ):
        raise ValueError("text byte bounds are invalid")
    if type(value) is not str:
        raise ValueError(f"{field_name} must be exact text")
    try:
        size = len(value.encode("utf-8"))
    except (MemoryError, UnicodeEncodeError):
        raise ValueError(f"{field_name} must contain valid bounded UTF-8") from None
    if not minimum_bytes <= size <= maximum_bytes:
        raise ValueError(
            f"{field_name} must contain {minimum_bytes} to {maximum_bytes} UTF-8 bytes"
        )
    for character in value:
        ordinal = ord(character)
        if ordinal == 0x7F or 0x80 <= ordinal <= 0x9F:
            raise ValueError(f"{field_name} contains a forbidden control character")
        if ordinal < 0x20 and not (multiline and character in {"\n", "\t"}):
            raise ValueError(f"{field_name} contains a forbidden control character")
    return value


def require_utc_datetime(value: datetime, field_name: str) -> datetime:
    """Return an aware timestamp normalized to the UTC singleton timezone."""

    if type(value) is not datetime or value.tzinfo is None:
        raise ValueError(f"{field_name} must be a timezone-aware datetime")
    try:
        if value.utcoffset() is None:
            raise ValueError(f"{field_name} must be a timezone-aware datetime")
        normalized = value.astimezone(UTC)
    except Exception:
        raise ValueError(f"{field_name} cannot be normalized safely to UTC") from None
    return normalized


def require_finite_number(value: float, field_name: str) -> float:
    """Return an exact finite float without coercing integers or booleans."""

    if type(value) is not float or not math.isfinite(value):
        raise ValueError(f"{field_name} must be an exact finite float")
    return value
