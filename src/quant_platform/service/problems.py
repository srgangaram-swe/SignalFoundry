"""Redacted RFC 9457 problem details for the read-only service boundary."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

from pydantic import Field

from quant_platform.service.contracts import StrictServiceContract, require_bounded_text
from quant_platform.service.manifests import ManifestValidationError
from quant_platform.tracking.contracts import (
    BusyError,
    CapacityError,
    IntegrityError,
    InvalidCursorError,
    NotFoundError,
    RegistryError,
    ValidationError,
    VerificationTimeoutError,
)
from quant_platform.tracking.read_ports import ReadTimeoutError


class ProblemDocument(StrictServiceContract):
    """RFC 9457 response with a stable service code and server request ID."""

    type: str
    title: str
    status: int = Field(ge=400, le=599)
    detail: str
    code: str
    request_id: str


@dataclass(frozen=True, slots=True)
class ProblemSpec:
    """Closed safe mapping from an internal failure family to public semantics."""

    status: int
    code: str
    title: str
    detail: str
    retry_after_seconds: int | None = None

    def __post_init__(self) -> None:
        if type(self.status) is not int or not 400 <= self.status <= 599:
            raise ValueError("problem status must be a 4xx or 5xx integer")
        require_bounded_text(self.code, "problem code", maximum_bytes=64)
        require_bounded_text(self.title, "problem title", maximum_bytes=128)
        require_bounded_text(self.detail, "problem detail", maximum_bytes=256)
        if self.retry_after_seconds is not None and (
            type(self.retry_after_seconds) is not int or not 1 <= self.retry_after_seconds <= 3_600
        ):
            raise ValueError("Retry-After must be a bounded positive integer")


BAD_REQUEST: Final = ProblemSpec(
    400,
    "invalid_request",
    "Invalid request",
    "The request violates the bounded read-only API contract.",
)
VALIDATION_FAILED: Final = ProblemSpec(
    422,
    "request_validation_failed",
    "Request validation failed",
    "One or more request values do not satisfy the public API schema.",
)
NOT_FOUND: Final = ProblemSpec(
    404,
    "not_found",
    "Resource not found",
    "The requested evidence resource does not exist.",
)
METHOD_NOT_ALLOWED: Final = ProblemSpec(
    405,
    "method_not_allowed",
    "Method not allowed",
    "This service exposes GET operations only.",
)
PAYLOAD_TOO_LARGE: Final = ProblemSpec(
    413,
    "request_body_forbidden",
    "Request body forbidden",
    "GET requests to this service may not carry a body.",
)
SATURATED: Final = ProblemSpec(
    429,
    "service_saturated",
    "Service saturated",
    "The bounded local service has reached its concurrency limit.",
    1,
)
UNAVAILABLE: Final = ProblemSpec(
    503,
    "evidence_unavailable",
    "Evidence unavailable",
    "Verified evidence storage is temporarily unavailable or not ready.",
    1,
)
INTEGRITY_FAILED: Final = ProblemSpec(
    503,
    "evidence_integrity_failed",
    "Evidence integrity verification failed",
    "The requested evidence did not satisfy its verified storage contract.",
)
INTERNAL: Final = ProblemSpec(
    500,
    "internal_error",
    "Internal service error",
    "The service could not complete the bounded read operation.",
)


def problem_for_exception(error: Exception) -> ProblemSpec:
    """Return a non-reflective public mapping for one internal exception."""

    if isinstance(error, NotFoundError):
        return NOT_FOUND
    if isinstance(error, CapacityError):
        return SATURATED
    if isinstance(error, (IntegrityError, ManifestValidationError)):
        return INTEGRITY_FAILED
    if isinstance(error, (BusyError, ReadTimeoutError, VerificationTimeoutError)):
        return UNAVAILABLE
    if isinstance(error, (InvalidCursorError, ValidationError, ValueError)):
        return BAD_REQUEST
    if isinstance(error, RegistryError):
        return UNAVAILABLE
    return INTERNAL


def build_problem(spec: ProblemSpec, request_id: str) -> ProblemDocument:
    """Build the canonical RFC 9457 payload without reflecting request data."""

    return ProblemDocument(
        type=f"urn:signalattice:problem:{spec.code}",
        title=spec.title,
        status=spec.status,
        detail=spec.detail,
        code=spec.code,
        request_id=request_id,
    )
