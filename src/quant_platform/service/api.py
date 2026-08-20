"""Version-1 local-only FastAPI adapter over injected evidence read ports."""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Annotated, Any, Final, Protocol, cast

from fastapi import FastAPI, Query, Request
from fastapi import Path as ApiPath
from fastapi.exceptions import RequestValidationError
from fastapi.openapi.utils import get_openapi
from fastapi.responses import Response
from starlette.exceptions import HTTPException as StarletteHTTPException

from quant_platform.service.admission import (
    AdmissionController,
    AdmissionLimits,
)
from quant_platform.service.console import ConsoleBundle
from quant_platform.service.governance_models import (
    LANE_RESPONSE_PAGE_LIMIT,
    ComparisonPageResponse,
    LaneDetailResponse,
    LanePageResponse,
)
from quant_platform.service.manifests import (
    DiagnosticsManifest,
    ForecastSummaryManifest,
    ManifestKind,
    ManifestValidationError,
    ModelCardManifest,
    parse_evidence_manifest,
)
from quant_platform.service.metrics import ServiceMetrics
from quant_platform.service.middleware import SecurityBoundaryMiddleware
from quant_platform.service.models import (
    DIAGNOSTICS_RESPONSE_PAGE_LIMIT,
    FORECAST_RESPONSE_PAGE_LIMIT,
    ArtifactPageResponse,
    ArtifactResponse,
    DiagnosticsPageResponse,
    DiagnosticsResponse,
    ForecastSummaryPageResponse,
    ForecastSummaryResponse,
    LiveResponse,
    ModelCardManifestResponse,
    ModelCardPageResponse,
    ModelCardResponse,
    ModelCardSummary,
    ReadyResponse,
    RunArtifactResponse,
    RunPageResponse,
    RunResponse,
    canonical_openapi_bytes,
    decode_run_reference,
)
from quant_platform.service.problems import (
    BAD_REQUEST,
    METHOD_NOT_ALLOWED,
    NOT_FOUND,
    VALIDATION_FAILED,
    ProblemDocument,
    build_problem,
    problem_for_exception,
)
from quant_platform.service.telemetry import ServiceTelemetry
from quant_platform.tracking.contracts import (
    ArtifactCursor,
    CapacityError,
    IntegrityError,
    NotFoundError,
    Page,
    RegistryError,
    RunCursor,
    RunStatus,
)
from quant_platform.tracking.read_ports import (
    ArtifactPageRequest,
    ArtifactQuery,
    ArtifactView,
    EvidenceReadiness,
    RegistryReadPorts,
    RunArtifactView,
    RunPageRequest,
    RunProvenance,
    RunQuery,
    RunReadModel,
)

_MANIFEST_MEDIA_TYPE = "application/vnd.signalattice.manifest+json"
_FORECAST_MANIFEST_BYTES = 128 * 1024
_DIAGNOSTICS_MANIFEST_BYTES = 512 * 1024
_MODEL_CARD_MANIFEST_BYTES = 96 * 1024
# The contract describes its own document locally rather than by reference to the
# remote OpenAPI meta-schema. A local-only, offline service whose published
# contract cannot be interpreted without a network fetch is not self-contained,
# and every consumer -- including the console's type generator -- would have to
# reach the public internet to read it.
_OPENAPI_DOCUMENT_SCHEMA_NAME = "OpenApiDocument"
_OPENAPI_DOCUMENT_SCHEMA_REF = f"#/components/schemas/{_OPENAPI_DOCUMENT_SCHEMA_NAME}"
_OPENAPI_DOCUMENT_SCHEMA: Final[dict[str, object]] = {
    "type": "object",
    "title": _OPENAPI_DOCUMENT_SCHEMA_NAME,
    "description": (
        "This service's own OpenAPI 3.1 document. Conformance to the OpenAPI "
        "meta-schema is asserted by the contract test suite rather than by a remote "
        "$ref, so the published contract resolves entirely offline."
    ),
    "additionalProperties": True,
}

_PROBLEM_DESCRIPTIONS = {
    400: "The bounded request contract was violated.",
    404: "The requested safe evidence projection does not exist.",
    405: "Only GET is supported.",
    413: "Request bodies are forbidden.",
    422: "A typed query or path value failed validation.",
    429: "A bounded rate or concurrency admission limit was reached.",
    500: "An unexpected failure was redacted at the service boundary.",
    503: "The service is draining or a verified evidence or response boundary is unavailable.",
}
_COMMON_PROBLEM_STATUSES = frozenset({400, 405, 413, 429, 500, 503})
_VALIDATED_PATHS = frozenset(
    {
        "/api/v1/runs",
        "/api/v1/runs/{run_id}",
        "/api/v1/runs/{run_id}/forecast-summaries",
        "/api/v1/runs/{run_id}/diagnostics",
        "/api/v1/runs/{run_id}/artifacts",
        "/api/v1/artifacts/{artifact_id}",
        "/api/v1/model-cards",
        "/api/v1/model-cards/{card_id}",
    }
)
_NOT_FOUND_PATHS = frozenset(
    {
        "/api/v1/runs/{run_id}",
        "/api/v1/runs/{run_id}/forecast-summaries",
        "/api/v1/runs/{run_id}/diagnostics",
        "/api/v1/runs/{run_id}/artifacts",
        "/api/v1/artifacts/{artifact_id}",
        "/api/v1/model-cards",
        "/api/v1/model-cards/{card_id}",
    }
)
_STORAGE_PATHS = frozenset(
    {
        "/api/v1/runs",
        "/api/v1/runs/{run_id}",
        "/api/v1/runs/{run_id}/forecast-summaries",
        "/api/v1/runs/{run_id}/diagnostics",
        "/api/v1/runs/{run_id}/artifacts",
        "/api/v1/artifacts/{artifact_id}",
        "/api/v1/model-cards",
        "/api/v1/model-cards/{card_id}",
    }
)

PageSize = Annotated[int, Query(ge=1, le=100, description="Page size; default 25, maximum 100.")]
CursorQuery = Annotated[
    str | None,
    Query(min_length=16, max_length=1_024, description="Opaque authenticated keyset cursor."),
]
RunReferenceQuery = Annotated[
    str,
    Query(min_length=5, max_length=176, pattern=r"^r1_[A-Za-z0-9_-]+$"),
]
RunReferencePath = Annotated[
    str,
    ApiPath(min_length=5, max_length=176, pattern=r"^r1_[A-Za-z0-9_-]+$"),
]
DigestPath = Annotated[str, ApiPath(min_length=64, max_length=64, pattern=r"^[0-9a-f]{64}$")]
# A governance lane cursor is a lane identity, so it is validated as one rather
# than as opaque text: an unparseable cursor fails at the boundary.
LaneCursorQuery = Annotated[
    str | None,
    Query(min_length=64, max_length=64, pattern=r"^[0-9a-f]{64}$"),
]
CardIdentifierPath = Annotated[
    str,
    ApiPath(min_length=1, max_length=128, pattern=r"^[A-Za-z0-9][A-Za-z0-9._~-]{0,127}$"),
]


class GovernanceReadPort(Protocol):
    """Structural contract for the bounded governance projection.

    Declared here rather than imported so the service package does not depend on
    the governance package. The hardened image ships the service modules without
    governance, and a concrete import would place the writable governance store
    inside a container whose whole point is that it cannot write.
    """

    def list_lanes(self, *, page_size: int = ..., cursor: str | None = ...) -> Any:
        """Return one bounded page of lane summaries."""

    def get_lane(self, lane_identity: str) -> Any:
        """Return one lane with a bounded slice of its history."""

    def list_comparisons(self, lane_identity: str) -> Any:
        """Return the lane's recorded promotion decisions."""


class EvidenceReadPort(Protocol):
    """Narrow framework-neutral behavior consumed by every HTTP route."""

    def probe_evidence_readiness(self) -> EvidenceReadiness: ...

    def get_run(self, run_id: str) -> RunReadModel: ...

    def list_runs(
        self,
        query: RunQuery | None = None,
        page: RunPageRequest | None = None,
    ) -> Page[RunReadModel, RunCursor]: ...

    def get_artifact(self, artifact_id: str) -> ArtifactView: ...

    def list_run_artifacts(
        self,
        run_id: str,
        page: ArtifactPageRequest | None = None,
        *,
        query: ArtifactQuery | None = None,
    ) -> Page[RunArtifactView, ArtifactCursor]: ...

    def read_verified_manifest(
        self,
        artifact_id: str,
        expected_media_type: str,
        max_bytes: int,
    ) -> bytes: ...


def _request_id(request: Request) -> str:
    state = request.scope.get("state", {})
    if type(state) is dict:
        value = state.get("signalattice_request_id")
        if type(value) is str and len(value) == 32 and value.isascii():
            return value
    return "0" * 32


def _problem_response(error: Exception, request: Request) -> Response:
    spec = problem_for_exception(error)
    document = build_problem(spec, _request_id(request))
    headers = {}
    if spec.retry_after_seconds is not None:
        headers["Retry-After"] = str(spec.retry_after_seconds)
    return Response(
        content=document.canonical_json_bytes(maximum_bytes=4_096),
        status_code=spec.status,
        media_type="application/problem+json",
        headers=headers,
    )


def _cursor_token(cursor: RunCursor | ArtifactCursor | None) -> str | None:
    return None if cursor is None else cursor.token


def _artifact_page(
    ports: EvidenceReadPort,
    *,
    run_id: str,
    page_size: int,
    cursor: str | None,
    role: str | None = None,
    maximum_page_size: int = 100,
) -> Page[RunArtifactView, ArtifactCursor]:
    if type(maximum_page_size) is not int or not 1 <= maximum_page_size <= 100:
        raise ValueError("maximum_page_size is outside the service page contract")
    page = ArtifactPageRequest(
        page_size=min(page_size, maximum_page_size),
        cursor=None if cursor is None else ArtifactCursor(cursor),
    )
    return ports.list_run_artifacts(
        run_id,
        page,
        query=ArtifactQuery(role=role),
    )


def _manifest_for_link(
    ports: EvidenceReadPort,
    link: RunArtifactView,
    *,
    kind: ManifestKind,
    run_id: str,
    max_bytes: int,
) -> ForecastSummaryManifest | DiagnosticsManifest | ModelCardManifest:
    payload = ports.read_verified_manifest(
        link.artifact.artifact_id,
        _MANIFEST_MEDIA_TYPE,
        max_bytes,
    )
    return parse_evidence_manifest(payload, expected_kind=kind, expected_run_id=run_id)


def _install_openapi_contract(app: FastAPI) -> None:
    """Replace framework-default errors with the exact public wire contract."""

    def custom_openapi() -> dict[str, object]:
        if app.openapi_schema is not None:
            return cast(dict[str, object], app.openapi_schema)
        document = cast(
            dict[str, object],
            get_openapi(
                title=app.title,
                version=app.version,
                summary=app.summary,
                description=app.description,
                routes=app.routes,
                openapi_version=app.openapi_version,
                servers=[],
            ),
        )
        components = cast(dict[str, object], document.setdefault("components", {}))
        schemas = cast(dict[str, object], components.setdefault("schemas", {}))
        schemas["ProblemDocument"] = ProblemDocument.model_json_schema(
            ref_template="#/components/schemas/{model}"
        )
        schemas[_OPENAPI_DOCUMENT_SCHEMA_NAME] = dict(_OPENAPI_DOCUMENT_SCHEMA)
        paths = cast(dict[str, object], document.get("paths", {}))
        for path, raw_path_item in paths.items():
            path_item = cast(dict[str, object], raw_path_item)
            operation = cast(dict[str, object], path_item["get"])
            responses = cast(dict[str, object], operation.setdefault("responses", {}))
            statuses = set(_COMMON_PROBLEM_STATUSES)
            if path in _VALIDATED_PATHS:
                statuses.add(422)
            if path in _NOT_FOUND_PATHS:
                statuses.add(404)
            if path in _STORAGE_PATHS:
                statuses.add(503)
            for status in sorted(statuses):
                key = str(status)
                problem_content = {
                    "application/problem+json": {
                        "schema": {"$ref": "#/components/schemas/ProblemDocument"}
                    }
                }
                if path == "/health/ready" and status == 503 and key in responses:
                    response = cast(dict[str, object], responses[key])
                    content = cast(dict[str, object], response.setdefault("content", {}))
                    content.update(problem_content)
                    continue
                responses[key] = {
                    "description": _PROBLEM_DESCRIPTIONS[status],
                    "content": problem_content,
                }
        schemas.pop("HTTPValidationError", None)
        schemas.pop("ValidationError", None)
        app.openapi_schema = document
        return document

    app.openapi = custom_openapi  # type: ignore[method-assign]


def create_app(
    ports: EvidenceReadPort,
    *,
    max_concurrency: int = 32,
    max_data_concurrency: int | None = None,
    allowed_port: int | None = None,
    admission: AdmissionController | None = None,
    telemetry: ServiceTelemetry | None = None,
    governance: GovernanceReadPort | None = None,
    console: ConsoleBundle | None = None,
) -> FastAPI:
    """Create a read-only app over already-initialized injected storage ports.

    Construction never creates, initializes, repairs, migrates, or mutates
    registry/CAS state.  The returned app is wrapped by a raw ASGI security
    boundary and intentionally has no interactive documentation endpoints.
    """

    if not isinstance(ports, RegistryReadPorts) and not all(
        callable(getattr(ports, name, None))
        for name in (
            "probe_evidence_readiness",
            "get_run",
            "list_runs",
            "get_artifact",
            "list_run_artifacts",
            "read_verified_manifest",
        )
    ):
        raise TypeError("ports must implement the bounded evidence read contract")
    resolved_data_concurrency = (
        min(24, max_concurrency) if max_data_concurrency is None else max_data_concurrency
    )
    resolved_admission = (
        AdmissionController(
            AdmissionLimits(
                global_concurrency=max_concurrency,
                data_concurrency=resolved_data_concurrency,
            )
        )
        if admission is None
        else admission
    )
    if type(resolved_admission) is not AdmissionController:
        raise TypeError("admission must be an AdmissionController")
    resolved_telemetry = ServiceTelemetry(ServiceMetrics()) if telemetry is None else telemetry
    if type(resolved_telemetry) is not ServiceTelemetry:
        raise TypeError("telemetry must be a ServiceTelemetry")

    @asynccontextmanager
    async def lifespan(_app: FastAPI) -> AsyncIterator[None]:
        await resolved_telemetry.start()
        try:
            yield
        finally:
            resolved_admission.begin_shutdown()
            resolved_telemetry.begin_draining()
            await resolved_telemetry.stop()

    app = FastAPI(
        title="Signalattice Read-Only Evidence API",
        summary="Local aggregate forecast and governance evidence",
        version="1.0.0",
        debug=False,
        redirect_slashes=False,
        strict_content_type=True,
        docs_url=None,
        redoc_url=None,
        swagger_ui_oauth2_redirect_url=None,
        openapi_url=None,
        servers=[],
        separate_input_output_schemas=False,
        lifespan=lifespan,
    )

    @app.exception_handler(RequestValidationError)
    async def request_validation_handler(
        request: Request,
        _error: RequestValidationError,
    ) -> Response:
        document = build_problem(VALIDATION_FAILED, _request_id(request))
        return Response(
            content=document.canonical_json_bytes(maximum_bytes=4_096),
            status_code=VALIDATION_FAILED.status,
            media_type="application/problem+json",
        )

    @app.exception_handler(StarletteHTTPException)
    async def http_error_handler(request: Request, error: StarletteHTTPException) -> Response:
        if error.status_code == 404:
            spec = NOT_FOUND
        elif error.status_code == 405:
            spec = METHOD_NOT_ALLOWED
        else:
            spec = BAD_REQUEST
        document = build_problem(spec, _request_id(request))
        return Response(
            content=document.canonical_json_bytes(maximum_bytes=4_096),
            status_code=spec.status,
            media_type="application/problem+json",
        )

    @app.exception_handler(RegistryError)
    async def registry_error_handler(request: Request, error: RegistryError) -> Response:
        return _problem_response(error, request)

    @app.exception_handler(ManifestValidationError)
    async def manifest_error_handler(
        request: Request,
        error: ManifestValidationError,
    ) -> Response:
        return _problem_response(error, request)

    @app.exception_handler(ValueError)
    async def value_error_handler(request: Request, error: ValueError) -> Response:
        return _problem_response(error, request)

    @app.get(
        "/health/live",
        response_model=LiveResponse,
        operation_id="getLiveness",
        tags=["health"],
    )
    def liveness() -> LiveResponse:
        return LiveResponse()

    @app.get(
        "/health/ready",
        response_model=ReadyResponse,
        operation_id="getReadiness",
        tags=["health"],
        responses={503: {"model": ReadyResponse, "description": "Storage is not ready."}},
    )
    def readiness() -> ReadyResponse | Response:
        result = ports.probe_evidence_readiness()
        response = ReadyResponse(
            ready=result.ready,
            code=result.code,
            registry_schema_version=result.schema_version,
            journal_mode=result.journal_mode,
            retryable=result.retryable,
        )
        if result.ready:
            return response
        headers = {"Retry-After": "1"} if result.retryable else None
        return Response(
            content=response.canonical_json_bytes(maximum_bytes=4_096),
            status_code=503,
            media_type="application/json",
            headers=headers,
        )

    @app.get(
        "/internal/metrics",
        operation_id="getInternalMetrics",
        include_in_schema=False,
        response_class=Response,
    )
    def internal_metrics() -> Response:
        snapshot = resolved_telemetry.metrics.snapshot()
        return Response(
            content=snapshot.body,
            headers={"content-type": snapshot.content_type},
        )

    @app.get(
        "/api/v1/runs",
        response_model=RunPageResponse,
        operation_id="listRunsV1",
        tags=["runs"],
    )
    def list_runs(
        page_size: PageSize = 25,
        cursor: CursorQuery = None,
        status: Annotated[RunStatus | None, Query()] = None,
        provenance: Annotated[RunProvenance | None, Query()] = None,
    ) -> RunPageResponse:
        page = ports.list_runs(
            RunQuery(status=status, provenance=provenance),
            RunPageRequest(
                page_size=page_size,
                cursor=None if cursor is None else RunCursor(cursor),
            ),
        )
        return RunPageResponse(
            items=tuple(RunResponse.from_read_model(item) for item in page.items),
            next_cursor=_cursor_token(page.next_cursor),
        )

    @app.get(
        "/api/v1/runs/{run_id}",
        response_model=RunResponse,
        operation_id="getRunV1",
        tags=["runs"],
    )
    def get_run(run_id: RunReferencePath) -> RunResponse:
        return RunResponse.from_read_model(ports.get_run(decode_run_reference(run_id)))

    @app.get(
        "/api/v1/runs/{run_id}/forecast-summaries",
        response_model=ForecastSummaryPageResponse,
        operation_id="listAggregateForecastSummariesV1",
        tags=["evidence"],
    )
    def list_forecast_summaries(
        run_id: RunReferencePath,
        page_size: PageSize = 25,
        cursor: CursorQuery = None,
    ) -> ForecastSummaryPageResponse:
        internal_run_id = decode_run_reference(run_id)
        page = _artifact_page(
            ports,
            run_id=internal_run_id,
            page_size=page_size,
            cursor=cursor,
            role=ManifestKind.FORECAST_SUMMARY.value,
            maximum_page_size=FORECAST_RESPONSE_PAGE_LIMIT,
        )
        manifests = tuple(
            cast(
                ForecastSummaryManifest,
                _manifest_for_link(
                    ports,
                    link,
                    kind=ManifestKind.FORECAST_SUMMARY,
                    run_id=internal_run_id,
                    max_bytes=_FORECAST_MANIFEST_BYTES,
                ),
            )
            for link in page.items
        )
        return ForecastSummaryPageResponse(
            items=tuple(ForecastSummaryResponse.from_manifest(item) for item in manifests),
            next_cursor=_cursor_token(page.next_cursor),
        )

    @app.get(
        "/api/v1/runs/{run_id}/diagnostics",
        response_model=DiagnosticsPageResponse,
        operation_id="listDiagnosticsV1",
        tags=["evidence"],
    )
    def list_diagnostics(
        run_id: RunReferencePath,
        page_size: PageSize = 25,
        cursor: CursorQuery = None,
    ) -> DiagnosticsPageResponse:
        internal_run_id = decode_run_reference(run_id)
        page = _artifact_page(
            ports,
            run_id=internal_run_id,
            page_size=page_size,
            cursor=cursor,
            role=ManifestKind.DIAGNOSTICS.value,
            maximum_page_size=DIAGNOSTICS_RESPONSE_PAGE_LIMIT,
        )
        manifests = tuple(
            cast(
                DiagnosticsManifest,
                _manifest_for_link(
                    ports,
                    link,
                    kind=ManifestKind.DIAGNOSTICS,
                    run_id=internal_run_id,
                    max_bytes=_DIAGNOSTICS_MANIFEST_BYTES,
                ),
            )
            for link in page.items
        )
        return DiagnosticsPageResponse(
            items=tuple(DiagnosticsResponse.from_manifest(item) for item in manifests),
            next_cursor=_cursor_token(page.next_cursor),
        )

    @app.get(
        "/api/v1/runs/{run_id}/artifacts",
        response_model=ArtifactPageResponse,
        operation_id="listRunArtifactsV1",
        tags=["artifacts"],
    )
    def list_artifacts(
        run_id: RunReferencePath,
        page_size: PageSize = 25,
        cursor: CursorQuery = None,
    ) -> ArtifactPageResponse:
        page = _artifact_page(
            ports,
            run_id=decode_run_reference(run_id),
            page_size=page_size,
            cursor=cursor,
        )
        return ArtifactPageResponse(
            items=tuple(RunArtifactResponse.from_view(item) for item in page.items),
            next_cursor=_cursor_token(page.next_cursor),
        )

    @app.get(
        "/api/v1/artifacts/{artifact_id}",
        response_model=ArtifactResponse,
        operation_id="getArtifactMetadataV1",
        tags=["artifacts"],
    )
    def get_artifact(artifact_id: DigestPath) -> ArtifactResponse:
        return ArtifactResponse.from_view(ports.get_artifact(artifact_id))

    @app.get(
        "/api/v1/model-cards",
        response_model=ModelCardPageResponse,
        operation_id="listModelCardsV1",
        tags=["model-cards"],
    )
    def list_model_cards(
        run_id: RunReferenceQuery,
        page_size: PageSize = 25,
        cursor: CursorQuery = None,
    ) -> ModelCardPageResponse:
        internal_run_id = decode_run_reference(run_id)
        page = _artifact_page(
            ports,
            run_id=internal_run_id,
            page_size=page_size,
            cursor=cursor,
            role=ManifestKind.MODEL_CARD.value,
        )
        summaries: list[ModelCardSummary] = []
        for link in page.items:
            manifest = cast(
                ModelCardManifest,
                _manifest_for_link(
                    ports,
                    link,
                    kind=ManifestKind.MODEL_CARD,
                    run_id=internal_run_id,
                    max_bytes=_MODEL_CARD_MANIFEST_BYTES,
                ),
            )
            summaries.append(
                ModelCardSummary(
                    artifact_id=link.artifact.artifact_id,
                    card_id=manifest.card_id,
                    model_name=manifest.model_name,
                    model_version=manifest.model_version,
                    generated_at=manifest.generated_at,
                    limitation_count=len(manifest.limitations),
                )
            )
        return ModelCardPageResponse(
            items=tuple(summaries),
            next_cursor=_cursor_token(page.next_cursor),
        )

    @app.get(
        "/api/v1/model-cards/{card_id}",
        response_model=ModelCardResponse,
        operation_id="getModelCardV1",
        tags=["model-cards"],
    )
    def get_model_card(
        card_id: CardIdentifierPath,
        run_id: RunReferenceQuery,
    ) -> ModelCardResponse:
        internal_run_id = decode_run_reference(run_id)
        page = _artifact_page(
            ports,
            run_id=internal_run_id,
            page_size=100,
            cursor=None,
            role=ManifestKind.MODEL_CARD.value,
        )
        matches: list[tuple[str, ModelCardManifest]] = []
        for link in page.items:
            manifest = cast(
                ModelCardManifest,
                _manifest_for_link(
                    ports,
                    link,
                    kind=ManifestKind.MODEL_CARD,
                    run_id=internal_run_id,
                    max_bytes=_MODEL_CARD_MANIFEST_BYTES,
                ),
            )
            if manifest.card_id == card_id:
                matches.append((link.artifact.artifact_id, manifest))
        if page.next_cursor is not None:
            raise CapacityError("model-card lookup exceeded its bounded scan")
        if len(matches) > 1:
            raise IntegrityError("model-card identifier is ambiguous within its run")
        if matches:
            artifact_id, manifest = matches[0]
            return ModelCardResponse(
                artifact_id=artifact_id,
                model_card=ModelCardManifestResponse.from_manifest(manifest),
            )
        raise NotFoundError("model card does not exist")

    @app.get(
        "/api/v1/openapi.json",
        operation_id="getOpenApiV1",
        include_in_schema=True,
        tags=["contract"],
        response_class=Response,
        responses={
            200: {
                "description": "The deterministic OpenAPI 3.1 service contract.",
                "content": {
                    "application/json": {
                        "schema": {"$ref": _OPENAPI_DOCUMENT_SCHEMA_REF},
                    }
                },
            }
        },
    )
    def get_openapi() -> Response:
        return Response(
            canonical_openapi_bytes(cast(dict[str, object], app.openapi())),
            media_type="application/json",
        )

    if governance is not None:
        governance_ports = governance

        @app.get(
            "/api/v1/governance/lanes",
            response_model=LanePageResponse,
            operation_id="listGovernanceLanesV1",
            tags=["governance"],
        )
        def list_governance_lanes(
            page_size: PageSize = 25,
            cursor: LaneCursorQuery = None,
        ) -> LanePageResponse:
            bounded = min(page_size, LANE_RESPONSE_PAGE_LIMIT)
            return LanePageResponse.from_projection(
                governance_ports.list_lanes(page_size=bounded, cursor=cursor)
            )

        @app.get(
            "/api/v1/governance/lanes/{lane_id}",
            response_model=LaneDetailResponse,
            operation_id="getGovernanceLaneV1",
            tags=["governance"],
        )
        def get_governance_lane(lane_id: DigestPath) -> LaneDetailResponse:
            return LaneDetailResponse.from_projection(governance_ports.get_lane(lane_id))

        @app.get(
            "/api/v1/governance/lanes/{lane_id}/comparisons",
            response_model=ComparisonPageResponse,
            operation_id="listGovernanceComparisonsV1",
            tags=["governance"],
        )
        def list_governance_comparisons(lane_id: DigestPath) -> ComparisonPageResponse:
            return ComparisonPageResponse.from_projections(
                governance_ports.list_comparisons(lane_id)
            )

    _install_openapi_contract(app)
    app.add_middleware(
        SecurityBoundaryMiddleware,
        max_concurrency=max_concurrency,
        max_data_concurrency=resolved_data_concurrency,
        allowed_port=allowed_port,
        admission=resolved_admission,
        telemetry=resolved_telemetry,
        console=console,
    )
    return app


def assert_read_only_route_inventory(app: FastAPI) -> None:
    """Fail if the application exposes anything outside the approved GET inventory."""

    expected = {
        "/health/live",
        "/health/ready",
        "/internal/metrics",
        "/api/v1/runs",
        "/api/v1/runs/{run_id}",
        "/api/v1/runs/{run_id}/forecast-summaries",
        "/api/v1/runs/{run_id}/diagnostics",
        "/api/v1/runs/{run_id}/artifacts",
        "/api/v1/artifacts/{artifact_id}",
        "/api/v1/model-cards",
        "/api/v1/model-cards/{card_id}",
        "/api/v1/openapi.json",
    }
    # Governance reads are mounted only when a governance port is injected, so
    # the inventory admits them conditionally rather than demanding them.
    optional = {
        "/api/v1/governance/lanes",
        "/api/v1/governance/lanes/{lane_id}",
        "/api/v1/governance/lanes/{lane_id}/comparisons",
    }
    observed: set[str] = set()
    operation_ids: set[str] = set()
    for route in app.routes:
        path = getattr(route, "path", None)
        methods = getattr(route, "methods", None)
        if type(path) is not str or not isinstance(methods, set):
            continue
        if methods != {"GET"}:
            raise RuntimeError("service route inventory contains a non-GET operation")
        observed.add(path)
        operation_id = getattr(route, "operation_id", None)
        if type(operation_id) is not str or not operation_id or operation_id in operation_ids:
            raise RuntimeError("service route inventory has a missing or duplicate operation ID")
        operation_ids.add(operation_id)
    unexpected = observed - expected - optional
    if unexpected or not expected <= observed:
        raise RuntimeError("service route inventory differs from the approved version-1 contract")
