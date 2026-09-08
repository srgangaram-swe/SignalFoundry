"""Versioned research and optional paper HTTP routes; no domain math."""

from __future__ import annotations

from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager
from typing import Annotated, Any

from fastapi import FastAPI, Header, Request
from fastapi.exceptions import RequestValidationError
from pydantic import ValidationError
from starlette.concurrency import run_in_threadpool
from starlette.exceptions import HTTPException
from starlette.responses import JSONResponse

from signal_foundry.boundary import FoundryError, decode
from signal_foundry.contracts import (
    AuditTrail,
    Catalog,
    Comparison,
    Job,
    JobPage,
    Problem,
    ResearchEvidence,
    ResearchRequest,
    Validation,
)
from signal_foundry.http_security import LocalBoundary
from signal_foundry.manager import Manager
from signal_foundry.nexus import NexusBundle, mount
from signal_foundry.trading.models import PaperStatus
from signal_foundry.trading.service import (
    Action,
    PaperResult,
    PaperService,
    StopRequest,
    unavailable,
)

BODY_SCHEMA = {
    "requestBody": {
        "required": True,
        "content": {
            "application/json": {
                "schema": {"$ref": "#/components/schemas/ResearchRequest"}
            }
        },
    }
}
ERROR_RESPONSES: dict[int | str, dict[str, Any]] = {
    status: {"model": Problem}
    for status in (400, 403, 404, 409, 413, 415, 422, 429, 500, 502, 503, 504, 507)
}


async def parsed(request: Request) -> ResearchRequest:
    try:
        return ResearchRequest.model_validate_json(await request.body())
    except ValidationError as exc:
        raise FoundryError(
            "invalid_request",
            "Research configuration violates its versioned schema; inspect field"
            " bounds.",
        ) from exc


def create_app(
    factory: Callable[[], Manager],
    *,
    port: int = 8765,
    nexus: NexusBundle | None = None,
    paper_factory: Callable[[], PaperService] | None = None,
) -> FastAPI:
    """Create without IO; only the lifespan acquires state and worker ownership."""
    if not 1024 <= port <= 65535:
        raise FoundryError("port_policy", "Choose an unprivileged local port.")

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        manager = await run_in_threadpool(factory)
        app.state.manager = manager
        try:
            app.state.paper = (
                await run_in_threadpool(paper_factory) if paper_factory else None
            )
            yield
        finally:
            await run_in_threadpool(manager.close)

    app = FastAPI(
        title="Signal Foundry research control plane",
        version="1.0.0",
        description=(
            "Local research and explicit Alpaca paper operations."
            " Live-order capability is absent."
        ),
        lifespan=lifespan,
        docs_url=None,
        redoc_url=None,
        openapi_url="/api/v1/openapi.json",
        responses=ERROR_RESPONSES,
    )
    app.add_middleware(LocalBoundary, port=port)

    @app.exception_handler(FoundryError)
    async def domain_error(request: Request, exc: FoundryError) -> JSONResponse:
        return JSONResponse(
            Problem(code=exc.code, detail=exc.detail).model_dump(),
            status_code=exc.status,
        )

    @app.exception_handler(RequestValidationError)
    async def schema_error(
        request: Request, exc: RequestValidationError
    ) -> JSONResponse:
        return JSONResponse(
            Problem(
                code="invalid_request",
                detail="Request parameters violate the versioned schema.",
            ).model_dump(),
            status_code=422,
        )

    @app.exception_handler(HTTPException)
    async def route_error(request: Request, exc: HTTPException) -> JSONResponse:
        return JSONResponse(
            Problem(
                code="route_unavailable",
                detail="No supported operation matches this request.",
            ).model_dump(),
            status_code=exc.status_code,
        )

    def owner(request: Request) -> Manager:
        manager: Manager = request.app.state.manager
        return manager

    @app.get("/api/v1/paper", response_model=PaperStatus, operation_id="paper_status")
    def paper_status(request: Request) -> PaperStatus:
        service: PaperService | None = request.app.state.paper
        return service.status() if service else unavailable()

    @app.post("/api/v1/paper", response_model=PaperResult, operation_id="paper_action")
    def paper_action(request: Request, action: Action) -> PaperResult:
        service: PaperService | None = request.app.state.paper
        if service is None:
            raise FoundryError(
                "paper_unavailable",
                "Configure the local paper service explicitly before operating it.",
                409,
            )
        return service.action(action)

    @app.post(
        "/api/v1/paper/stop", response_model=PaperResult, operation_id="paper_stop"
    )
    def paper_stop(request: Request, body: StopRequest) -> PaperResult:
        """Persist stop through the same service, using reserved HTTP admission."""
        return paper_action(request, Action(operation="stop"))

    @app.get("/api/v1/catalog", response_model=Catalog, operation_id="catalog")
    def catalog(request: Request) -> Catalog:
        return owner(request).runner.catalog()

    @app.post(
        "/api/v1/validate",
        response_model=Validation,
        operation_id="validate",
        openapi_extra=BODY_SCHEMA,
    )
    async def validate(request: Request) -> Validation:
        return await run_in_threadpool(
            owner(request).runner.validate, await parsed(request)
        )

    @app.post(
        "/api/v1/jobs",
        response_model=Job,
        status_code=202,
        operation_id="submit",
        openapi_extra=BODY_SCHEMA,
    )
    async def submit(
        request: Request,
        idempotency_key: Annotated[str, Header(min_length=16, max_length=128)],
    ) -> Job:
        return await run_in_threadpool(
            owner(request).submit, await parsed(request), idempotency_key
        )

    @app.get("/api/v1/jobs", response_model=JobPage, operation_id="jobs")
    def jobs(request: Request) -> JobPage:
        return owner(request).store.list()

    @app.get("/api/v1/jobs/{job_id}", response_model=Job, operation_id="status")
    def status(request: Request, job_id: str) -> Job:
        return owner(request).store.get(job_id)

    @app.post(
        "/api/v1/jobs/{job_id}/cancel",
        response_model=Job,
        operation_id="cancel",
        openapi_extra={
            "requestBody": {
                "required": True,
                "content": {
                    "application/json": {
                        "schema": {
                            "type": "object",
                            "additionalProperties": False,
                            "properties": {},
                        }
                    }
                },
            }
        },
    )
    async def cancel(request: Request, job_id: str) -> Job:
        if decode(await request.body()) != {}:
            raise FoundryError(
                "invalid_request", "Cancellation accepts only an empty JSON object."
            )
        return await run_in_threadpool(owner(request).cancel, job_id)

    @app.get(
        "/api/v1/jobs/{job_id}/evidence",
        response_model=ResearchEvidence,
        operation_id="evidence",
    )
    def evidence(request: Request, job_id: str) -> ResearchEvidence:
        return owner(request).store.evidence(job_id)

    @app.get(
        "/api/v1/jobs/{job_id}/audit", response_model=AuditTrail, operation_id="audit"
    )
    def audit(request: Request, job_id: str) -> AuditTrail:
        return owner(request).store.audit(job_id)

    @app.get(
        "/api/v1/compare/{left}/{right}",
        response_model=Comparison,
        operation_id="compare",
    )
    def compare(request: Request, left: str, right: str) -> Comparison:
        return owner(request).compare(left, right)

    if nexus is not None:
        mount(app, nexus)
    return app
