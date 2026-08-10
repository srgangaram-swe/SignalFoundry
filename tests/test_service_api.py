"""End-to-end and adversarial tests for the local read-only evidence API.

These tests deliberately exercise the raw ASGI boundary through HTTPX rather
than calling route functions.  Valid evidence crosses a real initialized
registry and content-addressed store; hostile inputs must fail before they can
select storage state or leak an internal exception, pathname, or artifact byte.
"""

from __future__ import annotations

import asyncio
import re
import sqlite3
import threading
from collections.abc import Awaitable, Callable
from contextlib import closing
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import httpx
import pytest
from asgi_lifespan import LifespanManager
from fastapi import FastAPI

from quant_platform.service.api import assert_read_only_route_inventory, create_app
from quant_platform.service.manifests import (
    DiagnosticCategory,
    DiagnosticsManifest,
    DiagnosticStatus,
    DiagnosticValue,
    ForecastAggregate,
    ForecastSplit,
    ForecastSummaryManifest,
    HorizonUnit,
    ModelCardManifest,
    ModelCardSection,
    ModelCardSectionName,
)
from quant_platform.service.models import encode_run_reference
from quant_platform.tracking.cas import ArtifactStore, PublishedArtifact
from quant_platform.tracking.contracts import (
    ArtifactClass,
    ArtifactCursor,
    ArtifactLink,
    EvidenceClass,
    Page,
    RegistryLimits,
    RunCursor,
    SubmissionRequest,
    TerminalRunRequest,
)
from quant_platform.tracking.read_ports import (
    ArtifactPageRequest,
    ArtifactQuery,
    ArtifactView,
    EvidenceReadiness,
    EvidenceReadinessCode,
    RegistryReadPorts,
    RunArtifactView,
    RunPageRequest,
    RunQuery,
    RunReadModel,
)
from quant_platform.tracking.registry import RunRegistry

NOW = datetime(2026, 8, 9, 12, 0, tzinfo=UTC)
SECRET = b"service-api-test-secret-material-32-bytes"
MANIFEST_MEDIA_TYPE = "application/vnd.signalattice.manifest+json"
RAW_SENTINEL = b"DO-NOT-EXPOSE-RAW-ARTIFACT-BYTES"
_REQUEST_ID = re.compile(r"^[0-9a-f]{32}$")
_SECURITY_HEADERS = {
    "cache-control": "no-store",
    "content-security-policy": (
        "default-src 'none'; base-uri 'none'; form-action 'none'; frame-ancestors 'none'"
    ),
    "cross-origin-resource-policy": "same-origin",
    "permissions-policy": "accelerometer=(), camera=(), geolocation=(), microphone=()",
    "referrer-policy": "no-referrer",
    "x-content-type-options": "nosniff",
    "x-frame-options": "DENY",
}


class _MutableClock:
    """Deterministic registry authority controlled only by fixture setup."""

    def __init__(self, current: datetime) -> None:
        self.current = current

    def now(self) -> datetime:
        return self.current


@dataclass(frozen=True, slots=True)
class _ServiceSystem:
    app: FastAPI
    registry: RunRegistry
    store: ArtifactStore
    ports: RegistryReadPorts
    root: Path
    run_id: str
    run_reference: str
    forecast_ids: tuple[str, ...]
    diagnostic_id: str
    model_card_id: str
    raw_artifact_id: str


async def _with_client[ResultT](
    app: FastAPI,
    operation: Callable[[httpx.AsyncClient], Awaitable[ResultT]],
) -> ResultT:
    """Run one bounded scenario inside the application's real lifespan."""

    async with LifespanManager(app):
        transport = httpx.ASGITransport(app=app, raise_app_exceptions=False)
        async with httpx.AsyncClient(
            transport=transport,
            base_url="http://127.0.0.1",
            timeout=2.0,
        ) as client:
            return await operation(client)


def _run_scenario[ResultT](
    app: FastAPI,
    operation: Callable[[httpx.AsyncClient], Awaitable[ResultT]],
) -> ResultT:
    return asyncio.run(_with_client(app, operation))


async def _single_request(
    client: httpx.AsyncClient,
    method: str,
    url: str,
    **kwargs: Any,
) -> httpx.Response:
    return await client.request(method, url, **kwargs)


def _request(
    app: FastAPI,
    method: str,
    url: str,
    **kwargs: Any,
) -> httpx.Response:
    async def exchange(client: httpx.AsyncClient) -> httpx.Response:
        return await _single_request(client, method, url, **kwargs)

    return _run_scenario(app, exchange)


def _assert_security_headers(response: httpx.Response) -> str:
    for name, expected in _SECURITY_HEADERS.items():
        assert response.headers[name] == expected
    request_id = response.headers["x-request-id"]
    assert _REQUEST_ID.fullmatch(request_id)
    assert "access-control-allow-origin" not in response.headers
    assert "access-control-allow-credentials" not in response.headers
    return request_id


def _assert_problem(response: httpx.Response, status: int, code: str) -> dict[str, object]:
    assert response.status_code == status
    assert response.headers["content-type"] == "application/problem+json"
    request_id = _assert_security_headers(response)
    document = response.json()
    assert document == {
        "code": code,
        "detail": document["detail"],
        "request_id": request_id,
        "status": status,
        "title": document["title"],
        "type": f"urn:signalattice:problem:{code}",
    }
    assert isinstance(document["detail"], str)
    assert isinstance(document["title"], str)
    return document


def _forecast_manifest(run_id: str, aggregate_id: str) -> ForecastSummaryManifest:
    return ForecastSummaryManifest(
        run_id=run_id,
        generated_at=NOW,
        limitations=("Historical aggregate fixture; no prospective trading claim.",),
        aggregates=(
            ForecastAggregate(
                aggregate_id=aggregate_id,
                target="return-1d",
                split=ForecastSplit.TEST,
                horizon_steps=1,
                horizon_unit=HorizonUnit.BUSINESS_DAYS,
                window_start=NOW - timedelta(days=90),
                window_end=NOW - timedelta(days=1),
                sample_count=512,
                mean_prediction=0.001,
                mean_observation=0.0008,
                mean_error=0.0002,
                mean_absolute_error=0.012,
                root_mean_squared_error=0.018,
                interval_coverage=0.81,
                mean_interval_width=0.043,
            ),
        ),
    )


def _diagnostics_manifest(run_id: str) -> DiagnosticsManifest:
    return DiagnosticsManifest(
        run_id=run_id,
        generated_at=NOW,
        limitations=("Synthetic API integration fixture only.",),
        values=(
            DiagnosticValue(
                code="ready-for-live",
                category=DiagnosticCategory.READINESS,
                value=False,
                status=DiagnosticStatus.FAIL,
                detail="No prospective paper-trading evidence or capital authorization.",
            ),
        ),
    )


def _maximum_shape_diagnostics_manifest(
    run_id: str,
    prefix: str,
) -> DiagnosticsManifest:
    """Return a legal near-maximum diagnostic payload for response-budget tests."""

    return DiagnosticsManifest(
        run_id=run_id,
        generated_at=NOW,
        limitations=tuple(f"{prefix}-limitation-{index}-" + "l" * 2_000 for index in range(16)),
        values=tuple(
            DiagnosticValue(
                code=f"{prefix}-{index:03d}",
                category=DiagnosticCategory.CALIBRATION,
                value="v" * 256,
                unit="unit" * 8,
                status=DiagnosticStatus.INFORMATIONAL,
                detail="d" * 1_024,
            )
            for index in range(256)
        ),
    )


def _model_card_sections() -> tuple[ModelCardSection, ...]:
    text = {
        ModelCardSectionName.OVERVIEW: "Aggregate research-model evidence.",
        ModelCardSectionName.INTENDED_USE: "Bounded historical research.",
        ModelCardSectionName.OUT_OF_SCOPE_USE: "Live trading and profit guarantees.",
        ModelCardSectionName.DATA: "Synthetic, versioned integration fixtures.",
        ModelCardSectionName.EVALUATION: "Walk-forward aggregate metrics.",
        ModelCardSectionName.LIMITATIONS: "No prospective trading evidence.",
        ModelCardSectionName.MONITORING: "Reassess drift before promotion.",
    }
    return tuple(
        ModelCardSection(
            name=name,
            title=name.value.replace("_", " ").title(),
            text=text[name],
        )
        for name in ModelCardSectionName
        if name in text
    )


def _model_card(run_id: str) -> ModelCardManifest:
    return ModelCardManifest(
        run_id=run_id,
        generated_at=NOW,
        limitations=("Historical fixture only.",),
        card_id="card-001",
        model_name="baseline-forecast",
        model_version="1.0.0",
        sections=_model_card_sections(),
    )


def _publish(
    system_root: Path,
    registry: RunRegistry,
    store: ArtifactStore,
    *,
    name: str,
    payload: bytes,
    artifact_class: ArtifactClass = ArtifactClass.METADATA,
    media_type: str = MANIFEST_MEDIA_TYPE,
) -> PublishedArtifact:
    source_root = system_root / "sources"
    source_root.mkdir(mode=0o700, exist_ok=True)
    source = source_root / name
    source.write_bytes(payload)
    published = store.publish(source)
    registry.register_artifact(
        published,
        artifact_class=artifact_class,
        media_type=media_type,
    )
    return published


def _complete_run(
    registry: RunRegistry,
    *,
    run_id: str,
    ordinal: int,
    links: tuple[ArtifactLink, ...],
) -> None:
    registry.submit(
        SubmissionRequest("forecast", {"fixture_ordinal": ordinal}),
        idempotency_key=f"service-api-fixture-{ordinal}",
    )
    claimed = registry.claim(worker_id=f"service-worker-{ordinal}", lease_seconds=60)
    assert claimed is not None
    registry.complete(
        claimed.lease,
        run=TerminalRunRequest(
            run_id=run_id,
            evidence_class=EvidenceClass.SIMULATED,
            started_at=NOW,
            source_commit="abcdef0123456789",
            data_identity="sha256:" + "a" * 64,
            limitation_summary="Synthetic API integration evidence only.",
        ),
        artifact_links=tuple(sorted(links)),
    )


@pytest.fixture
def service_system(tmp_path: Path) -> _ServiceSystem:
    root = (tmp_path / "service-system").resolve()
    root.mkdir(mode=0o700)
    store = ArtifactStore(root / "cas", max_artifact_bytes=1024 * 1024)
    store.initialize()
    registry = RunRegistry(
        root / "registry.sqlite",
        digest_secret=SECRET,
        limits=RegistryLimits(busy_timeout_ms=2_000),
        clock=_MutableClock(NOW),
        artifact_verifier=store,
    )
    registry.initialize()
    run_id = "run-api-001"
    forecast_manifests = (
        _forecast_manifest(run_id, "aggregate-001"),
        _forecast_manifest(run_id, "aggregate-002"),
    )
    forecasts = tuple(
        _publish(
            root,
            registry,
            store,
            name=f"forecast-{index}.json",
            payload=manifest.canonical_json_bytes(),
        )
        for index, manifest in enumerate(forecast_manifests)
    )
    diagnostics = _publish(
        root,
        registry,
        store,
        name="diagnostics.json",
        payload=_diagnostics_manifest(run_id).canonical_json_bytes(),
    )
    model_card = _publish(
        root,
        registry,
        store,
        name="model-card.json",
        payload=_model_card(run_id).canonical_json_bytes(),
    )
    raw = _publish(
        root,
        registry,
        store,
        name="private-raw.bin",
        payload=RAW_SENTINEL,
        artifact_class=ArtifactClass.OUTPUT,
        media_type="application/octet-stream",
    )
    _complete_run(
        registry,
        run_id=run_id,
        ordinal=1,
        links=(
            *(ArtifactLink("forecast_summary", artifact.digest) for artifact in forecasts),
            ArtifactLink("diagnostics", diagnostics.digest),
            ArtifactLink("model_card", model_card.digest),
            ArtifactLink("private_blob", raw.digest),
        ),
    )
    ports = RegistryReadPorts(registry, store)
    return _ServiceSystem(
        app=create_app(ports),
        registry=registry,
        store=store,
        ports=ports,
        root=root,
        run_id=run_id,
        run_reference=encode_run_reference(run_id),
        forecast_ids=tuple(artifact.digest for artifact in forecasts),
        diagnostic_id=diagnostics.digest,
        model_card_id=model_card.digest,
        raw_artifact_id=raw.digest,
    )


def test_every_approved_route_crosses_real_registry_and_cas_without_raw_leakage(
    service_system: _ServiceSystem,
) -> None:
    async def scenario(client: httpx.AsyncClient) -> dict[str, httpx.Response]:
        reference = service_system.run_reference
        requests = {
            "live": "/health/live",
            "ready": "/health/ready",
            "runs": "/api/v1/runs?status=succeeded&provenance=registry%2Fverified",
            "run": f"/api/v1/runs/{reference}",
            "forecasts": f"/api/v1/runs/{reference}/forecast-summaries",
            "diagnostics": f"/api/v1/runs/{reference}/diagnostics",
            "artifacts": f"/api/v1/runs/{reference}/artifacts",
            "artifact": f"/api/v1/artifacts/{service_system.raw_artifact_id}",
            "cards": f"/api/v1/model-cards?run_id={reference}",
            "card": f"/api/v1/model-cards/card-001?run_id={reference}",
            "openapi": "/api/v1/openapi.json",
        }
        return {name: await client.get(url) for name, url in requests.items()}

    responses = _run_scenario(service_system.app, scenario)

    assert all(response.status_code == 200 for response in responses.values())
    for response in responses.values():
        _assert_security_headers(response)
    assert responses["live"].json() == {"schema_version": 1, "status": "live"}
    assert responses["ready"].json() == {
        "schema_version": 1,
        "ready": True,
        "code": "ready",
        "registry_schema_version": service_system.ports.probe_evidence_readiness().schema_version,
        "journal_mode": "wal",
        "retryable": False,
    }
    run_page = responses["runs"].json()
    assert [item["run_id"] for item in run_page["items"]] == [service_system.run_reference]
    run = responses["run"].json()
    assert run["run_id"] == service_system.run_reference
    assert run["provenance"] == "registry/verified"
    assert run["status"] == "succeeded"
    assert run["evidence_class"] == "simulated"
    assert set(run["links"]) == {"forecast_summaries", "diagnostics", "artifacts"}

    forecasts = responses["forecasts"].json()
    assert forecasts["evidence_granularity"] == "aggregate"
    assert forecasts["row_level_available"] is False
    assert {item["kind"] for item in forecasts["items"]} == {"forecast_summary"}
    assert {item["run_id"] for item in forecasts["items"]} == {service_system.run_reference}
    assert {
        aggregate["aggregate_id"] for item in forecasts["items"] for aggregate in item["aggregates"]
    } == {"aggregate-001", "aggregate-002"}
    diagnostics = responses["diagnostics"].json()
    assert diagnostics["items"][0]["run_id"] == service_system.run_reference
    assert diagnostics["items"][0]["values"][0]["code"] == "ready-for-live"
    assert diagnostics["items"][0]["values"][0]["value"] is False
    artifacts = responses["artifacts"].json()
    assert len(artifacts["items"]) == 5
    assert {item["role"] for item in artifacts["items"]} == {
        "diagnostics",
        "forecast_summary",
        "model_card",
        "private_blob",
    }
    assert responses["artifact"].json()["artifact_id"] == service_system.raw_artifact_id
    assert responses["cards"].json()["items"][0]["card_id"] == "card-001"
    assert responses["card"].json()["model_card"]["text_format"] == "plain_text"
    assert responses["card"].json()["model_card"]["run_id"] == service_system.run_reference

    serialized = b"\n".join(
        response.content for name, response in responses.items() if name != "openapi"
    )
    assert RAW_SENTINEL not in serialized
    assert str(service_system.root).encode() not in serialized
    for forbidden_key in (
        b'"storage_key"',
        b'"storage_relpath"',
        b'"path"',
        b'"rows"',
        b'"predictions"',
        b'"observations"',
    ):
        assert forbidden_key not in serialized


def test_maximum_shape_diagnostics_paginate_below_response_ceiling(
    service_system: _ServiceSystem,
) -> None:
    run_id = "maximum-diagnostics-run"
    manifests = tuple(
        _maximum_shape_diagnostics_manifest(run_id, prefix) for prefix in ("a", "b", "c")
    )
    published = tuple(
        _publish(
            service_system.root,
            service_system.registry,
            service_system.store,
            name=f"maximum-diagnostics-{index}.json",
            payload=manifest.canonical_json_bytes(),
        )
        for index, manifest in enumerate(manifests)
    )
    _complete_run(
        service_system.registry,
        run_id=run_id,
        ordinal=10,
        links=tuple(ArtifactLink("diagnostics", artifact.digest) for artifact in published),
    )
    reference = encode_run_reference(run_id)

    async def scenario(client: httpx.AsyncClient) -> tuple[httpx.Response, ...]:
        responses: list[httpx.Response] = []
        cursor: str | None = None
        for _ in range(3):
            response = await client.get(
                f"/api/v1/runs/{reference}/diagnostics",
                params={"page_size": 100, **({"cursor": cursor} if cursor else {})},
            )
            responses.append(response)
            cursor = response.json()["next_cursor"]
        return tuple(responses)

    responses = _run_scenario(service_system.app, scenario)

    assert [response.status_code for response in responses] == [200, 200, 200]
    assert all(len(response.content) < 1024 * 1024 for response in responses)
    assert [len(response.json()["items"]) for response in responses] == [1, 1, 1]
    assert responses[0].json()["next_cursor"] is not None
    assert responses[1].json()["next_cursor"] is not None
    assert responses[2].json()["next_cursor"] is None
    assert {response.json()["items"][0]["run_id"] for response in responses} == {reference}


def test_manifest_binding_supports_full_registry_identifier_and_projects_reference(
    service_system: _ServiceSystem,
) -> None:
    run_id = "experiment:2026/run-1"
    manifest = _forecast_manifest(run_id, "aggregate-full-identifier")
    published = _publish(
        service_system.root,
        service_system.registry,
        service_system.store,
        name="full-identifier-forecast.json",
        payload=manifest.canonical_json_bytes(),
    )
    _complete_run(
        service_system.registry,
        run_id=run_id,
        ordinal=11,
        links=(ArtifactLink("forecast_summary", published.digest),),
    )
    reference = encode_run_reference(run_id)

    response = _request(
        service_system.app,
        "GET",
        f"/api/v1/runs/{reference}/forecast-summaries",
    )

    assert response.status_code == 200
    assert response.json()["items"][0]["run_id"] == reference
    assert run_id.encode() not in response.content


def test_duplicate_model_card_identifier_fails_integrity_closed(
    service_system: _ServiceSystem,
) -> None:
    run_id = "duplicate-model-card-run"
    manifests = tuple(
        ModelCardManifest(
            run_id=run_id,
            generated_at=NOW,
            limitations=("Synthetic ambiguity fixture only.",),
            card_id="duplicate-card",
            model_name="baseline-forecast",
            model_version=version,
            sections=_model_card_sections(),
        )
        for version in ("1.0.0", "1.0.1")
    )
    published = tuple(
        _publish(
            service_system.root,
            service_system.registry,
            service_system.store,
            name=f"duplicate-card-{index}.json",
            payload=manifest.canonical_json_bytes(),
        )
        for index, manifest in enumerate(manifests)
    )
    _complete_run(
        service_system.registry,
        run_id=run_id,
        ordinal=12,
        links=tuple(ArtifactLink("model_card", artifact.digest) for artifact in published),
    )

    response = _request(
        service_system.app,
        "GET",
        "/api/v1/model-cards/duplicate-card",
        params={"run_id": encode_run_reference(run_id)},
    )

    _assert_problem(response, 503, "evidence_integrity_failed")
    assert "retry-after" not in response.headers


def test_route_and_openapi_inventory_is_exact_read_only_and_has_no_trading_authority(
    service_system: _ServiceSystem,
) -> None:
    assert_read_only_route_inventory(service_system.app)
    routes = {
        route.path: route.methods
        for route in service_system.app.routes
        if hasattr(route, "methods")
    }
    assert all(methods == {"GET"} for methods in routes.values())
    forbidden_terms = {"order", "broker", "trade", "position", "execute", "write", "delete"}
    assert not any(term in path.lower() for term in forbidden_terms for path in routes)

    response = _request(service_system.app, "GET", "/api/v1/openapi.json")

    assert response.status_code == 200
    document = response.json()
    assert set(document["paths"]) == set(routes)
    assert all(set(path_item) == {"get"} for path_item in document["paths"].values())
    assert document.get("servers", []) == []
    schemas = document["components"]["schemas"]
    assert "ProblemDocument" in schemas
    assert "HTTPValidationError" not in schemas
    assert "ValidationError" not in schemas
    assert schemas["ReadyResponse"]["properties"]["code"] == {
        "$ref": "#/components/schemas/EvidenceReadinessCode"
    }
    assert schemas["RunResponse"]["properties"]["status"]["anyOf"][0] == {
        "$ref": "#/components/schemas/RunStatus"
    }
    assert schemas["RunResponse"]["properties"]["evidence_class"]["anyOf"][0] == {
        "$ref": "#/components/schemas/EvidenceClass"
    }
    assert schemas["ArtifactResponse"]["properties"]["artifact_class"] == {
        "$ref": "#/components/schemas/ArtifactClass"
    }
    assert schemas["ForecastSummaryPageResponse"]["properties"]["items"]["maxItems"] == 7
    assert schemas["DiagnosticsPageResponse"]["properties"]["items"]["maxItems"] == 1
    assert document["paths"]["/api/v1/openapi.json"]["get"]["responses"]["200"] == {
        "description": "The deterministic OpenAPI 3.1 service contract.",
        "content": {
            "application/json": {
                "schema": {"$ref": "https://spec.openapis.org/oas/3.1/schema/2025-11-23"}
            }
        },
    }
    for path, path_item in document["paths"].items():
        responses = path_item["get"]["responses"]
        for code in ("400", "405", "413", "429", "500"):
            assert responses[code]["content"] == {
                "application/problem+json": {
                    "schema": {"$ref": "#/components/schemas/ProblemDocument"}
                }
            }, (path, code)
    assert document["paths"]["/api/v1/runs"]["get"]["responses"]["422"]["content"] == {
        "application/problem+json": {"schema": {"$ref": "#/components/schemas/ProblemDocument"}}
    }


def test_pagination_cursors_are_run_filter_and_evidence_role_bound(
    service_system: _ServiceSystem,
) -> None:
    async def scenario(client: httpx.AsyncClient) -> tuple[httpx.Response, ...]:
        reference = service_system.run_reference
        first_forecast = await client.get(
            f"/api/v1/runs/{reference}/forecast-summaries?page_size=1"
        )
        forecast_cursor = first_forecast.json()["next_cursor"]
        second_forecast = await client.get(
            f"/api/v1/runs/{reference}/forecast-summaries",
            params={"page_size": 1, "cursor": forecast_cursor},
        )
        wrong_role = await client.get(
            f"/api/v1/runs/{reference}/diagnostics",
            params={"cursor": forecast_cursor},
        )
        first_artifact = await client.get(f"/api/v1/runs/{reference}/artifacts?page_size=1")
        artifact_cursor = first_artifact.json()["next_cursor"]
        role_from_unfiltered = await client.get(
            f"/api/v1/runs/{reference}/forecast-summaries",
            params={"cursor": artifact_cursor},
        )
        first_run = await client.get("/api/v1/runs?page_size=1")
        run_cursor = first_run.json()["next_cursor"]
        # One run has no continuation. Create a syntactically valid cursor from
        # the read port so the API test can prove filter binding at the boundary.
        if run_cursor is None:
            direct_page = service_system.ports.list_runs(page=RunPageRequest(page_size=1))
            run_cursor = None if direct_page.next_cursor is None else direct_page.next_cursor.token
        mismatched_run = (
            await client.get(
                "/api/v1/runs",
                params={"cursor": run_cursor, "status": "failed"},
            )
            if run_cursor is not None
            else first_run
        )
        return (
            first_forecast,
            second_forecast,
            wrong_role,
            role_from_unfiltered,
            first_run,
            mismatched_run,
        )

    first, second, wrong_role, unfiltered, first_run, mismatched_run = _run_scenario(
        service_system.app, scenario
    )

    assert first.status_code == second.status_code == 200
    assert len(first.json()["items"]) == len(second.json()["items"]) == 1
    observed_ids = {
        first.json()["items"][0]["aggregates"][0]["aggregate_id"],
        second.json()["items"][0]["aggregates"][0]["aggregate_id"],
    }
    assert observed_ids == {"aggregate-001", "aggregate-002"}
    _assert_problem(wrong_role, 400, "invalid_request")
    _assert_problem(unfiltered, 400, "invalid_request")
    assert first_run.status_code == 200
    if mismatched_run is not first_run:
        _assert_problem(mismatched_run, 400, "invalid_request")


def test_run_cursor_is_filter_bound_at_http_boundary(tmp_path: Path) -> None:
    system = _build_minimal_system(tmp_path, run_count=2)

    async def scenario(client: httpx.AsyncClient) -> tuple[httpx.Response, httpx.Response]:
        first = await client.get("/api/v1/runs?page_size=1")
        cursor = first.json()["next_cursor"]
        mismatched = await client.get(
            "/api/v1/runs",
            params={"page_size": 1, "cursor": cursor, "status": "failed"},
        )
        return first, mismatched

    first, mismatched = _run_scenario(system.app, scenario)

    assert first.status_code == 200
    assert first.json()["next_cursor"] is not None
    _assert_problem(mismatched, 400, "invalid_request")


def _build_minimal_system(tmp_path: Path, *, run_count: int) -> _ServiceSystem:
    root = (tmp_path / "minimal-service-system").resolve()
    root.mkdir(mode=0o700)
    store = ArtifactStore(root / "cas")
    store.initialize()
    registry = RunRegistry(
        root / "registry.sqlite",
        digest_secret=SECRET,
        limits=RegistryLimits(busy_timeout_ms=2_000),
        clock=_MutableClock(NOW),
        artifact_verifier=store,
    )
    registry.initialize()
    for ordinal in range(run_count):
        _complete_run(
            registry,
            run_id=f"minimal-run-{ordinal}",
            ordinal=100 + ordinal,
            links=(),
        )
    ports = RegistryReadPorts(registry, store)
    return _ServiceSystem(
        app=create_app(ports),
        registry=registry,
        store=store,
        ports=ports,
        root=root,
        run_id="minimal-run-0",
        run_reference=encode_run_reference("minimal-run-0"),
        forecast_ids=(),
        diagnostic_id="0" * 64,
        model_card_id="0" * 64,
        raw_artifact_id="0" * 64,
    )


@pytest.mark.parametrize(
    ("method", "url", "kwargs", "status", "code"),
    [
        ("POST", "/api/v1/runs", {"content": b"{}"}, 405, "method_not_allowed"),
        ("PUT", "/api/v1/runs", {"content": b"{}"}, 405, "method_not_allowed"),
        ("DELETE", "/api/v1/runs/r1_Zm9v", {}, 405, "method_not_allowed"),
        ("PATCH", "/health/ready", {"content": b"{}"}, 405, "method_not_allowed"),
        ("HEAD", "/health/live", {}, 405, "method_not_allowed"),
        ("TRACE", "/health/live", {}, 405, "method_not_allowed"),
        (
            "GET",
            "/health/live",
            {"content": b"forbidden-body"},
            413,
            "request_body_forbidden",
        ),
    ],
)
def test_methods_and_request_bodies_fail_at_raw_security_boundary(
    service_system: _ServiceSystem,
    method: str,
    url: str,
    kwargs: dict[str, object],
    status: int,
    code: str,
) -> None:
    response = _request(service_system.app, method, url, **kwargs)

    if method == "HEAD":
        assert response.status_code == status
        assert response.content == b""
        assert int(response.headers["content-length"]) > 0
        assert response.headers["content-type"] == "application/problem+json"
        _assert_security_headers(response)
    else:
        _assert_problem(response, status, code)


@pytest.mark.parametrize(
    "headers",
    [
        {"host": "attacker.example"},
        {"host": "127.0.0.1:00080"},
        {"forwarded": "for=203.0.113.7"},
        {"x-forwarded-for": "203.0.113.7"},
        {"x-forwarded-host": "attacker.example"},
        {"x-forwarded-proto": "https"},
        {"x-request-id": "caller-controlled"},
        {"transfer-encoding": "chunked"},
        {"expect": "100-continue"},
    ],
)
def test_host_proxy_request_id_and_transfer_metadata_fail_closed(
    service_system: _ServiceSystem,
    headers: dict[str, str],
) -> None:
    response = _request(service_system.app, "GET", "/health/live", headers=headers)

    document = _assert_problem(response, 400, "invalid_request")
    assert not any(value in response.text for value in headers.values())
    assert document["request_id"] != headers.get("x-request-id")


def test_docs_redirects_cors_and_unknown_routes_remain_disabled(
    service_system: _ServiceSystem,
) -> None:
    async def scenario(client: httpx.AsyncClient) -> tuple[httpx.Response, ...]:
        return (
            await client.get("/docs"),
            await client.get("/redoc"),
            await client.get("/openapi.json"),
            await client.get("/api/v1/runs/"),
            await client.options(
                "/api/v1/runs",
                headers={
                    "origin": "https://attacker.example",
                    "access-control-request-method": "GET",
                },
            ),
            await client.get(
                "/health/live",
                headers={"origin": "https://attacker.example"},
            ),
        )

    docs, redoc, default_schema, redirect, preflight, origin = _run_scenario(
        service_system.app, scenario
    )

    for response in (docs, redoc, default_schema, redirect):
        _assert_problem(response, 404, "not_found")
        assert response.history == []
    _assert_problem(preflight, 405, "method_not_allowed")
    assert origin.status_code == 200
    _assert_security_headers(origin)


def test_validation_and_not_found_errors_are_rfc9457_redacted(
    service_system: _ServiceSystem,
) -> None:
    async def scenario(client: httpx.AsyncClient) -> tuple[httpx.Response, ...]:
        return (
            await client.get("/api/v1/runs?page_size=0"),
            await client.get("/api/v1/runs?status=caller-secret"),
            await client.get("/api/v1/runs/not-a-canonical-reference"),
            await client.get("/api/v1/artifacts/not-a-secret-digest"),
            await client.get("/api/v1/artifacts/" + "f" * 64),
            await client.get(
                "/api/v1/model-cards/missing-card",
                params={"run_id": service_system.run_reference},
            ),
        )

    page, status, reference, malformed_digest, missing_artifact, missing_card = _run_scenario(
        service_system.app, scenario
    )

    _assert_problem(page, 422, "request_validation_failed")
    _assert_problem(status, 422, "request_validation_failed")
    _assert_problem(reference, 422, "request_validation_failed")
    _assert_problem(malformed_digest, 422, "request_validation_failed")
    _assert_problem(missing_artifact, 404, "not_found")
    _assert_problem(missing_card, 404, "not_found")
    combined = b"".join(
        response.content
        for response in (page, status, reference, malformed_digest, missing_artifact, missing_card)
    )
    for secret in (b"caller-secret", b"not-a-canonical-reference", b"not-a-secret-digest"):
        assert secret not in combined
    assert str(service_system.root).encode() not in combined


def test_legacy_artifact_route_rejects_unauthenticated_cursor(
    service_system: _ServiceSystem,
) -> None:
    legacy_run_id = "legacy-api-cursor"
    with closing(sqlite3.connect(service_system.registry.path)) as connection, connection:
        connection.execute("""
            CREATE TABLE runs (
                run_id TEXT PRIMARY KEY, experiment TEXT, name TEXT,
                started_at TEXT, ended_at TEXT, status TEXT, git_commit TEXT,
                data_hash TEXT, tickers TEXT, features TEXT, params TEXT,
                metrics TEXT, tags TEXT, artifacts TEXT
            )
            """)
        connection.execute(
            "INSERT INTO runs VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (
                legacy_run_id,
                "redacted-experiment",
                "redacted-name",
                NOW.isoformat(),
                (NOW + timedelta(seconds=1)).isoformat(),
                "completed",
                "abcdef0",
                "sha256:" + "b" * 64,
                "[]",
                "[]",
                "{}",
                "{}",
                "{}",
                "[]",
            ),
        )

    response = _request(
        service_system.app,
        "GET",
        f"/api/v1/runs/{encode_run_reference(legacy_run_id)}/artifacts",
        params={"cursor": "forged-cursor-00"},
    )

    _assert_problem(response, 400, "invalid_request")
    assert b"forged-cursor-00" not in response.content


@pytest.mark.parametrize("corruption", ["malformed", "schema", "kind", "media", "oversized"])
def test_malformed_wrong_schema_kind_and_media_manifests_fail_integrity_closed(
    service_system: _ServiceSystem,
    corruption: str,
) -> None:
    run_id = f"corrupt-{corruption}-run"
    valid = _forecast_manifest(run_id, "aggregate-corrupt").canonical_json_bytes()
    if corruption == "malformed":
        payload = b'{"kind":"forecast_summary","private":"do-not-reflect"'
        media_type = MANIFEST_MEDIA_TYPE
    elif corruption == "schema":
        payload = (
            b'{"aggregates":[],"generated_at":"2026-08-09T12:00:00Z",'
            b'"kind":"forecast_summary","limitations":[],"row_level_available":false,'
            b'"run_id":"corrupt-schema-run","schema_version":2}'
        )
        media_type = MANIFEST_MEDIA_TYPE
    elif corruption == "kind":
        payload = _diagnostics_manifest(run_id).canonical_json_bytes()
        media_type = MANIFEST_MEDIA_TYPE
    elif corruption == "media":
        payload = valid
        media_type = "application/json"
    else:
        payload = b"x" * (128 * 1024 + 1)
        media_type = MANIFEST_MEDIA_TYPE
    published = _publish(
        service_system.root,
        service_system.registry,
        service_system.store,
        name=f"corrupt-{corruption}.json",
        payload=payload,
        media_type=media_type,
    )
    _complete_run(
        service_system.registry,
        run_id=run_id,
        ordinal={"malformed": 20, "schema": 21, "kind": 22, "media": 23, "oversized": 24}[
            corruption
        ],
        links=(ArtifactLink("forecast_summary", published.digest),),
    )

    response = _request(
        service_system.app,
        "GET",
        f"/api/v1/runs/{encode_run_reference(run_id)}/forecast-summaries",
    )

    document = _assert_problem(response, 503, "evidence_integrity_failed")
    assert "retry-after" not in response.headers
    assert "corrupt" not in str(document["detail"]).lower()
    assert b"do-not-reflect" not in response.content
    assert str(service_system.root).encode() not in response.content


def test_liveness_and_readiness_never_create_missing_registry_or_cas(tmp_path: Path) -> None:
    root = (tmp_path / "missing-evidence").resolve()
    database = root / "registry" / "registry.sqlite"
    cas_root = root / "cas"
    ports = RegistryReadPorts(
        RunRegistry(database, digest_secret=SECRET),
        ArtifactStore(cas_root),
    )
    app = create_app(ports)
    assert not root.exists()

    async def scenario(client: httpx.AsyncClient) -> tuple[httpx.Response, httpx.Response]:
        return await client.get("/health/live"), await client.get("/health/ready")

    live, ready = _run_scenario(app, scenario)

    assert live.status_code == 200
    assert ready.status_code == 503
    assert "retry-after" not in ready.headers
    assert ready.json() == {
        "schema_version": 1,
        "ready": False,
        "code": "registry_missing",
        "registry_schema_version": None,
        "journal_mode": None,
        "retryable": False,
    }
    _assert_security_headers(live)
    _assert_security_headers(ready)
    assert not root.exists()


class _RetryableReadinessPorts:
    """Minimal injected port exposing one closed retryable readiness verdict."""

    def probe_evidence_readiness(self) -> EvidenceReadiness:
        return EvidenceReadiness(
            ready=False,
            code=EvidenceReadinessCode.REGISTRY_BUSY,
            schema_version=1,
            journal_mode="wal",
        )

    def get_run(self, run_id: str) -> RunReadModel:
        raise AssertionError(f"unexpected run read: {run_id}")

    def list_runs(
        self,
        query: RunQuery | None = None,
        page: RunPageRequest | None = None,
    ) -> Page[RunReadModel, RunCursor]:
        raise AssertionError(f"unexpected run listing: {query!r}, {page!r}")

    def get_artifact(self, artifact_id: str) -> ArtifactView:
        raise AssertionError(f"unexpected artifact read: {artifact_id}")

    def list_run_artifacts(
        self,
        run_id: str,
        page: ArtifactPageRequest | None = None,
        *,
        query: ArtifactQuery | None = None,
    ) -> Page[RunArtifactView, ArtifactCursor]:
        raise AssertionError(f"unexpected artifact listing: {run_id}, {page!r}, {query!r}")

    def read_verified_manifest(
        self,
        artifact_id: str,
        expected_media_type: str,
        max_bytes: int,
    ) -> bytes:
        raise AssertionError(
            "unexpected manifest read: " f"{artifact_id}, {expected_media_type}, {max_bytes}"
        )


def test_retry_after_is_emitted_only_for_retryable_readiness() -> None:
    app = create_app(_RetryableReadinessPorts())

    response = _request(app, "GET", "/health/ready")

    assert response.status_code == 503
    assert response.headers["retry-after"] == "1"
    assert response.json()["retryable"] is True


class _DelegatingPorts:
    """Explicit test adapter used to inject one controlled boundary behavior."""

    def __init__(self, delegate: RegistryReadPorts) -> None:
        self.delegate = delegate

    def probe_evidence_readiness(self) -> EvidenceReadiness:
        return self.delegate.probe_evidence_readiness()

    def get_run(self, run_id: str) -> RunReadModel:
        return self.delegate.get_run(run_id)

    def list_runs(
        self,
        query: RunQuery | None = None,
        page: RunPageRequest | None = None,
    ) -> Page[RunReadModel, RunCursor]:
        return self.delegate.list_runs(query, page)

    def get_artifact(self, artifact_id: str) -> ArtifactView:
        return self.delegate.get_artifact(artifact_id)

    def list_run_artifacts(
        self,
        run_id: str,
        page: ArtifactPageRequest | None = None,
        *,
        query: ArtifactQuery | None = None,
    ) -> Page[RunArtifactView, ArtifactCursor]:
        return self.delegate.list_run_artifacts(run_id, page, query=query)

    def read_verified_manifest(
        self,
        artifact_id: str,
        expected_media_type: str,
        max_bytes: int,
    ) -> bytes:
        return self.delegate.read_verified_manifest(artifact_id, expected_media_type, max_bytes)


class _ExplodingPorts(_DelegatingPorts):
    def get_run(self, run_id: str) -> RunReadModel:
        del run_id
        raise RuntimeError("private-path=/tmp/secret private-token=do-not-reflect")


def test_unexpected_application_failure_is_redacted_by_outer_asgi_boundary(
    service_system: _ServiceSystem,
) -> None:
    app = create_app(_ExplodingPorts(service_system.ports))

    response = _request(app, "GET", f"/api/v1/runs/{service_system.run_reference}")

    _assert_problem(response, 500, "internal_error")
    assert b"private-path" not in response.content
    assert b"private-token" not in response.content
    assert str(service_system.root).encode() not in response.content


class _BlockingReadinessPorts(_DelegatingPorts):
    def __init__(self, delegate: RegistryReadPorts) -> None:
        super().__init__(delegate)
        self.entered = threading.Event()
        self.release = threading.Event()

    def probe_evidence_readiness(self) -> EvidenceReadiness:
        self.entered.set()
        if not self.release.wait(timeout=2):
            raise RuntimeError("test readiness barrier timed out")
        return super().probe_evidence_readiness()


def test_concurrency_saturation_returns_bounded_429_without_wall_clock_sleep(
    service_system: _ServiceSystem,
) -> None:
    ports = _BlockingReadinessPorts(service_system.ports)
    app = create_app(ports, max_concurrency=1)

    async def scenario(client: httpx.AsyncClient) -> tuple[httpx.Response, httpx.Response]:
        blocked = asyncio.create_task(client.get("/health/ready"))
        entered = await asyncio.wait_for(asyncio.to_thread(ports.entered.wait, 1), timeout=1.5)
        assert entered
        saturated = await client.get("/health/live")
        ports.release.set()
        completed = await asyncio.wait_for(blocked, timeout=1.5)
        return saturated, completed

    saturated, completed = _run_scenario(app, scenario)

    _assert_problem(saturated, 429, "service_saturated")
    assert saturated.headers["retry-after"] == "1"
    assert completed.status_code == 200
    _assert_security_headers(completed)
