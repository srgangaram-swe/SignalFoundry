"""Benchmark the bounded read-only service through its public ASGI boundary.

The workload is deterministic in scenario construction, request ordering, and
synthetic aggregate inputs.  Timings and operating-system resource readings are
measured and therefore host-dependent.  Only aggregate distributions are
written; raw request samples remain process-local.

This is local engineering evidence.  It uses no network, credential, market
data, model, broker, order, or capital authority and cannot establish a
production SLO, trading readiness, or profitability.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import hmac
import json
import logging
import math
import os
import platform
import re
import resource
import selectors
import stat
import statistics
import subprocess
import sys
import tempfile
import threading
import time
from collections import Counter
from collections.abc import Callable, Sequence
from contextlib import suppress
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from enum import StrEnum
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any, Protocol

import httpx
from asgi_lifespan import LifespanManager

from quant_platform.service.admission import AdmissionController, AdmissionLimits
from quant_platform.service.api import EvidenceReadPort, create_app
from quant_platform.service.exporter import ExporterConfig
from quant_platform.service.manifests import (
    DiagnosticCategory,
    DiagnosticsManifest,
    DiagnosticStatus,
    DiagnosticValue,
)
from quant_platform.service.metrics import ServiceMetrics
from quant_platform.service.telemetry import ServiceTelemetry
from quant_platform.service.telemetry_contracts import TelemetryChannel
from quant_platform.tracking.contracts import (
    ArtifactClass,
    ArtifactCursor,
    BusyError,
    EvidenceClass,
    NotFoundError,
    Page,
    RunCursor,
    RunSnapshot,
    RunStatus,
)
from quant_platform.tracking.read_ports import (
    ArtifactPageRequest,
    ArtifactQuery,
    ArtifactView,
    EvidenceReadiness,
    EvidenceReadinessCode,
    RunArtifactView,
    RunPageRequest,
    RunQuery,
    RunReadModel,
)

SCHEMA_VERSION = "1.0.0"
EVIDENCE_TIME = datetime(2026, 8, 9, 18, 0, tzinfo=UTC)
SEED = 20260809
WARMUP_REQUESTS = 8
COLD_REPETITIONS = 7
STEADY_AB_ROUNDS = 4
STEADY_REQUESTS_PER_ROUND = 16
MAXIMUM_REQUESTS = 12
SATURATION_REQUESTS = 48
CONCURRENT_METRICS_REQUESTS = 8
SYNTHETIC_RUNS = 100
SYNTHETIC_DIAGNOSTICS = 256
MAX_RESPONSE_BYTES = 2 * 1024 * 1024
MAX_METRICS_BYTES = 256 * 1024
MAX_METRIC_SERIES = 599  # The issue contract requires strictly fewer than 600.
MAX_REJECTION_MILLISECONDS = 250.0
MAX_TELEMETRY_RSS_DELTA_BYTES = 100 * 1024 * 1024
MAX_EVIDENCE_FILE_BYTES = 2 * 1024 * 1024
_PUBLICATION_CHUNK_BYTES = 64 * 1024
_PUBLICATION_MODE = 0o644
_MANIFEST_MEDIA_TYPE = "application/vnd.signalattice.manifest+json"
_DIAGNOSTIC_DIGEST = "d" * 64
_PRIMARY_RUN_ID = "service-benchmark-run-000"
_SERVICE_CANARY_PATTERN = re.compile(r"^[A-Za-z0-9._~-]{16,128}$")
_STEADY_ROUTES = (
    "/api/v1/runs?page_size=5",
    "/api/v1/runs/r1_c2VydmljZS1iZW5jaG1hcmstcnVuLTAwMA",
    "/health/ready",
    "/health/live",
)
_ISOLATED_RSS_PROTOCOL = "signalattice-service-rss-v1"
_ISOLATED_RSS_READY = b"SIGNALATTICE_SERVICE_RSS_READY_V1\n"
_ISOLATED_RSS_START = b"SIGNALATTICE_SERVICE_RSS_START_V1\n"
_ISOLATED_RSS_RESULT_PREFIX = b"SIGNALATTICE_SERVICE_RSS_RESULT_V1 "
_ISOLATED_RSS_MAXIMUM_LINE_BYTES = 8 * 1024
_ISOLATED_RSS_HARD_TIMEOUT_SECONDS = 20.0
_ISOLATED_RSS_TERMINATION_GRACE_SECONDS = 0.5
_ISOLATED_RSS_MODES = ("disabled", "enabled")


class ASGIApplication(Protocol):
    """Structural ASGI application contract used by HTTPX and lifespan tooling."""

    async def __call__(
        self,
        scope: dict[str, Any],
        receive: Callable[[], Any],
        send: Callable[[dict[str, Any]], Any],
    ) -> None: ...


class AppFactory(Protocol):
    """Public benchmark integration point; implementations may inject telemetry."""

    def __call__(
        self,
        ports: EvidenceReadPort,
        *,
        telemetry_enabled: bool,
        admission: AdmissionController,
    ) -> ASGIApplication: ...


class _PortFault(StrEnum):
    NONE = "none"
    BUSY = "busy"
    CORRUPT_MANIFEST = "corrupt_manifest"


class _StepClock:
    """Thread-safe deterministic clock used only for admission refill decisions."""

    def __init__(self, step_seconds: float) -> None:
        if not math.isfinite(step_seconds) or step_seconds < 0.0:
            raise ValueError("step_seconds must be finite and non-negative")
        self._step_seconds = step_seconds
        self._value = 1_000.0
        self._lock = threading.Lock()

    def __call__(self) -> float:
        with self._lock:
            value = self._value
            self._value += self._step_seconds
            return value


class _BlockingGate:
    """Bounded test barrier for deterministic data-route saturation."""

    def __init__(self, target: int) -> None:
        if type(target) is not int or target < 1:
            raise ValueError("target must be a positive integer")
        self.target = target
        self._entered = 0
        self._lock = threading.Lock()
        self._release = threading.Event()

    @property
    def entered(self) -> int:
        with self._lock:
            return self._entered

    def wait(self) -> None:
        with self._lock:
            self._entered += 1
        if not self._release.wait(timeout=5.0):
            raise BusyError("synthetic bounded saturation barrier expired")

    def release(self) -> None:
        self._release.set()


class _AggregateSinkExporter:
    """Network-free sink that validates the real asynchronous export path."""

    async def export(
        self,
        channel: TelemetryChannel,
        records: tuple[bytes, ...],
    ) -> None:
        if type(channel) is not TelemetryChannel or not records:
            raise RuntimeError("benchmark exporter received an invalid bounded batch")
        if any(type(record) is not bytes or not record.endswith(b"\n") for record in records):
            raise RuntimeError("benchmark exporter received an invalid telemetry record")
        await asyncio.sleep(0)


class _SyntheticPorts:
    """Path-free, network-free aggregate fixtures for ASGI boundary measurements."""

    def __init__(
        self,
        *,
        fault: _PortFault = _PortFault.NONE,
        readiness: EvidenceReadinessCode = EvidenceReadinessCode.READY,
        gate: _BlockingGate | None = None,
        canary_marker: str | None = None,
    ) -> None:
        if canary_marker is not None and _SERVICE_CANARY_PATTERN.fullmatch(canary_marker) is None:
            raise ValueError("synthetic canary violates its bounded public test contract")
        self._fault = fault
        self._readiness = readiness
        self._gate = gate
        self._canary_marker = canary_marker
        self._runs = _synthetic_runs()
        self._diagnostic_bytes = _diagnostic_manifest().canonical_json_bytes()
        self._diagnostic_view = RunArtifactView(
            sequence=1,
            role="diagnostics",
            linked_at=EVIDENCE_TIME,
            artifact=ArtifactView(
                artifact_id=_DIAGNOSTIC_DIGEST,
                artifact_class=ArtifactClass.METADATA,
                byte_size=len(self._diagnostic_bytes),
                media_type=_MANIFEST_MEDIA_TYPE,
                created_at=EVIDENCE_TIME,
                pinned=False,
            ),
        )

    def probe_evidence_readiness(self) -> EvidenceReadiness:
        return EvidenceReadiness(
            ready=self._readiness is EvidenceReadinessCode.READY,
            code=self._readiness,
            schema_version=2 if self._readiness is EvidenceReadinessCode.READY else None,
            journal_mode="wal" if self._readiness is EvidenceReadinessCode.READY else None,
        )

    def get_run(self, run_id: str) -> RunReadModel:
        for run in self._runs:
            if run.run_id == run_id:
                return run
        raise NotFoundError("synthetic run does not exist")

    def list_runs(
        self,
        query: RunQuery | None = None,
        page: RunPageRequest | None = None,
    ) -> Page[RunReadModel, RunCursor]:
        del query
        if self._gate is not None:
            self._gate.wait()
        if self._fault is _PortFault.BUSY:
            raise BusyError("synthetic busy fault")
        page_size = 50 if page is None else page.page_size
        return Page(tuple(self._runs[:page_size]), None)

    def get_artifact(self, artifact_id: str) -> ArtifactView:
        if artifact_id != _DIAGNOSTIC_DIGEST:
            raise NotFoundError("synthetic artifact does not exist")
        return self._diagnostic_view.artifact

    def list_run_artifacts(
        self,
        run_id: str,
        page: ArtifactPageRequest | None = None,
        *,
        query: ArtifactQuery | None = None,
    ) -> Page[RunArtifactView, ArtifactCursor]:
        del page
        if run_id != _PRIMARY_RUN_ID:
            raise NotFoundError("synthetic run does not exist")
        if query is not None and query.role not in {None, "diagnostics"}:
            return Page((), None)
        return Page((self._diagnostic_view,), None)

    def read_verified_manifest(
        self,
        artifact_id: str,
        expected_media_type: str,
        max_bytes: int,
    ) -> bytes:
        if artifact_id != _DIAGNOSTIC_DIGEST or expected_media_type != _MANIFEST_MEDIA_TYPE:
            raise NotFoundError("synthetic manifest does not exist")
        if self._fault is _PortFault.CORRUPT_MANIFEST:
            marker = "synthetic-canary" if self._canary_marker is None else self._canary_marker
            return b'{"kind":"diagnostics","private":"' + marker.encode("ascii") + b'"'
        if len(self._diagnostic_bytes) > max_bytes:
            raise BusyError("synthetic manifest exceeded caller bound")
        return self._diagnostic_bytes


@dataclass(frozen=True, slots=True)
class _RequestSample:
    latency_ms: float
    status_code: int
    response_bytes: int
    problem_code: str | None


@dataclass(frozen=True, slots=True)
class _ResourceSnapshot:
    cpu_seconds: float
    maximum_rss_bytes: int
    file_descriptors: int | None
    threads: int


def _synthetic_runs() -> tuple[RunSnapshot, ...]:
    return tuple(
        RunSnapshot(
            sequence=index + 1,
            run_id=f"service-benchmark-run-{index:03d}",
            job_id=f"service-benchmark-job-{index:03d}",
            attempt=1,
            status=RunStatus.SUCCEEDED,
            evidence_class=EvidenceClass.SIMULATED,
            schema_version=1,
            created_at=EVIDENCE_TIME - timedelta(minutes=SYNTHETIC_RUNS - index),
            started_at=EVIDENCE_TIME - timedelta(minutes=SYNTHETIC_RUNS - index),
            ended_at=EVIDENCE_TIME - timedelta(minutes=SYNTHETIC_RUNS - index - 1),
            source_commit="abcdef0123456789",
            data_identity="sha256:" + f"{index:064x}",
            limitation_summary="Deterministic synthetic service benchmark; no trading claim.",
        )
        for index in range(SYNTHETIC_RUNS)
    )


def _diagnostic_manifest() -> DiagnosticsManifest:
    values = tuple(
        sorted(
            (
                DiagnosticValue(
                    code=f"benchmark-{index:03d}",
                    category=(
                        DiagnosticCategory.LATENCY
                        if index % 2 == 0
                        else DiagnosticCategory.READINESS
                    ),
                    value=float(index) / 10.0,
                    unit="synthetic-unit",
                    status=(DiagnosticStatus.INFORMATIONAL if index % 3 else DiagnosticStatus.WARN),
                    detail=(
                        "Synthetic aggregate diagnostic for bounded response mechanics. "
                        + "x" * 128
                    ),
                )
                for index in range(SYNTHETIC_DIAGNOSTICS)
            ),
            key=lambda value: (value.category.value, value.code),
        )
    )
    return DiagnosticsManifest(
        run_id=_PRIMARY_RUN_ID,
        generated_at=EVIDENCE_TIME,
        limitations=(
            "Synthetic local aggregate input only.",
            "No market, model, paper-trading, live-trading, or profitability evidence.",
        ),
        values=values,
    )


def _admission(*, step_seconds: float) -> AdmissionController:
    return AdmissionController(AdmissionLimits(), clock=_StepClock(step_seconds))


def _file_descriptor_count() -> int | None:
    for candidate in (Path("/proc/self/fd"), Path("/dev/fd")):
        try:
            with os.scandir(candidate) as entries:
                return sum(1 for _ in entries)
        except OSError:
            continue
    return None


def _maximum_rss_bytes() -> int:
    value = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    # Darwin reports bytes; Linux and the other supported CI Unix runners
    # report KiB.  The environment field retains the platform interpretation.
    return value if sys.platform == "darwin" else value * 1024


def _resource_snapshot() -> _ResourceSnapshot:
    usage = resource.getrusage(resource.RUSAGE_SELF)
    return _ResourceSnapshot(
        cpu_seconds=float(usage.ru_utime + usage.ru_stime),
        maximum_rss_bytes=_maximum_rss_bytes(),
        file_descriptors=_file_descriptor_count(),
        threads=threading.active_count(),
    )


def _linear_percentile(sorted_values: Sequence[float], percentile: float) -> float:
    if not sorted_values:
        raise ValueError("cannot summarize an empty distribution")
    if not 0.0 <= percentile <= 100.0:
        raise ValueError("percentile must be in [0, 100]")
    if len(sorted_values) == 1:
        return float(sorted_values[0])
    rank = percentile / 100.0 * (len(sorted_values) - 1)
    lower = math.floor(rank)
    upper = math.ceil(rank)
    fraction = rank - lower
    return float(sorted_values[lower] * (1.0 - fraction) + sorted_values[upper] * fraction)


def summarize_distribution(values: Sequence[float]) -> dict[str, object]:
    """Return a finite aggregate distribution without retaining raw samples."""

    if not values or any(not math.isfinite(value) or value < 0.0 for value in values):
        raise ValueError("distribution values must be non-empty, finite, and non-negative")
    ordered = sorted(float(value) for value in values)
    percentile_grid = (0.0, 10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0, 95.0, 99.0, 100.0)
    return {
        "count": len(ordered),
        "min": ordered[0],
        "mean": statistics.fmean(ordered),
        "variance": statistics.pvariance(ordered),
        "standard_deviation": statistics.pstdev(ordered),
        "p50": _linear_percentile(ordered, 50.0),
        "p95": _linear_percentile(ordered, 95.0),
        "p99": _linear_percentile(ordered, 99.0),
        "max": ordered[-1],
        "empirical_percentiles": [
            {"percentile": percentile, "value_ms": _linear_percentile(ordered, percentile)}
            for percentile in percentile_grid
        ],
    }


def _problem_code(response: httpx.Response) -> str | None:
    content_type = response.headers.get("content-type", "").split(";", 1)[0]
    if content_type not in {"application/problem+json", "application/json"}:
        return None
    if len(response.content) > 16 * 1024:
        return None
    try:
        document = response.json()
    except (UnicodeDecodeError, json.JSONDecodeError):
        return None
    if type(document) is not dict:
        return None
    code = document.get("code")
    return code if type(code) is str and len(code) <= 64 and code.isascii() else None


async def _request(
    client: httpx.AsyncClient, method: str, path: str, **kwargs: Any
) -> _RequestSample:
    started = time.perf_counter_ns()
    response = await client.request(method, path, **kwargs)
    latency_ms = (time.perf_counter_ns() - started) / 1_000_000.0
    return _RequestSample(
        latency_ms=latency_ms,
        status_code=response.status_code,
        response_bytes=len(response.content),
        problem_code=_problem_code(response),
    )


async def _open_client(app: ASGIApplication) -> tuple[LifespanManager, httpx.AsyncClient]:
    lifespan = LifespanManager(app)
    await lifespan.__aenter__()
    client = httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app, raise_app_exceptions=False),
        base_url="http://127.0.0.1",
        timeout=5.0,
    )
    return lifespan, client


async def _close_client(lifespan: LifespanManager, client: httpx.AsyncClient) -> None:
    await client.aclose()
    await lifespan.__aexit__(None, None, None)


def _scenario_record(
    *,
    name: str,
    display_name: str,
    telemetry_mode: str,
    samples: Sequence[_RequestSample],
    warmup_requests: int,
    before: _ResourceSnapshot,
    after: _ResourceSnapshot,
    wall_seconds: float,
) -> dict[str, object]:
    status_counts = Counter(str(sample.status_code) for sample in samples)
    problem_counts = Counter(
        sample.problem_code for sample in samples if sample.problem_code is not None
    )
    fd_delta = (
        None
        if before.file_descriptors is None or after.file_descriptors is None
        else after.file_descriptors - before.file_descriptors
    )
    return {
        "name": name,
        "display_name": display_name,
        "telemetry_mode": telemetry_mode,
        "sample_count": len(samples),
        "warmup_requests": warmup_requests,
        "latency_ms": summarize_distribution([sample.latency_ms for sample in samples]),
        "response_bytes": summarize_distribution(
            [float(sample.response_bytes) for sample in samples]
        ),
        "status_counts": dict(sorted(status_counts.items())),
        "problem_counts": dict(sorted(problem_counts.items())),
        "throughput_requests_per_second": len(samples) / wall_seconds,
        "resources": {
            "wall_seconds": wall_seconds,
            "cpu_seconds": after.cpu_seconds - before.cpu_seconds,
            "maximum_rss_before_bytes": before.maximum_rss_bytes,
            "maximum_rss_after_bytes": after.maximum_rss_bytes,
            "file_descriptors_before": before.file_descriptors,
            "file_descriptors_after": after.file_descriptors,
            "file_descriptor_delta": fd_delta,
            "threads_before": before.threads,
            "threads_after": after.threads,
            "processes": 1,
        },
    }


async def _measure_requests(
    app: ASGIApplication,
    paths: Sequence[str],
    *,
    warmup_paths: Sequence[str] = (),
) -> tuple[list[_RequestSample], _ResourceSnapshot, _ResourceSnapshot, float]:
    lifespan, client = await _open_client(app)
    try:
        for path in warmup_paths:
            warmup = await _request(client, "GET", path)
            if warmup.status_code != 200:
                raise RuntimeError("synthetic benchmark warm-up did not succeed")
        before = _resource_snapshot()
        started = time.perf_counter()
        samples = [await _request(client, "GET", path) for path in paths]
        wall_seconds = time.perf_counter() - started
        after = _resource_snapshot()
        return samples, before, after, wall_seconds
    finally:
        await _close_client(lifespan, client)


async def _cold_scenario(app_factory: AppFactory) -> dict[str, object]:
    samples: list[_RequestSample] = []
    before = _resource_snapshot()
    started = time.perf_counter()
    for _ in range(COLD_REPETITIONS):
        request_started = time.perf_counter_ns()
        app = app_factory(
            _SyntheticPorts(),
            telemetry_enabled=False,
            admission=_admission(step_seconds=0.5),
        )
        lifespan, client = await _open_client(app)
        try:
            response = await client.get("/health/live")
        finally:
            await _close_client(lifespan, client)
        samples.append(
            _RequestSample(
                latency_ms=(time.perf_counter_ns() - request_started) / 1_000_000.0,
                status_code=response.status_code,
                response_bytes=len(response.content),
                problem_code=_problem_code(response),
            )
        )
    wall_seconds = time.perf_counter() - started
    after = _resource_snapshot()
    return _scenario_record(
        name="cold_start",
        display_name="Cold app + liveness",
        telemetry_mode="disabled",
        samples=samples,
        warmup_requests=0,
        before=before,
        after=after,
        wall_seconds=wall_seconds,
    )


def _steady_workload() -> tuple[tuple[str, ...], tuple[str, ...]]:
    """Return the one canonical route mix shared by latency and RSS A/B arms."""

    paths = tuple(
        _STEADY_ROUTES[index % len(_STEADY_ROUTES)] for index in range(STEADY_REQUESTS_PER_ROUND)
    )
    warmup = tuple(_STEADY_ROUTES[index % len(_STEADY_ROUTES)] for index in range(WARMUP_REQUESTS))
    return paths, warmup


async def _steady_ab_scenarios(
    app_factory: AppFactory,
) -> tuple[dict[str, object], dict[str, object], list[list[str]]]:
    """Measure paired A/B arms in a balanced deterministic order.

    Alternating AB/BA rounds prevents either mode from owning every earlier
    process/cache state.  Each segment gets the same route order and warm-up.
    Request samples remain process-local; only aggregate distributions and
    per-round wall totals cross the evidence boundary.
    """

    paths, warmup = _steady_workload()
    schedule = [
        ["disabled", "enabled"] if round_index % 2 == 0 else ["enabled", "disabled"]
        for round_index in range(STEADY_AB_ROUNDS)
    ]
    segments: dict[
        bool,
        list[tuple[list[_RequestSample], _ResourceSnapshot, _ResourceSnapshot, float]],
    ] = {False: [], True: []}
    for round_order in schedule:
        for mode in round_order:
            telemetry_enabled = mode == "enabled"
            app = app_factory(
                _SyntheticPorts(),
                telemetry_enabled=telemetry_enabled,
                admission=_admission(step_seconds=0.5),
            )
            segments[telemetry_enabled].append(
                await _measure_requests(app, paths, warmup_paths=warmup)
            )

    records: list[dict[str, object]] = []
    for telemetry_enabled in (False, True):
        mode_segments = segments[telemetry_enabled]
        samples = [sample for segment in mode_segments for sample in segment[0]]
        first_before = mode_segments[0][1]
        last_after = mode_segments[-1][2]
        total_wall_seconds = sum(segment[3] for segment in mode_segments)
        record = _scenario_record(
            name="steady_route_mix",
            display_name="Steady route mix",
            telemetry_mode="enabled" if telemetry_enabled else "disabled",
            samples=samples,
            warmup_requests=len(warmup) * STEADY_AB_ROUNDS,
            before=first_before,
            after=last_after,
            wall_seconds=total_wall_seconds,
        )
        file_descriptor_delta = (
            None
            if first_before.file_descriptors is None or last_after.file_descriptors is None
            else last_after.file_descriptors - first_before.file_descriptors
        )
        record["resources"] = {
            "wall_seconds": total_wall_seconds,
            "round_wall_seconds": [segment[3] for segment in mode_segments],
            "cpu_seconds": sum(
                segment[2].cpu_seconds - segment[1].cpu_seconds for segment in mode_segments
            ),
            "maximum_rss_before_bytes": first_before.maximum_rss_bytes,
            "maximum_rss_after_bytes": max(
                segment[2].maximum_rss_bytes for segment in mode_segments
            ),
            "file_descriptors_before": first_before.file_descriptors,
            "file_descriptors_after": last_after.file_descriptors,
            "file_descriptor_delta": file_descriptor_delta,
            "threads_before": first_before.threads,
            "threads_after": last_after.threads,
            "processes": 1,
        }
        records.append(record)
    return records[0], records[1], schedule


async def _maximum_scenario(app_factory: AppFactory) -> dict[str, object]:
    app = app_factory(
        _SyntheticPorts(),
        telemetry_enabled=True,
        admission=_admission(step_seconds=0.5),
    )
    diagnostic_path = "/api/v1/runs/r1_c2VydmljZS1iZW5jaG1hcmstcnVuLTAwMA/diagnostics?page_size=1"
    routes = ("/api/v1/runs?page_size=100", diagnostic_path)
    paths = tuple(routes[index % 2] for index in range(MAXIMUM_REQUESTS))
    samples, before, after, wall_seconds = await _measure_requests(
        app,
        paths,
        warmup_paths=routes,
    )
    return _scenario_record(
        name="maximum_projection",
        display_name="Maximum page / diagnostics",
        telemetry_mode="enabled",
        samples=samples,
        warmup_requests=len(routes),
        before=before,
        after=after,
        wall_seconds=wall_seconds,
    )


async def _wait_for_gate(gate: _BlockingGate, maximum_yields: int = 100_000) -> None:
    for _ in range(maximum_yields):
        if gate.entered >= gate.target:
            return
        await asyncio.sleep(0)
    raise RuntimeError("saturation scenario did not reach the deterministic barrier")


async def _saturation_scenario(
    app_factory: AppFactory,
) -> tuple[dict[str, object], dict[str, object]]:
    gate = _BlockingGate(target=24)
    app = app_factory(
        _SyntheticPorts(gate=gate),
        telemetry_enabled=True,
        admission=_admission(step_seconds=0.0),
    )
    lifespan, client = await _open_client(app)
    before = _resource_snapshot()
    started = time.perf_counter()
    tasks = [
        asyncio.create_task(_request(client, "GET", "/api/v1/runs?page_size=1"))
        for _ in range(SATURATION_REQUESTS)
    ]
    try:
        await asyncio.wait_for(_wait_for_gate(gate), timeout=5.0)
        probe = await _request(client, "GET", "/health/live")
    finally:
        gate.release()
    samples = list(await asyncio.gather(*tasks))
    wall_seconds = time.perf_counter() - started
    after = _resource_snapshot()
    await _close_client(lifespan, client)
    rejection_latencies = [sample.latency_ms for sample in samples if sample.status_code != 200]
    record = _scenario_record(
        name="saturation",
        display_name="Data saturation",
        telemetry_mode="enabled",
        samples=samples,
        warmup_requests=0,
        before=before,
        after=after,
        wall_seconds=wall_seconds,
    )
    detail = {
        "submitted_requests": SATURATION_REQUESTS,
        "barrier_target": gate.target,
        "probe_status": probe.status_code,
        "probe_latency_ms": probe.latency_ms,
        "rejection_count": len(rejection_latencies),
        "maximum_rejection_latency_ms": max(rejection_latencies, default=0.0),
    }
    return record, detail


async def _metrics_scenario(
    app_factory: AppFactory,
) -> tuple[dict[str, object], bytes]:
    app = app_factory(
        _SyntheticPorts(),
        telemetry_enabled=True,
        admission=_admission(step_seconds=0.0),
    )
    lifespan, client = await _open_client(app)
    before = _resource_snapshot()
    started = time.perf_counter()
    samples = list(
        await asyncio.gather(
            *(
                _request(client, "GET", "/internal/metrics")
                for _ in range(CONCURRENT_METRICS_REQUESTS)
            )
        )
    )
    wall_seconds = time.perf_counter() - started
    # Use a fresh refill clock/app for the bounded exposition snapshot so the
    # concurrent rate-rejection evidence cannot suppress measurement.
    await _close_client(lifespan, client)
    after = _resource_snapshot()
    snapshot_app = app_factory(
        _SyntheticPorts(),
        telemetry_enabled=True,
        admission=_admission(step_seconds=0.5),
    )
    snapshot_lifespan, snapshot_client = await _open_client(snapshot_app)
    try:
        snapshot = await snapshot_client.get("/internal/metrics")
    finally:
        await _close_client(snapshot_lifespan, snapshot_client)
    if snapshot.status_code != 200:
        raise RuntimeError("metrics snapshot did not succeed")
    return (
        _scenario_record(
            name="concurrent_metrics",
            display_name="Concurrent metrics scrapes",
            telemetry_mode="enabled",
            samples=samples,
            warmup_requests=0,
            before=before,
            after=after,
            wall_seconds=wall_seconds,
        ),
        snapshot.content,
    )


async def _fault_scenario(
    app_factory: AppFactory,
    *,
    canary_marker: str | None,
) -> tuple[dict[str, object], list[dict[str, object]]]:
    cases: list[tuple[str, ASGIApplication, str, str, dict[str, object]]] = [
        (
            "readiness_unavailable",
            app_factory(
                _SyntheticPorts(readiness=EvidenceReadinessCode.CAS_UNAVAILABLE),
                telemetry_enabled=True,
                admission=_admission(step_seconds=0.5),
            ),
            "GET",
            "/health/ready",
            {},
        ),
        (
            "sqlite_busy",
            app_factory(
                _SyntheticPorts(fault=_PortFault.BUSY),
                telemetry_enabled=True,
                admission=_admission(step_seconds=0.5),
            ),
            "GET",
            "/api/v1/runs",
            {},
        ),
        (
            "corrupt_manifest",
            app_factory(
                _SyntheticPorts(
                    fault=_PortFault.CORRUPT_MANIFEST,
                    canary_marker=canary_marker,
                ),
                telemetry_enabled=True,
                admission=_admission(step_seconds=0.5),
            ),
            "GET",
            "/api/v1/runs/r1_c2VydmljZS1iZW5jaG1hcmstcnVuLTAwMA/diagnostics",
            {},
        ),
        (
            "oversized_query",
            app_factory(
                _SyntheticPorts(),
                telemetry_enabled=True,
                admission=_admission(step_seconds=0.5),
            ),
            "GET",
            "/health/live?" + "q" * (4 * 1024 + 1),
            {},
        ),
        (
            "forbidden_body",
            app_factory(
                _SyntheticPorts(),
                telemetry_enabled=True,
                admission=_admission(step_seconds=0.5),
            ),
            "GET",
            "/health/live",
            {"content": b"synthetic-forbidden-body"},
        ),
    ]
    before = _resource_snapshot()
    started = time.perf_counter()
    samples: list[_RequestSample] = []
    details: list[dict[str, object]] = []
    for name, app, method, path, kwargs in cases:
        lifespan, client = await _open_client(app)
        try:
            sample = await _request(client, method, path, **kwargs)
        finally:
            await _close_client(lifespan, client)
        samples.append(sample)
        details.append(
            {
                "case": name,
                "status": sample.status_code,
                "closed_code": sample.problem_code,
                "latency_ms": sample.latency_ms,
                "response_bytes": sample.response_bytes,
            }
        )
    wall_seconds = time.perf_counter() - started
    after = _resource_snapshot()
    return (
        _scenario_record(
            name="fault_injection",
            display_name="Injected bounded faults",
            telemetry_mode="enabled",
            samples=samples,
            warmup_requests=0,
            before=before,
            after=after,
            wall_seconds=wall_seconds,
        ),
        details,
    )


def _metric_snapshot(payload: bytes) -> dict[str, object]:
    if len(payload) > MAX_METRICS_BYTES:
        raise RuntimeError("metrics exposition exceeded the fixed byte bound")
    try:
        text = payload.decode("ascii")
    except UnicodeDecodeError as error:
        raise RuntimeError("metrics exposition was not bounded ASCII") from error
    series_lines = [
        line for line in text.splitlines() if line and not line.startswith("#") and " " in line
    ]
    if len(series_lines) > MAX_METRIC_SERIES:
        raise RuntimeError("metrics exposition exceeded the fixed cardinality bound")
    drop_total = 0.0
    for line in series_lines:
        name_and_labels, raw_value, *_ = line.split()
        if "drop" not in name_and_labels.lower():
            continue
        try:
            value = float(raw_value)
        except ValueError as error:
            raise RuntimeError("metrics drop counter was not numeric") from error
        if not math.isfinite(value) or value < 0.0:
            raise RuntimeError("metrics drop counter was outside its domain")
        drop_total += value
    return {
        "exposition_bytes": len(payload),
        "series_count": len(series_lines),
        "telemetry_drop_total": drop_total,
        "content_sha256": hashlib.sha256(payload).hexdigest(),
    }


def _dependency_versions() -> dict[str, str]:
    result: dict[str, str] = {}
    for package in ("asgi-lifespan", "fastapi", "httpx", "seaborn", "starlette", "uvicorn"):
        try:
            result[package] = version(package)
        except PackageNotFoundError:
            result[package] = "not-installed"
    return result


def _physical_memory_bytes() -> int | None:
    try:
        page_size = os.sysconf("SC_PAGE_SIZE")
        page_count = os.sysconf("SC_PHYS_PAGES")
    except (OSError, TypeError, ValueError):
        return None
    if type(page_size) is not int or type(page_count) is not int:
        return None
    if page_size <= 0 or page_count <= 0 or page_size > (2**63 - 1) // page_count:
        return None
    return page_size * page_count


def _service_source_files(root: Path) -> tuple[Path, ...]:
    """Return the complete deterministic source closure used by the service image."""

    tracking = root / "src" / "quant_platform" / "tracking"
    return (
        *sorted((root / "src" / "quant_platform" / "service").glob("*.py")),
        *(
            tracking / name
            for name in (
                "__init__.py",
                "cas.py",
                "contracts.py",
                "migrations.py",
                "read_ports.py",
                "registry.py",
                "retention.py",
            )
        ),
        root / "pyproject.toml",
        root / "uv.lock",
        root / "scripts" / "benchmark_service_operability.py",
        root / "scripts" / "plot_service_operability.py",
    )


def _source_tree_sha256_at(root: Path) -> str:
    """Hash one explicit service source root with path and byte framing."""

    digest = hashlib.sha256()
    for path in _service_source_files(root):
        relative = path.relative_to(root).as_posix().encode("utf-8")
        payload = path.read_bytes()
        digest.update(len(relative).to_bytes(4, "big"))
        digest.update(relative)
        digest.update(len(payload).to_bytes(8, "big"))
        digest.update(payload)
    return digest.hexdigest()


def _source_tree_sha256() -> str:
    return _source_tree_sha256_at(Path(__file__).resolve().parents[1])


def _integrity_payload(document: dict[str, object]) -> bytes:
    without_integrity = {key: value for key, value in document.items() if key != "integrity"}
    return json.dumps(
        without_integrity,
        ensure_ascii=True,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("ascii")


def _with_integrity(document: dict[str, object]) -> dict[str, object]:
    result = dict(document)
    result["integrity"] = {
        "algorithm": "sha256",
        "canonicalization": "sorted compact ASCII JSON excluding the integrity member",
        "canonical_payload_sha256": hashlib.sha256(_integrity_payload(result)).hexdigest(),
    }
    return result


class _IsolatedRssProtocolError(RuntimeError):
    """A bounded child process violated the private RSS measurement protocol."""


def _benchmark_exporter_config() -> ExporterConfig:
    """Return the canonical enabled-arm exporter configuration."""

    return ExporterConfig(
        endpoint="http://127.0.0.1:4318/benchmark",
        timeout_seconds=0.25,
        max_retries=0,
        queue_capacity=256,
        batch_size=32,
        shutdown_timeout_seconds=2.0,
    )


def _isolated_rss_workload_contract() -> dict[str, object]:
    """Describe every common input used by both isolated telemetry arms."""

    exporter = _benchmark_exporter_config()
    return {
        "assembly": "reference_app_factory",
        "transport": "in-process HTTPX ASGI transport",
        "route_cycle": list(_STEADY_ROUTES),
        "measured_requests": STEADY_REQUESTS_PER_ROUND,
        "warmup_requests": WARMUP_REQUESTS,
        "admission_clock_step_seconds": 0.5,
        "synthetic_fixture_time": EVIDENCE_TIME.isoformat().replace("+00:00", "Z"),
        "synthetic_runs": SYNTHETIC_RUNS,
        "synthetic_diagnostics": SYNTHETIC_DIAGNOSTICS,
        "telemetry_axis": {
            "disabled": "fixed metrics; asynchronous record queues and export disabled",
            "enabled": "fixed metrics plus bounded asynchronous records and local sink",
        },
        "enabled_exporter": {
            "endpoint": exporter.endpoint,
            "timeout_seconds": exporter.timeout_seconds,
            "max_retries": exporter.max_retries,
            "queue_capacity": exporter.queue_capacity,
            "batch_size": exporter.batch_size,
            "shutdown_timeout_seconds": exporter.shutdown_timeout_seconds,
            "worker_termination_timeout_seconds": exporter.worker_termination_timeout_seconds,
            "sink": "network-free aggregate sink",
        },
    }


def _isolated_rss_workload_sha256() -> str:
    payload = json.dumps(
        _isolated_rss_workload_contract(),
        ensure_ascii=True,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("ascii")
    return hashlib.sha256(payload).hexdigest()


def _bounded_protocol_text(value: object, field: str, *, maximum_bytes: int = 256) -> str:
    if type(value) is not str:
        raise _IsolatedRssProtocolError(f"isolated RSS {field} must be text")
    try:
        payload = value.encode("ascii")
    except UnicodeEncodeError as error:
        raise _IsolatedRssProtocolError(f"isolated RSS {field} must use bounded ASCII") from error
    if (
        not payload
        or len(payload) > maximum_bytes
        or any(byte < 0x20 or byte > 0x7E for byte in payload)
    ):
        raise _IsolatedRssProtocolError(f"isolated RSS {field} is outside its bound")
    return value


def _bounded_protocol_integer(
    value: object,
    field: str,
    *,
    minimum: int = 0,
    maximum: int = 2**63 - 1,
) -> int:
    if type(value) is not int or not minimum <= value <= maximum:
        raise _IsolatedRssProtocolError(f"isolated RSS {field} is outside its bound")
    return value


def _validate_isolated_rss_child_result(
    document: object,
    *,
    expected_mode: str,
) -> dict[str, object]:
    """Validate one untrusted child result against the exact closed schema."""

    if expected_mode not in _ISOLATED_RSS_MODES:
        raise ValueError("expected isolated RSS mode is unsupported")
    if type(document) is not dict:
        raise _IsolatedRssProtocolError("isolated RSS result must be an object")
    result = dict(document)
    expected_fields = {
        "protocol",
        "mode",
        "workload_sha256",
        "runtime_platform",
        "python",
        "python_implementation",
        "rss_source",
        "rss_unit",
        "rss_before_app_bytes",
        "rss_after_shutdown_peak_bytes",
        "workload_growth_bytes",
        "warmup_requests",
        "measured_requests",
        "successful_requests",
        "status_counts",
        "lifespan_shutdown_completed",
        "network_requests",
        "processes",
    }
    if set(result) != expected_fields:
        raise _IsolatedRssProtocolError("isolated RSS result fields violated the closed schema")
    if result["protocol"] != _ISOLATED_RSS_PROTOCOL or result["mode"] != expected_mode:
        raise _IsolatedRssProtocolError("isolated RSS protocol or mode is incompatible")
    workload_sha256 = _bounded_protocol_text(result["workload_sha256"], "workload identity")
    if (
        re.fullmatch(r"[0-9a-f]{64}", workload_sha256) is None
        or workload_sha256 != _isolated_rss_workload_sha256()
    ):
        raise _IsolatedRssProtocolError("isolated RSS workload identity is incompatible")
    runtime_platform = _bounded_protocol_text(result["runtime_platform"], "runtime platform")
    python = _bounded_protocol_text(result["python"], "Python version")
    python_implementation = _bounded_protocol_text(
        result["python_implementation"], "Python implementation"
    )
    if (
        runtime_platform != f"{platform.system()}-{platform.release()}-{platform.machine()}"
        or python != platform.python_version()
        or python_implementation != platform.python_implementation()
    ):
        raise _IsolatedRssProtocolError("isolated RSS runtime is incompatible")
    if result["rss_source"] != "resource.getrusage(RUSAGE_SELF).ru_maxrss":
        raise _IsolatedRssProtocolError("isolated RSS source is incompatible")
    if result["rss_unit"] != "bytes":
        raise _IsolatedRssProtocolError("isolated RSS unit is incompatible")

    before = _bounded_protocol_integer(
        result["rss_before_app_bytes"], "pre-application peak", minimum=1
    )
    after = _bounded_protocol_integer(
        result["rss_after_shutdown_peak_bytes"], "post-shutdown peak", minimum=1
    )
    growth = _bounded_protocol_integer(result["workload_growth_bytes"], "workload growth")
    if after < before or growth != after - before:
        raise _IsolatedRssProtocolError("isolated RSS peak chronology is invalid")
    warmup_requests = _bounded_protocol_integer(result["warmup_requests"], "warm-up count")
    measured_requests = _bounded_protocol_integer(
        result["measured_requests"], "measured count", minimum=1
    )
    successful_requests = _bounded_protocol_integer(
        result["successful_requests"], "successful count"
    )
    if (
        warmup_requests != WARMUP_REQUESTS
        or measured_requests != STEADY_REQUESTS_PER_ROUND
        or successful_requests != measured_requests
        or result["status_counts"] != {"200": measured_requests}
    ):
        raise _IsolatedRssProtocolError("isolated RSS workload outcome is incompatible")
    if result["lifespan_shutdown_completed"] is not True:
        raise _IsolatedRssProtocolError("isolated RSS child did not complete lifespan shutdown")
    if result["network_requests"] != 0 or type(result["network_requests"]) is not int:
        raise _IsolatedRssProtocolError("isolated RSS child crossed the network boundary")
    if result["processes"] != 1 or type(result["processes"]) is not int:
        raise _IsolatedRssProtocolError("isolated RSS child process count is incompatible")
    return result


async def _isolated_rss_child_measurement(mode: str) -> dict[str, object]:
    """Execute one canonical arm and observe its process-lifetime RSS peak."""

    if mode not in _ISOLATED_RSS_MODES:
        raise ValueError("isolated RSS mode is unsupported")
    paths, warmup = _steady_workload()
    rss_before = _maximum_rss_bytes()
    app = _reference_app_factory(
        _SyntheticPorts(),
        telemetry_enabled=mode == "enabled",
        admission=_admission(step_seconds=0.5),
    )
    samples, _, _, _ = await _measure_requests(app, paths, warmup_paths=warmup)
    rss_after = _maximum_rss_bytes()
    status_counts = Counter(str(sample.status_code) for sample in samples)
    return {
        "protocol": _ISOLATED_RSS_PROTOCOL,
        "mode": mode,
        "workload_sha256": _isolated_rss_workload_sha256(),
        "runtime_platform": f"{platform.system()}-{platform.release()}-{platform.machine()}",
        "python": platform.python_version(),
        "python_implementation": platform.python_implementation(),
        "rss_source": "resource.getrusage(RUSAGE_SELF).ru_maxrss",
        "rss_unit": "bytes",
        "rss_before_app_bytes": rss_before,
        "rss_after_shutdown_peak_bytes": rss_after,
        "workload_growth_bytes": rss_after - rss_before,
        "warmup_requests": len(warmup),
        "measured_requests": len(samples),
        "successful_requests": sum(sample.status_code == 200 for sample in samples),
        "status_counts": dict(sorted(status_counts.items())),
        "lifespan_shutdown_completed": True,
        "network_requests": 0,
        "processes": 1,
    }


def _isolated_rss_child_entrypoint(mode: str) -> int:
    """Serve one exact parent/child handshake without exposing failure details."""

    os.environ.pop("SERVICE_SECRET_CANARY", None)
    try:
        sys.stdout.buffer.write(_ISOLATED_RSS_READY)
        sys.stdout.buffer.flush()
        command = sys.stdin.buffer.readline(len(_ISOLATED_RSS_START) + 1)
        if command != _ISOLATED_RSS_START:
            return 2

        client_logger = logging.getLogger("httpx")
        was_disabled = client_logger.disabled
        client_logger.disabled = True
        try:
            result = asyncio.run(_isolated_rss_child_measurement(mode))
        finally:
            client_logger.disabled = was_disabled
        validated = _validate_isolated_rss_child_result(result, expected_mode=mode)
        payload = json.dumps(
            validated,
            ensure_ascii=True,
            allow_nan=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("ascii")
        line = _ISOLATED_RSS_RESULT_PREFIX + payload + b"\n"
        if len(line) > _ISOLATED_RSS_MAXIMUM_LINE_BYTES:
            return 2
        sys.stdout.buffer.write(line)
        sys.stdout.buffer.flush()
        return 0
    except Exception:
        return 2


def _isolated_child_command(mode: str) -> tuple[str, ...]:
    return (sys.executable, str(Path(__file__).resolve()), "--_isolated-rss-child", mode)


def _isolated_child_environment() -> dict[str, str]:
    """Return a credential-free allowlist for the measurement child."""

    return {
        "PATH": os.environ.get("PATH", os.defpath),
        "LC_ALL": "C",
        "PYTHONHASHSEED": "0",
        "PYTHONDONTWRITEBYTECODE": "1",
        "PYTHONNOUSERSITE": "1",
        "PYTHONUNBUFFERED": "1",
    }


def _read_bounded_protocol_line(
    stream: Any,
    *,
    deadline: float,
) -> bytes:
    """Read one newline-terminated pipe record without exceeding time or size."""

    selector = selectors.DefaultSelector()
    buffer = bytearray()
    try:
        selector.register(stream, selectors.EVENT_READ)
        while True:
            available = _ISOLATED_RSS_MAXIMUM_LINE_BYTES - len(buffer)
            if available <= 0:
                raise _IsolatedRssProtocolError("isolated RSS child exceeded its protocol bound")
            remaining = deadline - time.monotonic()
            if remaining <= 0.0 or not selector.select(remaining):
                raise _IsolatedRssProtocolError("isolated RSS child exceeded its hard deadline")
            chunk = os.read(stream.fileno(), min(4_096, available))
            if not chunk:
                raise _IsolatedRssProtocolError("isolated RSS child closed its protocol early")
            buffer.extend(chunk)
            newline = buffer.find(b"\n")
            if newline >= 0:
                if newline != len(buffer) - 1:
                    raise _IsolatedRssProtocolError(
                        "isolated RSS child emitted an ambiguous protocol record"
                    )
                return bytes(buffer)
    finally:
        selector.close()


def _terminate_and_reap_isolated_child(process: subprocess.Popen[bytes]) -> None:
    """Bound child cleanup so a protocol failure cannot leave work running."""

    try:
        if process.poll() is None:
            with suppress(ProcessLookupError):
                process.terminate()
            try:
                process.wait(timeout=_ISOLATED_RSS_TERMINATION_GRACE_SECONDS)
            except subprocess.TimeoutExpired:
                with suppress(ProcessLookupError):
                    process.kill()
                try:
                    process.wait(timeout=_ISOLATED_RSS_TERMINATION_GRACE_SECONDS)
                except subprocess.TimeoutExpired as error:
                    raise _IsolatedRssProtocolError(
                        "isolated RSS child could not be terminated within its bound"
                    ) from error
        else:
            process.wait()
    finally:
        for stream in (process.stdin, process.stdout):
            if stream is not None:
                stream.close()


def _run_isolated_rss_child(
    mode: str,
    *,
    timeout_seconds: float = _ISOLATED_RSS_HARD_TIMEOUT_SECONDS,
) -> dict[str, object]:
    """Run, hand-shake with, validate, and reap one fresh measurement process."""

    if mode not in _ISOLATED_RSS_MODES:
        raise ValueError("isolated RSS mode is unsupported")
    if not math.isfinite(timeout_seconds) or timeout_seconds <= 0.0:
        raise ValueError("isolated RSS timeout must be finite and positive")
    try:
        process = subprocess.Popen(
            _isolated_child_command(mode),
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            cwd=Path(__file__).resolve().parents[1],
            env=_isolated_child_environment(),
            bufsize=0,
            close_fds=True,
        )
    except OSError as error:
        raise _IsolatedRssProtocolError("unable to start isolated RSS child") from error

    deadline = time.monotonic() + timeout_seconds
    try:
        if process.stdin is None or process.stdout is None:
            raise _IsolatedRssProtocolError("isolated RSS child pipes are unavailable")
        ready = _read_bounded_protocol_line(process.stdout, deadline=deadline)
        if ready != _ISOLATED_RSS_READY:
            raise _IsolatedRssProtocolError("isolated RSS child handshake is incompatible")
        try:
            process.stdin.write(_ISOLATED_RSS_START)
            process.stdin.flush()
            process.stdin.close()
        except (BrokenPipeError, OSError) as error:
            raise _IsolatedRssProtocolError(
                "isolated RSS child rejected its start signal"
            ) from error

        line = _read_bounded_protocol_line(process.stdout, deadline=deadline)
        remaining = deadline - time.monotonic()
        if remaining <= 0.0:
            raise _IsolatedRssProtocolError("isolated RSS child exceeded its hard deadline")
        try:
            return_code = process.wait(timeout=remaining)
        except subprocess.TimeoutExpired as error:
            raise _IsolatedRssProtocolError(
                "isolated RSS child exceeded its hard deadline"
            ) from error
        if return_code != 0 or not line.startswith(_ISOLATED_RSS_RESULT_PREFIX):
            raise _IsolatedRssProtocolError("isolated RSS child result is unavailable")
        payload = line[len(_ISOLATED_RSS_RESULT_PREFIX) : -1]
        try:
            document = json.loads(payload.decode("ascii"))
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise _IsolatedRssProtocolError("isolated RSS child result is malformed") from error
        return _validate_isolated_rss_child_result(document, expected_mode=mode)
    finally:
        _terminate_and_reap_isolated_child(process)


def _aggregate_isolated_rss_measurements(
    disabled: object,
    enabled: object,
) -> dict[str, object]:
    """Build a compatible fresh-process delta while retaining its signed value."""

    validated_disabled = _validate_isolated_rss_child_result(disabled, expected_mode="disabled")
    validated_enabled = _validate_isolated_rss_child_result(enabled, expected_mode="enabled")
    disabled_peak = int(validated_disabled["rss_after_shutdown_peak_bytes"])
    enabled_peak = int(validated_enabled["rss_after_shutdown_peak_bytes"])
    signed_delta = enabled_peak - disabled_peak
    nonnegative_delta = max(0, signed_delta)
    if nonnegative_delta > MAX_TELEMETRY_RSS_DELTA_BYTES:
        raise RuntimeError("isolated telemetry RSS delta exceeded the fixed 100 MiB bound")
    return {
        "method": "fresh_process_peak_rss_comparison",
        "protocol": _ISOLATED_RSS_PROTOCOL,
        "process_order": list(_ISOLATED_RSS_MODES),
        "hard_timeout_seconds_per_process": _ISOLATED_RSS_HARD_TIMEOUT_SECONDS,
        "common_workload": _isolated_rss_workload_contract(),
        "workload_sha256": _isolated_rss_workload_sha256(),
        "disabled": validated_disabled,
        "enabled": validated_enabled,
        "comparison_compatible": True,
        "observed_signed_delta_bytes": signed_delta,
        "nonnegative_delta_bytes": nonnegative_delta,
        "limit_bytes": MAX_TELEMETRY_RSS_DELTA_BYTES,
        "within_limit": True,
        "negative_delta_policy": (
            "retain the signed observation; clamp only the incremental-overhead guard to zero"
        ),
    }


def _measure_isolated_telemetry_rss() -> dict[str, object]:
    """Measure disabled then enabled telemetry in separate canonical processes."""

    disabled = _run_isolated_rss_child("disabled")
    enabled = _run_isolated_rss_child("enabled")
    return _aggregate_isolated_rss_measurements(disabled, enabled)


async def _run_benchmark(
    app_factory: AppFactory,
    isolated_rss: dict[str, object],
) -> dict[str, object]:
    canary_marker = _service_canary_from_environment()
    measured_at = datetime.now(UTC)
    cold = await _cold_scenario(app_factory)
    steady_disabled, steady_enabled, steady_schedule = await _steady_ab_scenarios(app_factory)
    maximum = await _maximum_scenario(app_factory)
    saturation, saturation_detail = await _saturation_scenario(app_factory)
    metrics, metrics_payload = await _metrics_scenario(app_factory)
    faults, fault_details = await _fault_scenario(
        app_factory,
        canary_marker=canary_marker,
    )
    scenarios = [
        cold,
        steady_disabled,
        steady_enabled,
        maximum,
        saturation,
        metrics,
        faults,
    ]
    metric_snapshot = _metric_snapshot(metrics_payload)
    maximum_response = max(float(dict(scenario["response_bytes"])["max"]) for scenario in scenarios)
    maximum_rejection_latency = float(saturation_detail["maximum_rejection_latency_ms"])
    evidence = {
        "schema_version": SCHEMA_VERSION,
        "evidence_id": "signalattice-service-operability-synthetic-local-v1",
        "evidence_class": "measured_synthetic_local_engineering",
        "generated_at": measured_at.isoformat(timespec="seconds").replace("+00:00", "Z"),
        "provenance": {
            "network_requests": 0,
            "credentials_used": False,
            "market_or_model_data_used": False,
            "raw_request_samples_committed": False,
            "source_tree_sha256": _source_tree_sha256(),
        },
        "environment": {
            "platform": f"{platform.system()}-{platform.release()}-{platform.machine()}",
            "python": platform.python_version(),
            "python_implementation": platform.python_implementation(),
            "processor": platform.processor() or "unreported",
            "logical_cpu_count": os.cpu_count(),
            "physical_memory_bytes": _physical_memory_bytes(),
            "execution_context": "local_host_process",
            "virtualization": "not_detected_by_benchmark",
            "rss_unit_interpretation": (
                "bytes" if sys.platform == "darwin" else "KiB converted to bytes"
            ),
            "dependencies": _dependency_versions(),
        },
        "profile": {
            "transport_concurrency": 32,
            "data_concurrency": 24,
            "api_rate_per_second": 20.0,
            "api_burst": 40,
            "operations_rate_per_second": 2.0,
            "operations_burst": 4,
            "query_bytes": 4 * 1024,
            "cursor_bytes": 1024,
            "header_bytes": 16 * 1024,
            "response_bytes": MAX_RESPONSE_BYTES,
            "metrics_bytes": MAX_METRICS_BYTES,
            "metric_series_strict_upper_bound": 600,
        },
        "workload": {
            "seed": SEED,
            "synthetic_fixture_time": EVIDENCE_TIME.isoformat().replace("+00:00", "Z"),
            "warmup_requests": WARMUP_REQUESTS,
            "scenario_order": [str(scenario["name"]) for scenario in scenarios],
            "synthetic_runs": SYNTHETIC_RUNS,
            "synthetic_diagnostics": SYNTHETIC_DIAGNOSTICS,
            "sample_counts": {
                str(scenario["name"]): int(scenario["sample_count"]) for scenario in scenarios
            },
        },
        "candidate_slos": {
            "status": "aspirational_until_28_day_continuous_evidence",
            "interactive_p95_ms": 100,
            "interactive_p99_ms": 250,
            "maximum_page_diagnostic_p99_ms": 500,
            "readiness_p99_ms": 100,
            "excess_load_rejection_ms": 250,
            "corrupt_unverified_successes": 0,
            "semantics_document": "docs/service_operations.md#candidate-28-day-slo-semantics",
        },
        "telemetry_ab": {
            "disabled": (
                "fixed in-process metrics enabled; asynchronous record queues and export disabled"
            ),
            "enabled": (
                "fixed metrics plus bounded asynchronous records delivered to a network-free "
                "aggregate sink"
            ),
            "rounds": STEADY_AB_ROUNDS,
            "requests_per_round": STEADY_REQUESTS_PER_ROUND,
            "warmup_requests_per_round": WARMUP_REQUESTS,
            "balanced_round_order": steady_schedule,
            "isolated_process_rss": isolated_rss,
        },
        "scenarios": scenarios,
        "saturation": saturation_detail,
        "fault_cases": fault_details,
        "telemetry": {
            **metric_snapshot,
            "concurrent_scrape_requests": CONCURRENT_METRICS_REQUESTS,
            "exporter_mode": "bounded local benchmark sink; no network",
        },
        "bounds_observed": [
            {
                "name": "maximum_response_bytes",
                "display_name": "Largest response",
                "observed": maximum_response,
                "limit": MAX_RESPONSE_BYTES,
                "unit": "bytes",
            },
            {
                "name": "metrics_exposition_bytes",
                "display_name": "Metrics exposition",
                "observed": metric_snapshot["exposition_bytes"],
                "limit": MAX_METRICS_BYTES,
                "unit": "bytes",
            },
            {
                "name": "metric_series",
                "display_name": "Metric series",
                "observed": metric_snapshot["series_count"],
                "limit": 600,
                "unit": "series (strictly below)",
            },
            {
                "name": "rejection_latency_ms",
                "display_name": "Slowest overload rejection",
                "observed": maximum_rejection_latency,
                "limit": MAX_REJECTION_MILLISECONDS,
                "unit": "ms",
            },
            {
                "name": "telemetry_isolated_rss_delta_bytes",
                "display_name": "Telemetry RSS delta",
                "observed": isolated_rss["nonnegative_delta_bytes"],
                "limit": MAX_TELEMETRY_RSS_DELTA_BYTES,
                "unit": "bytes",
            },
        ],
        "limitations": [
            "In-process HTTPX ASGI transport; no kernel socket, TLS, proxy, or network timing.",
            "Primary scenarios use a single local process; the RSS A/B uses one fresh child per arm. Host-dependent timing and resource values are not portable.",
            "ru_maxrss is a process-lifetime peak, not current RSS or telemetry-object attribution; the signed disabled/enabled difference can be noisy.",
            "Deterministic synthetic aggregate fixtures; no market observations, models, or licensed data.",
            "Short bounded samples cannot establish the aspirational 28-day candidate SLOs.",
            "No production, market-scale, paper-trading, live-trading, capital, or profitability claim.",
            "Raw request samples are intentionally not committed; aggregate percentiles retain variability.",
        ],
    }
    if maximum_response > MAX_RESPONSE_BYTES:
        raise RuntimeError("observed response exceeded the fixed byte bound")
    if maximum_rejection_latency > MAX_REJECTION_MILLISECONDS:
        raise RuntimeError("overload rejection exceeded the fixed latency bound")
    if int(isolated_rss["nonnegative_delta_bytes"]) > MAX_TELEMETRY_RSS_DELTA_BYTES:
        raise RuntimeError("isolated telemetry RSS delta exceeded the fixed 100 MiB bound")
    bounded_evidence = _with_integrity(evidence)
    if canary_marker is not None and canary_marker.encode("ascii") in _integrity_payload(
        bounded_evidence
    ):
        raise RuntimeError("synthetic canary crossed the aggregate evidence boundary")
    return bounded_evidence


def _service_canary_from_environment() -> str | None:
    """Read one workflow-only synthetic marker without logging or persisting it."""

    marker = os.environ.get("SERVICE_SECRET_CANARY")
    if marker is None:
        return None
    if _SERVICE_CANARY_PATTERN.fullmatch(marker) is None:
        raise RuntimeError("SERVICE_SECRET_CANARY violates its bounded synthetic contract")
    return marker


def run_benchmark(app_factory: AppFactory) -> dict[str, object]:
    """Run every bounded scenario with the canonical production app assembly."""

    if app_factory is not _reference_app_factory:
        raise ValueError("isolated RSS evidence requires the canonical reference app factory")
    isolated_rss = _measure_isolated_telemetry_rss()

    # HTTPX's convenience access logger includes the full request URL.  The
    # workload deliberately includes hostile query input, so suppress that
    # client-side logger just as the production Uvicorn profile suppresses its
    # access log.  Restore caller state after the bounded run.
    client_logger = logging.getLogger("httpx")
    was_disabled = client_logger.disabled
    client_logger.disabled = True
    try:
        return asyncio.run(_run_benchmark(app_factory, isolated_rss))
    finally:
        client_logger.disabled = was_disabled


def _reference_app_factory(
    ports: EvidenceReadPort,
    *,
    telemetry_enabled: bool,
    admission: AdmissionController,
) -> ASGIApplication:
    """Assemble production admission/telemetry with a network-free benchmark sink."""

    metrics = ServiceMetrics()
    if telemetry_enabled:
        config = _benchmark_exporter_config()
        telemetry = ServiceTelemetry(metrics, config, _AggregateSinkExporter())
    else:
        telemetry = ServiceTelemetry(metrics)
    return create_app(ports, admission=admission, telemetry=telemetry)


def _file_identity(value: os.stat_result) -> tuple[int, int, int, int, int, int, int]:
    """Return fields that expose replacement or mutation during a bounded read."""

    return (
        value.st_dev,
        value.st_ino,
        value.st_mode,
        value.st_nlink,
        value.st_size,
        value.st_mtime_ns,
        value.st_ctime_ns,
    )


def _read_existing_publication(destination: Path, *, maximum_bytes: int) -> bytes:
    """Read one existing regular destination without following or blocking on it."""

    descriptor: int | None = None
    flags = (
        os.O_RDONLY
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0)
        | getattr(os, "O_NONBLOCK", 0)
    )
    try:
        descriptor = os.open(destination, flags)
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            raise ValueError("evidence destination must be a regular file")
        if stat.S_IMODE(before.st_mode) != _PUBLICATION_MODE:
            raise ValueError("existing evidence destination lacks canonical 0644 permissions")
        if before.st_size > maximum_bytes:
            raise ValueError("existing evidence destination exceeds the publication bound")

        chunks: list[bytes] = []
        observed = 0
        while True:
            chunk = os.read(
                descriptor,
                min(_PUBLICATION_CHUNK_BYTES, maximum_bytes + 1 - observed),
            )
            if not chunk:
                break
            chunks.append(chunk)
            observed += len(chunk)
            if observed > maximum_bytes:
                raise ValueError("existing evidence destination exceeds the publication bound")

        after = os.fstat(descriptor)
        rebound = os.stat(destination, follow_symlinks=False)
        if (
            _file_identity(before) != _file_identity(after)
            or _file_identity(after) != _file_identity(rebound)
            or observed != after.st_size
        ):
            raise ValueError("evidence destination changed during verification")
        return b"".join(chunks)
    except ValueError:
        raise
    except OSError as error:
        raise ValueError("unable to verify existing evidence destination") from error
    finally:
        if descriptor is not None:
            os.close(descriptor)


def _fsync_directory(directory: Path) -> None:
    """Persist the no-replace directory entry before reporting publication success."""

    descriptor = os.open(
        directory,
        os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_DIRECTORY", 0),
    )
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _publish_bytes_no_replace(payload: bytes, destination: Path) -> None:
    """Publish bounded bytes once, or verify an identical regular destination.

    The hard-link operation is the atomic no-replace commit point.  A concurrent
    creator therefore wins without being overwritten; only byte-identical,
    canonically permissioned output is accepted as an idempotent no-op.
    """

    if type(payload) is not bytes or not payload:
        raise ValueError("evidence publication payload must be non-empty bytes")
    if len(payload) > MAX_EVIDENCE_FILE_BYTES:
        raise ValueError("evidence publication payload exceeds the 2 MiB bound")

    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.",
        suffix=".tmp",
        dir=destination.parent,
    )
    temporary = Path(temporary_name)
    descriptor_open = True
    try:
        with os.fdopen(descriptor, "wb") as output:
            descriptor_open = False
            output.write(payload)
            output.flush()
            os.fchmod(output.fileno(), _PUBLICATION_MODE)
            os.fsync(output.fileno())
        try:
            os.link(temporary, destination, follow_symlinks=False)
        except FileExistsError:
            existing = _read_existing_publication(
                destination,
                maximum_bytes=MAX_EVIDENCE_FILE_BYTES,
            )
            if not hmac.compare_digest(existing, payload):
                raise ValueError("existing evidence destination contains different bytes") from None
        except OSError as error:
            raise ValueError("unable to publish evidence without replacement") from error
        else:
            _fsync_directory(destination.parent)
    finally:
        if descriptor_open:
            os.close(descriptor)
        temporary.unlink(missing_ok=True)


def write_evidence(document: dict[str, object], destination: Path) -> None:
    """Publish canonical aggregate JSON without replacing any existing destination."""

    payload = (
        json.dumps(
            document,
            ensure_ascii=True,
            allow_nan=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("ascii")
        + b"\n"
    )
    _publish_bytes_no_replace(payload, destination)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate aggregate synthetic/local evidence for the bounded read service."
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--_isolated-rss-child",
        choices=_ISOLATED_RSS_MODES,
        help=argparse.SUPPRESS,
    )
    arguments = parser.parse_args()
    if arguments._isolated_rss_child is not None:
        if arguments.output is not None:
            parser.error("--output cannot be combined with the internal child mode")
        raise SystemExit(_isolated_rss_child_entrypoint(arguments._isolated_rss_child))
    if arguments.output is None:
        parser.error("--output is required")
    evidence = run_benchmark(_reference_app_factory)
    write_evidence(evidence, arguments.output)


if __name__ == "__main__":
    main()
