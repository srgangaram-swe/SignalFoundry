#!/usr/bin/env python3
"""Fail-closed verifier for the Signalattice service-image evidence boundary.

The verifier treats Docker and scanner output as untrusted input. It never
renders finding identifiers, package names, filesystem paths, scanner stderr,
or canary values. Successful output is a bounded aggregate JSON document;
failures expose only a stable contract-level reason.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import stat
import subprocess
import sys
import tarfile
import tempfile
from dataclasses import dataclass
from json import JSONDecodeError
from pathlib import Path, PurePosixPath
from typing import Any, Final

MAX_IMAGE_BYTES: Final = 1_342_177_280  # 1.25 GiB, uncompressed Docker Size.
MAX_TELEMETRY_SOURCE_ARTIFACT_BYTES: Final = 104_857_600  # 100 MiB.
MAX_TELEMETRY_RSS_DELTA_BYTES: Final = 104_857_600  # 100 MiB.
MAX_JSON_BYTES: Final = 64 * 1024 * 1024
MAX_TAR_MEMBERS: Final = 300_000
MAX_TAR_MEMBER_BYTES: Final = 512 * 1024 * 1024
MAX_TAR_LOGICAL_BYTES: Final = MAX_IMAGE_BYTES
MAX_HISTORY_RECORDS: Final = 256
MAX_HISTORY_RECORD_BYTES: Final = 1024 * 1024
MAX_JSON_NODES: Final = 2_000_000
MAX_RUNTIME_LOG_BYTES: Final = 8 * 1024 * 1024
EXPECTED_SERVICE_PIDS: Final = 64
MIN_SATURATED_SERVICE_PIDS: Final = 26
MAX_SATURATED_SERVICE_PIDS: Final = 60
EXPECTED_PLATFORM: Final = "linux/amd64"
EXPECTED_SERVICE_ENTRYPOINT: Final = ["python", "-m", "quant_platform.service"]
EXPECTED_SERVICE_COMMAND: Final = [
    "--registry-db",
    "/var/lib/signalattice/registry/registry.sqlite3",
    "--cas-root",
    "/var/lib/signalattice/cas",
    "--socket-path",
    "/run/signalattice/api.sock",
    "--digest-key-file",
    "/run/secrets/registry-digest-key",
]
EXPECTED_HEALTHCHECK_TEST: Final = [
    "CMD",
    "python",
    "-c",
    (
        "import socket; s=socket.socket(socket.AF_UNIX); s.settimeout(1); "
        "s.connect('/run/signalattice/api.sock'); s.sendall(b'GET /health/ready "
        "HTTP/1.1\\r\\nHost: localhost\\r\\nConnection: close\\r\\n\\r\\n'); "
        "f=s.makefile('rb'); line=f.readline(4097); f.close(); s.close(); "
        "parts=line.split(b' ',2); raise SystemExit(0 if len(line)<=4096 and "
        "line.endswith(b'\\r\\n') and len(parts)>=2 and parts[0] in "
        "(b'HTTP/1.0',b'HTTP/1.1') and parts[1]==b'200' else 1)"
    ),
]
_DIGEST = re.compile(r"^sha256:[0-9a-f]{64}$")
_ARTIFACT_ID = re.compile(r"^(?:sha256:)?[0-9a-f]{64}$")
_SOURCE_REVISION = re.compile(r"^[0-9a-f]{40}$")
_IMAGE_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:/@-]{0,255}$")
_ENV_NAME = re.compile(r"^[A-Z][A-Z0-9_]{0,63}$")
_TRIVY_IMAGE_ARCHIVE_NAME: Final = "/scan/service-image.tar"
_FORBIDDEN_LICENSE = re.compile(
    r"(?:AGPL|Affero|SSPL|Server[ -]Side[ -]Public|Commons[ -]Clause|"
    r"Business[ -]Source|BUSL|BSL-1\.1|Elastic[ -]License)",
    re.IGNORECASE,
)
_FORBIDDEN_HISTORY = re.compile(
    rb"(?:BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY|https?://[^\s:@]+:[^\s@]+@)",
    re.IGNORECASE,
)
_WORKSTATION_PATH = re.compile(rb"/(?:Users|home/runner/work)/[A-Za-z0-9_.-]+/")
_UNRESOLVED_LICENSES: Final = frozenset({"", "NONE", "NOASSERTION", "UNKNOWN"})
_ACCEPTED_LICENSE_CATEGORIES: Final = frozenset(
    {"NOTICE", "PERMISSIVE", "RECIPROCAL", "UNENCUMBERED"}
)
_DOCKER_RUNTIME_INJECTED_PATHS: Final = frozenset({"etc/hostname", "etc/hosts", "etc/resolv.conf"})
_TELEMETRY_SOURCE_ARTIFACT_SUFFIXES: Final = frozenset(
    {
        "/quant_platform/service/exporter.py",
        "/quant_platform/service/metrics.py",
        "/quant_platform/service/telemetry.py",
        "/quant_platform/service/telemetry_contracts.py",
    }
)
_EXPECTED_SERVICE_SOURCE_FILES: Final = frozenset(
    {
        "app/src/quant_platform/__init__.py",
        "app/src/quant_platform/py.typed",
        "app/src/quant_platform/service/__init__.py",
        "app/src/quant_platform/service/__main__.py",
        "app/src/quant_platform/service/admission.py",
        "app/src/quant_platform/service/api.py",
        "app/src/quant_platform/service/contracts.py",
        "app/src/quant_platform/service/entrypoint.py",
        "app/src/quant_platform/service/exporter.py",
        "app/src/quant_platform/service/http_protocol.py",
        "app/src/quant_platform/service/manifests.py",
        "app/src/quant_platform/service/metrics.py",
        "app/src/quant_platform/service/middleware.py",
        "app/src/quant_platform/service/models.py",
        "app/src/quant_platform/service/problems.py",
        "app/src/quant_platform/service/server.py",
        "app/src/quant_platform/service/telemetry.py",
        "app/src/quant_platform/service/telemetry_contracts.py",
        "app/src/quant_platform/tracking/__init__.py",
        "app/src/quant_platform/tracking/cas.py",
        "app/src/quant_platform/tracking/contracts.py",
        "app/src/quant_platform/tracking/migrations.py",
        "app/src/quant_platform/tracking/read_ports.py",
        "app/src/quant_platform/tracking/registry.py",
        "app/src/quant_platform/tracking/retention.py",
    }
)
_EXPECTED_SERVICE_DISTRIBUTIONS: Final = frozenset(
    {
        "annotated_doc-0.0.4.dist-info",
        "annotated_types-0.8.0.dist-info",
        "anyio-4.14.2.dist-info",
        "click-8.4.2.dist-info",
        "fastapi-0.141.1.dist-info",
        "h11-0.16.0.dist-info",
        "idna-3.18.dist-info",
        "pydantic-2.13.4.dist-info",
        "pydantic_core-2.46.4.dist-info",
        "starlette-1.3.1.dist-info",
        "typing_extensions-4.16.0.dist-info",
        "typing_inspection-0.4.2.dist-info",
        "uvicorn-0.52.1.dist-info",
    }
)
_EXPECTED_SITE_PACKAGE_ROOTS: Final = _EXPECTED_SERVICE_DISTRIBUTIONS | frozenset(
    {
        "_virtualenv.pth",
        "_virtualenv.py",
        "annotated_doc",
        "annotated_types",
        "anyio",
        "click",
        "fastapi",
        "h11",
        "idna",
        "pydantic",
        "pydantic_core",
        "starlette",
        "typing_extensions.py",
        "typing_inspection",
        "uvicorn",
    }
)
_FORBIDDEN_APPLICATION_DIRECTORY_NAMES: Final = frozenset(
    {"dataset", "datasets", "fixture", "fixtures", "sample", "samples", "test", "tests"}
)
_REVIEWED_CONTAINER_LICENSE_FINDINGS: Final = frozenset(
    {
        (
            "opt/venv/lib/python3.13/site-packages/"
            "typing_extensions-4.16.0.dist-info/licenses/LICENSE",
            "BSD-0-Clause",
        ),
        (
            "opt/venv/lib/python3.13/site-packages/"
            "typing_extensions-4.16.0.dist-info/licenses/LICENSE",
            "BeOpen",
        ),
        (
            "opt/venv/lib/python3.13/site-packages/"
            "typing_extensions-4.16.0.dist-info/licenses/LICENSE",
            "CNRI-Python-GPL-Compatible",
        ),
    }
)


class VerificationError(RuntimeError):
    """One evidence artifact violated the closed service-image contract."""


@dataclass(frozen=True, slots=True)
class ImageSnapshot:
    """Sanitized, deterministic evidence extracted from one image."""

    application_digest: str
    application_regular_bytes: int
    config_digest: str
    filesystem_digest: str
    image_size_bytes: int
    history_record_count: int
    image_config_digest: str
    layer_count: int
    application_without_telemetry_source_artifacts_digest: str
    application_without_telemetry_source_artifacts_regular_bytes: int
    regular_file_count: int
    telemetry_source_artifact_bytes: int
    telemetry_source_artifact_digest: str


@dataclass(frozen=True, slots=True)
class FilesystemSnapshot:
    """Deterministic flattened-filesystem identity and artifact footprints."""

    application_digest: str
    application_regular_bytes: int
    filesystem_digest: str
    application_without_telemetry_source_artifacts_digest: str
    application_without_telemetry_source_artifacts_regular_bytes: int
    regular_file_count: int
    telemetry_source_artifact_bytes: int
    telemetry_source_artifact_digest: str


@dataclass(frozen=True, slots=True)
class ProvenanceIdentity:
    """Immutable identities emitted by one Buildx build."""

    config_digest: str
    manifest_digest: str


@dataclass(frozen=True, slots=True)
class TrivyEvidence:
    """Sanitized Trivy counts plus package-license resolution identities."""

    artifact_id: str | None
    counters: dict[str, int]
    licensed_packages: frozenset[str]
    report_digest: str


@dataclass(frozen=True, slots=True)
class SpdxEvidence:
    """Sanitized SPDX inventory and packages requiring cross-tool resolution."""

    license_expressions: int
    packages: int
    unresolved_license_packages: frozenset[str]


def _require_regular_file(path: Path, *, maximum_bytes: int) -> bytes:
    """Read one bounded, non-symlink regular file or fail closed."""

    try:
        metadata = path.lstat()
    except OSError as exc:
        raise VerificationError("required evidence file is unavailable") from exc
    if not stat.S_ISREG(metadata.st_mode) or metadata.st_size <= 0:
        raise VerificationError("evidence input must be a non-empty regular file")
    if metadata.st_size > maximum_bytes:
        raise VerificationError("evidence input exceeds its byte ceiling")
    try:
        return path.read_bytes()
    except OSError as exc:
        raise VerificationError("required evidence file is unreadable") from exc


def _load_json(path: Path) -> Any:
    payload = _require_regular_file(path, maximum_bytes=MAX_JSON_BYTES)

    def reject_nonfinite(_value: str) -> None:
        raise ValueError("non-finite JSON number")

    try:
        return json.loads(payload, parse_constant=reject_nonfinite)
    except (JSONDecodeError, RecursionError, UnicodeDecodeError, ValueError) as exc:
        raise VerificationError("evidence input is not bounded valid JSON") from exc


def _walk_strings(value: Any) -> tuple[str, ...]:
    """Return bounded JSON strings without recursive traversal."""

    pending = [value]
    strings: list[str] = []
    visited = 0
    while pending:
        current = pending.pop()
        visited += 1
        if visited > MAX_JSON_NODES:
            raise VerificationError("evidence JSON exceeds its structural ceiling")
        if isinstance(current, str):
            strings.append(current)
        elif isinstance(current, list):
            pending.extend(current)
        elif isinstance(current, dict):
            pending.extend(current.keys())
            pending.extend(current.values())
        elif current is not None and not isinstance(current, (bool, int, float)):
            raise VerificationError("evidence JSON contains an unsupported value")
    return tuple(strings)


def _assert_canary_absent(value: Any, canary: str) -> None:
    if any(canary in candidate for candidate in _walk_strings(value)):
        raise VerificationError("secret canary reached an evidence artifact")


def _expected_isolated_rss_workload() -> dict[str, Any]:
    """Return the exact workload shared by telemetry-disabled/enabled children."""

    return {
        "assembly": "reference_app_factory",
        "transport": "in-process HTTPX ASGI transport",
        "route_cycle": [
            "/api/v1/runs?page_size=5",
            "/api/v1/runs/r1_c2VydmljZS1iZW5jaG1hcmstcnVuLTAwMA",
            "/health/ready",
            "/health/live",
        ],
        "measured_requests": 16,
        "warmup_requests": 8,
        "admission_clock_step_seconds": 0.5,
        "synthetic_fixture_time": "2026-08-09T18:00:00Z",
        "synthetic_runs": 100,
        "synthetic_diagnostics": 256,
        "telemetry_axis": {
            "disabled": "fixed metrics; asynchronous record queues and export disabled",
            "enabled": "fixed metrics plus bounded asynchronous records and local sink",
        },
        "enabled_exporter": {
            "endpoint": "http://127.0.0.1:4318/benchmark",
            "timeout_seconds": 0.25,
            "max_retries": 0,
            "queue_capacity": 256,
            "batch_size": 32,
            "shutdown_timeout_seconds": 2.0,
            "worker_termination_timeout_seconds": 0.25,
            "sink": "network-free aggregate sink",
        },
    }


def _benchmark_integer(value: Any, *, minimum: int = 0) -> int:
    if type(value) is not int or not minimum <= value <= 2**63 - 1:
        raise VerificationError("benchmark isolated RSS integer is outside its bound")
    return value


def _benchmark_ascii(value: Any) -> str:
    if not isinstance(value, str) or not 1 <= len(value) <= 256:
        raise VerificationError("benchmark isolated RSS identity is outside its bound")
    try:
        encoded = value.encode("ascii")
    except UnicodeEncodeError as exc:
        raise VerificationError("benchmark isolated RSS identity must be ASCII") from exc
    if any(byte < 0x20 for byte in encoded):
        raise VerificationError("benchmark isolated RSS identity contains control data")
    return value


def _verify_isolated_rss_child(
    document: Any,
    *,
    expected_mode: str,
    workload_digest: str,
) -> dict[str, Any]:
    """Validate one fresh-process RSS arm against a closed evidence schema."""

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
    if not isinstance(document, dict) or set(document) != expected_fields:
        raise VerificationError("benchmark isolated RSS child schema is invalid")
    if (
        document.get("protocol") != "signalattice-service-rss-v1"
        or document.get("mode") != expected_mode
        or document.get("workload_sha256") != workload_digest
        or document.get("rss_source") != "resource.getrusage(RUSAGE_SELF).ru_maxrss"
        or document.get("rss_unit") != "bytes"
    ):
        raise VerificationError("benchmark isolated RSS child identity is incompatible")
    for field in ("runtime_platform", "python", "python_implementation"):
        _benchmark_ascii(document[field])
    before = _benchmark_integer(document["rss_before_app_bytes"], minimum=1)
    after = _benchmark_integer(document["rss_after_shutdown_peak_bytes"], minimum=1)
    growth = _benchmark_integer(document["workload_growth_bytes"])
    if after < before or growth != after - before:
        raise VerificationError("benchmark isolated RSS chronology is invalid")
    if (
        _benchmark_integer(document["warmup_requests"]) != 8
        or _benchmark_integer(document["measured_requests"], minimum=1) != 16
        or _benchmark_integer(document["successful_requests"]) != 16
        or document.get("status_counts") != {"200": 16}
        or document.get("lifespan_shutdown_completed") is not True
        or type(document.get("network_requests")) is not int
        or document.get("network_requests") != 0
        or type(document.get("processes")) is not int
        or document.get("processes") != 1
    ):
        raise VerificationError("benchmark isolated RSS workload outcome is incompatible")
    return document


def _verify_isolated_rss_evidence(document: Any) -> None:
    """Require compatible fresh processes and a bounded telemetry RSS delta."""

    expected_fields = {
        "method",
        "protocol",
        "process_order",
        "hard_timeout_seconds_per_process",
        "common_workload",
        "workload_sha256",
        "disabled",
        "enabled",
        "comparison_compatible",
        "observed_signed_delta_bytes",
        "nonnegative_delta_bytes",
        "limit_bytes",
        "within_limit",
        "negative_delta_policy",
    }
    if not isinstance(document, dict) or set(document) != expected_fields:
        raise VerificationError("benchmark isolated RSS evidence schema is invalid")
    common_workload = document.get("common_workload")
    if common_workload != _expected_isolated_rss_workload():
        raise VerificationError("benchmark isolated RSS workload has drifted")
    try:
        workload_payload = json.dumps(
            common_workload,
            ensure_ascii=True,
            allow_nan=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("ascii")
    except (TypeError, ValueError, UnicodeEncodeError) as exc:
        raise VerificationError("benchmark isolated RSS workload is not canonicalizable") from exc
    workload_digest = hashlib.sha256(workload_payload).hexdigest()
    if (
        document.get("method") != "fresh_process_peak_rss_comparison"
        or document.get("protocol") != "signalattice-service-rss-v1"
        or document.get("process_order") != ["disabled", "enabled"]
        or document.get("hard_timeout_seconds_per_process") != 20.0
        or document.get("workload_sha256") != workload_digest
        or document.get("comparison_compatible") is not True
        or document.get("limit_bytes") != MAX_TELEMETRY_RSS_DELTA_BYTES
        or document.get("within_limit") is not True
        or document.get("negative_delta_policy")
        != "retain the signed observation; clamp only the incremental-overhead guard to zero"
    ):
        raise VerificationError("benchmark isolated RSS contract is incompatible")
    disabled = _verify_isolated_rss_child(
        document.get("disabled"),
        expected_mode="disabled",
        workload_digest=workload_digest,
    )
    enabled = _verify_isolated_rss_child(
        document.get("enabled"),
        expected_mode="enabled",
        workload_digest=workload_digest,
    )
    if any(
        disabled[field] != enabled[field]
        for field in ("runtime_platform", "python", "python_implementation")
    ):
        raise VerificationError("benchmark isolated RSS processes are incompatible")
    signed_delta = (
        enabled["rss_after_shutdown_peak_bytes"] - disabled["rss_after_shutdown_peak_bytes"]
    )
    nonnegative_delta = max(0, signed_delta)
    if (
        type(document.get("observed_signed_delta_bytes")) is not int
        or document.get("observed_signed_delta_bytes") != signed_delta
        or _benchmark_integer(document.get("nonnegative_delta_bytes")) != nonnegative_delta
        or nonnegative_delta > MAX_TELEMETRY_RSS_DELTA_BYTES
    ):
        raise VerificationError("benchmark isolated telemetry RSS delta is invalid")


def verify_benchmark_canary_evidence(document: Any, *, canary: str) -> str:
    """Validate the bounded benchmark result and return its canonical digest.

    The benchmark injects the job-scoped marker into a malformed manifest.
    This independent boundary proves that neither the fault response nor any
    aggregate evidence retained it; it also prevents an empty substitute from
    satisfying the unified canary gate.
    """

    if not isinstance(document, dict):
        raise VerificationError("benchmark canary evidence has an invalid shape")
    _assert_canary_absent(document, canary)
    provenance = document.get("provenance")
    saturation = document.get("saturation")
    fault_cases = document.get("fault_cases")
    integrity = document.get("integrity")
    telemetry_ab = document.get("telemetry_ab")
    bounds_observed = document.get("bounds_observed")
    if (
        document.get("schema_version") != "1.0.0"
        or document.get("evidence_id") != "signalattice-service-operability-synthetic-local-v1"
        or document.get("evidence_class") != "measured_synthetic_local_engineering"
        or not isinstance(provenance, dict)
        or provenance.get("network_requests") != 0
        or provenance.get("credentials_used") is not False
        or provenance.get("market_or_model_data_used") is not False
        or not isinstance(provenance.get("source_tree_sha256"), str)
        or re.fullmatch(r"[0-9a-f]{64}", provenance["source_tree_sha256"]) is None
        or not isinstance(saturation, dict)
        or saturation.get("probe_status") != 200
        or type(saturation.get("rejection_count")) is not int
        or saturation["rejection_count"] < 1
        or not isinstance(fault_cases, list)
        or not isinstance(integrity, dict)
        or not isinstance(telemetry_ab, dict)
        or not isinstance(bounds_observed, list)
    ):
        raise VerificationError("benchmark canary evidence violated its closed contract")
    _verify_isolated_rss_evidence(telemetry_ab.get("isolated_process_rss"))
    rss_bounds = [
        bound
        for bound in bounds_observed
        if isinstance(bound, dict) and bound.get("name") == "telemetry_isolated_rss_delta_bytes"
    ]
    isolated_rss = telemetry_ab["isolated_process_rss"]
    if (
        len(rss_bounds) != 1
        or rss_bounds[0].get("observed") != isolated_rss["nonnegative_delta_bytes"]
        or rss_bounds[0].get("limit") != MAX_TELEMETRY_RSS_DELTA_BYTES
        or rss_bounds[0].get("unit") != "bytes"
    ):
        raise VerificationError("benchmark telemetry RSS bound is not evidence-linked")
    corrupt = [
        case
        for case in fault_cases
        if isinstance(case, dict) and case.get("case") == "corrupt_manifest"
    ]
    if (
        len(corrupt) != 1
        or corrupt[0].get("status") != 503
        or corrupt[0].get("closed_code") != "evidence_integrity_failed"
    ):
        raise VerificationError("benchmark malformed-input evidence is incomplete")
    expected_integrity = {
        "algorithm": "sha256",
        "canonicalization": "sorted compact ASCII JSON excluding the integrity member",
    }
    if any(integrity.get(key) != value for key, value in expected_integrity.items()):
        raise VerificationError("benchmark evidence integrity contract has drifted")
    without_integrity = {key: value for key, value in document.items() if key != "integrity"}
    try:
        canonical = json.dumps(
            without_integrity,
            ensure_ascii=True,
            allow_nan=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("ascii")
    except (TypeError, ValueError, UnicodeEncodeError) as exc:
        raise VerificationError("benchmark evidence is not canonicalizable") from exc
    digest = hashlib.sha256(canonical).hexdigest()
    if integrity.get("canonical_payload_sha256") != digest:
        raise VerificationError("benchmark evidence integrity verification failed")
    return digest


def _validated_canary(environment_name: str) -> str:
    if _ENV_NAME.fullmatch(environment_name) is None:
        raise VerificationError("canary environment name is invalid")
    canary = os.environ.get(environment_name)
    if canary is None or not 32 <= len(canary) <= 256 or not canary.isascii():
        raise VerificationError("secret canary is missing or outside its bounds")
    if any(character.isspace() or ord(character) < 0x21 for character in canary):
        raise VerificationError("secret canary must be bounded visible ASCII")
    return canary


def _stable_severity(value: Any) -> str:
    return value.upper() if isinstance(value, str) else "UNKNOWN"


def _is_reviewed_container_license_finding(finding: dict[str, Any]) -> bool:
    """Recognize only version- and path-bound Python license notices.

    ``typing-extensions`` 4.16.0 carries the composite Python license. Trivy's
    full-text scanner labels three component notices in that exact file as
    unknown. Any path, version, package, category, or severity drift fails
    rather than widening the reviewed application-dependency policy.
    """

    path = finding.get("FilePath")
    name = finding.get("Name")
    return (
        isinstance(path, str)
        and isinstance(name, str)
        and (path, name) in _REVIEWED_CONTAINER_LICENSE_FINDINGS
        and finding.get("PkgName") == ""
        and str(finding.get("Category", "")).upper() == "UNKNOWN"
        and _stable_severity(finding.get("Severity")) == "UNKNOWN"
    )


def _verify_trivy_metadata(
    metadata: Any,
    *,
    expected_type: str,
    expected_revision: str | None,
    expected_config_digest: str | None,
    expected_image_reference: str | None,
) -> None:
    """Bind native Trivy v0.73 metadata to the scanned source or image."""

    if not isinstance(metadata, dict):
        raise VerificationError("Trivy report metadata has an invalid shape")
    if expected_type == "repository":
        expected_fields = {
            "Author",
            "Branch",
            "Commit",
            "CommitMsg",
            "Committer",
            "RepoURL",
        }
        if (
            expected_revision is None
            or set(metadata) != expected_fields
            or metadata.get("Commit") != expected_revision
            or any(
                not isinstance(metadata.get(field), str)
                or len(metadata[field].encode("utf-8")) > 64 * 1024
                for field in expected_fields
            )
            or not metadata.get("RepoURL")
            or not metadata.get("Author")
            or not metadata.get("Committer")
        ):
            raise VerificationError("Trivy repository metadata is not source-bound")
        return

    if expected_type != "container_image":
        if expected_revision is not None and expected_revision not in _walk_strings(metadata):
            raise VerificationError("Trivy report is not bound to the requested source revision")
        if expected_config_digest is not None or expected_image_reference is not None:
            raise VerificationError("Trivy non-image report declared image identity")
        return

    expected_metadata_fields = {
        "DiffIDs",
        "ImageConfig",
        "ImageID",
        "Layers",
        "OS",
        "Reference",
        "RepoTags",
        "Size",
    }
    image_config = metadata.get("ImageConfig")
    diff_ids = metadata.get("DiffIDs")
    layers = metadata.get("Layers")
    operating_system = metadata.get("OS")
    size = metadata.get("Size")
    if (
        expected_revision is None
        or expected_config_digest is None
        or expected_image_reference is None
        or set(metadata) != expected_metadata_fields
        or metadata.get("ImageID") != expected_config_digest
        or metadata.get("Reference") != expected_image_reference
        or metadata.get("RepoTags") != [expected_image_reference]
        or type(size) is not int
        or not 1 <= size <= MAX_IMAGE_BYTES
        or not isinstance(diff_ids, list)
        or not 1 <= len(diff_ids) <= MAX_HISTORY_RECORDS
        or not all(isinstance(item, str) and _DIGEST.fullmatch(item) for item in diff_ids)
        or not isinstance(layers, list)
        or len(layers) != len(diff_ids)
        or not isinstance(operating_system, dict)
        or set(operating_system) != {"Family", "Name"}
        or not all(
            isinstance(operating_system.get(field), str)
            and 1 <= len(operating_system[field]) <= 128
            for field in ("Family", "Name")
        )
        or not isinstance(image_config, dict)
        or set(image_config)
        != {"architecture", "config", "created", "history", "os", "rootfs"}
        or image_config.get("architecture") != "amd64"
        or image_config.get("os") != "linux"
        or not isinstance(image_config.get("created"), str)
        or not 1 <= len(image_config["created"]) <= 128
        or not isinstance(image_config.get("history"), list)
        or len(image_config["history"]) > MAX_HISTORY_RECORDS
    ):
        raise VerificationError("Trivy image metadata is not artifact-bound")

    rootfs = image_config.get("rootfs")
    config = image_config.get("config")
    if (
        not isinstance(rootfs, dict)
        or rootfs != {"type": "layers", "diff_ids": diff_ids}
        or not isinstance(config, dict)
    ):
        raise VerificationError("Trivy image root filesystem metadata has drifted")
    labels = config.get("Labels")
    if (
        not isinstance(labels, dict)
        or labels.get("org.opencontainers.image.revision") != expected_revision
    ):
        raise VerificationError("Trivy image metadata is not source-bound")
    for index, layer in enumerate(layers):
        if (
            not isinstance(layer, dict)
            or set(layer) != {"DiffID", "Digest", "Size"}
            or layer.get("DiffID") != diff_ids[index]
            or not isinstance(layer.get("Digest"), str)
            or _DIGEST.fullmatch(layer["Digest"]) is None
            or type(layer.get("Size")) is not int
            or layer["Size"] < 0
        ):
            raise VerificationError("Trivy image layer metadata has drifted")


def verify_trivy_document(
    document: Any,
    *,
    canary: str,
    expected_name: str,
    expected_type: str,
    expected_revision: str | None,
    expected_config_digest: str | None = None,
    expected_image_reference: str | None = None,
    require_artifact_id: bool = True,
) -> TrivyEvidence:
    """Reject material or suppressed Trivy findings and bind the scan target."""

    if not isinstance(document, dict) or document.get("SchemaVersion") != 2:
        raise VerificationError("Trivy report has an unsupported schema")
    artifact_name = document.get("ArtifactName")
    artifact_type = document.get("ArtifactType")
    artifact_id = document.get("ArtifactID")
    report_id = document.get("ReportID")
    artifact_id_is_valid = (
        isinstance(artifact_id, str) and _ARTIFACT_ID.fullmatch(artifact_id) is not None
    )
    if (
        artifact_name != expected_name
        or artifact_type != expected_type
        or (require_artifact_id and not artifact_id_is_valid)
        or (not require_artifact_id and artifact_id is not None)
        or not isinstance(report_id, str)
        or not 1 <= len(report_id) <= 128
    ):
        raise VerificationError(
            "Trivy report target identity does not match the requested artifact"
        )
    if not isinstance(document.get("Results"), list) or not document["Results"]:
        raise VerificationError("Trivy report is missing scan results")
    _assert_canary_absent(document, canary)
    _verify_trivy_metadata(
        document.get("Metadata"),
        expected_type=expected_type,
        expected_revision=expected_revision,
        expected_config_digest=expected_config_digest,
        expected_image_reference=expected_image_reference,
    )

    counters = {
        "critical": 0,
        "high": 0,
        "fixed_vulnerabilities": 0,
        "unfixed_vulnerabilities": 0,
        "secrets": 0,
        "material_misconfigurations": 0,
        "incompatible_licenses": 0,
        "license_findings": 0,
        "resolved_package_licenses": 0,
        "reviewed_container_license_notices": 0,
        "suppressed_findings": 0,
    }
    licensed_packages: set[str] = set()
    reviewed_license_findings: set[tuple[str, str]] = set()
    for result in document["Results"]:
        if not isinstance(result, dict):
            raise VerificationError("Trivy result entry has an invalid shape")
        modified = result.get("ExperimentalModifiedFindings", [])
        if not isinstance(modified, list):
            raise VerificationError("Trivy modified-finding collection has an invalid shape")
        if modified:
            counters["suppressed_findings"] += len(modified)
        exceptions = result.get("Exceptions", [])
        if not isinstance(exceptions, list) or exceptions:
            raise VerificationError("scanner exceptions require an explicit reviewed policy")

        vulnerabilities = result.get("Vulnerabilities", [])
        secrets = result.get("Secrets", [])
        misconfigurations = result.get("Misconfigurations", [])
        licenses = result.get("Licenses", [])
        for collection in (vulnerabilities, secrets, misconfigurations, licenses):
            if not isinstance(collection, list):
                raise VerificationError("Trivy finding collection has an invalid shape")

        for finding in vulnerabilities:
            if not isinstance(finding, dict):
                raise VerificationError("vulnerability finding has an invalid shape")
            severity = _stable_severity(finding.get("Severity"))
            if finding.get("FixedVersion"):
                counters["fixed_vulnerabilities"] += 1
            else:
                counters["unfixed_vulnerabilities"] += 1
            if severity in {"HIGH", "CRITICAL"}:
                counters[severity.lower()] += 1

        if secrets:
            counters["secrets"] += len(secrets)

        for finding in misconfigurations:
            if not isinstance(finding, dict):
                raise VerificationError("misconfiguration finding has an invalid shape")
            status = str(finding.get("Status", "FAIL")).upper()
            severity = _stable_severity(finding.get("Severity"))
            if status not in {"PASS", "SKIP"} and severity in {"HIGH", "CRITICAL"}:
                counters["material_misconfigurations"] += 1

        for finding in licenses:
            if not isinstance(finding, dict):
                raise VerificationError("license finding has an invalid shape")
            counters["license_findings"] += 1
            if expected_type == "container_image" and _is_reviewed_container_license_finding(
                finding
            ):
                reviewed_identity = (finding["FilePath"], finding["Name"])
                if reviewed_identity in reviewed_license_findings:
                    raise VerificationError("reviewed container license notice was duplicated")
                reviewed_license_findings.add(reviewed_identity)
                counters["reviewed_container_license_notices"] += 1
                continue
            name = finding.get("Name")
            package = finding.get("PkgName", "")
            severity = _stable_severity(finding.get("Severity"))
            category = str(finding.get("Category", "UNKNOWN")).upper()
            if (
                not isinstance(name, str)
                or not name.strip()
                or name.strip().upper() in _UNRESOLVED_LICENSES
                or category not in _ACCEPTED_LICENSE_CATEGORIES
                or severity == "UNKNOWN"
            ):
                counters["incompatible_licenses"] += 1
                continue
            rendered = " ".join((name, category, severity))
            if (
                severity in {"HIGH", "CRITICAL"}
                or category in {"FORBIDDEN", "RESTRICTED"}
                or _FORBIDDEN_LICENSE.search(rendered) is not None
            ):
                counters["incompatible_licenses"] += 1
            elif isinstance(package, str) and 1 <= len(package) <= 256:
                licensed_packages.add(package.casefold())
                counters["resolved_package_licenses"] += 1

    if any(
        counters[key]
        for key in (
            "critical",
            "high",
            "secrets",
            "material_misconfigurations",
            "incompatible_licenses",
            "suppressed_findings",
        )
    ):
        raise VerificationError("security scan contains a release-blocking finding")
    try:
        canonical_report = json.dumps(
            document,
            ensure_ascii=True,
            allow_nan=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("ascii")
    except (TypeError, ValueError, UnicodeEncodeError) as exc:
        raise VerificationError("Trivy report is not canonicalizable") from exc
    return TrivyEvidence(
        artifact_id=artifact_id,
        counters=counters,
        licensed_packages=frozenset(licensed_packages),
        report_digest=hashlib.sha256(canonical_report).hexdigest(),
    )


def verify_spdx_document(
    document: Any,
    *,
    canary: str,
    expected_name: str,
    expected_revision: str,
) -> SpdxEvidence:
    """Validate a source- and image-bound SPDX 2.3 package inventory."""

    if not isinstance(document, dict) or document.get("spdxVersion") != "SPDX-2.3":
        raise VerificationError("SBOM must use SPDX 2.3")
    if document.get("SPDXID") != "SPDXRef-DOCUMENT":
        raise VerificationError("SBOM document identity is invalid")
    if document.get("dataLicense") != "CC0-1.0":
        raise VerificationError("SBOM data license is invalid")
    creation = document.get("creationInfo")
    packages = document.get("packages")
    creators = creation.get("creators") if isinstance(creation, dict) else None
    namespace = document.get("documentNamespace")
    if (
        document.get("name") != expected_name
        or not isinstance(creators, list)
        or not creators
        or not all(isinstance(creator, str) and creator for creator in creators)
    ):
        raise VerificationError("SBOM creation provenance is missing")
    if not isinstance(namespace, str) or not namespace.startswith("https://"):
        raise VerificationError("SBOM document namespace is invalid")
    if not isinstance(packages, list) or not packages:
        raise VerificationError("SBOM contains no packages")
    _assert_canary_absent(document, canary)

    license_expressions = 0
    unresolved: set[str] = set()
    root_found = False
    for package in packages:
        if not isinstance(package, dict):
            raise VerificationError("SBOM package entry has an invalid shape")
        package_name = package.get("name")
        if not isinstance(package_name, str) or not 1 <= len(package_name) <= 256:
            raise VerificationError("SBOM package identity is invalid")
        if package_name == expected_name:
            if package.get("versionInfo") != expected_revision:
                raise VerificationError("SBOM root package is not bound to the source revision")
            root_found = True
            continue
        resolved = False
        for field in ("licenseConcluded", "licenseDeclared"):
            value = package.get(field)
            if not isinstance(value, str):
                raise VerificationError("SBOM package lacks a license decision")
            normalized = value.strip().upper()
            if normalized not in _UNRESOLVED_LICENSES and "LICENSEREF-" not in normalized:
                license_expressions += 1
                resolved = True
                if _FORBIDDEN_LICENSE.search(value) is not None:
                    raise VerificationError("SBOM contains an incompatible license")
        if not resolved:
            unresolved.add(package_name.casefold())
    if not root_found:
        raise VerificationError("SBOM lacks the exact image identity root package")
    return SpdxEvidence(
        license_expressions=license_expressions,
        packages=len(packages),
        unresolved_license_packages=frozenset(unresolved),
    )


def verify_provenance_document(
    document: Any, *, canary: str, expected_revision: str
) -> ProvenanceIdentity:
    """Validate Buildx maximum provenance and return exact image identities."""

    if not isinstance(document, dict):
        raise VerificationError("build metadata has an invalid shape")
    _assert_canary_absent(document, canary)
    config_digest = document.get("containerimage.config.digest")
    manifest_digest = document.get("containerimage.digest")
    provenance = document.get("buildx.build.provenance")
    if (
        not isinstance(config_digest, str)
        or _DIGEST.fullmatch(config_digest) is None
        or not isinstance(manifest_digest, str)
        or _DIGEST.fullmatch(manifest_digest) is None
    ):
        raise VerificationError("build metadata lacks immutable config and manifest digests")
    if not isinstance(provenance, dict):
        raise VerificationError("Buildx provenance metadata is missing")
    if expected_revision not in _walk_strings(provenance):
        raise VerificationError("Buildx provenance is not bound to the source revision")
    build_type = provenance.get("buildType")
    materials = provenance.get("materials")
    if not isinstance(build_type, str) or not build_type.startswith("https://mobyproject.org/"):
        raise VerificationError("Buildx provenance type is invalid")
    if not isinstance(materials, list) or not materials:
        raise VerificationError("Buildx provenance materials are missing")
    if not any(
        isinstance(material, dict)
        and isinstance(material.get("uri"), str)
        and isinstance(material.get("digest"), dict)
        and any(
            isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value)
            for value in material["digest"].values()
        )
        for material in materials
    ):
        raise VerificationError("Buildx provenance materials lack an immutable digest")
    return ProvenanceIdentity(
        config_digest=config_digest,
        manifest_digest=manifest_digest,
    )


def _run(command: list[str], *, timeout_seconds: int = 60) -> bytes:
    """Run one fixed local command without reflecting command or process output."""

    try:
        result = subprocess.run(
            command,
            check=False,
            capture_output=True,
            timeout=timeout_seconds,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise VerificationError("required local container operation failed") from exc
    if result.returncode != 0:
        raise VerificationError("required local container operation failed")
    return result.stdout


def _validated_image_name(value: str) -> str:
    if _IMAGE_NAME.fullmatch(value) is None:
        raise VerificationError("container image reference is invalid")
    return value


def _validated_revision(value: str) -> str:
    if _SOURCE_REVISION.fullmatch(value) is None:
        raise VerificationError("source revision must be one full lowercase Git SHA")
    return value


def _verify_repository_source(root: Path, *, expected_revision: str) -> str:
    """Bind a filesystem scan to one canonical, clean Git checkout.

    Trivy recognizes an actions/checkout tree as a repository and emits its
    own immutable identity. The workflow still mounts this exact checkout at
    ``/workspace`` and independently proves it is the requested clean source,
    so neither scanner metadata nor the host checkout is a single trust root.
    """

    try:
        metadata = root.lstat()
        canonical = root.resolve(strict=True)
    except OSError as exc:
        raise VerificationError("repository scan root is unavailable") from exc
    if not stat.S_ISDIR(metadata.st_mode) or root.is_symlink() or canonical != root.absolute():
        raise VerificationError("repository scan root must be a canonical directory")
    head = _run(
        ["git", "-C", str(canonical), "rev-parse", "--verify", "HEAD"],
        timeout_seconds=30,
    )
    top_level = _run(
        ["git", "-C", str(canonical), "rev-parse", "--show-toplevel"],
        timeout_seconds=30,
    )
    status = _run(
        [
            "git",
            "-C",
            str(canonical),
            "status",
            "--porcelain=v1",
            "--untracked-files=all",
        ],
        timeout_seconds=30,
    )
    if len(head) > 128 or len(top_level) > 4_096 or len(status) > 1024 * 1024:
        raise VerificationError("repository identity output exceeded its byte ceiling")
    try:
        observed_head = head.decode("ascii").strip()
        observed_root = Path(top_level.decode("utf-8").strip()).resolve(strict=True)
    except (OSError, UnicodeDecodeError) as exc:
        raise VerificationError("repository identity output is invalid") from exc
    if observed_head != expected_revision or observed_root != canonical:
        raise VerificationError("repository scan root is not the requested source revision")
    if status:
        raise VerificationError("repository scan root is not a clean checkout")
    return hashlib.sha256((observed_head + "\n").encode("ascii")).hexdigest()


def _docker_inspect(image: str) -> dict[str, Any]:
    payload = _run(["docker", "image", "inspect", _validated_image_name(image)])
    if len(payload) > 4 * 1024 * 1024:
        raise VerificationError("Docker inspection output exceeds its byte ceiling")
    try:
        parsed = json.loads(payload)
    except (JSONDecodeError, UnicodeDecodeError, RecursionError) as exc:
        raise VerificationError("Docker inspection output is invalid") from exc
    if not isinstance(parsed, list) or len(parsed) != 1 or not isinstance(parsed[0], dict):
        raise VerificationError("Docker inspection returned an unexpected inventory")
    return parsed[0]


def _verify_healthcheck_config(healthcheck: Any) -> None:
    if not isinstance(healthcheck, dict) or healthcheck.get("Test") != EXPECTED_HEALTHCHECK_TEST:
        raise VerificationError("service image lacks the exact socket-aware health check")
    if (
        healthcheck.get("Interval") != 10_000_000_000
        or healthcheck.get("Timeout") != 2_000_000_000
        or healthcheck.get("StartPeriod") != 5_000_000_000
        or healthcheck.get("Retries") != 3
    ):
        raise VerificationError("service image health check bounds have drifted")


def _normalized_config(
    inspect: dict[str, Any], *, expected_revision: str
) -> tuple[dict[str, Any], int, str]:
    config = inspect.get("Config")
    rootfs = inspect.get("RootFS")
    size = inspect.get("Size")
    image_config_digest = inspect.get("Id")
    if not isinstance(config, dict) or not isinstance(rootfs, dict):
        raise VerificationError("image inspection lacks runtime configuration")
    if type(size) is not int or size <= 0:
        raise VerificationError("image inspection lacks a valid uncompressed size")
    if not isinstance(image_config_digest, str) or _DIGEST.fullmatch(image_config_digest) is None:
        raise VerificationError("image inspection lacks an immutable config digest")
    layers = rootfs.get("Layers")
    if (
        not isinstance(layers, list)
        or not layers
        or not all(isinstance(layer, str) for layer in layers)
    ):
        raise VerificationError("image inspection lacks an immutable layer inventory")
    if inspect.get("Os") != "linux" or inspect.get("Architecture") != "amd64":
        raise VerificationError("service image does not match the declared platform")
    if config.get("User") != "10001:10001":
        raise VerificationError("service image runtime identity is not UID/GID 10001")
    if (
        config.get("Entrypoint") != EXPECTED_SERVICE_ENTRYPOINT
        or config.get("Cmd") != EXPECTED_SERVICE_COMMAND
    ):
        raise VerificationError("service image command contract has drifted")
    if config.get("WorkingDir") != "/app":
        raise VerificationError("service image working directory has drifted")
    if config.get("ExposedPorts") not in (None, {}):
        raise VerificationError("service image must expose no TCP or UDP port")
    if config.get("Volumes") not in (None, {}):
        raise VerificationError("service image must not declare writable volumes")

    environment = config.get("Env")
    if not isinstance(environment, list) or not all(isinstance(item, str) for item in environment):
        raise VerificationError("service image environment has an invalid shape")
    for item in environment:
        name, _, value = item.partition("=")
        if re.search(r"(?:PASSWORD|PASSPHRASE|TOKEN|API_KEY|PRIVATE_KEY|CREDENTIAL)", name):
            raise VerificationError("service image embeds a credential-like environment entry")
        if name == "HOME" and value != "/nonexistent":
            raise VerificationError("service runtime home contract has drifted")
    if "PYTHONPATH=/app/src" not in environment:
        raise VerificationError("service source import boundary has drifted")

    healthcheck = config.get("Healthcheck")
    _verify_healthcheck_config(healthcheck)

    labels = config.get("Labels")
    if (
        not isinstance(labels, dict)
        or labels.get("org.opencontainers.image.revision") != expected_revision
        or labels.get("org.opencontainers.image.source")
        != "https://github.com/srgangaram-swe/Signalattice"
        or labels.get("org.opencontainers.image.licenses") != "MIT"
    ):
        raise VerificationError("service image source-identity labels have drifted")

    normalized = {
        "cmd": config.get("Cmd"),
        "entrypoint": config.get("Entrypoint"),
        "environment": sorted(environment),
        "healthcheck": healthcheck,
        "labels": labels,
        "user": config.get("User"),
        "working_directory": config.get("WorkingDir"),
    }
    return normalized, len(layers), image_config_digest


def _verify_history(image: str, *, canary: bytes) -> int:
    payload = _run(
        ["docker", "history", "--no-trunc", "--format", "{{json .}}", image],
        timeout_seconds=30,
    )
    if canary in payload or _FORBIDDEN_HISTORY.search(payload) is not None:
        raise VerificationError("image history contains credential material")
    records = payload.splitlines()
    if not records or len(records) > MAX_HISTORY_RECORDS:
        raise VerificationError("image history exceeds its record ceiling")
    if any(len(record) > MAX_HISTORY_RECORD_BYTES for record in records):
        raise VerificationError("image history record exceeds its byte ceiling")
    for record in records:
        try:
            parsed = json.loads(record)
        except (JSONDecodeError, UnicodeDecodeError, RecursionError) as exc:
            raise VerificationError("image history record is invalid") from exc
        if not isinstance(parsed, dict):
            raise VerificationError("image history record has an invalid shape")
    return len(records)


def _canonical_tar_path(raw: str) -> str:
    if (
        not raw
        or "\x00" in raw
        or "\\" in raw
        or raw.startswith("/")
        or any(ord(character) < 0x20 or ord(character) == 0x7F for character in raw)
    ):
        raise VerificationError("container filesystem contains an invalid path")
    path = PurePosixPath(raw)
    canonical = path.as_posix()
    if raw != canonical or any(part in {"", ".", ".."} for part in path.parts):
        raise VerificationError("container filesystem path is not canonical")
    return canonical


def _resolved_link_target(member_name: str, target: str, *, hard_link: bool) -> str:
    """Resolve one tar link lexically and reject escape or ambiguous spelling."""

    if (
        not target
        or "\x00" in target
        or "\\" in target
        or any(ord(character) < 0x20 or ord(character) == 0x7F for character in target)
    ):
        raise VerificationError("container filesystem link target is invalid")
    target_path = PurePosixPath(target)
    if target != target_path.as_posix():
        raise VerificationError("container filesystem link target is not canonical")
    if hard_link and target_path.is_absolute():
        raise VerificationError("container filesystem hard link target is invalid")

    parts = (
        []
        if target_path.is_absolute() or hard_link
        else list(PurePosixPath(member_name).parent.parts)
    )
    for part in target_path.parts:
        if part in {"", ".", "/"}:
            continue
        if part == "..":
            if not parts:
                raise VerificationError("container filesystem link escapes its root")
            parts.pop()
        else:
            parts.append(part)
    if not parts:
        raise VerificationError("container filesystem link target resolves to the root")
    return PurePosixPath(*parts).as_posix()


def _member_bytes(
    archive: tarfile.TarFile, member: tarfile.TarInfo, *, canary: bytes
) -> tuple[str, bytes]:
    source = archive.extractfile(member)
    if source is None:
        raise VerificationError("container filesystem member is unreadable")
    digest = hashlib.sha256()
    captured = bytearray()
    previous = b""
    remaining = member.size
    while remaining:
        chunk = source.read(min(1024 * 1024, remaining))
        if not chunk:
            raise VerificationError("container filesystem member is truncated")
        remaining -= len(chunk)
        digest.update(chunk)
        window = previous + chunk
        if canary in window or _WORKSTATION_PATH.search(window) is not None:
            raise VerificationError("container filesystem contains private material")
        if len(captured) < 64 * 1024:
            captured.extend(chunk[: 64 * 1024 - len(captured)])
        previous = window[-max(len(canary), 128) :]
    if source.read(1):
        raise VerificationError("container filesystem member exceeded its declared size")
    return digest.hexdigest(), bytes(captured)


def _scan_exported_filesystem(image: str, *, canary: bytes) -> FilesystemSnapshot:
    """Flatten, inventory, and hash one image without executing it."""

    with tempfile.TemporaryDirectory(prefix="signalattice-container-verify-") as directory:
        archive_path = Path(directory) / "rootfs.tar"
        container_id = (
            _run(["docker", "create", image], timeout_seconds=30)
            .decode("ascii", errors="strict")
            .strip()
        )
        if re.fullmatch(r"[0-9a-f]{64}", container_id) is None:
            raise VerificationError("Docker returned an invalid temporary container identity")
        try:
            _run(
                ["docker", "export", "--output", str(archive_path), container_id],
                timeout_seconds=180,
            )
        finally:
            _run(["docker", "rm", "--force", container_id], timeout_seconds=30)

        try:
            # Keep construction outside the iteration boundary so malformed-open and
            # malformed-member failures retain distinct sanitized causes; the archive
            # is closed unconditionally below.
            archive = tarfile.open(archive_path, mode="r:")  # noqa: SIM115
        except (OSError, tarfile.TarError) as exc:
            raise VerificationError("container filesystem export is invalid") from exc

        application_entries: list[str] = []
        application_without_telemetry_source_entries: list[str] = []
        telemetry_source_artifact_entries: list[str] = []
        filesystem_entries: list[str] = []
        seen_names: set[str] = set()
        hard_link_targets: list[str] = []
        regular_files = 0
        application_regular_bytes = 0
        telemetry_source_artifact_bytes = 0
        telemetry_source_artifact_paths: dict[str, str] = {}
        observed_service_distributions: set[str] = set()
        observed_service_source_files: set[str] = set()
        logical_bytes = 0
        passwd = b""
        group = b""
        required = {"opt/venv/bin/python": False, "quant_platform": False}
        runtime_directories = {
            "run/signalattice": False,
            "var/lib/signalattice/registry": False,
            "var/lib/signalattice/cas": False,
        }
        forbidden_prefixes = (
            ".git",
            "app/configs",
            "app/data",
            "app/docs",
            "app/models",
            "app/reports",
            "app/scripts",
            "app/tests",
            "build",
            "workspace",
        )
        forbidden_suffixes = (
            ".sqlite",
            ".sqlite3",
            ".db",
            ".duckdb",
            ".parquet",
            ".feather",
            ".arrow",
            ".avro",
            ".orc",
            ".csv",
            ".tsv",
            ".jsonl",
            ".ndjson",
            ".pkl",
            ".pickle",
            ".joblib",
            ".h5",
            ".hdf5",
            ".keras",
            ".npy",
            ".npz",
            ".onnx",
            ".pt",
            ".pth",
            ".safetensors",
            ".pyc",
        )
        try:
            for index, member in enumerate(archive, start=1):
                if index > MAX_TAR_MEMBERS:
                    raise VerificationError("container filesystem exceeds its member ceiling")
                name = _canonical_tar_path(member.name)
                if name in seen_names:
                    raise VerificationError("container filesystem contains a duplicate path")
                seen_names.add(name)
                if member.size < 0 or member.size > MAX_TAR_MEMBER_BYTES:
                    raise VerificationError("container filesystem member exceeds its byte ceiling")
                logical_bytes += member.size
                if logical_bytes > MAX_TAR_LOGICAL_BYTES:
                    raise VerificationError("container filesystem exceeds its logical byte ceiling")
                if member.isdev() or member.isfifo():
                    raise VerificationError("container filesystem contains a special file")
                if member.mode & 0o6000:
                    raise VerificationError("container filesystem contains a privileged mode bit")
                if any(
                    "security.capability" in f"{key}={value}".lower()
                    for key, value in member.pax_headers.items()
                ):
                    raise VerificationError("container filesystem contains a file capability")
                encoded_name = name.encode("utf-8", errors="strict")
                if canary in encoded_name or _WORKSTATION_PATH.search(encoded_name) is not None:
                    raise VerificationError("container filesystem contains private material")

                # Scan regular bytes and link targets before applying artifact-type
                # policy so a credential canary is never hidden behind a more
                # general extension or package-layout rejection.
                content_digest = "-"
                captured = b""
                if member.isfile():
                    regular_files += 1
                    content_digest, captured = _member_bytes(archive, member, canary=canary)
                    if name == "etc/passwd":
                        passwd = captured
                    elif name == "etc/group":
                        group = captured
                elif member.issym() or member.islnk():
                    target = member.linkname.encode("utf-8", errors="strict")
                    if canary in target or _WORKSTATION_PATH.search(target) is not None:
                        raise VerificationError(
                            "container filesystem link contains private material"
                        )
                    resolved_target = _resolved_link_target(
                        name,
                        member.linkname,
                        hard_link=member.islnk(),
                    )
                    if member.islnk():
                        hard_link_targets.append(resolved_target)
                    content_digest = hashlib.sha256(target).hexdigest()
                if any(part in {".git", ".hg", ".svn"} for part in PurePosixPath(name).parts):
                    raise VerificationError("container filesystem contains VCS metadata")
                if any(
                    name == prefix or name.startswith(prefix + "/") for prefix in forbidden_prefixes
                ):
                    raise VerificationError("service image contains an excluded project artifact")
                if name.startswith("app/") and not (
                    name == "app/src" or name.startswith("app/src/")
                ):
                    raise VerificationError("service image contains an unexpected application file")
                if (
                    (name.startswith("opt/venv/") or name.startswith("app/src/"))
                    and (member.isfile() or member.isdir())
                    and member.mode & 0o022
                ):
                    raise VerificationError(
                        "service application contains a writable group boundary"
                    )
                is_application_path = (
                    name == "opt/venv"
                    or name.startswith("opt/venv/")
                    or name == "app/src"
                    or name.startswith("app/src/")
                )
                if (
                    is_application_path
                    and name != "opt/venv/lib/python3.13/site-packages/_virtualenv.pth"
                    and name.lower().endswith(forbidden_suffixes)
                ):
                    raise VerificationError(
                        "service image contains an excluded data or model artifact"
                    )
                if is_application_path and any(
                    part.casefold() in _FORBIDDEN_APPLICATION_DIRECTORY_NAMES
                    for part in PurePosixPath(name).parts
                ):
                    raise VerificationError(
                        "service application contains a test, fixture, sample, or dataset directory"
                    )
                site_packages_prefix = "opt/venv/lib/python3.13/site-packages/"
                if name.startswith(site_packages_prefix):
                    site_relative = name[len(site_packages_prefix) :]
                    site_root = site_relative.split("/", 1)[0]
                    if site_root not in _EXPECTED_SITE_PACKAGE_ROOTS:
                        raise VerificationError(
                            "service dependency inventory contains an undeclared package root"
                        )
                    if site_root.endswith(".dist-info"):
                        observed_service_distributions.add(site_root)
                elif "site-packages/" in name:
                    raise VerificationError(
                        "service dependency inventory escaped its isolated environment"
                    )
                if is_application_path and any(
                    part.endswith((".dist-info", ".egg-info")) for part in PurePosixPath(name).parts
                ):
                    distribution_roots = {
                        part for part in PurePosixPath(name).parts if part.endswith(".dist-info")
                    }
                    if not distribution_roots <= _EXPECTED_SERVICE_DISTRIBUTIONS:
                        raise VerificationError(
                            "service dependency metadata is outside the locked allowlist"
                        )
                    if any(part.endswith(".egg-info") for part in PurePosixPath(name).parts):
                        raise VerificationError(
                            "service runtime contains editable or legacy package metadata"
                        )
                if name.startswith("app/src/"):
                    if member.isfile():
                        if (
                            name not in _EXPECTED_SERVICE_SOURCE_FILES
                            or member.uid != 0
                            or member.gid != 0
                            or member.mode != 0o444
                        ):
                            raise VerificationError(
                                "service source inventory violates its exact read-only allowlist"
                            )
                        observed_service_source_files.add(name)
                    elif member.isdir():
                        if not any(
                            source.startswith(name + "/")
                            for source in _EXPECTED_SERVICE_SOURCE_FILES
                        ):
                            raise VerificationError(
                                "service source inventory contains an unexpected directory"
                            )
                    else:
                        raise VerificationError(
                            "service source inventory contains a non-regular module"
                        )
                if name in {
                    "opt/venv/bin/pip",
                    "opt/venv/bin/pip3",
                    "opt/venv/bin/uv",
                    "usr/local/bin/pip",
                    "usr/local/bin/pip3",
                    "usr/local/bin/pip3.13",
                    "usr/local/lib/python3.13/ensurepip",
                } or name.startswith(
                    (
                        "usr/local/lib/python3.13/ensurepip/",
                        "usr/local/lib/python3.13/site-packages/pip",
                    )
                ):
                    raise VerificationError("service runtime contains a package installer")
                if name == "opt/venv/bin/python":
                    required[name] = True
                if name == "app/src/quant_platform/__init__.py":
                    required["quant_platform"] = True
                if name == "run/signalattice":
                    if (
                        not member.isdir()
                        or member.uid != 10001
                        or member.gid != 10001
                        or member.mode != 0o700
                    ):
                        raise VerificationError("socket parent permissions have drifted")
                    runtime_directories[name] = True
                elif name in {
                    "var/lib/signalattice/registry",
                    "var/lib/signalattice/cas",
                }:
                    if (
                        not member.isdir()
                        or member.uid != 0
                        or member.gid != 10001
                        or member.mode != 0o550
                    ):
                        raise VerificationError("read-only mountpoint permissions have drifted")
                    runtime_directories[name] = True

                record = f"{name}\0{member.type!r}\0{member.mode:o}\0{member.uid}\0{member.gid}\0{member.size}\0{content_digest}"
                if name in _DOCKER_RUNTIME_INJECTED_PATHS:
                    if not member.isfile():
                        raise VerificationError(
                            "Docker runtime-injected filesystem path is not a regular file"
                        )
                    # Docker replaces these files at container-create time with daemon
                    # state (container identity and DNS). Their bytes are still scanned
                    # above, while reproducibility is proved by the exact OCI config and
                    # immutable layer identities rather than random runtime injection.
                    record = f"{name}\0docker-runtime-injected-regular-file"
                filesystem_entries.append(record)
                if (
                    name == "opt/venv"
                    or name.startswith("opt/venv/")
                    or name == "app/src"
                    or name.startswith("app/src/")
                ):
                    application_entries.append(record)
                    telemetry_suffix = next(
                        (
                            suffix
                            for suffix in _TELEMETRY_SOURCE_ARTIFACT_SUFFIXES
                            if name.endswith(suffix)
                        ),
                        None,
                    )
                    if telemetry_suffix is None:
                        application_without_telemetry_source_entries.append(record)
                    else:
                        if (
                            not member.isfile()
                            or telemetry_suffix in telemetry_source_artifact_paths
                        ):
                            raise VerificationError(
                                "telemetry artifact inventory is not an exact regular-file set"
                            )
                        telemetry_source_artifact_paths[telemetry_suffix] = name
                        telemetry_source_artifact_entries.append(record)
                        telemetry_source_artifact_bytes += member.size
                    if member.isfile():
                        application_regular_bytes += member.size

                if (
                    member.uid == 10001
                    and member.mode & 0o200
                    and name != "run/signalattice"
                    and not name.startswith("run/signalattice/")
                ):
                    raise VerificationError(
                        "service filesystem grants unexpected owner write access"
                    )
        except (OSError, tarfile.TarError, UnicodeError) as exc:
            raise VerificationError("container filesystem could not be verified") from exc
        finally:
            archive.close()

        if any(target not in seen_names for target in hard_link_targets):
            raise VerificationError("container filesystem hard link target is missing")

    if (
        not all(required.values())
        or not all(runtime_directories.values())
        or not application_entries
        or observed_service_distributions != _EXPECTED_SERVICE_DISTRIBUTIONS
        or observed_service_source_files != _EXPECTED_SERVICE_SOURCE_FILES
        or set(telemetry_source_artifact_paths) != _TELEMETRY_SOURCE_ARTIFACT_SUFFIXES
        or not application_without_telemetry_source_entries
    ):
        raise VerificationError("service application inventory is incomplete")
    passwd_text = passwd.decode("utf-8", errors="strict").splitlines()
    group_text = group.decode("utf-8", errors="strict").splitlines()
    expected_passwd = "signalattice:x:10001:10001::/nonexistent:/usr/sbin/nologin"
    if passwd_text.count(expected_passwd) != 1:
        raise VerificationError("service runtime account contract has drifted")
    if sum(line.startswith("signalattice:x:10001:") for line in group_text) != 1:
        raise VerificationError("service runtime group contract has drifted")

    application_digest = hashlib.sha256("\n".join(sorted(application_entries)).encode()).hexdigest()
    application_without_telemetry_source_artifacts_digest = hashlib.sha256(
        "\n".join(sorted(application_without_telemetry_source_entries)).encode()
    ).hexdigest()
    telemetry_source_artifact_digest = hashlib.sha256(
        "\n".join(sorted(telemetry_source_artifact_entries)).encode()
    ).hexdigest()
    filesystem_digest = hashlib.sha256("\n".join(sorted(filesystem_entries)).encode()).hexdigest()
    application_without_telemetry_source_artifacts_regular_bytes = (
        application_regular_bytes - telemetry_source_artifact_bytes
    )
    _verify_telemetry_source_artifact_bytes(telemetry_source_artifact_bytes)
    if application_without_telemetry_source_artifacts_regular_bytes < 1:
        raise VerificationError("service artifact source-footprint baseline is invalid")
    return FilesystemSnapshot(
        application_digest=application_digest,
        application_regular_bytes=application_regular_bytes,
        filesystem_digest=filesystem_digest,
        application_without_telemetry_source_artifacts_digest=(
            application_without_telemetry_source_artifacts_digest
        ),
        application_without_telemetry_source_artifacts_regular_bytes=(
            application_without_telemetry_source_artifacts_regular_bytes
        ),
        regular_file_count=regular_files,
        telemetry_source_artifact_bytes=telemetry_source_artifact_bytes,
        telemetry_source_artifact_digest=telemetry_source_artifact_digest,
    )


def _verify_telemetry_source_artifact_bytes(value: int) -> None:
    """Reject absent, negative, or over-budget telemetry-only source bytes."""

    if type(value) is not int or not 1 <= value <= MAX_TELEMETRY_SOURCE_ARTIFACT_BYTES:
        raise VerificationError("telemetry source artifacts are outside the 100 MiB budget")


def inspect_service_image(image: str, *, canary: str, expected_revision: str) -> ImageSnapshot:
    """Return one sanitized service-image snapshot after all local checks pass."""

    image = _validated_image_name(image)
    inspect = _docker_inspect(image)
    normalized, layer_count, image_config_digest = _normalized_config(
        inspect,
        expected_revision=expected_revision,
    )
    size = inspect["Size"]
    if size > MAX_IMAGE_BYTES:
        raise VerificationError("service image exceeds the 1.25 GiB size budget")
    _assert_canary_absent(inspect, canary)
    history_record_count = _verify_history(image, canary=canary.encode("ascii"))
    filesystem = _scan_exported_filesystem(image, canary=canary.encode("ascii"))
    config_digest = hashlib.sha256(
        json.dumps(normalized, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    return ImageSnapshot(
        application_digest=filesystem.application_digest,
        application_regular_bytes=filesystem.application_regular_bytes,
        config_digest=config_digest,
        filesystem_digest=filesystem.filesystem_digest,
        image_size_bytes=size,
        history_record_count=history_record_count,
        image_config_digest=image_config_digest,
        layer_count=layer_count,
        application_without_telemetry_source_artifacts_digest=(
            filesystem.application_without_telemetry_source_artifacts_digest
        ),
        application_without_telemetry_source_artifacts_regular_bytes=(
            filesystem.application_without_telemetry_source_artifacts_regular_bytes
        ),
        regular_file_count=filesystem.regular_file_count,
        telemetry_source_artifact_bytes=filesystem.telemetry_source_artifact_bytes,
        telemetry_source_artifact_digest=filesystem.telemetry_source_artifact_digest,
    )


def _inspect_cli_size(image: str, *, canary: str) -> int:
    inspect = _docker_inspect(_validated_image_name(image))
    _assert_canary_absent(inspect, canary)
    size = inspect.get("Size")
    if type(size) is not int or size <= 0 or size > MAX_IMAGE_BYTES:
        raise VerificationError("CLI image size is outside the declared budget")
    return size


def _parse_tmpfs_size(value: str) -> int:
    match = re.fullmatch(r"([0-9]+)([kKmMgG]?)", value)
    if match is None:
        raise VerificationError("Compose tmpfs size is invalid")
    multiplier = {"": 1, "k": 1024, "m": 1024**2, "g": 1024**3}[match.group(2).lower()]
    return int(match.group(1)) * multiplier


def _verify_tmpfs_options(raw: Any, *, size: int, mode: int) -> None:
    if not isinstance(raw, str) or not 1 <= len(raw) <= 512:
        raise VerificationError("Compose tmpfs options are invalid")
    flags: set[str] = set()
    values: dict[str, str] = {}
    for token in raw.split(","):
        if not token or token in flags:
            raise VerificationError("Compose tmpfs options are ambiguous")
        if "=" in token:
            key, value = token.split("=", 1)
            if key in values or key not in {"size", "mode", "uid", "gid"} or not value:
                raise VerificationError("Compose tmpfs options are ambiguous")
            values[key] = value
        else:
            flags.add(token)
    if flags != {"rw", "noexec", "nosuid", "nodev"}:
        raise VerificationError("Compose tmpfs hardening options have drifted")
    try:
        parsed_mode = int(values.get("mode", ""), 8)
    except ValueError as exc:
        raise VerificationError("Compose tmpfs mode is invalid") from exc
    if (
        _parse_tmpfs_size(values.get("size", "")) != size
        or parsed_mode != mode
        or values.get("uid") != "10001"
        or values.get("gid") != "10001"
    ):
        raise VerificationError("Compose tmpfs ownership or resource bounds have drifted")


def verify_runtime_probe_document(document: Any, *, canary: str) -> None:
    """Validate the fixed in-container UDS and secret probe result."""

    _assert_canary_absent(document, canary)
    expected = {
        "result": "passed",
        "schema_version": 1,
        "secret_environment_present": False,
        "secret_file": {
            "gid": 10001,
            "matches_source": True,
            "mode": 0o400,
            "nlink": 1,
            "regular": True,
            "uid": 10001,
        },
        "socket": {
            "gid": 10001,
            "mode": 0o600,
            "socket": True,
            "uid": 10001,
        },
        "socket_parent": {
            "directory": True,
            "gid": 10001,
            "mode": 0o700,
            "uid": 10001,
        },
    }
    if document != expected:
        raise VerificationError("in-container Compose runtime probe violated its exact contract")


def verify_runtime_saturation_document(document: Any, *, canary: str) -> dict[str, Any]:
    """Validate actual-UDS overload, reserved-probe, PID, and recovery evidence."""

    _assert_canary_absent(document, canary)
    expected_keys = {
        "data_problem_code_counts",
        "data_status_counts",
        "docker_healthcheck_recovered",
        "liveness_latency_ms",
        "liveness_status",
        "pids_headroom",
        "pids_limit",
        "pids_peak",
        "readiness_code",
        "readiness_latency_ms",
        "readiness_retry_after_seconds",
        "readiness_retryable",
        "readiness_status",
        "result",
        "schema_version",
        "submitted_requests",
    }
    if not isinstance(document, dict) or set(document) != expected_keys:
        raise VerificationError("runtime saturation evidence has an invalid shape")
    counts = document.get("data_status_counts")
    problem_codes = document.get("data_problem_code_counts")
    if (
        document.get("schema_version") != 1
        or document.get("result") != "passed"
        or document.get("submitted_requests") != 48
        or document.get("liveness_status") != 200
        or document.get("readiness_status") != 503
        or document.get("readiness_code") != "registry_busy"
        or document.get("readiness_retryable") is not True
        or document.get("readiness_retry_after_seconds") != 1
        or document.get("pids_limit") != EXPECTED_SERVICE_PIDS
        or document.get("docker_healthcheck_recovered") is not True
        or not isinstance(counts, dict)
        or not counts
        or set(counts) != {"429", "503"}
        or any(type(value) is not int or value < 1 for value in counts.values())
        or sum(counts.values()) != 48
        or counts.get("429", 0) < 1
        or counts.get("503", 0) < 1
        or not isinstance(problem_codes, dict)
        or set(problem_codes) != {"evidence_unavailable", "service_saturated"}
        or any(type(value) is not int or value < 1 for value in problem_codes.values())
        or problem_codes.get("evidence_unavailable") != counts.get("503")
        or problem_codes.get("service_saturated") != counts.get("429")
    ):
        raise VerificationError("runtime saturation behavior violated its closed contract")
    liveness_latency = document.get("liveness_latency_ms")
    readiness_latency = document.get("readiness_latency_ms")
    if (
        isinstance(liveness_latency, bool)
        or not isinstance(liveness_latency, (int, float))
        or not math.isfinite(float(liveness_latency))
        or not 0.0 <= float(liveness_latency) <= 250.0
        or isinstance(readiness_latency, bool)
        or not isinstance(readiness_latency, (int, float))
        or not math.isfinite(float(readiness_latency))
        or not 0.0 <= float(readiness_latency) <= 1_000.0
    ):
        raise VerificationError("runtime reserved-probe latency exceeded its bound")
    pids_peak = document.get("pids_peak")
    pids_headroom = document.get("pids_headroom")
    if (
        type(pids_peak) is not int
        or not MIN_SATURATED_SERVICE_PIDS <= pids_peak <= MAX_SATURATED_SERVICE_PIDS
        or type(pids_headroom) is not int
        or pids_headroom != EXPECTED_SERVICE_PIDS - pids_peak
        or pids_headroom < EXPECTED_SERVICE_PIDS - MAX_SATURATED_SERVICE_PIDS
    ):
        raise VerificationError("runtime PID saturation evidence violated its bounded headroom")
    return {
        "data_problem_code_counts": dict(sorted(problem_codes.items())),
        "data_status_counts": dict(sorted(counts.items())),
        "docker_healthcheck_recovered": True,
        "liveness_latency_ms": float(liveness_latency),
        "pids_headroom": pids_headroom,
        "pids_limit": EXPECTED_SERVICE_PIDS,
        "pids_peak": pids_peak,
        "readiness_code": "registry_busy",
        "readiness_latency_ms": float(readiness_latency),
        "readiness_retry_after_seconds": 1,
        "readiness_retryable": True,
    }


def _verify_running_container(document: Any, *, canary: str) -> None:
    if not isinstance(document, list) or len(document) != 1 or not isinstance(document[0], dict):
        raise VerificationError("running-container inspection has an invalid shape")
    inspection = document[0]
    _assert_canary_absent(inspection, canary)
    config = inspection.get("Config")
    host = inspection.get("HostConfig")
    state = inspection.get("State")
    mounts = inspection.get("Mounts")
    network = inspection.get("NetworkSettings")
    if (
        not isinstance(config, dict)
        or not isinstance(host, dict)
        or not isinstance(state, dict)
        or not isinstance(network, dict)
    ):
        raise VerificationError("running-container inspection is incomplete")
    if not isinstance(mounts, list):
        raise VerificationError("running-container mount inventory is invalid")

    environment = config.get("Env")
    _verify_healthcheck_config(config.get("Healthcheck"))
    if (
        config.get("User") != "10001:10001"
        or config.get("Entrypoint") != EXPECTED_SERVICE_ENTRYPOINT
        or config.get("Cmd") != EXPECTED_SERVICE_COMMAND
        or config.get("StopSignal") != "SIGTERM"
        or config.get("ExposedPorts") not in (None, {})
        or not isinstance(environment, list)
        or not all(isinstance(value, str) for value in environment)
        or any(value.startswith("SIGNALATTICE_REGISTRY_DIGEST_KEY=") for value in environment)
    ):
        raise VerificationError("Compose service process authority has drifted")

    expected_ulimits = {
        "core": (0, 0),
        "nofile": (256, 256),
        "nproc": (EXPECTED_SERVICE_PIDS, EXPECTED_SERVICE_PIDS),
    }
    ulimits = host.get("Ulimits")
    if not isinstance(ulimits, list):
        raise VerificationError("Compose ulimit inventory is invalid")
    observed_ulimits: dict[str, tuple[int, int]] = {}
    for value in ulimits:
        if not isinstance(value, dict):
            raise VerificationError("Compose ulimit entry is invalid")
        name = value.get("Name")
        soft = value.get("Soft")
        hard = value.get("Hard")
        if not isinstance(name, str) or type(soft) is not int or type(hard) is not int:
            raise VerificationError("Compose ulimit entry is invalid")
        observed_ulimits[name] = (soft, hard)
    log_config = host.get("LogConfig")
    restart = host.get("RestartPolicy")
    if (
        host.get("NetworkMode") != "none"
        or host.get("ReadonlyRootfs") is not True
        or host.get("Privileged") is not False
        or host.get("Init") is not True
        or host.get("CapDrop") != ["ALL"]
        or host.get("CapAdd") not in (None, [])
        or host.get("SecurityOpt") != ["no-new-privileges:true"]
        or host.get("PidsLimit") != EXPECTED_SERVICE_PIDS
        or host.get("Memory") != 768 * 1024 * 1024
        or host.get("MemoryReservation") != 256 * 1024 * 1024
        or host.get("NanoCpus") != 1_000_000_000
        or host.get("ShmSize") != 16 * 1024 * 1024
        or host.get("PortBindings") not in (None, {})
        or observed_ulimits != expected_ulimits
        or log_config != {"Type": "local", "Config": {"max-file": "2", "max-size": "8m"}}
        or not isinstance(restart, dict)
        or restart.get("Name") != "no"
    ):
        raise VerificationError("Compose service isolation or resource bounds have drifted")

    tmpfs = host.get("Tmpfs")
    if not isinstance(tmpfs, dict) or set(tmpfs) != {"/run/signalattice", "/tmp"}:
        raise VerificationError("Compose writable runtime inventory has drifted")
    _verify_tmpfs_options(tmpfs["/run/signalattice"], size=1024 * 1024, mode=0o700)
    _verify_tmpfs_options(tmpfs["/tmp"], size=16 * 1024 * 1024, mode=0o1770)

    expected_destinations = {
        "/var/lib/signalattice/registry",
        "/var/lib/signalattice/cas",
        "/run/secrets/registry-digest-key",
    }
    observed_destinations: set[str] = set()
    for mount in mounts:
        if not isinstance(mount, dict):
            raise VerificationError("Compose mount entry is invalid")
        destination = mount.get("Destination")
        if destination in {"/run/signalattice", "/tmp"} and mount.get("Type") == "tmpfs":
            continue
        if (
            not isinstance(destination, str)
            or destination not in expected_destinations
            or mount.get("Type") != "bind"
            or mount.get("RW") is not False
        ):
            raise VerificationError("Compose mounted authority has drifted")
        observed_destinations.add(destination)
    if observed_destinations != expected_destinations:
        raise VerificationError("Compose mounted authority is incomplete")

    health = state.get("Health")
    if (
        state.get("Running") is not True
        or not isinstance(health, dict)
        or health.get("Status") != "healthy"
        or network.get("Ports") not in (None, {})
    ):
        raise VerificationError("Compose socket readiness or network isolation failed")


def _bounded_container_logs(container_id: str, *, canary: bytes) -> int:
    try:
        result = subprocess.run(
            ["docker", "logs", "--tail", "1000", container_id],
            check=False,
            capture_output=True,
            timeout=30,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise VerificationError("running-container logs could not be inspected") from exc
    logs = result.stdout + result.stderr
    if result.returncode != 0 or len(logs) > MAX_RUNTIME_LOG_BYTES:
        raise VerificationError("running-container logs could not be inspected within bounds")
    if canary in logs or _WORKSTATION_PATH.search(logs) is not None:
        raise VerificationError("running-container logs contain private material")
    return len(logs)


def _atomic_json(path: Path, document: dict[str, Any]) -> None:
    payload = json.dumps(document, sort_keys=True, separators=(",", ":")) + "\n"
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        parent_metadata = path.parent.lstat()
        if (
            not stat.S_ISDIR(parent_metadata.st_mode)
            or path.parent.resolve(strict=True) != path.parent.absolute()
        ):
            raise VerificationError("summary parent must be a canonical directory")
        if path.is_symlink() or (path.exists() and not path.is_file()):
            raise VerificationError("summary destination is not a regular file")
        descriptor, temporary_name = tempfile.mkstemp(
            prefix=".container-evidence-", dir=path.parent
        )
        temporary = Path(temporary_name)
        try:
            with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as stream:
                stream.write(payload)
                stream.flush()
                os.fsync(stream.fileno())
            temporary.chmod(0o600)
            os.replace(temporary, path)
        finally:
            if temporary.exists():
                temporary.unlink()
    except VerificationError:
        raise
    except OSError as exc:
        raise VerificationError("sanitized evidence summary could not be published") from exc


def _container_command(args: argparse.Namespace) -> None:
    canary = _validated_canary(args.canary_env)
    revision = _validated_revision(args.source_revision)
    first = inspect_service_image(
        args.service_image_a,
        canary=canary,
        expected_revision=revision,
    )
    second = inspect_service_image(
        args.service_image_b,
        canary=canary,
        expected_revision=revision,
    )
    if first.application_digest != second.application_digest:
        raise VerificationError("clean builds produced different application digests")
    if first.config_digest != second.config_digest:
        raise VerificationError("clean builds produced different runtime config digests")
    if first.filesystem_digest != second.filesystem_digest:
        raise VerificationError("clean builds produced different filesystem digests")
    if (
        first.image_size_bytes != second.image_size_bytes
        or first.application_regular_bytes != second.application_regular_bytes
        or first.history_record_count != second.history_record_count
        or first.layer_count != second.layer_count
        or first.regular_file_count != second.regular_file_count
        or first.application_without_telemetry_source_artifacts_digest
        != second.application_without_telemetry_source_artifacts_digest
        or first.application_without_telemetry_source_artifacts_regular_bytes
        != second.application_without_telemetry_source_artifacts_regular_bytes
        or first.telemetry_source_artifact_digest != second.telemetry_source_artifact_digest
        or first.telemetry_source_artifact_bytes != second.telemetry_source_artifact_bytes
    ):
        raise VerificationError("clean builds produced different bounded inventories")

    first_provenance = _load_json(args.metadata_a)
    second_provenance = _load_json(args.metadata_b)
    first_identity = verify_provenance_document(
        first_provenance,
        canary=canary,
        expected_revision=revision,
    )
    second_identity = verify_provenance_document(
        second_provenance,
        canary=canary,
        expected_revision=revision,
    )
    if (
        first_identity != second_identity
        or first_identity.config_digest != first.image_config_digest
        or second_identity.config_digest != second.image_config_digest
    ):
        raise VerificationError("clean builds or provenance produced different image identities")

    cli_size = _inspect_cli_size(args.cli_image, canary=canary)
    service_cli_size_delta = first.image_size_bytes - cli_size
    _verify_telemetry_source_artifact_bytes(first.telemetry_source_artifact_bytes)

    _atomic_json(
        args.summary,
        {
            "schema_version": 1,
            "result": "passed",
            "platform": EXPECTED_PLATFORM,
            "source_revision": revision,
            "image_name": _validated_image_name(args.service_image_a),
            "builds": 2,
            "application_digest": f"sha256:{first.application_digest}",
            "runtime_config_digest": f"sha256:{first.config_digest}",
            "filesystem_digest_a": f"sha256:{first.filesystem_digest}",
            "filesystem_digest_b": f"sha256:{second.filesystem_digest}",
            "oci_config_digest": first_identity.config_digest,
            "oci_manifest_digest": first_identity.manifest_digest,
            "service_size_bytes_max": max(first.image_size_bytes, second.image_size_bytes),
            "cli_size_bytes": cli_size,
            "service_cli_size_delta_bytes": service_cli_size_delta,
            "application_regular_bytes": first.application_regular_bytes,
            "application_without_telemetry_source_artifacts_digest": (
                "sha256:" f"{first.application_without_telemetry_source_artifacts_digest}"
            ),
            "application_without_telemetry_source_artifacts_regular_bytes": (
                first.application_without_telemetry_source_artifacts_regular_bytes
            ),
            "telemetry_source_artifact_digest": (
                f"sha256:{first.telemetry_source_artifact_digest}"
            ),
            "telemetry_source_artifact_bytes": first.telemetry_source_artifact_bytes,
            "size_budget_bytes": MAX_IMAGE_BYTES,
            "telemetry_source_artifact_budget_bytes": (MAX_TELEMETRY_SOURCE_ARTIFACT_BYTES),
            "history_record_count": first.history_record_count,
            "layer_count": first.layer_count,
            "regular_file_count": first.regular_file_count,
            "credentials_present": False,
            "network_exposed": False,
        },
    )


def _runtime_command(args: argparse.Namespace) -> None:
    canary = _validated_canary(args.canary_env)
    if re.fullmatch(r"[0-9a-f]{64}", args.container_id) is None:
        raise VerificationError("running-container identity is invalid")
    payload = _run(
        ["docker", "container", "inspect", args.container_id],
        timeout_seconds=30,
    )
    if len(payload) > 4 * 1024 * 1024:
        raise VerificationError("running-container inspection exceeds its byte ceiling")
    try:
        inspection = json.loads(
            payload,
            parse_constant=lambda _value: (_ for _ in ()).throw(ValueError()),
        )
    except (JSONDecodeError, UnicodeDecodeError, RecursionError, ValueError) as exc:
        raise VerificationError("running-container inspection is invalid") from exc
    _verify_running_container(inspection, canary=canary)
    verify_runtime_probe_document(_load_json(args.probe), canary=canary)
    saturation = verify_runtime_saturation_document(
        _load_json(args.saturation),
        canary=canary,
    )
    log_bytes = _bounded_container_logs(args.container_id, canary=canary.encode("ascii"))
    _atomic_json(
        args.summary,
        {
            "schema_version": 1,
            "result": "passed",
            "profile": "compose-default-service",
            "network_mode": "none",
            "read_only_root": True,
            "uid": 10001,
            "gid": 10001,
            "socket_mode": "0600",
            "socket_parent_mode": "0700",
            "secret_mode": "0400",
            "secret_environment_present": False,
            "secret_source_match": True,
            "runtime_log_bytes": log_bytes,
            "saturation": saturation,
            "credentials_present": False,
        },
    )


def _scans_command(args: argparse.Namespace) -> None:
    canary = _validated_canary(args.canary_env)
    revision = _validated_revision(args.source_revision)
    image_name = _validated_image_name(args.image_name)
    repository_source_digest = _verify_repository_source(
        args.repository_root,
        expected_revision=revision,
    )
    container_summary = _load_json(args.container_summary)
    runtime_summary = _load_json(args.runtime_summary)
    if not isinstance(container_summary, dict):
        raise VerificationError("container summary has an invalid shape")
    _assert_canary_absent(container_summary, canary)
    if not isinstance(runtime_summary, dict):
        raise VerificationError("runtime summary has an invalid shape")
    _assert_canary_absent(runtime_summary, canary)
    if (
        runtime_summary.get("schema_version") != 1
        or runtime_summary.get("result") != "passed"
        or runtime_summary.get("profile") != "compose-default-service"
        or runtime_summary.get("credentials_present") is not False
        or not isinstance(runtime_summary.get("saturation"), dict)
    ):
        raise VerificationError("runtime summary is not verified Compose evidence")
    runtime_digest = hashlib.sha256(
        json.dumps(runtime_summary, sort_keys=True, separators=(",", ":")).encode("ascii")
    ).hexdigest()
    config_digest = container_summary.get("oci_config_digest")
    manifest_digest = container_summary.get("oci_manifest_digest")
    if (
        container_summary.get("schema_version") != 1
        or container_summary.get("result") != "passed"
        or container_summary.get("source_revision") != revision
        or container_summary.get("image_name") != image_name
        or not isinstance(config_digest, str)
        or _DIGEST.fullmatch(config_digest) is None
        or not isinstance(manifest_digest, str)
        or _DIGEST.fullmatch(manifest_digest) is None
    ):
        raise VerificationError("scanner inputs are not bound to the verified image summary")
    sbom_identity = f"{image_name}@{manifest_digest}"
    sbom = verify_spdx_document(
        _load_json(args.sbom),
        canary=canary,
        expected_name=sbom_identity,
        expected_revision=revision,
    )
    repository = verify_trivy_document(
        _load_json(args.repository_scan),
        canary=canary,
        expected_name="/workspace",
        expected_type="repository",
        expected_revision=revision,
    )
    image_security = verify_trivy_document(
        _load_json(args.image_scan),
        canary=canary,
        expected_name=_TRIVY_IMAGE_ARCHIVE_NAME,
        expected_type="container_image",
        expected_revision=revision,
        expected_config_digest=config_digest,
        expected_image_reference=image_name,
    )
    image_licenses = verify_trivy_document(
        _load_json(args.image_license_scan),
        canary=canary,
        expected_name=_TRIVY_IMAGE_ARCHIVE_NAME,
        expected_type="container_image",
        expected_revision=revision,
        expected_config_digest=config_digest,
        expected_image_reference=image_name,
    )
    if image_security.artifact_id != image_licenses.artifact_id:
        raise VerificationError("image security and license scans target different artifacts")
    if (
        image_security.counters["license_findings"] != 0
        or image_licenses.counters["license_findings"] < 1
        or image_licenses.counters["fixed_vulnerabilities"] != 0
        or image_licenses.counters["unfixed_vulnerabilities"] != 0
        or image_licenses.counters["secrets"] != 0
        or image_licenses.counters["material_misconfigurations"] != 0
    ):
        raise VerificationError("image scan evidence crossed its declared scanner boundary")
    benchmark_digest = verify_benchmark_canary_evidence(
        _load_json(args.benchmark_evidence),
        canary=canary,
    )
    unresolved = sbom.unresolved_license_packages - image_licenses.licensed_packages
    if unresolved:
        raise VerificationError("SBOM contains packages without a reviewed license decision")
    _atomic_json(
        args.summary,
        {
            "schema_version": 1,
            "result": "passed",
            "source_revision": revision,
            "image_name": image_name,
            "image_config_digest": config_digest,
            "image_manifest_digest": manifest_digest,
            "sbom_format": "SPDX-2.3",
            "sbom_packages": sbom.packages,
            "sbom_license_expressions": sbom.license_expressions,
            "sbom_unresolved_licenses": 0,
            "repository_report_sha256": repository.report_digest,
            "repository_artifact_id": repository.artifact_id,
            "repository_source_identity_sha256": repository_source_digest,
            "image_artifact_id": image_security.artifact_id,
            "image_security_report_sha256": image_security.report_digest,
            "image_license_report_sha256": image_licenses.report_digest,
            "benchmark_canonical_payload_sha256": benchmark_digest,
            "runtime_summary_sha256": runtime_digest,
            "repository": repository.counters,
            "image_security": image_security.counters,
            "image_licenses": image_licenses.counters,
            "scanner_exceptions": 0,
            "scanner_suppressions": 0,
            "credentials_present": False,
        },
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    container = commands.add_parser("container", help="Verify clean image builds and metadata.")
    container.add_argument("--service-image-a", required=True)
    container.add_argument("--service-image-b", required=True)
    container.add_argument("--cli-image", required=True)
    container.add_argument("--metadata-a", type=Path, required=True)
    container.add_argument("--metadata-b", type=Path, required=True)
    container.add_argument("--canary-env", required=True)
    container.add_argument("--source-revision", required=True)
    container.add_argument("--summary", type=Path, required=True)
    container.set_defaults(handler=_container_command)

    runtime = commands.add_parser("runtime", help="Verify the exact Compose service runtime.")
    runtime.add_argument("--container-id", required=True)
    runtime.add_argument("--probe", type=Path, required=True)
    runtime.add_argument("--saturation", type=Path, required=True)
    runtime.add_argument("--canary-env", required=True)
    runtime.add_argument("--summary", type=Path, required=True)
    runtime.set_defaults(handler=_runtime_command)

    scans = commands.add_parser("scans", help="Verify SBOM and scanner results.")
    scans.add_argument("--sbom", type=Path, required=True)
    scans.add_argument("--repository-scan", type=Path, required=True)
    scans.add_argument("--repository-root", type=Path, required=True)
    scans.add_argument("--image-scan", type=Path, required=True)
    scans.add_argument("--image-license-scan", type=Path, required=True)
    scans.add_argument("--container-summary", type=Path, required=True)
    scans.add_argument("--runtime-summary", type=Path, required=True)
    scans.add_argument("--benchmark-evidence", type=Path, required=True)
    scans.add_argument("--image-name", required=True)
    scans.add_argument("--source-revision", required=True)
    scans.add_argument("--canary-env", required=True)
    scans.add_argument("--summary", type=Path, required=True)
    scans.set_defaults(handler=_scans_command)
    return parser


def main() -> int:
    args = _parser().parse_args()
    try:
        args.handler(args)
    except VerificationError as exc:
        print(f"service-container verification failed: {exc}", file=sys.stderr)
        return 1
    except Exception:
        # This is the outer trust boundary: fail the gate without reflecting an
        # unexpected library, filesystem, Docker, or parser diagnostic that may
        # contain a path, image label, scanner match, or credential fragment.
        print("service-container verification failed: unexpected internal failure", file=sys.stderr)
        return 1
    print("service-container evidence verified")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
