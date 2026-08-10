"""Static, adversarial, and fail-closed service-container contract tests."""

from __future__ import annotations

import hashlib
import importlib.util
import io
import json
import re
import shutil
import subprocess
import sys
import tarfile
from argparse import Namespace
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "verify_service_container.py"
BENCHMARK_SCRIPT = ROOT / "scripts" / "benchmark_service_operability.py"
CANARY = "SF_CONTAINER_CANARY_" + ("c" * 64)
REVISION = "0123456789abcdef0123456789abcdef01234567"
CONFIG_DIGEST = "sha256:" + ("d" * 64)
MANIFEST_DIGEST = "sha256:" + ("e" * 64)
IMAGE_NAME = "signalattice-service:clean-a"
TRIVY_IMAGE_ARCHIVE_NAME = "/scan/service-image.tar"
SBOM_NAME = f"{IMAGE_NAME}@{MANIFEST_DIGEST}"


def _load_verifier() -> ModuleType:
    spec = importlib.util.spec_from_file_location("verify_service_container", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


verifier = _load_verifier()


def _load_benchmark() -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        "benchmark_service_operability_container_contract",
        BENCHMARK_SCRIPT,
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _text(relative: str) -> str:
    return (ROOT / relative).read_text(encoding="utf-8")


def _minimal_trivy_result(**overrides: Any) -> dict[str, Any]:
    result: dict[str, Any] = {
        "Target": "redacted-target",
        "Class": "lang-pkgs",
        "Type": "python-pkg",
    }
    result.update(overrides)
    return {
        "SchemaVersion": 2,
        "ArtifactName": "signalattice-test-target",
        "ArtifactType": "filesystem",
        "ArtifactID": "a" * 64,
        "ReportID": "report-identity",
        "Metadata": {"Repo": {"Commit": REVISION}},
        "Results": [result],
    }


def _repository_metadata() -> dict[str, str]:
    return {
        "Author": "Signalattice Test <signalattice-test@example.invalid>",
        "Branch": "dev",
        "Commit": REVISION,
        "CommitMsg": "Exercise the repository scanner contract",
        "Committer": "Signalattice Test <signalattice-test@example.invalid>",
        "RepoURL": "https://github.com/srgangaram-swe/Signalattice.git",
    }


def _image_metadata() -> dict[str, Any]:
    diff_ids = ["sha256:" + ("1" * 64), "sha256:" + ("2" * 64)]
    return {
        "DiffIDs": diff_ids,
        "ImageConfig": {
            "architecture": "amd64",
            "config": {
                "Cmd": list(verifier.EXPECTED_SERVICE_COMMAND),
                "Entrypoint": list(verifier.EXPECTED_SERVICE_ENTRYPOINT),
                "Labels": {"org.opencontainers.image.revision": REVISION},
            },
            "created": "2026-08-09T00:00:00Z",
            "history": [],
            "os": "linux",
            "rootfs": {"type": "layers", "diff_ids": diff_ids},
        },
        "ImageID": CONFIG_DIGEST,
        "Layers": [
            {
                "DiffID": diff_id,
                "Digest": "sha256:" + (str(index + 3) * 64),
                "Size": 1_024 * (index + 1),
            }
            for index, diff_id in enumerate(diff_ids)
        ],
        "OS": {"Family": "debian", "Name": "12.13"},
        "Reference": IMAGE_NAME,
        "RepoTags": [IMAGE_NAME],
        "Size": 16 * 1024 * 1024,
    }


def _minimal_image_trivy_result(**overrides: Any) -> dict[str, Any]:
    document = _minimal_trivy_result(**overrides)
    document.update(
        {
            "ArtifactName": TRIVY_IMAGE_ARCHIVE_NAME,
            "ArtifactType": "container_image",
            "Metadata": _image_metadata(),
        }
    )
    return document


def _minimal_spdx(**overrides: Any) -> dict[str, Any]:
    document: dict[str, Any] = {
        "spdxVersion": "SPDX-2.3",
        "SPDXID": "SPDXRef-DOCUMENT",
        "dataLicense": "CC0-1.0",
        "name": SBOM_NAME,
        "documentNamespace": "https://anchore.com/syft/test",
        "creationInfo": {"creators": ["Tool: syft-1.50.0"]},
        "packages": [
            {
                "SPDXID": "SPDXRef-Package-signalattice",
                "name": "signalattice",
                "licenseConcluded": "MIT",
                "licenseDeclared": "MIT",
            },
            {
                "SPDXID": "SPDXRef-DocumentRoot-Container-service",
                "name": SBOM_NAME,
                "versionInfo": REVISION,
                "licenseConcluded": "NOASSERTION",
                "licenseDeclared": "NOASSERTION",
            },
        ],
    }
    document.update(overrides)
    return document


def _minimal_benchmark_evidence() -> dict[str, Any]:
    common_workload = verifier._expected_isolated_rss_workload()
    workload_sha256 = hashlib.sha256(
        json.dumps(
            common_workload,
            ensure_ascii=True,
            allow_nan=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("ascii")
    ).hexdigest()

    def rss_child(mode: str, peak: int) -> dict[str, Any]:
        before = 40 * 1024 * 1024
        return {
            "protocol": "signalattice-service-rss-v1",
            "mode": mode,
            "workload_sha256": workload_sha256,
            "runtime_platform": "Linux-test-amd64",
            "python": "3.13.7",
            "python_implementation": "CPython",
            "rss_source": "resource.getrusage(RUSAGE_SELF).ru_maxrss",
            "rss_unit": "bytes",
            "rss_before_app_bytes": before,
            "rss_after_shutdown_peak_bytes": peak,
            "workload_growth_bytes": peak - before,
            "warmup_requests": 8,
            "measured_requests": 16,
            "successful_requests": 16,
            "status_counts": {"200": 16},
            "lifespan_shutdown_completed": True,
            "network_requests": 0,
            "processes": 1,
        }

    disabled = rss_child("disabled", 50 * 1024 * 1024)
    enabled = rss_child("enabled", 60 * 1024 * 1024)
    delta = enabled["rss_after_shutdown_peak_bytes"] - disabled["rss_after_shutdown_peak_bytes"]
    isolated_rss = {
        "method": "fresh_process_peak_rss_comparison",
        "protocol": "signalattice-service-rss-v1",
        "process_order": ["disabled", "enabled"],
        "hard_timeout_seconds_per_process": 20.0,
        "common_workload": common_workload,
        "workload_sha256": workload_sha256,
        "disabled": disabled,
        "enabled": enabled,
        "comparison_compatible": True,
        "observed_signed_delta_bytes": delta,
        "nonnegative_delta_bytes": max(0, delta),
        "limit_bytes": verifier.MAX_TELEMETRY_RSS_DELTA_BYTES,
        "within_limit": True,
        "negative_delta_policy": (
            "retain the signed observation; clamp only the incremental-overhead guard to zero"
        ),
    }
    document: dict[str, Any] = {
        "schema_version": "1.0.0",
        "evidence_id": "signalattice-service-operability-synthetic-local-v1",
        "evidence_class": "measured_synthetic_local_engineering",
        "provenance": {
            "network_requests": 0,
            "credentials_used": False,
            "market_or_model_data_used": False,
            "source_tree_sha256": "a" * 64,
        },
        "saturation": {"probe_status": 200, "rejection_count": 24},
        "telemetry_ab": {"isolated_process_rss": isolated_rss},
        "bounds_observed": [
            {
                "name": "telemetry_isolated_rss_delta_bytes",
                "display_name": "Telemetry RSS delta",
                "observed": max(0, delta),
                "limit": verifier.MAX_TELEMETRY_RSS_DELTA_BYTES,
                "unit": "bytes",
            }
        ],
        "fault_cases": [
            {
                "case": "corrupt_manifest",
                "status": 503,
                "closed_code": "evidence_integrity_failed",
            }
        ],
    }
    canonical = json.dumps(
        document,
        ensure_ascii=True,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("ascii")
    document["integrity"] = {
        "algorithm": "sha256",
        "canonicalization": "sorted compact ASCII JSON excluding the integrity member",
        "canonical_payload_sha256": hashlib.sha256(canonical).hexdigest(),
    }
    return document


def _minimal_inspection() -> dict[str, Any]:
    return {
        "Id": CONFIG_DIGEST,
        "Architecture": "amd64",
        "Os": "linux",
        "Size": 512 * 1024 * 1024,
        "RootFS": {"Layers": ["sha256:" + ("a" * 64)]},
        "Config": {
            "User": "10001:10001",
            "Entrypoint": list(verifier.EXPECTED_SERVICE_ENTRYPOINT),
            "Cmd": list(verifier.EXPECTED_SERVICE_COMMAND),
            "WorkingDir": "/app",
            "ExposedPorts": None,
            "Volumes": None,
            "Env": [
                "HOME=/nonexistent",
                "PATH=/opt/venv/bin:/usr/bin:/bin",
                "PYTHONPATH=/app/src",
            ],
            "Healthcheck": {
                "Test": list(verifier.EXPECTED_HEALTHCHECK_TEST),
                "Interval": 10_000_000_000,
                "Timeout": 2_000_000_000,
                "StartPeriod": 5_000_000_000,
                "Retries": 3,
            },
            "Labels": {
                "org.opencontainers.image.licenses": "MIT",
                "org.opencontainers.image.revision": REVISION,
                "org.opencontainers.image.source": "https://github.com/srgangaram-swe/Signalattice",
            },
        },
    }


def _minimal_runtime_inspection() -> list[dict[str, Any]]:
    return [
        {
            "Config": {
                "User": "10001:10001",
                "Entrypoint": list(verifier.EXPECTED_SERVICE_ENTRYPOINT),
                "Cmd": list(verifier.EXPECTED_SERVICE_COMMAND),
                "StopSignal": "SIGTERM",
                "ExposedPorts": None,
                "Env": [
                    "HOME=/nonexistent",
                    "PATH=/opt/venv/bin:/usr/bin:/bin",
                    "PYTHONPATH=/app/src",
                ],
                "Healthcheck": {
                    "Test": list(verifier.EXPECTED_HEALTHCHECK_TEST),
                    "Interval": 10_000_000_000,
                    "Timeout": 2_000_000_000,
                    "StartPeriod": 5_000_000_000,
                    "Retries": 3,
                },
            },
            "HostConfig": {
                "NetworkMode": "none",
                "ReadonlyRootfs": True,
                "Privileged": False,
                "Init": True,
                "CapDrop": ["ALL"],
                "CapAdd": None,
                "SecurityOpt": ["no-new-privileges:true"],
                "PidsLimit": 64,
                "Memory": 768 * 1024 * 1024,
                "MemoryReservation": 256 * 1024 * 1024,
                "NanoCpus": 1_000_000_000,
                "ShmSize": 16 * 1024 * 1024,
                "PortBindings": None,
                "Ulimits": [
                    {"Name": "core", "Soft": 0, "Hard": 0},
                    {"Name": "nofile", "Soft": 256, "Hard": 256},
                    {"Name": "nproc", "Soft": 64, "Hard": 64},
                ],
                "LogConfig": {
                    "Type": "local",
                    "Config": {"max-file": "2", "max-size": "8m"},
                },
                "RestartPolicy": {"Name": "no"},
                "Tmpfs": {
                    "/run/signalattice": "rw,noexec,nosuid,nodev,size=1m,mode=0700,uid=10001,gid=10001",
                    "/tmp": "rw,noexec,nosuid,nodev,size=16m,mode=1770,uid=10001,gid=10001",
                },
            },
            "Mounts": [
                {
                    "Type": "bind",
                    "Destination": "/var/lib/signalattice/registry",
                    "RW": False,
                },
                {
                    "Type": "bind",
                    "Destination": "/var/lib/signalattice/cas",
                    "RW": False,
                },
                {
                    "Type": "bind",
                    "Destination": "/run/secrets/registry-digest-key",
                    "RW": False,
                },
            ],
            "State": {"Running": True, "Health": {"Status": "healthy"}},
            "NetworkSettings": {"Ports": None},
        }
    ]


def _minimal_runtime_probe() -> dict[str, Any]:
    return {
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


def _minimal_runtime_saturation() -> dict[str, Any]:
    return {
        "data_problem_code_counts": {
            "evidence_unavailable": 24,
            "service_saturated": 24,
        },
        "data_status_counts": {"429": 24, "503": 24},
        "docker_healthcheck_recovered": True,
        "liveness_latency_ms": 12.5,
        "liveness_status": 200,
        "pids_headroom": 32,
        "pids_limit": 64,
        "pids_peak": 32,
        "readiness_code": "registry_busy",
        "readiness_latency_ms": 251.0,
        "readiness_retry_after_seconds": 1,
        "readiness_retryable": True,
        "readiness_status": 503,
        "result": "passed",
        "schema_version": 1,
        "submitted_requests": 48,
    }


def _verify_trivy(
    document: dict[str, Any],
    *,
    expected_config_digest: str | None = None,
    expected_revision: str | None = REVISION,
    expected_type: str = "filesystem",
    require_artifact_id: bool = True,
) -> Any:
    return verifier.verify_trivy_document(
        document,
        canary=CANARY,
        expected_name=(
            TRIVY_IMAGE_ARCHIVE_NAME
            if expected_type == "container_image"
            else "signalattice-test-target"
        ),
        expected_type=expected_type,
        expected_revision=expected_revision,
        expected_config_digest=(
            CONFIG_DIGEST
            if expected_type == "container_image" and expected_config_digest is None
            else expected_config_digest
        ),
        expected_image_reference=(IMAGE_NAME if expected_type == "container_image" else None),
        require_artifact_id=require_artifact_id,
    )


def _write_rootfs_tar(
    path: Path,
    *,
    extra_path: str | None = None,
    extra_payload: bytes = b"",
    link: tuple[str, str, bool] | None = None,
    omit_distribution: str | None = None,
) -> None:
    def directory(name: str, *, mode: int = 0o755, uid: int = 0, gid: int = 0) -> tarfile.TarInfo:
        info = tarfile.TarInfo(name)
        info.type = tarfile.DIRTYPE
        info.mode = mode
        info.uid = uid
        info.gid = gid
        return info

    def regular(
        name: str, payload: bytes, *, mode: int = 0o644, uid: int = 0, gid: int = 0
    ) -> tuple[tarfile.TarInfo, io.BytesIO]:
        info = tarfile.TarInfo(name)
        info.type = tarfile.REGTYPE
        info.mode = mode
        info.uid = uid
        info.gid = gid
        info.size = len(payload)
        return info, io.BytesIO(payload)

    passwd = (
        b"root:x:0:0:root:/root:/bin/bash\n"
        b"signalattice:x:10001:10001::/nonexistent:/usr/sbin/nologin\n"
    )
    group = b"root:x:0:\nsignalattice:x:10001:\n"
    with tarfile.open(path, "w") as archive:
        directory_names = {
            "app",
            "app/src",
            "app/src/quant_platform",
            "app/src/quant_platform/service",
            "app/src/quant_platform/tracking",
            "etc",
            "opt",
            "opt/venv",
            "opt/venv/bin",
            "opt/venv/lib",
            "opt/venv/lib/python3.13",
            "opt/venv/lib/python3.13/site-packages",
            "run",
            "var",
            "var/lib",
            "var/lib/signalattice",
        }
        directory_names.update(
            "opt/venv/lib/python3.13/site-packages/" + distribution
            for distribution in verifier._EXPECTED_SERVICE_DISTRIBUTIONS
            if distribution != omit_distribution
        )
        for name in sorted(directory_names):
            archive.addfile(directory(name))
        for info in (
            directory("run/signalattice", mode=0o700, uid=10001, gid=10001),
            directory("var/lib/signalattice/registry", mode=0o550, gid=10001),
            directory("var/lib/signalattice/cas", mode=0o550, gid=10001),
        ):
            archive.addfile(info)
        for info, payload in (
            regular("etc/passwd", passwd),
            regular("etc/group", group),
            regular("opt/venv/bin/python", b"python\n", mode=0o555),
        ):
            archive.addfile(info, payload)
        for source_path in sorted(verifier._EXPECTED_SERVICE_SOURCE_FILES):
            info, payload = regular(source_path, b'"""reviewed source"""\n', mode=0o444)
            archive.addfile(info, payload)
        if extra_path is not None:
            info, payload = regular(extra_path, extra_payload)
            archive.addfile(info, payload)
        if link is not None:
            name, target, hard_link = link
            info = tarfile.TarInfo(name)
            info.type = tarfile.LNKTYPE if hard_link else tarfile.SYMTYPE
            info.mode = 0o777
            info.linkname = target
            archive.addfile(info)


def _install_fake_export(monkeypatch: pytest.MonkeyPatch, archive: Path) -> None:
    def fake_run(command: list[str], *, timeout_seconds: int = 60) -> bytes:
        del timeout_seconds
        if command[:2] == ["docker", "create"]:
            return ("a" * 64).encode()
        if command[:2] == ["docker", "export"]:
            shutil.copyfile(archive, Path(command[3]))
            return b""
        if command[:2] == ["docker", "rm"]:
            return b""
        raise AssertionError("unexpected test command")

    monkeypatch.setattr(verifier, "_run", fake_run)


def test_dockerfile_pins_every_frontend_and_base_identity() -> None:
    dockerfile = _text("Dockerfile")
    first_line = dockerfile.splitlines()[0]
    assert re.fullmatch(r"# syntax=docker/dockerfile:[^@\s]+@sha256:[0-9a-f]{64}", first_line)
    assert dockerfile.splitlines()[1] == "# check=error=true"

    from_lines = [line for line in dockerfile.splitlines() if line.startswith("FROM ")]
    image_references = [line.split()[1] for line in from_lines if "@sha256:" in line]
    assert len(image_references) == 3
    assert all(re.fullmatch(r"[^\s]+@sha256:[0-9a-f]{64}", image) for image in image_references)
    assert all("latest" not in image for image in image_references)
    assert dockerfile.count("python:3.13-slim-bookworm@sha256:") == 2
    assert "ghcr.io/astral-sh/uv:0.11.32@sha256:" in dockerfile
    assert 'org.opencontainers.image.revision="${VCS_REF}"' in dockerfile


def test_dockerfile_uses_only_frozen_locked_installs_and_no_runtime_installer() -> None:
    dockerfile = _text("Dockerfile")
    assert "pip install" not in dockerfile
    assert "apt-get" not in dockerfile
    assert "curl " not in dockerfile
    assert "wget " not in dockerfile
    assert "uv sync --locked --no-dev --no-editable" in dockerfile
    assert dockerfile.count("uv sync --locked --no-dev --no-editable") == 2
    assert "uv sync --locked --no-dev --no-editable --extra dev" in dockerfile
    assert "uv sync --locked --only-group service-runtime --no-install-project" in dockerfile
    assert "FROM cli-builder AS service-builder" not in dockerfile
    assert "FROM builder-base AS service-builder" in dockerfile
    assert "UV_NO_CACHE=1" in dockerfile
    assert "--mount=type=cache" not in dockerfile
    assert "UV_PYTHON_DOWNLOADS=never" in dockerfile
    assert "COPY --from=service-builder --chown=0:0 /opt/venv /opt/venv" in dockerfile
    assert "rm -rf /usr/local/lib/python3.13/site-packages" in dockerfile
    assert "/usr/local/lib/python3.13/ensurepip" in dockerfile
    assert "/usr/local/bin/pip3.13" in dockerfile
    assert "find / -xdev -type f -perm /6000 -exec chmod a-s {} +" in dockerfile


def test_dockerfile_preserves_cli_and_minimizes_dedicated_service_target() -> None:
    dockerfile = _text("Dockerfile")
    service_start = dockerfile.index("FROM runtime-base AS service")
    cli_start = dockerfile.index("FROM runtime-base AS cli")
    assert service_start < cli_start
    assert dockerfile.rstrip().endswith('CMD ["--help"]')
    service = dockerfile[service_start:cli_start]
    assert "COPY --chown=10001:10001 configs" not in service
    assert "COPY --chown=10001:10001 scripts" not in service
    assert "COPY --chown=10001:10001 data" not in service
    assert "EXPOSE" not in service
    assert "USER 10001:10001" in service
    assert "PYTHONPATH=/app/src" in service
    assert 'ENTRYPOINT ["python", "-m", "quant_platform.service"]' in service
    assert "src/quant_platform/service/entrypoint.py" in service
    assert "src/quant_platform/tracking/registry.py" in service
    assert "src/quant_platform/tracking/experiment.py" not in service
    assert "COPY src ./src" not in service
    copied_service_sources = set(
        re.findall(r"^\s+(src/quant_platform/[^\s]+)\s+\\?$", service, flags=re.MULTILINE)
    )
    assert {
        "app/" + source for source in copied_service_sources
    } == verifier._EXPECTED_SERVICE_SOURCE_FILES
    assert "/run/signalattice/api.sock" in service
    assert "/health/ready" in service
    assert "readline(4097)" in service
    assert ".recv(4096)" not in service
    for entrypoint_argument in verifier.EXPECTED_SERVICE_ENTRYPOINT:
        assert f'"{entrypoint_argument}"' in service
    for argument in verifier.EXPECTED_SERVICE_COMMAND:
        assert f'"{argument}"' in service
    healthcheck_json = next(
        line.strip()[4:]
        for line in dockerfile.splitlines()
        if line.strip().startswith('CMD ["python", "-c"')
    )
    assert ["CMD", *json.loads(healthcheck_json)] == verifier.EXPECTED_HEALTHCHECK_TEST


def test_exact_service_source_inventory_closes_recursive_imports_without_installed_project(
    tmp_path: Path,
) -> None:
    isolated_source = tmp_path / "isolated" / "src"
    for container_path in verifier._EXPECTED_SERVICE_SOURCE_FILES:
        repository_path = ROOT / Path(container_path).relative_to("app")
        destination = isolated_source / Path(container_path).relative_to("app/src")
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(repository_path, destination)

    program = """
import pathlib
import sys

root = pathlib.Path(sys.argv[1]).resolve()
sys.path.insert(0, str(root))
import quant_platform.service.entrypoint  # noqa: F401
import quant_platform.tracking.retention  # noqa: F401

for name, module in tuple(sys.modules.items()):
    if name == "quant_platform" or name.startswith("quant_platform."):
        path = pathlib.Path(module.__file__).resolve()
        path.relative_to(root)
"""
    result = subprocess.run(
        [sys.executable, "-B", "-I", "-c", program, str(isolated_source)],
        cwd=tmp_path,
        check=False,
        capture_output=True,
        text=True,
        timeout=20,
    )

    assert result.returncode == 0, result.stderr
    assert result.stdout == ""


def test_healthcheck_bounded_reader_accepts_a_fragmented_exact_200_status(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FragmentedStatus:
        def __init__(self) -> None:
            self.fragments = [b"HTTP/1.1 2", b"00 OK\r", b"\nunused"]
            self.requested_limit: int | None = None

        def readline(self, limit: int) -> bytes:
            self.requested_limit = limit
            result = bytearray()
            while self.fragments and len(result) < limit:
                fragment = self.fragments.pop(0)
                newline = fragment.find(b"\n")
                if newline >= 0:
                    result.extend(fragment[: newline + 1])
                    break
                result.extend(fragment)
            return bytes(result[:limit])

        def close(self) -> None:
            return None

    reader = FragmentedStatus()

    class FakeSocket:
        def settimeout(self, timeout: int) -> None:
            assert timeout == 1

        def connect(self, path: str) -> None:
            assert path == "/run/signalattice/api.sock"

        def sendall(self, request: bytes) -> None:
            assert request.startswith(b"GET /health/ready HTTP/1.1\r\n")

        def makefile(self, mode: str) -> FragmentedStatus:
            assert mode == "rb"
            return reader

        def recv(self, _size: int) -> bytes:
            raise AssertionError("healthcheck must not parse a single recv fragment")

        def close(self) -> None:
            return None

    socket_module = ModuleType("socket")
    socket_module.AF_UNIX = object()  # type: ignore[attr-defined]
    socket_module.socket = lambda _family: FakeSocket()  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "socket", socket_module)

    with pytest.raises(SystemExit) as exited:
        exec(verifier.EXPECTED_HEALTHCHECK_TEST[3], {})

    assert exited.value.code == 0
    assert reader.requested_limit == 4097


def test_runtime_identity_has_no_created_home_or_login_shell() -> None:
    dockerfile = _text("Dockerfile")
    assert "--uid 10001 --gid 10001 --no-create-home" in dockerfile
    assert "--home-dir /nonexistent --shell /usr/sbin/nologin" in dockerfile
    assert "-m 0700 /run/signalattice" in dockerfile
    assert "HOME=/nonexistent" in dockerfile
    runtime = dockerfile[dockerfile.rindex("FROM python:3.13-slim-bookworm") :]
    for line in runtime.splitlines():
        if line.startswith("RUN "):
            assert "--network=none" in line


def test_compose_service_is_local_read_only_and_capability_free() -> None:
    compose = yaml.safe_load(_text("docker-compose.yml"))
    service = compose["services"]["service"]
    assert service["build"]["target"] == "service"
    assert service["pull_policy"] == "never"
    assert service["user"] == "10001:10001"
    assert service["privileged"] is False
    assert service["read_only"] is True
    assert service["network_mode"] == "none"
    assert service["restart"] == "no"
    assert service["cap_drop"] == ["ALL"]
    assert service["security_opt"] == ["no-new-privileges:true"]
    assert service["pids_limit"] == 64
    assert service["cpus"] == 1.0
    assert service["mem_limit"] == "768m"
    assert service["shm_size"] == "16m"
    assert service["stop_grace_period"] == "15s"
    assert service["logging"] == {
        "driver": "local",
        "options": {"max-size": "8m", "max-file": "2"},
    }
    assert service["ulimits"]["nofile"] == {"soft": 256, "hard": 256}
    assert service["ulimits"]["nproc"] == {"soft": 64, "hard": 64}
    assert service["command"] == verifier.EXPECTED_SERVICE_COMMAND
    assert "ports" not in service and "expose" not in service
    assert "SIGNALATTICE_REGISTRY_DIGEST_KEY" not in service["environment"]
    assert not any("seccomp=unconfined" in option for option in service["security_opt"])


def test_compose_mounts_are_read_only_except_bounded_hardened_tmpfs() -> None:
    compose = yaml.safe_load(_text("docker-compose.yml"))
    service = compose["services"]["service"]
    assert len(service["volumes"]) == 2
    for mount in service["volumes"]:
        assert mount["type"] == "bind"
        assert mount["read_only"] is True
        assert mount["bind"]["create_host_path"] is False
    assert {mount["target"] for mount in service["volumes"]} == {
        "/var/lib/signalattice/registry",
        "/var/lib/signalattice/cas",
    }
    assert len(service["tmpfs"]) == 2
    for mount in service["tmpfs"]:
        assert all(option in mount for option in ("noexec", "nosuid", "nodev", "size="))
    assert "mode=0700" in service["tmpfs"][0]
    secret = service["secrets"][0]
    assert secret == {
        "source": "registry-digest-key",
        "target": "registry-digest-key",
        "uid": "10001",
        "gid": "10001",
        "mode": 0o400,
    }
    assert compose["secrets"]["registry-digest-key"] == {
        "environment": "SIGNALATTICE_REGISTRY_DIGEST_KEY"
    }


def test_compose_keeps_cli_explicit_and_networkless() -> None:
    compose = yaml.safe_load(_text("docker-compose.yml"))
    platform = compose["services"]["platform"]
    assert platform["profiles"] == ["cli"]
    assert platform["build"]["target"] == "cli"
    assert platform["image"] == "signalattice:latest"
    assert platform["network_mode"] == "none"
    assert platform["read_only"] is True
    assert platform["cap_drop"] == ["ALL"]


def test_runtime_verifier_accepts_exact_compose_inspection_and_probe() -> None:
    verifier._verify_running_container(_minimal_runtime_inspection(), canary=CANARY)
    verifier.verify_runtime_probe_document(_minimal_runtime_probe(), canary=CANARY)
    evidence = verifier.verify_runtime_saturation_document(
        _minimal_runtime_saturation(), canary=CANARY
    )
    assert evidence["pids_headroom"] == 32


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("liveness_status", 503, "behavior"),
        ("readiness_code", "evidence_unavailable", "behavior"),
        ("readiness_retryable", False, "behavior"),
        ("readiness_retry_after_seconds", 2, "behavior"),
        ("readiness_latency_ms", 1_001.0, "latency"),
        ("pids_peak", 61, "PID"),
        ("docker_healthcheck_recovered", False, "behavior"),
    ],
)
def test_runtime_saturation_verifier_rejects_probe_pid_and_recovery_drift(
    field: str, value: Any, message: str
) -> None:
    document = _minimal_runtime_saturation()
    document[field] = value
    if field == "pids_peak":
        document["pids_headroom"] = 64 - value
    with pytest.raises(verifier.VerificationError, match=message):
        verifier.verify_runtime_saturation_document(document, canary=CANARY)


def test_runtime_saturation_verifier_binds_statuses_to_problem_causes() -> None:
    document = _minimal_runtime_saturation()
    document["data_problem_code_counts"] = {
        "evidence_unavailable": 23,
        "service_saturated": 25,
    }
    with pytest.raises(verifier.VerificationError, match="behavior"):
        verifier.verify_runtime_saturation_document(document, canary=CANARY)


def test_runtime_saturation_verifier_rejects_lock_blocked_success() -> None:
    document = _minimal_runtime_saturation()
    document["data_status_counts"] = {"200": 1, "429": 23, "503": 24}
    document["data_problem_code_counts"] = {
        "evidence_unavailable": 24,
        "service_saturated": 23,
    }
    with pytest.raises(verifier.VerificationError, match="behavior"):
        verifier.verify_runtime_saturation_document(document, canary=CANARY)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("ReadonlyRootfs", False),
        ("NetworkMode", "bridge"),
        ("PidsLimit", 0),
        ("LogConfig", {"Type": "json-file", "Config": {}}),
    ],
)
def test_runtime_verifier_rejects_compose_authority_drift(field: str, value: Any) -> None:
    inspection = _minimal_runtime_inspection()
    inspection[0]["HostConfig"][field] = value
    with pytest.raises(verifier.VerificationError):
        verifier._verify_running_container(inspection, canary=CANARY)


def test_runtime_verifier_rejects_writable_mount_secret_env_and_probe_drift() -> None:
    writable = _minimal_runtime_inspection()
    writable[0]["Mounts"][0]["RW"] = True
    with pytest.raises(verifier.VerificationError, match="mounted authority"):
        verifier._verify_running_container(writable, canary=CANARY)

    disclosed = _minimal_runtime_inspection()
    disclosed[0]["Config"]["Env"].append(f"SIGNALATTICE_REGISTRY_DIGEST_KEY={CANARY}")
    with pytest.raises(verifier.VerificationError):
        verifier._verify_running_container(disclosed, canary=CANARY)

    probe = _minimal_runtime_probe()
    probe["secret_file"]["mode"] = 0o444
    with pytest.raises(verifier.VerificationError, match="probe"):
        verifier.verify_runtime_probe_document(probe, canary=CANARY)


def test_runtime_log_verifier_rejects_marker_without_reflecting_it(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        verifier.subprocess,
        "run",
        lambda *args, **kwargs: subprocess.CompletedProcess(
            args=args,
            returncode=0,
            stdout=f"startup {CANARY}".encode(),
            stderr=b"",
        ),
    )
    with pytest.raises(verifier.VerificationError, match="private material"):
        verifier._bounded_container_logs("a" * 64, canary=CANARY.encode())


def test_workflow_pins_actions_and_runs_with_read_only_authority() -> None:
    workflow_text = _text(".github/workflows/service-security.yml")
    workflow = yaml.safe_load(workflow_text)
    assert workflow["permissions"] == {"contents": "read"}
    job = workflow["jobs"]["service-supply-chain"]
    assert job["runs-on"] == "ubuntu-24.04"
    assert job["permissions"] == {"contents": "read"}
    uses = re.findall(r"^\s*-?\s*uses:\s*([^\s#]+)", workflow_text, flags=re.MULTILINE)
    assert len(uses) == 4
    assert all(re.fullmatch(r"[^@\s]+@[0-9a-f]{40}", value) for value in uses)
    assert re.search(r"moby/buildkit:v[^@\s]+@sha256:[0-9a-f]{64}", workflow_text)
    assert "secrets." not in workflow_text
    assert "id-token:" not in workflow_text
    assert "attestations:" not in workflow_text
    assert "permissions:\n  contents: write" not in workflow_text


def test_workflow_builds_two_clean_service_images_and_no_artifact_is_published() -> None:
    workflow = _text(".github/workflows/service-security.yml")
    assert workflow.count("--target service") == 2
    assert workflow.count("--metadata-file") == 2
    assert workflow.count("--provenance=mode=max") == 2
    assert workflow.count("--no-cache") == 4
    assert workflow.count("signalattice-service:clean-") == 2
    assert "SOURCE_DATE_EPOCH" in workflow
    assert workflow.count('--build-arg "VCS_REF=$GITHUB_SHA"') == 4
    assert "docker login" not in workflow
    assert "docker push" not in workflow
    assert "build-push-action" not in workflow
    assert "cosign" not in workflow.lower()
    assert "signing" not in workflow.lower()
    assert "publish" not in workflow.lower().replace("non-publishing", "")
    verifier_source = _text("scripts/verify_service_container.py")
    assert "service_growth_bytes" not in verifier_source
    assert "service_cli_size_delta_bytes" in verifier_source
    assert "telemetry_source_artifact_bytes" in verifier_source
    assert "application_without_telemetry_source_artifacts_digest" in verifier_source


def test_workflow_scans_repository_image_history_and_final_filesystem_fail_closed() -> None:
    workflow = _text(".github/workflows/service-security.yml")
    assert re.search(r"TRIVY_IMAGE: [^\s]+@sha256:[0-9a-f]{64}", workflow)
    assert re.search(r"SYFT_IMAGE: [^\s]+@sha256:[0-9a-f]{64}", workflow)
    assert "filesystem" in workflow
    assert "--image-config-scanners misconfig,secret" in workflow
    assert workflow.count("--scanners vuln,secret,misconfig,license") == 1
    assert workflow.count("--scanners vuln,secret,misconfig ") == 1
    assert workflow.count("--scanners license ") == 1
    assert workflow.count("--pkg-types library") == 1
    assert "--skip-dirs /usr" in workflow
    assert "--ignore-unfixed" not in workflow
    assert "--license-full" in workflow
    assert workflow.count("--show-suppressed") == 3
    assert workflow.count("--ignorefile /scan/empty.trivyignore") == 3
    assert workflow.count("--offline-scan") == 3
    assert workflow.count("--network none") >= 5
    assert "/var/run/docker.sock" not in workflow
    assert "--input /scan/service-image.tar" in workflow
    assert "docker compose --project-name" in workflow
    assert "compose up --no-build --detach --wait" in workflow
    assert "verify_service_container.py runtime" in workflow
    assert "--output spdx-json" in workflow
    assert "verify_service_container.py container" in workflow
    assert "verify_service_container.py scans" in workflow
    assert "scripts/benchmark_service_operability.py" in workflow
    assert "--env SERVICE_SECRET_CANARY" in workflow
    assert '"data_problem_code_counts"' in workflow
    assert 'problem_codes["service_saturated"] != counts["429"]' in workflow
    assert 'problem_codes["evidence_unavailable"] != counts["503"]' in workflow
    assert 'ready_code != "registry_busy"' in workflow
    assert "ready_retryable is not True" in workflow
    assert '--benchmark-evidence "$EVIDENCE_DIR/service-canary-evidence.json"' in workflow
    assert '--image-license-scan "$EVIDENCE_DIR/trivy-image-license.json"' in workflow
    assert '--runtime-summary "$EVIDENCE_DIR/runtime-summary.json"' in workflow
    assert '--saturation "$EVIDENCE_DIR/runtime-saturation.json"' in workflow
    verifier_source = _text("scripts/verify_service_container.py")
    assert '["docker", "history"' in verifier_source
    assert '["docker", "export"' in verifier_source


def test_workflow_never_uses_runner_authority_inside_the_private_runtime_root() -> None:
    workflow = _text(".github/workflows/service-security.yml")
    runtime_step = workflow[
        workflow.index('runtime_root="') : workflow.index(
            "- name: Export one immutable scanner input"
        )
    ]

    assert 'sudo chown -R 10001:10001 "$runtime_root"' in runtime_step
    assert '> "$runtime_root/' not in runtime_step
    assert '[[ -f "$runtime_root/' not in runtime_step
    assert '[[ ! -f "$runtime_root/' not in runtime_step
    assert 'sudo test -f "$runtime_root/sqlite-lock-ready"' in runtime_step
    assert '> "$EVIDENCE_DIR/lock-helper.log"' in runtime_step


def test_dockerignore_excludes_sensitive_and_generated_build_context() -> None:
    ignored = set(_text(".dockerignore").splitlines())
    for contract in {
        ".git",
        ".github",
        ".env",
        "*.key",
        "*credential*",
        "*secret*",
        "*.sqlite",
        "*.parquet",
        "*.pkl",
        "*.onnx",
        "reports",
        "models",
        "tests",
        "docs",
        "data/registry/**",
        "data/cas/**",
        "data/feature-store/**",
    }:
        assert contract in ignored


def test_trivy_verifier_counts_fixed_and_unfixed_without_ignoring_them() -> None:
    document = _minimal_trivy_result(
        Vulnerabilities=[
            {"Severity": "MEDIUM", "FixedVersion": "2.0"},
            {"Severity": "LOW", "FixedVersion": ""},
        ]
    )
    summary = _verify_trivy(document).counters
    assert summary["fixed_vulnerabilities"] == 1
    assert summary["unfixed_vulnerabilities"] == 1
    assert summary["high"] == summary["critical"] == 0


@pytest.mark.parametrize("fixed_version", ["", "9.9.9"])
@pytest.mark.parametrize("severity", ["HIGH", "CRITICAL"])
def test_trivy_verifier_rejects_fixed_and_unfixed_release_blockers(
    fixed_version: str, severity: str
) -> None:
    document = _minimal_trivy_result(
        Vulnerabilities=[{"Severity": severity, "FixedVersion": fixed_version}]
    )
    with pytest.raises(verifier.VerificationError, match="release-blocking"):
        _verify_trivy(document)


@pytest.mark.parametrize(
    "finding",
    [
        {"Secrets": [{"Severity": "HIGH"}]},
        {"Misconfigurations": [{"Status": "FAIL", "Severity": "HIGH"}]},
        {"Licenses": [{"Name": "SSPL-1.0", "Severity": "LOW"}]},
        {"Exceptions": [{"ID": "temporary-exception"}]},
    ],
)
def test_trivy_verifier_rejects_secret_policy_and_exception_bypasses(
    finding: dict[str, Any],
) -> None:
    with pytest.raises(verifier.VerificationError):
        _verify_trivy(_minimal_trivy_result(**finding))


def test_trivy_verifier_rejects_canary_and_malformed_scanner_output() -> None:
    with pytest.raises(verifier.VerificationError, match="canary"):
        _verify_trivy(_minimal_trivy_result(Target=f"target-{CANARY}"))
    with pytest.raises(verifier.VerificationError, match="schema"):
        _verify_trivy({"SchemaVersion": 1, "Results": []})
    with pytest.raises(verifier.VerificationError, match="scan results"):
        _verify_trivy(
            {
                "SchemaVersion": 2,
                "ArtifactName": "signalattice-test-target",
                "ArtifactType": "filesystem",
                "ArtifactID": "a" * 64,
                "ReportID": "report-identity",
                "Metadata": {"Repo": {"Commit": REVISION}},
                "Results": [],
            }
        )


def test_trivy_verifier_binds_identity_and_records_reviewed_package_license() -> None:
    document = _minimal_trivy_result(
        Licenses=[
            {
                "Name": "MIT",
                "PkgName": "signalattice",
                "Category": "permissive",
                "Severity": "LOW",
            }
        ]
    )
    evidence = _verify_trivy(document)
    assert evidence.artifact_id == "a" * 64
    assert (
        evidence.report_digest
        == hashlib.sha256(
            json.dumps(
                document,
                ensure_ascii=True,
                allow_nan=False,
                separators=(",", ":"),
                sort_keys=True,
            ).encode("ascii")
        ).hexdigest()
    )
    assert evidence.licensed_packages == frozenset({"signalattice"})
    assert evidence.counters["resolved_package_licenses"] == 1


def test_trivy_filesystem_report_requires_null_artifact_id_without_git_claim() -> None:
    document = _minimal_trivy_result()
    document["ArtifactID"] = None
    document["Metadata"] = {}

    evidence = _verify_trivy(
        document,
        expected_revision=None,
        require_artifact_id=False,
    )

    assert evidence.artifact_id is None
    document["ArtifactID"] = "a" * 64
    with pytest.raises(verifier.VerificationError, match="target identity"):
        _verify_trivy(
            document,
            expected_revision=None,
            require_artifact_id=False,
        )


def test_trivy_repository_report_requires_generated_id_and_exact_commit() -> None:
    document = _minimal_trivy_result()
    document["ArtifactType"] = "repository"
    document["Metadata"] = _repository_metadata()

    evidence = _verify_trivy(document, expected_type="repository")

    assert evidence.artifact_id == "a" * 64
    document["ArtifactID"] = None
    with pytest.raises(verifier.VerificationError, match="target identity"):
        _verify_trivy(document, expected_type="repository")


@pytest.mark.parametrize(
    "override",
    [
        {"ExperimentalModifiedFindings": [{"Status": "ignored"}]},
        {
            "Licenses": [
                {
                    "Name": "unclassified",
                    "PkgName": "unknown-package",
                    "Category": "unknown",
                    "Severity": "UNKNOWN",
                }
            ]
        },
    ],
)
def test_trivy_verifier_rejects_suppressed_and_unresolved_license_findings(
    override: dict[str, Any],
) -> None:
    with pytest.raises(verifier.VerificationError, match="release-blocking"):
        _verify_trivy(_minimal_trivy_result(**override))


def test_trivy_verifier_accepts_only_exact_version_bound_python_license_notices() -> None:
    typing_extensions_path = (
        "opt/venv/lib/python3.13/site-packages/"
        "typing_extensions-4.16.0.dist-info/licenses/LICENSE"
    )
    notices = [
        {
            "Name": name,
            "PkgName": "",
            "FilePath": path,
            "Category": "unknown",
            "Severity": "UNKNOWN",
        }
        for path in (typing_extensions_path,)
        for name in ("BSD-0-Clause", "BeOpen", "CNRI-Python-GPL-Compatible")
    ]
    image_document = _minimal_image_trivy_result(Licenses=notices)
    evidence = _verify_trivy(image_document, expected_type="container_image")
    assert evidence.counters["reviewed_container_license_notices"] == 3
    assert evidence.counters["incompatible_licenses"] == 0

    drifted = _minimal_image_trivy_result(Licenses=[dict(notices[0])])
    drifted["Results"][0]["Licenses"][0]["FilePath"] = typing_extensions_path.replace(
        "4.16.0", "4.16.1"
    )
    with pytest.raises(verifier.VerificationError, match="release-blocking"):
        _verify_trivy(drifted, expected_type="container_image")
    with pytest.raises(verifier.VerificationError, match="release-blocking"):
        _verify_trivy(_minimal_trivy_result(Licenses=[notices[0]]))

    duplicated = _minimal_image_trivy_result(Licenses=[dict(notices[0]), dict(notices[0])])
    with pytest.raises(verifier.VerificationError, match="duplicated"):
        _verify_trivy(duplicated, expected_type="container_image")


def test_trivy_verifier_rejects_target_revision_and_config_identity_drift() -> None:
    wrong_target = _minimal_trivy_result()
    wrong_target["ArtifactName"] = "other"
    with pytest.raises(verifier.VerificationError, match="target identity"):
        _verify_trivy(wrong_target)

    missing_revision = _minimal_trivy_result()
    missing_revision["Metadata"] = {}
    with pytest.raises(verifier.VerificationError, match="source revision"):
        _verify_trivy(missing_revision)

    image = _minimal_image_trivy_result()
    _verify_trivy(
        image,
        expected_type="container_image",
        expected_config_digest=CONFIG_DIGEST,
    )
    with pytest.raises(verifier.VerificationError, match="artifact-bound"):
        _verify_trivy(
            image,
            expected_type="container_image",
            expected_config_digest="sha256:" + ("f" * 64),
        )


def test_repository_source_binding_requires_exact_clean_head(tmp_path: Path) -> None:
    repository = tmp_path / "repository"
    repository.mkdir()
    subprocess.run(["git", "init", "--quiet", repository], check=True)
    tracked = repository / "tracked.txt"
    tracked.write_text("reviewed\n", encoding="utf-8")
    subprocess.run(["git", "-C", repository, "add", "tracked.txt"], check=True)
    subprocess.run(
        [
            "git",
            "-C",
            repository,
            "-c",
            "user.name=Signalattice Test",
            "-c",
            "user.email=signalattice-test@example.invalid",
            "commit",
            "--quiet",
            "-m",
            "Create test source",
        ],
        check=True,
    )
    head = subprocess.run(
        ["git", "-C", repository, "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()

    assert (
        verifier._verify_repository_source(
            repository,
            expected_revision=head,
        )
        == hashlib.sha256((head + "\n").encode("ascii")).hexdigest()
    )
    with pytest.raises(verifier.VerificationError, match="requested source revision"):
        verifier._verify_repository_source(repository, expected_revision="f" * 40)

    (repository / "untracked.txt").write_text("unreviewed\n", encoding="utf-8")
    with pytest.raises(verifier.VerificationError, match="clean checkout"):
        verifier._verify_repository_source(repository, expected_revision=head)


def test_scan_command_binds_repository_image_license_runtime_and_benchmark_evidence(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def evidence_file(name: str, document: dict[str, Any]) -> Path:
        path = tmp_path / name
        path.write_text(json.dumps(document), encoding="utf-8")
        return path

    repository = _minimal_trivy_result()
    repository.update(
        {
            "ArtifactName": "/workspace",
            "ArtifactType": "repository",
            "Metadata": _repository_metadata(),
        }
    )
    image_security = _minimal_image_trivy_result()
    license_path = (
        "opt/venv/lib/python3.13/site-packages/"
        "typing_extensions-4.16.0.dist-info/licenses/LICENSE"
    )
    image_licenses = _minimal_image_trivy_result(
        Licenses=[
            {
                "Name": name,
                "PkgName": "",
                "FilePath": license_path,
                "Category": "unknown",
                "Severity": "UNKNOWN",
            }
            for name in ("BSD-0-Clause", "BeOpen", "CNRI-Python-GPL-Compatible")
        ]
    )
    container_summary = {
        "schema_version": 1,
        "result": "passed",
        "source_revision": REVISION,
        "image_name": IMAGE_NAME,
        "oci_config_digest": CONFIG_DIGEST,
        "oci_manifest_digest": MANIFEST_DIGEST,
    }
    runtime_summary = {
        "schema_version": 1,
        "result": "passed",
        "profile": "compose-default-service",
        "credentials_present": False,
        "saturation": {"result": "passed"},
    }
    summary = tmp_path / "summary.json"
    args = Namespace(
        sbom=evidence_file("sbom.json", _minimal_spdx()),
        repository_scan=evidence_file("repository.json", repository),
        repository_root=tmp_path,
        image_scan=evidence_file("image-security.json", image_security),
        image_license_scan=evidence_file("image-licenses.json", image_licenses),
        container_summary=evidence_file("container-summary.json", container_summary),
        runtime_summary=evidence_file("runtime-summary.json", runtime_summary),
        benchmark_evidence=evidence_file("benchmark.json", _minimal_benchmark_evidence()),
        image_name=IMAGE_NAME,
        source_revision=REVISION,
        canary_env="SERVICE_SECRET_CANARY",
        summary=summary,
    )
    monkeypatch.setenv("SERVICE_SECRET_CANARY", CANARY)
    monkeypatch.setattr(
        verifier,
        "_verify_repository_source",
        lambda _root, *, expected_revision: hashlib.sha256(
            (expected_revision + "\n").encode("ascii")
        ).hexdigest(),
    )

    verifier._scans_command(args)

    sanitized = json.loads(summary.read_text(encoding="utf-8"))
    assert sanitized["result"] == "passed"
    assert sanitized["repository_artifact_id"] == "a" * 64
    assert sanitized["image_artifact_id"] == "a" * 64
    assert sanitized["image_security"]["license_findings"] == 0
    assert sanitized["image_licenses"]["reviewed_container_license_notices"] == 3
    assert sanitized["sbom_unresolved_licenses"] == 0

    image_licenses["ArtifactID"] = "b" * 64
    args.image_license_scan = evidence_file("image-licenses-drifted.json", image_licenses)
    with pytest.raises(verifier.VerificationError, match="different artifacts"):
        verifier._scans_command(args)


def test_benchmark_canary_verifier_accepts_integrity_bound_malformed_fault() -> None:
    document = _minimal_benchmark_evidence()
    digest = verifier.verify_benchmark_canary_evidence(document, canary=CANARY)
    assert digest == document["integrity"]["canonical_payload_sha256"]


def test_benchmark_verifier_workload_identity_matches_the_measurement_harness() -> None:
    benchmark = _load_benchmark()

    assert verifier._expected_isolated_rss_workload() == benchmark._isolated_rss_workload_contract()


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (("saturation", {"probe_status": 503, "rejection_count": 24}), "contract"),
        (("fault_cases", []), "malformed-input"),
        (("evidence_id", CANARY), "canary"),
    ],
)
def test_benchmark_canary_verifier_rejects_substitution_or_disclosure(
    mutation: tuple[str, Any], message: str
) -> None:
    document = _minimal_benchmark_evidence()
    field, value = mutation
    document[field] = value
    with pytest.raises(verifier.VerificationError, match=message):
        verifier.verify_benchmark_canary_evidence(document, canary=CANARY)


def test_benchmark_canary_verifier_rejects_integrity_drift() -> None:
    document = _minimal_benchmark_evidence()
    document["provenance"]["network_requests"] = 1
    with pytest.raises(verifier.VerificationError):
        verifier.verify_benchmark_canary_evidence(document, canary=CANARY)

    document = _minimal_benchmark_evidence()
    document["integrity"]["canonical_payload_sha256"] = "f" * 64
    with pytest.raises(verifier.VerificationError, match="integrity verification"):
        verifier.verify_benchmark_canary_evidence(document, canary=CANARY)


def test_spdx_verifier_accepts_exact_nonempty_23_contract() -> None:
    summary = verifier.verify_spdx_document(
        _minimal_spdx(),
        canary=CANARY,
        expected_name=SBOM_NAME,
        expected_revision=REVISION,
    )
    assert summary.packages == 2
    assert summary.license_expressions == 2
    assert summary.unresolved_license_packages == frozenset()


@pytest.mark.parametrize("unresolved", ["", "NONE", "NOASSERTION", "UNKNOWN", "LicenseRef-opaque"])
def test_spdx_verifier_surfaces_unresolved_dependency_license_for_cross_scan(
    unresolved: str,
) -> None:
    document = _minimal_spdx()
    document["packages"][0]["licenseConcluded"] = unresolved
    document["packages"][0]["licenseDeclared"] = unresolved
    evidence = verifier.verify_spdx_document(
        document,
        canary=CANARY,
        expected_name=SBOM_NAME,
        expected_revision=REVISION,
    )
    assert evidence.unresolved_license_packages == frozenset({"signalattice"})


def test_spdx_verifier_rejects_root_revision_or_document_identity_drift() -> None:
    wrong_revision = _minimal_spdx()
    wrong_revision["packages"][1]["versionInfo"] = "f" * 40
    with pytest.raises(verifier.VerificationError, match="source revision"):
        verifier.verify_spdx_document(
            wrong_revision,
            canary=CANARY,
            expected_name=SBOM_NAME,
            expected_revision=REVISION,
        )

    wrong_name = _minimal_spdx(name="other")
    with pytest.raises(verifier.VerificationError):
        verifier.verify_spdx_document(
            wrong_name,
            canary=CANARY,
            expected_name=SBOM_NAME,
            expected_revision=REVISION,
        )


@pytest.mark.parametrize(
    "document",
    [
        _minimal_spdx(spdxVersion="SPDX-2.2"),
        _minimal_spdx(packages=[]),
        _minimal_spdx(packages=[{"licenseDeclared": "AGPL-3.0-only"}]),
        _minimal_spdx(name=CANARY),
    ],
)
def test_spdx_verifier_rejects_wrong_empty_incompatible_or_disclosing_documents(
    document: dict[str, Any],
) -> None:
    with pytest.raises(verifier.VerificationError):
        verifier.verify_spdx_document(
            document,
            canary=CANARY,
            expected_name=SBOM_NAME,
            expected_revision=REVISION,
        )


def test_provenance_verifier_requires_materials_and_immutable_config() -> None:
    document: dict[str, Any] = {
        "containerimage.config.digest": CONFIG_DIGEST,
        "containerimage.digest": MANIFEST_DIGEST,
        "buildx.build.provenance": {
            "buildType": "https://mobyproject.org/buildkit@v1",
            "invocation": {"parameters": {"VCS_REF": REVISION}},
            "materials": [{"uri": "pkg:docker/python", "digest": {"sha256": "a" * 64}}],
        },
    }
    identity = verifier.verify_provenance_document(
        document,
        canary=CANARY,
        expected_revision=REVISION,
    )
    assert identity.config_digest == CONFIG_DIGEST
    assert identity.manifest_digest == MANIFEST_DIGEST
    with pytest.raises(verifier.VerificationError, match="materials"):
        document["buildx.build.provenance"]["materials"] = []
        verifier.verify_provenance_document(
            document,
            canary=CANARY,
            expected_revision=REVISION,
        )
    document["buildx.build.provenance"]["materials"] = [
        {"uri": "pkg:docker/python", "digest": {"sha256": "a" * 64}}
    ]
    document["buildx.build.provenance"]["invocation"] = {}
    with pytest.raises(verifier.VerificationError, match="source revision"):
        verifier.verify_provenance_document(
            document,
            canary=CANARY,
            expected_revision=REVISION,
        )


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (("Architecture", "arm64"), "platform"),
        (("Size", 0), "size"),
        (("Config.User", "0:0"), "UID/GID"),
        (("Config.ExposedPorts", {"8765/tcp": {}}), "expose"),
        (("Config.Volumes", {"/data": {}}), "writable volumes"),
        (("Config.Env", ["HOME=/root"]), "home"),
        (("Config.Healthcheck", None), "health check"),
        (("Config.Labels", {"org.opencontainers.image.licenses": "MIT"}), "source-identity"),
    ],
)
def test_image_config_verifier_rejects_privilege_and_authority_drift(
    mutation: tuple[str, Any], message: str
) -> None:
    inspection = _minimal_inspection()
    path, value = mutation
    if "." in path:
        parent, key = path.split(".", 1)
        inspection[parent][key] = value
    else:
        inspection[path] = value
    with pytest.raises(verifier.VerificationError, match=message):
        verifier._normalized_config(inspection, expected_revision=REVISION)


def test_image_config_digest_is_deterministic_for_environment_order() -> None:
    first, first_layers, first_config = verifier._normalized_config(
        _minimal_inspection(), expected_revision=REVISION
    )
    second_inspection = _minimal_inspection()
    second_inspection["Config"]["Env"].reverse()
    second, second_layers, second_config = verifier._normalized_config(
        second_inspection, expected_revision=REVISION
    )
    assert first == second
    assert first_config == second_config == CONFIG_DIGEST
    assert first_layers == second_layers == 1


def test_bounded_json_loader_rejects_symlink_and_invalid_json(tmp_path: Path) -> None:
    target = tmp_path / "target.json"
    target.write_text("{}", encoding="utf-8")
    link = tmp_path / "link.json"
    link.symlink_to(target)
    with pytest.raises(verifier.VerificationError, match="regular file"):
        verifier._load_json(link)

    invalid = tmp_path / "invalid.json"
    invalid.write_text("{", encoding="utf-8")
    with pytest.raises(verifier.VerificationError, match="valid JSON"):
        verifier._load_json(invalid)

    nonfinite = tmp_path / "nonfinite.json"
    nonfinite.write_text('{"value": NaN}', encoding="utf-8")
    with pytest.raises(verifier.VerificationError, match="valid JSON"):
        verifier._load_json(nonfinite)


@pytest.mark.parametrize(
    "path",
    [
        "../secret",
        "/absolute",
        "nested/../../escape",
        "./etc/passwd",
        "etc//passwd",
        "etc/passwd/",
        "",
    ],
)
def test_container_export_path_validation_rejects_noncanonical_names(path: str) -> None:
    with pytest.raises(verifier.VerificationError):
        verifier._canonical_tar_path(path)


def test_sanitized_summary_is_atomic_bounded_and_contains_no_canary(tmp_path: Path) -> None:
    destination = tmp_path / "evidence" / "summary.json"
    verifier._atomic_json(destination, {"schema_version": 1, "result": "passed"})
    assert json.loads(destination.read_text(encoding="utf-8")) == {
        "result": "passed",
        "schema_version": 1,
    }
    assert destination.stat().st_mode & 0o777 == 0o600
    assert CANARY not in destination.read_text(encoding="utf-8")


def test_cli_boundary_redacts_unexpected_internal_diagnostics(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    class Args:
        @staticmethod
        def handler(_args: object) -> None:
            raise RuntimeError(f"private diagnostic {CANARY}")

    class Parser:
        @staticmethod
        def parse_args() -> Args:
            return Args()

    monkeypatch.setattr(verifier, "_parser", lambda: Parser())

    assert verifier.main() == 1
    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err == ("service-container verification failed: unexpected internal failure\n")
    assert CANARY not in captured.err


def test_flattened_filesystem_verifier_accepts_only_the_declared_runtime_inventory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    archive = tmp_path / "rootfs.tar"
    _write_rootfs_tar(archive)
    _install_fake_export(monkeypatch, archive)

    snapshot = verifier._scan_exported_filesystem(
        "signalattice-service:test", canary=CANARY.encode()
    )

    assert re.fullmatch(r"[0-9a-f]{64}", snapshot.application_digest)
    assert re.fullmatch(r"[0-9a-f]{64}", snapshot.filesystem_digest)
    assert re.fullmatch(
        r"[0-9a-f]{64}",
        snapshot.application_without_telemetry_source_artifacts_digest,
    )
    assert re.fullmatch(r"[0-9a-f]{64}", snapshot.telemetry_source_artifact_digest)
    assert snapshot.application_regular_bytes > snapshot.telemetry_source_artifact_bytes > 0
    assert snapshot.application_without_telemetry_source_artifacts_regular_bytes == (
        snapshot.application_regular_bytes - snapshot.telemetry_source_artifact_bytes
    )
    assert snapshot.regular_file_count == 3 + len(verifier._EXPECTED_SERVICE_SOURCE_FILES)


def test_flattened_filesystem_verifier_requires_exact_locked_distribution_set(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    missing = tmp_path / "missing-distribution.tar"
    omitted = min(verifier._EXPECTED_SERVICE_DISTRIBUTIONS)
    _write_rootfs_tar(missing, omit_distribution=omitted)
    _install_fake_export(monkeypatch, missing)
    with pytest.raises(verifier.VerificationError, match="inventory is incomplete"):
        verifier._scan_exported_filesystem(
            "signalattice-service:test",
            canary=CANARY.encode(),
        )

    undeclared = tmp_path / "undeclared-distribution.tar"
    _write_rootfs_tar(
        undeclared,
        extra_path=("opt/venv/lib/python3.13/site-packages/" "numpy-9.9.9.dist-info/METADATA"),
        extra_payload=b"Name: numpy\nVersion: 9.9.9\n",
    )
    _install_fake_export(monkeypatch, undeclared)
    with pytest.raises(verifier.VerificationError, match="undeclared package root"):
        verifier._scan_exported_filesystem(
            "signalattice-service:test",
            canary=CANARY.encode(),
        )


@pytest.mark.parametrize(
    "value",
    [
        -1,
        0,
        verifier.MAX_TELEMETRY_SOURCE_ARTIFACT_BYTES + 1,
    ],
)
def test_telemetry_source_artifact_bytes_reject_negative_absent_and_over_budget(
    value: int,
) -> None:
    with pytest.raises(verifier.VerificationError, match="100 MiB"):
        verifier._verify_telemetry_source_artifact_bytes(value)


@pytest.mark.parametrize(
    ("extra_path", "extra_payload", "message"),
    [
        ("app/unexpected.txt", b"unexpected", "unexpected application"),
        (
            "opt/venv/lib/python3.13/site-packages/fastapi/private.txt",
            CANARY.encode(),
            "private material",
        ),
        (
            "opt/venv/lib/python3.13/site-packages/fastapi/model.onnx",
            b"model",
            "data or model",
        ),
        (
            "opt/venv/lib/python3.13/site-packages/fastapi/fixtures/licensed_sample.json",
            b"{}",
            "test, fixture, sample, or dataset",
        ),
        (
            "opt/venv/lib/python3.13/site-packages/fastapi/model.safetensors",
            b"model",
            "data or model",
        ),
        (
            "opt/venv/lib/python3.13/site-packages/fastapi/local.duckdb",
            b"database",
            "data or model",
        ),
        (
            "app/src/quant_platform/service/unreviewed.py",
            b'"""unreviewed"""\n',
            "exact read-only allowlist",
        ),
        (
            "usr/local/lib/python3.13/ensurepip/__init__.py",
            b'"""installer"""\n',
            "package installer",
        ),
        ("etc/hosts", CANARY.encode(), "private material"),
    ],
)
def test_flattened_filesystem_verifier_rejects_unexpected_canary_and_model_artifacts(
    extra_path: str,
    extra_payload: bytes,
    message: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    archive = tmp_path / "rootfs.tar"
    _write_rootfs_tar(archive, extra_path=extra_path, extra_payload=extra_payload)
    _install_fake_export(monkeypatch, archive)

    with pytest.raises(verifier.VerificationError, match=message):
        verifier._scan_exported_filesystem("signalattice-service:test", canary=CANARY.encode())


@pytest.mark.parametrize("path", sorted(verifier._DOCKER_RUNTIME_INJECTED_PATHS))
def test_filesystem_digest_normalizes_only_docker_runtime_injected_files(
    path: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    first_archive = tmp_path / "first.tar"
    second_archive = tmp_path / "second.tar"
    _write_rootfs_tar(first_archive, extra_path=path, extra_payload=b"first-daemon-value")
    _write_rootfs_tar(second_archive, extra_path=path, extra_payload=b"second-daemon-value-longer")

    _install_fake_export(monkeypatch, first_archive)
    first = verifier._scan_exported_filesystem("signalattice-service:test", canary=CANARY.encode())
    _install_fake_export(monkeypatch, second_archive)
    second = verifier._scan_exported_filesystem("signalattice-service:test", canary=CANARY.encode())

    assert first.application_digest == second.application_digest
    assert first.filesystem_digest == second.filesystem_digest
    assert first.telemetry_source_artifact_digest == second.telemetry_source_artifact_digest
    assert (
        first.regular_file_count
        == second.regular_file_count
        == (4 + len(verifier._EXPECTED_SERVICE_SOURCE_FILES))
    )


def test_filesystem_verifier_rejects_duplicate_paths_and_unsafe_links(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    duplicate = tmp_path / "duplicate.tar"
    _write_rootfs_tar(duplicate, extra_path="etc/passwd", extra_payload=b"shadow")
    _install_fake_export(monkeypatch, duplicate)
    with pytest.raises(verifier.VerificationError, match="duplicate"):
        verifier._scan_exported_filesystem("signalattice-service:test", canary=CANARY.encode())

    escaping = tmp_path / "escaping.tar"
    _write_rootfs_tar(escaping, link=("opt/escape", "../../../host", False))
    _install_fake_export(monkeypatch, escaping)
    with pytest.raises(verifier.VerificationError, match="escapes"):
        verifier._scan_exported_filesystem("signalattice-service:test", canary=CANARY.encode())

    missing = tmp_path / "missing-hard-link.tar"
    _write_rootfs_tar(missing, link=("opt/missing", "does/not/exist", True))
    _install_fake_export(monkeypatch, missing)
    with pytest.raises(verifier.VerificationError, match="target is missing"):
        verifier._scan_exported_filesystem("signalattice-service:test", canary=CANARY.encode())


def test_link_target_resolution_accepts_only_container_root_confined_targets() -> None:
    assert (
        verifier._resolved_link_target("etc/mtab", "../proc/mounts", hard_link=False)
        == "proc/mounts"
    )
    assert (
        verifier._resolved_link_target("etc/mtab", "/proc/mounts", hard_link=False) == "proc/mounts"
    )
    with pytest.raises(verifier.VerificationError, match="escapes"):
        verifier._resolved_link_target("etc/link", "../../host", hard_link=False)
