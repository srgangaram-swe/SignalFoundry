"""Security and failure-boundary tests for local service assembly."""

from __future__ import annotations

import asyncio
import json
import os
import socket
import stat
import subprocess
import tempfile
import threading
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest
from typer.testing import CliRunner

from quant_platform.cli import app
from quant_platform.service import server
from quant_platform.service.admission import AdmissionController, AdmissionLimits
from quant_platform.service.http_protocol import SignalatticeH11Protocol
from quant_platform.service.metrics import ServiceMetrics
from quant_platform.service.telemetry import ServiceTelemetry
from quant_platform.tracking.contracts import Page


class _NoReadPorts:
    """Minimal read-port shape; liveness must never invoke these methods."""

    @staticmethod
    def _unexpected(*_args: object, **_kwargs: object) -> Any:
        raise AssertionError("storage was accessed by a storage-independent operation")

    probe_evidence_readiness = _unexpected
    get_run = _unexpected
    list_runs = _unexpected
    get_artifact = _unexpected
    list_run_artifacts = _unexpected
    read_verified_manifest = _unexpected


class _BlockingReadPorts(_NoReadPorts):
    """Hold a fixed number of real routed reads without wall-clock polling."""

    def __init__(self, target: int) -> None:
        self._target = target
        self._entered = 0
        self._lock = threading.Lock()
        self._one_before_full = threading.Event()
        self._all_entered = threading.Event()
        self._release = threading.Event()

    def list_runs(self, *_args: object, **_kwargs: object) -> Page[Any, Any]:
        with self._lock:
            self._entered += 1
            if self._entered == self._target - 1:
                self._one_before_full.set()
            if self._entered == self._target:
                self._all_entered.set()
        if not self._release.wait(timeout=5):
            raise RuntimeError("bounded server-test barrier expired")
        return Page((), None)

    def wait_until_full(self, timeout: float) -> bool:
        return self._all_entered.wait(timeout)

    def wait_until_one_before_full(self, timeout: float) -> bool:
        if self._target <= 1:
            raise ValueError("one-before-full barrier requires a target greater than one")
        return self._one_before_full.wait(timeout)

    def release(self) -> None:
        self._release.set()


@pytest.mark.parametrize(
    "overrides",
    [
        {"host": "0.0.0.0"},
        {"host": "::1"},
        {"port": True},
        {"port": 1_023},
        {"transport_concurrency": 0},
        {"data_concurrency": 33, "transport_concurrency": 32},
        {"backlog": 1_025},
        {"keep_alive_seconds": 0},
        {"graceful_shutdown_seconds": 61},
        {"incomplete_event_bytes": 4_095},
    ],
)
def test_server_config_rejects_non_loopback_and_unbounded_profiles(
    overrides: dict[str, object],
) -> None:
    with pytest.raises(server.ServiceStartupError):
        server.ServerConfig(**overrides)  # type: ignore[arg-type]


def test_uvicorn_profile_disables_remote_authority_and_unbounded_features() -> None:
    config = server.build_uvicorn_config(_NoReadPorts(), server.ServerConfig())

    assert config.host == "127.0.0.1"
    assert config.port == 8_765
    assert isinstance(config.http, type)
    assert issubclass(config.http, SignalatticeH11Protocol)
    assert config.ws == "none"
    assert config.lifespan == "on"
    assert config.loop == "asyncio"
    assert config.interface == "asgi3"
    assert config.workers == 1
    assert config.reload is False
    assert config.proxy_headers is False
    assert config.forwarded_allow_ips == ""
    assert config.access_log is False
    assert config.log_level == "critical"
    assert config.server_header is False
    assert config.date_header is False
    assert config.limit_concurrency == 32
    assert config.backlog == 64
    assert config.timeout_keep_alive == 3
    assert config.timeout_graceful_shutdown == 10
    assert config.h11_max_incomplete_event_size == 16 * 1_024
    assert config.reset_contextvars is True
    assert config.uds is None


def test_uvicorn_profile_defers_private_unix_socket_to_prebound_descriptor() -> None:
    socket_path = Path("/run/signalattice/api.sock")
    config = server.build_uvicorn_config(
        _NoReadPorts(),
        server.ServerConfig(socket_path=socket_path),
    )

    assert config.uds is None
    assert config.access_log is False
    assert config.proxy_headers is False


@pytest.mark.parametrize("relative", [Path("api.sock"), Path("../api.sock")])
def test_server_config_rejects_relative_socket_paths(relative: Path) -> None:
    with pytest.raises(server.ServiceStartupError, match="absolute"):
        server.ServerConfig(socket_path=relative)


def test_server_config_rejects_platform_unsafe_unix_socket_path(tmp_path: Path) -> None:
    oversized = (tmp_path / ("s" * 101)).absolute()

    with pytest.raises(server.ServiceStartupError, match="bounded Unix-socket"):
        server.ServerConfig(socket_path=oversized)


def test_keychain_lookup_uses_argument_vector_and_keeps_secret_in_memory(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    observed: dict[str, object] = {}

    def completed(
        args: list[str],
        **kwargs: object,
    ) -> subprocess.CompletedProcess[bytes]:
        observed["args"] = args
        observed["kwargs"] = kwargs
        return subprocess.CompletedProcess(args, 0, stdout=(b"s" * 32) + b"\n", stderr=b"")

    monkeypatch.setattr(server.getpass, "getuser", lambda: "srgangaram-swe")
    monkeypatch.setattr(server.subprocess, "run", completed)

    secret = server.read_digest_secret_from_keychain(service="com.signal-foundry.test")

    assert secret == b"s" * 32
    assert observed["args"] == [
        "security",
        "find-generic-password",
        "-a",
        "srgangaram-swe",
        "-s",
        "com.signal-foundry.test",
        "-w",
    ]
    assert observed["kwargs"] == {
        "check": False,
        "capture_output": True,
        "timeout": 5,
    }


@pytest.mark.parametrize(
    ("stdout", "returncode"),
    [
        (b"short", 0),
        ((b"s" * 32) + b"\nembedded", 0),
        (b"s" * 32, 1),
        (b"s" * 257, 0),
    ],
)
def test_keychain_contract_fails_closed_without_reflecting_output(
    monkeypatch: pytest.MonkeyPatch,
    stdout: bytes,
    returncode: int,
) -> None:
    def completed(
        args: list[str],
        **_kwargs: object,
    ) -> subprocess.CompletedProcess[bytes]:
        return subprocess.CompletedProcess(
            args,
            returncode,
            stdout=stdout,
            stderr=b"private-keychain-diagnostic",
        )

    monkeypatch.setattr(server.subprocess, "run", completed)

    with pytest.raises(server.ServiceStartupError) as raised:
        server.read_digest_secret_from_keychain(account="owner")

    rendered = str(raised.value)
    assert "private-keychain-diagnostic" not in rendered
    assert stdout.decode("ascii", errors="ignore") not in rendered


def test_keychain_process_failure_is_redacted(monkeypatch: pytest.MonkeyPatch) -> None:
    def timeout(*_args: object, **_kwargs: object) -> subprocess.CompletedProcess[bytes]:
        raise subprocess.TimeoutExpired("security private-argument", timeout=5)

    monkeypatch.setattr(server.subprocess, "run", timeout)

    with pytest.raises(server.ServiceStartupError) as raised:
        server.read_digest_secret_from_keychain(account="owner")

    assert str(raised.value) == "unable to read the registry secret from macOS Keychain"


def test_keychain_public_identifiers_are_validated_before_process_launch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def unexpected(*_args: object, **_kwargs: object) -> subprocess.CompletedProcess[bytes]:
        raise AssertionError("subprocess must not run for invalid public metadata")

    monkeypatch.setattr(server.subprocess, "run", unexpected)

    with pytest.raises(server.ServiceStartupError, match="bounded public identifier"):
        server.read_digest_secret_from_keychain(service="invalid service", account="owner")
    with pytest.raises(server.ServiceStartupError, match="bounded public identifier"):
        server.read_digest_secret_from_keychain(service="valid", account="invalid account")


def test_private_digest_file_is_read_without_following_or_reflecting(tmp_path: Path) -> None:
    path = (tmp_path / "digest-key").resolve()
    path.write_bytes((b"k" * 32) + b"\n")
    path.chmod(0o400)

    assert server.read_digest_secret_from_file(path) == b"k" * 32


def test_digest_file_rejects_symlinks_permissions_size_and_embedded_lines(
    tmp_path: Path,
) -> None:
    private = (tmp_path / "private-key").resolve()
    private.write_bytes(b"s" * 32)
    private.chmod(0o400)
    linked = (tmp_path / "linked-key").resolve()
    linked.symlink_to(private)
    with pytest.raises(server.ServiceStartupError):
        server.read_digest_secret_from_file(linked)

    private.chmod(0o404)
    with pytest.raises(server.ServiceStartupError, match="private-file"):
        server.read_digest_secret_from_file(private)

    private.chmod(0o600)
    private.write_bytes(b"s" * 257)
    private.chmod(0o400)
    with pytest.raises(server.ServiceStartupError, match="32-to-256"):
        server.read_digest_secret_from_file(private)


def test_cli_preserves_secret_and_socket_path_identity_for_server_validation(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    registry = tmp_path / "registry.sqlite3"
    registry.write_bytes(b"existing")
    cas = tmp_path / "cas"
    cas.mkdir()
    private = tmp_path / "private-key"
    private.write_bytes(b"k" * 32)
    private.chmod(0o400)
    linked_key = tmp_path / "linked-key"
    linked_key.symlink_to(private)
    observed: dict[str, object] = {}

    def capture(
        registry_path: Path,
        cas_path: Path,
        **kwargs: object,
    ) -> None:
        observed.update(
            registry=registry_path,
            cas=cas_path,
            digest_key_file=kwargs["digest_key_file"],
            config=kwargs["config"],
        )

    monkeypatch.setattr(server, "serve_existing_evidence", capture)
    with tempfile.TemporaryDirectory(prefix="sf-cli-", dir="/tmp") as directory:
        short_root = Path(directory)
        runtime = short_root / "runtime"
        runtime.mkdir(mode=0o700)
        linked_runtime = short_root / "linked-runtime"
        linked_runtime.symlink_to(runtime, target_is_directory=True)
        result = CliRunner().invoke(
            app,
            [
                "serve-api",
                "--registry-db",
                str(registry),
                "--cas-root",
                str(cas),
                "--socket-path",
                str(linked_runtime / "api.sock"),
                "--digest-key-file",
                str(linked_key),
            ],
        )

        assert result.exit_code == 0, result.output
        assert observed["digest_key_file"] == linked_key
        config = observed["config"]
        assert isinstance(config, server.ServerConfig)
        assert config.socket_path == linked_runtime / "api.sock"

    private.chmod(0o600)
    private.write_bytes((b"s" * 32) + b"\nprivate")
    private.chmod(0o400)
    with pytest.raises(server.ServiceStartupError, match="32-to-256"):
        server.read_digest_secret_from_file(private)


def test_socket_runtime_requires_owned_private_directory_and_absent_target(
    tmp_path: Path,
) -> None:
    runtime = (tmp_path / "runtime").resolve()
    runtime.mkdir(mode=0o700)
    socket_path = runtime / "api.sock"

    server.validate_socket_runtime(socket_path)
    runtime.chmod(0o750)
    with pytest.raises(server.ServiceStartupError, match="private-directory"):
        server.validate_socket_runtime(socket_path)
    runtime.chmod(0o700)
    socket_path.write_bytes(b"stale")
    with pytest.raises(server.ServiceStartupError, match="already exists"):
        server.validate_socket_runtime(socket_path)


def test_socket_runtime_rejects_symlinked_parent(tmp_path: Path) -> None:
    target = (tmp_path / "target").resolve()
    target.mkdir(mode=0o700)
    linked = (tmp_path / "linked").resolve()
    linked.symlink_to(target, target_is_directory=True)

    with pytest.raises(server.ServiceStartupError, match="private-directory"):
        server.validate_socket_runtime(linked / "api.sock")


async def _unix_socket_scenario(tmp_path: Path) -> tuple[bytes, os.stat_result]:
    runtime = (tmp_path / "private-runtime").resolve()
    runtime.mkdir(mode=0o700)
    socket_path = runtime / "api.sock"
    binding = server._bind_private_unix_socket(socket_path, backlog=8)
    metadata = socket_path.lstat()
    config = server.build_uvicorn_config(
        _NoReadPorts(),
        server.ServerConfig(socket_path=socket_path, backlog=8),
    )
    instance = server.uvicorn.Server(config)
    task = asyncio.create_task(instance.serve(sockets=[binding.listener]))
    writer: asyncio.StreamWriter | None = None
    try:
        await asyncio.wait_for(_wait_until(lambda: instance.started or task.done()), timeout=5)
        if task.done():
            await task
            raise AssertionError("Uvicorn exited before accepting the Unix-socket request")
        reader, writer = await asyncio.wait_for(
            asyncio.open_unix_connection(socket_path),
            timeout=5,
        )
        writer.write(
            b"GET /health/live HTTP/1.1\r\n" b"Host: localhost\r\n" b"Connection: close\r\n\r\n"
        )
        await asyncio.wait_for(writer.drain(), timeout=5)
        chunks: list[bytes] = []
        response_bytes = 0
        while chunk := await asyncio.wait_for(reader.read(4 * 1_024), timeout=5):
            response_bytes += len(chunk)
            if response_bytes > 32 * 1_024:
                raise AssertionError("Unix-socket response exceeded its byte ceiling")
            chunks.append(chunk)
        return b"".join(chunks), metadata
    finally:
        if writer is not None:
            writer.close()
            await writer.wait_closed()
        instance.should_exit = True
        await asyncio.wait_for(task, timeout=5)
        server._close_private_unix_socket(binding)
        assert not socket_path.exists()


def test_prebound_unix_socket_is_private_live_and_removed_exactly() -> None:
    with tempfile.TemporaryDirectory(prefix="sf-uds-", dir="/tmp") as directory:
        response, metadata = asyncio.run(_unix_socket_scenario(Path(directory)))

    headers, separator, body = response.partition(b"\r\n\r\n")
    assert separator
    assert headers.startswith(b"HTTP/1.1 200 OK\r\n")
    assert body == b'{"schema_version":1,"status":"live"}'
    assert stat.S_ISSOCK(metadata.st_mode)
    assert stat.S_IMODE(metadata.st_mode) == 0o600
    assert metadata.st_uid == os.geteuid()


def test_unix_socket_cleanup_refuses_replaced_identity() -> None:
    with tempfile.TemporaryDirectory(prefix="sf-uds-", dir="/tmp") as directory:
        runtime = (Path(directory) / "private-runtime").resolve()
        runtime.mkdir(mode=0o700)
        socket_path = runtime / "api.sock"
        binding = server._bind_private_unix_socket(socket_path, backlog=8)
        binding.listener.close()
        socket_path.unlink()
        socket_path.write_bytes(b"replacement")

        with pytest.raises(server.ServiceStartupError, match="identity changed"):
            server._close_private_unix_socket(binding)

        assert socket_path.read_bytes() == b"replacement"


def test_private_bind_preserves_every_preexisting_path_identity() -> None:
    with tempfile.TemporaryDirectory(prefix="sf-uds-", dir="/tmp") as directory:
        runtime = (Path(directory) / "private-runtime").resolve()
        runtime.mkdir(mode=0o700)
        socket_path = runtime / "api.sock"

        socket_path.write_bytes(b"stale-file")
        regular = socket_path.lstat()
        with pytest.raises(server.ServiceStartupError, match="already exists"):
            server._bind_private_unix_socket(socket_path, backlog=8)
        assert socket_path.read_bytes() == b"stale-file"
        assert (socket_path.lstat().st_dev, socket_path.lstat().st_ino) == (
            regular.st_dev,
            regular.st_ino,
        )
        socket_path.unlink()

        target = runtime / "target"
        target.write_bytes(b"target")
        socket_path.symlink_to(target)
        symbolic = socket_path.lstat()
        with pytest.raises(server.ServiceStartupError, match="already exists"):
            server._bind_private_unix_socket(socket_path, backlog=8)
        assert socket_path.is_symlink()
        assert os.readlink(socket_path) == os.fspath(target)
        assert (socket_path.lstat().st_dev, socket_path.lstat().st_ino) == (
            symbolic.st_dev,
            symbolic.st_ino,
        )
        socket_path.unlink()

        foreign = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        try:
            foreign.bind(os.fspath(socket_path))
            existing_socket = socket_path.lstat()
            with pytest.raises(server.ServiceStartupError, match="already exists"):
                server._bind_private_unix_socket(socket_path, backlog=8)
            observed = socket_path.lstat()
            assert stat.S_ISSOCK(observed.st_mode)
            assert (observed.st_dev, observed.st_ino) == (
                existing_socket.st_dev,
                existing_socket.st_ino,
            )
        finally:
            foreign.close()
            socket_path.unlink(missing_ok=True)


async def _wait_until(
    predicate: Callable[[], bool],
    *,
    maximum_scheduler_yields: int = 10_000,
) -> None:
    """Wait for async state without using wall-clock sleeps or unbounded polling."""

    for _ in range(maximum_scheduler_yields):
        if predicate():
            return
        await asyncio.sleep(0)
    raise AssertionError("bounded async state transition did not complete")


async def _loopback_socket_scenario(
    request_chunks: tuple[bytes, ...] = (
        b"GET /health/live HTTP/1.1\r\n" b"Host: localhost\r\n" b"Connection: close\r\n\r\n",
    ),
) -> tuple[bytes, ServiceTelemetry]:
    listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    listener.bind(("127.0.0.1", 0))
    listener.listen(8)
    listener.setblocking(False)
    address = listener.getsockname()
    assert isinstance(address, tuple)
    port = int(address[1])
    telemetry = ServiceTelemetry(ServiceMetrics())
    config = server.build_uvicorn_config(
        _NoReadPorts(),
        server.ServerConfig(port=port, backlog=8),
        telemetry=telemetry,
    )
    instance = server.uvicorn.Server(config)
    task = asyncio.create_task(instance.serve(sockets=[listener]))
    writer: asyncio.StreamWriter | None = None
    try:
        await asyncio.wait_for(_wait_until(lambda: instance.started or task.done()), timeout=5)
        if task.done():
            await task
            raise AssertionError("Uvicorn exited before accepting the loopback smoke request")
        reader, writer = await asyncio.wait_for(
            asyncio.open_connection("127.0.0.1", port),
            timeout=5,
        )
        for request_chunk in request_chunks:
            writer.write(request_chunk)
            await asyncio.wait_for(writer.drain(), timeout=5)
            await asyncio.sleep(0)
        chunks: list[bytes] = []
        response_bytes = 0
        while chunk := await asyncio.wait_for(reader.read(4 * 1_024), timeout=5):
            response_bytes += len(chunk)
            if response_bytes > 32 * 1_024:
                raise AssertionError("loopback smoke response exceeded its byte ceiling")
            chunks.append(chunk)
        return b"".join(chunks), telemetry
    finally:
        if writer is not None:
            writer.close()
            await writer.wait_closed()
        instance.should_exit = True
        await asyncio.wait_for(task, timeout=5)
        listener.close()


def test_uvicorn_serves_one_bounded_loopback_liveness_request() -> None:
    response, _telemetry = asyncio.run(_loopback_socket_scenario())

    headers, separator, body = response.partition(b"\r\n\r\n")
    assert separator
    assert headers.startswith(b"HTTP/1.1 200 OK\r\n")
    assert b"server:" not in headers.lower()
    assert b"date:" not in headers.lower()
    assert b"cache-control: no-store" in headers.lower()
    assert b"x-content-type-options: nosniff" in headers.lower()
    assert body == b'{"schema_version":1,"status":"live"}'


@pytest.mark.parametrize(
    "request_chunks",
    [
        (
            b"GET /health/live HTTP/1.1\r\n"
            b"Host: localhost\r\n"
            b"X-Invalid: prefix\x00suffix\r\n\r\n",
        ),
        (
            b"GET /health/live HTTP/1.1\r\nHost: localhost\r\nX-Oversized: ",
            b"a" * (9 * 1_024),
            b"b" * (9 * 1_024),
        ),
    ],
)
def test_h11_parser_failures_are_structured_redacted_and_telemetered(
    request_chunks: tuple[bytes, ...],
) -> None:
    response, telemetry = asyncio.run(_loopback_socket_scenario(request_chunks))

    headers, separator, body = response.partition(b"\r\n\r\n")
    assert separator
    assert headers.startswith(b"HTTP/1.1 400 Bad Request\r\n")
    assert b"content-type: application/problem+json" in headers.lower()
    assert b"cache-control: no-store" in headers.lower()
    assert b"x-content-type-options: nosniff" in headers.lower()
    assert b"connection: close" in headers.lower()
    assert b"invalid http request received" not in response.lower()
    assert b"prefix" not in response and b"suffix" not in response
    problem = json.loads(body)
    assert problem == {
        "code": "invalid_request",
        "detail": "The request violates the bounded read-only API contract.",
        "request_id": problem["request_id"],
        "status": 400,
        "title": "Invalid request",
        "type": "urn:signalattice:problem:invalid_request",
    }
    assert len(problem["request_id"]) == 32

    metrics = telemetry.metrics.snapshot().body
    assert b'signalattice_http_requests_total{outcome="rejected",route="unmatched"} 1' in metrics
    assert b'signalattice_admission_rejections_total{reason="invalid_metadata"} 1' in metrics
    records = [json.loads(record) for record in telemetry.local_snapshot().records]
    assert any(
        record["event"] == "request_rejected"
        and record["attributes"]["route"] == "unmatched"
        and record["attributes"]["rejection"] == "invalid_metadata"
        for record in records
    )
    assert any(
        record["event"] == "request_span"
        and record["attributes"]["route"] == "unmatched"
        and record["attributes"]["rejection"] == "invalid_metadata"
        for record in records
    )


async def _rejected_body_connection_scenario(
    request: bytes,
) -> tuple[bytes, float, bool, bool, int, int, int]:
    """Exercise one pre-body rejection across a real bounded TCP connection."""

    listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    listener.bind(("127.0.0.1", 0))
    listener.listen(8)
    listener.setblocking(False)
    address = listener.getsockname()
    assert isinstance(address, tuple)
    port = int(address[1])
    admission = AdmissionController(
        AdmissionLimits(
            global_concurrency=32,
            data_concurrency=24,
            api_rate_per_second=10_000.0,
            api_burst=10_000,
            operations_rate_per_second=10_000.0,
            operations_burst=1_000,
        )
    )
    config = server.build_uvicorn_config(
        _NoReadPorts(),
        server.ServerConfig(port=port, backlog=8),
        admission=admission,
    )
    instance = server.uvicorn.Server(config)
    task = asyncio.create_task(instance.serve(sockets=[listener]))
    writer: asyncio.StreamWriter | None = None
    late_write_failed = False
    try:
        await asyncio.wait_for(_wait_until(lambda: instance.started or task.done()), timeout=5)
        if task.done():
            await task
            raise AssertionError("Uvicorn exited before the rejected-body socket test")

        started = time.monotonic()
        chunks: list[bytes] = []
        observed = 0
        async with asyncio.timeout(0.25):
            reader, writer = await asyncio.open_connection("127.0.0.1", port)
            writer.write(request)
            await writer.drain()
            while chunk := await reader.read(4 * 1_024):
                observed += len(chunk)
                if observed > 32 * 1_024:
                    raise AssertionError("rejected-body response exceeded its byte ceiling")
                chunks.append(chunk)
        elapsed = time.monotonic() - started
        assert reader.at_eof()

        try:
            writer.write(b"1\r\nx\r\n0\r\n\r\n")
            await asyncio.wait_for(writer.drain(), timeout=0.25)
        except (BrokenPipeError, ConnectionError, OSError):
            late_write_failed = True
        assert await asyncio.wait_for(reader.read(1), timeout=0.25) == b""
        closed_after_late_write = reader.at_eof()
        await asyncio.wait_for(
            _wait_until(
                lambda: not instance.server_state.connections
                and not instance.server_state.tasks
                and admission.snapshot().active_total == 0
            ),
            timeout=1,
        )
        return (
            b"".join(chunks),
            elapsed,
            late_write_failed,
            closed_after_late_write,
            admission.snapshot().active_total,
            len(instance.server_state.connections),
            len(instance.server_state.tasks),
        )
    finally:
        if writer is not None:
            writer.close()
            await asyncio.gather(writer.wait_closed(), return_exceptions=True)
        instance.should_exit = True
        await asyncio.wait_for(task, timeout=5)
        listener.close()


@pytest.mark.parametrize(
    ("wire_request", "expected_status", "expected_code"),
    [
        (
            b"GET /api/v1/runs HTTP/1.1\r\n"
            b"Host: localhost\r\n"
            b"Transfer-Encoding: chunked\r\n\r\n",
            400,
            "invalid_request",
        ),
        (
            b"GET /api/v1/runs HTTP/1.1\r\n"
            b"Host: localhost\r\n"
            b"Content-Length: 1048576\r\n\r\n",
            413,
            "request_body_forbidden",
        ),
    ],
)
def test_rejected_body_connections_close_promptly_and_release_every_bound(
    wire_request: bytes,
    expected_status: int,
    expected_code: str,
) -> None:
    response, elapsed, late_write_failed, closed_after_late_write, active, connections, tasks = (
        asyncio.run(_rejected_body_connection_scenario(wire_request))
    )

    headers, separator, body = response.partition(b"\r\n\r\n")
    assert separator
    assert headers.startswith(f"HTTP/1.1 {expected_status} ".encode("ascii"))
    assert b"content-type: application/problem+json" in headers.lower()
    assert b"connection: close" in headers.lower()
    assert json.loads(body)["code"] == expected_code
    assert elapsed <= 0.25
    assert late_write_failed or closed_after_late_write
    assert (active, connections, tasks) == (0, 0, 0)


async def _encoded_operation_rate_scenario() -> list[bytes]:
    """Send canonical-equivalent aliases through one fixed operations bucket."""

    listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    listener.bind(("127.0.0.1", 0))
    listener.listen(8)
    listener.setblocking(False)
    address = listener.getsockname()
    assert isinstance(address, tuple)
    port = int(address[1])
    admission = AdmissionController(AdmissionLimits(), clock=lambda: 100.0)
    config = server.build_uvicorn_config(
        _NoReadPorts(),
        server.ServerConfig(port=port, backlog=8),
        admission=admission,
    )
    instance = server.uvicorn.Server(config)
    task = asyncio.create_task(instance.serve(sockets=[listener]))
    writers: list[asyncio.StreamWriter] = []
    responses: list[bytes] = []
    try:
        await asyncio.wait_for(_wait_until(lambda: instance.started or task.done()), timeout=5)
        if task.done():
            await task
            raise AssertionError("Uvicorn exited before the encoded-path socket test")
        for _ in range(5):
            reader, writer = await asyncio.wait_for(
                asyncio.open_connection("127.0.0.1", port),
                timeout=1,
            )
            writers.append(writer)
            writer.write(b"GET /health/%6cive HTTP/1.1\r\n" b"Host: localhost\r\n\r\n")
            await asyncio.wait_for(writer.drain(), timeout=1)
            chunks: list[bytes] = []
            observed = 0
            while chunk := await asyncio.wait_for(reader.read(4 * 1_024), timeout=1):
                observed += len(chunk)
                if observed > 32 * 1_024:
                    raise AssertionError("encoded-path response exceeded its byte ceiling")
                chunks.append(chunk)
            responses.append(b"".join(chunks))
        await asyncio.wait_for(
            _wait_until(
                lambda: not instance.server_state.connections
                and not instance.server_state.tasks
                and admission.snapshot().active_total == 0
            ),
            timeout=1,
        )
        return responses
    finally:
        for writer in writers:
            writer.close()
        await asyncio.gather(*(writer.wait_closed() for writer in writers), return_exceptions=True)
        instance.should_exit = True
        await asyncio.wait_for(task, timeout=5)
        listener.close()


def test_real_encoded_operation_alias_cannot_escape_operations_rate_budget() -> None:
    responses = asyncio.run(_encoded_operation_rate_scenario())

    problems = [json.loads(response.partition(b"\r\n\r\n")[2]) for response in responses]
    assert [problem["status"] for problem in problems] == [400, 400, 400, 400, 429]
    assert [problem["code"] for problem in problems] == ["invalid_request"] * 4 + ["rate_limited"]
    assert all(b"connection: close" in response.lower() for response in responses)


async def _transport_saturation_scenario() -> tuple[bytes, float, ServiceTelemetry, list[bytes]]:
    listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    listener.bind(("127.0.0.1", 0))
    listener.listen(64)
    listener.setblocking(False)
    address = listener.getsockname()
    assert isinstance(address, tuple)
    port = int(address[1])
    ports = _BlockingReadPorts(target=32)
    admission = AdmissionController(
        AdmissionLimits(
            global_concurrency=32,
            data_concurrency=32,
            api_rate_per_second=10_000.0,
            api_burst=10_000,
            operations_rate_per_second=10_000.0,
            operations_burst=1_000,
        )
    )
    telemetry = ServiceTelemetry(ServiceMetrics())
    config = server.build_uvicorn_config(
        ports,  # type: ignore[arg-type]
        server.ServerConfig(
            port=port,
            transport_concurrency=32,
            data_concurrency=32,
            backlog=64,
        ),
        admission=admission,
        telemetry=telemetry,
    )
    instance = server.uvicorn.Server(config)
    server_task = asyncio.create_task(instance.serve(sockets=[listener]))
    writers: list[asyncio.StreamWriter] = []

    async def connect() -> tuple[asyncio.StreamReader, asyncio.StreamWriter]:
        reader, writer = await asyncio.wait_for(
            asyncio.open_connection("127.0.0.1", port),
            timeout=5,
        )
        writers.append(writer)
        writer.write(
            b"GET /api/v1/runs HTTP/1.1\r\n" b"Host: localhost\r\n" b"Connection: close\r\n\r\n"
        )
        await asyncio.wait_for(writer.drain(), timeout=5)
        return reader, writer

    async def read_response(reader: asyncio.StreamReader) -> bytes:
        chunks: list[bytes] = []
        observed = 0
        while chunk := await asyncio.wait_for(reader.read(4 * 1_024), timeout=5):
            observed += len(chunk)
            if observed > 32 * 1_024:
                raise AssertionError("transport test response exceeded its byte ceiling")
            chunks.append(chunk)
        return b"".join(chunks)

    try:
        await asyncio.wait_for(
            _wait_until(lambda: instance.started or server_task.done()), timeout=5
        )
        if server_task.done():
            await server_task
            raise AssertionError("Uvicorn exited before the saturation scenario")
        held = await asyncio.gather(*(connect() for _ in range(32)))
        entered = await asyncio.wait_for(
            asyncio.to_thread(ports.wait_until_full, 5.0),
            timeout=6,
        )
        assert entered

        started = time.monotonic()
        async with asyncio.timeout(0.25):
            rejected_reader, _rejected_writer = await connect()
            rejected = await read_response(rejected_reader)
        rejection_seconds = time.monotonic() - started
        await asyncio.wait_for(
            _wait_until(
                lambda: len(instance.server_state.connections) == 32
                and len(instance.server_state.tasks) == 32
            ),
            timeout=1,
        )
        ports.release()
        accepted = await asyncio.gather(*(read_response(reader) for reader, _ in held))
        return rejected, rejection_seconds, telemetry, accepted
    finally:
        ports.release()
        for writer in writers:
            writer.close()
        await asyncio.gather(*(writer.wait_closed() for writer in writers), return_exceptions=True)
        instance.should_exit = True
        await asyncio.wait_for(server_task, timeout=5)
        listener.close()


def test_transport_admits_32_and_structurally_rejects_the_33rd() -> None:
    rejected, rejection_seconds, telemetry, accepted = asyncio.run(_transport_saturation_scenario())

    assert len(accepted) == 32
    assert all(response.startswith(b"HTTP/1.1 200 OK\r\n") for response in accepted)
    headers, separator, body = rejected.partition(b"\r\n\r\n")
    assert separator
    assert headers.startswith(b"HTTP/1.1 429 Too Many Requests\r\n")
    assert b"content-type: application/problem+json" in headers.lower()
    assert b"retry-after: 1" in headers.lower()
    assert b"text/plain" not in rejected.lower()
    assert json.loads(body)["code"] == "service_saturated"
    assert rejection_seconds <= 0.25
    metrics = telemetry.metrics.snapshot().body
    assert b'signalattice_admission_rejections_total{reason="global_concurrency"} 1' in metrics


async def _keep_alive_transition_scenario() -> tuple[bytes, list[bytes]]:
    """Retire one keep-alive task while 31 other data exchanges remain active."""

    listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    listener.bind(("127.0.0.1", 0))
    listener.listen(64)
    listener.setblocking(False)
    address = listener.getsockname()
    assert isinstance(address, tuple)
    port = int(address[1])
    ports = _BlockingReadPorts(target=32)
    admission = AdmissionController(
        AdmissionLimits(
            global_concurrency=32,
            data_concurrency=32,
            api_rate_per_second=10_000.0,
            api_burst=10_000,
            operations_rate_per_second=10_000.0,
            operations_burst=1_000,
        )
    )
    config = server.build_uvicorn_config(
        ports,  # type: ignore[arg-type]
        server.ServerConfig(
            port=port,
            transport_concurrency=32,
            data_concurrency=32,
            backlog=64,
        ),
        admission=admission,
    )
    instance = server.uvicorn.Server(config)
    server_task = asyncio.create_task(instance.serve(sockets=[listener]))
    writers: list[asyncio.StreamWriter] = []

    async def open_data_request() -> tuple[asyncio.StreamReader, asyncio.StreamWriter]:
        reader, writer = await asyncio.wait_for(
            asyncio.open_connection("127.0.0.1", port),
            timeout=2,
        )
        writers.append(writer)
        writer.write(
            b"GET /api/v1/runs HTTP/1.1\r\n" b"Host: localhost\r\nConnection: close\r\n\r\n"
        )
        await asyncio.wait_for(writer.drain(), timeout=2)
        return reader, writer

    async def read_bounded(reader: asyncio.StreamReader, maximum: int) -> bytes:
        chunks: list[bytes] = []
        observed = 0
        while chunk := await asyncio.wait_for(reader.read(4 * 1_024), timeout=5):
            observed += len(chunk)
            if observed > maximum:
                raise AssertionError("keep-alive transport response exceeded its byte ceiling")
            chunks.append(chunk)
        return b"".join(chunks)

    try:
        await asyncio.wait_for(
            _wait_until(lambda: instance.started or server_task.done()), timeout=5
        )
        if server_task.done():
            await server_task
            raise AssertionError("Uvicorn exited before the keep-alive transition scenario")
        held = await asyncio.gather(*(open_data_request() for _ in range(31)))
        entered_31 = await asyncio.wait_for(
            asyncio.to_thread(ports.wait_until_one_before_full, 5.0),
            timeout=6,
        )
        assert entered_31

        pipeline_reader, pipeline_writer = await asyncio.wait_for(
            asyncio.open_connection("127.0.0.1", port),
            timeout=2,
        )
        writers.append(pipeline_writer)
        pipeline_writer.write(
            b"GET /health/live HTTP/1.1\r\nHost: localhost\r\n\r\n"
            b"GET /api/v1/runs HTTP/1.1\r\nHost: localhost\r\nConnection: close\r\n\r\n"
        )
        await asyncio.wait_for(pipeline_writer.drain(), timeout=2)
        entered_32 = await asyncio.wait_for(
            asyncio.to_thread(ports.wait_until_full, 5.0),
            timeout=6,
        )
        assert entered_32

        ports.release()
        held_responses = await asyncio.gather(
            *(read_bounded(reader, 32 * 1_024) for reader, _ in held)
        )
        pipeline_response = await read_bounded(pipeline_reader, 64 * 1_024)
        await asyncio.wait_for(
            _wait_until(
                lambda: not instance.server_state.connections
                and not instance.server_state.tasks
                and admission.snapshot().active_total == 0
            ),
            timeout=1,
        )
        return pipeline_response, held_responses
    finally:
        ports.release()
        for writer in writers:
            writer.close()
        await asyncio.gather(*(writer.wait_closed() for writer in writers), return_exceptions=True)
        instance.should_exit = True
        await asyncio.wait_for(server_task, timeout=5)
        listener.close()


def test_retiring_keep_alive_task_is_not_counted_against_its_buffered_successor() -> None:
    pipeline, held = asyncio.run(_keep_alive_transition_scenario())

    assert len(held) == 31
    assert all(response.startswith(b"HTTP/1.1 200 OK\r\n") for response in held)
    assert pipeline.count(b"HTTP/1.1 200 OK\r\n") == 2
    assert b"429 Too Many Requests" not in pipeline


def test_operator_server_rejects_internal_ephemeral_port_before_keychain_read(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    def unexpected_keychain_read(**_kwargs: object) -> bytes:
        raise AssertionError("keychain must not be read")

    monkeypatch.setattr(server, "read_digest_secret_from_keychain", unexpected_keychain_read)

    with pytest.raises(server.ServiceStartupError, match="explicit nonzero port"):
        server.serve_existing_evidence(
            tmp_path / "registry.db",
            tmp_path / "cas",
            config=server.ServerConfig(port=0),
        )
