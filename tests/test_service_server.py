"""Security and failure-boundary tests for local service assembly."""

from __future__ import annotations

import asyncio
import socket
import subprocess
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

from quant_platform.service import server


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


@pytest.mark.parametrize(
    "overrides",
    [
        {"host": "0.0.0.0"},
        {"host": "::1"},
        {"port": True},
        {"port": 1_023},
        {"max_concurrency": 0},
        {"transport_concurrency": 32, "max_concurrency": 32},
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
    assert config.http == "h11"
    assert config.ws == "none"
    assert config.lifespan == "on"
    assert config.loop == "asyncio"
    assert config.interface == "asgi3"
    assert config.workers == 1
    assert config.reload is False
    assert config.proxy_headers is False
    assert config.forwarded_allow_ips == ""
    assert config.access_log is False
    assert config.server_header is False
    assert config.date_header is False
    assert config.limit_concurrency == 64
    assert config.backlog == 64
    assert config.timeout_keep_alive == 3
    assert config.timeout_graceful_shutdown == 10
    assert config.h11_max_incomplete_event_size == 16 * 1_024
    assert config.reset_contextvars is True


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


async def _loopback_socket_scenario() -> bytes:
    listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    listener.bind(("127.0.0.1", 0))
    listener.listen(8)
    listener.setblocking(False)
    address = listener.getsockname()
    assert isinstance(address, tuple)
    port = int(address[1])
    config = server.build_uvicorn_config(
        _NoReadPorts(),
        server.ServerConfig(port=port, backlog=8),
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
        writer.write(
            b"GET /health/live HTTP/1.1\r\n" b"Host: localhost\r\n" b"Connection: close\r\n\r\n"
        )
        await asyncio.wait_for(writer.drain(), timeout=5)
        chunks: list[bytes] = []
        response_bytes = 0
        while chunk := await asyncio.wait_for(reader.read(4 * 1_024), timeout=5):
            response_bytes += len(chunk)
            if response_bytes > 32 * 1_024:
                raise AssertionError("loopback smoke response exceeded its byte ceiling")
            chunks.append(chunk)
        return b"".join(chunks)
    finally:
        if writer is not None:
            writer.close()
            await writer.wait_closed()
        instance.should_exit = True
        await asyncio.wait_for(task, timeout=5)
        listener.close()


def test_uvicorn_serves_one_bounded_loopback_liveness_request() -> None:
    response = asyncio.run(_loopback_socket_scenario())

    headers, separator, body = response.partition(b"\r\n\r\n")
    assert separator
    assert headers.startswith(b"HTTP/1.1 200 OK\r\n")
    assert b"server:" not in headers.lower()
    assert b"date:" not in headers.lower()
    assert b"cache-control: no-store" in headers.lower()
    assert b"x-content-type-options: nosniff" in headers.lower()
    assert body == b'{"schema_version":1,"status":"live"}'


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
