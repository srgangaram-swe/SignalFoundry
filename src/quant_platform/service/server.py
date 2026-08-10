"""Fail-closed assembly and Uvicorn profile for the local evidence API."""

from __future__ import annotations

import getpass
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Final

import uvicorn

from quant_platform.service.api import assert_read_only_route_inventory, create_app
from quant_platform.tracking.cas import ArtifactStore, ArtifactStoreError
from quant_platform.tracking.contracts import RegistryError, RegistryLimits, ValidationError
from quant_platform.tracking.read_ports import ReadPortLimits, RegistryReadPorts
from quant_platform.tracking.registry import RunRegistry

DEFAULT_KEYCHAIN_SERVICE: Final = "com.signal-foundry.signalattice-registry"
_KEYCHAIN_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")


class ServiceStartupError(RuntimeError):
    """Local service configuration or existing evidence state is not safe to serve."""


@dataclass(frozen=True, slots=True)
class ServerConfig:
    """Closed resource profile for one loopback Uvicorn process."""

    host: str = "127.0.0.1"
    port: int = 8765
    max_concurrency: int = 32
    transport_concurrency: int = 64
    backlog: int = 64
    keep_alive_seconds: int = 3
    graceful_shutdown_seconds: int = 10
    incomplete_event_bytes: int = 16 * 1024

    def __post_init__(self) -> None:
        if self.host != "127.0.0.1":
            raise ServiceStartupError("the evidence API may bind only to 127.0.0.1")
        if type(self.port) is not int or (self.port != 0 and not 1_024 <= self.port <= 65_535):
            raise ServiceStartupError(
                "port must be zero for an internal socket test or in [1024, 65535]"
            )
        bounds = {
            "max_concurrency": (self.max_concurrency, 1, 1_024),
            "transport_concurrency": (self.transport_concurrency, 2, 2_048),
            "backlog": (self.backlog, 1, 1_024),
            "keep_alive_seconds": (self.keep_alive_seconds, 1, 30),
            "graceful_shutdown_seconds": (self.graceful_shutdown_seconds, 1, 60),
            "incomplete_event_bytes": (self.incomplete_event_bytes, 4 * 1024, 64 * 1024),
        }
        for name, (value, lower, upper) in bounds.items():
            if type(value) is not int or not lower <= value <= upper:
                raise ServiceStartupError(f"{name} is outside the supported local-service bound")
        if self.transport_concurrency <= self.max_concurrency:
            raise ServiceStartupError(
                "transport_concurrency must exceed the application concurrency limit"
            )


def read_digest_secret_from_keychain(
    *,
    service: str = DEFAULT_KEYCHAIN_SERVICE,
    account: str | None = None,
) -> bytes:
    """Read a registry HMAC secret from macOS Keychain without shell expansion.

    The credential is captured only in process memory, is never accepted as a
    CLI argument, and is never included in an error message.  The keychain item
    must contain at least 32 and at most 256 raw bytes.  The single line ending
    emitted by ``security -w`` is removed; embedded line breaks are rejected.
    """

    if type(service) is not str or _KEYCHAIN_NAME.fullmatch(service) is None:
        raise ServiceStartupError("keychain service must be a bounded public identifier")
    resolved_account = getpass.getuser() if account is None else account
    if type(resolved_account) is not str or _KEYCHAIN_NAME.fullmatch(resolved_account) is None:
        raise ServiceStartupError("keychain account must be a bounded public identifier")
    try:
        completed = subprocess.run(
            [
                "security",
                "find-generic-password",
                "-a",
                resolved_account,
                "-s",
                service,
                "-w",
            ],
            check=False,
            capture_output=True,
            timeout=5,
        )
    except (OSError, subprocess.SubprocessError):
        raise ServiceStartupError(
            "unable to read the registry secret from macOS Keychain"
        ) from None
    secret = completed.stdout
    if secret.endswith(b"\r\n"):
        secret = secret[:-2]
    elif secret.endswith(b"\n"):
        secret = secret[:-1]
    if (
        completed.returncode != 0
        or not 32 <= len(secret) <= 256
        or b"\n" in secret
        or b"\r" in secret
    ):
        raise ServiceStartupError(
            "registry Keychain credential is missing or violates the 32-to-256-byte contract"
        )
    return secret


def open_existing_read_ports(
    registry_path: Path,
    cas_root: Path,
    *,
    digest_secret: bytes,
) -> RegistryReadPorts:
    """Open and verify existing registry/CAS state without bootstrap or migration."""

    try:
        registry = RunRegistry(
            registry_path,
            digest_secret=digest_secret,
            limits=RegistryLimits(
                busy_timeout_ms=500,
                verification_timeout_ms=1_000,
                max_page_size=100,
            ),
        )
        registry_state = registry.probe_readiness()
        if not registry_state.ready:
            raise ServiceStartupError("existing registry state is not ready for bounded reads")
        store_id = registry.artifact_store_id
        if store_id is None:
            raise ServiceStartupError("existing registry is not bound to an artifact store")
        store = ArtifactStore(cas_root, expected_store_id=store_id)
        # With an expected identity this is reopen-only: missing state fails and
        # no directory, marker, staging entry, or object may be created/repaired.
        store.initialize()
        ports = RegistryReadPorts(
            registry,
            store,
            limits=ReadPortLimits(
                max_page_size=100,
                query_timeout_ms=500,
                max_manifest_bytes=512 * 1024,
            ),
        )
        if not ports.probe_evidence_readiness().ready:
            raise ServiceStartupError("existing registry/CAS evidence boundary is not ready")
        return ports
    except ServiceStartupError:
        raise
    except (ArtifactStoreError, RegistryError, ValidationError, OSError):
        raise ServiceStartupError("existing evidence state failed startup verification") from None


def build_uvicorn_config(
    ports: RegistryReadPorts,
    config: ServerConfig,
) -> uvicorn.Config:
    """Return the audited local Uvicorn configuration without starting sockets."""

    if type(config) is not ServerConfig:
        raise ServiceStartupError("config must be a ServerConfig")
    app = create_app(
        ports,
        max_concurrency=config.max_concurrency,
        allowed_port=None if config.port == 0 else config.port,
    )
    assert_read_only_route_inventory(app)
    return uvicorn.Config(
        app,
        host=config.host,
        port=config.port,
        http="h11",
        ws="none",
        lifespan="on",
        loop="asyncio",
        interface="asgi3",
        workers=1,
        reload=False,
        log_config=None,
        proxy_headers=False,
        forwarded_allow_ips="",
        access_log=False,
        server_header=False,
        date_header=False,
        limit_concurrency=config.transport_concurrency,
        backlog=config.backlog,
        timeout_keep_alive=config.keep_alive_seconds,
        timeout_graceful_shutdown=config.graceful_shutdown_seconds,
        h11_max_incomplete_event_size=config.incomplete_event_bytes,
        reset_contextvars=True,
    )


def serve_existing_evidence(
    registry_path: Path,
    cas_root: Path,
    *,
    keychain_service: str = DEFAULT_KEYCHAIN_SERVICE,
    config: ServerConfig | None = None,
) -> None:
    """Verify existing state, then run one bounded loopback-only server."""

    server_config = ServerConfig() if config is None else config
    if type(server_config) is not ServerConfig:
        raise ServiceStartupError("config must be a ServerConfig")
    if server_config.port == 0:
        raise ServiceStartupError("operator service startup requires an explicit nonzero port")
    secret = read_digest_secret_from_keychain(service=keychain_service)
    ports = open_existing_read_ports(
        registry_path,
        cas_root,
        digest_secret=secret,
    )
    uvicorn.Server(build_uvicorn_config(ports, server_config)).run()
