"""Fail-closed assembly and Uvicorn profile for the local evidence API."""

from __future__ import annotations

import getpass
import os
import re
import socket
import stat
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Final

import uvicorn

from quant_platform.service.admission import AdmissionController
from quant_platform.service.api import assert_read_only_route_inventory, create_app
from quant_platform.service.http_protocol import configured_h11_protocol
from quant_platform.service.metrics import ServiceMetrics
from quant_platform.service.telemetry import ServiceTelemetry
from quant_platform.tracking.cas import ArtifactStore, ArtifactStoreError
from quant_platform.tracking.contracts import RegistryError, RegistryLimits, ValidationError
from quant_platform.tracking.read_ports import ReadPortLimits, RegistryReadPorts
from quant_platform.tracking.registry import RunRegistry

DEFAULT_KEYCHAIN_SERVICE: Final = "com.signal-foundry.signalattice-registry"
MAX_UNIX_SOCKET_PATH_BYTES: Final = 100
_KEYCHAIN_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")


class ServiceStartupError(RuntimeError):
    """Local service configuration or existing evidence state is not safe to serve."""


@dataclass(frozen=True, slots=True)
class _BoundUnixSocket:
    """One private listener and the exact filesystem identity it created."""

    listener: socket.socket
    path: Path
    device: int
    inode: int


@dataclass(frozen=True, slots=True)
class ServerConfig:
    """Closed resource profile for one loopback Uvicorn process."""

    host: str = "127.0.0.1"
    port: int = 8765
    socket_path: Path | None = None
    transport_concurrency: int = 32
    data_concurrency: int = 24
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
        if self.socket_path is not None:
            if not isinstance(self.socket_path, Path) or not self.socket_path.is_absolute():
                raise ServiceStartupError("socket_path must be an absolute pathlib.Path")
            rendered_socket = os.fspath(self.socket_path)
            try:
                encoded_socket = os.fsencode(rendered_socket)
            except (UnicodeEncodeError, ValueError):
                raise ServiceStartupError(
                    "socket_path violates the bounded Unix-socket contract"
                ) from None
            if (
                not 1 <= len(encoded_socket) <= MAX_UNIX_SOCKET_PATH_BYTES
                or self.socket_path.name in {"", ".", ".."}
                or b"\x00" in encoded_socket
            ):
                raise ServiceStartupError("socket_path violates the bounded Unix-socket contract")
        bounds = {
            "transport_concurrency": (self.transport_concurrency, 1, 1_024),
            "data_concurrency": (self.data_concurrency, 1, 1_024),
            "backlog": (self.backlog, 1, 1_024),
            "keep_alive_seconds": (self.keep_alive_seconds, 1, 30),
            "graceful_shutdown_seconds": (self.graceful_shutdown_seconds, 1, 60),
            "incomplete_event_bytes": (self.incomplete_event_bytes, 4 * 1024, 64 * 1024),
        }
        for name, (value, lower, upper) in bounds.items():
            if type(value) is not int or not lower <= value <= upper:
                raise ServiceStartupError(f"{name} is outside the supported local-service bound")
        if self.data_concurrency > self.transport_concurrency:
            raise ServiceStartupError("data_concurrency cannot exceed transport_concurrency")


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


def read_digest_secret_from_file(path: Path) -> bytes:
    """Read one private runtime secret without following links or reflecting data.

    The file must already exist, be a single regular file, and grant no access
    to ``other`` users.  Group read is accepted only when the file's group is
    the service process group.  At most 257 bytes are read, permitting a single
    terminal line ending while proving larger values are rejected.
    """

    if not isinstance(path, Path) or not path.is_absolute():
        raise ServiceStartupError("digest key file must be an absolute pathlib.Path")
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    descriptor: int | None = None
    try:
        descriptor = os.open(path, flags)
        metadata = os.fstat(descriptor)
        mode = stat.S_IMODE(metadata.st_mode)
        if (
            not stat.S_ISREG(metadata.st_mode)
            or metadata.st_nlink != 1
            or mode & 0o007
            or mode & 0o222
            or (mode & 0o040 and metadata.st_gid != os.getegid())
            or metadata.st_uid not in {0, os.geteuid()}
        ):
            raise ServiceStartupError("digest key file violates the private-file contract")
        chunks: list[bytes] = []
        remaining = 258
        while remaining:
            chunk = os.read(descriptor, remaining)
            if not chunk:
                break
            chunks.append(chunk)
            remaining -= len(chunk)
        secret = b"".join(chunks)
    except ServiceStartupError:
        raise
    except OSError:
        raise ServiceStartupError("unable to read the digest key file") from None
    finally:
        if descriptor is not None:
            os.close(descriptor)
    if secret.endswith(b"\r\n"):
        secret = secret[:-2]
    elif secret.endswith(b"\n"):
        secret = secret[:-1]
    if not 32 <= len(secret) <= 256 or b"\n" in secret or b"\r" in secret:
        raise ServiceStartupError("digest key file violates the 32-to-256-byte contract")
    return secret


def validate_socket_runtime(path: Path) -> None:
    """Fail closed unless an absent socket has one private owned directory."""

    if not isinstance(path, Path) or not path.is_absolute():
        raise ServiceStartupError("socket path must be absolute")
    try:
        parent_lstat = path.parent.lstat()
        resolved_parent = path.parent.resolve(strict=True)
    except OSError:
        raise ServiceStartupError("socket runtime directory is unavailable") from None
    mode = stat.S_IMODE(parent_lstat.st_mode)
    if (
        not stat.S_ISDIR(parent_lstat.st_mode)
        or resolved_parent != path.parent
        or parent_lstat.st_uid != os.geteuid()
        or mode & 0o077
    ):
        raise ServiceStartupError(
            "socket runtime directory violates the private-directory contract"
        )
    try:
        path.lstat()
    except FileNotFoundError:
        return
    except OSError:
        raise ServiceStartupError("socket path cannot be inspected safely") from None
    raise ServiceStartupError("socket path already exists; stale state requires explicit removal")


def _bind_private_unix_socket(path: Path, *, backlog: int) -> _BoundUnixSocket:
    """Bind one owner-only listener without exposing Uvicorn's mode-0666 race.

    Uvicorn 0.52.1 creates an absent UDS and then explicitly changes its mode to
    ``0666``, overriding the process umask.  Signalattice therefore creates the
    socket itself under umask ``0077``, narrows it to ``0600``, verifies the
    filesystem object, and passes the already-listening descriptor to Uvicorn.
    """

    if type(backlog) is not int or not 1 <= backlog <= 1_024:
        raise ServiceStartupError("socket backlog is outside the supported bound")
    validate_socket_runtime(path)
    listener: socket.socket | None = None
    binding: _BoundUnixSocket | None = None
    previous_umask = os.umask(0o077)
    try:
        listener = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        listener.bind(os.fspath(path))
        created = path.lstat()
        if not stat.S_ISSOCK(created.st_mode) or created.st_uid != os.geteuid():
            raise ServiceStartupError("bound socket violates the private-socket contract")
        binding = _BoundUnixSocket(
            listener=listener,
            path=path,
            device=created.st_dev,
            inode=created.st_ino,
        )
        os.chmod(path, 0o600, follow_symlinks=False)
        listener.listen(backlog)
        listener.setblocking(False)
        metadata = path.lstat()
        mode = stat.S_IMODE(metadata.st_mode)
        if (
            not stat.S_ISSOCK(metadata.st_mode)
            or metadata.st_uid != os.geteuid()
            or mode != 0o600
            or metadata.st_nlink != 1
            or metadata.st_dev != binding.device
            or metadata.st_ino != binding.inode
        ):
            raise ServiceStartupError("bound socket violates the private-socket contract")
        return binding
    except ServiceStartupError:
        if listener is not None:
            listener.close()
        if binding is not None:
            _unlink_matching_socket(path, expected=binding)
        raise
    except (OSError, ValueError):
        if listener is not None:
            listener.close()
        if binding is not None:
            _unlink_matching_socket(path, expected=binding)
        raise ServiceStartupError("unable to bind the private Unix socket") from None
    finally:
        os.umask(previous_umask)


def _unlink_matching_socket(
    path: Path,
    *,
    expected: _BoundUnixSocket,
) -> None:
    """Unlink only the exact owned socket created by this process."""

    try:
        metadata = path.lstat()
    except FileNotFoundError:
        return
    except OSError:
        raise ServiceStartupError("unable to inspect the Unix socket during cleanup") from None
    if (
        not stat.S_ISSOCK(metadata.st_mode)
        or metadata.st_uid != os.geteuid()
        or metadata.st_dev != expected.device
        or metadata.st_ino != expected.inode
    ):
        raise ServiceStartupError("Unix socket identity changed; refusing unsafe cleanup")
    try:
        path.unlink()
    except OSError:
        raise ServiceStartupError("unable to remove the owned Unix socket") from None


def _close_private_unix_socket(binding: _BoundUnixSocket) -> None:
    """Close one listener and remove only its unchanged filesystem identity."""

    if type(binding) is not _BoundUnixSocket:
        raise ServiceStartupError("socket binding has an invalid type")
    binding.listener.close()
    _unlink_matching_socket(binding.path, expected=binding)


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
    *,
    admission: AdmissionController | None = None,
    telemetry: ServiceTelemetry | None = None,
) -> uvicorn.Config:
    """Return the audited local Uvicorn configuration without starting sockets."""

    if type(config) is not ServerConfig:
        raise ServiceStartupError("config must be a ServerConfig")
    resolved_telemetry = ServiceTelemetry(ServiceMetrics()) if telemetry is None else telemetry
    if type(resolved_telemetry) is not ServiceTelemetry:
        raise ServiceStartupError("telemetry must be a ServiceTelemetry")
    if admission is not None and type(admission) is not AdmissionController:
        raise ServiceStartupError("admission must be an AdmissionController")
    if admission is not None and (
        admission.limits.global_concurrency != config.transport_concurrency
        or admission.limits.data_concurrency != config.data_concurrency
    ):
        raise ServiceStartupError("admission limits must match the server resource profile")
    app = create_app(
        ports,
        max_concurrency=config.transport_concurrency,
        max_data_concurrency=config.data_concurrency,
        allowed_port=(None if config.socket_path is not None or config.port == 0 else config.port),
        admission=admission,
        telemetry=resolved_telemetry,
    )
    assert_read_only_route_inventory(app)
    return uvicorn.Config(
        app,
        host=config.host,
        port=config.port,
        # The UDS profile passes an already-bound private descriptor to
        # ``Server.run``.  Leaving this unset prevents Uvicorn 0.52.1 from
        # changing the socket mode to 0666 after creation.
        uds=None,
        http=configured_h11_protocol(resolved_telemetry),
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
        log_level="critical",
        server_header=False,
        date_header=False,
        # The pinned h11 adapter corrects Uvicorn's inclusive connection-count
        # off-by-one and replaces its raw text/plain overload responder.  The
        # configured value is therefore the exact transport ceiling, while the
        # outer ASGI controller independently enforces the same exchange bound.
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
    digest_key_file: Path | None = None,
    config: ServerConfig | None = None,
) -> None:
    """Verify existing state, then run one bounded local-only server."""

    server_config = ServerConfig() if config is None else config
    if type(server_config) is not ServerConfig:
        raise ServiceStartupError("config must be a ServerConfig")
    if server_config.port == 0 and server_config.socket_path is None:
        raise ServiceStartupError("operator service startup requires an explicit nonzero port")
    if digest_key_file is not None and not isinstance(digest_key_file, Path):
        raise ServiceStartupError("digest_key_file must be a pathlib.Path")
    secret = (
        read_digest_secret_from_keychain(service=keychain_service)
        if digest_key_file is None
        else read_digest_secret_from_file(digest_key_file)
    )
    ports = open_existing_read_ports(
        registry_path,
        cas_root,
        digest_secret=secret,
    )
    uvicorn_config = build_uvicorn_config(ports, server_config)
    binding = (
        None
        if server_config.socket_path is None
        else _bind_private_unix_socket(
            server_config.socket_path,
            backlog=server_config.backlog,
        )
    )
    try:
        uvicorn.Server(uvicorn_config).run(sockets=None if binding is None else [binding.listener])
    finally:
        if binding is not None:
            _close_private_unix_socket(binding)
