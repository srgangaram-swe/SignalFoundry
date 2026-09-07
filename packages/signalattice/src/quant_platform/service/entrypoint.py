"""Narrow container entry point for the read-only evidence service.

The research CLI intentionally remains a separate Typer application. This
module uses only the standard library and the service/storage boundary so the
dedicated runtime does not import or install the numerical research stack.
Arguments are bounded and errors are non-reflective; the registry authority is
accepted only as an existing private file path, never as credential material.
"""

from __future__ import annotations

import argparse
import os
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Final, NoReturn, TextIO

from quant_platform.service.server import ServerConfig, ServiceStartupError, serve_existing_evidence

_MAX_ARGUMENTS: Final = 16
_MAX_ARGUMENT_BYTES: Final = 4 * 1_024
_REQUIRED_OPTIONS: Final = (
    "--registry-db",
    "--cas-root",
    "--socket-path",
    "--digest-key-file",
)


class ServiceArgumentError(ValueError):
    """Container arguments violate the closed non-reflective service contract."""


class _NonReflectiveParser(argparse.ArgumentParser):
    """Translate argparse failures without echoing attacker-controlled values."""

    def error(self, _message: str) -> NoReturn:
        raise ServiceArgumentError("invalid service arguments")


@dataclass(frozen=True, slots=True)
class ServiceArguments:
    """Validated absolute paths for one private Unix-socket service process."""

    registry_db: Path
    cas_root: Path
    socket_path: Path
    digest_key_file: Path

    def __post_init__(self) -> None:
        for name in ("registry_db", "cas_root", "socket_path", "digest_key_file"):
            value = getattr(self, name)
            if not isinstance(value, Path) or not value.is_absolute():
                raise ServiceArgumentError("service paths must be absolute")


def _bounded_argv(argv: Sequence[str]) -> tuple[str, ...]:
    if not isinstance(argv, Sequence) or isinstance(argv, (str, bytes)):
        raise ServiceArgumentError("service arguments must be a sequence")
    if not 1 <= len(argv) <= _MAX_ARGUMENTS:
        raise ServiceArgumentError("service argument count is outside the supported bound")
    bounded: list[str] = []
    for argument in argv:
        if type(argument) is not str:
            raise ServiceArgumentError("service arguments must be exact strings")
        try:
            encoded = os.fsencode(argument)
        except (UnicodeEncodeError, ValueError):
            raise ServiceArgumentError("service argument encoding is invalid") from None
        if (
            not 1 <= len(encoded) <= _MAX_ARGUMENT_BYTES
            or b"\x00" in encoded
            or any(byte < 0x20 or byte == 0x7F for byte in encoded)
        ):
            raise ServiceArgumentError("service argument bytes violate the bounded contract")
        bounded.append(argument)
    return tuple(bounded)


def parse_service_arguments(argv: Sequence[str]) -> ServiceArguments:
    """Parse the exact container surface without resolving or following paths."""

    bounded = _bounded_argv(argv)
    if len(bounded) != 2 * len(_REQUIRED_OPTIONS) or any(
        bounded.count(option) != 1 for option in _REQUIRED_OPTIONS
    ):
        raise ServiceArgumentError("invalid service arguments")
    parser = _NonReflectiveParser(
        prog="signalattice-service",
        description="Serve verified local evidence over one private Unix socket.",
        allow_abbrev=False,
        add_help=False,
    )
    parser.add_argument("--registry-db", required=True, type=Path)
    parser.add_argument("--cas-root", required=True, type=Path)
    parser.add_argument("--socket-path", required=True, type=Path)
    parser.add_argument("--digest-key-file", required=True, type=Path)
    namespace = parser.parse_args(bounded)
    return ServiceArguments(
        registry_db=namespace.registry_db,
        cas_root=namespace.cas_root,
        socket_path=namespace.socket_path,
        digest_key_file=namespace.digest_key_file,
    )


def main(argv: Sequence[str] | None = None, *, stderr: TextIO | None = None) -> int:
    """Run one local service and return a stable process status.

    ``0`` means Uvicorn stopped normally. Configuration, storage-readiness, and
    argument failures return ``2`` after a bounded public diagnostic. Internal
    exceptions are deliberately not flattened here: the process supervisor
    must observe an unexpected crash rather than a false graceful shutdown.
    """

    error_stream = sys.stderr if stderr is None else stderr
    try:
        arguments = parse_service_arguments(sys.argv[1:] if argv is None else argv)
        serve_existing_evidence(
            arguments.registry_db,
            arguments.cas_root,
            digest_key_file=arguments.digest_key_file,
            config=ServerConfig(socket_path=arguments.socket_path),
        )
    except (ServiceArgumentError, ServiceStartupError) as exc:
        print(f"signalattice-service: {exc}", file=error_stream)
        return 2
    return 0


if __name__ == "__main__":  # pragma: no cover - exercised through ``python -m`` smoke tests.
    raise SystemExit(main())
