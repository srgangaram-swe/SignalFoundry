"""Bounded JSON and local file primitives; no caller-controlled diagnostics."""

from __future__ import annotations

import hashlib
import json
import math
import os
import stat
from pathlib import Path
from typing import Any

MAX_REQUEST_BYTES = 16_384
MAX_EVIDENCE_BYTES = 4 * 1024 * 1024


class FoundryError(Exception):
    """Stable boundary failure; detail never contains an underlying payload/path."""

    def __init__(self, code: str, detail: str, status: int = 422) -> None:
        super().__init__(code)
        self.code = code
        self.detail = detail
        self.status = status


def _pairs(items: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in items:
        if key in result:
            raise ValueError("duplicate key")
        result[key] = value
    return result


def _constant(value: str) -> None:
    raise ValueError("non-finite JSON constant")


def decode(payload: bytes, maximum: int = MAX_REQUEST_BYTES) -> Any:
    """Decode one finite, unambiguous JSON tree within a byte bound."""
    if not 0 < len(payload) <= maximum:
        raise FoundryError(
            "payload_size", "Payload is empty or exceeds its byte limit.", 413
        )
    try:
        result = json.loads(payload, object_pairs_hook=_pairs, parse_constant=_constant)
        pending = [(result, 0)]
        visited = 0
        while pending:
            value, depth = pending.pop()
            visited += 1
            if depth > 32 or visited > 200_000:
                raise ValueError("JSON structural budget")
            if isinstance(value, float) and not math.isfinite(value):
                raise ValueError("non-finite JSON number")
            if isinstance(value, dict):
                pending.extend((child, depth + 1) for child in value.values())
            elif isinstance(value, list):
                pending.extend((child, depth + 1) for child in value)
        return result
    except (ValueError, UnicodeError, RecursionError) as exc:
        raise FoundryError(
            "invalid_json", "Expected bounded, finite JSON without duplicate keys."
        ) from exc


def encode(value: Any, maximum: int = MAX_EVIDENCE_BYTES) -> bytes:
    try:
        payload = json.dumps(
            value,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
        ).encode()
    except (ValueError, TypeError, RecursionError) as exc:
        raise FoundryError(
            "invalid_evidence", "Evidence is not finite JSON.", 500
        ) from exc
    if len(payload) > maximum:
        raise FoundryError(
            "evidence_limit", "Evidence exceeds the publication byte limit.", 507
        )
    return payload


def read_file(path: Path, maximum: int = MAX_EVIDENCE_BYTES) -> bytes:
    """Read a bounded regular final component; trusted non-symlink parents required."""
    try:
        descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
        with os.fdopen(descriptor, "rb") as stream:
            metadata = os.fstat(stream.fileno())
            if not stat.S_ISREG(metadata.st_mode) or metadata.st_size > maximum:
                raise FoundryError(
                    "unsafe_file", "Resource must be a bounded regular file."
                )
            payload = stream.read(maximum + 1)
    except OSError as exc:
        raise FoundryError(
            "resource_unavailable", "The selected local resource cannot be read.", 404
        ) from exc
    if len(payload) > maximum:
        raise FoundryError("unsafe_file", "Resource changed beyond its byte limit.")
    return payload


def private_directory(path: Path, *, create: bool = False) -> Path:
    """Reject symlink components before using a single-owner storage directory."""
    absolute = path.absolute()
    for component in (absolute, *absolute.parents):
        if component.is_symlink():
            raise FoundryError(
                "unsafe_directory", "Storage paths cannot contain symlinks."
            )
    if create:
        try:
            absolute.mkdir(mode=0o700, parents=True, exist_ok=True)
        except OSError as exc:
            raise FoundryError(
                "storage_unavailable", "Cannot create the local storage directory.", 503
            ) from exc
    if not absolute.is_dir():
        raise FoundryError(
            "storage_unavailable", "A configured local directory does not exist.", 503
        )
    if create:
        metadata = absolute.stat()
        if metadata.st_uid != os.getuid() or metadata.st_mode & 0o077:
            raise FoundryError(
                "unsafe_directory", "Private state must be owner-only (mode 0700)."
            )
    return absolute


def code_identity(root: Path) -> str:
    """Hash bounded control-plane source bytes in stable path order, O(source bytes)."""
    digest = hashlib.sha256()
    paths = sorted((root / "signal_foundry").rglob("*.py"))
    if not 1 <= len(paths) <= 128:
        raise FoundryError(
            "code_identity", "Control-plane source inventory is invalid.", 500
        )
    for path in paths:
        digest.update(
            path.relative_to(root).as_posix().encode()
            + b"\0"
            + read_file(path, 128 * 1024)
        )
    return digest.hexdigest()


def package_identity(root: Path, source: str) -> str:
    """Bind the imported Python/native implementation and its exact lock bytes."""
    if source not in {"alphaforge", "signalattice"}:
        raise FoundryError("unknown_package", "Unregistered package identity.")
    package = root / "packages" / source
    code = package / ("alphaforge" if source == "alphaforge" else "src/quant_platform")
    paths = (
        sorted(code.rglob("*.py")) + sorted(code.glob("*.so")) + [package / "uv.lock"]
    )
    if not 1 <= len(paths) <= 512:
        raise FoundryError("code_identity", "Package source inventory is invalid.", 500)
    digest = hashlib.sha256()
    for path in paths:
        digest.update(path.relative_to(package).as_posix().encode() + b"\0")
        digest.update(read_file(path, 8 * 1024 * 1024))
    return digest.hexdigest()
