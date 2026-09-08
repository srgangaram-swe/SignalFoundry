"""Internal bounded file snapshots and strict structured-document parsing.

These helpers sit on evidence/provenance trust boundaries.  A successful read
returns immutable bytes from one already-open regular-file descriptor; callers
must parse those bytes rather than reopening the pathname.
"""

from __future__ import annotations

import errno
import hashlib
import json
import os
import stat
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml
from yaml.events import AliasEvent
from yaml.nodes import MappingNode

READ_CHUNK_BYTES = 1024 * 1024


class BoundedIOError(ValueError):
    """Raised when a bounded file or document boundary fails closed."""


@dataclass(frozen=True)
class RegularFileSnapshot:
    """Immutable bytes and identity metadata from one regular-file descriptor."""

    path: Path
    data: bytes
    sha256: str

    @property
    def size(self) -> int:
        """Return the exact snapshot byte count."""

        return len(self.data)


def _metadata_identity(metadata: os.stat_result) -> tuple[int, int, int, int, int, int]:
    return (
        metadata.st_dev,
        metadata.st_ino,
        stat.S_IFMT(metadata.st_mode),
        metadata.st_size,
        metadata.st_mtime_ns,
        metadata.st_ctime_ns,
    )


def _reject_symlink_components(root: Path, path: Path) -> None:
    try:
        relative = path.relative_to(root)
    except ValueError as exc:
        raise BoundedIOError("bounded path escapes its declared root") from exc
    current = root
    if root.is_symlink():
        raise BoundedIOError("bounded path root must not be a symlink")
    for component in relative.parts:
        current = current / component
        try:
            metadata = os.stat(current, follow_symlinks=False)
        except OSError as exc:
            raise BoundedIOError(f"unable to inspect bounded path component: {current}") from exc
        if stat.S_ISLNK(metadata.st_mode):
            raise BoundedIOError(f"bounded path traverses a symlink: {current}")


def read_regular_file_snapshot(
    path: str | Path,
    *,
    max_bytes: int,
    root: str | Path | None = None,
) -> RegularFileSnapshot:
    """Read one non-empty regular file without accepting a symlink swap.

    The function opens the file once, reads at most ``max_bytes + 1`` bytes,
    compares descriptor metadata before and after the read, and confirms the
    final pathname still names that descriptor's inode.  When ``root`` is
    supplied, every lexical component below that root must remain a
    non-symlink both before and after the read.
    """

    if isinstance(max_bytes, bool) or not isinstance(max_bytes, int) or max_bytes < 1:
        raise BoundedIOError("max_bytes must be a positive integer")
    lexical = Path(os.path.abspath(path))
    root_path: Path | None = None
    if root is not None:
        root_path = Path(os.path.abspath(root))
        _reject_symlink_components(root_path, lexical)

    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_BINARY", 0)
    flags |= getattr(os, "O_NOFOLLOW", 0)
    descriptor = -1
    try:
        descriptor = os.open(lexical, flags)
        initial = os.fstat(descriptor)
        if not stat.S_ISREG(initial.st_mode):
            raise BoundedIOError(f"bounded path must be a regular file: {lexical}")
        if not 0 < initial.st_size <= max_bytes:
            raise BoundedIOError(f"bounded file bytes must be in [1, {max_bytes}]: {lexical}")

        chunks: list[bytes] = []
        total = 0
        while True:
            chunk = os.read(descriptor, min(READ_CHUNK_BYTES, max_bytes - total + 1))
            if not chunk:
                break
            total += len(chunk)
            if total > max_bytes:
                raise BoundedIOError(f"bounded file exceeds {max_bytes} bytes: {lexical}")
            chunks.append(chunk)

        final = os.fstat(descriptor)
        if _metadata_identity(initial) != _metadata_identity(final):
            raise BoundedIOError(f"bounded file changed during read: {lexical}")
        if total != initial.st_size:
            raise BoundedIOError(f"bounded file length changed during read: {lexical}")
        pathname = os.stat(lexical, follow_symlinks=False)
        if (
            not stat.S_ISREG(pathname.st_mode)
            or pathname.st_dev != final.st_dev
            or pathname.st_ino != final.st_ino
        ):
            raise BoundedIOError(f"bounded file pathname changed during read: {lexical}")
        if root_path is not None:
            _reject_symlink_components(root_path, lexical)
    except BoundedIOError:
        raise
    except OSError as exc:
        if exc.errno == errno.ELOOP:
            raise BoundedIOError(f"bounded path must not be a symlink: {lexical}") from exc
        raise BoundedIOError(f"unable to read bounded regular file: {lexical}") from exc
    finally:
        if descriptor >= 0:
            os.close(descriptor)

    data = b"".join(chunks)
    return RegularFileSnapshot(
        path=lexical,
        data=data,
        sha256=hashlib.sha256(data).hexdigest(),
    )


def _validate_structure(
    document: Any,
    *,
    maximum_depth: int,
    maximum_nodes: int,
) -> None:
    stack: list[tuple[Any, int]] = [(document, 1)]
    observed = 0
    while stack:
        value, depth = stack.pop()
        observed += 1
        if observed > maximum_nodes:
            raise BoundedIOError(f"structured document exceeds {maximum_nodes} nodes")
        if depth > maximum_depth:
            raise BoundedIOError(f"structured document exceeds depth {maximum_depth}")
        if isinstance(value, dict):
            stack.extend((item, depth + 1) for item in value.values())
        elif isinstance(value, list):
            stack.extend((item, depth + 1) for item in value)


def _strict_json_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    document: dict[str, Any] = {}
    for key, value in pairs:
        if key in document:
            raise BoundedIOError(f"JSON object contains duplicate key {key!r}")
        document[key] = value
    return document


def parse_strict_json(
    data: bytes,
    *,
    maximum_depth: int,
    maximum_nodes: int,
) -> Any:
    """Parse UTF-8 JSON with duplicate, constant, depth, and node rejection."""

    def reject_constant(value: str) -> None:
        raise BoundedIOError(f"JSON contains unsupported constant {value!r}")

    try:
        document = json.loads(
            data.decode("utf-8"),
            object_pairs_hook=_strict_json_object,
            parse_constant=reject_constant,
        )
    except BoundedIOError:
        raise
    except (UnicodeError, json.JSONDecodeError, RecursionError) as exc:
        raise BoundedIOError("document must be valid bounded UTF-8 JSON") from exc
    _validate_structure(
        document,
        maximum_depth=maximum_depth,
        maximum_nodes=maximum_nodes,
    )
    return document


class _StrictBoundedSafeLoader(yaml.SafeLoader):
    """SafeLoader variant rejecting aliases, duplicates, and deep node graphs."""

    maximum_depth = 64
    maximum_nodes = 100_000

    def __init__(self, stream: str) -> None:
        super().__init__(stream)
        self._compose_depth = 0
        self._composed_nodes = 0

    def compose_node(self, parent: Any, index: Any) -> Any:
        if self.check_event(AliasEvent):
            raise BoundedIOError("YAML aliases are not permitted")
        self._compose_depth += 1
        try:
            if self._compose_depth > self.maximum_depth:
                raise BoundedIOError(f"structured document exceeds depth {self.maximum_depth}")
            node = super().compose_node(parent, index)
            self._composed_nodes += 1
            if self._composed_nodes > self.maximum_nodes:
                raise BoundedIOError(f"structured document exceeds {self.maximum_nodes} nodes")
            return node
        finally:
            self._compose_depth -= 1

    def construct_mapping(self, node: MappingNode, deep: bool = False) -> dict[str, Any]:
        if not isinstance(node, MappingNode):
            raise BoundedIOError("YAML mapping node is invalid")
        mapping: dict[str, Any] = {}
        for key_node, value_node in node.value:
            key = self.construct_object(key_node, deep=deep)
            if not isinstance(key, str):
                raise BoundedIOError("YAML mapping keys must be strings")
            if key in mapping:
                raise BoundedIOError(f"YAML mapping contains duplicate key {key!r}")
            mapping[key] = self.construct_object(value_node, deep=deep)
        return mapping


def parse_strict_yaml(
    data: bytes,
    *,
    maximum_depth: int,
    maximum_nodes: int,
) -> Any:
    """Parse UTF-8 YAML without aliases, duplicate keys, or unbounded graphs."""

    loader_type = type(
        "_ConfiguredStrictBoundedSafeLoader",
        (_StrictBoundedSafeLoader,),
        {
            "maximum_depth": maximum_depth,
            "maximum_nodes": maximum_nodes,
        },
    )
    try:
        document = yaml.load(data.decode("utf-8"), Loader=loader_type)
    except BoundedIOError:
        raise
    except (UnicodeError, yaml.YAMLError, RecursionError) as exc:
        raise BoundedIOError("document must be valid bounded UTF-8 YAML") from exc
    _validate_structure(
        document,
        maximum_depth=maximum_depth,
        maximum_nodes=maximum_nodes,
    )
    return document


def bounded_diagnostic(prefix: str, detail: object, *, maximum_chars: int) -> str:
    """Return one deterministic diagnostic bounded including its prefix."""

    if maximum_chars < 16:
        raise ValueError("maximum_chars must be at least 16")
    message = f"{prefix}{detail}"
    if len(message) <= maximum_chars:
        return message
    marker = "...[truncated]"
    return message[: maximum_chars - len(marker)] + marker
