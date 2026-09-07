"""Bounded read-only Git object access, independent of user Git configuration."""

from __future__ import annotations

import os
import re
import selectors
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path


class PreservationError(ValueError):
    """Sanitized boundary failure; messages contain codes, never source payloads."""


@dataclass(frozen=True)
class Limits:
    """Hard per-command and whole-inventory limits; bytes are uncompressed."""

    command_seconds: float = 30.0
    total_seconds: float = 900.0
    output_bytes: int = 32 * 1024 * 1024
    blob_bytes: int = 8 * 1024 * 1024
    total_blob_bytes: int = 256 * 1024 * 1024
    objects: int = 100_000
    refs: int = 2_000
    paths: int = 20_000
    path_records: int = 200_000
    interface_records: int = 200_000
    ast_nodes: int = 100_000

    def __post_init__(self) -> None:
        for value in self.__dict__.values():
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise PreservationError("invalid-limit")
            if not 0 < value < 1_000_000_000:
                raise PreservationError("invalid-limit")


def bounded_run(command: list[str], cwd: Path, seconds: float, ceiling: int) -> bytes:
    """Run trusted arguments without shell/config/credentials; bound output and time.

    stderr counts against the same ceiling but is never exposed on failure.
    No source checkout, source execution, hooks, network, or environment secrets.
    """
    environment = {
        "PATH": os.defpath + ":/opt/homebrew/bin:/usr/local/bin",
        "LC_ALL": "C",
        "GIT_CONFIG_NOSYSTEM": "1",
        "GIT_CONFIG_GLOBAL": os.devnull,
        "GIT_NO_REPLACE_OBJECTS": "1",
        "GIT_TERMINAL_PROMPT": "0",
        "GIT_OPTIONAL_LOCKS": "0",
        "GIT_NO_LAZY_FETCH": "1",
        "HOME": str(Path.home()),
    }
    try:
        process = subprocess.Popen(
            command,
            cwd=cwd,
            env=environment,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
    except OSError as exc:
        raise PreservationError("process-unavailable") from exc
    output = bytearray()
    size = 0
    deadline = time.monotonic() + seconds
    try:
        with selectors.DefaultSelector() as selector:
            assert process.stdout is not None and process.stderr is not None
            selector.register(process.stdout, selectors.EVENT_READ, True)
            selector.register(process.stderr, selectors.EVENT_READ, False)
            while selector.get_map():
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise PreservationError("command-timeout")
                for key, _ in selector.select(min(remaining, 0.1)):
                    chunk = os.read(key.fd, 65536)
                    if not chunk:
                        selector.unregister(key.fileobj)
                        continue
                    size += len(chunk)
                    if size > ceiling:
                        raise PreservationError("command-output-limit")
                    if key.data:
                        output.extend(chunk)
            try:
                code = process.wait(timeout=max(0.001, deadline - time.monotonic()))
            except subprocess.TimeoutExpired as exc:
                raise PreservationError("command-timeout") from exc
            if code:
                raise PreservationError("command-failed")
    finally:
        if process.poll() is None:
            process.kill()
        process.wait()
        if process.stdout is not None:
            process.stdout.close()
        if process.stderr is not None:
            process.stderr.close()
    return bytes(output)


def oid(value: str) -> str:
    """Accept only full SHA-1 object names (source repositories use SHA-1)."""
    if re.fullmatch(r"[0-9a-f]{40}", value) is None:
        raise PreservationError("invalid-object-id")
    return value


DEFAULT_LIMITS = Limits()


class Git:
    """Read committed objects from a complete repository; never mutate its refs."""

    def __init__(self, root: Path, limits: Limits = DEFAULT_LIMITS) -> None:
        self.root = root.resolve()
        self.limits = limits
        self.deadline = time.monotonic() + limits.total_seconds
        if self.run("rev-parse", "--is-shallow-repository").strip() != b"false":
            raise PreservationError("shallow-repository")
        if self.run("rev-parse", "--show-object-format").strip() != b"sha1":
            raise PreservationError("unsupported-object-format")
        for name in ("info/grafts", "objects/info/alternates", "objects/info/http-alternates"):
            location = self.run("rev-parse", "--git-path", name).decode().strip()
            path = self.root / location
            if path.exists() or path.is_symlink():
                raise PreservationError("external-or-rewritten-history")
        if self.run("for-each-ref", "refs/replace").strip():
            raise PreservationError("replacement-history")
        configuration = self.run("config", "--local", "--list").lower()
        if b".promisor=" in configuration or b"extensions.partialclone=" in configuration:
            raise PreservationError("partial-repository")

    def run(self, *arguments: str, ceiling: int | None = None) -> bytes:
        """Internal plumbing interface; callers supply fixed verbs and checked IDs."""
        remaining = self.deadline - time.monotonic()
        if remaining <= 0:
            raise PreservationError("inventory-timeout")
        return bounded_run(
            ["git", "--no-pager", "-c", "core.hooksPath=/dev/null", *arguments],
            self.root,
            min(remaining, self.limits.command_seconds),
            self.limits.output_bytes if ceiling is None else ceiling,
        )

    def refs(self) -> dict[str, str]:
        """Snapshot every local mirror ref, including advertised PR refs."""
        result = {}
        for line in (
            self.run("for-each-ref", "--format=%(refname) %(objectname)").decode().splitlines()
        ):
            name, value = line.split(" ")
            if not name.startswith("refs/") or any(ord(c) < 32 for c in name):
                raise PreservationError("invalid-ref")
            result[name] = oid(value)
        if not result or len(result) > self.limits.refs:
            raise PreservationError("ref-count-limit")
        return dict(sorted(result.items()))
