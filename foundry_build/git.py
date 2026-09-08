"""Bounded Git plumbing with sanitized failures and no inherited credentials.

The streaming pipe pattern follows AlphaForge's preserved MIT-licensed
scripts/preservation/git.py. New writes are limited to explicit assembly actions.
"""

from __future__ import annotations

import os
import re
import selectors
import subprocess
import time
from pathlib import Path


class AssemblyError(ValueError):
    """Stable, non-sensitive boundary failure code."""


def oid(value: str) -> str:
    """Require an unabbreviated SHA-1 Git object identity."""
    if re.fullmatch(r"[0-9a-f]{40}", value) is None:
        raise AssemblyError("invalid-object-id")
    return value


def git(root: Path, *args: str, seconds: float = 60, ceiling: int = 32 << 20) -> bytes:
    """Run explicit Git arguments with a combined pipe-byte ceiling and deadline.

    No shell, credential variables, user/system Git configuration, replacements,
    lazy fetching, or hooks. The caller owns authorization and destination scope.
    Arguments must be fixed verbs and validated refs/paths, never shell fragments.
    """
    environment = {
        "PATH": os.defpath + ":/opt/homebrew/bin:/usr/local/bin",
        "HOME": str(Path.home()),
        "LC_ALL": "C",
        "GIT_CONFIG_NOSYSTEM": "1",
        "GIT_CONFIG_GLOBAL": os.devnull,
        "GIT_NO_REPLACE_OBJECTS": "1",
        "GIT_TERMINAL_PROMPT": "0",
        "GIT_NO_LAZY_FETCH": "1",
        "GIT_AUTHOR_NAME": "srgangaram-swe",
        "GIT_AUTHOR_EMAIL": "srgangaram-swe@users.noreply.github.com",
        "GIT_COMMITTER_NAME": "srgangaram-swe",
        "GIT_COMMITTER_EMAIL": "srgangaram-swe@users.noreply.github.com",
    }
    try:
        process = subprocess.Popen(
            ["git", "--no-pager", "-c", "core.hooksPath=/dev/null", *args],
            cwd=root,
            env=environment,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
    except OSError as exc:
        raise AssemblyError("git-unavailable") from exc
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
                    raise AssemblyError("git-timeout")
                for key, _ in selector.select(min(remaining, 0.1)):
                    chunk = os.read(key.fd, 65536)
                    if not chunk:
                        selector.unregister(key.fileobj)
                        continue
                    size += len(chunk)
                    if size > ceiling:
                        raise AssemblyError("git-output-limit")
                    if key.data:
                        output.extend(chunk)
            try:
                code = process.wait(timeout=max(0.001, deadline - time.monotonic()))
            except subprocess.TimeoutExpired as exc:
                raise AssemblyError("git-timeout") from exc
            if code:
                raise AssemblyError("git-command-failed")
    finally:
        if process.poll() is None:
            process.kill()
        process.wait()
        if process.stdout is not None:
            process.stdout.close()
        if process.stderr is not None:
            process.stderr.close()
    return bytes(output)


def resolve(root: Path, revision: str) -> str:
    return oid(git(root, "rev-parse", "--verify", revision).decode().strip())
