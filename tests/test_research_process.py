"""Real POSIX pipe/process lifecycle and deterministic resource-fault tests."""

from __future__ import annotations

import os
import subprocess
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import psutil
import pytest

import signal_foundry.process as process_module
from signal_foundry.boundary import FoundryError
from signal_foundry.process import execute


def run(code: str, tmp_path: Path, **options: Any) -> bytes:
    return execute(
        [sys.executable, "-I", "-c", code],
        cwd=tmp_path,
        environment={"PATH": "/usr/bin:/bin"},
        payload=options.pop("payload", b"hello"),
        cancel=options.pop("cancel", threading.Event()),
        timeout=options.pop("timeout", 3),
        **options,
    )


def test_success_is_bounded_and_does_not_inherit_secrets(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("NASDAQ_DATA_LINK_API_KEY", "MARKER_SECRET")
    result = run(
        "import os,sys; assert 'NASDAQ_DATA_LINK_API_KEY' not in os.environ;"
        " sys.stdout.buffer.write(sys.stdin.buffer.read())",
        tmp_path,
    )
    assert result == b"hello"
    assert (
        run("import sys; sys.stdout.write('empty')", tmp_path, payload=b"") == b"empty"
    )


@pytest.mark.parametrize(
    "code,options,error",
    [
        ("import sys; sys.exit(2)", {}, "worker_failed"),
        (
            "import sys; sys.stdout.write('x'*10000)",
            {"maximum_output": 100},
            "worker_output_limit",
        ),
        (
            "import sys; sys.stderr.write('x'*10000)",
            {"maximum_output": 100},
            "worker_output_limit",
        ),
        ("import time; time.sleep(10)", {"timeout": 0.1}, "worker_timeout"),
        ("pass", {"timeout": 0}, "execution_policy"),
        ("pass", {"payload": b"x" * 16385}, "execution_policy"),
        ("pass", {"maximum_rss": 1}, "execution_policy"),
    ],
)
def test_failure_paths_reap_children(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    code: str,
    options: dict[str, Any],
    error: str,
) -> None:
    spawned: list[subprocess.Popen[bytes]] = []
    original = subprocess.Popen

    def record(*args: Any, **kwargs: Any):
        child = original(*args, **kwargs)
        spawned.append(child)
        return child

    monkeypatch.setattr(process_module.subprocess, "Popen", record)
    with pytest.raises(FoundryError, match=error):
        run(code, tmp_path, **options)
    for child in spawned:
        assert child.poll() is not None
        assert all(
            stream.closed for stream in (child.stdin, child.stdout, child.stderr)
        )


def test_cancellation_before_and_after_spawn(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cancel = threading.Event()
    cancel.set()
    with pytest.raises(FoundryError, match="cancelled"):
        run("pass", tmp_path, cancel=cancel)
    cancel.clear()
    spawned = threading.Event()
    original = subprocess.Popen

    def record(*args: Any, **kwargs: Any):
        child = original(*args, **kwargs)
        spawned.set()
        return child

    monkeypatch.setattr(process_module.subprocess, "Popen", record)
    with ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(
            run, "import time; time.sleep(10)", tmp_path, cancel=cancel
        )
        assert spawned.wait(2)
        cancel.set()
        with pytest.raises(FoundryError, match="cancelled"):
            future.result(timeout=3)


@pytest.mark.parametrize(
    "fault,code",
    [
        ("rss", "worker_memory_limit"),
        ("children", "worker_process_limit"),
        ("permission", "worker_monitor"),
    ],
)
def test_monitor_faults_are_fail_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, fault: str, code: str
) -> None:
    def monitor(pid: int):
        if fault == "permission":
            raise psutil.AccessDenied(pid)
        return SimpleNamespace(
            children=lambda **_: [None] * 17 if fault == "children" else [],
            memory_info=lambda: SimpleNamespace(rss=3 << 30),
        )

    monkeypatch.setattr(process_module.psutil, "Process", monitor)
    with pytest.raises(FoundryError, match=code):
        run("import time; time.sleep(10)", tmp_path)


def test_missing_executable_preserves_cause(tmp_path: Path) -> None:
    with pytest.raises(FoundryError, match="worker_unavailable") as error:
        execute(
            [str(tmp_path / "missing")],
            cwd=tmp_path,
            environment={},
            payload=b"{}",
            cancel=threading.Event(),
            timeout=1,
        )
    assert isinstance(error.value.__cause__, OSError)


def test_pipe_io_fault_reaps_owned_process(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    original = os.set_blocking

    def fail(fd: int, blocking: bool) -> None:
        if not blocking:
            raise OSError("injected private IO message")
        original(fd, blocking)

    monkeypatch.setattr(process_module.os, "set_blocking", fail)
    with pytest.raises(FoundryError, match="worker_io") as error:
        run("pass", tmp_path)
    assert "private" not in error.value.detail
