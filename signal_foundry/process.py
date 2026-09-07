"""POSIX subprocess isolation with bounded pipes, cancellation and process cleanup.

Only Runner constructs executable commands. execute is the mechanism, accepting a
trusted argument vector; it is never exposed through HTTP. Polling blocks in the
selector and has a 50 ms cancellation observation bound, excluding OS scheduling.
"""

from __future__ import annotations

import os
import selectors
import signal
import subprocess
import threading
import time
from collections.abc import Mapping, Sequence
from pathlib import Path

import psutil

from signal_foundry.boundary import MAX_EVIDENCE_BYTES, MAX_REQUEST_BYTES, FoundryError


def execute(
    command: Sequence[str],
    *,
    cwd: Path,
    environment: Mapping[str, str],
    payload: bytes,
    cancel: threading.Event,
    timeout: float,
    maximum_output: int = MAX_EVIDENCE_BYTES,
    maximum_rss: int = 2 * 1024**3,
) -> bytes:
    """Return bounded stdout or raise a sanitized, cause-preserving failure.

    Combined stdout/stderr is capped. Every exit path reaps the process and closes
    all pipes; cancellation and timeout kill its whole newly owned process group.
    No shell or preexec callback runs. The child sets its own POSIX resource limits.
    """
    if (
        os.name != "posix"
        or not 0 < timeout <= 180
        or len(payload) > MAX_REQUEST_BYTES
        or not 64 << 20 <= maximum_rss <= 8 << 30
    ):
        raise FoundryError("execution_policy", "Unsupported process policy.")
    if cancel.is_set():
        raise FoundryError("cancelled", "Research was cancelled.", 409)
    try:
        process = subprocess.Popen(
            list(command),
            cwd=cwd,
            env=dict(environment),
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            start_new_session=True,
        )
    except OSError as exc:
        raise FoundryError(
            "worker_unavailable",
            "Install the documented locked package environment.",
            503,
        ) from exc
    assert (
        process.stdin is not None
        and process.stdout is not None
        and process.stderr is not None
    )
    pipes = (process.stdin, process.stdout, process.stderr)
    output = bytearray()
    total = 0
    written = 0
    deadline = time.monotonic() + timeout
    try:
        with selectors.DefaultSelector() as selector:
            for stream in pipes:
                os.set_blocking(stream.fileno(), False)
            selector.register(process.stdin, selectors.EVENT_WRITE)
            selector.register(process.stdout, selectors.EVENT_READ)
            selector.register(process.stderr, selectors.EVENT_READ)
            while selector.get_map():
                if cancel.is_set():
                    raise FoundryError("cancelled", "Research was cancelled.", 409)
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise FoundryError(
                        "worker_timeout", "Research exceeded its wall-time budget.", 504
                    )
                try:
                    monitored = psutil.Process(process.pid)
                    children = monitored.children(recursive=True)
                    if len(children) > 16:
                        raise FoundryError(
                            "worker_process_limit",
                            "Worker descendant count exceeded its budget.",
                            507,
                        )
                    resident = monitored.memory_info().rss
                    for child in children:
                        try:
                            resident += child.memory_info().rss
                        except psutil.NoSuchProcess:
                            continue
                    if resident > maximum_rss:
                        raise FoundryError(
                            "worker_memory_limit",
                            "Worker exceeded its sampled resident-memory budget.",
                            507,
                        )
                except psutil.NoSuchProcess:
                    # An exited process still has bounded pipe output to drain.
                    pass
                except psutil.AccessDenied as exc:
                    raise FoundryError(
                        "worker_monitor",
                        "Worker resource monitoring became unavailable.",
                        503,
                    ) from exc
                for key, _ in selector.select(min(0.05, remaining)):
                    ready_stream = key.fileobj
                    if ready_stream is process.stdin:
                        try:
                            written += os.write(
                                process.stdin.fileno(), payload[written:]
                            )
                        except BrokenPipeError:
                            selector.unregister(process.stdin)
                            process.stdin.close()
                            continue
                        if written == len(payload):
                            selector.unregister(process.stdin)
                            process.stdin.close()
                    else:
                        chunk = os.read(key.fd, 65_536)
                        if not chunk:
                            selector.unregister(ready_stream)
                            continue
                        total += len(chunk)
                        if total > maximum_output:
                            raise FoundryError(
                                "worker_output_limit",
                                "Worker output exceeded its byte budget.",
                                507,
                            )
                        if ready_stream is process.stdout:
                            output.extend(chunk)
            remaining = deadline - time.monotonic()
            try:
                process.wait(timeout=max(0.001, remaining))
            except subprocess.TimeoutExpired as exc:
                raise FoundryError(
                    "worker_timeout", "Worker did not exit within its budget.", 504
                ) from exc
        if cancel.is_set():
            raise FoundryError("cancelled", "Research was cancelled.", 409)
        if process.returncode != 0:
            raise FoundryError(
                "worker_failed",
                "Worker failed; no partial evidence was published.",
                422,
            )
        return bytes(output)
    except OSError as exc:
        raise FoundryError("worker_io", "Worker communication failed.", 503) from exc
    finally:
        # The group is exclusively created above. Also contain descendants left
        # behind after their parent exited; a vanished group is an expected race.
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        process.wait()
        for stream in pipes:
            stream.close()
