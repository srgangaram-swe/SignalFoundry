"""Fixed paper worker dispatch; parent owns deadlines and kills descendants.

The browser supplies an operation and a universe symbol, never a filesystem
path, endpoint, credential, order quantity or qualification verdict.
"""

from __future__ import annotations

import os
import sys
import threading
from pathlib import Path
from typing import Literal

from pydantic import Field

from signal_foundry.boundary import (
    FoundryError,
    decode,
    encode,
    private_directory,
    read_file,
)
from signal_foundry.process import execute
from signal_foundry.trading.alpaca import FORBIDDEN
from signal_foundry.trading.engine import Engine
from signal_foundry.trading.models import PaperConfig, PaperStatus, Record, Symbol
from signal_foundry.trading.store import Journal


class Action(Record):
    operation: Literal[
        "initialize",
        "probe",
        "acquire",
        "research",
        "qualify",
        "start",
        "cycle",
        "reconcile",
        "record-session",
        "campaign",
        "stop",
        "cancel",
    ]
    symbol: Symbol | None = None


class PaperResult(Record):
    status: PaperStatus
    artifact: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    account_digest: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")


class WorkerError(Record):
    code: str = Field(pattern=r"^[a-z_]{1,64}$")
    detail: str = Field(min_length=1, max_length=512)
    status: int = Field(ge=400, le=599, strict=True)


def configuration(path: Path) -> PaperConfig:
    private_directory(path.parent)
    return PaperConfig.model_validate_json(read_file(path, 16_384))


class PaperService:
    def __init__(self, root: Path, state: Path, config: Path) -> None:
        self.root = private_directory(root)
        self.state = private_directory(state, create=True)
        self.config_path = config.absolute()
        self.config = configuration(self.config_path)
        self._capacity = threading.BoundedSemaphore(1)

    def status(self) -> PaperStatus:
        journal = Journal(self.state)
        try:
            return Engine(self.root, journal, self.config).status()
        finally:
            journal.close()

    def action(self, action: Action) -> PaperResult:
        # Emergency stop never waits for a worker slot or network call.
        if action.operation == "stop":
            journal = Journal(self.state)
            try:
                journal.stop()
                return PaperResult(
                    status=Engine(self.root, journal, self.config).status()
                )
            finally:
                journal.close()
        if any(key in os.environ for key in FORBIDDEN):
            raise FoundryError(
                "credential_policy", "Remove broker credentials from the environment."
            )
        if not self._capacity.acquire(blocking=False):
            raise FoundryError(
                "paper_busy", "One paper operation is already active.", 409
            )
        try:
            result = execute(
                [
                    sys.executable,
                    "-I",
                    str(self.root / "signal_foundry/trading/entry.py"),
                ],
                cwd=self.root,
                environment={
                    "PATH": "/usr/bin:/bin",
                    "PYTHONNOUSERSITE": "1",
                    "OMP_NUM_THREADS": "1",
                    "OPENBLAS_NUM_THREADS": "1",
                },
                payload=encode(
                    {
                        "state": str(self.state),
                        "config": str(self.config_path),
                        "action": action.wire(),
                    }
                ),
                cancel=threading.Event(),
                timeout=30,
                maximum_output=65_536,
            )
            value = decode(result, 65_536)
            if not isinstance(value, dict) or set(value) not in ({"error"}, {"result"}):
                raise FoundryError("paper_envelope", "Invalid paper worker response.")
            if "error" in value:
                error = WorkerError.model_validate(value["error"])
                raise FoundryError(error.code, error.detail, error.status)
            return PaperResult.model_validate(value["result"])
        finally:
            self._capacity.release()


def unavailable() -> PaperStatus:
    return PaperStatus(
        configured=False,
        enabled=False,
        stopped=False,
        config_identity="",
        state="not_configured",
        orders=0,
        events=0,
        symbols=(),
        feed="none",
        maximum_order_notional="0",
        maximum_position_notional="0",
        maximum_session_loss="0",
        blockers=(
            "Start with an explicit local paper configuration.",
            "Live capability is absent.",
        ),
        last_action="none",
        paper_sessions=0,
    )
