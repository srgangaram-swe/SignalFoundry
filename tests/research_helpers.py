"""Explicit deterministic doubles for scheduling tests, never benchmark evidence."""

from __future__ import annotations

import threading

from signal_foundry.boundary import FoundryError
from signal_foundry.contracts import (
    Catalog,
    Column,
    EvidenceTable,
    ResearchEvidence,
    ResearchRequest,
    SeedAssignment,
    Validation,
)


def evidence(request: ResearchRequest | None = None) -> ResearchEvidence:
    request = request or ResearchRequest()
    return ResearchEvidence(
        request=request,
        request_hash=request.digest(),
        data_identity="synthetic:test",
        code_hash="a" * 64,
        source_code_hash="b" * 64,
        environment_hash="c" * 64,
        seed_map=(SeedAssignment(name="test", value=request.seed),),
        tables=(
            EvidenceTable(
                name="test",
                description="Test double only.",
                columns=(Column(name="value", unit="test units"),),
                rows=((1.0,),),
                total_rows=1,
            ),
        ),
        limitations=("Test double, not scientific evidence.",),
    )


class FakeRunner:
    def __init__(self) -> None:
        self.started = threading.Event()
        self.release = threading.Event()
        self.release.set()
        self.calls = 0
        self.failure: Exception | None = None

    def catalog(self) -> Catalog:
        return Catalog(
            models=(),
            strategies=(),
            baselines=(),
            datasets=(),
            limitations=("Test only.",),
        )

    def validate(self, request: ResearchRequest) -> Validation:
        return Validation(
            request_hash=request.digest(),
            data_identity="synthetic:test",
            observations=4500,
            symbols=9,
            sessions=500,
            limitations=("Test only.",),
        )

    def run(
        self, request: ResearchRequest, cancel: threading.Event
    ) -> ResearchEvidence:
        self.calls += 1
        self.started.set()
        if not self.release.wait(10):
            raise FoundryError("test_deadline", "Test synchronization timed out.")
        if self.failure is not None:
            raise self.failure
        return evidence(request)
