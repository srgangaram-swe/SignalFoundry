"""One bounded FIFO worker; cancellation and publication are linearized.

No database transaction spans research execution. The condition protects only
queue/lifecycle transitions; workers wait without polling. Fatal scheduler faults
stop new admission and remain observable, rather than abandoning queued work.
"""

from __future__ import annotations

import logging
import threading
from collections import deque
from typing import Protocol

from signal_foundry.boundary import FoundryError
from signal_foundry.contracts import (
    Catalog,
    Comparison,
    Job,
    JobState,
    ResearchEvidence,
    ResearchRequest,
    Validation,
)
from signal_foundry.store import TERMINAL, Store

LOGGER = logging.getLogger(__name__)


class ResearchRunner(Protocol):
    def catalog(self) -> Catalog: ...
    def validate(self, request: ResearchRequest) -> Validation: ...
    def run(
        self, request: ResearchRequest, cancel: threading.Event
    ) -> ResearchEvidence: ...


class Manager:
    """Own the store and single scheduler thread until close completes."""

    def __init__(self, store: Store, runner: ResearchRunner) -> None:
        self.store = store
        self.runner = runner
        self._condition = threading.Condition()
        self._queue: deque[str] = deque()
        self._active: tuple[str, threading.Event] | None = None
        self._closed = False
        self._faulted = False
        self._thread = threading.Thread(
            target=self._loop, name="foundry-research", daemon=True
        )
        self._thread.start()

    def _available(self) -> None:
        if self._closed or self._faulted:
            raise FoundryError(
                "scheduler_unavailable",
                "Research admission is stopped; restart after inspecting local state.",
                503,
            )

    def submit(self, request: ResearchRequest, key: str) -> Job:
        with self._condition:
            self._available()
            existing = self.store.lookup(request, key)
            if existing is not None:
                return existing
        validated = self.runner.validate(request)
        if validated.request_hash != request.digest():
            raise FoundryError(
                "validation_identity",
                "Preflight did not bind the requested configuration.",
                502,
            )
        with self._condition:
            self._available()
            job, created = self.store.submit(request, key)
            if created:
                self._queue.append(job.job_id)
                self._condition.notify()
            return job

    def cancel(self, job_id: str) -> Job:
        """Once cancelled is returned, late worker output cannot publish."""
        with self._condition:
            current = self.store.get(job_id)
            if current.state in TERMINAL:
                return current
            if self._active is not None and self._active[0] == job_id:
                self._active[1].set()
            else:
                self._queue.remove(job_id)
            result = self.store.transition(job_id, JobState.CANCELLED, "cancelled")
            self._condition.notify_all()
            return result

    def compare(self, left: str, right: str) -> Comparison:
        a, b = self.store.evidence(left), self.store.evidence(right)
        # Model and strategy may differ; data, chronology, costs and risk must
        # match. Unfair comparisons remain inspectable but explicitly unranked.
        fields = ("data", "feature_profile", "folds", "costs", "risk", "seed")
        compatible = (
            a.data_identity == b.data_identity
            and a.code_hash == b.code_hash
            and a.source_code_hash == b.source_code_hash
            and a.environment_hash == b.environment_hash
            and all(
                getattr(a.request, field) == getattr(b.request, field)
                for field in fields
            )
        )
        return Comparison(
            compatible=compatible,
            reason=(
                "Matched data, code, environment, seed, chronology, costs and risk;"
                " development evidence only."
                if compatible
                else (
                    "Unmatched comparison policy or provenance; do not rank these"
                    " results as like-for-like."
                )
            ),
            evidence=(a, b),
        )

    def _loop(self) -> None:
        try:
            self._consume()
        except Exception as exc:
            # The sole catch-all is the thread ownership boundary: record only
            # the exception class, stop admission, and preserve DB recovery.
            LOGGER.error("research scheduler stopped (%s)", type(exc).__name__)
            with self._condition:
                self._faulted = True
                if self._active is not None:
                    self._active[1].set()
                try:
                    for job in self.store.list().jobs:
                        if job.state not in TERMINAL:
                            self.store.transition(
                                job.job_id, JobState.FAILED, "scheduler_fault"
                            )
                    self._queue.clear()
                except FoundryError as persistence_error:
                    LOGGER.error(
                        "scheduler recovery requires restart (%s)",
                        persistence_error.code,
                    )
                self._condition.notify_all()

    def _consume(self) -> None:
        while True:
            with self._condition:
                self._condition.wait_for(lambda: self._closed or bool(self._queue))
                if self._closed:
                    return
                identity = self._queue.popleft()
                cancel = threading.Event()
                self._active = identity, cancel
                self.store.transition(identity, JobState.RUNNING)
            try:
                result = self.runner.run(self.store.request(identity), cancel)
            except FoundryError as exc:
                with self._condition:
                    if self.store.get(identity).state not in TERMINAL:
                        self.store.transition(identity, JobState.FAILED, exc.code)
            else:
                with self._condition:
                    if not cancel.is_set():
                        self.store.publish(identity, result)
            finally:
                with self._condition:
                    self._active = None
                    self._condition.notify_all()

    def wait(self, job_id: str, timeout: float = 180.0) -> Job:
        """Bounded CLI/test wait; the HTTP service uses nonblocking status GETs."""
        if not 0 < timeout <= 180:
            raise FoundryError(
                "wait_policy", "Wait duration must be in (0, 180] seconds."
            )
        with self._condition:
            finished = self._condition.wait_for(
                lambda: self._faulted or self.store.get(job_id).state in TERMINAL,
                timeout,
            )
            if self._faulted:
                raise FoundryError(
                    "scheduler_unavailable",
                    "Scheduler stopped; unfinished work fails on restart.",
                    503,
                )
            if not finished:
                raise FoundryError(
                    "wait_timeout",
                    "Job is still pending; its identity remains valid.",
                    504,
                )
            return self.store.get(job_id)

    def close(self) -> None:
        """Cancel owned work, reap the bounded runner, then release persistence."""
        with self._condition:
            if self._closed:
                return
            self._closed = True
            for identity in tuple(self._queue):
                self.cancel(identity)
            if self._active is not None:
                self.cancel(self._active[0])
            self._condition.notify_all()
        self._thread.join(timeout=181)
        if self._thread.is_alive():
            raise FoundryError(
                "shutdown_timeout",
                "The owned worker did not stop; state remains locked.",
                503,
            )
        self.store.close()
