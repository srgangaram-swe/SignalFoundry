"""Single-owner bounded SQLite job/evidence store with atomic state transitions.

Canonical evidence is an immutable, hash-bound SQLite BLOB: state, artifact and
audit publish in one transaction. No partial file can masquerade as a completed
run. A process-lifetime flock excludes another service instance. This is local
durability, not distributed consensus or protection against the owning user.
"""

from __future__ import annotations

import fcntl
import hashlib
import os
import re
import sqlite3
import stat
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path

from pydantic import ValidationError

from signal_foundry.boundary import MAX_EVIDENCE_BYTES, FoundryError, private_directory
from signal_foundry.contracts import (
    AuditEvent,
    AuditTrail,
    Job,
    JobPage,
    JobState,
    ResearchEvidence,
    ResearchRequest,
)

MAX_JOBS = 64
MAX_PENDING = 8
TERMINAL = frozenset({JobState.SUCCEEDED, JobState.FAILED, JobState.CANCELLED})
IDENTITY = re.compile(r"^[0-9a-f]{64}$")
KEY = re.compile(r"^[A-Za-z0-9_-]{16,128}$")


def now() -> str:
    return datetime.now(UTC).isoformat(timespec="milliseconds").replace("+00:00", "Z")


class Store:
    """All public operations are thread-safe; SQL values are bound parameters."""

    def __init__(self, directory: Path) -> None:
        self.directory = private_directory(directory, create=True)
        self._lock = threading.RLock()
        self._descriptor: int | None = None
        self._connection: sqlite3.Connection | None = None
        try:
            self._descriptor = os.open(
                self.directory / ".owner.lock",
                os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW | os.O_NONBLOCK,
                0o600,
            )
            lock_metadata = os.fstat(self._descriptor)
            if (
                not stat.S_ISREG(lock_metadata.st_mode)
                or lock_metadata.st_uid != os.getuid()
                or lock_metadata.st_mode & 0o077
            ):
                raise FoundryError(
                    "unsafe_store",
                    "State ownership lock must be an owner-only regular file.",
                )
            fcntl.flock(self._descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
            path = self.directory / "research.sqlite3"
            if path.is_symlink() or (
                path.exists()
                and (not path.is_file() or path.stat().st_size > 512 << 20)
            ):
                raise FoundryError(
                    "unsafe_store", "The state database violates the local file policy."
                )
            descriptor = os.open(path, os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW, 0o600)
            try:
                metadata = os.fstat(descriptor)
                if metadata.st_uid != os.getuid() or metadata.st_mode & 0o077:
                    raise FoundryError(
                        "unsafe_store", "State database must be owner-only."
                    )
            finally:
                os.close(descriptor)
            self._connection = sqlite3.connect(
                path, timeout=0.5, check_same_thread=False
            )
            self._connection.row_factory = sqlite3.Row
            self._initialize()
            self.recover()
        except (OSError, sqlite3.Error) as exc:
            self.close()
            raise FoundryError(
                "store_unavailable",
                "State is unavailable or owned by another service instance.",
                503,
            ) from exc
        except FoundryError:
            self.close()
            raise

    @property
    def connection(self) -> sqlite3.Connection:
        if self._connection is None:
            raise FoundryError("store_closed", "The research store is closed.", 503)
        return self._connection

    def _initialize(self) -> None:
        db = self.connection
        version = db.execute("PRAGMA user_version").fetchone()[0]
        if version not in {0, 1}:
            raise FoundryError(
                "store_version",
                "Unsupported state schema; do not downgrade this store.",
                503,
            )
        if (
            version == 0
            and db.execute(
                "SELECT count(*) FROM sqlite_master WHERE type='table'"
            ).fetchone()[0]
        ):
            raise FoundryError(
                "store_version", "Refusing an unrecognized existing database.", 503
            )
        db.executescript("""
            PRAGMA foreign_keys=ON;
            PRAGMA journal_mode=DELETE;
            PRAGMA synchronous=FULL;
            PRAGMA max_page_count=131072;
            CREATE TABLE IF NOT EXISTS jobs (
                job_id TEXT PRIMARY KEY, key_hash TEXT UNIQUE NOT NULL,
                request_hash TEXT NOT NULL, request_json BLOB NOT NULL,
                state TEXT NOT NULL CHECK(state IN (
                    'queued','running','succeeded','failed','cancelled')),
                created_at TEXT NOT NULL, updated_at TEXT NOT NULL,
                error_code TEXT, evidence_hash TEXT, evidence BLOB
            );
            CREATE TABLE IF NOT EXISTS audit (
                sequence INTEGER PRIMARY KEY AUTOINCREMENT,
                job_id TEXT NOT NULL REFERENCES jobs(job_id),
                state TEXT NOT NULL, at TEXT NOT NULL, code TEXT
            );
            CREATE INDEX IF NOT EXISTS audit_job ON audit(job_id,sequence);
            PRAGMA user_version=1;
        """)
        if db.execute("SELECT count(*) FROM jobs").fetchone()[0] > MAX_JOBS:
            raise FoundryError(
                "corrupt_store", "Stored jobs exceed the retention budget.", 500
            )

    @contextmanager
    def _read(self) -> Iterator[None]:
        """Serialize access and preserve database causes without leaking SQL."""
        with self._lock:
            try:
                yield
            except sqlite3.Error as exc:
                raise FoundryError(
                    "store_io", "State operation failed; no success is implied.", 503
                ) from exc

    @contextmanager
    def _write(self) -> Iterator[None]:
        with self._read(), self.connection:
            yield

    def _row(self, job_id: str) -> sqlite3.Row:
        if not IDENTITY.fullmatch(job_id):
            raise FoundryError("job_not_found", "No such research job.", 404)
        row: sqlite3.Row | None = self.connection.execute(
            "SELECT"
            " job_id,request_hash,state,created_at,updated_at,error_code,evidence_hash"
            " FROM jobs WHERE job_id=?",
            (job_id,),
        ).fetchone()
        if row is None:
            raise FoundryError("job_not_found", "No such research job.", 404)
        return row

    @staticmethod
    def _job(row: sqlite3.Row) -> Job:
        try:
            return Job(**{**dict(row), "state": JobState(row["state"])})
        except (ValueError, ValidationError) as exc:
            raise FoundryError(
                "corrupt_store", "Stored job metadata failed validation.", 500
            ) from exc

    def _audit(
        self, job_id: str, state: JobState, at: str, code: str | None = None
    ) -> None:
        self.connection.execute(
            "INSERT INTO audit(job_id,state,at,code) VALUES(?,?,?,?)",
            (job_id, state.value, at, code),
        )

    def get(self, job_id: str) -> Job:
        with self._read():
            return self._job(self._row(job_id))

    def list(self) -> JobPage:
        with self._read():
            rows = self.connection.execute(
                "SELECT"
                " job_id,request_hash,state,created_at,updated_at,"
                "error_code,evidence_hash"
                " FROM jobs ORDER BY created_at DESC,job_id LIMIT ?",
                (MAX_JOBS,),
            ).fetchall()
            return JobPage(jobs=tuple(self._job(row) for row in rows))

    def lookup(self, request: ResearchRequest, key: str) -> Job | None:
        """Retries return the original identity without repeating preflight."""
        if not KEY.fullmatch(key):
            raise FoundryError(
                "idempotency_key", "Use a 16–128 character URL-safe idempotency key."
            )
        key_hash = hashlib.sha256(key.encode()).hexdigest()
        with self._read():
            existing = self.connection.execute(
                "SELECT job_id,request_hash FROM jobs WHERE key_hash=?", (key_hash,)
            ).fetchone()
            if existing is not None:
                if existing["request_hash"] != request.digest():
                    raise FoundryError(
                        "idempotency_conflict",
                        "That idempotency key already binds a different request.",
                        409,
                    )
                return self.get(existing["job_id"])
            return None

    def submit(self, request: ResearchRequest, key: str) -> tuple[Job, bool]:
        with self._write():
            existing = self.lookup(request, key)
            if existing is not None:
                return existing, False
            key_hash = hashlib.sha256(key.encode()).hexdigest()
            request_hash = request.digest()
            count, pending = self.connection.execute(
                "SELECT count(*),coalesce(sum(state IN ('queued','running')),0) FROM"
                " jobs"
            ).fetchone()
            if count >= MAX_JOBS or pending >= MAX_PENDING:
                raise FoundryError(
                    "queue_capacity",
                    "The queue or retained-evidence budget is full; existing evidence"
                    " is preserved.",
                    429,
                )
            identity = hashlib.sha256((key_hash + request_hash).encode()).hexdigest()
            timestamp = now()
            self.connection.execute(
                "INSERT INTO"
                " jobs(job_id,key_hash,request_hash,request_json,state,"
                "created_at,updated_at)"
                " VALUES(?,?,?,?,?,?,?)",
                (
                    identity,
                    key_hash,
                    request_hash,
                    request.canonical(),
                    JobState.QUEUED.value,
                    timestamp,
                    timestamp,
                ),
            )
            self._audit(identity, JobState.QUEUED, timestamp)
            return self.get(identity), True

    def request(self, job_id: str) -> ResearchRequest:
        with self._read():
            job = self.get(job_id)
            payload = self.connection.execute(
                "SELECT substr(request_json,1,16385) FROM jobs WHERE job_id=?",
                (job_id,),
            ).fetchone()[0]
            try:
                request = ResearchRequest.model_validate_json(payload)
            except ValidationError as exc:
                raise FoundryError(
                    "corrupt_store", "Stored research request failed validation.", 500
                ) from exc
            if request.digest() != job.request_hash:
                raise FoundryError(
                    "corrupt_store",
                    "Stored request identity does not match its job.",
                    500,
                )
            return request

    def transition(self, job_id: str, state: JobState, code: str | None = None) -> Job:
        with self._write():
            current = self.get(job_id)
            if current.state in TERMINAL:
                if current.state == state:
                    return current
                raise FoundryError(
                    "terminal_job", "A terminal job cannot change state.", 409
                )
            allowed = {JobState.CANCELLED, JobState.FAILED}
            if current.state == JobState.QUEUED:
                allowed.add(JobState.RUNNING)
            if state not in allowed:
                raise FoundryError(
                    "invalid_transition", "This job transition is not permitted.", 409
                )
            timestamp = now()
            self.connection.execute(
                "UPDATE jobs SET state=?,updated_at=?,error_code=? WHERE job_id=?",
                (state.value, timestamp, code, job_id),
            )
            self._audit(job_id, state, timestamp, code)
            return self.get(job_id)

    def publish(self, job_id: str, evidence: ResearchEvidence) -> Job:
        payload = evidence.canonical()
        if len(payload) > MAX_EVIDENCE_BYTES:
            raise FoundryError(
                "evidence_limit", "Evidence exceeds the immutable artifact budget.", 507
            )
        with self._write():
            current = self.get(job_id)
            if (
                current.state != JobState.RUNNING
                or current.request_hash != evidence.request_hash
            ):
                raise FoundryError(
                    "publication_conflict",
                    "Evidence cannot be published for this job state or request.",
                    409,
                )
            timestamp = now()
            self.connection.execute(
                "UPDATE jobs SET state=?,updated_at=?,evidence_hash=?,evidence=? WHERE"
                " job_id=?",
                (
                    JobState.SUCCEEDED.value,
                    timestamp,
                    hashlib.sha256(payload).hexdigest(),
                    payload,
                    job_id,
                ),
            )
            self._audit(job_id, JobState.SUCCEEDED, timestamp)
            return self.get(job_id)

    def evidence(self, job_id: str) -> ResearchEvidence:
        with self._read():
            job = self.get(job_id)
            if job.state != JobState.SUCCEEDED:
                raise FoundryError(
                    "evidence_unavailable",
                    "This job has no completed immutable evidence.",
                    409,
                )
            payload = self.connection.execute(
                "SELECT substr(evidence,1,?) FROM jobs WHERE job_id=?",
                (MAX_EVIDENCE_BYTES + 1, job_id),
            ).fetchone()[0]
            if (
                not isinstance(payload, bytes)
                or len(payload) > MAX_EVIDENCE_BYTES
                or hashlib.sha256(payload).hexdigest() != job.evidence_hash
            ):
                raise FoundryError(
                    "corrupt_evidence",
                    "Stored evidence failed its hash or byte check.",
                    500,
                )
            try:
                result = ResearchEvidence.model_validate_json(payload)
            except ValidationError as exc:
                raise FoundryError(
                    "corrupt_evidence",
                    "Stored evidence failed schema verification.",
                    500,
                ) from exc
            if result.request_hash != job.request_hash:
                raise FoundryError(
                    "corrupt_evidence", "Stored evidence is not bound to its job.", 500
                )
            return result

    def audit(self, job_id: str) -> AuditTrail:
        with self._read():
            self.get(job_id)
            rows = self.connection.execute(
                "SELECT sequence,state,at,code FROM audit WHERE job_id=? ORDER BY"
                " sequence LIMIT 9",
                (job_id,),
            ).fetchall()
            try:
                return AuditTrail(
                    job_id=job_id,
                    events=tuple(
                        AuditEvent(**{**dict(row), "state": JobState(row["state"])})
                        for row in rows
                    ),
                )
            except ValueError as exc:
                raise FoundryError(
                    "corrupt_store", "Stored audit failed validation.", 500
                ) from exc

    def recover(self) -> None:
        """Interrupted work fails closed on restart; never automatically re-train."""
        with self._read():
            rows = self.connection.execute(
                "SELECT job_id FROM jobs WHERE state IN ('queued','running') LIMIT 65"
            ).fetchall()
            if len(rows) > MAX_JOBS:
                raise FoundryError(
                    "corrupt_store", "Stored queue exceeds its cardinality bound.", 500
                )
            for row in rows:
                self.transition(row["job_id"], JobState.FAILED, "interrupted")

    def close(self) -> None:
        with self._lock:
            if self._connection is not None:
                self._connection.close()
                self._connection = None
            if self._descriptor is not None:
                os.close(self._descriptor)
                self._descriptor = None
