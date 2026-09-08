"""Private SQLite journal: durable intent before I/O and append-only hash evidence.

One process owns ordinary operations through a nonblocking POSIX lock. Emergency
stop uses a separate short SQLite transaction, so it need not wait for network
I/O. A stop cannot recall an already dispatched order; cancellation is explicit.
"""

from __future__ import annotations

import fcntl
import hashlib
import os
import sqlite3
import tempfile
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from signal_foundry.boundary import (
    FoundryError,
    decode,
    encode,
    private_directory,
    read_file,
)

MAX_EVENTS = 100_000
ZERO = "0" * 64


class Journal:
    """Bounded state cache plus independently verified event chain.

    The hashes detect accidental corruption, not a malicious local owner capable
    of rewriting both data and hashes. No SQL transaction spans network I/O.
    """

    def __init__(self, root: Path) -> None:
        self.root = private_directory(root, create=True)
        path = self.root / "paper.sqlite"
        new = not path.exists()
        if new:
            try:
                descriptor = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
                os.close(descriptor)
            except FileExistsError:
                new = False
        if path.is_symlink() or path.stat().st_size > 64 << 20:
            raise FoundryError("paper_store", "Unsafe or oversized paper state.")
        if path.stat().st_mode & 0o077:
            raise FoundryError("paper_store", "Paper state must be owner-only.")
        self.connection = sqlite3.connect(path, timeout=2, isolation_level=None)
        try:
            self.connection.execute("PRAGMA synchronous=FULL")
            self.connection.execute("PRAGMA journal_mode=DELETE")
            if new:
                self.connection.executescript(
                    "CREATE TABLE state (key TEXT PRIMARY KEY, value BLOB NOT NULL,"
                    " digest TEXT NOT NULL);"
                    "CREATE TABLE events (seq INTEGER PRIMARY KEY, at TEXT NOT NULL,"
                    " kind TEXT NOT NULL, payload BLOB NOT NULL,"
                    " previous TEXT NOT NULL,"
                    " digest TEXT NOT NULL);"
                    "CREATE TRIGGER no_update BEFORE UPDATE ON events BEGIN"
                    " SELECT RAISE(ABORT, 'append-only'); END;"
                    "CREATE TRIGGER no_delete BEFORE DELETE ON events BEGIN"
                    " SELECT RAISE(ABORT, 'append-only'); END;"
                    "PRAGMA user_version=1;"
                )
            if self.connection.execute("PRAGMA user_version").fetchone()[0] != 1:
                raise FoundryError("paper_store", "Unknown paper journal version.")
            if self.connection.execute("PRAGMA quick_check").fetchone()[0] != "ok":
                raise FoundryError("paper_store", "Paper journal integrity failed.")
            self.verify()
        except (sqlite3.Error, FoundryError) as exc:
            self.connection.close()
            raise FoundryError(
                "paper_store", "Paper journal verification failed."
            ) from exc

    def close(self) -> None:
        self.connection.close()

    @contextmanager
    def exclusive(self) -> Iterator[None]:
        """Reject competing operators immediately; OS releases the lock on crash."""
        descriptor = os.open(
            self.root / "operator.lock", os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW, 0o600
        )
        try:
            try:
                fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as exc:
                raise FoundryError(
                    "paper_busy", "Another paper operation is active.", 409
                ) from exc
            yield
        finally:
            os.close(descriptor)

    @contextmanager
    def transaction(self) -> Iterator[None]:
        try:
            self.connection.execute("BEGIN IMMEDIATE")
            yield
            self.connection.execute("COMMIT")
        except BaseException:
            if self.connection.in_transaction:
                self.connection.execute("ROLLBACK")
            raise

    def get(self, key: str) -> Any:
        row = self.connection.execute(
            "SELECT value, digest FROM state WHERE key=?", (key,)
        ).fetchone()
        if row is None:
            return None
        if hashlib.sha256(row[0]).hexdigest() != row[1]:
            raise FoundryError("paper_integrity", "Paper state checksum differs.")
        return decode(row[0])

    def _set(self, key: str, value: object) -> None:
        payload = encode(value, 16_384)
        self.connection.execute(
            "INSERT INTO state VALUES (?, ?, ?) ON CONFLICT(key) DO UPDATE SET"
            " value=excluded.value, digest=excluded.digest",
            (key, payload, hashlib.sha256(payload).hexdigest()),
        )

    def set(self, key: str, value: object, *, kind: str) -> None:
        """Atomically update bounded state and record the same value in the audit."""
        with self.transaction():
            self._append(kind, {"key": key, "value": value})
            self._set(key, value)

    def _append(self, kind: str, payload: object) -> None:
        row = self.connection.execute(
            "SELECT seq,digest FROM events ORDER BY seq DESC LIMIT 1"
        ).fetchone()
        seq, previous = (row[0] + 1, row[1]) if row else (1, ZERO)
        if seq > MAX_EVENTS - 10 and kind != "stop":
            raise FoundryError(
                "paper_capacity", "Journal is full; stop and archive.", 507
            )
        if seq > MAX_EVENTS:
            raise FoundryError("paper_capacity", "Journal event ceiling reached.", 507)
        at = datetime.now(UTC).isoformat()
        raw = encode(payload, 16_384)
        digest = hashlib.sha256(encode([seq, at, kind, previous]) + raw).hexdigest()
        self.connection.execute(
            "INSERT INTO events VALUES (?, ?, ?, ?, ?, ?)",
            (seq, at, kind, raw, previous, digest),
        )

    def append(self, kind: str, payload: object) -> None:
        with self.transaction():
            self._append(kind, payload)

    def verify(self) -> None:
        """Verify O(events) bytes in sequence, bounded by the physical store cap."""
        previous = ZERO
        count = 0
        for seq, at, kind, payload, parent, digest in self.connection.execute(
            "SELECT * FROM events ORDER BY seq"
        ):
            count += 1
            expected = hashlib.sha256(
                encode([seq, at, kind, parent]) + payload
            ).hexdigest()
            if (
                seq != count
                or parent != previous
                or digest != expected
                or count > MAX_EVENTS
            ):
                raise FoundryError("paper_integrity", "Paper audit chain differs.")
            previous = digest

    def request(self, now: float) -> None:
        """Reserve one request before dispatch; a restart cannot reset the budget."""
        with self.transaction():
            prior = self.get("requests") or []
            if prior and now < prior[-1]:
                raise FoundryError("clock_rollback", "Request clock moved backwards.")
            recent = [value for value in prior if now - value < 60]
            if len(recent) >= 100:
                raise FoundryError(
                    "provider_rate", "Local provider request budget exhausted.", 429
                )
            self._set("requests", [*recent, now])

    def stop(self) -> None:
        """Idempotent, persistent, irreversible for this state root."""
        with self.transaction():
            if not self.get("stopped"):
                self._append("stop", {"reason": "operator_stop"})
                self._set("stopped", True)

    def artifact(self, value: object) -> str:
        """Publish content-addressed data without replacing an existing artifact."""
        directory = private_directory(self.root / "artifacts", create=True)
        payload = encode(value)
        identity = hashlib.sha256(payload).hexdigest()
        destination = directory / f"{identity}.json"
        if destination.exists():
            if read_file(destination) != payload:
                raise FoundryError("paper_integrity", "An immutable artifact changed.")
            return identity
        paths = list(directory.iterdir())
        if len(paths) >= 256 or (
            sum(p.lstat().st_size for p in paths) + len(payload) > 512 << 20
        ):
            raise FoundryError(
                "paper_capacity", "Private artifact budget exhausted.", 507
            )
        descriptor, temporary_name = tempfile.mkstemp(prefix=".pending-", dir=directory)
        temporary = Path(temporary_name)
        try:
            with os.fdopen(descriptor, "wb") as stream:
                stream.write(payload)
                stream.flush()
                os.fsync(stream.fileno())
            # link publishes the complete inode atomically and refuses overwrite.
            os.link(temporary, destination)
            directory_fd = os.open(directory, os.O_RDONLY | os.O_DIRECTORY)
            try:
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)
        finally:
            temporary.unlink(missing_ok=True)
        return identity

    def read_artifact(self, identity: str) -> Any:
        if len(identity) != 64 or any(c not in "0123456789abcdef" for c in identity):
            raise FoundryError("paper_integrity", "Invalid artifact identity.")
        payload = read_file(self.root / "artifacts" / f"{identity}.json")
        if hashlib.sha256(payload).hexdigest() != identity:
            raise FoundryError("paper_integrity", "Artifact digest differs.")
        return decode(payload, 4 << 20)

    def order_keys(self) -> list[str]:
        rows = self.connection.execute(
            "SELECT key FROM state WHERE key LIKE 'order/%' ORDER BY key LIMIT 1001"
        ).fetchall()
        if len(rows) > 1000:
            raise FoundryError("paper_capacity", "Paper order ceiling exceeded.")
        return [row[0] for row in rows]

    def audit(self, after: int = 0, limit: int = 100) -> dict[str, Any]:
        """Export an ordered private audit page with its verification hashes."""
        if (
            type(after) is not int
            or not 0 <= after <= MAX_EVENTS
            or type(limit) is not int
            or not 1 <= limit <= 256
        ):
            raise FoundryError(
                "audit_cursor", "Choose a valid sequence and 1–256 events."
            )
        rows = self.connection.execute(
            "SELECT * FROM events WHERE seq > ? ORDER BY seq LIMIT ?", (after, limit)
        ).fetchall()
        records = [
            {
                "sequence": seq,
                "at": at,
                "kind": kind,
                "payload": decode(payload),
                "previous": previous,
                "digest": digest,
            }
            for seq, at, kind, payload, previous, digest in rows
        ]
        return {"events": records, "next_after": rows[-1][0] if rows else after}

    def event_count(self) -> int:
        return int(self.connection.execute("SELECT COUNT(*) FROM events").fetchone()[0])
