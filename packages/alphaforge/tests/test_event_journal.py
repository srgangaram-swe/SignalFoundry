from __future__ import annotations

import hashlib
import os
import sqlite3
from collections.abc import Callable, Iterator
from datetime import date, timedelta
from pathlib import Path

import pytest

import alphaforge.backtesting.journal as journal_module
from alphaforge.backtesting.journal import (
    InMemoryJournal,
    Journal,
    JournalBusyError,
    JournalClosedError,
    JournalCollisionError,
    JournalIntegrityError,
    JournalOrderingError,
    JournalPathError,
    JournalResourceLimitError,
    JournalSchemaError,
    SQLiteJournal,
)
from alphaforge.execution.events import (
    EventCoordinate,
    EventPhase,
    ExecutionEvent,
    SignalAvailable,
)


def _event(index: int) -> ExecutionEvent:
    token = f"signal-{index}".encode()
    return ExecutionEvent(
        run_id="journal-run",
        correlation_id=f"correlation-{index}",
        entity_id=f"entity-{index}",
        coordinate=EventCoordinate(
            session=date(2024, 1, 2) + timedelta(days=index),
            bar_index=index,
            phase=EventPhase.SIGNAL,
        ),
        payload=SignalAvailable(
            signal_id=f"signal-{index}",
            model_id="model-1",
            signal_digest=hashlib.sha256(token).hexdigest(),
        ),
    )


@pytest.fixture(params=("memory", "sqlite"))
def journal(request: pytest.FixtureRequest, tmp_path: Path) -> Iterator[Journal]:
    instance: Journal
    if request.param == "memory":
        instance = InMemoryJournal()
    else:
        instance = SQLiteJournal(tmp_path / "events.sqlite3")
    try:
        yield instance
    finally:
        instance.close()


def test_append_is_deterministic_and_exact_duplicates_are_noops(journal: Journal) -> None:
    first, second = _event(0), _event(1)

    assert journal.append(first)
    first_head = journal.head_hash
    assert not journal.append(first)
    assert journal.count == 1
    assert journal.head_hash == first_head
    assert journal.append(second)

    expected = (first.canonical_bytes(), second.canonical_bytes())
    assert journal.export_canonical() == expected
    assert journal.export_canonical() == expected
    assert journal.events() == (first, second)
    verified = journal.verify()
    assert verified.count == 2
    assert verified.event_hash == journal.head_hash
    assert verified.event_hash != bytes(32)


def test_identifier_body_collision_fails_without_mutation(
    journal: Journal, monkeypatch: pytest.MonkeyPatch
) -> None:
    first, second = _event(0), _event(1)
    assert journal.append(first)
    original_head = journal.head_hash

    # A collision is computationally infeasible with content-derived event IDs,
    # so fault-inject the already validated (identifier, body) boundary.
    monkeypatch.setattr(
        journal_module,
        "_canonical_event",
        lambda _event_value, _limits_value: (first.event_id, second.canonical_bytes()),
    )
    with pytest.raises(JournalCollisionError, match="different canonical bytes"):
        journal.append(second)

    assert journal.count == 1
    assert journal.head_hash == original_head


def test_logical_time_cannot_move_backward(journal: Journal) -> None:
    later, earlier = _event(1), _event(0)
    assert journal.append(later)

    with pytest.raises(JournalOrderingError, match="logical coordinate"):
        journal.append(earlier)

    assert journal.events() == (later,)


def test_context_manager_closes_and_close_is_idempotent(tmp_path: Path) -> None:
    memory = InMemoryJournal()
    with memory as opened:
        assert opened is memory
        assert opened.count == 0
    memory.close()
    with pytest.raises(JournalClosedError, match="closed"):
        _ = memory.count

    durable = SQLiteJournal(tmp_path / "closed.sqlite3")
    with durable as opened_durable:
        assert opened_durable is durable
    durable.close()
    with pytest.raises(JournalClosedError, match="closed"):
        durable.verify()


@pytest.mark.parametrize(
    ("kwargs", "exception", "message"),
    (
        ({"max_events": True}, TypeError, "max_events"),
        ({"max_events": 0}, ValueError, "max_events"),
        ({"max_event_bytes": 0}, ValueError, "max_event_bytes"),
        (
            {"max_event_bytes": 100, "max_total_payload_bytes": 99},
            ValueError,
            "max_total_payload_bytes",
        ),
    ),
)
def test_in_memory_limit_configuration_is_strict(
    kwargs: dict[str, object], exception: type[Exception], message: str
) -> None:
    with pytest.raises(exception, match=message):
        InMemoryJournal(**kwargs)  # type: ignore[arg-type]


def test_in_memory_event_count_payload_and_aggregate_limits_are_atomic() -> None:
    first, second = _event(0), _event(1)
    first_bytes = first.canonical_bytes()
    second_bytes = second.canonical_bytes()

    count_limited = InMemoryJournal(max_events=1)
    assert count_limited.append(first)
    with pytest.raises(JournalResourceLimitError, match="event-count"):
        count_limited.append(second)
    assert count_limited.events() == (first,)

    event_limited = InMemoryJournal(
        max_event_bytes=len(first_bytes) - 1,
        max_total_payload_bytes=len(first_bytes),
    )
    with pytest.raises(JournalResourceLimitError, match="canonical event"):
        event_limited.append(first)
    assert event_limited.count == 0

    aggregate_limited = InMemoryJournal(
        max_event_bytes=max(len(first_bytes), len(second_bytes)),
        max_total_payload_bytes=len(first_bytes) + len(second_bytes) - 1,
    )
    assert aggregate_limited.append(first)
    with pytest.raises(JournalResourceLimitError, match="payload-size"):
        aggregate_limited.append(second)
    assert aggregate_limited.events() == (first,)


def test_sqlite_restart_verifies_chain_and_preserves_idempotency(tmp_path: Path) -> None:
    path = tmp_path / "restart.sqlite3"
    events = tuple(_event(index) for index in range(3))
    with SQLiteJournal(path) as first:
        for event in events:
            assert first.append(event)
        expected_head = first.head_hash
        expected_export = first.export_canonical()

    with SQLiteJournal(path) as restarted:
        assert restarted.verify().event_hash == expected_head
        assert restarted.export_canonical() == expected_export
        assert restarted.events() == events
        assert not restarted.append(events[-1])
        assert restarted.count == len(events)


def test_sqlite_enforces_wal_full_sync_and_append_only_triggers(tmp_path: Path) -> None:
    path = tmp_path / "protected.sqlite3"
    journal = SQLiteJournal(path)
    assert journal.append(_event(0))
    assert journal._connection.execute("PRAGMA journal_mode").fetchone() == ("wal",)
    assert journal._connection.execute("PRAGMA synchronous").fetchone() == (2,)
    journal.close()

    connection = sqlite3.connect(path, isolation_level=None)
    try:
        with pytest.raises(sqlite3.IntegrityError, match="append-only"):
            connection.execute("UPDATE journal_events SET payload_size = payload_size")
        with pytest.raises(sqlite3.IntegrityError, match="append-only"):
            connection.execute("DELETE FROM journal_events")
        with pytest.raises(sqlite3.IntegrityError, match="append-only"):
            connection.execute("DELETE FROM journal_metadata")
    finally:
        connection.close()

    with SQLiteJournal(path) as verified:
        assert verified.events() == (_event(0),)


def _replace_event_payload_behind_trigger(path: Path, replacement: bytes) -> None:
    connection = sqlite3.connect(path, isolation_level=None)
    try:
        trigger_sql_row = connection.execute(
            "SELECT sql FROM sqlite_master WHERE type = ? AND name = ?",
            ("trigger", "journal_events_no_update"),
        ).fetchone()
        assert trigger_sql_row is not None
        trigger_sql = trigger_sql_row[0]
        assert isinstance(trigger_sql, str)
        connection.execute("BEGIN IMMEDIATE")
        connection.execute("DROP TRIGGER journal_events_no_update")
        connection.execute(
            "UPDATE journal_events SET payload = ?, payload_size = ? WHERE ordinal = ?",
            (replacement, len(replacement), 1),
        )
        connection.execute(trigger_sql)
        connection.execute("COMMIT")
    finally:
        if connection.in_transaction:
            connection.execute("ROLLBACK")
        connection.close()


def test_sqlite_full_restart_verification_detects_payload_tampering(tmp_path: Path) -> None:
    path = tmp_path / "tampered.sqlite3"
    with SQLiteJournal(path) as journal:
        assert journal.append(_event(0))
        assert journal.append(_event(1))

    _replace_event_payload_behind_trigger(path, _event(2).canonical_bytes())

    with pytest.raises(JournalIntegrityError, match="event hash"):
        SQLiteJournal(path)


def test_sqlite_rejects_truncation_and_schema_version_mismatch(tmp_path: Path) -> None:
    truncated_path = tmp_path / "truncated.sqlite3"
    with SQLiteJournal(truncated_path) as journal:
        assert journal.append(_event(0))
    os.truncate(truncated_path, 64)
    with pytest.raises(JournalIntegrityError):
        SQLiteJournal(truncated_path)

    schema_path = tmp_path / "schema.sqlite3"
    with SQLiteJournal(schema_path):
        pass
    connection = sqlite3.connect(schema_path, isolation_level=None)
    try:
        connection.execute("PRAGMA user_version = 99")
    finally:
        connection.close()
    with pytest.raises(JournalSchemaError, match="schema version"):
        SQLiteJournal(schema_path)


def test_sqlite_missing_append_only_trigger_fails_schema_validation(tmp_path: Path) -> None:
    path = tmp_path / "missing-trigger.sqlite3"
    with SQLiteJournal(path):
        pass
    connection = sqlite3.connect(path, isolation_level=None)
    try:
        connection.execute("DROP TRIGGER journal_events_no_delete")
    finally:
        connection.close()

    with pytest.raises(JournalSchemaError, match="append-only triggers"):
        SQLiteJournal(path)


def test_sqlite_lock_contention_has_no_partial_event_or_head_update(tmp_path: Path) -> None:
    path = tmp_path / "busy.sqlite3"
    journal = SQLiteJournal(path, busy_timeout_ms=1)
    contender = sqlite3.connect(path, timeout=0.0, isolation_level=None)
    try:
        contender.execute("BEGIN IMMEDIATE")
        with pytest.raises(JournalBusyError, match="busy"):
            journal.append(_event(0))
        assert journal.count == 0
        assert journal.head_hash == bytes(32)
        contender.execute("ROLLBACK")

        assert journal.append(_event(0))
        assert journal.verify().count == 1
    finally:
        if contender.in_transaction:
            contender.execute("ROLLBACK")
        contender.close()
        journal.close()


def test_sqlite_mid_transaction_failure_rolls_back_event_and_head(tmp_path: Path) -> None:
    path = tmp_path / "mid-transaction-failure.sqlite3"
    with SQLiteJournal(path) as journal:
        first, second = _event(0), _event(1)
        assert journal.append(first)
        expected_payloads = journal.export_canonical()
        expected_head = journal.head_hash

        def deny_metadata_update(
            action: int,
            table: str | None,
            _column: str | None,
            _database: str | None,
            _trigger: str | None,
        ) -> int:
            if action == sqlite3.SQLITE_UPDATE and table == "journal_metadata":
                return sqlite3.SQLITE_DENY
            return sqlite3.SQLITE_OK

        journal._connection.set_authorizer(deny_metadata_update)
        try:
            with pytest.raises(JournalIntegrityError, match="integrity during append"):
                journal.append(second)
        finally:
            journal._connection.set_authorizer(None)

        assert not journal._connection.in_transaction
        assert journal.count == 1
        assert journal.head_hash == expected_head
        assert journal.export_canonical() == expected_payloads
        stored_rows = journal._connection.execute(
            "SELECT ordinal, event_id, payload FROM journal_events ORDER BY ordinal"
        ).fetchall()
        assert stored_rows == [(1, first.event_id, first.canonical_bytes())]


def test_sqlite_database_size_refusal_precedes_any_insert(tmp_path: Path) -> None:
    path = tmp_path / "bounded.sqlite3"
    with SQLiteJournal(
        path,
        max_event_bytes=2_000,
        max_total_payload_bytes=100_000,
        max_database_bytes=128 * 1024,
    ) as journal:
        with pytest.raises(JournalResourceLimitError, match="database-size"):
            journal.append(_event(0))
        assert journal.count == 0
        assert journal.verify().count == 0


def _insecure_file(root: Path) -> Path:
    path = root / "insecure.sqlite3"
    path.touch(mode=0o600)
    path.chmod(0o644)
    return path


def _symlink_file(root: Path) -> Path:
    target = root / "target.sqlite3"
    target.touch(mode=0o600)
    link = root / "linked.sqlite3"
    link.symlink_to(target)
    return link


def _symlink_parent(root: Path) -> Path:
    target = root / "real-parent"
    target.mkdir(mode=0o700)
    link = root / "linked-parent"
    link.symlink_to(target, target_is_directory=True)
    return link / "events.sqlite3"


def _hardlink_file(root: Path) -> Path:
    target = root / "hardlink-target.sqlite3"
    target.touch(mode=0o600)
    link = root / "hardlinked.sqlite3"
    os.link(target, link)
    return link


def _insecure_parent(root: Path) -> Path:
    parent = root / "insecure-parent"
    parent.mkdir(mode=0o700)
    parent.chmod(0o770)
    return parent / "events.sqlite3"


@pytest.mark.parametrize(
    "make_path",
    (
        lambda root: _insecure_file(root),
        lambda root: _symlink_file(root),
        lambda root: _symlink_parent(root),
        lambda root: _hardlink_file(root),
        lambda root: _insecure_parent(root),
    ),
)
def test_sqlite_rejects_insecure_or_symlinked_paths(
    tmp_path: Path, make_path: Callable[[Path], Path]
) -> None:
    with pytest.raises(JournalPathError):
        SQLiteJournal(make_path(tmp_path))


def test_sqlite_database_and_sidecar_permissions_are_owner_only(tmp_path: Path) -> None:
    path = tmp_path / "permissions.sqlite3"
    journal = SQLiteJournal(path)
    try:
        assert journal.append(_event(0))
        for candidate in (path, Path(f"{path}-wal"), Path(f"{path}-shm")):
            if candidate.exists():
                assert candidate.stat().st_mode & 0o077 == 0
    finally:
        journal.close()
