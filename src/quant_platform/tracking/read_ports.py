"""Framework-neutral, bounded read ports for durable experiment evidence.

The HTTP and console adapters introduced by later sprint slices depend on this
module rather than on SQLite or filesystem details.  Every database connection
is opened read-only and query-only, every page is an authenticated keyset
snapshot, and every returned object is frozen.  Relative CAS storage keys are
used only inside the verified-read boundary and are never part of a public
projection.

Legacy tracker rows are intentionally less expressive than registry runs.
They carry no durable job identity, attempt, evidence class, parameters, tags,
metrics, or artifact paths.  Their small safe projection is labelled
``legacy/unverified``; an active or unknown legacy status remains unsupported
rather than being promoted to terminal evidence.
"""

from __future__ import annotations

import hashlib
import hmac
import math
import re
import sqlite3
import time
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import StrEnum
from typing import Final, cast

from quant_platform.tracking.cas import (
    ArtifactStore,
    ArtifactStoreError,
    PublishedArtifact,
)
from quant_platform.tracking.contracts import (
    ArtifactClass,
    ArtifactCursor,
    ArtifactMetadata,
    BusyError,
    EvidenceClass,
    IntegrityError,
    InvalidCursorError,
    NotFoundError,
    Page,
    RegistryError,
    RegistryReadiness,
    RunCursor,
    RunSnapshot,
    RunStatus,
    ValidationError,
    decode_bounded_json,
    parse_stored_utc,
    require_digest,
    require_identifier,
    require_stored_int,
    require_utc,
    require_utf8_size,
)
from quant_platform.tracking.registry import RunRegistry

_MAX_SEQUENCE_COMPONENT: Final = 2**31 - 1
_LEGACY_POSITION_BASE: Final = 2**31
_SNAPSHOT_MARKER: Final = 2**62
_READ_PROGRESS_INSTRUCTIONS: Final = 1_000
_LEGACY_CURSOR_MARKER: Final = ";legacy="
_LEGACY_FINGERPRINT_DOMAIN: Final = b"signalattice.read-port.legacy-snapshot.v1\0"
_LEGACY_LIMITATION: Final = (
    "Legacy tracker row: lifecycle, parameters, metrics, tags, and artifact references are "
    "unverified and intentionally redacted."
)
_ACTIVE_LEGACY_LIMITATION: Final = (
    "Legacy tracker row reports non-terminal or unknown state; no terminal evidence is inferred."
)
_COMMIT = re.compile(r"^[0-9a-fA-F]{7,64}$")
_DATA_DIGEST = re.compile(r"^(?:sha256:)?[0-9a-f]{64}$")
_ARTIFACT_ROLE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
_MEDIA_TYPE = re.compile(r"^[a-z0-9][a-z0-9!#$&^_.+-]*/[a-z0-9][a-z0-9!#$&^_.+-]{0,126}$")
_ALLOWED_MANIFEST_MEDIA_TYPES: Final = frozenset(
    {
        "application/json",
        "application/manifest+json",
        "application/vnd.signalattice.manifest+json",
    }
)


class ReadTimeoutError(RegistryError):
    """A read exceeded its configured cooperative SQLite deadline."""

    code = "read_timeout"
    retryable = True


class RunProvenance(StrEnum):
    """Trust classification for a framework-neutral run projection."""

    REGISTRY_VERIFIED = "registry/verified"
    LEGACY_UNVERIFIED = "legacy/unverified"


@dataclass(frozen=True, slots=True)
class ReadPortLimits:
    """Independent resource ceilings for a read-port instance."""

    max_page_size: int = 100
    query_timeout_ms: int = 500
    max_legacy_json_bytes: int = 65_536
    max_manifest_bytes: int = 8 * 1024 * 1024

    def __post_init__(self) -> None:
        bounds = {
            "max_page_size": (self.max_page_size, 1, 1_000),
            "query_timeout_ms": (self.query_timeout_ms, 1, 60_000),
            "max_legacy_json_bytes": (self.max_legacy_json_bytes, 1_024, 1_048_576),
            "max_manifest_bytes": (self.max_manifest_bytes, 1, 64 * 1024 * 1024),
        }
        for name, (value, lower, upper) in bounds.items():
            if type(value) is not int or not lower <= value <= upper:
                raise ValidationError(f"{name} must be in [{lower}, {upper}]")


@dataclass(frozen=True, slots=True)
class RunQuery:
    """Closed, cursor-bound filters for a stable run listing."""

    status: RunStatus | None = None
    provenance: RunProvenance | None = None

    def __post_init__(self) -> None:
        if self.status is not None and type(self.status) is not RunStatus:
            raise ValidationError("status must be null or a RunStatus")
        if self.provenance is not None and type(self.provenance) is not RunProvenance:
            raise ValidationError("provenance must be null or a RunProvenance")

    @property
    def cursor_identity(self) -> str:
        """Return a bounded canonical identity authenticated into every cursor."""

        status = "*" if self.status is None else self.status.value
        provenance = "*" if self.provenance is None else self.provenance.value
        return f"status={status};provenance={provenance}"


@dataclass(frozen=True, slots=True)
class RunPageRequest:
    """Bounded page request separated from semantic filters."""

    page_size: int = 50
    cursor: RunCursor | None = None

    def __post_init__(self) -> None:
        if type(self.page_size) is not int or not 1 <= self.page_size <= 1_000:
            raise ValidationError("page_size must be in [1, 1000]")
        if self.cursor is not None and type(self.cursor) is not RunCursor:
            raise InvalidCursorError("run cursor has the wrong contract type")


@dataclass(frozen=True, slots=True)
class ArtifactPageRequest:
    """Bounded page request for immutable run-to-artifact links."""

    page_size: int = 50
    cursor: ArtifactCursor | None = None

    def __post_init__(self) -> None:
        if type(self.page_size) is not int or not 1 <= self.page_size <= 1_000:
            raise ValidationError("page_size must be in [1, 1000]")
        if self.cursor is not None and type(self.cursor) is not ArtifactCursor:
            raise InvalidCursorError("artifact cursor has the wrong contract type")


@dataclass(frozen=True, slots=True)
class LegacyRunSnapshot:
    """Minimal path- and credential-free view of a historical tracker row."""

    sequence: int
    run_id: str
    started_at: datetime
    ended_at: datetime | None
    reported_terminal_status: RunStatus | None
    source_commit: str | None
    data_identity: str | None
    provenance: RunProvenance = field(default=RunProvenance.LEGACY_UNVERIFIED, init=False)
    limitation_summary: str = field(default=_LEGACY_LIMITATION, init=False)

    def __post_init__(self) -> None:
        if type(self.sequence) is not int or not 1 <= self.sequence <= _MAX_SEQUENCE_COMPONENT:
            raise ValidationError("legacy sequence is outside the supported snapshot range")
        require_identifier(self.run_id, "run_id")
        object.__setattr__(self, "started_at", require_utc(self.started_at, "started_at"))
        if self.ended_at is not None:
            object.__setattr__(self, "ended_at", require_utc(self.ended_at, "ended_at"))
            if self.ended_at < self.started_at:
                raise ValidationError("legacy ended_at may not precede started_at")
        if (
            self.reported_terminal_status is not None
            and type(self.reported_terminal_status) is not RunStatus
        ):
            raise ValidationError("reported_terminal_status must be null or a RunStatus")
        if self.reported_terminal_status is None:
            object.__setattr__(self, "limitation_summary", _ACTIVE_LEGACY_LIMITATION)
        if self.source_commit is not None and not _COMMIT.fullmatch(self.source_commit):
            raise ValidationError("source_commit must be a hexadecimal revision")
        if self.data_identity is not None and not _DATA_DIGEST.fullmatch(self.data_identity):
            raise ValidationError("data_identity must be a SHA-256 identity")


type RunReadModel = RunSnapshot | LegacyRunSnapshot


@dataclass(frozen=True, slots=True)
class ArtifactView:
    """Path-free immutable artifact metadata safe for service adapters."""

    artifact_id: str
    artifact_class: ArtifactClass
    byte_size: int
    media_type: str
    created_at: datetime
    pinned: bool

    def __post_init__(self) -> None:
        require_digest(self.artifact_id, "artifact_id")
        if type(self.artifact_class) is not ArtifactClass:
            raise ValidationError("artifact_class must be an ArtifactClass")
        if type(self.byte_size) is not int or not 0 <= self.byte_size <= 2**63 - 1:
            raise ValidationError("byte_size must be a non-negative signed 64-bit integer")
        if type(self.media_type) is not str or not _MEDIA_TYPE.fullmatch(self.media_type):
            raise ValidationError("media_type must be a lowercase type/subtype")
        object.__setattr__(self, "created_at", require_utc(self.created_at, "created_at"))
        if type(self.pinned) is not bool:
            raise ValidationError("pinned must be a boolean")


@dataclass(frozen=True, slots=True)
class RunArtifactView:
    """One immutable run role linked to path-free artifact metadata."""

    sequence: int
    role: str
    linked_at: datetime
    artifact: ArtifactView

    def __post_init__(self) -> None:
        if type(self.sequence) is not int or not 1 <= self.sequence <= 2**63 - 1:
            raise ValidationError("artifact-link sequence must be positive")
        if type(self.role) is not str or not _ARTIFACT_ROLE.fullmatch(self.role):
            raise ValidationError("artifact role must be a path-free bounded identifier")
        object.__setattr__(self, "linked_at", require_utc(self.linked_at, "linked_at"))
        if type(self.artifact) is not ArtifactView:
            raise ValidationError("artifact must be an ArtifactView")


class RegistryReadPorts:
    """Read-only facade over a durable registry and initialized local CAS.

    Construction has no side effects.  In particular, it does not initialize,
    migrate, repair, create, or bind either storage system.  The supplied
    :class:`ArtifactStore` must already have been initialized by the owning
    process before a manifest read; listing metadata never touches the CAS.
    """

    def __init__(
        self,
        registry: RunRegistry,
        artifact_store: ArtifactStore,
        *,
        limits: ReadPortLimits | None = None,
    ) -> None:
        if type(registry) is not RunRegistry:
            raise ValidationError("registry must be a RunRegistry")
        if type(artifact_store) is not ArtifactStore:
            raise ValidationError("artifact_store must be an ArtifactStore")
        self._registry = registry
        self._artifact_store = artifact_store
        if limits is None:
            self._limits = ReadPortLimits(
                max_page_size=min(registry.limits.max_page_size, 100),
            )
        elif type(limits) is ReadPortLimits:
            self._limits = limits
        else:
            raise ValidationError("limits must be a ReadPortLimits")
        if self._limits.max_page_size > registry.limits.max_page_size:
            raise ValidationError("read page bound may not exceed the registry page bound")

    def probe_readiness(self) -> RegistryReadiness:
        """Inspect an existing database without creating or migrating any state."""

        return self._registry.probe_readiness()

    def get_run(self, run_id: str) -> RunReadModel:
        """Return one safe run projection, rejecting ambiguous source collisions."""

        require_identifier(run_id, "run_id")
        with self._reader() as connection:
            registry_row = connection.execute(
                "SELECT * FROM sl_registry_runs WHERE run_id = ?", (run_id,)
            ).fetchone()
            legacy_row = self._get_legacy_row(connection, run_id)
        if registry_row is not None and legacy_row is not None:
            raise IntegrityError("run identity is ambiguous across registry and legacy state")
        if registry_row is not None:
            return self._registry_run(registry_row)
        if legacy_row is not None:
            return self._legacy_run(legacy_row)
        raise NotFoundError("run does not exist")

    def list_runs(
        self,
        query: RunQuery | None = None,
        page: RunPageRequest | None = None,
    ) -> Page[RunReadModel, RunCursor]:
        """List a bounded run page under authenticated dual-source high-water marks."""

        if query is None:
            query = RunQuery()
        if page is None:
            page = RunPageRequest()
        if type(query) is not RunQuery:
            raise ValidationError("query must be a RunQuery")
        if type(page) is not RunPageRequest:
            raise ValidationError("page must be a RunPageRequest")
        self._validate_page_size(page.page_size)

        next_filter: str | None = None
        with self._reader() as connection:
            legacy_fingerprint: str | None = None
            expected_legacy_fingerprint: str | None = None
            if page.cursor is None:
                registry_max = self._max_sequence(connection, "sl_registry_runs")
                legacy_max = self._legacy_max_rowid(connection)
                snapshot = self._encode_snapshot(registry_max, legacy_max)
                after = 0
            else:
                snapshot, after, cursor_filter = self._registry.cursor_codec.decode_run(page.cursor)
                query_identity, expected_legacy_fingerprint = self._decode_cursor_filter(
                    cursor_filter
                )
                if query_identity != query.cursor_identity:
                    raise InvalidCursorError("run cursor filter does not match this request")
                registry_max, legacy_max = self._decode_snapshot(snapshot)
                self._validate_run_position(after)
                if self._max_sequence(connection, "sl_registry_runs") < registry_max:
                    raise InvalidCursorError("run cursor snapshot source regressed")
            if query.provenance is not RunProvenance.REGISTRY_VERIFIED:
                self._assert_safe_legacy_scalar_cells(
                    connection,
                    snapshot=legacy_max,
                )
                if expected_legacy_fingerprint is not None:
                    legacy_fingerprint = self._legacy_fingerprint(connection, legacy_max)
                    if not hmac.compare_digest(
                        legacy_fingerprint,
                        expected_legacy_fingerprint,
                    ):
                        raise InvalidCursorError(
                            "legacy run state changed since the snapshot cursor was issued"
                        )
            if query.provenance is None:
                self._assert_no_cross_source_run_ids(
                    connection,
                    registry_snapshot=registry_max,
                    legacy_snapshot=legacy_max,
                )
            if query.provenance is not RunProvenance.REGISTRY_VERIFIED:
                self._assert_unique_legacy_run_ids(connection, legacy_max)

            rows: list[tuple[int, RunReadModel]] = []
            fetch_limit = page.page_size + 1
            if (
                query.provenance is not RunProvenance.LEGACY_UNVERIFIED
                and after < _LEGACY_POSITION_BASE
            ):
                registry_rows = self._list_registry_rows(
                    connection,
                    status=query.status,
                    after=after,
                    snapshot=registry_max,
                    limit=fetch_limit,
                )
                rows.extend(
                    (
                        require_stored_int(
                            row["sequence"],
                            "run sequence",
                            1,
                            _MAX_SEQUENCE_COMPONENT,
                        ),
                        self._registry_run(row),
                    )
                    for row in registry_rows
                )

            if len(rows) < fetch_limit and query.provenance is not RunProvenance.REGISTRY_VERIFIED:
                legacy_after = max(0, after - _LEGACY_POSITION_BASE)
                legacy_rows = self._list_legacy_rows(
                    connection,
                    status=query.status,
                    after=legacy_after,
                    snapshot=legacy_max,
                    limit=fetch_limit - len(rows),
                )
                rows.extend(
                    (
                        _LEGACY_POSITION_BASE
                        + require_stored_int(
                            row["legacy_rowid"],
                            "legacy sequence",
                            1,
                            _MAX_SEQUENCE_COMPONENT,
                        ),
                        self._legacy_run(row),
                    )
                    for row in legacy_rows
                )

            if len(rows) > page.page_size:
                next_filter = self._encode_cursor_filter(
                    query,
                    (
                        "-"
                        if query.provenance is RunProvenance.REGISTRY_VERIFIED
                        else legacy_fingerprint or self._legacy_fingerprint(connection, legacy_max)
                    ),
                )

        visible = tuple(item for _, item in rows[: page.page_size])
        next_cursor = None
        if len(rows) > page.page_size:
            if next_filter is None:
                raise IntegrityError("run page cursor filter was not produced")
            next_cursor = self._registry.cursor_codec.encode_run(
                snapshot,
                rows[page.page_size - 1][0],
                next_filter,
            )
        return Page(visible, next_cursor)

    def get_artifact(self, artifact_id: str) -> ArtifactView:
        """Return immutable metadata without exposing any storage pathname."""

        metadata = self._get_artifact_metadata(artifact_id)
        return self._artifact_view(metadata)

    def list_run_artifacts(
        self,
        run_id: str,
        page: ArtifactPageRequest | None = None,
    ) -> Page[RunArtifactView, ArtifactCursor]:
        """List immutable path-free links under a run-bound snapshot cursor."""

        require_identifier(run_id, "run_id")
        if page is None:
            page = ArtifactPageRequest()
        if type(page) is not ArtifactPageRequest:
            raise ValidationError("page must be an ArtifactPageRequest")
        self._validate_page_size(page.page_size)

        with self._reader() as connection:
            registry_exists = connection.execute(
                "SELECT 1 FROM sl_registry_runs WHERE run_id = ?", (run_id,)
            ).fetchone()
            if registry_exists is None:
                if self._get_legacy_row(connection, run_id) is not None:
                    return Page((), None)
                raise NotFoundError("run does not exist")
            if page.cursor is None:
                row = connection.execute(
                    "SELECT MIN(sequence) AS minimum, MAX(sequence) AS maximum "
                    "FROM sl_registry_run_artifacts WHERE run_id = ?",
                    (run_id,),
                ).fetchone()
                snapshot = _sequence_maximum(
                    cast(sqlite3.Row, row),
                    field_name="artifact-link sequence",
                    maximum=2**63 - 1,
                )
                after = 0
            else:
                current_row = connection.execute(
                    "SELECT MIN(sequence) AS minimum, MAX(sequence) AS maximum "
                    "FROM sl_registry_run_artifacts WHERE run_id = ?",
                    (run_id,),
                ).fetchone()
                current_maximum = _sequence_maximum(
                    cast(sqlite3.Row, current_row),
                    field_name="artifact-link sequence",
                    maximum=2**63 - 1,
                )
                snapshot, after, cursor_run_id = self._registry.cursor_codec.decode_artifact(
                    page.cursor
                )
                if cursor_run_id != run_id:
                    raise InvalidCursorError(
                        "artifact cursor run filter does not match this request"
                    )
                if current_maximum < snapshot:
                    raise InvalidCursorError("artifact cursor snapshot source regressed")
            rows = connection.execute(
                """
                SELECT l.sequence, l.role, l.linked_at,
                       a.digest, a.artifact_class, a.byte_size, a.media_type,
                       a.storage_relpath, a.created_at, a.pinned
                FROM sl_registry_run_artifacts AS l
                LEFT JOIN sl_registry_artifacts AS a ON a.digest = l.artifact_digest
                WHERE l.run_id = ? AND l.sequence > ? AND l.sequence <= ?
                ORDER BY l.sequence
                LIMIT ?
                """,
                (run_id, after, snapshot, page.page_size + 1),
            ).fetchall()

        items = tuple(self._run_artifact(row) for row in rows[: page.page_size])
        next_cursor = None
        if len(rows) > page.page_size:
            next_cursor = self._registry.cursor_codec.encode_artifact(
                snapshot,
                require_stored_int(
                    rows[page.page_size - 1]["sequence"],
                    "artifact-link sequence",
                    1,
                    2**63 - 1,
                ),
                run_id,
            )
        return Page(items, next_cursor)

    def read_verified_manifest(
        self,
        artifact_id: str,
        expected_media_type: str,
        max_bytes: int,
    ) -> bytes:
        """Return one digest-, size-, key-, and media-verified immutable byte snapshot.

        The caller limit is mandatory and cannot exceed the configured port
        ceiling.  Only JSON manifest media types cross this narrow boundary.
        The CAS store performs descriptor-anchored containment, type, mutation,
        and digest verification before bytes are returned.
        """

        require_digest(artifact_id, "artifact_id")
        if (
            type(expected_media_type) is not str
            or expected_media_type not in _ALLOWED_MANIFEST_MEDIA_TYPES
        ):
            raise ValidationError("expected_media_type is not an allowlisted manifest type")
        if type(max_bytes) is not int or not 1 <= max_bytes <= self._limits.max_manifest_bytes:
            raise ValidationError(f"max_bytes must be in [1, {self._limits.max_manifest_bytes}]")
        metadata = self._get_artifact_metadata(artifact_id)
        if metadata.artifact_class is not ArtifactClass.METADATA:
            raise IntegrityError("artifact class is not authorized for manifest reads")
        if metadata.media_type != expected_media_type:
            raise IntegrityError("artifact media type does not match the requested manifest type")
        if metadata.byte_size > max_bytes:
            raise ValidationError("artifact exceeds the caller-declared manifest byte limit")
        try:
            bound_store_id = self._registry.artifact_store_id
            if bound_store_id is None:
                raise IntegrityError("registry is not bound to a CAS store identity")
            if not hmac.compare_digest(bound_store_id, self._artifact_store.store_id):
                raise IntegrityError("manifest CAS store does not match the registry binding")
            artifact = PublishedArtifact(
                digest=metadata.digest,
                byte_size=metadata.byte_size,
                storage_key=metadata.storage_relpath,
            )
            return self._artifact_store.read_verified(artifact, max_bytes=max_bytes)
        except ArtifactStoreError as exc:
            raise IntegrityError("artifact bytes failed verified CAS read") from exc

    @contextmanager
    def _reader(self) -> Iterator[sqlite3.Connection]:
        connection = self._registry._connect(
            readonly=True,
            busy_timeout_ms=min(
                self._registry.limits.busy_timeout_ms,
                self._limits.query_timeout_ms,
            ),
        )
        deadline = time.monotonic() + (self._limits.query_timeout_ms / 1_000)

        def expired() -> int:
            return int(time.monotonic() >= deadline)

        connection.set_progress_handler(expired, _READ_PROGRESS_INSTRUCTIONS)
        primary_error: BaseException | None = None
        try:
            connection.execute("BEGIN")
            yield connection
        except RegistryError as exc:
            primary_error = exc
            raise
        except sqlite3.Error as exc:
            primary_code = getattr(exc, "sqlite_errorcode", -1) & 0xFF
            if primary_code == sqlite3.SQLITE_INTERRUPT:
                mapped: RegistryError = ReadTimeoutError(
                    "registry read exceeded its configured deadline"
                )
            elif primary_code in {sqlite3.SQLITE_BUSY, sqlite3.SQLITE_LOCKED}:
                mapped = BusyError("registry read remained busy past its configured bound")
            else:
                mapped = IntegrityError("SQLite rejected a registry read")
            primary_error = mapped
            raise mapped from exc
        finally:
            cleanup_failures: list[BaseException] = []
            try:
                connection.set_progress_handler(None, 0)
            except sqlite3.Error as exc:
                cleanup_failures.append(exc)
            try:
                if connection.in_transaction:
                    connection.execute("ROLLBACK")
            except sqlite3.Error as exc:
                cleanup_failures.append(exc)
            try:
                connection.close()
            except sqlite3.Error as exc:
                cleanup_failures.append(exc)
            if cleanup_failures:
                if primary_error is not None:
                    names = ",".join(type(error).__name__ for error in cleanup_failures)
                    primary_error.add_note(f"registry read cleanup also failed: {names}")
                else:
                    raise IntegrityError("registry read cleanup failed") from cleanup_failures[0]

    def _get_artifact_metadata(self, artifact_id: str) -> ArtifactMetadata:
        require_digest(artifact_id, "artifact_id")
        with self._reader() as connection:
            row = connection.execute(
                "SELECT * FROM sl_registry_artifacts WHERE digest = ?", (artifact_id,)
            ).fetchone()
        if row is None:
            raise NotFoundError("artifact does not exist")
        return self._artifact_metadata(row)

    @staticmethod
    def _registry_run(row: sqlite3.Row) -> RunSnapshot:
        try:
            source_commit = _safe_commit(row["source_commit"])
            data_identity = _safe_data_identity(row["data_identity"])
            limitation = None if row["limitation_summary"] is None else "Limitations recorded."
            return RunSnapshot(
                sequence=_stored_int(row["sequence"], "run sequence"),
                run_id=_stored_text(row["run_id"], "run_id", 128),
                job_id=_stored_text(row["job_id"], "job_id", 128),
                attempt=_stored_int(row["attempt"], "attempt"),
                status=RunStatus(_stored_text(row["status"], "status", 32)),
                evidence_class=EvidenceClass(
                    _stored_text(row["evidence_class"], "evidence_class", 32)
                ),
                schema_version=_stored_int(row["schema_version"], "schema_version"),
                created_at=_stored_time(row["created_at"], "created_at"),
                started_at=_optional_stored_time(row["started_at"], "started_at"),
                ended_at=_stored_time(row["ended_at"], "ended_at"),
                source_commit=source_commit,
                data_identity=data_identity,
                limitation_summary=limitation,
            )
        except (ValueError, ValidationError) as exc:
            raise IntegrityError("stored registry run is malformed") from exc

    def _legacy_run(self, row: sqlite3.Row) -> LegacyRunSnapshot:
        try:
            for name in (
                "run_id",
                "started_at",
                "ended_at",
                "status",
                "git_commit",
                "data_hash",
            ):
                if (
                    _stored_int(
                        row[f"legacy_{name}_safe"],
                        f"legacy {name} safety flag",
                    )
                    != 1
                ):
                    raise IntegrityError(
                        "legacy scalar field is malformed or exceeds its byte bound"
                    )
            for name, expected in (
                ("tickers", list),
                ("features", list),
                ("params", dict),
                ("metrics", dict),
                ("tags", dict),
                ("artifacts", list),
            ):
                self._validate_legacy_json(row, name, expected)
            status = _legacy_status(row["legacy_status"])
            ended_at = _optional_legacy_stored_time(row["legacy_ended_at"], "legacy ended_at")
            if status is not None and ended_at is None:
                raise IntegrityError("terminal legacy run is missing ended_at")
            return LegacyRunSnapshot(
                sequence=_stored_int(row["legacy_rowid"], "legacy sequence"),
                run_id=_stored_text(row["legacy_run_id"], "legacy run_id", 128),
                started_at=_legacy_stored_time(row["legacy_started_at"], "legacy started_at"),
                ended_at=ended_at,
                reported_terminal_status=status,
                source_commit=_safe_commit(row["legacy_git_commit"]),
                data_identity=_safe_data_identity(row["legacy_data_hash"]),
            )
        except (ValueError, ValidationError) as exc:
            raise IntegrityError("legacy run cannot be projected safely") from exc

    def _validate_legacy_json(
        self,
        row: sqlite3.Row,
        name: str,
        expected_type: type[object],
    ) -> None:
        if _stored_int(row[f"legacy_{name}_safe"], f"legacy {name} safety flag") != 1:
            raise IntegrityError("legacy JSON field is malformed or exceeds its byte bound")
        raw = row[f"legacy_{name}"]
        if raw is None:
            return
        try:
            value = decode_bounded_json(
                raw,
                maximum_bytes=self._limits.max_legacy_json_bytes,
            )
        except ValidationError:
            raise IntegrityError("legacy JSON field is malformed or non-finite") from None
        if type(value) is not expected_type:
            raise IntegrityError("legacy JSON field has an unexpected shape")
        if name == "metrics":
            metrics = cast(Mapping[object, object], value)
            for key, metric in metrics.items():
                if type(key) is not str or type(metric) not in {int, float}:
                    raise IntegrityError("legacy metrics are not finite numeric scalars")
                if not math.isfinite(cast(int | float, metric)):
                    raise IntegrityError("legacy metrics are not finite numeric scalars")

    @staticmethod
    def _artifact_metadata(row: sqlite3.Row) -> ArtifactMetadata:
        try:
            pinned = _stored_int(row["pinned"], "artifact pinned")
            if pinned not in {0, 1}:
                raise IntegrityError("stored artifact pinned state is invalid")
            metadata = ArtifactMetadata(
                digest=_stored_text(row["digest"], "artifact digest", 64),
                artifact_class=ArtifactClass(
                    _stored_text(row["artifact_class"], "artifact_class", 32)
                ),
                byte_size=_stored_int(row["byte_size"], "artifact byte_size"),
                media_type=_stored_text(row["media_type"], "media_type", 128),
                storage_relpath=_stored_text(row["storage_relpath"], "storage key", 1_024),
                created_at=_stored_time(row["created_at"], "artifact created_at"),
                pinned=pinned == 1,
            )
            PublishedArtifact(
                digest=metadata.digest,
                byte_size=metadata.byte_size,
                storage_key=metadata.storage_relpath,
            )
            return metadata
        except ArtifactStoreError as exc:
            raise IntegrityError("stored artifact storage key is noncanonical") from exc
        except (ValueError, ValidationError) as exc:
            raise IntegrityError("stored artifact metadata is malformed") from exc

    @staticmethod
    def _artifact_view(metadata: ArtifactMetadata) -> ArtifactView:
        return ArtifactView(
            artifact_id=metadata.digest,
            artifact_class=metadata.artifact_class,
            byte_size=metadata.byte_size,
            media_type=metadata.media_type,
            created_at=metadata.created_at,
            pinned=metadata.pinned,
        )

    @classmethod
    def _run_artifact(cls, row: sqlite3.Row) -> RunArtifactView:
        if row["digest"] is None:
            raise IntegrityError("stored run-artifact link references missing metadata")
        try:
            return RunArtifactView(
                sequence=_stored_int(row["sequence"], "artifact-link sequence"),
                role=_stored_text(row["role"], "artifact role", 128),
                linked_at=_stored_time(row["linked_at"], "linked_at"),
                artifact=cls._artifact_view(cls._artifact_metadata(row)),
            )
        except (ValueError, ValidationError) as exc:
            raise IntegrityError("stored run-artifact link is malformed") from exc

    @staticmethod
    def _max_sequence(connection: sqlite3.Connection, table: str) -> int:
        if table != "sl_registry_runs":
            raise IntegrityError("unsupported run snapshot source")
        row = connection.execute(
            "SELECT MIN(sequence) AS minimum, MAX(sequence) AS maximum FROM sl_registry_runs"
        ).fetchone()
        return _sequence_maximum(
            cast(sqlite3.Row, row),
            field_name="run sequence",
            maximum=_MAX_SEQUENCE_COMPONENT,
        )

    def _legacy_max_rowid(self, connection: sqlite3.Connection) -> int:
        if not self._legacy_table_exists(connection):
            return 0
        row = cast(
            sqlite3.Row,
            connection.execute(
                "SELECT MIN(rowid) AS minimum, MAX(rowid) AS maximum FROM runs"
            ).fetchone(),
        )
        return _sequence_maximum(
            row,
            field_name="legacy sequence",
            maximum=_MAX_SEQUENCE_COMPONENT,
        )

    @staticmethod
    def _encode_snapshot(registry_max: int, legacy_max: int) -> int:
        if not (
            0 <= registry_max <= _MAX_SEQUENCE_COMPONENT
            and 0 <= legacy_max <= _MAX_SEQUENCE_COMPONENT
        ):
            raise IntegrityError("run snapshot components exceed their supported bounds")
        return _SNAPSHOT_MARKER | (registry_max << 31) | legacy_max

    @staticmethod
    def _decode_snapshot(snapshot: int) -> tuple[int, int]:
        if type(snapshot) is not int or snapshot & _SNAPSHOT_MARKER == 0:
            raise InvalidCursorError("run cursor snapshot has an unsupported encoding")
        payload = snapshot ^ _SNAPSHOT_MARKER
        if payload >= _SNAPSHOT_MARKER:
            raise InvalidCursorError("run cursor snapshot has an unsupported encoding")
        return payload >> 31, payload & _MAX_SEQUENCE_COMPONENT

    @staticmethod
    def _validate_run_position(after: int) -> None:
        if type(after) is not int or not 0 <= after <= 2**32 - 1:
            raise InvalidCursorError("run cursor position is outside the supported range")

    def _legacy_fingerprint(self, connection: sqlite3.Connection, snapshot: int) -> str:
        """Hash every bounded projection input at or below a legacy high-water mark.

        The digest detects updates, deletions, explicit low-rowid insertion, and
        rowid reuse between pages.  Large or malformed JSON is represented by
        its safe flag rather than materialized.  The reader transaction pins
        the fingerprint and page query to the same SQLite snapshot.
        """

        hasher = hashlib.sha256(_LEGACY_FINGERPRINT_DOMAIN)
        if not self._legacy_table_exists(connection):
            return hasher.hexdigest()
        cursor = connection.execute(
            self._legacy_select() + " WHERE rowid <= ? ORDER BY rowid",
            self._legacy_select_parameters() + [snapshot],
        )
        while row := cursor.fetchone():
            for value in row:
                if value is None:
                    hasher.update(b"n")
                    continue
                if type(value) is int:
                    encoded = str(value).encode("ascii")
                    hasher.update(b"i")
                elif type(value) is str:
                    try:
                        encoded = value.encode("utf-8")
                    except (MemoryError, UnicodeEncodeError):
                        raise IntegrityError(
                            "legacy snapshot contains text that is not bounded UTF-8"
                        ) from None
                    hasher.update(b"s")
                else:
                    raise IntegrityError("legacy snapshot contains an unsupported SQLite type")
                hasher.update(len(encoded).to_bytes(8, "big"))
                hasher.update(encoded)
        return hasher.hexdigest()

    def _assert_no_cross_source_run_ids(
        self,
        connection: sqlite3.Connection,
        *,
        registry_snapshot: int,
        legacy_snapshot: int,
    ) -> None:
        if not self._legacy_table_exists(connection):
            return
        identity_source = self._legacy_identity_select()
        collision = connection.execute(
            f"""
            SELECT 1
            FROM sl_registry_runs AS registry
            JOIN ({identity_source}) AS legacy
              ON legacy.legacy_run_id = registry.run_id
            WHERE registry.sequence <= ?
              AND legacy.legacy_rowid <= ?
              AND legacy.legacy_run_id_safe = 1
            LIMIT 1
            """,
            (registry_snapshot, legacy_snapshot),
        ).fetchone()
        if collision is not None:
            raise IntegrityError("run identity is ambiguous across registry and legacy state")

    def _assert_unique_legacy_run_ids(
        self,
        connection: sqlite3.Connection,
        snapshot: int,
    ) -> None:
        """Fail closed when an untrusted legacy table has ambiguous run identities."""

        if not self._legacy_table_exists(connection):
            return
        identity_source = self._legacy_identity_select()
        duplicate = connection.execute(
            f"""
            SELECT 1
            FROM ({identity_source}) AS legacy
            WHERE legacy_rowid <= ? AND legacy_run_id_safe = 1
            GROUP BY legacy_run_id
            HAVING count(*) > 1
            LIMIT 1
            """,
            (snapshot,),
        ).fetchone()
        if duplicate is not None:
            raise IntegrityError("legacy run identity is ambiguous")

    @staticmethod
    def _encode_cursor_filter(query: RunQuery, legacy_fingerprint: str) -> str:
        if legacy_fingerprint != "-" and not re.fullmatch(r"[0-9a-f]{64}", legacy_fingerprint):
            raise IntegrityError("legacy snapshot fingerprint is malformed")
        result = query.cursor_identity + _LEGACY_CURSOR_MARKER + legacy_fingerprint
        if len(result.encode("utf-8")) > 128:
            raise IntegrityError("run cursor filter exceeds its contract bound")
        return result

    @staticmethod
    def _decode_cursor_filter(value: str | None) -> tuple[str, str]:
        if value is None or _LEGACY_CURSOR_MARKER not in value:
            raise InvalidCursorError("run cursor is missing its legacy snapshot binding")
        query_identity, fingerprint = value.rsplit(_LEGACY_CURSOR_MARKER, 1)
        if fingerprint != "-" and not re.fullmatch(r"[0-9a-f]{64}", fingerprint):
            raise InvalidCursorError("run cursor legacy snapshot binding is malformed")
        return query_identity, fingerprint

    def _list_registry_rows(
        self,
        connection: sqlite3.Connection,
        *,
        status: RunStatus | None,
        after: int,
        snapshot: int,
        limit: int,
    ) -> list[sqlite3.Row]:
        status_clause = "" if status is None else " AND status = ?"
        parameters: list[object] = [after, snapshot]
        if status is not None:
            parameters.append(status.value)
        parameters.append(limit)
        return connection.execute(
            "SELECT * FROM sl_registry_runs "
            "WHERE sequence > ? AND sequence <= ?"
            f"{status_clause} ORDER BY sequence LIMIT ?",
            parameters,
        ).fetchall()

    def _list_legacy_rows(
        self,
        connection: sqlite3.Connection,
        *,
        status: RunStatus | None,
        after: int,
        snapshot: int,
        limit: int,
    ) -> list[sqlite3.Row]:
        if not self._legacy_table_exists(connection):
            return []
        status_clause = ""
        parameters: list[object] = [after, snapshot]
        if status is not None:
            legacy_values = {
                RunStatus.SUCCEEDED: ("completed", "succeeded"),
                RunStatus.FAILED: ("failed",),
                RunStatus.CANCELLED: ("cancelled",),
            }[status]
            placeholders = ",".join("?" for _ in legacy_values)
            status_clause = f" AND lower(legacy_status) IN ({placeholders})"
            parameters.extend(legacy_values)
        parameters.append(limit)
        return connection.execute(
            "SELECT * FROM ("
            + self._legacy_select()
            + ") AS legacy WHERE legacy_rowid > ? AND legacy_rowid <= ?"
            + status_clause
            + " ORDER BY legacy_rowid LIMIT ?",
            # The byte-limit placeholders occur in the SELECT before the
            # keyset/filter placeholders in the WHERE clause.
            self._legacy_select_parameters() + parameters,
        ).fetchall()

    def _get_legacy_row(self, connection: sqlite3.Connection, run_id: str) -> sqlite3.Row | None:
        if not self._legacy_table_exists(connection):
            return None
        self._assert_safe_legacy_scalar_cells(connection, snapshot=None)
        rows = connection.execute(
            "SELECT * FROM ("
            + self._legacy_select()
            + ") AS legacy WHERE legacy_run_id = ? LIMIT 2",
            self._legacy_select_parameters() + [run_id],
        ).fetchall()
        if len(rows) > 1:
            raise IntegrityError("legacy run identity is ambiguous")
        return None if not rows else cast(sqlite3.Row, rows[0])

    def _assert_safe_legacy_scalar_cells(
        self,
        connection: sqlite3.Connection,
        *,
        snapshot: int | None,
    ) -> None:
        """Reject malformed legacy scalars before equality, grouping, or case folding."""

        if not self._legacy_table_exists(connection):
            return
        conditions = tuple(
            self._safe_legacy_text_condition(column, maximum, nullable=nullable)
            for column, maximum, nullable in (
                ("run_id", 128, False),
                ("started_at", 64, False),
                ("ended_at", 64, True),
                ("status", 32, False),
                ("git_commit", 64, True),
                ("data_hash", 128, True),
            )
        )
        scope = "" if snapshot is None else "rowid <= ? AND "
        parameters: tuple[object, ...] = () if snapshot is None else (snapshot,)
        unsafe = connection.execute(
            "SELECT 1 FROM runs WHERE "
            + scope
            + "("
            + " OR ".join(f"NOT ({condition})" for condition in conditions)
            + ") LIMIT 1",
            parameters,
        ).fetchone()
        if unsafe is not None:
            raise IntegrityError("legacy scalar field is malformed or exceeds its byte bound")

    @classmethod
    def _legacy_identity_select(cls) -> str:
        return (
            "SELECT rowid AS legacy_rowid, "
            + cls._safe_legacy_text("run_id", "legacy_run_id", 128)
            + ", "
            + cls._safe_legacy_text_flag(
                "run_id",
                "legacy_run_id_safe",
                128,
            )
            + " FROM runs"
        )

    def _legacy_select(self) -> str:
        json_columns = ("tickers", "features", "params", "metrics", "tags", "artifacts")
        projections = ["rowid AS legacy_rowid"]
        for column, maximum, nullable in (
            ("run_id", 128, False),
            ("started_at", 64, False),
            ("ended_at", 64, True),
            ("status", 32, False),
            ("git_commit", 64, True),
            ("data_hash", 128, True),
        ):
            alias = f"legacy_{column}"
            projections.extend(
                (
                    self._safe_legacy_text(
                        column,
                        alias,
                        maximum,
                        nullable=nullable,
                    ),
                    self._safe_legacy_text_flag(
                        column,
                        f"{alias}_safe",
                        maximum,
                        nullable=nullable,
                    ),
                )
            )
        for name in json_columns:
            projections.extend(
                (
                    f"CASE WHEN {name} IS NULL THEN NULL "
                    f"WHEN typeof({name}) = 'text' AND length(CAST({name} AS BLOB)) <= ? "
                    f"THEN {name} ELSE NULL END AS legacy_{name}",
                    f"CASE WHEN {name} IS NULL OR (typeof({name}) = 'text' "
                    f"AND length(CAST({name} AS BLOB)) <= ?) THEN 1 ELSE 0 END "
                    f"AS legacy_{name}_safe",
                )
            )
        return "SELECT " + ", ".join(projections) + " FROM runs"

    def _legacy_select_parameters(self) -> list[object]:
        return [self._limits.max_legacy_json_bytes] * 12

    @staticmethod
    def _safe_legacy_text(
        column: str,
        alias: str,
        maximum: int,
        *,
        nullable: bool = False,
    ) -> str:
        condition = RegistryReadPorts._safe_legacy_text_condition(
            column,
            maximum,
            nullable=nullable,
        )
        return f"CASE WHEN {condition} THEN {column} ELSE NULL END AS {alias}"

    @staticmethod
    def _safe_legacy_text_flag(
        column: str,
        alias: str,
        maximum: int,
        *,
        nullable: bool = False,
    ) -> str:
        condition = RegistryReadPorts._safe_legacy_text_condition(
            column,
            maximum,
            nullable=nullable,
        )
        return f"CASE WHEN {condition} THEN 1 ELSE 0 END AS {alias}"

    @staticmethod
    def _safe_legacy_text_condition(
        column: str,
        maximum: int,
        *,
        nullable: bool = False,
    ) -> str:
        bounded_text = (
            f"(typeof({column}) = 'text' "
            f"AND length(CAST({column} AS BLOB)) BETWEEN 1 AND {maximum})"
        )
        return f"({column} IS NULL OR {bounded_text})" if nullable else bounded_text

    @staticmethod
    def _legacy_table_exists(connection: sqlite3.Connection) -> bool:
        return (
            connection.execute(
                "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = 'runs'"
            ).fetchone()
            is not None
        )

    def _validate_page_size(self, value: int) -> None:
        if type(value) is not int or not 1 <= value <= self._limits.max_page_size:
            raise ValidationError(f"page_size must be in [1, {self._limits.max_page_size}]")


def _stored_int(value: object, field_name: str) -> int:
    return require_stored_int(value, field_name, -(2**63), 2**63 - 1)


def _sequence_maximum(row: sqlite3.Row, *, field_name: str, maximum: int) -> int:
    """Validate a full keyset source range and return its safe high-water mark."""

    minimum_raw = row["minimum"]
    maximum_raw = row["maximum"]
    if minimum_raw is None and maximum_raw is None:
        return 0
    if minimum_raw is None or maximum_raw is None:
        raise IntegrityError(f"stored {field_name} range is malformed")
    minimum = _stored_int(minimum_raw, f"{field_name} minimum")
    observed_maximum = _stored_int(maximum_raw, f"{field_name} maximum")
    if minimum < 1 or observed_maximum > maximum or minimum > observed_maximum:
        raise IntegrityError(f"stored {field_name} exceeds its supported snapshot range")
    return observed_maximum


def _stored_text(value: object, field_name: str, maximum: int) -> str:
    if type(value) is not str:
        raise IntegrityError(f"stored {field_name} is not bounded text")
    try:
        size = require_utf8_size(value, field_name)
    except ValidationError:
        raise IntegrityError(f"stored {field_name} is not bounded text") from None
    if not 1 <= size <= maximum:
        raise IntegrityError(f"stored {field_name} is not bounded text")
    return value


def _stored_time(value: object, field_name: str) -> datetime:
    return parse_stored_utc(value, field_name)


def _optional_stored_time(value: object, field_name: str) -> datetime | None:
    return None if value is None else _stored_time(value, field_name)


def _legacy_stored_time(value: object, field_name: str) -> datetime:
    """Parse only exact historic Python-ISO UTC text emitted by the legacy adapter."""

    text = _stored_text(value, field_name, 64)
    try:
        parsed = datetime.fromisoformat(text)
    except (MemoryError, ValueError):
        raise IntegrityError(f"stored {field_name} is not canonical legacy UTC text") from None
    if parsed.tzinfo is None or parsed.utcoffset() != UTC.utcoffset(parsed):
        raise IntegrityError(f"stored {field_name} is not canonical legacy UTC text")
    if parsed.isoformat() != text:
        raise IntegrityError(f"stored {field_name} is not canonical legacy UTC text")
    return parsed.astimezone(UTC)


def _optional_legacy_stored_time(value: object, field_name: str) -> datetime | None:
    return None if value is None else _legacy_stored_time(value, field_name)


def _safe_commit(value: object) -> str | None:
    if type(value) is not str or not _COMMIT.fullmatch(value):
        return None
    return value.lower()


def _safe_data_identity(value: object) -> str | None:
    if type(value) is not str or not _DATA_DIGEST.fullmatch(value):
        return None
    return value


def _legacy_status(value: object) -> RunStatus | None:
    if type(value) is not str:
        return None
    normalized = value.lower()
    if normalized in {"completed", "succeeded"}:
        return RunStatus.SUCCEEDED
    if normalized == "failed":
        return RunStatus.FAILED
    if normalized == "cancelled":
        return RunStatus.CANCELLED
    return None
