"""Experiment tracker implementations and the public :func:`get_tracker` factory."""

from __future__ import annotations

import math
import os
import re
import sqlite3
import stat
import time
import uuid
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Final, cast

from quant_platform.config import TrackingConfig
from quant_platform.logging_utils import get_logger
from quant_platform.tracking.contracts import (
    ValidationError,
    canonical_json,
    decode_bounded_json,
)
from quant_platform.utils import ensure_dir, git_commit_hash, resolve_path

logger = get_logger(__name__)

_MAX_LEGACY_LIST_LIMIT: Final = 1_000
_MAX_LEGACY_JSON_SCAN: Final = 10_000
_MAX_LEGACY_RECORD_BYTES: Final = 512 * 1024
_MAX_LEGACY_JSON_FIELD_BYTES: Final = 65_536
_MAX_LEGACY_RESULT_BYTES: Final = 16 * 1024 * 1024
_MAX_LEGACY_SQLITE_VALUE_BYTES: Final = 1024 * 1024
_LEGACY_SQLITE_BUSY_TIMEOUT_MS: Final = 500
_LEGACY_SQLITE_QUERY_TIMEOUT_MS: Final = 500
_LEGACY_SQLITE_PROGRESS_INSTRUCTIONS: Final = 1_000
_LEGACY_JSON_COLUMNS: Final = (
    "tickers",
    "features",
    "params",
    "metrics",
    "tags",
    "artifacts",
)
_LEGACY_RECORD_COLUMNS: Final = (
    "run_id",
    "experiment",
    "name",
    "started_at",
    "ended_at",
    "status",
    "git_commit",
    "data_hash",
    *_LEGACY_JSON_COLUMNS,
)
_LEGACY_TEXT_LIMITS: Final = {
    "run_id": 128,
    "experiment": 256,
    "name": 256,
    "started_at": 64,
    "ended_at": 64,
    "status": 32,
    "git_commit": 128,
    "data_hash": 512,
}
_LEGACY_TERMINAL_STATUSES: Final = frozenset(
    {"running", "completed", "succeeded", "failed", "cancelled"}
)
_SAFE_LEGACY_DISPLAY: Final = re.compile(r"^[A-Za-z0-9][A-Za-z0-9 .,_:+()@'=-]*$")
_SECRETISH_LEGACY_WORD: Final = re.compile(
    r"(?i)\b(api[ _-]?key|authorization|bearer|password|private[ _-]?key|secret|token)\b"
)
_SECRETISH_LEGACY_TOKEN: Final = re.compile(r"[A-Za-z0-9_-]{32,}")
_SAFE_LEGACY_DIGEST: Final = re.compile(r"^(?:sha256:)?[0-9a-f]{64}$")
_SAFE_LEGACY_RUN_ID: Final = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")


class LegacyTrackingError(RuntimeError):
    """Stable, redacted failure at the bounded legacy compatibility boundary."""

    code = "legacy_tracking_error"


class LegacyTrackingReadError(LegacyTrackingError):
    """A legacy record could not be read within its integrity/resource contract."""

    code = "legacy_tracking_read_error"


class LegacyTrackingWriteError(LegacyTrackingError):
    """A legacy record could not be serialized within its persistence contract."""

    code = "legacy_tracking_write_error"


def _add_legacy_failure_note(primary_error: BaseException, note: str) -> None:
    """Attach bounded diagnostics without letting custom exceptions replace the primary."""

    try:
        primary_error.add_note(note)
    except Exception:
        return


@dataclass
class RunContext:
    """Accumulates everything logged during a single experiment run."""

    run_id: str
    experiment: str
    name: str
    started_at: str
    params: dict[str, Any] = field(default_factory=dict)
    metrics: dict[str, float] = field(default_factory=dict)
    tags: dict[str, Any] = field(default_factory=dict)
    artifacts: list[str] = field(default_factory=list)
    git_commit: str | None = None
    data_hash: str | None = None
    tickers: list[str] = field(default_factory=list)
    features: list[str] = field(default_factory=list)
    status: str = "running"

    # -- logging helpers (return self for chaining) --------------------------
    def log_params(self, params: dict[str, Any]) -> RunContext:
        try:
            flattened = _flatten(params)
        except (MemoryError, RecursionError, TypeError, ValueError, ValidationError):
            raise LegacyTrackingWriteError(
                "legacy parameters violate their bounded persistence contract"
            ) from None
        self.params.update(flattened)
        return self

    def log_metrics(self, metrics: dict[str, Any]) -> RunContext:
        for k, v in metrics.items():
            try:
                self.metrics[k] = float(v)
            except (TypeError, ValueError):
                self.tags[k] = v
        return self

    def log_tags(self, tags: dict[str, Any]) -> RunContext:
        self.tags.update(tags)
        return self

    def log_artifact(self, path: str | Path) -> RunContext:
        self.artifacts.append(str(path))
        return self

    def set_dataset(self, *, data_hash: str, tickers: list[str], features: list[str]) -> RunContext:
        self.data_hash = data_hash
        self.tickers = list(tickers)
        self.features = list(features)
        return self

    def to_record(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "experiment": self.experiment,
            "name": self.name,
            "started_at": self.started_at,
            "ended_at": datetime.now(UTC).isoformat(),
            "status": self.status,
            "git_commit": self.git_commit,
            "data_hash": self.data_hash,
            "tickers": self.tickers,
            "features": self.features,
            "params": self.params,
            "metrics": self.metrics,
            "tags": self.tags,
            "artifacts": self.artifacts,
        }


class ExperimentTracker:
    """Base tracker interface (also serves as the no-op tracker)."""

    backend = "none"

    def __init__(self, config: TrackingConfig, *, base_dir: str | None = None) -> None:
        self.config = config
        self.base_dir = base_dir

    @contextmanager
    def run(self, name: str) -> Iterator[RunContext]:
        """Yield one run and persist its terminal state exactly once.

        Persistence is a cleanup boundary.  When the run body has already
        failed, a second failure while recording that state must not replace
        the original domain exception.  The original exception receives a
        bounded note naming only the persistence error type; exception text is
        deliberately omitted because filesystem and provider errors can
        contain credentials or absolute workstation paths.  Conversely, a
        persistence failure after a successful run is raised: callers must not
        mistake an unrecorded run for durable success.
        """

        ctx = RunContext(
            run_id=uuid.uuid4().hex[:12],
            experiment=self.config.experiment_name,
            name=name,
            started_at=datetime.now(UTC).isoformat(),
            git_commit=git_commit_hash(),
        )
        logger.info("[%s] started run id=%s", self.backend, _safe_legacy_log_id(ctx.run_id))
        primary_error: BaseException | None = None
        try:
            yield ctx
            ctx.status = "completed"
        except BaseException as exc:
            ctx.status = "failed"
            primary_error = exc
            raise
        finally:
            persisted = _persist_preserving_primary(
                backend=self.backend,
                ctx=ctx,
                operation=lambda: self._persist(ctx),
                primary_error=primary_error,
            )
            logger.info(
                "[%s] finished run id=%s status=%s persisted=%s",
                self.backend,
                _safe_legacy_log_id(ctx.run_id),
                ctx.status,
                persisted,
            )

    def _persist(self, ctx: RunContext) -> None:  # pragma: no cover - no-op base
        pass

    def list_runs(self, *, limit: int | None = None) -> list[dict[str, Any]]:
        """Return at most ``limit`` legacy records without an unbounded read."""

        _legacy_list_limit(limit)
        return []


class JSONTracker(ExperimentTracker):
    """Persist each run as a JSON document under ``json_dir``."""

    backend = "json"

    def _dir(self, *, create: bool) -> Path:
        path = resolve_path(self.config.json_dir, self.base_dir)
        return ensure_dir(path) if create else path

    def _persist(self, ctx: RunContext) -> None:
        record = ctx.to_record()
        document, _ = _serialize_legacy_record(record)
        run_id = cast(str, record["run_id"])
        started_date = _legacy_timestamp(record["started_at"], role="started_at").date().isoformat()
        try:
            # Human-supplied run names belong inside the document, never in a pathname.
            path = self._dir(create=True) / f"{started_date}_{run_id}.json"
            with path.open("x", encoding="utf-8") as fh:
                fh.write(document)
                fh.write("\n")
        except (OSError, UnicodeError):
            raise LegacyTrackingWriteError("legacy JSON record persistence failed") from None
        logger.debug("Wrote legacy JSON experiment record id=%s", _safe_legacy_log_id(ctx.run_id))

    def list_runs(self, *, limit: int | None = None) -> list[dict[str, Any]]:
        query_limit = _legacy_list_limit(limit)
        directory = self._dir(create=False)
        flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_DIRECTORY", 0)
        flags |= getattr(os, "O_NOFOLLOW", 0)
        try:
            directory_descriptor = os.open(directory, flags)
        except FileNotFoundError:
            return []
        except OSError:
            raise LegacyTrackingReadError(
                "legacy experiment JSON root cannot be opened safely"
            ) from None
        try:
            before = os.fstat(directory_descriptor)
            if not stat.S_ISDIR(before.st_mode):
                raise LegacyTrackingReadError("legacy experiment JSON root is not a directory")
            try:
                entries: list[str] = []
                scanned = 0
                with os.scandir(directory_descriptor) as iterator:
                    for entry in iterator:
                        scanned += 1
                        if scanned > _MAX_LEGACY_JSON_SCAN:
                            raise LegacyTrackingReadError(
                                "legacy experiment JSON root exceeds its bounded scan limit"
                            )
                        if entry.name.endswith(".json"):
                            entries.append(entry.name)
                entries.sort()
            except LegacyTrackingReadError:
                raise
            except OSError:
                raise LegacyTrackingReadError(
                    "legacy experiment JSON root cannot be enumerated safely"
                ) from None
            runs: list[dict[str, Any]] = []
            aggregate_bytes = 0
            for name in entries[:query_limit]:
                payload = _read_legacy_json_record(directory_descriptor, name)
                aggregate_bytes = _add_legacy_result_bytes(aggregate_bytes, len(payload))
                runs.append(_decode_legacy_record(payload))
            after = os.fstat(directory_descriptor)
            if _filesystem_identity(before) != _filesystem_identity(after):
                raise LegacyTrackingReadError(
                    "legacy experiment JSON root changed during bounded listing"
                )
            return runs
        finally:
            os.close(directory_descriptor)


class SQLiteTracker(ExperimentTracker):
    """Persist runs into a single SQLite database (default backend)."""

    backend = "sqlite"

    def _db_path(self, *, create_parent: bool) -> Path:
        path = resolve_path(self.config.db_path, self.base_dir)
        if create_parent:
            ensure_dir(path.parent)
        return path

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(
            self._db_path(create_parent=True),
            timeout=_LEGACY_SQLITE_BUSY_TIMEOUT_MS / 1_000,
        )
        try:
            conn.execute(f"PRAGMA busy_timeout={_LEGACY_SQLITE_BUSY_TIMEOUT_MS}")
            conn.execute("""
                CREATE TABLE IF NOT EXISTS runs (
                    run_id TEXT PRIMARY KEY,
                    experiment TEXT,
                    name TEXT,
                    started_at TEXT,
                    ended_at TEXT,
                    status TEXT,
                    git_commit TEXT,
                    data_hash TEXT,
                    tickers TEXT,
                    features TEXT,
                    params TEXT,
                    metrics TEXT,
                    tags TEXT,
                    artifacts TEXT
                )
                """)
            return conn
        except BaseException as primary_error:
            try:
                conn.close()
            except sqlite3.Error as cleanup_error:
                _add_legacy_failure_note(
                    primary_error,
                    "legacy SQLite setup cleanup also failed: " f"{type(cleanup_error).__name__}",
                )
            raise

    def _persist(self, ctx: RunContext) -> None:
        rec = ctx.to_record()
        _, json_fields = _serialize_legacy_record(rec)
        conn: sqlite3.Connection | None = None
        primary_error: BaseException | None = None
        try:
            conn = self._connect()
            with conn:
                conn.execute(
                    """
                    INSERT INTO runs
                    (run_id, experiment, name, started_at, ended_at, status, git_commit,
                     data_hash, tickers, features, params, metrics, tags, artifacts)
                    VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                    """,
                    (
                        rec["run_id"],
                        rec["experiment"],
                        rec["name"],
                        rec["started_at"],
                        rec["ended_at"],
                        rec["status"],
                        rec["git_commit"],
                        rec["data_hash"],
                        json_fields["tickers"],
                        json_fields["features"],
                        json_fields["params"],
                        json_fields["metrics"],
                        json_fields["tags"],
                        json_fields["artifacts"],
                    ),
                )
        except LegacyTrackingWriteError as exc:
            primary_error = exc
            raise
        except (OSError, sqlite3.Error):
            mapped = LegacyTrackingWriteError("legacy SQLite record persistence failed")
            primary_error = mapped
            raise mapped from None
        finally:
            if conn is not None:
                try:
                    conn.close()
                except sqlite3.Error as exc:
                    if primary_error is not None:
                        _add_legacy_failure_note(
                            primary_error,
                            "legacy SQLite persistence cleanup also failed: "
                            f"{type(exc).__name__}",
                        )
                    else:
                        raise LegacyTrackingWriteError(
                            "legacy SQLite persistence cleanup failed"
                        ) from None
        logger.debug(
            "Inserted legacy SQLite experiment record id=%s",
            _safe_legacy_log_id(ctx.run_id),
        )

    def list_runs(self, *, limit: int | None = None) -> list[dict[str, Any]]:
        query_limit = _legacy_list_limit(limit)
        path = self._db_path(create_parent=False)
        try:
            before = path.lstat()
        except FileNotFoundError:
            return []
        except OSError:
            raise LegacyTrackingReadError(
                "legacy experiment database metadata cannot be inspected"
            ) from None
        if not stat.S_ISREG(before.st_mode):
            raise LegacyTrackingReadError(
                "legacy experiment database must be a regular non-symlink file"
            )
        connection: sqlite3.Connection | None = None
        primary_error: BaseException | None = None
        try:
            connection = sqlite3.connect(
                f"{path.absolute().as_uri()}?mode=ro",
                uri=True,
                timeout=_LEGACY_SQLITE_BUSY_TIMEOUT_MS / 1_000,
            )
            connection.row_factory = sqlite3.Row
            connection.execute("PRAGMA query_only=ON")
            connection.execute(f"PRAGMA busy_timeout={_LEGACY_SQLITE_BUSY_TIMEOUT_MS}")
            connection.setlimit(sqlite3.SQLITE_LIMIT_LENGTH, _MAX_LEGACY_SQLITE_VALUE_BYTES)
            deadline = time.monotonic_ns() + _LEGACY_SQLITE_QUERY_TIMEOUT_MS * 1_000_000
            connection.set_progress_handler(
                lambda: int(time.monotonic_ns() >= deadline),
                _LEGACY_SQLITE_PROGRESS_INSTRUCTIONS,
            )
            table = connection.execute(
                "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = 'runs'"
            ).fetchone()
            if table is None:
                return []
            cursor = connection.execute(
                _legacy_sqlite_projection()
                + " ORDER BY "
                + _safe_legacy_sqlite_text("started_at", 64)
                + " DESC, rowid DESC LIMIT ?",
                (query_limit,),
            )
            rows: list[dict[str, Any]] = []
            aggregate_bytes = 0
            while row := cursor.fetchone():
                _validate_legacy_sqlite_projection(row)
                raw = {column: row[column] for column in _LEGACY_RECORD_COLUMNS}
                aggregate_bytes = _add_legacy_result_bytes(
                    aggregate_bytes,
                    _legacy_sqlite_record_bytes(raw),
                )
                rows.append(_decode_legacy_sqlite_record(raw))
            return rows
        except LegacyTrackingError as exc:
            primary_error = exc
            raise
        except sqlite3.Error as exc:
            primary_code = getattr(exc, "sqlite_errorcode", -1) & 0xFF
            if primary_code == sqlite3.SQLITE_INTERRUPT:
                mapped = LegacyTrackingReadError(
                    "legacy experiment database read exceeded its deadline"
                )
            elif primary_code in {sqlite3.SQLITE_BUSY, sqlite3.SQLITE_LOCKED}:
                mapped = LegacyTrackingReadError(
                    "legacy experiment database remained busy past its deadline"
                )
            else:
                mapped = LegacyTrackingReadError("legacy experiment database read failed")
            primary_error = mapped
            raise mapped from None
        except OSError:
            mapped = LegacyTrackingReadError(
                "legacy experiment database changed during bounded listing"
            )
            primary_error = mapped
            raise mapped from None
        finally:
            cleanup_failures: list[BaseException] = []
            if connection is not None:
                try:
                    connection.set_progress_handler(None, 0)
                except sqlite3.Error as exc:
                    cleanup_failures.append(exc)
                try:
                    connection.close()
                except sqlite3.Error as exc:
                    cleanup_failures.append(exc)
            try:
                after = path.lstat()
                if _filesystem_identity(before) != _filesystem_identity(after):
                    raise LegacyTrackingReadError(
                        "legacy experiment database changed during bounded listing"
                    )
            except (FileNotFoundError, LegacyTrackingReadError, OSError) as exc:
                cleanup_failures.append(exc)
            if cleanup_failures:
                if primary_error is not None:
                    names = ",".join(type(error).__name__ for error in cleanup_failures)
                    _add_legacy_failure_note(
                        primary_error,
                        f"legacy read cleanup also failed: {names}",
                    )
                else:
                    raise LegacyTrackingReadError("legacy experiment read cleanup failed") from None


class MLflowTracker(ExperimentTracker):
    """Optional MLflow-backed tracker."""

    backend = "mlflow"

    @contextmanager
    def run(self, name: str) -> Iterator[RunContext]:  # pragma: no cover - optional dep
        try:
            import mlflow
        except ImportError as exc:
            raise ImportError(
                "MLflow backend requested but mlflow is not installed "
                "(`pip install '.[mlflow]'`)."
            ) from exc
        if self.config.mlflow_tracking_uri:
            mlflow.set_tracking_uri(self.config.mlflow_tracking_uri)
        mlflow.set_experiment(self.config.experiment_name)
        ctx = RunContext(
            run_id=uuid.uuid4().hex[:12],
            experiment=self.config.experiment_name,
            name=name,
            started_at=datetime.now(UTC).isoformat(),
            git_commit=git_commit_hash(),
        )
        logger.info("[%s] started run id=%s", self.backend, _safe_legacy_log_id(ctx.run_id))
        primary_error: BaseException | None = None
        try:
            with mlflow.start_run(run_name=name):
                try:
                    yield ctx
                    ctx.status = "completed"
                except BaseException as exc:
                    ctx.status = "failed"
                    primary_error = exc
                    raise
                finally:
                    persisted = _persist_preserving_primary(
                        backend=self.backend,
                        ctx=ctx,
                        operation=lambda: self._log_mlflow_run(mlflow, ctx),
                        primary_error=primary_error,
                    )
                    logger.info(
                        "[%s] finished run id=%s status=%s persisted=%s",
                        self.backend,
                        _safe_legacy_log_id(ctx.run_id),
                        ctx.status,
                        persisted,
                    )
        except BaseException as session_error:
            if primary_error is None or session_error is primary_error:
                raise
            _annotate_secondary_failure(
                backend=self.backend,
                ctx=ctx,
                primary_error=primary_error,
                secondary_error=session_error,
                phase="session finalization",
            )
            raise primary_error from None
        if primary_error is not None:
            # A third-party context manager must never suppress the run body's
            # exception.  Restore it even if the integration violates the
            # normal context-manager contract.
            _add_legacy_failure_note(
                primary_error,
                "mlflow session finalization suppressed the run exception",
            )
            logger.error(
                "[%s] session finalization suppressed the run exception for id=%s; "
                "restoring the original exception",
                self.backend,
                _safe_legacy_log_id(ctx.run_id),
            )
            raise primary_error from None

    @staticmethod
    def _log_mlflow_run(mlflow: Any, ctx: RunContext) -> None:
        """Publish the legacy MLflow record while its run context is active."""

        mlflow.log_params(ctx.params)
        mlflow.log_metrics(dict(ctx.metrics))
        mlflow.set_tags(
            {
                **ctx.tags,
                "git_commit": ctx.git_commit or "",
                "data_hash": ctx.data_hash or "",
            }
        )
        for artifact in ctx.artifacts:
            if Path(artifact).exists():
                mlflow.log_artifact(artifact)


def get_tracker(config: TrackingConfig, *, base_dir: str | None = None) -> ExperimentTracker:
    """Factory returning the configured tracker backend."""
    backend = config.backend
    if backend == "sqlite":
        return SQLiteTracker(config, base_dir=base_dir)
    if backend == "json":
        return JSONTracker(config, base_dir=base_dir)
    if backend == "mlflow":
        return MLflowTracker(config, base_dir=base_dir)
    return ExperimentTracker(config, base_dir=base_dir)


def _persist_preserving_primary(
    *,
    backend: str,
    ctx: RunContext,
    operation: Callable[[], None],
    primary_error: BaseException | None,
) -> bool:
    """Execute persistence without allowing cleanup to replace a run error.

    Returns ``True`` when persistence completed.  If no primary exception is
    active, any persistence exception propagates unchanged.  If the run body
    already failed, only the secondary exception's type is attached and
    logged; its potentially sensitive message is never emitted.
    """

    try:
        operation()
    except BaseException as persistence_error:
        if primary_error is None:
            raise
        _annotate_secondary_failure(
            backend=backend,
            ctx=ctx,
            primary_error=primary_error,
            secondary_error=persistence_error,
            phase="persistence",
        )
        return False
    return True


def _annotate_secondary_failure(
    *,
    backend: str,
    ctx: RunContext,
    primary_error: BaseException,
    secondary_error: BaseException,
    phase: str,
) -> None:
    """Record a bounded cleanup failure without exposing its message."""

    error_type = type(secondary_error).__name__
    _add_legacy_failure_note(
        primary_error,
        f"{backend} experiment {phase} also failed: {error_type}",
    )
    logger.error(
        "[%s] experiment %s failed for id=%s while the run exception was active; "
        "preserving the original exception (secondary_error=%s)",
        backend,
        phase,
        _safe_legacy_log_id(ctx.run_id),
        error_type,
    )


def _serialize_legacy_record(record: dict[str, Any]) -> tuple[str, dict[str, str]]:
    """Validate and serialize one complete legacy record before storage mutation."""

    try:
        document = _validated_legacy_document(record)
        fields: dict[str, str] = {}
        for name in _LEGACY_JSON_COLUMNS:
            encoded = canonical_json(record[name])
            if len(encoded.encode("utf-8")) > _MAX_LEGACY_JSON_FIELD_BYTES:
                raise ValidationError("legacy JSON field exceeds its byte ceiling")
            fields[name] = encoded
        return document, fields
    except (KeyError, TypeError, ValueError, UnicodeError, ValidationError):
        raise LegacyTrackingWriteError(
            "legacy experiment record violates its bounded persistence contract"
        ) from None


def _validated_legacy_document(record: object) -> str:
    """Return canonical JSON after exact shape, scalar, and aggregate validation."""

    if type(record) is not dict:
        raise ValidationError("legacy record must be a built-in object")
    typed = cast(dict[object, object], record)
    if set(typed) != set(_LEGACY_RECORD_COLUMNS):
        raise ValidationError("legacy record has an unexpected shape")
    if type(typed["run_id"]) is not str or _SAFE_LEGACY_RUN_ID.fullmatch(typed["run_id"]) is None:
        raise ValidationError("legacy run_id is not filename-safe")

    for name, maximum in _LEGACY_TEXT_LIMITS.items():
        value = typed[name]
        if name in {"ended_at", "git_commit", "data_hash"} and value is None:
            continue
        _bounded_legacy_text(value, maximum=maximum)

    if typed["status"] not in _LEGACY_TERMINAL_STATUSES:
        raise ValidationError("legacy status is outside the closed compatibility set")
    started_at = _legacy_timestamp(typed["started_at"], role="started_at")
    ended_value = typed["ended_at"]
    if ended_value is not None:
        ended_at = _legacy_timestamp(ended_value, role="ended_at")
        if ended_at < started_at:
            raise ValidationError("legacy ended_at precedes started_at")

    for name in ("tickers", "features", "artifacts"):
        value = typed[name]
        if value is None:
            continue
        if type(value) is not list:
            raise ValidationError("legacy sequence field has an unexpected type")
        item_limit = _MAX_LEGACY_JSON_FIELD_BYTES if name == "artifacts" else 1_024
        for item in cast(list[object], value):
            _bounded_legacy_text(item, maximum=item_limit, allow_empty=True)

    for name in ("params", "tags"):
        value = typed[name]
        if value is not None and type(value) is not dict:
            raise ValidationError("legacy object field has an unexpected type")

    metrics = typed["metrics"]
    if metrics is not None:
        if type(metrics) is not dict:
            raise ValidationError("legacy metrics have an unexpected type")
        for key, metric in cast(dict[object, object], metrics).items():
            _bounded_legacy_text(key, maximum=256)
            if type(metric) not in {int, float} or not math.isfinite(cast(int | float, metric)):
                raise ValidationError("legacy metrics must contain finite numeric scalars")

    document = canonical_json(record)
    if len(document.encode("utf-8")) > _MAX_LEGACY_RECORD_BYTES:
        raise ValidationError("legacy record exceeds its aggregate byte ceiling")
    return document


def _bounded_legacy_text(value: object, *, maximum: int, allow_empty: bool = False) -> str:
    if type(value) is not str:
        raise ValidationError("legacy text field has an unexpected type")
    try:
        size = len(value.encode("utf-8"))
    except UnicodeError:
        raise ValidationError("legacy text field is not valid UTF-8") from None
    minimum = 0 if allow_empty else 1
    if not minimum <= size <= maximum or "\x00" in value:
        raise ValidationError("legacy text field exceeds its bounded contract")
    return value


def _strict_legacy_json(payload: bytes | str, *, maximum_bytes: int) -> object:
    return decode_bounded_json(
        payload,
        maximum_bytes=maximum_bytes,
    )


def _decode_legacy_record(payload: bytes) -> dict[str, Any]:
    try:
        decoded = _strict_legacy_json(payload, maximum_bytes=_MAX_LEGACY_RECORD_BYTES)
        _validated_legacy_document(decoded)
        return _legacy_surface_projection(cast(dict[str, Any], decoded))
    except (
        RecursionError,
        TypeError,
        ValueError,
        UnicodeError,
        ValidationError,
    ):
        raise LegacyTrackingReadError(
            "legacy JSON record violates its bounded compatibility contract"
        ) from None


def _decode_legacy_sqlite_record(raw: dict[str, object]) -> dict[str, Any]:
    try:
        decoded = dict(raw)
        for name in _LEGACY_JSON_COLUMNS:
            value = decoded[name]
            decoded[name] = (
                None
                if value is None
                else _strict_legacy_json(
                    cast(str, value), maximum_bytes=_MAX_LEGACY_JSON_FIELD_BYTES
                )
            )
        _validated_legacy_document(decoded)
        return _legacy_surface_projection(cast(dict[str, Any], decoded))
    except (
        RecursionError,
        TypeError,
        ValueError,
        UnicodeError,
        ValidationError,
    ):
        raise LegacyTrackingReadError(
            "legacy SQLite record violates its bounded compatibility contract"
        ) from None


def _read_legacy_json_record(directory_descriptor: int, name: str) -> bytes:
    if type(name) is not str or not name.endswith(".json") or name in {".", ".."} or "/" in name:
        raise LegacyTrackingReadError("legacy JSON record name is not canonical")
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(name, flags, dir_fd=directory_descriptor)
    except OSError:
        raise LegacyTrackingReadError("legacy JSON record cannot be opened safely") from None
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode) or before.st_nlink != 1:
            raise LegacyTrackingReadError("legacy JSON record must be a contained regular file")
        if not 0 < before.st_size <= _MAX_LEGACY_RECORD_BYTES:
            raise LegacyTrackingReadError("legacy JSON record exceeds its byte ceiling")
        chunks: list[bytes] = []
        observed = 0
        while observed <= _MAX_LEGACY_RECORD_BYTES:
            chunk = os.read(
                descriptor,
                min(65_536, _MAX_LEGACY_RECORD_BYTES + 1 - observed),
            )
            if not chunk:
                break
            chunks.append(chunk)
            observed += len(chunk)
        if observed != before.st_size or observed > _MAX_LEGACY_RECORD_BYTES:
            raise LegacyTrackingReadError("legacy JSON record changed or exceeds its byte ceiling")
        after = os.fstat(descriptor)
        named = os.stat(name, dir_fd=directory_descriptor, follow_symlinks=False)
        if _filesystem_identity(before) != _filesystem_identity(after) or _filesystem_identity(
            after
        ) != _filesystem_identity(named):
            raise LegacyTrackingReadError("legacy JSON record changed during bounded read")
        return b"".join(chunks)
    except OSError:
        raise LegacyTrackingReadError("legacy JSON record read failed") from None
    finally:
        os.close(descriptor)


def _filesystem_identity(value: os.stat_result) -> tuple[int, int, int, int, int, int]:
    return (
        value.st_dev,
        value.st_ino,
        stat.S_IFMT(value.st_mode),
        value.st_size,
        value.st_mtime_ns,
        value.st_ctime_ns,
    )


def _legacy_timestamp(value: object, *, role: str) -> datetime:
    text = _bounded_legacy_text(value, maximum=64)
    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
        offset = parsed.utcoffset()
    except (OverflowError, ValueError):
        raise ValidationError(f"legacy {role} is not a safe ISO-8601 timestamp") from None
    if parsed.tzinfo is None or offset is None:
        raise ValidationError(f"legacy {role} is not timezone-aware")
    try:
        return parsed.astimezone(UTC)
    except (OverflowError, ValueError):
        raise ValidationError(f"legacy {role} cannot be normalized safely") from None


def _legacy_surface_projection(record: dict[str, Any]) -> dict[str, Any]:
    """Redact terminal-facing legacy text without mutating persisted evidence."""

    projected = dict(record)
    projected["name"] = _safe_legacy_display(record["name"], role="name")
    if record["data_hash"] is not None:
        projected["data_hash"] = _safe_legacy_display(record["data_hash"], role="data_hash")
    return projected


def _safe_legacy_display(value: object, *, role: str) -> str:
    """Return bounded one-line public text or one fixed redaction marker."""

    if type(value) is not str:
        return "[redacted]"
    if role == "data_hash" and _SAFE_LEGACY_DIGEST.fullmatch(value):
        return value
    try:
        encoded = value.encode("ascii")
    except UnicodeError:
        return "[redacted]"
    if (
        not encoded
        or len(encoded) > _LEGACY_TEXT_LIMITS[role]
        or _SAFE_LEGACY_DISPLAY.fullmatch(value) is None
        or "/" in value
        or "\\" in value
        or _SECRETISH_LEGACY_WORD.search(value) is not None
        or _SECRETISH_LEGACY_TOKEN.search(value) is not None
    ):
        return "[redacted]"
    return value


def _safe_legacy_log_id(value: object) -> str:
    """Keep mutable caller state from injecting paths or controls into logs."""

    if type(value) is str and _SAFE_LEGACY_RUN_ID.fullmatch(value):
        return value
    return "[redacted]"


def _add_legacy_result_bytes(current: int, added: int) -> int:
    if current < 0 or added < 0 or current > _MAX_LEGACY_RESULT_BYTES - added:
        raise LegacyTrackingReadError("legacy experiment result exceeds its aggregate byte ceiling")
    return current + added


def _safe_legacy_sqlite_text(column: str, maximum: int, *, nullable: bool = False) -> str:
    null_clause = f"{column} IS NULL OR " if nullable else ""
    return (
        f"CASE WHEN {null_clause}(typeof({column}) = 'text' "
        f"AND length(CAST({column} AS BLOB)) BETWEEN 1 AND {maximum}) "
        f"THEN {column} ELSE NULL END"
    )


def _legacy_sqlite_projection() -> str:
    projections: list[str] = []
    for name in _LEGACY_RECORD_COLUMNS:
        if name in _LEGACY_JSON_COLUMNS:
            safe = (
                f"{name} IS NULL OR (typeof({name}) = 'text' "
                f"AND length(CAST({name} AS BLOB)) BETWEEN 1 "
                f"AND {_MAX_LEGACY_JSON_FIELD_BYTES})"
            )
            projections.append(f"CASE WHEN {safe} THEN {name} ELSE NULL END AS {name}")
            projections.append(f"CASE WHEN {safe} THEN 1 ELSE 0 END AS __safe_{name}")
        else:
            nullable = name in {"ended_at", "git_commit", "data_hash"}
            safe_text = _safe_legacy_sqlite_text(
                name,
                _LEGACY_TEXT_LIMITS[name],
                nullable=nullable,
            )
            projections.append(f"{safe_text} AS {name}")
            valid = (f"{name} IS NULL OR " if nullable else "") + (
                f"(typeof({name}) = 'text' AND length(CAST({name} AS BLOB)) "
                f"BETWEEN 1 AND {_LEGACY_TEXT_LIMITS[name]})"
            )
            projections.append(f"CASE WHEN {valid} THEN 1 ELSE 0 END AS __safe_{name}")
    return "SELECT " + ", ".join(projections) + " FROM runs"


def _validate_legacy_sqlite_projection(row: sqlite3.Row) -> None:
    for name in _LEGACY_RECORD_COLUMNS:
        if row[f"__safe_{name}"] != 1:
            raise LegacyTrackingReadError(
                "legacy SQLite record contains an unsafe field representation"
            )


def _legacy_sqlite_record_bytes(record: dict[str, object]) -> int:
    total = 0
    try:
        for value in record.values():
            if value is not None:
                if type(value) is not str:
                    raise LegacyTrackingReadError(
                        "legacy SQLite record contains an unsupported storage type"
                    )
                encoded = value.encode("utf-8")
                if total > _MAX_LEGACY_RECORD_BYTES - len(encoded):
                    raise LegacyTrackingReadError(
                        "legacy SQLite record exceeds its aggregate byte ceiling"
                    )
                total += len(encoded)
        return total
    except UnicodeError:
        raise LegacyTrackingReadError("legacy SQLite record is not valid UTF-8") from None


def _legacy_list_limit(limit: int | None) -> int:
    """Validate the compatibility adapter's bounded list query."""

    if limit is None:
        return _MAX_LEGACY_LIST_LIMIT
    if type(limit) is not int:
        raise TypeError("limit must be an integer or None")
    if not 1 <= limit <= _MAX_LEGACY_LIST_LIMIT:
        raise ValueError(f"limit must be between 1 and {_MAX_LEGACY_LIST_LIMIT}")
    return limit


def _flatten(d: dict[str, Any], parent: str = "", sep: str = ".") -> dict[str, Any]:
    """Flatten validated built-in JSON objects without recursive call-stack growth."""

    if type(d) is not dict or type(parent) is not str or type(sep) is not str or not sep:
        raise ValidationError("legacy parameter flattening requires built-in bounded values")
    # Validate depth, nodes, finite numerics, strings, and exact container types before producing
    # any output. This keeps log_params transactional even for adversarial input.
    canonical_json(d)
    items: dict[str, Any] = {}
    pending: list[tuple[str, str, object]] = [
        (parent, key, value) for key, value in reversed(tuple(d.items()))
    ]
    while pending:
        current_parent, component, value = pending.pop()
        key = f"{current_parent}{sep}{component}" if current_parent else component
        if type(value) is dict:
            nested = cast(dict[str, object], value)
            pending.extend(
                (key, nested_key, nested_value)
                for nested_key, nested_value in reversed(tuple(nested.items()))
            )
        elif type(value) in {list, tuple}:
            if key in items:
                raise ValidationError("flattened legacy parameter keys are ambiguous")
            items[key] = canonical_json(value)
        else:
            if key in items:
                raise ValidationError("flattened legacy parameter keys are ambiguous")
            items[key] = value
    return items
