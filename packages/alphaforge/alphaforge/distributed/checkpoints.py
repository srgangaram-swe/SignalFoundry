"""Atomic versioned checkpoints that refuse to resume into a different world.

SF-S5-MR9. A checkpoint is only useful if resuming from it produces the run that
would have happened without the interruption. That requires binding more than
progress: the code, the data, the configuration, the dependency set, the seed,
and the task graph all have to be the same, because changing any of them makes
the resumed run a *different experiment wearing the original's name*.

So a :class:`CheckpointManifest` binds all six as content hashes, and
:func:`verify_resumable` refuses on any mismatch — naming which binding broke.
The work item's non-goal is "resuming from semantically incompatible state"; a
manifest that only recorded completed task IDs would satisfy the letter of
"checkpointing" while permitting exactly that.

**Writes are atomic.** A checkpoint is written to a temporary sibling and then
`os.replace`d into position, which is atomic within a filesystem. A crash
mid-write leaves the previous checkpoint intact, never a truncated one that
verifies as corrupt and destroys a resumable run.

**Concurrent writers are detected, not merged.** Two processes checkpointing one
experiment is not a supported configuration, and silently interleaving their
progress would produce a manifest describing work no single run performed. The
writer identity is recorded and a foreign writer is refused.
"""

from __future__ import annotations

import hashlib
import json
import os
import stat
import tempfile
import uuid
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Final

from alphaforge.distributed.tasks import TaskSpec

#: Bump only with a deliberate migration. A silently migrated checkpoint is how
#: a resumed run acquires semantics nobody chose.
CHECKPOINT_SCHEMA_VERSION: Final = 1

#: Refusal thresholds, not tuning knobs.
MAX_COMPLETED_TASKS: Final = 200_000
MAX_MANIFEST_BYTES: Final = 64 * 1024 * 1024
MAX_IDENTIFIER_BYTES: Final = 256

#: The bindings a resumed run must match. Named so a mismatch says which one.
REQUIRED_BINDINGS: Final[tuple[str, ...]] = (
    "code_hash",
    "data_hash",
    "config_hash",
    "dependency_hash",
    "task_graph_hash",
)


class CheckpointError(ValueError):
    """Raised when a checkpoint cannot be written or trusted."""


class CheckpointCorruptError(CheckpointError):
    """Raised when a stored checkpoint fails its integrity check."""


class CheckpointIncompatibleError(CheckpointError):
    """Raised when a checkpoint describes a different experiment.

    Distinct from corruption: the record is intact and describes work that is not
    the work about to resume.
    """


class ConcurrentWriterError(CheckpointError):
    """Raised when a second writer is detected for one experiment."""


def _canonical(payload: Any) -> bytes:
    return json.dumps(
        payload, sort_keys=True, separators=(",", ":"), allow_nan=False, ensure_ascii=True
    ).encode("utf-8")


def _digest(payload: Any) -> str:
    return hashlib.sha256(_canonical(payload)).hexdigest()


def _identifier(value: object, *, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip() or value != value.strip():
        raise CheckpointError(f"{field_name} must be a non-empty unpadded string")
    if len(value.encode("utf-8")) > MAX_IDENTIFIER_BYTES:
        raise CheckpointError(f"{field_name} exceeds {MAX_IDENTIFIER_BYTES} bytes")
    return value


def _hash_field(value: object, *, field_name: str) -> str:
    if not isinstance(value, str) or len(value) != 64:
        raise CheckpointError(f"{field_name} must be a full SHA-256 hex digest")
    if any(character not in "0123456789abcdef" for character in value):
        raise CheckpointError(f"{field_name} must be lowercase hexadecimal")
    return value


def task_graph_hash(tasks: Sequence[TaskSpec]) -> str:
    """Return a content hash over the whole task graph.

    Order-independent: the graph is the *set* of work, and a caller reordering
    submission has not changed the experiment. Sorting by identity first means
    the hash answers "is this the same work?" rather than "was it submitted the
    same way?".
    """
    return _digest(sorted(task.task_id for task in tasks))


@dataclass(frozen=True)
class CheckpointManifest:
    """Everything a resumed run must match, plus what has already finished.

    Raises:
        CheckpointError: On a malformed binding or an oversized completed set.
    """

    experiment_id: str
    schema_version: int
    code_hash: str
    data_hash: str
    config_hash: str
    dependency_hash: str
    task_graph_hash: str
    root_seed: int
    completed_task_ids: tuple[str, ...]
    writer_id: str
    written_at: datetime
    content_hash: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "experiment_id", _identifier(self.experiment_id, field_name="experiment_id")
        )
        object.__setattr__(self, "writer_id", _identifier(self.writer_id, field_name="writer_id"))
        if self.schema_version != CHECKPOINT_SCHEMA_VERSION:
            raise CheckpointIncompatibleError(
                f"checkpoint schema version {self.schema_version} does not match the running "
                f"version {CHECKPOINT_SCHEMA_VERSION}; a silently migrated checkpoint is how "
                "a resumed run acquires semantics nobody chose"
            )
        for field_name in REQUIRED_BINDINGS:
            object.__setattr__(
                self, field_name, _hash_field(getattr(self, field_name), field_name=field_name)
            )
        if isinstance(self.root_seed, bool) or not isinstance(self.root_seed, int):
            raise CheckpointError("root_seed must be an int")
        if self.root_seed < 0:
            raise CheckpointError("root_seed must be non-negative")
        completed = tuple(self.completed_task_ids)
        if len(completed) > MAX_COMPLETED_TASKS:
            raise CheckpointError(f"exceeds the {MAX_COMPLETED_TASKS}-task ceiling")
        if len(set(completed)) != len(completed):
            raise CheckpointError(
                "duplicate task id in completed set; a task counted twice would let the "
                "resumed run skip work it never did"
            )
        object.__setattr__(self, "completed_task_ids", tuple(sorted(completed)))
        if not isinstance(self.written_at, datetime) or self.written_at.tzinfo is None:
            raise CheckpointError("written_at must be a timezone-aware datetime")
        object.__setattr__(self, "written_at", self.written_at.astimezone(UTC))
        if not self.content_hash:
            object.__setattr__(self, "content_hash", _digest(self.bindings_payload()))
        else:
            _hash_field(self.content_hash, field_name="content_hash")

    def bindings_payload(self) -> dict[str, Any]:
        """Return exactly the fields the content hash covers."""
        return {
            "experiment_id": self.experiment_id,
            "schema_version": self.schema_version,
            "code_hash": self.code_hash,
            "data_hash": self.data_hash,
            "config_hash": self.config_hash,
            "dependency_hash": self.dependency_hash,
            "task_graph_hash": self.task_graph_hash,
            "root_seed": self.root_seed,
            "completed_task_ids": list(self.completed_task_ids),
            "writer_id": self.writer_id,
            "written_at": self.written_at.isoformat(),
        }

    def verify_integrity(self) -> None:
        """Refuse a manifest whose contents no longer match its hash.

        Raises:
            CheckpointCorruptError: On a mismatch.
        """
        recomputed = _digest(self.bindings_payload())
        if recomputed != self.content_hash:
            raise CheckpointCorruptError(
                f"checkpoint for {self.experiment_id!r} failed its integrity check: stored "
                f"{self.content_hash[:12]}, recomputed {recomputed[:12]}"
            )

    def remaining(self, tasks: Sequence[TaskSpec]) -> tuple[TaskSpec, ...]:
        """Return the tasks still to run, in identity order."""
        done = set(self.completed_task_ids)
        return tuple(sorted((t for t in tasks if t.task_id not in done), key=lambda t: t.task_id))

    def to_dict(self) -> dict[str, Any]:
        """Return the complete JSON-friendly manifest."""
        payload = self.bindings_payload()
        payload["content_hash"] = self.content_hash
        payload["completed_count"] = len(self.completed_task_ids)
        return payload


def verify_resumable(
    manifest: CheckpointManifest,
    *,
    code_hash: str,
    data_hash: str,
    config_hash: str,
    dependency_hash: str,
    tasks: Sequence[TaskSpec],
    root_seed: int,
) -> None:
    """Refuse to resume unless every binding matches.

    Checks integrity first, then each binding in turn, naming the one that broke.
    A resumed run whose code, data, configuration, dependencies, seed, or task
    graph differs is a different experiment wearing the original's name.

    Raises:
        CheckpointCorruptError: If the manifest fails its integrity check.
        CheckpointIncompatibleError: On any binding mismatch, naming which.
    """
    manifest.verify_integrity()
    observed = {
        "code_hash": code_hash,
        "data_hash": data_hash,
        "config_hash": config_hash,
        "dependency_hash": dependency_hash,
        "task_graph_hash": task_graph_hash(tasks),
    }
    explanations = {
        "code_hash": "the implementation changed, so resumed tasks would run different logic",
        "data_hash": "the inputs changed, so completed and remaining work saw different data",
        "config_hash": "the configuration changed, so the experiment's parameters are not the "
        "ones the completed work used",
        "dependency_hash": "the dependency set changed, so numerical results may differ between "
        "the completed and remaining halves",
        "task_graph_hash": "the set of work changed; completed ids may no longer correspond to "
        "tasks in this graph",
    }
    mismatched = [name for name in REQUIRED_BINDINGS if observed[name] != getattr(manifest, name)]
    if mismatched:
        details = "; ".join(
            f"{name} (stored {getattr(manifest, name)[:12]}, observed {observed[name][:12]}) — "
            f"{explanations[name]}"
            for name in mismatched
        )
        raise CheckpointIncompatibleError(
            f"cannot resume experiment {manifest.experiment_id!r}: {len(mismatched)} binding(s) "
            f"changed. {details}. Resuming would produce a different experiment under the "
            "original's name."
        )
    if root_seed != manifest.root_seed:
        raise CheckpointIncompatibleError(
            f"root seed changed from {manifest.root_seed} to {root_seed}; the remaining tasks "
            "would draw from a different stream than the completed ones"
        )
    unknown = set(manifest.completed_task_ids) - {task.task_id for task in tasks}
    if unknown:
        raise CheckpointIncompatibleError(
            f"{len(unknown)} completed task id(s) are absent from the supplied graph, e.g. "
            f"{sorted(unknown)[:3]}; the checkpoint describes work this batch does not contain"
        )


class CheckpointStore:
    """Atomic checkpoint persistence for one experiment.

    Writes go to a temporary sibling and are then `os.replace`d into position,
    which is atomic within a filesystem: a crash mid-write leaves the previous
    checkpoint intact rather than a truncated one.

    Raises:
        CheckpointError: On an unsafe directory.
    """

    def __init__(self, directory: str | os.PathLike[str], *, writer_id: str | None = None) -> None:
        path = Path(directory).expanduser()
        if path.exists():
            info = path.lstat()
            if stat.S_ISLNK(info.st_mode) or not stat.S_ISDIR(info.st_mode):
                raise CheckpointError("checkpoint directory must be a real directory")
            if hasattr(os, "getuid") and info.st_uid != os.getuid():
                raise CheckpointError("checkpoint directory must be owned by the current user")
        else:
            path.mkdir(parents=True, exist_ok=True)
            path.chmod(0o700)
        self._directory = path
        self._writer_id = writer_id or f"writer-{uuid.uuid4().hex[:16]}"

    @property
    def writer_id(self) -> str:
        """This store's writer identity."""
        return self._writer_id

    def _path_for(self, experiment_id: str) -> Path:
        return self._directory / f"{_identifier(experiment_id, field_name='experiment_id')}.json"

    def write(self, manifest: CheckpointManifest, *, allow_foreign_writer: bool = False) -> None:
        """Persist a checkpoint atomically.

        Args:
            manifest: The checkpoint to store.
            allow_foreign_writer: Permit overwriting a checkpoint written by a
                different writer. Defaults to ``False`` because two processes
                checkpointing one experiment is unsupported, and interleaving
                their progress would describe work no single run performed.

        Raises:
            ConcurrentWriterError: On a foreign writer without explicit consent.
            CheckpointError: On an oversized payload.
        """
        target = self._path_for(manifest.experiment_id)
        if target.exists() and not allow_foreign_writer:
            existing = self.read(manifest.experiment_id)
            if existing is not None and existing.writer_id != manifest.writer_id:
                raise ConcurrentWriterError(
                    f"experiment {manifest.experiment_id!r} was last checkpointed by writer "
                    f"{existing.writer_id!r}, not {manifest.writer_id!r}. Two writers on one "
                    "experiment would produce a manifest describing work no single run "
                    "performed."
                )
        encoded = _canonical(manifest.to_dict())
        if len(encoded) > MAX_MANIFEST_BYTES:
            raise CheckpointError(
                f"manifest is {len(encoded)} bytes, exceeding the {MAX_MANIFEST_BYTES} ceiling"
            )
        handle, temporary = tempfile.mkstemp(
            dir=self._directory, prefix=".checkpoint-", suffix=".tmp"
        )
        try:
            with os.fdopen(handle, "wb") as stream:
                stream.write(encoded)
                stream.flush()
                os.fsync(stream.fileno())
            os.chmod(temporary, 0o600)
            os.replace(temporary, target)
        except Exception:
            Path(temporary).unlink(missing_ok=True)
            raise

    def read(self, experiment_id: str) -> CheckpointManifest | None:
        """Return the stored checkpoint, or ``None`` when absent.

        Raises:
            CheckpointCorruptError: On unparseable or truncated content.
        """
        target = self._path_for(experiment_id)
        if not target.exists():
            return None
        try:
            payload = json.loads(target.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, UnicodeDecodeError) as error:
            raise CheckpointCorruptError(
                f"checkpoint for {experiment_id!r} is not readable JSON: {error}. A truncated "
                "checkpoint is not a partially usable one."
            ) from error
        return manifest_from_payload(payload)

    def clear(self, experiment_id: str) -> bool:
        """Delete a checkpoint. Returns whether one existed."""
        target = self._path_for(experiment_id)
        if target.exists():
            target.unlink()
            return True
        return False


def manifest_from_payload(payload: Mapping[str, Any]) -> CheckpointManifest:
    """Rebuild a manifest from its stored payload.

    Raises:
        CheckpointCorruptError: On missing fields.
    """
    if not isinstance(payload, Mapping):
        raise CheckpointCorruptError("checkpoint payload is not an object")
    required = {
        "experiment_id",
        "schema_version",
        "root_seed",
        "completed_task_ids",
        "writer_id",
        "written_at",
        "content_hash",
        *REQUIRED_BINDINGS,
    }
    missing = required - set(payload)
    if missing:
        raise CheckpointCorruptError(
            f"checkpoint payload is missing {sorted(missing)}; a truncated record is not a "
            "partially usable one"
        )
    return CheckpointManifest(
        experiment_id=str(payload["experiment_id"]),
        schema_version=int(payload["schema_version"]),
        code_hash=str(payload["code_hash"]),
        data_hash=str(payload["data_hash"]),
        config_hash=str(payload["config_hash"]),
        dependency_hash=str(payload["dependency_hash"]),
        task_graph_hash=str(payload["task_graph_hash"]),
        root_seed=int(payload["root_seed"]),
        completed_task_ids=tuple(payload["completed_task_ids"]),
        writer_id=str(payload["writer_id"]),
        written_at=datetime.fromisoformat(str(payload["written_at"])).astimezone(UTC),
        content_hash=str(payload["content_hash"]),
    )


__all__ = [
    "CHECKPOINT_SCHEMA_VERSION",
    "MAX_COMPLETED_TASKS",
    "REQUIRED_BINDINGS",
    "CheckpointCorruptError",
    "CheckpointError",
    "CheckpointIncompatibleError",
    "CheckpointManifest",
    "CheckpointStore",
    "ConcurrentWriterError",
    "manifest_from_payload",
    "task_graph_hash",
    "verify_resumable",
]
