"""Content-addressed, transactional AlphaForge Sprint 5 evidence.

The close-out plot is a view over two independently verifiable inputs: repeated
raw benchmark samples and frozen Git objects.  No performance observation or
delivery count is duplicated as a source constant.  Publication is bounded,
staged beside the final directory, verified byte-for-byte, and committed by one
directory rename.  It never connects to a provider, broker, or credential store.
"""

from __future__ import annotations

import ctypes
import errno
import hashlib
import importlib.metadata
import io
import json
import math
import os
import platform as platform_module
import re
import secrets
import stat
import sys
import warnings
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Final, NoReturn

import matplotlib

matplotlib.use("Agg")
import matplotlib.ft2font as ft2font
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from alphaforge.distributed.benchmark_evidence import (
    BenchmarkEvidence,
    BenchmarkEvidenceError,
    parse_benchmark_evidence_bytes,
    require_production_implementation,
    verify_production_implementation_sources,
)
from alphaforge.readiness.capital import ABSOLUTE_MAX_CAPITAL, LiveCapitalConfig
from alphaforge.readiness.checklist import (
    ReadinessDecision,
    evaluate_readiness,
    minimal_capital_checklist,
    render_readiness_report,
)
from alphaforge.readiness.sprint_5_inventory import (
    DeliveryInventory,
    Sprint5InventoryError,
    build_sprint_5_inventory,
)
from alphaforge.research._bounded_io import (
    BoundedIOError,
    RegularFileSnapshot,
    parse_strict_json,
    read_regular_file_snapshot,
)

EVIDENCE_SCHEMA_VERSION: Final = "2.0.0"
MAX_INPUT_BYTES: Final = 8 * 1024 * 1024
MAX_ARTIFACT_BYTES: Final = 24 * 1024 * 1024
MAX_BUNDLE_BYTES: Final = 48 * 1024 * 1024
MAX_ARTIFACTS: Final = 8
MAX_JSON_DEPTH: Final = 64
MAX_JSON_NODES: Final = 100_000
MAX_TEXT_CHARS: Final = 2048

_SHA256: Final = re.compile(r"^[0-9a-f]{64}$")
_SHA1: Final = re.compile(r"^[0-9a-f]{40}$")
_LOCK_BYTES: Final = b"AlphaForge Sprint 5 evidence publication in progress\n"
_MANIFEST_INTEGRITY: Final = (
    "The manifest does not recursively hash itself. Git anchors manifest.json; "
    "this index hashes every other artifact."
)
_LIMITATIONS: Final = (
    "Synthetic deterministic workload and Git metadata only; no market data or broker connection.",
    "Single-machine wall-clock measurements are environment-specific and are not a performance "
    "SLA or a cluster benchmark.",
    "Per-slice changed-path record counts describe frozen Git scope; a repository path can appear "
    "in more than one slice, and the counts are not tests, quality, effort, or semantic complexity "
    "metrics.",
    "The readiness framework records but cannot independently verify human policy or legal "
    "attestations.",
    "No strategy is qualified; no paper/live trading or capital is authorized.",
)

_GENERATOR_SOURCE_PATHS: Final = (
    "alphaforge/distributed/benchmark_evidence.py",
    "alphaforge/distributed/executor.py",
    "alphaforge/distributed/tasks.py",
    "alphaforge/readiness/capital.py",
    "alphaforge/readiness/checklist.py",
    "alphaforge/readiness/sprint_5_evidence.py",
    "alphaforge/readiness/sprint_5_inventory.py",
    "alphaforge/research/_bounded_io.py",
    "benchmarks/benchmark_distributed_crossover.py",
    "pyproject.toml",
    "scripts/publish_sprint_5_evidence.py",
    "uv.lock",
)
_RENDER_DEPENDENCIES: Final = ("matplotlib", "numpy", "pandas", "Pillow", "seaborn")

_PAYLOAD_ARTIFACTS: Final = frozenset(
    {
        "checklist.json",
        "delivery_inventory.json",
        "distribution_crossover.csv",
        "distribution_crossover_raw.json",
        "readiness_decision.json",
        "readiness_report.md",
        "sprint_5_closeout.png",
    }
)


class Sprint5EvidenceError(ValueError):
    """Raised when Sprint 5 evidence is unsafe, inconsistent, or incomplete."""


class Sprint5PostCommitError(Sprint5EvidenceError):
    """Raised after the bundle rename committed but final durability failed."""


class Sprint5LimitedVerificationWarning(UserWarning):
    """Warn that repository/Git/source provenance was not externally checked."""


@dataclass(frozen=True, slots=True)
class _Identity:
    """Non-following device/inode/type identity for one filesystem entry."""

    device: int
    inode: int
    directory: bool


@dataclass(slots=True)
class _ParentAnchor:
    """One validated repository-contained parent held open for the transaction."""

    path: Path
    descriptor: int
    identity: _Identity


@dataclass(slots=True)
class _OwnedFile:
    """One file created relative to an anchored directory."""

    name: str
    identity: _Identity
    descriptor: int = -1


@dataclass(slots=True)
class _StagingDirectory:
    """One owned staging directory and its flat owned artifact set."""

    name: str
    descriptor: int
    identity: _Identity
    artifacts: dict[str, _OwnedFile]


@dataclass(frozen=True, slots=True)
class _GeneratorSourceSnapshot:
    """Exact bytes and stable filesystem identity for one generator source."""

    path: Path
    data: bytes
    sha256: str
    device: int
    inode: int
    size: int
    modified_ns: int
    changed_ns: int


def _mapping(value: object, expected: set[str], *, field: str) -> Mapping[str, Any]:
    """Return an exact string-keyed object."""

    if not isinstance(value, dict) or any(not isinstance(key, str) for key in value):
        raise Sprint5EvidenceError(f"{field} must be an object with string keys")
    observed = set(value)
    missing = sorted(expected - observed)
    unknown = sorted(observed - expected)
    if missing or unknown:
        raise Sprint5EvidenceError(f"{field} keys differ: missing={missing}, unknown={unknown}")
    return value


def _integer(value: object, *, field: str, minimum: int, maximum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise Sprint5EvidenceError(f"{field} must be an integer, not a bool")
    if not minimum <= value <= maximum:
        raise Sprint5EvidenceError(f"{field} must lie in [{minimum}, {maximum}]")
    return value


def _text(value: object, *, field: str, maximum: int = MAX_TEXT_CHARS) -> str:
    if (
        not isinstance(value, str)
        or not value
        or value != value.strip()
        or len(value) > maximum
        or any(ord(character) < 32 or ord(character) == 127 for character in value)
    ):
        raise Sprint5EvidenceError(f"{field} must be bounded, non-empty, trimmed text")
    return value


def _digest(value: object, *, field: str, sha1: bool = False) -> str:
    pattern = _SHA1 if sha1 else _SHA256
    if not isinstance(value, str) or pattern.fullmatch(value) is None:
        kind = "SHA-1" if sha1 else "SHA-256"
        raise Sprint5EvidenceError(f"{field} must be a lowercase {kind} digest")
    return value


def _identity(metadata: os.stat_result, *, directory: bool, field: str) -> _Identity:
    expected = stat.S_ISDIR(metadata.st_mode) if directory else stat.S_ISREG(metadata.st_mode)
    if not expected:
        kind = "directory" if directory else "regular file"
        raise Sprint5EvidenceError(f"{field} must be a {kind}")
    return _Identity(device=metadata.st_dev, inode=metadata.st_ino, directory=directory)


def _same_identity(metadata: os.stat_result, expected: _Identity) -> bool:
    expected_type = (
        stat.S_ISDIR(metadata.st_mode) if expected.directory else stat.S_ISREG(metadata.st_mode)
    )
    return (
        expected_type and metadata.st_dev == expected.device and metadata.st_ino == expected.inode
    )


def _leaf_name(value: str, *, field: str) -> str:
    if (
        not isinstance(value, str)
        or not value
        or value in {".", ".."}
        or Path(value).name != value
        or len(os.fsencode(value)) > 200
        or "\x00" in value
        or any(ord(character) < 32 or ord(character) == 127 for character in value)
    ):
        raise Sprint5EvidenceError(f"{field} must be a bounded single path component")
    return value


def _stat_at(directory_descriptor: int, name: str) -> os.stat_result:
    return os.stat(
        _leaf_name(name, field="anchored entry name"),
        dir_fd=directory_descriptor,
        follow_symlinks=False,
    )


def _assert_identity_at(
    directory_descriptor: int,
    name: str,
    expected: _Identity,
    *,
    field: str,
) -> None:
    try:
        metadata = _stat_at(directory_descriptor, name)
    except FileNotFoundError as exc:
        raise Sprint5EvidenceError(f"{field} disappeared") from exc
    if not _same_identity(metadata, expected):
        raise Sprint5EvidenceError(f"{field} changed identity")


def _entry_exists_at(directory_descriptor: int, name: str) -> bool:
    try:
        _stat_at(directory_descriptor, name)
    except FileNotFoundError:
        return False
    return True


def _open_parent_anchor(repository: Path, parent: Path) -> _ParentAnchor:
    """Open and retain one real repository-contained destination parent."""

    try:
        repository_real = repository.resolve(strict=True)
        parent_real = parent.resolve(strict=True)
        parent_real.relative_to(repository_real)
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        raise Sprint5EvidenceError(
            "evidence destination parent must resolve inside repository_root"
        ) from exc
    if parent_real != parent:
        raise Sprint5EvidenceError(
            "evidence destination parent must not depend on symlink resolution"
        )
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_DIRECTORY", 0)
    flags |= getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(parent, flags)
    except OSError as exc:
        raise Sprint5EvidenceError("unable to anchor evidence destination parent") from exc
    try:
        descriptor_metadata = os.fstat(descriptor)
        pathname_metadata = os.stat(parent, follow_symlinks=False)
        identity = _identity(
            descriptor_metadata,
            directory=True,
            field="evidence destination parent",
        )
        if not _same_identity(pathname_metadata, identity):
            raise Sprint5EvidenceError("evidence destination parent changed during anchoring")
    except BaseException:
        os.close(descriptor)
        raise
    return _ParentAnchor(path=parent, descriptor=descriptor, identity=identity)


def _assert_parent_anchor_path(anchor: _ParentAnchor) -> None:
    """Confirm the public parent pathname still names the retained directory."""

    try:
        metadata = os.stat(anchor.path, follow_symlinks=False)
    except OSError as exc:
        raise Sprint5EvidenceError("evidence destination parent disappeared") from exc
    if not _same_identity(metadata, anchor.identity):
        raise Sprint5EvidenceError("evidence destination parent changed identity")


def _close_descriptor(descriptor: int, failures: list[BaseException]) -> None:
    if descriptor < 0:
        return
    try:
        os.close(descriptor)
    except BaseException as exc:
        failures.append(exc)


def _raise_with_cleanup(
    message: str,
    primary: BaseException,
    cleanup_failures: list[BaseException],
) -> NoReturn:
    if cleanup_failures:
        raise BaseExceptionGroup(message, [primary, *cleanup_failures]) from primary
    raise primary.with_traceback(primary.__traceback__)


def sprint_5_readiness() -> ReadinessDecision:
    """Return the repository's fail-closed readiness decision at Sprint 5 close.

    Evidence and attestations are intentionally empty.  AlphaForge has no
    qualified strategy, completed paper interval, owner approval, or legal and
    policy attestations; absence therefore remains a blocking failure.
    """

    return evaluate_readiness(
        minimal_capital_checklist(),
        evidence={},
        attestations={},
        now=datetime(2026, 8, 8, tzinfo=UTC),
    )


def _json_bytes(payload: Any) -> bytes:
    """Return deterministic UTF-8 JSON, refusing non-finite values."""

    try:
        document = json.dumps(
            payload,
            allow_nan=False,
            ensure_ascii=True,
            indent=2,
            sort_keys=True,
        )
    except (TypeError, ValueError) as exc:
        raise Sprint5EvidenceError("evidence payload is not finite canonical JSON") from exc
    return (document + "\n").encode("utf-8")


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _write_descriptor(descriptor: int, data: bytes) -> None:
    """Write every byte to an already-open descriptor or fail on no progress."""

    view = memoryview(data)
    while view:
        written = os.write(descriptor, view)
        if written <= 0:
            raise OSError("bounded descriptor write made no forward progress")
        view = view[written:]


def _create_owned_file(
    directory_descriptor: int,
    name: str,
    *,
    mode: int,
) -> _OwnedFile:
    """Create one regular file relative to an anchored directory."""

    name = _leaf_name(name, field="owned file name")
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_CLOEXEC", 0)
    try:
        descriptor = os.open(name, flags, mode, dir_fd=directory_descriptor)
    except FileExistsError as exc:
        raise FileExistsError(f"publisher-owned file already exists: {name}") from exc
    owned: _OwnedFile | None = None
    try:
        descriptor_metadata = os.fstat(descriptor)
        identity = _identity(
            descriptor_metadata,
            directory=False,
            field=f"publisher-owned file {name}",
        )
        owned = _OwnedFile(name=name, identity=identity, descriptor=descriptor)
        _assert_identity_at(
            directory_descriptor,
            name,
            identity,
            field=f"publisher-owned file {name}",
        )
        return owned
    except BaseException as primary:
        cleanup_failures: list[BaseException] = []
        if owned is not None:
            try:
                _assert_identity_at(
                    directory_descriptor,
                    name,
                    owned.identity,
                    field=f"failed publisher-owned file {name}",
                )
                os.unlink(name, dir_fd=directory_descriptor)
            except BaseException as exc:
                cleanup_failures.append(exc)
        _close_descriptor(descriptor, cleanup_failures)
        _raise_with_cleanup(
            "owned-file creation and cleanup both failed",
            primary,
            cleanup_failures,
        )


def _write_bytes_at(staging: _StagingDirectory, name: str, data: bytes) -> None:
    """Create, flush, and identity-bind one flat staging artifact."""

    if not data or len(data) > MAX_ARTIFACT_BYTES:
        raise Sprint5EvidenceError(f"artifact bytes must be in [1, {MAX_ARTIFACT_BYTES}]: {name}")
    if name in staging.artifacts:
        raise Sprint5EvidenceError(f"staging artifact was already created: {name}")
    owned = _create_owned_file(staging.descriptor, name, mode=0o644)
    staging.artifacts[name] = owned
    primary: BaseException | None = None
    try:
        _write_descriptor(owned.descriptor, data)
        os.fsync(owned.descriptor)
    except BaseException as exc:
        primary = exc
    close_failures: list[BaseException] = []
    descriptor = owned.descriptor
    owned.descriptor = -1
    _close_descriptor(descriptor, close_failures)
    if primary is not None:
        _raise_with_cleanup(
            "artifact write and descriptor close both failed",
            primary,
            close_failures,
        )
    if close_failures:
        raise close_failures[0]
    _assert_identity_at(
        staging.descriptor,
        name,
        owned.identity,
        field=f"staged artifact {name}",
    )


def _acquire_publication_lock(anchor: _ParentAnchor, name: str) -> _OwnedFile:
    """Create and retain one anchored cooperative-writer lock descriptor."""

    try:
        lock = _create_owned_file(anchor.descriptor, name, mode=0o600)
    except FileExistsError as exc:
        raise FileExistsError(f"evidence publisher lock already exists: {name}") from exc
    try:
        _write_descriptor(lock.descriptor, _LOCK_BYTES)
        os.fsync(lock.descriptor)
        _assert_identity_at(
            anchor.descriptor,
            name,
            lock.identity,
            field="publication lock",
        )
        return lock
    except BaseException as primary:
        cleanup_failures = _cleanup_lock(anchor, lock)
        _raise_with_cleanup(
            "publication lock initialization and cleanup both failed",
            primary,
            cleanup_failures,
        )


def _create_staging(anchor: _ParentAnchor, destination_name: str) -> _StagingDirectory:
    """Create and retain one high-entropy staging directory under the anchor."""

    prefix = f".{destination_name}.staging."
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_DIRECTORY", 0)
    flags |= getattr(os, "O_NOFOLLOW", 0)
    for _ in range(32):
        name = _leaf_name(prefix + secrets.token_hex(16), field="staging directory name")
        try:
            os.mkdir(name, mode=0o700, dir_fd=anchor.descriptor)
        except FileExistsError:
            continue
        identity = _identity(
            _stat_at(anchor.descriptor, name),
            directory=True,
            field="staging directory",
        )
        descriptor = -1
        try:
            descriptor = os.open(name, flags, dir_fd=anchor.descriptor)
            if not _same_identity(os.fstat(descriptor), identity):
                raise Sprint5EvidenceError("staging directory changed during acquisition")
            return _StagingDirectory(
                name=name,
                descriptor=descriptor,
                identity=identity,
                artifacts={},
            )
        except BaseException as primary:
            cleanup_failures: list[BaseException] = []
            _close_descriptor(descriptor, cleanup_failures)
            try:
                _assert_identity_at(
                    anchor.descriptor,
                    name,
                    identity,
                    field="failed staging directory",
                )
                os.rmdir(name, dir_fd=anchor.descriptor)
            except BaseException as exc:
                cleanup_failures.append(exc)
            _raise_with_cleanup(
                "staging acquisition and cleanup both failed",
                primary,
                cleanup_failures,
            )
    raise Sprint5EvidenceError("unable to allocate a unique staging directory")


def _cleanup_lock(anchor: _ParentAnchor, lock: _OwnedFile | None) -> list[BaseException]:
    failures: list[BaseException] = []
    if lock is None:
        return failures
    try:
        _assert_identity_at(
            anchor.descriptor,
            lock.name,
            lock.identity,
            field="publication lock cleanup target",
        )
        os.unlink(lock.name, dir_fd=anchor.descriptor)
    except BaseException as exc:
        failures.append(exc)
    descriptor = lock.descriptor
    lock.descriptor = -1
    _close_descriptor(descriptor, failures)
    return failures


def _cleanup_staging(
    anchor: _ParentAnchor,
    staging: _StagingDirectory | None,
    *,
    committed: bool,
) -> list[BaseException]:
    """Remove only still-identical invocation-owned staging entries."""

    failures: list[BaseException] = []
    if staging is None:
        return failures
    can_remove_directory = not committed
    if not committed:
        for name, artifact in sorted(staging.artifacts.items()):
            descriptor = artifact.descriptor
            artifact.descriptor = -1
            _close_descriptor(descriptor, failures)
            try:
                _assert_identity_at(
                    staging.descriptor,
                    name,
                    artifact.identity,
                    field=f"staged artifact cleanup target {name}",
                )
                os.unlink(name, dir_fd=staging.descriptor)
            except BaseException as exc:
                failures.append(exc)
                can_remove_directory = False
        try:
            unexpected = os.listdir(staging.descriptor)
            if unexpected:
                raise Sprint5EvidenceError(
                    "staging cleanup refused unowned or identity-replaced entries: "
                    f"{sorted(unexpected)}"
                )
        except BaseException as exc:
            failures.append(exc)
            can_remove_directory = False
        try:
            _assert_identity_at(
                anchor.descriptor,
                staging.name,
                staging.identity,
                field="staging directory cleanup target",
            )
        except BaseException as exc:
            failures.append(exc)
            can_remove_directory = False
    if can_remove_directory:
        try:
            os.rmdir(staging.name, dir_fd=anchor.descriptor)
        except BaseException as exc:
            failures.append(exc)
    descriptor = staging.descriptor
    staging.descriptor = -1
    _close_descriptor(descriptor, failures)
    return failures


def _read_regular_file_snapshot_at(
    directory_descriptor: int,
    name: str,
    *,
    max_bytes: int,
    display_root: Path,
) -> RegularFileSnapshot:
    """Read one immutable flat-file snapshot relative to an open directory."""

    if isinstance(max_bytes, bool) or not isinstance(max_bytes, int) or max_bytes < 1:
        raise Sprint5EvidenceError("snapshot max_bytes must be a positive integer")
    name = _leaf_name(name, field="snapshot file name")
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    descriptor = -1
    try:
        descriptor = os.open(name, flags, dir_fd=directory_descriptor)
        initial = os.fstat(descriptor)
        identity = _identity(initial, directory=False, field=f"snapshot {name}")
        if not 0 < initial.st_size <= max_bytes:
            raise Sprint5EvidenceError(f"snapshot bytes must be in [1, {max_bytes}]: {name}")
        chunks: list[bytes] = []
        total = 0
        while True:
            chunk = os.read(descriptor, min(1024 * 1024, max_bytes - total + 1))
            if not chunk:
                break
            total += len(chunk)
            if total > max_bytes:
                raise Sprint5EvidenceError(f"snapshot exceeds {max_bytes} bytes: {name}")
            chunks.append(chunk)
        final = os.fstat(descriptor)
        stable_metadata = (
            final.st_size == initial.st_size
            and final.st_mtime_ns == initial.st_mtime_ns
            and final.st_ctime_ns == initial.st_ctime_ns
        )
        if not _same_identity(final, identity) or not stable_metadata:
            raise Sprint5EvidenceError(f"snapshot changed during read: {name}")
        _assert_identity_at(
            directory_descriptor,
            name,
            identity,
            field=f"snapshot pathname {name}",
        )
    except Sprint5EvidenceError:
        raise
    except OSError as exc:
        raise Sprint5EvidenceError(f"unable to read anchored snapshot: {name}") from exc
    finally:
        if descriptor >= 0:
            os.close(descriptor)
    data = b"".join(chunks)
    if len(data) != initial.st_size:
        raise Sprint5EvidenceError(f"snapshot length changed during read: {name}")
    return RegularFileSnapshot(
        path=display_root / name,
        data=data,
        sha256=_sha256(data),
    )


def _safe_repository(repository_root: str | Path) -> Path:
    lexical = Path(os.path.abspath(repository_root))
    try:
        resolved = lexical.resolve(strict=True)
    except (FileNotFoundError, RuntimeError) as exc:
        raise Sprint5EvidenceError("repository_root must be a real, non-symlink directory") from exc
    if resolved != lexical or lexical.is_symlink() or not lexical.is_dir():
        raise Sprint5EvidenceError("repository_root must be a real, non-symlink directory")
    marker = lexical / ".git"
    if marker.is_symlink() or not marker.exists():
        raise Sprint5EvidenceError("repository_root must identify a Git worktree")
    return lexical


def _inside_repository(repository: Path, path: str | Path, *, role: str) -> Path:
    requested = Path(path)
    lexical = requested if requested.is_absolute() else repository / requested
    lexical = Path(os.path.abspath(lexical))
    try:
        relative = lexical.relative_to(repository)
    except ValueError as exc:
        raise Sprint5EvidenceError(f"{role} must remain inside repository_root") from exc
    if ".git" in relative.parts:
        raise Sprint5EvidenceError(f"{role} must not enter Git administrative storage")
    current = repository
    for component in relative.parts:
        current /= component
        if current.exists() and current.is_symlink():
            raise Sprint5EvidenceError(f"{role} path must not traverse a symlink")
    return lexical


def _benchmark_frames(payload: dict[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Derive raw and summary frames solely from validated samples."""

    config = payload["config"]
    task_count = config["task_count"]
    samples = pd.DataFrame(payload["samples"])
    if samples.empty:
        raise Sprint5EvidenceError("benchmark evidence contains no measured samples")
    rows = samples.assign(
        serial_ms=samples["serial_ns"] / 1_000_000.0,
        pool_ms=samples["pool_ns"] / 1_000_000.0,
        per_task_ms=samples["serial_ns"] / float(task_count) / 1_000_000.0,
        speedup=samples["serial_ns"] / samples["pool_ns"],
    )
    derived = rows.loc[
        :,
        [
            "iterations",
            "repetition",
            "backend_order",
            "serial_ns",
            "pool_ns",
            "serial_ms",
            "pool_ms",
            "per_task_ms",
            "speedup",
            "parity",
        ],
    ].copy()
    if not derived["parity"].all():
        raise Sprint5EvidenceError("benchmark evidence contains a backend parity failure")
    numeric = derived.select_dtypes(include="number")
    if not numeric.map(lambda value: math.isfinite(float(value))).all().all():
        raise Sprint5EvidenceError("benchmark derivation produced a non-finite value")

    summaries: list[dict[str, Any]] = []
    for iterations, group in derived.groupby("iterations", sort=True):
        record: dict[str, Any] = {
            "benchmark_id": payload["benchmark_id"],
            "iterations": int(iterations),
            "sample_count": int(len(group)),
            "task_count": int(config["task_count"]),
            "workers": int(config["workers"]),
        }
        for field in ("serial_ms", "pool_ms", "per_task_ms", "speedup"):
            values = group[field]
            record[f"{field}_median"] = float(values.median())
            record[f"{field}_q1"] = float(values.quantile(0.25))
            record[f"{field}_q3"] = float(values.quantile(0.75))
            record[f"{field}_min"] = float(values.min())
            record[f"{field}_max"] = float(values.max())
        summaries.append(record)
    return derived, pd.DataFrame(summaries)


def _inventory_frame(payload: dict[str, Any]) -> pd.DataFrame:
    """Return the exact per-slice changed-path counts from a validated inventory."""

    rows: list[dict[str, Any]] = []
    for item in payload["slices"]:
        paths = item["paths"]
        if item["changed_path_count"] != len(paths):
            raise Sprint5EvidenceError("delivery inventory path count does not reconcile")
        rows.append(
            {
                "slice": item["mr_group"],
                "changed_paths": len(paths),
                "commit": item["commit"],
            }
        )
    frame = pd.DataFrame(rows)
    if frame.empty or frame["slice"].duplicated().any():
        raise Sprint5EvidenceError("delivery inventory slices must be non-empty and unique")
    if int(frame["changed_paths"].sum()) != payload["totals"]["changed_path_count"]:
        raise Sprint5EvidenceError("delivery inventory total does not reconcile")
    return frame


def _observed_break_even(summary: pd.DataFrame) -> str:
    """Describe only the observed median bracket; never extrapolate a threshold."""

    ordered = summary.sort_values("per_task_ms_median")
    below = ordered.loc[ordered["speedup_median"] < 1.0]
    above = ordered.loc[ordered["speedup_median"] > 1.0]
    if below.empty or above.empty:
        return "No observed median break-even bracket"
    lower = below.loc[below["per_task_ms_median"].idxmax()]
    candidates = above.loc[above["per_task_ms_median"] > lower["per_task_ms_median"]]
    if candidates.empty:
        return "No ordered median break-even bracket"
    upper = candidates.loc[candidates["per_task_ms_median"].idxmin()]
    return (
        "Observed median break-even bracket: "
        f"{lower['per_task_ms_median']:.3g}–{upper['per_task_ms_median']:.3g} ms/task"
    )


def _draw_figure(
    decision: ReadinessDecision,
    benchmark: dict[str, Any],
    inventory: dict[str, Any],
    *,
    figure: Any,
    axes: Any,
) -> bytes:
    """Draw all four evidence panels on one caller-owned Matplotlib figure."""

    raw, summary = _benchmark_frames(benchmark)
    delivery = _inventory_frame(inventory)
    environment = benchmark["environment"]

    grouped = decision.unmet_by_category()
    gate_frame = pd.DataFrame(
        [{"category": key, "unmet": len(value)} for key, value in sorted(grouped.items())]
    )
    axis = axes[0][0]
    sns.barplot(data=gate_frame, y="category", x="unmet", ax=axis, color="#d95f02")
    axis.set_title(
        f"Live-readiness gate: {len(decision.unmet)}/{len(decision.results)} items UNMET",
        fontsize=15,
    )
    axis.set_xlabel("Unmet checklist items (count)")
    axis.set_ylabel("Category")
    for index, row in gate_frame.iterrows():
        axis.text(row["unmet"] + 0.05, index, str(row["unmet"]), va="center", fontsize=11)
    axis.set_xlim(0, max(gate_frame["unmet"]) + 1)

    inert = LiveCapitalConfig.inert()
    capital = pd.DataFrame(
        [
            {"state": "Permitted today\n(inert)", "usd": float(inert.capital_cap)},
            {"state": "Absolute code ceiling", "usd": float(ABSOLUTE_MAX_CAPITAL)},
        ]
    )
    axis = axes[0][1]
    sns.barplot(data=capital, x="state", y="usd", hue="state", legend=False, ax=axis)
    axis.set_title("Capital at risk: none authorized", fontsize=15)
    axis.set_ylabel("Configured capital cap (USD)")
    axis.set_xlabel("")
    axis.text(0, float(ABSOLUTE_MAX_CAPITAL) * 0.05, "$0", ha="center", fontweight="bold")

    axis = axes[1][0]
    sns.scatterplot(
        data=raw,
        x="per_task_ms",
        y="speedup",
        hue="iterations",
        palette="colorblind",
        alpha=0.65,
        s=70,
        ax=axis,
    )
    sns.lineplot(
        data=summary,
        x="per_task_ms_median",
        y="speedup_median",
        marker="o",
        color="#1b9e77",
        linewidth=2.5,
        label="median by work size",
        ax=axis,
    )
    axis.fill_between(
        summary["per_task_ms_median"].to_numpy(),
        summary["speedup_q1"].to_numpy(),
        summary["speedup_q3"].to_numpy(),
        color="#1b9e77",
        alpha=0.18,
        label="speedup IQR",
    )
    axis.axhline(1.0, linestyle="--", color="#d95f02", linewidth=2)
    axis.set_xscale("log")
    axis.set_title(
        f"Process-pool crossover — {_observed_break_even(summary)}\n"
        f"n={benchmark['config']['repetitions']} per work size; "
        f"{benchmark['config']['task_count']} tasks, {benchmark['config']['workers']} workers",
        fontsize=13,
    )
    axis.set_xlabel("Measured serial cost per task (ms, log scale)")
    axis.set_ylabel("Serial / process-pool wall time (×)")
    axis.legend(fontsize=8, title="iterations / statistic", loc="best")

    axis = axes[1][1]
    sns.barplot(data=delivery, y="slice", x="changed_paths", color="#7570b3", ax=axis)
    axis.set_title(
        f"Frozen Sprint 5 delivery scope: {inventory['totals']['changed_path_count']} "
        "per-slice path-change records\n(paths may repeat across slices)",
        fontsize=14,
    )
    axis.set_xlabel("Per-slice changed-path records in squash commit (count)")
    axis.set_ylabel("Merge-request slice")
    for index, row in delivery.iterrows():
        axis.text(row["changed_paths"] + 0.3, index, str(row["changed_paths"]), va="center")

    buffer = io.BytesIO()
    figure.suptitle(
        "AlphaForge Signal Foundry Sprint 5 track — broker boundary and readiness controls\n"
        "Synthetic workload and repository evidence only. No strategy qualified; "
        "no paper/live trading or capital authorized.\n"
        f"Benchmark: {environment['python_implementation']} {environment['python_version']}, "
        f"{environment['platform']} {environment['machine']}; raw repetitions shown.",
        fontsize=16,
    )
    figure.tight_layout(rect=(0, 0, 1, 0.92))
    figure.savefig(
        buffer,
        format="png",
        dpi=150,
        bbox_inches="tight",
        metadata={"Software": "AlphaForge Sprint 5 evidence publisher"},
    )
    payload = buffer.getvalue()
    if not payload.startswith(b"\x89PNG\r\n\x1a\n"):
        raise Sprint5EvidenceError("renderer did not produce a PNG artifact")
    if not 0 < len(payload) <= MAX_ARTIFACT_BYTES:
        raise Sprint5EvidenceError(f"rendered figure bytes must be in [1, {MAX_ARTIFACT_BYTES}]")
    return payload


def _figure(
    decision: ReadinessDecision,
    benchmark: dict[str, Any],
    inventory: dict[str, Any],
) -> bytes:
    """Render through Seaborn, closing Matplotlib resources on every failure path."""

    sns.set_theme(style="whitegrid", context="talk", palette="colorblind")
    figure, axes = plt.subplots(2, 2, figsize=(20, 15))
    try:
        return _draw_figure(
            decision,
            benchmark,
            inventory,
            figure=figure,
            axes=axes,
        )
    finally:
        plt.close(figure)


def _payload_snapshots_at(
    directory_descriptor: int,
    *,
    display_root: Path,
) -> tuple[dict[str, RegularFileSnapshot], int]:
    """Read the exact flat payload allowlist through one directory descriptor."""

    snapshots: dict[str, RegularFileSnapshot] = {}
    total = 0
    for name in sorted(_PAYLOAD_ARTIFACTS):
        artifact = _read_regular_file_snapshot_at(
            directory_descriptor,
            name,
            max_bytes=MAX_ARTIFACT_BYTES,
            display_root=display_root,
        )
        total += artifact.size
        if total > MAX_BUNDLE_BYTES:
            raise Sprint5EvidenceError(f"evidence payload exceeds {MAX_BUNDLE_BYTES} bytes")
        snapshots[name] = artifact
    return snapshots, total


def _artifact_records_at(
    staging: _StagingDirectory,
    *,
    display_root: Path,
) -> tuple[dict[str, dict[str, Any]], int]:
    _assert_staging_artifacts_owned(staging, expected=_PAYLOAD_ARTIFACTS)
    observed = set(os.listdir(staging.descriptor))
    if observed != _PAYLOAD_ARTIFACTS:
        raise Sprint5EvidenceError(
            "staged payload does not match the bounded artifact allowlist: "
            f"missing={sorted(_PAYLOAD_ARTIFACTS - observed)}, "
            f"extra={sorted(observed - _PAYLOAD_ARTIFACTS)}"
        )
    snapshots, total = _payload_snapshots_at(
        staging.descriptor,
        display_root=display_root,
    )
    records = {
        name: {"bytes": snapshot.size, "sha256": snapshot.sha256}
        for name, snapshot in snapshots.items()
    }
    return records, total


def _assert_staging_artifacts_owned(
    staging: _StagingDirectory,
    *,
    expected: set[str] | frozenset[str],
) -> None:
    """Require every staged pathname to retain its invocation-created inode."""

    expected_names = set(expected)
    if set(staging.artifacts) != expected_names:
        raise Sprint5EvidenceError("staging ownership records differ from expected artifacts")
    if set(os.listdir(staging.descriptor)) != expected_names:
        raise Sprint5EvidenceError("staging directory entries differ from expected artifacts")
    for name in sorted(expected_names):
        _assert_identity_at(
            staging.descriptor,
            name,
            staging.artifacts[name].identity,
            field=f"staged artifact ownership {name}",
        )


def _assert_bundle_bytes_at(
    directory_descriptor: int,
    *,
    display_root: Path,
    manifest: Mapping[str, Any],
) -> None:
    """Re-read the complete bundle and bind every final byte to its manifest."""

    expected = _PAYLOAD_ARTIFACTS | {"manifest.json"}
    if set(os.listdir(directory_descriptor)) != expected:
        raise Sprint5EvidenceError("evidence bundle changed after semantic verification")
    manifest_snapshot = _read_regular_file_snapshot_at(
        directory_descriptor,
        "manifest.json",
        max_bytes=MAX_ARTIFACT_BYTES,
        display_root=display_root,
    )
    if manifest_snapshot.data != _json_bytes(dict(manifest)):
        raise Sprint5EvidenceError("evidence manifest changed after semantic verification")
    snapshots, payload_bytes = _payload_snapshots_at(
        directory_descriptor,
        display_root=display_root,
    )
    if payload_bytes + manifest_snapshot.size > MAX_BUNDLE_BYTES:
        raise Sprint5EvidenceError(f"evidence bundle exceeds {MAX_BUNDLE_BYTES} bytes")
    for name, snapshot in snapshots.items():
        if manifest["artifacts"][name] != {
            "bytes": snapshot.size,
            "sha256": snapshot.sha256,
        }:
            raise Sprint5EvidenceError(f"evidence artifact changed after verification: {name}")


def _manifest_identity(payload: dict[str, Any]) -> str:
    unsigned = {key: value for key, value in payload.items() if key != "bundle_id"}
    return _sha256(_json_bytes(unsigned))


def _render_environment() -> dict[str, str]:
    return {name: importlib.metadata.version(name) for name in _RENDER_DEPENDENCIES}


def _render_runtime() -> dict[str, str]:
    """Return renderer details that can materially affect deterministic PNG bytes."""

    libc_name, libc_version = platform_module.libc_ver()
    return {
        "freetype_version": ft2font.__freetype_version__,
        "libc_name": libc_name or "unknown",
        "libc_version": libc_version or "unknown",
        "machine": platform_module.machine() or "unknown",
        "matplotlib_backend": str(matplotlib.get_backend()),
        "platform_release": platform_module.release() or "unknown",
        "platform_system": platform_module.system() or "unknown",
        "python_implementation": platform_module.python_implementation(),
        "python_version": platform_module.python_version(),
    }


def _source_snapshots(repository: Path) -> dict[str, _GeneratorSourceSnapshot]:
    """Snapshot every generator source once, including bytes and inode identity."""

    snapshots: dict[str, _GeneratorSourceSnapshot] = {}
    for relative in _GENERATOR_SOURCE_PATHS:
        path = _inside_repository(repository, relative, role="generator source")
        flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
        descriptor = -1
        try:
            descriptor = os.open(path, flags)
            initial = os.fstat(descriptor)
            identity = _identity(
                initial,
                directory=False,
                field=f"generator source {relative}",
            )
            if not 0 < initial.st_size <= MAX_INPUT_BYTES:
                raise Sprint5EvidenceError(
                    f"generator source bytes must be in [1, {MAX_INPUT_BYTES}]: {relative}"
                )
            chunks: list[bytes] = []
            total = 0
            while True:
                chunk = os.read(descriptor, min(1024 * 1024, MAX_INPUT_BYTES - total + 1))
                if not chunk:
                    break
                total += len(chunk)
                if total > MAX_INPUT_BYTES:
                    raise Sprint5EvidenceError(
                        f"generator source exceeds {MAX_INPUT_BYTES} bytes: {relative}"
                    )
                chunks.append(chunk)
            final = os.fstat(descriptor)
            stable_metadata = (
                final.st_size == initial.st_size
                and final.st_mtime_ns == initial.st_mtime_ns
                and final.st_ctime_ns == initial.st_ctime_ns
            )
            if not _same_identity(final, identity) or not stable_metadata:
                raise Sprint5EvidenceError(f"generator source changed during read: {relative}")
            pathname = os.stat(path, follow_symlinks=False)
            if not _same_identity(pathname, identity):
                raise Sprint5EvidenceError(
                    f"generator source pathname changed during read: {relative}"
                )
            _inside_repository(repository, path, role="generator source")
        except Sprint5EvidenceError:
            raise
        except OSError as exc:
            raise Sprint5EvidenceError(
                f"generator source cannot be content-identified: {relative}"
            ) from exc
        finally:
            if descriptor >= 0:
                os.close(descriptor)
        data = b"".join(chunks)
        if len(data) != initial.st_size:
            raise Sprint5EvidenceError(f"generator source length changed: {relative}")
        snapshots[relative] = _GeneratorSourceSnapshot(
            path=path,
            data=data,
            sha256=_sha256(data),
            device=identity.device,
            inode=identity.inode,
            size=len(data),
            modified_ns=initial.st_mtime_ns,
            changed_ns=initial.st_ctime_ns,
        )
    return snapshots


def _source_records(
    snapshots: Mapping[str, _GeneratorSourceSnapshot],
) -> dict[str, dict[str, Any]]:
    """Return manifest records from one already-captured generator snapshot set."""

    if set(snapshots) != set(_GENERATOR_SOURCE_PATHS):
        raise Sprint5EvidenceError("generator source snapshot set is incomplete")
    records: dict[str, dict[str, Any]] = {}
    for relative in _GENERATOR_SOURCE_PATHS:
        snapshot = snapshots[relative]
        records[relative] = {"bytes": snapshot.size, "sha256": snapshot.sha256}
    return records


def _assert_source_snapshots_unchanged(
    repository: Path,
    expected: Mapping[str, _GeneratorSourceSnapshot],
) -> None:
    """Re-read sources immediately before commit and require exact identity/bytes."""

    observed = _source_snapshots(repository)
    if set(observed) != set(expected):
        raise Sprint5EvidenceError("generator source snapshot set changed before commit")
    for relative in _GENERATOR_SOURCE_PATHS:
        if observed[relative] != expected[relative]:
            raise Sprint5EvidenceError(f"generator source changed before commit: {relative}")


def _raise_rename_error(error: int, destination: str) -> None:
    if error in {errno.EEXIST, errno.ENOTEMPTY}:
        raise FileExistsError(
            error,
            f"evidence destination already exists: {destination}",
            destination,
        )
    unsupported = {errno.EINVAL, errno.ENOSYS}
    if hasattr(errno, "ENOTSUP"):
        unsupported.add(errno.ENOTSUP)
    if hasattr(errno, "EOPNOTSUPP"):
        unsupported.add(errno.EOPNOTSUPP)
    if error in unsupported:
        raise Sprint5EvidenceError(
            "filesystem does not support an atomic no-replace directory rename"
        )
    raise OSError(error, os.strerror(error), destination)


def _sync_parent(anchor: _ParentAnchor) -> None:
    """Make anchored parent mutations durable without reopening its pathname."""

    os.fsync(anchor.descriptor)


def _rename_staging(
    anchor: _ParentAnchor,
    staging: _StagingDirectory,
    destination_name: str,
) -> None:
    """Atomically commit a directory without replacing a raced destination.

    macOS provides ``renamex_np(RENAME_EXCL)`` and Linux provides
    ``renameat2(RENAME_NOREPLACE)``.  There is deliberately no check-then-rename
    fallback: POSIX ``rename`` may replace an empty directory after the check.
    Unsupported kernels therefore fail closed.
    """

    if os.name != "posix":
        raise Sprint5EvidenceError("atomic no-replace publication requires a POSIX platform")
    destination_name = _leaf_name(destination_name, field="evidence destination name")
    _assert_parent_anchor_path(anchor)
    _assert_identity_at(
        anchor.descriptor,
        staging.name,
        staging.identity,
        field="staging directory rename source",
    )
    library = ctypes.CDLL(None, use_errno=True)
    source_bytes = os.fsencode(staging.name)
    destination_bytes = os.fsencode(destination_name)
    ctypes.set_errno(0)
    if sys.platform == "darwin":
        operation = getattr(library, "renameatx_np", None)
        if operation is None:
            raise Sprint5EvidenceError("macOS renameatx_np is unavailable")
        operation.argtypes = [
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_uint,
        ]
        operation.restype = ctypes.c_int
        result = operation(
            anchor.descriptor,
            source_bytes,
            anchor.descriptor,
            destination_bytes,
            0x00000004,
        )
    elif sys.platform.startswith("linux"):
        operation = getattr(library, "renameat2", None)
        if operation is None:
            raise Sprint5EvidenceError("Linux renameat2 is unavailable")
        operation.argtypes = [
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_uint,
        ]
        operation.restype = ctypes.c_int
        result = operation(
            anchor.descriptor,
            source_bytes,
            anchor.descriptor,
            destination_bytes,
            0x00000001,
        )
    else:
        raise Sprint5EvidenceError(
            f"atomic no-replace publication is unsupported on {sys.platform!r}"
        )
    if result != 0:
        _raise_rename_error(ctypes.get_errno(), destination_name)


def _csv_bytes(benchmark: Mapping[str, Any], benchmark_sha256: str) -> bytes:
    """Derive the complete summary CSV from validated raw benchmark samples."""

    _, summary = _benchmark_frames(dict(benchmark))
    summary_output = summary.copy()
    summary_output.insert(1, "benchmark_input_sha256", benchmark_sha256)
    return summary_output.to_csv(index=False, lineterminator="\n").encode("utf-8")


def _parse_json_snapshot(snapshot: RegularFileSnapshot, *, field: str) -> Any:
    try:
        return parse_strict_json(
            snapshot.data,
            maximum_depth=MAX_JSON_DEPTH,
            maximum_nodes=MAX_JSON_NODES,
        )
    except BoundedIOError as exc:
        raise Sprint5EvidenceError(f"{field} is not strict bounded JSON") from exc


def _validate_content_record(
    value: object,
    *,
    field: str,
    maximum_bytes: int,
) -> Mapping[str, Any]:
    record = _mapping(value, {"bytes", "sha256"}, field=field)
    _integer(record["bytes"], field=f"{field}.bytes", minimum=1, maximum=maximum_bytes)
    _digest(record["sha256"], field=f"{field}.sha256")
    return record


def _validate_manifest(snapshot: RegularFileSnapshot) -> dict[str, Any]:
    """Validate the exact schema and primitive bounds before trusting bindings."""

    value = _parse_json_snapshot(snapshot, field="evidence manifest")
    manifest = _mapping(
        value,
        {
            "absolute_code_ceiling_usd",
            "artifacts",
            "bundle_id",
            "capital_at_risk_usd",
            "checklist_identity",
            "delivery_source_head",
            "generator",
            "inputs",
            "limitations",
            "manifest_integrity",
            "measurement_environment",
            "schema_version",
            "sprint",
            "total_items",
            "track",
            "unmet_count",
            "verdict",
        },
        field="evidence manifest",
    )
    if manifest["schema_version"] != EVIDENCE_SCHEMA_VERSION:
        raise Sprint5EvidenceError("evidence manifest schema is unsupported")
    if manifest["track"] != "AlphaForge":
        raise Sprint5EvidenceError("evidence manifest track is unsupported")
    if _integer(manifest["sprint"], field="manifest.sprint", minimum=1, maximum=10_000) != 5:
        raise Sprint5EvidenceError("evidence manifest sprint is not Sprint 5")
    _digest(manifest["bundle_id"], field="manifest.bundle_id")
    _digest(manifest["checklist_identity"], field="manifest.checklist_identity")
    _digest(manifest["delivery_source_head"], field="manifest.delivery_source_head", sha1=True)
    _text(manifest["verdict"], field="manifest.verdict", maximum=64)
    _integer(manifest["unmet_count"], field="manifest.unmet_count", minimum=0, maximum=128)
    _integer(manifest["total_items"], field="manifest.total_items", minimum=1, maximum=128)
    _text(manifest["capital_at_risk_usd"], field="manifest.capital_at_risk_usd", maximum=64)
    _text(
        manifest["absolute_code_ceiling_usd"],
        field="manifest.absolute_code_ceiling_usd",
        maximum=64,
    )
    if manifest["manifest_integrity"] != _MANIFEST_INTEGRITY:
        raise Sprint5EvidenceError("evidence manifest integrity statement is unsupported")
    if manifest["limitations"] != list(_LIMITATIONS):
        raise Sprint5EvidenceError("evidence manifest limitations differ from the schema")

    inputs = _mapping(
        manifest["inputs"],
        {
            "benchmark_id",
            "benchmark_sha256",
            "delivery_inventory_id",
            "delivery_inventory_sha256",
        },
        field="manifest.inputs",
    )
    for field in inputs:
        _digest(inputs[field], field=f"manifest.inputs.{field}")

    artifact_records = _mapping(
        manifest["artifacts"], set(_PAYLOAD_ARTIFACTS), field="manifest.artifacts"
    )
    for name in _PAYLOAD_ARTIFACTS:
        _validate_content_record(
            artifact_records[name],
            field=f"manifest.artifacts.{name}",
            maximum_bytes=MAX_ARTIFACT_BYTES,
        )

    generator = _mapping(
        manifest["generator"],
        {"callable", "render_dependencies", "render_runtime", "source_files"},
        field="manifest.generator",
    )
    if generator["callable"] != (
        "alphaforge.readiness.sprint_5_evidence.publish_sprint_5_evidence"
    ):
        raise Sprint5EvidenceError("evidence generator callable is unsupported")
    source_files = _mapping(
        generator["source_files"],
        set(_GENERATOR_SOURCE_PATHS),
        field="manifest.generator.source_files",
    )
    for name in _GENERATOR_SOURCE_PATHS:
        _validate_content_record(
            source_files[name],
            field=f"manifest.generator.source_files.{name}",
            maximum_bytes=MAX_INPUT_BYTES,
        )
    dependencies = _mapping(
        generator["render_dependencies"],
        set(_RENDER_DEPENDENCIES),
        field="manifest.generator.render_dependencies",
    )
    for name, version in dependencies.items():
        _text(version, field=f"manifest.generator.render_dependencies.{name}", maximum=128)
    runtime = _mapping(
        generator["render_runtime"],
        {
            "freetype_version",
            "libc_name",
            "libc_version",
            "machine",
            "matplotlib_backend",
            "platform_release",
            "platform_system",
            "python_implementation",
            "python_version",
        },
        field="manifest.generator.render_runtime",
    )
    for name, version in runtime.items():
        _text(version, field=f"manifest.generator.render_runtime.{name}", maximum=128)
    if not isinstance(manifest["measurement_environment"], dict):
        raise Sprint5EvidenceError("manifest.measurement_environment must be an object")

    document = dict(manifest)
    if snapshot.data != _json_bytes(document):
        raise Sprint5EvidenceError("evidence manifest is not canonical JSON")
    if document["bundle_id"] != _manifest_identity(document):
        raise Sprint5EvidenceError("evidence manifest bundle identity does not verify")
    return document


def _verify_artifact_semantics(
    manifest: Mapping[str, Any],
    snapshots: Mapping[str, RegularFileSnapshot],
) -> tuple[BenchmarkEvidence, DeliveryInventory]:
    """Reconcile every machine-readable claim to its copied source artifacts."""

    try:
        benchmark = parse_benchmark_evidence_bytes(
            snapshots["distribution_crossover_raw.json"].data
        )
    except BenchmarkEvidenceError as exc:
        raise Sprint5EvidenceError("copied benchmark evidence is invalid") from exc
    try:
        require_production_implementation(benchmark.config.implementation)
    except BenchmarkEvidenceError as exc:
        raise Sprint5EvidenceError(
            "copied benchmark does not use the production execution contract"
        ) from exc
    benchmark_payload = benchmark.to_dict()
    inventory_value = _parse_json_snapshot(
        snapshots["delivery_inventory.json"], field="delivery inventory"
    )
    try:
        inventory = DeliveryInventory.from_dict(inventory_value)
    except Sprint5InventoryError as exc:
        raise Sprint5EvidenceError("copied delivery inventory is invalid") from exc
    if snapshots["delivery_inventory.json"].data != _json_bytes(inventory.to_dict()):
        raise Sprint5EvidenceError("delivery inventory is not canonical JSON")

    checklist = minimal_capital_checklist()
    decision = sprint_5_readiness()
    expected_artifacts = {
        "checklist.json": _json_bytes(checklist.to_dict()),
        "distribution_crossover.csv": _csv_bytes(
            benchmark_payload,
            snapshots["distribution_crossover_raw.json"].sha256,
        ),
        "readiness_decision.json": _json_bytes(decision.to_dict()),
        "readiness_report.md": (render_readiness_report(decision).rstrip() + "\n").encode("utf-8"),
    }
    for name, expected in expected_artifacts.items():
        if snapshots[name].data != expected:
            raise Sprint5EvidenceError(f"evidence artifact does not reconcile: {name}")
    expected_figure = _figure(decision, benchmark_payload, inventory.to_dict())
    if snapshots["sprint_5_closeout.png"].data != expected_figure:
        raise Sprint5EvidenceError("close-out figure does not byte-match a fresh render")

    inputs = manifest["inputs"]
    if inputs["benchmark_id"] != benchmark.benchmark_id:
        raise Sprint5EvidenceError("benchmark identity does not reconcile")
    if inputs["benchmark_sha256"] != snapshots["distribution_crossover_raw.json"].sha256:
        raise Sprint5EvidenceError("benchmark digest is not bound to its copied bytes")
    if inputs["delivery_inventory_id"] != inventory.inventory_id:
        raise Sprint5EvidenceError("delivery inventory identity does not reconcile")
    if inputs["delivery_inventory_sha256"] != snapshots["delivery_inventory.json"].sha256:
        raise Sprint5EvidenceError("delivery inventory digest is not bound to its copied bytes")
    if manifest["delivery_source_head"] != inventory.source_head:
        raise Sprint5EvidenceError("delivery source head does not reconcile")
    if _json_bytes(manifest["measurement_environment"]) != _json_bytes(
        benchmark_payload["environment"]
    ):
        raise Sprint5EvidenceError("measurement environment does not reconcile")
    if manifest["checklist_identity"] != checklist.identity:
        raise Sprint5EvidenceError("checklist identity does not reconcile")
    if manifest["verdict"] != decision.verdict.value:
        raise Sprint5EvidenceError("readiness verdict does not reconcile")
    if manifest["unmet_count"] != len(decision.unmet):
        raise Sprint5EvidenceError("readiness unmet count does not reconcile")
    if manifest["total_items"] != len(decision.results):
        raise Sprint5EvidenceError("readiness total item count does not reconcile")
    if manifest["capital_at_risk_usd"] != str(LiveCapitalConfig.inert().capital_cap):
        raise Sprint5EvidenceError("inert capital amount does not reconcile")
    if manifest["absolute_code_ceiling_usd"] != str(ABSOLUTE_MAX_CAPITAL):
        raise Sprint5EvidenceError("absolute capital ceiling does not reconcile")
    return benchmark, inventory


def _verify_repository_provenance(
    manifest: Mapping[str, Any],
    benchmark: BenchmarkEvidence,
    inventory: DeliveryInventory,
    repository: Path,
) -> None:
    """Rebuild every external Git/source claim from one exact repository root."""

    rebuilt = build_sprint_5_inventory(repository)
    if rebuilt.to_dict() != inventory.to_dict():
        raise Sprint5EvidenceError("delivery inventory differs from rebuilt frozen Git objects")
    current_sources = _source_records(_source_snapshots(repository))
    if manifest["generator"]["source_files"] != current_sources:
        raise Sprint5EvidenceError("generator source provenance differs from repository bytes")
    try:
        verify_production_implementation_sources(
            benchmark.config.implementation,
            current_sources,
        )
    except BenchmarkEvidenceError as exc:
        raise Sprint5EvidenceError(
            "benchmark implementation does not reconcile to repository sources"
        ) from exc


def _verify_bundle_at(
    directory_descriptor: int,
    *,
    display_root: Path,
    repository: Path | None,
) -> dict[str, Any]:
    """Verify a flat bundle through one already-open directory descriptor."""

    expected = _PAYLOAD_ARTIFACTS | {"manifest.json"}
    observed = set(os.listdir(directory_descriptor))
    if observed != expected or len(observed) > MAX_ARTIFACTS:
        raise Sprint5EvidenceError("evidence bundle contains missing, extra, or nested artifacts")
    manifest_snapshot = _read_regular_file_snapshot_at(
        directory_descriptor,
        "manifest.json",
        max_bytes=MAX_ARTIFACT_BYTES,
        display_root=display_root,
    )
    manifest = _validate_manifest(manifest_snapshot)
    snapshots, payload_bytes = _payload_snapshots_at(
        directory_descriptor,
        display_root=display_root,
    )
    if payload_bytes + manifest_snapshot.size > MAX_BUNDLE_BYTES:
        raise Sprint5EvidenceError(f"evidence bundle exceeds {MAX_BUNDLE_BYTES} bytes")
    records = manifest["artifacts"]
    for name, artifact in snapshots.items():
        if records[name] != {"bytes": artifact.size, "sha256": artifact.sha256}:
            raise Sprint5EvidenceError(f"evidence artifact integrity failure: {name}")
    generator = manifest["generator"]
    if generator["render_dependencies"] != _render_environment():
        raise Sprint5EvidenceError("renderer dependency versions differ from the manifest")
    if generator["render_runtime"] != _render_runtime():
        raise Sprint5EvidenceError("renderer runtime differs from the manifest")
    benchmark, inventory = _verify_artifact_semantics(manifest, snapshots)
    if repository is not None:
        _verify_repository_provenance(manifest, benchmark, inventory, repository)
    else:
        warnings.warn(
            "standalone Sprint 5 verification checked bundle bytes, schemas, cross-artifact "
            "semantics, and a byte-identical PNG re-render, but it did not rebuild Git inventory "
            "or compare generator sources; pass repository_root for full external provenance",
            Sprint5LimitedVerificationWarning,
            stacklevel=3,
        )
    _assert_bundle_bytes_at(
        directory_descriptor,
        display_root=display_root,
        manifest=manifest,
    )
    return manifest


def _open_bundle_directory(root: Path) -> tuple[int, _Identity]:
    """Open one real bundle directory and bind its pathname to that descriptor."""

    try:
        if root.resolve(strict=True) != root or root.is_symlink():
            raise Sprint5EvidenceError(
                "evidence bundle must be a real path without symlink traversal"
            )
    except (FileNotFoundError, RuntimeError) as exc:
        raise Sprint5EvidenceError("evidence bundle must be a real, non-symlink directory") from exc
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_DIRECTORY", 0)
    flags |= getattr(os, "O_NOFOLLOW", 0)
    descriptor = -1
    try:
        descriptor = os.open(root, flags)
        identity = _identity(os.fstat(descriptor), directory=True, field="evidence bundle")
        if not _same_identity(os.stat(root, follow_symlinks=False), identity):
            raise Sprint5EvidenceError("evidence bundle changed during acquisition")
        return descriptor, identity
    except Sprint5EvidenceError:
        if descriptor >= 0:
            os.close(descriptor)
        raise
    except OSError as exc:
        if descriptor >= 0:
            os.close(descriptor)
        raise Sprint5EvidenceError("unable to open evidence bundle directory") from exc


def verify_sprint_5_bundle(
    bundle: str | Path,
    *,
    repository_root: str | Path | None = None,
) -> dict[str, Any]:
    """Verify a Sprint 5 bundle, with optional external repository provenance.

    Byte integrity, strict schemas, cross-artifact semantics, renderer context,
    and an exact PNG re-render are always checked.  Passing ``repository_root``
    additionally rebuilds the frozen Git inventory and compares every generator
    source record.  Without it, verification emits
    :class:`Sprint5LimitedVerificationWarning` because those external claims
    cannot be established from a self-contained bundle alone.
    """

    root = Path(os.path.abspath(bundle))
    repository = _safe_repository(repository_root) if repository_root is not None else None
    descriptor, identity = _open_bundle_directory(root)
    try:
        manifest = _verify_bundle_at(
            descriptor,
            display_root=root,
            repository=repository,
        )
        if not _same_identity(os.stat(root, follow_symlinks=False), identity):
            raise Sprint5EvidenceError("evidence bundle changed during verification")
        return manifest
    except Sprint5EvidenceError:
        raise
    except OSError as exc:
        raise Sprint5EvidenceError("unable to verify evidence bundle directory") from exc
    finally:
        os.close(descriptor)


def _post_commit_error(destination: Path, cause: BaseException) -> Sprint5PostCommitError:
    error = Sprint5PostCommitError(
        f"evidence bundle committed at {destination}, but final identity, cleanup, or durability "
        "verification failed; inspect the committed bundle and do not retry blindly"
    )
    error.__cause__ = cause
    return error


def publish_sprint_5_evidence(
    *,
    repository_root: str | Path,
    benchmark_input: str | Path,
    output: str | Path,
) -> dict[str, Any]:
    """Atomically publish deterministic close-out evidence into an absent path.

    Args:
        repository_root: AlphaForge Git worktree containing all inputs and output.
        benchmark_input: Repository-local bounded raw benchmark JSON.
        output: Repository-local final directory, which must not exist.

    Raises:
        FileExistsError: If the output or cooperative-writer lock exists.
        Sprint5EvidenceError: If any input, Git object, artifact, or integrity
            invariant fails.  Before the rename, failure leaves no output.
        Sprint5PostCommitError: If the no-replace rename committed but a final
            identity, lock cleanup, or directory durability check failed.  The
            committed destination is deliberately preserved for inspection.
    """

    repository = _safe_repository(repository_root)
    benchmark_path = _inside_repository(repository, benchmark_input, role="benchmark input")
    destination = _inside_repository(repository, output, role="evidence output")
    if benchmark_path == destination or destination in benchmark_path.parents:
        raise Sprint5EvidenceError("benchmark input and evidence output paths overlap")
    destination_name = _leaf_name(destination.name, field="evidence destination name")
    lock_name = _leaf_name(
        f".{destination_name}.publish.lock",
        field="publication lock name",
    )

    anchor: _ParentAnchor | None = None
    lock: _OwnedFile | None = None
    staging: _StagingDirectory | None = None
    committed = False
    verified: dict[str, Any] | None = None
    primary: BaseException | None = None
    try:
        anchor = _open_parent_anchor(repository, destination.parent)
        if _entry_exists_at(anchor.descriptor, destination_name):
            raise FileExistsError(f"evidence destination already exists: {destination}")
        lock = _acquire_publication_lock(anchor, lock_name)

        try:
            benchmark_snapshot = read_regular_file_snapshot(
                benchmark_path,
                max_bytes=MAX_INPUT_BYTES,
                root=repository,
            )
        except BoundedIOError as exc:
            raise Sprint5EvidenceError("benchmark input is not a bounded immutable file") from exc
        try:
            benchmark = parse_benchmark_evidence_bytes(benchmark_snapshot.data)
        except BenchmarkEvidenceError as exc:
            raise Sprint5EvidenceError("benchmark input is not valid benchmark evidence") from exc
        try:
            require_production_implementation(benchmark.config.implementation)
        except BenchmarkEvidenceError as exc:
            raise Sprint5EvidenceError(
                "benchmark input does not use the production execution contract"
            ) from exc
        benchmark_payload = benchmark.to_dict()
        inventory = build_sprint_5_inventory(repository)
        inventory_payload = inventory.to_dict()
        inventory_bytes = _json_bytes(inventory_payload)
        _benchmark_frames(benchmark_payload)
        _inventory_frame(inventory_payload)
        decision = sprint_5_readiness()

        source_snapshots = _source_snapshots(repository)
        render_dependencies = _render_environment()
        render_runtime = _render_runtime()
        figure_bytes = _figure(decision, benchmark_payload, inventory_payload)

        staging = _create_staging(anchor, destination_name)
        staging_display = destination.parent / staging.name
        _write_bytes_at(staging, "distribution_crossover_raw.json", benchmark_snapshot.data)
        _write_bytes_at(staging, "delivery_inventory.json", inventory_bytes)
        _write_bytes_at(
            staging,
            "checklist.json",
            _json_bytes(minimal_capital_checklist().to_dict()),
        )
        _write_bytes_at(staging, "readiness_decision.json", _json_bytes(decision.to_dict()))
        _write_bytes_at(
            staging,
            "readiness_report.md",
            (render_readiness_report(decision).rstrip() + "\n").encode("utf-8"),
        )
        _write_bytes_at(
            staging,
            "distribution_crossover.csv",
            _csv_bytes(benchmark_payload, benchmark_snapshot.sha256),
        )
        _write_bytes_at(staging, "sprint_5_closeout.png", figure_bytes)

        records, payload_bytes = _artifact_records_at(
            staging,
            display_root=staging_display,
        )
        manifest: dict[str, Any] = {
            "schema_version": EVIDENCE_SCHEMA_VERSION,
            "sprint": 5,
            "track": "AlphaForge",
            "verdict": decision.verdict.value,
            "unmet_count": len(decision.unmet),
            "total_items": len(decision.results),
            "capital_at_risk_usd": str(LiveCapitalConfig.inert().capital_cap),
            "absolute_code_ceiling_usd": str(ABSOLUTE_MAX_CAPITAL),
            "checklist_identity": decision.checklist_identity,
            "delivery_source_head": inventory_payload["source_head"],
            "measurement_environment": benchmark_payload["environment"],
            "generator": {
                "callable": ("alphaforge.readiness.sprint_5_evidence.publish_sprint_5_evidence"),
                "render_dependencies": render_dependencies,
                "render_runtime": render_runtime,
                "source_files": _source_records(source_snapshots),
            },
            "inputs": {
                "benchmark_id": benchmark_payload["benchmark_id"],
                "benchmark_sha256": benchmark_snapshot.sha256,
                "delivery_inventory_id": inventory_payload["inventory_id"],
                "delivery_inventory_sha256": _sha256(inventory_bytes),
            },
            "artifacts": records,
            "manifest_integrity": _MANIFEST_INTEGRITY,
            "limitations": list(_LIMITATIONS),
        }
        manifest["bundle_id"] = _manifest_identity(manifest)
        manifest_bytes = _json_bytes(manifest)
        if payload_bytes + len(manifest_bytes) > MAX_BUNDLE_BYTES:
            raise Sprint5EvidenceError(f"evidence bundle exceeds {MAX_BUNDLE_BYTES} bytes")
        _write_bytes_at(staging, "manifest.json", manifest_bytes)
        os.fsync(staging.descriptor)
        verified = _verify_bundle_at(
            staging.descriptor,
            display_root=staging_display,
            repository=repository,
        )
        if verified != manifest:
            raise Sprint5EvidenceError("independent staged verification changed the manifest")
        _assert_source_snapshots_unchanged(repository, source_snapshots)
        _assert_staging_artifacts_owned(
            staging,
            expected=_PAYLOAD_ARTIFACTS | {"manifest.json"},
        )
        _assert_bundle_bytes_at(
            staging.descriptor,
            display_root=staging_display,
            manifest=manifest,
        )
        _rename_staging(anchor, staging, destination_name)
        committed = True
        _assert_identity_at(
            anchor.descriptor,
            destination_name,
            staging.identity,
            field="committed evidence destination",
        )
        _assert_staging_artifacts_owned(
            staging,
            expected=_PAYLOAD_ARTIFACTS | {"manifest.json"},
        )
        _assert_bundle_bytes_at(
            staging.descriptor,
            display_root=destination,
            manifest=manifest,
        )
        _assert_parent_anchor_path(anchor)
        _sync_parent(anchor)
    except BaseException as exc:
        primary = exc

    cleanup_failures: list[BaseException] = []
    if anchor is not None:
        cleanup_failures.extend(_cleanup_staging(anchor, staging, committed=committed))
        cleanup_failures.extend(_cleanup_lock(anchor, lock))
        try:
            _sync_parent(anchor)
        except BaseException as exc:
            cleanup_failures.append(exc)
        descriptor = anchor.descriptor
        anchor.descriptor = -1
        _close_descriptor(descriptor, cleanup_failures)

    if committed:
        if primary is not None and not isinstance(primary, Sprint5PostCommitError):
            primary = _post_commit_error(destination, primary)
        elif primary is None and cleanup_failures:
            cause = cleanup_failures.pop(0)
            primary = _post_commit_error(destination, cause)
    if primary is not None:
        _raise_with_cleanup(
            "evidence publication and invocation-owned cleanup both failed",
            primary,
            cleanup_failures,
        )
    if cleanup_failures:
        raise BaseExceptionGroup(
            "evidence transaction cleanup failed",
            cleanup_failures,
        )
    if verified is None:
        raise Sprint5EvidenceError("evidence publication produced no verified manifest")
    return verified


__all__ = [
    "EVIDENCE_SCHEMA_VERSION",
    "Sprint5EvidenceError",
    "Sprint5LimitedVerificationWarning",
    "Sprint5PostCommitError",
    "publish_sprint_5_evidence",
    "sprint_5_readiness",
    "verify_sprint_5_bundle",
]
