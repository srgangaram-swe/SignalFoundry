"""Deterministic, Git-derived inventory of the AlphaForge Sprint 5 delivery.

The close-out plot needs a truthful description of what each merged slice
changed.  Counting tests from prose is neither reproducible nor stable, so this
module inspects the six frozen squash commits directly.  It reads only the local
Git object database, disables lazy fetching and replacement objects, and never
consults the worktree, index, current branch, or network.

Security and reproducibility invariants:

* the checkout must be the root of the expected public AlphaForge repository;
* every commit is a full, frozen SHA-1 and the six commits form one linear chain;
* subprocess time, input, output, path count, path length, and blob size are
  bounded;
* changed paths are repository-relative, traversal-free, printable ASCII;
* symlinks, submodules, non-blob objects, rename-status records, and unsupported
  modes fail closed; rename detection is deliberately disabled, so a moved path
  is represented truthfully as one deletion plus one addition; and
* counts and the inventory identity are derived from the serialized path records.

Counts are therefore per-slice path-change records, not unique files or semantic
change counts.  The installed ``git`` executable and the local object database are explicit
trust boundaries.  Network and credential access are not required or enabled.

For an added or modified path, ``mode``, ``blob``, and ``bytes`` describe the
blob committed by that slice.  For a deleted path they describe the blob removed
from its first parent.  The status therefore makes the selected side explicit.
"""

from __future__ import annotations

import hashlib
import json
import os
import signal
import subprocess
import threading
from collections import Counter
from collections.abc import Sequence
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Final

INVENTORY_SCHEMA_VERSION: Final = "1.0.0"
EXPECTED_ORIGIN_URL: Final = "https://github.com/srgangaram-swe/AlphaForge.git"
_EXPECTED_ORIGIN_URLS: Final = frozenset(
    {
        EXPECTED_ORIGIN_URL,
        EXPECTED_ORIGIN_URL.removesuffix(".git"),
    }
)
SOURCE_HEAD: Final = "60f572b933d925dcdbd48c167e810be2910b3d25"

FULL_GIT_SHA_LENGTH: Final = 40
MAX_GIT_ARGUMENTS: Final = 32
MAX_GIT_ARGUMENT_BYTES: Final = 1024
MAX_GIT_STDIN_BYTES: Final = 512 * 1024
MAX_GIT_STDERR_BYTES: Final = 16 * 1024
MAX_GIT_OUTPUT_BYTES: Final = 4 * 1024 * 1024
MAX_CHANGED_PATHS: Final = 4096
MAX_PATH_BYTES: Final = 512
MAX_PATH_COMPONENT_BYTES: Final = 255
MAX_TRACKED_BLOB_BYTES: Final = 512 * 1024 * 1024
GIT_TIMEOUT_SECONDS: Final = 15

_ZERO_OID: Final = "0" * FULL_GIT_SHA_LENGTH
_ABSENT_MODE: Final = "000000"
_REGULAR_MODES: Final = frozenset({"100644", "100755"})
_SUPPORTED_STATUSES: Final = frozenset({"A", "D", "M"})
_CATEGORIES: Final = frozenset(
    {
        "automation",
        "benchmarks",
        "configuration",
        "documentation",
        "repository",
        "source",
        "tests",
        "tooling",
    }
)


class Sprint5InventoryError(ValueError):
    """Raised when frozen delivery provenance cannot be verified exactly."""


def _is_lower_hex(value: object, *, length: int = FULL_GIT_SHA_LENGTH) -> bool:
    return (
        isinstance(value, str)
        and len(value) == length
        and all(character in "0123456789abcdef" for character in value)
    )


def _mapping(value: object, expected: set[str], *, field: str) -> dict[str, Any]:
    """Return one exact string-keyed mapping at the inventory trust boundary."""

    if not isinstance(value, dict) or any(not isinstance(key, str) for key in value):
        raise Sprint5InventoryError(f"{field} must be an object with string keys")
    observed = set(value)
    missing = sorted(expected - observed)
    unknown = sorted(observed - expected)
    if missing or unknown:
        raise Sprint5InventoryError(f"{field} keys differ: missing={missing}, unknown={unknown}")
    return value


def _integer(value: object, *, field: str, minimum: int, maximum: int) -> int:
    """Return one bounded integer while refusing JSON booleans."""

    if isinstance(value, bool) or not isinstance(value, int):
        raise Sprint5InventoryError(f"{field} must be an integer, not a bool")
    if not minimum <= value <= maximum:
        raise Sprint5InventoryError(f"{field} must lie in [{minimum}, {maximum}]")
    return value


def _safe_text(value: object, *, field: str, maximum: int) -> str:
    if (
        not isinstance(value, str)
        or not value
        or value != value.strip()
        or not value.isascii()
        or len(value) > maximum
        or any(ord(character) < 32 or ord(character) == 127 for character in value)
    ):
        raise Sprint5InventoryError(
            f"{field} must be non-empty, trimmed printable ASCII of at most {maximum} characters"
        )
    return value


def _validate_repository_path(value: object) -> str:
    path = _safe_text(value, field="changed path", maximum=MAX_PATH_BYTES)
    if "\\" in path or ":" in path:
        raise Sprint5InventoryError("changed path must use a portable repository-relative form")
    parts = path.split("/")
    parsed = PurePosixPath(path)
    if (
        parsed.is_absolute()
        or parsed.as_posix() != path
        or any(part in {"", ".", ".."} for part in parts)
        or any(len(part.encode("ascii")) > MAX_PATH_COMPONENT_BYTES for part in parts)
        or any(part.endswith((" ", ".")) for part in parts)
    ):
        raise Sprint5InventoryError(
            "changed path must be a canonical traversal-free repository-relative POSIX path"
        )
    return path


def _classify_path(path: str) -> str:
    """Classify one validated path using stable, mutually exclusive rules."""

    if path.startswith("tests/"):
        return "tests"
    if path.startswith("alphaforge/"):
        return "source"
    if path.startswith("benchmarks/"):
        return "benchmarks"
    if path.startswith("scripts/"):
        return "tooling"
    if path.startswith("docs/"):
        return "documentation"
    if path.startswith(".github/"):
        return "automation"
    if path.startswith("configs/"):
        return "configuration"
    return "repository"


@dataclass(frozen=True, slots=True)
class DeliverySliceSpec:
    """Frozen Git and GitHub identity for one merged Sprint 5 slice."""

    mr_group: str
    issue_number: int
    commit: str

    def __post_init__(self) -> None:
        group = _safe_text(self.mr_group, field="mr_group", maximum=32)
        if not group.startswith("SF-S5-MR") or not group.removeprefix("SF-S5-MR").isdigit():
            raise Sprint5InventoryError("mr_group must use the SF-S5-MR<number> form")
        if (
            isinstance(self.issue_number, bool)
            or not isinstance(self.issue_number, int)
            or self.issue_number < 1
        ):
            raise Sprint5InventoryError("issue_number must be a positive integer")
        if not _is_lower_hex(self.commit):
            raise Sprint5InventoryError("commit must be a full lowercase SHA-1")


SPRINT_5_SLICES: Final[tuple[DeliverySliceSpec, ...]] = (
    DeliverySliceSpec(
        mr_group="SF-S5-MR2",
        issue_number=99,
        commit="975e015896d5cdf4f8f9808d44bad13362b83b9a",
    ),
    DeliverySliceSpec(
        mr_group="SF-S5-MR3",
        issue_number=45,
        commit="70fd5d76fedf21a18883a14e406ee8714c6015f7",
    ),
    DeliverySliceSpec(
        mr_group="SF-S5-MR4",
        issue_number=46,
        commit="b10aaf25f3407f6b61b37ac8c136534387a330dc",
    ),
    DeliverySliceSpec(
        mr_group="SF-S5-MR8",
        issue_number=47,
        commit="cb3a60efc47b4c7eba2a86054561126cecdd41fa",
    ),
    DeliverySliceSpec(
        mr_group="SF-S5-MR9",
        issue_number=48,
        commit="cb2310f4e4e8a25db07365d9ef77c45df1168a19",
    ),
    DeliverySliceSpec(
        mr_group="SF-S5-MR10",
        issue_number=49,
        commit=SOURCE_HEAD,
    ),
)


@dataclass(frozen=True, slots=True)
class DeliveryPath:
    """One changed regular blob observed in a frozen commit diff."""

    path: str
    status: str
    mode: str
    blob: str
    bytes: int
    category: str

    def __post_init__(self) -> None:
        validated_path = _validate_repository_path(self.path)
        if self.status not in _SUPPORTED_STATUSES:
            raise Sprint5InventoryError(f"unsupported changed-path status: {self.status!r}")
        if self.mode not in _REGULAR_MODES:
            raise Sprint5InventoryError(f"unsupported regular-blob mode: {self.mode!r}")
        if not _is_lower_hex(self.blob):
            raise Sprint5InventoryError("blob must be a full lowercase Git SHA-1")
        if (
            isinstance(self.bytes, bool)
            or not isinstance(self.bytes, int)
            or not 0 <= self.bytes <= MAX_TRACKED_BLOB_BYTES
        ):
            raise Sprint5InventoryError(f"blob bytes must be in [0, {MAX_TRACKED_BLOB_BYTES}]")
        if self.category not in _CATEGORIES or self.category != _classify_path(validated_path):
            raise Sprint5InventoryError(
                "changed-path category does not match the stable classifier"
            )

    def to_dict(self) -> dict[str, Any]:
        """Return the stable JSON-compatible path record."""

        return {
            "path": self.path,
            "status": self.status,
            "mode": self.mode,
            "blob": self.blob,
            "bytes": self.bytes,
            "category": self.category,
        }

    @classmethod
    def from_dict(cls, value: object) -> DeliveryPath:
        """Parse one exact path record and re-run every path/object invariant."""

        data = _mapping(
            value,
            {"path", "status", "mode", "blob", "bytes", "category"},
            field="delivery path",
        )
        return cls(
            path=data["path"],
            status=_safe_text(data["status"], field="delivery path status", maximum=4),
            mode=_safe_text(data["mode"], field="delivery path mode", maximum=6),
            blob=data["blob"],
            bytes=_integer(
                data["bytes"],
                field="delivery path bytes",
                minimum=0,
                maximum=MAX_TRACKED_BLOB_BYTES,
            ),
            category=_safe_text(data["category"], field="delivery path category", maximum=32),
        )


@dataclass(frozen=True, slots=True)
class DeliverySlice:
    """Verified first-parent delta and object identities for one sprint slice."""

    mr_group: str
    issue_number: int
    commit: str
    parent: str
    tree: str
    paths: tuple[DeliveryPath, ...]

    def __post_init__(self) -> None:
        DeliverySliceSpec(self.mr_group, self.issue_number, self.commit)
        if not _is_lower_hex(self.parent) or not _is_lower_hex(self.tree):
            raise Sprint5InventoryError("parent and tree must be full lowercase Git SHA-1 values")
        if not self.paths:
            raise Sprint5InventoryError("each frozen sprint slice must change at least one path")
        observed_paths = tuple(item.path for item in self.paths)
        if observed_paths != tuple(sorted(observed_paths)):
            raise Sprint5InventoryError("changed paths must be sorted deterministically")
        if len(observed_paths) != len(set(observed_paths)):
            raise Sprint5InventoryError("changed paths must be unique within one slice")
        if len(observed_paths) > MAX_CHANGED_PATHS:
            raise Sprint5InventoryError(
                f"changed path count exceeds the bound of {MAX_CHANGED_PATHS}"
            )

    @property
    def changed_path_count(self) -> int:
        """Return the exact number of serialized path changes."""

        return len(self.paths)

    @property
    def category_counts(self) -> dict[str, int]:
        """Return sorted category counts derived from ``paths``."""

        counts = Counter(item.category for item in self.paths)
        return {category: counts[category] for category in sorted(counts)}

    def to_dict(self) -> dict[str, Any]:
        """Return this slice with summaries reconciled to its path records."""

        return {
            "mr_group": self.mr_group,
            "issue_number": self.issue_number,
            "commit": self.commit,
            "parent": self.parent,
            "tree": self.tree,
            "paths": [item.to_dict() for item in self.paths],
            "category_counts": self.category_counts,
            "changed_path_count": self.changed_path_count,
        }

    @classmethod
    def from_dict(cls, value: object) -> DeliverySlice:
        """Parse and reconcile one exact frozen merge-request slice."""

        data = _mapping(
            value,
            {
                "mr_group",
                "issue_number",
                "commit",
                "parent",
                "tree",
                "paths",
                "category_counts",
                "changed_path_count",
            },
            field="delivery slice",
        )
        raw_paths = data["paths"]
        if not isinstance(raw_paths, list):
            raise Sprint5InventoryError("delivery slice paths must be an array")
        if not 1 <= len(raw_paths) <= MAX_CHANGED_PATHS:
            raise Sprint5InventoryError(
                f"delivery slice paths must contain 1-{MAX_CHANGED_PATHS} records"
            )
        paths = tuple(DeliveryPath.from_dict(path) for path in raw_paths)
        result = cls(
            mr_group=data["mr_group"],
            issue_number=_integer(
                data["issue_number"],
                field="delivery slice issue_number",
                minimum=1,
                maximum=2**31 - 1,
            ),
            commit=data["commit"],
            parent=data["parent"],
            tree=data["tree"],
            paths=paths,
        )
        changed_path_count = _integer(
            data["changed_path_count"],
            field="delivery slice changed_path_count",
            minimum=1,
            maximum=MAX_CHANGED_PATHS,
        )
        raw_categories = _mapping(
            data["category_counts"],
            set(result.category_counts),
            field="delivery slice category_counts",
        )
        category_counts = {
            category: _integer(
                count,
                field=f"delivery slice category_counts.{category}",
                minimum=1,
                maximum=MAX_CHANGED_PATHS,
            )
            for category, count in raw_categories.items()
        }
        if changed_path_count != result.changed_path_count:
            raise Sprint5InventoryError("delivery slice changed_path_count does not reconcile")
        if category_counts != result.category_counts:
            raise Sprint5InventoryError("delivery slice category_counts do not reconcile")
        return result


@dataclass(frozen=True, slots=True)
class DeliveryInventory:
    """Exact six planned capability slices used by Sprint 5 close-out evidence.

    Evidence-correction pull requests are intentionally outside this inventory:
    they add no trading capability and a publisher cannot bind its own future
    squash commit without circular provenance.
    """

    source_head: str
    slices: tuple[DeliverySlice, ...]

    def __post_init__(self) -> None:
        if self.source_head != SOURCE_HEAD:
            raise Sprint5InventoryError("source_head must equal the frozen Sprint 5 MR10 commit")
        expected = tuple(
            (item.mr_group, item.issue_number, item.commit) for item in SPRINT_5_SLICES
        )
        observed = tuple((item.mr_group, item.issue_number, item.commit) for item in self.slices)
        if observed != expected:
            raise Sprint5InventoryError(
                "delivery slices must match the six frozen Sprint 5 capability slices exactly"
            )
        for previous, current in zip(self.slices, self.slices[1:], strict=False):
            if current.parent != previous.commit:
                raise Sprint5InventoryError("frozen Sprint 5 commits do not form a linear chain")

    @property
    def changed_path_count(self) -> int:
        """Return the exact number of per-slice path changes."""

        return sum(item.changed_path_count for item in self.slices)

    @property
    def category_counts(self) -> dict[str, int]:
        """Return sorted category counts derived from every serialized path."""

        counts = Counter(path.category for item in self.slices for path in item.paths)
        return {category: counts[category] for category in sorted(counts)}

    def _identity_payload(self) -> dict[str, Any]:
        return {
            "schema_version": INVENTORY_SCHEMA_VERSION,
            "source_head": self.source_head,
            "slices": [item.to_dict() for item in self.slices],
            "totals": {
                "category_counts": self.category_counts,
                "changed_path_count": self.changed_path_count,
            },
        }

    @property
    def inventory_id(self) -> str:
        """Return a SHA-256 identity over the complete reconciled inventory."""

        canonical = json.dumps(
            self._identity_payload(),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
        ).encode("ascii")
        return hashlib.sha256(canonical).hexdigest()

    def to_dict(self) -> dict[str, Any]:
        """Return the required deterministic close-out JSON document."""

        payload = self._identity_payload()
        return {
            "schema_version": payload["schema_version"],
            "inventory_id": self.inventory_id,
            "source_head": payload["source_head"],
            "slices": payload["slices"],
            "totals": payload["totals"],
        }

    def to_json(self) -> str:
        """Serialize deterministically as pretty JSON with one terminal newline."""

        return json.dumps(self.to_dict(), indent=2, sort_keys=True, ensure_ascii=True) + "\n"

    @classmethod
    def from_dict(cls, value: object) -> DeliveryInventory:
        """Parse the complete strict inventory and reconcile all derived fields."""

        data = _mapping(
            value,
            {"schema_version", "inventory_id", "source_head", "slices", "totals"},
            field="delivery inventory",
        )
        if data["schema_version"] != INVENTORY_SCHEMA_VERSION:
            raise Sprint5InventoryError(
                f"unsupported inventory schema_version {data['schema_version']!r}"
            )
        raw_slices = data["slices"]
        if not isinstance(raw_slices, list):
            raise Sprint5InventoryError("delivery inventory slices must be an array")
        if len(raw_slices) != len(SPRINT_5_SLICES):
            raise Sprint5InventoryError(
                f"delivery inventory must contain exactly {len(SPRINT_5_SLICES)} slices"
            )
        result = cls(
            source_head=data["source_head"],
            slices=tuple(DeliverySlice.from_dict(item) for item in raw_slices),
        )
        totals = _mapping(
            data["totals"],
            {"category_counts", "changed_path_count"},
            field="delivery inventory totals",
        )
        changed_path_count = _integer(
            totals["changed_path_count"],
            field="delivery inventory changed_path_count",
            minimum=1,
            maximum=MAX_CHANGED_PATHS * len(SPRINT_5_SLICES),
        )
        raw_categories = _mapping(
            totals["category_counts"],
            set(result.category_counts),
            field="delivery inventory category_counts",
        )
        category_counts = {
            category: _integer(
                count,
                field=f"delivery inventory category_counts.{category}",
                minimum=1,
                maximum=MAX_CHANGED_PATHS * len(SPRINT_5_SLICES),
            )
            for category, count in raw_categories.items()
        }
        if changed_path_count != result.changed_path_count:
            raise Sprint5InventoryError("delivery inventory changed_path_count does not reconcile")
        if category_counts != result.category_counts:
            raise Sprint5InventoryError("delivery inventory category_counts do not reconcile")
        if not _is_lower_hex(data["inventory_id"], length=64):
            raise Sprint5InventoryError("inventory_id must be a full lowercase SHA-256 digest")
        if data["inventory_id"] != result.inventory_id:
            raise Sprint5InventoryError("delivery inventory identity does not reconcile")
        return result


@dataclass(frozen=True, slots=True)
class _DiffEntry:
    path: str
    status: str
    old_mode: str
    new_mode: str
    old_blob: str
    new_blob: str


@dataclass(frozen=True, slots=True)
class _CommitIdentity:
    commit: str
    tree: str
    parent: str


def _kill_process_tree(process: subprocess.Popen[bytes]) -> None:
    if os.name == "posix":
        with suppress(OSError):
            os.killpg(process.pid, signal.SIGKILL)
    else:
        with suppress(OSError):
            process.kill()


def _read_capped_stream(
    stream: Any,
    *,
    maximum: int,
    output: bytearray,
    overflow: threading.Event,
    failures: list[OSError],
    process: subprocess.Popen[bytes],
) -> None:
    try:
        while chunk := stream.read(64 * 1024):
            remaining = maximum + 1 - len(output)
            if remaining > 0:
                output.extend(chunk[:remaining])
            if len(output) > maximum or len(chunk) > max(remaining, 0):
                overflow.set()
                _kill_process_tree(process)
                return
    except OSError as exc:
        failures.append(exc)
        _kill_process_tree(process)
    finally:
        stream.close()


def _write_bounded_stdin(
    stream: Any,
    payload: bytes,
    *,
    failures: list[OSError],
    process: subprocess.Popen[bytes],
) -> None:
    try:
        offset = 0
        while offset < len(payload):
            written = stream.write(payload[offset : offset + 64 * 1024])
            if written is None or written < 1:
                raise OSError("short write to bounded Git subprocess")
            offset += written
        stream.flush()
    except (BrokenPipeError, OSError) as exc:
        failures.append(exc)
        _kill_process_tree(process)
    finally:
        stream.close()


def _run_git(
    checkout: Path,
    arguments: Sequence[str],
    *,
    maximum_stdout: int,
    input_bytes: bytes = b"",
) -> bytes:
    """Run one network-disabled Git command with hard resource bounds."""

    if (
        not 0 <= maximum_stdout <= MAX_GIT_OUTPUT_BYTES
        or len(arguments) > MAX_GIT_ARGUMENTS
        or len(input_bytes) > MAX_GIT_STDIN_BYTES
    ):
        raise Sprint5InventoryError("bounded Git command exceeds configured resource limits")
    for argument in arguments:
        if (
            not isinstance(argument, str)
            or "\x00" in argument
            or len(argument.encode("utf-8")) > MAX_GIT_ARGUMENT_BYTES
        ):
            raise Sprint5InventoryError("Git arguments must be bounded NUL-free text")

    environment = {
        "GIT_ATTR_NOSYSTEM": "1",
        "GIT_CONFIG_GLOBAL": os.devnull,
        "GIT_CONFIG_NOSYSTEM": "1",
        "GIT_NO_LAZY_FETCH": "1",
        "GIT_NO_REPLACE_OBJECTS": "1",
        "GIT_OPTIONAL_LOCKS": "0",
        "GIT_PROTOCOL_FROM_USER": "0",
        "GIT_TERMINAL_PROMPT": "0",
        "LC_ALL": "C",
        "PATH": os.environ.get("PATH", ""),
    }
    try:
        process = subprocess.Popen(
            ["git", "-C", os.fspath(checkout), *arguments],
            stdin=subprocess.PIPE if input_bytes else subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=environment,
            start_new_session=os.name == "posix",
        )
    except OSError as exc:
        raise Sprint5InventoryError("unable to start bounded local Git inspection") from exc
    if process.stdout is None or process.stderr is None:
        _kill_process_tree(process)
        process.wait()
        raise Sprint5InventoryError("unable to capture bounded local Git output")

    stdout = bytearray()
    stderr = bytearray()
    overflow = threading.Event()
    read_failures: list[OSError] = []
    write_failures: list[OSError] = []
    readers = (
        threading.Thread(
            target=_read_capped_stream,
            kwargs={
                "stream": process.stdout,
                "maximum": maximum_stdout,
                "output": stdout,
                "overflow": overflow,
                "failures": read_failures,
                "process": process,
            },
            daemon=True,
        ),
        threading.Thread(
            target=_read_capped_stream,
            kwargs={
                "stream": process.stderr,
                "maximum": MAX_GIT_STDERR_BYTES,
                "output": stderr,
                "overflow": overflow,
                "failures": read_failures,
                "process": process,
            },
            daemon=True,
        ),
    )
    for reader in readers:
        reader.start()
    writer: threading.Thread | None = None
    if input_bytes:
        if process.stdin is None:
            _kill_process_tree(process)
            process.wait()
            raise Sprint5InventoryError("unable to open bounded local Git input")
        writer = threading.Thread(
            target=_write_bounded_stdin,
            args=(process.stdin, input_bytes),
            kwargs={"failures": write_failures, "process": process},
            daemon=True,
        )
        writer.start()
    try:
        return_code = process.wait(timeout=GIT_TIMEOUT_SECONDS)
    except subprocess.TimeoutExpired as exc:
        _kill_process_tree(process)
        process.wait()
        for reader in readers:
            reader.join(timeout=1)
        if writer is not None:
            writer.join(timeout=1)
        raise Sprint5InventoryError("bounded local Git inspection timed out") from exc
    for reader in readers:
        reader.join()
    if writer is not None:
        writer.join()

    if read_failures:
        raise Sprint5InventoryError("unable to read bounded local Git output") from read_failures[0]
    if write_failures:
        raise Sprint5InventoryError("unable to supply bounded local Git input") from write_failures[
            0
        ]
    if overflow.is_set():
        raise Sprint5InventoryError("local Git inspection exceeded its output bound")
    if return_code != 0:
        diagnostic = bytes(stderr).decode("utf-8", errors="replace")
        diagnostic = " ".join(diagnostic.split())[:1000]
        raise Sprint5InventoryError(
            f"local Git inspection rejected frozen delivery evidence: {diagnostic or 'no detail'}"
        )
    return bytes(stdout)


def _single_ascii_line(payload: bytes, *, field: str, maximum: int) -> str:
    if len(payload) > maximum:
        raise Sprint5InventoryError(f"Git returned oversized {field}")
    try:
        text = payload.decode("ascii")
    except UnicodeError as exc:
        raise Sprint5InventoryError(f"Git returned non-ASCII {field}") from exc
    value = text.removesuffix("\n")
    if "\n" in value or "\r" in value or not value:
        raise Sprint5InventoryError(f"Git returned malformed {field}")
    return value


def _validate_checkout(repository_root: str | Path) -> Path:
    try:
        lexical = Path(os.path.abspath(os.fspath(repository_root)))
    except TypeError as exc:
        raise Sprint5InventoryError("repository_root must be a filesystem path") from exc
    if lexical.is_symlink() or not lexical.is_dir():
        raise Sprint5InventoryError("repository_root must be a real non-symlink directory")
    try:
        inside = _single_ascii_line(
            _run_git(
                lexical,
                ("rev-parse", "--is-inside-work-tree"),
                maximum_stdout=16,
            ),
            field="worktree status",
            maximum=16,
        )
    except Sprint5InventoryError as exc:
        raise Sprint5InventoryError("repository_root must name a Git worktree") from exc
    if inside != "true":
        raise Sprint5InventoryError("repository_root must be inside a Git worktree")
    root_text = _single_ascii_line(
        _run_git(
            lexical,
            ("rev-parse", "--show-toplevel"),
            maximum_stdout=4096,
        ),
        field="worktree root",
        maximum=4096,
    )
    root = Path(root_text)
    if root.is_symlink() or root.resolve() != lexical.resolve():
        raise Sprint5InventoryError("repository_root must name the exact Git worktree root")
    origin = _single_ascii_line(
        _run_git(
            root,
            ("config", "--local", "--get", "remote.origin.url"),
            maximum_stdout=1024,
        ),
        field="origin URL",
        maximum=1024,
    )
    if origin not in _EXPECTED_ORIGIN_URLS:
        raise Sprint5InventoryError("repository_root origin does not identify AlphaForge")
    return root


def _inspect_commit(checkout: Path, commit: str) -> _CommitIdentity:
    if not _is_lower_hex(commit):
        raise Sprint5InventoryError("commit inspection requires a full lowercase SHA-1")
    object_type = _single_ascii_line(
        _run_git(checkout, ("cat-file", "-t", commit), maximum_stdout=16),
        field="object type",
        maximum=16,
    )
    if object_type != "commit":
        raise Sprint5InventoryError(f"frozen object {commit} is not a commit")
    metadata = _run_git(
        checkout,
        (
            "show",
            "--no-patch",
            "--no-show-signature",
            "--format=format:%H%x00%T%x00%P",
            commit,
            "--",
        ),
        maximum_stdout=256,
    )
    parts = metadata.split(b"\x00")
    if len(parts) != 3:
        raise Sprint5InventoryError("Git returned malformed commit identity metadata")
    try:
        observed_commit, tree, parents_text = (part.decode("ascii") for part in parts)
    except UnicodeError as exc:
        raise Sprint5InventoryError("Git returned non-ASCII commit identity metadata") from exc
    parents = parents_text.split()
    if observed_commit != commit or not _is_lower_hex(tree):
        raise Sprint5InventoryError("Git returned the wrong frozen commit or tree identity")
    if len(parents) != 1 or not _is_lower_hex(parents[0]):
        raise Sprint5InventoryError("each frozen Sprint 5 slice must have exactly one parent")
    return _CommitIdentity(commit=observed_commit, tree=tree, parent=parents[0])


def _parse_raw_diff(payload: bytes) -> tuple[_DiffEntry, ...]:
    if len(payload) > MAX_GIT_OUTPUT_BYTES:
        raise Sprint5InventoryError("raw Git diff exceeds its output bound")
    tokens = payload.split(b"\x00")
    if not tokens or tokens[-1] != b"":
        raise Sprint5InventoryError("raw Git diff is not NUL terminated")
    records = tokens[:-1]
    if len(records) % 2 != 0:
        raise Sprint5InventoryError("raw Git diff contains an incomplete path record")
    if len(records) // 2 > MAX_CHANGED_PATHS:
        raise Sprint5InventoryError(f"raw Git diff exceeds {MAX_CHANGED_PATHS} paths")

    entries: list[_DiffEntry] = []
    observed_paths: set[str] = set()
    for position in range(0, len(records), 2):
        raw_metadata, raw_path = records[position : position + 2]
        fields = raw_metadata.split(b" ")
        if len(fields) != 5 or not fields[0].startswith(b":"):
            raise Sprint5InventoryError("raw Git diff contains malformed metadata")
        try:
            old_mode = fields[0][1:].decode("ascii")
            new_mode = fields[1].decode("ascii")
            old_blob = fields[2].decode("ascii")
            new_blob = fields[3].decode("ascii")
            status = fields[4].decode("ascii")
            path = raw_path.decode("ascii")
        except UnicodeError as exc:
            raise Sprint5InventoryError("raw Git diff fields must be ASCII") from exc
        path = _validate_repository_path(path)
        if path in observed_paths:
            raise Sprint5InventoryError(f"raw Git diff repeats changed path {path!r}")
        observed_paths.add(path)
        if status not in _SUPPORTED_STATUSES:
            raise Sprint5InventoryError(
                f"raw Git diff status {status!r} is unsupported; collection disables rename "
                "detection and records moves as one deletion plus one addition"
            )
        for name, mode in (("old", old_mode), ("new", new_mode)):
            if mode not in _REGULAR_MODES | {_ABSENT_MODE}:
                raise Sprint5InventoryError(
                    f"{name} mode {mode!r} is not a regular blob; symlinks and submodules fail closed"
                )
        if not _is_lower_hex(old_blob) or not _is_lower_hex(new_blob):
            raise Sprint5InventoryError("raw Git diff contains a malformed object identity")
        if status == "A":
            valid_shape = old_mode == _ABSENT_MODE and old_blob == _ZERO_OID
            valid_shape &= new_mode in _REGULAR_MODES and new_blob != _ZERO_OID
        elif status == "D":
            valid_shape = new_mode == _ABSENT_MODE and new_blob == _ZERO_OID
            valid_shape &= old_mode in _REGULAR_MODES and old_blob != _ZERO_OID
        else:
            valid_shape = old_mode in _REGULAR_MODES and new_mode in _REGULAR_MODES
            valid_shape &= old_blob != _ZERO_OID and new_blob != _ZERO_OID
        if not valid_shape:
            raise Sprint5InventoryError(
                f"raw Git diff has inconsistent {status} metadata for {path}"
            )
        entries.append(
            _DiffEntry(
                path=path,
                status=status,
                old_mode=old_mode,
                new_mode=new_mode,
                old_blob=old_blob,
                new_blob=new_blob,
            )
        )
    return tuple(sorted(entries, key=lambda item: item.path))


def _blob_sizes(checkout: Path, entries: Sequence[_DiffEntry]) -> dict[str, int]:
    object_ids = sorted(
        {
            object_id
            for entry in entries
            for object_id in (entry.old_blob, entry.new_blob)
            if object_id != _ZERO_OID
        }
    )
    if len(object_ids) > MAX_CHANGED_PATHS * 2:
        raise Sprint5InventoryError("raw Git diff references too many blob objects")
    request = "".join(f"{object_id}\n" for object_id in object_ids).encode("ascii")
    output = _run_git(
        checkout,
        ("cat-file", "--batch-check=%(objectname) %(objecttype) %(objectsize)"),
        maximum_stdout=max(1, len(object_ids) * 96),
        input_bytes=request,
    )
    lines = output.splitlines()
    if len(lines) != len(object_ids):
        raise Sprint5InventoryError("Git blob inventory did not reconcile with requested objects")
    sizes: dict[str, int] = {}
    for expected, line in zip(object_ids, lines, strict=True):
        fields = line.split(b" ")
        if len(fields) != 3:
            raise Sprint5InventoryError("Git returned malformed blob metadata")
        try:
            observed = fields[0].decode("ascii")
            object_type = fields[1].decode("ascii")
            size_text = fields[2].decode("ascii")
            size = int(size_text)
        except (UnicodeError, ValueError) as exc:
            raise Sprint5InventoryError("Git returned invalid blob metadata") from exc
        if observed != expected or object_type != "blob":
            raise Sprint5InventoryError("Git returned the wrong object or a non-blob path")
        if not 0 <= size <= MAX_TRACKED_BLOB_BYTES:
            raise Sprint5InventoryError(
                f"tracked blob exceeds the {MAX_TRACKED_BLOB_BYTES}-byte evidence bound"
            )
        sizes[observed] = size
    return sizes


def _inspect_slice(checkout: Path, spec: DeliverySliceSpec) -> DeliverySlice:
    identity = _inspect_commit(checkout, spec.commit)
    raw_diff = _run_git(
        checkout,
        (
            "diff-tree",
            "--no-commit-id",
            "--raw",
            "-r",
            "-z",
            "--abbrev=40",
            "--no-ext-diff",
            "--no-renames",
            identity.parent,
            identity.commit,
            "--",
        ),
        maximum_stdout=MAX_GIT_OUTPUT_BYTES,
    )
    entries = _parse_raw_diff(raw_diff)
    if not entries:
        raise Sprint5InventoryError(f"frozen sprint slice {spec.mr_group} changes no paths")
    sizes = _blob_sizes(checkout, entries)
    paths: list[DeliveryPath] = []
    for entry in entries:
        selected_mode = entry.old_mode if entry.status == "D" else entry.new_mode
        selected_blob = entry.old_blob if entry.status == "D" else entry.new_blob
        paths.append(
            DeliveryPath(
                path=entry.path,
                status=entry.status,
                mode=selected_mode,
                blob=selected_blob,
                bytes=sizes[selected_blob],
                category=_classify_path(entry.path),
            )
        )
    return DeliverySlice(
        mr_group=spec.mr_group,
        issue_number=spec.issue_number,
        commit=identity.commit,
        parent=identity.parent,
        tree=identity.tree,
        paths=tuple(paths),
    )


def build_sprint_5_inventory(repository_root: str | Path) -> DeliveryInventory:
    """Build the exact Sprint 5 inventory from a local AlphaForge object database.

    Args:
        repository_root: Exact root of an AlphaForge Git worktree. Passing a
            subdirectory, symlink, bare repository, or checkout with another
            origin fails closed.

    Returns:
        Immutable inventory whose summaries and identity reconcile exactly to
        the six frozen first-parent diffs.

    Raises:
        Sprint5InventoryError: If repository identity, Git objects, paths,
            modes, bounds, chain order, or reconciliation cannot be verified.
    """

    checkout = _validate_checkout(repository_root)
    slices = tuple(_inspect_slice(checkout, spec) for spec in SPRINT_5_SLICES)
    return DeliveryInventory(source_head=SOURCE_HEAD, slices=slices)


__all__ = [
    "DeliveryInventory",
    "DeliveryPath",
    "DeliverySlice",
    "EXPECTED_ORIGIN_URL",
    "INVENTORY_SCHEMA_VERSION",
    "SOURCE_HEAD",
    "SPRINT_5_SLICES",
    "Sprint5InventoryError",
    "build_sprint_5_inventory",
]
