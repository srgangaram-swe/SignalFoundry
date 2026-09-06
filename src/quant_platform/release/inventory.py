"""Content inventory for a release: what was built, and exactly which bytes.

SF-S5-SL-MR7. An inventory answers one question -- *are these the bytes that
were built?* -- and it has to answer it for every artifact, not the ones someone
remembered to list.

Detection properties, all tested:

* a **missing** subject is detected, because the inventory is the expectation;
* an **extra** file is detected, because the inventory is closed rather than a
  minimum;
* a **renamed** file is detected as one missing plus one extra rather than
  silently matched by digest, since a wheel under the wrong filename is not the
  same release;
* a **truncated** or **substituted** file is detected by digest;
* a **one-byte modification** is detected, which is the weakest tamper worth
  naming explicitly because a size-only check would miss it.

Paths are recorded relative to the staging root with forward slashes, so an
inventory produced on one platform verifies on another.
"""

from __future__ import annotations

import hashlib
import os
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

#: Read size for hashing. Large enough to be efficient, small enough that a
#: pathological artifact cannot force an unbounded allocation.
_CHUNK_BYTES: Final = 1024 * 1024

#: Refusal thresholds, not tuning knobs.
MAX_SUBJECTS: Final = 512
MAX_SUBJECT_BYTES: Final = 512 * 1024 * 1024


class InventoryError(ValueError):
    """Raised when an inventory cannot be built or does not verify."""


@dataclass(frozen=True, slots=True)
class Subject:
    """One release artifact, identified by path and content."""

    path: str
    sha256: str
    size_bytes: int

    def to_dict(self) -> dict[str, Any]:
        """Return the JSON-friendly record."""
        return {"path": self.path, "sha256": self.sha256, "size_bytes": self.size_bytes}

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> Subject:
        """Rebuild a subject from its record.

        Raises:
            InventoryError: On a malformed record.
        """
        try:
            path = payload["path"]
            digest = payload["sha256"]
            size = payload["size_bytes"]
        except KeyError as error:
            raise InventoryError(f"subject record is missing {error}") from error
        if not isinstance(path, str) or not path:
            raise InventoryError("subject path must be a non-empty string")
        if not isinstance(digest, str) or len(digest) != 64 or not _is_hex(digest):
            raise InventoryError(f"subject {path!r} has a malformed digest")
        if isinstance(size, bool) or not isinstance(size, int) or size < 0:
            raise InventoryError(f"subject {path!r} has a malformed size")
        return cls(path=path, sha256=digest, size_bytes=size)


def _is_hex(value: str) -> bool:
    """Return whether a string is lowercase hexadecimal."""
    return all(character in "0123456789abcdef" for character in value)


def digest_file(path: Path) -> tuple[str, int]:
    """Return the SHA-256 and byte size of one file.

    Streamed rather than read whole: an artifact set includes container layers,
    and reading one into memory to hash it is an avoidable ceiling.

    Raises:
        InventoryError: If the file is unreadable or above the size ceiling.
    """
    try:
        size = path.stat().st_size
    except OSError as error:
        raise InventoryError(f"cannot stat {path}") from error
    if size > MAX_SUBJECT_BYTES:
        raise InventoryError(f"{path.name} is {size} bytes, above the subject ceiling")
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            while chunk := handle.read(_CHUNK_BYTES):
                digest.update(chunk)
    except OSError as error:
        raise InventoryError(f"cannot read {path}") from error
    return digest.hexdigest(), size


def build_inventory(root: Path, *, exclude: Iterable[str] = ()) -> tuple[Subject, ...]:
    """Enumerate every file under ``root`` as a sorted subject list.

    Symlinks are refused rather than followed: a staging directory is a set of
    regular files, and a link would let the inventory describe content that
    lives somewhere else.

    Args:
        root: Staging directory holding the built artifacts.
        exclude: Relative paths to omit -- used for the manifest files that
            record the inventory itself, which cannot contain their own digest.

    Raises:
        InventoryError: On a missing root, a symlink, or too many subjects.
    """
    base = Path(root)
    if not base.is_dir():
        raise InventoryError(f"staging root {base} is not a directory")
    if base.is_symlink():
        raise InventoryError("staging root must not be a symlink")
    excluded = {item.replace(os.sep, "/") for item in exclude}

    subjects: list[Subject] = []
    for current, directories, files in os.walk(base, followlinks=False):
        directories.sort()
        if any((Path(current) / name).is_symlink() for name in directories):
            raise InventoryError("staging contains a symlink directory")
        for name in sorted(files):
            candidate = Path(current) / name
            if candidate.is_symlink():
                raise InventoryError(
                    f"{candidate.name} is a symlink; a release stage holds regular files so "
                    "that its inventory describes the bytes it actually ships"
                )
            relative = str(candidate.relative_to(base)).replace(os.sep, "/")
            if relative in excluded:
                continue
            digest, size = digest_file(candidate)
            subjects.append(Subject(path=relative, sha256=digest, size_bytes=size))
            if len(subjects) > MAX_SUBJECTS:
                raise InventoryError(f"release stage exceeds {MAX_SUBJECTS} subjects")
    return tuple(sorted(subjects, key=lambda item: item.path))


@dataclass(frozen=True, slots=True)
class InventoryDifference:
    """What changed between an expected inventory and an observed one."""

    missing: tuple[str, ...]
    unexpected: tuple[str, ...]
    modified: tuple[str, ...]

    @property
    def clean(self) -> bool:
        """Whether the observed set matches the expectation exactly."""
        return not (self.missing or self.unexpected or self.modified)

    def describe(self) -> str:
        """Return a bounded human-readable summary."""
        parts: list[str] = []
        if self.missing:
            parts.append(f"missing: {', '.join(self.missing[:10])}")
        if self.unexpected:
            parts.append(f"unexpected: {', '.join(self.unexpected[:10])}")
        if self.modified:
            parts.append(f"modified: {', '.join(self.modified[:10])}")
        return "; ".join(parts) if parts else "inventory matches"


def compare_inventories(
    expected: Sequence[Subject], observed: Sequence[Subject]
) -> InventoryDifference:
    """Compare two inventories by path and content.

    Matching is by path first, then digest. A file that moved is reported as one
    missing and one unexpected rather than matched by digest, because a wheel
    published under the wrong name is not the release the inventory describes.
    """
    expected_by_path = {item.path: item for item in expected}
    observed_by_path = {item.path: item for item in observed}
    missing = tuple(sorted(set(expected_by_path) - set(observed_by_path)))
    unexpected = tuple(sorted(set(observed_by_path) - set(expected_by_path)))
    modified = tuple(
        sorted(
            path
            for path in set(expected_by_path) & set(observed_by_path)
            if expected_by_path[path].sha256 != observed_by_path[path].sha256
            or expected_by_path[path].size_bytes != observed_by_path[path].size_bytes
        )
    )
    return InventoryDifference(missing=missing, unexpected=unexpected, modified=modified)


def assert_inventory_matches(expected: Sequence[Subject], root: Path) -> None:
    """Refuse unless the staging root reproduces the expected inventory exactly.

    Raises:
        InventoryError: Describing every difference found.
    """
    observed = build_inventory(root)
    difference = compare_inventories(expected, observed)
    if not difference.clean:
        raise InventoryError(f"release inventory does not verify: {difference.describe()}")


def inventory_to_dicts(subjects: Sequence[Subject]) -> list[dict[str, Any]]:
    """Return the JSON-friendly inventory, in deterministic path order."""
    return [item.to_dict() for item in sorted(subjects, key=lambda entry: entry.path)]


def inventory_from_dicts(payload: Sequence[Mapping[str, Any]]) -> tuple[Subject, ...]:
    """Rebuild an inventory from its records.

    Raises:
        InventoryError: On a malformed record or a duplicate path.
    """
    subjects = tuple(Subject.from_dict(item) for item in payload)
    paths = [item.path for item in subjects]
    if len(set(paths)) != len(paths):
        raise InventoryError("inventory contains a duplicate path")
    return subjects


__all__ = [
    "MAX_SUBJECTS",
    "MAX_SUBJECT_BYTES",
    "InventoryDifference",
    "InventoryError",
    "Subject",
    "assert_inventory_matches",
    "build_inventory",
    "compare_inventories",
    "digest_file",
    "inventory_from_dicts",
    "inventory_to_dicts",
]
