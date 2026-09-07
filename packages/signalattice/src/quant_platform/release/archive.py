"""Bounded, complete release transport independent of GitHub or credentials.

The ZIP preserves relative paths for the console, schemas, evidence and Python
distributions. Verification compares every downloaded member to the build stage;
it never extracts an untrusted archive or trusts remote asset counts alone.
"""

from __future__ import annotations

import hashlib
import stat
from pathlib import Path, PurePosixPath
from zipfile import ZIP_STORED, BadZipFile, ZipFile, ZipInfo

from quant_platform.release.inventory import (
    MAX_SUBJECT_BYTES,
    MAX_SUBJECTS,
    InventoryError,
    Subject,
    build_inventory,
    compare_inventories,
)

MAX_ARCHIVE_BYTES = 1024 * 1024 * 1024


def pack_stage(stage: Path, destination: Path) -> None:
    """Write a deterministic, exclusively created ZIP including all manifests.

    Refuse symlinks, a destination inside the stage, existing output, or total
    content above 1 GiB. A failed write removes only the file created here.
    """
    if destination.resolve().is_relative_to(stage.resolve()):
        raise InventoryError("archive destination must be outside the stage")
    subjects = build_inventory(stage)
    if not subjects or sum(item.size_bytes for item in subjects) > MAX_ARCHIVE_BYTES:
        raise InventoryError("archive must have bounded nonempty contents")
    with destination.open("xb") as output:
        try:
            with ZipFile(output, "w", compression=ZIP_STORED) as archive:
                for subject in subjects:
                    entry = ZipInfo(subject.path, date_time=(1980, 1, 1, 0, 0, 0))
                    entry.create_system = 3
                    entry.external_attr = (stat.S_IFREG | 0o644) << 16
                    with (
                        (stage / subject.path).open("rb") as source,
                        archive.open(entry, "w") as target,
                    ):
                        while block := source.read(1024 * 1024):
                            target.write(block)
        except BaseException:
            destination.unlink(missing_ok=True)
            raise


def archive_inventory(path: Path) -> tuple[Subject, ...]:
    """Hash bounded ZIP members without extraction; reject unsafe or duplicate names."""
    if path.is_symlink() or path.stat().st_size > MAX_ARCHIVE_BYTES + 1024 * 1024:
        raise InventoryError("archive is a symlink or exceeds its size limit")
    try:
        with ZipFile(path) as archive:
            entries = archive.infolist()
            if not entries or len(entries) > MAX_SUBJECTS:
                raise InventoryError("archive member count outside limits")
            seen: set[str] = set()
            total = 0
            result: list[Subject] = []
            for entry in entries:
                name = entry.filename
                parts = PurePosixPath(name).parts
                if (
                    not name
                    or len(name) > 512
                    or name.startswith("/")
                    or "\\" in name
                    or any(ord(character) < 32 for character in name)
                    or any(part in {".", ".."} for part in parts)
                    or str(PurePosixPath(name)) != name
                    or stat.S_IFMT(entry.external_attr >> 16) != stat.S_IFREG
                    or name.casefold() in seen
                    or entry.flag_bits & 1
                ):
                    raise InventoryError("unsafe or duplicate archive member")
                seen.add(name.casefold())
                total += entry.file_size
                if entry.file_size > MAX_SUBJECT_BYTES or total > MAX_ARCHIVE_BYTES:
                    raise InventoryError("archive expanded content exceeds limits")
                digest = hashlib.sha256()
                read = 0
                with archive.open(entry) as source:
                    while block := source.read(1024 * 1024):
                        read += len(block)
                        if read > entry.file_size:
                            raise InventoryError("archive member exceeds declared size")
                        digest.update(block)
                if read != entry.file_size:
                    raise InventoryError("archive member is truncated")
                result.append(Subject(name, digest.hexdigest(), read))
            return tuple(sorted(result, key=lambda item: item.path))
    except (BadZipFile, RuntimeError, NotImplementedError) as error:
        raise InventoryError("invalid or unsupported release archive") from error


def verify_archive(stage: Path, archive: Path) -> None:
    """Reject any missing, extra, renamed or changed downloaded stage member."""
    difference = compare_inventories(build_inventory(stage), archive_inventory(archive))
    if not difference.clean:
        raise InventoryError(f"downloaded archive differs: {difference.describe()}")


def verify_downloads(expected: Path, downloaded: Path) -> None:
    """Compare the complete flat GitHub asset set to the independently staged bytes."""
    difference = compare_inventories(build_inventory(expected), build_inventory(downloaded))
    if not difference.clean:
        raise InventoryError(f"downloaded release differs: {difference.describe()}")
