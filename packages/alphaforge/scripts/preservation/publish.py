"""Exclusive publication of safe ledger shards, plus deterministic verification."""

from __future__ import annotations

import json
import os
import re
import shutil
import tempfile
from pathlib import Path
from typing import Any

from scripts.preservation.git import PreservationError
from scripts.preservation.inventory import canonical, sha256


def _read_bounded(path: Path, limit: int) -> bytes:
    if path.is_symlink():
        raise PreservationError("symlink-publication")
    with path.open("rb") as stream:
        content = stream.read(limit + 1)
    if len(content) > limit:
        raise PreservationError("publication-byte-limit")
    return content


def publish(ledger: dict[str, Any], destination: Path) -> dict[str, Any]:
    """Publish content-addressed JSON shards; refuse existing destinations.

    A sibling reservation directory prevents competing compliant writers. Every
    file is bounded below the repository's one-MiB artifact policy. Atomic rename
    publishes only complete output; no raw source blobs or absolute paths appear.
    """
    if any(
        not isinstance(field, str) or re.fullmatch(r"[a-z_]+", field) is None for field in ledger
    ):
        raise PreservationError("invalid-ledger-field")
    parent = destination.parent.resolve()
    destination = parent / destination.name
    if not parent.is_dir() or destination.exists() or destination.is_symlink():
        raise PreservationError("destination-unavailable")
    reservation = parent / f".{destination.name}.reservation"
    try:
        reservation.mkdir()
    except OSError as exc:
        raise PreservationError("destination-reserved") from exc
    stage: Path | None = None
    try:
        stage = Path(tempfile.mkdtemp(prefix=".preservation-", dir=parent))
        manifest: dict[str, Any] = {
            "schema_version": 1,
            "ledger_sha256": sha256(canonical(ledger)),
            "shards": [],
        }
        for field in sorted(ledger):
            value = ledger[field]
            items = list(sorted(value.items())) if isinstance(value, dict) else None
            # Each keyed record stands alone: reproducible and reviewable diffs.
            groups: list[Any] = []
            if items:
                batch: dict[str, Any] = {}
                size = 0
                for key, item in items:
                    entry_size = len(canonical({key: item}))
                    if size + entry_size > 800_000 and batch:
                        groups.append(batch)
                        batch, size = {}, 0
                    batch[key] = item
                    size += entry_size
                groups.append(batch)
            else:
                groups = [value]
            for number, group in enumerate(groups):
                payload = canonical(group)
                if len(payload) > 950_000:
                    raise PreservationError("publication-shard-limit")
                name = f"{field}-{number:06d}.json"
                (stage / name).write_bytes(payload)
                manifest["shards"].append(
                    {
                        "field": field,
                        "path": name,
                        "sha256": sha256(payload),
                        "mapping": items is not None,
                    }
                )
        (stage / "manifest.json").write_bytes(canonical(manifest))
        if destination.exists() or destination.is_symlink():
            raise PreservationError("destination-race")
        os.rename(stage, destination)
        stage = None
        return manifest
    except OSError as exc:
        raise PreservationError("publication-io-failure") from exc
    finally:
        if stage is not None:
            shutil.rmtree(stage)
        reservation.rmdir()


def read_ledger(directory: Path) -> dict[str, Any]:
    """Verify membership, shard digests and reconstructed ledger identity."""
    try:
        manifest_path = directory / "manifest.json"
        if manifest_path.is_symlink() or manifest_path.stat().st_size > 16 * 1024 * 1024:
            raise PreservationError("manifest-byte-limit")
        manifest_bytes = _read_bounded(manifest_path, 16 * 1024 * 1024)
        manifest = json.loads(manifest_bytes)
        if not isinstance(manifest, dict) or manifest.get("schema_version") != 1:
            raise PreservationError("manifest-schema")
        if not isinstance(manifest.get("shards"), list) or len(manifest["shards"]) > 4096:
            raise PreservationError("shard-count-limit")
        result: dict[str, Any] = {}
        expected = {"manifest.json"}
        total_bytes = 0
        for shard in manifest["shards"]:
            name = shard["path"]
            if (
                not isinstance(name, str)
                or re.fullmatch(r"[a-z_]+-[0-9]{6}\.json", name) is None
                or name in expected
            ):
                raise PreservationError("invalid-shard-path")
            expected.add(name)
            path = directory / name
            if path.is_symlink() or path.stat().st_size > 950_000:
                raise PreservationError("invalid-shard-file")
            content = _read_bounded(path, 950_000)
            total_bytes += len(content)
            if total_bytes > 128 * 1024 * 1024:
                raise PreservationError("ledger-byte-limit")
            if sha256(content) != shard["sha256"]:
                raise PreservationError("shard-hash-mismatch")
            value = json.loads(content)
            field = shard["field"]
            if shard["mapping"]:
                current = result.setdefault(field, {})
                if set(current).intersection(value):
                    raise PreservationError("duplicate-shard-key")
                current.update(value)
            else:
                if field in result:
                    raise PreservationError("duplicate-shard-field")
                result[field] = value
        if {path.name for path in directory.iterdir()} != expected:
            raise PreservationError("shard-membership-mismatch")
        if sha256(canonical(result)) != manifest["ledger_sha256"]:
            raise PreservationError("ledger-hash-mismatch")
        return result
    except (OSError, KeyError, TypeError, ValueError, RecursionError) as exc:
        raise PreservationError("invalid-ledger-publication") from exc
