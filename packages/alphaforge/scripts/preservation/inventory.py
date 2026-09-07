"""Deterministic committed-object ledger and fail-closed migration decision."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

from scripts.preservation.contracts import category, findings, static_interfaces, validate_paths
from scripts.preservation.git import DEFAULT_LIMITS, Git, Limits, PreservationError, oid


def canonical(value: Any) -> bytes:
    """Stable UTF-8 JSON; non-finite numbers are never admitted."""
    return (
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False)
        + "\n"
    ).encode()


def sha256(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def tree_entries(git: Git, tree: str) -> list[dict[str, str]]:
    """Inventory every recursively tracked leaf, including unsupported modes."""
    result = []
    try:
        for raw in git.run("ls-tree", "-rz", oid(tree)).split(b"\0"):
            if not raw:
                continue
            header, path = raw.split(b"\t", 1)
            mode, kind, value = header.decode("ascii").split(" ")
            result.append(
                {"path": path.decode("utf-8"), "mode": mode, "kind": kind, "oid": oid(value)}
            )
    except (UnicodeError, ValueError) as exc:
        raise PreservationError("malformed-tree") from exc
    if len(result) > git.limits.paths:
        raise PreservationError("path-count-limit")
    validate_paths([row["path"] for row in result])
    return sorted(result, key=lambda row: row["path"])


def inventory(
    root: Path, source: str, licenses: dict[str, str], limits: Limits = DEFAULT_LIMITS
) -> dict[str, Any]:
    """Freeze all mirror refs, object identities, tip paths and static capabilities.

    Complexity is linear in distinct reachable object bytes plus historical tree
    entries and Python AST nodes. Blob bytes are discarded after hashing/scanning;
    source code is never serialized. No working-tree/ignored data is read.
    Historical object and root-tree coverage is complete; static declarations
    are not a claim to execute arbitrary historical APIs.
    """
    if source not in {"alphaforge", "signalattice"}:
        raise PreservationError("unknown-source")
    if not licenses or any(len(key) != 64 or not value for key, value in licenses.items()):
        raise PreservationError("invalid-license-policy")
    git = Git(root, limits)
    refs = git.refs()
    git.run("fsck", "--full", "--no-reflogs", "--no-dangling")
    objects = sorted(
        set(git.run("rev-list", "--objects", "--no-object-names", "--all").decode().splitlines())
    )
    if len(objects) > limits.objects:
        raise PreservationError("object-count-limit")
    metadata: dict[str, dict[str, Any]] = {}
    total = 0
    for value in objects:
        oid(value)
        kind = git.run("cat-file", "-t", value).decode().strip()
        size = int(git.run("cat-file", "-s", value))
        if size > limits.blob_bytes:
            raise PreservationError("object-byte-limit")
        total += size
        if total > limits.total_blob_bytes:
            raise PreservationError("total-object-byte-limit")
        content = git.run("cat-file", kind, value, ceiling=limits.blob_bytes)
        if len(content) != size:
            raise PreservationError("object-size-mismatch")
        if (
            hashlib.sha1(f"{kind} {size}\0".encode() + content, usedforsecurity=False).hexdigest()
            != value
        ):
            raise PreservationError("object-hash-mismatch")
        record: dict[str, Any] = {"kind": kind, "bytes": size, "sha256": sha256(content)}
        if kind == "commit":
            first = content.split(b"\n", 1)[0]
            if not first.startswith(b"tree "):
                raise PreservationError("malformed-commit")
            record["tree"] = oid(first[5:].decode("ascii"))
        if kind == "blob":
            record["content_findings"] = findings("", content)
        metadata[value] = record
    tips: dict[str, str] = {}
    trees: dict[str, list[dict[str, Any]]] = {}
    blockers: list[dict[str, str]] = []
    interfaces: dict[str, list[dict[str, Any]]] = {}
    path_count = 0
    interface_count = 0
    for ref, value in refs.items():
        # Peel annotated tags to a tree; tags to blobs are retained but block import.
        try:
            tree = oid(git.run("rev-parse", "--verify", f"{value}^{{tree}}").decode().strip())
        except PreservationError:
            blockers.append({"ref": ref, "code": "non-tree-ref"})
            continue
        tips[ref] = tree
    historical_trees = {row["tree"] for row in metadata.values() if row["kind"] == "commit"}
    for tree in sorted(historical_trees | set(tips.values())):
        rows: list[dict[str, Any]] = []
        found_license = False
        for entry in tree_entries(git, tree):
            path_count += 1
            if path_count > limits.path_records:
                raise PreservationError("total-path-record-limit")
            path, blob = entry["path"], entry["oid"]
            row: dict[str, Any] = {
                **entry,
                "target": f"packages/{source}/{path}",
                "category": category(path),
            }
            if entry["mode"] not in {"100644", "100755"} or entry["kind"] != "blob":
                blockers.append({"tree": tree, "path": path, "code": "unsupported-file-mode"})
            else:
                data = metadata[blob]
                row["sha256"] = data["sha256"]
                codes = findings(path, b"") + data["content_findings"]
                if path.endswith("/.gitkeep") and data["bytes"] == 0:
                    codes = [code for code in codes if code != "raw-or-runtime-data"]
                if path == "LICENSE":
                    found_license = True
                    row["license_review"] = licenses.get(data["sha256"], "UNREVIEWED")
                    if row["license_review"] == "UNREVIEWED":
                        codes.append("unreviewed-license")
                if path.endswith(".py") and blob not in interfaces and not data["content_findings"]:
                    content = git.run("cat-file", "blob", blob, ceiling=limits.blob_bytes)
                    interfaces[blob] = static_interfaces(path, content, limits.ast_nodes)
                    interface_count += len(interfaces[blob])
                    if interface_count > limits.interface_records:
                        raise PreservationError("total-interface-record-limit")
                for code in sorted(set(codes)):
                    blockers.append({"tree": tree, "path": path, "code": code})
            rows.append(row)
        if not found_license:
            blockers.append({"tree": tree, "code": "missing-license"})
        trees[tree] = rows
    # Historical blob findings are not dismissed because they disappeared at tips.
    for value, data in metadata.items():
        for code in data.get("content_findings", []):
            blockers.append({"oid": value, "code": f"historical-{code}"})
    if refs != git.refs():
        raise PreservationError("source-ref-race")
    return {
        "schema_version": 1,
        "source": source,
        "evidence_class": "offline-static-migration-plan",
        "refs": refs,
        "ref_trees": tips,
        "objects": metadata,
        "trees": trees,
        "interfaces": interfaces,
        "license_policy": licenses,
        "blockers": sorted(blockers, key=lambda row: canonical(row)),
        "migration_gate": "BLOCKED" if blockers else "STATIC_CHECKS_PASS_RUNTIME_PARITY_PENDING",
        "summary": {
            "refs": len(refs),
            "objects": len(metadata),
            "object_bytes": total,
            "tip_trees": len(set(tips.values())),
            "historical_trees": len(historical_trees),
            "historical_path_records": sum(map(len, trees.values())),
            "interface_records": sum(map(len, interfaces.values())),
            "blockers": len(blockers),
            "object_kinds": dict(sorted(Counter(row["kind"] for row in metadata.values()).items())),
        },
    }


def require_migration_clear(ledger: dict[str, Any]) -> None:
    """A planned mapping is never approval to import or a runtime parity claim."""
    if ledger.get("schema_version") != 1 or ledger.get("blockers") != []:
        raise PreservationError("migration-blocked")
    if ledger.get("migration_gate") != "STATIC_CHECKS_PASS_RUNTIME_PARITY_PENDING":
        raise PreservationError("invalid-migration-state")
