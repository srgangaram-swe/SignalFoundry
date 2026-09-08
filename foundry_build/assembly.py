"""Lossless ancestry import and independently verifiable source manifests.

Only the fixed two public source identities are supported. A previously scanned,
hash-bound ledger is required; a caller cannot turn arbitrary findings into a pass.
Historical missing-notice findings are resolved only for the owner's exact four
approved trees. This is a recorded rights determination, not a legal verification.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import stat
from pathlib import Path
from typing import Any

from foundry_build.git import AssemblyError, git, oid, resolve

SOURCES = {"alphaforge": "AlphaForge", "signalattice": "Signalattice"}
APPROVED_TREES = frozenset(
    {
        "2b7d1d052b4750e216ecf02d5573617e8ec1f1fe",
        "60a3b01ddfcbdba1a3257b67d8ddeaf518d1ae13",
        "7b8d4cee76db53e4e05ed5ad6d6315477ea101b2",
        "ea4fa50c22fc5aed14fff9b44312f613804107c3",
    }
)
DETERMINATION = (
    "https://github.com/srgangaram-swe/AlphaForge/issues/79#issuecomment-5563813017"
)


def canonical(value: Any) -> bytes:
    return (
        json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n"
    ).encode()


def read_json(path: Path, maximum: int = 32 << 20) -> Any:
    """Reject symlinks, oversized documents, duplicate keys and non-JSON constants."""
    try:
        descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
        with os.fdopen(descriptor, "rb") as stream:
            if not stat.S_ISREG(os.fstat(stream.fileno()).st_mode):
                raise AssemblyError("unsafe-json-file")
            payload = stream.read(maximum + 1)
        if len(payload) > maximum:
            raise AssemblyError("json-byte-limit")
    except OSError as exc:
        raise AssemblyError("unsafe-json-file") from exc

    def unique(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise AssemblyError("duplicate-json-key")
            result[key] = value
        return result

    def reject(value: str) -> Any:
        raise AssemblyError("invalid-json-constant")

    try:
        return json.loads(payload, object_pairs_hook=unique, parse_constant=reject)
    except (ValueError, UnicodeError, RecursionError) as exc:
        raise AssemblyError("invalid-json-document") from exc


def archive_ref(source: str, ref: str) -> str:
    """Retain advertised refs as namespaced tags, without development branches."""
    if source not in SOURCES or not ref.startswith(
        ("refs/heads/", "refs/tags/", "refs/pull/")
    ):
        raise AssemblyError("unsupported-source-ref")
    if any(character.isspace() for character in ref) or ".." in ref:
        raise AssemblyError("unsafe-source-ref")
    if ref.startswith("refs/tags/"):
        return f"refs/tags/{source}/{ref.removeprefix('refs/tags/')}"
    return f"refs/tags/archive/{source}/{ref.removeprefix('refs/')}"


def resolve_findings(
    source: str, findings: list[dict[str, str]]
) -> list[dict[str, str]]:
    """Resolve only exact missing-notice records covered by the owner determination."""
    resolved = []
    for finding in findings:
        if (
            source != "alphaforge"
            or set(finding) != {"code", "tree"}
            or finding["code"] != "missing-license"
            or finding["tree"] not in APPROVED_TREES
        ):
            raise AssemblyError("unresolved-source-finding")
        if finding in resolved:
            raise AssemblyError("duplicate-source-finding")
        resolved.append(finding)
    return resolved


def source_record(mirror: Path, source: str, ledger_dir: Path) -> dict[str, Any]:
    """Bind source tip, every advertised ref and every original commit to the freeze.

    The scanned ledger is regenerated separately by the source's original verifier.
    All shards are authenticated before reading fields; refs are checked against Git.
    """
    manifest = read_json(ledger_dir / "manifest.json", 1 << 20)
    fields: dict[str, Any] = {}
    for shard in manifest["shards"]:
        name = shard["path"]
        if Path(name).name != name:
            raise AssemblyError("unsafe-shard-path")
        path = ledger_dir / name
        data = read_json(path, 1 << 20)
        if hashlib.sha256(path.read_bytes()).hexdigest() != shard["sha256"]:
            raise AssemblyError("ledger-shard-hash-mismatch")
        if shard["mapping"]:
            fields.setdefault(shard["field"], {}).update(data)
        else:
            if shard["field"] in fields:
                raise AssemblyError("duplicate-ledger-field")
            fields[shard["field"]] = data
    serialized = (
        json.dumps(
            fields,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        )
        + "\n"
    ).encode()
    if hashlib.sha256(serialized).hexdigest() != manifest["ledger_sha256"]:
        raise AssemblyError("ledger-identity-mismatch")
    if fields["source"] != source:
        raise AssemblyError("wrong-source-ledger")
    refs = dict(
        line.split(" ", 1)
        for line in git(mirror, "for-each-ref", "--format=%(refname) %(objectname)")
        .decode()
        .splitlines()
    )
    if refs != fields["refs"] or len(refs) > 2000:
        raise AssemblyError("source-freeze-mismatch")
    resolved = resolve_findings(source, fields["blockers"])
    commits = sorted(
        value for value, row in fields["objects"].items() if row["kind"] == "commit"
    )
    tip = oid(refs["refs/heads/dev"])
    return {
        "repository": f"srgangaram-swe/{SOURCES[source]}",
        "dev": tip,
        "tree": resolve(mirror, f"{tip}^{{tree}}"),
        "refs": {
            ref: {"oid": oid(value), "target": archive_ref(source, ref)}
            for ref, value in sorted(refs.items())
        },
        "commits": commits,
        "objects": fields["objects"],
        "ledger_sha256": manifest["ledger_sha256"],
        "resolved_findings": resolved,
        "active_blockers": [],
        "owner_determination": DETERMINATION if resolved else None,
    }


def import_sources(root: Path, backup: Path) -> dict[str, Any]:
    """Attach preserved histories and exact prefixed trees on a new work branch.

    Refuse existing package trees, a dirty tracked index, or a protected branch.
    No network, source mutation, history rewriting or protected-branch write occurs.
    The operation is resumable by inspection, not destructive rollback on failure.
    """
    branch = git(root, "branch", "--show-current").decode().strip()
    if not branch.startswith("feat/") or (root / "packages").exists():
        raise AssemblyError("unsafe-import-destination")
    if git(root, "status", "--porcelain", "--untracked-files=no"):
        raise AssemblyError("dirty-tracked-destination")
    records = {
        source: source_record(
            backup / f"{source}.git", source, backup / f"{source}-ledger"
        )
        for source in SOURCES
    }
    parent = resolve(root, "HEAD")
    original_tree = resolve(root, "HEAD^{tree}")
    for source, record in records.items():
        mirror = backup / f"{source}.git"
        # Explicit refspecs; no mirror push, no implicit tag overwrites.
        refspecs = [f"{ref}:{row['target']}" for ref, row in record["refs"].items()]
        git(root, "fetch", "--no-tags", str(mirror), *refspecs)
        tips = sorted(
            {
                resolve(root, f"{row['oid']}^{{commit}}")
                for row in record["refs"].values()
            }
        )
        independent = (
            git(root, "merge-base", "--independent", *tips).decode().splitlines()
        )
        for offset in range(0, len(independent), 16):
            parents = [parent, *independent[offset : offset + 16]]
            args = [part for value in parents for part in ("-p", oid(value))]
            parent = oid(
                git(
                    root,
                    "commit-tree",
                    original_tree,
                    *args,
                    "-m",
                    f"chore(archive): retain complete {source} ancestry "
                    "without deploying abandoned changes",
                )
                .decode()
                .strip()
            )
        git(root, "read-tree", f"--prefix=packages/{source}/", "-u", record["dev"])
    imported_tree = resolve(root, git(root, "write-tree").decode().strip())
    imported = oid(
        git(
            root,
            "commit-tree",
            imported_tree,
            "-p",
            parent,
            "-m",
            "feat(assembly): preserve complete source histories "
            "and exact package trees",
        )
        .decode()
        .strip()
    )
    git(root, "update-ref", f"refs/heads/{branch}", imported, resolve(root, "HEAD"))
    result = {"schema_version": 1, "import_commit": imported, "sources": records}
    verify(root, result)
    return result


def verify(root: Path, manifest: dict[str, Any]) -> dict[str, Any]:
    """Prove every commit/object identity and initial/current subtree equality.

    One reachability traversal, O(commits + objects); Git validates object checksums.
    Runtime parity is a separate mandatory check, never inferred from tree equality.
    """
    if manifest.get("schema_version") != 1 or set(manifest["sources"]) != set(SOURCES):
        raise AssemblyError("invalid-assembly-manifest")
    reachable = set(git(root, "rev-list", "HEAD").decode().splitlines())
    git(root, "fsck", "--full", "--no-dangling")
    summary = {}
    for source, record in manifest["sources"].items():
        if not set(record["commits"]) <= reachable or record["active_blockers"]:
            raise AssemblyError("missing-ancestry-or-blocked-source")
        resolve_findings(source, record["resolved_findings"])
        if (
            record["resolved_findings"]
            and record["owner_determination"] != DETERMINATION
        ):
            raise AssemblyError("missing-owner-determination")
        for ref in ("HEAD", oid(manifest["import_commit"])):
            if resolve(root, f"{ref}:packages/{source}") != record["tree"]:
                raise AssemblyError("source-subtree-mismatch")
        for ref, row in record["refs"].items():
            if row["target"] != archive_ref(source, ref):
                raise AssemblyError("invalid-archive-mapping")
            if resolve(root, row["target"]) != row["oid"]:
                raise AssemblyError("archive-ref-mismatch")
        # Hash uncompressed original objects, not just their advertised names.
        for value, row in record["objects"].items():
            if row["kind"] not in {"commit", "tree", "blob", "tag"}:
                raise AssemblyError("invalid-object-kind")
            content = git(root, "cat-file", row["kind"], oid(value), ceiling=8 << 20)
            if (
                len(content) != row["bytes"]
                or hashlib.sha256(content).hexdigest() != row["sha256"]
            ):
                raise AssemblyError("source-object-mismatch")
        summary[source] = {
            "commits": len(record["commits"]),
            "objects": len(record["objects"]),
            "refs": len(record["refs"]),
            "tree": record["tree"],
            "active_blockers": 0,
        }
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("import", "verify"))
    parser.add_argument("--backup", type=Path)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    path = root / "provenance" / "assembly.json"
    try:
        if args.command == "import":
            if args.backup is None or path.exists():
                raise AssemblyError("import-requires-new-manifest-and-backup")
            result = import_sources(root, args.backup.resolve())
            path.parent.mkdir(exist_ok=True)
            with path.open("xb") as stream:
                stream.write(canonical(result))
            print(
                json.dumps(
                    {
                        source: len(record["commits"])
                        for source, record in result["sources"].items()
                    }
                )
            )
        else:
            print(json.dumps(verify(root, read_json(path)), sort_keys=True))
    except (AssemblyError, OSError, KeyError, TypeError, UnicodeError) as exc:
        parser.exit(2, f"assembly failed ({type(exc).__name__}); no gate bypassed\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
