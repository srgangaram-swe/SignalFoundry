"""Give an unchanged prefixed package its original offline Git provenance context.

No submodule and no external checkout: tests/installations run in packages/<name>.
The generated .git pointer addresses independently copied object storage under the
unified .git directory. This lets unchanged source tools resolve original tags,
root-relative historical paths and hashes. Context creation never fetches a network.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any

from foundry_build.assembly import SOURCES, canonical, read_json
from foundry_build.git import AssemblyError, git, oid, resolve


def prepare(root: Path, source: str, manifest: dict[str, Any]) -> Path:
    """Create once under exclusive reservation; reject drift and foreign .git files.

    Trusted local parent and cooperating writers are required. Copies are complete,
    never alternates/hardlinks; each original gate can operate independently. Time
    and Git output are bounded. Existing user Git state is never overwritten.
    """
    root = root.resolve(strict=True)
    if source not in SOURCES:
        raise AssemblyError("unknown-source")
    record = manifest["sources"][source]
    package = root / "packages" / source
    if package.is_symlink() or not package.is_dir():
        raise AssemblyError("unsafe-package-root")
    if resolve(root, f"HEAD:packages/{source}") != record["tree"]:
        raise AssemblyError("package-tree-drift")
    if git(root, "diff", "HEAD", "--", f"packages/{source}"):
        raise AssemblyError("dirty-package-tree")
    gitdir = Path(git(root, "rev-parse", "--absolute-git-dir").decode().strip())
    parent = gitdir / "source-contexts"
    parent.mkdir(exist_ok=True)
    destination = parent / f"{source}.git"
    marker = package / ".git"
    identity = hashlib.sha256(canonical(record)).hexdigest()
    pointer = f"gitdir: {destination}\n".encode()
    if marker.exists() or marker.is_symlink():
        if (
            marker.is_symlink()
            or not marker.is_file()
            or marker.read_bytes() != pointer
        ):
            raise AssemblyError("foreign-package-git-context")
        if (destination / "foundry-identity").read_text().strip() != identity:
            raise AssemblyError("stale-package-git-context")
        if resolve(package, "HEAD") != record["dev"]:
            raise AssemblyError("package-context-head-drift")
        return package
    reservation = parent / f".{source}.reservation"
    try:
        reservation.mkdir()
    except OSError as exc:
        raise AssemblyError("package-context-reserved") from exc
    stage: Path | None = None
    try:
        if destination.exists() or destination.is_symlink():
            raise AssemblyError("orphan-package-context")
        stage = Path(tempfile.mkdtemp(prefix=f".{source}-", dir=parent))
        # clone into a new child; the enclosing staging reservation remains owned.
        storage = stage / "repository.git"
        git(root, "clone", "--bare", "--no-hardlinks", str(root), str(storage))
        command = ("--git-dir", str(storage))
        current_refs = (
            git(root, *command, "for-each-ref", "--format=%(refname)")
            .decode()
            .splitlines()
        )
        for ref in current_refs:
            git(root, *command, "update-ref", "-d", ref)
        for ref, row in record["refs"].items():
            value = oid(row["oid"])
            git(root, *command, "update-ref", ref, value)
            if ref.startswith("refs/heads/"):
                git(
                    root,
                    *command,
                    "update-ref",
                    ref.replace("refs/heads/", "refs/remotes/origin/", 1),
                    value,
                )
        git(root, *command, "symbolic-ref", "HEAD", "refs/heads/dev")
        git(root, *command, "config", "core.bare", "false")
        git(root, *command, "config", "core.worktree", str(package))
        git(
            root,
            *command,
            "config",
            "remote.origin.url",
            f"https://github.com/{record['repository']}.git",
        )
        git(root, *command, "read-tree", oid(record["dev"]))
        (storage / "foundry-identity").write_text(identity + "\n")
        os.rename(storage, destination)
        # Exclusive create prevents replacing another actor's administrative file.
        with marker.open("xb") as stream:
            stream.write(pointer)
        if resolve(package, "HEAD") != record["dev"]:
            raise AssemblyError("package-context-verification-failed")
    except OSError as exc:
        raise AssemblyError("package-context-io-failure") from exc
    finally:
        if stage is not None:
            shutil.rmtree(stage)
        # Keep an already published context on a late failure for explicit recovery;
        # never unlink a marker that might belong to a concurrent writer.
        reservation.rmdir()
    return package


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", choices=tuple(SOURCES))
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    try:
        prepare(root, args.source, read_json(root / "provenance" / "assembly.json"))
    except (AssemblyError, OSError, KeyError, TypeError) as exc:
        parser.exit(2, f"source context refused ({type(exc).__name__})\n")
    print(json.dumps({"source": args.source, "offline_context": "verified"}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
