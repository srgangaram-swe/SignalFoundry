"""Explicit network adapter: freeze allowlisted public metadata through gh API.

Use only after gh auth status confirms the owner. Authentication remains in the
existing credential store. JSON contains no bodies, comments, or credentials.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

from scripts.preservation.git import Git, PreservationError, bounded_run
from scripts.preservation.github import capture
from scripts.preservation.inventory import canonical, oid


def freeze(repository: str, mirror: Path) -> dict[str, Any]:
    """Cross-check REST branch/tag identities and every advertised Git ref."""
    if repository not in {"srgangaram-swe/AlphaForge", "srgangaram-swe/Signalattice"}:
        raise PreservationError("unknown-github-repository")
    url = f"https://github.com/{repository}.git"
    deadline = time.monotonic() + 180
    before = bounded_run(["git", "ls-remote", "--refs", url], Path.cwd(), 30, 2_000_000)

    def request(endpoint: str) -> Any:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise PreservationError("metadata-timeout")
        content = bounded_run(["gh", "api", endpoint], Path.cwd(), min(30, remaining), 8_000_000)
        return json.loads(content)

    result = capture(repository, request)
    after = bounded_run(["git", "ls-remote", "--refs", url], Path.cwd(), 30, 2_000_000)
    if before != after:
        raise PreservationError("advertised-ref-race")
    advertised = {}
    for line in before.decode().splitlines():
        value, ref = line.split("\t")
        advertised[ref] = oid(value)
    git = Git(mirror)
    if git.refs() != advertised:
        raise PreservationError("mirror-advertised-ref-mismatch")
    for row in result["refs"]:
        reference_value = advertised.get(row["ref"])
        if reference_value is None:
            raise PreservationError("metadata-ref-missing")
        # REST tags return peeled commits; Git refs retain signed tag objects.
        peeled = git.run("rev-parse", "--verify", f"{reference_value}^{{commit}}").decode().strip()
        if peeled != row["oid"]:
            raise PreservationError("metadata-ref-mismatch")
    return {"schema_version": 1, "github": result, "advertised_refs": advertised}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--repository",
        choices=("srgangaram-swe/AlphaForge", "srgangaram-swe/Signalattice"),
        required=True,
    )
    parser.add_argument("--mirror", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        result = freeze(args.repository, args.mirror)
        with args.output.open("xb") as stream:
            stream.write(canonical(result))
    except (PreservationError, OSError, ValueError, KeyError, TypeError):
        parser.exit(2, "metadata freeze failed; no source or target mutation performed\n")


if __name__ == "__main__":
    main()
