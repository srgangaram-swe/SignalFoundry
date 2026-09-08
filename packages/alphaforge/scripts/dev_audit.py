"""Validate uv's hashed development export without installing or importing packages.

This is a verifier of an already solved lock, not a dependency resolver. An
iterative worklist visits each (package, requested extra) once, including cycles.
Errors carry stable codes and never include untrusted requirement text.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import stat
import sys
import tomllib
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from packaging.markers import InvalidMarker, Marker, default_environment
from packaging.requirements import InvalidRequirement, Requirement
from packaging.utils import canonicalize_name
from packaging.version import InvalidVersion, Version

MAX_BYTES = 4 * 1024 * 1024
MAX_PACKAGES = 2048
MAX_EDGES = 32768
HASH = re.compile(r"sha256:[0-9a-f]{64}\Z")
EXPORT_ARGUMENTS = (
    "export",
    "--locked",
    "--no-dev",
    "--extra",
    "dev",
    "--no-emit-project",
    "--no-header",
    "--no-annotate",
    "--format",
    "requirements-txt",
)
Key = tuple[str, Version]


class AuditContractError(ValueError):
    """A bounded, non-sensitive failure code at the audit trust boundary."""


def read_bounded(path: Path) -> bytes:
    """Read at most four MiB plus one sentinel byte; reject symlinks and nonfiles."""
    try:
        descriptor = os.open(path, os.O_RDONLY | os.O_NONBLOCK | os.O_NOFOLLOW)
        with os.fdopen(descriptor, "rb") as stream:
            if not stat.S_ISREG(os.fstat(stream.fileno()).st_mode):
                raise AuditContractError("unsafe-input-file")
            content = stream.read(MAX_BYTES + 1)
    except OSError as exc:
        raise AuditContractError("input-read-failed") from exc
    if not content or len(content) > MAX_BYTES:
        raise AuditContractError("input-size-limit")
    return content


def _mapping(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict) or not all(isinstance(key, str) for key in value):
        raise AuditContractError("invalid-lock-mapping")
    return value


def _sequence(value: Any) -> list[Any]:
    if not isinstance(value, list) or len(value) > MAX_EDGES:
        raise AuditContractError("invalid-lock-sequence")
    return value


def _name(value: Any) -> str:
    if not isinstance(value, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", value):
        raise AuditContractError("invalid-package-name")
    return canonicalize_name(value)


def _version(value: Any) -> Version:
    if not isinstance(value, str) or len(value) > 128:
        raise AuditContractError("invalid-package-version")
    try:
        return Version(value)
    except InvalidVersion as exc:
        raise AuditContractError("invalid-package-version") from exc


def _active(marker: Any, environment: dict[str, str]) -> bool:
    if marker is None:
        return True
    if not isinstance(marker, str) or len(marker) > 4096:
        raise AuditContractError("invalid-environment-marker")
    try:
        return Marker(marker).evaluate(environment=environment)
    except (InvalidMarker, KeyError, ValueError, RecursionError) as exc:
        raise AuditContractError("invalid-environment-marker") from exc


class LockGraph:
    """Indexed uv v1/revision-3 graph; only the trusted public registry is allowed.

    Construction is O(lock bytes). Closure is O(V + E) for this lock's unique
    package versions; explicitly versioned forks are indexed by package name.
    Requested extras have separate visit keys so diamond dependencies cannot
    suppress an extra reached later. Bounds apply before graph traversal.
    """

    def __init__(self, content: bytes) -> None:
        if not content or len(content) > MAX_BYTES:
            raise AuditContractError("input-size-limit")
        try:
            lock = tomllib.loads(content.decode("utf-8"))
        except (UnicodeError, tomllib.TOMLDecodeError, RecursionError) as exc:
            raise AuditContractError("invalid-lock-document") from exc
        if lock.get("version") != 1 or lock.get("revision") != 3:
            raise AuditContractError("unsupported-lock-schema")
        packages = _sequence(lock.get("package"))
        if not packages or len(packages) > MAX_PACKAGES:
            raise AuditContractError("package-count-limit")
        self.packages: dict[Key, dict[str, Any]] = {}
        self.names: dict[str, list[Key]] = defaultdict(list)
        self.hashes: dict[Key, frozenset[str]] = {}
        edges = 0
        for raw in packages:
            package = _mapping(raw)
            key = (_name(package.get("name")), _version(package.get("version")))
            if key in self.packages:
                raise AuditContractError("duplicate-lock-package")
            source = _mapping(package.get("source"))
            if source != (
                {"editable": "."}
                if key[0] == "alphaforge"
                else {"registry": "https://pypi.org/simple"}
            ):
                raise AuditContractError("untrusted-package-source")
            groups = _mapping(package.get("optional-dependencies", {}))
            dependencies = _sequence(package.get("dependencies", []))
            edges += len(dependencies) + sum(len(_sequence(group)) for group in groups.values())
            if edges > MAX_EDGES:
                raise AuditContractError("edge-count-limit")
            artifacts = [*(_sequence(package.get("wheels", [])))]
            if "sdist" in package:
                artifacts.append(package["sdist"])
            hashes: set[str] = set()
            for artifact in artifacts:
                digest = _mapping(artifact).get("hash")
                if not isinstance(digest, str) or not HASH.fullmatch(digest):
                    raise AuditContractError("invalid-lock-hash")
                hashes.add(digest)
            if key[0] != "alphaforge" and not hashes:
                raise AuditContractError("missing-lock-hashes")
            self.packages[key] = package
            self.names[key[0]].append(key)
            self.hashes[key] = frozenset(hashes)
        if len(self.names.get("alphaforge", [])) != 1:
            raise AuditContractError("missing-project-root")

    def closure(self, extras: tuple[str, ...], environment: dict[str, str]) -> set[Key]:
        """Return the complete active third-party closure for runtime plus extras."""
        root = self.names["alphaforge"][0]
        queue = [(root, ""), *((root, extra) for extra in extras)]
        visited: set[tuple[Key, str]] = set()
        while queue:
            key, extra = queue.pop()
            if (key, extra) in visited:
                continue
            visited.add((key, extra))
            package = self.packages[key]
            if extra:
                groups = _mapping(package.get("optional-dependencies", {}))
                if extra not in groups:
                    raise AuditContractError("unknown-requested-extra")
                edges = _sequence(groups[extra])
            else:
                edges = _sequence(package.get("dependencies", []))
            for raw in edges:
                edge = _mapping(raw)
                if set(edge) - {"name", "version", "source", "marker", "extra"}:
                    raise AuditContractError("unsupported-dependency-edge")
                if not _active(edge.get("marker"), environment):
                    continue
                candidates = self.names.get(_name(edge.get("name")), [])
                if "version" in edge:
                    candidates = [
                        item for item in candidates if item[1] == _version(edge["version"])
                    ]
                candidates = [
                    item
                    for item in candidates
                    if ("source" not in edge or edge["source"] == self.packages[item]["source"])
                    and (
                        not self.packages[item].get("resolution-markers")
                        or any(
                            _active(marker, environment)
                            for marker in _sequence(self.packages[item]["resolution-markers"])
                        )
                    )
                ]
                if len(candidates) != 1:
                    raise AuditContractError("missing-or-ambiguous-dependency")
                target = candidates[0]
                queue.append((target, ""))
                queue.extend((target, _name(name)) for name in _sequence(edge.get("extra", [])))
        return {key for key, _ in visited if key != root}


@dataclass(frozen=True)
class ExportPin:
    """An exact non-executable package pin and its lock-bound artifact hashes."""

    key: Key
    marker: str | None
    hashes: frozenset[str]


def parse_export(content: bytes) -> tuple[ExportPin, ...]:
    """Accept only uv's unannotated hashed pins; reject options, URLs and includes."""
    if not content or len(content) > MAX_BYTES:
        raise AuditContractError("input-size-limit")
    try:
        text = content.decode("ascii")
    except UnicodeError as exc:
        raise AuditContractError("invalid-export-encoding") from exc
    if any(ord(char) < 32 and char not in "\n\r\t" for char in text):
        raise AuditContractError("invalid-export-control-character")
    text = text.replace("\r\n", "\n").replace("\\\n", " ")
    pins = []
    for line in text.splitlines():
        if not line.strip():
            continue
        parts = line.split(" --hash=")
        try:
            requirement = Requirement(parts[0].strip())
        except (InvalidRequirement, RecursionError) as exc:
            raise AuditContractError("invalid-export-requirement") from exc
        specifiers = list(requirement.specifier)
        if (
            requirement.url
            or requirement.extras
            or len(specifiers) != 1
            or specifiers[0].operator != "=="
        ):
            raise AuditContractError("export-must-use-exact-pins")
        version = _version(specifiers[0].version)
        hashes = [part.strip() for part in parts[1:]]
        if (
            not hashes
            or any(not HASH.fullmatch(digest) for digest in hashes)
            or len(set(hashes)) != len(hashes)
        ):
            raise AuditContractError("invalid-export-hashes")
        pins.append(
            ExportPin(
                (_name(requirement.name), version),
                str(requirement.marker) if requirement.marker else None,
                frozenset(hashes),
            )
        )
        if len(pins) > MAX_PACKAGES:
            raise AuditContractError("package-count-limit")
    if not pins:
        raise AuditContractError("empty-export")
    return tuple(pins)


def validate_export(
    lock: bytes, exported: bytes, environment: dict[str, str] | None = None
) -> dict[str, Any]:
    """Verify all hashes and the exact host runtime+dev closure, including pip's floor.

    Inactive marker records still require valid pins and exact lock hashes. They
    are not claimed audited on this host. CI additionally tests the supported
    Python/platform selection matrix without performing network requests.
    """
    env = (
        {key: str(value) for key, value in default_environment().items()}
        if environment is None
        else dict(environment)
    )
    if set(env) != set(default_environment()) or not all(
        isinstance(value, str) for value in env.values()
    ):
        raise AuditContractError("incomplete-marker-environment")
    graph = LockGraph(lock)
    expected = graph.closure(("dev",), env)
    active: set[Key] = set()
    active_names: set[str] = set()
    seen: set[tuple[Key, str | None]] = set()
    pins = parse_export(exported)
    for pin in pins:
        if pin.hashes != graph.hashes.get(pin.key):
            raise AuditContractError("export-lock-hash-mismatch")
        if (pin.key, pin.marker) in seen:
            raise AuditContractError("duplicate-export-record")
        seen.add((pin.key, pin.marker))
        if _active(pin.marker, env):
            if pin.key[0] in active_names:
                raise AuditContractError("duplicate-active-package")
            active.add(pin.key)
            active_names.add(pin.key[0])
    if active != expected:
        raise AuditContractError("export-dependency-closure-mismatch")
    versions = {name: version for name, version in active}
    if "pip" not in versions or versions["pip"] < Version("26.2.0") or "pip-audit" not in versions:
        raise AuditContractError("unpatched-or-missing-audit-toolchain")
    runtime = graph.closure((), env)
    return {
        "schema_version": 1,
        "lock_sha256": hashlib.sha256(lock).hexdigest(),
        "export_sha256": hashlib.sha256(exported).hexdigest(),
        "environment": env,
        "export_records": len(pins),
        "runtime_packages": len(runtime),
        "additional_dev_packages": len(active - runtime),
        "active_packages": [
            {"name": name, "version": str(version)} for name, version in sorted(active)
        ],
        "scope": "host runtime plus dev extra; advisory audit is a separate required step",
    }


def main() -> int:
    """Exit 2 on invalid input, with no offending payload or traceback in stderr."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("requirements", type=Path)
    parser.add_argument("--lock", type=Path, default=Path("uv.lock"))
    args = parser.parse_args()
    try:
        result = validate_export(read_bounded(args.lock), read_bounded(args.requirements))
    except AuditContractError as exc:
        print(f"development audit contract: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
