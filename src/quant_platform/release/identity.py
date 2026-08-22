"""Canonical release identity and version consistency.

SF-S5-SL-MR7. A release has exactly one version, and every place that states a
version must agree with it. The failure this prevents is mundane and common: a
wheel says 0.3.0, the console bundle says 0.1.0, and the release notes say
something else, so nobody can later establish which bytes were released.

**One canonical source.** ``pyproject.toml`` ``[project].version`` is the
release version. ``quant_platform.__version__`` is derived from installed
package metadata rather than re-typed, so the two cannot drift; when the package
is not installed (a source checkout) it falls back to reading the manifest, and
a test asserts both paths agree.

**The API contract version is deliberately separate.** ``info.version`` in the
OpenAPI document names the *wire contract*, which is versioned independently of
the software: a 0.x release can serve a stable v1 contract, and forcing them to
match would either freeze the contract or lie about the software's maturity. The
descriptor records both and the console pins the contract it was built against.
"""

from __future__ import annotations

import re
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Final

#: PEP 440 subset this project uses: major.minor.patch with an optional
#: pre-release. Anything else is refused rather than normalised, because a
#: version the tooling had to reinterpret is a version nobody can reproduce.
_VERSION_PATTERN: Final = re.compile(r"^(?P<release>\d+\.\d+\.\d+)(?:(?P<pre>a|b|rc)(?P<n>\d+))?$")

#: The distribution name this repository publishes.
DISTRIBUTION_NAME: Final = "signalattice"


class ReleaseIdentityError(ValueError):
    """Raised when a release version cannot be resolved or does not agree."""


@dataclass(frozen=True, slots=True)
class VersionSource:
    """One place that states a version, and what it said."""

    name: str
    path: str
    version: str


def parse_version(value: object) -> str:
    """Return a validated version string.

    Raises:
        ReleaseIdentityError: On anything outside the supported PEP 440 subset.
    """
    if not isinstance(value, str) or _VERSION_PATTERN.match(value) is None:
        raise ReleaseIdentityError(
            f"version {value!r} is not major.minor.patch with an optional a/b/rc suffix; "
            "an unrecognised version cannot be reproduced or compared"
        )
    return value


def is_prerelease(version: str) -> bool:
    """Return whether the version carries a pre-release suffix."""
    match = _VERSION_PATTERN.match(parse_version(version))
    return match is not None and match.group("pre") is not None


def claims_stable_api(version: str) -> bool:
    """Return whether the version claims a stable public interface.

    Used by the publication policy: reaching 1.0 is a compatibility promise, not
    a milestone that arrives because release machinery exists.
    """
    return int(parse_version(version).split(".")[0]) >= 1


def canonical_version(repository_root: Path) -> str:
    """Return the single canonical release version from ``pyproject.toml``.

    Raises:
        ReleaseIdentityError: If the manifest is missing or states no version.
    """
    manifest = repository_root / "pyproject.toml"
    try:
        document = tomllib.loads(manifest.read_text(encoding="utf-8"))
    except OSError as error:
        raise ReleaseIdentityError(f"cannot read {manifest}") from error
    except tomllib.TOMLDecodeError as error:
        raise ReleaseIdentityError(f"{manifest} is not valid TOML") from error
    project = document.get("project")
    if not isinstance(project, dict) or "version" not in project:
        raise ReleaseIdentityError("pyproject.toml declares no [project].version")
    return parse_version(project["version"])


def _console_version(repository_root: Path) -> str | None:
    """Return the console's declared version, or ``None`` when absent."""
    manifest = repository_root / "web" / "package.json"
    if not manifest.is_file():
        return None
    import json

    try:
        document = json.loads(manifest.read_text(encoding="utf-8"))
    except ValueError as error:
        raise ReleaseIdentityError("web/package.json is not valid JSON") from error
    version = document.get("version")
    if not isinstance(version, str):
        raise ReleaseIdentityError("web/package.json declares no string version")
    return version


def _installed_version() -> str | None:
    """Return the installed distribution version, or ``None`` in a bare checkout."""
    from importlib.metadata import PackageNotFoundError, version

    try:
        return version(DISTRIBUTION_NAME)
    except PackageNotFoundError:
        return None


def collect_version_sources(repository_root: Path) -> tuple[VersionSource, ...]:
    """Return every place that states the release version.

    The API contract version is deliberately excluded: it names the wire
    contract, not the software, and is checked separately by
    :func:`assert_contract_pinned`.
    """
    sources = [
        VersionSource("pyproject", "pyproject.toml", canonical_version(repository_root)),
    ]
    console = _console_version(repository_root)
    if console is not None:
        sources.append(VersionSource("console", "web/package.json", console))
    installed = _installed_version()
    if installed is not None:
        sources.append(VersionSource("installed", f"{DISTRIBUTION_NAME} metadata", installed))
    return tuple(sources)


def assert_versions_agree(repository_root: Path) -> str:
    """Return the canonical version, refusing if any source disagrees.

    Raises:
        ReleaseIdentityError: Naming every disagreeing source, so one run
            reports all of them rather than one per fix-and-retry cycle.
    """
    sources = collect_version_sources(repository_root)
    expected = sources[0].version
    mismatched = [item for item in sources if item.version != expected]
    if mismatched:
        detail = ", ".join(f"{item.name} ({item.path}) = {item.version}" for item in mismatched)
        raise ReleaseIdentityError(
            f"release version disagreement: canonical is {expected}, but {detail}"
        )
    return expected


def assert_contract_pinned(repository_root: Path, *, expected_contract: str) -> str:
    """Return the served API contract version, refusing a mismatch.

    Raises:
        ReleaseIdentityError: If the committed contract does not state the
            expected version.
    """
    import json

    document_path = repository_root / "docs" / "api" / "openapi-v1.json"
    try:
        document = json.loads(document_path.read_text(encoding="utf-8"))
    except OSError as error:
        raise ReleaseIdentityError(f"cannot read {document_path}") from error
    except ValueError as error:
        raise ReleaseIdentityError(f"{document_path} is not valid JSON") from error
    served = document.get("info", {}).get("version")
    if served != expected_contract:
        raise ReleaseIdentityError(
            f"API contract version is {served!r} but the release expects {expected_contract!r}; "
            "the contract version names the wire format and is pinned by the console"
        )
    return str(served)


__all__ = [
    "DISTRIBUTION_NAME",
    "ReleaseIdentityError",
    "VersionSource",
    "assert_contract_pinned",
    "assert_versions_agree",
    "canonical_version",
    "claims_stable_api",
    "collect_version_sources",
    "is_prerelease",
    "parse_version",
]
