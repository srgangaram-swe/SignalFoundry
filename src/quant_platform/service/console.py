"""Bounded static delivery for the local forecast-observability console.

SF-S5-SL-MR6. This is an additive read-only boundary: it serves a pre-built
front-end bundle from one directory and nothing else. It grants no authority the
JSON API does not already have, and removing it removes the console without
touching the API.

The boundary is deliberately not ``StaticFiles``. A generic static handler
resolves whatever path it is given, guesses content types from the filesystem,
and follows symlinks; each of those is a way for a file outside the bundle to
reach a browser. Instead:

**The served set is enumerated at construction.** The directory is walked once,
every eligible file is recorded with its resolved path, size, and digest, and
requests are answered from that manifest. A path that was not present at
construction cannot be served, so a file appearing under the root later --
including through a symlink swapped in after the fact -- is not reachable.

**Content types come from a fixed map, never from the filesystem.** An extension
outside the map is not served at all, so a bundle cannot be made to emit
``text/html`` for something the build did not intend as a document.

**Route fallback is an allowlist, not a wildcard.** A single-page application
normally answers every unknown path with ``index.html``. That turns the console
into an oracle that returns 200 for anything. Only the seven declared console
routes fall back; everything else is a 404.

**The console content-security policy is separate from the API's.** API
responses keep ``default-src 'none'``. The document needs its own scripts and
styles, so it gets exactly ``'self'`` for those and nothing else: no inline, no
eval, no third-party origin, no websocket, no worker, and no remote image.
"""

from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Final

#: The seven console routes from the issue, and the only paths that fall back to
#: the application document. Adding an eighth route here is a visible change.
CONSOLE_ROUTES: Final[tuple[str, ...]] = (
    "/console",
    "/console/",
    "/console/runs",
    "/console/runs/",
    "/console/evidence",
    "/console/evidence/",
    "/console/comparison",
    "/console/comparison/",
    "/console/calibration",
    "/console/calibration/",
    "/console/operations",
    "/console/operations/",
    "/console/governance",
    "/console/governance/",
)

#: Extensions the build is permitted to emit, mapped to the exact content type
#: served. Anything else is not part of a console bundle and is not served.
_CONTENT_TYPES: Final[dict[str, str]] = {
    ".css": "text/css; charset=utf-8",
    ".html": "text/html; charset=utf-8",
    ".js": "text/javascript; charset=utf-8",
    ".json": "application/json",
    ".svg": "image/svg+xml",
    ".webmanifest": "application/manifest+json",
    ".woff2": "font/woff2",
}

#: Extensions whose file name contains a build hash and may be cached
#: indefinitely. The document itself never is: it names the current bundle, and
#: a stale document would load assets that no longer exist.
_IMMUTABLE_SUFFIXES: Final[frozenset[str]] = frozenset(
    {".css", ".js", ".svg", ".woff2", ".webmanifest"}
)

#: Refusal thresholds, not tuning knobs.
MAX_ASSET_BYTES: Final = 4 * 1024 * 1024
MAX_ASSETS: Final = 512
MAX_TOTAL_BYTES: Final = 16 * 1024 * 1024

#: The console document policy. Distinct from the API policy because a document
#: legitimately needs its own scripts and styles, and nothing else.
#:
#: ``style-src 'self'`` and not ``'unsafe-inline'``: the build emits a stylesheet
#: file, so inline style is never required. ``connect-src 'self'`` confines the
#: browser to the same origin that served it, which is the loopback service.
CONSOLE_CSP: Final = (
    "default-src 'none'; "
    "script-src 'self'; "
    "style-src 'self'; "
    "img-src 'self' data:; "
    "font-src 'self'; "
    "connect-src 'self'; "
    "base-uri 'none'; "
    "form-action 'none'; "
    "frame-ancestors 'none'; "
    "object-src 'none'; "
    "manifest-src 'self'; "
    "worker-src 'none'; "
    "child-src 'none'; "
    "frame-src 'none'; "
    "media-src 'none'; "
    "require-trusted-types-for 'script'"
)


class ConsoleBundleError(RuntimeError):
    """Raised when a console bundle cannot be served safely.

    Fail-closed: a bundle that cannot be validated is not served in a degraded
    form, because a partially served console would render evidence that the
    reader could not tell was incomplete.
    """


@dataclass(frozen=True, slots=True)
class ConsoleAsset:
    """One immutable file admitted to the served manifest."""

    route: str
    path: Path
    content_type: str
    size: int
    digest: str
    immutable: bool

    @property
    def etag(self) -> str:
        """Return the strong validator for this asset."""
        return f'"{self.digest[:32]}"'


def _is_safe_component(name: str) -> bool:
    """Return whether one path component is admissible.

    Rejects empty names, dot components, separators, NUL, and anything
    non-printable-ASCII. A bundle emitted by the pinned build uses a narrow,
    predictable character set; anything else is a signal, not a file to serve.
    """
    if not name or name in {".", ".."}:
        return False
    if any(character in name for character in ("/", "\\", "\x00")):
        return False
    return all("!" <= character <= "~" for character in name)


@dataclass(frozen=True, slots=True)
class ConsoleBundle:
    """An enumerated, immutable manifest of servable console assets."""

    root: Path
    assets: dict[str, ConsoleAsset]
    document: ConsoleAsset
    total_bytes: int

    def resolve(self, path: str) -> ConsoleAsset | None:
        """Return the asset for a request path, or ``None`` for a 404.

        A declared console route resolves to the application document; every
        other unknown path resolves to nothing. There is no wildcard fallback.
        """
        if path in self.assets:
            return self.assets[path]
        if path in CONSOLE_ROUTES:
            return self.document
        return None


def load_console_bundle(root: str | Path) -> ConsoleBundle:
    """Enumerate a built console bundle into an immutable served manifest.

    The directory is read exactly once. Every file is resolved and re-checked to
    live under the root, so a symlink escaping the bundle is refused at load
    rather than followed at request time.

    Args:
        root: Directory containing the built bundle, which must hold
            ``index.html``.

    Returns:
        The manifest the boundary answers requests from.

    Raises:
        ConsoleBundleError: If the directory is missing, holds no document,
            exceeds a ceiling, or contains a file that is not safe to serve.
    """
    base = Path(root).expanduser()
    try:
        resolved_root = base.resolve(strict=True)
    except OSError as error:
        raise ConsoleBundleError(f"console bundle root {base} is not readable") from error
    if not resolved_root.is_dir():
        raise ConsoleBundleError(f"console bundle root {resolved_root} is not a directory")

    assets: dict[str, ConsoleAsset] = {}
    total = 0
    for current, directories, files in os.walk(resolved_root, followlinks=False):
        directories.sort()
        for name in sorted(directories):
            if not _is_safe_component(name):
                raise ConsoleBundleError(f"console bundle contains an unsafe directory: {name!r}")
        for name in sorted(files):
            if not _is_safe_component(name):
                raise ConsoleBundleError(f"console bundle contains an unsafe file name: {name!r}")
            candidate = Path(current) / name
            if candidate.is_symlink():
                raise ConsoleBundleError(
                    f"console bundle contains a symlink: {candidate.name}; a bundle is a "
                    "set of regular files, and following links would serve content from "
                    "outside the build output"
                )
            try:
                real = candidate.resolve(strict=True)
            except OSError as error:
                raise ConsoleBundleError(f"console asset {name!r} is not readable") from error
            if not real.is_file():
                raise ConsoleBundleError(f"console asset {name!r} is not a regular file")
            if resolved_root not in real.parents and real.parent != resolved_root:
                raise ConsoleBundleError(f"console asset {name!r} resolves outside the bundle root")

            suffix = real.suffix.lower()
            content_type = _CONTENT_TYPES.get(suffix)
            if content_type is None:
                raise ConsoleBundleError(
                    f"console bundle contains an unservable file type {suffix!r}; the "
                    "served content types are fixed, so an unexpected artifact is a build "
                    "problem rather than something to guess at"
                )
            size = real.stat().st_size
            if size > MAX_ASSET_BYTES:
                raise ConsoleBundleError(f"console asset {name!r} exceeds {MAX_ASSET_BYTES} bytes")
            total += size
            if total > MAX_TOTAL_BYTES:
                raise ConsoleBundleError(f"console bundle exceeds {MAX_TOTAL_BYTES} bytes")

            payload = real.read_bytes()
            route = "/console/" + str(real.relative_to(resolved_root)).replace(os.sep, "/")
            assets[route] = ConsoleAsset(
                route=route,
                path=real,
                content_type=content_type,
                size=size,
                digest=hashlib.sha256(payload).hexdigest(),
                immutable=suffix in _IMMUTABLE_SUFFIXES,
            )
            if len(assets) > MAX_ASSETS:
                raise ConsoleBundleError(f"console bundle exceeds {MAX_ASSETS} files")

    document = assets.get("/console/index.html")
    if document is None:
        raise ConsoleBundleError("console bundle has no index.html document")
    return ConsoleBundle(root=resolved_root, assets=assets, document=document, total_bytes=total)


def console_headers(asset: ConsoleAsset) -> tuple[tuple[bytes, bytes], ...]:
    """Return the exact response headers for one console asset.

    The document is never cached, because it names the hashed asset files for
    the current build; a cached document outliving its bundle would request
    files that no longer exist. Hashed assets are immutable by construction and
    say so.
    """
    cache = b"public, max-age=31536000, immutable" if asset.immutable else b"no-store"
    return (
        (b"content-type", asset.content_type.encode("ascii")),
        (b"content-length", str(asset.size).encode("ascii")),
        (b"cache-control", cache),
        (b"etag", asset.etag.encode("ascii")),
        (b"content-security-policy", CONSOLE_CSP.encode("ascii")),
        (b"cross-origin-resource-policy", b"same-origin"),
        (b"cross-origin-opener-policy", b"same-origin"),
        (b"referrer-policy", b"no-referrer"),
        (b"x-content-type-options", b"nosniff"),
        (b"x-frame-options", b"DENY"),
        (
            b"permissions-policy",
            b"accelerometer=(), camera=(), geolocation=(), microphone=(), usb=()",
        ),
    )


__all__ = [
    "CONSOLE_CSP",
    "CONSOLE_ROUTES",
    "MAX_ASSETS",
    "MAX_ASSET_BYTES",
    "MAX_TOTAL_BYTES",
    "ConsoleAsset",
    "ConsoleBundle",
    "ConsoleBundleError",
    "console_headers",
    "load_console_bundle",
]
