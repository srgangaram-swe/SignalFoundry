"""Bounded immutable Nexus assets; requests never resolve a filesystem path.

The build manifest detects drift, not a malicious owner replacing trusted code.
No source package, private state, source map or arbitrary directory is served.
"""

from __future__ import annotations

import hashlib
import os
import re
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Literal

from fastapi import FastAPI, Request
from pydantic import Field, ValidationError, model_validator
from starlette.responses import RedirectResponse, Response

from signal_foundry.boundary import FoundryError, decode, private_directory, read_file
from signal_foundry.contracts import Contract, Digest

MAX_ASSET_BYTES = 1024 * 1024
MAX_ASSETS = 64
ASSET_NAME = re.compile(r"assets/[A-Za-z0-9_-]{1,120}\.(js|css|woff2|svg)")
MIME = {
    ".html": "text/html",
    ".js": "text/javascript",
    ".css": "text/css",
    ".svg": "image/svg+xml",
    ".woff2": "font/woff2",
    ".json": "application/json",
}


class AssetRecord(Contract):
    path: str = Field(min_length=1, max_length=144)
    bytes: int = Field(ge=1, le=MAX_ASSET_BYTES)
    sha256: Digest


class BuildSizes(Contract):
    javascript_gzip: int = Field(ge=0, le=250 * 1024)
    css_gzip: int = Field(ge=0, le=50 * 1024)
    total_bytes: int = Field(ge=1, le=MAX_ASSET_BYTES)


class BuildBudgets(Contract):
    javascript_gzip: Literal[256000]
    css_gzip: Literal[51200]
    total_bytes: Literal[1048576]


class BuildManifest(Contract):
    schema_version: Literal["1.0.0"]
    source_sha256: Digest
    lock_sha256: Digest
    contract_sha256: Digest
    assets: tuple[AssetRecord, ...] = Field(min_length=3, max_length=MAX_ASSETS)
    measured: BuildSizes
    budgets: BuildBudgets

    @model_validator(mode="after")
    def inventory(self) -> BuildManifest:
        names = {item.path for item in self.assets}
        if len(names) != len(self.assets) or "index.html" not in names:
            raise ValueError("duplicate or missing document in inventory")
        if any(
            name != "index.html" and not ASSET_NAME.fullmatch(name) for name in names
        ):
            raise ValueError("asset path outside allowlist")
        if sum(item.bytes for item in self.assets) != self.measured.total_bytes:
            raise ValueError("inventory byte count disagrees")
        return self


@dataclass(frozen=True)
class Asset:
    body: bytes
    media_type: str


@dataclass(frozen=True)
class NexusBundle:
    """Immutable in-memory snapshot; O(1) lookup and no request-time file IO."""

    assets: Mapping[str, Asset]


def _names(directory: Path, maximum: int) -> set[str]:
    names: set[str] = set()
    try:
        with os.scandir(directory) as entries:
            for entry in entries:
                if entry.is_symlink() or len(names) >= maximum:
                    raise FoundryError("unsafe_bundle", "Unexpected Nexus inventory.")
                names.add(entry.name)
    except OSError as exc:
        raise FoundryError(
            "bundle_unavailable", "Cannot inspect the Nexus build inventory.", 503
        ) from exc
    return names


def load_bundle(directory: Path, *, contract: Path) -> NexusBundle:
    """Verify the fixed build and current schema before acquiring service state.

    At most 64 assets and 1 MiB of asset bytes are admitted. Every byte is checked
    against the bounded manifest. Symlinks, extra files, paths and stale schema
    builds fail closed; startup never downloads, rebuilds or repairs anything.
    """
    root = private_directory(directory)
    private_directory(contract.parent)
    raw = read_file(root / "manifest.json", 16_384)
    decode(raw)
    try:
        manifest = BuildManifest.model_validate_json(raw)
    except ValidationError as exc:
        raise FoundryError(
            "invalid_bundle", "Nexus manifest violates the build contract."
        ) from exc
    if hashlib.sha256(read_file(contract)).hexdigest() != manifest.contract_sha256:
        raise FoundryError(
            "stale_bundle", "Rebuild Nexus against the current API contract.", 409
        )
    asset_root = private_directory(root / "assets")
    expected = {
        Path(item.path).name for item in manifest.assets if item.path != "index.html"
    }
    if (
        _names(root, 3) != {"index.html", "manifest.json", "assets"}
        or _names(asset_root, MAX_ASSETS) != expected
    ):
        raise FoundryError("unsafe_bundle", "Nexus files disagree with the manifest.")
    assets = {"manifest.json": Asset(raw, MIME[".json"])}
    for item in manifest.assets:
        body = read_file(root / item.path, MAX_ASSET_BYTES)
        if len(body) != item.bytes or hashlib.sha256(body).hexdigest() != item.sha256:
            raise FoundryError("corrupt_bundle", "Nexus asset failed integrity checks.")
        assets[item.path] = Asset(body, MIME[Path(item.path).suffix])
    return NexusBundle(MappingProxyType(assets))


def mount(app: FastAPI, bundle: NexusBundle) -> None:
    """Add only the document, fixed manifest and enumerated GET/HEAD assets."""

    @app.get("/", include_in_schema=False)
    def entry() -> RedirectResponse:
        return RedirectResponse("/nexus", status_code=307)

    @app.api_route("/nexus", methods=["GET", "HEAD"], include_in_schema=False)
    @app.api_route(
        "/nexus/{name:path}", methods=["GET", "HEAD"], include_in_schema=False
    )
    def asset(request: Request, name: str = "") -> Response:
        record = bundle.assets.get(name or "index.html")
        if record is None:
            raise FoundryError(
                "asset_unavailable", "No approved Nexus asset matches.", 404
            )
        return Response(
            content=b"" if request.method == "HEAD" else record.body,
            media_type=record.media_type,
            headers={"content-length": str(len(record.body))},
        )
