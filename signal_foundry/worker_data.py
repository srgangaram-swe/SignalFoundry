"""Signalattice adapter: validate producer bundles, export only bounded metadata."""

from __future__ import annotations

import re
from pathlib import Path

from signal_foundry.boundary import FoundryError, decode, private_directory, read_file
from signal_foundry.contracts import Dataset

DIGEST = re.compile(r"^[0-9a-f]{64}$")


def bundle_path(directory: Path, identity: str) -> Path:
    """Resolve only an immutable bundle name below an operator-selected directory."""
    if not DIGEST.fullmatch(identity):
        raise FoundryError(
            "invalid_dataset_id", "A dataset must have a content identity."
        )
    parent = private_directory(directory)
    bundle = private_directory(parent / identity)
    manifest = decode(read_file(bundle / "manifest.json", 1 << 20), 1 << 20)
    if not isinstance(manifest, dict) or type(manifest.get("rows")) is not int:
        raise FoundryError("invalid_dataset", "Dataset metadata is malformed.")
    if not 1 <= manifest["rows"] <= 40_000:
        raise FoundryError(
            "dataset_limit", "Dataset exceeds the interactive row budget."
        )
    if (
        not isinstance(manifest.get("tickers"), list)
        or not 2 <= len(manifest["tickers"]) <= 32
    ):
        raise FoundryError("dataset_limit", "Dataset exceeds the instrument budget.")
    total = 0
    count = 0
    for path in bundle.rglob("*"):
        count += 1
        if count > 128 or path.is_symlink():
            raise FoundryError(
                "unsafe_dataset", "Dataset layout exceeds the safe file policy."
            )
        if path.is_file():
            size = path.stat().st_size
            total += size
            if size > 16 << 20 or total > 128 << 20:
                raise FoundryError(
                    "dataset_limit", "Dataset exceeds the disk-input budget."
                )
        elif not path.is_dir():
            raise FoundryError(
                "unsafe_dataset",
                "Dataset entries must be regular files or directories.",
            )
    return bundle


def inspect_bundle(directory: Path, identity: str) -> Dataset:
    """Exercise the actual producer validator, never infer validation from a flag."""
    from quant_platform.data.signal_foundry_contract import (
        SignalFoundryContractError,
        validate_signal_foundry_bundle,
    )

    path = bundle_path(directory, identity)
    try:
        manifest = validate_signal_foundry_bundle(path)
    except (
        SignalFoundryContractError,
        OSError,
        ValueError,
        TypeError,
        KeyError,
    ) as exc:
        raise FoundryError(
            "invalid_dataset", "Signalattice rejected the selected bundle."
        ) from exc
    if manifest["bundle_id"] != identity:
        raise FoundryError(
            "dataset_identity", "Bundle directory and content identity disagree."
        )
    limits = manifest["point_in_time_limits"]
    limitations = ["Historical research data; no current-market freshness claim."]
    for key, explanation in (
        (
            "historical_revisions_complete",
            "Historical revisions are not proven complete.",
        ),
        (
            "universe_membership_point_in_time",
            (
                "Point-in-time universe membership is not proven; survivorship bias may"
                " remain."
            ),
        ),
        ("corporate_actions_complete", "Corporate-action completeness is not proven."),
    ):
        if limits.get(key) is not True:
            limitations.append(explanation)
    return Dataset(
        bundle_id=identity,
        rows=manifest["rows"],
        symbols=tuple(manifest["tickers"]),
        date_min=manifest["date_min"],
        date_max=manifest["date_max"],
        limitations=tuple(limitations),
    )


def discover(directory: Path | None) -> tuple[Dataset, ...]:
    if directory is None:
        return ()
    parent = private_directory(directory)
    identities: list[str] = []
    inspected = 0
    for child in parent.iterdir():
        inspected += 1
        if inspected > 64:
            raise FoundryError(
                "catalog_limit", "Bundle directory contains too many entries."
            )
        if DIGEST.fullmatch(child.name):
            identities.append(child.name)
            if len(identities) > 16:
                raise FoundryError(
                    "catalog_limit",
                    "At most sixteen approved bundles may be cataloged.",
                )
    return tuple(inspect_bundle(parent, identity) for identity in sorted(identities))
