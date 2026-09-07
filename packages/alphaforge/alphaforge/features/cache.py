"""Content-addressed, validated cache for registered feature matrices."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from alphaforge.features.registry import (
    FeatureContractError,
    FeatureRegistry,
    canonical_json,
    semantic_hash,
    validate_feature_frame,
)

CACHE_SCHEMA_VERSION = "1.0.0"
_MANIFEST_FIELDS = {
    "cache_schema_version",
    "cache_key",
    "lineage",
    "registry",
    "artifact",
    "artifact_sha256",
}


class FeatureCacheError(ValueError):
    """Raised when a feature cache entry is unsafe, corrupt, or incompatible."""


@dataclass(frozen=True)
class FeatureLineage:
    """Inputs that uniquely determine a materialized feature set."""

    dataset_reference: str
    dataset_content_id: str
    code_version: str
    registry_id: str
    parameters_id: str
    date_start: str
    date_end: str
    universe: tuple[str, ...]

    def __post_init__(self) -> None:
        for field_name in (
            "dataset_reference",
            "dataset_content_id",
            "code_version",
            "registry_id",
            "parameters_id",
            "date_start",
            "date_end",
        ):
            value = getattr(self, field_name)
            if not value or value != value.strip():
                raise FeatureContractError(f"{field_name} must be non-empty and trimmed")
        for field_name in ("dataset_content_id", "registry_id", "parameters_id"):
            value = getattr(self, field_name)
            if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
                raise FeatureContractError(f"{field_name} must be a lowercase SHA-256 digest")
        if len(self.code_version) not in {40, 64} or any(
            character not in "0123456789abcdef" for character in self.code_version
        ):
            raise FeatureContractError("code_version must be a full Git SHA or SHA-256 digest")
        try:
            start = pd.Timestamp(self.date_start)
            end = pd.Timestamp(self.date_end)
        except ValueError as exc:
            raise FeatureContractError("feature lineage dates must be valid timestamps") from exc
        if start > end:
            raise FeatureContractError("feature lineage date_start must not follow date_end")
        if not self.universe or self.universe != tuple(sorted(set(self.universe))):
            raise FeatureContractError("universe must be non-empty, sorted, and unique")
        if any(
            not symbol or symbol != symbol.strip() or not symbol.isascii()
            for symbol in self.universe
        ):
            raise FeatureContractError("universe symbols must be non-empty, trimmed ASCII")

    @property
    def cache_key(self) -> str:
        return semantic_hash(asdict(self))


@dataclass(frozen=True)
class FeatureSet:
    """A feature matrix paired with exact semantic lineage."""

    frame: pd.DataFrame
    registry: FeatureRegistry
    lineage: FeatureLineage
    cache_key: str
    cache_hit: bool


def fingerprint_frame(frame: pd.DataFrame) -> str:
    """Hash a frame deterministically, including schema, values, and row order."""

    if not isinstance(frame, pd.DataFrame) or frame.empty:
        raise FeatureContractError("cannot fingerprint an empty or non-DataFrame input")
    normalized = frame.copy()
    date_columns = [
        column
        for column in normalized.columns
        if pd.api.types.is_datetime64_any_dtype(normalized[column])
    ]
    for column in date_columns:
        normalized[column] = normalized[column].dt.strftime("%Y-%m-%dT%H:%M:%S.%f")
    schema = [(str(column), str(dtype)) for column, dtype in normalized.dtypes.items()]
    row_hashes = pd.util.hash_pandas_object(normalized, index=False, categorize=True).to_numpy()
    digest = hashlib.sha256()
    digest.update(canonical_json(schema).encode("utf-8"))
    digest.update(row_hashes.tobytes())
    return digest.hexdigest()


def build_feature_lineage(
    panel: pd.DataFrame,
    registry: FeatureRegistry,
    config: dict[str, Any],
    *,
    dataset_id: str,
    code_version: str,
) -> FeatureLineage:
    """Build the complete key material for one feature computation."""

    required = {"date", "symbol"}
    if not required.issubset(panel):
        raise FeatureContractError("panel must contain date and symbol for feature lineage")
    dates = pd.to_datetime(panel["date"], errors="raise")
    return FeatureLineage(
        dataset_reference=dataset_id,
        dataset_content_id=fingerprint_frame(panel),
        code_version=code_version,
        registry_id=registry.registry_id,
        parameters_id=semantic_hash(
            {
                key: value
                for key, value in config.items()
                if key not in {"cache_dir", "fitted_transform", "output_dir"}
            }
        ),
        date_start=dates.min().isoformat(),
        date_end=dates.max().isoformat(),
        universe=tuple(sorted(panel["symbol"].astype(str).unique())),
    )


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class FeatureCache:
    """Local immutable cache whose directory names are validated SHA-256 keys."""

    def __init__(self, root: str | Path, *, max_artifact_bytes: int = 512 * 1024 * 1024) -> None:
        self.root = Path(root)
        if self.root.is_symlink():
            raise FeatureCacheError("feature cache root must not be a symbolic link")
        if max_artifact_bytes <= 0:
            raise FeatureCacheError("max_artifact_bytes must be positive")
        self.max_artifact_bytes = max_artifact_bytes

    @staticmethod
    def _validate_key(key: str) -> None:
        if len(key) != 64 or any(character not in "0123456789abcdef" for character in key):
            raise FeatureCacheError("cache key must be a lowercase SHA-256 digest")

    def _entry(self, key: str) -> Path:
        self._validate_key(key)
        return self.root / key

    def load(self, lineage: FeatureLineage, registry: FeatureRegistry) -> pd.DataFrame | None:
        """Return a verified cache hit, ``None`` for a miss, or fail on corruption."""

        key = lineage.cache_key
        entry = self._entry(key)
        if not entry.exists():
            return None
        if entry.is_symlink() or not entry.is_dir():
            raise FeatureCacheError("cache entry must be a real directory")
        actual_names = {path.name for path in entry.iterdir()}
        if actual_names != {"manifest.json", "features.table.json"}:
            raise FeatureCacheError("cache entry contains missing or undeclared artifacts")
        manifest_path = entry / "manifest.json"
        artifact_path = entry / "features.table.json"
        if manifest_path.is_symlink() or artifact_path.is_symlink():
            raise FeatureCacheError("cache artifacts must not be symbolic links")
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            raise FeatureCacheError("cache manifest is unreadable") from exc
        if not isinstance(manifest, dict) or set(manifest) != _MANIFEST_FIELDS:
            raise FeatureCacheError("cache manifest fields mismatch")
        expected = json.loads(
            canonical_json(
                {
                    "cache_schema_version": CACHE_SCHEMA_VERSION,
                    "cache_key": key,
                    "lineage": asdict(lineage),
                    "registry": registry.manifest(),
                    "artifact": "features.table.json",
                    "artifact_sha256": _file_sha256(artifact_path),
                }
            )
        )
        if manifest != expected:
            raise FeatureCacheError("cache manifest, lineage, registry, or digest mismatch")
        try:
            from alphaforge.research.artifacts import read_frame_artifact

            frame = read_frame_artifact(artifact_path, max_bytes=self.max_artifact_bytes)
            validate_feature_frame(frame, registry)
        except (OSError, ValueError, TypeError) as exc:
            raise FeatureCacheError("cached feature artifact failed validation") from exc
        return frame

    def store(
        self, frame: pd.DataFrame, lineage: FeatureLineage, registry: FeatureRegistry
    ) -> Path:
        """Atomically publish an immutable feature entry after full validation."""

        validate_feature_frame(frame, registry)
        key = lineage.cache_key
        destination = self._entry(key)
        self.root.mkdir(parents=True, exist_ok=True)
        if destination.exists():
            existing = self.load(lineage, registry)
            if existing is None:  # pragma: no cover - destination existed above
                raise FeatureCacheError("cache entry disappeared during validation")
            return destination
        staging = Path(tempfile.mkdtemp(prefix=f".{key}.", dir=self.root))
        try:
            from alphaforge.research.artifacts import write_frame_artifact

            artifact = write_frame_artifact(frame, staging / "features.table.json")
            manifest = {
                "cache_schema_version": CACHE_SCHEMA_VERSION,
                "cache_key": key,
                "lineage": asdict(lineage),
                "registry": registry.manifest(),
                "artifact": artifact.name,
                "artifact_sha256": _file_sha256(artifact),
            }
            (staging / "manifest.json").write_text(
                canonical_json(manifest) + "\n", encoding="utf-8"
            )
            os.replace(staging, destination)
        except BaseException:
            shutil.rmtree(staging, ignore_errors=True)
            raise
        return destination
