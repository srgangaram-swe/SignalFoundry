"""Typed, local-only service boundary for immutable Signalattice evidence.

Framework-neutral contracts live here so HTTP and console adapters can share one
strict projection without depending on FastAPI, SQLite, or filesystem details.
"""

from quant_platform.service.manifests import (
    DiagnosticsManifest,
    ForecastSummaryManifest,
    ManifestKind,
    ModelCardManifest,
    parse_evidence_manifest,
)

__all__ = [
    "DiagnosticsManifest",
    "ForecastSummaryManifest",
    "ManifestKind",
    "ModelCardManifest",
    "parse_evidence_manifest",
]
