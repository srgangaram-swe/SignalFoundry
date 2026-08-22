"""Build and check the Sprint 5 machine-readable evidence index.

SF-S5-SL-MR8. The dossier makes claims; this index is what makes each of them
checkable. Every entry names the artifact that supports a claim, recomputes its
SHA-256, and records its evidence class.

**Evidence classes are never relabelled upward.** The four classes used here are,
in increasing strength:

``deterministic_synthetic``
    Generated from a seeded fixture. Proves mechanics, not market behaviour.
``historical_replay``
    Computed from committed historical data with no prospective element.
``local_engineering``
    Measured on one local machine. Not a capacity or SLO claim.
``prospective_wallclock``
    Accumulated in real elapsed time. **Sprint 5 has none of this**, which is
    why every readiness question defers to #63.

``build`` regenerates the index from the artifacts on disk. ``check`` recomputes
every digest and refuses when one drifts, a file is missing, or an entry claims
a class the sprint did not produce.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Final

#: Classes in increasing strength. An entry may never claim a class the sprint
#: did not actually produce, which is enforced rather than documented.
EVIDENCE_CLASSES: Final = (
    "deterministic_synthetic",
    "historical_replay",
    "local_engineering",
    "prospective_wallclock",
)

#: Sprint 5 produced no prospective wall-clock evidence. Any entry claiming it
#: is a relabelling error, so the index refuses to build.
FORBIDDEN_CLASSES: Final = frozenset({"prospective_wallclock"})

EXIT_OK: Final = 0
EXIT_REFUSED: Final = 2


class DossierError(RuntimeError):
    """Raised when the evidence index cannot be built or does not check."""


@dataclass(frozen=True, slots=True)
class EvidenceEntry:
    """One artifact and the claim it supports."""

    claim: str
    issue: str
    source_path: str
    sha256: str
    size_bytes: int
    evidence_class: str
    collection_method: str
    environment: str
    sample_context: str
    limitations: str


#: The evidence Sprint 5 actually produced, each tied to the issue that produced
#: it and the claim it supports. Paths are repository-relative.
_DECLARED: Final[tuple[dict[str, str], ...]] = (
    {
        "claim": "The bounded read-only service holds its admission, latency, and response bounds.",
        "issue": "#20",
        "source_path": "docs/benchmarks/service_operability_2026-08-09.json",
        "evidence_class": "local_engineering",
        "collection_method": "scripts/benchmark_service_operability.py against an in-process app",
        "environment": "single local machine, CPython, no network",
        "sample_context": "aggregate percentiles over synthetic request scenarios",
        "limitations": (
            "Laptop-scale measurements from one process. Not a service level objective, a "
            "capacity claim, or evidence of production readiness."
        ),
    },
    {
        "claim": (
            "The console meets its resource budgets, renders every honest state, and passes "
            "accessibility checks across four browser targets."
        ),
        "issue": "#19",
        "source_path": "docs/benchmarks/console_evidence_2026-08-20.json",
        "evidence_class": "local_engineering",
        "collection_method": "scripts/collect_console_evidence.py from real build and test runs",
        "environment": "single local machine, Node 24 toolchain, Chromium/Firefox/WebKit",
        "sample_context": "130 browser tests, 172 unit tests, 3 measured budgets",
        "limitations": (
            "Measured on one machine. Panel 4 of the figure is a static property of the view "
            "sources, not a claim about which states were exercised."
        ),
    },
    {
        "claim": (
            "The release dry run verifies end to end and refuses six deliberate tamper cases."
        ),
        "issue": "#23",
        "source_path": "docs/benchmarks/release_dry_run_2026-08-20.json",
        "evidence_class": "local_engineering",
        "collection_method": "scripts/collect_release_evidence.py against a non-publishing dry run",
        "environment": "single local machine, CPython 3.13, Node 24",
        "sample_context": "24 release subjects, 439 SBOM components, 6 tamper cases",
        "limitations": (
            "Digests establish content identity, not economic validity, independent review, or "
            "production readiness. No signing material exists yet."
        ),
    },
    {
        "claim": "Two clean builds of one commit produce byte-identical release artifacts.",
        "issue": "#23",
        "source_path": "docs/benchmarks/release_reproducibility_2026-08-20.json",
        "evidence_class": "local_engineering",
        "collection_method": "scripts/check_release_reproducible.py, two full builds compared",
        "environment": "single local machine; one platform only",
        "sample_context": "27 artifacts compared byte for byte",
        "limitations": (
            "Cross-platform byte identity is not established, and the container image is not yet "
            "a dry-run subject."
        ),
    },
    {
        "claim": (
            "The date-block bootstrap holds its nominal false-positive rate under intraday "
            "dependence where a naive per-forecast bootstrap does not."
        ),
        "issue": "#22",
        "source_path": "reports/figures/governance_promotion_evidence.json",
        "evidence_class": "deterministic_synthetic",
        "collection_method": "scripts/plot_governance_evidence.py, seeded null-calibration study",
        "environment": "single local machine, seeded numpy generator",
        "sample_context": "400 synthetic null cohorts per dependence level, seed 20260811",
        "limitations": (
            "Simulated evidence. It establishes the estimator's operating characteristics, not "
            "any model's performance, and is not market evidence."
        ),
    },
    {
        "claim": (
            "The Signal Foundry dataset contract round-trips a verified historical panel with "
            "recorded provenance."
        ),
        "issue": "#18",
        "source_path": "docs/benchmarks/signal_foundry_contract_1_1_2026-07-25.json",
        "evidence_class": "historical_replay",
        "collection_method": "scripts/benchmark_signal_foundry_contract.py over cached vintages",
        "environment": "single local machine, immutable verified cache, zero provider requests",
        "sample_context": "aggregate counts and hashes only; raw rows remain local",
        "limitations": (
            "Stale current-vintage public-domain data through 2018-03-27. Not a point-in-time "
            "universe, and vulnerable to survivorship and selection bias."
        ),
    },
    {
        "claim": "Deterministic cache replay reproduces an acquired panel without a provider request.",
        "issue": "#18",
        "source_path": "docs/benchmarks/nasdaq_cache_replay_2026-07-24.json",
        "evidence_class": "historical_replay",
        "collection_method": "scripts/benchmark_nasdaq_cache_replay.py against the immutable cache",
        "environment": "single local machine, cache-first, no network",
        "sample_context": "aggregate row counts and digests",
        "limitations": "Replay of already-acquired data. Establishes mechanics, not data quality.",
    },
)


def _digest(path: Path) -> tuple[str, int]:
    """Return the SHA-256 and size of one artifact.

    Raises:
        DossierError: If the artifact is missing or unreadable.
    """
    try:
        payload = path.read_bytes()
    except OSError as error:
        raise DossierError(f"evidence artifact is missing: {path}") from error
    return hashlib.sha256(payload).hexdigest(), len(payload)


def build_index(root: Path) -> dict[str, Any]:
    """Recompute every declared entry into a machine-readable index.

    Raises:
        DossierError: On a missing artifact or an unusable evidence class.
    """
    entries: list[EvidenceEntry] = []
    for declared in _DECLARED:
        evidence_class = declared["evidence_class"]
        if evidence_class not in EVIDENCE_CLASSES:
            raise DossierError(f"unknown evidence class {evidence_class!r}")
        if evidence_class in FORBIDDEN_CLASSES:
            raise DossierError(
                f"{declared['source_path']} claims {evidence_class!r}, which Sprint 5 did not "
                "produce; evidence is never relabelled upward"
            )
        path = root / declared["source_path"]
        digest, size = _digest(path)
        entries.append(
            EvidenceEntry(
                claim=declared["claim"],
                issue=declared["issue"],
                source_path=declared["source_path"],
                sha256=digest,
                size_bytes=size,
                evidence_class=evidence_class,
                collection_method=declared["collection_method"],
                environment=declared["environment"],
                sample_context=declared["sample_context"],
                limitations=declared["limitations"],
            )
        )

    adrs = sorted(path.name for path in (root / "docs" / "adr").glob("*.md"))
    return {
        "schema_version": 1,
        "sprint": "Signal Foundry Sprint 5 — Local ForecastOps, Shadow Evidence & Governance",
        "evidence_classes": list(EVIDENCE_CLASSES),
        "prospective_wallclock_evidence": None,
        "prospective_wallclock_note": (
            "Sprint 5 produced no prospective wall-clock evidence. Every readiness question "
            "defers to #63, and no result here may be read as live or paper-trading evidence."
        ),
        "adr_index": adrs,
        "entries": [asdict(entry) for entry in entries],
    }


def check_index(root: Path, index_path: Path) -> list[str]:
    """Return every discrepancy between the committed index and the artifacts."""
    try:
        committed = json.loads(index_path.read_text(encoding="utf-8"))
    except OSError:
        return [f"the evidence index is missing: {index_path}"]
    except ValueError as error:
        return [f"the evidence index is not valid JSON: {error}"]

    problems: list[str] = []
    rebuilt = build_index(root)

    if committed.get("entries") is None:
        return ["the evidence index declares no entries"]

    committed_by_path = {entry["source_path"]: entry for entry in committed["entries"]}
    rebuilt_by_path = {entry["source_path"]: entry for entry in rebuilt["entries"]}

    for path in sorted(set(rebuilt_by_path) - set(committed_by_path)):
        problems.append(f"{path}: declared in the builder but absent from the committed index")
    for path in sorted(set(committed_by_path) - set(rebuilt_by_path)):
        problems.append(f"{path}: present in the index but no longer declared")

    for path in sorted(set(committed_by_path) & set(rebuilt_by_path)):
        recorded = committed_by_path[path]
        current = rebuilt_by_path[path]
        if recorded["sha256"] != current["sha256"]:
            problems.append(
                f"{path}: digest drifted; the artifact changed after the index was written"
            )
        if recorded["evidence_class"] != current["evidence_class"]:
            problems.append(f"{path}: evidence class changed")
        if recorded["evidence_class"] in FORBIDDEN_CLASSES:
            problems.append(f"{path}: claims an evidence class Sprint 5 did not produce")

    if committed.get("adr_index") != rebuilt["adr_index"]:
        problems.append("the ADR index does not match docs/adr/")
    if committed.get("prospective_wallclock_evidence") is not None:
        problems.append("the index claims prospective wall-clock evidence, which does not exist")
    return problems


def main(argv: list[str] | None = None) -> int:
    """Build or check the evidence index."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument(
        "--index", type=Path, default=Path("docs/benchmarks/sprint5_evidence_index.json")
    )
    parser.add_argument("--check", action="store_true", help="verify instead of writing")
    arguments = parser.parse_args(argv)
    root = arguments.root.resolve()
    index_path = root / arguments.index

    try:
        if arguments.check:
            problems = check_index(root, index_path)
            if problems:
                print("sprint 5 evidence index does not check:")
                for problem in problems:
                    print(f"  - {problem}")
                return EXIT_REFUSED
            print("sprint 5 evidence index checks: every digest recomputes")
            return EXIT_OK
        index = build_index(root)
    except DossierError as error:
        print(f"dossier refused: {error}")
        return EXIT_REFUSED

    index_path.parent.mkdir(parents=True, exist_ok=True)
    index_path.write_text(json.dumps(index, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"wrote {index_path} with {len(index['entries'])} entries")
    return EXIT_OK


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    raise SystemExit(main())
