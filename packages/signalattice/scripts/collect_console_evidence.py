"""Collect machine-readable console evidence from real test and build runs.

SF-S5-SL-MR6. Every number this emits comes from an artifact a gate produced:
the bundle manifest from a build, the coverage summary from Vitest, the browser
results from Playwright, and the contrast ratios computed from the shipped
stylesheet. Nothing is typed in by hand, so the figure that renders this cannot
drift from what the gates actually measured.

The output is redistribution-safe: counts, ratios, and byte sizes only. No
source, no host path, no run identifier, and no market observation.
"""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

#: WCAG 2.2 AA minimum contrast for normal-size text.
AA_CONTRAST: Final = 4.5

#: The nine states every route and evidence panel must be able to represent.
EVIDENCE_STATES: Final = (
    "LOADING",
    "READY",
    "EMPTY",
    "PARTIAL",
    "INSUFFICIENT_EVIDENCE",
    "INVALID",
    "STALE",
    "UNAVAILABLE",
    "ERROR",
)

#: Maps a state constructor in the view source to the state it produces.
_CONSTRUCTORS: Final[dict[str, str]] = {
    "loading(": "LOADING",
    "ready(": "READY",
    "empty(": "EMPTY",
    "partial(": "PARTIAL",
    "insufficient(": "INSUFFICIENT_EVIDENCE",
    "invalid(": "INVALID",
    "stale(": "STALE",
    "unavailable(": "UNAVAILABLE",
    "errored(": "ERROR",
}

#: States every view inherits from the shared transport classifier, which maps
#: timeouts, refusals, oversized bodies, and decode failures for all of them.
_INHERITED: Final = ("LOADING", "EMPTY", "UNAVAILABLE", "ERROR")


class EvidenceError(RuntimeError):
    """Raised when required evidence is missing or unusable."""


@dataclass(frozen=True, slots=True)
class Budget:
    """One measured resource budget."""

    name: str
    observed: int
    limit: int

    @property
    def utilisation(self) -> float:
        """Fraction of the budget consumed."""
        return self.observed / self.limit if self.limit else 0.0


def _relative_luminance(hex_colour: str) -> float:
    """Return the WCAG relative luminance of an sRGB hex colour."""
    channels = [int(hex_colour[index : index + 2], 16) / 255 for index in (1, 3, 5)]
    linear = [
        channel / 12.92 if channel <= 0.04045 else ((channel + 0.055) / 1.055) ** 2.4
        for channel in channels
    ]
    return 0.2126 * linear[0] + 0.7152 * linear[1] + 0.0722 * linear[2]


def contrast_ratio(foreground: str, background: str) -> float:
    """Return the WCAG contrast ratio between two sRGB hex colours."""
    first = _relative_luminance(foreground)
    second = _relative_luminance(background)
    return (max(first, second) + 0.05) / (min(first, second) + 0.05)


def read_palette_contrast(stylesheet: Path) -> dict[str, float]:
    """Compute every light-palette token's contrast against the surface.

    Raises:
        EvidenceError: If the stylesheet does not declare the light palette.
    """
    text = stylesheet.read_text(encoding="utf-8")
    start = text.find(":root {")
    end = text.find("@media (prefers-color-scheme: dark)")
    if start < 0 or end < 0:
        raise EvidenceError("stylesheet does not declare a light palette block")
    block = text[start:end]
    tokens = dict(re.findall(r"--([a-z-]+):\s*(#[0-9a-f]{6})", block))
    surface = tokens.get("surface")
    if surface is None:
        raise EvidenceError("light palette declares no --surface")
    measured = ["ink", "ink-muted", "state-ready", "state-warn", "state-fail", "state-info"]
    return {name: contrast_ratio(tokens[name], surface) for name in measured if name in tokens}


def read_bundle_budgets(manifest: Path) -> tuple[Budget, ...]:
    """Return the budgets measured from a real build.

    Raises:
        EvidenceError: If the manifest is absent or malformed.
    """
    if not manifest.is_file():
        raise EvidenceError(f"{manifest} is missing; run `npm run budget` after a build")
    document = json.loads(manifest.read_text(encoding="utf-8"))
    budgets = document.get("budgets")
    if not isinstance(budgets, list) or not budgets:
        raise EvidenceError("bundle manifest declares no budgets")
    return tuple(
        Budget(name=str(item["name"]), observed=int(item["observed"]), limit=int(item["limit"]))
        for item in budgets
    )


def read_browser_results(report: Path) -> dict[str, dict[str, int]]:
    """Return per-project browser outcomes from a Playwright JSON report.

    Raises:
        EvidenceError: If the report is absent or has no specs.
    """
    if not report.is_file():
        raise EvidenceError(f"{report} is missing; run Playwright with the json reporter")
    document = json.loads(report.read_text(encoding="utf-8"))
    outcomes: dict[str, dict[str, int]] = {}

    def walk(node: dict[str, Any]) -> None:
        for spec in node.get("specs", []):
            for test in spec.get("tests", []):
                project = str(test.get("projectName", "unknown"))
                bucket = outcomes.setdefault(project, {"passed": 0, "skipped": 0, "failed": 0})
                status = str(test.get("status", "unknown"))
                if status == "expected":
                    bucket["passed"] += 1
                elif status == "skipped":
                    bucket["skipped"] += 1
                else:
                    bucket["failed"] += 1
        for child in node.get("suites", []):
            walk(child)

    for suite in document.get("suites", []):
        walk(suite)
    if not outcomes:
        raise EvidenceError("Playwright report contains no test outcomes")
    return outcomes


def read_coverage(summary: Path) -> dict[str, float]:
    """Return front-end coverage percentages.

    Raises:
        EvidenceError: If the summary is absent.
    """
    if not summary.is_file():
        raise EvidenceError(f"{summary} is missing; run `npm run test:coverage`")
    document = json.loads(summary.read_text(encoding="utf-8"))
    total = document.get("total", {})
    return {
        key: float(total[key]["pct"])
        for key in ("branches", "functions", "lines", "statements")
        if key in total
    }


def read_route_states(routes_directory: Path) -> dict[str, list[str]]:
    """Return the honest states each view is able to report.

    Derived by reading which state constructors each view calls, plus the states
    every view inherits from the shared transport classifier. This is a static
    fact about the source rather than a claim about test coverage, and the
    figure's caption says so.

    Raises:
        EvidenceError: If no view sources are found.
    """
    sources = sorted(routes_directory.glob("*.tsx"))
    if not sources:
        raise EvidenceError(f"no view sources under {routes_directory}")
    result: dict[str, list[str]] = {}
    for source in sources:
        text = source.read_text(encoding="utf-8")
        states = {state for token, state in _CONSTRUCTORS.items() if token in text}
        states.update(_INHERITED)
        result[source.stem] = sorted(states)
    return result


def collect(root: Path, *, playwright_report: Path) -> dict[str, Any]:
    """Assemble the complete redistribution-safe evidence document."""
    web = root / "web"
    return {
        "schema_version": 1,
        "evidence_class": "measured_local_engineering",
        "note": (
            "Counts, ratios, and byte sizes produced by real build and test runs on one "
            "local machine. Not a performance benchmark, a capacity claim, or evidence of "
            "profitability."
        ),
        "bundle_budgets": [
            {
                "name": budget.name,
                "observed": budget.observed,
                "limit": budget.limit,
                "utilisation": budget.utilisation,
            }
            for budget in read_bundle_budgets(web / "bundle-manifest.json")
        ],
        "coverage": read_coverage(web / "coverage" / "coverage-summary.json"),
        "browsers": read_browser_results(playwright_report),
        "palette_contrast": read_palette_contrast(web / "src" / "styles.css"),
        "aa_contrast_threshold": AA_CONTRAST,
        "route_states": read_route_states(web / "src" / "routes"),
        "evidence_states": list(EVIDENCE_STATES),
    }


def main(argv: list[str] | None = None) -> int:
    """Write the console evidence document."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--playwright-report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args(argv)

    try:
        document = collect(arguments.root, playwright_report=arguments.playwright_report)
    except EvidenceError as error:
        print(f"console evidence refused: {error}")
        return 2

    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(json.dumps(document, indent=2, sort_keys=True) + "\n")
    print(f"wrote {arguments.output}")
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    raise SystemExit(main())
