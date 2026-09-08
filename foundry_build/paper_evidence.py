"""Collect focused software checks and render their safe, measured evidence.

Rendering uses only committed aggregates. Test timing is a single local run,
not exchange latency or a performance benchmark; no broker credentials are read.
"""

from __future__ import annotations

import argparse
import json
import platform
import subprocess
import sys
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

ROOT = Path(__file__).resolve().parents[1]
DESTINATION = ROOT / "docs/evidence/paper"
TESTS = [
    f"tests/test_paper_{name}.py"
    for name in ("core", "transport", "integration", "guards", "stop")
]


def collect() -> dict[str, Any]:
    """Run only the paper slice, retaining every result and per-module coverage."""
    with tempfile.TemporaryDirectory(prefix="paper-evidence-") as temporary:
        directory = Path(temporary)
        subprocess.run(
            [
                sys.executable,
                "-m",
                "pytest",
                *TESTS,
                "--cov=signal_foundry.trading",
                "--cov-branch",
                "--cov-fail-under=90",
                f"--cov-report=json:{directory / 'coverage.json'}",
                f"--junitxml={directory / 'tests.xml'}",
            ],
            cwd=ROOT,
            check=True,
            timeout=180,
        )
        coverage = json.loads((directory / "coverage.json").read_text())
        tests = [
            {
                "case": row.attrib["name"],
                "group": row.attrib["classname"].rsplit("_", 1)[-1],
                "seconds": float(row.attrib["time"]),
                "passed": not any(
                    child.tag in {"failure", "error", "skipped"} for child in row
                ),
            }
            for row in ET.parse(directory / "tests.xml").iter("testcase")
        ]
        modules = []
        for name, value in coverage["files"].items():
            summary = value["summary"]
            if summary["num_branches"]:
                modules.append(
                    {
                        "module": Path(name).stem,
                        "covered_branches": summary["covered_branches"],
                        "branches": summary["num_branches"],
                    }
                )
        return {
            "schema_version": 1,
            "evidence_class": "software_fixture",
            "broker_requests": 0,
            "live_orders": 0,
            "environment": {
                "python": platform.python_version(),
                "system": platform.system(),
                "machine": platform.machine(),
            },
            "tests": tests,
            "modules": modules,
            "coverage": coverage["totals"],
            "limitations": (
                "One local test run; fixture transports, "
                "no broker acceptance or strategy evidence."
            ),
        }


def render(value: dict[str, Any], destination: Path) -> None:
    """Reproduce Seaborn coverage and timing distributions from safe aggregates."""
    sns.set_theme(context="notebook", style="whitegrid", palette="colorblind")
    modules = pd.DataFrame(value["modules"])
    modules["coverage"] = modules["covered_branches"] / modules["branches"] * 100
    tests = pd.DataFrame(value["tests"])
    tests["milliseconds"] = tests["seconds"] * 1000
    figure, axes = plt.subplots(1, 2, figsize=(14, 6))
    sns.barplot(
        data=modules, x="coverage", y="module", color=sns.color_palette()[0], ax=axes[0]
    )
    axes[0].set(
        xlim=(0, 100),
        xlabel="Covered branch outcomes (%)",
        ylabel="Paper module",
        title="All paper modules, including weaker paths",
    )
    sns.stripplot(
        data=tests,
        x="milliseconds",
        y="group",
        jitter=False,
        alpha=0.55,
        color=sns.color_palette()[1],
        ax=axes[1],
    )
    axes[1].set(
        xscale="symlog",
        xlabel="Test duration (ms, symlog; zero retained)",
        ylabel="Focused test group",
        title=f"All {len(tests)} test durations; one local run",
    )
    axes[1].set_xlim(0, max(1, float(tests["milliseconds"].max()) * 1.2))
    figure.suptitle(
        "Paper wiring: measured software checks, fixture broker only", fontsize=15
    )
    environment = value["environment"]
    figure.text(
        0.5,
        0.02,
        f"{environment['system']} {environment['machine']} · "
        f"Python {environment['python']} · "
        "deterministic fixtures / research seed 17\n"
        "No real broker requests, prospective sessions or live profitability evidence. "
        "Timing includes isolated worker startup.",
        ha="center",
        fontsize=10,
    )
    figure.tight_layout(rect=(0, 0.1, 1, 0.94))
    figure.savefig(destination / "software-checks.png", dpi=150)
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--collect", action="store_true")
    args = parser.parse_args()
    DESTINATION.mkdir(parents=True, exist_ok=True)
    path = DESTINATION / "measurements.json"
    if args.collect:
        path.write_text(json.dumps(collect(), indent=2, sort_keys=True) + "\n")
    render(json.loads(path.read_text()), DESTINATION)


if __name__ == "__main__":
    main()
