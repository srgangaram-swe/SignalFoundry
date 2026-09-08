"""Derive reviewable root workflows from the exact preserved source gates.

Only location/context, bounded job deadlines and locked bootstrap adaptations are
allowed. Original test commands, floors, action pins and permissions are retained.
Release publication is intentionally not activated by repository assembly.
"""

from __future__ import annotations

import argparse
import copy
from pathlib import Path
from typing import Any

import yaml

CHECKOUT = "actions/checkout@3d3c42e5aac5ba805825da76410c181273ba90b1"
PYTHON = "actions/setup-python@5fda3b95a4ea91299a34e894583c3862153e4b97"
UV = "astral-sh/setup-uv@c771a70e6277c0a99b617c7a806ffedaca235ff9"
NODE = "actions/setup-node@2028fbc5c25fe9cf00d9f06a71cc4710d4507903"


def relocate(source: str, original: dict[str, Any]) -> dict[str, Any]:
    """Relocate one job while preserving all test/security semantics."""
    job = copy.deepcopy(original)
    prefix = f"packages/{source}"
    job["name"] = f"{source}: {job.get('name', '')}".rstrip()
    matrix = job.get("strategy", {}).get("matrix", {})
    if "python-version" in matrix and "matrix.python-version" not in job["name"]:
        job["name"] += " (Python ${{ matrix.python-version }})"
    elif "include" in matrix and "matrix.extra" not in job["name"]:
        job["name"] += " (${{ matrix.extra }})"
    job.setdefault("timeout-minutes", 30)
    job.setdefault("defaults", {}).setdefault("run", {})["working-directory"] = prefix
    steps = []
    for step in job["steps"]:
        if "working-directory" in step:
            step["working-directory"] = f"{prefix}/{step['working-directory']}"
        uses = step.get("uses", "")
        options = step.get("with", {})
        for key in ("node-version-file", "cache-dependency-path"):
            if key in options:
                options[key] = f"{prefix}/{options[key]}"
        if uses.startswith("actions/setup-python@"):
            # The relocated environment uses uv's exact lock, not pip's free resolver.
            options.pop("cache", None)
        if uses.startswith("actions/upload-artifact@"):
            paths = options.get("path", "").splitlines()
            options["path"] = "\n".join(
                line if line.startswith(("$", "/")) else f"{prefix}/{line}"
                for line in paths
            )
            options["name"] = f"{source}-{options.get('name', 'artifact')}"
        if "run" in step:
            command = step["run"]
            for expression in ("${GITHUB_WORKSPACE}", "$GITHUB_WORKSPACE"):
                command = command.replace(expression, expression + "/" + prefix)
            if source == "signalattice":
                replacements = {
                    'python -m pip install -e ".[dev]"': "uv sync --locked --extra dev",
                    'python -m pip install -e ".[dev,torch]"': (
                        "uv sync --locked --extra dev --extra torch"
                    ),
                    'python -m pip install -e ".[dev,mlflow]"': (
                        "uv sync --locked --extra dev --extra mlflow"
                    ),
                    "python -m pip install --upgrade pip build": (
                        "uv sync --locked --extra dev"
                    ),
                    "python -m pip install uv": "uv --version",
                    "python -m pip install .": "uv sync --locked",
                    "uv sync --all-extras --frozen": "uv sync --all-extras --locked",
                }
                # Only whole bootstrap commands may change. Substring replacement
                # corrupts absolute wheel/sdist interpreter paths and their newlines.
                command = "\n".join(
                    replacements.get(line, line)
                    for line in command.split("\n")
                    if line != "python -m pip install --upgrade pip"
                )
            step["run"] = command
        if uses.startswith("actions/checkout@"):
            step.setdefault("with", {})["fetch-depth"] = 0
            step["with"]["persist-credentials"] = False
        steps.append(step)
        if uses.startswith("actions/checkout@"):
            steps.append(
                {
                    "name": "Verify offline package provenance context",
                    "working-directory": ".",
                    "run": f"python3 -m foundry_build.context {source}",
                }
            )
        if source == "signalattice" and uses.startswith("actions/setup-python@"):
            steps.extend(
                [
                    {"uses": UV, "with": {"version": "0.11.32"}},
                    {
                        "name": "Expose isolated package environment",
                        "run": (
                            f'echo "$GITHUB_WORKSPACE/{prefix}/.venv/bin" '
                            '>> "$GITHUB_PATH"'
                        ),
                    },
                ]
            )
    job["steps"] = steps
    return job


def generate(root: Path) -> dict[str, Any]:
    """Return the full CI replica and mandatory aggregate gates."""
    jobs: dict[str, Any] = {}
    required: dict[str, list[str]] = {"alphaforge": [], "signalattice": []}
    for source in required:
        paths = ["ci.yml", "security.yml"]
        if source == "signalattice":
            paths += ["release.yml", "service-security.yml"]
        for filename in paths:
            workflow = yaml.safe_load(
                (
                    root / "packages" / source / ".github" / "workflows" / filename
                ).read_text()
            )
            for key, original in workflow["jobs"].items():
                if filename == "release.yml" and key != "dry-run":
                    continue
                identifier = f"{source}-{key}"
                if identifier in jobs:
                    raise ValueError("duplicate source job identity")
                original = copy.deepcopy(original)
                original.setdefault("name", key)
                job = relocate(source, original)
                if "needs" in job:
                    dependencies = job["needs"]
                    job["needs"] = [
                        f"{source}-{dependency}" for dependency in dependencies
                    ]
                # Preserve each workflow's isolated environment.
                job["env"] = {**workflow.get("env", {}), **job.get("env", {})}
                jobs[identifier] = job
                # Source #67 is non-required and failing; retain it visibly,
                # never relabel it as successful or quietly activate a different policy.
                if filename != "service-security.yml":
                    required[source].append(identifier)
    for source, dependencies in required.items():
        jobs[source] = {
            "name": source,
            "runs-on": "ubuntu-latest",
            "timeout-minutes": 2,
            "needs": dependencies,
            "if": "always()",
            "steps": [
                {
                    "name": "Require every preserved source gate",
                    "env": {
                        "RESULTS": "${{ toJSON(needs) }}",
                        "EVENT": "${{ github.event_name }}",
                    },
                    "run": (
                        "python3 -c 'import json,os; "
                        'r=json.loads(os.environ["RESULTS"]); '
                        'assert all(v["result"] == "success" or '
                        '(k.endswith("-dependency-review") and '
                        'os.environ["EVENT"] != "pull_request" and '
                        'v["result"] == "skipped") for k,v in r.items()), r\''
                    ),
                }
            ],
        }
    jobs["assembly"] = {
        "name": "assembly",
        "needs": ["nexus"],
        "runs-on": "ubuntu-latest",
        "timeout-minutes": 15,
        "steps": [
            {
                "uses": CHECKOUT,
                "with": {"fetch-depth": 0, "persist-credentials": False},
            },
            {"uses": PYTHON, "with": {"python-version": "3.13"}},
            {"uses": UV, "with": {"version": "0.11.32"}},
            {"run": "uv sync --locked --extra dev"},
            {
                "run": (
                    "uv sync --project packages/alphaforge --locked --extra dev --extra"
                    " data"
                )
            },
            {"run": "uv sync --project packages/signalattice --locked --extra dev"},
            {"run": "uv run python -m foundry_build.context alphaforge"},
            {"run": "uv run python -m foundry_build.context signalattice"},
            {"run": "uv run black --check foundry_build signal_foundry tests"},
            {"run": "uv run ruff check foundry_build signal_foundry tests"},
            {"run": "uv run mypy foundry_build signal_foundry"},
            {"run": "uv run python -m foundry_build.workflows --check"},
            {
                "env": {"FOUNDRY_TEST_COVERAGE": "1"},
                "run": (
                    "uv run pytest --cov=foundry_build --cov=signal_foundry"
                    " --cov-branch --cov-report=term-missing --cov-fail-under=90"
                ),
            },
            {"run": "uv run python -m foundry_build.contracts --check"},
            {"uses": NODE, "with": {"node-version": "24"}},
            {"run": "npm --prefix contracts ci --ignore-scripts"},
            {"run": "npm --prefix contracts run generate"},
            {"run": "npm --prefix contracts run check"},
            {"run": "git diff --exit-code -- contracts"},
            {"run": "npm --prefix contracts audit"},
            {"run": "uv build --no-sources"},
            {"run": "uv run python -m foundry_build.assembly verify"},
        ],
    }
    jobs["nexus"] = {
        "name": "nexus",
        "runs-on": "macos-15",
        "timeout-minutes": 20,
        "steps": [
            {
                "uses": CHECKOUT,
                "with": {"fetch-depth": 0, "persist-credentials": False},
            },
            {"uses": PYTHON, "with": {"python-version": "3.13"}},
            {"uses": UV, "with": {"version": "0.11.32"}},
            {"uses": NODE, "with": {"node-version": "24"}},
            {"run": "uv sync --locked --extra dev"},
            {
                "run": (
                    "uv sync --project packages/alphaforge --locked --extra dev "
                    "--extra data"
                )
            },
            {"run": "uv sync --project packages/signalattice --locked --extra dev"},
            {"run": "uv run python -m foundry_build.context alphaforge"},
            {"run": "uv run python -m foundry_build.context signalattice"},
            {"run": "npm ci --ignore-scripts", "working-directory": "apps/nexus"},
            {
                "run": (
                    "npm run format:check && npm run lint && npm run build && "
                    "npm run test:coverage && npm audit"
                ),
                "working-directory": "apps/nexus",
            },
            {
                "run": "npx playwright install chromium",
                "working-directory": "apps/nexus",
            },
            {"run": "npm run e2e", "working-directory": "apps/nexus"},
            {
                "uses": "actions/upload-artifact@ea165f8d65b6e75b540449e92b4886f43607fa02",
                "if": "always()",
                "with": {
                    "name": "nexus-browser-evidence",
                    "path": "apps/nexus/test-results",
                    "retention-days": 7,
                    "if-no-files-found": "error",
                },
            },
        ],
    }
    jobs["security"] = {
        "name": "security",
        "runs-on": "ubuntu-latest",
        "timeout-minutes": 10,
        "steps": [
            {
                "uses": CHECKOUT,
                "with": {"fetch-depth": 0, "persist-credentials": False},
            },
            {"uses": PYTHON, "with": {"python-version": "3.13"}},
            {"uses": UV, "with": {"version": "0.11.32"}},
            {"run": "uv sync --locked --extra dev"},
            {
                "run": (
                    "uv export --locked --no-dev --extra dev --no-emit-project "
                    "--format requirements-txt --output-file /tmp/foundry-audit.txt"
                )
            },
            {
                "run": (
                    "uv run pip-audit --strict --require-hashes --disable-pip "
                    "--progress-spinner=off --requirement /tmp/foundry-audit.txt"
                )
            },
        ],
    }
    return {
        "name": "Signal Foundry qualification",
        "on": {
            "push": {"branches": ["dev", "prod", "main"]},
            "pull_request": {"branches": ["dev", "prod", "main"]},
        },
        "permissions": {"contents": "read"},
        "concurrency": {
            "group": "qualification-${{ github.ref }}",
            "cancel-in-progress": True,
        },
        "jobs": jobs,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    output = (
        "# Generated by python -m foundry_build.workflows; review gate adapters.\n"
        + yaml.safe_dump(generate(root), sort_keys=False, width=100)
    )
    path = root / ".github" / "workflows" / "qualification.yml"
    if args.check:
        if path.read_text() != output:
            parser.exit(2, "root workflow differs from its source-gate contract\n")
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
