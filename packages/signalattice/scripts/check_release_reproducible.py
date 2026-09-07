"""Prove that two clean release builds of one commit are byte-identical.

SF-S5-SL-MR7. Reproducibility is the property that makes every other release
guarantee checkable: if the same commit and the same locked inputs can produce
different bytes, then a digest identifies one particular build rather than the
release, and no later verification means anything.

Two staging directories are built from scratch and compared file by file. The
report names what differed, because "not reproducible" without the offending
paths sends an operator hunting through the whole artifact set.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Final

#: Bounded: a wedged build must fail rather than hold a runner open.
BUILD_TIMEOUT_SECONDS: Final = 1800

EXIT_OK: Final = 0
EXIT_NOT_REPRODUCIBLE: Final = 2
EXIT_ENVIRONMENT: Final = 3


def _digests(root: Path) -> dict[str, str]:
    """Return a path -> SHA-256 map for every file under ``root``."""
    return {
        str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


def main(argv: list[str] | None = None) -> int:
    """Build twice and compare."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--first", type=Path, default=Path("build/release-a"))
    parser.add_argument("--second", type=Path, default=Path("build/release-b"))
    arguments = parser.parse_args(argv)
    root = arguments.root.resolve()

    stages = [root / arguments.first, root / arguments.second]
    for stage in stages:
        if stage.exists():
            shutil.rmtree(stage)
        try:
            subprocess.run(  # noqa: S603 - fixed argv, never shell
                [sys.executable, "scripts/release.py", "dry-run", "--staging", str(stage)],
                cwd=root,
                check=True,
                capture_output=True,
                text=True,
                timeout=BUILD_TIMEOUT_SECONDS,
            )
        except subprocess.CalledProcessError as error:
            print(f"release build failed:\n{error.stdout[-1500:]}\n{error.stderr[-1500:]}")
            return EXIT_ENVIRONMENT
        except (OSError, subprocess.TimeoutExpired) as error:
            print(f"release build could not run: {error}")
            return EXIT_ENVIRONMENT

    first, second = _digests(stages[0]), _digests(stages[1])
    differing = sorted(path for path in first if path in second and first[path] != second[path])
    asymmetric = sorted(set(first) ^ set(second))

    report = {
        "artifacts": len(first),
        "identical": len(first) - len(differing),
        "differing": differing,
        "present_in_one_build_only": asymmetric,
        "reproducible": not (differing or asymmetric),
    }
    print(json.dumps(report, indent=2, sort_keys=True))
    return EXIT_OK if report["reproducible"] else EXIT_NOT_REPRODUCIBLE


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    raise SystemExit(main())
