from __future__ import annotations

import argparse
from pathlib import Path

from _common import latest_run_dir

from alphaforge.reporting import write_markdown_report
from alphaforge.research import refresh_experiment_manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate a markdown report for a run.")
    parser.add_argument("--run-dir")
    parser.add_argument("--latest", action="store_true")
    parser.add_argument("--output")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_dir = Path(args.run_dir) if args.run_dir else latest_run_dir()
    report = write_markdown_report(run_dir, output_path=args.output)
    refresh_experiment_manifest(run_dir)
    print(f"report written: {report}")


if __name__ == "__main__":
    main()
