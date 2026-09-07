"""Run offline: python -m scripts.preservation inventory --help."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from scripts.preservation.git import PreservationError
from scripts.preservation.inventory import inventory, require_migration_clear
from scripts.preservation.publish import publish, read_ledger


def main() -> int:
    """Return 2 for sanitized input/I/O errors or a blocked static import gate."""
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    generate = commands.add_parser("inventory", help="hash committed Git objects; never import")
    generate.add_argument("--repository", type=Path, required=True)
    generate.add_argument("--source", choices=("alphaforge", "signalattice"), required=True)
    generate.add_argument(
        "--licenses", type=Path, default=Path("configs/preservation_licenses.json")
    )
    generate.add_argument("--output", type=Path, required=True)
    verify = commands.add_parser("verify", help="verify shards and reject static blockers")
    verify.add_argument("directory", type=Path)
    args = parser.parse_args()
    try:
        if args.command == "inventory":
            if args.licenses.stat().st_size > 65_536:
                raise PreservationError("license-policy-size")
            policy = json.loads(args.licenses.read_bytes())
            ledger = inventory(args.repository, args.source, policy[args.source])
            publish(ledger, args.output)
        else:
            ledger = read_ledger(args.directory)
        print(json.dumps(ledger["summary"], sort_keys=True))
        require_migration_clear(ledger)
    except (PreservationError, OSError, KeyError, TypeError, ValueError):
        parser.exit(2, "preservation: validation failed; no import authorized\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
