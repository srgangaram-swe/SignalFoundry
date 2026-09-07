"""Pack or verify complete release bytes without network or signing authority."""

from __future__ import annotations

import argparse
from pathlib import Path

from quant_platform.release.archive import pack_stage, verify_archive, verify_downloads


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("pack", "verify-archive", "verify-downloads"))
    parser.add_argument("source", type=Path)
    parser.add_argument("destination", type=Path)
    arguments = parser.parse_args()
    actions = {
        "pack": pack_stage,
        "verify-archive": verify_archive,
        "verify-downloads": verify_downloads,
    }
    actions[arguments.command](arguments.source, arguments.destination)
    print(f"{arguments.command}: verified")


if __name__ == "__main__":
    main()
