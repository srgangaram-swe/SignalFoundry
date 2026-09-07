"""Generate or verify the deterministic OpenAPI source of client bindings."""

from __future__ import annotations

import argparse
from pathlib import Path

from signal_foundry.api import create_app
from signal_foundry.boundary import encode
from signal_foundry.manager import Manager


def schema() -> bytes:
    def no_runtime() -> Manager:
        raise RuntimeError("Schema generation must not start a research service")

    return encode(create_app(no_runtime).openapi()) + b"\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", type=Path, default=Path("contracts/openapi-v1.json")
    )
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    expected = schema()
    if args.check:
        if args.output.read_bytes() != expected:
            parser.error("OpenAPI differs from the reviewed generated contract")
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_bytes(expected)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
