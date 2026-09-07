"""Verify the committed Sprint 3 Signalattice provenance receipt locally."""

from __future__ import annotations

import argparse

from alphaforge.research.cross_repository_provenance import (
    verify_cross_repository_receipt,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Verify AlphaForge's aggregate Sprint 3 receipt against an existing "
            "local Signalattice Git object database without network access."
        )
    )
    parser.add_argument(
        "--receipt",
        default=("docs/evidence/signal_foundry_sprint_3/" "cross_repository_provenance.json"),
        help="Committed strict JSON receipt.",
    )
    parser.add_argument(
        "--signalattice",
        required=True,
        help="Existing local Signalattice checkout; no fetch or checkout occurs.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = verify_cross_repository_receipt(
        args.receipt,
        checkout=args.signalattice,
    )
    print(f"repository={result.repository}")
    print(f"commit={result.commit}")
    print(f"sources={result.source_count}")
    print(f"bytes={result.total_bytes}")
    print(f"receipt_sha256={result.receipt_sha256}")
    print("network_requests=0")


if __name__ == "__main__":
    main()
