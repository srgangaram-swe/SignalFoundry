"""Fixed operation dispatch in an isolated package environment, with OS limits."""

from __future__ import annotations

import os
import resource
import sys
from contextlib import redirect_stdout
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from signal_foundry.boundary import MAX_REQUEST_BYTES, FoundryError, decode, encode
from signal_foundry.contracts import ResearchRequest


def limits() -> None:
    """Apply child-only CPU/address-space/file-descriptor/file-size ceilings."""
    try:
        resource.setrlimit(resource.RLIMIT_CPU, (120, 120))
        if sys.platform == "linux":
            resource.setrlimit(resource.RLIMIT_AS, (8 << 30, 8 << 30))
        resource.setrlimit(resource.RLIMIT_NOFILE, (128, 128))
        resource.setrlimit(resource.RLIMIT_FSIZE, (64 << 20, 64 << 20))
    except (OSError, ValueError) as exc:
        raise FoundryError(
            "resource_policy", "Required POSIX worker limits are unavailable.", 503
        ) from exc


def dispatch(source: str, operation: str, payload: dict[str, Any], root: Path) -> Any:
    """Closed operation set; only approved backend source may execute an operation."""
    bundles = Path(payload["bundles"]) if payload.get("bundles") is not None else None
    if source == "signalattice":
        from signal_foundry.worker_data import discover, inspect_bundle

        if operation == "catalog":
            return [item.model_dump(mode="json") for item in discover(bundles)]
        if operation == "validate" and bundles is not None:
            return inspect_bundle(bundles, payload["bundle_id"]).model_dump(mode="json")
    elif source == "alphaforge":
        if operation == "paper-qualification":
            from signal_foundry.worker_qualification import verify

            return verify(payload["request"])
        if operation == "catalog":
            from signal_foundry.worker_catalog import catalog

            return catalog().model_dump(mode="json")
        request = ResearchRequest.model_validate_json(encode(payload["request"]))
        from signal_foundry.worker_alpha import prepare, research

        if operation == "validate":
            return prepare(request, bundles).validation.model_dump(mode="json")
        if operation == "run":
            return research(request, bundles, root).model_dump(mode="json")
    raise FoundryError("unknown_operation", "Unregistered worker operation.")


def main() -> int:
    try:
        limits()
        if (
            len(sys.argv) != 3
            or os.name != "posix"
            or sys.platform not in {"linux", "darwin"}
        ):
            raise FoundryError("worker_arguments", "Invalid fixed worker invocation.")
        payload = decode(sys.stdin.buffer.read(MAX_REQUEST_BYTES + 1))
        if not isinstance(payload, dict) or set(payload) - {
            "request",
            "bundles",
            "bundle_id",
        }:
            raise FoundryError("worker_payload", "Invalid worker envelope.")
        # Source libraries may emit ordinary progress logs during validation.
        # Reserve stdout exclusively for the typed envelope; stderr remains
        # inside the same combined byte/time bounds, never an API response.
        with redirect_stdout(sys.stderr):
            result = dispatch(
                sys.argv[1], sys.argv[2], payload, Path(__file__).resolve().parents[1]
            )
        response = {"result": result}
    except FoundryError as exc:
        response = {
            "error": {"code": exc.code, "detail": exc.detail, "status": exc.status}
        }
    except ValidationError:
        response = {
            "error": {
                "code": "invalid_contract",
                "detail": "Worker rejected the typed request.",
                "status": 422,
            }
        }
    sys.stdout.buffer.write(encode(response))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
