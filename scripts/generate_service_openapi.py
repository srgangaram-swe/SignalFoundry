"""Generate or verify the deterministic Signalattice evidence-service contract.

The generator assembles the real FastAPI adapter over a fresh, private registry
and content-addressed store inside a temporary directory.  It never reads
operator state, Keychain credentials, market data, or the network.  Persistent
filesystem output occurs only when ``--output`` is supplied explicitly.
"""

from __future__ import annotations

import argparse
import hashlib
import os
import sys
import tempfile
from pathlib import Path

from quant_platform.service.api import assert_read_only_route_inventory, create_app
from quant_platform.service.models import canonical_openapi_bytes
from quant_platform.tracking.cas import ArtifactStore
from quant_platform.tracking.read_ports import RegistryReadPorts
from quant_platform.tracking.registry import RunRegistry

_GENERATOR_SECRET = hashlib.sha256(b"signalattice.openapi.generator.ephemeral-registry.v1").digest()


def generate_openapi() -> bytes:
    """Return canonical OpenAPI bytes from an isolated real storage assembly.

    The fixed digest input is test material for a registry deleted before this
    function returns; it is not an operator credential.  Resolving the temporary
    root avoids traversing macOS's ``/var`` compatibility symlink through the
    CAS no-follow boundary.
    """

    with tempfile.TemporaryDirectory(prefix="signalattice-openapi-") as raw_root:
        root = Path(raw_root).resolve(strict=True)
        store = ArtifactStore(root / "cas")
        store.initialize()
        registry = RunRegistry(
            root / "registry.sqlite",
            digest_secret=_GENERATOR_SECRET,
            artifact_verifier=store,
        )
        registry.initialize()
        ports = RegistryReadPorts(registry, store)
        if not ports.probe_evidence_readiness().ready:
            raise RuntimeError("temporary evidence boundary did not become ready")
        app = create_app(ports)
        assert_read_only_route_inventory(app)
        document = app.openapi()
        return canonical_openapi_bytes(document)


def _atomic_write(destination: Path, payload: bytes) -> None:
    """Atomically replace one explicitly selected regular output file."""

    parent = destination.parent
    if not parent.is_dir():
        raise FileNotFoundError(f"output parent directory does not exist: {parent}")
    if destination.is_symlink() or (destination.exists() and not destination.is_file()):
        raise ValueError("output must be a regular file and may not be a symbolic link")

    descriptor, temporary_name = tempfile.mkstemp(
        dir=parent,
        prefix=f".{destination.name}.",
        suffix=".tmp",
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.chmod(temporary, 0o644)
        os.replace(temporary, destination)
        directory_descriptor = os.open(parent, os.O_RDONLY)
        try:
            os.fsync(directory_descriptor)
        finally:
            os.close(directory_descriptor)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


def _check(expected: Path, payload: bytes) -> bool:
    """Return whether ``expected`` exactly equals the generated contract."""

    if expected.is_symlink() or not expected.is_file():
        return False
    try:
        observed = expected.read_bytes()
    except OSError:
        return False
    return observed == payload


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    destination = parser.add_mutually_exclusive_group()
    destination.add_argument(
        "--output",
        type=Path,
        help="Atomically write the generated contract to this explicit path.",
    )
    destination.add_argument(
        "--check",
        type=Path,
        help="Verify that this existing file exactly matches without writing it.",
    )
    return parser


def main() -> None:
    """Generate to stdout, atomically write, or verify one checked artifact."""

    arguments = _parser().parse_args()
    payload = generate_openapi()
    if arguments.output is not None:
        _atomic_write(arguments.output, payload)
        print(f"wrote {arguments.output} ({len(payload)} bytes)")
        return
    if arguments.check is not None:
        if not _check(arguments.check, payload):
            print(
                f"OpenAPI contract differs from {arguments.check}; regenerate with --output",
                file=sys.stderr,
            )
            raise SystemExit(1)
        print(f"verified {arguments.check} ({len(payload)} bytes)")
        return
    sys.stdout.buffer.write(payload)


if __name__ == "__main__":
    main()
