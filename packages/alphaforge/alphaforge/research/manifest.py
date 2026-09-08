"""Versioned experiment identity, environment, seed, and artifact provenance.

The manifest boundary is deliberately independent of a workflow runner. It accepts
only already-validated domain configuration, converts it into a canonical semantic
identity, and records execution metadata separately so timestamps never change the
experiment identity.

Security invariants:

* environment capture is allowlisted rather than copied wholesale;
* CLI arguments with credential-like names are redacted;
* artifact paths must be relative to one run root; and
* hashes use canonical JSON or file bytes, never Python object serialization.
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import subprocess
import sys
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, replace
from datetime import UTC, date, datetime
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

MANIFEST_SCHEMA_VERSION = "1.0.0"
SHA256_HEX_LENGTH = 64
FULL_GIT_SHA_LENGTH = 40
SAFE_ENVIRONMENT_KEYS = (
    "CUDA_VISIBLE_DEVICES",
    "JAX_ENABLE_X64",
    "MKL_NUM_THREADS",
    "OMP_NUM_THREADS",
    "PYTHONHASHSEED",
    "TOKENIZERS_PARALLELISM",
)
DEFAULT_SEED_STREAMS = (
    "python",
    "numpy",
    "scikit_learn",
    "lightgbm",
    "xgboost",
    "pytorch",
    "tensorflow",
    "data_loader",
    "hyperparameter_search",
)
DEFAULT_DEPENDENCIES = (
    "alphaforge",
    "lightgbm",
    "matplotlib",
    "numpy",
    "pandas",
    "pyarrow",
    "pydantic",
    "PyYAML",
    "scikit-learn",
    "seaborn",
    "scipy",
    "tensorflow",
    "torch",
    "xgboost",
)
SENSITIVE_ARGUMENT_FRAGMENTS = ("api-key", "apikey", "password", "secret", "token")


class ManifestValidationError(ValueError):
    """Raised when experiment provenance is incomplete or internally inconsistent."""


def _is_lower_hex(value: object, length: int) -> bool:
    return (
        isinstance(value, str)
        and len(value) == length
        and value == value.lower()
        and all(character in "0123456789abcdef" for character in value)
    )


def _parse_utc_timestamp(value: object) -> datetime:
    if not isinstance(value, str):
        raise ManifestValidationError("execution timestamps must be ISO-8601 strings")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ManifestValidationError("execution timestamps must be valid ISO-8601") from exc
    if parsed.utcoffset() != UTC.utcoffset(parsed):
        raise ManifestValidationError("execution timestamps must be UTC")
    return parsed


def canonical_json(value: Any) -> bytes:
    """Serialize JSON deterministically for identities and manifests."""
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode(
        "utf-8"
    )


def sha256_bytes(payload: bytes) -> str:
    """Return a lowercase SHA-256 digest."""
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path) -> str:
    """Hash one file with bounded memory."""
    hasher = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def derive_seed_map(
    root_seed: int, streams: Sequence[str] = DEFAULT_SEED_STREAMS
) -> dict[str, int]:
    """Derive stable named 32-bit streams independent of candidate ordering."""
    if root_seed < 0:
        raise ManifestValidationError("root_seed must be nonnegative")
    normalized = tuple(stream.strip() for stream in streams)
    if not normalized or any(not stream for stream in normalized):
        raise ManifestValidationError("seed stream names must be non-empty")
    if len(set(normalized)) != len(normalized):
        raise ManifestValidationError("seed stream names must be unique")
    return {
        stream: int.from_bytes(
            hashlib.sha256(f"alphaforge-seed-v1:{root_seed}:{stream}".encode()).digest()[:4],
            byteorder="big",
        )
        for stream in sorted(normalized)
    }


def dependency_versions(packages: Sequence[str] = DEFAULT_DEPENDENCIES) -> dict[str, str]:
    """Resolve the exact installed versions that can affect numerical evidence."""
    resolved: dict[str, str] = {}
    for package in sorted(set(packages)):
        try:
            resolved[package] = version(package)
        except PackageNotFoundError:
            resolved[package] = "not-installed"
    return resolved


def capture_environment(
    *,
    environ: Mapping[str, str] | None = None,
    packages: Sequence[str] = DEFAULT_DEPENDENCIES,
) -> dict[str, Any]:
    """Capture an allowlisted, secret-safe runtime and hardware snapshot."""
    source = os.environ if environ is None else environ
    safe_environment = {key: source[key] for key in SAFE_ENVIRONMENT_KEYS if key in source}
    accelerator: dict[str, Any] = {
        "backend": "cpu",
        "cuda_runtime": "not-installed",
        "devices": [],
    }
    try:
        import torch

        torch_version = getattr(torch, "version", None)
        cuda = getattr(torch, "cuda", None)
        backends = getattr(torch, "backends", None)
        accelerator["cuda_runtime"] = getattr(torch_version, "cuda", None) or "not-available"
        if cuda is None:
            accelerator["probe_status"] = "incomplete-install"
        elif cuda.is_available():
            accelerator["backend"] = "cuda"
            accelerator["devices"] = [
                cuda.get_device_name(index) for index in range(cuda.device_count())
            ]
        elif backends is not None and hasattr(backends, "mps") and backends.mps.is_available():
            accelerator["backend"] = "mps"
            accelerator["devices"] = ["Apple Metal Performance Shaders"]
    except ImportError:
        pass
    return {
        "python": {
            "version": platform.python_version(),
            "implementation": platform.python_implementation(),
            "executable_name": Path(sys.executable).name,
        },
        "operating_system": {
            "system": platform.system(),
            "release": platform.release(),
            "machine": platform.machine(),
        },
        "hardware": {
            "logical_cpu_count": os.cpu_count(),
            "processor": platform.processor(),
            "accelerator": accelerator,
        },
        "dependencies": dependency_versions(packages),
        "environment": safe_environment,
    }


def capture_git_context(repository: str | Path = ".") -> dict[str, Any]:
    """Return full Git SHA, branch, and dirty state without mutating the repository."""

    def run(*arguments: str) -> str:
        result = subprocess.run(
            ["git", "-C", str(repository), *arguments],
            check=False,
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode != 0:
            raise ManifestValidationError(
                f"cannot determine Git context using {' '.join(arguments)}"
            )
        return result.stdout.strip()

    sha = run("rev-parse", "HEAD")
    if len(sha) != FULL_GIT_SHA_LENGTH:
        raise ManifestValidationError("experiment provenance requires a full Git SHA")
    branch = run("branch", "--show-current") or "detached"
    return {
        "sha": sha,
        "branch": branch,
        "dirty": bool(run("status", "--porcelain=v1")),
    }


def redact_cli_arguments(arguments: Sequence[str]) -> list[str]:
    """Redact values associated with credential-like CLI options."""
    redacted: list[str] = []
    redact_next = False
    for argument in arguments:
        lowered = argument.lower()
        if redact_next:
            redacted.append("[REDACTED]")
            redact_next = False
            continue
        if any(fragment in lowered for fragment in SENSITIVE_ARGUMENT_FRAGMENTS):
            if "=" in argument:
                redacted.append(f"{argument.split('=', 1)[0]}=[REDACTED]")
            else:
                redacted.append(argument)
                redact_next = True
            continue
        redacted.append(argument)
    return redacted


def inventory_artifacts(run_root: Path, *, exclude: Sequence[str] = ()) -> list[dict[str, Any]]:
    """Return deterministic relative-path, byte-size, and SHA-256 artifact records."""
    root = run_root.resolve()
    excluded = set(exclude)
    records: list[dict[str, Any]] = []
    for path in sorted(candidate for candidate in run_root.rglob("*") if candidate.is_file()):
        resolved = path.resolve()
        try:
            relative = resolved.relative_to(root)
        except ValueError as exc:
            raise ManifestValidationError("artifact path escapes the run root") from exc
        relative_text = relative.as_posix()
        if relative_text in excluded:
            continue
        records.append(
            {
                "path": relative_text,
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    return records


@dataclass(frozen=True)
class ExperimentManifest:
    """Validated manifest with semantic identity separated from execution metadata."""

    schema_version: str
    experiment_id: str
    config_sha256: str
    code: Mapping[str, Any]
    dataset: Mapping[str, Any]
    universe: tuple[str, ...]
    date_range: Mapping[str, str]
    features: Mapping[str, Any]
    label: Mapping[str, Any]
    models: tuple[Mapping[str, Any], ...]
    validation: Mapping[str, Any]
    transaction_costs: Mapping[str, Any]
    seeds: Mapping[str, int]
    environment: Mapping[str, Any]
    invocation: Mapping[str, Any]
    execution: Mapping[str, str]
    artifacts: tuple[Mapping[str, Any], ...]

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> ExperimentManifest:
        """Reconstruct a manifest from JSON-compatible data and revalidate it."""

        expected = set(cls.__dataclass_fields__)
        if set(value) != expected:
            raise ManifestValidationError("experiment manifest fields mismatch")
        normalized = dict(value)
        normalized["universe"] = tuple(value["universe"])
        normalized["models"] = tuple(value["models"])
        normalized["artifacts"] = tuple(value["artifacts"])
        manifest = cls(**normalized)
        manifest.validate()
        return manifest

    @staticmethod
    def _semantic_payload(
        *,
        code: Mapping[str, Any],
        dataset: Mapping[str, Any],
        universe: Sequence[str],
        date_range: Mapping[str, str],
        features: Mapping[str, Any],
        label: Mapping[str, Any],
        models: Sequence[Mapping[str, Any]],
        validation: Mapping[str, Any],
        transaction_costs: Mapping[str, Any],
        seeds: Mapping[str, int],
        environment: Mapping[str, Any],
    ) -> dict[str, Any]:
        return {
            "schema_version": MANIFEST_SCHEMA_VERSION,
            "code": dict(code),
            "dataset": dict(dataset),
            "universe": list(universe),
            "date_range": dict(date_range),
            "features": dict(features),
            "label": dict(label),
            "models": [dict(model) for model in models],
            "validation": dict(validation),
            "transaction_costs": dict(transaction_costs),
            "seeds": dict(seeds),
            "environment": dict(environment),
        }

    @classmethod
    def build(
        cls,
        *,
        code: Mapping[str, Any],
        dataset: Mapping[str, Any],
        universe: Sequence[str],
        date_range: Mapping[str, str],
        features: Mapping[str, Any],
        label: Mapping[str, Any],
        models: Sequence[Mapping[str, Any]],
        validation: Mapping[str, Any],
        transaction_costs: Mapping[str, Any],
        root_seed: int,
        environment: Mapping[str, Any],
        invocation: Mapping[str, Any],
        execution: Mapping[str, str],
        artifacts: Sequence[Mapping[str, Any]],
    ) -> ExperimentManifest:
        """Build and validate a manifest from domain-validated inputs."""
        normalized_universe = tuple(sorted(set(symbol.strip() for symbol in universe)))
        if not normalized_universe or any(not symbol for symbol in normalized_universe):
            raise ManifestValidationError("universe must contain non-empty symbols")
        seed_map = derive_seed_map(root_seed)
        semantic_config = cls._semantic_payload(
            code=code,
            dataset=dataset,
            universe=normalized_universe,
            date_range=date_range,
            features=features,
            label=label,
            models=models,
            validation=validation,
            transaction_costs=transaction_costs,
            seeds=seed_map,
            environment=environment,
        )
        config_sha256 = sha256_bytes(canonical_json(semantic_config))
        manifest = cls(
            schema_version=MANIFEST_SCHEMA_VERSION,
            experiment_id=config_sha256,
            config_sha256=config_sha256,
            code=dict(code),
            dataset=dict(dataset),
            universe=normalized_universe,
            date_range=dict(date_range),
            features=dict(features),
            label=dict(label),
            models=tuple(dict(model) for model in models),
            validation=dict(validation),
            transaction_costs=dict(transaction_costs),
            seeds=seed_map,
            environment=dict(environment),
            invocation=dict(invocation),
            execution=dict(execution),
            artifacts=tuple(dict(artifact) for artifact in artifacts),
        )
        manifest.validate()
        return manifest

    def validate(self) -> None:
        """Validate identities, time bounds, Git provenance, and artifact records."""
        if self.schema_version != MANIFEST_SCHEMA_VERSION:
            raise ManifestValidationError("unsupported experiment manifest schema")
        expected_identity = sha256_bytes(
            canonical_json(
                self._semantic_payload(
                    code=self.code,
                    dataset=self.dataset,
                    universe=self.universe,
                    date_range=self.date_range,
                    features=self.features,
                    label=self.label,
                    models=self.models,
                    validation=self.validation,
                    transaction_costs=self.transaction_costs,
                    seeds=self.seeds,
                    environment=self.environment,
                )
            )
        )
        if not (
            _is_lower_hex(self.experiment_id, SHA256_HEX_LENGTH)
            and _is_lower_hex(self.config_sha256, SHA256_HEX_LENGTH)
            and self.experiment_id == self.config_sha256 == expected_identity
        ):
            raise ManifestValidationError("experiment identity must be one canonical SHA-256")
        if set(self.code) != {"sha", "branch", "dirty"}:
            raise ManifestValidationError("code provenance requires sha, branch, and dirty")
        if (
            not _is_lower_hex(self.code["sha"], FULL_GIT_SHA_LENGTH)
            or not isinstance(self.code["branch"], str)
            or not self.code["branch"]
            or not isinstance(self.code["dirty"], bool)
        ):
            raise ManifestValidationError("code provenance requires a full Git SHA")
        dataset_id = self.dataset.get("bundle_id", self.dataset.get("id"))
        if not _is_lower_hex(dataset_id, SHA256_HEX_LENGTH):
            raise ManifestValidationError("dataset provenance requires a SHA-256 identity")
        if not self.universe or tuple(sorted(set(self.universe))) != self.universe:
            raise ManifestValidationError("universe must be non-empty, sorted, and unique")
        if set(self.date_range) != {"start", "end"}:
            raise ManifestValidationError("date_range must contain exactly start and end")
        try:
            start_date = date.fromisoformat(self.date_range["start"])
            end_date = date.fromisoformat(self.date_range["end"])
        except (TypeError, ValueError) as exc:
            raise ManifestValidationError("date_range values must be ISO-8601 dates") from exc
        if end_date < start_date:
            raise ManifestValidationError("date_range end precedes start")
        if not self.seeds or any(
            not isinstance(name, str)
            or not name
            or not isinstance(seed, int)
            or isinstance(seed, bool)
            or not 0 <= seed < 2**32
            for name, seed in self.seeds.items()
        ):
            raise ManifestValidationError("seed records require named unsigned 32-bit values")
        if len(set(self.seeds.values())) != len(self.seeds):
            raise ManifestValidationError("named seed values must be unique")
        if (
            set(self.invocation) != {"entrypoint", "arguments"}
            or not isinstance(self.invocation["entrypoint"], str)
            or not self.invocation["entrypoint"]
            or not isinstance(self.invocation["arguments"], Sequence)
            or isinstance(self.invocation["arguments"], (str, bytes))
            or any(not isinstance(value, str) for value in self.invocation["arguments"])
        ):
            raise ManifestValidationError("invocation requires an entrypoint and string arguments")
        if set(self.execution) != {"started_at", "finished_at"}:
            raise ManifestValidationError("execution must contain started_at and finished_at")
        started_at = _parse_utc_timestamp(self.execution["started_at"])
        finished_at = _parse_utc_timestamp(self.execution["finished_at"])
        if finished_at < started_at:
            raise ManifestValidationError("execution finished before it started")
        artifact_paths: set[str] = set()
        for artifact in self.artifacts:
            if set(artifact) != {"path", "bytes", "sha256"}:
                raise ManifestValidationError("artifact records require path, bytes, and sha256")
            path = Path(str(artifact["path"]))
            if path == Path(".") or path.is_absolute() or ".." in path.parts:
                raise ManifestValidationError("artifact paths must be safe and relative")
            if str(path) in artifact_paths:
                raise ManifestValidationError("artifact paths must be unique")
            artifact_paths.add(str(path))
            if (
                not isinstance(artifact["bytes"], int)
                or isinstance(artifact["bytes"], bool)
                or artifact["bytes"] < 0
                or not _is_lower_hex(artifact["sha256"], SHA256_HEX_LENGTH)
            ):
                raise ManifestValidationError("artifact size or SHA-256 is invalid")

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable mapping after revalidating invariants."""
        self.validate()
        return asdict(self)


def write_experiment_manifest(manifest: ExperimentManifest, path: str | Path) -> Path:
    """Atomically publish a validated experiment manifest."""

    destination = Path(path)
    if destination.name != "run_manifest.json" or destination.is_symlink():
        raise ManifestValidationError("experiment manifest path must name run_manifest.json")
    encoded = canonical_json(manifest.to_dict()) + b"\n"
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary_name: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            dir=destination.parent,
            prefix=".run_manifest.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary_name = handle.name
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, destination)
    except BaseException:
        if temporary_name is not None:
            Path(temporary_name).unlink(missing_ok=True)
        raise
    return destination


def refresh_experiment_manifest(
    run_root: str | Path,
    *,
    finished_at: str | None = None,
) -> ExperimentManifest:
    """Refresh a run's artifact inventory without changing semantic identity."""

    root = Path(run_root)
    path = root / "run_manifest.json"
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
        manifest = ExperimentManifest.from_dict(document)
    except (OSError, UnicodeError, json.JSONDecodeError, TypeError) as exc:
        raise ManifestValidationError(f"cannot refresh invalid experiment manifest {path}") from exc
    finish = finished_at or datetime.now(UTC).isoformat().replace("+00:00", "Z")
    refreshed = replace(
        manifest,
        execution={
            "started_at": str(manifest.execution["started_at"]),
            "finished_at": finish,
        },
        artifacts=tuple(inventory_artifacts(root, exclude=("run_manifest.json",))),
    )
    refreshed.validate()
    write_experiment_manifest(refreshed, path)
    return refreshed
