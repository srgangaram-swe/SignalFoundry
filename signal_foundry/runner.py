"""Fixed package-worker policy and typed cross-package composition."""

from __future__ import annotations

import os
import threading
from pathlib import Path
from typing import Any

from pydantic import TypeAdapter, ValidationError

from signal_foundry.boundary import (
    MAX_EVIDENCE_BYTES,
    FoundryError,
    code_identity,
    decode,
    encode,
    private_directory,
)
from signal_foundry.contracts import (
    Catalog,
    Dataset,
    ResearchEvidence,
    ResearchRequest,
    Validation,
)
from signal_foundry.process import execute


class Runner:
    """At most two independent bounded workers, including preflight requests.

    No inherited provider credentials, Python import path, Git authentication,
    shell variables or user site-packages cross this boundary. Source lock/runtime
    installation is an explicit CLI prerequisite, not an automatic network action.
    """

    def __init__(self, root: Path, state: Path, bundles: Path | None = None) -> None:
        self.root = private_directory(root)
        self.bundles = private_directory(bundles) if bundles is not None else None
        private_state = private_directory(state, create=True)
        self.cache = private_directory(private_state / "worker-cache", create=True)
        self._capacity = threading.BoundedSemaphore(2)

    def call(
        self,
        source: str,
        operation: str,
        payload: dict[str, Any],
        *,
        cancel: threading.Event | None = None,
        timeout: float = 150.0,
    ) -> Any:
        if source not in {"alphaforge", "signalattice"} or operation not in {
            "catalog",
            "validate",
            "run",
            "paper-qualification",
        }:
            raise FoundryError("unknown_operation", "Unregistered worker request.")
        if not self._capacity.acquire(blocking=False):
            raise FoundryError(
                "workers_busy", "Worker capacity is occupied; retry later.", 429
            )
        try:
            package = self.root / "packages" / source
            python = package / ".venv/bin/python"
            environment = {
                "PATH": str(python.parent) + os.pathsep + "/usr/bin:/bin",
                "PYTHONHASHSEED": "0",
                "PYTHONNOUSERSITE": "1",
                "OMP_NUM_THREADS": "1",
                "OPENBLAS_NUM_THREADS": "1",
                "MKL_NUM_THREADS": "1",
                "VECLIB_MAXIMUM_THREADS": "1",
                "ARROW_IO_THREADS": "1",
                "MPLCONFIGDIR": str(self.cache),
            }
            identity = code_identity(self.root)
            response = execute(
                [
                    str(python),
                    "-I",
                    str(self.root / "signal_foundry/_worker_entry.py"),
                    source,
                    operation,
                ],
                cwd=package,
                environment=environment,
                payload=encode(payload),
                cancel=cancel if cancel is not None else threading.Event(),
                timeout=timeout,
            )
            decoded = decode(response, MAX_EVIDENCE_BYTES)
            if not isinstance(decoded, dict) or set(decoded) not in (
                {"result"},
                {"error"},
            ):
                raise FoundryError(
                    "worker_contract", "Worker returned an invalid envelope.", 502
                )
            if "error" in decoded:
                error = decoded["error"]
                if not isinstance(error, dict) or set(error) != {
                    "code",
                    "detail",
                    "status",
                }:
                    raise FoundryError(
                        "worker_contract", "Worker returned an invalid error.", 502
                    )
                from signal_foundry.contracts import Problem

                problem = Problem.model_validate(
                    {"code": error["code"], "detail": error["detail"]}
                )
                if (
                    type(error["status"]) is not int
                    or not 400 <= error["status"] <= 599
                ):
                    raise FoundryError(
                        "worker_contract", "Worker returned an invalid status.", 502
                    )
                raise FoundryError(problem.code, problem.detail, error["status"])
            if identity != code_identity(self.root):
                raise FoundryError(
                    "code_changed",
                    "Code changed during execution; restart the service.",
                    409,
                )
            return decoded["result"]
        except ValidationError as exc:
            raise FoundryError(
                "worker_contract", "Worker returned an invalid typed contract.", 502
            ) from exc
        finally:
            self._capacity.release()

    def _location(self) -> dict[str, Any]:
        return {"bundles": str(self.bundles) if self.bundles is not None else None}

    def catalog(self) -> Catalog:
        try:
            catalog = Catalog.model_validate_json(
                encode(self.call("alphaforge", "catalog", {}))
            )
            datasets = TypeAdapter(tuple[Dataset, ...]).validate_json(
                encode(self.call("signalattice", "catalog", self._location()))
            )
            return Catalog(**{**catalog.model_dump(), "datasets": datasets})
        except ValidationError as exc:
            raise FoundryError(
                "catalog_contract", "A catalog failed schema verification.", 502
            ) from exc

    def validate(self, request: ResearchRequest) -> Validation:
        try:
            producer = None
            if request.data.kind == "bundle":
                producer = Dataset.model_validate_json(
                    encode(
                        self.call(
                            "signalattice",
                            "validate",
                            {
                                **self._location(),
                                "bundle_id": request.data.bundle_id,
                            },
                        )
                    )
                )
                if producer.bundle_id != request.data.bundle_id:
                    raise FoundryError(
                        "dataset_identity",
                        "Producer and requested identities disagree.",
                    )
            result = Validation.model_validate_json(
                encode(
                    self.call(
                        "alphaforge",
                        "validate",
                        {
                            **self._location(),
                            "request": request.model_dump(mode="json"),
                        },
                    )
                )
            )
            if result.request_hash != request.digest():
                raise FoundryError(
                    "validation_identity",
                    "Preflight configuration identity disagrees.",
                    502,
                )
            if producer is not None and (
                result.data_identity != "bundle:" + producer.bundle_id
                or result.observations != producer.rows
                or result.symbols != len(producer.symbols)
            ):
                raise FoundryError(
                    "dataset_identity",
                    "Producer and consumer data metadata disagree.",
                    502,
                )
            return result
        except ValidationError as exc:
            raise FoundryError(
                "validation_contract", "Preflight returned an invalid contract.", 502
            ) from exc

    def run(
        self, request: ResearchRequest, cancel: threading.Event
    ) -> ResearchEvidence:
        try:
            result = ResearchEvidence.model_validate_json(
                encode(
                    self.call(
                        "alphaforge",
                        "run",
                        {
                            **self._location(),
                            "request": request.model_dump(mode="json"),
                        },
                        cancel=cancel,
                    )
                )
            )
        except ValidationError as exc:
            raise FoundryError(
                "evidence_contract", "Research returned an invalid contract.", 502
            ) from exc
        if result.request_hash != request.digest() or result.code_hash != code_identity(
            self.root
        ):
            raise FoundryError(
                "evidence_identity",
                "Returned evidence does not bind the requested code/configuration.",
                502,
            )
        return result
