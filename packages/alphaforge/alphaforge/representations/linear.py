"""Deterministic raw and PCA-family representation mechanisms."""

from __future__ import annotations

import math
import time
from typing import Any, Self

import numpy as np
from sklearn.decomposition import PCA, IncrementalPCA

from alphaforge.representations.base import (
    BaseRepresentation,
    EmbeddingCollapseError,
    FloatArray,
    RepresentationBatch,
    RepresentationConfig,
    RepresentationError,
    RepresentationOutput,
    RepresentationSchemaError,
    RepresentationState,
    _readonly_float_array,
    apply_standardizer,
    array_sha256,
    canonicalize_component_signs,
    embedding_variance,
    fit_standardizer,
    invert_standardizer,
    stable_identity,
    subspace_fingerprint,
)

LINEAR_REPRESENTATION_KINDS = frozenset({"raw", "pca", "incremental_pca", "robust_pca"})


def _fit_robust_scaler(values: FloatArray) -> tuple[FloatArray, FloatArray]:
    center = np.median(values, axis=0)
    lower, upper = np.quantile(values, [0.25, 0.75], axis=0)
    scale = upper - lower
    scale = np.where(scale > np.finfo(float).eps, scale, 1.0)
    return (
        _readonly_float_array(center, name="robust center", dimensions=1),
        _readonly_float_array(scale, name="robust scale", dimensions=1),
    )


def _parameter_inventory(*arrays: object) -> tuple[int, int]:
    numeric = [np.asarray(array) for array in arrays]
    return (
        int(sum(array.size for array in numeric)),
        int(sum(array.nbytes for array in numeric)),
    )


class LinearRepresentation(BaseRepresentation):
    """Raw-standardized, PCA, incremental-PCA, or robust-scaling PCA.

    ``robust_pca`` means deterministic median/IQR scaling followed by ordinary
    full-SVD PCA.  It is intentionally not marketed as principal-component
    pursuit: sparse low-rank decomposition is a different estimator with
    different identifiability and compute assumptions.
    """

    def __init__(self, config: RepresentationConfig) -> None:
        super().__init__(config)
        if config.kind not in LINEAR_REPRESENTATION_KINDS:
            raise RepresentationError(f"LinearRepresentation does not support kind {config.kind!r}")
        self._center: FloatArray | None = None
        self._scale: FloatArray | None = None
        self._model: PCA | IncrementalPCA | None = None

    def fit(self, batch: RepresentationBatch) -> Self:
        self._validate_fit_batch(batch)
        wall_start = time.perf_counter()
        cpu_start = time.process_time()
        notes: tuple[str, ...]
        if self.config.kind == "robust_pca":
            center, scale = _fit_robust_scaler(batch.values)
            notes = (
                "median/IQR scaling limits marginal outlier leverage before full-SVD PCA",
                "this is robust scaling, not principal-component pursuit",
            )
        else:
            center, scale = fit_standardizer(batch.values)
            notes = ("mean/std normalization fitted on training rows only",)
        normalized = apply_standardizer(batch.values, center, scale)
        normalized_variance = embedding_variance(normalized)
        if normalized_variance < self.config.min_embedding_variance:
            raise EmbeddingCollapseError(
                f"{self.config.kind} normalized training variance "
                f"{normalized_variance:.3e} is below "
                f"{self.config.min_embedding_variance:.3e}"
            )

        model: PCA | IncrementalPCA | None = None
        learned_arrays: list[object] = [center, scale]
        iterations = 1
        if self.config.kind == "raw":
            embeddings = normalized
            reconstruction = batch.values
            output_features = tuple(f"raw::{name}" for name in batch.feature_names)
            subspace_id: str | None = None
        else:
            if self.config.kind == "incremental_pca":
                model = IncrementalPCA(
                    n_components=self.config.latent_dim,
                    batch_size=self.config.incremental_batch_size,
                )
                maximum_batches = max(1, batch.n_rows // self.config.latent_dim)
                requested_batches = math.ceil(batch.n_rows / self.config.incremental_batch_size)
                chunks = np.array_split(
                    np.arange(batch.n_rows),
                    min(requested_batches, maximum_batches),
                )
                for chunk in chunks:
                    model.partial_fit(normalized[chunk])
                iterations = len(chunks)
                notes += ("incremental updates use deterministic chronological contiguous batches",)
            else:
                model = PCA(
                    n_components=self.config.latent_dim,
                    svd_solver="full",
                    random_state=self.config.seed,
                )
                model.fit(normalized)
            model.components_ = canonicalize_component_signs(model.components_).copy()
            embeddings = _readonly_float_array(
                model.transform(normalized),
                name="linear embeddings",
                dimensions=2,
            )
            reconstructed_normalized = model.inverse_transform(embeddings)
            reconstruction = invert_standardizer(reconstructed_normalized, center, scale)
            output_features = tuple(
                f"{self.config.kind}::{index + 1:03d}" for index in range(self.config.latent_dim)
            )
            subspace_id = subspace_fingerprint(model.components_)
            learned_arrays.extend(
                (
                    model.components_,
                    model.mean_,
                    model.explained_variance_,
                    model.singular_values_,
                )
            )

        variance = embedding_variance(embeddings)
        if variance < self.config.min_embedding_variance:
            raise EmbeddingCollapseError(
                f"{self.config.kind} embedding variance {variance:.3e} is below "
                f"{self.config.min_embedding_variance:.3e}"
            )
        reconstruction_mse = float(np.mean(np.square(reconstruction - batch.values)))
        parameter_count, parameter_bytes = _parameter_inventory(*learned_arrays)
        state_payload: dict[str, Any] = {
            "schema_version": "1.0.0",
            "kind": self.config.kind,
            "config_id": self.config.config_id,
            "fit_batch_id": batch.batch_id,
            "subspace_id": subspace_id,
            "learned_array_digests": [array_sha256(value) for value in learned_arrays],
        }
        state_id = stable_identity(state_payload)
        wall_seconds = time.perf_counter() - wall_start
        cpu_seconds = time.process_time() - cpu_start
        self._center = center
        self._scale = scale
        self._model = model
        self.state_ = RepresentationState(
            schema_version="1.0.0",
            kind=self.config.kind,
            config_id=self.config.config_id,
            state_id=state_id,
            fit_batch_id=batch.batch_id,
            input_features=batch.feature_names,
            output_features=output_features,
            fit_start=batch.start,
            fit_end=batch.end,
            fit_rows=batch.n_rows,
            validation_rows=0,
            iterations=iterations,
            converged=True,
            stopping_reason="closed_form",
            reconstruction_mse=reconstruction_mse,
            embedding_variance=variance,
            parameter_count=parameter_count,
            parameter_bytes=parameter_bytes,
            fit_wall_seconds=wall_seconds,
            fit_cpu_seconds=cpu_seconds,
            device="cpu",
            notes=notes,
        )
        return self

    def _normalized(self, batch: RepresentationBatch) -> FloatArray:
        self._validate_inference_batch(batch)
        if self._center is None or self._scale is None:
            raise RepresentationError("linear normalization state is unavailable")
        return apply_standardizer(batch.values, self._center, self._scale)

    def transform(self, batch: RepresentationBatch) -> RepresentationOutput:
        normalized = self._normalized(batch)
        if self._model is None:
            values = normalized
        else:
            try:
                values = self._model.transform(normalized)
            except ValueError as exc:
                raise RepresentationSchemaError("linear representation rejected input") from exc
        return self._output(values, batch)

    def reconstruct(self, batch: RepresentationBatch) -> FloatArray:
        normalized = self._normalized(batch)
        if self._center is None or self._scale is None:  # pragma: no cover - guarded above
            raise RepresentationError("linear normalization state is unavailable")
        if self._model is None:
            reconstructed = normalized
        else:
            reconstructed = self._model.inverse_transform(self._model.transform(normalized))
        return invert_standardizer(reconstructed, self._center, self._scale)
