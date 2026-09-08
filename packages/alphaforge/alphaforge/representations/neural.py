"""Bounded deterministic CPU autoencoder and contrastive representations.

Torch is an optional dependency and is imported only when a neural
representation is instantiated.  The public fitted state contains hashes and
resource evidence, never model weights.  Augmentation exists only inside
``fit``; ``transform`` and ``reconstruct`` are deterministic.
"""

from __future__ import annotations

import copy
import math
import time
from collections.abc import Iterator
from typing import Any, Self

import numpy as np
import pandas as pd

from alphaforge.representations.base import (
    NEURAL_REPRESENTATION_KINDS,
    BaseRepresentation,
    EmbeddingCollapseError,
    FloatArray,
    RepresentationBatch,
    RepresentationCapabilityError,
    RepresentationConfig,
    RepresentationError,
    RepresentationOutput,
    RepresentationResourceError,
    RepresentationSchemaError,
    RepresentationState,
    _readonly_float_array,
    apply_standardizer,
    array_sha256,
    embedding_variance,
    fit_standardizer,
    invert_standardizer,
    named_seed,
    stable_identity,
)


def _require_torch() -> tuple[Any, Any]:
    try:
        import torch
        from torch import nn
    except ImportError as exc:  # pragma: no cover - exercised in isolated dependency checks
        raise RepresentationError(
            "neural representations require the locked 'torch' optional dependency"
        ) from exc
    return torch, nn


def _complete_date_split(
    batch: RepresentationBatch,
    validation_fraction: float,
) -> tuple[np.ndarray, np.ndarray]:
    dates = pd.DatetimeIndex(batch.dates)
    unique_dates = dates.unique().sort_values()
    if len(unique_dates) < 4:
        raise RepresentationSchemaError(
            "neural representation fitting requires at least four distinct dates"
        )
    n_validation_dates = max(1, int(math.ceil(len(unique_dates) * validation_fraction)))
    if n_validation_dates >= len(unique_dates) - 1:
        raise RepresentationSchemaError("validation split leaves insufficient training dates")
    validation_start = unique_dates[-n_validation_dates]
    validation = np.asarray(dates >= validation_start)
    training = ~validation
    if training.sum() < 2 or validation.sum() < 2:
        raise RepresentationSchemaError("neural fit requires at least two rows per split")
    return training, validation


def build_causal_windows(
    batch: RepresentationBatch,
    normalized_values: object,
    *,
    sequence_length: int,
    max_tensor_bytes: int,
) -> FloatArray:
    """Build left-padded per-symbol windows ending at each current row."""

    values = _readonly_float_array(
        normalized_values,
        name="causal-window values",
        dimensions=2,
    )
    if values.shape != batch.values.shape:
        raise RepresentationSchemaError("causal-window values must align with the batch")
    required_bytes = batch.n_rows * sequence_length * (batch.n_features + 1) * 8
    if required_bytes > max_tensor_bytes:
        raise RepresentationResourceError(
            f"causal windows require {required_bytes} bytes; limit is {max_tensor_bytes}"
        )
    windows = np.zeros(
        (batch.n_rows, sequence_length, batch.n_features + 1),
        dtype=np.float64,
    )
    symbols = np.asarray(batch.symbols, dtype=object)
    for symbol in sorted(set(batch.symbols)):
        rows = np.flatnonzero(symbols == symbol)
        for offset, row in enumerate(rows):
            history = rows[max(0, offset - sequence_length + 1) : offset + 1]
            width = len(history)
            windows[row, -width:, :-1] = values[history]
            windows[row, -width:, -1] = 1.0
    return _readonly_float_array(windows, name="causal windows", dimensions=3)


def augment_causal_windows(
    windows: object,
    config: RepresentationConfig,
    generator: np.random.Generator,
) -> FloatArray:
    """Apply causal jitter, positive scaling, and past-step masking.

    Augmentations never reverse time, mix symbols, or alter the current-step
    availability mask.  They are used only while fitting denoising or
    contrastive objectives.
    """

    values = _readonly_float_array(windows, name="augmentation windows", dimensions=3)
    if values.shape[0] == 0 or values.shape[1] < 2 or values.shape[2] < 2:
        raise RepresentationSchemaError("augmentation requires non-empty sequence windows")
    result = values.copy()
    observed = result[:, :, -1:] > 0.5
    features = result[:, :, :-1]
    if config.augmentation_scale_std > 0.0:
        scale = np.exp(
            generator.normal(
                0.0,
                config.augmentation_scale_std,
                size=(len(result), 1, features.shape[2]),
            )
        )
        features *= scale
    if config.augmentation_jitter_std > 0.0:
        noise = generator.normal(
            0.0,
            config.augmentation_jitter_std,
            size=features.shape,
        )
        features += noise * observed
    mask = generator.random(size=result.shape[:2]) < config.augmentation_mask_probability
    mask[:, -1] = False
    mask &= observed[:, :, 0]
    features[mask] = 0.0
    result[:, :, -1][mask] = 0.0
    result.setflags(write=False)
    return result


def _batches(indices: np.ndarray, size: int) -> Iterator[np.ndarray]:
    for start in range(0, len(indices), size):
        block = indices[start : start + size]
        if len(block):
            yield block


def _float_tensor(torch: Any, values: object) -> Any:
    """Copy read-only NumPy evidence into Torch-owned CPU storage."""

    return torch.tensor(np.asarray(values), dtype=torch.float32, device="cpu")


def _make_dense_network(
    nn: Any,
    input_dim: int,
    hidden_dim: int,
    latent_dim: int,
) -> Any:
    class DenseAutoencoder(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.encoder = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, latent_dim),
            )
            self.decoder = nn.Sequential(
                nn.Linear(latent_dim, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, input_dim),
            )

        def encode(self, values: Any) -> Any:
            return self.encoder(values)

        def decode(self, latent: Any) -> Any:
            return self.decoder(latent)

    return DenseAutoencoder()


def _make_variational_network(
    nn: Any,
    input_dim: int,
    hidden_dim: int,
    latent_dim: int,
) -> Any:
    class VariationalAutoencoder(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.hidden = nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.GELU())
            self.mean = nn.Linear(hidden_dim, latent_dim)
            self.log_variance = nn.Linear(hidden_dim, latent_dim)
            self.decoder = nn.Sequential(
                nn.Linear(latent_dim, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, input_dim),
            )

        def encode_stats(self, values: Any) -> tuple[Any, Any]:
            hidden = self.hidden(values)
            return self.mean(hidden), self.log_variance(hidden).clamp(-12.0, 12.0)

        def decode(self, latent: Any) -> Any:
            return self.decoder(latent)

    return VariationalAutoencoder()


def _make_sequence_network(
    nn: Any,
    input_dim: int,
    hidden_dim: int,
    latent_dim: int,
    *,
    contrastive: bool,
) -> Any:
    class SequenceEncoder(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.recurrent = nn.GRU(input_dim + 1, hidden_dim, batch_first=True)
            self.encoder = nn.Linear(hidden_dim, latent_dim)
            self.decoder = None if contrastive else nn.Linear(latent_dim, input_dim)
            self.projector = (
                nn.Sequential(
                    nn.Linear(latent_dim, latent_dim),
                    nn.GELU(),
                    nn.Linear(latent_dim, latent_dim),
                )
                if contrastive
                else None
            )

        def encode(self, windows: Any) -> Any:
            sequence, _ = self.recurrent(windows)
            return self.encoder(sequence[:, -1, :])

        def decode(self, latent: Any) -> Any:
            if self.decoder is None:
                raise RuntimeError("contrastive encoder has no decoder")
            return self.decoder(latent)

        def project(self, latent: Any) -> Any:
            if self.projector is None:
                return latent
            return self.projector(latent)

    return SequenceEncoder()


def _nt_xent_loss(torch: Any, first: Any, second: Any, temperature: float) -> Any:
    if first.shape != second.shape or first.ndim != 2 or len(first) < 2:
        raise RepresentationSchemaError(
            "contrastive batches require two aligned matrices with at least two rows"
        )
    first = torch.nn.functional.normalize(first, dim=1)
    second = torch.nn.functional.normalize(second, dim=1)
    combined = torch.cat((first, second), dim=0)
    logits = combined @ combined.T / temperature
    diagonal = torch.eye(len(combined), dtype=torch.bool, device=combined.device)
    logits = logits.masked_fill(diagonal, float("-inf"))
    count = len(first)
    labels = torch.cat(
        (
            torch.arange(count, 2 * count, device=combined.device),
            torch.arange(0, count, device=combined.device),
        )
    )
    return torch.nn.functional.cross_entropy(logits, labels)


class NeuralRepresentation(BaseRepresentation):
    """Dense, denoising, sequence, variational, or contrastive encoder."""

    def __init__(self, config: RepresentationConfig) -> None:
        super().__init__(config)
        if config.kind not in NEURAL_REPRESENTATION_KINDS:
            raise RepresentationError(f"NeuralRepresentation does not support kind {config.kind!r}")
        self._center: FloatArray | None = None
        self._scale: FloatArray | None = None
        self._model: Any | None = None

    @property
    def _sequence_kind(self) -> bool:
        return self.config.kind in {"sequence_autoencoder", "contrastive_timeseries"}

    def _build_model(self, input_dim: int) -> tuple[Any, Any]:
        torch, nn = _require_torch()
        torch.manual_seed(named_seed(self.config.seed, f"{self.config.kind}:initialization"))
        if self.config.kind == "variational_autoencoder":
            model = _make_variational_network(
                nn,
                input_dim,
                self.config.hidden_dim,
                self.config.latent_dim,
            )
        elif self._sequence_kind:
            model = _make_sequence_network(
                nn,
                input_dim,
                self.config.hidden_dim,
                self.config.latent_dim,
                contrastive=self.config.kind == "contrastive_timeseries",
            )
        else:
            model = _make_dense_network(
                nn,
                input_dim,
                self.config.hidden_dim,
                self.config.latent_dim,
            )
        return torch, model

    def _network_input(
        self,
        batch: RepresentationBatch,
        normalized: FloatArray,
    ) -> FloatArray:
        if not self._sequence_kind:
            return normalized
        return build_causal_windows(
            batch,
            normalized,
            sequence_length=self.config.sequence_length,
            max_tensor_bytes=self.config.max_tensor_bytes,
        )

    def _autoencoder_loss(
        self,
        torch: Any,
        model: Any,
        network_input: Any,
        clean_target: Any,
        *,
        generator: np.random.Generator,
        training: bool,
    ) -> Any:
        if self.config.kind == "variational_autoencoder":
            mean, log_variance = model.encode_stats(network_input)
            if training:
                epsilon = torch.as_tensor(
                    generator.normal(size=tuple(mean.shape)),
                    dtype=mean.dtype,
                    device=mean.device,
                )
                latent = mean + torch.exp(0.5 * log_variance) * epsilon
            else:
                latent = mean
            reconstruction = model.decode(latent)
            reconstruction_loss = torch.nn.functional.mse_loss(reconstruction, clean_target)
            divergence = -0.5 * torch.mean(1.0 + log_variance - mean.square() - log_variance.exp())
            return reconstruction_loss + self.config.vae_beta * divergence
        if self.config.kind == "sequence_autoencoder":
            latent = model.encode(network_input)
            reconstruction = model.decode(latent)
            return torch.nn.functional.mse_loss(reconstruction, clean_target)
        input_values = network_input
        if training and self.config.kind == "denoising_autoencoder":
            noise = torch.as_tensor(
                generator.normal(
                    0.0,
                    self.config.corruption_std,
                    size=tuple(input_values.shape),
                ),
                dtype=input_values.dtype,
                device=input_values.device,
            )
            input_values = input_values + noise
        latent = model.encode(input_values)
        reconstruction = model.decode(latent)
        return torch.nn.functional.mse_loss(reconstruction, clean_target)

    def _contrastive_loss(
        self,
        torch: Any,
        model: Any,
        windows: FloatArray,
        *,
        generator: np.random.Generator,
    ) -> Any:
        first = augment_causal_windows(windows, self.config, generator)
        second = augment_causal_windows(windows, self.config, generator)
        first_tensor = _float_tensor(torch, first)
        second_tensor = _float_tensor(torch, second)
        first_projection = model.project(model.encode(first_tensor))
        second_projection = model.project(model.encode(second_tensor))
        return _nt_xent_loss(
            torch,
            first_projection,
            second_projection,
            self.config.contrastive_temperature,
        )

    def fit(self, batch: RepresentationBatch) -> Self:
        self._validate_fit_batch(batch)
        training_mask, validation_mask = _complete_date_split(
            batch,
            self.config.validation_fraction,
        )
        center, scale = fit_standardizer(batch.values[training_mask])
        normalized = apply_standardizer(batch.values, center, scale)
        network_input = self._network_input(batch, normalized)
        torch, model = self._build_model(batch.n_features)
        parameter_count = int(sum(parameter.numel() for parameter in model.parameters()))
        parameter_bytes = int(
            sum(parameter.numel() * parameter.element_size() for parameter in model.parameters())
        )
        if parameter_count > self.config.max_parameters:
            raise RepresentationResourceError(
                f"parameter count {parameter_count} exceeds limit {self.config.max_parameters}"
            )
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
        )
        train_indices = np.flatnonzero(training_mask)
        validation_indices = np.flatnonzero(validation_mask)
        training_rng = np.random.default_rng(
            named_seed(self.config.seed, f"{self.config.kind}:training")
        )
        validation_rng_seed = named_seed(
            self.config.seed,
            f"{self.config.kind}:validation",
        )
        previous_deterministic = torch.are_deterministic_algorithms_enabled()
        torch.use_deterministic_algorithms(True)
        wall_start = time.perf_counter()
        cpu_start = time.process_time()
        best_loss = math.inf
        best_state: dict[str, Any] | None = None
        epochs_without_improvement = 0
        epochs_run = 0
        stopping_reason: str = "max_epochs"
        try:
            for epoch in range(self.config.max_epochs):
                epochs_run = epoch + 1
                model.train()
                order = training_rng.permutation(train_indices)
                for block in _batches(order, self.config.batch_size):
                    optimizer.zero_grad(set_to_none=True)
                    if self.config.kind == "contrastive_timeseries":
                        if len(block) < 2:
                            continue
                        loss = self._contrastive_loss(
                            torch,
                            model,
                            network_input[block],
                            generator=training_rng,
                        )
                    else:
                        input_tensor = _float_tensor(torch, network_input[block])
                        clean_target = _float_tensor(torch, normalized[block])
                        loss = self._autoencoder_loss(
                            torch,
                            model,
                            input_tensor,
                            clean_target,
                            generator=training_rng,
                            training=True,
                        )
                    if not bool(torch.isfinite(loss)):
                        raise RepresentationError("neural training produced a non-finite loss")
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                    optimizer.step()

                model.eval()
                with torch.no_grad():
                    if self.config.kind == "contrastive_timeseries":
                        validation_loss = self._contrastive_loss(
                            torch,
                            model,
                            network_input[validation_indices],
                            generator=np.random.default_rng(validation_rng_seed),
                        )
                    else:
                        validation_input = _float_tensor(
                            torch,
                            network_input[validation_indices],
                        )
                        validation_target = _float_tensor(
                            torch,
                            normalized[validation_indices],
                        )
                        validation_loss = self._autoencoder_loss(
                            torch,
                            model,
                            validation_input,
                            validation_target,
                            generator=np.random.default_rng(validation_rng_seed),
                            training=False,
                        )
                value = float(validation_loss.detach().cpu())
                if not math.isfinite(value):
                    raise RepresentationError("neural validation produced a non-finite loss")
                tolerance = 1e-10 * max(1.0, abs(best_loss)) if math.isfinite(best_loss) else 0.0
                if best_state is None or value < best_loss - tolerance:
                    best_loss = value
                    best_state = copy.deepcopy(model.state_dict())
                    epochs_without_improvement = 0
                else:
                    epochs_without_improvement += 1
                    if epochs_without_improvement >= self.config.patience:
                        stopping_reason = "validation_patience"
                        break
        finally:
            torch.use_deterministic_algorithms(previous_deterministic)
        if best_state is None:
            raise RepresentationError("neural representation produced no finite fitted state")
        model.load_state_dict(best_state)
        model.eval()
        wall_seconds = time.perf_counter() - wall_start
        cpu_seconds = time.process_time() - cpu_start

        self._center = center
        self._scale = scale
        self._model = model
        embeddings = self._encode(batch, normalized=normalized)
        variance = embedding_variance(embeddings)
        if variance < self.config.min_embedding_variance:
            self._model = None
            raise EmbeddingCollapseError(
                f"{self.config.kind} embedding variance {variance:.3e} is below "
                f"{self.config.min_embedding_variance:.3e}"
            )
        reconstruction_mse: float | None
        if self.config.kind == "contrastive_timeseries":
            reconstruction_mse = None
        else:
            reconstructed = self._reconstruct_array(batch, normalized=normalized)
            reconstruction_mse = float(np.mean(np.square(reconstructed - batch.values)))

        learned_arrays: list[object] = [center, scale]
        for name, tensor in sorted(model.state_dict().items()):
            del name
            learned_arrays.append(tensor.detach().cpu().numpy())
        state_id = stable_identity(
            {
                "schema_version": "1.0.0",
                "kind": self.config.kind,
                "config_id": self.config.config_id,
                "fit_batch_id": batch.batch_id,
                "learned_array_digests": [array_sha256(value) for value in learned_arrays],
            }
        )
        notes = [
            "normalization was fitted on the inner-training dates only",
            "validation dates were used only for bounded early stopping",
            "CPU deterministic algorithms and named random streams were used",
            "learned weights are retained in memory but omitted from public evidence",
        ]
        if self.config.kind == "denoising_autoencoder":
            notes.append("Gaussian corruption was applied to training inputs only")
        if self.config.kind == "variational_autoencoder":
            notes.append("deterministic posterior means are emitted at inference")
        if self._sequence_kind:
            notes.append("per-symbol windows are left padded and strictly causal")
        if self.config.kind == "contrastive_timeseries":
            notes.append(
                "positive scaling, jitter, and past-step masking were applied during fit only"
            )
        output_features = tuple(
            f"{self.config.kind}::{index + 1:03d}" for index in range(self.config.latent_dim)
        )
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
            validation_rows=int(validation_mask.sum()),
            iterations=epochs_run,
            converged=stopping_reason == "validation_patience",
            stopping_reason=stopping_reason,  # type: ignore[arg-type]
            reconstruction_mse=reconstruction_mse,
            embedding_variance=variance,
            parameter_count=parameter_count,
            parameter_bytes=parameter_bytes,
            fit_wall_seconds=wall_seconds,
            fit_cpu_seconds=cpu_seconds,
            device="cpu",
            notes=tuple(notes),
        )
        return self

    def _normalized(self, batch: RepresentationBatch) -> FloatArray:
        self._validate_inference_batch(batch)
        if self._center is None or self._scale is None:
            raise RepresentationError("neural normalization state is unavailable")
        return apply_standardizer(batch.values, self._center, self._scale)

    def _encode(
        self,
        batch: RepresentationBatch,
        *,
        normalized: FloatArray | None = None,
    ) -> FloatArray:
        if self._model is None:
            raise RepresentationError("neural model state is unavailable")
        torch, _ = _require_torch()
        values = self._normalized(batch) if normalized is None else normalized
        network_input = self._network_input(batch, values)
        outputs: list[np.ndarray] = []
        self._model.eval()
        with torch.no_grad():
            for block in _batches(np.arange(batch.n_rows), self.config.batch_size):
                tensor = _float_tensor(torch, network_input[block])
                if self.config.kind == "variational_autoencoder":
                    latent, _ = self._model.encode_stats(tensor)
                else:
                    latent = self._model.encode(tensor)
                outputs.append(latent.detach().cpu().numpy().astype(np.float64))
        return _readonly_float_array(
            np.concatenate(outputs, axis=0),
            name="neural embeddings",
            dimensions=2,
        )

    def transform(self, batch: RepresentationBatch) -> RepresentationOutput:
        self._validate_inference_batch(batch)
        return self._output(self._encode(batch), batch)

    def _reconstruct_array(
        self,
        batch: RepresentationBatch,
        *,
        normalized: FloatArray | None = None,
    ) -> FloatArray:
        if self.config.kind == "contrastive_timeseries":
            raise RepresentationCapabilityError(
                "contrastive_timeseries has no reconstruction decoder"
            )
        if self._model is None or self._center is None or self._scale is None:
            raise RepresentationError("neural reconstruction state is unavailable")
        torch, _ = _require_torch()
        values = self._normalized(batch) if normalized is None else normalized
        network_input = self._network_input(batch, values)
        blocks: list[np.ndarray] = []
        self._model.eval()
        with torch.no_grad():
            for block in _batches(np.arange(batch.n_rows), self.config.batch_size):
                tensor = _float_tensor(torch, network_input[block])
                if self.config.kind == "variational_autoencoder":
                    latent, _ = self._model.encode_stats(tensor)
                else:
                    latent = self._model.encode(tensor)
                reconstructed = self._model.decode(latent)
                blocks.append(reconstructed.detach().cpu().numpy().astype(np.float64))
        normalized_reconstruction = np.concatenate(blocks, axis=0)
        return invert_standardizer(
            normalized_reconstruction,
            self._center,
            self._scale,
        )

    def reconstruct(self, batch: RepresentationBatch) -> FloatArray:
        self._validate_inference_batch(batch)
        return self._reconstruct_array(batch)
