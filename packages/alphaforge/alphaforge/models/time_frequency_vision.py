"""Governed image models for causal financial time-frequency tensors.

Signalattice defines the upstream tensor as
``[sample, channel, frequency, within-window time]`` with a per-channel
observability mask.  This module consumes that shape without reconstructing the
transform.  It owns only downstream validation, train-fit normalization,
meaning-preserving robustness perturbations, bounded neural training, and the
machine-checkable architecture progression.

The progression is intentionally fail-closed:

1. a small CNN may run without prior evidence;
2. a ResNet requires a verified small-CNN gate record; and
3. a Vision Transformer requires a verified ResNet gate record.

Gate metrics come from a chronological validation interval.  The final test
interval is not accepted by the gate API and therefore cannot authorize a more
complex candidate.
"""

from __future__ import annotations

import hashlib
import json
import math
import time
from dataclasses import asdict, dataclass, field
from typing import Any, Literal

import numpy as np
import pandas as pd

from alphaforge.models.base import ModelError
from alphaforge.models.deep_sequence import DeviceName, resolve_sequence_device
from alphaforge.models.torch_models import TORCH_AVAILABLE, _require_torch

if TORCH_AVAILABLE:
    import torch
    from torch import nn

VisionArchitecture = Literal["small_cnn", "resnet", "vit"]
_ARCHITECTURES: tuple[VisionArchitecture, ...] = ("small_cnn", "resnet", "vit")


@dataclass(frozen=True)
class TimeFrequencyBatch:
    """Validated tensor, alignment, baseline features, and response.

    The batch is an in-memory research boundary.  Licensed observations and
    model-ready tensors remain local; only aggregate evidence may be published.
    """

    values: np.ndarray
    observed_mask: np.ndarray
    dates: np.ndarray
    symbols: np.ndarray
    target: np.ndarray
    time_features: np.ndarray
    spectral_features: np.ndarray
    channels: tuple[str, ...]
    frequency_values: tuple[float, ...]
    representation: Literal["spectrogram", "scalogram"]
    max_tensor_bytes: int = field(default=2_147_483_648, repr=False)

    def __post_init__(self) -> None:
        values = np.asarray(self.values)
        mask = np.asarray(self.observed_mask)
        if values.ndim != 4:
            raise ValueError(
                "time-frequency values must have shape [sample, channel, frequency, time]"
            )
        samples, channels, frequencies, time_steps = values.shape
        if min(samples, channels, frequencies, time_steps) < 1:
            raise ValueError("every tensor axis must be non-empty")
        if values.dtype not in {np.dtype(np.float32), np.dtype(np.float64)}:
            raise ValueError("time-frequency values must use float32 or float64")
        if values.nbytes + mask.nbytes > self.max_tensor_bytes:
            raise ValueError("time-frequency arrays exceed max_tensor_bytes")
        if mask.dtype != np.bool_ or mask.shape != (samples, channels):
            raise ValueError("observed_mask must be boolean with shape [sample, channel]")
        if len(self.channels) != channels or len(set(self.channels)) != channels:
            raise ValueError("channel names must be unique and align with the channel axis")
        if any(not name or not name.replace("_", "").isalnum() for name in self.channels):
            raise ValueError("channel names must be non-empty alphanumeric identifiers")
        frequency = np.asarray(self.frequency_values, dtype=np.float64)
        if frequency.shape != (frequencies,) or not np.isfinite(frequency).all():
            raise ValueError("frequency_values must be finite and align with the frequency axis")
        if not np.all(np.diff(frequency) > 0.0):
            raise ValueError("frequency_values must be strictly ascending")
        if self.representation not in {"spectrogram", "scalogram"}:
            raise ValueError("representation must be spectrogram or scalogram")

        dates = pd.to_datetime(np.asarray(self.dates), errors="raise")
        symbols = np.asarray(self.symbols, dtype=object)
        target = np.asarray(self.target, dtype=np.float64)
        if dates.shape != (samples,) or symbols.shape != (samples,) or target.shape != (samples,):
            raise ValueError("dates, symbols, and target must each align one-to-one with samples")
        if any(not isinstance(value, str) or not value for value in symbols):
            raise ValueError("symbols must be non-empty strings")
        if not np.isfinite(target).all():
            raise ValueError("targets must be finite")
        alignment = pd.MultiIndex.from_arrays([dates, symbols])
        if alignment.has_duplicates:
            raise ValueError("sample alignment contains duplicate (date, symbol) rows")

        for name, matrix in (
            ("time_features", self.time_features),
            ("spectral_features", self.spectral_features),
        ):
            array = np.asarray(matrix, dtype=np.float64)
            if array.ndim != 2 or array.shape[0] != samples or array.shape[1] < 1:
                raise ValueError(f"{name} must be a non-empty two-dimensional sample matrix")
            if not np.isfinite(array).all():
                raise ValueError(f"{name} must contain only finite values")

        observed_values = np.broadcast_to(mask[:, :, None, None], values.shape)
        if not np.isfinite(values[observed_values]).all():
            raise ValueError("observed tensor surfaces must be finite")
        if np.isinf(values).any():
            raise ValueError("tensor values may not contain infinity")
        if not mask.all(axis=1).any():
            raise ValueError("at least one sample must have all channels observed")

    @property
    def complete_samples(self) -> np.ndarray:
        """Return the mask for samples whose every channel is observed."""
        return np.asarray(self.observed_mask).all(axis=1)

    def take(self, rows: np.ndarray) -> TimeFrequencyBatch:
        """Return a validated row subset without changing tensor semantics."""
        selector = np.asarray(rows)
        if selector.dtype != np.bool_ or selector.shape != (len(self.target),):
            raise ValueError("rows must be a boolean mask aligned with samples")
        if not selector.any():
            raise ValueError("cannot create an empty time-frequency batch")
        return TimeFrequencyBatch(
            values=np.asarray(self.values)[selector],
            observed_mask=np.asarray(self.observed_mask)[selector],
            dates=np.asarray(self.dates)[selector],
            symbols=np.asarray(self.symbols)[selector],
            target=np.asarray(self.target)[selector],
            time_features=np.asarray(self.time_features)[selector],
            spectral_features=np.asarray(self.spectral_features)[selector],
            channels=self.channels,
            frequency_values=self.frequency_values,
            representation=self.representation,
            max_tensor_bytes=self.max_tensor_bytes,
        )


@dataclass(frozen=True)
class VisionAugmentationConfig:
    """Train-only robustness perturbations that preserve axis semantics."""

    probability: float = 0.35
    log_amplitude_std: float = 0.05
    max_frequency_mask_bins: int = 1

    def __post_init__(self) -> None:
        if not 0.0 <= self.probability <= 1.0:
            raise ValueError("augmentation probability must be in [0, 1]")
        if not math.isfinite(self.log_amplitude_std) or not 0.0 <= self.log_amplitude_std <= 0.5:
            raise ValueError("log_amplitude_std must be finite and in [0, 0.5]")
        if not 0 <= self.max_frequency_mask_bins <= 16:
            raise ValueError("max_frequency_mask_bins must be in [0, 16]")


@dataclass(frozen=True)
class VisionTrainingConfig:
    """Shared optimization, architecture, and resource policy."""

    width: int = 16
    residual_blocks: int = 2
    vit_layers: int = 2
    vit_heads: int = 4
    patch_frequency: int = 2
    patch_time: int = 1
    dropout: float = 0.0
    learning_rate: float = 1e-3
    weight_decay: float = 1e-5
    max_epochs: int = 20
    patience: int = 4
    batch_size: int = 128
    gradient_clip: float = 1.0
    max_parameters: int = 2_000_000
    max_tensor_bytes: int = 2_147_483_648
    seed: int = 42
    device: DeviceName = "auto"
    augmentation: VisionAugmentationConfig = field(default_factory=VisionAugmentationConfig)

    def __post_init__(self) -> None:
        if not 4 <= self.width <= 256:
            raise ValueError("width must be in [4, 256]")
        if not 1 <= self.residual_blocks <= 8:
            raise ValueError("residual_blocks must be in [1, 8]")
        if not 1 <= self.vit_layers <= 8:
            raise ValueError("vit_layers must be in [1, 8]")
        if not 1 <= self.vit_heads <= 16 or self.width % self.vit_heads:
            raise ValueError("vit_heads must divide width and be in [1, 16]")
        if not 1 <= self.patch_frequency <= 32 or not 1 <= self.patch_time <= 32:
            raise ValueError("patch dimensions must be in [1, 32]")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")
        for name, value in (
            ("learning_rate", self.learning_rate),
            ("weight_decay", self.weight_decay),
            ("gradient_clip", self.gradient_clip),
        ):
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and positive")
        if not 1 <= self.max_epochs <= 10_000:
            raise ValueError("max_epochs must be in [1, 10000]")
        if not 1 <= self.patience <= self.max_epochs:
            raise ValueError("patience must be in [1, max_epochs]")
        if not 1 <= self.batch_size <= 65_536:
            raise ValueError("batch_size must be in [1, 65536]")
        if not 1 <= self.max_parameters <= 100_000_000:
            raise ValueError("max_parameters must be in [1, 100000000]")
        if not 1_048_576 <= self.max_tensor_bytes <= 8_589_934_592:
            raise ValueError("max_tensor_bytes must be in [1048576, 8589934592]")
        if self.seed < 0:
            raise ValueError("seed must be non-negative")
        if self.device not in {"auto", "cpu", "cuda", "mps"}:
            raise ValueError("device must be one of auto, cpu, cuda, or mps")


@dataclass(frozen=True)
class ProgressionGatePolicy:
    """Frozen validation thresholds for one architecture transition."""

    minimum_observations: int = 64
    minimum_dates: int = 8
    minimum_rank_ic: float = 0.0
    minimum_incremental_rank_ic: float = 0.0
    maximum_rmse_ratio: float = 1.05

    def __post_init__(self) -> None:
        if not 8 <= self.minimum_observations <= 10_000_000:
            raise ValueError("minimum_observations must be in [8, 10000000]")
        if not 2 <= self.minimum_dates <= 100_000:
            raise ValueError("minimum_dates must be in [2, 100000]")
        for name, value in (
            ("minimum_rank_ic", self.minimum_rank_ic),
            ("minimum_incremental_rank_ic", self.minimum_incremental_rank_ic),
        ):
            if not math.isfinite(value) or not -1.0 <= value <= 1.0:
                raise ValueError(f"{name} must be finite and in [-1, 1]")
        if not math.isfinite(self.maximum_rmse_ratio) or not 0.1 <= self.maximum_rmse_ratio <= 10.0:
            raise ValueError("maximum_rmse_ratio must be finite and in [0.1, 10]")

    @property
    def identity(self) -> str:
        """Return the canonical SHA-256 identity of this policy."""
        payload = json.dumps(asdict(self), sort_keys=True, separators=(",", ":"), allow_nan=False)
        return hashlib.sha256(payload.encode()).hexdigest()


@dataclass(frozen=True)
class PredictionMetrics:
    """Predictive evidence on one named chronological partition."""

    partition: Literal["validation", "test"]
    observations: int
    dates: int
    rmse: float
    mae: float
    rank_ic: float

    def __post_init__(self) -> None:
        if self.observations < 1 or self.dates < 1:
            raise ValueError("prediction metrics require non-empty evidence")
        if any(not math.isfinite(value) for value in (self.rmse, self.mae, self.rank_ic)):
            raise ValueError("prediction metrics must be finite")


@dataclass(frozen=True)
class ProgressionGateEvidence:
    """Tamper-evident validation record authorizing at most one next stage."""

    candidate: VisionArchitecture
    baseline: str
    policy_id: str
    candidate_metrics: PredictionMetrics
    baseline_metrics: PredictionMetrics
    checks: tuple[tuple[str, bool], ...]
    passed: bool
    evidence_sha256: str

    def payload(self) -> dict[str, Any]:
        """Return the canonical payload covered by ``evidence_sha256``."""
        return {
            "candidate": self.candidate,
            "baseline": self.baseline,
            "policy_id": self.policy_id,
            "candidate_metrics": asdict(self.candidate_metrics),
            "baseline_metrics": asdict(self.baseline_metrics),
            "checks": [[name, passed] for name, passed in self.checks],
            "passed": self.passed,
        }

    def verify(self) -> None:
        """Reject tampering, test-set gates, or inconsistent check summaries."""
        if self.candidate_metrics.partition != "validation":
            raise ModelError("architecture gates may use validation evidence only")
        if self.baseline_metrics.partition != "validation":
            raise ModelError("architecture gates may use validation evidence only")
        expected = hashlib.sha256(
            json.dumps(
                self.payload(),
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            ).encode()
        ).hexdigest()
        if expected != self.evidence_sha256:
            raise ModelError("progression gate evidence digest mismatch")
        if self.passed != all(result for _, result in self.checks):
            raise ModelError("progression gate summary is inconsistent with its checks")


def evaluate_progression_gate(
    *,
    candidate: VisionArchitecture,
    baseline: str,
    candidate_metrics: PredictionMetrics,
    baseline_metrics: PredictionMetrics,
    policy: ProgressionGatePolicy,
) -> ProgressionGateEvidence:
    """Evaluate and sign a validation-only architecture progression decision."""
    if candidate not in {"small_cnn", "resnet"}:
        raise ValueError("only small_cnn and resnet can authorize a next architecture")
    if candidate_metrics.partition != "validation" or baseline_metrics.partition != "validation":
        raise ValueError("architecture gates may use validation evidence only")
    if candidate_metrics.observations != baseline_metrics.observations:
        raise ValueError("candidate and baseline gate observations must be matched")
    checks = (
        ("minimum_observations", candidate_metrics.observations >= policy.minimum_observations),
        ("minimum_dates", candidate_metrics.dates >= policy.minimum_dates),
        ("minimum_rank_ic", candidate_metrics.rank_ic >= policy.minimum_rank_ic),
        (
            "minimum_incremental_rank_ic",
            candidate_metrics.rank_ic - baseline_metrics.rank_ic
            >= policy.minimum_incremental_rank_ic,
        ),
        (
            "maximum_rmse_ratio",
            candidate_metrics.rmse <= baseline_metrics.rmse * policy.maximum_rmse_ratio,
        ),
    )
    unsigned = {
        "candidate": candidate,
        "baseline": baseline,
        "policy_id": policy.identity,
        "candidate_metrics": asdict(candidate_metrics),
        "baseline_metrics": asdict(baseline_metrics),
        "checks": [[name, passed] for name, passed in checks],
        "passed": all(result for _, result in checks),
    }
    digest = hashlib.sha256(
        json.dumps(unsigned, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()
    evidence = ProgressionGateEvidence(
        candidate=candidate,
        baseline=baseline,
        policy_id=policy.identity,
        candidate_metrics=candidate_metrics,
        baseline_metrics=baseline_metrics,
        checks=checks,
        passed=all(result for _, result in checks),
        evidence_sha256=digest,
    )
    evidence.verify()
    return evidence


def authorize_architecture(
    architecture: VisionArchitecture,
    *,
    policy: ProgressionGatePolicy,
    prior_gate: ProgressionGateEvidence | None,
) -> None:
    """Fail closed unless the exact preceding architecture passed its gate."""
    if architecture == "small_cnn":
        if prior_gate is not None:
            raise ModelError(
                "small_cnn is the mandatory first architecture and takes no prior gate"
            )
        return
    if prior_gate is None:
        raise ModelError(f"{architecture} requires verified prior gate evidence")
    prior_gate.verify()
    expected = "small_cnn" if architecture == "resnet" else "resnet"
    if prior_gate.candidate != expected:
        raise ModelError(f"{architecture} requires a passed {expected} gate")
    if prior_gate.policy_id != policy.identity:
        raise ModelError("progression gate policy does not match the frozen training policy")
    if not prior_gate.passed:
        raise ModelError(f"{architecture} is blocked because the {expected} gate failed")


def augment_time_frequency(
    values: np.ndarray,
    config: VisionAugmentationConfig,
    *,
    seed: int,
) -> np.ndarray:
    """Apply deterministic train-only amplitude and frequency perturbations.

    Positive multiplicative amplitude changes preserve sign/order semantics.
    A contiguous frequency mask models a missing or unreliable band without
    permuting frequency, time, channels, samples, or labels.
    """
    array = np.asarray(values, dtype=np.float32)
    if array.ndim != 4 or not np.isfinite(array).all():
        raise ValueError("augmentation requires a finite four-dimensional tensor")
    output = array.copy()
    if config.probability == 0.0:
        return output
    generator = np.random.default_rng(seed)
    selected = generator.random(len(output)) < config.probability
    if config.log_amplitude_std > 0.0:
        scale = generator.lognormal(
            mean=0.0,
            sigma=config.log_amplitude_std,
            size=(len(output), output.shape[1], 1, 1),
        ).astype(np.float32)
        output[selected] *= scale[selected]
    width = min(config.max_frequency_mask_bins, output.shape[2] - 1)
    if width > 0:
        for sample in np.flatnonzero(selected):
            actual = int(generator.integers(1, width + 1))
            start = int(generator.integers(0, output.shape[2] - actual + 1))
            output[sample, :, start : start + actual, :] = 0.0
    return output


if TORCH_AVAILABLE:

    class _SmallCNN(nn.Module):
        def __init__(self, channels: int, width: int, dropout: float):
            super().__init__()
            self.network = nn.Sequential(
                nn.Conv2d(channels, width, 3, padding=1),
                nn.GELU(),
                nn.Conv2d(width, width, 3, padding=1),
                nn.GELU(),
                nn.AdaptiveAvgPool2d((4, 4)),
                nn.Flatten(),
                nn.Dropout(dropout),
                nn.Linear(width * 16, 1),
            )

        def forward(self, values: torch.Tensor) -> torch.Tensor:
            return self.network(values).squeeze(-1)

    class _ResidualBlock(nn.Module):
        def __init__(self, width: int, dropout: float):
            super().__init__()
            self.network = nn.Sequential(
                nn.Conv2d(width, width, 3, padding=1, bias=False),
                nn.BatchNorm2d(width),
                nn.GELU(),
                nn.Dropout2d(dropout),
                nn.Conv2d(width, width, 3, padding=1, bias=False),
                nn.BatchNorm2d(width),
            )

        def forward(self, values: torch.Tensor) -> torch.Tensor:
            return nn.functional.gelu(values + self.network(values))

    class _ResNet(nn.Module):
        def __init__(self, channels: int, config: VisionTrainingConfig):
            super().__init__()
            self.stem = nn.Conv2d(channels, config.width, 3, padding=1)
            self.blocks = nn.Sequential(
                *(
                    _ResidualBlock(config.width, config.dropout)
                    for _ in range(config.residual_blocks)
                )
            )
            self.head = nn.Sequential(
                nn.AdaptiveAvgPool2d((4, 4)),
                nn.Flatten(),
                nn.Linear(config.width * 16, 1),
            )

        def forward(self, values: torch.Tensor) -> torch.Tensor:
            return self.head(self.blocks(nn.functional.gelu(self.stem(values)))).squeeze(-1)

    class _VisionTransformer(nn.Module):
        def __init__(
            self,
            channels: int,
            frequency_bins: int,
            time_steps: int,
            config: VisionTrainingConfig,
        ):
            super().__init__()
            if frequency_bins < config.patch_frequency or time_steps < config.patch_time:
                raise ValueError("ViT patch dimensions exceed the tensor surface")
            self.patch = nn.Conv2d(
                channels,
                config.width,
                kernel_size=(config.patch_frequency, config.patch_time),
                stride=(config.patch_frequency, config.patch_time),
            )
            patch_count = (frequency_bins // config.patch_frequency) * (
                time_steps // config.patch_time
            )
            self.position = nn.Parameter(torch.zeros(1, patch_count, config.width))
            layer = nn.TransformerEncoderLayer(
                d_model=config.width,
                nhead=config.vit_heads,
                dim_feedforward=config.width * 2,
                dropout=config.dropout,
                activation="gelu",
                batch_first=True,
            )
            self.encoder = nn.TransformerEncoder(layer, config.vit_layers)
            self.head = nn.Sequential(nn.LayerNorm(config.width), nn.Linear(config.width, 1))

        def forward(self, values: torch.Tensor) -> torch.Tensor:
            patches = self.patch(values).flatten(2).transpose(1, 2)
            encoded = self.encoder(patches + self.position[:, : patches.shape[1]])
            return self.head(encoded.mean(dim=1)).squeeze(-1)


@dataclass(frozen=True)
class VisionResourceEvidence:
    """Measured training and convergence evidence."""

    architecture: VisionArchitecture
    device: str
    parameter_count: int
    parameter_bytes: int
    fit_wall_seconds: float
    fit_cpu_seconds: float
    epochs_completed: int
    best_epoch: int
    best_validation_loss: float
    stopped_early: bool
    training_loss: tuple[float, ...] = field(repr=False)
    validation_loss: tuple[float, ...] = field(repr=False)

    def to_dict(self) -> dict[str, Any]:
        """Return deterministic JSON-compatible evidence."""
        return asdict(self)


class ControlledVisionModel:
    """Bounded time-frequency regressor with train-only fitted state."""

    def __init__(
        self,
        architecture: VisionArchitecture,
        config: VisionTrainingConfig,
        *,
        policy: ProgressionGatePolicy,
        prior_gate: ProgressionGateEvidence | None = None,
    ):
        _require_torch()
        if architecture not in _ARCHITECTURES:
            raise ValueError(f"architecture must be one of {_ARCHITECTURES}")
        authorize_architecture(architecture, policy=policy, prior_gate=prior_gate)
        self.architecture = architecture
        self.config = config
        self.policy = policy
        self.net: nn.Module | None = None
        self.mean_: np.ndarray | None = None
        self.scale_: np.ndarray | None = None
        self.target_mean_: float | None = None
        self.target_scale_: float | None = None
        self.evidence_: VisionResourceEvidence | None = None

    def _network(self, shape: tuple[int, int, int, int]) -> nn.Module:
        _, channels, frequencies, time_steps = shape
        if self.architecture == "small_cnn":
            return _SmallCNN(channels, self.config.width, self.config.dropout)
        if self.architecture == "resnet":
            return _ResNet(channels, self.config)
        return _VisionTransformer(channels, frequencies, time_steps, self.config)

    def fit(
        self,
        train_values: np.ndarray,
        train_target: np.ndarray,
        validation_values: np.ndarray,
        validation_target: np.ndarray,
    ) -> ControlledVisionModel:
        """Fit on training data and use validation only for early stopping."""
        train = _finite_tensor(train_values, "train_values", self.config.max_tensor_bytes)
        validation = _finite_tensor(
            validation_values, "validation_values", self.config.max_tensor_bytes
        )
        if train.shape[1:] != validation.shape[1:]:
            raise ValueError("training and validation tensor shapes are incompatible")
        y_train = _finite_target(train_target, len(train), "train_target")
        y_validation = _finite_target(validation_target, len(validation), "validation_target")
        if len(train) < 2 or len(validation) < 2:
            raise ValueError("training and validation partitions need at least two samples")

        self.mean_ = train.mean(axis=0, dtype=np.float64)
        scale = train.std(axis=0, dtype=np.float64)
        self.scale_ = np.where(scale < 1e-12, 1.0, scale)
        self.target_mean_ = float(y_train.mean())
        target_scale = float(y_train.std())
        self.target_scale_ = 1.0 if target_scale < 1e-12 else target_scale
        normalized_train = np.asarray((train - self.mean_) / self.scale_, dtype=np.float32)
        normalized_validation = np.asarray(
            (validation - self.mean_) / self.scale_,
            dtype=np.float32,
        )
        normalized_train_target = np.asarray(
            (y_train - self.target_mean_) / self.target_scale_,
            dtype=np.float32,
        )
        normalized_validation_target = np.asarray(
            (y_validation - self.target_mean_) / self.target_scale_,
            dtype=np.float32,
        )
        normalized_train = augment_time_frequency(
            normalized_train,
            self.config.augmentation,
            seed=self.config.seed,
        )

        device = resolve_sequence_device(self.config.device)
        torch.manual_seed(self.config.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.config.seed)
        network = self._network(train.shape).to(device)
        parameter_count = sum(parameter.numel() for parameter in network.parameters())
        if parameter_count > self.config.max_parameters:
            raise ModelError(
                f"{self.architecture} has {parameter_count} parameters; "
                f"max_parameters={self.config.max_parameters}"
            )
        optimizer = torch.optim.AdamW(
            network.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
        )
        loss_function = nn.HuberLoss()
        train_tensor = torch.as_tensor(normalized_train, dtype=torch.float32, device=device)
        train_y = torch.as_tensor(
            normalized_train_target,
            dtype=torch.float32,
            device=device,
        )
        validation_tensor = torch.as_tensor(
            normalized_validation, dtype=torch.float32, device=device
        )
        validation_y = torch.as_tensor(
            normalized_validation_target,
            dtype=torch.float32,
            device=device,
        )
        generator = torch.Generator(device="cpu").manual_seed(self.config.seed)

        best_state: dict[str, torch.Tensor] | None = None
        best_loss = math.inf
        best_epoch = 0
        stale = 0
        train_losses: list[float] = []
        validation_losses: list[float] = []
        start_wall = time.perf_counter()
        start_cpu = time.process_time()
        for epoch in range(1, self.config.max_epochs + 1):
            network.train()
            permutation = torch.randperm(len(train_tensor), generator=generator)
            batch_losses: list[float] = []
            for start in range(0, len(permutation), self.config.batch_size):
                rows = permutation[start : start + self.config.batch_size].to(device)
                optimizer.zero_grad(set_to_none=True)
                loss = loss_function(network(train_tensor[rows]), train_y[rows])
                if not torch.isfinite(loss):
                    raise ModelError(f"{self.architecture} produced non-finite training loss")
                loss.backward()
                nn.utils.clip_grad_norm_(network.parameters(), self.config.gradient_clip)
                optimizer.step()
                batch_losses.append(float(loss.detach().cpu()))
            network.eval()
            with torch.no_grad():
                validation_loss = float(
                    loss_function(network(validation_tensor), validation_y).detach().cpu()
                )
            train_losses.append(float(np.mean(batch_losses)))
            validation_losses.append(validation_loss)
            if validation_loss < best_loss - 1e-12:
                best_loss = validation_loss
                best_epoch = epoch
                best_state = {
                    name: parameter.detach().cpu().clone()
                    for name, parameter in network.state_dict().items()
                }
                stale = 0
            else:
                stale += 1
            if stale >= self.config.patience:
                break
        if best_state is None or not math.isfinite(best_loss):
            raise ModelError(f"{self.architecture} did not produce a finite validation state")
        network.load_state_dict(best_state)
        self.net = network
        self.evidence_ = VisionResourceEvidence(
            architecture=self.architecture,
            device=device,
            parameter_count=parameter_count,
            parameter_bytes=sum(
                parameter.numel() * parameter.element_size() for parameter in network.parameters()
            ),
            fit_wall_seconds=time.perf_counter() - start_wall,
            fit_cpu_seconds=time.process_time() - start_cpu,
            epochs_completed=len(validation_losses),
            best_epoch=best_epoch,
            best_validation_loss=best_loss,
            stopped_early=len(validation_losses) < self.config.max_epochs,
            training_loss=tuple(train_losses),
            validation_loss=tuple(validation_losses),
        )
        return self

    def predict(self, values: np.ndarray) -> np.ndarray:
        """Predict with the train-fitted normalization and best checkpoint."""
        if (
            self.net is None
            or self.mean_ is None
            or self.scale_ is None
            or self.target_mean_ is None
            or self.target_scale_ is None
        ):
            raise ModelError("vision model must be fit before prediction")
        array = _finite_tensor(values, "values", self.config.max_tensor_bytes)
        if array.shape[1:] != self.mean_.shape:
            raise ValueError("prediction tensor shape differs from the fitted tensor shape")
        normalized = np.asarray((array - self.mean_) / self.scale_, dtype=np.float32)
        device = next(self.net.parameters()).device
        predictions: list[np.ndarray] = []
        self.net.eval()
        with torch.no_grad():
            for start in range(0, len(normalized), self.config.batch_size):
                batch = torch.as_tensor(
                    normalized[start : start + self.config.batch_size],
                    dtype=torch.float32,
                    device=device,
                )
                predictions.append(self.net(batch).detach().cpu().numpy())
        output = np.concatenate(predictions).astype(np.float64, copy=False)
        output = output * self.target_scale_ + self.target_mean_
        if output.shape != (len(array),) or not np.isfinite(output).all():
            raise ModelError("vision model produced invalid predictions")
        return output

    def resource_evidence(self) -> VisionResourceEvidence:
        """Return measured evidence after a successful fit."""
        if self.evidence_ is None:
            raise ModelError("vision model must be fit before resource evidence is available")
        return self.evidence_


def prediction_metrics(
    target: np.ndarray,
    prediction: np.ndarray,
    dates: np.ndarray,
    *,
    partition: Literal["validation", "test"],
) -> PredictionMetrics:
    """Compute matched aggregate and cross-sectional prediction diagnostics."""
    actual = _finite_target(target, len(target), "target")
    estimated = _finite_target(prediction, len(actual), "prediction")
    date_values = pd.to_datetime(np.asarray(dates), errors="raise")
    if date_values.shape != (len(actual),):
        raise ValueError("dates must align with predictions")
    rank_values: list[float] = []
    frame = pd.DataFrame({"date": date_values, "target": actual, "prediction": estimated})
    for _, group in frame.groupby("date", sort=True):
        if len(group) >= 2 and group["target"].nunique() > 1 and group["prediction"].nunique() > 1:
            correlation = group["target"].rank().corr(group["prediction"].rank())
            if math.isfinite(correlation):
                rank_values.append(float(correlation))
    if not rank_values:
        raise ValueError("prediction partition has no finite cross-sectional rank IC")
    error = estimated - actual
    return PredictionMetrics(
        partition=partition,
        observations=len(actual),
        dates=int(frame["date"].nunique()),
        rmse=float(np.sqrt(np.mean(error**2))),
        mae=float(np.mean(np.abs(error))),
        rank_ic=float(np.mean(rank_values)),
    )


def _finite_tensor(values: np.ndarray, name: str, max_bytes: int) -> np.ndarray:
    array = np.asarray(values)
    if array.ndim != 4 or min(array.shape) < 1:
        raise ValueError(f"{name} must be a non-empty four-dimensional tensor")
    if array.dtype not in {np.dtype(np.float32), np.dtype(np.float64)}:
        raise ValueError(f"{name} must use float32 or float64")
    if array.nbytes > max_bytes:
        raise ValueError(f"{name} exceeds max_tensor_bytes")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} must contain only finite values")
    return np.asarray(array, dtype=np.float32)


def _finite_target(values: np.ndarray, expected: int, name: str) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    if array.shape != (expected,) or not np.isfinite(array).all():
        raise ValueError(f"{name} must be a finite vector aligned with samples")
    return array
