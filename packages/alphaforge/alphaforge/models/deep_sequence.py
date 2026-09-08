"""Controlled, causal deep-sequence benchmarks for panel return research.

The module intentionally separates four concerns:

* :func:`build_causal_windows` constructs bounded per-symbol histories without
  crossing an asset boundary;
* :class:`SequenceBenchmarkConfig` fixes a common training and resource policy;
* :class:`ControlledSequenceModel` supplies five architecture variants behind
  the shared :class:`~alphaforge.models.base.AlphaModel` contract; and
* :func:`evaluate_oos_predictions` computes comparable predictive,
  regression-calibration, and costed long-short diagnostics.

All learned transforms are fitted on the chronological inner-training segment.
Validation rows influence early stopping only. The implementation is a
research benchmark, not a live-trading component or evidence of persistent
profitability.
"""

from __future__ import annotations

import math
import time
from dataclasses import asdict, dataclass, field
from typing import Any, Literal

import numpy as np
import pandas as pd

from alphaforge.models.base import AlphaModel, ModelError, TrainingDiagnostics
from alphaforge.models.torch_models import TORCH_AVAILABLE, _require_torch

if TORCH_AVAILABLE:
    import torch
    from torch import nn

Architecture = Literal["cnn", "tcn", "lstm", "gru", "transformer"]
DeviceName = Literal["auto", "cpu", "cuda", "mps"]
_ARCHITECTURES: tuple[Architecture, ...] = ("cnn", "tcn", "lstm", "gru", "transformer")


@dataclass(frozen=True)
class SequenceBenchmarkConfig:
    """Common architecture, optimization, and resource policy.

    Defaults deliberately target reproducible CPU research. ``auto`` is a
    conservative CPU fallback; accelerators require an explicit request so
    experiment identity cannot silently change with host hardware.
    """

    seq_len: int = 32
    min_history: int | None = None
    hidden_size: int = 32
    n_layers: int = 2
    kernel_size: int = 3
    n_heads: int = 4
    dropout: float = 0.0
    learning_rate: float = 1e-3
    weight_decay: float = 1e-5
    max_epochs: int = 30
    patience: int = 5
    batch_size: int = 256
    validation_fraction: float = 0.15
    gradient_clip: float = 1.0
    clip_z: float = 8.0
    max_parameters: int = 2_000_000
    max_windows: int = 2_000_000
    max_tensor_bytes: int = 2_147_483_648
    seed: int = 42
    device: DeviceName = "auto"

    def __post_init__(self) -> None:
        if not 2 <= self.seq_len <= 512:
            raise ValueError("seq_len must be in [2, 512]")
        minimum = self.seq_len if self.min_history is None else self.min_history
        if not 1 <= minimum <= self.seq_len:
            raise ValueError("min_history must be in [1, seq_len]")
        if not 4 <= self.hidden_size <= 512:
            raise ValueError("hidden_size must be in [4, 512]")
        if not 1 <= self.n_layers <= 8:
            raise ValueError("n_layers must be in [1, 8]")
        if not 1 <= self.kernel_size <= 15 or self.kernel_size % 2 == 0:
            raise ValueError("kernel_size must be odd and in [1, 15]")
        if not 1 <= self.n_heads <= 16 or self.hidden_size % self.n_heads:
            raise ValueError("n_heads must divide hidden_size and be in [1, 16]")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")
        for name, value in (
            ("learning_rate", self.learning_rate),
            ("weight_decay", self.weight_decay),
            ("gradient_clip", self.gradient_clip),
            ("clip_z", self.clip_z),
        ):
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and positive")
        if not 1 <= self.max_epochs <= 10_000:
            raise ValueError("max_epochs must be in [1, 10000]")
        if not 1 <= self.patience <= self.max_epochs:
            raise ValueError("patience must be in [1, max_epochs]")
        if not 1 <= self.batch_size <= 65_536:
            raise ValueError("batch_size must be in [1, 65536]")
        if not 0.01 <= self.validation_fraction <= 0.5:
            raise ValueError("validation_fraction must be in [0.01, 0.5]")
        if not 1 <= self.max_parameters <= 100_000_000:
            raise ValueError("max_parameters must be in [1, 100000000]")
        if not 1 <= self.max_windows <= 10_000_000:
            raise ValueError("max_windows must be in [1, 10000000]")
        if not 1_048_576 <= self.max_tensor_bytes <= 8_589_934_592:
            raise ValueError("max_tensor_bytes must be in [1048576, 8589934592]")
        if self.seed < 0:
            raise ValueError("seed must be non-negative")
        if self.device not in {"auto", "cpu", "cuda", "mps"}:
            raise ValueError("device must be one of auto, cpu, cuda, or mps")

    @property
    def effective_min_history(self) -> int:
        """Return the validated minimum number of real observations."""
        return self.seq_len if self.min_history is None else self.min_history


@dataclass(frozen=True)
class CausalWindows:
    """Immutable window batch with left-padding and explicit validity masks."""

    values: np.ndarray
    valid_mask: np.ndarray
    row_positions: np.ndarray
    end_dates: np.ndarray
    symbols: np.ndarray
    targets: np.ndarray | None


@dataclass(frozen=True)
class SequenceResourceEvidence:
    """Bounded training-resource and convergence evidence for one fit."""

    architecture: Architecture
    device: str
    parameter_count: int
    parameter_bytes: int
    fit_wall_seconds: float
    fit_cpu_seconds: float
    peak_device_bytes: int
    epochs_completed: int
    best_epoch: int
    best_validation_loss: float
    stopped_early: bool
    training_loss: tuple[float, ...] = field(repr=False)
    validation_loss: tuple[float, ...] = field(repr=False)

    def to_dict(self) -> dict[str, Any]:
        """Return a deterministic JSON-compatible evidence record."""
        return asdict(self)


@dataclass(frozen=True)
class OOSPredictionEvidence:
    """Aggregate out-of-sample diagnostics for a single candidate."""

    model: str
    observations: int
    dates: int
    rmse: float
    mae: float
    pearson_correlation: float
    rank_ic: float
    calibration_intercept: float
    calibration_slope: float
    calibration_rmse: float
    gross_mean_daily_return: float
    net_mean_daily_return: float
    mean_daily_turnover: float
    transaction_cost_bps: float

    def to_dict(self) -> dict[str, Any]:
        """Return a deterministic JSON-compatible evidence record."""
        return asdict(self)


def _validate_panel_index(index: pd.Index) -> tuple[np.ndarray, np.ndarray]:
    if not isinstance(index, pd.MultiIndex) or index.nlevels != 2:
        raise ValueError("sequence data must use a two-level (date, symbol) MultiIndex")
    if index.has_duplicates:
        raise ValueError("sequence index must contain unique (date, symbol) rows")
    dates = pd.to_datetime(index.get_level_values(0), errors="raise").to_numpy()
    symbols = index.get_level_values(1).astype(str).to_numpy()
    if any(not symbol for symbol in symbols):
        raise ValueError("symbols must be non-empty")
    return dates, symbols


def build_causal_windows(
    X: pd.DataFrame,
    y: pd.Series | None,
    *,
    seq_len: int,
    min_history: int | None = None,
    max_windows: int = 2_000_000,
    max_tensor_bytes: int = 2_147_483_648,
) -> CausalWindows:
    """Build deterministic, left-padded windows ending at each prediction row.

    Rows are sorted by ``(symbol, date)`` internally and mapped back to their
    original positions. A window never includes another symbol and never
    includes an observation later than its end date.
    """
    if not isinstance(X, pd.DataFrame) or X.empty:
        raise ValueError("X must be a non-empty pandas DataFrame")
    if not 2 <= seq_len <= 512:
        raise ValueError("seq_len must be in [2, 512]")
    required = seq_len if min_history is None else min_history
    if not 1 <= required <= seq_len:
        raise ValueError("min_history must be in [1, seq_len]")
    if not 1 <= max_windows <= 10_000_000:
        raise ValueError("max_windows must be in [1, 10000000]")
    if not 1_048_576 <= max_tensor_bytes <= 8_589_934_592:
        raise ValueError("max_tensor_bytes must be in [1048576, 8589934592]")
    dates, symbols = _validate_panel_index(X.index)
    if y is not None and (not isinstance(y, pd.Series) or not X.index.equals(y.index)):
        raise ValueError("y must be a Series with the identical ordered index as X")
    try:
        raw_values = X.to_numpy(dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError("sequence features must be numeric") from exc
    if np.isinf(raw_values).any():
        raise ValueError("sequence features contain infinite values")
    target_values = None if y is None else y.to_numpy(dtype=np.float64)
    if target_values is not None and not np.isfinite(target_values).all():
        raise ValueError("sequence targets must be finite")

    order = np.lexsort((dates, symbols))
    sorted_symbols = symbols[order]
    boundaries = np.flatnonzero(np.r_[True, sorted_symbols[1:] != sorted_symbols[:-1]])
    count = sum(
        max(0, int(end - start) - required + 1)
        for start, end in zip(boundaries, np.r_[boundaries[1:], len(order)])
    )
    if count == 0:
        raise ValueError("not enough per-symbol history to construct a sequence")
    if count > max_windows:
        raise ValueError(f"sequence window count {count} exceeds max_windows={max_windows}")
    tensor_bytes = count * seq_len * X.shape[1] * np.dtype(np.float32).itemsize
    tensor_bytes += count * seq_len * np.dtype(bool).itemsize
    if tensor_bytes > max_tensor_bytes:
        raise ValueError(
            f"sequence tensors require {tensor_bytes} bytes; "
            f"max_tensor_bytes={max_tensor_bytes}"
        )

    values = np.zeros((count, seq_len, X.shape[1]), dtype=np.float32)
    masks = np.zeros((count, seq_len), dtype=bool)
    positions = np.empty(count, dtype=np.int64)
    end_dates = np.empty(count, dtype=dates.dtype)
    window_symbols = np.empty(count, dtype=object)
    targets = None if target_values is None else np.empty(count, dtype=np.float64)
    output = 0
    finite_values = np.nan_to_num(raw_values, nan=0.0)
    for start, end in zip(boundaries, np.r_[boundaries[1:], len(order)]):
        rows = order[start:end]
        for offset in range(required - 1, len(rows)):
            history = rows[max(0, offset - seq_len + 1) : offset + 1]
            history_length = len(history)
            values[output, -history_length:] = finite_values[history]
            masks[output, -history_length:] = True
            position = int(rows[offset])
            positions[output] = position
            end_dates[output] = dates[position]
            window_symbols[output] = symbols[position]
            if targets is not None and target_values is not None:
                targets[output] = target_values[position]
            output += 1
    return CausalWindows(values, masks, positions, end_dates, window_symbols, targets)


def resolve_sequence_device(requested: DeviceName) -> str:
    """Resolve an explicit device policy or fail closed when unavailable."""
    _require_torch()
    if requested in {"auto", "cpu"}:
        return "cpu"
    if requested == "cuda":
        if not torch.cuda.is_available():
            raise ModelError("CUDA was requested but is unavailable")
        return "cuda"
    if requested == "mps":
        backend = getattr(torch.backends, "mps", None)
        if backend is None or not backend.is_available():
            raise ModelError("Apple MPS was requested but is unavailable")
        return "mps"
    raise ModelError(f"unsupported device policy: {requested}")


if TORCH_AVAILABLE:

    class _CausalConvBlock(nn.Module):
        def __init__(self, width: int, kernel_size: int, dilation: int, dropout: float):
            super().__init__()
            self.left_padding = (kernel_size - 1) * dilation
            self.norm = nn.LayerNorm(width)
            self.conv = nn.Conv1d(width, width, kernel_size, dilation=dilation)
            self.dropout = nn.Dropout(dropout)

        def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
            residual = x
            hidden = self.norm(x).transpose(1, 2)
            hidden = nn.functional.pad(hidden, (self.left_padding, 0))
            hidden = self.conv(hidden).transpose(1, 2)
            hidden = self.dropout(nn.functional.gelu(hidden))
            hidden = (residual + hidden) * mask.unsqueeze(-1)
            return hidden

    class _MaskedRecurrent(nn.Module):
        """GRU/LSTM cells whose state is unchanged on padded steps."""

        def __init__(self, kind: Literal["gru", "lstm"], width: int, layers: int):
            super().__init__()
            cell = nn.GRUCell if kind == "gru" else nn.LSTMCell
            self.kind = kind
            self.cells = nn.ModuleList(cell(width, width) for _ in range(layers))

        def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
            batch, steps, width = x.shape
            hidden = [x.new_zeros((batch, width)) for _ in self.cells]
            cells = [x.new_zeros((batch, width)) for _ in self.cells]
            for step in range(steps):
                active = mask[:, step].unsqueeze(-1)
                layer_input = x[:, step]
                for layer, cell in enumerate(self.cells):
                    if self.kind == "lstm":
                        candidate_h, candidate_c = cell(layer_input, (hidden[layer], cells[layer]))
                        cells[layer] = torch.where(active, candidate_c, cells[layer])
                    else:
                        candidate_h = cell(layer_input, hidden[layer])
                    hidden[layer] = torch.where(active, candidate_h, hidden[layer])
                    layer_input = hidden[layer]
            return hidden[-1]

    class _SequenceNetwork(nn.Module):
        def __init__(
            self,
            architecture: Architecture,
            n_features: int,
            config: SequenceBenchmarkConfig,
        ):
            super().__init__()
            self.architecture = architecture
            self.projection = nn.Linear(n_features, config.hidden_size)
            if architecture == "cnn":
                self.encoder: nn.Module = _CausalConvBlock(
                    config.hidden_size, config.kernel_size, 1, config.dropout
                )
            elif architecture == "tcn":
                self.encoder = nn.ModuleList(
                    _CausalConvBlock(
                        config.hidden_size,
                        config.kernel_size,
                        2**layer,
                        config.dropout,
                    )
                    for layer in range(config.n_layers)
                )
            elif architecture in {"lstm", "gru"}:
                self.encoder = _MaskedRecurrent(architecture, config.hidden_size, config.n_layers)
            else:
                encoder_layer = nn.TransformerEncoderLayer(
                    d_model=config.hidden_size,
                    nhead=config.n_heads,
                    dim_feedforward=config.hidden_size * 2,
                    dropout=config.dropout,
                    activation="gelu",
                    batch_first=True,
                    norm_first=False,
                )
                self.encoder = nn.TransformerEncoder(encoder_layer, config.n_layers)
                self.position = nn.Parameter(torch.zeros(1, config.seq_len, config.hidden_size))
            self.head = nn.Sequential(
                nn.LayerNorm(config.hidden_size), nn.Linear(config.hidden_size, 1)
            )

        def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
            hidden = self.projection(x) * mask.unsqueeze(-1)
            if self.architecture == "cnn":
                hidden = self.encoder(hidden, mask)
                encoded = hidden[:, -1]
            elif self.architecture == "tcn":
                if not isinstance(self.encoder, nn.ModuleList):  # pragma: no cover - constructor
                    raise RuntimeError("TCN encoder invariant violated")
                for block in self.encoder:
                    hidden = block(hidden, mask)
                encoded = hidden[:, -1]
            elif self.architecture in {"lstm", "gru"}:
                encoded = self.encoder(hidden, mask)
            else:
                hidden = hidden + self.position[:, : hidden.shape[1]]
                causal_mask = torch.triu(
                    torch.ones(
                        hidden.shape[1],
                        hidden.shape[1],
                        dtype=torch.bool,
                        device=hidden.device,
                    ),
                    diagonal=1,
                )
                hidden = self.encoder(
                    hidden,
                    mask=causal_mask,
                    src_key_padding_mask=~mask,
                )
                encoded = hidden[:, -1]
            return self.head(encoded).squeeze(-1)


class ControlledSequenceModel(AlphaModel):
    """One member of the governed deep-sequence benchmark family."""

    needs_sequence_index = True

    def __init__(self, architecture: Architecture, **kwargs: Any):
        _require_torch()
        if architecture not in _ARCHITECTURES:
            raise ValueError(f"architecture must be one of {_ARCHITECTURES}")
        self.architecture = architecture
        self.config = SequenceBenchmarkConfig(**kwargs)
        self.net: nn.Module | None = None
        self.columns_: list[str] | None = None
        self.x_mean_: np.ndarray | None = None
        self.x_std_: np.ndarray | None = None
        self.y_mean_: float = 0.0
        self.y_std_: float = 1.0
        self.resource_evidence_: SequenceResourceEvidence | None = None

    def get_params(self) -> dict[str, Any]:
        return {"architecture": self.architecture, **asdict(self.config)}

    def training_diagnostics(self) -> TrainingDiagnostics:
        self._ensure_fitted()
        evidence = self.resource_evidence()
        return TrainingDiagnostics(
            backend=f"torch-{self.architecture}-{evidence.device}",
            status="completed" if evidence.stopped_early else "max_iterations",
            iterations=evidence.epochs_completed,
            iteration_limit=self.config.max_epochs,
            seed=self.config.seed,
            warnings=(
                ()
                if evidence.stopped_early
                else ("epoch budget exhausted before patience triggered",)
            ),
        )

    def resource_evidence(self) -> SequenceResourceEvidence:
        self._ensure_fitted()
        if self.resource_evidence_ is None:
            raise ModelError("training resource evidence is unavailable")
        return self.resource_evidence_

    def fit(self, X: pd.DataFrame, y: pd.Series) -> ControlledSequenceModel:
        dates, _ = _validate_panel_index(X.index)
        unique_dates = np.unique(dates)
        if len(unique_dates) < 3:
            raise ValueError("at least three unique dates are required")
        validation_dates = max(1, math.ceil(len(unique_dates) * self.config.validation_fraction))
        if validation_dates >= len(unique_dates):
            raise ValueError("chronological validation consumed all available dates")
        validation_start = unique_dates[-validation_dates]
        transform_rows = dates < validation_start
        if not transform_rows.any():
            raise ValueError("chronological inner-training segment is empty")

        self.columns_ = list(X.columns)
        raw = X.to_numpy(dtype=np.float64)
        with np.errstate(invalid="ignore"):
            self.x_mean_ = np.nan_to_num(np.nanmean(raw[transform_rows], axis=0), nan=0.0)
            self.x_std_ = np.nan_to_num(np.nanstd(raw[transform_rows], axis=0), nan=1.0)
        self.x_std_[self.x_std_ == 0.0] = 1.0
        y_train = y.to_numpy(dtype=np.float64)[transform_rows]
        self.y_mean_ = float(np.mean(y_train))
        self.y_std_ = float(np.std(y_train)) or 1.0

        scaled = X.copy()
        scaled.loc[:, :] = self._scale(raw)
        windows = build_causal_windows(
            scaled,
            y,
            seq_len=self.config.seq_len,
            min_history=self.config.effective_min_history,
            max_windows=self.config.max_windows,
            max_tensor_bytes=self.config.max_tensor_bytes,
        )
        is_validation = windows.end_dates >= validation_start
        if not is_validation.any() or is_validation.all():
            raise ValueError("window construction produced an empty train or validation segment")
        if windows.targets is None:  # pragma: no cover - y was supplied above
            raise RuntimeError("training window construction omitted targets")
        targets = (windows.targets - self.y_mean_) / self.y_std_
        device_name = resolve_sequence_device(self.config.device)
        device = torch.device(device_name)
        torch.manual_seed(self.config.seed)
        if device_name == "cuda":
            torch.cuda.manual_seed_all(self.config.seed)
            torch.cuda.reset_peak_memory_stats(device)

        self.net = _SequenceNetwork(self.architecture, len(self.columns_), self.config).to(device)
        parameter_count = sum(parameter.numel() for parameter in self.net.parameters())
        if parameter_count > self.config.max_parameters:
            self.net = None
            raise ModelError(
                f"{self.architecture} parameter count {parameter_count} exceeds "
                f"max_parameters={self.config.max_parameters}"
            )
        parameter_bytes = sum(
            parameter.numel() * parameter.element_size() for parameter in self.net.parameters()
        )
        x_tensor = torch.as_tensor(windows.values, dtype=torch.float32, device=device)
        mask_tensor = torch.as_tensor(windows.valid_mask, dtype=torch.bool, device=device)
        y_tensor = torch.as_tensor(targets, dtype=torch.float32, device=device)
        train_indices = torch.as_tensor(np.flatnonzero(~is_validation), dtype=torch.long)
        validation_indices = torch.as_tensor(np.flatnonzero(is_validation), dtype=torch.long)

        optimizer = torch.optim.AdamW(
            self.net.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
        )
        loss_function = nn.HuberLoss(delta=1.0)
        generator = torch.Generator().manual_seed(self.config.seed)
        best_loss = math.inf
        best_epoch = 0
        best_state: dict[str, torch.Tensor] | None = None
        bad_epochs = 0
        train_losses: list[float] = []
        validation_losses: list[float] = []
        started_wall = time.perf_counter()
        started_cpu = time.process_time()
        for epoch in range(self.config.max_epochs):
            self.net.train()
            permutation = train_indices[torch.randperm(len(train_indices), generator=generator)]
            epoch_losses: list[float] = []
            for start in range(0, len(permutation), self.config.batch_size):
                indices = permutation[start : start + self.config.batch_size].to(device)
                optimizer.zero_grad(set_to_none=True)
                prediction = self.net(x_tensor[indices], mask_tensor[indices])
                loss = loss_function(prediction, y_tensor[indices])
                if not torch.isfinite(loss):
                    raise ModelError("training produced a non-finite loss")
                loss.backward()
                nn.utils.clip_grad_norm_(self.net.parameters(), self.config.gradient_clip)
                optimizer.step()
                epoch_losses.append(float(loss.detach().cpu()))
            self.net.eval()
            with torch.no_grad():
                indices = validation_indices.to(device)
                validation_loss = float(
                    loss_function(
                        self.net(x_tensor[indices], mask_tensor[indices]),
                        y_tensor[indices],
                    ).cpu()
                )
            train_losses.append(float(np.mean(epoch_losses)))
            validation_losses.append(validation_loss)
            if validation_loss < best_loss - 1e-8:
                best_loss = validation_loss
                best_epoch = epoch + 1
                bad_epochs = 0
                best_state = {
                    name: tensor.detach().cpu().clone()
                    for name, tensor in self.net.state_dict().items()
                }
            else:
                bad_epochs += 1
                if bad_epochs >= self.config.patience:
                    break
        wall_seconds = time.perf_counter() - started_wall
        cpu_seconds = time.process_time() - started_cpu
        if best_state is None or not math.isfinite(best_loss):
            raise ModelError("training did not produce a finite validation checkpoint")
        self.net.load_state_dict(best_state)
        peak_device_bytes = (
            int(torch.cuda.max_memory_allocated(device)) if device_name == "cuda" else 0
        )
        self.resource_evidence_ = SequenceResourceEvidence(
            architecture=self.architecture,
            device=device_name,
            parameter_count=parameter_count,
            parameter_bytes=parameter_bytes,
            fit_wall_seconds=wall_seconds,
            fit_cpu_seconds=cpu_seconds,
            peak_device_bytes=peak_device_bytes,
            epochs_completed=len(train_losses),
            best_epoch=best_epoch,
            best_validation_loss=best_loss,
            stopped_early=len(train_losses) < self.config.max_epochs,
            training_loss=tuple(train_losses),
            validation_loss=tuple(validation_losses),
        )
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if self.net is None or self.columns_ is None:
            raise ModelError("fit or restore the model before prediction")
        scaled = X.loc[:, self.columns_].copy()
        scaled.loc[:, :] = self._scale(scaled.to_numpy(dtype=np.float64))
        try:
            windows = build_causal_windows(
                scaled,
                None,
                seq_len=self.config.seq_len,
                min_history=self.config.effective_min_history,
                max_windows=self.config.max_windows,
                max_tensor_bytes=self.config.max_tensor_bytes,
            )
        except ValueError as exc:
            if "not enough per-symbol history" in str(exc):
                return np.zeros(len(X), dtype=np.float64)
            raise
        device = next(self.net.parameters()).device
        output = np.zeros(len(X), dtype=np.float64)
        self.net.eval()
        with torch.no_grad():
            for start in range(0, len(windows.values), self.config.batch_size):
                stop = start + self.config.batch_size
                values = torch.as_tensor(
                    windows.values[start:stop], dtype=torch.float32, device=device
                )
                masks = torch.as_tensor(
                    windows.valid_mask[start:stop], dtype=torch.bool, device=device
                )
                prediction = self.net(values, masks).cpu().numpy()
                output[windows.row_positions[start:stop]] = prediction * self.y_std_ + self.y_mean_
        return output

    def _scale(self, values: np.ndarray) -> np.ndarray:
        if self.x_mean_ is None or self.x_std_ is None:
            raise ModelError("training transform is unavailable")
        scaled = (values - self.x_mean_) / self.x_std_
        return np.clip(
            np.nan_to_num(scaled, nan=0.0, posinf=self.config.clip_z, neginf=-self.config.clip_z),
            -self.config.clip_z,
            self.config.clip_z,
        )


def evaluate_oos_predictions(
    frame: pd.DataFrame,
    *,
    model: str,
    transaction_cost_bps: float,
    selection_fraction: float = 0.2,
) -> OOSPredictionEvidence:
    """Evaluate pre-existing OOS predictions under one comparable policy.

    The economic diagnostic ranks each date's cross-section, takes equal-weight
    long and short tails, and subtracts turnover-proportional cost. It is a
    deliberately simple comparison diagnostic, not an execution simulator.
    """
    required = {"date", "symbol", "target", "prediction"}
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"prediction frame is missing columns: {missing}")
    if not model:
        raise ValueError("model must be non-empty")
    if not math.isfinite(transaction_cost_bps) or transaction_cost_bps < 0.0:
        raise ValueError("transaction_cost_bps must be finite and non-negative")
    if not 0.0 < selection_fraction <= 0.5:
        raise ValueError("selection_fraction must be in (0, 0.5]")
    data = frame.loc[:, ["date", "symbol", "target", "prediction"]].copy()
    data["date"] = pd.to_datetime(data["date"], errors="raise")
    if data.duplicated(["date", "symbol"]).any():
        raise ValueError("prediction frame contains duplicate (date, symbol) rows")
    values = data[["target", "prediction"]].to_numpy(dtype=float)
    if len(data) < 2 or not np.isfinite(values).all():
        raise ValueError("prediction evidence requires at least two finite rows")

    target = values[:, 0]
    prediction = values[:, 1]
    residual = target - prediction
    pearson = (
        float(np.corrcoef(target, prediction)[0, 1])
        if np.std(target) > 0.0 and np.std(prediction) > 0.0
        else 0.0
    )
    date_rank_ics: list[float] = []
    gross_returns: list[float] = []
    turnovers: list[float] = []
    previous_weights: dict[str, float] = {}
    for _, group in data.sort_values(["date", "symbol"]).groupby("date", sort=True):
        if len(group) < 2:
            continue
        if group["prediction"].std() > 0.0 and group["target"].std() > 0.0:
            date_rank_ics.append(float(group["prediction"].rank().corr(group["target"].rank())))
        tail = max(1, int(math.floor(len(group) * selection_fraction)))
        ordered = group.sort_values(["prediction", "symbol"])
        short = ordered.head(tail)
        long = ordered.tail(tail)
        weights = {str(symbol): -0.5 / tail for symbol in short["symbol"]}
        weights.update({str(symbol): 0.5 / tail for symbol in long["symbol"]})
        realized = {str(row.symbol): float(row.target) for row in group.itertuples(index=False)}
        gross_returns.append(sum(weight * realized[symbol] for symbol, weight in weights.items()))
        universe = set(previous_weights).union(weights)
        turnovers.append(
            sum(
                abs(weights.get(symbol, 0.0) - previous_weights.get(symbol, 0.0))
                for symbol in universe
            )
        )
        previous_weights = weights
    if not gross_returns:
        raise ValueError("prediction evidence has no date with a tradable cross-section")
    cost_rate = transaction_cost_bps / 10_000.0
    net_returns = np.asarray(gross_returns) - cost_rate * np.asarray(turnovers)
    design = np.column_stack([np.ones(len(prediction)), prediction])
    coefficients, *_ = np.linalg.lstsq(design, target, rcond=None)
    calibrated = design @ coefficients
    return OOSPredictionEvidence(
        model=model,
        observations=len(data),
        dates=int(data["date"].nunique()),
        rmse=float(np.sqrt(np.mean(np.square(residual)))),
        mae=float(np.mean(np.abs(residual))),
        pearson_correlation=pearson,
        rank_ic=float(np.mean(date_rank_ics)) if date_rank_ics else 0.0,
        calibration_intercept=float(coefficients[0]),
        calibration_slope=float(coefficients[1]),
        calibration_rmse=float(np.sqrt(np.mean(np.square(target - calibrated)))),
        gross_mean_daily_return=float(np.mean(gross_returns)),
        net_mean_daily_return=float(np.mean(net_returns)),
        mean_daily_turnover=float(np.mean(turnovers)),
        transaction_cost_bps=float(transaction_cost_bps),
    )
