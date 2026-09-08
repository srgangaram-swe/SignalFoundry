"""TemporalAlphaNet: a production-grade neural time-series alpha model.

Architecture (per-symbol sequences of daily feature vectors):

    input projection -> stack of dilated causal Conv1d residual blocks (TCN)
    -> attention pooling over time -> MLP head(s)

Design rationale:

- **Dilated causal convolutions** capture multi-scale temporal structure
  (days to quarters) with a receptive field that grows exponentially in
  depth, train in parallel (unlike RNNs), and cannot peek forward by
  construction — causality is a property of the padding, not a promise.
- **Attention pooling** learns *which* days in the lookback matter instead
  of hard-coding "the last one" (GRU-style) or "all equally" (mean-pool).
- **Composite loss** = Huber on standardized returns + a differentiable
  cross-sectional IC penalty. Cross-sectional rank quality is what the
  portfolio actually monetizes, so the loss optimizes it directly; Huber
  keeps magnitudes calibrated and is robust to fat-tailed return outliers.
- **Date-batched sampling**: every mini-batch contains complete same-date
  cross-sections, which is what makes the IC term meaningful.
- **Optional auxiliary horizons** (multi-task): predicting 1/5/20-day
  returns jointly regularizes the shared encoder. The walk-forward driver
  uses the single-target contract; scripts/train_model.py exercises the
  multi-task path.

The training loop is the real thing: chronological train/validation split by
*date*, AdamW with cosine decay, gradient clipping, early stopping on
validation rank IC, best-checkpoint restore, per-epoch history persisted for
plotting, and deterministic seeding. Checkpoints round-trip via save()/load().

Torch is an optional dependency; the registry raises an actionable error when
it is missing.
"""

from __future__ import annotations

import warnings
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from alphaforge.models.base import AlphaModel, ModelError
from alphaforge.models.torch_models import TORCH_AVAILABLE, _build_sequences, _require_torch

if TORCH_AVAILABLE:
    import torch
    from torch import nn


@dataclass
class TemporalConfig:
    """Hyperparameters for TemporalAlphaNet. Defaults target daily equities."""

    seq_len: int = 64
    hidden_size: int = 64
    n_blocks: int = 4  # dilations 1,2,4,8 -> receptive field ≈ 61 days
    kernel_size: int = 3
    dropout: float = 0.15
    # optimization
    lr: float = 3e-4
    weight_decay: float = 1e-4
    max_epochs: int = 60
    patience: int = 8
    grad_clip: float = 1.0
    dates_per_batch: int = 16
    # loss
    ic_loss_weight: float = 1.0
    huber_delta: float = 1.0  # applied to standardized targets
    aux_loss_weight: float = 0.3
    # data handling
    val_fraction: float = 0.15
    clip_z: float = 5.0
    seed: int = 42
    device: str = "auto"

    def resolve_device(self) -> str:
        if self.device != "auto":
            return self.device
        if TORCH_AVAILABLE and torch.cuda.is_available():
            return "cuda"
        return "cpu"


@dataclass
class TrainingHistory:
    """Per-epoch record of the training loop, persisted for evaluation plots."""

    train_loss: list[float] = field(default_factory=list)
    val_loss: list[float] = field(default_factory=list)
    val_rank_ic: list[float] = field(default_factory=list)
    lr: list[float] = field(default_factory=list)
    best_epoch: int = -1

    def to_frame(self) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "epoch": np.arange(1, len(self.train_loss) + 1),
                "train_loss": self.train_loss,
                "val_loss": self.val_loss,
                "val_rank_ic": self.val_rank_ic,
                "lr": self.lr,
            }
        )


if TORCH_AVAILABLE:

    class _CausalBlock(nn.Module):
        """Dilated causal convolution residual block (pre-norm)."""

        def __init__(self, channels: int, kernel_size: int, dilation: int, dropout: float):
            super().__init__()
            self.pad = (kernel_size - 1) * dilation
            self.norm = nn.LayerNorm(channels)
            self.conv1 = nn.Conv1d(channels, channels, kernel_size, dilation=dilation)
            self.conv2 = nn.Conv1d(channels, channels, kernel_size, dilation=dilation)
            self.drop = nn.Dropout(dropout)

        def forward(self, x: torch.Tensor) -> torch.Tensor:  # x: (B, T, C)
            h = self.norm(x).transpose(1, 2)  # (B, C, T)
            h = nn.functional.pad(h, (self.pad, 0))  # left-pad only: causal
            h = nn.functional.gelu(self.conv1(h))
            h = nn.functional.pad(self.drop(h), (self.pad, 0))
            h = self.conv2(h).transpose(1, 2)  # (B, T, C)
            return x + self.drop(h)

    class _AttentionPool(nn.Module):
        """Learned softmax attention over time steps."""

        def __init__(self, channels: int):
            super().__init__()
            self.score = nn.Sequential(
                nn.Linear(channels, channels), nn.Tanh(), nn.Linear(channels, 1)
            )

        def forward(self, x: torch.Tensor) -> torch.Tensor:  # (B, T, C) -> (B, C)
            weights = torch.softmax(self.score(x), dim=1)
            return (weights * x).sum(dim=1)

    class _TemporalAlphaNet(nn.Module):
        def __init__(self, n_features: int, cfg: TemporalConfig, n_outputs: int = 1):
            super().__init__()
            self.input_proj = nn.Linear(n_features, cfg.hidden_size)
            self.blocks = nn.ModuleList(
                _CausalBlock(cfg.hidden_size, cfg.kernel_size, 2**i, cfg.dropout)
                for i in range(cfg.n_blocks)
            )
            self.pool = _AttentionPool(cfg.hidden_size)
            self.head = nn.Sequential(
                nn.LayerNorm(cfg.hidden_size),
                nn.Linear(cfg.hidden_size, cfg.hidden_size // 2),
                nn.GELU(),
                nn.Dropout(cfg.dropout),
                nn.Linear(cfg.hidden_size // 2, n_outputs),
            )

        def forward(self, x: torch.Tensor) -> torch.Tensor:  # (B, T, F) -> (B, n_outputs)
            h = self.input_proj(x)
            for block in self.blocks:
                h = block(h)
            return self.head(self.pool(h))


def _cross_sectional_ic_loss(
    pred: torch.Tensor, target: torch.Tensor, date_ids: torch.Tensor
) -> torch.Tensor:
    """Negative mean per-date Pearson correlation (differentiable IC proxy).

    Spearman is not differentiable; Pearson on standardized targets is the
    standard smooth surrogate. Dates with < 3 names are skipped.
    """
    losses = []
    for date_id in torch.unique(date_ids):
        mask = date_ids == date_id
        if int(mask.sum()) < 3:
            continue
        p, t = pred[mask], target[mask]
        p = p - p.mean()
        t = t - t.mean()
        denom = p.norm() * t.norm()
        if float(denom.detach()) > 1e-12:
            losses.append(-(p * t).sum() / denom)
    if not losses:
        return pred.new_zeros(())
    return torch.stack(losses).mean()


def _rank_ic_by_date(pred: np.ndarray, target: np.ndarray, dates: np.ndarray) -> float:
    frame = pd.DataFrame({"pred": pred, "target": target, "date": dates})
    ics = []
    for _, g in frame.groupby("date"):
        if len(g) >= 3 and g["pred"].std() > 0 and g["target"].std() > 0:
            ics.append(g["pred"].rank().corr(g["target"].rank()))
    return float(np.nanmean(ics)) if ics else float("nan")


class TemporalAlphaModel(AlphaModel):
    """AlphaModel wrapper around TemporalAlphaNet with the full training loop."""

    name = "temporal_alpha"
    needs_sequence_index = True

    def __init__(self, **kwargs):
        _require_torch()
        self.cfg = TemporalConfig(**kwargs)
        self.net: nn.Module | None = None
        self.columns_: list[str] | None = None
        self.x_mean_: np.ndarray | None = None
        self.x_std_: np.ndarray | None = None
        self.y_mean_: float = 0.0
        self.y_std_: float = 1.0
        self.history_: TrainingHistory | None = None
        self.n_outputs_: int = 1

    # ---------------------------------------------------------------- fit

    def fit(
        self, X: pd.DataFrame, y: pd.Series, aux_y: pd.DataFrame | None = None
    ) -> TemporalAlphaModel:
        """Train with chronological validation, early stopping, best restore.

        ``X`` must carry a (date, symbol) MultiIndex. ``aux_y`` (optional)
        holds extra same-index target columns for multi-task training.
        """
        torch.manual_seed(self.cfg.seed)
        np.random.seed(self.cfg.seed)
        device = self.cfg.resolve_device()

        self.columns_ = list(X.columns)
        x_values = X.to_numpy(dtype=np.float64)
        with warnings.catch_warnings():
            # all-NaN feature columns (e.g. long-window features on short
            # panels) scale to zero rather than warning
            warnings.simplefilter("ignore", RuntimeWarning)
            self.x_mean_ = np.nan_to_num(np.nanmean(x_values, axis=0), nan=0.0)
            self.x_std_ = np.nan_to_num(np.nanstd(x_values, axis=0), nan=1.0)
        self.x_std_[self.x_std_ == 0] = 1.0

        y_values = y.to_numpy(dtype=np.float64)
        finite_y = y_values[np.isfinite(y_values)]
        self.y_mean_ = float(finite_y.mean())
        self.y_std_ = float(finite_y.std()) or 1.0

        targets = [(y_values - self.y_mean_) / self.y_std_]
        if aux_y is not None:
            for col in aux_y.columns:
                a = aux_y[col].to_numpy(dtype=np.float64)
                a_fin = a[np.isfinite(a)]
                a_std = float(a_fin.std()) or 1.0
                targets.append((a - float(a_fin.mean())) / a_std)
        self.n_outputs_ = len(targets)
        target_matrix = np.column_stack(targets)

        Xs = X.copy()
        Xs.loc[:, :] = self._scale_x(x_values)
        seqs, _, positions = _build_sequences(Xs, None, self.cfg.seq_len)
        if len(seqs) == 0:
            raise ValueError("not enough history to build any training sequences")
        seq_targets = target_matrix[positions]
        valid = np.isfinite(seq_targets[:, 0])
        seqs, positions, seq_targets = seqs[valid], positions[valid], seq_targets[valid]
        seq_targets = np.nan_to_num(seq_targets, nan=0.0)

        end_dates = X.index.get_level_values(0).to_numpy()[positions]
        unique_dates = np.unique(end_dates)
        n_val_dates = max(1, int(len(unique_dates) * self.cfg.val_fraction))
        val_dates = set(unique_dates[-n_val_dates:])
        is_val = np.array([d in val_dates for d in end_dates])
        date_codes = pd.factorize(end_dates)[0]

        x_train = torch.as_tensor(seqs[~is_val], dtype=torch.float32, device=device)
        y_train = torch.as_tensor(seq_targets[~is_val], dtype=torch.float32, device=device)
        d_train = torch.as_tensor(date_codes[~is_val], dtype=torch.long, device=device)
        x_val = torch.as_tensor(seqs[is_val], dtype=torch.float32, device=device)
        y_val = torch.as_tensor(seq_targets[is_val], dtype=torch.float32, device=device)
        d_val = torch.as_tensor(date_codes[is_val], dtype=torch.long, device=device)
        val_dates_np = end_dates[is_val]
        if len(x_train) == 0 or len(x_val) == 0:
            raise ValueError("chronological split produced an empty train or validation set")

        self.net = _TemporalAlphaNet(len(self.columns_), self.cfg, self.n_outputs_).to(device)
        opt = torch.optim.AdamW(
            self.net.parameters(), lr=self.cfg.lr, weight_decay=self.cfg.weight_decay
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=self.cfg.max_epochs)
        huber = nn.HuberLoss(delta=self.cfg.huber_delta)

        train_date_codes = d_train.unique()
        history = TrainingHistory()
        best_ic, best_state, bad_epochs = -np.inf, None, 0
        generator = torch.Generator().manual_seed(self.cfg.seed)

        for epoch in range(self.cfg.max_epochs):
            self.net.train()
            epoch_losses = []
            perm = torch.randperm(len(train_date_codes), generator=generator)
            for i in range(0, len(perm), self.cfg.dates_per_batch):
                batch_dates = train_date_codes[perm[i : i + self.cfg.dates_per_batch]]
                mask = torch.isin(d_train, batch_dates)
                xb, yb, db = x_train[mask], y_train[mask], d_train[mask]
                if len(xb) < 4:
                    continue
                opt.zero_grad()
                out = self.net(xb)
                loss = huber(out[:, 0], yb[:, 0])
                loss = loss + self.cfg.ic_loss_weight * _cross_sectional_ic_loss(
                    out[:, 0], yb[:, 0], db
                )
                for k in range(1, self.n_outputs_):
                    loss = loss + self.cfg.aux_loss_weight * huber(out[:, k], yb[:, k])
                loss.backward()
                nn.utils.clip_grad_norm_(self.net.parameters(), self.cfg.grad_clip)
                opt.step()
                epoch_losses.append(float(loss.detach()))
            scheduler.step()

            self.net.eval()
            with torch.no_grad():
                val_out = self.net(x_val)
                val_loss = float(
                    huber(val_out[:, 0], y_val[:, 0])
                    + self.cfg.ic_loss_weight
                    * _cross_sectional_ic_loss(val_out[:, 0], y_val[:, 0], d_val)
                )
                val_ic = _rank_ic_by_date(
                    val_out[:, 0].cpu().numpy(), y_val[:, 0].cpu().numpy(), val_dates_np
                )

            history.train_loss.append(float(np.mean(epoch_losses)) if epoch_losses else np.nan)
            history.val_loss.append(val_loss)
            history.val_rank_ic.append(val_ic)
            history.lr.append(float(opt.param_groups[0]["lr"]))

            if np.isfinite(val_ic) and val_ic > best_ic + 1e-5:
                best_ic, bad_epochs = val_ic, 0
                history.best_epoch = epoch + 1
                best_state = {k: v.detach().clone() for k, v in self.net.state_dict().items()}
            else:
                bad_epochs += 1
                if bad_epochs >= self.cfg.patience:
                    break

        if best_state is not None:
            self.net.load_state_dict(best_state)
        self.history_ = history
        return self

    # ------------------------------------------------------------ predict

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if self.net is None:
            raise RuntimeError("fit or load the model before predicting")
        device = next(self.net.parameters()).device
        Xs = X[self.columns_].copy()
        Xs.loc[:, :] = self._scale_x(X[self.columns_].to_numpy(dtype=np.float64))
        seqs, _, positions = _build_sequences(Xs, None, self.cfg.seq_len)
        preds = np.zeros(len(X))  # zero fallback for rows lacking history
        if len(seqs) > 0:
            self.net.eval()
            with torch.no_grad():
                out = []
                for i in range(0, len(seqs), 4096):
                    chunk = torch.as_tensor(seqs[i : i + 4096], dtype=torch.float32, device=device)
                    out.append(self.net(chunk)[:, 0].cpu().numpy())
                preds[positions] = np.concatenate(out) * self.y_std_ + self.y_mean_
        return preds

    # ------------------------------------------------- persistence & misc

    def save(self, path: str | Path) -> None:
        if self.net is None:
            raise RuntimeError("nothing to save: model is not fit")
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "config": asdict(self.cfg),
                "columns": self.columns_,
                "x_mean": self.x_mean_,
                "x_std": self.x_std_,
                "y_mean": self.y_mean_,
                "y_std": self.y_std_,
                "n_outputs": self.n_outputs_,
                "state_dict": self.net.state_dict(),
            },
            path,
        )

    @classmethod
    def load(cls, path: str | Path, *, trusted: bool = False) -> TemporalAlphaModel:
        """Load a checkpoint after an explicit executable-artifact trust decision."""
        if not trusted:
            raise ModelError(
                "refusing to deserialize an executable torch checkpoint; "
                "pass trusted=True only for a verified, trusted artifact"
            )
        _require_torch()
        payload = torch.load(Path(path), map_location="cpu", weights_only=False)
        model = cls(**payload["config"])
        model.columns_ = payload["columns"]
        model.x_mean_ = payload["x_mean"]
        model.x_std_ = payload["x_std"]
        model.y_mean_ = payload["y_mean"]
        model.y_std_ = payload["y_std"]
        model.n_outputs_ = payload["n_outputs"]
        model.net = _TemporalAlphaNet(len(model.columns_), model.cfg, model.n_outputs_)
        model.net.load_state_dict(payload["state_dict"])
        # A restored model is fitted under the shared AlphaModel contract.
        model._feature_names = tuple(str(c) for c in model.columns_)
        model._is_fitted = True
        return model

    def _scale_x(self, values: np.ndarray) -> np.ndarray:
        z = (values - self.x_mean_) / self.x_std_
        return np.clip(np.nan_to_num(z, nan=0.0), -self.cfg.clip_z, self.cfg.clip_z)
