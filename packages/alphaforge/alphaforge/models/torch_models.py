"""PyTorch models: tabular MLP and sequence models (GRU, temporal CNN).

Torch is an optional dependency (`pip install 'alphaforge[torch]'`). Import
of this module fails gracefully at registry level when torch is absent.

Sequence models expect X to carry a (date, symbol) MultiIndex so causal
windows can be built per symbol. Rows without enough history fall back to a
zero prediction (documented, and consistent with the zero baseline).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

try:
    import torch
    from torch import nn

    TORCH_AVAILABLE = True
except ImportError:  # pragma: no cover - exercised only without torch
    TORCH_AVAILABLE = False

from alphaforge.models.base import AlphaModel


def _require_torch() -> None:
    if not TORCH_AVAILABLE:
        raise ImportError("PyTorch is required: pip install 'alphaforge[torch]'")


def _to_tensor(a: np.ndarray) -> torch.Tensor:
    return torch.as_tensor(np.nan_to_num(a, nan=0.0), dtype=torch.float32)


class _TorchBase(AlphaModel):
    def __init__(
        self,
        hidden_size: int = 64,
        epochs: int = 30,
        lr: float = 1e-3,
        batch_size: int = 1024,
        weight_decay: float = 1e-5,
        val_fraction: float = 0.1,
        patience: int = 5,
        seed: int = 42,
    ):
        _require_torch()
        self.hidden_size = hidden_size
        self.epochs = epochs
        self.lr = lr
        self.batch_size = batch_size
        self.weight_decay = weight_decay
        self.val_fraction = val_fraction
        self.patience = patience
        self.seed = seed
        self.net: nn.Module | None = None
        self.columns_: list[str] | None = None
        self.x_mean_: np.ndarray | None = None
        self.x_std_: np.ndarray | None = None

    def _require_net(self) -> nn.Module:
        if self.net is None:
            raise RuntimeError("fit the model before using it")
        return self.net

    def _scale_fit(self, X: np.ndarray) -> np.ndarray:
        self.x_mean_ = np.nanmean(X, axis=0)
        self.x_std_ = np.nanstd(X, axis=0)
        self.x_std_[self.x_std_ == 0] = 1.0
        return self._scale(X)

    def _scale(self, X: np.ndarray) -> np.ndarray:
        return np.clip((X - self.x_mean_) / self.x_std_, -5, 5)

    def _train_loop(self, X: torch.Tensor, y: torch.Tensor) -> None:
        torch.manual_seed(self.seed)
        net = self._require_net()
        # time-ordered validation split: last fraction of training rows
        n_val = max(1, int(len(X) * self.val_fraction))
        X_tr, y_tr, X_val, y_val = X[:-n_val], y[:-n_val], X[-n_val:], y[-n_val:]
        opt = torch.optim.Adam(net.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        loss_fn = nn.MSELoss()
        best_val, best_state, bad_epochs = np.inf, None, 0
        for _ in range(self.epochs):
            net.train()
            perm = torch.randperm(len(X_tr))
            for i in range(0, len(X_tr), self.batch_size):
                idx = perm[i : i + self.batch_size]
                opt.zero_grad()
                loss = loss_fn(net(X_tr[idx]).squeeze(-1), y_tr[idx])
                loss.backward()
                opt.step()
            net.eval()
            with torch.no_grad():
                val = float(loss_fn(net(X_val).squeeze(-1), y_val))
            if val < best_val - 1e-7:
                best_val, bad_epochs = val, 0
                best_state = {k: v.clone() for k, v in net.state_dict().items()}
            else:
                bad_epochs += 1
                if bad_epochs >= self.patience:
                    break
        if best_state is not None:
            net.load_state_dict(best_state)


class TorchMLP(_TorchBase):
    """Two-hidden-layer MLP on tabular features."""

    name = "torch_mlp"

    def fit(self, X: pd.DataFrame, y: pd.Series) -> TorchMLP:
        self.columns_ = list(X.columns)
        Xs = self._scale_fit(X.to_numpy(dtype=np.float64))
        self.net = nn.Sequential(
            nn.Linear(Xs.shape[1], self.hidden_size),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(self.hidden_size, self.hidden_size // 2),
            nn.ReLU(),
            nn.Linear(self.hidden_size // 2, 1),
        )
        self._train_loop(_to_tensor(Xs), _to_tensor(y.to_numpy(dtype=np.float64)))
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        net = self._require_net()
        Xs = self._scale(X[self.columns_].to_numpy(dtype=np.float64))
        net.eval()
        with torch.no_grad():
            return net(_to_tensor(Xs)).squeeze(-1).numpy()


def _build_sequences(
    X: pd.DataFrame, y: pd.Series | None, seq_len: int
) -> tuple[np.ndarray, np.ndarray | None, np.ndarray]:
    """Per-symbol causal windows: sequence ending at t predicts label at t.

    Returns (sequences, targets, row_positions) where row_positions maps each
    sequence back to its row in X. Rows with < seq_len history are skipped.
    """
    if not isinstance(X.index, pd.MultiIndex):
        raise ValueError("sequence models need X indexed by (date, symbol)")
    dates = X.index.get_level_values(0)
    symbols = X.index.get_level_values(1)
    values = np.nan_to_num(X.to_numpy(dtype=np.float64), nan=0.0)
    order = np.lexsort((dates, symbols))  # sort by symbol then date
    seqs, targets, positions = [], [], []
    y_arr = None if y is None else y.to_numpy(dtype=np.float64)
    sorted_symbols = symbols[order]
    boundaries = np.flatnonzero(np.r_[True, sorted_symbols[1:] != sorted_symbols[:-1]])
    for b_start, b_end in zip(boundaries, np.r_[boundaries[1:], len(order)]):
        rows = order[b_start:b_end]
        for i in range(seq_len - 1, len(rows)):
            window = rows[i - seq_len + 1 : i + 1]
            seqs.append(values[window])
            positions.append(rows[i])
            if y_arr is not None:
                targets.append(y_arr[rows[i]])
    return (
        np.asarray(seqs),
        None if y is None else np.asarray(targets),
        np.asarray(positions),
    )


class _TorchSequenceModel(_TorchBase):
    needs_sequence_index = True

    def __init__(self, seq_len: int = 20, **kwargs):
        super().__init__(**kwargs)
        self.seq_len = seq_len

    def _make_net(self, n_features: int) -> nn.Module:
        raise NotImplementedError

    def fit(self, X: pd.DataFrame, y: pd.Series) -> _TorchSequenceModel:
        self.columns_ = list(X.columns)
        self._scale_fit(X.to_numpy(dtype=np.float64))
        Xs = X.copy()
        Xs.loc[:, :] = self._scale(X.to_numpy(dtype=np.float64))
        seqs, targets, _ = _build_sequences(Xs, y, self.seq_len)
        if len(seqs) == 0:
            raise ValueError("not enough history to build any training sequences")
        if targets is None:
            raise RuntimeError("training sequence construction did not return targets")
        self.net = self._make_net(len(self.columns_))
        self._train_loop(_to_tensor(seqs), _to_tensor(targets))
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        net = self._require_net()
        Xs = X[self.columns_].copy()
        Xs.loc[:, :] = self._scale(X[self.columns_].to_numpy(dtype=np.float64))
        seqs, _, positions = _build_sequences(Xs, None, self.seq_len)
        preds = np.zeros(len(X))  # zero fallback for rows lacking history
        if len(seqs) > 0:
            net.eval()
            with torch.no_grad():
                out = net(_to_tensor(seqs)).squeeze(-1).numpy()
            preds[positions] = out
        return preds


class TorchGRU(_TorchSequenceModel):
    """GRU over the last ``seq_len`` daily feature vectors."""

    name = "torch_gru"

    def _make_net(self, n_features: int) -> nn.Module:
        class GRUHead(nn.Module):
            def __init__(self, n_in: int, hidden: int):
                super().__init__()
                self.gru = nn.GRU(n_in, hidden, batch_first=True)
                self.head = nn.Linear(hidden, 1)

            def forward(self, x):
                out, _ = self.gru(x)
                return self.head(out[:, -1])

        return GRUHead(n_features, self.hidden_size)


class TorchTemporalCNN(_TorchSequenceModel):
    """1-D causal convolution stack over the feature sequence."""

    name = "torch_tcn"

    def _make_net(self, n_features: int) -> nn.Module:
        hidden = self.hidden_size

        class TCN(nn.Module):
            def __init__(self):
                super().__init__()
                self.conv = nn.Sequential(
                    nn.Conv1d(n_features, hidden, kernel_size=3, padding=2, dilation=1),
                    nn.ReLU(),
                    nn.Conv1d(hidden, hidden, kernel_size=3, padding=4, dilation=2),
                    nn.ReLU(),
                )
                self.head = nn.Linear(hidden, 1)

            def forward(self, x):  # x: (batch, seq, feat)
                h = self.conv(x.transpose(1, 2))
                return self.head(h[:, :, -1])

        return TCN()
