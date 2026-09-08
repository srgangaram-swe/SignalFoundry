"""Shared utilities: logging, JSON output, seeding, and run IDs."""

from __future__ import annotations

import json
import logging
import random
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

ANNUALIZATION_DAYS = 252


def get_logger(name: str) -> logging.Logger:
    logger = logging.getLogger(name)
    if not logging.getLogger().handlers and not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(
            logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s", "%H:%M:%S")
        )
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    return logger


def save_json(obj: Any, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    def _default(o: Any) -> Any:
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, (np.floating,)):
            return float(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        if hasattr(o, "isoformat"):
            return o.isoformat()
        return str(o)

    with open(path, "w") as f:
        json.dump(obj, f, indent=2, default=_default)


def set_seed(seed: int) -> None:
    """Seed applicable Python, NumPy, PyTorch, and TensorFlow runtime RNGs."""
    if seed < 0:
        raise ValueError("seed must be non-negative")
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        if hasattr(torch, "manual_seed"):
            torch.manual_seed(seed)
        if hasattr(torch, "cuda") and torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        if hasattr(torch, "use_deterministic_algorithms"):
            torch.use_deterministic_algorithms(True, warn_only=True)
    except (ImportError, AttributeError):
        pass
    try:
        import tensorflow as tf

        tf.random.set_seed(seed)
        if hasattr(tf.config.experimental, "enable_op_determinism"):
            tf.config.experimental.enable_op_determinism()
    except (ImportError, AttributeError):
        pass


def timestamp_id(prefix: str = "run") -> str:
    return f"{prefix}_{datetime.now(UTC).strftime('%Y%m%d_%H%M%S')}"
