"""Model evaluation metrics for return prediction panels."""

from __future__ import annotations

import numpy as np
import pandas as pd


def _finite_xy(
    y_true: pd.Series | np.ndarray, y_pred: pd.Series | np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    yt = np.asarray(y_true, dtype=float)
    yp = np.asarray(y_pred, dtype=float)
    mask = np.isfinite(yt) & np.isfinite(yp)
    return yt[mask], yp[mask]


def _corr(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    if len(y_true) < 2 or np.std(y_true) == 0 or np.std(y_pred) == 0:
        return np.nan
    return float(np.corrcoef(y_true, y_pred)[0, 1])


def _rank_corr(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    if len(y_true) < 2:
        return np.nan
    ranked_true = pd.Series(y_true).rank().to_numpy()
    ranked_pred = pd.Series(y_pred).rank().to_numpy()
    return _corr(ranked_true, ranked_pred)


def regression_metrics(
    y_true: pd.Series | np.ndarray, y_pred: pd.Series | np.ndarray
) -> dict[str, float]:
    """Return core regression, direction, and rank metrics."""
    yt, yp = _finite_xy(y_true, y_pred)
    if len(yt) == 0:
        return {
            "n_obs": 0,
            "mse": np.nan,
            "mae": np.nan,
            "r2": np.nan,
            "directional_accuracy": np.nan,
            "ic": np.nan,
            "rank_ic": np.nan,
        }

    err = yp - yt
    sse = float(np.sum(err**2))
    sst = float(np.sum((yt - yt.mean()) ** 2))
    return {
        "n_obs": int(len(yt)),
        "mse": float(np.mean(err**2)),
        "mae": float(np.mean(np.abs(err))),
        "r2": np.nan if sst == 0 else 1.0 - sse / sst,
        "directional_accuracy": float(np.mean((yt > 0) == (yp > 0))),
        "ic": _corr(yt, yp),
        "rank_ic": _rank_corr(yt, yp),
    }


def information_coefficient_by_date(
    predictions: pd.DataFrame,
    target_col: str = "target",
    prediction_col: str = "prediction",
) -> pd.DataFrame:
    """Cross-sectional Pearson and Spearman IC for each date."""
    rows = []
    for date, g in predictions.groupby("date"):
        yt, yp = _finite_xy(g[target_col], g[prediction_col])
        rows.append(
            {
                "date": date,
                "n_obs": int(len(yt)),
                "ic": _corr(yt, yp),
                "rank_ic": _rank_corr(yt, yp),
            }
        )
    return pd.DataFrame(rows).sort_values("date").reset_index(drop=True)


def evaluate_prediction_panel(
    predictions: pd.DataFrame,
    target_col: str = "target",
    prediction_col: str = "prediction",
    group_cols: list[str] | None = None,
) -> pd.DataFrame:
    """Evaluate a prediction panel overall or by groups such as model/window."""
    group_cols = group_cols or []
    if not group_cols:
        return pd.DataFrame(
            [regression_metrics(predictions[target_col], predictions[prediction_col])]
        )

    rows = []
    for keys, g in predictions.groupby(group_cols):
        if not isinstance(keys, tuple):
            keys = (keys,)
        row = dict(zip(group_cols, keys, strict=False))
        row.update(regression_metrics(g[target_col], g[prediction_col]))
        rows.append(row)
    return pd.DataFrame(rows)


def quantile_return_table(
    predictions: pd.DataFrame,
    target_col: str = "target",
    prediction_col: str = "prediction",
    n_quantiles: int = 5,
) -> pd.DataFrame:
    """Mean realized return by same-date prediction quantile."""
    frames = []
    for date, g in predictions.groupby("date"):
        valid = g[[prediction_col, target_col]].replace([np.inf, -np.inf], np.nan).dropna()
        if valid[prediction_col].nunique() < 2:
            continue
        q = min(n_quantiles, valid[prediction_col].nunique())
        valid = valid.assign(
            quantile=pd.qcut(valid[prediction_col], q=q, labels=False, duplicates="drop") + 1
        )
        valid["date"] = date
        frames.append(valid)
    if not frames:
        return pd.DataFrame(columns=["quantile", "mean_return", "count"])
    panel = pd.concat(frames, ignore_index=True)
    return (
        panel.groupby("quantile")[target_col]
        .agg(mean_return="mean", median_return="median", count="size")
        .reset_index()
    )
