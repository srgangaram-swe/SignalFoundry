"""Portfolio construction from signal scores."""

from __future__ import annotations

import numpy as np
import pandas as pd

ID_COLUMNS = ["date", "symbol"]


def _normalize_scores(scores: pd.Series, max_gross: float) -> pd.Series:
    scores = scores.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    if scores.abs().sum() == 0:
        return scores
    pos = scores.clip(lower=0)
    neg = scores.clip(upper=0)
    if pos.sum() > 0 and neg.abs().sum() > 0:
        long_w = pos / pos.sum() * (max_gross / 2.0)
        short_w = neg / neg.abs().sum() * (max_gross / 2.0)
        return long_w + short_w
    return scores / scores.abs().sum() * max_gross


def _apply_caps(
    weights: pd.Series, max_weight: float, max_gross: float, max_net: float
) -> pd.Series:
    weights = weights.clip(lower=-max_weight, upper=max_weight)
    gross = weights.abs().sum()
    if gross > max_gross and gross > 0:
        weights = weights * (max_gross / gross)
    net = float(weights.sum())
    if net > max_net:
        positive = weights.clip(lower=0)
        if positive.sum() > 0:
            weights = weights - positive * ((net - max_net) / positive.sum())
    elif net < -max_net:
        negative = weights.clip(upper=0)
        if negative.abs().sum() > 0:
            weights = weights - negative * ((net + max_net) / negative.sum())
    return weights


def construct_portfolio(
    signals: pd.DataFrame,
    features: pd.DataFrame | None = None,
    config: dict | None = None,
) -> pd.DataFrame:
    """Convert signal scores into capped target weights by date."""
    cfg = config or {}
    allowed = {
        "scheme",
        "max_weight",
        "max_gross_exposure",
        "max_net_exposure",
        "inverse_vol_scaling",
        "turnover_cap",
        "vol_lookback",
        "cash_buffer",
    }
    unknown = set(cfg) - allowed
    if unknown:
        raise ValueError(f"unknown portfolio settings: {sorted(unknown)}")
    max_weight = float(cfg.get("max_weight", 0.10))
    cash_buffer = float(cfg.get("cash_buffer", 0.0))
    max_gross = float(cfg.get("max_gross_exposure", 1.0)) * (1.0 - cash_buffer)
    max_net = float(cfg.get("max_net_exposure", max_gross))
    scheme = str(cfg.get("scheme", "inverse_vol"))
    inverse_vol = bool(cfg.get("inverse_vol_scaling", scheme == "inverse_vol"))
    vol_column = f"vol_{int(cfg.get('vol_lookback', 20))}"
    turnover_cap = cfg.get("turnover_cap")
    turnover_cap = None if turnover_cap is None else float(turnover_cap)

    frame = signals[ID_COLUMNS + ["signal"]].copy()
    if features is not None and vol_column in features.columns:
        frame = frame.merge(features[ID_COLUMNS + [vol_column]], on=ID_COLUMNS, how="left")
    else:
        frame[vol_column] = np.nan

    rows = []
    prev = pd.Series(dtype=float)
    for date, g in frame.groupby("date", sort=True):
        scores = g.set_index("symbol")["signal"].astype(float)
        if scheme == "equal_weight":
            scores = np.sign(scores)
        if inverse_vol:
            vol = g.set_index("symbol")[vol_column].replace(0, np.nan)
            scores = scores / vol.fillna(vol.median()).fillna(1.0)
        weights = _normalize_scores(scores, max_gross=max_gross)
        weights = _apply_caps(
            weights,
            max_weight=max_weight,
            max_gross=max_gross,
            max_net=max_net,
        )

        if turnover_cap is not None and not prev.empty:
            all_symbols = prev.index.union(weights.index)
            delta = weights.reindex(all_symbols, fill_value=0.0) - prev.reindex(
                all_symbols, fill_value=0.0
            )
            turnover = float(delta.abs().sum())
            if turnover > turnover_cap > 0:
                scaled = prev.reindex(all_symbols, fill_value=0.0) + delta * (
                    turnover_cap / turnover
                )
                weights = scaled.reindex(weights.index, fill_value=0.0)

        prev = weights.copy()
        block = weights.rename("target_weight").reset_index()
        block["date"] = date
        rows.append(block[["date", "symbol", "target_weight"]])

    if not rows:
        return pd.DataFrame(columns=ID_COLUMNS + ["target_weight"])
    return pd.concat(rows, ignore_index=True).sort_values(ID_COLUMNS).reset_index(drop=True)
