"""Feature pipeline: panel in, feature matrix out — with train-only scaling.

Output layout: one row per (date, symbol) with feature columns. The
benchmark symbol contributes market-regime features but is excluded from
the tradable rows.
"""

from __future__ import annotations

import pandas as pd

from alphaforge.features.cache import FeatureCache, FeatureSet, build_feature_lineage
from alphaforge.features.registry import (
    FeatureContractError,
    build_default_registry,
    validate_feature_frame,
)
from alphaforge.features.technical import (
    compute_benchmark_relative,
    compute_market_regime,
    compute_symbol_features,
)
from alphaforge.models.regime import causal_stress_probability

ID_COLUMNS = ["date", "symbol"]


def build_features(
    panel: pd.DataFrame,
    benchmark_symbol: str,
    config: dict | None = None,
) -> pd.DataFrame:
    """Build and contract-validate the causal feature matrix."""
    cfg = config or {}
    registry = build_default_registry(cfg)
    features = _build_features(panel, benchmark_symbol, cfg)
    validate_feature_frame(features, registry)
    return features


def _build_features(
    panel: pd.DataFrame,
    benchmark_symbol: str,
    cfg: dict,
) -> pd.DataFrame:
    """Execute feature mathematics after registry construction succeeds."""

    panel = panel.sort_values(["symbol", "date"]).reset_index(drop=True)

    bench_bars = panel[panel["symbol"] == benchmark_symbol]
    if bench_bars.empty:
        raise ValueError(f"benchmark symbol {benchmark_symbol!r} not found in panel")
    regime = compute_market_regime(bench_bars, cfg)
    bench_ret = regime[["date", "bench_ret"]]

    frames = []
    for symbol, g in panel.groupby("symbol"):
        if symbol == benchmark_symbol:
            continue
        g = g.sort_values("date")
        feats = compute_symbol_features(g, cfg)
        rel = compute_benchmark_relative(g, bench_ret, cfg)
        block = pd.concat(
            [
                g[ID_COLUMNS].reset_index(drop=True),
                feats.reset_index(drop=True),
                rel.reset_index(drop=True),
            ],
            axis=1,
        )
        frames.append(block)

    features = pd.concat(frames, ignore_index=True)
    features = features.merge(regime.drop(columns=["bench_ret"]), on="date", how="left")

    if cfg.get("hmm_regime", True):
        # causal HMM stress probability: expanding refits + filtered inference
        stress = causal_stress_probability(
            regime["bench_ret"],
            refit_every=int(cfg.get("hmm_refit_every", 63)),
            min_train=int(cfg.get("hmm_min_train", 252)),
        )
        hmm_frame = pd.DataFrame(
            {"date": regime["date"].to_numpy(), "hmm_stress_prob": stress.to_numpy()}
        )
        features = features.merge(hmm_frame, on="date", how="left")

    if cfg.get("cross_sectional", True):
        features = _add_cross_sectional(features)

    features["date"] = pd.to_datetime(features["date"]).astype("datetime64[ns]")
    return features.sort_values(ID_COLUMNS).reset_index(drop=True)


def materialize_feature_set(
    panel: pd.DataFrame,
    benchmark_symbol: str,
    config: dict | None = None,
    *,
    dataset_id: str,
    code_version: str,
    cache: FeatureCache | None = None,
) -> FeatureSet:
    """Materialize registered features with full lineage and optional caching.

    The cache key binds the declared dataset reference, panel content, code,
    semantic feature parameters, date interval, universe, and registry. Cache
    misses execute the same validated feature path as :func:`build_features`.
    """

    cfg = config or {}
    registry = build_default_registry(cfg)
    unique_dates = pd.Index(pd.to_datetime(panel["date"], errors="raise").unique())
    required = registry.required_warmup_sessions + 1
    if len(unique_dates) < required:
        raise FeatureContractError(
            f"insufficient feature warm-up history: {len(unique_dates)} < {required}"
        )
    lineage = build_feature_lineage(
        panel,
        registry,
        cfg,
        dataset_id=dataset_id,
        code_version=code_version,
    )
    if cache is not None:
        cached = cache.load(lineage, registry)
        if cached is not None:
            return FeatureSet(cached, registry, lineage, lineage.cache_key, True)
    frame = _build_features(panel, benchmark_symbol, cfg)
    validate_feature_frame(frame, registry)
    if cache is not None:
        cache.store(frame, lineage, registry)
    return FeatureSet(frame, registry, lineage, lineage.cache_key, False)


def _add_cross_sectional(features: pd.DataFrame) -> pd.DataFrame:
    """Per-date cross-sectional ranks in [0, 1]. Uses only same-date data."""
    for col in ["momentum_20", "momentum_60", "vol_20", "ret_5", "rolling_sharpe"]:
        if col in features.columns:
            features[f"cs_rank_{col}"] = features.groupby("date")[col].rank(pct=True)
    return features


def feature_columns(features: pd.DataFrame) -> list[str]:
    """All model input columns (everything except identifiers)."""
    return [c for c in features.columns if c not in ID_COLUMNS]


class FeatureScaler:
    """Z-score scaler whose statistics are fit on training rows only.

    Fitting on the full panel would leak test-period distribution information
    into training — a classic subtle leak. This class makes the train-only
    contract explicit and testable.
    """

    def __init__(self, clip: float = 5.0):
        self.clip = clip
        self.means_: pd.Series | None = None
        self.stds_: pd.Series | None = None
        self.columns_: list[str] | None = None

    def fit(self, X: pd.DataFrame) -> FeatureScaler:
        self.columns_ = list(X.columns)
        self.means_ = X.mean()
        self.stds_ = X.std().replace(0, 1.0)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if self.means_ is None or self.stds_ is None:
            raise RuntimeError("FeatureScaler must be fit before transform")
        Z = (X[self.columns_] - self.means_) / self.stds_
        return Z.clip(-self.clip, self.clip)

    def fit_transform(self, X: pd.DataFrame) -> pd.DataFrame:
        return self.fit(X).transform(X)
