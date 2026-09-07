"""Statistical diagnostics for governed financial-label datasets."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np
import pandas as pd

from alphaforge.labels.contracts import LabelDataset

CATEGORICAL_KINDS = {
    "classification",
    "threshold",
    "quantile",
    "triple_barrier",
    "meta_label",
}


@dataclass(frozen=True)
class LabelDiagnostics:
    """Machine-readable evidence for dependence, balance, stability, and sensitivity."""

    summary: pd.DataFrame
    class_balance: pd.DataFrame
    temporal_stability: pd.DataFrame
    parameter_sensitivity: pd.DataFrame


def _weighted_autocorrelation(frame: pd.DataFrame, column: str, lag: int) -> tuple[float, int]:
    correlations: list[float] = []
    weights: list[int] = []
    for _, group in frame.groupby("symbol", sort=True):
        values = group.sort_values("date")[column].dropna()
        if len(values) <= lag + 1 or values.nunique() < 2:
            continue
        correlation = values.autocorr(lag=lag)
        if pd.notna(correlation):
            correlations.append(float(correlation))
            weights.append(len(values) - lag)
    if not correlations:
        return 0.0, 0
    return float(np.average(correlations, weights=weights)), int(sum(weights))


def _overlap_rate(events: pd.DataFrame) -> float:
    comparisons = 0
    overlaps = 0
    for _, group in events.loc[events["observable"]].groupby("symbol", sort=True):
        ordered = group.sort_values("date")
        previous_end = ordered["required_future_end"].shift(1)
        comparable = previous_end.notna()
        comparisons += int(comparable.sum())
        overlaps += int(ordered.loc[comparable, "date"].lt(previous_end[comparable]).sum())
    return float(overlaps / comparisons) if comparisons else 0.0


def _effective_sample_size(sample_size: int, autocorrelation: float) -> float:
    if sample_size <= 0:
        return 0.0
    bounded = float(np.clip(autocorrelation, -0.99, 0.99))
    estimate = sample_size * (1.0 - bounded) / (1.0 + bounded)
    return float(np.clip(estimate, 1.0, sample_size))


def _summary(dataset: LabelDataset, lag: int) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    total_rows = len(dataset.values)
    for definition in dataset.contract.definitions:
        values = dataset.values[["date", "symbol", definition.name]]
        valid = values[definition.name].dropna()
        autocorrelation, dependent_pairs = _weighted_autocorrelation(values, definition.name, lag)
        events = dataset.events.loc[dataset.events["label"].eq(definition.name)]
        sample_size = int(len(valid))
        rows.append(
            {
                "label": definition.name,
                "label_id": definition.identity,
                "kind": definition.kind,
                "horizon_sessions": definition.horizon,
                "observations": sample_size,
                "missing_fraction": float(1.0 - sample_size / total_rows),
                "overlap_rate": _overlap_rate(events),
                f"autocorrelation_lag_{lag}": autocorrelation,
                "autocorrelation_pairs": dependent_pairs,
                "effective_sample_size": _effective_sample_size(sample_size, autocorrelation),
                "mean": float(valid.mean()) if sample_size else np.nan,
                "standard_deviation": float(valid.std(ddof=1)) if sample_size > 1 else np.nan,
            }
        )
    return pd.DataFrame(rows).sort_values("label").reset_index(drop=True)


def _class_balance(dataset: LabelDataset) -> pd.DataFrame:
    definitions = {definition.name: definition for definition in dataset.contract.definitions}
    rows: list[dict[str, object]] = []
    for name, definition in definitions.items():
        if definition.kind not in CATEGORICAL_KINDS:
            continue
        counts = dataset.values[name].dropna().value_counts().sort_index()
        total = int(counts.sum())
        for value, count in counts.items():
            rows.append(
                {
                    "label": name,
                    "label_id": definition.identity,
                    "class": str(int(value)) if float(value).is_integer() else str(value),
                    "count": int(count),
                    "fraction": float(count / total),
                }
            )
    return pd.DataFrame(
        rows, columns=["label", "label_id", "class", "count", "fraction"]
    ).sort_values(["label", "class"], ignore_index=True)


def _temporal_stability(dataset: LabelDataset, periods: int) -> pd.DataFrame:
    unique_dates = pd.Index(sorted(dataset.values["date"].unique()))
    period_by_date = {
        date: min(periods - 1, position * periods // len(unique_dates))
        for position, date in enumerate(unique_dates)
    }
    rows: list[dict[str, object]] = []
    for definition in dataset.contract.definitions:
        frame = dataset.values[["date", definition.name]].dropna().copy()
        frame["period"] = frame["date"].map(period_by_date)
        for period, group in frame.groupby("period", sort=True):
            values = group[definition.name]
            rows.append(
                {
                    "label": definition.name,
                    "label_id": definition.identity,
                    "period": int(period) + 1,
                    "periods_total": periods,
                    "date_start": pd.Timestamp(group["date"].min()).date().isoformat(),
                    "date_end": pd.Timestamp(group["date"].max()).date().isoformat(),
                    "observations": int(len(values)),
                    "mean": float(values.mean()),
                    "standard_deviation": float(values.std(ddof=1)) if len(values) > 1 else np.nan,
                }
            )
    return pd.DataFrame(rows).sort_values(["label", "period"]).reset_index(drop=True)


def _parameter_sensitivity(
    baseline: LabelDataset,
    variants: Mapping[float, LabelDataset],
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    keys = ["date", "symbol"]
    definitions = {definition.name: definition for definition in baseline.contract.definitions}
    for scale, variant in sorted(variants.items()):
        if not np.isfinite(scale) or scale <= 0:
            raise ValueError("sensitivity variant scales must be finite and positive")
        other_definitions = {
            definition.name: definition for definition in variant.contract.definitions
        }
        if set(other_definitions) != set(definitions):
            raise ValueError("sensitivity variants must contain the exact baseline label names")
        for name, definition in definitions.items():
            other = other_definitions[name]
            if other.identity == definition.identity and scale != 1.0:
                continue
            merged = baseline.values[keys + [name]].merge(
                variant.values[keys + [name]],
                on=keys,
                how="inner",
                suffixes=("_baseline", "_variant"),
                validate="one_to_one",
            )
            if len(merged) != len(baseline.values) or len(merged) != len(variant.values):
                raise ValueError(
                    "sensitivity variants must contain the exact baseline observation keys"
                )
            left = merged[f"{name}_baseline"]
            right = merged[f"{name}_variant"]
            valid = left.notna() & right.notna()
            left = left.loc[valid]
            right = right.loc[valid]
            if len(left) > 1 and left.nunique() > 1 and right.nunique() > 1:
                correlation = float(left.corr(right))
            else:
                correlation = 1.0 if left.equals(right) else np.nan
            categorical = definition.kind in CATEGORICAL_KINDS
            rows.append(
                {
                    "label": name,
                    "baseline_label_id": definition.identity,
                    "variant_label_id": other.identity,
                    "parameter_scale": float(scale),
                    "observations": int(len(left)),
                    "correlation": correlation,
                    "mean_absolute_change": (
                        float((right - left).abs().mean()) if len(left) else np.nan
                    ),
                    "class_flip_rate": (
                        float(left.ne(right).mean()) if categorical and len(left) else np.nan
                    ),
                }
            )
    columns = [
        "label",
        "baseline_label_id",
        "variant_label_id",
        "parameter_scale",
        "observations",
        "correlation",
        "mean_absolute_change",
        "class_flip_rate",
    ]
    return pd.DataFrame(rows, columns=columns).sort_values(
        ["label", "parameter_scale"], ignore_index=True
    )


def diagnose_labels(
    dataset: LabelDataset,
    *,
    autocorrelation_lag: int = 1,
    periods: int = 4,
    sensitivity_variants: Mapping[float, LabelDataset] | None = None,
) -> LabelDiagnostics:
    """Compute deterministic reference diagnostics.

    ``sensitivity_variants`` must contain datasets built from the same panel and
    label names with deliberately perturbed parameters.  The caller controls
    which research parameters are examined; this function never searches for a
    favorable label.
    """

    if autocorrelation_lag < 1:
        raise ValueError("autocorrelation_lag must be positive")
    if periods < 2:
        raise ValueError("periods must be at least two")
    if periods > dataset.values["date"].nunique():
        raise ValueError("periods cannot exceed the number of observed session dates")
    variants = sensitivity_variants or {1.0: dataset}
    return LabelDiagnostics(
        summary=_summary(dataset, autocorrelation_lag),
        class_balance=_class_balance(dataset),
        temporal_stability=_temporal_stability(dataset, periods),
        parameter_sensitivity=_parameter_sensitivity(dataset, variants),
    )
