"""Versioned financial-label contract, mathematics, and boundary tests."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from alphaforge.labels import (
    LabelBoundaryError,
    LabelContract,
    LabelContractError,
    LabelDefinition,
    build_label_set,
)


def _contract(*definitions: LabelDefinition, **kwargs: object) -> LabelContract:
    return LabelContract(
        benchmark_symbol="BENCH",
        definitions=tuple(definitions),
        **kwargs,  # type: ignore[arg-type]
    )


def _definition(
    name: str = "return_2",
    kind: str = "regression",
    horizon: int = 2,
    **kwargs: object,
) -> LabelDefinition:
    return LabelDefinition(
        name=name,
        kind=kind,  # type: ignore[arg-type]
        horizon=horizon,
        **kwargs,  # type: ignore[arg-type]
    )


def _panel(
    closes: dict[str, list[float]],
    *,
    start: str = "2024-01-02",
) -> pd.DataFrame:
    dates = pd.bdate_range(start, periods=len(next(iter(closes.values()))))
    rows = [
        {"date": date, "symbol": symbol, "close": close}
        for symbol, values in closes.items()
        for date, close in zip(dates, values, strict=True)
    ]
    return pd.DataFrame(rows)


def test_definition_identity_records_complete_semantics() -> None:
    definition = _definition(
        name="barrier",
        kind="triple_barrier",
        horizon=5,
        upper_barrier=0.03,
        lower_barrier=0.02,
        overlap_policy="non_overlapping",
    )
    record = definition.to_dict()

    assert len(definition.identity) == 64
    assert record["identity"] == definition.identity
    assert record["timing_convention"] == "close_to_close"
    assert record["required_future_interval"] == "(t, t+h]"
    assert record["horizon_sessions"] == 5
    assert record["overlap_policy"] == "non_overlapping"
    assert record["parameters"] == {"lower_barrier": 0.02, "upper_barrier": 0.03}
    assert replace(definition, upper_barrier=0.04).identity != definition.identity


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"kind": "triple_barrier"}, "requires parameters"),
        (
            {
                "kind": "triple_barrier",
                "upper_barrier": 0.0,
                "lower_barrier": 0.02,
            },
            "upper_barrier",
        ),
        ({"kind": "quantile", "quantiles": 1}, "quantiles"),
        (
            {
                "kind": "meta_label",
                "threshold": 0.0,
                "side_source": "column",
            },
            "side_column",
        ),
        ({"kind": "regression", "threshold": 0.1}, "unsupported parameters"),
        ({"kind": "regression", "horizon": 2521}, "horizon"),
        ({"kind": "unsupported"}, "unsupported label kind"),
    ],
)
def test_invalid_definition_fails_closed(kwargs: dict[str, object], message: str) -> None:
    payload: dict[str, object] = {
        "name": "invalid",
        "kind": "regression",
        "horizon": 2,
    }
    payload.update(kwargs)
    with pytest.raises(LabelContractError, match=message):
        LabelDefinition(**payload)  # type: ignore[arg-type]


def test_contract_rejects_unsupported_policy_and_unbounded_definition_count() -> None:
    definition = _definition()
    with pytest.raises(LabelContractError, match="missing-price policy"):
        LabelContract(
            benchmark_symbol="BENCH",
            definitions=(definition,),
            missing_price_policy="nan",  # type: ignore[arg-type]
        )

    definitions = tuple(_definition(name=f"return_{position}") for position in range(65))
    with pytest.raises(LabelContractError, match="between 1 and 64"):
        LabelContract(benchmark_symbol="BENCH", definitions=definitions)


def test_reference_label_mathematics_and_future_intervals() -> None:
    panel = _panel(
        {
            "BENCH": [100, 100, 100, 100, 100, 100],
            "AAA": [100, 101, 104, 102, 106, 108],
            "BBB": [100, 99, 98, 97, 96, 95],
            "CCC": [100, 100, 101, 101, 102, 102],
        }
    )
    contract = _contract(
        _definition("return_2", "regression", 2),
        _definition("direction_2", "classification", 2),
        _definition("threshold_2", "threshold", 2, threshold=0.02),
        _definition("quantile_2", "quantile", 2, quantiles=3),
        _definition(
            "barrier_2",
            "triple_barrier",
            2,
            upper_barrier=0.03,
            lower_barrier=0.02,
        ),
        _definition("scaled_2", "volatility_scaled", 2, volatility_window=2),
        _definition(
            "meta_2",
            "meta_label",
            2,
            threshold=0.005,
            side_source="lagged_return",
        ),
    )
    result = build_label_set(panel, contract)
    aaa = result.values.loc[result.values["symbol"].eq("AAA")].reset_index(drop=True)

    assert aaa.loc[0, "return_2"] == pytest.approx(0.04)
    assert aaa.loc[0, "direction_2"] == 1.0
    assert aaa.loc[0, "threshold_2"] == 1.0
    assert aaa.loc[0, "quantile_2"] == 3.0
    assert aaa.loc[0, "barrier_2"] == 1.0
    assert pd.isna(aaa.loc[0, "scaled_2"])
    assert aaa.loc[2, "scaled_2"] == pytest.approx(
        (106 / 104 - 1)
        / (pd.Series([100, 101, 104]).pct_change().rolling(2).std().iloc[-1] * np.sqrt(2))
    )
    assert aaa.loc[1, "meta_2"] == 1.0

    event = result.events.loc[
        result.events["label"].eq("return_2")
        & result.events["symbol"].eq("AAA")
        & result.events["date"].eq(aaa.loc[0, "date"])
    ].iloc[0]
    assert event["required_future_start"] == aaa.loc[1, "date"]
    assert event["required_future_end"] == aaa.loc[2, "date"]
    assert event["observable"]
    assert result.manifest()["contract"]["identity"] == contract.identity


def test_triple_barrier_uses_first_close_hit_not_terminal_sign() -> None:
    panel = _panel(
        {
            "BENCH": [100, 100, 100, 100, 100],
            "AAA": [100, 103, 97, 105, 105],
        }
    )
    definition = _definition(
        "barrier",
        "triple_barrier",
        3,
        upper_barrier=0.02,
        lower_barrier=0.02,
    )
    result = build_label_set(panel, _contract(definition))
    assert result.values.loc[0, "barrier"] == 1.0


def test_non_overlapping_policy_masks_intersecting_events() -> None:
    panel = _panel(
        {
            "BENCH": [100] * 8,
            "AAA": [100, 101, 102, 103, 104, 105, 106, 107],
        }
    )
    definition = _definition(
        "return_3",
        "regression",
        3,
        overlap_policy="non_overlapping",
    )
    result = build_label_set(panel, _contract(definition))
    observable = result.events.loc[result.events["observable"], "date"].tolist()

    assert observable == list(pd.bdate_range("2024-01-02", periods=2, freq="3B"))
    assert result.values["return_3"].notna().sum() == 2


def test_future_mutation_is_isolated_by_symbol() -> None:
    closes: dict[str, list[float]] = {
        "BENCH": [100] * 10,
        "AAA": list(np.linspace(100, 109, 10)),
        "BBB": list(np.linspace(100, 91, 10)),
    }
    panel = _panel(closes)
    contract = _contract(_definition())
    baseline = build_label_set(panel, contract).values
    mutated = panel.copy()
    mutated.loc[mutated["symbol"].eq("BBB"), "close"] *= np.linspace(1, 10, 10)
    changed = build_label_set(mutated, contract).values

    pd.testing.assert_series_equal(
        baseline.loc[baseline["symbol"].eq("AAA"), "return_2"].reset_index(drop=True),
        changed.loc[changed["symbol"].eq("AAA"), "return_2"].reset_index(drop=True),
    )


def test_protected_holdout_crossing_fails_before_publication() -> None:
    panel = _panel({"BENCH": [100] * 8, "AAA": list(range(100, 108))})
    boundary = pd.Timestamp("2024-01-09")
    contract = _contract(_definition(horizon=3), protected_boundaries=(boundary,))

    with pytest.raises(LabelBoundaryError, match="crosses protected boundary"):
        build_label_set(panel, contract)


def test_unavailable_nonfinite_duplicate_and_horizon_overflow_fail() -> None:
    panel = _panel(
        {
            "BENCH": [100] * 6,
            "AAA": [100, 101, 102, 103, 104, 105],
        }
    )
    contract = _contract(_definition(horizon=2))

    missing = panel.drop(
        panel.index[(panel["symbol"].eq("AAA")) & (panel["date"].eq(pd.Timestamp("2024-01-04")))]
    )
    with pytest.raises(LabelContractError, match="unavailable close prices"):
        build_label_set(missing, contract)

    nonfinite = panel.copy()
    nonfinite["close"] = nonfinite["close"].astype(float)
    nonfinite.loc[nonfinite["symbol"].eq("AAA"), "close"] = np.inf
    with pytest.raises(LabelContractError, match="finite and strictly positive"):
        build_label_set(nonfinite, contract)

    duplicate = pd.concat([panel, panel.iloc[[0]]], ignore_index=True)
    with pytest.raises(LabelContractError, match="duplicate"):
        build_label_set(duplicate, contract)

    with pytest.raises(LabelContractError, match="overflows"):
        build_label_set(panel, _contract(_definition(horizon=6)))


def test_meta_label_column_side_is_validated() -> None:
    panel = _panel({"BENCH": [100] * 5, "AAA": [100, 101, 102, 103, 104]})
    definition = _definition(
        "meta",
        "meta_label",
        1,
        threshold=0.0,
        side_source="column",
        side_column="primary_side",
    )
    with pytest.raises(LabelContractError, match="unavailable"):
        build_label_set(panel, _contract(definition))

    panel["primary_side"] = 1.0
    panel.loc[panel["symbol"].eq("AAA") & panel["date"].eq(panel["date"].max()), "primary_side"] = 0
    with pytest.raises(LabelContractError, match="only -1 or 1"):
        build_label_set(panel, _contract(definition))
