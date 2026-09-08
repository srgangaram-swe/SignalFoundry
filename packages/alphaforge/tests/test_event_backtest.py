"""Integration tests for the event-sourced historical backtest boundary.

All fixtures are deterministic and synthetic.  The tests intentionally cover
the public ``run_backtest`` contract rather than rebuilding event-engine unit
tests: legacy research tables must remain stable while the appended event and
accounting tables make execution replayable and reconcilable.
"""

from __future__ import annotations

from dataclasses import fields

import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from alphaforge.backtesting import BacktestResult, run_backtest
from alphaforge.backtesting.event_engine import DeterministicEventEngine
from alphaforge.backtesting.journal import InMemoryJournal, Journal, SQLiteJournal
from alphaforge.execution.events import (
    CashChargeAccrued,
    EngineHalted,
    FillApplied,
    PortfolioMarked,
    TargetDecided,
)
from alphaforge.execution.frictions import standard_stress_profiles

_RUN_ID = "synthetic-event-backtest"
_NO_COSTS = {
    "commission_bps": 0.0,
    "half_spread_bps": 0.0,
    "slippage_bps": 0.0,
}
_LEGACY_FIELDS = (
    "equity_curve",
    "weights",
    "trades",
    "orders",
    "fills",
    "pnl_attribution",
)
_LEGACY_COLUMNS = {
    "equity_curve": [
        "date",
        "gross_return",
        "transaction_cost",
        "return",
        "equity",
        "cash",
        "market_pnl",
        "overnight_pnl",
        "intraday_pnl",
        "trading_cost",
        "benchmark_return",
        "turnover",
        "traded_notional",
        "gross_exposure",
        "net_exposure",
        "leverage",
        "active",
    ],
    "weights": [
        "date",
        "symbol",
        "shares",
        "mark_price",
        "market_value",
        "weight",
        "target_weight",
    ],
    "trades": [
        "date",
        "symbol",
        "trade_weight",
        "filled_shares",
        "reference_price",
        "fill_price",
        "traded_notional",
        "total_cost",
        "decision_date",
        "status",
        "residual_shares",
        "participation_rate",
    ],
    "orders": [
        "order_id",
        "symbol",
        "decision_date",
        "fill_date",
        "requested_shares",
        "requested_notional",
        "target_weight",
        "pretrade_equity",
    ],
    "fills": [
        "order_id",
        "symbol",
        "decision_date",
        "fill_date",
        "status",
        "requested_shares",
        "filled_shares",
        "residual_shares",
        "reference_price",
        "fill_price",
        "target_weight",
        "pretrade_equity",
        "lagged_adv_shares",
        "lagged_volatility",
        "participation_rate",
        "commission",
        "spread_cost",
        "fixed_slippage_cost",
        "impact_cost",
        "impact_bps",
        "traded_notional",
        "total_cost",
        "requested_notional",
        "lagged_adv_notional",
    ],
    "pnl_attribution": [
        "date",
        "symbol",
        "overnight_pnl",
        "intraday_pnl",
        "market_pnl",
        "trading_cost",
        "net_pnl",
    ],
}
_EVENT_COLUMNS = [
    "event_id",
    "run_id",
    "session",
    "bar_index",
    "phase",
    "ordinal",
    "event_type",
    "correlation_id",
    "entity_id",
    "causation_id",
]
_ACCOUNTING_COLUMNS = [
    "date",
    "cash",
    "equity",
    "gross_exposure",
    "net_exposure",
    "realized_pnl",
    "unrealized_pnl",
    "fees",
    "financing",
    "borrow",
    "other_charges",
    "total_charges",
    "net_pnl",
    "reconciliation_error",
    "reconciliation_tolerance",
    "bankrupt",
]
_FRICTION_MANIFEST_COLUMNS = [
    "model_type",
    "model_id",
    "version",
    "declaration_digest",
    "configuration_digest",
    "parameters_json",
    "units_json",
    "parameter_bounds_json",
    "calibration_provenance",
    "domain",
    "execution_timestamp",
    "failure_behavior",
    "limitations_json",
    "stress_profile",
]
_FRICTION_ATTRIBUTION_COLUMNS = [
    "date",
    "order_id",
    "symbol",
    "component",
    "accounting_path",
    "amount_usd",
    "rate",
    "rate_unit",
    "basis_usd",
    "model_digest",
    "record_digest",
    "stress_profile",
    "status",
    "detail",
]
_LATENCY_COLUMNS = [
    "origin_session",
    "data_available_session",
    "feature_available_session",
    "signal_available_session",
    "submission_session",
    "fill_session",
    "execution_lag_sessions",
    "model_digest",
    "schedule_digest",
    "stress_profile",
]


def _panel(
    symbols: tuple[str, ...] = ("AAA", "BBB"),
    *,
    periods: int = 6,
    start: str = "2025-01-02",
    price: float = 100.0,
    volume: float = 1_000_000.0,
) -> pd.DataFrame:
    dates = pd.bdate_range(start, periods=periods)
    return pd.DataFrame(
        [
            {
                "date": date,
                "symbol": symbol,
                "open": price,
                "high": price,
                "low": price,
                "close": price,
                "volume": volume,
            }
            for symbol in symbols
            for date in dates
        ]
    )


def _targets(
    panel: pd.DataFrame,
    weights: dict[str, float] | None = None,
    *,
    decision_index: int = 0,
) -> pd.DataFrame:
    sessions = pd.DatetimeIndex(sorted(panel["date"].unique()))
    chosen = weights or {"AAA": 0.6, "BBB": 0.4}
    return pd.DataFrame(
        [
            {
                "date": sessions[decision_index],
                "symbol": symbol,
                "target_weight": weight,
            }
            for symbol, weight in chosen.items()
        ]
    )


def _run(
    panel: pd.DataFrame,
    targets: pd.DataFrame,
    *,
    execution: dict[str, object] | None = None,
    execution_lag: int = 1,
    event_journal: Journal | None = None,
) -> BacktestResult:
    return run_backtest(
        panel,
        targets,
        costs=_NO_COSTS,
        execution=execution,
        execution_lag=execution_lag,
        event_journal=event_journal,
        event_run_id=_RUN_ID,
    )


def test_event_extension_preserves_every_legacy_table_schema() -> None:
    panel = _panel()
    result = _run(panel, _targets(panel))

    assert tuple(field.name for field in fields(BacktestResult)) == (
        *_LEGACY_FIELDS,
        "events",
        "accounting",
        "friction_model_manifest",
        "friction_attribution",
        "latency_schedule",
    )
    for field_name, expected_columns in _LEGACY_COLUMNS.items():
        table = getattr(result, field_name)
        assert list(table.columns) == expected_columns
    assert list(result.events.columns) == _EVENT_COLUMNS
    assert list(result.accounting.columns) == _ACCOUNTING_COLUMNS
    assert list(result.friction_model_manifest.columns) == _FRICTION_MANIFEST_COLUMNS
    assert list(result.friction_attribution.columns) == _FRICTION_ATTRIBUTION_COLUMNS
    assert list(result.latency_schedule.columns) == _LATENCY_COLUMNS


def test_two_asset_orders_are_all_submitted_before_any_execution() -> None:
    panel = _panel()
    result = _run(panel, _targets(panel))
    execution_session = pd.DatetimeIndex(sorted(panel["date"].unique()))[1]
    events = result.events.loc[result.events["session"] == execution_session].reset_index(drop=True)

    submissions = events.index[events["event_type"] == "order_submitted"].tolist()
    executions = events.index[
        events["event_type"].isin(["order_accepted", "order_rejected", "fill_applied"])
    ].tolist()
    assert len(submissions) == 2
    assert len(executions) == 4
    assert max(submissions) < min(executions)
    assert events.loc[submissions, "entity_id"].is_unique
    assert events.loc[executions, "causation_id"].notna().all()


@pytest.mark.parametrize(
    ("decision_index", "execution", "expected_status", "expected_events"),
    [
        (
            0,
            {},
            "filled",
            {"order_submitted", "order_accepted", "fill_applied"},
        ),
        (
            8,
            {
                "adv_lookback": 5,
                "volatility_lookback": 2,
                "max_participation_rate": 0.10,
            },
            "partial",
            {
                "order_submitted",
                "order_accepted",
                "fill_applied",
                "order_cancelled",
            },
        ),
        (
            0,
            {
                "adv_lookback": 5,
                "volatility_lookback": 2,
                "max_participation_rate": 0.10,
            },
            "rejected",
            {"order_submitted", "order_rejected"},
        ),
    ],
)
def test_fill_outcomes_map_to_explicit_order_lifecycle_events(
    decision_index: int,
    execution: dict[str, object],
    expected_status: str,
    expected_events: set[str],
) -> None:
    panel = _panel(("AAA",), periods=12, volume=1_000.0)
    result = _run(
        panel,
        _targets(panel, {"AAA": 1.0}, decision_index=decision_index),
        execution=execution,
    )
    fill = result.fills.iloc[0]
    execution_session = pd.Timestamp(fill["fill_date"])
    lifecycle = result.events.loc[
        (result.events["session"] == execution_session)
        & result.events["event_type"].isin(
            [
                "order_submitted",
                "order_accepted",
                "order_rejected",
                "fill_applied",
                "order_cancelled",
            ]
        ),
        "event_type",
    ]

    assert fill["status"] == expected_status
    assert set(lifecycle) == expected_events


def test_large_partial_fill_emits_explicit_residual_cancellation() -> None:
    panel = _panel(("AAA",), periods=12, price=1.0, volume=1_000_000_000.0)
    result = run_backtest(
        panel,
        _targets(panel, {"AAA": 1.0}, decision_index=8),
        initial_capital=1_000_000_000.0,
        costs=_NO_COSTS,
        execution={
            "adv_lookback": 5,
            "volatility_lookback": 2,
            "max_participation_rate": 0.999999,
        },
        event_run_id=_RUN_ID,
    )

    fill = result.fills.iloc[0]
    assert fill["status"] == "partial"
    assert fill["filled_shares"] == 999_999_000.0
    assert fill["residual_shares"] == 1_000.0
    execution_events = result.events.loc[
        result.events["session"] == pd.Timestamp(fill["fill_date"]), "event_type"
    ].tolist()
    assert "order_cancelled" in execution_events
    assert not bool(result.accounting.iloc[-1]["bankrupt"])


def test_repeated_run_is_byte_semantic_deterministic() -> None:
    panel = _panel()
    targets = _targets(panel)
    first = _run(panel, targets)
    second = _run(panel, targets)

    for field_name in (
        *_LEGACY_FIELDS,
        "events",
        "accounting",
        "friction_model_manifest",
        "friction_attribution",
        "latency_schedule",
    ):
        assert_frame_equal(
            getattr(first, field_name),
            getattr(second, field_name),
            check_exact=True,
        )


def test_lagged_target_is_not_eligible_before_its_frozen_session_and_replays() -> None:
    panel = _panel(("AAA",), periods=6)
    sessions = pd.DatetimeIndex(sorted(panel["date"].unique()))
    journal = InMemoryJournal()
    result = _run(
        panel,
        _targets(panel, {"AAA": 1.0}),
        execution_lag=3,
        event_journal=journal,
    )

    target_events = []
    for event in journal.events():
        if isinstance(event.payload, TargetDecided):
            target_events.append(event)
    submitted = result.events[result.events["event_type"] == "order_submitted"]
    assert len(target_events) == 1
    target_payload = target_events[0].payload
    assert isinstance(target_payload, TargetDecided)
    assert target_payload.eligible_session == sessions[3].date()
    assert submitted["session"].tolist() == [sessions[3]]
    assert submitted.iloc[0]["causation_id"] == target_events[0].event_id

    replayed = DeterministicEventEngine.replay(
        journal,
        calendar=tuple(session.date() for session in sessions),
        initial_cash=1_000_000.0,
        expected_run_id=_RUN_ID,
    )
    replayed_state = replayed.snapshot()
    latest = result.accounting.iloc[-1]
    assert replayed_state.processed_events == len(result.events)
    assert replayed_state.portfolio is not None
    assert replayed_state.portfolio.cash == latest["cash"]
    assert replayed_state.portfolio.equity == latest["equity"]


def test_cash_only_session_has_marks_but_no_order_and_commission_is_debited_once() -> None:
    panel = _panel(("AAA",), periods=4)
    sessions = pd.DatetimeIndex(sorted(panel["date"].unique()))
    result = run_backtest(
        panel,
        _targets(panel, {"AAA": 1.0}),
        initial_capital=1_000.0,
        costs={
            "commission_bps": 100.0,
            "half_spread_bps": 0.0,
            "slippage_bps": 0.0,
        },
        event_run_id=_RUN_ID,
    )

    first_session_events = result.events.loc[
        result.events["session"] == sessions[0], "event_type"
    ].tolist()
    first_account = result.accounting.loc[result.accounting["date"] == sessions[0]].iloc[0]
    assert "order_submitted" not in first_session_events
    assert {"portfolio_marked", "signal_available", "target_decided"}.issubset(first_session_events)
    assert first_account["cash"] == 1_000.0
    assert first_account["equity"] == 1_000.0

    commission = float(result.fills["commission"].sum())
    assert commission == pytest.approx(10.0)
    assert result.accounting.iloc[-1]["fees"] == pytest.approx(commission)
    assert result.accounting.iloc[-1]["total_charges"] == pytest.approx(commission)
    assert result.equity_curve.iloc[-1]["equity"] == pytest.approx(1_000.0 - commission)
    assert result.equity_curve["trading_cost"].sum() == pytest.approx(commission)


def test_component_costs_native_carry_and_attribution_reconcile() -> None:
    panel = _panel(("AAA", "BBB"), periods=12, volume=1_000_000.0)
    sessions = pd.DatetimeIndex(sorted(panel["date"].unique()))
    for index, session in enumerate(sessions):
        price = 100.0 + float(index % 2)
        selector = panel["date"] == session
        panel.loc[selector, ["open", "high", "low", "close"]] = price
    journal = InMemoryJournal()
    result = run_backtest(
        panel,
        _targets(panel, {"AAA": 1.4, "BBB": -0.4}, decision_index=5),
        initial_capital=100_000.0,
        costs={
            "commission_bps": 1.0,
            "commission_per_share_usd": 0.001,
            "exchange_fee_bps": 0.5,
            "half_spread_bps": 2.0,
            "slippage_bps": 1.0,
            "spread_slippage_multiplier": 0.5,
            "participation_slippage_bps": 5.0,
            "volatility_slippage_bps_per_1pct": 1.0,
            "calibration_provenance": "predeclared synthetic integration assumptions",
        },
        execution={
            "adv_lookback": 2,
            "volatility_lookback": 2,
            "impact_coefficient": 0.1,
            "impact_exponent": 0.5,
            "calibration_provenance": "predeclared synthetic integration assumptions",
        },
        carry={
            "cash_financing_bps_annual": 500.0,
            "short_borrow_bps_annual": 1_000.0,
            "sessions_per_year": 252,
            "calibration_provenance": "predeclared synthetic integration assumptions",
        },
        event_journal=journal,
        event_run_id=_RUN_ID,
    )

    components = set(result.friction_attribution["component"])
    assert {
        "commission",
        "exchange_fee",
        "spread",
        "fixed_slippage",
        "spread_slippage",
        "participation_slippage",
        "volatility_slippage",
        "market_impact",
        "cash_financing",
        "short_borrow",
    }.issubset(components)
    assert set(result.friction_model_manifest["model_type"]) == {
        "fill_cost",
        "execution_policy",
        "carry_cost",
        "latency",
        "stress_profile",
    }
    assert result.friction_model_manifest["declaration_digest"].str.len().eq(64).all()
    assert result.friction_model_manifest["configuration_digest"].str.len().eq(64).all()
    daily_components = result.friction_attribution.groupby("date")["amount_usd"].sum()
    curve_costs = result.equity_curve.set_index("date")["trading_cost"]
    pd.testing.assert_series_equal(
        daily_components.reindex(curve_costs.index, fill_value=0.0),
        curve_costs,
        check_names=False,
        rtol=1e-12,
        atol=1e-10,
    )
    latest = result.accounting.iloc[-1]
    cash_fees = result.friction_attribution.loc[
        result.friction_attribution["component"].isin(["commission", "exchange_fee"]),
        "amount_usd",
    ].sum()
    assert latest["fees"] == pytest.approx(cash_fees)
    assert latest["financing"] > 0.0
    assert latest["borrow"] > 0.0
    assert {"financing", "borrow"}.issubset(
        set(
            result.events.loc[result.events["event_type"] == "cash_charge_accrued", "entity_id"]
            .str.split("-")
            .str[0]
        )
    )
    replayed = DeterministicEventEngine.replay(
        journal,
        calendar=tuple(session.date() for session in sessions),
        initial_cash=100_000.0,
        expected_run_id=_RUN_ID,
    ).snapshot()
    assert replayed.portfolio is not None
    assert replayed.portfolio.cash == latest["cash"]
    assert replayed.portfolio.equity == latest["equity"]
    assert replayed.portfolio.charges["financing"] == latest["financing"]
    assert replayed.portfolio.charges["borrow"] == latest["borrow"]


def test_fill_attribution_uses_component_model_and_content_identities() -> None:
    panel = _panel(("AAA",), periods=10, volume=1_000_000.0)
    sessions = pd.DatetimeIndex(sorted(panel["date"].unique()))
    for index, session in enumerate(sessions):
        price = 100.0 + float(index % 2)
        selector = panel["date"] == session
        panel.loc[selector, ["open", "high", "low", "close"]] = price
    targets = _targets(panel, {"AAA": 1.0}, decision_index=4)
    common = {
        "panel": panel,
        "target_weights": targets,
        "costs": {
            "commission_bps": 1.0,
            "half_spread_bps": 2.0,
            "slippage_bps": 0.0,
        },
        "event_run_id": _RUN_ID,
    }
    lower = run_backtest(
        **common,
        execution={
            "adv_lookback": 2,
            "volatility_lookback": 2,
            "impact_coefficient": 0.1,
        },
    )
    higher = run_backtest(
        **common,
        execution={
            "adv_lookback": 2,
            "volatility_lookback": 2,
            "impact_coefficient": 0.2,
        },
    )

    lower_manifest = lower.friction_model_manifest.set_index("model_type")
    higher_manifest = higher.friction_model_manifest.set_index("model_type")
    lower_impact = lower.friction_attribution.loc[
        lower.friction_attribution["component"] == "market_impact"
    ].iloc[0]
    higher_impact = higher.friction_attribution.loc[
        higher.friction_attribution["component"] == "market_impact"
    ].iloc[0]
    lower_commission = lower.friction_attribution.loc[
        lower.friction_attribution["component"] == "commission"
    ].iloc[0]

    assert (
        lower_impact["model_digest"]
        == lower_manifest.at["execution_policy", "configuration_digest"]
    )
    assert (
        lower_commission["model_digest"] == lower_manifest.at["fill_cost", "configuration_digest"]
    )
    assert (
        lower_manifest.at["fill_cost", "configuration_digest"]
        == higher_manifest.at["fill_cost", "configuration_digest"]
    )
    assert lower_impact["model_digest"] != higher_impact["model_digest"]
    assert lower_impact["record_digest"] != higher_impact["record_digest"]
    assert higher_impact["amount_usd"] == pytest.approx(2.0 * lower_impact["amount_usd"])
    assert lower.friction_attribution["record_digest"].str.fullmatch(r"[0-9a-f]{64}").all()


def test_zero_lagged_liquidity_rejects_and_publishes_normalized_evidence() -> None:
    panel = _panel(("AAA",), periods=6, price=10.0, volume=0.0)
    result = run_backtest(
        panel,
        _targets(panel, {"AAA": 1.0}, decision_index=2),
        costs=_NO_COSTS,
        execution={"adv_lookback": 2, "max_participation_rate": 0.05},
        event_run_id=_RUN_ID,
    )

    assert result.fills["status"].tolist() == ["rejected"]
    rejection = result.friction_attribution.iloc[0]
    assert rejection["component"] == "execution_rejection"
    assert rejection["accounting_path"] == "none"
    assert rejection["amount_usd"] == 0.0
    assert rejection["detail"] == "required_lagged_adv_unavailable"
    assert len(rejection["model_digest"]) == 64
    assert len(rejection["record_digest"]) == 64
    assert "order_rejected" in set(result.events["event_type"])


def test_logical_latency_delays_each_stage_without_intraday_claims() -> None:
    panel = _panel(("AAA",), periods=10)
    sessions = pd.DatetimeIndex(sorted(panel["date"].unique()))
    result = run_backtest(
        panel,
        _targets(panel, {"AAA": 1.0}),
        costs=_NO_COSTS,
        latency={
            "data_delay_sessions": 1,
            "feature_delay_sessions": 1,
            "inference_delay_sessions": 1,
            "submission_delay_sessions": 1,
            "fill_delay_sessions": 1,
            "calibration_provenance": "predeclared synthetic logical delay",
        },
        event_run_id=_RUN_ID,
    )

    schedule = result.latency_schedule.iloc[0]
    assert schedule["origin_session"] == sessions[0]
    assert schedule["data_available_session"] == sessions[1]
    assert schedule["feature_available_session"] == sessions[2]
    assert schedule["signal_available_session"] == sessions[3]
    assert schedule["submission_session"] == sessions[4]
    assert schedule["fill_session"] == sessions[6]
    signal_sessions = result.events.loc[
        result.events["event_type"] == "signal_available", "session"
    ].tolist()
    submission_sessions = result.events.loc[
        result.events["event_type"] == "order_submitted", "session"
    ].tolist()
    assert signal_sessions == [sessions[3]]
    target_session = result.events.loc[
        result.events["event_type"] == "target_decided", "session"
    ].iloc[0]
    assert result.orders.iloc[0]["decision_date"] == target_session == sessions[3]
    assert result.orders.iloc[0]["decision_date"] >= schedule["feature_available_session"]
    # DAY lifecycle remains atomic on the future fill bar; submission_session
    # is a logical readiness diagnostic, not a fabricated resting order.
    assert submission_sessions == [sessions[6]]


def test_added_latency_fails_closed_when_a_baseline_eligible_target_exceeds_calendar() -> None:
    panel = _panel(("AAA",), periods=3)
    with pytest.raises(ValueError, match="exceeds the frozen calendar"):
        run_backtest(
            panel,
            _targets(panel, {"AAA": 1.0}, decision_index=1),
            costs=_NO_COSTS,
            latency={"fill_delay_sessions": 1},
        )


def test_every_standard_stress_profile_reruns_and_changes_its_event_path() -> None:
    panel = _panel(("AAA",), periods=12, price=10.0, volume=1_000.0)
    targets = _targets(panel, {"AAA": 1.0}, decision_index=3)
    profiles = standard_stress_profiles()
    results = {
        profile.name: run_backtest(
            panel,
            targets,
            initial_capital=1_000.0,
            costs={"commission_bps": 0.0, "half_spread_bps": 2.0, "slippage_bps": 0.0},
            execution={
                "adv_lookback": 2,
                "volatility_lookback": 2,
                "max_participation_rate": 0.05,
            },
            stress_profile=profile,
            event_run_id=_RUN_ID,
        )
        for profile in profiles
    }
    baseline = results["baseline"]
    doubled = results["doubled_costs"]
    tripled = results["tripled_costs"]
    adverse_spread = results["adverse_spread"]
    reduced_liquidity = results["reduced_liquidity"]
    delayed = results["delayed_signals"]
    partial_pressure = results["partial_fill_pressure"]
    capacity = results["capacity_scaling"]

    for name, result in results.items():
        assert result.friction_model_manifest["stress_profile"].eq(name).all()
        assert result.friction_attribution["stress_profile"].eq(name).all()
        assert {"order_submitted", "fill_applied", "order_cancelled"}.issubset(
            set(result.events["event_type"])
        )

    costs = [run.equity_curve["trading_cost"].sum() for run in (baseline, doubled, tripled)]
    assert costs == sorted(costs)
    assert costs[1] == pytest.approx(2.0 * costs[0])
    assert costs[2] == pytest.approx(3.0 * costs[0])
    assert adverse_spread.equity_curve["trading_cost"].sum() == pytest.approx(3.0 * costs[0])
    assert reduced_liquidity.fills.iloc[0]["filled_shares"] == pytest.approx(
        0.5 * baseline.fills.iloc[0]["filled_shares"]
    )
    assert partial_pressure.fills.iloc[0]["filled_shares"] == pytest.approx(
        0.5 * baseline.fills.iloc[0]["filled_shares"]
    )
    assert delayed.fills.iloc[0]["fill_date"] > baseline.fills.iloc[0]["fill_date"]
    assert capacity.orders["requested_notional"].sum() == pytest.approx(
        5.0 * baseline.orders["requested_notional"].sum()
    )


def test_missing_required_open_price_raises_without_publishing_a_result() -> None:
    panel = _panel(("AAA",), periods=3)
    sessions = pd.DatetimeIndex(sorted(panel["date"].unique()))
    panel.loc[panel["date"] == sessions[1], "open"] = np.nan
    journal = InMemoryJournal()

    with pytest.raises(ValueError, match="invalid open prices"):
        _run(
            panel,
            _targets(panel, {"AAA": 1.0}),
            event_journal=journal,
        )

    committed = journal.events()
    assert committed
    assert isinstance(committed[-1].payload, EngineHalted)
    assert committed[-1].payload.reason_code == "missing_open_price"
    assert all(event.coordinate.session <= sessions[1].date() for event in committed)
    assert not any(
        isinstance(event.payload, PortfolioMarked)
        and event.coordinate.session == sessions[1].date()
        and event.payload.mark_type == "close"
        for event in committed
    )


def test_fill_price_resource_failure_never_commits_an_open_order_prefix() -> None:
    panel = _panel(("AAA",), periods=3, price=1.0e12)
    journal = InMemoryJournal()

    with pytest.raises(ValueError, match="fill price exceeds"):
        run_backtest(
            panel,
            _targets(panel, {"AAA": 1.0}),
            costs={
                "commission_bps": 0.0,
                "half_spread_bps": 1.0,
                "slippage_bps": 0.0,
            },
            event_journal=journal,
            event_run_id=_RUN_ID,
        )

    event_types = {event.event_type for event in journal.events()}
    assert "order_submitted" not in event_types
    assert "order_accepted" not in event_types
    assert "fill_applied" not in event_types


def test_missing_required_close_price_records_halt_without_publishing_result() -> None:
    panel = _panel(("AAA",), periods=4)
    sessions = pd.DatetimeIndex(sorted(panel["date"].unique()))
    panel.loc[panel["date"] == sessions[1], "close"] = np.nan
    journal = InMemoryJournal()

    with pytest.raises(ValueError, match="invalid close prices"):
        _run(
            panel,
            _targets(panel, {"AAA": 1.0}),
            event_journal=journal,
        )

    committed = journal.events()
    assert committed
    assert isinstance(committed[-1].payload, EngineHalted)
    assert committed[-1].payload.reason_code == "missing_close_price"
    assert all(event.coordinate.session <= sessions[1].date() for event in committed)
    assert not any(
        isinstance(event.payload, PortfolioMarked)
        and event.coordinate.session == sessions[1].date()
        and event.payload.mark_type == "close"
        for event in committed
    )


def test_adverse_gap_bankruptcy_records_open_mark_then_terminal_halt() -> None:
    panel = _panel(("AAA",), periods=4, price=100.0)
    sessions = pd.DatetimeIndex(sorted(panel["date"].unique()))
    gap_session = sessions[2]
    panel.loc[panel["date"] == gap_session, ["open", "high", "low", "close"]] = 40.0
    journal = InMemoryJournal()

    with pytest.raises(RuntimeError, match="non-positive open equity"):
        _run(
            panel,
            _targets(panel, {"AAA": 2.0}),
            event_journal=journal,
        )

    committed = journal.events()
    assert isinstance(committed[-2].payload, PortfolioMarked)
    assert committed[-2].payload.mark_type == "open"
    assert committed[-2].coordinate.session == gap_session.date()
    assert committed[-2].payload.equity < 0.0
    assert isinstance(committed[-1].payload, EngineHalted)
    assert committed[-1].payload.reason_code == "bankruptcy"
    assert committed[-1].causation_id == committed[-2].event_id
    assert committed[-1].coordinate.session == gap_session.date()
    assert not any(
        event.coordinate.session >= gap_session.date()
        and isinstance(event.payload, PortfolioMarked)
        and event.payload.mark_type == "close"
        for event in committed
    )


def test_fee_induced_fill_bankruptcy_halts_before_later_execution() -> None:
    panel = _panel(("AAA", "BBB"), periods=4, price=100.0)
    journal = InMemoryJournal()

    with pytest.raises(RuntimeError, match="non-positive fill equity"):
        run_backtest(
            panel,
            _targets(panel, {"AAA": 0.5, "BBB": 0.5}),
            initial_capital=1_000.0,
            costs={
                "commission_bps": 20_000.0,
                "half_spread_bps": 0.0,
                "slippage_bps": 0.0,
            },
            event_journal=journal,
            event_run_id=_RUN_ID,
        )

    committed = journal.events()
    assert isinstance(committed[-2].payload, FillApplied)
    assert isinstance(committed[-1].payload, EngineHalted)
    assert committed[-1].payload.reason_code == "bankruptcy"
    assert committed[-1].causation_id == committed[-2].event_id
    assert not any(
        event.coordinate.session == committed[-1].coordinate.session
        and isinstance(event.payload, PortfolioMarked)
        and event.payload.mark_type == "close"
        for event in committed
    )


def test_financing_bankruptcy_records_charge_then_terminal_halt() -> None:
    panel = _panel(("AAA",), periods=4, price=10.0)
    journal = InMemoryJournal()

    with pytest.raises(RuntimeError, match="non-positive financing equity"):
        run_backtest(
            panel,
            _targets(panel, {"AAA": 2.0}),
            initial_capital=1_000.0,
            costs=_NO_COSTS,
            carry={
                "cash_financing_bps_annual": 1_000_000.0,
                "short_borrow_bps_annual": 0.0,
                "sessions_per_year": 1,
            },
            event_journal=journal,
            event_run_id=_RUN_ID,
        )

    committed = journal.events()
    assert isinstance(committed[-2].payload, CashChargeAccrued)
    assert committed[-2].payload.charge_type == "financing"
    assert isinstance(committed[-1].payload, EngineHalted)
    assert committed[-1].payload.reason_code == "bankruptcy"
    assert committed[-1].causation_id == committed[-2].event_id
    assert not any(
        event.coordinate.session == committed[-1].coordinate.session
        and isinstance(event.payload, PortfolioMarked)
        and event.payload.mark_type == "close"
        for event in committed
    )


def test_backtest_sqlite_journal_restarts_and_replays_exact_final_state(tmp_path) -> None:
    panel = _panel(("AAA",), periods=5)
    targets = _targets(panel, {"AAA": 1.0})
    sessions = pd.DatetimeIndex(sorted(panel["date"].unique()))
    path = tmp_path / "backtest.event-journal.sqlite3"

    with SQLiteJournal(path) as journal:
        result = _run(panel, targets, event_journal=journal)
        expected_count = journal.count
        expected_head = journal.head_hash
        assert expected_count == len(result.events)

    with SQLiteJournal(path) as restarted:
        assert restarted.count == expected_count
        assert restarted.head_hash == expected_head
        replayed = DeterministicEventEngine.replay(
            restarted,
            calendar=tuple(session.date() for session in sessions),
            initial_cash=1_000_000.0,
            expected_run_id=_RUN_ID,
        )
        portfolio = replayed.snapshot().portfolio
        assert portfolio is not None
        assert portfolio.cash == result.accounting.iloc[-1]["cash"]
        assert portfolio.equity == result.accounting.iloc[-1]["equity"]
