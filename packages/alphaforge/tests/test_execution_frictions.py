"""Adversarial tests for immutable execution-friction domain contracts."""

from __future__ import annotations

import math
from dataclasses import FrozenInstanceError, replace
from datetime import date, datetime, timedelta

import pytest

from alphaforge.execution.frictions import (
    MAX_CARRY_SYMBOLS,
    STANDARD_STRESS_PROFILE_ORDER,
    CarryAccrual,
    CarryCostModel,
    ExecutionStressProfile,
    FrictionContractError,
    LatencyModel,
    LatencySchedule,
    ModelDeclaration,
    standard_stress_profiles,
)


def _calendar(count: int = 32) -> tuple[date, ...]:
    start = date(2026, 1, 5)
    return tuple(start + timedelta(days=index) for index in range(count))


def _declaration() -> ModelDeclaration:
    return ModelDeclaration(
        model_id="reference-model",
        version="1.0.0",
        units=(("amount", "USD"), ("rate", "basis points")),
        calibration_provenance="predeclared synthetic reference",
        domain="deterministic unit test",
        parameter_bounds=(("amount", "finite and non-negative"),),
        execution_timestamp="one logical session",
        failure_behavior="reject malformed input",
        limitations=("Not observed execution quality.",),
    )


def test_model_declaration_is_frozen_canonical_and_content_addressed() -> None:
    declaration = _declaration()
    equivalent = _declaration()

    assert declaration == equivalent
    assert declaration.digest == equivalent.digest
    assert len(declaration.digest) == 64
    assert declaration.to_dict()["units"] == {"amount": "USD", "rate": "basis points"}
    with pytest.raises(FrozenInstanceError):
        declaration.model_id = "changed"  # type: ignore[misc]


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"model_id": "bad model"}, "identifier"),
        ({"calibration_provenance": ""}, "non-empty"),
        ({"units": (("z", "last"), ("a", "first"))}, "lexical order"),
        ({"units": (("amount", "USD"), ("amount", "USD"))}, "unique"),
        ({"limitations": ("same", "same")}, "unique"),
    ],
)
def test_model_declaration_rejects_ambiguous_metadata(
    overrides: dict[str, object], message: str
) -> None:
    values: dict[str, object] = {
        "model_id": "reference-model",
        "version": "1.0.0",
        "units": (("amount", "USD"),),
        "calibration_provenance": "predeclared synthetic reference",
        "domain": "deterministic unit test",
        "parameter_bounds": (("amount", "finite and non-negative"),),
        "execution_timestamp": "one logical session",
        "failure_behavior": "reject malformed input",
        "limitations": (),
    }
    values.update(overrides)
    with pytest.raises(FrictionContractError, match=message):
        ModelDeclaration(**values)  # type: ignore[arg-type]


def test_zero_stage_latency_preserves_irreducible_next_session_execution() -> None:
    sessions = _calendar()
    model = LatencyModel()

    schedule = model.schedule(sessions[3], calendar=sessions, execution_lag=1)

    assert schedule.stage_sessions == (
        sessions[3],
        sessions[3],
        sessions[3],
        sessions[3],
        sessions[3],
        sessions[4],
    )
    assert schedule.execution_lag_sessions == 1
    assert schedule.fill_session > schedule.origin_session
    assert schedule.model_digest == model.configuration_digest
    assert LatencySchedule.from_config(schedule.to_dict(), calendar=sessions) == schedule


def test_latency_stages_compose_cumulatively_on_frozen_calendar() -> None:
    sessions = _calendar()
    model = LatencyModel(
        data_delay_sessions=1,
        feature_delay_sessions=1,
        inference_delay_sessions=2,
        submission_delay_sessions=0,
        fill_delay_sessions=1,
        calibration_provenance="frozen before the synthetic test",
    )

    assert model.cumulative_offsets(execution_lag=1) == (1, 2, 4, 4, 6)
    assert model.stage_delay_sessions == 5
    schedule = model.schedule(sessions[2], calendar=sessions, execution_lag=1)
    assert schedule.stage_sessions == (
        sessions[2],
        sessions[3],
        sessions[4],
        sessions[6],
        sessions[6],
        sessions[8],
    )
    assert "logical trading sessions" in dict(model.declaration.units).values()
    assert "wall-clock" in " ".join(model.declaration.limitations)


@pytest.mark.parametrize(
    "stressed_model",
    [
        LatencyModel(data_delay_sessions=1),
        LatencyModel(feature_delay_sessions=1),
        LatencyModel(inference_delay_sessions=1),
        LatencyModel(submission_delay_sessions=1),
        LatencyModel(fill_delay_sessions=1),
    ],
)
def test_increasing_any_latency_stage_cannot_advance_information_or_fill(
    stressed_model: LatencyModel,
) -> None:
    sessions = _calendar()
    origin = sessions[2]
    baseline_model = LatencyModel()
    baseline = baseline_model.schedule(origin, calendar=sessions, execution_lag=1)
    stressed = stressed_model.schedule(
        origin,
        calendar=sessions,
        execution_lag=1,
    )

    baseline_positions = tuple(sessions.index(value) for value in baseline.stage_sessions)
    stressed_positions = tuple(sessions.index(value) for value in stressed.stage_sessions)
    assert all(
        stressed_index >= baseline_index
        for stressed_index, baseline_index in zip(
            stressed_positions, baseline_positions, strict=True
        )
    )
    assert stressed.fill_session > baseline.fill_session


def test_increasing_execution_lag_cannot_advance_fill() -> None:
    sessions = _calendar()
    model = LatencyModel(data_delay_sessions=1, inference_delay_sessions=1)
    fills = [
        model.schedule(sessions[1], calendar=sessions, execution_lag=lag).fill_session
        for lag in (1, 2, 3, 4)
    ]
    assert fills == sorted(fills)
    assert len(set(fills)) == len(fills)


@pytest.mark.parametrize(
    ("config", "message"),
    [
        ({"lookahead_sessions": 1}, "unknown latency"),
        ({"data_delay_sessions": True}, "integer"),
        ({"feature_delay_sessions": 1.0}, "integer"),
        ({"inference_delay_sessions": float("inf")}, "integer"),
        ({"submission_delay_sessions": -1}, "integer"),
        ({"fill_delay_sessions": -1}, "integer"),
        ({"calibration_provenance": ""}, "non-empty"),
    ],
)
def test_latency_config_rejects_unknown_coerced_or_unbounded_values(
    config: dict[str, object], message: str
) -> None:
    with pytest.raises(FrictionContractError, match=message):
        LatencyModel.from_config(config)


def test_latency_rejects_total_overflow_and_invalid_execution_lag() -> None:
    with pytest.raises(FrictionContractError, match="total logical latency"):
        LatencyModel(
            data_delay_sessions=2_520,
            feature_delay_sessions=2_520,
            inference_delay_sessions=1,
        )
    with pytest.raises(FrictionContractError, match="less than"):
        LatencyModel(data_delay_sessions=2_520, feature_delay_sessions=2_520)
    for invalid in (0, -1, True, 1.0, float("inf")):
        with pytest.raises(FrictionContractError, match="execution_lag"):
            LatencyModel().schedule(
                _calendar()[0],
                calendar=_calendar(),
                execution_lag=invalid,  # type: ignore[arg-type]
            )


def test_latency_schedule_fails_closed_outside_or_beyond_calendar() -> None:
    sessions = _calendar(6)
    with pytest.raises(FrictionContractError, match="origin_session"):
        LatencyModel().schedule(date(2030, 1, 1), calendar=sessions, execution_lag=1)
    with pytest.raises(FrictionContractError, match="exceeds the frozen calendar"):
        LatencyModel(fill_delay_sessions=1).schedule(
            sessions[-2], calendar=sessions, execution_lag=1
        )
    with pytest.raises(FrictionContractError, match="strictly increasing"):
        LatencyModel().schedule(
            sessions[0],
            calendar=(sessions[0], sessions[0]),
            execution_lag=1,
        )
    with pytest.raises(FrictionContractError, match="date without a time"):
        LatencyModel().schedule(
            sessions[0],
            calendar=(datetime(2026, 1, 5), date(2026, 1, 6)),
            execution_lag=1,
        )


def test_serialized_latency_schedule_rejects_missing_unknown_and_foreign_sessions() -> None:
    sessions = _calendar()
    schedule = LatencyModel().schedule(sessions[0], calendar=sessions, execution_lag=1)
    valid = schedule.to_dict()

    missing = dict(valid)
    missing.pop("fill_session")
    with pytest.raises(FrictionContractError, match="missing settings"):
        LatencySchedule.from_config(missing, calendar=sessions)
    with pytest.raises(FrictionContractError, match="unknown latency schedule"):
        LatencySchedule.from_config({**valid, "wall_clock_ns": 1}, calendar=sessions)
    with pytest.raises(FrictionContractError, match="outside the frozen calendar"):
        LatencySchedule.from_config(
            {**valid, "fill_session": "2030-01-01"},
            calendar=sessions,
        )
    with pytest.raises(FrictionContractError, match="monotonic"):
        LatencySchedule.from_config(
            {
                **valid,
                "data_available_session": sessions[1].isoformat(),
                "feature_available_session": sessions[0].isoformat(),
            },
            calendar=sessions,
        )
    with pytest.raises(FrictionContractError, match="preserve the declared execution lag"):
        LatencySchedule.from_config(
            {**valid, "execution_lag_sessions": 10},
            calendar=sessions,
        )


def test_latency_identity_is_deterministic_and_configuration_sensitive() -> None:
    first = LatencyModel.from_config({"data_delay_sessions": 1, "fill_delay_sessions": 2})
    second = LatencyModel.from_config({"fill_delay_sessions": 2, "data_delay_sessions": 1})
    changed = replace(first, fill_delay_sessions=3)

    assert first == second
    assert first.configuration_digest == second.configuration_digest
    assert changed.configuration_digest != first.configuration_digest


def _carry_model() -> CarryCostModel:
    return CarryCostModel(
        cash_financing_bps_annual=500.0,
        short_borrow_bps_annual=1_000.0,
        sessions_per_year=250,
        calibration_provenance="predeclared synthetic carry assumptions",
    )


def test_carry_accrual_matches_independent_reference_and_exact_fsum() -> None:
    model = _carry_model()
    accrual = model.accrue(
        session=date(2026, 1, 6),
        cash_usd=-100_000.0,
        positions={"B": -5.0, "LONG": 100.0, "A": -10.0},
        prices_usd={"B": 200.0, "A": 100.0},
    )

    expected_financing = 100_000.0 * (500.0 / 10_000.0) / 250.0
    expected_borrow_items = (
        ("A", 1_000.0 * (1_000.0 / 10_000.0) / 250.0),
        ("B", 1_000.0 * (1_000.0 / 10_000.0) / 250.0),
    )
    assert accrual.financing_basis_usd == 100_000.0
    assert accrual.financing_charge_usd == expected_financing
    assert accrual.short_market_values_usd == (("A", 1_000.0), ("B", 1_000.0))
    assert accrual.borrow_charges_usd == expected_borrow_items
    assert accrual.total_borrow_charge_usd == math.fsum(
        amount for _, amount in expected_borrow_items
    )
    assert accrual.total_charge_usd == math.fsum(
        (expected_financing, accrual.total_borrow_charge_usd)
    )
    assert accrual.currency == "USD"
    assert accrual.model_digest == model.configuration_digest


def test_carry_accrual_is_order_invariant_and_records_no_borrow_claim() -> None:
    model = _carry_model()
    first = model.accrue(
        session=date(2026, 1, 6),
        cash_usd=-500.0,
        positions={"B": -2.0, "A": -1.0},
        prices_usd={"B": 20.0, "A": 10.0},
    )
    second = model.accrue(
        session=date(2026, 1, 6),
        cash_usd=-500.0,
        positions={"A": -1.0, "B": -2.0},
        prices_usd={"A": 10.0, "B": 20.0},
    )

    assert first == second
    assert first.digest == second.digest
    assert first.borrow_availability == "not_modeled"
    assert first.locate_status == "not_modeled"
    limitations = " ".join(model.declaration.limitations).lower()
    assert "availability" in limitations
    assert "locates" in limitations
    assert "executable" in limitations


def test_positive_cash_and_long_positions_accrue_no_carry() -> None:
    accrual = _carry_model().accrue(
        session=date(2026, 1, 6),
        cash_usd=1_000.0,
        positions={"LONG": 5.0, "FLAT": 0.0},
        prices_usd={},
    )
    assert accrual.financing_basis_usd == 0.0
    assert accrual.financing_charge_usd == 0.0
    assert accrual.short_market_values_usd == ()
    assert accrual.borrow_charges_usd == ()
    assert accrual.total_charge_usd == 0.0


def test_carry_cost_is_monotonic_in_debt_short_size_and_price() -> None:
    model = _carry_model()
    debt_charges = [
        model.accrue(
            session=date(2026, 1, 6),
            cash_usd=-debt,
            positions={},
            prices_usd={},
        ).total_charge_usd
        for debt in (1.0, 10.0, 100.0, 1_000.0)
    ]
    short_charges = [
        model.accrue(
            session=date(2026, 1, 6),
            cash_usd=0.0,
            positions={"A": -quantity},
            prices_usd={"A": price},
        ).total_borrow_charge_usd
        for quantity, price in ((1.0, 10.0), (2.0, 10.0), (2.0, 20.0), (4.0, 20.0))
    ]
    assert debt_charges == sorted(debt_charges)
    assert short_charges == sorted(short_charges)
    assert all(right > left for left, right in zip(debt_charges, debt_charges[1:], strict=False))
    assert all(right > left for left, right in zip(short_charges, short_charges[1:], strict=False))


@pytest.mark.parametrize(
    ("config", "message"),
    [
        ({"unknown_rate": 1.0}, "unknown carry cost"),
        ({"cash_financing_bps_annual": True}, "finite real"),
        ({"cash_financing_bps_annual": -1.0}, "finite and in"),
        ({"short_borrow_bps_annual": float("inf")}, "finite and in"),
        ({"sessions_per_year": True}, "integer"),
        ({"sessions_per_year": 0}, "integer"),
        ({"calibration_provenance": ""}, "non-empty"),
    ],
)
def test_carry_config_rejects_unknown_coerced_or_nonfinite_values(
    config: dict[str, object], message: str
) -> None:
    with pytest.raises(FrictionContractError, match=message):
        CarryCostModel.from_config(config)


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"session": datetime(2026, 1, 6)}, "date without a time"),
        ({"cash_usd": True}, "finite real"),
        ({"cash_usd": float("inf")}, "finite and in"),
        ({"positions": {"BAD SYMBOL": -1.0}}, "market symbol"),
        ({"positions": {"A": True}}, "finite real"),
        ({"positions": {"A": -1.0}, "prices_usd": {}}, "missing positive"),
        ({"positions": {"A": -1.0}, "prices_usd": {"A": 0.0}}, "finite and in"),
        ({"prices_usd": {"UNUSED": float("nan")}}, "finite and in"),
        (
            {"positions": {"A": -1.0e15}, "prices_usd": {"A": 1.0e12}},
            "short market value",
        ),
    ],
)
def test_carry_inputs_fail_closed(overrides: dict[str, object], message: str) -> None:
    values: dict[str, object] = {
        "session": date(2026, 1, 6),
        "cash_usd": 0.0,
        "positions": {},
        "prices_usd": {},
    }
    values.update(overrides)
    with pytest.raises(FrictionContractError, match=message):
        _carry_model().accrue(**values)  # type: ignore[arg-type]


def test_carry_input_mappings_enforce_symbol_resource_ceiling_before_copy() -> None:
    oversized = {f"S{index:05d}": 1.0 for index in range(MAX_CARRY_SYMBOLS + 1)}
    with pytest.raises(FrictionContractError, match="positions exceeds"):
        _carry_model().accrue(
            session=date(2026, 1, 6),
            cash_usd=0.0,
            positions=oversized,
            prices_usd={},
        )
    with pytest.raises(FrictionContractError, match="prices_usd exceeds"):
        _carry_model().accrue(
            session=date(2026, 1, 6),
            cash_usd=0.0,
            positions={},
            prices_usd=oversized,
        )


def test_carry_accrual_rejects_forged_totals_and_claims_and_is_frozen() -> None:
    accrual = _carry_model().accrue(
        session=date(2026, 1, 6),
        cash_usd=-100.0,
        positions={"A": -1.0},
        prices_usd={"A": 10.0},
    )
    with pytest.raises(FrictionContractError, match="exact math.fsum"):
        replace(accrual, total_charge_usd=accrual.total_charge_usd + 1.0)
    with pytest.raises(FrictionContractError, match="cannot claim"):
        replace(accrual, locate_status="located")  # type: ignore[arg-type]
    with pytest.raises(FrozenInstanceError):
        accrual.currency = "EUR"  # type: ignore[misc, assignment]


def test_carry_accrual_requires_aligned_sorted_component_records() -> None:
    base = _carry_model().accrue(
        session=date(2026, 1, 6),
        cash_usd=0.0,
        positions={"A": -1.0},
        prices_usd={"A": 10.0},
    )
    with pytest.raises(FrictionContractError, match="same sorted symbols"):
        CarryAccrual(
            session=base.session,
            financing_basis_usd=0.0,
            financing_charge_usd=0.0,
            short_market_values_usd=(("A", 10.0),),
            borrow_charges_usd=(),
            total_borrow_charge_usd=0.0,
            total_charge_usd=0.0,
            model_digest=base.model_digest,
            calibration_provenance=base.calibration_provenance,
        )


def test_standard_stress_grid_has_canonical_order_coverage_and_unique_identity() -> None:
    first = standard_stress_profiles()
    second = standard_stress_profiles()
    by_name = {profile.name: profile for profile in first}

    assert tuple(profile.name for profile in first) == STANDARD_STRESS_PROFILE_ORDER
    assert first == second
    assert tuple(profile.digest for profile in first) == tuple(profile.digest for profile in second)
    assert len({profile.digest for profile in first}) == len(first)
    assert by_name["baseline"] == ExecutionStressProfile(name="baseline")
    assert by_name["doubled_costs"].cost_multiplier == 2.0
    assert by_name["tripled_costs"].cost_multiplier == 3.0
    assert by_name["adverse_spread"].effective_spread_multiplier == 3.0
    assert by_name["reduced_liquidity"].liquidity_multiplier == 0.5
    assert by_name["delayed_signals"].signal_delay_sessions == 1
    assert by_name["partial_fill_pressure"].participation_limit_multiplier == 0.5
    assert by_name["capacity_scaling"].capacity_multiplier == 5.0


def test_every_standard_stress_is_adverse_or_baseline() -> None:
    for profile in standard_stress_profiles():
        assert profile.cost_multiplier >= 1.0
        assert profile.spread_multiplier >= 1.0
        assert 0.0 < profile.liquidity_multiplier <= 1.0
        assert 0.0 < profile.participation_limit_multiplier <= 1.0
        assert profile.capacity_multiplier >= 1.0
        assert profile.signal_delay_sessions >= 0
        declaration = profile.declaration
        assert declaration.units
        assert declaration.parameter_bounds
        assert "test outcomes" in declaration.calibration_provenance
        assert "sensitivities" in " ".join(declaration.limitations)


def test_stress_profile_digest_is_mapping_order_invariant_and_value_sensitive() -> None:
    first = ExecutionStressProfile.from_config(
        {"name": "custom", "cost_multiplier": 2.0, "liquidity_multiplier": 0.75}
    )
    second = ExecutionStressProfile.from_config(
        {"liquidity_multiplier": 0.75, "cost_multiplier": 2.0, "name": "custom"}
    )
    changed = replace(first, cost_multiplier=3.0)

    assert first == second
    assert first.digest == second.digest
    assert first.digest != changed.digest
    with pytest.raises(FrozenInstanceError):
        first.name = "changed"  # type: ignore[misc]


@pytest.mark.parametrize(
    ("config", "message"),
    [
        ({}, "requires name"),
        ({"name": "x", "unknown": 1}, "unknown execution stress"),
        ({"name": "x", "cost_multiplier": True}, "finite real"),
        ({"name": "x", "cost_multiplier": 0.99}, "finite and in"),
        ({"name": "x", "spread_multiplier": float("inf")}, "finite and in"),
        ({"name": "x", "liquidity_multiplier": 0.0}, "finite and in"),
        ({"name": "x", "liquidity_multiplier": 1.01}, "finite and in"),
        ({"name": "x", "signal_delay_sessions": True}, "integer"),
        ({"name": "x", "signal_delay_sessions": -1}, "integer"),
        ({"name": "x", "participation_limit_multiplier": 0.0}, "finite and in"),
        ({"name": "x", "capacity_multiplier": 0.5}, "finite and in"),
        ({"name": "baseline", "cost_multiplier": 2.0}, "reserved stress profile"),
        ({"name": "bad name"}, "identifier"),
    ],
)
def test_stress_profile_rejects_favorable_ambiguous_or_unbounded_settings(
    config: dict[str, object], message: str
) -> None:
    with pytest.raises(FrictionContractError, match=message):
        ExecutionStressProfile.from_config(config)
