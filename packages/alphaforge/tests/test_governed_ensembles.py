"""Adversarial and mathematical tests for the governed OOF ensemble boundary."""

from __future__ import annotations

from dataclasses import FrozenInstanceError, replace

import numpy as np
import pandas as pd
import pytest

from alphaforge.models import (
    EnsembleAuditRecord,
    EnsembleContractError,
    EnsembleDecision,
    GovernedEnsembleConfig,
    GovernedEnsembleState,
    HoldoutLeakageError,
    IncompleteOOFError,
    InferenceBatch,
    TemporalOOFFold,
    TrainingOOFPanel,
    fit_governed_ensemble,
)

EXPERTS = ("defensive", "redundant", "trend")


def _training_frame(
    *, constant_target: bool = False
) -> tuple[pd.DataFrame, tuple[TemporalOOFFold, ...], str]:
    generator = np.random.default_rng(90210)
    dates = pd.bdate_range("2024-01-02", periods=24)
    symbols = ("AAA", "BBB", "CCC", "DDD")
    rows: list[dict[str, object]] = []
    for date_index, date_value in enumerate(dates):
        fold_id = date_index // 8
        regime = 0.12 if date_index < 12 else 0.88
        for symbol_index, symbol in enumerate(symbols):
            latent = (
                0.012 * np.sin(date_index / 3.0)
                + 0.003 * (symbol_index - 1.5)
                + generator.normal(0.0, 0.001)
            )
            target = 0.0 if constant_target else latent
            predictions = {
                "trend": latent + generator.normal(0.0, 0.002 if regime < 0.5 else 0.009),
                "defensive": latent + generator.normal(0.0, 0.008 if regime < 0.5 else 0.002),
                "redundant": latent + generator.normal(0.0, 0.0022 if regime < 0.5 else 0.0092),
            }
            for expert, prediction in predictions.items():
                rows.append(
                    {
                        "date": date_value,
                        "symbol": symbol,
                        "fold_id": fold_id,
                        "expert": expert,
                        "prediction": prediction,
                        "target": target,
                        "uncertainty": 0.004,
                        "regime_probability": regime,
                    }
                )
    folds = tuple(
        TemporalOOFFold(
            fold_id=fold_id,
            training_end=(dates[fold_id * 8] - pd.Timedelta(days=3)).date().isoformat(),
            validation_start=dates[fold_id * 8].date().isoformat(),
            validation_end=dates[fold_id * 8 + 7].date().isoformat(),
            embargo_days=2,
        )
        for fold_id in range(3)
    )
    holdout_start = (dates[-1] + pd.offsets.BDay()).date().isoformat()
    return pd.DataFrame(rows), folds, holdout_start


def _panel(*, constant_target: bool = False) -> TrainingOOFPanel:
    frame, folds, holdout_start = _training_frame(constant_target=constant_target)
    return TrainingOOFPanel.from_frame(
        frame,
        folds=folds,
        expected_experts=EXPERTS,
        holdout_start=holdout_start,
        source_id="synthetic-oof-v1",
    )


def _inference_frame(panel: TrainingOOFPanel, *, regime: float = 0.2) -> pd.DataFrame:
    date_value = panel.holdout_start
    rows = []
    for symbol_index, symbol in enumerate(("AAA", "BBB", "CCC", "DDD")):
        base = 0.002 * (symbol_index - 1.5)
        for expert_index, expert in enumerate(EXPERTS):
            rows.append(
                {
                    "date": date_value,
                    "symbol": symbol,
                    "expert": expert,
                    "prediction": base + 0.0002 * expert_index,
                    "uncertainty": 0.004,
                    "regime_probability": regime,
                }
            )
    return pd.DataFrame(rows)


def _expected_keys(frame: pd.DataFrame) -> tuple[tuple[str, str], ...]:
    return tuple(
        sorted(
            {
                (pd.Timestamp(row.date).date().isoformat(), str(row.symbol))
                for row in frame.itertuples(index=False)
            }
        )
    )


def _batch(
    frame: pd.DataFrame,
    *,
    expected_keys: tuple[tuple[str, str], ...] | None = None,
) -> InferenceBatch:
    return InferenceBatch.from_frame(
        frame,
        expected_keys=_expected_keys(frame) if expected_keys is None else expected_keys,
    )


def _config(method: str, **kwargs: object) -> GovernedEnsembleConfig:
    return GovernedEnsembleConfig(method=method, experts=EXPERTS, **kwargs)  # type: ignore[arg-type]


def test_oof_panel_is_order_isolated_and_byte_deterministic() -> None:
    frame, folds, holdout_start = _training_frame()
    first = TrainingOOFPanel.from_frame(
        frame,
        folds=folds,
        expected_experts=EXPERTS,
        holdout_start=holdout_start,
        source_id="synthetic-oof-v1",
    )
    second = TrainingOOFPanel.from_frame(
        frame.sample(frac=1.0, random_state=4),
        folds=tuple(reversed(folds)),
        expected_experts=tuple(reversed(EXPERTS)),
        holdout_start=holdout_start,
        source_id="synthetic-oof-v1",
    )

    assert first.identity == second.identity
    assert first.to_json() == second.to_json()
    restored = TrainingOOFPanel.from_json(first.to_json())
    assert restored == first
    assert restored.identity == first.identity


def test_oof_boundary_rejects_missing_expert_and_holdout_rows() -> None:
    frame, folds, holdout_start = _training_frame()
    missing = frame.drop(index=frame.index[0])
    with pytest.raises(IncompleteOOFError, match="complete expert"):
        TrainingOOFPanel.from_frame(
            missing,
            folds=folds,
            expected_experts=EXPERTS,
            holdout_start=holdout_start,
            source_id="synthetic-oof-v1",
        )

    leaked = frame.copy()
    leaked.loc[0, "date"] = holdout_start
    with pytest.raises(HoldoutLeakageError, match="holdout"):
        TrainingOOFPanel.from_frame(
            leaked,
            folds=folds,
            expected_experts=EXPERTS,
            holdout_start=holdout_start,
            source_id="synthetic-oof-v1",
        )


def test_oof_boundary_rejects_malformed_folds_values_and_resource_overflow() -> None:
    frame, folds, holdout_start = _training_frame()
    with pytest.raises(EnsembleContractError, match="embargo"):
        replace(
            folds[0],
            training_end=(pd.Timestamp(folds[0].validation_start) - pd.Timedelta(days=1))
            .date()
            .isoformat(),
        )
    non_integer_fold = frame.copy()
    non_integer_fold["fold_id"] = non_integer_fold["fold_id"].astype(float)
    with pytest.raises(EnsembleContractError, match="integer"):
        TrainingOOFPanel.from_frame(
            non_integer_fold,
            folds=folds,
            expected_experts=EXPERTS,
            holdout_start=holdout_start,
            source_id="synthetic-oof-v1",
        )
    mixed_date = frame.copy()
    first_date = mixed_date.loc[0, "date"]
    mixed_date.loc[
        (mixed_date["date"] == first_date) & (mixed_date["symbol"] == "AAA"),
        "fold_id",
    ] = 1
    with pytest.raises(EnsembleContractError, match="exactly one fold"):
        TrainingOOFPanel.from_frame(
            mixed_date,
            folds=folds,
            expected_experts=EXPERTS,
            holdout_start=holdout_start,
            source_id="synthetic-oof-v1",
        )
    non_string_symbol = frame.copy()
    non_string_symbol["symbol"] = non_string_symbol["symbol"].astype(object)
    non_string_symbol.loc[0, "symbol"] = 123
    with pytest.raises(EnsembleContractError, match="must be strings"):
        TrainingOOFPanel.from_frame(
            non_string_symbol,
            folds=folds,
            expected_experts=EXPERTS,
            holdout_start=holdout_start,
            source_id="synthetic-oof-v1",
        )
    numeric_date = frame.copy()
    numeric_date["date"] = numeric_date["date"].astype(object)
    numeric_date.loc[0, "date"] = 1
    with pytest.raises(EnsembleContractError, match="calendar date"):
        TrainingOOFPanel.from_frame(
            numeric_date,
            folds=folds,
            expected_experts=EXPERTS,
            holdout_start=holdout_start,
            source_id="synthetic-oof-v1",
        )
    duplicate_column = pd.concat([frame, frame[["target"]]], axis=1)
    with pytest.raises(EnsembleContractError, match="fields mismatch"):
        TrainingOOFPanel.from_frame(
            duplicate_column,
            folds=folds,
            expected_experts=EXPERTS,
            holdout_start=holdout_start,
            source_id="synthetic-oof-v1",
        )
    outside = frame.copy()
    outside.loc[0, "date"] = "2023-12-01"
    with pytest.raises(EnsembleContractError, match="outside"):
        TrainingOOFPanel.from_frame(
            outside,
            folds=folds,
            expected_experts=EXPERTS,
            holdout_start=holdout_start,
            source_id="synthetic-oof-v1",
        )
    non_finite = frame.copy()
    non_finite.loc[0, "prediction"] = np.inf
    with pytest.raises(EnsembleContractError, match="finite"):
        TrainingOOFPanel.from_frame(
            non_finite,
            folds=folds,
            expected_experts=EXPERTS,
            holdout_start=holdout_start,
            source_id="synthetic-oof-v1",
        )
    with pytest.raises(EnsembleContractError, match="max_records"):
        TrainingOOFPanel.from_frame(
            frame,
            folds=folds,
            expected_experts=EXPERTS,
            holdout_start=holdout_start,
            source_id="synthetic-oof-v1",
            max_records=1,
        )
    with pytest.raises(EnsembleContractError, match="positive integer"):
        TrainingOOFPanel.from_json(_panel().to_json(), max_bytes=float("nan"))  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "method",
    ["static", "rank_vote", "stacking", "bayesian", "dynamic", "regime_gate"],
)
def test_every_policy_is_deterministic_serializable_and_finite(method: str) -> None:
    panel = _panel()
    config = _config(method, min_regime_rows=8)
    first = fit_governed_ensemble(panel, config)
    second = fit_governed_ensemble(panel, config)

    assert first == second
    assert first.identity == second.identity
    restored = GovernedEnsembleState.from_json(first.to_json())
    assert restored == first
    decisions = restored.predict(_batch(_inference_frame(panel)))
    assert len(decisions) == 4
    assert all(decision.status == "combined" for decision in decisions)
    assert all(np.isfinite(decision.prediction) for decision in decisions)
    assert all(decision.uncertainty is not None for decision in decisions)
    assert [decision.identity for decision in decisions] == [
        decision.identity for decision in restored.predict(_batch(_inference_frame(panel)))
    ]


def test_stacking_matches_ridge_reference_and_meta_folds_are_strictly_prior() -> None:
    panel = _panel()
    penalty = 0.75
    state = fit_governed_ensemble(panel, _config("stacking", ridge_penalty=penalty))
    frame = panel.wide_frame()
    matrix = frame[list(EXPERTS)].to_numpy()
    target = frame["target"].to_numpy()
    means = matrix.mean(axis=0)
    scales = np.where(matrix.std(axis=0) <= 1e-12, 1.0, matrix.std(axis=0))
    normalized = (matrix - means) / scales
    expected = np.linalg.lstsq(
        normalized.T @ normalized + penalty * np.eye(len(EXPERTS)),
        normalized.T @ (target - target.mean()),
        rcond=None,
    )[0]

    np.testing.assert_allclose(
        list(dict(state.weights).values()),
        expected,
        rtol=1e-12,
        atol=1e-12,
    )
    assert state.audits[0].status == "fallback"
    assert state.audits[0].input_folds == ()
    assert len(state.audits[0].identity) == 64
    for audit in state.audits[1:]:
        current_fold = int(audit.effective_after.split("-")[1])
        assert all(fold < current_fold for fold in audit.input_folds)


def test_correlated_experts_produce_stable_stacking_and_bayesian_weights() -> None:
    frame, folds, holdout_start = _training_frame()
    trend = frame.loc[frame["expert"] == "trend", "prediction"].to_numpy()
    frame.loc[frame["expert"] == "redundant", "prediction"] = trend
    panel = TrainingOOFPanel.from_frame(
        frame,
        folds=folds,
        expected_experts=EXPERTS,
        holdout_start=holdout_start,
        source_id="correlated-oof",
    )
    stacking = fit_governed_ensemble(panel, _config("stacking", ridge_penalty=1.0))
    bayesian = fit_governed_ensemble(panel, _config("bayesian"))

    assert np.isfinite(stacking.condition_number)
    assert np.isfinite(list(dict(stacking.weights).values())).all()
    assert sum(dict(bayesian.weights).values()) == pytest.approx(1.0)
    assert min(dict(bayesian.weights).values()) > 0.0


def test_learned_weight_floor_is_feasible_and_holds_after_normalization() -> None:
    floor = 0.2
    panel = _panel()
    state = fit_governed_ensemble(panel, _config("bayesian", min_weight=floor))
    frame = panel.wide_frame()
    matrix = frame.loc[:, list(EXPERTS)].to_numpy(dtype=float)
    target = frame["target"].to_numpy(dtype=float)
    mse = np.maximum(np.mean((matrix - target[:, None]) ** 2, axis=0), 1e-12)
    scores = -0.5 * len(target) * np.log(mse)
    unbounded = np.exp(np.clip(scores - scores.max(), -700.0, 0.0))
    unbounded /= unbounded.sum()
    expected = floor + (1.0 - len(EXPERTS) * floor) * unbounded

    np.testing.assert_allclose(list(dict(state.weights).values()), expected)
    assert min(dict(state.weights).values()) >= floor
    dynamic = fit_governed_ensemble(panel, _config("dynamic", min_weight=floor))
    assert all(min(dict(audit.weights_after).values()) >= floor for audit in dynamic.audits)
    equal = fit_governed_ensemble(
        panel,
        _config("bayesian", min_weight=1.0 / len(EXPERTS)),
    )
    assert list(dict(equal.weights).values()) == pytest.approx([1.0 / len(EXPERTS)] * 3)
    with pytest.raises(EnsembleContractError, match="number of experts"):
        _config("bayesian", min_weight=0.34)


def test_zero_confidence_regime_threshold_belongs_only_to_stress_gate() -> None:
    frame, folds, holdout_start = _training_frame()
    first_date = frame["date"].min()
    at_threshold = frame.copy()
    at_threshold.loc[at_threshold["date"] == first_date, "regime_probability"] = 0.5
    just_stress = at_threshold.copy()
    just_stress.loc[just_stress["date"] == first_date, "regime_probability"] = 0.5000001

    def fit_gate(candidate: pd.DataFrame, source_id: str) -> GovernedEnsembleState:
        panel = TrainingOOFPanel.from_frame(
            candidate,
            folds=folds,
            expected_experts=EXPERTS,
            holdout_start=holdout_start,
            source_id=source_id,
        )
        return fit_governed_ensemble(
            panel,
            _config(
                "regime_gate",
                regime_min_confidence=0.0,
                min_regime_rows=8,
            ),
        )

    threshold_state = fit_gate(at_threshold, "threshold-regime")
    stress_state = fit_gate(just_stress, "stress-regime")
    assert threshold_state.regime_weights == stress_state.regime_weights
    decision = threshold_state.predict(_batch(_inference_frame(_panel(), regime=0.5)))[0]
    assert decision.status == "combined"
    assert decision.weights == dict(threshold_state.regime_weights)["stress"]


def test_degenerate_learning_target_emits_frozen_fallback_state() -> None:
    state = fit_governed_ensemble(_panel(constant_target=True), _config("stacking"))

    assert state.fit_status == "fallback"
    assert state.fallback_reason == "degenerate_training_target"
    decision = state.predict(_batch(_inference_frame(_panel(constant_target=True))))[0]
    assert decision.status == "abstained"
    assert decision.reason == "fit_degenerate_training_target"
    assert decision.prediction == 0.0


def test_stacking_rejects_an_insufficient_audit_budget_before_fitting() -> None:
    with pytest.raises(EnsembleContractError, match="folds exceed max_audit_records"):
        fit_governed_ensemble(
            _panel(),
            _config("stacking", max_audit_records=1),
        )


def test_extreme_finite_values_fail_through_the_structured_numeric_boundary() -> None:
    panel = _panel()
    state = fit_governed_ensemble(panel, _config("static"))
    inference = _inference_frame(panel)
    inference["uncertainty"] = 1e308
    with pytest.raises(EnsembleContractError, match="inference arithmetic"):
        state.predict(_batch(inference))

    frame, folds, holdout_start = _training_frame()
    frame["prediction"] = 1e308
    extreme_panel = TrainingOOFPanel.from_frame(
        frame,
        folds=folds,
        expected_experts=EXPERTS,
        holdout_start=holdout_start,
        source_id="extreme-finite-oof",
    )
    with pytest.raises(EnsembleContractError, match="training"):
        fit_governed_ensemble(extreme_panel, _config("stacking"))


def test_zero_coefficient_stacking_retains_nonzero_heuristic_dispersion() -> None:
    frame, folds, holdout_start = _training_frame()
    frame["prediction"] = 0.0
    panel = TrainingOOFPanel.from_frame(
        frame,
        folds=folds,
        expected_experts=EXPERTS,
        holdout_start=holdout_start,
        source_id="zero-coefficient-oof",
    )
    state = fit_governed_ensemble(panel, _config("stacking"))
    inference = _inference_frame(panel)
    inference["prediction"] = 0.0
    decisions = state.predict(_batch(inference))

    assert list(dict(state.weights).values()) == pytest.approx([0.0, 0.0, 0.0])
    for decision in decisions:
        assert decision.uncertainty is not None
        assert decision.uncertainty > 0.0


def test_missing_expert_and_uncertain_regime_emit_explicit_abstentions() -> None:
    panel = _panel()
    missing = _inference_frame(panel)
    missing = missing[~((missing["symbol"] == "AAA") & (missing["expert"] == "trend"))].reset_index(
        drop=True
    )
    static = fit_governed_ensemble(panel, _config("static"))
    decisions = static.predict(_batch(missing))
    aaa = next(decision for decision in decisions if decision.symbol == "AAA")
    assert aaa.status == "abstained"
    assert aaa.reason == "missing_experts:trend"

    gate = fit_governed_ensemble(panel, _config("regime_gate", min_regime_rows=8))
    uncertain = gate.predict(_batch(_inference_frame(panel, regime=0.5)))
    assert {decision.status for decision in uncertain} == {"abstained"}
    assert {decision.reason for decision in uncertain} == {"uncertain_regime"}
    missing_regime = _inference_frame(panel)
    missing_regime["regime_probability"] = None
    missing_decisions = gate.predict(_batch(missing_regime))
    assert {decision.reason for decision in missing_decisions} == {"missing_or_inconsistent_regime"}
    inconsistent_regime = _inference_frame(panel)
    inconsistent_regime.loc[
        (inconsistent_regime["symbol"] == "AAA") & (inconsistent_regime["expert"] == "trend"),
        "regime_probability",
    ] = 0.8
    inconsistent_decisions = gate.predict(_batch(inconsistent_regime))
    aaa = next(decision for decision in inconsistent_decisions if decision.symbol == "AAA")
    assert aaa.reason == "missing_or_inconsistent_regime"


def test_rank_vote_abstains_for_whole_incomplete_date() -> None:
    panel = _panel()
    complete = _inference_frame(panel)
    frame = complete.copy()
    frame = frame[~((frame["symbol"] == "AAA") & (frame["expert"] == "trend"))].reset_index(
        drop=True
    )
    state = fit_governed_ensemble(panel, _config("rank_vote"))
    decisions = state.predict(_batch(frame))

    assert {decision.status for decision in decisions} == {"abstained"}
    assert {decision.reason for decision in decisions} == {"incomplete_date_for_rank_vote"}

    missing_symbol = complete[complete["symbol"] != "AAA"].reset_index(drop=True)
    decisions = state.predict(_batch(missing_symbol, expected_keys=_expected_keys(complete)))
    assert len(decisions) == 4
    assert {decision.reason for decision in decisions} == {"incomplete_date_for_rank_vote"}

    one_symbol = complete[complete["symbol"] == "AAA"].reset_index(drop=True)
    decisions = state.predict(_batch(one_symbol))
    assert {decision.reason for decision in decisions} == {"insufficient_rank_cross_section"}


def test_dynamic_updates_are_causal_and_last_target_cannot_rewrite_prior_state() -> None:
    frame, folds, holdout_start = _training_frame()
    panel = TrainingOOFPanel.from_frame(
        frame,
        folds=folds,
        expected_experts=EXPERTS,
        holdout_start=holdout_start,
        source_id="causal-original",
    )
    mutated = frame.copy()
    last_date = mutated["date"].max()
    mutated.loc[mutated["date"] == last_date, "target"] *= -100.0
    changed = TrainingOOFPanel.from_frame(
        mutated,
        folds=folds,
        expected_experts=EXPERTS,
        holdout_start=holdout_start,
        source_id="causal-mutated",
    )
    original_state = fit_governed_ensemble(panel, _config("dynamic"))
    changed_state = fit_governed_ensemble(changed, _config("dynamic"))

    assert original_state.audits[:-1] == changed_state.audits[:-1]
    assert original_state.audits[-1].weights_before == changed_state.audits[-1].weights_before
    assert original_state.audits[-1].weights_after != changed_state.audits[-1].weights_after
    assert original_state.audits[1].weights_before == original_state.audits[0].weights_after


def test_candidate_order_isolation_extends_to_fitted_state() -> None:
    frame, folds, holdout_start = _training_frame()
    first = TrainingOOFPanel.from_frame(
        frame,
        folds=folds,
        expected_experts=EXPERTS,
        holdout_start=holdout_start,
        source_id="order-isolation",
    )
    second = TrainingOOFPanel.from_frame(
        frame.sample(frac=1.0, random_state=91),
        folds=tuple(reversed(folds)),
        expected_experts=tuple(reversed(EXPERTS)),
        holdout_start=holdout_start,
        source_id="order-isolation",
    )
    state_a = fit_governed_ensemble(first, _config("bayesian"))
    state_b = fit_governed_ensemble(
        second,
        GovernedEnsembleConfig(method="bayesian", experts=tuple(reversed(EXPERTS))),
    )

    assert state_a.to_json() == state_b.to_json()


def test_inference_boundary_has_no_target_escape_hatch_or_duplicates() -> None:
    frame = _inference_frame(_panel())
    frame["target"] = 123.0
    with pytest.raises(EnsembleContractError, match="fields mismatch"):
        _batch(frame)

    duplicate = pd.concat([frame.drop(columns="target"), frame.drop(columns="target").iloc[[0]]])
    with pytest.raises(EnsembleContractError, match="unique"):
        _batch(duplicate)
    numeric_date = frame.drop(columns="target").copy()
    numeric_date["date"] = numeric_date["date"].astype(object)
    numeric_date.loc[0, "date"] = 1
    with pytest.raises(EnsembleContractError, match="calendar date"):
        _batch(numeric_date, expected_keys=_expected_keys(frame.drop(columns="target")))
    with pytest.raises(EnsembleContractError, match="tuples"):
        InferenceBatch(
            predictions=(),
            expected_keys=(["2024-01-01", "AAA"],),  # type: ignore[arg-type]
        )


def test_static_weight_validation_fails_closed() -> None:
    with pytest.raises(EnsembleContractError, match="degenerate"):
        _config(
            "static",
            static_weights=tuple((expert, 0.0) for expert in EXPERTS),
        )
    with pytest.raises(EnsembleContractError, match="match"):
        _config("static", static_weights=(("trend", 1.0),))


def test_state_identity_detects_tampering_and_size_overflow() -> None:
    state = fit_governed_ensemble(_panel(), _config("bayesian"))
    document = state.to_json().replace('"fallback_prediction":0.0', '"fallback_prediction":1.0')
    with pytest.raises(EnsembleContractError, match="identity mismatch"):
        GovernedEnsembleState.from_json(document)
    with pytest.raises(EnsembleContractError, match="max_bytes"):
        GovernedEnsembleState.from_json(state.to_json(), max_bytes=1)
    with pytest.raises(EnsembleContractError, match="ISO-8601"):
        replace(state, holdout_start="not-a-date")
    with pytest.raises(EnsembleContractError, match="fit_status"):
        replace(state, fit_status="garbage")  # type: ignore[arg-type]
    with pytest.raises(EnsembleContractError, match="canonical expert order"):
        replace(state, weights=tuple(reversed(state.weights)))

    dynamic = fit_governed_ensemble(_panel(), _config("dynamic"))
    with pytest.raises(EnsembleContractError, match="max_audit_records"):
        GovernedEnsembleState.from_json(dynamic.to_json(), max_audit_records=1)
    invalid_audit = replace(dynamic.audits[0], action="unrelated_update")
    with pytest.raises(EnsembleContractError, match="dynamic audit"):
        replace(dynamic, audits=(invalid_audit, *dynamic.audits[1:]))
    holdout_audit = replace(dynamic.audits[-1], effective_after=dynamic.holdout_start)
    with pytest.raises(EnsembleContractError, match="frozen holdout"):
        replace(dynamic, audits=(*dynamic.audits[:-1], holdout_audit))


def test_decision_and_audit_records_reject_noncanonical_state() -> None:
    state = fit_governed_ensemble(_panel(), _config("dynamic"))
    decision = state.predict(_batch(_inference_frame(_panel())))[0]
    with pytest.raises(EnsembleContractError, match="status"):
        replace(decision, status="bogus")  # type: ignore[arg-type]
    with pytest.raises(EnsembleContractError, match="SHA-256"):
        replace(decision, state_id="g" * 64)
    with pytest.raises(EnsembleContractError, match="canonical"):
        replace(decision, weights=tuple(reversed(decision.weights)))
    with pytest.raises(EnsembleContractError, match="unsupported"):
        EnsembleDecision(
            date=decision.date,
            symbol=decision.symbol,
            method="unknown",
            prediction=decision.prediction,
            uncertainty=decision.uncertainty,
            status="combined",
            reason=None,
            weights=decision.weights,
            state_id=decision.state_id,
        )
    with pytest.raises(EnsembleContractError, match="non-negative integer"):
        EnsembleAuditRecord(
            sequence=-1,
            effective_after="2024-01-01",
            action="update_after_oof_target_observed",
            status="updated",
            input_folds=(0,),
            weights_before=decision.weights,
            weights_after=decision.weights,
        )


def test_final_holdout_predictions_do_not_mutate_state() -> None:
    panel = _panel()
    state = fit_governed_ensemble(panel, _config("dynamic"))
    before = state.to_json()
    state.predict(_batch(_inference_frame(panel)))
    assert state.to_json() == before


def test_panel_records_are_immutable() -> None:
    panel = _panel()
    with pytest.raises(FrozenInstanceError):
        panel.predictions[0].prediction = 99.0  # type: ignore[misc]
    assert replace(panel.predictions[0]) == panel.predictions[0]
