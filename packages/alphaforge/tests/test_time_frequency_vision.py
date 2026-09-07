"""Contracts, gates, robustness, and bounded-model tests for SF-S3-MR7."""

from __future__ import annotations

from dataclasses import replace
from typing import Any

import numpy as np
import pandas as pd
import pytest

from alphaforge.models.base import ModelError  # noqa: E402
from alphaforge.models.time_frequency_vision import (  # noqa: E402
    ControlledVisionModel,
    PredictionMetrics,
    ProgressionGatePolicy,
    TimeFrequencyBatch,
    VisionAugmentationConfig,
    VisionTrainingConfig,
    augment_time_frequency,
    authorize_architecture,
    evaluate_progression_gate,
    prediction_metrics,
)
from alphaforge.models.torch_models import TORCH_AVAILABLE  # noqa: E402


def _batch(
    *,
    dates: int = 12,
    symbols: int = 6,
    seed: int = 19,
) -> TimeFrequencyBatch:
    generator = np.random.default_rng(seed)
    samples = dates * symbols
    values = generator.lognormal(-1.0, 0.3, (samples, 2, 6, 3)).astype(np.float32)
    date_values = np.repeat(pd.bdate_range("2025-01-02", periods=dates).to_numpy(), symbols)
    symbol_values = np.tile(np.asarray([f"S{i}" for i in range(symbols)]), dates)
    target = 0.02 * (values[:, 0, :2, -1].mean(axis=1) - values[:, 1].mean(axis=(1, 2)))
    return TimeFrequencyBatch(
        values=values,
        observed_mask=np.ones((samples, 2), dtype=bool),
        dates=date_values,
        symbols=symbol_values.astype(object),
        target=target,
        time_features=np.column_stack([values.mean(axis=(1, 2, 3)), values.std(axis=(1, 2, 3))]),
        spectral_features=np.column_stack(
            [values[:, :, :2].mean(axis=(1, 2, 3)), values[:, :, -2:].mean(axis=(1, 2, 3))]
        ),
        channels=("return", "volume"),
        frequency_values=tuple(np.linspace(0.0, 0.5, 6)),
        representation="spectrogram",
    )


def _metrics(
    *,
    rank_ic: float = 0.2,
    rmse: float = 0.1,
    partition: str = "validation",
) -> PredictionMetrics:
    return PredictionMetrics(
        partition=partition,  # type: ignore[arg-type]
        observations=128,
        dates=16,
        rmse=rmse,
        mae=rmse * 0.8,
        rank_ic=rank_ic,
    )


def _passing_gate(
    candidate: str,
    policy: ProgressionGatePolicy,
) -> Any:
    return evaluate_progression_gate(
        candidate=candidate,  # type: ignore[arg-type]
        baseline="baseline",
        candidate_metrics=_metrics(rank_ic=0.3, rmse=0.08),
        baseline_metrics=_metrics(rank_ic=0.1, rmse=0.1),
        policy=policy,
    )


FAST_CONFIG = VisionTrainingConfig(
    width=8,
    residual_blocks=1,
    vit_layers=1,
    vit_heads=2,
    max_epochs=2,
    patience=2,
    batch_size=64,
    max_parameters=100_000,
    seed=20260726,
    device="cpu",
    augmentation=VisionAugmentationConfig(probability=0.0),
)


def test_batch_contract_and_subset_preserve_exact_alignment() -> None:
    batch = _batch()
    selector = np.arange(len(batch.target)) % 2 == 0

    subset = batch.take(selector)

    np.testing.assert_array_equal(subset.dates, batch.dates[selector])
    np.testing.assert_array_equal(subset.symbols, batch.symbols[selector])
    np.testing.assert_allclose(subset.target, batch.target[selector])
    assert subset.values.shape[0] == int(selector.sum())
    assert subset.complete_samples.all()


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda kwargs: kwargs.update(values=np.zeros((2, 3, 4))), "shape"),
        (
            lambda kwargs: kwargs.update(observed_mask=np.ones((72, 1), dtype=bool)),
            "observed_mask",
        ),
        (lambda kwargs: kwargs.update(channels=("return", "return")), "unique"),
        (
            lambda kwargs: kwargs.update(frequency_values=(0.0, 0.2, 0.1, 0.3, 0.4, 0.5)),
            "ascending",
        ),
        (lambda kwargs: kwargs.update(target=np.full(72, np.inf)), "finite"),
        (
            lambda kwargs: kwargs.update(time_features=np.full((72, 2), np.nan)),
            "time_features",
        ),
    ],
)
def test_batch_rejects_malformed_untrusted_inputs(mutation, message: str) -> None:
    batch = _batch()
    kwargs: dict[str, Any] = {
        "values": batch.values.copy(),
        "observed_mask": batch.observed_mask.copy(),
        "dates": batch.dates.copy(),
        "symbols": batch.symbols.copy(),
        "target": batch.target.copy(),
        "time_features": batch.time_features.copy(),
        "spectral_features": batch.spectral_features.copy(),
        "channels": batch.channels,
        "frequency_values": batch.frequency_values,
        "representation": batch.representation,
    }
    mutation(kwargs)
    with pytest.raises(ValueError, match=message):
        TimeFrequencyBatch(**kwargs)


def test_masked_surfaces_may_be_nan_but_observed_surfaces_may_not() -> None:
    batch = _batch()
    values = batch.values.copy()
    mask = batch.observed_mask.copy()
    mask[0, 0] = False
    values[0, 0] = np.nan
    accepted = replace(batch, values=values, observed_mask=mask)
    assert not accepted.complete_samples[0]

    mask[0, 0] = True
    with pytest.raises(ValueError, match="observed tensor"):
        replace(batch, values=values, observed_mask=mask)


def test_tensor_allocation_budget_fails_before_model_training() -> None:
    batch = _batch()
    with pytest.raises(ValueError, match="max_tensor_bytes"):
        replace(batch, max_tensor_bytes=1)


def test_augmentation_is_deterministic_positive_and_axis_preserving() -> None:
    values = np.ones((32, 2, 6, 3), dtype=np.float32)
    amplitude_only = VisionAugmentationConfig(
        probability=1.0,
        log_amplitude_std=0.1,
        max_frequency_mask_bins=0,
    )
    first = augment_time_frequency(values, amplitude_only, seed=7)
    second = augment_time_frequency(values, amplitude_only, seed=7)

    np.testing.assert_array_equal(first, second)
    assert first.shape == values.shape
    assert (first > 0.0).all()
    for sample in first:
        for channel in sample:
            assert np.unique(channel).size == 1

    masked = augment_time_frequency(
        values,
        VisionAugmentationConfig(
            probability=1.0,
            log_amplitude_std=0.0,
            max_frequency_mask_bins=2,
        ),
        seed=8,
    )
    assert (masked == 0.0).any()
    assert np.all((masked == 0.0).all(axis=(1, 3)).sum(axis=1) <= 2)


def test_augmentation_rejects_nonfinite_or_wrong_shape() -> None:
    with pytest.raises(ValueError, match="four-dimensional"):
        augment_time_frequency(np.zeros((2, 3)), VisionAugmentationConfig(), seed=1)
    values = np.zeros((2, 1, 2, 2), dtype=np.float32)
    values[0, 0, 0, 0] = np.inf
    with pytest.raises(ValueError, match="finite"):
        augment_time_frequency(values, VisionAugmentationConfig(), seed=1)


def test_gate_is_validation_only_tamper_evident_and_policy_bound() -> None:
    policy = ProgressionGatePolicy()
    gate = _passing_gate("small_cnn", policy)
    gate.verify()
    assert gate.passed
    assert dict(gate.checks) == {
        "minimum_observations": True,
        "minimum_dates": True,
        "minimum_rank_ic": True,
        "minimum_incremental_rank_ic": True,
        "maximum_rmse_ratio": True,
    }

    with pytest.raises(ModelError, match="digest mismatch"):
        replace(gate, baseline="tampered").verify()
    with pytest.raises(ValueError, match="validation"):
        evaluate_progression_gate(
            candidate="small_cnn",
            baseline="baseline",
            candidate_metrics=_metrics(partition="test"),
            baseline_metrics=_metrics(),
            policy=policy,
        )
    with pytest.raises(ValueError, match="only small_cnn and resnet"):
        evaluate_progression_gate(
            candidate="vit",
            baseline="baseline",
            candidate_metrics=_metrics(),
            baseline_metrics=_metrics(),
            policy=policy,
        )


def test_failed_or_wrong_predecessor_gate_blocks_larger_architecture() -> None:
    policy = ProgressionGatePolicy(minimum_rank_ic=0.5)
    failed = evaluate_progression_gate(
        candidate="small_cnn",
        baseline="spectral",
        candidate_metrics=_metrics(rank_ic=0.1),
        baseline_metrics=_metrics(rank_ic=0.2),
        policy=policy,
    )
    assert not failed.passed
    with pytest.raises(ModelError, match="blocked"):
        authorize_architecture("resnet", policy=policy, prior_gate=failed)
    with pytest.raises(ModelError, match="requires verified"):
        authorize_architecture("vit", policy=policy, prior_gate=None)

    passing_small = _passing_gate("small_cnn", ProgressionGatePolicy())
    with pytest.raises(ModelError, match="passed resnet gate"):
        authorize_architecture(
            "vit",
            policy=ProgressionGatePolicy(),
            prior_gate=passing_small,
        )


@pytest.mark.parametrize("architecture", ["small_cnn", "resnet", "vit"])
@pytest.mark.skipif(not TORCH_AVAILABLE, reason="torch optional dependency is unavailable")
def test_every_authorized_architecture_trains_under_common_resource_contract(
    architecture: str,
) -> None:
    batch = _batch(dates=10)
    rows = np.arange(len(batch.target))
    training = rows < 42
    validation = ~training
    policy = ProgressionGatePolicy()
    prior = None
    if architecture == "resnet":
        prior = _passing_gate("small_cnn", policy)
    elif architecture == "vit":
        prior = _passing_gate("resnet", policy)
    model = ControlledVisionModel(
        architecture,  # type: ignore[arg-type]
        FAST_CONFIG,
        policy=policy,
        prior_gate=prior,
    ).fit(
        batch.values[training],
        batch.target[training],
        batch.values[validation],
        batch.target[validation],
    )

    prediction = model.predict(batch.values[validation])
    evidence = model.resource_evidence()

    assert prediction.shape == (int(validation.sum()),)
    assert np.isfinite(prediction).all()
    assert evidence.architecture == architecture
    assert 0 < evidence.parameter_count <= FAST_CONFIG.max_parameters
    assert evidence.parameter_bytes >= evidence.parameter_count * 4
    assert evidence.fit_wall_seconds > 0.0
    assert evidence.fit_cpu_seconds > 0.0
    assert 1 <= evidence.best_epoch <= evidence.epochs_completed


@pytest.mark.skipif(not TORCH_AVAILABLE, reason="torch optional dependency is unavailable")
def test_normalization_is_fit_on_training_tensor_and_target_only() -> None:
    batch = _batch(dates=10)
    training = np.arange(len(batch.target)) < 42
    validation = ~training
    mutated_validation = batch.values[validation] * 10_000.0
    mutated_target = batch.target[validation] * -10_000.0

    model = ControlledVisionModel(
        "small_cnn",
        FAST_CONFIG,
        policy=ProgressionGatePolicy(),
    ).fit(
        batch.values[training],
        batch.target[training],
        mutated_validation,
        mutated_target,
    )

    assert model.mean_ is not None
    np.testing.assert_allclose(model.mean_, batch.values[training].mean(axis=0, dtype=np.float64))
    assert model.target_mean_ == pytest.approx(float(batch.target[training].mean()))


@pytest.mark.skipif(not TORCH_AVAILABLE, reason="torch optional dependency is unavailable")
def test_future_samples_cannot_change_past_predictions() -> None:
    batch = _batch(dates=10)
    training = np.arange(len(batch.target)) < 42
    validation = ~training
    model = ControlledVisionModel(
        "small_cnn",
        FAST_CONFIG,
        policy=ProgressionGatePolicy(),
    ).fit(
        batch.values[training],
        batch.target[training],
        batch.values[validation],
        batch.target[validation],
    )
    baseline = model.predict(batch.values)
    mutated = batch.values.copy()
    mutated[50:] *= 1_000.0

    changed = model.predict(mutated)

    np.testing.assert_allclose(baseline[:50], changed[:50], rtol=0.0, atol=1e-8)


@pytest.mark.skipif(not TORCH_AVAILABLE, reason="torch optional dependency is unavailable")
def test_parameter_budget_and_unavailable_state_fail_closed() -> None:
    batch = _batch(dates=10)
    model = ControlledVisionModel(
        "small_cnn",
        replace(FAST_CONFIG, max_parameters=1),
        policy=ProgressionGatePolicy(),
    )
    with pytest.raises(ModelError, match="max_parameters"):
        model.fit(
            batch.values[:42],
            batch.target[:42],
            batch.values[42:],
            batch.target[42:],
        )
    with pytest.raises(ModelError, match="fit before"):
        ControlledVisionModel(
            "small_cnn",
            FAST_CONFIG,
            policy=ProgressionGatePolicy(),
        ).predict(batch.values)


def test_prediction_metrics_are_cross_sectional_and_partition_typed() -> None:
    batch = _batch(dates=4)
    metrics = prediction_metrics(
        batch.target,
        batch.target * 2.0,
        batch.dates,
        partition="test",
    )
    assert metrics.observations == len(batch.target)
    assert metrics.dates == 4
    assert metrics.rank_ic == pytest.approx(1.0)
    assert metrics.rmse > 0.0

    with pytest.raises(ValueError, match="rank IC"):
        prediction_metrics(
            np.ones(8),
            np.ones(8),
            np.repeat(pd.Timestamp("2025-01-02"), 8),
            partition="validation",
        )
