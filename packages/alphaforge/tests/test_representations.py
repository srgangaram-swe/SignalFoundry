"""Invariant and adversarial tests for SF-S3-MR8 representation contracts."""

from __future__ import annotations

import inspect
from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from alphaforge.representations import (
    EmbeddingCollapseError,
    LinearRepresentation,
    RepresentationBatch,
    RepresentationCapabilityError,
    RepresentationConfig,
    RepresentationError,
    RepresentationNotFittedError,
    RepresentationResourceError,
    RepresentationSchemaError,
    canonicalize_component_signs,
    create_representation,
    subspace_distance,
    subspace_fingerprint,
)
from alphaforge.representations.neural import (
    augment_causal_windows,
    build_causal_windows,
)


def _batch(
    *,
    n_dates: int = 24,
    n_symbols: int = 3,
    n_features: int = 6,
    seed: int = 17,
) -> RepresentationBatch:
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2024-01-02", periods=n_dates)
    date_factor = rng.normal(size=(n_dates, 3))
    symbol_factor = rng.normal(size=(n_symbols, 3))
    loading = rng.normal(size=(3, n_features))
    rows = []
    for date_index in range(n_dates):
        for symbol_index in range(n_symbols):
            latent = date_factor[date_index] + 0.3 * symbol_factor[symbol_index]
            rows.append(latent @ loading + rng.normal(0.0, 0.02, size=n_features))
    return RepresentationBatch(
        values=np.asarray(rows),
        dates=tuple(str(value) for value in np.repeat(dates.to_numpy(), n_symbols)),
        symbols=tuple(
            str(value) for value in np.tile([f"S{index}" for index in range(n_symbols)], n_dates)
        ),
        feature_names=tuple(f"x{index}" for index in range(n_features)),
    )


def _config(kind: str, **overrides: object) -> RepresentationConfig:
    defaults: dict[str, object] = {
        "kind": kind,
        "latent_dim": 3,
        "seed": 29,
        "hidden_dim": 8,
        "sequence_length": 5,
        "batch_size": 16,
        "incremental_batch_size": 24,
        "max_epochs": 4,
        "patience": 2,
        "learning_rate": 0.005,
        "validation_fraction": 0.25,
        "max_samples": 1000,
        "max_features": 32,
        "max_tensor_bytes": 16 * 1024**2,
        "max_parameters": 100_000,
    }
    defaults.update(overrides)
    return RepresentationConfig(**defaults)  # type: ignore[arg-type]


def test_batch_copies_inputs_is_read_only_and_has_no_label_surface() -> None:
    values = np.arange(24, dtype=float).reshape(6, 4)
    batch = RepresentationBatch(
        values=values,
        dates=tuple(str(value) for value in np.repeat(pd.bdate_range("2024-01-02", periods=3), 2)),
        symbols=("A", "B") * 3,
        feature_names=("a", "b", "c", "d"),
    )
    values[:] = -1.0

    assert batch.values[0, 0] == 0.0
    assert not batch.values.flags.writeable
    assert "target" not in inspect.signature(RepresentationBatch).parameters
    with pytest.raises(ValueError, match="read-only"):
        batch.values[0, 0] = 7.0


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda values: values.__setitem__((0, 0), np.nan), "finite"),
        (lambda values: values.__setitem__((0, 0), np.inf), "finite"),
    ],
)
def test_batch_rejects_nonfinite_values(mutation, message: str) -> None:
    values = np.ones((4, 2))
    mutation(values)
    with pytest.raises(RepresentationSchemaError, match=message):
        RepresentationBatch(
            values=values,
            dates=("2024-01-01", "2024-01-01", "2024-01-02", "2024-01-02"),
            symbols=("A", "B", "A", "B"),
            feature_names=("x", "y"),
        )


def test_batch_rejects_duplicate_or_nonchronological_identity() -> None:
    with pytest.raises(RepresentationSchemaError, match="unique"):
        RepresentationBatch(
            values=np.ones((2, 2)),
            dates=("2024-01-01", "2024-01-01"),
            symbols=("A", "A"),
            feature_names=("x", "y"),
        )
    with pytest.raises(RepresentationSchemaError, match="monotonically"):
        RepresentationBatch(
            values=np.ones((2, 2)),
            dates=("2024-01-02", "2024-01-01"),
            symbols=("A", "A"),
            feature_names=("x", "y"),
        )


@pytest.mark.parametrize(
    "overrides",
    [
        {"latent_dim": 0},
        {"hidden_dim": 1, "latent_dim": 2},
        {"seed": -1},
        {"batch_size": 1},
        {"patience": 5, "max_epochs": 4},
        {"validation_fraction": 0.0},
        {"augmentation_mask_probability": 0.5},
        {"learning_rate": np.nan},
        {"max_tensor_bytes": 0},
    ],
)
def test_config_rejects_unsafe_or_unbounded_values(overrides: dict[str, object]) -> None:
    with pytest.raises(RepresentationError):
        _config("pca", **overrides)


def test_component_sign_and_rotation_ambiguity_have_invariant_subspace_identity() -> None:
    basis = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    angle = 0.37
    rotation = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
    rotated = rotation @ basis
    signed = basis * np.array([[-1.0], [1.0]])

    assert subspace_fingerprint(basis) == subspace_fingerprint(rotated)
    assert subspace_fingerprint(basis) == subspace_fingerprint(signed)
    assert subspace_distance(basis, rotated) < 1e-12
    canonical = canonicalize_component_signs(signed)
    assert canonical[0, np.argmax(np.abs(canonical[0]))] > 0.0


def test_pca_reconstructs_low_rank_reference_and_is_deterministic() -> None:
    rng = np.random.default_rng(4)
    factors = rng.normal(size=(90, 2))
    values = factors @ rng.normal(size=(2, 5))
    dates = pd.bdate_range("2024-01-02", periods=30)
    batch = RepresentationBatch(
        values=values,
        dates=tuple(str(value) for value in np.repeat(dates.to_numpy(), 3)),
        symbols=("A", "B", "C") * 30,
        feature_names=tuple(f"x{index}" for index in range(5)),
    )
    config = _config("pca", latent_dim=2)
    first = LinearRepresentation(config).fit(batch)
    second = LinearRepresentation(config).fit(batch)

    assert first.state_ is not None and second.state_ is not None
    assert first.state_.state_id == second.state_.state_id
    assert first.state_.fit_batch_id == batch.batch_id
    np.testing.assert_allclose(first.reconstruct(batch), values, atol=1e-10)
    np.testing.assert_array_equal(first.transform(batch).values, second.transform(batch).values)


def test_incremental_and_full_pca_recover_nearly_identical_subspaces() -> None:
    batch = _batch(n_dates=50)
    full = LinearRepresentation(_config("pca")).fit(batch)
    incremental = LinearRepresentation(_config("incremental_pca")).fit(batch)
    assert full._model is not None and incremental._model is not None
    assert subspace_distance(full._model.components_, incremental._model.components_) < 0.08
    assert incremental.state_ is not None
    assert incremental.state_.iterations > 1


def test_robust_scaling_pca_publishes_honest_mechanism_and_finite_output() -> None:
    batch = _batch()
    values = batch.values.copy()
    values[-1] += 1_000.0
    contaminated = RepresentationBatch(
        values=values,
        dates=batch.dates,
        symbols=batch.symbols,
        feature_names=batch.feature_names,
    )
    representation = LinearRepresentation(_config("robust_pca")).fit(contaminated)

    assert representation.state_ is not None
    assert any("not principal-component pursuit" in note for note in representation.state_.notes)
    assert np.isfinite(representation.transform(contaminated).values).all()
    assert np.isfinite(representation.reconstruct(contaminated)).all()


def test_holdout_mutation_cannot_change_linear_fitted_state() -> None:
    batch = _batch()
    train = batch.take(np.arange(0, 48))
    holdout = batch.take(np.arange(48, batch.n_rows))
    representation = LinearRepresentation(_config("pca")).fit(train)
    state_id = representation.state_.state_id if representation.state_ else ""
    mutated = RepresentationBatch(
        values=holdout.values * 1000.0,
        dates=holdout.dates,
        symbols=holdout.symbols,
        feature_names=holdout.feature_names,
    )

    representation.transform(mutated)
    assert representation.state_ is not None
    assert representation.state_.state_id == state_id


def test_inference_before_fit_schema_drift_and_resource_overflow_fail_closed() -> None:
    batch = _batch()
    representation = LinearRepresentation(_config("pca"))
    with pytest.raises(RepresentationNotFittedError):
        representation.transform(batch)

    representation.fit(batch)
    drifted = RepresentationBatch(
        values=batch.values,
        dates=batch.dates,
        symbols=batch.symbols,
        feature_names=tuple(reversed(batch.feature_names)),
    )
    with pytest.raises(RepresentationSchemaError, match="schema mismatch"):
        representation.transform(drifted)

    with pytest.raises(RepresentationResourceError, match="sample count"):
        LinearRepresentation(_config("pca", max_samples=10)).fit(batch)


def test_raw_control_round_trips_and_collapse_is_rejected() -> None:
    batch = _batch()
    raw = LinearRepresentation(_config("raw")).fit(batch)
    np.testing.assert_allclose(raw.reconstruct(batch), batch.values, atol=1e-12)
    assert raw.transform(batch).values.shape == batch.values.shape

    constant = RepresentationBatch(
        values=np.ones_like(batch.values),
        dates=batch.dates,
        symbols=batch.symbols,
        feature_names=batch.feature_names,
    )
    with pytest.raises(EmbeddingCollapseError):
        LinearRepresentation(_config("pca")).fit(constant)


def test_causal_windows_ignore_future_mutation_and_never_mix_symbols() -> None:
    batch = _batch(n_dates=12, n_symbols=2, n_features=3)
    first = build_causal_windows(
        batch,
        batch.values,
        sequence_length=4,
        max_tensor_bytes=2 * 1024**2,
    )
    values = batch.values.copy()
    values[16:] += 1000.0
    mutated = RepresentationBatch(
        values=values,
        dates=batch.dates,
        symbols=batch.symbols,
        feature_names=batch.feature_names,
    )
    second = build_causal_windows(
        mutated,
        mutated.values,
        sequence_length=4,
        max_tensor_bytes=2 * 1024**2,
    )

    np.testing.assert_array_equal(first[:16], second[:16])
    assert first[0, -1, -1] == 1.0
    assert first[0, :-1, -1].sum() == 0.0
    assert first[2, -2, 0] == pytest.approx(batch.values[0, 0])


def test_temporal_augmentation_is_deterministic_bounded_and_preserves_current_step() -> None:
    batch = _batch(n_dates=8, n_symbols=2, n_features=3)
    windows = build_causal_windows(
        batch,
        batch.values,
        sequence_length=4,
        max_tensor_bytes=2 * 1024**2,
    )
    config = _config("contrastive_timeseries", latent_dim=2)
    first = augment_causal_windows(windows, config, np.random.default_rng(9))
    second = augment_causal_windows(windows, config, np.random.default_rng(9))

    np.testing.assert_array_equal(first, second)
    np.testing.assert_array_equal(first[:, -1, -1], windows[:, -1, -1])
    assert not np.shares_memory(first, windows)


@pytest.mark.parametrize(
    "kind",
    [
        "dense_autoencoder",
        "sequence_autoencoder",
        "denoising_autoencoder",
        "variational_autoencoder",
        "contrastive_timeseries",
    ],
)
def test_neural_representations_are_cpu_deterministic_and_candidate_isolated(kind: str) -> None:
    pytest.importorskip("torch")
    batch = _batch(n_dates=18, n_symbols=3)
    config = _config(kind, max_epochs=3, patience=2)
    first = create_representation(config).fit(batch)
    second = create_representation(config).fit(batch)

    assert first.state_ is not None and second.state_ is not None
    assert first.state_.state_id == second.state_.state_id
    assert first.state_.device == "cpu"
    assert first.state_.parameter_count <= config.max_parameters
    np.testing.assert_allclose(
        first.transform(batch).values,
        second.transform(batch).values,
        rtol=0.0,
        atol=1e-7,
    )
    np.testing.assert_array_equal(
        first.transform(batch).values,
        first.transform(batch).values,
    )
    if kind == "contrastive_timeseries":
        with pytest.raises(RepresentationCapabilityError):
            first.reconstruct(batch)
    else:
        reconstruction = first.reconstruct(batch)
        assert reconstruction.shape == batch.values.shape
        assert np.isfinite(reconstruction).all()


def test_neural_collapse_threshold_and_extra_target_argument_fail_closed() -> None:
    pytest.importorskip("torch")
    batch = _batch(n_dates=12, n_symbols=2)
    representation = create_representation(
        _config(
            "dense_autoencoder",
            max_epochs=2,
            patience=1,
            min_embedding_variance=1e9,
        )
    )
    with pytest.raises(EmbeddingCollapseError):
        representation.fit(batch)

    ordinary = create_representation(_config("dense_autoencoder", max_epochs=2, patience=1))
    with pytest.raises(TypeError):
        ordinary.fit(batch, np.zeros(batch.n_rows))  # type: ignore[call-arg]


def test_candidate_named_seed_is_unchanged_by_unrelated_config_replacement() -> None:
    config = _config("pca")
    changed = replace(config, max_epochs=config.max_epochs + 1)
    assert config.seed == changed.seed
    assert config.config_id != changed.config_id
