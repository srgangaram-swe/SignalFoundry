"""Adversarial, reproducibility, and integration tests for SF-S3-MR6."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

torch = pytest.importorskip("torch")

from alphaforge.models.base import AlphaModel, ModelError  # noqa: E402
from alphaforge.models.deep_sequence import (  # noqa: E402
    Architecture,
    ControlledSequenceModel,
    build_causal_windows,
    evaluate_oos_predictions,
    resolve_sequence_device,
)
from alphaforge.models.registry import create_model, seed_model_specs  # noqa: E402
from alphaforge.training import run_walk_forward  # noqa: E402

ARCHITECTURES: tuple[Architecture, ...] = ("cnn", "tcn", "lstm", "gru", "transformer")
FAST_CONFIG: dict[str, Any] = {
    "seq_len": 5,
    "hidden_size": 8,
    "n_layers": 1,
    "n_heads": 2,
    "max_epochs": 2,
    "patience": 2,
    "batch_size": 64,
    "validation_fraction": 0.2,
    "seed": 20260726,
    "device": "cpu",
}


def _panel(
    *,
    symbols: int = 3,
    days: int = 28,
    seed: int = 7,
) -> tuple[pd.DataFrame, pd.Series]:
    generator = np.random.default_rng(seed)
    dates = pd.bdate_range("2024-01-02", periods=days)
    records: list[dict[str, object]] = []
    for symbol_index in range(symbols):
        feature = generator.normal(size=(days, 3))
        target = (
            0.4 * pd.Series(feature[:, 0]).rolling(3, min_periods=1).mean().to_numpy()
            - 0.2 * feature[:, 1]
        )
        for date, row, response in zip(dates, feature, target):
            records.append(
                {
                    "date": date,
                    "symbol": f"S{symbol_index}",
                    "f0": row[0],
                    "f1": row[1],
                    "f2": row[2],
                    "target": response,
                }
            )
    frame = pd.DataFrame(records).sort_values(["date", "symbol"]).reset_index(drop=True)
    index = pd.MultiIndex.from_frame(frame[["date", "symbol"]])
    features = frame[["f0", "f1", "f2"]].set_axis(index)
    target = pd.Series(frame["target"].to_numpy(), index=index, name="target")
    return features, target


def test_causal_windows_left_pad_and_never_cross_symbols() -> None:
    features, target = _panel(symbols=2, days=5)
    windows = build_causal_windows(
        features,
        target,
        seq_len=4,
        min_history=2,
        max_windows=20,
    )

    assert windows.values.shape == (8, 4, 3)
    assert windows.valid_mask.shape == (8, 4)
    assert np.all(windows.valid_mask.sum(axis=1) >= 2)
    for values, mask, position, symbol in zip(
        windows.values,
        windows.valid_mask,
        windows.row_positions,
        windows.symbols,
    ):
        assert not mask[: -int(mask.sum())].any()
        expected_symbol = features.index[position][1]
        assert symbol == expected_symbol
        source_rows = features.xs(symbol, level=1).loc[: features.index[position][0]].tail(4)
        np.testing.assert_allclose(values[mask], source_rows.to_numpy(dtype=np.float32))


def test_window_builder_is_input_order_independent() -> None:
    features, target = _panel()
    permutation = np.random.default_rng(17).permutation(len(features))
    shuffled_features = features.iloc[permutation]
    shuffled_target = target.iloc[permutation]

    first = build_causal_windows(features, target, seq_len=5)
    second = build_causal_windows(shuffled_features, shuffled_target, seq_len=5)
    first_order = np.lexsort((first.end_dates, first.symbols))
    second_order = np.lexsort((second.end_dates, second.symbols))

    np.testing.assert_allclose(first.values[first_order], second.values[second_order])
    assert first.targets is not None
    assert second.targets is not None
    np.testing.assert_allclose(first.targets[first_order], second.targets[second_order])


@pytest.mark.parametrize("architecture", ARCHITECTURES)
def test_every_architecture_obeys_common_contract_and_resource_budget(
    architecture: Architecture,
) -> None:
    features, target = _panel()
    model = ControlledSequenceModel(architecture, **FAST_CONFIG).fit(features, target)
    prediction = model.predict(features)
    evidence = model.resource_evidence()
    diagnostics = model.training_diagnostics()

    assert prediction.shape == (len(features),)
    assert np.isfinite(prediction).all()
    assert evidence.architecture == architecture
    assert 0 < evidence.parameter_count <= FAST_CONFIG.get("max_parameters", 2_000_000)
    assert evidence.parameter_bytes >= evidence.parameter_count * 4
    assert evidence.fit_wall_seconds > 0.0
    assert evidence.fit_cpu_seconds > 0.0
    assert evidence.device == "cpu"
    assert evidence.epochs_completed == len(evidence.validation_loss)
    assert 1 <= evidence.best_epoch <= evidence.epochs_completed
    assert np.isfinite(evidence.best_validation_loss)
    assert diagnostics.backend == f"torch-{architecture}-cpu"
    assert diagnostics.seed == FAST_CONFIG["seed"]


def test_transform_is_fitted_only_on_inner_training_dates() -> None:
    features, target = _panel(days=30)
    dates = features.index.get_level_values(0).to_numpy()
    unique_dates = np.unique(dates)
    validation_dates = int(np.ceil(len(unique_dates) * FAST_CONFIG["validation_fraction"]))
    validation_start = unique_dates[-validation_dates]
    expected = np.mean(features.to_numpy()[dates < validation_start], axis=0)

    model = ControlledSequenceModel("gru", **FAST_CONFIG).fit(features, target)

    assert model.x_mean_ is not None
    np.testing.assert_allclose(model.x_mean_, expected, rtol=0.0, atol=1e-12)


def test_future_mutation_cannot_change_past_predictions() -> None:
    features, target = _panel(days=32)
    model = ControlledSequenceModel("tcn", **FAST_CONFIG).fit(features, target)
    baseline = model.predict(features)
    dates = features.index.get_level_values(0)
    cutoff = dates.unique().sort_values()[20]
    mutated = features.copy()
    mutated.loc[dates > cutoff, :] *= 10_000.0

    changed = model.predict(mutated)

    np.testing.assert_allclose(baseline[dates <= cutoff], changed[dates <= cutoff], atol=1e-7)


def test_cross_asset_values_cannot_change_other_asset_predictions() -> None:
    features, target = _panel(days=30)
    model = ControlledSequenceModel("lstm", **FAST_CONFIG).fit(features, target)
    baseline = model.predict(features)
    mutated = features.copy()
    symbol = features.index.get_level_values(1)
    mutated.loc[symbol == "S2", :] = -99_999.0

    changed = model.predict(mutated)

    np.testing.assert_allclose(baseline[symbol == "S0"], changed[symbol == "S0"], atol=1e-7)


def test_batch_and_input_order_do_not_change_fitted_predictions() -> None:
    features, target = _panel(days=24)
    permutation = np.random.default_rng(99).permutation(len(features))
    first = ControlledSequenceModel("gru", **FAST_CONFIG).fit(features, target)
    second = ControlledSequenceModel("gru", **FAST_CONFIG).fit(
        features.iloc[permutation],
        target.iloc[permutation],
    )

    np.testing.assert_allclose(
        first.predict(features),
        second.predict(features),
        rtol=1e-6,
        atol=1e-6,
    )


def test_trusted_save_load_round_trip(tmp_path: Path) -> None:
    features, target = _panel(days=24)
    model = create_model("sequence_cnn", **FAST_CONFIG).fit(features, target)
    artifact = tmp_path / "sequence-model.joblib"
    model.save(artifact)

    with pytest.raises(ModelError, match="trusted=True"):
        AlphaModel.load(artifact)
    restored = AlphaModel.load(artifact, trusted=True)

    np.testing.assert_allclose(model.predict(features), restored.predict(features), atol=1e-7)


def test_auto_device_is_cpu_and_unavailable_accelerator_fails_closed(monkeypatch) -> None:
    assert resolve_sequence_device("auto") == "cpu"
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(ModelError, match="unavailable"):
        resolve_sequence_device("cuda")


def test_registry_and_seed_injection_cover_complete_family() -> None:
    names = [f"sequence_{architecture}" for architecture in ARCHITECTURES]
    specs = seed_model_specs([{"name": name, "params": {}} for name in names], 123)

    assert [spec["params"]["seed"] for spec in specs] == [123] * len(names)
    assert all(create_model(name).needs_sequence_index for name in names)


def test_walk_forward_emits_resource_and_convergence_evidence() -> None:
    features, target = _panel(days=28)
    identifiers = features.index.to_frame(index=False)
    feature_frame = pd.concat(
        [identifiers.reset_index(drop=True), features.reset_index(drop=True)],
        axis=1,
    )
    label_frame = identifiers.copy()
    label_frame["forward_return"] = target.to_numpy()

    result = run_walk_forward(
        feature_frame,
        label_frame,
        model_specs=[{"name": "sequence_gru", "params": FAST_CONFIG}],
        target="forward_return",
        config={
            "min_train_days": 15,
            "test_days": 5,
            "step_days": 5,
            "embargo_days": 1,
            "max_windows": 1,
        },
        max_horizon=1,
    )

    assert len(result.metrics) == 1
    row = result.metrics.iloc[0]
    assert row["training_backend"] == "torch-gru-cpu"
    assert row["parameter_count"] > 0
    assert row["fit_wall_seconds"] > 0.0
    assert np.isfinite(row["best_validation_loss"])


def test_oos_evidence_reports_calibration_costs_and_lightgbm_comparability() -> None:
    dates = np.repeat(pd.bdate_range("2025-01-02", periods=4), 5)
    symbols = np.tile([f"S{index}" for index in range(5)], 4)
    target = np.tile(np.linspace(-0.02, 0.02, 5), 4)
    frame = pd.DataFrame(
        {
            "date": dates,
            "symbol": symbols,
            "target": target,
            "prediction": target * 0.8,
        }
    )

    sequence = evaluate_oos_predictions(
        frame,
        model="sequence_tcn",
        transaction_cost_bps=5.0,
    )
    lightgbm = evaluate_oos_predictions(
        frame.assign(prediction=frame["prediction"] * 0.9),
        model="lightgbm",
        transaction_cost_bps=5.0,
    )

    assert sequence.observations == lightgbm.observations == 20
    assert sequence.dates == lightgbm.dates == 4
    assert sequence.calibration_slope == pytest.approx(1.25)
    assert sequence.net_mean_daily_return < sequence.gross_mean_daily_return
    assert sequence.transaction_cost_bps == lightgbm.transaction_cost_bps


@pytest.mark.parametrize(
    ("kwargs", "message"),
    (
        ({"seq_len": 1}, "seq_len"),
        ({"hidden_size": 7, "n_heads": 2}, "n_heads"),
        ({"max_windows": 0}, "max_windows"),
        ({"max_tensor_bytes": 1}, "max_tensor_bytes"),
        ({"device": "quantum"}, "device"),
        ({"learning_rate": float("inf")}, "learning_rate"),
    ),
)
def test_invalid_or_unbounded_configuration_fails_before_allocation(
    kwargs: dict[str, object],
    message: str,
) -> None:
    with pytest.raises((TypeError, ValueError), match=message):
        ControlledSequenceModel("cnn", **kwargs)


def test_nonfinite_input_and_parameter_budget_fail_closed() -> None:
    features, target = _panel()
    infinite = features.copy()
    infinite.iloc[0, 0] = np.inf
    with pytest.raises(ValueError, match="infinite"):
        build_causal_windows(infinite, target, seq_len=5)
    with pytest.raises(ModelError, match="max_parameters"):
        ControlledSequenceModel("transformer", **FAST_CONFIG, max_parameters=10).fit(
            features,
            target,
        )


def test_tensor_allocation_budget_fails_before_window_allocation() -> None:
    features, target = _panel(symbols=4, days=28)
    wide = pd.DataFrame(
        np.zeros((len(features), 3_000)),
        index=features.index,
        columns=[f"f{index}" for index in range(3_000)],
    )

    with pytest.raises(ValueError, match="max_tensor_bytes"):
        build_causal_windows(
            wide,
            target,
            seq_len=20,
            min_history=1,
            max_tensor_bytes=1_048_576,
        )
