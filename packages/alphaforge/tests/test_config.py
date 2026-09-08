"""Strict configuration boundary and cross-field invariant tests."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from alphaforge.config import SCHEMAS, ConfigValidationError, load_config


def _write_yaml(path: Path, payload: object) -> Path:
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    return path


@pytest.mark.parametrize("kind", sorted(SCHEMAS))
def test_every_committed_configuration_has_a_strict_schema(kind: str) -> None:
    config = load_config(Path("configs") / f"{kind}.yaml", kind)
    assert config


def test_wiki_bootstrap_profile_has_a_strict_pre_registered_boundary() -> None:
    config = load_config(
        Path("configs/signal_foundry_wiki_bootstrap.yaml"),
        "signal_foundry_research",
    )

    assert config["research"]["holdout_start"] == "2017-01-03"
    assert config["research"]["benchmark_symbol"] == "AAPL"
    assert config["readiness"]["require_complete_point_in_time"] is True
    assert config["readiness"]["minimum_holdout_days"] == 252


@pytest.mark.parametrize(
    ("kind", "mutation", "message"),
    [
        ("data", lambda cfg: cfg.update({"api_key": "must-not-appear"}), "extra"),
        (
            "features",
            lambda cfg: cfg["macd"].update({"future_window": 1}),
            "future_window",
        ),
        (
            "labels",
            lambda cfg: cfg["labels"][0].update({"future_window": 1}),
            "future_window",
        ),
        ("models", lambda cfg: cfg["walk_forward"].update({"embargo_days": 0}), "embargo"),
        (
            "backtest",
            lambda cfg: cfg["execution"].update({"price_field": "close"}),
            "price_field",
        ),
        ("portfolio", lambda cfg: cfg.update({"cash_buffer": 1.5}), "cash_buffer"),
        ("risk", lambda cfg: cfg.update({"var_confidence": 0.95}), "var_confidence"),
        (
            "signal_foundry_research",
            lambda cfg: cfg["readiness"].update({"unknown_gate": True}),
            "unknown_gate",
        ),
        (
            "research_governance",
            lambda cfg: cfg["correction"].update({"failed_trial_p_value": 0.5}),
            "failed_trial_p_value",
        ),
        (
            "decision_policy",
            lambda cfg: cfg["policy"].update({"broker_token": "forbidden"}),
            "broker_token",
        ),
    ],
)
def test_invalid_or_unknown_settings_fail_before_execution(
    tmp_path: Path,
    kind: str,
    mutation: object,
    message: str,
) -> None:
    original = yaml.safe_load(Path(f"configs/{kind}.yaml").read_text(encoding="utf-8"))
    assert isinstance(original, dict)
    mutation(original)  # type: ignore[operator]
    path = _write_yaml(tmp_path / f"{kind}.yaml", original)

    with pytest.raises(ConfigValidationError, match=message):
        load_config(path, kind)


@pytest.mark.parametrize("payload", [None, [], "not-a-mapping"])
def test_configuration_root_must_be_a_mapping(tmp_path: Path, payload: object) -> None:
    path = _write_yaml(tmp_path / "bad.yaml", payload)

    with pytest.raises(ConfigValidationError, match="root must be a mapping"):
        load_config(path, "data")


def test_data_configuration_rejects_unsafe_paths_and_date_order(tmp_path: Path) -> None:
    config = yaml.safe_load(Path("configs/data.yaml").read_text(encoding="utf-8"))
    config["cache_dir"] = "../outside"
    path = _write_yaml(tmp_path / "unsafe.yaml", config)
    with pytest.raises(ConfigValidationError, match="safe relative path"):
        load_config(path, "data")

    config["cache_dir"] = "data/cache"
    config["end"] = config["start"]
    path = _write_yaml(tmp_path / "dates.yaml", config)
    with pytest.raises(ConfigValidationError, match="earlier than end"):
        load_config(path, "data")


def test_model_parameter_names_are_not_an_untyped_escape_hatch(tmp_path: Path) -> None:
    config = yaml.safe_load(Path("configs/models.yaml").read_text(encoding="utf-8"))
    config["models"][0]["params"] = {"magic_accuracy": 1.0}
    path = _write_yaml(tmp_path / "models.yaml", config)

    with pytest.raises(ConfigValidationError, match="unknown parameters"):
        load_config(path, "models")


def test_governed_model_parameter_surfaces_are_explicit(tmp_path: Path) -> None:
    config = yaml.safe_load(Path("configs/models.yaml").read_text(encoding="utf-8"))
    config["models"] = [
        {"name": "huber", "params": {"epsilon": 1.35, "max_iter": 100}},
        {
            "name": "extra_trees",
            "params": {"n_estimators": 20, "n_jobs": 1, "random_state": 42},
        },
        {
            "name": "lightgbm",
            "params": {"n_estimators": 20, "n_jobs": 1, "random_state": 42},
        },
        {
            "name": "xgboost",
            "params": {"n_estimators": 20, "min_child_weight": 1.0, "n_jobs": 1},
        },
        {
            "name": "catboost",
            "params": {"n_estimators": 20, "min_samples_leaf": 2, "n_jobs": 1},
        },
        {
            "name": "small_mlp",
            "params": {"hidden_layer_sizes": [8], "max_iter": 20, "random_state": 42},
        },
    ]
    path = _write_yaml(tmp_path / "models.yaml", config)

    loaded = load_config(path, "models")

    assert [spec["name"] for spec in loaded["models"]] == [
        "huber",
        "extra_trees",
        "lightgbm",
        "xgboost",
        "catboost",
        "small_mlp",
    ]


def test_label_configuration_rejects_invalid_semantics(tmp_path: Path) -> None:
    config = yaml.safe_load(Path("configs/labels.yaml").read_text(encoding="utf-8"))
    barrier = next(label for label in config["labels"] if label["kind"] == "triple_barrier")
    barrier["upper_barrier"] = 0.0
    path = _write_yaml(tmp_path / "labels.yaml", config)
    with pytest.raises(ConfigValidationError, match="upper_barrier"):
        load_config(path, "labels")

    config = yaml.safe_load(Path("configs/labels.yaml").read_text(encoding="utf-8"))
    config["protected_boundaries"] = ["2025-01-01", "2024-01-01"]
    path = _write_yaml(tmp_path / "labels.yaml", config)
    with pytest.raises(ConfigValidationError, match="strictly increasing"):
        load_config(path, "labels")


def test_strategy_specific_unused_settings_are_rejected(tmp_path: Path) -> None:
    config = yaml.safe_load(Path("configs/backtest.yaml").read_text(encoding="utf-8"))
    config["strategy_params"]["top_k"] = 5
    path = _write_yaml(tmp_path / "backtest.yaml", config)

    with pytest.raises(ConfigValidationError, match="unused by long_short"):
        load_config(path, "backtest")
