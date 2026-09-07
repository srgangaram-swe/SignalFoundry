"""Contract and anti-corruption tests for Signal Foundry bundles."""

from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

import pandas as pd
import pytest

from alphaforge.data import (
    SignalFoundryDataError,
    load_prices,
    load_signal_foundry_dataset,
)

FIXTURE_ROOT = Path(__file__).parent / "fixtures/signal_foundry_v1"
UNIVERSE_COLUMNS = [
    "membership_id",
    "universe_id",
    "instrument_id",
    "ticker",
    "effective_at",
    "available_at",
    "observed_at",
    "provider_updated_at",
    "is_member",
    "reason",
    "source",
    "source_table",
]
CORPORATE_ACTION_COLUMNS = [
    "action_id",
    "instrument_id",
    "ticker",
    "action_type",
    "effective_at",
    "available_at",
    "observed_at",
    "provider_updated_at",
    "cash_amount",
    "split_ratio",
    "currency",
    "old_ticker",
    "new_ticker",
    "adjustment_state",
    "source",
    "source_table",
]
SEMANTIC_FIELDS = (
    "contract",
    "schema_version",
    "source_snapshot_hash",
    "source_manifest_sha256",
    "producer_git_sha",
    "files",
    "columns",
    "rows",
    "date_min",
    "date_max",
    "tickers",
    "temporal_contract",
    "point_in_time_limits",
    "license",
    "source_provenance",
)
V1_1_SEMANTIC_FIELDS = (
    *SEMANTIC_FIELDS,
    "universe_files",
    "universe_columns",
    "universe_rows",
    "corporate_action_files",
    "corporate_action_columns",
    "corporate_action_rows",
)


def _fixture_bundle() -> Path:
    pointer = json.loads((FIXTURE_ROOT / "current.json").read_text(encoding="utf-8"))
    return FIXTURE_ROOT / pointer["bundle_id"]


def _copied_bundle(tmp_path: Path) -> Path:
    tmp_path.mkdir(parents=True, exist_ok=True)
    destination = tmp_path / _fixture_bundle().name
    shutil.copytree(_fixture_bundle(), destination)
    return destination


def _write_family(bundle: Path, family: str, frame: pd.DataFrame) -> dict[str, object]:
    path = bundle / family / "part-00000.parquet"
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(path, index=False)
    return {
        "path": path.relative_to(bundle).as_posix(),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "rows": len(frame),
    }


def _rewrite_identity(bundle: Path) -> Path:
    manifest_path = bundle / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    fields = V1_1_SEMANTIC_FIELDS if manifest["schema_version"] == "1.1.0" else SEMANTIC_FIELDS
    semantic = {key: manifest[key] for key in fields}
    manifest["bundle_id"] = hashlib.sha256(
        json.dumps(semantic, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    destination = bundle.with_name(manifest["bundle_id"])
    bundle.rename(destination)
    return destination


def _schema_1_1_bundle(tmp_path: Path) -> Path:
    bundle = _copied_bundle(tmp_path)
    universe = pd.DataFrame(
        [
            {
                "membership_id": "atlas-aaa",
                "universe_id": "ATLAS",
                "instrument_id": "AAA",
                "ticker": "AAA",
                "effective_at": "2023-12-01T00:00:00Z",
                "available_at": "2023-12-01T01:00:00Z",
                "observed_at": "2026-07-23T00:00:00Z",
                "provider_updated_at": "2023-12-01T00:30:00Z",
                "is_member": True,
                "reason": "synthetic inclusion",
                "source": "synthetic_provider_fixture",
                "source_table": "TEST/UNIVERSE",
            },
            {
                "membership_id": "atlas-aaa",
                "universe_id": "ATLAS",
                "instrument_id": "AAA",
                "ticker": "AAA",
                "effective_at": "2024-01-04T00:00:00Z",
                "available_at": "2024-01-04T01:00:00Z",
                "observed_at": "2026-07-23T00:00:00Z",
                "provider_updated_at": "2024-01-04T00:30:00Z",
                "is_member": False,
                "reason": "synthetic delisting",
                "source": "synthetic_provider_fixture",
                "source_table": "TEST/UNIVERSE",
            },
            {
                "membership_id": "atlas-spy",
                "universe_id": "ATLAS",
                "instrument_id": "SPY",
                "ticker": "SPY",
                "effective_at": "2023-12-01T00:00:00Z",
                "available_at": "2023-12-01T01:00:00Z",
                "observed_at": "2026-07-23T00:00:00Z",
                "provider_updated_at": "2023-12-01T00:30:00Z",
                "is_member": True,
                "reason": "synthetic inclusion",
                "source": "synthetic_provider_fixture",
                "source_table": "TEST/UNIVERSE",
            },
        ],
        columns=UNIVERSE_COLUMNS,
    )
    corporate_actions = pd.DataFrame(
        [
            {
                "action_id": "aaa-dividend",
                "instrument_id": "AAA",
                "ticker": "AAA",
                "action_type": "cash_dividend",
                "effective_at": "2024-01-03T14:30:00Z",
                "available_at": "2024-01-02T22:00:00Z",
                "observed_at": "2026-07-23T00:00:00Z",
                "provider_updated_at": "2024-01-02T22:30:00Z",
                "cash_amount": 0.25,
                "split_ratio": None,
                "currency": "USD",
                "old_ticker": "",
                "new_ticker": "",
                "adjustment_state": "synthetic_fixture",
                "source": "synthetic_provider_fixture",
                "source_table": "TEST/ACTIONS",
            },
            {
                "action_id": "aaa-dividend",
                "instrument_id": "AAA",
                "ticker": "AAA",
                "action_type": "cash_dividend",
                "effective_at": "2024-01-03T14:30:00Z",
                "available_at": "2024-01-04T02:00:00Z",
                "observed_at": "2026-07-23T00:00:00Z",
                "provider_updated_at": "2024-01-04T02:30:00Z",
                "cash_amount": 0.30,
                "split_ratio": None,
                "currency": "USD",
                "old_ticker": "",
                "new_ticker": "",
                "adjustment_state": "synthetic_fixture",
                "source": "synthetic_provider_fixture",
                "source_table": "TEST/ACTIONS",
            },
            {
                "action_id": "spy-split",
                "instrument_id": "SPY",
                "ticker": "SPY",
                "action_type": "split",
                "effective_at": "2024-01-05T14:30:00Z",
                "available_at": "2024-01-01T12:00:00Z",
                "observed_at": "2026-07-23T00:00:00Z",
                "provider_updated_at": "2024-01-01T12:30:00Z",
                "cash_amount": None,
                "split_ratio": 2.0,
                "currency": "USD",
                "old_ticker": "",
                "new_ticker": "",
                "adjustment_state": "synthetic_fixture",
                "source": "synthetic_provider_fixture",
                "source_table": "TEST/ACTIONS",
            },
        ],
        columns=CORPORATE_ACTION_COLUMNS,
    )
    manifest_path = bundle / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest.update(
        {
            "schema_version": "1.1.0",
            "universe_files": [_write_family(bundle, "universe", universe)],
            "universe_columns": UNIVERSE_COLUMNS,
            "universe_rows": len(universe),
            "corporate_action_files": [
                _write_family(bundle, "corporate_actions", corporate_actions)
            ],
            "corporate_action_columns": CORPORATE_ACTION_COLUMNS,
            "corporate_action_rows": len(corporate_actions),
        }
    )
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    return _rewrite_identity(bundle)


def _rewrite_family(bundle: Path, family: str, frame: pd.DataFrame) -> Path:
    manifest_path = bundle / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    entry = _write_family(bundle, family, frame)
    key = "corporate_action_files" if family == "corporate_actions" else "universe_files"
    rows_key = "corporate_action_rows" if family == "corporate_actions" else "universe_rows"
    manifest[key] = [entry]
    manifest[rows_key] = len(frame)
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    return _rewrite_identity(bundle)


def test_committed_producer_fixture_validates_and_maps_adjusted_bars() -> None:
    dataset = load_signal_foundry_dataset(_fixture_bundle())

    assert dataset.bundle_id == _fixture_bundle().name
    assert dataset.manifest["schema_version"] == "1.0.0"
    assert dataset.manifest["license"]["observations_redistributable"] is True
    assert set(dataset.panel["symbol"]) == {"AAA", "SPY"}
    source_first = dataset.source_panel.iloc[0]
    panel_first = dataset.panel.loc[
        (dataset.panel["date"] == source_first["date"])
        & (dataset.panel["symbol"] == source_first["ticker"])
    ].iloc[0]
    adjustment_factor = source_first["adj_close"] / source_first["close"]
    assert panel_first["close"] == pytest.approx(source_first["adj_close"])
    assert panel_first["open"] == pytest.approx(source_first["open"] * adjustment_factor)
    assert dataset.decision_panel is not None
    assert dataset.decision_panel["date"].min() > dataset.panel["date"].min()
    assert len(dataset.decision_panel) < len(dataset.panel)
    assert dataset.universe_records.empty
    assert dataset.corporate_actions.empty
    assert dataset.point_in_time_diagnostics is not None
    assert dataset.point_in_time_diagnostics.survivorship_risk
    assert dataset.point_in_time_diagnostics.corporate_action_risk


def test_schema_1_1_reconstructs_membership_and_visible_action_revisions(
    tmp_path: Path,
) -> None:
    dataset = load_signal_foundry_dataset(_schema_1_1_bundle(tmp_path))
    legacy = load_signal_foundry_dataset(_fixture_bundle())

    assert dataset.point_in_time_diagnostics is not None
    assert not dataset.point_in_time_diagnostics.survivorship_risk
    assert not dataset.point_in_time_diagnostics.corporate_action_risk
    assert dataset.point_in_time_diagnostics.has_universe_records
    assert dataset.point_in_time_diagnostics.has_corporate_action_records
    pd.testing.assert_frame_equal(dataset.panel, legacy.panel)

    before_exit = dataset.universe_membership_as_of("ATLAS", "2024-01-03T23:59:59Z")
    after_exit = dataset.universe_membership_as_of("ATLAS", "2024-01-04T02:00:00Z")
    assert dict(zip(before_exit["ticker"], before_exit["is_member"], strict=True)) == {
        "AAA": True,
        "SPY": True,
    }
    assert dict(zip(after_exit["ticker"], after_exit["is_member"], strict=True)) == {
        "AAA": False,
        "SPY": True,
    }
    assert dataset.active_universe_as_of("ATLAS", "2024-01-04T02:00:00Z")["ticker"].tolist() == [
        "SPY"
    ]

    early_actions = dataset.corporate_actions_as_of("2024-01-03T15:00:00Z")
    later_actions = dataset.corporate_actions_as_of("2024-01-05T15:00:00Z")
    assert early_actions["action_id"].tolist() == ["aaa-dividend"]
    assert early_actions.iloc[0]["cash_amount"] == pytest.approx(0.25)
    assert set(later_actions["action_id"]) == {"aaa-dividend", "spy-split"}
    revised = later_actions.loc[later_actions["action_id"].eq("aaa-dividend")].iloc[0]
    assert revised["cash_amount"] == pytest.approx(0.30)


def test_dataset_as_of_bounds_every_record_family(tmp_path: Path) -> None:
    dataset = load_signal_foundry_dataset(
        _schema_1_1_bundle(tmp_path),
        as_of="2024-01-03T15:00:00Z",
    )

    assert dataset.as_of == pd.Timestamp("2024-01-03T15:00:00Z")
    assert dataset.source_panel["effective_at"].le(dataset.as_of).all()
    assert dataset.source_panel["available_at"].le(dataset.as_of).all()
    assert dataset.corporate_actions["action_id"].tolist() == ["aaa-dividend"]
    assert dataset.corporate_actions.iloc[0]["cash_amount"] == pytest.approx(0.25)
    with pytest.raises(SignalFoundryDataError, match="must equal the dataset as-of"):
        dataset.corporate_actions_as_of("2024-01-05T00:00:00Z")


def test_historical_universe_requires_complete_schema_1_1_evidence(
    tmp_path: Path,
) -> None:
    legacy = load_signal_foundry_dataset(_fixture_bundle())
    with pytest.raises(SignalFoundryDataError, match="requires schema 1.1"):
        legacy.universe_membership_as_of("ATLAS", "2024-01-01T00:00:00Z")

    bundle = _schema_1_1_bundle(tmp_path)
    manifest_path = bundle / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["point_in_time_limits"]["universe_membership_point_in_time"] = False
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    bundle = _rewrite_identity(bundle)
    incomplete = load_signal_foundry_dataset(bundle)
    with pytest.raises(SignalFoundryDataError, match="does not attest"):
        incomplete.active_universe_as_of("ATLAS", "2024-01-01T00:00:00Z")


def test_decision_timestamps_must_be_explicit_and_supported(tmp_path: Path) -> None:
    dataset = load_signal_foundry_dataset(_schema_1_1_bundle(tmp_path))

    with pytest.raises(SignalFoundryDataError, match="explicit timezone"):
        dataset.corporate_actions_as_of("2024-01-03")
    with pytest.raises(SignalFoundryDataError, match="non-empty"):
        dataset.active_universe_as_of("", "2024-01-03T00:00:00Z")
    with pytest.raises(SignalFoundryDataError, match="no visible membership"):
        dataset.active_universe_as_of("UNKNOWN", "2024-01-03T00:00:00Z")


def test_as_of_rule_excludes_future_available_observations() -> None:
    early = load_signal_foundry_dataset(
        _fixture_bundle(),
        as_of="2024-01-01T06:00:00Z",
    )
    later = load_signal_foundry_dataset(
        _fixture_bundle(),
        as_of="2024-01-04T06:00:00Z",
    )

    assert early.source_panel["available_at"].max() <= pd.Timestamp("2024-01-01T06:00:00Z")
    assert later.source_panel["available_at"].max() <= pd.Timestamp("2024-01-04T06:00:00Z")
    assert len(early.panel) < len(later.panel)


def test_future_rows_cannot_change_earlier_as_of_view(tmp_path: Path) -> None:
    original = load_signal_foundry_dataset(
        _fixture_bundle(),
        as_of="2024-01-03T04:59:59Z",
    ).panel
    copied = _copied_bundle(tmp_path)
    future_partition = copied / "prices/year=2024/part-00000.parquet"
    future_partition.write_bytes(b"adversarial future mutation")

    with pytest.raises(SignalFoundryDataError, match="hash mismatch"):
        load_signal_foundry_dataset(copied, as_of="2024-01-03T04:59:59Z")
    assert not original.empty


def test_corrupt_partition_and_path_traversal_fail_closed(tmp_path: Path) -> None:
    corrupt = _copied_bundle(tmp_path / "corrupt")
    next(corrupt.glob("prices/year=*/part-00000.parquet")).write_bytes(b"corrupt")
    with pytest.raises(SignalFoundryDataError, match="hash mismatch"):
        load_signal_foundry_dataset(corrupt)

    traversal = _copied_bundle(tmp_path / "traversal")
    manifest_path = traversal / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["files"][0]["path"] = "../licensed-data.parquet"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(SignalFoundryDataError, match="semantic identity mismatch|unsafe"):
        load_signal_foundry_dataset(traversal)


def test_auxiliary_hash_row_path_and_inventory_faults_fail_closed(tmp_path: Path) -> None:
    corrupt = _schema_1_1_bundle(tmp_path / "corrupt")
    (corrupt / "universe/part-00000.parquet").write_bytes(b"corrupt")
    with pytest.raises(SignalFoundryDataError, match="universe data hash mismatch"):
        load_signal_foundry_dataset(corrupt)

    wrong_rows = _schema_1_1_bundle(tmp_path / "rows")
    manifest_path = wrong_rows / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["corporate_action_files"][0]["rows"] += 1
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    wrong_rows = _rewrite_identity(wrong_rows)
    with pytest.raises(SignalFoundryDataError, match="row-count mismatch"):
        load_signal_foundry_dataset(wrong_rows)

    traversal = _schema_1_1_bundle(tmp_path / "traversal")
    manifest_path = traversal / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["universe_files"][0]["path"] = "../universe.parquet"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    traversal = _rewrite_identity(traversal)
    with pytest.raises(SignalFoundryDataError, match="unsafe bundle path"):
        load_signal_foundry_dataset(traversal)

    undeclared = _schema_1_1_bundle(tmp_path / "undeclared")
    shutil.copy2(
        undeclared / "universe/part-00000.parquet",
        undeclared / "undeclared.parquet",
    )
    with pytest.raises(SignalFoundryDataError, match="undeclared parquet"):
        load_signal_foundry_dataset(undeclared)


@pytest.mark.parametrize(
    ("case", "match"),
    [
        ("naive_timestamp", "explicit timezone-aware"),
        ("temporal_inversion", "observed_at precedes available_at"),
        ("duplicate_revision", "duplicate .* revision"),
        ("unsupported_action", "unsupported action types"),
        ("invalid_split", "split actions require split_ratio"),
    ],
)
def test_unsafe_auxiliary_semantics_fail_closed(
    tmp_path: Path,
    case: str,
    match: str,
) -> None:
    bundle = _schema_1_1_bundle(tmp_path)
    family = "universe" if case == "duplicate_revision" else "corporate_actions"
    path = bundle / family / "part-00000.parquet"
    frame = pd.read_parquet(path)
    if case == "naive_timestamp":
        frame["available_at"] = pd.Timestamp("2024-01-02")
    elif case == "temporal_inversion":
        frame["observed_at"] = pd.Timestamp("2020-01-01T00:00:00Z")
    elif case == "duplicate_revision":
        frame = pd.concat([frame, frame.iloc[[0]]], ignore_index=True)
    elif case == "unsupported_action":
        frame.loc[0, "action_type"] = "unknown"
    elif case == "invalid_split":
        frame.loc[frame["action_type"].eq("split"), "split_ratio"] = None
    else:  # pragma: no cover - parametrization is intentionally exhaustive.
        raise AssertionError(f"unknown case: {case}")
    bundle = _rewrite_family(bundle, family, frame)

    with pytest.raises(SignalFoundryDataError, match=match):
        load_signal_foundry_dataset(bundle)


@pytest.mark.parametrize(
    ("field", "value", "match"),
    [
        ("schema_version", "2.0.0", "unsupported schema"),
        ("columns", ["date"], "unsupported column"),
        ("point_in_time_limits", {}, "point-in-time"),
        (
            "license",
            {
                "observations_redistributable": True,
                "bundle_must_remain_local": True,
                "public_evidence_must_be_aggregate_or_synthetic": False,
            },
            "internally inconsistent",
        ),
    ],
)
def test_unsupported_or_ambiguous_manifest_policy_fails_closed(
    tmp_path: Path,
    field: str,
    value: object,
    match: str,
) -> None:
    bundle = _copied_bundle(tmp_path)
    manifest_path = bundle / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest[field] = value
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(SignalFoundryDataError, match=match):
        load_signal_foundry_dataset(bundle)


def test_generic_loader_cannot_collapse_market_and_decision_panels() -> None:
    with pytest.raises(ValueError, match="governed dual-panel"):
        load_prices(
            {
                "source": "signal_foundry",
                "bundle_dir": str(_fixture_bundle()),
                "benchmark": "SPY",
            }
        )


def test_signal_foundry_generic_loader_always_fails_closed() -> None:
    with pytest.raises(ValueError, match="governed dual-panel"):
        load_prices({"source": "signal_foundry", "benchmark": "SPY"})
    with pytest.raises(ValueError, match="governed dual-panel"):
        load_prices(
            {
                "source": "signal_foundry",
                "bundle_dir": str(_fixture_bundle()),
                "benchmark": "MISSING",
            }
        )


@pytest.mark.parametrize(
    ("column", "value", "match"),
    [
        ("exchange_calendar", "ambiguous", "unsupported exchange"),
        ("currency", "XYZ", "unsupported currencies"),
        ("effective_at", pd.Timestamp("2024-01-01"), "timezone-aware"),
    ],
)
def test_ambiguous_market_semantics_fail_closed(
    tmp_path: Path,
    column: str,
    value: object,
    match: str,
) -> None:
    bundle = _copied_bundle(tmp_path)
    partition_path = next(bundle.glob("prices/year=*/part-00000.parquet"))
    partition = pd.read_parquet(partition_path)
    partition[column] = value
    partition.to_parquet(partition_path, index=False)

    with pytest.raises(SignalFoundryDataError, match="hash mismatch"):
        load_signal_foundry_dataset(bundle)

    # Even a producer that recomputed only the file hash cannot bypass the
    # independent semantic checks; full bundle-identity rebuilding is tested
    # by the producer repository.
    manifest_path = bundle / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    entry = next(
        item
        for item in manifest["files"]
        if item["path"] == str(partition_path.relative_to(bundle))
    )
    entry["sha256"] = hashlib.sha256(partition_path.read_bytes()).hexdigest()
    semantic = {key: manifest[key] for key in SEMANTIC_FIELDS}
    manifest["bundle_id"] = hashlib.sha256(
        json.dumps(semantic, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    renamed = bundle.with_name(manifest["bundle_id"])
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    bundle.rename(renamed)

    with pytest.raises(SignalFoundryDataError, match=match):
        load_signal_foundry_dataset(renamed)
