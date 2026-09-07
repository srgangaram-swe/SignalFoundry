"""Pure diagnostic projection and hostile on-disk bundle shape boundaries."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from signal_foundry.boundary import FoundryError
from signal_foundry.worker_alpha import scalar, table
from signal_foundry.worker_data import bundle_path, discover


@pytest.mark.parametrize(
    "value,expected",
    [
        (None, None),
        (pd.NA, None),
        (pd.NaT, None),
        (pd.Timestamp("2024-01-02"), "2024-01-02"),
        (np.bool_(True), True),
        (np.int64(4), 4),
        (np.float64(1.5), 1.5),
        (np.nan, None),
        (np.inf, None),
        ("x", "x"),
    ],
)
def test_scalar_projection(value, expected) -> None:
    assert scalar(value) == expected


def test_projection_rejects_objects_and_marks_truncation() -> None:
    with pytest.raises(FoundryError, match="unsupported_evidence"):
        scalar(object())
    source = pd.DataFrame({"value": range(2050), "secret_field": "MARKER_SECRET"})
    result = table("test", source, {"value": "test units"}, "Test projection.")
    assert len(result.rows) == 2048 and result.total_rows == 2050
    assert "MARKER_SECRET" not in result.canonical().decode()
    empty = table("test", pd.DataFrame(), {"value": "test units"}, "No observations.")
    assert empty.rows == () and empty.columns[0].name == "value"
    with pytest.raises(FoundryError, match="evidence_schema"):
        table("test", source, {"missing": "test units"}, "Incompatible columns.")


@pytest.mark.parametrize(
    "manifest,code",
    [
        ({}, "invalid_dataset"),
        ({"rows": True}, "invalid_dataset"),
        ({"rows": 0}, "dataset_limit"),
        ({"rows": 40001}, "dataset_limit"),
        ({"rows": 4, "tickers": ["A"]}, "dataset_limit"),
        ({"rows": 4, "tickers": "AB"}, "dataset_limit"),
    ],
)
def test_bundle_manifest_bounds(tmp_path: Path, manifest: dict, code: str) -> None:
    bundle = tmp_path / ("a" * 64)
    bundle.mkdir()
    (bundle / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(FoundryError, match=code):
        bundle_path(tmp_path, bundle.name)


def test_bundle_traversal_links_and_disk_budget(tmp_path: Path) -> None:
    with pytest.raises(FoundryError, match="invalid_dataset_id"):
        bundle_path(tmp_path, "../secret")
    bundle = tmp_path / ("a" * 64)
    bundle.mkdir()
    (bundle / "manifest.json").write_text('{"rows":4,"tickers":["A","B"]}')
    assert bundle_path(tmp_path, bundle.name) == bundle
    link = bundle / "symlink"
    link.symlink_to(tmp_path)
    with pytest.raises(FoundryError, match="unsafe_dataset"):
        bundle_path(tmp_path, bundle.name)
    link.unlink()
    with (bundle / "large").open("wb") as stream:
        stream.truncate((16 << 20) + 1)
    with pytest.raises(FoundryError, match="dataset_limit"):
        bundle_path(tmp_path, bundle.name)


def test_discovery_bounds_before_any_validation(tmp_path: Path) -> None:
    assert discover(None) == ()
    for index in range(65):
        (tmp_path / f"unrecognized-{index}").mkdir()
    with pytest.raises(FoundryError, match="catalog_limit"):
        discover(tmp_path)
    other = tmp_path / "bounded"
    other.mkdir()
    for index in range(17):
        (other / f"{index:064x}").mkdir()
    with pytest.raises(FoundryError, match="catalog_limit"):
        discover(other)
