"""Versioned, non-executable tabular artifact contract tests."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from alphaforge.research import (
    ArtifactValidationError,
    read_frame_artifact,
    write_frame_artifact,
)


def _frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "date": pd.to_datetime(["2026-01-02", "2026-01-05"], utc=True),
            "symbol": ["AAPL", "MSFT"],
            "value": np.array([1.25, -2.5], dtype=np.float64),
            "window": np.array([1, 5], dtype=np.int64),
            "active": [True, False],
        }
    )


def test_tabular_artifact_is_versioned_deterministic_and_round_trips(tmp_path: Path) -> None:
    first = write_frame_artifact(_frame(), tmp_path / "first.table.json")
    second = write_frame_artifact(_frame(), tmp_path / "second.table.json")

    assert first.read_bytes() == second.read_bytes()
    document = json.loads(first.read_text(encoding="utf-8"))
    assert document["artifact_schema_version"] == "1.0.0"
    assert document["format"] == "pandas-table-json"
    expected = _frame()
    expected["date"] = expected["date"].astype("datetime64[ns, UTC]")
    pd.testing.assert_frame_equal(read_frame_artifact(first), expected)


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda doc: doc.update({"artifact_schema_version": "999.0.0"}), "schema version"),
        (lambda doc: doc.update({"unexpected": True}), "envelope fields"),
        (lambda doc: doc["table"]["data"][0].pop("symbol"), "row fields"),
        (lambda doc: doc["table"]["schema"]["fields"].append({"name": "symbol"}), "unique"),
    ],
)
def test_tampered_artifacts_fail_closed(tmp_path: Path, mutation: object, message: str) -> None:
    path = write_frame_artifact(_frame(), tmp_path / "frame.table.json")
    document = json.loads(path.read_text(encoding="utf-8"))
    mutation(document)  # type: ignore[operator]
    path.write_text(json.dumps(document), encoding="utf-8")

    with pytest.raises(ArtifactValidationError, match=message):
        read_frame_artifact(path)


def test_artifact_boundary_rejects_unsafe_names_symlinks_and_resource_overflow(
    tmp_path: Path,
) -> None:
    with pytest.raises(ArtifactValidationError, match=r"\.table\.json"):
        write_frame_artifact(_frame(), tmp_path / "frame.pkl")

    path = write_frame_artifact(_frame(), tmp_path / "frame.table.json")
    with pytest.raises(ArtifactValidationError, match="above"):
        read_frame_artifact(path, max_bytes=1)

    symlink = tmp_path / "link.table.json"
    symlink.symlink_to(path)
    with pytest.raises(ArtifactValidationError, match="symbolic link"):
        read_frame_artifact(symlink)


def test_failed_serialization_does_not_publish_partial_artifact(tmp_path: Path) -> None:
    path = tmp_path / "unsafe.table.json"
    frame = pd.DataFrame({"object": [object()]})

    with pytest.raises((TypeError, OverflowError)):
        write_frame_artifact(frame, path)
    assert not path.exists()
    assert not list(tmp_path.glob(".*.tmp"))
