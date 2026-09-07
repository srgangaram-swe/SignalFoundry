"""Safe, versioned tabular artifacts for research-pipeline boundaries.

Pickle is executable serialization and therefore cannot be an interchange
contract.  This module stores DataFrames as versioned JSON Table Schema
documents, validates the envelope before deserialization, and publishes with
an atomic same-filesystem replace.
"""

from __future__ import annotations

import json
import os
import tempfile
from datetime import date, datetime
from io import StringIO
from pathlib import Path
from typing import Any

import pandas as pd
from pandas.api.types import is_object_dtype

TABLE_ARTIFACT_VERSION = "1.0.0"
TABLE_ARTIFACT_FORMAT = "pandas-table-json"
DEFAULT_MAX_ARTIFACT_BYTES = 512 * 1024 * 1024
_ROOT_FIELDS = {"artifact_schema_version", "format", "table"}
_TABLE_FIELDS = {"schema", "data"}


class ArtifactValidationError(ValueError):
    """Raised when a tabular artifact violates its schema or safety boundary."""


def _validate_path(path: Path) -> None:
    if path.suffixes[-2:] != [".table", ".json"]:
        raise ArtifactValidationError("tabular artifact paths must end in .table.json")
    if path.is_symlink():
        raise ArtifactValidationError("tabular artifact path must not be a symbolic link")


def _validate_document(document: Any) -> dict[str, Any]:
    if not isinstance(document, dict) or set(document) != _ROOT_FIELDS:
        raise ArtifactValidationError("tabular artifact envelope fields mismatch")
    if document["artifact_schema_version"] != TABLE_ARTIFACT_VERSION:
        raise ArtifactValidationError("unsupported tabular artifact schema version")
    if document["format"] != TABLE_ARTIFACT_FORMAT:
        raise ArtifactValidationError("unsupported tabular artifact format")
    table = document["table"]
    if not isinstance(table, dict) or set(table) != _TABLE_FIELDS:
        raise ArtifactValidationError("JSON Table Schema payload fields mismatch")
    schema = table["schema"]
    rows = table["data"]
    if not isinstance(schema, dict) or not isinstance(rows, list):
        raise ArtifactValidationError("JSON Table Schema must contain a schema and row list")
    fields = schema.get("fields")
    if not isinstance(fields, list) or not fields:
        raise ArtifactValidationError("JSON Table Schema fields must be a non-empty list")
    names = [field.get("name") for field in fields if isinstance(field, dict)]
    if len(names) != len(fields) or any(not isinstance(name, str) or not name for name in names):
        raise ArtifactValidationError("JSON Table Schema field names are invalid")
    if len(names) != len(set(names)):
        raise ArtifactValidationError("JSON Table Schema field names must be unique")
    expected = set(names)
    for row in rows:
        if not isinstance(row, dict) or set(row) != expected:
            raise ArtifactValidationError("tabular artifact row fields do not match its schema")
    return document


def _validate_frame(frame: pd.DataFrame) -> None:
    if not isinstance(frame, pd.DataFrame):
        raise TypeError("frame must be a pandas DataFrame")
    if not all(isinstance(column, str) and column for column in frame.columns):
        raise ArtifactValidationError("tabular artifact columns must be non-empty strings")
    if frame.columns.has_duplicates:
        raise ArtifactValidationError("tabular artifact columns must be unique")
    scalar_types = (str, bool, int, float, date, datetime)
    for column in frame.columns:
        series = frame[column]
        if not is_object_dtype(series.dtype):
            continue
        unsupported = [value for value in series.dropna() if not isinstance(value, scalar_types)]
        if unsupported:
            raise TypeError(f"object column {column!r} contains unsupported non-scalar values")


def write_frame_artifact(frame: pd.DataFrame, path: str | Path) -> Path:
    """Publish ``frame`` atomically as a versioned, non-executable artifact.

    The caller supplies a ``*.table.json`` destination. Object columns must
    contain JSON-compatible scalar values; arbitrary Python objects fail during
    serialization instead of being embedded as executable state.
    """

    destination = Path(path)
    _validate_path(destination)
    _validate_frame(frame)
    destination.parent.mkdir(parents=True, exist_ok=True)
    table = json.loads(frame.to_json(orient="table", date_format="iso", index=False))
    document = _validate_document(
        {
            "artifact_schema_version": TABLE_ARTIFACT_VERSION,
            "format": TABLE_ARTIFACT_FORMAT,
            "table": table,
        }
    )
    encoded = (
        json.dumps(document, sort_keys=True, separators=(",", ":"), ensure_ascii=True) + "\n"
    ).encode("utf-8")

    temporary_name: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            dir=destination.parent,
            prefix=f".{destination.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary_name = handle.name
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, destination)
    except BaseException:
        if temporary_name is not None:
            Path(temporary_name).unlink(missing_ok=True)
        raise
    return destination


def read_frame_artifact(
    path: str | Path, *, max_bytes: int = DEFAULT_MAX_ARTIFACT_BYTES
) -> pd.DataFrame:
    """Validate and load a versioned tabular artifact without code execution."""

    source = Path(path)
    _validate_path(source)
    if max_bytes <= 0:
        raise ValueError("max_bytes must be positive")
    try:
        size = source.stat().st_size
    except OSError as exc:
        raise ArtifactValidationError(f"could not stat tabular artifact {source}") from exc
    if size > max_bytes:
        raise ArtifactValidationError(
            f"tabular artifact is {size} bytes, above the {max_bytes}-byte limit"
        )
    try:
        document = _validate_document(json.loads(source.read_text(encoding="utf-8")))
        table_text = json.dumps(document["table"], separators=(",", ":"), ensure_ascii=True)
        frame = pd.read_json(StringIO(table_text), orient="table")
    except (OSError, UnicodeError, json.JSONDecodeError, ValueError, TypeError) as exc:
        if isinstance(exc, ArtifactValidationError):
            raise
        raise ArtifactValidationError(f"invalid tabular artifact {source}") from exc
    return frame
