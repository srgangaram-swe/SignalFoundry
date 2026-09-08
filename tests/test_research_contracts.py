"""Wire shape, semantic, finite-JSON and path boundary invariants."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from hypothesis import given
from hypothesis import strategies as st
from pydantic import ValidationError

from signal_foundry.boundary import (
    FoundryError,
    decode,
    encode,
    private_directory,
    read_file,
)
from signal_foundry.contracts import (
    Column,
    EvidenceTable,
    ModelChoice,
    Parameter,
    ResearchEvidence,
    ResearchRequest,
)
from signal_foundry.policy import model_parameters
from tests.research_helpers import evidence


@pytest.mark.parametrize(
    "payload",
    [
        b"",
        b"x",
        b' {"a":1,"a":2}',
        b'{"x":NaN}',
        b'{"x":Infinity}',
        b'{"x":1e999}',
        b"\xff",
        b"[" * 40 + b"0" + b"]" * 40,
        b"x" * 16385,
    ],
)
def test_reject_ambiguous_or_unbounded_json(payload: bytes) -> None:
    with pytest.raises(FoundryError):
        decode(payload)


@given(
    st.dictionaries(
        st.text(max_size=10),
        st.one_of(
            st.none(),
            st.booleans(),
            st.integers(min_value=-(10**6), max_value=10**6),
            st.floats(allow_nan=False, allow_infinity=False),
            st.text(max_size=32),
        ),
        max_size=8,
    )
)
def test_finite_json_roundtrip(value: dict[str, Any]) -> None:
    assert decode(encode(value)) == value


@given(st.integers(min_value=0, max_value=2**32 - 1))
def test_request_hash_and_evidence_bind_configuration(seed: int) -> None:
    request = ResearchRequest(seed=seed)
    assert ResearchRequest.model_validate_json(request.canonical()) == request
    assert ResearchEvidence.model_validate_json(
        evidence(request).canonical()
    ) == evidence(request)
    changed = evidence(request).model_dump(mode="json")
    changed["request_hash"] = "0" * 64
    with pytest.raises(ValidationError):
        ResearchEvidence.model_validate_json(json.dumps(changed))


@pytest.mark.parametrize(
    "change",
    [
        {"unknown": True},
        {"seed": True},
        {"seed": -1},
        {"strategy": "../exec"},
        {"data": {"kind": "bundle"}},
        {"data": {"benchmark": "SPY"}},
        {"data": {"symbols": 31, "days": 2000}},
        {"folds": {"horizon": 10, "embargo_days": 1}},
        {"folds": {"train_days": 500}},
        {"risk": {"max_weight": 1, "max_gross": 0.5}},
        {"risk": {"max_net": 1, "max_gross": 0.5}},
        {"costs": {"execution_lag": 0}},
        {"baselines": ["ridge"]},
        {"baselines": ["zero_baseline", "zero_baseline"]},
        {
            "model": {
                "parameters": [
                    {"name": "alpha", "value": 1},
                    {"name": "alpha", "value": 2},
                ]
            }
        },
        {"model": {"parameters": [{"name": "alpha", "value": "../../artifact"}]}},
    ],
)
def test_invalid_research_fails_closed(change: dict[str, Any]) -> None:
    with pytest.raises(ValidationError):
        ResearchRequest.model_validate_json(json.dumps(change))


@pytest.mark.parametrize(
    "name,value,code",
    [
        ("path", 1, "unsupported_parameter"),
        ("alpha", True, "unsupported_parameter"),
        ("alpha", "abc", "unsupported_parameter"),
        ("alpha", -1, "parameter_limit"),
        ("epochs", 4.5, "parameter_type"),
        ("epochs", 51, "parameter_limit"),
    ],
)
def test_parameter_policy(name: str, value: Any, code: str) -> None:
    with pytest.raises(FoundryError, match=code):
        model_parameters(ModelChoice(parameters=(Parameter(name=name, value=value),)))


def test_valid_parameters_and_table_bounds() -> None:
    assert model_parameters(
        ModelChoice(parameters=(Parameter(name="alpha", value=2),))
    ) == {"alpha": 2}
    column = Column(name="value", unit="fraction")
    for columns, rows, total in [
        ((column, column), (), 0),
        ((column,), ((1, 2),), 1),
        ((column,), ((1,),), 0),
    ]:
        with pytest.raises(ValidationError):
            EvidenceTable(
                name="test",
                description="Test.",
                columns=columns,
                rows=rows,
                total_rows=total,
            )


def test_local_file_policy(tmp_path: Path) -> None:
    private = private_directory(tmp_path / "state", create=True)
    assert private.stat().st_mode & 0o077 == 0
    target = private / "input"
    target.write_bytes(b"123")
    assert read_file(target, 3) == b"123"
    with pytest.raises(FoundryError, match="unsafe_file"):
        read_file(target, 2)
    link = private / "link"
    link.symlink_to(target)
    with pytest.raises(FoundryError, match="resource_unavailable"):
        read_file(link)
    with pytest.raises(FoundryError, match="unsafe_directory"):
        private_directory(link)
    private.chmod(0o755)
    with pytest.raises(FoundryError, match="unsafe_directory"):
        private_directory(private, create=True)
    with pytest.raises(FoundryError, match="storage_unavailable"):
        private_directory(tmp_path / "missing")


@pytest.mark.parametrize("value", [float("nan"), float("inf"), object()])
def test_nonfinite_or_unserializable_evidence(value: Any) -> None:
    with pytest.raises(FoundryError, match="invalid_evidence"):
        encode(value)


def test_encoded_output_limit() -> None:
    with pytest.raises(FoundryError, match="evidence_limit"):
        encode("long", maximum=2)
