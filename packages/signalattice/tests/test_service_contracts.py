"""Unit tests for immutable service-wide value and serialization invariants."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta, timezone, tzinfo

import pytest
from pydantic import ValidationError as PydanticValidationError

from quant_platform.service.contracts import (
    MAX_CANONICAL_CONTRACT_BYTES,
    ServiceContractError,
    StrictServiceContract,
    require_bounded_text,
    require_finite_number,
    require_url_identifier,
    require_utc_datetime,
)


class _SampleContract(StrictServiceContract):
    alpha: float
    identifier: str
    observed_at: datetime


class _ExplosiveTimezone(tzinfo):
    """Adversarial timezone whose hooks must not escape the validation boundary."""

    def utcoffset(self, _value: datetime | None) -> timedelta | None:
        raise RuntimeError("sensitive timezone implementation detail")

    def dst(self, _value: datetime | None) -> timedelta | None:
        return None


class _ExplosiveBool:
    """Wrong-type option proving text validation never invokes caller truthiness."""

    def __bool__(self) -> bool:
        raise AssertionError("attacker-controlled truthiness executed")


def test_contract_is_strict_frozen_and_forbids_unknown_fields() -> None:
    contract = _SampleContract(alpha=1.5, identifier="run-1", observed_at=datetime.now(UTC))

    with pytest.raises(PydanticValidationError):
        _SampleContract(alpha="1.5", identifier="run-1", observed_at=datetime.now(UTC))  # type: ignore[arg-type]
    with pytest.raises(PydanticValidationError):
        _SampleContract.model_validate(
            {
                "alpha": 1.5,
                "identifier": "run-1",
                "observed_at": datetime.now(UTC),
                "unknown": True,
            }
        )
    with pytest.raises(PydanticValidationError):
        contract.alpha = 2.0


def test_canonical_json_is_finite_sorted_utf8_and_byte_bounded() -> None:
    contract = _SampleContract(
        alpha=1.5,
        identifier="run-1",
        observed_at=datetime(2026, 8, 9, 12, 0, tzinfo=UTC),
    )

    assert contract.canonical_json_bytes() == (
        b'{"alpha":1.5,"identifier":"run-1",' b'"observed_at":"2026-08-09T12:00:00Z"}'
    )
    with pytest.raises(ServiceContractError, match="byte bound"):
        contract.canonical_json_bytes(maximum_bytes=1)
    with pytest.raises(ServiceContractError, match="outside"):
        contract.canonical_json_bytes(maximum_bytes=MAX_CANONICAL_CONTRACT_BYTES + 1)


@pytest.mark.parametrize("value", ["run-1", "RUN_2", "sha256~evidence", "a.b"])
def test_url_identifier_accepts_only_bounded_unreserved_segments(value: str) -> None:
    assert require_url_identifier(value, "run_id") == value


@pytest.mark.parametrize(
    "value",
    ["", "/absolute", "relative/path", "query?value", "with space", "évidence", "a" * 129],
)
def test_url_identifier_rejects_path_or_non_ascii_ambiguity(value: str) -> None:
    with pytest.raises(ValueError, match="URL-safe"):
        require_url_identifier(value, "run_id")


def test_text_bound_is_utf8_bytes_and_rejects_controls() -> None:
    assert require_bounded_text("é", "label", maximum_bytes=2) == "é"
    assert require_bounded_text("line\nnext", "body", maximum_bytes=32, multiline=True)
    with pytest.raises(ValueError, match="UTF-8 bytes"):
        require_bounded_text("é", "label", maximum_bytes=1)
    with pytest.raises(ValueError, match="control"):
        require_bounded_text("line\nnext", "label", maximum_bytes=32)
    with pytest.raises(ValueError, match="control"):
        require_bounded_text("safe\x00unsafe", "label", maximum_bytes=32)
    with pytest.raises(ValueError, match="bounds"):
        require_bounded_text(
            "safe",
            "label",
            maximum_bytes=32,
            multiline=_ExplosiveBool(),  # type: ignore[arg-type]
        )


def test_utc_normalization_rejects_naive_and_accepts_aware_offsets() -> None:
    source = datetime(2026, 8, 9, 6, 0, tzinfo=timezone(timedelta(hours=-6)))

    assert require_utc_datetime(source, "observed_at") == datetime(2026, 8, 9, 12, 0, tzinfo=UTC)
    assert require_utc_datetime(source, "observed_at").tzinfo is UTC
    with pytest.raises(ValueError, match="timezone-aware"):
        require_utc_datetime(datetime(2026, 8, 9, 12, 0), "observed_at")
    with pytest.raises(ValueError, match="cannot be normalized") as failure:
        require_utc_datetime(
            datetime(2026, 8, 9, 12, 0, tzinfo=_ExplosiveTimezone()),
            "observed_at",
        )
    assert "sensitive timezone" not in str(failure.value)


@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
def test_finite_number_rejects_nonfinite_values(value: float) -> None:
    with pytest.raises(ValueError, match="finite"):
        require_finite_number(value, "metric")
    with pytest.raises(ValueError, match="exact"):
        require_finite_number(1, "metric")  # type: ignore[arg-type]
