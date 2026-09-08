"""Fail-closed contracts for bounded evidence I/O (SF-S3-MR11).

The synthesis engine reads files it did not write — configs, receipts, and
evidence from a second repository. Every one of those is adversarial input under
the security contract, so the reader is the trust boundary and its refusals are
the thing worth testing. These are the negative paths: a symlink swapped in mid
read, a YAML alias billion-laughs expansion, a duplicate JSON key that would let
two different documents hash the same, a file that grows past its ceiling.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from alphaforge.research._bounded_io import (
    BoundedIOError,
    bounded_diagnostic,
    parse_strict_json,
    parse_strict_yaml,
    read_regular_file_snapshot,
)

LIMITS = {"maximum_depth": 32, "maximum_nodes": 10_000}


# ---------------------------------------------------------------------------
# read_regular_file_snapshot
# ---------------------------------------------------------------------------


def test_reads_a_regular_file(tmp_path: Path) -> None:
    target = tmp_path / "evidence.json"
    target.write_bytes(b'{"a": 1}')
    snapshot = read_regular_file_snapshot(target, max_bytes=1024)
    assert snapshot.data == b'{"a": 1}'


@pytest.mark.parametrize("max_bytes", [0, -1, True, 1.5, "8"])
def test_max_bytes_must_be_a_positive_int(tmp_path: Path, max_bytes: object) -> None:
    target = tmp_path / "evidence.json"
    target.write_bytes(b"{}")
    with pytest.raises(BoundedIOError, match="max_bytes"):
        read_regular_file_snapshot(target, max_bytes=max_bytes)  # type: ignore[arg-type]


def test_empty_file_is_refused(tmp_path: Path) -> None:
    target = tmp_path / "empty.json"
    target.write_bytes(b"")
    with pytest.raises(BoundedIOError, match=r"must be in \[1,"):
        read_regular_file_snapshot(target, max_bytes=16)


def test_file_above_the_ceiling_is_refused(tmp_path: Path) -> None:
    target = tmp_path / "big.json"
    target.write_bytes(b"x" * 64)
    with pytest.raises(BoundedIOError, match="bounded file"):
        read_regular_file_snapshot(target, max_bytes=8)


def test_directory_is_not_a_regular_file(tmp_path: Path) -> None:
    with pytest.raises(BoundedIOError, match="regular file|unable to read"):
        read_regular_file_snapshot(tmp_path, max_bytes=16)


def test_missing_file_fails_closed(tmp_path: Path) -> None:
    with pytest.raises(BoundedIOError, match="unable to read"):
        read_regular_file_snapshot(tmp_path / "absent.json", max_bytes=16)


def test_symlinked_file_is_refused(tmp_path: Path) -> None:
    target = tmp_path / "real.json"
    target.write_bytes(b"{}")
    link = tmp_path / "link.json"
    link.symlink_to(target)
    with pytest.raises(BoundedIOError, match="symlink|unable to read"):
        read_regular_file_snapshot(link, max_bytes=16)


def test_path_escaping_the_declared_root_is_refused(tmp_path: Path) -> None:
    root = tmp_path / "root"
    root.mkdir()
    outside = tmp_path / "outside.json"
    outside.write_bytes(b"{}")
    with pytest.raises(BoundedIOError, match="escapes its declared root"):
        read_regular_file_snapshot(outside, max_bytes=16, root=root)


def test_symlinked_root_is_refused(tmp_path: Path) -> None:
    real = tmp_path / "real"
    real.mkdir()
    (real / "f.json").write_bytes(b"{}")
    link = tmp_path / "link"
    link.symlink_to(real, target_is_directory=True)
    with pytest.raises(BoundedIOError, match="root must not be a symlink"):
        read_regular_file_snapshot(link / "f.json", max_bytes=16, root=link)


def test_symlinked_intermediate_component_is_refused(tmp_path: Path) -> None:
    root = tmp_path / "root"
    (root / "real").mkdir(parents=True)
    (root / "real" / "f.json").write_bytes(b"{}")
    os.symlink(root / "real", root / "hop", target_is_directory=True)
    with pytest.raises(BoundedIOError, match="traverses a symlink"):
        read_regular_file_snapshot(root / "hop" / "f.json", max_bytes=16, root=root)


# ---------------------------------------------------------------------------
# parse_strict_json
# ---------------------------------------------------------------------------


def test_parses_bounded_json() -> None:
    assert parse_strict_json(b'{"a": [1, 2]}', **LIMITS) == {"a": [1, 2]}


def test_duplicate_json_keys_are_refused() -> None:
    """Two documents that differ must never parse to the same object."""
    with pytest.raises(BoundedIOError, match="duplicate key"):
        parse_strict_json(b'{"a": 1, "a": 2}', **LIMITS)


@pytest.mark.parametrize("payload", [b'{"a": NaN}', b'{"a": Infinity}', b'{"a": -Infinity}'])
def test_non_finite_json_constants_are_refused(payload: bytes) -> None:
    """NaN compares false against every bound, so it must not enter at all."""
    with pytest.raises(BoundedIOError, match="unsupported constant"):
        parse_strict_json(payload, **LIMITS)


@pytest.mark.parametrize("payload", [b"{not json", b"\xff\xfe not utf-8", b""])
def test_malformed_json_fails_closed(payload: bytes) -> None:
    with pytest.raises(BoundedIOError, match="bounded UTF-8 JSON"):
        parse_strict_json(payload, **LIMITS)


def test_json_depth_ceiling_is_enforced() -> None:
    payload = (b"[" * 40) + (b"]" * 40)
    with pytest.raises(BoundedIOError, match="depth"):
        parse_strict_json(payload, maximum_depth=8, maximum_nodes=10_000)


def test_json_node_ceiling_is_enforced() -> None:
    payload = b"[" + b",".join(b"1" for _ in range(50)) + b"]"
    with pytest.raises(BoundedIOError, match="nodes"):
        parse_strict_json(payload, maximum_depth=32, maximum_nodes=10)


# ---------------------------------------------------------------------------
# parse_strict_yaml
# ---------------------------------------------------------------------------


def test_parses_bounded_yaml() -> None:
    assert parse_strict_yaml(b"a:\n  - 1\n  - 2\n", **LIMITS) == {"a": [1, 2]}


def test_yaml_aliases_are_refused() -> None:
    """Aliases permit quadratic expansion from a tiny document."""
    payload = b"a: &anchor [1, 2]\nb: *anchor\n"
    with pytest.raises(BoundedIOError, match="aliases are not permitted"):
        parse_strict_yaml(payload, **LIMITS)


def test_duplicate_yaml_keys_are_refused() -> None:
    with pytest.raises(BoundedIOError, match="duplicate key"):
        parse_strict_yaml(b"a: 1\na: 2\n", **LIMITS)


def test_non_string_yaml_keys_are_refused() -> None:
    with pytest.raises(BoundedIOError, match="keys must be strings"):
        parse_strict_yaml(b"1: value\n", **LIMITS)


@pytest.mark.parametrize("payload", [b"a: [unclosed\n", b"\xff\xfe"])
def test_malformed_yaml_fails_closed(payload: bytes) -> None:
    with pytest.raises(BoundedIOError, match="bounded UTF-8 YAML"):
        parse_strict_yaml(payload, **LIMITS)


def test_yaml_depth_ceiling_is_enforced() -> None:
    payload = ("[" * 40 + "]" * 40).encode()
    with pytest.raises(BoundedIOError, match="depth"):
        parse_strict_yaml(payload, maximum_depth=8, maximum_nodes=10_000)


def test_yaml_node_ceiling_is_enforced() -> None:
    payload = ("[" + ",".join("1" for _ in range(50)) + "]").encode()
    with pytest.raises(BoundedIOError, match="nodes"):
        parse_strict_yaml(payload, maximum_depth=32, maximum_nodes=10)


# ---------------------------------------------------------------------------
# bounded_diagnostic
# ---------------------------------------------------------------------------


def test_short_diagnostics_pass_through() -> None:
    assert bounded_diagnostic("prefix: ", "detail", maximum_chars=64) == "prefix: detail"


def test_long_diagnostics_are_truncated_within_their_budget() -> None:
    message = bounded_diagnostic("p: ", "x" * 500, maximum_chars=32)
    assert len(message) == 32
    assert message.endswith("...[truncated]")


def test_diagnostic_budget_has_a_floor() -> None:
    with pytest.raises(ValueError, match="at least 16"):
        bounded_diagnostic("p: ", "detail", maximum_chars=8)
