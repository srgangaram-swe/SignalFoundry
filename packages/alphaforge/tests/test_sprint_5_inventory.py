"""Tests for the Git-derived Sprint 5 delivery inventory."""

from __future__ import annotations

import json
import subprocess
from dataclasses import replace
from pathlib import Path

import pytest
import yaml

import alphaforge.readiness.sprint_5_inventory as inventory_module
from alphaforge.readiness.sprint_5_inventory import (
    MAX_CHANGED_PATHS,
    MAX_GIT_ARGUMENTS,
    MAX_GIT_OUTPUT_BYTES,
    MAX_GIT_STDIN_BYTES,
    MAX_TRACKED_BLOB_BYTES,
    SOURCE_HEAD,
    SPRINT_5_SLICES,
    DeliveryInventory,
    DeliveryPath,
    DeliverySlice,
    DeliverySliceSpec,
    Sprint5InventoryError,
    _blob_sizes,
    _classify_path,
    _DiffEntry,
    _inspect_commit,
    _parse_raw_diff,
    _run_git,
    _single_ascii_line,
    _validate_repository_path,
    build_sprint_5_inventory,
)

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
ZERO_OID = "0" * 40


@pytest.fixture(scope="module")
def inventory() -> DeliveryInventory:
    """Build the real frozen inventory once for object-boundary tests."""

    return build_sprint_5_inventory(REPOSITORY_ROOT)


def _raw_record(
    *,
    path: str = "alphaforge/example.py",
    status: str = "A",
    old_mode: str = "000000",
    new_mode: str = "100644",
    old_blob: str = ZERO_OID,
    new_blob: str = "a" * 40,
) -> bytes:
    metadata = f":{old_mode} {new_mode} {old_blob} {new_blob} {status}".encode("ascii")
    return metadata + b"\x00" + path.encode("ascii") + b"\x00"


def test_inventory_uses_the_six_frozen_work_items(inventory: DeliveryInventory) -> None:
    assert inventory.source_head == SOURCE_HEAD
    assert [(item.mr_group, item.issue_number, item.commit) for item in inventory.slices] == [
        (item.mr_group, item.issue_number, item.commit) for item in SPRINT_5_SLICES
    ]
    assert inventory.slices[-1].commit == SOURCE_HEAD
    assert all(
        current.parent == previous.commit
        for previous, current in zip(inventory.slices, inventory.slices[1:], strict=False)
    )


def test_every_summary_reconciles_exactly_to_serialized_paths(
    inventory: DeliveryInventory,
) -> None:
    document = inventory.to_dict()
    assert set(document) == {
        "schema_version",
        "inventory_id",
        "source_head",
        "slices",
        "totals",
    }
    assert document["totals"]["changed_path_count"] == sum(
        len(item["paths"]) for item in document["slices"]
    )
    assert (
        sum(document["totals"]["category_counts"].values())
        == document["totals"]["changed_path_count"]
    )
    for slice_document in document["slices"]:
        assert slice_document["changed_path_count"] == len(slice_document["paths"])
        assert sum(slice_document["category_counts"].values()) == len(slice_document["paths"])
        assert [item["path"] for item in slice_document["paths"]] == sorted(
            item["path"] for item in slice_document["paths"]
        )


def test_path_records_contain_real_blob_identities(inventory: DeliveryInventory) -> None:
    records = [path for item in inventory.slices for path in item.paths]
    assert records
    for record in records:
        assert set(record.to_dict()) == {"path", "status", "mode", "blob", "bytes", "category"}
        assert len(record.blob) == 40
        assert record.mode in {"100644", "100755"}
        assert record.bytes >= 0
        assert record.category == _classify_path(record.path)


def test_docs_only_slice_is_derived_as_documentation(inventory: DeliveryInventory) -> None:
    mr2 = inventory.slices[0]
    assert mr2.mr_group == "SF-S5-MR2"
    assert mr2.category_counts == {"documentation": mr2.changed_path_count}
    assert {item.path for item in mr2.paths} == {
        "docs/adr/0015-broker-selection-for-paper-and-live-trading.md",
        "docs/backtesting.md",
        "docs/broker_connectivity_requirements.md",
        "docs/signal_foundry.md",
    }


def test_serialization_and_identity_are_deterministic(inventory: DeliveryInventory) -> None:
    rebuilt = build_sprint_5_inventory(REPOSITORY_ROOT)
    assert rebuilt == inventory
    assert rebuilt.inventory_id == inventory.inventory_id
    assert rebuilt.to_json().encode("utf-8") == inventory.to_json().encode("utf-8")
    assert inventory.to_json().endswith("\n")
    assert json.loads(inventory.to_json()) == inventory.to_dict()
    assert len(inventory.inventory_id) == 64


def test_strict_inventory_parser_round_trips_the_frozen_document(
    inventory: DeliveryInventory,
) -> None:
    assert DeliveryInventory.from_dict(inventory.to_dict()) == inventory


@pytest.mark.parametrize(
    "mutation",
    [
        "unknown_top_level_key",
        "boolean_total",
        "wrong_slice_total",
        "wrong_category_total",
        "wrong_identity",
        "unknown_path_key",
        "wrong_path_category_type",
    ],
)
def test_strict_inventory_parser_rejects_schema_and_reconciliation_mutations(
    inventory: DeliveryInventory,
    mutation: str,
) -> None:
    document = json.loads(inventory.to_json())
    if mutation == "unknown_top_level_key":
        document["unexpected"] = None
    elif mutation == "boolean_total":
        document["totals"]["changed_path_count"] = True
    elif mutation == "wrong_slice_total":
        document["slices"][0]["changed_path_count"] += 1
    elif mutation == "wrong_category_total":
        category = next(iter(document["totals"]["category_counts"]))
        document["totals"]["category_counts"][category] += 1
    elif mutation == "wrong_identity":
        document["inventory_id"] = "0" * 64
    elif mutation == "unknown_path_key":
        document["slices"][0]["paths"][0]["unexpected"] = None
    else:
        document["slices"][0]["paths"][0]["category"] = []

    with pytest.raises(Sprint5InventoryError):
        DeliveryInventory.from_dict(document)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("status", "R"),
        ("mode", "120000"),
        ("blob", "not-a-sha"),
        ("bytes", True),
        ("bytes", -1),
        ("bytes", MAX_TRACKED_BLOB_BYTES + 1),
        ("category", "source"),
        ("category", "invented"),
    ],
)
def test_malformed_delivery_paths_fail_closed(field: str, value: object) -> None:
    arguments: dict[str, object] = {
        "path": "docs/evidence.md",
        "status": "A",
        "mode": "100644",
        "blob": "a" * 40,
        "bytes": 1,
        "category": "documentation",
    }
    arguments[field] = value
    with pytest.raises(Sprint5InventoryError):
        DeliveryPath(**arguments)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("mr_group", "SF-S5-WORK2"),
        ("mr_group", "SF-S5-MR2\n"),
        ("issue_number", 0),
        ("issue_number", True),
        ("commit", "abc"),
        ("commit", "A" * 40),
    ],
)
def test_malformed_slice_specs_fail_closed(field: str, value: object) -> None:
    arguments: dict[str, object] = {
        "mr_group": "SF-S5-MR2",
        "issue_number": 99,
        "commit": "a" * 40,
    }
    arguments[field] = value
    with pytest.raises(Sprint5InventoryError):
        DeliverySliceSpec(**arguments)  # type: ignore[arg-type]


def test_missing_duplicate_and_wrong_slices_fail_exact_plan_reconciliation(
    inventory: DeliveryInventory,
) -> None:
    with pytest.raises(Sprint5InventoryError, match="six frozen Sprint 5 capability slices"):
        replace(inventory, slices=inventory.slices[:-1])
    with pytest.raises(Sprint5InventoryError, match="six frozen Sprint 5 capability slices"):
        replace(
            inventory,
            slices=(inventory.slices[0], inventory.slices[0], *inventory.slices[2:]),
        )
    wrong_commit = replace(inventory.slices[0], commit=inventory.slices[0].parent)
    with pytest.raises(Sprint5InventoryError, match="six frozen Sprint 5 capability slices"):
        replace(inventory, slices=(wrong_commit, *inventory.slices[1:]))


def test_slice_and_chain_invariants_fail_closed(inventory: DeliveryInventory) -> None:
    first = inventory.slices[0]
    with pytest.raises(Sprint5InventoryError, match="parent and tree"):
        replace(first, parent="not-a-sha")
    with pytest.raises(Sprint5InventoryError, match="at least one path"):
        replace(first, paths=())
    with pytest.raises(Sprint5InventoryError, match="sorted deterministically"):
        replace(first, paths=tuple(reversed(first.paths)))
    with pytest.raises(Sprint5InventoryError, match="unique"):
        replace(first, paths=(first.paths[0], first.paths[0]))

    oversized = tuple(
        DeliveryPath(
            path=f"docs/generated/{index:04d}.md",
            status="A",
            mode="100644",
            blob="a" * 40,
            bytes=1,
            category="documentation",
        )
        for index in range(MAX_CHANGED_PATHS + 1)
    )
    with pytest.raises(Sprint5InventoryError, match="exceeds the bound"):
        DeliverySlice(
            mr_group=first.mr_group,
            issue_number=first.issue_number,
            commit=first.commit,
            parent=first.parent,
            tree=first.tree,
            paths=oversized,
        )

    bad_chain_member = replace(inventory.slices[1], parent="a" * 40)
    with pytest.raises(Sprint5InventoryError, match="linear chain"):
        replace(inventory, slices=(inventory.slices[0], bad_chain_member, *inventory.slices[2:]))
    with pytest.raises(Sprint5InventoryError, match="source_head"):
        replace(inventory, source_head="0" * 40)


@pytest.mark.parametrize(
    "path",
    [
        "../escape.py",
        "docs/../escape.md",
        "/absolute.py",
        "double//separator.py",
        "windows\\path.py",
        "drive:path.py",
        "control\ncharacter.py",
        "trailing-dot.",
        f"docs/{'x' * 256}.md",
    ],
)
def test_unsafe_paths_fail_closed(path: str) -> None:
    with pytest.raises(Sprint5InventoryError):
        _validate_repository_path(path)


def test_raw_diff_parser_accepts_one_canonical_addition() -> None:
    (entry,) = _parse_raw_diff(_raw_record())
    assert entry.path == "alphaforge/example.py"
    assert entry.status == "A"
    assert entry.new_blob == "a" * 40


def test_raw_diff_parser_supports_regular_modifications_and_deletions() -> None:
    modified = _raw_record(
        status="M",
        old_mode="100644",
        new_mode="100755",
        old_blob="b" * 40,
        new_blob="c" * 40,
    )
    deleted = _raw_record(
        path="docs/deleted.md",
        status="D",
        old_mode="100644",
        new_mode="000000",
        old_blob="d" * 40,
        new_blob=ZERO_OID,
    )
    entries = _parse_raw_diff(modified + deleted)
    assert [(entry.path, entry.status) for entry in entries] == [
        ("alphaforge/example.py", "M"),
        ("docs/deleted.md", "D"),
    ]


def test_rename_accounting_is_explicit_delete_plus_add() -> None:
    assert "rename detection is deliberately disabled" in (inventory_module.__doc__ or "")
    assert "one deletion plus one addition" in (inventory_module.__doc__ or "")
    with pytest.raises(
        Sprint5InventoryError, match="records moves as one deletion plus one addition"
    ):
        _parse_raw_diff(_raw_record(status="R100"))


@pytest.mark.parametrize(
    "payload",
    [
        b"not raw diff\x00path.py\x00",
        _raw_record()[:-1],
        _raw_record(status="R100"),
        _raw_record(new_mode="120000"),
        _raw_record(new_mode="160000"),
        _raw_record(old_mode="100644", old_blob="b" * 40),
        _raw_record(new_blob="z" * 40),
        _raw_record()[: -len("alphaforge/example.py") - 1] + b"\xff\x00",
    ],
)
def test_malformed_or_unsafe_raw_diffs_fail_closed(payload: bytes) -> None:
    with pytest.raises(Sprint5InventoryError):
        _parse_raw_diff(payload)


def test_duplicate_paths_in_a_raw_diff_fail_closed() -> None:
    record = _raw_record()
    with pytest.raises(Sprint5InventoryError, match="repeats changed path"):
        _parse_raw_diff(record + record)


def test_incomplete_and_path_count_exhaustion_fail_closed() -> None:
    metadata_only = _raw_record().split(b"\x00", maxsplit=1)[0] + b"\x00"
    with pytest.raises(Sprint5InventoryError, match="incomplete path record"):
        _parse_raw_diff(metadata_only)
    with pytest.raises(Sprint5InventoryError, match="exceeds"):
        _parse_raw_diff(_raw_record() * (MAX_CHANGED_PATHS + 1))


def test_oversized_raw_diff_fails_before_parsing() -> None:
    with pytest.raises(Sprint5InventoryError, match="output bound"):
        _parse_raw_diff(b"x" * (MAX_GIT_OUTPUT_BYTES + 1))


def test_missing_commit_object_fails_closed() -> None:
    with pytest.raises(Sprint5InventoryError, match="rejected frozen delivery evidence"):
        _inspect_commit(REPOSITORY_ROOT, "f" * 40)


def test_commit_inspection_rejects_malformed_ids_and_non_commit_objects(
    inventory: DeliveryInventory,
) -> None:
    with pytest.raises(Sprint5InventoryError, match="full lowercase SHA-1"):
        _inspect_commit(REPOSITORY_ROOT, "not-a-commit")
    blob = inventory.slices[0].paths[0].blob
    with pytest.raises(Sprint5InventoryError, match="is not a commit"):
        _inspect_commit(REPOSITORY_ROOT, blob)


@pytest.mark.parametrize(
    ("payload", "maximum"),
    [
        (b"too long", 2),
        (b"\xff", 8),
        (b"", 8),
        (b"two\nlines\n", 32),
        (b"carriage\rreturn", 32),
    ],
)
def test_single_line_git_metadata_is_strict(payload: bytes, maximum: int) -> None:
    with pytest.raises(Sprint5InventoryError):
        _single_ascii_line(payload, field="test metadata", maximum=maximum)


def test_git_process_bounds_and_output_caps_fail_closed(monkeypatch: pytest.MonkeyPatch) -> None:
    with pytest.raises(Sprint5InventoryError, match="resource limits"):
        _run_git(REPOSITORY_ROOT, ("status",), maximum_stdout=-1)
    with pytest.raises(Sprint5InventoryError, match="resource limits"):
        _run_git(REPOSITORY_ROOT, ("status",) * (MAX_GIT_ARGUMENTS + 1), maximum_stdout=1)
    with pytest.raises(Sprint5InventoryError, match="resource limits"):
        _run_git(
            REPOSITORY_ROOT,
            ("status",),
            maximum_stdout=1,
            input_bytes=b"x" * (MAX_GIT_STDIN_BYTES + 1),
        )
    with pytest.raises(Sprint5InventoryError, match="NUL-free"):
        _run_git(REPOSITORY_ROOT, ("bad\x00argument",), maximum_stdout=1)
    with pytest.raises(Sprint5InventoryError, match="output bound"):
        _run_git(REPOSITORY_ROOT, ("rev-parse", "HEAD"), maximum_stdout=0)

    monkeypatch.setenv("PATH", "/definitely/missing")
    with pytest.raises(Sprint5InventoryError, match="unable to start"):
        _run_git(REPOSITORY_ROOT, ("status",), maximum_stdout=1)


@pytest.mark.parametrize(
    "response",
    [
        b"",
        b"malformed\n",
        f"{'a' * 40} blob not-an-int\n".encode("ascii"),
        f"{'b' * 40} blob 1\n".encode("ascii"),
        f"{'a' * 40} tree 1\n".encode("ascii"),
        f"{'a' * 40} blob -1\n".encode("ascii"),
        f"{'a' * 40} blob {MAX_TRACKED_BLOB_BYTES + 1}\n".encode("ascii"),
    ],
)
def test_blob_metadata_must_reconcile_exactly(
    response: bytes,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    entry = _DiffEntry(
        path="docs/example.md",
        status="A",
        old_mode="000000",
        new_mode="100644",
        old_blob=ZERO_OID,
        new_blob="a" * 40,
    )
    monkeypatch.setattr(inventory_module, "_run_git", lambda *args, **kwargs: response)
    with pytest.raises(Sprint5InventoryError):
        _blob_sizes(REPOSITORY_ROOT, (entry,))


def test_non_repository_and_symlink_roots_fail_closed(tmp_path: Path) -> None:
    with pytest.raises(Sprint5InventoryError, match="Git worktree"):
        build_sprint_5_inventory(tmp_path)
    linked = tmp_path / "linked-repository"
    linked.symlink_to(REPOSITORY_ROOT, target_is_directory=True)
    with pytest.raises(Sprint5InventoryError, match="non-symlink"):
        build_sprint_5_inventory(linked)


def test_a_repository_with_the_wrong_origin_fails_before_object_inspection(tmp_path: Path) -> None:
    checkout = tmp_path / "wrong-origin"
    subprocess.run(
        ["git", "init", "--quiet", str(checkout)],
        check=True,
        capture_output=True,
        timeout=5,
    )
    subprocess.run(
        [
            "git",
            "-C",
            str(checkout),
            "remote",
            "add",
            "origin",
            "https://github.com/example/not-alphaforge.git",
        ],
        check=True,
        capture_output=True,
        timeout=5,
    )
    with pytest.raises(Sprint5InventoryError, match="does not identify AlphaForge"):
        build_sprint_5_inventory(checkout)


@pytest.mark.parametrize(
    "origin",
    [
        "https://github.com/srgangaram-swe/AlphaForge.git",
        "https://github.com/srgangaram-swe/AlphaForge",
    ],
)
def test_checkout_accepts_only_exact_github_https_origin_forms(
    tmp_path: Path,
    origin: str,
) -> None:
    checkout = tmp_path / "expected-origin"
    subprocess.run(
        ["git", "init", "--quiet", str(checkout)],
        check=True,
        capture_output=True,
        timeout=5,
    )
    subprocess.run(
        ["git", "-C", str(checkout), "remote", "add", "origin", origin],
        check=True,
        capture_output=True,
        timeout=5,
    )

    assert inventory_module._validate_checkout(checkout) == checkout


@pytest.mark.parametrize(
    "origin",
    [
        "https://token@github.com/srgangaram-swe/AlphaForge.git",
        "https://github.com/srgangaram-swe/AlphaForge.git?ref=main",
        "https://github.com/srgangaram-swe/AlphaForge/",
        "https://github.com/srgangaram-swe/alphaforge.git",
        "git@github.com:srgangaram-swe/AlphaForge.git",
    ],
)
def test_checkout_rejects_lookalike_or_credential_bearing_origins(
    tmp_path: Path,
    origin: str,
) -> None:
    checkout = tmp_path / "lookalike-origin"
    subprocess.run(
        ["git", "init", "--quiet", str(checkout)],
        check=True,
        capture_output=True,
        timeout=5,
    )
    subprocess.run(
        ["git", "-C", str(checkout), "remote", "add", "origin", origin],
        check=True,
        capture_output=True,
        timeout=5,
    )

    with pytest.raises(Sprint5InventoryError, match="does not identify AlphaForge"):
        inventory_module._validate_checkout(checkout)


@pytest.mark.parametrize(
    ("path", "category"),
    [
        (".github/workflows/ci.yml", "automation"),
        ("alphaforge/readiness/checklist.py", "source"),
        ("benchmarks/example.py", "benchmarks"),
        ("configs/example.yaml", "configuration"),
        ("docs/example.md", "documentation"),
        ("pyproject.toml", "repository"),
        ("scripts/example.py", "tooling"),
        ("tests/test_example.py", "tests"),
    ],
)
def test_path_classification_is_stable_and_mutually_exclusive(path: str, category: str) -> None:
    assert _classify_path(_validate_repository_path(path)) == category


def test_ci_inventory_verification_fetches_frozen_git_objects() -> None:
    """Keep the remote test matrix capable of resolving frozen Sprint 5 commits."""

    workflow = yaml.safe_load(
        (REPOSITORY_ROOT / ".github/workflows/ci.yml").read_text(encoding="utf-8")
    )
    checkout = next(
        step
        for step in workflow["jobs"]["test"]["steps"]
        if str(step.get("uses", "")).startswith("actions/checkout@")
    )

    assert checkout["with"]["fetch-depth"] == 0
    assert checkout["with"]["persist-credentials"] is False
