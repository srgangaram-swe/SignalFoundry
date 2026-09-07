"""Adversarial tests for transactional AlphaForge Sprint 5 evidence (MR11)."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import replace
from functools import lru_cache
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest

import alphaforge.readiness.sprint_5_evidence as evidence_module
from alphaforge.distributed.benchmark_evidence import (
    TEST_EXECUTION_PROFILE,
    BenchmarkConfig,
    BenchmarkEvidence,
    load_benchmark_evidence,
    parse_benchmark_evidence_bytes,
    run_crossover_benchmark,
    write_benchmark_evidence,
)
from alphaforge.readiness import Verdict, minimal_capital_checklist
from alphaforge.readiness.sprint_5_evidence import (
    Sprint5EvidenceError,
    Sprint5LimitedVerificationWarning,
    Sprint5PostCommitError,
    publish_sprint_5_evidence,
    sprint_5_readiness,
    verify_sprint_5_bundle,
)
from benchmarks.benchmark_distributed_crossover import (
    benchmark_implementation,
    build_batch,
    busy_work,
)
from benchmarks.benchmark_distributed_crossover import (
    main as benchmark_main,
)

REPOSITORY = Path(__file__).resolve().parents[1]
BENCHMARK = Path("docs/evidence/signal_foundry_sprint_5/inputs/distribution_crossover_raw.json")
EXPECTED_PAYLOADS = {
    "checklist.json",
    "delivery_inventory.json",
    "distribution_crossover.csv",
    "distribution_crossover_raw.json",
    "readiness_decision.json",
    "readiness_report.md",
    "sprint_5_closeout.png",
}


@contextmanager
def _publication_parent() -> Iterator[Path]:
    parent = Path(tempfile.mkdtemp(prefix=".sprint-5-evidence-test.", dir=REPOSITORY))
    try:
        yield parent
    finally:
        shutil.rmtree(parent, ignore_errors=True)


@lru_cache(maxsize=1)
def _valid_benchmark_bytes() -> bytes:
    """Measure current-source production paths once; never relabel old timings.

    A dependency update legitimately invalidates the historical benchmark's lock
    binding. Tiny real serial/process-pool measurements exercise publication
    without treating the archived reference as a run under a different lock.
    Timings are not golden values; repeated publications share this one snapshot.
    """
    measured = run_crossover_benchmark(
        BenchmarkConfig(
            implementation=benchmark_implementation(),
            task_count=2,
            workers=2,
            iteration_counts=(10, 100),
            warmups=1,
            repetitions=7,
            max_total_seconds=60.0,
        ),
        function=busy_work,
        task_builder=build_batch,
        harness=benchmark_main,
    )
    payload = measured.canonical_bytes()
    evidence = parse_benchmark_evidence_bytes(payload)
    assert evidence.canonical_bytes() == payload
    return payload


@pytest.fixture(scope="module", autouse=True)
def _measure_before_fault_injection() -> Iterator[None]:
    """Freeze real measurements before any test monkeypatches shared I/O hooks."""
    _valid_benchmark_bytes()
    yield
    _valid_benchmark_bytes.cache_clear()


def _publish(destination: Path, benchmark: Path | None = None) -> dict[str, Any]:
    source = benchmark
    if source is None:
        source = destination.parent / "validated-benchmark-input.json"
        if not source.exists():
            source.write_bytes(_valid_benchmark_bytes())
    return publish_sprint_5_evidence(
        repository_root=REPOSITORY,
        benchmark_input=source,
        output=destination,
    )


def _assert_no_transaction_residue(parent: Path, destination: Path) -> None:
    assert not destination.exists()
    assert not (parent / f".{destination.name}.publish.lock").exists()
    assert not list(parent.glob(f".{destination.name}.staging.*"))


def _write_manifest(destination: Path, manifest: dict[str, Any]) -> None:
    manifest["bundle_id"] = evidence_module._manifest_identity(manifest)
    (destination / "manifest.json").write_bytes(evidence_module._json_bytes(manifest))


def _rebind_artifact(
    destination: Path,
    name: str,
    payload: bytes,
    *,
    bind_input: str | None = None,
) -> None:
    (destination / name).write_bytes(payload)
    manifest = json.loads((destination / "manifest.json").read_text(encoding="utf-8"))
    manifest["artifacts"][name] = {
        "bytes": len(payload),
        "sha256": hashlib.sha256(payload).hexdigest(),
    }
    if bind_input is not None:
        manifest["inputs"][bind_input] = hashlib.sha256(payload).hexdigest()
    _write_manifest(destination, manifest)


def _reidentified_benchmark(
    evidence: BenchmarkEvidence,
    *,
    implementation_changes: dict[str, str],
) -> BenchmarkEvidence:
    """Recompute every internal identity around a forged implementation claim."""

    implementation = replace(evidence.config.implementation, **implementation_changes)
    config = replace(evidence.config, implementation=implementation)
    samples = tuple(replace(sample, config_sha256=config.identity) for sample in evidence.samples)
    return BenchmarkEvidence.from_samples(
        config=config,
        environment=evidence.environment,
        samples=samples,
        limitations=evidence.limitations,
    )


def test_the_published_verdict_is_fail_closed() -> None:
    decision = sprint_5_readiness()
    assert decision.verdict is Verdict.NOT_READY
    assert len(decision.unmet) == len(minimal_capital_checklist().items) == 17


def test_historical_timings_cannot_be_relabelled_as_current_lock_evidence() -> None:
    historical = load_benchmark_evidence(REPOSITORY / BENCHMARK)
    current = parse_benchmark_evidence_bytes(_valid_benchmark_bytes())
    assert (
        historical.config.implementation.dependency_lock_sha256
        != current.config.implementation.dependency_lock_sha256
    )
    with _publication_parent() as parent:
        destination = parent / "stale-benchmark"
        with pytest.raises(Sprint5EvidenceError, match="does not reconcile to repository sources"):
            _publish(destination, BENCHMARK)
        _assert_no_transaction_residue(parent, destination)


def test_bundle_hashes_every_non_manifest_artifact_and_verifies() -> None:
    with _publication_parent() as parent:
        destination = parent / "closeout"
        manifest = _publish(destination)
        assert set(manifest["artifacts"]) == EXPECTED_PAYLOADS
        assert "manifest.json" not in manifest["artifacts"]
        assert "does not recursively hash itself" in manifest["manifest_integrity"]
        assert verify_sprint_5_bundle(destination, repository_root=REPOSITORY) == manifest
        for name, record in manifest["artifacts"].items():
            artifact = destination / name
            assert artifact.stat().st_size == record["bytes"] > 0
            assert len(record["sha256"]) == 64


def test_manifest_binds_every_material_source_and_renderer_input() -> None:
    with _publication_parent() as parent:
        destination = parent / "closeout"
        manifest = _publish(destination)
        source_files = manifest["generator"]["source_files"]
        assert set(source_files) == set(evidence_module._GENERATOR_SOURCE_PATHS)
        for relative, record in source_files.items():
            payload = (REPOSITORY / relative).read_bytes()
            assert record == {
                "bytes": len(payload),
                "sha256": hashlib.sha256(payload).hexdigest(),
            }
        assert set(manifest["generator"]["render_dependencies"]) == {
            "matplotlib",
            "numpy",
            "pandas",
            "Pillow",
            "seaborn",
        }
        assert manifest["generator"]["render_runtime"]["matplotlib_backend"].lower() == "agg"
        assert set(manifest["generator"]["render_runtime"]) == {
            "freetype_version",
            "libc_name",
            "libc_version",
            "machine",
            "matplotlib_backend",
            "platform_release",
            "platform_system",
            "python_implementation",
            "python_version",
        }
        assert "source_ref" not in manifest
        assert len(manifest["delivery_source_head"]) == 40


@pytest.mark.parametrize(
    ("implementation_changes", "match"),
    [
        ({"executor_source_sha256": "a" * 64}, "does not reconcile to repository sources"),
        (
            {"execution_profile": TEST_EXECUTION_PROFILE},
            "does not use the production execution contract",
        ),
    ],
)
def test_publisher_rejects_reidentified_forged_or_test_runtime_evidence(
    implementation_changes: dict[str, str],
    match: str,
) -> None:
    with _publication_parent() as parent:
        source = parent / "reidentified-forgery.json"
        original = parse_benchmark_evidence_bytes(_valid_benchmark_bytes())
        forged = _reidentified_benchmark(
            original,
            implementation_changes=implementation_changes,
        )
        source.write_bytes(forged.canonical_bytes())
        destination = parent / "closeout"

        with pytest.raises(Sprint5EvidenceError, match=match):
            _publish(destination, source)
        _assert_no_transaction_residue(parent, destination)


def test_one_bounded_benchmark_snapshot_is_parsed_and_copied_exactly(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with _publication_parent() as parent:
        benchmark = parent / "benchmark-race.json"
        original = _valid_benchmark_bytes()
        benchmark.write_bytes(original)
        frozen_inventory = evidence_module.build_sprint_5_inventory(REPOSITORY)
        original_reader = evidence_module.read_regular_file_snapshot
        source_reads = 0

        def observe_snapshot(path: str | Path, **kwargs: Any) -> Any:
            nonlocal source_reads
            if Path(path).resolve() == benchmark.resolve():
                source_reads += 1
            return original_reader(path, **kwargs)

        def mutate_after_snapshot(repository: Path) -> Any:
            assert repository == REPOSITORY
            benchmark.write_bytes(b'{"replaced_after_snapshot":true}\n')
            return frozen_inventory

        monkeypatch.setattr(evidence_module, "read_regular_file_snapshot", observe_snapshot)
        monkeypatch.setattr(
            evidence_module,
            "build_sprint_5_inventory",
            mutate_after_snapshot,
        )
        destination = parent / "closeout"
        manifest = _publish(destination, benchmark.relative_to(REPOSITORY))
        assert source_reads == 1
        assert (destination / "distribution_crossover_raw.json").read_bytes() == original
        assert manifest["inputs"]["benchmark_sha256"] == hashlib.sha256(original).hexdigest()


def test_bundle_is_byte_identical_including_png() -> None:
    with _publication_parent() as parent:
        first = parent / "first"
        second = parent / "second"
        assert _publish(first) == _publish(second)
        assert {path.name for path in first.iterdir()} == {path.name for path in second.iterdir()}
        for source in first.iterdir():
            assert source.read_bytes() == (second / source.name).read_bytes(), source.name


def test_changed_valid_raw_input_changes_bundle_identity() -> None:
    with _publication_parent() as parent:
        original_path = parent / "original.json"
        original_path.write_bytes(_valid_benchmark_bytes())
        original = load_benchmark_evidence(original_path)
        changed_sample = replace(
            original.samples[0],
            serial_ns=original.samples[0].serial_ns + 1,
        )
        changed = BenchmarkEvidence.from_samples(
            config=original.config,
            environment=original.environment,
            samples=(changed_sample, *original.samples[1:]),
            limitations=original.limitations,
        )
        changed_path = write_benchmark_evidence(changed, parent / "changed.json")
        baseline_manifest = _publish(parent / "baseline")
        changed_manifest = _publish(parent / "changed", changed_path.relative_to(REPOSITORY))
        assert changed_manifest["bundle_id"] != baseline_manifest["bundle_id"]
        assert changed_manifest["inputs"]["benchmark_id"] == changed.benchmark_id


@pytest.mark.parametrize(
    "mutation",
    [
        "unknown_top_level_key",
        "boolean_sprint",
        "boolean_artifact_bytes",
        "missing_source_file",
        "uppercase_digest",
        "wrong_delivery_head",
        "wrong_measurement_type",
        "capital_claim",
    ],
)
def test_verifier_rejects_reidentified_manifest_schema_and_semantic_mutations(
    mutation: str,
) -> None:
    with _publication_parent() as parent:
        destination = parent / "closeout"
        manifest = _publish(destination)
        if mutation == "unknown_top_level_key":
            manifest["unexpected"] = None
        elif mutation == "boolean_sprint":
            manifest["sprint"] = True
        elif mutation == "boolean_artifact_bytes":
            manifest["artifacts"]["checklist.json"]["bytes"] = True
        elif mutation == "missing_source_file":
            manifest["generator"]["source_files"].pop("uv.lock")
        elif mutation == "uppercase_digest":
            manifest["inputs"]["benchmark_sha256"] = "A" * 64
        elif mutation == "wrong_delivery_head":
            manifest["delivery_source_head"] = "0" * 40
        elif mutation == "wrong_measurement_type":
            manifest["measurement_environment"]["logical_cpu_count"] = True
        else:
            manifest["capital_at_risk_usd"] = "1"
        _write_manifest(destination, manifest)
        with pytest.raises(Sprint5EvidenceError):
            verify_sprint_5_bundle(destination)


@pytest.mark.parametrize(
    ("artifact", "bind_input"),
    [
        ("checklist.json", None),
        ("delivery_inventory.json", "delivery_inventory_sha256"),
        ("distribution_crossover.csv", None),
        ("distribution_crossover_raw.json", "benchmark_sha256"),
        ("readiness_decision.json", None),
        ("readiness_report.md", None),
    ],
)
def test_verifier_rejects_rehashed_cross_artifact_semantic_mutations(
    artifact: str,
    bind_input: str | None,
) -> None:
    with _publication_parent() as parent:
        destination = parent / "closeout"
        _publish(destination)
        payload = (destination / artifact).read_bytes()
        if artifact == "delivery_inventory.json":
            document = json.loads(payload)
            document["totals"]["changed_path_count"] += 1
            changed = evidence_module._json_bytes(document)
        elif artifact == "distribution_crossover_raw.json":
            document = json.loads(payload)
            document["benchmark_id"] = "0" * 64
            changed = evidence_module._json_bytes(document)
        else:
            changed = payload + b"mutation\n"
        _rebind_artifact(destination, artifact, changed, bind_input=bind_input)
        with pytest.raises(Sprint5EvidenceError):
            verify_sprint_5_bundle(destination)


def test_standalone_verification_warns_about_external_provenance_limits() -> None:
    with _publication_parent() as parent:
        destination = parent / "closeout"
        manifest = _publish(destination)
        with pytest.warns(Sprint5LimitedVerificationWarning, match="did not rebuild Git"):
            assert verify_sprint_5_bundle(destination) == manifest


def test_rehashed_png_is_rejected_by_byte_exact_rerender() -> None:
    with _publication_parent() as parent:
        destination = parent / "closeout"
        _publish(destination)
        png = (destination / "sprint_5_closeout.png").read_bytes()
        _rebind_artifact(destination, "sprint_5_closeout.png", png + b"rehashed-tamper")
        with pytest.raises(Sprint5EvidenceError, match="byte-match a fresh render"):
            verify_sprint_5_bundle(destination, repository_root=REPOSITORY)


def test_rehashed_generator_record_requires_repository_aware_verification() -> None:
    with _publication_parent() as parent:
        destination = parent / "closeout"
        manifest = _publish(destination)
        relative = evidence_module._GENERATOR_SOURCE_PATHS[0]
        manifest["generator"]["source_files"][relative]["sha256"] = "0" * 64
        _write_manifest(destination, manifest)

        with pytest.warns(Sprint5LimitedVerificationWarning):
            verify_sprint_5_bundle(destination)
        with pytest.raises(Sprint5EvidenceError, match="source provenance"):
            verify_sprint_5_bundle(destination, repository_root=REPOSITORY)


def test_fabricated_rehashed_inventory_is_rejected_against_git_objects() -> None:
    with _publication_parent() as parent:
        destination = parent / "closeout"
        _publish(destination)
        inventory_path = destination / "delivery_inventory.json"
        inventory = json.loads(inventory_path.read_text(encoding="utf-8"))
        inventory["slices"][0]["paths"][0]["blob"] = "f" * 40
        identity_payload = {key: value for key, value in inventory.items() if key != "inventory_id"}
        inventory["inventory_id"] = hashlib.sha256(
            json.dumps(
                identity_payload,
                ensure_ascii=True,
                separators=(",", ":"),
                sort_keys=True,
            ).encode("ascii")
        ).hexdigest()
        inventory_bytes = evidence_module._json_bytes(inventory)
        _rebind_artifact(
            destination,
            "delivery_inventory.json",
            inventory_bytes,
            bind_input="delivery_inventory_sha256",
        )
        manifest = json.loads((destination / "manifest.json").read_text(encoding="utf-8"))
        manifest["inputs"]["delivery_inventory_id"] = inventory["inventory_id"]
        _write_manifest(destination, manifest)

        with pytest.warns(Sprint5LimitedVerificationWarning):
            verify_sprint_5_bundle(destination)
        with pytest.raises(Sprint5EvidenceError, match="rebuilt frozen Git objects"):
            verify_sprint_5_bundle(destination, repository_root=REPOSITORY)


def test_source_snapshot_occurs_before_render_and_is_rechecked_before_commit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with _publication_parent() as parent:
        destination = parent / "closeout"
        source_snapshots = evidence_module._source_snapshots(REPOSITORY)
        original_figure = evidence_module._figure
        source_calls = 0
        events: list[str] = []

        def controlled_sources(repository: Path) -> Any:
            nonlocal source_calls
            assert repository == REPOSITORY
            source_calls += 1
            events.append(f"source-{source_calls}")
            if source_calls < 3:
                return source_snapshots
            changed = dict(source_snapshots)
            relative = evidence_module._GENERATOR_SOURCE_PATHS[0]
            snapshot = changed[relative]
            payload = snapshot.data + b"simulated concurrent mutation"
            changed[relative] = replace(
                snapshot,
                data=payload,
                sha256=hashlib.sha256(payload).hexdigest(),
                size=len(payload),
            )
            return changed

        def observed_figure(*args: Any, **kwargs: Any) -> bytes:
            events.append("render")
            return original_figure(*args, **kwargs)

        monkeypatch.setattr(evidence_module, "_source_snapshots", controlled_sources)
        monkeypatch.setattr(evidence_module, "_figure", observed_figure)
        with pytest.raises(Sprint5EvidenceError, match="changed before commit"):
            _publish(destination)
        assert events[0:2] == ["source-1", "render"]
        assert events[-1] == "source-3"
        _assert_no_transaction_residue(parent, destination)


def test_manifest_binds_raw_digest_to_csv_and_inventory() -> None:
    with _publication_parent() as parent:
        destination = parent / "closeout"
        manifest = _publish(destination)
        csv_text = (destination / "distribution_crossover.csv").read_text(encoding="utf-8")
        assert manifest["inputs"]["benchmark_sha256"] in csv_text
        inventory = json.loads(
            (destination / "delivery_inventory.json").read_text(encoding="utf-8")
        )
        assert inventory["inventory_id"] == manifest["inputs"]["delivery_inventory_id"]
        assert sum(item["changed_path_count"] for item in inventory["slices"]) == (
            inventory["totals"]["changed_path_count"]
        )


def test_figure_uses_seaborn_for_raw_summary_and_delivery_views() -> None:
    with _publication_parent() as parent:
        with (
            patch.object(
                evidence_module.sns,
                "scatterplot",
                wraps=evidence_module.sns.scatterplot,
            ) as scatter,
            patch.object(
                evidence_module.sns,
                "lineplot",
                wraps=evidence_module.sns.lineplot,
            ) as line,
            patch.object(
                evidence_module.sns,
                "barplot",
                wraps=evidence_module.sns.barplot,
            ) as bars,
        ):
            _publish(parent / "closeout")
        assert scatter.call_count >= 1
        assert line.call_count >= 1
        assert bars.call_count >= 3


@pytest.mark.parametrize("failure_point", ["raw", "figure", "manifest", "rename"])
def test_injected_failure_leaves_no_destination_staging_or_lock(
    monkeypatch: pytest.MonkeyPatch,
    failure_point: str,
) -> None:
    with _publication_parent() as parent:
        destination = parent / "closeout"
        original_write = evidence_module._write_bytes_at

        if failure_point == "raw":

            def fail_raw(staging: Any, name: str, data: bytes) -> None:
                original_write(staging, name, data)
                if name == "distribution_crossover_raw.json":
                    raise OSError("injected raw writer failure")

            monkeypatch.setattr(evidence_module, "_write_bytes_at", fail_raw)
        elif failure_point == "figure":
            monkeypatch.setattr(
                evidence_module,
                "_figure",
                lambda *args, **kwargs: (_ for _ in ()).throw(OSError("injected render failure")),
            )
        elif failure_point == "manifest":

            def fail_manifest(staging: Any, name: str, data: bytes) -> None:
                original_write(staging, name, data)
                if name == "manifest.json":
                    raise OSError("injected manifest writer failure")

            monkeypatch.setattr(evidence_module, "_write_bytes_at", fail_manifest)
        else:
            monkeypatch.setattr(
                evidence_module,
                "_rename_staging",
                lambda *args, **kwargs: (_ for _ in ()).throw(OSError("injected rename failure")),
            )

        with pytest.raises(OSError, match="injected"):
            _publish(destination)
        _assert_no_transaction_residue(parent, destination)


def test_atomic_no_replace_preserves_a_destination_that_wins_the_race(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with _publication_parent() as parent:
        destination = parent / "closeout"
        original_rename = evidence_module._rename_staging
        raced_identity: tuple[int, int] | None = None

        def raced_rename(anchor: Any, staging: Any, requested_name: str) -> None:
            nonlocal raced_identity
            os.mkdir(requested_name, dir_fd=anchor.descriptor)
            metadata = os.stat(
                requested_name,
                dir_fd=anchor.descriptor,
                follow_symlinks=False,
            )
            raced_identity = (metadata.st_dev, metadata.st_ino)
            original_rename(anchor, staging, requested_name)

        monkeypatch.setattr(evidence_module, "_rename_staging", raced_rename)
        with pytest.raises(FileExistsError, match="already exists"):
            _publish(destination)
        metadata = destination.stat()
        assert raced_identity == (metadata.st_dev, metadata.st_ino)
        assert list(destination.iterdir()) == []
        assert not (parent / ".closeout.publish.lock").exists()
        assert not list(parent.glob(".closeout.staging.*"))


def test_parent_directory_identity_swap_fails_before_anchored_rename(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with _publication_parent() as parent:
        destination = parent / "closeout"
        moved = parent.with_name(f"{parent.name}.moved")
        original_rename = evidence_module._rename_staging

        def swap_parent(anchor: Any, staging: Any, requested_name: str) -> None:
            parent.rename(moved)
            parent.mkdir()
            try:
                original_rename(anchor, staging, requested_name)
            finally:
                parent.rmdir()
                moved.rename(parent)

        monkeypatch.setattr(evidence_module, "_rename_staging", swap_parent)
        with pytest.raises(Sprint5EvidenceError, match="parent changed identity"):
            _publish(destination)
        _assert_no_transaction_residue(parent, destination)


def test_staging_name_identity_swap_is_rejected_before_rename(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with _publication_parent() as parent:
        destination = parent / "closeout"
        original_rename = evidence_module._rename_staging

        def swap_staging(anchor: Any, staging: Any, requested_name: str) -> None:
            held_name = ".attacker-held-staging"
            os.rename(
                staging.name,
                held_name,
                src_dir_fd=anchor.descriptor,
                dst_dir_fd=anchor.descriptor,
            )
            os.mkdir(staging.name, dir_fd=anchor.descriptor)
            try:
                original_rename(anchor, staging, requested_name)
            finally:
                os.rmdir(staging.name, dir_fd=anchor.descriptor)
                os.rename(
                    held_name,
                    staging.name,
                    src_dir_fd=anchor.descriptor,
                    dst_dir_fd=anchor.descriptor,
                )

        monkeypatch.setattr(evidence_module, "_rename_staging", swap_staging)
        with pytest.raises(Sprint5EvidenceError, match="changed identity"):
            _publish(destination)
        _assert_no_transaction_residue(parent, destination)


def test_post_rename_destination_identity_swap_is_reported_and_preserved(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with _publication_parent() as parent:
        destination = parent / "closeout"
        held = parent / ".attacker-held-committed-bundle"
        original_rename = evidence_module._rename_staging
        replacement_identity: tuple[int, int] | None = None

        def swap_after_rename(anchor: Any, staging: Any, requested_name: str) -> None:
            nonlocal replacement_identity
            original_rename(anchor, staging, requested_name)
            os.rename(
                requested_name,
                held.name,
                src_dir_fd=anchor.descriptor,
                dst_dir_fd=anchor.descriptor,
            )
            os.mkdir(requested_name, dir_fd=anchor.descriptor)
            metadata = os.stat(
                requested_name,
                dir_fd=anchor.descriptor,
                follow_symlinks=False,
            )
            replacement_identity = (metadata.st_dev, metadata.st_ino)

        monkeypatch.setattr(evidence_module, "_rename_staging", swap_after_rename)
        with pytest.raises(Sprint5PostCommitError, match="do not retry blindly"):
            _publish(destination)
        metadata = destination.stat()
        assert replacement_identity == (metadata.st_dev, metadata.st_ino)
        assert list(destination.iterdir()) == []
        assert (held / "manifest.json").is_file()
        destination.rmdir()
        shutil.rmtree(held)


def test_failure_cleanup_does_not_remove_another_invocations_residue(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with _publication_parent() as parent:
        destination = parent / "closeout"
        unrelated = parent / ".closeout.staging.preexisting"
        unrelated.mkdir()
        marker = unrelated / "owned-by-another-invocation.txt"
        marker.write_text("preserve me\n", encoding="utf-8")
        monkeypatch.setattr(
            evidence_module,
            "_figure",
            lambda *args, **kwargs: (_ for _ in ()).throw(OSError("injected render failure")),
        )
        with pytest.raises(OSError, match="injected render failure"):
            _publish(destination)
        assert marker.read_text(encoding="utf-8") == "preserve me\n"
        assert not (parent / ".closeout.publish.lock").exists()


def test_lock_initialization_failure_is_guarded_and_cleaned(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with _publication_parent() as parent:
        destination = parent / "closeout"
        original_write = evidence_module._write_descriptor

        def fail_lock(descriptor: int, data: bytes) -> None:
            if data == evidence_module._LOCK_BYTES:
                raise OSError("injected lock write failure")
            original_write(descriptor, data)

        monkeypatch.setattr(evidence_module, "_write_descriptor", fail_lock)
        with pytest.raises(OSError, match="injected lock write failure"):
            _publish(destination)
        _assert_no_transaction_residue(parent, destination)


def test_lock_identity_swap_cleanup_preserves_the_replacement(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with _publication_parent() as parent:
        destination = parent / "closeout"
        lock = parent / ".closeout.publish.lock"
        moved = parent / ".attacker-held-lock"
        replacement_identity: tuple[int, int] | None = None

        def replace_lock_then_fail(*args: Any, **kwargs: Any) -> bytes:
            nonlocal replacement_identity
            lock.rename(moved)
            lock.write_bytes(b"replacement owned by another invocation\n")
            metadata = lock.stat()
            replacement_identity = (metadata.st_dev, metadata.st_ino)
            raise OSError("primary render failure")

        monkeypatch.setattr(evidence_module, "_figure", replace_lock_then_fail)
        with pytest.raises(BaseExceptionGroup):
            _publish(destination)
        metadata = lock.stat()
        assert replacement_identity == (metadata.st_dev, metadata.st_ino)
        assert moved.read_bytes() == evidence_module._LOCK_BYTES
        lock.unlink()
        moved.unlink()


def test_cleanup_failure_is_surfaced_without_skipping_other_owned_cleanup(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with _publication_parent() as parent:
        destination = parent / "closeout"
        monkeypatch.setattr(
            evidence_module,
            "_figure",
            lambda *args, **kwargs: (_ for _ in ()).throw(OSError("primary failure")),
        )
        original_cleanup = evidence_module._cleanup_staging

        def cleanup_with_injected_failure(*args: Any, **kwargs: Any) -> list[BaseException]:
            failures = original_cleanup(*args, **kwargs)
            failures.append(OSError("cleanup failure"))
            return failures

        monkeypatch.setattr(
            evidence_module,
            "_cleanup_staging",
            cleanup_with_injected_failure,
        )
        with pytest.raises(BaseExceptionGroup) as captured:
            _publish(destination)
        messages = {str(error) for error in captured.value.exceptions}
        assert messages == {"primary failure", "cleanup failure"}
        assert not (parent / ".closeout.publish.lock").exists()
        assert not list(parent.glob(".closeout.staging.*"))


def test_post_rename_parent_fsync_failure_preserves_the_committed_bundle(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with _publication_parent() as parent:
        destination = parent / "closeout"
        original_sync = evidence_module._sync_parent
        calls = 0

        def fail_first_sync(anchor: Any) -> None:
            nonlocal calls
            calls += 1
            if calls == 1:
                raise OSError("injected parent fsync failure")
            original_sync(anchor)

        monkeypatch.setattr(evidence_module, "_sync_parent", fail_first_sync)
        with pytest.raises(Sprint5PostCommitError, match="do not retry blindly"):
            _publish(destination)
        assert destination.is_dir()
        assert (
            verify_sprint_5_bundle(destination, repository_root=REPOSITORY)["verdict"]
            == "NOT_READY"
        )
        assert not (parent / ".closeout.publish.lock").exists()


def test_existing_destination_and_writer_lock_are_refused() -> None:
    with _publication_parent() as parent:
        destination = parent / "closeout"
        destination.mkdir()
        with pytest.raises(FileExistsError, match="already exists"):
            _publish(destination)
        destination.rmdir()
        lock = parent / ".closeout.publish.lock"
        lock.write_text("active\n", encoding="utf-8")
        with pytest.raises(FileExistsError, match="lock already exists"):
            _publish(destination)
        assert lock.read_text(encoding="utf-8") == "active\n"


@pytest.mark.parametrize("mutation", ["tamper", "missing", "extra", "symlink"])
def test_verifier_rejects_artifact_set_and_integrity_attacks(mutation: str) -> None:
    with _publication_parent() as parent:
        destination = parent / "closeout"
        _publish(destination)
        report = destination / "readiness_report.md"
        if mutation == "tamper":
            report.write_bytes(report.read_bytes() + b"x")
        elif mutation == "missing":
            report.unlink()
        elif mutation == "extra":
            (destination / "unexpected.txt").write_text("x\n", encoding="utf-8")
        else:
            report.unlink()
            report.symlink_to(destination / "checklist.json")
        with pytest.raises(Sprint5EvidenceError):
            verify_sprint_5_bundle(destination)


def test_paths_fail_closed_on_escape_and_symlink_input() -> None:
    with _publication_parent() as parent:
        valid_benchmark = parent / "valid-benchmark.json"
        valid_benchmark.write_bytes(_valid_benchmark_bytes())
        with pytest.raises(Sprint5EvidenceError, match="inside repository_root"):
            _publish(
                REPOSITORY.parent / "escaped-closeout",
                valid_benchmark.relative_to(REPOSITORY),
            )
        linked = parent / "benchmark-link.json"
        linked.symlink_to(REPOSITORY / BENCHMARK)
        with pytest.raises((Sprint5EvidenceError, ValueError), match="symlink|regular file"):
            _publish(parent / "closeout", linked.relative_to(REPOSITORY))
        real_parent = parent / "real-parent"
        real_parent.mkdir()
        linked_parent = parent / "linked-parent"
        linked_parent.symlink_to(real_parent, target_is_directory=True)
        with pytest.raises(Sprint5EvidenceError, match="symlink"):
            _publish(
                linked_parent / "closeout",
                valid_benchmark.relative_to(REPOSITORY),
            )


def test_text_artifacts_end_with_newline_and_limitations_are_honest() -> None:
    with _publication_parent() as parent:
        destination = parent / "closeout"
        manifest = _publish(destination)
        for name in EXPECTED_PAYLOADS - {"sprint_5_closeout.png"}:
            assert (destination / name).read_bytes().endswith(b"\n"), name
        limitations = " ".join(manifest["limitations"])
        assert "not a performance SLA" in limitations
        assert "not tests, quality, effort" in limitations
        assert "no paper/live trading or capital is authorized" in limitations
        assert manifest["capital_at_risk_usd"] == "0"
        assert manifest["verdict"] == "NOT_READY"
