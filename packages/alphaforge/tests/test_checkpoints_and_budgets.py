"""Tests for checkpointing, resumption, and enforced budgets (SF-S5-MR9).

Grouped by the guarantee each protects. The ones carrying the most weight:

* **A checkpoint refuses to resume into a different world.** Changing code,
  data, config, dependencies, seed, or the task graph makes the resumed run a
  different experiment wearing the original's name.
* **Budgets are hard.** Admission refuses before any worker starts; runtime
  enforcement catches the task that declared one hour and is four hours in.
* **A crash mid-write leaves the previous checkpoint intact**, never a truncated
  one that verifies as corrupt and destroys a resumable run.
* **Concurrent writers are detected, not merged.**
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest

from alphaforge.distributed import (
    CHECKPOINT_SCHEMA_VERSION,
    REQUIRED_BINDINGS,
    BudgetError,
    BudgetExceededError,
    CheckpointCorruptError,
    CheckpointError,
    CheckpointIncompatibleError,
    CheckpointManifest,
    CheckpointStore,
    ConcurrentWriterError,
    ExperimentBudget,
    LimitKind,
    ResourceRequest,
    ResourceUsage,
    TaskSpec,
    admit,
    breach_report,
    declared_usage,
    enforce,
    manifest_from_payload,
    task_graph_hash,
    verify_resumable,
)

NOW = datetime(2026, 8, 8, 12, 0, tzinfo=UTC)
CODE = "1" * 64
DATA = "2" * 64
CONFIG = "3" * 64
DEPS = "4" * 64


def _resources(**kw: Any) -> ResourceRequest:
    base: dict[str, Any] = {
        "cpus": 1.0,
        "memory_mb": 512,
        "gpus": 0,
        "scratch_mb": 100,
        "expected_seconds": 10.0,
    }
    base.update(kw)
    return ResourceRequest(**base)


def _task(value: int = 0, **kw: Any) -> TaskSpec:
    base: dict[str, Any] = {
        "name": f"task-{value}",
        "payload": {"value": value},
        "seed": value,
        "resources": _resources(),
        "timeout_seconds": 60.0,
    }
    base.update(kw)
    return TaskSpec(**base)


def _tasks(count: int = 4) -> list[TaskSpec]:
    return [_task(index) for index in range(count)]


def _budget(**kw: Any) -> ExperimentBudget:
    base: dict[str, Any] = {
        "wall_seconds": 3_600.0,
        "gpu_hours": 10.0,
        "concurrent_tasks": 8,
        "memory_mb": 16_384,
        "storage_mb": 10_000,
        "cost_units": 100.0,
    }
    base.update(kw)
    return ExperimentBudget(**base)


def _manifest(tasks: list[TaskSpec] | None = None, **kw: Any) -> CheckpointManifest:
    tasks = tasks if tasks is not None else _tasks()
    base: dict[str, Any] = {
        "experiment_id": "exp-1",
        "schema_version": CHECKPOINT_SCHEMA_VERSION,
        "code_hash": CODE,
        "data_hash": DATA,
        "config_hash": CONFIG,
        "dependency_hash": DEPS,
        "task_graph_hash": task_graph_hash(tasks),
        "root_seed": 42,
        "completed_task_ids": (tasks[0].task_id,),
        "writer_id": "writer-a",
        "written_at": NOW,
    }
    base.update(kw)
    return CheckpointManifest(**base)


def _verify(manifest: CheckpointManifest, tasks: list[TaskSpec], **kw: Any) -> None:
    base: dict[str, Any] = {
        "code_hash": CODE,
        "data_hash": DATA,
        "config_hash": CONFIG,
        "dependency_hash": DEPS,
        "tasks": tasks,
        "root_seed": 42,
    }
    base.update(kw)
    verify_resumable(manifest, **base)


# ---------------------------------------------------------------------------
# Checkpoints bind their whole world
# ---------------------------------------------------------------------------


def test_a_matching_world_resumes() -> None:
    tasks = _tasks()
    _verify(_manifest(tasks), tasks)


@pytest.mark.parametrize("binding", ["code_hash", "data_hash", "config_hash", "dependency_hash"])
def test_a_changed_binding_refuses_resumption(binding: str) -> None:
    """Resuming under changed code, data, config, or deps is a different experiment."""
    tasks = _tasks()
    with pytest.raises(CheckpointIncompatibleError, match=binding):
        _verify(_manifest(tasks), tasks, **{binding: "9" * 64})


def test_the_refusal_explains_why_each_binding_matters() -> None:
    tasks = _tasks()
    with pytest.raises(CheckpointIncompatibleError) as caught:
        _verify(_manifest(tasks), tasks, code_hash="9" * 64)
    assert "different logic" in str(caught.value)
    assert "different experiment under the original's name" in str(caught.value)


def test_a_changed_task_graph_refuses_resumption() -> None:
    tasks = _tasks(4)
    manifest = _manifest(tasks)
    with pytest.raises(CheckpointIncompatibleError, match="task_graph_hash"):
        _verify(manifest, _tasks(5))


def test_a_changed_seed_refuses_resumption() -> None:
    """Remaining tasks would draw from a different stream than completed ones."""
    tasks = _tasks()
    with pytest.raises(CheckpointIncompatibleError, match="root seed changed"):
        _verify(_manifest(tasks), tasks, root_seed=43)


def test_a_completed_id_absent_from_the_graph_refuses() -> None:
    tasks = _tasks()
    manifest = _manifest(tasks, completed_task_ids=("f" * 64,))
    with pytest.raises(CheckpointIncompatibleError, match="absent from the supplied graph"):
        _verify(manifest, tasks)


def test_the_task_graph_hash_is_order_independent() -> None:
    """Reordering submission has not changed the experiment."""
    tasks = _tasks(5)
    assert task_graph_hash(tasks) == task_graph_hash(list(reversed(tasks)))


def test_the_task_graph_hash_changes_with_membership() -> None:
    assert task_graph_hash(_tasks(4)) != task_graph_hash(_tasks(5))


def test_every_required_binding_is_declared() -> None:
    assert set(REQUIRED_BINDINGS) == {
        "code_hash",
        "data_hash",
        "config_hash",
        "dependency_hash",
        "task_graph_hash",
    }


def test_an_incompatible_schema_version_is_refused() -> None:
    with pytest.raises(CheckpointIncompatibleError, match="schema version"):
        _manifest(schema_version=99)


def test_a_tampered_manifest_fails_its_integrity_check() -> None:
    tasks = _tasks()
    manifest = _manifest(tasks)
    tampered = CheckpointManifest(
        experiment_id=manifest.experiment_id,
        schema_version=manifest.schema_version,
        code_hash=manifest.code_hash,
        data_hash=manifest.data_hash,
        config_hash=manifest.config_hash,
        dependency_hash=manifest.dependency_hash,
        task_graph_hash=manifest.task_graph_hash,
        root_seed=999,  # changed after the hash was computed
        completed_task_ids=manifest.completed_task_ids,
        writer_id=manifest.writer_id,
        written_at=manifest.written_at,
        content_hash=manifest.content_hash,
    )
    with pytest.raises(CheckpointCorruptError, match="integrity check"):
        tampered.verify_integrity()


def test_duplicate_completed_ids_are_refused() -> None:
    """A task counted twice would let the resumed run skip work it never did."""
    tasks = _tasks()
    with pytest.raises(CheckpointError, match="duplicate task id"):
        _manifest(tasks, completed_task_ids=(tasks[0].task_id, tasks[0].task_id))


def test_a_malformed_binding_hash_is_refused() -> None:
    with pytest.raises(CheckpointError, match="SHA-256"):
        _manifest(code_hash="tooshort")


def test_an_uppercase_hash_is_refused() -> None:
    with pytest.raises(CheckpointError, match="lowercase"):
        _manifest(code_hash="A" * 64)


def test_a_naive_timestamp_is_refused() -> None:
    with pytest.raises(CheckpointError, match="timezone-aware"):
        _manifest(written_at=datetime(2026, 8, 8, 12, 0))  # noqa: DTZ001


def test_remaining_excludes_completed_work_in_identity_order() -> None:
    tasks = _tasks(4)
    remaining = _manifest(tasks).remaining(tasks)
    assert len(remaining) == 3
    assert [item.task_id for item in remaining] == sorted(item.task_id for item in remaining)


# ---------------------------------------------------------------------------
# Atomic persistence and concurrent writers
# ---------------------------------------------------------------------------


def test_a_checkpoint_round_trips(tmp_path: Path) -> None:
    store = CheckpointStore(tmp_path, writer_id="writer-a")
    tasks = _tasks()
    store.write(_manifest(tasks))
    recovered = store.read("exp-1")
    assert recovered is not None
    recovered.verify_integrity()
    assert recovered.completed_task_ids == (tasks[0].task_id,)


def test_an_absent_checkpoint_reads_as_none(tmp_path: Path) -> None:
    assert CheckpointStore(tmp_path).read("never-written") is None


def test_a_failed_write_leaves_the_previous_checkpoint_intact(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Atomic replace: a crash mid-write must not truncate the existing record."""
    store = CheckpointStore(tmp_path, writer_id="writer-a")
    tasks = _tasks()
    store.write(_manifest(tasks))
    good = store.read("exp-1")
    assert good is not None

    def failing_replace(source: Any, target: Any) -> None:
        raise OSError("simulated crash between write and rename")

    monkeypatch.setattr("alphaforge.distributed.checkpoints.os.replace", failing_replace)
    with pytest.raises(OSError, match="simulated crash"):
        store.write(_manifest(tasks, completed_task_ids=(tasks[0].task_id, tasks[1].task_id)))
    monkeypatch.undo()

    after = store.read("exp-1")
    assert after is not None
    assert after.content_hash == good.content_hash, "the previous checkpoint must survive"
    leftovers = list(tmp_path.glob(".checkpoint-*.tmp"))
    assert leftovers == [], f"temporary files must be cleaned up, found {leftovers}"


def test_a_truncated_file_is_not_partially_usable(tmp_path: Path) -> None:
    store = CheckpointStore(tmp_path, writer_id="writer-a")
    store.write(_manifest())
    (tmp_path / "exp-1.json").write_text('{"experiment_id": "exp-1"', encoding="utf-8")
    with pytest.raises(CheckpointCorruptError, match="not readable JSON"):
        store.read("exp-1")


def test_a_payload_missing_fields_is_refused(tmp_path: Path) -> None:
    store = CheckpointStore(tmp_path, writer_id="writer-a")
    store.write(_manifest())
    payload = json.loads((tmp_path / "exp-1.json").read_text(encoding="utf-8"))
    del payload["code_hash"]
    (tmp_path / "exp-1.json").write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(CheckpointCorruptError, match="missing"):
        store.read("exp-1")


def test_a_second_writer_is_detected_not_merged(tmp_path: Path) -> None:
    """Interleaving two writers describes work no single run performed."""
    CheckpointStore(tmp_path, writer_id="writer-a").write(_manifest(writer_id="writer-a"))
    second = CheckpointStore(tmp_path, writer_id="writer-b")
    with pytest.raises(ConcurrentWriterError, match="last checkpointed by writer"):
        second.write(_manifest(writer_id="writer-b"))


def test_a_foreign_writer_may_take_over_deliberately(tmp_path: Path) -> None:
    CheckpointStore(tmp_path, writer_id="writer-a").write(_manifest(writer_id="writer-a"))
    second = CheckpointStore(tmp_path, writer_id="writer-b")
    second.write(_manifest(writer_id="writer-b"), allow_foreign_writer=True)
    recovered = second.read("exp-1")
    assert recovered is not None
    assert recovered.writer_id == "writer-b"


def test_the_same_writer_may_update_its_own_checkpoint(tmp_path: Path) -> None:
    store = CheckpointStore(tmp_path, writer_id="writer-a")
    tasks = _tasks()
    store.write(_manifest(tasks))
    store.write(_manifest(tasks, completed_task_ids=(tasks[0].task_id, tasks[1].task_id)))
    recovered = store.read("exp-1")
    assert recovered is not None
    assert len(recovered.completed_task_ids) == 2


def test_clearing_removes_the_checkpoint(tmp_path: Path) -> None:
    store = CheckpointStore(tmp_path, writer_id="writer-a")
    store.write(_manifest())
    assert store.clear("exp-1") is True
    assert store.clear("exp-1") is False
    assert store.read("exp-1") is None


def test_a_symlinked_directory_is_refused(tmp_path: Path) -> None:
    real = tmp_path / "real"
    real.mkdir()
    link = tmp_path / "link"
    link.symlink_to(real)
    with pytest.raises(CheckpointError, match="real directory"):
        CheckpointStore(link)


def test_a_store_generates_a_writer_id_when_none_is_given(tmp_path: Path) -> None:
    assert CheckpointStore(tmp_path).writer_id.startswith("writer-")


def test_a_manifest_rebuilds_from_its_payload() -> None:
    original = _manifest()
    rebuilt = manifest_from_payload(original.to_dict())
    assert rebuilt.content_hash == original.content_hash
    rebuilt.verify_integrity()


# ---------------------------------------------------------------------------
# Budget admission
# ---------------------------------------------------------------------------


def test_a_fitting_batch_is_admitted() -> None:
    decision = admit(_budget(), _tasks(4), concurrency=4, now=NOW)
    assert decision.admitted
    assert decision.breaches == ()


def test_a_batch_that_could_never_fit_is_refused_before_any_worker_starts() -> None:
    decision = admit(_budget(wall_seconds=1.0), _tasks(4), concurrency=1, now=NOW)
    assert not decision.admitted
    assert LimitKind.WALL_SECONDS in {item.kind for item in decision.breaches}


def test_wall_time_uses_the_critical_path_not_the_serial_total() -> None:
    """Charging the serial total would refuse work that fits comfortably."""
    tasks = [
        _task(i, resources=_resources(expected_seconds=100.0), timeout_seconds=600.0)
        for i in range(8)
    ]
    serial = declared_usage(tasks, concurrency=1)
    parallel = declared_usage(tasks, concurrency=8)
    assert serial.wall_seconds == pytest.approx(800.0)
    assert parallel.wall_seconds == pytest.approx(100.0)


def test_a_batch_cannot_finish_faster_than_its_slowest_task() -> None:
    tasks = [
        _task(0, resources=_resources(expected_seconds=500.0), timeout_seconds=600.0),
        _task(1, resources=_resources(expected_seconds=1.0)),
    ]
    assert declared_usage(tasks, concurrency=64).wall_seconds == pytest.approx(500.0)


def test_memory_is_the_peak_concurrent_requirement() -> None:
    tasks = [_task(i, resources=_resources(memory_mb=1_000)) for i in range(10)]
    assert declared_usage(tasks, concurrency=3).memory_mb == 3_000


def test_storage_accumulates_regardless_of_overlap() -> None:
    tasks = [_task(i, resources=_resources(scratch_mb=100)) for i in range(10)]
    assert declared_usage(tasks, concurrency=2).storage_mb == 1_000


def test_gpu_hours_accumulate() -> None:
    tasks = [
        _task(i, resources=_resources(gpus=2, expected_seconds=3_600.0), timeout_seconds=7_200.0)
        for i in range(2)
    ]
    assert declared_usage(tasks, concurrency=2).gpu_hours == pytest.approx(4.0)


def test_an_over_memory_batch_is_refused() -> None:
    tasks = [_task(i, resources=_resources(memory_mb=10_000)) for i in range(4)]
    decision = admit(_budget(memory_mb=1_000), tasks, concurrency=4, now=NOW)
    assert not decision.admitted
    assert LimitKind.MEMORY_MB in {item.kind for item in decision.breaches}


def test_an_over_storage_batch_is_refused() -> None:
    tasks = [_task(i, resources=_resources(scratch_mb=5_000)) for i in range(4)]
    decision = admit(_budget(storage_mb=1_000), tasks, concurrency=4, now=NOW)
    assert LimitKind.STORAGE_MB in {item.kind for item in decision.breaches}


def test_an_over_gpu_batch_is_refused() -> None:
    tasks = [
        _task(i, resources=_resources(gpus=4, expected_seconds=3_600.0), timeout_seconds=7_200.0)
        for i in range(10)
    ]
    decision = admit(_budget(gpu_hours=1.0), tasks, concurrency=10, now=NOW)
    assert LimitKind.GPU_HOURS in {item.kind for item in decision.breaches}


def test_a_refused_decision_must_name_a_breach() -> None:
    """An unexplained refusal cannot be acted on."""
    from alphaforge.distributed.budgets import AdmissionDecision

    with pytest.raises(BudgetError, match="must name at least one breach"):
        AdmissionDecision(
            admitted=False,
            budget=_budget(),
            declared=ResourceUsage(),
            breaches=(),
            decided_at=NOW.isoformat(),
        )


def test_an_empty_batch_cannot_be_costed() -> None:
    with pytest.raises(BudgetError, match="empty batch"):
        declared_usage([], concurrency=1)


# ---------------------------------------------------------------------------
# Runtime enforcement
# ---------------------------------------------------------------------------


def test_observed_overage_is_caught_even_when_admission_passed() -> None:
    """Admission trusts the declaration; enforcement does not."""
    budget = _budget(wall_seconds=100.0)
    enforce(budget, ResourceUsage(wall_seconds=50.0))
    with pytest.raises(BudgetExceededError, match="wall time"):
        enforce(budget, ResourceUsage(wall_seconds=400.0))


def test_enforcement_names_every_breached_limit() -> None:
    with pytest.raises(BudgetExceededError) as caught:
        enforce(
            _budget(wall_seconds=10.0, memory_mb=100),
            ResourceUsage(wall_seconds=99.0, memory_mb=9_999),
        )
    message = str(caught.value)
    assert "2 budget limit(s) exceeded" in message
    assert "no best-effort mode" in message


def test_exact_limit_consumption_is_permitted() -> None:
    """The bound is inclusive; only exceeding it fails."""
    enforce(_budget(wall_seconds=100.0), ResourceUsage(wall_seconds=100.0))


def test_concurrency_overage_is_caught() -> None:
    with pytest.raises(BudgetExceededError, match="simultaneously running"):
        enforce(_budget(concurrent_tasks=4), ResourceUsage(concurrent_tasks=5))


def test_cost_overage_is_caught() -> None:
    with pytest.raises(BudgetExceededError, match="cost units"):
        enforce(_budget(cost_units=10.0), ResourceUsage(cost_units=11.0))


def test_there_is_no_soft_or_best_effort_mode() -> None:
    """Guards the non-goal: budgets are hard refusals.

    Parses the AST rather than grepping the source, so the module may *describe*
    the modes it refuses to offer without the test matching its own prose.
    """
    import ast
    import inspect

    from alphaforge.distributed import budgets

    forbidden = {"soft", "warn_only", "allow_overage", "best_effort", "force", "override"}
    tree = ast.parse(inspect.getsource(budgets))
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            names = {arg.arg for arg in node.args.args + node.args.kwonlyargs}
            assert not (names & forbidden), f"{node.name} exposes {names & forbidden}"


# ---------------------------------------------------------------------------
# Evidence and validation
# ---------------------------------------------------------------------------


def test_breach_evidence_is_machine_readable() -> None:
    decision = admit(_budget(wall_seconds=1.0), _tasks(4), concurrency=1, now=NOW)
    payload = json.loads(breach_report(decision.breaches))
    assert payload["breach_count"] >= 1
    assert "not advisory" in payload["action"]
    assert payload["breaches"][0]["overage"] > 0


def test_the_admission_decision_serializes() -> None:
    payload = admit(_budget(), _tasks(), concurrency=4, now=NOW).to_dict()
    assert json.loads(json.dumps(payload))
    assert "before any worker starts" in payload["policy"]


@pytest.mark.parametrize(
    "override",
    [
        {"wall_seconds": 0.0},
        {"wall_seconds": -1.0},
        {"gpu_hours": float("nan")},
        {"concurrent_tasks": 0},
        {"memory_mb": -1},
        {"cost_units": 0.0},
    ],
)
def test_an_unusable_budget_is_refused(override: dict[str, Any]) -> None:
    with pytest.raises(BudgetError):
        _budget(**override)


def test_negative_observed_usage_is_refused() -> None:
    with pytest.raises(BudgetError, match="non-negative"):
        ResourceUsage(wall_seconds=-1.0)


def test_a_resumed_run_is_deterministic(tmp_path: Path) -> None:
    """Interrupt and resume must yield the same completed set as running through."""
    tasks = _tasks(6)
    store = CheckpointStore(tmp_path, writer_id="writer-a")
    store.write(_manifest(tasks, completed_task_ids=tuple(t.task_id for t in tasks[:3])))
    recovered = store.read("exp-1")
    assert recovered is not None
    _verify(recovered, tasks)
    remaining = recovered.remaining(tasks)
    combined = sorted([*recovered.completed_task_ids, *(t.task_id for t in remaining)])
    assert combined == sorted(t.task_id for t in tasks)
