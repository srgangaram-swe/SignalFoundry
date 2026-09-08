"""Backward-compatibility and failure-precedence tests for experiment tracking."""

from __future__ import annotations

import json
import logging
import sqlite3
import subprocess
import sys
import time
from contextlib import closing
from pathlib import Path
from types import ModuleType
from typing import Any, Literal, cast

import pytest
from typer.testing import CliRunner

import quant_platform.tracking.experiment as experiment_module
from quant_platform.cli import app
from quant_platform.config import AppConfig, TrackingConfig
from quant_platform.pipeline import Pipeline
from quant_platform.tracking import (
    ExperimentTracker,
    LegacyTrackingReadError,
    LegacyTrackingWriteError,
    get_tracker,
)
from quant_platform.tracking.experiment import (
    MLflowTracker,
    RunContext,
)


class _PipelineFailure(RuntimeError):
    """Sentinel failure raised by a pipeline stage."""


class _PersistenceFailure(OSError):
    """Sentinel failure raised by a persistence boundary."""


class _HostilePrimary(_PipelineFailure):
    """Primary whose diagnostic hook must never become an error replacement."""

    def add_note(self, note: str) -> None:
        del note
        raise RuntimeError("hostile add_note replacement payload")


class _HostileInt(int):
    """Integer subclass whose comparisons prove exact public validation ordering."""

    def __ge__(self, other: object) -> bool:
        del other
        raise RuntimeError("hostile comparison payload")


class _FailingTracker(ExperimentTracker):
    backend = "failing-test"

    def __init__(self, error: BaseException) -> None:
        super().__init__(TrackingConfig(backend="none"))
        self.error = error
        self.persisted_context: RunContext | None = None

    def _persist(self, ctx: RunContext) -> None:
        self.persisted_context = ctx
        raise self.error


class _MLflowSession:
    def __init__(
        self,
        *,
        exit_error: BaseException | None = None,
        suppress_body_error: bool = False,
    ) -> None:
        self.exit_error = exit_error
        self.suppress_body_error = suppress_body_error

    def __enter__(self) -> object:
        return object()

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: object,
    ) -> bool:
        if self.exit_error is not None:
            raise self.exit_error
        return self.suppress_body_error


def _install_fake_mlflow(
    monkeypatch: pytest.MonkeyPatch,
    *,
    log_error: BaseException | None = None,
    exit_error: BaseException | None = None,
    suppress_body_error: bool = False,
) -> None:
    module = ModuleType("mlflow")
    module.set_tracking_uri = lambda _uri: None  # type: ignore[attr-defined]
    module.set_experiment = lambda _name: None  # type: ignore[attr-defined]
    module.start_run = lambda **_kwargs: _MLflowSession(  # type: ignore[attr-defined]
        exit_error=exit_error,
        suppress_body_error=suppress_body_error,
    )
    module.log_params = lambda _params: None  # type: ignore[attr-defined]

    def log_metrics(_metrics: dict[str, float]) -> None:
        if log_error is not None:
            raise log_error

    module.log_metrics = log_metrics  # type: ignore[attr-defined]
    module.set_tags = lambda _tags: None  # type: ignore[attr-defined]
    module.log_artifact = lambda _artifact: None  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "mlflow", module)


_LEGACY_RECORD_KEYS = {
    "artifacts",
    "data_hash",
    "ended_at",
    "experiment",
    "features",
    "git_commit",
    "metrics",
    "name",
    "params",
    "run_id",
    "started_at",
    "status",
    "tags",
    "tickers",
}


def _legacy_record() -> dict[str, Any]:
    return {
        "run_id": "bounded-run",
        "experiment": "compatibility",
        "name": "bounded",
        "started_at": "2026-08-08T00:00:00+00:00",
        "ended_at": "2026-08-08T00:00:01+00:00",
        "status": "completed",
        "git_commit": None,
        "data_hash": None,
        "tickers": [],
        "features": [],
        "params": {},
        "metrics": {},
        "tags": {},
        "artifacts": [],
    }


def _deep_legacy_value() -> dict[str, object]:
    value: object = "leaf"
    for _ in range(18):
        value = {"next": value}
    return cast(dict[str, object], value)


@pytest.mark.parametrize("backend", ["json", "sqlite"])
def test_legacy_list_record_shape_is_unchanged(
    tmp_path: Path,
    backend: Literal["json", "sqlite"],
) -> None:
    config = TrackingConfig(
        backend=backend,
        experiment_name="shape-contract",
        db_path=str(tmp_path / "experiments.sqlite"),
        json_dir=str(tmp_path / "runs"),
    )
    tracker = get_tracker(config, base_dir=str(tmp_path))

    with tracker.run("shape-run") as context:
        context.log_params({"model": {"depth": 3}})
        context.log_metrics({"bt_sharpe": 0.5})
        context.log_tags({"purpose": "compatibility"})
        context.set_dataset(data_hash="dataset-1", tickers=["SPY"], features=["f_return"])

    runs = tracker.list_runs()
    assert len(runs) == 1
    assert set(runs[0]) == _LEGACY_RECORD_KEYS
    assert runs[0]["params"] == {"model.depth": 3}
    assert runs[0]["metrics"] == {"bt_sharpe": 0.5}
    assert runs[0]["tickers"] == ["SPY"]


def test_log_params_rejects_deep_input_without_partial_context_mutation() -> None:
    context = RunContext(
        run_id="bounded-params",
        experiment="compatibility",
        name="deep-input",
        started_at="2026-08-08T00:00:00+00:00",
    )
    value: object = "leaf"
    for _ in range(5_000):
        value = {"next": value}

    with pytest.raises(LegacyTrackingWriteError, match="bounded persistence contract"):
        context.log_params(cast(dict[str, Any], value))

    assert context.params == {}


def test_list_experiments_cli_shape_is_unchanged(tmp_path: Path) -> None:
    database = tmp_path / "experiments.sqlite"
    config = AppConfig.model_validate(
        {
            "project": {"name": "cli-shape"},
            "tracking": {
                "backend": "sqlite",
                "experiment_name": "compatibility",
                "db_path": str(database),
            },
        }
    )
    tracker = get_tracker(config.tracking)
    with tracker.run("visible-run") as context:
        context.log_metrics({"bt_sharpe": 0.75})
    config_path = tmp_path / "config.yaml"
    config.to_yaml(config_path)

    result = CliRunner().invoke(
        app,
        ["list-experiments", "--config", str(config_path), "--limit", "1"],
    )

    assert result.exit_code == 0, result.stdout
    assert "visible-run" in result.stdout
    assert "status=completed" in result.stdout
    assert "bt_sharpe=0.75" in result.stdout


def test_legacy_sqlite_collision_preserves_existing_row(tmp_path: Path) -> None:
    config = TrackingConfig(backend="sqlite", db_path=str(tmp_path / "experiments.sqlite"))
    tracker = get_tracker(config)
    original = RunContext(
        run_id="stable-run",
        experiment="compatibility",
        name="original",
        started_at="2026-08-08T00:00:00+00:00",
        status="completed",
    )
    replacement = RunContext(
        run_id="stable-run",
        experiment="compatibility",
        name="replacement",
        started_at="2026-08-08T01:00:00+00:00",
        status="failed",
    )

    tracker._persist(original)
    with pytest.raises(LegacyTrackingWriteError, match="persistence failed"):
        tracker._persist(replacement)

    assert tracker.list_runs(limit=1)[0]["name"] == "original"


def test_legacy_list_is_bounded_and_read_does_not_create_database(tmp_path: Path) -> None:
    database = tmp_path / "nested" / "experiments.sqlite"
    tracker = get_tracker(TrackingConfig(backend="sqlite", db_path=str(database)))

    assert tracker.list_runs(limit=1) == []
    assert not database.exists()
    assert not database.parent.exists()
    with pytest.raises(ValueError, match="between 1 and 1000"):
        tracker.list_runs(limit=1_001)


def test_all_legacy_list_surfaces_reject_hostile_integer_subclasses(
    tmp_path: Path,
) -> None:
    trackers = (
        ExperimentTracker(TrackingConfig(backend="none")),
        get_tracker(
            TrackingConfig(backend="json", json_dir=str(tmp_path / "json-runs")),
        ),
        get_tracker(
            TrackingConfig(backend="sqlite", db_path=str(tmp_path / "runs.sqlite")),
        ),
        MLflowTracker(TrackingConfig(backend="mlflow")),
    )

    for tracker in trackers:
        with pytest.raises(TypeError, match="integer or None"):
            tracker.list_runs(limit=_HostileInt(1))

    assert not (tmp_path / "json-runs").exists()
    assert not (tmp_path / "runs.sqlite").exists()


def test_legacy_json_listing_fails_closed_at_scan_bound(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    directory = tmp_path / "runs"
    directory.mkdir()
    for index in range(3):
        suffix = ".json" if index == 0 else ".ignored"
        (directory / f"run-{index}{suffix}").write_text("{}", encoding="utf-8")
    monkeypatch.setattr(experiment_module, "_MAX_LEGACY_JSON_SCAN", 2)
    tracker = get_tracker(
        TrackingConfig(backend="json", json_dir=str(directory)),
        base_dir=str(tmp_path),
    )

    with pytest.raises(LegacyTrackingReadError, match="bounded scan limit"):
        tracker.list_runs(limit=1)


def test_legacy_json_oversize_fails_before_decode(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    directory = tmp_path / "runs"
    directory.mkdir()
    (directory / "oversized.json").write_bytes(
        b"x" * (experiment_module._MAX_LEGACY_RECORD_BYTES + 1)
    )
    decoded = False

    def forbidden_decode(*_args: object, **_kwargs: object) -> object:
        nonlocal decoded
        decoded = True
        raise AssertionError("oversized bytes reached JSON decoding")

    monkeypatch.setattr(json, "loads", forbidden_decode)
    tracker = get_tracker(
        TrackingConfig(backend="json", json_dir=str(directory)),
        base_dir=str(tmp_path),
    )

    with pytest.raises(LegacyTrackingReadError, match="byte ceiling"):
        tracker.list_runs(limit=1)
    assert not decoded


@pytest.mark.parametrize(
    "mutation",
    [
        lambda record: record.__setitem__("metrics", {"bad": float("nan")}),
        lambda record: record.__setitem__("tags", _deep_legacy_value()),
    ],
)
def test_legacy_json_rejects_nonfinite_and_deep_records(
    tmp_path: Path,
    mutation: Any,
) -> None:
    directory = tmp_path / "runs"
    directory.mkdir()
    record = _legacy_record()
    mutation(record)
    (directory / "unsafe.json").write_text(
        json.dumps(record, allow_nan=True),
        encoding="utf-8",
    )
    tracker = get_tracker(
        TrackingConfig(backend="json", json_dir=str(directory)),
        base_dir=str(tmp_path),
    )

    with pytest.raises(LegacyTrackingReadError, match="bounded compatibility contract"):
        tracker.list_runs(limit=1)


@pytest.mark.parametrize("backend", ["json", "sqlite"])
def test_legacy_persistence_rejects_nonfinite_metrics_before_writing(
    tmp_path: Path,
    backend: Literal["json", "sqlite"],
) -> None:
    database = tmp_path / "experiments.sqlite"
    directory = tmp_path / "runs"
    tracker = get_tracker(
        TrackingConfig(
            backend=backend,
            db_path=str(database),
            json_dir=str(directory),
        ),
        base_dir=str(tmp_path),
    )

    with (
        pytest.raises(LegacyTrackingWriteError, match="bounded persistence contract"),
        tracker.run("unsafe-metric") as context,
    ):
        context.log_metrics({"unsafe": float("nan")})

    assert not database.exists()
    assert not directory.exists()


def test_legacy_sqlite_rejects_oversized_text_without_returning_partial_rows(
    tmp_path: Path,
) -> None:
    database = tmp_path / "experiments.sqlite"
    tracker = get_tracker(TrackingConfig(backend="sqlite", db_path=str(database)))
    with tracker.run("safe-row"):
        pass
    with closing(sqlite3.connect(database)) as connection, connection:
        connection.execute(
            "UPDATE runs SET metrics = ?",
            ("x" * (experiment_module._MAX_LEGACY_JSON_FIELD_BYTES + 1),),
        )
    with pytest.raises(sqlite3.ProgrammingError):
        connection.execute("SELECT 1")

    with pytest.raises(LegacyTrackingReadError, match="unsafe field"):
        tracker.list_runs(limit=1)


def test_legacy_sqlite_deadline_is_typed_and_redacted(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    database = tmp_path / "experiments.sqlite"
    tracker = get_tracker(TrackingConfig(backend="sqlite", db_path=str(database)))
    with tracker.run("deadline-row"):
        pass
    calls = 0

    def expired_clock() -> int:
        nonlocal calls
        calls += 1
        return 0 if calls == 1 else 1_000_000_000

    monkeypatch.setattr(experiment_module, "_LEGACY_SQLITE_PROGRESS_INSTRUCTIONS", 1)
    monkeypatch.setattr(time, "monotonic_ns", expired_clock)

    with pytest.raises(LegacyTrackingReadError, match="deadline") as caught:
        tracker.list_runs(limit=1)
    assert "sensitive" not in str(caught.value)


def test_legacy_json_run_name_never_becomes_a_path(tmp_path: Path) -> None:
    directory = tmp_path / "runs"
    tracker = get_tracker(
        TrackingConfig(backend="json", json_dir=str(directory)),
        base_dir=str(tmp_path),
    )

    with tracker.run("../../operator-secret"):
        pass

    records = list(directory.iterdir())
    assert len(records) == 1
    assert records[0].is_file()
    assert records[0].parent == directory
    assert "operator-secret" not in records[0].name
    assert not (tmp_path / "operator-secret").exists()


@pytest.mark.parametrize(
    ("field", "unsafe_value"),
    [
        ("run_id", "../../escape"),
        ("started_at", "../../escape/2026-08-08"),
    ],
)
def test_legacy_json_mutable_context_cannot_escape_record_root(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
    field: str,
    unsafe_value: str,
) -> None:
    directory = tmp_path / "runs"
    tracker = get_tracker(
        TrackingConfig(backend="json", json_dir=str(directory)),
        base_dir=str(tmp_path),
    )

    with (
        caplog.at_level(logging.INFO),
        pytest.raises(LegacyTrackingWriteError),
        tracker.run("mutable-boundary") as context,
    ):
        setattr(context, field, unsafe_value)

    assert "../" not in caplog.text
    assert "escape" not in caplog.text
    assert not directory.exists()
    assert not (tmp_path / "escape").exists()


def test_legacy_sqlite_storage_failure_is_typed_and_redacted(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tracker = get_tracker(
        TrackingConfig(backend="sqlite", db_path=str(tmp_path / "experiments.sqlite"))
    )

    def fail_connect(*_args: object, **_kwargs: object) -> sqlite3.Connection:
        raise sqlite3.OperationalError("sensitive absolute /private/operator/database")

    monkeypatch.setattr(sqlite3, "connect", fail_connect)

    with (
        pytest.raises(LegacyTrackingWriteError, match="persistence failed") as caught,
        tracker.run("storage-failure"),
    ):
        pass
    assert caught.value.__cause__ is None
    assert "sensitive" not in str(caught.value)
    assert "/private" not in str(caught.value)


def test_legacy_debug_logs_do_not_disclose_workstation_paths(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    database = tmp_path / "sensitive-workstation" / "experiments.sqlite"
    tracker = get_tracker(TrackingConfig(backend="sqlite", db_path=str(database)))

    with caplog.at_level(logging.DEBUG), tracker.run("redacted-path"):
        pass

    assert str(tmp_path) not in caplog.text


def test_legacy_logs_cli_and_summarizer_redact_unsafe_terminal_fields(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    database = tmp_path / "experiments.sqlite"
    unsafe_name = "/Users/example/private/credential-marker\n\x1b[31mINJECTED-ROW"
    unsafe_hash = "api_key=do-not-emit-this-credential-value-123456789"
    config = AppConfig.model_validate(
        {
            "tracking": {
                "backend": "sqlite",
                "experiment_name": "terminal-safety",
                "db_path": str(database),
            }
        }
    )
    tracker = get_tracker(config.tracking)
    with caplog.at_level(logging.INFO), tracker.run(unsafe_name) as context:
        context.set_dataset(data_hash=unsafe_hash, tickers=[], features=[])

    assert "credential-marker" not in caplog.text
    assert "INJECTED-ROW" not in caplog.text
    record = tracker.list_runs(limit=1)[0]
    assert record["name"] == "[redacted]"
    assert record["data_hash"] == "[redacted]"

    config_path = tmp_path / "unsafe-config.yaml"
    config.to_yaml(config_path)
    cli_result = CliRunner().invoke(
        app,
        ["list-experiments", "--config", str(config_path), "--limit", "1"],
    )
    assert cli_result.exit_code == 0, cli_result.output
    assert "[redacted]" in cli_result.output
    assert "credential-marker" not in cli_result.output
    assert "INJECTED-ROW" not in cli_result.output
    assert "do-not-emit" not in cli_result.output

    summary = subprocess.run(
        [
            sys.executable,
            "scripts/summarize_experiments.py",
            "--config",
            str(config_path),
            "--limit",
            "1",
        ],
        cwd=Path(__file__).resolve().parents[1],
        check=False,
        capture_output=True,
        text=True,
        timeout=10,
    )
    combined = summary.stdout + summary.stderr
    assert summary.returncode == 0, combined
    assert "[redacted]" in summary.stdout
    assert "credential-marker" not in combined
    assert "INJECTED-ROW" not in combined
    assert "do-not-emit" not in combined


def test_summarizer_bounds_limit_and_missing_store_is_read_only(tmp_path: Path) -> None:
    database = tmp_path / "missing" / "experiments.sqlite"
    config = AppConfig.model_validate({"tracking": {"backend": "sqlite", "db_path": str(database)}})
    config_path = tmp_path / "missing-config.yaml"
    config.to_yaml(config_path)
    command = [
        sys.executable,
        "scripts/summarize_experiments.py",
        "--config",
        str(config_path),
    ]

    missing = subprocess.run(
        command,
        cwd=Path(__file__).resolve().parents[1],
        check=False,
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert missing.returncode == 0, missing.stdout + missing.stderr
    assert "No tracked runs found." in missing.stdout
    assert not database.exists()
    assert not database.parent.exists()

    unbounded = subprocess.run(
        [*command, "--limit", "1001"],
        cwd=Path(__file__).resolve().parents[1],
        check=False,
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert unbounded.returncode == 2
    assert "between 1 and 1000" in unbounded.stderr
    assert not database.exists()


def test_list_experiments_rejects_unbounded_limit(tmp_path: Path) -> None:
    config = AppConfig.model_validate(
        {"tracking": {"backend": "sqlite", "db_path": str(tmp_path / "experiments.sqlite")}}
    )
    config_path = tmp_path / "config.yaml"
    config.to_yaml(config_path)

    result = CliRunner().invoke(
        app,
        ["list-experiments", "--config", str(config_path), "--limit", "1001"],
    )

    assert result.exit_code == 2
    assert "1000" in result.stdout + result.stderr


def test_pipeline_failure_is_durably_recorded_by_legacy_sqlite(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    database = tmp_path / "experiments.sqlite"
    config = AppConfig.model_validate(
        {
            "project": {"name": "failure-integration"},
            "tracking": {"backend": "sqlite", "db_path": str(database)},
        }
    )
    pipeline = Pipeline(config, base_dir=str(tmp_path))
    expected = _PipelineFailure("stage failed")

    def fail_ingest(*, force: bool = False) -> Any:
        del force
        raise expected

    monkeypatch.setattr(pipeline, "ingest", fail_ingest)

    with pytest.raises(_PipelineFailure) as caught:
        pipeline.run_full()

    assert caught.value is expected
    runs = get_tracker(config.tracking, base_dir=str(tmp_path)).list_runs()
    assert len(runs) == 1
    assert runs[0]["name"] == "failure-integration"
    assert runs[0]["status"] == "failed"
    assert runs[0]["ended_at"]


def test_persistence_failure_does_not_mask_run_failure(
    caplog: pytest.LogCaptureFixture,
) -> None:
    persistence_error = _PersistenceFailure("sensitive-do-not-log opaque-provider-detail")
    tracker = _FailingTracker(persistence_error)
    expected = _PipelineFailure("domain failure")

    with (
        caplog.at_level(logging.ERROR),
        pytest.raises(_PipelineFailure) as caught,
        tracker.run("negative-path"),
    ):
        raise expected

    assert caught.value is expected
    assert tracker.persisted_context is not None
    assert tracker.persisted_context.status == "failed"
    assert expected.__notes__ == [
        "failing-test experiment persistence also failed: _PersistenceFailure"
    ]
    assert "sensitive-do-not-log" not in caplog.text
    assert "opaque-provider-detail" not in caplog.text
    assert "secondary_error=_PersistenceFailure" in caplog.text


def test_persistence_failure_after_success_propagates_unchanged() -> None:
    expected = _PersistenceFailure("durability failed")
    tracker = _FailingTracker(expected)

    with (
        pytest.raises(_PersistenceFailure) as caught,
        tracker.run("apparently-successful"),
    ):
        pass

    assert caught.value is expected
    assert tracker.persisted_context is not None
    assert tracker.persisted_context.status == "completed"


def test_hostile_primary_diagnostic_hook_cannot_replace_base_tracker_failure() -> None:
    tracker = _FailingTracker(_PersistenceFailure("sensitive persistence payload"))
    expected = _HostilePrimary("authoritative run failure")

    with pytest.raises(_HostilePrimary) as caught, tracker.run("hostile-primary"):
        raise expected

    assert caught.value is expected
    assert tracker.persisted_context is not None
    assert tracker.persisted_context.status == "failed"


def test_mlflow_logging_failure_does_not_mask_run_failure(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    persistence_error = _PersistenceFailure("sensitive provider response")
    _install_fake_mlflow(monkeypatch, log_error=persistence_error)
    tracker = MLflowTracker(TrackingConfig(backend="mlflow"))
    expected = _PipelineFailure("model failed")

    with (
        caplog.at_level(logging.ERROR),
        pytest.raises(_PipelineFailure) as caught,
        tracker.run("mlflow-negative"),
    ):
        raise expected

    assert caught.value is expected
    assert expected.__notes__ == ["mlflow experiment persistence also failed: _PersistenceFailure"]
    assert "sensitive provider response" not in caplog.text


def test_mlflow_session_failure_does_not_mask_run_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    session_error = _PersistenceFailure("session teardown failed")
    _install_fake_mlflow(monkeypatch, exit_error=session_error)
    tracker = MLflowTracker(TrackingConfig(backend="mlflow"))
    expected = _PipelineFailure("pipeline failed")

    with (
        pytest.raises(_PipelineFailure) as caught,
        tracker.run("mlflow-session-negative"),
    ):
        raise expected

    assert caught.value is expected
    assert expected.__notes__ == [
        "mlflow experiment session finalization also failed: _PersistenceFailure"
    ]


def test_mlflow_session_cannot_suppress_run_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_fake_mlflow(monkeypatch, suppress_body_error=True)
    tracker = MLflowTracker(TrackingConfig(backend="mlflow"))
    expected = _PipelineFailure("must remain observable")

    with (
        pytest.raises(_PipelineFailure) as caught,
        tracker.run("mlflow-suppression-negative"),
    ):
        raise expected

    assert caught.value is expected
    assert expected.__notes__ == ["mlflow session finalization suppressed the run exception"]


@pytest.mark.parametrize("secondary_phase", ["logging", "session", "suppression"])
def test_hostile_primary_diagnostic_hook_cannot_replace_mlflow_failure(
    monkeypatch: pytest.MonkeyPatch,
    secondary_phase: str,
) -> None:
    _install_fake_mlflow(
        monkeypatch,
        log_error=(
            _PersistenceFailure("sensitive logging payload")
            if secondary_phase == "logging"
            else None
        ),
        exit_error=(
            _PersistenceFailure("sensitive session payload")
            if secondary_phase == "session"
            else None
        ),
        suppress_body_error=secondary_phase == "suppression",
    )
    tracker = MLflowTracker(TrackingConfig(backend="mlflow"))
    expected = _HostilePrimary("authoritative MLflow run failure")

    with (
        pytest.raises(_HostilePrimary) as caught,
        tracker.run(f"hostile-primary-{secondary_phase}"),
    ):
        raise expected

    assert caught.value is expected
