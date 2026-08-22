"""Offline integration evidence for the optional MLflow tracking client."""

from __future__ import annotations

from pathlib import Path

import pytest

from quant_platform.config import TrackingConfig
from quant_platform.tracking.experiment import MLflowTracker

mlflow = pytest.importorskip(
    "mlflow",
    reason="the optional MLflow client is exercised by its dedicated CI job",
)


def test_mlflow_skinny_persists_a_complete_synthetic_run(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Exercise the real fluent client without a server, socket, credential, or external data."""

    # MLflow keeps its filesystem adapter only for migration compatibility.  This test opts in for
    # one isolated temporary directory so the client contract can be proved without network access;
    # operator configurations must use an explicitly approved remote tracking service.
    monkeypatch.setenv("MLFLOW_ALLOW_FILE_STORE", "true")
    tracking_root = tmp_path / "tracking"
    artifact = tmp_path / "synthetic-evidence.txt"
    artifact.write_text("synthetic integration evidence\n", encoding="utf-8")
    experiment_name = "signalattice-offline-client-integration"
    tracker = MLflowTracker(
        TrackingConfig(
            backend="mlflow",
            experiment_name=experiment_name,
            mlflow_tracking_uri=tracking_root.resolve().as_uri(),
        )
    )

    with tracker.run("offline-client-smoke") as context:
        context.params["source"] = "synthetic"
        context.metrics["finite_metric"] = 1.0
        context.tags["evidence_class"] = "synthetic"
        context.artifacts.append(str(artifact))

    client = mlflow.tracking.MlflowClient(tracking_uri=tracking_root.resolve().as_uri())
    experiment = client.get_experiment_by_name(experiment_name)
    assert experiment is not None
    runs = client.search_runs([experiment.experiment_id])
    assert len(runs) == 1
    assert runs[0].data.params == {"source": "synthetic"}
    assert runs[0].data.metrics == {"finite_metric": 1.0}
    assert runs[0].data.tags["evidence_class"] == "synthetic"
    assert [item.path for item in client.list_artifacts(runs[0].info.run_id)] == [artifact.name]
    assert context.status == "completed"
