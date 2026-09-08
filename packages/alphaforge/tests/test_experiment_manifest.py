"""Experiment-manifest, environment, seed, and artifact provenance tests."""

from __future__ import annotations

from pathlib import Path

import pytest

from alphaforge.research.manifest import (
    ExperimentManifest,
    ManifestValidationError,
    capture_environment,
    derive_seed_map,
    inventory_artifacts,
    redact_cli_arguments,
    refresh_experiment_manifest,
    write_experiment_manifest,
)


def _manifest(tmp_path: Path, *, date_end: str = "2024-12-31") -> ExperimentManifest:
    artifact = tmp_path / "metrics.json"
    artifact.write_text('{"score":0.0}\n', encoding="utf-8")
    return ExperimentManifest.build(
        code={"sha": "a" * 40, "branch": "feat/manifest", "dirty": False},
        dataset={"id": "b" * 64, "version": "1.0.0"},
        universe=["MSFT", "AAPL", "AAPL"],
        date_range={"start": "2020-01-01", "end": date_end},
        features={"names": ["return_1"], "version": "1"},
        label={"name": "fwd_ret_5", "horizon": 5},
        models=[{"name": "zero_baseline"}],
        validation={"scheme": "walk_forward", "embargo_days": 5},
        transaction_costs={"commission_bps": 1.0},
        root_seed=42,
        environment=capture_environment(
            environ={
                "PYTHONHASHSEED": "42",
                "NASDAQ_DATA_LINK_API_KEY": "must-not-appear",
            },
            packages=("numpy",),
        ),
        invocation={
            "entrypoint": "alphaforge-research",
            "arguments": redact_cli_arguments(["--api-key", "must-not-appear", "--seed=42"]),
        },
        execution={
            "started_at": "2026-07-24T00:00:00Z",
            "finished_at": "2026-07-24T00:00:01Z",
        },
        artifacts=inventory_artifacts(tmp_path),
    )


def test_named_seed_streams_are_order_independent_and_unique() -> None:
    first = derive_seed_map(42, ("model", "data", "search"))
    second = derive_seed_map(42, ("search", "model", "data"))

    assert first == second
    assert len(set(first.values())) == len(first)
    assert all(0 <= seed < 2**32 for seed in first.values())


@pytest.mark.parametrize("streams", [(), ("",), ("duplicate", "duplicate")])
def test_invalid_seed_streams_fail_closed(streams: tuple[str, ...]) -> None:
    with pytest.raises(ManifestValidationError):
        derive_seed_map(42, streams)


def test_environment_capture_is_allowlisted_and_secret_safe() -> None:
    snapshot = capture_environment(
        environ={
            "OMP_NUM_THREADS": "4",
            "NASDAQ_DATA_LINK_API_KEY": "must-not-appear",
            "OTHER_SECRET": "must-not-appear",
        },
        packages=(),
    )

    assert snapshot["environment"] == {"OMP_NUM_THREADS": "4"}
    assert snapshot["hardware"]["accelerator"]["backend"] in {"cpu", "cuda", "mps"}
    assert "must-not-appear" not in str(snapshot)


def test_cli_redaction_covers_split_and_inline_sensitive_options() -> None:
    arguments = redact_cli_arguments(
        ["run", "--api-key", "secret-1", "--token=secret-2", "--config", "safe.yaml"]
    )

    assert arguments == [
        "run",
        "--api-key",
        "[REDACTED]",
        "--token=[REDACTED]",
        "--config",
        "safe.yaml",
    ]
    assert "secret" not in str(arguments)


def test_manifest_identity_is_canonical_and_artifacts_are_hashed(tmp_path: Path) -> None:
    first = _manifest(tmp_path)
    second = _manifest(tmp_path)

    assert first.experiment_id == second.experiment_id
    assert first.universe == ("AAPL", "MSFT")
    assert first.artifacts[0]["path"] == "metrics.json"
    assert len(str(first.artifacts[0]["sha256"])) == 64
    assert "must-not-appear" not in str(first.to_dict())


def test_manifest_rejects_unsafe_artifact_and_invalid_time(tmp_path: Path) -> None:
    valid = _manifest(tmp_path).to_dict()
    valid["artifacts"][0]["path"] = "../escape"

    unsafe = ExperimentManifest(**valid)
    with pytest.raises(ManifestValidationError, match="safe and relative"):
        unsafe.validate()

    valid = _manifest(tmp_path).to_dict()
    valid["execution"] = {
        "started_at": "2026-07-24T00:00:02Z",
        "finished_at": "2026-07-24T00:00:01Z",
    }
    backwards = ExperimentManifest(**valid)
    with pytest.raises(ManifestValidationError, match="finished before"):
        backwards.validate()


def test_manifest_rejects_tampered_semantics_and_malformed_provenance(tmp_path: Path) -> None:
    valid = _manifest(tmp_path).to_dict()
    valid["label"]["horizon"] = 20

    tampered = ExperimentManifest(**valid)
    with pytest.raises(ManifestValidationError, match="canonical SHA-256"):
        tampered.validate()

    valid = _manifest(tmp_path).to_dict()
    valid["code"]["sha"] = "not-a-git-sha".ljust(40, "x")
    invalid_code = ExperimentManifest(**valid)
    with pytest.raises(ManifestValidationError, match="canonical SHA-256"):
        invalid_code.validate()


def test_manifest_rejects_non_utc_or_malformed_dates(tmp_path: Path) -> None:
    valid = _manifest(tmp_path).to_dict()
    valid["execution"]["finished_at"] = "2026-07-24T00:00:01-06:00"
    non_utc = ExperimentManifest(**valid)
    with pytest.raises(ManifestValidationError, match="must be UTC"):
        non_utc.validate()

    with pytest.raises(ManifestValidationError, match="ISO-8601 dates"):
        _manifest(tmp_path, date_end="2024-99-99")


def test_manifest_publication_and_artifact_refresh_preserve_identity(tmp_path: Path) -> None:
    original = _manifest(tmp_path)
    path = write_experiment_manifest(original, tmp_path / "run_manifest.json")
    (tmp_path / "later.csv").write_text("value\n1\n", encoding="utf-8")

    refreshed = refresh_experiment_manifest(
        tmp_path,
        finished_at="2026-07-24T00:00:03Z",
    )

    assert refreshed.experiment_id == original.experiment_id
    assert refreshed.execution["finished_at"] == "2026-07-24T00:00:03Z"
    assert {artifact["path"] for artifact in refreshed.artifacts} == {
        "later.csv",
        "metrics.json",
    }
    assert path.read_text(encoding="utf-8").endswith("\n")
    assert not list(tmp_path.glob(".run_manifest.*.tmp"))


def test_manifest_publication_rejects_unsafe_destination(tmp_path: Path) -> None:
    with pytest.raises(ManifestValidationError, match="run_manifest.json"):
        write_experiment_manifest(_manifest(tmp_path), tmp_path / "manifest.json")
