"""Measure real bounded local workers and render reproducible Seaborn evidence.

Synthetic aggregates are redistributable; no provider data or credentials enter
this harness. Timing samples are measured, not deterministic. Rendering from the
saved machine-readable report is deterministic under the committed lock.
"""

from __future__ import annotations

import argparse
import hashlib
import os
import platform
import tempfile
import threading
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd
import psutil
import seaborn as sns
from matplotlib.ticker import MaxNLocator, PercentFormatter

from signal_foundry.boundary import (
    MAX_EVIDENCE_BYTES,
    FoundryError,
    decode,
    encode,
    read_file,
)
from signal_foundry.contracts import ResearchEvidence, ResearchRequest
from signal_foundry.runner import Runner


def measure(operation: str, phase: str, action: Callable[[], Any]) -> dict[str, Any]:
    """Measure wall latency and sampled parent/descendant RSS at 20 ms cadence."""
    stop = threading.Event()
    samples: list[int] = []

    def sample() -> None:
        parent = psutil.Process()
        while not stop.is_set():
            resident = parent.memory_info().rss
            for child in parent.children(recursive=True):
                try:
                    resident += child.memory_info().rss
                except psutil.NoSuchProcess:
                    continue
            samples.append(resident)
            stop.wait(0.02)

    sampler = threading.Thread(target=sample, name="evidence-rss")
    sampler.start()
    start = time.perf_counter()
    status = "completed"
    try:
        action()
    except FoundryError as exc:
        if exc.code != "workers_busy":
            raise
        status = "capacity_rejected"
    finally:
        elapsed = time.perf_counter() - start
        stop.set()
        sampler.join(timeout=2)
    if sampler.is_alive() or not samples:
        raise RuntimeError("resource sampler failed to terminate or record evidence")
    return {
        "operation": operation,
        "phase": phase,
        "status": status,
        "wall_seconds": elapsed,
        "peak_tree_rss_mib": max(samples) / 2**20,
        "rss_samples": len(samples),
    }


def benchmark(root: Path, state: Path) -> dict[str, Any]:
    request = ResearchRequest()
    runner = Runner(root, state)
    records = []
    for _ in range(3):
        records.append(
            measure("catalog", "fresh_runner", lambda: Runner(root, state).catalog())
        )
    for _ in range(5):
        records.append(measure("catalog", "reused_runner", runner.catalog))
    for _ in range(3):
        records.append(
            measure("validate", "reused_runner", lambda: runner.validate(request))
        )
    evidence: list[ResearchEvidence] = []
    for _ in range(3):
        records.append(
            measure(
                "research",
                "reused_runner",
                lambda: evidence.append(runner.run(request, threading.Event())),
            )
        )
    if len({item.digest() for item in evidence}) != 1:
        raise RuntimeError("identical synthetic requests did not reproduce")
    barrier = threading.Barrier(8)

    def saturated() -> dict[str, Any]:
        barrier.wait(timeout=5)
        return measure("validate", "saturated_8", lambda: runner.validate(request))

    batch_start = time.perf_counter()
    with ThreadPoolExecutor(max_workers=8) as pool:
        saturation = list(pool.map(lambda _: saturated(), range(8)))
    batch_seconds = time.perf_counter() - batch_start
    records.extend(saturation)
    completed = sum(item["status"] == "completed" for item in saturation)
    return {
        "schema_version": "1.0.0",
        "kind": "measured_local_development_simulation",
        "environment": {
            "python": platform.python_version(),
            "system": platform.system(),
            "release": platform.release(),
            "architecture": platform.machine(),
            "logical_cpus": psutil.cpu_count(),
            "physical_cpus": psutil.cpu_count(logical=False),
            "ram_gib": round(psutil.virtual_memory().total / 2**30, 2),
        },
        "method": {
            "seed": 42,
            "rss_sample_seconds": 0.02,
            "saturation_clients": 8,
            "worker_slots": 2,
            "warmup": "none; all samples retained",
            "cold_definition": (
                "fresh Runner; subprocesses always cold; filesystem caches uncontrolled"
            ),
            "memory_scope": (
                "process tree, including concurrent work and sampler threads"
            ),
            "limitations": (
                "Small laptop sample; no live-order, market latency or performance"
                " improvement claim."
            ),
        },
        "samples": records,
        "saturation": {
            "batch_wall_seconds": batch_seconds,
            "completed": completed,
            "rejected": 8 - completed,
            "completed_per_second": completed / batch_seconds,
            "scope": (
                "Eight-client burst including thread startup/sampling; not sustained"
                " throughput."
            ),
        },
        "harness_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "repeatable_research_hash": evidence[0].digest(),
        "research": evidence[0].model_dump(mode="json"),
    }


def frame(evidence: ResearchEvidence, name: str) -> pd.DataFrame:
    table = next(item for item in evidence.tables if item.name == name)
    return pd.DataFrame(table.rows, columns=[item.name for item in table.columns])


def plot(report: dict[str, Any], output: Path) -> None:
    """Restore plotting state and close every owned figure even on export faults."""
    existing = set(plt.get_fignums())
    try:
        with plt.rc_context():
            _plot(report, output)
    finally:
        for identity in set(plt.get_fignums()) - existing:
            plt.close(identity)


def _plot(report: dict[str, Any], output: Path) -> None:
    """Render all measured outcomes, including denied work and losing curves."""
    sns.set_theme(style="whitegrid", palette="colorblind", font_scale=0.9)
    records = pd.DataFrame(report["samples"])
    fig, axes = plt.subplots(1, 3, figsize=(16, 5), constrained_layout=True)
    sns.scatterplot(
        data=records,
        x="operation",
        y="wall_seconds",
        hue="phase",
        style="phase",
        ax=axes[0],
    )
    axes[0].set(
        yscale="log",
        title="Local operation latency · every sample",
        ylabel="Wall seconds (log scale)",
        xlabel="Actual package-worker operation",
    )
    sns.scatterplot(
        data=records,
        x="operation",
        y="peak_tree_rss_mib",
        hue="phase",
        style="phase",
        ax=axes[1],
    )
    axes[1].set(
        title="Sampled process-tree memory",
        ylabel="Peak RSS (MiB; 20 ms sampling)",
        xlabel="Overlapping work shares process-tree RSS",
    )
    saturated = records.loc[records["phase"].eq("saturated_8")]
    sns.countplot(
        data=saturated, x="status", color=sns.color_palette("colorblind")[0], ax=axes[2]
    )
    axes[2].set(
        title="Eight simultaneous validation clients",
        xlabel="Two worker slots; excess requests denied",
        ylabel="Requests",
    )
    fig.suptitle(
        "Measured workstation boundary · not market execution latency · seed 42"
    )
    fig.savefig(
        output / "local-resources.png", dpi=150, metadata={"Software": "Signal Foundry"}
    )
    plt.close(fig)

    result = ResearchEvidence.model_validate_json(encode(report["research"]))
    curves = []
    for name in (result.request.model.name, *result.request.baselines):
        curve = frame(result, "equity_" + name)
        curve["model"] = name
        curve["date"] = pd.to_datetime(curve["date"])
        curves.append(curve)
    all_curves = pd.concat(curves, ignore_index=True)
    fig, axes = plt.subplots(2, 3, figsize=(17, 10), constrained_layout=True)
    for metric, axis, title in (
        ("cumulative_return", axes[0, 0], "Net costed return · common OOS calendar"),
        ("drawdown", axes[0, 1], "Drawdown · losses remain visible"),
    ):
        sns.lineplot(
            data=all_curves, x="date", y=metric, hue="model", style="model", ax=axis
        )
        axis.set(title=title, ylabel="Return fraction", xlabel="Synthetic market date")
        axis.tick_params(axis="x", rotation=25)
    sns.lineplot(
        data=frame(result, "learning"),
        x="window_id",
        y="mse",
        hue="model",
        style="model",
        markers=True,
        ax=axes[0, 2],
    )
    axes[0, 2].set(
        title="Learning diagnostic · out-of-sample MSE",
        ylabel="Squared daily-return fraction",
        xlabel="Chronological fold (not optimization epochs)",
    )
    sns.barplot(
        data=frame(result, "prediction_quantiles"),
        x="quantile",
        y="mean_return",
        color=sns.color_palette("colorblind")[0],
        ax=axes[1, 0],
    )
    axes[1, 0].set(
        title="Candidate prediction ranks vs realized targets",
        ylabel="Mean target return fraction",
        xlabel="Within-date prediction quantile",
    )
    sns.lineplot(
        data=all_curves,
        x="date",
        y="transaction_cost",
        hue="model",
        style="model",
        ax=axes[1, 1],
    )
    axes[1, 1].set(
        title="Execution costs · modeled, not observed",
        ylabel="Previous-equity fraction per session",
        xlabel="Synthetic market date",
    )
    axes[1, 1].tick_params(axis="x", rotation=25)
    intervals = frame(result, "uncertainty").melt(
        id_vars="model",
        value_vars=["lower", "mean", "upper"],
        var_name="statistic",
        value_name="daily_return",
    )
    sns.scatterplot(
        data=intervals,
        x="daily_return",
        y="model",
        hue="statistic",
        style="statistic",
        s=90,
        ax=axes[1, 2],
    )
    axes[1, 2].set(
        title="Mean-return uncertainty · 200 block resamples",
        xlabel="Daily net return fraction; 95% exploratory bounds",
        ylabel="Forecast plus identical signal/risk/cost policy",
    )
    axes[1, 2].xaxis.set_major_locator(MaxNLocator(5))
    axes[1, 2].xaxis.set_major_formatter(PercentFormatter(xmax=1, decimals=2))
    axes[1, 2].axvline(0, color="0.5", linestyle=":", linewidth=1)
    fig.suptitle(
        "Synthetic development evidence · seed 42 · no holdout/selection adjustment ·"
        " NOT_READY\n500 sessions, 8 tradable assets, 4 chronological folds;"
        " 242 common evaluation sessions"
    )
    fig.savefig(
        output / "research-diagnostics.png",
        dpi=150,
        metadata={"Software": "Signal Foundry"},
    )
    plt.close(fig)


def publish(report: dict[str, Any], destination: Path) -> None:
    """Publish all plots/inputs/hashes atomically, without replacing evidence."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() or destination.is_symlink():
        raise FoundryError(
            "evidence_exists", "Reviewed evidence cannot be overwritten.", 409
        )
    reservation = destination.with_name("." + destination.name + ".reservation")
    try:
        reservation.mkdir()
    except OSError as exc:
        raise FoundryError(
            "evidence_reserved",
            "Another publisher owns this evidence destination.",
            409,
        ) from exc
    try:
        with tempfile.TemporaryDirectory(
            prefix=".research-evidence-", dir=destination.parent
        ) as temporary:
            stage = Path(temporary) / "published"
            stage.mkdir()
            (stage / "measurements.json").write_bytes(encode(report) + b"\n")
            plot(report, stage)
            manifest = {
                path.name: hashlib.sha256(path.read_bytes()).hexdigest()
                for path in sorted(stage.iterdir())
            }
            (stage / "manifest.json").write_bytes(encode(manifest) + b"\n")
            if destination.exists() or destination.is_symlink():
                raise FoundryError(
                    "evidence_race",
                    "Evidence destination changed during publication.",
                    409,
                )
            os.rename(stage, destination)
    except OSError as exc:
        raise FoundryError(
            "publication_io",
            "Evidence publication failed; no completed artifact is implied.",
            503,
        ) from exc
    finally:
        reservation.rmdir()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--state", type=Path, default=Path("var/benchmark"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--input", type=Path, help="Render saved measurements without rerunning work"
    )
    args = parser.parse_args()
    report = (
        decode(read_file(args.input), MAX_EVIDENCE_BYTES)
        if args.input
        else benchmark(args.root, args.state)
    )
    publish(report, args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
