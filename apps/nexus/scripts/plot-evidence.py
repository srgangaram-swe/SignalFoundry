"""Plot measured browser samples; never fabricate timing or overwrite evidence."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("measurements", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("output exists; choose a new path and review before replacing")
    report = json.loads(args.measurements.read_text())
    frame = pd.DataFrame(report["samples"])
    sns.set_theme(style="whitegrid", palette="colorblind", font_scale=1.0)
    figure, axes = plt.subplots(1, 3, figsize=(14, 4.8), layout="constrained")
    try:
        latency = (
            frame[["cold_load_ms", "interaction_ms"]]
            .rename(
                columns={
                    "cold_load_ms": "Cold navigation",
                    "interaction_ms": "Table page",
                }
            )
            .melt(var_name="Operation", value_name="Milliseconds")
        )
        sns.stripplot(
            data=latency, x="Operation", y="Milliseconds", jitter=False, ax=axes[0]
        )
        axes[0].set(title="Every latency sample (n=8 per operation)", ylim=(0, None))
        sns.lineplot(
            x=range(1, len(frame) + 1),
            y=frame["renderer_cpu_s"] * 1000,
            marker="o",
            ax=axes[1],
        )
        axes[1].set(
            title="Renderer task time during table paging",
            xlabel="Sample",
            ylabel="Milliseconds",
            ylim=(0, None),
        )
        sns.lineplot(
            x=range(1, len(frame) + 1),
            y=frame["javascript_heap_bytes"] / 1048576,
            marker="s",
            ax=axes[2],
        )
        axes[2].set(
            title="JavaScript heap (not process RSS)",
            xlabel="Sample",
            ylabel="MiB",
            ylim=(0, None),
        )
        env = report["environment"]
        figure.suptitle(
            f"Nexus: measured local Chromium {env['browser']} on {env['platform']} {env['arch']}\nSynthetic 2,048-row fixture; 40 rendered data rows; seed not applicable; no trading-performance claim",
            fontsize=11,
        )
        figure.savefig(
            args.output, dpi=150, metadata={"Software": "Signal Foundry Nexus"}
        )
    finally:
        plt.close(figure)


if __name__ == "__main__":
    main()
