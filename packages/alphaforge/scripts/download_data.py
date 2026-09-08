from __future__ import annotations

import argparse
from pathlib import Path

from alphaforge.config import load_data_config
from alphaforge.data import data_quality_report, load_prices
from alphaforge.research import write_frame_artifact


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download or load market data.")
    parser.add_argument("--config", default="configs/data.yaml")
    parser.add_argument("--output", default="data/processed/panel.table.json")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_data_config(args.config)
    panel, benchmark = load_prices(cfg)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    write_frame_artifact(panel, output)
    if cfg.get("quality_report"):
        data_quality_report(panel, cfg["quality_report"])
    print(f"saved {len(panel):,} rows to {output}; benchmark={benchmark}")


if __name__ == "__main__":
    main()
