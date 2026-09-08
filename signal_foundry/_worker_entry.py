"""Fixed isolated interpreter entry: import only this reviewed repository root."""

from __future__ import annotations

import sys
from pathlib import Path

if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from signal_foundry.worker import main

    raise SystemExit(main())
