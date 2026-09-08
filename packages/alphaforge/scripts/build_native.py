"""Build the C++ extension without CMake: one compiler invocation.

Usage: python scripts/build_native.py
Requires: a C++17 compiler and `pip install pybind11`.
"""

from __future__ import annotations

import subprocess
import sys
import sysconfig
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    try:
        import pybind11
    except ImportError:
        print("pybind11 is required: pip install pybind11", file=sys.stderr)
        return 1

    suffix = sysconfig.get_config_var("EXT_SUFFIX") or ".so"
    out = ROOT / "alphaforge" / f"alphaforge_native{suffix}"
    cmd = [
        "c++",
        "-O3",
        "-std=c++17",
        "-shared",
        "-fPIC",
        f"-I{pybind11.get_include()}",
        f"-I{sysconfig.get_paths()['include']}",
        f"-I{ROOT / 'cpp' / 'include'}",
        str(ROOT / "cpp" / "src" / "bindings.cpp"),
        "-o",
        str(out),
    ]
    if sys.platform == "darwin":
        cmd.insert(1, "-undefined")
        cmd.insert(2, "dynamic_lookup")

    print(" ".join(cmd))
    subprocess.run(cmd, check=True)
    print(f"built {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
