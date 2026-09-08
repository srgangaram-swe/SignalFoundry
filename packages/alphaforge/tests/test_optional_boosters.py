"""Locked-extra contract tests for optional native tree boosters.

Each backend runs in a fresh interpreter. This is intentional: mixing several
OpenMP/native model runtimes with pickle-family restoration in one long-lived
process has produced a native XGBoost crash on macOS. A clean worker is the
supported optional-artifact restoration boundary.
"""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

CASES: tuple[tuple[str, dict[str, Any]], ...] = (
    (
        "lightgbm",
        {
            "n_estimators": 10,
            "max_depth": 3,
            "min_samples_leaf": 2,
            "n_jobs": 1,
        },
    ),
    (
        "xgboost",
        {
            "n_estimators": 10,
            "max_depth": 3,
            "min_child_weight": 1.0,
            "n_jobs": 1,
        },
    ),
    (
        "catboost",
        {
            "n_estimators": 10,
            "max_depth": 3,
            "min_samples_leaf": 2,
            "n_jobs": 1,
        },
    ),
)

_ISOLATED_CONTRACT = r"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from alphaforge.models import AlphaModel, ModelError, create_model

name = sys.argv[1]
params = json.loads(sys.argv[2])
artifact = Path(sys.argv[3]) / f"{name}.joblib"
generator = np.random.default_rng(20260725)
values = generator.normal(size=(96, 4))
features = pd.DataFrame(values, columns=["value", "quality", "trend", "risk"])
features.loc[::13, "quality"] = np.nan
target = pd.Series(0.5 * values[:, 0] - 0.25 * values[:, 2])

first = create_model(name, **params).fit(features, target)
second = create_model(name, **params).fit(features, target)
first_prediction = first.predict(features)
np.testing.assert_allclose(first_prediction, second.predict(features), rtol=0.0, atol=1e-12)
assert np.isfinite(first_prediction).all()
assert first.metadata().params["n_jobs"] == 1
diagnostics = first.training_diagnostics()
assert diagnostics is not None
assert diagnostics.backend.lower().startswith(name)
assert diagnostics.status == "completed"
assert diagnostics.iteration_limit == 10
assert diagnostics.seed == 42

first.save(artifact)
try:
    AlphaModel.load(artifact)
except ModelError as exc:
    assert "executable binary" in str(exc)
else:
    raise AssertionError("untrusted optional model artifact was accepted")
restored = AlphaModel.load(artifact, trusted=True)
np.testing.assert_allclose(restored.predict(features), first_prediction)
assert restored.training_diagnostics() == diagnostics
"""


@pytest.mark.parametrize(("name", "params"), CASES)
def test_optional_booster_is_deterministic_finite_persistent_and_diagnostic(
    name: str, params: dict[str, Any], tmp_path: Path
) -> None:
    if importlib.util.find_spec(name) is None:
        pytest.skip(f"{name} is exercised by the locked ml-extra CI job")

    completed = subprocess.run(
        [sys.executable, "-c", _ISOLATED_CONTRACT, name, json.dumps(params), str(tmp_path)],
        check=False,
        capture_output=True,
        text=True,
        timeout=60,
    )

    assert completed.returncode == 0, (
        f"{name} isolated contract failed\nstdout:\n{completed.stdout}\n"
        f"stderr:\n{completed.stderr}"
    )
