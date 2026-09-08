"""Release-version consistency contracts."""

from importlib.metadata import version

import alphaforge


def test_distribution_metadata_matches_runtime_version() -> None:
    assert version("alphaforge") == alphaforge.__version__
