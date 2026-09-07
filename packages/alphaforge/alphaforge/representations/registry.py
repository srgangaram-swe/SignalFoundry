"""Fail-closed construction for the frozen representation family."""

from __future__ import annotations

from alphaforge.representations.base import (
    NEURAL_REPRESENTATION_KINDS,
    BaseRepresentation,
    RepresentationConfig,
)
from alphaforge.representations.linear import LinearRepresentation


def create_representation(config: RepresentationConfig) -> BaseRepresentation:
    """Construct exactly the mechanism declared by ``config.kind``."""

    if config.kind in NEURAL_REPRESENTATION_KINDS:
        from alphaforge.representations.neural import NeuralRepresentation

        return NeuralRepresentation(config)
    return LinearRepresentation(config)
