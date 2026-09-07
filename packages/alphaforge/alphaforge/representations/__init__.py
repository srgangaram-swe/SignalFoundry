"""Leakage-safe latent and self-supervised representation contracts."""

from alphaforge.representations.base import (
    NEURAL_REPRESENTATION_KINDS,
    REPRESENTATION_KINDS,
    REPRESENTATION_SCHEMA_VERSION,
    BaseRepresentation,
    EmbeddingCollapseError,
    RepresentationBatch,
    RepresentationCapabilityError,
    RepresentationConfig,
    RepresentationError,
    RepresentationKind,
    RepresentationNotFittedError,
    RepresentationOutput,
    RepresentationResourceError,
    RepresentationSchemaError,
    RepresentationState,
    canonicalize_component_signs,
    named_seed,
    subspace_distance,
    subspace_fingerprint,
)
from alphaforge.representations.linear import (
    LINEAR_REPRESENTATION_KINDS,
    LinearRepresentation,
)
from alphaforge.representations.registry import create_representation

__all__ = [
    "NEURAL_REPRESENTATION_KINDS",
    "REPRESENTATION_KINDS",
    "REPRESENTATION_SCHEMA_VERSION",
    "BaseRepresentation",
    "EmbeddingCollapseError",
    "LINEAR_REPRESENTATION_KINDS",
    "LinearRepresentation",
    "RepresentationBatch",
    "RepresentationCapabilityError",
    "RepresentationConfig",
    "RepresentationError",
    "RepresentationKind",
    "RepresentationNotFittedError",
    "RepresentationOutput",
    "RepresentationResourceError",
    "RepresentationSchemaError",
    "RepresentationState",
    "canonicalize_component_signs",
    "create_representation",
    "named_seed",
    "subspace_distance",
    "subspace_fingerprint",
]
