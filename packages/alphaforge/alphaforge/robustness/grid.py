"""Frozen robustness grid and named seed streams (SF-S4-MR6).

A robustness study is only evidence if its grid was fixed before anyone saw a
result. Otherwise "we swept the parameters" means "we searched until something
looked good", and the sweep becomes the fitting procedure it was supposed to
audit.

This module makes that freeze mechanical rather than procedural:

* :class:`RobustnessGrid` enumerates every parameter point, feature-family
  ablation, and negative control **up front**, publishes a content-derived
  identity, and refuses to change afterwards.
* :func:`verify_frozen` compares an executing grid against the identity recorded
  before execution, so a grid edited mid-study is a detectable error rather than
  a silent one.
* Seed streams are **named and derived**, never drawn from a shared mutable
  generator. Trial ``k``'s randomness is a pure function of
  ``(root_seed, stream_name, index)``, so running trials in a different order,
  in parallel, or resuming after a crash reproduces identical draws.

That last property is what the issue calls candidate-order isolation, and it is
the difference between a study that can be re-run and one whose numbers depend on
the order somebody happened to iterate a dictionary.
"""

from __future__ import annotations

import hashlib
import itertools
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Final, Literal

import numpy as np

#: Refusal thresholds, not tuning knobs. A grid large enough to exhaust memory or
#: wall-clock is a specification mistake, and discovering it after six hours of
#: compute is the expensive way to find out.
MAX_GRID_POINTS: Final = 20_000
MAX_AXES: Final = 24
MAX_AXIS_VALUES: Final = 200
MAX_FAMILIES: Final = 64
MAX_NAME_CHARS: Final = 64

#: The negative controls the issue requires. Each answers a different question,
#: and a study that runs only one of them has not established much:
#:
#: - ``feature_permutation`` breaks the feature/label correspondence while keeping
#:   both marginal distributions, testing whether the model used the *pairing*.
#: - ``randomized_labels`` destroys the signal entirely, giving the null a
#:   distribution rather than a point, which is what a p-value needs.
#: - ``representation_placebo`` substitutes a structurally similar but
#:   information-free representation, testing whether the *representation* earned
#:   its keep as opposed to the pipeline around it.
ControlKind = Literal["feature_permutation", "randomized_labels", "representation_placebo"]

CONTROL_KINDS: Final[tuple[ControlKind, ...]] = (
    "feature_permutation",
    "randomized_labels",
    "representation_placebo",
)


class RobustnessGridError(ValueError):
    """Raised when a grid, axis, family, or seed request is unusable."""


def _name(value: object, *, field_name: str) -> str:
    """Return a bounded identifier suitable for a seed-stream name."""
    if not isinstance(value, str):
        raise RobustnessGridError(f"{field_name} must be a string, got {type(value).__name__}")
    text = value.strip()
    if not text or text != value:
        raise RobustnessGridError(f"{field_name} must be non-empty and free of padding")
    if len(text) > MAX_NAME_CHARS:
        raise RobustnessGridError(f"{field_name} exceeds {MAX_NAME_CHARS} characters")
    if not text.isascii() or not all(part.isalnum() or part in "._-" for part in text):
        raise RobustnessGridError(f"{field_name} must be ASCII alphanumeric with . _ - only")
    return text


def _json_scalar(value: object, *, field_name: str) -> Any:
    """Return a JSON-safe, hashable, finite scalar for a grid axis value."""
    if isinstance(value, bool) or value is None or isinstance(value, str):
        return value
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        if not np.isfinite(value):
            raise RobustnessGridError(f"{field_name} must be finite, got {value}")
        return float(value)
    raise RobustnessGridError(
        f"{field_name} must be a JSON scalar (str, int, float, bool, None), "
        f"got {type(value).__name__}"
    )


def canonical_digest(payload: Any) -> str:
    """Return a deterministic SHA-256 over a canonical JSON payload."""
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


@dataclass(frozen=True, slots=True)
class GridPoint:
    """One fully specified parameter combination."""

    point_id: str
    parameters: Mapping[str, Any]

    def __post_init__(self) -> None:
        object.__setattr__(self, "point_id", _name(self.point_id, field_name="point_id"))
        if not isinstance(self.parameters, Mapping) or not self.parameters:
            raise RobustnessGridError("a grid point must carry a non-empty parameter mapping")
        object.__setattr__(self, "parameters", dict(sorted(self.parameters.items())))

    @property
    def digest(self) -> str:
        """Content identity of this point's parameters."""
        return canonical_digest({"point": dict(self.parameters)})


@dataclass(frozen=True, slots=True)
class FeatureFamily:
    """A named group of features that is ablated as a unit.

    Ablating individual columns answers a question nobody asked: correlated
    columns substitute for one another, so dropping one changes nothing and the
    analysis concludes — wrongly — that the information was worthless. Families
    are the unit at which redundancy is actually detectable.
    """

    name: str
    members: tuple[str, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "name", _name(self.name, field_name="family name"))
        members = tuple(self.members)
        if not members:
            raise RobustnessGridError(f"family {self.name!r} must contain at least one member")
        if len(set(members)) != len(members):
            raise RobustnessGridError(f"family {self.name!r} contains duplicate members")
        for member in members:
            _name(member, field_name=f"family {self.name!r} member")
        object.__setattr__(self, "members", members)


@dataclass(frozen=True, slots=True)
class NegativeControl:
    """One frozen null control and the family it challenges.

    ``replicates`` is the number of independent draws. A single draw yields a
    point, not a distribution, and cannot support a p-value — so a replicate
    count below two is refused rather than silently producing a degenerate null.
    """

    name: str
    kind: ControlKind
    challenges: str
    replicates: int

    def __post_init__(self) -> None:
        object.__setattr__(self, "name", _name(self.name, field_name="control name"))
        object.__setattr__(self, "challenges", _name(self.challenges, field_name="challenges"))
        if self.kind not in CONTROL_KINDS:
            raise RobustnessGridError(
                f"unsupported control kind {self.kind!r}; expected one of {list(CONTROL_KINDS)}"
            )
        if isinstance(self.replicates, bool) or not isinstance(self.replicates, int):
            raise RobustnessGridError("replicates must be an int")
        if not 2 <= self.replicates <= 10_000:
            raise RobustnessGridError(
                "replicates must be in [2, 10000]; a single draw is a point, not a "
                "null distribution, and cannot support a p-value"
            )


@dataclass(frozen=True)
class RobustnessGrid:
    """The complete, frozen specification of a robustness study.

    Args:
        study_id: Stable identifier for this study.
        axes: Parameter name → ordered candidate values. Order is preserved
            because adjacency in a sweep is meaningful — sensitivity cliffs are
            defined between *neighbouring* values.
        families: Feature families available for ablation.
        controls: Frozen negative controls.
        root_seed: Root of every derived seed stream.
        metric_name: The single frozen metric every trial reports. Declared here
            so a study cannot quietly switch to whichever metric looks best.
        higher_is_better: Direction of that metric.

    Raises:
        RobustnessGridError: On any malformed or oversized specification.
    """

    study_id: str
    axes: Mapping[str, Sequence[Any]]
    families: tuple[FeatureFamily, ...] = ()
    controls: tuple[NegativeControl, ...] = ()
    root_seed: int = 20260802
    metric_name: str = "net_sharpe"
    higher_is_better: bool = True
    _points: tuple[GridPoint, ...] = field(default=(), init=False, repr=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "study_id", _name(self.study_id, field_name="study_id"))
        object.__setattr__(self, "metric_name", _name(self.metric_name, field_name="metric_name"))
        if isinstance(self.root_seed, bool) or not isinstance(self.root_seed, int):
            raise RobustnessGridError("root_seed must be an int")
        if not 0 <= self.root_seed < 2**32:
            raise RobustnessGridError("root_seed must be in [0, 2**32)")
        if not isinstance(self.higher_is_better, bool):
            raise RobustnessGridError("higher_is_better must be a bool")

        if not isinstance(self.axes, Mapping) or not self.axes:
            raise RobustnessGridError("a grid must declare at least one axis")
        if len(self.axes) > MAX_AXES:
            raise RobustnessGridError(f"grid exceeds the {MAX_AXES}-axis ceiling")
        cleaned: dict[str, tuple[Any, ...]] = {}
        for axis, values in self.axes.items():
            axis_name = _name(axis, field_name="axis name")
            if isinstance(values, (str, bytes)) or not isinstance(values, Sequence):
                raise RobustnessGridError(f"axis {axis_name!r} must be a sequence of values")
            ordered = tuple(
                _json_scalar(item, field_name=f"axis {axis_name!r} value") for item in values
            )
            if not ordered:
                raise RobustnessGridError(f"axis {axis_name!r} must offer at least one value")
            if len(ordered) > MAX_AXIS_VALUES:
                raise RobustnessGridError(
                    f"axis {axis_name!r} exceeds the {MAX_AXIS_VALUES}-value ceiling"
                )
            if len(set(map(repr, ordered))) != len(ordered):
                raise RobustnessGridError(f"axis {axis_name!r} contains duplicate values")
            cleaned[axis_name] = ordered
        object.__setattr__(self, "axes", dict(sorted(cleaned.items())))

        families = tuple(self.families)
        if len(families) > MAX_FAMILIES:
            raise RobustnessGridError(f"grid exceeds the {MAX_FAMILIES}-family ceiling")
        names = [family.name for family in families]
        if len(set(names)) != len(names):
            raise RobustnessGridError("feature family names must be unique")
        object.__setattr__(self, "families", families)

        controls = tuple(self.controls)
        control_names = [control.name for control in controls]
        if len(set(control_names)) != len(control_names):
            raise RobustnessGridError("negative control names must be unique")
        known = set(names) | {"candidate"}
        for control in controls:
            if control.challenges not in known:
                raise RobustnessGridError(
                    f"control {control.name!r} challenges unknown target "
                    f"{control.challenges!r}; expected a declared family or 'candidate'"
                )
        object.__setattr__(self, "controls", controls)

        total = 1
        for values in self.axes.values():
            total *= len(values)
            if total > MAX_GRID_POINTS:
                raise RobustnessGridError(
                    f"grid enumerates more than {MAX_GRID_POINTS} points; narrow the sweep "
                    "rather than discovering the cost after the compute is spent"
                )
        object.__setattr__(self, "_points", self._enumerate())

    def _enumerate(self) -> tuple[GridPoint, ...]:
        """Enumerate the full Cartesian product in a deterministic order."""
        axis_names = tuple(self.axes)
        combinations = itertools.product(*(self.axes[axis] for axis in axis_names))
        return tuple(
            GridPoint(
                point_id=f"p{index:05d}",
                parameters=dict(zip(axis_names, values, strict=True)),
            )
            for index, values in enumerate(combinations)
        )

    @property
    def points(self) -> tuple[GridPoint, ...]:
        """Every parameter combination, in deterministic order."""
        return self._points

    def __len__(self) -> int:
        return len(self._points)

    @property
    def identity(self) -> str:
        """Content identity frozen before execution.

        Covers axes, families, controls, seed, and metric. Two studies that
        differ in any of those are different studies, and the identity is what
        makes an accidental mid-study edit detectable.
        """
        return canonical_digest(
            {
                "study_id": self.study_id,
                "axes": {axis: list(values) for axis, values in self.axes.items()},
                "families": [
                    {"name": family.name, "members": list(family.members)}
                    for family in self.families
                ],
                "controls": [
                    {
                        "name": control.name,
                        "kind": control.kind,
                        "challenges": control.challenges,
                        "replicates": control.replicates,
                    }
                    for control in self.controls
                ],
                "root_seed": self.root_seed,
                "metric_name": self.metric_name,
                "higher_is_better": self.higher_is_better,
            }
        )

    def seed_stream(self, name: str, index: int = 0) -> np.random.Generator:
        """Return the named, derived generator for one trial.

        Derived from ``(root_seed, name, index)`` rather than drawn from a shared
        advancing generator. That is what makes trial ``k`` reproducible whether
        the study runs in order, in parallel, or resumes after a crash — and what
        stops a candidate and its own null control from sharing mutable state.
        """
        stream = _name(name, field_name="seed stream name")
        if isinstance(index, bool) or not isinstance(index, int) or index < 0:
            raise RobustnessGridError("seed stream index must be a non-negative int")
        label = int.from_bytes(hashlib.sha256(stream.encode("utf-8")).digest()[:4], "big")
        return np.random.default_rng(np.random.SeedSequence([self.root_seed, label, index]))

    def ablation_points(self) -> tuple[tuple[str, tuple[str, ...]], ...]:
        """Return each ablation as ``(label, retained families)``.

        Includes the ``full`` arm. A leave-one-family-out sweep without the
        complete book has no reference to be worse than.
        """
        names = tuple(family.name for family in self.families)
        arms: list[tuple[str, tuple[str, ...]]] = [("full", names)]
        for family in self.families:
            retained = tuple(item for item in names if item != family.name)
            arms.append((f"drop_{family.name}", retained))
        return tuple(arms)

    def to_dict(self) -> dict[str, Any]:
        """Return the JSON-safe frozen plan for the ledger and manifest."""
        return {
            "study_id": self.study_id,
            "identity": self.identity,
            "axes": {axis: list(values) for axis, values in self.axes.items()},
            "n_points": len(self._points),
            "families": [
                {"name": family.name, "members": list(family.members)} for family in self.families
            ],
            "ablations": [label for label, _ in self.ablation_points()],
            "controls": [
                {
                    "name": control.name,
                    "kind": control.kind,
                    "challenges": control.challenges,
                    "replicates": control.replicates,
                }
                for control in self.controls
            ],
            "root_seed": self.root_seed,
            "metric_name": self.metric_name,
            "higher_is_better": self.higher_is_better,
            "frozen_before_execution": True,
        }


def verify_frozen(grid: RobustnessGrid, expected_identity: str) -> None:
    """Refuse to execute a grid that differs from the one that was frozen.

    Raises:
        RobustnessGridError: If the identity does not match, naming both so the
            divergence can be diagnosed rather than merely reported.
    """
    if not isinstance(expected_identity, str) or len(expected_identity) != 64:
        raise RobustnessGridError("expected_identity must be a full SHA-256 digest")
    if grid.identity != expected_identity:
        raise RobustnessGridError(
            f"grid {grid.study_id!r} does not match the frozen plan: "
            f"executing {grid.identity[:12]}, frozen {expected_identity[:12]}. "
            "A grid edited after freezing invalidates the study's multiple-testing "
            "correction and its null distributions."
        )
