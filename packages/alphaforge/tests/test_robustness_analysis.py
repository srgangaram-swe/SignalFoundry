"""Tests for parameter, feature, and stable-region robustness (SF-S4-MR6).

Grouped by acceptance criterion. The invariants that carry the most weight:

* **The grid is frozen before execution** and a mid-study edit is detectable.
* **Controls never see the holdout and never share mutable random state** with
  the candidates they challenge.
* **Analysis reports regions, not the maximum** — and relative thresholds are
  measured against the global metric range so a no-effect axis cannot
  manufacture a cliff out of its own noise.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import pytest

from alphaforge.robustness import (
    CONTROL_KINDS,
    ControlOutcome,
    FeatureFamily,
    Fold,
    NegativeControl,
    NegativeControlError,
    RobustnessGrid,
    RobustnessGridError,
    StabilityError,
    assert_streams_isolated,
    contiguous_folds,
    family_redundancy,
    permute_within_folds,
    randomize_labels_within_folds,
    representation_placebo,
    run_control,
    sensitivity_cliffs,
    stability_report,
    stable_regions,
    verify_frozen,
    verify_rerun_determinism,
)

N_OBS = 240


def build_grid(**overrides: Any) -> RobustnessGrid:
    fields: dict[str, Any] = {
        "study_id": "s4mr6",
        "axes": {
            "lookback": [5, 10, 21, 63, 126],
            "hold": [1, 5, 21],
            "cost_bps": [1.0, 10.0, 50.0],
        },
        "families": (
            FeatureFamily(name="momentum", members=("mom_21", "mom_63")),
            FeatureFamily(name="volatility", members=("vol_21",)),
            FeatureFamily(name="noise", members=("rand_a",)),
        ),
        "controls": (
            NegativeControl(
                name="perm_mom", kind="feature_permutation", challenges="momentum", replicates=32
            ),
            NegativeControl(
                name="rand_labels", kind="randomized_labels", challenges="candidate", replicates=32
            ),
            NegativeControl(
                name="placebo_vol",
                kind="representation_placebo",
                challenges="volatility",
                replicates=32,
            ),
        ),
    }
    fields.update(overrides)
    return RobustnessGrid(**fields)


def planted_metrics(grid: RobustnessGrid, *, seed: int = 1) -> dict[str, float]:
    """A surface with a real lookback plateau and a real cost cliff, no hold effect."""
    rng = np.random.default_rng(seed)
    metrics: dict[str, float] = {}
    for point in grid.points:
        lookback = point.parameters["lookback"]
        cost = point.parameters["cost_bps"]
        value = 1.2 if lookback in (10, 21, 63) else 0.3
        value -= 0.9 if cost >= 50.0 else 0.0
        metrics[point.point_id] = value + 0.02 * rng.normal()
    return metrics


@pytest.fixture
def grid() -> RobustnessGrid:
    return build_grid()


@pytest.fixture
def features() -> pd.DataFrame:
    rng = np.random.default_rng(7)
    return pd.DataFrame(
        {
            "mom_21": rng.normal(size=N_OBS),
            "mom_63": rng.normal(size=N_OBS),
            "vol_21": np.abs(rng.normal(size=N_OBS)) + 0.1,
            "rand_a": rng.normal(size=N_OBS),
        }
    )


@pytest.fixture
def labels() -> pd.Series:
    return pd.Series(np.random.default_rng(8).normal(size=N_OBS), name="forward_return")


# ---------------------------------------------------------------------------
# AC1: the grid is frozen before execution
# ---------------------------------------------------------------------------


def test_grid_enumerates_the_full_product_deterministically(grid: RobustnessGrid) -> None:
    assert len(grid) == 5 * 3 * 3
    assert [point.point_id for point in grid.points] == [
        f"p{index:05d}" for index in range(len(grid))
    ]
    assert build_grid().identity == grid.identity


def test_identity_changes_with_every_frozen_input(grid: RobustnessGrid) -> None:
    assert len(grid.identity) == 64
    assert build_grid(root_seed=999).identity != grid.identity
    assert build_grid(metric_name="net_return").identity != grid.identity
    assert build_grid(higher_is_better=False).identity != grid.identity
    assert build_grid(axes={"lookback": [5, 10], "hold": [1], "cost_bps": [1.0]}).identity != (
        grid.identity
    )


def test_a_grid_edited_after_freezing_is_refused(grid: RobustnessGrid) -> None:
    """The whole point of freezing: an edit must be detectable, not silent."""
    verify_frozen(grid, grid.identity)
    edited = build_grid(axes={"lookback": [5, 10, 21], "hold": [1], "cost_bps": [1.0]})
    with pytest.raises(RobustnessGridError, match="does not match the frozen plan"):
        verify_frozen(edited, grid.identity)
    with pytest.raises(RobustnessGridError, match="full SHA-256"):
        verify_frozen(grid, "short")


def test_ablation_arms_include_the_full_book(grid: RobustnessGrid) -> None:
    """A leave-one-out sweep with no complete arm has nothing to be worse than."""
    labels_seen = [label for label, _ in grid.ablation_points()]
    assert labels_seen[0] == "full"
    assert set(labels_seen) == {"full", "drop_momentum", "drop_volatility", "drop_noise"}
    retained = dict(grid.ablation_points())
    assert set(retained["full"]) == {"momentum", "volatility", "noise"}
    assert "momentum" not in retained["drop_momentum"]


def test_plan_records_that_it_was_frozen(grid: RobustnessGrid) -> None:
    plan = grid.to_dict()
    assert plan["frozen_before_execution"] is True
    assert plan["n_points"] == len(grid)
    assert plan["identity"] == grid.identity
    assert len(plan["controls"]) == 3


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"axes": {}}, "at least one axis"),
        ({"axes": {"a": []}}, "at least one value"),
        ({"axes": {"a": [1, 1]}}, "duplicate values"),
        ({"axes": {"a": [float("nan")]}}, "finite"),
        ({"axes": {"a": [object()]}}, "JSON scalar"),
        ({"axes": {"bad name": [1]}}, "axis name"),
        ({"root_seed": -1}, "root_seed"),
        ({"root_seed": True}, "root_seed must be an int"),
        ({"higher_is_better": "yes"}, "higher_is_better"),
        ({"study_id": ""}, "study_id"),
    ],
)
def test_grid_specification_fails_closed(overrides: dict[str, Any], message: str) -> None:
    with pytest.raises(RobustnessGridError, match=message):
        build_grid(**overrides)


def test_oversized_grid_is_refused_before_the_compute_is_spent() -> None:
    with pytest.raises(RobustnessGridError, match="narrow the sweep"):
        build_grid(axes={f"axis_{index}": list(range(20)) for index in range(6)})


def test_duplicate_families_and_controls_are_refused() -> None:
    duplicate = (
        FeatureFamily(name="a", members=("x",)),
        FeatureFamily(name="a", members=("y",)),
    )
    with pytest.raises(RobustnessGridError, match="family names must be unique"):
        build_grid(families=duplicate, controls=())
    controls = (
        NegativeControl(name="c", kind="randomized_labels", challenges="candidate", replicates=4),
        NegativeControl(name="c", kind="randomized_labels", challenges="candidate", replicates=4),
    )
    with pytest.raises(RobustnessGridError, match="control names must be unique"):
        build_grid(controls=controls)


def test_control_must_challenge_a_declared_target() -> None:
    with pytest.raises(RobustnessGridError, match="challenges unknown target"):
        build_grid(
            controls=(
                NegativeControl(
                    name="c", kind="randomized_labels", challenges="absent", replicates=4
                ),
            )
        )


def test_a_single_replicate_control_is_refused() -> None:
    """One draw is a point, not a null distribution."""
    with pytest.raises(RobustnessGridError, match="a point, not a null distribution"):
        NegativeControl(name="c", kind="randomized_labels", challenges="candidate", replicates=1)


def test_unsupported_control_kind_is_refused() -> None:
    with pytest.raises(RobustnessGridError, match="unsupported control kind"):
        NegativeControl(name="c", kind="vibes", challenges="candidate", replicates=4)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# AC3: seed streams, candidate-order isolation, rerun determinism
# ---------------------------------------------------------------------------


def test_seed_streams_are_named_derived_and_reproducible(grid: RobustnessGrid) -> None:
    first = grid.seed_stream("perm_mom", 3).random(5)
    second = grid.seed_stream("perm_mom", 3).random(5)
    np.testing.assert_array_equal(first, second)


def test_seed_streams_differ_by_name_and_by_index(grid: RobustnessGrid) -> None:
    """A control sharing a candidate's randomness measures the candidate."""
    base = grid.seed_stream("perm_mom", 0).random(5)
    assert not np.array_equal(base, grid.seed_stream("rand_labels", 0).random(5))
    assert not np.array_equal(base, grid.seed_stream("perm_mom", 1).random(5))
    assert_streams_isolated(grid, ["perm_mom", "rand_labels", "placebo_vol", "candidate"])


def test_stream_order_does_not_affect_draws(grid: RobustnessGrid) -> None:
    """Candidate-order isolation: replicate k is identical however you get to it."""
    forward = [grid.seed_stream("perm_mom", index).random(3) for index in range(4)]
    backward = [grid.seed_stream("perm_mom", index).random(3) for index in reversed(range(4))]
    for index, expected in enumerate(forward):
        np.testing.assert_array_equal(expected, backward[3 - index])


def test_a_different_root_seed_changes_every_stream(grid: RobustnessGrid) -> None:
    other = build_grid(root_seed=4242)
    assert not np.array_equal(
        grid.seed_stream("perm_mom", 0).random(5), other.seed_stream("perm_mom", 0).random(5)
    )


def test_invalid_stream_requests_are_refused(grid: RobustnessGrid) -> None:
    with pytest.raises(RobustnessGridError, match="index"):
        grid.seed_stream("perm_mom", -1)
    with pytest.raises(RobustnessGridError, match="seed stream name"):
        grid.seed_stream("bad name", 0)


def test_rerun_determinism_is_verified(grid: RobustnessGrid) -> None:
    metrics = planted_metrics(grid)
    verify_rerun_determinism(metrics, planted_metrics(grid))
    drifted = dict(metrics)
    drifted["p00000"] += 1e-12
    with pytest.raises(RobustnessGridError, match="rerun diverged"):
        verify_rerun_determinism(metrics, drifted)
    with pytest.raises(RobustnessGridError, match="different point sets"):
        verify_rerun_determinism(metrics, {"p00000": 1.0})


# ---------------------------------------------------------------------------
# AC3: negative controls are leakage-safe
# ---------------------------------------------------------------------------


def test_permutation_stays_inside_each_fold(features: pd.DataFrame) -> None:
    """Global shuffling would move rows across the temporal boundary."""
    folds = contiguous_folds(len(features), n_folds=4)
    generator = np.random.default_rng(0)
    permuted = permute_within_folds(features, folds, generator=generator)
    for fold in folds:
        original = features.iloc[fold.start : fold.stop]
        shuffled = permuted.iloc[fold.start : fold.stop]
        # The multiset of rows inside a fold is preserved; only the order moves.
        np.testing.assert_allclose(
            np.sort(original["mom_21"].to_numpy()), np.sort(shuffled["mom_21"].to_numpy())
        )
    assert not np.allclose(features["mom_21"].to_numpy(), permuted["mom_21"].to_numpy())


def test_permutation_can_target_one_family(features: pd.DataFrame) -> None:
    folds = contiguous_folds(len(features), n_folds=3)
    permuted = permute_within_folds(
        features, folds, generator=np.random.default_rng(0), columns=["mom_21", "mom_63"]
    )
    np.testing.assert_array_equal(features["vol_21"].to_numpy(), permuted["vol_21"].to_numpy())
    assert not np.allclose(features["mom_21"].to_numpy(), permuted["mom_21"].to_numpy())


def test_randomized_labels_preserve_each_fold_distribution(labels: pd.Series) -> None:
    folds = contiguous_folds(len(labels), n_folds=4)
    randomized = randomize_labels_within_folds(labels, folds, generator=np.random.default_rng(0))
    for fold in folds:
        original = labels.iloc[fold.start : fold.stop].to_numpy()
        shuffled = randomized.iloc[fold.start : fold.stop].to_numpy()
        np.testing.assert_allclose(np.sort(original), np.sort(shuffled))
    assert not np.allclose(labels.to_numpy(), randomized.to_numpy())


def test_representation_placebo_matches_moments_but_not_content(
    features: pd.DataFrame,
) -> None:
    placebo = representation_placebo(
        features, generator=np.random.default_rng(0), columns=["vol_21"]
    )
    original = features["vol_21"].to_numpy()
    surrogate = placebo["vol_21"].to_numpy()
    assert abs(surrogate.mean() - original.mean()) < 0.3 * original.std()
    assert abs(surrogate.std() - original.std()) < 0.4 * original.std()
    assert abs(np.corrcoef(original, surrogate)[0, 1]) < 0.3
    np.testing.assert_array_equal(features["mom_21"].to_numpy(), placebo["mom_21"].to_numpy())


def test_placebo_of_a_constant_column_does_not_invent_variance(
    features: pd.DataFrame,
) -> None:
    constant = features.assign(vol_21=1.0)
    placebo = representation_placebo(
        constant, generator=np.random.default_rng(0), columns=["vol_21"]
    )
    np.testing.assert_allclose(placebo["vol_21"].to_numpy(), 1.0)


def test_controls_refuse_too_few_observations() -> None:
    tiny = pd.DataFrame({"a": np.arange(5.0)})
    with pytest.raises(NegativeControlError, match="at least 30 observations"):
        permute_within_folds(
            tiny, contiguous_folds(5, n_folds=1), generator=np.random.default_rng(0)
        )
    with pytest.raises(NegativeControlError, match="at least 30 observations"):
        randomize_labels_within_folds(
            pd.Series(np.arange(5.0)),
            contiguous_folds(5, n_folds=1),
            generator=np.random.default_rng(0),
        )


def test_controls_refuse_unknown_columns_and_empty_folds(features: pd.DataFrame) -> None:
    folds = contiguous_folds(len(features), n_folds=2)
    with pytest.raises(NegativeControlError, match="unknown columns"):
        permute_within_folds(
            features, folds, generator=np.random.default_rng(0), columns=["absent"]
        )
    with pytest.raises(NegativeControlError, match="unknown columns for placebo"):
        representation_placebo(features, generator=np.random.default_rng(0), columns=["absent"])
    with pytest.raises(NegativeControlError, match="at least one fold"):
        permute_within_folds(features, [], generator=np.random.default_rng(0))
    with pytest.raises(NegativeControlError, match="extends past"):
        permute_within_folds(
            features,
            [Fold(name="f", start=0, stop=len(features) + 5)],
            generator=np.random.default_rng(0),
        )


def test_folds_are_contiguous_and_ordered() -> None:
    folds = contiguous_folds(100, n_folds=4)
    assert len(folds) == 4
    for earlier, later in zip(folds, folds[1:], strict=False):
        assert earlier.stop == later.start
    with pytest.raises(NegativeControlError, match="positive int"):
        contiguous_folds(100, n_folds=0)
    with pytest.raises(NegativeControlError, match="fewer observations"):
        contiguous_folds(3, n_folds=10)
    with pytest.raises(NegativeControlError, match="empty or inverted"):
        Fold(name="bad", start=5, stop=5)


# ---------------------------------------------------------------------------
# AC2/AC3: candidates versus their frozen nulls
# ---------------------------------------------------------------------------


def test_control_outcome_reports_a_conservative_p_value() -> None:
    outcome = ControlOutcome(
        control="rand_labels",
        kind="randomized_labels",
        challenges="candidate",
        replicate_metrics=tuple(np.linspace(-0.2, 0.2, 40)),
        candidate_metric=1.5,
        higher_is_better=True,
    )
    # Nothing in the null reaches the candidate, but the p-value is never zero:
    # a finite number of draws cannot support that much certainty.
    assert outcome.exceedances == 0
    assert outcome.p_value == pytest.approx(1 / 41)
    assert outcome.p_value > 0.0
    assert outcome.exceeds_null(0.05)


def test_a_candidate_inside_its_null_does_not_exceed_it() -> None:
    outcome = ControlOutcome(
        control="perm_mom",
        kind="feature_permutation",
        challenges="momentum",
        replicate_metrics=tuple(np.linspace(-1.0, 1.0, 40)),
        candidate_metric=0.0,
        higher_is_better=True,
    )
    assert not outcome.exceeds_null(0.05)
    assert outcome.p_value > 0.05


def test_control_outcome_validates_its_inputs() -> None:
    with pytest.raises(NegativeControlError, match="at least two replicates"):
        ControlOutcome(
            control="c",
            kind="randomized_labels",
            challenges="candidate",
            replicate_metrics=(0.1,),
            candidate_metric=1.0,
            higher_is_better=True,
        )
    with pytest.raises(NegativeControlError, match="must be finite"):
        ControlOutcome(
            control="c",
            kind="randomized_labels",
            challenges="candidate",
            replicate_metrics=(0.1, float("nan")),
            candidate_metric=1.0,
            higher_is_better=True,
        )


def test_run_control_uses_isolated_streams_and_is_reproducible(grid: RobustnessGrid) -> None:
    control = grid.controls[1]
    seen: list[float] = []

    def evaluate(kind: str, generator: np.random.Generator, replicate: int) -> float:
        assert kind in CONTROL_KINDS
        value = float(generator.normal())
        seen.append(value)
        return value

    first = run_control(control, grid, candidate_metric=3.0, evaluate=evaluate)
    second = run_control(control, grid, candidate_metric=3.0, evaluate=evaluate)
    assert first.replicate_metrics == second.replicate_metrics
    assert len(first.replicate_metrics) == control.replicates


def test_run_control_refuses_a_non_finite_or_non_numeric_metric(grid: RobustnessGrid) -> None:
    control = grid.controls[1]
    with pytest.raises(NegativeControlError, match="non-finite"):
        run_control(
            control,
            grid,
            candidate_metric=1.0,
            evaluate=lambda kind, generator, replicate: float("nan"),
        )
    with pytest.raises(NegativeControlError, match="expected a real metric"):
        run_control(
            control,
            grid,
            candidate_metric=1.0,
            evaluate=lambda kind, generator, replicate: "nope",
        )


# ---------------------------------------------------------------------------
# AC2: regions, cliffs, redundancy — not a single optimum
# ---------------------------------------------------------------------------


def test_stable_regions_find_the_planted_plateau(grid: RobustnessGrid) -> None:
    regions = {
        region.axis: list(region.values) for region in stable_regions(grid, planted_metrics(grid))
    }
    assert regions["lookback"] == [10, 21, 63]
    assert regions["cost_bps"] == [1.0, 10.0]


def test_an_axis_with_no_effect_is_wholly_stable(grid: RobustnessGrid) -> None:
    """`hold` has no planted effect, so every setting must qualify."""
    regions = {
        region.axis: list(region.values) for region in stable_regions(grid, planted_metrics(grid))
    }
    assert regions["hold"] == [1, 5, 21]


def test_cliffs_find_the_planted_edges(grid: RobustnessGrid) -> None:
    cliffs = sensitivity_cliffs(grid, planted_metrics(grid))
    edges = {(cliff.axis, cliff.from_value, cliff.to_value) for cliff in cliffs}
    assert ("cost_bps", 10.0, 50.0) in edges
    assert ("lookback", 63, 126) in edges


def test_a_no_effect_axis_cannot_manufacture_a_cliff(grid: RobustnessGrid) -> None:
    """Regression: relative drops are scaled by the global metric range.

    Dividing a noise-sized drop by an axis's own noise-sized aggregate spread
    produced a spurious cliff on `hold`, which has no planted effect at all.
    """
    cliffs = sensitivity_cliffs(grid, planted_metrics(grid))
    assert all(cliff.axis != "hold" for cliff in cliffs), [c.axis for c in cliffs]


def test_a_flat_surface_has_no_cliffs_and_one_region(grid: RobustnessGrid) -> None:
    flat = {point.point_id: 1.0 for point in grid.points}
    assert sensitivity_cliffs(grid, flat) == ()
    for region in stable_regions(grid, flat):
        assert region.size == len(grid.axes[region.axis])


def test_redundant_family_is_identified(grid: RobustnessGrid) -> None:
    contributions = family_redundancy(
        grid,
        {"full": 1.20, "drop_momentum": 0.55, "drop_volatility": 1.05, "drop_noise": 1.199},
    )
    by_name = {item.family: item for item in contributions}
    assert by_name["noise"].redundant
    assert not by_name["momentum"].redundant
    assert by_name["momentum"].contribution > by_name["volatility"].contribution


def test_partial_ablation_sets_are_refused(grid: RobustnessGrid) -> None:
    """Reporting only the ablations somebody chose is selection by another name."""
    with pytest.raises(StabilityError, match="selection by another name"):
        family_redundancy(grid, {"full": 1.0, "drop_momentum": 0.5})


def test_metrics_must_cover_exactly_the_frozen_grid(grid: RobustnessGrid) -> None:
    metrics = planted_metrics(grid)
    partial = dict(list(metrics.items())[:-1])
    with pytest.raises(StabilityError, match="must cover exactly the frozen grid"):
        stable_regions(grid, partial)
    extra = {**metrics, "p99999": 1.0}
    with pytest.raises(StabilityError, match="must cover exactly the frozen grid"):
        stable_regions(grid, extra)


def test_a_failed_trial_cannot_be_smuggled_in_as_a_number(grid: RobustnessGrid) -> None:
    metrics = planted_metrics(grid)
    metrics["p00000"] = float("nan")
    with pytest.raises(StabilityError, match="recorded as a failure"):
        stable_regions(grid, metrics)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"tolerance": 0.0}, "tolerance"),
        ({"tolerance": 1.0}, "tolerance"),
        ({"min_size": 0}, "min_size"),
    ],
)
def test_stable_region_bounds(grid: RobustnessGrid, kwargs: dict, message: str) -> None:
    with pytest.raises(StabilityError, match=message):
        stable_regions(grid, planted_metrics(grid), **kwargs)


@pytest.mark.parametrize("kwargs", [{"threshold": 0.0}, {"threshold": 1.0}, {"consistency": 0.0}])
def test_cliff_bounds(grid: RobustnessGrid, kwargs: dict) -> None:
    with pytest.raises(StabilityError):
        sensitivity_cliffs(grid, planted_metrics(grid), **kwargs)


# ---------------------------------------------------------------------------
# The assembled record
# ---------------------------------------------------------------------------


def test_report_states_that_the_maximum_is_untrustworthy(grid: RobustnessGrid) -> None:
    """The caveat must travel with the data, not live only in a document."""
    report = stability_report(grid, planted_metrics(grid))
    assert "least trustworthy" in report["interpretation"]
    assert "best_point" in report
    assert report["stable_regions"]
    assert report["grid_identity"] == grid.identity


def test_report_carries_ablations_and_controls(grid: RobustnessGrid) -> None:
    outcome = ControlOutcome(
        control="rand_labels",
        kind="randomized_labels",
        challenges="candidate",
        replicate_metrics=tuple(np.linspace(-0.2, 0.2, 40)),
        candidate_metric=1.2,
        higher_is_better=True,
    )
    report = stability_report(
        grid,
        planted_metrics(grid),
        ablation_metrics={
            "full": 1.20,
            "drop_momentum": 0.55,
            "drop_volatility": 1.05,
            "drop_noise": 1.199,
        },
        control_outcomes=[outcome],
    )
    assert report["redundant_families"] == ["noise"]
    assert report["controls_all_exceeded"] is True
    assert "uncorrected" in report["control_note"]
    assert "family-wise correction" in report["control_note"].lower()


def test_report_is_deterministic(grid: RobustnessGrid) -> None:
    metrics = planted_metrics(grid)
    assert stability_report(grid, metrics) == stability_report(grid, metrics)
