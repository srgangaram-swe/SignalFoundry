# Parameter, feature, and stable-region robustness (SF-S4-MR6)

A frozen sweep, three negative controls under a leakage-safe temporal protocol,
and an analysis that reports **stable regions instead of a single optimum**.

> **No qualification claim.** This MR supplies the robustness machinery and its
> correctness evidence. Whether any candidate survives it is SF-S4-MR9's frozen
> decision. Simulation only.

---

## 1. Why the maximum is the wrong number

The argmax of a noisy sweep is the **least trustworthy point in it**. It is where
skill and luck happen to align, and by construction it is the point most likely to
regress. A setting whose neighbours also work is evidence of a real effect; a
spike surrounded by mediocrity is evidence of overfitting — and the two are
indistinguishable from the maximum alone.

The issue's non-goal says it directly: do not report only the best parameter
point. This module replaces that number with the three things that actually
predict out-of-sample survival.

## 2. The grid is frozen, mechanically

A robustness study is evidence only if its grid was fixed before anyone saw a
result. Otherwise "we swept the parameters" means "we searched until something
looked good", and the sweep becomes the fitting procedure it was meant to audit.

`RobustnessGrid` enumerates every parameter point, ablation arm, and negative
control up front and publishes a **content-derived identity** over axes,
families, controls, seed, and metric. `verify_frozen` compares the executing grid
against the identity recorded beforehand, so a mid-study edit is a detectable
error rather than a silent one — and the error message says why it matters: an
edited grid invalidates both the multiple-testing correction and the null
distributions.

The metric is frozen in the grid too, so a study cannot quietly switch to
whichever metric looks best.

**Ceilings are refusals, not tuning knobs.** A grid above 20 000 points is a
specification mistake, and discovering it after six hours of compute is the
expensive way to find out.

## 3. Named seed streams and candidate-order isolation

Every draw comes from `seed_stream(name, index)`, derived from
`(root_seed, name, index)` — never from a shared advancing generator.

That single choice buys three properties the issue requires:

- **Reproducibility** — replicate *k* is identical whether the study runs in
  order, in parallel, or resumes after a crash.
- **Candidate-order isolation** — no result depends on the order somebody
  happened to iterate a dictionary.
- **No shared mutable state** — a candidate and its own null control cannot draw
  the same randomness. `assert_streams_isolated` proves it, and its failure
  message states the consequence: a control sharing a candidate's randomness
  measures the candidate.

## 4. Three negative controls, because they fail differently

| control | what it destroys | what it therefore tests |
|---|---|---|
| `feature_permutation` | the feature/label pairing, keeping both marginals | whether the model used the *pairing* or just the shapes |
| `randomized_labels` | the signal entirely, repeatedly | gives the null a *distribution*, which a p-value needs |
| `representation_placebo` | the representation, keeping its moments and position | whether the *representation* earned its keep, or the surrounding pipeline did |

Two invariants are enforced structurally rather than by convention:

**Controls never see the holdout.** The control functions take only training-fold
data; there is no parameter through which holdout outcomes could arrive. A
control fit on the holdout would calibrate the null against the very data the
candidate is judged on.

**Permutation is within folds, never across them.** Shuffling globally would move
observations across the temporal boundary and leak future rows into a training
fold — producing a null that is *easier* than reality and a candidate that clears
it too easily.

The empirical p-value uses the conservative `(exceedances + 1) / (replicates + 1)`
correction. The uncorrected form can report exactly zero, which claims more
certainty than a finite number of draws can support, and a zero would also break
the downstream multiple-testing correction.

Per-control p-values are **uncorrected**. Family-wise correction is applied by
the governed ledger over the complete eligible family, and the record says so — a
candidate that clears an individual control has not thereby cleared the family.

## 5. The three analyses

**Stable regions** — contiguous axis runs whose every setting stays within
`tolerance` of the best. Aggregation across other axes is by **median**, not
mean: one catastrophic combination should not disqualify a sound setting, and one
spectacular combination should not rescue a bad one. `min_size` defaults to 2,
because a "region" of one point is exactly the spike this analysis exists to
distinguish from a plateau.

**Sensitivity cliffs** — adjacent settings between which the metric collapses.
Operating next to a cliff is materially different from operating on a plateau at
the same metric: the plateau tolerates the drift live operation guarantees, the
cliff edge does not, and the point estimate is identical in both cases. Only
deteriorating steps are reported; a sharp improvement is not a risk to the
operator standing on the good side of it.

**Family redundancy** — leave-one-family-out ablation. Families rather than
individual columns, because correlated columns substitute for one another: drop
one and nothing changes, and the analysis concludes — wrongly — that the
information was worthless. A partial ablation set is refused, since reporting only
the ablations somebody chose to run is selection by another name.

### A defect worth recording

The first implementation scaled relative drops by **each axis's own aggregate
spread**. On an axis with no effect that spread is pure noise, so a noise-sized
drop divided by a noise-sized denominator produced a large "relative drop" and a
confident cliff on a parameter that did nothing. It survived a
consistency-across-slices check, because the artefact is in the denominator
rather than the comparison.

All relative thresholds are now measured against the **global metric range**, so
"a 35% cliff" always means the same fraction of the performance range the study
actually spans. A regression test plants a surface with a real lookback plateau
and a real cost cliff but deliberately **no** `hold` effect, and asserts no cliff
is ever reported on `hold`.

## 6. Evidence

`tests/test_robustness_analysis.py` — 57 tests: grid enumeration and identity
sensitivity to every frozen input; freeze verification; ablation arms including
the full book; 10 specification-failure cases; the oversized-grid refusal;
duplicate family/control refusal; single-replicate refusal; seed-stream
reproducibility, name/index separation, order independence, and isolation; rerun
determinism; permutation staying inside folds and targeting one family;
label randomization preserving per-fold distributions; placebo moment matching
and constant-column handling; observation-count and unknown-column refusals;
conservative p-values in both directions; region and cliff detection on a planted
surface; the no-effect-axis regression; flat-surface behaviour; redundancy
detection; partial-ablation and partial-metric refusals; the non-finite-metric
refusal; and report determinism.

Coverage: `grid.py` 91%, `stability.py` 88%, `controls.py` 87% (branch).
Repository total 84.12% against an unchanged 78% floor.

## 7. Residual limitations

- **This MR covers *how* a candidate was configured, not *when* or *where* it was
  measured.** Calendar, regime, and universe robustness is SF-S4-MR7 — see
  [temporal, regime, and universe robustness](temporal_regime_robustness.md).
- **Redundancy is a claim about this grid and this metric.** A family that adds
  nothing on average may still matter in a regime the sweep did not contain, so
  the flag marks a candidate for removal, not a decision to remove it.
- **Median aggregation hides within-cell dispersion.** Two settings with equal
  medians can have very different tails; the report carries the metric
  distribution so that is visible, but the region logic itself does not use it.
- **Controls are only as strong as the evaluation function supplied to them.**
  This module deliberately knows nothing about models, portfolios, or costs —
  that separation is what stops a control from acquiring holdout access through a
  shared object — but it also means a caller can pass an evaluator that leaks.
- **No strategy is qualified here**, and the multiple-testing correction that
  makes a family-level claim lives in the governed ledger, not in this package.
- **Synthetic evidence only.**
