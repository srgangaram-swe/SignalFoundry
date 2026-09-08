# ADR 0006: Govern ensembles at a temporal-OOF prediction boundary

- Status: Accepted
- Date: 2026-07-26
- Decision owners: AlphaForge
- Scope: SF-S3-MR9, issue #35

## Context

The original `EnsembleModel` combined feature-level member models and estimated
optional IC weights from one chronological row tail. It had no stable identity
for the prediction panel, could split a cross-section inside a date, caught any
member exception and converted it to zero skill, and offered no way to prove
that stacking, dynamic weights, or regime gates were isolated from the final
holdout. Extending that class with more conditionals would preserve the wrong
trust boundary.

An ensemble is a second-stage learner. Its training data are expert
predictions—not the original feature matrix—and those predictions must already
be out of fold. Its final-evaluation API must not accept outcomes at all.

## Decision

Advanced ensembles cross two separate immutable contracts:

1. `TrainingOOFPanel` contains canonical
   `(date, symbol, fold_id, expert, prediction, uncertainty,
   regime_probability)` records plus one aligned target per
   `(date, symbol, fold_id)` and temporal fold attestations.
2. `InferenceBatch` contains only
   `(date, symbol, expert, prediction, uncertainty, regime_probability)`.
   It has no target field and carries the predeclared expected `(date, symbol)`
   key set. Unknown fields and unexpected keys fail validation.

The training panel enforces:

- every target row has exactly one prediction from every declared expert;
- expert, row, target, and fold identities are unique;
- `training_end < validation_start <= prediction_date <= validation_end`, with
  at least the declared number of intervening calendar embargo days;
- validation folds are ordered and disjoint;
- every prediction and fold ends before the frozen `holdout_start`;
- predictions, targets, uncertainties, and probabilities are finite and in
  their declared domains;
- candidate order is canonical; and
- prediction, date, expert, serialized-byte, and audit counts are bounded.

Canonical sorted-key JSON and SHA-256 identify panels, configurations, fitted
states, audit transitions, and decisions. State deserialization checks exact
fields, schema version, resource limits, semantic audit continuity, and the
embedded integrity digest before constructing a typed object. The unkeyed
digest is not authentication; provenance-sensitive callers compare it with an
independently trusted expected identity. No executable serialization format is
supported.

The six policies share this boundary:

- **Static blend:** validated non-negative weights normalized to one.
- **Rank/vote:** complete-date, within-expert cross-sectional percentile ranks
  are centered, averaged, and rescaled by training-target dispersion.
- **Temporal OOF stacking:** each diagnostic meta-fold is fit only from earlier
  OOF folds. The final frozen ridge state uses all training OOF rows, with
  means and scales fitted there and reused unchanged at holdout inference.
- **Bayesian averaging:** a stable log-evidence approximation from OOF residual
  variance is converted with shifted/clipped softmax and a feasible declared
  post-normalization floor \(f\), using
  \(w_j=f+(1-nf)\operatorname{softmax}_j(\ell)\).
- **Causal dynamic weighting:** the prediction for date \(t\) uses weights
  available before \(y_t\); only after that date's OOF outcomes are observed is
  the exponentially weighted loss state updated. The final OOF-trained state
  is frozen for holdout inference.
- **Regime-gated experts:** calm and stress weights are learned only from
  confident training-OOF regime rows. With zero confidence margin, threshold
  equality belongs only to stress, matching inference. A missing,
  inconsistent, or uncertain holdout regime abstains.

For stacking, with training-OOF expert matrix \(P\), column means \(\mu\),
scales \(s\), target \(y\), and penalty \(\lambda>0\):

\[
Z_{ij} = \frac{P_{ij}-\mu_j}{s_j},\qquad
\beta = (Z^\top Z+\lambda I)^\dagger Z^\top(y-\bar y).
\]

`numpy.linalg.lstsq` solves the regularized system. The positive ridge term
bounds collinearity; the condition number remains evidence rather than being
hidden. Constant columns receive scale one. A degenerate target creates an
explicit fallback state instead of meaningless learned weights.

For dynamic weighting, expert loss evolves as:

\[
L_{j,t}=\rho L_{j,t-1}+(1-\rho)
        \operatorname{mean}_{i\in t}(p_{ijt}-y_{it})^2,
\quad
w_{j,t+1}\propto\exp\left(-L_{j,t}/(\tau\,\operatorname{median}L_t)\right).
\]

The audit record stores before/after weights and `effective_after=t`, making
the causal ordering reviewable.

Every expected inference `(date, symbol)` group produces `EnsembleDecision`. A
missing expert or whole symbol, insufficient/incomplete rank date, uncertain
regime, or fallback fitted state yields
`status="abstained"`, a machine-readable reason, zero active weights, and the
declared fail-closed prediction. Exceptions are never converted into skill
scores or partial blends.

## Compatibility

The registry name `ensemble` remains available so existing
`configs/models.yaml` and walk-forward callers do not break. Its equal and
chronological IC modes now validate weights, propagate member failures, and
publish `weighting_status_` when insufficient rows require an explicit equal
weight fallback. It is a compatibility adapter, not the advanced ensemble
contract. Stacking, Bayesian, dynamic, and gated policies are configured and
run through `configs/ensemble_benchmark.yaml` and the governed OOF API.

## Alternatives considered

### Extend the feature-level `EnsembleModel`

Rejected. It cannot establish expert-prediction provenance, complete dates, or
a target-free holdout interface without becoming a second training framework
inside a model wrapper.

### Accept a permissive DataFrame

Rejected. Column typos, duplicate identities, partial dates, final targets, and
unbounded rows would become silent research-policy changes.

### Silently renormalize available experts

Rejected. Missing experts change the candidate and can make a favorable result
depend on operational failure. Abstention is observable and safer.

### Update dynamic weights on final-holdout outcomes

Rejected for this slice. Online adaptation may be a later pre-registered
evaluation protocol, but the frozen MR9 holdout must not mutate ensemble state.

## Consequences

Benefits:

- holdout labels are structurally excluded from ensemble inference;
- full-date and fold provenance are independently testable;
- candidate order, replay, and state transitions are deterministic;
- correlated experts are numerically stable and diagnostically visible; and
- fallback behavior is explicit, typed, and countable.

Costs:

- upstream experts must publish complete temporal OOF predictions and fold
  attestations;
- the immutable tuple representation uses more memory than an unvalidated
  matrix; resource ceilings bound that cost; and
- the boundary can verify dates and declared provenance but cannot prove that
  an external expert producer was honest. Independent upstream generation and
  lineage validation remain required.

## Security, privacy, and rollback

Inputs are untrusted until exact-schema validation completes. Serialization is
non-executable JSON with byte limits. No credentials, licensed observations,
model binaries, or row-level holdout outputs are committed. Evidence contains
only aggregate tables, state hashes, and a Seaborn image.

Rollback removes the governed runner/config/docs and stops calling
`fit_governed_ensemble`; the compatibility registry path and all prior evidence
remain readable. A rollback must not reinterpret already published state IDs.
