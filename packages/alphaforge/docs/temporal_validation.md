# Temporal validation contract

AlphaForge treats every validation split as a data-access policy, not an array
partition. The interval-aware planner in
`alphaforge/training/temporal_validation.py` assigns observed market sessions to
explicit training, validation, test, purge, embargo, overlap, and final-holdout
roles. It supports expanding and fixed-length rolling training histories.

The final holdout is never returned as a train, validation, or test sample. Its
dates appear only in metadata and visual evidence so a reviewer can verify the
boundary. Candidate and hyperparameter selection must use development
validation evidence only. A frozen candidate may enter the separate governed
final-holdout workflow once per immutable run identity.

## Interval mathematics

For sample \(i\), let \(s_i\) be its market-session date and \(e_i\) the last
future session required to compute its label. A sample assigned to a role whose
next protected boundary is \(b\) is eligible only when:

\[
e_i < b.
\]

This exact containment rule is stronger than assuming every target has the same
horizon. If any supplied asset or label on a session crosses the boundary, the
planner conservatively assigns that session to `overlap`. Explicit purge and
embargo gaps are counts of observed sessions, so holidays and unexpected
exchange closures do not become fabricated observations.

Purged K-fold and CPCV accept the same normalized label-event table. For every
contiguous test block \([a,b]\), a training event \([s_i,e_i]\) is removed when
it intersects the test information interval. CPCV applies this rule separately
to each selected test group; it does not incorrectly erase training groups
between non-adjacent test blocks.

## Public API and configuration

```python
from alphaforge.training import (
    TemporalValidationConfig,
    make_temporal_validation_plan,
)

config = TemporalValidationConfig(
    scheme="expanding",
    min_train_sessions=756,
    validation_sessions=126,
    test_sessions=126,
    step_sessions=126,
    purge_sessions=20,
    embargo_sessions=5,
    final_holdout_sessions=252,
)
folds = make_temporal_validation_plan(
    market_dates,
    config,
    label_events=label_dataset.events,
)
```

Configuration and input validation are bounded and fail closed. Session counts
must be valid, calendars must be finite and timezone-naive, event intervals must
satisfy `date < required_future_start <= required_future_end`, event dates must
belong to the supplied calendar, and every completed fold must retain non-empty
train, validation, and test roles.

`fold_assignments`, `fold_metadata`, and `temporal_plan_identity` provide the
machine-readable allocation, non-observation metadata, and SHA-256 identity for
run provenance. The identity changes when any role assignment changes.

## Reproducible evidence

Generate the committed synthetic reference evidence without network access:

```bash
make temporal-evidence OUTPUT=/absolute/path/to/new/evidence-directory
```

The command refuses to overwrite an existing directory and publishes
transactionally. It emits CSV assignments and role bounds, a canonical manifest,
and a Seaborn visualization. The plot contains only synthetic calendar roles—no
market observations, labels, predictions, or returns.

## Risks, rollback, and limitations

- Exact protection requires honest event intervals. Without them, only declared
  session gaps apply.
- A conservative per-date maximum event end may remove more observations when
  sparse assets or multiple targets have heterogeneous horizons. That reduces
  sample size instead of accepting ambiguous leakage.
- CPCV supplies distributional validation paths; it does not replace the
  chronological walk-forward backtest or justify tuning on a final holdout.
- No split method repairs survivorship, revision, adjustment, licensing, or
  point-in-time-universe deficiencies in the upstream dataset.

The feature is additive and configuration-gated. A normal revert removes the
new planner and exact-event options while preserving the established
walk-forward API. Weakening a boundary or reclassifying crossing samples as
eligible is not an acceptable rollback.
