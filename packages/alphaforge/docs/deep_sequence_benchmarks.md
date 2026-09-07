# Controlled deep-sequence benchmarks

SF-S3-MR6 adds an opt-in family for comparing five causal sequence
architectures against the frozen Sprint 2 LightGBM reference. It is an
experimental-control capability, not a claim of market edge or trading
readiness.

## Public contract

The registry names are `sequence_cnn`, `sequence_tcn`, `sequence_lstm`,
`sequence_gru`, and `sequence_transformer`. All accept the same bounded
`SequenceBenchmarkConfig` fields and the standard `AlphaModel.fit/predict`
surface. Inputs use a unique two-level `(date, symbol)` index. Output retains
one prediction per input row; rows without the declared minimum history receive
the documented zero fallback.

`build_causal_windows` sorts internally by symbol and date, left-pads short
histories, publishes a boolean validity mask, and maps every output to the
original row position. Tests prove that future mutations cannot affect past
predictions, one symbol cannot contaminate another, shuffled input and
candidate order do not alter results, and padding cannot update hidden state.

## Equivalent experiment policy

Every variant shares:

1. the same outer walk-forward folds and label embargo;
2. inner chronological validation;
3. feature and target scaling fitted only before the inner-validation start;
4. Huber loss, AdamW, gradient clipping, seeded batch permutations, patience,
   and best-state restoration;
5. bounded sequence length, history, windows, tensor bytes, parameters, layers,
   batch, and epochs; and
6. CPU reference execution unless a named accelerator is explicitly selected.

`resource_evidence()` returns parameters, parameter bytes, wall and process
CPU time, accelerator peak allocation, epochs, best epoch, validation loss,
and full loss histories. The walk-forward metrics table carries the aggregate
resource fields with backend and termination status. CPU peak memory is not
currently sampled, so `peak_device_bytes=0` means “not an accelerator
measurement,” not zero host memory.

`evaluate_oos_predictions` consumes already out-of-sample
`date/symbol/target/prediction` rows. It reports RMSE, MAE, Pearson
correlation, per-date rank IC, descriptive regression-calibration
intercept/slope/RMSE, turnover, and gross/net mean daily return after a declared
basis-point cost. The long-short diagnostic is deliberately simpler than the
event-driven execution simulator and must not be presented as execution
evidence.

## Example

```python
from alphaforge.models import create_model

model = create_model(
    "sequence_tcn",
    seq_len=32,
    hidden_size=32,
    n_layers=2,
    max_epochs=30,
    patience=5,
    max_parameters=2_000_000,
    device="cpu",
    seed=42,
)
model.fit(training_features, training_returns)
prediction = model.predict(test_features)
resources = model.resource_evidence()
```

The training and test matrices must retain their per-symbol history. In the
current walk-forward driver, early rows of a test fold do not borrow hidden
history from the training fold; they remain the documented zero fallback until
the test matrix contains enough causal history. This conservative behavior
avoids state leakage but reduces usable OOS rows and remains a limitation.

## Validation and comparison

Run:

```bash
.venv/bin/python -m pytest tests/test_deep_sequence.py
.venv/bin/python -m pytest -m "not network"
```

A serious study must pre-register folds, feature and label versions, model
parameters, cost policy, compute budget, baseline, ablations, rejection rules,
and the final untouched interval before fitting. Compare every candidate with
LightGBM on identical OOS rows. Report failed candidates and unfavorable
metrics. Do not tune sequence length, width, depth, or the number of attempts
against the protected holdout.

The bounded reference run and its unfavorable as well as favorable outcomes
are documented in the [Sprint 3 deep-sequence report](sprint_3_deep_sequence_report.md).

The public WIKI bootstrap is suitable for engineering validation only. It is
stale, current-vintage, lacks point-in-time membership and complete corporate
actions, and carries survivorship and selection bias. Results from it cannot
qualify paper or live readiness. Licensed point-in-time data, realistic
execution and financing, multiplicity correction, regime/capacity stability,
paper-trading duration, risk controls, and explicit owner approval remain
required before capital is considered.
