# Calibration and uncertainty contracts

SF-S2-MR6 adds conservative baseline tools for probability calibration and
prediction uncertainty. They are research-evidence components, not guarantees
of future coverage or profitability.

## Trust boundary

`OOFProbabilityData` and `OOFRegressionData` accept only time-ordered,
training-derived out-of-fold observations with at least two fold identities.
They intentionally cannot represent an in-sample fit or a final-holdout fit.
Probability comparison and block-conformal application require dates strictly
after the fitted OOF period.

This type boundary prevents accidental leakage through the supported API. It
does not independently reconstruct how a caller produced each prediction.
Run manifests must therefore retain the validation-plan identity and fold
assignments that support the `training_oof` declaration.

## Probability calibration

The probability module supplies:

- Brier score, fixed-width reliability tables, and count-weighted expected
  calibration error (ECE);
- Platt scaling on clipped raw-probability log-odds;
- monotone isotonic calibration;
- immutable metadata containing method, fit period, sample count, fold count,
  prevalence, and limitations; and
- atomic, bounded, versioned JSON persistence with no executable object
  deserialization.

Reliability tables retain empty bins explicitly. Probabilities outside
`[0, 1]`, non-binary outcomes, non-finite values, constant scores, single-class
outcomes, insufficient samples, and malformed artifacts fail closed.

`compare_calibration` evaluates only a post-fit period. It publishes raw and
calibrated Brier/ECE values plus a paired moving-block bootstrap interval for
the Brier improvement. A positive estimate is evidence for that period; it is
not a claim that calibration will remain beneficial.

## Dependence-aware intervals

`BlockBootstrapConfig` fixes the resample count, contiguous block length,
confidence level, circular-boundary policy, and NumPy generator seed. The
implementation resamples ordered contiguous blocks and reports its stationarity
and block-length assumptions. It does not expose an IID bootstrap mode.

`BlockConformalInterval` groups absolute OOF residuals into non-overlapping
temporal blocks and calibrates on the block maxima using the conservative
finite-sample quantile. Its interval requires:

1. OOF residuals from multiple temporal folds;
2. enough complete blocks under the predeclared block length;
3. application after the fit period; and
4. residual-block exchangeability.

The final assumption is strong in non-stationary markets. Coverage can fail
under drift, regime change, or changed model/feature policy, so every fitted
artifact records that limitation.

## Quantile regression

`QuantileRegressionInterval` fits bounded linear lower, median, and upper
quantile regressions on a declared training fold. It records fit dates, feature
schema, sample/feature counts, regularization, solver iterations, and
limitations. Non-finite inputs, changed schemas, oversized work, and quantile
crossing fail closed. Crossing is never hidden by sorting the outputs.

The model uses deterministic HiGHS optimization and persists only numeric
coefficients and metadata in the same bounded JSON format. Quantile regression
alone does not provide finite-sample coverage; block conformal is the more
conservative residual baseline when its assumptions are defensible.

## Configuration

`configs/calibration.yaml` is the strict predeclared policy. Unknown keys,
unbounded work, invalid probabilities, and inconsistent bootstrap/conformal
block lengths fail during `make config-check`.

No API key, licensed observation, prediction panel, trained artifact, or final
holdout is committed. The focused tests use deterministic synthetic fixtures.

## Rollback and operational limitations

Rollback removes the new evaluation exports and calibration configuration; it
does not change the model contract or existing walk-forward results. These
tools do not authorize paper or live trading. Before capital use, calibration
and coverage must be re-evaluated on licensed point-in-time data across regimes,
with drift monitoring and explicit failure actions.
