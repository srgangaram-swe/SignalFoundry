# Signal Foundry Sprint 3 — governed ensemble engineering report

## Outcome

SF-S3-MR9 now has one typed temporal-OOF boundary for static blends, rank/vote,
ridge stacking, Bayesian averaging, causal dynamic weighting, and
regime-gated experts. The final-holdout inference type has no target field.
Missing experts and uncertain regimes publish explicit abstentions, while
panel, configuration, state, audit, and decision identities are deterministic.

This report covers synthetic contract/algorithm evidence only. It does not
accept or reject an ensemble for market use.

## Frozen reference

`configs/ensemble_benchmark.yaml` fixes generator seed 3519, bootstrap seed
97531, 96 training dates, four temporal OOF folds, 32 untouched synthetic
holdout dates, eight symbols, four experts, six ensemble policies, a 5 bps
turnover diagnostic, and bounded record and audit budgets. Named generator
child streams and their SHA-256-derived seeds are recorded in `summary.json`.
The in-memory training boundary contained 3,072 expert OOF rows. The holdout
contained 256 `(date, symbol)` outcomes, evaluated only after all six fitted
states were frozen.

No row prediction or target is published. The aggregate evidence is under
[`docs/evidence/signal_foundry_sprint_3/ensembles`](evidence/signal_foundry_sprint_3/ensembles/README.md).

## Aggregate observations

The best single synthetic expert was `stable`, with MSE `4.0978e-05`. All six
policy point estimates were lower on this deliberately complementary draw.
The predeclared paired 95% moving-block interval remained below zero for
regime gating, stacking, dynamic weighting, static blending, and Bayesian
averaging; rank/vote crossed zero. Bayesian's `1.66e-10` delta is practically
negligible even though the deterministic paired interval excludes zero.

| policy | MSE | delta vs best single (paired 95% interval) | within-date rank IC (valid dates) | fallback rate |
|---|---:|---:|---:|---:|
| regime gate | 1.9236e-05 | -2.1742e-05 [-2.8641e-05, -1.4215e-05] | 0.343 (31/32) | 3.125% |
| stacking | 2.1328e-05 | -1.9651e-05 [-2.3788e-05, -1.4664e-05] | 0.254 (32/32) | 0% |
| dynamic | 2.5710e-05 | -1.5268e-05 [-2.0959e-05, -9.3169e-06] | 0.240 (32/32) | 0% |
| static | 2.8468e-05 | -1.2510e-05 [-2.1224e-05, -2.6350e-06] | 0.209 (32/32) | 0% |
| rank/vote | 3.5647e-05 | -5.3309e-06 [-1.3366e-05, 1.9125e-06] | 0.294 (32/32) | 0% |
| Bayesian | 4.0978e-05 | -1.6577e-10 [-2.0368e-10, -1.2487e-10] | 0.196 (32/32) | 0% |

Intervals use 512 circular moving-block resamples of five complete dates at
bootstrap seed 97531. CSVs publish confidence bounds, standard errors,
variances, sample counts, and the full resampling policy. These intervals
describe this frozen synthetic holdout; they do not cover generator choice,
candidate selection, regime uncertainty, or future data.

The regime gate's lower error is expected because the synthetic reference
deliberately gives different experts complementary calm/stress noise. Its
3.125% fallback rate is equally important: ambiguous probabilities abstained
instead of being forced through the better-looking gate. This is machinery
recovery evidence, not a market-regime discovery.

The drop-one tables show positive mean marginal MSE contribution for all six
policies, but prediction/error correlations remain material. The stacking
regularized system condition number was approximately 796; the state records
it rather than presenting correlated experts as independent information.
Heuristic dispersion hit rates ranged from 76.2% to 91.4% for non-gated
ensembles; the regime gate reached 85.1% over combined rows. These are not
one-sigma coverage claims: the dispersion combines supplied expert scales,
training-OOF residual MSE, and disagreement and is neither calibrated nor
like-for-like with a single expert's noise scale.

The simplified 5 bps diagnostic reduced mean signed synthetic return by
`1.62e-04` to `2.51e-04` per date across ensembles. Regime gating had the
lowest measured turnover (0.324) and rank/vote had the weakest net directional
result despite improving MSE. These disagreements are why the report does not
select a winner from one metric.

## Verification

Focused tests cover:

- exact folds, calendar embargo, complete expected `(date, symbol)` keys,
  holdout exclusion, numeric date rejection, and inference target rejection;
- mathematical ridge agreement and correlated-column stability;
- first-fold stacking fallback and strictly prior meta-fold inputs;
- update-after-observation dynamic causality;
- missing experts and whole symbols, one-symbol/incomplete rank dates,
  uncertain/missing/equality-boundary regimes, degenerate targets, infeasible
  weight floors, audit ceilings, and structured numeric overflow;
- deterministic state/panel serialization, tamper rejection, bounded bytes,
  immutable records, and candidate-order isolation; and
- strict direct/config-file construction, atomic/no-overwrite publication,
  complete artifact inventory, exact aggregate CSV schemas, zero row-output
  inventory, named-stream/candidate-order replay, and plot dimensions.

The committed Seaborn plot was visually inspected at 1,950 × 1,350 pixels. It
shows relative holdout MSE with block intervals, the error-correlation matrix,
mean drop-one marginal contribution, and turnover versus the costed
directional diagnostic with distinct generator/bootstrap seeds, sample count,
block policy, and synthetic scope in the title.

## Decision and remaining work

MR9 accepts the ensemble *engineering boundary*. It does not promote an expert
or policy. Market evaluation must regenerate complete OOF expert panels from
licensed point-in-time data; bind expert feature/model/fold identities to the
panel; freeze multiplicity, ablation, cost, capacity, and rejection rules; and
keep the final holdout outside tuning. Online adaptation on final-holdout
outcomes is explicitly outside this slice.
