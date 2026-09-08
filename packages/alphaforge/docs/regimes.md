# Regime and change-point model contracts (SF-S3-MR4)

Rule-based, Gaussian-mixture, hidden-Markov, CUSUM, and Bayesian online
change-point models behind one contract, plus retrospective segmenters and an
incremental-value harness that measures what a regime is actually worth.

> **No acceptance claim.** This MR delivers the models and the machinery for
> judging them. Whether any regime earns its place is SF-S3-MR11's frozen study.
> The issue's non-goal is the organizing principle here: never accept a regime
> because its chart looks intuitive, and never relabel states after inspecting a
> holdout.

---

## 1. Why a regime model needs an unusual amount of scaffolding

A regime model is exceptionally easy to fool yourself with. It produces a chart
that looks obviously correct — calm stretches blue, the crash red — and the chart
is worthless as evidence, because the model was fit on the whole sample
*including* the crash. Four properties in
[`base.py`](../alphaforge/regimes/base.py) exist to make that mistake hard to
commit by accident.

### Causality is declared, not assumed

Every model carries a class-level `causal` flag, and
`expanding_state_probabilities` — the only sanctioned way to turn a regime into a
feature — **refuses** a non-causal model outright. For each block it fits a
*fresh* model on `series[:k]`, filters forward to the block end, and keeps only
that block's rows. A factory is required rather than an instance, because reusing
one model across refits would carry state across the very boundary the refit
exists to enforce.

### Posteriors, not labels

Every model returns a full state posterior; `confidence` and `entropy` are
derived from it. A hard label discards the model's own uncertainty, which is
precisely the quantity that should shrink a position when the regime call is
marginal. `hard_labels()` exists for reporting and says so in its docstring.

### Canonical label ordering

EM has no preferred labelling — two runs can find the same two states and swap
their indices. Every model sorts states by a declared key (increasing variance
for GMM/HMM), so "state 1" means the same thing across runs, folds, and refits.
Without this, a regime-conditioned backtest silently mixes two different
definitions across refit boundaries. `ordering_rule` publishes the key so a
consumer never has to read the implementation to know what state 0 means.

### Deterministic identity

`identity` hashes configuration **and** fitted parameters. Configuration alone
would collide two fits on different training windows; parameters alone would
collide two different specifications that happened to converge to similar
numbers.

---

## 2. The models

| model | kind | causal | notes |
|---|---|---|---|
| `RuleBasedRegime` | threshold on a trailing statistic | yes | the baseline everything else must beat |
| `GaussianMixtureRegime` | EM mixture | yes | no time structure; flips on outliers |
| `GaussianHMMRegime` | Baum-Welch, K states | yes | persistence is *fitted*, not assumed |
| `CusumRegime` | sequential test | yes | minimal state, near-optimal detection delay |
| `BayesianOnlineChangePoint` | run-length posterior | yes | causal by construction |
| `segment_bayesian` | penalized DP | **no** | retrospective only |
| `segment_kernel` | RBF kernel DP | **no** | retrospective only |

### Rule-based — the baseline

Four trailing statistics: `volatility`, `trend` (mean over its own noise),
`drawdown`, and `volatility_change` (log ratio of short to long volatility, the
*leading* one, since expansion precedes the stress a level rule only confirms
afterwards).

Two design choices matter. Memberships are **soft**: a hard cut at the 70th
percentile flips the regime on a rounding difference when volatility sits at the
boundary, and that flicker goes straight into position sizing. And thresholds are
quantiles of the **training window only** — recomputing a quantile over the full
sample would define "high volatility" partly by volatility that had not happened
yet.

`drawdown` inverts its direction (`elevated_above = False`), because a deeper
drawdown is a *lower* number and getting that backwards silently inverts the
regime.

### GMM and HMM

Deliberately complementary. GMM treats observations as independent draws, so it
knows nothing about time and its posterior can flip on a single outlier. HMM adds
a transition matrix, so persistence is fitted. If the fitted transition matrix
comes out near-uniform, the HMM found no persistence and its extra parameters
bought nothing — checkable from `fitted_parameters()`, not assumed.

`expected_durations()` returns `1/(1 - a_ii)` per state. It is the most
interpretable summary of a fitted HMM: **a state with an expected duration of 1.2
bars is not a regime, it is a relabelled outlier detector**, and the number says
so directly.

Inference uses **filtered** posteriors `P(state_t | x_1..t)`, never smoothed.
Smoothing conditions on the entire sample and is the single most common way a
regime feature acquires lookahead — the smoothed series looks cleaner precisely
because it has seen the future.

Numerical care: log-space with log-sum-exp throughout, a variance floor against
the classic degenerate EM solution (a component collapsing onto one point, driving
variance to zero and likelihood to infinity — which looks like spectacular
convergence), and deterministic quantile initialization so the fit is
reproducible and the identity hash meaningful.

### CUSUM

Accumulates standardized deviations and alarms past a threshold. `drift` is the
slack that makes it usable — without it the statistic accumulates noise
indefinitely and eventually crosses any threshold. The baseline uses median and
MAD rather than mean and standard deviation, so it is not dragged by the very
excursions it exists to detect. After an alarm the statistic **resets**, so the
detector finds the next change instead of latching permanently.

Its posterior is a bounded monotone map of the exceedance, **not a calibrated
probability**. CUSUM is a test statistic, not a generative model, and
`fitted_parameters()` records `posterior_is_calibrated: False` so the number is
never mistaken for one.

### Bayesian online change-point detection

Adams & MacKay (2007): a posterior over the **run length** since the last change,
updated recursively. Causal by construction rather than by discipline — the
recursion has no access to future data, so there is no smoothed variant to
accidentally use.

Normal-Inverse-Gamma conjugate prior, so the predictive is Student-t. That matters
here: heavy tails stop a single large return from being read as a certain regime
change, which a Gaussian predictive does routinely on financial data. The
run-length posterior is truncated at `max_run_length` with the tail folded into
the last bin, conserving probability mass rather than leaking it, and bounding
both memory and per-bar time.

### Retrospective segmenters

`segment_bayesian` (penalized Gaussian model evidence) and `segment_kernel`
(RBF kernel scatter) both use exact dynamic programming, so each returns the
global optimum for its cost — not a greedy approximation whose output depends on
scan order.

They return `Segmentation`, **not** a `RegimeModel`, so they cannot be handed to
the causal driver at all. The type system enforces the separation rather than a
docstring warning, and `Segmentation` raises if constructed with `causal=True`.

They are still worth having: retrospective segmentation is the reference against
which an online detector's lag and false-alarm rate are measured. The comparison
is the deliverable, not the segmentation.

Kernel segmentation earns its place by detecting what a Gaussian segmenter is
blind to — a change in *shape* (skew, tails, multimodality) at matched mean and
variance. There is a test for exactly that case.

---

## 3. Incremental-value evidence

[`evidence.py`](../alphaforge/regimes/evidence.py) is what replaces the chart. It
measures a regime series against an explicit **no-regime baseline** on the same
rows, along the five axes the sprint plan names: forecast, calibration, sizing,
drawdown, and stability.

Every measure is a **difference**, never a level. A regime-conditioned Sharpe of
1.2 says nothing alone; against an unconditional 1.3 it says the regime destroyed
value, and only the differenced form makes that impossible to miss.

**Sign conventions differ by field and are stated on each one**, because a
reversed sign turns a harmful regime into an apparent improvement. `mse_delta`
and `brier_delta` improve when negative; `sharpe_delta` and `max_drawdown_delta`
improve when positive — the latter because a drawdown is a negative number, so a
shallower one is the larger value. (This was got backwards in the first draft and
caught by running it; hence the emphasis.)

Stability is reported alongside performance, not after it: `flip_rate`,
`mean_duration`, `mean_entropy`, and `exposure_turnover`. A regime that changes
every other bar can post an attractive Sharpe purely by trading noise and will
not survive costs.

**Honest limitation, stated in the code:** the forecast axis computes conditional
means on the same rows it scores, so `mse_delta` is an in-sample *upper bound* on
forecast value, not an estimate of out-of-sample skill. It is the easiest number
in the record to over-read.

`improves_nothing()` is a reporting convenience, explicitly **not** a rejection
rule — the acceptance threshold must be frozen before results are seen and
belongs to the study. `compare_regime_models` sorts by model name rather than by
any metric, because ordering by performance invites reading the top row as a
winner.

---

## 4. Measured behaviour

Synthetic calm/stress/calm series (σ 0.006 → 0.025 → 0.006, switch at bar 400),
expanding refits from bar 300 every 100 bars:

| model | Sharpe Δ | drawdown Δ | Brier Δ | flip rate | mean duration | mean entropy |
|---|---|---|---|---|---|---|
| rule | +1.60 | +0.585 | −0.113 | 0.013 | 66.6 | 0.381 |
| gmm | +1.28 | +0.585 | +0.045 | 0.135 | 7.3 | 0.391 |
| **hmm** | **+1.73** | +0.557 | +0.015 | **0.005** | **149.8** | **0.050** |
| cusum | +0.40 | +0.443 | −0.075 | 0.171 | 5.8 | 0.499 |
| bocpd | −0.21 | +0.016 | +0.074 | 0.007 | 119.8 | 0.241 |

Read honestly: the HMM's transition matrix buys it a flip rate 27× lower than the
GMM's, which is exactly the persistence argument. The rule-based baseline is
*competitive* — it beats the GMM, CUSUM, and BOCPD on Sharpe delta while needing
no EM, no seed, and no convergence check. And BOCPD's negative Sharpe delta is
reported as-is; it detects changes well and sizes badly on this series.

**This is synthetic data with a regime deliberately planted in it.** It
demonstrates that the machinery recovers a regime that is genuinely there. It is
not evidence that regimes exist in markets, and nothing here should be quoted as
such.

Retrospective segmentation locates the planted break at bar 403 (truth 400).

---

## 5. Compute and bounds

Python 3.13, macOS 15.5 arm64, single process, median of 3.

**Expanding causal estimation, 2520 bars, 32 refits:**

| model | median |
|---|---|
| rule | 0.01 s |
| gmm | 0.09 s |
| cusum | 0.09 s |
| hmm (2 states) | 1.17 s |
| bocpd | 1.59 s |
| hmm (3 states) | 4.22 s |

The HMM's cost is ~3.6× per added state, since Baum-Welch is `O(T·K²)` per
iteration and every refit runs it from scratch. Budget before raising `n_states`.

**Retrospective segmentation** is quadratic in series length: 0.06 s (Bayesian)
and 0.15 s (kernel) at n=2000, against 0.01 s at n=500. Hence the
`MAX_SEGMENT_SERIES = 5000` ceiling — an unbounded request is a configuration
mistake, not a long wait.

Other ceilings: `MAX_STATES` 8, `MAX_EM_ITERATIONS` 2000,
`MIN_FIT_OBSERVATIONS` 30, `MAX_CHANGE_POINTS` 50, BOCPD `max_run_length` ≤ 10000.

---

## 6. Evidence

`tests/test_regimes.py` — 102 tests:

* **Causality/leakage** — the mutation test for all five causal models: rewrite
  every bar after a cutoff, assert earlier posteriors are bit-identical. Warm-up
  rows carry no estimate. The driver refuses a non-causal model and a non-model.
* **Identity/labelling** — deterministic and training-window sensitive; different
  models with similar parameters do not collide; states canonically
  variance-ordered for K ∈ {2,3,4}; identity before fit refused.
* **Posteriors** — sum to one, non-negative, confidence in [0.5, 1], entropy in
  [0, 1], NaN where unobserved; container validation.
* **Recovery/no-change/abrupt/gradual** — stress posterior higher inside the
  planted window; a constant series does not manufacture switching; retrospective
  segmentation locates an abrupt break within 25 bars; BOCPD's change posterior
  rises after a step; a ramp is detected late but detected.
* **Degenerate/non-convergence** — short series, a training window shorter than
  the rule's own lookback, missing observations not imputed, exhausted EM budget
  reported as `converged=False`, 14 out-of-range configuration refusals.
* **Retrospective separation** — segmenters are never causal, are not
  `RegimeModel`s, refuse a causal claim, refuse non-finite/oversized input, and
  a higher penalty never yields more change points.
* **Evidence** — forward returns strictly forward; every model produces evidence;
  a de-risking regime improves drawdown; a *constant* posterior adds exactly zero
  sizing value; non-causal states refused; misalignment refused; comparison table
  deterministic under input order.

Coverage: `base.py` 91%, `changepoint.py` 93%, `evidence.py` 93%,
`mixture.py` 95%, `rules.py` 89% (branch).

---

## 7. Planned ablations — not yet run

Designed, deliberately not executed before the Sprint 2 baselines are frozen.
Against the no-regime baseline on identical folds with a pre-registered rejection
rule:

1. no regime (baseline);
2. rule-based regime;
3. GMM;
4. HMM at K ∈ {2, 3};
5. CUSUM;
6. BOCPD;

with online-vs-retrospective detection lag as a diagnostic, and regime-conditioned
sizing evaluated net of the turnover each regime implies. SF-S3-MR11 publishes the
decision.

---

## 8. Residual limitations

* **No acceptance evidence exists**, and none is implied. The measured table is
  synthetic-recovery evidence that the machinery works.
* **`mse_delta` is an in-sample upper bound**, not out-of-sample forecast skill.
* **CUSUM's posterior is not calibrated** and is flagged as such in its
  parameters.
* **A regime series inherits refit staleness.** Parameters are up to
  `refit_every` bars old; larger cadence is cheaper and staler, and the trade is
  explicit rather than hidden in a default.
* **State count is a declared choice, not inferred.** Nothing here selects K, and
  selecting it on the data being evaluated would be the same error the frozen
  study exists to prevent.
* **Univariate.** Every model observes one series. Correlation and liquidity
  regimes from the sprint plan need a multivariate emission model and are not
  implemented; claiming them from a univariate fit would be inventing evidence.
* **State-space models** (Kalman, dynamic linear, particle filters) from
  Workstream 3.3 are not in this MR.
* **Synthetic evidence only.** Nothing here is market evidence.
