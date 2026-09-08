# Metric governance and time-series distributions

SF-S2-MR7 establishes one interpretation boundary for predictive, calibration,
information-coefficient, return, risk, drawdown, turnover, exposure, capacity,
and execution-cost evidence. A metric without a declared unit, aggregation,
sample requirement, missing-data policy, annualization rule, benchmark, and
invalid-state behavior is not publishable through this boundary.

These metrics describe a frozen historical evaluation. They do not establish a
tradable edge, future profitability, paper readiness, or permission to place an
order.

## Input and trust boundary

`evaluate_metric_suite` consumes two chronological tables:

- a prediction panel with `date`, `target`, and `prediction`, optionally paired
  `probability` and binary `outcome`; and
- one unique-date trading record with `date` and decimal `return`, optionally
  including benchmark return, turnover, exposure, accounting cost, gross traded
  notional, and capacity diagnostics.

Dates must already be monotonically ordered. Required values and supplied
optional metric columns must be finite. The API rejects missing values rather
than silently dropping or imputing them, rejects returns below -100%, and
requires cost/notional reconciliation. It does not reconstruct whether the
prediction panel is genuinely out of sample; the upstream fold and experiment
manifests remain the authority for that provenance.

`MetricContract` records the interpretation of every supported metric.
`MetricEstimate` represents an undefined quantity with `value=None`,
`status="undefined"`, and a reason. Undefined Sharpe for constant returns,
undefined cost bps for no trades, and undefined IC for tied/constant
cross-sections never become a favorable zero or an unlabelled NaN.

## Mathematics

For prediction error \(e_i = \hat y_i-y_i\), MSE and MAE are arithmetic means
of \(e_i^2\) and \(|e_i|\). Directional accuracy treats zero as non-positive.
Pearson IC and Spearman rank IC are computed within each date and then averaged
over valid dates; Spearman ties use average ranks. Constant cross-sections are
undefined and excluded from the IC time series, with the retained date count
published.

Brier score is the mean squared binary-probability error. ECE uses fixed,
predeclared bins and count weighting. Its value remains bin-policy dependent.

Trading wealth begins at one:

\[
W_t=\prod_{i=1}^{t}(1+r_i), \qquad
D_t=\frac{W_t}{\max_{0\leq j\leq t}W_j}-1.
\]

Total return is \(W_T-1\), and maximum drawdown is \(\min_t D_t\). A -100%
period bankrupts the path; later positive returns cannot restore zero wealth.
Annualized return uses the actual elapsed calendar span:

\[
(1+R)^{365.2425/\Delta_{\text{days}}}-1.
\]

The effective observation frequency is
\((n-1)365.2425/\Delta_{\text{days}}\). Volatility, Sharpe, tracking error,
and benchmark comparisons use that declared frequency instead of assuming 252
observations for irregular calendars. Sharpe uses the arithmetic sample mean
and sample standard deviation and is undefined at zero variance.

Cost bps is \(10{,}000\sum c_i/\sum |N_i|\) over active trades. Cost must be
zero wherever gross traded notional is zero. Capacity fill and constrained
fractions are bounded descriptive means; they inherit the upstream capacity
model's lagged-liquidity and sensitivity limitations.

## Dependence-aware uncertainty

Every defined scalar is replayed under a moving-block bootstrap using
`BlockBootstrapConfig`. Each `TimeSeriesDistribution` publishes:

- point estimate, percentile interval, variance, and standard error;
- observation and resample counts;
- contiguous block length, seed, confidence level, and circular-boundary
  policy; and
- stationarity, block-adequacy, and interpretation assumptions.

There is no IID mode. Pairwise statistics keep their rows paired during block
resampling. Prediction-panel resampling selects contiguous dates and retains
each selected date's complete cross-section; it never splits a same-date asset
set into artificial time steps. The implementation fails closed if a statistic
becomes undefined under the predeclared resampling policy. Bootstrap variance quantifies
resampling variability under these assumptions; it is not parameter certainty
and does not cover regime change, selection bias, data revisions, or an
unrecorded research-trial history.

## Configuration and reproducibility

`configs/metrics.yaml` freezes minimum sample sizes, reliability bins,
benchmark identity, resample count, block length, confidence level, circular
policy, and seed. Its strict schema rejects unknown fields, coercion, unbounded
work, and a block length longer than either minimum sample requirement.

The reference tests use deterministic synthetic tables and independent NumPy
calculations. They cover no trades, constant returns, bankruptcy, irregular
calendars, cross-sectional ties, non-finite values, short histories, duplicate
or reversed time, partial calibration columns, invalid probabilities, sparse
trades, and cost/notional inconsistencies. No credential, licensed observation,
row-level market artifact, or final holdout is committed.

## Rollback and residual risk

Rollback removes `metric_suite.py`, its exports, strict configuration, and
documentation. It does not rewrite existing backtest artifacts or change the
legacy metric functions, so callers can revert this slice without a storage
migration.

Before paper or live consideration, the suite must run on licensed
point-in-time data with a frozen benchmark and evaluation policy. Block length,
market regimes, execution costs, capacity, tail behavior, and strategy
selection must receive independent sensitivity and adversarial review. Metrics
are one evidence layer; operational controls and explicit owner approval remain
separate mandatory gates.
