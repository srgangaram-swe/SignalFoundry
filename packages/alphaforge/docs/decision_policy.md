# Cost- and uncertainty-aware decision policy

`alphaforge.decision` is a pure eligibility boundary between model research and
portfolio construction. It decides whether one forecast is supported strongly
enough to continue downstream. It does not size a position, choose a venue,
construct an order, access an account, contact a broker, or authorize paper or
live trading. The boundary is currently opt-in and exercised only by the
standalone synthetic study; AlphaForge's active signal-to-portfolio path does
not call it.

## Contract and mathematics

One immutable `DecisionSignal` carries:

- a bounded signal and model identity;
- timezone-aware decision and data-availability times;
- signed expected return;
- expected cost and cost-estimation uncertainty;
- predictive uncertainty and model disagreement;
- typed regime support and regime uncertainty; and
- a drift score.

All return and cost values use decimal return per proposed unit of exposure.
For a structurally valid, finite, in-range signal, the policy computes

\[
C^+ =
\lambda_c\left(\widehat C + \lambda_{C,u}U_C\right),
\qquad
U^+ = \lambda_u U_p,
\]

\[
V^- = |\widehat R| - C^+ - U^+.
\]

Here \(\widehat R\) is signed expected return, \(\widehat C\) is expected cost,
\(U_C\) is cost uncertainty, \(U_p\) is predictive uncertainty,
\(\lambda_c \ge 1\) is the conservative cost multiplier, and
\(\lambda_{C,u},\lambda_u \ge 0\) are uncertainty multipliers. A signal is
eligible only when

\[
V^- > m,
\]

where \(m\) is the required margin. Equality abstains. The policy uses
`abs(expected_return)` solely to evaluate long or short forecast magnitude; the
forecast sign is retained as research metadata, not an executable side.

Independent maximum gates use `>` semantics, so equality to a declared maximum
passes. The final value test remains strictly greater than the margin. This
distinction is covered by boundary tests.

## Fail-closed gates and reason order

A decision accumulates all applicable reasons in this stable order:

1. `non_finite_input`
2. `out_of_bounds_input`
3. `future_data`
4. `stale_data`
5. `excess_cost`
6. `high_disagreement`
7. `unsupported_regime`
8. `uncertain_regime`
9. `drift_detected`
10. `excess_uncertainty`
11. `insufficient_margin`

Non-finite and invalid-range inputs never enter arithmetic. Their decisions
remain JSON-safe and retain the failed field names. Future data means
`data_available_at > decision_time`; stale data means age strictly exceeds the
configured maximum. Both `unsupported` and `unknown` regimes abstain.
One malformed estimate suppresses only the arithmetic or semantic gate that
requires that estimate. Other independently evaluable cost, disagreement,
regime, drift, uncertainty, and freshness failures remain visible.

`DecisionPolicy.evaluate_many` consumes no more than the configured batch
limit, rejects duplicate signal identities, and returns decisions sorted by
stable decision identity. Therefore caller iteration order cannot change a
decision or replay serialization.

## Identity and serialization

The policy ID is SHA-256 over every threshold field. A decision ID is SHA-256
over that policy ID and the complete signal input, including timestamps and
exact hexadecimal finite-float representations. Canonical tokens also identify
NaN and infinities so malformed estimates still receive reproducible audit
identities without emitting non-standard JSON.

`Decision.to_json()` sorts keys, uses stable enum values, and rejects JSON NaN
extensions. IDs are deterministic for the same typed inputs and policy; they
are not signatures and do not prove who produced a decision.

## Frozen offline study

`configs/decision_policy.yaml` freezes thresholds, seed, synthetic generator,
resource ceiling, expected-cost range, uncertainty/disagreement/regime/drift
rates, freshness faults, capacity proxy, and two ablations:

- `always_trade` removes all abstention gates;
- `never_trade` removes all eligibility.

Run the study into a new directory:

```bash
make decision-policy-evidence OUTPUT=/absolute/path/to/new/evidence
```

The generator creates 2,400 synthetic opportunities across 120 periods using
NumPy PCG64 and seed `20260726`. A `DecisionStudyObservation` stores the
`DecisionSignal` separately from realized return and realized cost.
`DecisionPolicy` sees only the signal. Tests replace every realized label and
prove that decisions and reason counts do not change.

The study publishes only three aggregate policy rows, non-exclusive reason
counts, a JSON summary, and a Seaborn plot. It never publishes synthetic signal
rows or realized labels. Publication stages a new directory and renames it
atomically; an existing destination or staging directory fails closed.

Metrics are:

- **coverage:** eligible decisions divided by opportunities;
- **selective risk:** loss frequency conditional on eligibility, undefined for
  never-trade;
- **mean net value:** realized direction-adjusted return minus realized cost,
  averaged over all opportunities with abstentions set to zero;
- **missed opportunity:** positive realized net value among abstentions,
  averaged over all opportunities;
- **turnover demand:** configured round-trip units per selected opportunity,
  averaged by period; and
- **capacity demand:** selected unit notionals divided by the configured
  per-period capacity proxy.

Turnover and capacity are pre-portfolio demand diagnostics, not orders or
fills. The reference generator is not calibrated to market or execution data.

## Complexity, security, and limitations

Single-signal evaluation is constant time. Batch validation and evaluation are
linear, followed by deterministic decision-ID ordering in \(O(n\log n)\)
comparison time; memory is bounded \(O(n)\) with
`observation_count <= maximum_batch_size <= 100,000`. The operation-count
benchmark exercises batches of 1, 64, and 4,096 and proves one evaluation per
accepted signal; a 4,097th item fails before any policy arithmetic. No
wall-clock threshold is a correctness gate. The study is deterministic,
CPU-only, network-independent, and accepts no credential, executable artifact,
pickle, market row, tensor, or model state.

This policy cannot guarantee a loss will be avoided. Its thresholds can become
miscalibrated as forecast, cost, or regime distributions shift. Expected cost
and uncertainty estimates can be wrong together. A favorable selective-risk
result may hide economically important missed opportunities. Thresholds must
be frozen on development evidence and may not be chosen on a protected final
holdout.

See [ADR 0007](adr/0007-fail-closed-decision-eligibility.md) and the
[SF-S3-MR10 evidence report](sprint_3_abstention_policy_report.md).
