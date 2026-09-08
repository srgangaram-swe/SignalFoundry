# Qualification dossier — sf-s4-synthetic-development

**Verdict: REJECTED**

> **No paper or live evaluation is authorized.** Every blocking failure below must be resolved and the candidate re-scored against a rubric frozen before the new evidence was seen.

- Rubric: `paper-v1` (`e8437e2b680c`)
- Frozen research plan: `ea7bfaa97729`
- Decided at: 2026-08-02T00:00:00

## Blocking failures (7)

### adjusted_p_value

Does the edge survive multiple-testing correction across the complete trial family?

- Observed: **not measured**
- Required: at most 0.05
- Reason: no observation supplied; an unmeasured criterion cannot pass

### capacity_utilization

Does the strategy fit inside its measured liquidity and borrow capacity?

- Observed: **not measured**
- Required: at most 1
- Reason: no observation supplied; an unmeasured criterion cannot pass

### max_drawdown

Is the worst peak-to-trough loss within the declared tolerance?

- Observed: **-0.538136**
- Required: at least -0.25
- Reason: missing required evidence: dataset, risk_attribution; an unevidenced metric is a failure, not a weaker pass

### net_return_over_baseline

Does the candidate beat a simple investable baseline after all modeled costs?

- Observed: **-0.196773**
- Required: at least 0
- Reason: missing required evidence: code, configuration, dataset; an unevidenced metric is a failure, not a weaker pass

### stress_downside_net_return

Under frozen execution perturbation, does the 5th-percentile path remain non-negative?

- Observed: **-0.330317**
- Required: at least 0
- Reason: observed -0.330317 fails required >= 0

### top_name_concentration

Is the result carried by more than a handful of names?

- Observed: **not measured**
- Required: at most 0.35
- Reason: no observation supplied; an unmeasured criterion cannot pass

### uncertainty_lower_bound

Is the dependence-aware lower confidence bound on mean return above zero?

- Observed: **not measured**
- Required: at least 0
- Reason: no observation supplied; an unmeasured criterion cannot pass

## Satisfied criteria (1)

| Criterion | Observed | Required | Evidence |
| --- | --- | --- | --- |
| stress_insolvent_paths | 0 | at most 0 | `stress_study:ea7bfaa9` |

## Limitations

- Every number above is simulated. No live or paper capital has been at risk.
- Monte Carlo frequencies come from a declared perturbation family and are not probabilities of loss in the market.
- A frozen rubric prevents moving the bar; it does not make the bar correct. The thresholds are judgements, recorded so they can be argued with.
- Passing every criterion establishes that the candidate was not obviously broken, not that it will earn money.
