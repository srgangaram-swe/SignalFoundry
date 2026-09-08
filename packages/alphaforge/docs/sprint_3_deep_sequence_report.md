# Sprint 3 deep-sequence benchmark report

This report records the SF-S3-MR6 engineering result. It does not select a
production model, access a protected final holdout, authorize paper or live
trading, or claim a persistent edge.

## Frozen run

The committed `configs/deep_sequence_benchmark.yaml` profile compared
LightGBM, causal CNN, TCN, masked LSTM, masked GRU, and causal Transformer on
one matched development fold. Every candidate produced 1,134 OOS predictions
across 126 sessions. Neural candidates used four epochs, CPU execution, seed
42, a common 20-session window, width 16, and the same validation and
optimization policy. The simple diagnostic charged 5.5 bps against
turnover. It is not the event-driven backtester.

The source was the verified cached Signalattice WIKI bundle
`63bc9af39ed5199cf4027355163c95767c8530b31af9d986a9e7a18006c0e26f`:
13,169 rows, ten symbols, 2013-01-02 through 2018-03-27. Cache replay made no
provider request. The bundle lacks complete historical revisions,
point-in-time universe membership, universe records, complete corporate
actions, and corporate-action records. It therefore remains engineering data
with survivorship, selection, and adjustment risks.

## Result

| Candidate | Rank IC | Net mean daily diagnostic | Parameters | Fit wall seconds |
|---|---:|---:|---:|---:|
| LightGBM | -0.05331 | -0.002074 | not measured | not measured |
| causal CNN | -0.00997 | -0.002661 | 1,505 | 0.500 |
| TCN | 0.08832 | -0.000040 | 2,321 | 1.472 |
| LSTM | 0.10047 | 0.001945 | 5,041 | 0.767 |
| GRU | 0.01776 | -0.001234 | 3,953 | 0.649 |
| causal Transformer | 0.02773 | -0.000040 | 5,457 | 0.694 |

The single-fold result rejects any blanket claim that deep sequence models
improve net results: only LSTM was positive under this simplified diagnostic,
and four of the six candidates were negative after costs. LSTM's result is a
development observation, not a confirmed discovery. TCN and Transformer were
approximately flat after the simplified cost model. No candidate may be tuned
against a final interval because of this table.

Regression-calibration intercept, slope, and RMSE, predictive errors,
correlations, turnover, parameter bytes, CPU/wall measurements, convergence,
and best validation loss are retained in the
[machine-readable model summary](evidence/signal_foundry_sprint_3/deep_sequence/model_summary.csv).
The [Seaborn comparison
plot](evidence/signal_foundry_sprint_3/deep_sequence/model_comparison.png) was
generated directly from that summary and visually inspected at 2,991 by 929
pixels. Runtime and data limitations are in
[`summary.json`](evidence/signal_foundry_sprint_3/deep_sequence/summary.json).
No ticker-level observations, predictions, model checkpoints, licensed raw
data, or credentials are published.

## Decision and next evidence

MR6 qualifies the causal implementation and comparison machinery. It does not
promote a candidate. The next legitimate sequence-model study needs multiple
matched development folds, uncertainty around fold-level differences,
pre-registered ablations, correction for every attempted variant, the full
event-driven cost model, regime and capacity sensitivity, and licensed
point-in-time data. A final holdout stays untouched until one candidate and
rejection rule are frozen.

CPU peak host memory is not sampled; `peak_device_bytes=0` means no accelerator
allocation measurement. LightGBM parameter and fit-resource fields were not
captured by the new Torch resource contract. These are explicit limitations,
not zeros.
