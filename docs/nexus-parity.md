# Source and workstation parity

Parity has two distinct evidence boundaries: exact source preservation and
runtime compatibility. `uv run python -m foundry_build.assembly verify` verifies
all 167 AlphaForge and 115 Signalattice original commits, 1,972/1,246 objects,
88/52 refs and both exact current package trees. Current source remote `dev`
identities match the frozen inputs in [the assembly record](assembly.md).
This covers every tracked source file, public symbol, CLI/API implementation,
config/schema, test, fixture, document, evidence artifact and dependency lock.
The GitHub metadata ledgers retain original issue/milestone/PR/release links.
Tree equality proves preservation, not that an existing source defect is fixed.

| Source capability | Unified access | Qualification |
| --- | --- | --- |
| AlphaForge data source, symbols/days and benchmark | Nexus catalog/configuration; retained dashboard | Catalog/real-worker integration and `test_backtest_ui.py` |
| Every model registry entry, optional backend status | Nexus catalog selector; original registry/extras retained | Runtime catalog and unchanged registry; unavailable extras remain explicit |
| long_short, long_only_topk, rank_weighted, confidence_weighted | Nexus selector and original dashboard | Catalog contract and retained dashboard tests |
| Baselines, seed, model parameters | Complete configuration; original dashboard | Schema/preflight and real-worker tests |
| Costs, risk, portfolio limits, causal folds/regimes | Nexus grouped controls and complete configuration | Schema, preflight and existing worker evidence |
| Backtest returns, costs, exposures, model metrics and turnover | Nexus evidence/charts; original dashboard | Real-worker browser test and bounded evidence tables |
| IC, rank IC and prediction quantiles | Nexus evidence tables; original Signal Quality tab | Unchanged source helpers plus real-worker projection |
| Regime performance, stress, capacity, holdings and concentration | Nexus evidence tables; original Risk & Regimes tab | Real-worker evidence; capacity is a proxy |
| Latest-run artifacts, deflated Sharpe, PBO and detailed fills | Retained AlphaForge dashboard | Exact source tree and dashboard tests; not all projected by API v1 |
| Simulated paper orders and execution/native core | Original AlphaForge CLI/dashboard/core | Exact source and historical qualification; no real brokerage |
| Signalattice liveness, readiness and degraded states | Retained `/console` | 82 console boundary/integration/HTTP tests |
| Run catalog with keyset pagination | Retained `/console/runs` | Same console tests and unchanged route implementation |
| Artifact lineage, integrity and evidence state | Retained `/console/evidence`; Nexus research hashes | Console tests plus browser evidence inspection |
| Promotion comparisons, gates and corrections | Retained `/console/comparison` | Console tests; distinct from Nexus's research pair comparison |
| Proper-score calibration and uncertainty | Retained `/console/calibration` | Console tests; never invented for regression-only Nexus runs |
| Drift, latency and operational telemetry | Retained `/console/operations` | Console tests; not an execution-monitoring claim |
| Governance lane state and chain health | Retained `/console/governance` | Console boundary tests; read-only authority retained |
| Ingestion, forecasting, CLI/API, data contracts and providers | Original Signalattice package entry points | Exact source/lock preservation and real cross-package research integration |

Launch the original AlphaForge dashboard after preparing its compatibility context:

```bash
cd packages/alphaforge
uv run streamlit run apps/dashboard.py --server.address 127.0.0.1
```

Signalattice's [console guide](../packages/signalattice/docs/console.md) retains its
locked build and explicit read-port/service construction. No private store is
silently discovered or mounted by Nexus. Its seven views remain separate, working
applications; Nexus is an additive projection of API v1, not a claim that every
legacy view has been ported into one screen.

No unexplained source functionality loss was found at the frozen inputs. Runtime
evidence is bounded to the focused tests and preserved earlier source qualification;
optional backends are not claimed exercised by this local run. The existing
Signalattice container reproducibility, governance lifecycle, console goldens,
shadow ADR and wall-clock campaign issues remain visible in the source repository.
Neither source repository is deleted, archived or rewritten by this release.
