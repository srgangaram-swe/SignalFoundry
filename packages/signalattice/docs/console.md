# Forecast-observability console

A local-only, read-only browser projection of the evidence this repository already stores. It shows
lineage, calibration, uncertainty, drift, evidence sufficiency, and governance state, each with its
limitations attached.

It is not a trading system. Nothing in it authorizes deployment, capital, paper trading, or live
trading, and no view is a claim of profitability. It cannot approve, apply, roll back, unfreeze, or
waive anything — see [ADR 0006](adr/0006-local-forecast-observability-console.md).

## Supported environment

| Component | Version | Role |
| --- | --- | --- |
| Node.js | 24 (`.nvmrc`) | build and test boundary only |
| React | 19.2 | rendering |
| React Router | 8 | routing |
| Vite | 8 | bundler |
| TypeScript | 5.9.3 | strict compiler (see the ADR for why not 7) |
| Vitest | 4.1 | unit and component tests |
| Playwright | 1.62 | browser tests |
| Zod | 4 | runtime decoding |

The hardened service container ships **without** Node and without a browser. The console is built
ahead of time and served as static files.

## Build and run

```bash
cd web
npm ci                 # locked install; no network beyond the registry
npm run build          # type-check, then bundle into web/dist
```

Then start the service with the bundle mounted:

```python
from quant_platform.governance.read_ports import GovernanceReadPorts
from quant_platform.service.api import create_app
from quant_platform.service.console import load_console_bundle

app = create_app(
    ports,                                        # existing evidence read ports
    governance=GovernanceReadPorts(registry_path),
    console=load_console_bundle("web/dist"),
)
```

The console is then at `http://127.0.0.1:<port>/console`. The service refuses any `Host` other than
`127.0.0.1` or `localhost`, so it is unreachable from another machine by construction.

Omitting `console=` leaves the boundary unmounted; omitting `governance=` leaves the governance
routes unregistered. Both are additive.

## The seven views

| Route | Shows |
| --- | --- |
| `/console` | liveness, readiness, contract version, explicit degraded state |
| `/console/runs` | keyset-paginated run summaries, no total count |
| `/console/evidence` | one run's artifacts, lineage, and verification status |
| `/console/comparison` | recorded promotion decisions with gates, tests, and correction |
| `/console/calibration` | proper scores beside their sample counts, and limitations |
| `/console/operations` | bounded telemetry samples |
| `/console/governance` | lane state, chain health, and the authority boundary |

There is no eighth view. The allowlist is enforced in both `CONSOLE_ROUTES` (Python) and
`CONSOLE_VIEWS` (TypeScript), and tested in both.

## State semantics

Every view and evidence panel resolves to one of nine states. They are deliberately distinct:

| State | Meaning |
| --- | --- |
| `LOADING` | the read is in flight |
| `READY` | complete, trustworthy evidence |
| `EMPTY` | the server answered; nothing is recorded |
| `PARTIAL` | some evidence is present and some is missing, named |
| `INSUFFICIENT_EVIDENCE` | data exists but cannot support the claim — a finding, not an absence |
| `INVALID` | the evidence contradicts itself or failed verification |
| `STALE` | real evidence, older than its freshness contract (24 hours) |
| `UNAVAILABLE` | the service could not answer |
| `ERROR` | the console could not interpret what it received |

A state carrying data renders **both** the banner and the data. Hiding an insufficient result would
make it indistinguishable from an empty one, and hiding an invalid governance lane would remove
exactly the lane worth investigating.

## Resource limits

| Limit | Value |
| --- | --- |
| Requests in flight | 4 |
| Per-request deadline | 10 s, hard abort |
| Automatic retry | none; retry is an explicit human action |
| Polling | none |
| Page size | 25 default, 100 maximum |
| Marks per panel | 256 |
| Initial JavaScript | 250 KiB gzip (currently 94.7 KiB) |
| Initial CSS | 50 KiB gzip (currently 1.5 KiB) |
| All emitted assets | 1 MiB (currently 335 KiB) |

Budgets are measured from the emitted bundle by `npm run budget`, which writes
`web/bundle-manifest.json` and fails the build when a budget is exceeded.

## Accessibility

Targets WCAG 2.2 AA. Status is conveyed by a word, a glyph, and a colour, so no single channel is
required. Every chart has a tabular equivalent in the same row. The layout reflows to 320 CSS
pixels and survives 200% zoom without horizontal document scroll; wide tables scroll inside their
own focusable region. Reduced motion and forced-colors are honoured.

`npm run e2e` runs axe against every route and against a failure state, with no serious or critical
violation waived.

## Troubleshooting

| Symptom | Likely cause |
| --- | --- |
| Every panel reads `UNAVAILABLE` | the service is not running, or is bound to a different port |
| A panel reads `ERROR: did not match the version-1 contract` | the served contract has drifted from the committed one; run `npm run api:generate` |
| The console 404s entirely | `console=` was not passed to `create_app`, or `web/dist` is absent |
| An asset 404s | the build emitted a file type the boundary does not serve; check `npm run build` output |
| `npm ci` fails on peer dependencies | the pinned TypeScript is 5.9.3 by design; see ADR 0006 |

## Evidence

`docs/benchmarks/console_evidence_2026-08-20.json` is collected from real build and test runs by
`scripts/collect_console_evidence.py`; `scripts/plot_console_evidence.py` renders it to
`reports/figures/console_evidence.png`. The figure refuses to render if a budget is exceeded, a
browser test failed, or a palette token falls below the AA threshold.

## Rollback

Pass `console=None` and delete `web/`. The API, container, registry, CAS, and OpenAPI contract are
untouched. Do not roll back by expanding CORS, exposing a remote listener, relaxing the CSP, or
adding a browser mutation path.
