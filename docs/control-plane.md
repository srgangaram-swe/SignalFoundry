# Local research control plane (API v1)

This additive interface composes the preserved AlphaForge and Signalattice
packages. It runs development simulations, not live orders. A succeeded job
means validated evidence was published atomically—not that a strategy passed
qualification. Every result retains `live_readiness: NOT_READY`.

## Installation and launch

Use Python 3.13 on macOS or Linux, Node 24 for client-contract tooling, and the
committed locks. From the unified repository root:

```bash
uv sync --locked --extra dev
uv sync --project packages/alphaforge --locked --extra dev --extra data
uv sync --project packages/signalattice --locked --extra dev
uv run python -m foundry_build.context alphaforge
uv run python -m foundry_build.context signalattice
uv run signal-foundry serve
```

The API is at `http://127.0.0.1:8765/api/v1/catalog`; its OpenAPI document is at
`/api/v1/openapi.json`. Nexus is a separate, subsequent delivery; this API MR
does not claim the new workstation exists. State is private, ignored
`var/research`. Do not run a second instance against that directory. Existing
state must be owned by the launching user, with directory mode 0700 and database
mode 0600. The process will reject broad permissions instead of silently changing
another application's files. The initial platform is POSIX-only.

For approved historical bundles, add `--bundles /absolute/local/bundle-parent`
**before** `serve`. The browser selects content IDs, never filesystem paths.
No provider key is required to replay an already verified bundle. WIKI history
ends in March 2018 and lacks complete point-in-time universe, corporate-action
and revision evidence. It is suitable for exercising mechanics, not today's
paper-trading inputs or proof of a tradable edge.

## Guided HTTP workflow

1. `GET /api/v1/catalog`: actual regression/strategy registries, dependency
   availability, immutable bundle metadata and limitations.
2. `POST /api/v1/validate`: validate a `ResearchRequest` across producer and
   consumer; reject unregistered models, dangerous parameters and invalid data.
3. `POST /api/v1/jobs`: same request plus `Idempotency-Key` (16–128 URL-safe
   characters). The key binds one immutable request; a conflicting retry is 409.
4. `GET /api/v1/jobs/{id}`: queued/running/succeeded/failed/cancelled status.
   `POST /api/v1/jobs/{id}/cancel` with `{}` cancels pending work. Terminal
   cancellation is idempotent and cannot undo completed publication.
5. `GET /api/v1/jobs/{id}/evidence` and `/audit`: hash-verified canonical
   aggregates and ordered state transitions. No raw market bars or model files.
6. `GET /api/v1/compare/{left}/{right}`: provenance/policy compatibility and
   both immutable artifacts. An incompatible pair is explicitly unranked.

Every POST requires exactly `Content-Type: application/json` and
`X-Signal-Foundry-Client: nexus`. Request bodies are bounded to 16 KiB and JSON
depth to 32; duplicate keys, nonfinite values, unknown fields and coerced
booleans fail closed. Structured `Problem` responses never echo invalid inputs.
`signal-foundry example` prints the complete default request for local editing;
`signal-foundry validate request.json` and `signal-foundry run request.json`
exercise the same preflight and scheduler without a browser. The CLI run prints
the durable job identity; retrieve its artifact through the API afterward.

## Scientific contract

The fixed `technical_v1_no_hmm` profile invokes AlphaForge's registered technical
feature pipeline with HMM disabled. The HMM warmup otherwise leaves an entirely
missing feature in the initial 252-session training window. This is an explicit
profile, not a fitted-transform fallback. Standardization is fitted in each
training fold; baselines that require raw features retain the source behavior.
Test windows do not overlap, and embargo must cover the prediction horizon.
Trailing sessions are reserved for source-engine terminal liquidation, accounting
for the rebalance interval and execution lag.

Models and forecast baselines use a common out-of-sample calendar and identical
portfolio/cost policies. A forecast baseline name does not mean a passive
investable portfolio. Transaction-cost curve values are fractions of previous
equity; P&L attribution is in simulated dollars. Missing/nonfinite diagnostics
remain null, never a fabricated zero. Large tables disclose retained and total
row counts. Training termination and warnings are evidence, not automatically
successful convergence. Interactive exploration is not an untouched final test.

Risk limits constrain targets; marked holdings can drift. Holdings, concentration
and first-order beta-one stress use the last nonzero snapshot before liquidation,
not fictitious current positions. Daily-bar participation capacity is a proxy,
not a liquidity guarantee. Moving-block mean-return intervals use 200 exploratory
resamples and do not correct repeated model selection. Costs, borrow, funding and
fills are assumed rather than broker observations. No current borrow inventory,
tax, intraday queue-position or live-latency evidence is established.

## Resource and failure policy

One FIFO research worker, two total package subprocesses (including preflight),
eight pending jobs, 64 retained jobs, 40,000 observations, 32 instruments, 2,000
sessions, 4 MiB per evidence artifact and eight concurrent HTTP requests are
hard admission bounds. Jobs have a 150-second wall deadline and worker CPU limit
of 120 seconds. Numerical libraries receive one thread. The worker/descendant RSS
stop threshold is 2 GiB, sampled every 50 ms; sampling can overshoot. Linux also
applies an 8 GiB address-space limit. These are workstation limits, not a hardened
untrusted-code sandbox or a market execution speed claim.

SQLite publishes success, canonical evidence, its hash and audit in one FULL-sync
transaction. A process-lifetime lock excludes a second owner. Interrupted jobs
fail on restart and are never automatically retrained. Cancellation linearizes
against publication: late cancelled worker output is discarded. Completed
artifacts are not evicted to make room; choose a new private workspace after
reviewing the retention limit. Keep previous workspaces for provenance.

## Security boundary and remote exposure

Only 127.0.0.1 is an allowed CLI bind. Host and Origin are restricted to the exact
configured localhost/127.0.0.1 port; cross-site browser fetches, duplicate security
headers and simple form POSTs are rejected. CORS is absent. CSP disallows external
scripts, framing, objects and foreign connections. Responses are no-store/nosniff.
The worker inherits no provider credentials, Git auth, HOME, PYTHONPATH or user
site packages. Paths are operator-only, symlink components are rejected, SQL
values are bound, and subprocess commands come only from reviewed fixed policy.

This does **not** authenticate other processes belonging to the local user, defend
against a malicious owner replacing executable/source files, or provide kernel
container isolation. Do not port-forward, reverse-proxy or expose this service.
Remote use requires a separate reviewed TLS, authentication, authorization,
CSRF/session, tenancy, data entitlement and deployment design. No configuration
flag turns this release into that system. Broker connectivity remains absent.

## Preserved dashboards and rollback

Original package trees and entry points are unchanged. In the AlphaForge package,
the validated macOS Streamlit launch profile is:

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
VECLIB_MAXIMUM_THREADS=1 .venv/bin/streamlit run apps/dashboard.py \
  --server.address 127.0.0.1 --server.port 8501 --server.headless true \
  --browser.gatherUsageStats false
```

An unrestricted macOS run crashed in Arrow's allocator; three independent
browser sessions completed backtests with the bounded profile. That is a tested
launch constraint, not a proven upstream fix. Original Signalattice console/API
instructions remain under its package documentation. Rollback the additive root
adapter without changing either source package or deleting any historical refs.
