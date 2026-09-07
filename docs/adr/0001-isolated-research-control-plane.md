# ADR 0001 — Isolated local research workers

Status: accepted for AlphaForge #80, extending source ADR 0023 without changing
the preserved source packages. This contract authorizes simulation only.

## Decision

Keep the root HTTP application, AlphaForge and Signalattice in separate locked
Python environments. Their pandas major versions differ; a dependency union is
neither necessary nor qualified. Fixed, reviewed worker entry points compose the
existing public data validators and research functions. No mathematical algorithm
is reimplemented in the API or browser. No arbitrary subprocess, import, path,
provider URL, credential or broker operation is accepted over HTTP.

The immutable Pydantic contracts generate one versioned OpenAPI document and its
TypeScript bindings. Catalog metadata identifies unavailable implementations;
validation checks actual package boundaries before admitting research. Data IDs
refer only to a bounded operator-selected local bundle directory. Both producer
and consumer validate a selected bundle independently. Raw observations stay in
the worker and are never HTTP responses.

One durable bounded job queue owns state transitions, idempotency, cancellation,
recovery, audit and atomic evidence publication. Subprocesses isolate native
crashes from the API. Concurrency, output, wall/CPU time, numerical threads,
dataset shape, retained jobs and artifact bytes are bounded. Limits describe
local interactive research, not distributed throughput or low-latency trading.
Linux additionally applies an 8 GiB address-space limit. macOS rejects that limit;
both hosts instead enforce a sampled 2 GiB aggregate worker/descendant RSS stop
threshold. A 50 ms sampling interval can overshoot and is not a kernel-enforced
resident-memory boundary. CPU, file size and descriptor ceilings remain OS limits.
The initial implementation is POSIX-only and single-process/single-owner; it is
not an adversarial code sandbox or a multi-tenant scheduler.

Bind only loopback. Reject foreign Host/Origin and browser cross-site requests,
require JSON plus an explicit client header for mutations, and do not enable
CORS. This limits browser-driven local-service abuse; it does not authenticate
other processes belonging to the local user. Remote exposure requires a separate
reviewed authentication, authorization, TLS and deployment design. It is not a
configuration switch in this release.

## Scientific boundary

Chronological train/test folds and embargo are explicit. Learned preprocessing
is fitted inside each training fold by the existing AlphaForge transformer.
Models and baselines receive identical data, folds and cost policies. Diagnostics
carry units, counts and assumptions; missing evidence is not a zero. Interactive
walk-forward exploration is development evidence, not a frozen final holdout or
strategy qualification. WIKI's historical/current-vintage and survivorship limits
remain visible. Successful execution never changes the live-readiness verdict.

## Alternatives and consequences

An in-process dependency union was rejected for incompatible locks and shared
native failure state. An external task broker would add deployment, credential
and retry semantics unnecessary for one local operator. Copying the domain math
would create two implementations requiring scientific reconciliation. The chosen
boundary pays cold subprocess/import overhead for explicit isolation; benchmarks
must report it. Original CLI/API/dashboards remain usable independently.

The assembly's local GUI check found an unrestricted macOS Arrow allocator crash;
three browser-session backtests passed with numerical thread counts bounded to
one. This is evidence for an explicit launch profile, not an upstream allocator
fix. Preserve that constraint in workers and legacy launch instructions.

References: [FastAPI security](https://fastapi.tiangolo.com/tutorial/security/),
[Python subprocess lifecycle](https://docs.python.org/3/library/subprocess.html),
[Arrow runtime/thread settings](https://arrow.apache.org/docs/cpp/env_vars.html).
