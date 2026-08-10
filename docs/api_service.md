# Local read-only evidence API

Signalattice can serve verified run summaries and aggregate forecast evidence from its durable local
registry and content-addressed artifact store (CAS). The API is a single-owner, loopback-only
inspection boundary. It is not a hosted service, an execution engine, a broker interface, or a
paper/live-trading control plane.

[ADR 0003](adr/0003-local-read-only-evidence-api.md) defines the trust boundary and durable design
decision. [The run-registry guide](run_registry.md) defines storage bootstrap, lifecycle, artifact,
backup, and retention invariants.

## Capability and limitations

The version-1 service exposes:

- bounded registry-native and explicitly `legacy/unverified` run summaries;
- verified path-free artifact metadata;
- schema-1 aggregate forecast summaries;
- typed finite diagnostics; and
- structured plain-text model cards.

It does not expose forecast observations, ticker-level predictions, licensed/provider rows,
arbitrary reports, artifact bytes, filesystem paths, SQL, parameters, tags, model binaries, jobs,
cancellation, retention, data ingestion, broker controls, orders, positions, or any mutation. Row-
level pre-outcome forecasts and reconciliation are deferred to
[#21](https://github.com/srgangaram-swe/Signalattice/issues/21).

The API can inspect historical research evidence. It cannot establish that a model is profitable,
current, production-ready, paper-trading-ready, or safe to trade with capital.

## Install the optional service surface

The core package intentionally omits the HTTP framework. The current `serve-api` operator assembly
requires macOS because it reads registry authority from Keychain; the HTTP contracts do not provide
an environment-variable or CLI-secret fallback. Install the locked service extra in an isolated
environment:

```bash
uv sync --extra service
```

For an installed wheel:

```bash
python -m pip install "signalattice[service]"
```

Do not install `fastapi[standard]` or `uvicorn[standard]` as a workaround. Signalattice qualifies a
smaller dependency boundary and records its complete resolution in `uv.lock`.

## Prerequisites

The server opens existing state only. Before startup, an owning process must have:

1. explicitly initialized the durable registry and CAS under the procedure in
   [the registry guide](run_registry.md#explicit-bootstrap);
2. bound that registry to the exact CAS identity;
3. completed all required forward migrations using the owning bootstrap process; and
4. stored the same 32-to-256-byte registry HMAC secret in macOS Keychain.

The Keychain account defaults to the current macOS account and the service label defaults to
`com.signal-foundry.signalattice-registry`. To create or update that item without placing its value
in shell history or the process list, put `-w` last and enter the value only at the secure prompt:

```bash
security add-generic-password \
  -U \
  -a "$USER" \
  -s com.signal-foundry.signalattice-registry \
  -w
```

For an existing registry, enter the **exact secret used at initialization**. A new or rotated value
will fail authority verification; this schema does not support implicit secret rotation. Never echo
the secret, put it in a URL or CLI argument, commit it, save it in an environment file, or reuse an
API key/idempotency key as the registry authority.

## Start and stop

Start one server over ignored local state:

```bash
signalattice serve-api \
  --registry-db experiments/control-plane/registry.sqlite \
  --cas-root experiments/control-plane/cas \
  --port 8765
```

There is deliberately no host option. The process binds only to `127.0.0.1`, uses one worker, and
refuses ports below 1024. It retrieves the registry secret directly from Keychain, verifies the
existing registry and registry-bound CAS, checks the approved GET-only route inventory, and only
then starts the socket. It never initializes, migrates, repairs, or replaces evidence state.

Stop with the process supervisor's normal termination or `Ctrl-C`. Uvicorn permits up to ten
seconds for graceful shutdown. Do not kill the process during a diagnostic simply to hide an
integrity or readiness failure; preserve the redacted request ID and inspect local storage under the
registry runbook.

The hardened container profile documented in the
[service operations guide](service_operations.md) uses an owner-permissioned Unix-domain socket
instead of TCP, a read-only root filesystem, and a file-mounted runtime secret. The loopback command
above remains the explicit workstation CLI profile; it is not the container default.

## Health and initial verification

Use the numeric loopback listener for operator checks:

```bash
curl --fail --silent --show-error \
  http://127.0.0.1:8765/health/live

curl --fail --silent --show-error \
  http://127.0.0.1:8765/health/ready

curl --fail --silent --show-error \
  http://127.0.0.1:8765/api/v1/openapi.json
```

Liveness is intentionally storage-independent. `{"schema_version":1,"status":"live"}` proves
only that the process can answer. Readiness performs a bounded, read-only registry/CAS probe. A
healthy boundary reports `ready: true` and the closed code `ready`. A 503 means callers must stop
using evidence responses until the underlying condition is resolved; it never authorizes a
fallback to raw SQLite, arbitrary JSON, or unverified files.

The readiness body's `retryable` field is the semantic decision: only bounded busy/unavailable
states may become healthy without operator repair, and only those responses carry
`Retry-After: 1`. Integrity, authority, missing-state, and identity results omit that header and
require operator investigation rather than a retry loop.

Every response carries a server-generated `X-Request-ID`, `Cache-Control: no-store`, a deny-all
content security policy, `X-Content-Type-Options: nosniff`, `Referrer-Policy: no-referrer`, frame
denial, same-origin resource policy, and a restrictive permissions policy.

## Route contract

Only the following GET operations exist:

| Operation ID | Route | Result |
| --- | --- | --- |
| `getLiveness` | `/health/live` | process-only liveness |
| `getReadiness` | `/health/ready` | bounded combined registry/CAS readiness |
| `listRunsV1` | `/api/v1/runs` | snapshot/keyset page of safe run summaries |
| `getRunV1` | `/api/v1/runs/{run_id}` | one run summary and evidence links |
| `listAggregateForecastSummariesV1` | `/api/v1/runs/{run_id}/forecast-summaries` | aggregate-only forecast manifests |
| `listDiagnosticsV1` | `/api/v1/runs/{run_id}/diagnostics` | typed diagnostic manifests |
| `listRunArtifactsV1` | `/api/v1/runs/{run_id}/artifacts` | role-linked path-free metadata |
| `getArtifactMetadataV1` | `/api/v1/artifacts/{artifact_id}` | one path-free artifact projection |
| `listModelCardsV1` | `/api/v1/model-cards?run_id={run_id}` | run-scoped model-card summaries |
| `getModelCardV1` | `/api/v1/model-cards/{card_id}?run_id={run_id}` | one structured model card |
| `getOpenApiV1` | `/api/v1/openapi.json` | deterministic OpenAPI 3.1 contract |

The approved inventory contains no POST, PUT, PATCH, DELETE, broker, order, position, submission,
cancellation, retention, portfolio-control, paper-trading, or live-trading route. Interactive
Swagger and ReDoc endpoints are disabled.

### Run references

Begin with `GET /api/v1/runs`. Its `run_id` values are canonical public references such as
`r1_...`, not raw database identifiers. Follow the relative evidence links returned with the run;
do not hand-encode or decode a reference. References are bounded path segments and aliases are
rejected.

Optional run-list filters are the closed `status` and `provenance` values in the checked OpenAPI
schema. `provenance=registry%2Fverified` selects durable registry evidence;
`provenance=legacy%2Funverified` selects the restricted historical projection. Legacy status,
parameters, tags, metrics, and artifact paths are not promoted to verified evidence.

### Artifact and manifest semantics

Artifact identities are complete lowercase SHA-256 digests. Metadata includes class, size, media
type, creation time, and pin state but never a storage key or bytes. Evidence routes accept only
verified CAS artifacts with media type `application/vnd.signalattice.manifest+json`, exact role,
exact run binding, and schema version 1.
Stored manifests bind the registry's complete internal identifier contract, including `:` and `/`;
HTTP responses replace that value with the canonical `r1_` public reference used by run routes.

Forecast-summary responses state `evidence_granularity: "aggregate"` and
`row_level_available: false`. Each aggregate covers at least two observations and carries a split,
horizon, UTC window, sample count, finite error metrics, and optional paired interval statistics.
This contract cannot represent observation rows.

Diagnostics expose at most 256 finite scalar values in the closed categories `calibration`,
`uncertainty`, `latency`, `lineage`, and `readiness`. Model cards contain ordered structured
plain-text sections; they are never interpreted as Markdown or HTML and contain at most 64 KiB of
cumulative text.

Wrong media types, oversized bytes, digest mismatch, CAS substitution, cross-run identity,
duplicate JSON keys, noncanonical JSON, unknown fields, non-finite numbers, unsupported versions,
newer schemas, or a single-card lookup whose requested ID appears more than once within the bounded
run inventory fail closed as integrity failures.

## Pagination

Run, artifact, forecast-summary, diagnostic, and model-card collections use authenticated
snapshot/keyset pagination:

- `page_size` defaults to 25 and must be between 1 and 100; it is a requested upper bound, not a
  promise that every response contains that many records;
- forecast-summary pages contain at most 7 manifests and diagnostics pages at most 1 manifest so
  every maximum-shape legal page stays below the 2 MiB buffered-response boundary;
- `cursor` is opaque and at most 1,024 characters;
- `next_cursor: null` marks the end of the snapshot; and
- no route computes a total count or accepts an offset.

To continue, send the returned cursor unchanged with the same route, run, page semantics, and
filters:

```bash
curl --fail --silent --show-error --get \
  --data-urlencode "page_size=25" \
  --data-urlencode "cursor=${next_cursor}" \
  http://127.0.0.1:8765/api/v1/runs
```

The first page fixes immutable source high-water marks. Concurrent inserts appear only in a new
traversal and cannot create duplicates in the current one. A cursor is authenticated to its exact
authority and filters; tampering, using it with a different run/role/filter, rolling back a source,
or mutating projected legacy rows fails safely. Cursors are coordination tokens, not durable
research evidence, and clients must not decode them.

Model-card routes require a run reference so the service never performs a global artifact scan. A
single-card lookup is intentionally bounded to at most 100 cards for that run; capacity failure is
safer than an unbounded search.

## Errors

Application and request-boundary failures use bounded RFC 9457 `application/problem+json`
documents. A storage-unready `/health/ready` response is the deliberate exception: it remains a
typed readiness verdict with HTTP 503, not a problem document. It carries `Retry-After: 1` only
when its `retryable` field is true.

```json
{
  "code": "not_found",
  "detail": "The requested evidence resource does not exist.",
  "request_id": "00000000000000000000000000000000",
  "status": 404,
  "title": "Resource not found",
  "type": "urn:signalattice:problem:not_found"
}
```

The shown request ID is illustrative. Runtime IDs are random server-owned 32-character lowercase
hex values. A client-supplied `X-Request-ID` is rejected, not trusted.

| HTTP | Code | Client/operator action |
| --- | --- | --- |
| 400 | `invalid_request` | correct the bounded path/query/cursor/request contract |
| 404 | `not_found` | stop; the safe projection is absent |
| 405 | `method_not_allowed` | use GET only |
| 413 | `request_body_forbidden` | remove the request body and body transport metadata |
| 422 | `request_validation_failed` | correct the typed query/path value |
| 429 | `service_saturated` | honor `Retry-After: 1`; reduce local concurrency |
| 429 | `rate_limited` | honor `Retry-After: 1`; reduce the request rate |
| 503 | `service_draining` | stop new work and allow bounded shutdown to finish |
| 503 | `response_limit_exceeded` | narrow the projection; do not raise the ceiling ad hoc |
| 503 | `evidence_unavailable` | honor `Retry-After: 1`; inspect readiness and storage activity |
| 503 | `evidence_integrity_failed` | stop consuming evidence; preserve state and investigate |
| 500 | `internal_error` | record the request ID; inspect sanitized local logs |

Problem details never echo request values, cursor contents, SQL, paths, credentials, manifest
contents, exceptions, or stack traces. Unexpected failures produce only the closed
`internal_error` outcome and allowlisted route/operation attributes; exception classes and request
IDs do not enter telemetry. Ordinary Uvicorn access logs are disabled, so query strings are not
retained by the service. Every perimeter problem response closes its connection.

## Resource and transport bounds

| Boundary | Default/maximum |
| --- | --- |
| Bind | exactly `127.0.0.1` |
| Workers / protocol | one / h11; WebSockets disabled |
| Application concurrency | 32 globally / 24 data requests; probe headroom reserved |
| Transport concurrency / backlog | 32 / 64 |
| Keep-alive / graceful shutdown | 3 seconds / 10 seconds |
| Admission rate | API 20/s burst 40 / probes and metrics 2/s burst 4 |
| Headers | 64 entries and 16 KiB total; RFC-token names and visible-ASCII values only |
| Path / query | 512 ASCII bytes / 4 KiB and at most 32 components |
| Incomplete h11 event | 16 KiB |
| Request body | forbidden on every route |
| Buffered response / metrics | 2 MiB / 256 KiB and fewer than 600 series |
| Requested page / cursor | at most 100 records / 1,024 characters |
| Effective manifest page | forecast 7 / diagnostics 1; continuation preserves the snapshot |
| Forecast / diagnostics / model-card manifest | 128 KiB / 512 KiB / 96 KiB |
| Diagnostic values / model-card text | 256 / 64 KiB |

Forwarding headers, transfer encoding, content encoding, `Expect`, noncanonical or hostile `Host`,
client request IDs, CORS, proxy trust, reload, and WebSockets are not supported. Do not put this
listener behind a proxy or tunnel; that changes the threat model even if the local bind remains.
The stricter value grammar accepts space and visible ASCII and rejects HTAB, every other C0 byte,
DEL, and non-ASCII `obs-text`; this removes parser-dependent whitespace/control ambiguity. The
decoded path must equal its canonical ASCII wire spelling, so percent-encoded aliases cannot move
fixed operations onto the larger data-route rate budget.

## Deterministic OpenAPI evidence

The checked [OpenAPI artifact](api/openapi-v1.json) is generated from the real application factory,
not maintained by hand. Generation creates a fresh private registry and CAS under a temporary
directory, verifies the combined evidence boundary, asserts the exact GET-only route inventory,
and serializes the schema with sorted keys and finite JSON. It does not read Keychain, operator
state, market data, credentials, or the network. Temporary state is removed when generation ends.
The contract endpoint declares `application/json` and references the OpenAPI Initiative's pinned
2025-11-23 schema for OpenAPI 3.1 documents. Serving or generating it never resolves that external
schema reference over the network.

Regenerate only through an explicit output path:

```bash
python scripts/generate_service_openapi.py \
  --output docs/api/openapi-v1.json
```

Check drift without writing:

```bash
python scripts/generate_service_openapi.py \
  --check docs/api/openapi-v1.json
```

With neither flag, the script writes the canonical document to standard output and does not create
or replace a repository file. Any intended wire change requires review of the generated diff,
versioning consequences, route inventory, clients, and ADR impact. Never hand-edit the checked
artifact.

## Operational failure guide

### Startup refuses the registry authority

Confirm that the Keychain account and service label identify the exact secret originally used for
the registry. Do not rotate or guess the value against existing state. A wrong secret fails closed
and should not be replaced by a less protected CLI or environment fallback.

### Startup reports existing evidence is not ready

Stop and run the registry's bounded readiness and integrity procedures in the owning maintenance
context. Check that the database and CAS are on one supported local filesystem, have exact private
ownership/modes, retain the expected migration ledger and store identity, and are not partial
copies. Do not create missing CAS children, edit the store marker, down-migrate SQLite, adopt a
different CAS, or bypass readiness.

### Readiness becomes unready after startup

Stop evidence-consuming clients. A retryable busy/unavailable code can be retried only after its
`Retry-After` interval and after bounded local contention subsides. Identity or integrity failures
require preservation and investigation, not retry loops.

### Requests return 429

Reduce client parallelism and honor the one-second retry instruction. Do not increase server limits
without measured load evidence, memory accounting, and a reviewed configuration change. The
[bounded service operator guide](service_operations.md) defines measured synthetic load and
telemetry evidence, candidate SLO semantics, containment, and residual limits.

### A cursor is rejected

Restart the traversal from its first page with the intended filters. Do not decode or modify the
cursor. Repeated rejection can indicate the wrong registry authority, source rollback, filter
reuse, or mutation of legacy state; preserve that evidence for diagnosis.

## Security checklist

Before each operator session:

- verify the listener is exactly `127.0.0.1` and do not expose it through a proxy, tunnel, port
  forward, container publish, or shared host;
- use only the approved Keychain secret retrieval; never place the credential in arguments, URLs,
  logs, environment files, or repository state;
- verify `/health/ready` before consuming evidence and stop on any integrity result;
- keep registry/CAS directories owner-only and on one supported local filesystem;
- treat cursors, stored manifests, registry rows, and CAS bytes as untrusted;
- keep licensed observations, private account data, API keys, models, and broker credentials out of
  the served manifest families; and
- do not infer paper/live readiness, capital authorization, or expected profit from API
  availability or historical aggregate evidence.

Loopback and browser same-origin policy are not authentication. Another process running as the same
OS user can query this endpoint, and a compromised owner account can attack both service and local
storage. Remote, multi-user, or same-origin console operation requires a superseding authenticated
transport design. The dedicated local container remains single-owner and socket-bound; do not
weaken either transport profile locally.

## Rollback

Stop `serve-api` and remove the optional service dependencies if the HTTP surface must be disabled.
The core pipeline, durable registry, and CAS remain usable through their existing non-HTTP
boundaries. Do not down-migrate storage, delete artifacts, expose a replacement remote listener,
enable interactive docs, or serve untyped legacy/report data as a fallback.
