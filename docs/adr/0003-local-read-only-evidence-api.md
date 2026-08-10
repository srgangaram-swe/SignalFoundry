# ADR 0003 — Local-only, read-only forecast-evidence API

- **Status:** Accepted
- **Date:** 2026-08-09
- **Work item:** SF-S5-SL-MR2 ([#17](https://github.com/srgangaram-swe/Signalattice/issues/17))
- **Depends on:** [ADR 0002](0002-durable-local-registry.md)
- **Supersedes:** none

## Context

Signalattice now has a durable local registry, an immutable content-addressed artifact store (CAS),
and framework-neutral bounded read ports. A local operator console and later service-operability
work need one stable HTTP contract over that evidence. Exposing SQLite, CAS paths, generic JSON, or
the legacy tracker directly would couple clients to persistence, bypass artifact verification, and
risk reflecting paths, parameters, tags, licensed rows, malformed values, or other sensitive state.

This repository does not yet persist observation-level forecast history. The available safe
boundary consists of run summaries, path-free artifact metadata, aggregate forecast evaluation,
typed diagnostics, and structured model cards. A route that appeared to serve forecasts but
actually reconstructed rows from reports or arbitrary artifacts would create a false contract and
an unsafe parser boundary. Pre-outcome rows and delayed reconciliation remain deferred to
[#21](https://github.com/srgangaram-swe/Signalattice/issues/21).

The first HTTP slice is single-owner laptop infrastructure. It must remain reproducible and useful
without a remote database, gateway, TLS certificate, identity provider, broker connection, market
data request, or cloud deployment. That narrow environment does not make local HTTP inherently
trusted: request metadata, cursor tokens, stored manifests, registry state, and CAS bytes remain
untrusted and resource-bounded.

## Decision

### Adapter and dependency direction

Implement a FastAPI adapter whose application factory receives an already assembled evidence-read
port. Every route uses only these framework-neutral operations:

- combined registry/CAS readiness;
- get/list safe run projections;
- get/list path-free artifact projections; and
- size- and media-type-constrained verified manifest reads.

The HTTP package does not issue SQL, traverse storage paths, initialize storage, run migrations,
publish artifacts, or perform lifecycle transitions. Storage bootstrap remains explicit under ADR
0002. Service startup opens an existing registry with its matching local HMAC authority, obtains
the registry-bound CAS identity, reopens that exact CAS, and refuses to listen unless the complete
evidence boundary is ready. In expected-identity mode, CAS initialization is a reopen-only
verification operation: missing or partial state is not created or repaired.

FastAPI, Starlette, and Uvicorn live in the optional locked `service` dependency extra. Core and
offline installations do not import that framework surface until the operator selects `serve-api`.
The current operator assembly is macOS-local because it retrieves registry authority through the
macOS `security` command; another platform needs a separately reviewed secret-provider adapter.
The server uses a single audited Uvicorn configuration rather than exposing general framework
settings through the CLI.

### Versioned, closed projections

The public prefix is `/api/v1`. Every response model is an immutable strict Pydantic-v2 contract:
unknown fields are forbidden, coercion is disabled, non-finite numbers are rejected, timestamps
are UTC-aware, identifiers and text are byte-bounded, and JSON is finite. Stable explicit operation
IDs make generated clients and contract diffs reviewable.

Internal durable run identifiers do not become path text. Run responses expose a canonical
base64url reference prefixed by `r1_`; decoding rejects padding, aliases, non-ASCII text, excessive
length, and invalid underlying identifiers. Artifact identities remain complete lowercase SHA-256
digests. Artifact responses include immutable metadata only—class, byte size, media type, creation
time, and pin state—and never include a storage key, absolute path, file bytes, download link, or
model binary.

Registry-native runs are labelled `registry/verified`. Historical tracker rows are labelled
`legacy/unverified` and use ADR 0002's restricted projection. Arbitrary legacy parameters, tags,
metrics, artifact path strings, malformed JSON, and non-finite values never cross HTTP. A malformed
or ambiguous legacy row fails the request closed.

Within version 1, changes must preserve the documented wire meaning and pass a checked OpenAPI
diff. Adding a required request value, removing or renaming a field, narrowing an existing value
domain, changing pagination or error semantics, or reinterpreting evidence requires a new API
major version. `schema_version: 1` travels with versioned response and manifest contracts. The
checked [OpenAPI 3.1 artifact](../api/openapi-v1.json) is generated from the application factory;
it is evidence of the contract, not a separate source of truth.

### Aggregate manifests only

Evidence routes parse only immutable CAS artifacts with media type
`application/vnd.signalattice.manifest+json`. The artifact role selected by the route, requested
run, manifest `kind`, manifest `run_id`, and schema version must all agree. Parsing requires the
canonical representation and rejects duplicate keys, unknown fields, unsupported/newer schemas,
wrong media types, wrong roles, cross-run substitution, excessive nesting, excessive bytes,
non-finite values, and noncanonical timestamps.
The stored manifest binds the registry's complete internal run identifier; the HTTP projection
replaces it with the canonical `r1_` public reference before serialization.

Three closed version-1 manifest families are accepted:

- `forecast_summary`: at most 100 sorted aggregate evaluations, each representing at least two
  observations; `evidence_granularity` is exactly `aggregate` and `row_level_available` is exactly
  `false`;
- `diagnostics`: at most 256 sorted finite scalar values across calibration, uncertainty, latency,
  lineage, and readiness categories; and
- `model_card`: ordered structured plain-text sections with required intended-use, out-of-scope,
  data, evaluation, limitations, and monitoring content, with at most 64 KiB of cumulative text.

Verified-read byte ceilings are 128 KiB for a forecast-summary manifest, 512 KiB for diagnostics,
and 96 KiB for a model-card manifest. The service never parses Markdown or HTML, arbitrary report
JSON, serialized estimators, joblib/pickle, raw observations, provider rows, ticker-level
predictions, or licensed market data at request time.

### Snapshot/keyset pagination

List routes use the authenticated snapshot/keyset pagination owned by ADR 0002. Requested page
size defaults to 25 and is capped at 100. It is an upper bound: forecast-summary pages are capped
at 7 and diagnostics pages at 1 so a maximum-shape legal response stays below the 1 MiB envelope.
The smaller page retains an authenticated continuation. A continuation is opaque, at least 16 and
at most 1,024 characters,
authenticated to the local registry authority, snapshot high-water marks, and exact filters. Run
cursors bind status and provenance; artifact cursors bind the run and exact role filter.

The first page fixes the source snapshot. Rows inserted later do not appear in its continuation;
the client begins a new traversal to observe them. Continuations reject tampering, wrong authority,
filter or run reuse, source regression, stale mutable legacy state, and unsafe source data. This
avoids duplicates and gaps under supported concurrent insertion without offset scans or expensive
total counts. Clients must not decode, synthesize, store as durable evidence, or reuse a cursor
with different filters.

Model-card collection routes require a run reference. A single-card lookup scans exactly one
bounded 100-item run-scoped page before returning a result, rejects duplicate semantic card IDs as
integrity failure, and returns capacity failure if an additional page exists; it never turns a card
identifier into an unbounded global artifact scan.

### Error semantics

Application and request-boundary failures use bounded `application/problem+json` following
[RFC 9457](https://www.rfc-editor.org/rfc/rfc9457.html). The document contains a stable Signalattice
problem URI, title, HTTP status, non-reflective detail, stable code, and a 128-bit server-generated
request ID. Request values, cursor contents, filesystem paths, SQL, manifest contents, credentials,
exceptions, and stack traces are not reflected. An unready `/health/ready` response is a typed
health-state verdict rather than an exception: it retains the `ReadyResponse` schema with HTTP 503
and carries `Retry-After: 1` only for a retryable busy/unavailable code.

The closed mapping is:

| Status | Stable code | Meaning |
| --- | --- | --- |
| 400 | `invalid_request` | malformed scope, cursor, public reference, or bounded request |
| 404 | `not_found` | requested safe projection does not exist |
| 405 | `method_not_allowed` | any method other than GET |
| 413 | `request_body_forbidden` | non-empty body on a GET request |
| 422 | `request_validation_failed` | typed query or path validation failed |
| 429 | `service_saturated` | application concurrency ceiling reached; retry after 1 second |
| 503 | `evidence_unavailable` | busy, timed-out, or unready storage; retry after 1 second |
| 503 | `evidence_integrity_failed` | non-retryable CAS or typed-manifest verification failure; stop and investigate |
| 500 | `internal_error` | unexpected bounded application failure |

Not-found responses do not reveal whether an omitted value failed in SQLite, the CAS, or manifest
selection. Integrity errors remain distinct from ordinary absence so an operator does not treat
corrupt evidence as a harmless cache miss.

### Loopback serving and ASGI security boundary

The only accepted bind host is the IPv4 loopback address `127.0.0.1`. It is an invariant in the
frozen server configuration, not a CLI default that can be overridden. The server runs one worker
with h11, WebSockets disabled, reload disabled, proxy headers disabled, an empty forwarding
allowlist, access logging disabled, interactive Swagger/ReDoc disabled, and server/date headers
disabled. `Host` must be exactly `127.0.0.1` or `localhost`, optionally followed by the configured
canonical nonzero port; a different explicit port is rejected.

A raw ASGI middleware surrounds FastAPI so invalid traffic and framework failures share the same
redacted boundary. It:

- accepts GET only and rejects request bodies, transfer encoding, content encoding, `Expect`,
  forwarding headers, and client-supplied request IDs;
- caps the path at 512 ASCII bytes, query string at 4 KiB and 32 components, headers at 64 entries
  and 8 KiB total, the h11 incomplete event at 16 KiB, and a buffered response at 1 MiB;
- caps application work at 32 concurrent requests, while Uvicorn caps transport concurrency at 64
  with a backlog of 64, three-second keep-alive, and ten-second graceful shutdown;
- generates an independent 128-bit request ID and logs only that ID plus the exception class for an
  unexpected application failure; and
- overwrites response security policy with `Cache-Control: no-store`,
  `X-Content-Type-Options: nosniff`, `Referrer-Policy: no-referrer`, a deny-all content security
  policy, same-origin resource policy, restrictive permissions policy, frame denial, and the
  server-owned request ID.

No CORS middleware is installed. Browser scripts on other origins therefore cannot read responses,
but this is not authentication and does not defeat malware or another process running as the same
user. Access logging is disabled so query cursors and identifiers are not copied into ordinary
server logs.

### Health semantics

`/health/live` proves only that the application can answer; it never touches storage. It must stay
live when the registry or CAS is unavailable so a supervisor can distinguish process failure from
evidence failure.

`/health/ready` performs the bounded, read-only combined registry/CAS probe. It validates known
registry schema and authority, CAS presence and initialization, immutable store identity and
registry binding, and returns a closed path-free code. It never creates, migrates, repairs, binds,
publishes, or sweeps state. An unready verdict returns 503 and must remove the service from use;
only a retryable busy/unavailable result carries `Retry-After`. This is not permission to return
unverified evidence.

## Security analysis

Threats include DNS-rebinding-style hostile `Host` values, accidental remote binding, proxy-header
spoofing, method confusion, request smuggling metadata, oversized paths/headers/query strings,
body-based resource abuse on GET, response amplification, cursor tampering, identifier traversal,
legacy-data reflection, non-finite JSON, malicious manifests, cross-run artifact substitution,
CAS corruption, exception disclosure, and accidental mutable or trading routes.

Controls are immutable configuration, exact loopback binding, raw ASGI validation, explicit route
inventory, generated operation IDs, strict projections, authenticated cursors, full content
identities, media/kind/run/schema binding, canonical bounded parsing, verified CAS reads, redacted
problem documents, owner-only storage, Keychain retrieval without shell interpolation, minimal
optional dependencies, and no mutation/broker authority. The route-inventory assertion fails
startup and contract generation if an unapproved or non-GET route appears.

The service has no application authentication because it is constrained to a single-user loopback
profile over owner-protected local evidence. This is an intentionally narrow exception, not a
general authorization architecture. Remote binding, a shared workstation trust model, a container
listener, TLS termination, same-origin browser deployment, or another user requires the superseding
threat model and authenticated gateway planned in
[#20](https://github.com/srgangaram-swe/Signalattice/issues/20). Adding a bearer token while
retaining an unsafe listener would not satisfy that requirement.

Residual risks remain:

- any process able to act as the same OS user can query the loopback endpoint and may also attack
  the underlying owner-accessible files;
- loopback does not provide TLS, origin authentication, multi-user authorization, or protection
  from a compromised host;
- a hostile browser origin can attempt a GET even though same-origin policy prevents script access
  to the response; there are no mutable routes to exploit through cross-origin submission;
- buffering responses to enforce a hard ceiling consumes up to the configured bound per active
  request;
- cooperative SQLite and CAS deadlines do not preempt arbitrary in-process framework code;
- this slice establishes neither production load evidence nor telemetry/SLOs; and
- verified aggregate historical evidence is not current market data, paper-trading evidence,
  capital authorization, a profitability guarantee, or live-trading readiness.

## Alternatives considered

**Expose SQLite or a generic object endpoint.** Rejected. It would leak persistence semantics,
permit unbounded/ad hoc queries, and bypass safe legacy and artifact projections.

**Serve arbitrary JSON, Markdown, HTML reports, or artifact bytes.** Rejected. Content identity does
not make a parser or renderer semantically safe, and download routes increase disclosure and
resource-abuse risk.

**Bind remotely and add a token.** Rejected for this slice. Remote serving needs TLS, identity,
authorization, secret rotation, rate/load controls, telemetry, deployment hardening, and a distinct
threat model; a token alone does not supply those controls.

**Use offset pagination and totals.** Rejected because concurrent insertion can create duplicates
or omissions and total scans defeat the bounded local read contract.

**Add a service database or denormalized cache.** Rejected. The framework-neutral registry/CAS
ports already provide the required evidence; another persistence layer would create consistency,
migration, and rollback complexity without an established scale need.

**Run multiple workers or background tasks.** Rejected. The local SQLite/CAS boundary and current
evidence volume do not justify duplicated process state, cross-worker capacity accounting, or
background mutation.

## Consequences

- Console and automation clients receive one deterministic, path-free, versioned read contract.
- HTTP cannot mutate jobs, evidence, orders, positions, or trading state.
- Strict manifests require producers to publish explicit aggregate schemas instead of relying on
  informal reports.
- Snapshot pagination remains stable under supported concurrent insertion at the cost of opaque,
  authority-specific cursors and no total count.
- The optional service extra increases dependency and vulnerability-review surface only for users
  who install it.
- The current CLI service assembly requires macOS Keychain; support for another secret provider is
  an explicit portability change, not an environment-variable fallback.
- Loopback-only operation is deliberately less convenient than remote access; production service
  controls remain separate work.

## Rollback and containment

Stop the `serve-api` process and remove or disable that CLI entry point. Core pipeline and registry
operation continue without the optional service extra. Revert the additive service adapter,
projection contracts, checked OpenAPI document, and service documentation if required.

Rollback must not down-migrate or rewrite the registry, alter the CAS identity, delete evidence,
expose a different listener, enable interactive docs, fall back to arbitrary legacy JSON, or replace
typed manifest failures with unverified output. Preserve storage and redacted diagnostics for
investigation. A wire-incompatible replacement uses a new versioned route and a superseding ADR.

## Explicit non-goals

Remote/public exposure, TLS, gateway authentication, multi-user authorization, same-origin console
deployment, production telemetry/SLOs/load evidence, container hardening, row-level forecasts,
delayed reconciliation, background execution, ingestion, mutation, artifact download, broker
integration, orders, positions, portfolio controls, paper/live trading, capital authorization,
production readiness, or profitability claims.
