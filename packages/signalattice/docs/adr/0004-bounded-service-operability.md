# ADR 0004: Bound the local evidence service before operational use

- **Status:** Accepted
- **Date:** 2026-08-09
- **Issue:** [#20](https://github.com/srgangaram-swe/Signalattice/issues/20)

## Context

ADR 0003 established a loopback-only, read-only HTTP projection over the durable registry and
content-addressed store. Correct projections and redacted failures are necessary, but they do not
by themselves establish a credible operating envelope. Unbounded admission, high-cardinality
telemetry, blocking exporters, or an over-privileged image could turn a read-only endpoint into a
resource-exhaustion or disclosure boundary.

This repository needs an inspectable local deployment profile for later shadow and governance
work. The profile must preserve the registry/CAS integrity boundary and version-1 HTTP contract. It
is not a remote service, a production deployment, a paper- or live-trading system, or evidence of
profitability or market-scale capacity.

## Decision

### Admission and response bounds

Admission is layered across the pinned Uvicorn protocol adapter and the outer ASGI boundary. Both
use fixed, process-global state rather than attacker-keyed maps:

- Uvicorn is configured at 32; the pinned h11 adapter admits exactly 32 normal parsed exchanges and
  sends the next one through a bounded structured rejection task, while the outer boundary mirrors
  the 32-exchange ceiling and limits data routes to 24, reserving probe capacity;
- API routes use one 20-request/second token bucket with burst 40; probes and metrics share a
  separate 2-request/second bucket with burst 4;
- every parsed request acquires its fixed lane before semantic validation, canonical decoded
  operation paths consume the operations lane, and percent-encoded aliases are rejected rather
  than receiving a larger budget;
- excess rate and concurrency fail promptly with closed 429 `rate_limited` or
  `service_saturated` problem codes and `Retry-After`; 503 remains reserved for draining and
  unavailable, integrity, or response-boundary failures; every perimeter problem response closes
  its connection and the service performs no internal request retry;
- query strings are capped at 4 KiB, cursors at 1 KiB, headers at 16 KiB, model-card text at
  64 KiB, projected responses at 2 MiB, and metrics exposition at 256 KiB;
- request field names are RFC tokens while values are limited to space and visible ASCII (HTAB,
  other C0 bytes, DEL, and `obs-text` are rejected); admission is held before waiting for the first
  ASGI body event; and
- every body is rejected, including transfer-encoded, content-encoded, and deferred ASGI body
  messages.

SQLite busy waits, progress deadlines, parameterized queries, page ceilings, and verified CAS
reads remain independent lower-layer controls. HTTP cancellation is not treated as a database
deadline or an integrity control.

Pinned Uvicorn 0.52.1 is configured with `limit_concurrency=32`, but its native comparison includes
the current connection, rejects normal exchange 32 rather than 33, and emits an uninstrumented
`text/plain` response outside ASGI. A small pinned h11 adapter preserves parser/lifecycle behavior,
uses the configured value as an exact pre-ASGI gate, and temporarily bypasses only that raw native
responder. It admits 32 normal exchanges and routes exchange 33 through a short canonical rejection
task with problem details, telemetry, connection close, and `Retry-After`; the outer
`AdmissionController` independently mirrors the global 32 ceiling and owns the 24-data split and
fixed rate buckets. Real-socket tests put one 250-ms deadline around connection, write, and complete
structured rejection, then prove task/connection accounting returns to baseline. The same adapter
converts h11 parser failures—including fragmented oversized headers—to a redacted structured 400
and closed unmatched/invalid-metadata telemetry event. Changing the Uvicorn minor line requires
revalidating these adapter and socket tests.

### Privacy-safe telemetry

Metrics, spans, and structured logs use reviewed route templates and closed operation, outcome,
and rejection enums. They never carry a URL, query, header, cookie, client address, user agent,
run/model/forecast/ticker identifier, market value, artifact path, exception text, credential, or
caller-provided label. Cardinality is a design invariant: fewer than 600 metric series under
adversarial high-uniqueness requests.

Every process retains a 512-record ring of canonical structured logs and spans even when external
export is disabled. Oldest-record eviction and sink failure increment distinct closed counters;
startup, ready, draining, stopping, and stopped are explicit local lifecycle events. There is no
fallback to ad-hoc plaintext exception logging.

External trace/log delivery uses bounded non-blocking queues, bounded attempts, and bounded shutdown
flushes. Export is disabled by default. A rejected, unavailable, slow, or malformed exporter
increments a closed drop reason but cannot fail or delay an admitted evidence request. Exporter
configuration accepts only HTTPS without URL credentials, query, or fragment, plus cleartext HTTP
to literal loopback IP addresses; resolver names such as `localhost` are rejected. TLS verification
cannot be disabled. Every injected transport request carries `follow_redirects=false`; this sprint
bundles and qualifies no concrete remote transport, so honoring that invariant remains the embedding
transport's responsibility.

Shutdown applies two independent ceilings: at most 2 seconds to flush queued records and, after
cancellation, at most 250 milliseconds to observe worker termination. A worker exceeding the
second deadline increments `signalattice_telemetry_worker_termination_failures_total`, leaves the
lifecycle in `termination_timeout`, and requires supervisor termination rather than an invented
clean-shutdown claim. Python cannot preempt arbitrary in-process code that deliberately suppresses
cancellation; remote export therefore remains disabled in the standard assembly and no concrete
remote transport is qualified by this sprint.

The Prometheus-compatible endpoint is `GET /internal/metrics`. It is omitted from OpenAPI, served
only through the local trust boundary, rate limited, size capped, and not an authorization
mechanism. Uvicorn access logs remain disabled.

### Deployment and supply chain

The dedicated service image defaults to the permissioned Unix-domain socket
`/run/signalattice/api.sock`; the default Compose profile publishes no TCP port and permits no
outbound network. Signalattice binds that socket itself under umask `0077`, narrows it to mode
`0600`, verifies owner/type/link identity, passes the already-listening descriptor to Uvicorn, and
unlinks only the unchanged device/inode it created. This avoids Uvicorn 0.52.1's absent-UDS
`chmod(0666)` behavior without a post-start permission race. The final image runs as UID/GID 10001
with `/usr/sbin/nologin` and no writable home; the Debian base still contains `/bin/sh`, but neither
the service command nor healthcheck invokes it. The root filesystem and registry/CAS mounts are
read-only, writable space is one bounded `noexec,nosuid,nodev` tmpfs, all capabilities are dropped,
and `no-new-privileges`, default seccomp, one worker, and explicit CPU, memory, PID,
file-descriptor, and shutdown ceilings apply.

The service builder does not inherit the research CLI environment. It synchronizes only the locked
`service-runtime` dependency group without installing project metadata, then the final image copies
the exact service and durable-read modules needed by `python -m quant_platform.service`. The
tracking package initializes its legacy experiment adapters lazily, so importing registry reads no
longer imports NumPy, pandas, YAML, Typer, Rich, or the broader research graph. CI verifies the
closed distribution/module inventory and globally rejects fixture/data/model/database/report
artifacts rather than relying only on a first-party path check.

Build inputs and Actions are immutable; Python resolution comes from `uv.lock`; the final stage has
no compiler. CI compares application/configuration digests from two clean builds, caps the
uncompressed image at 1.25 GiB, records the first-party telemetry source-artifact footprint, and
requires a compatible fresh-process telemetry-enabled-minus-disabled peak-RSS delta between zero
and 100 MiB. It also produces SPDX 2.3 SBOM,
provenance, vulnerability, secret, license, misconfiguration, history, and final-filesystem
evidence. High/Critical findings, secrets, incompatible licenses, material misconfiguration,
scanner/database failure, or unexplained scan exceptions fail closed. This decision neither
publishes nor signs an image.

### Evidence and candidate SLOs

The repository commits only redistribution-safe aggregate results from a deterministic,
network-independent synthetic workload. Raw request samples remain untracked or become bounded CI
artifacts. The workload includes cold construction, a steady route mix, maximum pages and
diagnostics, telemetry disabled/enabled, saturation, concurrent metrics scraping, and injected
storage/export faults. It records latency distributions, outcomes, throughput, bytes, CPU/wall
time, RSS, file descriptors, threads/processes, rejections, telemetry drops, metric series,
environment, seed, warm-up, and sample counts. A Seaborn plot is generated directly from that JSON.

The 28-day objectives in the operator guide are **candidate** SLOs. Local synthetic observations
do not prove them. CI fails on deterministic correctness and resource bounds, not on noisy laptop
percentiles.

## Consequences

Positive consequences:

- excess work receives deterministic bounded rejection instead of consuming unbounded resources;
- telemetry remains useful without creating an identifier or credential exfiltration surface;
- probes remain available under data-route saturation;
- the local container boundary and its dependency/image inputs are independently auditable; and
- claims can be traced to aggregate evidence with explicit environment and limitations.

Costs and tradeoffs:

- fixed global buckets intentionally sacrifice per-client fairness and multi-tenant semantics;
- response buffering consumes up to the declared ceiling for each admitted request;
- one process and one local socket provide neither high availability nor horizontal scale;
- the native Uvicorn responder is not a safe structured-response boundary; the pinned adapter owns
  the exact configured request transition. Partially formed or idle connections exist before that
  transition and consume the separately bounded h11 bytes, socket backlog, and container file
  descriptors until they close;
- the local ring may evict oldest records and bounded export queues may drop new records under
  pressure, so separate drop counters and shutdown summaries are required; and
- an injected in-process exporter that suppresses cancellation can outlive the application-level
  termination deadline; the degraded lifecycle and metric expose this state, but the process
  supervisor remains the final hard-stop boundary; and
- local timing distributions are useful regression evidence but cannot establish a 28-day SLO or
  market-scale performance.

## Alternatives considered

### Per-client limits

Rejected. Client identity is unauthenticated in this local profile, while a map keyed by hostile
headers or addresses creates cardinality and memory-denial risk.

### Blocking, lossless telemetry

Rejected. A local exporter must not become part of request correctness or availability. Durable
business evidence belongs in the registry/CAS; telemetry is bounded operational evidence.

### Remote TCP with a bearer token

Rejected. Remote service authority requires a superseding design for TLS, authenticated identity,
authorization, secret rotation, gateway controls, audit, and incident response. A token alone does
not provide those properties.

### Gate CI on local latency percentiles

Rejected. Shared-runner and laptop latency is noisy. CI instead enforces deterministic ceilings,
status semantics, cardinality, byte bounds, non-disclosure, and reproducible evidence structure.

## Rollback

Revert this slice and remove only its locally built service image and generated local raw benchmark
samples. The MR1 registry/CAS and MR2 version-1 API contracts remain intact. Disable any configured
exporter first; no registry, artifact, credential, source data, model, order, or capital state is
modified by rollback.
