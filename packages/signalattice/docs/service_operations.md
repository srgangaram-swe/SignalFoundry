# Bounded local evidence-service operations

This guide operates Signalattice's read-only evidence API inside its deliberately narrow local
trust boundary. Read [ADR 0004](adr/0004-bounded-service-operability.md), the
[service API contract](api_service.md), and the [service threat model](threat_model.md) before
changing the profile.

This profile is synthetic/local engineering infrastructure. It is not evidence of production,
paper-trading, live-trading, profitability, market-scale capacity, remote-service security, or a
proven service-level objective (SLO). It has no broker, order, position, or capital authority.

## Trust boundary and fixed budgets

The supported profile is one owner-controlled process reading one owner-controlled local SQLite
registry and CAS. Another process running as the same operating-system user remains inside the
trust boundary. A shared host, remote client, TCP listener, reverse proxy, network filesystem, or
multi-user authorization model requires a new threat model.

| Boundary | Fixed ceiling or policy |
| --- | --- |
| Transport concurrency | Uvicorn configured at 32; pinned h11 gate admits 32 normal exchanges and structurally rejects the next; outer ASGI gate independently mirrors 32 |
| Data-route concurrency | 24 exchanges at the outer gate; probe capacity remains reserved |
| API admission | 20 requests/second, burst 40, one global token bucket |
| Probe and metrics admission | 2 requests/second, burst 4, one global token bucket |
| Query / cursor | 4 KiB / 1 KiB |
| Request headers | 16 KiB after canonical byte accounting |
| Request body | Forbidden, including transfer/content encoding |
| Model-card text | 64 KiB |
| Projected response | 2 MiB |
| Metrics exposition | 256 KiB and fewer than 600 series |
| Telemetry record / fields | 8 KiB / at most 32 reviewed fields |
| Process-local telemetry | 512-record canonical JSON ring; at most 4 MiB by construction |
| Export | Disabled by default; bounded, non-blocking, allowlisted destination |
| Process / telemetry workers / retry | One Uvicorn process worker; one exporter worker per channel (two total); no internal request retry; 2-s flush plus 250-ms worker-termination ceilings |
| Telemetry RSS comparison | Fresh-process enabled-minus-disabled peak delta, clamped to zero only for the overhead guard; at most 100 MiB |

The fixed buckets are capacity controls, not user fairness. Every parsed request consumes its
closed lane before semantic validation; canonical decoded probe paths use the operations lane and
percent-encoded aliases are rejected. A 429 carries either `rate_limited` or
`service_saturated`; a 503 carries a closed draining, evidence-unavailable, integrity, or response-
boundary code. Clients must inspect that code instead of inferring cause from status alone. Every
perimeter problem response closes the connection, and only a retryable response carries
`Retry-After`. Never add a retry loop that can amplify saturation.

## Startup and readiness

1. Verify the registry and CAS are existing owner-only local state. Never initialize, migrate,
   repair, or adopt them from the service process.
2. For the workstation profile, retrieve the registry authority directly from macOS Keychain
   through the documented argument-vector path. For the container profile, pass the same credential
   only to the single bounded Compose process; do not `export` it or write it to a repository,
   environment file, command argument, log, or image layer:

   ```bash
   SIGNALATTICE_REGISTRY_DIGEST_KEY="$(
     security find-generic-password \
       -a "$USER" \
       -s com.signal-foundry.signalattice-registry \
       -w
   )" docker compose up --build service
   ```

   Compose creates `/run/secrets/registry-digest-key` inside the service as UID/GID 10001, mode
   `0400`; it does not add the credential to the container environment. The service independently
   validates file type, ownership, links, size, content, and permissions before opening evidence.
   The container profile requires a 32--256-byte newline-free credential representable by the host
   process environment; use a random hex/base64 authority, never an API key.
3. Start one Uvicorn process worker under the dedicated service profile. The default container profile creates the
   owner-permissioned Unix-domain socket `/run/signalattice/api.sock`, publishes no TCP port, and
   has no outbound network. Signalattice pre-binds the socket as the service owner under umask
   `0077`, verifies exact mode `0600` plus type and filesystem identity, passes the already-listening
   descriptor to Uvicorn, and removes only that unchanged identity during shutdown. Do not
   pre-create, replace, or chmod the socket; stale state requires explicit operator inspection and
   removal.
   The image enters through `python -m quant_platform.service`, not the broad research CLI. Its
   builder synchronizes only the locked `service-runtime` dependency group and copies an explicit
   service/storage module allowlist. Treat any numerical research package, legacy experiment
   adapter import, fixture, dataset, model, report, database, or unapproved module in that image as
   a release-blocking inventory failure.
   Startup, ready, draining, stopping, and stopped transitions are retained as closed local
   lifecycle records. Missing or out-of-order transitions fail the lifecycle integration gate.
4. Use the socket-aware healthcheck. `/health/live` proves only that the process can answer;
   `/health/ready` performs the bounded registry/CAS verification. Never route evidence reads on a
   non-ready verdict.
5. Confirm `/internal/metrics` is absent from the OpenAPI document, contains fewer than 600 series,
   is no larger than 256 KiB, and contains no caller/business identifiers. The endpoint is local
   operational evidence, not authentication.

A host checkout may retain MR2's explicitly loopback-only diagnostic profile. Do not expose that
listener outside the owner-controlled host. Container operation uses the Unix socket by default.

## Telemetry configuration

Telemetry uses route templates and closed outcome, operation, and rejection values. Uvicorn access
logging remains disabled. Logs, spans, and metrics must not include raw paths, URLs, query strings,
headers, cookies, client addresses, user agents, run/model/forecast/ticker identifiers, market
values, artifact paths, exception messages, credentials, or arbitrary labels.

The standard CLI/container assembly always keeps a 512-record process-local ring of canonical JSON
logs and spans. It evicts the oldest record at capacity, increments a fixed channel/reason counter,
and is diagnostic evidence rather than a durable audit store. Serialization/sink failures have a
separate closed counter and never fall back to plaintext Python logging.

External export remains disabled in that standard assembly. The narrow programmatic adapter can be
enabled only by an embedding owner-controlled assembly that supplies both an allowlisted destination
and an injected transport; this sprint deliberately bundles no collector, network transport, token,
or remote endpoint configuration. At that adapter boundary:

- HTTPS is required for non-loopback destinations;
- cleartext HTTP is accepted only for literal loopback IP addresses; resolver names such as
  `localhost` are rejected;
- URL credentials, query strings, fragments, and disabled TLS verification are rejected;
- every injected transport receives `follow_redirects=false` and must honor it; no concrete remote
  transport is bundled or independently qualified by this sprint; and
- queue capacity, whole-attempt timeout, export attempts, shutdown flush, and post-cancel worker
  termination are independently bounded.

An exporter failure must increment a closed drop/failure reason and leave the admitted response
unchanged. Investigate unexplained drops; do not make request completion wait for an exporter.
If a worker misses its 250-ms termination deadline,
`signalattice_telemetry_worker_termination_failures_total` increments and the local lifecycle ends
at `termination_timeout`, not `stopped`. Treat that as degraded shutdown and let the supervisor
enforce its process grace period. Python cannot preempt an injected in-process transport that
deliberately suppresses cancellation, which is one reason no remote transport is enabled or
qualified in the standard profile.

At a flush deadline, records still waiting in a queue are definite `shutdown_timeout` drops.
Records already handed to an injected exporter are instead counted by
`signalattice_telemetry_delivery_indeterminate_records_total`: the runtime cannot honestly know
whether a cancellation-suppressing transport delivered them. A non-terminating worker therefore
keeps the exporter state at `stopping`, records `termination_timeout`, and never emits `stopped`.
The worker exits without taking another batch if the transport later cooperates; until then the
supervisor's process deadline is the containment boundary.

Request field names must be RFC token bytes. Values use this profile's intentionally stricter
ASCII policy: space and visible bytes `0x20`--`0x7e` are accepted, while HTAB, every other C0 byte,
DEL, and non-ASCII `obs-text` are rejected. H11 permits several of those control values, so both a
real h11 parse-to-ASGI test and direct hostile-ASGI tests enforce the tighter application boundary.

## Candidate 28-day SLO semantics

These are aspirational objectives for a future continuous shadow/paper campaign. The committed
synthetic benchmark cannot prove any 28-day objective.

The rolling window is the immediately preceding 28 complete UTC days. A request is *eligible* only
when it reached the application through the supported local profile and passed method, host,
metadata, body, rate, and concurrency admission. Deliberately rejected excess load, unsupported
methods, malformed requests, operator-declared maintenance windows, and benchmark/fault-injection
traffic are excluded. Storage busy, timeout, corruption, and internal failures remain included for
an eligible request; an operator may not relabel them as client failures.

| Candidate | Numerator | Denominator / population | Alert semantics |
| --- | --- | --- | --- |
| Availability >= 99.9% | Eligible requests returning their documented success status | All eligible admitted requests | Page when both a 5-minute fast-burn and 1-hour confirmation window consume the 28-day error budget at >=14.4x; ticket on 6-hour >=1x burn. |
| Interactive p95 <= 100 ms; p99 <= 250 ms | End-to-end monotonic duration from admission through final response byte | Successful eligible `/api/v1/runs`, single-run, artifact-metadata, model-card, and forecast-summary requests, partitioned by route template | Page only after both 5-minute and 1-hour windows exceed the objective with at least 100 samples; ticket if a complete 6-hour window exceeds it. |
| Maximum-page/diagnostic p99 <= 500 ms | Same duration | Successful eligible requests for maximum legal pages and diagnostic projections, with at least 100 observations | Ticket after a complete 6-hour breach; page on simultaneous 5-minute and 1-hour breach once the sample floor is met. |
| Readiness p99 <= 100 ms | Same duration | Every supported `/health/ready` response, ready or unready; liveness is reported separately | Page on simultaneous 5-minute and 1-hour breach with at least 100 samples or any bounded-readiness deadline violation. |
| Corrupt/unverified responses = 0 | Successful evidence responses whose post-read digest/schema/binding verification later proves false | All successful evidence responses | Page immediately and remove readiness; this is an integrity invariant, not an error-budget trade. |
| Excess load rejected within 250 ms = 100% | Rate/concurrency rejections completed within 250 ms | All deliberate 429/503 admission rejections | Page on any confirmed violation in a 5-minute window; containment takes precedence over availability. |
| Exporter blocking/unexplained drops = 0 | Requests whose duration/result changed because of export, plus drops without a closed reason | All eligible admitted requests and all attempted telemetry records | Page on any exporter-induced request change; ticket immediately for an unexplained drop. Expected bounded queue drops are visible and investigated, not excluded from telemetry quality reporting. |

Percentiles are empirical request distributions, not averages of per-host percentiles. Report route
populations and sample counts. A missing series, non-finite value, clock regression, unknown route,
or unknown outcome fails the SLO calculation closed rather than disappearing from its denominator.

## Benchmark and visual evidence

Regenerate the network-independent aggregate evidence from a supported isolated development
environment into a fresh, Git-ignored staging directory. The writers use an atomic no-replace
publication contract: a new regular file is created with mode `0644`, an already-identical regular
file is a verified no-op, and a symlink, non-regular target, concurrent different file, or existing
different content fails closed. They never overwrite the committed reference artifacts and have no
force mode.

```bash
mkdir -p build
service_evidence_run="$(mktemp -d build/service-operability.XXXXXX)"
python scripts/benchmark_service_operability.py \
  --output "${service_evidence_run}/candidate.json"
python scripts/plot_service_operability.py \
  --input "${service_evidence_run}/candidate.json" \
  --output "${service_evidence_run}/candidate.png"
python scripts/plot_service_operability.py \
  --input docs/benchmarks/service_operability_2026-09-06_patch1.json \
  --output "${service_evidence_run}/committed-input.png"
shasum -a 256 \
  "${service_evidence_run}/candidate.json" \
  docs/benchmarks/service_operability_2026-09-06_patch1.json \
  "${service_evidence_run}/candidate.png" \
  docs/assets/service_operability_2026-09-06_patch1.png
regenerated_reference_plot_sha="$(
  shasum -a 256 "${service_evidence_run}/committed-input.png" | cut -d ' ' -f 1
)"
committed_reference_plot_sha="$(
  shasum -a 256 docs/assets/service_operability_2026-09-06_patch1.png | cut -d ' ' -f 1
)"
test "${regenerated_reference_plot_sha}" = "${committed_reference_plot_sha}"
pytest -q tests/test_service_operability_evidence.py
```

Keep the fresh directory until its JSON, image, and reported hashes have been reviewed. Candidate
benchmark hashes normally differ from the committed reference because measured wall/CPU/resource
observations and the anonymous runtime environment are part of the evidence. A hash difference is
therefore a review signal, not permission to overwrite the reference. The plot rendered from the
unchanged committed JSON is byte-for-byte reproducible and its hash equality is a required renderer
gate. Publish a newly reviewed reference only as an intentional source change in its own work
branch; never point either writer directly at a nonempty committed path.

`make benchmark-service-operability` applies the same policy: it retains a fresh
`build/service-operability.XXXXXX` candidate directory, prints candidate and reference hashes, and
never targets committed paths for output. Review the retained directory before selecting new,
empty dated reference paths on a dedicated work branch; never replace current references in place.

The script exercises the public ASGI boundary without a network or credential. It uses synthetic
aggregate manifests and fixed scenario ordering/seed. Wall/CPU latency and resource observations
are inherently host-dependent; the evidence records the exact anonymous environment and retains
aggregate distributions rather than raw request records. The telemetry A/B keeps fixed in-process
metrics enabled in both arms: the disabled arm has no asynchronous export, while the enabled arm
drains bounded log/span queues into a network-free aggregate sink. It therefore measures the
incremental record/queue machinery without contacting an exporter endpoint.

Latency A/B rounds remain balanced inside the primary benchmark process. RSS is measured
separately because a later process-lifetime peak cannot be subtracted meaningfully from an earlier
peak in the same process. The harness launches exactly one fresh disabled child followed by one
fresh enabled child under the same interpreter, synthetic fixtures, route cycle, warm-up,
admission clock, and service assembly. Each child emits an exact ready marker, waits for an exact
start marker, runs 8 warm-up and 16 measured requests, completes ASGI lifespan shutdown, and
returns one schema-validated aggregate record. Each process has a 20-second hard deadline;
malformed protocol, early exit, or timeout triggers bounded terminate/kill/reap cleanup. The child
receives only an allowlisted locale/Python/PATH environment—not the parent credential or canary
environment—and both arms use in-process HTTPX transport with zero network requests.

The RSS source is `resource.getrusage(RUSAGE_SELF).ru_maxrss`, normalized to bytes. It is the total
process-lifetime peak—not current RSS and not precise attribution to telemetry objects—so
allocator/import noise can make enabled minus disabled negative. Evidence retains that raw signed
difference. Only the incremental-overhead guard is `max(0, signed difference)`, and it must remain
at or below 100 MiB. This is a coarse local engineering guard, not a production capacity or SLO
claim. The plot is generated through Seaborn from the committed JSON. See the
[committed aggregate evidence](benchmarks/service_operability_2026-09-06_patch1.json) and
[visual summary](assets/service_operability_2026-09-06_patch1.png).

The 2026-09-06 patch-1 reference reruns the unchanged synthetic workload against
the 0.3.1 package/lock metadata and GitPython 3.1.59 security patch. Both earlier
references remain available as historical evidence; these measurements are not
a controlled performance comparison.
The 3,016 x 1,869 reference image was generated from aggregate JSON SHA-256
`b55e537f963e8ecbeeebcab18f6f3a3f17ff9e364c4899321ccf38119adabd15`; its
service-operability plot SHA-256: `57e03fd337030c6f06cb0de4018754dac6f5eadb3bfede1a261dcd1f54c821a7`.
The implementing engineering agent visually inspected it at original resolution and confirmed
that the four panels, candidate reference lines, measured percentile curves, A/B bars, explicit
fault/rejection outcomes, limit labels, units, source caption, and synthetic-local disclaimer are
legible; axes are not truncated; and the colorblind palette keeps distinctions visible. The
near-zero overload-bound bar remains intentionally visible through its exact annotation rather
than a distorted axis. This records bounded implementation inspection only; it does not claim
owner review, production evidence, or SLO attainment.

## Incident containment and shutdown

For an integrity, disclosure, unexplained cardinality, resource, or exporter incident:

1. Fail readiness and stop accepting evidence traffic. Do not convert an integrity failure into a
   not-found response.
2. Disable the exporter and preserve bounded, sanitized counters. Do not copy raw secrets or
   hostile payloads into the incident record.
3. Send SIGTERM once and allow only the configured bounded queue flush and graceful-shutdown
   interval. Escalate to supervisor termination after the deadline; never wait indefinitely.
4. Preserve the stopped registry, WAL/SHM state, CAS, image digest, SBOM/provenance, sanitized
   scanner summaries, and aggregate service telemetry. Do not modify or repair storage in place.
5. Check Keychain exposure, mounts, container history/final filesystem, package/image inputs,
   permissions, symlinks, CAS identity/digests, migration ledger, and scanner/database freshness.
6. Restore only from a coherent owner-controlled backup or rebuild from locked inputs. Re-run the
   canary non-disclosure, fault, bounds, image, and readiness gates before returning to service.

For pure overload, leave liveness available, preserve probe capacity, stop the load source, and
wait for fixed buckets/active work to drain. Do not increase a ceiling without measured evidence,
an updated threat analysis, and review.

## Rollback

Disable export, stop the service, remove only the generated local service container and untracked
raw benchmark samples, and revert the MR3 service-operability slice. Do not delete or rewrite the
registry/CAS. MR1 storage and MR2 API remain readable through their previously qualified local
profile. Rebuild from the last accepted locked revision, verify readiness, and record the rollback
reason outside secret-bearing logs.

## Residual limitations

- Same-user malware can query the local socket and attack owner-readable storage.
- The bounded Compose process briefly holds the registry authority in its host environment;
  same-user process inspection remains a residual local risk. Never persist or export that value.
- Unix-socket permissions are not remote identity, tenant isolation, TLS, or host-compromise
  protection.
- Uvicorn remains configured at 32, but the pinned h11 adapter corrects its inclusive off-by-one
  and bypasses only the uninstrumented native text responder. The adapter is the exact pre-ASGI
  request gate; the outer controller independently mirrors 32 active exchanges and the 24-data
  split. Incomplete or idle connections remain separately contained by h11's 16-KiB
  incomplete-event ceiling, the container's bounded backlog/file descriptors, and the
  owner-controlled local trust boundary.
- One process and SQLite/CAS offer no high availability, disaster recovery, or distributed scale.
- The isolated RSS comparison has one fresh process per arm; `ru_maxrss` is host-dependent peak
  total-process memory and cannot isolate allocator noise or attribute bytes to one telemetry type.
- Response buffering can consume up to the fixed per-request ceiling.
- Cooperative deadlines do not preempt arbitrary in-process Python or kernel stalls.
- An injected exporter that suppresses cancellation can survive the application worker deadline;
  `termination_timeout` plus the fixed failure counter expose it, while the container/supervisor
  process deadline remains the final hard stop.
- The local telemetry ring evicts oldest records and bounded export queues may drop new records
  under pressure; separate counters expose but do not recover either loss mode.
- Synthetic local evidence does not establish current market-data quality, model validity,
  prospective shadow duration, paper/live trading readiness, capital authorization, future profit,
  or a production SLO.
