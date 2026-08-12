# Service security and privacy threat model

## Scope and assets

This model covers the Sprint 5 bounded read-only service introduced by
[#20](https://github.com/srgangaram-swe/Signalattice/issues/20): its outer ASGI admission boundary,
privacy-safe telemetry adapters, hidden local metrics endpoint, optional exporter, Unix-domain
socket service container, locked build inputs, and aggregate benchmark evidence. ADR 0003 continues
to govern version-1 projection semantics; the registry/CAS threat model remains in the
[registry operator guide](run_registry.md).

Assets are registry/CAS integrity and availability, aggregate evidence confidentiality, the local
authority loaded from Keychain, host and container resources, telemetry correctness, dependency
and image provenance, and an operator's ability to distinguish overload from corruption. Raw
market data, model bytes, credentials, paths, and arbitrary report content are never service
outputs.

This is a single-owner local profile. The owner-controlled host process, pre-existing registry/CAS,
Keychain, locked source tree, and pinned CI/scanner inputs are trusted only for their declared
roles. Every HTTP/ASGI field, cursor, stored row, manifest, artifact, filesystem object, package,
image layer, exporter endpoint, exporter response, scanner result, and build artifact is untrusted
until validated. Same-UID host compromise remains outside the protection boundary.

## Security invariants

- Only reviewed GET routes exist; there is no mutation, download, authentication, broker, order,
  position, market-ingestion, or capital authority.
- The default container listens on `/run/signalattice/api.sock`, publishes no TCP port, and has no
  outbound network. Signalattice pre-binds the socket under umask `0077`, verifies exact mode
  `0600` plus owner/type/device/inode, and unlinks only that unchanged identity; Uvicorn never
  creates or chmods the path. Socket permissions restrict access to the owner-operated profile.
- Admission uses fixed global state at two layers. Uvicorn remains configured at 32; the pinned h11
  adapter admits 32 normal parsed exchanges and structurally rejects the next instead of invoking
  Uvicorn's raw responder. The outer ASGI gate independently mirrors 32 active exchanges, limits
  data to 24, and applies API 20/s burst 40 or probes/metrics 2/s burst 4 before semantic validation.
  Attacker values never allocate a bucket, metric, queue, or logger.
- Queries, cursors, headers, bodies, model-card text, responses, metrics, queues, timeouts, retries,
  graceful shutdown, logs, and series cardinality are independently bounded.
- Telemetry values come only from closed enums and route templates. Caller/business values and raw
  exception text cannot cross telemetry or response boundaries.
- A 512-record local canonical JSON ring remains active when remote export is disabled. Capacity
  eviction and sink failure have distinct fixed counters; there is no plaintext exception-log
  fallback.
- Export cannot block or fail an admitted request. It is disabled by default and restricted to
  verified HTTPS or literal loopback-IP cleartext without credentials, query, fragment, or disabled
  TLS verification. Injected transports receive an explicit no-redirect request invariant; this
  sprint bundles and qualifies no concrete remote transport.
- Registry/CAS evidence is returned only after schema, authority, content identity, media type,
  role, run binding, size, and safe projection verification. Integrity failure is distinct from
  absence and fails readiness closed.
- The final image is non-root, read-only, capability-free, local-only, resource-bounded, built from
  locked/pinned inputs, and scanned fail-closed. UID/GID 10001 has `nologin` and no writable home;
  the Debian base contains `/bin/sh`, but the service and healthcheck do not invoke it. No
  credential, raw/licensed data, local database, model, report, fixture, VCS metadata, or
  workstation path belongs in it. The builder installs only the locked `service-runtime` group and
  the final source-module allowlist; it never inherits the numerical research CLI environment.

## Threat analysis

| Threat / abuse case | Control and negative evidence | Residual risk / response |
| --- | --- | --- |
| Telemetry exfiltration through URLs, queries, headers, identifiers, market values, paths, or exceptions | Closed route/operation/outcome/rejection enums; allowlisted fields; access logs disabled; 32-field/8-KiB JSON cap; canary marker injected through every boundary and asserted absent from response, log, metric, span, benchmark, SBOM, image history, and final filesystem | A new field can violate the allowlist; schema/inventory tests and review must precede release. Stop service and rotate any exposed credential. |
| Label-cardinality denial of service with thousands of unique identifiers | Route templates only, no client-keyed buckets, closed labels, fewer than 600 series, 256-KiB exposition cap, adversarial uniqueness test | Fixed series can still consume the declared budget; alert on approach to either ceiling and reject exposition fail-closed. |
| Exporter SSRF, downgrade, credential forwarding, redirect, or unbounded retry | Export off by default; strict URL parser; HTTPS except literal loopback-IP HTTP; reject resolver-name cleartext and URL credentials/query/fragment/TLS-disable; pass an exact `follow_redirects=false` transport request; bounded attempts, body, queue, and timeouts; no request-coupled retry | DNS and local host trust remain relevant for configured HTTPS. An injected transport must honor the no-redirect contract and is not qualified here. Disable exporter on any ambiguity. |
| Metrics disclosure to another local process | Schema-hidden endpoint, socket/loopback boundary, independent low-rate bucket, aggregate closed labels, no secrets or business identifiers | Same-UID processes remain able to read local metrics. Do not expose the socket or bind remotely. |
| Log injection through control characters, ANSI, multiline values, or malicious exception text | Callers cannot provide log fields; canonical JSON encoding escapes controls; record field/byte caps; raw exception messages and class names forbidden; no ad-hoc plaintext fallback; canary/control-character tests parse each record as one object | Terminal/rendering bugs remain possible outside the adapter; retain structured sinks and never interpolate a payload. |
| Body, transfer, content-encoding, malformed-header, or decompression abuse | Every body is forbidden; admission is acquired before validation/body wait; reject `Content-Length` other than zero, transfer/content encoding, `Expect`, and any later ASGI body message; every perimeter problem response closes the connection; no decompressor or multipart parser; real-socket chunked/nonzero-length tests prove bounded problem, EOF, and released accounting; pinned h11 adapter converts fragmented oversized/control-byte parser failures to a redacted structured 400 plus closed telemetry | Idle/partial connections exist before a complete parsed ASGI exchange. H11's 16-KiB incomplete-event bound and container backlog/FD ceilings contain them; the owner-controlled loopback profile is not a remote denial-of-service boundary. |
| Query/header/response amplification and resource exhaustion | 4-KiB query, 1-KiB cursor, RFC-token header names, visible-ASCII-only values (HTAB/C0/DEL/obs-text rejected), 16-KiB headers, 64-KiB card text, 2-MiB response, fixed admission before semantic validation, exact Uvicorn-adapter 32 gate plus outer 32/24 split, fixed buckets, canonical path/lane test, page caps, no internal retries, bounded SQLite progress deadline | Buffered responses use up to 2 MiB each; monitor RSS and reject before data work. Do not raise bounds without evidence. |
| Probe starvation during data saturation | Data concurrency stops at 24 under a global ceiling of 32; probes/metrics have a separate bucket; deterministic barrier test verifies prompt rejection and probe completion | CPU/kernel starvation can still delay probes; container CPU/PID/FD/memory bounds and supervisor deadline contain it. |
| SQLite lock, missing/newer/corrupt schema, malformed row, or deadline evasion | Read-only/query-only connections, busy timeout, progress handler/deadline, checksummed known migrations and schema inventory, typed stored-state validation, redacted retryability, fault injection | SQLite/kernel stalls are cooperatively bounded only where execution returns control. Fail readiness and stop traffic on integrity/newer-schema results. |
| CAS corruption, cross-run substitution, symlink/path traversal, hard-link or mutation race | Descriptor-anchored no-follow reads, owner/mode/link checks, registry/store identity binding, artifact role/media/size/digest/run binding, bounded verified manifest parser, corruption/path fault tests | Same-UID namespace mutation remains an authority-boundary risk. Stop all same-UID writers and preserve evidence for investigation. |
| Client cancellation causing leaked work or corrupted accounting | Cancellation is not a DB deadline; admission release is in a `finally` path; bounded storage deadline continues; deterministic cancellation test verifies slot/metric accounting | Sync storage work may continue briefly after HTTP cancellation. Fixed concurrency and query deadlines contain it. |
| Full local telemetry ring/queues, malformed or unavailable exporter, slow response, or shutdown hang | Canonical 512-record local ring; non-blocking bounded export enqueue; distinct capacity/sink/export drop reasons; bounded request-independent worker; capped response/timeouts/attempts; bounded flush and join; fault and lifecycle tests | Local records can be evicted and export records dropped by design. Loss remains explicit and does not alter durable registry/CAS evidence. |
| Container escape or host modification | UID/GID 10001 has a `nologin` account and no writable home; service/healthcheck commands do not invoke the base image's retained `/bin/sh`; root and evidence mounts are read-only; tmpfs is bounded `noexec,nosuid,nodev`; all capabilities are dropped; no-new-privileges, default seccomp, local socket, and CPU/memory/PID/FD limits apply | Kernel/runtime vulnerabilities and same-host privilege remain. Patch pinned bases deliberately and stop/remove a suspect container. |
| Secret persistence in layer, cache, log, SBOM, history, final filesystem, or benchmark | Keychain retrieval only at runtime; one bounded Compose-process environment assignment becomes a UID/GID 10001, mode-`0400` runtime secret file; no secret-bearing CLI argument, exported/persisted environment, build input, or container environment; canary scans and final-filesystem allowlist | Same-user malware can inspect the short-lived Compose environment or runtime file. The image build receives no secret. Rotate any non-synthetic marker after suspected exposure. |
| Malicious package, base image, action, scanner, or build input | Full digests/SHAs, `uv.lock`, no unconstrained installer upgrade, isolated `service-runtime` group, lazy legacy tracking imports, exact runtime distribution/module inventory, global fixture/data suffix and path rejection, minimal final stage, two clean-build application/config digest comparison, SPDX/provenance, repo and image scans including unfixed findings | Reproducible application/config digests do not prove bit-for-bit base-image reproducibility. Preserve digests and investigate unexplained differences. |
| Hidden High/Critical advisory, incompatible license, secret, material misconfiguration, or failed scanner database | CI fails closed on finding, stale/unavailable database, parser failure, or unexplained exception; exception requires exact identity, containment, owner, issue, and <=30-day expiry | Scanners have false negatives. Layer independent scans, review transitive dependencies, and keep exception scope narrow and expiring. |
| Evidence benchmark leaks host identity or overstates readiness | Aggregate only; no raw request records, hostname, user, absolute path, credential, or business data; anonymous environment, fixed seed/scenarios, explicit synthetic-local limitations; Seaborn source digest | Timing remains host-specific and can fluctuate. It qualifies engineering mechanics only and does not gate on noisy percentiles. |

## Canary non-disclosure protocol

One app integration scenario reads a synthetic marker from the bounded CI environment (or uses a
deterministic network-independent fallback), injects it independently through a header value, query
value, malformed registry record, handled exception, and aggregate artifact metadata, and deletes
its ephemeral registry/CAS root in a `finally` path. It searches bounded responses, parsed local
structured logs, metric exposition, and captured spans. The container gate uses the same CI marker
for its runtime-secret, image-history, SPDX, scanner, and final-filesystem checks. Any marker
occurrence is a release-blocking failure. Separate direct-ASGI and real-h11 cases cover C0, HTAB,
DEL, `obs-text`, invalid header-name tokens, ANSI bytes, and canonical one-object JSON framing.

The canary is test data, never a real API key or password. Passing this test proves only coverage of
the enumerated paths; field allowlists, route inventory, image allowlists, and human review remain
independent controls.

## Containment, shutdown, and recovery

Integrity, disclosure, unexplained telemetry growth, or supply-chain ambiguity makes readiness
false. Stop intake, disable export, issue one SIGTERM, enforce the bounded flush/grace interval, and
let the supervisor terminate after that deadline. Preserve stopped registry/WAL/CAS state,
container and input digests, SBOM/provenance, and sanitized aggregate telemetry. Do not repair the
database, CAS, image, migration ledger, or scan output in place.

Telemetry shutdown separately bounds queue flush and post-cancel worker observation. A worker that
misses the latter increments a closed channel metric and emits `termination_timeout`; it is not
reported as stopped. Arbitrary in-process Python can suppress cancellation, so only the process
supervisor supplies a hard termination boundary. External export remains disabled by default, and
this sprint qualifies no concrete remote transport.

The container startup command gives the registry authority only to one bounded host-side Compose
process so Compose can create a UID/GID 10001, mode-`0400` secret file rather than a world-readable
or incorrectly owned bind mount. The value is not a container environment variable, but same-user
host process inspection can still observe that temporary Compose environment. Never export it,
persist it in `.env`, or run the profile on a shared or compromised owner account.

For overload without integrity failure, retain liveness and reserved probe capacity, stop the load
source, and let fixed active work/buckets drain. For rollback, revert only this additive slice and
remove its local image/evidence outputs; preserve MR1 registry/CAS and MR2 HTTP contracts. The
[operator guide](service_operations.md) provides the exact sequence.

## Explicitly unsupported claims

The Debian runtime base contains `/bin/sh`; the proven restriction is that UID/GID 10001 has
`nologin` and no writable home and that service/healthcheck commands never invoke a shell.
Uvicorn remains configured at 32. The pinned h11 adapter corrects its inclusive decision and
bypasses only the raw overload responder; the outer gate independently mirrors 32 active exchanges
and the 24-data split. Incomplete or idle connections exist before a parsed request transition and
are separately contained by h11's 16-KiB incomplete-event ceiling and the container
backlog/file-descriptor limits.

This threat model does not establish remote authentication, TLS termination, tenant isolation,
Kubernetes or multi-worker safety, high availability, disaster recovery, a 28-day SLO, production
readiness, current market-data validity, paper/live trading readiness, capital authorization,
profitability, or protection from a compromised owner account.

## Promotion governance

`quant_platform.governance` records champion assignments in a per-lane append-only event chain.
Each event binds its payload digest, its predecessor's chain digest, and its own sequence and kind,
and `verify_chain` reports the first sequence that fails to reproduce. The lane head is a cache:
`rebuild_head` replays the events and raises if the stored projection disagrees, because the events
are the authority and a projection rebuilt from a broken chain would launder the break.

Threats addressed:

- **Approval replay and recycling.** An approval is bound to one decision identity and expires after
  seven days, so it cannot be reused on a later comparison that produced a different result.
- **Concurrent apply.** Application is a compare-and-swap on generation *and* champion inside one
  `BEGIN IMMEDIATE` transaction, verified by a barrier-based concurrency test with exactly one
  winner. A refused application leaves no event behind.
- **Split records.** The head update and its event are written in the same transaction, so a crash
  cannot leave a head that moved without recording why.
- **Idempotency-key disclosure.** Only a one-way digest of a caller's key is stored; the key itself
  never reaches the database file. Reusing a key for different content is a conflict, not a silent
  no-op.
- **Automation acquiring authority.** No `force`, `waive`, `skip`, `override`, `bypass`, `unsafe`,
  or `ignore_gates` parameter exists in the package, proven by parsing the module ASTs, and there is
  no `unfreeze` method and no HTTP mutation route.
- **Post-hoc threshold changes.** The frozen policy carries a content identity that the decision
  path checks.
- **Non-finite and naive values.** Canonical digests reject NaN and infinity; every instant must be
  timezone-aware.

**Explicitly not established.** The hash chain is *not* externally tamper-proof: anyone able to
rewrite a row can recompute the remainder of the chain, and restoring an older database file
wholesale leaves a self-consistent chain that verifies. What is detected is divergence within a
chain and between the chain and its projection. Signing and external anchoring are separate work.
The approver field is an **owner assertion**, not cryptographic identity, independent validation, or
separation of duties — a single local operator is both requester and approver. There is no
independent time source, so clock trust rests on the caller.
