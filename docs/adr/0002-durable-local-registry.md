# ADR 0002 — Durable local registry and content-addressed evidence store

- **Status:** Accepted
- **Date:** 2026-08-08
- **Work item:** SF-S5-SL-MR1 ([#18](https://github.com/srgangaram-swe/Signalattice/issues/18))
- **Supersedes:** none

## Context

Signalattice historically persisted a completed in-process experiment as one row in a SQLite
`runs` table. That adapter is useful for notebook and CLI compatibility, but it is not a durable
control plane: `INSERT OR REPLACE` can erase prior state, arbitrary JSON and path strings are stored
without an integrity contract, and no job, lease, event, migration, artifact, cancellation, or
retention protocol exists. A later HTTP service must not expose those implementation details or
interpret legacy rows as verified evidence.

Sprint 5 needs a local process to submit work that a later process can claim. Submission can be
idempotent, but executing Python and filesystem side effects cannot honestly be described as
exactly once. The design therefore separates durable state transitions from user work and treats
the execution boundary as at-least-once. Downstream effects require their own idempotency.

The registry processes untrusted identifiers, JSON, timestamps, database state, and artifact bytes.
It must remain useful on a laptop without introducing a network service, broker, daemon, or remote
database, and it must preserve every existing tracker backend and the legacy `runs` table.

## Decision

### Local storage and migration boundary

The control-plane kernel uses Python's standard-library SQLite driver and a local-filesystem
database. Explicit bootstrap alone may create the database and creates it as an owner-only,
single-link regular file. Operational writers use SQLite `mode=rw`; readers use `mode=ro` plus
`query_only`, so missing state is never created implicitly. The database and WAL/SHM companions
must remain owner-only regular files with stable open identities. Every connection enables foreign
keys and a bounded busy timeout; writers require `journal_mode=WAL` and `synchronous=FULL`. A
separate bounded verification deadline governs startup/readiness integrity scans, which advance in
keyset batches and fail with a retryable typed timeout rather than returning a partial verdict.
Write operations use short `BEGIN IMMEDIATE` transactions. Callers never receive a connection or
cursor.

Namespaced `sl_registry_*` tables are installed by checksummed, forward-only migrations. A migration records
its ordered version, name, SHA-256 digest, and application time in
`sl_registry_schema_migrations` in the
same transaction as its schema changes. Startup rejects a missing migration in an otherwise
namespaced database, a changed digest, a gap, an unknown newer version, noncanonical table/index/
trigger DDL, an unexpected trigger, and a failed integrity or foreign-key check. Bootstrap also
persists a versioned, domain-separated HMAC-key verifier; a different secret fails readiness and
all operations. Secret rotation requires a future reviewed migration. Migrations never rewrite or
down-migrate the historical `runs` table.

Readiness probing is deliberately different from bootstrapping. It opens an existing database in
SQLite read-only mode, enables `query_only`, applies bounded read settings, verifies the known
migration ledger and database checks, and closes the connection. It never creates a table,
migration, primary database, or directory. SQLite may create WAL/SHM coordination sidecars
while opening an existing live WAL database; those files remain contained by the validated
owner-only parent and must pass the same private-file checks. `immutable=1` is intentionally not
used because another local process may hold or advance the WAL.

SQLite WAL is supported only on a local filesystem. NFS, SMB, object-backed mounts, multi-host
coordination, and a generally distributed database are outside this decision. Operators must stop
new claims before copying or restoring the database and its WAL state.

### Jobs and lifecycle evidence

A submission contains a bounded, canonical request and a transient idempotency key. The registry
stores only an HMAC-SHA-256 digest of that key using a caller-supplied local secret; it never stores
or logs the key. Repeating the same key and semantically identical request returns the existing
job. Reusing it for different canonical request bytes raises a typed conflict.

Claiming returns both the secret lease capability and a bounded `SubmissionRequest` reparsed from
the canonical stored bytes after its SHA-256 digest and denormalized fields are rechecked. The job
state machine is explicit:

```text
queued ──claim──> running ──complete──> succeeded
  │                  │  └──fail───────> failed
  │                  ├──lease expiry──> queued (while attempts remain)
  │                  └──lease expiry──> failed (attempts exhausted)
  └──cancel────────> cancelled

running ──cancel request──> running + cancellation_requested
```

Terminal state is immutable. A claim is a compare-and-set transition under one write transaction;
at most one worker receives the lease generation. Heartbeats and completion must present the exact
worker, lease token, generation, and unexpired lease. Lease recovery is explicit and bounded; there
is no hidden retry loop. Cancellation records intent but cannot forcibly preempt arbitrary Python.

Every accepted state transition appends one immutable, monotonically ordered
`sl_registry_events` row in the same transaction as the job snapshot update. Heartbeats update the
bounded lease snapshot without appending an event, preventing an otherwise unbounded heartbeat
history. Event payloads are bounded canonical JSON and do not contain raw idempotency keys,
artifact bytes, credentials, exception traces, or local paths.

Startup and readiness do not trust schema-valid rows. Under one deadline they keyset-scan jobs and
replay each bounded event chain from its sole `submitted` event, requiring contiguous claims,
exact attempts, legal requeue/terminal transitions, matching cancellation intent, canonical UTC
chronology, and agreement between the final event and job snapshot. Every terminal run must match
a real claimed attempt and its closing event; lease-expiry and queued-cancellation paths retain
their documented no-run semantics. A defense-in-depth insert trigger also permits a run only for
the referenced job's current running attempt. Persisted heartbeat events are unsupported and
rejected: the bounded job snapshot is their only durable representation.

Migration-ledger rows deny update and delete while retaining forward append authority. Readiness
also validates their exact integer versions and canonical application timestamps. All durable JSON
is decoded with byte, duplicate-key, non-finite-number, depth, node, string, and memory guards;
SQLite integer semantics require exact INTEGER storage rather than lossy REAL coercion.

Queue admission, attempts, lease duration, identifier/text/JSON sizes, query pages, busy waits, and
event counts are bounded. Stable pagination orders by immutable sequence and uses opaque,
authenticated cursors tied to the query snapshot and filter. Filtered job cursors also bind the
remaining snapshot membership and fail closed if an in-place state transition would make a later
page inconsistent. Each keyset source must have an exact integer sequence range inside its cursor
domain; negative or oversized durable positions fail closed instead of falling below the first
page. Later pages reject regression of the authenticated high-water mark, exposing rollback or tail
deletion instead of returning a silently truncated snapshot. No query reports an unbounded total
count.

### Runs, legacy projections, and read ports

New `sl_registry_runs` records and their lifecycle evidence are immutable projections keyed by stable
identifiers. Legacy `runs` rows remain byte-for-byte unchanged and are projected only through a
restricted `legacy/unverified` view. Malformed legacy JSON, non-finite metrics, arbitrary params,
tags, filesystem paths, and artifact strings are not exposed through the service boundary.
Duplicate legacy run identities are ambiguous corruption: both get and list fail closed instead of
selecting an arbitrary row. Legacy JSON uses the same bounded strict decoder as registry evidence.

The legacy table name, columns, backend selection, and returned record shape remain compatible, but
destructive duplicate-key overwrite semantics intentionally do not. New compatibility writes use
plain `INSERT`, not historical `INSERT OR REPLACE`: a duplicate `run_id` raises the typed, redacted
`LegacyTrackingWriteError` and preserves the original row byte-for-byte. This is a deliberate safety
correction to the data-loss behavior identified in this ADR's context, not a claim of behavioral
compatibility for callers that depended on silent replacement.

The framework-neutral read boundary exposes typed, immutable projections only:

- `probe_readiness()`;
- `get_run()` and bounded, snapshot-stable `list_runs()`;
- `get_artifact()` and bounded `list_run_artifacts()`; and
- `read_verified_manifest()` for an allowlisted media type and caller-declared byte limit.

Those ports return no SQL handle, CAS pathname, arbitrary JSON document, credential, or mutable
object. A later FastAPI adapter depends on these ports rather than on SQLite or filesystem details.

### Content-addressed artifacts

Artifacts live beneath one configured, ignored CAS root. Publication accepts only a bounded,
non-symlink regular source file. It retains an open source descriptor, hashes and copies through a
new private staging file, rechecks source identity and size, flushes bytes, and publishes by the
full lowercase SHA-256 digest. Destination fan-out is deterministic and bounded.

The CAS mechanism is deliberately POSIX-only even though the rest of the Python package remains
importable on other operating systems. Construction fails through the typed artifact boundary,
before creating or changing storage, when advisory `flock`, directory-descriptor, or no-follow
primitives are unavailable. Source copying and hashing may proceed in parallel without a CAS lock.
After staging bytes are complete, publication takes a five-second bounded exclusive advisory lock
on the verified root descriptor only for destination linking or reuse checks and removal of the
transient staging link. Verification, reads, and direct inspection take the corresponding bounded
shared lock only while opening and validating a canonical single-link object descriptor, then
release it before O(bytes) hashing. This narrow protocol prevents cooperating processes from
mistaking publication's temporary second hard link for an external escape without serializing
artifact I/O.

The store refuses traversal, symlink components, special files, unsafe permissions or ownership,
hard links, oversized input, mutation during copy, digest mismatch, and a conflicting existing
object. Reuse succeeds only after independently verifying the existing object's type, identity,
size, and digest. Publication flushes the file and containing directories.

Registration and terminal linking perform their full descriptor-safe CAS hash before acquiring the
SQLite writer lock. The short write transaction rechecks the exact verifier object, CAS identity,
durable binding, immutable metadata, retention tombstones, and lifecycle state applicable to the
operation; it performs no object-sized I/O. Private read-only single-link objects that are never
replaced, together with append-only artifact metadata, make this safe for supported cooperative
concurrency and prevent large hashes from blocking unrelated writes. The filesystem/database gap
can still leave a valid unlinked object after a crash. It cannot atomically exclude a malicious or
buggy same-UID namespace mutation between hash and commit, so CAS maintenance must be serialized and
subsequent reads re-verify the bytes. Descriptor-anchored, bounded object and stale-staging
inventories make publication crash state discoverable without authorizing deletion.

The registry stores metadata and immutable run/artifact links, never source paths or artifact
contents. Artifact classes are a closed allowlist. Pinning and evidence classification are explicit
policy, not filename inference.

### Retention

Retention is disabled by default. Planning is a read-only operation that returns a bounded,
canonical plan, its plain SHA-256 payload digest, and a separate HMAC-SHA-256 plan identity bound to
the registry instance, CAS identity, and local authority secret. Execution requires the exact
authority-bound plan identity and uses the registry's bound clock. Before Phase A creates durable
intent for an unstaged plan, a read-side preflight repeats registry eligibility and
descriptor-safely verifies every exact CAS generation and physical grace cutoff. Phase A then
repeats active-work, policy, candidate-metadata, link, pin, prior-intent, count, and byte invariants
inside a short write transaction and persists both the signed plan envelope and pending append-only
tombstones. Phase B rechecks all generations and grace before the first unlink, retains a per-object
generation compare-and-delete guard, verifies and unlinks only those CAS objects outside the
database writer, then finalizes each tombstone in another short transaction. Pending intent makes
the unavoidable crash window visible and resumable. Retention never removes run/job/event/artifact
metadata, active or pinned evidence, or anything outside the CAS root. A changed candidate
invalidates the plan; there is no hidden best-effort sweep.

The raw, vendor, processed, feature-store, model, report, and Signal Foundry bundle roots are not CAS
retention targets.

### Failure semantics and observability

Expected failures use typed exceptions that retain stable machine-readable codes. Errors gain
bounded context at the registry boundary without embedding secrets, payloads, SQL, paths, or stack
traces. Busy state, capacity exhaustion, conflicts, stale leases, invalid transitions, integrity
failure, migration drift, and unsupported schema are distinct.

A pipeline exception remains the primary exception. The SQLite adapter attempts to durably record
the failed run; if that persistence also fails, it records a sanitized diagnostic without masking
the pipeline failure. A persistence failure after successful user work remains an error.

## Security analysis

Threats include SQL injection, malicious JSON, oversized inputs, Unicode ambiguity, replayed
idempotency keys, guessed cursors, stale or stolen leases, database rollback/corruption, migration
drift, symlink and path traversal, source mutation, special files, raced destination creation,
artifact substitution, retention-plan replay, and sensitive-data disclosure through logs or read
projections.

Controls are parameterized SQL, canonical bounded encodings, normalized identifiers, HMAC key and
cursor digests, explicit state transitions, lease generations, checksummed migrations, SQLite
integrity/foreign-key checks, descriptor-oriented filesystem validation, content identities,
fail-closed publication/reuse, classified metadata projections, no raw paths, and bounded resource
use. CAS initialization adds a private, read-only, checksummed random store marker. Exact owner-only
modes, descriptor-based macOS and POSIX ACL rejection, store-and-cutoff-bound typed inventory
cursors, exact-object generation inspection, absence proofs, and pre/post-unlink containment checks
close accidental cross-store, advancing-cutoff, replay, and detached-parent failure modes. This is a
single-user local trust boundary, not authentication or multi-tenant isolation.

Only the invocation whose descriptor walk creates the CAS root may create its marker, staging
directory, and object directory. A pre-existing incomplete root is evidence of corruption or an
interrupted bootstrap and fails without repair. Exact-object absence requires an acquired and
rebound canonical fan-out; missing fan-out components never count as proof. The optional externally
bound expected store identity detects substitution across process restart.

Residual risks remain:

- SQLite and the local filesystem cannot form one atomic transaction; orphan discovery is required.
- A successful preflight hash and a later SQLite commit cannot be one atomic operation. Owner-only,
  read-only, single-link CAS objects plus append-only metadata protect cooperative operations, but
  same-UID CAS maintenance must be serialized against registration and terminal linking.
- A process with write access to both database and CAS can corrupt or delete evidence. Verification
  detects substitution but does not prevent host compromise.
- Cooperative timeouts cannot preempt arbitrary in-process Python.
- POSIX cannot atomically couple `unlinkat` to proof that an already-open parent descriptor remains
  beneath the CAS root. The implementation rebinds immediately before and after unlink, but
  maintenance still must serialize against malicious or buggy same-UID rename operations.
- The store-marker checksum detects malformed or corrupted content; it does not authenticate a
  canonical replacement created by the owner. Durable registry/external binding through the
  expected store identity is required to detect that substitution after process restart.
- HMAC avoids storing raw idempotency keys and authenticates the persisted authority verifier,
  cursors, and retention plans only while the local secret remains private. It does not encrypt
  cursor or plan payloads and cannot defend an authority secret compromised in the owner context.
- WAL durability depends on the filesystem and host honoring flush semantics.

## Alternatives considered

**Continue using the legacy `runs` row.** Rejected because replacement writes, unchecked JSON, and
path strings cannot provide lifecycle or evidence integrity.

**Redis/Celery or a background worker framework.** Rejected for this slice. It adds network,
deployment, and operational trust boundaries without improving the required local evidence kernel.

**PostgreSQL.** Deferred until multi-host concurrency or scale evidence justifies it. SQLite keeps
the current slice network-independent and reproducible.

**Store artifact bytes in SQLite.** Rejected because large BLOB transactions increase contention
and make independent content verification and bounded retention harder.

**Claim exactly-once execution.** Rejected. A database transition cannot prove that arbitrary model
training or filesystem effects occurred exactly once across a crash.

**Automatically delete unreferenced content.** Rejected. Retention is a destructive policy action
and therefore requires a deterministic, confirmed plan.

## Consequences

- Later service and console layers receive a stable, redacted, framework-neutral read contract.
- Submission and claiming survive process boundaries, while execution semantics remain honestly
  at-least-once.
- Artifact bytes become immutable and independently verifiable, at the cost of explicit orphan
  reconciliation and local storage management.
- The legacy adapter's backend/table/record contract and existing stored rows remain compatible;
  duplicate IDs now fail closed instead of replacing evidence, and new code must not infer verified
  evidence from legacy state.
- The control plane remains deliberately local and single-host.

## Rollback

Stop new submissions and claims, allow or explicitly expire active leases, and return callers to the
legacy tracker adapter. Do not down-migrate, edit terminal events, delete CAS bytes, truncate the
migration ledger, or mutate the historical `runs` table. Preserve the database, WAL, CAS, and
diagnostics for investigation. A corrected implementation moves forward through a new checksummed
migration.

## Non-goals

FastAPI/OpenAPI, browser UI, a worker daemon or pool, remote/object storage, multi-host operation,
authentication, automatic model publication, licensed market-data storage, broker connectivity,
orders, positions, paper/live trading, capital authorization, production readiness, or
profitability claims.
