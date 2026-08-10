# Durable local run registry

Signalattice's durable registry is a single-host control-plane kernel for submitted research jobs,
terminal runs, append-only lifecycle events, and content-addressed evidence. It lets a submission
outlive its process and be claimed after restart. It does **not** make user code exactly once:
execution is at least once, and every downstream side effect still needs its own idempotency key.

The registry is local research infrastructure. It is not an authentication system, a multi-tenant
service, a distributed queue, a trading engine, or evidence that any strategy is ready for paper or
live use. [ADR 0002](adr/0002-durable-local-registry.md) records the durable design decision.

## Trust boundary

The database, job JSON, timestamps, legacy rows, cursor tokens, artifact metadata, CAS tree, and
artifact bytes are untrusted on every read. Public APIs return frozen typed records, never a SQLite
connection or an absolute CAS path. The implementation is fail-closed and resource-bounded:

- one explicitly initialized SQLite database on a local filesystem;
- owner-only database, WAL, shared-memory, CAS, and staging permissions;
- parameterized SQL, foreign keys, `WAL`, `synchronous=FULL`, bounded busy waits, and a separate
  bounded startup/readiness verification deadline;
- checksummed forward-only migrations and canonical schema-object verification;
- a persisted, domain-separated verifier that rejects the wrong local HMAC secret after restart;
- canonical bounded JSON, authenticated cursors, closed reason codes, and redacted diagnostics;
- descriptor-anchored CAS access that rejects traversal, symlinks, special files, mutation, and
  digest or size disagreement; and
- no network access, credential access, licensed-data access, broker access, or automatic trade.

Do not place the database or CAS on NFS, SMB, an object-backed mount, a synchronizing cloud folder,
or a multi-host volume. SQLite WAL and the CAS protocol are supported only on one local filesystem.

## Explicit bootstrap

Construction is side-effect free. Only the owning process may explicitly initialize storage:

```python
from pathlib import Path

from quant_platform.tracking.cas import ArtifactStore
from quant_platform.tracking.registry import RunRegistry

state_root = Path("experiments/control-plane")  # ignored local state
store = ArtifactStore(state_root / "cas")
store.initialize()

# Load this from an approved local secret store. Never log it, commit it, put it
# in a URL or CLI argument, or reuse an idempotency key as the HMAC secret.
digest_secret: bytes = load_local_registry_secret()
registry = RunRegistry(
    state_root / "registry.sqlite",
    digest_secret=digest_secret,
    artifact_verifier=store,
)
registry.initialize()
```

Initialization securely creates an owner-only regular database, applies every known migration in
one forward direction, records migration checksums, binds the database to the secret verifier, and
validates the resulting schema. Operational methods open existing state only; they do not create,
adopt, migrate, or repair the primary database or schema implicitly. `probe_readiness()` is
read-only/query-only and does not create a parent directory, primary database, CAS directory, or
migration. SQLite may still create its WAL/SHM coordination sidecars while opening an existing live
WAL database; the validated owner-only directory contains that runtime behavior, and every sidecar
is required to be a private regular file. The implementation deliberately does not use SQLite's
`immutable=1` URI mode because another local process may hold or advance the WAL.

Keep the same secret for the lifetime of a registry. A missing or different secret fails readiness
and operations rather than silently creating new idempotency or cursor identities. Secret rotation
requires a separately reviewed migration and is not supported by this schema.

## Durable lifecycle

Submit one bounded immutable request with a transient idempotency key. Only a domain-separated HMAC
digest of the key is stored. Repeating the key with the exact semantic request returns the original
job; repeating it with different semantics raises a typed conflict.

```python
from quant_platform.tracking.contracts import SubmissionRequest

job = registry.submit(
    SubmissionRequest(
        kind="forecast",
        payload={"dataset_id": "fixture-v1", "seed": 20260808},
        max_attempts=3,
    ),
    idempotency_key=transient_submission_key,
)
```

A later process claims the next eligible job. The returned claimed-job contract contains a secret
lease capability and the canonical request reparsed from durable bytes and rechecked against its
stored digest. Authority-bearing operations obtain time from the clock injected when the registry
is constructed; callers cannot backdate or future-date individual transitions.

```python
claimed = registry.claim(worker_id="local-worker-1", lease_seconds=60)
if claimed is not None:
    request = claimed.request
    lease = claimed.lease
```

The state machine is:

```text
queued --claim--> running --complete--> succeeded
  |                  |  \--fail------> failed
  |                  |  \--ack cancel> cancelled
  |                  \--lease expiry-> queued or failed at the attempt bound
  \--cancel---------------------------> cancelled
```

Every accepted state transition appends an immutable event in the same short write transaction as
the job snapshot. A bounded heartbeat extends the lease snapshot without adding an event, so a
worker cannot grow lifecycle history indefinitely. A lease is valid only for its exact job,
worker, token digest, attempt generation, and unexpired interval. Cancellation is durable intent;
it cannot forcibly interrupt arbitrary Python. Terminal jobs, runs, events, artifact metadata, and
run links are immutable. Closed cancellation and failure codes cross the boundary; raw exception
text, paths, credentials, SQL, and tracebacks do not.

The event ceiling is a safety limit, not a target. Workers should heartbeat only often enough to
retain a realistic lease and must treat capacity or lease-loss errors as control-plane failures.
There is no hidden retry loop.

Readiness replays every bounded lifecycle rather than trusting a row merely because it satisfies
SQLite `CHECK` constraints. The first event must be the unique canonical submission; claims and
attempts must be contiguous; event state/time chains, cancellation fields, lease snapshots,
terminal timestamps, and result/failure fields must agree; and each stored run must match a real
claim and legal attempt-closing event. A run-insert trigger requires the current running attempt.
Heartbeat events are deliberately absent from the durable event enum and schema because heartbeat
state is snapshot-only.

## Artifact publication and terminal runs

Publish bytes before registering or linking them:

1. `ArtifactStore.publish()` opens a bounded non-symlink regular source through descriptors.
2. It copies and hashes into a private same-filesystem staging file while checking source identity.
3. It flushes and makes the fully hashed staging object read-only, then takes a five-second bounded
   advisory publisher lock on the verified CAS root only for destination publication/reuse checks
   and staging-link removal. This serializes the short-lived second hard link without placing source
   I/O or O(bytes) hashing under the exclusive lock. Verification, reads, and direct inspection take
   a bounded shared lock only while opening and validating a canonical single-link descriptor, then
   release it before hashing. Persistent link count greater than one still fails ordinary publish
   reuse, verification, reads, inspection, and retention.
4. Registration and terminal linking perform their full descriptor-safe CAS hash through the exact
   bound verifier **before** acquiring SQLite's process-wide writer lock.
5. The short write transaction rechecks the same verifier object, CAS identity, and durable
   registry/CAS binding. Registration then inserts immutable metadata. Terminal linking also
   re-reads the exact immutable registered metadata, rejects any retention tombstone, and atomically
   inserts the terminal run, links, job transition, and lifecycle event.

The filesystem and SQLite cannot share one transaction, and a byte hash cannot be held atomically
across the later SQLite commit. Under the supported cooperative CAS protocol, objects are private,
read-only, single-link files that are never replaced, while artifact metadata and terminal links are
append-only. Those invariants make hashing outside the writer safe for legitimate concurrent
operations and keep unrelated registry writes available while large objects are verified. A crash
after CAS publication but before metadata registration can leave a valid unlinked object; it cannot
create a link to partially published bytes through the supported protocol. Bounded object and
stale-staging inventory make that crash window discoverable after a grace period. Discovery is
diagnostic only: it never deletes automatically.

Owner-level access remains a trust boundary: a malicious or buggy same-UID process could mutate the
CAS namespace between a successful preflight hash and the database commit. CAS maintenance must be
serialized against registry publication/linking, and later verified reads fail closed on any
identity, type, size, digest, hard-link, permission, or mutation disagreement. This protocol does
not claim an atomic filesystem/database commit or protection from a compromised owner account.

Initialization durably publishes a checksummed random `.store-id` marker with exact mode `0400`;
the root and every CAS directory require exact mode `0700`. The marker, directories, objects, and
staging files are opened through descriptors and reject wrong ownership, extended macOS ACLs,
POSIX ACL xattrs, symlinks, special files, unsafe modes, hard links, substitution, and mutation as
applicable. The path-free `store_id` survives process restarts and copied backups. Inventory
continuations are frozen typed values bound to that identity and the exact grace cutoff, so a
caller cannot advance a cutoff between pages and silently skip newly eligible lower keys.

The initializer may create `.store-id`, `.staging`, and `objects` only when an unbound descriptor
walk created the CAS root itself in that invocation; that new root receives a fresh random identity.
If the root already exists, every component must already be present and valid; a missing marker or
directory is corruption and is never repaired or adopted automatically. The marker checksum detects
corruption but is not authentication: a process with owner access can construct another valid
marker. Reopen through `ArtifactStore(..., expected_store_id=<registry-bound-id>)` when a durable
external authority is available. This mode is reopen-only: a missing root fails without creating
path components, the expected value never seeds a replacement store, and any mismatch is rejected
before accepting the store.

Run evidence records an honest evidence class (`measured`, `simulated`, `backtested`,
`paper_traded`, `live`, or `not_applicable`) and an explicit limitation. Classification does not
authorize publication or trading. Manifest reads require both a manifest media type and the
metadata artifact class, then recheck canonical storage key, size, digest, file type, and mutation
before returning caller-bounded bytes. They also authenticate the registry HMAC authority and require
the reader's initialized CAS identity to equal the immutable registry binding. Streaming artifact
publication and metadata are capped at 1 GiB, while any in-memory verified read is capped at 64 MiB
(the default manifest ceiling remains 8 MiB).

## Framework-neutral reads

`RegistryReadPorts` is the only boundary later HTTP and console adapters should consume. It offers:

- `probe_readiness()`;
- `get_run()` and authenticated, snapshot-bound `list_runs()`;
- `get_artifact()` and run-bound `list_run_artifacts()`; and
- `read_verified_manifest()`.

Views are frozen and path-free. Legacy `runs` rows are explicitly `legacy/unverified`; arbitrary
params, tags, metrics, and artifact paths are redacted. Malformed, non-finite, oversized, ambiguous,
out-of-domain sequence, or cross-source-colliding legacy state fails closed. Every native keyset
source validates its exact integer minimum and maximum before paging, and an authenticated cursor is
rejected if its source high-water mark regresses. Because the historical table predates
immutability, continued legacy pagination fingerprints all bounded projected rows at or below its
high-water mark. That is intentionally local-scale O(history) compatibility work, not a claim of
large-scale query performance. New service traffic should use registry-native runs.

The rollback-only `JSONTracker` and `SQLiteTracker` adapters preserve their historical record shape
but now apply the same finite/depth/node/string validation before persistence and before returning a
record. A record is capped at 512 KiB, each encoded JSON field at 64 KiB, one list result at 16 MiB,
and one list call at 1,000 records; SQLite reads also have a 500 ms cooperative deadline and a 1 MiB
engine value ceiling. JSON reads use no-follow descriptors and reject mutation or extra hard links.
SQLite reads are read-only/query-only with an explicit safe projection. Caller-mutated JSON run IDs
cannot become filename components. Unsafe path-, control-, newline-, ANSI-, or credential-shaped
legacy name/data text is never logged and is returned to terminal adapters as `[redacted]`.
Persistence and read failures expose typed redacted errors, never raw SQLite or filesystem text.
Compatibility does not retain destructive `INSERT OR REPLACE` behavior: a duplicate SQLite
`run_id` raises `LegacyTrackingWriteError` and leaves the original row unchanged.

## Retention and orphan review

Retention is disabled by default. Enabling it requires an explicit bounded policy and an operator
workflow with five ordered steps:

1. plan eligible, old, unlinked, unpinned objects under a read snapshot;
2. inspect and confirm the exact canonical plan digest; and
3. for an unstaged plan, repeat the read-side registry checks and descriptor-safely preflight every
   exact CAS generation and physical grace cutoff before durable deletion intent exists;
4. in a short Phase A writer transaction, repeat the active-work, policy, candidate-metadata,
   link, pin, prior-intent, count, and byte invariants, then persist the signed plan envelope and
   pending append-only tombstones; and
5. in Phase B, verify and unlink each exact CAS generation outside the SQLite writer, then finalize
   its permanent tombstone in a short transaction.

The controller rechecks the full active policy, grace cutoff, candidate identities, aggregate byte
and count ceilings, absence of active work, links, pins, prior deletion, and committed plan intent.
Pending tombstones block new links and allow restart to resume an interrupted Phase B. Metadata and
tombstones are never removed. A partial filesystem failure is explicit and resumable; it is not
reported as full success.

The pre-stage CAS inspection is not deletion authority by itself. Phase B repeats generation and
physical-grace verification before the first unlink and retains the independent
`unlink_verified(..., expected_generation=...)` compare-and-delete guard for every object, so a
change after preflight fails closed. The signed envelope carries both exact `last_changed_ns` for
grace decisions and a typed opaque generation digest bound to the store ID, artifact digest,
device, inode, file type, size, modification nanoseconds, and metadata-change nanoseconds. The
microsecond `last_changed_at` field is display evidence only and never deletion authority. Raw
device and inode values are not exposed. It uses
`verify_absent()` to prove replayed finalized tombstones remain absent without creating fan-out
directories. The absence proof requires the prior canonical fan-out to be acquired and rebound and
accepts only `ENOENT` for the exact object name; a missing fan-out fails closed. It rebinds the
opened fan-out beneath the initialized object tree both before and after the final absence check.
Immediately before `unlinkat`, the destructive primitive likewise rebinds the opened fan-out parent
and repeats the check afterward. POSIX cannot make either "absent at return" or "unlink only while
this dirfd remains beneath that root" atomic against a same-UID namespace writer after the final
check. Maintenance must therefore be serialized against same-UID processes. Exact owner-only modes
exclude other users, and any observed rebind, replacement, mutation, or generation mismatch fails
closed. A filesystem that reuses the complete bound tuple, including inode and both exact
nanosecond timestamps, could reproduce a token; same-UID serialization remains the authority
boundary for that residual case.

Retention cannot target raw, vendor, processed, feature-store, model, report, or Signal Foundry
bundle roots. The CAS root is the only byte-deletion boundary. Object/staging inventory may identify
old content absent from the registry, but an operator must investigate the publication crash window
before any separately authorized cleanup. This MR provides no automatic orphan deletion.

## Recovery runbook

### Process or worker crash

1. Preserve the database, WAL, shared-memory file, and CAS tree.
2. Verify `probe_readiness()` with the original secret.
3. Run bounded lease recovery. An expired attempt returns to `queued` only while attempts remain;
   otherwise it becomes `failed` with the closed exhaustion code.
4. Claim normally. Assume prior user code may have run and make downstream effects idempotent.
5. Reconcile pending retention intents and inspect old CAS/staging inventory.

### Integrity, schema, or secret failure

1. Stop submissions, claims, retention, and service traffic.
2. Do not edit the migration ledger, triggers, tables, hashes, or tombstones by hand.
3. Copy the complete stopped database/WAL state and CAS for investigation.
4. Identify whether the cause is the wrong secret, permissions, schema drift, corruption, a newer
   unsupported schema, non-local storage, or host compromise.
5. Restore only from a coherent owner-controlled backup, then re-run readiness. A code fix moves
   forward through a new checksummed migration; it never rewrites migration history.

### Busy, verification-deadline, or capacity failure

`busy_timeout_ms` bounds SQLite lock acquisition. The distinct `verification_timeout_ms` bounds
startup/readiness full-integrity work (default 2,000 ms) using keyset batches plus monotonic
SQLite/Python deadline checks; expiry returns the retryable machine code
`registry_verification_timeout`, never a partial readiness verdict. Queue capacity, event count,
page size, request size, attempts, leases, artifacts per run, manifest bytes, retention candidates,
and retention bytes are likewise deliberate ceilings. Reduce concurrency or work size,
finish/cancel existing work, or propose a measured configuration change. Do not add retries that
hide contention or raise a limit without workload and memory/disk evidence.

The same pass validates append-only migration-ledger timestamps, exact SQLite INTEGER storage,
canonical UTC timestamps, strict bounded JSON, immutable run/artifact/link scalars, canonical CAS
storage keys, and retention envelope bindings. Duplicate keys, non-finite numbers, excess depth or
nodes, malformed UTF-8, REAL values in integer domains, and semantically forged lifecycle rows
return a typed `integrity_error`; attacker-controlled bytes are not reflected in the verdict.

## Backup, restore, and rollback

Quiesce submissions and claims and finish or explicitly expire active work before backup. Use a
SQLite-supported coherent snapshot while preserving the corresponding CAS. Copying only the main
database file while WAL is active is not a backup. A restored database and CAS must retain owner-only
permissions and pass readiness, foreign-key, integrity, schema, secret-binding, and CAS verification
before operations resume.

Rollback stops new registry work and returns callers to the legacy adapter. It does not down-migrate,
truncate events, rewrite a terminal run, delete a tombstone, mutate a legacy row, or remove CAS
bytes. Preserve failed state for diagnosis.

## Performance boundary and residual risk

Writes deliberately serialize through short `BEGIN IMMEDIATE` transactions. Artifact hashing is
O(bytes) and occurs outside database transactions; deterministic blocking-verifier tests prove that
an unrelated submission can still acquire the writer while registration or terminal-link hashing
is in progress. Startup/readiness integrity work uses bounded keyset batches and a distinct
verification deadline. Ordinary keyset reads are indexed and bounded; legacy fingerprinting and
filesystem inventory are local-scale maintenance scans. No throughput, latency, multi-process
saturation, NFS, remote-database, or distributed-worker claim is made in this slice.

A process with owner write access to both the database and CAS can still corrupt or remove evidence.
Verification detects many substitutions but cannot defeat host compromise, malicious owner-level
code, storage firmware lies, or loss of both primary state and backup. Do not place API credentials,
licensed observations, private account data, or trading secrets in job payloads, events, metadata,
limitations, or legacy compatibility fields.

## Validation

Before release, run the repository's complete offline CI boundary:

```bash
make ci
```

The MR also requires focused transition, migration, contention, wrong-secret, schema-drift,
filesystem race, source-mutation, CAS corruption, read-port, legacy-compatibility, retention,
fault-injection, package, and synthetic pipeline tests. Remote Python 3.12–3.14, container,
dependency, and secret checks must pass without weakened settings.
