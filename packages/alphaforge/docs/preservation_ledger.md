# Signal Foundry preservation ledger

This implements AlphaForge #78, the first Sprint 6 mini-sprint MR. It plans and
checks preservation; it does not create SignalFoundry, import source, change runtime
behavior, archive a repository, or enable trading. [ADR 0023](adr/0023-lossless-signal-foundry-preservation.md)
defines the accepted architecture, alternatives, source-of-truth transition and gates.

## Reproduce offline

Use the locked AlphaForge development environment and complete frozen source mirrors.
Never point the tool at a shallow, partial, alternate-object or rewritten clone.

```bash
python -m scripts.preservation inventory --source alphaforge \
  --repository /path/to/frozen/alphaforge.git --output /new/alphaforge-ledger
python -m scripts.preservation inventory --source signalattice \
  --repository /path/to/frozen/signalattice.git --output /new/signalattice-ledger
python -m scripts.preservation verify /new/signalattice-ledger
python -m scripts.publish_preservation_evidence \
  --ledgers /new/alphaforge-ledger /new/signalattice-ledger --output /new/evidence
```

Inventory writes a complete safe ledger and exits 2 if static blockers exist.
Verify also exits 2 on a blocker or corrupted/malformed publication. Neither command
imports anything. An existing destination is refused. AlphaForge's historical license
gaps make its current static gate **BLOCKED**, intentionally. Signalattice's static
gate passes; its runtime migration parity is still **NOT RUN**.

Every shard is bound to `manifest.json`; every manifest binds the reconstructed
canonical ledger. Ref mappings address complete historical root trees, which map
each file to its source object and intended `packages/<source>/<original-path>`.
Python interfaces are keyed by source blob and include source lines. GUI, OpenAPI,
CLI config, tests, workflows, docs, notices and evidence remain exact file objects.
Dynamic registration stays unresolved rather than becoming an invented static API.

## Explicit online freeze

After confirming owner authentication using `gh auth status`, create complete public
mirrors with `git clone --mirror`, then capture each source's allowlisted metadata:

```bash
python -m scripts.capture_preservation_metadata \
  --repository srgangaram-swe/AlphaForge --mirror /path/to/alphaforge.git \
  --output /new/alphaforge-github.json
```

Repeat for Signalattice. Every advertised ref must match the mirror and remain
unchanged across capture; REST branches and peeled tags are independently matched.
Metadata contains URLs/states/IDs, not issue bodies, comments, API responses wholesale,
credentials or private content. Future remote changes do not mutate the frozen record.
Re-freeze and review the delta before MR2, including this planning MR itself.

## Bounds and complexity

Default limits: 30 seconds per Git command, 900 seconds per inventory; 32 MiB command
output, 8 MiB per object, 256 MiB cumulative uncompressed object contents, 100,000
objects, 2,000 refs, 20,000 paths per historical root, 200,000 total path records,
200,000 total interface records, and 100,000 AST nodes per blob.
The online metadata step has a 180-second request deadline, bounded 100-record pages
and at most 20 pages per endpoint. Limits fail explicitly, not by truncation.

The scan is O(distinct object bytes + historical tree path records + AST nodes).
Casefold/NFC collision checking is O(path components). Blob contents are discarded
after verification; historical root records and parsed interface metadata are retained
within the declared input/output bounds. Git output is streamed through bounded
pipes; source code is never executed and Git lazy-fetch/replacements are disabled.

Publication shards are at most 950,000 bytes; a reservation protects compliant
concurrent publishers and atomic rename exposes only complete ledger directories.
Use trusted local output parents: this is not a hostile multi-user directory sandbox.
Metadata and plot commands likewise require new destinations. No scanner can prove
absence of every secret or determine licenses automatically.

## Reference evidence and recovery

See the [MR1 evidence report](sprint_6_preservation_report.md) for exact counts,
inspected plot, recovery receipts, hashes and unresolved gates.

The committed [reference directory](evidence/signal_foundry_sprint_6/) contains
the exhaustive safe ledgers, GitHub snapshots and derived Seaborn evidence. The
source freeze precedes this MR; source refs and exact hashes live in those records.
Both source repositories remain intact. Verified self-contained private bundles
were restored into fresh mirrors and independently inventoried to check byte-identical
canonical ledgers. The original annotated tag objects remain unchanged.

The fixture integration drill uses Git plumbing to prefix a tree while retaining
all original commits (including an unmerged branch) as ancestry. It proves exact
subtree IDs, executable modes, tag-object identity, unchanged source refs, bundle
verification and recovery. It does not claim a real import has happened.

Tests cover malformed repositories, hostile paths, randomized path-order invariance,
unreviewed/missing licenses, symlink/LFS/secret/raw-data rejection, time/output limits,
static discovery without imports, GitHub pagination/privacy/races, publication
integrity and the public CLI. Existing runtime suites and remote gates remain intact.

## Capability and follow-up matrix

| Preserved capability | Exact mapping / parity requirement |
| --- | --- |
| AlphaForge Python and C++ research/execution | `packages/alphaforge/`; independent original Python/native parity suites |
| Signalattice data/forecast/release package | `packages/signalattice/`; original package, temporal, service and distribution suites |
| CLI/API/OpenAPI | Same source files and static declaration identities; help/route/schema tests after prefixing |
| Both existing GUIs | Original paths below package prefixes; browser/accessibility tests before Nexus cutover |
| Configs/locks/build contexts | Original relative layouts and independent environments until a separate compatibility decision |
| Docs/evidence/notices/workflows | All historical path/blob identities; root workflow replacements require gate/pin/permission parity |
| Nexus workstation | `apps/nexus/` planned in #81, not implemented in MR1 |
| GitHub issues/PRs/milestones/releases | Qualified source identities; later target mapping never replaces original discussion URLs |

MR2 (#79) must resolve the recorded early AlphaForge license gaps through an owner
determination, re-freeze source advances, assemble original ancestry without squash,
and prove runtime compatibility. MR3/#80 owns the unified control plane; MR4/#81
owns Nexus. Neither the planned integration nor the later live-capital issue #88
authorizes trading in this MR. Unfavorable source evidence, Signalattice #67, and
prospective #63 remain visible rather than being upgraded by repository consolidation.
