# ADR 0023: preserve provenance before consolidating Signal Foundry

Status: Accepted for architecture and the executable pre-import contract in #78.
Import authorization is deliberately separate; static blockers and runtime parity
must be resolved by #79 before cutover. This ADR does not authorize live trading.

## Context and decision

Signalattice owns data acquisition, provenance, quality, forecast observability,
and release evidence. AlphaForge owns research, portfolios, execution simulation,
risk, and qualification evidence. They form one evidence chain, not interchangeable
model demos. Combining them must preserve their mathematics, limitations, interfaces,
tests, legal notices, security gates, and full reachable histories.

Adopt a new **SignalFoundry** repository with these initial boundaries:

| Boundary | Initial contents | Dependency rule |
| --- | --- | --- |
| `packages/alphaforge/` | Exact AlphaForge dev tree, unchanged relative paths | Research consumes versioned data contracts |
| `packages/signalattice/` | Exact Signalattice dev tree, unchanged relative paths | Data platform does not depend on research implementations |
| `apps/nexus/` | Planned local-first accessible workstation, not implemented here | UI consumes the typed control plane, never provider/broker credentials |
| Root orchestration | New, reviewed build/CI/docs and later control-plane integration | Invoke independent package environments from their package working directories |

Keep both original repositories authoritative, public, and recoverable during the
entire mini-sprint. Do not archive or delete them. MR1 changes only AlphaForge
planning tools, tests and evidence; no source import or runtime refactor occurs.

## Alternatives

1. A copy or squash import is simple but disconnects original commit ancestry and
   signed tag identity. Rejected.
2. Rewriting every historical tree into a prefix produces convenient historical
   paths but changes original object IDs and signatures. Rejected.
3. Submodules preserve external pointers but not a self-contained reachable object
   closure; they add checkout and recovery failure modes. Rejected for initial import.
4. Preserve original objects and attach them as merge ancestry, with a prefixed
   current tree and namespaced original refs. Selected. Historical commits keep
   their original root layouts; consumers must distinguish historical and new paths.

## Freeze and preservation contract

The offline ledger reads only committed objects in a complete mirror. It records
every advertised head, tag and PR ref, every reachable commit/tree/blob/tag object
ID and SHA-256 of its uncompressed contents, every historical commit root's tracked
path/mode/blob identity, and each path's target mapping and capability class.
Deduplication is by Git object/tree identity, never by approximate similarity.
Every historical Python blob encountered at a tracked path receives a static
definition/reexport/binding/CLI/API declaration inventory without importing code.

The GitHub adapter is an explicit, bounded network step through the authenticated
owner's `gh api`. It records default branch, license, all paginated issues/PRs,
milestones/releases, and protection settings. Bodies/comments/credentials are not
retained. All advertised Git refs are compared before/after metadata capture and
against the frozen mirror. REST tags are cross-checked as peeled commits while the
ledger retains original annotated tag objects. Discussions are a metadata snapshot,
not a claim that GitHub provides an atomic transaction over all discussions.

Shards contain metadata and hashes only, not source blobs, raw observations, models,
credentials or machine-local paths. Canonical serialization and per-shard/whole-ledger
hashes make regeneration independently checkable. Frozen input refs deliberately
precede this planning MR to avoid self-reference. MR2 must re-freeze and review a
delta that includes this MR and any other approved source advances.

## Import mechanics and recovery proof

In MR2, bootstrap the new repository and protected dev/prod/main branches before
implementation. Preserve source refs under `refs/archive/<source>/...`, with public
namespaced tags `refs/tags/<source>/<original-tag>` pointing to the **same tag object**.
Original signed tags are not recreated or re-signed. Archive refs not accepted by a
hosting provider require an explicit published ref mapping and reachable archive
branch anchors; never rely on unadvertised, garbage-collectable objects.

Fetch the complete frozen object closure. Construct the new tip using Git's
`read-tree --prefix` mechanics so each source subtree equals the exact original
tree ID. Attach original commit tips as merge parents, including unmerged PR and
branch tips not reachable from dev. Use bounded archive-anchor merge commits when
many parents are needed; their unchanged deployment tree means ancestry preservation,
**not** deployment of abandoned changes. Source-history imports must use merge
commits, not the ordinary squash method for new work.

Before pushing, independently prove every original commit is an ancestor, every
required ref/tag object is identical, and each target subtree equals the original
source tree. Retain author/committer/timestamp/signature bytes of historical objects.
New commits use only the owner's identity and no coauthor trailers. Never rewrite
history to modify the contributor display.

Create self-contained `git bundle --all` backups before import; verify and restore
them into independent mirrors, run `git fsck --full`, and compare every ref/object
inventory. Private bundles remain outside normal build-clean paths and are not
committed. The test drill proves prefixing, executable modes, annotated tags,
unmerged-branch ancestry, immutable source refs, and bundle restoration using
temporary fixtures. The reference run separately restores both actual public
source bundles and compares their deterministic ledgers.

## Fail-closed boundaries

Reject shallow, partial/promisor, grafted, replaced, alternate-object, corrupt or
unsupported-hash repositories. Validate NFC/casefold collisions, file/directory
overlaps, traversal, control characters and nonportable paths. Symlinks, submodules,
LFS pointers, unreviewed/missing license notices, raw/runtime data and high-confidence
secret markers block import. Empty `.gitkeep` files are allowed only with zero
verified bytes; this is not an exception for data. Scans never print matched payloads.

License policy binds exact reviewed notice hashes and preserves the Signalattice
historical combined MIT/disclaimer as well as its canonical MIT plus separate
DISCLAIMER.md. Some early AlphaForge historical trees lack LICENSE. They remain in
the inventory with a blocking finding: the next import requires a recorded owner
licensing determination covering those exact trees. Do not manufacture a historical
notice, omit those objects, or silently waive the finding. Architecture acceptance
does not mean the pre-import gate is clear.

This bounded scanner is not a proof of secret absence or a legal opinion. Run both
source repositories' pinned security checks and review all scan findings before an
import. API registration can be dynamic; static discovery is not runtime parity.
Keep unresolved declarations visible and qualify them using existing CLI/help,
OpenAPI, route and GUI tests. Historical doc/evidence hashes are preserved even when
their conclusions are unfavorable or have since been superseded.

## Compatibility and cutover gates

Keep both Python namespaces, package manifests, lock files and supported toolchains
independent initially. Do not merge dependency resolutions on assumption. MR3 needs
an explicit compatible control-plane environment decision and integration proof.
Keep native build contexts and package-relative data/config defaults intact.

Source workflows move as preserved but inert files below package prefixes; reviewed
root workflows must invoke all existing gates with original SHA pins and minimum
permissions. Separate test processes prevent import/test-name collisions. Git-root,
archive and provenance helpers that assume standalone layouts need explicit MR2
compatibility tests; this ADR does not claim they already support prefixing.

Retain strict branch protections and required checks. New ordinary work is squashed
into dev; checked sprint promotions use dev -> prod -> main merge commits. Strict
up-to-date protection can require source-identical ancestry reconciliation after
release; that exception needs the owner's explicit authorization, never a weakened
rule, direct protected push, force push, or unexplained reverse promotion.

Cutover requires both source suites, unified integration, contract inventories,
security/notice review, generated-file hygiene, reference evidence, installation,
accessible Nexus tests, and checked forward promotions. Signalattice #67's OCI
configuration reproducibility failure remains explicit; #63's prospective campaign
is not completed by migration. No source repository is archived at cutover here.

## GitHub traceability

Use `(source repository, issue-or-PR, number)` as the key, not a bare number. Preserve
source URLs and states; create target tracking issues only for active work during
the later cutover, with bidirectional links and a reviewed mapping. Historical
discussion identity stays at the source URL because not every GitHub discussion can
be transferred. Qualify cross-repository issue references and audit imported commit
messages so old `Closes #N` text does not close unrelated target issues. Preserve
release URLs and signed artifact records; tags alone do not recreate a GitHub Release.

## Consequences, limits and rollback

The ledger is larger than a summary because it is an exhaustive contract. It makes
review and reproducibility cheaper by separating human-readable decisions from
generated facts. Prefixing retains current files but does not itself establish a
working monorepo. Runtime compatibility, dynamic interface coverage and historical
licensing determinations remain explicit MR2 gates.

Any failed parity/security/license/protection check stops migration. Source remotes
and worktrees remain unchanged; restore the scratch target from verified bundles
or discard only the newly created scratch target. After a reviewed deployment,
rollback is a new reviewed revert or corrected release, never rewritten history.

## Primary references

- [Git bundle](https://git-scm.com/docs/git-bundle): offline object/ref recovery and verification.
- [Git read-tree](https://git-scm.com/docs/git-read-tree): index prefix mechanics.
- [Git rev-list](https://git-scm.com/docs/git-rev-list): reachable object closure.
- [GitHub issues REST API](https://docs.github.com/en/rest/issues/issues): paginated source discussion identities, including PRs.

See [the executable preservation guide](../preservation_ledger.md) for exact commands,
resource limits, evidence and unresolved gates.
