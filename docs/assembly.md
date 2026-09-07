# Lossless source assembly

Assembly implementation for
[AlphaForge #79](https://github.com/srgangaram-swe/AlphaForge/issues/79).

## Frozen inputs and ownership decision

| Source | Exact dev input | Original commits | Objects | Advertised refs |
| --- | --- | ---: | ---: | ---: |
| AlphaForge | `e3548cc4d08f32e08a706ccad28c5b0706c737f2` | 167 | 1,972 | 88 |
| Signalattice | `c5b3d2ec3a1f3b585ee42a0a60638cdf7cc41768` | 115 | 1,246 | 52 |

The owner confirmed rights and authorized import under the existing MIT terms in
the [recorded determination](https://github.com/srgangaram-swe/AlphaForge/issues/79#issuecomment-5563813017).
It covers these exact AlphaForge historical root trees:

- `2b7d1d052b4750e216ecf02d5573617e8ec1f1fe`
- `60a3b01ddfcbdba1a3257b67d8ddeaf518d1ae13`
- `7b8d4cee76db53e4e05ed5ad6d6315477ea101b2`
- `ea4fa50c22fc5aed14fff9b44312f613804107c3`

They are no longer active import blockers. The current resolution record preserves
the original finding, exact tree identity and decision URL. No historical LICENSE
file is invented, no object is dropped, and no third-party notice is removed.
This records the owner's determination; software cannot verify legal ownership.

## Ancestry and tree invariants

Original commits, trees, blobs and annotated tags retain their exact identities.
Git `read-tree --prefix` creates the current package trees. Independent source
tips, including abandoned branches and PRs, become parents of bounded archive
anchor commits (at most sixteen source parents per anchor). Archive anchors retain
the bootstrap deployment tree: preserving ancestry does not deploy old branches.
The assembly commit attaches the exact current source trees below package prefixes.

`provenance/assembly.json` records every original object SHA-256 and size, all
original commits, ref mappings and initial subtree identities. Verification uses
one reachability traversal, checks every original object, verifies Git integrity,
and compares both initial and current subtrees. The original frozen ledgers account
for every historical path/blob; the source manifest binds those ledger identities.
No source-file exclusions are used.

`provenance/ledgers/` contains the re-frozen historical path/blob shards.
`provenance/github/` contains hash-bound, allowlisted GitHub issue/PR, milestone,
release, license and protection snapshots, with original discussion URLs preserved.
Snapshots omit bodies, comments and credentials; they are dated frozen metadata,
not a claim that future source discussions stop changing.

GitHub-friendly public mapping:

- Original `refs/tags/<tag>` → `refs/tags/<source>/<tag>`, retaining the same tag object.
- Other `refs/<suffix>` → `refs/tags/archive/<source>/<suffix>`.

These are immutable provenance anchors, not additional development branches. A
full clone retrieves all source objects and namespaced refs without contacting the
old repositories. Old commit messages remain intact; use qualified source URLs for
issue tracking. Do not create unrelated target issues that old unqualified closing
references could affect during later default-branch promotion.

## Relocated source Git contexts

Some source provenance and release helpers deliberately require a `.git` marker
at their package root and look up original root-relative paths. Changing those
helpers during a no-loss import would mix migration with a domain refactor.

`foundry_build.context` therefore creates an ignored `.git` pointer at each package
root, backed by an independent, non-hardlinked object copy under the unified Git
administrative directory. It restores only that source's original refs and origin
identity, indexes the exact source tree and keeps HEAD at frozen source dev.
The package files remain at their actual `packages/<source>` paths; these are not
shadow source checkouts. Original release dry runs and historical hash verification
operate without weakened assertions. Root verification separately identifies the
unified commit and source manifests.

Context creation refuses dirty package trees, stale identities, unknown sources,
symlinks, foreign `.git` markers and competing reservations. Failed staged creation
cleans owned temporary storage; a late published context is retained for inspection,
not destructively overwritten. Source contexts are generated local state and must
never be committed, pushed or used to commit new unified work.

## Gate parity and known failure

Root workflows derive from preserved source workflows. Tests, native parity,
coverage floors (AlphaForge 78%, Signalattice 80%), browser matrices, accessibility,
dependency/secret gates, package builds and release dry runs remain required.
Root assembly tooling adds a 90% coverage gate. Location-dependent paths and
environment bootstrap are adapted explicitly; separate locked environments avoid
silently resolving incompatible source dependencies into one environment.

The existing non-required [Signalattice #67](https://github.com/srgangaram-swe/Signalattice/issues/67)
service-image reproducibility failure remains a visible job and unresolved issue.
It is not a passing result, not newly waived by import, and not part of a claim of
deployment readiness. No source-required gate is removed.

## Recovery and source independence

Before import, self-contained `git bundle --all` backups were restored into separate
mirrors. Both restores passed full Git integrity checks and exact ref comparisons.
Backups are private local artifacts outside repositories and build-clean paths.
The integration tests independently import fixture histories, preserve abandoned
branches and annotated tags, compare original refs, and verify a clean-room clone.

Failure stops the assembly branch. Both source repositories remain unchanged and
recoverable. Do not delete a remote repository without a separate owner decision.
After a reviewed merge, use another reviewed correction; never rewrite protected
history or silently drop imported ancestry.

Primary mechanics: [Git read-tree](https://git-scm.com/docs/git-read-tree),
[Git bundle](https://git-scm.com/docs/git-bundle), and
[GitHub protected branches](https://docs.github.com/en/rest/branches/branch-protection).
