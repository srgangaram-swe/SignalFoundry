# ADR 0008: Bind release publication to signed tags and downloaded bytes

- Status: Accepted
- Date: 2026-09-06
- Issue: [#24](https://github.com/srgangaram-swe/Signalattice/issues/24)
- Extends: [ADR 0007](0007-reproducible-signed-release.md)

## Failure and decision

The approved publication run failed because checkout disabled persisted credentials and
`git push` had no credential helper. Review also found that the tag command created an
unsigned annotation, the descriptor retained dry-run identity, and verification rehashed
local files while omitting console/schema/evidence files from the upload.

Keep publication in the protected `release` environment, after a human decision. The build
tool gains an offline `publication` build mode and `publication-gate`: the former produces
publication identity without upload authority; the latter verifies the exact current main
commit, a clean checkout, both merge promotions, and an unused tag. The workflow refreshes
origin refs immediately before checking. Dry-run verification compares the tag set before
and after; an existing release does not make a later dry run fail.

Use a new Ed25519 SSH signing key for each approved publication attempt. Its private half
exists only in a mode-0700 runner temporary directory and is removed by an EXIT trap after
tag creation. The owner remains the sole tagger. The public key and signed tag object are
release assets alongside the complete stage ZIP, Python distributions and manifests.
The tag uses `git tag -s` and `git verify-tag` before pushing. The push uses a command-scoped
GitHub CLI credential helper and the short-lived job token; credentials never enter a URL
or persistent Git configuration.

The public key is **not trusted merely because it appears in the release**. GitHub's OIDC
build-provenance attestation binds every uploaded asset, including the public key, tag
object and full archive, to this repository's release workflow, exact source digest and
`refs/heads/main`. First verify that attestation identity, then verify the Git signature
using the authenticated public key. Another key, workflow or commit must fail. GitHub's
account-level Verified badge is not claimed; trust is in the approved workflow identity.

## Complete transport and remote verification

A flat wheel/JSON collection cannot preserve the console's nested paths. Ship a
deterministic stored ZIP containing every staged file, including manifests, console, API
schema, evidence and license. Verify without extraction: bound member counts and sizes;
reject traversal, duplicate/case-folded names, encrypted entries and nonregular members.

Create a draft, download every asset into a new directory, compare exact membership, sizes
and SHA-256 digests to the build, and compare every archive member to the stage. Verify all
attestations with repository, workflow, main ref and source digest pinned. Reverify the
remote tag signature and object bytes, then make the verified draft public. A partial
failure leaves diagnostic state, never an overwritten or reused tag. If public state
exists, corrective publication uses a new version.

## Alternatives, trust and rollback

A long-lived GPG/SSH secret needs new credential provisioning, rotation, revocation and
account administration. An unsigned tag fails the existing contract. A job-local key
authenticated by OIDC avoids durable signing secrets while retaining cryptographic
verification tied to the approved workflow.

Residual trust remains in GitHub OIDC, its verification root, the runner and reviewed
workflow. A compromised approved job could sign malicious bytes. This is not independent
build verification, hardware-rooted signing, research review or trading authorization.
A compromised run requires an advisory naming affected digests and a corrective version;
never change a published tag or asset. Key destruction prevents reuse but is not a claim
of forensic erasure. Before publication, revert through the normal PR/promotion path.
The new archive code has no network or signing authority. Research behavior is unchanged.

## Validation

Tests exercise deterministic complete archives, missing/extra/renamed/modified members,
hostile paths, special entries, malformed archives, symlink directories, substituted
downloads, source-commit substitution, and real Git rejection of unsigned/wrong-key tags.
Workflow tests enforce the publication gate, full attestation scope and download checks
before draft publication.

References: [Git tag](https://git-scm.com/docs/git-tag) and
[GitHub artifact attestations](https://docs.github.com/en/actions/how-tos/secure-your-work/use-artifact-attestations/use-artifact-attestations).
