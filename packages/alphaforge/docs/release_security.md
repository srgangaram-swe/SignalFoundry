# Release and security governance

AlphaForge treats dependencies, workflow code, release artifacts, and local
market data as separate trust boundaries. These controls reduce known risks;
they do not prove the absence of vulnerabilities or authorize deployment.

## Pull-request and scheduled gates

The `Security` workflow runs with read-only repository contents and no market
data or provider credentials. It:

- rejects pull requests that introduce dependencies with moderate-or-higher
  known advisories or denied licenses through GitHub's dependency review;
- retains the strict hashed runtime/data/ML dependency audit;
- verifies a separate hashed runtime-plus-development export against the exact
  lock graph, installs it into a clean environment, and audits those pins in
  strict mode without a second resolver;
- scans full reachable Git history with Gitleaks; and
- runs again on protected-branch pushes and a weekly schedule where applicable.

GitHub secret scanning and push protection remain enabled on the public
repository. These scanners complement one another: pattern scanning cannot
guarantee that a novel or encoded secret is absent. API keys, licensed bundles,
provider responses, and governed run roots must remain in approved local secret
or ignored-data storage.

All third-party GitHub Actions are pinned to full commit SHAs, checkout does
not persist credentials, jobs have hard timeouts, and workflow permissions are
deny-by-default. Dependabot proposes weekly Python and Actions updates; each
proposal still follows the protected `dev -> prod -> main` review flow.

The development gate addresses [GHSA-qwm4-qh6w-59xr](https://github.com/advisories/GHSA-qwm4-qh6w-59xr).
The dev extra explicitly requires pip >=26.2.0 (locked as the equivalent 26.2)
and declares the already-locked `packaging` parser dependency. No audit ignore or
severity downgrade is used. [Scope, reproducibility, and measured evidence](development_audit.md)
describe the additional coverage and remaining gaps. A passing `dev` audit does
not close a default-branch Dependabot alert before a checked forward promotion.

## Release automation

A semantic `vMAJOR.MINOR.PATCH` tag triggers the `Release` workflow only. The
workflow fails unless:

- the tagged commit is an ancestor of `origin/main`;
- the tag exactly matches `alphaforge.__version__`;
- locked wheel and source distributions build;
- a clean virtual environment installs and imports the wheel; and
- SHA-256 checksums are created.

The workflow retains the immutable candidate artifacts for 30 days and creates
one GitHub release from the verified tag. It does not publish to PyPI, mutate a
package version, create a tag, weaken branch protection, or receive market-data
credentials. A pre-existing release for the tag fails closed instead of being
silently replaced.

## Branch and rollback policy

`dev`, `prod`, and `main` prohibit force pushes and deletion, require current
status checks and resolved conversations, and enforce the repository's
work-branch and promotion-PR flow. Ordinary work is squash-merged to `dev`;
validated promotions use merge commits from `dev` to `prod` and `prod` to
`main`.

Rollback preserves history: revert the affected squash or release merge, open
the corresponding PR through the same branch flow, and issue a new semantic
release. Never move an existing release tag or replace published checksums.
