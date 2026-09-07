# Contributing

The maintained branches are `dev` (integration), `prod` (release candidate) and
`main` (released history). Create a scoped work branch from current `origin/dev`,
test it, and open a PR targeting `dev`. Never push implementation directly to a
protected branch, force-push protected history, or bypass a required check.

Ordinary work uses squash merges. Source-history assembly uses a merge commit so
original objects remain ancestors; promotions use `dev → prod → main` merge commits.
New authored commits belong to `srgangaram-swe`, without coauthor trailers. Original
historical authors, timestamps and signed objects are preserved, not rewritten to
change a contributor display. GitHub may use its standard web-flow committer for
server-generated merges.

Keep source namespaces and environments independent. Run commands inside each
package only after `python -m foundry_build.context <source>`. Git metadata there is
a compatibility view of original history; commit new work only from the root.
Do not change preserved package trees as incidental cleanup. Such changes require
a separately reviewed compatibility/provenance design and its own evidence.

Root checks:

```bash
uv sync --locked --extra dev
uv sync --project packages/alphaforge --locked --extra dev --extra data
uv sync --project packages/signalattice --locked --extra dev
uv run python -m foundry_build.context alphaforge
uv run python -m foundry_build.context signalattice
uv run black --check foundry_build signal_foundry tests
uv run ruff check foundry_build signal_foundry tests
uv run mypy foundry_build signal_foundry
FOUNDRY_TEST_COVERAGE=1 uv run pytest --cov=foundry_build \
  --cov=signal_foundry --cov-branch --cov-fail-under=90
uv run python -m foundry_build.contracts --check
npm --prefix contracts ci --ignore-scripts
npm --prefix contracts run generate
npm --prefix contracts run check
git diff --exit-code -- contracts
uv run python -m foundry_build.workflows --check
uv run python -m foundry_build.assembly verify
```

The root workflow generator retains source test commands, coverage thresholds,
action pins and minimum permissions. Its reviewed adaptations are package working
directories, Git compatibility contexts, locked environment bootstrap, prefixed
cache/artifact paths and explicit timeouts. Source release publication is not
activated here; the source release dry-run gate remains mandatory.

Research integration tests instrument the real isolated package workers when
`FOUNDRY_TEST_COVERAGE=1`; this is a test-only boundary, not a runtime feature or
inherited provider environment. There is no skipped stand-in for these tests.
The two source locks remain independent, including their pandas major versions.
Optional source model backends are catalogued as unavailable until explicitly
installed through their own locked extras. Root typing treats these independently
installed packages as external dependencies; typed wire validation and actual
cross-package tests qualify their composition.

PRs must map acceptance criteria to evidence, explain trust boundaries, algorithms,
complexity, errors, tests, compatibility, resource limits, measured outcomes,
limitations and rollback. Do not claim test coverage or remote CI success without
observing it. After merge, explicitly close resolved source issues as Completed,
record the merge/evidence mapping, verify Project Done and milestone counts, and
remove the merged work branch after verifying `dev`.

Never commit local credentials, raw market data, model binaries, environments,
private run artifacts or `ai.md`. Safe reference plots require reproducible
machine-readable evidence and visual inspection. Backtests and synthetic fixtures
must be labelled honestly; no trading-profit or production-readiness claim follows
from a software gate.
