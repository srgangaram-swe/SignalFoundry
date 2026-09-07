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
uv run black --check foundry_build tests
uv run ruff check foundry_build tests
uv run mypy foundry_build
uv run pytest --cov=foundry_build --cov-branch --cov-fail-under=90
uv run python -m foundry_build.workflows --check
uv run python -m foundry_build.assembly verify
```

The root workflow generator retains source test commands, coverage thresholds,
action pins and minimum permissions. Its reviewed adaptations are package working
directories, Git compatibility contexts, locked environment bootstrap, prefixed
cache/artifact paths and explicit timeouts. Source release publication is not
activated here; the source release dry-run gate remains mandatory.

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
