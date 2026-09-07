# Signal Foundry

Local-first quantitative research, from data provenance to strategy and risk evidence.

This repository assembles [AlphaForge](https://github.com/srgangaram-swe/AlphaForge)
and [Signalattice](https://github.com/srgangaram-swe/Signalattice) without rewriting
their histories or merging their dependency environments. Both source repositories
remain intact. The Nexus workstation and unified control plane are not yet
delivered on this branch.

**Research and simulation only. No broker connection or live-order route.**
The recorded capital-readiness verdict remains `NOT_READY`. Tests establish
software properties, not a profitable trading strategy.

## Repository layout

- `packages/alphaforge`: unchanged research, strategy, portfolio, risk and execution
  simulation package, with its original tests, C++ core, dashboards and evidence.
- `packages/signalattice`: unchanged data, provenance, forecasting and read-only
  observability package, including its existing web console.
- `foundry_build`: offline history verification and package-context adapters.
- `provenance/assembly.json`: frozen source identities, object hashes, namespaced
  ref mappings and the recorded historical-license determination.
- `apps/nexus`: planned in [AlphaForge #81](https://github.com/srgangaram-swe/AlphaForge/issues/81).

## Verify a checkout

Use a full clone, Python 3.13 and the committed uv resolution. Source environments
remain separate: AlphaForge and Signalattice currently lock different pandas and
other integration dependencies.

```bash
git clone https://github.com/srgangaram-swe/SignalFoundry.git
cd SignalFoundry
git switch dev
uv sync --locked --extra dev
uv run python -m foundry_build.assembly verify
uv run python -m foundry_build.context alphaforge
uv run python -m foundry_build.context signalattice
uv run pytest
```

The context commands generate ignored package-local `.git` pointers backed by
independent object copies inside the unified `.git` directory. They do not clone
source worktrees or fetch credentials/network data. They allow original tools to
resolve original root-relative source paths and tags while operating **inside the
prefixed package directories**. Do not commit from these compatibility contexts;
make reviewed changes on a work branch at the unified repository root.

See [assembly architecture and recovery](docs/assembly.md),
[validation evidence](docs/assembly_validation.md), [contribution workflow](CONTRIBUTING.md)
and [security boundaries](SECURITY.md).

![Verified source preservation counts](docs/evidence/assembly/preservation.png)

## Preserved limitation

[Signalattice #67](https://github.com/srgangaram-swe/Signalattice/issues/67)
tracks an existing clean-container metadata reproducibility failure. Its source
workflow is retained as a visible, non-required job, matching the source branch
policy. All source-required checks remain mandatory through aggregate gates.
Assembly does not resolve that issue or qualify live deployment.
