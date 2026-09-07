# Security boundary

This is a local research repository, not a brokerage or production trading system.
It has no authority to place a live order. Do not put provider or broker credentials
in browser forms, committed configuration, URLs or logs. Use the operating-system
credential store through a bounded, explicitly authorized process when needed.

Assembly tooling accepts reviewed local mirrors and source-generated preservation
ledgers. Hashes establish integrity, not the trustworthiness or ownership of code.
The complete source scanner and pinned security gates remain necessary. Root Git
commands have deadlines and combined stdout/stderr ceilings, disable hooks and
replacement/lazy-fetch behavior, and do not inherit credential environment variables.
Input JSON rejects nonregular files, symlinks, oversized payloads, duplicate keys
and nonstandard numeric constants. Context publication requires a trusted local
parent directory and cooperating writers; it is not an adversarial multi-user
filesystem sandbox.

The owner's determination resolves only the four exact historical missing-license
findings named in the current provenance record. Other license, secret, raw-data,
symlink and object-corruption findings still fail closed. Historical records are
not falsified or removed.

Root CI is read-only, uses full-SHA Action pins and separate package environments.
Source credentialed publication workflows remain inert below package prefixes.
Dependency audit results are time-dependent; passing audits do not prove that all
code, Actions, native libraries or future resolutions are safe.

Report a suspected vulnerability privately using GitHub's private vulnerability
reporting when enabled. Never put exploit credentials, sensitive data or proprietary
material in a public issue. No security reporting channel authorizes access to
accounts, external services or live funds.
