# Intraday paper operations

Signal Foundry can explicitly acquire Alpaca intraday history, evaluate two frozen
hypotheses, verify the preserved AlphaForge qualification rubric, and operate a
bounded paper account through the CLI and Nexus. Default launch has no paper
configuration. Live capability is absent. No qualified strategy or real prospective
campaign is supplied by this release; the diagnostic report always says `NO_GO`.

## Setup and account binding

Prepare the root and AlphaForge locked environments as described in the README.
Use a dedicated Alpaca **paper** account with no positions or working orders.
Create two generic-password items in macOS Keychain Access, with service
`com.signal-foundry.alpaca-paper` and account labels `api-key-id` and
`api-secret-key`. Enter values through Keychain Access, not a shell argument,
repository file, browser form or chat. Broker environment variables are refused.
The worker reads these two items with a five-second credential lookup deadline.

Copy `configs/paper.example.json` under ignored `var/`, retain `enabled: false`,
and choose a prospective research plan before acquiring data. The example is a
schema example, not a selected trading universe or a profitable strategy. Each
command below uses an explicit config and state root:

```bash
mkdir -p var
cp configs/paper.example.json var/paper-config.json
chmod 600 var/paper-config.json
uv run signal-foundry --paper-config var/paper-config.json \
  --paper-state var/paper-probe paper initialize
uv run signal-foundry --paper-config var/paper-config.json \
  --paper-state var/paper-probe paper probe
```

`probe` returns only the hashed account identity and checks the paper clock. Bind
that digest in the local config, choose conservative limits, and initialize a
**new** state root. To make later paper submission possible, set `enabled: true`
before freezing this new root; it still cannot start without qualification.
Changing enabled, candidate, plan, limits, account or root Python code invalidates
the frozen identity. Keep the probe root; do not rewrite its history.

Use the following explicit prefix for subsequent commands (ordinary shell
function, containing no credentials):

```bash
paper() {
  uv run signal-foundry --paper-config var/paper-config.json \
    --paper-state var/paper-campaign paper "$@"
}
paper initialize
paper status
paper acquire --symbol AAPL
paper acquire --symbol MSFT
paper acquire --symbol SPY
paper research
```

Replace the example symbols with every member of the frozen plan. Acquisitions
are cache-first with immutable raw provider pages, normalized minute bars,
retrieval/plan/config/feed identities and official exchange calendars. Histories
must end before acquisition; holidays and early closes come from the provider.
Provider inclusive endpoints are converted to the internal half-open interval;
implicit current-day symbol remapping is disabled (`asof=-`).
Only regular-session bars enter diagnostics. At most five pages/20,000 bars per
symbol are accepted; a longer interval is refused, never silently truncated.
The 366-day plan ceiling is not a promise to acquire a year within that row budget.

IEX is a single-exchange feed, not consolidated SIP. Entitlements determine
availability. Raw current-vintage bars do not prove historical revision,
corporate-action or point-in-time universe completeness: all three flags remain
false. Provider data stays private under the state root. No additional Nasdaq
bootstrap request is needed for this paper adapter.

## Research and qualification

The momentum and mean-reversion candidates compare the last completed close with
a rolling window mean. Replay executes a signal at the **next** minute open,
charges configured costs on both sides, resets across gaps/session boundaries,
and evaluates 1×, 2× and 3× cost stress. Gap liquidation at the preceding close is
an explicit diagnostic assumption, not proof of an executable fill. Replay is
O(bars), with O(days + window) retained state.

Results are daily USD P&L per one share, with cash and intraday buy-and-hold
baselines; they are not portfolio returns. Selection uses only dates before the
frozen split, with untouched dates retained for assessment. Seeded 500-replicate,
five-day moving-block bootstrap and two-candidate Bonferroni adjustment expose
uncertainty. A 20-date minimum is an engineering sample floor, not a power analysis.
No signal result automatically grants paper qualification.

Obtain an independently reviewed source qualification decision from actual
research. Keep its evidence private. Each evidence document must bind
`plan_identity`, `config_identity`, `evidence_kind` and `evidence_class: "measured"`,
plus the actual measurement, inputs, method and limitations. Fixture evidence
cannot qualify. Import each document explicitly:

```bash
paper import-evidence --file var/reviewed-evidence.json
paper show-artifact --artifact <returned-sha256>
```

Import only preserves content; it does not verify scientific truth. Do not relabel
synthetic results as measured evidence. Place `qualification.json` (mode 0600) in
the campaign root with exactly these envelope fields:

```text
config_identity: SHA-256 of the frozen validated configuration
code_identity: code identity recorded at initialization
valid_until: aware UTC timestamp, after now and at most 30 days after decided_at
decision:
  candidate_id: momentum or mean_reversion, matching the configuration
  plan_hash: frozen Plan identity
  rubric_identity: standard_paper_rubric().identity from preserved AlphaForge
  decided_at: aware UTC timestamp, no later than now
  criteria: all eight source criteria, each with name, observed and evidence
    evidence: [{kind, identifier, content_hash: imported artifact SHA-256}]
```

The unchanged source rubric requires net return over baseline ≥0, adjusted p-value
≤0.05, stress downside net return ≥0, zero insolvent stress paths, maximum drawdown
≥−0.25, top-name concentration ≤0.35, capacity utilization ≤1 and uncertainty lower
bound ≥0, with its original required evidence families. The root worker recomputes
it and calls the original paper authorization contract. Caller `verdict`/`passed`
flags are ignored. Any missing, expired, mismatched or rejected evidence refuses
submission. Verified dossier versions are preserved immutably on renewal.

## Operate, inspect and stop

```bash
paper qualify
paper start
paper cycle --symbol AAPL
paper run --cycles 10
paper reconcile
paper audit --after 0 --limit 100 > var/private-audit-page.json
paper record-session
paper campaign
```

`start` binds a dedicated flat paper account. `run` explicitly performs 1–390
minute iterations across the frozen symbols, never a daemon or automatic startup.
An overrun, error or interrupt persists a stop; uncertain orders are not retried.
Normal completion returns to the operator. Begin only after the full causal
feature window is available in the current regular session. No overnight or
extended-hours decisions are supported; targets become flat in the final ten
minutes, but limit-order nonfill can leave exposure that requires operator action.

To use the same service from Nexus:

```bash
uv run signal-foundry --paper-config var/paper-config.json \
  --paper-state var/paper-campaign serve --nexus
```

Open Paper operations. The panel shows the feed, limits, blockers, last reconciled
cash/equity/positions, observation count and private artifact identity. It exposes
bounded actions and a universe selector; configuration paths, raw history,
credentials and order quantities never come from the browser. Refresh explicitly
while a CLI session runs. Emergency stop remains available during an operation.

```bash
paper stop
paper cancel
paper reconcile
```

Stop is permanent for the state root and prevents future admission. It cannot
recall an in-flight request. Cancel stops first, checks the account, and requests
cancellation of owned orders only. It does **not** flatten positions; inspect the
paper broker, resolve remaining inventory, then reconcile and archive. Never reset
an account or erase an unknown intent to manufacture clean evidence.

`audit` exports ordered private pages with event and parent hashes. Continue from
`next_after`; retain exports under ignored private storage and do not publish raw
account/audit contents.

`record-session` requires current broker time, official close plus one minute,
reconciliation and no working orders; one immutable observation per actual date.
`campaign` shows every missing scheduled date and remaining overnight inventory.
The gate requires at least 42 calendar days, 30 scheduled market sessions and 20
reconciled flat observations; these are necessary, not sufficient. Independent
review, baseline and capital checklist remain separate unmet obligations.

## Bounds and failure handling

| Boundary | Limit / response |
| --- | --- |
| HTTPS | Fixed paper/data hosts; no redirects, proxies, compression or retries |
| Worker | 30 seconds parent deadline, 25 seconds transport budget, ≤32 requests |
| Request rate | ≤100/minute persisted across restarts; clock rollback refuses |
| Input | ≤4 MiB provider response/artifact; bounded strict JSON and Decimal strings |
| Exposure | Whole-share target; supported fractional partial-fill residuals retained; long/flat only |
| Limits | ≤10,000 USD gross, ≤1,000 USD loss ceiling; lower configured values apply |
| Orders | 1–1,000 total intents per root; default 100, never reset automatically |
| Freshness | Clock ≤5 seconds; quotes ≤configured 1–30 seconds; account ≤15 seconds at dispatch |
| Journal | 64 MiB, 100,000 events with stop reserve; 256 artifacts / 512 MiB |

`submission_unknown` preserves ambiguous intent; locate it at the broker and
reconcile, never resubmit. `account_divergence`, `external_order` and
`order_divergence` require accounting investigation. `stale_quote`,
`feature_freshness`, `feature_gap`, `admission_expired` and `market_session` block
stale/cross-session decisions. `provider_auth` requires local credential/account
repair; provider errors do not reflect response bodies. `paper_identity` requires
a new explicitly reviewed root. `paper_capacity` requires stopping and archiving,
not pruning audit history. Private file permissions and checksums detect accidental
corruption; a malicious machine owner is outside this trust boundary.

Cash accounting uses cumulative fill quantity × average price and a one-cent
tolerance. Terminal orders are cached; late corrections, fees, dividends, funding
and other external activity cause conservative breaks instead of silent balance
adjustments. REST snapshots are not atomic. This implementation has no streaming
fill ledger, corporate-action accounting, realistic queue/impact model or live
adapter. Paper fills do not establish live liquidity, capacity or profitability.

## Evidence and outstanding work

See [software validation](paper-validation.md), [the architecture decision](adr/0003-bounded-paper-operations.md)
and [source preservation](nexus-parity.md). Issues #7–#12 retain their empirical
acceptance criteria. Implementation #13 does not close those obligations.

Before risking capital, obtain licensed current point-in-time data and symbol/
corporate-action history; preregister a small economic hypothesis family; require
untouched net performance after realistic spread, fees, impact and selection-bias
correction; conduct the prospective paper campaign and incident rehearsals; then
have independent risk/account-policy review and a new explicit owner decision on
bounded capital. Failure at any evidence gate means no deployment. More code or
an attractive backtest cannot supply favorable odds without that evidence.
