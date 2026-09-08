import { useEffect, useRef, useState } from "react";
import {
  ApiError,
  type PaperAction,
  type PaperStatus,
  type ResearchClient,
} from "./api";

const actions: readonly [PaperAction["operation"], string][] = [
  ["initialize", "Freeze configuration"],
  ["probe", "Check paper connection"],
  ["acquire", "Acquire selected history"],
  ["research", "Evaluate frozen hypotheses"],
  ["qualify", "Verify qualification"],
  ["start", "Start qualified paper session"],
  ["cycle", "Run one decision"],
  ["reconcile", "Reconcile account"],
  ["record-session", "Record completed session"],
  ["campaign", "Audit paper campaign"],
  ["cancel", "Stop and cancel owned orders"],
];

/** No orders, credentials, paths or qualification verdicts originate in the UI. */
export function PaperConsole({ client }: { client: ResearchClient }) {
  const [status, setStatus] = useState<PaperStatus | null>(null);
  const [symbol, setSymbol] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [artifact, setArtifact] = useState<string | null>(null);
  const [accountDigest, setAccountDigest] = useState<string | null>(null);
  const epoch = useRef(0);
  const pending = useRef(new Set<AbortController>());

  useEffect(() => {
    const controller = new AbortController();
    const controllers = pending.current;
    controllers.add(controller);
    const generation = ++epoch.current;
    client
      .paperStatus(controller.signal)
      .then((value) => {
        if (epoch.current === generation) setStatus(value);
      })
      .catch((cause: unknown) => {
        if (!controller.signal.aborted && epoch.current === generation)
          setError(
            cause instanceof ApiError
              ? cause.message
              : "Paper status is unavailable.",
          );
      })
      .finally(() => controllers.delete(controller));
    return () => {
      epoch.current += 1;
      controllers.forEach((item) => {
        item.abort();
      });
      controllers.clear();
    };
  }, [client]);

  async function operate(operation: PaperAction["operation"] | "refresh") {
    const controller = new AbortController();
    pending.current.add(controller);
    const generation = ++epoch.current;
    setBusy(true);
    setError(null);
    try {
      const selected = symbol ?? status?.symbols[0] ?? null;
      const result =
        operation === "refresh"
          ? {
              status: await client.paperStatus(controller.signal),
              artifact: null,
              account_digest: null,
            }
          : await client.paperAction(
              { operation, symbol: selected },
              controller.signal,
            );
      // A late response from before an emergency stop cannot overwrite it.
      if (epoch.current === generation) {
        setStatus(result.status);
        setArtifact(result.artifact);
        setAccountDigest(result.account_digest);
      }
    } catch (cause) {
      if (!controller.signal.aborted && epoch.current === generation)
        setError(
          cause instanceof ApiError
            ? cause.message
            : "Paper operation failed. Reconcile before another decision.",
        );
    } finally {
      pending.current.delete(controller);
      if (epoch.current === generation) setBusy(false);
    }
  }

  return (
    <section className="panel" aria-labelledby="paper-heading">
      <h2 id="paper-heading">Paper operations</h2>
      <p>
        <strong>Alpaca paper only · Live capability absent</strong>
      </p>
      <p>
        Every decision requires verified qualification and current admission
        checks. Stop prevents future submissions; it does not flatten positions
        or recall an order already in flight.
      </p>
      {error && <p role="alert">{error}</p>}
      {!status && !error && <p role="status">Loading paper state…</p>}
      {status && (
        <>
          <p role="status">
            {status.state} ·{" "}
            {status.stopped ? "STOP ENGAGED" : "Stop available"} ·{" "}
            {status.orders} orders · {status.paper_sessions} session
            observations
          </p>
          <p>
            Feed: {status.feed}. Paper limits (USD):{" "}
            {status.maximum_order_notional} per order /{" "}
            {status.maximum_position_notional} gross position /{" "}
            {status.maximum_session_loss} loss.
          </p>
          <ul>
            {status.blockers.map((blocker) => (
              <li key={blocker}>{blocker}</li>
            ))}
          </ul>
          <p>
            Last reconciled snapshot:{" "}
            {status.account_observed_at ?? "unavailable"}. Paper equity:{" "}
            {status.equity ?? "unknown"} USD; cash: {status.cash ?? "unknown"}{" "}
            USD.
          </p>
          <ul aria-label="Reconciled paper positions">
            {status.positions.map((position) => (
              <li key={position.symbol}>
                {position.symbol}: {position.quantity} shares ·{" "}
                {position.market_value} USD
              </li>
            ))}
          </ul>
          <label>
            Paper universe symbol
            <select
              value={symbol ?? status.symbols[0] ?? ""}
              disabled={busy || !status.configured}
              onChange={(event) => {
                setSymbol(event.target.value);
              }}
            >
              {status.symbols.map((name) => (
                <option key={name}>{name}</option>
              ))}
            </select>
          </label>
          <div className="actions">
            <button
              disabled={busy}
              onClick={() => {
                void operate("refresh");
              }}
            >
              Refresh paper state
            </button>
            {actions.map(([operation, label]) => (
              <button
                key={operation}
                disabled={
                  busy ||
                  !status.configured ||
                  (status.stopped && ["start", "cycle"].includes(operation))
                }
                onClick={() => {
                  void operate(operation);
                }}
              >
                {label}
              </button>
            ))}
            <button
              className="danger"
              disabled={!status.configured || status.stopped}
              onClick={() => {
                void operate("stop");
              }}
            >
              Emergency stop
            </button>
          </div>
          {accountDigest && (
            <p>
              Observed paper account digest: <code>{accountDigest}</code>. Bind
              this digest in the local configuration before starting a session.
            </p>
          )}
          {artifact && (
            <p>
              Private artifact identity: <code>{artifact}</code>. Inspect it
              with the CLI; licensed observations stay outside the browser.
            </p>
          )}
        </>
      )}
    </section>
  );
}
