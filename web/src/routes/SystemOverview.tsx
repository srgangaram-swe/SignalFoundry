/**
 * View 1: liveness, readiness, and the contract identity the console binds to.
 *
 * Readiness is reported from the service's own field rather than inferred from
 * a 200. A service can answer `/health/ready` successfully while reporting that
 * its storage is degraded, and reading the status code alone would render that
 * as healthy.
 */
import { useCallback, useState } from "react";

import { liveSchema, readySchema, SUPPORTED_SCHEMA_VERSION } from "../api/decoders";
import { EvidencePanel } from "../components/EvidencePanel";
import type { EvidenceStatus } from "../state/evidenceState";
import { partial, ready, unavailable } from "../state/evidenceState";
import { useEvidence } from "../hooks/useEvidence";

interface Live {
  readonly status: string;
}

interface Ready {
  readonly status: string;
  // `| undefined` is explicit because exactOptionalPropertyTypes distinguishes
  // "key absent" from "key present and undefined", and the decoder produces the
  // latter.
  readonly readiness?: string | undefined;
}

export function SystemOverview(): React.JSX.Element {
  const [token, setToken] = useState(0);
  const refresh = useCallback(() => {
    setToken((value) => value + 1);
  }, []);

  const interpretLive = useCallback((value: Live): EvidenceStatus<Live> => {
    return value.status.toLowerCase() === "live"
      ? ready(value, "The local service process is answering.")
      : unavailable<Live>(`The service reports liveness "${value.status}".`);
  }, []);

  const interpretReady = useCallback((value: Ready): EvidenceStatus<Ready> => {
    const state = value.status.toLowerCase();
    if (state === "ready") return ready(value, "Storage is attached and the contract is served.");
    // Degraded is real evidence, so it renders alongside its warning rather
    // than being replaced by an empty panel.
    return partial(value, `The service reports readiness "${value.status}".`);
  }, []);

  const live = useEvidence({
    path: "/health/live",
    schema: liveSchema,
    interpret: interpretLive,
    refreshToken: token,
  });
  const readiness = useEvidence({
    path: "/health/ready",
    schema: readySchema,
    interpret: interpretReady,
    refreshToken: token,
  });

  return (
    <>
      <section className="panel" aria-label="Console scope">
        <h3>Scope</h3>
        <p className="detail">
          This console is a bounded read-only projection of locally stored evidence. It is not a
          trading system, and nothing shown here authorizes deployment, capital, paper trading, or
          live trading, or constitutes a claim of profitability.
        </p>
        <p className="detail">
          Contract version {SUPPORTED_SCHEMA_VERSION}. Evidence that does not match this version is
          reported as incompatible rather than rendered.
        </p>
        <button type="button" onClick={refresh}>
          Re-read service status
        </button>
        <p className="detail">
          Refresh is manual. The console never polls, so it cannot add load to a service that is
          already saturated.
        </p>
      </section>

      <EvidencePanel title="Liveness" status={live} headingLevel={3}>
        {(value) => <p>Reported status: {value.status}</p>}
      </EvidencePanel>

      <EvidencePanel title="Readiness" status={readiness} headingLevel={3}>
        {(value) => (
          <dl>
            <dt>Status</dt>
            <dd>{value.status}</dd>
            {value.readiness === undefined ? null : (
              <>
                <dt>Detail</dt>
                <dd>{value.readiness}</dd>
              </>
            )}
          </dl>
        )}
      </EvidencePanel>
    </>
  );
}
