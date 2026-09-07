/**
 * View 7: lane state, gates, chain history, and the authority boundary.
 *
 * A lane whose event chain fails verification renders as INVALID and keeps
 * rendering its state. Dropping it would remove exactly the lane an operator
 * needs to investigate, and rendering it as healthy would be worse.
 *
 * The non-authorization language is part of the view, not a footnote: this is
 * the screen where a reader is most likely to believe they are looking at a
 * control surface.
 */
import { useCallback, useState } from "react";

import { lanePageSchema, type LanePage, type LaneSummary } from "../api/decoders";
import { DEFAULT_PAGE_SIZE, buildQuery } from "../api/client";
import { EvidencePanel } from "../components/EvidencePanel";
import { DigestText } from "../components/DigestText";
import type { EvidenceStatus } from "../state/evidenceState";
import { empty, invalid, partial, ready } from "../state/evidenceState";
import { useEvidence } from "../hooks/useEvidence";
import { ScrollableTable } from "../components/ScrollableTable";

const STATE_TONE: Readonly<Record<LaneSummary["state"], "ready" | "warn" | "muted">> = {
  active: "ready",
  frozen: "warn",
  unassigned: "muted",
};

export function GovernanceReadiness(): React.JSX.Element {
  const [cursor, setCursor] = useState<string | undefined>(undefined);

  const interpret = useCallback((value: LanePage): EvidenceStatus<LanePage> => {
    if (value.items.length === 0) {
      return empty<LanePage>("No governance lanes are registered.");
    }
    const broken = value.items.filter((lane) => !lane.chain_verified);
    if (broken.length > 0) {
      return invalid(
        `${String(broken.length)} of ${String(value.items.length)} lanes have an event chain that `
          + "does not verify. Their recorded history disagrees with itself and must be "
          + "investigated before any state shown here is relied on.",
        value,
      );
    }
    if (value.next_cursor !== null) {
      return partial(value, "More lanes exist than are shown on this page.");
    }
    return ready(value, `${String(value.items.length)} lanes, all chains verified.`);
  }, []);

  const lanes = useEvidence({
    path: `/api/v1/governance/lanes${buildQuery({ page_size: DEFAULT_PAGE_SIZE, cursor })}`,
    schema: lanePageSchema,
    interpret,
  });

  const next = lanes.data?.next_cursor ?? null;

  return (
    <>
      <section className="panel" aria-label="Human authority boundary">
        <h3>Human authority boundary</h3>
        <p className="detail">
          This view is evidence only. No control in this console -- and no request it can send --
          can approve, apply, roll back, unfreeze, or waive a gate. Promotion requires a separate
          local action by a person, recorded as its own append-only event.
        </p>
        <p className="detail">
          Nothing here authorizes production deployment, capital allocation, paper trading, or live
          trading, and no lane state is a claim of profitability.
        </p>
      </section>

      <EvidencePanel title="Governance lanes" status={lanes} headingLevel={3}>
        {(value) => (
          <>
            <ScrollableTable label="Governance lanes">
              <table>
                <caption>
                  A champion is held relative to a lane, never globally. One revision may be
                  champion in one lane and a rejected challenger in another.
                </caption>
                <thead>
                  <tr>
                    <th scope="col">Lane</th>
                    <th scope="col">Purpose</th>
                    <th scope="col">State</th>
                    <th scope="col">Champion</th>
                    <th scope="col" className="numeric">
                      Generation
                    </th>
                    <th scope="col">Chain</th>
                  </tr>
                </thead>
                <tbody>
                  {value.items.map((lane) => (
                    <tr key={lane.lane_identity}>
                      <th scope="row">
                        <DigestText value={lane.lane_identity} />
                      </th>
                      <td>
                        {lane.purpose} / {lane.target} / {lane.horizon_days}d
                      </td>
                      <td>
                        <span className="status" data-tone={STATE_TONE[lane.state]}>
                          <span className="glyph" aria-hidden="true">
                            {lane.state === "active" ? "✓" : lane.state === "frozen" ? "⏸" : "∅"}
                          </span>
                          <span>{lane.state}</span>
                        </span>
                        {lane.freeze_trigger === null ? null : (
                          <div className="detail">Trigger: {lane.freeze_trigger}</div>
                        )}
                      </td>
                      <td>
                        {lane.champion_revision === null ? (
                          "none assigned"
                        ) : (
                          <DigestText value={lane.champion_revision} />
                        )}
                      </td>
                      <td className="numeric">{lane.generation}</td>
                      <td>
                        <span
                          className="status"
                          data-tone={lane.chain_verified ? "ready" : "fail"}
                        >
                          <span className="glyph" aria-hidden="true">
                            {lane.chain_verified ? "✓" : "✕"}
                          </span>
                          <span>{lane.chain_verified ? "Verified" : "Broken"}</span>
                        </span>
                        {lane.chain_fault === null ? null : (
                          <div className="detail">{lane.chain_fault}</div>
                        )}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </ScrollableTable>
            <button
              type="button"
              disabled={next === null}
              onClick={() => {
                if (next !== null) setCursor(next);
              }}
            >
              {next === null ? "No further pages" : "Next page"}
            </button>
            <p className="detail">
              Local hash chains detect accidental divergence. They are not externally tamper-proof:
              anyone able to rewrite a row can recompute the rest of the chain.
            </p>
          </>
        )}
      </EvidencePanel>
    </>
  );
}
