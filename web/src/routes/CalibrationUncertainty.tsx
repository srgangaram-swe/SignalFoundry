/**
 * View 5: calibration, proper scores, intervals, and their sample counts.
 *
 * Sample counts are rendered next to every score rather than in a footnote. A
 * Brier score over eleven observations and one over eleven thousand are not
 * comparable numbers, and presenting them identically invites exactly that
 * comparison.
 *
 * Where the aggregate does not carry enough observations to support a
 * calibration claim, the panel says so instead of printing the number.
 */
import { useCallback, useId, useState } from "react";
import { z } from "zod";

import { MAX_MARKS_PER_PANEL, SUPPORTED_SCHEMA_VERSION } from "../api/decoders";
import { EvidencePanel } from "../components/EvidencePanel";
import type { EvidenceStatus } from "../state/evidenceState";
import { empty, insufficient, partial, ready } from "../state/evidenceState";
import { useEvidence } from "../hooks/useEvidence";
import { ScrollableTable } from "../components/ScrollableTable";

/** Below this, an aggregate cannot support a calibration claim. */
const MIN_SCORED_OBSERVATIONS = 30;

const aggregateSchema = z.looseObject({
  label: z.string().max(64).optional(),
  observations: z.number().int().min(0).optional(),
  brier_score: z.number().nullable().optional(),
  log_score: z.number().nullable().optional(),
});

const summarySchema = z.looseObject({
  schema_version: z.literal(SUPPORTED_SCHEMA_VERSION),
  aggregates: z.array(aggregateSchema).max(MAX_MARKS_PER_PANEL).optional(),
  limitations: z.array(z.string().max(512)).max(32).optional(),
});

const summaryPageSchema = z.looseObject({
  schema_version: z.literal(SUPPORTED_SCHEMA_VERSION),
  items: z.array(summarySchema).max(MAX_MARKS_PER_PANEL),
  next_cursor: z.string().max(512).nullable(),
});

type SummaryPage = z.infer<typeof summaryPageSchema>;

function totalObservations(page: SummaryPage): number {
  let total = 0;
  for (const item of page.items) {
    for (const aggregate of item.aggregates ?? []) {
      total += aggregate.observations ?? 0;
    }
  }
  return total;
}

export function CalibrationUncertainty(): React.JSX.Element {
  const [runId, setRunId] = useState("");
  const [applied, setApplied] = useState("");
  const inputId = useId();

  const interpret = useCallback((value: SummaryPage): EvidenceStatus<SummaryPage> => {
    if (value.items.length === 0) {
      return empty<SummaryPage>("No forecast summaries are recorded for this run.");
    }
    const observations = totalObservations(value);
    if (observations < MIN_SCORED_OBSERVATIONS) {
      return insufficient(
        `Only ${String(observations)} scored observations are recorded, below the `
          + `${String(MIN_SCORED_OBSERVATIONS)} needed to support a calibration claim. `
          + "The scores are shown, but they do not establish calibration.",
        value,
      );
    }
    if (value.next_cursor !== null) {
      return partial(value, "More summaries exist than are shown on this page.");
    }
    return ready(value, `${String(observations)} scored observations across this run.`);
  }, []);

  const summaries = useEvidence({
    path: `/api/v1/runs/${encodeURIComponent(applied)}/forecast-summaries`,
    schema: summaryPageSchema,
    interpret,
    skip: applied === "",
    skipDetail: "Enter a run reference to load its calibration evidence.",
  });

  return (
    <>
      <section className="panel" aria-label="Select a run">
        <h3>Select a run</h3>
        <label htmlFor={inputId}>Run reference</label>
        <input
          id={inputId}
          value={runId}
          onChange={(event) => {
            setRunId(event.target.value);
          }}
          type="text"
          maxLength={200}
        />
        <button
          type="button"
          onClick={() => {
            setApplied(runId.trim());
          }}
        >
          Load calibration evidence
        </button>
      </section>

      <EvidencePanel title="Proper scores and sample counts" status={summaries} headingLevel={3}>
        {(value) => (
          <>
            <ScrollableTable label="Proper scores and sample counts">
              <table>
                <caption>
                  Brier and log score are strictly proper: lower is better, and both are minimised
                  only by reporting the true probability. Sample counts are shown beside every
                  score because scores over different sample sizes are not comparable.
                </caption>
                <thead>
                  <tr>
                    <th scope="col">Class</th>
                    <th scope="col" className="numeric">
                      Observations
                    </th>
                    <th scope="col" className="numeric">
                      Brier
                    </th>
                    <th scope="col" className="numeric">
                      Log score
                    </th>
                  </tr>
                </thead>
                <tbody>
                  {value.items.flatMap((item, index) =>
                    (item.aggregates ?? []).map((aggregate) => (
                      <tr key={`${String(index)}-${aggregate.label ?? "unlabelled"}`}>
                        <th scope="row">{aggregate.label ?? "unlabelled"}</th>
                        <td className="numeric">{aggregate.observations ?? 0}</td>
                        <td className="numeric">
                          {typeof aggregate.brier_score === "number"
                            ? aggregate.brier_score.toFixed(4)
                            : "not reported"}
                        </td>
                        <td className="numeric">
                          {typeof aggregate.log_score === "number"
                            ? aggregate.log_score.toFixed(4)
                            : "not reported"}
                        </td>
                      </tr>
                    )),
                  )}
                </tbody>
              </table>
            </ScrollableTable>
            <h4>Stated limitations</h4>
            <ul>
              {value.items.flatMap((item, index) =>
                (item.limitations ?? []).map((limitation) => (
                  <li key={`${String(index)}-${limitation.slice(0, 32)}`}>{limitation}</li>
                )),
              )}
            </ul>
          </>
        )}
      </EvidencePanel>
    </>
  );
}
