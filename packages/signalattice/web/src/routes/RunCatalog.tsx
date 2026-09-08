/**
 * View 2: keyset-paginated run summaries.
 *
 * There is no total count and no page number. The registry is append-only, so
 * any count is a snapshot that is stale before it renders; presenting one would
 * be a fiction the console cannot support. Navigation is cursor-based and
 * forward-only, which is what the server's stable ordering actually provides.
 */
import { useCallback, useState } from "react";
import { z } from "zod";

import { DEFAULT_PAGE_SIZE, buildQuery } from "../api/client";
import { MAX_ITEMS_PER_PAGE, SUPPORTED_SCHEMA_VERSION } from "../api/decoders";
import { EvidencePanel } from "../components/EvidencePanel";
import { DigestText } from "../components/DigestText";
import type { EvidenceStatus } from "../state/evidenceState";
import { empty, ready } from "../state/evidenceState";
import { useEvidence } from "../hooks/useEvidence";
import { ScrollableTable } from "../components/ScrollableTable";

const runSummarySchema = z.looseObject({
  run_id: z.string().max(200),
  status: z.string().max(32),
  evidence_class: z.string().max(64).optional(),
  created_at: z.string().max(64).optional(),
});

const runPageSchema = z.looseObject({
  schema_version: z.literal(SUPPORTED_SCHEMA_VERSION),
  items: z.array(runSummarySchema).max(MAX_ITEMS_PER_PAGE),
  next_cursor: z.string().max(512).nullable(),
});

type RunPage = z.infer<typeof runPageSchema>;

export function RunCatalog(): React.JSX.Element {
  const [cursor, setCursor] = useState<string | undefined>(undefined);

  const interpret = useCallback((value: RunPage): EvidenceStatus<RunPage> => {
    if (value.items.length === 0) {
      return empty<RunPage>("No runs are recorded in this registry.");
    }
    return ready(value, `${String(value.items.length)} runs on this page.`);
  }, []);

  const page = useEvidence({
    path: `/api/v1/runs${buildQuery({ page_size: DEFAULT_PAGE_SIZE, cursor })}`,
    schema: runPageSchema,
    interpret,
  });

  const next = page.data?.next_cursor ?? null;

  return (
    <EvidencePanel title="Run catalog" status={page} headingLevel={3}>
      {(value) => (
        <>
          <ScrollableTable label="Run catalog">
            <table>
              <caption>
                Runs are listed in a stable order with a forward cursor. No total count is shown:
                the registry is append-only, so a count would be stale before it rendered.
              </caption>
              <thead>
                <tr>
                  <th scope="col">Run</th>
                  <th scope="col">Status</th>
                  <th scope="col">Evidence class</th>
                  <th scope="col">Created</th>
                </tr>
              </thead>
              <tbody>
                {value.items.map((run) => (
                  <tr key={run.run_id}>
                    <th scope="row">
                      <DigestText value={run.run_id} />
                    </th>
                    <td>{run.status}</td>
                    <td>{run.evidence_class ?? "not recorded"}</td>
                    <td>{run.created_at ?? "not recorded"}</td>
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
        </>
      )}
    </EvidencePanel>
  );
}
