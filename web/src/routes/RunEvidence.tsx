/**
 * View 3: one run's immutable identity, artifacts, and stated limitations.
 *
 * The run is chosen by identifier rather than discovered, because a console
 * that auto-selected "the latest run" would silently change what a reader is
 * looking at between visits.
 */
import { useCallback, useId, useState } from "react";
import { z } from "zod";

import { SUPPORTED_SCHEMA_VERSION, MAX_MARKS_PER_PANEL } from "../api/decoders";
import { EvidencePanel } from "../components/EvidencePanel";
import { DigestText } from "../components/DigestText";
import type { EvidenceStatus } from "../state/evidenceState";
import { partial, ready } from "../state/evidenceState";
import { useEvidence } from "../hooks/useEvidence";
import { ScrollableTable } from "../components/ScrollableTable";

const artifactSchema = z.looseObject({
  digest: z.string().max(200),
  role: z.string().max(64).optional(),
  media_type: z.string().max(128).optional(),
  verified: z.boolean().optional(),
});

const artifactPageSchema = z.looseObject({
  schema_version: z.literal(SUPPORTED_SCHEMA_VERSION),
  items: z.array(artifactSchema).max(MAX_MARKS_PER_PANEL),
  next_cursor: z.string().max(512).nullable(),
});

type ArtifactPage = z.infer<typeof artifactPageSchema>;

export function RunEvidence(): React.JSX.Element {
  const [runId, setRunId] = useState("");
  const [applied, setApplied] = useState("");
  const inputId = useId();

  const interpret = useCallback((value: ArtifactPage): EvidenceStatus<ArtifactPage> => {
    const unverified = value.items.filter((item) => item.verified === false).length;
    if (unverified > 0) {
      // Unverified artifacts are shown *and* flagged. Hiding them would make an
      // incomplete lineage look complete.
      return partial(
        value,
        `${String(unverified)} of ${String(value.items.length)} artifacts are not verified.`,
      );
    }
    if (value.next_cursor !== null) {
      return partial(value, "More artifacts exist than are shown on this page.");
    }
    return ready(value, `${String(value.items.length)} artifacts, all verified.`);
  }, []);

  const artifacts = useEvidence({
    path: `/api/v1/runs/${encodeURIComponent(applied)}/artifacts`,
    schema: artifactPageSchema,
    interpret,
    skip: applied === "",
    skipDetail: "Enter a run reference to load its evidence.",
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
          // Text only: this value becomes a path segment and is encoded before
          // it is sent, and it is never interpreted as markup on the way back.
          type="text"
          inputMode="text"
          maxLength={200}
        />
        <button
          type="button"
          onClick={() => {
            setApplied(runId.trim());
          }}
        >
          Load evidence
        </button>
      </section>

      <EvidencePanel title="Artifacts and lineage" status={artifacts} headingLevel={3}>
        {(value) => (
          <ScrollableTable label="Run artifacts and lineage">
            <table>
              <caption>
                Every artifact bound to this run, including any the registry could not verify.
                Exclusions are reported, never dropped.
              </caption>
              <thead>
                <tr>
                  <th scope="col">Digest</th>
                  <th scope="col">Role</th>
                  <th scope="col">Media type</th>
                  <th scope="col">Verified</th>
                </tr>
              </thead>
              <tbody>
                {value.items.map((artifact) => (
                  <tr key={artifact.digest}>
                    <th scope="row">
                      <DigestText value={artifact.digest} />
                    </th>
                    <td>{artifact.role ?? "not recorded"}</td>
                    <td>{artifact.media_type ?? "not recorded"}</td>
                    <td>
                      <span
                        className="status"
                        data-tone={artifact.verified === true ? "ready" : "fail"}
                      >
                        <span className="glyph" aria-hidden="true">
                          {artifact.verified === true ? "✓" : "✕"}
                        </span>
                        <span>{artifact.verified === true ? "Verified" : "Not verified"}</span>
                      </span>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </ScrollableTable>
        )}
      </EvidencePanel>
    </>
  );
}
