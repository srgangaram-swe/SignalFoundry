/**
 * View 4: champion versus challenger, from recorded governance decisions.
 *
 * The scores are never collapsed into a single ranking number. A leaderboard
 * that sorted by one figure would recreate exactly the weighted-score behaviour
 * the governance layer refuses: a challenger that wins on one metric while
 * failing a gate must not out-rank one that passes everything.
 *
 * Each decision therefore renders its own gates, its own tests with intervals,
 * and the multiplicity correction applied to the family -- and a decision whose
 * cohort could not be compared renders as invalid rather than as a low score.
 */
import { useCallback, useId, useState } from "react";

import { comparisonPageSchema, type ComparisonPage } from "../api/decoders";
import { EvidencePanel } from "../components/EvidencePanel";
import { GateTable } from "../components/GateTable";
import { IntervalBar } from "../components/IntervalBar";
import { DigestText } from "../components/DigestText";
import type { EvidenceStatus } from "../state/evidenceState";
import { empty, insufficient, invalid, partial, ready } from "../state/evidenceState";
import { useEvidence } from "../hooks/useEvidence";
import { ScrollableTable } from "../components/ScrollableTable";

/** Display domain for interval bars, in Brier units. */
const INTERVAL_DOMAIN = 0.1;

export function ModelComparison(): React.JSX.Element {
  const [lane, setLane] = useState("");
  const [applied, setApplied] = useState("");
  const inputId = useId();

  const interpret = useCallback((value: ComparisonPage): EvidenceStatus<ComparisonPage> => {
    if (value.items.length === 0) {
      return empty<ComparisonPage>("No comparison has been recorded for this lane.");
    }
    const latest = value.items[value.items.length - 1];
    if (latest === undefined) {
      return empty<ComparisonPage>("No comparison has been recorded for this lane.");
    }
    // The most recent decision determines how the panel reports itself, because
    // that is the decision a reader would otherwise act on.
    if (latest.recommendation === "invalid") {
      return invalid(
        "The most recent comparison is INVALID: its cohort could not be compared honestly.",
        value,
      );
    }
    if (latest.recommendation === "insufficient_evidence") {
      return insufficient(
        "The most recent comparison did not meet the preregistered evidence floor. "
          + "This is a finding about the evidence, not about the models.",
        value,
      );
    }
    if (latest.recommendation === "unknown") {
      return invalid("The recorded recommendation is not one this console understands.", value);
    }
    if (value.truncated) {
      return partial(value, "Older comparisons exist than are shown here.");
    }
    return ready(value, `${String(value.items.length)} recorded comparisons.`);
  }, []);

  const comparisons = useEvidence({
    path: `/api/v1/governance/lanes/${encodeURIComponent(applied)}/comparisons`,
    schema: comparisonPageSchema,
    interpret,
    skip: applied === "",
    skipDetail: "Enter a governance lane identity to load its comparisons.",
  });

  return (
    <>
      <section className="panel" aria-label="Select a lane">
        <h3>Select a governance lane</h3>
        <label htmlFor={inputId}>Lane identity (64 hex characters)</label>
        <input
          id={inputId}
          value={lane}
          onChange={(event) => {
            setLane(event.target.value);
          }}
          type="text"
          inputMode="text"
          maxLength={64}
          pattern="[0-9a-f]{64}"
        />
        <button
          type="button"
          onClick={() => {
            setApplied(lane.trim());
          }}
        >
          Load comparisons
        </button>
        <p className="detail">
          Lane identities are listed in the governance and readiness view.
        </p>
      </section>

      <EvidencePanel title="Recorded comparisons" status={comparisons} headingLevel={3}>
        {(value) => (
          <>
            {value.items.map((comparison) => (
              <section
                key={comparison.sequence}
                className="panel"
                aria-label={`Comparison ${String(comparison.sequence)}`}
              >
                <h4>
                  Decision {comparison.sequence}: {comparison.recommendation.replace(/_/gu, " ")}
                </h4>
                <dl>
                  <dt>Cohort identity</dt>
                  <dd>
                    <DigestText value={comparison.cohort_identity} />
                  </dd>
                  <dt>Frozen policy identity</dt>
                  <dd>
                    <DigestText value={comparison.policy_identity} />
                  </dd>
                  <dt>Multiplicity correction</dt>
                  <dd>
                    {comparison.correction_method === null
                      ? "None recorded"
                      : `${comparison.correction_method} across ${String(
                          comparison.family_size ?? 0,
                        )} tests at alpha ${String(comparison.correction_alpha ?? 0)}`}
                  </dd>
                </dl>

                <GateTable gates={comparison.gates} truncated={comparison.truncated_gates} />

                <ScrollableTable label="Hypothesis tests">
                  <table>
                    <caption>
                      Each test keeps its own interval and uncorrected p-value. The familywise
                      decision is the correction above, not any single p-value here.
                      {comparison.truncated_tests ? " Additional tests were not returned." : ""}
                    </caption>
                    <thead>
                      <tr>
                        <th scope="col">Test</th>
                        <th scope="col">Metric</th>
                        <th scope="col">Verdict</th>
                        <th scope="col" className="numeric">
                          Observations
                        </th>
                        <th scope="col" className="numeric">
                          Blocks
                        </th>
                        <th scope="col">Interval (challenger minus champion)</th>
                        <th scope="col" className="numeric">
                          p (uncorrected)
                        </th>
                      </tr>
                    </thead>
                    <tbody>
                      {comparison.tests.map((test) => (
                        <tr key={`${String(comparison.sequence)}-${test.name}`}>
                          <th scope="row">{test.name.replace(/_/gu, " ")}</th>
                          <td>{test.metric}</td>
                          <td>{test.verdict.replace(/_/gu, " ")}</td>
                          <td className="numeric">{test.observations}</td>
                          <td className="numeric">{test.blocks}</td>
                          <td>
                            {test.verdict === "underpowered" ? (
                              "No interval: the test was underpowered and reports no estimate."
                            ) : (
                              <IntervalBar
                                low={test.interval_low}
                                high={test.interval_high}
                                point={test.point_estimate}
                                domain={INTERVAL_DOMAIN}
                                label={`${test.name} on ${test.metric}`}
                              />
                            )}
                          </td>
                          <td className="numeric">
                            {test.p_value_uncorrected === null
                              ? "not reported"
                              : test.p_value_uncorrected.toFixed(4)}
                          </td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </ScrollableTable>
              </section>
            ))}
            <p className="detail">
              A recommendation is not an authorization. Applying one requires a separate local human
              action that this console cannot perform.
            </p>
          </>
        )}
      </EvidencePanel>
    </>
  );
}
