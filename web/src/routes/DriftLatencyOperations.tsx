/**
 * View 6: bounded operational evidence.
 *
 * The metrics endpoint returns Prometheus text exposition, not JSON, so this
 * view parses a deliberately narrow subset: sample name, labels, and value. A
 * permissive parser here would let arbitrary exposition text drive the DOM.
 *
 * Series are capped. A metric with unbounded label cardinality is reported as
 * truncated rather than expanded into thousands of rows, because the console
 * must not become the thing that exhausts the machine it is observing.
 */
import { useCallback, useState } from "react";

import { MAX_MARKS_PER_PANEL } from "../api/decoders";
import { EvidencePanel } from "../components/EvidencePanel";
import { readText } from "../api/client";
import type { EvidenceStatus } from "../state/evidenceState";
import { empty, errored, loading, partial, ready, classifyTransport } from "../state/evidenceState";
import { ScrollableTable } from "../components/ScrollableTable";

export interface MetricSample {
  readonly name: string;
  readonly labels: string;
  readonly value: number;
}

export interface MetricsView {
  readonly samples: readonly MetricSample[];
  readonly truncated: boolean;
}

/** One exposition line: `name{labels} value`. Anything else is skipped. */
const SAMPLE_LINE = /^([a-zA-Z_:][a-zA-Z0-9_:]*)(\{[^}\n]*\})?\s+(-?[0-9.eE+]+)$/u;

/**
 * Parse Prometheus text exposition into bounded samples.
 *
 * Comments, help text, and malformed lines are skipped rather than rendered.
 * Non-finite values are dropped: `NaN` is a legal exposition value and would
 * otherwise print as a measurement.
 */
export function parseExposition(text: string): MetricsView {
  const samples: MetricSample[] = [];
  let truncated = false;
  for (const raw of text.split("\n")) {
    const line = raw.trim();
    if (line === "" || line.startsWith("#")) continue;
    const match = SAMPLE_LINE.exec(line);
    if (match === null) continue;
    const value = Number(match[3]);
    if (!Number.isFinite(value)) continue;
    if (samples.length >= MAX_MARKS_PER_PANEL) {
      truncated = true;
      break;
    }
    samples.push({ name: match[1] ?? "", labels: match[2] ?? "", value });
  }
  return { samples, truncated };
}

export function DriftLatencyOperations(): React.JSX.Element {
  const [status, setStatus] = useState<EvidenceStatus<MetricsView>>(() => loading<MetricsView>());

  const load = useCallback(() => {
    setStatus(loading<MetricsView>());
    void (async () => {
      const result = await readText("/internal/metrics");
      if (result.kind !== "ok-text") {
        setStatus(
          classifyTransport<MetricsView>(result)
            ?? errored<MetricsView>("The metrics endpoint returned an unusable response."),
        );
        return;
      }
      const view = parseExposition(result.text);
      if (view.samples.length === 0) {
        setStatus(
          empty<MetricsView>(
            "No metric samples were readable. Telemetry may be disabled for this service.",
          ),
        );
        return;
      }
      setStatus(
        view.truncated
          ? partial(view, `Showing the first ${String(MAX_MARKS_PER_PANEL)} samples.`)
          : ready(view, `${String(view.samples.length)} metric samples.`),
      );
    })();
  }, []);

  return (
    <>
      <section className="panel" aria-label="Operational evidence scope">
        <h3>Scope and limits</h3>
        <p className="detail">
          These are local engineering measurements from one process on one machine. They are not a
          service level objective, a capacity claim, or evidence of production readiness.
        </p>
        <button type="button" onClick={load}>
          Read service telemetry
        </button>
        <p className="detail">
          Reading is manual. The console does not poll, so it cannot contribute load to a service
          it is meant to be observing.
        </p>
      </section>

      <EvidencePanel title="Telemetry samples" status={status} headingLevel={3}>
        {(view) => (
          <ScrollableTable label="Telemetry samples">
            <table>
              <caption>
                Bounded to {MAX_MARKS_PER_PANEL} samples. Series with unbounded label cardinality
                are truncated rather than expanded into the page.
              </caption>
              <thead>
                <tr>
                  <th scope="col">Metric</th>
                  <th scope="col">Labels</th>
                  <th scope="col" className="numeric">
                    Value
                  </th>
                </tr>
              </thead>
              <tbody>
                {view.samples.map((sample) => (
                  <tr key={`${sample.name}${sample.labels}`}>
                    <th scope="row">{sample.name}</th>
                    <td>{sample.labels === "" ? "none" : sample.labels}</td>
                    <td className="numeric">{sample.value}</td>
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
