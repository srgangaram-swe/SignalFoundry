/**
 * Components are asserted on the properties a reader depends on: status is
 * never colour alone, evidence is never canvas-only, and a panel never renders
 * data over a state that says the data is untrustworthy.
 */
import { describe, expect, it } from "vitest";
import { render, screen, within } from "@testing-library/react";

import { EvidencePanel } from "../src/components/EvidencePanel";
import { GateTable } from "../src/components/GateTable";
import { IntervalBar } from "../src/components/IntervalBar";
import { DigestText } from "../src/components/DigestText";
import { StateBanner, toneFor } from "../src/components/StateBanner";
import { EVIDENCE_STATES } from "../src/state/evidenceState";
import {
  empty,
  errored,
  insufficient,
  invalid,
  loading,
  partial,
  ready,
  stale,
  unavailable,
} from "../src/state/evidenceState";

const DIGEST = "b".repeat(64);

describe("StateBanner", () => {
  it.each(EVIDENCE_STATES)("renders %s with a word and a glyph, not colour alone", (state) => {
    render(<StateBanner state={state} detail="why this state" />);
    const banner = screen.getByTestId("state-banner");
    expect(banner).toHaveAttribute("data-state", state);
    // The label is real text, so a screen reader and a forced-colors theme both
    // convey the same finding the colour does.
    expect(within(banner).getByText(toneFor(state).label)).toBeInTheDocument();
    expect(within(banner).getByText(/why this state/u)).toBeInTheDocument();
  });

  it("announces politely rather than interrupting", () => {
    render(<StateBanner state="ERROR" detail="bad" />);
    const banner = screen.getByTestId("state-banner");
    expect(banner).toHaveAttribute("role", "status");
    expect(banner).toHaveAttribute("aria-live", "polite");
  });

  it("gives the four failure states visibly different labels", () => {
    const labels = new Set(
      (["INSUFFICIENT_EVIDENCE", "INVALID", "STALE", "UNAVAILABLE"] as const).map(
        (state) => toneFor(state).label,
      ),
    );
    expect(labels.size).toBe(4);
  });
});

describe("EvidencePanel", () => {
  it("renders children when the evidence is ready", () => {
    render(
      <EvidencePanel title="Scores" status={ready({ n: 1 })}>
        {(data) => <p>observations {data.n}</p>}
      </EvidencePanel>,
    );
    expect(screen.getByText(/observations 1/u)).toBeInTheDocument();
    expect(screen.queryByTestId("state-banner")).not.toBeInTheDocument();
  });

  it.each([
    ["loading", loading<{ n: number }>()],
    ["empty", empty<{ n: number }>("none")],
    ["insufficient without data", insufficient<{ n: number }>("thin")],
    ["invalid without data", invalid<{ n: number }>("broken")],
    ["unavailable", unavailable<{ n: number }>("down")],
    ["error", errored<{ n: number }>("bad")],
  ])("renders the banner alone when the %s state carries no data", (_name, status) => {
    render(
      <EvidencePanel title="Scores" status={status}>
        {(data) => <p>observations {data.n}</p>}
      </EvidencePanel>,
    );
    expect(screen.getByTestId("state-banner")).toBeInTheDocument();
    expect(screen.queryByText(/observations/u)).not.toBeInTheDocument();
  });

  it.each([
    ["partial", partial({ n: 7 }, "some missing")],
    ["stale", stale({ n: 7 }, "aged")],
    ["insufficient with data", insufficient("below the floor", { n: 7 })],
    ["invalid with data", invalid("chain broken", { n: 7 })],
  ])("shows both the banner and the data for the %s state", (_name, status) => {
    // The evidence is real; hiding it would lose information, and showing it
    // without the banner would present it as current, complete, and valid.
    render(
      <EvidencePanel title="Scores" status={status}>
        {(data) => <p>observations {data.n}</p>}
      </EvidencePanel>,
    );
    expect(screen.getByTestId("state-banner")).toBeInTheDocument();
    expect(screen.getByText(/observations 7/u)).toBeInTheDocument();
  });

  it("names the region so assistive technology can navigate to it", () => {
    render(
      <EvidencePanel title="Governance lanes" status={ready({ n: 1 })}>
        {() => <p>content</p>}
      </EvidencePanel>,
    );
    expect(screen.getByRole("region", { name: "Governance lanes" })).toBeInTheDocument();
  });
});

describe("IntervalBar", () => {
  it("prints every value it draws", () => {
    render(<IntervalBar low={-0.04} high={-0.01} point={-0.02} domain={0.1} label="brier" />);
    expect(screen.getByText(/-0\.0400 to -0\.0100/u)).toBeInTheDocument();
    expect(screen.getByText(/point -0\.0200/u)).toBeInTheDocument();
  });

  it("labels an unbounded end rather than closing it at a number", () => {
    render(<IntervalBar low={null} high={-0.005} point={-0.02} domain={0.1} label="brier" />);
    expect(screen.getByText(/unbounded to -0\.0050/u)).toBeInTheDocument();
  });

  it("gives the drawn bar an accessible name carrying the same numbers", () => {
    render(<IntervalBar low={-0.04} high={0.01} point={null} domain={0.1} label="log score" />);
    expect(
      screen.getByRole("img", { name: /log score: interval -0\.0400 to 0\.0100/u }),
    ).toBeInTheDocument();
  });

  it("survives a degenerate domain without producing a non-finite width", () => {
    render(<IntervalBar low={-1} high={1} point={0} domain={0} label="degenerate" />);
    expect(screen.getByTestId("interval-track")).toBeInTheDocument();
  });
});

describe("GateTable", () => {
  const gates = [
    { schema_version: 1 as const, name: "minimum_pairs", satisfied: false, detail: "12 of 200" },
    { schema_version: 1 as const, name: "cohort_comparable", satisfied: true, detail: "symmetric" },
  ];

  it("lists every gate, passed and failed", () => {
    render(<GateTable gates={gates} truncated={false} />);
    expect(screen.getByRole("rowheader", { name: "minimum_pairs" })).toBeInTheDocument();
    expect(screen.getByRole("rowheader", { name: "cohort_comparable" })).toBeInTheDocument();
    expect(screen.getByText(/1 failed/u)).toBeInTheDocument();
  });

  it("states that gates are absolute so a reader does not average them", () => {
    render(<GateTable gates={gates} truncated={false} />);
    expect(screen.getByText(/no gate can be traded off/u)).toBeInTheDocument();
  });

  it("says so when gates were withheld", () => {
    render(<GateTable gates={gates} truncated />);
    expect(screen.getByText(/Additional gates exist/u)).toBeInTheDocument();
  });
});

describe("DigestText", () => {
  it("abbreviates visually while keeping the full value available", () => {
    render(<DigestText value={DIGEST} />);
    expect(screen.getByText(DIGEST)).toBeInTheDocument();
    expect(screen.getByTitle(DIGEST)).toBeInTheDocument();
  });
});
