/**
 * The remaining view behaviours: the shell's navigation contract, the run
 * evidence view, the operations view's live read, and the interaction paths
 * (pagination, refresh, skipped reads) the state tests do not reach.
 */
import { describe, expect, it, vi } from "vitest";
import { render, screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { MemoryRouter } from "react-router";

import { CONSOLE_VIEWS } from "../src/App";
import { CalibrationUncertainty } from "../src/routes/CalibrationUncertainty";
import { DriftLatencyOperations } from "../src/routes/DriftLatencyOperations";
import { GovernanceReadiness } from "../src/routes/GovernanceReadiness";
import { RunCatalog } from "../src/routes/RunCatalog";
import { RunEvidence } from "../src/routes/RunEvidence";
import { SystemOverview } from "../src/routes/SystemOverview";
import { jsonResponse } from "./setup";

const DIGEST = "a".repeat(64);

function stub(handler: (path: string) => Response): void {
  vi.stubGlobal(
    "fetch",
    vi.fn(async (input: string) => handler(input)),
  );
}

describe("console view inventory", () => {
  it("declares exactly the seven specified views", () => {
    expect(CONSOLE_VIEWS.map((view) => view.path)).toEqual([
      "/console",
      "/console/runs",
      "/console/evidence",
      "/console/comparison",
      "/console/calibration",
      "/console/operations",
      "/console/governance",
    ]);
  });

  it.each(["admin", "sql", "orders", "positions", "pnl", "settings", "export"])(
    "has no %s view",
    (forbidden) => {
      expect(CONSOLE_VIEWS.some((view) => view.path.includes(forbidden))).toBe(false);
    },
  );

  it("gives every view a distinct human-readable label", () => {
    const labels = new Set(CONSOLE_VIEWS.map((view) => view.label));
    expect(labels.size).toBe(CONSOLE_VIEWS.length);
  });
});

describe("run evidence", () => {
  it("waits for a run reference rather than guessing one", () => {
    stub(() => jsonResponse({}));
    render(<RunEvidence />);
    expect(screen.getByText(/Enter a run reference/u)).toBeInTheDocument();
  });

  it("flags unverified artifacts while still showing them", async () => {
    stub(() =>
      jsonResponse({
        schema_version: 1,
        items: [
          { digest: "d1", role: "forecast", media_type: "application/json", verified: true },
          { digest: "d2", role: "model_card", media_type: "application/json", verified: false },
        ],
        next_cursor: null,
      }),
    );
    render(<RunEvidence />);
    await userEvent.type(screen.getByLabelText(/Run reference/u), "r1_abc");
    await userEvent.click(screen.getByRole("button", { name: /Load evidence/u }));
    await waitFor(() => {
      expect(screen.getByText(/1 of 2 artifacts are not verified/u)).toBeInTheDocument();
    });
    // Hiding the unverified row would make an incomplete lineage look complete.
    // Scoped to the body because "Verified" is also the column header.
    const body = screen.getAllByRole("rowgroup")[1];
    expect(body).toBeDefined();
    expect(within(body!).getByText("Not verified")).toBeInTheDocument();
    expect(within(body!).getByText("Verified")).toBeInTheDocument();
  });

  it("reports a further page as partial", async () => {
    stub(() =>
      jsonResponse({
        schema_version: 1,
        items: [{ digest: "d1", role: "forecast", verified: true }],
        next_cursor: "next",
      }),
    );
    render(<RunEvidence />);
    await userEvent.type(screen.getByLabelText(/Run reference/u), "r1_abc");
    await userEvent.click(screen.getByRole("button", { name: /Load evidence/u }));
    await waitFor(() => {
      expect(screen.getByText(/More artifacts exist/u)).toBeInTheDocument();
    });
  });

  it("renders 'not recorded' rather than an empty cell for absent metadata", async () => {
    stub(() =>
      jsonResponse({ schema_version: 1, items: [{ digest: "d1", verified: true }], next_cursor: null }),
    );
    render(<RunEvidence />);
    await userEvent.type(screen.getByLabelText(/Run reference/u), "r1_abc");
    await userEvent.click(screen.getByRole("button", { name: /Load evidence/u }));
    await waitFor(() => {
      expect(screen.getAllByText("not recorded").length).toBeGreaterThan(0);
    });
  });
});

describe("operations", () => {
  it("does not read telemetry until asked", () => {
    const spy = vi.fn(async () => new Response("x 1", { status: 200 }));
    vi.stubGlobal("fetch", spy);
    render(<DriftLatencyOperations />);
    // No polling: opening the view must not add load to the service.
    expect(spy).not.toHaveBeenCalled();
  });

  it("states that these are local measurements, not an objective", () => {
    stub(() => new Response("x 1", { status: 200 }));
    render(<DriftLatencyOperations />);
    expect(screen.getByText(/not a service level objective/u)).toBeInTheDocument();
  });

  it("reports unreadable exposition as empty rather than inventing samples", async () => {
    stub(() => new Response("# HELP only, no samples", { status: 200 }));
    render(<DriftLatencyOperations />);
    await userEvent.click(screen.getByRole("button", { name: /Read service telemetry/u }));
    await waitFor(() => {
      expect(screen.getByText(/No metric samples were readable/u)).toBeInTheDocument();
    });
  });

  it("surfaces an unavailable telemetry endpoint", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn(async () => {
        throw new TypeError("refused");
      }),
    );
    render(<DriftLatencyOperations />);
    await userEvent.click(screen.getByRole("button", { name: /Read service telemetry/u }));
    await waitFor(() => {
      expect(screen.getByText(/not reachable at this origin/u)).toBeInTheDocument();
    });
  });
});

describe("pagination and refresh", () => {
  it("advances the run catalog by cursor and never shows a page number", async () => {
    let call = 0;
    vi.stubGlobal(
      "fetch",
      vi.fn(async (input: string) => {
        call += 1;
        const first = !input.includes("cursor=");
        return jsonResponse({
          schema_version: 1,
          items: [{ run_id: first ? "r1_first" : "r1_second", status: "ok" }],
          next_cursor: first ? "cursor-2" : null,
        });
      }),
    );
    render(<RunCatalog />);
    // DigestText renders the value twice on purpose: an abbreviated form for
    // sighted readers and the full value for assistive technology.
    await waitFor(() => {
      expect(screen.getAllByText(/r1_first/u).length).toBeGreaterThan(0);
    });
    await userEvent.click(screen.getByRole("button", { name: "Next page" }));
    await waitFor(() => {
      expect(screen.getAllByText(/r1_second/u).length).toBeGreaterThan(0);
    });
    expect(call).toBe(2);
    expect(screen.queryByText(/page 2/iu)).not.toBeInTheDocument();
  });

  it("advances governance lanes by cursor", async () => {
    const lane = (identity: string, cursor: string | null): Response =>
      jsonResponse({
        schema_version: 1,
        items: [
          {
            schema_version: 1,
            lane_identity: identity,
            purpose: "shadow-eval",
            target: "direction",
            horizon_days: 5,
            frequency: "daily",
            universe: "us-large-cap",
            decision_policy: "long-short",
            environment: "local",
            state: "unassigned",
            champion_revision: null,
            generation: 0,
            freeze_trigger: null,
            created_at: "2026-08-01T00:00:00+00:00",
            event_count: 0,
            chain_verified: true,
            chain_fault: null,
            events_by_kind: [],
          },
        ],
        next_cursor: cursor,
      });
    vi.stubGlobal(
      "fetch",
      vi.fn(async (input: string) =>
        input.includes("cursor=") ? lane("c".repeat(64), null) : lane(DIGEST, "b".repeat(64)),
      ),
    );
    render(<GovernanceReadiness />);
    await waitFor(() => {
      expect(screen.getByRole("button", { name: "Next page" })).toBeEnabled();
    });
    await userEvent.click(screen.getByRole("button", { name: "Next page" }));
    await waitFor(() => {
      expect(screen.getByRole("button", { name: /No further pages/u })).toBeDisabled();
    });
    // An unassigned lane reports no champion rather than an empty cell.
    expect(screen.getByText("none assigned")).toBeInTheDocument();
  });

  it("re-reads service status when the operator asks", async () => {
    const spy = vi.fn(async (input: string) =>
      jsonResponse(input.includes("ready") ? { status: "ready" } : { status: "live" }),
    );
    vi.stubGlobal("fetch", spy);
    render(<SystemOverview />);
    await waitFor(() => {
      expect(spy).toHaveBeenCalledTimes(2);
    });
    await userEvent.click(screen.getByRole("button", { name: /Re-read service status/u }));
    await waitFor(() => {
      expect(spy).toHaveBeenCalledTimes(4);
    });
  });

  it("reports a non-live service as unavailable rather than as live", async () => {
    stub((input) =>
      jsonResponse(input.includes("ready") ? { status: "ready" } : { status: "draining" }),
    );
    render(<SystemOverview />);
    await waitFor(() => {
      expect(screen.getByText(/reports liveness "draining"/u)).toBeInTheDocument();
    });
  });
});

describe("calibration sufficiency", () => {
  it("accepts an aggregate that clears the observation floor", async () => {
    stub(() =>
      jsonResponse({
        schema_version: 1,
        items: [
          {
            schema_version: 1,
            aggregates: [{ label: "up", observations: 200, brier_score: 0.18, log_score: 0.52 }],
            limitations: ["Historical fixture only; no prospective claim."],
          },
        ],
        next_cursor: null,
      }),
    );
    render(<CalibrationUncertainty />);
    await userEvent.type(screen.getByLabelText(/Run reference/u), "r1_abc");
    await userEvent.click(screen.getByRole("button", { name: /Load calibration evidence/u }));
    // READY renders no banner, so the evidence itself is the assertion.
    await waitFor(() => {
      expect(screen.getByRole("table")).toBeInTheDocument();
    });
    expect(screen.queryByTestId("state-banner")).not.toBeInTheDocument();
    const table = screen.getByRole("table");
    expect(within(table).getByText("0.1800")).toBeInTheDocument();
    expect(screen.getByText(/no prospective claim/u)).toBeInTheDocument();
  });

  it("reports an empty summary set as empty", async () => {
    stub(() => jsonResponse({ schema_version: 1, items: [], next_cursor: null }));
    render(<CalibrationUncertainty />);
    await userEvent.type(screen.getByLabelText(/Run reference/u), "r1_abc");
    await userEvent.click(screen.getByRole("button", { name: /Load calibration evidence/u }));
    await waitFor(() => {
      expect(screen.getByText(/No forecast summaries are recorded/u)).toBeInTheDocument();
    });
  });

  it("prints 'not reported' when a score is absent rather than a zero", async () => {
    stub(() =>
      jsonResponse({
        schema_version: 1,
        items: [{ schema_version: 1, aggregates: [{ label: "up", observations: 100 }] }],
        next_cursor: null,
      }),
    );
    render(<CalibrationUncertainty />);
    await userEvent.type(screen.getByLabelText(/Run reference/u), "r1_abc");
    await userEvent.click(screen.getByRole("button", { name: /Load calibration evidence/u }));
    await waitFor(() => {
      expect(screen.getAllByText("not reported").length).toBe(2);
    });
  });

  it("reports a further summary page as partial", async () => {
    stub(() =>
      jsonResponse({
        schema_version: 1,
        items: [
          { schema_version: 1, aggregates: [{ label: "up", observations: 100, brier_score: 0.2 }] },
        ],
        next_cursor: "more",
      }),
    );
    render(<CalibrationUncertainty />);
    await userEvent.type(screen.getByLabelText(/Run reference/u), "r1_abc");
    await userEvent.click(screen.getByRole("button", { name: /Load calibration evidence/u }));
    await waitFor(() => {
      expect(screen.getByText(/More summaries exist/u)).toBeInTheDocument();
    });
  });
});

describe("shell routing", () => {
  it("renders an unknown path as a bounded message, not a view", async () => {
    const { App } = await import("../src/App");
    stub(() => jsonResponse({ status: "live" }));
    render(
      <MemoryRouter initialEntries={["/console/nope"]}>
        <App />
      </MemoryRouter>,
    );
    expect(screen.getByText(/This console has exactly seven views/u)).toBeInTheDocument();
  });

  it("offers a skip link before the navigation", async () => {
    const { App } = await import("../src/App");
    stub(() => jsonResponse({ status: "live" }));
    render(
      <MemoryRouter initialEntries={["/console"]}>
        <App />
      </MemoryRouter>,
    );
    expect(screen.getByRole("link", { name: /Skip to main content/u })).toBeInTheDocument();
    expect(screen.getByRole("navigation", { name: "Console views" })).toBeInTheDocument();
  });
});
