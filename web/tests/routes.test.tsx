/**
 * Every view driven through the states it must represent, against injected
 * responses rather than a live service.
 *
 * The recurring assertion is that an unfavourable or unavailable outcome
 * renders as itself: no view is permitted to fall back to something that looks
 * like a result.
 */
import { describe, expect, it, vi } from "vitest";
import { render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { MemoryRouter } from "react-router";

import { CalibrationUncertainty } from "../src/routes/CalibrationUncertainty";
import { GovernanceReadiness } from "../src/routes/GovernanceReadiness";
import { ModelComparison } from "../src/routes/ModelComparison";
import { RunCatalog } from "../src/routes/RunCatalog";
import { SystemOverview } from "../src/routes/SystemOverview";
import { parseExposition } from "../src/routes/DriftLatencyOperations";
import { jsonResponse } from "./setup";

const DIGEST = "a".repeat(64);
const COHORT = "d".repeat(64);

function lane(overrides: Record<string, unknown> = {}): Record<string, unknown> {
  return {
    schema_version: 1,
    lane_identity: DIGEST,
    purpose: "shadow-eval",
    target: "direction",
    horizon_days: 5,
    frequency: "daily",
    universe: "us-large-cap",
    decision_policy: "long-short",
    environment: "local",
    state: "active",
    champion_revision: DIGEST,
    generation: 1,
    freeze_trigger: null,
    created_at: "2026-08-01T00:00:00+00:00",
    event_count: 2,
    chain_verified: true,
    chain_fault: null,
    events_by_kind: [["assignment", 1]],
    ...overrides,
  };
}

function comparison(overrides: Record<string, unknown> = {}): Record<string, unknown> {
  return {
    schema_version: 1,
    sequence: 1,
    recorded_at: "2026-08-01T00:00:00+00:00",
    recommendation: "retain_champion",
    policy_identity: DIGEST,
    cohort_identity: COHORT,
    decided_at: "2026-08-01T00:00:00+00:00",
    gates: [{ schema_version: 1, name: "minimum_pairs", satisfied: false, detail: "12 of 200" }],
    tests: [
      {
        schema_version: 1,
        name: "superiority",
        metric: "brier",
        verdict: "inconclusive",
        point_estimate: 0.001,
        interval_low: -0.01,
        interval_high: 0.02,
        p_value_uncorrected: 0.4,
        blocks: 30,
        observations: 240,
        margin: null,
      },
    ],
    correction_method: "holm_bonferroni",
    correction_alpha: 0.05,
    family_size: 1,
    truncated_gates: false,
    truncated_tests: false,
    ...overrides,
  };
}

/** Route the injected responses by path so a view sees only its own data. */
function stubFetch(routes: Record<string, () => Response>): void {
  vi.stubGlobal(
    "fetch",
    vi.fn(async (input: string) => {
      const path = input.split("?")[0] ?? input;
      const handler = routes[path];
      if (handler === undefined) {
        return jsonResponse({ status: 404, code: "not_found", title: "x", detail: "y" }, { status: 404 });
      }
      return handler();
    }),
  );
}

/**
 * Views are rendered directly rather than through the app shell.
 *
 * The shell loads every secondary view as a lazy chunk, and a lazily imported
 * module does not resolve inside this DOM environment, so driving the views
 * through it would test the Suspense fallback instead of the view. The composed
 * shell -- routing, code splitting, and focus management -- is covered end to
 * end by Playwright against a real browser.
 */
const VIEWS = {
  "/console": SystemOverview,
  "/console/runs": RunCatalog,
  "/console/comparison": ModelComparison,
  "/console/calibration": CalibrationUncertainty,
  "/console/governance": GovernanceReadiness,
} as const;

function renderAt(path: keyof typeof VIEWS): void {
  const View = VIEWS[path];
  render(
    <MemoryRouter initialEntries={[path]}>
      <View />
    </MemoryRouter>,
  );
}

describe("system overview", () => {
  it("reports a degraded readiness rather than a healthy default", async () => {
    stubFetch({
      "/health/live": () => jsonResponse({ status: "live" }),
      // A 200 that reports degraded storage must not read as ready.
      "/health/ready": () => jsonResponse({ status: "degraded", readiness: "cas_unverified" }),
    });
    renderAt("/console");
    await waitFor(() => {
      expect(screen.getByText(/reports readiness "degraded"/u)).toBeInTheDocument();
    });
    expect(screen.getByText(/cas_unverified/u)).toBeInTheDocument();
  });

  it("states that it authorizes nothing", () => {
    stubFetch({});
    renderAt("/console");
    expect(screen.getByText(/nothing shown here authorizes deployment/u)).toBeInTheDocument();
  });

  it("surfaces an unreachable service as unavailable", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn(async () => {
        throw new TypeError("refused");
      }),
    );
    renderAt("/console");
    await waitFor(() => {
      expect(screen.getAllByText(/not reachable at this origin/u).length).toBeGreaterThan(0);
    });
  });
});

describe("run catalog", () => {
  it("shows no total count", async () => {
    stubFetch({
      "/api/v1/runs": () =>
        jsonResponse({
          schema_version: 1,
          items: [{ run_id: "r1_abc", status: "succeeded", evidence_class: "synthetic" }],
          next_cursor: null,
        }),
    });
    renderAt("/console/runs");
    await waitFor(() => {
      expect(screen.getByText(/No total count is shown/u)).toBeInTheDocument();
    });
    expect(screen.getByRole("button", { name: /No further pages/u })).toBeDisabled();
  });

  it("reports an empty registry as empty, not as a failure", async () => {
    stubFetch({
      "/api/v1/runs": () => jsonResponse({ schema_version: 1, items: [], next_cursor: null }),
    });
    renderAt("/console/runs");
    await waitFor(() => {
      expect(screen.getByText(/No runs are recorded/u)).toBeInTheDocument();
    });
  });

  it("rejects a response that does not match the contract", async () => {
    stubFetch({
      "/api/v1/runs": () => jsonResponse({ schema_version: 2, items: [], next_cursor: null }),
    });
    renderAt("/console/runs");
    await waitFor(() => {
      expect(screen.getByText(/did not match the version-1 contract/u)).toBeInTheDocument();
    });
  });
});

describe("model comparison", () => {
  it("reports an insufficient decision as a finding about the evidence", async () => {
    stubFetch({
      [`/api/v1/governance/lanes/${DIGEST}/comparisons`]: () =>
        jsonResponse({
          schema_version: 1,
          items: [comparison({ recommendation: "insufficient_evidence" })],
          truncated: false,
        }),
    });
    renderAt("/console/comparison");
    await userEvent.type(screen.getByLabelText(/Lane identity/u), DIGEST);
    await userEvent.click(screen.getByRole("button", { name: /Load comparisons/u }));
    await waitFor(() => {
      expect(
        screen.getByText(/finding about the evidence, not about the models/u),
      ).toBeInTheDocument();
    });
  });

  it("reports an invalid cohort as invalid rather than as a low score", async () => {
    stubFetch({
      [`/api/v1/governance/lanes/${DIGEST}/comparisons`]: () =>
        jsonResponse({
          schema_version: 1,
          items: [comparison({ recommendation: "invalid" })],
          truncated: false,
        }),
    });
    renderAt("/console/comparison");
    await userEvent.type(screen.getByLabelText(/Lane identity/u), DIGEST);
    await userEvent.click(screen.getByRole("button", { name: /Load comparisons/u }));
    await waitFor(() => {
      expect(screen.getByText(/could not be compared honestly/u)).toBeInTheDocument();
    });
  });

  it("shows failed gates and the multiplicity correction, and refuses to rank", async () => {
    stubFetch({
      [`/api/v1/governance/lanes/${DIGEST}/comparisons`]: () =>
        jsonResponse({ schema_version: 1, items: [comparison()], truncated: false }),
    });
    renderAt("/console/comparison");
    await userEvent.type(screen.getByLabelText(/Lane identity/u), DIGEST);
    await userEvent.click(screen.getByRole("button", { name: /Load comparisons/u }));
    await waitFor(() => {
      expect(screen.getByText(/12 of 200/u)).toBeInTheDocument();
    });
    expect(screen.getByText(/holm_bonferroni across 1 tests/u)).toBeInTheDocument();
    expect(screen.getByText(/A recommendation is not an authorization/u)).toBeInTheDocument();
  });

  it("prints no interval for an underpowered test", async () => {
    stubFetch({
      [`/api/v1/governance/lanes/${DIGEST}/comparisons`]: () =>
        jsonResponse({
          schema_version: 1,
          items: [
            comparison({
              tests: [
                {
                  schema_version: 1,
                  name: "superiority",
                  metric: "brier",
                  verdict: "underpowered",
                  point_estimate: null,
                  interval_low: null,
                  interval_high: null,
                  p_value_uncorrected: null,
                  blocks: 3,
                  observations: 12,
                  margin: null,
                },
              ],
            }),
          ],
          truncated: false,
        }),
    });
    renderAt("/console/comparison");
    await userEvent.type(screen.getByLabelText(/Lane identity/u), DIGEST);
    await userEvent.click(screen.getByRole("button", { name: /Load comparisons/u }));
    await waitFor(() => {
      expect(screen.getByText(/underpowered and reports no estimate/u)).toBeInTheDocument();
    });
  });
});

describe("calibration", () => {
  it("declines to claim calibration below the observation floor", async () => {
    stubFetch({
      "/api/v1/runs/r1_abc/forecast-summaries": () =>
        jsonResponse({
          schema_version: 1,
          items: [{ schema_version: 1, aggregates: [{ label: "up", observations: 4, brier_score: 0.2 }] }],
          next_cursor: null,
        }),
    });
    renderAt("/console/calibration");
    await userEvent.type(screen.getByLabelText(/Run reference/u), "r1_abc");
    await userEvent.click(screen.getByRole("button", { name: /Load calibration evidence/u }));
    await waitFor(() => {
      expect(screen.getByText(/do not establish calibration/u)).toBeInTheDocument();
    });
    // The scores are still shown, with their sample count beside them.
    expect(screen.getByText("0.2000")).toBeInTheDocument();
  });
});

describe("governance and readiness", () => {
  it("renders a broken chain as invalid while still showing the lane", async () => {
    stubFetch({
      "/api/v1/governance/lanes": () =>
        jsonResponse({
          schema_version: 1,
          items: [lane({ chain_verified: false, chain_fault: "sequence gap at 3" })],
          next_cursor: null,
        }),
    });
    renderAt("/console/governance");
    await waitFor(() => {
      expect(screen.getByText(/does not verify/u)).toBeInTheDocument();
    });
    // Dropping the lane would hide exactly the one worth investigating.
    expect(screen.getByText(/sequence gap at 3/u)).toBeInTheDocument();
    expect(screen.getByText("Broken")).toBeInTheDocument();
  });

  it("states the human authority boundary", async () => {
    stubFetch({
      "/api/v1/governance/lanes": () =>
        jsonResponse({ schema_version: 1, items: [lane()], next_cursor: null }),
    });
    renderAt("/console/governance");
    await waitFor(() => {
      expect(
      screen.getByText(/can approve, apply, roll back, unfreeze, or waive a gate/u),
    ).toBeInTheDocument();
    });
    expect(screen.getByText(/not externally tamper-proof/u)).toBeInTheDocument();
  });

  it("shows a frozen lane with its trigger", async () => {
    stubFetch({
      "/api/v1/governance/lanes": () =>
        jsonResponse({
          schema_version: 1,
          items: [lane({ state: "frozen", freeze_trigger: "hard_integrity" })],
          next_cursor: null,
        }),
    });
    renderAt("/console/governance");
    await waitFor(() => {
      expect(screen.getByText(/Trigger: hard_integrity/u)).toBeInTheDocument();
    });
  });

  it("reports more pages as partial rather than implying completeness", async () => {
    stubFetch({
      "/api/v1/governance/lanes": () =>
        jsonResponse({ schema_version: 1, items: [lane()], next_cursor: "b".repeat(64) }),
    });
    renderAt("/console/governance");
    await waitFor(() => {
      expect(screen.getByText(/More lanes exist/u)).toBeInTheDocument();
    });
  });
});

describe("operations exposition parsing", () => {
  it("skips comments, help text, and malformed lines", () => {
    const view = parseExposition(
      ["# HELP x help", "# TYPE x counter", "x 1", "garbage line", 'y{a="b"} 2'].join("\n"),
    );
    expect(view.samples).toEqual([
      { name: "x", labels: "", value: 1 },
      { name: "y", labels: '{a="b"}', value: 2 },
    ]);
  });

  it("drops a non-finite value rather than printing it as a measurement", () => {
    expect(parseExposition("x NaN\ny 3").samples).toEqual([{ name: "y", labels: "", value: 3 }]);
  });

  it("bounds the sample count and reports the truncation", () => {
    const lines = Array.from({ length: 400 }, (_v, index) => `m${String(index)} 1`).join("\n");
    const view = parseExposition(lines);
    expect(view.samples).toHaveLength(256);
    expect(view.truncated).toBe(true);
  });

  it("returns nothing for empty exposition instead of inventing a sample", () => {
    expect(parseExposition("").samples).toHaveLength(0);
  });
});
