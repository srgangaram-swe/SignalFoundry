/**
 * Branch cases the happy paths do not reach: refused reads, cancelled reads,
 * skipped reads, truncated collections, and the degenerate shapes a server is
 * permitted to send.
 */
import { describe, expect, it, vi } from "vitest";
import { render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { renderHook } from "@testing-library/react";
import { z } from "zod";

import { useEvidence } from "../src/hooks/useEvidence";
import { decode, laneSummarySchema } from "../src/api/decoders";
import { CalibrationUncertainty } from "../src/routes/CalibrationUncertainty";
import { DriftLatencyOperations } from "../src/routes/DriftLatencyOperations";
import { ModelComparison } from "../src/routes/ModelComparison";
import { jsonResponse } from "./setup";

const DIGEST = "a".repeat(64);
const COHORT = "d".repeat(64);
const okSchema = z.looseObject({ ok: z.boolean() });

describe("useEvidence", () => {
  it("reports a refused path as an error instead of hanging on LOADING", async () => {
    // The transport throws for a path the console may not send. A panel left on
    // LOADING for ever would leave the reader waiting on evidence never coming.
    vi.stubGlobal("fetch", vi.fn(async () => jsonResponse({ ok: true })));
    const { result } = renderHook(() =>
      useEvidence({ path: "https://elsewhere.example/api/v1/runs", schema: okSchema }),
    );
    await waitFor(() => {
      expect(result.current.state).toBe("ERROR");
    });
    expect(result.current.detail).toMatch(/could not issue this read/u);
  });

  it("reports a skipped read as empty with the caller's reason", () => {
    vi.stubGlobal("fetch", vi.fn(async () => jsonResponse({ ok: true })));
    const { result } = renderHook(() =>
      useEvidence({
        path: "/api/v1/runs",
        schema: okSchema,
        skip: true,
        skipDetail: "Choose a run first.",
      }),
    );
    expect(result.current.state).toBe("EMPTY");
    expect(result.current.detail).toBe("Choose a run first.");
  });

  it("falls back to a default reason when a skipped read gives none", () => {
    vi.stubGlobal("fetch", vi.fn(async () => jsonResponse({ ok: true })));
    const { result } = renderHook(() =>
      useEvidence({ path: "/api/v1/runs", schema: okSchema, skip: true }),
    );
    expect(result.current.detail).toMatch(/Select a subject/u);
  });

  it("does not apply a response that arrives after unmount", async () => {
    let release: ((value: Response) => void) | undefined;
    vi.stubGlobal(
      "fetch",
      vi.fn(
        () =>
          new Promise<Response>((resolve) => {
            release = resolve;
          }),
      ),
    );
    const { result, unmount } = renderHook(() =>
      useEvidence({ path: "/api/v1/runs", schema: okSchema }),
    );
    expect(result.current.state).toBe("LOADING");
    unmount();
    release?.(jsonResponse({ ok: true }));
    await Promise.resolve();
    // A late response must not overwrite a newer view's state.
    expect(result.current.state).toBe("LOADING");
  });

  it("returns READY without an interpreter when the decode succeeds", async () => {
    vi.stubGlobal("fetch", vi.fn(async () => jsonResponse({ ok: true })));
    const { result } = renderHook(() =>
      useEvidence({ path: "/api/v1/runs", schema: okSchema }),
    );
    await waitFor(() => {
      expect(result.current.state).toBe("READY");
    });
  });
});

describe("decoder reporting", () => {
  it("names the root when the payload is not an object at all", () => {
    const result = decode(laneSummarySchema, "not-an-object");
    expect(result.ok).toBe(false);
    if (!result.ok) expect(result.reason.length).toBeGreaterThan(0);
  });

  it("names the root for a null payload", () => {
    expect(decode(laneSummarySchema, null).ok).toBe(false);
  });
});

describe("model comparison edges", () => {
  function comparison(overrides: Record<string, unknown> = {}): Record<string, unknown> {
    return {
      schema_version: 1,
      sequence: 1,
      recorded_at: "2026-08-01T00:00:00+00:00",
      recommendation: "promote",
      policy_identity: DIGEST,
      cohort_identity: COHORT,
      decided_at: "2026-08-01T00:00:00+00:00",
      gates: [{ schema_version: 1, name: "g", satisfied: true, detail: "ok" }],
      tests: [
        {
          schema_version: 1,
          name: "superiority",
          metric: "brier",
          verdict: "favours_challenger",
          point_estimate: -0.02,
          interval_low: -0.04,
          interval_high: -0.01,
          p_value_uncorrected: 0.001,
          blocks: 30,
          observations: 240,
          margin: null,
        },
      ],
      correction_method: null,
      correction_alpha: null,
      family_size: null,
      truncated_gates: false,
      truncated_tests: false,
      ...overrides,
    };
  }

  async function load(body: unknown): Promise<void> {
    vi.stubGlobal("fetch", vi.fn(async () => jsonResponse(body)));
    render(<ModelComparison />);
    await userEvent.type(screen.getByLabelText(/Lane identity/u), DIGEST);
    await userEvent.click(screen.getByRole("button", { name: /Load comparisons/u }));
  }

  it("waits for a lane rather than guessing one", () => {
    vi.stubGlobal("fetch", vi.fn(async () => jsonResponse({})));
    render(<ModelComparison />);
    expect(screen.getByText(/Enter a governance lane identity/u)).toBeInTheDocument();
  });

  it("reports an empty comparison history as empty", async () => {
    await load({ schema_version: 1, items: [], truncated: false });
    await waitFor(() => {
      expect(screen.getByText(/No comparison has been recorded/u)).toBeInTheDocument();
    });
  });

  it("reports an unrecognised recommendation as invalid rather than promoting", async () => {
    await load({ schema_version: 1, items: [comparison({ recommendation: "unknown" })], truncated: false });
    await waitFor(() => {
      expect(screen.getByText(/not one this console understands/u)).toBeInTheDocument();
    });
  });

  it("reports withheld history as partial", async () => {
    await load({ schema_version: 1, items: [comparison()], truncated: true });
    await waitFor(() => {
      expect(screen.getByText(/Older comparisons exist/u)).toBeInTheDocument();
    });
  });

  it("says so when no multiplicity correction was recorded", async () => {
    await load({ schema_version: 1, items: [comparison()], truncated: false });
    await waitFor(() => {
      expect(screen.getByText("None recorded")).toBeInTheDocument();
    });
  });

  it("prints 'not reported' for a missing p-value", async () => {
    await load({
      schema_version: 1,
      items: [
        comparison({
          tests: [
            {
              schema_version: 1,
              name: "non_inferiority",
              metric: "brier",
              verdict: "inconclusive",
              point_estimate: null,
              interval_low: null,
              interval_high: 0.01,
              p_value_uncorrected: null,
              blocks: 30,
              observations: 240,
              margin: 0.01,
            },
          ],
        }),
      ],
      truncated: false,
    });
    await waitFor(() => {
      expect(screen.getByText("not reported")).toBeInTheDocument();
    });
  });
});

describe("operations edges", () => {
  it("labels a sample with no labels rather than leaving a blank cell", async () => {
    vi.stubGlobal("fetch", vi.fn(async () => new Response("x 1", { status: 200 })));
    render(<DriftLatencyOperations />);
    await userEvent.click(screen.getByRole("button", { name: /Read service telemetry/u }));
    await waitFor(() => {
      expect(screen.getByText("none")).toBeInTheDocument();
    });
  });

  it("surfaces a rate-limited telemetry read as unavailable", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn(async () => jsonResponse({ status: 429 }, { status: 429 })),
    );
    render(<DriftLatencyOperations />);
    await userEvent.click(screen.getByRole("button", { name: /Read service telemetry/u }));
    await waitFor(() => {
      expect(screen.getByText(/rate limiting/u)).toBeInTheDocument();
    });
  });
});

describe("calibration edges", () => {
  it("counts an aggregate with no observation field as contributing none", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn(async () =>
        jsonResponse({
          schema_version: 1,
          items: [{ schema_version: 1, aggregates: [{ label: "up", brier_score: 0.2 }] }],
          next_cursor: null,
        }),
      ),
    );
    render(<CalibrationUncertainty />);
    await userEvent.type(screen.getByLabelText(/Run reference/u), "r1_abc");
    await userEvent.click(screen.getByRole("button", { name: /Load calibration evidence/u }));
    await waitFor(() => {
      expect(screen.getByText(/Only 0 scored observations/u)).toBeInTheDocument();
    });
  });

  it("handles a summary with no aggregates at all", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn(async () =>
        jsonResponse({
          schema_version: 1,
          items: [{ schema_version: 1 }],
          next_cursor: null,
        }),
      ),
    );
    render(<CalibrationUncertainty />);
    await userEvent.type(screen.getByLabelText(/Run reference/u), "r1_abc");
    await userEvent.click(screen.getByRole("button", { name: /Load calibration evidence/u }));
    await waitFor(() => {
      expect(screen.getByText(/Only 0 scored observations/u)).toBeInTheDocument();
    });
  });
});
