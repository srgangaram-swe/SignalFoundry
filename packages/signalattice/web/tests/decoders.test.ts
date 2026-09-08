/**
 * Runtime decoding is the check that runs on the bytes that actually arrive.
 * These tests are all negative cases, because the positive one is trivially
 * true and the whole value of the decoder is what it refuses.
 */
import { describe, expect, it } from "vitest";

import {
  MAX_ITEMS_PER_PAGE,
  SUPPORTED_SCHEMA_VERSION,
  comparisonSchema,
  decode,
  hypothesisTestSchema,
  laneSummarySchema,
  lanePageSchema,
} from "../src/api/decoders";

const DIGEST = "a".repeat(64);

function lane(overrides: Record<string, unknown> = {}): Record<string, unknown> {
  return {
    schema_version: SUPPORTED_SCHEMA_VERSION,
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

function test_(overrides: Record<string, unknown> = {}): Record<string, unknown> {
  return {
    schema_version: SUPPORTED_SCHEMA_VERSION,
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
    ...overrides,
  };
}

describe("lane decoding", () => {
  it("accepts a well-formed lane", () => {
    expect(decode(laneSummarySchema, lane()).ok).toBe(true);
  });

  it("refuses an unknown field rather than ignoring it", () => {
    // A server that started sending a new field is a contract change the
    // console must surface, not silently drop.
    const result = decode(laneSummarySchema, lane({ surprise: 1 }));
    expect(result.ok).toBe(false);
  });

  it("refuses a wrong schema version", () => {
    const result = decode(laneSummarySchema, lane({ schema_version: 2 }));
    expect(result.ok).toBe(false);
  });

  it.each([["short"], ["Z".repeat(64)], ["A".repeat(64)], [""]])(
    "refuses the malformed digest %s",
    (value) => {
      expect(decode(laneSummarySchema, lane({ lane_identity: value })).ok).toBe(false);
    },
  );

  it("refuses an unrecognised lane state instead of defaulting it", () => {
    expect(decode(laneSummarySchema, lane({ state: "promoted" })).ok).toBe(false);
  });

  it("refuses an invalid timestamp", () => {
    expect(decode(laneSummarySchema, lane({ created_at: "not-a-date" })).ok).toBe(false);
  });

  it("accepts an unassigned lane with no champion", () => {
    const result = decode(
      laneSummarySchema,
      lane({ state: "unassigned", champion_revision: null, generation: 0 }),
    );
    expect(result.ok).toBe(true);
  });

  it("carries a chain fault through rather than rejecting the lane", () => {
    const result = decode(
      laneSummarySchema,
      lane({ chain_verified: false, chain_fault: "sequence gap at 3" }),
    );
    expect(result.ok).toBe(true);
  });
});

describe("numeric safety", () => {
  it.each([[Number.NaN], [Number.POSITIVE_INFINITY], [Number.NEGATIVE_INFINITY]])(
    "refuses the non-finite estimate %s",
    (value) => {
      expect(decode(hypothesisTestSchema, test_({ point_estimate: value })).ok).toBe(false);
    },
  );

  it("keeps a one-sided interval's unbounded end null", () => {
    const result = decode(hypothesisTestSchema, test_({ interval_low: null }));
    expect(result.ok).toBe(true);
    if (result.ok) expect(result.value.interval_low).toBeNull();
  });

  it("refuses a p-value outside [0, 1]", () => {
    expect(decode(hypothesisTestSchema, test_({ p_value_uncorrected: 1.2 })).ok).toBe(false);
  });

  it("refuses a negative observation count", () => {
    expect(decode(hypothesisTestSchema, test_({ observations: -1 })).ok).toBe(false);
  });

  it("accepts an underpowered test with no estimate at all", () => {
    const result = decode(
      hypothesisTestSchema,
      test_({
        verdict: "underpowered",
        point_estimate: null,
        interval_low: null,
        interval_high: null,
        p_value_uncorrected: null,
      }),
    );
    expect(result.ok).toBe(true);
  });
});

describe("collection bounds", () => {
  it("refuses a page larger than the ceiling", () => {
    const items = Array.from({ length: MAX_ITEMS_PER_PAGE + 1 }, () => lane());
    const result = decode(lanePageSchema, {
      schema_version: SUPPORTED_SCHEMA_VERSION,
      items,
      next_cursor: null,
    });
    expect(result.ok).toBe(false);
  });

  it("accepts an empty page", () => {
    const result = decode(lanePageSchema, {
      schema_version: SUPPORTED_SCHEMA_VERSION,
      items: [],
      next_cursor: null,
    });
    expect(result.ok).toBe(true);
  });
});

describe("decode reporting", () => {
  it("reports the failing field path without echoing the received value", () => {
    const hostile = "<script>alert(1)</script>";
    const result = decode(laneSummarySchema, lane({ state: hostile }));
    expect(result.ok).toBe(false);
    if (!result.ok) {
      expect(result.reason).toContain("state");
      expect(result.reason).not.toContain("<script>");
    }
  });

  it("bounds the reason so a long server message cannot flood the DOM", () => {
    const result = decode(comparisonSchema, { schema_version: 1 });
    expect(result.ok).toBe(false);
    if (!result.ok) expect(result.reason.length).toBeLessThanOrEqual(200);
  });
});
