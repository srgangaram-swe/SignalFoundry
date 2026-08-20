/**
 * The state model exists so that "we could not tell" never renders as
 * "nothing is wrong". These tests hold those states apart.
 */
import { describe, expect, it } from "vitest";

import {
  EVIDENCE_STATES,
  FRESHNESS_WINDOW_MS,
  classifyTransport,
  empty,
  errored,
  insufficient,
  invalid,
  isStale,
  loading,
  partial,
  ready,
  stale,
  unavailable,
} from "../src/state/evidenceState";

describe("state constructors", () => {
  it("declares exactly the nine states the console must render", () => {
    expect([...EVIDENCE_STATES]).toEqual([
      "LOADING",
      "READY",
      "EMPTY",
      "PARTIAL",
      "INSUFFICIENT_EVIDENCE",
      "INVALID",
      "STALE",
      "UNAVAILABLE",
      "ERROR",
    ]);
  });

  it("marks only states with a trustworthy measurement as measuring", () => {
    expect(ready(1).measuring).toBe(true);
    expect(partial(1, "some").measuring).toBe(true);
    expect(stale(1, "old").measuring).toBe(true);
    for (const status of [
      loading<number>(),
      empty<number>("none"),
      insufficient<number>("thin"),
      invalid<number>("broken"),
      unavailable<number>("down"),
      errored<number>("bad"),
    ]) {
      expect(status.measuring).toBe(false);
    }
  });

  it("keeps insufficient and invalid distinct, both carrying their data", () => {
    // Detail first for these two: data is optional because the shortfall is
    // often the whole finding.
    const thin = insufficient("below the floor", 42);
    const broken = invalid("cohort not comparable", 42);
    expect(thin.state).not.toBe(broken.state);
    expect(thin.data).toBe(42);
    expect(broken.data).toBe(42);
  });

  it("gives every non-ready state a reason a reader can act on", () => {
    for (const status of [
      loading<number>(),
      empty<number>("none recorded"),
      insufficient<number>("thin"),
      invalid<number>("broken"),
      unavailable<number>("down"),
      errored<number>("bad"),
    ]) {
      expect(status.detail.length).toBeGreaterThan(0);
    }
  });
});

describe("transport classification", () => {
  it("passes a successful read through untouched", () => {
    expect(classifyTransport({ kind: "ok" })).toBeNull();
  });

  it.each([
    ["timeout", "UNAVAILABLE"],
    ["offline", "UNAVAILABLE"],
    ["too-large", "ERROR"],
    ["malformed", "ERROR"],
  ])("maps %s to %s", (kind, expected) => {
    expect(classifyTransport({ kind })?.state).toBe(expected);
  });

  it.each([
    [404, "EMPTY"],
    [503, "UNAVAILABLE"],
    [429, "UNAVAILABLE"],
    [400, "ERROR"],
    [500, "ERROR"],
  ])("maps problem status %s to %s", (status, expected) => {
    expect(classifyTransport({ kind: "problem", status })?.state).toBe(expected);
  });

  it("treats a problem with no status as an error rather than a success", () => {
    expect(classifyTransport({ kind: "problem" })?.state).toBe("ERROR");
  });

  it("refuses to interpret an outcome it does not recognise", () => {
    expect(classifyTransport({ kind: "surprise" })?.state).toBe("ERROR");
  });
});

describe("freshness", () => {
  const now = Date.parse("2026-08-01T12:00:00Z");

  it("treats evidence inside the window as fresh", () => {
    expect(isStale(now - 1000, now)).toBe(false);
  });

  it("treats evidence past the window as stale", () => {
    expect(isStale(now - FRESHNESS_WINDOW_MS - 1, now)).toBe(true);
  });

  it("treats an unusable instant as stale rather than fresh", () => {
    // Failing closed: an unparseable timestamp must not read as current.
    expect(isStale(Number.NaN, now)).toBe(true);
    expect(isStale(now, Number.NaN)).toBe(true);
  });
});
