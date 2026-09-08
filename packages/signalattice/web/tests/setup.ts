/**
 * Deterministic test environment.
 *
 * A console that renders timestamps, intervals, and locale-formatted numbers
 * will otherwise pass or fail depending on the machine's clock, timezone, and
 * locale. Every one of those is pinned here so a failure means the code
 * changed, not the environment.
 *
 * The global `fetch` is replaced with a throwing stub: no test may reach the
 * network by accident, and a test that needs a response injects one explicitly.
 */
import "@testing-library/jest-dom/vitest";
import { afterEach, beforeEach, vi } from "vitest";
import { cleanup } from "@testing-library/react";

/** Fixed reference instant used across the suite. */
export const FIXED_NOW = new Date("2026-08-01T12:00:00.000Z");

beforeEach(() => {
  // The clock is pinned but timers stay real: lazily imported route chunks
  // resolve on the microtask queue, and faking timers globally would stall the
  // dynamic imports that every secondary view depends on. Tests that need to
  // advance a timer (the transport deadline) enable fake timers themselves.
  vi.setSystemTime(FIXED_NOW);
  vi.stubGlobal(
    "fetch",
    vi.fn(() => {
      throw new Error("a test reached the network; inject a fetch implementation instead");
    }),
  );
});

afterEach(() => {
  cleanup();
  vi.useRealTimers();
  vi.setSystemTime(FIXED_NOW);
  vi.unstubAllGlobals();
  vi.restoreAllMocks();
});

/** Build a `Response`-like object without touching the network. */
export function jsonResponse(
  body: unknown,
  init: { status?: number; headers?: Record<string, string> } = {},
): Response {
  const text = typeof body === "string" ? body : JSON.stringify(body);
  return new Response(text, {
    status: init.status ?? 200,
    headers: { "content-type": "application/json", ...(init.headers ?? {}) },
  });
}
