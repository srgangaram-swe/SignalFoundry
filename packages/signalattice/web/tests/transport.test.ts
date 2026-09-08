/**
 * The transport is the console's only network capability, so its refusals are
 * the console's actual security boundary rather than a convention.
 */
import { describe, expect, it, vi } from "vitest";

import {
  MAX_IN_FLIGHT,
  MAX_RESPONSE_BYTES,
  REQUEST_DEADLINE_MS,
  TransportPathError,
  assertSafePath,
  buildQuery,
  inFlightCount,
  readJson,
} from "../src/api/client";
import { jsonResponse } from "./setup";

describe("path allowlist", () => {
  it.each([
    "/api/v1/runs",
    "/api/v1/governance/lanes",
    "/health/live",
    "/internal/metrics",
  ])("admits the same-origin API path %s", (path) => {
    expect(assertSafePath(path)).toBe(path);
  });

  it.each([
    "https://example.com/api/v1/runs",
    "//example.com/api/v1/runs",
    "http://127.0.0.1/api/v1/runs",
    "/api/v1/../../etc/passwd",
    "/api/v1//runs",
    "api/v1/runs",
    "/console/index.html",
    "/",
    "",
  ])("refuses %s", (path) => {
    expect(() => assertSafePath(path)).toThrow(TransportPathError);
  });

  it("refuses a non-string path rather than coercing it", () => {
    expect(() => assertSafePath(undefined as unknown as string)).toThrow(TransportPathError);
  });
});

describe("query building", () => {
  it("omits undefined values and encodes the rest", () => {
    expect(buildQuery({ page_size: 25, cursor: undefined })).toBe("?page_size=25");
  });

  it("returns an empty string when nothing is set", () => {
    expect(buildQuery({ cursor: undefined })).toBe("");
  });

  it("escapes values rather than interpolating them", () => {
    expect(buildQuery({ cursor: "a b&c=d" })).toBe("?cursor=a+b%26c%3Dd");
  });

  it("refuses a non-integer number", () => {
    expect(() => buildQuery({ page_size: 1.5 })).toThrow(TransportPathError);
  });
});

describe("reads", () => {
  it("sends GET without credentials and with no-store caching", async () => {
    const fetchImpl = vi.fn(async () => jsonResponse({ ok: true }));
    await readJson("/api/v1/runs", { fetchImpl: fetchImpl as unknown as typeof fetch });
    const call = fetchImpl.mock.calls[0] as unknown as [string, RequestInit];
    const init = call[1];
    expect(init.method).toBe("GET");
    expect(init.credentials).toBe("omit");
    expect(init.cache).toBe("no-store");
    expect(init.redirect).toBe("error");
    expect(init.referrerPolicy).toBe("no-referrer");
  });

  it("returns the decoded body on success", async () => {
    const fetchImpl = vi.fn(async () => jsonResponse({ items: [] }));
    const result = await readJson("/api/v1/runs", {
      fetchImpl: fetchImpl as unknown as typeof fetch,
    });
    expect(result).toEqual({ kind: "ok", status: 200, body: { items: [] } });
  });

  it("classifies a problem response without throwing", async () => {
    const fetchImpl = vi.fn(async () => jsonResponse({ status: 404 }, { status: 404 }));
    const result = await readJson("/api/v1/runs", {
      fetchImpl: fetchImpl as unknown as typeof fetch,
    });
    expect(result.kind).toBe("problem");
  });

  it("reports a non-JSON body as malformed rather than crashing", async () => {
    const fetchImpl = vi.fn(async () => jsonResponse("<html>not json</html>"));
    const result = await readJson("/api/v1/runs", {
      fetchImpl: fetchImpl as unknown as typeof fetch,
    });
    expect(result).toEqual({ kind: "malformed", reason: "response body is not JSON" });
  });

  it("refuses an oversized response by its declared length before parsing", async () => {
    const fetchImpl = vi.fn(async () =>
      jsonResponse({}, { headers: { "content-length": String(MAX_RESPONSE_BYTES + 1) } }),
    );
    const result = await readJson("/api/v1/runs", {
      fetchImpl: fetchImpl as unknown as typeof fetch,
    });
    expect(result.kind).toBe("too-large");
  });

  it("still enforces the ceiling when no length is declared", async () => {
    // A chunked response has no content-length, so trusting the header alone
    // would leave the ceiling unenforced.
    const body = JSON.stringify({ blob: "x".repeat(MAX_RESPONSE_BYTES + 16) });
    const fetchImpl = vi.fn(async () => new Response(body, { status: 200 }));
    const result = await readJson("/api/v1/runs", {
      fetchImpl: fetchImpl as unknown as typeof fetch,
    });
    expect(result.kind).toBe("too-large");
  });

  it("reports a network failure as offline", async () => {
    const fetchImpl = vi.fn(async () => {
      throw new TypeError("connection refused");
    });
    const result = await readJson("/api/v1/runs", {
      fetchImpl: fetchImpl as unknown as typeof fetch,
    });
    expect(result).toEqual({ kind: "offline" });
  });

  it("aborts at the deadline and never retries", async () => {
    // Fake timers are enabled locally so the deadline can be advanced without
    // stalling dynamic imports in the rest of the suite.
    vi.useFakeTimers({ shouldAdvanceTime: true });
    let attempts = 0;
    const fetchImpl = vi.fn(
      (_path: string, init?: RequestInit) =>
        new Promise<Response>((_resolve, reject) => {
          attempts += 1;
          init?.signal?.addEventListener("abort", () => {
            reject(new DOMException("aborted", "AbortError"));
          });
        }),
    );
    const pending = readJson("/api/v1/runs", {
      fetchImpl: fetchImpl as unknown as typeof fetch,
    });
    await vi.advanceTimersByTimeAsync(REQUEST_DEADLINE_MS + 10);
    expect(await pending).toEqual({ kind: "timeout" });
    expect(attempts).toBe(1);
    vi.useRealTimers();
  });

  it("refuses to exceed the in-flight ceiling", async () => {
    const gate: (() => void)[] = [];
    const fetchImpl = vi.fn(
      () =>
        new Promise<Response>((resolve) => {
          gate.push(() => {
            resolve(jsonResponse({ ok: true }));
          });
        }),
    );
    const held = Array.from({ length: MAX_IN_FLIGHT }, () =>
      readJson("/api/v1/runs", { fetchImpl: fetchImpl as unknown as typeof fetch }),
    );
    // Let the held requests register before the ceiling is probed.
    await Promise.resolve();
    expect(inFlightCount()).toBe(MAX_IN_FLIGHT);
    const refused = await readJson("/api/v1/runs", {
      fetchImpl: fetchImpl as unknown as typeof fetch,
    });
    expect(refused).toEqual({ kind: "malformed", reason: "console concurrency limit reached" });
    for (const release of gate) release();
    await Promise.all(held);
    expect(inFlightCount()).toBe(0);
  });

  it("releases its slot even when the read fails", async () => {
    const fetchImpl = vi.fn(async () => {
      throw new TypeError("boom");
    });
    await readJson("/api/v1/runs", { fetchImpl: fetchImpl as unknown as typeof fetch });
    expect(inFlightCount()).toBe(0);
  });

  it("cannot express a method, so a mutation is unrepresentable", () => {
    // The signature is the guarantee: there is no verb parameter to set.
    expect(readJson.length).toBe(1);
  });
});
