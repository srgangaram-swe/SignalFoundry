import { describe, expect, it, vi } from "vitest";
import {
  ApiError,
  assertEvidence,
  assertJob,
  DEADLINE_MS,
  finiteTree,
  identity,
  ResearchClient,
  RESPONSE_BYTES,
} from "../src/api";
import { isCatalog, isResearchRequest } from "../src/generated/validators.cjs";
import { catalog, evidence, job, json, request, validation } from "./fixtures";

const signal = () => new AbortController().signal;

describe("resolved contracts and semantic boundaries", () => {
  it("requires complete server records and retains nulls", () => {
    expect(isCatalog(catalog)).toBe(true);
    expect(isResearchRequest(request)).toBe(true);
    expect(isResearchRequest({})).toBe(false);
    expect(isCatalog({ ...catalog, live_readiness: "READY" })).toBe(false);
    expect(isCatalog({ ...catalog, unknown: 1 })).toBe(false);
  });

  it.each(["x", "a".repeat(63) + "\n", "a".repeat(64) + "/", "A".repeat(64)])(
    "rejects unsafe identity %s",
    (value) => {
      expect(() => identity(value)).toThrow(ApiError);
    },
  );

  it("rejects inconsistent job and table states", () => {
    expect(() => {
      assertJob(job);
    }).not.toThrow();
    expect(() => {
      assertJob({ ...job, evidence_hash: null });
    }).toThrow(ApiError);
    expect(() => {
      assertJob({ ...job, error_code: "fault" });
    }).toThrow(ApiError);
    expect(() => {
      assertEvidence(evidence);
    }).not.toThrow();
    const first = evidence.tables[0];
    if (!first) throw new Error("Missing fixture table");
    expect(() => {
      assertEvidence({ ...evidence, tables: [first, first] });
    }).toThrow(ApiError);
    expect(() => {
      assertEvidence({ ...evidence, tables: [{ ...first, rows: [[1]] }] });
    }).toThrow(ApiError);
    expect(() => {
      assertEvidence({ ...evidence, tables: [{ ...first, total_rows: 0 }] });
    }).toThrow(ApiError);
    expect(() => {
      assertEvidence({
        ...evidence,
        tables: [{ ...first, columns: [...first.columns, ...first.columns] }],
      });
    }).toThrow(ApiError);
  });

  it("bounds nesting, nodes and numerical overflow without recursion", () => {
    expect(() => {
      finiteTree({ valid: [null, true, 2, "text"] });
    }).not.toThrow();
    expect(() => {
      finiteTree({ invalid: Infinity });
    }).toThrow(ApiError);
    let deep: unknown = null;
    for (let index = 0; index < 34; index += 1) deep = [deep];
    expect(() => {
      finiteTree(deep);
    }).toThrow(ApiError);
    expect(() => {
      finiteTree(Array.from({ length: 450001 }, () => null));
    }).toThrow(ApiError);
  });
});

describe("bounded client", () => {
  it("accepts globally allocated audit sequences with gaps between jobs", async () => {
    const trail = {
      schema_version: "1.0.0",
      job_id: job.job_id,
      events: [
        { sequence: 17, state: "queued", at: job.created_at, code: null },
        { sequence: 23, state: "running", at: job.created_at, code: null },
        { sequence: 30, state: "succeeded", at: job.updated_at, code: null },
      ],
    };
    const client = new ResearchClient(
      vi.fn<typeof fetch>().mockResolvedValue(json(trail)),
    );
    expect(await client.audit(job.job_id, signal())).toEqual(trail);
  });
  it("uses only fixed same-origin routes and no credentials", async () => {
    const transport = vi.fn<typeof fetch>().mockResolvedValue(json(catalog));
    const client = new ResearchClient(transport);
    expect(await client.catalog(signal())).toEqual(catalog);
    expect(transport).toHaveBeenCalledWith(
      "/api/v1/catalog",
      expect.objectContaining({
        mode: "same-origin",
        credentials: "omit",
        redirect: "error",
        cache: "no-store",
      }),
    );
  });

  it("posts explicit research headers and preserves the caller's idempotency key", async () => {
    const transport = vi
      .fn<typeof fetch>()
      .mockResolvedValueOnce(json(validation))
      .mockResolvedValueOnce(json(job, 202))
      .mockResolvedValueOnce(json(job));
    const client = new ResearchClient(transport);
    expect(await client.validate(request, signal())).toEqual(validation);
    expect(
      await client.submit(request, "stable_submission_key", signal()),
    ).toEqual(job);
    expect(await client.cancel(job.job_id, signal())).toEqual(job);
    expect(transport.mock.calls[1]?.[1]?.headers).toEqual({
      Accept: "application/json",
      "Content-Type": "application/json",
      "X-Signal-Foundry-Client": "nexus",
      "Idempotency-Key": "stable_submission_key",
    });
    await expect(
      client.submit(request, "bad\nsubmission_key", signal()),
    ).rejects.toMatchObject({ code: "invalid_key" });
  });

  it("checks evidence, comparison and audit identities", async () => {
    const trail = {
      schema_version: "1.0.0",
      job_id: job.job_id,
      events: [
        { sequence: 1, state: "succeeded", at: job.updated_at, code: null },
      ],
    };
    const pair = {
      schema_version: "1.0.0",
      compatible: true,
      reason: "Test fixture",
      evidence: [evidence, evidence],
    };
    const transport = vi
      .fn<typeof fetch>()
      .mockResolvedValueOnce(json({ schema_version: "1.0.0", jobs: [job] }))
      .mockResolvedValueOnce(json(evidence))
      .mockResolvedValueOnce(json(trail))
      .mockResolvedValueOnce(json(pair))
      .mockResolvedValueOnce(
        json({ ...evidence, request_hash: "b".repeat(64) }),
      )
      .mockResolvedValueOnce(
        json({
          ...trail,
          events: [
            { ...trail.events[0], sequence: 2 },
            { ...trail.events[0], sequence: 1 },
          ],
        }),
      )
      .mockResolvedValueOnce(
        json({
          ...pair,
          evidence: [{ ...evidence, request_hash: "b".repeat(64) }, evidence],
        }),
      );
    const client = new ResearchClient(transport);
    expect((await client.jobs(signal())).jobs).toEqual([job]);
    expect(await client.evidence(job, signal())).toEqual(evidence);
    expect(await client.audit(job.job_id, signal())).toEqual(trail);
    expect(await client.compare(job, job, signal())).toEqual(pair);
    await expect(client.evidence(job, signal())).rejects.toMatchObject({
      code: "invalid_response",
    });
    await expect(client.audit(job.job_id, signal())).rejects.toMatchObject({
      code: "invalid_response",
    });
    await expect(client.compare(job, job, signal())).rejects.toMatchObject({
      code: "invalid_response",
    });
  });

  it.each([
    () =>
      new Response("not JSON", { headers: { "Content-Type": "text/plain" } }),
    () =>
      new Response("{", { headers: { "Content-Type": "application/json" } }),
    () =>
      new Response(new Uint8Array([255]), {
        headers: { "Content-Type": "application/json" },
      }),
    () => json({ ...catalog, models: [catalog.models[0], catalog.models[0]] }),
    () => json({}),
    () => json({}, 500),
  ])(
    "rejects malformed responses without reflecting their bytes",
    async (response) => {
      const client = new ResearchClient(
        vi.fn<typeof fetch>().mockResolvedValue(response()),
      );
      await expect(client.catalog(signal())).rejects.toMatchObject({
        code: "invalid_response",
      });
    },
  );

  it("propagates the typed problem but never retries a mutation", async () => {
    const transport = vi.fn<typeof fetch>().mockResolvedValue(
      json(
        {
          schema_version: "1.0.0",
          code: "workers_busy",
          detail: "Worker capacity is occupied.",
        },
        429,
      ),
    );
    await expect(
      new ResearchClient(transport).validate(request, signal()),
    ).rejects.toMatchObject({ code: "workers_busy" });
    expect(transport).toHaveBeenCalledTimes(1);
  });

  it("rejects declared oversized bodies before reading", async () => {
    const response = new Response("{}", {
      headers: {
        "Content-Type": "application/json",
        "Content-Length": String(RESPONSE_BYTES + 1),
      },
    });
    await expect(
      new ResearchClient(
        vi.fn<typeof fetch>().mockResolvedValue(response),
      ).catalog(signal()),
    ).rejects.toMatchObject({ code: "response_size" });
  });

  it("bounds fragmented and undeclared oversized streams and cancels readers", async () => {
    for (const oversized of [false, true]) {
      let cancelled = false;
      const body = new ReadableStream<Uint8Array>({
        start(controller) {
          if (oversized) controller.enqueue(new Uint8Array(RESPONSE_BYTES + 1));
          else
            for (let index = 0; index < 4097; index += 1)
              controller.enqueue(new Uint8Array([32]));
        },
        cancel() {
          cancelled = true;
        },
      });
      const response = new Response(body, {
        headers: { "Content-Type": "application/json" },
      });
      await expect(
        new ResearchClient(
          vi.fn<typeof fetch>().mockResolvedValue(response),
        ).catalog(signal()),
      ).rejects.toMatchObject({ code: "response_size" });
      expect(cancelled).toBe(true);
    }
  });

  it("enforces four admissions and releases them after faults", async () => {
    let release: (() => void) | undefined;
    const barrier = new Promise<void>((resolve) => {
      release = resolve;
    });
    const transport = vi.fn<typeof fetch>(async () => {
      await barrier;
      return json(catalog);
    });
    const client = new ResearchClient(transport);
    const pending = Array.from({ length: 4 }, () => client.catalog(signal()));
    await expect(client.catalog(signal())).rejects.toMatchObject({
      code: "client_busy",
    });
    release?.();
    await Promise.all(pending);
    expect(await client.catalog(signal())).toEqual(catalog);
  });

  it("bounds deadlines and distinguishes user abort from network failure", async () => {
    vi.useFakeTimers();
    try {
      const transport = vi.fn<typeof fetch>(
        (_input, options) =>
          new Promise<Response>((_resolve, reject) => {
            options?.signal?.addEventListener(
              "abort",
              () => {
                reject(new Error("private transport fault"));
              },
              { once: true },
            );
          }),
      );
      const client = new ResearchClient(transport);
      const waiting = expect(client.catalog(signal())).rejects.toMatchObject({
        code: "deadline",
      });
      await vi.advanceTimersByTimeAsync(DEADLINE_MS);
      await waiting;
      const stop = new AbortController();
      const aborted = expect(client.catalog(stop.signal)).rejects.toMatchObject(
        { code: "aborted" },
      );
      stop.abort();
      await aborted;
      const offline = new ResearchClient(
        vi
          .fn<typeof fetch>()
          .mockRejectedValue(new Error("private network failure")),
      );
      await expect(offline.catalog(signal())).rejects.toMatchObject({
        code: "network",
      });
    } finally {
      vi.useRealTimers();
    }
  });
});
