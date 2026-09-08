/** Bounded same-origin transport. No retries, credentials or executable inputs. */
import * as guards from "./generated/validators.cjs";
import type {
  AuditTrail,
  Catalog,
  Comparison,
  Evidence,
  Job,
  JobPage,
  ResearchRequest,
  Validation,
} from "./types";

export const RESPONSE_BYTES = 8 * 1024 * 1024 + 65536;
export const REQUEST_BYTES = 16384;
export const MAX_IN_FLIGHT = 4;
export const DEADLINE_MS = 30000;
const MAX_CHUNKS = 4096;
type Guard<T> = (value: unknown) => value is T;

export class ApiError extends Error {
  constructor(
    readonly code: string,
    message: string,
    options?: ErrorOptions,
  ) {
    super(message, options);
    this.name = "ApiError";
  }
}

export function identity(value: string): string {
  if (value.length !== 64 || /[^0-9a-f]/u.test(value)) {
    throw new ApiError(
      "invalid_identity",
      "The selected evidence identity is invalid.",
    );
  }
  return value;
}

/** O(nodes), with bounded explicit stack; reject overflow before semantic use. */
export function finiteTree(value: unknown): void {
  const stack: { value: unknown; depth: number }[] = [{ value, depth: 0 }];
  let visited = 0;
  while (stack.length > 0) {
    const item = stack.pop();
    if (!item) break;
    visited += 1;
    if (visited > 450000 || item.depth > 32) {
      throw new ApiError(
        "invalid_response",
        "Response exceeds its structural limits.",
      );
    }
    if (typeof item.value === "number" && !Number.isFinite(item.value)) {
      throw new ApiError(
        "invalid_response",
        "Response contains a non-finite number.",
      );
    }
    if (item.value && typeof item.value === "object") {
      const children: unknown[] = Object.values(item.value);
      if (children.length + stack.length + visited > 450000) {
        throw new ApiError(
          "invalid_response",
          "Response exceeds its structural limits.",
        );
      }
      for (const child of children)
        stack.push({ value: child, depth: item.depth + 1 });
    }
  }
}

function unique(values: readonly string[]): boolean {
  return new Set(values).size === values.length;
}

export function assertJob(job: Job): void {
  identity(job.job_id);
  identity(job.request_hash);
  if (
    (job.state === "succeeded") !== (job.evidence_hash !== null) ||
    (["queued", "running", "succeeded"].includes(job.state) &&
      job.error_code !== null)
  ) {
    throw new ApiError(
      "invalid_response",
      "Job state contradicts its evidence state.",
    );
  }
}

export function assertEvidence(evidence: Evidence): void {
  [
    evidence.request_hash,
    evidence.code_hash,
    evidence.source_code_hash,
    evidence.environment_hash,
  ].forEach(identity);
  if (!unique(evidence.tables.map((table) => table.name))) {
    throw new ApiError(
      "invalid_response",
      "Duplicate evidence table identities.",
    );
  }
  for (const table of evidence.tables) {
    if (
      !unique(table.columns.map((column) => column.name)) ||
      table.total_rows < table.rows.length ||
      table.rows.some((row) => row.length !== table.columns.length)
    ) {
      throw new ApiError(
        "invalid_response",
        "Evidence table shape or count is inconsistent.",
      );
    }
  }
}

async function boundedJson(
  response: Response,
  signal: AbortSignal,
): Promise<unknown> {
  if (
    response.headers.get("content-type")?.split(";", 1)[0] !==
    "application/json"
  ) {
    throw new ApiError(
      "invalid_response",
      "Expected a versioned JSON response.",
    );
  }
  const declared = response.headers.get("content-length");
  if (
    declared !== null &&
    (!/^\d{1,9}$/u.test(declared) || Number(declared) > RESPONSE_BYTES)
  ) {
    throw new ApiError(
      "response_size",
      "Response exceeds its declared byte limit.",
    );
  }
  const reader = response.body?.getReader();
  if (!reader)
    throw new ApiError("invalid_response", "The server returned no JSON body.");
  const chunks: Uint8Array[] = [];
  let size = 0;
  try {
    for (
      let part = await reader.read();
      !part.done;
      part = await reader.read()
    ) {
      signal.throwIfAborted();
      size += part.value.byteLength;
      if (size > RESPONSE_BYTES || chunks.length >= MAX_CHUNKS) {
        throw new ApiError(
          "response_size",
          "Response exceeds its byte or fragment limit.",
        );
      }
      chunks.push(part.value);
    }
  } finally {
    try {
      await reader.cancel();
    } finally {
      reader.releaseLock();
    }
  }
  const bytes = new Uint8Array(size);
  let offset = 0;
  for (const chunk of chunks) {
    bytes.set(chunk, offset);
    offset += chunk.byteLength;
  }
  try {
    const value: unknown = JSON.parse(
      new TextDecoder("utf-8", { fatal: true }).decode(bytes),
    );
    finiteTree(value);
    return value;
  } catch (cause) {
    if (cause instanceof ApiError) throw cause;
    throw new ApiError(
      "invalid_response",
      "Response is not valid finite UTF-8 JSON.",
      { cause },
    );
  }
}

export class ResearchClient {
  private active = 0;

  constructor(
    private readonly transport: typeof globalThis.fetch = globalThis.fetch.bind(
      globalThis,
    ),
  ) {}

  /** Every admission releases in finally; an abort never submits a cancel order. */
  private async request<T>(
    path: string,
    guard: Guard<T>,
    signal: AbortSignal,
    mutation?: { body: ResearchRequest | Record<string, never>; key?: string },
  ): Promise<T> {
    if (this.active >= MAX_IN_FLIGHT) {
      throw new ApiError(
        "client_busy",
        "Four local requests are already in progress.",
      );
    }
    this.active += 1;
    const deadline = new AbortController();
    const timer = setTimeout(() => {
      deadline.abort();
    }, DEADLINE_MS);
    const combined = AbortSignal.any([signal, deadline.signal]);
    try {
      const headers: Record<string, string> = { Accept: "application/json" };
      let body: string | undefined;
      if (mutation) {
        body = JSON.stringify(mutation.body);
        if (new TextEncoder().encode(body).byteLength > REQUEST_BYTES) {
          throw new ApiError(
            "request_size",
            "Configuration exceeds the request byte limit.",
          );
        }
        headers["Content-Type"] = "application/json";
        headers["X-Signal-Foundry-Client"] = "nexus";
        if (mutation.key !== undefined)
          headers["Idempotency-Key"] = mutation.key;
      }
      const response = await this.transport("/api/v1/" + path, {
        method: mutation ? "POST" : "GET",
        headers,
        ...(body === undefined ? {} : { body }),
        signal: combined,
        credentials: "omit",
        mode: "same-origin",
        redirect: "error",
        cache: "no-store",
        referrerPolicy: "no-referrer",
      });
      const value = await boundedJson(response, combined);
      if (!response.ok) {
        if (!guards.isProblem(value)) {
          throw new ApiError(
            "invalid_response",
            "Server failure violates its error contract.",
          );
        }
        throw new ApiError(value.code, value.detail);
      }
      if (!guard(value))
        throw new ApiError(
          "invalid_response",
          "Response violates the resolved API contract.",
        );
      return value;
    } catch (cause) {
      if (cause instanceof ApiError) throw cause;
      if (signal.aborted)
        throw new ApiError(
          "aborted",
          "Request stopped; any submitted job may continue.",
          { cause },
        );
      if (deadline.signal.aborted)
        throw new ApiError(
          "deadline",
          "Request deadline exceeded; inspect the queue before retrying.",
          { cause },
        );
      throw new ApiError(
        "network",
        "Local service could not be reached or the response was interrupted.",
        { cause },
      );
    } finally {
      clearTimeout(timer);
      deadline.abort();
      this.active -= 1;
    }
  }

  async catalog(signal: AbortSignal): Promise<Catalog> {
    const value = await this.request("catalog", guards.isCatalog, signal);
    if (
      !unique(value.models.map((item) => item.name)) ||
      !unique(value.strategies.map((item) => item.name)) ||
      !unique(value.datasets.map((item) => item.bundle_id))
    ) {
      throw new ApiError(
        "invalid_response",
        "Catalog contains duplicate identities.",
      );
    }
    return value;
  }

  validate(request: ResearchRequest, signal: AbortSignal): Promise<Validation> {
    return this.request("validate", guards.isValidation, signal, {
      body: request,
    });
  }

  async submit(
    request: ResearchRequest,
    key: string,
    signal: AbortSignal,
  ): Promise<Job> {
    if (key.length < 16 || key.length > 128 || /[^A-Za-z0-9_-]/u.test(key))
      throw new ApiError("invalid_key", "Invalid research submission key.");
    const job = await this.request("jobs", guards.isJob, signal, {
      body: request,
      key,
    });
    assertJob(job);
    return job;
  }

  async jobs(signal: AbortSignal): Promise<JobPage> {
    const page = await this.request("jobs", guards.isJobPage, signal);
    page.jobs.forEach(assertJob);
    if (!unique(page.jobs.map((job) => job.job_id)))
      throw new ApiError("invalid_response", "Duplicate job identities.");
    return page;
  }

  async cancel(jobId: string, signal: AbortSignal): Promise<Job> {
    const job = await this.request(
      "jobs/" + identity(jobId) + "/cancel",
      guards.isJob,
      signal,
      { body: {} },
    );
    assertJob(job);
    if (job.job_id !== jobId)
      throw new ApiError(
        "invalid_response",
        "Cancellation returned another job identity.",
      );
    return job;
  }

  async evidence(job: Job, signal: AbortSignal): Promise<Evidence> {
    const evidence = await this.request(
      "jobs/" + identity(job.job_id) + "/evidence",
      guards.isResearchEvidence,
      signal,
    );
    assertEvidence(evidence);
    if (evidence.request_hash !== job.request_hash)
      throw new ApiError(
        "invalid_response",
        "Evidence is not bound to the selected job.",
      );
    return evidence;
  }

  async audit(jobId: string, signal: AbortSignal): Promise<AuditTrail> {
    const trail = await this.request(
      "jobs/" + identity(jobId) + "/audit",
      guards.isAuditTrail,
      signal,
    );
    if (
      trail.job_id !== jobId ||
      // Sequence is global to the durable store, so interleaved jobs leave gaps.
      trail.events.some(
        (event, index) =>
          event.sequence <= (trail.events[index - 1]?.sequence ?? 0),
      )
    )
      throw new ApiError(
        "invalid_response",
        "Audit identity or event ordering is inconsistent.",
      );
    return trail;
  }

  async compare(
    left: Job,
    right: Job,
    signal: AbortSignal,
  ): Promise<Comparison> {
    const pair = await this.request(
      "compare/" + identity(left.job_id) + "/" + identity(right.job_id),
      guards.isComparison,
      signal,
    );
    pair.evidence.forEach(assertEvidence);
    if (
      pair.evidence[0].request_hash !== left.request_hash ||
      pair.evidence[1].request_hash !== right.request_hash
    )
      throw new ApiError(
        "invalid_response",
        "Comparison is not bound to the selected jobs.",
      );
    return pair;
  }
}
