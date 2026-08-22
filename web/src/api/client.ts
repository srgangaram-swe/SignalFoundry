/**
 * The console's only path to the network.
 *
 * Every property here is a refusal rather than a convenience, because a browser
 * reading governance evidence is exactly where an accidental capability becomes
 * a real one.
 *
 * - **GET only.** There is no method parameter. A mutation cannot be expressed,
 *   so it cannot be sent by mistake or added by a later edit that "just needs a
 *   POST".
 * - **Same-origin only.** Paths must be root-relative and are rejected if they
 *   carry a scheme, an authority, or a protocol-relative prefix. The console is
 *   served by the local service and talks to nothing else.
 * - **No credentials.** `credentials: "omit"` is explicit: the local authority
 *   boundary is the loopback socket, not an ambient cookie.
 * - **Bounded concurrency and time.** At most `MAX_IN_FLIGHT` requests, each
 *   with a hard `REQUEST_DEADLINE_MS` abort. There is no automatic retry: a
 *   failed read is surfaced so a human decides, because silently retrying a
 *   saturated local service is how a console becomes the load.
 * - **Bounded response size.** A response larger than `MAX_RESPONSE_BYTES` is
 *   refused before parsing rather than expanded into the DOM.
 */

/** Maximum simultaneous reads. The service reserves capacity for probes. */
export const MAX_IN_FLIGHT = 4;

/** Hard abort deadline for one read. */
export const REQUEST_DEADLINE_MS = 10_000;

/** Refused before parsing; the service's own ceiling is 2 MiB. */
export const MAX_RESPONSE_BYTES = 2 * 1024 * 1024;

/** Server default and ceiling, mirrored so the console cannot ask for more. */
export const DEFAULT_PAGE_SIZE = 25;
export const MAX_PAGE_SIZE = 100;

/** The path portion that is safe to send to the same origin. */
const SAFE_PATH = /^\/(?:api\/v1|health|internal)\/[A-Za-z0-9/_.~-]*$/;

/**
 * The query portion, validated separately from the path.
 *
 * Percent-encoding is permitted because `buildQuery` produces it; the character
 * set deliberately excludes anything that could introduce a second path or a
 * fragment.
 */
const SAFE_QUERY = /^[A-Za-z0-9_.~=&%+-]*$/;

/** Total request-target ceiling, matching the service's own path/query bounds. */
const MAX_TARGET_CHARS = 1024;

/** Discriminated transport outcome. Nothing throws across this boundary. */
export type TransportResult =
  | { readonly kind: "ok"; readonly status: number; readonly body: unknown }
  | { readonly kind: "ok-text"; readonly status: number; readonly text: string }
  | { readonly kind: "problem"; readonly status: number; readonly body: unknown }
  | { readonly kind: "timeout" }
  | { readonly kind: "offline" }
  | { readonly kind: "too-large"; readonly bytes: number }
  | { readonly kind: "malformed"; readonly reason: string };

export class TransportPathError extends Error {
  public constructor(path: string) {
    super(`refusing a request path the console may not send: ${path}`);
    this.name = "TransportPathError";
  }
}

/**
 * Reject anything that is not a root-relative same-origin API path.
 *
 * Checked with an allowlist rather than by blocking known-bad prefixes: a
 * denylist has to anticipate every spelling of "somewhere else", and this one
 * only has to describe the handful of paths the console actually reads.
 */
export function assertSafePath(path: string): string {
  if (typeof path !== "string" || path.length === 0 || path.length > MAX_TARGET_CHARS) {
    throw new TransportPathError(path);
  }
  // Path and query are validated separately: they permit different characters,
  // and checking the whole target with one pattern would have to admit `=` and
  // `&` inside the path as well.
  const separator = path.indexOf("?");
  const route = separator === -1 ? path : path.slice(0, separator);
  const query = separator === -1 ? "" : path.slice(separator + 1);
  if (!SAFE_PATH.test(route) || !SAFE_QUERY.test(query)) {
    throw new TransportPathError(path);
  }
  if (route.includes("..") || route.includes("//") || path.includes("#")) {
    throw new TransportPathError(path);
  }
  return path;
}

/** Bounded query building; refuses values the server would reject anyway. */
export function buildQuery(params: Readonly<Record<string, string | number | undefined>>): string {
  const search = new URLSearchParams();
  for (const [key, value] of Object.entries(params)) {
    if (value === undefined) continue;
    if (typeof value === "number") {
      if (!Number.isSafeInteger(value)) {
        throw new TransportPathError(`${key} must be a safe integer`);
      }
      search.set(key, String(value));
      continue;
    }
    search.set(key, value);
  }
  const encoded = search.toString();
  return encoded ? `?${encoded}` : "";
}

/** Slots currently held; the module owns this because the limit is global. */
let inFlight = 0;

/** Exposed for tests and for the operations view's own honesty. */
export function inFlightCount(): number {
  return inFlight;
}

type FetchLike = typeof globalThis.fetch;

/**
 * Perform one bounded, same-origin GET.
 *
 * @param path Root-relative API path, validated against the allowlist.
 * @param options.signal Caller cancellation, combined with the deadline.
 * @param options.fetchImpl Injected for tests; defaults to the global.
 * @returns A discriminated result. This function does not throw for network,
 *   timeout, size, or protocol failures -- each is a state the console renders.
 * @throws TransportPathError if the path is not one the console may request.
 *   That is a programming error, not a runtime condition, so it is loud.
 */
export async function readJson(
  path: string,
  options: { readonly signal?: AbortSignal; readonly fetchImpl?: FetchLike } = {},
): Promise<TransportResult> {
  const raw = await readRaw(path, "application/json", options);
  if (raw.kind !== "ok-text") return raw;
  let body: unknown;
  try {
    body = JSON.parse(raw.text);
  } catch {
    return { kind: "malformed", reason: "response body is not JSON" };
  }
  if (raw.status >= 400) {
    return { kind: "problem", status: raw.status, body };
  }
  return { kind: "ok", status: raw.status, body };
}

/**
 * Read a bounded text response under exactly the same guards as
 * :func:`readJson`.
 *
 * The metrics endpoint serves Prometheus text exposition, not JSON. Forcing it
 * through the JSON reader would mean re-encoding text as a JSON string and
 * parsing the result, which corrupts the very content the operations view is
 * trying to display.
 */
export async function readText(
  path: string,
  options: { readonly signal?: AbortSignal; readonly fetchImpl?: FetchLike } = {},
): Promise<TransportResult> {
  const raw = await readRaw(path, "text/plain", options);
  if (raw.kind === "ok-text" && raw.status >= 400) {
    return { kind: "problem", status: raw.status, body: null };
  }
  return raw;
}

async function readRaw(
  path: string,
  accept: string,
  options: { readonly signal?: AbortSignal; readonly fetchImpl?: FetchLike },
): Promise<TransportResult> {
  const safePath = assertSafePath(path);
  if (inFlight >= MAX_IN_FLIGHT) {
    return { kind: "malformed", reason: "console concurrency limit reached" };
  }

  const deadline = new AbortController();
  const timer = setTimeout(() => {
    deadline.abort();
  }, REQUEST_DEADLINE_MS);
  const signals: AbortSignal[] = [deadline.signal];
  if (options.signal) signals.push(options.signal);
  const combined = AbortSignal.any(signals);
  const request = options.fetchImpl ?? globalThis.fetch.bind(globalThis);

  inFlight += 1;
  try {
    const response = await request(safePath, {
      method: "GET",
      credentials: "omit",
      cache: "no-store",
      redirect: "error",
      referrerPolicy: "no-referrer",
      headers: { accept },
      signal: combined,
    });

    const declared = response.headers.get("content-length");
    if (declared !== null && Number(declared) > MAX_RESPONSE_BYTES) {
      return { kind: "too-large", bytes: Number(declared) };
    }

    const text = await response.text();
    // Re-checked after reading: a chunked response has no declared length, so
    // trusting the header alone would leave the ceiling unenforced.
    if (text.length > MAX_RESPONSE_BYTES) {
      return { kind: "too-large", bytes: text.length };
    }

    return { kind: "ok-text", status: response.status, text };
  } catch (error: unknown) {
    if (deadline.signal.aborted) return { kind: "timeout" };
    if (options.signal?.aborted) return { kind: "offline" };
    if (error instanceof TypeError) return { kind: "offline" };
    return { kind: "malformed", reason: "transport failure" };
  } finally {
    clearTimeout(timer);
    inFlight -= 1;
  }
}
