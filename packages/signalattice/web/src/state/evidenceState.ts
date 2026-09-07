/**
 * The console's honest-state model.
 *
 * Every route and every reusable evidence panel resolves to exactly one of
 * these states. The point of enumerating them is that the failure modes are
 * *different* and must stay different:
 *
 * - `EMPTY` means the server answered and there is nothing to show.
 * - `INSUFFICIENT_EVIDENCE` means there is data, but not enough of it to
 *   support the claim the panel exists to make. That is a finding, not an
 *   absence.
 * - `INVALID` means the evidence contradicts itself or failed verification. It
 *   is never a smaller version of `INSUFFICIENT_EVIDENCE`.
 * - `STALE` means the evidence is real but older than its freshness contract.
 * - `UNAVAILABLE` means the service could not answer at all.
 * - `ERROR` means the console could not interpret what it received.
 *
 * Collapsing any of these into "no data" would let a reader conclude "nothing
 * is wrong" from "we could not tell", which is the specific mistake this model
 * exists to prevent.
 *
 * State is derived from validated server fields only. It is never inferred from
 * an HTTP status alone, from a missing value, from a colour, or from the
 * absence of a reported failure.
 */

export const EVIDENCE_STATES = [
  "LOADING",
  "READY",
  "EMPTY",
  "PARTIAL",
  "INSUFFICIENT_EVIDENCE",
  "INVALID",
  "STALE",
  "UNAVAILABLE",
  "ERROR",
] as const;

export type EvidenceState = (typeof EVIDENCE_STATES)[number];

/** States in which a panel shows no trustworthy measurement. */
const NON_MEASURING: ReadonlySet<EvidenceState> = new Set<EvidenceState>([
  "LOADING",
  "EMPTY",
  "INSUFFICIENT_EVIDENCE",
  "INVALID",
  "UNAVAILABLE",
  "ERROR",
]);

/**
 * A resolved panel state with the reason a reader needs.
 *
 * `detail` is always present for a non-`READY` state: a banner that says
 * "insufficient" without saying insufficient *for what* sends the reader back
 * to the raw API, which defeats the console.
 */
export interface EvidenceStatus<T> {
  readonly state: EvidenceState;
  readonly detail: string;
  readonly data: T | null;
  /** True when a measurement may be read from `data`. */
  readonly measuring: boolean;
}

function status<T>(state: EvidenceState, detail: string, data: T | null): EvidenceStatus<T> {
  return { state, detail, data, measuring: !NON_MEASURING.has(state) };
}

export const loading = <T,>(): EvidenceStatus<T> =>
  status<T>("LOADING", "Reading local evidence.", null);

export const ready = <T,>(data: T, detail = "Evidence complete."): EvidenceStatus<T> =>
  status<T>("READY", detail, data);

export const empty = <T,>(detail: string): EvidenceStatus<T> => status<T>("EMPTY", detail, null);

/** Some of the requested evidence is present; the shortfall is named. */
export const partial = <T,>(data: T, detail: string): EvidenceStatus<T> =>
  status<T>("PARTIAL", detail, data);

/**
 * Note the argument order: `detail` comes first for `insufficient` and
 * `invalid` because their data is optional -- the shortfall is often the entire
 * finding and there is nothing to show. The states that always carry data
 * (`ready`, `partial`, `stale`) take the data first.
 */
export const insufficient = <T,>(detail: string, data: T | null = null): EvidenceStatus<T> =>
  status<T>("INSUFFICIENT_EVIDENCE", detail, data);

export const invalid = <T,>(detail: string, data: T | null = null): EvidenceStatus<T> =>
  status<T>("INVALID", detail, data);

export const stale = <T,>(data: T, detail: string): EvidenceStatus<T> =>
  status<T>("STALE", detail, data);

export const unavailable = <T,>(detail: string): EvidenceStatus<T> =>
  status<T>("UNAVAILABLE", detail, null);

export const errored = <T,>(detail: string): EvidenceStatus<T> => status<T>("ERROR", detail, null);

/**
 * Freshness window beyond which evidence is reported `STALE` rather than
 * `READY`. Local evidence does not expire on its own, so the console names an
 * explicit contract instead of implying the newest row is current.
 */
export const FRESHNESS_WINDOW_MS = 24 * 60 * 60 * 1000;

/**
 * Classify a transport failure into the state a reader should see.
 *
 * A 4xx that names a bounded problem is not the same as a service that did not
 * answer, and neither is the same as a body the console could not decode.
 */
export function classifyTransport<T>(
  result: { readonly kind: string; readonly status?: number },
): EvidenceStatus<T> | null {
  switch (result.kind) {
    case "ok":
      return null;
    case "timeout":
      return unavailable<T>(
        "The local service did not answer within 10 seconds. It may be saturated or stopped.",
      );
    case "offline":
      return unavailable<T>("The local service is not reachable at this origin.");
    case "too-large":
      return errored<T>("The response exceeded the console's size ceiling and was not parsed.");
    case "malformed":
      return errored<T>("The response could not be read as JSON evidence.");
    case "problem": {
      const code = result.status ?? 0;
      if (code === 404) return empty<T>("No evidence of this kind is recorded.");
      if (code === 503) return unavailable<T>("The local service reports its storage unavailable.");
      if (code === 429) {
        return unavailable<T>("The local service is rate limiting; retry is a manual action.");
      }
      return errored<T>(`The local service refused the read (status ${String(code)}).`);
    }
    default:
      return errored<T>("The console received an outcome it does not recognise.");
  }
}

/**
 * Report whether an instant is inside the freshness window.
 *
 * @param observedAt Evidence timestamp in epoch milliseconds.
 * @param now Reference instant, injected so tests are not clock dependent.
 */
export function isStale(observedAt: number, now: number): boolean {
  if (!Number.isFinite(observedAt) || !Number.isFinite(now)) return true;
  return now - observedAt > FRESHNESS_WINDOW_MS;
}
