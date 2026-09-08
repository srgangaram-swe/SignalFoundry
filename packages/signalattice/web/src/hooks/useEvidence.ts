/**
 * One bounded read, decoded, and resolved to an honest state.
 *
 * Deliberate non-features:
 *
 * - **No automatic retry.** A failed read stays failed until a person asks
 *   again. Retrying a saturated local service automatically is how a read-only
 *   console becomes the load that keeps it saturated.
 * - **No polling by default.** `refreshToken` changes only when a user acts.
 * - **No optimistic default.** The initial state is `LOADING`, never an empty
 *   success, so a render that beats its data cannot look like "no findings".
 *
 * A skipped read is computed during render rather than written from an effect:
 * a state that is a pure function of the props does not need a round trip
 * through the scheduler, and setting it in an effect causes a cascading render.
 *
 * The request is aborted when the component unmounts or the inputs change, so a
 * late response can never overwrite a newer one.
 */
import { useEffect, useRef, useState } from "react";
import type { z } from "zod";

import { readJson } from "../api/client";
import { decode } from "../api/decoders";
import type { EvidenceStatus } from "../state/evidenceState";
import { classifyTransport, empty, errored, loading, ready } from "../state/evidenceState";

export interface UseEvidenceOptions<T, U> {
  /** Root-relative API path; validated by the transport. */
  readonly path: string;
  readonly schema: z.ZodType<T>;
  /**
   * Maps decoded evidence to a final state. This is where a view says
   * "this is insufficient" or "this is stale" using validated server fields;
   * omitting it means a successful decode is simply `READY`.
   */
  readonly interpret?: (value: T) => EvidenceStatus<U>;
  /** Increment to re-read. Never changed by this hook itself. */
  readonly refreshToken?: number;
  /** Skip the read entirely, e.g. while a required parameter is absent. */
  readonly skip?: boolean;
  readonly skipDetail?: string;
}

export function useEvidence<T, U = T>(options: UseEvidenceOptions<T, U>): EvidenceStatus<U> {
  const { path, schema, interpret, refreshToken = 0, skip = false, skipDetail } = options;
  // The request this status belongs to. Keeping it beside the status is what
  // lets a changed path read as LOADING during render, rather than needing a
  // synchronous setState in the effect to clear a stale result.
  const key = `${path}|${String(refreshToken)}`;
  const [entry, setEntry] = useState<{ key: string; status: EvidenceStatus<U> }>(() => ({
    key: "",
    status: loading<U>(),
  }));
  // Tracks whether this effect instance is still the current one. A ref rather
  // than a closure flag so the type checker can see it genuinely changes.
  const generation = useRef(0);

  useEffect(() => {
    if (skip) return undefined;

    generation.current += 1;
    const mine = generation.current;
    const controller = new AbortController();
    const isCurrent = (): boolean => generation.current === mine;
    const setStatus = (status: EvidenceStatus<U>): void => {
      setEntry({ key, status });
    };

    void (async () => {
      let result;
      try {
        result = await readJson(path, { signal: controller.signal });
      } catch (error: unknown) {
        // A refused path is a programming error, but a panel stuck on LOADING
        // for ever is worse than a visible one: the reader would wait on
        // evidence that is never coming and could not tell why.
        if (!isCurrent()) return;
        const reason = error instanceof Error ? error.message : "unknown transport fault";
        setStatus(errored<U>(`The console could not issue this read (${reason}).`));
        return;
      }
      if (!isCurrent()) return;

      const transportState = classifyTransport<U>(result);
      if (transportState !== null) {
        setStatus(transportState);
        return;
      }

      const body = "body" in result ? result.body : undefined;
      const decoded = decode(schema, body);
      if (!decoded.ok) {
        // An unknown field or unrecognised enum lands here rather than being
        // coerced: incompatible evidence is a finding, not a rendering detail.
        setStatus(
          errored<U>(`The response did not match the version-1 contract (${decoded.reason}).`),
        );
        return;
      }

      setStatus(
        interpret === undefined ? ready<U>(decoded.value as unknown as U) : interpret(decoded.value),
      );
    })();

    return () => {
      // Invalidate this instance so a late response cannot apply.
      generation.current += 1;
      controller.abort();
    };
  }, [key, path, schema, interpret, skip]);

  if (skip) {
    return empty<U>(skipDetail ?? "Select a subject to load its evidence.");
  }
  // A result recorded for a different request is not this request's result.
  return entry.key === key ? entry.status : loading<U>();
}
