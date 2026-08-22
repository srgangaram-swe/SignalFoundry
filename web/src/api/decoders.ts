/**
 * Independent strict runtime decoding for every response the console reads.
 *
 * The generated OpenAPI types in `schema.ts` describe what the contract
 * *promises*. They are erased at runtime and therefore prove nothing about the
 * bytes that actually arrive. These decoders are the check that runs on the
 * real payload, and they are deliberately written by hand rather than derived
 * from the same document: a decoder generated from the contract would agree
 * with the contract even when the server does not.
 *
 * Everything fails closed. Unknown fields, non-finite numbers, malformed
 * digests, invalid timestamps, oversized collections, wrong schema versions,
 * and unrecognised enum members are rejected rather than coerced, because each
 * of those is a way for a console to render a number nobody produced.
 */
import { z } from "zod";

/** Contract version the console understands. A mismatch is incompatible evidence. */
export const SUPPORTED_SCHEMA_VERSION = 1;

/** Collection ceilings; the DOM is bounded before it is built. */
export const MAX_ITEMS_PER_PAGE = 100;
export const MAX_MARKS_PER_PANEL = 256;
export const MAX_TEXT_CHARS = 4096;

const digest = z.string().regex(/^[0-9a-f]{64}$/u, "expected a lowercase SHA-256 digest");

/**
 * A JSON number the console may plot or print.
 *
 * `z.number()` alone accepts NaN and infinities, both of which render as
 * "NaN"/"Infinity" in a table and silently poison any aggregate computed from
 * them, so finiteness is required explicitly.
 */
const finite = z.number().refine(Number.isFinite, { message: "must be finite" });

const boundedText = z.string().max(MAX_TEXT_CHARS);

/** An ISO-8601 instant that actually parses; a string that does not is invalid. */
const instant = z.string().refine((value) => Number.isFinite(Date.parse(value)), {
  message: "must be an ISO-8601 instant",
});

const schemaVersion = z.literal(SUPPORTED_SCHEMA_VERSION);

/** `strictObject` refuses unknown keys, so added server fields surface loudly. */
export const gateResultSchema = z.strictObject({
  schema_version: schemaVersion,
  name: z.string().min(1).max(64),
  satisfied: z.boolean(),
  detail: boundedText,
});

export const hypothesisTestSchema = z.strictObject({
  schema_version: schemaVersion,
  name: z.string().min(1).max(64),
  metric: z.string().min(1).max(64),
  verdict: z.enum([
    "favours_challenger",
    "favours_champion",
    "inconclusive",
    "underpowered",
    "unknown",
  ]),
  point_estimate: finite.nullable(),
  // Independently nullable: a one-sided test has a genuinely unbounded end, and
  // substituting a number there would assert a bound nobody established.
  interval_low: finite.nullable(),
  interval_high: finite.nullable(),
  p_value_uncorrected: finite.min(0).max(1).nullable(),
  blocks: z.number().int().min(0),
  observations: z.number().int().min(0),
  margin: finite.nullable(),
});

export const comparisonSchema = z.strictObject({
  schema_version: schemaVersion,
  sequence: z.number().int().min(1),
  recorded_at: instant,
  recommendation: z.enum([
    "promote",
    "retain_champion",
    "insufficient_evidence",
    "invalid",
    "unknown",
  ]),
  policy_identity: digest,
  cohort_identity: digest,
  decided_at: boundedText,
  gates: z.array(gateResultSchema).max(MAX_MARKS_PER_PANEL),
  tests: z.array(hypothesisTestSchema).max(MAX_MARKS_PER_PANEL),
  correction_method: z.string().max(64).nullable(),
  correction_alpha: finite.gt(0).lt(0.5).nullable(),
  family_size: z.number().int().min(1).nullable(),
  truncated_gates: z.boolean(),
  truncated_tests: z.boolean(),
});

export const laneEventSchema = z.strictObject({
  schema_version: schemaVersion,
  sequence: z.number().int().min(1),
  kind: z.enum([
    "policy",
    "comparison",
    "request",
    "approval",
    "assignment",
    "monitoring",
    "freeze",
  ]),
  recorded_at: instant,
  chain_digest: digest,
  summary: boundedText,
});

export const laneSummarySchema = z.strictObject({
  schema_version: schemaVersion,
  lane_identity: digest,
  purpose: z.string().min(1).max(64),
  target: z.string().min(1).max(64),
  horizon_days: z.number().int().min(1).max(365),
  frequency: z.string().min(1).max(64),
  universe: z.string().min(1).max(64),
  decision_policy: z.string().min(1).max(64),
  environment: z.string().min(1).max(64),
  state: z.enum(["unassigned", "active", "frozen"]),
  champion_revision: digest.nullable(),
  generation: z.number().int().min(0),
  freeze_trigger: z.enum(["hard_integrity", "consecutive_soft_breach"]).nullable(),
  created_at: instant,
  event_count: z.number().int().min(0),
  chain_verified: z.boolean(),
  chain_fault: boundedText.nullable(),
  events_by_kind: z.array(z.tuple([z.string().max(64), z.number().int().min(0)])).max(32),
});

export const lanePageSchema = z.strictObject({
  schema_version: schemaVersion,
  items: z.array(laneSummarySchema).max(MAX_ITEMS_PER_PAGE),
  next_cursor: digest.nullable(),
});

export const laneDetailSchema = z.strictObject({
  schema_version: schemaVersion,
  lane: laneSummarySchema,
  events: z.array(laneEventSchema).max(MAX_MARKS_PER_PANEL),
  truncated: z.boolean(),
  authority: boundedText,
});

export const comparisonPageSchema = z.strictObject({
  schema_version: schemaVersion,
  items: z.array(comparisonSchema).max(MAX_ITEMS_PER_PAGE),
  truncated: z.boolean(),
});

export const liveSchema = z.looseObject({ status: z.string().max(32) });

export const readySchema = z.looseObject({
  status: z.string().max(32),
  readiness: z.string().max(64).optional(),
});

export const problemSchema = z.looseObject({
  code: z.string().max(64),
  title: z.string().max(200),
  detail: boundedText,
  status: z.number().int().min(100).max(599),
});

export type GateResult = z.infer<typeof gateResultSchema>;
export type HypothesisTest = z.infer<typeof hypothesisTestSchema>;
export type Comparison = z.infer<typeof comparisonSchema>;
export type LaneEvent = z.infer<typeof laneEventSchema>;
export type LaneSummary = z.infer<typeof laneSummarySchema>;
export type LanePage = z.infer<typeof lanePageSchema>;
export type LaneDetail = z.infer<typeof laneDetailSchema>;
export type ComparisonPage = z.infer<typeof comparisonPageSchema>;
export type Problem = z.infer<typeof problemSchema>;

/** Outcome of decoding, with the reason available for the reader. */
export type DecodeResult<T> =
  | { readonly ok: true; readonly value: T }
  | { readonly ok: false; readonly reason: string };

/**
 * Decode one payload, returning a reason instead of throwing.
 *
 * The reason is truncated and carries only the failing field path and message.
 * A raw Zod dump can echo the received value, and echoing an unvalidated server
 * string into the DOM is the thing this whole module exists to prevent.
 */
export function decode<T>(schema: z.ZodType<T>, payload: unknown): DecodeResult<T> {
  const parsed = schema.safeParse(payload);
  if (parsed.success) {
    return { ok: true, value: parsed.data };
  }
  const first = parsed.error.issues[0];
  const path = first?.path.join(".") ?? "(root)";
  const message = first?.message ?? "did not match the expected shape";
  return { ok: false, reason: `${path}: ${message}`.slice(0, 200) };
}
