/** Redistributable synthetic reference only; actual-worker E2E is separate. */
import report from "../../../docs/evidence/control-plane/measurements.json" with { type: "json" };
import { isResearchEvidence } from "../src/generated/validators.cjs";
import type { Catalog, Job, Validation } from "../src/types";

const candidate: unknown = report.research;
if (!isResearchEvidence(candidate))
  throw new Error("Synthetic reference violates the resolved contract");
export const evidence = candidate;
export const request = evidence.request;
export const validation: Validation = {
  schema_version: "1.0.0",
  request_hash: evidence.request_hash,
  data_identity: evidence.data_identity,
  observations: 4500,
  symbols: 9,
  sessions: 500,
  limitations: evidence.limitations,
};
export const job: Job = {
  schema_version: "1.0.0",
  job_id: "a".repeat(64),
  request_hash: evidence.request_hash,
  state: "succeeded",
  created_at: "2026-09-01T00:00:00Z",
  updated_at: "2026-09-01T00:00:10Z",
  error_code: null,
  evidence_hash: report.repeatable_research_hash,
};
export const catalog: Catalog = {
  schema_version: "1.0.0",
  default_request: request,
  models: [
    {
      name: "ridge",
      available: true,
      parameters: ["alpha"],
      reason: "Synthetic test fixture registry.",
    },
    {
      name: "random_forest",
      available: true,
      parameters: ["n_estimators", "max_depth"],
      reason: "Synthetic test fixture registry.",
    },
    {
      name: "torch_mlp",
      available: false,
      parameters: [],
      reason: "Optional test backend unavailable.",
    },
  ],
  strategies: [
    "long_short",
    "long_only_topk",
    "rank_weighted",
    "confidence_weighted",
    "threshold",
  ].map((name) => ({
    name,
    available: true,
    parameters: [],
    reason: "Synthetic fixture capability.",
  })),
  baselines: ["historical_mean", "momentum_baseline", "zero"],
  datasets: [],
  limitations: ["Synthetic development fixture; not a trading qualification."],
  mode: "development_simulation",
  live_readiness: "NOT_READY",
};

export function json(value: unknown, status = 200): Response {
  return new Response(JSON.stringify(value), {
    status,
    headers: { "Content-Type": "application/json" },
  });
}
