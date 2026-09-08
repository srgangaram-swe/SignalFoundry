/** Fully serialized server records; API defaults are not invented by the UI. */
import type { components } from "../../../contracts/research-v1";

export type Resolved<T> = T extends object
  ? { readonly [K in keyof T]-?: Resolved<Exclude<T[K], undefined>> }
  : T;

type Schema<K extends keyof components["schemas"]> = Resolved<
  components["schemas"][K]
>;
export type Catalog = Schema<"Catalog">;
export type ResearchRequest = Schema<"ResearchRequest">;
export type Validation = Schema<"Validation">;
export type Job = Schema<"Job">;
export type JobPage = Schema<"JobPage">;
export type Evidence = Schema<"ResearchEvidence">;
export type EvidenceTable = Schema<"EvidenceTable">;
export type Comparison = Schema<"Comparison">;
export type AuditTrail = Schema<"AuditTrail">;
export type Problem = Schema<"Problem">;
export type Capability = Schema<"Capability">;
