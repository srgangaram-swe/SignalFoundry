// Generated from contracts/openapi-v1.json; do not hand-edit.
import type { components } from "../../../../contracts/research-v1";
import type { Resolved } from "../types";
export declare function isCatalog(value: unknown): value is Resolved<components["schemas"]["Catalog"]>;
export declare function isResearchRequest(value: unknown): value is Resolved<components["schemas"]["ResearchRequest"]>;
export declare function isValidation(value: unknown): value is Resolved<components["schemas"]["Validation"]>;
export declare function isJob(value: unknown): value is Resolved<components["schemas"]["Job"]>;
export declare function isJobPage(value: unknown): value is Resolved<components["schemas"]["JobPage"]>;
export declare function isResearchEvidence(value: unknown): value is Resolved<components["schemas"]["ResearchEvidence"]>;
export declare function isComparison(value: unknown): value is Resolved<components["schemas"]["Comparison"]>;
export declare function isAuditTrail(value: unknown): value is Resolved<components["schemas"]["AuditTrail"]>;
export declare function isProblem(value: unknown): value is Resolved<components["schemas"]["Problem"]>;
