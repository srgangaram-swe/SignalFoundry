/** Pure configure/preflight/submission state machine; effects live outside it. */
import type { Job, ResearchRequest, Validation } from "./types";

export type Workflow =
  | { readonly phase: "editing"; readonly revision: number }
  | {
      readonly phase: "validating";
      readonly revision: number;
      readonly request: ResearchRequest;
    }
  | {
      readonly phase: "validated";
      readonly revision: number;
      readonly request: ResearchRequest;
      readonly validation: Validation;
    }
  | {
      readonly phase: "submitting" | "uncertain";
      readonly revision: number;
      readonly request: ResearchRequest;
      readonly validation: Validation;
      readonly key: string;
    }
  | {
      readonly phase: "submitted";
      readonly revision: number;
      readonly job: Job;
    };

export type WorkflowEvent =
  | { readonly type: "edit" }
  | { readonly type: "validate"; readonly request: ResearchRequest }
  | {
      readonly type: "validated";
      readonly revision: number;
      readonly validation: Validation;
    }
  | { readonly type: "validation_failed"; readonly revision: number }
  | { readonly type: "submit"; readonly key: string }
  | { readonly type: "submission_unknown" }
  | { readonly type: "retry_submission" }
  | { readonly type: "submitted"; readonly job: Job };

export const INITIAL: Workflow = { phase: "editing", revision: 0 };

/** Invalid or stale transitions are inert; ambiguity never creates a fresh key. */
export function transition(state: Workflow, event: WorkflowEvent): Workflow {
  switch (event.type) {
    case "edit":
      return ["submitting", "uncertain"].includes(state.phase)
        ? state
        : { phase: "editing", revision: state.revision + 1 };
    case "validate":
      return state.phase === "editing"
        ? {
            phase: "validating",
            revision: state.revision,
            request: event.request,
          }
        : state;
    case "validated":
      return state.phase === "validating" && state.revision === event.revision
        ? { ...state, phase: "validated", validation: event.validation }
        : state;
    case "validation_failed":
      return state.phase === "validating" && state.revision === event.revision
        ? { phase: "editing", revision: state.revision }
        : state;
    case "submit":
      return state.phase === "validated"
        ? { ...state, phase: "submitting", key: event.key }
        : state;
    case "submission_unknown":
      return state.phase === "submitting"
        ? { ...state, phase: "uncertain" }
        : state;
    case "retry_submission":
      return state.phase === "uncertain"
        ? { ...state, phase: "submitting" }
        : state;
    case "submitted":
      return state.phase === "submitting" &&
        state.validation.request_hash === event.job.request_hash
        ? { phase: "submitted", revision: state.revision, job: event.job }
        : state;
  }
}
