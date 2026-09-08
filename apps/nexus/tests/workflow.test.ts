import { describe, expect, it } from "vitest";
import { INITIAL, transition } from "../src/workflow";
import type { Workflow, WorkflowEvent } from "../src/workflow";
import { job, request, validation } from "./fixtures";

function validated(): Workflow {
  return transition(transition(INITIAL, { type: "validate", request }), {
    type: "validated",
    revision: 0,
    validation,
  });
}

describe("workflow invariants", () => {
  it("requires preflight and invalidates it on every edit", () => {
    expect(transition(INITIAL, { type: "submit", key: "x" })).toBe(INITIAL);
    expect(validated().phase).toBe("validated");
    expect(transition(validated(), { type: "edit" })).toEqual({
      phase: "editing",
      revision: 1,
    });
    const waiting = transition(INITIAL, { type: "validate", request });
    const changed = transition(waiting, { type: "edit" });
    expect(
      transition(changed, { type: "validated", revision: 0, validation }),
    ).toBe(changed);
    expect(
      transition(waiting, { type: "validation_failed", revision: 0 }),
    ).toEqual(INITIAL);
    expect(
      transition(waiting, { type: "validation_failed", revision: 1 }),
    ).toBe(waiting);
  });

  it("holds one key/request through ambiguity and accepts only the matching job", () => {
    const submitting = transition(validated(), {
      type: "submit",
      key: "fixed_submission_key",
    });
    const uncertain = transition(submitting, { type: "submission_unknown" });
    expect(uncertain.phase).toBe("uncertain");
    expect(transition(uncertain, { type: "edit" })).toBe(uncertain);
    expect(transition(uncertain, { type: "submit", key: "another_key" })).toBe(
      uncertain,
    );
    const retry = transition(uncertain, { type: "retry_submission" });
    expect(retry).toEqual(submitting);
    expect(
      transition(retry, {
        type: "submitted",
        job: { ...job, request_hash: "b".repeat(64) },
      }),
    ).toBe(retry);
    const done = transition(retry, { type: "submitted", job });
    expect(done.phase).toBe("submitted");
    expect(transition(done, { type: "edit" }).phase).toBe("editing");
  });

  it("ignores every inapplicable action without hidden effects", () => {
    const invalid: WorkflowEvent[] = [
      { type: "validated", revision: 0, validation },
      { type: "validation_failed", revision: 0 },
      { type: "submission_unknown" },
      { type: "retry_submission" },
      { type: "submitted", job },
    ];
    for (const event of invalid)
      expect(transition(INITIAL, event)).toBe(INITIAL);
    expect(transition(validated(), { type: "validate", request }).phase).toBe(
      "validated",
    );
  });
});
