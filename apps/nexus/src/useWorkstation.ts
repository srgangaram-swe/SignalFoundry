import { useEffect, useMemo, useReducer, useRef, useState } from "react";
import { type ResearchClient, ApiError } from "./api";
import { message, parseConfiguration } from "./configuration";
import { INITIAL, transition } from "./workflow";
import type { AuditTrail, Catalog, Comparison, Evidence, Job } from "./types";

const POLL_MS = 2000;
const MAX_POLLS = 120;

/** Own bounded effect lifetimes and workflow transitions, separate from views. */
export function useWorkstation(client: ResearchClient) {
  const [catalog, setCatalog] = useState<Catalog | null>(null);
  const [text, setText] = useState("");
  const [workflow, dispatch] = useReducer(transition, INITIAL);
  const [jobs, setJobs] = useState<readonly Job[]>([]);
  const [error, setError] = useState("");
  const [busy, setBusy] = useState(false);
  const [evidence, setEvidence] = useState<Evidence | null>(null);
  const [audit, setAudit] = useState<AuditTrail | null>(null);
  const [comparison, setComparison] = useState<Comparison | null>(null);
  const [selected, setSelected] = useState<readonly string[]>([]);
  const [light, setLight] = useState(false);
  const [monitor, setMonitor] = useState(0);
  const lifetime = useRef(new AbortController());

  useEffect(() => {
    const controller = new AbortController();
    lifetime.current = controller;
    void Promise.all([
      client.catalog(controller.signal),
      client.jobs(controller.signal),
    ])
      .then(([catalogValue, page]) => {
        if (controller.signal.aborted) return;
        setCatalog(catalogValue);
        setText(JSON.stringify(catalogValue.default_request, null, 2));
        setJobs(page.jobs);
      })
      .catch((cause: unknown) => {
        if (!controller.signal.aborted) setError(message(cause));
      });
    return () => {
      controller.abort();
    };
  }, [client]);

  const pending = jobs.some(
    (job) => job.state === "queued" || job.state === "running",
  );
  useEffect(() => {
    if (!pending) return;
    const controller = new AbortController();
    let count = 0;
    let timer: ReturnType<typeof setTimeout>;
    const poll = async () => {
      try {
        const page = await client.jobs(controller.signal);
        if (controller.signal.aborted) return;
        setJobs(page.jobs);
        count += 1;
        if (count >= MAX_POLLS) {
          setError(
            "Automatic monitoring stopped after 120 checks. Refresh jobs to resume; research may still be running.",
          );
          return;
        }
        timer = setTimeout(() => {
          void poll();
        }, POLL_MS);
      } catch (cause) {
        if (!controller.signal.aborted) setError(message(cause));
      }
    };
    timer = setTimeout(() => {
      void poll();
    }, POLL_MS);
    return () => {
      controller.abort();
      clearTimeout(timer);
    };
  }, [client, pending, monitor]);

  async function operation(action: (signal: AbortSignal) => Promise<void>) {
    setBusy(true);
    setError("");
    try {
      await action(lifetime.current.signal);
    } catch (cause) {
      if (!lifetime.current.signal.aborted) setError(message(cause));
    } finally {
      if (!lifetime.current.signal.aborted) setBusy(false);
    }
  }

  function edit(value: string) {
    setText(value);
    dispatch({ type: "edit" });
  }
  function choose(field: "model" | "strategy" | "data", value: string) {
    try {
      const request = parseConfiguration(text);
      const changed =
        field === "model"
          ? { ...request, model: { name: value, parameters: [] } }
          : field === "strategy"
            ? { ...request, strategy: value }
            : {
                ...request,
                data: {
                  ...request.data,
                  kind: value === "synthetic" ? "synthetic" : "bundle",
                  bundle_id: value === "synthetic" ? null : value,
                },
              };
      edit(JSON.stringify(changed, null, 2));
    } catch (cause) {
      setError(message(cause));
    }
  }
  async function validate(signal: AbortSignal) {
    const request = parseConfiguration(text);
    const revision = workflow.revision;
    dispatch({ type: "validate", request });
    try {
      const validation = await client.validate(request, signal);
      if (!signal.aborted)
        dispatch({ type: "validated", revision, validation });
    } catch (cause) {
      dispatch({ type: "validation_failed", revision });
      throw cause;
    }
  }
  async function submit(signal: AbortSignal) {
    if (workflow.phase !== "validated" && workflow.phase !== "uncertain")
      return;
    const key =
      workflow.phase === "uncertain" ? workflow.key : crypto.randomUUID();
    dispatch(
      workflow.phase === "uncertain"
        ? { type: "retry_submission" }
        : { type: "submit", key },
    );
    try {
      const job = await client.submit(workflow.request, key, signal);
      if (job.request_hash !== workflow.validation.request_hash)
        throw new ApiError(
          "invalid_response",
          "Submission identity differs from preflight. Retry the same request.",
        );
      if (!signal.aborted) {
        dispatch({ type: "submitted", job });
        setJobs((previous) =>
          [job, ...previous.filter((item) => item.job_id !== job.job_id)].slice(
            0,
            64,
          ),
        );
      }
    } catch (cause) {
      dispatch({ type: "submission_unknown" });
      throw cause;
    }
  }
  const locked =
    busy || workflow.phase === "uncertain" || workflow.phase === "submitting";
  const draft = useMemo(() => {
    try {
      return parseConfiguration(text);
    } catch {
      return null;
    }
  }, [text]);
  return {
    catalog,
    text,
    workflow,
    dispatch,
    jobs,
    setJobs,
    error,
    busy,
    evidence,
    setEvidence,
    audit,
    setAudit,
    comparison,
    setComparison,
    selected,
    setSelected,
    light,
    setLight,
    setMonitor,
    operation,
    edit,
    choose,
    validate,
    submit,
    locked,
    draft,
  };
}
