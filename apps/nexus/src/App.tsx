import { ResearchClient } from "./api";
import { EvidenceView } from "./EvidenceView";
import { ConfigurationFields } from "./ConfigurationFields";
import { useWorkstation } from "./useWorkstation";

const DEFAULT_CLIENT = new ResearchClient();

/** Accessible workstation presentation; workflow ownership lives in its hook. */
export function App({
  client = DEFAULT_CLIENT,
}: {
  readonly client?: ResearchClient;
}) {
  const {
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
  } = useWorkstation(client);
  return (
    <div className={light ? "app light" : "app"}>
      <a className="skip" href="#workspace">
        Skip to workspace
      </a>
      <header>
        <a className="brand" href="/nexus">
          SIGNAL FOUNDRY <strong>NEXUS</strong>
        </a>
        <span className="badge">Research workstation</span>
        <button
          onClick={() => {
            setLight(!light);
          }}
        >
          {light ? "Dark theme" : "Light theme"}
        </button>
      </header>
      <main id="workspace">
        <div className="hero">
          <div>
            <p className="eyebrow">LOCAL RESEARCH / EVIDENCE FIRST</p>
            <h1>
              Test the thesis.
              <br />
              Inspect the evidence.
            </h1>
            <p>Configure → validate → run → compare → inspect</p>
          </div>
          <aside>
            <strong>NOT_READY</strong>
            <p>Development simulation</p>
            <p>
              No live execution. A completed run does not establish a trading
              edge.
            </p>
          </aside>
        </div>
        {error && (
          <div role="alert" className="error">
            {error}
          </div>
        )}
        {!catalog && !error && <p role="status">Loading local capabilities…</p>}
        {catalog && (
          <div className="workspace-grid">
            <section className="panel">
              <h2>01 / Configure research</h2>
              <p>
                Defaults and available capabilities come from the local service.
                Edit the complete configuration for parameters, folds, costs,
                risk and baselines.
              </p>
              <fieldset disabled={locked}>
                <legend>Capabilities</legend>
                <label>
                  Model
                  <select
                    value={draft?.model.name ?? ""}
                    onChange={(event) => {
                      choose("model", event.target.value);
                    }}
                  >
                    {catalog.models.map((item) => (
                      <option
                        key={item.name}
                        value={item.name}
                        disabled={!item.available}
                      >
                        {item.name}
                        {item.available ? "" : " (unavailable)"}
                      </option>
                    ))}
                  </select>
                </label>
                <label>
                  Strategy
                  <select
                    value={draft?.strategy ?? ""}
                    onChange={(event) => {
                      choose("strategy", event.target.value);
                    }}
                  >
                    {catalog.strategies.map((item) => (
                      <option key={item.name} disabled={!item.available}>
                        {item.name}
                      </option>
                    ))}
                  </select>
                </label>
                <label>
                  Dataset
                  <select
                    value={draft?.data.bundle_id ?? "synthetic"}
                    onChange={(event) => {
                      choose("data", event.target.value);
                    }}
                  >
                    <option value="synthetic">
                      Synthetic engineering panel
                    </option>
                    {catalog.datasets.map((item) => (
                      <option key={item.bundle_id} value={item.bundle_id}>
                        {item.bundle_id.slice(0, 12)} · {item.rows} rows
                      </option>
                    ))}
                  </select>
                </label>
                {draft && <ConfigurationFields request={draft} change={edit} />}
                <label>
                  Complete research configuration
                  <textarea
                    spellCheck={false}
                    value={text}
                    onChange={(event) => {
                      edit(event.target.value);
                    }}
                  />
                </label>
              </fieldset>
              <div className="actions">
                <button
                  disabled={locked || workflow.phase !== "editing"}
                  onClick={() => {
                    void operation(validate);
                  }}
                >
                  Validate configuration
                </button>
                <button
                  disabled={
                    busy || !["validated", "uncertain"].includes(workflow.phase)
                  }
                  onClick={() => {
                    void operation(submit);
                  }}
                >
                  {workflow.phase === "uncertain"
                    ? "Retry same submission"
                    : "Run simulation"}
                </button>
              </div>
              <p role="status">Workflow: {workflow.phase}</p>
              {"validation" in workflow && (
                <div>
                  <p>
                    Preflight: {workflow.validation.observations} observations ·{" "}
                    {workflow.validation.symbols} symbols ·{" "}
                    {workflow.validation.sessions} sessions
                  </p>
                  <ul>
                    {workflow.validation.limitations.map((item) => (
                      <li key={item}>{item}</li>
                    ))}
                  </ul>
                </div>
              )}
              {workflow.phase === "submitted" && (
                <button
                  onClick={() => {
                    dispatch({ type: "edit" });
                  }}
                >
                  Configure another run
                </button>
              )}
              <details>
                <summary>Catalog limitations and optional backends</summary>
                <ul>
                  {catalog.limitations.map((item) => (
                    <li key={item}>{item}</li>
                  ))}
                  {catalog.models
                    .filter((item) => !item.available)
                    .map((item) => (
                      <li key={item.name}>
                        {item.name}: {item.reason}
                      </li>
                    ))}
                </ul>
              </details>
            </section>
            <section className="panel">
              <h2>02 / Experiment queue</h2>
              <button
                disabled={busy}
                onClick={() => {
                  void operation(async (signal) => {
                    setJobs((await client.jobs(signal)).jobs);
                    setMonitor((previous) => previous + 1);
                  });
                }}
              >
                Refresh jobs
              </button>
              {jobs.length === 0 && (
                <p>No research jobs yet. Validate a configuration to begin.</p>
              )}
              <ul className="jobs">
                {jobs.map((job) => (
                  <li key={job.job_id}>
                    <code>{job.job_id.slice(0, 12)}</code>
                    <strong>{job.state}</strong>
                    {job.error_code && <p>{job.error_code}</p>}
                    {job.state === "succeeded" && (
                      <>
                        <label>
                          <input
                            type="checkbox"
                            checked={selected.includes(job.job_id)}
                            disabled={
                              !selected.includes(job.job_id) &&
                              selected.length >= 2
                            }
                            onChange={(event) => {
                              setSelected(
                                event.target.checked
                                  ? [...selected, job.job_id]
                                  : selected.filter((id) => id !== job.job_id),
                              );
                            }}
                          />
                          Compare {job.job_id.slice(0, 8)}
                        </label>
                        <button
                          disabled={busy}
                          onClick={() => {
                            void operation(async (signal) => {
                              setEvidence(await client.evidence(job, signal));
                            });
                          }}
                        >
                          Inspect {job.job_id.slice(0, 8)}
                        </button>
                      </>
                    )}
                    {["queued", "running"].includes(job.state) && (
                      <button
                        disabled={busy}
                        onClick={() => {
                          void operation(async (signal) => {
                            const cancelled = await client.cancel(
                              job.job_id,
                              signal,
                            );
                            setJobs((previous) =>
                              previous.map((item) =>
                                item.job_id === cancelled.job_id
                                  ? cancelled
                                  : item,
                              ),
                            );
                          });
                        }}
                      >
                        Cancel research {job.job_id.slice(0, 8)}
                      </button>
                    )}
                    <button
                      disabled={busy}
                      onClick={() => {
                        void operation(async (signal) => {
                          setAudit(await client.audit(job.job_id, signal));
                        });
                      }}
                    >
                      Audit {job.job_id.slice(0, 8)}
                    </button>
                  </li>
                ))}
              </ul>
              <button
                disabled={busy || selected.length !== 2}
                onClick={() => {
                  void operation(async (signal) => {
                    const left = jobs.find((job) => job.job_id === selected[0]);
                    const right = jobs.find(
                      (job) => job.job_id === selected[1],
                    );
                    if (left && right)
                      setComparison(await client.compare(left, right, signal));
                  });
                }}
              >
                Compare selected runs
              </button>
              {audit && (
                <section>
                  <h3>Audit trail</h3>
                  <ol>
                    {audit.events.map((event) => (
                      <li key={event.sequence}>
                        {event.at} · {event.state} {event.code}
                      </li>
                    ))}
                  </ol>
                </section>
              )}
            </section>
          </div>
        )}
        {comparison && (
          <section className="panel">
            <h2>03 / Compare</h2>
            <p>
              {comparison.compatible
                ? "Compatible comparison"
                : "Incompatible comparison"}
              : {comparison.reason}
            </p>
            <div className="comparison">
              {comparison.evidence.map((item, index) => (
                <EvidenceView key={index} evidence={item} />
              ))}
            </div>
          </section>
        )}
        {evidence && (
          <section className="panel">
            <EvidenceView evidence={evidence} />
          </section>
        )}
        <footer>
          Loopback only · No credentials or live-order controls · Legacy
          research and data consoles remain available through their documented
          launch commands.
        </footer>
      </main>
    </div>
  );
}
