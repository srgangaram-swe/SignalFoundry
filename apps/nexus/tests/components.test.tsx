import {
  act,
  fireEvent,
  render,
  screen,
  waitFor,
  within,
} from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";
import { App } from "../src/App";
import { ResearchClient, ApiError } from "../src/api";
import { parseConfiguration, message } from "../src/configuration";
import { EvidenceView, TableView } from "../src/EvidenceView";
import { chartSeries, EquityChart } from "../src/EquityChart";
import { catalog, evidence, job, json, request, validation } from "./fixtures";

function clientWithJobs(completed = false) {
  const transport = vi.fn<typeof fetch>((url) =>
    Promise.resolve(
      json(
        typeof url === "string" && url.endsWith("catalog")
          ? catalog
          : {
              schema_version: "1.0.0",
              jobs: completed ? [job, { ...job, job_id: "b".repeat(64) }] : [],
            },
      ),
    ),
  );
  return { client: new ResearchClient(transport), transport };
}

describe("configuration boundary", () => {
  it("retains the complete request and rejects malformed, excessive and unsafe input", () => {
    expect(parseConfiguration(JSON.stringify(request))).toEqual(request);
    for (const text of [
      "{",
      "{}",
      " ".repeat(16385),
      JSON.stringify({ ...request, seed: -1 }),
      JSON.stringify({ ...request, private: "x" }),
    ])
      expect(() => parseConfiguration(text)).toThrow(ApiError);
    expect(message(new Error("private"))).not.toContain("private");
    expect(message(new ApiError("fixture", "Actionable"))).toBe("Actionable");
  });
});

describe("evidence projections", () => {
  it("retains chart gaps and constant or missing evidence without false interpolation", () => {
    const table = {
      name: "equity_fixture",
      description: "Fixture",
      columns: [{ name: "return", unit: "fraction" }],
      rows: [[0], [null], [2], [-1]],
      total_rows: 4,
    };
    const chart = chartSeries([table], "return");
    expect(chart.low).toBe(-1);
    expect(chart.high).toBe(2);
    expect(chart.series[0]?.path.match(/M/gu)).toHaveLength(2);
    expect(chartSeries([{ ...table, rows: [[0]] }], "return").zero).toBe(200);
    expect(
      chartSeries([{ ...table, rows: [] }], "missing").series[0]?.first,
    ).toBe("unavailable");
    expect(chartSeries([table], "missing").series[0]?.path.trim()).toBe("");
    render(<EquityChart tables={[]} title="Net return" metric="return" />);
    expect(screen.getByText("Net return: unavailable.")).toBeVisible();
  });
  it("distinguishes the strategy and all three allowed baselines in chart and legend", () => {
    const tables = Array.from({ length: 4 }, (_, index) => ({
      name: `equity_model${String(index)}`,
      description: "Four-series boundary",
      columns: [{ name: "return", unit: "fraction" }],
      rows: [[0], [index + 1]],
      total_rows: 2,
    }));
    const { container } = render(
      <EquityChart tables={tables} title="Net return" metric="return" />,
    );
    const paths = [...container.querySelectorAll('svg[role="img"] path')];
    const legends = [...container.querySelectorAll(".chart-legend path")];
    expect(new Set(paths.map((path) => path.getAttribute("class"))).size).toBe(
      4,
    );
    expect(legends.map((path) => path.getAttribute("class"))).toEqual(
      paths.map((path) => path.getAttribute("class")),
    );
  });
  it("windows 2048 records, preserves missing values and bounds DOM rows", () => {
    render(
      <TableView
        table={{
          name: "bounded",
          description: "Synthetic boundary fixture",
          columns: [{ name: "value", unit: "fraction" }],
          rows: Array.from({ length: 2048 }, (_, i) => [i === 0 ? null : i]),
          total_rows: 4000,
        }}
      />,
    );
    expect(screen.getAllByRole("row")).toHaveLength(41);
    expect(screen.getByText("Unavailable")).toBeVisible();
    expect(
      screen.getByRole("button", { name: "Previous rows" }),
    ).toBeDisabled();
    fireEvent.click(screen.getByRole("button", { name: "Next rows" }));
    expect(screen.getByRole("status")).toHaveTextContent("Rows 41–80");
    fireEvent.click(screen.getByRole("button", { name: "Previous rows" }));
    expect(screen.getByRole("status")).toHaveTextContent("Rows 1–40");
  });
  it("represents empty records and terminal windows honestly", () => {
    render(
      <TableView
        table={{
          name: "empty",
          description: "No result",
          columns: [{ name: "value", unit: "ratio" }],
          rows: [],
          total_rows: 0,
        }}
      />,
    );
    expect(screen.getByText("No records available.")).toBeVisible();
    expect(screen.getByRole("button", { name: "Next rows" })).toBeDisabled();
    expect(screen.getByRole("status")).toHaveTextContent("Rows 0–0");
  });
  it("renders adversarial evidence as text and never upgrades readiness", () => {
    const { container } = render(
      <EvidenceView
        evidence={{ ...evidence, limitations: ["<script>private()</script>"] }}
      />,
    );
    expect(container.querySelector("script")).toBeNull();
    expect(screen.getByText("<script>private()</script>")).toBeVisible();
    expect(
      screen.getByText(/Development simulation · NOT_READY/),
    ).toBeVisible();
  });
});

describe("workstation integration", () => {
  it("cancels a real job state and displays its structured failure", async () => {
    const queued = { ...job, state: "queued", evidence_hash: null };
    const transport = vi.fn<typeof fetch>((url) =>
      Promise.resolve(
        json(
          typeof url === "string" && url.endsWith("catalog")
            ? catalog
            : { schema_version: "1.0.0", jobs: [queued] },
        ),
      ),
    );
    render(<App client={new ResearchClient(transport)} />);
    await screen.findByRole("textbox");
    transport.mockResolvedValueOnce(
      json({ ...queued, state: "cancelled", error_code: "owner_cancelled" }),
    );
    fireEvent.click(
      screen.getByRole("button", { name: "Cancel research aaaaaaaa" }),
    );
    await screen.findByText("owner_cancelled");
    expect(
      screen.queryByRole("button", { name: /Cancel research/ }),
    ).toBeNull();
  });
  it("polls serially, stops at its count budget and aborts on unmount", async () => {
    const queued = { ...job, state: "queued", evidence_hash: null };
    const transport = vi.fn<typeof fetch>((url) =>
      Promise.resolve(
        json(
          typeof url === "string" && url.endsWith("catalog")
            ? catalog
            : { schema_version: "1.0.0", jobs: [queued] },
        ),
      ),
    );
    vi.useFakeTimers();
    try {
      const view = render(<App client={new ResearchClient(transport)} />);
      await act(async () => {
        await vi.advanceTimersByTimeAsync(0);
      });
      await act(async () => {
        await vi.advanceTimersByTimeAsync(242000);
      });
      expect(screen.getByRole("alert")).toHaveTextContent("120 checks");
      expect(transport).toHaveBeenCalledTimes(122);
      view.unmount();
      await vi.advanceTimersByTimeAsync(10000);
      expect(transport).toHaveBeenCalledTimes(122);
    } finally {
      vi.useRealTimers();
    }
  });
  it("stops polling on a malformed response and rejects changed submission identities", async () => {
    const { client, transport } = clientWithJobs();
    render(<App client={client} />);
    await screen.findByRole("textbox");
    transport.mockResolvedValueOnce(json(validation));
    fireEvent.click(
      screen.getByRole("button", { name: "Validate configuration" }),
    );
    await waitFor(() => {
      expect(
        screen.getByRole("button", { name: "Run simulation" }),
      ).toBeEnabled();
    });
    transport.mockResolvedValueOnce(
      json({ ...job, request_hash: "b".repeat(64) }),
    );
    fireEvent.click(screen.getByRole("button", { name: "Run simulation" }));
    await waitFor(() => {
      expect(screen.getByRole("alert")).toHaveTextContent(
        "differs from preflight",
      );
    });
    transport.mockResolvedValueOnce(
      json({ ...job, state: "queued", evidence_hash: null }),
    );
    vi.useFakeTimers();
    try {
      fireEvent.click(
        screen.getByRole("button", { name: "Retry same submission" }),
      );
      await act(async () => {
        await vi.advanceTimersByTimeAsync(0);
      });
      transport.mockResolvedValueOnce(json({ invalid: true }));
      await act(async () => {
        await vi.advanceTimersByTimeAsync(2000);
      });
      expect(screen.getByRole("alert")).toHaveTextContent("contract");
    } finally {
      vi.useRealTimers();
    }
  });
  it("loads capabilities, changes themes and invalidates preflight after edits", async () => {
    const { client, transport } = clientWithJobs();
    render(<App client={client} />);
    expect(screen.getByRole("status")).toHaveTextContent("Loading");
    const configuration = await screen.findByRole("textbox");
    expect(screen.getByText(/No research jobs yet/)).toBeVisible();
    fireEvent.click(screen.getByText("Execution and financing costs"));
    fireEvent.change(screen.getByLabelText("commission bps (basis points)"), {
      target: { value: "3" },
    });
    expect(
      parseConfiguration((configuration as HTMLTextAreaElement).value).costs
        .commission_bps,
    ).toBe(3);
    fireEvent.click(screen.getByText("Portfolio risk limits"));
    fireEvent.click(screen.getByLabelText("inverse volatility"));
    expect(
      parseConfiguration((configuration as HTMLTextAreaElement).value).risk
        .inverse_volatility,
    ).toBe(false);
    fireEvent.click(screen.getByRole("button", { name: "Light theme" }));
    fireEvent.click(screen.getByRole("button", { name: "Dark theme" }));
    fireEvent.change(screen.getByLabelText("Model"), {
      target: { value: "random_forest" },
    });
    expect((configuration as HTMLTextAreaElement).value).toContain(
      "random_forest",
    );
    fireEvent.change(screen.getByLabelText("Strategy"), {
      target: { value: "threshold" },
    });
    fireEvent.change(screen.getByLabelText("Dataset"), {
      target: { value: "synthetic" },
    });
    transport.mockResolvedValueOnce(json(validation));
    fireEvent.click(
      screen.getByRole("button", { name: "Validate configuration" }),
    );
    await waitFor(() => {
      expect(
        screen.getByRole("button", { name: "Run simulation" }),
      ).toBeEnabled();
    });
    fireEvent.change(configuration, {
      target: { value: JSON.stringify(request) },
    });
    expect(
      screen.getByRole("button", { name: "Run simulation" }),
    ).toBeDisabled();
  });
  it("submits only after validation and retains the same key after an ambiguous failure", async () => {
    const { client, transport } = clientWithJobs();
    render(<App client={client} />);
    await screen.findByRole("textbox");
    transport.mockResolvedValueOnce(json(validation));
    fireEvent.click(
      screen.getByRole("button", { name: "Validate configuration" }),
    );
    await waitFor(() => {
      expect(
        screen.getByRole("button", { name: "Run simulation" }),
      ).toBeEnabled();
    });
    transport.mockRejectedValueOnce(new Error("private transport failure"));
    fireEvent.click(screen.getByRole("button", { name: "Run simulation" }));
    const retry = await screen.findByRole("button", {
      name: "Retry same submission",
    });
    await waitFor(() => {
      expect(retry).toBeEnabled();
    });
    expect(screen.getByRole("textbox")).toBeDisabled();
    const previous = transport.mock.calls.at(-1)?.[1];
    transport.mockResolvedValueOnce(json(job));
    fireEvent.click(retry);
    await screen.findByRole("button", { name: "Configure another run" });
    expect(transport.mock.calls.at(-1)?.[1]?.headers).toEqual(
      previous?.headers,
    );
    expect(transport.mock.calls.at(-1)?.[1]?.body).toEqual(previous?.body);
    fireEvent.click(
      screen.getByRole("button", { name: "Configure another run" }),
    );
    expect(
      screen.getByRole("button", { name: "Validate configuration" }),
    ).toBeEnabled();
  });
  it("shows validation errors and preserves explicit retry", async () => {
    const { client, transport } = clientWithJobs();
    render(<App client={client} />);
    const field = await screen.findByRole("textbox");
    fireEvent.change(field, { target: { value: "{" } });
    fireEvent.change(screen.getByLabelText("Model"), {
      target: { value: "random_forest" },
    });
    expect(screen.getByRole("alert")).toHaveTextContent("valid JSON");
    fireEvent.click(
      screen.getByRole("button", { name: "Validate configuration" }),
    );
    await waitFor(() => {
      expect(
        screen.getByRole("button", { name: "Validate configuration" }),
      ).toBeEnabled();
    });
    fireEvent.change(field, { target: { value: JSON.stringify(request) } });
    transport.mockResolvedValueOnce(
      json(
        {
          schema_version: "1.0.0",
          code: "invalid_request",
          detail: "Invalid fold policy.",
        },
        422,
      ),
    );
    fireEvent.click(
      screen.getByRole("button", { name: "Validate configuration" }),
    );
    await waitFor(() => {
      expect(screen.getByRole("alert")).toHaveTextContent(
        "Invalid fold policy.",
      );
    });
    expect(
      screen.getByRole("button", { name: "Run simulation" }),
    ).toBeDisabled();
  });
  it("inspects evidence, audits, refreshes and compares selected jobs", async () => {
    const { client, transport } = clientWithJobs(true);
    render(<App client={client} />);
    await screen.findByRole("textbox");
    transport.mockResolvedValueOnce(json(evidence));
    fireEvent.click(screen.getByRole("button", { name: "Inspect aaaaaaaa" }));
    await screen.findByRole("heading", { name: "Inspect evidence" });
    transport.mockResolvedValueOnce(
      json({
        schema_version: "1.0.0",
        job_id: job.job_id,
        events: [
          { sequence: 1, state: "succeeded", at: job.updated_at, code: null },
        ],
      }),
    );
    fireEvent.click(screen.getByRole("button", { name: "Audit aaaaaaaa" }));
    await screen.findByRole("heading", { name: "Audit trail" });
    for (const checkbox of screen.getAllByRole("checkbox", {
      name: /^Compare /,
    }))
      fireEvent.click(checkbox);
    transport.mockResolvedValueOnce(
      json({
        schema_version: "1.0.0",
        compatible: false,
        reason: "Different data.",
        evidence: [evidence, evidence],
      }),
    );
    fireEvent.click(
      screen.getByRole("button", { name: "Compare selected runs" }),
    );
    await screen.findByText(/Incompatible comparison/);
    fireEvent.click(screen.getByRole("checkbox", { name: "Compare aaaaaaaa" }));
    expect(
      screen.getByRole("button", { name: "Compare selected runs" }),
    ).toBeDisabled();
    fireEvent.click(screen.getByRole("button", { name: "Refresh jobs" }));
    await waitFor(() => {
      expect(
        screen.getByRole("button", { name: "Refresh jobs" }),
      ).toBeEnabled();
    });
    expect(
      within(screen.getByRole("main")).getAllByRole("article"),
    ).toHaveLength(3);
  });
  it("shows loading failures without exposing unexpected exception details", async () => {
    const client = new ResearchClient(
      vi.fn<typeof fetch>().mockRejectedValue(new Error("private")),
    );
    render(<App client={client} />);
    expect(await screen.findByRole("alert")).not.toHaveTextContent("private");
  });
});
