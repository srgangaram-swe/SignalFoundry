import { act, render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it, vi } from "vitest";
import { ApiError, ResearchClient, type PaperStatus } from "../src/api";
import { PaperConsole } from "../src/PaperConsole";

const status: PaperStatus = {
  environment: "alpaca-paper",
  configured: true,
  enabled: true,
  stopped: false,
  config_identity: "a".repeat(64),
  state: "ready",
  orders: 0,
  events: 0,
  symbols: ["AAA", "BBB", "CCC"],
  feed: "iex",
  maximum_order_notional: "100",
  maximum_position_notional: "500",
  maximum_session_loss: "25",
  blockers: ["Live capability is absent."],
  last_action: "none",
  paper_sessions: 0,
  live_authorized: false,
  cash: null,
  equity: null,
  account_observed_at: null,
  positions: [],
};
const json = (value: unknown) =>
  new Response(JSON.stringify(value), {
    headers: { "Content-Type": "application/json" },
  });

describe("paper operation boundary", () => {
  it("uses generated response guards and sends no arbitrary order inputs", async () => {
    const requests: RequestInit[] = [];
    const client = new ResearchClient(
      vi.fn(async (_url: RequestInfo | URL, options?: RequestInit) => {
        requests.push(options ?? {});
        return json(
          options?.method === "POST"
            ? { status, artifact: "b".repeat(64), account_digest: null }
            : status,
        );
      }),
    );
    render(<PaperConsole client={client} />);
    await screen.findByText(/0 orders/);
    const user = userEvent.setup();
    await user.selectOptions(
      screen.getByLabelText("Paper universe symbol"),
      "BBB",
    );
    for (const label of [
      "Freeze configuration",
      "Check paper connection",
      "Acquire selected history",
      "Evaluate frozen hypotheses",
      "Verify qualification",
      "Start qualified paper session",
      "Run one decision",
      "Reconcile account",
      "Record completed session",
      "Audit paper campaign",
      "Stop and cancel owned orders",
    ]) {
      await user.click(screen.getByRole("button", { name: label }));
      await waitFor(() =>
        expect(screen.getByRole("button", { name: label })).toBeEnabled(),
      );
    }
    await user.click(
      screen.getByRole("button", { name: "Refresh paper state" }),
    );
    const mutations = requests.filter((r) => r.method === "POST");
    expect(mutations).toHaveLength(11);
    for (const request of mutations) {
      expect(JSON.parse(request.body as string) as unknown).toEqual({
        operation: expect.any(String) as unknown,
        symbol: "BBB",
      });
      expect(request.credentials).toBe("omit");
    }
  });

  it("keeps emergency stop available while a request is busy and ignores its late result", async () => {
    let release: ((value: Response) => void) | undefined;
    const client = new ResearchClient(
      vi.fn(async (_url: RequestInfo | URL, options?: RequestInit) => {
        if (options?.method !== "POST") return json(status);
        const action = JSON.parse(options.body as string) as {
          operation: string;
        };
        if (action.operation === "stop")
          return json({
            status: { ...status, stopped: true },
            artifact: null,
            account_digest: null,
          });
        return new Promise<Response>((resolve) => {
          release = resolve;
        });
      }),
    );
    render(<PaperConsole client={client} />);
    await screen.findByText(/0 orders/);
    const user = userEvent.setup();
    await user.click(screen.getByRole("button", { name: "Run one decision" }));
    expect(
      screen.getByRole("button", { name: "Reconcile account" }),
    ).toBeDisabled();
    await user.click(screen.getByRole("button", { name: "Emergency stop" }));
    await screen.findByText(/STOP ENGAGED/);
    await act(async () => {
      release?.(json({ status, artifact: null, account_digest: null }));
    });
    expect(screen.getByText(/STOP ENGAGED/)).toBeInTheDocument();
    expect(
      screen.getByRole("button", { name: "Run one decision" }),
    ).toBeDisabled();
  });

  it("disables unavailable configuration and rejects a live response", async () => {
    const client = new ResearchClient(
      vi.fn(async () => json({ ...status, configured: false, symbols: [] })),
    );
    render(<PaperConsole client={client} />);
    await screen.findByText(/0 orders/);
    expect(
      screen.getByRole("button", { name: "Emergency stop" }),
    ).toBeDisabled();
    expect(
      screen.getByRole("button", { name: "Start qualified paper session" }),
    ).toBeDisabled();
    const malformed = new ResearchClient(
      vi.fn(async () => json({ ...status, live_authorized: true })),
    );
    await expect(
      malformed.paperStatus(new AbortController().signal),
    ).rejects.toBeInstanceOf(ApiError);
  });

  it.each([
    new ApiError("refused", "Gate refused"),
    new Error("private cause"),
  ])("shows bounded status and operation failures", async (cause) => {
    const client = new ResearchClient();
    vi.spyOn(client, "paperStatus")
      .mockRejectedValueOnce(cause)
      .mockResolvedValue(status);
    const view = render(<PaperConsole client={client} />);
    expect(await screen.findByRole("alert")).toHaveTextContent(
      cause instanceof ApiError
        ? "Gate refused"
        : "Paper status is unavailable.",
    );
    view.unmount();
    vi.spyOn(client, "paperAction").mockRejectedValue(cause);
    render(<PaperConsole client={client} />);
    await screen.findByText(/0 orders/);
    await userEvent.click(
      screen.getByRole("button", { name: "Run one decision" }),
    );
    expect(await screen.findByRole("alert")).toHaveTextContent(
      cause instanceof ApiError
        ? "Gate refused"
        : "Reconcile before another decision.",
    );
    expect(screen.queryByText("private cause")).not.toBeInTheDocument();
  });

  it("aborts pending work on unmount without pretending to cancel broker orders", async () => {
    const signals: AbortSignal[] = [];
    const client = new ResearchClient(
      vi.fn(async (_url: RequestInfo | URL, options?: RequestInit) => {
        const signal = options?.signal;
        if (!signal) throw new Error("Expected cancellation signal");
        signals.push(signal);
        return new Promise<Response>((_resolve, reject) => {
          signal.addEventListener(
            "abort",
            () => {
              reject(new Error("Aborted"));
            },
            {
              once: true,
            },
          );
        });
      }),
    );
    const view = render(<PaperConsole client={client} />);
    view.unmount();
    await act(async () => {
      await Promise.resolve();
    });
    expect(signals[0]?.aborted).toBe(true);
    expect(signals).toHaveLength(1);
  });
});
