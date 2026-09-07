/**
 * Shared browser-test helpers.
 *
 * `stubApi` intercepts at the network layer so the console's own transport
 * still runs: the request it issues, the headers it sets, and the decoding it
 * performs are all exercised, and only the service is replaced.
 */
import type { Page, Route } from "@playwright/test";

export const DIGEST = "a".repeat(64);
export const COHORT = "d".repeat(64);

export function lane(overrides: Record<string, unknown> = {}): Record<string, unknown> {
  return {
    schema_version: 1,
    lane_identity: DIGEST,
    purpose: "shadow-eval",
    target: "direction",
    horizon_days: 5,
    frequency: "daily",
    universe: "us-large-cap",
    decision_policy: "long-short",
    environment: "local",
    state: "active",
    champion_revision: DIGEST,
    generation: 1,
    freeze_trigger: null,
    created_at: "2026-08-01T00:00:00+00:00",
    event_count: 2,
    chain_verified: true,
    chain_fault: null,
    events_by_kind: [["assignment", 1]],
    ...overrides,
  };
}

/** A route body, or the sentinel that makes the read fail as if offline. */
export type ApiRoutes = Record<string, unknown>;

/** Sentinel body meaning "abort this request as a connection failure". */
export const UNAVAILABLE = "unavailable";

/** Serve the declared routes; anything else is a bounded 404 problem. */
export async function stubApi(page: Page, routes: ApiRoutes): Promise<void> {
  await page.route("**/api/v1/**", async (route: Route) => {
    const path = new URL(route.request().url()).pathname;
    const body = routes[path];
    if (body === undefined) {
      await route.fulfill({
        status: 404,
        contentType: "application/problem+json",
        body: JSON.stringify({ code: "not_found", title: "Not found", detail: "x", status: 404 }),
      });
      return;
    }
    if (body === "unavailable") {
      await route.abort("connectionrefused");
      return;
    }
    await route.fulfill({
      status: 200,
      contentType: "application/json",
      body: JSON.stringify(body),
    });
  });
  await page.route("**/health/**", async (route: Route) => {
    const path = new URL(route.request().url()).pathname;
    const body = routes[path] ?? { status: path.endsWith("ready") ? "ready" : "live" };
    if (body === "unavailable") {
      await route.abort("connectionrefused");
      return;
    }
    await route.fulfill({ status: 200, contentType: "application/json", body: JSON.stringify(body) });
  });
  await page.route("**/internal/metrics", async (route: Route) => {
    await route.fulfill({
      status: 200,
      contentType: "text/plain",
      body: 'sl_requests_total{route="runs"} 12\nsl_latency_seconds 0.004\n',
    });
  });
}
