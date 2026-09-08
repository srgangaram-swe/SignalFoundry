import { test, expect } from "@playwright/test";
import AxeBuilder from "@axe-core/playwright";
import { catalog } from "../tests/fixtures";

test("paper panel defaults to unavailable through the real HTTP service", async ({
  page,
}) => {
  await page.route("**/api/v1/catalog", (route) =>
    route.fulfill({ json: catalog }),
  );
  await page.goto("/nexus");
  await page
    .getByRole("button", { name: "Open paper operations", exact: true })
    .click();
  await expect(page.getByText(/not_configured/)).toBeVisible();
  await expect(
    page.getByRole("button", { name: "Emergency stop" }),
  ).toBeDisabled();
  expect(
    (
      await new AxeBuilder({ page })
        .withTags(["wcag2a", "wcag2aa", "wcag22aa"])
        .analyze()
    ).violations,
  ).toEqual([]);
});

test("paper fixture remains legible and stoppable at both viewport widths", async ({
  page,
}) => {
  const status = {
    environment: "alpaca-paper",
    configured: true,
    enabled: true,
    stopped: false,
    config_identity: "a".repeat(64),
    symbols: ["AAA", "BBB", "CCC"],
    feed: "iex",
    maximum_order_notional: "100",
    maximum_position_notional: "500",
    maximum_session_loss: "25",
    state: "reconciliation_required",
    orders: 1,
    events: 12,
    blockers: [
      "Fixture only: account reconciliation is required.",
      "Live capability is absent.",
    ],
    last_action: "cycle",
    paper_sessions: 0,
    cash: "9899.00",
    equity: "10000.00",
    account_observed_at: "2026-09-08T15:00:00Z",
    positions: [{ symbol: "AAA", quantity: "1", market_value: "101" }],
    live_authorized: false,
  };
  await page.route("**/api/v1/catalog", (route) =>
    route.fulfill({ json: catalog }),
  );
  await page.route("**/api/v1/jobs", (route) =>
    route.fulfill({ json: { schema_version: "1.0.0", jobs: [] } }),
  );
  await page.route("**/api/v1/paper{,/stop}", (route) =>
    route.fulfill({
      json:
        route.request().method() === "GET"
          ? status
          : {
              status: { ...status, stopped: true },
              artifact: null,
              account_digest: null,
            },
    }),
  );
  for (const width of [1440, 390]) {
    await page.setViewportSize({ width, height: 1000 });
    await page.goto("/nexus");
    await page
      .getByRole("button", { name: "Open paper operations", exact: true })
      .click();
    await expect(page.getByText(/9899.00/)).toBeVisible();
    await expect(
      page.getByRole("list", { name: "Reconciled paper positions" }),
    ).toContainText("AAA: 1 shares");
    expect(
      await page.evaluate(
        () => document.documentElement.scrollWidth <= innerWidth,
      ),
    ).toBe(true);
    expect(
      (
        await new AxeBuilder({ page })
          .withTags(["wcag2a", "wcag2aa", "wcag22aa"])
          .analyze()
      ).violations,
    ).toEqual([]);
    await page.screenshot({
      path: test.info().outputPath(`paper-${String(width)}.png`),
      fullPage: true,
    });
    await page.getByRole("button", { name: "Emergency stop" }).click();
    await expect(page.getByText(/STOP ENGAGED/)).toBeVisible();
    await expect(
      page.getByRole("button", { name: "Run one decision" }),
    ).toBeDisabled();
  }
});
