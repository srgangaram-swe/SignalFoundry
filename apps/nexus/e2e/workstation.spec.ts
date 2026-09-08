import { test, expect, type Page } from "@playwright/test";
import AxeBuilder from "@axe-core/playwright";
import { catalog, evidence, job } from "../tests/fixtures";

/** Catalog starts two real isolated workers; await its existing transport boundary. */
async function openWorkstation(page: Page) {
  const started = performance.now();
  const response = page.waitForResponse(
    (value) => new URL(value.url()).pathname === "/api/v1/catalog",
    { timeout: 30000 },
  );
  await page.goto("/nexus");
  const catalog = await response;
  await test.info().attach("catalog-readiness", {
    body: JSON.stringify({
      status: catalog.status(),
      elapsed_ms: performance.now() - started,
    }),
    contentType: "application/json",
  });
  expect(catalog.ok()).toBe(true);
  await expect(page.getByRole("textbox")).toBeVisible();
}

test("real isolated workers: configure, validate, run, inspect and audit", async ({
  page,
}) => {
  const violations: string[] = [];
  page.on("console", (event) => {
    if (event.type() === "error") violations.push(event.text());
  });
  await openWorkstation(page);
  await page.getByRole("button", { name: "Validate configuration" }).click();
  await expect(
    page.getByRole("button", { name: "Run simulation" }),
  ).toBeEnabled({ timeout: 60000 });
  await page.getByRole("button", { name: "Run simulation" }).click();
  await expect(
    page.getByRole("button", { name: /^Inspect / }).first(),
  ).toBeVisible({ timeout: 150000 });
  await page
    .getByRole("button", { name: /^Inspect / })
    .first()
    .click();
  await expect(
    page.getByRole("heading", { name: "Inspect evidence" }),
  ).toBeVisible();
  await expect(page.getByRole("img", { name: /^Drawdown/ })).toBeVisible();
  await page
    .getByRole("button", { name: /^Audit / })
    .first()
    .click();
  await expect(
    page.getByRole("heading", { name: "Audit trail" }),
  ).toBeVisible();
  expect(
    (
      await new AxeBuilder({ page })
        .withTags(["wcag2a", "wcag2aa", "wcag21aa", "wcag22aa"])
        .analyze()
    ).violations,
  ).toEqual([]);
  expect(violations).toEqual([]);
});

test("keyboard, responsive themes, reduced motion and deterministic appearance", async ({
  page,
}) => {
  // Empty queue stabilizes the reviewable visual fixture; the other test uses real workers.
  await page.route("**/api/v1/jobs", (route) =>
    route.fulfill({ json: { schema_version: "1.0.0", jobs: [] } }),
  );
  await page.emulateMedia({ reducedMotion: "reduce" });
  for (const width of [1440, 390]) {
    await page.setViewportSize({ width, height: 1000 });
    await openWorkstation(page);
    await page.keyboard.press("Tab");
    await expect(
      page.getByRole("link", { name: "Skip to workspace" }),
    ).toBeFocused();
    await page.keyboard.press("Enter");
    for (const theme of ["dark", "light"]) {
      if (theme === "light")
        await page.getByRole("button", { name: "Light theme" }).click();
      expect(
        (
          await new AxeBuilder({ page })
            .withTags(["wcag2a", "wcag2aa", "wcag21aa", "wcag22aa"])
            .analyze()
        ).violations,
      ).toEqual([]);
      expect(
        await page.evaluate(
          () => document.documentElement.scrollWidth <= innerWidth,
        ),
      ).toBe(true);
      await expect(page).toHaveScreenshot(
        `nexus-${String(width)}-${theme}.png`,
        {
          fullPage: true,
          animations: "disabled",
        },
      );
    }
  }
});

test("four model lines and legend swatches remain distinct without color", async ({
  page,
}) => {
  const tables = Array.from({ length: 4 }, (_, index) => ({
    name: `equity_model${String(index)}`,
    description: "Synthetic four-series boundary",
    columns: [
      { name: "date", unit: "UTC date" },
      { name: "cumulative_return", unit: "fraction" },
      { name: "drawdown", unit: "fraction" },
    ],
    rows: [
      ["2026-01-01", 0, 0],
      ["2026-01-02", (index + 1) / 10, -index / 10],
    ],
    total_rows: 2,
  }));
  await page.route("**/api/v1/catalog", (route) =>
    route.fulfill({ json: catalog }),
  );
  await page.route("**/api/v1/jobs", (route) =>
    route.fulfill({ json: { schema_version: "1.0.0", jobs: [job] } }),
  );
  await page.route("**/api/v1/jobs/*/evidence", (route) =>
    route.fulfill({ json: { ...evidence, tables } }),
  );
  await openWorkstation(page);
  await page.getByRole("button", { name: "Inspect aaaaaaaa" }).click();
  const chart = page.getByRole("img", { name: /^Net cumulative/ });
  await expect(chart).toBeVisible();
  const patterns = await chart
    .locator("path")
    .evaluateAll((paths) =>
      paths.map((path) => getComputedStyle(path).strokeDasharray),
    );
  expect(new Set(patterns).size).toBe(4);
  const legends = page.locator(".chart-legend").first().locator("path");
  expect(
    await legends.evaluateAll((paths) =>
      paths.map((path) => getComputedStyle(path).strokeDasharray),
    ),
  ).toEqual(patterns);
  const swatch = await page.locator(".swatch").first().boundingBox();
  expect(swatch?.height).toBe(8);
  expect(swatch?.width).toBe(26);
  expect(
    (
      await new AxeBuilder({ page })
        .withTags(["wcag2a", "wcag2aa", "wcag21aa", "wcag22aa"])
        .analyze()
    ).violations,
  ).toEqual([]);
  await page.screenshot({
    path: test.info().outputPath("four-series.png"),
    fullPage: true,
  });
});
