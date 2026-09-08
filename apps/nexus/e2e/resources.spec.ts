import { test, expect } from "@playwright/test";
import { performance } from "node:perf_hooks";
import { execFileSync } from "node:child_process";
import { platform, arch, release } from "node:os";
import { evidence, job, catalog } from "../tests/fixtures";

test("bounded synthetic table resource evidence", async ({
  page,
  browser,
}, info) => {
  const table = {
    name: "bounded",
    description: "Synthetic 2048-row stress fixture; no market observations.",
    columns: [{ name: "value", unit: "index" }],
    rows: Array.from({ length: 2048 }, (_, index) => [index]),
    total_rows: 2048,
  };
  await page.route("**/api/v1/catalog", (route) =>
    route.fulfill({ json: catalog }),
  );
  await page.route("**/api/v1/jobs", (route) =>
    route.fulfill({ json: { schema_version: "1.0.0", jobs: [job] } }),
  );
  await page.route("**/api/v1/jobs/*/evidence", (route) =>
    route.fulfill({ json: { ...evidence, tables: [table] } }),
  );
  const session = await page.context().newCDPSession(page);
  await session.send("Performance.enable");
  const samples: {
    cold_load_ms: number;
    interaction_ms: number;
    renderer_cpu_s: number;
    javascript_heap_bytes: number;
    rendered_rows: number;
    browser_process_rss_bytes: number;
  }[] = [];
  for (let sample = 0; sample < 8; sample += 1) {
    const start = performance.now();
    await page.goto("/nexus");
    await expect(page.getByRole("textbox")).toBeVisible();
    const cold = performance.now() - start;
    await page.getByRole("button", { name: "Inspect aaaaaaaa" }).click();
    await expect(page.getByRole("table")).toBeVisible();
    const before: { metrics: { name: string; value: number }[] } =
      await session.send("Performance.getMetrics");
    const interaction = performance.now();
    await page.getByRole("button", { name: "Next rows" }).click();
    await expect(page.getByText("Rows 41–80")).toBeVisible();
    const latency = performance.now() - interaction;
    const after: { metrics: { name: string; value: number }[] } =
      await session.send("Performance.getMetrics");
    const metric = (values: typeof before, name: string) => {
      const found = values.metrics.find((item) => item.name === name);
      if (!found) throw new Error("Missing browser resource metric: " + name);
      return found.value;
    };
    const row = {
      cold_load_ms: cold,
      interaction_ms: latency,
      renderer_cpu_s:
        metric(after, "TaskDuration") - metric(before, "TaskDuration"),
      javascript_heap_bytes: metric(after, "JSHeapUsedSize"),
      rendered_rows: await page.getByRole("row").count(),
      browser_process_rss_bytes: 0,
    };
    const browserSession = await browser.newBrowserCDPSession();
    try {
      const processes: { processInfo: { id: number }[] } =
        await browserSession.send("SystemInfo.getProcessInfo");
      const pids = processes.processInfo.map((item) => String(item.id));
      const rss = execFileSync("ps", ["-o", "rss=", "-p", pids.join(",")], {
        encoding: "utf8",
        timeout: 3000,
        maxBuffer: 16384,
      });
      const values = rss.trim().split(/\s+/u).map(Number);
      if (
        !values.length ||
        values.some((value) => !Number.isFinite(value) || value <= 0)
      )
        throw new Error("Invalid process RSS sample");
      row.browser_process_rss_bytes =
        values.reduce((total, value) => total + value, 0) * 1024;
    } finally {
      await browserSession.detach();
    }
    expect(row.rendered_rows).toBe(41);
    samples.push(row);
  }
  const budgets = {
    cold_load_ms: 5000,
    interaction_ms: 1000,
    renderer_cpu_s: 2,
    javascript_heap_bytes: 134217728,
    rendered_rows: 41,
    browser_process_rss_bytes: 1073741824,
  };
  for (const row of samples)
    for (const key of Object.keys(budgets) as (keyof typeof budgets)[])
      expect(row[key]).toBeLessThanOrEqual(budgets[key]);
  await info.attach("resources.json", {
    body: JSON.stringify(
      {
        schema_version: "1.0.0",
        environment: {
          platform: platform(),
          arch: arch(),
          release: release(),
          browser: browser.version(),
          node: process.version,
        },
        fixture: "synthetic 2048 rows; seed not applicable",
        samples,
        budgets,
        limitations: [
          "Local Chromium only; renderer task time is not total host CPU.",
          "Process RSS sums browser processes, counts shared pages repeatedly, and is sampled rather than peak RSS.",
          "Cold navigation disables HTTP cache; operating-system and browser startup caches are not reset.",
          "No exchange latency or trading-capacity inference.",
        ],
      },
      null,
      2,
    ),
    contentType: "application/json",
  });
});
