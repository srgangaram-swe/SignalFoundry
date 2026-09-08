/**
 * Browser tests for the console.
 *
 * These run against the built bundle served by a static preview, not against a
 * development server, because the thing under test is what actually ships.
 *
 * The API is stubbed at the browser boundary by each test rather than by a live
 * service: it lets a test drive the states -- unavailable, invalid, stale --
 * that a healthy service would never produce, and those are exactly the states
 * the console exists to render honestly.
 */
import { defineConfig, devices } from "@playwright/test";

export default defineConfig({
  testDir: "./e2e",
  fullyParallel: true,
  forbidOnly: Boolean(process.env["CI"]),
  retries: 0, // A retry would conceal a flake rather than reveal it.
  // Bounded in CI; local runs use Playwright's own default.
  ...(process.env["CI"] === undefined ? {} : { workers: 2 }),
  reporter: process.env["CI"] === undefined ? [["list"]] : [["list"], ["json", { outputFile: "playwright-report/results.json" }]],
  timeout: 30_000,
  expect: { timeout: 10_000 },
  use: {
    baseURL: "http://127.0.0.1:4183",
    trace: "retain-on-failure",
    screenshot: "only-on-failure",
    video: "off",
  },
  projects: [
    { name: "chromium", use: { ...devices["Desktop Chrome"] } },
    { name: "firefox", use: { ...devices["Desktop Firefox"] } },
    { name: "webkit", use: { ...devices["Desktop Safari"] } },
    // The narrow viewport is a supported target, not an afterthought: the
    // layout must remain usable at 320 CSS pixels.
    { name: "mobile-narrow", use: { ...devices["iPhone SE"], viewport: { width: 320, height: 640 } } },
  ],
  webServer: {
    command: "npx vite preview --port 4183 --strictPort --host 127.0.0.1",
    url: "http://127.0.0.1:4183/console/",
    reuseExistingServer: process.env["CI"] === undefined,
    timeout: 60_000,
  },
});
