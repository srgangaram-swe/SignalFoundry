import { defineConfig, devices } from "@playwright/test";

export default defineConfig({
  testDir: "e2e",
  timeout: 180000,
  expect: { timeout: 10000 },
  retries: 0,
  workers: 1,
  reporter: [["list"], ["json", { outputFile: "test-results/browser.json" }]],
  use: { baseURL: "http://127.0.0.1:8765", trace: "retain-on-failure" },
  projects: [{ name: "chromium", use: { ...devices["Desktop Chrome"] } }],
  webServer: {
    command:
      "../../.venv/bin/signal-foundry --root ../.. --state ../../var/nexus-e2e serve --nexus",
    url: "http://127.0.0.1:8765/nexus",
    reuseExistingServer: false,
    timeout: 60000,
  },
});
