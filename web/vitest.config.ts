/**
 * Vitest configuration, kept separate from the build config.
 *
 * The 90% branch floor is a gate, not a target: it is the level below which a
 * console state has demonstrably never been rendered in a test. Every state in
 * the honest-state model has a branch somewhere, so a shortfall means one of
 * them is unexercised.
 *
 * Time, locale, timezone, randomness, and network are all pinned in
 * `tests/setup.ts`, because a console that renders timestamps and intervals
 * would otherwise pass or fail depending on where it ran.
 */
import { defineConfig } from "vitest/config";
import react from "@vitejs/plugin-react";

export default defineConfig({
  plugins: [react()],
  test: {
    environment: "happy-dom",
    globals: true,
    setupFiles: ["./tests/setup.ts"],
    include: ["tests/**/*.test.ts", "tests/**/*.test.tsx"],
    restoreMocks: true,
    unstubEnvs: true,
    unstubGlobals: true,
    coverage: {
      provider: "v8",
      reporter: ["text-summary", "json-summary"],
      reportsDirectory: "coverage",
      include: ["src/**/*.ts", "src/**/*.tsx"],
      // schema.ts is generated type-only output with no runtime branches;
      // main.tsx is the DOM bootstrap, exercised end to end by Playwright.
      exclude: ["src/api/schema.ts", "src/main.tsx", "src/vite-env.d.ts"],
      thresholds: { branches: 90, functions: 90, lines: 90, statements: 90 },
    },
  },
});
