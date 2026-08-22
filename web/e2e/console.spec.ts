/**
 * End-to-end behaviour of the shipped bundle in a real browser.
 *
 * These cover what a component test cannot: the lazy route chunks actually
 * loading, keyboard-only navigation, focus moving on route change, deep links,
 * and the layout surviving a 320-pixel viewport and 200% zoom.
 */
import { expect, test } from "@playwright/test";
import AxeBuilder from "@axe-core/playwright";

import { lane, stubApi } from "./fixtures";

const VIEWS = [
  { path: "/console/", heading: "System overview" },
  { path: "/console/runs", heading: "Run catalog" },
  { path: "/console/evidence", heading: "Run evidence" },
  { path: "/console/comparison", heading: "Model comparison" },
  { path: "/console/calibration", heading: "Calibration and uncertainty" },
  { path: "/console/operations", heading: "Drift, latency and operations" },
  { path: "/console/governance", heading: "Governance and readiness" },
] as const;

const DEFAULT_ROUTES = {
  "/api/v1/runs": { schema_version: 1, items: [{ run_id: "r1_demo", status: "succeeded" }], next_cursor: null },
  "/api/v1/governance/lanes": { schema_version: 1, items: [lane()], next_cursor: null },
};

test.describe("routes", () => {
  for (const view of VIEWS) {
    test(`deep link to ${view.path} renders ${view.heading}`, async ({ page }) => {
      await stubApi(page, DEFAULT_ROUTES);
      await page.goto(view.path);
      // A lazily loaded chunk must actually arrive; the heading proves it did.
      await expect(page.getByRole("heading", { level: 2, name: view.heading })).toBeVisible();
    });
  }

  test("an unknown console path renders the bounded message, not a view", async ({ page }) => {
    await stubApi(page, DEFAULT_ROUTES);
    await page.goto("/console/not-a-view");
    await expect(page.getByText(/exactly seven views/)).toBeVisible();
  });
});

test.describe("keyboard access", () => {
  test("the skip link precedes the navigation and reaches main content", async ({ page }) => {
    await stubApi(page, DEFAULT_ROUTES);
    await page.goto("/console/");

    // On load the shell has already moved focus to the main heading, which is
    // why a fresh Tab does not land on the skip link: the reader is in the
    // content rather than ahead of it. The link matters once focus returns to
    // the navigation, so the property under test is its position in the tab
    // order, not what happens to be focused first.
    const firstFocusable = await page.evaluate(() => {
      const candidates = document.querySelectorAll<HTMLElement>(
        'a[href], button:not([disabled]), input, [tabindex]:not([tabindex="-1"])',
      );
      return candidates[0]?.className ?? "";
    });
    expect(firstFocusable).toContain("skip-link");

    // It becomes visible on focus rather than staying off-screen.
    const link = page.getByRole("link", { name: /Skip to main content/ });
    await link.focus();
    await expect(link).toBeFocused();
    await expect(link).toBeInViewport();

    await page.keyboard.press("Enter");
    await expect(page).toHaveURL(/#main$/);
    await expect(page.locator("#main")).toBeVisible();
  });

  test("every view is reachable and activatable by keyboard alone", async ({ page }) => {
    // Asserted by focusing and activating each link rather than by counting Tab
    // presses: the number of stops between two elements differs across engines
    // (WebKit only includes links when full keyboard access is enabled), so a
    // count would be testing the browser rather than the console.
    await stubApi(page, DEFAULT_ROUTES);
    await page.goto("/console/");
    for (const view of VIEWS) {
      const link = page.getByRole("link", { name: view.heading });
      await link.focus();
      await expect(link).toBeFocused();
      await page.keyboard.press("Enter");
      await expect(page.getByRole("heading", { level: 2, name: view.heading })).toBeVisible();
    }
  });

  test("the focusable order is the reading order on every engine", async ({ page }) => {
    // Engine-independent: read the tab order from the DOM rather than by
    // pressing Tab, so the assertion holds even where the platform declines to
    // visit buttons and links.
    await stubApi(page, DEFAULT_ROUTES);
    await page.goto("/console/evidence");
    // The view is a lazy chunk; reading the DOM before it renders would sample
    // the Suspense fallback instead of the page.
    await expect(page.getByLabel(/Run reference/)).toBeVisible();
    const order = await page.evaluate(() =>
      [
        ...document.querySelectorAll<HTMLElement>(
          'a[href], button:not([disabled]), input, [tabindex]:not([tabindex="-1"])',
        ),
      ].map((element) => `${element.tagName}:${element.getAttribute("aria-label") ?? element.textContent.trim().slice(0, 24)}`),
    );
    // The skip link is first, and the field precedes the button that submits it.
    expect(order[0]).toContain("Skip to main content");
    const field = order.findIndex((entry) => entry.startsWith("INPUT"));
    const submit = order.findIndex((entry) => entry.includes("Load evidence"));
    expect(field).toBeGreaterThan(-1);
    expect(submit).toBeGreaterThan(field);
    // No positive tabindex anywhere: those override document order and are the
    // usual cause of an unpredictable keyboard path.
    const positive = await page.evaluate(
      () => document.querySelectorAll('[tabindex]:not([tabindex="0"]):not([tabindex="-1"])').length,
    );
    expect(positive).toBe(0);
  });

  test("focus moves to the new heading on route change", async ({ page }) => {
    await stubApi(page, DEFAULT_ROUTES);
    await page.goto("/console/");
    await page.getByRole("link", { name: "Governance and readiness" }).click();
    // Without this a screen-reader user is left on a stale element with no
    // indication the page changed.
    await expect(page.getByRole("heading", { level: 2, name: "Governance and readiness" })).toBeFocused();
  });

  test("no focus trap: focus moves between adjacent controls both ways", async ({
    page,
    browserName,
  }) => {
    // Skipped on WebKit, not weakened: macOS Safari omits buttons and links
    // from the Tab order entirely unless "Full Keyboard Access" is enabled in
    // system settings, and Playwright cannot set it. Asserting synthetic Tab
    // movement there would measure that setting rather than this console. The
    // DOM-order property is asserted separately on every engine below.
    test.skip(browserName === "webkit", "macOS WebKit omits buttons from the tab order by default");
    await stubApi(page, DEFAULT_ROUTES);
    await page.goto("/console/evidence");
    const field = page.getByLabel(/Run reference/);
    const submit = page.getByRole("button", { name: /Load evidence/ });
    await field.focus();
    await expect(field).toBeFocused();
    await page.keyboard.press("Tab");
    await expect(submit).toBeFocused();
    await page.keyboard.press("Shift+Tab");
    // The round trip is the portable statement of "nothing traps focus".
    await expect(field).toBeFocused();
  });
});

test.describe("accessibility", () => {
  for (const view of VIEWS) {
    test(`axe reports no serious or critical violation on ${view.path}`, async ({ page }) => {
      await stubApi(page, DEFAULT_ROUTES);
      await page.goto(view.path);
      await expect(page.getByRole("heading", { level: 2, name: view.heading })).toBeVisible();
      const results = await new AxeBuilder({ page })
        .withTags(["wcag2a", "wcag2aa", "wcag21a", "wcag21aa", "wcag22aa"])
        .analyze();
      const blocking = results.violations.filter(
        (violation) => violation.impact === "serious" || violation.impact === "critical",
      );
      // Nothing is waived: the assertion names the rules so a failure is
      // actionable rather than a bare count.
      expect(blocking.map((violation) => violation.id)).toEqual([]);
    });
  }

  test("failure states remain accessible", async ({ page }) => {
    await stubApi(page, {
      "/api/v1/governance/lanes": {
        schema_version: 1,
        items: [lane({ chain_verified: false, chain_fault: "sequence gap at 3" })],
        next_cursor: null,
      },
    });
    await page.goto("/console/governance");
    await expect(page.getByText(/does not verify/)).toBeVisible();
    const results = await new AxeBuilder({ page })
      .withTags(["wcag2a", "wcag2aa", "wcag21aa", "wcag22aa"])
      .analyze();
    expect(
      results.violations
        .filter((v) => v.impact === "serious" || v.impact === "critical")
        .map((v) => v.id),
    ).toEqual([]);
  });

  test("status is conveyed by text, not colour alone", async ({ page }) => {
    await stubApi(page, DEFAULT_ROUTES);
    await page.goto("/console/governance");
    // The word is present in the accessible name, so the finding survives a
    // reader who cannot distinguish the palette.
    await expect(page.getByText("Verified").first()).toBeVisible();
    await expect(page.getByText("active").first()).toBeVisible();
  });
});

test.describe("layout", () => {
  test("no horizontal document scroll at 320 pixels", async ({ page }) => {
    await stubApi(page, DEFAULT_ROUTES);
    await page.setViewportSize({ width: 320, height: 640 });
    await page.goto("/console/governance");
    await expect(page.getByRole("table")).toBeVisible();
    const overflow = await page.evaluate(
      () => document.documentElement.scrollWidth > document.documentElement.clientWidth + 1,
    );
    // Wide tables scroll inside their own container, never the document.
    expect(overflow).toBe(false);
  });

  test("remains usable at 200% zoom", async ({ page }) => {
    await stubApi(page, DEFAULT_ROUTES);
    await page.setViewportSize({ width: 640, height: 480 });
    await page.goto("/console/runs");
    await page.evaluate(() => {
      document.documentElement.style.fontSize = "32px";
    });
    await expect(page.getByRole("heading", { level: 2, name: "Run catalog" })).toBeVisible();
    const overflow = await page.evaluate(
      () => document.documentElement.scrollWidth > document.documentElement.clientWidth + 1,
    );
    expect(overflow).toBe(false);
  });

  test("honours reduced motion", async ({ page }) => {
    await page.emulateMedia({ reducedMotion: "reduce" });
    await stubApi(page, DEFAULT_ROUTES);
    await page.goto("/console/");
    const seconds = await page.evaluate(() => {
      const element = document.querySelector(".panel");
      if (element === null) return 0;
      return Number.parseFloat(getComputedStyle(element).transitionDuration);
    });
    // The override collapses every transition to effectively zero.
    expect(seconds).toBeLessThan(0.01);
  });
});

test.describe("failure and recovery", () => {
  test("an unreachable service renders as unavailable, not as empty", async ({ page }) => {
    await stubApi(page, { "/health/live": "unavailable", "/health/ready": "unavailable" });
    await page.goto("/console/");
    await expect(page.getByText(/not reachable at this origin/).first()).toBeVisible();
  });

  test("recovers when the operator retries after the service returns", async ({ page }) => {
    let healthy = false;
    await page.route("**/health/**", async (route) => {
      if (!healthy) {
        await route.abort("connectionrefused");
        return;
      }
      await route.fulfill({
        status: 200,
        contentType: "application/json",
        body: JSON.stringify({ status: route.request().url().endsWith("ready") ? "ready" : "live" }),
      });
    });
    await page.goto("/console/");
    await expect(page.getByText(/not reachable/).first()).toBeVisible();
    healthy = true;
    // Retry is an explicit human action; the console never retries on its own.
    await page.getByRole("button", { name: /Re-read service status/ }).click();
    await expect(page.getByText(/Reported status: live/)).toBeVisible();
  });

  test("incompatible evidence is reported rather than rendered", async ({ page }) => {
    await stubApi(page, {
      "/api/v1/governance/lanes": { schema_version: 99, items: [], next_cursor: null },
    });
    await page.goto("/console/governance");
    await expect(page.getByText(/did not match the version-1 contract/)).toBeVisible();
  });
});

test.describe("security posture", () => {
  test("the console issues only same-origin GET requests", async ({ page }) => {
    const observed: { method: string; origin: string }[] = [];
    page.on("request", (request) => {
      const url = new URL(request.url());
      observed.push({ method: request.method(), origin: url.origin });
    });
    await stubApi(page, DEFAULT_ROUTES);
    await page.goto("/console/governance");
    await expect(page.getByRole("table")).toBeVisible();
    expect(observed.every((entry) => entry.method === "GET")).toBe(true);
    expect(new Set(observed.map((entry) => entry.origin))).toEqual(
      new Set(["http://127.0.0.1:4183"]),
    );
  });

  test("the console persists nothing in the browser", async ({ page }) => {
    await stubApi(page, DEFAULT_ROUTES);
    await page.goto("/console/governance");
    await expect(page.getByRole("table")).toBeVisible();
    const stored = await page.evaluate(() => ({
      local: window.localStorage.length,
      session: window.sessionStorage.length,
      cookies: document.cookie,
    }));
    expect(stored).toEqual({ local: 0, session: 0, cookies: "" });
  });

  test("no service worker is registered", async ({ page }) => {
    await stubApi(page, DEFAULT_ROUTES);
    await page.goto("/console/");
    const registrations = await page.evaluate(async () =>
      "serviceWorker" in navigator ? (await navigator.serviceWorker.getRegistrations()).length : 0,
    );
    expect(registrations).toBe(0);
  });

  test("server text is rendered as text, never as markup", async ({ page }) => {
    await stubApi(page, {
      "/api/v1/governance/lanes": {
        schema_version: 1,
        items: [
          lane({
            chain_verified: false,
            chain_fault: "<img src=x onerror=alert(1)>broken at 3",
          }),
        ],
        next_cursor: null,
      },
    });
    await page.goto("/console/governance");
    await expect(page.getByText(/onerror=alert\(1\)/)).toBeVisible();
    // The markup arrived as characters and stayed characters.
    expect(await page.locator("img").count()).toBe(0);
  });

  test("the lane identity input cannot escape into the request path", async ({ page }) => {
    const requested: string[] = [];
    page.on("request", (request) => {
      requested.push(new URL(request.url()).pathname);
    });
    await stubApi(page, {});
    await page.goto("/console/comparison");
    await page.getByLabel(/Lane identity/).fill("../../etc/passwd");
    await page.getByRole("button", { name: /Load comparisons/ }).click();
    await expect(page.getByText(/could not issue this read|refused/).first()).toBeVisible();
    expect(requested.some((path) => path.includes("etc/passwd"))).toBe(false);
  });
});
