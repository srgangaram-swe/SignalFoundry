import { test, expect } from "@playwright/test";
import AxeBuilder from "@axe-core/playwright";

test("real isolated workers: configure, validate, run, inspect and audit", async ({
  page,
}) => {
  const violations: string[] = [];
  page.on("console", (event) => {
    if (event.type() === "error") violations.push(event.text());
  });
  await page.goto("/nexus");
  await expect(page.getByRole("textbox")).toBeVisible();
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
    await page.goto("/nexus");
    await expect(page.getByRole("textbox")).toBeVisible();
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
