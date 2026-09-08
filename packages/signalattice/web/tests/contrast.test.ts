/**
 * The palette is checked numerically so a contrast regression fails here rather
 * than in a browser audit.
 *
 * This exists because the first version of this palette used the published
 * Okabe-Ito values directly. Those are selected for hue separation under colour
 * vision deficiency, not for luminance contrast, and two of them fell below the
 * 4.5:1 that WCAG 2.2 AA requires for text on white.
 */
import { readFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";
import { describe, expect, it } from "vitest";

// Resolved through the filesystem rather than a URL: the test environment does
// not serve `import.meta.url` as a file URL.
const CSS = readFileSync(
  join(dirname(fileURLToPath(import.meta.url)), "..", "src", "styles.css"),
  "utf8",
);

/** WCAG relative luminance for an sRGB hex colour. */
function luminance(hex: string): number {
  const channels = [1, 3, 5].map((offset) => Number.parseInt(hex.slice(offset, offset + 2), 16) / 255);
  const linear = channels.map((channel) =>
    channel <= 0.04045 ? channel / 12.92 : ((channel + 0.055) / 1.055) ** 2.4,
  );
  return 0.2126 * (linear[0] ?? 0) + 0.7152 * (linear[1] ?? 0) + 0.0722 * (linear[2] ?? 0);
}

function contrast(foreground: string, background: string): number {
  const a = luminance(foreground);
  const b = luminance(background);
  return (Math.max(a, b) + 0.05) / (Math.min(a, b) + 0.05);
}

/** Read a custom property from the first (light) `:root` block. */
function token(name: string): string {
  const block = CSS.slice(CSS.indexOf(":root {"), CSS.indexOf("@media (prefers-color-scheme: dark)"));
  const match = new RegExp(`--${name}:\\s*(#[0-9a-f]{6})`, "u").exec(block);
  if (match?.[1] === undefined) throw new Error(`token --${name} not found in the light palette`);
  return match[1];
}

describe("light palette contrast", () => {
  const surface = token("surface");

  it.each(["state-ready", "state-warn", "state-fail", "state-info", "state-muted"])(
    "%s meets WCAG 2.2 AA against the surface",
    (name) => {
      expect(contrast(token(name), surface)).toBeGreaterThanOrEqual(4.5);
    },
  );

  it("body text clears AA comfortably", () => {
    expect(contrast(token("ink"), surface)).toBeGreaterThanOrEqual(7);
  });

  it("muted text still clears AA", () => {
    expect(contrast(token("ink-muted"), surface)).toBeGreaterThanOrEqual(4.5);
  });

  it("keeps the state colours distinguishable from one another", () => {
    // Contrast alone is not the goal: the tones must remain separable, which is
    // why they are darkened hues rather than shades of one colour.
    const tones = ["state-ready", "state-warn", "state-fail", "state-info"].map(token);
    expect(new Set(tones).size).toBe(tones.length);
  });
});
