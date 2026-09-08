/**
 * Enforce the console's declared resource budgets against a real build.
 *
 * Budgets are measured from the emitted files rather than from the build log,
 * because a log line is a claim and the files are the evidence. The manifest
 * this writes is committed as machine-readable proof alongside the figure that
 * renders it.
 *
 * Failing loudly matters more than the exact numbers: a console that quietly
 * grew past its budget is a console that stopped being cheap to open, and the
 * whole point of the limits is that they are not negotiable at review time.
 */
import { createHash } from "node:crypto";
import { gzipSync } from "node:zlib";
import { readdirSync, readFileSync, statSync, writeFileSync } from "node:fs";
import { join, relative, extname } from "node:path";

/** Initial-route JavaScript, gzip encoded. */
const MAX_INITIAL_JS_GZIP = 250 * 1024;
/** Initial stylesheet, gzip encoded. */
const MAX_INITIAL_CSS_GZIP = 50 * 1024;
/** Every emitted static asset, uncompressed. */
const MAX_TOTAL_BYTES = 1024 * 1024;

const DIST = new URL("../dist/", import.meta.url).pathname;

function walk(directory) {
  const found = [];
  for (const entry of readdirSync(directory, { withFileTypes: true })) {
    const full = join(directory, entry.name);
    if (entry.isDirectory()) {
      found.push(...walk(full));
    } else if (entry.isFile()) {
      found.push(full);
    }
  }
  return found;
}

function main() {
  let files;
  try {
    files = walk(DIST);
  } catch {
    console.error("bundle budget: dist/ is missing; run `npm run build` first");
    process.exit(2);
  }

  const document = readFileSync(join(DIST, "index.html"), "utf8");
  // The initial route is exactly what the document references. Lazy chunks are
  // fetched on navigation and are deliberately outside the initial budget.
  const referenced = new Set(
    [...document.matchAll(/(?:src|href)="\/console\/([^"]+)"/gu)].map((match) => match[1]),
  );

  const entries = [];
  let totalBytes = 0;
  let initialJsGzip = 0;
  let initialCssGzip = 0;

  for (const file of files.sort()) {
    const relativePath = relative(DIST, file).split("\\").join("/");
    const bytes = readFileSync(file);
    const gzip = gzipSync(bytes, { level: 9 }).length;
    const size = statSync(file).size;
    totalBytes += size;
    const initial = referenced.has(relativePath);
    if (initial && extname(file) === ".js") initialJsGzip += gzip;
    if (initial && extname(file) === ".css") initialCssGzip += gzip;
    entries.push({
      path: relativePath,
      bytes: size,
      gzip_bytes: gzip,
      initial,
      sha256: createHash("sha256").update(bytes).digest("hex"),
    });
  }

  const budgets = [
    { name: "initial_js_gzip", observed: initialJsGzip, limit: MAX_INITIAL_JS_GZIP },
    { name: "initial_css_gzip", observed: initialCssGzip, limit: MAX_INITIAL_CSS_GZIP },
    { name: "total_bytes", observed: totalBytes, limit: MAX_TOTAL_BYTES },
  ];

  const manifest = {
    schema_version: 1,
    evidence_class: "measured_local_build",
    note:
      "Sizes measured from the emitted bundle. Lazy route chunks are excluded from the "
      + "initial budgets and counted in total_bytes.",
    budgets,
    files: entries,
  };
  writeFileSync(
    new URL("../bundle-manifest.json", import.meta.url),
    `${JSON.stringify(manifest, null, 2)}\n`,
  );

  const failures = budgets.filter((budget) => budget.observed > budget.limit);
  for (const budget of budgets) {
    const verdict = budget.observed > budget.limit ? "OVER" : "ok";
    console.log(
      `${budget.name}: ${String(budget.observed)} / ${String(budget.limit)} bytes [${verdict}]`,
    );
  }
  if (failures.length > 0) {
    console.error(`bundle budget: ${String(failures.length)} budget(s) exceeded`);
    process.exit(1);
  }
}

main();
