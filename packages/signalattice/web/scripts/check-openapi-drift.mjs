/**
 * Fail if the committed generated bindings no longer match the contract.
 *
 * Generation is re-run into a temporary file and compared byte for byte. A
 * console whose types drifted from the served contract compiles happily and
 * then renders fields the service does not send, which is exactly the class of
 * failure runtime decoding exists to catch at the boundary -- this check moves
 * it earlier, to build time.
 */
import { execFileSync } from "node:child_process";
import { mkdtempSync, readFileSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";

const CONTRACT = new URL("../../docs/api/openapi-v1.json", import.meta.url).pathname;
const COMMITTED = new URL("../src/api/schema.ts", import.meta.url).pathname;

const scratch = mkdtempSync(join(tmpdir(), "openapi-drift-"));
const candidate = join(scratch, "schema.ts");
try {
  execFileSync(
    "npx",
    ["--no-install", "openapi-typescript", CONTRACT, "-o", candidate],
    { stdio: "pipe" },
  );
  const fresh = readFileSync(candidate, "utf8");
  const committed = readFileSync(COMMITTED, "utf8");
  if (fresh !== committed) {
    console.error(
      "openapi drift: src/api/schema.ts does not match docs/api/openapi-v1.json.\n"
        + "Run `npm run api:generate` and commit the result.",
    );
    process.exit(1);
  }
  console.log("openapi drift: generated bindings match the committed contract");
} finally {
  rmSync(scratch, { recursive: true, force: true });
}
