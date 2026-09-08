/** Pin a bounded build inventory; no environment, private state or raw data. */
import { createHash } from "node:crypto";
import { lstatSync, readdirSync, readFileSync, writeFileSync } from "node:fs";
import { resolve, relative, extname } from "node:path";
import { gzipSync } from "node:zlib";

const root = resolve(import.meta.dirname, "..");
const dist = resolve(root, "dist");
const digest = (bytes) => createHash("sha256").update(bytes).digest("hex");
const allowed =
  /^(index\.html|assets\/[A-Za-z0-9_-]{1,120}\.(js|css|woff2|svg))$/;

function files(directory, depth = 0) {
  if (depth > 4) throw new Error("Build inventory depth exceeded");
  const entries = readdirSync(directory).sort();
  if (entries.length > 128) throw new Error("Build inventory count exceeded");
  return entries.flatMap((name) => {
    const path = resolve(directory, name);
    const metadata = lstatSync(path);
    if (metadata.isSymbolicLink())
      throw new Error("Symlink in build inventory");
    if (metadata.isDirectory()) return files(path, depth + 1);
    if (!metadata.isFile() || metadata.size > 1024 * 1024)
      throw new Error("Invalid build inventory file");
    return [path];
  });
}

const assets = files(dist).filter(
  (path) => relative(dist, path) !== "manifest.json",
);
if (assets.length < 3 || assets.length > 64)
  throw new Error("Asset count outside policy");
const totals = { javascript_gzip: 0, css_gzip: 0, total_bytes: 0 };
const inventory = assets.map((path) => {
  const name = relative(dist, path);
  if (!allowed.test(name)) throw new Error("Unexpected emitted asset");
  const bytes = readFileSync(path);
  const suffix = extname(path);
  totals.total_bytes += bytes.length;
  if (suffix === ".js") totals.javascript_gzip += gzipSync(bytes).length;
  if (suffix === ".css") totals.css_gzip += gzipSync(bytes).length;
  return { path: name, bytes: bytes.length, sha256: digest(bytes) };
});
const budgets = {
  javascript_gzip: 250 * 1024,
  css_gzip: 50 * 1024,
  total_bytes: 1024 * 1024,
};
for (const key of Object.keys(budgets)) {
  if (totals[key] > budgets[key])
    throw new Error("Bundle budget exceeded: " + key);
}
const source = createHash("sha256");
for (const path of [
  ...files(resolve(root, "src")),
  ...files(resolve(root, "scripts")),
  resolve(root, "index.html"),
  resolve(root, "vite.config.ts"),
].sort()) {
  source.update(relative(root, path) + "\0");
  source.update(readFileSync(path));
}
const manifest = {
  schema_version: "1.0.0",
  source_sha256: source.digest("hex"),
  lock_sha256: digest(readFileSync(resolve(root, "package-lock.json"))),
  contract_sha256: digest(
    readFileSync(resolve(root, "../../contracts/openapi-v1.json")),
  ),
  assets: inventory,
  measured: totals,
  budgets,
};
const encoded = JSON.stringify(manifest) + "\n";
if (process.argv.includes("--check")) {
  if (readFileSync(resolve(dist, "manifest.json"), "utf8") !== encoded)
    throw new Error("Build manifest drift");
} else {
  writeFileSync(resolve(dist, "manifest.json"), encoded);
}
console.log(JSON.stringify({ measured: totals, budgets }));
