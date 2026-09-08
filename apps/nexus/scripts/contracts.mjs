/** Generate CSP-safe validators and form bounds from the sole API schema. */
import { readFileSync, mkdirSync, writeFileSync } from "node:fs";
import { resolve } from "node:path";
import Ajv2020 from "ajv/dist/2020.js";
import standalone from "ajv/dist/standalone/index.js";

const root = resolve(import.meta.dirname, "../../..");
const document = JSON.parse(
  readFileSync(resolve(root, "contracts/openapi-v1.json"), "utf8"),
);
const names = [
  "Catalog",
  "ResearchRequest",
  "Validation",
  "Job",
  "JobPage",
  "ResearchEvidence",
  "Comparison",
  "AuditTrail",
  "Problem",
];

function references(value) {
  if (Array.isArray(value)) return value.map(references);
  if (value && typeof value === "object") {
    const resolved = Object.fromEntries(
      Object.entries(value).map(([key, item]) => [
        key,
        key === "$ref"
          ? item.replace("#/components/schemas/", "#/$defs/")
          : references(item),
      ]),
    );
    // The server emits fully resolved models. Require their declared properties
    // rather than silently inventing omitted response defaults in the browser.
    if (value.properties) resolved.required = Object.keys(value.properties);
    return resolved;
  }
  return value;
}

const schema = {
  $id: "urn:signal-foundry:research:v1",
  $defs: references(document.components.schemas),
};
const ajv = new Ajv2020({
  strict: true,
  allErrors: false,
  ownProperties: true,
  inlineRefs: false,
  // CommonJS lets the bundler resolve Ajv's small Unicode-length runtime helper
  // without leaving a Node `require` inside an otherwise-ESM browser module.
  code: { source: true, esm: false, optimize: true },
});
ajv.addSchema(schema);
const exports = Object.fromEntries(
  names.map((name) => ["is" + name, schema.$id + "#/$defs/" + name]),
);
const validators = standalone(ajv, exports);
const declarations = [
  "// Generated from contracts/openapi-v1.json; do not hand-edit.",
  'import type { components } from "../../../../contracts/research-v1";',
  'import type { Resolved } from "../types";',
  ...names.map(
    (name) =>
      `export declare function is${name}(value: unknown): value is Resolved<components["schemas"]["${name}"]>;`,
  ),
  "",
].join("\n");
const fields = Object.fromEntries(
  [
    "DataChoice",
    "FoldPolicy",
    "CostPolicy",
    "RiskPolicy",
    "ResearchRequest",
  ].map((name) => [
    name,
    Object.fromEntries(
      Object.entries(document.components.schemas[name].properties)
        .filter(([, item]) =>
          ["integer", "number", "boolean"].includes(item.type),
        )
        .map(([key, item]) => [
          key,
          Object.fromEntries(
            Object.entries(item).filter(([field]) =>
              [
                "type",
                "minimum",
                "maximum",
                "exclusiveMinimum",
                "default",
              ].includes(field),
            ),
          ),
        ]),
    ),
  ]),
);
const output = resolve(root, "apps/nexus/src/generated");
mkdirSync(output, { recursive: true });
for (const [name, content] of [
  ["validators.cjs", validators + "\n"],
  ["validators.d.cts", declarations],
  ["fields.json", JSON.stringify(fields) + "\n"],
]) {
  const path = resolve(output, name);
  if (process.argv.includes("--check")) {
    if (readFileSync(path, "utf8") !== content)
      throw new Error("Generated contract drift: " + name);
  } else {
    writeFileSync(path, content);
  }
}
