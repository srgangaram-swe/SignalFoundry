// Flat ESLint configuration. Type-aware rules run against the project's own
// tsconfig so lint sees the same types the build does.
import js from "@eslint/js";
import tseslint from "typescript-eslint";
import reactHooks from "eslint-plugin-react-hooks";

export default tseslint.config(
  { ignores: ["dist", "coverage", "src/api/schema.ts", "playwright-report", "test-results"] },
  js.configs.recommended,
  ...tseslint.configs.strictTypeChecked,
  ...tseslint.configs.stylisticTypeChecked,
  {
    languageOptions: {
      parserOptions: { projectService: true, tsconfigRootDir: import.meta.dirname },
    },
    plugins: { "react-hooks": reactHooks },
    rules: {
      ...reactHooks.configs.recommended.rules,
      // The console renders evidence as text. Any DOM sink that accepts markup
      // is a way for stored text to become script, so they are refused here as
      // well as by the content-security policy.
      "no-restricted-properties": [
        "error",
        { object: "document", property: "write", message: "The console renders text as text." },
        { property: "innerHTML", message: "Use text nodes; server text is never markup." },
        { property: "outerHTML", message: "Use text nodes; server text is never markup." },
      ],
      "no-restricted-globals": [
        "error",
        { name: "fetch", message: "Use the bounded transport in src/api/client.ts." },
      ],
      "no-restricted-syntax": [
        "error",
        {
          selector: "JSXAttribute[name.name='dangerouslySetInnerHTML']",
          message: "The console never injects server-provided markup.",
        },
        {
          selector: "NewExpression[callee.name='WebSocket']",
          message: "The console is read-only and polls nothing; no socket is permitted.",
        },
        {
          selector: "CallExpression[callee.object.name='navigator'][callee.property.name='sendBeacon']",
          message: "No telemetry leaves the console.",
        },
        {
          selector: "MemberExpression[object.name='localStorage']",
          message: "The console persists nothing in the browser.",
        },
        {
          selector: "MemberExpression[object.name='sessionStorage']",
          message: "The console persists nothing in the browser.",
        },
      ],
      "@typescript-eslint/no-non-null-assertion": "error",
      "@typescript-eslint/consistent-type-imports": "error",
    },
  },
  {
    // Build scripts and flat-config files are plain modules outside the app's
    // TypeScript project, so type-aware rules cannot apply to them.
    files: ["scripts/**/*.mjs", "eslint.config.js"],
    extends: [tseslint.configs.disableTypeChecked],
    languageOptions: { parserOptions: { projectService: false } },
    rules: { "no-undef": "off", "no-restricted-globals": "off" },
  },
  {
    files: ["tests/**", "e2e/**", "scripts/**"],
    rules: {
      "@typescript-eslint/no-non-null-assertion": "off",
      // Test doubles for `fetch` must return a promise to match the signature
      // they stand in for, so an async function with no await is correct here.
      // The rule stays on for `src/`, where it does catch real mistakes.
      "@typescript-eslint/require-await": "off",
      "no-restricted-globals": "off",
    },
  },
);
