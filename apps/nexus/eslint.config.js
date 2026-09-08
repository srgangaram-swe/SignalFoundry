import js from "@eslint/js";
import hooks from "eslint-plugin-react-hooks";
import ts from "typescript-eslint";

export default ts.config(
  {
    ignores: [
      "dist/**",
      "coverage/**",
      "src/generated/**",
      "test-results/**",
      "playwright-report/**",
    ],
  },
  js.configs.recommended,
  ...ts.configs.strictTypeChecked,
  ...ts.configs.stylisticTypeChecked,
  {
    languageOptions: {
      parserOptions: {
        projectService: true,
        tsconfigRootDir: import.meta.dirname,
      },
    },
    plugins: { "react-hooks": hooks },
    rules: {
      ...hooks.configs.recommended.rules,
      "@typescript-eslint/consistent-type-imports": "error",
      "@typescript-eslint/no-non-null-assertion": "error",
      "no-restricted-properties": [
        "error",
        { property: "innerHTML" },
        { property: "outerHTML" },
        { object: "document", property: "write" },
      ],
      "no-restricted-syntax": [
        "error",
        {
          selector: "JSXAttribute[name.name='dangerouslySetInnerHTML']",
          message: "Evidence is text, never markup.",
        },
        {
          selector: "NewExpression[callee.name='WebSocket']",
          message: "Use the bounded HTTP client.",
        },
      ],
    },
  },
  {
    files: ["scripts/**/*.mjs", "eslint.config.js"],
    extends: [ts.configs.disableTypeChecked],
    languageOptions: {
      parserOptions: { projectService: false },
      globals: {
        process: "readonly",
        console: "readonly",
        Buffer: "readonly",
        URL: "readonly",
      },
    },
  },
  {
    files: ["tests/**", "e2e/**"],
    rules: { "@typescript-eslint/require-await": "off" },
  },
);
