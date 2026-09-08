/** Parse editable configuration before invoking the authoritative preflight. */
import { ApiError, finiteTree, REQUEST_BYTES } from "./api";
import { isResearchRequest } from "./generated/validators.cjs";
import type { ResearchRequest } from "./types";

export function parseConfiguration(text: string): ResearchRequest {
  if (new TextEncoder().encode(text).byteLength > REQUEST_BYTES)
    throw new ApiError("request_size", "Configuration exceeds 16 KiB.");
  let value: unknown;
  try {
    value = JSON.parse(text);
  } catch (cause) {
    throw new ApiError(
      "invalid_configuration",
      "Enter a valid JSON configuration.",
      { cause },
    );
  }
  finiteTree(value);
  if (!isResearchRequest(value))
    throw new ApiError(
      "invalid_configuration",
      "Configuration violates the API schema. Check field names, types and bounds.",
    );
  return value;
}

export function message(error: unknown): string {
  return error instanceof ApiError
    ? error.message
    : "The local operation failed. Inspect the service diagnostics and retry.";
}
