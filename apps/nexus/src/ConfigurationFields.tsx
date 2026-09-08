import fields from "./generated/fields.json";
import type { ResearchRequest } from "./types";

const GROUPS = [
  ["data", "DataChoice", "Synthetic panel"],
  ["folds", "FoldPolicy", "Walk-forward validation"],
  ["costs", "CostPolicy", "Execution and financing costs"],
  ["risk", "RiskPolicy", "Portfolio risk limits"],
] as const;

/** Schema-derived bounds; server validation remains required before submission. */
export function ConfigurationFields({
  request,
  change,
}: {
  readonly request: ResearchRequest;
  readonly change: (text: string) => void;
}) {
  return (
    <div className="configuration-fields">
      {GROUPS.map(([group, schema, title]) => (
        <details key={group}>
          <summary>{title}</summary>
          {Object.entries(fields[schema]).map(([name, definition]) => {
            const current: unknown = Object.entries(request[group]).find(
              ([key]) => key === name,
            )?.[1];
            const label =
              name.replaceAll("_", " ") +
              (name.includes("bps")
                ? " (basis points)"
                : name.includes("days") ||
                    name === "horizon" ||
                    name === "execution_lag"
                  ? " (sessions)"
                  : "");
            function update(value: boolean | number) {
              change(
                JSON.stringify(
                  { ...request, [group]: { ...request[group], [name]: value } },
                  null,
                  2,
                ),
              );
            }
            return (
              <label key={name}>
                {label}
                {typeof current === "boolean" ? (
                  <input
                    type="checkbox"
                    checked={current}
                    onChange={(event) => {
                      update(event.target.checked);
                    }}
                  />
                ) : (
                  <input
                    type="number"
                    value={typeof current === "number" ? current : ""}
                    min={
                      "minimum" in definition ? definition.minimum : undefined
                    }
                    max={
                      "maximum" in definition ? definition.maximum : undefined
                    }
                    step={definition.type === "integer" ? 1 : "any"}
                    onChange={(event) => {
                      update(event.target.valueAsNumber);
                    }}
                  />
                )}
              </label>
            );
          })}
        </details>
      ))}
    </div>
  );
}
