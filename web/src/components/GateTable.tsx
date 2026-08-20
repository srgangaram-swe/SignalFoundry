/**
 * Absolute gate outcomes as a semantic table.
 *
 * Every gate is listed, satisfied or not. Filtering to failures would hide how
 * much was checked, and filtering to passes would be a lie of omission; both
 * matter to a reader deciding whether a recommendation is trustworthy.
 */
import type { GateResult } from "../api/decoders";
import { ScrollableTable } from "./ScrollableTable";

export function GateTable({
  gates,
  truncated,
}: {
  readonly gates: readonly GateResult[];
  readonly truncated: boolean;
}): React.JSX.Element {
  const failed = gates.filter((gate) => !gate.satisfied).length;
  return (
    <ScrollableTable label="Absolute gate outcomes">
      <table>
        <caption>
          {`${String(gates.length)} gates evaluated, ${String(failed)} failed.`}
          {truncated ? " Additional gates exist but were not returned." : ""}
          {" Gates are absolute: no gate can be traded off against another."}
        </caption>
        <thead>
          <tr>
            <th scope="col">Gate</th>
            <th scope="col">Outcome</th>
            <th scope="col">Detail</th>
          </tr>
        </thead>
        <tbody>
          {gates.map((gate) => (
            <tr key={gate.name}>
              <th scope="row">{gate.name}</th>
              <td>
                <span className="status" data-tone={gate.satisfied ? "ready" : "fail"}>
                  <span className="glyph" aria-hidden="true">
                    {gate.satisfied ? "✓" : "✕"}
                  </span>
                  <span>{gate.satisfied ? "Satisfied" : "Failed"}</span>
                </span>
              </td>
              <td>{gate.detail}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </ScrollableTable>
  );
}
