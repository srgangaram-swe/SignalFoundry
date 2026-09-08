import { useState } from "react";
import type { Evidence, EvidenceTable } from "./types";
import { EquityChart } from "./EquityChart";

const PAGE_ROWS = 40;

/** Bounded window: at most 40 retained records enter the accessible DOM. */
export function TableView({ table }: { readonly table: EvidenceTable }) {
  const [page, setPage] = useState(0);
  const start = Math.min(page * PAGE_ROWS, Math.max(0, table.rows.length - 1));
  const rows = table.rows.slice(start, start + PAGE_ROWS);
  return (
    <section className="evidence-table" aria-label={table.name}>
      <h3>{table.name.replaceAll("_", " ")}</h3>
      <p>{table.description}</p>
      <p className="muted">
        {table.rows.length} retained / {table.total_rows} total records. Missing
        values remain unavailable.
      </p>
      <div
        className="table-scroll"
        tabIndex={0}
        role="region"
        aria-label={`${table.name} records`}
      >
        <table aria-rowcount={table.rows.length + 1}>
          <thead>
            <tr>
              {table.columns.map((column) => (
                <th key={column.name} scope="col">
                  {column.name}
                  <small>{column.unit}</small>
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {rows.map((row, index) => (
              <tr key={start + index} aria-rowindex={start + index + 2}>
                {row.map((value, cell) => (
                  <td key={cell}>
                    {value === null ? "Unavailable" : String(value)}
                  </td>
                ))}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      {rows.length === 0 && <p>No records available.</p>}
      <div className="actions">
        <button
          disabled={page === 0}
          onClick={() => {
            setPage(page - 1);
          }}
        >
          Previous rows
        </button>
        <span role="status">
          Rows {rows.length ? start + 1 : 0}–{start + rows.length}
        </span>
        <button
          disabled={start + PAGE_ROWS >= table.rows.length}
          onClick={() => {
            setPage(page + 1);
          }}
        >
          Next rows
        </button>
      </div>
    </section>
  );
}

export function EvidenceView({ evidence }: { readonly evidence: Evidence }) {
  return (
    <article>
      <h2>Inspect evidence</h2>
      <p className="badge">Development simulation · NOT_READY</p>
      <dl>
        <dt>Data identity</dt>
        <dd>{evidence.data_identity}</dd>
        <dt>Request hash · verified by server</dt>
        <dd>{evidence.request_hash}</dd>
        <dt>Source code hash</dt>
        <dd>{evidence.source_code_hash}</dd>
        <dt>Environment hash</dt>
        <dd>{evidence.environment_hash}</dd>
      </dl>
      <ul>
        {evidence.limitations.map((item) => (
          <li key={item}>{item}</li>
        ))}
      </ul>
      <p>
        Calibration is unavailable for regression-only artifacts. Capacity
        proxies and simulated costs do not establish executable liquidity or
        profitable trading.
      </p>
      <EquityChart
        tables={evidence.tables}
        metric="cumulative_return"
        title="Net cumulative return"
      />
      <EquityChart
        tables={evidence.tables}
        metric="drawdown"
        title="Drawdown"
      />
      {evidence.tables.map((table) => (
        <TableView key={evidence.request_hash + table.name} table={table} />
      ))}
    </article>
  );
}
