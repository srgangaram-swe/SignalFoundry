import { useId } from "react";
import type { EvidenceTable } from "./types";

/** O(retained rows), preserves gaps; the shared scale always includes zero. */
export function chartSeries(tables: readonly EvidenceTable[], metric: string) {
  const series = tables
    .filter((table) => table.name.startsWith("equity_"))
    .map((table) => {
      const column = table.columns.findIndex((item) => item.name === metric);
      return {
        name: table.name.slice(7),
        values: table.rows.map((row) => {
          const value = row[column];
          return typeof value === "number" ? value : null;
        }),
        first: String(table.rows[0]?.[0] ?? "unavailable"),
        last: String(table.rows.at(-1)?.[0] ?? "unavailable"),
      };
    });
  let low = 0;
  let high = 0;
  for (const item of series)
    for (const value of item.values)
      if (value !== null) {
        low = Math.min(low, value);
        high = Math.max(high, value);
      }
  const span = high - low || 1;
  const y = (value: number) => 200 - ((value - low) / span) * 160;
  return {
    low,
    high,
    zero: y(0),
    series: series.map((item) => {
      let connected = false;
      const path = item.values
        .map((value, index) => {
          if (value === null) {
            connected = false;
            return "";
          }
          const prefix = connected ? "L" : "M";
          connected = true;
          return `${prefix}${String(70 + (index / Math.max(1, item.values.length - 1)) * 500)},${String(y(value))}`;
        })
        .join(" ");
      return { ...item, path };
    }),
  };
}

export function EquityChart({
  tables,
  metric,
  title,
}: {
  readonly tables: readonly EvidenceTable[];
  readonly metric: string;
  readonly title: string;
}) {
  const id = useId();
  const chart = chartSeries(tables, metric);
  if (!chart.series.length) return <p>{title}: unavailable.</p>;
  return (
    <figure className="chart">
      <figcaption>{title} · return fraction</figcaption>
      <svg viewBox="0 0 620 250" role="img" aria-labelledby={id}>
        <title id={id}>
          {title}. Simulated strategy and baselines; exact values follow in the
          evidence tables. Missing observations break each line.
        </title>
        <line
          x1="70"
          x2="570"
          y1={chart.zero}
          y2={chart.zero}
          className="zero-line"
        />
        <text x="5" y="45">
          {chart.high.toPrecision(3)}
        </text>
        <text x="5" y="200">
          {chart.low.toPrecision(3)}
        </text>
        {chart.series.map((item, index) => (
          <path
            key={item.name}
            d={item.path}
            className={`series series-${String(index % 4)}`}
          />
        ))}
        <text x="70" y="235">
          {chart.series[0]?.first}
        </text>
        <text x="465" y="235">
          {chart.series[0]?.last}
        </text>
      </svg>
      <ul className="chart-legend">
        {chart.series.map((item, index) => (
          <li key={item.name}>
            <svg className="swatch" viewBox="0 0 26 8" aria-hidden="true">
              <path
                d="M0,4 H26"
                className={`series series-${String(index % 4)}`}
              />
            </svg>
            {item.name} · {item.values.length} retained sessions
          </li>
        ))}
      </ul>
      <p>
        All retained observations; zero reference included. Line patterns
        identify models. No interpolation across missing values.
      </p>
    </figure>
  );
}
