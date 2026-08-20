/**
 * A confidence interval drawn as a bar, always beside its own numbers.
 *
 * The bar is decoration: every value it encodes is printed as text in the same
 * row, so the evidence survives a screen reader, a forced-colors theme, and a
 * failure to paint. A canvas-only interval would make the finding unreadable
 * for anyone not looking at pixels.
 *
 * A one-sided interval is drawn to the edge of the track and labelled
 * "unbounded", never closed at an arbitrary number.
 */
export interface IntervalBarProps {
  readonly low: number | null;
  readonly high: number | null;
  readonly point: number | null;
  /** Symmetric display domain; values outside are clamped and labelled. */
  readonly domain: number;
  readonly label: string;
}

function formatBound(value: number | null, fallback: string): string {
  return value === null ? fallback : value.toFixed(4);
}

export function IntervalBar({
  low,
  high,
  point,
  domain,
  label,
}: IntervalBarProps): React.JSX.Element {
  const span = domain <= 0 ? 1 : domain;
  const toPercent = (value: number): number =>
    Math.min(100, Math.max(0, ((value + span) / (2 * span)) * 100));

  const left = low === null ? 0 : toPercent(low);
  const right = high === null ? 100 : toPercent(high);
  const width = Math.max(1, right - left);

  const text = `${formatBound(low, "unbounded")} to ${formatBound(high, "unbounded")}`;
  return (
    <div className="interval">
      <div
        className="interval-track"
        role="img"
        aria-label={`${label}: interval ${text}`}
        data-testid="interval-track"
      >
        <span className="interval-zero" style={{ left: "50%" }} aria-hidden="true" />
        <span
          className="interval-span"
          style={{ left: `${String(left)}%`, width: `${String(width)}%` }}
          aria-hidden="true"
        />
      </div>
      <span className="numeric">
        {text}
        {point === null ? null : ` (point ${point.toFixed(4)})`}
      </span>
    </div>
  );
}
