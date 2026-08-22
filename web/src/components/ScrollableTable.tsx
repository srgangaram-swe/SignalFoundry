/**
 * A horizontally scrollable container that a keyboard user can actually scroll.
 *
 * At narrow widths an evidence table overflows and its container scrolls. A
 * scrollable region that is not focusable is unreachable without a pointer, so
 * the container takes `tabindex="0"` and an accessible name. This is the
 * `scrollable-region-focusable` rule, and it is a real barrier rather than a
 * lint detail: on a 320-pixel viewport the overflowing columns are the evidence.
 */
export function ScrollableTable({
  label,
  children,
}: {
  readonly label: string;
  readonly children: React.ReactNode;
}): React.JSX.Element {
  return (
    <div className="table-scroll" tabIndex={0} role="region" aria-label={label}>
      {children}
    </div>
  );
}
