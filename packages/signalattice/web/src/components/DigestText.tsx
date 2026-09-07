/**
 * Renders a content digest in an abbreviated but recoverable form.
 *
 * The short form is what a human compares; the full value stays available to
 * assistive technology and to copy, so abbreviating never loses the evidence.
 */
export function DigestText({ value }: { readonly value: string }): React.JSX.Element {
  return (
    <span className="digest" title={value}>
      <span aria-hidden="true">{value.slice(0, 12)}…</span>
      <span className="visually-hidden">{value}</span>
    </span>
  );
}
