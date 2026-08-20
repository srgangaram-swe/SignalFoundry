/**
 * A titled region that renders either its evidence or the reason it cannot.
 *
 * The rule is: **render whatever evidence exists, and always say what is wrong
 * with it.** A state that carries data renders both the banner and the data,
 * because the numbers are real and hiding them loses information a reader
 * needs -- an `INSUFFICIENT_EVIDENCE` panel that showed nothing would be
 * indistinguishable from an empty one, and an `INVALID` governance lane that
 * vanished would hide precisely the lane worth investigating.
 *
 * States that genuinely have nothing to show (`LOADING`, `EMPTY`,
 * `UNAVAILABLE`, `ERROR`) carry `null` data and therefore render the banner
 * alone. The panel does not need to special-case them.
 */
import type { EvidenceStatus } from "../state/evidenceState";
import { StateBanner } from "./StateBanner";

export interface EvidencePanelProps<T> {
  readonly title: string;
  readonly status: EvidenceStatus<T>;
  readonly headingLevel?: 2 | 3;
  readonly children: (data: T) => React.ReactNode;
}

export function EvidencePanel<T>({
  title,
  status,
  headingLevel = 2,
  children,
}: EvidencePanelProps<T>): React.JSX.Element {
  const Heading = headingLevel === 2 ? "h2" : "h3";
  const data = status.data;
  return (
    <section className="panel" aria-label={title}>
      <Heading>{title}</Heading>
      {status.state === "READY" ? null : (
        <StateBanner state={status.state} detail={status.detail} subject={title} />
      )}
      {data === null ? null : children(data)}
    </section>
  );
}
