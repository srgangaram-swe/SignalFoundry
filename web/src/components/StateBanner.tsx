/**
 * Renders one honest-state outcome.
 *
 * Status is carried three ways -- a word, a glyph, and a colour -- so no single
 * channel is load-bearing. A reader using a screen reader hears the word, a
 * reader with a forced high-contrast theme sees the glyph and border, and a
 * reader with colour vision differences distinguishes the shapes.
 *
 * The live region is `polite` and its content is the *detail*, not the state
 * name alone. Announcing "error" without saying what failed makes a screen
 * reader user open the JSON API to find out.
 */
import type { EvidenceState } from "../state/evidenceState";

interface Tone {
  readonly tone: "ready" | "warn" | "fail" | "info" | "muted";
  readonly glyph: string;
  readonly label: string;
}

/**
 * Presentation for each state.
 *
 * `INSUFFICIENT_EVIDENCE`, `INVALID`, `STALE`, and `UNAVAILABLE` are given
 * visibly different glyphs and labels because conflating them is precisely the
 * failure this console is built to avoid.
 */
const TONES: Readonly<Record<EvidenceState, Tone>> = {
  LOADING: { tone: "muted", glyph: "…", label: "Loading" },
  READY: { tone: "ready", glyph: "✓", label: "Ready" },
  EMPTY: { tone: "muted", glyph: "∅", label: "No evidence recorded" },
  PARTIAL: { tone: "warn", glyph: "◐", label: "Partial evidence" },
  INSUFFICIENT_EVIDENCE: { tone: "warn", glyph: "⚖", label: "Insufficient evidence" },
  INVALID: { tone: "fail", glyph: "✕", label: "Invalid evidence" },
  STALE: { tone: "warn", glyph: "⏱", label: "Stale evidence" },
  UNAVAILABLE: { tone: "fail", glyph: "⚠", label: "Service unavailable" },
  ERROR: { tone: "fail", glyph: "!", label: "Incompatible evidence" },
};

export interface StateBannerProps {
  readonly state: EvidenceState;
  readonly detail: string;
  /** Optional heading context, e.g. the panel this state belongs to. */
  readonly subject?: string;
}

export function StateBanner({ state, detail, subject }: StateBannerProps): React.JSX.Element {
  const tone = TONES[state];
  return (
    <div
      className="state-banner"
      // `status` rather than `alert`: an alert interrupts, and a console that
      // interrupts on every panel transition is unusable with a screen reader.
      role="status"
      aria-live="polite"
      data-state={state}
      data-testid="state-banner"
    >
      <span className="status" data-tone={tone.tone}>
        <span className="glyph" aria-hidden="true">
          {tone.glyph}
        </span>
        <span>{tone.label}</span>
      </span>
      <p className="detail">
        {subject === undefined ? null : <span className="visually-hidden">{subject}: </span>}
        {detail}
      </p>
    </div>
  );
}

/** Exposed so tests can assert the model and the presentation stay in step. */
export function toneFor(state: EvidenceState): Tone {
  return TONES[state];
}
