/**
 * Routing shell.
 *
 * Two accessibility behaviours live here because they are properties of
 * navigation rather than of any one view:
 *
 * - A skip link precedes the navigation so a keyboard user reaches content in
 *   one keystroke.
 * - On every route change, focus moves to the main region's heading and the new
 *   view is announced. Without this, a single-page navigation leaves a screen
 *   reader user on a stale element with no indication anything happened.
 */
import { Suspense, lazy, useEffect, useRef } from "react";
import { NavLink, Route, Routes, useLocation } from "react-router";

import { StateBanner } from "./components/StateBanner";
// The landing view is bundled with the shell so the first paint needs one
// request. Every secondary view is a separate chunk, which is what keeps the
// initial JavaScript inside its declared budget as views are added.
import { SystemOverview } from "./routes/SystemOverview";

const RunCatalog = lazy(async () => ({ default: (await import("./routes/RunCatalog")).RunCatalog }));
const RunEvidence = lazy(async () => ({
  default: (await import("./routes/RunEvidence")).RunEvidence,
}));
const ModelComparison = lazy(async () => ({
  default: (await import("./routes/ModelComparison")).ModelComparison,
}));
const CalibrationUncertainty = lazy(async () => ({
  default: (await import("./routes/CalibrationUncertainty")).CalibrationUncertainty,
}));
const DriftLatencyOperations = lazy(async () => ({
  default: (await import("./routes/DriftLatencyOperations")).DriftLatencyOperations,
}));
const GovernanceReadiness = lazy(async () => ({
  default: (await import("./routes/GovernanceReadiness")).GovernanceReadiness,
}));

/** The seven views. There is deliberately no eighth. */
export const CONSOLE_VIEWS = [
  { path: "/console", label: "System overview", end: true },
  { path: "/console/runs", label: "Run catalog", end: false },
  { path: "/console/evidence", label: "Run evidence", end: false },
  { path: "/console/comparison", label: "Model comparison", end: false },
  { path: "/console/calibration", label: "Calibration and uncertainty", end: false },
  { path: "/console/operations", label: "Drift, latency and operations", end: false },
  { path: "/console/governance", label: "Governance and readiness", end: false },
] as const;

export function App(): React.JSX.Element {
  const location = useLocation();
  const headingRef = useRef<HTMLHeadingElement>(null);

  useEffect(() => {
    // Focus without scrolling: moving the viewport as well would disorient a
    // sighted keyboard user who is already looking at the right place.
    headingRef.current?.focus({ preventScroll: true });
  }, [location.pathname]);

  const current =
    CONSOLE_VIEWS.find((view) =>
      view.end ? location.pathname === view.path : location.pathname.startsWith(view.path),
    ) ?? CONSOLE_VIEWS[0];

  return (
    <>
      <a className="skip-link" href="#main">
        Skip to main content
      </a>
      <div className="layout">
        <header className="sidebar">
          <h1>Signalattice evidence console</h1>
          <p className="detail">
            Read-only local evidence. This console cannot approve, promote, roll back, trade, or
            change any recorded result.
          </p>
          <nav aria-label="Console views">
            <ul>
              {CONSOLE_VIEWS.map((view) => (
                <li key={view.path}>
                  <NavLink to={view.path} end={view.end}>
                    {view.label}
                  </NavLink>
                </li>
              ))}
            </ul>
          </nav>
        </header>
        <main className="content" id="main">
          <h2 ref={headingRef} tabIndex={-1}>
            {current.label}
          </h2>
          <Suspense
            fallback={
              <StateBanner state="LOADING" detail="Loading this view's code." subject="View" />
            }
          >
            <Routes>
              <Route path="/console" element={<SystemOverview />} />
              <Route path="/console/runs" element={<RunCatalog />} />
              <Route path="/console/evidence" element={<RunEvidence />} />
              <Route path="/console/comparison" element={<ModelComparison />} />
              <Route path="/console/calibration" element={<CalibrationUncertainty />} />
              <Route path="/console/operations" element={<DriftLatencyOperations />} />
              <Route path="/console/governance" element={<GovernanceReadiness />} />
              <Route
                path="*"
                element={
                  <section className="panel" aria-label="Unknown view">
                    <h3>No such view</h3>
                    <p className="detail">
                      This console has exactly seven views. Use the navigation to reach one.
                    </p>
                  </section>
                }
              />
            </Routes>
          </Suspense>
        </main>
      </div>
    </>
  );
}
