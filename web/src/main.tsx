/** DOM bootstrap. Kept trivial; behaviour lives in App and the views. */
import { StrictMode } from "react";
import { createRoot } from "react-dom/client";
import { BrowserRouter } from "react-router";

import { App } from "./App";
import "./styles.css";

const container = document.getElementById("root");
if (container === null) {
  throw new Error("console root element is missing");
}

createRoot(container).render(
  <StrictMode>
    <BrowserRouter>
      <App />
    </BrowserRouter>
  </StrictMode>,
);
