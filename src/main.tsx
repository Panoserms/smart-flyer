import { StrictMode } from "react";
import { createRoot } from "react-dom/client";
import SmartFlyerShowcase from "./components/SmartFlyerShowcase";
import "./index.css";

createRoot(document.getElementById("root")!).render(
  <StrictMode>
    <SmartFlyerShowcase />
  </StrictMode>,
);
