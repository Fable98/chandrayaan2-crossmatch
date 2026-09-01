"use client";

import dynamic from "next/dynamic";

// Leaflet reaches for `window` at import time, so it can't be part of the
// server-rendered bundle — ssr:false keeps it out of the initial HTML and
// loads it only in the browser.
const MapPanel = dynamic(() => import("./MapPanel"), {
  ssr: false,
  loading: () => (
    <div className="flex h-full items-center justify-center text-2xs font-mono text-ink-faint">
      Loading map…
    </div>
  ),
});

export default MapPanel;
