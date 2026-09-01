import Console from "@/components/Console";

// This is an interactive console reading live data from a local FastAPI
// backend — there's nothing to statically prerender, and Leaflet's
// import graph touches `window` at module-eval time, which breaks static
// export. Render per-request instead.
export const dynamic = "force-dynamic";

export default function Home() {
  return <Console />;
}
