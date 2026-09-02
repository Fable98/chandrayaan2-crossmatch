# SIH26166 — Lunar Correspondence Console

Next.js frontend for the OHRC/TMC-2/IIRS registration backend. Consumes
`/triplets`, `/triplets/{id}`, `/triplets/{id}/matches`,
`/triplets/{id}/iirs-overlay`, and `/images/{sensor}/{tile_id}` exactly as
built and tested on the backend side — the types in `src/lib/types.ts`
mirror `backend/schemas.py` field-for-field.

## Setup

```bash
npm install
cp .env.local.example .env.local   # point at your running FastAPI instance
npm run dev
```

Open http://localhost:3000. Your FastAPI backend needs to be running
separately (`uvicorn main:app --reload`) and needs CORS configured to allow
`http://localhost:3000` as an origin — the backend's CORS setup already
does this by default per the earlier integration work, but confirm before
assuming.

## What's built

- **Region list** (left sidebar): pulls `/triplets`, shows footprint size
  in km computed the same way the backend's own sanity checks did
  (`R_moon = 1737.4 km`), flags which regions have DEM available.
- **Fused map** tab: Leaflet map showing the shared footprint rectangle
  (all sensors share one bbox by pipeline design — see backend notes) with
  toggleable IIRS mineralogy and DEM elevation overlays via
  `L.imageOverlay`. Longitude values in the real processed data use the
  standard 0–360° lunar convention; the current region_005 / region_006
  footprints stay in the 234° range rather than crossing 180°.
- **Linked cursor** tab: click a point on the OHRC tile, and the nearest
  LoFTR+RANSAC match (by pixel distance, within a 40px threshold) is
  highlighted on the corresponding TMC-2 tile, with its confidence score.
  This is the core "click a crater, see it match" demo interaction.

## Known gaps / next steps — be upfront about these, don't paper over them

- **No basemap tile layer.** There's no verified public lunar WMTS tile
  source wired in, so the map background is empty except for the
  footprint rectangle and overlays. This is a deliberate choice over
  faking a tile URL that might 404 or silently show Earth/Mars tiles
  under lunar data — swap in a real lunar tile source if the team finds
  one and wants basemap context for the demo.
- **512×512 tile size is hardcoded** (`TILE_PX` in
  `LinkedCursorPanel.tsx`) based on the pipeline's Stage 4 grid
  standardization. If any region ever isn't exactly 512×512, this needs
  to become per-triplet data instead of a constant.
- **Nearest-match threshold (40px) is a starting guess**, not tuned
  against real match density. Since matches are sparse by design, this
  may need adjusting once you've clicked around on real data — if
  clicks feel like they're missing obviously-nearby matches, raise it;
  if they're matching to something clearly unrelated, lower it.
- **No loading skeleton for images** — a missing/renamed tile file fails
  silently (image just doesn't render) rather than showing an error
  state. Fine for a demo where you control the data, worth hardening if
  this goes beyond the hackathon.
- **`npm audit` reports 2 high-severity advisories** in Next.js's bundled
  PostCSS (a source-map path traversal issue). Only exploitable at build
  time with attacker-controlled source maps — low real risk for a local
  hackathon build, but worth `npm audit fix --force` (a breaking Next.js
  15/16 upgrade) if this goes further than the demo.

## Design notes

Dark "mission console" theme (deep space background, amber/regolith
accent for OHRC-side data, teal/parallax accent for TMC-side and layer
toggles) rather than a generic SaaS dashboard look — deliberately avoids
the warm-cream-plus-terracotta and rounded-card-with-shadow defaults.
IBM Plex Sans for UI text, IBM Plex Mono for coordinates, product IDs,
and anything tabular — the numeric data is the actual content here, so it
gets a typeface built for it.
