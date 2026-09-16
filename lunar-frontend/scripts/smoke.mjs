/**
 * smoke.mjs — Step 13/14 frontend smoke test (no browser, no backend needed).
 *
 *   npm run smoke   (node --test scripts/smoke.mjs)
 *
 * Contract assertions:
 *   1. Kill-backend => the API layer throws ApiError (never fallback data).
 *   2. imageUrl() maps /dynamic_runs/* onto the backend origin.
 *   3. Single API base: ingest-api shares api.ts's API_BASE.
 *   4. No silent-fallback / fake-auth / fake-coordinate strings in src.
 *   5. Generated contract (backend-types.ts) is imported by the app layer.
 *
 * Pure node, zero dependencies. TS lib files use erasable syntax only, so
 * node type-stripping can import them directly.
 */

import { describe, it } from "node:test";
import assert from "node:assert/strict";
import { readFileSync, existsSync, readdirSync, statSync } from "fs";
import { join, dirname } from "path";
import { fileURLToPath } from "url";

const ROOT = join(dirname(fileURLToPath(import.meta.url)), "..");
const SRC = join(ROOT, "src");

process.env.NEXT_PUBLIC_API_BASE_URL = "http://127.0.0.1:9"; // dead port

// Import the real TS api layer via the tsc transpile helper (no browser).
const { importLib } = await import("./tsimport.mjs");
const api = await importLib("api.ts");

describe("frontend contract smoke", () => {
  it("kill-backend => listTriplets throws ApiError, not fake data", async () => {
    await assert.rejects(api.api.listTriplets(), (err) => {
      assert.equal(err?.name, "ApiError");
      assert.equal(err?.status, 0);
      return true;
    });
  });

  it("kill-backend => getTriplet/getMatches/getIirsOverlay throw ApiError", async () => {
    for (const call of [
      () => api.api.getTriplet("region_001"),
      () => api.api.getMatches("region_001"),
      () => api.api.getIirsOverlay("region_001"),
    ]) {
      await assert.rejects(call, (err) => err?.name === "ApiError");
    }
  });

  it("imageUrl() prefixes /dynamic_runs/* with the backend origin", () => {
    const url = api.imageUrl("/dynamic_runs/abc/output/registered_checkerboard.png");
    assert.ok(url.startsWith("http://127.0.0.1:9/dynamic_runs/"), url);
  });

  it("imageUrl() keeps bundled /images/* relative", () => {
    assert.equal(api.imageUrl("/images/ohrc/region_001"), "/images/ohrc/region_001.png");
  });

  it("single API base shared with ingest-api", async () => {
    const src = readFileSync(join(SRC, "lib", "ingest-api.ts"), "utf-8");
    assert.ok(src.includes("from './api'"), "ingest-api must import API_BASE from ./api");
    assert.ok(!src.includes("localhost:8000"), "no second hardcoded backend origin");
    assert.ok(src.includes("getAuthHeaders"), "ingest calls must send auth");
  });

  it("no silent-fallback or fake-auth strings in src", async () => {
    const banned = [
      "fallbackData",
      "FALLBACK_TRIPLETS",
      "FALLBACK_MATCHES",
      "offline_jwt_",
      "demo_jwt_",
      "loginAsDemo",
      "Quick Demo Access",
    ];
    const offenders = [];
    const scan = (dir) => {
      for (const name of readdirSync(dir)) {
        const p = join(dir, name);
        if (statSync(p).isDirectory()) {
          scan(p);
          continue;
        }
        if (!/\.(ts|tsx)$/.test(name)) continue;
        const text = readFileSync(p, "utf-8");
        for (const b of banned) {
          if (text.includes(b)) offenders.push(`${p}: ${b}`);
        }
      }
    };
    scan(SRC);
    assert.deepEqual(offenders, []);
  });

  it("generated contract exists and is consumed by types.ts", () => {
    const gen = join(SRC, "lib", "backend-types.ts");
    assert.ok(existsSync(gen), "run npm run gen:api");
    const types = readFileSync(join(SRC, "lib", "types.ts"), "utf-8");
    assert.ok(types.includes("./backend-types"), "types.ts must re-export the generated contract");
  });

  it("no demo-patch coordinates in backend registration router", () => {
    const backend = join(ROOT, "..", "backend", "routers", "registration.py");
    const text = readFileSync(backend, "utf-8");
    assert.ok(!text.includes("336.0 +"), "demo lat/lon patch must stay deleted");
    assert.ok(text.includes("georeferenced"), "no-georef flag must be served");
  });

  it("traffic-light badge exists, is null-safe, and is rendered", () => {
    const badge = join(SRC, "components", "TrafficLightBadge.tsx");
    assert.ok(existsSync(badge), "TrafficLightBadge.tsx must exist");
    const text = readFileSync(badge, "utf-8");
    for (const label of ["Photogrammetric Grade", "Acceptable / Review", "Rejected / Low Confidence"]) {
      assert.ok(text.includes(label), `badge must label verdicts (${label})`);
    }
    assert.ok(text.includes("Unverified"), "badge must handle null/unknown color");
    assert.ok(text.includes("confidence_score"), "badge must accept confidence_score");
    assert.ok(text.includes("ssim_score"), "badge must accept ssim_score");
    for (const host of ["components/RegistrationLauncher.tsx", "components/ingest/ResultsTable.tsx"]) {
      const hostText = readFileSync(join(SRC, host), "utf-8");
      assert.ok(hostText.includes("TrafficLightBadge"), `${host} must render the badge`);
    }
  });
});

describe("phase 4 hardening", () => {
  it("API_BASE treats empty-string env as unset (||, not ??)", () => {
    const text = readFileSync(join(SRC, "lib", "api.ts"), "utf-8");
    assert.ok(!text.includes('NEXT_PUBLIC_API_BASE_URL ??'), "empty-string env must fall back (?? keeps '')");
    assert.ok(text.includes('NEXT_PUBLIC_API_BASE_URL ||'), "must use || for the base URL");
  });

  it("api layer has a GET timeout budget and a 401 branch", async () => {
    const text = readFileSync(join(SRC, "lib", "api.ts"), "utf-8");
    assert.ok(text.includes("AbortSignal.timeout"), "GETs must carry an abort timeout");
    assert.ok(text.includes("API_GET_TIMEOUT_MS"), "timeout budget must be a named constant");
    assert.equal(api.API_GET_TIMEOUT_MS, 15000);
    assert.ok(text.includes("status === 401"), "401 must throw a re-login ApiError, not a generic failure");
  });

  it("auth decodes exp client-side; expired tokens never authenticate", async () => {
    const auth = await importLib("auth.ts");
    const b64 = (o) => Buffer.from(JSON.stringify(o)).toString("base64url");
    const future = `h.${b64({ exp: Math.floor(Date.now() / 1000) + 3600 })}.s`;
    const past = `h.${b64({ exp: Math.floor(Date.now() / 1000) - 3600 })}.s`;
    assert.equal(auth.isTokenExpired(future), false);
    assert.equal(auth.isTokenExpired(past), true);
    assert.equal(auth.isTokenExpired("garbage"), true);
    assert.equal(auth.isTokenExpired(null), true);
  });

  it("landing page validates the token against the backend on mount", () => {
    const text = readFileSync(join(SRC, "app", "page.tsx"), "utf-8");
    assert.ok(text.includes("fetchCurrentUser"), "must validate via /auth/me, not trust localStorage");
  });

  it("Console LRO fallback fires on 404 only", () => {
    const text = readFileSync(join(SRC, "components", "Console.tsx"), "utf-8");
    assert.ok(text.includes("err.status === 404"), "matches fallback must be 404-gated");
  });

  it("RegistrationLauncher sends auth on moon-points/matches and budgets fetches", () => {
    const text = readFileSync(join(SRC, "components", "RegistrationLauncher.tsx"), "utf-8");
    assert.ok(text.includes("moon-points/${jobId}`"), "moon-points fetch must exist");
    assert.ok(text.includes("getAuthHeaders()"), "all backend fetches must carry auth headers");
    assert.ok(text.includes("REGISTER_TIMEOUT_MS"), "POST /register needs its own (longer) budget");
    assert.ok(!text.includes('await import("@/lib/auth")'), "no per-call dynamic auth imports");
  });

  it("LinkedCursor math scales by natural dims with confidence guards", () => {
    const text = readFileSync(join(SRC, "components", "LinkedCursorPanel.tsx"), "utf-8");
    assert.ok(!text.includes("coords[0] / TILE_PX"), "dot fractions must not hardcode 512");
    assert.ok(text.includes("nat.w") && text.includes("nat.h"), "dots and clicks scale by natural dims");
    assert.ok(text.includes("p.confidence ?? 0"), "missing confidence must render 0%, never NaN%");
  });

  it("MapPanel rewrites unrenderable TIFF overlays and badges the fallback", () => {
    const text = readFileSync(join(SRC, "components", "MapPanel.tsx"), "utf-8");
    assert.ok(/\.tiff\?/.test(text), "must detect .tif/.tiff overlay URLs");
    assert.ok(text.includes("TIFF"), "must badge the PNG-preview fallback");
  });

  it("sensors: unknown ids stay unknown; filter labels derive from SENSOR_META", async () => {
    const sensors = await importLib("sensors.ts");
    assert.equal(sensors.normalizeSensorId("TMC-2"), "tmc");
    assert.equal(sensors.normalizeSensorId("LRO_NAC"), "lro_nac");
    assert.equal(sensors.normalizeSensorId("FUTURE-X"), "unknown");
    assert.ok(sensors.sensorFilterLabel("tmc").includes("5"), "TMC label must track the 5.0 spec, not a 4m string");
    assert.ok(sensors.sensorFilterLabel("iirs").includes("80"), "IIRS label must track the 80.0 spec, not a 70m string");
  });

  it("IngestPage poll is bounded with backoff", () => {
    const text = readFileSync(join(SRC, "components", "ingest", "IngestPage.tsx"), "utf-8");
    assert.ok(text.includes("MAX_POLL_FAILURES"), "poll loop must stop after N consecutive failures");
    assert.ok(text.includes("2 **"), "backoff must grow exponentially between retries");
  });

  it("DropZone loops readEntries, recurses, and toasts ignored files", () => {
    const text = readFileSync(join(SRC, "components", "ingest", "DropZone.tsx"), "utf-8");
    assert.ok(text.includes("for (;;)"), "directory reads must loop until readEntries drains");
    assert.ok(text.includes("ignoredNotice") || text.includes("onIgnored"), "ignored non-.zip files must surface, not vanish");
  });

  it("ResultsTable accounts for LRO-excluded rows and keys by triplet id", () => {
    const text = readFileSync(join(SRC, "components", "ingest", "ResultsTable.tsx"), "utf-8");
    assert.ok(text.includes("excludedCount"), "must show how many LRO/external rows were filtered");
    assert.ok(!text.includes("key={i}"), "rows must not key by array index");
    assert.ok(text.includes("region_id"), "row keys must derive from the triplet id");
  });

  it("image proxy contains resolved paths and 502s when the backend is down", () => {
    const text = readFileSync(join(SRC, "app", "images", "[...slug]", "route.ts"), "utf-8");
    assert.ok(text.includes("startsWith(publicDir"), "candidates must resolve inside publicDir (traversal guard)");
    assert.ok(text.includes("status: 502"), "upstream-down must be 502, not 404");
    assert.ok(text.includes("promises as fs"), "must use async fs/promises, not sync fs");
    assert.ok(!text.includes("fs.existsSync") && !text.includes("fs.readFileSync") && !text.includes("fs.statSync"), "no sync fs calls on the hot path");
  });
});
