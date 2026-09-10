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
});
