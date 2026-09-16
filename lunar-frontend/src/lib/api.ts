import type {
  TripletListResponse,
  TripletSummary,
  MatchesResponse,
  IIRSOverlay,
} from "./types";
import { getAuthHeaders } from "./auth";

// Single API base for the whole frontend (Step 13 contract). ingest-api.ts
// imports API_BASE from here — no second base URL is allowed, so staging /
// production can never split-brain between two backends.
//
// Point this at your running FastAPI instance. Override at build/run time
// with NEXT_PUBLIC_API_BASE_URL if the backend isn't on localhost:8000 —
// e.g. NEXT_PUBLIC_API_BASE_URL=http://192.168.1.20:8000 npm run dev
// NEXT_PUBLIC_API_BASE_URL is baked at build time; an empty string env
// (e.g. `NEXT_PUBLIC_API_BASE_URL= docker build`) must fall back to
// localhost — `??` would keep the empty string and every fetch would go
// same-origin. `||` treats "" as unset.
export const API_BASE =
  (process.env.NEXT_PUBLIC_API_BASE_URL || "").replace(/\/$/, "") ||
  "http://localhost:8000";

// Aborted/slow-backend budget for idempotent GETs. POST /register holds a
// CFOG worker for ~6-8s, so mutating uploads use their own longer budget.
export const API_GET_TIMEOUT_MS = 15000;

function timeoutSignal(ms: number): AbortSignal | undefined {
  try {
    if (typeof AbortSignal !== "undefined" && typeof AbortSignal.timeout === "function") {
      return AbortSignal.timeout(ms);
    }
  } catch {
    // Older runtimes without AbortSignal.timeout — fetch without a signal.
  }
  return undefined;
}

export class ApiError extends Error {
  // Plain field declarations (no TS parameter properties) so node
  // type-stripping can import this module in scripts/smoke.mjs.
  status: number;
  url: string;
  constructor(message: string, status: number, url: string) {
    super(message);
    this.name = "ApiError";
    this.status = status;
    this.url = url;
  }
}

async function getJson<T>(path: string): Promise<T> {
  const url = `${API_BASE}${path}`;
  let res: Response;
  try {
    res = await fetch(url, {
      cache: "no-store",
      signal: timeoutSignal(API_GET_TIMEOUT_MS),
      headers: {
        ...getAuthHeaders(),
      },
    });
  } catch (err) {
    if (err instanceof Error && err.name === "TimeoutError") {
      throw new ApiError(
        `Backend request timed out after ${API_GET_TIMEOUT_MS / 1000}s: ${path}. The server may be busy — retry in a moment.`,
        0,
        url
      );
    }
    throw new ApiError(
      `Could not reach the backend at ${API_BASE}. Is FastAPI running and is CORS configured for this origin?`,
      0,
      url
    );
  }
  if (res.status === 401) {
    // Stale/rotated JWT: callers (Console error banner, IngestPage session
    // wall) match on the 401 status to force a re-login instead of showing
    // a generic fetch failure.
    throw new ApiError("Unauthorized — your session expired. Please sign in again.", 401, url);
  }
  if (!res.ok) {
    throw new ApiError(`${res.status} ${res.statusText}`, res.status, url);
  }
  return res.json() as Promise<T>;
}

export function imageUrl(path: string): string {
  if (!path) return "";
  if (path.startsWith("http")) return path;

  const cleanPath = path.startsWith("/") ? path : `/${path}`;

  // Backend-computed artifacts (/dynamic_runs/...) live on the FastAPI host,
  // NOT on the Next.js origin: a relative URL would resolve against the
  // frontend and silently 404. Prefix the single API base (Step 13 fix).
  if (cleanPath.startsWith("/dynamic_runs/")) {
    return `${API_BASE}${cleanPath}`;
  }

  // Bundled static lunar imagery (/images/...) ships in public/images/ and
  // is served by Next.js / Vercel Edge CDN.
  if (cleanPath.startsWith("/images/")) {
    if (
      !cleanPath.endsWith(".png") &&
      !cleanPath.endsWith(".jpg") &&
      !cleanPath.endsWith(".jpeg") &&
      !cleanPath.endsWith(".json")
    ) {
      // Ensure an image extension so browsers receive an image content-type.
      return `${cleanPath}.png`;
    }
    return cleanPath;
  }

  // Unknown relative path: assume a backend route and make it absolute so a
  // missing backend surfaces as a fetch error, never a same-origin 404 page.
  return `${API_BASE}${cleanPath}`;
}

// Step 13 contract: NO silent fallback data. Every method below throws
// ApiError when the backend is unreachable or returns an error status, and
// callers render the shared error banner (Console: "Archive Connection
// Failed"). Kill-backend => error banner, never fabricated archive data.
export const api = {
  listTriplets: (): Promise<TripletListResponse> =>
    getJson<TripletListResponse>("/triplets"),
  deleteTriplet: async (id: string, force = false): Promise<{ triplet_id: string; removed: string[] }> => {
    const res = await fetch(
      `${API_BASE}/triplets/${encodeURIComponent(id)}${force ? "?force=true" : ""}`,
      { method: "DELETE", headers: { ...getAuthHeaders() }, signal: timeoutSignal(API_GET_TIMEOUT_MS) }
    );
    if (res.status === 401) {
      throw new ApiError("Unauthorized — your session expired. Please sign in again.", 401, id);
    }
    if (!res.ok) {
      const err = await res.json().catch(() => ({ detail: res.statusText }));
      throw new ApiError(err.detail || "Delete failed", res.status, id);
    }
    return res.json();
  },
  getTriplet: (id: string): Promise<TripletSummary> =>
    getJson<TripletSummary>(`/triplets/${id}`),
  getMatches: (id: string): Promise<MatchesResponse> =>
    getJson<MatchesResponse>(`/triplets/${id}/matches`),
  getIirsOverlay: (id: string): Promise<IIRSOverlay> =>
    getJson<IIRSOverlay>(`/triplets/${id}/iirs-overlay`),
  getLroCandidates: (id: string) =>
    getJson<{
      triplet_id: string;
      candidates: Array<{
        product_id: string;
        label_url?: string | null;
        download_urls: string[];
        footprint_bounds?: {
          west_lon: number;
          east_lon: number;
          south_lat: number;
          north_lat: number;
        } | null;
        incidence_angle_deg?: number | null;
        overlap_score?: number;
        ranking_score?: number;
      }>;
    }>(`/triplets/${id}/lro-candidates`),
};
