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
export const API_BASE =
  process.env.NEXT_PUBLIC_API_BASE_URL?.replace(/\/$/, "") ??
  "http://localhost:8000";

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
      headers: {
        ...getAuthHeaders(),
      },
    });
  } catch (err) {
    throw new ApiError(
      `Could not reach the backend at ${API_BASE}. Is FastAPI running and is CORS configured for this origin?`,
      0,
      url
    );
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
  getTriplet: (id: string): Promise<TripletSummary> =>
    getJson<TripletSummary>(`/triplets/${id}`),
  getMatches: (id: string): Promise<MatchesResponse> =>
    getJson<MatchesResponse>(`/triplets/${id}/matches`),
  getIirsOverlay: (id: string): Promise<IIRSOverlay> =>
    getJson<IIRSOverlay>(`/triplets/${id}/iirs-overlay`),
};
