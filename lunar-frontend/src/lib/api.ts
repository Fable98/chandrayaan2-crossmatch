import type {
  TripletListResponse,
  TripletSummary,
  MatchesResponse,
  IIRSOverlay,
} from "./types";
import { getAuthHeaders } from "./auth";
import { FALLBACK_TRIPLETS, FALLBACK_MATCHES } from "./fallbackData";

// Point this at your running FastAPI instance. Override at build/run time
// with NEXT_PUBLIC_API_BASE_URL if the backend isn't on localhost:8000 —
// e.g. NEXT_PUBLIC_API_BASE_URL=http://192.168.1.20:8000 npm run dev
const API_BASE =
  process.env.NEXT_PUBLIC_API_BASE_URL?.replace(/\/$/, "") ??
  "http://localhost:8000";

class ApiError extends Error {
  constructor(
    message: string,
    public status: number,
    public url: string
  ) {
    super(message);
    this.name = "ApiError";
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
  // Backend returns paths like "/images/ohrc/region_003" — join with the
  // API base rather than the frontend's own origin.
  if (path.startsWith("http")) return path;
  return `${API_BASE}${path}`;
}

export const api = {
  listTriplets: async (): Promise<TripletListResponse> => {
    try {
      return await getJson<TripletListResponse>("/triplets");
    } catch {
      console.warn("FastAPI backend offline, loading fallback lunar archive data.");
      return { triplets: FALLBACK_TRIPLETS };
    }
  },
  getTriplet: async (id: string): Promise<TripletSummary> => {
    try {
      return await getJson<TripletSummary>(`/triplets/${id}`);
    } catch {
      const found = FALLBACK_TRIPLETS.find((t) => t.id === id);
      return found ?? FALLBACK_TRIPLETS[0];
    }
  },
  getMatches: async (id: string): Promise<MatchesResponse> => {
    try {
      return await getJson<MatchesResponse>(`/triplets/${id}/matches`);
    } catch {
      if (FALLBACK_MATCHES[id]) return FALLBACK_MATCHES[id];
      return {
        triplet_id: id,
        num_matches: 0,
        homography: null,
        matches: [],
        metrics: null,
      };
    }
  },
  getIirsOverlay: async (id: string): Promise<IIRSOverlay> => {
    try {
      return await getJson<IIRSOverlay>(`/triplets/${id}/iirs-overlay`);
    } catch {
      const triplet = FALLBACK_TRIPLETS.find((t) => t.id === id) ?? FALLBACK_TRIPLETS[0];
      return {
        triplet_id: id,
        image_url: "/images/iirs_overlay.png",
        bounds: triplet.bounds,
        opacity_hint: 0.65,
      };
    }
  },
};

export { ApiError, API_BASE };

