import type {
  TripletListResponse,
  TripletSummary,
  MatchesResponse,
  IIRSOverlay,
} from "./types";
import { getAuthHeaders } from "./auth";

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
  listTriplets: () => getJson<TripletListResponse>("/triplets"),
  getTriplet: (id: string) => getJson<TripletSummary>(`/triplets/${id}`),
  getMatches: (id: string) =>
    getJson<MatchesResponse>(`/triplets/${id}/matches`),
  getIirsOverlay: (id: string) =>
    getJson<IIRSOverlay>(`/triplets/${id}/iirs-overlay`),
};

export { ApiError, API_BASE };
