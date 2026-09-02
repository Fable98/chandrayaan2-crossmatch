"use client";

import { useEffect, useState } from "react";
import { api, ApiError, imageUrl } from "@/lib/api";
import { footprintSizeKm } from "@/lib/geo";
import type { TripletSummary, MatchPoint, IIRSOverlay, MatchMetrics } from "@/lib/types";
import RegionList from "./RegionList";
import MapPanel from "./DynamicMapPanel";
import LinkedCursorPanel from "./LinkedCursorPanel";

type View = "map" | "linked-cursor" | "registration";

export default function Console() {
  const [triplets, setTriplets] = useState<TripletSummary[]>([]);
  const [selectedId, setSelectedId] = useState<string | null>(null);
  const [detail, setDetail] = useState<TripletSummary | null>(null);
  const [matches, setMatches] = useState<MatchPoint[]>([]);
  const [metrics, setMetrics] = useState<MatchMetrics | null>(null);
  const [iirsOverlay, setIirsOverlay] = useState<IIRSOverlay | null>(null);
  const [view, setView] = useState<View>("map");
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);

  // Load the region list once on mount.
  useEffect(() => {
    api
      .listTriplets()
      .then((res) => {
        setTriplets(res.triplets);
        if (res.triplets.length > 0) setSelectedId(res.triplets[0].id);
      })
      .catch((err) => setError(describeError(err)))
      .finally(() => setLoading(false));
  }, []);

  // Load per-triplet detail, matches, and IIRS overlay whenever selection changes.
  useEffect(() => {
    if (!selectedId) return;
    setDetail(null);
    setMatches([]);
    setMetrics(null);
    setIirsOverlay(null);

    Promise.all([
      api.getTriplet(selectedId),
      api.getMatches(selectedId),
      api.getIirsOverlay(selectedId).catch(() => null),
    ])
      .then(([d, m, iirs]) => {
        setDetail(d);
        setMatches(m.matches);
        setMetrics(m.metrics ?? null);
        setIirsOverlay(iirs);
      })
      .catch((err) => setError(describeError(err)));
  }, [selectedId]);

  if (error) {
    return (
      <div className="flex h-screen items-center justify-center bg-void px-6">
        <div className="max-w-md border border-alert/40 bg-alert/5 p-5">
          <p className="text-sm font-medium text-alert">Backend unreachable</p>
          <p className="mt-2 text-sm text-ink-dim">{error}</p>
          <p className="mt-3 text-2xs font-mono text-ink-faint">
            Set NEXT_PUBLIC_API_BASE_URL if the API isn't at
            http://localhost:8000, and confirm CORS allows this origin.
          </p>
        </div>
      </div>
    );
  }

  return (
    <div className="grid h-screen grid-cols-[260px_1fr] grid-rows-[56px_1fr]">
      <header className="col-span-2 flex items-center justify-between border-b border-border px-4">
        <div className="flex items-baseline gap-3">
          <span className="font-mono text-sm text-regolith">SIH26166</span>
          <span className="text-sm text-ink-dim">
            Lunar Correspondence Console
          </span>
        </div>
        <nav className="flex gap-1">
          <TabButton active={view === "map"} onClick={() => setView("map")}>
            Fused map
          </TabButton>
          <TabButton
            active={view === "linked-cursor"}
            onClick={() => setView("linked-cursor")}
          >
            Linked cursor
          </TabButton>
          <TabButton
            active={view === "registration"}
            onClick={() => setView("registration")}
          >
            Registration QA
          </TabButton>
        </nav>
      </header>

      <aside className="row-start-2 border-r border-border">
        <RegionList
          triplets={triplets}
          selectedId={selectedId}
          onSelect={setSelectedId}
        />
      </aside>

      <main className="row-start-2 flex flex-col overflow-hidden">
        {loading && (
          <Centered>Loading regions from the backend…</Centered>
        )}
        {!loading && !detail && selectedId && (
          <Centered>Loading {selectedId}…</Centered>
        )}
        {!loading && !selectedId && (
          <Centered>No regions available.</Centered>
        )}
        {detail && view === "map" && (
          <MapPanel triplet={detail} iirsOverlay={iirsOverlay} />
        )}
        {detail && view === "linked-cursor" && (
          <LinkedCursorPanel tripletId={detail.id} points={matches} />
        )}
        {detail && view === "registration" && (
          <RegistrationPanel tripletId={detail.id} metrics={metrics} />
        )}
        {detail && <MetaBar triplet={detail} metrics={metrics} />}
      </main>
    </div>
  );
}

function RegistrationPanel({
  tripletId,
  metrics,
}: {
  tripletId: string;
  metrics: MatchMetrics | null;
}) {
  return (
    <div className="flex h-full flex-col overflow-y-auto p-6">
      <div className="mb-4 flex items-center justify-between border-b border-border pb-3">
        <div>
          <h2 className="text-sm font-semibold text-ink">
            Geometric Registration Output · {tripletId}
          </h2>
          <p className="text-2xs text-ink-faint">
            Source (OHRC) warped via estimated homography into target (TMC-2) coordinate space.
          </p>
        </div>
        {metrics && (
          <div className="flex gap-3 text-2xs font-mono">
            <span className="rounded bg-panel-raised px-2 py-1 text-teal">
              RMSE: {metrics.rmse_px.toFixed(3)} px
            </span>
            <span
              className={`rounded px-2 py-1 ${
                metrics.sub_pixel_accurate
                  ? "bg-teal/20 text-teal"
                  : "bg-amber-500/20 text-amber-300"
              }`}
            >
              {metrics.sub_pixel_accurate ? "Sub-pixel: YES" : "Sub-pixel: NO"}
            </span>
            <span className="rounded bg-panel-raised px-2 py-1 text-ink-dim">
              Inliers: {metrics.num_inliers}
            </span>
            <span className="rounded bg-panel-raised px-2 py-1 text-ink-dim">
              Coverage: {(metrics.combined_coverage_score * 100).toFixed(0)}%
            </span>
          </div>
        )}
      </div>

      <div className="grid grid-cols-3 gap-4">
        <div className="flex flex-col items-center gap-2">
          <div className="relative aspect-square w-full overflow-hidden border border-border bg-panel">
            {/* eslint-disable-next-line @next/next/no-img-element */}
            <img
              src={imageUrl(`/images/registered/${tripletId}/registered_ohrc.png`)}
              alt="Warped Source (OHRC)"
              className="h-full w-full object-cover"
              onError={(e) => {
                (e.currentTarget as HTMLImageElement).src = imageUrl(`/images/ohrc/${tripletId}`);
              }}
            />
          </div>
          <span className="font-mono text-2xs text-ink-dim">1. Warped Source (OHRC)</span>
        </div>

        <div className="flex flex-col items-center gap-2">
          <div className="relative aspect-square w-full overflow-hidden border border-border bg-panel">
            {/* eslint-disable-next-line @next/next/no-img-element */}
            <img
              src={imageUrl(`/images/registered/${tripletId}/blend_overlay.png`)}
              alt="Blend Overlay (50% Alpha)"
              className="h-full w-full object-cover"
              onError={(e) => {
                (e.currentTarget as HTMLImageElement).src = imageUrl(`/images/tmc/${tripletId}`);
              }}
            />
          </div>
          <span className="font-mono text-2xs text-ink-dim">2. Blend Overlay (50% Alpha)</span>
        </div>

        <div className="flex flex-col items-center gap-2">
          <div className="relative aspect-square w-full overflow-hidden border border-border bg-panel">
            {/* eslint-disable-next-line @next/next/no-img-element */}
            <img
              src={imageUrl(`/images/registered/${tripletId}/checkerboard_qa.png`)}
              alt="Checkerboard QA"
              className="h-full w-full object-cover"
              onError={(e) => {
                (e.currentTarget as HTMLImageElement).src = imageUrl(`/images/tmc/${tripletId}`);
              }}
            />
          </div>
          <span className="font-mono text-2xs text-ink-dim">3. Checkerboard Continuity QA</span>
        </div>
      </div>
    </div>
  );
}

function MetaBar({
  triplet,
  metrics,
}: {
  triplet: TripletSummary;
  metrics: MatchMetrics | null;
}) {
  const { widthKm, heightKm } = footprintSizeKm(triplet.bounds);
  return (
    <div className="flex flex-wrap items-center justify-between border-t border-border bg-panel px-4 py-2 text-2xs font-mono text-ink-faint">
      <div className="flex flex-wrap gap-x-6 gap-y-1">
        <span>OHRC {triplet.ohrc_product_id ?? triplet.id}</span>
        <span>TMC-2 {triplet.tmc2_product_id ?? "—"}</span>
        <span>IIRS {triplet.iirs_product_id ?? "—"}</span>
        <span>
          {widthKm.toFixed(2)} × {heightKm.toFixed(2)} km
        </span>
        <span>
          lon {triplet.bounds.west_lon.toFixed(4)}°–{triplet.bounds.east_lon.toFixed(4)}°
        </span>
        <span>
          lat {triplet.bounds.south_lat.toFixed(4)}°–{triplet.bounds.north_lat.toFixed(4)}°
        </span>
      </div>
      {metrics && (
        <div className="flex gap-3 text-teal">
          <span>RMSE: {metrics.rmse_px.toFixed(2)}px</span>
          <span>Coverage: {(metrics.combined_coverage_score * 100).toFixed(0)}%</span>
        </div>
      )}
    </div>
  );
}

function TabButton({
  active,
  onClick,
  children,
}: {
  active: boolean;
  onClick: () => void;
  children: React.ReactNode;
}) {
  return (
    <button
      onClick={onClick}
      className={`px-3 py-1.5 text-sm transition-colors ${
        active
          ? "bg-panel-raised text-ink"
          : "text-ink-faint hover:text-ink-dim"
      }`}
    >
      {children}
    </button>
  );
}

function Centered({ children }: { children: React.ReactNode }) {
  return (
    <div className="flex flex-1 items-center justify-center text-sm text-ink-faint">
      {children}
    </div>
  );
}

function describeError(err: unknown): string {
  if (err instanceof ApiError) return err.message;
  if (err instanceof Error) return err.message;
  return "Unknown error.";
}
