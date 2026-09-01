"use client";

import { useEffect, useState } from "react";
import { api, ApiError } from "@/lib/api";
import { footprintSizeKm } from "@/lib/geo";
import type { TripletSummary, MatchPoint, IIRSOverlay } from "@/lib/types";
import RegionList from "./RegionList";
import MapPanel from "./DynamicMapPanel";
import LinkedCursorPanel from "./LinkedCursorPanel";

type View = "map" | "linked-cursor";

export default function Console() {
  const [triplets, setTriplets] = useState<TripletSummary[]>([]);
  const [selectedId, setSelectedId] = useState<string | null>(null);
  const [detail, setDetail] = useState<TripletSummary | null>(null);
  const [matches, setMatches] = useState<MatchPoint[]>([]);
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
    setIirsOverlay(null);

    Promise.all([
      api.getTriplet(selectedId),
      api.getMatches(selectedId),
      api.getIirsOverlay(selectedId).catch(() => null),
    ])
      .then(([d, m, iirs]) => {
        setDetail(d);
        setMatches(m.matches);
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
        {detail && <MetaBar triplet={detail} />}
      </main>
    </div>
  );
}

function MetaBar({ triplet }: { triplet: TripletSummary }) {
  const { widthKm, heightKm } = footprintSizeKm(triplet.bounds);
  return (
    <div className="flex flex-wrap gap-x-6 gap-y-1 border-t border-border bg-panel px-4 py-2 text-2xs font-mono text-ink-faint">
      <span>OHRC {triplet.ohrc_product_id ?? triplet.id}</span>
      <span>TMC-2 {triplet.tmc2_product_id ?? "—"}</span>
      <span>IIRS {triplet.iirs_product_id ?? "—"}</span>
      <span>
        {widthKm.toFixed(2)} × {heightKm.toFixed(2)} km
      </span>
      <span>
        lon {triplet.bounds.west_lon.toFixed(4)}°–
        {triplet.bounds.east_lon.toFixed(4)}°
      </span>
      <span>
        lat {triplet.bounds.south_lat.toFixed(4)}°–
        {triplet.bounds.north_lat.toFixed(4)}°
      </span>
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
