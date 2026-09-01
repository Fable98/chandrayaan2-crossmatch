"use client";

import type { TripletSummary } from "@/lib/types";
import { footprintSizeKm } from "@/lib/geo";

interface Props {
  triplets: TripletSummary[];
  selectedId: string | null;
  onSelect: (id: string) => void;
}

export default function RegionList({ triplets, selectedId, onSelect }: Props) {
  return (
    <div className="flex h-full flex-col">
      <div className="border-b border-border px-4 py-3">
        <h2 className="text-2xs uppercase tracking-wide text-ink-faint">
          Regions
        </h2>
        <p className="mt-0.5 text-2xs text-ink-faint">
          {triplets.length} triplet{triplets.length === 1 ? "" : "s"} loaded
        </p>
      </div>
      <div className="flex-1 overflow-y-auto">
        {triplets.map((t) => {
          const { widthKm, heightKm } = footprintSizeKm(t.bounds);
          const active = t.id === selectedId;
          return (
            <button
              key={t.id}
              onClick={() => onSelect(t.id)}
              className={`block w-full border-b border-border px-4 py-3 text-left transition-colors ${
                active
                  ? "bg-panel-raised border-l-2 border-l-regolith"
                  : "border-l-2 border-l-transparent hover:bg-panel-raised/50"
              }`}
            >
              <div className="flex items-baseline justify-between">
                <span
                  className={`font-mono text-sm ${
                    active ? "text-regolith" : "text-ink"
                  }`}
                >
                  {t.id}
                </span>
                {t.dem_available && (
                  <span className="text-2xs text-parallax font-mono">DEM</span>
                )}
              </div>
              <div className="mt-1 text-2xs text-ink-dim font-mono">
                {widthKm.toFixed(1)} × {heightKm.toFixed(1)} km
              </div>
            </button>
          );
        })}
        {triplets.length === 0 && (
          <p className="px-4 py-6 text-sm text-ink-faint">
            No regions returned by the backend yet.
          </p>
        )}
      </div>
    </div>
  );
}
