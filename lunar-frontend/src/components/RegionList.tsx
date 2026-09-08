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
      <div className="border-b border-border bg-panel px-4 py-3.5 flex items-center justify-between">
        <div>
          <h2 className="text-xs font-semibold uppercase tracking-wider text-white">
            Regions
          </h2>
          <p className="mt-0.5 text-2xs font-mono text-ink-faint">
            {triplets.length} validated triplets
          </p>
        </div>
        <span className="rounded-full border border-teal/30 bg-teal/10 px-2.5 py-0.5 font-mono text-[10px] text-teal font-semibold">
          8/8 Ready
        </span>
      </div>

      <div className="flex-1 overflow-y-auto p-3 space-y-2">
        {triplets.map((t) => {
          const { widthKm, heightKm } = footprintSizeKm(t.bounds);
          const active = t.id === selectedId;
          return (
            <button
              key={t.id}
              onClick={() => onSelect(t.id)}
              className={`group block w-full rounded-xl p-3 text-left transition-all duration-200 ${
                active
                  ? "bg-teal/10 border border-teal/50 shadow-[0_0_15px_rgba(63,181,201,0.15)]"
                  : "bg-panel-raised/60 border border-white/5 hover:border-white/15 hover:bg-panel-raised"
              }`}
            >
              <div className="flex items-center justify-between">
                <span
                  className={`font-mono text-xs font-medium transition-colors ${
                    active ? "text-teal font-bold" : "text-white group-hover:text-teal"
                  }`}
                >
                  {t.id}
                </span>
                {t.dem_available && (
                  <span className="rounded-full border border-teal/30 bg-teal/10 px-1.5 py-0.5 font-mono text-[9px] text-teal font-medium">
                    DEM
                  </span>
                )}
              </div>
              <div className="mt-1.5 flex items-center justify-between font-mono text-2xs text-ink-dim">
                <span>
                  {widthKm.toFixed(1)} × {heightKm.toFixed(1)} km
                </span>
                <span className="text-[10px] text-ink-faint group-hover:text-ink-dim">
                  {t.bounds.north_lat > 0 ? "North" : "South"}
                </span>
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
