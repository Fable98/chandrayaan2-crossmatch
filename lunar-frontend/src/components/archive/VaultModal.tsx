"use client";

import { useState } from "react";
import { imageUrl } from "@/lib/api";
import { footprintSizeKm } from "@/lib/geo";
import type { TripletSummary } from "@/lib/types";

export type PayloadFilter = "all" | "ohrc" | "tmc" | "iirs" | "qa";

interface Props {
  triplets: TripletSummary[];
  initialFilter?: PayloadFilter;
  onClose: () => void;
  onSelectRegion: (tripletId: string, preferredView?: "registration" | "linked-cursor" | "map") => void;
}

export default function VaultModal({
  triplets,
  initialFilter = "all",
  onClose,
  onSelectRegion,
}: Props) {
  const [filter, setFilter] = useState<PayloadFilter>(initialFilter);
  const [search, setSearch] = useState("");

  const filteredTriplets = triplets.filter((t) =>
    t.id.toLowerCase().includes(search.toLowerCase())
  );

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/80 p-4 backdrop-blur-2xl animate-fade-in">
      <div className="relative flex max-h-[90vh] w-full max-w-6xl flex-col overflow-hidden rounded-3xl border border-white/15 bg-[#0c0e24]/90 text-white shadow-[0_30px_90px_rgba(0,0,0,0.9)] ring-1 ring-white/10">
        {/* Header Bar */}
        <div className="flex flex-wrap items-center justify-between gap-4 border-b border-white/10 bg-white/[0.02] px-6 py-4 backdrop-blur-md">
          <div className="flex items-center gap-3">
            <div>
              <div className="flex items-center gap-2">
                <span className="rounded-full border border-purple-400/30 bg-purple-500/10 px-2.5 py-0.5 text-[10px] font-semibold uppercase tracking-wider text-purple-300">
                  Archive Vault
                </span>
                <span className="text-base font-bold text-white">
                  Lunar Cross-Match Products
                </span>
              </div>
              <p className="mt-0.5 text-xs text-slate-400">
                {triplets.length} validated regions · Chandrayaan-2 multi-sensor archive
              </p>
            </div>
          </div>

          <button
            onClick={onClose}
            className="flex h-7 w-7 items-center justify-center rounded-full border border-white/10 bg-white/[0.04] text-xs text-slate-300 transition hover:bg-white/[0.08] hover:text-white"
            title="Close"
          >
            ✕
          </button>
        </div>

        {/* Filter & Search Toolbar */}
        <div className="flex flex-wrap items-center justify-between gap-4 border-b border-white/[0.06] bg-white/[0.015] px-6 py-3">
          {/* Filter Pills */}
          <div className="flex flex-wrap gap-2">
            <FilterButton
              active={filter === "all"}
              onClick={() => setFilter("all")}
              label="All Payloads"
            />
            <FilterButton
              active={filter === "ohrc"}
              onClick={() => setFilter("ohrc")}
              label="OHRC (0.25m)"
            />
            <FilterButton
              active={filter === "tmc"}
              onClick={() => setFilter("tmc")}
              label="TMC-2 (4m)"
            />
            <FilterButton
              active={filter === "iirs"}
              onClick={() => setFilter("iirs")}
              label="IIRS (70m)"
            />
            <FilterButton
              active={filter === "qa"}
              onClick={() => setFilter("qa")}
              label="Registration QA"
            />
          </div>

          {/* Search */}
          <div className="flex items-center gap-2 rounded-full border border-white/10 bg-white/[0.04] px-4 py-1.5 text-xs text-slate-300 focus-within:border-purple-400/50">
            <svg className="h-3.5 w-3.5 text-slate-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0z" />
            </svg>
            <input
              type="text"
              value={search}
              onChange={(e) => setSearch(e.target.value)}
              placeholder="Search regions…"
              className="w-40 bg-transparent text-xs text-white placeholder-slate-400 focus:outline-none sm:w-56"
            />
          </div>
        </div>

        {/* Vault Grid Content */}
        <div className="flex-1 overflow-y-auto p-6">
          <div className="grid grid-cols-1 gap-4 sm:grid-cols-2 lg:grid-cols-4">
            {filteredTriplets.map((t, idx) => {
              const { widthKm, heightKm } = footprintSizeKm(t.bounds);

              let thumbUrl = imageUrl(`/images/ohrc/${t.id}`);
              let badge = "OHRC 0.25m";
              let targetView: "registration" | "linked-cursor" | "map" = "linked-cursor";

              if (filter === "tmc") {
                thumbUrl = imageUrl(`/images/tmc/${t.id}`);
                badge = "TMC-2 4m";
                targetView = "linked-cursor";
              } else if (filter === "iirs") {
                thumbUrl = imageUrl(`/images/iirs/${t.id}`);
                badge = "IIRS 70m";
                targetView = "map";
              } else if (filter === "qa") {
                thumbUrl = imageUrl(`/images/registered/${t.id}/blend_overlay.png`);
                badge = "Co-Reg QA";
                targetView = "registration";
              }

              return (
                <div
                  key={t.id}
                  className="group flex flex-col justify-between rounded-2xl border border-white/[0.07] bg-white/[0.025] p-4 backdrop-blur-xl transition-all hover:border-purple-400/40 hover:bg-white/[0.06] hover:shadow-[0_10px_30px_rgba(168,85,247,0.15)]"
                >
                  <div>
                    {/* Top line */}
                    <div className="mb-2 flex items-center justify-between">
                      <span className="font-mono text-[11px] text-slate-400">
                        #{String(idx + 1).padStart(2, "0")}
                      </span>
                      <span className="rounded-full border border-purple-400/30 bg-purple-500/10 px-2 py-0.5 text-[9px] font-semibold text-purple-300">
                        {badge}
                      </span>
                    </div>

                    {/* Image Preview */}
                    <div className="relative aspect-square overflow-hidden rounded-xl border border-white/[0.06] bg-black">
                      {/* eslint-disable-next-line @next/next/no-img-element */}
                      <img
                        src={thumbUrl}
                        alt={t.id}
                        className="h-full w-full object-cover"
                        onError={(e) => {
                          (e.currentTarget as HTMLImageElement).src = imageUrl(filter === "iirs" ? "/images/iirs/iirs_overlay.png" : `/images/ohrc/${t.id}`);
                        }}
                      />
                    </div>

                    {/* Region Metadata */}
                    <div className="mt-3">
                      <h4 className="text-xs font-semibold text-white group-hover:text-purple-200 transition">
                        {t.id}
                      </h4>
                      <p className="mt-0.5 text-xs text-slate-400">
                        {widthKm.toFixed(1)} × {heightKm.toFixed(1)} km
                      </p>
                      <p className="mt-0.5 text-[11px] text-slate-500 font-mono">
                        {t.bounds.west_lon.toFixed(2)}°E, {t.bounds.north_lat.toFixed(2)}°N
                      </p>
                    </div>
                  </div>

                  {/* Actions */}
                  <div className="mt-4 pt-3 border-t border-white/[0.05] flex items-center justify-between">
                    <span className="text-[10px] text-slate-400">
                      {t.dem_available ? "DEM available" : "Stereo"}
                    </span>
                    <button
                      onClick={() => {
                        onClose();
                        onSelectRegion(t.id, targetView);
                      }}
                      className="rounded-full border border-purple-400/30 bg-purple-500/10 px-3 py-1 text-xs font-medium text-purple-200 transition-all hover:bg-purple-600 hover:text-white hover:shadow-[0_0_15px_rgba(168,85,247,0.4)]"
                    >
                      Inspect →
                    </button>
                  </div>
                </div>
              );
            })}
          </div>

          {filteredTriplets.length === 0 && (
            <div className="py-20 text-center text-xs text-slate-400">
              No regions matching "{search}".
            </div>
          )}
        </div>

        {/* Footer */}
        <div className="flex items-center justify-between border-t border-white/10 bg-white/[0.02] px-6 py-3 text-xs text-slate-400">
          <span>Chandrayaan-2 Cross-Match Repository</span>
          <button
            onClick={onClose}
            className="hover:text-white transition-colors"
          >
            Close ✕
          </button>
        </div>
      </div>
    </div>
  );
}

function FilterButton({
  active,
  onClick,
  label,
}: {
  active: boolean;
  onClick: () => void;
  label: string;
}) {
  return (
    <button
      onClick={onClick}
      className={`rounded-full px-4 py-1 text-xs font-semibold transition-all ${
        active
          ? "bg-gradient-to-r from-purple-600 to-indigo-600 text-white shadow-[0_0_15px_rgba(168,85,247,0.35)]"
          : "border border-white/10 bg-white/[0.04] text-slate-300 hover:bg-white/[0.08] hover:text-white"
      }`}
    >
      {label}
    </button>
  );
}
