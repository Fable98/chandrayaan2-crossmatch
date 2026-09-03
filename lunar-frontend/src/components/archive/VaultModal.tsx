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
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/85 p-4 backdrop-blur-md">
      <div className="relative flex max-h-[92vh] w-full max-w-6xl flex-col overflow-hidden border border-[#23211d] bg-[#08080a] text-[#f0f2f5] shadow-[0_0_60px_rgba(0,0,0,0.9)]">
        {/* Header Bar */}
        <div className="flex flex-wrap items-center justify-between gap-4 border-b border-[#23211d] bg-[#0e0e12] px-6 py-4">
          <div className="flex items-center gap-3">
            <svg className="h-5 w-5 text-[#d4af37]" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M5 8h14M5 8a2 2 0 110-4h14a2 2 0 110 4M5 8v10a2 2 0 002 2h10a2 2 0 002-2V8m-9 4h4" />
            </svg>
            <div>
              <div className="flex items-center gap-2">
                <span className="font-mono text-xs text-[#d4af37]">[ LUNAR GLASS VAULT ]</span>
                <span className="font-mono text-xs font-bold uppercase tracking-wider text-white">
                  ALL REGISTRATION PRODUCTS &amp; RAW SENSORS
                </span>
              </div>
              <p className="font-mono text-[10px] text-[#6b665f]">
                {triplets.length} validated regions · Chandrayaan-2 multi-spectral repository
              </p>
            </div>
          </div>

          <button
            onClick={onClose}
            className="flex h-7 w-7 items-center justify-center border border-[#23211d] font-mono text-xs text-[#9a958e] transition-colors hover:border-[#d4af37] hover:text-[#d4af37] focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-[#d4af37]"
            title="Close Vault"
          >
            ✕
          </button>
        </div>

        {/* Filter & Search Toolbar */}
        <div className="flex flex-wrap items-center justify-between gap-4 border-b border-[#23211d] bg-[#121217] px-6 py-3">
          {/* Filter Pills */}
          <div className="flex flex-wrap gap-2">
            <FilterButton
              active={filter === "all"}
              onClick={() => setFilter("all")}
              label="ALL PAYLOADS"
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
              label="REGISTRATION QA"
            />
          </div>

          {/* Search */}
          <div className="flex items-center gap-2 border border-[#23211d] bg-[#08080a] px-3 py-1 font-mono text-xs text-[#6b665f]">
            <svg className="h-3.5 w-3.5 text-[#6b665f]" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0z" />
            </svg>
            <input
              type="text"
              value={search}
              onChange={(e) => setSearch(e.target.value)}
              placeholder="FILTER REGIONS..."
              className="w-40 bg-transparent text-xs text-white placeholder-[#4f4b45] focus:outline-none sm:w-56"
            />
          </div>
        </div>

        {/* Vault Grid Content */}
        <div className="flex-1 overflow-y-auto p-6">
          <div className="grid grid-cols-1 gap-6 sm:grid-cols-2 lg:grid-cols-4">
            {filteredTriplets.map((t, idx) => {
              const { widthKm, heightKm } = footprintSizeKm(t.bounds);

              // Decide thumbnail based on filter
              let thumbUrl = imageUrl(`/images/ohrc/${t.id}`);
              let badge = "OHRC 0.25M";
              let targetView: "registration" | "linked-cursor" | "map" = "linked-cursor";

              if (filter === "tmc") {
                thumbUrl = imageUrl(`/images/tmc/${t.id}`);
                badge = "TMC-2 4M";
                targetView = "linked-cursor";
              } else if (filter === "iirs") {
                thumbUrl = imageUrl(`/images/iirs/${t.id}`);
                badge = "IIRS 70M";
                targetView = "map";
              } else if (filter === "qa") {
                thumbUrl = imageUrl(`/images/registered/${t.id}/blend_overlay.png`);
                badge = "CO-REG QA";
                targetView = "registration";
              }

              return (
                <div
                  key={t.id}
                  className="group flex flex-col justify-between border border-[#23211d] bg-[#0d0d11] p-4 transition-all duration-200 hover:border-[#d4af37] hover:bg-[#14141a]"
                >
                  <div>
                    {/* Top line */}
                    <div className="mb-2 flex items-center justify-between">
                      <span className="font-mono text-[10px] text-[#6b665f]">
                        {idx + 1 < 10 ? `0${idx + 1}` : idx + 1}.
                      </span>
                      <span className="rounded border border-[#38342d] bg-[#121217] px-1.5 py-0.5 font-mono text-[9px] uppercase tracking-wider text-[#d4af37]">
                        {badge}
                      </span>
                    </div>

                    {/* Image Preview */}
                    <div className="relative aspect-square overflow-hidden border border-[#23211d] bg-black">
                      {/* eslint-disable-next-line @next/next/no-img-element */}
                      <img
                        src={thumbUrl}
                        alt={t.id}
                        className="h-full w-full object-cover contrast-[1.2] brightness-95 transition-transform duration-300 group-hover:scale-105"
                        onError={(e) => {
                          (e.currentTarget as HTMLImageElement).src = imageUrl(filter === "iirs" ? "/images/iirs/iirs_overlay.png" : `/images/ohrc/${t.id}`);
                        }}
                      />
                    </div>

                    {/* Region Metadata */}
                    <div className="mt-3">
                      <h4 className="font-mono text-xs font-semibold text-white group-hover:text-[#d4af37] transition-colors">
                        {t.id}
                      </h4>
                      <p className="mt-1 font-mono text-[10px] text-[#9a958e]">
                        {widthKm.toFixed(1)} × {heightKm.toFixed(1)} km
                      </p>
                      <p className="mt-0.5 font-mono text-[9px] text-[#6b665f]">
                        {t.bounds.west_lon.toFixed(2)}°E, {t.bounds.north_lat.toFixed(2)}°N
                      </p>
                    </div>
                  </div>

                  {/* Actions */}
                  <div className="mt-4 pt-3 border-t border-[#1e1c18] flex items-center justify-between">
                    <span className="font-mono text-[9px] text-[#d4af37]">
                      {t.dem_available ? "DEM READY" : "STEREO"}
                    </span>
                    <button
                      onClick={() => {
                        onClose();
                        onSelectRegion(t.id, targetView);
                      }}
                      className="border border-[#38342d] bg-[#121217] px-2.5 py-1 font-mono text-[10px] uppercase tracking-wider text-white transition-colors hover:border-[#d4af37] hover:bg-[#d4af37] hover:text-black focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-[#d4af37]"
                    >
                      INSPECT →
                    </button>
                  </div>
                </div>
              );
            })}
          </div>

          {filteredTriplets.length === 0 && (
            <div className="py-20 text-center font-mono text-xs text-[#9a958e]">
              No regions matching "{search}".
            </div>
          )}
        </div>

        {/* Footer */}
        <div className="flex items-center justify-between border-t border-[#23211d] bg-[#0e0e12] px-6 py-3 font-mono text-[11px] text-[#6b665f]">
          <span>Obsidian Archive Repository · GeoTIFF &amp; PDS4 Data Standard</span>
          <button
            onClick={onClose}
            className="text-[#9a958e] hover:text-[#d4af37] transition-colors"
          >
            Close Archive ✕
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
      className={`rounded border px-3 py-1 font-mono text-xs transition-all duration-200 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-[#d4af37] ${
        active
          ? "border-[#d4af37] bg-[#2c2619] font-semibold text-[#f3df9b]"
          : "border-[#23211d] bg-[#08080a] text-[#9a958e] hover:border-[#38342d] hover:text-white"
      }`}
    >
      {label}
    </button>
  );
}
