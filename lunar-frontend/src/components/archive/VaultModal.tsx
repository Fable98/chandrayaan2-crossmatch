"use client";

import { useState } from "react";
import { API_BASE, imageUrl } from "@/lib/api";
import { footprintSizeKm } from "@/lib/geo";
import type { TripletSummary } from "@/lib/types";
import { sensorBadge, sensorFilterLabel } from "@/lib/sensors";

export type PayloadFilter = "all" | "ohrc" | "tmc" | "iirs" | "lro" | "qa";

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
  // Tiles that failed to load, keyed by resolved URL. LRO tiles NEVER fall
  // back to OHRC pixels (that mislabels one sensor as another); they render
  // an explicit unavailable placeholder instead (Step 13 honesty rule).
  const [failedThumbs, setFailedThumbs] = useState<Set<string>>(new Set());

  const filteredTriplets = triplets.filter((t) => {
    const matchesSearch = t.id.toLowerCase().includes(search.toLowerCase());
    if (!matchesSearch) return false;
    if (filter === "lro") {
      return Boolean(t.lro_nac_available);
    }
    return true;
  });

  return (
    <div className="fixed inset-0 z-[2000] flex items-center justify-center bg-slate-900/40 p-4 backdrop-blur-sm animate-fade-in">
      <div className="relative flex max-h-[90vh] w-full max-w-6xl flex-col overflow-hidden rounded-2xl border border-slate-200/80 bg-white text-slate-800 shadow-2xl">
        {/* Header Bar */}
        <div className="flex flex-wrap items-center justify-between gap-4 border-b border-slate-100 bg-white px-6 py-4">
          <div className="flex items-center gap-3">
            <div>
              <div className="flex items-center gap-2">
                <span className="rounded-lg border border-indigo-100 bg-indigo-50 px-2.5 py-0.5 text-[11px] font-bold uppercase tracking-wider text-[#4F46E5]">
                  Archive Vault
                </span>
                <span className="text-base font-bold text-slate-900">
                  Lunar Cross-Match Products
                </span>
              </div>
              <p className="mt-0.5 text-xs text-slate-500">
                {triplets.length} validated regions · Chandrayaan-2 multi-sensor archive
              </p>
            </div>
          </div>

          <button
            onClick={onClose}
            className="flex h-7 w-7 items-center justify-center rounded-lg border border-slate-200 bg-white text-xs text-slate-400 transition hover:bg-slate-50 hover:text-slate-700"
            title="Close"
          >
            ✕
          </button>
        </div>

        {/* Filter & Search Toolbar */}
        <div className="flex flex-wrap items-center justify-between gap-4 border-b border-slate-100 bg-slate-50/50 px-6 py-3">
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
              label={sensorFilterLabel("ohrc")}
            />
            <FilterButton
              active={filter === "tmc"}
              onClick={() => setFilter("tmc")}
              label={sensorFilterLabel("tmc")}
            />
            <FilterButton
              active={filter === "iirs"}
              onClick={() => setFilter("iirs")}
              label={sensorFilterLabel("iirs")}
            />
            <FilterButton
              active={filter === "lro"}
              onClick={() => setFilter("lro")}
              label={sensorFilterLabel("lro")}
            />
            <FilterButton
              active={filter === "qa"}
              onClick={() => setFilter("qa")}
              label="Registration QA"
            />
          </div>

          {/* Search */}
          <div className="flex items-center gap-2 rounded-xl border border-slate-200 bg-white px-3.5 py-1.5 text-xs text-slate-700 focus-within:border-[#4F46E5]">
            <svg className="h-3.5 w-3.5 text-slate-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0z" />
            </svg>
            <input
              type="text"
              value={search}
              onChange={(e) => setSearch(e.target.value)}
              placeholder="Search regions…"
              className="w-40 bg-transparent text-xs text-slate-800 placeholder-slate-400 focus:outline-none sm:w-56"
            />
          </div>
        </div>

        {/* Vault Grid Content */}
        <div className="flex-1 overflow-y-auto p-6 bg-slate-50/30">
          <div className="grid grid-cols-1 gap-4 sm:grid-cols-2 lg:grid-cols-4">
            {filteredTriplets.map((t, idx) => {
              const { widthKm, heightKm } = footprintSizeKm(t.bounds);

              let thumbUrl = imageUrl(`/images/ohrc/${t.id}`);
              let badge = sensorBadge("ohrc", t);
              let targetView: "registration" | "linked-cursor" | "map" = "linked-cursor";

              if (filter === "tmc") {
                thumbUrl = imageUrl(`/images/tmc/${t.id}`);
                badge = sensorBadge("tmc", t);
                targetView = "linked-cursor";
              } else if (filter === "iirs") {
                thumbUrl = imageUrl(`/images/iirs/${t.id}`);
                badge = sensorBadge("iirs", t);
                targetView = "map";
              } else if (filter === "lro") {
                thumbUrl = imageUrl(`/images/lro_nac/${t.id}`);
                badge = sensorBadge("lro", t);
                targetView = "linked-cursor";
              } else if (filter === "qa") {
                thumbUrl = imageUrl(`/images/registered/${t.id}/blend_overlay.png`);
                badge = sensorBadge("qa", t);
                targetView = "registration";
              }

              return (
                <div
                  key={t.id}
                  className="group flex flex-col justify-between rounded-2xl border border-slate-200/80 bg-white p-4 shadow-sm transition-all hover:border-[#4F46E5]/40 hover:shadow-md"
                >
                  <div>
                    {/* Top line */}
                    <div className="mb-2 flex items-center justify-between">
                      <span className="font-mono text-[11px] text-slate-400">
                        #{String(idx + 1).padStart(2, "0")}
                      </span>
                      <span className="rounded-md border border-indigo-100 bg-indigo-50 px-2 py-0.5 text-[10px] font-bold text-[#4F46E5]">
                        {badge}
                      </span>
                    </div>

                    {/* Image Preview: contain-fit so non-square tiles (e.g. LRO
                        reference swaths) are never edge-cropped. */}
                    <div className="relative aspect-square overflow-hidden rounded-xl border border-slate-100 bg-black">
                      {filter === "lro" && failedThumbs.has(thumbUrl) ? (
                        <div className="flex h-full w-full flex-col items-center justify-center gap-1.5 p-4 text-center">
                          <span className="rounded-md border border-amber-300 bg-amber-50 px-2 py-0.5 text-[9px] font-black uppercase tracking-wider text-amber-700">
                            LRO reference unavailable
                          </span>
                          <span className="text-[10px] leading-relaxed text-slate-400">
                            Real CDR tile missing for {t.id} — no proxy substituted
                          </span>
                        </div>
                      ) : (
                        /* eslint-disable-next-line @next/next/no-img-element */
                        <img
                          src={thumbUrl}
                          alt={t.id}
                          className="h-full w-full object-contain transition-transform duration-300 group-hover:scale-105"
                          onError={(e) => {
                            if (filter === "lro") {
                              setFailedThumbs((prev) => new Set(prev).add(thumbUrl));
                            } else {
                              (e.currentTarget as HTMLImageElement).src = imageUrl(filter === "iirs" ? "/images/iirs/iirs_overlay.png" : `/images/ohrc/${t.id}`);
                            }
                          }}
                        />
                      )}
                    </div>

                    {/* Region Metadata */}
                    <div className="mt-3">
                      <h4 className="text-xs font-bold text-slate-900 group-hover:text-[#4F46E5] transition">
                        {t.id}
                      </h4>
                      <p className="mt-0.5 text-xs text-slate-500">
                        {widthKm.toFixed(1)} × {heightKm.toFixed(1)} km
                      </p>
                      <p className="mt-0.5 text-[11px] text-slate-400 font-mono">
                        {t.bounds.west_lon.toFixed(2)}°E, {t.bounds.north_lat.toFixed(2)}°N
                      </p>
                    </div>
                  </div>

                  {/* Actions */}
                  <div className="mt-4 pt-3 border-t border-slate-100 flex items-center justify-between">
                    <div className="flex items-center gap-2">
                      <span className="text-[10px] font-medium text-slate-500">
                        {t.dem_available ? "DEM available" : "Mono"}
                      </span>
                      <a
                        href={`${API_BASE}/api/registration/report/${t.id}`}
                        target="_blank"
                        rel="noopener noreferrer"
                        className="inline-flex items-center gap-1 rounded-md border border-slate-200 bg-slate-50 px-1.5 py-0.5 text-[10px] font-semibold text-slate-600 hover:bg-rose-50 hover:text-rose-700 hover:border-rose-200 transition-all"
                        title="Download ISRO Verification Report (PDF)"
                        onClick={(e) => e.stopPropagation()}
                      >
                        <svg className="h-2.5 w-2.5 text-rose-500" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 10v6m0 0l-3-3m3 3l3-3m2 8H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
                        </svg>
                        <span>PDF</span>
                      </a>
                    </div>
                    <button
                      onClick={() => {
                        onClose();
                        onSelectRegion(t.id, targetView);
                      }}
                      className="rounded-xl bg-[#4F46E5] hover:bg-[#4338CA] px-3 py-1.5 text-xs font-semibold text-white transition-all shadow-sm"
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
              {search ? (
                <>No regions matching "{search}".</>
              ) : filter === "lro" ? (
                <div className="space-y-3">
                  <p className="font-semibold text-slate-600">No external reference datasets matched current filters.</p>
                  <button
                    onClick={() => setFilter("all")}
                    className="rounded-xl bg-[#4F46E5] px-4 py-2 text-xs font-bold text-white shadow-sm hover:bg-[#4338CA] transition"
                  >
                    View All Payloads
                  </button>
                </div>
              ) : (
                <>No regions found.</>
              )}
            </div>
          )}
        </div>

        {/* Footer */}
        <div className="flex items-center justify-between border-t border-slate-100 bg-white px-6 py-3 text-xs text-slate-500">
          <span>Chandrayaan-2 Cross-Match Repository</span>
          <button
            onClick={onClose}
            className="hover:text-slate-900 font-medium transition-colors"
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
      className={`rounded-xl px-3.5 py-1.5 text-xs font-semibold transition-all ${
        active
          ? "bg-[#4F46E5] text-white shadow-sm"
          : "border border-slate-200 bg-white text-slate-600 hover:bg-slate-50 hover:text-slate-900"
      }`}
    >
      {label}
    </button>
  );
}
