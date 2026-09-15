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
  onDeleteRegion?: (tripletId: string) => Promise<void>;
}

export default function VaultModal({
  triplets,
  initialFilter = "all",
  onClose,
  onSelectRegion,
  onDeleteRegion,
}: Props) {
  const [filter, setFilter] = useState<PayloadFilter>(initialFilter);
  const [search, setSearch] = useState("");
  const [deletingId, setDeletingId] = useState<string | null>(null);
  const [deleteError, setDeleteError] = useState<string | null>(null);
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
      <div className="relative flex max-h-[90vh] w-full max-w-6xl flex-col overflow-hidden rounded-2xl border border-slate-200/80 dark:border-[#1b2029] bg-white dark:bg-[#0e1117] text-slate-800 dark:text-slate-200 shadow-2xl">
        {/* Header Bar */}
        <div className="flex flex-wrap items-center justify-between gap-4 border-b border-slate-100 dark:border-[#1b2029] bg-white dark:bg-[#0e1117] px-6 py-4">
          <div className="flex items-center gap-3">
            <div>
              <div className="flex items-center gap-2">
                <span className="rounded-lg border border-indigo-100 dark:border-indigo-900/50 bg-indigo-50 dark:bg-indigo-950/30 px-2.5 py-0.5 text-[11px] font-bold uppercase tracking-wider text-[#4F46E5] dark:text-indigo-300">
                  Archive Vault
                </span>
                <span className="text-base font-bold text-slate-900 dark:text-slate-100">
                  Lunar Cross-Match Products
                </span>
              </div>
              <p className="mt-0.5 text-xs text-slate-500 dark:text-slate-400">
                {triplets.length} validated regions · Chandrayaan-2 multi-sensor archive
              </p>
            </div>
          </div>

          <button
            onClick={onClose}
            className="flex h-7 w-7 items-center justify-center rounded-lg border border-slate-200 dark:border-[#1b2029] bg-white dark:bg-white/5 text-xs text-slate-400 transition hover:bg-slate-50 dark:hover:bg-white/10 hover:text-slate-700 dark:hover:text-slate-200"
            title="Close"
          >
            ✕
          </button>
        </div>

        {/* Filter & Search Toolbar */}
        <div className="flex flex-wrap items-center justify-between gap-4 border-b border-slate-100 dark:border-[#1b2029] bg-slate-50/50 dark:bg-white/5 px-6 py-3">
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
          <div className="flex items-center gap-2 rounded-xl border border-slate-200 dark:border-[#1b2029] bg-white dark:bg-white/5 px-3.5 py-1.5 text-xs text-slate-700 dark:text-slate-200 focus-within:border-[#4F46E5]">
            <svg className="h-3.5 w-3.5 text-slate-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0z" />
            </svg>
            <input
              type="text"
              value={search}
              onChange={(e) => setSearch(e.target.value)}
              placeholder="Search regions…"
              className="w-40 bg-transparent text-xs text-slate-800 dark:text-slate-200 placeholder-slate-400 focus:outline-none sm:w-56"
            />
          </div>
        </div>

        {/* Vault Grid Content */}
        <div className="flex-1 overflow-y-auto p-6 bg-slate-50/30 dark:bg-white/5">
          {deleteError && (
            <div className="mb-4 rounded-xl border border-rose-200 dark:border-rose-900/50 bg-rose-50 dark:bg-rose-950/30 px-4 py-2.5 text-xs font-medium text-rose-700 dark:text-rose-300">
              Delete failed: {deleteError}
            </div>
          )}
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
                  className="group flex flex-col justify-between rounded-2xl border border-slate-200/80 dark:border-[#1b2029] bg-white dark:bg-white/5 p-4 shadow-sm transition-all hover:border-[#4F46E5]/40 hover:shadow-md"
                >
                  <div>
                    {/* Top line */}
                    <div className="mb-2 flex items-center justify-between">
                      <span className="font-mono text-[11px] text-slate-400">
                        #{String(idx + 1).padStart(2, "0")}
                      </span>
                      <span className="rounded-md border border-indigo-100 dark:border-indigo-900/50 bg-indigo-50 dark:bg-indigo-950/30 px-2 py-0.5 text-[10px] font-bold text-[#4F46E5] dark:text-indigo-300">
                        {badge}
                      </span>
                    </div>

                    {/* Image Preview: contain-fit so non-square tiles (e.g. LRO
                        reference swaths) are never edge-cropped. */}
                    <div className="relative aspect-square overflow-hidden rounded-xl border border-slate-100 dark:border-[#1b2029] bg-black">
                      {filter === "lro" && failedThumbs.has(thumbUrl) ? (
                        <div className="flex h-full w-full flex-col items-center justify-center gap-1.5 p-4 text-center">
                          <span className="rounded-md border border-amber-300 dark:border-amber-900/50 bg-amber-50 dark:bg-amber-950/30 px-2 py-0.5 text-[9px] font-black uppercase tracking-wider text-amber-700 dark:text-amber-300">
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
                              // Guard against error-loop: only swap when the
                              // fallback differs from the URL that just failed
                              // (previously re-set the identical ohrc URL, so a
                              // genuine 404 retried forever on a black tile).
                              const fallback = imageUrl(filter === "iirs" ? "/images/iirs/iirs_overlay.png" : `/images/ohrc/${t.id}`);
                              const img = e.currentTarget as HTMLImageElement;
                              if (img.src !== fallback && !img.dataset.fbk) {
                                img.dataset.fbk = "1";
                                img.src = fallback;
                              } else {
                                img.style.opacity = "0.25";
                              }
                            }
                          }}
                        />
                      )}
                    </div>

                    {/* Region Metadata */}
                    <div className="mt-3">
                      <h4 className="text-xs font-bold text-slate-900 dark:text-slate-100 group-hover:text-[#4F46E5] dark:group-hover:text-indigo-300 transition">
                        {t.id}
                      </h4>
                      <p className="mt-0.5 text-xs text-slate-500 dark:text-slate-400">
                        {widthKm.toFixed(1)} × {heightKm.toFixed(1)} km
                      </p>
                      <p className="mt-0.5 text-[11px] text-slate-400 font-mono">
                        {t.bounds.west_lon.toFixed(2)}°E, {t.bounds.north_lat.toFixed(2)}°N
                      </p>
                    </div>
                  </div>

                  {/* Actions */}
                  <div className="mt-4 pt-3 border-t border-slate-100 dark:border-[#1b2029] flex items-center justify-between">
                    <div className="flex items-center gap-2">
                      <span className="text-[10px] font-medium text-slate-500 dark:text-slate-400">
                        {t.dem_available ? "DEM available" : "Mono"}
                      </span>
                      <a
                        href={`${API_BASE}/api/registration/report/${t.id}`}
                        target="_blank"
                        rel="noopener noreferrer"
                        className="inline-flex items-center gap-1 rounded-md border border-slate-200 dark:border-[#1b2029] bg-slate-50 dark:bg-white/5 px-1.5 py-0.5 text-[10px] font-semibold text-slate-600 dark:text-slate-300 hover:bg-rose-50 dark:hover:bg-rose-950/30 hover:text-rose-700 dark:hover:text-rose-300 hover:border-rose-200 dark:hover:border-rose-900/50 transition-all"
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
                    {onDeleteRegion && (
                      <button
                        disabled={deletingId === t.id}
                        title={/^region_\d+$/i.test(t.id) ? "Delete curated seed (requires force confirm)" : "Delete this dataset"}
                        onClick={async (e) => {
                          e.stopPropagation();
                          const curated = /^region_\d+$/i.test(t.id);
                          const ok = window.confirm(
                            curated
                              ? `Delete curated seed '${t.id}'? Its on-disk bundle will be removed (restorable via git). Continue?`
                              : `Delete dataset '${t.id}'? Its tiles, matches and registration products will be removed from disk. Continue?`
                          );
                          if (!ok) return;
                          setDeleteError(null);
                          setDeletingId(t.id);
                          try {
                            await onDeleteRegion(t.id);
                          } catch (err) {
                            setDeleteError(err instanceof Error ? err.message : "Delete failed");
                          } finally {
                            setDeletingId(null);
                          }
                        }}
                        className="rounded-xl border border-slate-200 dark:border-[#1b2029] bg-white dark:bg-white/5 px-2.5 py-1.5 text-xs font-semibold text-slate-400 hover:bg-rose-50 dark:hover:bg-rose-950/30 hover:text-rose-700 dark:hover:text-rose-300 hover:border-rose-200 dark:hover:border-rose-900/50 transition-all shadow-sm disabled:opacity-50"
                      >
                        {deletingId === t.id ? "…" : "🗑"}
                      </button>
                    )}
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
                  <p className="font-semibold text-slate-600 dark:text-slate-300">No external reference datasets matched current filters.</p>
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
        <div className="flex items-center justify-between border-t border-slate-100 dark:border-[#1b2029] bg-white dark:bg-[#0e1117] px-6 py-3 text-xs text-slate-500 dark:text-slate-400">
          <span>Chandrayaan-2 Cross-Match Repository</span>
          <button
            onClick={onClose}
            className="hover:text-slate-900 dark:hover:text-slate-100 font-medium transition-colors"
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
          : "border border-slate-200 dark:border-[#1b2029] bg-white dark:bg-white/5 text-slate-600 dark:text-slate-300 hover:bg-slate-50 dark:hover:bg-white/10 hover:text-slate-900 dark:hover:text-slate-100"
      }`}
    >
      {label}
    </button>
  );
}
