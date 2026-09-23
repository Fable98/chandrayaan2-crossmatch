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
    <div className="fixed inset-0 z-[2000] flex items-center justify-center bg-black/60 p-3 sm:p-6 backdrop-blur-xs animate-fade-in font-mono">
      <div className="retro-outset relative flex max-h-[92vh] w-full max-w-6xl flex-col overflow-hidden bg-[#E7E2D6] dark:bg-[#1A201E] text-[#1E2321] dark:text-[#E7E2D6] shadow-2xl">
        {/* Retro Window Titlebar */}
        <div className="flex items-center justify-between bg-[#1F4743] px-3 py-1.5 text-xs font-bold font-mono text-white select-none shrink-0 border-b border-[#143532]">
          <div className="flex items-center gap-2">
            <span className="text-sm">🗄️</span>
            <span className="tracking-wider uppercase">
              ARCHIVE VAULT // LUNAR MULTI-SENSOR PRODUCTS
            </span>
            <span className="retro-inset px-2 py-0.5 text-[10px] font-mono font-bold text-black border-t-[#8B8579] border-l-[#8B8579] border-r-white border-b-white">
              [{filteredTriplets.length}/{triplets.length} REGIONS]
            </span>
          </div>

          <div className="flex items-center space-x-1">
            <button
              type="button"
              className="window-ctrl-btn"
              title="Minimize"
              tabIndex={-1}
            >
              _
            </button>
            <button
              type="button"
              className="window-ctrl-btn"
              title="Maximize"
              tabIndex={-1}
            >
              □
            </button>
            <button
              type="button"
              onClick={onClose}
              className="window-ctrl-btn hover:bg-rose-700 hover:text-white font-bold"
              title="Close [Esc]"
            >
              ✕
            </button>
          </div>
        </div>

        {/* Filter & Search Toolbar */}
        <div className="flex flex-wrap items-center justify-between gap-3 border-b border-[#8B8579] dark:border-[#2D3835] bg-[#DED8CB] dark:bg-[#141817] px-4 py-2.5 shrink-0">
          {/* Filter Buttons */}
          <div className="flex flex-wrap items-center gap-1.5">
            <span className="text-[11px] font-bold text-[#555C58] dark:text-[#8C9893] uppercase mr-1">Filter:</span>
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

          {/* Search Box */}
          <div className="flex items-center gap-2 retro-inset bg-white dark:bg-[#0A0D0C] px-2.5 py-1 text-xs text-[#1E2321] dark:text-[#E7E2D6]">
            <span className="text-[#79827D] text-xs">🔍</span>
            <input
              type="text"
              value={search}
              onChange={(e) => setSearch(e.target.value)}
              placeholder="Search regions by ID..."
              className="w-40 bg-transparent text-xs font-mono text-[#1E2321] dark:text-[#E7E2D6] placeholder-[#79827D] outline-none sm:w-56"
            />
            {search && (
              <button
                type="button"
                onClick={() => setSearch("")}
                className="text-[10px] text-[#79827D] hover:text-[#1E2321] dark:hover:text-white"
              >
                ✕
              </button>
            )}
          </div>
        </div>

        {/* Vault Grid Recessed Well */}
        <div className="flex-1 overflow-y-auto p-3 sm:p-4 bg-[#D6D0C2] dark:bg-[#0F1312] retro-inset m-2 sm:m-3">
          {deleteError && (
            <div className="retro-inset mb-3 border-l-4 border-l-rose-600 bg-rose-100 dark:bg-rose-950/50 p-2.5 text-xs font-mono text-rose-800 dark:text-rose-200">
              [SYSTEM ERROR] Delete failed: {deleteError}
            </div>
          )}

          <div className="grid grid-cols-1 gap-3 sm:grid-cols-2 lg:grid-cols-4">
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
                  className="retro-outset group flex flex-col justify-between p-3 bg-[#E7E2D6] dark:bg-[#1A201E] hover:border-[#1F4743] transition-colors"
                >
                  <div>
                    {/* Top line */}
                    <div className="mb-2 flex items-center justify-between">
                      <span className="font-mono text-xs font-bold text-[#555C58] dark:text-[#8C9893]">
                        #{String(idx + 1).padStart(2, "0")}
                      </span>
                      <span className="retro-inset px-2 py-0.5 text-[10px] font-mono font-bold bg-[#F6F3EC] dark:bg-[#141817] text-[#1F4743] dark:text-teal-300">
                        {badge}
                      </span>
                    </div>

                    {/* Image Preview */}
                    <div className="retro-inset relative aspect-square overflow-hidden bg-[#0A0D0C] p-1 flex items-center justify-center">
                      {filter === "lro" && failedThumbs.has(thumbUrl) ? (
                        <div className="flex h-full w-full flex-col items-center justify-center gap-1.5 p-3 text-center">
                          <span className="retro-inset px-1.5 py-0.5 text-[9px] font-black uppercase tracking-wider bg-amber-100 dark:bg-amber-950/60 text-amber-800 dark:text-amber-300 border border-amber-500">
                            LRO UNAVAILABLE
                          </span>
                          <span className="text-[10px] leading-relaxed text-[#8C9893]">
                            Real CDR tile missing for {t.id}
                          </span>
                        </div>
                      ) : (
                        /* eslint-disable-next-line @next/next/no-img-element */
                        <img
                          src={thumbUrl}
                          alt={t.id}
                          className="h-full w-full object-contain transition-transform duration-200 group-hover:scale-102"
                          onError={(e) => {
                            if (filter === "lro") {
                              setFailedThumbs((prev) => new Set(prev).add(thumbUrl));
                            } else {
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
                    <div className="mt-2.5">
                      <h4 className="text-xs font-bold font-mono text-[#1E2321] dark:text-[#E7E2D6] truncate">
                        {t.id}
                      </h4>
                      <p className="mt-0.5 text-xs font-mono text-[#555C58] dark:text-[#8C9893]">
                        {widthKm.toFixed(1)} × {heightKm.toFixed(1)} km
                      </p>
                      <p className="mt-0.5 text-[11px] text-[#79827D] dark:text-[#697571] font-mono">
                        {t.bounds.west_lon.toFixed(2)}°E, {t.bounds.north_lat.toFixed(2)}°N
                      </p>
                    </div>
                  </div>

                  {/* Actions Toolbar */}
                  <div className="mt-3 pt-2.5 border-t border-[#B9B2A5] dark:border-[#2D3835] flex items-center justify-between gap-1.5">
                    <div className="flex items-center gap-1.5">
                      <span className="retro-inset px-1.5 py-0.5 text-[9px] font-mono font-bold bg-[#EFEBE0] dark:bg-[#141817] text-[#555C58] dark:text-[#8C9893]">
                        {t.dem_available ? "DEM 3D" : "MONO"}
                      </span>
                      <a
                        href={`${API_BASE}/api/registration/report/${t.id}`}
                        target="_blank"
                        rel="noopener noreferrer"
                        className="retro-button inline-flex items-center gap-1 px-2 py-0.5 text-[10px] font-mono font-bold text-rose-700 dark:text-rose-400 hover:bg-rose-100 dark:hover:bg-rose-950/40"
                        title="Download ISRO Verification Report (PDF)"
                        onClick={(e) => e.stopPropagation()}
                      >
                        <span>📄</span>
                        <span>PDF</span>
                      </a>
                    </div>

                    <div className="flex items-center gap-1.5">
                      <button
                        onClick={() => {
                          onClose();
                          onSelectRegion(t.id, targetView);
                        }}
                        className="retro-button-primary px-2.5 py-1 text-xs font-mono font-bold flex items-center gap-1"
                      >
                        <span>Inspect</span>
                        <span>→</span>
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
                          className="retro-button px-2 py-1 text-xs font-mono font-bold text-rose-700 dark:text-rose-400 hover:bg-rose-100 dark:hover:bg-rose-950/40 disabled:opacity-50"
                        >
                          {deletingId === t.id ? "…" : "🗑"}
                        </button>
                      )}
                    </div>
                  </div>
                </div>
              );
            })}
          </div>

          {filteredTriplets.length === 0 && (
            <div className="retro-outset p-10 text-center text-xs font-mono text-[#555C58] dark:text-[#8C9893] bg-[#E7E2D6] dark:bg-[#1A201E] my-6">
              {search ? (
                <>No region matches search query "{search}".</>
              ) : filter === "lro" ? (
                <div className="space-y-3">
                  <p className="font-bold text-[#1E2321] dark:text-[#E7E2D6]">
                    No external NASA LRO reference datasets matched active filter.
                  </p>
                  <button
                    onClick={() => setFilter("all")}
                    className="retro-button-primary px-4 py-1.5 text-xs font-mono font-bold"
                  >
                    View All Payloads
                  </button>
                </div>
              ) : (
                <>No archive products found.</>
              )}
            </div>
          )}
        </div>

        {/* Retro Window Bottom Status Bar */}
        <div className="flex items-center justify-between border-t border-[#8B8579] dark:border-[#2D3835] bg-[#DED8CB] dark:bg-[#141817] px-4 py-2 text-xs font-mono text-[#555C58] dark:text-[#8C9893] shrink-0">
          <div className="flex items-center gap-2">
            <span className="w-2 h-2 bg-emerald-500 inline-block animate-pulse" />
            <span>Chandrayaan-2 Cross-Match Multi-Sensor Repository // SAC-ISRO</span>
          </div>
          <button
            onClick={onClose}
            className="retro-button px-3 py-1 text-xs font-mono font-bold text-[#1E2321] dark:text-[#E7E2D6]"
          >
            Close [Esc] ✕
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
      className={`px-2.5 py-1 text-xs font-mono font-bold ${
        active
          ? "retro-button-primary active-pressed"
          : "retro-button bg-[#E7E2D6] dark:bg-[#222927] text-[#1E2321] dark:text-[#E7E2D6] hover:bg-[#F0ECE1] dark:hover:bg-[#2D3835]"
      }`}
    >
      {label}
    </button>
  );
}


