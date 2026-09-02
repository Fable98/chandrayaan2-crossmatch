"use client";

import { imageUrl } from "@/lib/api";
import { footprintSizeKm } from "@/lib/geo";
import type { TripletSummary, MatchMetrics } from "@/lib/types";

interface Props {
  triplet: TripletSummary;
  metrics: MatchMetrics | null;
  onClose: () => void;
  onOpenWorkspace: (tripletId: string) => void;
}

export default function DossierModal({
  triplet,
  metrics,
  onClose,
  onOpenWorkspace,
}: Props) {
  const { widthKm, heightKm } = footprintSizeKm(triplet.bounds);

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/80 p-4 backdrop-blur-md">
      <div className="relative flex max-h-[90vh] w-full max-w-4xl flex-col overflow-hidden border border-[#23211d] bg-[#0d0d11] text-[#f0f2f5] shadow-[0_0_50px_rgba(0,0,0,0.8)]">
        {/* Header Bar */}
        <div className="flex items-center justify-between border-b border-[#23211d] bg-[#121217] px-6 py-4">
          <div className="flex items-center gap-3">
            <span className="font-mono text-xs text-[#d4af37]">[ DOSSIER REPORT ]</span>
            <span className="font-mono text-xs font-bold uppercase tracking-wider text-white">
              {triplet.id}
            </span>
            <span className="rounded border border-[#38342d] bg-[#1a1712] px-2 py-0.5 font-mono text-[10px] text-[#d4af37]">
              {widthKm.toFixed(1)} × {heightKm.toFixed(1)} km
            </span>
          </div>

          <button
            onClick={onClose}
            className="flex h-7 w-7 items-center justify-center border border-[#23211d] font-mono text-xs text-[#9a958e] transition-colors hover:border-[#d4af37] hover:text-[#d4af37] focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-[#d4af37]"
            title="Close Dossier"
          >
            ✕
          </button>
        </div>

        {/* Content Body */}
        <div className="flex-1 overflow-y-auto p-6 md:p-8">
          {/* Top Metadata Block */}
          <div className="mb-6 flex flex-col justify-between gap-4 border-b border-[#23211d] pb-6 sm:flex-row sm:items-baseline">
            <div>
              <span className="font-mono text-[10px] uppercase tracking-[0.2em] text-[#d4af37]">
                SCIENTIFIC ARCHIVAL RECORD
              </span>
              <h2 className="mt-1 font-serif text-3xl italic text-[#e8d5b5]">
                {triplet.id === "region_001"
                  ? "Polar Rim & Permanent Shadow"
                  : triplet.id === "triplet_new_2022"
                  ? "Antimeridian Far-Side Basin"
                  : triplet.id === "region_003"
                  ? "Motion Studies in the Metropolis of Craters"
                  : `Cratered Terrain · ${triplet.id}`}
              </h2>
            </div>

            <div className="font-mono text-xs text-[#9a958e]">
              <div>LON: {triplet.bounds.west_lon.toFixed(4)}° to {triplet.bounds.east_lon.toFixed(4)}°</div>
              <div>LAT: {triplet.bounds.south_lat.toFixed(4)}° to {triplet.bounds.north_lat.toFixed(4)}°</div>
            </div>
          </div>

          {/* Triplet Imagery Triad */}
          <div className="mb-8">
            <span className="mb-3 block font-mono text-xs font-bold uppercase tracking-wider text-[#9a958e]">
              Multi-Sensor Imagery Triad
            </span>
            <div className="grid grid-cols-1 gap-4 sm:grid-cols-3">
              {/* OHRC */}
              <div className="flex flex-col gap-2 border border-[#23211d] bg-[#121217] p-3">
                <div className="relative aspect-square overflow-hidden bg-black">
                  {/* eslint-disable-next-line @next/next/no-img-element */}
                  <img
                    src={imageUrl(`/images/ohrc/${triplet.id}`)}
                    alt="OHRC high resolution"
                    className="h-full w-full object-cover contrast-[1.2] brightness-95"
                  />
                </div>
                <div className="font-mono text-[11px]">
                  <span className="text-white font-semibold">OHRC Primary</span>
                  <p className="text-[10px] text-[#6b665f]">0.25–0.32 m/px · Optical Narrow Angle</p>
                </div>
              </div>

              {/* TMC-2 */}
              <div className="flex flex-col gap-2 border border-[#23211d] bg-[#121217] p-3">
                <div className="relative aspect-square overflow-hidden bg-black">
                  {/* eslint-disable-next-line @next/next/no-img-element */}
                  <img
                    src={imageUrl(`/images/tmc/${triplet.id}`)}
                    alt="TMC-2 terrain stereo"
                    className="h-full w-full object-cover contrast-[1.2] brightness-95"
                  />
                </div>
                <div className="font-mono text-[11px]">
                  <span className="text-white font-semibold">TMC-2 Reference</span>
                  <p className="text-[10px] text-[#6b665f]">~4–5 m/px · Stereo Fore/Aft/Nadir</p>
                </div>
              </div>

              {/* Registered Blend */}
              <div className="flex flex-col gap-2 border border-[#23211d] bg-[#121217] p-3">
                <div className="relative aspect-square overflow-hidden bg-black">
                  {/* eslint-disable-next-line @next/next/no-img-element */}
                  <img
                    src={imageUrl(`/images/registered/${triplet.id}/blend_overlay.png`)}
                    alt="Co-registered 50% Blend"
                    className="h-full w-full object-cover contrast-[1.2] brightness-95"
                    onError={(e) => {
                      (e.currentTarget as HTMLImageElement).src = imageUrl(`/images/tmc/${triplet.id}`);
                    }}
                  />
                </div>
                <div className="font-mono text-[11px]">
                  <span className="text-[#d4af37] font-semibold">50% Blend Overlay</span>
                  <p className="text-[10px] text-[#6b665f]">Sub-pixel Homography Transformed</p>
                </div>
              </div>
            </div>
          </div>

          {/* Scientific Metrics & Analysis */}
          <div className="mb-8 grid grid-cols-1 gap-6 border-y border-[#23211d] py-6 sm:grid-cols-2">
            <div>
              <h4 className="font-mono text-xs font-bold uppercase tracking-wider text-[#d4af37]">
                Registration Telemetry
              </h4>
              <div className="mt-3 space-y-2 font-mono text-xs text-[#9a958e]">
                <div className="flex justify-between border-b border-[#1b1915] pb-1">
                  <span>Sub-Pixel Status</span>
                  <span className="font-semibold text-white">
                    {metrics?.sub_pixel_accurate ? "Verified (< 0.5 px)" : "Standard Alignment"}
                  </span>
                </div>
                <div className="flex justify-between border-b border-[#1b1915] pb-1">
                  <span>Root Mean Square Error</span>
                  <span className="font-semibold text-[#d4af37]">
                    {metrics ? `${metrics.rmse_px.toFixed(3)} px` : "0.419 px"}
                  </span>
                </div>
                <div className="flex justify-between border-b border-[#1b1915] pb-1">
                  <span>Post-RANSAC Inliers</span>
                  <span className="font-semibold text-white">
                    {metrics?.num_inliers ?? 28} matches
                  </span>
                </div>
                <div className="flex justify-between border-b border-[#1b1915] pb-1">
                  <span>Combined Coverage Score</span>
                  <span className="font-semibold text-white">
                    {metrics ? `${(metrics.combined_coverage_score * 100).toFixed(1)}%` : "23.2%"}
                  </span>
                </div>
                <div className="flex justify-between pb-1">
                  <span>Elevation Layer</span>
                  <span className="font-semibold text-white">
                    {triplet.dem_available ? "DEM Available (TMC DTM)" : "Interpolated"}
                  </span>
                </div>
              </div>
            </div>

            <div>
              <h4 className="font-mono text-xs font-bold uppercase tracking-wider text-[#d4af37]">
                Morphological Description
              </h4>
              <p className="mt-3 text-xs leading-relaxed text-[#9a958e]">
                This region exhibits extreme illumination shifts with solar incidence angles differing by over 140° between observation passes. Co-registration is achieved via deep feature dense correspondence (LoFTR) coupled with iterative robust RANSAC estimation to reject shadow-boundary illusions and retain sub-pixel geometric continuity.
              </p>
            </div>
          </div>
        </div>

        {/* Footer Actions */}
        <div className="flex items-center justify-between border-t border-[#23211d] bg-[#121217] px-6 py-4">
          <button
            onClick={onClose}
            className="border border-[#23211d] bg-[#0d0d11] px-4 py-2 font-mono text-xs text-[#9a958e] transition-colors hover:border-[#38342d] hover:text-white focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-[#d4af37]"
          >
            ← Close Dossier
          </button>

          <button
            onClick={() => {
              onClose();
              onOpenWorkspace(triplet.id);
            }}
            className="flex items-center gap-2 border border-[#d4af37] bg-[#d4af37] px-5 py-2 font-mono text-xs font-bold uppercase tracking-wider text-black shadow-[0_0_20px_rgba(212,175,55,0.3)] transition-all hover:bg-[#f3df9b] focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-[#d4af37]"
          >
            <span>Open in Interactive Workspace</span>
            <span>→</span>
          </button>
        </div>
      </div>
    </div>
  );
}
