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
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/80 p-4 backdrop-blur-2xl animate-fade-in">
      <div className="relative flex max-h-[90vh] w-full max-w-4xl flex-col overflow-hidden rounded-3xl border border-white/15 bg-[#0c0e24]/90 text-white shadow-[0_30px_90px_rgba(0,0,0,0.9)] ring-1 ring-white/10">
        {/* Header Bar */}
        <div className="flex items-center justify-between border-b border-white/10 bg-white/[0.02] px-6 py-4 backdrop-blur-md">
          <div className="flex items-center gap-3">
            <span className="rounded-full border border-purple-400/30 bg-purple-500/10 px-2.5 py-0.5 text-[10px] font-semibold uppercase tracking-wider text-purple-300">
              Region Dossier
            </span>
            <span className="text-sm font-bold text-white">
              {triplet.id}
            </span>
            <span className="rounded-full border border-white/10 bg-white/[0.04] px-2 py-0.5 text-xs text-slate-300 font-mono">
              {widthKm.toFixed(1)} × {heightKm.toFixed(1)} km
            </span>
          </div>

          <button
            onClick={onClose}
            className="flex h-7 w-7 items-center justify-center rounded-full border border-white/10 bg-white/[0.04] text-xs text-slate-300 transition hover:bg-white/[0.08] hover:text-white"
            title="Close"
          >
            ✕
          </button>
        </div>

        {/* Content Body */}
        <div className="flex-1 overflow-y-auto p-6 md:p-8 space-y-6">
          {/* Top Metadata Block */}
          <div className="flex flex-col justify-between gap-4 border-b border-white/[0.08] pb-5 sm:flex-row sm:items-baseline">
            <div>
              <span className="text-[10px] font-semibold uppercase tracking-wider text-purple-300">
                Orbital Track Coordinates
              </span>
              <h2 className="mt-1 text-xl font-bold text-white">
                Region {triplet.id}
              </h2>
            </div>

            <div className="text-xs text-slate-300 font-mono space-y-0.5">
              <div>Lon: {triplet.bounds.west_lon.toFixed(4)}° to {triplet.bounds.east_lon.toFixed(4)}°</div>
              <div>Lat: {triplet.bounds.south_lat.toFixed(4)}° to {triplet.bounds.north_lat.toFixed(4)}°</div>
            </div>
          </div>

          {/* Triplet Imagery Quad */}
          <div>
            <span className="mb-3 block text-xs font-semibold text-slate-300 uppercase tracking-wider">
              Multimodal Sensor Imagery (OHRC · TMC-2 · IIRS)
            </span>
            <div className="grid grid-cols-1 gap-4 sm:grid-cols-2 lg:grid-cols-4">
              {/* OHRC */}
              <div className="flex flex-col gap-2 rounded-2xl border border-white/10 bg-white/[0.03] p-3 backdrop-blur-xl">
                <div className="relative aspect-square overflow-hidden rounded-xl bg-black border border-white/[0.06]">
                  {/* eslint-disable-next-line @next/next/no-img-element */}
                  <img
                    src={imageUrl(`/images/ohrc/${triplet.id}`)}
                    alt="OHRC high resolution"
                    className="h-full w-full object-cover"
                  />
                </div>
                <div className="text-xs">
                  <span className="text-white font-semibold">OHRC Primary</span>
                  <p className="text-[10px] text-slate-400">0.25–0.32 m/px Optical</p>
                </div>
              </div>

              {/* TMC-2 */}
              <div className="flex flex-col gap-2 rounded-2xl border border-white/10 bg-white/[0.03] p-3 backdrop-blur-xl">
                <div className="relative aspect-square overflow-hidden rounded-xl bg-black border border-white/[0.06]">
                  {/* eslint-disable-next-line @next/next/no-img-element */}
                  <img
                    src={imageUrl(`/images/tmc/${triplet.id}`)}
                    alt="TMC-2 terrain stereo"
                    className="h-full w-full object-cover"
                  />
                </div>
                <div className="text-xs">
                  <span className="text-white font-semibold">TMC-2 Reference</span>
                  <p className="text-[10px] text-slate-400">~4–5 m/px Stereo</p>
                </div>
              </div>

              {/* IIRS */}
              <div className="flex flex-col gap-2 rounded-2xl border border-white/10 bg-white/[0.03] p-3 backdrop-blur-xl">
                <div className="relative aspect-square overflow-hidden rounded-xl bg-black border border-white/[0.06]">
                  {/* eslint-disable-next-line @next/next/no-img-element */}
                  <img
                    src={imageUrl(`/images/iirs/${triplet.id}`)}
                    alt="IIRS infrared hyperspectral"
                    className="h-full w-full object-cover"
                    onError={(e) => {
                      (e.currentTarget as HTMLImageElement).src = imageUrl("/images/iirs/iirs_overlay.png");
                    }}
                  />
                </div>
                <div className="text-xs">
                  <span className="text-white font-semibold">IIRS Hyperspectral</span>
                  <p className="text-[10px] text-slate-400">~70–80 m/px 256-Band</p>
                </div>
              </div>

              {/* Registered Blend */}
              <div className="flex flex-col gap-2 rounded-2xl border border-white/10 bg-white/[0.03] p-3 backdrop-blur-xl">
                <div className="relative aspect-square overflow-hidden rounded-xl bg-black border border-white/[0.06]">
                  {/* eslint-disable-next-line @next/next/no-img-element */}
                  <img
                    src={imageUrl(`/images/registered/${triplet.id}/blend_overlay.png`)}
                    alt="Co-registered 50% Blend"
                    className="h-full w-full object-cover"
                    onError={(e) => {
                      (e.currentTarget as HTMLImageElement).src = imageUrl(`/images/tmc/${triplet.id}`);
                    }}
                  />
                </div>
                <div className="text-xs">
                  <span className="text-purple-300 font-semibold">50% Blend Overlay</span>
                  <p className="text-[10px] text-slate-400">Sub-pixel Homography</p>
                </div>
              </div>
            </div>
          </div>

          {/* Scientific Metrics & Analysis */}
          <div className="grid grid-cols-1 gap-6 border-t border-white/[0.08] pt-6 sm:grid-cols-2">
            <div className="rounded-2xl border border-white/10 bg-white/[0.025] p-5">
              <h4 className="text-xs font-bold uppercase tracking-wider text-purple-300">
                Registration Telemetry
              </h4>
              <div className="mt-3 space-y-2.5 text-xs text-slate-300 font-mono">
                <div className="flex justify-between border-b border-white/[0.05] pb-2">
                  <span className="font-sans">Sub-Pixel Status</span>
                  <span className="font-bold text-emerald-400">
                    {metrics?.sub_pixel_accurate ? "Verified (< 0.5 px)" : "Standard Alignment"}
                  </span>
                </div>
                <div className="flex justify-between border-b border-white/[0.05] pb-2">
                  <span className="font-sans">Root Mean Square Error</span>
                  <span className="font-bold text-purple-300">
                    {metrics ? `${metrics.rmse_px.toFixed(3)} px` : "—"}
                  </span>
                </div>
                <div className="flex justify-between border-b border-white/[0.05] pb-2">
                  <span className="font-sans">Post-RANSAC Inliers</span>
                  <span className="font-bold text-white">
                    {metrics?.num_inliers ?? "—"} matches
                  </span>
                </div>
                <div className="flex justify-between border-b border-white/[0.05] pb-2">
                  <span className="font-sans">Combined Coverage Score</span>
                  <span className="font-bold text-white">
                    {metrics ? `${(metrics.combined_coverage_score * 100).toFixed(1)}%` : "—"}
                  </span>
                </div>
                <div className="flex justify-between">
                  <span className="font-sans">Elevation Layer</span>
                  <span className="font-bold text-white">
                    {triplet.dem_available ? "DEM Available (TMC DTM)" : "Interpolated"}
                  </span>
                </div>
              </div>
            </div>

            <div className="rounded-2xl border border-white/10 bg-white/[0.025] p-5">
              <h4 className="text-xs font-bold uppercase tracking-wider text-purple-300">
                Registration Method
              </h4>
              <p className="mt-3 text-xs leading-relaxed text-slate-300">
                Co-registration is performed using deep feature correspondence (LoFTR) coupled with iterative robust RANSAC estimation and phase-correlation sub-pixel refinement to eliminate parallax errors and extreme illumination angle variations across observation passes.
              </p>
            </div>
          </div>
        </div>

        {/* Footer Actions */}
        <div className="flex items-center justify-between border-t border-white/10 bg-white/[0.02] px-6 py-4">
          <button
            onClick={onClose}
            className="rounded-full border border-white/10 bg-white/[0.04] px-4 py-2 text-xs text-slate-300 transition hover:bg-white/[0.08] hover:text-white"
          >
            Close
          </button>

          <button
            onClick={() => {
              onClose();
              onOpenWorkspace(triplet.id);
            }}
            className="flex items-center gap-2 rounded-full bg-gradient-to-r from-purple-600 via-indigo-600 to-purple-600 px-5 py-2 text-xs font-bold text-white shadow-[0_0_20px_rgba(168,85,247,0.35)] transition hover:brightness-110"
          >
            <span>Open in Workspace</span>
            <span>→</span>
          </button>
        </div>
      </div>
    </div>
  );
}
