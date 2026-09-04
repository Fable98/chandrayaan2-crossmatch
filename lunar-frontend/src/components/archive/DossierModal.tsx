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
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/70 p-4">
      <div className="relative flex max-h-[90vh] w-full max-w-4xl flex-col overflow-hidden rounded-lg border border-[#4A4A4A] bg-[#1a1d20] text-[#FFFFE3] shadow-xl">
        {/* Header Bar */}
        <div className="flex items-center justify-between border-b border-[#4A4A4A] bg-[#282c30] px-6 py-4">
          <div className="flex items-center gap-3">
            <span className="text-xs font-semibold uppercase tracking-wider text-[#4682B4]">
              Region Report
            </span>
            <span className="text-sm font-semibold text-[#FFFFE3]">
              {triplet.id}
            </span>
            <span className="rounded bg-[#1a1d20] px-2 py-0.5 text-xs text-[#CBCBCB]">
              {widthKm.toFixed(1)} × {heightKm.toFixed(1)} km
            </span>
          </div>

          <button
            onClick={onClose}
            className="flex h-7 w-7 items-center justify-center rounded border border-[#4A4A4A] bg-[#1a1d20] text-xs text-[#CBCBCB] transition-colors hover:border-[#565c63] hover:text-[#FFFFE3]"
            title="Close"
          >
            ✕
          </button>
        </div>

        {/* Content Body */}
        <div className="flex-1 overflow-y-auto p-6 md:p-8 space-y-6">
          {/* Top Metadata Block */}
          <div className="flex flex-col justify-between gap-4 border-b border-[#4A4A4A] pb-6 sm:flex-row sm:items-baseline">
            <div>
              <span className="text-xs font-medium uppercase tracking-wider text-[#a2a8b0]">
                Coordinates
              </span>
              <h2 className="mt-1 text-lg font-semibold text-[#FFFFE3]">
                Region {triplet.id}
              </h2>
            </div>

            <div className="text-xs text-[#CBCBCB]">
              <div>Lon: {triplet.bounds.west_lon.toFixed(4)}° to {triplet.bounds.east_lon.toFixed(4)}°</div>
              <div>Lat: {triplet.bounds.south_lat.toFixed(4)}° to {triplet.bounds.north_lat.toFixed(4)}°</div>
            </div>
          </div>

          {/* Triplet Imagery Quad */}
          <div>
            <span className="mb-3 block text-xs font-medium text-[#CBCBCB]">
              Sensor Imagery (OHRC · TMC-2 · IIRS)
            </span>
            <div className="grid grid-cols-1 gap-4 sm:grid-cols-2 lg:grid-cols-4">
              {/* OHRC */}
              <div className="flex flex-col gap-2 rounded-md border border-[#4A4A4A] bg-[#282c30] p-3">
                <div className="relative aspect-square overflow-hidden rounded bg-black">
                  {/* eslint-disable-next-line @next/next/no-img-element */}
                  <img
                    src={imageUrl(`/images/ohrc/${triplet.id}`)}
                    alt="OHRC high resolution"
                    className="h-full w-full object-cover"
                  />
                </div>
                <div className="text-xs">
                  <span className="text-[#FFFFE3] font-medium">OHRC Primary</span>
                  <p className="text-[11px] text-[#a2a8b0]">0.25–0.32 m/px</p>
                </div>
              </div>

              {/* TMC-2 */}
              <div className="flex flex-col gap-2 rounded-md border border-[#4A4A4A] bg-[#282c30] p-3">
                <div className="relative aspect-square overflow-hidden rounded bg-black">
                  {/* eslint-disable-next-line @next/next/no-img-element */}
                  <img
                    src={imageUrl(`/images/tmc/${triplet.id}`)}
                    alt="TMC-2 terrain stereo"
                    className="h-full w-full object-cover"
                  />
                </div>
                <div className="text-xs">
                  <span className="text-[#FFFFE3] font-medium">TMC-2 Reference</span>
                  <p className="text-[11px] text-[#a2a8b0]">~4–5 m/px stereo</p>
                </div>
              </div>

              {/* IIRS */}
              <div className="flex flex-col gap-2 rounded-md border border-[#4A4A4A] bg-[#282c30] p-3">
                <div className="relative aspect-square overflow-hidden rounded bg-black">
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
                  <span className="text-[#FFFFE3] font-medium">IIRS Hyperspectral</span>
                  <p className="text-[11px] text-[#a2a8b0]">~70–80 m/px</p>
                </div>
              </div>

              {/* Registered Blend */}
              <div className="flex flex-col gap-2 rounded-md border border-[#4A4A4A] bg-[#282c30] p-3">
                <div className="relative aspect-square overflow-hidden rounded bg-black">
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
                  <span className="text-[#4682B4] font-medium">50% Blend Overlay</span>
                  <p className="text-[11px] text-[#a2a8b0]">Homography transformed</p>
                </div>
              </div>
            </div>
          </div>

          {/* Scientific Metrics & Analysis */}
          <div className="grid grid-cols-1 gap-6 border-t border-[#4A4A4A] pt-6 sm:grid-cols-2">
            <div>
              <h4 className="text-xs font-semibold uppercase tracking-wider text-[#CBCBCB]">
                Registration Telemetry
              </h4>
              <div className="mt-3 space-y-2 text-xs text-[#CBCBCB]">
                <div className="flex justify-between border-b border-[#4A4A4A] pb-1.5">
                  <span>Sub-Pixel Status</span>
                  <span className="font-medium text-[#FFFFE3]">
                    {metrics?.sub_pixel_accurate ? "Verified (< 0.5 px)" : "Standard Alignment"}
                  </span>
                </div>
                <div className="flex justify-between border-b border-[#4A4A4A] pb-1.5">
                  <span>Root Mean Square Error</span>
                  <span className="font-medium text-[#4682B4]">
                    {metrics ? `${metrics.rmse_px.toFixed(3)} px` : "—"}
                  </span>
                </div>
                <div className="flex justify-between border-b border-[#4A4A4A] pb-1.5">
                  <span>Post-RANSAC Inliers</span>
                  <span className="font-medium text-[#FFFFE3]">
                    {metrics?.num_inliers ?? "—"} matches
                  </span>
                </div>
                <div className="flex justify-between border-b border-[#4A4A4A] pb-1.5">
                  <span>Combined Coverage Score</span>
                  <span className="font-medium text-[#FFFFE3]">
                    {metrics ? `${(metrics.combined_coverage_score * 100).toFixed(1)}%` : "—"}
                  </span>
                </div>
                <div className="flex justify-between pb-1">
                  <span>Elevation Layer</span>
                  <span className="font-medium text-[#FFFFE3]">
                    {triplet.dem_available ? "DEM Available (TMC DTM)" : "Interpolated"}
                  </span>
                </div>
              </div>
            </div>

            <div>
              <h4 className="text-xs font-semibold uppercase tracking-wider text-[#CBCBCB]">
                Registration Method
              </h4>
              <p className="mt-3 text-xs leading-relaxed text-[#CBCBCB]">
                Co-registration is performed using deep feature correspondence (LoFTR) coupled with iterative robust RANSAC estimation and phase-correlation sub-pixel refinement to eliminate parallax errors and illumination variations across passes.
              </p>
            </div>
          </div>
        </div>

        {/* Footer Actions */}
        <div className="flex items-center justify-between border-t border-[#4A4A4A] bg-[#282c30] px-6 py-4">
          <button
            onClick={onClose}
            className="rounded-md border border-[#4A4A4A] bg-[#1a1d20] px-4 py-2 text-xs text-[#CBCBCB] transition-colors hover:bg-[#4A4A4A] hover:text-[#FFFFE3]"
          >
            Close
          </button>

          <button
            onClick={() => {
              onClose();
              onOpenWorkspace(triplet.id);
            }}
            className="flex items-center gap-2 rounded-md bg-[#4682B4] px-4 py-2 text-xs font-semibold text-[#FFFFE3] transition-colors hover:bg-[#3a6d96]"
          >
            <span>Open in Workspace</span>
            <span>→</span>
          </button>
        </div>
      </div>
    </div>
  );
}
