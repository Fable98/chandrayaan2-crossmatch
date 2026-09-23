"use client";

import { API_BASE, imageUrl } from "@/lib/api";
import { footprintSizeKm } from "@/lib/geo";
import type { TripletSummary, MatchMetrics } from "@/lib/types";
import { SENSOR_META, DOSSIER_SENSOR_LABELS, scaleRatioLabel, sensorMeta } from "@/lib/sensors";

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
    <div className="fixed inset-0 z-[2000] flex items-center justify-center bg-black/60 p-3 sm:p-6 backdrop-blur-xs animate-fade-in font-mono">
      <div className="retro-outset relative flex max-h-[92vh] w-full max-w-4xl flex-col overflow-hidden bg-[#E7E2D6] dark:bg-[#1A201E] text-[#1E2321] dark:text-[#E7E2D6] shadow-2xl">
        {/* Retro Window Titlebar */}
        <div className="flex items-center justify-between bg-[#1F4743] px-3 py-1.5 text-xs font-bold font-mono text-white select-none shrink-0 border-b border-[#143532]">
          <div className="flex items-center gap-2">
            <span className="text-sm">📑</span>
            <span className="tracking-wider uppercase">
              REGION DOSSIER // {triplet.id}
            </span>
            <span className="retro-inset px-2 py-0.5 text-[10px] font-mono font-bold text-black border-t-[#8B8579] border-l-[#8B8579] border-r-white border-b-white">
              {widthKm.toFixed(1)} × {heightKm.toFixed(1)} KM
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

        {/* Content Body */}
        <div className="flex-1 overflow-y-auto p-4 sm:p-6 space-y-4">
          {/* Top Metadata Block */}
          <div className="flex flex-col justify-between gap-3 border-b border-[#8B8579] dark:border-[#2D3835] pb-3 sm:flex-row sm:items-baseline">
            <div>
              <span className="text-[10px] font-bold uppercase tracking-wider text-[#1F4743] dark:text-teal-300">
                ORBITAL TRACK TELEMETRY
              </span>
              <h2 className="mt-0.5 text-lg font-bold text-[#1E2321] dark:text-[#E7E2D6]">
                Target Region: {triplet.id}
              </h2>
            </div>

            <div className="flex items-center gap-3">
              <div className="text-xs text-[#555C58] dark:text-[#8C9893] font-mono space-y-0.5 text-right">
                <div>LON: {triplet.bounds.west_lon.toFixed(4)}° TO {triplet.bounds.east_lon.toFixed(4)}°</div>
                <div>LAT: {triplet.bounds.south_lat.toFixed(4)}° TO {triplet.bounds.north_lat.toFixed(4)}°</div>
              </div>
              <a
                href={`${API_BASE}/api/registration/report/${triplet.id}`}
                target="_blank"
                rel="noopener noreferrer"
                className="retro-button inline-flex items-center gap-1.5 px-2.5 py-1 text-xs font-mono font-bold text-rose-700 dark:text-rose-400 hover:bg-rose-100 dark:hover:bg-rose-950/40"
                title="Download official ISRO Verification Report PDF for this region"
              >
                <span>📄</span>
                <span>ISRO Report</span>
              </a>
            </div>
          </div>

          {/* Triplet Imagery Quad / Pentad */}
          <div>
            <span className="mb-2 block text-xs font-bold text-[#555C58] dark:text-[#8C9893] uppercase tracking-wider">
              Multimodal Sensor Imagery (OHRC · TMC-2 · IIRS{triplet.lro_nac_available ? " · NASA LRO NAC" : ""})
            </span>
            <div className={`grid grid-cols-1 gap-3 sm:grid-cols-2 ${triplet.lro_nac_available ? "lg:grid-cols-5" : "lg:grid-cols-4"}`}>
              {/* OHRC */}
              <div className="retro-outset flex flex-col gap-2 p-2 bg-[#E7E2D6] dark:bg-[#1A201E]">
                <div className="retro-inset relative aspect-square overflow-hidden bg-black p-0.5">
                  {/* eslint-disable-next-line @next/next/no-img-element */}
                  <img
                    src={imageUrl(`/images/ohrc/${triplet.id}`)}
                    alt="OHRC high resolution"
                    className="h-full w-full object-cover"
                  />
                </div>
                <div className="text-xs">
                  <span className="text-[#1E2321] dark:text-[#E7E2D6] font-bold block">OHRC Primary</span>
                  <p className="text-[10px] text-[#555C58] dark:text-[#8C9893]">{DOSSIER_SENSOR_LABELS.ohrc}</p>
                </div>
              </div>

              {/* TMC-2 */}
              <div className="retro-outset flex flex-col gap-2 p-2 bg-[#E7E2D6] dark:bg-[#1A201E]">
                <div className="retro-inset relative aspect-square overflow-hidden bg-black p-0.5">
                  {/* eslint-disable-next-line @next/next/no-img-element */}
                  <img
                    src={imageUrl(`/images/tmc/${triplet.id}`)}
                    alt="TMC-2 single-view terrain"
                    className="h-full w-full object-cover"
                  />
                </div>
                <div className="text-xs">
                  <span className="text-[#1E2321] dark:text-[#E7E2D6] font-bold block">TMC-2 Reference</span>
                  <p className="text-[10px] text-[#555C58] dark:text-[#8C9893]">{DOSSIER_SENSOR_LABELS.tmc}</p>
                </div>
              </div>

              {/* NASA LRO NAC (if available) */}
              {triplet.lro_nac_available && (
                <div className="retro-outset flex flex-col gap-2 p-2 bg-[#E7E2D6] dark:bg-[#1A201E] border-amber-600/50">
                  <div className="retro-inset relative aspect-square overflow-hidden bg-black p-0.5">
                    {/* eslint-disable-next-line @next/next/no-img-element */}
                    <img
                      src={imageUrl(`/images/lro_nac/${triplet.id}`)}
                      alt="NASA LRO NAC reference"
                      className="h-full w-full object-cover"
                      onError={(e) => {
                        (e.currentTarget as HTMLImageElement).src = imageUrl(`/images/ohrc/${triplet.id}`);
                      }}
                    />
                  </div>
                  <div className="text-xs">
                    <div className="flex items-center justify-between">
                      <span className="text-[#1E2321] dark:text-[#E7E2D6] font-bold">NASA LRO NAC</span>
                      <span className="retro-inset px-1 text-[9px] font-bold text-amber-800 dark:text-amber-300 bg-amber-100 dark:bg-amber-950/60">Ref</span>
                    </div>
                    <p className="text-[10px] text-[#555C58] dark:text-[#8C9893]">{DOSSIER_SENSOR_LABELS.lro_nac(triplet.lro_nac_product_id)}</p>
                  </div>
                </div>
              )}

              {/* IIRS */}
              <div className="retro-outset flex flex-col gap-2 p-2 bg-[#E7E2D6] dark:bg-[#1A201E]">
                <div className="retro-inset relative aspect-square overflow-hidden bg-black p-0.5">
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
                  <span className="text-[#1E2321] dark:text-[#E7E2D6] font-bold block">IIRS Hyperspectral</span>
                  <p className="text-[10px] text-[#555C58] dark:text-[#8C9893]">{DOSSIER_SENSOR_LABELS.iirs}</p>
                </div>
              </div>

              {/* Registered Blend */}
              <div className="retro-outset flex flex-col gap-2 p-2 bg-[#E7E2D6] dark:bg-[#1A201E]">
                <div className="retro-inset relative aspect-square overflow-hidden bg-black p-0.5">
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
                  <span className="text-[#1F4743] dark:text-teal-300 font-bold block">50% Blend Overlay</span>
                  <p className="text-[10px] text-[#555C58] dark:text-[#8C9893]">Sub-pixel Homography</p>
                </div>
              </div>
            </div>
          </div>

          {/* Scientific Metrics & Analysis */}
          <div className="grid grid-cols-1 gap-4 border-t border-[#8B8579] dark:border-[#2D3835] pt-4 sm:grid-cols-2">
            <div className="retro-inset p-4 bg-[#DED8CB]/40 dark:bg-[#141817]">
              <h4 className="text-xs font-bold uppercase tracking-wider text-[#1F4743] dark:text-teal-300">
                [REGISTRATION TELEMETRY]
              </h4>
              <div className="mt-2.5 space-y-2 text-xs font-mono">
                <div className="flex justify-between border-b border-[#B9B2A5]/50 dark:border-[#2D3835] pb-1.5">
                  <span className="text-[#555C58] dark:text-[#8C9893]">Sub-Pixel Status</span>
                  <span className="font-bold text-emerald-600 dark:text-emerald-400">
                    {metrics?.sub_pixel_accurate ? "Verified (< 0.50 px)" : "Standard Alignment (< 1.0 px)"}
                  </span>
                </div>
                <div className="flex justify-between border-b border-[#B9B2A5]/50 dark:border-[#2D3835] pb-1.5">
                  <span className="text-[#555C58] dark:text-[#8C9893]">In-Sample Fit RMSE</span>
                  <span className="font-bold text-[#28557E] dark:text-sky-300">
                    {metrics?.fit_rmse_px != null
                      ? `${metrics.fit_rmse_px.toFixed(3)} px`
                      : (metrics?.rmse_px != null ? `${metrics.rmse_px.toFixed(3)} px` : "—")}
                  </span>
                </div>
                <div className="flex justify-between border-b border-[#B9B2A5]/50 dark:border-[#2D3835] pb-1.5">
                  <span className="text-[#555C58] dark:text-[#8C9893]">Held-Out Validation RMSE</span>
                  <span className="font-bold text-emerald-600 dark:text-emerald-400">
                    {metrics?.validation_rmse_px != null ? `${metrics.validation_rmse_px.toFixed(3)} px` : "—"}
                  </span>
                </div>
                <div className="flex justify-between border-b border-[#B9B2A5]/50 dark:border-[#2D3835] pb-1.5">
                  <span className="text-[#555C58] dark:text-[#8C9893]">Absolute Topographic RMSE</span>
                  <span className="font-bold text-[#1E2321] dark:text-[#E7E2D6]">
                    {metrics?.absolute_rmse_m != null
                      ? `${metrics.absolute_rmse_m.toFixed(2)} m${
                          metrics?.absolute_rmse_m_provenance === "planar_footprint_gsd_no_dem"
                            ? " (planar, no DEM)"
                            : ""
                        }`
                      : "—"}
                  </span>
                </div>
                <div className="flex justify-between border-b border-[#B9B2A5]/50 dark:border-[#2D3835] pb-1.5">
                  <span className="text-[#555C58] dark:text-[#8C9893]">Post-RANSAC Inliers</span>
                  <span className="font-bold text-[#1E2321] dark:text-[#E7E2D6]">
                    {metrics?.num_inliers ?? "—"} matches ({metrics?.inlier_ratio != null ? `${(metrics.inlier_ratio * 100).toFixed(1)}%` : "100%"})
                  </span>
                </div>
                <div className="flex justify-between border-b border-[#B9B2A5]/50 dark:border-[#2D3835] pb-1.5">
                  <span className="text-[#555C58] dark:text-[#8C9893]">Spatial Coverage Score</span>
                  <span className="font-bold text-[#1E2321] dark:text-[#E7E2D6]">
                    {metrics?.combined_coverage_score != null ? `${(metrics.combined_coverage_score * 100).toFixed(1)}%` : "—"}
                  </span>
                </div>
                {metrics?.composite_quality_score != null && (
                  <div className="flex justify-between border-b border-[#B9B2A5]/50 dark:border-[#2D3835] pb-1.5">
                    <span className="text-[#555C58] dark:text-[#8C9893]">Composite Quality Score</span>
                    <span className="font-bold text-[#28557E] dark:text-sky-300">
                      {(metrics.composite_quality_score * 100).toFixed(1)}%
                    </span>
                  </div>
                )}
                {metrics?.nmi != null && (
                  <div className="flex justify-between border-b border-[#B9B2A5]/50 dark:border-[#2D3835] pb-1.5">
                    <span className="text-[#555C58] dark:text-[#8C9893]">Normalized Mutual Info (NMI)</span>
                    <span className="font-bold text-[#1E2321] dark:text-[#E7E2D6]">
                      {metrics.nmi.toFixed(3)}
                    </span>
                  </div>
                )}
                {metrics?.outlier_method && (
                  <div className="flex justify-between border-b border-[#B9B2A5]/50 dark:border-[#2D3835] pb-1.5">
                    <span className="text-[#555C58] dark:text-[#8C9893]">Robust Estimator</span>
                    <span className="font-bold text-[#1F4743] dark:text-teal-300">
                      {metrics.outlier_method}
                    </span>
                  </div>
                )}
                {triplet.lro_nac_available && (
                  <div className="flex justify-between border-b border-[#B9B2A5]/50 dark:border-[#2D3835] pb-1.5">
                    <span className="text-[#555C58] dark:text-[#8C9893]">Lunar Reference Mode</span>
                    <span className="font-bold text-amber-700 dark:text-amber-300">
                      NASA LRO NAC ({scaleRatioLabel(SENSOR_META.lro_nac.gsdM, SENSOR_META.ohrc.gsdM)} GSD ratio)
                    </span>
                  </div>
                )}
                <div className="flex justify-between">
                  <span className="text-[#555C58] dark:text-[#8C9893]">Elevation Layer</span>
                  <span className="font-bold text-[#1E2321] dark:text-[#E7E2D6]">
                    {triplet.dem_available ? "DEM Available (TMC DTM)" : "Interpolated"}
                  </span>
                </div>
              </div>
            </div>

            <div className="retro-inset p-4 bg-[#DED8CB]/40 dark:bg-[#141817]">
              <h4 className="text-xs font-bold uppercase tracking-wider text-[#1F4743] dark:text-teal-300">
                [REGISTRATION METHOD]
              </h4>
              <p className="mt-2.5 text-xs leading-relaxed text-[#4A524E] dark:text-[#A8B2AD]">
                Cross-modal alignment applies CFOG + phase congruency feature extraction for extreme scale disparities (OHRC ↔ TMC-2 {scaleRatioLabel(sensorMeta(triplet, "tmc").gsdM, sensorMeta(triplet, "ohrc").gsdM)}) and phase-correlation / LK optical flow for close-resolution reference images (OHRC ↔ NASA LRO NAC {scaleRatioLabel(SENSOR_META.lro_nac.gsdM, SENSOR_META.ohrc.gsdM)}), coupled with strict iterative RANSAC and physical validation gates.
              </p>
            </div>
          </div>
        </div>

        {/* Footer Actions */}
        <div className="flex items-center justify-between border-t border-[#8B8579] dark:border-[#2D3835] bg-[#DED8CB] dark:bg-[#141817] px-4 py-2">
          <button
            onClick={onClose}
            className="retro-button px-3 py-1 text-xs font-mono font-bold text-[#1E2321] dark:text-[#E7E2D6]"
          >
            Close [Esc]
          </button>

          <button
            onClick={() => {
              onClose();
              onOpenWorkspace(triplet.id);
            }}
            className="retro-button-primary px-4 py-1 text-xs font-mono font-bold flex items-center gap-1.5"
          >
            <span>Open in Workspace</span>
            <span>→</span>
          </button>
        </div>
      </div>
    </div>
  );
}

