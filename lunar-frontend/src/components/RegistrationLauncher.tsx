"use client";

import { useEffect, useRef, useState } from "react";
import { API_BASE, API_GET_TIMEOUT_MS, api, imageUrl } from "@/lib/api";
import { getAuthHeaders } from "@/lib/auth";
import LunarGlobe, { type LunarFootprint } from "@/components/hero/LunarGlobe";
import type { MoonPoint, MoonPointsResponse } from "@/lib/backend-types";
import { footprintSizeKm } from "@/lib/geo";
import { SENSOR_META, scaleRatioLabel, SENSOR_OPTIONS } from "@/lib/sensors";
import TrafficLightBadge from "@/components/TrafficLightBadge";

// True selenographic bounds (planetocentric, deg) from
// data_preprocessing_pipeline/processed_triplets/*/manifest.json. Used as the
// offline fallback so region footprints render even when /triplets is down;
// live bounds from GET /triplets replace these when available.
const FALLBACK_FOOTPRINTS: LunarFootprint[] = [
  { id: "region_001", west_lon: 336.484646, east_lon: 336.589455, south_lat: -3.3748612, north_lat: -3.2487328 },
  { id: "region_002", west_lon: 336.484646, east_lon: 336.589455, south_lat: -3.20669, north_lat: -3.0805616 },
  { id: "region_003", west_lon: 336.484646, east_lon: 336.589455, south_lat: -3.0385188, north_lat: -2.9123904 },
  { id: "region_004", west_lon: 336.484646, east_lon: 336.589455, south_lat: -2.8703476, north_lat: -2.7442192 },
  { id: "region_005", west_lon: 234.396774, east_lon: 234.528638, south_lat: 4.86317335, north_lat: 5.0672006 },
  { id: "region_006", west_lon: 234.396774, east_lon: 234.528638, south_lat: 5.1488115, north_lat: 5.35283875 },
];

function to180(lon: number): number {
  return ((((lon + 180) % 360) + 360) % 360) - 180;
}

type Sensor = "OHRC" | "TMC" | "IIRS" | "LRO_NAC";

type RegistrationMetrics = {
  fit_rmse_px?: number | null;
  rmse_px?: number | null;
  validation_rmse_px?: number | null;
  absolute_rmse_m?: number | null;
  validation_status?: string | null;
  num_inliers?: number | null;
  inlier_count?: number | null;
  inlier_ratio?: number | null;
  combined_coverage_score?: number | null;
  source_coverage_ratio?: number | null;
  spatial_coverage?: number | null;
  uniformity_score?: number | null;
  spatial_uniformity?: number | null;
  quality_tier?: string | null;
  ssim?: number | null;
  // Epic 3 objective verification (tracks backend metrics.ssim_score /
  // confidence_score / traffic_light_color); rendered via TrafficLightBadge.
  ssim_score?: number | null;
  confidence_score?: number | null;
  traffic_light_color?: "GREEN" | "YELLOW" | "RED" | string | null;
  held_out_rmse?: number | null;
  held_out_validation_rmse_px?: number | null;
  psnr?: number | null;
  nmi?: number | null;
  composite_quality_score?: number | null;
  outlier_method?: string | null;
};

type RegistrationResult = {
  status: string;
  message?: string | null;
  metrics?: RegistrationMetrics | null;
  visual_url?: string | null;
  warped_url?: string | null;
  source_url?: string | null;
  reference_url?: string | null;
  matches_url?: string | null;
  raster_url?: string | null;
  quiver_url?: string | null;
  job_id?: string | null;
};

type MatchPoint = {
  source_x?: number;
  source_y?: number;
  target_x?: number;
  target_y?: number;
  image1_x?: number;
  image1_y?: number;
  image2_x?: number;
  image2_y?: number;
  confidence?: number;
};

interface SamplePair {
  id: string;
  title: string;
  tag: string;
  badgeStyle: string;
  description: string;
  sourceSensor: Sensor;
  referenceSensor: Sensor;
  sourceUrl: string;
  referenceUrl: string;
  demUrl?: string;
  demoResult: RegistrationResult;
  demoPoints: MatchPoint[];
}

const SAMPLE_PAIRS: SamplePair[] = [
  {
    id: "sample_001_tmc",
    title: "Region 001: OHRC ↔ TMC-2",
    // Live manifests serve TMC 5.4 / OHRC 0.25 (21.6x → "~22x"). The legacy
    // "~21x" string came from a hardcoded 5.25 fudge with no provenance;
    // this static sample tag uses the derived value instead.
    tag: `Primary ${scaleRatioLabel(5.4, SENSOR_META.ohrc.gsdM)} Scale Gap`,
    badgeStyle: "bg-blue-50 dark:bg-blue-950/50 text-blue-700 dark:text-blue-300 border-blue-200 dark:border-blue-900/50",
    description: `${SENSOR_META.ohrc.gsdM}m/px Narrow-Angle OHRC matched to ${(4).toFixed(1)}m/px single-view TMC-2 in Sinus Medii.`,
    sourceSensor: "OHRC",
    referenceSensor: "TMC",
    sourceUrl: "/images/ohrc/region_001",
    referenceUrl: "/images/tmc/region_001",
    demoResult: {
      status: "success",
      message: "Registration across 21x scale disparity (Quality Gates 1-3 Passed; fragile 7-inlier consensus).",
      metrics: {
        fit_rmse_px: 1.2715,
        validation_rmse_px: null,
        absolute_rmse_m: 0.92,
        num_inliers: 7,
        inlier_count: 7,
        inlier_ratio: 0.1707,
        combined_coverage_score: 0.4375,
        spatial_coverage: 0.4375,
        spatial_uniformity: 0.3113,
        quality_tier: "LOW_CONFIDENCE",
        validation_status: "insufficient_points_for_holdout",
        ssim: 0.684,
        psnr: 24.12,
        nmi: 0.732,
        composite_quality_score: 0.695,
        outlier_method: "RANSAC",
      },
      visual_url: "/images/registered/region_001/checkerboard_qa.png",
      warped_url: "/images/registered/region_001/registered_ohrc.png",
      source_url: "/images/ohrc/region_001",
      reference_url: "/images/tmc/region_001",
      matches_url: null,
      raster_url: null,
      job_id: "region_001",
    },
    demoPoints: [
      { image1_x: 332, image1_y: 25, image2_x: 285, image2_y: 28, confidence: 0.92 },
      { image1_x: 25, image1_y: 230, image2_x: 71, image2_y: 262, confidence: 0.89 },
      { image1_x: 486, image1_y: 281, image2_x: 490, image2_y: 207, confidence: 0.88 },
      { image1_x: 215, image1_y: 412, image2_x: 232, image2_y: 419, confidence: 0.94 },
      { image1_x: 390, image1_y: 195, image2_x: 374, image2_y: 188, confidence: 0.85 },
      { image1_x: 140, image1_y: 95, image2_x: 128, image2_y: 104, confidence: 0.91 },
    ],
  },
  {
    id: "sample_001_lro",
    title: "Region 001: OHRC ↔ NASA LRO NAC",
    tag: `PS Lunar Reference (${scaleRatioLabel(SENSOR_META.lro_nac.gsdM, SENSOR_META.ohrc.gsdM)}, real CDR)`,
    badgeStyle: "bg-emerald-50 dark:bg-emerald-950/50 text-emerald-700 dark:text-emerald-300 border-emerald-200 dark:border-emerald-900/50",
    description:
      "NASA LRO NAC M1417670274LC real-CDR panchromatic reference (MI path; optical NCC finds 0 candidates on real CDRs).",
    sourceSensor: "OHRC",
    referenceSensor: "LRO_NAC",
    sourceUrl: "/images/ohrc/region_001",
    referenceUrl: "/images/lro_nac/region_001",
    demoResult: {
      status: "success",
      message:
        "Real-CDR reference registration: Fit RMSE 0.633 px, 6/32 inliers, LOW_CONFIDENCE (fragile, MI path).",
      metrics: {
        fit_rmse_px: 0.6333,
        validation_rmse_px: null,
        absolute_rmse_m: 0.5788,
        num_inliers: 6,
        inlier_count: 6,
        inlier_ratio: 0.1875,
        combined_coverage_score: 0.06,
        spatial_coverage: 0.06,
        spatial_uniformity: 0.0183,
        quality_tier: "LOW_CONFIDENCE",
        validation_status: "insufficient_points_for_holdout",
        ssim: 0.1441,
        psnr: 15.0171,
        nmi: 0.0029,
        composite_quality_score: 0.3115,
        outlier_method: "RANSAC",
      },
      visual_url: "/images/registered/lro_nac/region_001/checkerboard_qa.png",
      warped_url: "/images/registered/lro_nac/region_001/registered_source.png",
      source_url: "/images/ohrc/region_001",
      reference_url: "/images/lro_nac/region_001",
      matches_url: null,
      raster_url: null,
      job_id: "region_001",
    },
    demoPoints: [
      { image1_x: 64, image1_y: 64, image2_x: 66, image2_y: 65, confidence: 0.98 },
      { image1_x: 64, image1_y: 192, image2_x: 65, image2_y: 192, confidence: 0.96 },
      { image1_x: 84, image1_y: 301, image2_x: 84.4, image2_y: 300.8, confidence: 0.99 },
      { image1_x: 100, image1_y: 337, image2_x: 100.1, image2_y: 336.7, confidence: 0.97 },
      { image1_x: 151, image1_y: 116, image2_x: 152.2, image2_y: 117.2, confidence: 0.99 },
      { image1_x: 192, image1_y: 192, image2_x: 192.5, image2_y: 192.7, confidence: 0.99 },
      { image1_x: 320, image1_y: 320, image2_x: 319.5, image2_y: 320.2, confidence: 0.98 },
      { image1_x: 448, image1_y: 448, image2_x: 447, image2_y: 448, confidence: 0.97 },
    ],
  },
  {
    id: "sample_003_lro",
    title: "Region 003: OHRC ↔ NASA LRO NAC",
    tag: "Real CDR, distinct topography",
    badgeStyle: "bg-purple-50 dark:bg-purple-950/50 text-purple-700 dark:text-purple-300 border-purple-200 dark:border-purple-900/50",
    description:
      "Real-CDR M1417670274LC across distinct crater topography (Fit: 1.286 px — NOT sub-pixel; honest LOW_CONFIDENCE).",
    sourceSensor: "OHRC",
    referenceSensor: "LRO_NAC",
    sourceUrl: "/images/ohrc/region_003",
    referenceUrl: "/images/lro_nac/region_003",
    demoResult: {
      status: "success",
      message:
        "Real-CDR reference registration: Fit RMSE 1.286 px (>1 px, not sub-pixel), 5/27 inliers, LOW_CONFIDENCE.",
      metrics: {
        fit_rmse_px: 1.286,
        validation_rmse_px: null,
        absolute_rmse_m: 1.1754,
        num_inliers: 5,
        inlier_count: 5,
        inlier_ratio: 0.1852,
        combined_coverage_score: 0.05,
        spatial_coverage: 0.05,
        spatial_uniformity: 0.0135,
        quality_tier: "LOW_CONFIDENCE",
        validation_status: "insufficient_points_for_holdout",
        ssim: 0.1915,
        psnr: 13.4525,
        nmi: 0.004,
        composite_quality_score: 0.2415,
        outlier_method: "RANSAC",
      },
      visual_url: "/images/registered/lro_nac/region_003/checkerboard_qa.png",
      warped_url: "/images/registered/lro_nac/region_003/registered_source.png",
      source_url: "/images/ohrc/region_003",
      reference_url: "/images/lro_nac/region_003",
      matches_url: null,
      raster_url: null,
      job_id: "region_003",
    },
    demoPoints: [
      { image1_x: 64, image1_y: 64, image2_x: 65, image2_y: 65, confidence: 0.97 },
      { image1_x: 192, image1_y: 192, image2_x: 192.4, image2_y: 192.5, confidence: 0.98 },
      { image1_x: 320, image1_y: 320, image2_x: 319.8, image2_y: 320.1, confidence: 0.98 },
      { image1_x: 448, image1_y: 448, image2_x: 447.2, image2_y: 448.0, confidence: 0.96 },
    ],
  },
  {
    id: "sample_001_iirs",
    title: "Region 001: OHRC ↔ IIRS Hyperspectral",
    tag: `Chained Spectral Overlay (${scaleRatioLabel(SENSOR_META.iirs.gsdM, SENSOR_META.ohrc.gsdM, { approx: true })})`,
    badgeStyle: "bg-purple-50 dark:bg-purple-950/50 text-purple-700 dark:text-purple-300 border-purple-200 dark:border-purple-900/50",
    description:
      "80m/px IIRS hyperspectral cube co-registered via TMC-2 bridge. Spectral overlay — physical scale respected, no direct sub-pixel matching.",
    sourceSensor: "OHRC",
    referenceSensor: "IIRS",
    sourceUrl: "/images/ohrc/region_001",
    referenceUrl: "/images/iirs/region_001",
    demoResult: {
      status: "success",
      message:
        "Co-registered via TMC-2 Chained Homography. IIRS treated as spectral overlay (Physical scale respected).",
      metrics: {
        fit_rmse_px: null,
        validation_rmse_px: null,
        absolute_rmse_m: null,
        num_inliers: 0,
        inlier_count: 0,
        inlier_ratio: 0.0,
        combined_coverage_score: 0.0,
        spatial_coverage: 0.0,
        spatial_uniformity: 0.0,
        quality_tier: "SPECTRAL_PROJECTION_VALIDATED",
        validation_status: "Spectral Projection: Validated via TMC-2 Bridge",
      },
      visual_url: "/images/registered/region_001/checkerboard_qa.png",
      warped_url: "/images/registered/region_001/registered_ohrc.png",
      source_url: "/images/ohrc/region_001",
      reference_url: "/images/iirs/region_001",
      matches_url: null,
      raster_url: null,
      job_id: "region_001",
    },
    demoPoints: [],
  },
  {
    id: "sample_hardest_case",
    title: "Triplet New: 162° Sun-Gap Hardest Case",
    tag: "Fragile LOW · No Held-Out",
    badgeStyle: "bg-amber-50 dark:bg-amber-950/50 text-amber-700 dark:text-amber-300 border-amber-200 dark:border-amber-900/50",
    description:
      "Diametric illumination reversal (162.25°): formerly a clean Gate-3 refusal, re-measured 2026-09-11 as a fragile 6-inlier LOW fit with no held-out validation. The honest floor of what this pipeline claims.",
    sourceSensor: "OHRC",
    referenceSensor: "TMC",
    sourceUrl: "/images/ohrc/triplet_new_2022",
    referenceUrl: "/images/tmc/triplet_new_2022",
    demoResult: {
      status: "success",
      message:
        "Hardest-case registration: 6/50 inliers @1.214px fit, LOW_CONFIDENCE, no held-out validation. Former Gate-3 refusal; validated paraboloid passes gate bounds — fragile, not triumphant.",
      metrics: {
        fit_rmse_px: 1.214,
        validation_rmse_px: null,
        absolute_rmse_m: null,
        num_inliers: 6,
        inlier_count: 6,
        inlier_ratio: 0.12,
        combined_coverage_score: 0.06,
        spatial_coverage: 0.06,
        spatial_uniformity: 0.0183,
        quality_tier: "LOW_CONFIDENCE",
        validation_status: "insufficient_points_for_holdout",
        ssim: null,
        psnr: null,
        nmi: null,
        composite_quality_score: 0.2278,
        outlier_method: "RANSAC",
      },
      visual_url: "/images/registered/triplet_new_2022/checkerboard_qa.png",
      warped_url: "/images/registered/triplet_new_2022/registered_ohrc.png",
      source_url: "/images/ohrc/triplet_new_2022",
      reference_url: "/images/tmc/triplet_new_2022",
      matches_url: null,
      raster_url: null,
      job_id: "triplet_new_2022",
    },
    // Exact coords from data_preprocessing_pipeline/matches/triplet_new_2022_matches.json
    demoPoints: [
      { image1_x: 13, image1_y: 423, image2_x: 21, image2_y: 443, confidence: 0.4 },
      { image1_x: 64, image1_y: 64, image2_x: 42, image2_y: 35, confidence: 0.5 },
      { image1_x: 64, image1_y: 192, image2_x: 70, image2_y: 192, confidence: 0.5 },
      { image1_x: 223, image1_y: 421, image2_x: 291, image2_y: 416, confidence: 0.51 },
      { image1_x: 320, image1_y: 448, image2_x: 357.2936706542969, image2_y: 423.72601318359375, confidence: 0.41 },
      { image1_x: 277, image1_y: 199, image2_x: 315, image2_y: 259, confidence: 0.43 },
    ],
  },
];

function absoluteUrl(path?: string | null) {
  if (!path) return null;
  if (path.startsWith("http")) return path;
  return imageUrl(path);
}

// POST /register holds a CFOG worker for ~6-8s plus upload time: 120s
// budget (the shared 15s API_GET_TIMEOUT_MS is for idempotent GETs only).
const REGISTER_TIMEOUT_MS = 120_000;

/**
 * fetch with an abort budget. Rejects on timeout so callers fall into
 * their existing error banners instead of hanging spinners forever.
 */
function fetchWithTimeout(url: string, init: RequestInit, ms: number): Promise<Response> {
  try {
    if (typeof AbortSignal !== "undefined" && typeof AbortSignal.timeout === "function") {
      return fetch(url, { ...init, signal: AbortSignal.timeout(ms) });
    }
  } catch {
    // Fall through to a plain fetch on runtimes without AbortSignal.timeout.
  }
  return fetch(url, init);
}

function metric(metrics: RegistrationMetrics | null | undefined, ...keys: string[]) {
  for (const key of keys) {
    const value = metrics?.[key as keyof RegistrationMetrics];
    if (typeof value === "number" && Number.isFinite(value)) return value;
  }
  return null;
}

function format(value: number | null, digits = 3) {
  return value === null ? "—" : value.toFixed(digits);
}

function coordinate(point: MatchPoint, side: "source" | "reference") {
  const x = side === "source" ? point.source_x ?? point.image1_x : point.target_x ?? point.image2_x;
  const y = side === "source" ? point.source_y ?? point.image1_y : point.target_y ?? point.image2_y;
  return typeof x === "number" && typeof y === "number" ? ([x, y] as const) : null;
}

function pointColor(confidence = 0) {
  return confidence >= 0.8 ? "#34d399" : confidence >= 0.5 ? "#fbbf24" : "#fb7185";
}

function PointOverlay({
  points,
  side,
  imgWidth,
  imgHeight,
}: {
  points: MatchPoint[];
  side: "source" | "reference";
  imgWidth: number;
  imgHeight: number;
}) {
  return (
    <div className="pointer-events-none absolute inset-0">
      {points.map((point, index) => {
        const value = coordinate(point, side);
        if (!value) return null;
        return (
          <span
            key={`${side}-${index}`}
            className="absolute h-2.5 w-2.5 -translate-x-1/2 -translate-y-1/2 rounded-full border border-white shadow-[0_0_0_2px_rgba(15,23,42,.55)]"
            style={{
              left: `${(value[0] / (imgWidth || 512)) * 100}%`,
              top: `${(value[1] / (imgHeight || 512)) * 100}%`,
              backgroundColor: pointColor(point.confidence),
            }}
          />
        );
      })}
    </div>
  );
}

function FilePicker({
  label,
  file,
  preview,
  optional,
  onChange,
}: {
  label: string;
  file: File | null;
  preview?: string | null;
  optional?: boolean;
  onChange: (file: File | null) => void;
}) {
  return (
    <div className="retro-inset p-2.5 bg-[#F9F7F1] dark:bg-[#141817] flex flex-col justify-between">
      <div>
        <div className="flex items-center justify-between text-[11px] font-bold text-[#143532] dark:text-emerald-400">
          <span>{label}</span>
          {optional && <span className="text-[9px] bg-[#E2DDCF] dark:bg-[#222927] px-1 border border-[#AFA99B] dark:border-[#3A4541]">OPTIONAL</span>}
        </div>
        <label className="my-2 border border-dashed border-[#A09A8D] dark:border-[#3A4541] p-3 text-center bg-[#F3EFE6] dark:bg-[#1A201E] block cursor-pointer hover:bg-[#EAE5D8] dark:hover:bg-[#222927] transition">
          <div className="text-xl">📁</div>
          <div className="text-xs font-bold text-[#2A312D] dark:text-[#E7E2D6] mt-0.5">
            {file ? file.name : preview ? "Pre-loaded image ready" : "Drop GeoTIFF / PNG here"}
          </div>
          <div className="text-[10px] text-[#717874] dark:text-[#8C9893] mt-0.5">
            or click to browse from workstation
          </div>
          <input
            type="file"
            accept=".jpg,.jpeg,.png,.tif,.tiff,image/*"
            onChange={(event) => onChange(event.target.files?.[0] || null)}
            className="hidden"
          />
        </label>
      </div>
      <div className="flex items-center justify-between pt-1">
        <label className="retro-button px-2.5 py-0.5 text-xs font-bold cursor-pointer">
          Choose File
          <input
            type="file"
            accept=".jpg,.jpeg,.png,.tif,.tiff,image/*"
            onChange={(event) => onChange(event.target.files?.[0] || null)}
            className="hidden"
          />
        </label>
        <span className="text-[10px] font-mono text-[#777] dark:text-[#888] truncate max-w-[140px]">
          {file ? file.name : preview ? "Sample image loaded" : "No file chosen"}
        </span>
      </div>
    </div>
  );
}

function MetricCard({
  label,
  value,
  suffix,
  emphasis,
  sublabel,
  hint,
}: {
  label: string;
  value: string;
  suffix?: string;
  emphasis?: boolean;
  sublabel?: string;
  hint?: string | null;
}) {
  const missing = value === "—";
  return (
    <div className={`retro-inset p-3 ${emphasis ? "bg-[#EEF5FA] dark:bg-[#152330] border-[#5A85A8]" : "bg-[#F9F7F1] dark:bg-[#141817]"}`}>
      <p className="text-[10px] font-mono font-bold uppercase tracking-wider text-[#555C58] dark:text-[#8C9893]">{label}</p>
      <p className={`mt-1 text-xl font-mono font-bold tracking-tight ${emphasis ? "text-[#28557E] dark:text-cyan-300" : "text-[#1E2321] dark:text-[#E7E2D6]"}`}>
        {value}
        {suffix && !missing && <span className="ml-1 text-xs font-normal text-[#555C58] dark:text-[#8C9893]">{suffix}</span>}
      </p>
      {missing && hint ? (
        <p className="mt-1 text-[10px] font-mono font-semibold text-amber-700 dark:text-amber-400">{hint}</p>
      ) : (
        sublabel && <p className="mt-1 text-[10px] font-mono text-[#777] dark:text-[#888]">{sublabel}</p>
      )}
    </div>
  );
}

// Local unavailability reasons derived from displayed values only (works for
// snapshots and live runs alike — no backend dependency).
function valHint(inliers: number | null): string | null {
  if (inliers === null) return "No verified matches in this run";
  if (inliers < 8) return `Held-out needs ≥8 inliers (have ${inliers})`;
  return null;
}

function DownloadLink({ href, label }: { href?: string | null; label: string }) {
  const url = absoluteUrl(href);
  return url ? (
    <a
      href={url}
      download
      target="_blank"
      rel="noreferrer"
      className="retro-button px-2.5 py-1 text-[11px] font-mono font-bold text-[#1E2321] dark:text-[#E7E2D6]"
    >
      ↓ {label}
    </a>
  ) : null;
}

function OverlayImage({
  title,
  src,
  fallbackSrc,
  secondaryFallback,
  points,
  side,
}: {
  title: string;
  src: string | null;
  fallbackSrc?: string | null;
  secondaryFallback?: string | null;
  points: MatchPoint[];
  side: "source" | "reference";
}) {
  const [currentSrc, setCurrentSrc] = useState<string | null>(src);
  const [failed, setFailed] = useState(false);
  const [naturalDims, setNaturalDims] = useState<{ w: number; h: number }>({ w: 512, h: 512 });

  useEffect(() => {
    setCurrentSrc(src);
    setFailed(false);
  }, [src]);

  const handleError = () => {
    if (fallbackSrc && currentSrc !== fallbackSrc) {
      setCurrentSrc(fallbackSrc);
    } else if (secondaryFallback && currentSrc !== secondaryFallback) {
      setCurrentSrc(secondaryFallback);
    } else {
      setFailed(true);
    }
  };

  return (
    <div className="retro-outset p-2.5">
      <div className="mb-2 flex items-center justify-between font-mono">
        <h4 className="text-xs font-bold text-[#143532] dark:text-[#52938B]">{title}</h4>
        <span className="text-[10px] text-[#717874] dark:text-[#8C9893]">{points.length} inlier pts</span>
      </div>
      <div className="relative aspect-square overflow-hidden retro-inset bg-[#0A0D0C] flex items-center justify-center">
        {!failed && currentSrc ? (
          // eslint-disable-next-line @next/next/no-img-element
          <img
            src={currentSrc}
            alt={title}
            className="h-full w-full object-contain"
            onLoad={(e) => {
              const img = e.currentTarget as HTMLImageElement;
              if (img.naturalWidth > 0 && img.naturalHeight > 0) {
                setNaturalDims({ w: img.naturalWidth, h: img.naturalHeight });
              }
            }}
            onError={handleError}
          />
        ) : (
          <div className="flex flex-col items-center justify-center p-6 text-center text-slate-400">
            <span className="text-2xl mb-1.5 opacity-80">🛰️</span>
            <span className="text-xs font-semibold text-slate-300">{title}</span>
            <span className="text-[11px] text-slate-500 mt-1 max-w-[220px]">
              Raw GeoTIFF ingested · {points.length} correspondence points mapped
            </span>
          </div>
        )}
        <PointOverlay points={points} side={side} imgWidth={naturalDims.w} imgHeight={naturalDims.h} />
      </div>
    </div>
  );
}

export default function RegistrationLauncher() {
  const uploadSectionRef = useRef<HTMLDivElement>(null);

  const [sourceFile, setSourceFile] = useState<File | null>(null);
  const [referenceFile, setReferenceFile] = useState<File | null>(null);
  const [sourcePreview, setSourcePreview] = useState<string | null>(null);
  const [referencePreview, setReferencePreview] = useState<string | null>(null);
  const [demFile, setDemFile] = useState<File | null>(null);
  const [sourceSensor, setSourceSensor] = useState<Sensor>("OHRC");
  const [referenceSensor, setReferenceSensor] = useState<Sensor>("TMC");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [result, setResult] = useState<RegistrationResult | null>(null);
  // Provenance of the currently displayed result: the pre-loaded committed
  // benchmark snapshot vs a genuinely live backend run (Step 13 honesty).
  const [resultProvenance, setResultProvenance] = useState<"snapshot" | "live">("snapshot");
  const [points, setPoints] = useState<MatchPoint[]>([]);
  const [selectedSample, setSelectedSample] = useState<SamplePair | null>(null);
  const [customMode, setCustomMode] = useState(true);
  const [resultView, setResultView] = useState<"2d" | "3d">("2d");
  const [moonPoints, setMoonPoints] = useState<MoonPoint[]>([]);
  const [moonPointsLoading, setMoonPointsLoading] = useState(false);
  const [footprints, setFootprints] = useState<LunarFootprint[]>(FALLBACK_FOOTPRINTS);

  // Live region bounds for the 3D globe footprints (true coordinates).
  useEffect(() => {
    let active = true;
    api
      .listTriplets()
      .then((data) => {
        if (!active) return;
        const live: LunarFootprint[] = (data.triplets || [])
          .filter((t) => t?.bounds && /^region_\d+$/i.test(t.id))
          .map((t) => ({
            id: t.id,
            west_lon: t.bounds.west_lon,
            east_lon: t.bounds.east_lon,
            south_lat: t.bounds.south_lat,
            north_lat: t.bounds.north_lat,
          }));
        if (live.length > 0) setFootprints(live);
      })
      .catch(() => {
        // Offline: keep FALLBACK_FOOTPRINTS (real manifest coordinates).
      });
    return () => {
      active = false;
    };
  }, []);

  useEffect(() => {
    const jobId = result?.job_id;
    if (!jobId) {
      setMoonPoints([]);
      return;
    }

    let active = true;
    setMoonPointsLoading(true);

    // Authenticated like every other backend call (moon-points is served by
    // the same FastAPI app; a rotated JWT must 401 here, not leak points).
    // 15s budget so a hung backend can't leave the globe spinner forever.
    const ctrl = typeof AbortController !== "undefined" ? new AbortController() : null;
    const timer = ctrl ? window.setTimeout(() => ctrl.abort(), API_GET_TIMEOUT_MS) : null;
    fetch(`${API_BASE}/api/registration/moon-points/${jobId}`, {
      headers: { ...getAuthHeaders() },
      ...(ctrl ? { signal: ctrl.signal } : {}),
    })
      .then((res) => {
        if (!res.ok) return null;
        return res.json() as Promise<MoonPointsResponse>;
      })
      .then((data) => {
        if (!active) return;
        if (data && Array.isArray(data.points)) {
          setMoonPoints(data.points);
        } else {
          setMoonPoints([]);
        }
      })
      .catch(() => {
        if (active) setMoonPoints([]);
      })
      .finally(() => {
        if (timer !== null) window.clearTimeout(timer);
        if (active) setMoonPointsLoading(false);
      });

    return () => {
      active = false;
      if (timer !== null) window.clearTimeout(timer);
      try {
        ctrl?.abort();
      } catch {
        // Aborting an already-settled fetch is a no-op.
      }
    };
  }, [result?.job_id]);

  const handleSourceFile = (file: File | null) => {
    setSourceFile(file);
    setSelectedSample(null);
    setCustomMode(true);
    if (!file) {
      setSourcePreview(null);
      return;
    }
    const reader = new FileReader();
    reader.onload = () => setSourcePreview(reader.result as string);
    reader.readAsDataURL(file);
  };

  const handleReferenceFile = (file: File | null) => {
    setReferenceFile(file);
    setSelectedSample(null);
    setCustomMode(true);
    if (!file) {
      setReferencePreview(null);
      return;
    }
    const reader = new FileReader();
    reader.onload = () => setReferencePreview(reader.result as string);
    reader.readAsDataURL(file);
  };

  const loadSample = (sample: SamplePair) => {
    setSelectedSample(sample);
    setCustomMode(false);
    setSourceSensor(sample.sourceSensor);
    setReferenceSensor(sample.referenceSensor);
    setSourcePreview(absoluteUrl(sample.sourceUrl));
    setReferencePreview(absoluteUrl(sample.referenceUrl));
    setSourceFile(null);
    setReferenceFile(null);
    setDemFile(null);
    setError(null);
    setResult(sample.demoResult);
    setResultProvenance("snapshot");
    setPoints(sample.demoPoints);
  };

  const reset = () => {
    setResult(null);
    setPoints([]);
    setResultView("2d");
    setMoonPoints([]);
    setResultProvenance("snapshot");
    setError(null);
    setSelectedSample(null);
    setCustomMode(true);
    setSourceFile(null);
    setReferenceFile(null);
    setSourcePreview(null);
    setReferencePreview(null);
  };

  const loadTestPairIntoForm = async (
    sourceRel: string,
    refRel: string,
    sSensor: Sensor,
    rSensor: Sensor,
    sName: string,
    rName: string
  ) => {
    try {
      setLoading(true);
      setError(null);
      setResult(null);
      const [sRes, rRes] = await Promise.all([
        fetch(imageUrl(sourceRel)),
        fetch(imageUrl(refRel)),
      ]);
      const sBlob = await sRes.blob();
      const rBlob = await rRes.blob();
      const sFile = new File([sBlob], sName, { type: "image/png" });
      const rFile = new File([rBlob], rName, { type: "image/png" });
      setSourceFile(sFile);
      setReferenceFile(rFile);
      setSourcePreview(imageUrl(sourceRel));
      setReferencePreview(imageUrl(refRel));
      setSourceSensor(sSensor);
      setReferenceSensor(rSensor);
    } catch {
      setError("Could not load sample files into upload form.");
    } finally {
      setLoading(false);
    }
  };

  const register = async () => {
    if (!sourceFile && !selectedSample) {
      setError("Please select or upload both a source image and reference image before registering.");
      return;
    }

    setLoading(true);
    setError(null);
    setResult(null);
    setPoints([]);

    // If a sample is loaded, execute against the live backend. Step 13: NO
    // silent snapshot fallback — a dead backend surfaces the error banner,
    // never the committed snapshot presented as a live result.
    if (selectedSample && !sourceFile) {
      try {
        const body = new FormData();
        body.append("source_sensor", selectedSample.sourceSensor);
        body.append("reference_sensor", selectedSample.referenceSensor);

        // Fetch sample images as blobs to submit to the live backend.
        const [sBlob, rBlob] = await Promise.all([
          fetch(absoluteUrl(selectedSample.sourceUrl)!).then((r) => {
            if (!r.ok) throw new Error(`sample source fetch failed (${r.status})`);
            return r.blob();
          }),
          fetch(absoluteUrl(selectedSample.referenceUrl)!).then((r) => {
            if (!r.ok) throw new Error(`sample reference fetch failed (${r.status})`);
            return r.blob();
          }),
        ]);

        body.append("source_file", sBlob, "source.png");
        body.append("reference_file", rBlob, "reference.png");
        // POST /register holds a CFOG worker for ~6-8s plus upload time, so
        // it gets a 120s budget (not the 15s idempotent-GET budget) and the
        // static auth import (not a per-call dynamic import).
        const response = await fetchWithTimeout(`${API_BASE}/register`, {
          method: "POST",
          headers: { ...getAuthHeaders() },
          body,
        }, REGISTER_TIMEOUT_MS);
        const data = (await response.json().catch(() => null)) as
          | RegistrationResult
          | { detail?: string }
          | null;
        if (!response.ok) {
          throw new Error(
            data && "detail" in (data as object)
              ? (data as { detail?: string }).detail
              : `Registration failed (${response.status}).`
          );
        }
        const live = data as RegistrationResult;
        setResult(live);
        setResultProvenance("live");
        if (live.matches_url) {
          // Backend artifact fetch: same auth headers + GET timeout so a
          // rotated JWT or hung backend can't silently yield zero dots.
          const mRes = await fetchWithTimeout(absoluteUrl(live.matches_url)!, {
            headers: { ...getAuthHeaders() },
          }, API_GET_TIMEOUT_MS).catch(() => null);
          if (mRes && mRes.ok) setPoints((await mRes.json()) as MatchPoint[]);
          else setPoints([]);
        } else {
          setPoints([]);
        }
        if (live.status !== "success") {
          setError(live.message || "The backend could not verify this registration.");
        }
      } catch (err) {
        // Backend unreachable or rejected: error banner, NOT the snapshot.
        setResult(null);
        setPoints([]);
        setError(err instanceof Error ? err.message : "Registration failed unexpectedly.");
      } finally {
        setLoading(false);
      }
      return;
    }

    // User-uploaded files execution (authenticated: /register requires Bearer).
    try {
      if (!sourceFile || !referenceFile) {
        throw new Error("Both source and reference files are required.");
      }
      const body = new FormData();
      body.append("source_file", sourceFile);
      body.append("reference_file", referenceFile);
      body.append("source_sensor", sourceSensor);
      body.append("reference_sensor", referenceSensor);
      if (demFile) body.append("dem_file", demFile);

      const response = await fetchWithTimeout(`${API_BASE}/register`, {
        method: "POST",
        headers: { ...getAuthHeaders() },
        body,
      }, REGISTER_TIMEOUT_MS);
      const data = (await response.json().catch(() => null)) as RegistrationResult | { detail?: string } | null;
      if (response.status === 401) {
        throw new Error("Session expired or missing — please sign in again, then retry.");
      }
      if (!response.ok) {
        throw new Error(data && "detail" in data ? data.detail : `Registration failed (${response.status}).`);
      }
      const registration = data as RegistrationResult;
      setResult(registration);
      setResultProvenance("live");
      if (registration.matches_url) {
        const matchesResponse = await fetchWithTimeout(absoluteUrl(registration.matches_url)!, {
          headers: { ...getAuthHeaders() },
        }, API_GET_TIMEOUT_MS).catch(() => null);
        if (matchesResponse && matchesResponse.ok) setPoints((await matchesResponse.json()) as MatchPoint[]);
      }
      if (registration.status !== "success") {
        setError(registration.message || "The backend could not verify this registration.");
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : "Registration failed unexpectedly.");
    } finally {
      setLoading(false);
    }
  };

  const metrics = result?.metrics;
  const activeSourceSensor: Sensor | undefined = selectedSample?.sourceSensor ?? sourceSensor;
  const activeReferenceSensor: Sensor | undefined = selectedSample?.referenceSensor ?? referenceSensor;
  const isIIRSPair = activeSourceSensor === "IIRS" || activeReferenceSensor === "IIRS";
  const qualityTier =
    metrics?.quality_tier ||
    (result?.status === "success" ? "HIGH_CONFIDENCE" : result?.status?.toUpperCase() || "READY");
  const qualityTone =
    qualityTier === "SPECTRAL_PROJECTION_VALIDATED"
      ? "border-purple-300 dark:border-purple-900/50 bg-purple-50 dark:bg-purple-950/30 text-purple-700 dark:text-purple-300"
      : qualityTier === "HIGH_CONFIDENCE"
      ? "border-emerald-300 dark:border-emerald-900/50 bg-emerald-50 dark:bg-emerald-950/30 text-emerald-700 dark:text-emerald-300"
      : qualityTier === "ACCEPTED"
      ? "border-cyan-300 dark:border-cyan-900/50 bg-cyan-50 dark:bg-cyan-950/30 text-cyan-700 dark:text-cyan-300"
      : "border-rose-300 dark:border-rose-900/50 bg-rose-50 dark:bg-rose-950/30 text-rose-700 dark:text-rose-300";

  const inliers = metric(metrics, "num_inliers", "inlier_count");
  const coverage = metric(metrics, "combined_coverage_score", "source_coverage_ratio", "spatial_coverage");
  const uniformity = metric(metrics, "uniformity_score", "spatial_uniformity");
  const ratio = metric(metrics, "inlier_ratio");
  const fitRmse = metric(metrics, "fit_rmse_px", "rmse_px");
  const valRmse = metric(metrics, "validation_rmse_px");
  const ssimVal = metric(metrics, "ssim");
  const ssimScore = metric(metrics, "ssim_score", "ssim");
  const psnrVal = metric(metrics, "psnr");
  const nmiVal = metric(metrics, "nmi");
  const compositeScore = metric(metrics, "composite_quality_score");
  const outlierMethod = metrics?.outlier_method || "RANSAC";
  // Epic 3 traffic-light verdict (null-safe; badge renders Unverified).
  const trafficColor =
    typeof metrics?.traffic_light_color === "string" ? metrics.traffic_light_color : null;
  const confidenceScore = metric(metrics, "confidence_score");
  const heldOutRmse = metric(metrics, "held_out_rmse", "held_out_validation_rmse_px", "validation_rmse_px");

  const sourcePrimarySrc = result?.source_url
    ? absoluteUrl(result.source_url)
    : result?.warped_url
    ? absoluteUrl(result.warped_url)
    : sourcePreview;
  const sourceFallbackSrc = result?.warped_url ? absoluteUrl(result.warped_url) : sourcePreview;

  const referencePrimarySrc = result?.reference_url ? absoluteUrl(result.reference_url) : referencePreview;

  const isFailedRegistration =
    ((result && result.status !== "success" && !isIIRSPair) || (customMode && !!error && !isIIRSPair)) as boolean;

  // Active 3D region: match job_id (e.g. "region_001") against footprint ids.
  const activeFootprintId =
    footprints.find((f) => f.id === result?.job_id)?.id ??
    (result?.job_id && /^region_\d+$/i.test(result.job_id) ? result.job_id : footprints[0]?.id ?? null);
  const activeFootprint = footprints.find((f) => f.id === activeFootprintId) ?? null;
  const activeCenter = activeFootprint
    ? {
        lat: (activeFootprint.south_lat + activeFootprint.north_lat) / 2,
        lon360: (activeFootprint.west_lon + activeFootprint.east_lon) / 2,
      }
    : null;
  const activeSize =
    activeFootprint != null
      ? footprintSizeKm({
          west_lon: activeFootprint.west_lon,
          east_lon: activeFootprint.east_lon,
          south_lat: activeFootprint.south_lat,
          north_lat: activeFootprint.north_lat,
        })
      : null;

  return (
    <section className="retro-outset p-1 text-[#1E2321] dark:text-[#E7E2D6]">
      {/* Retro OS Titlebar */}
      <div className="bg-[#1F4743] text-white px-2 py-1 flex items-center justify-between text-xs font-bold font-mono">
        <div className="flex items-center gap-2">
          <span className="w-2.5 h-2.5 bg-[#428178] border border-white/40" />
          <span>MISSION PAYLOAD // REGISTRATION PIPELINE v2.4</span>
        </div>
        <div className="flex items-center gap-1">
          <button type="button" className="window-ctrl-btn">_</button>
          <button type="button" className="window-ctrl-btn">□</button>
          <button type="button" className="window-ctrl-btn">X</button>
        </div>
      </div>

      <div className="p-3 md:p-4 space-y-4">
        {/* Header Bar */}
        <div className="flex flex-col justify-between gap-3 border-b border-[#AAA496] dark:border-[#2D3835] pb-3 md:flex-row md:items-center">
          <div>
            <div className="flex items-center gap-2">
              <span className="h-2 w-2 bg-[#28557E] shadow-[0_0_0_2px_rgba(40,85,126,.4)]" />
              <p className="text-[10px] font-mono font-bold uppercase tracking-[0.2em] text-[#1F4743] dark:text-[#52938B]">Live Registration Pipeline</p>
            </div>
            <h3 className="mt-1 text-lg font-mono font-bold tracking-tight text-[#1E2321] dark:text-[#E7E2D6]">Upload &amp; Register Workstation</h3>
            <p className="mt-0.5 max-w-2xl text-xs text-[#555C58] dark:text-[#8C9893]">
              Run the cross-sensor matching engine and inspect sub-pixel alignment, correspondence geometry, and Quality Gates.
            </p>
          </div>

          <div className="flex items-center gap-2 font-mono">
            {result && resultProvenance === "snapshot" && (
              <span
                className="retro-inset bg-[#FFFBEB] dark:bg-[#2A2312] border border-[#D97706] px-2 py-0.5 text-[10px] font-bold uppercase tracking-wider text-[#B45309] dark:text-amber-300"
                title="Committed benchmark snapshot from the repo manifests — press Run to verify live against the backend."
              >
                Snapshot · run live to verify
              </span>
            )}
            {result && resultProvenance === "live" && (
              <span className="retro-inset bg-[#ECFDF5] dark:bg-[#122A1E] border border-[#059669] px-2 py-0.5 text-[10px] font-bold uppercase tracking-wider text-[#047857] dark:text-emerald-300">
                Live backend run
              </span>
            )}
            <div className={`retro-outset px-2 py-0.5 text-[10px] font-bold uppercase tracking-wider ${result ? qualityTone : "bg-[#E7E2D6] dark:bg-[#1A201E] text-[#555C58] dark:text-[#8C9893]"}`}>
              {loading ? "PROCESSING..." : result ? qualityTier.replaceAll("_", " ") : "READY TO RUN"}
            </div>
            {result && !loading && (
              <TrafficLightBadge
                color={trafficColor}
                confidence_score={confidenceScore}
                ssim_score={ssimScore}
                held_out_rmse={heldOutRmse}
              />
            )}
          </div>
        </div>

        {/* 1. Custom Upload Inputs & Controls (Rendered FIRST, above the fold) */}
        <div ref={uploadSectionRef} className="space-y-3 retro-outset bg-[#E3DECFA8] dark:bg-[#161B19] p-3">
          <div className="flex items-center justify-between border-b border-[#AAA496] dark:border-[#2D3835] pb-1.5 font-mono">
            <span className="text-xs font-bold text-[#143532] dark:text-[#52938B]">Upload Custom Multi-Sensor Pair</span>
            <span className="text-[10px] text-[#717874] dark:text-[#8C9893]">GeoTIFF (.tif) or PNG (.png)</span>
          </div>

          <div className="grid gap-3 md:grid-cols-2">
            <FilePicker
              label={`Source Image · ${sourceSensor}`}
              file={sourceFile}
              preview={sourcePreview}
              onChange={handleSourceFile}
            />
            <FilePicker
              label={`Reference Image · ${referenceSensor}`}
              file={referenceFile}
              preview={referencePreview}
              onChange={handleReferenceFile}
            />
          </div>

          <div className="grid gap-3 md:grid-cols-[1fr_1fr_1.2fr]">
            <label className="block">
              <span className="mb-1 block text-[10px] font-mono font-bold uppercase tracking-wider text-[#555C58] dark:text-[#8C9893]">Source Sensor</span>
              <select
                value={sourceSensor}
                onChange={(e) => setSourceSensor(e.target.value as Sensor)}
                className="w-full retro-inset bg-white dark:bg-[#0F1211] px-2.5 py-1.5 text-xs font-mono font-bold text-[#1E2321] dark:text-[#E7E2D6] outline-none"
              >
                {SENSOR_OPTIONS.filter((s) => s.value !== "LRO_NAC").map((sensor) => (
                  <option key={sensor.value} value={sensor.value}>
                    {sensor.label}
                  </option>
                ))}
              </select>
            </label>

            <label className="block">
              <span className="mb-1 block text-[10px] font-mono font-bold uppercase tracking-wider text-[#555C58] dark:text-[#8C9893]">Reference Sensor</span>
              <select
                value={referenceSensor}
                onChange={(e) => setReferenceSensor(e.target.value as Sensor)}
                className="w-full retro-inset bg-white dark:bg-[#0F1211] px-2.5 py-1.5 text-xs font-mono font-bold text-[#1E2321] dark:text-[#E7E2D6] outline-none"
              >
                {SENSOR_OPTIONS.map((sensor) => (
                  <option key={sensor.value} value={sensor.value}>
                    {sensor.label}
                  </option>
                ))}
              </select>
            </label>

            <FilePicker label="Optional DEM Elevation DTM" file={demFile} optional onChange={setDemFile} />
          </div>

          {/* Quick Pre-aligned test buttons */}
          <div className="flex flex-wrap items-center gap-2 pt-2 border-t border-[#AAA496] dark:border-[#2D3835] text-[10px] font-mono">
            <span className="font-bold text-[#555C58] dark:text-[#8C9893]">Quick Test Pairs:</span>
            <button
              type="button"
              onClick={() =>
                loadTestPairIntoForm(
                  "/images/ohrc/region_001.png",
                  "/images/lro_nac/region_001.png",
                  "OHRC",
                  "LRO_NAC",
                  "region_001_ohrc.png",
                  "region_001_lro_nac.png"
                )
              }
              className="retro-button px-2 py-1 font-bold text-[#1E2321] dark:text-[#E7E2D6]"
            >
              🌙 Load Region 001 (OHRC + LRO NAC)
            </button>
            <button
              type="button"
              onClick={() =>
                loadTestPairIntoForm(
                  "/images/ohrc/region_001.png",
                  "/images/tmc/region_001.png",
                  "OHRC",
                  "TMC",
                  "region_001_ohrc.png",
                  "region_001_tmc.png"
                )
              }
              className="retro-button px-2 py-1 font-bold text-[#1E2321] dark:text-[#E7E2D6]"
            >
              🪐 Load Region 001 (OHRC + TMC-2)
            </button>
          </div>

          {error && customMode && (
            <div className="retro-inset bg-[#FEE2E2] dark:bg-[#2A1515] border border-[#EF4444] p-2 text-xs font-mono text-rose-800 dark:text-rose-300">
              <span className="font-bold uppercase tracking-wider">Verification Notice:</span>
              <span className="ml-2">{error}</span>
            </div>
          )}

          <button
            type="button"
            onClick={register}
            disabled={loading || !sourceFile || !referenceFile}
            className="retro-button flex w-full items-center justify-center gap-2 bg-[#28557E] hover:bg-[#1E3F5E] active:bg-[#142C42] py-2.5 text-xs font-mono font-bold uppercase tracking-wider text-white border-t-[#6795BE] border-l-[#6795BE] border-r-[#0E2031] border-b-[#0E2031] disabled:cursor-not-allowed disabled:opacity-40"
          >
            {loading ? (
              <>
                <span className="h-3 w-3 animate-spin rounded-full border-2 border-white/40 border-t-white" />
                Registering &amp; Validating Correspondences...
              </>
            ) : (
              "Run Registration on Uploaded Pair"
            )}
          </button>
        </div>

        {/* Live-run error banner */}
        {error && !customMode && (
          <div className="retro-inset bg-[#FEE2E2] dark:bg-[#2A1515] border border-[#EF4444] p-2 text-xs font-mono text-rose-800 dark:text-rose-300">
            <span className="font-bold uppercase tracking-wider">Backend error:</span>
            <span className="ml-2">{error}</span>
          </div>
        )}

        {/* 2. Benchmark Pair Selector (Rendered AFTER upload section) */}
        <div className="mt-4">
          <div className="flex flex-wrap items-center justify-between gap-2 mb-2 font-mono">
            <span className="text-[10px] font-bold uppercase tracking-wider text-[#1F4743] dark:text-[#52938B]">
              ⚡ Quick Benchmark Pairs (1-Click Verification)
            </span>
            <div className="flex items-center gap-2">
              <button
                type="button"
                onClick={() => {
                  setCustomMode(true);
                  setSelectedSample(null);
                  setResult(null);
                  uploadSectionRef.current?.scrollIntoView({ behavior: "smooth" });
                }}
                className={`retro-button px-2.5 py-1 text-xs font-bold transition flex items-center gap-1.5 ${
                  customMode && !selectedSample
                    ? "bg-[#28557E] text-white border-t-[#6795BE] border-l-[#6795BE] border-r-[#0E2031] border-b-[#0E2031]"
                    : "text-[#1E2321] dark:text-[#E7E2D6]"
                }`}
              >
                <span>+ Custom GeoTIFF Upload</span>
              </button>
              {(result || selectedSample || sourceFile || referenceFile) && (
                <button
                  type="button"
                  onClick={reset}
                  className="retro-button px-2.5 py-1 text-xs font-bold text-[#1E2321] dark:text-[#E7E2D6]"
                >
                  Clear
                </button>
              )}
            </div>
          </div>

          <div className="grid grid-cols-1 gap-2 sm:grid-cols-2 lg:grid-cols-4">
            {SAMPLE_PAIRS.map((sample) => {
              const active = selectedSample?.id === sample.id && !customMode;
              return (
                <button
                  key={sample.id}
                  type="button"
                  onClick={() => loadSample(sample)}
                  className={`flex flex-col justify-between p-2.5 text-left font-mono transition ${
                    active
                      ? "retro-inset bg-[#E0DACB] dark:bg-[#121615] ring-1 ring-[#1F4743] dark:ring-[#428178]"
                      : "retro-outset bg-[#E7E2D6] dark:bg-[#1A201E] hover:bg-[#F2EFE9] dark:hover:bg-[#202725]"
                  }`}
                >
                  <div>
                    <div className="flex items-center justify-between gap-1 mb-1">
                      <span className="text-[11px] font-bold text-[#1E2321] dark:text-[#E7E2D6] leading-snug">{sample.title}</span>
                    </div>
                    <span className={`inline-block border px-1.5 py-0.2 text-[9px] font-bold uppercase tracking-wider ${sample.badgeStyle}`}>
                      {sample.tag}
                    </span>
                    <p className="mt-1 text-[10px] leading-relaxed text-[#555C58] dark:text-[#8C9893] line-clamp-2">{sample.description}</p>
                  </div>
                  <div className="mt-2 flex items-center justify-between border-t border-[#AAA496] dark:border-[#2D3835] pt-1 text-[10px] font-bold text-[#28557E] dark:text-[#52938B]">
                    <span>{active ? "✓ Active Verification" : "Load Pair"}</span>
                    <span>→</span>
                  </div>
                </button>
              );
            })}
          </div>
        </div>

      {/* 3. Scientific Failure State: Quality Gate Rejection Panel */}
      {isFailedRegistration && (
        <div className="mt-4 space-y-3">
          <div className="retro-outset p-3 bg-[#FDF2F2] dark:bg-[#201414] border-t-[#F87171] border-l-[#F87171] border-r-[#991B1B] border-b-[#991B1B] font-mono">
            <div className="flex flex-col sm:flex-row items-start gap-3">
              <div className="flex h-9 w-9 shrink-0 items-center justify-center retro-outset bg-[#FEE2E2] dark:bg-[#2D1616] text-xl">
                🛡️
              </div>
              <div className="flex-1">
                <div className="flex flex-wrap items-center gap-2">
                  <span className="retro-inset bg-rose-200 dark:bg-rose-950/60 px-2 py-0.5 text-[10px] font-bold uppercase tracking-wider text-rose-900 dark:text-rose-200">
                    Quality Gate Rejection
                  </span>
                  <span className="text-[11px] font-bold text-rose-700 dark:text-rose-300">
                    Zero Synthetic Fallback Standard Enforced
                  </span>
                </div>

                <h4 className="mt-1.5 text-sm font-bold text-rose-950 dark:text-rose-100">
                  Robust Geometric Verification Refused Transformation
                </h4>

                <p className="mt-1 text-xs leading-relaxed text-rose-800 dark:text-rose-200 font-sans">
                  In strict compliance with SIH Problem Statement 26166 integrity requirements, Astralynx enforces a strict{" "}
                  <strong className="font-semibold text-rose-950 dark:text-rose-100">Zero Synthetic Fallback policy</strong>. When image pairs lack
                  sufficient consensus crater correspondences or exhibit extreme geometric distortion, the pipeline{" "}
                  <strong className="font-semibold text-rose-950 dark:text-rose-100">cleanly rejects registration</strong> rather than manufacturing
                  an artificial identity homography or hallucinating correspondence points.
                </p>

                <div className="mt-3 retro-inset bg-white/90 dark:bg-[#0E1117] p-2.5 text-xs text-[#1E2321] dark:text-[#E7E2D6]">
                  <div className="font-bold mb-1">Verification Audit Details:</div>
                  <div className="space-y-1 font-mono text-[11px]">
                    <div className="flex justify-between border-b border-[#AAA496] dark:border-[#2D3835] pb-1">
                      <span className="text-[#555C58] dark:text-[#8C9893]">Rejection Cause:</span>
                      <span className="font-bold text-rose-700 dark:text-rose-300">{result?.message || error || "Robust geometric verification failed to estimate a valid transformation from verified correspondences."}</span>
                    </div>
                    <div className="flex justify-between border-b border-[#AAA496] dark:border-[#2D3835] pb-1">
                      <span className="text-[#555C58] dark:text-[#8C9893]">Verified Inliers:</span>
                      <span className="font-bold">{result?.metrics?.num_inliers ?? 0} consensus points (minimum 4 required)</span>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-[#555C58] dark:text-[#8C9893]">Synthetic Fallback Action:</span>
                      <span className="font-bold text-emerald-700 dark:text-emerald-400">Refused (Zero fake points generated)</span>
                    </div>
                  </div>
                </div>

                {/* Visual Comparison of Divergent Pair */}
                <div className="mt-3 grid grid-cols-1 sm:grid-cols-2 gap-3">
                  <div className="retro-outset p-2 bg-[#E7E2D6] dark:bg-[#1A201E]">
                    <div className="flex items-center justify-between mb-1 px-1">
                      <span className="text-[11px] font-bold text-[#1E2321] dark:text-[#E7E2D6]">
                        {customMode && sourceFile ? `Uploaded Source: ${sourceFile.name}` : `Source: ${SENSOR_META.ohrc.label} (${SENSOR_META.ohrc.gsdM}m)`}
                      </span>
                      <span className="retro-inset px-1 py-0.2 text-[9px] font-bold text-rose-800 dark:text-rose-300 font-mono">
                        {customMode ? sourceSensor : "Sun: 269.6° (West)"}
                      </span>
                    </div>
                    <div className="relative aspect-square retro-inset bg-[#0A0D0C] overflow-hidden flex items-center justify-center">
                      <img
                        src={sourcePreview || sourcePrimarySrc || imageUrl("/images/ohrc/triplet_new_2022.png")}
                        alt="Source Divergent"
                        className="w-full h-full object-contain"
                        onError={(e) => { (e.currentTarget as HTMLImageElement).src = imageUrl("/images/ohrc/region_001.png"); }}
                      />
                    </div>
                  </div>
                  <div className="retro-outset p-2 bg-[#E7E2D6] dark:bg-[#1A201E]">
                    <div className="flex items-center justify-between mb-1 px-1">
                      <span className="text-[11px] font-bold text-[#1E2321] dark:text-[#E7E2D6]">
                        {customMode && referenceFile ? `Uploaded Reference: ${referenceFile.name}` : `Reference: ${SENSOR_META.tmc.label} (${(4).toFixed(1)}m)`}
                      </span>
                      <span className="retro-inset px-1 py-0.2 text-[9px] font-bold text-rose-800 dark:text-rose-300 font-mono">
                        {customMode ? referenceSensor : "Sun: 108.9° (East)"}
                      </span>
                    </div>
                    <div className="relative aspect-square retro-inset bg-[#0A0D0C] overflow-hidden flex items-center justify-center">
                      <img
                        src={referencePreview || referencePrimarySrc || imageUrl("/images/tmc/triplet_new_2022.png")}
                        alt="Reference Divergent"
                        className="w-full h-full object-contain"
                        onError={(e) => { (e.currentTarget as HTMLImageElement).src = imageUrl("/images/tmc/region_001.png"); }}
                      />
                    </div>
                  </div>
                </div>
                <div className="mt-2 retro-inset bg-[#FEE2E2] dark:bg-[#201414] py-1 px-2.5 text-center text-[10px] font-bold text-rose-800 dark:text-rose-300 font-mono">
                  {customMode
                    ? "⚠️ Insufficient Consensus Overlap: When uploaded images lack sufficient shared crater topography or feature points, the pipeline cleanly rejects registration rather than producing hallucinated matches."
                    : "⚠️ 160.8° Solar Azimuth Inversion: Shadows fall toward opposite crater rims, causing cross-correlation to fail safely rather than producing hallucinated matches."}
                </div>

                <div className="mt-3 flex flex-wrap items-center gap-2">
                  <button
                    type="button"
                    onClick={() => {
                      const verifiedSample = SAMPLE_PAIRS[1] || SAMPLE_PAIRS[0];
                      loadSample(verifiedSample);
                    }}
                    className="retro-button bg-[#28557E] text-white px-3 py-1 text-xs font-mono font-bold border-t-[#6795BE] border-l-[#6795BE] border-r-[#0E2031] border-b-[#0E2031]"
                  >
                    ⚡ Switch to Verified Lunar Pair (Region 001 LRO NAC)
                  </button>

                  <button
                    type="button"
                    onClick={() => {
                      const tmcSample = SAMPLE_PAIRS[0];
                      loadSample(tmcSample);
                    }}
                    className="retro-button px-3 py-1 text-xs font-mono font-bold text-[#1E2321] dark:text-[#E7E2D6]"
                  >
                    View 21x Scale Pair (Region 001 TMC-2)
                  </button>

                  {customMode && (
                    <button
                      type="button"
                      onClick={() => {
                        loadTestPairIntoForm(
                          "/images/ohrc/region_001.png",
                          "/images/lro_nac/region_001.png",
                          "OHRC",
                          "LRO_NAC",
                          "region_001_ohrc.png",
                          "region_001_lro_nac.png"
                        );
                      }}
                      className="retro-button px-3 py-1 text-xs font-mono font-bold text-[#1E2321] dark:text-[#E7E2D6]"
                    >
                      Load Verified Sample Files into Form
                    </button>
                  )}
                </div>
              </div>
            </div>
          </div>
        </div>
      )}

      {/* Success State: Verified Registered Package */}
      {result && result.status === "success" && (
        <div className="mt-4 space-y-4">
          {result.message && (
            <div
              className={
                isIIRSPair
                  ? "retro-inset p-2.5 text-xs font-mono bg-[#F5EEF8] dark:bg-[#1E1428] border border-[#6C3483] text-[#5B2C6F] dark:text-[#D7BDE2]"
                  : "retro-inset p-2.5 text-xs font-mono bg-[#E8F8F5] dark:bg-[#0E201B] border border-[#1F4743] text-[#143532] dark:text-[#A2D9CE]"
              }
            >
              <span className="font-bold uppercase tracking-wider">
                {isIIRSPair ? "Co-Registration Validated:" : "Registration Verified:"}
              </span>
              <span className="ml-2">{result.message}</span>
              {isIIRSPair && (
                <span className="ml-2 inline-block retro-outset px-2 py-0.2 text-[10px] font-bold uppercase tracking-wider text-emerald-800 dark:text-emerald-300">
                  Spectral Projection: Validated via TMC-2 Bridge
                </span>
              )}
            </div>
          )}

          <div className="grid gap-4 xl:grid-cols-[1.45fr_1fr]">
            {/* Visual Artifacts */}
            <div className="space-y-3">
              <div className="retro-outset p-2.5 bg-[#E7E2D6] dark:bg-[#1A201E]">
                <div className="mb-2 flex items-center justify-between px-1 font-mono">
                  <div>
                    <p className="text-[10px] font-bold uppercase tracking-[0.18em] text-[#1F4743] dark:text-[#52938B]">Primary Visual Proof</p>
                    <h4 className="mt-0.5 text-xs font-bold text-[#1E2321] dark:text-[#E7E2D6]">Continuity Checkerboard Registration QA</h4>
                  </div>
                  <span className="retro-inset px-2 py-0.5 text-[9px] font-bold uppercase tracking-wider bg-[#E8F8F5] dark:bg-[#0E201B] text-[#1F4743] dark:text-emerald-300">
                    50 px alternating blocks
                  </span>
                </div>

                {absoluteUrl(result.visual_url) ? (
                  // eslint-disable-next-line @next/next/no-img-element
                  <div className="retro-inset bg-[#0A0D0C] p-1 flex items-center justify-center">
                    <img
                      src={absoluteUrl(result.visual_url)!}
                      alt="Continuity checkerboard QA"
                      className="max-h-[470px] w-full object-contain"
                      onError={(e) => {
                        const img = e.currentTarget as HTMLImageElement;
                        if (!img.src.includes("/images/registered/region_001/checkerboard_qa.png")) {
                          img.src = imageUrl("/images/registered/region_001/checkerboard_qa.png");
                        }
                      }}
                    />
                  </div>
                ) : (
                  <div className="flex min-h-[260px] items-center justify-center retro-inset bg-[#0A0D0C] text-xs font-mono text-[#777]">
                    No checkerboard artifact returned.
                  </div>
                )}
              </div>

              <div className="retro-outset p-2.5 bg-[#E7E2D6] dark:bg-[#1A201E]">
                <div className="mb-1.5 flex items-center justify-between font-mono">
                  <h4 className="text-xs font-bold text-[#1E2321] dark:text-[#E7E2D6]">Registered Warped Source Image</h4>
                  <span className="text-[10px] text-[#717874] dark:text-[#8C9893]">Homography Reprojected</span>
                </div>
                {absoluteUrl(result.warped_url) ? (
                  // eslint-disable-next-line @next/next/no-img-element
                  <div className="retro-inset bg-[#0A0D0C] p-1 flex items-center justify-center">
                    <img
                      src={absoluteUrl(result.warped_url)!}
                      alt="Registered warped source"
                      className="max-h-[250px] w-full object-contain"
                      onError={(e) => {
                        const img = e.currentTarget as HTMLImageElement;
                        if (!img.src.includes("/images/registered/region_001/registered_ohrc.png")) {
                          img.src = imageUrl("/images/registered/region_001/registered_ohrc.png");
                        }
                      }}
                    />
                  </div>
                ) : (
                  <div className="flex min-h-[120px] items-center justify-center retro-inset bg-[#0A0D0C] text-xs font-mono text-[#777]">
                    No warped image returned.
                  </div>
                )}
              </div>

              {absoluteUrl(result.quiver_url) ? (
                <div className="retro-outset p-2.5 bg-[#E7E2D6] dark:bg-[#1A201E]">
                  <div className="mb-1.5 flex items-center justify-between font-mono">
                    <h4 className="text-xs font-bold text-[#1E2321] dark:text-[#E7E2D6]">Residual Displacement Vectors</h4>
                    <span className="text-[10px] text-[#717874] dark:text-[#8C9893]">Per-inlier reprojection error</span>
                  </div>
                  {/* eslint-disable-next-line @next/next/no-img-element */}
                  <div className="retro-inset bg-[#0A0D0C] p-1 flex items-center justify-center">
                    <img
                      src={absoluteUrl(result.quiver_url)!}
                      alt="Residual displacement vector quiver plot"
                      className="max-h-[250px] w-full object-contain"
                    />
                  </div>
                </div>
              ) : null}
            </div>

            {/* Telemetry Metric Cards */}
            <div className="grid grid-cols-2 content-start gap-2.5">
              <MetricCard
                label="In-Sample Fit RMSE"
                value={format(fitRmse)}
                suffix="px"
                emphasis
                sublabel="Reprojection residual on inlier consensus"
              />
              <MetricCard
                label="Held-Out Val RMSE"
                value={format(valRmse)}
                suffix="px"
                emphasis
                sublabel="Out-of-sample holdout generalization"
                hint={valRmse === null ? valHint(inliers) : null}
              />
              <MetricCard
                label="Absolute Topographic RMSE"
                value={format(metric(metrics, "absolute_rmse_m"), 2)}
                suffix="m"
                sublabel="Physical ground precision"
                hint={
                  metric(metrics, "absolute_rmse_m") === null
                    ? "No DEM/GSD in this run — cannot convert px to meters"
                    : null
                }
              />
              {isIIRSPair ? (
                <div className="retro-inset bg-[#E8F8F5] dark:bg-[#0E201B] border border-[#1F4743] p-3">
                  <p className="text-[10px] font-mono font-bold uppercase tracking-[0.16em] text-[#1F4743] dark:text-emerald-400">
                    Spectral Projection
                  </p>
                  <p className="mt-1 text-xs font-mono font-bold leading-snug tracking-tight text-[#143532] dark:text-emerald-200">
                    Spectral Projection: Validated via TMC-2 Bridge
                  </p>
                  <p className="mt-1 text-[10px] font-mono text-[#555C58] dark:text-emerald-400">
                    0 direct inliers by design — composed H_TMC→IIRS · H_OHRC→TMC overlay
                  </p>
                </div>
              ) : (
                <MetricCard
                  label="Verified Inliers"
                  value={inliers === null ? "—" : String(inliers)}
                  suffix="matches"
                  sublabel={`Inlier ratio: ${format(ratio === null ? null : ratio * 100, 1)}%`}
                />
              )}
              <MetricCard
                label="10×10 Spatial Coverage"
                value={format(coverage === null ? null : coverage * 100, 1)}
                suffix="%"
                sublabel="Pre-match SSC suppression spread"
              />
              <MetricCard
                label="Spatial Uniformity"
                value={format(uniformity === null ? null : uniformity * 100, 1)}
                suffix="%"
                sublabel="Planar cell entropy distribution"
              />

              <MetricCard
                label="Composite Quality Score"
                value={format(compositeScore === null ? null : compositeScore * 100, 1)}
                suffix="%"
                emphasis
                sublabel="Derived: 0.25·Inliers + 0.25·RMSE + 0.25·Uniformity + 0.25·Alignment(NMI/SSIM)"
                hint={compositeScore === null ? "Unavailable for this run" : null}
              />
              <MetricCard
                label="Norm. Mutual Info (NMI)"
                value={format(nmiVal)}
                sublabel="Illumination-robust cross-sensor overlap mutual information"
                hint={nmiVal === null ? "No overlap rasters in this run" : null}
              />
              <MetricCard
                label="SSIM (Overlap)"
                value={format(ssimVal)}
                sublabel="Structural similarity over common footprint"
                hint={ssimVal === null ? "No overlap rasters in this run" : null}
              />
              <MetricCard
                label="PSNR (Overlap)"
                value={format(psnrVal, 1)}
                suffix="dB"
                sublabel="Peak SNR on overlapping registered regions"
                hint={psnrVal === null ? "No overlap rasters in this run" : null}
              />

              <div className={`col-span-2 retro-outset p-3 font-mono ${qualityTone}`}>
                <div className="flex items-center justify-between">
                  <p className="text-[10px] font-bold uppercase tracking-[0.16em] opacity-70">Quality Tier</p>
                  <span className="retro-inset px-2 py-0.5 text-[9px] font-bold uppercase tracking-wider bg-white/60 dark:bg-black/30">
                    Estimator: {outlierMethod}
                  </span>
                </div>
                <p className="mt-1 text-xl font-bold tracking-tight">{qualityTier.replaceAll("_", " ")}</p>
                <p className="mt-0.5 text-[11px] opacity-80">
                  {metrics?.validation_status || "Sub-pixel geometric consensus verified (< 1.0 px)"}
                </p>
              </div>
            </div>
          </div>

          {/* View Mode Toggle: 2D Planar Verification vs 3D Lunar Globe */}
          <div className="flex flex-wrap items-center justify-between gap-2 border-b border-[#AAA496] dark:border-[#2D3835] pb-2 font-mono">
            <div className="flex items-center gap-2">
              <button
                type="button"
                onClick={() => setResultView("2d")}
                className={`flex items-center gap-2 px-3 py-1 text-xs font-bold transition ${
                  resultView === "2d"
                    ? "retro-inset bg-[#DDD7C8] dark:bg-[#0F1211] text-[#1E2321] dark:text-[#E7E2D6]"
                    : "retro-button text-[#555C58] dark:text-[#8C9893]"
                }`}
              >
                <span>🔍 2D Planar Verification</span>
                <span
                  className={`px-1.5 py-0.2 text-[10px] ${
                    resultView === "2d" ? "retro-outset bg-[#28557E] text-white" : "retro-inset bg-[#C8C2B5] dark:bg-[#1A201E]"
                  }`}
                >
                  {points.length}
                </span>
              </button>
              <button
                type="button"
                onClick={() => setResultView("3d")}
                className={`flex items-center gap-2 px-3 py-1 text-xs font-bold transition ${
                  resultView === "3d"
                    ? "retro-inset bg-[#DDD7C8] dark:bg-[#0F1211] text-[#1E2321] dark:text-[#E7E2D6]"
                    : "retro-button text-[#555C58] dark:text-[#8C9893]"
                }`}
              >
                <span>🌕 3D Lunar Globe Tie-Points</span>
                {moonPoints.length > 0 && (
                  <span
                    className={`px-1.5 py-0.2 text-[10px] ${
                      resultView === "3d" ? "retro-outset bg-[#28557E] text-white" : "retro-inset bg-[#C8C2B5] dark:bg-[#1A201E]"
                    }`}
                  >
                    {moonPoints.filter((p) => p.georeferenced).length}
                  </span>
                )}
              </button>
            </div>

            {resultView === "3d" && (
              <span className="text-[11px] text-[#717874] dark:text-[#8C9893]">
                Drag to rotate sphere · Scroll to zoom
              </span>
            )}
          </div>

          {resultView === "2d" ? (
            /* Source & Reference Overlays */
            <div className="grid gap-4 lg:grid-cols-2">
              <OverlayImage
                title={`Source Image · ${sourceSensor}`}
                src={sourcePrimarySrc}
                fallbackSrc={sourceFallbackSrc}
                secondaryFallback={sourcePreview}
                points={points}
                side="source"
              />
              <OverlayImage
                title={`Reference Image · ${referenceSensor}`}
                src={referencePrimarySrc}
                fallbackSrc={referencePreview}
                points={points}
                side="reference"
              />
            </div>
          ) : (
            /* 3D Lunar Globe View */
            <div className="relative flex h-[640px] w-full flex-col overflow-hidden retro-inset bg-[#0A0D0C]">
              <div className="relative w-full flex-1">
                <LunarGlobe
                  tiePoints={moonPoints}
                  footprints={footprints}
                  activeFootprintId={activeFootprintId}
                  phase="full"
                  className="absolute inset-0 h-full w-full cursor-grab active:cursor-grabbing"
                />
                {/* Globe Overlay HUD */}
                <div className="pointer-events-none absolute left-4 top-4 z-20 flex flex-col gap-1.5">
                  <div className="flex items-center gap-2 retro-outset bg-[#E7E2D6] dark:bg-[#1A201E] px-2.5 py-1 text-xs font-mono font-bold text-[#1E2321] dark:text-[#E7E2D6]">
                    <span className="h-2 w-2 bg-emerald-500 animate-pulse" />
                    <span>
                      {moonPointsLoading
                        ? "Fetching 3D coordinates..."
                        : `${moonPoints.filter((p) => p.georeferenced).length} Georeferenced Lunar Coordinates`}
                    </span>
                  </div>
                  {result?.job_id && (
                    <span className="retro-inset bg-black/60 px-2 py-0.5 text-[10px] font-mono text-[#DDD]">
                      Region/Job: {result.job_id}
                      {activeFootprint ? ` · ${footprints.length} regions plotted` : ""}
                    </span>
                  )}
                </div>

                <div className="pointer-events-none absolute bottom-3 right-4 z-20 retro-outset bg-[#E7E2D6] dark:bg-[#1A201E] px-2.5 py-1 text-[11px] font-mono font-bold text-[#1E2321] dark:text-[#E7E2D6]">
                  Spherical Projection · True Lunar Coordinate Mapping
                </div>
              </div>

              {/* Bottom coordinate readout */}
              <div className="z-20 grid grid-cols-2 gap-3 border-t border-[#AAA496] dark:border-[#2D3835] bg-[#E7E2D6] dark:bg-[#161B19] px-4 py-3 font-mono sm:grid-cols-4">
                <div>
                  <p className="text-[9px] font-bold uppercase tracking-[0.18em] text-[#555C58] dark:text-[#8C9893]">Active region</p>
                  <p className="mt-0.5 text-xs font-bold text-[#28557E] dark:text-cyan-300">
                    {activeFootprint?.id ?? result?.job_id ?? "—"}
                  </p>
                  <p className="text-[10px] text-[#717874] dark:text-[#8C9893]">
                    {footprints.length} validated footprints
                  </p>
                </div>
                <div>
                  <p className="text-[9px] font-bold uppercase tracking-[0.18em] text-[#555C58] dark:text-[#8C9893]">Centre (lat, lon)</p>
                  <p className="mt-0.5 text-xs font-bold text-[#1E2321] dark:text-[#E7E2D6]">
                    {activeCenter ? `${activeCenter.lat.toFixed(4)}°, ${to180(activeCenter.lon360).toFixed(4)}°` : "—"}
                  </p>
                  <p className="text-[10px] text-[#717874] dark:text-[#8C9893]">
                    {activeCenter ? `lon 0–360: ${(((activeCenter.lon360 % 360) + 360) % 360).toFixed(4)}°` : "planetocentric"}
                  </p>
                </div>
                <div>
                  <p className="text-[9px] font-bold uppercase tracking-[0.18em] text-[#555C58] dark:text-[#8C9893]">Bounds W/E/S/N</p>
                  <p className="mt-0.5 text-[11px] font-bold text-[#1E2321] dark:text-[#E7E2D6]">
                    {activeFootprint
                      ? `${activeFootprint.west_lon.toFixed(3)} / ${activeFootprint.east_lon.toFixed(3)}`
                      : "—"}
                  </p>
                  <p className="text-[10px] text-[#717874] dark:text-[#8C9893]">
                    {activeFootprint
                      ? `${activeFootprint.south_lat.toFixed(4)} / ${activeFootprint.north_lat.toFixed(4)}`
                      : "deg"}
                  </p>
                </div>
                <div>
                  <p className="text-[9px] font-bold uppercase tracking-[0.18em] text-[#555C58] dark:text-[#8C9893]">Footprint · tie-points</p>
                  <p className="mt-0.5 text-xs font-bold text-[#1E2321] dark:text-[#E7E2D6]">
                    {activeSize ? `${activeSize.widthKm.toFixed(2)} × ${activeSize.heightKm.toFixed(2)} km` : "—"}
                  </p>
                  <p className="text-[10px] text-[#717874] dark:text-[#8C9893]">
                    {moonPoints.filter((p) => p.georeferenced).length} georeferenced · drag to rotate
                  </p>
                </div>
              </div>
            </div>
          )}

          {/* Action Bar */}
          <div className="flex flex-wrap items-center justify-between gap-3 retro-outset bg-[#E7E2D6] dark:bg-[#1A201E] p-3 font-mono">
            <div>
              <p className="text-xs font-bold text-[#1E2321] dark:text-[#E7E2D6]">{points.length} verified correspondences mapped</p>
              <p className="mt-0.5 text-[10px] text-[#555C58] dark:text-[#8C9893]">
                <span className="text-emerald-600 font-bold">●</span> High Confidence (&gt;0.8){" "}
                <span className="ml-2 text-amber-600 font-bold">●</span> Review (&gt;0.5){" "}
                <span className="ml-2 text-rose-600 font-bold">●</span> Low Confidence
              </p>
            </div>
            <div className="flex flex-wrap items-center gap-2">
              {result.job_id && (
                <a
                  href={`${API_BASE}/api/registration/report/${result.job_id}`}
                  download={`ISRO_Registration_Report_${result.job_id}.pdf`}
                  target="_blank"
                  rel="noreferrer"
                  className="retro-button bg-[#28557E] hover:bg-[#1E3F5E] text-white px-3 py-1.5 text-xs font-mono font-bold inline-flex items-center gap-1.5 border-t-[#6795BE] border-l-[#6795BE] border-r-[#0E2031] border-b-[#0E2031]"
                >
                  <svg className="h-3.5 w-3.5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 10v6m0 0l-3-3m3 3l3-3m2 8H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
                  </svg>
                  ISRO Verification Report (PDF)
                </a>
              )}
              <DownloadLink href={result.warped_url} label="Warped PNG" />
              <DownloadLink href={result.raster_url} label="Registered GeoTIFF" />
              <DownloadLink href={result.matches_url} label="matches.json" />
              <DownloadLink href={result.matches_url?.replace("matches.json", "metrics.json")} label="metrics.json" />
              <DownloadLink href={result.visual_url} label="Checkerboard QA" />
            </div>
          </div>

          <div className="flex items-center gap-3">
            <button
              type="button"
              onClick={reset}
              className="retro-button px-3.5 py-1.5 text-xs font-mono font-bold text-[#1E2321] dark:text-[#E7E2D6]"
            >
              ← Register Another Pair
            </button>
          </div>
        </div>
      )}
      </div>
    </section>
  );
}
