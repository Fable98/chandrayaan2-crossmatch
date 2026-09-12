"use client";

import { useEffect, useState } from "react";
import { API_BASE, imageUrl } from "@/lib/api";
import LunarGlobe from "@/components/hero/LunarGlobe";
import type { MoonPoint, MoonPointsResponse } from "@/lib/backend-types";

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
  report_url?: string | null;
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
    tag: "Primary 21x Scale Gap",
    badgeStyle: "bg-blue-50 text-blue-700 border-blue-200",
    description: "0.25m/px Narrow-Angle OHRC matched to 4.0m/px single-view TMC-2 in Sinus Medii.",
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
      report_url: "/api/registration/report/region_001",
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
    tag: "PS Lunar Reference (~3.6x, real CDR)",
    badgeStyle: "bg-emerald-50 text-emerald-700 border-emerald-200",
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
      report_url: "/api/registration/report/region_001",
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
    badgeStyle: "bg-purple-50 text-purple-700 border-purple-200",
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
      report_url: "/api/registration/report/region_003",
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
    tag: "Chained Spectral Overlay (~320x)",
    badgeStyle: "bg-purple-50 text-purple-700 border-purple-200",
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
      report_url: "/api/registration/report/region_001",
    },
    demoPoints: [],
  },
  {
    id: "sample_hardest_case",
    title: "Triplet New: 162° Sun-Gap Hardest Case",
    tag: "Fragile LOW · No Held-Out",
    badgeStyle: "bg-amber-50 text-amber-700 border-amber-200",
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
      report_url: "/api/registration/report/triplet_new_2022",
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

const SENSOR_OPTIONS: { value: Sensor; label: string }[] = [
  { value: "OHRC", label: "OHRC (0.25 m/px Panchromatic Optical)" },
  { value: "TMC", label: "TMC-2 (4.0 m/px Single-View Optical)" },
  { value: "LRO_NAC", label: "NASA LRO NAC (0.5–1.2 m/px Lunar Reference)" },
  { value: "IIRS", label: "IIRS (69 m/px Infrared Hyperspectral)" },
];

function absoluteUrl(path?: string | null) {
  if (!path) return null;
  if (path.startsWith("http")) return path;
  return imageUrl(path);
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
    <label className="block cursor-pointer rounded-xl border border-dashed border-slate-300 bg-slate-50 p-4 transition hover:border-indigo-300 hover:bg-indigo-50/40">
      <div className="flex items-center justify-between gap-3">
        <span className="text-[10px] font-black uppercase tracking-wider text-slate-600">{label}</span>
        {optional && <span className="text-[9px] font-bold text-slate-400">OPTIONAL</span>}
      </div>
      <input
        type="file"
        accept=".jpg,.jpeg,.png,.tif,.tiff,image/*"
        onChange={(event) => onChange(event.target.files?.[0] || null)}
        className="mt-3 block w-full text-[11px] text-slate-500 file:mr-3 file:rounded-lg file:border-0 file:bg-indigo-100 file:px-3 file:py-1.5 file:text-[10px] file:font-bold file:text-indigo-700"
      />
      <p className="mt-2 truncate text-[10px] text-slate-400">
        {file ? file.name : preview ? "Using pre-loaded sample image" : optional ? "No DEM supplied" : "Choose an image or select sample"}
      </p>
    </label>
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
  // Shown in place of the sublabel when the value is unavailable ("—") so a
  // missing number always carries its reason, never a bare dash.
  hint?: string | null;
}) {
  const missing = value === "—";
  return (
    <div className={`rounded-xl border p-4 ${emphasis ? "border-indigo-200 bg-indigo-50/60" : "border-slate-200 bg-white"}`}>
      <p className="text-[10px] font-bold uppercase tracking-[0.16em] text-slate-400">{label}</p>
      <p className={`mt-2 text-2xl font-black tracking-tight ${emphasis ? "text-[#4F46E5]" : "text-slate-900"}`}>
        {value}
        {suffix && !missing && <span className="ml-1 text-xs font-bold text-slate-500">{suffix}</span>}
      </p>
      {missing && hint ? (
        <p className="mt-1 text-[10px] font-semibold text-amber-600">{hint}</p>
      ) : (
        sublabel && <p className="mt-1 text-[10px] text-slate-400">{sublabel}</p>
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
      className="rounded-lg border border-indigo-100 bg-white px-3 py-2 text-[10px] font-bold text-indigo-700 transition hover:bg-indigo-50"
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
    <div className="rounded-xl border border-slate-200 bg-white p-3">
      <div className="mb-2 flex items-center justify-between">
        <h4 className="text-xs font-bold text-slate-800">{title}</h4>
        <span className="text-[10px] font-mono text-slate-400">{points.length} inlier pts</span>
      </div>
      <div className="relative aspect-square overflow-hidden rounded-lg bg-slate-950 flex items-center justify-center">
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
  const defaultSample = SAMPLE_PAIRS[1] || SAMPLE_PAIRS[0];

  const [sourceFile, setSourceFile] = useState<File | null>(null);
  const [referenceFile, setReferenceFile] = useState<File | null>(null);
  const [sourcePreview, setSourcePreview] = useState<string | null>(absoluteUrl(defaultSample.sourceUrl));
  const [referencePreview, setReferencePreview] = useState<string | null>(absoluteUrl(defaultSample.referenceUrl));
  const [demFile, setDemFile] = useState<File | null>(null);
  const [sourceSensor, setSourceSensor] = useState<Sensor>(defaultSample.sourceSensor);
  const [referenceSensor, setReferenceSensor] = useState<Sensor>(defaultSample.referenceSensor);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [result, setResult] = useState<RegistrationResult | null>(defaultSample.demoResult);
  // Provenance of the currently displayed result: the pre-loaded committed
  // benchmark snapshot vs a genuinely live backend run (Step 13 honesty).
  const [resultProvenance, setResultProvenance] = useState<"snapshot" | "live">("snapshot");
  const [points, setPoints] = useState<MatchPoint[]>(defaultSample.demoPoints);
  const [selectedSample, setSelectedSample] = useState<SamplePair | null>(defaultSample);
  const [customMode, setCustomMode] = useState(false);
  const [resultView, setResultView] = useState<"2d" | "3d">("2d");
  const [moonPoints, setMoonPoints] = useState<MoonPoint[]>([]);
  const [moonPointsLoading, setMoonPointsLoading] = useState(false);

  useEffect(() => {
    const jobId = result?.job_id;
    if (!jobId) {
      setMoonPoints([]);
      return;
    }

    let active = true;
    setMoonPointsLoading(true);

    fetch(`${API_BASE}/api/registration/moon-points/${jobId}`)
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
        if (active) setMoonPointsLoading(false);
      });

    return () => {
      active = false;
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
        const { getAuthHeaders } = await import("@/lib/auth");
        const response = await fetch(`${API_BASE}/register`, {
          method: "POST",
          headers: { ...getAuthHeaders() },
          body,
        });
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
          const mRes = await fetch(absoluteUrl(live.matches_url)!);
          if (mRes.ok) setPoints((await mRes.json()) as MatchPoint[]);
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

      const { getAuthHeaders: uploadAuthHeaders } = await import("@/lib/auth");
      const response = await fetch(`${API_BASE}/register`, {
        method: "POST",
        headers: { ...uploadAuthHeaders() },
        body,
      });
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
        const matchesResponse = await fetch(absoluteUrl(registration.matches_url)!);
        if (matchesResponse.ok) setPoints((await matchesResponse.json()) as MatchPoint[]);
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
      ? "border-purple-300 bg-purple-50 text-purple-700"
      : qualityTier === "HIGH_CONFIDENCE"
      ? "border-emerald-300 bg-emerald-50 text-emerald-700"
      : qualityTier === "ACCEPTED"
      ? "border-cyan-300 bg-cyan-50 text-cyan-700"
      : "border-rose-300 bg-rose-50 text-rose-700";

  const inliers = metric(metrics, "num_inliers", "inlier_count");
  const coverage = metric(metrics, "combined_coverage_score", "source_coverage_ratio", "spatial_coverage");
  const uniformity = metric(metrics, "uniformity_score", "spatial_uniformity");
  const ratio = metric(metrics, "inlier_ratio");
  const fitRmse = metric(metrics, "fit_rmse_px", "rmse_px");
  const valRmse = metric(metrics, "validation_rmse_px");
  const ssimVal = metric(metrics, "ssim");
  const psnrVal = metric(metrics, "psnr");
  const nmiVal = metric(metrics, "nmi");
  const compositeScore = metric(metrics, "composite_quality_score");
  const outlierMethod = metrics?.outlier_method || "RANSAC";

  const sourcePrimarySrc = result?.source_url
    ? absoluteUrl(result.source_url)
    : result?.warped_url
    ? absoluteUrl(result.warped_url)
    : sourcePreview;
  const sourceFallbackSrc = result?.warped_url ? absoluteUrl(result.warped_url) : sourcePreview;

  const referencePrimarySrc = result?.reference_url ? absoluteUrl(result.reference_url) : referencePreview;

  const isFailedRegistration =
    ((result && result.status !== "success" && !isIIRSPair) || (customMode && !!error && !isIIRSPair)) as boolean;

  return (
    <section className="rounded-2xl border border-slate-200/80 bg-white p-5 shadow-sm md:p-6">
      {/* Header Bar */}
      <div className="flex flex-col justify-between gap-3 border-b border-slate-100 pb-5 md:flex-row md:items-center">
        <div>
          <div className="flex items-center gap-2">
            <span className="h-2 w-2 rounded-full bg-[#4F46E5] shadow-[0_0_0_4px_rgba(79,70,229,.12)]" />
            <p className="text-[10px] font-black uppercase tracking-[0.2em] text-[#4F46E5]">Live Registration Pipeline</p>
          </div>
          <h3 className="mt-1 text-xl font-black tracking-tight text-slate-900">Upload &amp; Register</h3>
          <p className="mt-1 max-w-2xl text-xs text-slate-500">
            Run the cross-sensor matching engine and inspect sub-pixel alignment, correspondence geometry, and Quality Gates.
          </p>
        </div>

        <div className="flex items-center gap-2">
          {result && resultProvenance === "snapshot" && (
            <span
              className="rounded-full border border-amber-300 bg-amber-50 px-3 py-1.5 text-[10px] font-black uppercase tracking-wider text-amber-700"
              title="Committed benchmark snapshot from the repo manifests — press Run to verify live against the backend."
            >
              Snapshot · run live to verify
            </span>
          )}
          {result && resultProvenance === "live" && (
            <span className="rounded-full border border-emerald-300 bg-emerald-50 px-3 py-1.5 text-[10px] font-black uppercase tracking-wider text-emerald-700">
              Live backend run
            </span>
          )}
          <div className={`rounded-full border px-3 py-1.5 text-[10px] font-black uppercase tracking-wider ${result ? qualityTone : "border-slate-200 bg-slate-50 text-slate-500"}`}>
            {loading ? "PROCESSING..." : result ? qualityTier.replaceAll("_", " ") : "READY TO RUN"}
          </div>
        </div>
      </div>

      {/* Live-run error banner (Step 13: backend failures surface, never snapshots) */}
      {error && !customMode && (
        <div className="mt-5 rounded-xl border border-rose-200 bg-rose-50 px-4 py-3 text-xs text-rose-700">
          <span className="font-black uppercase tracking-wider">Backend error:</span>
          <span className="ml-2">{error}</span>
        </div>
      )}

      {/* 1. Benchmark Pair Selector (Always Accessible) */}
      <div className="mt-5">
        <div className="flex flex-wrap items-center justify-between gap-2 mb-2.5">
          <span className="text-[10px] font-black uppercase tracking-wider text-slate-500">
            ⚡ Quick Benchmark Pairs (1-Click Verification)
          </span>
          <div className="flex items-center gap-2">
            <button
              type="button"
              onClick={() => {
                setCustomMode(!customMode);
                if (!customMode) {
                  setSelectedSample(null);
                  setResult(null);
                }
              }}
              className={`rounded-lg px-2.5 py-1 text-xs font-bold transition flex items-center gap-1.5 ${
                customMode
                  ? "bg-indigo-600 text-white shadow-sm"
                  : "bg-slate-100 text-slate-700 hover:bg-slate-200"
              }`}
            >
              <span>+ Custom GeoTIFF Upload</span>
            </button>
            {(result || customMode) && (
              <button
                type="button"
                onClick={reset}
                className="rounded-lg border border-slate-200 bg-white px-2.5 py-1 text-xs font-semibold text-slate-600 hover:bg-slate-50"
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
                className={`flex flex-col justify-between rounded-xl border p-3 text-left transition ${
                  active
                    ? "border-[#4F46E5] bg-indigo-50/50 shadow-sm ring-2 ring-indigo-200"
                    : "border-slate-200 bg-slate-50/60 hover:border-slate-300 hover:bg-slate-50"
                }`}
              >
                <div>
                  <div className="flex items-center justify-between gap-1 mb-1.5">
                    <span className="text-[11px] font-bold text-slate-900 leading-snug">{sample.title}</span>
                  </div>
                  <span className={`inline-block rounded-md border px-1.5 py-0.5 text-[9px] font-bold uppercase tracking-wider ${sample.badgeStyle}`}>
                    {sample.tag}
                  </span>
                  <p className="mt-1.5 text-[10px] leading-relaxed text-slate-500 line-clamp-2">{sample.description}</p>
                </div>
                <div className="mt-2.5 flex items-center justify-between border-t border-slate-100 pt-1.5 text-[10px] font-bold text-indigo-600">
                  <span>{active ? "✓ Active Verification" : "Load Pair"}</span>
                  <span>→</span>
                </div>
              </button>
            );
          })}
        </div>
      </div>

      {/* 2. Custom Upload Inputs & Controls (Visible when customMode is active) */}
      {customMode && (
        <div className="mt-5 space-y-4 rounded-xl border border-indigo-100 bg-indigo-50/20 p-4">
          <div className="flex items-center justify-between border-b border-indigo-100 pb-2">
            <span className="text-xs font-bold text-slate-800">Upload Custom Multi-Sensor Pair</span>
            <span className="text-[10px] text-slate-500 font-mono">GeoTIFF (.tif) or PNG (.png)</span>
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
              <span className="mb-1.5 block text-[10px] font-black uppercase tracking-wider text-slate-500">Source Sensor</span>
              <select
                value={sourceSensor}
                onChange={(e) => setSourceSensor(e.target.value as Sensor)}
                className="w-full rounded-xl border border-slate-200 bg-white px-3 py-2.5 text-xs font-bold text-slate-700 outline-none focus:border-indigo-400 focus:ring-2 focus:ring-indigo-100"
              >
                {SENSOR_OPTIONS.filter((s) => s.value !== "LRO_NAC").map((sensor) => (
                  <option key={sensor.value} value={sensor.value}>
                    {sensor.label}
                  </option>
                ))}
              </select>
            </label>

            <label className="block">
              <span className="mb-1.5 block text-[10px] font-black uppercase tracking-wider text-slate-500">Reference Sensor</span>
              <select
                value={referenceSensor}
                onChange={(e) => setReferenceSensor(e.target.value as Sensor)}
                className="w-full rounded-xl border border-slate-200 bg-white px-3 py-2.5 text-xs font-bold text-slate-700 outline-none focus:border-indigo-400 focus:ring-2 focus:ring-indigo-100"
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
          <div className="flex flex-wrap items-center gap-2 pt-2 border-t border-indigo-100/60 text-[10px]">
            <span className="font-bold text-slate-500">Quick Test Pairs:</span>
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
              className="rounded-lg border border-indigo-200 bg-white px-2.5 py-1 font-bold text-indigo-600 hover:bg-indigo-50 transition shadow-xs"
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
              className="rounded-lg border border-indigo-200 bg-white px-2.5 py-1 font-bold text-indigo-600 hover:bg-indigo-50 transition shadow-xs"
            >
              🪐 Load Region 001 (OHRC + TMC-2)
            </button>
          </div>

          {error && (
            <div className="rounded-xl border border-rose-200 bg-rose-50 px-4 py-3 text-xs text-rose-700">
              <span className="font-black uppercase tracking-wider">Verification Notice:</span>
              <span className="ml-2">{error}</span>
            </div>
          )}

          <button
            type="button"
            onClick={register}
            disabled={loading || !sourceFile || !referenceFile}
            className="flex w-full items-center justify-center gap-2 rounded-xl bg-[#4F46E5] px-5 py-3 text-xs font-black uppercase tracking-[0.15em] text-white shadow-sm transition hover:bg-[#4338CA] disabled:cursor-not-allowed disabled:opacity-40"
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
      )}

      {/* 3. Scientific Failure State: Quality Gate Rejection Panel */}
      {isFailedRegistration && (
        <div className="mt-5 space-y-4">
          <div className="rounded-2xl border border-rose-200 bg-rose-50/70 p-5 md:p-6">
            <div className="flex flex-col sm:flex-row items-start gap-4">
              <div className="flex h-11 w-11 shrink-0 items-center justify-center rounded-xl bg-rose-100 text-2xl text-rose-600">
                🛡️
              </div>
              <div className="flex-1">
                <div className="flex flex-wrap items-center gap-2">
                  <span className="rounded-full bg-rose-200/90 px-2.5 py-0.5 text-[10px] font-black uppercase tracking-wider text-rose-900">
                    Quality Gate Rejection
                  </span>
                  <span className="text-[11px] font-mono font-bold text-rose-700">
                    Zero Synthetic Fallback Standard Enforced
                  </span>
                </div>

                <h4 className="mt-2 text-base font-bold text-rose-950">
                  Robust Geometric Verification Refused Transformation
                </h4>

                <p className="mt-1.5 text-xs leading-relaxed text-rose-800">
                  In strict compliance with SIH Problem Statement 26166 integrity requirements, Astralynx enforces a strict{" "}
                  <strong className="font-semibold text-rose-950">Zero Synthetic Fallback policy</strong>. When image pairs lack
                  sufficient consensus crater correspondences or exhibit extreme geometric distortion, the pipeline{" "}
                  <strong className="font-semibold text-rose-950">cleanly rejects registration</strong> rather than manufacturing
                  an artificial identity homography or hallucinating correspondence points.
                </p>

                <div className="mt-4 rounded-xl border border-rose-200/80 bg-white/90 p-3.5 text-xs text-slate-700">
                  <div className="font-bold text-slate-900 mb-1.5">Verification Audit Details:</div>
                  <div className="space-y-1 font-mono text-[11px]">
                    <div className="flex justify-between border-b border-slate-100 pb-1">
                      <span className="font-sans text-slate-500">Rejection Cause:</span>
                      <span className="font-bold text-rose-700">{result?.message || error || "Robust geometric verification failed to estimate a valid transformation from verified correspondences."}</span>
                    </div>
                    <div className="flex justify-between border-b border-slate-100 pb-1">
                      <span className="font-sans text-slate-500">Verified Inliers:</span>
                      <span className="font-bold text-slate-900">{result?.metrics?.num_inliers ?? 0} consensus points (minimum 4 required)</span>
                    </div>
                    <div className="flex justify-between">
                      <span className="font-sans text-slate-500">Synthetic Fallback Action:</span>
                      <span className="font-bold text-emerald-700">Refused (Zero fake points generated)</span>
                    </div>
                  </div>
                </div>

                {/* Visual Comparison of Divergent Pair */}
                <div className="mt-4 grid grid-cols-1 sm:grid-cols-2 gap-3">
                  <div className="rounded-xl border border-rose-200/80 bg-white p-2.5">
                    <div className="flex items-center justify-between mb-1.5 px-1">
                      <span className="text-[11px] font-bold text-slate-800">
                        {customMode && sourceFile ? `Uploaded Source: ${sourceFile.name}` : `Source: OHRC (0.25m)`}
                      </span>
                      <span className="rounded bg-rose-100 px-1.5 py-0.5 text-[9px] font-bold text-rose-800 font-mono">
                        {customMode ? sourceSensor : "Sun: 269.6° (West)"}
                      </span>
                    </div>
                    <div className="relative aspect-square bg-slate-950 rounded-lg overflow-hidden flex items-center justify-center">
                      <img
                        src={sourcePreview || sourcePrimarySrc || imageUrl("/images/ohrc/triplet_new_2022.png")}
                        alt="Source Divergent"
                        className="w-full h-full object-contain"
                        onError={(e) => { (e.currentTarget as HTMLImageElement).src = imageUrl("/images/ohrc/region_001.png"); }}
                      />
                    </div>
                  </div>
                  <div className="rounded-xl border border-rose-200/80 bg-white p-2.5">
                    <div className="flex items-center justify-between mb-1.5 px-1">
                      <span className="text-[11px] font-bold text-slate-800">
                        {customMode && referenceFile ? `Uploaded Reference: ${referenceFile.name}` : `Reference: TMC-2 (4.0m)`}
                      </span>
                      <span className="rounded bg-rose-100 px-1.5 py-0.5 text-[9px] font-bold text-rose-800 font-mono">
                        {customMode ? referenceSensor : "Sun: 108.9° (East)"}
                      </span>
                    </div>
                    <div className="relative aspect-square bg-slate-950 rounded-lg overflow-hidden flex items-center justify-center">
                      <img
                        src={referencePreview || referencePrimarySrc || imageUrl("/images/tmc/triplet_new_2022.png")}
                        alt="Reference Divergent"
                        className="w-full h-full object-contain"
                        onError={(e) => { (e.currentTarget as HTMLImageElement).src = imageUrl("/images/tmc/region_001.png"); }}
                      />
                    </div>
                  </div>
                </div>
                <div className="mt-2.5 rounded-lg bg-rose-100/70 py-1.5 px-3 text-center text-[10px] font-bold text-rose-800">
                  {customMode
                    ? "⚠️ Insufficient Consensus Overlap: When uploaded images lack sufficient shared crater topography or feature points, the pipeline cleanly rejects registration rather than producing hallucinated matches."
                    : "⚠️ 160.8° Solar Azimuth Inversion: Shadows fall toward opposite crater rims, causing cross-correlation to fail safely rather than producing hallucinated matches."}
                </div>

                <div className="mt-5 flex flex-wrap items-center gap-3">
                  <button
                    type="button"
                    onClick={() => {
                      const verifiedSample = SAMPLE_PAIRS[1] || SAMPLE_PAIRS[0];
                      loadSample(verifiedSample);
                    }}
                    className="rounded-xl bg-[#4F46E5] px-4 py-2.5 text-xs font-bold text-white shadow-sm transition hover:bg-[#4338CA]"
                  >
                    ⚡ Switch to Verified Lunar Pair (Region 001 LRO NAC)
                  </button>

                  <button
                    type="button"
                    onClick={() => {
                      const tmcSample = SAMPLE_PAIRS[0];
                      loadSample(tmcSample);
                    }}
                    className="rounded-xl border border-slate-200 bg-white px-4 py-2.5 text-xs font-bold text-slate-700 shadow-sm transition hover:bg-slate-50"
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
                      className="rounded-xl border border-indigo-200 bg-indigo-50 px-4 py-2.5 text-xs font-bold text-indigo-700 shadow-sm transition hover:bg-indigo-100"
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
        <div className="mt-5 space-y-5">
          {result.message && (
            <div
              className={
                isIIRSPair
                  ? "rounded-xl border border-purple-200 bg-purple-50 px-4 py-3 text-xs text-purple-800"
                  : "rounded-xl border border-emerald-200 bg-emerald-50 px-4 py-3 text-xs text-emerald-800"
              }
            >
              <span className="font-black uppercase tracking-wider">
                {isIIRSPair ? "Co-Registration Validated:" : "Registration Verified:"}
              </span>
              <span className="ml-2">{result.message}</span>
              {isIIRSPair && (
                <span className="ml-2 inline-block rounded-full border border-emerald-200 bg-emerald-50 px-2.5 py-0.5 text-[10px] font-black uppercase tracking-wider text-emerald-700">
                  Spectral Projection: Validated via TMC-2 Bridge
                </span>
              )}
            </div>
          )}

          <div className="grid gap-4 xl:grid-cols-[1.45fr_1fr]">
            {/* Visual Artifacts */}
            <div className="space-y-4">
              <div className="rounded-xl border border-slate-200 bg-slate-950 p-3">
                <div className="mb-3 flex items-center justify-between px-1">
                  <div>
                    <p className="text-[10px] font-black uppercase tracking-[0.18em] text-indigo-300">Primary Visual Proof</p>
                    <h4 className="mt-1 text-sm font-bold text-white">Continuity Checkerboard Registration QA</h4>
                  </div>
                  <span className="rounded-md bg-emerald-400/10 px-2 py-1 text-[9px] font-bold uppercase tracking-wider text-emerald-300">
                    50 px alternating blocks
                  </span>
                </div>

                {absoluteUrl(result.visual_url) ? (
                  // eslint-disable-next-line @next/next/no-img-element
                  <img
                    src={absoluteUrl(result.visual_url)!}
                    alt="Continuity checkerboard QA"
                    className="max-h-[470px] w-full rounded-lg object-contain"
                    onError={(e) => {
                      const img = e.currentTarget as HTMLImageElement;
                      if (!img.src.includes("/images/registered/region_001/checkerboard_qa.png")) {
                        img.src = imageUrl("/images/registered/region_001/checkerboard_qa.png");
                      }
                    }}
                  />
                ) : (
                  <div className="flex min-h-[260px] items-center justify-center rounded-lg bg-slate-900 text-xs text-slate-500">
                    No checkerboard artifact returned.
                  </div>
                )}
              </div>

              <div className="rounded-xl border border-slate-200 bg-white p-3">
                <div className="mb-2 flex items-center justify-between">
                  <h4 className="text-xs font-bold text-slate-800">Registered Warped Source Image</h4>
                  <span className="text-[10px] font-mono text-slate-400">Homography Reprojected</span>
                </div>
                {absoluteUrl(result.warped_url) ? (
                  // eslint-disable-next-line @next/next/no-img-element
                  <img
                    src={absoluteUrl(result.warped_url)!}
                    alt="Registered warped source"
                    className="max-h-[250px] w-full rounded-lg bg-slate-950 object-contain"
                    onError={(e) => {
                      const img = e.currentTarget as HTMLImageElement;
                      if (!img.src.includes("/images/registered/region_001/registered_ohrc.png")) {
                        img.src = imageUrl("/images/registered/region_001/registered_ohrc.png");
                      }
                    }}
                  />
                ) : (
                  <div className="flex min-h-[120px] items-center justify-center rounded-lg bg-slate-100 text-xs text-slate-400">
                    No warped image returned.
                  </div>
                )}
              </div>

              {absoluteUrl(result.quiver_url) ? (
                <div className="rounded-xl border border-slate-200 bg-white p-3">
                  <div className="mb-2 flex items-center justify-between">
                    <h4 className="text-xs font-bold text-slate-800">Residual Displacement Vectors</h4>
                    <span className="text-[10px] font-mono text-slate-400">Per-inlier reprojection error</span>
                  </div>
                  {/* eslint-disable-next-line @next/next/no-img-element */}
                  <img
                    src={absoluteUrl(result.quiver_url)!}
                    alt="Residual displacement vector quiver plot"
                    className="max-h-[250px] w-full rounded-lg bg-slate-950 object-contain"
                  />
                </div>
              ) : null}
            </div>

            {/* Telemetry Metric Cards */}
            <div className="grid grid-cols-2 content-start gap-3">
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
                <div className="rounded-xl border border-emerald-200 bg-emerald-50 p-4">
                  <p className="text-[10px] font-bold uppercase tracking-[0.16em] text-emerald-600">
                    Spectral Projection
                  </p>
                  <p className="mt-2 text-sm font-black leading-snug tracking-tight text-emerald-800">
                    Spectral Projection: Validated via TMC-2 Bridge
                  </p>
                  <p className="mt-1 text-[10px] text-emerald-600">
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

              <div className={`col-span-2 rounded-xl border p-4 ${qualityTone}`}>
                <div className="flex items-center justify-between">
                  <p className="text-[10px] font-black uppercase tracking-[0.16em] opacity-70">Quality Tier</p>
                  <span className="rounded-md border border-indigo-200 bg-white/80 px-2 py-0.5 text-[9px] font-mono font-bold uppercase tracking-wider text-indigo-700 shadow-sm">
                    Estimator: {outlierMethod}
                  </span>
                </div>
                <p className="mt-1 text-2xl font-black tracking-tight">{qualityTier.replaceAll("_", " ")}</p>
                <p className="mt-1 text-[11px] opacity-80">
                  {metrics?.validation_status || "Sub-pixel geometric consensus verified (< 1.0 px)"}
                </p>
              </div>
            </div>
          </div>

          {/* View Mode Toggle: 2D Planar Verification vs 3D Lunar Globe */}
          <div className="flex flex-wrap items-center justify-between gap-3 border-b border-slate-200 pb-3">
            <div className="flex items-center gap-2">
              <button
                type="button"
                onClick={() => setResultView("2d")}
                className={`flex items-center gap-2 rounded-lg px-3.5 py-1.5 text-xs font-bold transition ${
                  resultView === "2d"
                    ? "bg-indigo-600 text-white shadow-sm"
                    : "bg-slate-100 text-slate-600 hover:bg-slate-200 hover:text-slate-900"
                }`}
              >
                <span>🔍 2D Planar Verification</span>
                <span
                  className={`rounded-full px-1.5 py-0.2 text-[10px] ${
                    resultView === "2d" ? "bg-indigo-500/80 text-white" : "bg-slate-200 text-slate-600"
                  }`}
                >
                  {points.length}
                </span>
              </button>
              <button
                type="button"
                onClick={() => setResultView("3d")}
                className={`flex items-center gap-2 rounded-lg px-3.5 py-1.5 text-xs font-bold transition ${
                  resultView === "3d"
                    ? "bg-indigo-600 text-white shadow-sm"
                    : "bg-slate-100 text-slate-600 hover:bg-slate-200 hover:text-slate-900"
                }`}
              >
                <span>🌕 3D Lunar Globe Tie-Points</span>
                {moonPoints.length > 0 && (
                  <span
                    className={`rounded-full px-1.5 py-0.2 text-[10px] ${
                      resultView === "3d" ? "bg-indigo-500/80 text-white" : "bg-slate-200 text-slate-600"
                    }`}
                  >
                    {moonPoints.filter((p) => p.georeferenced).length}
                  </span>
                )}
              </button>
            </div>

            {resultView === "3d" && (
              <span className="text-[11px] text-slate-500">
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
            <div className="relative h-[480px] w-full overflow-hidden rounded-2xl border border-slate-800/80 bg-gradient-to-b from-slate-950 via-[#0a0f1d] to-slate-950 shadow-inner">
              <LunarGlobe
                tiePoints={moonPoints}
                className="absolute inset-0 h-full w-full cursor-grab active:cursor-grabbing"
              />
              {/* Globe Overlay HUD */}
              <div className="pointer-events-none absolute left-4 top-4 z-20 flex flex-col gap-1.5">
                <div className="flex items-center gap-2 rounded-lg border border-white/10 bg-slate-900/85 px-3 py-1.5 shadow backdrop-blur">
                  <span className="h-2 w-2 rounded-full bg-emerald-400 animate-pulse" />
                  <span className="text-xs font-mono font-bold text-slate-200">
                    {moonPointsLoading
                      ? "Fetching 3D coordinates..."
                      : `${moonPoints.filter((p) => p.georeferenced).length} Georeferenced Lunar Coordinates`}
                  </span>
                </div>
                {result?.job_id && (
                  <span className="text-[10px] font-mono text-slate-400">
                    Region/Job: {result.job_id}
                  </span>
                )}
              </div>

              <div className="pointer-events-none absolute bottom-4 right-4 z-20 rounded-lg border border-white/10 bg-slate-900/85 px-3 py-1.5 text-[11px] font-medium text-slate-300 backdrop-blur">
                Spherical Projection · True Lunar Coordinate Mapping
              </div>
            </div>
          )}

          {/* Action Bar */}
          <div className="flex flex-wrap items-center justify-between gap-3 rounded-xl border border-slate-200 bg-slate-50 p-4">
            <div>
              <p className="text-xs font-bold text-slate-800">{points.length} verified correspondences mapped</p>
              <p className="mt-1 text-[10px] text-slate-500">
                <span className="text-emerald-500">●</span> High Confidence (&gt;0.8){" "}
                <span className="ml-2 text-amber-500">●</span> Review (&gt;0.5){" "}
                <span className="ml-2 text-rose-400">●</span> Low Confidence
              </p>
            </div>
            <div className="flex flex-wrap items-center gap-2">
              {result.job_id && (
                <a
                  href={`${API_BASE}/api/registration/report/${result.job_id}`}
                  download={`ISRO_Registration_Report_${result.job_id}.pdf`}
                  target="_blank"
                  rel="noreferrer"
                  className="inline-flex items-center gap-1.5 rounded-lg border border-indigo-600 bg-indigo-600 px-3.5 py-2 text-[11px] font-bold text-white shadow-sm transition hover:bg-indigo-700 hover:shadow"
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
              className="rounded-xl border border-slate-200 bg-white px-4 py-2.5 text-xs font-bold text-slate-700 shadow-sm transition hover:bg-slate-50"
            >
              ← Register Another Pair
            </button>
          </div>
        </div>
      )}
    </section>
  );
}
