"use client";

import { useEffect, useMemo, useState } from "react";
import { API_BASE } from "@/lib/api";

type Sensor = "OHRC" | "TMC" | "IIRS";
type RegistrationMetrics = {
  fit_rmse_px?: number | null;
  rmse_px?: number | null;
  validation_rmse_px?: number | null;
  validation_status?: string | null;
  num_inliers?: number | null;
  inlier_count?: number | null;
  inlier_ratio?: number | null;
  combined_coverage_score?: number | null;
  source_coverage_ratio?: number | null;
  uniformity_score?: number | null;
  spatial_uniformity?: number | null;
  quality_tier?: string | null;
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

const sensors: Sensor[] = ["OHRC", "TMC", "IIRS"];

function absoluteUrl(path?: string | null) {
  if (!path) return null;
  return path.startsWith("http") ? path : `${API_BASE}${path}`;
}

function metric(metrics: RegistrationMetrics | null | undefined, ...keys: string[]) {
  for (const key of keys) {
    const value = metrics?.[key as keyof RegistrationMetrics];
    if (typeof value === "number" && Number.isFinite(value)) return value;
  }
  return null;
}

function format(value: number | null, digits = 3) {
  return value === null ? "-" : value.toFixed(digits);
}

function coordinate(point: MatchPoint, side: "source" | "reference") {
  const x = side === "source" ? point.source_x ?? point.image1_x : point.target_x ?? point.image2_x;
  const y = side === "source" ? point.source_y ?? point.image1_y : point.target_y ?? point.image2_y;
  return typeof x === "number" && typeof y === "number" ? [x, y] as const : null;
}

function pointColor(confidence = 0) {
  return confidence >= 0.8 ? "#34d399" : confidence >= 0.5 ? "#fbbf24" : "#fb7185";
}

function PointOverlay({ points, side }: { points: MatchPoint[]; side: "source" | "reference" }) {
  return <div className="pointer-events-none absolute inset-0">{points.map((point, index) => {
    const value = coordinate(point, side);
    if (!value) return null;
    return <span key={`${side}-${index}`} className="absolute h-2.5 w-2.5 -translate-x-1/2 -translate-y-1/2 rounded-full border border-white shadow-[0_0_0_2px_rgba(15,23,42,.55)]" style={{ left: `${value[0] / 512 * 100}%`, top: `${value[1] / 512 * 100}%`, backgroundColor: pointColor(point.confidence) }} />;
  })}</div>;
}

function FilePicker({ label, file, optional, onChange }: { label: string; file: File | null; optional?: boolean; onChange: (file: File | null) => void }) {
  return <label className="block cursor-pointer rounded-xl border border-dashed border-slate-300 bg-slate-50 p-4 transition hover:border-indigo-300 hover:bg-indigo-50/40">
    <div className="flex items-center justify-between gap-3"><span className="text-[10px] font-black uppercase tracking-wider text-slate-600">{label}</span>{optional && <span className="text-[9px] font-bold text-slate-400">OPTIONAL</span>}</div>
    <input type="file" accept=".jpg,.jpeg,.png,.tif,.tiff,image/*" onChange={(event) => onChange(event.target.files?.[0] || null)} className="mt-3 block w-full text-[11px] text-slate-500 file:mr-3 file:rounded-lg file:border-0 file:bg-indigo-100 file:px-3 file:py-1.5 file:text-[10px] file:font-bold file:text-indigo-700" />
    <p className="mt-2 truncate text-[10px] text-slate-400">{file ? file.name : optional ? "No DEM supplied" : "Choose an image"}</p>
  </label>;
}

function MetricCard({ label, value, suffix, emphasis }: { label: string; value: string; suffix?: string; emphasis?: boolean }) {
  return <div className={`rounded-xl border p-4 ${emphasis ? "border-indigo-200 bg-indigo-50" : "border-slate-200 bg-white"}`}><p className="text-[10px] font-bold uppercase tracking-[0.16em] text-slate-400">{label}</p><p className={`mt-2 text-2xl font-black tracking-tight ${emphasis ? "text-[#4F46E5]" : "text-slate-900"}`}>{value}{suffix && <span className="ml-1 text-xs font-bold text-slate-500">{suffix}</span>}</p></div>;
}

function StatusMessage({ tone, message }: { tone: "error" | "success"; message: string }) {
  return <div className={`rounded-xl border px-4 py-3 text-xs ${tone === "error" ? "border-rose-200 bg-rose-50 text-rose-700" : "border-emerald-200 bg-emerald-50 text-emerald-700"}`}><span className="font-black uppercase tracking-wider">{tone === "error" ? "Registration attention" : "Pipeline complete"}</span><span className="ml-2">{message}</span></div>;
}

function DownloadLink({ href, label }: { href?: string | null; label: string }) {
  const url = absoluteUrl(href);
  return url ? <a href={url} download target="_blank" rel="noreferrer" className="rounded-lg border border-indigo-100 bg-white px-3 py-2 text-[10px] font-bold text-indigo-700 transition hover:bg-indigo-50">↓ {label}</a> : null;
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
        <span className="text-[10px] font-mono text-slate-400">{points.length} pts</span>
      </div>
      <div className="relative aspect-square overflow-hidden rounded-lg bg-slate-950 flex items-center justify-center">
        {!failed && currentSrc ? (
          <img
            src={currentSrc}
            alt={title}
            className="h-full w-full object-contain"
            onError={handleError}
          />
        ) : (
          <div className="flex flex-col items-center justify-center p-6 text-center text-slate-400">
            <span className="text-2xl mb-1.5 opacity-80">🛰️</span>
            <span className="text-xs font-semibold text-slate-300">{title}</span>
            <span className="text-[11px] text-slate-500 mt-1 max-w-[220px]">
              Raw GeoTIFF/TIFF ingested · {points.length} correspondence points mapped
            </span>
          </div>
        )}
        <PointOverlay points={points} side={side} />
      </div>
    </div>
  );
}

export default function RegistrationLauncher() {
  const [sourceFile, setSourceFile] = useState<File | null>(null);
  const [referenceFile, setReferenceFile] = useState<File | null>(null);
  const [demFile, setDemFile] = useState<File | null>(null);
  const [sourceSensor, setSourceSensor] = useState<Sensor>("OHRC");
  const [referenceSensor, setReferenceSensor] = useState<Sensor>("TMC");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [result, setResult] = useState<RegistrationResult | null>(null);
  const [points, setPoints] = useState<MatchPoint[]>([]);
  const sourcePreview = useMemo(() => sourceFile ? URL.createObjectURL(sourceFile) : null, [sourceFile]);
  const referencePreview = useMemo(() => referenceFile ? URL.createObjectURL(referenceFile) : null, [referenceFile]);

  useEffect(() => () => {
    if (sourcePreview) URL.revokeObjectURL(sourcePreview);
    if (referencePreview) URL.revokeObjectURL(referencePreview);
  }, [sourcePreview, referencePreview]);

  const reset = () => { setResult(null); setPoints([]); setError(null); };

  const register = async () => {
    if (!sourceFile || !referenceFile) { setError("Choose both a source image and a reference image before registering."); return; }
    setLoading(true); setError(null); setResult(null); setPoints([]);
    try {
      const body = new FormData();
      body.append("source_file", sourceFile); body.append("reference_file", referenceFile);
      body.append("source_sensor", sourceSensor); body.append("reference_sensor", referenceSensor);
      if (demFile) body.append("dem_file", demFile);
      const response = await fetch(`${API_BASE}/register`, { method: "POST", body });
      const data = await response.json().catch(() => null) as RegistrationResult | { detail?: string } | null;
      if (!response.ok) throw new Error(data && "detail" in data ? data.detail : `Registration failed (${response.status}).`);
      const registration = data as RegistrationResult;
      setResult(registration);
      if (registration.matches_url) {
        const matchesResponse = await fetch(absoluteUrl(registration.matches_url) as string);
        if (matchesResponse.ok) setPoints(await matchesResponse.json() as MatchPoint[]);
      }
      if (registration.status !== "success") setError(registration.message || "The backend could not verify this registration.");
    } catch (err) {
      setError(err instanceof Error ? err.message : "Registration failed unexpectedly.");
    } finally { setLoading(false); }
  };

  const metrics = result?.metrics;
  const qualityTier = metrics?.quality_tier || (result?.status === "success" ? "UNSPECIFIED" : result?.status?.toUpperCase() || "READY");
  const qualityTone = qualityTier === "HIGH_CONFIDENCE" ? "border-emerald-300 bg-emerald-50 text-emerald-700" : qualityTier === "ACCEPTED" ? "border-cyan-300 bg-cyan-50 text-cyan-700" : qualityTier === "FAILED" ? "border-rose-300 bg-rose-50 text-rose-700" : "border-amber-300 bg-amber-50 text-amber-700";
  const inliers = metric(metrics, "num_inliers", "inlier_count");
  const coverage = metric(metrics, "combined_coverage_score", "source_coverage_ratio");
  const uniformity = metric(metrics, "uniformity_score", "spatial_uniformity");
  const ratio = metric(metrics, "inlier_ratio");

  return <section className="rounded-2xl border border-slate-200/80 bg-white p-5 shadow-sm md:p-6">
    <div className="flex flex-col justify-between gap-3 border-b border-slate-100 pb-5 md:flex-row md:items-center"><div><div className="flex items-center gap-2"><span className="h-2 w-2 rounded-full bg-[#4F46E5] shadow-[0_0_0_4px_rgba(79,70,229,.12)]" /><p className="text-[10px] font-black uppercase tracking-[0.2em] text-[#4F46E5]">Live pipeline</p></div><h3 className="mt-1 text-xl font-black tracking-tight text-slate-900">Upload &amp; Register</h3><p className="mt-1 max-w-2xl text-xs text-slate-500">Run the full cross-sensor matcher and inspect the registered product, correspondence geometry, and quality gates in one place.</p></div><div className={`rounded-full border px-3 py-1.5 text-[10px] font-black uppercase tracking-wider ${result ? qualityTone : "border-slate-200 bg-slate-50 text-slate-500"}`}>{loading ? "PROCESSING" : result ? qualityTier.replaceAll("_", " ") : "READY"}</div></div>

    {!result && <div className="mt-5 space-y-5"><div className="grid gap-3 md:grid-cols-2"><FilePicker label={`Source image · ${sourceSensor}`} file={sourceFile} onChange={setSourceFile} /><FilePicker label={`Reference image · ${referenceSensor}`} file={referenceFile} onChange={setReferenceFile} /></div><div className="grid gap-3 md:grid-cols-[1fr_1fr_1.3fr]"><label className="block"><span className="mb-2 block text-[10px] font-black uppercase tracking-wider text-slate-500">Source sensor</span><select value={sourceSensor} onChange={(event) => setSourceSensor(event.target.value as Sensor)} className="w-full rounded-xl border border-slate-200 bg-white px-3 py-3 text-xs font-bold text-slate-700 outline-none focus:border-indigo-400 focus:ring-2 focus:ring-indigo-100">{sensors.map((sensor) => <option key={sensor}>{sensor}</option>)}</select></label><label className="block"><span className="mb-2 block text-[10px] font-black uppercase tracking-wider text-slate-500">Reference sensor</span><select value={referenceSensor} onChange={(event) => setReferenceSensor(event.target.value as Sensor)} className="w-full rounded-xl border border-slate-200 bg-white px-3 py-3 text-xs font-bold text-slate-700 outline-none focus:border-indigo-400 focus:ring-2 focus:ring-indigo-100">{sensors.map((sensor) => <option key={sensor}>{sensor}</option>)}</select></label><FilePicker label="Optional DEM elevation" file={demFile} optional onChange={setDemFile} /></div>{error && <StatusMessage tone="error" message={error} />}<button type="button" onClick={register} disabled={loading || !sourceFile || !referenceFile} className="flex w-full items-center justify-center gap-2 rounded-xl bg-[#4F46E5] px-5 py-3.5 text-xs font-black uppercase tracking-[0.15em] text-white shadow-sm transition hover:bg-[#4338CA] disabled:cursor-not-allowed disabled:opacity-40">{loading ? <><span className="h-3 w-3 animate-spin rounded-full border-2 border-white/40 border-t-white" /> Registering pair...</> : "Register pair"}</button></div>}

    {result && <div className="mt-5 space-y-5">{error && <StatusMessage tone="error" message={error} />}{result.message && result.status === "success" && <StatusMessage tone="success" message={result.message} />}<div className="grid gap-4 xl:grid-cols-[1.45fr_1fr]"><div className="space-y-4"><div className="rounded-xl border border-slate-200 bg-slate-950 p-3"><div className="mb-3 flex items-center justify-between px-1"><div><p className="text-[10px] font-black uppercase tracking-[0.18em] text-indigo-300">Primary visual proof</p><h4 className="mt-1 text-sm font-bold text-white">Checkerboard registration QA</h4></div><span className="rounded-md bg-emerald-400/10 px-2 py-1 text-[9px] font-bold uppercase tracking-wider text-emerald-300">50 px blocks</span></div>{absoluteUrl(result.visual_url) ? <img src={absoluteUrl(result.visual_url) as string} alt="Checkerboard registration quality assurance" className="max-h-[470px] w-full rounded-lg object-contain" /> : <div className="flex min-h-[260px] items-center justify-center rounded-lg bg-slate-900 text-xs text-slate-500">No checkerboard artifact returned.</div>}</div><div className="rounded-xl border border-slate-200 bg-white p-3"><div className="mb-2 flex items-center justify-between"><h4 className="text-xs font-bold text-slate-800">Registered warped source</h4><span className="text-[10px] font-mono text-slate-400">Output artifact</span></div>{absoluteUrl(result.warped_url) ? <img src={absoluteUrl(result.warped_url) as string} alt="Registered warped source image" className="max-h-[250px] w-full rounded-lg bg-slate-950 object-contain" /> : <div className="flex min-h-[120px] items-center justify-center rounded-lg bg-slate-100 text-xs text-slate-400">No warped image returned.</div>}</div></div><div className="grid grid-cols-2 content-start gap-3"><MetricCard label="Fit RMSE" value={format(metric(metrics, "fit_rmse_px", "rmse_px"))} suffix="px" emphasis /><MetricCard label="Validation RMSE" value={format(metric(metrics, "validation_rmse_px"))} suffix="px" /><MetricCard label="Verified inliers" value={inliers === null ? "-" : String(inliers)} /><MetricCard label="Inlier ratio" value={format(ratio === null ? null : ratio * 100, 1)} suffix="%" /><MetricCard label="Spatial coverage" value={format(coverage === null ? null : coverage * 100, 1)} suffix="%" /><MetricCard label="Uniformity score" value={format(uniformity === null ? null : uniformity * 100, 1)} suffix="%" /><div className={`col-span-2 rounded-xl border p-4 ${qualityTone}`}><p className="text-[10px] font-black uppercase tracking-[0.16em] opacity-70">Quality tier</p><p className="mt-1 text-2xl font-black tracking-tight">{qualityTier.replaceAll("_", " ")}</p><p className="mt-1 text-[11px] opacity-80">{metrics?.validation_status || "Backend geometric verification result"}</p></div></div></div><div className="grid gap-4 lg:grid-cols-2"><OverlayImage title={`Source · ${sourceSensor}`} src={result.source_url ? absoluteUrl(result.source_url) : sourcePreview} fallbackSrc={result.source_url ? absoluteUrl(result.source_url) : (result.warped_url ? absoluteUrl(result.warped_url) : sourcePreview)} secondaryFallback={result.warped_url ? absoluteUrl(result.warped_url) : null} points={points} side="source" /><OverlayImage title={`Reference · ${referenceSensor}`} src={result.reference_url ? absoluteUrl(result.reference_url) : referencePreview} fallbackSrc={referencePreview} points={points} side="reference" /></div><div className="flex flex-wrap items-center justify-between gap-3 rounded-xl border border-slate-200 bg-slate-50 p-4"><div><p className="text-xs font-bold text-slate-800">{points.length} correspondence points loaded</p><p className="mt-1 text-[10px] text-slate-500"><span className="text-emerald-500">●</span> high confidence <span className="ml-2 text-amber-500">●</span> review <span className="ml-2 text-rose-400">●</span> low confidence</p></div><div className="flex flex-wrap gap-2"><DownloadLink href={result.warped_url} label="Warped PNG" /><DownloadLink href={result.raster_url} label="Registered GeoTIFF" /><DownloadLink href={result.matches_url} label="matches.json" /><DownloadLink href={result.matches_url?.replace("matches.json", "metrics.json")} label="metrics.json" /><DownloadLink href={result.visual_url} label="Checkerboard" /></div></div><button type="button" onClick={reset} className="rounded-xl border border-slate-200 bg-white px-4 py-2.5 text-xs font-bold text-slate-700 transition hover:bg-slate-50">Register another pair</button></div>}
  </section>;
}
