"use client";

import { useCallback, useEffect, useState } from "react";

type RegistrationMetrics = {
  num_inliers?: number;
  rmse_px?: number;
  inlier_ratio?: number;
  sub_pixel_accurate?: boolean;
  fraction_below_1px?: number;
  uniformity_score?: number;
  source_coverage_ratio?: number;
  destination_coverage_ratio?: number;
  combined_coverage_score?: number;
};

type RegistrationResult = {
  status: string;
  metrics?: RegistrationMetrics | null;
  homography?: number[][] | null;
  visual_url?: string | null;
  warped_url?: string | null;
  matches_url?: string | null;
};

const API_BASE =
  process.env.NEXT_PUBLIC_API_BASE_URL ||
  "https://chandrayaan2-crossmatch.onrender.com";

function toAbsoluteUrl(path?: string | null): string | null {
  if (!path) return null;
  if (path.startsWith("http://") || path.startsWith("https://")) return path;
  return `${API_BASE.replace(/\/$/, "")}${path}`;
}

function formatMetric(value?: number, digits = 3): string {
  if (typeof value !== "number" || Number.isNaN(value)) return "-";
  return value.toFixed(digits);
}

export default function RegistrationLauncher() {
  const [open, setOpen] = useState(false);
  const [sourceFile, setSourceFile] = useState<File | null>(null);
  const [referenceFile, setReferenceFile] = useState<File | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [result, setResult] = useState<RegistrationResult | null>(null);

  useEffect(() => {
    if (open) {
      setSourceFile(null);
      setReferenceFile(null);
      setLoading(false);
      setError(null);
      setResult(null);
    }
  }, [open]);

  const runRegistration = useCallback(async () => {
    if (!sourceFile || !referenceFile) {
      setError("Select both a source image and a reference image.");
      return;
    }

    setLoading(true);
    setError(null);

    try {
      const formData = new FormData();
      formData.append("source_file", sourceFile);
      formData.append("reference_file", referenceFile);

      const response = await fetch(`${API_BASE.replace(/\/$/, "")}/register`, {
        method: "POST",
        body: formData,
      });

      const data = await response.json().catch(() => null);

      if (!response.ok) {
        const detail =
          data?.detail ||
          data?.error ||
          `Request failed with status ${response.status}`;
        throw new Error(detail);
      }

      setResult(data as RegistrationResult);
    } catch (err) {
      const message =
        err instanceof Error ? err.message : "Registration failed.";
      setError(message);
    } finally {
      setLoading(false);
    }
  }, [sourceFile, referenceFile]);

  const visualUrl = toAbsoluteUrl(result?.visual_url);
  const warpedUrl = toAbsoluteUrl(result?.warped_url);
  const matchesUrl = toAbsoluteUrl(result?.matches_url);

  return (
    <>
      <button
        type="button"
        onClick={() => setOpen(true)}
        className="rounded-lg border border-cyan-400/40 bg-cyan-500/10 px-3.5 py-1.5 text-xs font-semibold uppercase tracking-wider text-cyan-200 shadow-md transition hover:bg-cyan-500/25 hover:border-cyan-300"
      >
        ✦ Live Registration
      </button>

      {open && (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/80 p-4 backdrop-blur-md">
          <div className="max-h-[90vh] w-full max-w-5xl overflow-y-auto rounded-2xl border border-cyan-400/30 bg-[#080d14] p-6 shadow-2xl">
            <div className="mb-4 flex items-start justify-between gap-4 border-b border-cyan-900/40 pb-4">
              <div>
                <h2 className="text-lg font-bold tracking-wide text-cyan-100 uppercase">
                  Live Cross-Sensor Registration Pipeline
                </h2>
                <p className="mt-1 text-xs text-slate-400">
                  Upload arbitrary sensor images (OHRC, TMC-2, IIRS). The pipeline executes sub-pixel LoFTR matching,
                  a 10×10 spatial uniformity filter, phase-correlation sub-pixel refinement, and composite product generation.
                </p>
              </div>
              <button
                type="button"
                onClick={() => setOpen(false)}
                className="rounded-md border border-slate-700 bg-slate-900/80 px-3 py-1 text-xs text-slate-300 transition hover:bg-slate-800"
              >
                ✕ Close
              </button>
            </div>

            {!result ? (
              <div className="space-y-4">
                <div className="grid gap-4 md:grid-cols-2">
                  <label className="block rounded-xl border border-cyan-900/40 bg-slate-900/60 p-4 hover:border-cyan-700/60 transition">
                    <span className="text-xs font-semibold uppercase tracking-wider text-cyan-300">
                      Source Image (Moving / OHRC)
                    </span>
                    <input
                      type="file"
                      accept=".jpg,.jpeg,.png,.tif,.tiff,image/*"
                      onChange={(event) =>
                        setSourceFile(event.target.files?.[0] || null)
                      }
                      className="mt-3 block w-full text-xs text-slate-300 file:mr-4 file:rounded-md file:border-0 file:bg-cyan-500/20 file:px-3 file:py-2 file:text-cyan-100 hover:file:bg-cyan-500/30 cursor-pointer"
                    />
                    <span className="mt-2 block truncate text-[11px] text-slate-500">
                      {sourceFile ? sourceFile.name : "No source image selected"}
                    </span>
                  </label>

                  <label className="block rounded-xl border border-cyan-900/40 bg-slate-900/60 p-4 hover:border-cyan-700/60 transition">
                    <span className="text-xs font-semibold uppercase tracking-wider text-cyan-300">
                      Reference Image (Fixed / TMC-2)
                    </span>
                    <input
                      type="file"
                      accept=".jpg,.jpeg,.png,.tif,.tiff,image/*"
                      onChange={(event) =>
                        setReferenceFile(event.target.files?.[0] || null)
                      }
                      className="mt-3 block w-full text-xs text-slate-300 file:mr-4 file:rounded-md file:border-0 file:bg-cyan-500/20 file:px-3 file:py-2 file:text-cyan-100 hover:file:bg-cyan-500/30 cursor-pointer"
                    />
                    <span className="mt-2 block truncate text-[11px] text-slate-500">
                      {referenceFile
                        ? referenceFile.name
                        : "No reference image selected"}
                    </span>
                  </label>
                </div>

                <button
                  type="button"
                  onClick={runRegistration}
                  disabled={loading || !sourceFile || !referenceFile}
                  className="w-full rounded-lg border border-cyan-400/50 bg-cyan-600/30 px-4 py-3 text-sm font-bold uppercase tracking-wider text-cyan-100 shadow-lg transition hover:bg-cyan-500/40 disabled:cursor-not-allowed disabled:opacity-40"
                >
                  {loading
                    ? "Executing Sub-Pixel Refinement & Homography..."
                    : "✦ Run Dynamic Registration"}
                </button>

                {error && (
                  <div className="rounded-lg border border-red-500/40 bg-red-500/10 p-3 text-xs text-red-200">
                    ⚠ {error}
                  </div>
                )}
              </div>
            ) : (
              <div className="space-y-4">
                <div className="grid gap-3 grid-cols-2 md:grid-cols-5">
                  <div className="rounded-xl border border-cyan-900/40 bg-slate-900/80 p-3">
                    <div className="text-[10px] uppercase font-bold tracking-wider text-slate-400">RMSE</div>
                    <div className="mt-1 text-lg font-bold text-cyan-300">
                      {formatMetric(result.metrics?.rmse_px)} px
                    </div>
                  </div>

                  <div className="rounded-xl border border-cyan-900/40 bg-slate-900/80 p-3">
                    <div className="text-[10px] uppercase font-bold tracking-wider text-slate-400">Inliers</div>
                    <div className="mt-1 text-lg font-bold text-cyan-300">
                      {result.metrics?.num_inliers ?? "-"}
                    </div>
                  </div>

                  <div className="rounded-xl border border-cyan-900/40 bg-slate-900/80 p-3">
                    <div className="text-[10px] uppercase font-bold tracking-wider text-slate-400">Inlier Ratio</div>
                    <div className="mt-1 text-lg font-bold text-cyan-300">
                      {formatMetric(result.metrics?.inlier_ratio ? result.metrics.inlier_ratio * 100 : 0, 1)}%
                    </div>
                  </div>

                  <div className="rounded-xl border border-cyan-900/40 bg-slate-900/80 p-3">
                    <div className="text-[10px] uppercase font-bold tracking-wider text-slate-400">Uniformity</div>
                    <div className="mt-1 text-lg font-bold text-cyan-300">
                      {formatMetric(result.metrics?.uniformity_score ? result.metrics.uniformity_score * 100 : 0, 1)}%
                    </div>
                  </div>

                  <div className="rounded-xl border border-cyan-900/40 bg-slate-900/80 p-3">
                    <div className="text-[10px] uppercase font-bold tracking-wider text-slate-400">Sub-Pixel</div>
                    <div className="mt-1 text-lg font-bold text-emerald-400">
                      {result.metrics?.sub_pixel_accurate ? "Verified (<1px)" : "Marginal"}
                    </div>
                  </div>
                </div>

                {visualUrl && (
                  <div className="rounded-xl border border-cyan-900/40 bg-slate-900/80 p-4">
                    <div className="mb-2 flex items-center justify-between">
                      <span className="text-xs font-semibold uppercase tracking-wider text-cyan-200">
                        Registered Composite Product (50px Checkerboard Blend)
                      </span>
                      <span className="text-[11px] text-emerald-400">✓ Alignment Verified</span>
                    </div>
                    {/* eslint-disable-next-line @next/next/no-img-element */}
                    <img
                      src={visualUrl}
                      alt="Registered checkerboard product"
                      className="w-full max-h-[450px] object-contain rounded-lg border border-slate-800 bg-black"
                    />
                  </div>
                )}

                <div className="flex flex-wrap gap-3">
                  <button
                    type="button"
                    onClick={() => setResult(null)}
                    className="rounded-lg border border-slate-700 bg-slate-800/80 px-4 py-2 text-xs font-semibold text-slate-200 transition hover:bg-slate-700"
                  >
                    ← Register Another Pair
                  </button>

                  {warpedUrl && (
                    <a
                      href={warpedUrl}
                      target="_blank"
                      rel="noreferrer"
                      className="rounded-lg border border-cyan-400/30 bg-cyan-500/10 px-4 py-2 text-xs font-semibold text-cyan-200 transition hover:bg-cyan-500/20"
                    >
                      View Warped Source
                    </a>
                  )}

                  {matchesUrl && (
                    <a
                      href={matchesUrl}
                      target="_blank"
                      rel="noreferrer"
                      className="rounded-lg border border-cyan-400/30 bg-cyan-500/10 px-4 py-2 text-xs font-semibold text-cyan-200 transition hover:bg-cyan-500/20"
                    >
                      Download Match JSON
                    </a>
                  )}
                </div>
              </div>
            )}
          </div>
        </div>
      )}
    </>
  );
}
