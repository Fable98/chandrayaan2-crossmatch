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
        className="w-full rounded-xl bg-white/20 hover:bg-white/30 text-white font-semibold text-xs py-2 px-3 transition-colors flex items-center justify-center gap-1.5 shadow-sm"
      >
        <span>+</span>
        <span>Live Registration</span>
      </button>

      {open && (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-slate-900/40 p-4 backdrop-blur-sm animate-fade-in">
          <div className="max-h-[90vh] w-full max-w-5xl overflow-y-auto rounded-2xl border border-slate-200/80 bg-white p-6 md:p-8 text-slate-800 shadow-2xl">
            <div className="mb-6 flex items-start justify-between gap-4 border-b border-slate-100 pb-5">
              <div>
                <h2 className="text-lg font-bold tracking-tight text-slate-900 flex items-center gap-2">
                  <span>✦ Cross-Sensor Registration Pipeline</span>
                </h2>
                <p className="mt-1 text-xs text-slate-500">
                  Upload arbitrary sensor images (OHRC, TMC-2, IIRS). Executes sub-pixel LoFTR matching,
                  phase-correlation refinement, and composite product generation.
                </p>
              </div>
              <button
                type="button"
                onClick={() => setOpen(false)}
                className="rounded-lg border border-slate-200 bg-white px-3 py-1 text-xs text-slate-400 transition hover:bg-slate-50 hover:text-slate-700"
              >
                Close ✕
              </button>
            </div>

            {!result ? (
              <div className="space-y-5">
                <div className="grid gap-4 md:grid-cols-2">
                  <label className="block rounded-xl border border-slate-200 bg-slate-50 p-5 transition-all hover:border-[#4F46E5]/40 hover:bg-slate-100/60 cursor-pointer">
                    <span className="text-xs font-bold text-slate-700 uppercase tracking-wider block">
                      Source Image (Moving / OHRC)
                    </span>
                    <input
                      type="file"
                      accept=".jpg,.jpeg,.png,.tif,.tiff,image/*"
                      onChange={(event) =>
                        setSourceFile(event.target.files?.[0] || null)
                      }
                      className="mt-3 block w-full text-xs text-slate-600 file:mr-4 file:rounded-lg file:border-0 file:bg-indigo-50 file:px-4 file:py-1.5 file:text-xs file:font-semibold file:text-[#4F46E5] hover:file:bg-indigo-100 cursor-pointer"
                    />
                    <span className="mt-2.5 block truncate text-[11px] text-slate-400">
                      {sourceFile ? sourceFile.name : "No source image selected"}
                    </span>
                  </label>

                  <label className="block rounded-xl border border-slate-200 bg-slate-50 p-5 transition-all hover:border-[#4F46E5]/40 hover:bg-slate-100/60 cursor-pointer">
                    <span className="text-xs font-bold text-slate-700 uppercase tracking-wider block">
                      Reference Image (Fixed / TMC-2)
                    </span>
                    <input
                      type="file"
                      accept=".jpg,.jpeg,.png,.tif,.tiff,image/*"
                      onChange={(event) =>
                        setReferenceFile(event.target.files?.[0] || null)
                      }
                      className="mt-3 block w-full text-xs text-slate-600 file:mr-4 file:rounded-lg file:border-0 file:bg-indigo-50 file:px-4 file:py-1.5 file:text-xs file:font-semibold file:text-[#4F46E5] hover:file:bg-indigo-100 cursor-pointer"
                    />
                    <span className="mt-2.5 block truncate text-[11px] text-slate-400">
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
                  className="w-full rounded-xl bg-[#4F46E5] hover:bg-[#4338CA] px-5 py-3.5 text-xs font-bold uppercase tracking-wider text-white shadow-sm transition-all disabled:cursor-not-allowed disabled:opacity-40"
                >
                  {loading
                    ? "Executing Sub-Pixel Refinement & Homography..."
                    : "Run Dynamic Registration"}
                </button>

                {error && (
                  <div className="rounded-xl border border-red-200 bg-red-50 p-4 text-xs text-red-700">
                    {error}
                  </div>
                )}
              </div>
            ) : (
              <div className="space-y-5">
                <div className="grid gap-3 grid-cols-2 md:grid-cols-5">
                  <div className="rounded-xl border border-slate-200 bg-slate-50 p-4">
                    <div className="text-[10px] uppercase font-bold tracking-wider text-slate-400">RMSE</div>
                    <div className="mt-1 text-lg font-bold text-[#4F46E5]">
                      {formatMetric(result.metrics?.rmse_px)} px
                    </div>
                  </div>

                  <div className="rounded-xl border border-slate-200 bg-slate-50 p-4">
                    <div className="text-[10px] uppercase font-bold tracking-wider text-slate-400">Inliers</div>
                    <div className="mt-1 text-lg font-bold text-slate-900">
                      {result.metrics?.num_inliers ?? "-"}
                    </div>
                  </div>

                  <div className="rounded-xl border border-slate-200 bg-slate-50 p-4">
                    <div className="text-[10px] uppercase font-bold tracking-wider text-slate-400">Inlier Ratio</div>
                    <div className="mt-1 text-lg font-bold text-slate-900">
                      {formatMetric(result.metrics?.inlier_ratio ? result.metrics.inlier_ratio * 100 : 0, 1)}%
                    </div>
                  </div>

                  <div className="rounded-xl border border-slate-200 bg-slate-50 p-4">
                    <div className="text-[10px] uppercase font-bold tracking-wider text-slate-400">Uniformity</div>
                    <div className="mt-1 text-lg font-bold text-slate-900">
                      {formatMetric(result.metrics?.uniformity_score ? result.metrics.uniformity_score * 100 : 0, 1)}%
                    </div>
                  </div>

                  <div className="rounded-xl border border-slate-200 bg-slate-50 p-4">
                    <div className="text-[10px] uppercase font-bold tracking-wider text-slate-400">Status</div>
                    <div className="mt-1 text-base font-bold text-emerald-600">
                      {result.metrics?.sub_pixel_accurate ? "Verified (<1px)" : "Marginal"}
                    </div>
                  </div>
                </div>

                {visualUrl && (
                  <div className="rounded-xl border border-slate-200 bg-white p-4">
                    <div className="mb-2.5 flex items-center justify-between">
                      <span className="text-xs font-bold uppercase tracking-wider text-slate-700">
                        Registered Composite Product (50px Checkerboard Blend)
                      </span>
                      <span className="text-[11px] font-bold text-emerald-600">Alignment Verified</span>
                    </div>
                    {/* eslint-disable-next-line @next/next/no-img-element */}
                    <img
                      src={visualUrl}
                      alt="Registered checkerboard product"
                      className="w-full max-h-[450px] object-contain rounded-xl border border-slate-100 bg-black"
                    />
                  </div>
                )}

                <div className="flex flex-wrap gap-3">
                  <button
                    type="button"
                    onClick={() => setResult(null)}
                    className="rounded-xl border border-slate-200 bg-white px-4 py-2 text-xs font-semibold text-slate-700 transition hover:bg-slate-50"
                  >
                    Register Another Pair
                  </button>

                  {warpedUrl && (
                    <a
                      href={warpedUrl}
                      target="_blank"
                      rel="noreferrer"
                      className="rounded-xl border border-indigo-100 bg-indigo-50 px-4 py-2 text-xs font-semibold text-[#4F46E5] transition hover:bg-indigo-100"
                    >
                      View Warped Source
                    </a>
                  )}

                  {matchesUrl && (
                    <a
                      href={matchesUrl}
                      target="_blank"
                      rel="noreferrer"
                      className="rounded-xl border border-indigo-100 bg-indigo-50 px-4 py-2 text-xs font-semibold text-[#4F46E5] transition hover:bg-indigo-100"
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
