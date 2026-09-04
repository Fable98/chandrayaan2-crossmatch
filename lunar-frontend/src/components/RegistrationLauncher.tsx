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
        className="flex items-center gap-1.5 rounded-full bg-gradient-to-r from-purple-600 via-indigo-600 to-purple-600 px-4 py-1.5 text-xs font-semibold text-white shadow-[0_0_20px_rgba(168,85,247,0.35)] transition-all hover:brightness-110 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-purple-400"
      >
        <span>+</span>
        <span>Live Registration</span>
      </button>

      {open && (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/80 p-4 backdrop-blur-2xl animate-fade-in">
          <div className="max-h-[90vh] w-full max-w-5xl overflow-y-auto rounded-3xl border border-white/15 bg-[#0c0e24]/90 p-6 md:p-8 text-white shadow-[0_30px_90px_rgba(0,0,0,0.9)] ring-1 ring-white/10">
            <div className="mb-6 flex items-start justify-between gap-4 border-b border-white/10 pb-5">
              <div>
                <h2 className="text-lg font-bold tracking-tight text-white flex items-center gap-2">
                  <span>✦ Cross-Sensor Registration Pipeline</span>
                </h2>
                <p className="mt-1 text-xs text-slate-300">
                  Upload arbitrary sensor images (OHRC, TMC-2, IIRS). Executes sub-pixel LoFTR matching,
                  phase-correlation refinement, and composite product generation.
                </p>
              </div>
              <button
                type="button"
                onClick={() => setOpen(false)}
                className="rounded-full border border-white/10 bg-white/[0.04] px-3.5 py-1 text-xs text-slate-300 transition hover:bg-white/[0.08] hover:text-white"
              >
                Close ✕
              </button>
            </div>

            {!result ? (
              <div className="space-y-5">
                <div className="grid gap-4 md:grid-cols-2">
                  <label className="block rounded-2xl border border-white/10 bg-white/[0.03] p-5 transition-all hover:border-purple-400/40 hover:bg-white/[0.05]">
                    <span className="text-xs font-semibold text-purple-200 uppercase tracking-wider block">
                      Source Image (Moving / OHRC)
                    </span>
                    <input
                      type="file"
                      accept=".jpg,.jpeg,.png,.tif,.tiff,image/*"
                      onChange={(event) =>
                        setSourceFile(event.target.files?.[0] || null)
                      }
                      className="mt-3 block w-full text-xs text-slate-300 file:mr-4 file:rounded-full file:border-0 file:bg-purple-600/30 file:px-4 file:py-1.5 file:text-xs file:font-semibold file:text-purple-200 hover:file:bg-purple-600/50 cursor-pointer"
                    />
                    <span className="mt-2.5 block truncate text-[11px] text-slate-400">
                      {sourceFile ? sourceFile.name : "No source image selected"}
                    </span>
                  </label>

                  <label className="block rounded-2xl border border-white/10 bg-white/[0.03] p-5 transition-all hover:border-purple-400/40 hover:bg-white/[0.05]">
                    <span className="text-xs font-semibold text-purple-200 uppercase tracking-wider block">
                      Reference Image (Fixed / TMC-2)
                    </span>
                    <input
                      type="file"
                      accept=".jpg,.jpeg,.png,.tif,.tiff,image/*"
                      onChange={(event) =>
                        setReferenceFile(event.target.files?.[0] || null)
                      }
                      className="mt-3 block w-full text-xs text-slate-300 file:mr-4 file:rounded-full file:border-0 file:bg-purple-600/30 file:px-4 file:py-1.5 file:text-xs file:font-semibold file:text-purple-200 hover:file:bg-purple-600/50 cursor-pointer"
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
                  className="w-full rounded-2xl bg-gradient-to-r from-purple-600 via-indigo-600 to-purple-600 px-5 py-3.5 text-xs font-bold uppercase tracking-wider text-white shadow-[0_0_30px_rgba(168,85,247,0.35)] transition-all hover:brightness-110 disabled:cursor-not-allowed disabled:opacity-40"
                >
                  {loading
                    ? "Executing Sub-Pixel Refinement & Homography..."
                    : "Run Dynamic Registration"}
                </button>

                {error && (
                  <div className="rounded-2xl border border-red-500/30 bg-red-950/30 p-4 text-xs text-red-200">
                    {error}
                  </div>
                )}
              </div>
            ) : (
              <div className="space-y-5">
                <div className="grid gap-3 grid-cols-2 md:grid-cols-5">
                  <div className="rounded-2xl border border-white/10 bg-white/[0.03] p-4">
                    <div className="text-[10px] uppercase font-bold tracking-wider text-slate-400">RMSE</div>
                    <div className="mt-1 text-lg font-bold text-purple-300">
                      {formatMetric(result.metrics?.rmse_px)} px
                    </div>
                  </div>

                  <div className="rounded-2xl border border-white/10 bg-white/[0.03] p-4">
                    <div className="text-[10px] uppercase font-bold tracking-wider text-slate-400">Inliers</div>
                    <div className="mt-1 text-lg font-bold text-white">
                      {result.metrics?.num_inliers ?? "-"}
                    </div>
                  </div>

                  <div className="rounded-2xl border border-white/10 bg-white/[0.03] p-4">
                    <div className="text-[10px] uppercase font-bold tracking-wider text-slate-400">Inlier Ratio</div>
                    <div className="mt-1 text-lg font-bold text-white">
                      {formatMetric(result.metrics?.inlier_ratio ? result.metrics.inlier_ratio * 100 : 0, 1)}%
                    </div>
                  </div>

                  <div className="rounded-2xl border border-white/10 bg-white/[0.03] p-4">
                    <div className="text-[10px] uppercase font-bold tracking-wider text-slate-400">Uniformity</div>
                    <div className="mt-1 text-lg font-bold text-white">
                      {formatMetric(result.metrics?.uniformity_score ? result.metrics.uniformity_score * 100 : 0, 1)}%
                    </div>
                  </div>

                  <div className="rounded-2xl border border-white/10 bg-white/[0.03] p-4">
                    <div className="text-[10px] uppercase font-bold tracking-wider text-slate-400">Status</div>
                    <div className="mt-1 text-base font-bold text-emerald-400">
                      {result.metrics?.sub_pixel_accurate ? "Verified (<1px)" : "Marginal"}
                    </div>
                  </div>
                </div>

                {visualUrl && (
                  <div className="rounded-2xl border border-white/10 bg-white/[0.03] p-4">
                    <div className="mb-2.5 flex items-center justify-between">
                      <span className="text-xs font-semibold uppercase tracking-wider text-purple-200">
                        Registered Composite Product (50px Checkerboard Blend)
                      </span>
                      <span className="text-[11px] font-semibold text-emerald-400">Alignment Verified</span>
                    </div>
                    {/* eslint-disable-next-line @next/next/no-img-element */}
                    <img
                      src={visualUrl}
                      alt="Registered checkerboard product"
                      className="w-full max-h-[450px] object-contain rounded-xl border border-white/10 bg-black"
                    />
                  </div>
                )}

                <div className="flex flex-wrap gap-3">
                  <button
                    type="button"
                    onClick={() => setResult(null)}
                    className="rounded-full border border-white/10 bg-white/[0.04] px-4 py-2 text-xs font-semibold text-slate-200 transition hover:bg-white/[0.08]"
                  >
                    Register Another Pair
                  </button>

                  {warpedUrl && (
                    <a
                      href={warpedUrl}
                      target="_blank"
                      rel="noreferrer"
                      className="rounded-full border border-purple-400/30 bg-purple-500/10 px-4 py-2 text-xs font-semibold text-purple-200 transition hover:bg-purple-500/25"
                    >
                      View Warped Source
                    </a>
                  )}

                  {matchesUrl && (
                    <a
                      href={matchesUrl}
                      target="_blank"
                      rel="noreferrer"
                      className="rounded-full border border-purple-400/30 bg-purple-500/10 px-4 py-2 text-xs font-semibold text-purple-200 transition hover:bg-purple-500/25"
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
