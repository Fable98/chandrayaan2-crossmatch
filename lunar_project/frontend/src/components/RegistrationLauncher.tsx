"use client";

import { useCallback, useEffect, useState } from "react";

type RegistrationMetrics = {
  num_inliers?: number;
  rmse_px?: number;
  inlier_ratio?: number;
  sub_pixel_accurate?: boolean;
  fraction_below_1px?: number;
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
  ((import.meta as any).env?.VITE_API_URL as string | undefined) ||
  "http://localhost:8000";

function toAbsoluteUrl(path?: string | null): string | null {
  if (!path) return null;
  if (path.startsWith("http://") || path.startsWith("https://")) return path;
  return `${API_BASE}${path}`;
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

      const response = await fetch(`${API_BASE}/register`, {
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
        className="rounded-lg border border-cyan-400/30 bg-cyan-500/10 px-4 py-2 text-sm font-medium text-cyan-100 shadow-lg transition hover:bg-cyan-500/20"
      >
        Live Cross-Sensor Registration
      </button>

      {open && (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/70 p-4 backdrop-blur-sm">
          <div className="max-h-[90vh] w-full max-w-5xl overflow-y-auto rounded-2xl border border-cyan-400/20 bg-slate-950/95 p-6 shadow-2xl">
            <div className="mb-4 flex items-start justify-between gap-4">
              <div>
                <h2 className="text-lg font-semibold text-cyan-100">
                  Live Cross-Sensor Registration
                </h2>
                <p className="mt-1 text-sm text-slate-400">
                  Upload a source and reference image. The backend will compute
                  sub-pixel correspondences, uniform match distribution, and a
                  registered checkerboard product.
                </p>
              </div>
              <button
                type="button"
                onClick={() => setOpen(false)}
                className="rounded-md border border-slate-600 px-3 py-1 text-sm text-slate-300 transition hover:bg-slate-800"
              >
                Close
              </button>
            </div>

            {!result ? (
              <div className="space-y-4">
                <div className="grid gap-4 md:grid-cols-2">
                  <label className="block rounded-xl border border-slate-700 bg-slate-900/70 p-4">
                    <span className="text-sm font-medium text-slate-200">
                      Source Image / Moving
                    </span>
                    <input
                      type="file"
                      accept=".jpg,.jpeg,.png,.tif,.tiff,image/*"
                      onChange={(event) =>
                        setSourceFile(event.target.files?.[0] || null)
                      }
                      className="mt-3 block w-full text-sm text-slate-300 file:mr-4 file:rounded-md file:border-0 file:bg-cyan-500/20 file:px-3 file:py-2 file:text-cyan-100 hover:file:bg-cyan-500/30"
                    />
                    <span className="mt-2 block truncate text-xs text-slate-500">
                      {sourceFile ? sourceFile.name : "No source image selected"}
                    </span>
                  </label>

                  <label className="block rounded-xl border border-slate-700 bg-slate-900/70 p-4">
                    <span className="text-sm font-medium text-slate-200">
                      Reference Image / Fixed
                    </span>
                    <input
                      type="file"
                      accept=".jpg,.jpeg,.png,.tif,.tiff,image/*"
                      onChange={(event) =>
                        setReferenceFile(event.target.files?.[0] || null)
                      }
                      className="mt-3 block w-full text-sm text-slate-300 file:mr-4 file:rounded-md file:border-0 file:bg-cyan-500/20 file:px-3 file:py-2 file:text-cyan-100 hover:file:bg-cyan-500/30"
                    />
                    <span className="mt-2 block truncate text-xs text-slate-500">
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
                  className="w-full rounded-lg border border-cyan-400/40 bg-cyan-500/20 px-4 py-3 text-sm font-semibold text-cyan-100 transition hover:bg-cyan-500/30 disabled:cursor-not-allowed disabled:opacity-50"
                >
                  {loading
                    ? "Calculating Sub-Pixel Homography..."
                    : "Run Sub-Pixel Registration"}
                </button>

                {error && (
                  <div className="rounded-lg border border-red-500/30 bg-red-500/10 p-3 text-sm text-red-200">
                    {error}
                  </div>
                )}
              </div>
            ) : (
              <div className="space-y-4">
                <div className="grid gap-3 md:grid-cols-4">
                  <div className="rounded-xl border border-slate-700 bg-slate-900/70 p-3">
                    <div className="text-xs text-slate-400">RMSE (px)</div>
                    <div className="mt-1 text-lg font-semibold text-cyan-100">
                      {formatMetric(result.metrics?.rmse_px)}
                    </div>
                  </div>

                  <div className="rounded-xl border border-slate-700 bg-slate-900/70 p-3">
                    <div className="text-xs text-slate-400">Inlier Count</div>
                    <div className="mt-1 text-lg font-semibold text-cyan-100">
                      {result.metrics?.num_inliers ?? "-"}
                    </div>
                  </div>

                  <div className="rounded-xl border border-slate-700 bg-slate-900/70 p-3">
                    <div className="text-xs text-slate-400">Inlier Ratio</div>
                    <div className="mt-1 text-lg font-semibold text-cyan-100">
                      {formatMetric(result.metrics?.inlier_ratio)}
                    </div>
                  </div>

                  <div className="rounded-xl border border-slate-700 bg-slate-900/70 p-3">
                    <div className="text-xs text-slate-400">Sub-pixel</div>
                    <div className="mt-1 text-lg font-semibold text-cyan-100">
                      {result.metrics?.sub_pixel_accurate ? "Yes" : "No"}
                    </div>
                  </div>
                </div>

                {visualUrl && (
                  <div className="rounded-xl border border-slate-700 bg-slate-900/70 p-3">
                    <div className="mb-2 text-sm font-medium text-slate-200">
                      Registered Checkerboard Product
                    </div>
                    {/* eslint-disable-next-line @next/next/no-img-element */}
                    <img
                      src={visualUrl}
                      alt="Registered checkerboard product"
                      className="w-full rounded-lg border border-slate-800"
                    />
                  </div>
                )}

                <div className="flex flex-wrap gap-3">
                  <button
                    type="button"
                    onClick={() => setResult(null)}
                    className="rounded-lg border border-slate-600 px-4 py-2 text-sm text-slate-200 transition hover:bg-slate-800"
                  >
                    Register Another Pair
                  </button>

                  {warpedUrl && (
                    <a
                      href={warpedUrl}
                      target="_blank"
                      rel="noreferrer"
                      className="rounded-lg border border-cyan-400/30 bg-cyan-500/10 px-4 py-2 text-sm text-cyan-100 transition hover:bg-cyan-500/20"
                    >
                      Open Warped Source
                    </a>
                  )}

                  {matchesUrl && (
                    <a
                      href={matchesUrl}
                      target="_blank"
                      rel="noreferrer"
                      className="rounded-lg border border-cyan-400/30 bg-cyan-500/10 px-4 py-2 text-sm text-cyan-100 transition hover:bg-cyan-500/20"
                    >
                      Open Match JSON
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