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
        className="rounded-md border border-[#4A4A4A] bg-[#282c30] px-3 py-1.5 text-xs font-medium text-[#CBCBCB] transition-colors hover:bg-[#4A4A4A] hover:text-[#FFFFE3] focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-[#6D8196]"
      >
        Live Registration
      </button>

      {open && (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/70 p-4">
          <div className="max-h-[90vh] w-full max-w-5xl overflow-y-auto rounded-lg border border-[#4A4A4A] bg-[#1a1d20] p-6 text-[#FFFFE3] shadow-xl">
            <div className="mb-6 flex items-start justify-between gap-4 border-b border-[#4A4A4A] pb-4">
              <div>
                <h2 className="text-base font-semibold text-[#FFFFE3]">
                  Cross-Sensor Registration Pipeline
                </h2>
                <p className="mt-1 text-xs text-[#CBCBCB]">
                  Upload arbitrary sensor images (OHRC, TMC-2, IIRS). Runs sub-pixel LoFTR matching,
                  spatial uniformity filtering, phase-correlation refinement, and composite generation.
                </p>
              </div>
              <button
                type="button"
                onClick={() => setOpen(false)}
                className="rounded-md border border-[#4A4A4A] bg-[#282c30] px-3 py-1.5 text-xs text-[#CBCBCB] transition-colors hover:bg-[#4A4A4A] hover:text-[#FFFFE3]"
              >
                Close
              </button>
            </div>

            {!result ? (
              <div className="space-y-4">
                <div className="grid gap-4 md:grid-cols-2">
                  <label className="block rounded-lg border border-[#4A4A4A] bg-[#282c30] p-4 transition-colors hover:border-[#565c63]">
                    <span className="text-xs font-medium text-[#CBCBCB]">
                      Source Image (Moving / OHRC)
                    </span>
                    <input
                      type="file"
                      accept=".jpg,.jpeg,.png,.tif,.tiff,image/*"
                      onChange={(event) =>
                        setSourceFile(event.target.files?.[0] || null)
                      }
                      className="mt-3 block w-full text-xs text-[#CBCBCB] file:mr-4 file:rounded-md file:border file:border-[#4A4A4A] file:bg-[#1a1d20] file:px-3 file:py-1.5 file:text-xs file:font-medium file:text-[#FFFFE3] hover:file:bg-[#4A4A4A] cursor-pointer"
                    />
                    <span className="mt-2 block truncate text-[11px] text-[#a2a8b0]">
                      {sourceFile ? sourceFile.name : "No source image selected"}
                    </span>
                  </label>

                  <label className="block rounded-lg border border-[#4A4A4A] bg-[#282c30] p-4 transition-colors hover:border-[#565c63]">
                    <span className="text-xs font-medium text-[#CBCBCB]">
                      Reference Image (Fixed / TMC-2)
                    </span>
                    <input
                      type="file"
                      accept=".jpg,.jpeg,.png,.tif,.tiff,image/*"
                      onChange={(event) =>
                        setReferenceFile(event.target.files?.[0] || null)
                      }
                      className="mt-3 block w-full text-xs text-[#CBCBCB] file:mr-4 file:rounded-md file:border file:border-[#4A4A4A] file:bg-[#1a1d20] file:px-3 file:py-1.5 file:text-xs file:font-medium file:text-[#FFFFE3] hover:file:bg-[#4A4A4A] cursor-pointer"
                    />
                    <span className="mt-2 block truncate text-[11px] text-[#a2a8b0]">
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
                  className="w-full rounded-md bg-[#6D8196] px-4 py-2.5 text-xs font-semibold text-[#FFFFE3] transition-colors hover:bg-[#5a6d80] disabled:cursor-not-allowed disabled:opacity-40"
                >
                  {loading
                    ? "Executing Registration & Sub-Pixel Refinement…"
                    : "Run Registration"}
                </button>

                {error && (
                  <div className="rounded-md border border-red-900/50 bg-red-950/30 p-3 text-xs text-red-300">
                    {error}
                  </div>
                )}
              </div>
            ) : (
              <div className="space-y-4">
                <div className="grid gap-3 grid-cols-2 md:grid-cols-5">
                  <div className="rounded-md border border-[#4A4A4A] bg-[#282c30] p-3">
                    <div className="text-[10px] uppercase tracking-wider text-[#a2a8b0]">RMSE</div>
                    <div className="mt-1 text-base font-semibold text-[#FFFFE3]">
                      {formatMetric(result.metrics?.rmse_px)} px
                    </div>
                  </div>

                  <div className="rounded-md border border-[#4A4A4A] bg-[#282c30] p-3">
                    <div className="text-[10px] uppercase tracking-wider text-[#a2a8b0]">Inliers</div>
                    <div className="mt-1 text-base font-semibold text-[#FFFFE3]">
                      {result.metrics?.num_inliers ?? "-"}
                    </div>
                  </div>

                  <div className="rounded-md border border-[#4A4A4A] bg-[#282c30] p-3">
                    <div className="text-[10px] uppercase tracking-wider text-[#a2a8b0]">Inlier Ratio</div>
                    <div className="mt-1 text-base font-semibold text-[#FFFFE3]">
                      {formatMetric(result.metrics?.inlier_ratio ? result.metrics.inlier_ratio * 100 : 0, 1)}%
                    </div>
                  </div>

                  <div className="rounded-md border border-[#4A4A4A] bg-[#282c30] p-3">
                    <div className="text-[10px] uppercase tracking-wider text-[#a2a8b0]">Uniformity</div>
                    <div className="mt-1 text-base font-semibold text-[#FFFFE3]">
                      {formatMetric(result.metrics?.uniformity_score ? result.metrics.uniformity_score * 100 : 0, 1)}%
                    </div>
                  </div>

                  <div className="rounded-md border border-[#4A4A4A] bg-[#282c30] p-3">
                    <div className="text-[10px] uppercase tracking-wider text-[#a2a8b0]">Status</div>
                    <div className={`mt-1 text-base font-semibold ${result.metrics?.sub_pixel_accurate ? "text-[#10b981]" : "text-[#CBCBCB]"}`}>
                      {result.metrics?.sub_pixel_accurate ? "Sub-Pixel" : "Standard"}
                    </div>
                  </div>
                </div>

                {visualUrl && (
                  <div className="rounded-md border border-[#4A4A4A] bg-[#282c30] p-4">
                    <div className="mb-2 flex items-center justify-between">
                      <span className="text-xs font-medium text-[#CBCBCB]">
                        Composite Product (Checkerboard Blend)
                      </span>
                      <span className="text-[11px] text-[#10b981]">Aligned</span>
                    </div>
                    {/* eslint-disable-next-line @next/next/no-img-element */}
                    <img
                      src={visualUrl}
                      alt="Registered checkerboard product"
                      className="w-full max-h-[450px] object-contain rounded border border-[#4A4A4A] bg-black"
                    />
                  </div>
                )}

                <div className="flex flex-wrap gap-3">
                  <button
                    type="button"
                    onClick={() => setResult(null)}
                    className="rounded-md border border-[#4A4A4A] bg-[#282c30] px-3.5 py-1.5 text-xs font-medium text-[#CBCBCB] transition-colors hover:bg-[#4A4A4A] hover:text-[#FFFFE3]"
                  >
                    Register Another Pair
                  </button>

                  {warpedUrl && (
                    <a
                      href={warpedUrl}
                      target="_blank"
                      rel="noreferrer"
                      className="rounded-md border border-[#4A4A4A] bg-[#282c30] px-3.5 py-1.5 text-xs font-medium text-[#6D8196] transition-colors hover:bg-[#4A4A4A]"
                    >
                      View Warped Source
                    </a>
                  )}

                  {matchesUrl && (
                    <a
                      href={matchesUrl}
                      target="_blank"
                      rel="noreferrer"
                      className="rounded-md border border-[#4A4A4A] bg-[#282c30] px-3.5 py-1.5 text-xs font-medium text-[#6D8196] transition-colors hover:bg-[#4A4A4A]"
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
