"use client";

interface Props {
  onClose: () => void;
}

export default function TheoryModal({ onClose }: Props) {
  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-slate-900/40 p-4 backdrop-blur-sm animate-fade-in">
      <div className="relative flex max-h-[90vh] w-full max-w-3xl flex-col overflow-hidden rounded-2xl border border-slate-200/80 bg-white text-slate-800 shadow-2xl">
        {/* Header Bar */}
        <div className="flex items-center justify-between border-b border-slate-100 bg-white px-6 py-4">
          <div className="flex items-center gap-2.5">
            <span className="rounded-lg border border-indigo-100 bg-indigo-50 px-2.5 py-0.5 text-[11px] font-bold uppercase tracking-wider text-[#4F46E5]">
              Methodology
            </span>
            <span className="text-sm font-bold text-slate-900">
              Projective Homography &amp; Registration Pipeline
            </span>
          </div>

          <button
            onClick={onClose}
            className="flex h-7 w-7 items-center justify-center rounded-lg border border-slate-200 bg-white text-xs text-slate-400 transition hover:bg-slate-50 hover:text-slate-700"
            title="Close"
          >
            ✕
          </button>
        </div>

        {/* Body Content */}
        <div className="flex-1 overflow-y-auto p-6 md:p-8 space-y-6">
          <div>
            <h2 className="text-lg font-bold text-slate-900">
              Mathematical &amp; Algorithmic Framework
            </h2>
            <p className="mt-1 text-xs text-slate-500">
              Cross-sensor alignment between disparate orbital passes and extreme solar incidence reversals.
            </p>
          </div>

          <div className="border-t border-slate-100 pt-4 text-xs leading-relaxed text-slate-600 space-y-4">
            <p>
              In planetary cross-matching between high-resolution optical cameras (OHRC, 0.25 m/px) and stereo/hyperspectral terrain cameras (TMC-2, 4 m/px), sensor viewing geometries differ radically. Due to non-repeat orbital tracks, the angle of solar incidence often reverses by &gt;160°, rendering traditional pixel intensity metrics invalid.
            </p>

            <div className="rounded-xl border border-slate-200/80 bg-slate-50 p-4 text-xs">
              <span className="text-[#4F46E5] font-mono block mb-1 font-bold">Planar Projective Transform:</span>
              <span className="font-mono text-slate-800">s · [x&apos;, y&apos;, 1]ᵀ = H · [x, y, 1]ᵀ</span>
              <p className="mt-2 text-[11px] text-slate-500">
                Where H is a 3×3 matrix with 8 degrees of freedom calculated via Random Sample Consensus (RANSAC) on dense Transformer-based correspondences (LoFTR).
              </p>
            </div>

            <h4 className="text-xs font-bold text-slate-900 uppercase tracking-wider">
              Key Pipeline Steps:
            </h4>
            <ul className="list-disc list-inside space-y-2 text-slate-600">
              <li>
                <strong className="text-slate-900">Local Feature Transformer (LoFTR):</strong> Establishes semi-dense correspondences without explicit detector bottlenecks, allowing matching inside steep crater shadows.
              </li>
              <li>
                <strong className="text-slate-900">RANSAC Homography:</strong> Filters out erroneous correspondences caused by inverted shadow edges with sub-pixel tolerance (threshold &lt; 3.0 px).
              </li>
              <li>
                <strong className="text-slate-900">Sub-Pixel Refinement:</strong> Minimizes reprojection error to achieve an RMSE &lt; 0.5 px across the shared terrain footprint.
              </li>
            </ul>
          </div>
        </div>

        {/* Footer */}
        <div className="flex items-center justify-between border-t border-slate-100 bg-slate-50/50 px-6 py-3 text-xs">
          <span className="text-slate-500 font-medium">Chandrayaan-2 Registration Pipeline</span>
          <button
            onClick={onClose}
            className="rounded-xl bg-[#4F46E5] hover:bg-[#4338CA] px-4 py-1.5 text-xs font-semibold text-white transition shadow-sm"
          >
            Done
          </button>
        </div>
      </div>
    </div>
  );
}
