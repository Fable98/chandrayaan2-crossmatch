"use client";

interface Props {
  onClose: () => void;
}

export default function TheoryModal({ onClose }: Props) {
  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/80 p-4 backdrop-blur-2xl animate-fade-in">
      <div className="relative flex max-h-[90vh] w-full max-w-3xl flex-col overflow-hidden rounded-3xl border border-white/15 bg-[#0c0e24]/90 text-white shadow-[0_30px_90px_rgba(0,0,0,0.9)] ring-1 ring-white/10">
        {/* Header Bar */}
        <div className="flex items-center justify-between border-b border-white/10 bg-white/[0.02] px-6 py-4 backdrop-blur-md">
          <div className="flex items-center gap-2">
            <span className="rounded-full border border-purple-400/30 bg-purple-500/10 px-2.5 py-0.5 text-[10px] font-semibold uppercase tracking-wider text-purple-300">
              Methodology
            </span>
            <span className="text-sm font-bold text-white">
              Projective Homography &amp; Registration Pipeline
            </span>
          </div>

          <button
            onClick={onClose}
            className="flex h-7 w-7 items-center justify-center rounded-full border border-white/10 bg-white/[0.04] text-xs text-slate-300 transition hover:bg-white/[0.08] hover:text-white"
            title="Close"
          >
            ✕
          </button>
        </div>

        {/* Body Content */}
        <div className="flex-1 overflow-y-auto p-6 md:p-8 space-y-6">
          <div>
            <h2 className="text-lg font-bold text-white">
              Mathematical &amp; Algorithmic Framework
            </h2>
            <p className="mt-1 text-xs text-slate-300">
              Cross-sensor alignment between disparate orbital passes and extreme solar incidence reversals.
            </p>
          </div>

          <div className="border-t border-white/[0.08] pt-4 text-xs leading-relaxed text-slate-300 space-y-4">
            <p>
              In planetary cross-matching between high-resolution optical cameras (OHRC, 0.25 m/px) and stereo/hyperspectral terrain cameras (TMC-2, 4 m/px), sensor viewing geometries differ radically. Due to non-repeat orbital tracks, the angle of solar incidence often reverses by &gt;160°, rendering traditional pixel intensity metrics invalid.
            </p>

            <div className="rounded-2xl border border-white/10 bg-white/[0.03] p-4 text-xs text-white">
              <span className="text-purple-300 font-mono block mb-1 font-semibold">Planar Projective Transform:</span>
              <span className="font-mono text-slate-200">s · [x&apos;, y&apos;, 1]ᵀ = H · [x, y, 1]ᵀ</span>
              <p className="mt-2 text-[11px] text-slate-400">
                Where H is a 3×3 matrix with 8 degrees of freedom calculated via Random Sample Consensus (RANSAC) on dense Transformer-based correspondences (LoFTR).
              </p>
            </div>

            <h4 className="text-xs font-bold text-white uppercase tracking-wider">
              Key Pipeline Steps:
            </h4>
            <ul className="list-disc list-inside space-y-2 text-slate-300">
              <li>
                <strong className="text-white">Local Feature Transformer (LoFTR):</strong> Establishes semi-dense correspondences without explicit detector bottlenecks, allowing matching inside steep crater shadows.
              </li>
              <li>
                <strong className="text-white">RANSAC Homography:</strong> Filters out erroneous correspondences caused by inverted shadow edges with sub-pixel tolerance (threshold &lt; 3.0 px).
              </li>
              <li>
                <strong className="text-white">Sub-Pixel Refinement:</strong> Minimizes reprojection error to achieve an RMSE &lt; 0.5 px across the shared terrain footprint.
              </li>
            </ul>
          </div>
        </div>

        {/* Footer */}
        <div className="flex items-center justify-between border-t border-white/10 bg-white/[0.02] px-6 py-3 text-xs">
          <span className="text-slate-400">Chandrayaan-2 Registration Pipeline</span>
          <button
            onClick={onClose}
            className="rounded-full border border-white/10 bg-white/[0.04] px-4 py-1.5 text-xs text-slate-300 hover:bg-white/[0.08] transition"
          >
            Close
          </button>
        </div>
      </div>
    </div>
  );
}
