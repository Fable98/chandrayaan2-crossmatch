"use client";

interface Props {
  onClose: () => void;
}

export default function TheoryModal({ onClose }: Props) {
  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/80 p-4 backdrop-blur-md">
      <div className="relative flex max-h-[90vh] w-full max-w-3xl flex-col overflow-hidden rounded-2xl border border-white/15 bg-[#080b12]/90 text-[#f0f2f5] shadow-[0_20px_60px_rgba(0,0,0,0.9)] backdrop-blur-2xl">
        {/* Header Bar */}
        <div className="flex items-center justify-between border-b border-white/10 bg-white/[0.03] px-6 py-4 backdrop-blur-md">
          <div className="flex items-center gap-2">
            <span className="font-mono text-xs text-[#d4af37]">[ THEORY DOSSIER ]</span>
            <span className="font-mono text-xs font-bold uppercase tracking-wider text-white">
              PROJECTIVE HOMOGRAPHY &amp; SUB-PIXEL GEOMETRY
            </span>
          </div>

          <button
            onClick={onClose}
            className="flex h-7 w-7 items-center justify-center rounded-lg border border-white/15 bg-white/[0.04] font-mono text-xs text-[#9a958e] backdrop-blur-sm transition-colors hover:border-[#d4af37] hover:text-[#d4af37] focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-[#d4af37]"
            title="Close Theory"
          >
            ✕
          </button>
        </div>

        {/* Body Content */}
        <div className="flex-1 overflow-y-auto p-6 md:p-8 space-y-6">
          <div>
            <span className="font-mono text-[10px] uppercase tracking-[0.2em] text-[#d4af37]">
              MATHEMATICAL FRAMEWORK
            </span>
            <h2 className="mt-1 font-serif text-3xl italic text-[#e8d5b5]">
              "The grid is a conceptual framework, an intellectual construct. It is not an image."
            </h2>
            <p className="mt-2 font-mono text-xs text-[#6b665f]">
              REF-99 · Epipolar Transformations across Orbital Passes
            </p>
          </div>

          <div className="border-t border-[#23211d] pt-4 text-xs leading-relaxed text-[#9a958e] space-y-4">
            <p>
              In planetary cross-matching between high-resolution optical cameras (OHRC, 0.25 m/px) and stereo/hyperspectral terrain cameras (TMC-2, 4 m/px), sensor viewing geometries differ radically. Due to non-repeat orbital tracks, the angle of solar incidence often reverses by &gt;160°, rendering traditional pixel intensity metrics (like cross-correlation or MSE) completely invalid.
            </p>

            <div className="border border-[#23211d] bg-[#121217] p-4 font-mono text-xs text-[#e8d5b5]">
              <span className="text-[#d4af37] block mb-1">// Planar Projective Transform:</span>
              <span>s · [x&apos;, y&apos;, 1]ᵀ = H · [x, y, 1]ᵀ</span>
              <p className="mt-2 text-[11px] text-[#6b665f]">
                Where H is a 3×3 matrix with 8 degrees of freedom calculated via Random Sample Consensus (RANSAC) on dense Transformer-based correspondences (LoFTR).
              </p>
            </div>

            <h4 className="font-mono text-xs font-bold uppercase tracking-wider text-white">
              Key Invariant Steps in SIH26166 Pipeline:
            </h4>
            <ul className="list-disc list-inside space-y-2 text-[#9a958e]">
              <li>
                <strong className="text-[#e8d5b5]">Local Feature Transformer (LoFTR):</strong> Establishes semi-dense correspondences without explicit detector bottlenecks, allowing matching inside steep crater shadows.
              </li>
              <li>
                <strong className="text-[#e8d5b5]">RANSAC Homography:</strong> Filters out erroneous correspondences caused by inverted shadow edges with sub-pixel tolerance (threshold &lt; 3.0 px).
              </li>
              <li>
                <strong className="text-[#e8d5b5]">Sub-Pixel Refinement:</strong> Minimizes reprojection error to achieve an RMSE &lt; 0.5 px across the shared terrain footprint.
              </li>
            </ul>
          </div>
        </div>

        {/* Footer */}
        <div className="flex items-center justify-between border-t border-[#23211d] bg-[#121217] px-6 py-3 font-mono text-[11px]">
          <span className="text-[#6b665f]">Chandrayaan-2 Algorithmic Foundation</span>
          <button
            onClick={onClose}
            className="border border-[#38342d] bg-[#0d0d11] px-4 py-1.5 text-[#d4af37] hover:bg-[#2c2619] transition-colors focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-[#d4af37]"
          >
            Understood ✕
          </button>
        </div>
      </div>
    </div>
  );
}
