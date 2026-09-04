"use client";

interface Props {
  onClose: () => void;
}

export default function TheoryModal({ onClose }: Props) {
  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/70 p-4">
      <div className="relative flex max-h-[90vh] w-full max-w-3xl flex-col overflow-hidden rounded-lg border border-[#4A4A4A] bg-[#1a1d20] text-[#FFFFE3] shadow-xl">
        {/* Header Bar */}
        <div className="flex items-center justify-between border-b border-[#4A4A4A] bg-[#282c30] px-6 py-4">
          <div className="flex items-center gap-2">
            <span className="text-xs font-semibold uppercase tracking-wider text-[#6D8196]">
              Methodology
            </span>
            <span className="text-sm font-semibold text-[#FFFFE3]">
              Projective Homography &amp; Registration Pipeline
            </span>
          </div>

          <button
            onClick={onClose}
            className="flex h-7 w-7 items-center justify-center rounded border border-[#4A4A4A] bg-[#1a1d20] text-xs text-[#CBCBCB] transition-colors hover:border-[#565c63] hover:text-[#FFFFE3]"
            title="Close"
          >
            ✕
          </button>
        </div>

        {/* Body Content */}
        <div className="flex-1 overflow-y-auto p-6 md:p-8 space-y-6">
          <div>
            <h2 className="text-base font-semibold text-[#FFFFE3]">
              Mathematical &amp; Algorithmic Framework
            </h2>
            <p className="mt-1 text-xs text-[#CBCBCB]">
              Cross-sensor alignment between disparate orbital passes and illumination angles.
            </p>
          </div>

          <div className="border-t border-[#4A4A4A] pt-4 text-xs leading-relaxed text-[#CBCBCB] space-y-4">
            <p>
              In planetary cross-matching between high-resolution optical cameras (OHRC, 0.25 m/px) and stereo/hyperspectral terrain cameras (TMC-2, 4 m/px), sensor viewing geometries differ radically. Due to non-repeat orbital tracks, the angle of solar incidence often reverses by &gt;160°, rendering traditional pixel intensity metrics invalid.
            </p>

            <div className="rounded-md border border-[#4A4A4A] bg-[#282c30] p-4 text-xs text-[#FFFFE3]">
              <span className="text-[#6D8196] font-mono block mb-1">Planar Projective Transform:</span>
              <span className="font-mono text-[#FFFFE3]">s · [x&apos;, y&apos;, 1]ᵀ = H · [x, y, 1]ᵀ</span>
              <p className="mt-2 text-[11px] text-[#CBCBCB]">
                Where H is a 3×3 matrix with 8 degrees of freedom calculated via Random Sample Consensus (RANSAC) on dense Transformer-based correspondences (LoFTR).
              </p>
            </div>

            <h4 className="text-xs font-semibold text-[#FFFFE3] uppercase tracking-wider">
              Key Pipeline Steps:
            </h4>
            <ul className="list-disc list-inside space-y-2 text-[#CBCBCB]">
              <li>
                <strong className="text-[#FFFFE3]">Local Feature Transformer (LoFTR):</strong> Establishes semi-dense correspondences without explicit detector bottlenecks, allowing matching inside steep crater shadows.
              </li>
              <li>
                <strong className="text-[#FFFFE3]">RANSAC Homography:</strong> Filters out erroneous correspondences caused by inverted shadow edges with sub-pixel tolerance (threshold &lt; 3.0 px).
              </li>
              <li>
                <strong className="text-[#FFFFE3]">Sub-Pixel Refinement:</strong> Minimizes reprojection error to achieve an RMSE &lt; 0.5 px across the shared terrain footprint.
              </li>
            </ul>
          </div>
        </div>

        {/* Footer */}
        <div className="flex items-center justify-between border-t border-[#4A4A4A] bg-[#282c30] px-6 py-3 text-xs">
          <span className="text-[#CBCBCB]">Chandrayaan-2 Registration Pipeline</span>
          <button
            onClick={onClose}
            className="rounded border border-[#4A4A4A] bg-[#1a1d20] px-4 py-1.5 text-xs text-[#CBCBCB] hover:bg-[#4A4A4A] transition-colors"
          >
            Close
          </button>
        </div>
      </div>
    </div>
  );
}
