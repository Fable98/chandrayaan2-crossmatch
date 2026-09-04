"use client";

interface Props {
  onClose: () => void;
}

export default function TheoryModal({ onClose }: Props) {
  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/70 p-4">
      <div className="relative flex max-h-[90vh] w-full max-w-3xl flex-col overflow-hidden rounded-lg border border-[#1f326e] bg-[#091540] text-white shadow-xl">
        {/* Header Bar */}
        <div className="flex items-center justify-between border-b border-[#1f326e] bg-[#12235c] px-6 py-4">
          <div className="flex items-center gap-2">
            <span className="text-xs font-semibold uppercase tracking-wider text-[#7692FF]">
              Methodology
            </span>
            <span className="text-sm font-semibold text-white">
              Projective Homography &amp; Registration Pipeline
            </span>
          </div>

          <button
            onClick={onClose}
            className="flex h-7 w-7 items-center justify-center rounded border border-[#1f326e] bg-[#091540] text-xs text-[#ABD2FA] transition-colors hover:border-[#283e84] hover:text-white"
            title="Close"
          >
            ✕
          </button>
        </div>

        {/* Body Content */}
        <div className="flex-1 overflow-y-auto p-6 md:p-8 space-y-6">
          <div>
            <h2 className="text-base font-semibold text-white">
              Mathematical &amp; Algorithmic Framework
            </h2>
            <p className="mt-1 text-xs text-[#ABD2FA]">
              Cross-sensor alignment between disparate orbital passes and illumination angles.
            </p>
          </div>

          <div className="border-t border-[#1f326e] pt-4 text-xs leading-relaxed text-[#ABD2FA] space-y-4">
            <p>
              In planetary cross-matching between high-resolution optical cameras (OHRC, 0.25 m/px) and stereo/hyperspectral terrain cameras (TMC-2, 4 m/px), sensor viewing geometries differ radically. Due to non-repeat orbital tracks, the angle of solar incidence often reverses by &gt;160°, rendering traditional pixel intensity metrics invalid.
            </p>

            <div className="rounded-md border border-[#1f326e] bg-[#12235c] p-4 text-xs text-white">
              <span className="text-[#7692FF] font-mono block mb-1">Planar Projective Transform:</span>
              <span className="font-mono text-white">s · [x&apos;, y&apos;, 1]ᵀ = H · [x, y, 1]ᵀ</span>
              <p className="mt-2 text-[11px] text-[#ABD2FA]">
                Where H is a 3×3 matrix with 8 degrees of freedom calculated via Random Sample Consensus (RANSAC) on dense Transformer-based correspondences (LoFTR).
              </p>
            </div>

            <h4 className="text-xs font-semibold text-white uppercase tracking-wider">
              Key Pipeline Steps:
            </h4>
            <ul className="list-disc list-inside space-y-2 text-[#ABD2FA]">
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
        <div className="flex items-center justify-between border-t border-[#1f326e] bg-[#12235c] px-6 py-3 text-xs">
          <span className="text-[#ABD2FA]">Chandrayaan-2 Registration Pipeline</span>
          <button
            onClick={onClose}
            className="rounded border border-[#1f326e] bg-[#091540] px-4 py-1.5 text-xs text-[#ABD2FA] hover:bg-[#1f326e] transition-colors"
          >
            Close
          </button>
        </div>
      </div>
    </div>
  );
}
