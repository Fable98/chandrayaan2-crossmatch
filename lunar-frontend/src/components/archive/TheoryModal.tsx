"use client";

interface Props {
  onClose: () => void;
}

export default function TheoryModal({ onClose }: Props) {
  return (
    <div className="fixed inset-0 z-[2000] flex items-center justify-center bg-black/60 p-4 backdrop-blur-xs animate-fade-in font-mono">
      <div className="retro-outset relative flex max-h-[90vh] w-full max-w-3xl flex-col overflow-hidden bg-[#E7E2D6] dark:bg-[#1A201E] text-[#1E2321] dark:text-[#E7E2D6] shadow-2xl">
        {/* Retro Window Titlebar */}
        <div className="flex items-center justify-between bg-[#1F4743] px-3 py-1.5 text-xs font-bold font-mono text-white select-none shrink-0 border-b border-[#143532]">
          <div className="flex items-center gap-2">
            <span className="text-sm">📐</span>
            <span className="tracking-wider uppercase">
              METHODOLOGY // PROJECTIVE HOMOGRAPHY &amp; REGISTRATION PIPELINE
            </span>
          </div>

          <div className="flex items-center space-x-1">
            <button
              type="button"
              className="window-ctrl-btn"
              title="Minimize"
              tabIndex={-1}
            >
              _
            </button>
            <button
              type="button"
              className="window-ctrl-btn"
              title="Maximize"
              tabIndex={-1}
            >
              □
            </button>
            <button
              type="button"
              onClick={onClose}
              className="window-ctrl-btn hover:bg-rose-700 hover:text-white font-bold"
              title="Close [Esc]"
            >
              ✕
            </button>
          </div>
        </div>

        {/* Body Content */}
        <div className="flex-1 overflow-y-auto p-4 sm:p-6 space-y-4">
          <div className="border-b border-[#8B8579] dark:border-[#2D3835] pb-3">
            <h2 className="text-base font-bold text-[#1E2321] dark:text-[#E7E2D6]">
              Mathematical &amp; Algorithmic Framework
            </h2>
            <p className="mt-1 text-xs text-[#555C58] dark:text-[#8C9893]">
              Cross-sensor alignment between disparate orbital passes and extreme solar incidence reversals.
            </p>
          </div>

          <div className="text-xs leading-relaxed text-[#4A524E] dark:text-[#A8B2AD] space-y-3">
            <p>
              In planetary cross-matching between high-resolution optical cameras (OHRC, 0.25 m/px) and monoscopic terrain camera (TMC-2 single view, 4 m/px) and hyperspectral camera, sensor viewing geometries differ radically. Due to non-repeat orbital tracks, the angle of solar incidence often reverses by &gt;160°, rendering traditional pixel intensity metrics invalid.
            </p>

            <div className="retro-inset p-3 bg-[#DED8CB]/50 dark:bg-[#141817] text-xs">
              <span className="text-[#1F4743] dark:text-teal-300 font-mono block mb-1 font-bold">
                [PLANAR PROJECTIVE TRANSFORM FORMULATION]
              </span>
              <span className="font-mono font-bold text-[#1E2321] dark:text-[#E7E2D6] block py-1">
                s · [x&apos;, y&apos;, 1]ᵀ = H · [x, y, 1]ᵀ
              </span>
              <p className="mt-1 text-[11px] text-[#555C58] dark:text-[#8C9893]">
                Where H is a 3×3 matrix with 8 degrees of freedom calculated via Random Sample Consensus (RANSAC) on dense Transformer-based correspondences (LoFTR).
              </p>
            </div>

            <h4 className="text-xs font-bold text-[#1E2321] dark:text-[#E7E2D6] uppercase tracking-wider pt-2">
              Key Pipeline Steps:
            </h4>
            <ul className="list-disc list-inside space-y-1.5 text-[#4A524E] dark:text-[#A8B2AD]">
              <li>
                <strong className="text-[#1E2321] dark:text-[#E7E2D6]">Local Feature Transformer (LoFTR):</strong> Establishes semi-dense correspondences without explicit detector bottlenecks, allowing matching inside steep crater shadows.
              </li>
              <li>
                <strong className="text-[#1E2321] dark:text-[#E7E2D6]">RANSAC Homography:</strong> Filters out erroneous correspondences caused by inverted shadow edges with sub-pixel tolerance (threshold &lt; 3.0 px).
              </li>
              <li>
                <strong className="text-[#1E2321] dark:text-[#E7E2D6]">Sub-Pixel Refinement:</strong> Minimizes reprojection error to achieve an RMSE &lt; 0.5 px across the shared terrain footprint.
              </li>
            </ul>
          </div>
        </div>

        {/* Footer */}
        <div className="flex items-center justify-between border-t border-[#8B8579] dark:border-[#2D3835] bg-[#DED8CB] dark:bg-[#141817] px-4 py-2 text-xs">
          <span className="text-[#555C58] dark:text-[#8C9893] font-medium">ISRO Chandrayaan-2 Registration Pipeline</span>
          <button
            onClick={onClose}
            className="retro-button-primary px-4 py-1 text-xs font-mono font-bold"
          >
            Acknowledge [OK]
          </button>
        </div>
      </div>
    </div>
  );
}

