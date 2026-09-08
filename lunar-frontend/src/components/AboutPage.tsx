"use client";

import React from "react";

interface Props {
  onBackToHero?: () => void;
  onOpenConsole?: () => void;
}

export default function AboutPage({ onBackToHero, onOpenConsole }: Props) {
  return (
    <div className="min-h-screen bg-[#08080a] font-sans text-[#f0f2f5] selection:bg-[#2c2619] selection:text-[#f3df9b]">
      {/* 1. Sticky Navigation Header */}
      <header className="sticky top-0 z-50 flex h-16 items-center justify-between border-b border-[#23211d] bg-[#08080a]/90 px-6 backdrop-blur-md md:px-12">
        <button
          onClick={onBackToHero}
          className="group flex items-center gap-2 font-mono text-xs text-[#9a958e] transition-colors hover:text-white"
        >
          <span className="transition-transform duration-200 group-hover:-translate-x-1">
            ←
          </span>
          <span>Return to Lunar Globe</span>
        </button>

        <div className="hidden items-center gap-2 font-mono text-xs font-semibold uppercase tracking-widest text-[#d4af37] md:flex">
          <span className="h-1.5 w-1.5 rounded-full bg-[#d4af37]" />
          <span>SIH 26166 · Mission Briefing</span>
        </div>

        <button
          onClick={onOpenConsole}
          className="flex items-center gap-2 rounded-full border border-teal/40 bg-teal/10 px-4 py-1.5 font-mono text-xs font-semibold text-teal backdrop-blur-sm transition-all duration-200 hover:bg-teal/20 hover:scale-105"
        >
          <span>Open Dashboard</span>
          <span>↗</span>
        </button>
      </header>

      {/* 2. Hero Headline Section */}
      <section className="relative overflow-hidden border-b border-[#23211d] px-6 py-16 md:px-12 md:py-24">
        {/* Ambient background glow */}
        <div className="pointer-events-none absolute -top-24 left-1/2 h-96 w-[700px] -translate-x-1/2 rounded-full bg-teal/5 blur-[120px]" />

        <div className="relative mx-auto max-w-5xl text-left">
          <div className="inline-flex items-center gap-2 rounded-full border border-teal/30 bg-teal/10 px-3 py-1 font-mono text-2xs uppercase tracking-wider text-teal">
            <span>ISRO Space Applications Centre (SAC)</span>
            <span>·</span>
            <span>SIH 26166</span>
          </div>

          <h1 className="mt-6 text-3xl font-extrabold tracking-tight text-white md:text-5xl lg:text-6xl lg:leading-[1.1]">
            Multi-Modal, Sun-Angle &amp; Scale-Invariant Image Correspondence
          </h1>

          <p className="mt-6 max-w-3xl text-base leading-relaxed text-[#9a958e] md:text-lg md:leading-relaxed">
            Autonomous geometric co-registration and sub-pixel tie-point extraction across
            Chandrayaan-2&apos;s <span className="text-white font-medium">OHRC</span> (0.25 m/px),{" "}
            <span className="text-white font-medium">TMC-2</span> (5.0 m/px), and{" "}
            <span className="text-white font-medium">IIRS</span> (80 m/px) planetary rasters under
            extreme illumination, relief, and scale disparities.
          </p>

          {/* Quick Metrics Badges */}
          <div className="mt-8 flex flex-wrap gap-3 font-mono text-xs">
            <div className="rounded-lg border border-[#23211d] bg-[#121217] px-3.5 py-2">
              <span className="text-ink-faint">Scale Disparity: </span>
              <span className="text-teal font-semibold">20× to 320×</span>
            </div>
            <div className="rounded-lg border border-[#23211d] bg-[#121217] px-3.5 py-2">
              <span className="text-ink-faint">Illumination: </span>
              <span className="text-[#d4af37] font-semibold">Phase Congruency (CFOG)</span>
            </div>
            <div className="rounded-lg border border-[#23211d] bg-[#121217] px-3.5 py-2">
              <span className="text-ink-faint">Precision: </span>
              <span className="text-emerald-400 font-semibold">0.1 px Sub-Pixel Peak</span>
            </div>
            <div className="rounded-lg border border-[#23211d] bg-[#121217] px-3.5 py-2">
              <span className="text-ink-faint">Geodesy: </span>
              <span className="text-white font-semibold">IAU Moon2000 Equirectangular</span>
            </div>
          </div>
        </div>
      </section>

      {/* 3. The Multi-Sensor Optical Disparity Challenge */}
      <section className="border-b border-[#23211d] bg-[#0d0d11] px-6 py-16 md:px-12 md:py-20">
        <div className="mx-auto max-w-5xl">
          <div className="flex items-center gap-2 font-mono text-xs font-bold uppercase tracking-[0.2em] text-[#d4af37]">
            <span className="h-1.5 w-1.5 bg-[#d4af37]" />
            <span>01 · The Planetary Problem Space</span>
          </div>

          <h2 className="mt-3 text-2xl font-bold tracking-tight text-white md:text-3xl">
            Why Classical Computer Vision Fails On The Moon
          </h2>

          <p className="mt-4 max-w-3xl text-sm leading-relaxed text-[#9a958e]">
            The lunar surface presents unique optical challenges. Without an atmosphere to diffuse light,
            shadows cast by crater rims are pitch black and non-linear. As the sun moves, shadow vectors
            invert and elongate by hundreds of percent, rendering standard gradient-based descriptors
            (SIFT, SURF, ORB) and photometric matchers completely ineffective.
          </p>

          {/* 3 Payload Cards */}
          <div className="mt-10 grid gap-6 md:grid-cols-3">
            {/* Card 1: OHRC */}
            <div className="rounded-2xl border border-[#23211d] bg-[#121217] p-6 shadow-xl transition-all duration-300 hover:border-teal/40">
              <div className="flex items-center justify-between">
                <span className="font-mono text-xs font-bold uppercase tracking-wider text-teal">
                  OHRC
                </span>
                <span className="rounded bg-teal/10 px-2 py-0.5 font-mono text-2xs text-teal font-semibold">
                  0.25–0.32 m/px
                </span>
              </div>
              <h3 className="mt-3 text-base font-semibold text-white">
                Optical High Resolution Camera
              </h3>
              <p className="mt-2 text-xs leading-relaxed text-[#9a958e]">
                Narrow-angle framing camera designed for lunar hazard detection and landing site certification.
                Exhibits exquisite meter-scale morphology, crater textures, and severe relief displacement from steep slopes.
              </p>
              <div className="mt-4 border-t border-[#23211d] pt-3 font-mono text-2xs text-ink-faint">
                Role: Moving / Source Image ($I_1$)
              </div>
            </div>

            {/* Card 2: TMC-2 */}
            <div className="rounded-2xl border border-[#23211d] bg-[#121217] p-6 shadow-xl transition-all duration-300 hover:border-[#d4af37]/40">
              <div className="flex items-center justify-between">
                <span className="font-mono text-xs font-bold uppercase tracking-wider text-[#d4af37]">
                  TMC-2
                </span>
                <span className="rounded bg-[#d4af37]/10 px-2 py-0.5 font-mono text-2xs text-[#d4af37] font-semibold">
                  ~5.0 m/px
                </span>
              </div>
              <h3 className="mt-3 text-base font-semibold text-white">
                Terrain Mapping Camera-2
              </h3>
              <p className="mt-2 text-xs leading-relaxed text-[#9a958e]">
                Along-track stereo camera producing Fore, Nadir, and Aft view strips used to derive 3D Digital
                Elevation Models (DEMs). Provides intermediate regional topographic context across a 20× scale gap.
              </p>
              <div className="mt-4 border-t border-[#23211d] pt-3 font-mono text-2xs text-ink-faint">
                Role: Fixed / Geometric Reference ($I_2$)
              </div>
            </div>

            {/* Card 3: IIRS */}
            <div className="rounded-2xl border border-[#23211d] bg-[#121217] p-6 shadow-xl transition-all duration-300 hover:border-purple-400/40">
              <div className="flex items-center justify-between">
                <span className="font-mono text-xs font-bold uppercase tracking-wider text-purple-300">
                  IIRS
                </span>
                <span className="rounded bg-purple-500/10 px-2 py-0.5 font-mono text-2xs text-purple-300 font-semibold">
                  ~80.0 m/px
                </span>
              </div>
              <h3 className="mt-3 text-base font-semibold text-white">
                Imaging Infrared Spectrometer
              </h3>
              <p className="mt-2 text-xs leading-relaxed text-[#9a958e]">
                Hyperspectral sensor capturing 256 contiguous spectral channels from 0.8 to 5.0 µm for lunar
                mineralogy, hydration, and OH/H₂O absorption mapping. Spans a 320× scale gap relative to OHRC.
              </p>
              <div className="mt-4 border-t border-[#23211d] pt-3 font-mono text-2xs text-ink-faint">
                Role: Mineralogical Spectral Overlay
              </div>
            </div>
          </div>
        </div>
      </section>

      {/* 4. Algorithmic Solution Architecture */}
      <section className="border-b border-[#23211d] px-6 py-16 md:px-12 md:py-20">
        <div className="mx-auto max-w-5xl">
          <div className="flex items-center gap-2 font-mono text-xs font-bold uppercase tracking-[0.2em] text-teal">
            <span className="h-1.5 w-1.5 bg-teal" />
            <span>02 · Algorithmic Innovation</span>
          </div>

          <h2 className="mt-3 text-2xl font-bold tracking-tight text-white md:text-3xl">
            The Multi-Scale Registration Architecture
          </h2>

          <div className="mt-10 space-y-8">
            {/* Step 1 */}
            <div className="rounded-2xl border border-[#23211d] bg-[#0d0d11] p-6 md:p-8">
              <div className="flex items-start gap-4">
                <span className="flex h-8 w-8 shrink-0 items-center justify-center rounded-lg border border-teal/40 bg-teal/10 font-mono text-sm font-bold text-teal">
                  1
                </span>
                <div>
                  <h3 className="text-lg font-bold text-white">
                    2D Phase Congruency &amp; CFOG Structural Transformation
                  </h3>
                  <p className="mt-2 text-xs leading-relaxed text-[#9a958e] md:text-sm">
                    Rather than operating on fragile raw DN intensities, the pipeline decomposes images into
                    illumination-invariant structural representations using 2D Log-Gabor filter banks across multiple scales and orientations.
                    Phase Congruency detects significant structural boundaries where Fourier phase components align, completely independent of illumination intensity or shadow direction.
                    Channel Features of Oriented Gradients (CFOG) compile these structural vectors into dense directional tensors.
                  </p>
                </div>
              </div>
            </div>

            {/* Step 2 */}
            <div className="rounded-2xl border border-[#23211d] bg-[#0d0d11] p-6 md:p-8">
              <div className="flex items-start gap-4">
                <span className="flex h-8 w-8 shrink-0 items-center justify-center rounded-lg border border-[#d4af37]/40 bg-[#d4af37]/10 font-mono text-sm font-bold text-[#d4af37]">
                  2
                </span>
                <div>
                  <h3 className="text-lg font-bold text-white">
                    Common Physical-GSD Normalization &amp; Canvas Preservation
                  </h3>
                  <p className="mt-2 text-xs leading-relaxed text-[#9a958e] md:text-sm">
                    The registration engine reads genuine physical GSD metadata from PDS4 labels and normalizes both rasters to a mathematically conditioned working resolution.
                    Grid search windows and template patches are sized proportionally to prevent template starvation, ensuring that downsampled canvases retain sufficient spatial support for robust correlation.
                  </p>
                </div>
              </div>
            </div>

            {/* Step 3 */}
            <div className="rounded-2xl border border-[#23211d] bg-[#0d0d11] p-6 md:p-8">
              <div className="flex items-start gap-4">
                <span className="flex h-8 w-8 shrink-0 items-center justify-center rounded-lg border border-emerald-400/40 bg-emerald-400/10 font-mono text-sm font-bold text-emerald-400">
                  3
                </span>
                <div>
                  <h3 className="text-lg font-bold text-white">
                    Fourier Phase Correlation with 0.1 px Sub-Pixel Refinement
                  </h3>
                  <p className="mt-2 text-xs leading-relaxed text-[#9a958e] md:text-sm">
                    Candidate structural tie points are refined in the frequency domain using Fourier Phase Correlation:
                  </p>
                  <div className="my-2.5 rounded-lg border border-emerald-500/20 bg-black/60 px-3.5 py-2 font-mono text-xs text-emerald-300">
                    Q(u, v) = [ F₁(u, v) · F₂*(u, v) ] / | F₁(u, v) · F₂*(u, v) |
                  </div>
                  <p className="text-xs leading-relaxed text-[#9a958e] md:text-sm">
                    Cross-power spectrum peaks are interpolated via 2D quadratic parabolic peak estimation to determine fractional offsets with true sub-pixel precision down to 0.1 pixels.
                  </p>
                </div>
              </div>
            </div>

            {/* Step 4 */}
            <div className="rounded-2xl border border-[#23211d] bg-[#0d0d11] p-6 md:p-8">
              <div className="flex items-start gap-4">
                <span className="flex h-8 w-8 shrink-0 items-center justify-center rounded-lg border border-amber-400/40 bg-amber-400/10 font-mono text-sm font-bold text-amber-400">
                  4
                </span>
                <div>
                  <h3 className="text-lg font-bold text-white">
                    10×10 Spatial Uniformity Gate &amp; Homography Conditioning
                  </h3>
                  <p className="mt-2 text-xs leading-relaxed text-[#9a958e] md:text-sm">
                    A cluster of 50 tie points on a single high-contrast crater rim produces a degenerate, rank-deficient homography matrix that shears the rest of the image.
                    Our pipeline partitions the image canvas into a 10×10 spatial grid, keeping only the highest-confidence correspondence per cell.
                    Strict quality gates verify condition numbers (cond(H) &lt; 10³), eigenvalue determinants, and spatial quadrant coverage before any matrix is certified.
                  </p>
                </div>
              </div>
            </div>

            {/* Step 5 */}
            <div className="rounded-2xl border border-[#23211d] bg-[#0d0d11] p-6 md:p-8">
              <div className="flex items-start gap-4">
                <span className="flex h-8 w-8 shrink-0 items-center justify-center rounded-lg border border-cyan-400/40 bg-cyan-400/10 font-mono text-sm font-bold text-cyan-400">
                  5
                </span>
                <div>
                  <h3 className="text-lg font-bold text-white">
                    Hierarchical Scale-Bridge Architecture for 320× Triplet Closure
                  </h3>
                  <p className="mt-2 text-xs leading-relaxed text-[#9a958e] md:text-sm">
                    Direct feature matching between 0.25 m OHRC and 80 m IIRS is physically unconditioned (an entire 128m OHRC tile collapses into a 1.6-pixel dot on IIRS).
                    Our architecture transitively composes the transform using TMC-2 as the physical intermediate geometric bridge:
                  </p>
                  <div className="my-2.5 rounded-lg border border-cyan-500/20 bg-black/60 px-3.5 py-2 font-mono text-xs text-cyan-300">
                    H(OHRC → IIRS) = H(TMC → IIRS) · H(OHRC → TMC)
                  </div>
                  <p className="text-xs leading-relaxed text-[#9a958e] md:text-sm">
                    This decomposes the impossible 320× gap into two physically tractable steps (20× and 16×).
                  </p>
                </div>
              </div>
            </div>
          </div>
        </div>
      </section>

      {/* 5. Registered Deliverables & Products */}
      <section className="border-b border-[#23211d] bg-[#0d0d11] px-6 py-16 md:px-12 md:py-20">
        <div className="mx-auto max-w-5xl">
          <div className="flex items-center gap-2 font-mono text-xs font-bold uppercase tracking-[0.2em] text-[#d4af37]">
            <span className="h-1.5 w-1.5 bg-[#d4af37]" />
            <span>03 · Registered Output Products</span>
          </div>

          <h2 className="mt-3 text-2xl font-bold tracking-tight text-white md:text-3xl">
            Production-Grade Geospatial Deliverables
          </h2>

          <div className="mt-10 grid gap-6 md:grid-cols-2">
            <div className="rounded-2xl border border-[#23211d] bg-[#121217] p-6">
              <h3 className="font-mono text-sm font-bold text-white uppercase tracking-wider">
                1. 50px Alternating Checkerboard QA
              </h3>
              <p className="mt-2 text-xs leading-relaxed text-[#9a958e]">
                Interleaves 50×50 px blocks of the warped source and reference image to visually inspect crater rim continuity across boundaries.
                Alignment accuracy is immediately verifiable: crater circles remain unbroken across block borders.
              </p>
            </div>

            <div className="rounded-2xl border border-[#23211d] bg-[#121217] p-6">
              <h3 className="font-mono text-sm font-bold text-white uppercase tracking-wider">
                2. Standard GeoTIFF Planetary Rasters
              </h3>
              <p className="mt-2 text-xs leading-relaxed text-[#9a958e]">
                Exports fully georeferenced GeoTIFF files encoded with standard IAU/IAG Moon2000 Equirectangular Coordinate Reference System (CRS)
                and affine transformations for ingestion into QGIS, ArcGIS, or planetary GIS pipelines.
              </p>
            </div>

            <div className="rounded-2xl border border-[#23211d] bg-[#121217] p-6">
              <h3 className="font-mono text-sm font-bold text-white uppercase tracking-wider">
                3. Interactive Linked Cursor Visualizer
              </h3>
              <p className="mt-2 text-xs leading-relaxed text-[#9a958e]">
                Synchronized dual-viewport correspondence tool displaying verified tie points on both OHRC and TMC-2 with real-time confidence scores,
                interactive hover linking, and coordinate readouts.
              </p>
            </div>

            <div className="rounded-2xl border border-[#23211d] bg-[#121217] p-6">
              <h3 className="font-mono text-sm font-bold text-white uppercase tracking-wider">
                4. Audit-Grade Verification Metrics
              </h3>
              <p className="mt-2 text-xs leading-relaxed text-[#9a958e]">
                Outputs canonical RMSE, inlier ratios, reprojection error distribution (mean, median, max), and spatial uniformity scores.
                Zero synthetic identity matrix fallbacks — degenerate geometry is honestly reported rather than concealed.
              </p>
            </div>
          </div>
        </div>
      </section>

      {/* 6. Call to Action Footer */}
      <section className="px-6 py-20 text-center md:px-12 md:py-24">
        <div className="mx-auto max-w-3xl">
          <h2 className="text-3xl font-extrabold tracking-tight text-white md:text-4xl">
            Experience The Co-Registration Pipeline
          </h2>
          <p className="mt-4 text-sm text-[#9a958e]">
            Inspect precomputed Chandrayaan-2 regions, explore interactive dual-sensor match points,
            or upload custom arbitrary sensor images for live sub-pixel registration.
          </p>

          <div className="mt-8 flex flex-wrap items-center justify-center gap-4">
            <button
              onClick={onOpenConsole}
              className="rounded-full border border-teal/50 bg-teal px-8 py-3.5 font-mono text-sm font-bold text-black shadow-[0_0_30px_rgba(63,181,201,0.5)] transition-all duration-200 hover:bg-[#52cde3] hover:shadow-[0_0_40px_rgba(63,181,201,0.8)] hover:scale-105 active:scale-95"
            >
              Launch Planetary Dashboard →
            </button>
            <button
              onClick={onBackToHero}
              className="rounded-full border border-[#23211d] bg-[#121217] px-6 py-3.5 font-mono text-sm text-[#9a958e] transition-colors hover:border-white/20 hover:text-white"
            >
              Back to Hero
            </button>
          </div>
        </div>
      </section>
    </div>
  );
}
