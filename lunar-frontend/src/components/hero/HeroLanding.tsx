"use client";

import { useState } from "react";
import dynamic from "next/dynamic";
import CosmicBackground from "./CosmicBackground";
import type { PayloadMode, LunarPhase } from "./LunarGlobe";

// Dynamic import for Three.js WebGL canvas to guarantee clean client-side rendering
const LunarGlobe = dynamic(() => import("./LunarGlobe"), {
  ssr: false,
  loading: () => (
    <div className="flex h-full w-full items-center justify-center">
      <div className="h-24 w-24 animate-pulse rounded-full bg-border/40" />
    </div>
  ),
});

interface Props {
  onLaunchConsole: () => void;
}

export default function HeroLanding({ onLaunchConsole }: Props) {
  const [payloadMode, setPayloadMode] = useState<PayloadMode>("optical");
  const [phase, setPhase] = useState<LunarPhase>("crescent");
  const [autoRotate, setAutoRotate] = useState(true);

  return (
    <main className="relative h-screen w-screen select-none overflow-hidden bg-[#02040a] font-sans text-ink">
      {/* 1. Deep Space Cosmic Starfield & Meteors */}
      <CosmicBackground />

      {/* 2. Interactive Three.js 3D Moon Canvas */}
      <div className="absolute inset-0 z-10 flex items-center justify-center">
        <LunarGlobe
          payloadMode={payloadMode}
          phase={phase}
          autoRotate={autoRotate}
        />
      </div>

      {/* 3. Top Navigation Bar (Glassmorphism) */}
      <header className="relative z-30 flex items-center justify-between border-b border-white/5 bg-void/40 px-6 py-4 backdrop-blur-md">
        <div className="flex items-center gap-3">
          <div className="flex h-8 w-8 items-center justify-center rounded-full border border-regolith/40 bg-regolith/10 text-xs font-mono font-bold text-regolith">
            C2
          </div>
          <div>
            <div className="flex items-center gap-2">
              <span className="font-mono text-xs font-semibold tracking-wider text-regolith">
                SIH26166
              </span>
              <span className="rounded bg-white/10 px-1.5 py-0.5 text-[10px] font-medium text-white/80 uppercase tracking-widest">
                ISRO
              </span>
            </div>
            <p className="text-[11px] text-ink-dim">Chandrayaan-2 Lunar Crossmatch</p>
          </div>
        </div>

        {/* Live Payload Chips */}
        <div className="hidden items-center gap-2 md:flex">
          <span className="inline-flex items-center gap-1.5 rounded-full border border-white/10 bg-white/5 px-2.5 py-1 text-[11px] text-ink-dim">
            <span className="h-1.5 w-1.5 rounded-full bg-emerald-400" />
            OHRC 0.25m
          </span>
          <span className="inline-flex items-center gap-1.5 rounded-full border border-white/10 bg-white/5 px-2.5 py-1 text-[11px] text-ink-dim">
            <span className="h-1.5 w-1.5 rounded-full bg-cyan-400" />
            TMC-2 Stereo
          </span>
          <span className="inline-flex items-center gap-1.5 rounded-full border border-white/10 bg-white/5 px-2.5 py-1 text-[11px] text-ink-dim">
            <span className="h-1.5 w-1.5 rounded-full bg-purple-400" />
            IIRS Hyperspectral
          </span>
        </div>

        {/* Action Button */}
        <div className="flex items-center gap-3">
          <button
            onClick={onLaunchConsole}
            className="group relative flex items-center gap-2 overflow-hidden rounded-full border border-regolith/50 bg-regolith/15 px-4 py-1.5 text-xs font-medium text-regolith shadow-[0_0_20px_rgba(232,163,61,0.2)] transition-all duration-300 hover:border-regolith hover:bg-regolith hover:text-black hover:shadow-[0_0_28px_rgba(232,163,61,0.4)]"
          >
            <span>Launch Console</span>
            <span className="transition-transform duration-200 group-hover:translate-x-0.5">
              →
            </span>
          </button>
        </div>
      </header>

      {/* 4. Central Hero Overlay */}
      <div className="pointer-events-none relative z-20 flex h-[calc(100vh-140px)] flex-col items-center justify-between px-6 pt-8 pb-4 text-center">
        {/* Title Section (Top) */}
        <div className="flex max-w-3xl flex-col items-center">
          <div className="mb-3 inline-flex items-center gap-2 rounded-full border border-white/10 bg-white/5 px-3 py-1 text-xs font-mono text-ink-dim backdrop-blur-sm">
            <span className="h-1.5 w-1.5 animate-ping rounded-full bg-regolith" />
            <span>Multi-Modal Lunar Image Registration</span>
          </div>

          <h1 className="text-4xl font-extrabold tracking-tight text-white drop-shadow-[0_4px_28px_rgba(0,0,0,0.95)] md:text-6xl lg:text-7xl">
            Lunar Correspondence
          </h1>

          <p className="mt-3 max-w-xl text-sm font-medium text-slate-200/90 drop-shadow-[0_2px_12px_rgba(0,0,0,0.95)] md:text-base">
            Illumination-robust and sub-pixel correspondence across Chandrayaan-2's
            OHRC, TMC-2, and IIRS optical payloads.
          </p>

          <p className="mt-3 text-xs font-mono tracking-wide text-amber-200/90 drop-shadow-[0_2px_8px_rgba(0,0,0,0.9)]">
            (Click and drag to rotate the Moon · Scroll to zoom)
          </p>

          {/* Direct CTA Buttons */}
          <div className="pointer-events-auto mt-6 flex flex-wrap items-center justify-center gap-4">
            <button
              onClick={onLaunchConsole}
              className="flex items-center gap-2 rounded-full bg-gradient-to-r from-regolith to-amber-500 px-6 py-3 text-sm font-semibold text-black shadow-[0_0_25px_rgba(232,163,61,0.35)] transition-all duration-300 hover:scale-105 hover:shadow-[0_0_35px_rgba(232,163,61,0.55)] active:scale-95"
            >
              <span>Launch Correspondence Console</span>
              <span>→</span>
            </button>
            <button
              onClick={onLaunchConsole}
              className="flex items-center gap-2 rounded-full border border-white/20 bg-white/5 px-5 py-3 text-sm font-medium text-white backdrop-blur-md transition-all duration-200 hover:border-white/40 hover:bg-white/10"
            >
              <span>Explore 8 Validated Regions</span>
            </button>
          </div>
        </div>

        {/* Bottom Key Metrics Badges */}
        <div className="pointer-events-auto grid w-full max-w-4xl grid-cols-2 gap-3 sm:grid-cols-4">
          <div className="rounded-xl border border-white/10 bg-white/[0.03] p-3 text-left backdrop-blur-md">
            <p className="font-mono text-lg font-bold text-white md:text-xl">0.25 m</p>
            <p className="text-[11px] text-ink-dim">OHRC Resolution</p>
          </div>
          <div className="rounded-xl border border-white/10 bg-white/[0.03] p-3 text-left backdrop-blur-md">
            <p className="font-mono text-lg font-bold text-white md:text-xl">8 Regions</p>
            <p className="text-[11px] text-ink-dim">Multi-Sensor Triplets</p>
          </div>
          <div className="rounded-xl border border-white/10 bg-white/[0.03] p-3 text-left backdrop-blur-md">
            <p className="font-mono text-lg font-bold text-emerald-400 md:text-xl">&lt; 0.5 px</p>
            <p className="text-[11px] text-ink-dim">Sub-Pixel Accuracy (RMSE)</p>
          </div>
          <div className="rounded-xl border border-white/10 bg-white/[0.03] p-3 text-left backdrop-blur-md">
            <p className="font-mono text-lg font-bold text-parallax md:text-xl">LoFTR + RANSAC</p>
            <p className="text-[11px] text-ink-dim">Invariant Crossmatch</p>
          </div>
        </div>
      </div>

      {/* 5. Bottom Interactive Controls Toolbar */}
      <footer className="relative z-30 flex flex-wrap items-center justify-between border-t border-white/5 bg-void/50 px-6 py-3 text-xs backdrop-blur-md">
        {/* Left: Payload Shading Mode */}
        <div className="flex items-center gap-2">
          <span className="text-ink-faint">Payload Mode:</span>
          <div className="flex rounded-lg border border-white/10 bg-white/5 p-0.5">
            <button
              onClick={() => setPayloadMode("optical")}
              className={`rounded px-2.5 py-1 transition-all ${
                payloadMode === "optical"
                  ? "bg-white/20 text-white font-medium shadow"
                  : "text-ink-dim hover:text-white"
              }`}
            >
              Optical (OHRC/TMC)
            </button>
            <button
              onClick={() => setPayloadMode("iirs")}
              className={`rounded px-2.5 py-1 transition-all ${
                payloadMode === "iirs"
                  ? "bg-purple-500/30 text-purple-200 font-medium shadow"
                  : "text-ink-dim hover:text-white"
              }`}
            >
              IIRS Mineralogy
            </button>
            <button
              onClick={() => setPayloadMode("dem")}
              className={`rounded px-2.5 py-1 transition-all ${
                payloadMode === "dem"
                  ? "bg-emerald-500/30 text-emerald-200 font-medium shadow"
                  : "text-ink-dim hover:text-white"
              }`}
            >
              DEM Elevation
            </button>
          </div>
        </div>

        {/* Center: Lunar Lighting Phases */}
        <div className="hidden items-center gap-2 lg:flex">
          <span className="text-ink-faint">Sun Illumination:</span>
          <div className="flex gap-1">
            {(
              [
                { id: "crescent", label: "Crescent" },
                { id: "quarter", label: "Quarter" },
                { id: "gibbous", label: "Gibbous" },
                { id: "full", label: "Full" },
                { id: "new", label: "New" },
              ] as const
            ).map((p) => (
              <button
                key={p.id}
                onClick={() => setPhase(p.id)}
                className={`rounded px-2 py-0.5 font-mono text-[11px] transition-colors ${
                  phase === p.id
                    ? "border border-regolith/50 bg-regolith/20 text-regolith font-medium"
                    : "border border-white/5 text-ink-faint hover:text-white"
                }`}
              >
                {p.label}
              </button>
            ))}
          </div>
        </div>

        {/* Right: Orbit control & sound */}
        <div className="flex items-center gap-3">
          <label className="flex cursor-pointer items-center gap-2 text-ink-dim hover:text-white">
            <input
              type="checkbox"
              checked={autoRotate}
              onChange={(e) => setAutoRotate(e.target.checked)}
              className="accent-regolith"
            />
            <span className="font-mono text-[11px]">Auto Orbit</span>
          </label>

          <span className="h-3 w-px bg-white/10" />

          <button
            onClick={onLaunchConsole}
            className="flex items-center gap-1.5 font-mono text-regolith transition-colors hover:text-amber-300"
          >
            <span>Open Console</span>
            <span>↗</span>
          </button>
        </div>
      </footer>
    </main>
  );
}
