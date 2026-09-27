"use client";

import { useState, useEffect } from "react";
import Link from "next/link";

interface Props {
  onOpenConsole?: () => void;
  onOpenAbout?: () => void;
  onLogout?: () => void;
  userName?: string;
}

export default function ExploreMoonHero({
  onOpenConsole,
  onOpenAbout,
  onLogout,
  userName,
}: Props) {
  const [mousePos, setMousePos] = useState({ x: 0, y: 0 });
  const [isLoaded, setIsLoaded] = useState(false);

  useEffect(() => {
    setIsLoaded(true);

    const handleMouseMove = (e: MouseEvent) => {
      const x = (e.clientX / window.innerWidth - 0.5) * 2;
      const y = (e.clientY / window.innerHeight - 0.5) * 2;
      setMousePos({ x, y });
    };

    window.addEventListener("mousemove", handleMouseMove);
    return () => window.removeEventListener("mousemove", handleMouseMove);
  }, []);

  return (
    <section className="dark relative h-screen w-screen select-none overflow-hidden bg-[#0A0D0C] font-mono text-[#E7E2D6]">
      {/* 1. Full-Bleed Video Background with Subtle Parallax & Atmospheric Vignettes */}
      <div
        className="absolute inset-0 z-0 h-[106%] w-[106%] -left-[3%] -top-[3%] transition-transform duration-700 ease-out pointer-events-none"
        style={{
          transform: `translate3d(${mousePos.x * -8}px, ${mousePos.y * -8}px, 0)`,
        }}
      >
        <video
          autoPlay
          loop
          muted
          playsInline
          className="h-full w-full object-cover object-center"
        >
          <source src="/bgsih.mp4" type="video/mp4" />
        </video>

        {/* Cinematic High-Legibility Overlays */}
        <div className="pointer-events-none absolute inset-0 bg-gradient-to-t from-[#0A0D0C] via-black/40 to-black/75" />
        <div className="pointer-events-none absolute inset-0 bg-gradient-to-r from-black/80 via-transparent to-black/60" />
      </div>

      {/* 2. Twinkling Deep-Space Starlight Overlay */}
      <div className="pointer-events-none absolute inset-0 z-[1] overflow-hidden">
        {[
          { top: "12%", left: "18%", delay: "0s", duration: "3s" },
          { top: "25%", left: "75%", delay: "1.2s", duration: "4s" },
          { top: "45%", left: "88%", delay: "0.5s", duration: "3.5s" },
          { top: "70%", left: "62%", delay: "1.8s", duration: "4.5s" },
          { top: "82%", left: "28%", delay: "2.1s", duration: "3.2s" },
          { top: "35%", left: "42%", delay: "0.8s", duration: "4s" },
        ].map((star, i) => (
          <div
            key={i}
            className="absolute h-1 w-1 rounded-full bg-white shadow-[0_0_8px_#3fb5c9] animate-star-twinkle"
            style={{
              top: star.top,
              left: star.left,
              animationDelay: star.delay,
              animationDuration: star.duration,
            }}
          />
        ))}
      </div>

      {/* 3. Top Navigation Bar: Retro OS Dashboard Style */}
      <header
        className={`relative z-30 p-2 sm:px-6 transition-all duration-700 ${
          isLoaded ? "opacity-100 translate-y-0" : "opacity-0 -translate-y-4"
        }`}
      >
        <div className="retro-outset bg-[#E7E2D6] dark:bg-[#1A201E] p-1.5 flex items-center justify-between shadow-xl">
          {/* Brand Header & System Status */}
          <div className="flex items-center gap-3">
            <div className="bg-[#1F4743] text-white px-2 py-1 flex items-center gap-2 text-xs font-bold font-mono tracking-wider">
              <span className="w-4 h-4 bg-white/20 flex items-center justify-center text-[10px] text-white font-mono font-bold">
                C2
              </span>
              <span>CHANDRAYAAN-2</span>
            </div>

            <div className="hidden sm:flex items-center gap-2">
              <span className="retro-inset px-2 py-0.5 text-[10px] font-mono font-bold text-[#143532] dark:text-emerald-400 bg-[#E9E4D8] dark:bg-[#161B19] border-t-[#8B8579] border-l-[#8B8579] border-r-white border-b-white">
                SIH26166 // ISRO SAC
              </span>
              <div className="flex items-center gap-1.5 font-mono text-[10px] text-[#555C58] dark:text-[#8C9893]">
                <span className="h-2 w-2 rounded-full bg-emerald-500 animate-pulse" />
                <span className="hidden md:inline">ORBITAL PIPELINE ACTIVE</span>
              </div>
            </div>
          </div>

          {/* Top-Right Navigation Controls */}
          <nav className="flex items-center gap-2">
            <button
              onClick={onOpenAbout ?? onOpenConsole}
              className="retro-button px-2.5 py-1 text-xs font-mono font-semibold text-[#1E2321] dark:text-[#E7E2D6] hover:bg-[#D9D3C5] dark:hover:bg-white/10"
            >
              Mission Science
            </button>

            <Link
              href="/ingest"
              className="retro-button px-2.5 py-1 text-xs font-mono font-semibold text-[#1E2321] dark:text-[#E7E2D6] hover:bg-[#D9D3C5] dark:hover:bg-white/10 flex items-center gap-1.5"
            >
              <span>⚡</span>
              <span className="hidden sm:inline">Ingest Rasters</span>
            </Link>

            <button
              onClick={onOpenConsole}
              className="hidden sm:inline-flex retro-button px-2.5 py-1 text-xs font-mono font-semibold text-[#1E2321] dark:text-[#E7E2D6] hover:bg-[#D9D3C5] dark:hover:bg-white/10"
            >
              Dataset Vault
            </button>

            {/* Launch Console Action */}
            <button
              onClick={onOpenConsole}
              className="retro-button-primary px-3 py-1 text-xs font-mono font-bold flex items-center gap-1.5 tracking-wider uppercase shadow-md active-pressed"
            >
              <span>Launch Console</span>
              <span>↗</span>
            </button>

            {/* User Profile / Logout */}
            {userName && (
              <div className="flex items-center gap-2 ml-1 pl-2 border-l border-[#8B8579] dark:border-[#2D3835]">
                <span className="retro-inset px-1.5 py-0.5 text-[10px] font-mono font-bold text-[#1E2321] dark:text-[#E7E2D6] bg-white/20">
                  {userName.charAt(0).toUpperCase()}
                </span>
                <button
                  onClick={onLogout}
                  className="retro-button px-2 py-0.5 text-[10px] font-mono text-rose-600 dark:text-rose-400 hover:bg-rose-700 hover:text-white"
                  title="Sign out"
                >
                  Logout
                </button>
              </div>
            )}
          </nav>
        </div>
      </header>

      {/* 4. Centered Content: Dashboard Styled Mission Window HUD */}
      <div
        className={`relative z-20 flex h-[calc(100vh-130px)] w-full items-center justify-center px-4 sm:px-6 transition-all duration-700 delay-100 ${
          isLoaded ? "opacity-100 translate-y-0" : "opacity-0 translate-y-6"
        }`}
      >
        <div className="retro-outset bg-[#E7E2D6]/95 dark:bg-[#1A201E]/95 backdrop-blur-md p-1 shadow-2xl text-[#1E2321] dark:text-[#E7E2D6] w-full max-w-4xl mx-auto">
          {/* Workstation Titlebar */}
          <div className="bg-[#1F4743] px-3 py-1.5 text-white flex items-center justify-between text-xs font-bold font-mono tracking-wider select-none border-b border-[#143532]">
            <div className="flex items-center gap-2">
              <span className="h-2 w-2 rounded-full bg-emerald-400 animate-pulse" />
              <span>TERMINAL // CHANDRAYAAN-2 MULTI-MODAL CROSS-SENSOR REGISTRATION</span>
            </div>
            <div className="flex items-center gap-1">
              <span className="window-ctrl-btn">_</span>
              <span className="window-ctrl-btn">□</span>
              <span className="window-ctrl-btn">✕</span>
            </div>
          </div>

          {/* Sub-Header Status Strip */}
          <div className="px-3 py-1 bg-[#DED8CB] dark:bg-[#141817] border-b border-[#8B8579] dark:border-[#2D3835] flex items-center justify-between text-[11px] font-mono">
            <span className="text-[#555C58] dark:text-[#8C9893]">
              STATUS: <strong className="text-emerald-700 dark:text-emerald-400">READY</strong> · GSD 0.25m ↔ 5.0m ↔ 80m
            </span>
            <span className="retro-inset px-2 py-0.2 text-[10px] bg-white dark:bg-[#0A0D0C] text-[#1E2321] dark:text-[#E7E2D6]">
              SYS_REV: 2026.09.27
            </span>
          </div>

          {/* Window Body */}
          <div className="p-4 sm:p-6 space-y-5">
            {/* Eyebrow Chip */}
            <div className="inline-flex items-center gap-2 retro-inset px-2.5 py-1 text-[11px] font-mono bg-white dark:bg-[#0A0D0C] text-[#1E2321] dark:text-[#E7E2D6]">
              <span className="h-2 w-2 rounded-full bg-emerald-500 animate-pulse" />
              <span className="font-bold text-[#143532] dark:text-emerald-400">ISRO SAC PROBLEM STATEMENT 26166</span>
              <span className="text-[#8B8579]">|</span>
              <span className="text-[#555C58] dark:text-[#8C9893]">SUN ANGLE &amp; SCALE INVARIANT MATCHING</span>
            </div>

            {/* Headline */}
            <div>
              <h1 className="font-black tracking-tight leading-none text-4xl sm:text-5xl md:text-6xl text-[#1E2321] dark:text-white uppercase font-sans">
                EXPLORE <span className="text-[#1F4743] dark:text-teal">THE MOON</span>
              </h1>
              <p className="mt-2 text-xs sm:text-sm text-[#4A524E] dark:text-[#A8B2AD] leading-relaxed font-sans max-w-2xl">
                Deterministic sub-pixel geometric co-registration between Chandrayaan-2's{" "}
                <strong className="text-[#1E2321] dark:text-white">OHRC (0.25m)</strong>,{" "}
                <strong className="text-[#1E2321] dark:text-white">TMC-2 (5.0m)</strong>, and{" "}
                <strong className="text-[#1E2321] dark:text-white">IIRS (80m)</strong> payloads.
                Resolves extreme shadow inversion and ~20× scale disparity with Phase Congruency &amp; CFOG.
              </p>
            </div>

            {/* 4 Telemetry HUD Cards styled like Dashboard Insets */}
            <div className="grid grid-cols-2 sm:grid-cols-4 gap-2 font-mono">
              <div className="retro-inset p-2.5 bg-white dark:bg-[#0A0D0C]">
                <span className="text-[10px] text-[#555C58] dark:text-[#8C9893] block">Optical GSD</span>
                <span className="text-sm font-bold text-[#1F4743] dark:text-teal block mt-0.5">0.25 m/px</span>
                <span className="text-[9px] text-[#8B8579] block">OHRC Narrow</span>
              </div>
              <div className="retro-inset p-2.5 bg-white dark:bg-[#0A0D0C]">
                <span className="text-[10px] text-[#555C58] dark:text-[#8C9893] block">Sub-Pixel Fit</span>
                <span className="text-sm font-bold text-emerald-700 dark:text-emerald-400 block mt-0.5">&lt; 0.28 px</span>
                <span className="text-[9px] text-[#8B8579] block">Verified RMSE</span>
              </div>
              <div className="retro-inset p-2.5 bg-white dark:bg-[#0A0D0C]">
                <span className="text-[10px] text-[#555C58] dark:text-[#8C9893] block">Scale Invariant</span>
                <span className="text-sm font-bold text-[#28557E] dark:text-sky-400 block mt-0.5">20× – 320×</span>
                <span className="text-[9px] text-[#8B8579] block">Multi-Sensor</span>
              </div>
              <div className="retro-inset p-2.5 bg-white dark:bg-[#0A0D0C]">
                <span className="text-[10px] text-[#555C58] dark:text-[#8C9893] block">Planetary CRS</span>
                <span className="text-sm font-bold text-amber-700 dark:text-amber-400 block mt-0.5">Moon2000</span>
                <span className="text-[9px] text-[#8B8579] block">IAU Selenodesy</span>
              </div>
            </div>

            {/* Dashboard Buttons */}
            <div className="pt-2 flex flex-wrap items-center gap-3">
              <button
                onClick={onOpenConsole}
                className="retro-button-primary px-5 py-2.5 text-xs font-mono font-bold tracking-wider uppercase flex items-center gap-2 shadow-md active-pressed"
                title="Launch Planetary Registration Dashboard"
              >
                <span>Mission Dashboard</span>
                <span className="font-bold">&gt;&gt;</span>
              </button>

              <button
                onClick={onOpenAbout ?? onOpenConsole}
                className="retro-button px-4 py-2.5 text-xs font-mono font-bold text-[#1E2321] dark:text-[#E7E2D6] hover:bg-[#D9D3C5] dark:hover:bg-white/10 flex items-center gap-1.5"
              >
                <span>Science Briefing</span>
                <span>↗</span>
              </button>

              <Link
                href="/ingest"
                className="retro-button px-4 py-2.5 text-xs font-mono font-bold text-[#1E2321] dark:text-[#E7E2D6] hover:bg-[#D9D3C5] dark:hover:bg-white/10 flex items-center gap-1.5"
              >
                <span>⚡</span>
                <span>Ingest Rasters</span>
              </Link>
            </div>
          </div>
        </div>
      </div>

      {/* 5. Bottom Workstation Footer Status Bar */}
      <footer
        className={`absolute bottom-3 left-0 right-0 z-20 px-4 sm:px-6 transition-all duration-700 delay-200 ${
          isLoaded ? "opacity-100" : "opacity-0"
        }`}
      >
        <div className="retro-outset bg-[#E7E2D6]/95 dark:bg-[#1A201E]/95 backdrop-blur-md px-3 py-1.5 flex flex-col sm:flex-row items-center justify-between gap-2 text-xs font-mono text-[#555C58] dark:text-[#8C9893]">
          <div className="flex items-center gap-2 text-[11px]">
            <span className="font-bold text-[#1E2321] dark:text-[#E7E2D6]">ISRO SAC</span>
            <span>·</span>
            <span>Smart India Hackathon 2024</span>
          </div>

          <div className="retro-inset px-2.5 py-0.5 text-[10px] bg-white dark:bg-[#0A0D0C] text-[#1E2321] dark:text-[#E7E2D6] flex items-center gap-2">
            <span className="h-1.5 w-1.5 rounded-full bg-emerald-500 animate-pulse" />
            <span>FASTAPI RUNTIME: ONLINE · PYTORCH CFOG · THREE.JS GLOBE</span>
          </div>

          <div className="flex items-center gap-3 text-[11px]">
            <button
              onClick={onOpenAbout}
              className="hover:underline text-[#1E2321] dark:text-[#E7E2D6]"
            >
              Algorithm Specs
            </button>
            <span>·</span>
            <button
              onClick={onOpenConsole}
              className="font-bold text-[#28557E] dark:text-teal hover:underline"
            >
              Open Console
            </button>
          </div>
        </div>
      </footer>
    </section>
  );
}
