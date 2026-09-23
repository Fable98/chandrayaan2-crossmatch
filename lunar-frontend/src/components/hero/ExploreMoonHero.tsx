"use client";

import { useState, useEffect } from "react";
import Image from "next/image";
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
    <section className="dark relative h-screen w-screen select-none overflow-hidden bg-[#000000] font-sans text-white">
      {/* 1. Full-Bleed Crescent Moon & Lunar Surface Background with Smooth Parallax */}
      <div
        className="absolute inset-0 z-0 h-[106%] w-[106%] -left-[3%] -top-[3%] transition-transform duration-700 ease-out pointer-events-none"
        style={{
          transform: `translate3d(${mousePos.x * -12}px, ${mousePos.y * -12}px, 0)`,
        }}
      >
        <Image
          src="/lunar_crescent_backdrop.jpg"
          alt="Chandrayaan-2 Lunar Exploration"
          fill
          priority
          sizes="100vw"
          className="object-cover object-center"
        />

        {/* Cinematic Vignettes */}
        <div className="pointer-events-none absolute inset-0 bg-gradient-to-t from-black via-transparent to-black/70" />
        <div className="pointer-events-none absolute inset-0 bg-gradient-to-r from-black/50 via-transparent to-black/70" />
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
          { top: "18%", left: "92%", delay: "1.5s", duration: "3.8s" },
          { top: "60%", left: "82%", delay: "2.4s", duration: "4.2s" },
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

        {/* Ambient Subtle Radar Scan Ring around Moon */}
        <div
          className="absolute left-[18%] top-[45%] -translate-x-1/2 -translate-y-1/2 h-[520px] w-[520px] rounded-full border border-teal/10 opacity-30 pointer-events-none"
          style={{
            transform: `translate3d(${mousePos.x * -6}px, ${mousePos.y * -6}px, 0)`,
          }}
        >
          <div className="absolute inset-0 rounded-full border border-dashed border-teal/15 animate-radar-sweep" />
          <div className="absolute -top-1.5 left-1/2 -translate-x-1/2 h-3 w-3 rounded-full bg-teal shadow-[0_0_12px_#3fb5c9]" />
        </div>
      </div>

      {/* 3. Top Navigation Bar */}
      <header
        className={`relative z-30 flex h-16 items-center justify-between border-b border-white/[0.08] bg-black/30 backdrop-blur-xl px-6 transition-all duration-1000 md:px-12 ${
          isLoaded ? "opacity-100 translate-y-0" : "opacity-0 -translate-y-4"
        }`}
      >
        {/* Brand Logo & Mission Telemetry */}
        <div className="flex items-center gap-3">
          <div className="flex h-9 w-9 items-center justify-center rounded-xl bg-gradient-to-br from-teal to-[#1B2CC1] p-0.5 shadow-[0_0_16px_rgba(63,181,201,0.4)]">
            <div className="flex h-full w-full items-center justify-center rounded-[10px] bg-black/90">
              <span className="font-mono text-xs font-black tracking-tighter text-teal">
                C2
              </span>
            </div>
          </div>
          <div>
            <div className="flex items-center gap-2">
              <span className="font-bold text-xs tracking-wider text-white">
                CHANDRAYAAN-2
              </span>
              <span className="rounded bg-teal/15 border border-teal/30 px-1.5 py-0.2 text-[9px] font-mono font-semibold text-teal uppercase tracking-wider">
                SIH26166
              </span>
            </div>
            <div className="flex items-center gap-1.5">
              <span className="relative flex h-1.5 w-1.5">
                <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-emerald-400 opacity-75" />
                <span className="relative inline-flex rounded-full h-1.5 w-1.5 bg-emerald-500" />
              </span>
              <span className="font-mono text-[10px] text-ink-dim tracking-wide">
                ORBITAL CO-REGISTRATION PIPELINE
              </span>
            </div>
          </div>
        </div>

        {/* Top-Right: Nav Links */}
        <nav className="flex items-center gap-3 md:gap-4">
          <button
            onClick={onOpenAbout ?? onOpenConsole}
            className="rounded-lg px-3 py-1.5 text-xs font-medium tracking-wide text-ink-dim transition-all hover:bg-white/5 hover:text-white"
          >
            Mission Science
          </button>
          <Link
            href="/ingest"
            className="rounded-lg px-3 py-1.5 text-xs font-medium tracking-wide text-ink-dim transition-all hover:bg-white/5 hover:text-white"
          >
            Ingest Rasters
          </Link>
          <button
            onClick={onOpenConsole}
            className="hidden sm:inline-flex rounded-lg px-3 py-1.5 text-xs font-medium tracking-wide text-ink-dim transition-all hover:bg-white/5 hover:text-white"
          >
            Dataset Vault
          </button>

          {/* Launch Console Action */}
          <button
            onClick={onOpenConsole}
            className="group flex items-center gap-2 rounded-full border border-teal/50 bg-gradient-to-r from-teal/20 to-teal/10 px-4 py-1.5 font-mono text-xs font-semibold text-teal shadow-[0_0_20px_rgba(63,181,201,0.25)] backdrop-blur-md transition-all duration-200 hover:border-teal hover:bg-teal/30 hover:scale-105 active:scale-95"
          >
            <span>Launch Console</span>
            <span className="transition-transform duration-200 group-hover:translate-x-0.5">
              ↗
            </span>
          </button>

          {/* User / Logout */}
          {userName && (
            <div className="flex items-center gap-2 ml-2 pl-3 border-l border-white/10">
              <div className="h-7 w-7 rounded-full bg-gradient-to-br from-teal/30 to-teal-dark/30 border border-teal/30 flex items-center justify-center text-xs font-bold text-teal">
                {userName.charAt(0).toUpperCase()}
              </div>
              <button
                onClick={onLogout}
                className="text-xs font-normal text-ink-faint transition-colors hover:text-red-400"
                title="Sign out"
              >
                Logout
              </button>
            </div>
          )}
        </nav>
      </header>

      {/* 4. Main Center/Right Content: EXPLORE THE MOON + Telemetry HUD */}
      <div
        className={`relative z-20 flex h-[calc(100vh-140px)] flex-col justify-center px-6 transition-all duration-1000 delay-200 md:px-12 lg:ml-[36vw] ${
          isLoaded ? "opacity-100 translate-y-0" : "opacity-0 translate-y-6"
        }`}
      >
        <div className="max-w-2xl text-left">
          {/* Eyebrow Tag */}
          <div className="inline-flex items-center gap-2 rounded-full border border-teal/40 bg-teal/10 px-3.5 py-1 text-2xs font-mono font-semibold uppercase tracking-wider text-teal shadow-[0_0_20px_rgba(63,181,201,0.25)] backdrop-blur-sm">
            <span className="h-1.5 w-1.5 rounded-full bg-teal animate-pulse" />
            <span>ISRO Space Applications Centre</span>
            <span className="text-white/30">·</span>
            <span>SIH 26166</span>
          </div>

          {/* Giant Bold Headline with Metallic Gradient */}
          <h1 className="mt-4 font-extrabold uppercase tracking-tight leading-[0.92]">
            <span className="block text-5xl sm:text-6xl md:text-7xl lg:text-[5.5rem] bg-gradient-to-b from-white via-[#f0f0f0] to-[#8f96a0] bg-clip-text text-transparent drop-shadow-[0_4px_30px_rgba(0,0,0,0.9)]">
              EXPLORE
            </span>
            <span className="block text-5xl sm:text-6xl md:text-7xl lg:text-[5.5rem] bg-gradient-to-b from-white via-[#e8e8e8] to-[#717782] bg-clip-text text-transparent drop-shadow-[0_4px_30px_rgba(0,0,0,0.9)] mt-1">
              THE MOON
            </span>
          </h1>

          {/* Technical Problem Statement Description */}
          <div className="mt-6 max-w-xl">
            <p className="text-xs leading-relaxed text-white/85 sm:text-sm sm:leading-relaxed">
              Autonomous sub-pixel geometric co-registration across Chandrayaan-2's{" "}
              <span className="text-teal font-semibold">OHRC (0.25m)</span>,{" "}
              <span className="text-teal font-semibold">TMC-2 (5.0m)</span>, and{" "}
              <span className="text-teal font-semibold">IIRS (80m)</span> payloads. Resolves
              extreme relief displacement and sun-angle disparities with phase-congruent
              CFOG correspondence.
            </p>
          </div>

          {/* Floating Key Metrics HUD Badges */}
          <div className="mt-6 grid grid-cols-2 sm:grid-cols-4 gap-2.5 max-w-xl font-mono">
            <div className="glass-hud glass-hud-hover rounded-xl p-2.5 transition-all duration-300">
              <span className="text-[10px] text-ink-dim block">Optical GSD</span>
              <span className="text-sm font-bold text-teal block mt-0.5">0.25 m/px</span>
              <span className="text-[9px] text-ink-faint block">OHRC Narrow</span>
            </div>
            <div className="glass-hud glass-hud-hover rounded-xl p-2.5 transition-all duration-300">
              <span className="text-[10px] text-ink-dim block">Sub-Pixel Fit</span>
              <span className="text-sm font-bold text-emerald-400 block mt-0.5">&lt; 0.28 px</span>
              <span className="text-[9px] text-ink-faint block">Verified RMSE</span>
            </div>
            <div className="glass-hud glass-hud-hover rounded-xl p-2.5 transition-all duration-300">
              <span className="text-[10px] text-ink-dim block">Scale Invariant</span>
              <span className="text-sm font-bold text-white block mt-0.5">20× – 320×</span>
              <span className="text-[9px] text-ink-faint block">Multi-Sensor</span>
            </div>
            <div className="glass-hud glass-hud-hover rounded-xl p-2.5 transition-all duration-300">
              <span className="text-[10px] text-ink-dim block">Planetary CRS</span>
              <span className="text-sm font-bold text-amber-300 block mt-0.5">Moon2000</span>
              <span className="text-[9px] text-ink-faint block">IAU Selenodesy</span>
            </div>
          </div>

          {/* Action Buttons */}
          <div className="mt-7 flex flex-wrap items-center gap-3">
            <button
              onClick={onOpenConsole}
              className="group inline-flex items-center justify-center gap-3 rounded-full border border-teal/60 bg-teal px-8 py-3.5 text-sm font-bold tracking-wide text-black shadow-[0_0_35px_rgba(63,181,201,0.5)] transition-all duration-300 hover:bg-[#52cde3] hover:shadow-[0_0_50px_rgba(63,181,201,0.8)] hover:scale-105 active:scale-95"
              title="Launch Planetary Registration Dashboard"
            >
              <span>Mission Dashboard</span>
              <span className="transition-transform duration-300 group-hover:translate-x-1 font-bold">
                &gt;&gt;
              </span>
            </button>

            <button
              onClick={onOpenAbout ?? onOpenConsole}
              className="inline-flex items-center justify-center gap-2 rounded-full border border-white/15 bg-white/5 px-6 py-3.5 text-xs font-semibold tracking-wide text-white/90 backdrop-blur-md transition-all duration-200 hover:bg-white/10 hover:border-white/30 hover:text-white"
            >
              <span>Science Briefing</span>
              <span>↗</span>
            </button>

            <Link
              href="/ingest"
              className="inline-flex items-center justify-center gap-2 rounded-full border border-white/10 bg-transparent px-5 py-3.5 text-xs font-semibold tracking-wide text-ink-dim transition-all duration-200 hover:text-white hover:border-white/20"
            >
              <span>Ingest Rasters</span>
            </Link>
          </div>
        </div>
      </div>

      {/* 5. Bottom Modern Aerospace Footer */}
      <footer
        className={`absolute bottom-5 left-0 right-0 z-20 flex flex-col sm:flex-row items-center justify-between gap-3 px-6 text-xs text-ink-faint transition-all duration-1000 delay-700 md:bottom-6 md:px-12 ${
          isLoaded ? "opacity-100" : "opacity-0"
        }`}
      >
        {/* Left: Organization */}
        <div className="flex items-center gap-2 font-mono text-2xs tracking-wide">
          <span className="text-white/60">ISRO SAC</span>
          <span className="text-white/20">·</span>
          <span className="text-teal font-medium">Smart India Hackathon 2024</span>
        </div>

        {/* Center: System Telemetry Status */}
        <div className="flex items-center gap-2 rounded-full border border-white/[0.06] bg-black/40 px-3 py-1 font-mono text-2xs backdrop-blur-md">
          <span className="h-1.5 w-1.5 rounded-full bg-emerald-400" />
          <span className="text-ink-dim">FASTAPI RUNTIME · PYTORCH CFOG · THREE.JS GLOBE</span>
        </div>

        {/* Right: Quick Links */}
        <div className="flex items-center gap-4 text-2xs">
          <button
            onClick={onOpenAbout}
            className="text-ink-dim hover:text-white transition-colors"
          >
            Algorithm Specs
          </button>
          <span className="text-white/20">·</span>
          <button
            onClick={onOpenConsole}
            className="text-teal hover:underline transition-colors font-medium"
          >
            Open Console
          </button>
        </div>
      </footer>
    </section>
  );
}
