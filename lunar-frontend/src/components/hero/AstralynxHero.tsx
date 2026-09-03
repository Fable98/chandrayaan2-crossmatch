"use client";

import { useState, useEffect } from "react";
import Image from "next/image";

interface Props {
  onOpenConsole?: () => void;
}

export default function AstralynxHero({ onOpenConsole }: Props) {
  const [mousePos, setMousePos] = useState({ x: 0, y: 0 });
  const [isLoaded, setIsLoaded] = useState(false);
  const [mobileMenuOpen, setMobileMenuOpen] = useState(false);

  useEffect(() => {
    setIsLoaded(true);

    const handleMouseMove = (e: MouseEvent) => {
      // Normalized mouse offset from center (-1 to 1)
      const x = (e.clientX / window.innerWidth - 0.5) * 2;
      const y = (e.clientY / window.innerHeight - 0.5) * 2;
      setMousePos({ x, y });
    };

    window.addEventListener("mousemove", handleMouseMove);
    return () => window.removeEventListener("mousemove", handleMouseMove);
  }, []);

  return (
    <section className="relative h-screen w-screen select-none overflow-hidden bg-[#070d14] font-sans text-[#f0f4f5]">
      {/* 1. Full-Bleed Parallax Background with Atmospheric Zoom & Vignette */}
      <div
        className="absolute inset-0 z-0 h-[108%] w-[108%] -left-[4%] -top-[4%] transition-transform duration-700 ease-out"
        style={{
          transform: `translate3d(${mousePos.x * -14}px, ${mousePos.y * -14}px, 0) scale(1.02)`,
        }}
      >
        <Image
          src="/astralynx_portal.jpg"
          alt="Astralynx Celestial Portal"
          fill
          priority
          sizes="100vw"
          className="object-cover object-center animate-slow-pulse"
        />

        {/* Soft Radial Portal Glow & Bloom */}
        <div className="pointer-events-none absolute inset-0 bg-[radial-gradient(ellipse_at_center,rgba(63,181,201,0.18)_0%,rgba(10,21,32,0.4)_55%,rgba(7,13,20,0.85)_100%)]" />

        {/* Outer Edge Vignette */}
        <div className="pointer-events-none absolute inset-0 bg-[radial-gradient(circle_at_center,transparent_45%,rgba(5,9,15,0.75)_90%,rgba(4,7,12,0.95)_100%)]" />
      </div>

      {/* 2. Top Navigation Bar (Transparent, sleek, clean) */}
      <header
        className={`relative z-30 flex items-center justify-between px-6 py-6 transition-all duration-1000 md:px-12 ${
          isLoaded ? "opacity-100 translate-y-0" : "opacity-0 -translate-y-4"
        }`}
      >
        {/* Left: Logo Text */}
        <div className="flex items-center gap-3">
          <a
            href="#"
            className="group flex items-center gap-2.5 text-lg font-medium tracking-wide text-white transition-opacity hover:opacity-90 md:text-xl"
          >
            <span className="font-sans font-semibold tracking-wider">Chandrayaan-2</span>
            <span className="rounded border border-[#3fb5c9]/40 bg-[#3fb5c9]/10 px-1.5 py-0.5 font-mono text-[10px] font-semibold text-[#3fb5c9]">
              SIH26166
            </span>
          </a>
        </div>

        {/* Right: Nav Links (Desktop) */}
        <nav className="hidden items-center gap-8 md:flex">
          <a
            href="#payloads"
            className="text-sm font-normal tracking-wide text-white/80 transition-colors hover:text-white"
          >
            Payloads
          </a>
          <a
            href="#regions"
            className="text-sm font-normal tracking-wide text-white/80 transition-colors hover:text-white"
          >
            8 Validated Regions
          </a>
          <a
            href="#qa"
            className="text-sm font-normal tracking-wide text-white/80 transition-colors hover:text-white"
          >
            Registration QA
          </a>

          {/* Optional Portal Switcher to Chandrayaan-2 Console */}
          {onOpenConsole && (
            <button
              onClick={onOpenConsole}
              className="ml-2 flex items-center gap-1.5 rounded-full border border-white/20 bg-white/5 px-3.5 py-1 font-mono text-xs text-white/90 backdrop-blur-sm transition-all hover:border-[#3fb5c9]/60 hover:bg-[#3fb5c9]/10 hover:text-[#3fb5c9]"
            >
              <span>Launch Console</span>
              <span>↗</span>
            </button>
          )}
        </nav>

        {/* Mobile Hamburger Toggle */}
        <div className="md:hidden">
          <button
            onClick={() => setMobileMenuOpen(!mobileMenuOpen)}
            className="rounded p-1 text-white/80 hover:text-white"
            aria-label="Toggle Navigation"
          >
            <svg
              className="h-6 w-6"
              fill="none"
              stroke="currentColor"
              viewBox="0 0 24 24"
            >
              {mobileMenuOpen ? (
                <path
                  strokeLinecap="round"
                  strokeLinejoin="round"
                  strokeWidth="2"
                  d="M6 18L18 6M6 6l12 12"
                />
              ) : (
                <path
                  strokeLinecap="round"
                  strokeLinejoin="round"
                  strokeWidth="2"
                  d="M4 6h16M4 12h16M4 18h16"
                />
              )}
            </svg>
          </button>
        </div>
      </header>

      {/* Mobile Drawer */}
      {mobileMenuOpen && (
        <div className="relative z-30 flex flex-col gap-4 border-b border-white/10 bg-[#0a1520]/95 px-6 py-4 backdrop-blur-md md:hidden">
          <a href="#payloads" className="text-sm text-white/80 hover:text-white">
            Payloads
          </a>
          <a href="#regions" className="text-sm text-white/80 hover:text-white">
            8 Validated Regions
          </a>
          <a href="#qa" className="text-sm text-white/80 hover:text-white">
            Registration QA
          </a>
          {onOpenConsole && (
            <button
              onClick={onOpenConsole}
              className="flex items-center justify-between rounded-lg border border-[#3fb5c9]/30 bg-[#3fb5c9]/10 px-3 py-2 text-xs text-[#3fb5c9]"
            >
              <span>Open Lunar Console</span>
              <span>↗</span>
            </button>
          )}
        </div>
      )}

      {/* 3. Main Center Headline Block (Inside the Circle, Upper-Middle Area) */}
      <div
        className={`relative z-20 mx-auto flex max-w-3xl flex-col items-center px-6 pt-4 text-center transition-all duration-1000 delay-200 md:pt-10 lg:pt-14 ${
          isLoaded ? "opacity-100 translate-y-0" : "opacity-0 translate-y-6"
        }`}
      >
        {/* Large Bold Headline (2 Lines, White, Centered, Clean Modern Sans-Serif) */}
        <h1 className="text-3xl font-bold tracking-tight text-white drop-shadow-[0_4px_24px_rgba(0,0,0,0.85)] sm:text-4xl md:text-5xl lg:text-[3.25rem] lg:leading-[1.15]">
          Precision Alignment Across <br className="hidden sm:inline" />
          The Lunar Frontier
        </h1>

        {/* Smaller Subtext (2 lines, muted white/gray) */}
        <p className="mt-3.5 max-w-xl text-xs font-normal leading-relaxed text-white/85 drop-shadow-[0_2px_12px_rgba(0,0,0,0.95)] sm:text-sm md:text-[0.9375rem] md:leading-normal">
          Illumination-robust and sub-pixel deep correspondence across Chandrayaan-2's{" "}
          <br className="hidden sm:inline" />
          OHRC, TMC-2, and IIRS payloads, achieving precision co-registration.
        </p>

        {/* Pill-Shaped CTA Button with Teal/Cyan Background */}
        <div className="mt-5">
          <a
            href="#explore"
            onClick={(e) => {
              if (onOpenConsole) {
                e.preventDefault();
                onOpenConsole();
              }
            }}
            className="group inline-flex items-center gap-2 rounded-full bg-[#3fb5c9] px-6 py-2.5 text-xs font-medium text-[#07131b] shadow-[0_0_28px_rgba(63,181,201,0.45)] transition-all duration-300 hover:scale-105 hover:bg-[#52cde3] hover:shadow-[0_0_36px_rgba(63,181,201,0.7)] active:scale-95 sm:text-sm sm:px-7 sm:py-2.5"
          >
            <span>Launch Registration Console</span>
            <span className="transition-transform duration-200 group-hover:translate-x-0.5">
              ↗
            </span>
          </a>
        </div>
      </div>

      {/* 4. Bottom Content Blocks (Left & Right) */}
      <div
        className={`absolute bottom-6 left-0 right-0 z-20 flex flex-col justify-between gap-6 px-6 transition-all duration-1000 delay-500 md:bottom-10 md:flex-row md:items-end md:px-12 ${
          isLoaded ? "opacity-100 translate-y-0" : "opacity-0 translate-y-6"
        }`}
      >
        {/* Bottom-Left Content Block */}
        <div className="max-w-xl text-left">
          {/* Small eyebrow label with bullet */}
          <p className="mb-2 flex items-center gap-1.5 text-xs font-semibold tracking-wider text-[#3fb5c9] uppercase">
            <span>•</span>
            <span>The Edge of Resolution</span>
          </p>

          {/* Large heading below it (bold white sans-serif) */}
          <h2 className="text-xl font-bold tracking-tight text-white drop-shadow-[0_2px_12px_rgba(0,0,0,0.85)] sm:text-2xl md:text-3xl">
            Every Crater Unfolds Precision,
          </h2>

          {/* Second line in elegant italic serif font, teal-tinted */}
          <p className="mt-1 font-serif text-lg italic tracking-normal text-[#8be2ef] drop-shadow-[0_2px_12px_rgba(0,0,0,0.85)] sm:text-xl md:text-2xl">
            Harmonizing Optical, Stereo, and Hyperspectral Vision.
          </p>
        </div>

        {/* Bottom-Right Content Block */}
        <div className="flex max-w-sm flex-col items-start text-left md:items-end md:text-right">
          {/* Small paragraph (right-aligned, muted white/gray, 2-3 lines) */}
          <p className="text-xs leading-relaxed text-white/80 drop-shadow-[0_2px_10px_rgba(0,0,0,0.9)] sm:text-[0.8125rem]">
            Powered by deep invariant feature matching and robust RANSAC filtering, our
            pipeline resolves 160° solar illumination shifts into sub-pixel correspondences.
          </p>

          {/* Text link with arrow */}
          <a
            href="#how-it-works"
            onClick={(e) => {
              if (onOpenConsole) {
                e.preventDefault();
                onOpenConsole();
              }
            }}
            className="group mt-2.5 inline-flex items-center gap-1.5 text-xs font-medium text-[#f0f4f5] transition-colors hover:text-[#3fb5c9]"
          >
            <span>Inspect 8 Regions &amp; QA</span>
            <span className="transition-transform duration-200 group-hover:translate-x-1">
              →
            </span>
          </a>
        </div>
      </div>
    </section>
  );
}
