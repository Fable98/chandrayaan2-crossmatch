"use client";

import { useState, useEffect } from "react";
import Image from "next/image";

interface Props {
  onOpenConsole?: () => void;
  onOpenAbout?: () => void;
  onLogout?: () => void;
  userName?: string;
}

export default function ExploreMoonHero({ onOpenConsole, onOpenAbout, onLogout, userName }: Props) {
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
    <section className="relative h-screen w-screen select-none overflow-hidden bg-[#000000] font-sans text-white">
      {/* 1. Full-Bleed Crescent Moon & Lunar Surface Background with Subtle Parallax */}
      <div
        className="absolute inset-0 z-0 h-[106%] w-[106%] -left-[3%] -top-[3%] transition-transform duration-700 ease-out"
        style={{
          transform: `translate3d(${mousePos.x * -10}px, ${mousePos.y * -10}px, 0)`,
        }}
      >
        <Image
          src="/lunar_crescent_backdrop.jpg"
          alt="Lunar Crescent Exploration"
          fill
          priority
          sizes="100vw"
          className="object-cover object-center"
        />

        {/* Subtle Radial Vignette for Pure Contrast */}
        <div className="pointer-events-none absolute inset-0 bg-radial-gradient from-transparent via-transparent to-black/60" />
      </div>

      {/* 2. Top Navigation Bar */}
      {/* 2. Top Navigation Bar (Shared Height, Padding & Alignment) */}
      <header
        className={`relative z-30 flex h-16 items-center justify-between border-b border-white/5 px-6 transition-all duration-1000 md:px-12 ${
          isLoaded ? "opacity-100 translate-y-0" : "opacity-0 -translate-y-4"
        }`}
      >
        <div />

        {/* Top-Right: Nav Links */}
        <nav className="flex items-center gap-4 md:gap-5">
          <button
            onClick={onOpenAbout ?? onOpenConsole}
            className="text-xs font-normal tracking-wide text-ink-dim transition-colors hover:text-white"
          >
            About
          </button>
          <button
            onClick={onOpenConsole}
            className="rounded-full bg-teal px-4 py-1.5 text-xs font-semibold text-black shadow-[0_0_20px_rgba(63,181,201,0.4)] transition-all duration-200 hover:bg-[#52cde3] hover:scale-105 active:scale-95"
          >
            Explore
          </button>
          <button
            onClick={onOpenConsole}
            className="text-xs font-normal tracking-wide text-ink-dim transition-colors hover:text-white"
          >
            Gallery
          </button>
          <button
            onClick={onOpenConsole}
            className="flex items-center gap-1.5 rounded-full border border-teal/40 bg-teal/10 px-3.5 py-1 font-mono text-xs text-teal backdrop-blur-sm transition-all duration-200 hover:bg-teal/20"
          >
            <span>Console</span>
            <span>↗</span>
          </button>

          {/* User / Logout */}
          {userName && (
            <div className="flex items-center gap-2 ml-2">
              <div className="h-7 w-7 rounded-full bg-gradient-to-br from-teal/30 to-teal-dark/30 border border-teal/20 flex items-center justify-center text-xs font-bold text-teal">
                {userName.charAt(0).toUpperCase()}
              </div>
              <button
                onClick={onLogout}
                className="text-xs font-normal tracking-wide text-ink-faint transition-colors hover:text-red-400"
                title="Sign out"
              >
                Logout
              </button>
            </div>
          )}
        </nav>
      </header>

      {/* 3. Main Center/Right Content: EXPLORE THE MOON + Descriptions */}
      <div
        className={`relative z-20 flex h-[calc(100vh-160px)] flex-col justify-center px-6 transition-all duration-1000 delay-200 md:px-12 lg:ml-[38vw] ${
          isLoaded ? "opacity-100 translate-y-0" : "opacity-0 translate-y-6"
        }`}
      >
        {/* Giant Bold Headline with Metallic Gradient */}
        <div className="max-w-2xl text-left">
          <h1 className="font-extrabold uppercase tracking-tight leading-[0.92]">
            <span className="block text-5xl sm:text-6xl md:text-7xl lg:text-[5.75rem] bg-gradient-to-b from-white via-[#f0f0f0] to-[#9aa0a8] bg-clip-text text-transparent drop-shadow-[0_4px_24px_rgba(0,0,0,0.9)]">
              EXPLORE
            </span>
            <span className="block text-5xl sm:text-6xl md:text-7xl lg:text-[5.75rem] bg-gradient-to-b from-white via-[#e8e8e8] to-[#7a808a] bg-clip-text text-transparent drop-shadow-[0_4px_24px_rgba(0,0,0,0.9)] mt-1">
              THE MOON
            </span>
          </h1>

          {/* Description Block */}
          <div className="mt-8 max-w-xl">
            <p className="text-xs leading-relaxed text-white/80 md:text-sm md:leading-normal">
              Moon is an astronomical body that orbits planet Earth and it's Earth's
              only permanent natural satellite. Cross-matched across Chandrayaan-2's
              OHRC, TMC-2 &amp; IIRS payloads with sub-pixel accuracy.
            </p>
          </div>

          {/* Action Button: Dashboard >> */}
          <div className="mt-8">
            <button
              onClick={onOpenConsole}
              className="group inline-flex items-center justify-center gap-3 rounded-full border border-teal/50 bg-teal px-8 py-3 text-sm font-bold tracking-wide text-black shadow-[0_0_30px_rgba(63,181,201,0.5)] transition-all duration-200 hover:bg-[#52cde3] hover:shadow-[0_0_40px_rgba(63,181,201,0.7)] hover:scale-105 active:scale-95"
              title="Launch Planetary Registration Dashboard"
            >
              <span>Dashboard</span>
              <span className="transition-transform duration-200 group-hover:translate-x-1 font-bold">
                &gt;&gt;
              </span>
            </button>
          </div>
        </div>
      </div>

      {/* 5. Bottom Footer (Credits & Social Icons) */}
      <footer
        className={`absolute bottom-6 left-0 right-0 z-20 flex items-center justify-between px-6 text-xs text-ink-faint transition-all duration-1000 delay-700 md:bottom-8 md:px-12 ${
          isLoaded ? "opacity-100" : "opacity-0"
        }`}
      >
        {/* Center Credits */}
        <div className="absolute left-1/2 -translate-x-1/2 text-center">
          <span className="font-mono text-2xs tracking-wide">
            created for <span className="text-teal font-semibold">ISRO · SIH26166</span>
            <span className="mx-2 text-white/20">·</span>
            <span className="text-ink-dim">Chandrayaan-2 Crossmatch</span>
          </span>
        </div>

        {/* Right Social Icons */}
        <div className="ml-auto flex items-center gap-5">
          {/* YouTube */}
          <a
            href="https://youtube.com"
            target="_blank"
            rel="noreferrer"
            className="text-white/70 transition-colors hover:text-white"
            aria-label="YouTube"
          >
            <svg className="h-4 w-4 fill-current" viewBox="0 0 24 24">
              <path d="M23.498 6.186a3.016 3.016 0 0 0-2.122-2.136C19.505 3.545 12 3.545 12 3.545s-7.505 0-9.377.505A3.017 3.017 0 0 0 .502 6.186C0 8.07 0 12 0 12s0 3.93.502 5.814a3.016 3.016 0 0 0 2.122 2.136c1.871.505 9.376.505 9.376.505s7.505 0 9.377-.505a3.015 3.015 0 0 0 2.122-2.136C24 15.93 24 12 24 12s0-3.93-.502-5.814zM9.545 15.568V8.432L15.818 12l-6.273 3.568z" />
            </svg>
          </a>

          {/* Instagram */}
          <a
            href="https://instagram.com"
            target="_blank"
            rel="noreferrer"
            className="text-white/70 transition-colors hover:text-white"
            aria-label="Instagram"
          >
            <svg className="h-4 w-4 fill-current" viewBox="0 0 24 24">
              <path d="M12 2.163c3.204 0 3.584.012 4.85.07 3.252.148 4.771 1.691 4.919 4.919.058 1.265.069 1.645.069 4.849 0 3.205-.012 3.584-.069 4.849-.149 3.225-1.664 4.771-4.919 4.919-1.266.058-1.644.07-4.85.07-3.204 0-3.584-.012-4.849-.07-3.26-.149-4.771-1.699-4.919-4.92-.058-1.265-.07-1.644-.07-4.849 0-3.204.013-3.583.07-4.849.149-3.227 1.664-4.771 4.919-4.919 1.266-.057 1.645-.069 4.849-.069zm0-2.163c-3.259 0-3.667.014-4.947.072-4.358.2-6.78 2.618-6.98 6.98-.059 1.281-.073 1.689-.073 4.948 0 3.259.014 3.668.072 4.948.2 4.358 2.618 6.78 6.98 6.98 1.281.058 1.689.072 4.948.072 3.259 0 3.668-.014 4.948-.072 4.354-.2 6.782-2.618 6.979-6.98.059-1.28.073-1.689.073-4.948 0-3.259-.014-3.667-.072-4.947-.196-4.354-2.617-6.78-6.979-6.98-1.281-.059-1.69-.073-4.949-.073zm0 5.838c-3.403 0-6.162 2.759-6.162 6.162s2.759 6.163 6.162 6.163 6.162-2.759 6.162-6.163c0-3.403-2.759-6.162-6.162-6.162zm0 10.162c-2.209 0-4-1.79-4-4 0-2.209 1.791-4 4-4s4 1.791 4 4c0 2.21-1.791 4-4 4zm6.406-11.845c-.796 0-1.441.645-1.441 1.44s.645 1.44 1.441 1.44c.795 0 1.439-.645 1.439-1.44s-.644-1.44-1.439-1.44z" />
            </svg>
          </a>

          {/* Pinterest */}
          <a
            href="https://pinterest.com"
            target="_blank"
            rel="noreferrer"
            className="text-white/70 transition-colors hover:text-white"
            aria-label="Pinterest"
          >
            <svg className="h-4 w-4 fill-current" viewBox="0 0 24 24">
              <path d="M12 0C5.373 0 0 5.372 0 12c0 5.084 3.163 9.426 7.627 11.174-.105-.949-.2-2.405.042-3.441.218-.937 1.407-5.965 1.407-5.965s-.359-.719-.359-1.782c0-1.668.967-2.914 2.171-2.914 1.023 0 1.518.769 1.518 1.69 0 1.029-.655 2.568-.994 3.995-.283 1.194.599 2.169 1.777 2.169 2.133 0 3.772-2.249 3.772-5.495 0-2.873-2.064-4.882-5.012-4.882-3.414 0-5.418 2.561-5.418 5.207 0 1.031.397 2.138.893 2.738.098.119.112.224.083.345-.09.375-.291 1.199-.334 1.357-.053.225-.174.271-.401.165-1.495-.69-2.433-2.878-2.433-4.646 0-3.776 2.748-7.252 7.92-7.252 4.158 0 7.392 2.967 7.392 6.923 0 4.135-2.607 7.462-6.233 7.462-1.214 0-2.354-.629-2.758-1.379l-.749 2.848c-.269 1.045-1.004 2.352-1.498 3.146 1.123.345 2.306.535 3.55.535 6.627 0 12-5.373 12-12 0-6.628-5.373-12-12-12z" />
            </svg>
          </a>
        </div>
      </footer>
    </section>
  );
}
