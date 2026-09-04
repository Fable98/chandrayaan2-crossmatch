"use client";

import React, { useEffect, useState, useRef } from "react";
import { useSearchParams } from "next/navigation";
import { api, ApiError, imageUrl } from "@/lib/api";
import { footprintSizeKm } from "@/lib/geo";
import type { TripletSummary, MatchPoint, IIRSOverlay, MatchMetrics } from "@/lib/types";
import MapPanel from "./DynamicMapPanel";
import LinkedCursorPanel from "./LinkedCursorPanel";
import DossierModal from "./archive/DossierModal";
import VaultModal, { PayloadFilter } from "./archive/VaultModal";
import TheoryModal from "./archive/TheoryModal";
import InfoModal, { InfoModalContent } from "./archive/InfoModal";
import RegistrationLauncher from "./RegistrationLauncher";

type View = "registration" | "linked-cursor" | "map";

interface Props {
  onBackToHero?: () => void;
}

export default function Console({ onBackToHero }: Props = {}) {
  const searchParams = useSearchParams();
  const subviewParam = searchParams.get("subview") as View | null;
  const modalParam = searchParams.get("modal");

  const [triplets, setTriplets] = useState<TripletSummary[]>([]);
  const [selectedId, setSelectedId] = useState<string | null>(null);
  const [detail, setDetail] = useState<TripletSummary | null>(null);
  const [matches, setMatches] = useState<MatchPoint[]>([]);
  const [metrics, setMetrics] = useState<MatchMetrics | null>(null);
  const [iirsOverlay, setIirsOverlay] = useState<IIRSOverlay | null>(null);
  const [view, setView] = useState<View>(
    subviewParam === "linked-cursor" || subviewParam === "map" ? subviewParam : "registration"
  );
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);
  const [searchQuery, setSearchQuery] = useState("");

  // Modals & Interactivity State
  const [activeDossierTriplet, setActiveDossierTriplet] = useState<TripletSummary | null>(null);
  const [activeDossierMetrics, setActiveDossierMetrics] = useState<MatchMetrics | null>(null);
  const [vaultOpen, setVaultOpen] = useState(false);
  const [vaultInitialFilter, setVaultInitialFilter] = useState<PayloadFilter>("all");
  const [theoryModalOpen, setTheoryModalOpen] = useState(false);
  const [infoModalContent, setInfoModalContent] = useState<InfoModalContent | null>(null);
  const [toastMessage, setToastMessage] = useState<string | null>(null);

  const arenaRef = useRef<HTMLDivElement>(null);

  // Handle modal query param for demo linkability
  useEffect(() => {
    if (modalParam === "vault") {
      setVaultOpen(true);
    } else if (modalParam === "theory") {
      setTheoryModalOpen(true);
    } else if (modalParam === "dossier" && triplets.length > 0) {
      handleOpenDossierModal(triplets[0]);
    }
  }, [modalParam, triplets]);

  // Load regions once on mount
  useEffect(() => {
    api
      .listTriplets()
      .then((res) => {
        setTriplets(res.triplets);
        if (res.triplets.length > 0) setSelectedId(res.triplets[0].id);
      })
      .catch((err) => setError(describeError(err)))
      .finally(() => setLoading(false));
  }, []);

  // Load triplet detail, matches, and IIRS overlay
  useEffect(() => {
    if (!selectedId) return;
    setDetail(null);
    setMatches([]);
    setMetrics(null);
    setIirsOverlay(null);

    Promise.all([
      api.getTriplet(selectedId),
      api.getMatches(selectedId),
      api.getIirsOverlay(selectedId).catch(() => null),
    ])
      .then(([d, m, iirs]) => {
        setDetail(d);
        setMatches(m.matches);
        setMetrics(m.metrics ?? null);
        setIirsOverlay(iirs);
      })
      .catch((err) => setError(describeError(err)));
  }, [selectedId]);

  const showToast = (msg: string) => {
    setToastMessage(msg);
    setTimeout(() => {
      setToastMessage(null);
    }, 3200);
  };

  const scrollToArena = () => {
    if (arenaRef.current) {
      arenaRef.current.scrollIntoView({ behavior: "smooth", block: "start" });
    }
  };

  const handleSelectRegionAndScroll = (
    tripletId: string,
    preferredView?: View
  ) => {
    setSelectedId(tripletId);
    if (preferredView) setView(preferredView);
    scrollToArena();
  };

  const handleOpenDossierModal = async (triplet: TripletSummary) => {
    setActiveDossierTriplet(triplet);
    if (triplet.id === selectedId && metrics) {
      setActiveDossierMetrics(metrics);
    } else {
      try {
        const res = await api.getMatches(triplet.id);
        setActiveDossierMetrics(res.metrics ?? null);
      } catch {
        setActiveDossierMetrics(null);
      }
    }
  };

  const filteredTriplets = triplets.filter((t) =>
    t.id.toLowerCase().includes(searchQuery.toLowerCase())
  );

  const currentIndex = triplets.findIndex((t) => t.id === selectedId);

  const handlePrev = () => {
    if (triplets.length === 0) return;
    const prevIdx = (currentIndex - 1 + triplets.length) % triplets.length;
    setSelectedId(triplets[prevIdx].id);
  };

  const handleNext = () => {
    if (triplets.length === 0) return;
    const nextIdx = (currentIndex + 1) % triplets.length;
    setSelectedId(triplets[nextIdx].id);
  };

  const cycleView = () => {
    if (view === "registration") setView("linked-cursor");
    else if (view === "linked-cursor") setView("map");
    else setView("registration");
  };

  const openVaultWithFilter = (filter: PayloadFilter) => {
    setVaultInitialFilter(filter);
    setVaultOpen(true);
  };

  if (error) {
    return (
      <div className="flex h-screen items-center justify-center bg-[#070913] px-6">
        <div className="max-w-md rounded-2xl border border-red-500/30 bg-[#0d1024]/80 p-6 text-center shadow-2xl backdrop-blur-xl">
          <p className="font-mono text-sm font-semibold uppercase tracking-wider text-red-400">
            Archive Connection Failed
          </p>
          <p className="mt-2 text-xs text-slate-400">{error}</p>
          <p className="mt-4 font-mono text-[10px] text-slate-500">
            Confirm FastAPI server is active on port 8000.
          </p>
        </div>
      </div>
    );
  }

  return (
    <div className="relative min-h-screen overflow-hidden bg-[#070913] font-sans text-slate-100 antialiased selection:bg-purple-500/30 selection:text-purple-200">
      {/* Radiant Atmospheric Lighting (Ambient Orbs & Cosmic Gradient) */}
      <div className="pointer-events-none absolute -top-40 left-1/2 -translate-x-1/2 h-[600px] w-[1000px] rounded-full bg-gradient-to-b from-purple-600/25 via-indigo-600/15 to-transparent blur-[160px]" />
      <div className="pointer-events-none absolute top-1/4 -left-48 h-[600px] w-[600px] rounded-full bg-indigo-700/15 blur-[160px]" />
      <div className="pointer-events-none absolute top-1/3 -right-48 h-[600px] w-[600px] rounded-full bg-purple-700/15 blur-[160px]" />

      {/* Toast Notification */}
      {toastMessage && (
        <div className="fixed bottom-6 right-6 z-[9999] flex items-center gap-3 rounded-full border border-purple-500/30 bg-[#0e122b]/90 px-5 py-3 text-sm text-purple-100 shadow-[0_0_30px_rgba(168,85,247,0.35)] backdrop-blur-xl animate-fade-in">
          <span className="h-2 w-2 rounded-full bg-purple-400 shadow-[0_0_8px_#c084fc]" />
          <span>{toastMessage}</span>
        </div>
      )}

      {/* MAIN COCKPIT: Floating Ultra-Glass Window Container */}
      <div className="relative mx-auto max-w-[1600px] p-4 sm:p-6 lg:p-8">
        <div className="relative overflow-hidden rounded-[32px] border border-white/[0.1] bg-[#0b0e22]/70 shadow-[0_30px_90px_rgba(0,0,0,0.85)] backdrop-blur-2xl ring-1 ring-white/[0.05]">
          
          {/* 1. Header Bar: Brand Logo, Pill Search, and Action Launcher */}
          <header className="flex flex-wrap items-center justify-between gap-4 border-b border-white/[0.08] bg-white/[0.02] px-6 py-4 backdrop-blur-md">
            {/* Left: Brand Identity & Hero Link */}
            <div className="flex items-center gap-4">
              {onBackToHero && (
                <button
                  onClick={onBackToHero}
                  className="flex items-center gap-2 rounded-full border border-white/10 bg-white/[0.04] px-3.5 py-1.5 text-xs text-slate-300 transition hover:border-purple-400/40 hover:bg-white/[0.08] hover:text-white focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-purple-400"
                >
                  <span>←</span>
                  <span>Back</span>
                </button>
              )}
              <div className="flex items-center gap-3">
                <div className="flex h-9 w-9 items-center justify-center rounded-2xl bg-gradient-to-tr from-purple-600 to-indigo-500 shadow-[0_0_16px_rgba(147,51,234,0.5)]">
                  <span className="text-sm font-bold text-white">☾</span>
                </div>
                <div>
                  <h2 className="text-sm font-bold tracking-tight text-white flex items-center gap-2">
                    Chandrayaan-2
                    <span className="rounded-full border border-purple-400/30 bg-purple-500/10 px-2 py-0.5 text-[9px] font-semibold text-purple-300">
                      v2.4
                    </span>
                  </h2>
                  <p className="text-[10px] text-slate-400 font-mono">ISRO Multi-Sensor Cross-Match</p>
                </div>
              </div>
            </div>

            {/* Center: Command Pill Search Bar */}
            <div className="relative flex-1 max-w-md hidden sm:block">
              <div className="relative flex items-center rounded-full border border-white/10 bg-white/[0.04] px-4 py-2 transition-all focus-within:border-purple-400/50 focus-within:bg-white/[0.07] focus-within:shadow-[0_0_20px_rgba(168,85,247,0.2)]">
                <svg className="h-4 w-4 text-slate-400 mr-2.5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0z" />
                </svg>
                <input
                  type="text"
                  value={searchQuery}
                  onChange={(e) => setSearchQuery(e.target.value)}
                  placeholder="Search regions, coordinates, sensors..."
                  className="w-full bg-transparent text-xs text-white placeholder-slate-400 focus:outline-none"
                />
                <kbd className="hidden md:inline-flex items-center gap-0.5 rounded border border-white/10 bg-white/[0.06] px-1.5 py-0.5 text-[10px] text-slate-400 font-mono">
                  ⌘K
                </kbd>
              </div>
            </div>

            {/* Right: Actions & User Avatar */}
            <div className="flex items-center gap-3">
              <RegistrationLauncher />
              <button
                onClick={() => openVaultWithFilter("all")}
                className="flex items-center gap-2 rounded-full border border-white/10 bg-white/[0.04] px-4 py-1.5 text-xs text-slate-300 transition hover:border-purple-400/40 hover:bg-white/[0.08] hover:text-white focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-purple-400"
              >
                <span className="h-2 w-2 rounded-full bg-emerald-400 shadow-[0_0_8px_#34d399]" />
                <span>{triplets.length} Regions</span>
              </button>
              <div className="relative flex h-8 w-8 items-center justify-center rounded-full bg-gradient-to-br from-indigo-500 to-purple-600 text-xs font-bold text-white ring-2 ring-purple-400/30">
                <span>IS</span>
                <span className="absolute bottom-0 right-0 h-2 w-2 rounded-full bg-emerald-400 ring-2 ring-[#0b0e22]" />
              </div>
            </div>
          </header>

          {/* 2. Three-Column Workstation Cockpit */}
          <div className="grid grid-cols-1 lg:grid-cols-12 gap-6 p-6 md:p-8">
            
            {/* ======================================================== */}
            {/* LEFT COLUMN: Navigation Tabs, View Switcher & Region List */}
            {/* ======================================================== */}
            <div className="lg:col-span-3 flex flex-col gap-5">
              {/* Primary Navigation / View Modes */}
              <div className="rounded-2xl border border-white/[0.08] bg-white/[0.025] p-3 backdrop-blur-xl shadow-lg">
                <span className="px-3 py-1.5 text-[11px] font-semibold uppercase tracking-wider text-slate-400 block">
                  Workspace Views
                </span>
                <nav className="mt-1 space-y-1">
                  {[
                    { id: "registration", label: "Registration QA", icon: "✦" },
                    { id: "linked-cursor", label: "Linked Cursor", icon: "◎" },
                    { id: "map", label: "Fused Planetary Map", icon: "☵" },
                  ].map((item) => {
                    const active = view === item.id;
                    return (
                      <button
                        key={item.id}
                        onClick={() => { setView(item.id as View); scrollToArena(); }}
                        className={`group flex w-full items-center justify-between rounded-xl px-3.5 py-2.5 text-xs font-medium transition-all ${
                          active
                            ? "bg-gradient-to-r from-purple-600/35 via-indigo-600/25 to-purple-600/10 border border-purple-400/40 text-white shadow-[0_0_20px_rgba(168,85,247,0.25)]"
                            : "text-slate-300 hover:bg-white/[0.05] hover:text-white border border-transparent"
                        }`}
                      >
                        <div className="flex items-center gap-2.5">
                          <span className={active ? "text-purple-300" : "text-slate-400 group-hover:text-white"}>
                            {item.icon}
                          </span>
                          <span>{item.label}</span>
                        </div>
                        {active && (
                          <span className="h-1.5 w-1.5 rounded-full bg-purple-400 shadow-[0_0_6px_#c084fc]" />
                        )}
                      </button>
                    );
                  })}
                </nav>

                <div className="mt-3 pt-3 border-t border-white/[0.06] flex items-center justify-between px-2">
                  <button
                    onClick={() => setTheoryModalOpen(true)}
                    className="text-[11px] text-slate-400 hover:text-purple-300 transition flex items-center gap-1"
                  >
                    <span>📖 Theory &amp; Math</span>
                  </button>
                  <button
                    onClick={() => openVaultWithFilter("all")}
                    className="text-[11px] text-purple-300 hover:text-purple-200 transition"
                  >
                    Vault ↗
                  </button>
                </div>
              </div>

              {/* Region Directory */}
              <div className="flex-1 rounded-2xl border border-white/[0.08] bg-white/[0.025] p-4 backdrop-blur-xl shadow-lg flex flex-col">
                <div className="flex items-center justify-between border-b border-white/[0.06] pb-3">
                  <div>
                    <span className="text-xs font-semibold text-white">Regions</span>
                    <p className="text-[10px] text-slate-400 font-mono">{filteredTriplets.length} datasets available</p>
                  </div>
                  <div className="flex items-center gap-1">
                    <button
                      onClick={handlePrev}
                      className="flex h-7 w-7 items-center justify-center rounded-lg border border-white/10 bg-white/[0.04] text-slate-300 transition hover:bg-white/[0.08] hover:text-white focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-purple-400"
                      title="Previous Region"
                    >
                      ‹
                    </button>
                    <button
                      onClick={handleNext}
                      className="flex h-7 w-7 items-center justify-center rounded-lg border border-white/10 bg-white/[0.04] text-slate-300 transition hover:bg-white/[0.08] hover:text-white focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-purple-400"
                      title="Next Region"
                    >
                      ›
                    </button>
                  </div>
                </div>

                <div className="mt-3 flex-1 space-y-2 overflow-y-auto max-h-[380px] pr-1">
                  {filteredTriplets.map((t, i) => {
                    const active = t.id === selectedId;
                    const { widthKm, heightKm } = footprintSizeKm(t.bounds);
                    return (
                      <button
                        key={t.id}
                        onClick={() => setSelectedId(t.id)}
                        className={`group flex w-full flex-col rounded-xl p-3 text-left transition-all ${
                          active
                            ? "border border-purple-400/40 bg-white/[0.08] shadow-[0_0_20px_rgba(168,85,247,0.18)]"
                            : "border border-white/[0.04] bg-white/[0.02] hover:border-white/10 hover:bg-white/[0.05]"
                        }`}
                      >
                        <div className="flex items-center justify-between">
                          <span className="font-mono text-[10px] text-slate-400">
                            #{String(i + 1).padStart(2, "0")}
                          </span>
                          {t.dem_available && (
                            <span className="rounded-full border border-purple-400/30 bg-purple-500/10 px-2 py-0.5 font-mono text-[9px] text-purple-300">
                              DEM
                            </span>
                          )}
                        </div>
                        <span className={`mt-1 text-xs font-semibold ${active ? "text-white" : "text-slate-200"}`}>
                          {t.id}
                        </span>
                        <span className="mt-0.5 text-[10px] text-slate-400">
                          {widthKm.toFixed(1)} × {heightKm.toFixed(1)} km
                        </span>
                      </button>
                    );
                  })}
                </div>
              </div>
            </div>

            {/* ======================================================== */}
            {/* CENTER COLUMN: Central Inspection Canvas & 4 Action Cards */}
            {/* ======================================================== */}
            <div ref={arenaRef} className="lg:col-span-6 flex flex-col gap-6 scroll-mt-20">
              {/* Center Headline */}
              <div className="text-center pt-2 pb-1">
                <h1 className="text-2xl sm:text-3xl font-bold tracking-tight text-white">
                  What region would you like to inspect today?
                </h1>
                <p className="mt-1.5 text-xs text-slate-400">
                  Select a region or load dynamic imagery to verify sub-pixel multimodality.
                </p>
              </div>

              {/* Central Viewer Stage */}
              <div className="relative overflow-hidden rounded-2xl border border-white/[0.1] bg-black/40 backdrop-blur-2xl p-5 shadow-2xl flex flex-col min-h-[460px]">
                {/* Viewport Top Bar */}
                <div className="mb-4 flex items-center justify-between">
                  <div className="flex items-center gap-2">
                    <span className="h-2 w-2 rounded-full bg-purple-400 shadow-[0_0_8px_#c084fc]" />
                    <span className="text-xs font-semibold text-white">
                      {detail?.id ?? "No Region Selected"}
                    </span>
                    <span className="rounded-full border border-white/10 bg-white/[0.04] px-2 py-0.5 text-[10px] text-slate-400 font-mono">
                      {view.replace("-", " ").toUpperCase()}
                    </span>
                  </div>

                  <div className="flex items-center gap-2">
                    {detail && (
                      <button
                        onClick={() => handleOpenDossierModal(detail)}
                        className="rounded-full border border-purple-400/30 bg-purple-500/10 px-3 py-1 text-xs text-purple-200 transition hover:bg-purple-500/20"
                      >
                        Report ↗
                      </button>
                    )}
                    <button
                      onClick={cycleView}
                      className="rounded-full border border-white/10 bg-white/[0.04] px-3 py-1 text-xs text-slate-300 transition hover:bg-white/[0.08] hover:text-white"
                      title="Switch View"
                    >
                      Cycle View →
                    </button>
                  </div>
                </div>

                {/* Main View Area */}
                <div className="relative flex-1 overflow-hidden">
                  {loading && (
                    <div className="flex h-full items-center justify-center text-sm text-slate-400">
                      Loading regions…
                    </div>
                  )}
                  {!loading && !detail && (
                    <div className="flex h-full items-center justify-center text-sm text-slate-400">
                      No region selected.
                    </div>
                  )}

                  {/* Registration QA View */}
                  {detail && view === "registration" && (
                    <div className="flex h-full flex-col justify-between">
                      <div className="grid grid-cols-3 gap-3.5">
                        {[
                          { src: `/images/registered/${detail.id}/registered_ohrc.png`, fallback: `/images/ohrc/${detail.id}`, label: "Warped OHRC" },
                          { src: `/images/registered/${detail.id}/blend_overlay.png`, fallback: `/images/tmc/${detail.id}`, label: "Blend 50%" },
                          { src: `/images/registered/${detail.id}/checkerboard_qa.png`, fallback: `/images/tmc/${detail.id}`, label: "Checkerboard" },
                        ].map((img, idx) => (
                          <div key={idx} className="flex flex-col gap-2">
                            <div className="relative aspect-square overflow-hidden rounded-xl border border-white/[0.08] bg-black shadow-inner">
                              {/* eslint-disable-next-line @next/next/no-img-element */}
                              <img
                                src={imageUrl(img.src)}
                                alt={img.label}
                                className="h-full w-full object-cover"
                                onError={(e) => {
                                  (e.currentTarget as HTMLImageElement).src = imageUrl(img.fallback);
                                }}
                              />
                            </div>
                            <span className="text-[11px] font-medium text-slate-300 text-center">{img.label}</span>
                          </div>
                        ))}
                      </div>

                      <div className="mt-4 flex flex-wrap items-center justify-between rounded-xl border border-white/[0.08] bg-white/[0.03] p-3.5 text-xs">
                        <div className="flex items-center gap-2">
                          <span className="h-2 w-2 rounded-full bg-emerald-400 shadow-[0_0_6px_#34d399]" />
                          <span className="text-slate-300">
                            Status: <strong className="text-white font-semibold">{metrics?.sub_pixel_accurate ? "Sub-Pixel Verified (<0.5 px)" : "Standard Match"}</strong>
                          </span>
                        </div>
                        <span className="text-slate-400">
                          Inliers: <strong className="text-purple-300 font-mono">{metrics?.num_inliers ?? 0} matches</strong>
                        </span>
                      </div>
                    </div>
                  )}

                  {/* Linked Cursor View */}
                  {detail && view === "linked-cursor" && (
                    <div className="h-full">
                      <LinkedCursorPanel tripletId={detail.id} points={matches} />
                    </div>
                  )}

                  {/* Map View */}
                  {detail && view === "map" && (
                    <div className="h-full min-h-[380px] rounded-xl overflow-hidden border border-white/[0.08]">
                      <MapPanel triplet={detail} iirsOverlay={iirsOverlay} />
                    </div>
                  )}
                </div>
              </div>

              {/* 4 Quick-Action Frosted Glass Cards (Matches reference UI exactly!) */}
              <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
                {[
                  {
                    title: "Warped OHRC",
                    desc: "0.25m Sub-pixel",
                    icon: "🎯",
                    action: () => setView("registration"),
                  },
                  {
                    title: "Blend 50%",
                    desc: "Co-Registration",
                    icon: "⚖️",
                    action: () => setView("registration"),
                  },
                  {
                    title: "Checkerboard",
                    desc: "Continuity QA",
                    icon: "▦",
                    action: () => setView("registration"),
                  },
                  {
                    title: "Linked Cursor",
                    desc: "Point Inspection",
                    icon: "◎",
                    action: () => setView("linked-cursor"),
                  },
                ].map((card, i) => (
                  <button
                    key={i}
                    onClick={card.action}
                    className="group flex flex-col justify-between rounded-2xl border border-white/[0.07] bg-white/[0.03] p-3.5 text-left backdrop-blur-xl transition-all hover:border-purple-400/40 hover:bg-white/[0.07] hover:shadow-[0_8px_25px_rgba(168,85,247,0.15)]"
                  >
                    <span className="text-lg">{card.icon}</span>
                    <div className="mt-3">
                      <h4 className="text-xs font-semibold text-white group-hover:text-purple-200 transition">
                        {card.title}
                      </h4>
                      <p className="mt-0.5 text-[10px] text-slate-400">
                        {card.desc}
                      </p>
                    </div>
                  </button>
                ))}
              </div>
            </div>

            {/* ======================================================== */}
            {/* RIGHT COLUMN: Pipeline Model, Capabilities & Telemetry    */}
            {/* ======================================================== */}
            <div className="lg:col-span-3 flex flex-col gap-5">
              
              {/* 1. Model / Engine Card (Matches reference GPT-4o card) */}
              <div className="relative overflow-hidden rounded-2xl border border-white/[0.08] bg-gradient-to-b from-purple-900/20 via-white/[0.03] to-white/[0.01] p-4 backdrop-blur-xl shadow-lg">
                <div className="flex items-center justify-between">
                  <span className="text-xs font-bold text-white flex items-center gap-1.5">
                    LoFTR Engine ✦
                  </span>
                  <span className="rounded-full border border-purple-400/30 bg-purple-500/20 px-2 py-0.5 text-[9px] font-semibold text-purple-200">
                    Active
                  </span>
                </div>
                <p className="mt-1.5 text-[11px] text-slate-300 leading-relaxed">
                  Sub-pixel planetary cross-match model with phase correlation refinement.
                </p>

                {/* 3D Glass Layer Visual Graphic */}
                <div className="relative mt-3 h-20 w-full overflow-hidden rounded-xl border border-white/10 bg-gradient-to-tr from-purple-950/60 via-indigo-950/40 to-slate-900/50 p-2 flex items-center justify-center">
                  <div className="relative flex items-center justify-center">
                    <div className="h-10 w-10 rounded-full bg-purple-500/30 blur-sm shadow-[0_0_20px_#a855f7]" />
                    <div className="absolute h-8 w-8 rounded-xl bg-gradient-to-tr from-purple-500 to-indigo-400 rotate-12 opacity-80 shadow-md" />
                    <div className="absolute h-7 w-7 rounded-xl bg-gradient-to-tr from-indigo-400 to-cyan-300 -rotate-6 opacity-75 shadow-md" />
                  </div>
                </div>
              </div>

              {/* 2. Sensor Capabilities Checklist (Matches reference Capabilities card) */}
              <div className="rounded-2xl border border-white/[0.08] bg-white/[0.025] p-4 backdrop-blur-xl shadow-lg">
                <span className="text-[11px] font-semibold uppercase tracking-wider text-slate-400 block mb-3">
                  Sensor Capabilities
                </span>
                <div className="space-y-2.5 text-xs">
                  <div className="flex items-center justify-between text-slate-300">
                    <span className="flex items-center gap-2">
                      <span className="text-purple-400">📷</span>
                      <span>OHRC 0.25m/px</span>
                    </span>
                    <span className="text-emerald-400">✓</span>
                  </div>
                  <div className="flex items-center justify-between text-slate-300">
                    <span className="flex items-center gap-2">
                      <span className="text-indigo-400">🔭</span>
                      <span>TMC-2 Stereo 4.0m</span>
                    </span>
                    <span className="text-emerald-400">✓</span>
                  </div>
                  <div className="flex items-center justify-between text-slate-300">
                    <span className="flex items-center gap-2">
                      <span className="text-cyan-400">🌈</span>
                      <span>IIRS Hyperspectral</span>
                    </span>
                    <span className="text-emerald-400">✓</span>
                  </div>
                  <div className="flex items-center justify-between text-slate-300">
                    <span className="flex items-center gap-2">
                      <span className="text-blue-400">⛰️</span>
                      <span>DTM Elevation (DEM)</span>
                    </span>
                    <span className={detail?.dem_available ? "text-emerald-400" : "text-slate-500"}>
                      {detail?.dem_available ? "✓" : "—"}
                    </span>
                  </div>
                </div>
              </div>

              {/* 3. Live Telemetry Activity Card (Matches reference Activity card) */}
              <div className="rounded-2xl border border-white/[0.08] bg-white/[0.025] p-4 backdrop-blur-xl shadow-lg flex flex-col justify-between">
                <div>
                  <div className="flex items-center justify-between mb-3">
                    <span className="text-[11px] font-semibold uppercase tracking-wider text-slate-400">
                      Registration Telemetry
                    </span>
                    <span className="rounded-full bg-emerald-500/10 px-2 py-0.5 text-[9px] font-semibold text-emerald-300 border border-emerald-500/20">
                      Sub-Pixel
                    </span>
                  </div>

                  <div className="space-y-2.5 text-xs font-mono">
                    <div className="flex items-center justify-between text-slate-300 border-b border-white/[0.04] pb-2">
                      <span className="text-slate-400 font-sans text-xs">RMSE Error</span>
                      <span className="text-white font-bold">{metrics ? `${metrics.rmse_px.toFixed(3)} px` : "—"}</span>
                    </div>
                    <div className="flex items-center justify-between text-slate-300 border-b border-white/[0.04] pb-2">
                      <span className="text-slate-400 font-sans text-xs">Post-RANSAC Inliers</span>
                      <span className="text-purple-300 font-bold">{metrics ? `${metrics.num_inliers}` : "—"}</span>
                    </div>
                    <div className="flex items-center justify-between text-slate-300 border-b border-white/[0.04] pb-2">
                      <span className="text-slate-400 font-sans text-xs">Coverage Score</span>
                      <span className="text-white font-bold">{metrics ? `${(metrics.combined_coverage_score * 100).toFixed(0)}%` : "—"}</span>
                    </div>
                  </div>
                </div>

                {/* Animated Wave Sparkline Graphic (Exactly from reference image) */}
                <div className="mt-4 pt-2">
                  <svg className="w-full h-8 overflow-visible" viewBox="0 0 200 30" fill="none">
                    <path
                      d="M0 25 C 20 25, 30 15, 50 18 C 70 21, 80 5, 100 8 C 120 11, 130 22, 150 15 C 170 8, 180 2, 200 6"
                      stroke="url(#purpleGrad)"
                      strokeWidth="2.5"
                      strokeLinecap="round"
                    />
                    <path
                      d="M0 25 C 20 25, 30 15, 50 18 C 70 21, 80 5, 100 8 C 120 11, 130 22, 150 15 C 170 8, 180 2, 200 6 L 200 30 L 0 30 Z"
                      fill="url(#purpleGlow)"
                      opacity="0.3"
                    />
                    <defs>
                      <linearGradient id="purpleGrad" x1="0" y1="0" x2="200" y2="0" gradientUnits="userSpaceOnUse">
                        <stop stopColor="#818cf8" />
                        <stop offset="0.5" stopColor="#c084fc" />
                        <stop offset="1" stopColor="#e879f9" />
                      </linearGradient>
                      <linearGradient id="purpleGlow" x1="0" y1="0" x2="0" y2="30" gradientUnits="userSpaceOnUse">
                        <stop stopColor="#c084fc" stopOpacity="0.5" />
                        <stop offset="1" stopColor="#c084fc" stopOpacity="0" />
                      </linearGradient>
                    </defs>
                  </svg>
                </div>
              </div>
            </div>

          </div>

          {/* 3. Bottom Dossier Grid Section: All Validated Regions in Frosted Glass */}
          <div className="border-t border-white/[0.08] bg-white/[0.015] p-6 md:p-8">
            <div className="mb-5 flex items-center justify-between">
              <div>
                <h3 className="text-sm font-bold text-white">All Validated Lunar Regions</h3>
                <p className="text-xs text-slate-400">Complete multi-sensor repository ready for co-registration</p>
              </div>
              <button
                onClick={() => openVaultWithFilter("all")}
                className="text-xs font-semibold text-purple-300 hover:text-purple-200 transition"
              >
                Open Full Vault →
              </button>
            </div>

            <div className="grid grid-cols-1 sm:grid-cols-2 md:grid-cols-3 lg:grid-cols-4 gap-4">
              {triplets.map((t, i) => {
                const { widthKm, heightKm } = footprintSizeKm(t.bounds);
                return (
                  <div
                    key={t.id}
                    onClick={() => handleOpenDossierModal(t)}
                    className="group cursor-pointer rounded-2xl border border-white/[0.07] bg-white/[0.025] p-4 backdrop-blur-xl transition-all hover:border-purple-400/40 hover:bg-white/[0.06] hover:shadow-[0_10px_30px_rgba(168,85,247,0.15)]"
                  >
                    <div className="flex items-center justify-between mb-2">
                      <span className="font-mono text-[10px] text-slate-400">
                        #{String(i + 1).padStart(2, "0")}
                      </span>
                      <span className="text-[10px] text-purple-300 opacity-0 group-hover:opacity-100 transition">
                        Inspect →
                      </span>
                    </div>
                    <h4 className="text-xs font-semibold text-white group-hover:text-purple-200 transition">
                      {t.id}
                    </h4>
                    <p className="mt-1 text-[11px] text-slate-400">
                      {widthKm.toFixed(1)} × {heightKm.toFixed(1)} km
                      {t.dem_available && " · DEM"}
                    </p>

                    <div className="mt-3 pt-3 border-t border-white/[0.05] flex items-center gap-2 text-[10px]">
                      <button
                        onClick={(e) => { e.stopPropagation(); handleOpenDossierModal(t); }}
                        className="rounded-full border border-purple-400/30 bg-purple-500/10 px-2.5 py-0.5 text-purple-200 hover:bg-purple-500/20"
                      >
                        Report
                      </button>
                      <button
                        onClick={(e) => { e.stopPropagation(); handleSelectRegionAndScroll(t.id, "registration"); }}
                        className="rounded-full border border-white/10 bg-white/[0.04] px-2.5 py-0.5 text-slate-300 hover:bg-white/[0.08] hover:text-white"
                      >
                        Workspace
                      </button>
                    </div>
                  </div>
                );
              })}
            </div>
          </div>

          {/* Cockpit Bottom Footnote */}
          <footer className="border-t border-white/[0.06] bg-black/20 px-8 py-4 text-xs text-slate-400 flex flex-wrap items-center justify-between gap-2">
            <span>ISRO Chandrayaan-2 · SIH26166 · Sub-Pixel Homography Pipeline</span>
            <span>{triplets.length} Regions Loaded · Real-Time Telemetry</span>
          </footer>
        </div>
      </div>

      {/* Modals with Matching Glassmorphic Aesthetics */}
      {activeDossierTriplet && (
        <DossierModal
          triplet={activeDossierTriplet}
          metrics={activeDossierMetrics}
          onClose={() => setActiveDossierTriplet(null)}
          onOpenWorkspace={(id) => handleSelectRegionAndScroll(id, "registration")}
        />
      )}
      {vaultOpen && (
        <VaultModal
          triplets={triplets}
          initialFilter={vaultInitialFilter}
          onClose={() => setVaultOpen(false)}
          onSelectRegion={(id, preferredView) => handleSelectRegionAndScroll(id, preferredView)}
        />
      )}
      {theoryModalOpen && (
        <TheoryModal onClose={() => setTheoryModalOpen(false)} />
      )}
      {infoModalContent && (
        <InfoModal
          content={infoModalContent}
          onClose={() => setInfoModalContent(null)}
        />
      )}
    </div>
  );
}

function describeError(err: unknown): string {
  if (err instanceof ApiError) return err.message;
  if (err instanceof Error) return err.message;
  return "Unknown error.";
}
