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

  const currentFootprint = detail ? footprintSizeKm(detail.bounds) : null;

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
          
          {/* Header Bar: Clean & Functional */}
          <header className="flex flex-wrap items-center justify-between gap-4 border-b border-white/[0.08] bg-white/[0.02] px-6 py-4 backdrop-blur-md">
            {/* Left: Brand Identity & Back Link */}
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
                  <h2 className="text-sm font-bold tracking-tight text-white">
                    Chandrayaan-2 Console
                  </h2>
                  <p className="text-[10px] text-slate-400 font-mono">ISRO Planetary Cross-Matching</p>
                </div>
              </div>
            </div>

            {/* Center: Search Filter Bar */}
            <div className="relative flex-1 max-w-md hidden sm:block">
              <div className="relative flex items-center rounded-full border border-white/10 bg-white/[0.04] px-4 py-2 transition-all focus-within:border-purple-400/50 focus-within:bg-white/[0.07] focus-within:shadow-[0_0_20px_rgba(168,85,247,0.2)]">
                <svg className="h-4 w-4 text-slate-400 mr-2.5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0z" />
                </svg>
                <input
                  type="text"
                  value={searchQuery}
                  onChange={(e) => setSearchQuery(e.target.value)}
                  placeholder="Filter regions..."
                  className="w-full bg-transparent text-xs text-white placeholder-slate-400 focus:outline-none"
                />
              </div>
            </div>

            {/* Right: Actions */}
            <div className="flex items-center gap-3">
              <RegistrationLauncher />
              <button
                onClick={() => openVaultWithFilter("all")}
                className="flex items-center gap-2 rounded-full border border-white/10 bg-white/[0.04] px-4 py-1.5 text-xs text-slate-300 transition hover:border-purple-400/40 hover:bg-white/[0.08] hover:text-white focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-purple-400"
                title="Browse Full Archive Vault"
              >
                <span className="h-2 w-2 rounded-full bg-emerald-400 shadow-[0_0_8px_#34d399]" />
                <span>{triplets.length} Regions</span>
              </button>
            </div>
          </header>

          {/* 3-Column Workstation Cockpit */}
          <div className="grid grid-cols-1 lg:grid-cols-12 gap-6 p-6 md:p-8">
            
            {/* ======================================================== */}
            {/* LEFT COLUMN: Workspace Views & Region Directory          */}
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
                    <span>Theory &amp; Framework ↗</span>
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
                    <p className="text-[10px] text-slate-400 font-mono">{filteredTriplets.length} loaded</p>
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

                <div className="mt-3 flex-1 space-y-2 overflow-y-auto max-h-[440px] pr-1">
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
            {/* CENTER COLUMN: Central Inspection Stage                   */}
            {/* ======================================================== */}
            <div ref={arenaRef} className="lg:col-span-6 flex flex-col gap-5 scroll-mt-20">
              
              {/* Region Header */}
              <div className="flex flex-wrap items-center justify-between gap-3 px-1">
                <div>
                  <h1 className="text-xl sm:text-2xl font-bold tracking-tight text-white flex items-center gap-2">
                    <span>{detail?.id ?? "Select a region"}</span>
                    {detail?.dem_available && (
                      <span className="rounded-full border border-purple-400/30 bg-purple-500/10 px-2.5 py-0.5 text-[10px] font-semibold text-purple-200">
                        DEM Ready
                      </span>
                    )}
                  </h1>
                  <p className="mt-0.5 text-xs text-slate-400">
                    {currentFootprint
                      ? `${currentFootprint.widthKm.toFixed(1)} × ${currentFootprint.heightKm.toFixed(1)} km terrain footprint`
                      : "Multi-sensor cross-match arena"}
                  </p>
                </div>

                <div className="flex items-center gap-2">
                  {detail && (
                    <button
                      onClick={() => handleOpenDossierModal(detail)}
                      className="rounded-full border border-purple-400/30 bg-purple-500/10 px-3.5 py-1 text-xs font-semibold text-purple-200 transition hover:bg-purple-500/20"
                    >
                      Dossier Report ↗
                    </button>
                  )}
                  <button
                    onClick={cycleView}
                    className="rounded-full border border-white/10 bg-white/[0.04] px-3.5 py-1 text-xs text-slate-300 transition hover:bg-white/[0.08] hover:text-white"
                    title="Switch inspection mode"
                  >
                    {view === "registration" ? "Switch to Linked Cursor →" : view === "linked-cursor" ? "Switch to Map →" : "Switch to Registration →"}
                  </button>
                </div>
              </div>

              {/* Central Viewer Stage */}
              <div className="relative overflow-hidden rounded-2xl border border-white/[0.1] bg-black/40 backdrop-blur-2xl p-5 shadow-2xl flex flex-col min-h-[500px]">
                
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
                          { src: `/images/registered/${detail.id}/checkerboard_qa.png`, fallback: `/images/tmc/${detail.id}`, label: "Checkerboard QA" },
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
                        <span className="text-slate-400 font-mono">
                          Inliers: <strong className="text-purple-300">{metrics?.num_inliers ?? 0} matches</strong>
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
                    <div className="h-full min-h-[420px] rounded-xl overflow-hidden border border-white/[0.08]">
                      <MapPanel triplet={detail} iirsOverlay={iirsOverlay} />
                    </div>
                  )}
                </div>
              </div>
            </div>

            {/* ======================================================== */}
            {/* RIGHT COLUMN: Real Telemetry & Region Metadata           */}
            {/* ======================================================== */}
            <div className="lg:col-span-3 flex flex-col gap-5">
              
              {/* 1. Region Metadata Card */}
              <div className="rounded-2xl border border-white/[0.08] bg-white/[0.025] p-5 backdrop-blur-xl shadow-lg">
                <span className="text-[11px] font-semibold uppercase tracking-wider text-slate-400 block mb-3">
                  Selected Footprint
                </span>
                <div className="space-y-2.5 text-xs font-mono">
                  <div className="flex items-center justify-between text-slate-300 border-b border-white/[0.04] pb-2">
                    <span className="text-slate-400 font-sans">Longitude</span>
                    <span className="text-white">{detail ? `${detail.bounds.west_lon.toFixed(2)}° to ${detail.bounds.east_lon.toFixed(2)}°` : "—"}</span>
                  </div>
                  <div className="flex items-center justify-between text-slate-300 border-b border-white/[0.04] pb-2">
                    <span className="text-slate-400 font-sans">Latitude</span>
                    <span className="text-white">{detail ? `${detail.bounds.south_lat.toFixed(2)}° to ${detail.bounds.north_lat.toFixed(2)}°` : "—"}</span>
                  </div>
                  <div className="flex items-center justify-between text-slate-300 border-b border-white/[0.04] pb-2">
                    <span className="text-slate-400 font-sans">Dimensions</span>
                    <span className="text-white">{currentFootprint ? `${currentFootprint.widthKm.toFixed(1)} × ${currentFootprint.heightKm.toFixed(1)} km` : "—"}</span>
                  </div>
                  <div className="flex items-center justify-between text-slate-300">
                    <span className="text-slate-400 font-sans">Elevation Data</span>
                    <span className={detail?.dem_available ? "text-emerald-400 font-semibold" : "text-slate-500"}>
                      {detail?.dem_available ? "DTM Available" : "Interpolated"}
                    </span>
                  </div>
                </div>
              </div>

              {/* 2. Live Telemetry Card (Real metrics from backend!) */}
              <div className="rounded-2xl border border-white/[0.08] bg-white/[0.025] p-5 backdrop-blur-xl shadow-lg flex flex-col justify-between">
                <div>
                  <div className="flex items-center justify-between mb-3">
                    <span className="text-[11px] font-semibold uppercase tracking-wider text-slate-400">
                      Registration Telemetry
                    </span>
                    <span className={`rounded-full px-2 py-0.5 text-[9px] font-semibold border ${
                      metrics?.sub_pixel_accurate
                        ? "bg-emerald-500/10 text-emerald-300 border-emerald-500/20"
                        : "bg-purple-500/10 text-purple-300 border-purple-500/20"
                    }`}>
                      {metrics?.sub_pixel_accurate ? "Sub-Pixel Verified" : "Standard"}
                    </span>
                  </div>

                  <div className="space-y-3 text-xs font-mono">
                    <div className="rounded-xl border border-white/[0.06] bg-white/[0.02] p-3">
                      <div className="flex items-center justify-between text-slate-400 font-sans text-[11px]">
                        <span>RMSE Error</span>
                        <span className="text-[10px]">Threshold &lt; 0.5 px</span>
                      </div>
                      <div className="mt-1 text-base font-bold text-purple-300">
                        {metrics ? `${metrics.rmse_px.toFixed(3)} px` : "—"}
                      </div>
                    </div>

                    <div className="rounded-xl border border-white/[0.06] bg-white/[0.02] p-3">
                      <div className="flex items-center justify-between text-slate-400 font-sans text-[11px]">
                        <span>Inlier Matches</span>
                        <span className="text-[10px]">Post-RANSAC</span>
                      </div>
                      <div className="mt-1 text-base font-bold text-white">
                        {metrics ? `${metrics.num_inliers} inliers` : "—"}
                      </div>
                    </div>

                    <div className="rounded-xl border border-white/[0.06] bg-white/[0.02] p-3">
                      <div className="flex items-center justify-between text-slate-400 font-sans text-[11px]">
                        <span>Spatial Coverage</span>
                        <span className="text-[10px]">Uniformity</span>
                      </div>
                      <div className="mt-1 text-base font-bold text-white">
                        {metrics ? `${(metrics.combined_coverage_score * 100).toFixed(1)}%` : "—"}
                      </div>
                    </div>
                  </div>
                </div>

                {detail && (
                  <div className="mt-5 pt-3 border-t border-white/[0.06]">
                    <button
                      onClick={() => handleOpenDossierModal(detail)}
                      className="w-full rounded-xl bg-gradient-to-r from-purple-600/30 to-indigo-600/30 border border-purple-400/30 py-2.5 text-xs font-semibold text-purple-200 transition hover:bg-purple-600/40 hover:text-white"
                    >
                      Open Full Dossier Report ↗
                    </button>
                  </div>
                )}
              </div>

            </div>

          </div>

          {/* Cockpit Bottom Footnote */}
          <footer className="border-t border-white/[0.06] bg-black/20 px-8 py-3.5 text-xs text-slate-400 flex flex-wrap items-center justify-between gap-2">
            <span>ISRO Chandrayaan-2 · SIH26166 · Sub-Pixel Homography Pipeline</span>
            <span>{triplets.length} Regions Loaded</span>
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
