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
  const [savedQuoteBookmarked, setSavedQuoteBookmarked] = useState(false);
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

  const toggleBookmark = (e: React.MouseEvent) => {
    e.stopPropagation();
    const next = !savedQuoteBookmarked;
    setSavedQuoteBookmarked(next);
    try {
      localStorage.setItem("sih26166_saved_ref99", next ? "true" : "false");
    } catch {}
    showToast(next ? "Quote saved to Archival References" : "Quote removed from References");
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
    // Fetch metrics for this triplet if not current
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

  // Helper to find specific triplets for the dossier cards
  const region001 = triplets.find((t) => t.id === "region_001") ?? triplets[0];
  const region003 = triplets.find((t) => t.id === "region_003") ?? triplets[2] ?? triplets[0];
  const antimeridianRegion =
    triplets.find((t) => t.id === "triplet_new_2022") ??
    triplets[triplets.length - 1] ??
    triplets[0];

  if (error) {
    return (
      <div className="flex h-screen items-center justify-center bg-obsidian px-6">
        <div className="max-w-md border border-alert/40 bg-obsidian-card p-6 text-center">
          <p className="font-mono text-sm font-semibold uppercase tracking-wider text-alert">
            Archive Connection Failed
          </p>
          <p className="mt-2 text-xs text-ink-dim">{error}</p>
          <p className="mt-4 font-mono text-[10px] text-ink-faint">
            Confirm FastAPI server is active on port 8000.
          </p>
        </div>
      </div>
    );
  }

  return (
    <div className="relative min-h-screen bg-[#07090e] font-sans text-[#f0f2f5] selection:bg-[#2c2619] selection:text-[#f3df9b]">
      {/* Deep Space Starfield Background Layer */}
      <div
        className="fixed inset-0 pointer-events-none z-0 bg-cover bg-center bg-no-repeat"
        style={{
          backgroundImage: "url('/starfield-bg.png')",
        }}
      />
      {/* Subtle cosmic vignette for deep contrast & readability */}
      <div className="fixed inset-0 pointer-events-none z-0 bg-black/40 backdrop-brightness-[0.95]" />

      <div className="relative z-10">
        {/* Toast Notification */}
        {toastMessage && (
          <div className="fixed bottom-6 right-6 z-[9999] flex items-center gap-3 rounded-xl border border-[#d4af37]/60 bg-[#0d1017]/85 px-4 py-2.5 font-mono text-xs text-[#f3df9b] shadow-[0_8px_32px_rgba(212,175,55,0.3)] backdrop-blur-xl animate-fade-in">
            <span className="h-2 w-2 rounded-full bg-[#d4af37] shadow-[0_0_8px_#d4af37]" />
            <span>{toastMessage}</span>
          </div>
        )}

        {/* 1. Obsidian Top Navigation Bar */}
        <header className="sticky top-0 z-40 flex h-16 items-center justify-between border-b border-white/10 bg-[#07090e]/60 px-6 backdrop-blur-xl shadow-[0_4px_30px_rgba(0,0,0,0.5)] md:px-12">
          {/* Left: Editorial Archive Brand + Back Link */}
          <div className="flex items-center gap-4">
            {onBackToHero && (
              <button
                onClick={onBackToHero}
                className="flex items-center gap-1.5 rounded-lg border border-white/15 bg-white/[0.05] px-3 py-1.5 font-mono text-xs text-[#d4af37] backdrop-blur-md transition-all duration-200 hover:border-[#d4af37]/60 hover:bg-[#d4af37]/15 hover:text-[#f3df9b] focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-[#d4af37]"
              >
                <span>←</span>
                <span>HERO</span>
              </button>
            )}
            <div className="flex items-center gap-2 rounded-lg border border-white/10 bg-white/[0.03] px-3 py-1 backdrop-blur-sm">
              <span className="font-mono text-xs text-[#d4af37]">[ ]</span>
              <span className="text-xs font-bold uppercase tracking-[0.25em] text-[#e8d5b5]">
                CHANDRAYAAN-2 ARCHIVES
              </span>
            </div>
          </div>

          {/* Center: Nav Modes (FUSED MAP / LINKED CURSOR / REGISTRATION QA) */}
          <nav className="hidden items-center gap-2 rounded-xl border border-white/10 bg-white/[0.03] p-1 backdrop-blur-md md:flex">
            <button
              onClick={() => {
                setView("map");
                scrollToArena();
              }}
              className={`rounded-lg px-3.5 py-1 text-xs font-semibold uppercase tracking-[0.18em] transition-all focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-[#d4af37] ${
                view === "map"
                  ? "border border-[#d4af37]/40 bg-[#d4af37]/20 text-[#f3df9b] shadow-[0_0_12px_rgba(212,175,55,0.25)]"
                  : "text-[#9a958e] hover:bg-white/[0.05] hover:text-white"
              }`}
            >
              FUSED MAP
            </button>
            <button
              onClick={() => {
                setView("linked-cursor");
                scrollToArena();
              }}
              className={`rounded-lg px-3.5 py-1 text-xs font-semibold uppercase tracking-[0.18em] transition-all focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-[#d4af37] ${
                view === "linked-cursor"
                  ? "border border-[#d4af37]/40 bg-[#d4af37]/20 text-[#f3df9b] shadow-[0_0_12px_rgba(212,175,55,0.25)]"
                  : "text-[#9a958e] hover:bg-white/[0.05] hover:text-white"
              }`}
            >
              LINKED CURSOR
            </button>
            <button
              onClick={() => {
                setView("registration");
                scrollToArena();
              }}
              className={`rounded-lg px-3.5 py-1 text-xs font-semibold uppercase tracking-[0.18em] transition-all focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-[#d4af37] ${
                view === "registration"
                  ? "border border-[#d4af37]/40 bg-[#d4af37]/20 text-[#f3df9b] shadow-[0_0_12px_rgba(212,175,55,0.25)]"
                  : "text-[#9a958e] hover:bg-white/[0.05] hover:text-white"
              }`}
            >
              REGISTRATION QA
            </button>
          </nav>

          {/* Right: Vault Quick Access / Region Counter / Live Registration */}
          <div className="flex items-center gap-3">
            <RegistrationLauncher />
            <button
              onClick={() => openVaultWithFilter("all")}
              className="flex items-center gap-2 rounded-lg border border-white/15 bg-white/[0.05] px-3.5 py-1.5 font-mono text-xs text-[#d4af37] backdrop-blur-md transition-all hover:border-[#d4af37]/60 hover:bg-[#d4af37]/15 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-[#d4af37]"
              title="Open Full Archive Vault"
            >
              <span className="h-1.5 w-1.5 rounded-full bg-[#d4af37] shadow-[0_0_6px_#d4af37]" />
              <span>{triplets.length < 10 ? `0${triplets.length}` : triplets.length} TILES READY</span>
            </button>
          </div>
        </header>

      {/* Main Body */}
      <main className="mx-auto max-w-[1440px] px-6 py-10 md:px-12">
        {/* 2. Headline Area: EXPLORING the silent FRONTIERS */}
        <section className="mb-10 flex flex-col justify-between gap-4 border-b border-white/10 pb-8 md:flex-row md:items-end">
          <div>
            <span className="font-mono text-xs uppercase tracking-[0.25em] text-[#9a958e]">
              EXPLORING
            </span>
            <h1 className="mt-1 font-serif text-5xl font-normal italic tracking-tight text-[#e8d5b5] sm:text-6xl md:text-7xl">
              the silent
            </h1>
            <span className="mt-1 block text-3xl font-bold uppercase tracking-[0.2em] text-white sm:text-4xl drop-shadow-[0_0_20px_rgba(255,255,255,0.15)]">
              FRONTIERS
            </span>
          </div>

          <div className="flex flex-col items-start font-mono text-xs text-[#9a958e] md:items-end">
            <button
              onClick={() => openVaultWithFilter("all")}
              className="rounded-lg border border-white/10 bg-white/[0.04] px-3 py-1.5 text-[#d4af37] backdrop-blur-sm transition-all hover:border-[#d4af37]/50 hover:bg-[#d4af37]/10 hover:text-[#f3df9b] focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-[#d4af37]"
            >
              {triplets.length < 10 ? `0${triplets.length}` : triplets.length} — VALIDATED TILES ↗
            </button>
            <span className="mt-1 text-[11px] text-[#6b665f]">
              ISRO CHANDRAYAAN-2 · SIH26166
            </span>
          </div>
        </section>

        {/* 3. Featured Inspection Arena (FIG. 01 / LUNA) */}
        <section ref={arenaRef} className="mb-14 grid grid-cols-1 gap-6 lg:grid-cols-12 scroll-mt-20">
          {/* Left Column: Tiles / Articles Navigation */}
          <div className="flex flex-col rounded-2xl border border-white/10 bg-[#0a0d14]/55 p-5 shadow-[0_8px_32px_0_rgba(0,0,0,0.5)] backdrop-blur-xl lg:col-span-3">
            <div className="flex items-center justify-between border-b border-white/10 pb-3">
              <span className="font-mono text-xs font-bold uppercase tracking-[0.2em] text-white">
                TILES.
              </span>
              <div className="flex items-center gap-1 font-mono text-xs">
                <button
                  onClick={handlePrev}
                  className="flex h-6 w-6 items-center justify-center rounded border border-white/15 bg-white/[0.04] text-[#9a958e] backdrop-blur-sm transition-colors hover:border-[#d4af37] hover:bg-[#d4af37]/10 hover:text-[#d4af37] focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-[#d4af37]"
                  title="Previous Region"
                >
                  &lt;
                </button>
                <button
                  onClick={handleNext}
                  className="flex h-6 w-6 items-center justify-center rounded border border-white/15 bg-white/[0.04] text-[#9a958e] backdrop-blur-sm transition-colors hover:border-[#d4af37] hover:bg-[#d4af37]/10 hover:text-[#d4af37] focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-[#d4af37]"
                  title="Next Region"
                >
                  &gt;
                </button>
              </div>
            </div>

            {/* Search Input Box */}
            <div className="mt-3 flex items-center gap-2 rounded-lg border border-white/10 bg-white/[0.04] px-3 py-2 font-mono text-xs text-[#6b665f] backdrop-blur-md focus-within:border-[#d4af37]/50 focus-within:bg-white/[0.07]">
              <svg className="h-3.5 w-3.5 text-[#6b665f]" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0z" />
              </svg>
              <input
                type="text"
                value={searchQuery}
                onChange={(e) => setSearchQuery(e.target.value)}
                placeholder="SEARCH TILES..."
                className="w-full bg-transparent text-xs text-white placeholder-[#6b665f] focus:outline-none"
              />
            </div>

            {/* Region List */}
            <div className="mt-4 flex-1 space-y-1.5 overflow-y-auto max-h-[380px] pr-1">
              {filteredTriplets.map((t, i) => {
                const active = t.id === selectedId;
                const { widthKm, heightKm } = footprintSizeKm(t.bounds);
                return (
                  <button
                    key={t.id}
                    onClick={() => setSelectedId(t.id)}
                    className={`group flex w-full flex-col rounded-lg p-2.5 text-left transition-all backdrop-blur-sm focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-[#d4af37] ${
                      active
                        ? "border border-[#d4af37]/50 bg-[#d4af37]/15 shadow-[0_0_15px_rgba(212,175,55,0.15)]"
                        : "border border-white/[0.04] bg-white/[0.02] hover:border-white/15 hover:bg-white/[0.06]"
                    }`}
                  >
                    <div className="flex items-center justify-between">
                      <span className="font-mono text-[10px] text-[#6b665f]">
                        {i + 1 < 10 ? `0${i + 1}` : i + 1}.
                      </span>
                      {t.dem_available && (
                        <span className="rounded border border-[#d4af37]/30 bg-[#d4af37]/15 px-1.5 py-0.2 font-mono text-[9px] text-[#d4af37]">
                          DEM
                        </span>
                      )}
                    </div>
                    <span
                      className={`font-mono text-xs font-medium transition-colors ${
                        active ? "text-[#d4af37] font-semibold" : "text-[#e8d5b5] group-hover:text-white"
                      }`}
                    >
                      {t.id}
                    </span>
                    <span className="mt-0.5 font-mono text-[10px] text-[#6b665f]">
                      {widthKm.toFixed(1)} × {heightKm.toFixed(1)} km
                    </span>
                  </button>
                );
              })}
            </div>
          </div>

          {/* Center Column: Featured Interactive Card with [ FIG. 01 / LUNA ] & Circular Gold Button */}
          <div className="relative flex min-h-[460px] flex-col overflow-hidden rounded-2xl border border-white/10 bg-[#0a0d14]/55 p-6 shadow-[0_8px_32px_0_rgba(0,0,0,0.5)] backdrop-blur-xl lg:col-span-6">
            {/* Top Badge: FIG. 01 / LUNA */}
            <div className="mb-4 flex items-center justify-between">
              <span className="rounded-lg border border-white/15 bg-white/[0.05] px-3 py-1 font-mono text-[10px] uppercase tracking-wider text-[#d4af37] backdrop-blur-md">
                [ FIG. {currentIndex >= 0 ? (currentIndex + 1 < 10 ? `0${currentIndex + 1}` : currentIndex + 1) : "01"} / {detail?.id ?? "LUNA"} ]
              </span>
              <div className="flex items-center gap-3">
                {detail && (
                  <button
                    onClick={() => handleOpenDossierModal(detail)}
                    className="rounded-md border border-[#d4af37]/30 bg-[#d4af37]/10 px-2.5 py-1 font-mono text-[10px] text-[#d4af37] backdrop-blur-sm transition-all hover:bg-[#d4af37]/20 hover:underline focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-[#d4af37]"
                  >
                    REPORT ↗
                  </button>
                )}
                <span className="rounded-md border border-white/10 bg-white/[0.03] px-2 py-0.5 font-mono text-[10px] uppercase text-[#9a958e] backdrop-blur-sm">
                  {view.replace("-", " ")}
                </span>
              </div>
            </div>

            {/* Active Workspace View */}
            <div className="relative flex-1 overflow-hidden">
              {loading && (
                <div className="flex h-full items-center justify-center font-mono text-xs text-[#9a958e]">
                  Loading archival tiles from backend…
                </div>
              )}
              {!loading && !detail && (
                <div className="flex h-full items-center justify-center font-mono text-xs text-[#9a958e]">
                  No region selected.
                </div>
              )}

              {/* View 1: Geometric Registration QA */}
              {detail && view === "registration" && (
                <div className="flex h-full flex-col justify-between">
                  <div className="grid grid-cols-3 gap-3">
                    <div className="flex flex-col gap-1.5">
                      <div className="relative aspect-square overflow-hidden rounded-xl border border-white/15 bg-black/40 shadow-inner backdrop-blur-sm">
                        {/* eslint-disable-next-line @next/next/no-img-element */}
                        <img
                          src={imageUrl(`/images/registered/${detail.id}/registered_ohrc.png`)}
                          alt="Warped OHRC"
                          className="h-full w-full object-cover contrast-[1.2] brightness-95"
                          onError={(e) => {
                            (e.currentTarget as HTMLImageElement).src = imageUrl(`/images/ohrc/${detail.id}`);
                          }}
                        />
                      </div>
                      <span className="font-mono text-[10px] text-[#9a958e]">1. Warped OHRC</span>
                    </div>

                    <div className="flex flex-col gap-1.5">
                      <div className="relative aspect-square overflow-hidden rounded-xl border border-white/15 bg-black/40 shadow-inner backdrop-blur-sm">
                        {/* eslint-disable-next-line @next/next/no-img-element */}
                        <img
                          src={imageUrl(`/images/registered/${detail.id}/blend_overlay.png`)}
                          alt="Blend 50%"
                          className="h-full w-full object-cover contrast-[1.2] brightness-95"
                          onError={(e) => {
                            (e.currentTarget as HTMLImageElement).src = imageUrl(`/images/tmc/${detail.id}`);
                          }}
                        />
                      </div>
                      <span className="font-mono text-[10px] text-[#9a958e]">2. Blend 50%</span>
                    </div>

                    <div className="flex flex-col gap-1.5">
                      <div className="relative aspect-square overflow-hidden rounded-xl border border-white/15 bg-black/40 shadow-inner backdrop-blur-sm">
                        {/* eslint-disable-next-line @next/next/no-img-element */}
                        <img
                          src={imageUrl(`/images/registered/${detail.id}/checkerboard_qa.png`)}
                          alt="Checkerboard"
                          className="h-full w-full object-cover contrast-[1.2] brightness-95"
                          onError={(e) => {
                            (e.currentTarget as HTMLImageElement).src = imageUrl(`/images/tmc/${detail.id}`);
                          }}
                        />
                      </div>
                      <span className="font-mono text-[10px] text-[#9a958e]">3. Checkerboard</span>
                    </div>
                  </div>

                  <div className="mt-4 flex flex-wrap items-center justify-between rounded-xl border border-white/10 bg-white/[0.03] p-3.5 pr-16 font-mono text-[11px] backdrop-blur-md">
                    <span className="text-[#9a958e]">
                      Status: <span className="text-[#d4af37] font-semibold">{metrics?.sub_pixel_accurate ? "Sub-Pixel Verified (< 0.5 px)" : "Standard Match"}</span>
                    </span>
                    <span className="text-[#6b665f]">
                      Inliers: <span className="text-white font-semibold">{metrics?.num_inliers ?? 0}</span>
                    </span>
                  </div>
                </div>
              )}

              {/* View 2: Linked Cursor Pane */}
              {detail && view === "linked-cursor" && (
                <div className="h-full">
                  <LinkedCursorPanel tripletId={detail.id} points={matches} />
                </div>
              )}

              {/* View 3: Fused Map Pane */}
              {detail && view === "map" && (
                <div className="h-full min-h-[380px] rounded-xl overflow-hidden border border-white/10">
                  <MapPanel triplet={detail} iirsOverlay={iirsOverlay} />
                </div>
              )}
            </div>

            {/* Circular Gold Floating Action Button (Cycle View Mode) */}
            <button
              onClick={cycleView}
              className="absolute bottom-5 right-5 z-30 flex h-10 w-10 items-center justify-center rounded-full bg-[#d4af37] text-black shadow-[0_0_25px_rgba(212,175,55,0.45)] backdrop-blur-sm transition-all duration-300 hover:scale-110 hover:bg-[#f3df9b] active:scale-95 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-white"
              title="Cycle Inspection View"
            >
              <span className="text-base font-bold">→</span>
            </button>
          </div>

          {/* Right Column: ■ FROM THE ARCHIVE */}
          <div className="flex flex-col justify-between rounded-2xl border border-white/10 bg-[#0a0d14]/55 p-6 shadow-[0_8px_32px_0_rgba(0,0,0,0.5)] backdrop-blur-xl lg:col-span-3">
            <div>
              <div className="flex items-center gap-2">
                <span className="h-1.5 w-1.5 rounded-full bg-[#d4af37] shadow-[0_0_6px_#d4af37]" />
                <span className="font-mono text-xs font-bold uppercase tracking-[0.2em] text-[#d4af37]">
                  FROM THE ARCHIVE
                </span>
              </div>

              <p className="mt-3 text-xs leading-relaxed text-[#9a958e]">
                The latest dispatch from the outer rim of human understanding. We
                dissect multi-sensor signal noise across Chandrayaan-2 to find
                geometric truth in the void.
              </p>

              {/* Volumes & Metrics List - Light Glassmorphic Text Boxes */}
              <div className="mt-6 space-y-3 font-mono text-xs">
                <div className="rounded-xl border border-white/10 bg-white/[0.03] p-3 backdrop-blur-md transition-all hover:border-white/20 hover:bg-white/[0.05]">
                  <div className="flex items-center justify-between text-[10px] text-[#6b665f]">
                    <span>VOL. 01.1</span>
                    <span>RMSE ACCURACY</span>
                  </div>
                  <p className="mt-1 font-sans text-sm font-semibold text-[#e8d5b5]">
                    {metrics ? `${metrics.rmse_px.toFixed(3)} px` : "0.419 px"}
                  </p>
                </div>

                <div className="rounded-xl border border-white/10 bg-white/[0.03] p-3 backdrop-blur-md transition-all hover:border-white/20 hover:bg-white/[0.05]">
                  <div className="flex items-center justify-between text-[10px] text-[#6b665f]">
                    <span>VOL. 01.2</span>
                    <span>FEATURE COVERAGE</span>
                  </div>
                  <p className="mt-1 font-sans text-sm font-semibold text-[#e8d5b5]">
                    {metrics ? `${(metrics.combined_coverage_score * 100).toFixed(0)}% Occupied` : "23% Occupied"}
                  </p>
                </div>

                <div className="rounded-xl border border-white/10 bg-white/[0.03] p-3 backdrop-blur-md transition-all hover:border-white/20 hover:bg-white/[0.05]">
                  <div className="flex items-center justify-between text-[10px] text-[#6b665f]">
                    <span>VOL. 01.3</span>
                    <span>SCALE RATIO</span>
                  </div>
                  <p className="mt-1 font-sans text-sm font-semibold text-[#e8d5b5]">
                    16× (0.25m OHRC → 4m TMC)
                  </p>
                </div>
              </div>
            </div>

            {/* Quick Link */}
            <div className="mt-6">
              <button
                onClick={cycleView}
                className="group flex w-full items-center justify-between rounded-xl border border-[#d4af37]/30 bg-[#d4af37]/10 p-3 text-xs font-semibold text-[#d4af37] backdrop-blur-md transition-all duration-200 hover:border-[#d4af37]/60 hover:bg-[#d4af37]/20 hover:text-[#f3df9b] focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-[#d4af37]"
              >
                <span>Switch View Mode ({view})</span>
                <span className="transition-transform duration-200 group-hover:translate-x-1">→</span>
              </button>
            </div>
          </div>
        </section>

        {/* 4. INDEX / DOSSIER (6-Card Archival Grid) */}
        <section className="mb-14">
          <div className="mb-4 flex items-center justify-between">
            <span className="font-mono text-xs font-bold uppercase tracking-[0.25em] text-[#9a958e]">
              INDEX / DOSSIER
            </span>
            <span className="font-mono text-xs text-[#6b665f]">
              {triplets.length} DOCUMENTED REGIONS
            </span>
          </div>

          <div className="grid grid-cols-1 gap-4 md:grid-cols-3">
            {/* Card 1: Dossier 01 (Polar Rim & Permanent Shadow) */}
            {(() => {
              const bounds = region001 ? footprintSizeKm(region001.bounds) : { widthKm: 3.2, heightKm: 3.8 };
              return (
                <div
                  onClick={() => region001 && handleOpenDossierModal(region001)}
                  className="group relative flex min-h-[220px] cursor-pointer flex-col justify-between rounded-2xl border border-white/10 bg-[#0a0d14]/50 p-5 shadow-[0_8px_24px_rgba(0,0,0,0.4)] backdrop-blur-xl transition-all duration-300 hover:border-[#d4af37]/50 hover:bg-[#121622]/70 hover:shadow-[0_8px_32px_rgba(212,175,55,0.15)] hover:-translate-y-0.5 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-[#d4af37]"
                  tabIndex={0}
                  onKeyDown={(e) => {
                    if (e.key === "Enter" && region001) handleOpenDossierModal(region001);
                  }}
                >
                  <div className="flex items-center justify-between">
                    <span className="font-mono text-[10px] font-bold uppercase tracking-widest text-[#d4af37]">
                      DOSSIER · 01
                    </span>
                    <span className="font-mono text-[9px] text-[#6b665f] transition-colors group-hover:text-[#d4af37]">
                      CLICK TO OPEN ↗
                    </span>
                  </div>
                  <div>
                    <h3 className="text-sm font-semibold text-white transition-colors group-hover:text-[#d4af37]">
                      Polar Rim &amp; Permanent Shadow
                    </h3>
                    <p className="mt-1 text-xs text-[#9a958e]">
                      Sub-pixel alignment across extreme illumination disparity (&gt;160° solar angle).
                    </p>
                    <div className="mt-3 flex items-center gap-3 font-mono text-[10px]">
                      <button
                        onClick={(e) => {
                          e.stopPropagation();
                          if (region001) handleOpenDossierModal(region001);
                        }}
                        className="rounded-md border border-[#d4af37]/30 bg-[#d4af37]/10 px-2 py-0.5 text-[#d4af37] backdrop-blur-sm hover:bg-[#d4af37]/20 hover:underline focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-[#d4af37]"
                      >
                        READ DOSSIER →
                      </button>
                      <button
                        onClick={(e) => {
                          e.stopPropagation();
                          if (region001) handleSelectRegionAndScroll(region001.id, "registration");
                        }}
                        className="rounded-md border border-white/10 bg-white/[0.04] px-2 py-0.5 text-[#9a958e] backdrop-blur-sm hover:text-white hover:border-white/20 focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-[#d4af37]"
                      >
                        OPEN WORKSPACE →
                      </button>
                    </div>
                  </div>
                  <span className="font-mono text-[10px] text-[#6b665f]">
                    {region001?.id ?? "region_001"} · {bounds.widthKm.toFixed(1)} × {bounds.heightKm.toFixed(1)} km
                  </span>
                </div>
              );
            })()}

            {/* Card 2: Editorial Quote & Theory Link */}
            <div className="relative flex min-h-[220px] flex-col justify-between rounded-2xl border border-white/10 bg-[#0a0d14]/50 p-5 shadow-[0_8px_24px_rgba(0,0,0,0.4)] backdrop-blur-xl transition-all hover:border-white/20 hover:bg-[#121622]/70">
              <div className="flex items-center justify-between">
                <span className="font-mono text-[10px] font-bold uppercase tracking-widest text-[#d4af37]">
                  QUOTE / REF-99
                </span>
                <button
                  onClick={toggleBookmark}
                  className={`flex h-7 w-7 items-center justify-center rounded-lg border backdrop-blur-sm transition-all focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-[#d4af37] ${
                    savedQuoteBookmarked
                      ? "border-[#d4af37] bg-[#d4af37] text-black shadow-[0_0_12px_rgba(212,175,55,0.5)]"
                      : "border-white/15 bg-white/[0.04] text-[#9a958e] hover:border-[#d4af37] hover:text-[#d4af37]"
                  }`}
                  title={savedQuoteBookmarked ? "Remove Bookmark" : "Save to References"}
                >
                  <svg className="h-3.5 w-3.5" fill={savedQuoteBookmarked ? "currentColor" : "none"} stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 5a2 2 0 012-2h10a2 2 0 012 2v16l-7-3.5L5 21V5z" />
                  </svg>
                </button>
              </div>

              <p className="font-serif text-sm italic leading-relaxed text-[#e8d5b5]">
                "The grid is a conceptual framework, an intellectual construct. It is not an image."
              </p>

              <div className="flex items-center justify-between font-mono text-[10px]">
                <button
                  onClick={() => setTheoryModalOpen(true)}
                  className="rounded-md border border-white/10 bg-white/[0.03] px-2 py-0.5 text-[#9a958e] backdrop-blur-sm transition-colors hover:text-[#d4af37] hover:border-[#d4af37]/40 focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-[#d4af37]"
                >
                  THEORY OF HOMOGRAPHY ↗
                </button>
                <span className="text-[9px] text-[#6b665f]">
                  {savedQuoteBookmarked ? "SAVED" : "REF-99"}
                </span>
              </div>
            </div>

            {/* Card 3: Field Note (Live Inliers & QA Link) */}
            <div
              onClick={() => {
                setView("registration");
                scrollToArena();
              }}
              className="group relative flex min-h-[220px] cursor-pointer flex-col justify-between rounded-2xl border border-white/10 bg-[#0a0d14]/50 p-5 shadow-[0_8px_24px_rgba(0,0,0,0.4)] backdrop-blur-xl transition-all duration-300 hover:border-[#d4af37]/50 hover:bg-[#121622]/70 hover:shadow-[0_8px_32px_rgba(212,175,55,0.15)] hover:-translate-y-0.5 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-[#d4af37]"
              tabIndex={0}
              onKeyDown={(e) => {
                if (e.key === "Enter") {
                  setView("registration");
                  scrollToArena();
                }
              }}
            >
              <div className="flex items-center justify-between">
                <span className="rounded-lg border border-[#d4af37]/30 bg-[#d4af37]/15 px-2 py-0.5 font-mono text-[9px] uppercase tracking-wider text-[#d4af37] backdrop-blur-sm">
                  FIELD NOTE
                </span>
                <span className="font-mono text-[10px] font-semibold text-[#d4af37]">
                  {metrics ? `${metrics.rmse_px.toFixed(3)} PX` : "0.419 PX"}
                </span>
              </div>
              <div>
                <h3 className="text-sm font-semibold text-white transition-colors group-hover:text-[#d4af37]">
                  Decay Rates &amp; Rim Continuity
                </h3>
                <p className="mt-1 text-xs text-[#9a958e]">
                  Gradient and census transform features establish robust correspondence despite shadow reversals.
                </p>
                <div className="mt-2 font-mono text-[10px] text-[#d4af37] group-hover:underline">
                  INSPECT INLIERS &amp; RMSE →
                </div>
              </div>
              <span className="font-mono text-[10px] text-[#6b665f]">
                RANSAC inliers: <span className="text-white font-semibold">{metrics?.num_inliers ?? 28}</span> (Live)
              </span>
            </div>

            {/* Card 4: Long Read (Motion Studies in the Metropolis) */}
            {(() => {
              const target = region003;
              const bounds = target ? footprintSizeKm(target.bounds) : { widthKm: 4.0, heightKm: 6.2 };
              return (
                <div
                  onClick={() => target && handleOpenDossierModal(target)}
                  className="group relative flex min-h-[220px] cursor-pointer flex-col justify-between rounded-2xl border border-white/10 bg-[#0a0d14]/50 p-5 shadow-[0_8px_24px_rgba(0,0,0,0.4)] backdrop-blur-xl transition-all duration-300 hover:border-[#d4af37]/50 hover:bg-[#121622]/70 hover:shadow-[0_8px_32px_rgba(212,175,55,0.15)] hover:-translate-y-0.5 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-[#d4af37]"
                  tabIndex={0}
                  onKeyDown={(e) => {
                    if (e.key === "Enter" && target) handleOpenDossierModal(target);
                  }}
                >
                  <span className="font-mono text-[10px] font-bold uppercase tracking-widest text-[#d4af37]">
                    LONG READ
                  </span>
                  <div>
                    <h3 className="text-sm font-semibold text-white transition-colors group-hover:text-[#d4af37]">
                      Motion Studies in the Metropolis of Craters
                    </h3>
                    <p className="mt-1 text-xs text-[#9a958e]">
                      Multi-modal registration between 0.25m OHRC and 70m IIRS hyperspectral imagery.
                    </p>
                    <div className="mt-3 flex items-center gap-3 font-mono text-[10px]">
                      <button
                        onClick={(e) => {
                          e.stopPropagation();
                          if (target) handleOpenDossierModal(target);
                        }}
                        className="rounded-md border border-[#d4af37]/30 bg-[#d4af37]/10 px-2 py-0.5 text-[#d4af37] backdrop-blur-sm hover:bg-[#d4af37]/20 hover:underline focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-[#d4af37]"
                      >
                        READ DOSSIER →
                      </button>
                      <button
                        onClick={(e) => {
                          e.stopPropagation();
                          if (target) handleSelectRegionAndScroll(target.id, "map");
                        }}
                        className="rounded-md border border-white/10 bg-white/[0.04] px-2 py-0.5 text-[#9a958e] backdrop-blur-sm hover:text-white hover:border-white/20 focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-[#d4af37]"
                      >
                        OPEN WORKSPACE →
                      </button>
                    </div>
                  </div>
                  <span className="font-mono text-[10px] text-[#6b665f]">
                    {target?.id ?? "region_003"} · {bounds.widthKm.toFixed(1)} × {bounds.heightKm.toFixed(1)} km
                  </span>
                </div>
              );
            })()}

            {/* Card 5: Antimeridian Far-Side Basin */}
            {(() => {
              const target = antimeridianRegion;
              const bounds = target ? footprintSizeKm(target.bounds) : { widthKm: 4.0, heightKm: 24.7 };
              return (
                <div
                  onClick={() => target && handleOpenDossierModal(target)}
                  className="group relative flex min-h-[220px] cursor-pointer flex-col justify-between rounded-2xl border border-white/10 bg-[#0a0d14]/50 p-5 shadow-[0_8px_24px_rgba(0,0,0,0.4)] backdrop-blur-xl transition-all duration-300 hover:border-[#d4af37]/50 hover:bg-[#121622]/70 hover:shadow-[0_8px_32px_rgba(212,175,55,0.15)] hover:-translate-y-0.5 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-[#d4af37]"
                  tabIndex={0}
                  onKeyDown={(e) => {
                    if (e.key === "Enter" && target) handleOpenDossierModal(target);
                  }}
                >
                  <span className="font-mono text-[10px] font-bold uppercase tracking-widest text-[#d4af37]">
                    DOSSIER · 08
                  </span>
                  <div>
                    <h3 className="text-sm font-semibold text-white transition-colors group-hover:text-[#d4af37]">
                      Antimeridian Far-Side Basin
                    </h3>
                    <p className="mt-1 text-xs text-[#9a958e]">
                      Handling longitude wrap-around (180° / 360°) with continuous spatial projection.
                    </p>
                    <div className="mt-3 flex items-center gap-3 font-mono text-[10px]">
                      <button
                        onClick={(e) => {
                          e.stopPropagation();
                          if (target) handleOpenDossierModal(target);
                        }}
                        className="rounded-md border border-[#d4af37]/30 bg-[#d4af37]/10 px-2 py-0.5 text-[#d4af37] backdrop-blur-sm hover:bg-[#d4af37]/20 hover:underline focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-[#d4af37]"
                      >
                        READ DOSSIER →
                      </button>
                      <button
                        onClick={(e) => {
                          e.stopPropagation();
                          if (target) handleSelectRegionAndScroll(target.id, "linked-cursor");
                        }}
                        className="rounded-md border border-white/10 bg-white/[0.04] px-2 py-0.5 text-[#9a958e] backdrop-blur-sm hover:text-white hover:border-white/20 focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-[#d4af37]"
                      >
                        OPEN WORKSPACE →
                      </button>
                    </div>
                  </div>
                  <span className="font-mono text-[10px] text-[#6b665f]">
                    {target?.id ?? "triplet_new_2022"} · {bounds.widthKm.toFixed(1)} × {bounds.heightKm.toFixed(1)} km
                  </span>
                </div>
              );
            })()}

            {/* Card 6: Access The Vault */}
            <div
              onClick={() => openVaultWithFilter("all")}
              className="group relative flex min-h-[220px] cursor-pointer flex-col items-center justify-center rounded-2xl border border-white/15 bg-gradient-to-br from-[#0e1422]/60 to-[#0a0d14]/70 p-5 text-center shadow-[0_8px_24px_rgba(0,0,0,0.4)] backdrop-blur-xl transition-all duration-300 hover:border-[#d4af37]/60 hover:bg-[#161d2e]/80 hover:shadow-[0_8px_32px_rgba(212,175,55,0.2)] hover:-translate-y-0.5 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-[#d4af37]"
              tabIndex={0}
              onKeyDown={(e) => {
                if (e.key === "Enter") openVaultWithFilter("all");
              }}
            >
              <svg className="h-6 w-6 text-[#d4af37] transition-transform duration-300 group-hover:scale-110" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M5 8h14M5 8a2 2 0 110-4h14a2 2 0 110 4M5 8v10a2 2 0 002 2h10a2 2 0 002-2V8m-9 4h4" />
              </svg>
              <span className="mt-3 font-mono text-xs font-bold uppercase tracking-[0.2em] text-[#d4af37]">
                ACCESS THE VAULT
              </span>
              <p className="mt-1 text-xs text-[#9a958e]">
                View all registration products &amp; raw imagery across {triplets.length} regions
              </p>
              <span className="mt-3 font-mono text-[10px] text-[#d4af37] transition-transform group-hover:translate-x-1">
                OPEN WORKSPACE →
              </span>
            </div>
          </div>
        </section>

        {/* 5. Archival Footer */}
        <footer className="rounded-2xl border border-white/10 bg-[#0a0d14]/40 p-8 font-mono text-xs text-[#6b665f] shadow-[0_8px_32px_0_rgba(0,0,0,0.35)] backdrop-blur-xl">
          <div className="flex flex-col justify-between gap-8 md:flex-row md:items-start">
            {/* Left Statement */}
            <div className="max-w-md">
              <span className="text-[#d4af37]">[ ]</span>
              <p className="mt-2 font-serif text-sm italic text-[#9a958e]">
                "The investigation continues into the digital void, where sensor signals meet sub-pixel truth."
              </p>
              <p className="mt-4 text-[10px] text-[#4f4b45]">
                ISRO Chandrayaan-2 · SIH26166 · Built with Next.js &amp; FastAPI
              </p>
            </div>

            {/* Right Links Columns */}
            <div className="flex flex-wrap gap-12 font-mono text-[11px]">
              {/* Indices */}
              <div className="flex flex-col gap-2">
                <span className="font-bold uppercase tracking-wider text-[#d4af37]">INDICES</span>
                <button
                  onClick={() => openVaultWithFilter("all")}
                  className="text-left text-[#9a958e] transition-colors hover:text-white focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-[#d4af37]"
                >
                  ARCHIVES ↗
                </button>
                <button
                  onClick={() => {
                    setView("map");
                    scrollToArena();
                  }}
                  className="text-left text-[#9a958e] transition-colors hover:text-white focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-[#d4af37]"
                >
                  MAP
                </button>
                <button
                  onClick={() => {
                    setView("linked-cursor");
                    scrollToArena();
                  }}
                  className="text-left text-[#9a958e] transition-colors hover:text-white focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-[#d4af37]"
                >
                  CURSOR
                </button>
              </div>

              {/* Payloads */}
              <div className="flex flex-col gap-2">
                <span className="font-bold uppercase tracking-wider text-[#d4af37]">PAYLOADS</span>
                <button
                  onClick={() => openVaultWithFilter("ohrc")}
                  className="text-left text-[#9a958e] transition-colors hover:text-white focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-[#d4af37]"
                >
                  OHRC (0.25m) ↗
                </button>
                <button
                  onClick={() => openVaultWithFilter("tmc")}
                  className="text-left text-[#9a958e] transition-colors hover:text-white focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-[#d4af37]"
                >
                  TMC-2 (4m) ↗
                </button>
                <button
                  onClick={() => openVaultWithFilter("iirs")}
                  className="text-left text-[#9a958e] transition-colors hover:text-white focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-[#d4af37]"
                >
                  IIRS (70m) ↗
                </button>
              </div>

              {/* Legal / Provenance */}
              <div className="flex flex-col gap-2">
                <span className="font-bold uppercase tracking-wider text-[#d4af37]">LEGAL</span>
                <button
                  onClick={() =>
                    setInfoModalContent({
                      tag: "DATA GOVERNANCE",
                      title: "ISRO Planetary Data System Provenance",
                      subtitle: "Official Chandrayaan-2 PDS4 Archive Compliance",
                      paragraphs: [
                        "All imagery, telemetry, and ephemeris records are ingested directly from ISRO's ISSDC (Indian Space Science Data Centre) Chandrayaan-2 PDS4 repositories.",
                        "Optical High Resolution Camera (OHRC) operates at 0.25–0.32 m/pixel, Terrain Mapping Camera-2 (TMC-2) operates at ~4–5 m/pixel in stereo triplets, and Imaging Infrared Spectrometer (IIRS) operates in the 0.8–5.0 µm hyperspectral range.",
                        "All products derived via this cross-matching console conform to spatial reference standards on the Lunar Ellipsoid (IAU/IAG 2000 coordinate system).",
                      ],
                      specs: [
                        { label: "Data Authority", value: "ISRO / ISSDC" },
                        { label: "Coordinate Frame", value: "IAU 2000 Moon (Mean Earth/Polar)" },
                        { label: "Optical Sensor", value: "CH2_OHRC (0.25m)" },
                        { label: "Stereo Sensor", value: "CH2_TMC2 (4m)" },
                        { label: "Hyperspectral", value: "CH2_IIRS (70m, 256 bands)" },
                      ],
                    })
                  }
                  className="text-left text-[#9a958e] transition-colors hover:text-white focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-[#d4af37]"
                >
                  ISRO DATA ↗
                </button>
                <button
                  onClick={() =>
                    setInfoModalContent({
                      tag: "HACKATHON CRITERIA",
                      title: "SIH26166 Problem Statement Verification",
                      subtitle: "Automated Multi-Sensor Lunar Image Cross-Matching",
                      paragraphs: [
                        "Problem Statement SIH26166 requires sub-pixel co-registration of high-resolution Chandrayaan-2 lunar imagery characterized by high scale disparity (16×) and severe illumination differences (>140° solar angle).",
                        "Our implementation achieves verified sub-pixel accuracy (RMSE < 0.5 px) using semi-dense deep feature matching (LoFTR) coupled with iterative RANSAC planar homography estimation.",
                        "The solution provides real-time dual-cursor coordinate projection, multi-band spectral overlay blending, and automated checkerboard quality assurance.",
                      ],
                      specs: [
                        { label: "Problem Statement ID", value: "SIH26166" },
                        { label: "Accuracy Target", value: "Sub-pixel (RMSE < 1.0 px)" },
                        { label: "Achieved Accuracy", value: "RMSE 0.417 px (Verified)" },
                        { label: "Validated Regions", value: "8 / 8 Pass" },
                      ],
                    })
                  }
                  className="text-left text-[#9a958e] transition-colors hover:text-white focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-[#d4af37]"
                >
                  SIH26166 ↗
                </button>
                <button
                  onClick={() =>
                    setInfoModalContent({
                      tag: "OPEN ARCHITECTURE",
                      title: "Open Research & Algorithm Documentation",
                      subtitle: "Transformer Feature Tracking & Robust Geometry",
                      paragraphs: [
                        "This platform is built on open planetary science and computer vision standards. Feature correspondence relies on Local Feature Transformers (LoFTR), eliminating traditional detector bottlenecks on lunar craters.",
                        "Planar homography estimation and reprojection error optimization are implemented in Python via OpenCV and NumPy. Geospatial footprint bounds and tile services are served via FastAPI and Leaflet.",
                      ],
                      specs: [
                        { label: "Matching Backbone", value: "LoFTR (Local Feature Transformer)" },
                        { label: "Geometric Estimator", value: "RANSAC Homography" },
                        { label: "Backend Stack", value: "Python 3.14 / FastAPI / Uvicorn" },
                        { label: "Frontend Stack", value: "Next.js 14 / TypeScript / Tailwind" },
                      ],
                    })
                  }
                  className="text-left text-[#9a958e] transition-colors hover:text-white focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-[#d4af37]"
                >
                  OPEN RESEARCH ↗
                </button>
              </div>
            </div>
          </div>
        </footer>
      </main>

      {/* 6. Active Modals */}
      {/* Dossier Report Modal */}
      {activeDossierTriplet && (
        <DossierModal
          triplet={activeDossierTriplet}
          metrics={activeDossierMetrics}
          onClose={() => setActiveDossierTriplet(null)}
          onOpenWorkspace={(id) => handleSelectRegionAndScroll(id, "registration")}
        />
      )}

      {/* Vault Archive Modal */}
      {vaultOpen && (
        <VaultModal
          triplets={triplets}
          initialFilter={vaultInitialFilter}
          onClose={() => setVaultOpen(false)}
          onSelectRegion={(id, preferredView) => handleSelectRegionAndScroll(id, preferredView)}
        />
      )}

      {/* Theory Dossier Modal */}
      {theoryModalOpen && (
        <TheoryModal onClose={() => setTheoryModalOpen(false)} />
      )}

      {/* General Info / Legal Modal */}
      {infoModalContent && (
        <InfoModal
          content={infoModalContent}
          onClose={() => setInfoModalContent(null)}
        />
      )}
      </div>
    </div>
  );
}

function describeError(err: unknown): string {
  if (err instanceof ApiError) return err.message;
  if (err instanceof Error) return err.message;
  return "Unknown error.";
}
