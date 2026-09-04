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
    <div className="min-h-screen bg-[#111827] font-sans text-[#e5e7eb]">
      {/* Toast Notification */}
      {toastMessage && (
        <div className="fixed bottom-6 right-6 z-[9999] flex items-center gap-2 rounded-md border border-[#374151] bg-[#1f2937] px-4 py-2.5 text-sm text-[#e5e7eb] shadow-lg">
          <span className="h-2 w-2 rounded-full bg-[#60a5fa]" />
          <span>{toastMessage}</span>
        </div>
      )}

      {/* Top Navigation Bar */}
      <header className="sticky top-0 z-40 flex h-14 items-center justify-between border-b border-[#1f2937] bg-[#111827] px-6 md:px-10">
        <div className="flex items-center gap-4">
          {onBackToHero && (
            <button
              onClick={onBackToHero}
              className="flex items-center gap-1.5 rounded-md border border-[#374151] bg-[#1f2937] px-3 py-1.5 text-xs text-[#9ca3af] transition-colors hover:bg-[#374151] hover:text-white focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-[#60a5fa]"
            >
              <span>←</span>
              <span>Back</span>
            </button>
          )}
          <span className="text-sm font-semibold tracking-wide text-white">
            Chandrayaan-2 Console
          </span>
        </div>

        {/* Center Nav Tabs */}
        <nav className="hidden items-center gap-1 rounded-md border border-[#1f2937] bg-[#0d1117] p-0.5 md:flex">
          {(["map", "linked-cursor", "registration"] as View[]).map((v) => (
            <button
              key={v}
              onClick={() => { setView(v); scrollToArena(); }}
              className={`rounded px-3 py-1 text-xs font-medium transition-colors focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-[#60a5fa] ${
                view === v
                  ? "bg-[#1f2937] text-white"
                  : "text-[#6b7280] hover:text-[#d1d5db]"
              }`}
            >
              {v === "map" ? "Map" : v === "linked-cursor" ? "Linked Cursor" : "Registration QA"}
            </button>
          ))}
        </nav>

        <div className="flex items-center gap-3">
          <RegistrationLauncher />
          <button
            onClick={() => openVaultWithFilter("all")}
            className="flex items-center gap-2 rounded-md border border-[#374151] bg-[#1f2937] px-3 py-1.5 text-xs text-[#9ca3af] transition-colors hover:bg-[#374151] hover:text-white focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-[#60a5fa]"
          >
            <span>{triplets.length} Regions</span>
          </button>
        </div>
      </header>

      {/* Main Body */}
      <main className="mx-auto max-w-[1400px] px-6 py-8 md:px-10">
        {/* Page Header */}
        <section className="mb-8 flex flex-col justify-between gap-3 border-b border-[#1f2937] pb-6 md:flex-row md:items-end">
          <div>
            <h1 className="text-2xl font-semibold text-white">
              Multi-Sensor Registration Dashboard
            </h1>
            <p className="mt-1 text-sm text-[#6b7280]">
              ISRO Chandrayaan-2 · OHRC, TMC-2, IIRS cross-matching console
            </p>
          </div>
          <div className="flex items-center gap-3 text-sm">
            <button
              onClick={() => openVaultWithFilter("all")}
              className="rounded-md bg-[#1f2937] px-3 py-1.5 text-xs text-[#60a5fa] transition-colors hover:bg-[#374151] focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-[#60a5fa]"
            >
              {triplets.length} validated tiles →
            </button>
          </div>
        </section>

        {/* Inspection Arena */}
        <section ref={arenaRef} className="mb-10 grid grid-cols-1 gap-5 lg:grid-cols-12 scroll-mt-16">
          {/* Left: Region List */}
          <div className="flex flex-col rounded-lg border border-[#1f2937] bg-[#0d1117] p-4 lg:col-span-3">
            <div className="flex items-center justify-between border-b border-[#1f2937] pb-3">
              <span className="text-xs font-semibold text-[#d1d5db]">
                Regions
              </span>
              <div className="flex items-center gap-1 text-xs">
                <button
                  onClick={handlePrev}
                  className="flex h-6 w-6 items-center justify-center rounded border border-[#374151] bg-[#1f2937] text-[#6b7280] transition-colors hover:text-white focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-[#60a5fa]"
                >
                  &lt;
                </button>
                <button
                  onClick={handleNext}
                  className="flex h-6 w-6 items-center justify-center rounded border border-[#374151] bg-[#1f2937] text-[#6b7280] transition-colors hover:text-white focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-[#60a5fa]"
                >
                  &gt;
                </button>
              </div>
            </div>

            {/* Search */}
            <div className="mt-3 flex items-center gap-2 rounded-md border border-[#374151] bg-[#1f2937] px-3 py-2 text-xs focus-within:border-[#60a5fa]">
              <svg className="h-3.5 w-3.5 text-[#6b7280]" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0z" />
              </svg>
              <input
                type="text"
                value={searchQuery}
                onChange={(e) => setSearchQuery(e.target.value)}
                placeholder="Search regions..."
                className="w-full bg-transparent text-xs text-white placeholder-[#6b7280] focus:outline-none"
              />
            </div>

            {/* Region List */}
            <div className="mt-3 flex-1 space-y-1 overflow-y-auto max-h-[380px]">
              {filteredTriplets.map((t, i) => {
                const active = t.id === selectedId;
                const { widthKm, heightKm } = footprintSizeKm(t.bounds);
                return (
                  <button
                    key={t.id}
                    onClick={() => setSelectedId(t.id)}
                    className={`flex w-full flex-col rounded-md p-2.5 text-left transition-colors focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-[#60a5fa] ${
                      active
                        ? "bg-[#1f2937] border border-[#374151]"
                        : "border border-transparent hover:bg-[#1f2937]/50"
                    }`}
                  >
                    <div className="flex items-center justify-between">
                      <span className="text-[10px] text-[#6b7280]">
                        {String(i + 1).padStart(2, "0")}.
                      </span>
                      {t.dem_available && (
                        <span className="rounded bg-[#1e3a5f] px-1.5 py-px text-[9px] text-[#60a5fa]">
                          DEM
                        </span>
                      )}
                    </div>
                    <span className={`text-xs font-medium ${active ? "text-white" : "text-[#d1d5db]"}`}>
                      {t.id}
                    </span>
                    <span className="mt-0.5 text-[10px] text-[#6b7280]">
                      {widthKm.toFixed(1)} × {heightKm.toFixed(1)} km
                    </span>
                  </button>
                );
              })}
            </div>
          </div>

          {/* Center: Viewer */}
          <div className="relative flex min-h-[460px] flex-col overflow-hidden rounded-lg border border-[#1f2937] bg-[#0d1117] p-5 lg:col-span-6">
            <div className="mb-4 flex items-center justify-between">
              <span className="text-xs text-[#6b7280]">
                {currentIndex >= 0 ? `#${currentIndex + 1}` : "—"} / {detail?.id ?? "Select a region"}
              </span>
              <div className="flex items-center gap-2">
                {detail && (
                  <button
                    onClick={() => handleOpenDossierModal(detail)}
                    className="rounded-md bg-[#1f2937] px-2.5 py-1 text-[10px] text-[#60a5fa] transition-colors hover:bg-[#374151] focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-[#60a5fa]"
                  >
                    Report →
                  </button>
                )}
                <span className="rounded-md bg-[#1f2937] px-2 py-0.5 text-[10px] text-[#6b7280]">
                  {view.replace("-", " ")}
                </span>
              </div>
            </div>

            <div className="relative flex-1 overflow-hidden">
              {loading && (
                <div className="flex h-full items-center justify-center text-sm text-[#6b7280]">
                  Loading regions…
                </div>
              )}
              {!loading && !detail && (
                <div className="flex h-full items-center justify-center text-sm text-[#6b7280]">
                  No region selected.
                </div>
              )}

              {/* Registration QA View */}
              {detail && view === "registration" && (
                <div className="flex h-full flex-col justify-between">
                  <div className="grid grid-cols-3 gap-3">
                    {[
                      { src: `/images/registered/${detail.id}/registered_ohrc.png`, fallback: `/images/ohrc/${detail.id}`, label: "Warped OHRC" },
                      { src: `/images/registered/${detail.id}/blend_overlay.png`, fallback: `/images/tmc/${detail.id}`, label: "Blend 50%" },
                      { src: `/images/registered/${detail.id}/checkerboard_qa.png`, fallback: `/images/tmc/${detail.id}`, label: "Checkerboard" },
                    ].map((img, idx) => (
                      <div key={idx} className="flex flex-col gap-1.5">
                        <div className="relative aspect-square overflow-hidden rounded-md border border-[#1f2937] bg-black">
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
                        <span className="text-[10px] text-[#6b7280]">{idx + 1}. {img.label}</span>
                      </div>
                    ))}
                  </div>

                  <div className="mt-4 flex flex-wrap items-center justify-between rounded-md border border-[#1f2937] bg-[#1f2937] p-3 text-xs">
                    <span className="text-[#9ca3af]">
                      Status: <span className="text-white font-medium">{metrics?.sub_pixel_accurate ? "Sub-Pixel Verified (< 0.5 px)" : "Standard Match"}</span>
                    </span>
                    <span className="text-[#6b7280]">
                      Inliers: <span className="text-white font-medium">{metrics?.num_inliers ?? 0}</span>
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
                <div className="h-full min-h-[380px] rounded-md overflow-hidden border border-[#1f2937]">
                  <MapPanel triplet={detail} iirsOverlay={iirsOverlay} />
                </div>
              )}
            </div>

            {/* Cycle View Button */}
            <button
              onClick={cycleView}
              className="absolute bottom-4 right-4 z-30 flex h-9 w-9 items-center justify-center rounded-md bg-[#374151] text-white shadow-md transition-colors hover:bg-[#4b5563] focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-[#60a5fa]"
              title="Cycle View"
            >
              <span className="text-sm">→</span>
            </button>
          </div>

          {/* Right: Metrics Sidebar */}
          <div className="flex flex-col justify-between rounded-lg border border-[#1f2937] bg-[#0d1117] p-5 lg:col-span-3">
            <div>
              <span className="text-xs font-semibold text-[#d1d5db]">
                Metrics
              </span>
              <p className="mt-2 text-xs leading-relaxed text-[#6b7280]">
                Registration quality metrics for the currently selected region.
              </p>

              <div className="mt-5 space-y-3 text-xs">
                <div className="rounded-md border border-[#1f2937] bg-[#1f2937] p-3">
                  <div className="flex items-center justify-between text-[10px] text-[#6b7280]">
                    <span>RMSE</span>
                    <span>Accuracy</span>
                  </div>
                  <p className="mt-1 text-sm font-medium text-white">
                    {metrics ? `${metrics.rmse_px.toFixed(3)} px` : "—"}
                  </p>
                </div>

                <div className="rounded-md border border-[#1f2937] bg-[#1f2937] p-3">
                  <div className="flex items-center justify-between text-[10px] text-[#6b7280]">
                    <span>Coverage</span>
                    <span>Spatial</span>
                  </div>
                  <p className="mt-1 text-sm font-medium text-white">
                    {metrics ? `${(metrics.combined_coverage_score * 100).toFixed(0)}%` : "—"}
                  </p>
                </div>

                <div className="rounded-md border border-[#1f2937] bg-[#1f2937] p-3">
                  <div className="flex items-center justify-between text-[10px] text-[#6b7280]">
                    <span>Scale</span>
                    <span>OHRC → TMC</span>
                  </div>
                  <p className="mt-1 text-sm font-medium text-white">
                    16× (0.25m → 4m)
                  </p>
                </div>
              </div>
            </div>

            <div className="mt-5">
              <button
                onClick={cycleView}
                className="flex w-full items-center justify-between rounded-md bg-[#1f2937] p-3 text-xs text-[#9ca3af] transition-colors hover:bg-[#374151] hover:text-white focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-[#60a5fa]"
              >
                <span>Switch View ({view.replace("-", " ")})</span>
                <span>→</span>
              </button>
            </div>
          </div>
        </section>

        {/* Region Cards */}
        <section className="mb-10">
          <div className="mb-4 flex items-center justify-between">
            <span className="text-xs font-semibold text-[#d1d5db]">
              All Regions
            </span>
            <span className="text-xs text-[#6b7280]">
              {triplets.length} datasets
            </span>
          </div>

          <div className="grid grid-cols-1 gap-3 md:grid-cols-2 lg:grid-cols-3">
            {triplets.map((t, i) => {
              const { widthKm, heightKm } = footprintSizeKm(t.bounds);
              return (
                <div
                  key={t.id}
                  onClick={() => handleOpenDossierModal(t)}
                  className="group cursor-pointer rounded-lg border border-[#1f2937] bg-[#0d1117] p-4 transition-colors hover:border-[#374151] hover:bg-[#1f2937]/50 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-[#60a5fa]"
                  tabIndex={0}
                  onKeyDown={(e) => {
                    if (e.key === "Enter") handleOpenDossierModal(t);
                  }}
                >
                  <div className="flex items-center justify-between mb-2">
                    <span className="text-[10px] font-medium text-[#6b7280]">
                      #{String(i + 1).padStart(2, "0")}
                    </span>
                    <span className="text-[10px] text-[#6b7280] group-hover:text-[#60a5fa]">
                      Open →
                    </span>
                  </div>
                  <h3 className="text-sm font-medium text-white">
                    {t.id}
                  </h3>
                  <p className="mt-1 text-xs text-[#6b7280]">
                    {widthKm.toFixed(1)} × {heightKm.toFixed(1)} km
                    {t.dem_available && " · DEM"}
                  </p>
                  <div className="mt-3 flex items-center gap-2 text-[10px]">
                    <button
                      onClick={(e) => { e.stopPropagation(); handleOpenDossierModal(t); }}
                      className="rounded bg-[#1f2937] px-2 py-0.5 text-[#60a5fa] hover:bg-[#374151] focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-[#60a5fa]"
                    >
                      Report
                    </button>
                    <button
                      onClick={(e) => { e.stopPropagation(); handleSelectRegionAndScroll(t.id, "registration"); }}
                      className="rounded bg-[#1f2937] px-2 py-0.5 text-[#9ca3af] hover:bg-[#374151] hover:text-white focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-[#60a5fa]"
                    >
                      Workspace
                    </button>
                  </div>
                </div>
              );
            })}
          </div>
        </section>

        {/* Footer */}
        <footer className="border-t border-[#1f2937] pt-6 pb-4 text-xs text-[#4b5563]">
          <div className="flex items-center justify-between">
            <span>ISRO Chandrayaan-2 · SIH26166 · Next.js + FastAPI</span>
            <span>{triplets.length} regions loaded</span>
          </div>
        </footer>
      </main>

      {/* Modals */}
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
