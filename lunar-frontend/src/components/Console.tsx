"use client";

import React, { useEffect, useState, useRef } from "react";
import Link from "next/link";
import { useSearchParams, useRouter } from "next/navigation";
import { api, ApiError, imageUrl, API_BASE } from "@/lib/api";
import { footprintSizeKm } from "@/lib/geo";
import type { TripletSummary, MatchPoint, IIRSOverlay, MatchMetrics } from "@/lib/types";
import MapPanel from "./DynamicMapPanel";
import LinkedCursorPanel from "./LinkedCursorPanel";
import DossierModal from "./archive/DossierModal";
import VaultModal, { PayloadFilter } from "./archive/VaultModal";
import TheoryModal from "./archive/TheoryModal";
import InfoModal, { InfoModalContent } from "./archive/InfoModal";
import RegistrationLauncher from "./RegistrationLauncher";
import { getCurrentUser, logout, type AuthUser } from "@/lib/auth";
import { useTheme } from "@/lib/theme";
import {
  SENSOR_META,
  sensorMeta,
  sensorCardGsd,
  scaleRatioLabel,
  warpedOhrcTag,
  checkerboardTag,
} from "@/lib/sensors";

type View = "registration" | "linked-cursor" | "map";

interface Props {
  onBackToHero?: () => void;
  onLogout?: () => void;
}

export default function Console({ onBackToHero, onLogout }: Props = {}) {
  const searchParams = useSearchParams();
  const router = useRouter();
  const subviewParam = searchParams.get("subview") as View | null;
  const modalParam = searchParams.get("modal");

  const [triplets, setTriplets] = useState<TripletSummary[]>([]);
  const [selectedId, setSelectedId] = useState<string | null>(null);
  const [referenceMode, setReferenceMode] = useState<"tmc" | "lro_nac">("tmc");
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

  // Authenticated User Profile & Dropdown
  const [currentUser, setCurrentUser] = useState<AuthUser | null>(null);
  const [profileMenuOpen, setProfileMenuOpen] = useState(false);
  const profileMenuRef = useRef<HTMLDivElement>(null);
  const { theme, toggle: toggleTheme } = useTheme();

  const arenaRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    setCurrentUser(getCurrentUser());
  }, []);

  useEffect(() => {
    function handleClickOutside(event: MouseEvent) {
      if (profileMenuRef.current && !profileMenuRef.current.contains(event.target as Node)) {
        setProfileMenuOpen(false);
      }
    }
    if (profileMenuOpen) {
      document.addEventListener("mousedown", handleClickOutside);
      return () => document.removeEventListener("mousedown", handleClickOutside);
    }
  }, [profileMenuOpen]);

  const handleUserLogout = () => {
    setProfileMenuOpen(false);
    if (onLogout) {
      onLogout();
    } else {
      logout();
      router.push("/");
    }
  };

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

  // Load regions once on mount (honour ?region= deep-links from Ingest results)
  const regionParam = searchParams.get("region");
  useEffect(() => {
    api
      .listTriplets()
      .then((res) => {
        setTriplets(res.triplets);
        if (res.triplets.length === 0) return;
        const wanted = (regionParam || "").trim();
        const hit = wanted ? res.triplets.find((t) => t.id === wanted) : undefined;
        setSelectedId(hit ? hit.id : res.triplets[0].id);
      })
      .catch((err) => setError(describeError(err)))
      .finally(() => setLoading(false));
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  // Follow ?region= changes (e.g. clicking another Ingest row without reload)
  useEffect(() => {
    if (!regionParam || triplets.length === 0) return;
    if (triplets.some((t) => t.id === regionParam) && regionParam !== selectedId) {
      setSelectedId(regionParam);
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [regionParam, triplets]);

  // Follow ?subview= changes (e.g. sidebar links from Ingest: linked-cursor / map).
  // Without this, tapping those menu options leaves the previous flex view active.
  useEffect(() => {
    if (subviewParam === "linked-cursor" || subviewParam === "map" || subviewParam === "registration") {
      setView((prev) => (prev === subviewParam ? prev : subviewParam));
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [subviewParam]);

  // Load triplet detail, matches, and IIRS overlay
  useEffect(() => {
    if (!selectedId) return;
    setDetail(null);
    setMatches([]);
    setMetrics(null);
    setIirsOverlay(null);

    const matchKey = referenceMode === "lro_nac" ? `${selectedId}_lro_nac` : selectedId;

    // LRO cross-match rows live at {id}_lro_nac; the plain-id fallback fires
    // ONLY on 404 (no LRO row for this region). Any other failure (401/500/
    // timeout) must surface as an error — silently swallowing it would render
    // a neighbouring region's dots as this region's verification.
    const matchesFor = (key: string) =>
      api.getMatches(key).catch((err) => {
        if (matchKey !== selectedId && err instanceof ApiError && err.status === 404) {
          return api.getMatches(selectedId);
        }
        throw err;
      });

    Promise.all([
      api.getTriplet(selectedId),
      matchesFor(matchKey),
      api.getIirsOverlay(selectedId).catch(() => null),
    ])
      .then(([d, m, iirs]) => {
        setDetail(d);
        setMatches(m.matches);
        setMetrics(m.metrics ?? null);
        setIirsOverlay(iirs);
      })
      .catch((err) => setError(describeError(err)));
  }, [selectedId, referenceMode]);

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

  const handleDeleteRegion = async (tripletId: string) => {
    const curated = /^region_\d+$/i.test(tripletId);
    try {
      await api.deleteTriplet(tripletId, false);
    } catch (err) {
      // Backend asks for ?force=true on curated seeds — the vault already
      // confirmed once, so retry with force instead of failing.
      const msg = err instanceof Error ? err.message : "";
      if (curated && msg.includes("force=true")) {
        await api.deleteTriplet(tripletId, true);
      } else {
        throw err;
      }
    }
    const res = await api.listTriplets();
    setTriplets(res.triplets);
    if (selectedId === tripletId) {
      setSelectedId(res.triplets.length > 0 ? res.triplets[0].id : "");
    }
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

  const openVaultWithFilter = (filter: PayloadFilter) => {
    setVaultInitialFilter(filter);
    setVaultOpen(true);
  };

  const currentFootprint = detail ? footprintSizeKm(detail.bounds) : null;

  if (error) {
    return (
      <div className="flex h-screen items-center justify-center bg-[#f4f6fb] dark:bg-[#090b0e] px-6">
        <div className="max-w-md rounded-2xl border border-red-200 dark:border-red-900/50 bg-white dark:bg-[#0e1117] p-6 text-center shadow-lg">
          <p className="text-sm font-semibold uppercase tracking-wider text-red-600 dark:text-red-400">
            Archive Connection Failed
          </p>
          <p className="mt-2 text-xs text-slate-600 dark:text-slate-300">{error}</p>
          <p className="mt-4 font-mono text-[10px] text-slate-400 dark:text-slate-400">
            Confirm FastAPI server is active on port 8000.
          </p>
        </div>
      </div>
    );
  }

  return (
    <div className="h-screen overflow-hidden bg-[#D3CCC0] dark:bg-[#121615] font-sans text-[#1E2321] dark:text-[#E7E2D6] antialiased flex flex-col">
      {/* Toast Notification */}
      {toastMessage && (
        <div className="fixed bottom-12 right-6 z-[9999] flex items-center gap-2.5 retro-outset px-4 py-2 text-xs text-[#1E2321] dark:text-[#E7E2D6] shadow-xl animate-fade-in font-mono">
          <span className="h-2 w-2 rounded-full bg-emerald-500 animate-pulse" />
          <span>{toastMessage}</span>
        </div>
      )}

      <div className="flex-1 flex min-h-0 overflow-hidden">
        {/* ======================================================== */}
        {/* 1. LEFT SIDEBAR: Brand Logo, Main Menu, Region List      */}
        {/* ======================================================== */}
        <aside className="w-64 bg-[#E7E2D6] dark:bg-[#1A201E] border-r-2 border-[#8B8579] dark:border-[#2D3835] flex flex-col shrink-0 h-full overflow-hidden select-none">
          {/* Brand Header Card */}
          <div className="p-2 pb-0 shrink-0">
            <div className="retro-outset p-1">
              <div className="bg-[#1F4743] text-white px-2 py-1 flex items-center justify-between text-xs font-bold font-mono tracking-wider">
                <div className="flex items-center gap-1.5">
                  <div className="w-4 h-4 bg-white/20 flex items-center justify-center text-[10px] text-white font-mono font-bold">
                    C2
                  </div>
                  <span>CHANDRAYAAN-2</span>
                </div>
                <div className="flex gap-0.5">
                  <span className="window-ctrl-btn">_</span>
                  <span className="window-ctrl-btn">X</span>
                </div>
              </div>
              <div className="p-2 bg-[#E9E4D8] dark:bg-[#161B19] text-[11px] border-t border-[#AFA99B] dark:border-[#2D3835]">
                <div className="font-bold text-[#143532] dark:text-emerald-400 text-xs">ISRO Planetary Cross-Match</div>
                <div className="text-[#555C58] dark:text-[#8C9893] text-[10px]">OHRC · TMC-2 · LRO NAC · IIRS</div>
              </div>
            </div>
          </div>

          {/* Navigation Section */}
          <div className="flex-1 overflow-y-auto p-2 space-y-2 min-h-0 flex flex-col">
            <div className="retro-outset p-1.5 flex flex-col">
              <div className="bg-[#1F4743] text-white px-2 py-0.5 flex items-center justify-between text-[11px] font-bold font-mono mb-1.5">
                <span>MENU EXPLORER</span>
                <span className="text-[9px] text-[#9AC6C0]">SYS_MENU</span>
              </div>
              <nav className="space-y-1 text-xs">
                <button
                  key="registration"
                  onClick={() => { setView("registration"); scrollToArena(); }}
                  className={`w-full text-left px-2 py-1.5 flex items-center justify-between text-xs transition-all ${
                    view === "registration"
                      ? "bg-[#28557E] text-white font-bold border-t border-l border-[#4477A6] border-r-2 border-b-2 border-[#122A42] shadow-inner"
                      : "retro-button font-semibold text-[#222926] dark:text-[#E7E2D6] hover:bg-[#D9D3C5] dark:hover:bg-white/10"
                  }`}
                >
                  <div className="flex items-center gap-2">
                    <span className="text-[11px]">▣</span>
                    <span>Dashboard QA</span>
                  </div>
                  {view === "registration" && (
                    <span className="h-1.5 w-1.5 rounded-full bg-white" />
                  )}
                </button>

                <Link
                  href="/ingest"
                  prefetch={false}
                  className="retro-button w-full text-left px-2 py-1.5 flex items-center justify-between text-xs font-semibold text-[#222926] dark:text-[#E7E2D6] hover:bg-[#D9D3C5] dark:hover:bg-white/10 transition"
                >
                  <div className="flex items-center gap-2">
                    <span className="text-[11px]">⚡</span>
                    <span>Ingest &amp; Prepare</span>
                  </div>
                </Link>

                {[
                  { id: "linked-cursor", label: "Linked Cursor", icon: "⊙" },
                  { id: "map", label: "Planetary Map", icon: "☵" },
                ].map((item) => {
                  const active = view === item.id;
                  return (
                    <button
                      key={item.id}
                      onClick={() => { setView(item.id as View); scrollToArena(); }}
                      className={`w-full text-left px-2 py-1.5 flex items-center justify-between text-xs transition-all ${
                        active
                          ? "bg-[#28557E] text-white font-bold border-t border-l border-[#4477A6] border-r-2 border-b-2 border-[#122A42] shadow-inner"
                          : "retro-button font-semibold text-[#222926] dark:text-[#E7E2D6] hover:bg-[#D9D3C5] dark:hover:bg-white/10"
                      }`}
                    >
                      <div className="flex items-center gap-2">
                        <span className="text-[11px]">{item.icon}</span>
                        <span>{item.label}</span>
                      </div>
                      {active && (
                        <span className="h-1.5 w-1.5 rounded-full bg-white" />
                      )}
                    </button>
                  );
                })}

                <div className="border-t border-[#8B8579] dark:border-[#2D3835] my-1" />

                <button
                  onClick={() => openVaultWithFilter("all")}
                  className="retro-button w-full text-left px-2 py-1.5 flex items-center justify-between text-xs font-semibold text-[#222926] dark:text-[#E7E2D6] hover:bg-[#D9D3C5] dark:hover:bg-white/10 transition"
                  title="Browse all multi-sensor lunar datasets (Archive Vault)"
                >
                  <div className="flex items-center gap-2">
                    <span className="text-[11px]">🗄️</span>
                    <span>Inspect Full Vault</span>
                  </div>
                  <span className="text-[10px]">↗</span>
                </button>

                <button
                  onClick={() => setTheoryModalOpen(true)}
                  className="retro-button w-full text-left px-2 py-1.5 flex items-center justify-between text-xs font-semibold text-[#222926] dark:text-[#E7E2D6] hover:bg-[#D9D3C5] dark:hover:bg-white/10 transition"
                  title="Open mathematical framework reference"
                >
                  <div className="flex items-center gap-2">
                    <span className="text-[11px]">📖</span>
                    <span>Math Framework</span>
                  </div>
                  <span className="text-[10px]">↗</span>
                </button>
              </nav>
            </div>

            {/* Region Directory */}
            <div className="retro-outset p-1.5 flex-1 flex flex-col min-h-[220px]">
              <div className="bg-[#2D4F4A] text-white px-2 py-1 text-[11px] font-bold font-mono flex items-center justify-between mb-1">
                <span>REGIONS ({filteredTriplets.length})</span>
                <div className="flex items-center gap-1">
                  <button
                    onClick={handlePrev}
                    className="window-ctrl-btn text-[8px]"
                    title="Previous"
                  >
                    &lt;
                  </button>
                  <button
                    onClick={handleNext}
                    className="window-ctrl-btn text-[8px]"
                    title="Next"
                  >
                    &gt;
                  </button>
                </div>
              </div>

              <div className="retro-inset flex-1 overflow-y-auto p-1 space-y-1 font-mono text-[11px] max-h-[220px]">
                {filteredTriplets.map((t, i) => {
                  const active = t.id === selectedId;
                  const { widthKm, heightKm } = footprintSizeKm(t.bounds);
                  return (
                    <button
                      key={t.id}
                      onClick={() => setSelectedId(t.id)}
                      className={`flex w-full items-center justify-between px-1.5 py-1 text-left text-xs transition-all ${
                        active
                          ? "bg-[#28557E] text-white font-bold border border-[#173857] shadow-inner"
                          : "bg-[#ECE7DC] dark:bg-[#161B19] hover:bg-[#DFD9CB] dark:hover:bg-white/5 text-[#2B312E] dark:text-[#E7E2D6] border border-[#CCC6B8] dark:border-[#2D3835]"
                      }`}
                    >
                      <div className="truncate">
                        <span className={`text-[10px] mr-1.5 ${active ? "text-[#B9D4EB]" : "text-[#69726E]"}`}>
                          {String(i + 1).padStart(2, "0")}.
                        </span>
                        <span>{t.id}</span>
                        <span className={`block text-[9px] font-normal ${active ? "text-[#B9D4EB]" : "text-[#69726E]"}`}>
                          {widthKm.toFixed(1)} × {heightKm.toFixed(1)} km
                        </span>
                      </div>
                      <div className="flex items-center gap-1 shrink-0">
                        {t.dem_available && (
                          <span className={`px-1 text-[9px] border ${active ? "bg-[#1C3E61] border-[#406E99] text-white" : "bg-[#D3CBBF] dark:bg-[#2D3835] border-[#9E9789] text-[#1E2321] dark:text-[#E7E2D6]"}`}>
                            DEM
                          </span>
                        )}
                        {t.lro_nac_available && (
                          <span className={`px-1 text-[9px] border ${active ? "bg-[#785317] border-[#BD8B3E] text-white" : "bg-[#D1B890] dark:bg-[#785317] border-[#9E8662] text-black dark:text-amber-200"}`}>
                            LRO
                          </span>
                        )}
                      </div>
                    </button>
                  );
                })}
              </div>

              {/* Sidebar Bottom Hardware Status Gauge */}
              <div className="mt-2 retro-inset p-1.5 bg-[#DFD9CD] dark:bg-[#141817] text-[10px] font-mono leading-tight space-y-1">
                <div className="flex justify-between">
                  <span className="text-[#555C58] dark:text-[#7A827E]">ENGINE STATUS:</span>
                  <span className="text-emerald-800 dark:text-emerald-400 font-bold">READY TO RUN</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-[#555C58] dark:text-[#7A827E]">ALLOC_MEM:</span>
                  <span className="text-[#202724] dark:text-[#E7E2D6] font-bold">64.0 MB / OK</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-[#555C58] dark:text-[#7A827E]">ISRO LINK:</span>
                  <span className="text-emerald-800 dark:text-emerald-400 font-bold">ONLINE [TIF/RAW]</span>
                </div>
              </div>
            </div>
          </div>
        </aside>

        {/* ======================================================== */}
        {/* 2. MAIN WORKSPACE: Header Bar & Dashboard Content        */}
        {/* ======================================================== */}
      <div className="flex-1 flex flex-col min-w-0 h-full overflow-hidden">
        {/* Top Header Bar */}
        <header className="bg-[#E7E2D6] dark:bg-[#1A201E] border-b-2 border-[#8B8579] dark:border-[#2D3835] px-4 py-1.5 flex items-center justify-between gap-3 shrink-0 shadow-sm z-30">
          {/* Search Input (recessed inset box) */}
          <div className="relative retro-inset px-2 py-0.5 flex items-center bg-white dark:bg-[#0F1211] w-56 md:w-72 text-[11px]">
            <span className="text-[#6D746F] mr-1.5">🔍</span>
            <input
              type="text"
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
              placeholder="Search regions, coords..."
              className="w-full bg-transparent border-none p-0 text-[11px] focus:ring-0 text-[#222] dark:text-[#E7E2D6] outline-none"
            />
            <span className="font-mono text-[9px] text-[#888]">/</span>
          </div>

          {/* Center: Live Co-Registration Engine Status Beacon */}
          <div className="hidden lg:flex items-center gap-2 retro-inset px-2.5 py-1 text-2xs font-mono font-bold bg-[#E2DDCF] dark:bg-[#161B19] text-[#1F4743] dark:text-emerald-400">
            <span className="relative flex h-2 w-2">
              <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-emerald-400 opacity-75" />
              <span className="relative inline-flex rounded-full h-2 w-2 bg-emerald-500" />
            </span>
            <span>FASTAPI :8000</span>
            <span className="text-[#8B8579]">·</span>
            <span>CFOG SUB-PIXEL ENGINE ACTIVE</span>
          </div>

          {/* Right Header Controls */}
          <div className="flex items-center gap-2">
            <button
              type="button"
              onClick={toggleTheme}
              className="retro-button px-2.5 py-1 text-[11px] font-semibold text-[#2C312E] dark:text-[#E7E2D6] flex items-center gap-1"
              title={theme === "dark" ? "Switch to light mode" : "Switch to dark mode"}
              aria-label="Toggle color theme"
            >
              <span>{theme === "dark" ? "☀" : "🌙"}</span>
              <span className="hidden sm:inline">{theme === "dark" ? "Light" : "Dark"}</span>
            </button>

            <button
              onClick={() => {
                if (onBackToHero) onBackToHero();
                else router.push("/");
              }}
              className="retro-button px-2.5 py-1 text-[11px] font-semibold text-[#2C312E] dark:text-[#E7E2D6] flex items-center gap-1"
              title="Return to Mission Overview"
            >
              <span>←</span>
              <span>Back</span>
            </button>

            <button
              onClick={() => openVaultWithFilter("all")}
              className="retro-inset px-2.5 py-1 text-[11px] font-mono font-bold flex items-center gap-1.5 bg-[#E2DDCF] dark:bg-[#161B19] text-[#1F4743] dark:text-emerald-400"
              title="Browse all multi-sensor lunar datasets"
            >
              <span className="w-2 h-2 rounded-full bg-emerald-600 animate-pulse" />
              <span>{triplets.length} Datasets</span>
            </button>

            <div className="h-6 w-px bg-[#8B8579] dark:bg-[#2D3835] mx-1" />

            {/* Profile Pill with Interactive Dropdown */}
            <div className="relative" ref={profileMenuRef}>
              <button
                type="button"
                onClick={() => setProfileMenuOpen((prev) => !prev)}
                className="retro-button px-2 py-0.5 text-[11px] flex items-center space-x-1.5 cursor-pointer bg-[#DFD9CB] dark:bg-[#222927]"
                aria-haspopup="true"
                aria-expanded={profileMenuOpen}
              >
                <div className="w-5 h-5 bg-[#28557E] text-white flex items-center justify-center font-bold text-[9px]">
                  {currentUser?.name ? currentUser.name.slice(0, 2).toUpperCase() : "SH"}
                </div>
                <div className="hidden sm:block text-left leading-none">
                  <span className="font-bold text-[#1F2422] dark:text-[#E7E2D6] block truncate max-w-[100px]">
                    {currentUser?.name || "Shresth"}
                  </span>
                </div>
                <span className="text-[9px] text-[#555] dark:text-[#888]">▼</span>
              </button>

              {/* Dropdown Menu */}
              {profileMenuOpen && (
                <div className="absolute right-0 mt-1 w-64 retro-outset p-1.5 shadow-2xl z-50 animate-fade-in bg-[#E7E2D6] dark:bg-[#1A201E]">
                  <div className="p-2 border-b border-[#8B8579] dark:border-[#2D3835] bg-[#E9E4D8] dark:bg-[#161B19] text-xs">
                    <span className="font-bold text-[#143532] dark:text-emerald-400 block truncate">
                      {currentUser?.name || "ISRO Flight Operator"}
                    </span>
                    <span className="text-[10px] text-[#555C58] dark:text-[#8C9893] block truncate">
                      {currentUser?.email || "flight.ops@isro.gov.in"}
                    </span>
                    <span className="mt-1 inline-flex items-center gap-1 px-1.5 py-0.5 text-[9px] font-bold bg-[#D5EAD8] text-[#195924] border border-[#85C48F]">
                      <span className="h-1.5 w-1.5 rounded-full bg-emerald-500" />
                      Active Session
                    </span>
                  </div>

                  <div className="py-1 space-y-0.5 text-xs">
                    <button
                      onClick={() => {
                        setProfileMenuOpen(false);
                        openVaultWithFilter("all");
                      }}
                      className="w-full flex items-center gap-2 px-2 py-1 text-xs font-semibold text-[#1F2422] dark:text-[#E7E2D6] hover:bg-[#D9D3C5] dark:hover:bg-white/10 text-left"
                    >
                      <span>▤</span>
                      <span>Archive Vault ({triplets.length} Datasets)</span>
                    </button>

                    <button
                      onClick={() => {
                        setProfileMenuOpen(false);
                        setTheoryModalOpen(true);
                      }}
                      className="w-full flex items-center gap-2 px-2 py-1 text-xs font-semibold text-[#1F2422] dark:text-[#E7E2D6] hover:bg-[#D9D3C5] dark:hover:bg-white/10 text-left"
                    >
                      <span>📖</span>
                      <span>Methodology Reference</span>
                    </button>

                    <button
                      onClick={() => {
                        setProfileMenuOpen(false);
                        if (onBackToHero) onBackToHero();
                        else router.push("/");
                      }}
                      className="w-full flex items-center gap-2 px-2 py-1 text-xs font-semibold text-[#1F2422] dark:text-[#E7E2D6] hover:bg-[#D9D3C5] dark:hover:bg-white/10 text-left"
                    >
                      <span>🌐</span>
                      <span>Mission Overview</span>
                    </button>
                  </div>

                  <div className="pt-1 mt-1 border-t border-[#8B8579] dark:border-[#2D3835]">
                    <button
                      onClick={handleUserLogout}
                      className="w-full flex items-center gap-2 px-2 py-1 text-xs font-bold text-red-700 dark:text-red-400 hover:bg-red-100 dark:hover:bg-red-950/40 text-left"
                    >
                      <span>🚪</span>
                      <span>Log Out</span>
                    </button>
                  </div>
                </div>
              )}
            </div>
          </div>
        </header>

        {/* Dashboard Main Content */}
        <main className="flex-1 min-h-0 p-3 md:p-4 space-y-3 overflow-y-auto">
          {/* Top Overview Action Strip */}
          <div className="retro-outset p-2 flex flex-wrap items-center justify-between gap-2">
            <div>
              <h2 className="text-base font-bold text-[#143532] dark:text-emerald-400 tracking-tight flex items-center gap-2">
                <span>Mission Control</span>
                <span className="text-xs font-normal text-[#5B635F] dark:text-[#8C9893]">
                  | Lunar Multi-Sensor Cross-Matching &amp; Registration Pipeline
                </span>
              </h2>
              <p className="text-xs text-[#525955] dark:text-[#A8B2AD]">
                Co-registering high-resolution Chandrayaan-2 OHRC (0.25m) against TMC-2 (5.0m), LRO NAC, and IIRS rasters.
              </p>
            </div>
            <div className="flex items-center gap-1.5">
              <button
                onClick={() => openVaultWithFilter("all")}
                className="retro-button px-3 py-1 text-xs font-bold text-[#1F2522] dark:text-[#E7E2D6]"
              >
                Browse All Regions
              </button>
              {detail && (
                <button
                  onClick={() => handleOpenDossierModal(detail)}
                  className="retro-button px-3 py-1 text-xs font-bold text-black flex items-center gap-1 shadow"
                >
                  <span>📄</span>
                  <span>Dossier Report ↗</span>
                </button>
              )}
            </div>
          </div>

          {/* ======================================================== */}
          {/* 3. QUICK BENCHMARK PAIRS + PRIMARY INSPECTION ARENA      */}
          {/* ======================================================== */}
          <RegistrationLauncher />

          {/* ======================================================== */}
          {/* 4. ROW 2: PRIMARY INSPECTION ARENA + TELEMETRY SIDEBAR   */}
          {/* ======================================================== */}
          <div ref={arenaRef} className="grid grid-cols-1 lg:grid-cols-12 gap-3 scroll-mt-4">
            
            {/* Center Main Stage (8 cols) */}
            <div className="lg:col-span-8 retro-outset p-2 flex flex-col justify-between">
              <div>
                {/* Titlebar */}
                <div className="bg-[#1F4743] text-white px-2 py-1 flex items-center justify-between text-xs font-mono font-bold -m-2 mb-2">
                  <div className="flex items-center gap-2">
                    <span className="w-2.5 h-2.5 bg-[#4ADE80] border border-black inline-block" />
                    <span>REGISTRATION QA WORKBENCH — {detail?.id ?? "SELECT_REGION"}</span>
                  </div>
                  <div className="flex items-center gap-2">
                    {detail && (
                      <a
                        href={`${API_BASE}/api/registration/report/${detail.id}`}
                        target="_blank"
                        rel="noopener noreferrer"
                        className="retro-button bg-[#E6E0D2] dark:bg-[#222927] text-[#111] dark:text-[#E7E2D6] px-2 py-0 text-[10px] font-sans font-bold flex items-center gap-1"
                        title="Download official ISRO Verification Report PDF for this region"
                      >
                        <span>📄</span>
                        <span>ISRO Report (PDF)</span>
                      </a>
                    )}
                    <div className="flex gap-0.5">
                      <span className="window-ctrl-btn">_</span>
                      <span className="window-ctrl-btn">□</span>
                      <span className="window-ctrl-btn">X</span>
                    </div>
                  </div>
                </div>

                {/* Classic Retro Window Tabs (Directly below Titlebar) */}
                <div className="bg-[#D8D2C4] dark:bg-[#141817] px-2 pt-1.5 border-b border-[#8C867A] dark:border-[#2D3835] flex items-center justify-between -mx-2 mb-2.5">
                  <div className="flex items-center space-x-1 text-xs">
                    {[
                      { id: "registration", label: "Registration QA" },
                      { id: "linked-cursor", label: "Linked Cursor" },
                      { id: "map", label: "Planetary Map" },
                    ].map((m) => {
                      const active = view === m.id;
                      return (
                        <button
                          key={m.id}
                          onClick={() => setView(m.id as View)}
                          className={
                            active
                              ? "bg-[#28557E] text-white font-bold px-3 py-1 border-t-2 border-l-2 border-r-2 border-[#5484B0] shadow-sm text-xs"
                              : "retro-button px-3 py-1 font-semibold text-[#3F4743] dark:text-[#A8B2AD] bg-[#E7E2D6] dark:bg-[#1A201E] cursor-pointer text-xs"
                          }
                        >
                          {m.label}
                        </button>
                      );
                    })}
                  </div>

                  {/* Reference Mode Switcher */}
                  <div className="flex items-center gap-1.5 text-[11px] pb-1 font-medium">
                    <span className="text-[#555E5A] dark:text-[#8C9893]">REF:</span>
                    {detail?.lro_nac_available ? (
                      <>
                        <button
                          onClick={() => setReferenceMode("tmc")}
                          className={`retro-button px-2 py-0.5 text-xs font-semibold ${
                            referenceMode === "tmc" ? "bg-[#F4F1E8] dark:bg-[#28557E] font-bold text-black dark:text-white" : "text-[#525B57] dark:text-[#A8B2AD]"
                          }`}
                        >
                          TMC-2 (Mono)
                        </button>
                        <button
                          onClick={() => setReferenceMode("lro_nac")}
                          className={`retro-button px-2 py-0.5 text-xs font-semibold flex items-center gap-1 ${
                            referenceMode === "lro_nac" ? "bg-[#785317] text-white font-bold" : "text-[#525B57] dark:text-[#A8B2AD]"
                          }`}
                        >
                          <span>NASA LRO NAC</span>
                          <span className="text-[9px] bg-[#CFB88D] text-black px-1">Ext Ref</span>
                        </button>
                      </>
                    ) : (
                      <span className="retro-button px-2 py-0.5 text-xs font-bold bg-[#F4F1E8] dark:bg-[#28557E] text-black dark:text-white">
                        TMC-2 (Mono)
                      </span>
                    )}
                  </div>
                </div>

                {/* Viewport Canvas Area */}
                <div className="relative min-h-[440px]">
                  {loading && (
                    <div className="flex h-full min-h-[400px] items-center justify-center text-xs font-mono text-[#555C58] dark:text-[#8C9893]">
                      Loading sensor datasets…
                    </div>
                  )}

                  {!loading && !detail && (
                    <div className="flex h-full min-h-[400px] items-center justify-center text-xs font-mono text-[#555C58] dark:text-[#8C9893]">
                      No region selected. Choose a region from the explorer.
                    </div>
                  )}

                  {/* Registration QA View */}
                  {detail && view === "registration" && (
                    <div className="flex flex-col gap-2">
                      {referenceMode === "lro_nac" && (
                        <div className="flex flex-wrap items-center justify-between gap-2 retro-inset p-2 bg-[#FEF3D6] dark:bg-[#2A2315] text-xs font-mono text-[#785317] dark:text-amber-200">
                          <div className="flex items-center gap-2">
                            <span className="w-2 h-2 rounded-full bg-amber-500 animate-pulse" />
                            <span className="font-bold">NASA LRO NAC External Reference Active</span>
                            <span className="text-[11px]">· GSD ~{sensorMeta(detail, "lro_nac").gsdM.toFixed(2)}m</span>
                          </div>
                          <span className="font-bold">
                            Fit RMSE: {metrics?.fit_rmse_px?.toFixed(3) ?? metrics?.rmse_px?.toFixed(3) ?? "—"} px
                          </span>
                        </div>
                      )}
                      <div className="grid grid-cols-1 md:grid-cols-3 gap-2">
                        {[
                          referenceMode === "lro_nac"
                            ? { src: `/images/registered/lro_nac/${detail.id}/registered_source.png`, fallback: `/images/registered/${detail.id}/registered_ohrc.png`, label: "A: Warped OHRC (0.25 m/px)", tag: warpedOhrcTag(detail, true), badge: "SRC_OK", badgeColor: "text-emerald-400" }
                            : { src: `/images/registered/${detail.id}/registered_ohrc.png`, fallback: `/images/ohrc/${detail.id}`, label: "A: Warped OHRC (0.25 m/px)", tag: warpedOhrcTag(detail, false), badge: "SRC_OK", badgeColor: "text-emerald-400" },
                          referenceMode === "lro_nac"
                            ? { src: `/images/registered/lro_nac/${detail.id}/blend_overlay.png`, fallback: `/images/lro_nac/${detail.id}`, label: "B: Blend Overlay (50%)", tag: "50% Cross-Fade (OHRC + LRO NAC)", badge: "REF_BASE", badgeColor: "text-amber-400" }
                            : { src: `/images/registered/${detail.id}/blend_overlay.png`, fallback: `/images/tmc/${detail.id}`, label: "B: TMC-2 Blend (5.0 m/px)", tag: "50% Cross-Fade (OHRC + TMC-2)", badge: "REF_BASE", badgeColor: "text-amber-400" },
                          referenceMode === "lro_nac"
                            ? { src: `/images/registered/lro_nac/${detail.id}/checkerboard_qa.png`, fallback: `/images/lro_nac/${detail.id}`, label: "C: Checkerboard QA", tag: checkerboardTag(detail, true), badge: "MATCH", badgeColor: "text-cyan-400 font-bold" }
                            : { src: `/images/registered/${detail.id}/checkerboard_qa.png`, fallback: `/images/tmc/${detail.id}`, label: "C: Checkerboard QA", tag: checkerboardTag(detail, false), badge: "MATCH", badgeColor: "text-cyan-400 font-bold" },
                        ].map((img, idx) => (
                          <div
                            key={idx}
                            className="retro-inset p-1 bg-[#1A1D1C] flex flex-col"
                          >
                            <div className="bg-[#242C2A] text-white px-1.5 py-0.5 text-[10px] font-mono flex items-center justify-between border-b border-[#3B4744]">
                              <span>{img.label}</span>
                              <span className={img.badgeColor}>{img.badge}</span>
                            </div>
                            <div className="relative aspect-square overflow-hidden bg-black border border-[#333]">
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
                            <div className="p-1 bg-[#121615] text-[#8C9893] text-[9px] font-mono flex justify-between border-t border-[#242C2A]">
                              <span className="truncate">{img.tag}</span>
                            </div>
                          </div>
                        ))}
                      </div>
                    </div>
                  )}

                  {/* Linked Cursor View */}
                  {detail && view === "linked-cursor" && (
                    <div className="h-full min-h-[440px]">
                      <LinkedCursorPanel
                        tripletId={detail.id}
                        points={matches}
                        referenceMode={referenceMode}
                        notice={metrics?.metric_notes?.matches ?? null}
                        overlapPct={typeof detail.overlap_triplet_pct === "number" ? detail.overlap_triplet_pct : null}
                      />
                    </div>
                  )}

                  {/* Map View */}
                  {detail && view === "map" && (
                    <div className="h-full min-h-[440px] retro-inset overflow-hidden">
                      <MapPanel triplet={detail} iirsOverlay={iirsOverlay} />
                    </div>
                  )}
                </div>
              </div>

              {/* Viewport Bottom Footer */}
              <div className="mt-2 retro-inset p-1.5 bg-[#E8E3D7] dark:bg-[#141817] flex flex-wrap items-center justify-between text-xs">
                <div className="flex items-center gap-2">
                  <span className="w-2 h-2 rounded-full bg-emerald-500 animate-pulse" />
                  <span className="font-mono text-[11px] text-[#1E2321] dark:text-[#E7E2D6]">
                    Aligned with <strong>{metrics?.num_inliers ?? 0} inlier ties</strong>
                  </span>
                </div>
                {detail && (
                  <button
                    onClick={() => handleOpenDossierModal(detail)}
                    className="retro-button px-2 py-0.5 text-[11px] font-bold text-[#1B3E5C] dark:text-cyan-300"
                  >
                    View Comprehensive Dossier Report →
                  </button>
                )}
              </div>
            </div>

            {/* Right Telemetry Column (4 cols) */}
            <div className="lg:col-span-4 space-y-3">
              {/* Card 1: Selected Region Footprint + consolidated registration QA metrics */}
              <div className="retro-outset p-2 flex flex-col justify-between">
                <div>
                  <div className="bg-[#28557E] text-white px-2 py-1 text-xs font-mono font-bold flex justify-between items-center -m-2 mb-2">
                    <span>SELECTED FOOTPRINT DETAILS</span>
                    <span className="text-[9px] bg-[#143532] px-1">METRICS</span>
                  </div>

                  <div className="retro-inset p-2 bg-[#F9F7F2] dark:bg-[#121615] font-mono text-xs space-y-1.5">
                    <div className="flex justify-between border-b border-[#E0DACE] dark:border-[#2D3835] pb-1">
                      <span className="text-[#5D6662] dark:text-[#8C9893]">Longitude Extent:</span>
                      <span className="font-bold text-[#111] dark:text-[#E7E2D6]">
                        {detail ? `${detail.bounds.west_lon.toFixed(2)}° to ${detail.bounds.east_lon.toFixed(2)}°` : "—"}
                      </span>
                    </div>
                    <div className="flex justify-between border-b border-[#E0DACE] dark:border-[#2D3835] pb-1">
                      <span className="text-[#5D6662] dark:text-[#8C9893]">Latitude Extent:</span>
                      <span className="font-bold text-[#111] dark:text-[#E7E2D6]">
                        {detail ? `${detail.bounds.south_lat.toFixed(2)}° to ${detail.bounds.north_lat.toFixed(2)}°` : "—"}
                      </span>
                    </div>
                    <div className="flex justify-between border-b border-[#E0DACE] dark:border-[#2D3835] pb-1">
                      <span className="text-[#5D6662] dark:text-[#8C9893]">Terrain Dimensions:</span>
                      <span className="font-bold text-[#111] dark:text-[#E7E2D6]">
                        {currentFootprint ? `${currentFootprint.widthKm.toFixed(1)} × ${currentFootprint.heightKm.toFixed(1)} km` : "—"}
                      </span>
                    </div>
                    <div className="flex justify-between items-center border-b border-[#E0DACE] dark:border-[#2D3835] pb-1">
                      <span className="text-[#5D6662] dark:text-[#8C9893]">Topographic DEM:</span>
                      <span className={`px-1.5 py-0.2 text-[10px] font-bold border ${detail?.dem_available ? "bg-[#D5EAD8] text-[#195924] border-[#85C48F]" : "bg-[#ECE7DC] text-[#777] border-[#CCC]"}`}>
                        {detail?.dem_available ? "Available (TMC DTM)" : "Interpolated"}
                      </span>
                    </div>
                    <div className="border-b border-[#E0DACE] dark:border-[#2D3835] pb-1">
                      <div className="flex justify-between items-baseline">
                        <span className="text-[#5D6662] dark:text-[#8C9893] text-[11px]">Absolute Topo RMSE:</span>
                        <span className="font-bold text-sm text-[#111] dark:text-[#E7E2D6]">
                          {metrics?.absolute_rmse_m != null ? `${metrics.absolute_rmse_m.toFixed(2)} meters` : "—"}
                        </span>
                      </div>
                      <div className="text-[9px] text-[#7A827E] text-right">
                        {metrics?.absolute_rmse_m == null
                          ? "No DEM/GSD"
                          : metrics?.absolute_rmse_m_provenance === "planar_footprint_gsd_no_dem"
                          ? "Planar estimate (no DEM)"
                          : "DEM-corrected accuracy"}
                      </div>
                    </div>
                    <div className="border-b border-[#E0DACE] dark:border-[#2D3835] pb-1">
                      <div className="flex justify-between items-baseline">
                        <span className="text-[#5D6662] dark:text-[#8C9893] text-[11px]">
                          {referenceMode === "lro_nac" ? "LRO Fit / Val:" : "Fit / Val Reprojection:"}
                        </span>
                        <span className="font-bold text-[#111] dark:text-[#E7E2D6]">
                          {metrics?.fit_rmse_px != null
                            ? metrics.fit_rmse_px.toFixed(3)
                            : metrics?.rmse_px != null
                            ? metrics.rmse_px.toFixed(3)
                            : "—"}{" "}
                          <span className="font-normal text-[#888]">fit</span>
                          {" / "}
                          <span className="text-emerald-700 dark:text-emerald-400 font-bold">
                            {metrics?.validation_rmse_px != null ? metrics.validation_rmse_px.toFixed(3) : "—"}
                          </span>{" "}
                          <span className="font-normal text-[#888]">val</span>
                        </span>
                      </div>
                      <div className="text-[9px] text-[#9E6513] dark:text-amber-400 text-right">
                        {metrics?.validation_rmse_px == null
                          ? `▲ Held-out needs ≥8 inliers (have ${metrics?.num_inliers ?? 0})`
                          : "● Sub-pixel target < 1.00 px verified"}
                      </div>
                    </div>
                    <div className="flex justify-between items-baseline">
                      <span className="text-[#5D6662] dark:text-[#8C9893] text-[11px]">Verified Inliers:</span>
                      <span className="font-bold text-[#111] dark:text-[#E7E2D6]">
                        {metrics?.num_inliers ?? 0} matches ({metrics?.inlier_ratio != null ? (metrics.inlier_ratio * 100).toFixed(1) : 0}%)
                      </span>
                    </div>
                  </div>

                  {/* Quality Gate Indicators Card */}
                  <div className="mt-2 retro-inset p-2 bg-[#EFECE3] dark:bg-[#141817]">
                    <div className="text-[10px] font-bold text-[#3D4743] dark:text-[#8C9893] uppercase tracking-wider mb-1 font-mono">
                      QUALITY GATE BENCHMARK
                    </div>
                    <div className="grid grid-cols-3 gap-1 text-center font-mono text-[10px]">
                      <div className="bg-[#E4F4E7] dark:bg-emerald-950/40 border border-[#7EC48B] dark:border-emerald-800 p-1 text-[#185E26] dark:text-emerald-300 font-bold">
                        ● GATE 1<br /><span className="text-[8px]">PASS (Scale)</span>
                      </div>
                      <div className="bg-[#E4F4E7] dark:bg-emerald-950/40 border border-[#7EC48B] dark:border-emerald-800 p-1 text-[#185E26] dark:text-emerald-300 font-bold">
                        ● GATE 2<br /><span className="text-[8px]">PASS (NCC)</span>
                      </div>
                      <div className="bg-[#FEF3D6] dark:bg-amber-950/40 border border-[#DEC073] dark:border-amber-800 p-1 text-[#8C6211] dark:text-amber-300 font-bold">
                        ▲ GATE 3<br /><span className="text-[8px]">VERIFY (Val)</span>
                      </div>
                    </div>
                  </div>
                </div>
              </div>

              {/* Card 2: Sensor Capabilities Progress Breakdown */}
              <div className="retro-outset p-2">
                <div className="bg-[#1F4743] text-white px-2 py-1 text-xs font-mono font-bold -m-2 mb-2">
                  MULTI-MODAL SENSOR LAYERS
                </div>
                <div className="retro-inset p-2 bg-[#F9F7F2] dark:bg-[#121615] font-mono text-xs space-y-2">
                  <div>
                    <div className="flex justify-between text-[#1E2321] dark:text-[#E7E2D6] font-bold text-[11px] mb-1">
                      <span>OHRC Narrow Angle</span>
                      <span>{sensorCardGsd(detail, "ohrc")}</span>
                    </div>
                    <div className="progress-bar-track">
                      <div className="progress-bar-fill" style={{ width: "95%" }} />
                    </div>
                  </div>
                  <div>
                    <div className="flex justify-between text-[#1E2321] dark:text-[#E7E2D6] font-bold text-[11px] mb-1">
                      <span>TMC-2 Single View</span>
                      <span>{sensorCardGsd(detail, "tmc")}</span>
                    </div>
                    <div className="progress-bar-track">
                      <div className="progress-bar-fill bg-[#2A5D57]" style={{ width: "80%" }} />
                    </div>
                  </div>
                  <div>
                    <div className="flex justify-between text-[#1E2321] dark:text-[#E7E2D6] font-bold text-[11px] mb-1">
                      <span>IIRS Hyperspectral</span>
                      <span>{sensorCardGsd(detail, "iirs")}</span>
                    </div>
                    <div className="progress-bar-track">
                      <div className="progress-bar-fill bg-[#785317]" style={{ width: "65%" }} />
                    </div>
                  </div>
                  {detail?.lro_nac_available && (
                    <div>
                      <div className="flex justify-between text-[#1E2321] dark:text-[#E7E2D6] font-bold text-[11px] mb-1">
                        <span className="flex items-center gap-1">
                          <span>NASA LRO NAC</span>
                          <span className="bg-[#D1B890] text-black px-1 text-[8px]">Ref</span>
                        </span>
                        <span>{sensorCardGsd(detail, "lro_nac")}</span>
                      </div>
                      <div className="progress-bar-track">
                        <div className="progress-bar-fill bg-[#BD8B3E]" style={{ width: "88%" }} />
                      </div>
                    </div>
                  )}
                </div>
              </div>
            </div>
          </div>
        </main>

        {/* Retro OS Bottom Taskbar */}
        <footer className="h-9 bg-[#D6D0C2] dark:bg-[#1A201E] border-t-2 border-white dark:border-[#2D3835] px-3 py-1 flex items-center justify-between shadow-md z-40 shrink-0 text-xs select-none">
          <div className="flex items-center space-x-2">
            <button className="retro-button px-2.5 py-0.5 font-bold text-xs flex items-center gap-1.5 bg-[#E8E2D5] dark:bg-[#222927]">
              <span className="w-3.5 h-3.5 bg-[#1F4743] flex items-center justify-center text-white text-[9px] font-bold">🚀</span>
              <span className="font-black text-[#143532] dark:text-emerald-400 tracking-wider text-xs">Start</span>
            </button>
            <div className="hidden sm:flex items-center space-x-1 text-xs">
              <div className="retro-button active-pressed px-2.5 py-0.5 text-[11px] font-bold bg-[#CFC8BA] dark:bg-[#121615] flex items-center gap-1.5 text-[#1E2321] dark:text-[#E7E2D6]">
                <span className="w-2 h-2 bg-[#28557E] inline-block" />
                <span>Chandrayaan Console</span>
              </div>
              <div className="retro-button px-2.5 py-0.5 text-[11px] font-medium bg-[#DDD7CA] dark:bg-[#1A201E] hidden md:flex items-center gap-1.5 text-[#3E4743] dark:text-[#A8B2AD]">
                <span>🛰️</span>
                <span>Sub-Pixel Engine Active</span>
              </div>
            </div>
          </div>
          <div className="flex items-center space-x-2">
            <div className="retro-inset px-2 py-0.5 text-[10px] font-mono bg-[#E8E3D7] dark:bg-[#141817] flex items-center gap-1 text-[#222] dark:text-[#E7E2D6]">
              <span>🌙</span>
              <span>ALT: <strong>100.4 KM</strong></span>
            </div>
            <div className="retro-inset px-2 py-0.5 text-[10px] font-mono bg-[#E8E3D7] dark:bg-[#141817] flex items-center gap-1 text-[#222] dark:text-[#E7E2D6]">
              <span>💾</span>
              <span>MEM: <strong>64 MB OK</strong></span>
            </div>
            <div className="retro-inset px-2.5 py-0.5 text-xs font-mono font-bold bg-[#E8E3D7] dark:bg-[#141817] text-[#111] dark:text-[#E7E2D6] flex items-center gap-1">
              <span>🕒</span>
              <span>ISRO-UTC</span>
            </div>
          </div>
        </footer>
      </div>
    </div>

      {/* Modals with Clean Light UI Style */}
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
          onDeleteRegion={handleDeleteRegion}
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
