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
import { getCurrentUser, logout, type AuthUser } from "@/lib/auth";

type View = "registration" | "linked-cursor" | "map";

interface Props {
  onBackToHero?: () => void;
  onLogout?: () => void;
}

export default function Console({ onBackToHero, onLogout }: Props = {}) {
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

  // Authenticated User Profile & Dropdown
  const [currentUser, setCurrentUser] = useState<AuthUser | null>(null);
  const [profileMenuOpen, setProfileMenuOpen] = useState(false);
  const profileMenuRef = useRef<HTMLDivElement>(null);

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
      window.location.href = "/";
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

  const openVaultWithFilter = (filter: PayloadFilter) => {
    setVaultInitialFilter(filter);
    setVaultOpen(true);
  };

  const currentFootprint = detail ? footprintSizeKm(detail.bounds) : null;

  if (error) {
    return (
      <div className="flex h-screen items-center justify-center bg-[#f4f6fb] px-6">
        <div className="max-w-md rounded-2xl border border-red-200 bg-white p-6 text-center shadow-lg">
          <p className="text-sm font-semibold uppercase tracking-wider text-red-600">
            Archive Connection Failed
          </p>
          <p className="mt-2 text-xs text-slate-600">{error}</p>
          <p className="mt-4 font-mono text-[10px] text-slate-400">
            Confirm FastAPI server is active on port 8000.
          </p>
        </div>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-[#f4f6fb] font-sans text-slate-800 antialiased flex">
      {/* Toast Notification */}
      {toastMessage && (
        <div className="fixed bottom-6 right-6 z-[9999] flex items-center gap-2.5 rounded-xl border border-indigo-100 bg-white px-5 py-3 text-sm text-slate-800 shadow-xl animate-fade-in">
          <span className="h-2 w-2 rounded-full bg-[#4F46E5]" />
          <span>{toastMessage}</span>
        </div>
      )}

      {/* ======================================================== */}
      {/* 1. LEFT SIDEBAR: Brand Logo, Main Menu, Region List      */}
      {/* ======================================================== */}
      <aside className="w-64 bg-white border-r border-slate-200/80 flex flex-col justify-between shrink-0 min-h-screen">
        <div>
          {/* Brand Header */}
          <div className="p-6 pb-5 flex items-center justify-between border-b border-slate-100">
            <div className="flex items-center gap-2.5">
              <div className="flex h-9 w-9 items-center justify-center rounded-xl bg-[#4F46E5] text-white font-black text-base shadow-sm">
                c
              </div>
              <div>
                <h1 className="text-base font-extrabold tracking-tight text-slate-900 leading-none">
                  chandrayaan
                </h1>
                <span className="text-[10px] font-medium text-slate-400">Cross-Match Console</span>
              </div>
            </div>
          </div>

          {/* Navigation Section */}
          <div className="p-4 space-y-6">
            <div>
              <span className="px-3 text-[11px] font-bold uppercase tracking-wider text-slate-400 block mb-2">
                Menu
              </span>
              <nav className="space-y-1">
                {[
                  { id: "registration", label: "Dashboard QA", icon: "⊞" },
                  { id: "linked-cursor", label: "Linked Cursor", icon: "⊙" },
                  { id: "map", label: "Planetary Map", icon: "☵" },
                ].map((item) => {
                  const active = view === item.id;
                  return (
                    <button
                      key={item.id}
                      onClick={() => { setView(item.id as View); scrollToArena(); }}
                      className={`group flex w-full items-center justify-between rounded-xl px-3.5 py-2.5 text-xs font-semibold transition-all ${
                        active
                          ? "bg-[#EEF2FF] text-[#4F46E5] border-l-4 border-[#4F46E5]"
                          : "text-slate-600 hover:bg-slate-50 hover:text-slate-900 border-l-4 border-transparent"
                      }`}
                    >
                      <div className="flex items-center gap-3">
                        <span className={`text-sm ${active ? "text-[#4F46E5]" : "text-slate-400 group-hover:text-slate-600"}`}>
                          {item.icon}
                        </span>
                        <span>{item.label}</span>
                      </div>
                      {active && (
                        <span className="h-1.5 w-1.5 rounded-full bg-[#4F46E5]" />
                      )}
                    </button>
                  );
                })}

                <button
                  onClick={() => openVaultWithFilter("all")}
                  className="group flex w-full items-center gap-3 rounded-xl px-3.5 py-2.5 text-xs font-semibold text-slate-600 hover:bg-slate-50 hover:text-slate-900 transition border-l-4 border-transparent"
                >
                  <span className="text-sm text-slate-400 group-hover:text-slate-600">▤</span>
                  <span>Archive Vault</span>
                </button>

                <button
                  onClick={() => setTheoryModalOpen(true)}
                  className="group flex w-full items-center gap-3 rounded-xl px-3.5 py-2.5 text-xs font-semibold text-slate-600 hover:bg-slate-50 hover:text-slate-900 transition border-l-4 border-transparent"
                >
                  <span className="text-sm text-slate-400 group-hover:text-slate-600">📖</span>
                  <span>Methodology</span>
                </button>
              </nav>
            </div>

            {/* Region Directory */}
            <div>
              <div className="flex items-center justify-between px-3 mb-2">
                <span className="text-[11px] font-bold uppercase tracking-wider text-slate-400 block">
                  Regions ({filteredTriplets.length})
                </span>
                <div className="flex items-center gap-1">
                  <button
                    onClick={handlePrev}
                    className="flex h-5 w-5 items-center justify-center rounded border border-slate-200 text-xs text-slate-500 hover:bg-slate-100 transition"
                    title="Previous"
                  >
                    ‹
                  </button>
                  <button
                    onClick={handleNext}
                    className="flex h-5 w-5 items-center justify-center rounded border border-slate-200 text-xs text-slate-500 hover:bg-slate-100 transition"
                    title="Next"
                  >
                    ›
                  </button>
                </div>
              </div>

              <div className="space-y-1 max-h-[260px] overflow-y-auto pr-1">
                {filteredTriplets.map((t, i) => {
                  const active = t.id === selectedId;
                  const { widthKm, heightKm } = footprintSizeKm(t.bounds);
                  return (
                    <button
                      key={t.id}
                      onClick={() => setSelectedId(t.id)}
                      className={`flex w-full items-center justify-between rounded-xl px-3 py-2 text-left text-xs transition-all ${
                        active
                          ? "bg-slate-100 font-bold text-slate-900 shadow-sm"
                          : "text-slate-600 hover:bg-slate-50"
                      }`}
                    >
                      <div className="truncate">
                        <span className="text-[10px] font-mono text-slate-400 mr-1.5">
                          {String(i + 1).padStart(2, "0")}.
                        </span>
                        <span>{t.id}</span>
                        <span className="block text-[10px] text-slate-400 font-normal">
                          {widthKm.toFixed(1)} × {heightKm.toFixed(1)} km
                        </span>
                      </div>
                      {t.dem_available && (
                        <span className="rounded-md bg-indigo-50 px-1.5 py-0.5 text-[9px] font-bold text-[#4F46E5] shrink-0">
                          DEM
                        </span>
                      )}
                    </button>
                  );
                })}
              </div>
            </div>
          </div>
        </div>

      </aside>

      {/* ======================================================== */}
      {/* 2. MAIN WORKSPACE: Header Bar & Dashboard Content        */}
      {/* ======================================================== */}
      <div className="flex-1 flex flex-col min-w-0">
        {/* Top Header Bar */}
        <header className="h-16 bg-white border-b border-slate-200/80 px-8 flex items-center justify-between gap-4 shrink-0">
          {/* Search Input */}
          <div className="relative w-80">
            <svg className="absolute left-3.5 top-1/2 -translate-y-1/2 h-4 w-4 text-slate-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0z" />
            </svg>
            <input
              type="text"
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
              placeholder="Search for regions, coordinates..."
              className="w-full pl-10 pr-4 py-2 bg-slate-50 border border-slate-200/80 rounded-xl text-xs text-slate-700 placeholder-slate-400 focus:outline-none focus:ring-2 focus:ring-[#4F46E5]/20 focus:border-[#4F46E5] transition"
            />
          </div>

          {/* Right Header Controls */}
          <div className="flex items-center gap-3">
            <button
              onClick={() => {
                if (onBackToHero) onBackToHero();
                else window.location.href = "/";
              }}
              className="rounded-xl border border-slate-200 bg-white px-3.5 py-2 text-xs font-semibold text-slate-600 hover:bg-slate-50 transition shadow-sm flex items-center gap-1.5"
              title="Return to Mission Overview"
            >
              <span>←</span>
              <span>Back</span>
            </button>

            <button
              onClick={() => openVaultWithFilter("all")}
              className="rounded-xl border border-slate-200 bg-white px-4 py-2 text-xs font-semibold text-slate-700 hover:bg-slate-50 transition shadow-sm flex items-center gap-2"
              title="Browse all multi-sensor lunar datasets"
            >
              <span className="h-2 w-2 rounded-full bg-emerald-500" />
              <span>{triplets.length} Datasets</span>
            </button>

            <div className="h-8 w-px bg-slate-200 mx-1" />

            {/* Profile Pill with Interactive Dropdown */}
            <div className="relative" ref={profileMenuRef}>
              <button
                type="button"
                onClick={() => setProfileMenuOpen((prev) => !prev)}
                className="flex items-center gap-2.5 rounded-xl border border-slate-200/80 bg-white p-1.5 pr-3 hover:bg-slate-50 transition shadow-sm focus:outline-none focus:ring-2 focus:ring-[#4F46E5]/20"
                aria-haspopup="true"
                aria-expanded={profileMenuOpen}
              >
                <div className="h-8 w-8 rounded-lg bg-gradient-to-tr from-indigo-500 to-[#4F46E5] text-white font-bold text-xs flex items-center justify-center shadow-sm">
                  {currentUser?.name ? currentUser.name.slice(0, 2).toUpperCase() : "IS"}
                </div>
                <div className="hidden sm:block text-left">
                  <span className="text-xs font-bold text-slate-800 block leading-tight truncate max-w-[120px]">
                    {currentUser?.name || "ISRO Pilot"}
                  </span>
                  <span className="text-[10px] text-slate-400 block">Operator</span>
                </div>
                <svg
                  className={`h-3.5 w-3.5 text-slate-400 transition-transform duration-200 ${profileMenuOpen ? "rotate-180" : ""}`}
                  fill="none"
                  stroke="currentColor"
                  viewBox="0 0 24 24"
                >
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 9l-7 7-7-7" />
                </svg>
              </button>

              {/* Dropdown Menu */}
              {profileMenuOpen && (
                <div className="absolute right-0 mt-2 w-64 rounded-2xl border border-slate-200/80 bg-white p-2 shadow-2xl z-50 animate-fade-in">
                  <div className="p-3 border-b border-slate-100">
                    <span className="text-xs font-bold text-slate-900 block truncate">
                      {currentUser?.name || "ISRO Flight Operator"}
                    </span>
                    <span className="text-[11px] text-slate-500 block truncate">
                      {currentUser?.email || "flight.ops@isro.gov.in"}
                    </span>
                    <span className="mt-2 inline-flex items-center gap-1 rounded-md bg-emerald-50 px-2 py-0.5 text-[10px] font-bold text-emerald-700">
                      <span className="h-1.5 w-1.5 rounded-full bg-emerald-500" />
                      Active Session
                    </span>
                  </div>

                  <div className="py-1 space-y-0.5">
                    <button
                      onClick={() => {
                        setProfileMenuOpen(false);
                        openVaultWithFilter("all");
                      }}
                      className="w-full flex items-center gap-2.5 rounded-xl px-3 py-2 text-xs font-semibold text-slate-700 hover:bg-slate-50 transition text-left"
                    >
                      <span className="text-slate-400">▤</span>
                      <span>Archive Vault ({triplets.length} Datasets)</span>
                    </button>

                    <button
                      onClick={() => {
                        setProfileMenuOpen(false);
                        setTheoryModalOpen(true);
                      }}
                      className="w-full flex items-center gap-2.5 rounded-xl px-3 py-2 text-xs font-semibold text-slate-700 hover:bg-slate-50 transition text-left"
                    >
                      <span className="text-slate-400">📖</span>
                      <span>Methodology Reference</span>
                    </button>

                    <button
                      onClick={() => {
                        setProfileMenuOpen(false);
                        if (onBackToHero) onBackToHero();
                        else window.location.href = "/";
                      }}
                      className="w-full flex items-center gap-2.5 rounded-xl px-3 py-2 text-xs font-semibold text-slate-700 hover:bg-slate-50 transition text-left"
                    >
                      <span className="text-slate-400">🌐</span>
                      <span>Mission Overview</span>
                    </button>
                  </div>

                  <div className="pt-1 mt-1 border-t border-slate-100">
                    <button
                      onClick={handleUserLogout}
                      className="w-full flex items-center gap-2.5 rounded-xl px-3 py-2 text-xs font-bold text-red-600 hover:bg-red-50 transition text-left"
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
        <main className="p-8 space-y-6 overflow-y-auto">
          {/* Page Title & Subtitle */}
          <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4">
            <div>
              <h2 className="text-2xl font-bold tracking-tight text-slate-900">
                Dashboard
              </h2>
              <p className="text-xs text-slate-500 mt-0.5">
                A quick view of your planetary cross-matching and multi-sensor registration pipeline.
              </p>
            </div>
            <div className="flex items-center gap-2">
              <button
                onClick={() => openVaultWithFilter("all")}
                className="rounded-xl border border-slate-200 bg-white px-4 py-2 text-xs font-bold text-slate-700 hover:bg-slate-50 transition shadow-sm"
              >
                Browse All Regions
              </button>
              {detail && (
                <button
                  onClick={() => handleOpenDossierModal(detail)}
                  className="rounded-xl bg-[#4F46E5] px-4 py-2 text-xs font-bold text-white hover:bg-[#4338CA] transition shadow-sm flex items-center gap-1.5"
                >
                  <span>Dossier Report</span>
                  <span>↗</span>
                </button>
              )}
            </div>
          </div>

          <RegistrationLauncher />

          {/* ======================================================== */}
          {/* 3. ROW 1: TOP 4 KPI METRIC CARDS                         */}
          {/* (Exact layout & visual styling from reference image!)     */}
          {/* ======================================================== */}
          <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
            {/* Card 1: Solid Vibrant Indigo Accent Card */}
            <div className="rounded-2xl bg-[#4F46E5] p-5 text-white shadow-sm flex flex-col justify-between">
              <div className="flex items-center justify-between">
                <span className="text-xs font-medium text-indigo-100">Sub-Pixel Status</span>
                <span className="flex h-6 w-6 items-center justify-center rounded-full bg-white/20 text-xs">
                  ↗
                </span>
              </div>
              <div className="my-3">
                <div className="text-3xl font-extrabold tracking-tight">
                  {metrics?.sub_pixel_accurate ? "< 0.50" : "Active"}
                  {metrics?.sub_pixel_accurate && <span className="text-sm font-semibold ml-1">px</span>}
                </div>
              </div>
              <div className="flex items-center gap-1.5 text-[11px] text-indigo-100">
                <span className="h-1.5 w-1.5 rounded-full bg-emerald-300" />
                <span>Verified sub-pixel alignment</span>
              </div>
            </div>

            {/* Card 2: RMSE Reprojection Accuracy */}
            <div className="rounded-2xl border border-slate-200/80 bg-white p-5 shadow-sm flex flex-col justify-between">
              <div className="flex items-center justify-between">
                <span className="text-xs font-medium text-slate-500">Planar Fit RMSE</span>
                <span className="flex h-6 w-6 items-center justify-center rounded-full bg-slate-100 text-xs text-slate-600">
                  ↗
                </span>
              </div>
              <div className="my-3">
                <div className="text-3xl font-extrabold tracking-tight text-slate-900">
                  {metrics?.rmse_px != null ? metrics.rmse_px.toFixed(3) : "—"}
                  {metrics?.rmse_px != null && <span className="text-sm font-semibold text-slate-500 ml-1">px</span>}
                </div>
              </div>
              <div className="flex items-center gap-1.5 text-[11px] text-slate-500">
                <span>Threshold &lt; 0.50 px target</span>
              </div>
            </div>

            {/* Card 3: Inlier Correspondences */}
            <div className="rounded-2xl border border-slate-200/80 bg-white p-5 shadow-sm flex flex-col justify-between">
              <div className="flex items-center justify-between">
                <span className="text-xs font-medium text-slate-500">Verified Inliers</span>
                <span className="flex h-6 w-6 items-center justify-center rounded-full bg-slate-100 text-xs text-slate-600">
                  ↗
                </span>
              </div>
              <div className="my-3">
                <div className="text-3xl font-extrabold tracking-tight text-slate-900">
                  {metrics?.num_inliers ?? 0}
                  <span className="text-sm font-semibold text-slate-500 ml-1">matches</span>
                </div>
              </div>
              <div className="flex items-center gap-1.5 text-[11px] text-slate-500">
                <span>{metrics?.inlier_ratio != null ? (metrics.inlier_ratio * 100).toFixed(1) : 0}% inlier ratio</span>
              </div>
            </div>

            {/* Card 4: Spatial Coverage Score */}
            <div className="rounded-2xl border border-slate-200/80 bg-white p-5 shadow-sm flex flex-col justify-between">
              <div className="flex items-center justify-between">
                <span className="text-xs font-medium text-slate-500">Spatial Coverage</span>
                <span className="flex h-6 w-6 items-center justify-center rounded-full bg-slate-100 text-xs text-slate-600">
                  ↗
                </span>
              </div>
              <div className="my-3">
                <div className="text-3xl font-extrabold tracking-tight text-slate-900">
                  {metrics?.combined_coverage_score != null ? `${(metrics.combined_coverage_score * 100).toFixed(0)}%` : "—"}
                </div>
              </div>
              <div className="flex items-center gap-1.5 text-[11px] text-slate-500">
                <span>10×10 Grid Non-Max Suppression</span>
              </div>
            </div>
          </div>

          {/* ======================================================== */}
          {/* 4. ROW 2: PRIMARY INSPECTION ARENA + TELEMETRY SIDEBAR   */}
          {/* ======================================================== */}
          <div ref={arenaRef} className="grid grid-cols-1 lg:grid-cols-12 gap-6 scroll-mt-6">
            
            {/* Center Main Stage (8 cols) */}
            <div className="lg:col-span-8 rounded-2xl border border-slate-200/80 bg-white p-6 shadow-sm flex flex-col justify-between">
              <div>
                {/* Viewport Top Controls */}
                <div className="flex flex-wrap items-center justify-between gap-3 border-b border-slate-100 pb-4 mb-5">
                  <div className="flex items-center gap-3">
                    <span className="text-sm font-bold text-slate-900">
                      {detail?.id ?? "Select a region"}
                    </span>
                    {currentFootprint && (
                      <span className="rounded-md bg-slate-100 px-2 py-0.5 text-xs text-slate-600 font-mono">
                        {currentFootprint.widthKm.toFixed(1)} × {currentFootprint.heightKm.toFixed(1)} km
                      </span>
                    )}
                  </div>

                  {/* Mode Pill Switcher */}
                  <div className="flex items-center bg-slate-100 p-1 rounded-xl gap-1">
                    {[
                      { id: "registration", label: "Registration QA" },
                      { id: "linked-cursor", label: "Linked Cursor" },
                      { id: "map", label: "Planetary Map" },
                    ].map((m) => (
                      <button
                        key={m.id}
                        onClick={() => setView(m.id as View)}
                        className={`rounded-lg px-3 py-1.5 text-xs font-semibold transition-all ${
                          view === m.id
                            ? "bg-white text-slate-900 shadow-sm"
                            : "text-slate-500 hover:text-slate-800"
                        }`}
                      >
                        {m.label}
                      </button>
                    ))}
                  </div>
                </div>

                {/* Viewport Canvas Area */}
                <div className="relative min-h-[460px]">
                  {loading && (
                    <div className="flex h-full min-h-[400px] items-center justify-center text-sm text-slate-400">
                      Loading sensor datasets…
                    </div>
                  )}

                  {!loading && !detail && (
                    <div className="flex h-full min-h-[400px] items-center justify-center text-sm text-slate-400">
                      No region selected.
                    </div>
                  )}

                  {/* Registration QA View */}
                  {detail && view === "registration" && (
                    <div className="flex flex-col gap-4">
                      <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                        {[
                          { src: `/images/registered/${detail.id}/registered_ohrc.png`, fallback: `/images/ohrc/${detail.id}`, label: "Warped OHRC", tag: "0.25m Primary" },
                          { src: `/images/registered/${detail.id}/blend_overlay.png`, fallback: `/images/tmc/${detail.id}`, label: "Blend Overlay", tag: "50% Cross-Fade" },
                          { src: `/images/registered/${detail.id}/checkerboard_qa.png`, fallback: `/images/tmc/${detail.id}`, label: "Checkerboard QA", tag: "Continuity Verification" },
                        ].map((img, idx) => (
                          <div key={idx} className="flex flex-col rounded-xl border border-slate-200/70 overflow-hidden bg-slate-50">
                            <div className="relative aspect-square overflow-hidden bg-black">
                              {/* eslint-disable-next-line @next/next/no-img-element */}
                              <img
                                src={imageUrl(img.src)}
                                alt={img.label}
                                className="h-full w-full object-cover transition-transform duration-300 hover:scale-105"
                                onError={(e) => {
                                  (e.currentTarget as HTMLImageElement).src = imageUrl(img.fallback);
                                }}
                              />
                            </div>
                            <div className="p-3 bg-white">
                              <span className="text-xs font-bold text-slate-900 block">{img.label}</span>
                              <span className="text-[10px] text-slate-500">{img.tag}</span>
                            </div>
                          </div>
                        ))}
                      </div>
                    </div>
                  )}

                  {/* Linked Cursor View */}
                  {detail && view === "linked-cursor" && (
                    <div className="h-full min-h-[440px]">
                      <LinkedCursorPanel tripletId={detail.id} points={matches} />
                    </div>
                  )}

                  {/* Map View */}
                  {detail && view === "map" && (
                    <div className="h-full min-h-[440px] rounded-xl overflow-hidden border border-slate-200">
                      <MapPanel triplet={detail} iirsOverlay={iirsOverlay} />
                    </div>
                  )}
                </div>
              </div>

              {/* Viewport Bottom Footer */}
              <div className="mt-4 pt-4 border-t border-slate-100 flex flex-wrap items-center justify-between gap-3 text-xs text-slate-500">
                <div className="flex items-center gap-2">
                  <span className="h-2 w-2 rounded-full bg-emerald-500" />
                  <span>Aligned with <strong>{metrics?.num_inliers ?? 0} inlier ties</strong></span>
                </div>
                {detail && (
                  <button
                    onClick={() => handleOpenDossierModal(detail)}
                    className="text-[#4F46E5] font-bold hover:underline"
                  >
                    View Comprehensive Dossier Report →
                  </button>
                )}
              </div>
            </div>

            {/* Right Telemetry Column (4 cols) */}
            <div className="lg:col-span-4 space-y-6">
              {/* Card 1: Selected Region Footprint */}
              <div className="rounded-2xl border border-slate-200/80 bg-white p-5 shadow-sm">
                <h3 className="text-xs font-bold uppercase tracking-wider text-slate-400 mb-3">
                  Selected Footprint Details
                </h3>
                <div className="space-y-3 text-xs">
                  <div className="flex justify-between border-b border-slate-100 pb-2">
                    <span className="text-slate-500">Longitude Extent</span>
                    <span className="font-semibold text-slate-900">
                      {detail ? `${detail.bounds.west_lon.toFixed(2)}° to ${detail.bounds.east_lon.toFixed(2)}°` : "—"}
                    </span>
                  </div>
                  <div className="flex justify-between border-b border-slate-100 pb-2">
                    <span className="text-slate-500">Latitude Extent</span>
                    <span className="font-semibold text-slate-900">
                      {detail ? `${detail.bounds.south_lat.toFixed(2)}° to ${detail.bounds.north_lat.toFixed(2)}°` : "—"}
                    </span>
                  </div>
                  <div className="flex justify-between border-b border-slate-100 pb-2">
                    <span className="text-slate-500">Terrain Dimensions</span>
                    <span className="font-semibold text-slate-900">
                      {currentFootprint ? `${currentFootprint.widthKm.toFixed(1)} × ${currentFootprint.heightKm.toFixed(1)} km` : "—"}
                    </span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-slate-500">Topographic DEM</span>
                    <span className={`font-semibold ${detail?.dem_available ? "text-emerald-600" : "text-slate-400"}`}>
                      {detail?.dem_available ? "Available (TMC DTM)" : "Interpolated"}
                    </span>
                  </div>
                </div>
              </div>

              {/* Card 2: Sensor Capabilities Progress Breakdown */}
              <div className="rounded-2xl border border-slate-200/80 bg-white p-5 shadow-sm">
                <h3 className="text-xs font-bold uppercase tracking-wider text-slate-400 mb-3">
                  Multi-Modal Sensor Layers
                </h3>
                <div className="space-y-3 text-xs">
                  <div>
                    <div className="flex justify-between text-slate-700 font-semibold mb-1">
                      <span>OHRC Narrow Angle</span>
                      <span>0.25 m/px</span>
                    </div>
                    <div className="w-full bg-slate-100 h-2 rounded-full overflow-hidden">
                      <div className="bg-[#4F46E5] h-full rounded-full" style={{ width: "95%" }} />
                    </div>
                  </div>
                  <div>
                    <div className="flex justify-between text-slate-700 font-semibold mb-1">
                      <span>TMC-2 Terrain Stereo</span>
                      <span>4.0 m/px</span>
                    </div>
                    <div className="w-full bg-slate-100 h-2 rounded-full overflow-hidden">
                      <div className="bg-indigo-400 h-full rounded-full" style={{ width: "80%" }} />
                    </div>
                  </div>
                  <div>
                    <div className="flex justify-between text-slate-700 font-semibold mb-1">
                      <span>IIRS Hyperspectral</span>
                      <span>70 m/px</span>
                    </div>
                    <div className="w-full bg-slate-100 h-2 rounded-full overflow-hidden">
                      <div className="bg-cyan-500 h-full rounded-full" style={{ width: "65%" }} />
                    </div>
                  </div>
                </div>
              </div>

              {/* Card 3: Quick Action Launchers */}
              <div className="rounded-2xl border border-slate-200/80 bg-white p-5 shadow-sm space-y-2.5">
                <h3 className="text-xs font-bold uppercase tracking-wider text-slate-400 mb-2">
                  Actions
                </h3>
                <button
                  onClick={() => openVaultWithFilter("all")}
                  className="w-full rounded-xl border border-slate-200 bg-slate-50 py-2.5 text-xs font-bold text-slate-700 hover:bg-slate-100 transition text-center block"
                >
                  Inspect Full Dataset Vault ↗
                </button>
                <button
                  onClick={() => setTheoryModalOpen(true)}
                  className="w-full rounded-xl border border-slate-200 bg-slate-50 py-2.5 text-xs font-bold text-slate-700 hover:bg-slate-100 transition text-center block"
                >
                  View Mathematical Framework ↗
                </button>
              </div>
            </div>

          </div>
        </main>
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
