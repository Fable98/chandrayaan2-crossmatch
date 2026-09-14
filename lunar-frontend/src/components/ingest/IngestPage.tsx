"use client";

import React, { useCallback, useEffect, useRef, useState } from "react";
import Link from "next/link";
import { useRouter } from "next/navigation";
import { api } from "@/lib/api";
import { footprintSizeKm } from "@/lib/geo";
import type { TripletSummary } from "@/lib/types";
import { isAuthenticated, getCurrentUser, logout, type AuthUser } from "@/lib/auth";
import VaultModal from "../archive/VaultModal";
import TheoryModal from "../archive/TheoryModal";
import DropZone from "./DropZone";
import ProcessingProgress from "./ProcessingProgress";
import ResultsTable from "./ResultsTable";
import {
  uploadZips,
  pollStatus,
  getResults,
  listJobs,
  DEFAULT_CONFIG,
  type IngestConfig,
  type JobStatus,
  type IngestJobSummary,
} from "@/lib/ingest-api";

type Phase = "idle" | "queued" | "processing" | "done" | "error";

function fmtSize(bytes: number): string {
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
  return `${(bytes / 1024 / 1024).toFixed(1)} MB`;
}

export default function IngestPage() {
  const router = useRouter();

  // Core Ingestion State
  const [files, setFiles] = useState<File[]>([]);
  const [config, setConfig] = useState<IngestConfig>({ ...DEFAULT_CONFIG });
  const [showConfig, setShowConfig] = useState(false);
  const [phase, setPhase] = useState<Phase>("idle");
  const [jobId, setJobId] = useState<string | null>(null);
  const [status, setStatus] = useState<JobStatus | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [resultTriplets, setResultTriplets] = useState<Record<string, any>[]>([]);
  const [activeTab, setActiveTab] = useState<"new" | "history">("new");
  const [historyJobs, setHistoryJobs] = useState<IngestJobSummary[]>([]);
  const [historyLoading, setHistoryLoading] = useState(false);
  const [historyError, setHistoryError] = useState<string | null>(null);
  const [sessionExpired, setSessionExpired] = useState(false);
  const pollRef = useRef<number | null>(null);

  // Backend 401 details for a dead/rotated JWT ("Invalid or expired token",
  // "Authentication required", "User not found"). A stale localStorage token
  // must never leave the operator on a dead-end error banner.
  function isAuthError(msg: string): boolean {
    return /invalid or expired token|authentication required|user not found|not authenticated|\b401\b/i.test(
      msg || ""
    );
  }

  // Shell & Navigation State
  const [triplets, setTriplets] = useState<TripletSummary[]>([]);
  const [searchQuery, setSearchQuery] = useState("");
  const [currentUser, setCurrentUser] = useState<AuthUser | null>(null);
  const [isAuthed, setIsAuthed] = useState(false);
  const [authChecked, setAuthChecked] = useState(false);
  const [profileMenuOpen, setProfileMenuOpen] = useState(false);
  const profileMenuRef = useRef<HTMLDivElement>(null);

  // Modals State
  const [vaultOpen, setVaultOpen] = useState(false);
  const [theoryModalOpen, setTheoryModalOpen] = useState(false);

  useEffect(() => {
    setIsAuthed(isAuthenticated());
    setCurrentUser(getCurrentUser());
    setAuthChecked(true);
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
    logout();
    setIsAuthed(false);
    setCurrentUser(null);
    router.push("/");
  };

  // Load regions list on mount for sidebar & dataset pill
  useEffect(() => {
    api
      .listTriplets()
      .then((res) => {
        setTriplets(res.triplets || []);
      })
      .catch(() => {
        // Handled gracefully
      });
  }, []);

  const filteredTriplets = triplets.filter((t) =>
    t.id.toLowerCase().includes(searchQuery.toLowerCase())
  );

  const loadHistoryJobs = useCallback(async () => {
    setHistoryLoading(true);
    setHistoryError(null);
    try {
      const jobs = await listJobs();
      setHistoryJobs(jobs);
    } catch (err: any) {
      setHistoryError(err.message || "Failed to load past ingestion jobs");
    } finally {
      setHistoryLoading(false);
    }
  }, []);

  useEffect(() => {
    loadHistoryJobs();
  }, [loadHistoryJobs]);

  const handleSelectHistoricalJob = useCallback(async (selectedId: string) => {
    setError(null);
    try {
      const results = await getResults(selectedId);
      setJobId(selectedId);
      setResultTriplets(results.triplets || []);
      setPhase("done");
      setActiveTab("new");
    } catch (err: any) {
      setError(`Failed to fetch results for job ${selectedId}: ${err.message}`);
    }
  }, []);

  // Add files (dedup by name)
  const handleFilesSelected = useCallback((newFiles: File[]) => {
    setFiles((prev) => {
      const existing = new Set(prev.map((f) => f.name));
      const added = newFiles.filter((f) => !existing.has(f.name));
      return [...prev, ...added];
    });
  }, []);

  // Remove a file
  const removeFile = useCallback((name: string) => {
    setFiles((prev) => prev.filter((f) => f.name !== name));
  }, []);

  // Clear all files
  const clearFiles = useCallback(() => {
    setFiles([]);
    setPhase("idle");
    setJobId(null);
    setStatus(null);
    setError(null);
    setSessionExpired(false);
    setResultTriplets([]);
  }, []);

  // Start processing
  const handleStart = useCallback(async () => {
    if (files.length === 0) return;

    setPhase("queued");
    setError(null);
    setSessionExpired(false);

    try {
      const res = await uploadZips(files, config);
      setJobId(res.job_id);
      setPhase("processing");
    } catch (e: any) {
      const msg = e.message || "Upload failed";
      if (isAuthError(msg)) {
        // Stale/rotated JWT: drop it so the sign-in wall appears, and say
        // exactly what happened instead of a generic failure.
        logout();
        setCurrentUser(null);
        setSessionExpired(true);
        setError("Your session has expired. Please sign in again, then return here to retry the upload.");
      } else {
        setError(msg);
      }
      setPhase("error");
    }
  }, [files, config]);

  // Poll loop
  useEffect(() => {
    if (phase !== "processing" || !jobId) return;

    const poll = async () => {
      try {
        const s = await pollStatus(jobId);
        setStatus(s);

        if (s.status === "completed") {
          setPhase("done");
          try {
            const results = await getResults(jobId);
            setResultTriplets(results.triplets || []);
          } catch {
            // Triplets stay empty
          }
        } else if (s.status === "failed") {
          setPhase("error");
          setError(s.error || "Pipeline failed");
        }
      } catch {
        // Transient error, keep polling
      }
    };

    poll();
    const id = window.setInterval(poll, 1500);
    pollRef.current = id;

    return () => {
      if (pollRef.current !== null) {
        window.clearInterval(pollRef.current);
      }
    };
  }, [phase, jobId]);

  const totalSize = files.reduce((s, f) => s + f.size, 0);
  const isProcessing = phase === "processing" || phase === "queued";

  return (
    <div className="min-h-screen bg-[#f4f6fb] text-slate-800 flex">
      {/* ======================================================== */}
      {/* 1. LEFT SIDEBAR: Brand Logo, Main Menu, Region List      */}
      {/* ======================================================== */}
      <aside className="w-64 bg-white border-r border-slate-200/80 flex flex-col justify-between shrink-0 min-h-screen">
        <div>
          {/* Brand Header */}
          <div className="p-6 pb-5 flex items-center justify-between border-b border-slate-100">
            <Link href="/" className="flex items-center gap-2.5 group">
              <div className="flex h-9 w-9 items-center justify-center rounded-xl bg-[#4F46E5] text-white font-black text-base shadow-sm group-hover:scale-105 transition-transform">
                c
              </div>
              <div>
                <h1 className="text-base font-extrabold tracking-tight text-slate-900 leading-none">
                  chandrayaan
                </h1>
                <span className="text-[10px] font-medium text-slate-400">Cross-Match Console</span>
              </div>
            </Link>
          </div>

          {/* Navigation Section */}
          <div className="p-4 space-y-6">
            <div>
              <span className="px-3 text-[11px] font-bold uppercase tracking-wider text-slate-400 block mb-2">
                Menu
              </span>
              <nav className="space-y-1">
                <Link
                  href="/?view=console"
                  className="group flex w-full items-center justify-between rounded-xl px-3.5 py-2.5 text-xs font-semibold text-slate-600 hover:bg-slate-50 hover:text-slate-900 transition border-l-4 border-transparent"
                >
                  <div className="flex items-center gap-3">
                    <span className="text-sm text-slate-400 group-hover:text-slate-600">
                      ⊞
                    </span>
                    <span>Dashboard QA</span>
                  </div>
                </Link>

                <div
                  className="group flex w-full items-center justify-between rounded-xl px-3.5 py-2.5 text-xs font-semibold bg-[#EEF2FF] text-[#4F46E5] border-l-4 border-[#4F46E5]"
                >
                  <div className="flex items-center gap-3">
                    <span className="text-sm text-[#4F46E5]">
                      ⚡
                    </span>
                    <span>Ingest &amp; Prepare</span>
                  </div>
                  <span className="h-1.5 w-1.5 rounded-full bg-[#4F46E5]" />
                </div>

                <Link
                  href="/?view=console&subview=linked-cursor"
                  className="group flex w-full items-center justify-between rounded-xl px-3.5 py-2.5 text-xs font-semibold text-slate-600 hover:bg-slate-50 hover:text-slate-900 transition border-l-4 border-transparent"
                >
                  <div className="flex items-center gap-3">
                    <span className="text-sm text-slate-400 group-hover:text-slate-600">
                      ⊙
                    </span>
                    <span>Linked Cursor</span>
                  </div>
                </Link>

                <Link
                  href="/?view=console&subview=map"
                  className="group flex w-full items-center justify-between rounded-xl px-3.5 py-2.5 text-xs font-semibold text-slate-600 hover:bg-slate-50 hover:text-slate-900 transition border-l-4 border-transparent"
                >
                  <div className="flex items-center gap-3">
                    <span className="text-sm text-slate-400 group-hover:text-slate-600">
                      ☵
                    </span>
                    <span>Planetary Map</span>
                  </div>
                </Link>

                <button
                  type="button"
                  onClick={() => setVaultOpen(true)}
                  className="group flex w-full items-center gap-3 rounded-xl px-3.5 py-2.5 text-xs font-semibold text-slate-600 hover:bg-slate-50 hover:text-slate-900 transition border-l-4 border-transparent"
                >
                  <span className="text-sm text-slate-400 group-hover:text-slate-600">▤</span>
                  <span>Archive Vault</span>
                </button>

                <button
                  type="button"
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
              </div>

              <div className="space-y-1 max-h-[260px] overflow-y-auto pr-1">
                {filteredTriplets.map((t, i) => {
                  const { widthKm, heightKm } = footprintSizeKm(t.bounds);
                  return (
                    <Link
                      key={t.id}
                      href={`/?view=console&region=${t.id}`}
                      className="flex w-full items-center justify-between rounded-xl px-3 py-2 text-left text-xs transition-all text-slate-600 hover:bg-slate-50"
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
                      <div className="flex items-center gap-1 shrink-0">
                        {t.dem_available && (
                          <span className="rounded-md bg-indigo-50 px-1.5 py-0.5 text-[9px] font-bold text-[#4F46E5]">
                            DEM
                          </span>
                        )}
                        {t.lro_nac_available && (
                          <span className="rounded-md bg-amber-50 px-1.5 py-0.5 text-[9px] font-bold text-amber-700">
                            LRO
                          </span>
                        )}
                      </div>
                    </Link>
                  );
                })}
              </div>
            </div>
          </div>
        </div>
      </aside>

      {/* ======================================================== */}
      {/* 2. MAIN WORKSPACE: Header Bar & Ingest Content           */}
      {/* ======================================================== */}
      <div className="flex-1 flex flex-col min-w-0">
        {/* Top Header Bar */}
        <header className="h-16 bg-white border-b border-slate-200/80 px-8 flex items-center justify-between gap-4 shrink-0">
          {/* Search Input */}
          <div className="relative w-80">
            <svg
              className="absolute left-3.5 top-1/2 -translate-y-1/2 h-4 w-4 text-slate-400"
              fill="none"
              stroke="currentColor"
              viewBox="0 0 24 24"
            >
              <path
                strokeLinecap="round"
                strokeLinejoin="round"
                strokeWidth={2}
                d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0z"
              />
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
              onClick={() => router.push("/")}
              className="rounded-xl border border-slate-200 bg-white px-3.5 py-2 text-xs font-semibold text-slate-600 hover:bg-slate-50 transition shadow-sm flex items-center gap-1.5"
              title="Return to Mission Overview"
            >
              <span>←</span>
              <span>Back</span>
            </button>

            <button
              onClick={() => setVaultOpen(true)}
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
                  {currentUser?.name ? currentUser.name.slice(0, 2).toUpperCase() : "SH"}
                </div>
                <div className="hidden sm:block text-left">
                  <span className="text-xs font-bold text-slate-800 block leading-tight truncate max-w-[120px]">
                    {currentUser?.name || "Shresth"}
                  </span>
                  <span className="text-[10px] text-slate-400 block">Operator</span>
                </div>
                <svg
                  className={`h-3.5 w-3.5 text-slate-400 transition-transform duration-200 ${
                    profileMenuOpen ? "rotate-180" : ""
                  }`}
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
                        setVaultOpen(true);
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
                        router.push("/");
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

        {/* Ingest Main Content */}
        <main className="p-8 space-y-6 overflow-y-auto">
          {/* Top Title & Subtitle + Action Tabs */}
          <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4">
            <div>
              <h2 className="text-2xl font-bold tracking-tight text-slate-900">
                Ingest &amp; Prepare
              </h2>
              <p className="text-xs text-slate-500 mt-0.5">
                Drop your PRADAN zip files below to automatically discover, match, and
                process Chandrayaan-2 OHRC + TMC-2 + IIRS triplets.
              </p>
            </div>

            {/* Tab Controls & Direct Link to Dashboard */}
            <div className="flex items-center gap-2.5 shrink-0 flex-wrap">
              <button
                type="button"
                onClick={() => setActiveTab("new")}
                className={`rounded-xl px-4 py-2 text-xs font-bold transition shadow-sm ${
                  activeTab === "new"
                    ? "bg-[#4F46E5] text-white shadow-indigo-200"
                    : "border border-slate-200 bg-white text-slate-700 hover:bg-slate-50"
                }`}
              >
                ＋ New Batch Ingestion
              </button>

              <button
                type="button"
                onClick={() => {
                  setActiveTab("history");
                  loadHistoryJobs();
                }}
                className={`flex items-center gap-2 rounded-xl px-4 py-2 text-xs font-bold transition shadow-sm ${
                  activeTab === "history"
                    ? "bg-[#4F46E5] text-white shadow-indigo-200"
                    : "border border-slate-200 bg-white text-slate-700 hover:bg-slate-50"
                }`}
              >
                <span>📂 Previous Ingestion Runs</span>
                {historyJobs.length > 0 && (
                  <span
                    className={`rounded-full px-2 py-0.5 text-[10px] font-bold ${
                      activeTab === "history"
                        ? "bg-white/20 text-white"
                        : "bg-slate-100 text-slate-600"
                    }`}
                  >
                    {historyJobs.length}
                  </span>
                )}
              </button>

              <Link
                href="/?view=console"
                className="rounded-xl border border-slate-200 bg-white px-3.5 py-2 text-xs font-semibold text-slate-700 hover:bg-slate-50 transition shadow-sm flex items-center gap-1.5"
              >
                <span>Open Dashboard</span>
                <span>↗</span>
              </Link>
            </div>
          </div>

          {/* Auth Guard Check */}
          {!authChecked ? (
            <div className="rounded-2xl border border-slate-200/80 bg-white p-12 text-center shadow-sm max-w-xl mx-auto space-y-4">
              <div className="inline-block h-8 w-8 animate-spin rounded-full border-4 border-[#4F46E5] border-r-transparent" />
              <p className="text-xs text-slate-500 font-medium">Verifying credentials...</p>
            </div>
          ) : !isAuthed ? (
            <div className="rounded-2xl border border-slate-200/80 bg-white p-12 text-center shadow-sm max-w-xl mx-auto space-y-4">
              <div className="text-4xl">🔐</div>
              <h3 className="text-base font-bold text-slate-900">
                Sign in Required for Ingestion
              </h3>
              <p className="text-xs text-slate-500 leading-relaxed">
                The ingest pipeline writes to disk and launches processing jobs, so uploads require an
                authenticated operator session. Please sign in from the home page, then return here.
              </p>
              <div className="pt-2">
                <Link
                  href="/"
                  className="inline-flex items-center gap-2 rounded-xl bg-[#4F46E5] px-5 py-2.5 text-xs font-bold text-white shadow-sm hover:bg-[#4338CA] transition"
                >
                  Go to Sign In
                </Link>
              </div>
            </div>
          ) : (
            <>
              {/* TAB 1: NEW BATCH INGESTION */}
              {activeTab === "new" && (
                <section className="rounded-2xl border border-slate-200/80 bg-white p-5 shadow-sm md:p-6 space-y-6">
                  {/* Card Header Bar */}
                  <div className="flex flex-col justify-between gap-3 border-b border-slate-100 pb-5 md:flex-row md:items-center">
                    <div>
                      <div className="flex items-center gap-2">
                        <span className="h-2 w-2 rounded-full bg-[#4F46E5] shadow-[0_0_0_4px_rgba(79,70,229,.12)]" />
                        <p className="text-[10px] font-black uppercase tracking-[0.2em] text-[#4F46E5]">
                          Live Ingestion Pipeline
                        </p>
                      </div>
                      <h3 className="mt-1 text-xl font-black tracking-tight text-slate-900">
                        Upload PRADAN Archives
                      </h3>
                      <p className="mt-1 max-w-2xl text-xs text-slate-500">
                        Drop your PRADAN zip files below to automatically extract PDS4 metadata, match
                        overlapping OHRC + TMC-2 + IIRS triplets, and generate pipeline artifacts.
                      </p>
                    </div>

                    <div className="flex items-center gap-2">
                      <div
                        className={`rounded-full border px-3 py-1.5 text-[10px] font-black uppercase tracking-wider ${
                          phase === "processing" || phase === "queued"
                            ? "border-indigo-300 bg-indigo-50 text-[#4F46E5] animate-pulse"
                            : phase === "done"
                            ? "border-emerald-300 bg-emerald-50 text-emerald-700"
                            : phase === "error"
                            ? "border-rose-300 bg-rose-50 text-rose-700"
                            : "border-slate-200 bg-slate-50 text-slate-500"
                        }`}
                      >
                        {isProcessing
                          ? "PROCESSING..."
                          : phase === "done"
                          ? "COMPLETED"
                          : phase === "error"
                          ? "FAILED"
                          : "READY TO RUN"}
                      </div>
                    </div>
                  </div>

                  {/* Drop Zone */}
                  {(phase === "idle" || phase === "queued") && (
                    <DropZone
                      onFilesSelected={handleFilesSelected}
                      disabled={isProcessing}
                    />
                  )}

                  {/* Uploaded File List & Actions */}
                  {files.length > 0 && phase !== "done" && (
                    <div className="space-y-4 animate-fade-in">
                      <div className="flex flex-wrap gap-2 pt-1">
                        {files.map((f) => (
                          <div
                            key={f.name}
                            className="inline-flex items-center gap-2 rounded-xl border border-slate-200 bg-slate-50 px-3 py-1.5 text-xs text-slate-700 shadow-xs"
                          >
                            <span>📦</span>
                            <span className="font-medium max-w-[200px] truncate">{f.name}</span>
                            <span className="font-mono text-[10px] text-slate-400">
                              {fmtSize(f.size)}
                            </span>
                            {!isProcessing && (
                              <button
                                type="button"
                                onClick={() => removeFile(f.name)}
                                className="text-slate-400 hover:text-rose-600 transition px-1 text-sm leading-none"
                                title="Remove file"
                              >
                                &times;
                              </button>
                            )}
                          </div>
                        ))}
                      </div>

                      {/* Stats and Action Buttons */}
                      <div className="flex flex-wrap items-center justify-between gap-4 pt-2 border-t border-slate-100">
                        <div className="flex items-center gap-4 text-xs text-slate-500 font-medium">
                          <span>
                            Files: <strong className="font-mono text-slate-800">{files.length}</strong>
                          </span>
                          <span>
                            Total: <strong className="font-mono text-slate-800">{fmtSize(totalSize)}</strong>
                          </span>
                        </div>

                        <div className="flex items-center gap-2">
                          <button
                            type="button"
                            onClick={clearFiles}
                            disabled={isProcessing}
                            className="rounded-xl border border-slate-200 bg-white px-3.5 py-2 text-xs font-semibold text-slate-600 hover:bg-slate-50 transition shadow-sm disabled:opacity-40"
                          >
                            Clear All
                          </button>

                          <button
                            type="button"
                            onClick={handleStart}
                            disabled={isProcessing}
                            className="flex items-center gap-2 rounded-xl bg-[#4F46E5] px-6 py-2.5 text-xs font-black uppercase tracking-[0.12em] text-white shadow-sm transition hover:bg-[#4338CA] disabled:cursor-not-allowed disabled:opacity-40"
                          >
                            {isProcessing ? (
                              <>
                                <span className="h-3 w-3 animate-spin rounded-full border-2 border-white/40 border-t-white" />
                                <span>Running Pipeline...</span>
                              </>
                            ) : (
                              <span>Start Ingestion ({files.length})</span>
                            )}
                          </button>
                        </div>
                      </div>
                    </div>
                  )}

                  {/* Advanced Pipeline Configuration Accordion */}
                  {phase === "idle" && (
                    <div className="rounded-xl border border-slate-200/80 bg-slate-50/50 p-4 transition">
                      <button
                        type="button"
                        onClick={() => setShowConfig(!showConfig)}
                        className="w-full flex items-center justify-between text-left focus:outline-none"
                      >
                        <span className="text-xs font-bold uppercase tracking-wider text-slate-600">
                          Pipeline Configuration
                        </span>
                        <span className="text-xs font-bold text-[#4F46E5]">
                          {showConfig ? "▲ Hide Configuration" : "▼ Show Advanced"}
                        </span>
                      </button>

                      {showConfig && (
                        <div className="mt-4 pt-4 border-t border-slate-200/70 space-y-4 animate-fade-in">
                          <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
                            <div>
                              <label className="block text-[10px] font-bold uppercase tracking-wider text-slate-500 mb-1.5">
                                Min Containment ({Math.round(config.containment * 100)}%)
                              </label>
                              <input
                                type="range"
                                min="0.5"
                                max="1.0"
                                step="0.05"
                                value={config.containment}
                                onChange={(e) =>
                                  setConfig({ ...config, containment: parseFloat(e.target.value) })
                                }
                                className="w-full accent-[#4F46E5]"
                              />
                            </div>

                            <div>
                              <label className="block text-[10px] font-bold uppercase tracking-wider text-slate-500 mb-1.5">
                                Tile Size (px)
                              </label>
                              <select
                                value={config.tileSize}
                                onChange={(e) =>
                                  setConfig({ ...config, tileSize: parseInt(e.target.value) })
                                }
                                className="w-full rounded-xl border border-slate-200 bg-white px-3 py-2 text-xs font-semibold text-slate-700 outline-none focus:border-[#4F46E5] focus:ring-2 focus:ring-indigo-100"
                              >
                                <option value={256}>256 × 256</option>
                                <option value={512}>512 × 512 (Standard)</option>
                                <option value={1024}>1024 × 1024</option>
                              </select>
                            </div>
                          </div>

                          <div>
                            <label className="block text-[10px] font-bold uppercase tracking-wider text-slate-500 mb-1.5">
                              Max Time Gap (Days, Optional)
                            </label>
                            <input
                              type="number"
                              placeholder="e.g. 180 (empty = any)"
                              value={config.maxTimeGapDays ?? ""}
                              onChange={(e) =>
                                setConfig({
                                  ...config,
                                  maxTimeGapDays: e.target.value ? parseFloat(e.target.value) : null,
                                })
                              }
                              className="w-full rounded-xl border border-slate-200 bg-white px-3 py-2 text-xs font-mono text-slate-700 outline-none focus:border-[#4F46E5] focus:ring-2 focus:ring-indigo-100"
                            />
                          </div>

                          <div className="flex flex-wrap gap-4 pt-1">
                            <label className="flex items-center gap-2 text-xs font-medium text-slate-700 cursor-pointer">
                              <input
                                type="checkbox"
                                checked={config.noLargeAoi}
                                onChange={(e) =>
                                  setConfig({ ...config, noLargeAoi: e.target.checked })
                                }
                                className="rounded accent-[#4F46E5]"
                              />
                              Skip Large-AOI IIRS
                            </label>

                            <label className="flex items-center gap-2 text-xs font-medium text-slate-700 cursor-pointer">
                              <input
                                type="checkbox"
                                checked={config.noInvariants}
                                onChange={(e) =>
                                  setConfig({ ...config, noInvariants: e.target.checked })
                                }
                                className="rounded accent-[#4F46E5]"
                              />
                              Skip Invariant Maps
                            </label>

                            <label className="flex items-center gap-2 text-xs font-medium text-slate-700 cursor-pointer">
                              <input
                                type="checkbox"
                                checked={config.requireDates}
                                onChange={(e) =>
                                  setConfig({ ...config, requireDates: e.target.checked })
                                }
                                className="rounded accent-[#4F46E5]"
                              />
                              Require Dates
                            </label>
                          </div>
                        </div>
                      )}
                    </div>
                  )}

                  {/* Error banner */}
                  {phase === "error" && error && !status && (
                    <div className="rounded-xl border border-rose-200 bg-rose-50 p-4 text-xs text-rose-700 font-medium space-y-2 animate-fade-in">
                      <div>
                        <strong className="font-bold">Error:</strong> {error}
                      </div>
                      <div className="flex items-center gap-2">
                        {sessionExpired ? (
                          <Link
                            href="/"
                            className="rounded-lg bg-[#4F46E5] px-3 py-1.5 text-xs font-bold text-white hover:bg-[#4338CA] transition"
                          >
                            Go to Sign In
                          </Link>
                        ) : (
                          <button
                            type="button"
                            onClick={clearFiles}
                            className="rounded-lg border border-rose-300 bg-white px-3 py-1 text-xs font-semibold text-rose-700 hover:bg-rose-50"
                          >
                            Start Over
                          </button>
                        )}
                      </div>
                    </div>
                  )}

                  {/* Processing progress */}
                  {status && (phase === "processing" || phase === "done" || phase === "error") && (
                    <ProcessingProgress status={status} />
                  )}

                  {/* Results */}
                  {phase === "done" && (resultTriplets.length > 0 || status) && (
                    <div className="space-y-4 animate-fade-in">
                      <ResultsTable
                        triplets={resultTriplets}
                        containment={config.containment}
                      />

                      <div className="pt-2 text-center">
                        <button
                          type="button"
                          onClick={clearFiles}
                          className="rounded-xl bg-[#4F46E5] px-6 py-3 text-xs font-black uppercase tracking-wider text-white shadow-sm hover:bg-[#4338CA] transition"
                        >
                          Process Another Batch
                        </button>
                      </div>
                    </div>
                  )}
                </section>
              )}

              {/* TAB 2: PREVIOUS INGESTION RUNS */}
              {activeTab === "history" && (
                <section className="rounded-2xl border border-slate-200/80 bg-white p-5 shadow-sm md:p-6 space-y-5 animate-fade-in">
                  <div className="flex items-center justify-between border-b border-slate-100 pb-4">
                    <div>
                      <h3 className="text-lg font-bold text-slate-900">
                        Previous Ingestion Runs
                      </h3>
                      <p className="text-xs text-slate-500">
                        Inspect history and replay discovery results from past PRADAN archive batches.
                      </p>
                    </div>

                    <button
                      type="button"
                      onClick={loadHistoryJobs}
                      disabled={historyLoading}
                      className="flex items-center gap-1.5 rounded-xl border border-slate-200 bg-white px-3.5 py-1.5 text-xs font-semibold text-slate-700 hover:bg-slate-50 transition shadow-sm disabled:opacity-50"
                    >
                      <span>{historyLoading ? "Refreshing..." : "↻ Refresh History"}</span>
                    </button>
                  </div>

                  {historyError && (
                    <div className="rounded-xl border border-rose-200 bg-rose-50 p-4 text-xs text-rose-700 font-medium">
                      {historyError}
                    </div>
                  )}

                  {historyLoading && historyJobs.length === 0 && (
                    <div className="flex h-48 flex-col items-center justify-center gap-3 rounded-xl border border-slate-200 bg-slate-50/50 p-8 text-center">
                      <div className="h-6 w-6 animate-spin rounded-full border-2 border-[#4F46E5] border-t-transparent" />
                      <p className="font-mono text-xs text-slate-500">Loading historical ingestion records...</p>
                    </div>
                  )}

                  {!historyLoading && historyJobs.length === 0 && !historyError && (
                    <div className="flex flex-col items-center justify-center gap-3 rounded-xl border border-dashed border-slate-200 bg-slate-50/50 p-12 text-center">
                      <span className="text-3xl">📦</span>
                      <h4 className="text-sm font-bold text-slate-900">No Previous Ingestion Runs</h4>
                      <p className="max-w-md text-xs text-slate-500 leading-relaxed">
                        No past ingestion jobs were found on the backend. Upload raw Chandrayaan-2 PRADAN ZIP bundles to start your first discovery run.
                      </p>
                      <button
                        type="button"
                        onClick={() => setActiveTab("new")}
                        className="mt-2 rounded-xl bg-[#4F46E5] px-4 py-2 text-xs font-bold text-white shadow-sm hover:bg-[#4338CA] transition"
                      >
                        Upload PRADAN Files Now
                      </button>
                    </div>
                  )}

                  {historyJobs.length > 0 && (
                    <div className="overflow-hidden rounded-xl border border-slate-200/80 bg-white">
                      <table className="w-full text-left text-xs border-collapse">
                        <thead>
                          <tr className="border-b border-slate-200/80 bg-slate-50 text-[10px] font-bold uppercase tracking-wider text-slate-500">
                            <th className="py-3 px-4">Job ID</th>
                            <th className="py-3 px-4">Status</th>
                            <th className="py-3 px-4">Current Stage</th>
                            <th className="py-3 px-4">Progress</th>
                            <th className="py-3 px-4">Started</th>
                            <th className="py-3 px-4 text-right">Action</th>
                          </tr>
                        </thead>
                        <tbody className="divide-y divide-slate-100">
                          {historyJobs.map((job) => {
                            const statusColor =
                              job.status === "completed"
                                ? "border-emerald-200 bg-emerald-50 text-emerald-700"
                                : job.status === "running"
                                ? "border-indigo-200 bg-indigo-50 text-[#4F46E5] animate-pulse"
                                : job.status === "failed"
                                ? "border-rose-200 bg-rose-50 text-rose-700"
                                : "border-slate-200 bg-slate-50 text-slate-500";

                            return (
                              <tr key={job.job_id} className="transition hover:bg-slate-50/60">
                                <td className="py-3.5 px-4 font-mono font-semibold text-slate-900">
                                  {job.job_id.slice(0, 8)}...{job.job_id.slice(-4)}
                                </td>
                                <td className="py-3.5 px-4">
                                  <span
                                    className={`inline-flex items-center gap-1 rounded-md border px-2.5 py-0.5 text-[10px] font-bold uppercase tracking-wider ${statusColor}`}
                                  >
                                    <span className="h-1.5 w-1.5 rounded-full bg-current" />
                                    {job.status}
                                  </span>
                                </td>
                                <td className="py-3.5 px-4 text-slate-600 font-mono text-[11px]">
                                  {job.stage || "—"}
                                </td>
                                <td className="py-3.5 px-4">
                                  <div className="flex items-center gap-2">
                                    <div className="h-1.5 w-24 overflow-hidden rounded-full bg-slate-100">
                                      <div
                                        className="h-full bg-[#4F46E5] transition-all duration-300"
                                        style={{ width: `${job.progress_pct || 0}%` }}
                                      />
                                    </div>
                                    <span className="font-mono text-[10px] text-slate-500 font-semibold">
                                      {Math.round(job.progress_pct || 0)}%
                                    </span>
                                  </div>
                                </td>
                                <td className="py-3.5 px-4 font-mono text-[10px] text-slate-500">
                                  {job.started_at ? new Date(job.started_at).toLocaleString() : "—"}
                                </td>
                                <td className="py-3.5 px-4 text-right">
                                  {job.status === "completed" && (
                                    <button
                                      type="button"
                                      onClick={() => handleSelectHistoricalJob(job.job_id)}
                                      className="rounded-lg border border-indigo-200 bg-indigo-50 px-3 py-1 font-mono text-xs font-bold text-[#4F46E5] hover:bg-indigo-100 transition shadow-xs"
                                    >
                                      Load Results →
                                    </button>
                                  )}
                                  {job.status === "running" && (
                                    <button
                                      type="button"
                                      onClick={() => {
                                        setJobId(job.job_id);
                                        setPhase("processing");
                                        setActiveTab("new");
                                      }}
                                      className="rounded-lg border border-blue-200 bg-blue-50 px-3 py-1 font-mono text-xs font-bold text-blue-700 hover:bg-blue-100 transition shadow-xs"
                                    >
                                      Attach &amp; Monitor
                                    </button>
                                  )}
                                </td>
                              </tr>
                            );
                          })}
                        </tbody>
                      </table>
                    </div>
                  )}
                </section>
              )}
            </>
          )}
        </main>
      </div>

      {/* Modals Shared with Dashboard */}
      {vaultOpen && (
        <VaultModal
          triplets={triplets}
          initialFilter="all"
          onClose={() => setVaultOpen(false)}
          onSelectRegion={(id) => {
            setVaultOpen(false);
            router.push(`/?view=console&region=${id}`);
          }}
        />
      )}

      {theoryModalOpen && (
        <TheoryModal onClose={() => setTheoryModalOpen(false)} />
      )}
    </div>
  );
}
