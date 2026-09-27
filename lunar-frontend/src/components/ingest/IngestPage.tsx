"use client";

import React, { useCallback, useEffect, useRef, useState } from "react";
import Link from "next/link";
import { useRouter } from "next/navigation";
import { api } from "@/lib/api";
import { footprintSizeKm } from "@/lib/geo";
import type { TripletSummary } from "@/lib/types";
import { isAuthenticated, getCurrentUser, logout, type AuthUser } from "@/lib/auth";
import { useTheme } from "@/lib/theme";
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
  deleteJob,
  deleteJobs,
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
  const { theme, toggle: toggleTheme } = useTheme();

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
  const [selectedJobs, setSelectedJobs] = useState<Set<string>>(new Set());
  const [deletingJobs, setDeletingJobs] = useState(false);
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
      // Drop selections for jobs that no longer exist.
      setSelectedJobs((prev) => {
        const ids = new Set(jobs.map((j) => j.job_id));
        const next = new Set<string>();
        prev.forEach((id) => {
          if (ids.has(id)) next.add(id);
        });
        return next;
      });
    } catch (err: any) {
      setHistoryError(err.message || "Failed to load past ingestion jobs");
    } finally {
      setHistoryLoading(false);
    }
  }, []);

  const handleDeleteHistoryJob = useCallback(async (id: string) => {
    if (!window.confirm(`Delete ingestion history entry '${id}'? This removes the job record and its staging upload files.`)) return;
    setDeletingJobs(true);
    setHistoryError(null);
    try {
      await deleteJob(id);
      setHistoryJobs((prev) => prev.filter((j) => j.job_id !== id));
      setSelectedJobs((prev) => {
        const next = new Set(prev);
        next.delete(id);
        return next;
      });
    } catch (err: any) {
      setHistoryError(err.message || "Failed to delete job");
    } finally {
      setDeletingJobs(false);
    }
  }, []);

  const handleDeleteSelectedJobs = useCallback(async () => {
    if (selectedJobs.size === 0) return;
    if (!window.confirm(`Delete ${selectedJobs.size} ingestion histor${selectedJobs.size === 1 ? "y entry" : "y entries"}?`)) return;
    setDeletingJobs(true);
    setHistoryError(null);
    try {
      const res = await deleteJobs([...selectedJobs]);
      const gone = new Set(res.deleted || []);
      setHistoryJobs((prev) => prev.filter((j) => !gone.has(j.job_id)));
      setSelectedJobs(new Set());
      const errs = res.errors || {};
      const failed = Object.entries(errs);
      if (failed.length > 0) {
        setHistoryError(`Some entries could not be deleted: ${failed.map(([id, e]) => `${id} (${e})`).join("; ")}`);
      }
    } catch (err: any) {
      setHistoryError(err.message || "Failed to delete jobs");
    } finally {
      setDeletingJobs(false);
    }
  }, [selectedJobs]);

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

  // Poll loop: bounded consecutive-failure budget with exponential backoff.
  // The old loop retried forever every 1.5s — a dead backend spun the
  // spinner (and the backend) indefinitely. After MAX_POLL_FAILURES straight
  // failures the job errors out with a retry path instead of hanging.
  useEffect(() => {
    if (phase !== "processing" || !jobId) return;

    const MAX_POLL_FAILURES = 10;
    const BASE_POLL_MS = 1500;
    const MAX_POLL_MS = 15000;
    let failures = 0;
    let timer: number | null = null;
    let cancelled = false;

    const poll = async () => {
      if (cancelled) return;
      try {
        const s = await pollStatus(jobId);
        failures = 0;
        if (cancelled) return;
        setStatus(s);

        if (s.status === "completed") {
          setPhase("done");
          try {
            const results = await getResults(jobId);
            if (!cancelled) setResultTriplets(results.triplets || []);
          } catch {
            // Triplets stay empty
          }
          return;
        }
        if (s.status === "failed") {
          setPhase("error");
          setError(s.error || "Pipeline failed");
          return;
        }
        timer = window.setTimeout(poll, BASE_POLL_MS);
      } catch {
        failures += 1;
        if (failures >= MAX_POLL_FAILURES) {
          setPhase("error");
          setError(
            `Lost contact with the ingest backend after ${MAX_POLL_FAILURES} attempts. ` +
              "The job may still be running — check Previous Ingestion Runs, then retry."
          );
          return;
        }
        // Exponential backoff: 1.5s → 3s → 6s … capped at 15s.
        timer = window.setTimeout(poll, Math.min(BASE_POLL_MS * 2 ** (failures - 1), MAX_POLL_MS));
      }
    };

    poll();

    return () => {
      cancelled = true;
      if (timer !== null) window.clearTimeout(timer);
      if (pollRef.current !== null) {
        window.clearInterval(pollRef.current);
        pollRef.current = null;
      }
    };
  }, [phase, jobId]);

  const totalSize = files.reduce((s, f) => s + f.size, 0);
  const isProcessing = phase === "processing" || phase === "queued";

  return (
    <div className="h-screen overflow-hidden bg-[#D3CCC0] dark:bg-[#121615] font-sans text-[#1E2321] dark:text-[#E7E2D6] antialiased flex flex-col">
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
                <Link
                  href="/?view=console"
                  prefetch={false}
                  className="retro-button w-full text-left px-2 py-1.5 flex items-center justify-between text-xs font-semibold text-[#222926] dark:text-[#E7E2D6] hover:bg-[#D9D3C5] dark:hover:bg-white/10 transition"
                >
                  <div className="flex items-center gap-2">
                    <span className="text-[11px]">▣</span>
                    <span>Dashboard QA</span>
                  </div>
                </Link>

                <div
                  className="w-full text-left px-2 py-1.5 flex items-center justify-between text-xs transition-all bg-[#28557E] text-white font-bold border-t border-l border-[#4477A6] border-r-2 border-b-2 border-[#122A42] shadow-inner"
                >
                  <div className="flex items-center gap-2">
                    <span className="text-[11px]">⚡</span>
                    <span>Ingest &amp; Prepare</span>
                  </div>
                  <span className="h-1.5 w-1.5 rounded-full bg-white" />
                </div>

                <Link
                  href="/?view=console&subview=linked-cursor"
                  prefetch={false}
                  className="retro-button w-full text-left px-2 py-1.5 flex items-center justify-between text-xs font-semibold text-[#222926] dark:text-[#E7E2D6] hover:bg-[#D9D3C5] dark:hover:bg-white/10 transition"
                >
                  <div className="flex items-center gap-2">
                    <span className="text-[11px]">⊙</span>
                    <span>Linked Cursor</span>
                  </div>
                </Link>

                <Link
                  href="/?view=console&subview=map"
                  prefetch={false}
                  className="retro-button w-full text-left px-2 py-1.5 flex items-center justify-between text-xs font-semibold text-[#222926] dark:text-[#E7E2D6] hover:bg-[#D9D3C5] dark:hover:bg-white/10 transition"
                >
                  <div className="flex items-center gap-2">
                    <span className="text-[11px]">☵</span>
                    <span>Planetary Map</span>
                  </div>
                </Link>

                <div className="border-t border-[#8B8579] dark:border-[#2D3835] my-1" />

                <button
                  onClick={() => setVaultOpen(true)}
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
              </div>

              <div className="retro-inset flex-1 overflow-y-auto p-1 space-y-1 font-mono text-[11px] max-h-[220px]">
                {filteredTriplets.map((t, i) => {
                  const { widthKm, heightKm } = footprintSizeKm(t.bounds);
                  return (
                    <Link
                      key={t.id}
                      href={`/?view=console&region=${t.id}`}
                      className="flex w-full items-center justify-between px-1.5 py-1 text-left text-xs transition-all bg-[#ECE7DC] dark:bg-[#161B19] hover:bg-[#DFD9CB] dark:hover:bg-white/5 text-[#2B312E] dark:text-[#E7E2D6] border border-[#CCC6B8] dark:border-[#2D3835]"
                    >
                      <div className="truncate">
                        <span className="text-[10px] mr-1.5 text-[#69726E]">
                          {String(i + 1).padStart(2, "0")}.
                        </span>
                        <span>{t.id}</span>
                        <span className="block text-[9px] font-normal text-[#69726E]">
                          {widthKm.toFixed(1)} × {heightKm.toFixed(1)} km
                        </span>
                      </div>
                      <div className="flex items-center gap-1 shrink-0">
                        {t.dem_available && (
                          <span className="px-1 text-[9px] border bg-[#D3CBBF] dark:bg-[#2D3835] border-[#9E9789] text-[#1E2321] dark:text-[#E7E2D6]">
                            DEM
                          </span>
                        )}
                        {t.lro_nac_available && (
                          <span className="px-1 text-[9px] border bg-[#D1B890] dark:bg-[#785317] border-[#9E8662] text-black dark:text-amber-200">
                            LRO
                          </span>
                        )}
                      </div>
                    </Link>
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
        {/* 2. MAIN WORKSPACE: Header Bar & Ingest Content           */}
        {/* ======================================================== */}
        <div className="flex-1 flex flex-col min-w-0 h-full overflow-hidden">
          {/* Top Header Bar */}
          <header className="bg-[#E7E2D6] dark:bg-[#1A201E] border-b-2 border-[#8B8579] dark:border-[#2D3835] px-4 py-1.5 flex items-center justify-between gap-3 shrink-0 shadow-sm z-30 select-none">
            {/* Search Input */}
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

            {/* Center: Live Ingestion Pipeline Status Beacon */}
            <div className="hidden lg:flex items-center gap-2 retro-inset px-2.5 py-1 text-2xs font-mono font-bold bg-[#E2DDCF] dark:bg-[#161B19] text-[#1F4743] dark:text-emerald-400">
              <span className="relative flex h-2 w-2">
                <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-emerald-400 opacity-75" />
                <span className="relative inline-flex rounded-full h-2 w-2 bg-emerald-500" />
              </span>
              <span>FASTAPI :8000</span>
              <span className="text-[#8B8579]">·</span>
              <span>INGESTION PIPELINE READY</span>
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
                onClick={() => router.push("/")}
                className="retro-button px-2.5 py-1 text-[11px] font-semibold text-[#2C312E] dark:text-[#E7E2D6] flex items-center gap-1"
                title="Return to Mission Overview"
              >
                <span>←</span>
                <span>Back</span>
              </button>

              <button
                onClick={() => setVaultOpen(true)}
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
                          setVaultOpen(true);
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
                          router.push("/");
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

          {/* Ingest Main Content */}
          <main className="flex-1 min-h-0 overflow-y-auto p-3 sm:p-4 space-y-3">
            {/* Top Title & Subtitle + Action Tabs */}
            <div className="retro-outset p-3">
              <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3">
                <div>
                  <div className="flex items-center gap-2">
                    <h2 className="text-base font-bold font-mono tracking-tight text-[#1E2321] dark:text-[#E7E2D6]">
                      INGEST &amp; PREPARE
                    </h2>
                    <span className="retro-inset px-2 py-0.5 text-[9px] font-mono font-bold bg-[#1F4743] text-white">
                      ISRO SIH26166
                    </span>
                  </div>
                  <p className="text-[11px] text-[#555C58] dark:text-[#8C9893] mt-0.5 font-sans">
                    Drop raw PRADAN archives to automatically discover, match, and process Chandrayaan-2 OHRC + TMC-2 + IIRS triplets with sub-pixel verification.
                  </p>
                </div>

                {/* Tab Controls & Direct Link to Dashboard */}
                <div className="flex items-center gap-2 shrink-0 flex-wrap">
                  <button
                    type="button"
                    onClick={() => setActiveTab("new")}
                    className={`px-3 py-1.5 text-xs font-bold transition-all ${
                      activeTab === "new"
                        ? "retro-button-primary font-bold shadow-inner"
                        : "retro-button text-[#222926] dark:text-[#E7E2D6]"
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
                    className={`flex items-center gap-1.5 px-3 py-1.5 text-xs font-bold transition-all ${
                      activeTab === "history"
                        ? "retro-button-primary font-bold shadow-inner"
                        : "retro-button text-[#222926] dark:text-[#E7E2D6]"
                    }`}
                  >
                    <span>📂 Previous Runs</span>
                    {historyJobs.length > 0 && (
                      <span className="retro-inset px-1 py-0.2 text-[9px] font-mono bg-[#143532] text-white">
                        {historyJobs.length}
                      </span>
                    )}
                  </button>

                  <Link
                    href="/?view=console"
                    className="retro-button px-3 py-1.5 text-xs font-bold text-[#222926] dark:text-[#E7E2D6] flex items-center gap-1"
                  >
                    <span>Mission Dashboard</span>
                    <span>↗</span>
                  </Link>
                </div>
              </div>
            </div>

            {/* Auth Guard Check */}
            {!authChecked ? (
              <div className="retro-outset p-8 text-center max-w-md mx-auto space-y-3 font-mono">
                <div className="inline-block h-6 w-6 animate-spin border-2 border-[#1F4743] border-r-transparent" />
                <p className="text-xs text-[#555C58] dark:text-[#8C9893]">Verifying credentials...</p>
              </div>
            ) : !isAuthed ? (
              <div className="retro-outset p-8 text-center max-w-md mx-auto space-y-3 font-mono">
                <div className="text-3xl">🔐</div>
                <h3 className="text-sm font-bold text-[#1E2321] dark:text-[#E7E2D6]">
                  SIGN IN REQUIRED FOR INGESTION
                </h3>
                <p className="text-xs text-[#555C58] dark:text-[#8C9893] leading-relaxed font-sans">
                  The ingest pipeline writes to disk and launches processing jobs, so uploads require an
                  authenticated operator session. Please sign in from the home page, then return here.
                </p>
                <div className="pt-2">
                  <Link
                    href="/"
                    className="retro-button-primary px-4 py-2 text-xs font-bold font-mono inline-block"
                  >
                    GO TO SIGN IN
                  </Link>
                </div>
              </div>
            ) : (
              <>
                {/* TAB 1: NEW BATCH INGESTION */}
                {activeTab === "new" && (
                  <section className="retro-outset p-1 space-y-3">
                    {/* Card Header Bar */}
                    <div className="bg-[#1F4743] text-white px-2.5 py-1 flex items-center justify-between text-xs font-bold font-mono tracking-wider">
                      <div className="flex items-center gap-2">
                        <span className="h-2 w-2 rounded-full bg-emerald-400" />
                        <span>LIVE INGESTION PIPELINE // UPLOAD PRADAN ARCHIVES</span>
                      </div>

                      <div className="retro-inset px-2 py-0.5 bg-[#143532] text-white text-[10px] font-mono font-bold">
                        {isProcessing
                          ? "PROCESSING..."
                          : phase === "done"
                          ? "COMPLETED"
                          : phase === "error"
                          ? "FAILED"
                          : "READY TO RUN"}
                      </div>
                    </div>

                    <div className="p-3 bg-[#E7E2D6] dark:bg-[#1A201E] space-y-3">
                      {/* Drop Zone */}
                      {(phase === "idle" || phase === "queued") && (
                        <DropZone
                          onFilesSelected={handleFilesSelected}
                          disabled={isProcessing}
                        />
                      )}

                      {/* Uploaded File List & Actions */}
                      {files.length > 0 && phase !== "done" && (
                        <div className="space-y-3 animate-fade-in">
                          <div className="flex flex-wrap gap-1.5 pt-1">
                            {files.map((f) => (
                              <div
                                key={f.name}
                                className="retro-inset px-2.5 py-1 text-xs font-mono text-[#1E2321] dark:text-[#E7E2D6] bg-[#ECE7DC] dark:bg-[#161B19] flex items-center gap-2"
                              >
                                <span>📦</span>
                                <span className="font-semibold max-w-[200px] truncate">{f.name}</span>
                                <span className="text-[10px] text-[#555C58] dark:text-[#8C9893]">
                                  {fmtSize(f.size)}
                                </span>
                                {!isProcessing && (
                                  <button
                                    type="button"
                                    onClick={() => removeFile(f.name)}
                                    className="text-[#888] hover:text-red-600 transition px-1 text-sm leading-none font-bold"
                                    title="Remove file"
                                  >
                                    &times;
                                  </button>
                                )}
                              </div>
                            ))}
                          </div>

                          {/* Stats and Action Buttons */}
                          <div className="flex flex-wrap items-center justify-between gap-3 pt-2 border-t border-[#8B8579]/40 dark:border-[#2D3835]">
                            <div className="flex items-center gap-3 text-xs font-mono text-[#555C58] dark:text-[#8C9893]">
                              <span>
                                FILES: <strong className="text-[#1E2321] dark:text-[#E7E2D6]">{files.length}</strong>
                              </span>
                              <span>
                                TOTAL: <strong className="text-[#1E2321] dark:text-[#E7E2D6]">{fmtSize(totalSize)}</strong>
                              </span>
                            </div>

                            <div className="flex items-center gap-2">
                              <button
                                type="button"
                                onClick={clearFiles}
                                disabled={isProcessing}
                                className="retro-button px-3 py-1.5 text-xs font-bold text-[#1E2321] dark:text-[#E7E2D6] disabled:opacity-40"
                              >
                                Clear All
                              </button>

                              <button
                                type="button"
                                onClick={handleStart}
                                disabled={isProcessing}
                                className="retro-button-primary px-4 py-1.5 text-xs font-mono font-bold uppercase tracking-wider flex items-center gap-2 disabled:opacity-40"
                              >
                                {isProcessing ? (
                                  <>
                                    <span className="h-2.5 w-2.5 animate-spin rounded-full border-2 border-white/40 border-t-white" />
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
                        <div className="retro-outset p-2 bg-[#E9E4D8] dark:bg-[#161B19]">
                          <button
                            type="button"
                            onClick={() => setShowConfig(!showConfig)}
                            className="w-full flex items-center justify-between text-left font-mono"
                          >
                            <span className="text-xs font-bold uppercase tracking-wider text-[#1E2321] dark:text-[#E7E2D6]">
                              Pipeline Configuration
                            </span>
                            <span className="text-xs font-bold text-[#28557E] dark:text-cyan-400">
                              {showConfig ? "▲ Hide Configuration" : "▼ Show Advanced"}
                            </span>
                          </button>

                          {showConfig && (
                            <div className="mt-3 pt-3 border-t border-[#8B8579]/40 dark:border-[#2D3835] space-y-3 font-mono animate-fade-in">
                              <div className="grid grid-cols-1 sm:grid-cols-2 gap-3">
                                <div>
                                  <label className="block text-[10px] font-bold uppercase tracking-wider text-[#555C58] dark:text-[#8C9893] mb-1">
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
                                    className="w-full accent-[#28557E]"
                                  />
                                </div>

                                <div>
                                  <label className="block text-[10px] font-bold uppercase tracking-wider text-[#555C58] dark:text-[#8C9893] mb-1">
                                    Tile Size (px)
                                  </label>
                                  <select
                                    value={config.tileSize}
                                    onChange={(e) =>
                                      setConfig({ ...config, tileSize: parseInt(e.target.value) })
                                    }
                                    className="retro-inset w-full px-2 py-1 text-xs font-mono font-bold bg-[#ECE7DC] dark:bg-[#161B19] text-[#1E2321] dark:text-[#E7E2D6] outline-none"
                                  >
                                    <option value={256}>256 × 256</option>
                                    <option value={512}>512 × 512 (Standard)</option>
                                    <option value={1024}>1024 × 1024</option>
                                  </select>
                                </div>
                              </div>

                              <div>
                                <label className="block text-[10px] font-bold uppercase tracking-wider text-[#555C58] dark:text-[#8C9893] mb-1">
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
                                  className="retro-inset w-full px-2 py-1 text-xs font-mono bg-white dark:bg-[#0F1211] text-[#1E2321] dark:text-[#E7E2D6] outline-none"
                                />
                              </div>

                              <div className="flex flex-wrap gap-3 pt-1 text-xs text-[#1E2321] dark:text-[#E7E2D6]">
                                <label className="flex items-center gap-1.5 cursor-pointer select-none">
                                  <input
                                    type="checkbox"
                                    checked={config.noLargeAoi}
                                    onChange={(e) =>
                                      setConfig({ ...config, noLargeAoi: e.target.checked })
                                    }
                                    className="accent-[#28557E]"
                                  />
                                  Skip Large-AOI IIRS
                                </label>

                                <label className="flex items-center gap-1.5 cursor-pointer select-none">
                                  <input
                                    type="checkbox"
                                    checked={config.noInvariants}
                                    onChange={(e) =>
                                      setConfig({ ...config, noInvariants: e.target.checked })
                                    }
                                    className="accent-[#28557E]"
                                  />
                                  Skip Invariant Maps
                                </label>

                                <label className="flex items-center gap-1.5 cursor-pointer select-none">
                                  <input
                                    type="checkbox"
                                    checked={config.requireDates}
                                    onChange={(e) =>
                                      setConfig({ ...config, requireDates: e.target.checked })
                                    }
                                    className="accent-[#28557E]"
                                  />
                                  Require Dates
                                </label>

                                <label className="flex items-center gap-1.5 cursor-pointer select-none">
                                  <input
                                    type="checkbox"
                                    checked={config.noMatching}
                                    onChange={(e) =>
                                      setConfig({ ...config, noMatching: e.target.checked })
                                    }
                                    className="accent-[#28557E]"
                                  />
                                  Skip Matching (linked-cursor dots)
                                </label>

                                <label className="flex items-center gap-1.5 cursor-pointer select-none">
                                  <input
                                    type="checkbox"
                                    checked={config.noRegistration}
                                    onChange={(e) =>
                                      setConfig({ ...config, noRegistration: e.target.checked })
                                    }
                                    className="accent-[#28557E]"
                                  />
                                  Skip Registration QA (grid/blend/quiver)
                                </label>
                              </div>
                            </div>
                          )}
                        </div>
                      )}

                      {/* Error banner */}
                      {phase === "error" && error && !status && (
                        <div className="retro-outset p-3 text-xs font-mono text-rose-800 dark:text-rose-200 bg-[#EED2D2] dark:bg-[#3D1A1A] border border-rose-600 animate-fade-in space-y-2">
                          <div>
                            <strong className="font-bold">ERROR:</strong> {error}
                          </div>
                          <div className="flex items-center gap-2">
                            {sessionExpired ? (
                              <Link
                                href="/"
                                className="retro-button-primary px-3 py-1 text-xs font-bold font-mono"
                              >
                                Go to Sign In
                              </Link>
                            ) : (
                              <button
                                type="button"
                                onClick={clearFiles}
                                className="retro-button px-3 py-1 text-xs font-bold text-[#1E2321] dark:text-[#E7E2D6]"
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
                        <div className="space-y-3 animate-fade-in">
                          <ResultsTable
                            triplets={resultTriplets}
                            containment={config.containment}
                          />

                          <div className="pt-2 text-center">
                            <button
                              type="button"
                              onClick={clearFiles}
                              className="retro-button-primary px-5 py-2 text-xs font-mono font-bold uppercase tracking-wider"
                            >
                              Process Another Batch
                            </button>
                          </div>
                        </div>
                      )}
                    </div>
                  </section>
                )}

                {/* TAB 2: PREVIOUS INGESTION RUNS */}
                {activeTab === "history" && (
                  <section className="retro-outset p-1 space-y-3 animate-fade-in">
                    <div className="bg-[#1F4743] text-white px-2.5 py-1 flex items-center justify-between text-xs font-bold font-mono tracking-wider">
                      <span>HISTORICAL INGESTION RUNS // ARCHIVE JOB JOURNAL</span>
                      <div className="flex items-center gap-1.5">
                        {selectedJobs.size > 0 && (
                          <button
                            type="button"
                            onClick={handleDeleteSelectedJobs}
                            disabled={deletingJobs}
                            className="retro-button px-2 py-0 text-[10px] font-mono font-bold text-red-700 dark:text-red-400 disabled:opacity-50"
                          >
                            {deletingJobs ? "DELETING..." : `DELETE SELECTED (${selectedJobs.size})`}
                          </button>
                        )}
                        <button
                          type="button"
                          onClick={loadHistoryJobs}
                          disabled={historyLoading}
                          className="retro-button px-2 py-0 text-[10px] font-mono font-bold text-[#1E2321] dark:text-[#E7E2D6] disabled:opacity-50"
                        >
                          {historyLoading ? "REFRESHING..." : "↻ REFRESH"}
                        </button>
                      </div>
                    </div>

                    <div className="p-3 bg-[#E7E2D6] dark:bg-[#1A201E] space-y-3">
                      {historyError && (
                        <div className="retro-outset p-2 text-xs font-mono text-rose-800 dark:text-rose-200 bg-[#EED2D2] dark:bg-[#3D1A1A] border border-rose-600">
                          {historyError}
                        </div>
                      )}

                      {historyLoading && historyJobs.length === 0 && (
                        <div className="retro-inset p-8 text-center space-y-2 bg-[#DFD9CD] dark:bg-[#141817]">
                          <div className="h-5 w-5 animate-spin mx-auto border-2 border-[#1F4743] border-t-transparent" />
                          <p className="font-mono text-xs text-[#555C58] dark:text-[#8C9893]">
                            Loading historical ingestion records...
                          </p>
                        </div>
                      )}

                      {!historyLoading && historyJobs.length === 0 && !historyError && (
                        <div className="retro-inset p-8 text-center space-y-2 bg-[#DFD9CD] dark:bg-[#141817] font-mono">
                          <span className="text-2xl">📦</span>
                          <h4 className="text-xs font-bold text-[#1E2321] dark:text-[#E7E2D6]">
                            NO PREVIOUS INGESTION RUNS
                          </h4>
                          <p className="text-[11px] text-[#555C58] dark:text-[#8C9893] max-w-md mx-auto">
                            No past ingestion jobs were found on the backend. Upload raw Chandrayaan-2 PRADAN ZIP bundles to start your first discovery run.
                          </p>
                          <button
                            type="button"
                            onClick={() => setActiveTab("new")}
                            className="retro-button-primary px-3 py-1 text-xs font-bold mt-2"
                          >
                            Upload PRADAN Files Now
                          </button>
                        </div>
                      )}

                      {historyJobs.length > 0 && (
                        <div className="overflow-x-auto retro-inset p-0.5 bg-[#DFD9CD] dark:bg-[#161B19]">
                          <table className="w-full text-left text-xs border-collapse">
                            <thead>
                              <tr className="border-b-2 border-[#8B8579] dark:border-[#2D3835] bg-[#E7E2D6] dark:bg-[#1A201E] text-[10px] font-mono font-bold uppercase tracking-wider text-[#1E2321] dark:text-[#E7E2D6]">
                                <th className="py-2 px-3">
                                  <input
                                    type="checkbox"
                                    aria-label="Select all jobs"
                                    checked={historyJobs.length > 0 && selectedJobs.size === historyJobs.length}
                                    onChange={(e) => {
                                      if (e.target.checked) {
                                        setSelectedJobs(new Set(historyJobs.map((j) => j.job_id)));
                                      } else {
                                        setSelectedJobs(new Set());
                                      }
                                    }}
                                    className="accent-[#28557E]"
                                  />
                                </th>
                                <th className="py-2 px-3">Job ID</th>
                                <th className="py-2 px-3">Status</th>
                                <th className="py-2 px-3">Current Stage</th>
                                <th className="py-2 px-3">Progress</th>
                                <th className="py-2 px-3">Started</th>
                                <th className="py-2 px-3 text-right">Action</th>
                              </tr>
                            </thead>
                            <tbody className="divide-y divide-[#8B8579]/30 dark:divide-[#2D3835] font-mono text-xs">
                              {historyJobs.map((job) => {
                                const statusColor =
                                  job.status === "completed"
                                    ? "bg-[#D3E8D7] dark:bg-[#193A24] text-[#134E26] dark:text-[#88D49E] border border-[#7BB887]"
                                    : job.status === "running"
                                    ? "bg-[#28557E] text-white border border-[#173857] animate-pulse"
                                    : job.status === "failed"
                                    ? "bg-[#EED2D2] dark:bg-[#3D1A1A] text-[#7A1D1D] dark:text-[#E89898] border border-rose-500"
                                    : "bg-[#DFD9CD] dark:bg-[#141817] text-[#555C58] dark:text-[#8C9893]";

                                return (
                                  <tr key={job.job_id} className="hover:bg-[#D5CFC1] dark:hover:bg-white/5 transition">
                                    <td className="py-2 px-3">
                                      <input
                                        type="checkbox"
                                        aria-label={`Select job ${job.job_id}`}
                                        checked={selectedJobs.has(job.job_id)}
                                        onChange={(e) => {
                                          setSelectedJobs((prev) => {
                                            const next = new Set(prev);
                                            if (e.target.checked) {
                                              next.add(job.job_id);
                                            } else {
                                              next.delete(job.job_id);
                                            }
                                            return next;
                                          });
                                        }}
                                        className="accent-[#28557E]"
                                      />
                                    </td>
                                    <td className="py-2 px-3 font-bold text-[#1E2321] dark:text-[#E7E2D6]">
                                      {job.job_id.slice(0, 8)}...{job.job_id.slice(-4)}
                                    </td>
                                    <td className="py-2 px-3">
                                      <span className={`retro-inset px-2 py-0.5 text-[9px] font-bold uppercase tracking-wider inline-flex items-center gap-1 ${statusColor}`}>
                                        <span className="h-1.5 w-1.5 rounded-full bg-current" />
                                        {job.status}
                                      </span>
                                    </td>
                                    <td className="py-2 px-3 text-[#555C58] dark:text-[#8C9893] text-[11px]">
                                      {job.stage || "—"}
                                    </td>
                                    <td className="py-2 px-3">
                                      <div className="flex items-center gap-2">
                                        <div className="h-2 w-20 retro-inset bg-[#ECE7DC] dark:bg-[#141817] overflow-hidden">
                                          <div
                                            className="h-full bg-[#28557E]"
                                            style={{ width: `${job.progress_pct || 0}%` }}
                                          />
                                        </div>
                                        <span className="text-[10px] text-[#555C58] dark:text-[#8C9893] font-bold">
                                          {Math.round(job.progress_pct || 0)}%
                                        </span>
                                      </div>
                                    </td>
                                    <td className="py-2 px-3 text-[10px] text-[#555C58] dark:text-[#8C9893]">
                                      {job.started_at ? new Date(job.started_at).toLocaleString() : "—"}
                                    </td>
                                    <td className="py-2 px-3 text-right">
                                      <div className="flex items-center justify-end gap-1.5">
                                        {job.status === "completed" && (
                                          <button
                                            type="button"
                                            onClick={() => handleSelectHistoricalJob(job.job_id)}
                                            className="retro-button-primary px-2 py-0.5 text-[10px] font-mono font-bold"
                                          >
                                            LOAD RESULTS →
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
                                            className="retro-button px-2 py-0.5 text-[10px] font-mono font-bold text-[#1E2321] dark:text-[#E7E2D6]"
                                          >
                                            ATTACH &amp; MONITOR
                                          </button>
                                        )}
                                        {job.status !== "running" && job.status !== "pending" && (
                                          <button
                                            type="button"
                                            title="Delete this history entry"
                                            onClick={() => handleDeleteHistoryJob(job.job_id)}
                                            disabled={deletingJobs}
                                            className="retro-button px-2 py-0.5 text-xs text-rose-700 dark:text-rose-400 hover:bg-rose-100 dark:hover:bg-rose-950/40 disabled:opacity-50"
                                          >
                                            🗑
                                          </button>
                                        )}
                                      </div>
                                    </td>
                                  </tr>
                                );
                              })}
                            </tbody>
                          </table>
                        </div>
                      )}
                    </div>
                  </section>
                )}
              </>
            )}
          </main>
        </div>
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
