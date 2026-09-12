"use client";

import React, { useCallback, useEffect, useRef, useState } from 'react';
import Link from 'next/link';
import { isAuthenticated } from '@/lib/auth';
import DropZone from './DropZone';
import ProcessingProgress from './ProcessingProgress';
import ResultsTable from './ResultsTable';
import {
  uploadZips,
  pollStatus,
  getResults,
  listJobs,
  DEFAULT_CONFIG,
  type IngestConfig,
  type JobStatus,
  type IngestJobSummary,
} from '@/lib/ingest-api';

type Phase = 'idle' | 'queued' | 'processing' | 'done' | 'error';

const styles: Record<string, React.CSSProperties> = {
  page: {
    maxWidth: '1100px',
    margin: '0 auto',
    padding: '32px 24px 80px',
  },
  hero: {
    textAlign: 'center' as const,
    marginBottom: '40px',
  },
  heroTitle: {
    fontSize: '2rem',
    fontWeight: 800,
    background: 'linear-gradient(135deg, var(--accent-primary), var(--accent-secondary), #a78bfa)',
    WebkitBackgroundClip: 'text',
    WebkitTextFillColor: 'transparent',
    marginBottom: '8px',
    letterSpacing: '-0.02em',
  },
  heroSub: {
    color: 'var(--text-secondary)',
    fontSize: '0.95rem',
    maxWidth: '540px',
    margin: '0 auto',
    lineHeight: 1.7,
  },
  section: {
    marginBottom: '28px',
  },
  fileList: {
    display: 'flex',
    flexWrap: 'wrap' as const,
    gap: '8px',
    marginTop: '16px',
  },
  fileChip: {
    display: 'inline-flex',
    alignItems: 'center',
    gap: '8px',
    padding: '6px 14px',
    borderRadius: 'var(--radius-full)',
    background: 'var(--bg-elevated)',
    border: '1px solid var(--border-subtle)',
    fontSize: '0.75rem',
    color: 'var(--text-secondary)',
    transition: 'all 150ms ease',
  },
  fileSize: {
    fontSize: '0.65rem',
    color: 'var(--text-muted)',
    fontFamily: 'var(--font-mono)',
  },
  removeBtn: {
    background: 'none',
    border: 'none',
    color: 'var(--text-muted)',
    cursor: 'pointer',
    fontSize: '1rem',
    lineHeight: 1,
    padding: '0 2px',
    transition: 'color 150ms ease',
  },
  configPanel: {
    padding: '20px 24px',
    borderRadius: 'var(--radius-lg)',
    background: 'var(--bg-glass)',
    backdropFilter: 'blur(20px)',
    border: '1px solid var(--border-subtle)',
    marginTop: '20px',
  },
  configToggle: {
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'space-between',
    cursor: 'pointer',
    userSelect: 'none' as const,
  },
  configTitle: {
    fontSize: '0.8rem',
    fontWeight: 600,
    color: 'var(--text-secondary)',
    textTransform: 'uppercase' as const,
    letterSpacing: '0.06em',
  },
  configGrid: {
    display: 'grid',
    gridTemplateColumns: 'repeat(auto-fill, minmax(200px, 1fr))',
    gap: '16px',
    marginTop: '16px',
  },
  fieldLabel: {
    fontSize: '0.7rem',
    fontWeight: 600,
    color: 'var(--text-muted)',
    marginBottom: '4px',
    textTransform: 'uppercase' as const,
    letterSpacing: '0.04em',
  },
  fieldInput: {
    width: '100%',
    padding: '8px 12px',
    borderRadius: 'var(--radius-sm)',
    border: '1px solid var(--border-default)',
    background: 'var(--bg-secondary)',
    color: 'var(--text-primary)',
    fontFamily: 'var(--font-mono)',
    fontSize: '0.8rem',
    outline: 'none',
    transition: 'border-color 150ms ease',
  },
  checkboxRow: {
    display: 'flex',
    alignItems: 'center',
    gap: '8px',
    fontSize: '0.8rem',
    color: 'var(--text-secondary)',
    cursor: 'pointer',
    paddingTop: '4px',
  },
  actionsRow: {
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'space-between',
    gap: '12px',
    marginTop: '24px',
    flexWrap: 'wrap' as const,
  },
  stats: {
    display: 'flex',
    gap: '24px',
    fontSize: '0.8rem',
    color: 'var(--text-muted)',
  },
  statValue: {
    fontWeight: 700,
    color: 'var(--text-primary)',
    fontFamily: 'var(--font-mono)',
  },
};

function fmtSize(bytes: number): string {
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
  return `${(bytes / 1024 / 1024).toFixed(1)} MB`;
}

export default function IngestPage() {
  const [files, setFiles] = useState<File[]>([]);
  const [config, setConfig] = useState<IngestConfig>({ ...DEFAULT_CONFIG });
  const [showConfig, setShowConfig] = useState(false);
  const [phase, setPhase] = useState<Phase>('idle');
  const [jobId, setJobId] = useState<string | null>(null);
  const [status, setStatus] = useState<JobStatus | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [resultTriplets, setResultTriplets] = useState<Record<string, any>[]>([]);
  const [activeTab, setActiveTab] = useState<'new' | 'history'>('new');
  const [historyJobs, setHistoryJobs] = useState<IngestJobSummary[]>([]);
  const [historyLoading, setHistoryLoading] = useState(false);
  const [historyError, setHistoryError] = useState<string | null>(null);
  const pollRef = useRef<number | null>(null);

  const loadHistoryJobs = useCallback(async () => {
    setHistoryLoading(true);
    setHistoryError(null);
    try {
      const jobs = await listJobs();
      setHistoryJobs(jobs);
    } catch (err: any) {
      setHistoryError(err.message || 'Failed to load past ingestion jobs');
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
      setPhase('done');
      setActiveTab('new');
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
    setPhase('idle');
    setJobId(null);
    setStatus(null);
    setError(null);
    setResultTriplets([]);
  }, []);

  // Start processing
  const handleStart = useCallback(async () => {
    if (files.length === 0) return;

    setPhase('queued');
    setError(null);

    try {
      const res = await uploadZips(files, config);
      setJobId(res.job_id);
      setPhase('processing');
    } catch (e: any) {
      setError(e.message || 'Upload failed');
      setPhase('error');
    }
  }, [files, config]);

  // Poll loop
  useEffect(() => {
    if (phase !== 'processing' || !jobId) return;

    const poll = async () => {
      try {
        const s = await pollStatus(jobId);
        setStatus(s);

        if (s.status === 'completed') {
          setPhase('done');
          // Fetch full results
          try {
            const results = await getResults(jobId);
            setResultTriplets(results.triplets || []);
          } catch {
            // Results fetch failed, triplets stay empty
          }
        } else if (s.status === 'failed') {
          setPhase('error');
          setError(s.error || 'Pipeline failed');
        }
      } catch {
        // Transient error, keep polling
      }
    };

    poll(); // initial
    const id = window.setInterval(poll, 1500);
    pollRef.current = id;

    return () => {
      if (pollRef.current !== null) {
        window.clearInterval(pollRef.current);
      }
    };
  }, [phase, jobId]);

  const totalSize = files.reduce((s, f) => s + f.size, 0);
  const isProcessing = phase === 'processing' || phase === 'queued';

  // Uploads require a Bearer token (backend enforces auth); explain instead
  // of showing a form that can only 401.
  if (!isAuthenticated()) {
    return (
      <div style={styles.page}>
        <div style={{ ...styles.configPanel, textAlign: 'center' as const, padding: '48px 24px' }}>
          <div style={{ fontSize: '2rem', marginBottom: '12px' }}>🔐</div>
          <h1 style={{ fontSize: '1.25rem', fontWeight: 800, color: 'var(--text-primary)' }}>
            Sign in required
          </h1>
          <p style={{ ...styles.heroSub, marginTop: '8px' }}>
            The ingest pipeline writes to disk and launches processing jobs, so
            uploads require an operator session. Please sign in from the home
            page, then return here.
          </p>
          <div style={{ marginTop: '20px' }}>
            <Link href="/" className="btn btn-primary">
              Go to Sign In
            </Link>
          </div>
        </div>
      </div>
    );
  }

  return (
    <div style={styles.page}>
      {/* Top Navigation Header */}
      <header className="flex items-center justify-between py-4 mb-8 border-b border-white/10">
        <Link
          href="/"
          className="group flex items-center gap-2 font-mono text-xs text-[#9a958e] transition-colors hover:text-white"
        >
          <span className="transition-transform duration-200 group-hover:-translate-x-1">
            ←
          </span>
          <span>Return to Lunar Globe</span>
        </Link>

        <div className="hidden items-center gap-2 font-mono text-xs font-semibold uppercase tracking-widest text-[#d4af37] md:flex">
          <span className="h-1.5 w-1.5 rounded-full bg-[#d4af37]" />
          <span>SIH 26166 · Ingest &amp; Prepare Pipeline</span>
        </div>

        <Link
          href="/?view=console"
          className="flex items-center gap-2 rounded-full border border-teal/40 bg-teal/10 px-4 py-1.5 font-mono text-xs font-semibold text-teal backdrop-blur-sm transition-all duration-200 hover:bg-teal/20 hover:scale-105"
        >
          <span>Open Dashboard</span>
          <span>↗</span>
        </Link>
      </header>

      {/* Hero */}
      <div style={styles.hero} className="animate-fade-in">
        <h1 style={styles.heroTitle}>Ingest &amp; Prepare</h1>
        <p style={styles.heroSub}>
          Drop your PRADAN zip files below to automatically discover, match, and
          process Chandrayaan-2 OHRC + TMC-2 + IIRS triplets.
        </p>
      </div>

      {/* Tab Switcher */}
      <div className="flex items-center justify-center gap-3 mb-8">
        <button
          type="button"
          onClick={() => setActiveTab('new')}
          className={`flex items-center gap-2 rounded-xl px-5 py-2.5 font-mono text-xs font-bold transition-all ${
            activeTab === 'new'
              ? 'bg-[#d4af37] text-black shadow-lg shadow-[#d4af37]/20 scale-105'
              : 'border border-white/10 bg-white/5 text-white/70 hover:bg-white/10 hover:text-white'
          }`}
        >
          <span>＋ New Batch Ingestion</span>
        </button>
        <button
          type="button"
          onClick={() => {
            setActiveTab('history');
            loadHistoryJobs();
          }}
          className={`flex items-center gap-2 rounded-xl px-5 py-2.5 font-mono text-xs font-bold transition-all ${
            activeTab === 'history'
              ? 'bg-[#d4af37] text-black shadow-lg shadow-[#d4af37]/20 scale-105'
              : 'border border-white/10 bg-white/5 text-white/70 hover:bg-white/10 hover:text-white'
          }`}
        >
          <span>📂 Previous Ingestion Runs</span>
          {historyJobs.length > 0 && (
            <span className="rounded-full bg-white/20 px-2 py-0.5 text-[10px] font-bold">
              {historyJobs.length}
            </span>
          )}
        </button>
      </div>

      {/* Tab 1: New Batch Ingestion */}
      {activeTab === 'new' && (
        <>
          {/* Drop Zone */}
          {(phase === 'idle' || phase === 'queued') && (
            <div style={styles.section}>
              <DropZone
                onFilesSelected={handleFilesSelected}
                disabled={isProcessing}
              />
            </div>
          )}

          {/* File list */}
          {files.length > 0 && phase !== 'done' && (
            <div style={styles.section} className="animate-fade-in">
              <div style={styles.fileList}>
                {files.map((f) => (
                  <div key={f.name} style={styles.fileChip}>
                    <span>&#128230;</span>
                    <span>{f.name}</span>
                    <span style={styles.fileSize}>{fmtSize(f.size)}</span>
                    {!isProcessing && (
                      <button
                        style={styles.removeBtn}
                        onClick={() => removeFile(f.name)}
                        title="Remove file"
                        onMouseEnter={(e) => {
                          (e.target as HTMLElement).style.color = 'var(--accent-danger)';
                        }}
                        onMouseLeave={(e) => {
                          (e.target as HTMLElement).style.color = 'var(--text-muted)';
                        }}
                      >
                        &times;
                      </button>
                    )}
                  </div>
                ))}
              </div>

              {/* Stats + Actions */}
              <div style={styles.actionsRow}>
                <div style={styles.stats}>
                  <span>
                    Files: <span style={styles.statValue}>{files.length}</span>
                  </span>
                  <span>
                    Total: <span style={styles.statValue}>{fmtSize(totalSize)}</span>
                  </span>
                </div>
                <div style={{ display: 'flex', gap: '8px' }}>
                  <button
                    className="btn btn-secondary btn-sm"
                    onClick={clearFiles}
                    disabled={isProcessing}
                  >
                    Clear All
                  </button>
                  <button
                    className="btn btn-primary btn-md"
                    onClick={handleStart}
                    disabled={isProcessing}
                  >
                    Start Ingestion ({files.length})
                  </button>
                </div>
              </div>
            </div>
          )}

          {/* Config panel toggle */}
          {phase === 'idle' && (
            <div style={styles.configPanel}>
              <div
                style={styles.configToggle}
                onClick={() => setShowConfig(!showConfig)}
              >
                <span style={styles.configTitle}>
                  Pipeline Configuration
                </span>
                <span style={{ fontSize: '0.75rem', color: 'var(--text-muted)' }}>
                  {showConfig ? '▲ Hide' : '▼ Show Advanced'}
                </span>
              </div>

              {showConfig && (
                <div style={{ display: 'flex', flexDirection: 'column', gap: '14px', marginTop: '16px' }}>
                  <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(200px, 1fr))', gap: '12px' }}>
                    <div>
                      <label style={styles.label}>
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
                        style={{ width: '100%' }}
                      />
                    </div>
                    <div>
                      <label style={styles.label}>Tile Size (px)</label>
                      <select
                        style={styles.input}
                        value={config.tileSize}
                        onChange={(e) =>
                          setConfig({ ...config, tileSize: parseInt(e.target.value) })
                        }
                      >
                        <option value={256}>256 × 256</option>
                        <option value={512}>512 × 512 (Standard)</option>
                        <option value={1024}>1024 × 1024</option>
                      </select>
                    </div>
                  </div>
                  <div>
                    <label style={styles.label}>Max Time Gap (Days, Optional)</label>
                    <input
                      type="number"
                      placeholder="e.g. 180 (empty = any)"
                      style={styles.input}
                      value={config.maxTimeGapDays ?? ''}
                      onChange={(e) =>
                        setConfig({
                          ...config,
                          maxTimeGapDays: e.target.value ? parseFloat(e.target.value) : null,
                        })
                      }
                      onFocus={(e) => { e.target.style.borderColor = 'var(--accent-primary)'; }}
                      onBlur={(e) => { e.target.style.borderColor = 'var(--border-default)'; }}
                    />
                  </div>
                  <div style={{ display: 'flex', flexDirection: 'column', gap: '8px' }}>
                    <label style={styles.checkboxRow}>
                      <input
                        type="checkbox"
                        checked={config.noLargeAoi}
                        onChange={(e) =>
                          setConfig({ ...config, noLargeAoi: e.target.checked })
                        }
                      />
                      Skip Large-AOI IIRS
                    </label>
                    <label style={styles.checkboxRow}>
                      <input
                        type="checkbox"
                        checked={config.noInvariants}
                        onChange={(e) =>
                          setConfig({ ...config, noInvariants: e.target.checked })
                        }
                      />
                      Skip Invariant Maps
                    </label>
                    <label style={styles.checkboxRow}>
                      <input
                        type="checkbox"
                        checked={config.requireDates}
                        onChange={(e) =>
                          setConfig({ ...config, requireDates: e.target.checked })
                        }
                      />
                      Require Dates
                    </label>
                  </div>
                </div>
              )}
            </div>
          )}

          {/* Error banner */}
          {phase === 'error' && error && !status && (
            <div
              style={{
                marginTop: '24px',
                padding: '16px 20px',
                borderRadius: 'var(--radius-md)',
                background: 'var(--accent-danger-bg)',
                border: '1px solid rgba(248, 113, 113, 0.3)',
                color: 'var(--accent-danger)',
                fontSize: '0.9rem',
              }}
              className="animate-fade-in"
            >
              <strong>Error:</strong> {error}
              <div style={{ marginTop: '12px' }}>
                <button className="btn btn-secondary btn-sm" onClick={clearFiles}>
                  Start Over
                </button>
              </div>
            </div>
          )}

          {/* Processing progress */}
          {status && (phase === 'processing' || phase === 'done' || phase === 'error') && (
            <div style={styles.section}>
              <ProcessingProgress status={status} />
            </div>
          )}

          {/* Results */}
          {phase === 'done' && (resultTriplets.length > 0 || status) && (
            <div style={styles.section} className="animate-fade-in">
              <ResultsTable
                triplets={resultTriplets}
                containment={config.containment}
              />

              <div style={{ marginTop: '20px', textAlign: 'center' }}>
                <button className="btn btn-primary btn-lg" onClick={clearFiles}>
                  Process Another Batch
                </button>
              </div>
            </div>
          )}
        </>
      )}

      {/* Tab 2: Previous Ingestion Runs */}
      {activeTab === 'history' && (
        <div className="space-y-4 animate-fade-in">
          <div className="flex items-center justify-between border-b border-white/10 pb-4">
            <div>
              <h2 className="text-lg font-bold text-white">Previous Ingestion Runs</h2>
              <p className="text-xs text-slate-400">
                Inspect history and replay discovery results from past PRADAN archive batches.
              </p>
            </div>
            <button
              type="button"
              onClick={loadHistoryJobs}
              disabled={historyLoading}
              className="flex items-center gap-1.5 rounded-lg border border-white/10 bg-white/5 px-3 py-1.5 font-mono text-xs font-semibold text-white/85 transition hover:bg-white/10"
            >
              <span>{historyLoading ? "Refreshing..." : "↻ Refresh History"}</span>
            </button>
          </div>

          {historyError && (
            <div className="rounded-xl border border-rose-500/30 bg-rose-500/10 p-4 text-xs text-rose-300">
              {historyError}
            </div>
          )}

          {historyLoading && historyJobs.length === 0 && (
            <div className="flex h-48 flex-col items-center justify-center gap-3 rounded-2xl border border-white/10 bg-white/[0.02] p-8 text-center">
              <div className="h-6 w-6 animate-spin rounded-full border-2 border-[#d4af37] border-t-transparent" />
              <p className="font-mono text-xs text-slate-400">Loading historical ingestion records...</p>
            </div>
          )}

          {!historyLoading && historyJobs.length === 0 && !historyError && (
            <div className="flex flex-col items-center justify-center gap-3 rounded-2xl border border-dashed border-white/10 bg-white/[0.02] p-12 text-center">
              <span className="text-3xl">📦</span>
              <h3 className="text-sm font-bold text-white">No Previous Ingestion Runs</h3>
              <p className="max-w-md text-xs text-slate-400 leading-relaxed">
                No past ingestion jobs were found on the backend. Upload raw Chandrayaan-2 PRADAN ZIP bundles to start your first discovery run.
              </p>
              <button
                type="button"
                onClick={() => setActiveTab('new')}
                className="mt-2 rounded-xl bg-[#d4af37] px-4 py-2 font-mono text-xs font-bold text-black shadow-sm transition hover:bg-[#c29f2f]"
              >
                Upload PRADAN Files Now
              </button>
            </div>
          )}

          {historyJobs.length > 0 && (
            <div className="overflow-hidden rounded-2xl border border-white/10 bg-black/40 backdrop-blur-md shadow-xl">
              <table className="w-full text-left font-sans text-xs">
                <thead>
                  <tr className="border-b border-white/10 bg-white/5 font-mono text-[10px] uppercase tracking-wider text-slate-400">
                    <th className="py-3 px-4">Job ID</th>
                    <th className="py-3 px-4">Status</th>
                    <th className="py-3 px-4">Current Stage</th>
                    <th className="py-3 px-4">Progress</th>
                    <th className="py-3 px-4">Started</th>
                    <th className="py-3 px-4 text-right">Action</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-white/5">
                  {historyJobs.map((job) => {
                    const statusColor =
                      job.status === 'completed'
                        ? 'border-emerald-500/30 bg-emerald-500/10 text-emerald-400'
                        : job.status === 'running'
                        ? 'border-blue-500/30 bg-blue-500/10 text-blue-400 animate-pulse'
                        : job.status === 'failed'
                        ? 'border-rose-500/30 bg-rose-500/10 text-rose-400'
                        : 'border-slate-500/30 bg-slate-500/10 text-slate-400';

                    return (
                      <tr key={job.job_id} className="transition hover:bg-white/[0.03]">
                        <td className="py-3.5 px-4 font-mono font-medium text-white/90">
                          {job.job_id.slice(0, 8)}...{job.job_id.slice(-4)}
                        </td>
                        <td className="py-3.5 px-4">
                          <span className={`inline-flex items-center gap-1 rounded-full border px-2.5 py-0.5 text-[10px] font-bold uppercase tracking-wider ${statusColor}`}>
                            <span className="h-1.5 w-1.5 rounded-full bg-current" />
                            {job.status}
                          </span>
                        </td>
                        <td className="py-3.5 px-4 text-slate-300 font-mono text-[11px]">
                          {job.stage || '—'}
                        </td>
                        <td className="py-3.5 px-4">
                          <div className="flex items-center gap-2">
                            <div className="h-1.5 w-24 overflow-hidden rounded-full bg-white/10">
                              <div
                                className="h-full bg-[#d4af37] transition-all duration-300"
                                style={{ width: `${job.progress_pct || 0}%` }}
                              />
                            </div>
                            <span className="font-mono text-[10px] text-slate-400">
                              {Math.round(job.progress_pct || 0)}%
                            </span>
                          </div>
                        </td>
                        <td className="py-3.5 px-4 font-mono text-[10px] text-slate-400">
                          {job.started_at ? new Date(job.started_at).toLocaleString() : '—'}
                        </td>
                        <td className="py-3.5 px-4 text-right">
                          {job.status === 'completed' && (
                            <button
                              type="button"
                              onClick={() => handleSelectHistoricalJob(job.job_id)}
                              className="rounded-lg border border-[#d4af37]/30 bg-[#d4af37]/10 px-3 py-1 font-mono text-xs font-semibold text-[#d4af37] transition hover:bg-[#d4af37]/20"
                            >
                              Load Results →
                            </button>
                          )}
                          {job.status === 'running' && (
                            <button
                              type="button"
                              onClick={() => {
                                setJobId(job.job_id);
                                setPhase('processing');
                                setActiveTab('new');
                              }}
                              className="rounded-lg border border-blue-400/30 bg-blue-400/10 px-3 py-1 font-mono text-xs font-semibold text-blue-300 transition hover:bg-blue-400/20"
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
        </div>
      )}
    </div>
  );
}
