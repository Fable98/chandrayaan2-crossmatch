"use client";

import React, { useCallback, useEffect, useRef, useState } from 'react';
import DropZone from './DropZone';
import ProcessingProgress from './ProcessingProgress';
import ResultsTable from './ResultsTable';
import {
  uploadZips,
  pollStatus,
  getResults,
  DEFAULT_CONFIG,
  type IngestConfig,
  type JobStatus,
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
  const pollRef = useRef<number | null>(null);

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

  return (
    <div style={styles.page}>
      {/* Hero */}
      <div style={styles.hero} className="animate-fade-in">
        <h1 style={styles.heroTitle}>Ingest &amp; Prepare</h1>
        <p style={styles.heroSub}>
          Drop your PRADAN zip files below to automatically discover, match, and
          process Chandrayaan-2 OHRC + TMC-2 + IIRS triplets.
        </p>
      </div>

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
                <span style={styles.statValue}>{files.length}</span> file(s)
              </span>
              <span>
                <span style={styles.statValue}>{fmtSize(totalSize)}</span> total
              </span>
            </div>
            <div style={{ display: 'flex', gap: '8px' }}>
              {!isProcessing && (
                <button className="btn btn-secondary btn-sm" onClick={clearFiles}>
                  Clear All
                </button>
              )}
              {!isProcessing && (
                <button
                  className="btn btn-primary"
                  onClick={handleStart}
                  disabled={files.length === 0}
                >
                  Start Processing
                </button>
              )}
            </div>
          </div>
        </div>
      )}

      {/* Config panel */}
      {files.length > 0 && !isProcessing && phase !== 'done' && (
        <div style={styles.configPanel}>
          <div
            style={styles.configToggle}
            onClick={() => setShowConfig(!showConfig)}
          >
            <span style={styles.configTitle}>Pipeline Configuration</span>
            <span style={{ color: 'var(--text-muted)', fontSize: '0.8rem' }}>
              {showConfig ? '▲' : '▼'}
            </span>
          </div>

          {showConfig && (
            <div style={styles.configGrid} className="animate-fade-in">
              <div>
                <div style={styles.fieldLabel}>Containment Threshold</div>
                <input
                  type="number"
                  step="0.05"
                  min="0"
                  max="1"
                  style={styles.fieldInput}
                  value={config.containment}
                  onChange={(e) =>
                    setConfig({ ...config, containment: parseFloat(e.target.value) || 0.8 })
                  }
                  onFocus={(e) => { e.target.style.borderColor = 'var(--accent-primary)'; }}
                  onBlur={(e) => { e.target.style.borderColor = 'var(--border-default)'; }}
                />
              </div>
              <div>
                <div style={styles.fieldLabel}>Tile Size (px)</div>
                <input
                  type="number"
                  step="64"
                  min="128"
                  max="2048"
                  style={styles.fieldInput}
                  value={config.tileSize}
                  onChange={(e) =>
                    setConfig({ ...config, tileSize: parseInt(e.target.value) || 512 })
                  }
                  onFocus={(e) => { e.target.style.borderColor = 'var(--accent-primary)'; }}
                  onBlur={(e) => { e.target.style.borderColor = 'var(--border-default)'; }}
                />
              </div>
              <div>
                <div style={styles.fieldLabel}>Max Time Gap (days)</div>
                <input
                  type="number"
                  step="1"
                  min="0"
                  placeholder="unlimited"
                  style={styles.fieldInput}
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
      {phase === 'done' && status && (
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
    </div>
  );
}
