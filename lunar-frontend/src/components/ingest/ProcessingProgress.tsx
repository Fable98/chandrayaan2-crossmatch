import React, { useEffect, useRef } from 'react';
import type { JobStatus } from '@/lib/ingest-api';

interface ProcessingProgressProps {
  status: JobStatus;
}

const STAGE_LABELS = [
  'Uploading files...',
  'Unzipping & discovering files...',
  'Parsing PDS4 metadata...',
  'Matching triplets...',
  'Processing crops & tiles...',
  'Updating manifest...',
  'Generating summary...',
  'Done!',
];

const styles: Record<string, React.CSSProperties> = {
  wrapper: {
    padding: '28px',
    borderRadius: 'var(--radius-lg)',
    background: 'var(--bg-glass)',
    backdropFilter: 'blur(20px)',
    border: '1px solid var(--border-subtle)',
    boxShadow: 'var(--shadow-md)',
  },
  header: {
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'space-between',
    marginBottom: '20px',
  },
  headerLeft: {
    display: 'flex',
    alignItems: 'center',
    gap: '12px',
  },
  stageLabel: {
    fontSize: '1rem',
    fontWeight: 600,
    color: 'var(--text-primary)',
  },
  pctLabel: {
    fontSize: '0.8rem',
    fontWeight: 700,
    color: 'var(--accent-secondary)',
    fontFamily: 'var(--font-mono)',
  },
  stagesRow: {
    display: 'flex',
    gap: '4px',
    marginBottom: '24px',
    marginTop: '12px',
  },
  stageChip: {
    flex: 1,
    height: '4px',
    borderRadius: '2px',
    transition: 'all 400ms cubic-bezier(0.16, 1, 0.3, 1)',
  },
  logHeader: {
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'space-between',
    marginTop: '20px',
    marginBottom: '8px',
  },
  logTitle: {
    fontSize: '0.75rem',
    fontWeight: 600,
    color: 'var(--text-muted)',
    textTransform: 'uppercase' as const,
    letterSpacing: '0.08em',
  },
  errorBanner: {
    marginTop: '16px',
    padding: '12px 16px',
    borderRadius: 'var(--radius-md)',
    background: 'var(--accent-danger-bg)',
    border: '1px solid rgba(248, 113, 113, 0.3)',
    color: 'var(--accent-danger)',
    fontSize: '0.85rem',
    fontWeight: 500,
  },
};

function getStageIndex(stage: string): number {
  const idx = STAGE_LABELS.findIndex(
    (s) => stage.toLowerCase().includes(s.toLowerCase().slice(0, 10))
  );
  return idx >= 0 ? idx : -1;
}

function classifyLogLine(line: string): string {
  if (line.includes('Stage ') && line.includes('/6'))  return 'log-line log-line--stage';
  if (line.includes('ERROR') || line.includes('FAIL')) return 'log-line log-line--error';
  if (line.includes('[OK]') || line.includes('PASS'))  return 'log-line log-line--ok';
  return 'log-line log-line--info';
}

export default function ProcessingProgress({ status }: ProcessingProgressProps) {
  const logRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (logRef.current) {
      logRef.current.scrollTop = logRef.current.scrollHeight;
    }
  }, [status.log_lines]);

  const stageIdx = getStageIndex(status.stage);
  const isRunning = status.status === 'running' || status.status === 'pending';
  const isDone = status.status === 'completed';
  const isFailed = status.status === 'failed';

  return (
    <div style={styles.wrapper} className="animate-fade-in">
      {/* Header */}
      <div style={styles.header}>
        <div style={styles.headerLeft}>
          {isRunning && <div className="spinner" />}
          {isDone && <span style={{ fontSize: '1.2rem' }}>&#10003;</span>}
          {isFailed && <span style={{ fontSize: '1.2rem', color: 'var(--accent-danger)' }}>&#10007;</span>}
          <span style={styles.stageLabel}>{status.stage}</span>
        </div>
        <span style={styles.pctLabel}>
          {Math.round(status.progress_pct)}%
        </span>
      </div>

      {/* Progress bar */}
      <div className="progress-bar-track">
        <div
          className="progress-bar-fill"
          style={{ width: `${status.progress_pct}%` }}
        />
      </div>

      {/* Stage chips */}
      <div style={styles.stagesRow}>
        {STAGE_LABELS.slice(0, 7).map((label, i) => {
          let bg = 'var(--bg-elevated)';
          if (i < stageIdx || isDone) bg = 'var(--accent-success)';
          else if (i === stageIdx && isRunning) bg = 'var(--accent-primary)';
          else if (isFailed && i === stageIdx) bg = 'var(--accent-danger)';
          return (
            <div
              key={label}
              style={{ ...styles.stageChip, background: bg }}
              title={label}
            />
          );
        })}
      </div>

      {/* Error banner */}
      {isFailed && status.error && (
        <div style={styles.errorBanner}>
          <strong>Error:</strong> {status.error}
        </div>
      )}

      {/* Log console */}
      <div style={styles.logHeader}>
        <span style={styles.logTitle}>Pipeline Output</span>
        <span style={{ fontSize: '0.7rem', color: 'var(--text-muted)' }}>
          {status.log_lines.length} line(s)
        </span>
      </div>
      <div className="log-console" ref={logRef}>
        {status.log_lines.map((line, i) => (
          <div key={i} className={classifyLogLine(line)}>
            {line}
          </div>
        ))}
        {status.log_lines.length === 0 && (
          <div className="log-line log-line--info" style={{ opacity: 0.5 }}>
            Waiting for pipeline output...
          </div>
        )}
      </div>
    </div>
  );
}
