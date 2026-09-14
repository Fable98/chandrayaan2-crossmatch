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

function getStageIndex(stage: string): number {
  const idx = STAGE_LABELS.findIndex(
    (s) => stage.toLowerCase().includes(s.toLowerCase().slice(0, 10))
  );
  return idx >= 0 ? idx : -1;
}

function getLogLineClass(line: string): string {
  if (line.includes('Stage ') && line.includes('/6')) return 'text-indigo-400 font-bold';
  if (line.includes('ERROR') || line.includes('FAIL')) return 'text-rose-400 font-bold';
  if (line.includes('[OK]') || line.includes('PASS')) return 'text-emerald-400 font-semibold';
  return 'text-slate-300';
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
    <div className="rounded-2xl border border-slate-200/80 dark:border-[#1b2029] bg-white dark:bg-[#0e1117] p-6 shadow-sm space-y-5 animate-fade-in">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-3">
          {isRunning && (
            <span className="relative flex h-3 w-3">
              <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-indigo-400 opacity-75" />
              <span className="relative inline-flex rounded-full h-3 w-3 bg-[#4F46E5]" />
            </span>
          )}
          {isDone && (
            <span className="flex h-6 w-6 items-center justify-center rounded-full bg-emerald-100 dark:bg-emerald-950/50 text-emerald-700 dark:text-emerald-300 text-xs font-bold">
              ✓
            </span>
          )}
          {isFailed && (
            <span className="flex h-6 w-6 items-center justify-center rounded-full bg-rose-100 dark:bg-rose-950/50 text-rose-700 dark:text-rose-300 text-xs font-bold">
              ✕
            </span>
          )}
          <span className="text-sm font-bold text-slate-900 dark:text-slate-100">{status.stage}</span>
        </div>
        <span className="font-mono text-xs font-bold text-[#4F46E5] dark:text-indigo-300 bg-indigo-50 dark:bg-indigo-950/30 px-2.5 py-1 rounded-lg border border-indigo-100 dark:border-indigo-900/50">
          {Math.round(status.progress_pct)}%
        </span>
      </div>

      {/* Progress bar */}
      <div className="h-2 w-full rounded-full bg-slate-100 dark:bg-white/10 overflow-hidden">
        <div
          className="h-full bg-[#4F46E5] transition-all duration-300 rounded-full"
          style={{ width: `${status.progress_pct}%` }}
        />
      </div>

      {/* Stage chips */}
      <div className="flex gap-1.5">
        {STAGE_LABELS.slice(0, 7).map((label, i) => {
          let bg = 'bg-slate-200';
          if (i < stageIdx || isDone) bg = 'bg-emerald-500';
          else if (i === stageIdx && isRunning) bg = 'bg-[#4F46E5]';
          else if (isFailed && i === stageIdx) bg = 'bg-rose-500';
          return (
            <div
              key={label}
              className={`h-1.5 flex-1 rounded-full transition-all duration-300 ${bg}`}
              title={label}
            />
          );
        })}
      </div>

      {/* Error banner */}
      {isFailed && status.error && (
        <div className="rounded-xl border border-rose-200 dark:border-rose-900/50 bg-rose-50 dark:bg-rose-950/30 p-4 text-xs text-rose-700 dark:text-rose-300 font-medium">
          <strong className="font-bold">Error:</strong> {status.error}
        </div>
      )}

      {/* Log console */}
      <div className="space-y-2">
        <div className="flex items-center justify-between text-[11px] font-bold uppercase tracking-wider text-slate-400">
          <span>Pipeline Output</span>
          <span className="font-mono">{status.log_lines.length} line(s)</span>
        </div>
        <div
          ref={logRef}
          className="rounded-xl bg-slate-950 p-4 font-mono text-xs leading-relaxed text-slate-300 border border-slate-800 shadow-inner max-h-72 overflow-y-auto space-y-1"
        >
          {status.log_lines.map((line, i) => (
            <div key={i} className={getLogLineClass(line)}>
              {line}
            </div>
          ))}
          {status.log_lines.length === 0 && (
            <div className="text-slate-500 italic">
              Waiting for pipeline output...
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
