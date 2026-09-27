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

  const copyLogs = () => {
    if (status.log_lines) {
      navigator.clipboard.writeText(status.log_lines.join('\n'));
    }
  };

  return (
    <div className="rounded-2xl border border-slate-200/80 dark:border-[#1b2029] bg-white dark:bg-[#0e1117] p-6 shadow-sm space-y-5 animate-fade-in">
      {/* Header */}
      <div className="flex flex-wrap items-center justify-between gap-3">
        <div className="flex items-center gap-3">
          {isRunning && (
            <span className="relative flex h-3 w-3">
              <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-indigo-400 opacity-75" />
              <span className="relative inline-flex rounded-full h-3 w-3 bg-[#4F46E5]" />
            </span>
          )}
          {isDone && (
            <span className="flex h-6 w-6 items-center justify-center rounded-full bg-emerald-100 dark:bg-emerald-950/50 text-emerald-700 dark:text-emerald-300 text-xs font-bold shadow-sm">
              ✓
            </span>
          )}
          {isFailed && (
            <span className="flex h-6 w-6 items-center justify-center rounded-full bg-rose-100 dark:bg-rose-950/50 text-rose-700 dark:text-rose-300 text-xs font-bold shadow-sm">
              ✕
            </span>
          )}
          <div>
            <span className="text-sm font-extrabold text-slate-900 dark:text-slate-100 tracking-tight block">
              {status.stage}
            </span>
            <span className="text-[10px] font-mono text-slate-400">
              Pipeline Stage {(stageIdx >= 0 ? stageIdx + 1 : 1)} of {STAGE_LABELS.length - 1}
            </span>
          </div>
        </div>

        <div className="flex items-center gap-2">
          <span className="hidden sm:inline-flex items-center gap-1.5 rounded-full border border-slate-200 dark:border-[#1b2029] bg-slate-50 dark:bg-white/5 px-2.5 py-1 font-mono text-[10px] text-slate-600 dark:text-slate-300">
            <span className="h-1.5 w-1.5 rounded-full bg-emerald-400" />
            ISRO PIPELINE DAEMON
          </span>
          <span className="font-mono text-xs font-extrabold text-[#4F46E5] dark:text-indigo-300 bg-indigo-50 dark:bg-indigo-950/40 px-3 py-1 rounded-xl border border-indigo-100 dark:border-indigo-900/50 shadow-xs">
            {Math.round(status.progress_pct)}%
          </span>
        </div>
      </div>

      {/* Modern Shimmer Progress Bar */}
      <div className="relative h-2.5 w-full rounded-full bg-slate-100 dark:bg-white/10 overflow-hidden shadow-inner">
        <div
          className="h-full bg-gradient-to-r from-indigo-500 via-[#4F46E5] to-cyan-400 transition-all duration-300 rounded-full relative"
          style={{ width: `${status.progress_pct}%` }}
        >
          <div className="absolute inset-0 bg-gradient-to-r from-transparent via-white/25 to-transparent animate-[progress-shimmer_2s_infinite]" />
        </div>
      </div>

      {/* Stage chips */}
      <div className="flex gap-1.5">
        {STAGE_LABELS.slice(0, 7).map((label, i) => {
          let bg = 'bg-slate-200 dark:bg-white/10';
          if (i < stageIdx || isDone) bg = 'bg-emerald-500 shadow-[0_0_8px_rgba(16,185,129,0.3)]';
          else if (i === stageIdx && isRunning) bg = 'bg-[#4F46E5] shadow-[0_0_10px_rgba(79,70,229,0.4)] animate-pulse';
          else if (isFailed && i === stageIdx) bg = 'bg-rose-500 shadow-[0_0_8px_rgba(244,63,94,0.3)]';
          return (
            <div
              key={label}
              className={`h-2 flex-1 rounded-full transition-all duration-300 ${bg}`}
              title={label}
            />
          );
        })}
      </div>

      {/* Error banner */}
      {isFailed && status.error && (
        <div className="rounded-xl border border-rose-200 dark:border-rose-900/50 bg-rose-50 dark:bg-rose-950/30 p-4 text-xs text-rose-700 dark:text-rose-300 font-medium animate-fade-in space-y-1">
          <div className="font-bold flex items-center gap-1.5">
            <span>⚠️</span> Pipeline Failure:
          </div>
          <div className="font-mono text-[11px] leading-relaxed break-words">{status.error}</div>
        </div>
      )}

      {/* Terminal / Log console with Aerospace HUD styling */}
      <div className="space-y-2 pt-1">
        <div className="flex items-center justify-between text-[11px] font-bold uppercase tracking-wider text-slate-400">
          <div className="flex items-center gap-2">
            {/* Terminal Window Controls */}
            <div className="flex items-center gap-1.5">
              <span className="h-2.5 w-2.5 rounded-full bg-rose-500/80" />
              <span className="h-2.5 w-2.5 rounded-full bg-amber-500/80" />
              <span className="h-2.5 w-2.5 rounded-full bg-emerald-500/80" />
            </div>
            <span className="font-mono text-[10px] text-slate-500 dark:text-slate-400 ml-1">
              pipeline-runtime.log
            </span>
          </div>

          <div className="flex items-center gap-2">
            <span className="font-mono text-[10px] text-slate-400">
              {status.log_lines.length} lines
            </span>
            {status.log_lines.length > 0 && (
              <button
                type="button"
                onClick={copyLogs}
                className="rounded-md border border-slate-200 dark:border-slate-800 bg-white/60 dark:bg-white/5 px-2 py-0.5 text-[10px] font-mono text-slate-600 dark:text-slate-300 hover:bg-slate-100 dark:hover:bg-white/10 transition"
                title="Copy log to clipboard"
              >
                Copy
              </button>
            )}
          </div>
        </div>

        <div
          ref={logRef}
          className="rounded-xl bg-[#090d14] p-4 font-mono text-xs leading-relaxed text-slate-300 border border-slate-800/80 shadow-2xl max-h-72 overflow-y-auto space-y-1"
        >
          {status.log_lines.map((line, i) => (
            <div key={i} className={getLogLineClass(line)}>
              {line}
            </div>
          ))}
          {status.log_lines.length === 0 && (
            <div className="text-slate-500 italic flex items-center gap-2">
              <span className="h-2 w-2 animate-ping rounded-full bg-indigo-400" />
              <span>Awaiting telemetry stream from ingestion backend...</span>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
