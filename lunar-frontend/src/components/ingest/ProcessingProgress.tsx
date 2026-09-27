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
    <div className="retro-outset p-1 animate-fade-in">
      {/* Window Title Bar */}
      <div className="bg-[#1F4743] text-white px-2.5 py-1 flex items-center justify-between text-xs font-bold font-mono tracking-wider">
        <div className="flex items-center gap-2">
          {isRunning && (
            <span className="relative flex h-2 w-2">
              <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-emerald-400 opacity-75" />
              <span className="relative inline-flex rounded-full h-2 w-2 bg-emerald-500" />
            </span>
          )}
          {isDone && <span className="text-emerald-300 font-bold">✓</span>}
          {isFailed && <span className="text-rose-300 font-bold">✕</span>}
          <span>PIPELINE TELEMETRY MONITOR // {status.stage.toUpperCase()}</span>
        </div>
        <div className="flex items-center gap-2 font-mono text-[10px]">
          <span className="retro-inset px-2 py-0.5 bg-[#143532] text-white font-bold">
            STAGE {(stageIdx >= 0 ? stageIdx + 1 : 1)} / {STAGE_LABELS.length - 1}
          </span>
          <span className="retro-inset px-2 py-0.5 bg-[#28557E] text-white font-bold">
            {Math.round(status.progress_pct)}%
          </span>
        </div>
      </div>

      <div className="p-3 bg-[#E7E2D6] dark:bg-[#1A201E] space-y-3">
        {/* Retro Inset Progress Bar */}
        <div className="retro-inset h-4 bg-[#DFD9CD] dark:bg-[#0A0D0C] p-0.5 overflow-hidden">
          <div
            className="h-full bg-[#28557E] transition-all duration-300 flex items-center justify-end pr-1 text-[9px] font-mono text-white font-bold"
            style={{ width: `${Math.max(5, status.progress_pct)}%` }}
          >
            {Math.round(status.progress_pct)}%
          </div>
        </div>

        {/* Stage Chips */}
        <div className="grid grid-cols-2 sm:grid-cols-4 md:grid-cols-7 gap-1">
          {STAGE_LABELS.slice(0, 7).map((label, i) => {
            const finished = i < stageIdx || isDone;
            const current = i === stageIdx && isRunning;
            const failed = isFailed && i === stageIdx;
            return (
              <div
                key={label}
                className={`retro-inset px-1.5 py-1 text-center font-mono text-[9px] truncate ${
                  finished
                    ? 'bg-[#D3E8D7] dark:bg-[#193A24] text-[#134E26] dark:text-[#88D49E] font-bold border border-[#7BB887]'
                    : current
                    ? 'bg-[#28557E] text-white font-bold border border-[#173857]'
                    : failed
                    ? 'bg-[#EED2D2] dark:bg-[#3D1A1A] text-[#7A1D1D] dark:text-[#E89898] font-bold'
                    : 'bg-[#DFD9CD] dark:bg-[#141817] text-[#69726E] dark:text-[#7A8581]'
                }`}
                title={label}
              >
                {i + 1}. {label.replace('...', '')}
              </div>
            );
          })}
        </div>

        {/* Error banner */}
        {isFailed && status.error && (
          <div className="retro-outset p-2 text-xs font-mono text-rose-800 dark:text-rose-200 bg-[#EED2D2] dark:bg-[#3D1A1A] border border-rose-600 animate-fade-in space-y-1">
            <div className="font-bold flex items-center gap-1.5 text-xs">
              <span>⚠️</span> PIPELINE EXECUTION FAILURE:
            </div>
            <div className="text-[11px] leading-relaxed break-words">{status.error}</div>
          </div>
        )}

        {/* Terminal Console */}
        <div className="retro-outset p-0.5">
          <div className="bg-[#2D4F4A] text-white px-2 py-0.5 text-[10px] font-mono font-bold flex items-center justify-between">
            <div className="flex items-center gap-1.5">
              <span>CONSOLE LOG // pipeline-runtime.log</span>
              <span className="text-[#88BDB6]">({status.log_lines.length} lines)</span>
            </div>
            <div className="flex items-center gap-1">
              {status.log_lines.length > 0 && (
                <button
                  type="button"
                  onClick={copyLogs}
                  className="retro-button px-1.5 py-0 text-[9px] font-mono text-[#222] dark:text-[#E7E2D6]"
                  title="Copy log to clipboard"
                >
                  COPY
                </button>
              )}
              <span className="window-ctrl-btn text-[8px]">_</span>
              <span className="window-ctrl-btn text-[8px]">X</span>
            </div>
          </div>

          <div
            ref={logRef}
            className="retro-inset-dark p-3 font-mono text-[11px] leading-relaxed text-[#4ADE80] max-h-72 overflow-y-auto space-y-0.5 select-text"
          >
            {status.log_lines.map((line, i) => (
              <div key={i} className={getLogLineClass(line)}>
                {line}
              </div>
            ))}
            {status.log_lines.length === 0 && (
              <div className="text-[#889B95] italic flex items-center gap-2">
                <span className="h-2 w-2 animate-ping rounded-full bg-emerald-400" />
                <span>Awaiting telemetry stream from ingestion backend...</span>
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}
