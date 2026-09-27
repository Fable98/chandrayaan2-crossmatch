import React, { useCallback, useRef, useState } from 'react';

interface DropZoneProps {
  onFilesSelected: (files: File[]) => void;
  disabled?: boolean;
  /** Called with the count of dropped non-.zip files the pipeline ignored. */
  onIgnored?: (ignored: number) => void;
}

export default function DropZone({ onFilesSelected, disabled = false, onIgnored }: DropZoneProps) {
  const [isDragOver, setIsDragOver] = useState(false);
  const [ignoredNotice, setIgnoredNotice] = useState<string | null>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);
  const folderInputRef = useRef<HTMLInputElement>(null);
  const noticeTimer = useRef<number | null>(null);

  const flagIgnored = useCallback((count: number) => {
    if (count <= 0) return;
    onIgnored?.(count);
    setIgnoredNotice(
      `${count} non-.zip file${count === 1 ? " was" : "s were"} ignored — the pipeline only accepts PRADAN .zip archives.`
    );
    if (noticeTimer.current !== null) window.clearTimeout(noticeTimer.current);
    noticeTimer.current = window.setTimeout(() => setIgnoredNotice(null), 5000);
  }, [onIgnored]);

  const handleDragEnter = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    if (!disabled) setIsDragOver(true);
  }, [disabled]);

  const handleDragLeave = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    if (e.currentTarget === e.target) {
      setIsDragOver(false);
    }
  }, []);

  const handleDragOver = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
  }, []);

  const extractFilesFromEntry = async (entry: FileSystemEntry): Promise<File[]> => {
    if (entry.isFile) {
      return new Promise((resolve) => {
        (entry as FileSystemFileEntry).file(
          (f) => resolve([f]),
          () => resolve([])
        );
      });
    }
    if (entry.isDirectory) {
      // readEntries returns at most ~100 entries per call: loop until empty,
      // then recurse into every entry (subfolders included).
      const reader = (entry as FileSystemDirectoryEntry).createReader();
      const all: FileSystemEntry[] = [];
      for (;;) {
        const chunk: FileSystemEntry[] = await new Promise((resolve) => {
          reader.readEntries((entries) => resolve(entries), () => resolve([]));
        });
        if (chunk.length === 0) break;
        all.push(...chunk);
      }
      const nested = await Promise.all(all.map(extractFilesFromEntry));
      return nested.flat();
    }
    return [];
  };

  const handleDrop = useCallback(async (e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    setIsDragOver(false);
    if (disabled) return;

    const items = e.dataTransfer.items;
    const allFiles: File[] = [];

    if (items) {
      const entries: FileSystemEntry[] = [];
      for (let i = 0; i < items.length; i++) {
        const entry = items[i].webkitGetAsEntry?.();
        if (entry) entries.push(entry);
      }
      for (const entry of entries) {
        const files = await extractFilesFromEntry(entry);
        allFiles.push(...files);
      }
    } else {
      for (let i = 0; i < e.dataTransfer.files.length; i++) {
        allFiles.push(e.dataTransfer.files[i]);
      }
    }

    // Filter to .zip files; surface the ignored count instead of dropping silently.
    const zipFiles = allFiles.filter(
      (f) => f.name.toLowerCase().endsWith('.zip')
    );
    flagIgnored(allFiles.length - zipFiles.length);

    if (zipFiles.length > 0) {
      onFilesSelected(zipFiles);
    }
  }, [disabled, onFilesSelected, flagIgnored]);

  const handleFileInput = useCallback((e: React.ChangeEvent<HTMLInputElement>) => {
    const fileList = e.target.files;
    if (!fileList) return;
    const files: File[] = [];
    let ignored = 0;
    for (let i = 0; i < fileList.length; i++) {
      if (fileList[i].name.toLowerCase().endsWith('.zip')) {
        files.push(fileList[i]);
      } else {
        ignored += 1;
      }
    }
    flagIgnored(ignored);
    if (files.length > 0) {
      onFilesSelected(files);
    }
    // Reset input so re-selecting the same files works
    e.target.value = '';
  }, [onFilesSelected, flagIgnored]);

  return (
    <div
      className={`relative rounded-2xl border-2 border-dashed p-10 text-center transition-all duration-300 cursor-pointer overflow-hidden ${
        disabled
          ? 'opacity-50 cursor-not-allowed pointer-events-none border-slate-200 dark:border-[#1b2029] bg-slate-50/50 dark:bg-white/5'
          : isDragOver
          ? 'border-[#4F46E5] bg-indigo-50/50 dark:bg-indigo-950/50 shadow-[0_0_30px_rgba(79,70,229,0.15)] scale-[1.01]'
          : 'border-slate-300 dark:border-[#2a3140] bg-slate-50/60 dark:bg-white/5 hover:bg-indigo-50/20 dark:hover:bg-indigo-950/30 hover:border-[#4F46E5]/60 hover:shadow-md'
      }`}
      onDragEnter={handleDragEnter}
      onDragLeave={handleDragLeave}
      onDragOver={handleDragOver}
      onDrop={handleDrop}
      onClick={() => !disabled && fileInputRef.current?.click()}
      role="button"
      tabIndex={0}
      onKeyDown={(e) => {
        if ((e.key === 'Enter' || e.key === ' ') && !disabled) {
          e.preventDefault();
          fileInputRef.current?.click();
        }
      }}
      aria-label="Drop zone for zip files"
    >
      <div className="flex flex-col items-center justify-center">
        {/* Animated Beacon / Icon */}
        <div className="relative mb-4 flex items-center justify-center">
          <div className="flex h-14 w-14 items-center justify-center rounded-2xl bg-indigo-50 dark:bg-indigo-950/60 border border-indigo-200 dark:border-indigo-800/50 text-2xl shadow-sm">
            {isDragOver ? '📂' : '🛰️'}
          </div>
          <span className="absolute -top-1 -right-1 flex h-3 w-3">
            <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-indigo-400 opacity-75" />
            <span className="relative inline-flex rounded-full h-3 w-3 bg-[#4F46E5]" />
          </span>
        </div>

        <h3 className="text-base font-extrabold text-slate-900 dark:text-slate-100 mb-1 tracking-tight">
          {isDragOver ? 'Release to Stage Raw Archives' : 'Drop PRADAN ZIP Bundles or Folders'}
        </h3>

        <p className="text-xs text-slate-500 dark:text-slate-400 max-w-md mb-4">
          Accepts raw PDS4 archives from ISSDC PRADAN — OHRC, TMC-2, and IIRS products
        </p>

        {/* Supported Sensor Payload Badges */}
        <div className="flex flex-wrap items-center justify-center gap-2 mb-6">
          <span className="inline-flex items-center gap-1 rounded-full border border-slate-200 dark:border-[#1b2029] bg-white/80 dark:bg-white/5 px-2.5 py-0.5 font-mono text-[10px] font-semibold text-slate-700 dark:text-slate-300">
            <span className="h-1.5 w-1.5 rounded-full bg-indigo-500" />
            OHRC (0.25m)
          </span>
          <span className="inline-flex items-center gap-1 rounded-full border border-slate-200 dark:border-[#1b2029] bg-white/80 dark:bg-white/5 px-2.5 py-0.5 font-mono text-[10px] font-semibold text-slate-700 dark:text-slate-300">
            <span className="h-1.5 w-1.5 rounded-full bg-cyan-500" />
            TMC-2 (5.0m)
          </span>
          <span className="inline-flex items-center gap-1 rounded-full border border-slate-200 dark:border-[#1b2029] bg-white/80 dark:bg-white/5 px-2.5 py-0.5 font-mono text-[10px] font-semibold text-slate-700 dark:text-slate-300">
            <span className="h-1.5 w-1.5 rounded-full bg-emerald-500" />
            IIRS (80m)
          </span>
          <span className="inline-flex items-center gap-1 rounded-full border border-slate-200 dark:border-[#1b2029] bg-white/80 dark:bg-white/5 px-2.5 py-0.5 font-mono text-[10px] font-semibold text-slate-600 dark:text-slate-400">
            PDS4 XML + IMG
          </span>
        </div>

        <div className="flex items-center gap-3">
          <button
            type="button"
            onClick={(e) => {
              e.stopPropagation();
              fileInputRef.current?.click();
            }}
            className="inline-flex items-center gap-2 rounded-xl border border-slate-200 dark:border-[#1b2029] bg-white dark:bg-white/5 px-4 py-2.5 text-xs font-bold text-slate-700 dark:text-slate-200 shadow-sm hover:bg-slate-50 dark:hover:bg-white/10 hover:border-indigo-300 hover:scale-[1.02] active:scale-[0.98] transition-all"
          >
            <span>📁</span>
            <span>Browse Files</span>
          </button>

          <button
            type="button"
            onClick={(e) => {
              e.stopPropagation();
              folderInputRef.current?.click();
            }}
            className="inline-flex items-center gap-2 rounded-xl border border-slate-200 dark:border-[#1b2029] bg-white dark:bg-white/5 px-4 py-2.5 text-xs font-bold text-slate-700 dark:text-slate-200 shadow-sm hover:bg-slate-50 dark:hover:bg-white/10 hover:border-indigo-300 hover:scale-[1.02] active:scale-[0.98] transition-all"
          >
            <span>📂</span>
            <span>Browse Folder</span>
          </button>
        </div>

        <p className="text-[11px] text-slate-400 font-mono mt-5">
          Mixed sensor archives are auto-classified — pipeline correlates overlapping footprints automatically
        </p>
        {ignoredNotice && (
          <p role="status" className="mt-3 rounded-xl border border-amber-200 dark:border-amber-900/50 bg-amber-50 dark:bg-amber-950/30 px-4 py-2 text-[11px] font-semibold text-amber-700 dark:text-amber-300 animate-fade-in">
            {ignoredNotice}
          </p>
        )}
      </div>

      <input
        ref={fileInputRef}
        type="file"
        accept=".zip"
        multiple
        className="hidden"
        onChange={handleFileInput}
      />
      <input
        ref={folderInputRef}
        type="file"
        {...({ webkitdirectory: '' } as { webkitdirectory: string })}
        multiple
        className="hidden"
        onChange={handleFileInput}
      />
    </div>
  );
}
