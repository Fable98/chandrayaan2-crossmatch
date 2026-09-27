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
      className={`relative retro-inset p-8 text-center transition-all cursor-pointer select-none overflow-hidden ${
        disabled
          ? 'opacity-50 cursor-not-allowed pointer-events-none bg-[#D8D2C4] dark:bg-[#111413]'
          : isDragOver
          ? 'bg-[#C7C0B0] dark:bg-[#202724] border-2 border-dashed border-[#28557E]'
          : 'bg-[#DFD9CD] dark:bg-[#141817] hover:bg-[#D5CFC1] dark:hover:bg-[#1B211F]'
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
        <div className="relative mb-3 flex items-center justify-center">
          <div className="flex h-12 w-12 items-center justify-center retro-outset bg-[#E7E2D6] dark:bg-[#1A201E] text-2xl shadow-sm">
            {isDragOver ? '📂' : '🛰️'}
          </div>
          <span className="absolute -top-1 -right-1 flex h-3 w-3">
            <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-emerald-400 opacity-75" />
            <span className="relative inline-flex rounded-full h-3 w-3 bg-emerald-500" />
          </span>
        </div>

        <h3 className="text-sm font-bold font-mono text-[#1E2321] dark:text-[#E7E2D6] mb-1 tracking-wider uppercase">
          {isDragOver ? 'RELEASE TO STAGE RAW ARCHIVES' : 'DROP PRADAN ZIP BUNDLES OR FOLDERS'}
        </h3>

        <p className="text-[11px] text-[#555C58] dark:text-[#8C9893] max-w-md mb-3 font-sans">
          Accepts raw PDS4 archives from ISSDC PRADAN — OHRC, TMC-2, and IIRS products
        </p>

        {/* Supported Sensor Payload Badges */}
        <div className="flex flex-wrap items-center justify-center gap-1.5 mb-4">
          <span className="retro-inset px-2 py-0.5 font-mono text-[10px] font-bold text-[#1E2321] dark:text-[#E7E2D6] bg-[#ECE7DC] dark:bg-[#161B19] flex items-center gap-1">
            <span className="h-1.5 w-1.5 rounded-full bg-[#28557E]" />
            OHRC (0.25m)
          </span>
          <span className="retro-inset px-2 py-0.5 font-mono text-[10px] font-bold text-[#1E2321] dark:text-[#E7E2D6] bg-[#ECE7DC] dark:bg-[#161B19] flex items-center gap-1">
            <span className="h-1.5 w-1.5 rounded-full bg-cyan-600" />
            TMC-2 (5.0m)
          </span>
          <span className="retro-inset px-2 py-0.5 font-mono text-[10px] font-bold text-[#1E2321] dark:text-[#E7E2D6] bg-[#ECE7DC] dark:bg-[#161B19] flex items-center gap-1">
            <span className="h-1.5 w-1.5 rounded-full bg-emerald-600" />
            IIRS (80m)
          </span>
          <span className="retro-inset px-2 py-0.5 font-mono text-[10px] font-bold text-[#555C58] dark:text-[#8C9893] bg-[#ECE7DC] dark:bg-[#161B19]">
            PDS4 XML + IMG
          </span>
        </div>

        <div className="flex items-center gap-2">
          <button
            type="button"
            onClick={(e) => {
              e.stopPropagation();
              fileInputRef.current?.click();
            }}
            className="retro-button px-3.5 py-1.5 text-xs font-semibold text-[#1E2321] dark:text-[#E7E2D6] flex items-center gap-1.5"
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
            className="retro-button px-3.5 py-1.5 text-xs font-semibold text-[#1E2321] dark:text-[#E7E2D6] flex items-center gap-1.5"
          >
            <span>📂</span>
            <span>Browse Folder</span>
          </button>
        </div>

        <p className="text-[10px] text-[#69726E] dark:text-[#7A8581] font-mono mt-3">
          Mixed sensor archives are auto-classified — pipeline correlates overlapping footprints automatically
        </p>
        {ignoredNotice && (
          <p role="status" className="mt-2 retro-outset px-3 py-1 text-[10px] font-mono font-bold text-amber-900 dark:text-amber-200 bg-[#EFE3C6] dark:bg-[#342813] border border-amber-600">
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
