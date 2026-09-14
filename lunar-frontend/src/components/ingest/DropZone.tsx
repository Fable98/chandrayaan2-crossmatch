import React, { useCallback, useRef, useState } from 'react';

interface DropZoneProps {
  onFilesSelected: (files: File[]) => void;
  disabled?: boolean;
}

export default function DropZone({ onFilesSelected, disabled = false }: DropZoneProps) {
  const [isDragOver, setIsDragOver] = useState(false);
  const fileInputRef = useRef<HTMLInputElement>(null);
  const folderInputRef = useRef<HTMLInputElement>(null);

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
      const reader = (entry as FileSystemDirectoryEntry).createReader();
      return new Promise((resolve) => {
        reader.readEntries(
          async (entries) => {
            const nested = await Promise.all(entries.map(extractFilesFromEntry));
            resolve(nested.flat());
          },
          () => resolve([])
        );
      });
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

    // Filter to .zip files
    const zipFiles = allFiles.filter(
      (f) => f.name.toLowerCase().endsWith('.zip')
    );

    if (zipFiles.length > 0) {
      onFilesSelected(zipFiles);
    }
  }, [disabled, onFilesSelected]);

  const handleFileInput = useCallback((e: React.ChangeEvent<HTMLInputElement>) => {
    const fileList = e.target.files;
    if (!fileList) return;
    const files: File[] = [];
    for (let i = 0; i < fileList.length; i++) {
      if (fileList[i].name.toLowerCase().endsWith('.zip')) {
        files.push(fileList[i]);
      }
    }
    if (files.length > 0) {
      onFilesSelected(files);
    }
    // Reset input so re-selecting the same files works
    e.target.value = '';
  }, [onFilesSelected]);

  return (
    <div
      className={`relative rounded-2xl border-2 border-dashed p-10 text-center transition-all duration-200 cursor-pointer overflow-hidden ${
        disabled
          ? 'opacity-50 cursor-not-allowed pointer-events-none border-slate-200 dark:border-[#1b2029] bg-slate-50/50 dark:bg-white/5'
          : isDragOver
          ? 'border-[#4F46E5] bg-indigo-50/40 dark:bg-indigo-950/40 shadow-inner scale-[1.01]'
          : 'border-slate-300 dark:border-[#2a3140] bg-slate-50/60 dark:bg-white/5 hover:bg-indigo-50/20 dark:hover:bg-indigo-950/30 hover:border-[#4F46E5]/50 shadow-xs'
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
        <span className="text-4xl mb-3">
          {isDragOver ? '📂' : '🌙'}
        </span>

        <h3 className="text-base font-bold text-slate-900 dark:text-slate-100 mb-1">
          {isDragOver ? 'Release to upload' : 'Drop PRADAN zip files or a folder here'}
        </h3>

        <p className="text-xs text-slate-500 dark:text-slate-400 max-w-md mb-5">
          Accepts .zip files from ISSDC PRADAN — OHRC, TMC-2, and IIRS products
        </p>

        <div className="flex items-center gap-3">
          <button
            type="button"
            onClick={(e) => {
              e.stopPropagation();
              fileInputRef.current?.click();
            }}
            className="inline-flex items-center gap-2 rounded-xl border border-slate-200 dark:border-[#1b2029] bg-white dark:bg-white/5 px-4 py-2 text-xs font-bold text-slate-700 dark:text-slate-200 shadow-sm hover:bg-slate-50 dark:hover:bg-white/10 hover:border-indigo-300 transition"
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
            className="inline-flex items-center gap-2 rounded-xl border border-slate-200 dark:border-[#1b2029] bg-white dark:bg-white/5 px-4 py-2 text-xs font-bold text-slate-700 dark:text-slate-200 shadow-sm hover:bg-slate-50 dark:hover:bg-white/10 hover:border-indigo-300 transition"
          >
            <span>📂</span>
            <span>Browse Folder</span>
          </button>
        </div>

        <p className="text-[11px] text-slate-400 font-mono mt-5">
          Mixed sensor types are fine — the pipeline auto-discovers OHRC + TMC-2 + IIRS triplets
        </p>
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
