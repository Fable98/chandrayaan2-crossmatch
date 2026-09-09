import React, { useCallback, useRef, useState } from 'react';

interface DropZoneProps {
  onFilesSelected: (files: File[]) => void;
  disabled?: boolean;
}

const styles: Record<string, React.CSSProperties> = {
  wrapper: {
    position: 'relative',
    padding: '48px 32px',
    borderRadius: 'var(--radius-xl)',
    border: '2px dashed var(--border-default)',
    background: 'var(--bg-surface)',
    textAlign: 'center',
    cursor: 'pointer',
    transition: 'all 250ms cubic-bezier(0.16, 1, 0.3, 1)',
    overflow: 'hidden',
  },
  wrapperActive: {
    borderColor: 'var(--accent-primary)',
    background: 'rgba(99, 102, 241, 0.06)',
    boxShadow: '0 0 40px rgba(99, 102, 241, 0.15), inset 0 0 60px rgba(99, 102, 241, 0.05)',
    transform: 'scale(1.01)',
  },
  wrapperDisabled: {
    opacity: 0.5,
    cursor: 'not-allowed',
    pointerEvents: 'none' as const,
  },
  icon: {
    fontSize: '3rem',
    marginBottom: '16px',
    display: 'block',
  },
  title: {
    fontSize: '1.2rem',
    fontWeight: 600,
    color: 'var(--text-primary)',
    marginBottom: '8px',
  },
  subtitle: {
    fontSize: '0.875rem',
    color: 'var(--text-secondary)',
    marginBottom: '20px',
  },
  browseBtn: {
    display: 'inline-flex',
    alignItems: 'center',
    gap: '6px',
    padding: '8px 20px',
    background: 'var(--bg-elevated)',
    border: '1px solid var(--border-default)',
    borderRadius: 'var(--radius-md)',
    color: 'var(--text-secondary)',
    fontSize: '0.8rem',
    fontWeight: 500,
    cursor: 'pointer',
    transition: 'all 150ms ease',
    fontFamily: 'var(--font-sans)',
  },
  hiddenInput: {
    display: 'none',
  },
  hint: {
    fontSize: '0.7rem',
    color: 'var(--text-muted)',
    marginTop: '16px',
  },
  pulseRing: {
    position: 'absolute' as const,
    top: '50%',
    left: '50%',
    width: '120px',
    height: '120px',
    marginTop: '-60px',
    marginLeft: '-60px',
    borderRadius: '50%',
    border: '2px solid var(--accent-primary)',
    opacity: 0,
    animation: 'pulse-ring 1.5s ease-out infinite',
    pointerEvents: 'none' as const,
  },
};

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
    // Only set false if leaving the drop zone itself
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
      // Fallback
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

  const combinedStyle: React.CSSProperties = {
    ...styles.wrapper,
    ...(isDragOver ? styles.wrapperActive : {}),
    ...(disabled ? styles.wrapperDisabled : {}),
  };

  return (
    <div
      style={combinedStyle}
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
      {isDragOver && <div style={styles.pulseRing} />}

      <span style={styles.icon}>
        {isDragOver ? '📂' : '🌙'}
      </span>

      <div style={styles.title}>
        {isDragOver
          ? 'Release to upload'
          : 'Drop PRADAN zip files or a folder here'}
      </div>

      <div style={styles.subtitle}>
        Accepts .zip files from ISSDC PRADAN — OHRC, TMC-2, and IIRS products
      </div>

      <div style={{ display: 'flex', gap: '10px', justifyContent: 'center' }}>
        <button
          style={styles.browseBtn}
          onClick={(e) => {
            e.stopPropagation();
            fileInputRef.current?.click();
          }}
          onMouseEnter={(e) => {
            (e.target as HTMLElement).style.borderColor = 'var(--accent-primary)';
            (e.target as HTMLElement).style.color = 'var(--text-primary)';
          }}
          onMouseLeave={(e) => {
            (e.target as HTMLElement).style.borderColor = 'var(--border-default)';
            (e.target as HTMLElement).style.color = 'var(--text-secondary)';
          }}
          type="button"
        >
          Browse Files
        </button>
        <button
          style={styles.browseBtn}
          onClick={(e) => {
            e.stopPropagation();
            folderInputRef.current?.click();
          }}
          onMouseEnter={(e) => {
            (e.target as HTMLElement).style.borderColor = 'var(--accent-primary)';
            (e.target as HTMLElement).style.color = 'var(--text-primary)';
          }}
          onMouseLeave={(e) => {
            (e.target as HTMLElement).style.borderColor = 'var(--border-default)';
            (e.target as HTMLElement).style.color = 'var(--text-secondary)';
          }}
          type="button"
        >
          Browse Folder
        </button>
      </div>

      <input
        ref={fileInputRef}
        type="file"
        accept=".zip"
        multiple
        style={styles.hiddenInput}
        onChange={handleFileInput}
      />
      <input
        ref={folderInputRef}
        type="file"
        {...({ webkitdirectory: '' } as { webkitdirectory: string })}
        multiple
        style={styles.hiddenInput}
        onChange={handleFileInput}
      />

      <div style={styles.hint}>
        Mixed sensor types are fine — the pipeline auto-discovers OHRC + TMC-2 + IIRS triplets
      </div>
    </div>
  );
}
