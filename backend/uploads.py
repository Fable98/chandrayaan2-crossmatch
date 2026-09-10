"""
uploads.py — Step 12: upload validation + FileResponse allowlist + run TTL.

All user-supplied files (source, reference, DEM, ingest zips) go through:
  * extension allowlist (checked BEFORE any disk write),
  * filename sanitization (no directories, no traversal, safe charset),
  * streamed writes with a hard byte cap (oversize rejected mid-stream with
    413; partial files removed so a 100MB probe cannot fill the disk).
"""

from __future__ import annotations

import logging
import os
import re
import time
from pathlib import Path
from typing import Iterable, Optional

from fastapi import HTTPException, UploadFile

logger = logging.getLogger("backend.uploads")

ALLOWED_UPLOAD_EXTENSIONS = {".jpg", ".jpeg", ".png", ".tif", ".tiff"}
ALLOWED_INGEST_EXTENSIONS = {".zip"}
_SAFE_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
CHUNK_SIZE = 1024 * 1024  # 1 MB streaming chunks


def max_upload_bytes() -> int:
    """Per-file cap in bytes (env MAX_UPLOAD_MB, default 20MB)."""
    try:
        from config import settings  # type: ignore[import-not-found]

        return int(getattr(settings, "MAX_UPLOAD_MB", 20)) * 1024 * 1024
    except Exception:
        return 20 * 1024 * 1024


def sanitize_upload_filename(filename: Optional[str], allowed: Iterable[str] = ALLOWED_UPLOAD_EXTENSIONS) -> str:
    """Return a safe basename or raise 400/415. Never returns a path."""
    if not filename or not filename.strip():
        raise HTTPException(status_code=400, detail="Upload is missing a filename.")
    name = os.path.basename(filename.strip().replace("\\", "/"))
    if name in ("", ".", "..") or not _SAFE_NAME.match(name):
        raise HTTPException(status_code=400, detail=f"Unsafe filename rejected: {filename!r}")
    ext = os.path.splitext(name)[1].lower()
    if ext not in set(allowed):
        raise HTTPException(
            status_code=415,
            detail=f"Unsupported file type '{ext}'. Allowed: {sorted(set(allowed))}",
        )
    return name


async def save_upload_capped(upload: UploadFile, dest: Path, max_bytes: Optional[int] = None) -> int:
    """Stream an upload to dest with a hard byte cap. 413 on overflow.

    The cap is enforced DURING the write (not after), so oversized probes
    never land on disk. Partial output is removed on any failure.
    """
    cap = max_bytes if max_bytes is not None else max_upload_bytes()
    size = 0
    try:
        dest.parent.mkdir(parents=True, exist_ok=True)
        with dest.open("wb") as out:
            while True:
                chunk = await upload.read(CHUNK_SIZE)
                if not chunk:
                    break
                size += len(chunk)
                if size > cap:
                    raise HTTPException(
                        status_code=413,
                        detail=f"File '{upload.filename}' exceeds the {cap // (1024 * 1024)}MB limit.",
                    )
                out.write(chunk)
        return size
    except HTTPException:
        try:
            if dest.exists():
                dest.unlink()
        except Exception:
            pass
        raise
    except Exception as exc:
        try:
            if dest.exists():
                dest.unlink()
        except Exception:
            pass
        raise HTTPException(status_code=500, detail=f"Failed to save upload: {exc}") from exc
    finally:
        try:
            await upload.close()
        except Exception:
            pass


def is_path_within_roots(path: str | Path, roots: Iterable[str | Path]) -> Optional[Path]:
    """Resolve symlinks/.. and return the real path iff inside an allowed root."""
    try:
        real = Path(path).resolve()
    except Exception:
        return None
    for root in roots:
        try:
            root_real = Path(root).resolve()
            if root_real in real.parents or real == root_real:
                return real
        except Exception:
            continue
    return None


def check_file_response_allowed(path: str | Path, roots: Iterable[str | Path],
                                allowed_exts: Iterable[str]) -> Path:
    """Allowlist gate for every FileResponse: containment + extension. 404 otherwise."""
    real = is_path_within_roots(path, roots)
    if real is None or not real.is_file():
        raise HTTPException(status_code=404, detail="File not found")
    if real.suffix.lower() not in set(allowed_exts):
        raise HTTPException(status_code=404, detail="File not found")
    return real


def purge_expired_runs(data_dir: str | Path, ttl_hours: Optional[float] = None) -> int:
    """Delete dynamic_runs/* entries older than the TTL. Returns purged count."""
    try:
        if ttl_hours is None:
            try:
                from config import settings  # type: ignore[import-not-found]

                ttl_hours = float(getattr(settings, "DYNAMIC_RUNS_TTL_HOURS", 24))
            except Exception:
                ttl_hours = 24
        cutoff = time.time() - float(ttl_hours) * 3600.0
        runs_dir = Path(data_dir) / "dynamic_runs"
        if not runs_dir.is_dir():
            return 0
        purged = 0
        for entry in runs_dir.iterdir():
            try:
                if entry.stat().st_mtime < cutoff:
                    if entry.is_dir() and not entry.is_symlink():
                        import shutil

                        shutil.rmtree(entry, ignore_errors=True)
                    else:
                        entry.unlink(missing_ok=True)
                    purged += 1
            except Exception as exc:
                logger.warning("Could not purge expired run %s: %s", entry, exc)
        if purged:
            logger.info("Purged %d expired dynamic run(s) (TTL %.1fh).", purged, float(ttl_hours))
        return purged
    except Exception as exc:
        logger.warning("Run purge failed: %s", exc)
        return 0
