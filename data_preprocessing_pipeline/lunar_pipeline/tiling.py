from __future__ import annotations

import concurrent.futures
from pathlib import Path

import numpy as np
import rasterio
from rasterio.transform import Affine, array_bounds
from rasterio.windows import Window

from lunar_pipeline.models import ImageMetadata, TileRecord


def _safe_stem(product_id: str) -> str:
    return "".join(c if c.isalnum() or c in "-_" else "_" for c in product_id)[:80]


def tile_windows(width: int, height: int, tile_size: int, overlap: int) -> list[tuple[int, int, Window]]:
    stride = max(1, tile_size - overlap)
    windows = []
    row_i = 0
    r = 0
    while r < height:
        col_i = 0
        c = 0
        h = min(tile_size, height - r)
        while c < width:
            w = min(tile_size, width - c)
            windows.append((row_i, col_i, Window(c, r, w, h)))
            if c + w >= width:
                break
            c += stride
            col_i += 1
        if r + h >= height:
            break
        r += stride
        row_i += 1
    return windows


def _write_tif(path: Path, data: np.ndarray, profile: dict, transform: Affine) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 1 if data.ndim == 2 else data.shape[0]
    h, w = (data.shape if data.ndim == 2 else data.shape[1:])
    prof = {
        "driver": "GTiff",
        "height": h,
        "width": w,
        "count": count,
        "dtype": "float32",
        "transform": transform,
        "compress": profile.get("compress", "lzw"),
        "crs": profile.get("crs"),
        "nodata": profile.get("nodata"),
        # Gigapixel-safe: allow >4GB outputs, tiled layout for partial reads,
        # all CPUs for compression. write_invariant=False avoids the extra
        # checksum pass.
        "BIGTIFF": profile.get("BIGTIFF", "YES"),
        "TILED": profile.get("TILED", "YES"),
        "NUM_THREADS": profile.get("NUM_THREADS", "ALL_CPUS"),
    }
    arr = data if data.ndim == 3 else data[np.newaxis, ...]
    with rasterio.open(path, "w", **{k: v for k, v in prof.items() if v is not None}) as dst:
        dst.write(arr.astype(np.float32))


def write_tiles(
    arr: np.ndarray,
    profile: dict,
    meta: ImageMetadata,
    out_dir: Path,
    tile_size: int,
    overlap: int,
    invariants: dict[str, np.ndarray] | None = None,
    shadow: np.ndarray | None = None,
    level: int = 0,
) -> list[TileRecord]:
    transform: Affine = profile["transform"]
    crs = profile.get("crs")
    crs_str = str(crs) if crs is not None else ""
    windows = tile_windows(arr.shape[2], arr.shape[1], tile_size, overlap)
    stem = _safe_stem(meta.product_id)

    def _write_one(rcw: tuple[int, int, Window]) -> TileRecord:
        row, col, win = rcw
        tile_id = f"{stem}_{meta.sensor}_L{level}_r{row}_c{col}"
        t = rasterio.windows.transform(win, transform)
        sl = (slice(int(win.row_off), int(win.row_off + win.height)), slice(int(win.col_off), int(win.col_off + win.width)))
        # Copy the windowed patch up front: the worker threads must not race
        # on lazy views if the caller mutates arr during the write batch.
        patch = np.ascontiguousarray(arr[:, sl[0], sl[1]], dtype=np.float32)
        files: dict[str, str] = {}

        jobs: list[tuple[Path, np.ndarray, dict]] = [
            (out_dir / "tiles" / f"{tile_id}.tif", patch, profile)
        ]
        if shadow is not None:
            sm = np.ascontiguousarray(shadow[sl[0], sl[1]], dtype=np.float32)
            jobs.append((out_dir / "tiles" / f"{tile_id}_shadow.tif", sm, {**profile, "count": 1}))
        if invariants:
            for name, inv in invariants.items():
                inv_patch = inv[:, sl[0], sl[1]] if inv.ndim == 3 else inv[sl[0], sl[1]]
                inv_patch = np.ascontiguousarray(inv_patch, dtype=np.float32)
                jobs.append((out_dir / "tiles" / f"{tile_id}_{name}.tif", inv_patch, profile))
        # Batch the band writes for this tile concurrently (GTiff compression
        # releases the GIL; thread win scales with band count).
        with concurrent.futures.ThreadPoolExecutor(max_workers=min(4, len(jobs))) as io_pool:
            futs = [io_pool.submit(_write_tif, p, d, prof, t) for p, d, prof in jobs]
            for f in concurrent.futures.as_completed(futs):
                f.result()
        for p, _d, _prof in jobs:
            key = "intensity" if p.name == f"{tile_id}.tif" else (
                "shadow_mask" if p.name == f"{tile_id}_shadow.tif"
                else f"invariant_{p.stem.replace(tile_id + '_', '')}"
            )
            files[key] = str(p)

        west, south, east, north = array_bounds(int(win.height), int(win.width), t)
        return TileRecord(
            tile_id=tile_id,
            product_id=meta.product_id,
            sensor=meta.sensor,
            row=row,
            col=col,
            level=level,
            gsd_m=meta.gsd_m,
            working_gsd_m=meta.working_gsd_m,
            scale_factor=meta.scale_factor,
            sun_azimuth_deg=meta.sun_azimuth_deg,
            sun_elevation_deg=meta.sun_elevation_deg,
            incidence_deg=meta.incidence_deg,
            emission_deg=meta.emission_deg,
            phase_deg=meta.phase_deg,
            acquisition_utc=meta.acquisition_utc,
            footprint=meta.footprint,
            bbox=[west, south, east, north],
            crs=crs_str,
            files=files,
        )

    # Tiles are independent: write them across workers (order restored after).
    max_workers = min(8, max(1, len(windows)))
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as pool:
        records = list(pool.map(_write_one, windows))
    records.sort(key=lambda r: r.tile_id)
    return records
