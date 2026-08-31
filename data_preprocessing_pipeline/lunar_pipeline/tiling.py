from __future__ import annotations

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
    records: list[TileRecord] = []

    for row, col, win in windows:
        tile_id = f"{stem}_{meta.sensor}_L{level}_r{row}_c{col}"
        t = rasterio.windows.transform(win, transform)
        sl = (slice(int(win.row_off), int(win.row_off + win.height)), slice(int(win.col_off), int(win.col_off + win.width)))
        patch = arr[:, sl[0], sl[1]]
        files: dict[str, str] = {}

        intensity_path = out_dir / "tiles" / f"{tile_id}.tif"
        _write_tif(intensity_path, patch, profile, t)
        files["intensity"] = str(intensity_path)

        if shadow is not None:
            sm = shadow[sl[0], sl[1]]
            sp = out_dir / "tiles" / f"{tile_id}_shadow.tif"
            _write_tif(sp, sm.astype(np.float32), {**profile, "count": 1}, t)
            files["shadow_mask"] = str(sp)

        if invariants:
            for name, inv in invariants.items():
                inv_patch = inv[:, sl[0], sl[1]] if inv.ndim == 3 else inv[sl[0], sl[1]]
                ip = out_dir / "tiles" / f"{tile_id}_{name}.tif"
                _write_tif(ip, inv_patch, profile, t)
                files[f"invariant_{name}"] = str(ip)

        west, south, east, north = array_bounds(int(win.height), int(win.width), t)
        rec = TileRecord(
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
        records.append(rec)
    return records
