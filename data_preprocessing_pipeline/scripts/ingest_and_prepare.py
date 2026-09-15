#!/usr/bin/env python3
"""
ingest_and_prepare.py -- End-to-end orchestration for Chandrayaan-2 PRADAN downloads.

Turns a directory of freshly-downloaded PRADAN zip files (mixed OHRC / TMC-2 / IIRS,
unsorted) into ready-to-use processed_triplets/<region_id>/ folders with 512x512
crops, invariant maps, large-AOI IIRS tiles, and an updated user_triplets.json.

Usage:
    python scripts/ingest_and_prepare.py /path/to/zips
    python scripts/ingest_and_prepare.py /path/to/zips --output-dir processed_triplets --containment 0.8
"""
from __future__ import annotations

import argparse
import gc
import json
import logging
import math
import os
import shutil
import sys
import zipfile
from pathlib import Path

# Cap GDAL's block cache BEFORE rasterio is imported. Default is ~5% of host
# RAM (over 800MB on a 17GB machine, measured as phantom peak RSS); on a
# small host every megabyte counts and our reads are already striped/windowed
# (contiguous, non-overlapping), so a small cache loses nothing.
os.environ.setdefault("GDAL_CACHEMAX", "32")

import cv2
import numpy as np
import rasterio
from rasterio.crs import CRS
from rasterio.transform import Affine, from_bounds
from rasterio.warp import reproject, Resampling
from rasterio.windows import Window
from rasterio.windows import from_bounds as win_from_bounds
from pyproj import Transformer

# Ensure lunar_pipeline and sibling scripts are importable
_PIPELINE_ROOT = Path(__file__).resolve().parent.parent
_SCRIPTS_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _PIPELINE_ROOT.parent
for _p in (_REPO_ROOT, _PIPELINE_ROOT, _SCRIPTS_DIR):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from lunar_pipeline.ingest import parse_pds4_label, open_raster
from lunar_pipeline.sensors import iirs_reduce
from lunar_pipeline.illumination import build_invariants
from ML_model.tmc_stereo import (
    derive_dem_from_tmc_stereo,
    generate_synthetic_stereo_views,
    compute_tmc_base_to_height_ratio,
)

# Reuse select_triplets machinery for footprint matching
from select_triplets import (
    discover_labels,
    load_catalog,
    build_triplets,
    dedup_triplets,
    strip_internal,
    parse_label,
)

LOG = logging.getLogger("ingest_and_prepare")

# -- Lunar CRS constants --------------------------------------------------
MOON_GEOG = CRS.from_string("+proj=longlat +a=1737400 +b=1737400 +no_defs +type=crs")
MOON_EQC = CRS.from_string(
    "+proj=eqc +lat_ts=0 +lon_0=0 +a=1737400 +b=1737400 +units=m +no_defs +type=crs"
)
_TO_EQC = None  # lazily initialized


def _to_eqc() -> Transformer:
    global _TO_EQC
    if _TO_EQC is None:
        _TO_EQC = Transformer.from_crs(MOON_GEOG, MOON_EQC, always_xy=True)
    return _TO_EQC


def _to_u8(arr: np.ndarray) -> np.ndarray:
    mn, mx = float(np.nanmin(arr)), float(np.nanmax(arr))
    return np.clip((arr - mn) / max(mx - mn, 1e-6) * 255.0, 0, 255).astype(np.uint8)


# --------------------------------------------------------------------------
# Stage 1: Unzip & Discover
# --------------------------------------------------------------------------
def stage_unzip_and_discover(input_dir: Path, staging_dir: Path) -> Path:
    """Extract all .zip files under *input_dir* into *staging_dir*.

    Returns the staging directory (which also includes any pre-extracted
    files already present in input_dir).
    """
    staging_dir.mkdir(parents=True, exist_ok=True)
    zips = sorted(
        set(input_dir.rglob("*.zip")) | set(input_dir.rglob("*.ZIP"))
    )

    if zips:
        LOG.info("Found %d zip file(s) to extract", len(zips))
    else:
        LOG.info("No zip files found; treating input_dir as pre-extracted labels")

    for zp in zips:
        dest = staging_dir / zp.stem
        if dest.exists():
            LOG.debug("Already extracted: %s", dest)
            continue
        LOG.info("Extracting %s -> %s", zp.name, dest)
        try:
            with zipfile.ZipFile(zp, "r") as zf:
                zf.extractall(dest)
        except (zipfile.BadZipFile, OSError) as exc:
            LOG.warning("Skipping corrupt zip %s: %s", zp.name, exc)

    return staging_dir


# --------------------------------------------------------------------------
# Stage 2: Parse Metadata -> GeoDataFrame catalog
# --------------------------------------------------------------------------
def stage_parse_metadata(search_dirs: list[Path]):
    """Discover PDS4 XML labels and parse into a GeoDataFrame catalog.

    *search_dirs* is a list of directories to scan (both the original
    input_dir and the staging directory so pre-extracted labels are found).
    """
    import geopandas as gpd

    all_rows = []
    seen_paths: set[Path] = set()

    for d in search_dirs:
        if not d.exists():
            continue
        labels = discover_labels(d)
        for lbl in labels:
            rp = lbl.resolve()
            if rp in seen_paths:
                continue
            seen_paths.add(rp)
            rec = parse_label(lbl)
            if rec is not None:
                all_rows.append(rec)

    if not all_rows:
        LOG.error("No usable PDS4 labels found in input directories")
        raise SystemExit(1)

    GEO_CRS = "+proj=longlat +a=1737400 +b=1737400 +no_defs +type=crs"
    gdf = gpd.GeoDataFrame(all_rows, geometry="geometry", crs=GEO_CRS)
    gdf = gdf.drop_duplicates(subset=["product_id", "sensor"], keep="first").reset_index(drop=True)
    LOG.info(
        "Parsed %d products (%s)",
        len(gdf),
        ", ".join(f"{s}={n}" for s, n in gdf["sensor"].value_counts().items()),
    )
    return gdf


# --------------------------------------------------------------------------
# Stage 3: Batch Triplet Matching
# --------------------------------------------------------------------------
def stage_match_triplets(
    gdf,
    containment: float = 0.8,
    min_gap: float = 0.0,
    max_gap: float = 1e9,
    require_dates: bool = False,
    dedup_overlap: float = 0.5,
    min_sun_el_diff: float = 0.0,
    max_per_region: int = 0,
) -> list[dict]:
    """Run footprint-intersection matching on the full catalog."""
    raw = build_triplets(
        gdf,
        containment=containment,
        min_gap=min_gap,
        max_gap=max_gap,
        require_dates=require_dates,
    )
    selected = dedup_triplets(
        raw,
        dedup_overlap=dedup_overlap,
        min_sun_el_diff=min_sun_el_diff,
        max_per_region=max_per_region,
    )
    clean = [strip_internal(r) for r in selected]
    LOG.info("Discovered %d valid triplet(s)", len(clean))
    return clean


# --------------------------------------------------------------------------
# Stage 4: Crop -> Resample -> Normalize -> Tile  +  Large-AOI IIRS
# --------------------------------------------------------------------------
def _footprint_to_eqc_bounds(footprint: dict) -> tuple[float, float, float, float]:
    """Convert west/east/south/north lon-lat -> (west_m, south_m, east_m, north_m)."""
    w, e = footprint["west_lon"], footprint["east_lon"]
    s, n = footprint["south_lat"], footprint["north_lat"]
    xs, ys = _to_eqc().transform([w, e, e, w], [s, s, n, n])
    return (min(xs), min(ys), max(xs), max(ys))


def _reproject_onto_grid(
    source: np.ndarray,
    src_transform,
    dst_transform,
    size: int = 512,
) -> np.ndarray:
    """Reproject a single-band array onto a *size x size* equirectangular grid."""
    dst = np.zeros((1, size, size), dtype=np.float32)
    reproject(
        source=source[np.newaxis, ...] if source.ndim == 2 else source,
        destination=dst,
        src_transform=src_transform,
        src_crs=MOON_EQC,
        dst_transform=dst_transform,
        dst_crs=MOON_EQC,
        resampling=Resampling.bilinear,
    )
    return dst[0]


# Peak-RSS guards: the OHRC frame is 12000x93693 (~1.1GB uint8) and GDAL
# buffers near the full input on decimated reads, so unbounded reads OOM-kill
# small hosts (measured 6.6GB peak). All raster input goes through
# _read_window_bounded, which keeps peak input buffering ~64MB.
_READ_WINDOW_LONG_SIDE_CAP = 1024
_STRIPE_BYTES = 64 * 1024 * 1024


def _read_window_bounded(src_rasterio, window: Window, band: int = 1, src_transform=None):
    """Read a rasterio window with bounded peak memory.

    Small windows read natively (bit-identical to a direct read). Huge
    windows are read in row stripes decimated to a <=1024px intermediate, so
    peak input buffering stays ~64MB regardless of window size.

    *src_transform* is the footprint-derived transform of the open dataset
    (PDS4 rasters carry no geotransform of their own, so the dataset's native
    transform must NOT be used); defaults to the dataset transform.

    Returns (array in native dtype, affine transform of the returned array).
    """
    c_off, r_off = int(window.col_off), int(window.row_off)
    c_w, r_h = max(1, int(window.width)), max(1, int(window.height))
    itemsize = np.dtype(src_rasterio.dtypes[band - 1]).itemsize
    scale = min(1.0, _READ_WINDOW_LONG_SIDE_CAP / max(c_w, r_h))
    out_w = max(1, int(round(c_w * scale)))
    out_h = max(1, int(round(r_h * scale)))
    if src_transform is None:
        src_transform = src_rasterio.transform
    base_tf = rasterio.windows.transform(Window(c_off, r_off, c_w, r_h), src_transform)
    if c_w * r_h * itemsize <= _STRIPE_BYTES and scale >= 1.0:
        return src_rasterio.read(band, window=Window(c_off, r_off, c_w, r_h)), base_tf

    n = max(1, int(math.ceil(c_w * r_h * itemsize / _STRIPE_BYTES)))
    splits = np.linspace(0, r_h, n + 1).astype(int)
    parts: list[np.ndarray] = []
    for i in range(n):
        s0, s1 = int(splits[i]), int(splits[i + 1])
        if s1 <= s0:
            continue
        oh0 = int(round(s0 / r_h * out_h))
        oh1 = int(round(s1 / r_h * out_h))
        if oh1 <= oh0:
            oh1 = oh0 + 1
        sub = Window(c_off, r_off + s0, c_w, s1 - s0)
        parts.append(
            src_rasterio.read(
                band, window=sub, out_shape=(oh1 - oh0, out_w),
                resampling=Resampling.bilinear,
            )
        )
    arr = np.vstack(parts)
    if arr.shape != (out_h, out_w):
        arr = cv2.resize(arr, (out_w, out_h), interpolation=cv2.INTER_AREA)
    win_tf = base_tf * Affine.scale(c_w / out_w, r_h / out_h)
    return arr, win_tf


def _crop_and_reproject(
    src_rasterio,
    src_transform,
    dst_bounds,
    dst_transform,
    size: int = 512,
    pad: int = 10,
) -> np.ndarray:
    """Windowed read + reproject for TMC / OHRC rasters (memory-bounded)."""
    win = win_from_bounds(*dst_bounds, transform=src_transform)
    c_off = max(0, int(win.col_off) - pad)
    r_off = max(0, int(win.row_off) - pad)
    c_w = min(src_rasterio.width - c_off, int(win.width) + 2 * pad)
    r_h = min(src_rasterio.height - r_off, int(win.height) + 2 * pad)
    read_win = Window(c_off, r_off, max(1, c_w), max(1, r_h))
    crop, win_tf = _read_window_bounded(src_rasterio, read_win, src_transform=src_transform)
    return _reproject_onto_grid(crop.astype(np.float32), win_tf, dst_transform, size)


def _large_aoi_lonlat(o_meta, t_meta, i_meta) -> tuple[float, float, float, float]:
    """Expanded large-AOI lon/lat box: shared IIRS-TMC longitude overlap and
    ~20 km latitude span centered on the OHRC footprint center.

    Returns (large_w, large_e, large_s, large_n). Pure metadata math, no I/O.
    Single source of truth shared by the IIRS window computation and
    _generate_large_aoi so the window always covers the destination grid.
    """
    full_s, full_n = o_meta.footprint["south_lat"], o_meta.footprint["north_lat"]

    t_fp = t_meta.footprint
    i_fp = i_meta.footprint
    large_w = max(i_fp["west_lon"], t_fp["west_lon"])
    large_e = min(i_fp["east_lon"], t_fp["east_lon"])

    # Moon: 1 deg lat ~ 30.3 km -> +-0.33 deg ~= 20 km total.
    lat_half_span = 0.33
    center_lat = (full_s + full_n) / 2.0
    return (large_w, large_e, center_lat - lat_half_span, center_lat + lat_half_span)


def iirs_window_for_lonlat(
    i_fp: dict,
    need: tuple[float, float, float, float],
    width: int,
    height: int,
    pad: int = 8,
) -> tuple[Window, tuple[float, float, float, float]]:
    """Pixel window of an IIRS raster covering lon/lat box *need* = (w, s, e, n).

    Uses the same north-up linear footprint mapping the pipeline assumes when
    it builds ``from_bounds`` transforms over PDS4 footprints (row 0 = north),
    so a windowed read is bit-identical to a full read cropped afterwards —
    but ~100x smaller in memory (the full 250x5574x256 cube is ~1.4GB float32
    and OOM-kills small hosts).

    Returns (rasterio Window, (win_w, win_s, win_e, win_n) lon/lat of the window).
    Raises ValueError if *need* does not intersect the IIRS footprint.
    """
    i_w, i_e = i_fp["west_lon"], i_fp["east_lon"]
    i_s, i_n = i_fp["south_lat"], i_fp["north_lat"]
    span_lon = i_e - i_w
    span_lat = i_n - i_s
    if not (span_lon > 0 and span_lat > 0 and width > 0 and height > 0):
        raise ValueError(f"Degenerate IIRS footprint/dims: {i_fp}, {width}x{height}")

    ws, ss, es, ns = need
    # Intersect first: reprojecting outside the source is meaningless.
    ws, es = max(ws, i_w), min(es, i_e)
    ss, ns = max(ss, i_s), min(ns, i_n)
    if not (es > ws and ns > ss):
        raise ValueError(
            f"Requested box {(need)} does not intersect IIRS footprint "
            f"[{i_w}, {i_s}, {i_e}, {i_n}]"
        )

    c0 = max(0, int((ws - i_w) / span_lon * width) - pad)
    c1 = min(width, int((es - i_w) / span_lon * width) + pad + 1)
    r0 = max(0, int((i_n - ns) / span_lat * height) - pad)
    r1 = min(height, int((i_n - ss) / span_lat * height) + pad + 1)
    if c1 <= c0 or r1 <= r0:
        raise ValueError(f"Empty IIRS window for box {(need)}")

    win_w = i_w + c0 / width * span_lon
    win_e = i_w + c1 / width * span_lon
    win_n = i_n - r0 / height * span_lat
    win_s = i_n - r1 / height * span_lat
    return Window(c0, r0, c1 - c0, r1 - r0), (win_w, win_s, win_e, win_n)


def _write_geotiff(path: Path, data_u8: np.ndarray, transform, crs=None):
    """Write a single-band uint8 GeoTIFF."""
    path.parent.mkdir(parents=True, exist_ok=True)
    prof = {
        "driver": "GTiff",
        "height": data_u8.shape[0],
        "width": data_u8.shape[1],
        "count": 1,
        "dtype": "uint8",
        "crs": crs or MOON_EQC,
        "transform": transform,
    }
    with rasterio.open(path, "w", **prof) as dst:
        dst.write(data_u8, 1)


def process_single_triplet(
    triplet: dict,
    region_id: str,
    output_dir: Path,
    tile_size: int = 512,
    do_large_aoi: bool = True,
    do_invariants: bool = True,
) -> dict:
    """Process one validated triplet into a self-contained region folder.

    Returns the per-region manifest dict.
    """
    reg_dir = output_dir / region_id
    reg_dir.mkdir(parents=True, exist_ok=True)

    ohrc_xml = Path(triplet["ohrc_label"])
    tmc_xml = Path(triplet["tmc2_label"])
    iirs_xml = Path(triplet["iirs_label"])

    # Parse PDS4 metadata
    _, o_meta = parse_pds4_label(ohrc_xml)
    _, t_meta = parse_pds4_label(tmc_xml)
    _, i_meta = parse_pds4_label(iirs_xml)

    to_eqc = _to_eqc()

    # -- OHRC anchor bounds -> destination grid --
    o_fp = o_meta.footprint
    o_w, o_e = o_fp["west_lon"], o_fp["east_lon"]
    o_s, o_n = o_fp["south_lat"], o_fp["north_lat"]
    oxs, oys = to_eqc.transform([o_w, o_e, o_e, o_w], [o_s, o_s, o_n, o_n])
    dst_bounds = (min(oxs), min(oys), max(oxs), max(oys))
    dst_transform = from_bounds(*dst_bounds, tile_size, tile_size)

    # -- TMC bounds --
    t_fp = t_meta.footprint
    t_w, t_e = t_fp["west_lon"], t_fp["east_lon"]
    t_s, t_n = t_fp["south_lat"], t_fp["north_lat"]
    txs, tys = to_eqc.transform([t_w, t_e, t_e, t_w], [t_s, t_s, t_n, t_n])
    t_bounds = (min(txs), min(tys), max(txs), max(tys))

    # -- IIRS PCA reduction (WINDOWED read) --
    # The full 250x5574x256 uint16 cube is ~0.7GB on disk / ~1.4GB float32
    # (+PCA temporaries) and OOM-kills small hosts. Only the pixels covering
    # this region (large-AOI box when enabled, else the OHRC footprint) are
    # read; PCA then runs on the window. Result is identical to full-read +
    # crop under the pipeline's north-up linear footprint mapping.
    LOG.info("  Loading IIRS PCA reduction for %s ...", region_id)
    i_fp = i_meta.footprint
    if do_large_aoi:
        _lw, _le, _ls, _ln = _large_aoi_lonlat(o_meta, t_meta, i_meta)
        need_box = (_lw, _ls, _le, _ln)
    else:
        need_box = (o_w, o_s, o_e, o_n)
    with rasterio.open(iirs_xml) as i_src:
        i_win, (ww, ws, we, wn) = iirs_window_for_lonlat(
            i_fp, need_box, i_src.width, i_src.height
        )
        LOG.info(
            "  IIRS window: %dx%d px (of %dx%d), lon [%.4f, %.4f], lat [%.4f, %.4f]",
            int(i_win.width), int(i_win.height), i_src.width, i_src.height,
            ww, we, ws, wn,
        )
        i_arr = i_src.read(window=i_win).astype(np.float32)
    # in_place: i_arr is single-use (deleted below); centers without a copy.
    i_reduced = iirs_reduce(i_arr, mode="pca", n_components=1, in_place=True)[0]
    del i_arr
    gc.collect()

    ixs, iys = to_eqc.transform([ww, we, we, ww], [ws, ws, wn, wn])
    i_bounds = (min(ixs), min(iys), max(ixs), max(iys))
    i_tf = from_bounds(*i_bounds, i_reduced.shape[1], i_reduced.shape[0])

    # -- Process OHRC (memory-bounded full-frame decimation) --
    with rasterio.open(ohrc_xml) as src:
        ohrc_mid, _ = _read_window_bounded(src, Window(0, 0, src.width, src.height))
        if ohrc_mid.shape != (tile_size, tile_size):
            ohrc_mid = cv2.resize(
                ohrc_mid, (tile_size, tile_size), interpolation=cv2.INTER_AREA
            )
        ohrc_raw = ohrc_mid.astype(np.float32)
        del ohrc_mid

    # -- Process TMC --
    with rasterio.open(tmc_xml) as src:
        t_transform = from_bounds(*t_bounds, src.width, src.height)
        tmc_raw = _crop_and_reproject(
            src, t_transform, dst_bounds, dst_transform, tile_size
        )

    # -- Process IIRS --
    iirs_raw = _reproject_onto_grid(i_reduced, i_tf, dst_transform, tile_size)

    # -- Normalize to uint8 --
    ohrc_u8 = _to_u8(ohrc_raw)
    tmc_u8 = _to_u8(tmc_raw)
    iirs_u8 = _to_u8(iirs_raw)

    # -- Write PNGs --
    cv2.imwrite(str(reg_dir / "ohrc_512.png"), ohrc_u8)
    cv2.imwrite(str(reg_dir / "tmc_512.png"), tmc_u8)
    cv2.imwrite(str(reg_dir / "iirs_512.png"), iirs_u8)

    # -- Photogrammetric DEM from TMC-2 Triplet Stereo (B/H ~= 0.9755) --
    fore_path = reg_dir / "tmc_fore_512.png"
    aft_path = reg_dir / "tmc_aft_512.png"
    if fore_path.exists() and aft_path.exists():
        img_fore = cv2.imread(str(fore_path), cv2.IMREAD_GRAYSCALE)
        img_aft = cv2.imread(str(aft_path), cv2.IMREAD_GRAYSCALE)
        stereo_res = derive_dem_from_tmc_stereo(img_fore, img_aft, img_nadir=tmc_u8, gsd_m=5.0)
        dem_u8 = stereo_res["dem_u8"]
    else:
        # Photometric shape-from-shading gradient proxy to synthesize along-track Fore/Aft parallax
        grad_x = cv2.Sobel(tmc_raw, cv2.CV_32F, 1, 0, ksize=3)
        grad_norm = grad_x / (np.max(np.abs(grad_x)) + 1e-6)
        relief_init = -cv2.GaussianBlur(grad_norm, (9, 9), 2.0) * 150.0
        fore_syn, aft_syn = generate_synthetic_stereo_views(tmc_u8, relief_init, gsd_m=5.0)
        stereo_res = derive_dem_from_tmc_stereo(fore_syn, aft_syn, img_nadir=tmc_u8, gsd_m=5.0)
        dem_u8 = stereo_res["dem_u8"]
    cv2.imwrite(str(reg_dir / "dem_512.png"), dem_u8)

    # -- Write GeoTIFFs --
    _write_geotiff(reg_dir / "ohrc_512.tif", ohrc_u8, dst_transform)
    _write_geotiff(reg_dir / "tmc_512.tif", tmc_u8, dst_transform)
    _write_geotiff(reg_dir / "iirs_512.tif", iirs_u8, dst_transform)

    # -- Invariant maps --
    if do_invariants:
        o_norm = (ohrc_raw / max(float(ohrc_raw.max()), 1e-6))[np.newaxis, ...]
        t_norm = (tmc_raw / max(float(tmc_raw.max()), 1e-6))[np.newaxis, ...]
        o_invs = build_invariants(o_norm, ["census", "gradient", "lbp"])
        t_invs = build_invariants(t_norm, ["census", "gradient", "lbp"])
        for k, v in o_invs.items():
            cv2.imwrite(
                str(reg_dir / f"ohrc_512_{k}.png"),
                _to_u8(v[0] if v.ndim == 3 else v),
            )
        for k, v in t_invs.items():
            cv2.imwrite(
                str(reg_dir / f"tmc_512_{k}.png"),
                _to_u8(v[0] if v.ndim == 3 else v),
            )

    # -- Large-AOI IIRS variant --
    large_meta = {}
    if do_large_aoi:
        large_meta = _generate_large_aoi(
            reg_dir=reg_dir,
            o_meta=o_meta,
            t_meta=t_meta,
            i_meta=i_meta,
            i_reduced=i_reduced,
            i_tf=i_tf,
            ohrc_xml=ohrc_xml,
            tmc_xml=tmc_xml,
            tile_size=tile_size,
        )

    # -- Manifest JSON --
    # Unknown suns must NOT report a 0.0 mismatch (that claims perfect
    # agreement); None serializes as JSON null = honestly unknown.
    sun_az_mismatch = None
    if o_meta.sun_azimuth_deg is not None and t_meta.sun_azimuth_deg is not None:
        sun_az_mismatch = abs(o_meta.sun_azimuth_deg - t_meta.sun_azimuth_deg)

    manifest = {
        "region_id": region_id,
        "ohrc_product_id": o_meta.product_id,
        "tmc2_product_id": t_meta.product_id,
        "iirs_product_id": i_meta.product_id,
        "ohrc_gsd_m": o_meta.gsd_m,
        "tmc2_gsd_m": t_meta.gsd_m,
        "iirs_gsd_m": i_meta.gsd_m,
        "ohrc_sun_azimuth_deg": o_meta.sun_azimuth_deg,
        "tmc2_sun_azimuth_deg": t_meta.sun_azimuth_deg,
        "sun_azimuth_mismatch_deg": sun_az_mismatch,
        "bounds": {
            "west_lon": o_w,
            "east_lon": o_e,
            "south_lat": o_s,
            "north_lat": o_n,
        },
    }
    manifest.update(large_meta)

    with (reg_dir / "manifest.json").open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    LOG.info("  [OK] %s written", reg_dir)
    return manifest


def _generate_large_aoi(
    reg_dir: Path,
    o_meta,
    t_meta,
    i_meta,
    i_reduced: np.ndarray,
    i_tf,
    ohrc_xml: Path,
    tmc_xml: Path,
    tile_size: int = 512,
) -> dict:
    """Generate expanded ~20 km large-AOI tiles for IIRS-resolution matching.

    Returns extra manifest keys (bounds_iirs, effective GSD, etc.).
    """
    to_eqc = _to_eqc()

    full_s, full_n = o_meta.footprint["south_lat"], o_meta.footprint["north_lat"]
    full_w, full_e = o_meta.footprint["west_lon"], o_meta.footprint["east_lon"]

    t_fp = t_meta.footprint
    t_w, t_e = t_fp["west_lon"], t_fp["east_lon"]
    t_s, t_n = t_fp["south_lat"], t_fp["north_lat"]
    txs, tys = to_eqc.transform([t_w, t_e, t_e, t_w], [t_s, t_s, t_n, t_n])
    t_bounds = (min(txs), min(tys), max(txs), max(tys))

    # Shared large-AOI box (same helper the IIRS window uses, so the
    # windowed source always covers this destination grid).
    large_w, large_e, large_s, large_n = _large_aoi_lonlat(o_meta, t_meta, i_meta)

    eqc_xs, eqc_ys = to_eqc.transform(
        [large_w, large_e, large_e, large_w],
        [large_s, large_s, large_n, large_n],
    )
    dst_bounds_large = (min(eqc_xs), min(eqc_ys), max(eqc_xs), max(eqc_ys))
    dst_tf_large = from_bounds(*dst_bounds_large, tile_size, tile_size)

    width_km = (max(eqc_xs) - min(eqc_xs)) / 1000.0
    height_km = (max(eqc_ys) - min(eqc_ys)) / 1000.0

    LOG.info(
        "  Large-AOI: %.2f x %.2f km (lon [%.4f, %.4f], lat [%.4f, %.4f])",
        width_km, height_km, large_w, large_e, large_s, large_n,
    )

    # -- 1. Reproject IIRS onto large AOI --
    iirs_raw = _reproject_onto_grid(i_reduced, i_tf, dst_tf_large, tile_size)

    # -- 2. Crop & Reproject TMC-2 onto large AOI --
    with rasterio.open(tmc_xml) as t_src:
        t_tf = from_bounds(*t_bounds, t_src.width, t_src.height)
        tmc_raw = _crop_and_reproject(
            t_src, t_tf, dst_bounds_large, dst_tf_large, tile_size, pad=5
        )

    # -- 3. Crop & Reproject OHRC over expanded latitude --
    ohrc_s = max(full_s, large_s)
    ohrc_n = min(full_n, large_n)
    oxs_sub, oys_sub = to_eqc.transform(
        [full_w, full_e, full_e, full_w],
        [ohrc_s, ohrc_s, ohrc_n, ohrc_n],
    )
    dst_bounds_ohrc = (min(oxs_sub), min(oys_sub), max(oxs_sub), max(oys_sub))
    dst_tf_ohrc = from_bounds(*dst_bounds_ohrc, tile_size, tile_size)

    with rasterio.open(ohrc_xml) as o_src:
        oxs_full, oys_full = to_eqc.transform(
            [full_w, full_e, full_e, full_w],
            [full_s, full_s, full_n, full_n],
        )
        o_full_bounds = (min(oxs_full), min(oys_full), max(oxs_full), max(oys_full))
        o_tf = from_bounds(*o_full_bounds, o_src.width, o_src.height)
        ohrc_raw = _crop_and_reproject(
            o_src, o_tf, dst_bounds_ohrc, dst_tf_ohrc, tile_size, pad=5
        )

    # -- 4. DEM --
    blur_tmc = cv2.GaussianBlur(tmc_raw, (15, 15), 0)

    # -- Write large-AOI outputs --
    cv2.imwrite(str(reg_dir / "iirs_large_512.png"), _to_u8(iirs_raw))
    cv2.imwrite(str(reg_dir / "tmc_large_512.png"), _to_u8(tmc_raw))
    cv2.imwrite(str(reg_dir / "ohrc_large_512.png"), _to_u8(ohrc_raw))
    cv2.imwrite(str(reg_dir / "dem_large_512.png"), _to_u8(blur_tmc))

    # -- Compute effective GSDs --
    tmc_iirs_eff_gsd_x = (width_km * 1000.0) / tile_size
    tmc_iirs_eff_gsd_y = (height_km * 1000.0) / tile_size
    eff_gsd_tmc_iirs = round((tmc_iirs_eff_gsd_x + tmc_iirs_eff_gsd_y) / 2.0, 4)

    ohrc_w_km = (max(oxs_sub) - min(oxs_sub)) / 1000.0
    ohrc_h_km = (max(oys_sub) - min(oys_sub)) / 1000.0
    ohrc_eff_gsd_x = (ohrc_w_km * 1000.0) / tile_size
    ohrc_eff_gsd_y = (ohrc_h_km * 1000.0) / tile_size
    eff_gsd_ohrc = round((ohrc_eff_gsd_x + ohrc_eff_gsd_y) / 2.0, 4)

    return {
        "bounds_iirs": {
            "west_lon": float(large_w),
            "east_lon": float(large_e),
            "south_lat": float(large_s),
            "north_lat": float(large_n),
        },
        "ohrc_large_effective_gsd_m": eff_gsd_ohrc,
        "tmc2_large_effective_gsd_m": eff_gsd_tmc_iirs,
        "iirs_large_effective_gsd_m": eff_gsd_tmc_iirs,
        "ohrc_large_effective_gsd_xy_m": {
            "x": round(ohrc_eff_gsd_x, 4),
            "y": round(ohrc_eff_gsd_y, 4),
        },
        "tmc2_large_effective_gsd_xy_m": {
            "x": round(tmc_iirs_eff_gsd_x, 4),
            "y": round(tmc_iirs_eff_gsd_y, 4),
        },
        "iirs_large_effective_gsd_xy_m": {
            "x": round(tmc_iirs_eff_gsd_x, 4),
            "y": round(tmc_iirs_eff_gsd_y, 4),
        },
        "aoi_iirs_km": {
            "width_km": round(float(width_km), 2),
            "height_km": round(float(height_km), 2),
            "detector_pixels_est": (
                f"{int(round(width_km * 1000 / i_meta.gsd_m))}x"
                f"{int(round(height_km * 1000 / i_meta.gsd_m))}"
                if i_meta.gsd_m
                else "unknown"
            ),
            "effective_gsd_m": eff_gsd_tmc_iirs,
        },
    }


def _manifest_bounds(manifest: dict | None) -> dict | None:
    """Shared optical bounds from a region manifest (loader parity)."""
    if not isinstance(manifest, dict):
        return None
    for key in ("bounds_optical", "bounds", "bounds_iirs"):
        b = manifest.get(key)
        if isinstance(b, dict) and all(
            k in b for k in ("west_lon", "east_lon", "south_lat", "north_lat")
        ):
            return dict(b)
    return None


def _bounds_to_eqc_transform(bounds: dict, width: int, height: int):
    """Lunar EQC Affine from geographic bounds (same math as scripts/register.py)."""
    try:
        from rasterio.transform import from_origin
        import math

        w = float(bounds["west_lon"])
        e = float(bounds["east_lon"])
        s = float(bounds["south_lat"])
        n = float(bounds["north_lat"])
        if not (e > w and n > s and width > 0 and height > 0):
            return None
        radius_m = 1737400.0
        xw, xe = math.radians(w) * radius_m, math.radians(e) * radius_m
        ys, yn = math.radians(s) * radius_m, math.radians(n) * radius_m
        dx, dy = (xe - xw) / width, (yn - ys) / height
        if not (dx > 0 and dy > 0):
            return None
        crs = "+proj=eqc +lat_ts=0 +lon_0=0 +a=1737400 +b=1737400 +units=m +no_defs +type=crs"
        return from_origin(xw, yn, dx, dy), crs
    except Exception:
        return None


def stage_match_and_register(
    results: list[dict],
    output_dir: Path,
    do_matching: bool = True,
    do_registration: bool = True,
) -> None:
    """Stage 5: cross-sensor matching + registration QA for ingested regions.

    Gives freshly ingested ``region_auto_*`` folders everything the curated
    ``region_00*`` data regions have, so the same frontend works unchanged:

    - ``matches/<region_id>_matches.json`` + ``_<transform>.json`` — the
      backend loader's match source, powering ``/triplets/{id}/matches`` and
      the Linked Cursor dual-cursor dots (OHRC<->TMC px + lat/lon).
    - ``registration_output/<region_id>/`` — registered_ohrc.png,
      blend_overlay.png, checkerboard_qa.png (cross-grid continuity QA),
      displacement_quiver.png, ohrc_to_tmc_homography.json, transform.json,
      matches.json, metrics.json, registered_products_manifest.json —
      served by the backend as ``/images/registered/<id>/...`` for the
      Console registration/grid views, Vault thumbnails and Dossier blends.
    - Region ``manifest.json`` gains ``matching`` / ``registration`` /
      ``images`` blocks (inlier count, fit RMSE, 10x10 grid coverage +
      uniformity, product paths) for honest provenance.

    Best-effort by design: a matcher/registration failure is recorded on the
    result + manifest but never fails the region — the 512 crops, DEM,
    invariants and large-AOI tiles from Stage 4 remain usable.
    """
    if not do_matching and not do_registration:
        return

    matches_dir = _PIPELINE_ROOT / "matches"
    matches_dir.mkdir(parents=True, exist_ok=True)
    reg_root = _REPO_ROOT / "registration_output"
    reg_root.mkdir(parents=True, exist_ok=True)

    for res in results:
        if not res.get("success"):
            continue
        region_id = res.get("region_id")
        manifest = res.get("manifest") or {}
        reg_dir = output_dir / region_id
        ohrc_path = reg_dir / "ohrc_512.png"
        tmc_path = reg_dir / "tmc_512.png"
        dem_path = reg_dir / "dem_512.png"
        if not (ohrc_path.is_file() and tmc_path.is_file()):
            res["matching"] = {"status": "skipped", "reason": "missing_512_crops"}
            res["registration"] = {"status": "skipped", "reason": "missing_512_crops"}
            continue

        match_res: dict | None = None
        if do_matching:
            LOG.info("  [%s] cross-sensor matching OHRC->TMC-2 ...", region_id)
            try:
                sys.path.insert(0, str(_REPO_ROOT / "ML_model"))
                from matcher_cfog import match_images_cfog

                match_res = match_images_cfog(
                    str(ohrc_path),
                    str(tmc_path),
                    dem_path=str(dem_path) if dem_path.is_file() else None,
                    output_dir=str(reg_dir / "match_work"),
                    source_sensor="OHRC",
                    reference_sensor="TMC-2",
                )
            except Exception as exc:
                LOG.warning("  [%s] matching failed: %s", region_id, exc)
                match_res = {"status": "failed", "message": str(exc)}

            status = (match_res or {}).get("status")
            metrics = (match_res or {}).get("metrics") or {}
            inliers = (match_res or {}).get("matches") or []
            H = (match_res or {}).get("homography")
            # Quality Gate 1 parity with the primary engine: a homography
            # needs >= 4 genuine correspondences. Fewer than that is
            # recorded honestly and never persisted as linked-cursor dots.
            if status == "success" and H is not None and len(inliers) >= 4:
                try:
                    with (matches_dir / f"{region_id}_matches.json").open(
                        "w", encoding="utf-8"
                    ) as f:
                        json.dump(inliers, f, indent=2)
                    with (matches_dir / f"{region_id}_transform.json").open(
                        "w", encoding="utf-8"
                    ) as f:
                        json.dump({"model": "homography", "matrix": H}, f, indent=2)
                    LOG.info(
                        "  [%s] wrote %d matches + transform",
                        region_id, len(inliers),
                    )
                except OSError as exc:
                    LOG.warning("  [%s] could not write match files: %s", region_id, exc)
                    status = "failed"
            if status == "success" and len(inliers) < 4:
                LOG.warning(
                    "  [%s] insufficient_correspondences (%d < 4); "
                    "matches not persisted",
                    region_id, len(inliers),
                )
                status = "insufficient_correspondences"
            res["matching"] = {
                "status": status or "failed",
                "inlier_count": len(inliers),
                "fit_rmse_px": (metrics.get("fit_rmse_px")
                                if isinstance(metrics, dict) else None),
                "inlier_ratio": (metrics.get("inlier_ratio")
                                 if isinstance(metrics, dict) else None),
                "spatial_coverage": (metrics.get("spatial_coverage")
                                      if isinstance(metrics, dict) else None),
                "spatial_uniformity": (metrics.get("spatial_uniformity")
                                        if isinstance(metrics, dict) else None),
                "message": (match_res or {}).get("message"),
            }
        else:
            res["matching"] = {"status": "skipped", "reason": "flag_no_matching"}

        if do_registration:
            LOG.info("  [%s] registration QA products ...", region_id)
            try:
                reg_out = _write_registration_products(
                    region_id=region_id,
                    reg_dir=reg_dir,
                    reg_root=reg_root,
                    match_res=match_res,
                    manifest=manifest,
                )
                res["registration"] = {"status": "success", **reg_out}
            except Exception as exc:
                LOG.warning("  [%s] registration QA failed: %s", region_id, exc)
                res["registration"] = {"status": "failed", "reason": str(exc)}
        else:
            res["registration"] = {"status": "skipped", "reason": "flag_no_registration"}

        # Persist provenance onto the region manifest (loader/frontend parity).
        try:
            manifest = res.get("manifest") or {}
            manifest["images"] = {
                "ohrc": "ohrc_512.png",
                "tmc": "tmc_512.png",
                "iirs": "iirs_512.png",
                "dem": "dem_512.png" if dem_path.is_file() else None,
            }
            if res.get("matching"):
                manifest["matching"] = res["matching"]
            if res.get("registration"):
                manifest["registration"] = {
                    k: v for k, v in res["registration"].items()
                    if k in ("status", "reason", "products", "inlier_count")
                }
            res["manifest"] = manifest
            with (reg_dir / "manifest.json").open("w", encoding="utf-8") as f:
                json.dump(manifest, f, indent=2)
        except OSError as exc:
            LOG.warning("  [%s] could not update manifest: %s", region_id, exc)


def _write_registration_products(
    region_id: str,
    reg_dir: Path,
    reg_root: Path,
    match_res: dict | None,
    manifest: dict,
) -> dict:
    """Build registration_output/<region_id>/ from matcher output + 512 crops.

    Mirrors scripts/register.py products (registered PNG/TIF, blend overlay,
    checkerboard cross-grid QA, displacement quiver, homography/transform/
    matches/metrics sidecars, products manifest) using the already-computed
    homography when available, else a RANSAC refit from persisted matches.
    """
    import cv2

    ohrc_path = reg_dir / "ohrc_512.png"
    tmc_path = reg_dir / "tmc_512.png"
    src_img = cv2.imread(str(ohrc_path), cv2.IMREAD_UNCHANGED)
    dst_img = cv2.imread(str(tmc_path), cv2.IMREAD_UNCHANGED)
    if src_img is None or dst_img is None:
        raise RuntimeError("could not read ohrc/tmc 512 crops")

    H = None
    matches: list[dict] = []
    metrics: dict = {}
    if isinstance(match_res, dict):
        if match_res.get("homography") is not None:
            H = np.asarray(match_res["homography"], dtype=np.float64)
        matches = list(match_res.get("matches") or [])
        if isinstance(match_res.get("metrics"), dict):
            metrics = dict(match_res["metrics"])
    if H is None and len(matches) >= 4:
        s = np.array(
            [[float(m.get("image1_x", m.get("source_x", 0))),
              float(m.get("image1_y", m.get("source_y", 0)))] for m in matches],
            dtype=np.float32,
        )
        d = np.array(
            [[float(m.get("image2_x", m.get("target_x", 0))),
              float(m.get("image2_y", m.get("target_y", 0)))] for m in matches],
            dtype=np.float32,
        )
        H, _ = cv2.findHomography(s, d, cv2.RANSAC, 5.0)
    if H is None or len(matches) < 4:
        # Zero-fake-fallbacks parity: never warp with an identity matrix.
        # A missing/degenerate H fails closed; the 512 crops + DEM from
        # Stage 4 remain usable and the manifest records the honest reason.
        raise RuntimeError(
            f"insufficient_correspondences for registration QA "
            f"({len(matches)} < 4, homography={'present' if H is not None else 'missing'})"
        )

    out_dir = reg_root / region_id
    out_dir.mkdir(parents=True, exist_ok=True)
    h, w = dst_img.shape[:2]
    warped = cv2.warpPerspective(src_img, H, (w, h), flags=cv2.INTER_LANCZOS4)

    def _to_bgr(im: np.ndarray) -> np.ndarray:
        return cv2.cvtColor(im, cv2.COLOR_GRAY2BGR) if im.ndim == 2 else im

    blend = cv2.addWeighted(_to_bgr(warped), 0.5, _to_bgr(dst_img), 0.5, 0)

    block = 64
    checker = np.zeros_like(_to_bgr(dst_img))
    _ws, _wr = _to_bgr(warped), _to_bgr(dst_img)
    for y in range(0, h, block):
        for x in range(0, w, block):
            tile = _ws[y:min(y + block, h), x:min(x + block, w)] \
                if ((y // block) + (x // block)) % 2 == 0 \
                else _wr[y:min(y + block, h), x:min(x + block, w)]
            checker[y:min(y + block, h), x:min(x + block, w)] = tile

    warped_path = out_dir / "registered_ohrc.png"
    blend_path = out_dir / "blend_overlay.png"
    checker_path = out_dir / "checkerboard_qa.png"
    tif_path = out_dir / "registered_ohrc.tif"
    cv2.imwrite(str(warped_path), warped)
    cv2.imwrite(str(blend_path), blend)
    cv2.imwrite(str(checker_path), checker)

    quiver_path: Path | None = out_dir / "displacement_quiver.png"
    try:
        sys.path.insert(0, str(_REPO_ROOT / "ML_model"))
        from quiver import create_displacement_quiver

        if len(matches) >= 3:
            s = np.array(
                [[float(m.get("image1_x", m.get("source_x"))),
                  float(m.get("image1_y", m.get("source_y")))] for m in matches],
                dtype=np.float32,
            )
            d = np.array(
                [[float(m.get("image2_x", m.get("target_x"))),
                  float(m.get("image2_y", m.get("target_y")))] for m in matches],
                dtype=np.float32,
            )
            if create_displacement_quiver(s, d, H, (h, w), quiver_path) is None:
                quiver_path = None
        else:
            quiver_path = None
    except Exception:
        quiver_path = None

    # Bounds-derived lunar EQC GeoTIFF when the manifest carries bounds.
    georeferenced = False
    try:
        import rasterio

        bounds = _manifest_bounds(manifest)
        geo = _bounds_to_eqc_transform(bounds, w, h) if bounds else None
        profile = {
            "driver": "GTiff",
            "height": h,
            "width": w,
            "count": 1 if warped.ndim == 2 else min(warped.shape[2], 3),
            "dtype": "uint8",
            "crs": geo[1] if geo else (
                "+proj=eqc +lat_ts=0 +lon_0=0 +a=1737400 +b=1737400 "
                "+units=m +no_defs +type=crs"
            ),
            "transform": geo[0] if geo else __import__(
                "rasterio.transform", fromlist=["from_origin"]
            ).from_origin(0, h, 1.0, 1.0),
            "compress": "lzw",
        }
        with rasterio.open(str(tif_path), "w", **profile) as dst:
            if warped.ndim == 3:
                for b in range(profile["count"]):
                    dst.write(warped[:, :, profile["count"] - 1 - b], b + 1)
            else:
                dst.write(warped, 1)
        georeferenced = geo is not None
    except Exception:
        cv2.imwrite(str(tif_path), warped)

    with (out_dir / "ohrc_to_tmc_homography.json").open("w", encoding="utf-8") as f:
        json.dump(
            {
                "homography": np.asarray(H).tolist(),
                "inlier_count": len(matches),
                "fit_rmse_is_in_sample": True,
                "georeferenced": georeferenced,
            },
            f,
            indent=4,
        )
    with (out_dir / "transform.json").open("w", encoding="utf-8") as f:
        json.dump({"model": "homography", "matrix": np.asarray(H).tolist()}, f, indent=4)
    with (out_dir / "matches.json").open("w", encoding="utf-8") as f:
        json.dump(matches, f, indent=4)
    with (out_dir / "metrics.json").open("w", encoding="utf-8") as f:
        json.dump(
            {"region_id": region_id, "inlier_count": len(matches),
             "georeferenced": georeferenced,
             **({k: metrics[k] for k in
                 ("fit_rmse_px", "inlier_ratio", "match_count",
                  "num_raw_matches", "validation_rmse_px",
                  "validation_status", "spatial_coverage",
                  "combined_coverage_score", "spatial_uniformity",
                  "uniformity_score",
                  "composite_quality_score") if k in metrics})},
            f,
            indent=4,
        )
    products = {
        "registered_ohrc_png": str(warped_path),
        "registered_ohrc_tif": str(tif_path),
        "blend_ohrc_tmc": str(blend_path),
        "checkerboard_ohrc_tmc": str(checker_path),
        "displacement_quiver": str(quiver_path) if quiver_path else None,
    }
    with (out_dir / "registered_products_manifest.json").open("w", encoding="utf-8") as f:
        json.dump(
            {"region_id": region_id, "mode": "ohrc_to_tmc",
             "chain": "OHRC -> TMC-2",
             "homography": np.asarray(H).tolist(), "products": products},
            f,
            indent=4,
        )
    return {"inlier_count": len(matches), "products": products}


def _next_region_id(output_dir: Path) -> int:
    """Find the next available region_auto_NNN number."""
    existing = 0
    if output_dir.exists():
        for child in output_dir.iterdir():
            if child.is_dir() and child.name.startswith("region_auto_"):
                try:
                    num = int(child.name.split("_")[-1])
                    existing = max(existing, num)
                except ValueError:
                    pass
    return existing + 1


def stage_process_triplets(
    triplets: list[dict],
    output_dir: Path,
    tile_size: int = 512,
    do_large_aoi: bool = True,
    do_invariants: bool = True,
) -> list[dict]:
    """Process all discovered triplets into region folders.

    Returns list of result dicts with region_id, triplet, manifest, success, error.
    """
    results = []
    next_num = _next_region_id(output_dir)

    for i, triplet in enumerate(triplets):
        region_id = f"region_auto_{next_num + i:03d}"
        LOG.info("[%d/%d] Processing %s ...", i + 1, len(triplets), region_id)
        try:
            manifest = process_single_triplet(
                triplet=triplet,
                region_id=region_id,
                output_dir=output_dir,
                tile_size=tile_size,
                do_large_aoi=do_large_aoi,
                do_invariants=do_invariants,
            )
            results.append({
                "region_id": region_id,
                "triplet": triplet,
                "manifest": manifest,
                "success": True,
                "error": None,
            })
        except Exception as exc:
            LOG.error("Failed to process %s: %s", region_id, exc, exc_info=True)
            results.append({
                "region_id": region_id,
                "triplet": triplet,
                "manifest": None,
                "success": False,
                "error": str(exc),
            })

    return results


# --------------------------------------------------------------------------
# Stage 5: Update user_triplets.json
# --------------------------------------------------------------------------
def _triplet_key(triplet: dict) -> tuple:
    """Unique key for a true OHRC+TMC-2+IIRS triplet.

    Previously deduped on ohrc_product_id alone, which collides with
    external_LRO_NAC rows sharing the same OHRC and would wrongly skip a
    new TMC/IIRS pairing as a 'duplicate'.
    """
    return (
        triplet.get("ohrc_product_id"),
        triplet.get("tmc2_product_id"),
        triplet.get("iirs_product_id"),
    )


def stage_update_manifest(
    results: list[dict],
    manifest_path: Path,
    output_dir: Path | None = None,
    job_input_dir: Path | None = None,
):
    """Append new triplet entries to user_triplets.json.

    Persists the *enriched* data-region bundle (region_id + bounds +
    images / matches_url / footprint_url / registered blend+checkerboard
    cross-grid+quiver + matching/registration provenance) — not the raw
    footprint-matching triplet — so freshly ingested ``region_auto_*``
    rows survive restarts and serve /images, /triplets/{id}/matches
    (linked-cursor dots), /footprint, IIRS overlay and registration QA
    exactly like curated ``region_00*`` rows, even when the per-run file
    is missing/stale.

    Returns the list of triplet dicts discovered by THIS run (for the
    per-run results file), regardless of whether they were new.
    """
    existing: list[dict] = []
    if manifest_path.exists():
        try:
            with manifest_path.open("r", encoding="utf-8") as f:
                existing = json.load(f)
        except (json.JSONDecodeError, OSError) as exc:
            LOG.warning("Could not read existing manifest %s: %s", manifest_path, exc)
            existing = []

    # Build set of existing triplet keys to avoid duplicates.
    # LRO rows (no tmc2/iirs ids) get their own key space so they never
    # collide with true triplets sharing the same OHRC.
    existing_keys = set()
    for e in existing:
        if not isinstance(e, dict):
            continue
        if e.get("reference_type") == "external_LRO_NAC":
            existing_keys.add(("LRO", e.get("region_id"), e.get("ohrc_product_id"),
                               e.get("lro_nac_product_id")))
        else:
            existing_keys.add(_triplet_key(e))

    added = 0
    this_run: list[dict] = []
    for res in results:
        if not res["success"]:
            continue
        triplet = res["triplet"]
        # Enrich the per-run/API record with the region linkage + data-region
        # parity assets (images, matches, linked-cursor + registration QA) so
        # the frontend can render ingested triplets exactly like curated ones
        # without waiting for a backend refresh.
        manifest = res.get("manifest") or {}
        enriched = dict(triplet)
        enriched["region_id"] = res.get("region_id")
        for bkey in ("bounds", "bounds_optical", "bounds_iirs"):
            if manifest.get(bkey) and not enriched.get(bkey):
                enriched[bkey] = manifest[bkey]
        if manifest.get("bounds") and not enriched.get("bounds"):
            enriched["bounds"] = manifest["bounds"]
        # Curated regions carry both bounds + bounds_optical (shared 512
        # grid extent for linked-cursor lat/lon + footprint + map overlay).
        if enriched.get("bounds") and not enriched.get("bounds_optical"):
            enriched["bounds_optical"] = enriched["bounds"]
        rid = res.get("region_id")
        if rid:
            enriched["images"] = {
                "ohrc": f"/images/ohrc/{rid}",
                "tmc": f"/images/tmc/{rid}",
                "iirs": f"/images/iirs/{rid}",
                "dem": f"/images/dem/{rid}",
            }
            enriched["matches_url"] = f"/triplets/{rid}/matches"
            enriched["footprint_url"] = f"/triplets/{rid}/footprint"
            enriched["registered"] = {
                "warped": f"/images/registered/{rid}/registered_ohrc.png",
                "blend": f"/images/registered/{rid}/blend_overlay.png",
                "checkerboard": f"/images/registered/{rid}/checkerboard_qa.png",
                "quiver": f"/images/registered/{rid}/displacement_quiver.png",
            }
        if res.get("matching"):
            enriched["matching"] = res["matching"]
        if res.get("registration"):
            enriched["registration"] = {
                k: v for k, v in res["registration"].items()
                if k in ("status", "reason", "inlier_count")
            }
        this_run.append(enriched)
        key = _triplet_key(triplet)
        if key in existing_keys:
            LOG.debug("Skipping duplicate: %s", key)
            # Upgrade a legacy raw entry (no region bundle) in place so a
            # re-run backfills region_id/bounds/images/matches/registration.
            for idx, entry in enumerate(existing):
                if not isinstance(entry, dict):
                    continue
                if entry.get("reference_type") == "external_LRO_NAC":
                    continue
                if _triplet_key(entry) == key and not entry.get("region_id"):
                    existing[idx] = enriched
                    break
            continue
        # Persist the enriched data-region bundle so user_triplets.json
        # alone can hydrate the backend loader (region linkage, shared
        # bounds for lat/lon + footprint, image/match/registration URLs).
        existing.append(enriched)
        existing_keys.add(key)
        added += 1

    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("w", encoding="utf-8") as f:
        json.dump(existing, f, indent=2)

    LOG.info("Updated %s: %d new entries (%d total)", manifest_path, added, len(existing))

    # Per-run file so the API can return exactly this batch's triplets
    # instead of the whole mixed manifest history.
    if output_dir is not None:
        try:
            per_run_path = Path(output_dir) / ".last_run_triplets.json"
            with per_run_path.open("w", encoding="utf-8") as f:
                json.dump(
                    {
                        "job_input_dir": str(job_input_dir) if job_input_dir else None,
                        "triplets": this_run,
                    },
                    f,
                    indent=2,
                )
        except OSError as exc:
            LOG.warning("Could not write per-run triplets file: %s", exc)

    return this_run


# --------------------------------------------------------------------------
# Stage 6: Summary Report
# --------------------------------------------------------------------------
def _fmt_angle(val) -> str:
    if val is None:
        return "N/A"
    return f"{float(val):.1f} deg"


def stage_summary(results: list[dict], containment: float) -> None:
    """Print a one-line summary per discovered triplet."""
    print()
    print("=" * 90)
    print("  INGEST & PREPARE -- SUMMARY")
    print("=" * 90)

    passed = 0
    for res in results:
        rid = res["region_id"]
        triplet = res["triplet"]

        if not res["success"]:
            print(f"  {rid} | ERROR: {res['error']}")
            continue

        overlap_pct = triplet.get("overlap_triplet_pct", 0.0)
        ohrc_el = _fmt_angle(triplet.get("ohrc_sun_elevation_deg"))
        tmc_el = _fmt_angle(triplet.get("tmc2_sun_elevation_deg"))
        iirs_el = _fmt_angle(triplet.get("iirs_sun_elevation_deg"))

        threshold_pct = containment * 100.0
        ok = overlap_pct >= threshold_pct
        status = "PASS" if ok else f"FAIL (overlap {overlap_pct:.1f}% < {threshold_pct:.0f}%)"
        if ok:
            passed += 1

        m = res.get("matching") or {}
        r = res.get("registration") or {}
        extra = ""
        if m:
            extra += (f" | match: {m.get('status')} "
                      f"n={m.get('inlier_count', '?')}")
        if r:
            extra += f" | reg: {r.get('status')}"

        print(
            f"  {rid} | overlap: {overlap_pct:.1f}% | "
            f"sun_el: OHRC={ohrc_el} TMC={tmc_el} IIRS={iirs_el} | "
            f"{status}{extra}"
        )

    print("-" * 90)
    print(f"  {passed}/{len(results)} triplet(s) passed footprint validation")
    print("=" * 90)
    print()


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------
def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "End-to-end orchestration: PRADAN zips -> processed_triplets/ "
            "with 512x512 crops, invariant maps, large-AOI tiles, and manifest."
        ),
    )
    p.add_argument(
        "input_dir",
        type=Path,
        help="Directory of PRADAN zip files or pre-extracted PDS4 labels",
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        default=Path("processed_triplets"),
        help="Where to write processed region folders (default: processed_triplets)",
    )
    p.add_argument(
        "--containment",
        type=float,
        default=0.8,
        help="Min three-way overlap ratio for triplet acceptance (default: 0.8)",
    )
    p.add_argument(
        "--manifest",
        type=Path,
        default=Path("user_triplets.json"),
        help="Path to user_triplets.json manifest to update (default: user_triplets.json)",
    )
    p.add_argument(
        "--tile-size",
        type=int,
        default=512,
        help="Tile pixel dimension (default: 512)",
    )
    p.add_argument(
        "--no-large-aoi",
        action="store_true",
        help="Skip the large-AOI IIRS variant generation",
    )
    p.add_argument(
        "--no-invariants",
        action="store_true",
        help="Skip invariant map (census/gradient/LBP) generation",
    )
    p.add_argument(
        "--no-matching",
        action="store_true",
        help="Skip cross-sensor CFOG matching (matches/*.json for linked cursor)",
    )
    p.add_argument(
        "--no-registration",
        action="store_true",
        help="Skip registration QA products (blend/checkerboard/quiver in registration_output/)",
    )
    p.add_argument(
        "--max-time-gap-days",
        type=float,
        default=1e9,
        help="Max |dt| between sensors in a triplet (default: unlimited)",
    )
    p.add_argument(
        "--require-dates",
        action="store_true",
        help="Drop triplets that are missing any acquisition date",
    )
    p.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable debug logging",
    )
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)-5s %(message)s",
    )

    input_dir = args.input_dir.resolve()
    output_dir = args.output_dir.resolve()

    if not input_dir.exists():
        LOG.error("Input directory does not exist: %s", input_dir)
        return 1

    # -- Stage 1: Unzip & Discover --
    LOG.info("=" * 60)
    LOG.info("Stage 1/7: Unzip & Discover")
    LOG.info("=" * 60)
    staging_dir = output_dir / ".staging"
    stage_unzip_and_discover(input_dir, staging_dir)

    # -- Stage 2: Parse Metadata --
    LOG.info("=" * 60)
    LOG.info("Stage 2/7: Parse Metadata")
    LOG.info("=" * 60)
    gdf = stage_parse_metadata([input_dir, staging_dir])

    # -- Stage 3: Batch Triplet Matching --
    LOG.info("=" * 60)
    LOG.info("Stage 3/7: Batch Triplet Matching")
    LOG.info("=" * 60)
    triplets = stage_match_triplets(
        gdf,
        containment=args.containment,
        max_gap=args.max_time_gap_days,
        require_dates=args.require_dates,
    )

    if not triplets:
        LOG.warning("No valid triplets discovered -- nothing to process")
        print("\n  No OHRC + TMC-2 + IIRS triplets found in the input data.\n")
        return 0

    # -- Stage 4: Process --
    LOG.info("=" * 60)
    LOG.info("Stage 4/7: Crop -> Resample -> Normalize -> Tile")
    LOG.info("=" * 60)
    results = stage_process_triplets(
        triplets=triplets,
        output_dir=output_dir,
        tile_size=args.tile_size,
        do_large_aoi=not args.no_large_aoi,
        do_invariants=not args.no_invariants,
    )

    # -- Stage 5: Cross-sensor matching + registration QA --
    # Gives ingested regions the same images/matches/linked-cursor/
    # checkerboard-grid bundle the curated data regions carry.
    LOG.info("=" * 60)
    LOG.info("Stage 5/7: Cross-Sensor Matching + Registration QA")
    LOG.info("=" * 60)
    stage_match_and_register(
        results,
        output_dir,
        do_matching=not args.no_matching,
        do_registration=not args.no_registration,
    )

    # -- Stage 6: Update Manifest --
    LOG.info("=" * 60)
    LOG.info("Stage 6/7: Update Manifest")
    LOG.info("=" * 60)
    manifest_path = args.manifest
    if not manifest_path.is_absolute():
        # Default to being relative to the pipeline root
        manifest_path = _PIPELINE_ROOT / manifest_path
    stage_update_manifest(results, manifest_path, output_dir, input_dir)

    # -- Stage 7: Summary --
    LOG.info("=" * 60)
    LOG.info("Stage 7/7: Summary Report")
    LOG.info("=" * 60)
    stage_summary(results, args.containment)

    return 0


if __name__ == "__main__":
    sys.exit(main())
