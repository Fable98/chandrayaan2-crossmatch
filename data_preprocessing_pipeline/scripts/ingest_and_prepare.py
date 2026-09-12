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
import json
import logging
import shutil
import sys
import zipfile
from pathlib import Path

import cv2
import numpy as np
import rasterio
from rasterio.crs import CRS
from rasterio.transform import from_bounds
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


def _crop_and_reproject(
    src_rasterio,
    src_transform,
    dst_bounds,
    dst_transform,
    size: int = 512,
    pad: int = 10,
) -> np.ndarray:
    """Windowed read + reproject for TMC / OHRC rasters."""
    win = win_from_bounds(*dst_bounds, transform=src_transform)
    c_off = max(0, int(win.col_off) - pad)
    r_off = max(0, int(win.row_off) - pad)
    c_w = min(src_rasterio.width - c_off, int(win.width) + 2 * pad)
    r_h = min(src_rasterio.height - r_off, int(win.height) + 2 * pad)
    read_win = Window(c_off, r_off, max(1, c_w), max(1, r_h))
    crop = src_rasterio.read(1, window=read_win).astype(np.float32)
    win_tf = rasterio.windows.transform(read_win, src_transform)
    return _reproject_onto_grid(crop, win_tf, dst_transform, size)


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

    # -- IIRS PCA reduction --
    LOG.info("  Loading IIRS PCA reduction for %s ...", region_id)
    i_arr, _, _ = open_raster(iirs_xml)
    i_reduced = iirs_reduce(i_arr, mode="pca", n_components=1)[0]

    i_fp = i_meta.footprint
    i_w, i_e = i_fp["west_lon"], i_fp["east_lon"]
    i_s, i_n = i_fp["south_lat"], i_fp["north_lat"]
    ixs, iys = to_eqc.transform([i_w, i_e, i_e, i_w], [i_s, i_s, i_n, i_n])
    i_bounds = (min(ixs), min(iys), max(ixs), max(iys))
    i_tf = from_bounds(*i_bounds, i_reduced.shape[1], i_reduced.shape[0])

    # -- Process OHRC --
    with rasterio.open(ohrc_xml) as src:
        ohrc_raw = src.read(
            1, out_shape=(tile_size, tile_size), resampling=Resampling.bilinear
        ).astype(np.float32)

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
    sun_az_mismatch = 0.0
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

    i_fp = i_meta.footprint
    i_w, i_e = i_fp["west_lon"], i_fp["east_lon"]

    # Shared IIRS-TMC longitude overlap
    large_w = max(i_w, t_w)
    large_e = min(i_e, t_e)

    # ~20 km latitude span centered on OHRC footprint center
    # Moon: 1 deg lat ~ 30.3 km
    lat_half_span = 0.33  # ~20 km total
    center_lat = (full_s + full_n) / 2.0
    large_s = center_lat - lat_half_span
    large_n = center_lat + lat_half_span

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
def stage_update_manifest(
    results: list[dict],
    manifest_path: Path,
):
    """Append new triplet entries to user_triplets.json."""
    existing: list[dict] = []
    if manifest_path.exists():
        try:
            with manifest_path.open("r", encoding="utf-8") as f:
                existing = json.load(f)
        except (json.JSONDecodeError, OSError) as exc:
            LOG.warning("Could not read existing manifest %s: %s", manifest_path, exc)
            existing = []

    # Build set of existing OHRC product IDs to avoid duplicates
    existing_ids = {
        e.get("ohrc_product_id") for e in existing if isinstance(e, dict)
    }

    added = 0
    for res in results:
        if not res["success"]:
            continue
        triplet = res["triplet"]
        ohrc_id = triplet.get("ohrc_product_id")
        if ohrc_id in existing_ids:
            LOG.debug("Skipping duplicate: %s", ohrc_id)
            continue
        existing.append(triplet)
        existing_ids.add(ohrc_id)
        added += 1

    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("w", encoding="utf-8") as f:
        json.dump(existing, f, indent=2)

    LOG.info("Updated %s: %d new entries (%d total)", manifest_path, added, len(existing))


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

        print(
            f"  {rid} | overlap: {overlap_pct:.1f}% | "
            f"sun_el: OHRC={ohrc_el} TMC={tmc_el} IIRS={iirs_el} | "
            f"{status}"
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
    LOG.info("Stage 1/6: Unzip & Discover")
    LOG.info("=" * 60)
    staging_dir = output_dir / ".staging"
    stage_unzip_and_discover(input_dir, staging_dir)

    # -- Stage 2: Parse Metadata --
    LOG.info("=" * 60)
    LOG.info("Stage 2/6: Parse Metadata")
    LOG.info("=" * 60)
    gdf = stage_parse_metadata([input_dir, staging_dir])

    # -- Stage 3: Batch Triplet Matching --
    LOG.info("=" * 60)
    LOG.info("Stage 3/6: Batch Triplet Matching")
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
    LOG.info("Stage 4/6: Crop -> Resample -> Normalize -> Tile")
    LOG.info("=" * 60)
    results = stage_process_triplets(
        triplets=triplets,
        output_dir=output_dir,
        tile_size=args.tile_size,
        do_large_aoi=not args.no_large_aoi,
        do_invariants=not args.no_invariants,
    )

    # -- Stage 5: Update Manifest --
    LOG.info("=" * 60)
    LOG.info("Stage 5/6: Update Manifest")
    LOG.info("=" * 60)
    manifest_path = args.manifest
    if not manifest_path.is_absolute():
        # Default to being relative to the pipeline root
        manifest_path = _PIPELINE_ROOT / manifest_path
    stage_update_manifest(results, manifest_path)

    # -- Stage 6: Summary --
    LOG.info("=" * 60)
    LOG.info("Stage 6/6: Summary Report")
    LOG.info("=" * 60)
    stage_summary(results, args.containment)

    return 0


if __name__ == "__main__":
    sys.exit(main())
