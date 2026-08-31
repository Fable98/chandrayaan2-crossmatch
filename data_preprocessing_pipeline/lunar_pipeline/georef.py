from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import numpy as np
import rasterio
from rasterio.crs import CRS
from rasterio.enums import Resampling
from rasterio.transform import from_bounds, array_bounds
from rasterio.warp import calculate_default_transform, reproject

from lunar_pipeline.models import ImageMetadata

MOON_RADIUS_M = 1_737_400.0
RESAMPLE = {
    "nearest": Resampling.nearest,
    "bilinear": Resampling.bilinear,
    "cubic": Resampling.cubic,
}


def moon_crs(proj4: str) -> CRS:
    return CRS.from_string(proj4)


def _lonlat_to_eqc_bounds(footprint: dict, crs: CRS) -> tuple[float, float, float, float]:
    from pyproj import Transformer

    moon_geog = CRS.from_string("+proj=longlat +a=1737400 +b=1737400 +no_defs +type=crs")
    to_m = Transformer.from_crs(moon_geog, crs, always_xy=True)
    xs, ys = to_m.transform(
        [footprint["west_lon"], footprint["east_lon"], footprint["east_lon"], footprint["west_lon"]],
        [footprint["south_lat"], footprint["south_lat"], footprint["north_lat"], footprint["north_lat"]],
    )
    return float(min(xs)), float(min(ys)), float(max(xs)), float(max(ys))


def assign_georef_from_footprint(
    arr: np.ndarray,
    profile: dict,
    meta: ImageMetadata,
    target_crs: CRS,
) -> tuple[np.ndarray, dict]:
    if not meta.footprint:
        return arr, profile
    h, w = arr.shape[-2], arr.shape[-1]
    west, south, east, north = _lonlat_to_eqc_bounds(meta.footprint, target_crs)
    transform = from_bounds(west, south, east, north, w, h)
    profile = dict(profile)
    profile.update(
        {
            "crs": target_crs,
            "transform": transform,
            "width": w,
            "height": h,
            "count": arr.shape[0],
            "dtype": "float32",
        }
    )
    if meta.gsd_m is None:
        meta.gsd_m = abs(transform.a)
    return arr, profile


def warp_to_moon(
    arr: np.ndarray,
    profile: dict,
    target_crs: CRS,
    resampling: str = "bilinear",
) -> tuple[np.ndarray, dict]:
    src_crs = profile.get("crs")
    src_transform = profile.get("transform")
    if src_crs is None or src_transform is None:
        return arr, profile

    src_crs = CRS.from_user_input(src_crs)
    if src_crs == target_crs:
        return arr, profile

    dst_transform, width, height = calculate_default_transform(
        src_crs,
        target_crs,
        profile["width"],
        profile["height"],
        *array_bounds(profile["height"], profile["width"], src_transform),
    )
    dst = np.zeros((arr.shape[0], height, width), dtype=np.float32)
    reproject(
        source=arr,
        destination=dst,
        src_transform=src_transform,
        src_crs=src_crs,
        dst_transform=dst_transform,
        dst_crs=target_crs,
        resampling=RESAMPLE.get(resampling, Resampling.bilinear),
        src_nodata=profile.get("nodata"),
        dst_nodata=0,
    )
    out_profile = dict(profile)
    out_profile.update(
        {
            "crs": target_crs,
            "transform": dst_transform,
            "width": width,
            "height": height,
            "count": dst.shape[0],
            "dtype": "float32",
        }
    )
    return dst, out_profile


def try_isis_import(label_path: Path, work_dir: Path) -> Path | None:
    """Optional ISIS3 path: isisimport → .cub. Returns cube path or None."""
    if shutil.which("isisimport") is None:
        return None
    work_dir.mkdir(parents=True, exist_ok=True)
    cub = work_dir / f"{label_path.stem}.cub"
    cmd = ["isisimport", f"from={label_path}", f"to={cub}"]
    try:
        subprocess.run(cmd, check=True, capture_output=True, text=True)
        return cub if cub.exists() else None
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None


def georeference(
    arr: np.ndarray,
    profile: dict,
    meta: ImageMetadata,
    target_proj4: str,
    resampling: str = "bilinear",
    try_isis: bool = False,
    work_dir: Path | None = None,
) -> tuple[np.ndarray, dict]:
    crs = moon_crs(target_proj4)
    if try_isis and meta.label_path and work_dir is not None:
        cub = try_isis_import(Path(meta.label_path), work_dir / "isis")
        if cub is not None:
            with rasterio.open(cub) as src:
                arr = src.read().astype(np.float32)
                profile = src.profile.copy()
                meta.width, meta.height, meta.bands = src.width, src.height, src.count
                if src.res and src.res[0]:
                    meta.gsd_m = abs(src.res[0])

    has_geo = profile.get("crs") not in (None, "") and profile.get("transform") is not None
    if not has_geo:
        arr, profile = assign_georef_from_footprint(arr, profile, meta, crs)
    arr, profile = warp_to_moon(arr, profile, crs, resampling=resampling)
    if profile.get("transform") is not None and meta.gsd_m is None:
        meta.gsd_m = abs(profile["transform"].a)
    return arr, profile
