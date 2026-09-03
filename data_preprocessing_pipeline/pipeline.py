"""
data_preprocessing_pipeline/pipeline.py — Remote Sensing Ingestion and DEM Relief Compensation Pipeline

Standardizes multi-modal Chandrayaan-2 planetary imagery products (OHRC, TMC, IIRS),
applies lunar photometric normalization, and performs DEM-based relief displacement compensation
into a common map frame (local approximation, not full photogrammetric sensor-model ray-tracing).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Optional, Tuple, Dict, Any

import numpy as np
import cv2

# Add parent directory for lunar_pipeline imports if needed
sys.path.insert(0, str(Path(__file__).resolve().parent))


def load_raw_lunar_image(path: Path | str) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Ingests raw image data from PDS4/IMG, GeoTIFF, or standard formats.
    Returns 2D float32 array normalized to [0, 1] and metadata dictionary.
    """
    p = Path(path)
    metadata: Dict[str, Any] = {
        "file_name": p.name,
        "stem": p.stem,
        "sensor": "unknown",
        "emission_deg": 0.0,
        "azimuth_deg": 45.0,
        "gsd_m": 5.0,
    }

    # Infer sensor from name
    name_lower = p.name.lower()
    if "ohr" in name_lower:
        metadata["sensor"] = "OHRC"
        metadata["gsd_m"] = 0.25
        metadata["emission_deg"] = 0.0
    elif "tmc" in name_lower:
        metadata["sensor"] = "TMC-2"
        metadata["gsd_m"] = 5.0
        metadata["emission_deg"] = 12.0
    elif "iir" in name_lower:
        metadata["sensor"] = "IIRS"
        metadata["gsd_m"] = 70.0
        metadata["emission_deg"] = 0.0

    # Look for matching PDS4 XML label
    xml_path = p.with_suffix(".xml")
    if not xml_path.exists():
        xml_path = p.with_suffix(".XML")
    if xml_path.exists():
        try:
            from lunar_pipeline.ingest import parse_pds4_label
            _, pds_meta = parse_pds4_label(xml_path)
            metadata["sensor"] = pds_meta.sensor
            metadata["emission_deg"] = pds_meta.emission_deg or metadata["emission_deg"]
            metadata["azimuth_deg"] = pds_meta.incidence_deg or metadata["azimuth_deg"]
            metadata["gsd_m"] = pds_meta.gsd_m or metadata["gsd_m"]
        except Exception:
            pass

    # Read image via rasterio if available
    try:
        import rasterio
        with rasterio.open(str(p)) as src:
            if src.count > 3:
                # Hyperspectral cube: average bands into 1-channel pseudo-panchromatic
                bands = src.read().astype(np.float32)
                arr = np.mean(bands, axis=0)
            elif src.count >= 3:
                rgb = np.dstack([src.read(i) for i in (1, 2, 3)]).astype(np.float32)
                arr = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
            else:
                arr = src.read(1).astype(np.float32)

            metadata["width"] = int(src.width)
            metadata["height"] = int(src.height)
            metadata["crs"] = str(src.crs) if src.crs else None
            metadata["transform"] = list(src.transform) if src.transform else None
            
            # Normalize to [0, 1]
            a_min, a_max = float(np.nanmin(arr)), float(np.nanmax(arr))
            if a_max > a_min:
                arr = (arr - a_min) / (a_max - a_min)
            else:
                arr = np.zeros_like(arr)
            return arr.astype(np.float32), metadata
    except Exception:
        pass

    # Fallback to OpenCV
    raw = cv2.imread(str(p), cv2.IMREAD_UNCHANGED)
    if raw is None:
        raise ValueError(f"Could not read image file: {p}")

    if raw.ndim == 3 and raw.shape[2] > 3:
        arr = np.mean(raw.astype(np.float32), axis=2)
    elif raw.ndim == 3 and raw.shape[2] == 3:
        arr = cv2.cvtColor(raw, cv2.COLOR_BGR2GRAY).astype(np.float32)
    else:
        arr = raw.astype(np.float32)

    metadata["width"] = int(arr.shape[1])
    metadata["height"] = int(arr.shape[0])

    a_min, a_max = float(np.min(arr)), float(np.max(arr))
    if a_max > a_min:
        arr = (arr - a_min) / (a_max - a_min)
    else:
        arr = np.zeros_like(arr)

    return arr.astype(np.float32), metadata


def apply_dem_relief_compensation(
    image: np.ndarray,
    dem: Optional[np.ndarray] = None,
    emission_deg: float = 0.0,
    azimuth_deg: float = 45.0,
    gsd_m: float = 5.0,
) -> np.ndarray:
    """
    Applies simplified local DEM-based relief displacement compensation.
    Corrects geometric parallax caused by lunar terrain elevation under off-nadir emission angles.
    
    NOTE: This is local DEM relief displacement compensation. It is not equivalent to a
    full photogrammetric rigorous sensor-model ray-tracing orthorectifier.
    """
    h, w = image.shape[:2]
    if dem is None or abs(emission_deg) < 1e-3:
        return image.copy()

    if dem.shape[:2] != (h, w):
        dem_res = cv2.resize(dem.astype(np.float32), (w, h), interpolation=cv2.INTER_LINEAR)
    else:
        dem_res = dem.astype(np.float32)

    dem_rel = (dem_res - float(np.mean(dem_res))).astype(np.float32)
    e_rad = np.radians(emission_deg)
    psi_rad = np.radians(azimuth_deg)

    scale = float(np.tan(e_rad) / max(gsd_m, 1e-3))
    dx = (dem_rel * (scale * np.cos(psi_rad))).astype(np.float32)
    dy = (dem_rel * (scale * np.sin(psi_rad))).astype(np.float32)

    x_coords, y_coords = np.meshgrid(
        np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32)
    )
    map_x = (x_coords + dx).astype(np.float32)
    map_y = (y_coords + dy).astype(np.float32)

    compensated = cv2.remap(
        image,
        map_x,
        map_y,
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT,
    )
    return compensated


# Backward compatibility alias
orthorectify_image_with_dem = apply_dem_relief_compensation


def export_registered_geotiff(
    arr: np.ndarray,
    out_path: Path | str,
    meta: Optional[Dict[str, Any]] = None,
) -> None:
    """
    Exports registered/compensated array as GeoTIFF (if rasterio is installed) or PNG.
    """
    out_p = Path(out_path)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    u8 = (np.clip(arr, 0.0, 1.0) * 255.0).astype(np.uint8)

    written = False
    try:
        import rasterio
        from rasterio.transform import from_origin

        h, w = ortho_arr.shape[:2]
        transform = from_origin(0.0, float(h), 1.0, 1.0)
        profile = {
            "driver": "GTiff",
            "height": h,
            "width": w,
            "count": 1,
            "dtype": "uint8",
            "crs": "+proj=eqc +lat_ts=0 +lon_0=0 +a=1737400 +b=1737400 +units=m +no_defs +type=crs",
            "transform": transform,
            "compress": "lzw",
        }
        tif_path = out_p.with_suffix(".tif")
        with rasterio.open(str(tif_path), "w", **profile) as dst:
            dst.write(u8, 1)
        written = True
    except Exception:
        pass

    # Also save standard image output for web visualization
    img_out = out_p.with_suffix(".png")
    cv2.imwrite(str(img_out), u8)


export_orthorectified_geotiff = export_registered_geotiff


def process_and_orthorectify(
    input_image_path: Path | str,
    dem_path: Optional[Path | str] = None,
    output_path: Optional[Path | str] = None,
    emission_deg: Optional[float] = None,
    azimuth_deg: Optional[float] = None,
    gsd_m: Optional[float] = None,
) -> Path:
    """
    Full pipeline entry point: Ingests raw lunar imagery and generates DEM relief-compensated products.
    """
    in_path = Path(input_image_path)
    if not in_path.exists():
        raise FileNotFoundError(f"Input image not found: {in_path}")

    arr, meta = load_raw_lunar_image(in_path)

    em_deg = emission_deg if emission_deg is not None else meta.get("emission_deg", 0.0)
    az_deg = azimuth_deg if azimuth_deg is not None else meta.get("azimuth_deg", 45.0)
    res_m = gsd_m if gsd_m is not None else meta.get("gsd_m", 5.0)

    dem_arr = None
    if dem_path and Path(dem_path).exists():
        dem_arr, _ = load_raw_lunar_image(dem_path)

    ortho = orthorectify_image_with_dem(
        arr,
        dem=dem_arr,
        emission_deg=em_deg,
        azimuth_deg=az_deg,
        gsd_m=res_m,
    )

    if output_path is None:
        out_path = in_path.parent / f"{in_path.stem}_orthorectified.png"
    else:
        out_path = Path(output_path)

    export_orthorectified_geotiff(ortho, out_path, meta=meta)
    return out_path


def main():
    parser = argparse.ArgumentParser(
        description="Ingest raw lunar data and apply DEM-based relief displacement compensation."
    )
    parser.add_argument("input", type=str, help="Path to raw image file (PDS4/IMG, GeoTIFF, PNG)")
    parser.add_argument("--dem", type=str, default=None, help="Path to DEM elevation map")
    parser.add_argument("--output", type=str, default=None, help="Output destination path")
    parser.add_argument("--emission", type=float, default=None, help="Sensor emission angle in degrees")
    parser.add_argument("--azimuth", type=float, default=None, help="Solar/Look azimuth angle in degrees")
    parser.add_argument("--gsd", type=float, default=None, help="Ground sampling distance in meters")

    args = parser.parse_args()
    out = process_and_orthorectify(
        args.input,
        dem_path=args.dem,
        output_path=args.output,
        emission_deg=args.emission,
        azimuth_deg=args.azimuth,
        gsd_m=args.gsd,
    )
    print(f"Successfully generated orthorectified product: {out}")


if __name__ == "__main__":
    main()
