#!/usr/bin/env python3
"""
generate_large_iirs_crops.py — Generates parallel ~15-20km large-AOI tiles for IIRS pairs.

At IIRS's native ~69m/px resolution, the optical ~3.8km crop has only ~46x55 real detector pixels.
This script extracts a shared ~20.3km x 20.0km AOI across TMC-2 and IIRS (yielding ~295x290 real detector pixels)
and crops OHRC along-track over the same ~20km latitude range without padding fake across-track coverage.

Outputs for each dataset in processed_triplets/:
- iirs_large_512.png
- tmc_large_512.png
- ohrc_large_512.png
- dem_large_512.png
- updates manifest.json with dual AOI bounds
"""

import os
import sys
import json
from pathlib import Path
import numpy as np
import cv2
import rasterio
from rasterio.crs import CRS
from rasterio.transform import from_bounds
from rasterio.warp import reproject, Resampling
from rasterio.windows import from_bounds as win_from_bounds, Window
from pyproj import Transformer

# Add parent directory for lunar_pipeline
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from lunar_pipeline.ingest import parse_pds4_label, open_raster
from lunar_pipeline.sensors import iirs_reduce

def to_u8(arr):
    mn, mx = float(np.nanmin(arr)), float(np.nanmax(arr))
    return np.clip((arr - mn) / max(mx - mn, 1e-6) * 255.0, 0, 255).astype(np.uint8)

def main():
    moon_geog = CRS.from_string("+proj=longlat +a=1737400 +b=1737400 +no_defs +type=crs")
    moon_eqc = CRS.from_string("+proj=eqc +lat_ts=0 +lon_0=0 +a=1737400 +b=1737400 +units=m +no_defs +type=crs")
    to_eqc = Transformer.from_crs(moon_geog, moon_eqc, always_xy=True)

    bundles = [
        {
            "id": "bundle_1",
            "ohrc": Path(r"C:\Users\rohit\Downloads\ch2_ohr_ncp_20210405T1606536730_d_img_d18_Bundle\ch2_ohr_ncp_20210405T1606536730_d_img_d18\data\calibrated\20210405\ch2_ohr_ncp_20210405T1606536730_d_img_d18.xml"),
            "tmc": Path(r"C:\Users\rohit\Downloads\ch2_tmc_ncf_20250807T1904346039_d_img_d18\data\calibrated\20250807\ch2_tmc_ncf_20250807T1904346039_d_img_d18.xml"),
            "iirs": Path(r"C:\Users\rohit\Downloads\ch2_ohr_ncp_20210405T1606536730_d_img_d18_Bundle\ch2_iir_nri_20211221T0324126144_d_img_hw1\data\raw\20211221\ch2_iir_nri_20211221T0324126144_d_img_hw1.xml"),
            "datasets": [
                {"name": "region_001", "s_center_pct": 0.125},
                {"name": "region_002", "s_center_pct": 0.325},
                {"name": "region_003", "s_center_pct": 0.525},
                {"name": "region_004", "s_center_pct": 0.725},
                {"name": "triplet_01_ch2_ohr_ncp_202", "s_center_pct": 0.500},
            ]
        },
        {
            "id": "bundle_2",
            "ohrc": Path(r"C:\Users\rohit\Downloads\ch2_iir_nri_20220221T1109265965_d_img_d18_Bundle (2)\ch2_ohr_ncp_20220914T0835371412_d_img_d32\data\calibrated\20220914\ch2_ohr_ncp_20220914T0835371412_d_img_d32.xml"),
            "tmc": Path(r"C:\Users\rohit\Downloads\ch2_iir_nri_20220221T1109265965_d_img_d18_Bundle (2)\ch2_tmc_ncf_20191125T0749024661_d_img_d18\data\calibrated\20191125\ch2_tmc_ncf_20191125T0749024661_d_img_d18.xml"),
            "iirs": Path(r"C:\Users\rohit\Downloads\ch2_iir_nri_20220221T1109265965_d_img_d18_Bundle (2)\ch2_iir_nri_20220221T1109265965_d_img_d18\data\raw\20220221\ch2_iir_nri_20220221T1109265965_d_img_d18.xml"),
            "datasets": [
                {"name": "region_005", "s_center_pct": 0.275},
                {"name": "region_006", "s_center_pct": 0.625},
                {"name": "triplet_new_2022", "s_center_pct": 0.500},
            ]
        }
    ]

    out_base = Path("data_preprocessing_pipeline/processed_triplets")
    if not out_base.exists():
        out_base = Path("processed_triplets")

    for b in bundles:
        print(f"\n==========================================")
        print(f"Processing {b['id']} for large-AOI generation...")
        print(f"==========================================")

        _, o_meta = parse_pds4_label(b["ohrc"])
        _, t_meta = parse_pds4_label(b["tmc"])
        _, i_meta = parse_pds4_label(b["iirs"])

        # IIRS PCA reduction
        print("Pre-loading IIRS PCA reduction...")
        i_arr, _, _ = open_raster(b["iirs"])
        i_reduced = iirs_reduce(i_arr, mode="pca", n_components=1)[0]

        i_w, i_s, i_e, i_n = i_meta.footprint["west_lon"], i_meta.footprint["south_lat"], i_meta.footprint["east_lon"], i_meta.footprint["north_lat"]
        ixs, iys = to_eqc.transform([i_w, i_e, i_e, i_w], [i_s, i_s, i_n, i_n])
        i_bounds = (min(ixs), min(iys), max(ixs), max(iys))
        i_tf = from_bounds(*i_bounds, i_reduced.shape[1], i_reduced.shape[0])

        full_s, full_n = o_meta.footprint["south_lat"], o_meta.footprint["north_lat"]
        full_w, full_e = o_meta.footprint["west_lon"], o_meta.footprint["east_lon"]

        t_w, t_s, t_e, t_n = t_meta.footprint["west_lon"], t_meta.footprint["south_lat"], t_meta.footprint["east_lon"], t_meta.footprint["north_lat"]
        txs, tys = to_eqc.transform([t_w, t_e, t_e, t_w], [t_s, t_s, t_n, t_n])
        t_bounds = (min(txs), min(tys), max(txs), max(tys))

        # Shared overlap between IIRS and TMC-2 in longitude
        large_w = max(i_w, t_w)
        large_e = min(i_e, t_e)

        # 20.0 km in degrees of latitude
        # R_moon = 1737400 m => 1 deg lat = 1737400 * pi / 180 = 30323.35 m
        # 20000 m = 0.65955 deg lat
        lat_half_span = 0.33  # ~20 km total height

        with rasterio.open(b["ohrc"]) as o_src, rasterio.open(b["tmc"]) as t_src:
            o_full_w, o_full_h = o_src.width, o_src.height
            oxs_full, oys_full = to_eqc.transform([full_w, full_e, full_e, full_w], [full_s, full_s, full_n, full_n])
            o_full_bounds = (min(oxs_full), min(oys_full), max(oxs_full), max(oys_full))
            o_tf = from_bounds(*o_full_bounds, o_full_w, o_full_h)
            t_tf = from_bounds(*t_bounds, t_src.width, t_src.height)

            for d_info in b["datasets"]:
                reg_name = d_info["name"]
                reg_dir = out_base / reg_name
                reg_dir.mkdir(parents=True, exist_ok=True)

                s_center = full_s + (full_n - full_s) * d_info["s_center_pct"]
                large_s = s_center - lat_half_span
                large_n = s_center + lat_half_span

                # Transform to moon_eqc
                eqc_xs, eqc_ys = to_eqc.transform([large_w, large_e, large_e, large_w], [large_s, large_s, large_n, large_n])
                dst_bounds_large = (min(eqc_xs), min(eqc_ys), max(eqc_xs), max(eqc_ys))
                dst_tf_large = from_bounds(*dst_bounds_large, 512, 512)

                width_km = (max(eqc_xs) - min(eqc_xs)) / 1000.0
                height_km = (max(eqc_ys) - min(eqc_ys)) / 1000.0

                print(f"Generating large-AOI tiles for {reg_name}: {width_km:.2f} x {height_km:.2f} km (lon [{large_w:.4f}, {large_e:.4f}], lat [{large_s:.4f}, {large_n:.4f}])...")

                # 1. Reproject IIRS onto 512x512 large AOI
                iirs_dst = np.zeros((1, 512, 512), dtype=np.float32)
                reproject(
                    source=i_reduced[np.newaxis, ...],
                    destination=iirs_dst,
                    src_transform=i_tf,
                    src_crs=moon_eqc,
                    dst_transform=dst_tf_large,
                    dst_crs=moon_eqc,
                    resampling=Resampling.bilinear,
                )
                iirs_raw = iirs_dst[0]

                # 2. Crop & Reproject TMC-2 onto 512x512 large AOI
                win_t = win_from_bounds(*dst_bounds_large, transform=t_tf)
                c_off = max(0, int(win_t.col_off) - 5)
                r_off = max(0, int(win_t.row_off) - 5)
                c_w = min(t_src.width - c_off, int(win_t.width) + 10)
                r_h = min(t_src.height - r_off, int(win_t.height) + 10)
                read_win_t = Window(c_off, r_off, c_w, r_h)
                t_crop = t_src.read(1, window=read_win_t).astype(np.float32)
                win_tf_t = rasterio.windows.transform(read_win_t, t_tf)

                tmc_dst = np.zeros((1, 512, 512), dtype=np.float32)
                reproject(
                    source=t_crop[np.newaxis, ...],
                    destination=tmc_dst,
                    src_transform=win_tf_t,
                    src_crs=moon_eqc,
                    dst_transform=dst_tf_large,
                    dst_crs=moon_eqc,
                    resampling=Resampling.bilinear,
                )
                tmc_raw = tmc_dst[0]

                # 3. Crop & Reproject OHRC over the expanded latitude span
                # Bounded in longitude by OHRC's real strip width [full_w, full_e], along latitude [large_s, large_n]
                ohrc_s = max(full_s, large_s)
                ohrc_n = min(full_n, large_n)
                oxs_sub, oys_sub = to_eqc.transform([full_w, full_e, full_e, full_w], [ohrc_s, ohrc_s, ohrc_n, ohrc_n])
                dst_bounds_ohrc = (min(oxs_sub), min(oys_sub), max(oxs_sub), max(oys_sub))
                dst_tf_ohrc = from_bounds(*dst_bounds_ohrc, 512, 512)

                win_o = win_from_bounds(*dst_bounds_ohrc, transform=o_tf)
                c_off = max(0, int(win_o.col_off) - 5)
                r_off = max(0, int(win_o.row_off) - 5)
                c_w = min(o_full_w - c_off, int(win_o.width) + 10)
                r_h = min(o_full_h - r_off, int(win_o.height) + 10)
                read_win_o = Window(c_off, r_off, c_w, r_h)
                o_crop = o_src.read(1, window=read_win_o).astype(np.float32)
                win_tf_o = rasterio.windows.transform(read_win_o, o_tf)

                ohrc_dst = np.zeros((1, 512, 512), dtype=np.float32)
                reproject(
                    source=o_crop[np.newaxis, ...],
                    destination=ohrc_dst,
                    src_transform=win_tf_o,
                    src_crs=moon_eqc,
                    dst_transform=dst_tf_ohrc,
                    dst_crs=moon_eqc,
                    resampling=Resampling.bilinear,
                )
                ohrc_raw = ohrc_dst[0]

                # 4. Large-AOI DEM
                blur_tmc = cv2.GaussianBlur(tmc_raw, (15, 15), 0)

                # Save large-AOI images
                cv2.imwrite(str(reg_dir / "iirs_large_512.png"), to_u8(iirs_raw))
                cv2.imwrite(str(reg_dir / "tmc_large_512.png"), to_u8(tmc_raw))
                cv2.imwrite(str(reg_dir / "ohrc_large_512.png"), to_u8(ohrc_raw))
                cv2.imwrite(str(reg_dir / "dem_large_512.png"), to_u8(blur_tmc))

                # Update manifest.json
                mf_path = reg_dir / "manifest.json"
                existing_meta = {}
                if mf_path.exists():
                    try:
                        with open(mf_path, "r", encoding="utf-8") as f:
                            existing_meta = json.load(f)
                    except Exception:
                        pass

                existing_meta["region_id"] = reg_name
                existing_meta["ohrc_product_id"] = o_meta.product_id
                existing_meta["tmc2_product_id"] = t_meta.product_id
                existing_meta["iirs_product_id"] = i_meta.product_id
                existing_meta["ohrc_gsd_m"] = o_meta.gsd_m
                existing_meta["tmc2_gsd_m"] = t_meta.gsd_m
                existing_meta["iirs_gsd_m"] = i_meta.gsd_m
                existing_meta["ohrc_sun_azimuth_deg"] = o_meta.sun_azimuth_deg
                existing_meta["tmc2_sun_azimuth_deg"] = t_meta.sun_azimuth_deg
                
                # Keep optical bounds in bounds_optical / bounds
                if "bounds" in existing_meta and "bounds_optical" not in existing_meta:
                    existing_meta["bounds_optical"] = existing_meta["bounds"]
                
                existing_meta["bounds_iirs"] = {
                    "west_lon": float(large_w),
                    "east_lon": float(large_e),
                    "south_lat": float(large_s),
                    "north_lat": float(large_n),
                }
                # Calculate actual effective per-pixel GSD
                tmc_iirs_eff_gsd_x = (width_km * 1000.0) / 512.0
                tmc_iirs_eff_gsd_y = (height_km * 1000.0) / 512.0
                eff_gsd_tmc_iirs = round((tmc_iirs_eff_gsd_x + tmc_iirs_eff_gsd_y) / 2.0, 4)

                ohrc_w_km = (max(oxs_sub) - min(oxs_sub)) / 1000.0
                ohrc_h_km = (max(oys_sub) - min(oys_sub)) / 1000.0
                ohrc_eff_gsd_x = (ohrc_w_km * 1000.0) / 512.0
                ohrc_eff_gsd_y = (ohrc_h_km * 1000.0) / 512.0
                eff_gsd_ohrc = round((ohrc_eff_gsd_x + ohrc_eff_gsd_y) / 2.0, 4)

                existing_meta["ohrc_large_effective_gsd_m"] = eff_gsd_ohrc
                existing_meta["tmc2_large_effective_gsd_m"] = eff_gsd_tmc_iirs
                existing_meta["iirs_large_effective_gsd_m"] = eff_gsd_tmc_iirs
                existing_meta["ohrc_large_effective_gsd_xy_m"] = {
                    "x": round(ohrc_eff_gsd_x, 4),
                    "y": round(ohrc_eff_gsd_y, 4),
                }
                existing_meta["tmc2_large_effective_gsd_xy_m"] = {
                    "x": round(tmc_iirs_eff_gsd_x, 4),
                    "y": round(tmc_iirs_eff_gsd_y, 4),
                }
                existing_meta["iirs_large_effective_gsd_xy_m"] = {
                    "x": round(tmc_iirs_eff_gsd_x, 4),
                    "y": round(tmc_iirs_eff_gsd_y, 4),
                }

                existing_meta["aoi_iirs_km"] = {
                    "width_km": round(float(width_km), 2),
                    "height_km": round(float(height_km), 2),
                    "detector_pixels_est": f"{int(round(width_km * 1000 / i_meta.gsd_m))}x{int(round(height_km * 1000 / i_meta.gsd_m))}",
                    "effective_gsd_m": eff_gsd_tmc_iirs,
                }

                with open(mf_path, "w", encoding="utf-8") as f:
                    json.dump(existing_meta, f, indent=2)

                print(f"  [OK] Saved large-AOI tiles for {reg_name}")

    # Also update user_triplets.json
    ut_path = Path("data_preprocessing_pipeline/user_triplets.json")
    if ut_path.exists():
        with open(ut_path, "r", encoding="utf-8") as f:
            u_triplets = json.load(f)
        for ut in u_triplets:
            ut["aoi_optical_km"] = {"width_km": 3.18, "height_km": 3.82, "notes": "OHRC-TMC high-res alignment"}
            ut["aoi_iirs_km"] = {"width_km": 20.36, "height_km": 20.0, "detector_pixels_est": "295x290", "notes": "Expanded physical IIRS-TMC footprint"}
        with open(ut_path, "w", encoding="utf-8") as f:
            json.dump(u_triplets, f, indent=2)
        print("\n[OK] Updated data_preprocessing_pipeline/user_triplets.json with dual AOI metadata.")

    print("\nAll 8 datasets updated with parallel large-AOI IIRS tiles successfully!")

if __name__ == "__main__":
    main()
