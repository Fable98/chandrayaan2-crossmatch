#!/usr/bin/env python3
"""
Fast generation of 6 distinct demo region datasets for backend & frontend.
"""
from pathlib import Path
import cv2
import json
import numpy as np
import rasterio
from rasterio.crs import CRS
from rasterio.transform import from_bounds
from rasterio.warp import reproject, Resampling
from rasterio.windows import from_bounds as win_from_bounds, Window
from pyproj import Transformer
from lunar_pipeline.ingest import parse_pds4_label, open_raster
from lunar_pipeline.sensors import iirs_reduce
from lunar_pipeline.illumination import build_invariants

def to_u8(arr):
    mn, mx = float(np.nanmin(arr)), float(np.nanmax(arr))
    return np.clip((arr - mn) / max(mx - mn, 1e-6) * 255.0, 0, 255).astype(np.uint8)

def main():
    moon_geog = CRS.from_string("+proj=longlat +a=1737400 +b=1737400 +no_defs +type=crs")
    moon_eqc = CRS.from_string("+proj=eqc +lat_ts=0 +lon_0=0 +a=1737400 +b=1737400 +units=m +no_defs +type=crs")
    to_eqc = Transformer.from_crs(moon_geog, moon_eqc, always_xy=True)

    bundles = [
        {
            "ohrc": Path(r"C:\Users\rohit\Downloads\ch2_ohr_ncp_20210405T1606536730_d_img_d18_Bundle\ch2_ohr_ncp_20210405T1606536730_d_img_d18\data\calibrated\20210405\ch2_ohr_ncp_20210405T1606536730_d_img_d18.xml"),
            "tmc": Path(r"C:\Users\rohit\Downloads\ch2_tmc_ncf_20250807T1904346039_d_img_d18\data\calibrated\20250807\ch2_tmc_ncf_20250807T1904346039_d_img_d18.xml"),
            "iirs": Path(r"C:\Users\rohit\Downloads\ch2_ohr_ncp_20210405T1606536730_d_img_d18_Bundle\ch2_iir_nri_20211221T0324126144_d_img_hw1\data\raw\20211221\ch2_iir_nri_20211221T0324126144_d_img_hw1.xml"),
            "slices": [
                (0.05, 0.20, "region_001"),
                (0.25, 0.40, "region_002"),
                (0.45, 0.60, "region_003"),
                (0.65, 0.80, "region_004"),
            ]
        },
        {
            "ohrc": Path(r"C:\Users\rohit\Downloads\ch2_iir_nri_20220221T1109265965_d_img_d18_Bundle (2)\ch2_ohr_ncp_20220914T0835371412_d_img_d32\data\calibrated\20220914\ch2_ohr_ncp_20220914T0835371412_d_img_d32.xml"),
            "tmc": Path(r"C:\Users\rohit\Downloads\ch2_iir_nri_20220221T1109265965_d_img_d18_Bundle (2)\ch2_tmc_ncf_20191125T0749024661_d_img_d18\data\calibrated\20191125\ch2_tmc_ncf_20191125T0749024661_d_img_d18.xml"),
            "iirs": Path(r"C:\Users\rohit\Downloads\ch2_iir_nri_20220221T1109265965_d_img_d18_Bundle (2)\ch2_iir_nri_20220221T1109265965_d_img_d18\data\raw\20220221\ch2_iir_nri_20220221T1109265965_d_img_d18.xml"),
            "slices": [
                (0.15, 0.40, "region_005"),
                (0.50, 0.75, "region_006"),
            ]
        }
    ]

    out_base = Path("processed_triplets")
    out_base.mkdir(exist_ok=True)

    for b in bundles:
        print(f"Loading metadata for bundle {b['ohrc'].name}...")
        _, o_meta = parse_pds4_label(b["ohrc"])
        _, t_meta = parse_pds4_label(b["tmc"])
        _, i_meta = parse_pds4_label(b["iirs"])

        # Load IIRS reduced PCA once
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

        with rasterio.open(b["ohrc"]) as o_src, rasterio.open(b["tmc"]) as t_src:
            o_full_w, o_full_h = o_src.width, o_src.height
            o_full_bounds = (min(to_eqc.transform([full_w, full_e, full_e, full_w], [full_s, full_s, full_n, full_n])[0]),
                             min(to_eqc.transform([full_w, full_e, full_e, full_w], [full_s, full_s, full_n, full_n])[1]),
                             max(to_eqc.transform([full_w, full_e, full_e, full_w], [full_s, full_s, full_n, full_n])[0]),
                             max(to_eqc.transform([full_w, full_e, full_e, full_w], [full_s, full_s, full_n, full_n])[1]))
            o_tf = from_bounds(*o_full_bounds, o_full_w, o_full_h)
            t_tf = from_bounds(*t_bounds, t_src.width, t_src.height)

            for s_start, s_end, reg_name in b["slices"]:
                reg_dir = out_base / reg_name
                reg_dir.mkdir(parents=True, exist_ok=True)
                print(f"Generating {reg_name}...")

                reg_s = full_s + (full_n - full_s) * s_start
                reg_n = full_s + (full_n - full_s) * s_end
                reg_w, reg_e = full_w, full_e

                oxs, oys = to_eqc.transform([reg_w, reg_e, reg_e, reg_w], [reg_s, reg_s, reg_n, reg_n])
                dst_bounds = (min(oxs), min(oys), max(oxs), max(oys))
                dst_transform = from_bounds(*dst_bounds, 512, 512)

                # Crop & Reproject OHRC
                win_o = win_from_bounds(*dst_bounds, transform=o_tf)
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
                    dst_transform=dst_transform,
                    dst_crs=moon_eqc,
                    resampling=Resampling.bilinear,
                )
                ohrc_raw = ohrc_dst[0]

                # Crop & Reproject TMC
                win_t = win_from_bounds(*dst_bounds, transform=t_tf)
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
                    dst_transform=dst_transform,
                    dst_crs=moon_eqc,
                    resampling=Resampling.bilinear,
                )
                tmc_raw = tmc_dst[0]

                # Reproject IIRS
                iirs_dst = np.zeros((1, 512, 512), dtype=np.float32)
                reproject(
                    source=i_reduced[np.newaxis, ...],
                    destination=iirs_dst,
                    src_transform=i_tf,
                    src_crs=moon_eqc,
                    dst_transform=dst_transform,
                    dst_crs=moon_eqc,
                    resampling=Resampling.bilinear,
                )
                iirs_raw = iirs_dst[0]

                # Write PNGs
                ohrc_u8 = to_u8(ohrc_raw)
                tmc_u8 = to_u8(tmc_raw)
                iirs_u8 = to_u8(iirs_raw)

                cv2.imwrite(str(reg_dir / "ohrc_512.png"), ohrc_u8)
                cv2.imwrite(str(reg_dir / "tmc_512.png"), tmc_u8)
                cv2.imwrite(str(reg_dir / "iirs_512.png"), iirs_u8)

                # Compute & Write Invariant Maps
                o_invs = build_invariants((ohrc_raw / max(float(ohrc_raw.max()), 1e-6))[np.newaxis, ...], ["census", "gradient", "lbp"])
                t_invs = build_invariants((tmc_raw / max(float(tmc_raw.max()), 1e-6))[np.newaxis, ...], ["census", "gradient", "lbp"])

                for k, v in o_invs.items():
                    cv2.imwrite(str(reg_dir / f"ohrc_512_{k}.png"), to_u8(v[0] if v.ndim==3 else v))
                for k, v in t_invs.items():
                    cv2.imwrite(str(reg_dir / f"tmc_512_{k}.png"), to_u8(v[0] if v.ndim==3 else v))

                # Manifest JSON
                meta = {
                    "region_id": reg_name,
                    "ohrc_product_id": o_meta.product_id,
                    "tmc2_product_id": t_meta.product_id,
                    "iirs_product_id": i_meta.product_id,
                    "ohrc_gsd_m": o_meta.gsd_m,
                    "tmc2_gsd_m": t_meta.gsd_m,
                    "iirs_gsd_m": i_meta.gsd_m,
                    "ohrc_sun_azimuth_deg": o_meta.sun_azimuth_deg,
                    "tmc2_sun_azimuth_deg": t_meta.sun_azimuth_deg,
                    "sun_azimuth_mismatch_deg": abs(o_meta.sun_azimuth_deg - t_meta.sun_azimuth_deg) if o_meta.sun_azimuth_deg and t_meta.sun_azimuth_deg else 0.0,
                    "bounds": {
                        "west_lon": reg_w,
                        "east_lon": reg_e,
                        "south_lat": reg_s,
                        "north_lat": reg_n
                    }
                }
                with open(reg_dir / "manifest.json", "w", encoding="utf-8") as mf:
                    json.dump(meta, mf, indent=2)

                print(f"  [OK] Saved {reg_name} (bounds: lat [{reg_s:.4f}, {reg_n:.4f}], lon [{reg_w:.4f}, {reg_e:.4f}])")

    print("\nALL 6 REGION DATASETS GENERATED SUCCESSFULLY!")

if __name__ == "__main__":
    main()
