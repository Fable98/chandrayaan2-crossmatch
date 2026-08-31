#!/usr/bin/env python3
"""
Batch generate 512x512 triplets (OHRC + TMC + IIRS) for all validated triplets.
"""
from pathlib import Path
import cv2
import json
import numpy as np
import rasterio
from rasterio.crs import CRS
from rasterio.transform import from_bounds
from rasterio.warp import reproject, Resampling
from pyproj import Transformer
from lunar_pipeline.ingest import parse_pds4_label, open_raster
from lunar_pipeline.sensors import iirs_reduce

def process_triplets(triplets_file: Path, out_dir: Path):
    with open(triplets_file, "r", encoding="utf-8") as f:
        triplets = json.load(f)

    print(f"Found {len(triplets)} triplet(s) in {triplets_file.name}")

    moon_geog = CRS.from_string("+proj=longlat +a=1737400 +b=1737400 +no_defs +type=crs")
    moon_eqc = CRS.from_string("+proj=eqc +lat_ts=0 +lon_0=0 +a=1737400 +b=1737400 +units=m +no_defs +type=crs")
    to_eqc = Transformer.from_crs(moon_geog, moon_eqc, always_xy=True)

    out_dir.mkdir(parents=True, exist_ok=True)

    for i, t in enumerate(triplets):
        pid_stem = "".join(c if c.isalnum() or c in "-_" else "_" for c in t.get("ohrc_product_id", "OHRC"))[:15]
        t_dir = out_dir / f"triplet_{i:02d}_{pid_stem}"
        
        ohrc_xml = Path(t["ohrc_label"])
        tmc_xml = Path(t["tmc2_label"])
        iirs_xml = Path(t["iirs_label"])

        if not (ohrc_xml.exists() and tmc_xml.exists() and iirs_xml.exists()):
            print(f"Skipping triplet {i} ({pid_stem}): label files missing.")
            continue

        t_dir.mkdir(parents=True, exist_ok=True)
        print(f"[{i+1}/{len(triplets)}] Processing {t_dir.name}...")

        try:
            _, o_meta = parse_pds4_label(ohrc_xml)
            _, t_meta = parse_pds4_label(tmc_xml)
            _, i_meta = parse_pds4_label(iirs_xml)

            o_w = o_meta.footprint["west_lon"]
            o_s = o_meta.footprint["south_lat"]
            o_e = o_meta.footprint["east_lon"]
            o_n = o_meta.footprint["north_lat"]

            oxs, oys = to_eqc.transform([o_w, o_e, o_e, o_w], [o_s, o_s, o_n, o_n])
            dst_bounds = (min(oxs), min(oys), max(oxs), max(oys))
            dst_transform = from_bounds(*dst_bounds, 512, 512)

            # OHRC
            with rasterio.open(ohrc_xml) as src:
                ohrc_raw = src.read(1, out_shape=(512, 512), resampling=Resampling.bilinear).astype(np.float32)

            # TMC
            t_w, t_s, t_e, t_n = t_meta.footprint["west_lon"], t_meta.footprint["south_lat"], t_meta.footprint["east_lon"], t_meta.footprint["north_lat"]
            txs, tys = to_eqc.transform([t_w, t_e, t_e, t_w], [t_s, t_s, t_n, t_n])
            t_bounds = (min(txs), min(tys), max(txs), max(tys))

            with rasterio.open(tmc_xml) as src:
                t_transform = from_bounds(*t_bounds, src.width, src.height)
                from rasterio.windows import from_bounds as win_from_bounds
                win = win_from_bounds(*dst_bounds, transform=t_transform)
                c_off = max(0, int(win.col_off) - 10)
                r_off = max(0, int(win.row_off) - 10)
                c_w = min(src.width - c_off, int(win.width) + 20)
                r_h = min(src.height - r_off, int(win.height) + 20)
                read_win = rasterio.windows.Window(c_off, r_off, c_w, r_h)
                crop = src.read(1, window=read_win).astype(np.float32)
                win_tf = rasterio.windows.transform(read_win, t_transform)

                tmc_dst = np.zeros((1, 512, 512), dtype=np.float32)
                reproject(
                    source=crop[np.newaxis, ...],
                    destination=tmc_dst,
                    src_transform=win_tf,
                    src_crs=moon_eqc,
                    dst_transform=dst_transform,
                    dst_crs=moon_eqc,
                    resampling=Resampling.bilinear,
                )
                tmc_raw = tmc_dst[0]

            # IIRS
            i_w, i_s, i_e, i_n = i_meta.footprint["west_lon"], i_meta.footprint["south_lat"], i_meta.footprint["east_lon"], i_meta.footprint["north_lat"]
            ixs, iys = to_eqc.transform([i_w, i_e, i_e, i_w], [i_s, i_s, i_n, i_n])
            i_bounds = (min(ixs), min(iys), max(ixs), max(iys))

            i_arr, _, _ = open_raster(iirs_xml)
            i_reduced = iirs_reduce(i_arr, mode="pca", n_components=1)[0]
            i_tf = from_bounds(*i_bounds, i_reduced.shape[1], i_reduced.shape[0])

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

            def to_u8(arr):
                mn, mx = float(np.nanmin(arr)), float(np.nanmax(arr))
                return np.clip((arr - mn) / max(mx - mn, 1e-6) * 255.0, 0, 255).astype(np.uint8)

            ohrc_u8 = to_u8(ohrc_raw)
            tmc_u8 = to_u8(tmc_raw)
            iirs_u8 = to_u8(iirs_raw)

            cv2.imwrite(str(t_dir / "ohrc_512.png"), ohrc_u8)
            cv2.imwrite(str(t_dir / "tmc_512.png"), tmc_u8)
            cv2.imwrite(str(t_dir / "iirs_512.png"), iirs_u8)

            prof = {
                "driver": "GTiff",
                "height": 512,
                "width": 512,
                "count": 1,
                "dtype": "uint8",
                "crs": moon_eqc,
                "transform": dst_transform,
            }
            with rasterio.open(t_dir / "ohrc_512.tif", "w", **prof) as dst:
                dst.write(ohrc_u8, 1)
            with rasterio.open(t_dir / "tmc_512.tif", "w", **prof) as dst:
                dst.write(tmc_u8, 1)
            with rasterio.open(t_dir / "iirs_512.tif", "w", **prof) as dst:
                dst.write(iirs_u8, 1)

            meta = {
                "ohrc_product_id": t["ohrc_product_id"],
                "tmc2_product_id": t["tmc2_product_id"],
                "iirs_product_id": t["iirs_product_id"],
                "bounds": {
                    "west_lon": o_w,
                    "east_lon": o_e,
                    "south_lat": o_s,
                    "north_lat": o_n
                }
            }
            with open(t_dir / "manifest.json", "w", encoding="utf-8") as mf:
                json.dump(meta, mf, indent=2)

            print(f"Done: {t_dir.name}")
        except Exception as err:
            print(f"Error processing triplet {i}: {err}")

if __name__ == "__main__":
    process_triplets(Path("triplets.json"), Path("processed_triplets"))
