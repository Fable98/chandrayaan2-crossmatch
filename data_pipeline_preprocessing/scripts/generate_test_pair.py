#!/usr/bin/env python3
"""
Generate a 512x512 test triplet (OHRC + TMC + IIRS) cropped to the common footprint of the validated triplet.
"""
from pathlib import Path
import cv2
import numpy as np
import rasterio
from rasterio.crs import CRS
from rasterio.transform import from_bounds
from rasterio.warp import reproject, Resampling
from pyproj import Transformer
from lunar_pipeline.ingest import parse_pds4_label, open_raster
from lunar_pipeline.sensors import iirs_reduce

def main():
    # 1. Product paths
    ohrc_xml = Path(r"C:\Users\rohit\Downloads\ch2_ohr_ncp_20210405T1606536730_d_img_d18_Bundle\ch2_ohr_ncp_20210405T1606536730_d_img_d18\data\calibrated\20210405\ch2_ohr_ncp_20210405T1606536730_d_img_d18.xml")
    tmc_xml = Path(r"C:\Users\rohit\Downloads\ch2_tmc_ncf_20250807T1904346039_d_img_d18\data\calibrated\20250807\ch2_tmc_ncf_20250807T1904346039_d_img_d18.xml")
    iirs_xml = Path(r"C:\Users\rohit\Downloads\ch2_ohr_ncp_20210405T1606536730_d_img_d18_Bundle\ch2_iir_nri_20211221T0324126144_d_img_hw1\data\raw\20211221\ch2_iir_nri_20211221T0324126144_d_img_hw1.xml")

    # Parse metadata
    _, o_meta = parse_pds4_label(ohrc_xml)
    _, t_meta = parse_pds4_label(tmc_xml)
    _, i_meta = parse_pds4_label(iirs_xml)

    # Lunar projections
    moon_geog = CRS.from_string("+proj=longlat +a=1737400 +b=1737400 +no_defs +type=crs")
    moon_eqc = CRS.from_string("+proj=eqc +lat_ts=0 +lon_0=0 +a=1737400 +b=1737400 +units=m +no_defs +type=crs")
    to_eqc = Transformer.from_crs(moon_geog, moon_eqc, always_xy=True)

    # 2. Common Footprint Bounds (OHRC's exact bounds from manifest)
    o_w = o_meta.footprint["west_lon"]    # 336.484646
    o_s = o_meta.footprint["south_lat"]   # -3.416904
    o_e = o_meta.footprint["east_lon"]    # 336.589455
    o_n = o_meta.footprint["north_lat"]   # -2.576048

    oxs, oys = to_eqc.transform([o_w, o_e, o_e, o_w], [o_s, o_s, o_n, o_n])
    dst_bounds = (min(oxs), min(oys), max(oxs), max(oys))
    dst_transform = from_bounds(*dst_bounds, 512, 512)

    # --- OHRC Processing ---
    with rasterio.open(ohrc_xml) as src:
        ohrc_raw = src.read(1, out_shape=(512, 512), resampling=Resampling.bilinear).astype(np.float32)

    # --- TMC Processing ---
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

    # --- IIRS Processing ---
    i_w, i_s, i_e, i_n = i_meta.footprint["west_lon"], i_meta.footprint["south_lat"], i_meta.footprint["east_lon"], i_meta.footprint["north_lat"]
    ixs, iys = to_eqc.transform([i_w, i_e, i_e, i_w], [i_s, i_s, i_n, i_n])
    i_bounds = (min(ixs), min(iys), max(ixs), max(iys))

    i_arr, _, _ = open_raster(iirs_xml)
    i_reduced = iirs_reduce(i_arr, mode="pca", n_components=1)[0]
    i_transform = from_bounds(*i_bounds, i_reduced.shape[1], i_reduced.shape[0])

    iirs_dst = np.zeros((1, 512, 512), dtype=np.float32)
    reproject(
        source=i_reduced[np.newaxis, ...],
        destination=iirs_dst,
        src_transform=i_transform,
        src_crs=moon_eqc,
        dst_transform=dst_transform,
        dst_crs=moon_eqc,
        resampling=Resampling.bilinear,
    )
    iirs_raw = iirs_dst[0]

    # Normalize to uint8 (0-255)
    def to_u8(arr):
        mn, mx = float(np.nanmin(arr)), float(np.nanmax(arr))
        return np.clip((arr - mn) / max(mx - mn, 1e-6) * 255.0, 0, 255).astype(np.uint8)

    ohrc_u8 = to_u8(ohrc_raw)
    tmc_u8 = to_u8(tmc_raw)
    iirs_u8 = to_u8(iirs_raw)

    # Save PNGs
    cv2.imwrite("ohrc_512.png", ohrc_u8)
    cv2.imwrite("tmc_512.png", tmc_u8)
    cv2.imwrite("iirs_512.png", iirs_u8)

    # Save GeoTIFFs
    profile = {
        "driver": "GTiff",
        "height": 512,
        "width": 512,
        "count": 1,
        "dtype": "uint8",
        "crs": moon_eqc,
        "transform": dst_transform,
    }
    with rasterio.open("ohrc_512.tif", "w", **profile) as dst:
        dst.write(ohrc_u8, 1)

    with rasterio.open("tmc_512.tif", "w", **profile) as dst:
        dst.write(tmc_u8, 1)

    with rasterio.open("iirs_512.tif", "w", **profile) as dst:
        dst.write(iirs_u8, 1)

    print("SUCCESS: OHRC, TMC, and IIRS 512x512 test triplet generated.")
    print(f"Footprint bounds: lon [{o_w:.6f}°, {o_e:.6f}°], lat [{o_s:.6f}°, {o_n:.6f}°]")

if __name__ == "__main__":
    main()
