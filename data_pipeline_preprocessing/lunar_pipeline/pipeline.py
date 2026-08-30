from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
from tqdm import tqdm

from lunar_pipeline.config import PipelineConfig
from lunar_pipeline.georef import georeference
from lunar_pipeline.illumination import apply_photometry, build_invariants, photometric_factor, shadow_mask
from lunar_pipeline.ingest import discover_products, open_raster, write_sidecar_json
from lunar_pipeline.scale import gaussian_pyramid, resample_to_gsd
from lunar_pipeline.sensors import sensor_cleanup
from lunar_pipeline.tiling import write_tiles


def _clahe(arr: np.ndarray) -> np.ndarray:
    import cv2

    out = np.empty_like(arr)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    for i in range(arr.shape[0]):
        band = arr[i]
        finite = np.isfinite(band)
        if not finite.any():
            out[i] = band
            continue
        lo, hi = np.percentile(band[finite], (1, 99))
        scaled = np.clip((band - lo) / max(hi - lo, 1e-6), 0, 1)
        u8 = (scaled * 255).astype(np.uint8)
        eq = clahe.apply(u8).astype(np.float32) / 255.0
        out[i] = eq
    return out


def process_product(path: Path, out_dir: Path, cfg: PipelineConfig) -> list[dict]:
    arr, profile, meta = open_raster(path)
    product_dir = out_dir / "products" / meta.product_id.replace(":", "_")
    product_dir.mkdir(parents=True, exist_ok=True)

    arr = sensor_cleanup(
        arr,
        meta.sensor,
        destripe=cfg.sensors.destripe,
        iirs_mode=cfg.sensors.iirs_reduce,
        iirs_components=cfg.sensors.iirs_pca_components,
        iirs_band=cfg.sensors.iirs_band_index,
    )
    profile["count"] = arr.shape[0]
    profile["height"] = arr.shape[1]
    profile["width"] = arr.shape[2]
    profile["compress"] = cfg.output.compress

    arr, profile = georeference(
        arr,
        profile,
        meta,
        target_proj4=cfg.georef.crs,
        resampling=cfg.georef.resampling,
        try_isis=cfg.georef.try_isis,
        work_dir=product_dir,
    )

    factor = photometric_factor(
        cfg.illumination.photometric_model,
        meta.incidence_deg,
        meta.emission_deg,
        meta.phase_deg,
    )
    photo = apply_photometry(arr, factor)
    photo = _clahe(photo)

    native_gsd = meta.gsd_m
    working, transform, sf = resample_to_gsd(
        photo,
        profile.get("transform"),
        native_gsd,
        cfg.working_gsd_m,
    )
    profile = dict(profile)
    if transform is not None:
        profile["transform"] = transform
    elif profile.get("transform") is None:
        from rasterio.transform import from_origin

        profile["transform"] = from_origin(0, working.shape[1], 1, 1)
    profile["height"] = working.shape[1]
    profile["width"] = working.shape[2]
    profile["count"] = working.shape[0]
    meta.working_gsd_m = cfg.working_gsd_m
    meta.scale_factor = sf
    if native_gsd is None and transform is not None:
        meta.gsd_m = abs(transform.a)

    write_sidecar_json(meta, product_dir / "metadata.json")

    shadow = shadow_mask(
        working,
        percentile=cfg.illumination.shadow_percentile,
        incidence_deg=meta.incidence_deg,
        incidence_limit=cfg.illumination.incidence_shadow_deg,
    )
    invariants = {}
    if cfg.illumination.write_invariant:
        invariants = build_invariants(working, cfg.illumination.invariant_modes)

    pyramids = gaussian_pyramid(working, max_layer=max(0, cfg.pyramid_levels - 1))
    np.savez_compressed(product_dir / "pyramid_shapes.npz", **{f"L{i}": p.shape for i, p in enumerate(pyramids)})

    records = write_tiles(
        working,
        profile,
        meta,
        out_dir,
        tile_size=cfg.tile_size,
        overlap=cfg.overlap,
        invariants=invariants,
        shadow=shadow if cfg.output.write_shadow_mask else None,
        level=0,
    )
    return [r.to_dict() for r in records]


def _write_catalog(records: list[dict], out_dir: Path, cfg: PipelineConfig) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / cfg.output.catalog_json
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(records, f, indent=2)

    csv_path = out_dir / cfg.output.catalog_csv
    if not records:
        csv_path.write_text("", encoding="utf-8")
        return
    flat = []
    for r in records:
        row = {k: v for k, v in r.items() if k not in {"files", "footprint", "bbox"}}
        row["bbox"] = " ".join(str(x) for x in r.get("bbox", []))
        row["footprint"] = json.dumps(r.get("footprint") or {})
        for fk, fv in (r.get("files") or {}).items():
            row[f"file_{fk}"] = fv
        flat.append(row)
    fieldnames = sorted({k for row in flat for k in row})
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(flat)


def run_pipeline(input_dir: Path, out_dir: Path, config_path: Path | None = None) -> list[dict]:
    cfg = PipelineConfig.load(config_path)
    input_dir = Path(input_dir)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    products = discover_products(input_dir)
    if not products:
        raise FileNotFoundError(f"No PDS4 XML / GeoTIFF / cube products under {input_dir}")

    all_records: list[dict] = []
    for path in tqdm(products, desc="products"):
        all_records.extend(process_product(path, out_dir, cfg))
    _write_catalog(all_records, out_dir, cfg)
    return all_records
