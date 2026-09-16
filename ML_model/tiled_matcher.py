"""
ML_model/tiled_matcher.py — Multi-Threaded Tiled Matching for Full-Swath Planetary Imagery

Partitions gigapixel / full-swath lunar images into overlapping tiles (e.g. 1024x1024 with 15% overlap),
executes cross-sensor Phase Congruency & CFOG matching concurrently using ThreadPoolExecutor,
and seamlessly stitches correspondences back into global image coordinates to prevent OOM errors.
"""

from __future__ import annotations

import concurrent.futures
from pathlib import Path
from typing import Optional, Tuple, Dict, Any, List
import numpy as np
import cv2

from matcher_cfog import match_images_cfog, load_as_float_and_color
from metrics import compute_canonical_metrics, verify_transformation_quality
from geometry import warp_piecewise_affine


def match_single_tile(
    tile_crop1: np.ndarray,
    tile_crop2: np.ndarray,
    offset1: Tuple[int, int],
    offset2: Tuple[int, int],
    tile_id: int,
    source_sensor: Optional[str] = None,
    reference_sensor: Optional[str] = None,
    working_dir: Optional[Path] = None,
) -> Dict[str, Any]:
    """Processes a single tile pair and offsets match coordinates back to global space.

    Perf: crops are passed in-memory straight into ``match_images_cfog``
    (ndarray passthrough) — no PNG tile I/O. ``working_dir`` only hosts the
    matcher's standard output bundle (metrics/matches JSON), not re-encoded tiles.
    """
    ox1, oy1 = offset1
    ox2, oy2 = offset2

    tile_dir = working_dir / f"tile_{tile_id:04d}" if working_dir else Path(f"tile_{tile_id:04d}")
    tile_dir.mkdir(parents=True, exist_ok=True)

    try:
        res = match_images_cfog(
            np.ascontiguousarray(tile_crop1),
            np.ascontiguousarray(tile_crop2),
            output_dir=tile_dir,
            source_sensor=source_sensor,
            reference_sensor=reference_sensor,
        )
        if res.get("status") == "success" and res.get("matches"):
            # Offset tile-local matches to global image space
            global_matches = []
            for m in res["matches"]:
                m_copy = dict(m)
                m_copy["source_x"] = round(float(m["source_x"] + ox1), 2)
                m_copy["source_y"] = round(float(m["source_y"] + oy1), 2)
                m_copy["target_x"] = round(float(m["target_x"] + ox2), 2)
                m_copy["target_y"] = round(float(m["target_y"] + oy2), 2)
                m_copy["image1_x"] = m_copy["source_x"]
                m_copy["image1_y"] = m_copy["source_y"]
                m_copy["image2_x"] = m_copy["target_x"]
                m_copy["image2_y"] = m_copy["target_y"]
                m_copy["tile_id"] = tile_id
                global_matches.append(m_copy)
            return {"status": "success", "tile_id": tile_id, "matches": global_matches}
    except Exception as e:
        return {"status": "failed", "tile_id": tile_id, "error": str(e), "matches": []}
    return {"status": "failed", "tile_id": tile_id, "matches": []}


def match_images_tiled(
    img_path1: str | Path,
    img_path2: str | Path,
    tile_size: int = 1024,
    overlap_ratio: float = 0.15,
    max_workers: int = 4,
    dem_path: Optional[str | Path] = None,
    output_dir: str | Path = "tiled_output",
    source_sensor: Optional[str] = None,
    reference_sensor: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Full-swath tiled registration runner.
    
    1. Loads images and checks dimensions.
    2. Partitions scenes into tiles (default 1024x1024 with 15% overlap).
    3. Concurrently processes tiles using ThreadPoolExecutor.
    4. Stitches correspondence points and runs global verification.
    """
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    # Windowed I/O plan: for raster inputs, query (h, w) without reading pixels,
    # then read per-tile windows inside workers. Falls back to full-load slicing
    # for plain PNG/JPG inputs (identical numerics to the legacy path).
    def _raster_shape(p) -> Optional[Tuple[int, int]]:
        try:
            import rasterio
            with rasterio.open(str(p)) as _src:
                return int(_src.height), int(_src.width)
        except Exception:
            return None

    def _read_window(p, x0: int, y0: int, x1: int, y1: int) -> np.ndarray:
        import rasterio
        from rasterio.windows import Window as _W
        with rasterio.open(str(p)) as _src:
            _win = _W(int(x0), int(y0), max(1, int(x1 - x0)), max(1, int(y1 - y0)))
            _arr = _src.read(window=_win)  # (B, H, W)
            if _arr.shape[0] == 1:
                return _arr[0]
            if _arr.shape[0] in (3, 4):
                return np.moveaxis(_arr[:3], 0, -1)
            return _arr

    _sh1, _sh2 = _raster_shape(img_path1), _raster_shape(img_path2)
    windowed = _sh1 is not None and _sh2 is not None
    if windowed:
        h1, w1 = _sh1  # type: ignore[misc]
        h2, w2 = _sh2  # type: ignore[misc]
        img1 = img2 = None  # never materialized; workers read windows
    else:
        img1, _, meta1 = load_as_float_and_color(img_path1)
        img2, _, meta2 = load_as_float_and_color(img_path2)
        h1, w1 = img1.shape[:2]
        h2, w2 = img2.shape[:2]

    # If small enough, run standard single-pass matching
    if max(h1, w1, h2, w2) <= tile_size:
        return match_images_cfog(
            img_path1,
            img_path2,
            dem_path=dem_path,
            output_dir=output_dir,
            source_sensor=source_sensor,
            reference_sensor=reference_sensor,
        )

    step1 = max(64, int(tile_size * (1.0 - overlap_ratio)))
    scale_x = float(w2) / float(w1)
    scale_y = float(h2) / float(h1)

    tile_tasks = []
    tile_id = 0

    for y0_1 in range(0, h1, step1):
        y1_1 = min(h1, y0_1 + tile_size)
        for x0_1 in range(0, w1, step1):
            x1_1 = min(w1, x0_1 + tile_size)

            # Map corresponding tile bounding box in image 2
            x0_2 = int(x0_1 * scale_x)
            y0_2 = int(y0_1 * scale_y)
            x1_2 = min(w2, int(x1_1 * scale_x))
            y1_2 = min(h2, int(y1_1 * scale_y))

            if windowed:
                # Defer pixel reads to workers (windowed I/O); keep boxes only.
                if x1_1 > x0_1 and y1_1 > y0_1 and x1_2 > x0_2 and y1_2 > y0_2:
                    tile_tasks.append((
                        None, None, (x0_1, y0_1), (x0_2, y0_2), tile_id,
                        (x0_1, y0_1, x1_1, y1_1), (x0_2, y0_2, x1_2, y1_2),
                    ))
                    tile_id += 1
            else:
                crop1 = img1[y0_1:y1_1, x0_1:x1_1]
                crop2 = img2[y0_2:y1_2, x0_2:x1_2]
                if crop1.size > 0 and crop2.size > 0:
                    tile_tasks.append((
                        crop1, crop2, (x0_1, y0_1), (x0_2, y0_2), tile_id,
                        None, None,
                    ))
                    tile_id += 1

    def _run_task(task) -> Dict[str, Any]:
        c1, c2, off1, off2, tid, box1, box2 = task
        try:
            if c1 is None and windowed:
                c1 = _read_window(img_path1, *box1)  # type: ignore[arg-type]
                c2 = _read_window(img_path2, *box2)  # type: ignore[arg-type]
                if getattr(c1, "size", 0) == 0 or getattr(c2, "size", 0) == 0:
                    return {"status": "failed", "tile_id": tid, "matches": []}
            return match_single_tile(
                c1, c2, off1, off2, tid,
                source_sensor=source_sensor,
                reference_sensor=reference_sensor,
                working_dir=out_path / "tiles",
            )
        except Exception as e:
            return {"status": "failed", "tile_id": tid, "error": str(e), "matches": []}

    # Execute tile matching in parallel using ThreadPoolExecutor.
    # RNG honesty: cv2's RNG is process-global and shared across these threads,
    # so bit-reproducibility is NOT claimed for threaded runs. Decorrelation IS
    # guaranteed: match_images_cfog seeds SEED ^ hash(pair) per call and every
    # tile writes distinct tile_{id} paths, so tiles never lock-step onto
    # identical RANSAC draws (the old per-call setRNGSeed(42) did exactly that).
    all_stitched_matches = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(_run_task, t) for t in tile_tasks]
        for f in concurrent.futures.as_completed(futures):
            res = f.result()
            if res.get("status") == "success" and res.get("matches"):
                all_stitched_matches.extend(res["matches"])

    # Deduplicate matches across overlapping tile seams (within 3.0 pixels)
    unique_matches = []
    seen_coords = set()
    for m in all_stitched_matches:
        grid_key = (int(round(m["source_x"] / 3.0)), int(round(m["source_y"] / 3.0)))
        if grid_key not in seen_coords:
            seen_coords.add(grid_key)
            unique_matches.append(m)

    if len(unique_matches) < 4:
        return {
            "status": "insufficient_matches",
            "message": f"Tiled matching found {len(unique_matches)} unique correspondences across {len(tile_tasks)} tiles.",
            "match_count": len(unique_matches),
            "inlier_count": 0,
            "metrics": None,
            "matches": unique_matches,
        }

    # Fit global transformation
    src_pts = np.array([[m["source_x"], m["source_y"]] for m in unique_matches], dtype=np.float32)
    dst_pts = np.array([[m["target_x"], m["target_y"]] for m in unique_matches], dtype=np.float32)

    H_global, inlier_mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

    metrics = compute_canonical_metrics(
        src_pts, dst_pts, inlier_mask, H_global, (h2, w2), grid_size=10
    )

    return {
        "status": "success",
        "match_count": len(unique_matches),
        "inlier_count": metrics.get("inlier_count", 0),
        "metrics": metrics,
        "homography": H_global.tolist() if H_global is not None else None,
        "matches": unique_matches,
        "tiles_processed": len(tile_tasks),
    }
