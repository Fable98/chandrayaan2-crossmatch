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
    """Processes a single tile pair and offsets match coordinates back to global space."""
    ox1, oy1 = offset1
    ox2, oy2 = offset2

    tile_dir = working_dir / f"tile_{tile_id:04d}" if working_dir else Path(f"tile_{tile_id:04d}")
    tile_dir.mkdir(parents=True, exist_ok=True)

    p1 = tile_dir / "tile1.png"
    p2 = tile_dir / "tile2.png"

    u8_1 = (np.clip(tile_crop1, 0.0, 1.0) * 255.0).astype(np.uint8) if tile_crop1.max() <= 1.0 else tile_crop1.astype(np.uint8)
    u8_2 = (np.clip(tile_crop2, 0.0, 1.0) * 255.0).astype(np.uint8) if tile_crop2.max() <= 1.0 else tile_crop2.astype(np.uint8)

    cv2.imwrite(str(p1), u8_1)
    cv2.imwrite(str(p2), u8_2)

    try:
        res = match_images_cfog(
            p1,
            p2,
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

            crop1 = img1[y0_1:y1_1, x0_1:x1_1]
            crop2 = img2[y0_2:y1_2, x0_2:x1_2]

            if crop1.size > 0 and crop2.size > 0:
                tile_tasks.append((
                    crop1, crop2, (x0_1, y0_1), (x0_2, y0_2), tile_id,
                ))
                tile_id += 1

    # Execute tile matching in parallel using ThreadPoolExecutor
    all_stitched_matches = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [
            executor.submit(
                match_single_tile,
                c1, c2, off1, off2, tid,
                source_sensor=source_sensor,
                reference_sensor=reference_sensor,
                working_dir=out_path / "tiles",
            )
            for (c1, c2, off1, off2, tid) in tile_tasks
        ]
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
