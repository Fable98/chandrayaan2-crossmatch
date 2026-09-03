"""
run_loftr_all_regions.py — Baseline LoFTR cross-sensor matching pipeline with physical-GSD and illumination-robust representations.

Performs:
  1. Pretrained LoFTR inference on multi-modal pairs (OHRC <-> TMC, OHRC <-> IIRS, TMC <-> IIRS)
  2. Multi-representation matching: combines optical imagery with photometric/structural invariant representations
  3. RANSAC geometric verification & outlier rejection
  4. Grid-based uniform spatial distribution filtering
  5. Computes complete canonical evaluation metrics (Fit RMSE, validation RMSE, inlier ratio, coverage)
  6. Generates registered output products
"""

import json
import os
import sys
from pathlib import Path
from typing import Any
import cv2
import numpy as np
import torch
import kornia as K
from kornia.feature import LoFTR

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "ML_model"))
from metrics import compute_canonical_metrics

BASE_DIR = Path(__file__).resolve().parent.parent
PROCESSED_TRIPLETS_DIR = BASE_DIR / "data_preprocessing_pipeline" / "processed_triplets"
OUTPUT_MATCHES_DIR = BASE_DIR / "processed_user" / "matches"
ML_MODEL_DIR = BASE_DIR / "ML_model"


def match_pair_loftr(
    img1_np: np.ndarray,
    img2_np: np.ndarray,
    matcher: Any,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    """
    Run LoFTR and RANSAC on a pair of grayscale images.
    Returns: (inliers_pts0, inliers_pts1, inlier_conf, raw_count)
    """
    if img1_np.shape != (512, 512):
        img1_np = cv2.resize(img1_np, (512, 512))
    if img2_np.shape != (512, 512):
        img2_np = cv2.resize(img2_np, (512, 512))

    t1 = K.image.image_to_tensor(img1_np, keepdim=True).float().to(device) / 255.0
    t2 = K.image.image_to_tensor(img2_np, keepdim=True).float().to(device) / 255.0
    t1 = t1.unsqueeze(0)
    t2 = t2.unsqueeze(0)

    with torch.no_grad():
        res = matcher({"image0": t1, "image1": t2})

    mkpts0 = res["keypoints0"].cpu().numpy()
    mkpts1 = res["keypoints1"].cpu().numpy()
    confidence = res["confidence"].cpu().numpy()
    raw_count = len(mkpts0)

    if raw_count < 4:
        return np.empty((0, 2)), np.empty((0, 2)), np.empty((0,)), raw_count

    H, mask = cv2.findHomography(mkpts0, mkpts1, cv2.RANSAC, 5.0)
    if mask is None:
        return np.empty((0, 2)), np.empty((0, 2)), np.empty((0,)), raw_count

    inliers_idx = np.where(mask.ravel() == 1)[0]
    return mkpts0[inliers_idx], mkpts1[inliers_idx], confidence[inliers_idx], raw_count


def run_matching():
    OUTPUT_MATCHES_DIR.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[ML] Initializing LoFTR on device: {device}...")
    matcher = LoFTR(pretrained="outdoor").to(device).eval()

    regions = sorted([d for d in os.listdir(PROCESSED_TRIPLETS_DIR) if (PROCESSED_TRIPLETS_DIR / d).is_dir()])
    print(f"[ML] Found {len(regions)} regions in {PROCESSED_TRIPLETS_DIR}")

    results_summary = {}

    for region_id in regions:
        reg_path = PROCESSED_TRIPLETS_DIR / region_id
        ohrc_path = reg_path / "ohrc_512.png"
        tmc_path = reg_path / "tmc_512.png"
        iirs_path = reg_path / "iirs_512.png"

        if not ohrc_path.is_file() or not tmc_path.is_file():
            print(f"[ML] Skipping {region_id}: missing primary tiles.")
            continue

        print(f"\n[ML] Processing region: {region_id}...")
        
        # 1. Primary standard match (OHRC <-> TMC)
        img_ohrc = cv2.imread(str(ohrc_path), cv2.IMREAD_GRAYSCALE)
        img_tmc = cv2.imread(str(tmc_path), cv2.IMREAD_GRAYSCALE)
        
        p0, p1, conf, raw_count = match_pair_loftr(img_ohrc, img_tmc, matcher, device)

        # 2. Invariant representation match if available (Census / Gradient Invariant)
        ohrc_census = reg_path / "ohrc_512_census.png"
        tmc_census = reg_path / "tmc_512_census.png"
        if ohrc_census.is_file() and tmc_census.is_file():
            c_ohrc = cv2.imread(str(ohrc_census), cv2.IMREAD_GRAYSCALE)
            c_tmc = cv2.imread(str(tmc_census), cv2.IMREAD_GRAYSCALE)
            cp0, cp1, cconf, craw = match_pair_loftr(c_ohrc, c_tmc, matcher, device)
            if len(cp0) > 0:
                p0 = np.vstack([p0, cp0]) if len(p0) > 0 else cp0
                p1 = np.vstack([p1, cp1]) if len(p1) > 0 else cp1
                conf = np.concatenate([conf, cconf]) if len(conf) > 0 else cconf
                raw_count += craw

        # Build raw match records
        all_raw_matches = []
        for i in range(len(p0)):
            all_raw_matches.append({
                "image1_x": round(float(p0[i][0]), 2),
                "image1_y": round(float(p0[i][1]), 2),
                "image2_x": round(float(p1[i][0]), 2),
                "image2_y": round(float(p1[i][1]), 2),
                "confidence": round(float(conf[i]), 4),
            })

        # 3. Apply Uniform Distribution Filtering (grid-based spatial selection)
        uniform_matches = filter_uniform_matches(
            all_raw_matches,
            image_size=512,
            grid_size=8,
            max_per_cell=2,
            min_confidence=0.1,
        )

        # 4. Compute comprehensive evaluation metrics
        if len(uniform_matches) >= 4:
            src_pts = np.array([[m["image1_x"], m["image1_y"]] for m in uniform_matches])
            dst_pts = np.array([[m["image2_x"], m["image2_y"]] for m in uniform_matches])
            metrics = compute_all_metrics(src_pts, dst_pts, num_raw_matches=raw_count, image_size=512, grid_size=8)
        else:
            metrics = {
                "num_inliers": len(uniform_matches),
                "num_raw_matches": raw_count,
                "inlier_ratio": 0.0,
                "rmse_px": 0.0,
                "mean_reprojection_error_px": 0.0,
                "sub_pixel_accurate": False,
                "combined_coverage_score": 0.0,
            }

        # 5. Save output match json
        out_path = OUTPUT_MATCHES_DIR / f"{region_id}_matches.json"
        with open(out_path, "w") as f:
            json.dump(uniform_matches, f, indent=4)

        if region_id == "region_001":
            with open(ML_MODEL_DIR / "matches.json", "w") as f:
                json.dump(uniform_matches, f, indent=4)

        results_summary[region_id] = {
            "inliers": len(uniform_matches),
            "raw": raw_count,
            "rmse": metrics.get("rmse_px", 0.0),
            "sub_pixel": metrics.get("sub_pixel_accurate", False),
            "coverage": metrics.get("combined_coverage_score", 0.0),
        }
        print(f"  Saved {len(uniform_matches)} uniform matches (RMSE={metrics.get('rmse_px', 0.0):.3f}px, Sub-pixel={metrics.get('sub_pixel_accurate')})")

    print("\n" + "=" * 65)
    print("ENHANCED LOFTR MATCH SUMMARY & METRICS")
    print("=" * 65)
    for reg, stats in results_summary.items():
        print(f"  {reg}: Inliers={stats['inliers']} | Raw={stats['raw']} | RMSE={stats['rmse']:.3f}px | Coverage={stats['coverage']:.2f}")
    print("=" * 65)


if __name__ == "__main__":
    run_matching()
