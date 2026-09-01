"""
run_loftr_all_regions.py — Run LoFTR feature matching across all processed triplets.

Processes OHRC and TMC tile pairs for each region in processed_triplets/
and saves verified post-RANSAC correspondence points into processed_user/matches/.
"""

import json
import os
from pathlib import Path
import cv2
import numpy as np
import torch
import kornia as K
from kornia.feature import LoFTR

BASE_DIR = Path(__file__).resolve().parent.parent
PROCESSED_TRIPLETS_DIR = BASE_DIR / "processed_triplets"
OUTPUT_MATCHES_DIR = BASE_DIR / "processed_user" / "matches"
ML_MODEL_DIR = BASE_DIR / "ML_model"


def run_matching():
    OUTPUT_MATCHES_DIR.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[ML] Initializing LoFTR on device: {device}...")
    matcher = LoFTR(pretrained="outdoor").to(device).eval()

    regions = sorted([d for d in os.listdir(PROCESSED_TRIPLETS_DIR) if os.path.isdir(PROCESSED_TRIPLETS_DIR / d)])
    print(f"[ML] Found {len(regions)} regions: {regions}")

    results_summary = {}

    for region_id in regions:
        ohrc_path = PROCESSED_TRIPLETS_DIR / region_id / "ohrc_512.png"
        tmc_path = PROCESSED_TRIPLETS_DIR / region_id / "tmc_512.png"

        if not ohrc_path.is_file() or not tmc_path.is_file():
            print(f"[ML] Skipping {region_id}: missing ohrc_512.png or tmc_512.png")
            continue

        print(f"\n[ML] Processing {region_id}...")
        img1_np = cv2.imread(str(ohrc_path), cv2.IMREAD_GRAYSCALE)
        img2_np = cv2.imread(str(tmc_path), cv2.IMREAD_GRAYSCALE)

        if img1_np.shape != (512, 512):
            img1_np = cv2.resize(img1_np, (512, 512))
        if img2_np.shape != (512, 512):
            img2_np = cv2.resize(img2_np, (512, 512))

        img1 = K.image.image_to_tensor(img1_np, keepdim=True).float() / 255.0
        img2 = K.image.image_to_tensor(img2_np, keepdim=True).float() / 255.0
        img1 = img1.unsqueeze(0).to(device)
        img2 = img2.unsqueeze(0).to(device)

        with torch.no_grad():
            res = matcher({"image0": img1, "image1": img2})

        mkpts0 = res["keypoints0"].cpu().numpy()
        mkpts1 = res["keypoints1"].cpu().numpy()
        confidence = res["confidence"].cpu().numpy()

        print(f"  Raw keypoints found: {len(mkpts0)}")

        matches_data = []
        if len(mkpts0) >= 4:
            H, mask = cv2.findHomography(mkpts0, mkpts1, cv2.RANSAC, 5.0)
            if mask is not None:
                inliers_idx = np.where(mask.ravel() == 1)[0]
                good_mkpts0 = mkpts0[inliers_idx]
                good_mkpts1 = mkpts1[inliers_idx]
                good_conf = confidence[inliers_idx]

                for i in range(len(good_mkpts0)):
                    matches_data.append({
                        "image1_x": round(float(good_mkpts0[i][0]), 2),
                        "image1_y": round(float(good_mkpts0[i][1]), 2),
                        "image2_x": round(float(good_mkpts1[i][0]), 2),
                        "image2_y": round(float(good_mkpts1[i][1]), 2),
                        "confidence": round(float(good_conf[i]), 4),
                    })

                # Sort by confidence descending
                matches_data.sort(key=lambda x: x["confidence"], reverse=True)

        out_path = OUTPUT_MATCHES_DIR / f"{region_id}_matches.json"
        with open(out_path, "w") as f:
            json.dump(matches_data, f, indent=4)

        results_summary[region_id] = len(matches_data)
        print(f"  Saved {len(matches_data)} verified matches to {out_path.name}")

        # If region_001, also update ML_model/matches.json
        if region_id == "region_001":
            ml_out = ML_MODEL_DIR / "matches.json"
            with open(ml_out, "w") as f:
                json.dump(matches_data, f, indent=4)

    print("\n" + "=" * 50)
    print("LOFTR MATCH GENERATION SUMMARY")
    print("=" * 50)
    for reg, cnt in results_summary.items():
        print(f"  {reg}: {cnt} matches")
    print("=" * 50)


if __name__ == "__main__":
    run_matching()
