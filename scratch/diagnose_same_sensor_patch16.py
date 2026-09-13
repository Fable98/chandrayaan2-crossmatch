"""Diagnose why half_p=16 breaks same-sensor matching."""

import sys
from pathlib import Path
import numpy as np
import cv2
import math

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "ML_model"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from config import SEED
from evaluation.eval_candidate_generation import get_train_test_split
from scripts.generate_ground_truth_matches import (
    create_random_homography,
    apply_photometric_perturbation,
)
from matcher_cfog import (
    compute_phase_congruency,
    detect_salient_keypoints,
    suppression_via_square_covering,
    find_best_correspondence_unified,
    subpixel_phase_correlation,
    compute_spatial_quality_score,
    compute_descriptor_match_features,
    build_cfog_descriptor_gallery,
    estimate_weighted_homography,
    calculate_reprojection_errors,
)

train_groups, test_groups = get_train_test_split()
same_sensor_train = [g for g in train_groups if "same_sensor" in g]
print(f"Same-sensor training groups: {len(same_sensor_train)}")

triplets_dir = REPO_ROOT / "data_preprocessing_pipeline/processed_triplets"
p = triplets_dir / "region_001/ohrc_512.png"
img_gray = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
h, w = img_gray.shape[:2]

rng = np.random.default_rng(SEED)
H_gt = create_random_homography(w, h, rng)
warped = cv2.warpPerspective(img_gray, H_gt, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
warped_ss = apply_photometric_perturbation(warped, rng)

pc1 = compute_phase_congruency(img_gray)
pc2 = compute_phase_congruency(warped_ss)

kps_raw = detect_salient_keypoints(pc1, max_corners=300, quality_level=0.008)
kps_ssc = suppression_via_square_covering(kps_raw, num_ret_points=60, tolerance=0.12, cols=w, rows=h)
kps2_raw = detect_salient_keypoints(pc2, max_corners=200, quality_level=0.01)
tgt_gallery = build_cfog_descriptor_gallery(pc2, kps2_raw)

print(f"kps_raw: {len(kps_raw)}, kps_ssc: {len(kps_ssc)}")

for hp in [8, 16]:
    search_rad = 20
    jitter = 8.0
    matched = []
    skipped_reasons = {"border_src": 0, "tmpl_zero": 0, "border_tgt": 0, "search_small": 0, "subpixel_fail": 0, "label_none": 0}
    errs = []
    sub_errs = []
    scores = []

    for item in kps_ssc:
        kx, ky = float(item[0]), float(item[1])
        cx, cy = int(round(kx)), int(round(ky))
        if cy < hp or cy >= h - hp or cx < hp or cx >= w - hp:
            skipped_reasons["border_src"] += 1
            continue
        tmpl = pc1[cy - hp : cy + hp, cx - hp : cx + hp]
        if float(np.std(tmpl)) < 1e-4:
            skipped_reasons["tmpl_zero"] += 1
            continue

        p_gt = cv2.perspectiveTransform(np.array([[[kx, ky]]], dtype=np.float64), H_gt).reshape(-1)
        gt_x, gt_y = float(p_gt[0]), float(p_gt[1])
        search_cx = int(round(gt_x + float(rng.uniform(-jitter, jitter))))
        search_cy = int(round(gt_y + float(rng.uniform(-jitter, jitter))))
        s_min_x, s_max_x = max(0, search_cx - search_rad), min(w, search_cx + search_rad)
        s_min_y, s_max_y = max(0, search_cy - search_rad), min(h, search_cy + search_rad)
        if s_max_x - s_min_x <= tmpl.shape[1] or s_max_y - s_min_y <= tmpl.shape[0]:
            skipped_reasons["search_small"] += 1
            continue
        search_region = pc2[s_min_y:s_max_y, s_min_x:s_max_x]
        if float(np.std(search_region)) < 1e-4:
            skipped_reasons["search_small"] += 1
            continue

        score, loc = find_best_correspondence_unified(search_region, tmpl, multimodal_pair=False)
        found_x = float(s_min_x + loc[0] + hp)
        found_y = float(s_min_y + loc[1] + hp)

        raw_err = math.hypot(found_x - gt_x, found_y - gt_y)

        # Subpixel refinement
        ref_dx, ref_dy = 0.0, 0.0
        refined = False
        ibx, iby = int(round(found_x)), int(round(found_y))
        if iby >= hp and iby + hp <= h and ibx >= hp and ibx + hp <= w:
            p_ref = pc2[iby - hp : iby + hp, ibx - hp : ibx + hp]
            if p_ref.shape == tmpl.shape:
                dx, dy, _peak, valid = subpixel_phase_correlation(tmpl, p_ref)
                if valid and abs(dx) < 2.0 and abs(dy) < 2.0:
                    ref_dx, ref_dy = float(dx), float(dy)
                    found_x += ref_dx
                    found_y += ref_dy
                    refined = True

        final_err = math.hypot(found_x - gt_x, found_y - gt_y)
        errs.append(final_err)
        scores.append(score)

        # Check label
        # true_px = 2.0, false_px = 5.0
        if final_err <= 2.0:
            lbl = True
        elif final_err >= 5.0:
            lbl = False
        else:
            lbl = None
            skipped_reasons["label_none"] += 1
            continue

        matched.append({
            "source_x": kx, "source_y": ky, "target_x": found_x, "target_y": found_y,
            "confidence": score, "ground_truth_label": lbl, "err": final_err,
        })

    pts1 = np.array([[m["source_x"], m["source_y"]] for m in matched], dtype=np.float32)
    pts2 = np.array([[m["target_x"], m["target_y"]] for m in matched], dtype=np.float32)
    w_u = np.ones(len(matched))
    H_est, mask, _ = estimate_weighted_homography(pts1, pts2, w_u, ransac_reproj_threshold=5.0, rng_seed=42)

    corners = np.array([[[0.0, 0.0]], [[w, 0.0]], [[w, h]], [[0.0, h]]], dtype=np.float64)
    proj_gt = cv2.perspectiveTransform(corners, H_gt).reshape(-1, 2)
    if H_est is not None:
        proj_est = cv2.perspectiveTransform(corners, H_est).reshape(-1, 2)
        corner_err = float(np.mean(np.linalg.norm(proj_est - proj_gt, axis=1)))
    else:
        corner_err = float("inf")

    n_inl = int(np.sum(mask)) if mask is not None else 0
    inlier_labels = [matched[i]["ground_truth_label"] for i in range(len(matched)) if mask is not None and mask[i]]
    tp = sum(1 for l in inlier_labels if l is True)
    fp = sum(1 for l in inlier_labels if l is False)

    print(f"\n=================== half_p = {hp} ===================")
    print(f"Skipped reasons: {skipped_reasons}")
    print(f"Retained candidates: {len(matched)} / {len(kps_ssc)}")
    print(f"True candidates (err <= 2px): {sum(1 for e in errs if e <= 2.0)} ({sum(1 for e in errs if e <= 2.0)/max(1, len(errs))*100:.1f}%)")
    print(f"Mean error: {np.mean(errs):.3f} px, Median error: {np.median(errs):.3f} px")
    print(f"Mean score: {np.mean(scores):.3f}")
    print(f"RANSAC inliers: {n_inl}/{len(matched)} (TP={tp}, FP={fp})")
    print(f"Corner error vs H_gt: {corner_err:.3f} px")
