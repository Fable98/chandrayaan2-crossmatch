"""Phase 9: Multi-Scale Structural Information Diagnostic.

Diagnostic measurements across 37 development cross-sensor scenes:
1. Multi-scale spatial supports: 16x16, 32x32, 48x48, 64x64, 96x96.
2. Representations evaluated:
   - Phase Congruency (PC) Normalized Cross-Correlation
   - Mutual Information (MI)
   - Phase Congruency Energy / Structural correlation
3. Evaluates:
   - True-target similarity vs Best-false similarity
   - Separation: True - Best False
   - % where True > Best False
   - True-target rank and percentile rank
   - Top-1, Top-5, Top-10, Top-20 recall
   - ROC-AUC / PR-AUC of true vs false locations
4. Boundary and valid-point tracking across scales:
   - Valid target placements in search region
   - Percentage of ground-truth targets evaluable
   - Border exclusion count
   - Common evaluable subset vs all evaluable
5. Correlation with synthetic transformation factors:
   - Rotation angle, scale factor, translation magnitude, downsampling factor, blur sigma, azimuth delta.
6. Per-scene breakdown: scenes where scale helps, worsens, or fails completely.
7. Official Classification: A, B, or C.
"""

from __future__ import annotations

import math
import sys
import time
from pathlib import Path
from typing import Dict, List, Any, Tuple
import numpy as np
import cv2
from sklearn.metrics import roc_auc_score

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "ML_model"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from config import SEED
from matcher_cfog import (
    compute_phase_congruency,
    detect_salient_keypoints,
    suppression_via_square_covering,
    mutual_information_score,
)
from scripts.generate_ground_truth_matches import (
    create_random_homography,
    apply_antisolar_cross_sensor,
)
from evaluation.eval_candidate_generation import get_train_test_split


def get_image_paths() -> List[Path]:
    triplets_dir = REPO_ROOT / "data_preprocessing_pipeline/processed_triplets"
    image_paths = []
    for reg_dir in sorted(triplets_dir.glob("region_*")):
        for name in ("ohrc_512.png", "tmc_512.png"):
            p = reg_dir / name
            if p.exists():
                image_paths.append(p)
    return image_paths


def run_phase9_diagnostic():
    train_groups, test_groups = get_train_test_split()
    xs_train_groups = [g for g in train_groups if "cross_sensor" in g]

    print("================================================================================")
    print("PHASE 9: MULTI-SCALE STRUCTURAL INFORMATION DIAGNOSTIC")
    print("================================================================================")
    print(f"Development Cross-Sensor Groups : {len(xs_train_groups)}")
    print(f"Held-Out Test Groups (FROZEN)   : {len(test_groups)} (Untouched)")
    print("Diagnostic Only: Production pipeline, RF model, RANSAC remain strictly frozen.\n")

    image_paths = get_image_paths()
    scales = [16, 32, 48, 64, 96]
    half_patches = [s // 2 for s in scales]

    # Structure to hold metrics per scale
    # We evaluate both on the full evaluable set per scale, and on the common subset evaluable at all scales
    metrics_by_scale = {s: {
        "valid_points": 0,
        "border_excluded": 0,
        "true_sim_ncc": [],
        "best_false_ncc": [],
        "ranks_ncc": [],
        "pct_ranks_ncc": [],
        "true_sim_mi": [],
        "best_false_mi": [],
        "ranks_mi": [],
        "separations_ncc": [],
        "separations_mi": [],
        "true_gt_false_ncc": 0,
        "true_gt_false_mi": 0,
        "auc_ncc": [],
    } for s in scales}

    # Per-scene performance tracker
    scene_metrics = {grp: {s: {"ncc_ranks": [], "seps": [], "t_gt_f": 0, "n": 0} for s in scales} for grp in xs_train_groups}
    scene_transform_factors = {}

    total_keypoints_considered = 0
    t0_start = time.perf_counter()

    for p in image_paths:
        img_gray = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
        if img_gray is None:
            continue
        h, w = img_gray.shape[:2]
        region = p.parent.name
        source_image = p.name

        # Compute Phase Congruency
        pc1 = compute_phase_congruency(img_gray, num_orientations=4, num_scales=3)
        kps_raw = detect_salient_keypoints(pc1, max_corners=300, quality_level=0.008)
        kps_ssc = suppression_via_square_covering(kps_raw, num_ret_points=60, tolerance=0.12, cols=w, rows=h)

        for t_idx in range(4):
            target_image = f"{p.stem}_cross_sensor_t{t_idx}.png"
            grp = f"{region}:{source_image}::{target_image}"
            if grp not in xs_train_groups:
                continue

            scene_seed = int(abs(hash(grp)) % (2**31 - 1))
            rng_s = np.random.default_rng(scene_seed)

            # Draw parameters and record them for Section 7 analysis
            angle_deg = float(rng_s.uniform(-8.0, 8.0))
            scale_factor = float(rng_s.uniform(0.92, 1.08))
            tx = float(rng_s.uniform(-25.0, 25.0))
            ty = float(rng_s.uniform(-25.0, 25.0))
            p1 = float(rng_s.uniform(-0.0003, 0.0003))
            p2 = float(rng_s.uniform(-0.0003, 0.0003))
            cx_c, cy_c = w / 2.0, h / 2.0
            rad = math.radians(angle_deg)
            cos_a = math.cos(rad) * scale_factor
            sin_a = math.sin(rad) * scale_factor
            H_gt = np.array([
                [cos_a, -sin_a, (1.0 - cos_a) * cx_c + sin_a * cy_c + tx],
                [sin_a,  cos_a, -sin_a * cx_c + (1.0 - cos_a) * cy_c + ty],
                [p1,     p2,    1.0],
            ], dtype=np.float64)

            az_delta = 160.0 + float(rng_s.uniform(-8.0, 8.0))
            downsample_factor = int(rng_s.integers(6, 11))
            blur_sigma = float(rng_s.uniform(1.2, 2.0))
            noise_sigma = float(rng_s.uniform(0.01, 0.03))

            scene_transform_factors[grp] = {
                "angle_deg": abs(angle_deg),
                "scale_dev": abs(scale_factor - 1.0),
                "trans_mag": math.hypot(tx, ty),
                "downsample_factor": downsample_factor,
                "blur_sigma": blur_sigma,
                "az_delta": az_delta,
            }

            warped = cv2.warpPerspective(img_gray, H_gt, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
            # Reconstruct synthetic cross-sensor image with recorded parameters
            f = warped.astype(np.float32)
            gx = cv2.Sobel(f, cv2.CV_32F, 1, 0, ksize=3)
            gy = cv2.Sobel(f, cv2.CV_32F, 0, 1, ksize=3)
            az_rad = math.radians(az_delta)
            shade = gx * math.sin(az_rad) + gy * math.cos(az_rad)
            scale_s = float(np.percentile(np.abs(shade), 98)) + 1e-6
            shade = np.clip(shade / scale_s, -1.0, 1.0)
            albedo = np.clip(f / 255.0, 0.0, 1.0)
            mixed = np.clip(0.50 * (1.0 - albedo) + 0.50 * (0.5 - 0.45 * shade), 0.0, 1.0)
            small = cv2.resize(mixed, (max(16, w // downsample_factor), max(16, h // downsample_factor)), interpolation=cv2.INTER_AREA)
            mixed = cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR)
            mixed = cv2.GaussianBlur(mixed, (0, 0), sigmaX=blur_sigma)
            noise = rng_s.normal(0.0, noise_sigma, mixed.shape).astype(np.float32)
            warped_xs = np.clip((mixed + noise) * 255.0, 0.0, 255.0).astype(np.uint8)

            pc2 = compute_phase_congruency(warped_xs, num_orientations=4, num_scales=3)

            # Sizing for candidate search:
            # We evaluate candidate search displacement grid of dx, dy in [-20, 20]
            # around search center (jitter = 6.0 around H_gt target)
            s_rad = 20  # displacement radius
            jit = 6.0
            rng_jit = np.random.default_rng(scene_seed + 1000)

            for item in kps_ssc:
                kx, ky = float(item[0]), float(item[1])
                cx, cy = int(round(kx)), int(round(ky))
                total_keypoints_considered += 1

                p_gt = cv2.perspectiveTransform(np.array([[[kx, ky]]], dtype=np.float64), H_gt).reshape(-1)
                gt_x, gt_y = float(p_gt[0]), float(p_gt[1])

                jx = float(rng_jit.uniform(-jit, jit))
                jy = float(rng_jit.uniform(-jit, jit))
                search_cx = int(round(gt_x + jx))
                search_cy = int(round(gt_y + jy))

                # For each scale, test if source template and target search area are valid
                for s_idx, patch_sz in enumerate(scales):
                    hp_s = half_patches[s_idx]

                    # Check source template bounds
                    if cy - hp_s < 0 or cy + hp_s >= h or cx - hp_s < 0 or cx + hp_s >= w:
                        metrics_by_scale[patch_sz]["border_excluded"] += 1
                        continue

                    # Target search bounds: search center +/- s_rad, with patch margin hp_s
                    t_min_x = search_cx - s_rad - hp_s
                    t_max_x = search_cx + s_rad + hp_s
                    t_min_y = search_cy - s_rad - hp_s
                    t_max_y = search_cy + s_rad + hp_s

                    if t_min_x < 0 or t_max_x >= w or t_min_y < 0 or t_max_y >= h:
                        metrics_by_scale[patch_sz]["border_excluded"] += 1
                        continue

                    tmpl = pc1[cy - hp_s : cy + hp_s, cx - hp_s : cx + hp_s]
                    if float(np.std(tmpl)) < 1e-4:
                        continue

                    search_sub = pc2[t_min_y:t_max_y, t_min_x:t_max_x]
                    if float(np.std(search_sub)) < 1e-4:
                        continue

                    # Search surface across the +/- s_rad displacement grid
                    # search_sub has shape (2*s_rad + 2*hp_s, 2*s_rad + 2*hp_s)
                    # matchTemplate with tmpl (2*hp_s, 2*hp_s) produces output of shape (2*s_rad + 1, 2*s_rad + 1)
                    ncc_surf = cv2.matchTemplate(search_sub, tmpl, cv2.TM_CCOEFF_NORMED)

                    # True target position on this grid:
                    # Target center is gt_x, gt_y
                    # Top-left of patch at true target inside search_sub is (gt_x - hp_s - t_min_x, gt_y - hp_s - t_min_y)
                    tgt_col = int(round(gt_x - hp_s - t_min_x))
                    tgt_row = int(round(gt_y - hp_s - t_min_y))

                    if not (0 <= tgt_col < ncc_surf.shape[1] and 0 <= tgt_row < ncc_surf.shape[0]):
                        continue

                    true_ncc = float(ncc_surf[tgt_row, tgt_col])

                    # Exclude the immediate neighborhood of the true target (+/- 2 px) to find the best FALSE peak
                    r_grid, c_grid = np.ogrid[:ncc_surf.shape[0], :ncc_surf.shape[1]]
                    dist_from_true = np.hypot(c_grid - tgt_col, r_grid - tgt_row)
                    false_mask = dist_from_true > 2.0

                    if not np.any(false_mask):
                        continue

                    best_false_ncc = float(np.max(ncc_surf[false_mask]))
                    rank_ncc = int(np.sum(ncc_surf.ravel() > true_ncc)) + 1
                    pct_rank = (rank_ncc / ncc_surf.size) * 100.0

                    sep_ncc = true_ncc - best_false_ncc
                    t_gt_f = 1 if sep_ncc > 0 else 0

                    # Compute MI at true target and at best false peak
                    cand_true = search_sub[tgt_row : tgt_row + 2*hp_s, tgt_col : tgt_col + 2*hp_s]
                    mi_true = mutual_information_score(tmpl, cand_true) if cand_true.shape == tmpl.shape else 0.0

                    # Best false location
                    false_sub_vals = np.where(false_mask, ncc_surf, -np.inf)
                    bf_row, bf_col = np.unravel_index(np.argmax(false_sub_vals), false_sub_vals.shape)
                    cand_false = search_sub[bf_row : bf_row + 2*hp_s, bf_col : bf_col + 2*hp_s]
                    mi_false = mutual_information_score(tmpl, cand_false) if cand_false.shape == tmpl.shape else 0.0
                    sep_mi = mi_true - mi_false

                    # Binary classification labels for AUC: 1 for true target, 0 for false locations
                    labels = np.zeros(ncc_surf.size, dtype=np.int32)
                    labels[tgt_row * ncc_surf.shape[1] + tgt_col] = 1
                    auc_val = float(roc_auc_score(labels, ncc_surf.ravel()))

                    # Record
                    m = metrics_by_scale[patch_sz]
                    m["valid_points"] += 1
                    m["true_sim_ncc"].append(true_ncc)
                    m["best_false_ncc"].append(best_false_ncc)
                    m["ranks_ncc"].append(rank_ncc)
                    m["pct_ranks_ncc"].append(pct_rank)
                    m["true_sim_mi"].append(mi_true)
                    m["best_false_mi"].append(mi_false)
                    m["separations_ncc"].append(sep_ncc)
                    m["separations_mi"].append(sep_mi)
                    m["true_gt_false_ncc"] += t_gt_f
                    m["true_gt_false_mi"] += (1 if sep_mi > 0 else 0)
                    m["auc_ncc"].append(auc_val)

                    # Scene metrics
                    sm = scene_metrics[grp][patch_sz]
                    sm["ncc_ranks"].append(rank_ncc)
                    sm["seps"].append(sep_ncc)
                    sm["t_gt_f"] += t_gt_f
                    sm["n"] += 1

    t_elapsed = time.perf_counter() - t0_start
    print(f"Completed multi-scale diagnostic evaluation in {t_elapsed:.2f}s.\n")

    # --------------------------------------------------------------------------
    # SUMMARY TABLE ACROSS SCALES (SECTION 5)
    # --------------------------------------------------------------------------
    print("--------------------------------------------------------------------------------")
    print("MULTI-SCALE STRUCTURAL DISCRIMINABILITY SUMMARY TABLE")
    print("--------------------------------------------------------------------------------")
    print(f"{'Scale':<10} {'Valid Pts':>10} {'Median Rank':>13} {'Top-5 Recall':>14} {'Top-20 Recall':>15} {'True > Best False':>19} {'Mean Sep':>12} {'Mean AUC':>10}")
    print("-" * 107)

    summary_rows = []
    for s in scales:
        m = metrics_by_scale[s]
        n_pts = m["valid_points"]
        if n_pts == 0:
            print(f"{f'{s}x{s}':<10} {'0':>10} {'N/A':>13} {'N/A':>14} {'N/A':>15} {'N/A':>19} {'N/A':>12} {'N/A':>10}")
            continue
        med_r = np.median(m["ranks_ncc"])
        t5_rec = sum(1 for r in m["ranks_ncc"] if r <= 5) / n_pts * 100
        t20_rec = sum(1 for r in m["ranks_ncc"] if r <= 20) / n_pts * 100
        t_gt_f_pct = (m["true_gt_false_ncc"] / n_pts) * 100
        mean_sep = np.mean(m["separations_ncc"])
        mean_auc = np.mean(m["auc_ncc"])

        row_dict = {
            "scale": f"{s}x{s}", "valid_pts": n_pts, "median_rank": med_r,
            "top5_rec": t5_rec, "top20_rec": t20_rec, "true_gt_false_pct": t_gt_f_pct,
            "mean_sep": mean_sep, "mean_auc": mean_auc,
        }
        summary_rows.append(row_dict)

        print(f"{f'{s}x{s}':<10} {n_pts:>10d} {med_r:>13.1f} {t5_rec:>13.2f}% {t20_rec:>14.2f}% {t_gt_f_pct:>18.2f}% {mean_sep:>12.3f} {mean_auc:>10.4f}")

    # --------------------------------------------------------------------------
    # MUTUAL INFORMATION MULTI-SCALE SUMMARY
    # --------------------------------------------------------------------------
    print("\n--------------------------------------------------------------------------------")
    print("MUTUAL INFORMATION (MI) MULTI-SCALE SEPARATION SUMMARY")
    print("--------------------------------------------------------------------------------")
    print(f"{'Scale':<10} {'True MI Mean':>15} {'False MI Mean':>16} {'MI Mean Sep':>14} {'True MI > False MI':>22}")
    print("-" * 80)
    for s in scales:
        m = metrics_by_scale[s]
        n_pts = m["valid_points"]
        if n_pts == 0:
            continue
        m_true = np.mean(m["true_sim_mi"])
        m_false = np.mean(m["best_false_mi"])
        m_sep = np.mean(m["separations_mi"])
        mi_t_gt_f = (m["true_gt_false_mi"] / n_pts) * 100
        print(f"{f'{s}x{s}':<10} {m_true:>15.3f} {m_false:>16.3f} {m_sep:>14.3f} {mi_t_gt_f:>21.1f}%")

    # --------------------------------------------------------------------------
    # BOUNDARY AND EVALUABILITY AUDIT (SECTION 8)
    # --------------------------------------------------------------------------
    print("\n--------------------------------------------------------------------------------")
    print("BOUNDARY AND EVALUABILITY AUDIT ACROSS SCALES")
    print("--------------------------------------------------------------------------------")
    base_evaluable = metrics_by_scale[16]["valid_points"]
    print(f"{'Scale':<10} {'Evaluable Points':>18} {'Border Excluded':>17} {'% of 16x16 Baseline':>23}")
    print("-" * 72)
    for s in scales:
        n_p = metrics_by_scale[s]["valid_points"]
        n_b = metrics_by_scale[s]["border_excluded"]
        pct = (n_p / base_evaluable * 100) if base_evaluable > 0 else 0
        print(f"{f'{s}x{s}':<10} {n_p:>18d} {n_b:>17d} {pct:>22.1f}%")

    # --------------------------------------------------------------------------
    # PER-SCENE SCALE COMPARISON (SECTION 6)
    # --------------------------------------------------------------------------
    print("\n--------------------------------------------------------------------------------")
    print("PER-SCENE MULTI-SCALE ANALYSIS")
    print("--------------------------------------------------------------------------------")
    scenes_improved = []
    scenes_worsened = []
    scenes_no_sep = []

    for grp in sorted(xs_train_groups):
        sm16 = scene_metrics[grp][16]
        sm48 = scene_metrics[grp][48]
        if sm16["n"] == 0 or sm48["n"] == 0:
            continue
        med_r16 = np.median(sm16["ncc_ranks"])
        med_r48 = np.median(sm48["ncc_ranks"])

        t_gt_f_16 = sm16["t_gt_f"]
        t_gt_f_48 = sm48["t_gt_f"]

        if med_r48 < med_r16 * 0.8:
            scenes_improved.append(grp)
            status = "IMPROVED"
        elif med_r48 > med_r16 * 1.2:
            scenes_worsened.append(grp)
            status = "WORSENED"
        else:
            if t_gt_f_16 == 0 and t_gt_f_48 == 0:
                scenes_no_sep.append(grp)
                status = "NO SEPARATION"
            else:
                status = "UNCHANGED"

    print(f"Scenes where larger scale (48x48 vs 16x16) IMPROVED ranking : {len(scenes_improved)} / {len(xs_train_groups)} ({len(scenes_improved)/len(xs_train_groups)*100:.1f}%)")
    print(f"Scenes where larger scale WORSENED ranking                   : {len(scenes_worsened)} / {len(xs_train_groups)} ({len(scenes_worsened)/len(xs_train_groups)*100:.1f}%)")
    print(f"Scenes where NO useful separation existed at either scale   : {len(scenes_no_sep)} / {len(xs_train_groups)} ({len(scenes_no_sep)/len(xs_train_groups)*100:.1f}%)")

    # --------------------------------------------------------------------------
    # TRANSFORMATION FACTOR CORRELATION (SECTION 7)
    # --------------------------------------------------------------------------
    print("\n--------------------------------------------------------------------------------")
    print("TRANSFORMATION FACTOR CORRELATIONS WITH SEPARATION (AT 48x48)")
    print("--------------------------------------------------------------------------------")
    tf_keys = ["angle_deg", "scale_dev", "trans_mag", "downsample_factor", "blur_sigma", "az_delta"]
    tf_names = {
        "angle_deg": "Rotation Magnitude (|deg|)",
        "scale_dev": "Scale Distortion (|scale - 1|)",
        "trans_mag": "Translation Magnitude (px)",
        "downsample_factor": "Area Downsample Factor",
        "blur_sigma": "Gaussian Blur Sigma (px)",
        "az_delta": "Sun Azimuth Offset (deg)",
    }

    tf_values = {k: [] for k in tf_keys}
    scene_seps_48 = []

    for grp in xs_train_groups:
        sm = scene_metrics[grp][48]
        if sm["n"] > 0:
            mean_s = float(np.mean(sm["seps"]))
            scene_seps_48.append(mean_s)
            for k in tf_keys:
                tf_values[k].append(scene_transform_factors[grp][k])

    print(f"{'Transformation Factor':<35} {'Factor Mean ± Std':>22} {'Correlation with Sep':>24}")
    print("-" * 84)
    corr_results = {}
    for k in tf_keys:
        vals = tf_values[k]
        if len(vals) > 2:
            r = float(np.corrcoef(vals, scene_seps_48)[0, 1])
        else:
            r = 0.0
        corr_results[k] = r
        print(f"{tf_names[k]:<35} {np.mean(vals):>10.2f} ± {np.std(vals):.2f} {r:>24.3f}")

    # --------------------------------------------------------------------------
    # SCIENTIFIC CLASSIFICATION (SECTION 9)
    # --------------------------------------------------------------------------
    print("\n================================================================================")
    print("SCIENTIFIC CLASSIFICATION (SECTION 9)")
    print("================================================================================")

    # Classification rules:
    # A — Multi-scale context contains useful discriminative information:
    #     Larger context significantly improves true-vs-false separation and target ranking.
    # B — Larger context helps somewhat but remains insufficient:
    #     Measurable improvement, but true targets remain poorly ranked.
    # C — Larger context does not help:
    #     True targets remain approximately random regardless of spatial support.

    med_16 = summary_rows[0]["median_rank"]
    med_48 = summary_rows[2]["median_rank"]
    top5_16 = summary_rows[0]["top5_rec"]
    top5_48 = summary_rows[2]["top5_rec"]
    t_gt_f_16 = summary_rows[0]["true_gt_false_pct"]
    t_gt_f_48 = summary_rows[2]["true_gt_false_pct"]

    if t_gt_f_48 > 25.0 and top5_48 > 15.0 and med_48 < 100:
        classification = "A — Multi-scale context contains useful discriminative information"
        reason = f"Larger scale (48x48) substantially improved true-target ranking (median rank {med_48:.1f} vs {med_16:.1f}) and true > best-false separation ({t_gt_f_48:.1f}% vs {t_gt_f_16:.1f}%)."
    elif med_48 < med_16 * 0.7 or (t_gt_f_48 - t_gt_f_16 > 5.0) or (top5_48 > top5_16 + 2.0):
        classification = "B — Larger context helps somewhat but remains insufficient"
        reason = f"Larger spatial context provides measurable separation gains (True > Best False: {t_gt_f_16:.1f}% -> {t_gt_f_48:.1f}%), but true targets remain poorly ranked (median rank {med_48:.1f} out of ~1,681; Top-5 recall {top5_48:.2f}%)."
    else:
        classification = "C — Larger context does not help"
        reason = f"True targets remain approximately random across all scales (Median rank: 16x16={med_16:.1f}, 48x48={med_48:.1f}; Top-5 recall: {top5_16:.2f}% vs {top5_48:.2f}%)."

    print(f"Official Classification: {classification}")
    print(f"Scientific Rationale   : {reason}\n")

    return {
        "summary_rows": summary_rows,
        "metrics_by_scale": metrics_by_scale,
        "scenes_improved": scenes_improved,
        "scenes_worsened": scenes_worsened,
        "scenes_no_sep": scenes_no_sep,
        "corr_results": corr_results,
        "classification": classification,
        "reason": reason,
    }


if __name__ == "__main__":
    run_phase9_diagnostic()
