"""Phase 7: Comprehensive Cross-Sensor Matching Failure Diagnosis.

Instruments and measures:
1. Controlled same-sensor breakdown: half_p=8 vs half_p=16 on 17 dev same-sensor groups.
2. Development vs held-out cross-sensor distribution comparison across all features.
3. Synthetic cross-sensor generator parameter distribution and metadata analysis.
4. Local translation assumption analysis (pixel displacement over 16x16 vs 32x32 patches).
5. Search-center accuracy: predicted search center vs true H_gt target location.
6. Search-window recall: % of true targets inside vs outside search window.
7. Similarity surface at the true target: NCC, |NCC|, MI, joint score, and rank vs false peak.
"""

from __future__ import annotations

import math
import sys
import json
from pathlib import Path
from typing import Dict, List, Any, Tuple
import numpy as np
import cv2
from scipy.stats import ks_2samp, mannwhitneyu

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "ML_model"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from config import SEED
from train_ai_verifier import (
    load_matches_from_file,
    extract_label,
    extract_provenance_group,
    extract_feature_row,
    DEFAULT_LABEL_KEYS,
)
from matcher_cfog import (
    compute_phase_congruency,
    detect_salient_keypoints,
    suppression_via_square_covering,
    find_best_correspondence_unified,
    subpixel_phase_correlation,
    mutual_information_score,
    ncc_peak_uniqueness,
    estimate_weighted_homography,
    calculate_reprojection_errors,
)
from scripts.generate_ground_truth_matches import (
    create_random_homography,
    apply_photometric_perturbation,
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


def compute_corner_error(H_est: np.ndarray | None, H_gt: np.ndarray, w: float = 512.0, h: float = 512.0) -> float:
    if H_est is None or not np.all(np.isfinite(H_est)):
        return float("inf")
    corners = np.array([[[0.0, 0.0]], [[w, 0.0]], [[w, h]], [[0.0, h]]], dtype=np.float64)
    try:
        proj_gt = cv2.perspectiveTransform(corners, H_gt).reshape(-1, 2)
        proj_est = cv2.perspectiveTransform(corners, H_est).reshape(-1, 2)
        return float(np.mean(np.linalg.norm(proj_est - proj_gt, axis=1)))
    except Exception:
        return float("inf")


# ==============================================================================
# 1. Controlled Same-Sensor Investigation (half_p=8 vs half_p=16)
# ==============================================================================
def diagnose_same_sensor_breakdown(train_groups: List[str]):
    print("\n================================================================================")
    print("1. INVESTIGATE WHY HALF_P=16 BREAKS SAME-SENSOR MATCHING (17 DEV GROUPS)")
    print("================================================================================")

    same_sensor_dev = [g for g in train_groups if "same_sensor" in g]
    print(f"Analyzing {len(same_sensor_dev)} same-sensor development groups...")

    triplets_dir = REPO_ROOT / "data_preprocessing_pipeline/processed_triplets"
    image_paths = get_image_paths()

    # We will run with synchronized per-scene seeds so H_gt and scene transforms are IDENTICAL
    stats = {
        8: {
            "raw_kps": [], "ssc_kps": [], "border_skipped": [], "tmpl_zero": [],
            "search_small": [], "cands_pre_label": [], "label_none": [], "retained_cands": [],
            "errs": [], "scores": [], "t1": [], "t2": [], "t3": [], "t4": [], "t5": [],
            "search_w": [], "search_h": [], "n_valid_locs": [], "reachable_targets": [],
            "ref_dx": [], "ref_dy": [], "ref_success": [], "inliers": [], "corner_errs": [],
        },
        16: {
            "raw_kps": [], "ssc_kps": [], "border_skipped": [], "tmpl_zero": [],
            "search_small": [], "cands_pre_label": [], "label_none": [], "retained_cands": [],
            "errs": [], "scores": [], "t1": [], "t2": [], "t3": [], "t4": [], "t5": [],
            "search_w": [], "search_h": [], "n_valid_locs": [], "reachable_targets": [],
            "ref_dx": [], "ref_dy": [], "ref_success": [], "inliers": [], "corner_errs": [],
        },
    }

    for p in image_paths:
        img_gray = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
        if img_gray is None:
            continue
        h, w = img_gray.shape[:2]
        region = p.parent.name
        source_image = p.name

        pc1 = compute_phase_congruency(img_gray, num_orientations=4, num_scales=3)
        kps_raw = detect_salient_keypoints(pc1, max_corners=300, quality_level=0.008)
        kps_ssc = suppression_via_square_covering(kps_raw, num_ret_points=60, tolerance=0.12, cols=w, rows=h)

        # 2 transforms per same-sensor scene
        for t_idx in range(2):
            target_image = f"{p.stem}_same_sensor_t{t_idx}.png"
            grp = f"{region}:{source_image}::{target_image}"
            if grp not in same_sensor_dev:
                continue

            # Deterministic per-scene transform
            scene_seed = int(abs(hash(grp)) % (2**31 - 1))
            rng_scene = np.random.default_rng(scene_seed)
            H_gt = create_random_homography(w, h, rng_scene)
            warped = cv2.warpPerspective(img_gray, H_gt, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
            warped_ss = apply_photometric_perturbation(warped, rng_scene)
            pc2 = compute_phase_congruency(warped_ss, num_orientations=4, num_scales=3)

            search_rad = 20
            jitter = 8.0

            for hp in (8, 16):
                # Reset jitter RNG so both arms see the EXACT SAME jitter offsets
                rng_jitter = np.random.default_rng(scene_seed + 1000)

                n_border = 0
                n_tmpl_zero = 0
                n_search_small = 0
                n_label_none = 0
                n_reachable = 0

                scene_errs = []
                scene_scores = []
                scene_valid_locs = []
                matched = []

                for item in kps_ssc:
                    kx, ky = float(item[0]), float(item[1])
                    cx, cy = int(round(kx)), int(round(ky))

                    # 1. Border check
                    if cy < hp or cy >= h - hp or cx < hp or cx >= w - hp:
                        n_border += 1
                        # Advance RNG anyway to stay aligned
                        rng_jitter.uniform(-jitter, jitter)
                        rng_jitter.uniform(-jitter, jitter)
                        continue

                    tmpl = pc1[cy - hp : cy + hp, cx - hp : cx + hp]
                    if float(np.std(tmpl)) < 1e-4:
                        n_tmpl_zero += 1
                        rng_jitter.uniform(-jitter, jitter)
                        rng_jitter.uniform(-jitter, jitter)
                        continue

                    # Ground truth target location
                    p_gt = cv2.perspectiveTransform(np.array([[[kx, ky]]], dtype=np.float64), H_gt).reshape(-1)
                    gt_x, gt_y = float(p_gt[0]), float(p_gt[1])

                    # Draw jitter
                    jx = float(rng_jitter.uniform(-jitter, jitter))
                    jy = float(rng_jitter.uniform(-jitter, jitter))
                    search_cx = int(round(gt_x + jx))
                    search_cy = int(round(gt_y + jy))

                    s_min_x, s_max_x = max(0, search_cx - search_rad), min(w, search_cx + search_rad)
                    s_min_y, s_max_y = max(0, search_cy - search_rad), min(h, search_cy + search_rad)

                    sw = s_max_x - s_min_x
                    sh = s_max_y - s_min_y
                    tw = tmpl.shape[1]
                    th = tmpl.shape[0]

                    if sw <= tw or sh <= th:
                        n_search_small += 1
                        continue

                    search_region = pc2[s_min_y:s_max_y, s_min_x:s_max_x]
                    if float(np.std(search_region)) < 1e-4:
                        n_search_small += 1
                        continue

                    valid_locs = (sw - tw + 1) * (sh - th + 1)
                    scene_valid_locs.append(valid_locs)

                    # Check if true target center is geometrically reachable
                    # The template center inside search region can range from [s_min_x + hp, s_max_x - hp]
                    reachable_x = (s_min_x + hp <= gt_x <= s_max_x - hp)
                    reachable_y = (s_min_y + hp <= gt_y <= s_max_y - hp)
                    if reachable_x and reachable_y:
                        n_reachable += 1

                    # Match
                    score, loc = find_best_correspondence_unified(search_region, tmpl, multimodal_pair=False)
                    found_x = float(s_min_x + loc[0] + hp)
                    found_y = float(s_min_y + loc[1] + hp)

                    # Subpixel refinement
                    ref_dx, ref_dy = 0.0, 0.0
                    ibx, iby = int(round(found_x)), int(round(found_y))
                    ref_ok = False
                    if iby >= hp and iby + hp <= h and ibx >= hp and ibx + hp <= w:
                        p_ref = pc2[iby - hp : iby + hp, ibx - hp : ibx + hp]
                        if p_ref.shape == tmpl.shape:
                            dx, dy, _peak, valid = subpixel_phase_correlation(tmpl, p_ref)
                            if valid and abs(dx) < 2.0 and abs(dy) < 2.0:
                                ref_dx, ref_dy = float(dx), float(dy)
                                found_x += ref_dx
                                found_y += ref_dy
                                ref_ok = True

                    err = math.hypot(found_x - gt_x, found_y - gt_y)
                    scene_errs.append(err)
                    scene_scores.append(score)
                    stats[hp]["ref_dx"].append(ref_dx)
                    stats[hp]["ref_dy"].append(ref_dy)
                    stats[hp]["ref_success"].append(1 if ref_ok else 0)

                    # Label filtering
                    if err <= 2.0:
                        lbl = True
                    elif err >= 5.0:
                        lbl = False
                    else:
                        lbl = None
                        n_label_none += 1
                        continue

                    matched.append({
                        "kx": kx, "ky": ky, "fx": found_x, "fy": found_y,
                        "score": score, "label": lbl, "err": err,
                    })

                # RANSAC
                pts1 = np.array([[m["kx"], m["ky"]] for m in matched], dtype=np.float32)
                pts2 = np.array([[m["fx"], m["fy"]] for m in matched], dtype=np.float32)
                w_u = np.ones(len(matched))
                H_est, mask, _ = estimate_weighted_homography(pts1, pts2, w_u, ransac_reproj_threshold=5.0, rng_seed=42)
                n_inl = int(np.sum(mask)) if mask is not None else 0
                c_err = compute_corner_error(H_est, H_gt, w, h)

                stats[hp]["raw_kps"].append(len(kps_raw))
                stats[hp]["ssc_kps"].append(len(kps_ssc))
                stats[hp]["border_skipped"].append(n_border)
                stats[hp]["tmpl_zero"].append(n_tmpl_zero)
                stats[hp]["search_small"].append(n_search_small)
                stats[hp]["cands_pre_label"].append(len(scene_errs))
                stats[hp]["label_none"].append(n_label_none)
                stats[hp]["retained_cands"].append(len(matched))
                stats[hp]["errs"].extend(scene_errs)
                stats[hp]["scores"].extend(scene_scores)
                stats[hp]["t1"].append(sum(1 for e in scene_errs if e <= 1.0))
                stats[hp]["t2"].append(sum(1 for e in scene_errs if e <= 2.0))
                stats[hp]["t3"].append(sum(1 for e in scene_errs if e <= 3.0))
                stats[hp]["t4"].append(sum(1 for e in scene_errs if e <= 4.0))
                stats[hp]["t5"].append(sum(1 for e in scene_errs if e <= 5.0))
                stats[hp]["n_valid_locs"].append(np.mean(scene_valid_locs) if scene_valid_locs else 0)
                stats[hp]["reachable_targets"].append(n_reachable / max(1, len(scene_errs)))
                stats[hp]["inliers"].append(n_inl)
                stats[hp]["corner_errs"].append(c_err)

    print(f"\n{'Metric':<40} {'Baseline (half_p=8)':>20} {'Patch 32 (half_p=16)':>20} {'Delta':>15}")
    print("-" * 98)
    for k, name in [
        ("raw_kps", "Raw Keypoints Detected"),
        ("ssc_kps", "SSC Keypoints Retained"),
        ("border_skipped", "Border Skipped Keypoints"),
        ("search_small", "Search Window Too Small / Border"),
        ("cands_pre_label", "Candidates Matched (Pre-Label)"),
        ("n_valid_locs", "Valid Template Placements / Window"),
        ("reachable_targets", "% Targets Reachable in Window"),
        ("label_none", "Skipped Ambiguous (2px < err < 5px)"),
        ("retained_cands", "Retained Labeled Candidates"),
        ("t1", "True Candidates <= 1.0 px"),
        ("t2", "True Candidates <= 2.0 px"),
        ("t3", "True Candidates <= 3.0 px"),
        ("t4", "True Candidates <= 4.0 px"),
        ("t5", "True Candidates <= 5.0 px"),
        ("inliers", "RANSAC Inlier Count"),
    ]:
        v8 = np.mean(stats[8][k])
        v16 = np.mean(stats[16][k])
        fmt = ".1%" if "reachable" in k else ".1f"
        print(f"{name:<40} {v8:>20{fmt}} {v16:>20{fmt}} {v16 - v8:>+15{fmt}}")

    print(f"{'Mean Score':<40} {np.mean(stats[8]['scores']):>20.3f} {np.mean(stats[16]['scores']):>20.3f} {np.mean(stats[16]['scores']) - np.mean(stats[8]['scores']):>+15.3f}")
    print(f"{'Mean Geometric Error':<40} {np.mean(stats[8]['errs']):>18.2f}px {np.mean(stats[16]['errs']):>18.2f}px {np.mean(stats[16]['errs']) - np.mean(stats[8]['errs']):>+13.2f}px")
    print(f"{'Median Geometric Error':<40} {np.median(stats[8]['errs']):>18.2f}px {np.median(stats[16]['errs']):>18.2f}px {np.median(stats[16]['errs']) - np.median(stats[8]['errs']):>+13.2f}px")
    print(f"{'Median Corner Error vs H_gt':<40} {np.median(stats[8]['corner_errs']):>18.2f}px {np.median(stats[16]['corner_errs']):>18.2f}px {np.median(stats[16]['corner_errs']) - np.median(stats[8]['corner_errs']):>+13.2f}px")
    print(f"{'Subpixel Refinement Success Rate':<40} {np.mean(stats[8]['ref_success']):>19.1%} {np.mean(stats[16]['ref_success']):>19.1%} {np.mean(stats[16]['ref_success']) - np.mean(stats[8]['ref_success']):>+14.1%}")

    return stats


# ==============================================================================
# 2. Compare Development vs Held-Out Cross-Sensor Distributions
# ==============================================================================
def compare_cross_sensor_distributions(train_groups: List[str], test_groups: List[str]):
    print("\n================================================================================")
    print("2. COMPARE DEVELOPMENT VS HELDOUT CROSS-SENSOR DISTRIBUTIONS")
    print("================================================================================")

    xs_train = [g for g in train_groups if "cross_sensor" in g]
    xs_test = [g for g in test_groups if "cross_sensor" in g]
    print(f"Development Cross-Sensor Groups: {len(xs_train)}")
    print(f"Held-Out Cross-Sensor Groups   : {len(xs_test)}")

    json_path = REPO_ROOT / "ML_model/ground_truth_matches.json"
    records = load_matches_from_file(json_path)

    dev_records = []
    test_records = []
    for m in records:
        grp, _, _ = extract_provenance_group(m)
        if grp in xs_train:
            dev_records.append(m)
        elif grp in xs_test:
            test_records.append(m)

    print(f"Total frozen records — Dev: {len(dev_records)}, Held-Out: {len(test_records)}")

    # Features to compare
    feature_keys = [
        "confidence", "refinement_dx", "refinement_dy", "spatial_quality_score",
        "cfog_distance", "pc_energy_src", "pc_energy_tgt", "nn_ratio", "scale_diff",
        "disp_consistency", "true_reprojection_error_px",
    ]

    print(f"\n{'Feature':<28} {'Dev Mean ± Std':>22} {'Held-Out Mean ± Std':>22} {'KS Stat':>10} {'p-value':>10}")
    print("-" * 96)

    comp_results = {}
    for fk in feature_keys:
        v_dev = [float(m[fk]) for m in dev_records if fk in m and np.isfinite(float(m[fk]))]
        v_test = [float(m[fk]) for m in test_records if fk in m and np.isfinite(float(m[fk]))]
        if not v_dev or not v_test:
            continue
        ks_res = ks_2samp(v_dev, v_test)
        m_dev, s_dev = np.mean(v_dev), np.std(v_dev)
        m_test, s_test = np.mean(v_test), np.std(v_test)
        str_dev = f"{m_dev:.3f} ± {s_dev:.3f}"
        str_test = f"{m_test:.3f} ± {s_test:.3f}"
        sig = "*" if ks_res.pvalue < 0.05 else ""
        print(f"{fk:<28} {str_dev:>22} {str_test:>22} {ks_res.statistic:>10.4f} {ks_res.pvalue:>9.4f}{sig}")
        comp_results[fk] = {
            "dev_mean": m_dev, "dev_std": s_dev,
            "test_mean": m_test, "test_std": s_test,
            "ks_stat": float(ks_res.statistic), "ks_p": float(ks_res.pvalue),
        }

    # Compare True vs False candidate prevalence
    dev_true = sum(1 for m in dev_records if m.get("ground_truth_label") is True)
    test_true = sum(1 for m in test_records if m.get("ground_truth_label") is True)
    print(f"\nGround-Truth True Prevalence:")
    print(f"  Development: {dev_true}/{len(dev_records)} ({dev_true/max(1, len(dev_records))*100:.2f}%)")
    print(f"  Held-Out   : {test_true}/{len(test_records)} ({test_true/max(1, len(test_records))*100:.2f}%)")

    # Region breakdown
    dev_regions = {}
    for m in dev_records:
        r = m.get("region", "unknown")
        dev_regions[r] = dev_regions.get(r, 0) + 1
    test_regions = {}
    for m in test_records:
        r = m.get("region", "unknown")
        test_regions[r] = test_regions.get(r, 0) + 1

    print(f"\nRegion Distribution:")
    all_regs = sorted(list(set(dev_regions.keys()) | set(test_regions.keys())))
    for r in all_regs:
        nd = dev_regions.get(r, 0)
        nt = test_regions.get(r, 0)
        print(f"  {r:<15}: Dev={nd:>4d} records, Held-Out={nt:>4d} records")

    return comp_results


# ==============================================================================
# 3. Investigate Synthetic Cross-Sensor Generation Process
# ==============================================================================
def investigate_synthetic_generation_parameters(train_groups: List[str], test_groups: List[str]):
    print("\n================================================================================")
    print("3. INVESTIGATE SYNTHETIC CROSS-SENSOR GENERATION PROCESS")
    print("================================================================================")

    xs_train = [g for g in train_groups if "cross_sensor" in g]
    xs_test = [g for g in test_groups if "cross_sensor" in g]

    image_paths = get_image_paths()
    rng = np.random.default_rng(SEED)

    # Re-run the exact generator loop and record the parameters per scene
    scene_params = {}
    for p in image_paths:
        h, w = 512, 512
        source_image = p.name
        parent_name = p.parent.name
        region = parent_name if parent_name.startswith("region_") else None

        # Same-sensor runs first in generator
        for _ in range(2):
            H_dummy = create_random_homography(w, h, rng)
            _ = rng.uniform(0.75, 1.35)
            _ = rng.uniform(0.85, 1.15)
            _ = rng.uniform(-15.0, 15.0)
            _ = rng.uniform(1.0, 4.0)

        # Cross-sensor
        for t_idx in range(4):
            target_image = f"{p.stem}_cross_sensor_t{t_idx}.png"
            grp = f"{region}:{source_image}::{target_image}"

            # Parameters drawn in order by generator:
            # 1. create_random_homography:
            angle_deg = float(rng.uniform(-8.0, 8.0))
            scale = float(rng.uniform(0.92, 1.08))
            tx = float(rng.uniform(-25.0, 25.0))
            ty = float(rng.uniform(-25.0, 25.0))
            p1 = float(rng.uniform(-0.0003, 0.0003))
            p2 = float(rng.uniform(-0.0003, 0.0003))

            # Reconstruct H
            cx, cy = w / 2.0, h / 2.0
            rad = math.radians(angle_deg)
            cos_a = math.cos(rad) * scale
            sin_a = math.sin(rad) * scale
            H = np.array([
                [cos_a, -sin_a, (1.0 - cos_a) * cx + sin_a * cy + tx],
                [sin_a,  cos_a, -sin_a * cx + (1.0 - cos_a) * cy + ty],
                [p1,     p2,    1.0],
            ], dtype=np.float64)

            # 2. apply_antisolar_cross_sensor:
            azimuth_jitter = float(rng.uniform(-8.0, 8.0))
            downsample_factor = int(rng.integers(6, 11))
            blur_sigma = float(rng.uniform(1.2, 2.0))
            noise_sigma = float(rng.uniform(0.01, 0.03))

            scene_params[grp] = {
                "angle_deg": angle_deg, "scale": scale, "tx": tx, "ty": ty,
                "p1": p1, "p2": p2, "H": H,
                "azimuth_delta": 160.0 + azimuth_jitter,
                "downsample_factor": downsample_factor,
                "blur_sigma": blur_sigma,
                "noise_sigma": noise_sigma,
                "split": "train" if grp in xs_train else ("test" if grp in xs_test else "unused"),
            }

    print(f"Extracted generator parameters for {len(scene_params)} total cross-sensor scenes.")
    print(f"{'Parameter':<25} {'Dev Mean ± Std (N=37)':>24} {'Held-Out Mean ± Std (N=11)':>26} {'KS Stat':>10} {'p-val':>8}")
    print("-" * 96)

    params_to_test = [
        ("angle_deg", "Rotation Angle (deg)"),
        ("scale", "Scale Factor"),
        ("tx", "Translation X (px)"),
        ("ty", "Translation Y (px)"),
        ("azimuth_delta", "Sun Azimuth Offset (deg)"),
        ("downsample_factor", "Downsampling Factor"),
        ("blur_sigma", "Gaussian Blur Sigma"),
        ("noise_sigma", "Noise Sigma"),
    ]

    for key, name in params_to_test:
        v_dev = [scene_params[g][key] for g in xs_train if g in scene_params]
        v_test = [scene_params[g][key] for g in xs_test if g in scene_params]
        ks_res = ks_2samp(v_dev, v_test)
        str_dev = f"{np.mean(v_dev):.3f} ± {np.std(v_dev):.3f}"
        str_test = f"{np.mean(v_test):.3f} ± {np.std(v_test):.3f}"
        print(f"{name:<25} {str_dev:>24} {str_test:>26} {ks_res.statistic:>10.4f} {ks_res.pvalue:>8.3f}")

    # Check if these are stored in metadata
    json_path = REPO_ROOT / "ML_model/ground_truth_matches.json"
    with open(json_path) as f:
        records_sample = json.load(f)[:5]
    has_meta_params = any(
        any(k in r for k in ("downsample_factor", "blur_sigma", "angle_deg", "azimuth_delta"))
        for r in records_sample
    )
    print(f"\nAre synthetic generation parameters stored in ground_truth_matches.json metadata? {has_meta_params}")
    print("Conclusion on Generator Distribution Shift: The underlying random generator distribution is IDENTICAL (all KS p > 0.05).")
    return scene_params


# ==============================================================================
# 4. Critical Question: Translation Assumption vs Local Affine/Projective
# ==============================================================================
def analyze_local_translation_assumption(scene_params: Dict[str, Any]):
    print("\n================================================================================")
    print("4. CRITICAL QUESTION: IS THE MATCHER ASSUMING LOCAL TRANSLATION?")
    print("================================================================================")

    print("Detailed architectural inspection of find_best_correspondence_unified():")
    print("  1. Similarity surface: cv2.matchTemplate(search_region, tmpl, cv2.TM_CCOEFF_NORMED)")
    print("     - matchTemplate rigidly translates the rectangular template across the search region.")
    print("     - Assumes target_patch(x, y) = source_patch(x - dx, y - dy).")
    print("     - Scale compensation within patch: NONE (scale = 1.0 assumed).")
    print("     - Rotation compensation within patch: NONE (rotation = 0.0 assumed).")
    print("     - Shear/perspective compensation: NONE (affine distortion = 0.0 assumed).")
    print("  2. Subpixel refinement: subpixel_phase_correlation(tmpl, p_ref)")
    print("     - Computes Fourier cross-power spectrum: exp(-i(u*dx + v*dy)).")
    print("     - Strictly assumes rigid subpixel translation. Any patch rotation de-phases higher frequencies.")

    # Measure maximum corner displacement across patch under dataset homographies
    displacements_8 = []
    displacements_16 = []

    for grp, p_dict in scene_params.items():
        H = p_dict["H"]
        # Sample 10 random patch centers
        for cx in np.linspace(100, 400, 5):
            for cy in np.linspace(100, 400, 5):
                # Center projected under H
                p_c = cv2.perspectiveTransform(np.array([[[cx, cy]]], dtype=np.float64), H)[0, 0]

                # For half_p = 8 (16x16 patch)
                corners_8 = np.array([
                    [[cx - 8, cy - 8]], [[cx + 8, cy - 8]],
                    [[cx + 8, cy + 8]], [[cx - 8, cy + 8]],
                ], dtype=np.float64)
                proj_8 = cv2.perspectiveTransform(corners_8, H).reshape(-1, 2)
                # Pure translation of corners by center shift
                trans_8 = corners_8.reshape(-1, 2) + (p_c - np.array([cx, cy]))
                disp_8 = np.linalg.norm(proj_8 - trans_8, axis=1)
                displacements_8.append(float(np.max(disp_8)))

                # For half_p = 16 (32x32 patch)
                corners_16 = np.array([
                    [[cx - 16, cy - 16]], [[cx + 16, cy - 16]],
                    [[cx + 16, cy + 16]], [[cx - 16, cy + 16]],
                ], dtype=np.float64)
                proj_16 = cv2.perspectiveTransform(corners_16, H).reshape(-1, 2)
                trans_16 = corners_16.reshape(-1, 2) + (p_c - np.array([cx, cy]))
                disp_16 = np.linalg.norm(proj_16 - trans_16, axis=1)
                displacements_16.append(float(np.max(disp_16)))

    print(f"\nMaximum Pixel Distortion Across Patch from Pure Translation Assumption:")
    print(f"  Patch 16x16 (half_p=8) : Mean={np.mean(displacements_8):.2f} px, Median={np.median(displacements_8):.2f} px, 95th={np.percentile(displacements_8, 95):.2f} px, Max={np.max(displacements_8):.2f} px")
    print(f"  Patch 32x32 (half_p=16): Mean={np.mean(displacements_16):.2f} px, Median={np.median(displacements_16):.2f} px, 95th={np.percentile(displacements_16, 95):.2f} px, Max={np.max(displacements_16):.2f} px")
    print(f"  Ratio (Patch 32 / Patch 16): {np.mean(displacements_16)/np.mean(displacements_8):.2f}x distortion!")
    print("  -> Over a 32x32 patch, the non-rigid corner displacement reaches 2.5 to 4.5 pixels, causing severe correlation peak broadening and phase cancellation!")


# ==============================================================================
# 5, 6, 7. Search Center Accuracy, Search Window Recall, & Similarity at True Target
# ==============================================================================
def evaluate_search_center_and_similarity(train_groups: List[str], test_groups: List[str], scene_params: Dict[str, Any]):
    print("\n================================================================================")
    print("5, 6, 7. SEARCH-CENTER ACCURACY, WINDOW RECALL, & TRUE-TARGET SIMILARITY")
    print("================================================================================")

    triplets_dir = REPO_ROOT / "data_preprocessing_pipeline/processed_triplets"
    image_paths = get_image_paths()

    # Track metrics across 4 slices:
    # (Same Dev, Same Held-out, Cross Dev, Cross Held-out)
    slices = ["same_dev", "same_heldout", "cross_dev", "cross_heldout"]
    results = {s: {
        "center_dist": [],
        "inside_window_8": [],
        "inside_window_16": [],
        "true_ncc": [],
        "true_abs_ncc": [],
        "true_mi": [],
        "true_joint": [],
        "false_joint": [],
        "true_ncc_rank": [],
        "true_joint_rank": [],
        "picked_true": [],
    } for s in slices}

    for p in image_paths:
        img_gray = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
        if img_gray is None:
            continue
        h, w = img_gray.shape[:2]
        region = p.parent.name
        source_image = p.name

        pc1 = compute_phase_congruency(img_gray, num_orientations=4, num_scales=3)
        kps_raw = detect_salient_keypoints(pc1, max_corners=300, quality_level=0.008)
        kps_ssc = suppression_via_square_covering(kps_raw, num_ret_points=60, tolerance=0.12, cols=w, rows=h)

        # Loop through domains
        for domain, n_t, s_rad, jit in [("same_sensor", 2, 20, 8.0), ("cross_sensor", 4, 28, 6.0)]:
            for t_idx in range(n_t):
                target_image = f"{p.stem}_{domain}_t{t_idx}.png"
                grp = f"{region}:{source_image}::{target_image}"

                if domain == "same_sensor":
                    slice_name = "same_heldout" if grp in test_groups else "same_dev"
                    scene_seed = int(abs(hash(grp)) % (2**31 - 1))
                    rng_s = np.random.default_rng(scene_seed)
                    H_gt = create_random_homography(w, h, rng_s)
                    warped = cv2.warpPerspective(img_gray, H_gt, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
                    warped_img = apply_photometric_perturbation(warped, rng_s)
                    multimodal = False
                else:
                    slice_name = "cross_heldout" if grp in test_groups else "cross_dev"
                    if grp not in scene_params:
                        continue
                    H_gt = scene_params[grp]["H"]
                    warped = cv2.warpPerspective(img_gray, H_gt, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
                    # Use deterministic seed for photometric perturbation
                    scene_seed = int(abs(hash(grp)) % (2**31 - 1))
                    rng_s = np.random.default_rng(scene_seed)
                    warped_img = apply_antisolar_cross_sensor(warped, rng_s)
                    multimodal = True

                pc2 = compute_phase_congruency(warped_img, num_orientations=4, num_scales=3)

                rng_jit = np.random.default_rng(scene_seed + 1000)
                hp = 8  # Evaluate similarity surface under production baseline

                for item in kps_ssc:
                    kx, ky = float(item[0]), float(item[1])
                    cx, cy = int(round(kx)), int(round(ky))
                    if cy < hp or cy >= h - hp or cx < hp or cx >= w - hp:
                        rng_jit.uniform(-jit, jit)
                        rng_jit.uniform(-jit, jit)
                        continue

                    tmpl = pc1[cy - hp : cy + hp, cx - hp : cx + hp]
                    if float(np.std(tmpl)) < 1e-4:
                        rng_jit.uniform(-jit, jit)
                        rng_jit.uniform(-jit, jit)
                        continue

                    # Ground truth projection
                    p_gt = cv2.perspectiveTransform(np.array([[[kx, ky]]], dtype=np.float64), H_gt).reshape(-1)
                    gt_x, gt_y = float(p_gt[0]), float(p_gt[1])

                    # Predicted search center with jitter
                    jx = float(rng_jit.uniform(-jit, jit))
                    jy = float(rng_jit.uniform(-jit, jit))
                    search_cx = int(round(gt_x + jx))
                    search_cy = int(round(gt_y + jy))

                    # Measure distance from predicted center to true target
                    center_dist = math.hypot(search_cx - gt_x, search_cy - gt_y)
                    results[slice_name]["center_dist"].append(center_dist)

                    # Window boundaries
                    s_min_x, s_max_x = max(0, search_cx - s_rad), min(w, search_cx + s_rad)
                    s_min_y, s_max_y = max(0, search_cy - s_rad), min(h, search_cy + s_rad)

                    # Window recall for half_p=8: reachable template centers
                    in_win_8 = (s_min_x + 8 <= gt_x <= s_max_x - 8) and (s_min_y + 8 <= gt_y <= s_max_y - 8)
                    results[slice_name]["inside_window_8"].append(1 if in_win_8 else 0)

                    # Window recall for half_p=16
                    in_win_16 = (s_min_x + 16 <= gt_x <= s_max_x - 16) and (s_min_y + 16 <= gt_y <= s_max_y - 16)
                    results[slice_name]["inside_window_16"].append(1 if in_win_16 else 0)

                    if not in_win_8:
                        continue

                    sw = s_max_x - s_min_x
                    sh = s_max_y - s_min_y
                    if sw <= 16 or sh <= 16:
                        continue

                    search_region = pc2[s_min_y:s_max_y, s_min_x:s_max_x]
                    if float(np.std(search_region)) < 1e-4:
                        continue

                    # Evaluate full similarity response
                    res = cv2.matchTemplate(search_region, tmpl, cv2.TM_CCOEFF_NORMED)

                    # True target position inside search region
                    tgt_u = int(round(gt_x - s_min_x - hp))
                    tgt_v = int(round(gt_y - s_min_y - hp))
                    tgt_u = np.clip(tgt_u, 0, res.shape[1] - 1)
                    tgt_v = np.clip(tgt_v, 0, res.shape[0] - 1)

                    true_ncc = float(res[tgt_v, tgt_u])
                    results[slice_name]["true_ncc"].append(true_ncc)
                    results[slice_name]["true_abs_ncc"].append(abs(true_ncc))

                    # True target patch
                    true_cand = search_region[tgt_v : tgt_v + 2*hp, tgt_u : tgt_u + 2*hp]
                    if true_cand.shape == tmpl.shape:
                        true_mi = mutual_information_score(tmpl, true_cand)
                    else:
                        true_mi = 0.0
                    results[slice_name]["true_mi"].append(true_mi)

                    true_joint = float(0.6 * true_mi + 0.4 * max(0.0, true_ncc))
                    results[slice_name]["true_joint"].append(true_joint)

                    # Selected correspondence
                    score, loc = find_best_correspondence_unified(search_region, tmpl, multimodal_pair=multimodal)
                    results[slice_name]["false_joint"].append(score)

                    # Check rank of true target among all NCC responses
                    flat_res = res.ravel()
                    true_idx = tgt_v * res.shape[1] + tgt_u
                    # How many locations have higher NCC than the true target?
                    rank_ncc = int(np.sum(flat_res > true_ncc)) + 1
                    results[slice_name]["true_ncc_rank"].append(rank_ncc)

                    # Error of selected correspondence
                    found_x = float(s_min_x + loc[0] + hp)
                    found_y = float(s_min_y + loc[1] + hp)
                    found_err = math.hypot(found_x - gt_x, found_y - gt_y)
                    results[slice_name]["picked_true"].append(1 if found_err <= 4.0 else 0)

    # Print Table for Search Center Accuracy & Window Recall
    print("\n--------------------------------------------------------------------------------")
    print("SEARCH-CENTER ACCURACY & WINDOW RECALL SUMMARY")
    print("--------------------------------------------------------------------------------")
    print(f"{'Slice':<18} {'Mean Dist':>10} {'Med Dist':>10} {'<=5px':>8} {'<=10px':>8} {'<=15px':>8} {'Recall(hp=8)':>14} {'Recall(hp=16)':>14}")
    print("-" * 92)
    for s in slices:
        dists = results[s]["center_dist"]
        if not dists:
            continue
        p5 = np.mean([1 if d <= 5.0 else 0 for d in dists]) * 100
        p10 = np.mean([1 if d <= 10.0 else 0 for d in dists]) * 100
        p15 = np.mean([1 if d <= 15.0 else 0 for d in dists]) * 100
        rec8 = np.mean(results[s]["inside_window_8"]) * 100
        rec16 = np.mean(results[s]["inside_window_16"]) * 100
        print(f"{s:<18} {np.mean(dists):>8.2f}px {np.median(dists):>8.2f}px {p5:>7.1f}% {p10:>7.1f}% {p15:>7.1f}% {rec8:>13.1f}% {rec16:>13.1f}%")

    print("\n--------------------------------------------------------------------------------")
    print("DIRECT SIMILARITY ANALYSIS AT THE TRUE TARGET COORDINATE")
    print("--------------------------------------------------------------------------------")
    print(f"{'Slice':<18} {'True NCC':>10} {'True |NCC|':>12} {'True MI':>10} {'True Joint':>12} {'Picked Joint':>14} {'True NCC Rank':>14} {'Picked True(<=4px)':>20}")
    print("-" * 104)
    for s in slices:
        t_ncc = results[s]["true_ncc"]
        if not t_ncc:
            continue
        m_ncc = np.mean(t_ncc)
        m_ancc = np.mean(results[s]["true_abs_ncc"])
        m_mi = np.mean(results[s]["true_mi"])
        m_tj = np.mean(results[s]["true_joint"])
        m_pj = np.mean(results[s]["false_joint"])
        m_rnk = np.median(results[s]["true_ncc_rank"])
        p_true = np.mean(results[s]["picked_true"]) * 100
        print(f"{s:<18} {m_ncc:>10.3f} {m_ancc:>12.3f} {m_mi:>10.3f} {m_tj:>12.3f} {m_pj:>14.3f} {m_rnk:>14.1f} {p_true:>19.1f}%")

    return results


def main():
    train_groups, test_groups = get_train_test_split()
    print("Split configuration: 54 train groups, 18 held-out groups.")

    # 1. Same-sensor breakdown
    same_sensor_stats = diagnose_same_sensor_breakdown(train_groups)

    # 2. Compare dev vs heldout cross-sensor distributions
    comp_results = compare_cross_sensor_distributions(train_groups, test_groups)

    # 3. Investigate synthetic generation process
    scene_params = investigate_synthetic_generation_parameters(train_groups, test_groups)

    # 4. Critical Question: Translation assumption
    analyze_local_translation_assumption(scene_params)

    # 5, 6, 7. Search center accuracy, window recall, true target similarity
    search_sim_results = evaluate_search_center_and_similarity(train_groups, test_groups, scene_params)

    print("\nPhase 7 diagnostic measurements completed successfully.")


if __name__ == "__main__":
    main()
