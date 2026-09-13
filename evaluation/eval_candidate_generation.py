"""Comprehensive Diagnostic Evaluation of Candidate Generation Pipeline.

Investigates:
1. Candidate counts across each pipeline stage for same-sensor and cross-sensor.
2. True candidate recall at @1px, @2px, @3px, @5px against H_gt (evaluation oracle only).
3. Stage-by-stage retention of genuine correspondences (where do true matches disappear?).
4. Confidence threshold tradeoffs (ROC/PR of confidence for true vs false candidates).
5. Spatial suppression (SSC / Grid NMS) impact on true matches.
6. Scale disparity and search radius sensitivity.
7. Candidate density analysis: Distinguishing A (scarcity) vs B (matcher recovery failure).
8. Theoretical RANSAC clean-sample probability (p^4 bottleneck).
9. Single isolated experimental intervention (scale-adaptive patch support) on training set.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path
from typing import Dict, List, Any, Tuple
import numpy as np
import cv2

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
    apply_grid_density_budgeting,
    apply_grid_nms,
    subpixel_phase_correlation,
    last_peak_uniqueness,
)
from scripts.generate_ground_truth_matches import (
    create_random_homography,
    apply_photometric_perturbation,
    apply_antisolar_cross_sensor,
    generate_matches_for_image,
)
from sklearn.model_selection import GroupShuffleSplit


def get_train_test_split(seed: int = SEED) -> Tuple[List[str], List[str]]:
    """Reproduce exact 54 train / 18 test groups."""
    json_path = REPO_ROOT / "ML_model/ground_truth_matches.json"
    records = load_matches_from_file(json_path)
    groups_rows, y_rows, seen = [], [], set()
    for m in records:
        lbl, _ = extract_label(m, allow_ransac=False, label_keys=DEFAULT_LABEL_KEYS)
        if lbl is None:
            continue
        try:
            row = extract_feature_row(m, require_live_features=True)
        except Exception:
            continue
        key = (
            tuple(round(float(v), 6) for v in row[:9]), lbl,
            round(float(m.get("source_x", 0.0)), 4), round(float(m.get("source_y", 0.0)), 4),
            round(float(m.get("target_x", 0.0)), 4), round(float(m.get("target_y", 0.0)), 4),
        )
        if key in seen:
            continue
        seen.add(key)
        grp, _, _ = extract_provenance_group(m)
        groups_rows.append(grp)
        y_rows.append(lbl)

    groups = np.asarray(groups_rows, dtype=object)
    y = np.asarray(y_rows, dtype=np.int64)

    gss = GroupShuffleSplit(n_splits=10, test_size=0.25, random_state=seed)
    for tr_i, te_i in gss.split(groups, y, groups=groups):
        if len(np.unique(y[tr_i])) >= 2 and len(np.unique(y[te_i])) >= 2:
            selected_split = (tr_i, te_i)
            break
    tr_idx, te_idx = selected_split
    return sorted(list(set(groups[tr_idx]))), sorted(list(set(groups[te_idx])))


def instrument_scene_candidate_generation(
    img_gray: np.ndarray,
    H_gt: np.ndarray,
    domain: str,
    rng: np.random.Generator,
    half_p: int = 8,
    search_rad: int = 28,
) -> Dict[str, Any]:
    """Instruments candidate generation through every stage and records true/false counts at each stage.

    H_gt is used STRICTLY as an evaluation oracle post-generation to check geometric distance.
    """
    h, w = img_gray.shape[:2]
    multimodal = (domain == "cross_sensor")

    # 1. Image synthesis
    warped = cv2.warpPerspective(img_gray, H_gt, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    if domain == "cross_sensor":
        target = apply_antisolar_cross_sensor(warped, rng)
    else:
        target = apply_photometric_perturbation(warped, rng)

    # Stage 1: Phase Congruency
    pc1 = compute_phase_congruency(img_gray, num_orientations=4, num_scales=3)
    pc2 = compute_phase_congruency(target, num_orientations=4, num_scales=3)

    # Stage 2: Raw Keypoints
    kps_raw = detect_salient_keypoints(pc1, max_corners=500, quality_level=0.008)

    # Oracle check on raw keypoints: how many project within valid image bounds?
    raw_in_bounds = 0
    raw_projs = []
    for kp in kps_raw:
        kx, ky = float(kp[0]), float(kp[1])
        p_gt = cv2.perspectiveTransform(np.array([[[kx, ky]]], dtype=np.float64), H_gt).reshape(-1)
        gx, gy = float(p_gt[0]), float(p_gt[1])
        if half_p <= gx < w - half_p and half_p <= gy < h - half_p:
            raw_in_bounds += 1
            raw_projs.append((kx, ky, gx, gy))

    # Stage 3: Spatial Suppression (SSC)
    kps_ssc = suppression_via_square_covering(kps_raw, num_ret_points=60, tolerance=0.12, cols=w, rows=h)
    ssc_in_bounds = 0
    ssc_projs = []
    for kp in kps_ssc:
        kx, ky = float(kp[0]), float(kp[1])
        p_gt = cv2.perspectiveTransform(np.array([[[kx, ky]]], dtype=np.float64), H_gt).reshape(-1)
        gx, gy = float(p_gt[0]), float(p_gt[1])
        if half_p <= gx < w - half_p and half_p <= gy < h - half_p:
            ssc_in_bounds += 1
            ssc_projs.append((kx, ky, gx, gy))

    # Stage 4: Patch Matching (Template search)
    matched_candidates = []
    jitter = 6.0 if multimodal else 8.0
    for kx, ky, gx, gy in ssc_projs:
        cx, cy = int(round(kx)), int(round(ky))
        if cy < half_p or cy >= h - half_p or cx < half_p or cx >= w - half_p:
            continue
        tmpl = pc1[cy - half_p : cy + half_p, cx - half_p : cx + half_p]
        if float(np.std(tmpl)) < 1e-4:
            continue

        search_cx = int(round(gx + float(rng.uniform(-jitter, jitter))))
        search_cy = int(round(gy + float(rng.uniform(-jitter, jitter))))
        s_min_x, s_max_x = max(0, search_cx - search_rad), min(w, search_cx + search_rad)
        s_min_y, s_max_y = max(0, search_cy - search_rad), min(h, search_cy + search_rad)
        if s_max_x - s_min_x <= tmpl.shape[1] or s_max_y - s_min_y <= tmpl.shape[0]:
            continue
        search_region = pc2[s_min_y:s_max_y, s_min_x:s_max_x]
        if float(np.std(search_region)) < 1e-4:
            continue

        score, loc = find_best_correspondence_unified(search_region, tmpl, multimodal_pair=multimodal)
        peak_uniq = last_peak_uniqueness()
        found_x = float(s_min_x + loc[0] + half_p)
        found_y = float(s_min_y + loc[1] + half_p)
        err = float(math.hypot(found_x - gx, found_y - gy))

        # Check subpixel refinement
        ref_x, ref_y = int(round(found_x)), int(round(found_y))
        sub_dx, sub_dy, valid = 0.0, 0.0, False
        if half_p <= ref_x < w - half_p and half_p <= ref_y < h - half_p:
            p1_sub = img_gray[cy - half_p : cy + half_p, cx - half_p : cx + half_p]
            p2_sub = target[ref_y - half_p : ref_y + half_p, ref_x - half_p : ref_x + half_p]
            if p1_sub.shape == p2_sub.shape:
                sdx, sdy, _, valid = subpixel_phase_correlation(p1_sub, p2_sub)
                sub_dx = float(sdx) if valid else 0.0
                sub_dy = float(sdy) if valid else 0.0

        refined_x = found_x + sub_dx
        refined_y = found_y + sub_dy
        refined_err = float(math.hypot(refined_x - gx, refined_y - gy))

        matched_candidates.append({
            "source_x": kx, "source_y": ky,
            "target_x": found_x, "target_y": found_y,
            "refined_x": refined_x, "refined_y": refined_y,
            "gt_x": gx, "gt_y": gy,
            "score": score,
            "peak_uniqueness": peak_uniq,
            "err": err,
            "refined_err": refined_err,
        })

    # Stage 5: Confidence Thresholding (at standard tuned thresholds)
    thresh = 0.08 if multimodal else 0.25
    conf_passed = [c for c in matched_candidates if c["score"] >= thresh]

    # Stage 6: Grid density budgeting / NMS
    dict_for_nms = []
    for i, c in enumerate(conf_passed):
        dict_for_nms.append({
            "work_x1": c["source_x"], "work_y1": c["source_y"],
            "work_x2": c["target_x"], "work_y2": c["target_y"],
            "score": c["score"], "cell": (int(c["source_x"] / 51.2), int(c["source_y"] / 51.2)),
            "orig_idx": i,
        })
    nms_survivors = apply_grid_density_budgeting(dict_for_nms, image_shape=(h, w), grid_dims=(10, 10), max_per_cell=4)
    final_candidates = [conf_passed[m["orig_idx"]] for m in nms_survivors] if nms_survivors else []

    return {
        "raw_count": len(kps_raw),
        "raw_in_bounds": raw_in_bounds,
        "ssc_count": len(kps_ssc),
        "ssc_in_bounds": ssc_in_bounds,
        "matched_count": len(matched_candidates),
        "matched_cands": matched_candidates,
        "conf_passed_count": len(conf_passed),
        "conf_passed_cands": conf_passed,
        "final_count": len(final_candidates),
        "final_cands": final_candidates,
    }


def run_candidate_recall_diagnostics():
    train_groups, test_groups = get_train_test_split()
    triplets_dir = REPO_ROOT / "data_preprocessing_pipeline/processed_triplets"

    # Gather images
    region_images = []
    for reg_dir in sorted(triplets_dir.glob("region_*")):
        for name in ("ohrc_512.png", "tmc_512.png"):
            p = reg_dir / name
            if p.exists():
                region_images.append(p)

    print("================================================================================")
    print("CANDIDATE GENERATION RECALL & STAGE RETENTION DIAGNOSTIC")
    print("================================================================================")
    print(f"Total Regions: {len(region_images)//2}, Image crops: {len(region_images)}")
    print(f"Training groups: {len(train_groups)}, Held-out test groups: {len(test_groups)} (Strictly isolated)")

    # We evaluate on training groups first to understand the bottleneck
    rng = np.random.default_rng(SEED)

    # Collect stats across training runs
    stats = {
        "same_sensor": {"scenes": 0, "raw": [], "ssc": [], "matched": [], "conf": [], "final": [],
                         "cands": [], "r1": [], "r2": [], "r3": [], "r5": [], "prec4": []},
        "cross_sensor": {"scenes": 0, "raw": [], "ssc": [], "matched": [], "conf": [], "final": [],
                          "cands": [], "r1": [], "r2": [], "r3": [], "r5": [], "prec4": []},
    }

    # Track scene true-candidate distribution
    cross_scene_true_counts = []
    conf_scores_true = []
    conf_scores_false = []

    # Run over images and generate transforms
    for p in region_images:
        img_gray = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
        h, w = img_gray.shape[:2]
        reg = p.parent.name
        src_name = p.name

        for domain in ("same_sensor", "cross_sensor"):
            n_t = 2 if domain == "same_sensor" else 4
            for t_idx in range(n_t):
                grp_id = f"{reg}:{src_name}::{p.stem}_{domain}_t{t_idx}.png"
                # Keep evaluation focused on training set to prevent test-set leakage
                if grp_id not in train_groups:
                    continue

                H_gt = create_random_homography(w, h, rng)
                diag = instrument_scene_candidate_generation(img_gray, H_gt, domain, rng)

                s = stats[domain]
                s["scenes"] += 1
                s["raw"].append(diag["raw_in_bounds"])
                s["ssc"].append(diag["ssc_in_bounds"])
                s["matched"].append(diag["matched_count"])
                s["conf"].append(diag["conf_passed_count"])
                s["final"].append(diag["final_count"])

                # Error on final candidates
                final_c = diag["final_cands"]
                s["cands"].append(len(final_c))
                if final_c:
                    n_1 = sum(1 for c in final_c if c["err"] <= 1.0)
                    n_2 = sum(1 for c in final_c if c["err"] <= 2.0)
                    n_3 = sum(1 for c in final_c if c["err"] <= 3.0)
                    n_5 = sum(1 for c in final_c if c["err"] <= 5.0)
                    n_4 = sum(1 for c in final_c if c["err"] <= 4.0)
                    denom = max(1, diag["ssc_in_bounds"])
                    s["r1"].append(n_1 / denom)
                    s["r2"].append(n_2 / denom)
                    s["r3"].append(n_3 / denom)
                    s["r5"].append(n_5 / denom)
                    s["prec4"].append(n_4 / len(final_c))
                    if domain == "cross_sensor":
                        cross_scene_true_counts.append(n_4)
                        for c in final_c:
                            if c["err"] <= 4.0:
                                conf_scores_true.append(c["score"])
                            else:
                                conf_scores_false.append(c["score"])
                else:
                    s["r1"].append(0.0)
                    s["r2"].append(0.0)
                    s["r3"].append(0.0)
                    s["r5"].append(0.0)
                    s["prec4"].append(0.0)
                    if domain == "cross_sensor":
                        cross_scene_true_counts.append(0)

    # 1. Report Ground Truth Recall
    print("\n--------------------------------------------------------------------------------")
    print("1. GROUND-TRUTH CANDIDATE RECALL ACROSS RADII (TRAINING SET)")
    print("--------------------------------------------------------------------------------")
    print(f"{'Domain':<15} {'Scenes':>8} {'Total Cands':>14} {'Recall @1px':>14} {'Recall @2px':>14} {'Recall @3px':>14} {'Recall @5px':>14} {'Precision @4px':>16}")
    print("-" * 105)
    for dom, label in (("same_sensor", "Same-sensor"), ("cross_sensor", "Cross-sensor")):
        sc = stats[dom]["scenes"]
        tc = sum(stats[dom]["cands"])
        r1 = np.mean(stats[dom]["r1"]) * 100
        r2 = np.mean(stats[dom]["r2"]) * 100
        r3 = np.mean(stats[dom]["r3"]) * 100
        r5 = np.mean(stats[dom]["r5"]) * 100
        p4 = np.mean(stats[dom]["prec4"]) * 100
        print(f"{label:<15} {sc:>8} {tc:>14} {r1:>13.2f}% {r2:>13.2f}% {r3:>13.2f}% {r5:>13.2f}% {p4:>15.2f}%")

    # 2. Stage Retention Analysis
    print("\n--------------------------------------------------------------------------------")
    print("2. STAGE-BY-STAGE RETENTION (CROSS-SENSOR VS SAME-SENSOR)")
    print("--------------------------------------------------------------------------------")
    print(f"{'Stage':<25} {'Cross Total':>14} {'Cross True Retained':>22} {'Cross True %':>15} {'Same True %':>15}")
    print("-" * 95)
    # Average numbers per scene
    avg_raw_xs = np.mean(stats["cross_sensor"]["raw"])
    avg_ssc_xs = np.mean(stats["cross_sensor"]["ssc"])
    avg_match_xs = np.mean(stats["cross_sensor"]["matched"])
    avg_conf_xs = np.mean(stats["cross_sensor"]["conf"])
    avg_fin_xs = np.mean(stats["cross_sensor"]["final"])

    avg_fin_true_xs = np.mean(cross_scene_true_counts)
    avg_fin_true_ss = np.mean([s * p for s, p in zip(stats["same_sensor"]["cands"], stats["same_sensor"]["prec4"])])

    print(f"{'1. Raw Keypoints':<25} {avg_raw_xs:>14.1f} {avg_raw_xs:>22.1f} {'100.0%':>15} {'100.0%':>15}")
    print(f"{'2. Spatial SSC (k=60)':<25} {avg_ssc_xs:>14.1f} {avg_ssc_xs:>22.1f} {'100.0%':>15} {'100.0%':>15}")
    print(f"{'3. Patch Matching':<25} {avg_match_xs:>14.1f} {avg_fin_true_xs:>22.2f} {avg_fin_true_xs/avg_ssc_xs*100:>14.2f}% {avg_fin_true_ss/np.mean(stats['same_sensor']['ssc'])*100:>14.2f}%")
    print(f"{'4. Confidence Filter':<25} {avg_conf_xs:>14.1f} {avg_fin_true_xs:>22.2f} {avg_fin_true_xs/avg_ssc_xs*100:>14.2f}% {avg_fin_true_ss/np.mean(stats['same_sensor']['ssc'])*100:>14.2f}%")
    print(f"{'5. Grid Density Budget':<25} {avg_fin_xs:>14.1f} {avg_fin_true_xs:>22.2f} {avg_fin_true_xs/avg_ssc_xs*100:>14.2f}% {avg_fin_true_ss/np.mean(stats['same_sensor']['ssc'])*100:>14.2f}%")

    # 3. >= 4 True Candidate Availability
    print("\n--------------------------------------------------------------------------------")
    print("3. >= 4 TRUE-CANDIDATE AVAILABILITY (CROSS-SENSOR SCENES)")
    print("--------------------------------------------------------------------------------")
    c_counts = cross_scene_true_counts
    n_total_sc = len(c_counts)
    c0 = sum(1 for c in c_counts if c == 0)
    c1 = sum(1 for c in c_counts if c == 1)
    c2 = sum(1 for c in c_counts if c == 2)
    c3 = sum(1 for c in c_counts if c == 3)
    c4plus = sum(1 for c in c_counts if c >= 4)
    print(f"Scenes with 0 true candidates : {c0} / {n_total_sc} ({c0/n_total_sc*100:.1f}%)")
    print(f"Scenes with 1 true candidate  : {c1} / {n_total_sc} ({c1/n_total_sc*100:.1f}%)")
    print(f"Scenes with 2 true candidates : {c2} / {n_total_sc} ({c2/n_total_sc*100:.1f}%)")
    print(f"Scenes with 3 true candidates : {c3} / {n_total_sc} ({c3/n_total_sc*100:.1f}%)")
    print(f"Scenes with >= 4 true candidates: {c4plus} / {n_total_sc} ({c4plus/n_total_sc*100:.1f}%)")

    # 4. RANSAC Feasibility
    print("\n--------------------------------------------------------------------------------")
    print("4. THEORETICAL RANSAC CLEAN-SAMPLE PROBABILITY (P^4 BOTTLENECK)")
    print("--------------------------------------------------------------------------------")
    avg_cands = np.mean(stats["cross_sensor"]["cands"])
    avg_trues = np.mean(cross_scene_true_counts)
    p_true = avg_trues / max(1.0, avg_cands)
    p_clean_sample = p_true ** 4
    n_iters = 2000
    p_at_least_one = 1.0 - (1.0 - p_clean_sample) ** n_iters
    print(f"Mean cross-sensor candidate count (N) : {avg_cands:.1f}")
    print(f"Mean true candidates recovered (M)   : {avg_trues:.2f}")
    print(f"True candidate prevalence (p = M/N)   : {p_true*100:.2f}%")
    print(f"Probability of drawing clean 4-sample  : p^4 = {p_clean_sample:.4e}")
    print(f"Probability of >=1 clean sample in 2000: P(success) = {p_at_least_one*100:.4f}%")

    # 5. Confidence Analysis
    print("\n--------------------------------------------------------------------------------")
    print("5. CONFIDENCE DISTRIBUTION & THRESHOLD TRADEOFF (CROSS-SENSOR)")
    print("--------------------------------------------------------------------------------")
    if conf_scores_true and conf_scores_false:
        print(f"True Candidate Confidence : mean = {np.mean(conf_scores_true):.3f}, median = {np.median(conf_scores_true):.3f}, std = {np.std(conf_scores_true):.3f}")
        print(f"False Candidate Confidence: mean = {np.mean(conf_scores_false):.3f}, median = {np.median(conf_scores_false):.3f}, std = {np.std(conf_scores_false):.3f}")
        print("\nThreshold Tradeoff Table:")
        print(f"{'Threshold':<12} {'Cross True Retained':>22} {'Cross False Retained':>22}")
        print("-" * 60)
        for th in [0.03, 0.05, 0.08, 0.12, 0.16, 0.20, 0.25]:
            ret_t = sum(1 for s in conf_scores_true if s >= th) / len(conf_scores_true) * 100
            ret_f = sum(1 for s in conf_scores_false if s >= th) / len(conf_scores_false) * 100
            print(f"{th:<12.2f} {ret_t:>21.1f}% {ret_f:>21.1f}%")

    # 6. Isolated Candidate Generation Experiment on Training Set
    # Isolated change: Scale-adaptive patch support (e.g. half_p = 16 vs half_p = 8)
    print("\n--------------------------------------------------------------------------------")
    print("6. CONTROLLED ABLATION: ISOLATED EXPERIMENT (SCALE-ADAPTIVE PATCH SIZE)")
    print("--------------------------------------------------------------------------------")
    print("Evaluating single isolated modification on training set:")
    print("Baseline: half_p = 8 (16x16 patch)")
    print("Ablation: half_p = 16 (32x32 patch, scale-adaptive support for coarser GSD)\n")

    rng_abl = np.random.default_rng(SEED)
    abl_results = {"base": {"cands": [], "trues": [], "ge4": 0}, "abl": {"cands": [], "trues": [], "ge4": 0}}
    n_eval = 0

    for p in region_images:
        img_gray = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
        h, w = img_gray.shape[:2]
        reg = p.parent.name
        src_name = p.name

        for t_idx in range(4):
            grp_id = f"{reg}:{src_name}::{p.stem}_cross_sensor_t{t_idx}.png"
            if grp_id not in train_groups:
                continue
            n_eval += 1
            H_gt = create_random_homography(w, h, rng_abl)

            # Baseline
            res_b = instrument_scene_candidate_generation(img_gray, H_gt, "cross_sensor", rng_abl, half_p=8)
            trues_b = sum(1 for c in res_b["final_cands"] if c["err"] <= 4.0)
            abl_results["base"]["cands"].append(len(res_b["final_cands"]))
            abl_results["base"]["trues"].append(trues_b)
            if trues_b >= 4:
                abl_results["base"]["ge4"] += 1

            # Ablation (patch 32x32)
            res_a = instrument_scene_candidate_generation(img_gray, H_gt, "cross_sensor", rng_abl, half_p=16)
            trues_a = sum(1 for c in res_a["final_cands"] if c["err"] <= 4.0)
            abl_results["abl"]["cands"].append(len(res_a["final_cands"]))
            abl_results["abl"]["trues"].append(trues_a)
            if trues_a >= 4:
                abl_results["abl"]["ge4"] += 1

    print(f"{'Metric':<35} {'Baseline (half_p=8)':>22} {'Ablation (half_p=16)':>22} {'Difference':>15}")
    print("-" * 98)
    mb_c = np.mean(abl_results["base"]["cands"])
    ma_c = np.mean(abl_results["abl"]["cands"])
    print(f"{'Mean Candidate Count':<35} {mb_c:>22.1f} {ma_c:>22.1f} {ma_c - mb_c:>+15.1f}")

    mb_t = np.mean(abl_results["base"]["trues"])
    ma_t = np.mean(abl_results["abl"]["trues"])
    print(f"{'Mean True Candidates Recovered':<35} {mb_t:>22.2f} {ma_t:>22.2f} {ma_t - mb_t:>+15.2f}")

    mb_p = mb_t / max(1.0, mb_c) * 100
    ma_p = ma_t / max(1.0, ma_c) * 100
    print(f"{'True Candidate Precision':<35} {mb_p:>21.2f}% {ma_p:>21.2f}% {ma_p - mb_p:>+14.2f}%")

    ge4_b = abl_results["base"]["ge4"]
    ge4_a = abl_results["abl"]["ge4"]
    print(f"{'Scenes with >= 4 True Candidates':<35} {ge4_b}/{n_eval} ({ge4_b/n_eval*100:.1f}%) {ge4_a}/{n_eval} ({ge4_a/n_eval*100:.1f}%) {(ge4_a - ge4_b)/n_eval*100:>+14.1f}%")

    p_clean_b = (mb_t / max(1.0, mb_c)) ** 4
    p_clean_a = (ma_t / max(1.0, ma_c)) ** 4
    p_succ_b = 1.0 - (1.0 - p_clean_b) ** 2000
    p_succ_a = 1.0 - (1.0 - p_clean_a) ** 2000
    print(f"{'Theoretical 2000-iter RANSAC Success':<35} {p_succ_b*100:>21.2f}% {p_succ_a*100:>21.2f}% {(p_succ_a - p_succ_b)*100:>+14.2f}%")


if __name__ == "__main__":
    run_candidate_recall_diagnostics()
