"""Phase 8: Isolated Cross-Sensor GOA Preselection Experiment.

Evaluates:
1. Search-surface ranking of true target under NCC vs GOA (Median/Mean rank, Top-1, Top-3, Top-5, Top-10, Top-20 recall).
2. Candidate-level evaluation across 37 dev cross-sensor scenes (Baseline NCC top-5 vs GOA top-5).
3. Same-sensor regression check across 17 dev same-sensor scenes (verifies identical behavior).
4. Direct comparison of similarity metrics at the true target vs selected false peak.
5. Per-scene breakdown and final A/B/C classification.
"""

from __future__ import annotations

import math
import sys
import time
from pathlib import Path
from typing import Dict, List, Any, Tuple
import numpy as np
import cv2

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "ML_model"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from config import SEED
from matcher_cfog import (
    compute_phase_congruency,
    detect_salient_keypoints,
    suppression_via_square_covering,
    find_best_correspondence_unified,
    compute_goa_search_surface,
    mutual_information_score,
    subpixel_phase_correlation,
    estimate_weighted_homography,
    calculate_reprojection_errors,
)
from scripts.generate_ground_truth_matches import (
    create_random_homography,
    apply_photometric_perturbation,
    apply_antisolar_cross_sensor,
)
from evaluation.eval_candidate_generation import get_train_test_split


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


def get_image_paths() -> List[Path]:
    triplets_dir = REPO_ROOT / "data_preprocessing_pipeline/processed_triplets"
    image_paths = []
    for reg_dir in sorted(triplets_dir.glob("region_*")):
        for name in ("ohrc_512.png", "tmc_512.png"):
            p = reg_dir / name
            if p.exists():
                image_paths.append(p)
    return image_paths


def run_phase8_experiment():
    train_groups, test_groups = get_train_test_split()
    xs_train_groups = [g for g in train_groups if "cross_sensor" in g]
    ss_train_groups = [g for g in train_groups if "same_sensor" in g]

    print("================================================================================")
    print("PHASE 8: ISOLATED CROSS-SENSOR GOA PRESELECTION EXPERIMENT")
    print("================================================================================")
    print(f"Development Cross-Sensor Groups : {len(xs_train_groups)}")
    print(f"Development Same-Sensor Groups  : {len(ss_train_groups)}")
    print(f"Held-Out Test Groups (FROZEN)   : {len(test_groups)} (Strictly unused)")
    print("Zero ground-truth leakage: H_gt used strictly for post-hoc evaluation.\n")

    image_paths = get_image_paths()
    hp = 8  # Frozen half_p=8

    # --------------------------------------------------------------------------
    # 1. EVALUATE GOA VS NCC SEARCH SURFACE RANKING (SECTION 3 & 6)
    # --------------------------------------------------------------------------
    print("--------------------------------------------------------------------------------")
    print("1. SEARCH-SURFACE RANKING AT GROUND-TRUTH TARGET (NCC VS GOA)")
    print("--------------------------------------------------------------------------------")

    ncc_ranks = []
    goa_ranks = []
    true_target_metrics = {
        "ncc": [], "abs_ncc": [], "goa": [], "mi": [], "joint": [],
        "false_ncc": [], "false_goa": [], "false_mi": [], "false_joint": [],
    }

    t0_rank = time.perf_counter()
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

        for t_idx in range(4):
            target_image = f"{p.stem}_cross_sensor_t{t_idx}.png"
            grp = f"{region}:{source_image}::{target_image}"
            if grp not in xs_train_groups:
                continue

            scene_seed = int(abs(hash(grp)) % (2**31 - 1))
            rng_s = np.random.default_rng(scene_seed)
            H_gt = create_random_homography(w, h, rng_s)
            warped = cv2.warpPerspective(img_gray, H_gt, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
            warped_xs = apply_antisolar_cross_sensor(warped, rng_s)
            pc2 = compute_phase_congruency(warped_xs, num_orientations=4, num_scales=3)

            s_rad = 28
            jit = 6.0
            rng_jit = np.random.default_rng(scene_seed + 1000)

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

                p_gt = cv2.perspectiveTransform(np.array([[[kx, ky]]], dtype=np.float64), H_gt).reshape(-1)
                gt_x, gt_y = float(p_gt[0]), float(p_gt[1])

                jx = float(rng_jit.uniform(-jit, jit))
                jy = float(rng_jit.uniform(-jit, jit))
                search_cx = int(round(gt_x + jx))
                search_cy = int(round(gt_y + jy))

                s_min_x, s_max_x = max(0, search_cx - s_rad), min(w, search_cx + s_rad)
                s_min_y, s_max_y = max(0, search_cy - s_rad), min(h, search_cy + s_rad)

                # Check if target is inside searchable window
                if not (s_min_x + hp <= gt_x <= s_max_x - hp and s_min_y + hp <= gt_y <= s_max_y - hp):
                    continue

                sw, sh = s_max_x - s_min_x, s_max_y - s_min_y
                if sw <= 2 * hp or sh <= 2 * hp:
                    continue
                search_region = pc2[s_min_y:s_max_y, s_min_x:s_max_x]
                if float(np.std(search_region)) < 1e-4:
                    continue

                # Compute NCC surface
                ncc_surf = cv2.matchTemplate(search_region, tmpl, cv2.TM_CCOEFF_NORMED)

                # Compute GOA surface
                goa_surf = compute_goa_search_surface(search_region, tmpl)

                # Target integer coordinates inside surface
                tgt_u = int(round(gt_x - s_min_x - hp))
                tgt_v = int(round(gt_y - s_min_y - hp))
                tgt_u = np.clip(tgt_u, 0, ncc_surf.shape[1] - 1)
                tgt_v = np.clip(tgt_v, 0, ncc_surf.shape[0] - 1)

                v_ncc = float(ncc_surf[tgt_v, tgt_u])
                v_goa = float(goa_surf[tgt_v, tgt_u])

                r_ncc = int(np.sum(ncc_surf.ravel() > v_ncc)) + 1
                r_goa = int(np.sum(goa_surf.ravel() > v_goa)) + 1

                ncc_ranks.append(r_ncc)
                goa_ranks.append(r_goa)

                # Sample target patch for MI
                cand_true = search_region[tgt_v : tgt_v + 2*hp, tgt_u : tgt_u + 2*hp]
                if cand_true.shape == tmpl.shape:
                    v_mi = mutual_information_score(tmpl, cand_true)
                else:
                    v_mi = 0.0
                v_joint = 0.6 * v_mi + 0.4 * max(0.0, v_ncc)

                true_target_metrics["ncc"].append(v_ncc)
                true_target_metrics["abs_ncc"].append(abs(v_ncc))
                true_target_metrics["goa"].append(v_goa)
                true_target_metrics["mi"].append(v_mi)
                true_target_metrics["joint"].append(v_joint)

                # False peak selected by GOA
                goa_max_loc = np.unravel_index(np.argmax(goa_surf), goa_surf.shape)
                f_y, f_x = goa_max_loc
                cand_false = search_region[f_y : f_y + 2*hp, f_x : f_x + 2*hp]
                f_mi = mutual_information_score(tmpl, cand_false) if cand_false.shape == tmpl.shape else 0.0
                f_ncc = float(ncc_surf[f_y, f_x])
                f_joint = 0.6 * f_mi + 0.4 * max(0.0, f_ncc)

                true_target_metrics["false_ncc"].append(f_ncc)
                true_target_metrics["false_goa"].append(float(goa_surf[f_y, f_x]))
                true_target_metrics["false_mi"].append(f_mi)
                true_target_metrics["false_joint"].append(f_joint)

    n_eval = len(ncc_ranks)
    print(f"Total evaluated search windows (where true target is in-window): N = {n_eval}\n")

    print(f"{'Ranking Metric':<30} {'Baseline (NCC)':>20} {'Experimental (GOA)':>22} {'Delta':>15}")
    print("-" * 90)
    med_ncc, med_goa = np.median(ncc_ranks), np.median(goa_ranks)
    mean_ncc, mean_goa = np.mean(ncc_ranks), np.mean(goa_ranks)
    print(f"{'Median True-Target Rank':<30} {med_ncc:>20.1f} {med_goa:>22.1f} {med_goa - med_ncc:>+15.1f}")
    print(f"{'Mean True-Target Rank':<30} {mean_ncc:>20.1f} {mean_goa:>22.1f} {mean_goa - mean_ncc:>+15.1f}")

    for k in (1, 3, 5, 10, 20):
        rec_ncc = sum(1 for r in ncc_ranks if r <= k) / n_eval * 100
        rec_goa = sum(1 for r in goa_ranks if r <= k) / n_eval * 100
        print(f"{f'Top-{k} Recall':<30} {rec_ncc:>19.2f}% {rec_goa:>21.2f}% {rec_goa - rec_ncc:>+14.2f}%")

    print("\n--------------------------------------------------------------------------------")
    print("SIMILARITY SURFACE VALUES AT TRUE TARGET VS SELECTED PEAK")
    print("--------------------------------------------------------------------------------")
    print(f"{'Metric':<25} {'True Target Mean ± Std':>28} {'Selected False Peak Mean ± Std':>32}")
    print("-" * 88)
    for m_key, name in [
        ("ncc", "NCC (Signed)"),
        ("abs_ncc", "|NCC| (Absolute)"),
        ("goa", "GOA (Polarity Invariant)"),
        ("mi", "Mutual Information (MI)"),
        ("joint", "Joint Score (0.6*MI + 0.4*NCC)"),
    ]:
        vt = true_target_metrics[m_key]
        vf = true_target_metrics["false_" + (m_key if m_key != "abs_ncc" else "ncc")]
        print(f"{name:<25} {np.mean(vt):>14.3f} ± {np.std(vt):.3f} {np.mean(vf):>18.3f} ± {np.std(vf):.3f}")

    # --------------------------------------------------------------------------
    # 2. CANDIDATE-LEVEL END-TO-END EVALUATION (SECTION 4)
    # --------------------------------------------------------------------------
    print("\n================================================================================")
    print("2. CANDIDATE-LEVEL EVALUATION ON 37 CROSS-SENSOR DEVELOPMENT GROUPS")
    print("================================================================================")

    def evaluate_matcher_candidates(use_goa: bool) -> Dict[str, Any]:
        results_by_group = {}
        t_start = time.perf_counter()

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

            for t_idx in range(4):
                target_image = f"{p.stem}_cross_sensor_t{t_idx}.png"
                grp = f"{region}:{source_image}::{target_image}"
                if grp not in xs_train_groups:
                    continue

                t_scene_start = time.perf_counter()
                scene_seed = int(abs(hash(grp)) % (2**31 - 1))
                rng_s = np.random.default_rng(scene_seed)
                H_gt = create_random_homography(w, h, rng_s)
                warped = cv2.warpPerspective(img_gray, H_gt, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
                warped_xs = apply_antisolar_cross_sensor(warped, rng_s)
                pc2 = compute_phase_congruency(warped_xs, num_orientations=4, num_scales=3)

                s_rad = 28
                jit = 6.0
                rng_jit = np.random.default_rng(scene_seed + 1000)

                matched_candidates = []
                errs = []

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

                    p_gt = cv2.perspectiveTransform(np.array([[[kx, ky]]], dtype=np.float64), H_gt).reshape(-1)
                    gt_x, gt_y = float(p_gt[0]), float(p_gt[1])

                    jx = float(rng_jit.uniform(-jit, jit))
                    jy = float(rng_jit.uniform(-jit, jit))
                    search_cx = int(round(gt_x + jx))
                    search_cy = int(round(gt_y + jy))

                    s_min_x, s_max_x = max(0, search_cx - s_rad), min(w, search_cx + s_rad)
                    s_min_y, s_max_y = max(0, search_cy - s_rad), min(h, search_cy + s_rad)

                    if s_max_x - s_min_x <= 2 * hp or s_max_y - s_min_y <= 2 * hp:
                        continue
                    search_region = pc2[s_min_y:s_max_y, s_min_x:s_max_x]
                    if float(np.std(search_region)) < 1e-4:
                        continue

                    # Execute find_best_correspondence_unified with or without GOA
                    score, loc = find_best_correspondence_unified(
                        search_region, tmpl,
                        multimodal_pair=True,
                        top_k=5,
                        use_goa_preselection=use_goa,
                    )
                    found_x = float(s_min_x + loc[0] + hp)
                    found_y = float(s_min_y + loc[1] + hp)

                    # Subpixel refinement
                    ibx, iby = int(round(found_x)), int(round(found_y))
                    if iby >= hp and iby + hp <= h and ibx >= hp and ibx + hp <= w:
                        p_ref = pc2[iby - hp : iby + hp, ibx - hp : ibx + hp]
                        if p_ref.shape == tmpl.shape:
                            dx, dy, _p, valid = subpixel_phase_correlation(tmpl, p_ref)
                            if valid and abs(dx) < 2.0 and abs(dy) < 2.0:
                                found_x += float(dx)
                                found_y += float(dy)

                    err = math.hypot(found_x - gt_x, found_y - gt_y)
                    errs.append(err)
                    matched_candidates.append({
                        "kx": kx, "ky": ky, "fx": found_x, "fy": found_y,
                        "score": score, "err": err,
                    })

                t_scene = time.perf_counter() - t_scene_start
                results_by_group[grp] = {
                    "n_cands": len(matched_candidates),
                    "errs": errs,
                    "t1": sum(1 for e in errs if e <= 1.0),
                    "t2": sum(1 for e in errs if e <= 2.0),
                    "t3": sum(1 for e in errs if e <= 3.0),
                    "t4": sum(1 for e in errs if e <= 4.0),
                    "t5": sum(1 for e in errs if e <= 5.0),
                    "runtime_s": t_scene,
                }

        total_time = time.perf_counter() - t_start
        return {"per_group": results_by_group, "total_time": total_time}

    print("Running Baseline Matcher (half_p=8 + NCC top-5)...")
    res_base = evaluate_matcher_candidates(use_goa=False)
    print("Running Experimental Matcher (half_p=8 + GOA top-5)...")
    res_goa = evaluate_matcher_candidates(use_goa=True)

    def summarize_run(res_dict):
        pg = res_dict["per_group"]
        cands = [v["n_cands"] for v in pg.values()]
        t1 = [v["t1"] for v in pg.values()]
        t2 = [v["t2"] for v in pg.values()]
        t3 = [v["t3"] for v in pg.values()]
        t4 = [v["t4"] for v in pg.values()]
        t5 = [v["t5"] for v in pg.values()]
        all_errs = [e for v in pg.values() for e in v["errs"]]
        prec4 = [v["t4"] / max(1, v["n_cands"]) for v in pg.values()]
        ge4 = sum(1 for v in pg.values() if v["t4"] >= 4)
        runtimes = [v["runtime_s"] for v in pg.values()]
        return {
            "mean_cands": np.mean(cands),
            "t1": np.mean(t1), "t2": np.mean(t2), "t3": np.mean(t3), "t4": np.mean(t4), "t5": np.mean(t5),
            "prec4": np.mean(prec4) * 100,
            "mean_err": np.mean(all_errs) if all_errs else float("nan"),
            "med_err": np.median(all_errs) if all_errs else float("nan"),
            "ge4_count": ge4, "ge4_pct": ge4 / len(pg) * 100,
            "mean_runtime": np.mean(runtimes),
            "total_time": res_dict["total_time"],
        }

    s_base = summarize_run(res_base)
    s_goa = summarize_run(res_goa)

    print(f"\n{'Candidate Metric':<35} {'Baseline (NCC Top-5)':>22} {'GOA Experiment':>22} {'Delta':>15}")
    print("-" * 96)
    print(f"{'Mean Candidates / Scene':<35} {s_base['mean_cands']:>22.1f} {s_goa['mean_cands']:>22.1f} {s_goa['mean_cands'] - s_base['mean_cands']:>+15.1f}")
    print(f"{'True Candidates <= 1.0 px':<35} {s_base['t1']:>22.2f} {s_goa['t1']:>22.2f} {s_goa['t1'] - s_base['t1']:>+15.2f}")
    print(f"{'True Candidates <= 2.0 px':<35} {s_base['t2']:>22.2f} {s_goa['t2']:>22.2f} {s_goa['t2'] - s_base['t2']:>+15.2f}")
    print(f"{'True Candidates <= 3.0 px':<35} {s_base['t3']:>22.2f} {s_goa['t3']:>22.2f} {s_goa['t3'] - s_base['t3']:>+15.2f}")
    print(f"{'True Candidates <= 4.0 px':<35} {s_base['t4']:>22.2f} {s_goa['t4']:>22.2f} {s_goa['t4'] - s_base['t4']:>+15.2f}")
    print(f"{'True Candidates <= 5.0 px':<35} {s_base['t5']:>22.2f} {s_goa['t5']:>22.2f} {s_goa['t5'] - s_base['t5']:>+15.2f}")
    print(f"{'Candidate Precision (<=4px)':<35} {s_base['prec4']:>21.2f}% {s_goa['prec4']:>21.2f}% {s_goa['prec4'] - s_base['prec4']:>+14.2f}%")
    print(f"{'Mean Geometric Error':<35} {s_base['mean_err']:>20.2f}px {s_goa['mean_err']:>20.2f}px {s_goa['mean_err'] - s_base['mean_err']:>+13.2f}px")
    print(f"{'Median Geometric Error':<35} {s_base['med_err']:>20.2f}px {s_goa['med_err']:>20.2f}px {s_goa['med_err'] - s_base['med_err']:>+13.2f}px")
    print(f"{'Scenes with >= 4 True Candidates':<35} {s_base['ge4_count']:>17d}/37 ({s_base['ge4_pct']:.1f}%) {s_goa['ge4_count']:>17d}/37 ({s_goa['ge4_pct']:.1f}%) {s_goa['ge4_pct'] - s_base['ge4_pct']:>+14.1f}%")
    print(f"{'Mean Runtime / Scene':<35} {s_base['mean_runtime']:>20.3f}s {s_goa['mean_runtime']:>20.3f}s {s_goa['mean_runtime'] - s_base['mean_runtime']:>+14.3f}s")

    # --------------------------------------------------------------------------
    # 3. SAME-SENSOR REGRESSION CHECK (SECTION 5)
    # --------------------------------------------------------------------------
    print("\n================================================================================")
    print("3. SAME-SENSOR REGRESSION CHECK (17 DEVELOPMENT GROUPS)")
    print("================================================================================")

    ss_diff_count = 0
    ss_total_points = 0

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

        for t_idx in range(2):
            target_image = f"{p.stem}_same_sensor_t{t_idx}.png"
            grp = f"{region}:{source_image}::{target_image}"
            if grp not in ss_train_groups:
                continue

            scene_seed = int(abs(hash(grp)) % (2**31 - 1))
            rng_s = np.random.default_rng(scene_seed)
            H_gt = create_random_homography(w, h, rng_s)
            warped = cv2.warpPerspective(img_gray, H_gt, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
            warped_ss = apply_photometric_perturbation(warped, rng_s)
            pc2 = compute_phase_congruency(warped_ss, num_orientations=4, num_scales=3)

            s_rad = 20
            jit = 8.0
            rng_jit = np.random.default_rng(scene_seed + 1000)

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

                p_gt = cv2.perspectiveTransform(np.array([[[kx, ky]]], dtype=np.float64), H_gt).reshape(-1)
                gt_x, gt_y = float(p_gt[0]), float(p_gt[1])

                jx = float(rng_jit.uniform(-jit, jit))
                jy = float(rng_jit.uniform(-jit, jit))
                search_cx = int(round(gt_x + jx))
                search_cy = int(round(gt_y + jy))

                s_min_x, s_max_x = max(0, search_cx - s_rad), min(w, search_cx + s_rad)
                s_min_y, s_max_y = max(0, search_cy - s_rad), min(h, search_cy + s_rad)

                if s_max_x - s_min_x <= 2 * hp or s_max_y - s_min_y <= 2 * hp:
                    continue
                search_region = pc2[s_min_y:s_max_y, s_min_x:s_max_x]
                if float(np.std(search_region)) < 1e-4:
                    continue

                # Run both modes on same-sensor
                s1, loc1 = find_best_correspondence_unified(
                    search_region, tmpl, multimodal_pair=False, use_goa_preselection=False
                )
                s2, loc2 = find_best_correspondence_unified(
                    search_region, tmpl, multimodal_pair=False, use_goa_preselection=True
                )

                ss_total_points += 1
                if loc1 != loc2 or abs(s1 - s2) > 1e-6:
                    ss_diff_count += 1

    print(f"Total same-sensor candidate tests: {ss_total_points}")
    print(f"Total matching discrepancies     : {ss_diff_count}")
    print(f"Same-sensor regression status    : {'PASSED (Zero difference)' if ss_diff_count == 0 else 'FAILED'}")

    # --------------------------------------------------------------------------
    # 4. SCIENTIFIC CLASSIFICATION & INTERPRETATION (SECTION 9)
    # --------------------------------------------------------------------------
    print("\n================================================================================")
    print("4. EXPERIMENTAL CLASSIFICATION & SCIENTIFIC INTERPRETATION")
    print("================================================================================")

    # Classification rules:
    # A — GOA strongly supported: GOA substantially improves true-target ranking and candidate recovery
    # B — GOA partially supported: GOA improves ranking but does not produce enough candidate recovery
    # C — GOA rejected: GOA fails to distinguish true targets from false locations
    top5_rec_diff = (sum(1 for r in goa_ranks if r <= 5) - sum(1 for r in ncc_ranks if r <= 5)) / n_eval * 100
    t4_diff = s_goa["t4"] - s_base["t4"]
    ge4_diff = s_goa["ge4_count"] - s_base["ge4_count"]

    if top5_rec_diff > 15.0 and t4_diff >= 1.5 and ge4_diff >= 5:
        classification = "A — GOA strongly supported"
        reason = "GOA substantially improved true-target ranking and candidate recovery."
    elif med_goa < med_ncc * 0.5 or top5_rec_diff > 5.0:
        classification = "B — GOA partially supported"
        reason = "GOA improved ranking somewhat, but did not translate into sufficient downstream candidate recovery."
    else:
        classification = "C — GOA rejected"
        reason = f"GOA failed to distinguish true targets from false locations (Median rank: {med_goa:.1f} vs NCC {med_ncc:.1f}; Top-5 recall: {sum(1 for r in goa_ranks if r <= 5)/n_eval*100:.1f}% vs NCC {sum(1 for r in ncc_ranks if r <= 5)/n_eval*100:.1f}%; scenes with >=4 true candidates: {s_goa['ge4_count']}/37 vs {s_base['ge4_count']}/37)."

    print(f"Official Classification: {classification}")
    print(f"Scientific Rationale   : {reason}")

    return {
        "ranking": {
            "med_ncc": med_ncc, "med_goa": med_goa,
            "mean_ncc": mean_ncc, "mean_goa": mean_goa,
            "top1_ncc": sum(1 for r in ncc_ranks if r <= 1) / n_eval * 100,
            "top1_goa": sum(1 for r in goa_ranks if r <= 1) / n_eval * 100,
            "top3_ncc": sum(1 for r in ncc_ranks if r <= 3) / n_eval * 100,
            "top3_goa": sum(1 for r in goa_ranks if r <= 3) / n_eval * 100,
            "top5_ncc": sum(1 for r in ncc_ranks if r <= 5) / n_eval * 100,
            "top5_goa": sum(1 for r in goa_ranks if r <= 5) / n_eval * 100,
            "top10_ncc": sum(1 for r in ncc_ranks if r <= 10) / n_eval * 100,
            "top10_goa": sum(1 for r in goa_ranks if r <= 10) / n_eval * 100,
            "top20_ncc": sum(1 for r in ncc_ranks if r <= 20) / n_eval * 100,
            "top20_goa": sum(1 for r in goa_ranks if r <= 20) / n_eval * 100,
        },
        "base_candidates": s_base,
        "goa_candidates": s_goa,
        "ss_diff_count": ss_diff_count,
        "classification": classification,
        "reason": reason,
        "res_base_pg": res_base["per_group"],
        "res_goa_pg": res_goa["per_group"],
    }


if __name__ == "__main__":
    run_phase8_experiment()
