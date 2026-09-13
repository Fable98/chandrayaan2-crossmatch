"""Phase 5: Cross-Sensor Matcher Fixes — Isolated NCC and Patch-Aperture Experiments.

Strictly follows:
- Evaluates only on the 54 development/training groups (18 test groups held out).
- H_gt used strictly as an independent post-hoc evaluation oracle.
- Tests:
    1. Baseline: half_p=8, top-5 positive NCC preselection.
    2. Experiment A (NCC preselection fix): half_p=8, absolute / bipolar NCC peak preselection
       so negative/near-zero NCC locations are not discarded before MI evaluation.
       Joint MI/NCC scoring is preserved.
    3. Experiment B (Scale-adaptive patch support): half_p=16, baseline matching algorithm.
    4. Experiment C (Combined): half_p=16 + NCC preselection fix.
- Reports per-scene and aggregate metrics:
    - Total candidates
    - True candidates at <=1 px, <=2 px, <=3 px, <=5 px
    - True candidate precision at <=4 px
    - Mean and median geometric error
    - Scenes with >=4 true candidates (count and %)
    - Runtime
    - Approximate explanatory RANSAC clean-sample statistic
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
    mutual_information_score,
    ncc_peak_uniqueness,
    apply_grid_density_budgeting,
    subpixel_phase_correlation,
)
from scripts.generate_ground_truth_matches import (
    create_random_homography,
    apply_antisolar_cross_sensor,
)
from sklearn.model_selection import GroupShuffleSplit


def get_train_groups(seed: int = SEED) -> List[str]:
    """Reproduce exact 54 train groups."""
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
    tr_idx, _ = selected_split
    return sorted(list(set(groups[tr_idx])))


def match_template_unified_custom(
    search_region: np.ndarray,
    tmpl: np.ndarray,
    multimodal_pair: bool = True,
    w_mi: float = 0.6,
    w_ncc: float = 0.4,
    top_k: int = 5,
    ncc_selection_mode: str = "baseline_pos",
) -> Tuple[float, Tuple[int, int]]:
    """Custom unified correspondence matcher supporting baseline vs fixed NCC preselection."""
    res = cv2.matchTemplate(search_region, tmpl, cv2.TM_CCOEFF_NORMED)
    if not multimodal_pair:
        _, max_val, _, max_loc = cv2.minMaxLoc(res)
        return float(max_val), max_loc

    th, tw = tmpl.shape[:2]
    flat = res.ravel()
    k = min(top_k, flat.size)
    if k <= 1:
        _, max_val, _, max_loc = cv2.minMaxLoc(res)
        cand = search_region[max_loc[1] : max_loc[1] + th, max_loc[0] : max_loc[0] + tw]
        mi = mutual_information_score(tmpl, cand) if cand.shape == tmpl.shape else 0.0
        score = w_mi * mi + w_ncc * max(0.0, float(max_val))
        return float(score), max_loc

    if ncc_selection_mode == "baseline_pos":
        # Baseline: top-K algebraically highest (positive) NCC
        top_indices = np.argpartition(-flat, k)[:k]
        top_indices = top_indices[np.argsort(-flat[top_indices])]
    elif ncc_selection_mode == "abs_and_bipolar":
        # Experiment A: Sample top correlation magnitudes (|NCC|) so negative and
        # near-zero locations with significant statistical dependence are not discarded.
        # Evaluate top k/2 positive peaks and top k/2 negative troughs + top absolute.
        k_half = max(1, k // 2)
        top_pos = np.argpartition(-flat, k_half)[:k_half]
        top_neg = np.argpartition(flat, k_half)[:k_half]
        abs_flat = np.abs(flat)
        top_abs = np.argpartition(-abs_flat, k)[:k]
        top_indices = np.unique(np.concatenate([top_pos, top_neg, top_abs]))
    else:
        raise ValueError(f"Unknown ncc_selection_mode: {ncc_selection_mode}")

    best_score = -1.0
    best_loc = (0, 0)
    for idx in top_indices:
        cy, cx = np.unravel_index(idx, res.shape)
        cand = search_region[cy : cy + th, cx : cx + tw]
        if cand.shape != tmpl.shape:
            continue
        # Preserve existing joint MI/NCC scoring mechanism
        ncc_val = max(0.0, float(res[cy, cx]))
        mi_val = mutual_information_score(tmpl, cand)
        joint_score = float(w_mi * mi_val + w_ncc * ncc_val)
        if joint_score > best_score:
            best_score = joint_score
            best_loc = (int(cx), int(cy))

    if best_score < 0.0:
        _, max_val, _, max_loc = cv2.minMaxLoc(res)
        return float(max_val), max_loc

    return float(best_score), best_loc


def run_experiment_on_scene(
    img_gray: np.ndarray,
    H_gt: np.ndarray,
    rng: np.random.Generator,
    half_p: int,
    ncc_selection_mode: str,
    search_rad: int = 28,
) -> Dict[str, Any]:
    """Execute candidate generation on a single cross-sensor scene under specified configuration."""
    h, w = img_gray.shape[:2]
    t0 = time.perf_counter()

    warped = cv2.warpPerspective(img_gray, H_gt, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    target = apply_antisolar_cross_sensor(warped, rng)

    pc1 = compute_phase_congruency(img_gray, num_orientations=4, num_scales=3)
    pc2 = compute_phase_congruency(target, num_orientations=4, num_scales=3)

    kps_raw = detect_salient_keypoints(pc1, max_corners=500, quality_level=0.008)
    kps_ssc = suppression_via_square_covering(kps_raw, num_ret_points=60, tolerance=0.12, cols=w, rows=h)

    jitter = 6.0
    matched_candidates = []

    for kp in kps_ssc:
        kx, ky = float(kp[0]), float(kp[1])
        cx, cy = int(round(kx)), int(round(ky))
        if cy < half_p or cy >= h - half_p or cx < half_p or cx >= w - half_p:
            continue
        tmpl = pc1[cy - half_p : cy + half_p, cx - half_p : cx + half_p]
        if float(np.std(tmpl)) < 1e-4:
            continue

        p_gt = cv2.perspectiveTransform(np.array([[[kx, ky]]], dtype=np.float64), H_gt).reshape(-1)
        gx, gy = float(p_gt[0]), float(p_gt[1])
        if gx < half_p or gx >= w - half_p or gy < half_p or gy >= h - half_p:
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

        score, loc = match_template_unified_custom(
            search_region, tmpl, multimodal_pair=True,
            top_k=10 if ncc_selection_mode != "baseline_pos" else 5,
            ncc_selection_mode=ncc_selection_mode,
        )
        found_x = float(s_min_x + loc[0] + half_p)
        found_y = float(s_min_y + loc[1] + half_p)

        ref_x, ref_y = int(round(found_x)), int(round(found_y))
        sub_dx, sub_dy = 0.0, 0.0
        if half_p <= ref_x < w - half_p and half_p <= ref_y < h - half_p:
            p1_sub = img_gray[cy - half_p : cy + half_p, cx - half_p : cx + half_p]
            p2_sub = target[ref_y - half_p : ref_y + half_p, ref_x - half_p : ref_x + half_p]
            if p1_sub.shape == p2_sub.shape:
                sdx, sdy, _, valid = subpixel_phase_correlation(p1_sub, p2_sub)
                if valid:
                    sub_dx, sub_dy = float(sdx), float(sdy)

        final_x = found_x + sub_dx
        final_y = found_y + sub_dy
        err = float(math.hypot(final_x - gx, final_y - gy))

        matched_candidates.append({
            "source_x": kx, "source_y": ky,
            "target_x": final_x, "target_y": final_y,
            "score": score,
            "err": err,
        })

    # Confidence filter & grid density budgeting
    thresh = 0.08
    conf_passed = [c for c in matched_candidates if c["score"] >= thresh]
    dict_for_nms = []
    for i, c in enumerate(conf_passed):
        dict_for_nms.append({
            "work_x1": c["source_x"], "work_y1": c["source_y"],
            "work_x2": c["target_x"], "work_y2": c["target_y"],
            "score": c["score"], "cell": (int(c["source_x"] / 51.2), int(c["source_y"] / 51.2)),
            "orig_idx": i,
        })
    nms_survivors = apply_grid_density_budgeting(dict_for_nms, image_shape=(h, w), grid_dims=(10, 10), max_per_cell=4)
    final_cands = [conf_passed[m["orig_idx"]] for m in nms_survivors] if nms_survivors else []

    runtime_s = time.perf_counter() - t0
    return {
        "candidates": final_cands,
        "runtime_s": runtime_s,
    }


def evaluate_all_experiments():
    train_groups = get_train_groups()
    triplets_dir = REPO_ROOT / "data_preprocessing_pipeline/processed_triplets"
    region_images = []
    for reg_dir in sorted(triplets_dir.glob("region_*")):
        for name in ("ohrc_512.png", "tmc_512.png"):
            p = reg_dir / name
            if p.exists():
                region_images.append(p)

    print("================================================================================")
    print("PHASE 5: CROSS-SENSOR MATCHER EXPERIMENTS (TRAINING GROUPS ONLY)")
    print("================================================================================")
    print(f"Total training scenes available: 37 cross-sensor image-pair transforms")
    print("Configs:")
    print("  Baseline     : half_p = 8  (16x16 px patch), Top-5 positive NCC preselection")
    print("  Experiment A : half_p = 8  (16x16 px patch), Bipolar/Abs NCC preselection fix")
    print("  Experiment B : half_p = 16 (32x32 px patch), Top-5 positive NCC preselection")
    print("  Experiment C : half_p = 16 (32x32 px patch), Bipolar/Abs NCC preselection fix\n")

    experiments = {
        "Baseline": {"half_p": 8, "ncc_mode": "baseline_pos"},
        "Exp A (NCC Fix)": {"half_p": 8, "ncc_mode": "abs_and_bipolar"},
        "Exp B (Patch 32)": {"half_p": 16, "ncc_mode": "baseline_pos"},
        "Exp C (Combined)": {"half_p": 16, "ncc_mode": "abs_and_bipolar"},
    }

    results: Dict[str, Dict[str, List[Any]]] = {
        exp_name: {
            "total_cands": [], "t1": [], "t2": [], "t3": [], "t5": [], "t4": [],
            "prec4": [], "errs_all": [], "scenes_ge4": 0, "runtimes": [],
        }
        for exp_name in experiments
    }

    n_scenes = 0

    for p in region_images:
        img_gray = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
        h, w = img_gray.shape[:2]
        reg = p.parent.name
        src_name = p.name

        for t_idx in range(4):
            grp_id = f"{reg}:{src_name}::{p.stem}_cross_sensor_t{t_idx}.png"
            if grp_id not in train_groups:
                continue
            n_scenes += 1

            # Seed per scene so all 4 experiments receive identical geometric transforms
            scene_seed = SEED + t_idx * 100 + hash(grp_id) % 10000

            for exp_name, cfg in experiments.items():
                rng = np.random.default_rng(scene_seed)
                H_gt = create_random_homography(w, h, rng)
                out = run_experiment_on_scene(
                    img_gray, H_gt, rng,
                    half_p=cfg["half_p"],
                    ncc_selection_mode=cfg["ncc_mode"],
                )
                cands = out["candidates"]
                n_c = len(cands)
                errs = [c["err"] for c in cands]

                n_t1 = sum(1 for e in errs if e <= 1.0)
                n_t2 = sum(1 for e in errs if e <= 2.0)
                n_t3 = sum(1 for e in errs if e <= 3.0)
                n_t5 = sum(1 for e in errs if e <= 5.0)
                n_t4 = sum(1 for e in errs if e <= 4.0)

                res_dict = results[exp_name]
                res_dict["total_cands"].append(n_c)
                res_dict["t1"].append(n_t1)
                res_dict["t2"].append(n_t2)
                res_dict["t3"].append(n_t3)
                res_dict["t5"].append(n_t5)
                res_dict["t4"].append(n_t4)
                res_dict["prec4"].append(n_t4 / max(1, n_c))
                res_dict["errs_all"].extend(errs)
                if n_t4 >= 4:
                    res_dict["scenes_ge4"] += 1
                res_dict["runtimes"].append(out["runtime_s"])

    # Aggregate & print results
    print(f"Evaluated {n_scenes} cross-sensor training scenes across all configurations.\n")

    # Table 1: Per-scene true candidates & precision
    print("-------------------------------------------------------------------------------------------------------------")
    print("1. CANDIDATE RECOVERY & PRECISION PER SCENE (CROSS-SENSOR DEV SET, N=37)")
    print("-------------------------------------------------------------------------------------------------------------")
    print(f"{'Experiment':<18} {'Mean Cands':>11} {'True<=1px':>10} {'True<=2px':>10} {'True<=3px':>10} {'True<=5px':>10} {'Prec<=4px':>11} {'Mean Err':>10} {'Med Err':>9} {'Runtime/sc':>11}")
    print("-" * 115)

    for exp_name in experiments:
        rd = results[exp_name]
        m_cands = np.mean(rd["total_cands"])
        m_t1 = np.mean(rd["t1"])
        m_t2 = np.mean(rd["t2"])
        m_t3 = np.mean(rd["t3"])
        m_t5 = np.mean(rd["t5"])
        m_p4 = np.mean(rd["prec4"]) * 100
        m_err = np.mean(rd["errs_all"]) if rd["errs_all"] else float("nan")
        med_err = np.median(rd["errs_all"]) if rd["errs_all"] else float("nan")
        m_rt = np.mean(rd["runtimes"])
        print(f"{exp_name:<18} {m_cands:>11.1f} {m_t1:>10.2f} {m_t2:>10.2f} {m_t3:>10.2f} {m_t5:>10.2f} {m_p4:>10.2f}% {m_err:>9.2f}px {med_err:>8.2f}px {m_rt*1000:>9.1f}ms")

    # Table 2: >= 4 True Candidates & Explanatory RANSAC Feasibility
    print("\n-------------------------------------------------------------------------------------------------------------")
    print("2. SCENES REACHING >= 4 TRUE CANDIDATES & EXPLANATORY RANSAC CLEAN-SAMPLE PROBABILITY")
    print("-------------------------------------------------------------------------------------------------------------")
    print(f"{'Experiment':<18} {'Scenes >= 4 True':>18} {'Prevalence (p)':>16} {'Clean Draw p^4 (approx)':>25} {'2000-iter P(succ) approx':>26}")
    print("-" * 115)

    for exp_name in experiments:
        rd = results[exp_name]
        ge4_count = rd["scenes_ge4"]
        pct_ge4 = (ge4_count / n_scenes) * 100
        p_prev = np.mean(rd["t4"]) / max(1.0, np.mean(rd["total_cands"]))
        p4_clean = p_prev ** 4
        p_succ_2k = 1.0 - (1.0 - p4_clean) ** 2000
        print(f"{exp_name:<18} {ge4_count:>3}/{n_scenes} ({pct_ge4:>5.1f}%) {p_prev*100:>15.2f}% {p4_clean:>25.4e} {p_succ_2k*100:>25.2f}%")

    print("\nNote: p^4 and P(success) are approximate explanatory statistics describing minimal-sample drawing")
    print("feasibility under uniform random sampling; they are not exact predictions of RANSAC termination.")


if __name__ == "__main__":
    evaluate_all_experiments()
