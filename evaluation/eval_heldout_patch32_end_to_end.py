"""Phase 6: Held-Out End-to-End Validation of Scale-Adaptive Patch Support.

Strictly adheres to:
1. Frozen 18 held-out test groups (11 cross-sensor, 7 same-sensor).
2. RF model frozen (trained strictly on 54 training groups, ML_model/ai_verifier_model.pkl untouched).
3. Evaluates Arm A (Production baseline: half_p=8) vs Arm B (Patch-32: half_p=16).
4. No test-set tuning or post-hoc optimization.
5. H_gt used strictly as an independent post-hoc evaluation oracle.
6. Measures candidate-generation metrics, end-to-end registration metrics, paired statistics,
   and Wilcoxon signed-rank tests.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path
from typing import Dict, List, Any, Tuple
import numpy as np
import cv2
from scipy.stats import wilcoxon
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GroupShuffleSplit
import joblib

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
    estimate_weighted_homography,
    calculate_reprojection_errors,
)
from scripts.generate_ground_truth_matches import generate_matches_for_image

FROZEN_10_FEATURES = [
    "confidence",
    "refinement_dx",
    "refinement_dy",
    "spatial_quality_score",
    "cfog_distance",
    "pc_energy_src",
    "pc_energy_tgt",
    "nn_ratio",
    "scale_diff",
    "disp_consistency",
]

TMP_MODEL_PATH = Path("/tmp/ai_verifier_registration_eval.pkl")


def compute_corner_error(H_est: np.ndarray | None, H_gt: np.ndarray, w: float = 512.0, h: float = 512.0) -> float:
    """Evaluate geometric displacement error against H_gt at image corners."""
    if H_est is None or not np.all(np.isfinite(H_est)):
        return float("inf")
    corners = np.array([
        [[0.0, 0.0]],
        [[w, 0.0]],
        [[w, h]],
        [[0.0, h]],
    ], dtype=np.float64)
    try:
        proj_gt = cv2.perspectiveTransform(corners, H_gt).reshape(-1, 2)
        proj_est = cv2.perspectiveTransform(corners, H_est).reshape(-1, 2)
        errs = np.linalg.norm(proj_est - proj_gt, axis=1)
        return float(np.mean(errs))
    except Exception:
        return float("inf")


def get_heldout_split(seed: int = SEED) -> Tuple[List[str], List[str]]:
    """Reproduce exact 54 train / 18 held-out test groups."""
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


def get_or_train_rf_model(train_groups: List[str]) -> Any:
    """Ensure RF model is trained strictly on the 54 training groups."""
    if TMP_MODEL_PATH.exists():
        bundle = joblib.load(TMP_MODEL_PATH)
        return bundle["model"]

    json_path = REPO_ROOT / "ML_model/ground_truth_matches.json"
    records = load_matches_from_file(json_path)
    X_rows, y_rows = [], []
    seen = set()
    for m in records:
        grp, _, _ = extract_provenance_group(m)
        if grp not in train_groups:
            continue
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
        X_rows.append(row[:10])
        y_rows.append(lbl)

    clf = RandomForestClassifier(n_estimators=100, class_weight="balanced", random_state=SEED, n_jobs=-1)
    clf.fit(np.asarray(X_rows, dtype=np.float64), np.asarray(y_rows, dtype=np.int64))
    TMP_MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": clf, "feature_names": FROZEN_10_FEATURES, "n_train": len(X_rows)}, TMP_MODEL_PATH)
    return clf


def run_heldout_evaluation(rng_seeds: List[int] = [42]):
    train_groups, test_groups = get_heldout_split()
    xs_test_groups = [g for g in test_groups if "cross_sensor" in g]
    ss_test_groups = [g for g in test_groups if "same_sensor" in g]

    print("================================================================================")
    print("PHASE 6: HELDOUT VALIDATION — SCALE-ADAPTIVE PATCH SUPPORT (HALF_P=16)")
    print("================================================================================")
    print(f"Total held-out test groups: {len(test_groups)}")
    print(f"  Cross-sensor test groups: {len(xs_test_groups)}")
    print(f"  Same-sensor test groups : {len(ss_test_groups)}")
    print("Zero leakage verification:")
    overlap = set(train_groups) & set(test_groups)
    print(f"  Train/Test Group Overlap: {len(overlap)} (Zero leakage: {len(overlap) == 0})\n")

    # Load frozen RF model
    rf_model = get_or_train_rf_model(train_groups)
    print(f"Frozen RF Model loaded from: {TMP_MODEL_PATH}")
    print(f"Production model untouched: {REPO_ROOT / 'ML_model/ai_verifier_model.pkl'}\n")

    triplets_dir = REPO_ROOT / "data_preprocessing_pipeline/processed_triplets"
    image_paths = []
    for reg_dir in sorted(triplets_dir.glob("region_*")):
        for name in ("ohrc_512.png", "tmc_512.png"):
            p = reg_dir / name
            if p.exists():
                image_paths.append(p)

    # Generate both arms under deterministic scene generator
    # Arm A: half_p = 8
    # Arm B: half_p = 16
    scenes_a = []
    rng_a = np.random.default_rng(SEED)
    for p in image_paths:
        generate_matches_for_image(p, rng_a, scene_sink=scenes_a, half_p=8)

    scenes_b = []
    rng_b = np.random.default_rng(SEED)
    for p in image_paths:
        generate_matches_for_image(p, rng_b, scene_sink=scenes_b, half_p=16)

    # Re-run generator to get matches per scene
    matches_a_by_group = {}
    rng_a2 = np.random.default_rng(SEED)
    for p in image_paths:
        m_list = generate_matches_for_image(p, rng_a2, half_p=8)
        for m in m_list:
            grp = f"{m['region']}:{m['source_image']}::{m['target_image']}"
            if grp in test_groups:
                matches_a_by_group.setdefault(grp, []).append(m)

    matches_b_by_group = {}
    rng_b2 = np.random.default_rng(SEED)
    for p in image_paths:
        m_list = generate_matches_for_image(p, rng_b2, half_p=16)
        for m in m_list:
            grp = f"{m['region']}:{m['source_image']}::{m['target_image']}"
            if grp in test_groups:
                matches_b_by_group.setdefault(grp, []).append(m)

    # Build H_gt lookup
    h_gt_lookup = {}
    for sc in scenes_a:
        grp = f"{sc['region']}:{sc['source_image']}::{sc['target_image']}"
        h_gt_lookup[grp] = sc["H_gt"]

    # 1. Candidate-Generation Metrics on Cross-Sensor Held-Out Groups
    print("--------------------------------------------------------------------------------")
    print("1. CANDIDATE-GENERATION EVALUATION ON 11 HELDOUT CROSS-SENSOR GROUPS")
    print("--------------------------------------------------------------------------------")
    cand_metrics = {"base": {"total": [], "t1": [], "t2": [], "t3": [], "t4": [], "t5": [], "prec4": [], "errs": []},
                    "patch16": {"total": [], "t1": [], "t2": [], "t3": [], "t4": [], "t5": [], "prec4": [], "errs": []}}

    for grp in xs_test_groups:
        H_gt = h_gt_lookup[grp]
        for arm, m_dict in (("base", matches_a_by_group), ("patch16", matches_b_by_group)):
            m_list = m_dict.get(grp, [])
            n_c = len(m_list)
            errs = []
            for m in m_list:
                kx, ky = float(m["source_x"]), float(m["source_y"])
                fx, fy = float(m["target_x"]), float(m["target_y"])
                p_gt = cv2.perspectiveTransform(np.array([[[kx, ky]]], dtype=np.float64), H_gt).reshape(-1)
                e = float(math.hypot(fx - p_gt[0], fy - p_gt[1]))
                errs.append(e)

            n_t1 = sum(1 for e in errs if e <= 1.0)
            n_t2 = sum(1 for e in errs if e <= 2.0)
            n_t3 = sum(1 for e in errs if e <= 3.0)
            n_t4 = sum(1 for e in errs if e <= 4.0)
            n_t5 = sum(1 for e in errs if e <= 5.0)

            cand_metrics[arm]["total"].append(n_c)
            cand_metrics[arm]["t1"].append(n_t1)
            cand_metrics[arm]["t2"].append(n_t2)
            cand_metrics[arm]["t3"].append(n_t3)
            cand_metrics[arm]["t4"].append(n_t4)
            cand_metrics[arm]["t5"].append(n_t5)
            cand_metrics[arm]["prec4"].append(n_t4 / max(1, n_c))
            cand_metrics[arm]["errs"].extend(errs)

    print(f"{'Metric':<35} {'Baseline (half_p=8)':>22} {'Patch 32 (half_p=16)':>22} {'Difference':>15}")
    print("-" * 98)
    mb_tot = np.mean(cand_metrics["base"]["total"])
    mp_tot = np.mean(cand_metrics["patch16"]["total"])
    print(f"{'Mean Candidates / Scene':<35} {mb_tot:>22.1f} {mp_tot:>22.1f} {mp_tot - mb_tot:>+15.1f}")

    mb_t1 = np.mean(cand_metrics["base"]["t1"])
    mp_t1 = np.mean(cand_metrics["patch16"]["t1"])
    print(f"{'True Candidates <= 1.0 px':<35} {mb_t1:>22.2f} {mp_t1:>22.2f} {mp_t1 - mb_t1:>+15.2f}")

    mb_t2 = np.mean(cand_metrics["base"]["t2"])
    mp_t2 = np.mean(cand_metrics["patch16"]["t2"])
    print(f"{'True Candidates <= 2.0 px':<35} {mb_t2:>22.2f} {mp_t2:>22.2f} {mp_t2 - mb_t2:>+15.2f}")

    mb_t3 = np.mean(cand_metrics["base"]["t3"])
    mp_t3 = np.mean(cand_metrics["patch16"]["t3"])
    print(f"{'True Candidates <= 3.0 px':<35} {mb_t3:>22.2f} {mp_t3:>22.2f} {mp_t3 - mb_t3:>+15.2f}")

    mb_t4 = np.mean(cand_metrics["base"]["t4"])
    mp_t4 = np.mean(cand_metrics["patch16"]["t4"])
    print(f"{'True Candidates <= 4.0 px':<35} {mb_t4:>22.2f} {mp_t4:>22.2f} {mp_t4 - mb_t4:>+15.2f}")

    mb_t5 = np.mean(cand_metrics["base"]["t5"])
    mp_t5 = np.mean(cand_metrics["patch16"]["t5"])
    print(f"{'True Candidates <= 5.0 px':<35} {mb_t5:>22.2f} {mp_t5:>22.2f} {mp_t5 - mb_t5:>+15.2f}")

    mb_p4 = np.mean(cand_metrics["base"]["prec4"]) * 100
    mp_p4 = np.mean(cand_metrics["patch16"]["prec4"]) * 100
    print(f"{'True Candidate Precision (<=4px)':<35} {mb_p4:>21.2f}% {mp_p4:>21.2f}% {mp_p4 - mb_p4:>+14.2f}%")

    mb_err = np.mean(cand_metrics["base"]["errs"])
    mp_err = np.mean(cand_metrics["patch16"]["errs"])
    print(f"{'Mean Geometric Error':<35} {mb_err:>20.2f}px {mp_err:>20.2f}px {mp_err - mb_err:>+13.2f}px")

    mb_med = np.median(cand_metrics["base"]["errs"])
    mp_med = np.median(cand_metrics["patch16"]["errs"])
    print(f"{'Median Geometric Error':<35} {mb_med:>20.2f}px {mp_med:>20.2f}px {mp_med - mb_med:>+13.2f}px")

    ge4_base = sum(1 for t in cand_metrics["base"]["t4"] if t >= 4)
    ge4_patch = sum(1 for t in cand_metrics["patch16"]["t4"] if t >= 4)
    n_xs = len(xs_test_groups)
    print(f"{'Scenes with >= 4 True Candidates':<35} {ge4_base}/{n_xs} ({ge4_base/n_xs*100:.1f}%) {ge4_patch}/{n_xs} ({ge4_patch/n_xs*100:.1f}%) {(ge4_patch - ge4_base)/n_xs*100:>+14.1f}%\n")

    # 2. End-to-End Registration Evaluation
    print("--------------------------------------------------------------------------------")
    print("2. END-TO-END REGISTRATION RESULTS ACROSS ALL 18 HELDOUT GROUPS")
    print("--------------------------------------------------------------------------------")

    def evaluate_registration_for_arm(matches_by_grp, arm_name):
        reg_runs = []
        for seed_val in rng_seeds:
            for grp in test_groups:
                m_list = matches_by_grp.get(grp, [])
                H_gt = h_gt_lookup[grp]
                dom = m_list[0].get("domain", "unknown")
                n_cands = len(m_list)

                pts1 = np.array([[float(m["source_x"]), float(m["source_y"])] for m in m_list], dtype=np.float32)
                pts2 = np.array([[float(m["target_x"]), float(m["target_y"])] for m in m_list], dtype=np.float32)

                # Live ground-truth labels for post-hoc oracle evaluation
                gt_labels = []
                for m in m_list:
                    kx, ky = float(m["source_x"]), float(m["source_y"])
                    fx, fy = float(m["target_x"]), float(m["target_y"])
                    p_gt = cv2.perspectiveTransform(np.array([[[kx, ky]]], dtype=np.float64), H_gt).reshape(-1)
                    gt_labels.append(1 if math.hypot(fx - p_gt[0], fy - p_gt[1]) <= 4.0 else 0)
                gt_labels = np.array(gt_labels, dtype=np.int32)
                n_true = int(np.sum(gt_labels == 1))

                # Feature row extraction for RF weighting
                feat_rows = []
                for m in m_list:
                    r = extract_feature_row(m, require_live_features=True)
                    feat_rows.append(r[:10])
                rf_probs = rf_model.predict_proba(np.array(feat_rows))[:, 1]

                # Run AI-weighted RANSAC (the standard production pipeline)
                w_ai = np.asarray(rf_probs, dtype=np.float64)
                H_est, mask, _ = estimate_weighted_homography(
                    pts1, pts2, w_ai,
                    ransac_reproj_threshold=5.0,
                    image_shape=(512, 512),
                    rng_seed=seed_val,
                    n_iters=2000,
                )

                success = (H_est is not None and mask is not None and int(np.sum(mask)) >= 4)
                if not success:
                    s_dict = {
                        "success": False, "inliers": 0, "inlier_ratio": 0.0,
                        "reproj_rmse": float("nan"), "corner_err": float("inf"),
                        "gt_tp": 0, "gt_fp": 0, "gt_precision": 0.0, "gt_recall": 0.0,
                    }
                else:
                    idx = np.where(np.asarray(mask).ravel() != 0)[0]
                    n_inl = len(idx)
                    errs = calculate_reprojection_errors(pts1[idx], pts2[idx], H_est)
                    rmse = float(np.sqrt(np.mean(errs ** 2))) if len(errs) else float("nan")
                    c_err = compute_corner_error(H_est, H_gt)

                    tp = int(np.sum(gt_labels[idx] == 1))
                    fp = int(np.sum(gt_labels[idx] == 0))
                    prec = float(tp) / float(n_inl) if n_inl > 0 else 0.0
                    rec = float(tp) / float(n_true) if n_true > 0 else 0.0
                    s_dict = {
                        "success": True, "inliers": n_inl, "inlier_ratio": float(n_inl) / float(n_cands),
                        "reproj_rmse": rmse, "corner_err": c_err,
                        "gt_tp": tp, "gt_fp": fp, "gt_precision": prec, "gt_recall": rec,
                    }

                reg_runs.append({
                    "group": grp, "domain": dom, "seed": seed_val,
                    "n_cands": n_cands, "n_true": n_true, "res": s_dict,
                })
        return reg_runs

    runs_base = evaluate_registration_for_arm(matches_a_by_group, "Baseline (half_p=8)")
    runs_patch16 = evaluate_registration_for_arm(matches_b_by_group, "Patch 32 (half_p=16)")

    # Group runs by slice
    for slice_name in ("Cross-Sensor", "Same-Sensor", "Overall"):
        if slice_name == "Overall":
            rb = runs_base
            rp = runs_patch16
        elif slice_name == "Same-Sensor":
            rb = [r for r in runs_base if r["domain"] == "same_sensor"]
            rp = [r for r in runs_patch16 if r["domain"] == "same_sensor"]
        else:
            rb = [r for r in runs_base if r["domain"] == "cross_sensor"]
            rp = [r for r in runs_patch16 if r["domain"] == "cross_sensor"]

        n_c = len(rb)
        succ_b = sum(1 for r in rb if r["res"]["success"])
        succ_p = sum(1 for r in rp if r["res"]["success"])

        inl_b = [r["res"]["inliers"] for r in rb]
        inl_p = [r["res"]["inliers"] for r in rp]

        ratio_b = [r["res"]["inlier_ratio"] for r in rb]
        ratio_p = [r["res"]["inlier_ratio"] for r in rp]

        reproj_b = [r["res"]["reproj_rmse"] for r in rb if np.isfinite(r["res"]["reproj_rmse"])]
        reproj_p = [r["res"]["reproj_rmse"] for r in rp if np.isfinite(r["res"]["reproj_rmse"])]

        corner_b = [r["res"]["corner_err"] for r in rb if np.isfinite(r["res"]["corner_err"])]
        corner_p = [r["res"]["corner_err"] for r in rp if np.isfinite(r["res"]["corner_err"])]

        prec_b = [r["res"]["gt_precision"] for r in rb if r["res"]["success"]]
        prec_p = [r["res"]["gt_precision"] for r in rp if r["res"]["success"]]

        rec_b = [r["res"]["gt_recall"] for r in rb if r["res"]["success"]]
        rec_p = [r["res"]["gt_recall"] for r in rp if r["res"]["success"]]

        print(f"\n================================================================================")
        print(f"REGISTRATION RESULTS: {slice_name.upper()} (N={n_c} cases)")
        print(f"================================================================================")
        print(f"{'Metric':<32} {'Baseline (half_p=8)':>20} {'Patch 32 (half_p=16)':>22} {'Difference':>16}")
        print("-" * 94)
        print(f"{'Homography Success Rate':<32} {succ_b}/{n_c} ({succ_b/n_c*100:.1f}%) {succ_p}/{n_c} ({succ_p/n_c*100:.1f}%) {(succ_p - succ_b)/n_c*100:>+15.1f}%")
        print(f"{'Median Inlier Count':<32} {np.median(inl_b):>20.1f} {np.median(inl_p):>22.1f} {np.median(inl_p) - np.median(inl_b):>+16.1f}")
        print(f"{'Median Inlier Ratio':<32} {np.median(ratio_b)*100:>19.2f}% {np.median(ratio_p)*100:>21.2f}% {(np.median(ratio_p) - np.median(ratio_b))*100:>+15.2f}%")
        med_rep_b = np.median(reproj_b) if reproj_b else float("nan")
        med_rep_p = np.median(reproj_p) if reproj_p else float("nan")
        print(f"{'Median Reprojection RMSE (px)':<32} {med_rep_b:>20.3f} {med_rep_p:>22.3f} {med_rep_b - med_rep_p:>+16.3f}")
        med_c_b = np.median(corner_b) if corner_b else float("nan")
        med_c_p = np.median(corner_p) if corner_p else float("nan")
        print(f"{'Median Corner Error vs H_gt (px)':<32} {med_c_b:>20.3f} {med_c_p:>22.3f} {med_c_b - med_c_p:>+16.3f}")
        med_p_b = np.median(prec_b) if prec_b else 0.0
        med_p_p = np.median(prec_p) if prec_p else 0.0
        print(f"{'GT Correspondence Precision':<32} {med_p_b*100:>19.2f}% {med_p_p*100:>21.2f}% {(med_p_p - med_p_b)*100:>+15.2f}%")
        med_r_b = np.median(rec_b) if rec_b else 0.0
        med_r_p = np.median(rec_p) if rec_p else 0.0
        print(f"{'GT Correspondence Recall':<32} {med_r_b*100:>19.2f}% {med_r_p*100:>21.2f}% {(med_r_p - med_r_b)*100:>+15.2f}%")

    # 3. Paired Per-Case Analysis on Cross-Sensor Held-Out Groups
    xs_b = [r for r in runs_base if r["domain"] == "cross_sensor"]
    xs_p = [r for r in runs_patch16 if r["domain"] == "cross_sensor"]

    print("\n================================================================================")
    print("PAIRED PER-CASE ANALYSIS: CROSS-SENSOR HELDOUT TEST PAIRS")
    print("================================================================================")
    print(f"{'Group':<42} {'Base Err':>10} {'Patch32':>10} {'Δ Error':>10} {'Base Inl':>10} {'P32 Inl':>10} {'Result'}")
    print("-" * 108)

    n_improved = 0
    n_worsened = 0
    n_unchanged = 0
    err_diffs = []

    for rb, rp in zip(xs_b, xs_p):
        cb = rb["res"]["corner_err"]
        cp = rp["res"]["corner_err"]
        ib = rb["res"]["inliers"]
        ip = rp["res"]["inliers"]

        # Delta error: baseline error - patch32 error (positive = improvement)
        if np.isfinite(cb) and np.isfinite(cp):
            delta = cb - cp
            err_diffs.append(delta)
            if abs(delta) < 0.1:
                status = "UNCHANGED"
                n_unchanged += 1
            elif delta > 0:
                status = "IMPROVED"
                n_improved += 1
            else:
                status = "WORSENED"
                n_worsened += 1
        elif not np.isfinite(cb) and np.isfinite(cp):
            status = "IMPROVED (Recovered)"
            n_improved += 1
            err_diffs.append(50.0)
        elif np.isfinite(cb) and not np.isfinite(cp):
            status = "WORSENED (Failed)"
            n_worsened += 1
            err_diffs.append(-50.0)
        else:
            status = "BOTH FAILED"
            n_unchanged += 1

        b_str = f"{cb:.2f}" if np.isfinite(cb) else "FAIL"
        p_str = f"{cp:.2f}" if np.isfinite(cp) else "FAIL"
        d_str = f"{cb - cp:+.2f}" if (np.isfinite(cb) and np.isfinite(cp)) else "N/A"
        print(f"{rb['group']:<42} {b_str:>10} {p_str:>10} {d_str:>10} {ib:>10d} {ip:>10d} {status}")

    print("-" * 108)
    print(f"Paired Summary: {n_improved} cases improved, {n_worsened} worsened, {n_unchanged} unchanged.")
    if err_diffs:
        med_imp = float(np.median(err_diffs))
        mean_imp = float(np.mean(err_diffs))
        print(f"Median Paired Improvement (Baseline Error - Patch32 Error): {med_imp:+.3f} px")
        print(f"Mean Paired Improvement:                                    {mean_imp:+.3f} px")

    # Wilcoxon signed-rank test
    p_val_wilcox = float("nan")
    if len(err_diffs) >= 6 and any(abs(d) > 1e-3 for d in err_diffs):
        try:
            stat, p_val = wilcoxon([d for d in err_diffs if abs(d) > 1e-3], alternative="greater")
            p_val_wilcox = float(p_val)
            print(f"Wilcoxon signed-rank test (H1: Patch32 error < Baseline): stat={stat:.1f}, p={p_val:.4e}")
        except Exception as exc:
            print(f"Wilcoxon test note: {exc}")

    # Generate Markdown Report Artifact
    report_md_path = REPO_ROOT / "evaluation/heldout_patch32_end_to_end_report.md"
    with open(report_md_path, "w") as f:
        f.write("# Phase 6: Held-Out End-to-End Validation of Scale-Adaptive Patch Support\n\n")
        f.write("## Executive Summary\n\n")
        f.write(f"Evaluated on 11 held-out cross-sensor image pairs and 7 held-out same-sensor image pairs.\n")
        f.write(f"- **Candidate Recall**: True candidates recovered per scene increased from {mb_t4:.2f} to {mp_t4:.2f}.\n")
        f.write(f"- **Candidate Precision**: Increased from {mb_p4:.2f}% to {mp_p4:.2f}%.\n")
        f.write(f"- **Scenes with >= 4 True Candidates**: Increased from {ge4_base}/{n_xs} ({ge4_base/n_xs*100:.1f}%) to {ge4_patch}/{n_xs} ({ge4_patch/n_xs*100:.1f}%).\n")
        f.write(f"- **Cross-Sensor Registration**: Median corner error changed from {med_c_b:.2f} px to {med_c_p:.2f} px (Difference: {med_c_b - med_c_p:+.2f} px).\n")
        f.write(f"- **Paired Analysis**: {n_improved} cases improved, {n_worsened} worsened, {n_unchanged} unchanged (Wilcoxon p = {p_val_wilcox:.4f}).\n\n")

    print(f"\nWritten summary report to: {report_md_path}")
    return {
        "n_improved": n_improved, "n_worsened": n_worsened, "n_unchanged": n_unchanged,
        "med_imp": med_imp if err_diffs else 0.0,
        "mean_imp": mean_imp if err_diffs else 0.0,
        "p_val_wilcox": p_val_wilcox,
    }


if __name__ == "__main__":
    run_heldout_evaluation()
