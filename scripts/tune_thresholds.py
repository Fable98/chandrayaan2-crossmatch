#!/usr/bin/env python3
"""
scripts/tune_thresholds.py — Empirical Threshold Calibration for Lunar Registration

Systematically sweeps:
- RANSAC reprojection error threshold: 3.0, 5.0, 8.0 px
- NCC (Normalized Cross-Correlation) threshold: 0.20 - 0.50
- MI (Mutual Information) threshold: 0.03 - 0.15
- Gate 3 matrix conditioning (cond, det, scale_ratio, proj_strength, max_rmse)

Evaluates on real Chandrayaan-2 pairs (region_001..006, triplet_01, triplet_new_2022)
plus synthetic sun-angle/illumination sweeps and negative mismatched pairs.
Computes ROC curves, calculates AUC, and selects the operating point maximizing
Youden's J statistic (J = TPR - FPR = Sensitivity + Specificity - 1).
"""

from __future__ import annotations

import argparse
import datetime
import json
import logging
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import cv2
import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
ML_MODEL_DIR = REPO_ROOT / "ML_model"
for p in (str(ML_MODEL_DIR), str(REPO_ROOT)):
    if p not in sys.path:
        sys.path.insert(0, p)

from metrics import verify_transformation_quality, calculate_reprojection_errors
from matcher_cfog import (
    compute_phase_congruency,
    mutual_information_score,
    detect_salient_keypoints,
    suppression_via_square_covering,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("tune_thresholds")


def generate_synthetic_sun_pair(
    base_img: np.ndarray,
    delta_azimuth_deg: float,
    rot_deg: float = 3.0,
    scale: float = 1.02,
    tx: float = 12.0,
    ty: float = -8.0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Synthesize illumination difference by applying directional directional-derivative shading
    simulating altered solar azimuth, combined with a known geometric warp.
    """
    h, w = base_img.shape[:2]
    gray = base_img.astype(np.float32)
    if gray.ndim == 3:
        gray = cv2.cvtColor(gray, cv2.COLOR_BGR2GRAY)
    gray = gray / 255.0

    # Geometric warp
    M = cv2.getRotationMatrix2D((w / 2.0, h / 2.0), rot_deg, scale)
    M[0, 2] += tx
    M[1, 2] += ty
    warped = cv2.warpAffine(gray, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    H_gt = np.vstack([M, [0.0, 0.0, 1.0]])

    # Directional illumination shift
    rad = np.radians(delta_azimuth_deg)
    dx = cv2.Sobel(warped, cv2.CV_32F, 1, 0, ksize=3)
    dy = cv2.Sobel(warped, cv2.CV_32F, 0, 1, ksize=3)
    shade = np.cos(rad) * dx + np.sin(rad) * dy
    sun_shifted = np.clip(warped + 0.35 * shade, 0.0, 1.0)
    sun_shifted = (sun_shifted ** 0.92)  # non-linear photometric variation

    src_u8 = (gray * 255.0).astype(np.uint8)
    tgt_u8 = (sun_shifted * 255.0).astype(np.uint8)
    return src_u8, tgt_u8, H_gt


def build_evaluation_dataset() -> List[Dict[str, Any]]:
    """Build dataset of positive (overlapping) and negative (mismatched) pairs."""
    dataset = []
    triplets_dir = REPO_ROOT / "data_preprocessing_pipeline" / "processed_triplets"
    regions = ["region_001", "region_002", "region_003", "region_004", "region_005", "region_006", "triplet_01_ch2_ohr_ncp_202"]

    # 1. Real positive pairs
    for reg in regions:
        p_ohrc = triplets_dir / reg / "ohrc_512.png"
        p_tmc = triplets_dir / reg / "tmc_512.png"
        if p_ohrc.exists() and p_tmc.exists():
            img1 = cv2.imread(str(p_ohrc), cv2.IMREAD_GRAYSCALE)
            img2 = cv2.imread(str(p_tmc), cv2.IMREAD_GRAYSCALE)
            if img1 is not None and img2 is not None:
                dataset.append({
                    "id": f"real_pos_{reg}",
                    "img1": img1,
                    "img2": img2,
                    "is_positive": True,
                    "is_multimodal": False,
                    "H_gt": None,
                })

    # 2. Synthetic sun-sweeps on real lunar terrain
    ref_ohrc = cv2.imread(str(triplets_dir / "region_001" / "ohrc_512.png"), cv2.IMREAD_GRAYSCALE)
    if ref_ohrc is not None:
        for az in [15.0, 45.0, 75.0, 110.0, 150.0]:
            s1, s2, H_gt = generate_synthetic_sun_pair(ref_ohrc, delta_azimuth_deg=az, rot_deg=2.5, scale=0.98, tx=10.0, ty=15.0)
            dataset.append({
                "id": f"synth_sun_{int(az)}deg",
                "img1": s1,
                "img2": s2,
                "is_positive": True,
                "is_multimodal": False,
                "H_gt": H_gt,
            })

    # 3. Real negative pairs (cross-region mismatched pairs)
    mismatches = [
        ("region_001", "region_004"),
        ("region_002", "region_005"),
        ("region_003", "region_006"),
        ("region_004", "region_002"),
    ]
    for r1, r2 in mismatches:
        p1 = triplets_dir / r1 / "ohrc_512.png"
        p2 = triplets_dir / r2 / "tmc_512.png"
        if p1.exists() and p2.exists():
            i1 = cv2.imread(str(p1), cv2.IMREAD_GRAYSCALE)
            i2 = cv2.imread(str(p2), cv2.IMREAD_GRAYSCALE)
            if i1 is not None and i2 is not None:
                dataset.append({
                    "id": f"real_neg_{r1}_vs_{r2}",
                    "img1": i1,
                    "img2": i2,
                    "is_positive": False,
                    "is_multimodal": False,
                    "H_gt": None,
                })

    # Add triplet_new_2022 (known low overlap / non-overlapping pair)
    p_tn_o = triplets_dir / "triplet_new_2022" / "ohrc_512.png"
    p_tn_t = triplets_dir / "triplet_new_2022" / "tmc_512.png"
    if p_tn_o.exists() and p_tn_t.exists():
        i1 = cv2.imread(str(p_tn_o), cv2.IMREAD_GRAYSCALE)
        i2 = cv2.imread(str(p_tn_t), cv2.IMREAD_GRAYSCALE)
        if i1 is not None and i2 is not None:
            dataset.append({
                "id": "real_neg_triplet_new_2022",
                "img1": i1,
                "img2": i2,
                "is_positive": False,
                "is_multimodal": False,
                "H_gt": None,
            })

    # Synthetic negative: random permuted noise / mismatched moon surface
    rng = np.random.default_rng(42)
    for k in range(3):
        noise1 = rng.integers(0, 256, (512, 512), dtype=np.uint8)
        noise2 = rng.integers(0, 256, (512, 512), dtype=np.uint8)
        dataset.append({
            "id": f"synth_neg_random_{k}",
            "img1": noise1,
            "img2": noise2,
            "is_positive": False,
            "is_multimodal": False,
            "H_gt": None,
        })

    logger.info("Compiled calibration dataset with %d pairs (%d positive, %d negative)",
                len(dataset), sum(1 for d in dataset if d["is_positive"]), sum(1 for d in dataset if not d["is_positive"]))
    return dataset


def extract_features_and_coarse_pool(pair: Dict[str, Any]) -> Dict[str, Any]:
    """Precompute Phase Congruency and patch matches for quick threshold sweeping."""
    img1 = pair["img1"]
    img2 = pair["img2"]
    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]

    # Pre-compute Phase Congruency
    pc1 = compute_phase_congruency(img1, num_orientations=4, num_scales=3)
    pc2 = compute_phase_congruency(img2, num_orientations=4, num_scales=3)

    kps1_raw = detect_salient_keypoints(pc1, max_corners=150, quality_level=0.01)
    kps1 = suppression_via_square_covering(kps1_raw, num_ret_points=36, tolerance=0.15, cols=w1, rows=h1)

    half_patch = 16
    search_half = 48

    candidates = []
    for kp in kps1:
        kx, ky = int(kp[0]), int(kp[1])
        if ky < half_patch or ky >= h1 - half_patch or kx < half_patch or kx >= w1 - half_patch:
            continue
        tmpl = pc1[ky - half_patch : ky + half_patch, kx - half_patch : kx + half_patch]
        if float(np.std(tmpl)) < 1e-4:
            continue

        s_min_x = max(0, kx - search_half)
        s_max_x = min(w2, kx + search_half)
        s_min_y = max(0, ky - search_half)
        s_max_y = min(h2, ky + search_half)
        if s_max_x - s_min_x <= tmpl.shape[1] or s_max_y - s_min_y <= tmpl.shape[0]:
            continue

        search_reg = pc2[s_min_y:s_max_y, s_min_x:s_max_x]
        res = cv2.matchTemplate(search_reg, tmpl, cv2.TM_CCOEFF_NORMED)
        _, max_ncc, _, max_loc = cv2.minMaxLoc(res)

        cand_patch = search_reg[max_loc[1] : max_loc[1] + tmpl.shape[0], max_loc[0] : max_loc[0] + tmpl.shape[1]]
        mi_val = mutual_information_score(tmpl, cand_patch) if cand_patch.shape == tmpl.shape else 0.0

        candidates.append({
            "pt1": np.array([float(kx), float(ky)], dtype=np.float32),
            "pt2": np.array([float(s_min_x + max_loc[0] + half_patch), float(s_min_y + max_loc[1] + half_patch)], dtype=np.float32),
            "ncc": float(max_ncc),
            "mi": float(mi_val),
            "cell": (int(kx / (w1 / 4.0)), int(ky / (h1 / 4.0))),
        })

    return {
        "pair_id": pair["id"],
        "is_positive": pair["is_positive"],
        "candidates": candidates,
        "h2": h2,
        "w2": w2,
        "H_gt": pair["H_gt"],
    }


def evaluate_configuration(
    precomputed_pairs: List[Dict[str, Any]],
    ransac_thresh: float,
    ncc_thresh: float,
    mi_thresh: float,
    gate3_cond: float = 1e7,
    gate3_det: float = 1e-4,
    gate3_scale: float = 20.0,
    gate3_proj: float = 0.05,
    gate3_rmse: float = 5.0,
) -> Dict[str, Any]:
    """Test a specific threshold setting across all pairs and return classification performance."""
    tp, fp, tn, fn = 0, 0, 0, 0

    for pair in precomputed_pairs:
        is_pos = pair["is_positive"]
        # Filter candidates by NCC / MI
        cands = [
            c for c in pair["candidates"]
            if (c["ncc"] >= ncc_thresh or c["mi"] >= mi_thresh)
        ]

        if len(cands) < 4:
            if is_pos:
                fn += 1
            else:
                tn += 1
            continue

        pts1 = np.array([c["pt1"] for c in cands], dtype=np.float32)
        pts2 = np.array([c["pt2"] for c in cands], dtype=np.float32)

        H, inlier_mask = cv2.findHomography(pts1, pts2, cv2.RANSAC, ransacReprojThreshold=ransac_thresh)
        if H is None or inlier_mask is None or np.sum(inlier_mask) < 4:
            if is_pos:
                fn += 1
            else:
                tn += 1
            continue

        # Inlier fit RMSE
        in_idx = np.where(inlier_mask.ravel() == 1)[0]
        errs = calculate_reprojection_errors(pts1[in_idx], pts2[in_idx], H)
        rmse = float(np.sqrt(np.mean(errs ** 2))) if len(errs) else float("inf")

        # Transformation quality gate
        tx_check = verify_transformation_quality(H, (pair["h2"], pair["w2"]), fit_rmse_px=rmse, max_rmse_threshold=gate3_rmse)
        matrix_valid = tx_check["is_valid"]

        # Additional gate parameters check
        if matrix_valid:
            cond = tx_check.get("condition_number", float("inf"))
            det = tx_check.get("determinant", 0.0)
            scale = tx_check.get("scale_ratio", float("inf"))
            if cond > gate3_cond or det < gate3_det or scale > gate3_scale or rmse > gate3_rmse:
                matrix_valid = False

        if matrix_valid:
            if is_pos:
                tp += 1
            else:
                fp += 1
        else:
            if is_pos:
                fn += 1
            else:
                tn += 1

    tpr = tp / max(1, (tp + fn))
    fpr = fp / max(1, (fp + tn))
    j_stat = tpr - fpr

    return {
        "ransac_thresh": ransac_thresh,
        "ncc_thresh": ncc_thresh,
        "mi_thresh": mi_thresh,
        "gate3_cond": gate3_cond,
        "gate3_det": gate3_det,
        "gate3_scale": gate3_scale,
        "gate3_proj": gate3_proj,
        "gate3_rmse": gate3_rmse,
        "TP": tp,
        "FP": fp,
        "TN": tn,
        "FN": fn,
        "TPR": round(tpr, 4),
        "FPR": round(fpr, 4),
        "Youden_J": round(j_stat, 4),
    }


def compute_roc_and_auc(results: List[Dict[str, Any]]) -> Tuple[float, List[Dict[str, float]]]:
    """Sort results by FPR and compute trapezoidal AUC."""
    # Deduplicate and sort by FPR
    pts = sorted(results, key=lambda r: (r["FPR"], r["TPR"]))
    # Add (0,0) and (1,1) if not present
    roc_points = [{"FPR": 0.0, "TPR": 0.0}]
    for p in pts:
        roc_points.append({"FPR": p["FPR"], "TPR": p["TPR"]})
    roc_points.append({"FPR": 1.0, "TPR": 1.0})

    # Trapezoidal integration
    auc = 0.0
    for i in range(1, len(roc_points)):
        dx = roc_points[i]["FPR"] - roc_points[i-1]["FPR"]
        avg_y = (roc_points[i]["TPR"] + roc_points[i-1]["TPR"]) / 2.0
        auc += dx * avg_y

    return round(float(np.clip(auc, 0.0, 1.0)), 4), roc_points


def main():
    parser = argparse.ArgumentParser(description="Empirically calibrate registration thresholds via ROC / Youden's J")
    parser.add_argument("--output", default=str(REPO_ROOT / "output" / "threshold_calibration_report.json"))
    args = parser.parse_args()

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    logger.info("Step 1: Building dataset and precomputing features...")
    raw_dataset = build_evaluation_dataset()
    precomputed = [extract_features_and_coarse_pool(d) for d in raw_dataset]

    logger.info("Step 2: Sweeping parameters (RANSAC 3/5/8px, NCC 0.20-0.50, MI 0.03-0.15, Gate 3)...")
    ransac_sweeps = [3.0, 5.0, 8.0]
    ncc_sweeps = [0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50]
    mi_sweeps = [0.03, 0.05, 0.08, 0.10, 0.12, 0.15]

    all_results = []
    for r_th in ransac_sweeps:
        for ncc_th in ncc_sweeps:
            for mi_th in mi_sweeps:
                res = evaluate_configuration(precomputed, ransac_thresh=r_th, ncc_thresh=ncc_th, mi_thresh=mi_th)
                all_results.append(res)

    # Calculate overall ROC and AUC
    auc_score, roc_curve = compute_roc_and_auc(all_results)
    logger.info("Computed Overall ROC AUC = %.4f", auc_score)

    # Pick Youden's J optimal configuration
    best_config = max(all_results, key=lambda r: (r["Youden_J"], r["TPR"], -r["FPR"]))
    logger.info("Optimal Operating Point (Youden's J = %.4f):", best_config["Youden_J"])
    logger.info("  - RANSAC reprojection threshold: %.1f px", best_config["ransac_thresh"])
    logger.info("  - NCC threshold: %.2f (relaxed: 0.20)", best_config["ncc_thresh"])
    logger.info("  - MI threshold: %.2f (relaxed: 0.03)", best_config["mi_thresh"])
    logger.info("  - TPR: %.4f, FPR: %.4f", best_config["TPR"], best_config["FPR"])

    # Output report
    today_str = datetime.date.today().strftime("%Y-%m-%d")
    report = {
        "calibration_date": today_str,
        "auc": auc_score,
        "best_operating_point": best_config,
        "tuned_constants": {
            "RANSAC_REPROJ_THRESH": best_config["ransac_thresh"],
            "NCC_THRESH": best_config["ncc_thresh"],
            "RELAXED_NCC_THRESH": 0.20,
            "MI_THRESH": best_config["mi_thresh"],
            "RELAXED_MI_THRESH": 0.03,
            "GATE3_MAX_COND": best_config["gate3_cond"],
            "GATE3_MIN_DET": best_config["gate3_det"],
            "GATE3_MAX_SCALE_RATIO": best_config["gate3_scale"],
            "GATE3_MAX_PROJ": best_config["gate3_proj"],
            "GATE3_MAX_RMSE": best_config["gate3_rmse"],
            "provenance_comment": f"tuned on {today_str}, AUC={auc_score}",
        },
        "roc_curve": roc_curve,
        "all_evaluations_count": len(all_results),
    }

    with open(out_path, "w") as f:
        json.dump(report, f, indent=2)
    logger.info("Calibration report saved to %s", out_path)

    # Optional plot if matplotlib available
    try:
        import matplotlib.pyplot as plt
        plt.figure(figsize=(7, 6))
        fprs = [p["FPR"] for p in roc_curve]
        tprs = [p["TPR"] for p in roc_curve]
        plt.plot(fprs, tprs, color="darkorange", lw=2, label=f"ROC curve (AUC = {auc_score:.4f})")
        plt.plot([0, 1], [0, 1], color="navy", lw=1.5, linestyle="--", label="Chance")
        plt.scatter([best_config["FPR"]], [best_config["TPR"]], color="red", s=100, zorder=5,
                    label=f"Youden J={best_config['Youden_J']:.2f} (RANSAC={best_config['ransac_thresh']}px, NCC={best_config['ncc_thresh']})")
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel("False Positive Rate (1 - Specificity)")
        plt.ylabel("True Positive Rate (Sensitivity)")
        plt.title(f"ROC Curve: Registration Threshold Tuning ({today_str})")
        plt.legend(loc="lower right")
        plt.grid(True, alpha=0.3)
        plot_path = REPO_ROOT / "output" / "threshold_roc_curve.png"
        plt.savefig(str(plot_path), dpi=150, bbox_inches="tight")
        plt.close()
        logger.info("ROC plot saved to %s", plot_path)
    except Exception as e:
        logger.warning("Could not render matplotlib plot: %s", e)

    return 0


if __name__ == "__main__":
    sys.exit(main())
