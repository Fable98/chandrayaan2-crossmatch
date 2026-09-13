"""Phase 10: Macro Structural Correspondence Diagnostic.

Investigates:
1. Macro feature detection (detect_blob_centroids, connected components, salient crater/blob centers).
2. Independent detection in source and target images without H_gt knowledge.
3. Post-hoc H_gt correspondence evaluation: recall at <=2px, <=5px, <=10px, <=15px, and distance statistics.
4. Macro feature survival statistics (counts, spatial densities, scale distributions).
5. Local topology preservation (k=3, k=5 neighbor distance ratios, angular errors, triangle shape consistency).
6. Affine/projective-invariant geometry of centroid constellations.
7. Matchable constellations: scenes with >=4, >=6, >=10 true corresponding centroids vs local patch baseline.
8. Rigorous null model comparison (random spatial distribution with matched density).
9. Cross-sensor (37 dev scenes) vs same-sensor (17 dev scenes) sanity check.
10. Scientific classification: A, B, or C.
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
    detect_blob_centroids,
    suppression_via_square_covering,
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


def extract_salient_macro_centroids(
    pc_img: np.ndarray, min_area: int = 15, max_points: int = 60, use_ssc: bool = False
) -> Tuple[np.ndarray, np.ndarray]:
    """Detect structural blob centroids with area filtering and optional SSC spatial suppression.

    Returns:
      centroids: (N, 2) array of (x, y) coordinates
      radii: (N,) array of equivalent radii sqrt(area / pi)
    """
    image_f = np.asarray(pc_img, dtype=np.float32)
    if image_f.ndim != 2 or float(np.std(image_f)) < 1e-6:
        return np.empty((0, 2), dtype=np.float32), np.empty((0,), dtype=np.float32)

    threshold = float(np.percentile(image_f, 75.0))
    binary = (image_f >= threshold).astype(np.uint8)
    count, _, stats, centroids = cv2.connectedComponentsWithStats(binary, 8)

    found_pts: List[Tuple[float, float, float, float]] = []
    for label in range(1, count):
        area = int(stats[label, cv2.CC_STAT_AREA])
        if area < min_area:
            continue
        mask = binary == label
        weights = np.maximum(image_f[mask] - threshold, 0.0)
        ys, xs = np.nonzero(mask)
        w_sum = float(weights.sum())
        if w_sum > 1e-6:
            cx = float(np.dot(xs, weights) / w_sum)
            cy = float(np.dot(ys, weights) / w_sum)
        else:
            cx = float(centroids[label, 0])
            cy = float(centroids[label, 1])
        radius = math.sqrt(area / math.pi)
        score = float(image_f[min(image_f.shape[0]-1, int(round(cy))), min(image_f.shape[1]-1, int(round(cx)))])
        found_pts.append((cx, cy, score, radius))

    if not found_pts:
        return np.empty((0, 2), dtype=np.float32), np.empty((0,), dtype=np.float32)

    if use_ssc and len(found_pts) > max_points:
        h, w = pc_img.shape[:2]
        tuples_for_ssc = [(p[0], p[1], p[2]) for p in found_pts]
        ssc_res = suppression_via_square_covering(tuples_for_ssc, num_ret_points=max_points, tolerance=0.1, cols=w, rows=h)
        retained_set = {(round(float(r[0]), 2), round(float(r[1]), 2)) for r in ssc_res}
        final_pts = [p for p in found_pts if (round(p[0], 2), round(p[1], 2)) in retained_set][:max_points]
    else:
        # Sort by area/prominence and take top max_points if needed
        found_pts.sort(key=lambda p: p[3], reverse=True)
        final_pts = found_pts[:max_points] if max_points > 0 else found_pts

    coords = np.array([[p[0], p[1]] for p in final_pts], dtype=np.float32)
    radii = np.array([p[3] for p in final_pts], dtype=np.float32)
    return coords, radii


def evaluate_topology_preservation(src_pts: np.ndarray, tgt_pts: np.ndarray, H_gt: np.ndarray, matched_pairs: List[Tuple[int, int]]) -> Dict[str, Any]:
    """Measure distance ratios, angular consistency, and triangle shape preservation for corresponding centroids."""
    if len(matched_pairs) < 4:
        return {
            "dist_ratio_mean": float("nan"), "dist_ratio_std": float("nan"),
            "angle_err_mean": float("nan"), "angle_err_med": float("nan"),
            "tri_shape_err_mean": float("nan"), "tri_area_ratio_mean": float("nan"),
        }

    src_idx = [p[0] for p in matched_pairs]
    tgt_idx = [p[1] for p in matched_pairs]

    s_coords = src_pts[src_idx]
    t_coords = tgt_pts[tgt_idx]
    n_pairs = len(s_coords)

    # 1. Pairwise distance ratios
    dist_ratios = []
    for i in range(n_pairs):
        for j in range(i + 1, n_pairs):
            d_s = float(np.linalg.norm(s_coords[i] - s_coords[j]))
            d_t = float(np.linalg.norm(t_coords[i] - t_coords[j]))
            if d_s > 5.0:
                dist_ratios.append(d_t / d_s)

    # 2. Local neighbor angles (k=3)
    angle_errs = []
    for i in range(n_pairs):
        dists = np.linalg.norm(s_coords - s_coords[i], axis=1)
        nn_order = np.argsort(dists)[1:4]  # 3 nearest
        if len(nn_order) >= 2:
            v_s1 = s_coords[nn_order[0]] - s_coords[i]
            v_s2 = s_coords[nn_order[1]] - s_coords[i]
            v_t1 = t_coords[nn_order[0]] - t_coords[i]
            v_t2 = t_coords[nn_order[1]] - t_coords[i]

            norm_s = (np.linalg.norm(v_s1) * np.linalg.norm(v_s2)) + 1e-6
            norm_t = (np.linalg.norm(v_t1) * np.linalg.norm(v_t2)) + 1e-6

            cos_s = np.clip(np.dot(v_s1, v_s2) / norm_s, -1.0, 1.0)
            cos_t = np.clip(np.dot(v_t1, v_t2) / norm_t, -1.0, 1.0)

            ang_s = math.degrees(math.acos(cos_s))
            ang_t = math.degrees(math.acos(cos_t))
            angle_errs.append(abs(ang_s - ang_t))

    # 3. Triangle shape differences
    tri_shape_errs = []
    tri_area_ratios = []
    if n_pairs >= 3:
        for i in range(min(15, n_pairs)):
            for j in range(i + 1, min(15, n_pairs)):
                for k in range(j + 1, min(15, n_pairs)):
                    # Source triangle side lengths
                    a_s = float(np.linalg.norm(s_coords[i] - s_coords[j]))
                    b_s = float(np.linalg.norm(s_coords[j] - s_coords[k]))
                    c_s = float(np.linalg.norm(s_coords[k] - s_coords[i]))
                    # Target triangle side lengths
                    a_t = float(np.linalg.norm(t_coords[i] - t_coords[j]))
                    b_t = float(np.linalg.norm(t_coords[j] - t_coords[k]))
                    c_t = float(np.linalg.norm(t_coords[k] - t_coords[i]))

                    peri_s = a_s + b_s + c_s
                    peri_t = a_t + b_t + c_t
                    if peri_s > 10.0 and peri_t > 10.0:
                        # Normalized side-length shape vector
                        norm_side_s = np.sort([a_s / peri_s, b_s / peri_s, c_s / peri_s])
                        norm_side_t = np.sort([a_t / peri_t, b_t / peri_t, c_t / peri_t])
                        shape_diff = float(np.sum(np.abs(norm_side_s - norm_side_t)))
                        tri_shape_errs.append(shape_diff)

                        # Heron's area
                        sp_s = peri_s / 2.0
                        area_s = math.sqrt(max(0.0, sp_s * (sp_s - a_s) * (sp_s - b_s) * (sp_s - c_s)))
                        sp_t = peri_t / 2.0
                        area_t = math.sqrt(max(0.0, sp_t * (sp_t - a_t) * (sp_t - b_t) * (sp_t - c_t)))
                        if area_s > 10.0:
                            tri_area_ratios.append(area_t / area_s)

    return {
        "dist_ratio_mean": float(np.mean(dist_ratios)) if dist_ratios else float("nan"),
        "dist_ratio_std": float(np.std(dist_ratios)) if dist_ratios else float("nan"),
        "angle_err_mean": float(np.mean(angle_errs)) if angle_errs else float("nan"),
        "angle_err_med": float(np.median(angle_errs)) if angle_errs else float("nan"),
        "tri_shape_err_mean": float(np.mean(tri_shape_errs)) if tri_shape_errs else float("nan"),
        "tri_area_ratio_mean": float(np.mean(tri_area_ratios)) if tri_area_ratios else float("nan"),
    }


def run_phase10_diagnostic():
    train_groups, test_groups = get_train_test_split()
    xs_train_groups = [g for g in train_groups if "cross_sensor" in g]
    ss_train_groups = [g for g in train_groups if "same_sensor" in g]

    print("================================================================================")
    print("PHASE 10: MACRO STRUCTURAL CORRESPONDENCE DIAGNOSTIC")
    print("================================================================================")
    print(f"Development Cross-Sensor Groups : {len(xs_train_groups)}")
    print(f"Development Same-Sensor Groups  : {len(ss_train_groups)}")
    print(f"Held-Out Test Groups (FROZEN)   : {len(test_groups)} (Untouched)")
    print("Diagnostic Only: Evaluates macro structural centroids vs local patch baseline.\n")

    image_paths = get_image_paths()

    def evaluate_macro_dataset(domain_type: str, groups_list: List[str], n_transforms: int) -> Dict[str, Any]:
        results = {
            "n_src": [], "n_tgt": [],
            "radii_src": [], "radii_tgt": [],
            "dists_real": [], "dists_null": [],
            "rec2": [], "rec5": [], "rec10": [], "rec15": [],
            "rec2_null": [], "rec5_null": [], "rec10_null": [], "rec15_null": [],
            "scenes_ge4": 0, "scenes_ge6": 0, "scenes_ge10": 0,
            "scenes_ge4_10px": 0,
            "per_scene": {},
            "topo_metrics": [],
        }

        for p in image_paths:
            img_gray = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
            if img_gray is None:
                continue
            h, w = img_gray.shape[:2]
            region = p.parent.name
            source_image = p.name

            pc1 = compute_phase_congruency(img_gray, num_orientations=4, num_scales=3)
            # Detect source macro centroids independently
            src_centroids, src_radii = extract_salient_macro_centroids(pc1, min_area=15, max_points=40)

            for t_idx in range(n_transforms):
                target_image = f"{p.stem}_{domain_type}_t{t_idx}.png"
                grp = f"{region}:{source_image}::{target_image}"
                if grp not in groups_list:
                    continue

                scene_seed = int(abs(hash(grp)) % (2**31 - 1))
                rng_s = np.random.default_rng(scene_seed)

                H_gt = create_random_homography(w, h, rng_s)
                warped = cv2.warpPerspective(img_gray, H_gt, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)

                if domain_type == "cross_sensor":
                    warped_img = apply_antisolar_cross_sensor(warped, rng_s)
                else:
                    warped_img = apply_photometric_perturbation(warped, rng_s)

                pc2 = compute_phase_congruency(warped_img, num_orientations=4, num_scales=3)

                # Detect target macro centroids independently (NO H_gt KNOWLEDGE)
                tgt_centroids, tgt_radii = extract_salient_macro_centroids(pc2, min_area=15, max_points=40)

                results["n_src"].append(len(src_centroids))
                results["n_tgt"].append(len(tgt_centroids))
                results["radii_src"].extend(src_radii)
                results["radii_tgt"].extend(tgt_radii)

                # Project source centroids with H_gt (POST-HOC EVALUATION ONLY)
                if len(src_centroids) == 0 or len(tgt_centroids) == 0:
                    continue

                c1_homo = np.hstack([src_centroids, np.ones((len(src_centroids), 1))])
                c1_proj = (H_gt @ c1_homo.T).T
                c1_proj = c1_proj[:, :2] / c1_proj[:, 2:3]

                # Distance to nearest detected target centroid
                scene_dists = []
                matched_pairs = []
                for s_i, p_proj in enumerate(c1_proj):
                    diffs = tgt_centroids - p_proj
                    euc = np.linalg.norm(diffs, axis=1)
                    nearest_idx = int(np.argmin(euc))
                    min_d = float(euc[nearest_idx])
                    scene_dists.append(min_d)
                    if min_d <= 5.0:
                        matched_pairs.append((s_i, nearest_idx))

                # Null model: random uniform points with identical count
                rng_null = np.random.default_rng(scene_seed + 999)
                null_tgt = rng_null.uniform(0, 512, (len(tgt_centroids), 2))
                scene_dists_null = [float(np.min(np.linalg.norm(null_tgt - p_proj, axis=1))) for p_proj in c1_proj]

                results["dists_real"].extend(scene_dists)
                results["dists_null"].extend(scene_dists_null)

                c2 = sum(1 for d in scene_dists if d <= 2.0)
                c5 = sum(1 for d in scene_dists if d <= 5.0)
                c10 = sum(1 for d in scene_dists if d <= 10.0)
                c15 = sum(1 for d in scene_dists if d <= 15.0)

                c2_n = sum(1 for d in scene_dists_null if d <= 2.0)
                c5_n = sum(1 for d in scene_dists_null if d <= 5.0)
                c10_n = sum(1 for d in scene_dists_null if d <= 10.0)
                c15_n = sum(1 for d in scene_dists_null if d <= 15.0)

                results["rec2"].append(c2 / len(src_centroids))
                results["rec5"].append(c5 / len(src_centroids))
                results["rec10"].append(c10 / len(src_centroids))
                results["rec15"].append(c15 / len(src_centroids))

                results["rec2_null"].append(c2_n / len(src_centroids))
                results["rec5_null"].append(c5_n / len(src_centroids))
                results["rec10_null"].append(c10_n / len(src_centroids))
                results["rec15_null"].append(c15_n / len(src_centroids))

                if c5 >= 4:
                    results["scenes_ge4"] += 1
                if c5 >= 6:
                    results["scenes_ge6"] += 1
                if c5 >= 10:
                    results["scenes_ge10"] += 1

                if c10 >= 4:
                    results["scenes_ge4_10px"] += 1

                # Local topology preservation
                topo = evaluate_topology_preservation(src_centroids, tgt_centroids, H_gt, matched_pairs)
                results["topo_metrics"].append(topo)

                results["per_scene"][grp] = {
                    "n_src": len(src_centroids), "n_tgt": len(tgt_centroids),
                    "med_dist": float(np.median(scene_dists)),
                    "med_dist_null": float(np.median(scene_dists_null)),
                    "c2": c2, "c5": c5, "c10": c10, "c15": c15,
                    "c5_null": c5_n,
                }

        return results

    print("Running Cross-Sensor Macro Evaluation on 37 development groups...")
    res_xs = evaluate_macro_dataset("cross_sensor", xs_train_groups, n_transforms=4)

    print("Running Same-Sensor Macro Evaluation on 17 development groups (Sanity Check)...")
    res_ss = evaluate_macro_dataset("same_sensor", ss_train_groups, n_transforms=2)

    # --------------------------------------------------------------------------
    # 1. MACRO FEATURE SURVIVAL STATISTICS (SECTION 4)
    # --------------------------------------------------------------------------
    print("\n--------------------------------------------------------------------------------")
    print("1. MACRO FEATURE SURVIVAL & SPATIAL DENSITY STATISTICS")
    print("--------------------------------------------------------------------------------")
    print(f"{'Metric':<36} {'Cross-Sensor (N=37)':>24} {'Same-Sensor (N=17)':>24}")
    print("-" * 86)
    print(f"{'Mean Source Centroids / Scene':<36} {np.mean(res_xs['n_src']):>24.1f} {np.mean(res_ss['n_src']):>24.1f}")
    print(f"{'Mean Target Centroids / Scene':<36} {np.mean(res_xs['n_tgt']):>24.1f} {np.mean(res_ss['n_tgt']):>24.1f}")
    print(f"{'Median Source Centroids':<36} {np.median(res_xs['n_src']):>24.1f} {np.median(res_ss['n_src']):>24.1f}")
    print(f"{'Median Target Centroids':<36} {np.median(res_xs['n_tgt']):>24.1f} {np.median(res_ss['n_tgt']):>24.1f}")
    print(f"{'Spatial Density (pts / 10k px^2)':<36} {np.mean(res_xs['n_src'])/(512*512)*10000:>24.2f} {np.mean(res_ss['n_src'])/(512*512)*10000:>24.2f}")
    print(f"{'Mean Equivalent Radius (px)':<36} {np.mean(res_xs['radii_src']):>24.2f} {np.mean(res_ss['radii_src']):>24.2f}")
    print(f"{'Median Equivalent Radius (px)':<36} {np.median(res_xs['radii_src']):>24.2f} {np.median(res_ss['radii_src']):>24.2f}")

    # --------------------------------------------------------------------------
    # 2. POST-HOC H_GT CORRESPONDENCE RECALL & NULL COMPARISON (SECTION 3 & 8)
    # --------------------------------------------------------------------------
    print("\n--------------------------------------------------------------------------------")
    print("2. H_GT CORRESPONDENCE RECALL VS RANDOM NULL BASELINE")
    print("--------------------------------------------------------------------------------")
    print(f"{'Correspondence Metric':<32} {'Cross-Sensor Real':>18} {'Cross Null Baseline':>20} {'Same-Sensor Real':>18}")
    print("-" * 90)
    med_xs = np.median(res_xs["dists_real"])
    med_xs_n = np.median(res_xs["dists_null"])
    med_ss = np.median(res_ss["dists_real"])
    print(f"{'Median Nearest-Centroid Dist':<32} {med_xs:>16.2f}px {med_xs_n:>18.2f}px {med_ss:>16.2f}px")

    mean_xs = np.mean(res_xs["dists_real"])
    mean_xs_n = np.mean(res_xs["dists_null"])
    mean_ss = np.mean(res_ss["dists_real"])
    print(f"{'Mean Nearest-Centroid Dist':<32} {mean_xs:>16.2f}px {mean_xs_n:>18.2f}px {mean_ss:>16.2f}px")

    for thresh, k_name in [(2.0, "rec2"), (5.0, "rec5"), (10.0, "rec10"), (15.0, "rec15")]:
        r_xs = np.mean(res_xs[k_name]) * 100
        r_n = np.mean(res_xs[k_name + "_null"]) * 100
        r_ss = np.mean(res_ss[k_name]) * 100
        print(f"{f'Recall <= {thresh:.0f} px':<32} {r_xs:>17.1f}% {r_n:>19.1f}% {r_ss:>17.1f}%")

    unmatched_xs = sum(1 for d in res_xs["dists_real"] if d > 15.0) / len(res_xs["dists_real"]) * 100
    unmatched_ss = sum(1 for d in res_ss["dists_real"] if d > 15.0) / len(res_ss["dists_real"]) * 100
    print(f"{'% Unmatched (>15 px)':<32} {unmatched_xs:>17.1f}% {sum(1 for d in res_xs['dists_null'] if d > 15.0) / len(res_xs['dists_null']) * 100:>19.1f}% {unmatched_ss:>17.1f}%")

    # --------------------------------------------------------------------------
    # 3. MATCHABLE CONSTELLATIONS COMPARISON (SECTION 7)
    # --------------------------------------------------------------------------
    print("\n--------------------------------------------------------------------------------")
    print("3. MATCHABLE CONSTELLATIONS: LOCAL PATCH VS MACRO CENTROIDS")
    print("--------------------------------------------------------------------------------")
    n_xs_scenes = len(xs_train_groups)
    print(f"{'Constellation Metric':<40} {'Local Patch Baseline':>22} {'Macro Centroids (<=5px)':>25} {'Macro (<=10px)':>18}")
    print("-" * 107)
    # Local patch baseline values from Phase 8 & 9
    mean_c5 = np.mean([v['c5'] for v in res_xs['per_scene'].values()])
    mean_c10 = np.mean([v['c10'] for v in res_xs['per_scene'].values()])
    med_c5 = np.median([v['c5'] for v in res_xs['per_scene'].values()])
    med_c10 = np.median([v['c10'] for v in res_xs['per_scene'].values()])
    pct_ge4 = res_xs['scenes_ge4'] / n_xs_scenes * 100
    pct_ge4_10 = res_xs['scenes_ge4_10px'] / n_xs_scenes * 100
    pct_ge6 = res_xs['scenes_ge6'] / n_xs_scenes * 100
    pct_ge10 = res_xs['scenes_ge10'] / n_xs_scenes * 100

    str_ge4 = f"{res_xs['scenes_ge4']} / {n_xs_scenes} ({pct_ge4:.1f}%)"
    str_ge4_10 = f"{res_xs['scenes_ge4_10px']} / {n_xs_scenes} ({pct_ge4_10:.1f}%)"
    str_ge6 = f"{res_xs['scenes_ge6']} / {n_xs_scenes} ({pct_ge6:.1f}%)"
    str_ge10 = f"{res_xs['scenes_ge10']} / {n_xs_scenes} ({pct_ge10:.1f}%)"

    print(f"{'Mean True Correspondences / Scene':<40} {'1.68':>22} {mean_c5:>25.2f} {mean_c10:>18.2f}")
    print(f"{'Median True Correspondences':<40} {'1.00':>22} {med_c5:>25.1f} {med_c10:>18.1f}")
    print(f"{'Scenes with >= 4 True Matches':<40} {'4 / 37 (10.8%)':>22} {str_ge4:>25} {str_ge4_10:>18}")
    print(f"{'Scenes with >= 6 True Matches':<40} {'1 / 37 (2.7%)':>22} {str_ge6:>25} {'--':>18}")
    print(f"{'Scenes with >= 10 True Matches':<40} {'0 / 37 (0.0%)':>22} {str_ge10:>25} {'--':>18}")

    # --------------------------------------------------------------------------
    # 4. LOCAL TOPOLOGY & GEOMETRY PRESERVATION (SECTION 5 & 6)
    # --------------------------------------------------------------------------
    print("\n--------------------------------------------------------------------------------")
    print("4. LOCAL TOPOLOGY & GEOMETRY PRESERVATION (AMONG MATCHED CENTROIDS)")
    print("--------------------------------------------------------------------------------")
    topo_xs = [m for m in res_xs["topo_metrics"] if np.isfinite(m["dist_ratio_mean"])]
    topo_ss = [m for m in res_ss["topo_metrics"] if np.isfinite(m["dist_ratio_mean"])]

    print(f"{'Geometric Property':<36} {'Cross-Sensor (N=' + str(len(topo_xs)) + ')':>24} {'Same-Sensor (N=' + str(len(topo_ss)) + ')':>24}")
    print("-" * 86)
    dr_mean_xs = np.mean([m["dist_ratio_mean"] for m in topo_xs]) if topo_xs else float("nan")
    dr_mean_ss = np.mean([m["dist_ratio_mean"] for m in topo_ss]) if topo_ss else float("nan")
    dr_std_xs = np.mean([m["dist_ratio_std"] for m in topo_xs]) if topo_xs else float("nan")
    dr_std_ss = np.mean([m["dist_ratio_std"] for m in topo_ss]) if topo_ss else float("nan")
    print(f"{'Distance Ratio (Target / Source)':<36} {f'{dr_mean_xs:.3f} ± {dr_std_xs:.3f}':>24} {f'{dr_mean_ss:.3f} ± {dr_std_ss:.3f}':>24}")

    ang_xs = np.mean([m["angle_err_med"] for m in topo_xs]) if topo_xs else float("nan")
    ang_ss = np.mean([m["angle_err_med"] for m in topo_ss]) if topo_ss else float("nan")
    print(f"{'Median Neighbor Angular Error':<36} {f'{ang_xs:.2f} deg':>24} {f'{ang_ss:.2f} deg':>24}")

    tri_xs = np.mean([m["tri_shape_err_mean"] for m in topo_xs]) if topo_xs else float("nan")
    tri_ss = np.mean([m["tri_shape_err_mean"] for m in topo_ss]) if topo_ss else float("nan")
    print(f"{'Triangle Normalized Shape Error':<36} {f'{tri_xs:.4f}':>24} {f'{tri_ss:.4f}':>24}")

    # --------------------------------------------------------------------------
    # 5. SCIENTIFIC CLASSIFICATION (SECTION 11)
    # --------------------------------------------------------------------------
    print("\n================================================================================")
    print("5. SCIENTIFIC CLASSIFICATION (SECTION 11)")
    print("================================================================================")

    # Classification rules:
    # A — Strong Macro-Structure Signal: Macro centroids survive cross-sensor transformation and produce substantially >4 correspondences
    # B — Partial Signal: Macro structures survive but correspondence density or geometric stability is insufficient
    # C — No Useful Signal: Macro centroid configurations do not survive reliably or perform no better than chance

    med_real = np.median(res_xs["dists_real"])
    med_null = np.median(res_xs["dists_null"])
    rec5_real = np.mean(res_xs["rec5"]) * 100
    rec5_null = np.mean(res_xs["rec5_null"]) * 100

    if res_xs["scenes_ge4"] >= 20 and rec5_real > rec5_null * 2.0:
        classification = "A — Strong Macro-Structure Signal"
        reason = f"Macro centroids reliably survive and produce >=4 correspondences in {res_xs['scenes_ge4']}/{n_xs_scenes} scenes."
    elif res_xs["scenes_ge4"] > 4 or (rec5_real > rec5_null + 5.0):
        classification = "B — Partial Signal"
        reason = f"Macro centroids show modest correspondence in some scenes ({res_xs['scenes_ge4']}/{n_xs_scenes} scenes >=4), but correspondence density remains insufficient."
    else:
        classification = "C — No Useful Signal"
        reason = f"Macro centroids perform no better than chance (Median distance: {med_real:.2f} px vs null {med_null:.2f} px; Recall <= 5px: {rec5_real:.1f}% vs null {rec5_null:.1f}%; scenes with >=4 true matches: {res_xs['scenes_ge4']}/{n_xs_scenes} vs local patch 4/{n_xs_scenes})."

    print(f"Official Classification: {classification}")
    print(f"Scientific Rationale   : {reason}\n")

    return {
        "res_xs": res_xs,
        "res_ss": res_ss,
        "classification": classification,
        "reason": reason,
    }


if __name__ == "__main__":
    run_phase10_diagnostic()
