"""Evaluation of End-to-End Geometric Registration: Standard RANSAC vs. AI-Weighted RANSAC.

Strict, fair, leak-free paired evaluation across held-out unseen test image pairs.
Compares:
1. Standard/unweighted baseline: uniform RANSAC weighting.
2. AI-weighted: PROSAC-style weighted RANSAC using fresh RF probabilities (9 base features + disp_consistency).

Guarantees:
- Model trained strictly on 54 training groups (18 test groups held out).
- Same candidate correspondences fed to both methods.
- Same RANSAC budget (2000 iters), threshold (5.0 px), and estimator settings.
- H_gt used strictly as an independent evaluation oracle after estimation.
"""

import sys
from pathlib import Path
import numpy as np
import cv2
from scipy.stats import wilcoxon
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GroupShuffleSplit
import joblib

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "ML_model"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from train_ai_verifier import (
    load_matches_from_file,
    extract_label,
    extract_provenance_group,
    extract_feature_row,
    DEFAULT_LABEL_KEYS,
    SEED,
)
from matcher_cfog import (
    estimate_weighted_homography,
    calculate_reprojection_errors,
    verify_transformation_quality,
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


def get_cached_model_path(features: list[str], seed: int, train_groups: list[str] | None = None) -> Path:
    """Compute deterministic SHA-256 hash for temporary evaluation model cache."""
    import hashlib
    tg_str = ','.join(sorted(train_groups)) if train_groups else "all"
    key = f"{','.join(features)}:{seed}:{tg_str}"
    h = hashlib.sha256(key.encode("utf-8")).hexdigest()[:12]
    return Path(f"/tmp/ai_verifier_reg_eval_{h}.pkl")


def compute_mcnemar_test(arm1_succ: list[bool], arm2_succ: list[bool]) -> dict[str, Any]:
    """Compute exact McNemar test on paired binary registration outcomes."""
    b = sum(1 for s1, s2 in zip(arm1_succ, arm2_succ) if s1 and not s2)
    c = sum(1 for s1, s2 in zip(arm1_succ, arm2_succ) if not s1 and s2)
    n_disc = b + c
    if n_disc == 0:
        return {"p_value": 1.0, "b": 0, "c": 0, "n_discordant": 0, "method": "identical"}
    try:
        from scipy.stats import binomtest
        res = binomtest(min(b, c), n_disc, 0.5, alternative="two-sided")
        p_val = float(res.pvalue)
        method = "exact_binomial"
    except Exception:
        from scipy.stats import chi2
        stat = (abs(b - c) - 1.0) ** 2 / max(1, n_disc)
        p_val = float(chi2.sf(stat, df=1))
        method = "continuity_corrected_chi2"
    return {"p_value": p_val, "b": b, "c": c, "n_discordant": n_disc, "method": method}


def train_fresh_temporary_rf(X_train: np.ndarray, y_train: np.ndarray, feature_names: list[str], train_groups: list[str] | None = None) -> Path:
    """Train fresh RF strictly on training groups with frozen 10-feature set and hashed cache."""
    cache_path = get_cached_model_path(feature_names, SEED, train_groups)
    if cache_path.exists():
        return cache_path

    clf = RandomForestClassifier(
        n_estimators=100,
        class_weight="balanced",
        random_state=SEED,
        n_jobs=-1,
    )
    clf.fit(X_train, y_train)
    bundle = {
        "model": clf,
        "feature_names": feature_names,
        "n_train": len(X_train),
        "random_state": SEED,
    }
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, cache_path)
    return cache_path


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


def evaluate_candidate_ranking(matches_per_group: dict[str, list[dict]], rf_model, feature_indices: list[int]):
    """Measure Top-K precision for existing confidence vs RF probability."""
    k_vals = [5, 10, 20, 50]
    results = {"cross_sensor": {"conf": {k: [] for k in k_vals}, "rf": {k: [] for k in k_vals}},
               "same_sensor": {"conf": {k: [] for k in k_vals}, "rf": {k: [] for k in k_vals}}}

    for grp, m_list in matches_per_group.items():
        dom = m_list[0].get("domain", "unknown")
        if dom not in results:
            continue
        n_m = len(m_list)
        if n_m == 0:
            continue

        # Extract features
        feat_rows = []
        labels = []
        confs = []
        for m in m_list:
            r = extract_feature_row(m, require_live_features=True)
            feat_rows.append([r[idx] for idx in feature_indices])
            labels.append(1 if m.get("ground_truth_label") else 0)
            confs.append(float(m.get("confidence", 0.0)))

        rf_probs = rf_model.predict_proba(np.array(feat_rows))[:, 1]
        labels = np.array(labels)
        confs = np.array(confs)

        for k in k_vals:
            act_k = min(k, n_m)
            # Conf top-K
            top_conf_idx = np.argsort(-confs)[:act_k]
            p_conf = float(np.sum(labels[top_conf_idx])) / float(act_k)
            results[dom]["conf"][k].append(p_conf)

            # RF top-K
            top_rf_idx = np.argsort(-rf_probs)[:act_k]
            p_rf = float(np.sum(labels[top_rf_idx])) / float(act_k)
            results[dom]["rf"][k].append(p_rf)

    return results


def run_evaluation(rng_seeds: list[int] = [SEED]):
    json_path = REPO_ROOT / "ML_model/ground_truth_matches.json"
    records = load_matches_from_file(json_path)

    # 1. Reproduce exact grouped split
    X_rows, y_rows, groups_rows, domains_rows, meta_rows = [], [], [], [], []
    seen = set()
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
        grp, dom, _ = extract_provenance_group(m)
        groups_rows.append(grp)
        domains_rows.append(dom)
        X_rows.append(row)
        y_rows.append(lbl)
        meta_rows.append(m)

    X_all = np.asarray(X_rows, dtype=np.float64)  # 11 features
    y = np.asarray(y_rows, dtype=np.int64)
    groups = np.asarray(groups_rows, dtype=object)
    domains = np.asarray(domains_rows, dtype=object)

    # 10 features: 9 base + disp_consistency (col 9)
    X_10 = X_all[:, :10]

    gss = GroupShuffleSplit(n_splits=10, test_size=0.25, random_state=SEED)
    for tr_i, te_i in gss.split(X_10, y, groups=groups):
        if len(np.unique(y[tr_i])) >= 2 and len(np.unique(y[te_i])) >= 2:
            selected_split = (tr_i, te_i)
            break

    tr_idx, te_idx = selected_split
    tr_groups = sorted(list(set(groups[tr_idx])))
    te_groups = sorted(list(set(groups[te_idx])))

    # Persist split for auditability
    eval_dir = REPO_ROOT / "evaluation"
    eval_dir.mkdir(parents=True, exist_ok=True)
    with open(eval_dir / "train_groups.json", "w") as f:
        json.dump(tr_groups, f, indent=2)
    with open(eval_dir / "test_groups.json", "w") as f:
        json.dump(te_groups, f, indent=2)

    # 2. Recover oracle H_gt and canvas shapes for each group
    triplets_dir = REPO_ROOT / "data_preprocessing_pipeline/processed_triplets"
    image_paths = []
    for reg_dir in sorted(triplets_dir.glob("region_*")):
        for name in ("ohrc_512.png", "tmc_512.png"):
            p = reg_dir / name
            if p.exists():
                image_paths.append(p)

    rng = np.random.default_rng(SEED)
    scenes = []
    for p in image_paths:
        generate_matches_for_image(p, rng, scene_sink=scenes)

    h_gt_lookup = {}
    canvas_shape_lookup = {}
    for sc in scenes:
        grp = f"{sc['region']}:{sc['source_image']}::{sc['target_image']}"
        h_gt_lookup[grp] = sc["H_gt"]
        sh = sc.get("target_shape")
        if sh is None:
            try:
                probe = cv2.imread(str(sc.get("target_path", p)), cv2.IMREAD_GRAYSCALE)
                sh = probe.shape[:2] if probe is not None else (512, 512)
            except Exception:
                sh = (512, 512)
        canvas_shape_lookup[grp] = sh

    # 3. Train fresh RF strictly on training set
    print("Training fresh temporary RF on 54 training groups (frozen 10-feature set)...")
    model_path = train_fresh_temporary_rf(X_10[tr_idx], y[tr_idx], FROZEN_10_FEATURES, tr_groups)
    rf_bundle = joblib.load(model_path)
    rf_model = rf_bundle["model"]
    print(f"Temporary RF loaded from: {model_path}")

    # Group test matches by scene group
    test_matches_by_group = {}
    for idx in te_idx:
        grp = groups[idx]
        test_matches_by_group.setdefault(grp, []).append(meta_rows[idx])

    xs_test_groups = [g for g in te_groups if "cross_sensor" in g]
    ss_test_groups = [g for g in te_groups if "same_sensor" in g]

    print("\n================================================================================")
    print("1. DATASET CHARACTERISTICS")
    print("================================================================================")
    print(f"Total image-pair groups   : {len(set(groups))}")
    print(f"Held-out test groups      : {len(te_groups)} (Zero leakage: {len(set(tr_groups) & set(te_groups)) == 0})")
    print(f"Same-sensor test groups   : {len(ss_test_groups)}")
    print(f"Cross-sensor test groups  : {len(xs_test_groups)}")
    print(f"Total test candidates     : {len(te_idx)}")
    print(f"True test correspondences : {int(np.sum(y[te_idx] == 1))}")
    print(f"False test correspondences: {int(np.sum(y[te_idx] == 0))}")

    # 4. Candidate Ranking Evaluation (Top-K Precision)
    ranking_res = evaluate_candidate_ranking(test_matches_by_group, rf_model, list(range(10)))
    print("\n================================================================================")
    print("2. CANDIDATE RANKING: TOP-K PRECISION (CROSS-SENSOR UNSEEN TEST PAIRS)")
    print("================================================================================")
    print(f"{'Method':<25} {'Top-5 Precision':>18} {'Top-10 Precision':>18} {'Top-20 Precision':>18} {'Top-50 Precision':>18}")
    print("-" * 105)
    for method, label in [("conf", "Existing confidence"), ("rf", "RF probability (10 feats)")]:
        p5 = np.mean(ranking_res["cross_sensor"][method][5]) if ranking_res["cross_sensor"][method][5] else 0.0
        p10 = np.mean(ranking_res["cross_sensor"][method][10]) if ranking_res["cross_sensor"][method][10] else 0.0
        p20 = np.mean(ranking_res["cross_sensor"][method][20]) if ranking_res["cross_sensor"][method][20] else 0.0
        p50 = np.mean(ranking_res["cross_sensor"][method][50]) if ranking_res["cross_sensor"][method][50] else 0.0
        print(f"{label:<25} {p5*100:>17.2f}% {p10*100:>17.2f}% {p20*100:>17.2f}% {p50*100:>17.2f}%")

    # 5. Paired Registration Evaluation
    # For each test group, run Baseline RANSAC and AI-weighted RANSAC with identical points and seeds
    all_runs = []

    for seed_val in rng_seeds:
        for grp in te_groups:
            m_list = test_matches_by_group[grp]
            H_gt = h_gt_lookup.get(grp)
            dom = m_list[0].get("domain", "unknown")
            n_cands = len(m_list)

            pts1 = np.array([[float(m["source_x"]), float(m["source_y"])] for m in m_list], dtype=np.float32)
            pts2 = np.array([[float(m["target_x"]), float(m["target_y"])] for m in m_list], dtype=np.float32)
            gt_labels = np.array([1 if m.get("ground_truth_label") else 0 for m in m_list], dtype=np.int32)
            n_true = int(np.sum(gt_labels == 1))
            n_false = int(np.sum(gt_labels == 0))

            # Feature rows for RF
            feat_rows = []
            for m in m_list:
                r = extract_feature_row(m, require_live_features=True)
                feat_rows.append(r[:10])
            rf_probs = rf_model.predict_proba(np.array(feat_rows))[:, 1]

            # Run Method A: Standard Uniform RANSAC
            t_shape = canvas_shape_lookup.get(grp, (512, 512))
            w_uniform = np.ones(n_cands, dtype=np.float64)
            H_base, mask_base, tag_base = estimate_weighted_homography(
                pts1, pts2, w_uniform,
                ransac_reproj_threshold=5.0,
                image_shape=t_shape,
                rng_seed=seed_val,
                n_iters=2000,
            )

            # Run Method B: AI-Weighted RANSAC
            w_ai = np.asarray(rf_probs, dtype=np.float64)
            H_ai, mask_ai, tag_ai = estimate_weighted_homography(
                pts1, pts2, w_ai,
                ransac_reproj_threshold=5.0,
                image_shape=t_shape,
                rng_seed=seed_val,
                n_iters=2000,
            )

            def score_run(H, mask):
                success = (H is not None and mask is not None and int(np.sum(mask)) >= 4)
                if not success:
                    return {
                        "success": False, "inliers": 0, "inlier_ratio": 0.0,
                        "reproj_rmse": float("nan"), "corner_err": float("inf"),
                        "gt_tp": 0, "gt_fp": 0, "gt_precision": 0.0, "gt_recall": 0.0,
                    }
                idx = np.where(np.asarray(mask).ravel() != 0)[0]
                n_inl = len(idx)
                errs = calculate_reprojection_errors(pts1[idx], pts2[idx], H)
                rmse = float(np.sqrt(np.mean(errs ** 2))) if len(errs) else float("nan")
                c_err = compute_corner_error(H, H_gt, w=float(t_shape[1]), h=float(t_shape[0])) if H_gt is not None else float("nan")

                tp = int(np.sum(gt_labels[idx] == 1))
                fp = int(np.sum(gt_labels[idx] == 0))
                prec = float(tp) / float(n_inl) if n_inl > 0 else 0.0
                rec = float(tp) / float(n_true) if n_true > 0 else 0.0

                return {
                    "success": True, "inliers": n_inl, "inlier_ratio": float(n_inl) / float(n_cands),
                    "reproj_rmse": rmse, "corner_err": c_err,
                    "gt_tp": tp, "gt_fp": fp, "gt_precision": prec, "gt_recall": rec,
                }

            s_base = score_run(H_base, mask_base)
            s_ai = score_run(H_ai, mask_ai)

            all_runs.append({
                "group": grp, "domain": dom, "seed": seed_val,
                "n_cands": n_cands, "n_true": n_true, "n_false": n_false,
                "base": s_base, "ai": s_ai,
            })

    # Separate by slice
    for slice_name in ("Overall", "Same-Sensor", "Cross-Sensor"):
        if slice_name == "Overall":
            runs = all_runs
        elif slice_name == "Same-Sensor":
            runs = [r for r in all_runs if r["domain"] == "same_sensor"]
        else:
            runs = [r for r in all_runs if r["domain"] == "cross_sensor"]

        n_cases = len(runs)
        succ_base = sum(1 for r in runs if r["base"]["success"])
        succ_ai = sum(1 for r in runs if r["ai"]["success"])

        inl_base = [r["base"]["inliers"] for r in runs]
        inl_ai = [r["ai"]["inliers"] for r in runs]

        ratio_base = [r["base"]["inlier_ratio"] for r in runs]
        ratio_ai = [r["ai"]["inlier_ratio"] for r in runs]

        reproj_base = [r["base"]["reproj_rmse"] for r in runs if np.isfinite(r["base"]["reproj_rmse"])]
        reproj_ai = [r["ai"]["reproj_rmse"] for r in runs if np.isfinite(r["ai"]["reproj_rmse"])]

        # Valid corner error cases
        corner_base = [r["base"]["corner_err"] for r in runs if np.isfinite(r["base"]["corner_err"])]
        corner_ai = [r["ai"]["corner_err"] for r in runs if np.isfinite(r["ai"]["corner_err"])]

        gt_prec_base = [r["base"]["gt_precision"] for r in runs if r["base"]["success"]]
        gt_prec_ai = [r["ai"]["gt_precision"] for r in runs if r["ai"]["success"]]

        gt_rec_base = [r["base"]["gt_recall"] for r in runs if r["base"]["success"]]
        gt_rec_ai = [r["ai"]["gt_recall"] for r in runs if r["ai"]["success"]]

        print(f"\n================================================================================")
        print(f"REGISTRATION RESULTS: {slice_name.upper()} (N={n_cases} image-pair cases)")
        print(f"================================================================================")
        print(f"{'Metric':<32} {'Standard RANSAC':>18} {'AI-Weighted RANSAC':>20} {'Difference':>16}")
        print("-" * 90)
        print(f"{'Homography Success Rate':<32} {succ_base}/{n_cases} ({succ_base/n_cases*100:.1f}%) {succ_ai}/{n_cases} ({succ_ai/n_cases*100:.1f}%) {(succ_ai - succ_base)/n_cases*100:>+15.1f}%")
        print(f"{'Median Inlier Count':<32} {np.median(inl_base):>18.1f} {np.median(inl_ai):>20.1f} {np.median(inl_ai) - np.median(inl_base):>+16.1f}")
        print(f"{'Median Inlier Ratio':<32} {np.median(ratio_base)*100:>17.2f}% {np.median(ratio_ai)*100:>19.2f}% {(np.median(ratio_ai) - np.median(ratio_base))*100:>+15.2f}%")
        med_rep_b = np.median(reproj_base) if reproj_base else float("nan")
        med_rep_a = np.median(reproj_ai) if reproj_ai else float("nan")
        print(f"{'Median Reprojection Error (px)':<32} {med_rep_b:>18.3f} {med_rep_a:>20.3f} {med_rep_b - med_rep_a:>+16.3f}")
        med_c_b = np.median(corner_base) if corner_base else float("nan")
        med_c_a = np.median(corner_ai) if corner_ai else float("nan")
        print(f"{'Median Corner Error vs H_gt (px)':<32} {med_c_b:>18.3f} {med_c_a:>20.3f} {med_c_b - med_c_a:>+16.3f}")
        med_p_b = np.median(gt_prec_base) if gt_prec_base else 0.0
        med_p_a = np.median(gt_prec_ai) if gt_prec_ai else 0.0
        print(f"{'GT Correspondence Precision':<32} {med_p_b*100:>17.2f}% {med_p_a*100:>19.2f}% {(med_p_a - med_p_b)*100:>+15.2f}%")
        med_r_b = np.median(gt_rec_base) if gt_rec_base else 0.0
        med_r_a = np.median(gt_rec_ai) if gt_rec_ai else 0.0
        print(f"{'GT Correspondence Recall':<32} {med_r_b*100:>17.2f}% {med_r_a*100:>19.2f}% {(med_r_a - med_r_b)*100:>+15.2f}%")

    # 6. Paired Per-Case Analysis on Cross-Sensor Test Groups
    xs_runs = [r for r in all_runs if r["domain"] == "cross_sensor"]
    print("\n================================================================================")
    print("PAIRED PER-CASE ANALYSIS: CROSS-SENSOR UNSEEN TEST PAIRS")
    print("================================================================================")
    print(f"{'Group':<42} {'Base Err':>10} {'AI Err':>10} {'Δ Error':>10} {'Base Inl':>10} {'AI Inl':>10} {'Result'}")
    print("-" * 105)

    n_improved = 0
    n_worsened = 0
    n_unchanged = 0
    finite_err_diffs = []
    diag_penalty = math.hypot(512.0, 512.0)

    for r in xs_runs:
        c_base = r["base"]["corner_err"]
        c_ai = r["ai"]["corner_err"]
        i_base = r["base"]["inliers"]
        i_ai = r["ai"]["inliers"]

        # Improvement: baseline error - AI error (positive = improvement)
        if np.isfinite(c_base) and np.isfinite(c_ai):
            delta = c_base - c_ai
            finite_err_diffs.append(delta)
            if abs(delta) < 0.1:
                status = "UNCHANGED"
                n_unchanged += 1
            elif delta > 0:
                status = "IMPROVED"
                n_improved += 1
            else:
                status = "WORSENED"
                n_worsened += 1
        elif not np.isfinite(c_base) and np.isfinite(c_ai):
            status = "IMPROVED (Recovered)"
            n_improved += 1
        elif np.isfinite(c_base) and not np.isfinite(c_ai):
            status = "WORSENED (Failed)"
            n_worsened += 1
        else:
            status = "BOTH FAILED"
            n_unchanged += 1

        b_str = f"{c_base:.2f}" if np.isfinite(c_base) else "FAIL"
        a_str = f"{c_ai:.2f}" if np.isfinite(c_ai) else "FAIL"
        d_str = f"{c_base - c_ai:+.2f}" if (np.isfinite(c_base) and np.isfinite(c_ai)) else "N/A"
        print(f"{r['group']:<42} {b_str:>10} {a_str:>10} {d_str:>10} {i_base:>10d} {i_ai:>10d} {status}")

    print("-" * 105)
    print(f"Paired Summary: {n_improved} cases improved, {n_worsened} worsened, {n_unchanged} unchanged.")
    if finite_err_diffs:
        print(f"Median Paired Improvement (Finite Pairs Only): {np.median(finite_err_diffs):+.3f} px")
        print(f"Mean Paired Improvement (Finite Pairs Only):   {np.mean(finite_err_diffs):+.3f} px")

    # Wilcoxon signed-rank test strictly on finite pairs
    p_val_wilcox = float("nan")
    valid_diffs = [d for d in finite_err_diffs if abs(d) > 1e-3]
    if len(valid_diffs) >= 6:
        try:
            stat, p_val = wilcoxon(valid_diffs, alternative="greater")
            p_val_wilcox = float(p_val)
            print(f"Wilcoxon signed-rank test (finite pairs only, N={len(valid_diffs)}): stat={stat:.1f}, p={p_val:.4e}")
        except Exception as exc:
            print(f"Wilcoxon test note: {exc}")

    # McNemar test for success rate
    succ_base_flags = [bool(r["base"]["success"]) for r in xs_runs]
    succ_ai_flags = [bool(r["ai"]["success"]) for r in xs_runs]
    mcnemar_res = compute_mcnemar_test(succ_base_flags, succ_ai_flags)
    p_val_mcnemar = mcnemar_res["p_value"]
    print(f"McNemar test on success rate: b={mcnemar_res['b']}, c={mcnemar_res['c']}, p={p_val_mcnemar:.4e} ({mcnemar_res['method']})")

    # Side-by-side reporting: Success-rate, conditional-mean, Intent-to-Treat
    n_xs = len(xs_runs)
    succ_rate_b = sum(succ_base_flags) / max(1, n_xs) * 100.0
    succ_rate_a = sum(succ_ai_flags) / max(1, n_xs) * 100.0

    finite_cb = [r["base"]["corner_err"] for r in xs_runs if np.isfinite(r["base"]["corner_err"])]
    finite_ca = [r["ai"]["corner_err"] for r in xs_runs if np.isfinite(r["ai"]["corner_err"])]
    mean_cond_b = float(np.mean(finite_cb)) if finite_cb else float("nan")
    mean_cond_a = float(np.mean(finite_ca)) if finite_ca else float("nan")

    itt_cb = [r["base"]["corner_err"] if np.isfinite(r["base"]["corner_err"]) else diag_penalty for r in xs_runs]
    itt_ca = [r["ai"]["corner_err"] if np.isfinite(r["ai"]["corner_err"]) else diag_penalty for r in xs_runs]
    mean_itt_b = float(np.mean(itt_cb)) if itt_cb else float("nan")
    mean_itt_a = float(np.mean(itt_ca)) if itt_ca else float("nan")

    print("\n================================================================================")
    print("METHODOLOGY METRICS COMPARISON (SIDE-BY-SIDE: CROSS-SENSOR HELDOUT)")
    print("================================================================================")
    print(f"{'Metric':<35} {'Baseline (Uniform)':>20} {'AI-Weighted':>22} {'Difference':>16}")
    print("-" * 96)
    print(f"{'Registration Success Rate':<35} {succ_rate_b:>19.1f}% {succ_rate_a:>21.1f}% {succ_rate_a - succ_rate_b:>+15.1f}%")
    print(f"{'Conditional Mean Corner Error (px)':<35} {mean_cond_b:>20.2f} {mean_cond_a:>22.2f} {mean_cond_a - mean_cond_b:>+16.2f}")
    print(f"{'Intent-to-Treat Mean Error (px)':<35} {mean_itt_b:>20.2f} {mean_itt_a:>22.2f} {mean_itt_a - mean_itt_b:>+16.2f}")
    print(f"{'McNemar Test p-value':<35} {'--':>20} {p_val_mcnemar:>22.4e} {'b=' + str(mcnemar_res['b']) + ',c=' + str(mcnemar_res['c']):>16}")
    print(f"{'Wilcoxon (finite pairs) p-value':<35} {'--':>20} {p_val_wilcox:>22.4e} {'N=' + str(len(valid_diffs)):>16}")

if __name__ == "__main__":
    run_evaluation()
