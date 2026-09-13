"""Feature Ablation Evaluation on Grouped Held-Out Unseen Test Pairs.

Evaluates:
- Baseline: Existing 9 features
- Experiment 1: 9 + disp_consistency (Candidate F)
- Experiment 2: 9 + disp_consistency + pairwise_dist_ratio (Candidates F & G)
- Experiment 3: disp_consistency + pairwise_dist_ratio (confidence removed)

Maintains identical GroupShuffleSplit, RF configurations, random seeds, and class weights.
"""

import sys
from pathlib import Path
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
    roc_auc_score,
    average_precision_score,
)
from sklearn.model_selection import GroupShuffleSplit

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "ML_model"))

from train_ai_verifier import (
    load_matches_from_file,
    extract_label,
    extract_provenance_group,
    extract_feature_row,
    DEFAULT_LABEL_KEYS,
    SEED,
)

def run_feature_ablation():
    json_path = REPO_ROOT / "ML_model/ground_truth_matches.json"
    records = load_matches_from_file(json_path)

    X_rows = []
    y_rows = []
    groups_rows = []
    domains_rows = []
    seen = set()

    for m in records:
        lbl, _ = extract_label(m, allow_ransac=False, label_keys=DEFAULT_LABEL_KEYS)
        if lbl is None:
            continue
        try:
            row = extract_feature_row(m, require_live_features=True)
        except Exception:
            continue

        # row has 11 features: 9 base + disp_c + dist_r
        key = (
            tuple(round(float(v), 6) for v in row[:9]),
            lbl,
            round(float(m.get("source_x", 0.0)), 4),
            round(float(m.get("source_y", 0.0)), 4),
            round(float(m.get("target_x", 0.0)), 4),
            round(float(m.get("target_y", 0.0)), 4),
        )
        if key in seen:
            continue
        seen.add(key)
        grp, dom, _ = extract_provenance_group(m)
        groups_rows.append(grp)
        domains_rows.append(dom)
        X_rows.append(row)
        y_rows.append(lbl)

    X_all = np.asarray(X_rows, dtype=np.float64)  # (N, 11)
    y = np.asarray(y_rows, dtype=np.int64)
    groups = np.asarray(groups_rows, dtype=object)
    domains = np.asarray(domains_rows, dtype=object)

    # Recreate the deterministic grouped split
    gss = GroupShuffleSplit(n_splits=10, test_size=0.25, random_state=SEED)
    selected_split = None
    for tr_i, te_i in gss.split(X_all, y, groups=groups):
        if len(np.unique(y[tr_i])) >= 2 and len(np.unique(y[te_i])) >= 2:
            selected_split = (tr_i, te_i)
            break
    if selected_split is None:
        selected_split = next(gss.split(X_all, y, groups=groups))

    tr_idx, te_idx = selected_split
    dom_te = domains[te_idx]
    y_te = y[te_idx]
    xs_mask = (dom_te == "cross_sensor")
    ss_mask = (dom_te == "same_sensor")

    print("================================================================================")
    print("FEATURE ABLATION EVALUATION (GROUPED HELD-OUT UNSEEN TEST PAIRS)")
    print("================================================================================")
    print(f"Total Rows: {len(X_all)} | Train: {len(tr_idx)} rows | Test: {len(te_idx)} rows")
    print(f"Test Slices: Cross-Sensor N={np.sum(xs_mask)} (True={np.sum(y_te[xs_mask]==1)}, False={np.sum(y_te[xs_mask]==0)}) | Same-Sensor N={np.sum(ss_mask)}")

    # Configurations:
    # 0..8: Base 9 features
    # 9: disp_consistency
    # 10: pairwise_dist_ratio
    configs = [
        ("Baseline (Existing 9)", X_all[:, :9]),
        ("Exp 1 (9 + disp_consistency)", X_all[:, :10]),
        ("Exp 2 (9 + disp_c + dist_ratio)", X_all[:, :11]),
        ("Exp 3 (disp_c + dist_r, conf removed)", np.column_stack([X_all[:, 1:9], X_all[:, 9:11]])),
    ]

    ablation_summary = []

    for name, feat_mat in configs:
        clf = RandomForestClassifier(
            n_estimators=100,
            class_weight="balanced",
            random_state=SEED,
            n_jobs=-1,
        )
        clf.fit(feat_mat[tr_idx], y[tr_idx])
        probs_te = clf.predict_proba(feat_mat[te_idx])[:, 1]
        preds_te = clf.predict(feat_mat[te_idx])

        # Overall metrics
        acc = accuracy_score(y_te, preds_te)
        p_ov, r_ov, f_ov, _ = precision_recall_fscore_support(y_te, preds_te, labels=[0, 1], zero_division=0)
        cm_ov = confusion_matrix(y_te, preds_te)

        # Cross-sensor metrics
        y_xs = y_te[xs_mask]
        p_xs = probs_te[xs_mask]
        pred_xs = preds_te[xs_mask]
        roc_xs = roc_auc_score(y_xs, p_xs)
        pr_xs = average_precision_score(y_xs, p_xs)
        p_x, r_x, f_x, _ = precision_recall_fscore_support(y_xs, pred_xs, labels=[0, 1], zero_division=0)
        cm_xs = confusion_matrix(y_xs, pred_xs)

        # Same-sensor metrics
        y_ss = y_te[ss_mask]
        pred_ss = preds_te[ss_mask]
        p_s, r_s, f_s, _ = precision_recall_fscore_support(y_ss, pred_ss, labels=[0, 1], zero_division=0)

        ablation_summary.append({
            "name": name,
            "roc_xs": roc_xs,
            "pr_xs": pr_xs,
            "r_xs": r_x[1],
            "p_xs": p_x[1],
            "f_xs": f_x[1],
            "cm_xs": cm_xs,
            "r_ss": r_s[1],
            "p_ss": p_s[1],
            "f_ss": f_s[1],
            "acc": acc,
            "f_ov": f_ov[1],
        })

        print(f"\n--- {name} ---")
        print(f"Cross-Sensor: ROC-AUC={roc_xs:.4f}, PR-AUC={pr_xs:.4f}, Recall={r_x[1]:.4f}, F1={f_x[1]:.4f}")
        print(f"              CM: [[TN={cm_xs[0,0]}, FP={cm_xs[0,1]}], [FN={cm_xs[1,0]}, TP={cm_xs[1,1]}]]")
        print(f"Same-Sensor : Inlier Recall={r_s[1]:.4f}, Precision={p_s[1]:.4f}, F1={f_s[1]:.4f}")
        print(f"Overall     : Accuracy={acc*100:.2f}%, Inlier F1={f_ov[1]:.4f}")

    print("\n================================================================================")
    print("ABLATION SUMMARY TABLE")
    print("================================================================================")
    print(f"{'Feature Set':<38} {'Cross ROC':>10} {'Cross PR':>10} {'Cross Rec':>10} {'Cross F1':>10} {'Same F1':>10} {'Overall Acc':>12}")
    print("-" * 105)
    for row in ablation_summary:
        print(
            f"{row['name']:<38} {row['roc_xs']:>10.4f} {row['pr_xs']:>10.4f} {row['r_xs']:>10.4f} "
            f"{row['f_xs']:>10.4f} {row['f_ss']:>10.4f} {row['acc']*100:>11.2f}%"
        )

if __name__ == "__main__":
    run_feature_ablation()
