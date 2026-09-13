"""
train_ai_verifier.py — Train & Evaluate AIMatchVerifier RandomForest on Ground-Truth Matches.

Supervision and Anti-RANSAC Integrity:
  * Uses synthetic geometric ground truth (``ground_truth_label``, ``label_source=synthetic_geometry``)
    computed from known orbital projective transformations H_gt and reprojection error thresholds.
  * Backward compatibility: accepts legacy ``human_label`` (or manual aliases), stamping
    ``label_source=hand`` only if genuine manual provenance is present.
  * Training on downstream RANSAC inlier/outlier consensus is CIRCULAR (the filter runs
    before RANSAC; training on RANSAC memorizes the coarse matcher's own confidence and
    vetoes true low-confidence cross-sensor matches). Circular labels are REFUSED by default
    (exit 2).

Evaluation and Leakage Resistance:
  * Supports grouped evaluation (``--split grouped``, default when groups exist):
    Grouped by provenance hierarchy (image pair > region/source) to prevent leaking highly
    correlated correspondences from the same synthetic scene into both train and test splits.
  * Train and test groups are strictly disjoint (zero group leakage).
  * The model is fitted ONLY on the training split, and evaluated on the held-out test split.
  * Reports disaggregated metrics: Overall, Cross-Sensor, and Same-Sensor (precision, recall,
    F1, support, confusion matrix).
  * Random correspondence split (``--split random``) is preserved as a diagnostic baseline.

Features (order is the load-time contract with ``ai_verifier.py``):
    1. ``confidence``               (falls back to ``score``)
    2. ``refinement_dx``            (sub-pixel shift X)
    3. ``refinement_dy``            (sub-pixel shift Y)
    4. ``spatial_quality_score``
    5. ``cfog_distance``            (Euclidean CFOG descriptor distance)
    6. ``pc_energy_src``            (max Phase Congruency energy at source)
    7. ``pc_energy_tgt``            (max Phase Congruency energy at target)
    8. ``nn_ratio``                 (Lowe 1st/2nd NN descriptor distance ratio)
    9. ``scale_diff``               (abs blob-scale difference)

Model:
    ``RandomForestClassifier(n_estimators=100, class_weight='balanced')``

Bundle Output:
    Saves joblib dictionary with model, feature names, label source, n_samples,
    n_train, n_test, split strategy, group key, and evaluation metrics.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import joblib
import numpy as np

try:
    from config import SEED
except ImportError:
    from ML_model.config import SEED

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("ML_model.train_ai_verifier")

# Feature order is the contract with ai_verifier.AIMatchVerifier.extract_features.
BASE_FEATURE_NAMES = [
    "confidence",
    "refinement_dx",
    "refinement_dy",
    "spatial_quality_score",
    "cfog_distance",
    "pc_energy_src",
    "pc_energy_tgt",
    "nn_ratio",
    "scale_diff",
]

NEW_STRUCTURAL_FEATURE_NAMES = [
    "disp_consistency",
    "pairwise_dist_ratio",
]

FEATURE_NAMES = list(BASE_FEATURE_NAMES) + list(NEW_STRUCTURAL_FEATURE_NAMES)

DEFAULT_SEARCH_ROOTS = [
    "data_preprocessing_pipeline/processed_triplets",
    "data_preprocessing_pipeline/matches",
    "ML_model",
    "output",
    "results",
    "evaluation_output",
    "registration_output_demo",
]

# ---------------------------------------------------------------------------
# Parsing & Provenance helpers
# ---------------------------------------------------------------------------

def _as_bool_label(value: Any) -> bool | None:
    """Interpret a raw label value; return None if uninterpretable."""
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float, np.integer, np.floating)):
        return bool(value)
    if isinstance(value, str):
        v = value.strip().lower()
        if v in ("true", "1", "yes", "inlier", "y"):
            return True
        if v in ("false", "0", "no", "outlier", "n"):
            return False
    return None


GROUND_TRUTH_LABEL_KEYS = ("ground_truth_label",)
HAND_LABEL_KEYS = ("human_label", "hand_label", "manual_label", "verified_label")
DEFAULT_LABEL_KEYS = GROUND_TRUTH_LABEL_KEYS + HAND_LABEL_KEYS
RANSAC_LABEL_KEYS = ("is_inlier", "inlier", "ransac_inlier", "label")


def extract_label(
    match: dict,
    allow_ransac: bool = False,
    label_keys: tuple[str, ...] = DEFAULT_LABEL_KEYS,
) -> tuple[int | None, str]:
    """Extract verified binary label: 1 = true, 0 = false. Returns (label, source).

    source is "synthetic_geometry", "hand", "ransac" (only when allow_ransac=True), or "none".
    RANSAC consensus keys are IGNORED unless explicitly allowed, because they
    are circular supervision for a pre-RANSAC filter.
    """
    for key in label_keys:
        if key in match:
            b = _as_bool_label(match.get(key))
            if b is not None:
                src_field = match.get("label_source")
                if src_field == "synthetic_geometry":
                    source = "synthetic_geometry"
                elif src_field in ("hand", "manual", "hand_labelled"):
                    source = "hand"
                elif key in GROUND_TRUTH_LABEL_KEYS:
                    source = "synthetic_geometry"
                else:
                    source = "hand"
                return (1 if b else 0), source

    if allow_ransac:
        for key in RANSAC_LABEL_KEYS:
            if key in match:
                b = _as_bool_label(match.get(key))
                if b is not None:
                    return (1 if b else 0), "ransac"
        if "is_outlier" in match:
            b = _as_bool_label(match.get("is_outlier"))
            if b is not None:
                return (0 if b else 1), "ransac"

    return None, "none"


def extract_provenance_group(match: dict) -> tuple[str | None, str, str]:
    """Extract (group_id, domain, group_level) for grouped train/test splitting.

    Hierarchy:
      1. Image pair: f"{region}:{src_img}::{tgt_img}" or f"{src_img}::{tgt_img}"
      2. Region or source_image
    """
    src_img = match.get("source_image")
    tgt_img = match.get("target_image")
    region = match.get("region")
    domain = str(match.get("domain", "unknown"))

    if src_img and tgt_img:
        group = f"{region}:{src_img}::{tgt_img}" if region else f"{src_img}::{tgt_img}"
        level = "image_pair"
    elif region:
        group = str(region)
        level = "region"
    elif src_img:
        group = str(src_img)
        level = "source_image"
    else:
        group = None
        level = "none"

    return group, domain, level


def extract_feature_row(match: dict, require_live_features: bool = True) -> list[float]:
    """Build one feature row conforming to the 9-feature contract."""
    if require_live_features:
        has_dx = "refinement_dx" in match or "ref_dx" in match
        has_dy = "refinement_dy" in match or "ref_dy" in match
        has_sp = "spatial_quality_score" in match or "spatial_score" in match
        has_desc = all(k in match for k in (
            "cfog_distance", "pc_energy_src", "pc_energy_tgt", "nn_ratio", "scale_diff",
        ))
        if not (has_dx and has_dy and has_sp and has_desc):
            raise KeyError("missing live matcher features")

    conf = float(match.get("confidence", match.get("score", 0.0) or 0.0))
    dx = float(match.get("refinement_dx", match.get("ref_dx", 0.0) or 0.0))
    dy = float(match.get("refinement_dy", match.get("ref_dy", 0.0) or 0.0))
    spatial_raw = match.get("spatial_quality_score", match.get("spatial_score", None))
    if spatial_raw is None:
        raise KeyError("spatial_quality_score")
    spatial = float(spatial_raw)
    cfog_d = float(match.get("cfog_distance", 0.0) or 0.0)
    pc_src = float(match.get("pc_energy_src", 0.0) or 0.0)
    pc_tgt = float(match.get("pc_energy_tgt", 0.0) or 0.0)
    nn_ratio = float(match["nn_ratio"]) if "nn_ratio" in match else 1.0
    scale_diff = float(match.get("scale_diff", 0.0) or 0.0)
    disp_c = float(match.get("disp_consistency", 0.5) if match.get("disp_consistency") is not None else 0.5)
    dist_r = float(match.get("pairwise_dist_ratio", 0.5) if match.get("pairwise_dist_ratio") is not None else 0.5)
    return [conf, dx, dy, spatial, cfog_d, pc_src, pc_tgt, nn_ratio, scale_diff, disp_c, dist_r]


def iter_match_files(search_roots: list[Path]) -> list[Path]:
    """Recursively collect *matches.json files under the given roots."""
    files: list[Path] = []
    for root in search_roots:
        if not root.exists():
            continue
        if root.is_file() and root.name.endswith(".json"):
            files.append(root)
        elif root.is_dir():
            files.extend(sorted(root.rglob("*matches.json")))
    seen, unique = set(), []
    for f in files:
        key = str(f.resolve())
        if key not in seen:
            seen.add(key)
            unique.append(f)
    return unique


def load_matches_from_file(path: Path) -> list[dict]:
    """Load a matches file; supports a bare list or a dict with matches/all_matches."""
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        logger.warning("Skipping %s: unreadable JSON (%s)", path, exc)
        return []
    if isinstance(data, dict):
        for key in ("all_matches", "matches", "correspondences"):
            if isinstance(data.get(key), list):
                data = data[key]
                break
        else:
            return []
    if not isinstance(data, list):
        return []
    return [m for m in data if isinstance(m, dict)]


def build_dataset(
    search_roots: list[Path],
    allow_ransac: bool = False,
    label_keys: tuple[str, ...] = DEFAULT_LABEL_KEYS,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Parse all match files into (X, y, stats). Rows without labels are skipped."""
    files = iter_match_files(search_roots)
    logger.info("Found %d *matches.json file(s) to parse.", len(files))
    X_rows: list[list[float]] = []
    y_rows: list[int] = []
    groups_rows: list[str | None] = []
    domains_rows: list[str] = []
    group_levels: set[str] = set()

    seen_rows: set[tuple] = set()
    n_duplicates = 0
    stats = {
        "files": len(files),
        "parsed": 0,
        "skipped_no_label": 0,
        "skipped_missing_features": 0,
        "per_file": [],
        "n_synthetic_geometry": 0,
        "n_hand": 0,
        "n_ransac": 0,
        "allow_ransac": bool(allow_ransac),
    }

    for f in files:
        records = load_matches_from_file(f)
        n_in, n_out, n_skip, n_feat = 0, 0, 0, 0
        for m in records:
            label, source = extract_label(m, allow_ransac=allow_ransac, label_keys=label_keys)
            if label is None:
                n_skip += 1
                continue
            try:
                row = extract_feature_row(m, require_live_features=True)
            except Exception:
                n_feat += 1
                continue

            if source == "synthetic_geometry":
                stats["n_synthetic_geometry"] += 1
            elif source == "hand":
                stats["n_hand"] += 1
            elif source == "ransac":
                stats["n_ransac"] += 1

            dedup_key = (
                tuple(round(float(v), 6) for v in row),
                label,
                round(float(m.get("source_x", m.get("image1_x", 0.0)) or 0.0), 4),
                round(float(m.get("source_y", m.get("image1_y", 0.0)) or 0.0), 4),
                round(float(m.get("target_x", m.get("image2_x", 0.0)) or 0.0), 4),
                round(float(m.get("target_y", m.get("image2_y", 0.0)) or 0.0), 4),
            )
            if dedup_key in seen_rows:
                n_duplicates += 1
                continue
            seen_rows.add(dedup_key)

            grp, dom, lvl = extract_provenance_group(m)
            groups_rows.append(grp)
            domains_rows.append(dom)
            if lvl != "none":
                group_levels.add(lvl)

            X_rows.append(row)
            y_rows.append(label)
            if label == 1:
                n_in += 1
            else:
                n_out += 1

        stats["parsed"] += n_in + n_out
        stats["skipped_no_label"] += n_skip
        stats["skipped_missing_features"] += n_feat
        stats["per_file"].append({
            "file": str(f), "inliers": n_in, "outliers": n_out,
            "skipped": n_skip, "skipped_missing_features": n_feat,
        })
        logger.info(
            "  %s: %d unique inliers, %d unique outliers, %d skipped (no label), "
            "%d skipped (missing live features)",
            f, n_in, n_out, n_skip, n_feat,
        )

    stats["duplicates_removed"] = n_duplicates
    if n_duplicates:
        logger.info("Removed %d exact-duplicate row(s) shared across output dirs.", n_duplicates)

    X = np.asarray(X_rows, dtype=np.float64)
    y = np.asarray(y_rows, dtype=np.int64)
    stats["groups"] = np.asarray(groups_rows, dtype=object)
    stats["domains"] = np.asarray(domains_rows, dtype=object)
    stats["group_key"] = (
        "image_pair" if "image_pair" in group_levels
        else ("region" if "region" in group_levels
              else ("source_image" if "source_image" in group_levels else "none"))
    )
    return X, y, stats


# ---------------------------------------------------------------------------
# Metrics & Reporting helpers
# ---------------------------------------------------------------------------

def _compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, title: str) -> dict:
    """Compute precision, recall, F1, accuracy, support, and confusion matrix."""
    from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support

    if len(y_true) == 0:
        return {"title": title, "support": 0, "accuracy": 0.0}

    acc = float(accuracy_score(y_true, y_pred))
    prec, rec, f1, supp = precision_recall_fscore_support(
        y_true, y_pred, labels=[0, 1], zero_division=0
    )
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1]).tolist()

    return {
        "title": title,
        "n_samples": int(len(y_true)),
        "accuracy": float(round(acc, 4)),
        "class_0_outlier": {
            "precision": float(round(prec[0], 4)),
            "recall": float(round(rec[0], 4)),
            "f1": float(round(f1[0], 4)),
            "support": int(supp[0]),
        },
        "class_1_inlier": {
            "precision": float(round(prec[1], 4)),
            "recall": float(round(rec[1], 4)),
            "f1": float(round(f1[1], 4)),
            "support": int(supp[1]),
        },
        "confusion_matrix": cm,
    }


def _log_metrics_block(metrics: dict):
    title = metrics.get("title", "Metrics")
    if metrics.get("n_samples", 0) == 0:
        logger.info("  [%s] No evaluation samples in this subset.", title)
        return

    c0 = metrics["class_0_outlier"]
    c1 = metrics["class_1_inlier"]
    cm = metrics["confusion_matrix"]
    logger.info("  [%s] (N=%d, Accuracy: %.2f%%)", title, metrics["n_samples"], metrics["accuracy"] * 100.0)
    logger.info("    Class 0 (Outlier): Precision=%.3f, Recall=%.3f, F1=%.3f (Support=%d)",
                c0["precision"], c0["recall"], c0["f1"], c0["support"])
    logger.info("    Class 1 (Inlier) : Precision=%.3f, Recall=%.3f, F1=%.3f (Support=%d)",
                c1["precision"], c1["recall"], c1["f1"], c1["support"])
    logger.info("    Confusion Matrix [[TN=%d, FP=%d], [FN=%d, TP=%d]]",
                cm[0][0], cm[0][1], cm[1][0], cm[1][1])


# ---------------------------------------------------------------------------
# Training & Split
# ---------------------------------------------------------------------------

def train(
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray | None = None,
    domains: np.ndarray | None = None,
    split_strategy: str = "grouped",
    test_size: float = 0.25,
    random_state: int = SEED,
    n_estimators: int = 100,
) -> tuple[Any, dict, tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
    """Train RandomForest with honest, leak-free train/test evaluation.

    Returns:
        (clf, eval_results, (X_tr, y_tr, X_te, y_te))
    """
    from sklearn.ensemble import RandomForestClassifier

    classes = np.unique(y)
    if len(classes) < 2:
        raise ValueError(
            f"Cannot train a binary verifier: only class {classes.tolist()} present "
            f"(n={len(y)}). A one-sided label set cannot supervise a true/false "
            f"filter — collect counterexamples of the missing class and re-run. "
            f"Refusing to save a degenerate single-class model."
        )

    # 1. Train / Test Split
    if split_strategy == "grouped":
        valid_groups = [g for g in groups if g is not None] if groups is not None else []
        n_unique_groups = len(set(valid_groups))
        if groups is None or n_unique_groups < 2 or len(valid_groups) < len(y):
            raise ValueError(
                f"Grouped evaluation requested, but dataset has insufficient provenance groups "
                f"(found {n_unique_groups} distinct group(s) across {len(valid_groups)} rows out of {len(y)}). "
                f"Grouped train/test split requires at least 2 distinct groups. "
                f"Refusing to silently fall back to random split."
            )

        from sklearn.model_selection import GroupShuffleSplit
        gss = GroupShuffleSplit(n_splits=10, test_size=test_size, random_state=random_state)
        selected_split = None
        for tr_i, te_i in gss.split(X, y, groups=groups):
            if len(np.unique(y[tr_i])) >= 2 and len(np.unique(y[te_i])) >= 2:
                selected_split = (tr_i, te_i)
                break
        if selected_split is None:
            for tr_i, te_i in gss.split(X, y, groups=groups):
                if len(np.unique(y[tr_i])) >= 2:
                    selected_split = (tr_i, te_i)
                    break
            if selected_split is None:
                selected_split = next(gss.split(X, y, groups=groups))

        tr_idx, te_idx = selected_split

        # Anti-leakage assertion: train groups and test groups must be strictly disjoint
        tr_groups = set(groups[tr_idx])
        te_groups = set(groups[te_idx])
        overlap = tr_groups & te_groups
        if overlap:
            raise RuntimeError(f"Group leakage detected! Overlapping groups: {overlap}")

        logger.info("\n=== Grouped Evaluation Split ===")
        logger.info("Total distinct groups: %d", n_unique_groups)
        logger.info("Train groups (%d): %s", len(tr_groups), sorted(list(tr_groups))[:5] + (["..."] if len(tr_groups) > 5 else []))
        logger.info("Test groups (%d): %s", len(te_groups), sorted(list(te_groups))[:5] + (["..."] if len(te_groups) > 5 else []))
        logger.info("Group leakage verified: train and test groups are strictly disjoint.")

    elif split_strategy == "random":
        logger.info("\n=== Random Evaluation Split (Diagnostic Baseline) ===")
        logger.warning(
            "Running random correspondence-level split. "
            "WARNING: This diagnostic baseline can leak correspondences from the same scene into both train and test."
        )
        from sklearn.model_selection import train_test_split
        can_stratify = test_size > 0.0 and min(np.bincount(y)) >= 2
        if can_stratify:
            indices = np.arange(len(y))
            tr_idx, te_idx = train_test_split(
                indices, test_size=test_size, random_state=random_state, stratify=y
            )
        else:
            tr_idx = np.arange(len(y))
            te_idx = np.arange(len(y))
            logger.info("Minority class has <2 samples: evaluating on full training set.")
    else:
        raise ValueError(f"Unknown split strategy: {split_strategy}")

    X_tr, y_tr = X[tr_idx], y[tr_idx]
    X_te, y_te = X[te_idx], y[te_idx]
    dom_te = domains[te_idx] if domains is not None else np.array(["unknown"] * len(te_idx))

    # 2. Fit model ONLY on training split (no leakage)
    clf = RandomForestClassifier(
        n_estimators=n_estimators,
        class_weight="balanced",
        random_state=random_state,
        n_jobs=-1,
    )
    clf.fit(X_tr, y_tr)

    # 3. Predict ONLY on held-out test split
    y_pred = clf.predict(X_te)

    # 4. Disaggregated metrics
    overall_metrics = _compute_metrics(y_te, y_pred, "Overall")

    xs_mask = (dom_te == "cross_sensor")
    if np.any(xs_mask):
        xs_metrics = _compute_metrics(y_te[xs_mask], y_pred[xs_mask], "Cross-Sensor")
    else:
        xs_metrics = {"title": "Cross-Sensor", "n_samples": 0}

    ss_mask = (dom_te == "same_sensor")
    if np.any(ss_mask):
        ss_metrics = _compute_metrics(y_te[ss_mask], y_pred[ss_mask], "Same-Sensor")
    else:
        ss_metrics = {"title": "Same-Sensor", "n_samples": 0}

    eval_results = {
        "overall": overall_metrics,
        "cross_sensor": xs_metrics,
        "same_sensor": ss_metrics,
    }

    # 5. Log comprehensive report
    logger.info("\n=== Evaluation Results (%s Split) ===", split_strategy.upper())
    logger.info("Training size: %d rows (Class 0=%d, Class 1=%d)", len(y_tr), int(np.sum(y_tr == 0)), int(np.sum(y_tr == 1)))
    logger.info("Testing size:  %d rows (Class 0=%d, Class 1=%d)", len(y_te), int(np.sum(y_te == 0)), int(np.sum(y_te == 1)))
    _log_metrics_block(overall_metrics)
    _log_metrics_block(xs_metrics)
    _log_metrics_block(ss_metrics)

    importances = np.round(clf.feature_importances_, 4).tolist()
    logger.info("\n=== Feature Importances ===")
    for fname, fimp in zip(FEATURE_NAMES, importances):
        logger.info("  %-25s: %.4f", fname, fimp)

    return clf, eval_results, (X_tr, y_tr, X_te, y_te)


# ---------------------------------------------------------------------------
# CLI Entrypoint
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Train AIMatchVerifier RandomForest with grouped/leakage-resistant evaluation."
    )
    parser.add_argument(
        "--inputs", nargs="*", default=None,
        help="Explicit matches.json files or directories to parse. "
             "Default: recursively search known output roots.",
    )
    parser.add_argument(
        "--output", default=None,
        help="Where to save the .pkl bundle. (Default: None — will not save unless specified).",
    )
    parser.add_argument(
        "--split", choices=["grouped", "random", "auto"], default="auto",
        help="Split strategy for evaluation: 'grouped' (hierarchy: image_pair > region), "
             "'random' (correspondence-level baseline), or 'auto' (grouped if groups exist, else random).",
    )
    parser.add_argument("--test-size", type=float, default=0.25)
    parser.add_argument("--random-state", type=int, default=SEED)
    parser.add_argument("--n-estimators", type=int, default=100)
    parser.add_argument(
        "--label-key", default="ground_truth_label",
        help="Primary key holding binary ground-truth label. "
             "Default: ground_truth_label (aliases: human_label, hand_label, manual_label).",
    )
    parser.add_argument(
        "--allow-ransac-labels", action="store_true",
        help="DANGEROUS: fall back to circular RANSAC consensus labels. The saved bundle "
             "is stamped label_source=ransac-acknowledged and is REFUSED by "
             "AIMatchVerifier at load time (research artifact only).",
    )
    args = parser.parse_args(argv)

    this_dir = Path(__file__).resolve().parent
    project_root = this_dir.parent

    if args.inputs:
        search_roots = [Path(p) if Path(p).is_absolute() else (Path.cwd() / p) for p in args.inputs]
    else:
        search_roots = []
        for rel in DEFAULT_SEARCH_ROOTS:
            search_roots.append(project_root / rel)
            search_roots.append(this_dir / Path(rel).name)

    label_keys = (args.label_key,) + tuple(k for k in DEFAULT_LABEL_KEYS if k != args.label_key)

    if args.allow_ransac_labels:
        logger.warning(
            "DANGEROUS: --allow-ransac-labels accepted. RANSAC consensus labels are "
            "circular supervision for a pre-RANSAC filter; the saved bundle will be "
            "stamped label_source=ransac-acknowledged and REFUSED by AIMatchVerifier."
        )

    X, y, stats = build_dataset(
        search_roots,
        allow_ransac=args.allow_ransac_labels,
        label_keys=label_keys,
    )

    logger.info(
        "Total labeled matches: %d (%d true, %d false), %d skipped (no label).",
        len(y),
        int(np.sum(y == 1)),
        int(np.sum(y == 0)),
        stats["skipped_no_label"],
    )

    if len(y) == 0:
        logger.error(
            "ERROR: no valid ground-truth matches found under keys %s. "
            "Nothing to train on. Supply synthetic geometric ground truth (ground_truth_label) "
            "or genuine human labels; RANSAC consensus labels are refused without --allow-ransac-labels.",
            label_keys,
        )
        return 2

    # Determine split strategy
    groups = stats.get("groups")
    domains = stats.get("domains")
    valid_groups = [g for g in groups if g is not None] if groups is not None else []
    n_distinct_groups = len(set(valid_groups))

    if args.split == "auto":
        actual_split = "grouped" if n_distinct_groups >= 2 and len(valid_groups) == len(y) else "random"
    else:
        actual_split = args.split

    try:
        clf, eval_results, (X_tr, y_tr, X_te, y_te) = train(
            X, y,
            groups=groups,
            domains=domains,
            split_strategy=actual_split,
            test_size=args.test_size,
            random_state=args.random_state,
            n_estimators=args.n_estimators,
        )
    except ValueError as exc:
        logger.error("ERROR: %s", exc)
        return 2

    # Determine accurate label_source for metadata
    if args.allow_ransac_labels:
        label_source = "ransac-acknowledged"
    elif stats["n_synthetic_geometry"] > 0:
        label_source = "synthetic_geometry"
    elif stats["n_hand"] > 0:
        label_source = "hand"
    else:
        label_source = "synthetic_geometry"

    # Only save bundle if output argument was provided
    if args.output:
        out_path = Path(args.output)
        bundle = {
            "model": clf,
            "feature_names": FEATURE_NAMES,
            "label_source": label_source,
            "label_key": args.label_key,
            "n_samples": int(len(y)),
            "n_train": int(len(X_tr)),
            "n_test": int(len(X_te)),
            "class_counts": {int(k): int(v) for k, v in zip(*np.unique(y, return_counts=True))},
            "split_strategy": actual_split,
            "group_key": stats.get("group_key", "none"),
            "random_seed": args.random_state,
            "evaluation_metrics": eval_results,
        }
        joblib.dump(bundle, out_path)
        logger.info("Saved trained verifier bundle -> %s", out_path)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
