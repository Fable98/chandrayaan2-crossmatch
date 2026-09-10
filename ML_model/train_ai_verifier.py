"""
train_ai_verifier.py — Train the AIMatchVerifier RandomForest from HAND-LABELLED matches.

Training on RANSAC inlier/outlier consensus is CIRCULAR (the filter runs
before RANSAC, so the model memorizes the coarse matcher's own confidence
and vetoes true low-confidence matches on real CDR pairs) and is REFUSED by
default. Supply hand-labelled true/false correspondences instead: each match
record must carry a human verdict under ``human_label`` (aliases
``hand_label``, ``manual_label``, ``verified_label``; 1/True = true
correspondence, 0/False = false).

Features (per match, all read with ``.get()`` defaults so missing keys are safe):
    1. ``confidence``               (falls back to ``score``)
    2. ``refinement_dx``            (sub-pixel shift X, default 0.0)
    3. ``refinement_dy``            (sub-pixel shift Y, default 0.0)
    4. ``spatial_quality_score``    (default 0.5; falls back to ``spatial_score``,
       else 1.0 if ``is_refined`` else 0.5 so legacy files still carry signal)

Model:
    ``RandomForestClassifier(n_estimators=100, class_weight='balanced')`` —
    no deep learning, fast inference.

Output:
    ``ai_verifier_model.pkl`` (joblib) next to this script by default, as a
    bundle dict ``{"model": clf, "feature_names": [...], "label_source": "hand", ...}``
    so ``ai_verifier.AIMatchVerifier`` can load it directly. Bundles trained
    with --allow-ransac-labels are stamped label_source="ransac-acknowledged"
    and are REFUSED at load time (research artifacts only).

Failure contract (loud, non-zero exit):
    * no hand-labelled rows found  -> exit 2 with an explicit error.
    * single-class labels          -> exit 2 with an explicit error
      (a one-sided set cannot supervise a binary verifier).

Usage:
    python train_ai_verifier.py --inputs hand_labels/matches.json [--output ai_verifier_model.pkl]
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import joblib
import numpy as np

try:
    from config import SEED
except ImportError:
    from ML_model.config import SEED

logger = logging.getLogger("ML_model.train_ai_verifier")

# Feature order is the contract with ai_verifier.AIMatchVerifier.extract_features.
FEATURE_NAMES = ["confidence", "refinement_dx", "refinement_dy", "spatial_quality_score"]

# Directories searched (recursively) for *matches.json when no explicit input is given.
# NOTE: data_preprocessing_pipeline/processed_triplets currently holds manifests + PNGs,
# while the actual RANSAC-labeled correspondences live in
# data_preprocessing_pipeline/matches/*_matches.json — so we search both, plus every
# other known output location (output/, results/, evaluation_output/, ...).
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
# Parsing helpers
# ---------------------------------------------------------------------------

def _as_bool_label(value) -> bool | None:
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


HAND_LABEL_KEYS = ("human_label", "hand_label", "manual_label", "verified_label")
RANSAC_LABEL_KEYS = ("is_inlier", "inlier", "ransac_inlier", "label")


def extract_label(match: dict, allow_ransac: bool = False,
                  hand_keys: tuple[str, ...] = HAND_LABEL_KEYS) -> tuple[int | None, str]:
    """Hand-verdict label: 1 = true, 0 = false. Returns (label, source).

    source is "hand", "ransac" (only when allow_ransac=True), or "none".
    RANSAC consensus keys are IGNORED unless explicitly allowed, because they
    are circular supervision for a pre-RANSAC filter.
    """
    for key in hand_keys:
        if key in match:
            b = _as_bool_label(match.get(key))
            if b is not None:
                return (1 if b else 0), "hand"
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


def extract_feature_row(match: dict) -> list[float]:
    """Build one feature row; every key uses .get() with a safe default."""
    conf = float(match.get("confidence", match.get("score", 0.0) or 0.0))
    dx = float(match.get("refinement_dx", match.get("ref_dx", 0.0) or 0.0))
    dy = float(match.get("refinement_dy", match.get("ref_dy", 0.0) or 0.0))
    spatial_raw = match.get("spatial_quality_score", match.get("spatial_score", None))
    if spatial_raw is None:
        # Legacy files predate spatial_quality_score: derive a neutral prior
        # from whether sub-pixel refinement succeeded.
        spatial = 1.0 if match.get("is_refined", False) else 0.5
    else:
        try:
            spatial = float(spatial_raw)
        except (TypeError, ValueError):
            spatial = 0.5
    mag_guard = match.get("refinement_mag", None)  # optional extra, folded into dx/dy only
    _ = mag_guard  # documented: kept for forward-compat, not a separate feature
    return [conf, dx, dy, spatial]


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
    # De-duplicate while preserving order.
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


def build_dataset(search_roots: list[Path], allow_ransac: bool = False,
                  hand_keys: tuple[str, ...] = HAND_LABEL_KEYS) -> tuple[np.ndarray, np.ndarray, dict]:
    """Parse all match files into (X, y, stats). Rows without labels are skipped."""
    files = iter_match_files(search_roots)
    logger.info("Found %d *matches.json file(s) to parse.", len(files))
    X_rows: list[list[float]] = []
    y_rows: list[int] = []
    # Several output dirs contain byte-identical copies of the same matches
    # (e.g. registration_output_demo vs evaluation_output/...). Deduplicate
    # exact-repeat rows so they neither triple-count nor leak across the
    # train/test split.
    seen_rows: set[tuple] = set()
    n_duplicates = 0
    stats = {"files": len(files), "parsed": 0, "skipped_no_label": 0, "per_file": [],
             "n_hand": 0, "n_ransac": 0, "allow_ransac": bool(allow_ransac)}
    for f in files:
        records = load_matches_from_file(f)
        n_in, n_out, n_skip = 0, 0, 0
        for m in records:
            label, source = extract_label(m, allow_ransac=allow_ransac, hand_keys=hand_keys)
            if label is None:
                n_skip += 1
                continue
            if source == "hand":
                stats["n_hand"] += 1
            else:
                stats["n_ransac"] += 1
            try:
                row = extract_feature_row(m)
            except Exception:
                n_skip += 1
                continue
            dedup_key = (
                round(row[0], 6), round(row[1], 6), round(row[2], 6), round(row[3], 6),
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
            X_rows.append(row)
            y_rows.append(label)
            if label == 1:
                n_in += 1
            else:
                n_out += 1
        stats["parsed"] += n_in + n_out
        stats["skipped_no_label"] += n_skip
        stats["per_file"].append({"file": str(f), "inliers": n_in, "outliers": n_out, "skipped": n_skip})
        logger.info("  %s: %d unique inliers, %d unique outliers, %d skipped (no label)", f, n_in, n_out, n_skip)
    stats["duplicates_removed"] = n_duplicates
    if n_duplicates:
        logger.info("Removed %d exact-duplicate row(s) shared across output dirs.", n_duplicates)
    X = np.asarray(X_rows, dtype=np.float64)
    y = np.asarray(y_rows, dtype=np.int64)
    return X, y, stats


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train(
    X: np.ndarray,
    y: np.ndarray,
    test_size: float = 0.25,
    random_state: int = SEED,
    n_estimators: int = 100,
):
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import classification_report, confusion_matrix
    from sklearn.model_selection import train_test_split

    clf = RandomForestClassifier(
        n_estimators=n_estimators,
        class_weight="balanced",  # inliers usually outnumber outliers (or vice versa)
        random_state=random_state,
        n_jobs=-1,
    )
    classes = np.unique(y)
    if len(classes) < 2:
        raise ValueError(
            f"Cannot train a binary verifier: only class {classes.tolist()} present "
            f"(n={len(y)}). A one-sided label set cannot supervise a true/false "
            f"filter — collect hand-labelled counterexamples of the missing class "
            f"and re-run. Refusing to save a degenerate single-class model."
        )

    # Stratified split; fall back to train-on-all if the minority class is tiny.
    can_stratify = test_size > 0.0 and min(np.bincount(y)) >= 2
    if can_stratify:
        X_tr, X_te, y_tr, y_te = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
    else:
        X_tr, y_tr, X_te, y_te = X, y, X, y
        logger.info("Minority class has <2 samples: evaluating on the training set.")

    clf.fit(X_tr, y_tr)
    y_pred = clf.predict(X_te)
    logger.info("\n=== Classification report ===\n%s", classification_report(y_te, y_pred, target_names=["outlier(0)", "inlier(1)"]))
    logger.info("Confusion matrix (rows=true, cols=pred):\n%s", confusion_matrix(y_te, y_pred))
    logger.info("Train size: %d, Test size: %d", len(y_tr), len(y_te))
    logger.info("Feature importances (%s): %s", FEATURE_NAMES, np.round(clf.feature_importances_, 4).tolist())
    return clf


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Train AIMatchVerifier RandomForest from RANSAC labels.")
    parser.add_argument(
        "--inputs", nargs="*", default=None,
        help="Explicit matches.json files or directories to parse. "
             "Default: recursively search known output roots.",
    )
    parser.add_argument("--output", default=None,
                        help="Where to save the .pkl (default: <this-dir>/ai_verifier_model.pkl).")
    parser.add_argument("--test-size", type=float, default=0.25)
    parser.add_argument("--random-state", type=int, default=SEED)
    parser.add_argument("--n-estimators", type=int, default=100)
    parser.add_argument("--label-key", default="human_label",
                        help="Match-record key holding the HAND verdict (aliases: hand_label, "
                             "manual_label, verified_label). RANSAC consensus keys are ignored "
                             "unless --allow-ransac-labels is passed.")
    parser.add_argument("--allow-ransac-labels", action="store_true",
                        help="DANGEROUS: fall back to RANSAC consensus labels. The saved bundle "
                             "is stamped label_source=ransac-acknowledged and is REFUSED by "
                             "AIMatchVerifier at load time (research artifact only).")
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

    hand_keys: tuple[str, ...] = (args.label_key,) + tuple(
        k for k in HAND_LABEL_KEYS if k != args.label_key
    )
    if args.allow_ransac_labels:
        logger.warning(
            "DANGEROUS: --allow-ransac-labels accepted. RANSAC consensus labels are "
            "circular supervision for a pre-RANSAC filter; the saved bundle will be "
            "stamped label_source=ransac-acknowledged and REFUSED by AIMatchVerifier."
        )
    X, y, stats = build_dataset(search_roots, allow_ransac=args.allow_ransac_labels,
                                hand_keys=hand_keys)
    logger.info(
        "Total labeled matches: %d (%d true, %d false), %d skipped (no hand label).",
        len(y),
        int(np.sum(y == 1)),
        int(np.sum(y == 0)),
        stats["skipped_no_label"],
    )
    if len(y) == 0:
        logger.error(
            "ERROR: no hand-labelled matches found under key '%s' (aliases: %s). "
            "Nothing to train on. Label correspondences by hand as true/false and "
            "re-run; RANSAC consensus labels are refused without --allow-ransac-labels.",
            args.label_key, ", ".join(HAND_LABEL_KEYS),
        )
        return 2

    try:
        clf = train(X, y, test_size=args.test_size,
                    random_state=args.random_state, n_estimators=args.n_estimators)
    except ValueError as exc:
        logger.error("ERROR: %s", exc)
        return 2

    out_path = Path(args.output) if args.output else (this_dir / "ai_verifier_model.pkl")
    bundle = {
        "model": clf,
        "feature_names": FEATURE_NAMES,
        "label_source": "ransac-acknowledged" if args.allow_ransac_labels else "hand",
        "label_key": args.label_key,
        "n_train": int(len(y)),
        "class_counts": {int(k): int(v) for k, v in zip(*np.unique(y, return_counts=True))},
    }
    joblib.dump(bundle, out_path)
    logger.info("Saved trained verifier -> %s", out_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
