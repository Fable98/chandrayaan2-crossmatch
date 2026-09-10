"""
train_ai_verifier.py — Train the AIMatchVerifier RandomForest from RANSAC-labeled matches.

Parses every ``*matches.json`` produced by the CFOG pipeline
(RANSAC has already labeled each correspondence as inlier / outlier via
geometric consensus) and trains a lightweight binary classifier.

Features (per match, all read with ``.get()`` defaults so missing keys are safe):
    1. ``confidence``               (falls back to ``score``)
    2. ``refinement_dx``            (sub-pixel shift X, default 0.0)
    3. ``refinement_dy``            (sub-pixel shift Y, default 0.0)
    4. ``spatial_quality_score``    (default 0.5; falls back to ``spatial_score``,
       else 1.0 if ``is_refined`` else 0.5 so legacy files still carry signal)

Labels (RANSAC consensus):
    1 for inliers, 0 for outliers. Accepts ``is_inlier`` (bool/int/str),
    plus legacy aliases ``inlier`` / ``label`` / ``ransac_inlier``.

Model:
    ``RandomForestClassifier(n_estimators=100, class_weight='balanced')`` —
    no deep learning, fast inference.

Output:
    ``ai_verifier_model.pkl`` (joblib) next to this script by default, as a
    bundle dict ``{"model": clf, "feature_names": [...], ...}`` so
    ``ai_verifier.AIMatchVerifier`` can load it directly.

Usage:
    python train_ai_verifier.py [--output ai_verifier_model.pkl] [--test-size 0.25]
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import joblib
import numpy as np

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


def extract_label(match: dict) -> int | None:
    """RANSAC consensus label: 1 = inlier, 0 = outlier, None = unknown/skip."""
    for key in ("is_inlier", "inlier", "ransac_inlier", "label"):
        if key in match:
            b = _as_bool_label(match.get(key))
            if b is not None:
                return 1 if b else 0
    # Explicit outlier flag without an inlier flag.
    if "is_outlier" in match:
        b = _as_bool_label(match.get("is_outlier"))
        if b is not None:
            return 0 if b else 1
    return None


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


def build_dataset(search_roots: list[Path]) -> tuple[np.ndarray, np.ndarray, dict]:
    """Parse all match files into (X, y, stats). Rows without labels are skipped."""
    files = iter_match_files(search_roots)
    print(f"Found {len(files)} *matches.json file(s) to parse.")
    X_rows: list[list[float]] = []
    y_rows: list[int] = []
    # Several output dirs contain byte-identical copies of the same matches
    # (e.g. registration_output_demo vs evaluation_output/...). Deduplicate
    # exact-repeat rows so they neither triple-count nor leak across the
    # train/test split.
    seen_rows: set[tuple] = set()
    n_duplicates = 0
    stats = {"files": len(files), "parsed": 0, "skipped_no_label": 0, "per_file": []}
    for f in files:
        records = load_matches_from_file(f)
        n_in, n_out, n_skip = 0, 0, 0
        for m in records:
            label = extract_label(m)
            if label is None:
                n_skip += 1
                continue
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
        print(f"  {f}: {n_in} unique inliers, {n_out} unique outliers, {n_skip} skipped (no label)")
    stats["duplicates_removed"] = n_duplicates
    if n_duplicates:
        print(f"Removed {n_duplicates} exact-duplicate row(s) shared across output dirs.")
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
    random_state: int = 42,
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
            f"(n={len(y)}). The parsed matches.json files contain no outlier "
            f"(is_inlier=False) examples. Re-run the CFOG matcher so failed legs "
            f"persist 'all_matches' with RANSAC labels instead of inliers only."
        )

    # Stratified split; fall back to train-on-all if the minority class is tiny.
    can_stratify = test_size > 0.0 and min(np.bincount(y)) >= 2
    if can_stratify:
        X_tr, X_te, y_tr, y_te = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
    else:
        X_tr, y_tr, X_te, y_te = X, y, X, y
        print("Minority class has <2 samples: evaluating on the training set.")

    clf.fit(X_tr, y_tr)
    y_pred = clf.predict(X_te)
    print("\n=== Classification report ===")
    print(classification_report(y_te, y_pred, target_names=["outlier(0)", "inlier(1)"]))
    print("Confusion matrix (rows=true, cols=pred):")
    print(confusion_matrix(y_te, y_pred))
    print(f"Train size: {len(y_tr)}, Test size: {len(y_te)}")
    print(f"Feature importances ({FEATURE_NAMES}): {np.round(clf.feature_importances_, 4).tolist()}")
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
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--n-estimators", type=int, default=100)
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

    X, y, stats = build_dataset(search_roots)
    print(f"\nTotal labeled matches: {len(y)} "
          f"({int(np.sum(y == 1))} inliers, {int(np.sum(y == 0))} outliers), "
          f"{stats['skipped_no_label']} skipped.")
    if len(y) == 0:
        print("ERROR: no labeled matches found. Nothing to train on.", file=sys.stderr)
        return 2

    try:
        clf = train(X, y, test_size=args.test_size,
                    random_state=args.random_state, n_estimators=args.n_estimators)
    except ValueError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    out_path = Path(args.output) if args.output else (this_dir / "ai_verifier_model.pkl")
    bundle = {
        "model": clf,
        "feature_names": FEATURE_NAMES,
        "n_train": int(len(y)),
        "class_counts": {int(k): int(v) for k, v in zip(*np.unique(y, return_counts=True))},
    }
    joblib.dump(bundle, out_path)
    print(f"\nSaved trained verifier -> {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
