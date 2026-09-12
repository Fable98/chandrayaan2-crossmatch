"""
backfill_descriptor_features.py — Enrich hand-labelled GT with descriptor features.

Loads ML_model/ground_truth_matches.json and writes the same records back with
five NEW measured fields:

    cfog_distance, pc_energy_src, pc_energy_tgt, nn_ratio, scale_diff

Every existing field (human_label, label_source, domain, coordinates, ...) is
left untouched. This script does NOT run RANSAC and does NOT assign 1/0 labels
from an inlier mask.

How the five fields are recovered
---------------------------------
Existing rows do not store the source/target rasters. The original generator
(scripts/generate_ground_truth_matches.py) is deterministic given SEED and the
same region images, and adding descriptor computation does not consume RNG.
We replay that generator, then copy *only* the five new keys onto the matching
existing row (matched by coordinates + domain + human_label).

Rows that cannot be joined to a replayed pair are filled by measuring the five
fields at the *stored* (source, target) coordinates on a (pc1, pc2) scene
collected during that same replay (same domain, highest PC energy). Labels
are never rewritten.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any

import numpy as np

THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = THIS_DIR.parent
sys.path.insert(0, str(THIS_DIR))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from config import SEED  # noqa: E402
from generate_ground_truth_matches import generate_matches_for_image  # noqa: E402
from matcher_cfog import DESCRIPTOR_FEATURE_KEYS, compute_descriptor_match_features  # noqa: E402

logger = logging.getLogger("ML_model.backfill_descriptor_features")

NEW_KEYS = DESCRIPTOR_FEATURE_KEYS


def _row_key(rec: dict, nd: int = 4) -> tuple:
    return (
        round(float(rec.get("source_x", 0.0)), nd),
        round(float(rec.get("source_y", 0.0)), nd),
        round(float(rec.get("target_x", 0.0)), nd),
        round(float(rec.get("target_y", 0.0)), nd),
        rec.get("domain"),
        bool(rec.get("human_label")),
    )


def _feat_key(rec: dict) -> tuple:
    return (
        round(float(rec.get("source_x", 0.0)), 2),
        round(float(rec.get("source_y", 0.0)), 2),
        round(float(rec.get("confidence", rec.get("score", 0.0))), 6),
        rec.get("domain"),
        bool(rec.get("human_label")),
    )


def _discover_images() -> list[Path]:
    triplets_dir = REPO_ROOT / "data_preprocessing_pipeline/processed_triplets"
    image_paths: list[Path] = []
    for reg_dir in sorted(triplets_dir.glob("region_*")):
        for name in ("ohrc_512.png", "tmc_512.png"):
            p = reg_dir / name
            if p.exists():
                image_paths.append(p)
    return image_paths


def replay_feature_index(
    image_paths: list[Path],
) -> tuple[dict[tuple, dict], dict[tuple, dict], list[dict[str, Any]]]:
    rng = np.random.default_rng(SEED)
    by_xy: dict[tuple, dict] = {}
    by_feat: dict[tuple, dict] = {}
    scenes: list[dict[str, Any]] = []
    for p in image_paths:
        logger.info("Replaying generator on %s...", p.relative_to(REPO_ROOT))
        recs = generate_matches_for_image(p, rng, scene_sink=scenes)
        for rec in recs:
            feats = {k: float(rec[k]) for k in NEW_KEYS}
            by_xy[_row_key(rec)] = feats
            by_xy[_row_key(rec, nd=2)] = feats
            by_feat[_feat_key(rec)] = feats
        logger.info("  index size=%d scenes=%d", len(by_xy), len(scenes))
    return by_xy, by_feat, scenes


def _in_bounds(x: float, y: float, pc: np.ndarray, margin: int = 2) -> bool:
    h, w = pc.shape[:2]
    return margin <= x < (w - margin) and margin <= y < (h - margin)


def measure_on_best_scene(rec: dict, scenes: list[dict[str, Any]]) -> dict[str, float] | None:
    sx = float(rec["source_x"])
    sy = float(rec["source_y"])
    tx = float(rec["target_x"])
    ty = float(rec["target_y"])
    domain = rec.get("domain")
    best_e = -1.0
    best: dict[str, float] | None = None
    for scene in scenes:
        if scene["domain"] != domain:
            continue
        pc1, pc2 = scene["pc1"], scene["pc2"]
        if not (_in_bounds(sx, sy, pc1) and _in_bounds(tx, ty, pc2)):
            continue
        feats = compute_descriptor_match_features(
            pc1, pc2, sx, sy, tx, ty, tgt_gallery=scene["gallery"],
        )
        energy = float(feats["pc_energy_src"]) + float(feats["pc_energy_tgt"])
        if energy > best_e:
            best_e = energy
            best = feats
    return best


def backfill(
    records: list[dict],
    by_xy: dict[tuple, dict],
    by_feat: dict[tuple, dict],
    scenes: list[dict[str, Any]],
) -> tuple[int, int, int]:
    filled = 0
    via_scene = 0
    missing = 0
    for rec in records:
        feats = by_xy.get(_row_key(rec)) or by_xy.get(_row_key(rec, nd=2)) or by_feat.get(_feat_key(rec))
        source = "replay"
        if feats is None:
            feats = measure_on_best_scene(rec, scenes)
            source = "scene"
        if feats is None:
            missing += 1
            continue
        for k in NEW_KEYS:
            rec[k] = float(feats[k])
        filled += 1
        if source == "scene":
            via_scene += 1
    return filled, via_scene, missing


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        default=str(THIS_DIR / "ground_truth_matches.json"),
        help="Existing hand-labelled matches JSON (modified in place by default).",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Write path (default: overwrite --input).",
    )
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    in_path = Path(args.input)
    out_path = Path(args.output) if args.output else in_path
    if not in_path.exists():
        logger.error("Ground-truth file not found: %s", in_path)
        return 2

    records = json.loads(in_path.read_text(encoding="utf-8"))
    if not isinstance(records, list):
        logger.error("Expected a JSON list of match records.")
        return 2

    n_before = len(records)
    labels_before = [bool(r.get("human_label")) for r in records]
    sources_before = [r.get("label_source") for r in records]
    logger.info(
        "Loaded %d records from %s (label_source values: %s)",
        n_before, in_path, set(sources_before),
    )

    image_paths = _discover_images()
    if not image_paths:
        logger.error("No region images found to replay descriptor features from.")
        return 2

    by_xy, by_feat, scenes = replay_feature_index(image_paths)
    filled, via_scene, missing = backfill(records, by_xy, by_feat, scenes)
    logger.info("Filled %d / %d rows (scene-fallback=%d, unmatched=%d).", filled, n_before, via_scene, missing)

    if filled < n_before:
        logger.error(
            "Refusing to write: %d existing hand-labelled rows could not be matched. "
            "No RANSAC fallback is provided.",
            missing,
        )
        return 2

    if [bool(r.get("human_label")) for r in records] != labels_before:
        logger.error("Refusing to write: human_label values changed during backfill.")
        return 2
    if [r.get("label_source") for r in records] != sources_before:
        logger.error("Refusing to write: label_source values changed during backfill.")
        return 2

    out_path.write_text(json.dumps(records, indent=2), encoding="utf-8")
    logger.info("Wrote enriched ground truth -> %s", out_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
