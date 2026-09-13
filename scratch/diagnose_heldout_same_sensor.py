"""Diagnose Phase 6 held-out same-sensor evaluation."""

import sys
from pathlib import Path
import numpy as np
import cv2

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "ML_model"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from config import SEED
from evaluation.eval_candidate_generation import get_train_test_split
from scripts.generate_ground_truth_matches import generate_matches_for_image
from matcher_cfog import estimate_weighted_homography

train_groups, test_groups = get_train_test_split()
ss_test_groups = [g for g in test_groups if "same_sensor" in g]

triplets_dir = REPO_ROOT / "data_preprocessing_pipeline/processed_triplets"
image_paths = []
for reg_dir in sorted(triplets_dir.glob("region_*")):
    for name in ("ohrc_512.png", "tmc_512.png"):
        p = reg_dir / name
        if p.exists():
            image_paths.append(p)

scenes_sink = []
rng = np.random.default_rng(SEED)
for p in image_paths:
    generate_matches_for_image(p, rng, scene_sink=scenes_sink, half_p=8)

h_gt_lookup = {}
for sc in scenes_sink:
    grp = f"{sc['region']}:{sc['source_image']}::{sc['target_image']}"
    h_gt_lookup[grp] = sc["H_gt"]

print(f"Held-out same-sensor groups: {ss_test_groups}")

for grp in ss_test_groups[:3]:
    reg_src, tgt_name = grp.split("::")
    reg, src_name = reg_src.split(":")
    p = triplets_dir / reg / src_name

    # Generate matches with half_p=8 vs 16
    rng_8 = np.random.default_rng(SEED)
    m8_all = generate_matches_for_image(p, rng_8, half_p=8)
    m8 = [m for m in m8_all if f"{m['region']}:{m['source_image']}::{m['target_image']}" == grp]

    rng_16 = np.random.default_rng(SEED)
    m16_all = generate_matches_for_image(p, rng_16, half_p=16)
    m16 = [m for m in m16_all if f"{m['region']}:{m['source_image']}::{m['target_image']}" == grp]

    H_gt = h_gt_lookup[grp]
    print(f"\nGroup: {grp}")
    print(f"  half_p=8 : N={len(m8)}")
    if m8:
        e8 = [m['true_reprojection_error_px'] for m in m8]
        print(f"    mean_err={np.mean(e8):.2f}, med={np.median(e8):.2f}, <=2px: {sum(1 for e in e8 if e<=2.0)}")
    print(f"  half_p=16: N={len(m16)}")
    if m16:
        e16 = [m['true_reprojection_error_px'] for m in m16]
        print(f"    mean_err={np.mean(e16):.2f}, med={np.median(e16):.2f}, <=2px: {sum(1 for e in e16 if e<=2.0)}")
