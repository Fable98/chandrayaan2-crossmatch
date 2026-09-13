"""Verify RNG desynchronization hypothesis between half_p=8 and half_p=16."""

import sys
from pathlib import Path
import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from scripts.generate_ground_truth_matches import generate_matches_for_image

p = REPO_ROOT / "data_preprocessing_pipeline/processed_triplets/region_001/ohrc_512.png"

scenes_a = []
rng_a = np.random.default_rng(42)
generate_matches_for_image(p, rng_a, scene_sink=scenes_a, half_p=8)

scenes_b = []
rng_b = np.random.default_rng(42)
generate_matches_for_image(p, rng_b, scene_sink=scenes_b, half_p=16)

print(f"Scenes generated in A: {len(scenes_a)}, in B: {len(scenes_b)}")
for i in range(len(scenes_a)):
    Ha = scenes_a[i]["H_gt"]
    Hb = scenes_b[i]["H_gt"]
    diff = np.max(np.abs(Ha - Hb))
    print(f"Scene {i} ({scenes_a[i]['target_image']}): max H_gt difference = {diff:.6f}")
