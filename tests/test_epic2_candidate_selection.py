"""
tests/test_epic2_candidate_selection.py — Epic 2: GOA pre-selection funnel.

Failure definition: under extreme sun-angle shift (cross-sensor,
multimodal_pair=True), scalar NCC drops to noise (~0.004) and discards the
true match before MI is evaluated. Structural Gradient Orientation Agreement
(GOA) / Phase Congruency Energy Correlation must keep the TRUE ground-truth
coordinate within the top-5 pre-selection candidates.

Synthetic cross-sensor pair: Phase Congruency structural map, template
polarity-inverted (simulates ~160-180 deg azimuth shadow reversal) plus
coarse-sensor blur/downsample simulation.
"""
import sys
from pathlib import Path

import cv2
import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "ML_model"))

from matcher_cfog import (
    adaptive_illumination_normalization,
    compute_goa_search_surface,
    compute_phase_congruency,
    find_best_correspondence_unified,
)

TOL_PX = 3.0


def _load_structural_map() -> np.ndarray:
    sample = REPO_ROOT / "sample_data" / "ohrc_sample.png"
    if sample.exists():
        raw = cv2.imread(str(sample), cv2.IMREAD_GRAYSCALE)
        img = (raw.astype(np.float32) / 255.0).astype(np.float32)
    else:
        rng = np.random.RandomState(7)
        base = rng.uniform(50.0, 180.0, size=(256, 256)).astype(np.float32)
        base = cv2.GaussianBlur(base, (15, 15), 3.0)
        for _ in range(12):
            cx, cy = rng.randint(20, 236), rng.randint(20, 236)
            rad = rng.randint(8, 30)
            y, x = np.ogrid[:256, :256]
            d = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)
            base[d <= rad] *= 0.65
            base[(d > rad - 2) & (d <= rad + 2)] *= 1.35
        base = np.clip(base, 10, 245)
        img = ((base - base.min()) / (base.max() - base.min())).astype(np.float32)
    normed, _ = adaptive_illumination_normalization(img, enable_tophat=False)
    return compute_phase_congruency(normed)


def _build_cross_sensor_pair():
    """Returns (search_region, xsensor_tmpl, true_loc)."""
    pc = _load_structural_map()
    th, tw = 32, 32
    tx, ty = 100, 120  # template origin inside pc
    # Search window with 40px margin -> true loc inside search = (40, 40).
    sr = pc[ty - 40: ty + th + 40, tx - 40: tx + tw + 40].copy()
    assert sr.shape == (th + 80, tw + 80)
    true_loc = (40, 40)  # (x, y) in search-region coords
    tmpl = pc[ty: ty + th, tx: tx + tw].copy()
    # Cross-sensor simulation: polarity flip (160-180 deg sun shift) +
    # coarse-sensor blur + 2x down/up sampling (scale-gap proxy; full 8x
    # handled by physical GSD resampling upstream).
    xsensor = (1.0 - tmpl).astype(np.float32)
    xsensor = cv2.GaussianBlur(xsensor, (3, 3), 0.8)
    small = cv2.resize(xsensor, (tw // 2, th // 2), interpolation=cv2.INTER_AREA)
    xsensor = cv2.resize(small, (tw, th), interpolation=cv2.INTER_LINEAR).astype(np.float32)
    return sr, xsensor, true_loc


def _top_k_locs(surface: np.ndarray, k: int = 5):
    flat = surface.ravel()
    k = min(k, flat.size)
    idx = np.argpartition(-flat, k)[:k]
    idx = idx[np.argsort(-flat[idx])]
    locs = []
    for i in idx:
        cy, cx = np.unravel_index(i, surface.shape)
        locs.append((int(cx), int(cy)))
    return locs


def _min_dist_to_truth(locs, truth) -> float:
    return min(float(np.hypot(x - truth[0], y - truth[1])) for x, y in locs)


def test_epic2_goa_funnel_contains_truth_in_top5():
    sr, xsensor, truth = _build_cross_sensor_pair()
    goa = compute_goa_search_surface(sr, xsensor)
    assert goa.shape == (sr.shape[0] - xsensor.shape[0] + 1,
                         sr.shape[1] - xsensor.shape[1] + 1)
    top5 = _top_k_locs(goa, k=5)
    dist = _min_dist_to_truth(top5, truth)
    print(f"GOA top5={top5} truth={truth} min_dist={dist:.2f}")
    assert dist <= TOL_PX, (
        f"GOA pre-selection funnel lost the true match: min_dist={dist:.2f}px "
        f"> {TOL_PX}px (top5={top5}, truth={truth})"
    )


def test_epic2_unified_multimodal_uses_structural_preselection():
    """Default multimodal path must return the true target (GOA, not NCC).

    Calls WITHOUT an explicit use_goa flag so the production default is
    exercised. Before the Epic-2 fix (NCC pre-selection) this returns a
    wrong peak; after the fix it must land within tolerance of truth.
    """
    sr, xsensor, truth = _build_cross_sensor_pair()
    score, loc = find_best_correspondence_unified(
        sr, xsensor, multimodal_pair=True
    )
    dist = float(np.hypot(loc[0] - truth[0], loc[1] - truth[1]))
    print(f"unified multimodal score={score:.4f} loc={loc} truth={truth} dist={dist:.2f}")
    assert dist <= TOL_PX, (
        f"multimodal pre-selection collapsed: returned {loc} "
        f"{dist:.2f}px from truth {truth} (score={score:.4f})"
    )
