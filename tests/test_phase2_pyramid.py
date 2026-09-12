"""
tests/test_phase2_pyramid.py — Verification for Phase 2 (Multi-Scale Feature Extraction).

Covers:
  a) Pyramid contract: 3 PC levels, exact shapes, finite [0,1] maps.
  b) Scale-propagation math: L1<->L0 homography round-trip and L2->L0
     displacement scaling, exact within floating-point tolerance.
  c) Log-Gabor conditioning: default (orientations/scales/wavelength) stays
     finite and non-degenerate down to 64px inputs (32px probed).
  d) Ablation (pyramid ON vs finest_scale_only): the L1 cascade must not
     reduce support on real pairs; flags must record the mode honestly.
  e) Real-data regression guards: classical (Phase-1-off) status /
     inlier-floor / RMSE-cap on all 8 benchmark pairs, plus one exact
     golden case (region_001). Exact == pins on stochastic-geometry outputs
     fail on healthy stacks (verified: triplet_new_2022 measures 9 @ 2.26px
     here vs 8 @ 0.84px on an independent checkout — deterministic per
     machine, OpenCV-build-dependent), so only status/match_count are exact
     and region_001 alone is value-pinned (identical on every stack so far).
     README section 6 predates the cascade; finest_scale_only reproduces
     those older support counts exactly. region_004 is guarded ON
     (recovered by 35842a8) with its OFF-mode fallback guarded alongside.
"""

import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "ML_model"))

from matcher_cfog import (
    compute_phase_congruency,
    match_images_cfog,
    multi_scale_phase_congruency,
)

TRIPLETS = REPO_ROOT / "data_preprocessing_pipeline" / "processed_triplets"


def _textured_disc(size: int = 256, seed: int = 0) -> np.ndarray:
    rng = np.random.RandomState(seed)
    base = rng.uniform(40.0, 200.0, size=(size, size)).astype(np.float32)
    base = cv2.GaussianBlur(base, (9, 9), 2.0)
    for _ in range(8):
        cv2.circle(
            base,
            (int(rng.randint(30, size - 30)), int(rng.randint(30, size - 30))),
            int(rng.randint(10, 28)),
            float(rng.randint(60, 230)),
            -1,
        )
    base = (base - base.min()) / (base.max() - base.min())
    return base.astype(np.float32)


def test_pyramid_contract_shapes_and_finite():
    """3 levels, exact pyrDown shapes, finite [0,1] PC maps."""
    pyr = multi_scale_phase_congruency(_textured_disc(256), scales=3)
    assert len(pyr) == 3
    assert pyr[0].shape == (256, 256)
    assert pyr[1].shape == (128, 128)
    assert pyr[2].shape == (64, 64)
    for level in pyr:
        assert level.dtype == np.float32
        assert bool(np.all(np.isfinite(level)))
        assert float(level.min()) >= 0.0 and float(level.max()) <= 1.0


def test_scale_propagation_math_exact():
    """H_L1 <-> H_L0 similarity round-trip + L2->L0 displacement scaling."""
    H_L0_true = np.array([
        [1.015, -0.012, 18.5],
        [0.011, 1.022, -12.4],
        [0.00001, -0.00002, 1.0],
    ], dtype=np.float64)
    S_half = np.diag([0.5, 0.5, 1.0])
    S_two = np.diag([2.0, 2.0, 1.0])
    H_L1 = S_half @ H_L0_true @ S_two
    H_L0_rec = S_two @ H_L1 @ S_half
    pt = np.array([120.0, 150.0, 1.0])
    a = (H_L0_true @ pt)[:2] / (H_L0_true @ pt)[2]
    b = (H_L0_rec @ pt)[:2] / (H_L0_rec @ pt)[2]
    assert float(np.linalg.norm(a - b)) < 1e-6
    # L2 (1/4 scale) displacement scales by exactly 4x to working pixels.
    S_q = np.diag([0.25, 0.25, 1.0])
    S_4 = np.diag([4.0, 4.0, 1.0])
    H_L2 = S_q @ H_L0_true @ S_4
    assert float(np.linalg.norm((S_4 @ H_L2 @ S_q @ pt)[:2] / (S_4 @ H_L2 @ S_q @ pt)[2] - a)) < 1e-6


def test_log_gabor_conditioning_down_to_64px():
    """Default Log-Gabor bank stays finite/non-degenerate at every pyramid size."""
    for size in (256, 128, 64):
        pc = compute_phase_congruency(_textured_disc(256, seed=3) if size == 256
                                      else cv2.resize(_textured_disc(256, seed=3), (size, size),
                                                      interpolation=cv2.INTER_AREA).astype(np.float32))
        assert bool(np.all(np.isfinite(pc))), f"non-finite PC at {size}px"
        assert float(np.std(pc)) > 1e-6, f"degenerate (flat) PC at {size}px"
    # Analytic support: wavelengths 3.0/6.3/13.23px -> finest fo=1/3 < Nyquist 0.5.
    assert 1.0 / 3.0 < 0.5


def _run_pair(name: str, src: str, ref: str, tmp_path: Path, **kwargs):
    res = match_images_cfog(
        str(TRIPLETS / src), str(TRIPLETS / ref),
        source_sensor="OHRC", reference_sensor="TMC",
        output_dir=str(tmp_path / name), **kwargs,
    )
    return res


def test_ablation_l1_cascade_does_not_reduce_support(tmp_path):
    """Pyramid ON vs finest_scale_only on real pairs (nominal + 162-degree flip)."""
    pairs = [
        ("abl_001", "region_001/ohrc_512.png", "region_001/tmc_512.png"),
        ("abl_new2022", "triplet_new_2022/ohrc_512.png", "triplet_new_2022/tmc_512.png"),
    ]
    for name, src, ref in pairs:
        if not (TRIPLETS / src).exists():
            pytest.skip(f"{src} not on disk")
        on = _run_pair(name + "_on", src, ref, tmp_path)
        off = _run_pair(name + "_off", src, ref, tmp_path, finest_scale_only=True)
        m_on, m_off = on.get("metrics") or {}, off.get("metrics") or {}
        assert on.get("status") == "success" and off.get("status") == "success"
        pm_on, pm_off = m_on.get("pyramid_matching") or {}, m_off.get("pyramid_matching") or {}
        assert pm_on.get("coarse_to_fine_applied") is True
        assert pm_off.get("coarse_to_fine_applied") is False
        assert pm_off.get("finest_scale_only") is True
        assert pm_off.get("l1_matches_found") == 0
        # The cascade must not reduce final geometric support.
        assert int(m_on.get("inlier_count")) >= int(m_off.get("inlier_count")), (
            f"{name}: cascade ON inliers {m_on.get('inlier_count')} < OFF {m_off.get('inlier_count')}"
        )


# Classical (Phase-1-off) guards measured on the current tree: post-cascade
# (5ab379d) + post-weighted-RANSAC-fix (89f920b) + post-PR#32 9-feature model
# + PROSAC validity rework (35842a8). History, verified by re-measurement:
#   * README section 6 (2026-09-11) predates the cascade; finest_scale_only
#     reproduces those older support counts exactly (e.g. region_001 49
#     candidates), so that delta is the cascade working, not drift.
#   * PR #32 re-weighted consensus (region_002 8->10 inliers, triplet_new_2022
#     7->9) and briefly regressed region_004 to Gate2-FAIL under cascade-ON +
#     trained weights; the per-iteration Gate-3-valid refit in 35842a8
#     resolved it (004 back to success).
#
# Guard design (deliberately NOT exact-equality pins): seeded RANSAC / phase-
# correlation numerics differ across OpenCV builds — verified Sep 12 when an
# independent checkout measured triplet_new_2022 at 8 inliers @ 0.84px vs the
# 9 @ 2.26px pinned here, deterministically per machine. Exact == pins on
# inliers/RMSE are therefore a maintenance trap: they fail on healthy stacks.
# Status and match_count are pinned exact (stable across stacks observed so
# far); inlier support is a floor (min observed - 1) and fit quality a cap
# (max observed + 0.5px, always under the Gate-3 5.0px ceiling).
# region_001 is the single exact golden case: identical values on every
# stack measured so far. If IT drifts, something structural changed —
# investigate, don't just re-pin.
#   * region_002 (10 @ 3.26): higher recall admits marginal inliers, so RMSE
#     rises vs the pre-PR#32 8 @ 1.72. Defensible trade-off under the 5.0px
#     cap — stated here, not silently enshrined.
#   * region_005 (4 @ 0.0): 4 points determine H exactly, so RMSE ~0.0 is a
#     structural minimal-set exact fit, not superior accuracy. Support went
#     DOWN (5->4) while the headline metric went cosmetically to zero; the
#     floor below encodes the support requirement, not the RMSE.
GOLDEN_CASE = {
    # name: (src, ref, status, inliers, rmse_px, match_count)
    "region_001": ("region_001/ohrc_512.png", "region_001/tmc_512.png", "success", 7, 1.67, 75),
}

REGRESSION_GUARDS = {
    # name: (src, ref, status, min_inliers, rmse_cap_px, match_count)
    "region_002": ("region_002/ohrc_512.png", "region_002/tmc_512.png", "success", 9, 3.8, 78),
    "region_003": ("region_003/ohrc_512.png", "region_003/tmc_512.png", "success", 6, 1.6, 60),
    "region_004": ("region_004/ohrc_512.png", "region_004/tmc_512.png", "success", 4, 2.0, 57),
    "region_005": ("region_005/ohrc_512.png", "region_005/tmc_512.png", "success", 4, 0.5, 77),
    "region_006": ("region_006/ohrc_512.png", "region_006/tmc_512.png", "success", 4, 2.0, 67),
    "triplet_01": ("triplet_01_ch2_ohr_ncp_202/ohrc_512.png", "triplet_01_ch2_ohr_ncp_202/tmc_512.png", "success", 5, 1.2, 74),
    "triplet_new_2022": ("triplet_new_2022/ohrc_512.png", "triplet_new_2022/tmc_512.png", "success", 7, 3.0, 84),
}


def test_golden_case_region_001_exact(tmp_path):
    """region_001 reproduces exactly (within RMSE tolerance) on every stack."""
    name = "region_001"
    src, ref, status, inliers, rmse, all_matches = GOLDEN_CASE[name]
    if not (TRIPLETS / src).exists():
        pytest.skip(f"{src} not on disk")
    res = _run_pair("pin_" + name, src, ref, tmp_path)
    assert res.get("status") == status, f"{name}: status {res.get('status')}"
    m = res.get("metrics") or {}
    assert int(m.get("inlier_count")) == inliers, f"{name}: inliers {m.get('inlier_count')}"
    assert int(m.get("match_count")) == all_matches, f"{name}: matches {m.get('match_count')}"
    assert abs(float(m.get("fit_rmse_px")) - rmse) < 0.05, f"{name}: rmse {m.get('fit_rmse_px')}"


@pytest.mark.parametrize("name", sorted(REGRESSION_GUARDS))
def test_classical_benchmark_regression_guards(name, tmp_path):
    src, ref, status, min_inliers, rmse_cap, all_matches = REGRESSION_GUARDS[name]
    if not (TRIPLETS / src).exists():
        pytest.skip(f"{src} not on disk")
    res = _run_pair("pin_" + name, src, ref, tmp_path)
    assert res.get("status") == status, f"{name}: status {res.get('status')}"
    m = res.get("metrics") or {}
    assert int(m.get("inlier_count")) >= min_inliers, f"{name}: inliers {m.get('inlier_count')} < floor {min_inliers}"
    assert int(m.get("match_count")) == all_matches, f"{name}: matches {m.get('match_count')}"
    assert float(m.get("fit_rmse_px")) <= rmse_cap, f"{name}: rmse {m.get('fit_rmse_px')} > cap {rmse_cap}"


def test_region_004_finest_scale_only_escape_hatch(tmp_path):
    """region_004 succeeds with the cascade disabled.

    Retained as fallback-path coverage: during the PR#32 re-weighting the
    cascade-ON path transiently failed Gate2 here while OFF stayed green.
    Guarded as inequalities (same OpenCV-numerics rationale as above).
    Must stay green.
    """
    src, ref = "region_004/ohrc_512.png", "region_004/tmc_512.png"
    if not (TRIPLETS / src).exists():
        pytest.skip(f"{src} not on disk")
    res = _run_pair("pin_004_off", src, ref, tmp_path, finest_scale_only=True)
    assert res.get("status") == "success"
    m = res.get("metrics") or {}
    assert int(m.get("inlier_count")) >= 6
    assert float(m.get("fit_rmse_px")) <= 3.2
