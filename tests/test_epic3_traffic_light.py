"""
tests/test_epic3_traffic_light.py — Epic 3: objective verification.

Asserts a successful registration's metrics.json contains:
  ssim_score (float 0-1), confidence_score (int 0-100),
  traffic_light_color ("GREEN" | "YELLOW" | "RED").
Also covers compute_ssim_within_inliers + calculate_traffic_light contracts
and the Task-5 guardrail (inlier_ratio < 0.3 -> RED, never a forced matrix).
"""
import json
import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "ML_model"))

from metrics import (
    calculate_traffic_light,
    compute_canonical_metrics,
    compute_ssim_within_inliers,
)


def _synthetic_pair():
    rng = np.random.RandomState(42)
    h, w = 256, 256
    base = np.zeros((h, w), dtype=np.uint8)
    for _ in range(25):
        cx, cy = rng.randint(30, 226), rng.randint(30, 226)
        rad = rng.randint(10, 30)
        val = int(rng.randint(120, 255))
        cv2.circle(base, (cx, cy), rad, val, -1)
        cv2.circle(base, (cx, cy), max(2, rad - 5), val // 2, -1)
    base = cv2.add(base, rng.randint(0, 25, (h, w)).astype(np.uint8))
    M = np.float32([[1, 0, 6.0], [0, 1, -4.0]])
    shifted = cv2.warpAffine(base, M, (w, h))
    return base, shifted


def test_compute_ssim_within_inliers_contract():
    ref, shifted = _synthetic_pair()
    h, w = ref.shape[:2]
    # Inliers spread across the frame -> convex hull covers most of it.
    pts = np.array([[20, 20], [w - 20, 20], [w - 20, h - 20], [20, h - 20],
                    [w // 2, h // 2], [w // 3, h // 3]], dtype=np.float64)
    # Aligned pair (identical geometry + tiny noise) must score high.
    rng = np.random.RandomState(0)
    ref_noisy = np.clip(ref.astype(np.float32) + rng.normal(0, 1.0, ref.shape), 0, 255)
    ssim_val = compute_ssim_within_inliers(ref, ref_noisy, pts)
    assert isinstance(ssim_val, float), f"ssim must be float, got {type(ssim_val)}"
    assert 0.0 <= ssim_val <= 1.0, f"ssim out of bounds: {ssim_val}"
    # Near-aligned pair must score structurally high.
    assert ssim_val > 0.5, f"aligned pair SSIM too low: {ssim_val}"
    # Misaligned (shifted, unwarped) pair must still return a valid float.
    ssim_shifted = compute_ssim_within_inliers(ref, shifted, pts)
    assert isinstance(ssim_shifted, float)
    assert 0.0 <= ssim_shifted <= 1.0
    # JSON-serializable.
    json.dumps({"ssim_score": ssim_val})


def test_calculate_traffic_light_contract():
    # GREEN: held_out < 1.5 AND ssim > 0.65 AND ratio > 0.5.
    green = calculate_traffic_light({
        "held_out_rmse": 0.8, "ssim_score": 0.8, "ssim": 0.8,
        "inlier_ratio": 0.7, "inlier_count": 40,
    })
    assert green["traffic_light_color"] == "GREEN"
    assert isinstance(green["confidence_score"], int)
    assert 90 <= green["confidence_score"] <= 100

    # YELLOW: passed guardrails but borderline.
    yellow = calculate_traffic_light({
        "held_out_rmse": 2.0, "ssim_score": 0.5, "ssim": 0.5,
        "inlier_ratio": 0.4, "inlier_count": 12,
    })
    assert yellow["traffic_light_color"] == "YELLOW"
    assert 60 <= yellow["confidence_score"] <= 89

    # RED: guardrail trip (inlier_ratio < 0.3) -> honest failure, never forced.
    red = calculate_traffic_light({
        "held_out_rmse": 5.0, "ssim_score": 0.2, "ssim": 0.2,
        "inlier_ratio": 0.1, "inlier_count": 3,
    })
    assert red["traffic_light_color"] == "RED"
    assert 0 <= red["confidence_score"] < 60
    json.dumps(red)


def test_successful_registration_metrics_json_has_traffic_light():
    from matcher_cfog import match_images_cfog

    ref, src = _synthetic_pair()
    with tempfile.TemporaryDirectory() as td:
        p_src = Path(td) / "source.png"
        p_ref = Path(td) / "reference.png"
        cv2.imwrite(str(p_src), src)
        cv2.imwrite(str(p_ref), ref)
        out = Path(td) / "out"
        res = match_images_cfog(
            str(p_src), str(p_ref), output_dir=str(out),
            explicit_gsd1=5.0, explicit_gsd2=5.0,
        )
        assert res["status"] == "success", f"registration failed: {res.get('message')}"
        metrics_path = Path(res["outputs"]["metrics"])
        assert metrics_path.exists()
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))

        for key in ("ssim_score", "confidence_score", "traffic_light_color"):
            assert key in metrics, f"metrics.json missing required key {key!r}"

        assert isinstance(metrics["ssim_score"], float)
        assert 0.0 <= metrics["ssim_score"] <= 1.0
        assert isinstance(metrics["confidence_score"], int)
        assert 0 <= metrics["confidence_score"] <= 100
        assert metrics["traffic_light_color"] in ("GREEN", "YELLOW", "RED")
        # Successful registration must never be RED via a forced matrix.
        assert metrics["traffic_light_color"] in ("GREEN", "YELLOW"), (
            f"successful registration flagged RED: {metrics}"
        )
        # Task-5 guardrail intact: honest ratio present.
        assert metrics["inlier_ratio"] >= 0.3
