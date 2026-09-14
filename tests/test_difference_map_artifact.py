"""
tests/test_difference_map_artifact.py — SIH Task 4/6 difference-map artifact (Task 6).

On successful registration the pipeline must write:
  difference_map.png, metrics.json, registered/warped raster, matches JSON —
and metrics.json must contain in_sample_rmse, held_out_rmse, inlier_ratio,
uniformity_score, cyclic_rmse, ssim_score, confidence_score,
traffic_light_color. Failure paths must not fabricate outputs (covered by
test_insufficient_matches_never_creates_fake_points).
"""
import json
import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "ML_model"))

REQUIRED_METRIC_KEYS = (
    "in_sample_rmse",
    "held_out_rmse",
    "inlier_ratio",
    "uniformity_score",
    "cyclic_rmse",
    "ssim_score",
    "confidence_score",
    "traffic_light_color",
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
    return base, cv2.warpAffine(base, M, (w, h))


def test_difference_map_artifact_generated_on_success():
    from matcher_cfog import match_images_cfog

    src, ref = _synthetic_pair()
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
        outputs = res["outputs"]

        diff_p = out / "difference_map.png"
        assert diff_p.exists(), "difference_map.png not written on success"
        assert diff_p.stat().st_size > 0, "difference_map.png is empty"

        metrics_p = Path(outputs["metrics"])
        assert metrics_p.exists()
        metrics = json.loads(metrics_p.read_text(encoding="utf-8"))
        for key in REQUIRED_METRIC_KEYS:
            assert key in metrics, f"metrics.json missing required key {key!r}"

        assert Path(outputs["registered_raster"]).exists()
        assert Path(outputs["matches"]).exists()
        matches = json.loads(Path(outputs["matches"]).read_text(encoding="utf-8"))
        assert isinstance(matches, list) and len(matches) >= 4
