"""Quality Gate 3 must not be poisoned by the weighted-sampling 4-point DLT."""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "ML_model"))

from matcher_cfog import estimate_weighted_homography  # noqa: E402
from metrics import verify_transformation_quality  # noqa: E402


def test_weighted_sampling_does_not_return_ill_conditioned_h(monkeypatch):
    """High-weight collinear outliers used to yield a Gate-3-invalid 4-point H.

    Force the OpenCV ``weights=`` kwarg path to TypeError (OpenCV 4/5) so the
    manual sampler runs, then assert the returned H still passes Gate 3.
    """
    rng = np.random.default_rng(42)
    H_true = np.array([[1.0, 0.0, 12.0], [0.0, 1.0, -8.0], [0.0, 0.0, 1.0]], dtype=np.float64)

    xs, ys = np.meshgrid(np.linspace(40, 470, 8), np.linspace(40, 470, 8))
    src_good = np.column_stack([xs.ravel(), ys.ravel()]).astype(np.float32)
    dst_h = cv2.perspectiveTransform(src_good.reshape(-1, 1, 2), H_true).reshape(-1, 2)
    dst_good = dst_h + rng.normal(0.0, 0.15, dst_h.shape).astype(np.float32)

    src_bad = np.array([[10.0 + 0.2 * i, 50.0] for i in range(6)], dtype=np.float32)
    dst_bad = np.array([[400.0, 10.0 + 80.0 * i] for i in range(6)], dtype=np.float32)

    pts1 = np.vstack([src_good, src_bad])
    pts2 = np.vstack([dst_good, dst_bad])
    weights = np.concatenate([
        np.full(len(src_good), 0.05, dtype=np.float64),
        np.full(len(src_bad), 1.0, dtype=np.float64),
    ])

    real_fh = cv2.findHomography

    def _no_native_weights(*args, **kwargs):
        if "weights" in kwargs:
            raise TypeError("OpenCV build has no weights kwarg")
        return real_fh(*args, **kwargs)

    monkeypatch.setattr(cv2, "findHomography", _no_native_weights)

    H, mask, tag = estimate_weighted_homography(
        pts1, pts2, weights,
        estimator_method=cv2.RANSAC,
        ransac_reproj_threshold=5.0,
        image_shape=(512, 512),
        rng_seed=42,
        n_iters=800,
    )
    assert H is not None and mask is not None
    assert int(np.sum(mask)) >= 4
    assert tag in {
        "sampling_weighted_dlt",
        "sampling_unweighted_dlt",
        "standard_ransac",
        "native_weights",
    }
    check = verify_transformation_quality(H, (512, 512))
    assert check["is_valid"], f"Gate 3 rejected weighted H ({check['reason']}, tag={tag})"
    # Translation should stay near the true warp, not a collapsed 4-point DLT.
    assert abs(float(H[0, 2]) - 12.0) < 6.0
    assert abs(float(H[1, 2]) + 8.0) < 6.0
