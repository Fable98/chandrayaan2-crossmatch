"""
tests/test_matcher_magsac.py — Unit tests for MAGSAC++ (cv2.USAC_MAGSAC) outlier rejection option.
"""

import sys
import tempfile
from pathlib import Path
import numpy as np
import cv2
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "ML_model"))

from matcher_cfog import match_images_cfog


def _generate_synthetic_test_pair(tmpdir: Path) -> tuple[Path, Path]:
    """Creates a pair of synthetic test images with known translation for registration testing."""
    h, w = 256, 256
    img1 = np.zeros((h, w), dtype=np.uint8)

    # Draw craters / distinct features
    np.random.seed(42)
    for _ in range(25):
        cx = np.random.randint(30, w - 30)
        cy = np.random.randint(30, h - 30)
        rad = np.random.randint(10, 25)
        val = int(np.random.randint(100, 240))
        cv2.circle(img1, (cx, cy), rad, val, -1)
        cv2.circle(img1, (cx, cy), max(2, rad - 5), int(val * 0.4), -1)

    img1 = cv2.GaussianBlur(img1, (5, 5), 1.0)

    # Reference is shifted by dx=+6, dy=-4
    M = np.float32([[1, 0, 6], [0, 1, -4]])
    img2 = cv2.warpAffine(img1, M, (w, h))

    p1 = tmpdir / "source.png"
    p2 = tmpdir / "reference.png"
    cv2.imwrite(str(p1), img1)
    cv2.imwrite(str(p2), img2)
    return p1, p2


def test_magsac_outlier_rejection_option():
    """
    Validates that match_images_cfog accepts outlier_method='magsac',
    correctly configures the estimator, logs the method in metrics,
    and successfully registers the pair.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        p1, p2 = _generate_synthetic_test_pair(tmp_path)
        out_dir = tmp_path / "out_magsac"

        res = match_images_cfog(
            p1,
            p2,
            output_dir=out_dir,
            outlier_method="magsac",
            explicit_gsd1=1.0,
            explicit_gsd2=1.0,
            grid_size=6,
        )

        assert res is not None
        assert "outlier_method" in res
        assert res["outlier_method"] == "magsac"
        assert "metrics" in res and res["metrics"] is not None
        assert res["metrics"].get("outlier_method") == "magsac"
        assert res["status"] == "success"
        assert res["metrics"]["inlier_count"] >= 4


def test_default_outlier_rejection_is_ransac():
    """
    Validates that the default behavior remains 'ransac' to ensure complete
    backward compatibility with existing verified results.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        p1, p2 = _generate_synthetic_test_pair(tmp_path)
        out_dir = tmp_path / "out_ransac"

        res = match_images_cfog(
            p1,
            p2,
            output_dir=out_dir,
            explicit_gsd1=1.0,
            explicit_gsd2=1.0,
            # outlier_method omitted; defaults to "ransac"
            grid_size=6,
        )

        assert res is not None
        assert "outlier_method" in res
        assert res["outlier_method"] == "ransac"
        assert "metrics" in res and res["metrics"] is not None
        assert res["metrics"].get("outlier_method") == "ransac"
        assert res["status"] == "success"
