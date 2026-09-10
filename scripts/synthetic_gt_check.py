#!/usr/bin/env python3
"""
scripts/synthetic_gt_check.py — CI-ONLY synthetic ground-truth regression check.

*** THIS IS A SOFTWARE REGRESSION TEST, NOT BENCHMARK EVIDENCE ***
It warps a real tile by a KNOWN affine transform (+ mild gamma), runs the
matcher, and asserts the pipeline recovers approximately that transform.
Recovering a known synthetic warp proves the estimator machinery works; it
says NOTHING about cross-sensor/illumination robustness (that evidence lives
in docs/benchmark_sun_gap.md, measured on real orbital pairs). Never cite
numbers from this script as benchmark results.

Usage:
    python scripts/synthetic_gt_check.py [--src <tile.png>] [--out <dir>]
Exit code 0 on pass, 1 on fail. Prints a small JSON summary to stdout.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "ML_model"))

from matcher_cfog import match_images_cfog  # noqa: E402

# Known synthetic warp (rotation + scale + translation + photometric tilt).
ROT_DEG = 5.0
SCALE = 0.95
TX, TY = 20.0, 15.0
GAMMA = 0.9

# Lenient smoke thresholds: this guards against total breakage (crashes,
# empty outputs, wild transforms), NOT accuracy claims.
MIN_INLIERS = 4
MAX_CORNER_ERR_PX = 12.0


def build_synthetic_pair(src: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    h, w = src.shape[:2]
    gray = src if src.ndim == 2 else cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
    M = cv2.getRotationMatrix2D((w / 2.0, h / 2.0), ROT_DEG, SCALE)
    M[0, 2] += TX
    M[1, 2] += TY
    warped = cv2.warpAffine(gray, M, (w, h), flags=cv2.INTER_LINEAR,
                            borderMode=cv2.BORDER_REFLECT)
    f = np.clip((warped.astype(np.float32) / 255.0) ** GAMMA, 0, 1)
    H = np.vstack([M, [0.0, 0.0, 1.0]])
    return (f * 255).astype(np.uint8), H


def corner_error(H_est: np.ndarray, H_gt: np.ndarray, shape) -> float:
    h, w = shape[:2]
    corners = np.array([[[0, 0], [w, 0], [w, h], [0, h]]], dtype=np.float64)
    a = cv2.perspectiveTransform(corners, H_est).reshape(-1, 2)
    b = cv2.perspectiveTransform(corners, H_gt).reshape(-1, 2)
    return float(np.mean(np.linalg.norm(a - b, axis=1)))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--src", default=str(
        REPO_ROOT / "data_preprocessing_pipeline" / "processed_triplets"
        / "region_001" / "ohrc_512.png"))
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    import tempfile
    out = Path(args.out) if args.out else Path(tempfile.mkdtemp(prefix="synthgt_"))
    out.mkdir(parents=True, exist_ok=True)

    src = cv2.imread(args.src, cv2.IMREAD_GRAYSCALE)
    if src is None:
        print(json.dumps({"status": "error", "reason": f"cannot read {args.src}"}))
        return 1
    tgt, H_gt = build_synthetic_pair(src)
    p_src, p_tgt = out / "gt_source.png", out / "gt_target.png"
    cv2.imwrite(str(p_src), src)
    cv2.imwrite(str(p_tgt), tgt)

    res = match_images_cfog(
        str(p_src), str(p_tgt), output_dir=str(out / "work"),
        source_sensor="OHRC", reference_sensor="OHRC",
        explicit_gsd1=5.0, explicit_gsd2=5.0,
    )
    summary: dict = {"status": res.get("status"),
                     "known_warp": {"rot_deg": ROT_DEG, "scale": SCALE,
                                    "tx": TX, "ty": TY, "gamma": GAMMA}}
    ok = False
    if res.get("status") == "success" and res.get("homography") is not None:
        met = res.get("metrics") or {}
        H_est = np.asarray(res["homography"], dtype=np.float64)
        try:
            err = corner_error(H_est, H_gt, src.shape)
        except Exception:
            err = None
        summary.update({"inliers": met.get("inlier_count"),
                        "fit_rmse_px": met.get("fit_rmse_px"),
                        "corner_err_px": err})
        ok = (met.get("inlier_count", 0) >= MIN_INLIERS
              and err is not None and err <= MAX_CORNER_ERR_PX)
    summary["pass"] = bool(ok)
    print(json.dumps(summary, indent=2))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
