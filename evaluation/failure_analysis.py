"""
evaluation/failure_analysis.py — Automated Failure Case Analysis and Diagnostics

Detects registration failure conditions (e.g. RANSAC rejection, insufficient inliers, excessive RMSE),
analyzes physical planetary remote sensing factors:
1. Low texture variance (Laplacian variance < 15)
2. Extreme solar illumination discrepancy (|Sun angle delta| > 45 deg)
3. Extreme resolution scale ratio (> 10x)
4. Deep shadow occlusion or detector saturation
Generates diagnostic logs and annotated diagnostic images.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Any, Tuple, Optional, List
import numpy as np
import cv2


def analyze_failure_case(
    img1: np.ndarray | str | Path,
    img2: np.ndarray | str | Path,
    meta1: Optional[Dict[str, Any]] = None,
    meta2: Optional[Dict[str, Any]] = None,
    failure_reason: str = "geometric_verification_failed",
    rmse_px: Optional[float] = None,
    max_rmse_threshold: float = 5.0,
    out_dir: Optional[str | Path] = None,
) -> Dict[str, Any]:
    """
    Analyzes physical remote sensing failure causes when cross-sensor registration fails.
    """
    def to_u8_gray(item):
        if isinstance(item, (str, Path)):
            raw = cv2.imread(str(item), cv2.IMREAD_GRAYSCALE)
            if raw is None:
                return np.zeros((256, 256), dtype=np.uint8)
            return raw
        arr = np.asarray(item)
        if arr.ndim == 3:
            arr = cv2.cvtColor(arr, cv2.COLOR_BGR2GRAY)
        if arr.dtype != np.uint8:
            arr = (np.clip(arr / 255.0 if arr.max() > 1.0 else arr, 0, 1) * 255.0).astype(np.uint8)
        return arr

    u8_1 = to_u8_gray(img1)
    u8_2 = to_u8_gray(img2)

    m1 = meta1 or {}
    m2 = meta2 or {}

    root_causes: List[str] = []
    diagnostics: Dict[str, Any] = {}

    # 1. Texture Variance Analysis (Laplacian Variance)
    lap1 = cv2.Laplacian(u8_1, cv2.CV_32F)
    lap2 = cv2.Laplacian(u8_2, cv2.CV_32F)
    var1 = float(np.var(lap1))
    var2 = float(np.var(lap2))

    diagnostics["texture_variance"] = {"source_laplacian_var": round(var1, 2), "reference_laplacian_var": round(var2, 2)}
    if var1 < 15.0 or var2 < 15.0:
        root_causes.append(
            f"Low texture variance (Source: {var1:.1f}, Ref: {var2:.1f} < 15.0; smooth/shadowed terrain)"
        )

    # 2. Solar Illumination Discrepancy
    az1 = float(m1.get("azimuth_deg", m1.get("sun_azimuth_deg", 45.0)))
    az2 = float(m2.get("azimuth_deg", m2.get("sun_azimuth_deg", 45.0)))
    delta_az = abs((az1 - az2 + 180.0) % 360.0 - 180.0)

    diagnostics["illumination"] = {
        "source_sun_azimuth_deg": az1,
        "reference_sun_azimuth_deg": az2,
        "delta_azimuth_deg": round(delta_az, 2),
    }
    if delta_az > 45.0:
        root_causes.append(
            f"Extreme illumination difference (Sun angle delta {delta_az:.1f} deg > 45 deg; shadow reversal)"
        )

    # 3. Ground Sample Distance (Scale Gap)
    gsd1 = float(m1.get("gsd_m", 5.0))
    gsd2 = float(m2.get("gsd_m", 5.0))
    scale_ratio = float(max(gsd1, gsd2) / max(min(gsd1, gsd2), 1e-4))

    diagnostics["scale"] = {"source_gsd_m": gsd1, "reference_gsd_m": gsd2, "scale_ratio": round(scale_ratio, 2)}
    if scale_ratio > 10.0:
        root_causes.append(f"Extreme scale ratio ({scale_ratio:.1f}x > 10x scale disparity)")

    # 4. Deep Shadows / Saturation Occlusion
    dark_frac1 = float(np.mean(u8_1 < 10))
    dark_frac2 = float(np.mean(u8_2 < 10))
    sat_frac1 = float(np.mean(u8_1 > 245))
    sat_frac2 = float(np.mean(u8_2 > 245))

    diagnostics["dynamic_range"] = {
        "source_dark_fraction": round(dark_frac1, 3),
        "reference_dark_fraction": round(dark_frac2, 3),
        "source_saturated_fraction": round(sat_frac1, 3),
        "reference_saturated_fraction": round(sat_frac2, 3),
    }
    if dark_frac1 > 0.40 or dark_frac2 > 0.40:
        root_causes.append("Severe shadow occlusion (>40% of pixels in deep permanent shadow)")
    if sat_frac1 > 0.30 or sat_frac2 > 0.30:
        root_causes.append("Detector saturation (>30% pixels clipped at maximum radiance)")

    # 5. Reprojection RMSE Check
    if rmse_px is not None and rmse_px > max_rmse_threshold:
        root_causes.append(
            f"Reprojection error ({rmse_px:.2f} px) exceeded maximum acceptance gate ({max_rmse_threshold:.2f} px)"
        )

    if not root_causes:
        root_causes.append(f"Geometric verification rejected: {failure_reason}")

    report = {
        "status": "failure_diagnosed",
        "failure_reason": failure_reason,
        "identified_root_causes": root_causes,
        "primary_root_cause": root_causes[0],
        "diagnostics": diagnostics,
    }

    # Save diagnostic artifacts if out_dir specified
    if out_dir:
        out_p = Path(out_dir)
        out_p.mkdir(parents=True, exist_ok=True)
        json_path = out_p / "failure_diagnostic_report.json"
        with open(json_path, "w") as f:
            json.dump(report, f, indent=4)

        # Generate side-by-side failure preview
        h = max(u8_1.shape[0], u8_2.shape[0])
        w = u8_1.shape[1] + u8_2.shape[1]
        side = np.zeros((h, w), dtype=np.uint8)
        side[:u8_1.shape[0], :u8_1.shape[1]] = u8_1
        side[:u8_2.shape[0], u8_1.shape[1]:u8_1.shape[1] + u8_2.shape[1]] = u8_2
        side_bgr = cv2.cvtColor(side, cv2.COLOR_GRAY2BGR)

        # Annotate
        cv2.putText(side_bgr, f"FAILED: {root_causes[0][:50]}", (20, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.imwrite(str(out_p / "failure_diagnostic_preview.png"), side_bgr)
        report["diagnostic_json"] = str(json_path)
        report["diagnostic_image"] = str(out_p / "failure_diagnostic_preview.png")

    return report
