"""
benchmark_subpixel.py — Rigorous Multi-Directional Synthetic Sub-Pixel Precision Benchmark

Evaluates the Fourier Phase Correlation 2D Log-Gaussian peak refinement algorithm across:
- 8 Cardinal and Intercardinal Directions (+X, -X, +Y, -Y, +X/+Y, +X/-Y, -X/+Y, -X/-Y)
- 6 Fractional Magnitudes (0.05, 0.10, 0.20, 0.30, 0.50, 0.75 px)
- Five separately seeded synthetic terrain realizations per combination.

Reports full statistical error distributions (MAE, RMSE, Median, 95th Percentile, Biases, Threshold Fractions).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, Any, List, Tuple

import numpy as np
import cv2

# Add ML_model to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "ML_model"))
from matcher_cfog import subpixel_phase_correlation


DIRECTIONS: Dict[str, Tuple[float, float]] = {
    "+X": (1.0, 0.0),
    "-X": (-1.0, 0.0),
    "+Y": (0.0, 1.0),
    "-Y": (0.0, -1.0),
    "diag_+X_+Y": (1.0 / np.sqrt(2.0), 1.0 / np.sqrt(2.0)),
    "diag_+X_-Y": (1.0 / np.sqrt(2.0), -1.0 / np.sqrt(2.0)),
    "diag_-X_+Y": (-1.0 / np.sqrt(2.0), 1.0 / np.sqrt(2.0)),
    "diag_-X_-Y": (-1.0 / np.sqrt(2.0), -1.0 / np.sqrt(2.0)),
}

DIRECTION_SEEDS: Dict[str, int] = {
    "+X": 11,
    "-X": 23,
    "+Y": 37,
    "-Y": 41,
    "diag_+X_+Y": 53,
    "diag_+X_-Y": 67,
    "diag_-X_+Y": 79,
    "diag_-X_-Y": 97,
}


def generate_synthetic_lunar_patch(size: int = 64, seed: int = 42) -> np.ndarray:
    """
    Generates a realistic synthetic lunar terrain patch with multi-scale cratering,
    fractal elevation noise, and directional shading.
    """
    rng = np.random.RandomState(seed)
    y, x = np.mgrid[:size, :size].astype(np.float32)

    # Base low-frequency mare topography
    patch = 30.0 * np.sin(x / 20.0) * np.cos(y / 25.0)

    # Multi-scale craters
    num_craters = 8
    for _ in range(num_craters):
        cx = rng.uniform(15, size - 15)
        cy = rng.uniform(15, size - 15)
        radius = rng.uniform(5, 18)
        depth = rng.uniform(30, 80)

        dist_sq = (x - cx) ** 2 + (y - cy) ** 2
        bowl = -depth * np.exp(-dist_sq / (2.0 * (radius * 0.7) ** 2))
        rim = (depth * 0.35) * np.exp(-((np.sqrt(dist_sq) - radius) ** 2) / (2.0 * (radius * 0.25) ** 2))
        patch += bowl + rim

    # Shading (solar azimuth ~45 deg)
    sun_angle = np.radians(45.0)
    gx = cv2.Sobel(patch, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(patch, cv2.CV_32F, 0, 1, ksize=3)
    shading = (gx * np.cos(sun_angle) + gy * np.sin(sun_angle))

    # Regolith high-frequency noise
    regolith_noise = rng.normal(0, 2.0, (size, size)).astype(np.float32)
    final_patch = 128.0 + shading + regolith_noise

    p_min, p_max = float(np.min(final_patch)), float(np.max(final_patch))
    normalized = (final_patch - p_min) / max(p_max - p_min, 1e-5)
    return (normalized * 255.0).astype(np.float32)


def apply_fourier_fractional_shift(
    image: np.ndarray, shift_x: float, shift_y: float
) -> np.ndarray:
    """
    Shifts an image by exact sub-pixel amounts using the Fourier Shift Theorem:
    F{f(x - dx, y - dy)} = F{f(x, y)} * exp(-2j * pi * (u*dx/W + v*dy/H))
    Guarantees mathematically exact fractional translation without interpolation artifacts.
    """
    h, w = image.shape[:2]
    F = np.fft.fft2(image)

    y_freq = np.fft.fftfreq(h).astype(np.float32)
    x_freq = np.fft.fftfreq(w).astype(np.float32)
    xv, yv = np.meshgrid(x_freq, y_freq)

    phase_ramp = np.exp(-2j * np.pi * (xv * shift_x + yv * shift_y))
    shifted = np.real(np.fft.ifft2(F * phase_ramp))
    return np.clip(shifted, 0.0, 255.0).astype(np.float32)


def run_subpixel_benchmark(
    tested_shifts: List[float] = [0.05, 0.10, 0.20, 0.30, 0.50, 0.75],
    num_seeds_per_combination: int = 5,
    patch_size: int = 64,
) -> Dict[str, Any]:
    """
    Executes full multi-directional sub-pixel benchmark across 8 directions,
    6 displacement magnitudes, and five separately seeded synthetic terrain realizations.
    """
    all_errors: List[float] = []
    dx_errors: List[float] = []
    dy_errors: List[float] = []

    per_direction_breakdown: Dict[str, Any] = {}
    per_magnitude_breakdown: Dict[str, Any] = {}

    for dir_name, (u_dir, v_dir) in DIRECTIONS.items():
        dir_errors: List[float] = []

        for shift_mag in tested_shifts:
            true_dx = float(shift_mag * u_dir)
            true_dy = float(shift_mag * v_dir)

            for seed_idx in range(num_seeds_per_combination):
                seed = 1000 * seed_idx + int(round(shift_mag * 100)) + DIRECTION_SEEDS[dir_name]
                base_patch = generate_synthetic_lunar_patch(patch_size, seed=seed)
                shifted_patch = apply_fourier_fractional_shift(base_patch, true_dx, true_dy)

                est_dx, est_dy, peak, valid = subpixel_phase_correlation(base_patch, shifted_patch)

                if valid:
                    err_x = est_dx - true_dx
                    err_y = est_dy - true_dy
                    err_mag = float(np.hypot(err_x, err_y))

                    all_errors.append(err_mag)
                    dx_errors.append(err_x)
                    dy_errors.append(err_y)
                    dir_errors.append(err_mag)

                    mag_key = f"{shift_mag:.2f}_px"
                    per_magnitude_breakdown.setdefault(mag_key, []).append(err_mag)

        per_direction_breakdown[dir_name] = {
            "mean_error_px": round(float(np.mean(dir_errors)), 4),
            "median_error_px": round(float(np.median(dir_errors)), 4),
            "max_error_px": round(float(np.max(dir_errors)), 4),
        }

    mag_summary: Dict[str, Any] = {}
    for mag_key, err_list in per_magnitude_breakdown.items():
        arr_m = np.array(err_list)
        mag_summary[mag_key] = {
            "mean_error_px": round(float(np.mean(arr_m)), 4),
            "median_error_px": round(float(np.median(arr_m)), 4),
            "rmse_px": round(float(np.sqrt(np.mean(arr_m ** 2))), 4),
            "p95_error_px": round(float(np.percentile(arr_m, 95)), 4),
            "max_error_px": round(float(np.max(arr_m)), 4),
        }

    all_arr = np.array(all_errors)
    summary = {
        "benchmark_name": "Multi-Directional Synthetic Lunar Terrain Sub-Pixel Benchmark",
        "total_trials": len(all_errors),
        "tested_directions": list(DIRECTIONS.keys()),
        "tested_shifts_px": tested_shifts,
        "mean_absolute_error_px": round(float(np.mean(all_arr)), 4),
        "median_absolute_error_px": round(float(np.median(all_arr)), 4),
        "rmse_px": round(float(np.sqrt(np.mean(all_arr ** 2))), 4),
        "p95_error_px": round(float(np.percentile(all_arr, 95)), 4),
        "max_absolute_error_px": round(float(np.max(all_arr)), 4),
        "bias_x_px": round(float(np.mean(dx_errors)), 4),
        "bias_y_px": round(float(np.mean(dy_errors)), 4),
        "fraction_below_0_10px": round(float(np.mean(all_arr < 0.10)), 4),
        "fraction_below_0_20px": round(float(np.mean(all_arr < 0.20)), 4),
        "fraction_below_0_25px": round(float(np.mean(all_arr < 0.25)), 4),
        "fraction_below_0_50px": round(float(np.mean(all_arr < 0.50)), 4),
        "fraction_below_1_00px": round(float(np.mean(all_arr < 1.00)), 4),
        "per_direction_breakdown": per_direction_breakdown,
        "per_magnitude_breakdown": mag_summary,
    }

    return summary


def main():
    parser = argparse.ArgumentParser(description="Run strengthened synthetic sub-pixel precision benchmark.")
    parser.add_argument("--output", type=str, default="benchmarks/subpixel_benchmark_report.json")
    args = parser.parse_args()

    out_p = Path(args.output)
    out_p.parent.mkdir(parents=True, exist_ok=True)

    print("Running multi-directional synthetic sub-pixel benchmark across 8 directions & 6 magnitudes...")
    results = run_subpixel_benchmark()

    with open(out_p, "w") as f:
        json.dump(results, f, indent=4)

    print(f"\n=================================================================")
    print(f" Sub-Pixel Multi-Directional Benchmark Results")
    print(f" Total Trials:          {results['total_trials']}")
    print(f" Mean Absolute Error:   {results['mean_absolute_error_px']} px")
    print(f" Median Absolute Error: {results['median_absolute_error_px']} px")
    print(f" RMSE:                  {results['rmse_px']} px")
    print(f" 95th Percentile:       {results['p95_error_px']} px")
    print(f" Max Error:             {results['max_absolute_error_px']} px")
    print(f" Bias X / Bias Y:       {results['bias_x_px']} px / {results['bias_y_px']} px")
    print(f" Fraction < 0.10 px:    {results['fraction_below_0_10px'] * 100:.1f}%")
    print(f" Fraction < 0.20 px:    {results['fraction_below_0_20px'] * 100:.1f}%")
    print(f" Fraction < 0.25 px:    {results['fraction_below_0_25px'] * 100:.1f}%")
    print(f" Fraction < 0.50 px:    {results['fraction_below_0_50px'] * 100:.1f}%")
    print(f" Fraction < 1.00 px:    {results['fraction_below_1_00px'] * 100:.1f}%")
    print(f"=================================================================\n")
    print(f"Report saved to: {out_p}")


if __name__ == "__main__":
    main()
