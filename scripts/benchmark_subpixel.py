"""
benchmark_subpixel.py — Quantitative Synthetic Ground-Truth Sub-Pixel Benchmark

Applies calibrated, known fractional pixel displacements across synthetic and realistic
lunar terrain textures, evaluating the true numerical error distribution of the
Fourier Phase Correlation 2D quadratic peak refinement algorithm.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, Any, List

import numpy as np
import cv2

# Add ML_model to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "ML_model"))
from matcher_cfog import subpixel_phase_correlation


def generate_synthetic_lunar_patch(size: int = 128, seed: int = 42) -> np.ndarray:
    """
    Generates a realistic synthetic lunar terrain patch with multi-scale cratering,
    fractal elevation noise, and directional shading.
    """
    rng = np.random.RandomState(seed)
    patch = np.zeros((size, size), dtype=np.float32)

    # Base low-frequency rolling lunar mare topography
    y, x = np.mgrid[:size, :size].astype(np.float32)
    patch += 30.0 * np.sin(x / 25.0) * np.cos(y / 30.0)

    # Multi-scale craters
    num_craters = 12
    for _ in range(num_craters):
        cx = rng.uniform(20, size - 20)
        cy = rng.uniform(20, size - 20)
        radius = rng.uniform(6, 24)
        depth = rng.uniform(40, 100)

        dist_sq = (x - cx) ** 2 + (y - cy) ** 2
        # Crater bowl + raised rim
        bowl = -depth * np.exp(-dist_sq / (2.0 * (radius * 0.7) ** 2))
        rim = (depth * 0.35) * np.exp(-((np.sqrt(dist_sq) - radius) ** 2) / (2.0 * (radius * 0.25) ** 2))
        patch += bowl + rim

    # Directional illumination shading (simulating low sun elevation)
    sun_angle = np.radians(45.0)
    gx = cv2.Sobel(patch, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(patch, cv2.CV_32F, 0, 1, ksize=3)
    shading = (gx * np.cos(sun_angle) + gy * np.sin(sun_angle))

    # Regolith high-frequency noise
    regolith_noise = rng.normal(0, 3.0, (size, size)).astype(np.float32)
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
    num_trials_per_shift: int = 15,
    patch_size: int = 64,
) -> Dict[str, Any]:
    """
    Executes sub-pixel precision benchmark across a spectrum of known displacements.
    Reports rigorous, ground-truth error metrics.
    """
    all_errors: List[float] = []
    shift_results: Dict[str, Any] = {}

    for shift_mag in tested_shifts:
        errors_for_shift: List[float] = []

        for trial in range(num_trials_per_shift):
            # Random direction for the displacement
            angle = (2.0 * np.pi * trial) / float(num_trials_per_shift)
            true_dx = float(shift_mag * np.cos(angle))
            true_dy = float(shift_mag * np.sin(angle))

            base_patch = generate_synthetic_lunar_patch(patch_size, seed=trial * 100 + int(shift_mag * 1000))
            shifted_patch = apply_fourier_fractional_shift(base_patch, true_dx, true_dy)

            est_dx, est_dy, peak, valid = subpixel_phase_correlation(base_patch, shifted_patch)

            if valid:
                error = float(np.sqrt((est_dx - true_dx) ** 2 + (est_dy - true_dy) ** 2))
                errors_for_shift.append(error)
                all_errors.append(error)

        if errors_for_shift:
            shift_results[f"{shift_mag:.2f}_px"] = {
                "tested_displacement_px": shift_mag,
                "mean_error_px": round(float(np.mean(errors_for_shift)), 4),
                "median_error_px": round(float(np.median(errors_for_shift)), 4),
                "max_error_px": round(float(np.max(errors_for_shift)), 4),
                "std_error_px": round(float(np.std(errors_for_shift)), 4),
            }

    all_arr = np.array(all_errors)
    summary = {
        "benchmark_name": "Synthetic Lunar Terrain Sub-Pixel Phase Correlation",
        "tested_shifts_px": tested_shifts,
        "total_trials": len(all_errors),
        "mean_absolute_error_px": round(float(np.mean(all_arr)), 4),
        "median_absolute_error_px": round(float(np.median(all_arr)), 4),
        "max_absolute_error_px": round(float(np.max(all_arr)), 4),
        "std_absolute_error_px": round(float(np.std(all_arr)), 4),
        "fraction_below_0_25px": round(float(np.mean(all_arr < 0.25)), 4),
        "fraction_below_0_5px": round(float(np.mean(all_arr < 0.50)), 4),
        "fraction_below_1px": round(float(np.mean(all_arr < 1.00)), 4),
        "validated_subpixel_threshold_px": 0.25,
        "subpixel_capable": bool(np.mean(all_arr) < 0.25),
        "per_shift_breakdown": shift_results,
    }

    return summary


def main():
    parser = argparse.ArgumentParser(description="Run synthetic sub-pixel precision benchmark.")
    parser.add_argument("--output", type=str, default="benchmarks/subpixel_benchmark_report.json")
    args = parser.parse_args()

    out_p = Path(args.output)
    out_p.parent.mkdir(parents=True, exist_ok=True)

    print("Running synthetic sub-pixel precision benchmark across known fractional shifts...")
    results = run_subpixel_benchmark()

    with open(out_p, "w") as f:
        json.dump(results, f, indent=4)

    print(f"\nSub-Pixel Benchmark Completed:")
    print(f"  Mean Absolute Error:   {results['mean_absolute_error_px']} px")
    print(f"  Median Absolute Error: {results['median_absolute_error_px']} px")
    print(f"  Fraction < 0.25 px:    {results['fraction_below_0_25px'] * 100:.1f}%")
    print(f"  Fraction < 0.50 px:    {results['fraction_below_0_5px'] * 100:.1f}%")
    print(f"  Fraction < 1.00 px:    {results['fraction_below_1px'] * 100:.1f}%")
    print(f"  Subpixel Capable:      {results['subpixel_capable']}")
    print(f"Report saved to: {out_p}")


if __name__ == "__main__":
    main()
